Arya Forge:

# Auditing Validation and Release Approval for Financial Instruction Data

Kushagra Bhatnagar Deekshith Marla

Arya.ai

## ABSTRACT

Financial instruction-data pipelines must distinguish content validity from permission to release a particular dataset. Arya Forge combines typed records, current-content checks, and approval coverage bound to a dataset manifest. We evaluate its release boundaries in a financial-data systems case study. An integrity audit covers 100,000 synthetic instruction records; a separate arithmetic implementation agrees with all 15,000 numeric records across twelve formula families. In 240 scenarios with 2,400 treatment rows, adding financial and release-record checks to a short-instruction filtering recipe rejects all injected numeric faults and policy holds while retaining 220 of 240 clean controls. The twenty lost controls comprise every tested cash-conversion-cycle example. Nineteen persisted-release cases show that eleven selected negative cases are blocked, four equivalent representations are rejected, and changes to external evidence after binding are outside ordinary Gold promotion’s checks. Local processing measurements and controlled checkpoint resumptions characterize cost and recovery through 100,000 input rows. The results expose a tradeoff between strict producer-contract enforcement and useful-representation retention, and identify the protected inputs required for manifest-bound approval. These findings apply to the tested implementation and workloads; they do not establish general financial correctness, authenticated approval, expert-validated quality, or downstream model improvement.

Keywords: financial instruction data, dataset validation, provenance, release governance, reproducibility

## 1 Introduction

Financial instruction data supply both answers and examples of reasoning to language models. A record can pass a schema check while its answer has changed since validation, its cited evidence applies to another period, or its approval covers different content. Conversely, a valid explanation can fail a strict producer contract because it expresses the same quantity differently. Release controls therefore need to be evaluated for both invalid acceptance and useful-content loss.

Consider a debt-service coverage ratio example whose stored calculation flag remains PASS after the assistant answer is edited. A release decision based only on that flag can admit stale supervision. Recomputing supported quantities and checking the released response can reject the mutation. Yet requiring the factory’s exact wording also rejects some valid alternatives. Binding approval to the resulting manifest addresses a separate question: whether the recorded decision covers this particular release. Arithmetic, representation compatibility, and authorization are distinct properties.

Arya Forge implements auditing, synthetic instruction generation, qualification, deduplication, selection, and dataset promotion. We study three questions: (RQ1) which checks and trusted inputs connect financial-record validation and approval coverage to the released artifact; (RQ2) which constructed failures are rejected and which valid representations are lost; and (RQ3) what local processing cost and checkpoint-recovery behavior are observed? Independent financial quality and downstream model utility remain separate evaluation targets.

The paper makes three contributions:

1. A specification of the evaluated release interfaces, distinguishing fresh content checks, stored validation assertions, unresolved findings, and manifest-scoped approval coverage.

2. An empirical analysis of enforcement and retention: component diagnostics, paired filter treatments, and persisted-release cases expose both blocked faults and rejected valid representations.

3. Supporting integrity, processing, and recovery measurements, with an explicit distinction between portable code checks and privately retained experimental evidence.

The contribution is the measured interaction of these mechanisms in financial instruction-data production. Artifact binding, executable validation, and release gates are established techniques. We do not claim a new cryptographic primitive or infer a model benefit from results elsewhere in the larger repository. An exploratory AI-assisted record review is reported separately in Appendix A.

## 2 Related work

Data processing and validation. Data-Juicer 2.0 and DataTrove provide extensible processing infrastructure (Chen et al., 2025; Penedo et al., 2024); both can host application-specific checks. TensorFlow Data Validation and Deequ establish schema, anomaly, and executable constraint checking (Breck et al., 2019; Schelter et al., 2018). Our filter experiment measures two pinned DataTrove blocks with added Forge mechanisms. It does not compare equally informed financial validators or rank processing frameworks. Dolma and datasheets motivate explicit curation and artifact documentation (Soldaini et al., 2024; Gebru et al., 2021). DataComp-LM evaluates data choices under controlled training recipes (Li et al., 2024); comparable downstream evidence is outside this study.

Evidence and release control. in-toto binds authorized supply-chain steps to artifacts through signed metadata (Torres-Arias et al., 2019). Attesting LLM Pipelines proposes an attestation-aware admission gate and an evaluation blueprint for training and release claims (Tan et al., 2026). Reusing Assurance Evidence formalizes candidate relative evidence validity, implements signed checks, and studies stale dependency reuse in public DVC pipelines (Mukhopadhyay and Nalla, 2026). Structure Before Intelligence evaluates how validation, recovery, and release policies affect invalid outputs, yield, and cost in scientific workflows (Acharya et al., 2026). These works establish release gating and evidence binding as prior art. Forge examines financial instruction-record contracts and approval coverage, including their compatibility costs; it provides neither signature authentication nor incremental depen dency analysis. Table 1 locates the empirical distinction. The two October 2026 papers were incorporated during manuscript revision after our original experiments.

The Data Provenance Initiative audits dataset licensing and attribution (Longpre et al., 2024); an approval assertion in Forge does not independently establish either. OriginBlame propagates record- and token-level provenance and resolves revocation queries (Xue, 2026). Forge instead evaluates eligibility at a particular dataset-release boundary, with no external revocation discovery service.

Financial instruction data. FinQA supplies expert-authored questions and reasoning programs; FinanceBench supplies evidence-grounded financial QA (Chen et al., 2021; Islam et al., 2023). PIXIU and FinDAP study financial instruction data, capabilities, and model adaptation (Xie et al., 2023; Ke et al., 2025). From Documents to Reasoning combines synthetic financial QA validation with downstream adaptation and compares evaluated arithmetic expressions to reduce sensitivity to answer representation (Birla et al., 2026). Forge’s exact native text projection enforces a narrower producer contract and rejects some equivalent wording. Financial generation and arithmetic checking alone are therefore insufficient novelty claims. External QA datasets would require compatible adapters, documented source-overlap checks, and explicit accounting of unsupported cases.

## 3 System and trust boundary

## 3.1 Typed records and stages

Forge defines strict record contracts for continued pretraining and supervised fine-tuning. They carry content, task metadata, jurisdiction, temporal fields, provenance, rights assertions, quality and validation metadata, and trans formation lineage. Figure 1 shows the stages and the approval boundary. Raw denotes source bytes and inventory;

Table 1: Closest release-control comparisons. Entries describe the cited studies, not exhaustive limits of their software.
<table><tr><td>Study</td><td>Evidence and decision unit</td><td>Relation to the present evaluation</td></tr><tr><td>in-toto</td><td>Signed supply-chain steps and artifact identity.</td><td>Establishes authenticated step binding; Forge trusts supplied authority assertions.</td></tr><tr><td>Attesting LLM Pipelines</td><td>Attested training/release claims at admission.</td><td>Gate design and evaluation blueprint; Forge reports executed record and release probes.</td></tr><tr><td>Reusing Assurance Evidence</td><td>Signed control evidence valid for a candidate and its dependencies.</td><td>Studies stale evidence reuse; Forge tests current financial records and approval scope.</td></tr><tr><td>Structure Before Intelligence</td><td>Validation and release of workflow outputs.</td><td>Measures safety, yield, and cost; Forge examines instruction-data representation and</td></tr><tr><td>Arya Forge</td><td>Typed instruction records and approval bound to a manifest.</td><td>approval coverage. Measures selected faults, clean-template losses, and equivalent-representation rejection; no authority authentication.</td></tr></table>

Bronze denotes canonical migration; Silver denotes selected and validated candidate content; Gold denotes a release with explicit approval coverage. The protocol intends Gold to be an approval state, not a synonym for high quality. Historical implementation gaps described below show why that intention must be tested at every promotion entry point.

Releases include JSONL and Parquet representations, statistics, source inventories, rights and quality reports, and manifests. Manifests bind artifact hashes, record-set hashes, configuration, validator versions, and parent releases. Output construction uses a staging directory and refuses an existing immutable destination. These controls detect divergence relative to the recorded hashes; they do not prove that the source facts or approvals were correct when recorded.

![](images/5cf986285b3b1c90c0f7b66a4b6fb5ca7c8def80d03725f4779fe1e19469bc8c.jpg)  
Figure 1: Dataset stages and the ordinary approval path. Solid arrows denote processing; dashed arrows denote inputs to binding or promotion. Ordinary Gold promotion rechecks current records, parent integrity, and active coverage, but does not reopen external evidence. Supplemental promotion has a different evidence read boundary (Section 3.2). Authority assertions, accepted identities, and evaluation time remain trusted inputs.

## 3.2 Approval coverage

The promotion registry binds to an exact Silver manifest. Each retained record must be covered by one active sourceor record-scoped decision, with no missing or ambiguous coverage. Decision metadata includes an approval refer ence, evidence digest, permission assertions, obligations, and validity information. A named governance authority remains responsible for the authenticity and substance of the decision.

Let r be a record, m its parent manifest, g the supplied approval registry, and t the evaluation time. Write $C _ { v } ( r )$ for the supported current-content checks under validator version $v , Q ( r )$ for absence of unresolved quarantine conditions, $A ( r , m , g , t )$ for unique active approval coverage, and $I ( m )$ for manifest-listed artifact integrity. The intended promotion condition is

$$
\mathrm { P r o m o t e } ( r , m , g , t ) \Rightarrow C _ { v } ( r ) \wedge Q ( r ) \wedge A ( r , m , g , t ) \wedge I ( m ) .
$$

This is a design requirement, not a formal proof about all program paths. Approval resolves only the rights condi tions within its scope. It must not resolve an unrelated privacy, arithmetic, or source-applicability finding. Table 2 specifies the tested transition sequence.

Table 2: Checks in the ordinary governed promotion path. Registry binding and Gold promotion are separate operations.
<table><tr><td>Operation</td><td>Required evidence and effect</td></tr><tr><td>Load Silver</td><td>Check stage and manifest-listed artifact hashes; parse typed records and check count.</td></tr><tr><td>Bind registry</td><td>Check external evidence bytes and decision activity; bind decisions to this manifest and require unique coverage.</td></tr><tr><td>Evaluate Gold</td><td>Rerun current record checks and quarantine; evaluate manifest pin and unique active approval coverage at the supplied time.</td></tr><tr><td>Write Gold</td><td>Record approval and parent lineage, embed registry and coverage report, and refuse an existing immutable destination.</td></tr></table>

The trusted inputs are the verifier implementation, caller-supplied evaluation time, governance authority, and accepted manifest/registry identities. Activity includes approval/effective dates, exclusive expiry, and a recorded revocation date. The library does not discover external revocations. Hashes detect divergence from an accepted reference; an actor able to replace content and references can create a self-consistent release. The ordinary promotion function accepts a registry file, without a separate expected-registry-digest argument. Protecting that accepted registry and the evaluation time is a caller/deployment responsibility, not an additional authentication check performed by the function.

Evidence freshness. The ordinary path has a binding-time evidence boundary: binding checks external evidence bytes, and promotion subsequently trusts the bound registry. A deployment using snapshot approval must retain the approved evidence version and protect its reference; the ordinary Gold artifact embeds the registry and coverage report, not the external evidence itself. A deployment requiring current evidence or live revocation status needs an additional freshness mechanism. These are different operational requirements, and neither is established by hashing alone.

Entry-point coverage. The nineteen persisted-release cases exercise ordinary promotion. Supplemental Gold promotion reuses the current record gate but also reopens referenced corpus approval evidence while deriving it registry, and checks selection/governance lineage. It is covered by software regressions, not by the nineteen-case experiment. The low-level DatasetReleaseBuilder is a trusted writer and does not implement the full approval gate. Consequently, ordinary promotion’s accepted post-binding mutation must not be generalized to the supple mental path.

## 3.3 What is revalidated at promotion

Every record entering the shared gate is parsed against the current typed schema. The gate checks score, route, lineage, unresolved recorded findings, and current text against implemented sensitive-data patterns. For recognized native numeric contracts, it recomputes supported quantities and compares the question, derivation, and response with the producer’s canonical projection. Unsupported recognized numeric contracts are refused. Unmarked arithmetic elsewhere is not comprehensively discovered.

Task validation also consults stored assertions: schema/task error flags, a numeric-valid flag when calculation is required, and grounding/citation flags when retrieval is required. The shared promotion gate does not rerun every domain-specific generator or qualification validator. Table 3 makes that coverage explicit across the ten audited domains. Fresh privacy-pattern checks and numeric-contract checks should therefore not be read as fresh expert validation of every financial statement.

## 3.4 Corpus processing and selection

The streaming corpus path uses disk-backed fingerprints, bounded audit shards, checkpoint state, and source/configuration pins. Qualification generates rule-level findings and an inventory for rights review. Candidate selection combines eligibility, near-duplicate review, tokenizer accounting, and curriculum limits. The ordinary small-release builder materializes records in memory; the existence of a streaming audit does not imply that every command is memory bounded.

Stored quality scores and deterministic PASS labels are operational signals. Some synthetic factories assign fixed scores rather than estimating calibrated accuracy. In general, absence of a finding means no implemented check rejected the record; it is not proof of factual truth. These distinctions govern all interpretations in this paper.

## 4 Evaluation design

## 4.1 Frozen local assets

The asset audit enumerates ten local Gold releases and verifies manifest-listed artifact digests, JSONL counts, Parquet metadata counts, record-set hashes, and source-manifest hashes. It inventories declared domains and source identifiers. Synthetic status is reported from the explicit source prefixes; it does not independently establish ownership. A subsequent independently written consistency checker decodes every Parquet record payload and compares it with the corresponding JSONL row, rejecting duplicate JSON keys and nonfinite values. It separately reconstructs all twelve projected Parquet columns from JSONL and checks their types and values; this supplements the original count-and-hash audit.

The audit does not rewrite historical data or replace their original validation metadata. A new report binds the current verifier source hashes and runtime versions to its observations. Because the working tree contains unrelated ongoing changes, the research package records the base revision and exact relevant source snapshots; a bare Git commit alone is not claimed to reproduce the repaired state.

## 4.2 Separate arithmetic implementation

The historical numeric factory set independently\_verified by serializing a case and repeating the same computation. A deterministic bug can therefore satisfy the old check. The revised default uses a separate module that does not call the production calculator or generator. Ratio, loss, and conversion expressions use rational arithmetic; EMI is recomputed from the sum of discounted instalments; CAGR uses logarithm and exponential evaluation. For principal P, monthly rate $i ,$ and n instalments, the reference computes

$$
\mathrm { E M I } _ { \mathrm { r e f } } = \frac { P } { \sum _ { t = 1 } ^ { n } ( 1 + i ) ^ { - t } } .
$$

At zero interest this reduces to $P / n$ . Typed input constraints are enforced before the legacy EMI reference path, including a bounded tenure.

The reference checks output names and units, exact half-up displayed rounding, and unrounded agreement within

$$
\operatorname* { m a x } \left( 1 0 ^ { - 2 0 } , \ : \left| v _ { \mathrm { r e f } } \right| 1 0 ^ { - 2 4 } \right) .
$$

The tolerance accommodates the production implementation’s decimal precision. New generation records a new verification version. Explicit legacy reproduction remains available for historical artifacts and retains the old version identifier; those artifacts receive a fresh check before new promotion.

Separate implementation reduces shared-code failure modes, but the implementations retain common mathematical specifications and typed assumptions. Agreement is neither expert validation of those assumptions nor evidence that generated scenarios represent real borrowers.

## 4.3 Current-content promotion checks

New numeric release validation recomputes supported native factory cases, checks their structured results and derivations, and binds the canonical question and assistant response to those results. Changing domain metadata alone cannot disable checks when a native numeric contract or source marker remains. Records recognized by the numericcontract dispatch are rejected when their format is unsupported; unmarked arithmetic elsewhere is not comprehen sively discovered. The native contract is intentionally strict: alternate valid prose may require explicit migration rather than passing automatically.

New quarantine validation scans current training content with the existing sensitive-data patterns and rejects unresolved recorded sensitive-data findings. A generic human-review boolean cannot dismiss a finding. Rights-only quarantine remains compatible with the approval path; unknown or unrelated quarantine reasons remain blocked. The scanner is limited to its implemented patterns and is not a comprehensive privacy detector.

## 4.4 Overlap detection and near deduplication

The revised overlap audit aligns recognized question, request, structured-input, and conversation envelopes. It retains substantive context, numeric values, and prior conversation turns while ignoring absent optional message metadata. It preserves existing identity/source/content checks and rejects empty or missing training inputs rather than returning a vacuous clean result. This is exact normalized overlap detection, not semantic decontamination or a proof of template independence.

Near deduplication now preserves Unicode word characters and financially significant literals. A near-match may reject a record only when its ordered financial-literal signature agrees with that of the candidate. This prevents lexical similarity alone from erasing changed values or signs. Empty token streams cannot near-match. The policy is deliberately conservative: equivalent numeric formatting can retain extra examples, and matching literals do not establish semantic equivalence. The SQLite index stores the normalization and detector policy and rejects incompatible resume state.

## 4.5 Targeted component diagnostics

The diagnostic suite preserves pre-repair tracked source from the recorded repository revision and loads those com ponent snapshots under separate module names. It compares them with the repaired components on author-selected clean controls and fault probes. Cases include deliberately incorrect arithmetic, changed responses with stale flags, unresolved sensitive findings, repeated prompts across formats, and financially distinct near-duplicate candidates.

The cases were authored in response to observed defects. They are retrospective regression diagnostics, not a randomly sampled test set, an independent benchmark, a production incidence estimate, or a comparison with DataTrove/Data-Juicer. The harness reruns the substantive cases to check deterministic equality and records each expectation and observation. The baseline is component replay with shared current dependencies, not a complete historical environment reconstruction.

## 4.6 Controlled filter composition

We freeze a second study before drawing its new parameters and observing filter outcomes. The study remains internally designed after the initial defect analysis, not externally preregistered. It contains 240 test scenarios, twenty per numeric formula, allocated evenly between two supported native answer formats. Each scenario receives ten treatments: one clean control; four numeric inconsistencies affecting response value, response unit, structured truth, or prompt value; four release-policy holds affecting current sensitive content, a retained sensitive finding, a rejected route, or a below-threshold quality score; and one repetition probe. Thus there are 2,400 paired test rows, not 2,400 independent financial situations. An additional 24 development scenarios yield 240 rows. All content is locally generated synthetic material covered by the recorded project-owner authorization; policy holds are not characterized as legally unauthorized source data.

The baseline executes Gopher repetition and quality heuristics (Rae et al., 2021) through the actual GopherRepetitionFilter and GopherQualityFilter blocks from DataTrove 0.10.1. It sees the concatenated system, user, and assistant messages, without hidden quality metadata or expected labels. We report its defaults as a sensitivity analysis and prespecify a short-instruction recipe: minimum five words, minimum average word length one, minimum zero stop words, and max\_non\_alpha\_words\_ratio=0.3; other setting remain at package defaults. The latter implementation parameter acts as a minimum fraction of words containing an alphabetic character. Two further arms append either the numeric validator or the actual Forge release-record gate with affirmative rights eligibility. The final arm includes numeric checks, current and retained sensitive-data findings, route, score, and lineage checks. It does not execute the complete raw-to-Gold pipeline, approval-registry coverage, corpus selection, or either framework’s distributed executor.

Test/development numeric parameter hashes are disjoint, and test hashes are compared with the existing numeric Gold tranche. Formula definitions, generators, and answer templates remain shared. Outcomes are clean-control retention and acceptance of each constructed fault/hold category. We retain per-scenario decisions and report three repeated passes; repeated treatments and shared templates preclude treating aggregate counts as independent binomial observations. The experiment does not estimate naturally occurring defect prevalence or valid-paraphrase rejection. Four treatments alter only metadata unavailable to the text baseline: structured truth, a retained sensitive finding, rejected route, and quality score. Baseline acceptance of these rows is expected from its input boundary; it is not evidence that a text filter fails at its intended task.

## 4.7 Persisted release-boundary experiment

A further code-informed protocol freezes nineteen cases and hashes of 100 source and configuration files before runner execution. Every case starts with the same two deterministic native numeric records (DSCR seed 17 and EMI seed 29), builds a persisted Bronze release, and invokes the production Bronze-to-Silver promotion. It then binds explicitly fictional approval evidence and attempts governed Silver-to-Gold promotion. Two cases terminate during evidence binding; seventeen reach a Gold attempt. Dates and identifiers are fixed. These dependent case test selected mechanisms, not a representative sample or real approval authority.

Cases cover source- and record-scoped controls; incomplete, overlapping, expired, and wrong-parent approvals; dataset-byte tampering; incorrect current answers with stale flags; current and stored sensitive findings; equivalent representations; and evidence changes before or after binding. To isolate content revalidation, content-mutation cases coherently update both dataset formats, artifact and record-set hashes, and bind fresh fictional approval to the changed Silver manifest. This is not a bypass of an unchanged prior approval. Four constrained semantic variants change a prompt synonym, add an answer introduction, spell the ratio unit as “times,” or add redundant derivation parentheses; their values and parameters remain fixed. A separate JSON whitespace/member-order control preserves the parsed object.

Accepted outputs are checked for artifact and row consistency, coverage, lineage, training-eligibility metadata, and immutable-destination refusal. All Gold attempts check that their Silver parent remains unchanged. Two complete runs compare result bytes and all persisted file hashes. An initial runner assertion expected lowercase sft instead of the schema’s SFT; the instrumentation-only repair and aborted artifacts are retained. No case expectation, protocol, or core implementation changed.

## 4.8 Local processing and recovery experiment

We freeze a separate workload and measurement specification before timing three actual corpus stages: streaming audit, Bronze migration, and route/truth qualification. Each ten-row block contains seven distinct single-chunk CPT records, two distinct SFT records, and one exact duplicate of the second SFT record. The raw workload therefore has 10% duplicates and a 70:30 CPT:SFT mixture. We run 1,000, 10,000, and 100,000 raw rows with three repetitions per size. This authored fixed-length workload tests execution behavior, not financial correctness or realistic document diversity.

Each stage uses a fresh Python subprocess on an Apple M3 Pro with 18 GiB RAM, Python 3.13.7, and macOS. Stage time excludes interpreter startup/imports; peak resident memory includes them and is measured using the child’s platform-corrected ru\_maxrss. Logical output size includes stage artifacts and checkpoint state. The OS cache is not flushed, and unrelated desktop workloads are uncontrolled. Qualification receives 90% of raw rows after duplicate exclusion. At the 10K scale, public interruption hooks stop each stage after five committed chunks or shards; a new process resumes it. We compare substantive row-multiset hashes and summaries with the un interrupted run, allowing provenance paths to differ. Bronze and qualification reuse uninterrupted parent reports, isolating stage-local recovery rather than a chained end-to-end recovery. This tests controlled post-checkpoint re covery, not arbitrary process kills or power loss.

## 5 Results

## 5.1 Asset integrity and composition

The audit verifies all ten releases, all 80 manifest-listed artifacts, and 100,000 instruction records. The subsequent exhaustive format check finds 100,000 matching decoded row payloads and 1,200,000 matching projected field values, with no discrepancies. This establishes representation consistency for these bytes, not factual validity. Table 3 reports the observed mixture. Every record has a synthetic source prefix; none has its stored human-review flag set. The latter describes metadata, not a finding that no person ever reviewed any related material.

Table 3: Audited synthetic Gold mixture and shared promotion-gate coverage. All domains receive current schema, privacy-pattern, and unresolved-finding checks. “None” means no additional domain-semantic recomputation in this shared gate; domain-specific generation or qualification checks may have run earlier, and stored validation assertions are still consulted.
<table><tr><td>Domain</td><td>Records</td><td>Additional fresh semantic check</td></tr><tr><td>Corporate credit</td><td>20,000</td><td>None</td></tr><tr><td>Numeric reasoning</td><td>15,000</td><td>Supported native arithmetic and canonical text; supported legacy EMI</td></tr><tr><td>Grounded regulation</td><td>15,000</td><td>None; grounding/citation flags are stored assertions</td></tr><tr><td>Financial analysis</td><td>15,000</td><td>None</td></tr><tr><td>SME credit</td><td>10,000</td><td>None</td></tr><tr><td>Trade finance</td><td>8,000</td><td>None</td></tr><tr><td>AML/KYC</td><td>7,000</td><td>None</td></tr><tr><td>Transaction banking</td><td>5,000</td><td>None</td></tr><tr><td>Refusal/safety</td><td>3,000</td><td>None</td></tr><tr><td>General retail banking</td><td>2,000</td><td>None</td></tr><tr><td>Total</td><td>100,000</td><td>Arithmetic cross-check covers the 15,000-record numeric tranche</td></tr></table>

All 15,000 numeric records pass both the separate arithmetic implementation and fresh native-contract validation. There are 1,250 records in each of twelve formula families. The families are EMI, debt-service and interest coverage, fixed-obligation-to-income, loan-to-value, debt-to-equity, net-debt-to-EBITDA, liquidity ratios, cash conversion cycle, expected loss, compound annual growth, and foreign-exchange conversion. Their historical verifier identifier is financial-computation-v1; the new result is a retrospective check of those immutable records, not a reclassification of the original verifier as independent. Exhaustive agreement on this fixed population does not yield an external error rate for other cases, domains, or specifications.

## 5.2 Repaired boundaries and exact overlap

All fifteen targeted diagnostic cases satisfy their expected behavior under the repaired components; the preserved pre repair components satisfy five and fail ten (Table 4). The five clean controls conform under both implementations. The ten fault/edge probes were selected to expose known weaknesses, so the comparison must not be reported as an accuracy improvement from 33% to 100%. The diagnostic rows reproduce exactly on repeated execution after excluding runtime metadata. Preserved snapshots can be replayed with hash checking without requiring Git.

The refreshed overlap audit checks all 100,000 local training records against 1,280 evaluation cases across nine releases and detects no identity/source/exact-normalized collisions under its implemented comparisons. A separate integrity check verifies all nine evaluation manifests and 27 listed artifacts against the index and release metadata, with no discrepancies. Shared generators, paraphrases and templates remain outside this result; different numeric seeds do not establish an independent generalization benchmark.

Table 4: Conformance on author-selected component diagnostics. Counts are not population accuracy.
<table><tr><td>Component</td><td>Cases</td><td>Pre-repair conforming</td><td>Repaired conforming</td></tr><tr><td>Numeric factory</td><td>2</td><td>1</td><td>2</td></tr><tr><td>Promotion boundary</td><td>7</td><td>3</td><td>7</td></tr><tr><td>Overlap audit</td><td>3</td><td>0</td><td>3</td></tr><tr><td>Near deduplication</td><td>3</td><td>1</td><td>3</td></tr><tr><td>Total</td><td>15</td><td>5</td><td>15</td></tr></table>

A current-content quarantine scan over the 100K local records returns no implemented-pattern or retained-flag findings and no scan/schema errors. This does not estimate privacy-detector recall. Ordinary and supplemental Gold promotion are both covered by current checks; the low-level release writer remains a trusted construction API and is not itself a complete approval authority. Regression tests additionally cover metadata dispatch changes, invalid legacy numeric inputs, and promotion of old Silver releases. These software tests support the scoped behavior without supplying independent human validation.

## 5.3 Incremental checks on constructed faults

Figure 2 shows category acceptance with exact counts for the 2,400 test rows. The default web-text filters reject every clean control, illustrating an unsuitable operating point for short numerical instruction examples. The pre specified SFT recipe retains 220 of 240 controls and accepts 880 of 960 numeric-inconsistency probes. Adding the numeric validator rejects all those numeric probes but leaves 880 policy-held rows accepted. Adding the release record gate rejects all constructed numeric faults and policy holds. Every arm rejects the 240 repetition probes.

(a) Clean retained Higher is better

![](images/0165cbfadb426eadd46d0895f4b2e1d383c60c1ed0fae1dfce9f196985570522.jpg)  
(b) Numeric faults Lower is better

![](images/6a8dd50ea48c1870eefe0175c42ee9676ddf3a4dfe7c907fcd586c2c2ad53e0c.jpg)  
(c) Policy holds Lower is better

![](images/45e069c21db9e8c027cc2cd0114073bca751946b7cfc42c28dfd406f9ec5f6a7.jpg)  
Figure 2: Retention and acceptance on the 2,400 constructed test rows. Labels give exact accepted/total counts; all recipes accept 0/240 repetition probes. Clean retention is desirable, whereas accepting the injected numeric faults or release-policy holds is undesirable. Treatments are paired within 240 scenarios. Policy holds are eligibility conditions, not necessarily financial errors; four treatments alter metadata unavailable to the text-only baseline. The final arm is a composed record gate, not the full pipeline.

For the native contracts, visible prompt and answer mutations are rejected by comparison with a regenerated canonical representation. This demonstrates contract enforcement, not semantic validation of independently authored questions or valid paraphrases. Separate arithmetic recomputation checks structured values under the shared formula specification. Likewise, the metadata-only treatments give the composed gate information absent from the text baseline; the figure reports incremental mechanisms, not equal-information classifier superiority.

The twenty rejected clean controls are every cash-conversion-cycle control, while all twenty controls in each of the other eleven families survive (Table 5). Repeated derivation/substitution content triggers the repetition filter’s duplicate-line character criterion for ten text-format controls and its duplicated five-gram criterion for ten JSONformat controls. Both augmented arms inherit these losses. The aggregate 20/240 (8.3%) clean rejection therefore conceals complete loss of one tested task family. No setting is changed after observing this result. Retention should be evaluated as coverage of useful tasks and representations, as well as an overall percentage.

Table 5: Clean-control retention by family and representation. The SFT recipe and both augmented arms have identical counts. These are the existing 240 controls, not an additional experiment.
<table><tr><td>Formula group</td><td>Text retained</td><td>JSON retained</td><td>Total retained</td></tr><tr><td>Cash conversion cycle</td><td>0/10</td><td>0/10</td><td>0/20</td></tr><tr><td>Each of the other eleven families</td><td>10/10</td><td>10/10</td><td>20/20</td></tr><tr><td>All twelve families</td><td>110/120</td><td>110/120</td><td>220/240</td></tr></table>

There are no exact numeric-parameter overlaps between the 240 test scenarios and either the 24 development scenarios or the 15,000 existing Gold cases. This is parameter separation, not template-independent evaluation. Repeated decision hashes agree across three passes. Median warm pass times over all 2,640 development-plus test rows are 3.47 s for the SFT recipe, 3.83 s with numeric checks, and 3.89 s with release-record checks. These fixed-order, single-process observations do not estimate distributed throughput or broad comparative efficiency.

## 5.4 Release boundary: rejection and acceptance limits

The three ordinary controls produce Gold and eleven selected negative cases are blocked (Table 6). All four constrained semantic variants are rejected by the canonical numeric contract. These are losses of valid representations under the constructed equivalence criterion, even though rejection conforms to the implemented producer contract. They are not a population false-rejection rate. The JSON formatting control passes because equality is assessed after parsing.

Table 6: Nineteen dependent persisted-release cases. Counts are case tallies; all semantic variants target the DSCR record.
<table><tr><td>Case category</td><td>Accepted</td><td>Rejected</td></tr><tr><td>Complete source/record approvals; JSON formatting</td><td></td><td>0</td></tr><tr><td>Partial, overlapping, expired, wrong-parent approvals</td><td>0</td><td>4</td></tr><tr><td>JSONL/Parquet byte tampering</td><td>0</td><td>2</td></tr><tr><td>Wrong answer with stale flag and fresh approval</td><td>0</td><td>1</td></tr><tr><td>Current or stored sensitive findings</td><td>0</td><td>2</td></tr><tr><td>Missing or changed evidence before binding</td><td>0</td><td>2</td></tr><tr><td>Equivalent wording, unit spelling, derivation</td><td>0</td><td>4</td></tr><tr><td>Evidence modified after binding</td><td>1</td><td>0</td></tr><tr><td>Total</td><td>4</td><td>15</td></tr></table>

Changing external evidence after binding still permits ordinary Gold promotion: the unchanged bound registry is trusted. This is a freshness limitation when deployment policy requires the current external evidence. Under snapshot approval, the relevant requirement is preservation and protection of the approved evidence version; the experiment does not validate an operational snapshot store or revocation service. All four accepted releases satisfy the artifact/content/lineage and immutable-destination checks. Both runs match in result bytes and all 357 files in each hash inventory, excluding the inventory itself. Nine runner tests pass. The experiment excludes low-level builder misuse, concurrent writers, filesystem access controls, and supplemental Gold, which has a different evidence-read boundary.

## 5.5 Processing cost and controlled recovery

All 27 planned uninterrupted stage runs complete with stable source hashes and no omissions. Figure 3 shows pro cessing time and memory across scales; Table 8 in Appendix C preserves the numerical measurements and output sizes. At 100K raw rows, median stage times are 31.36 s for audit, 13.48 s for Bronze, and 4.02 s for qualification. Their observed ranges are 31.35–33.60, 13.42–14.03, and 3.68–4.05 s respectively. These stages write approxi mately 197.9, 242.7, and 45.8 MiB of logical output at that scale, versus an 87.0 MiB raw source; the output sizes describe retained artifact footprint, not bytes written by the storage device.

![](images/6fa1e05220ab62e3334b5374a223023d5a6d204efb85b363c7c9863c58bdb818.jpg)

![](images/94244266ecf277027a2e1d82557406d8c6ea05ccec6d8aa870e65ecbf4cd3ab9.jpg)  
Audit Bronze migration Qualification  
Figure 3: Local processing measurements on the authored workload, with three uninterrupted repetitions per stage and size. Left: median stage time and observed minimum–maximum bars (not confidence intervals); small ranges can be hidden by markers. Right: the largest observed peak RSS in each group of three runs, on a linear scale starting at zero. Time excludes interpreter startup/imports; peak RSS includes them. Qualification reads 0.9N rows after deduplication. Connecting lines guide the eye; they do not fit or extrapolate a scaling law.

Exactly 10% of raw rows are recognized as duplicates; migration produces the expected 90% typed rows without migration errors. The raw workload contains no initially canonical, schema-valid records, and all migrated rows retain a QUARANTINE route. Qualification records missing-temporal-metadata warnings for the CPT rows and retains review-required rights status. These measurements therefore cover staging and rule execution, not production of approved training data. All three 10K interruption/resume comparisons match their uninterrupted substantive rows and summaries. The result supports the tested checkpoint boundaries; it says nothing about uncommitted-write durability, near deduplication, tokenizer cost, approval adjudication, or full release-builder memory.

## 5.6 Reproducibility and scope

The installed code-only distribution passes all 182 shipped tests in the recorded fresh-environment check. A separate data-inclusive extraction reproduces selected audit, overlap, and release results on the same host. The suites overlap

and do not constitute independent replication; Appendix D gives their exact scope. The full research inputs remain private, and the code package alone cannot reconstruct all reported results.

An exploratory review of 200 retained records by two same-model AI agents flags possible usefulness and evidence problems despite arithmetic agreement. These unweighted judgments motivate qualified review; they are not expert labels or population defect estimates (Appendix A).

## 6 Limitations and claims boundary

The verified dataset is synthetic and comes from a single project. Numerical agreement across two implementations does not detect a shared specification error. Most nonnumeric domains have only operational metadata and domain specific rules, not independent expert truth labels. Pattern-based sensitive-data scanning has limited coverage, and lexical deduplication cannot establish semantic uniqueness. The stricter numeric contract rejects all four selected equivalent representations; this small DSCR-based probe cannot estimate the frequency of such loss in other tasks or natural data.

The overlap audit does not establish separation of templates, source knowledge, or model pretraining data. Hashes establish identity relative to supplied inputs, not source correctness or authenticated approval. Study designs were informed by inspected code and observed defects; the later frozen experiments remain internal and may underrepresent unknown failure modes. The nineteen release cases share two source records and do not test a hostile deployment or authenticated approval service. The comparative study uses two actual DataTrove blocks, not a tuned competitive corpus pipeline; its shared generators and native contracts can favor Forge’s validators. Its rejection labels derive from explicit perturbations and policies, not blinded expert truth. Likewise, two same-model AI reviewers can agree and still be wrong.

The full-corpus bundles, independent human labels, natural-defect curation comparison, and matched-budget model experiments remain unavailable or unexecuted. Performance observations are limited to one laptop and a repetitive authored workload; a fresh virtual environment still shares the host and package cache. The full research artifacts remain private, limiting external verification until confidential reviewer access is arranged. Appendix B records the additional evaluations needed for stronger claims. These limits permit a bounded systems claim, not broad financial validity, framework superiority, or causal downstream model benefit.

## 7 Broader impact and assistance disclosure

Binding validation evidence to released content can make unnoticed changes and unresolved release conditions easier to audit. However, financial training data can still contain unsupported decisions, omitted risks, misleading precision, or jurisdictional assumptions even when its arithmetic is correct. Using a clean gate report as a substitute for substantive financial review could propagate those weaknesses to consequential applications. Strict contracts and generic filters can also remove useful examples and narrow coverage. Our reports therefore separate arithmetic, policy, uncertainty, and retention outcomes; no artifact certifies suitability for lending, compliance, or customer decisions.

AI assistants were used to inspect and modify software, construct experimental runners, assist manuscript draft ing, and conduct the explicitly labeled exploratory review. Separate reviewer agents shared the same underlying model family and received fresh review contexts. They are not authors or qualified human participants. Their labels are retained as model outputs, and responsibility for the final scientific claims, authorship, release permissions, and submission rests with the human authors.

## Funding and competing interests

This work was funded by Arya.ai. Both authors are affiliated with Arya.ai, the developer of the system studied. No external financial experts participated in the reported AI-assisted review.

## 8 Conclusion

This case study connects financial instruction-record checks to manifest-scoped release approval and measures their interaction. Fresh native-contract validation rejects the selected stale-content faults, while the composed filtering recipe removes every tested clean cash-conversion-cycle example and the strict contract rejects four equivalent DSCR representations. The ordinary and supplemental promotion paths also differ in when they read approval evidence. These findings support explicit interface specifications, protected approval inputs, and retention reporting by task family. Integrity audits and controlled recovery provide supporting implementation evidence. Independent examples, qualified labels, and equal-input mechanism comparisons are the next tests of generality; model benefit requires a separate training study.

## Data and code availability

A standalone code distribution, version 0.1.0, is prepared under the Apache License 2.0, copyright Arya.ai, as arya-forge-0.1.0-source-20261025.tar.gz. It contains the allowlisted implementation, documentation, integrity manifest, and portable tests, with a separately built wheel. Publication at a public repository has not yet occurred. The synthetic populations, case-level results, AI annotations, approval fixtures, and research evidence remain private and are not covered by the code distribution’s license. A separate confidential reproduction package preserves these supporting materials. Controlled reviewer access arrangements have not been established. No public dataset, code-hosting URL, or artifact DOI is claimed.

## References

A. Acharya, A. Al Amin, and H. Qin. Structure Before Intelligence: Gating What Agentic Scientific Workflows Release. arXiv:2610.19507v1, 2026. https://arxiv.org/abs/2610.19507.

L. Birla, M. Savagaonkar, V. Srinivasan, S. Rasipuram, and S. Sengupta. From Documents to Reasoning: A Validated Synthetic Data Pipeline and Semantic-Aware Fine-Tuning for Financial Numerical Reasoning. arXiv:2608.27919v1, 2026. https://arxiv.org/abs/2608.27919.

E. Breck, N. Polyzotis, S. Roy, S. E. Whang, and M. Zinkevich. Data Validation for Machine Learning. Proceedings of Machine Learning and Systems, 1, 2019. https://proceedings.mlsys.org/paper\_files/ paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html.

D. Chen et al. Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models. Advances in Neural Information Processing Systems, 38, Datasets and Benchmarks Track, 2025. DOI: 10.52202/085713-2768. Official proceedings.

Z. Chen et al. FinQA: A Dataset of Numerical Reasoning over Financial Data. Proceedings of EMNLP, pages 3697–3711, 2021. https://doi.org/10.18653/v1/2021.emnlp-main.300.

T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and K. Crawford. Datasheets for Datasets. Communications of the ACM, 64(12):86–92, 2021. https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/.

P. Islam, A. Kannappan, D. Kiela, R. Qian, N. Scherrer, and B. Vidgen. FinanceBench: A New Benchmark for Financial Question Answering. arXiv:2311.11944, 2023. https://arxiv.org/abs/2311.11944.

Z. Ke, Y. Ming, X.-P. Nguyen, C. Xiong, and S. Joty. Demystifying Domain-adaptive Post-training for Financial LLMs. arXiv:2501.04961v4, 2025. https://arxiv.org/abs/2501.04961.

J. Li et al. DataComp-LM: In search of the next generation of training sets for language models. arXiv:2406.11794, 2024. https://arxiv.org/abs/2406.11794.

S. Longpre et al. A large-scale audit of dataset licensing and attribution in AI. Nature Machine Intelligence, 6:975–987, 2024. https://doi.org/10.1038/s42256-024-00878-8.

A. Mukhopadhyay and S. R. Nalla. Reusing Assurance Evidence in AI Pipelines: Validity Conditions, a Fail-Closed Release Gate, and a Replay of 714 Public Data Version Control Pipelines. arXiv:2610.27672v1, 2026. https://arxiv.org/abs/2610.27672.

G. Penedo et al. DataTrove: large scale data processing. Software, 2024. Evaluated version 0.10.1; accessed 7 October 2026. https://pypi.org/project/datatrove/0.10.1/.

J. W. Rae et al. Scaling Language Models: Methods, Analysis & Insights from Training Gopher. arXiv:2112.11446, 2021. https://arxiv.org/abs/2112.11446.

S. Schelter, D. Lange, P. Schmidt, M. Celikel, F. Biessmann, and A. Grafberger. Automating Large-Scale Data Quality Verification. Proceedings of the VLDB Endowment, 11(12):1781–1794, 2018. https://doi.org/10.14778/3229863.3229867.

L. Soldaini et al. Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. Proceedings of ACL, Volume 1: Long Papers, pages 15725–15788, 2024. https://doi.org/10.18653/v1/2024.acl-long.840.

Z. Tan, J. Singer, and C. Anagnostopoulos. Attesting LLM Pipelines: Enforcing Verifiable Training and Release Claims. arXiv:2603.28988v1, 2026. https://arxiv.org/abs/2603.28988.

S. Torres-Arias, H. Afzali, T. K. Kuppusamy, R. Curtmola, and J. Cappos. in-toto: Providing farm-to-table guarantees for bits and bytes. 28th USENIX Security Symposium, pages 1393–1410, 2019. https://www.usenix.org/conference/usenixsecurity19/presentation/torres-arias.

Q. Xie, W. Han, X. Zhang, Y. Lai, M. Peng, A. Lopez-Lira, and J. Huang. PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance. arXiv:2306.05443, 2023. https://arxiv.org/abs/2306.05443.

H. Xue. OriginBlame: Record- and Token-Level Data Provenance for AI Training Datasets. arXiv:2607.13037v2, 2026. https://arxiv.org/abs/2607.13037.

## Appendix A. Exploratory AI-assisted assessment

## A.1 Exploratory AI-assisted record review

At the project owner’s request, two fresh, same-base-model assistant agents separately review the 200-record retained packet. Each receives only the common rubric and visible conversations, without the coordinator’s domain mapping, pipeline scores, factory implementations, reference judgments, the manuscript, or the other agent’s annotations. Each may use independently written arithmetic code; no external factual source is authenticated by this procedure. Labels are locked separately before coordinator analysis, and the original blank human-review packet is preserved.

The rubric covers numeric correctness, support by supplied evidence, jurisdiction/date applicability, and training usefulness, with explicit uncertainty and not-applicable categories. Analysis reports paired categorical agreement and Cohen’s κ, including those categories, and retains disagreements without manufacturing a consensus truth label. These are AI judgments from a shared model family, not independent financial experts, a human-subject study, or established data-quality error rates. The model/API revision and sampling seed were not pinned, so preserving the labels enables analysis replay rather than deterministic recreation of the model judgments. Repeated templates and correlated model assumptions further limit their evidential strength.

## A.2 AI review agreement and unresolved findings

Both agents complete and lock all 200 records. Reviewer A labels 151 useful, 34 not useful, and 15 uncertain; reviewer B labels 128 useful, 32 not useful, and 40 uncertain. They jointly label 31 records not useful. At least one adverse label appears on 61 records, and at least one rubric dimension differs on 58 records. These are unweighted descriptions of the equally allocated sample, not estimated defect rates in the 100K population.

Table 7: Descriptive agreement of two same-model AI reviewers over 200 paired records. Uncertain and notapplicable labels remain categories; agreement is not accuracy.
<table><tr><td>Dimension</td><td>Raw agreement</td><td>Cohen&#x27;s κ</td></tr><tr><td>Numeric correctness</td><td>80.0%</td><td>0.573</td></tr><tr><td>Evidence support</td><td>84.0%</td><td>0.698</td></tr><tr><td>Jurisdiction/date applicability</td><td>91.0%</td><td>0.832</td></tr><tr><td>Training usefulness</td><td>86.5%</td><td>0.716</td></tr></table>

Reviewer A records 395 recomputed or checked values, including 392 explicit numerical comparisons; reviewer B records 395 explicit comparisons. Neither checker reports a mismatch, conditional on its formulas and assumptions. The counts differ because three entries in A’s filing fallback record supplied source values without comparing them with a candidate output; the final audit preserves this distinction rather than treating every computed entry as a comparison. Nevertheless, their rubric interpretations differ: reviewer B marks forty credit confidence scores uncertain because their numerical meaning is unspecified, while reviewer A treats them as separate uncalibrated signals and assesses the displayed financial calculations. This accounts for all forty numeric-label disagreements. It illustrates measurement ambiguity rather than forty discovered arithmetic errors.

Their notes flag omitted credit risks or covenant breaches, AML closure decisions without a supplied closure rule, benign requests answered only with an offer of assistance, unstated interest conventions, and a materiality ranking based on unavailable weights. These are candidates for qualified review, not adjudicated findings. The analysis preserves both original labels and does not convert agreement into a reference answer. Automated arithmetic confor mance therefore coexists with unresolved judgments about completeness, evidence, and usefulness; the study does not collapse them into a single quality score.

## Appendix B. Further evaluation protocol

These extensions are unexecuted and are not prerequisites for interpreting the bounded systems observations as reported. They are required before stronger claims about expert-validated quality, generalization, or downstream utility. The original prospective plan is retained separately from dated execution updates; no experiment is described as externally preregistered.

Independent representation holdout. Separate new cases from the retrospective defect probes and previously inspected controls. An author or panel independent of validator implementation should establish expected behavior for valid paraphrases, unit aliases, equivalent derivations, and faults in sign, scale, period, source selection, or assumptions. Cover every supported formula and more than one independently authored template family. Freeze development and holdout membership before tuning; changes prompted by the current cash-conversion-cycle and DSCR failures belong in development. Report unsupported formats separately, and choose sample sizes against a stated precision target. A new internally generated template set alone would not establish independent financial validation.

Equal-input mechanism comparison. For content checking, give all arms the same structured record and compare schema/cached flags, arithmetic recomputation, canonical projection checking, and their composition. Preserve arithmetic-only and text-only ablations to identify what each rejects. For release control, hold content validators and approval inputs fixed while varying cached versus fresh checks and unbound versus manifest-bound coverage.

Include held-out nonnumeric stale-grounding cases, approval lifecycle events, and both promotion entry points. Measure invalid acceptance, useful-content rejection, unsupported cases, and cost by scenario and family. Paired treatments and shared templates require clustered uncertainty rather than independent-row binomial intervals. These proposed arms have not been executed.

Qualified assessment. A private collection package is prepared for the unchanged 200-record retained sample, with twenty records per domain, deterministic selection, and randomized presentation. Reviewers have not been selected and no human ratings have been collected. The plan calls for two qualified, independent human judgments per record, allowing domain-specialist panels, followed by separate adjudication. A twenty-record pilot (two per domain) precedes the remaining 180; generic rubric ambiguities may be resolved without disclosing counterpart case labels. Any changed rubric requires preserved earlier versions and independent pilot rerating. The four dimensions separate financial calculations, evidence support, jurisdiction/date applicability, and training usefulness. Forms begin blank, exclude AI annotations and private provenance, and allow reassignment outside competence. Analysis requires locked complete pairs; report per-domain counts, descriptive agreement, and explicitly weighted retainedmixture fractions without treating uncertain labels as errors. A separate excluded-record arm is needed for rejection error, with source/domain/reason strata frozen after the primary excluded corpus is available. This preparation is not a completed human study or an external preregistration.

Curation and model utility. A broader comparison should hold source inventory, tokenizer, and training budget fixed across generic curation, added financial checks, and complete selection/release processing. Evaluate naturally occurring retained defects and useful-content losses using independent adjudication. Approval eligibility must not be equated with model accuracy. For model comparisons, freeze revision, tokenizer, optimizer, checkpoint-selection rule, total tokens, and three training seeds before execution; report a matched-record sensitivity analysis separately. Evaluate independently authored cases with source, organization, period, and template-family separation where applicable. External tasks such as FinQA require a compatible adapter and documented permitted use and overlap checks. Report paired results with uncertainty clustered by source or scenario family, retaining delivery failures and excluding selection exposure from the final holdout.

Systems and replication. Extend heterogeneous document lengths and hardware beyond the current laptop work load. Measure near deduplication, tokenization, selection, and full release construction separately. Inject process termination between committed checkpoints and test chained recovery, concurrent use, and externally managed approval/revocation updates. The present post-checkpoint experiment does not cover those conditions. Independent replication requires access to the private study materials; a new environment on the original host is insufficient.

Appendix C. Detailed processing measurements

Table 8: Three repetitions per cell on the authored laptop workload. N is raw input scale; qualification reads 0.9N canonical rows. Time excludes interpreter startup, while peak RSS includes it.
<table><tr><td>N</td><td>Stage</td><td>Median seconds</td><td>Maximum RSS (MiB)</td><td>Output (MiB)</td></tr><tr><td>1,000</td><td>Audit</td><td>0.225</td><td>44.1</td><td>2.0</td></tr><tr><td rowspan="4">10,000</td><td>Bronze</td><td>0.124</td><td>48.8</td><td>2.5</td></tr><tr><td>Qualification</td><td>0.038</td><td>41.0</td><td>0.5</td></tr><tr><td>Audit</td><td>2.403</td><td>46.9</td><td>19.7</td></tr><tr><td>Bronze</td><td>1.235</td><td>53.4</td><td>24.3</td></tr><tr><td rowspan="4">100,000</td><td>Qualification</td><td>0.353</td><td>42.2</td><td>4.6</td></tr><tr><td>Audit</td><td>31.356</td><td>50.4</td><td>197.9</td></tr><tr><td>Bronze</td><td>13.485</td><td>56.6</td><td>242.7</td></tr><tr><td>Qualification</td><td>4.022</td><td>44.8</td><td>45.8</td></tr></table>

## Appendix D. Reproduction scope and missing evidence

## D.1 Recorded reproduction checks

The fifteen component diagnostics reproduce from an extracted, hash-checked source archive after installing pinned dependencies into a new virtual environment. Of the 190 focused tests that pass in the original workspace, 169 also pass without private dataset fixtures. The initial source-only run records 21 failures caused by unavailable release fixtures; these are explicitly enumerated and excluded from the portable rerun rather than counted as passes. Both runs and installation logs are retained. This verifies dependency installation and portable checks on the same host, not an independent laboratory replication. The earlier broader caller-suite run remains incomplete.

A refreshed code-only distribution contains 167 files, including implementation, configuration, documentation, portable tests, and its integrity manifest. It excludes research populations, review labels, governance evidence, and Git history; its only JSONL fixture contains six explicitly synthetic regression probes. A non-editably installed wheel built from a clean archive extraction passes all 182 shipped tests with zero skips or failures in a fresh environment outside the checkout, including generation of twelve Bronze examples and release reporting. Packaging regressions verify that the installed CLI loads the bundled audit configuration and that generated-data ignore rules preserve evaluation source files. The 113 included runtime source identities match the evaluated snapshot; two evaluation package initializers retain the narrowed public interfaces. Two adapters requiring the separate serv ing/orchestration application are outside the standalone toolkit. These packaging checks overlap with the repository tests and must not be summed as unique coverage. The refresh ran on Python 3.13.7 and macOS 27.0.1 on ARM64; Linux CI and optional tokenizer integration were not executed.

In the recorded private reproduction check, a confidential data-inclusive archive is extracted into a separate local directory. With the existing host environment, its portable wrapper reproduces the per-release and per-formula 100K audit outputs and the 1,280-case overlap result. All 190 focused tests pass from that data-inclusive extraction with zero exclusions, resolving the source-only fixture omissions. The extracted release-boundary result bytes, locked label analysis, and component diagnostics agree with their recorded counterparts; one 1,000-row audit worker is a performance smoke check. This verifies selected relocation paths and supplied data, not a new-machine replication, a complete benchmark rerun, or recreation of AI judgments. The subsequent manuscript and public-package refresh preserves the empirical inputs; it is not counted as another execution of those 190 tests.

## D.2 Unavailable corpus evidence

The primary full-corpus audit, qualification, candidate, and promotion bundles were not located in the local ev idence inventory. Larger counts described in repository documentation are therefore excluded from this paper’s verified totals. The documented exclusion process combines duplication, validation, authorization policy, and mixture selection; its aggregate rejection cannot be interpreted as an invalid-data percentage. In particular, exclusion for unbound rights evidence does not establish that a source legally forbids use.

## Appendix E. Research artifact documentation

Purpose and composition. The study materials support checks of representation consistency, supported arithmetic contracts, release decisions, and local execution. The 100,000 retained records and domain counts are described in Table 3. Additional authored cases support the filter, release-boundary, and processing experiments; they do not form an independently authored financial benchmark. The material is not intended to certify suitability for lending, compliance, investment, or customer decisions.

Collection and labels. Synthetic generators, source identifiers, transformation lineage, and validator versions are preserved in the source and release manifests. Synthetic source markers do not themselves prove ownership or factual validity. Historical validation metadata remain unchanged; the current audit supplies separate retrospective observations. The exploratory review labels are AI outputs. There are no collected qualified-human labels, and no human-participant approval or exemption is asserted for a future study.

Access and maintenance. Research artifacts remain private at the project owner’s direction. The public code license grants no release of the separate research materials. A controlled-access contact, terms, hosting, and retention arrangements have not been established. A portable core experiment would need cleared synthetic generators, protocols, runners, case-level outputs, and pinned dependencies; this is an outstanding dissemination step. Corrections should preserve earlier evidence and record new versions. The code-only distribution cannot reproduce the full empirical study without its private inputs.

## Appendix F. Artifact map

Within the confidential archive, the empirical root is outputs/forge-research-20261007/. Its REPRODUCE.txt maps the audit, component diagnostics, filter, processing, and release-boundary runners to their frozen evidence. The final-audit/ directory contains independent representation checks; ai-review/ contains locked model labels, while review/ and human-review/ contain the separate review preparations. These are private evidence locations, not public download links.

The current submission source is manuscript/paper.tex in the delivery workspace; the arXiv source bundle contains it as paper.tex, with inline figures and bibliography. The later research revision changes presentation and analysis of existing results, without replacing the archived experiments or frozen implementation. Manuscripts inside the preserved confidential archive are earlier versions. Only the separately identified Apache-licensed code distribution is a public-release candidate; neither the surrounding research archive nor the coordinator’s review kit is the public code release.