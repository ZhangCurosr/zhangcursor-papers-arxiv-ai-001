# Claim-Gated Source-Risk Auditing for Generative Search

Kainan Zhou   
Google LLC   
Mountain View, USA   
zhoumark@google.com   
Chuhong Xu   
Sony Corporate of America   
San Jose, USA   
chuhong.xu@sony.com

Gangzhen Qian Google LLC Mountain View, USA irisqian@google.com

Zhaoyi Li   
Intuit Inc   
Mountain View, USA   
lzy9776@gmail.com

Abstract—A generative search answer can cite a supported passage yet omit a source relationship that changes its interpretation. We specify a claim-gated audit of the query–source– answer tuple. An omission is resolved only when relationship evidence, answer adoption, materiality, and disclosure are all observed; incomplete evidence remains unresolved rather than being treated as independence. The specification separates this endpoint from citation support and review priority, and binds decisions to versioned evidence spans. A reference checker makes the record contract executable. On an exhaustive synthetic suite, it reproduces all 81 three-state predicate combinations and rejects 192 deliberately malformed records. Common-guard baselines and predicate ablations isolate endpoint logic from missing-evidence handling, while controlled transitions check support separation and evidence removal. These are finite contract-conformance results, not detector accuracy or evidence of improved user outcomes. We define the independent annotation, held-out evaluation, and paired utility tests still required to establish semantic validity and deployment benefit.

Index Terms—generative search, source provenance, audit specification, missing evidence, conformance testing

## I. INTRODUCTION

Generative search combines retrieval, source selection, attribution, and generation in a single answer. Existing verifiability audits ask whether citations support the statements attached to them [1]. A different question arises when a supported statement comes from a source with a relationship to the queried target. The statement may be accurate, but interpreting it as an independent assessment may still be unwarranted.

Consider an illustrative answer that recommends a product using a review written by a paid affiliate. If the answer adopts the recommendation without identifying the relationship, citation support does not resolve the missing context. Conversely, a company-authored specification can be used appropriately when its origin is clear and the answer makes a factual comparison. Neither ownership nor commercial status alone establishes an omission. The relevant unit must include the query, the source, and the answer’s treatment of that source.

Citation-generation benchmarks measure correctness and citation quality [2]. Publisher-side optimization of answer inclusion provides a further reason to retain source context [3], but visibility or persuasive language is not evidence of a particular relationship. The audit therefore requires recoverable relationship evidence and a separate judgment of whether its omission matters for the adopted claim.

This paper contributes an executable specification rather than a new neural architecture. Its first contribution is a four-predicate omission endpoint with an explicit completecase policy. Its second is a record contract that separates source evidence, answer-level judgments, and post hoc review priority. Its third is a reproducible conformance suite with common-guard baselines, component ablations, and malformedrecord tests. These components make a narrower claim than a deployed detector: given explicit judgments, the implementation preserves their meaning and reports only the validation stage actually completed.

The distinction matters for evaluation. An implementation can reproduce a formula without recognizing relationships in Web text, and a valid evidence pointer can lead to an incorrect human judgment. We test the former, not the latter. The resulting artifact is a basis for subsequent semantic evaluation; it does not estimate omission prevalence, demonstrate better answers, or certify publisher trustworthiness.

## II. RELATED WORK AND SCOPE

## A. Citation Support and RAG Evaluation

RAGTruth supplies a hallucination corpus for retrievalaugmented generation [4], while RAGAs evaluates retrieval and generation components [5]. Their content-level measurements are relevant to factual support but do not by themselves supply the archived source–target relationship required here. Support is retained as a separate field so that factual and contextual failures can be inspected together without being conflated.

ARES uses learned evaluators for retrieval-augmented systems [6]. Such evaluators could help identify candidate tuples or assist an annotator, but their predictions would remain distinct from relationship evidence. Our reference checker does not run these systems or compare their published scores: its inputs are structured judgments, whereas their evaluation tasks involve generated text and retrieved passages.

## B. Relationship Evidence and Disclosure

Research on affiliate abuse traces relationships through transaction context [7]. AdIntuition studies disclosure of likely endorsements on video pages [8]. These settings motivate a separation between finding a relationship and deciding whether the audited answer explains it adequately. A disclosure keyword elsewhere on a page is not sufficient evidence that the answer preserves the relevant context.

## C. Behavioral Tests and the Claim Boundary

CheckList uses targeted behavioral tests to expose failures that an aggregate metric can miss [9]. We adopt that testing principle at the structured-record level: change one field, register the expected endpoint transition, and preserve the rest. Unlike a natural-language benchmark, our exhaustive state space has specification-derived outcomes. Agreement with those outcomes establishes implementation conformance only; it cannot establish that the predicates are meaningful or reliably recoverable from natural text.

## III. AUDIT SPECIFICATION AND REFERENCE CHECKER

## A. Unit, Predicates, and Complete-Case Endpoint

Let $r = ( q , u , a )$ denote a query, an archived source available to the generator or cited in its output, and the resulting answer. Multiple sources produce separate tuples. A versioned codebook assigns four predicates in {0, 1, U}: I records a query-relevant relationship; A records adoption of the source’s claim; M records whether the relationship affects interpretation; and D records adequate disclosure in the answer. Here U means that the available record does not resolve the judgment, not that the relationship is absent.

We use the complete-case endpoint

$$
R _ { \mathrm { O } } ( r ) = \left\{ { \begin{array} { l l } { \mathrm { U } , } & { { \mathrm { i f ~ a n y ~ o f ~ } } I , A , M , D { \mathrm { ~ i s ~ U } } , } \\ { I A M ( 1 - D ) , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{1}
$$

Thus (1, 1, 1, 0) is an omission and (1, 1, 1, 1) is not. The tuple (0, 1, 1, U) remains unresolved even though short-circuit Boolean reasoning could return zero. This is a deliberate documentation requirement: a resolved record must contain all four judgments. It is stricter than merely determining whether the conjunction can be true.

The endpoint concerns omission of material context in the observed answer. It does not attribute intent, identify deception, or establish a causal effect of the source on the generator. Adoption and materiality require their own evidence; neither follows automatically from a relationship label.

## B. Evidence Vector and Review Priority

The record retains five dimensions,

$$
\mathbf { d } ( r ) = \left( d _ { Q } , d _ { I } , d _ { A } , d _ { M } , d _ { G } \right) \in \left( \left[ 0 , 1 \right] \cup \left\{ \mathrm { U } \right\} \right) ^ { 5 } .\tag{2}
$$

They represent query susceptibility, relationship evidence, adoption, materiality, and disclosure deficit. In the reference checker, resolved entries map explicitly as $d _ { I } = I , d _ { A } = A$ $d _ { M } = M ,$ , and $d _ { G } = 1 - D ;$ an unresolved predicate maps to U. These entries are not class probabilities and need not sum to one. A later graded codebook would require a separately validated mapping to the predicates.

When all five dimensions are observed, review priority is

$$
{ \cal T } _ { \mathrm { p o s t } } ( r ) = \sum _ { k } w _ { k } d _ { k } , \quad w _ { k } \geq 0 , \quad \sum _ { k } w _ { k } = 1 .\tag{3}
$$

This score orders review; it never changes (1). The checker returns an unresolved priority when any dimension is missing, even when its weight is zero. The experiments use equal weights as an illustrative configuration, not as an optimized policy. Query susceptibility can support sampling, but answerdependent dimensions cannot select the evidence used to generate that same answer.

![](images/15763c27546d4cb10d309590e62dba04b143eb08458b2daabc01bdee7239a5d2.jpg)  
distinct outputs for empirical sampling and review.Fig. 1. Evidence flow for one tuple. The relationship, adoption, materiality, and disclosure fields resolve $R _ { \mathrm { O } } ; \bar { d } _ { Q }$ affects review priority only. The reference checker consumes explicit judgments, not raw-text predictions.

## C. Module Interfaces and Execution Order

Figure 1 separates the inputs and outputs. An archive adapter stores the query, source and answer snapshots, and pipeline configuration. An annotation adapter supplies the codebook version, predicate values, and evidence spans. The contract checker verifies representation and traceability. Only a conforming record reaches the endpoint resolver and priority scorer; their outputs then enter a claim ledger.

The execution order is deterministic: validate required fields and hashes; validate predicate types and evidence offsets; verify the dimension mapping and weights; resolve $R _ { \mathrm { O } } ;$ compute priority; and attach the completed validation gate. A malformed record is rejected with a reason rather than converted into a semantic label. A structurally valid record with unresolved judgments is accepted and returns U. This distinction prevents parsing failures from silently becoming negative findings.

The implemented adapter generates synthetic records and supplies their judgments directly. Web collection, entity resolution, natural-language annotation, and generation remain external interfaces, not implemented capabilities of this checker. An optional learned annotator would need to be frozen and evaluated against independent labels before any accuracy claim.

## D. Source Roles and Admissible Evidence

Table I retains the operational source-role taxonomy. Roles determine what evidence to seek and what disclosure the codebook should require. They are not standalone risk scores. Current ownership evidence may resolve I, but A, M, and D still concern the particular answer.

Evidence must be recoverable from an archived observation: for example, ownership records, sponsorship disclosures, affiliate records, or reproducible identity metadata. Tone, rank, similarity, and commercial status may guide investigation but do not resolve a relationship. Each decision is tied to the source version and time window used for the answer. Later evidence creates a new record version rather than rewriting the earlier observation.

TABLE I  
OPERATIONAL SOURCE-ROLE TAXONOMY
<table><tr><td>Role</td><td>Operational interpretation</td></tr><tr><td>Target owner</td><td>Observable ownership of the target-related source. Evidenced related party promoting or comparing</td></tr><tr><td>Target affiliate</td><td>the target.</td></tr><tr><td>Sponsored third party</td><td>An evidenced material sponsorship relationship.</td></tr><tr><td>Advocacy aligned</td><td>Observable organizational or cause alignment. Commercial publisher; relationship subtype un-</td></tr><tr><td>Commercial third party</td><td>specified.</td></tr><tr><td>Apparently independent</td><td>No visible ownership claim, but an archived material tie.</td></tr><tr><td>No observed relation</td><td>Prescribed search found no tie; independence unverified.</td></tr><tr><td>Community / UGC</td><td>Identity or coordination remains uncertain.</td></tr></table>

## E. Record Contract and Invariants

The contract stores query and tuple identifiers, original and canonical URLs, rank, collection time, source and answer snapshots, pipeline metadata, source role, codebook version, predicate judgments, and evidence offsets. Every resolved judgment has a recoverable span; unresolved synthetic judgments use JSON null. Missing keys, the string "U", and numeric zero are not interchangeable.

The reference implementation derives snapshot digests with SHA-256. A tuple digest binds the query identity, source and answer digests, and pipeline configuration; a record digest additionally binds annotations and other stored fields. An annotation revision retains the tuple identity, changes the record identity, and points to its parent. Hashes support change detection, not authenticity or semantic correctness. Appendonly storage enforcement remains a deployment responsibility.

Six invariants constrain the design: citation support is separate from source risk; findings remain tuple-local; decisions retain evidence pointers; missing evidence cannot create a resolved low-risk conclusion; generation and post hoc review remain separate; and stronger empirical claims require stronger validation. The conformance suite tests the specified record transitions, not every possible implementation of these princi ples.

## IV. EXECUTABLE CONFORMANCE EVALUATION

## A. Questions, Fixtures, and Oracle

The evaluation asks whether the checker implements the declared endpoint, whether simpler rules preserve that contract, and whether the record validator rejects selected structural defects. We enumerate the complete state space $\{ 0 , 1 , \mathrm { U } \} ^ { 4 } \colon$ : 81 tuples comprising one positive, 15 negatives, and 65 unresolved cases under (1). Each tuple has a synthetic query, source and answer snapshot, version metadata, and explicit judgment spans. The text contains fixture annotations, not naturally occurring Web claims.

There is no training, preprocessing of a real dataset, or train/test split. Enumerating a finite contract requires neither model fitting nor statistical sampling. Expected outputs are stored separately using membership in the explicit positive set {(1, 1, 1, 0)} for fully observed states and U otherwise. This is an independently expressed test oracle for the same rule, not independent semantic ground truth. The artifact contains no human labels and no live-Web observations.

TABLE II  
CONTRACT COMPARISONS ON THE EXHAUSTIVE STATE SPACE
<table><tr><td>Method</td><td> $E _ { 1 6 }$ </td><td> $F _ { 6 5 }$ </td><td> $U _ { 8 1 }$ </td></tr><tr><td>Complete-case reference</td><td>0</td><td>0</td><td>65</td></tr><tr><td>Support-only</td><td>1</td><td>65</td><td>0</td></tr><tr><td>Relationship-only</td><td>7</td><td>38</td><td>27</td></tr><tr><td>Disclosure-only</td><td>7</td><td>38</td><td>27</td></tr><tr><td>Support-only + guard</td><td>1</td><td>0</td><td>65</td></tr><tr><td>Relationship-only + guard</td><td>7</td><td>0</td><td>65</td></tr><tr><td>Disclosure-only + guard</td><td>7</td><td>0</td><td>65</td></tr><tr><td>Drop I + guard</td><td>1</td><td>0</td><td>65</td></tr><tr><td>Drop A + guard</td><td>1</td><td>0</td><td>65</td></tr><tr><td>Drop M + guard</td><td>1</td><td>0</td><td>65</td></tr><tr><td>Drop  $D + { \mathrm { g u a r d } }$ </td><td>1</td><td>0</td><td>65</td></tr><tr><td>Drop  $d _ { Q }$  (priority only)</td><td>0</td><td>0</td><td>65</td></tr></table>

$\overline { { E _ { 1 6 } } } \mathrm { : }$ : mismatches on 16 resolved contract states. ${ \overline { { F _ { 6 5 } } } } \colon$ resolutions contrary to the complete case contract on 65 incomplete states. $U _ { 8 1 } \mathbf { : }$ : unresolved outputs among all 81 states. These are specification counts, not estimated detector error rates.

All experiments run in Python 3.13.5 using only the standard library. The supplied scripts generate JSONL fixtures, the expected state table, per-case predictions, mutation logs, CSV summaries, and a checksum manifest. The paper’s experimental tables are generated from these outputs. No paid API, pretrained model, or network access is required.

## B. Baselines and Common-Guard Comparison

We compare three explicit rule projections. Support-only returns zero when passage support is present; all main fixtures have support fixed to one. Relationship-only returns I. Disclosure-only returns 1 − D when D is observed and U otherwise. The last rule is an oracle-feature proxy for a disclosure-only policy, not an executed keyword detector. All methods consume the same structured fixture judgments, so the comparison measures information discarded by each rule rather than text-understanding performance.

Table II reports mismatches on the 16 resolved contract states, resolutions issued on the 65 states that the contract requires to remain unresolved, and the total number of unresolved outputs. We then apply the identical four-predicate completeness guard to all three baselines. This controls for missingness policy instead of crediting the reference method merely for having a stricter guard.

Without the guard, support-only misses the sole positive state and resolves all 65 incomplete states. Relationship-only and disclosure-only each disagree on seven resolved states and resolve 38 incomplete states. With the common guard, those unsupported resolutions disappear, but the resolved-state mismatches remain. A guarded support-only rule agrees on 15 of 16 complete states simply because the positive state is rare in this enumeration. Its apparently high agreement therefore does not show that support can replace relationship auditing.

TABLE III  
FINITE CONFORMANCE CHECKS, NOT SEMANTIC ACCURACY
<table><tr><td>Check family</td><td>Passed / tested</td></tr><tr><td>Complete truth table</td><td>81/81</td></tr><tr><td>Evidence removal</td><td>216/216</td></tr><tr><td>Support separation</td><td>81/81</td></tr><tr><td>Query-priority separation</td><td>891/891</td></tr><tr><td>Controlled binary pairs</td><td>32/32</td></tr><tr><td>Canonical round trip</td><td>81/81</td></tr><tr><td>Annotation-version lineage</td><td>81/81</td></tr><tr><td>Reject malformed records</td><td>192/192</td></tr></table>

## C. Predicate Ablations and Coverage Trade-off

Each endpoint ablation preserves the same completeness guard and removes one Boolean requirement: dropping I, A, or M substitutes one; dropping D substitutes zero. Each introduces one additional positive on the 16 fully observed states. For example, dropping A incorrectly treats (1, 0, 1, 0) as an omission although the answer does not adopt the source claim. Removing $d _ { Q }$ from the endpoint interface changes no outcome, as intended; this is a separation control rather than evidence that query susceptibility is useful for triage.

A legitimate alternative is to resolve a negative as soon as $I \ = \ 0 ,$ A = 0, $M \ = \ 0 ,$ , or D = 1, even if another predicate is unknown. That short-circuit policy resolves 66 of 81 states, compared with 16 for our complete-case contract. Its 50 additional resolved negatives are not Boolean errors. They expose the cost of requiring a fully documented audit record. These synthetic coverage counts are not deployment rates, and the stricter policy is not claimed to be universally preferable.

## D. Controlled Transitions and Structural Mutations

Table III summarizes the completed checks. The resolver matches all 81/81 oracle outputs. Replacing one observed predicate with U produces 216 directed evidence-removal transitions; every destination remains unresolved. Toggling passage support leaves all 81 endpoints unchanged. Sweeping d<sub>Q</sub> from zero to one in steps of 0.1 yields 891 checks without changing an endpoint. These counts share base states and must not be treated as independent observations.

The binary suite flips one predicate while fixing the other three, giving 32 controlled pairs. For disclosure, changing D from zero to one removes the omission only when $I = A =$ M = 1; the other seven pairs remain negative. The equivalent relationship, adoption, and materiality pairs test their registered directions. These are record-level interventions with supplied judgments, not paired generated answers evaluated by users.

All 81 valid records survive canonical serialization and a version-lineage check. The mutation suite applies 12 defect types to each of the 16 fully observed records, producing 192 malformed cases. Defects cover identity, snapshot and record checksums, missing codebook or pipeline fields, invalid nullable labels, inconsistent dimensions, missing or invalid spans, mismatched quotes, and unnormalized weights. All 192 are rejected. Except for deliberate checksum or identity defects, tion domain and a prespecified risk band. Domains registered for<sub>records are resealed after mutation so that a stale outer digest</sub> calibration do not enter the test partition. The report also states<sub>cannot mask the intended structural test. This curated suite is</sub> not arbitrary-input fuzzing or a security proof.

![](images/d87f6501256792c0e783d136ce622adb421be1923ff01e12d0c7c1b478d956f7.jpg)  
Fig. 2. Required semantic validation sequence. These human-label and modelevaluation stages remain future work; they are not established by the synthetic conformance checks.

## E. Measured Checker Cost and Interpretation

sagreements. The codebook permits U and requires an evidence<sub>A single-process microbenchmark on an Intel Xeon Platinum</sub> span for every resolved predicate. Reports preserve pre-adjudication<sub>8573C</sub> <sub>host</sub> <sub>records</sub> <sub>21</sub> <sub>repeated</sub> <sub>batch</sub> <sub>means</sub> <sub>after</sub> <sub>warm-up.</sub> The endpoint loop processes 16,200 tuples per repeat; the full record checker processes 810. The median batch-mean cost is 0.49 µs per tuple for the endpoint and 51.20 µs for without its archived observation cannot resolve a source role.validation plus auditing. These measurements exclude archive I/O, retrieval, generation, and annotation; they are neither endto-end latency nor per-request tail-latency estimates.

The experiment now supplies an executable record checker and numerical comparisons, but its success criterion is agreement with a chosen specification. An incorrect materiality label can still pass every structural test. No confidence interval or significance test is attached to the exhaustive counts, and repeated template cases are not used to claim a large empirical sample. The completed claim is finite conformance of the supplied implementation.

## V. SEMANTIC VALIDATION AND APPLICATION BOUNDARY

## A. Independent Labels and Held-Out Evaluation

Figure 2 shows the next validation stage, which has not been completed. A Web study must archive observations before annotation, stratify by application domain and prespecified risk bands, and record inclusion probabilities. Target families, related domains, and query templates should not leak from calibration into the held-out partition. Synthetic fixtures remain outside the human-gold test set.

At least two trained annotators should label each tuple independently, blinded to model scores and one another’s decisions. They must be able to assign U and cite evidence for resolved predicates. Adjudication follows, while reliability is reported on the original independent labels using a coefficient appropriate to the label scale [10]. The codebook, pipeline, and operating thresholds must be frozen before opening the test labels.

The primary semantic measurements should separate each predicate from the derived endpoint. Confusion matrices, classspecific counts, unresolved coverage, and domain-clustered intervals expose errors hidden by aggregate classification measures [11]. A learned judge should not become its own gold standard: work on LLM-based evaluation also reports potential evaluator biases [12]. Any judge-assisted annotation would therefore need a separately labeled audit subset and explicit reporting of assistance.

TABLE IV  
CLAIM GATES AND EVIDENCE STATUS
<table><tr><td>Claim</td><td>Required evidence</td><td>Status</td></tr><tr><td>Finite conformance</td><td>Executable oracle, fixture predictions, mutation logs</td><td>Completed</td></tr><tr><td>Real-Web finding</td><td>Archived observations, timestamps, re- trieval provenance</td><td>Not tested</td></tr><tr><td>Semantic validity</td><td>Independent labels, reliability, adjudi- cation</td><td>Not tested</td></tr><tr><td>Detector accuracy</td><td>Frozen system, held-out labels, class metrics and intervals</td><td>Not tested</td></tr><tr><td>Application benefit</td><td>Matched interventions, omission and utility measures</td><td>Not tested</td></tr></table>

## B. Tool Utility and Deployment Decisions

Application benefit requires a matched comparison of generated answers before and after an audit-informed intervention, with generation settings held fixed and evaluators blinded to condition. The primary endpoint is material relationship omission; utility measures include answerability, retained evidence, factual support, abstention, and end-to-end latency. Merely suppressing answers or removing related sources does not establish a useful defense.

A policy may return PASS, CONTEXTUALIZE, REVIEW, or ABSTAIN. PASS means that the configured review require ment is met for the observed tuple, not that the publisher is independent or universally reliable. CONTEXTUALIZE can request adequate relationship attribution while retaining useful evidence. REVIEW preserves unresolved cases; ABSTAIN requires an explicit answerability or evidence policy. These actions are proposed interfaces, not validated production controls.

## C. Limitations, Falsifiability, and Release

Table IV makes the remaining claim boundaries explicit. The present artifact validates typed synthetic records and deterministic logic. It does not establish source discovery, temporal validity of real relationships, annotation reliability, multilingual transfer, or changes in reader interpretation. Materiality and adequate disclosure remain task-dependent judgments. A hash valid record can preserve an erroneous or outdated judgment perfectly.

Several findings would count against the specification: annotators cannot apply M consistently; the role taxonomy systematically excludes relevant ties; a simpler provenance rule performs equally well against independent labels; the complete-case requirement causes impractical review load; or contextualization harms answer utility without reducing omissions. Any such result requires revising the affected rule rather than treating conformance as proof of usefulness.

The accompanying package includes the checker, fixture generator, test oracle, per-case outputs, aggregate CSVs, and table-generation script. A checksum manifest binds the files used for each run; it is not a digital signature.

The future Web release must additionally document archive permissions, exclusions, codebook changes, adjudication, and split assignments. Restricted evidence can retain authorized pointers, but redaction must not silently change the reported sample.

## VI. CONCLUSION

Source-risk auditing asks whether a generated answer preserves a material relationship, not only whether a citation supports its words. We formalize that question at the query– source–answer level and implement a record checker with explicit missingness, evidence pointers, and separate review priority. Exhaustive predicate enumeration, common-guard comparisons, component ablations, and structural mutations establish finite conformance of the implementation. They also expose the coverage cost of a complete-case policy. Independent human labels and paired answer-level evaluations remain necessary before claiming semantic accuracy, omission prevalence, or application benefit.

## ACKNOWLEDGMENT

We acknowledge that this work was supported by Beijing Institute of Technology, Zhuhai (Project No. 2026039DCXM).

## REFERENCES

[1] N. F. Liu, T. Zhang, and P. Liang, “Evaluating verifiability in generative search engines,” in Findings of EMNLP, 2023, pp. 7001–7025, doi: 10.18653/v1/2023.findings-emnlp.467.

[2] T. Gao, H. Yen, J. Yu, and D. Chen, “Enabling large language models to generate text with citations,” in Proc. EMNLP, 2023, pp. 6465–6488, doi: 10.18653/v1/2023.emnlp-main.398.

[3] P. Aggarwal, V. Murahari, T. Rajpurohit, A. Kalyan, K. Narasimhan, and A. Deshpande, “GEO: Generative engine optimization,” in Proc. ACM SIGKDD, 2024, pp. 5–16, doi: 10.1145/3637528.3671900.

[4] C. Niu et al., “RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models,” in Proc. ACL, 2024, pp. 10862–10878, doi: 10.18653/v1/2024.acl-long.585.

[5] S. Es, J. James, L. Espinosa Anke, and S. Schockaert, “RAGAs: Automated evaluation of retrieval augmented generation,” in Proc. EACL: System Demonstrations, 2024, pp. 150–158, doi: 10.18653/v1/2024.eacldemo.16.

[6] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “ARES: An automated evaluation framework for retrieval-augmented generation systems,” in Proc. NAACL, 2024, pp. 338–354, doi: 10.18653/v1/2024.naacllong.20.

[7] N. Chachra, S. Savage, and G. M. Voelker, “Affiliate Crookies: Characterizing affiliate marketing abuse,” in Proc. ACM Internet Measuremen Conf., 2015, pp. 41–47, doi: 10.1145/2815675.2815720.

[8] M. Swart, Y. Lopez, A. Mathur, and M. Chetty, “Is this an ad?: Automatically disclosing online endorsements on YouTube with AdIntuition,” in Proc. CHI, 2020, pp. 1–12, doi: 10.1145/3313831.3376178.

[9] M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh, “Beyond accuracy: Behavioral testing of NLP models with CheckList,” in Proc. ACL, 2020, pp. 4902–4912, doi: 10.18653/v1/2020.acl-main.442.

[10] R. Artstein and M. Poesio, “Inter-coder agreement for computational linguistics,” Computational Linguistics, vol. 34, no. 4, pp. 555–596, 2008, doi: 10.1162/coli.07-034-R2.

[11] M. Sokolova and G. Lapalme, “A systematic analysis of performance measures for classification tasks,” Information Processing & Management, vol. 45, no. 4, pp. 427–437, 2009, doi: 10.1016/j.ipm.2009.03.002.

[12] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu, “G-Eval: NLG evaluation using GPT-4 with better human alignment,” in Proc. EMNLP, 2023, pp. 2511–2522, doi: 10.18653/v1/2023.emnlp-main.153.