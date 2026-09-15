# Four Ledgers, Not One Score: Responsible Communication of LLM-Judge Calibration in Biomedical ML

Sidi Chang<sup>\*</sup> <sup>†</sup> schang@blossomai.co Blossom AI San Francisco, CA, USA

Peiying Zhu

peiying@blossomai.co

Blossom AI

San Francisco, CA, USA

## Abstract

Synthetic perturbations appear to ofer inexpensive calibration data for LLM evaluators in biomedical ML, where expert review is scarce. Yet a planted mutation key is neither a detector output nor automatically human ground truth. We formalize four distinct ledgers: planted perturbations, independent detector outputs, source-linked human dispositions, and human-added discoveries. We then audit the evaluation design, scoring code, read paths, and current human records of a private synthetic Japanese care-handof workflow. The factory stored 69 planted error cards across 47 targets. Final review covers 22 targets and contains 22 confirmed imported proposals, 9 rejected proposals, and 79 human-added cards; only 3 reviewed targets are double annotated. Passing imported plant keys to a generic detector scorer yields 22/(22 + 9) = 0.710 and 22/(22 + 79) = 0.218. A direct audit identity shows that these values are proposal-confirmation yield and submitted-ledger composition, not judge precision and recall, because no independent detector realization was preserved for the audited proposals in the available records. The audit also finds source-name collisions, row shadowing, forced severity, vacuous ratio defaults, and unsupported zero-support field weights. We contribute a provenance-aware claim audit, a storage contract, and a minimum calibration gate for responsibly communicating biomedical ML capability claims. This single-workflow forensic case is an existence proof of a failure mode, not an estimate of its prevalence: existing human work supports an exploratory audit of synthetic proposals, but not LLM-judge operating characteristics, clinical validity, corpus prevalence, or robust inter-annotator agreement.

Keywords: LLM-as-a-judge, biomedical machine learning, evaluation provenance, synthetic perturbations, measurement validity

## 1 Introduction

LLM judges promise scalable evaluation when biomedical outputs resist exact matching and expert review is expensive. Synthetic data seems to make calibration cheap: plant a known omission or fabrication, ask a judge to find defects, and compare the output with the key. This can be valid only if the plant, the judge output, and the human decision remain separate records.

That separation is easy to lose. A list of planted cards may be imported into a table named for judge results, displayed as judgeCards, and consumed by a precision/recall function. Reviewers can confirm, reject, or add cards. The resulting counts have the algebraic shape of true positives, false positives, and false negatives, but their scientific meaning depends on who produced each card and how it was selected. Metric-shaped values can communicate a biomedical capability that was never measured.

We treat this as a provenance and responsiblecommunication problem. For each item, we distinguish what the synthetic factory intentionally planted (P), what an independent detector actually emitted (D), how humans disposed of source-linked proposals (C, R), and what humans discovered beyond them (A). Human-centered evaluation research shows that framing, instructions, rater populations, and task constructs afect reported outcomes [10, 11, 4]. Model judges likewise show systematic biases [13, 2, 3]. Our contribution precedes model choice: an evaluator cannot have its precision or recall estimated when its outputs are not observed as an independent ledger.

We use a private Japanese care-handof workflow as a forensic case study. Synthetic spoken reports are paired with structured six-field notes, and controlled omissions or fabrications are created for stress testing. We inspect stored aggregates, handof specifications, scoring semantics, and relevant read-path and annotation-interface code. We do not rank a model or certify a clinical system. We ask what quantities the current records identify and how those quantities should be communicated.

Our contributions are: (i) a four-ledger measurement model and claim-audit identity; (ii) a traceable audit showing what current human work does and does not establish; (iii) implementation mechanisms that turn provenance ambiguity into plausible metrics; and (iv) a minimum storage and reporting gate for judge-calibration claims in biomedical ML.

## 2 Related work

Human evaluation guidance calls for explicit constructs, criteria, rater recruitment, agreement, and uncertainty [11,

4]. Clinical note evaluation particularly needs grounded criteria that surface omission and hallucination beyond lexical similarity [1, 9, 14]. Static audio metrics may also weakly predict situated preferences [6].

LLM judges can correlate with human preferences but remain sensitive to order, style, verbosity, and model relationships [13, 2]. Factuality reliability varies across task, prompt, and model [3]; surveys therefore treat LLM-asjudge as a measurement system needing meta-evaluation [5]. Synthetic examples can train quality classifiers or provide weak supervision [12, 8, 7], but a generated label remains a product of a policy. It can be useful without being an unbiased sample of natural errors, an independent model prediction, or final human truth.

## 3 Four-ledger measurement model

For item $i ,$ let $\mathcal { P } _ { i }$ be intentionally planted perturbations and $\mathcal { D } _ { i }$ cards independently emitted by detector g. A disposition ledger $\mathcal { H } _ { i }$ stores tuples $( q , s , o )$ for reviewed card $q ,$ explicit source $s \in \{ P , D \}$ , and outcome $o \in$ {confirm, reject, uncertain}. Let $\mathcal { C } _ { i } ^ { s }$ and $\mathcal { R } _ { i } ^ { s }$ be sourcespecific confirmed and rejected cards, and $\mathcal { A } _ { i } ^ { s }$ cards added beyond the displayed source set. Additions are reviewer assertions unless separately adjudicated. Treating $\mathcal { A } _ { i } ^ { D }$ as detector misses additionally requires a documented, suficiently complete miss-search process.

If all detector proposals are adjudicated, detector microprecision is

$$
\widehat { \mathrm { P r e c } } _ { D } = \frac { \sum _ { i } | \mathcal { C } _ { i } ^ { D } | } { \sum _ { i } ( | \mathcal { C } _ { i } ^ { D } | + | \mathcal { R } _ { i } ^ { D } | ) } .\tag{1}
$$

Under the stronger complete-search assumption, microrecall is

$$
\widehat { \mathrm { R e c } } _ { D } = \frac { \sum _ { i } | \mathcal { C } _ { i } ^ { D } | } { \sum _ { i } ( | \mathcal { C } _ { i } ^ { D } | + | \mathcal { A } _ { i } ^ { D } | ) } .\tag{2}
$$

When humans instead review plant proposals, the identified quantity is proposal-confirmation yield,

$$
\widehat { Y } _ { P } = \frac { \sum _ { i } | { \mathcal { C } } _ { i } ^ { P } | } { \sum _ { i } ( | { \mathcal { C } } _ { i } ^ { P } | + | { \mathcal { R } } _ { i } ^ { P } | ) } .\tag{3}
$$

The human-added share of the submitted final-review ledger is

$$
\widehat { S } _ { A } = \frac { \sum _ { i } | \mathcal { A } _ { i } ^ { P } | } { \sum _ { i } ( | \mathcal { C } _ { i } ^ { P } | + | \mathcal { A } _ { i } ^ { P } | ) } .\tag{4}
$$

Neither is a detector operating characteristic.

Audit identity (detector non-identifiability). Suppose the stored proposal list supplied to Equations 1–2 is ${ \mathcal { P } } _ { i } ,$ source-P human dispositions are treated as source-D dispositions, and $\mathcal { D } _ { i }$ is not separately observed. Then numerical “precision” reduces to $\widehat { Y } _ { P }$ and numerical “recall” to the confirmed-plant share of the submitted final-review ledger. Neither identifies an operating characteristic of $g .$

Derivation. Substitute $\mathcal { P } _ { i }$ for $\mathcal { D } _ { i }$ and $\mathcal { H } _ { i } ^ { P }$ for $\mathcal { H } _ { i } ^ { D }$ . Confirmed and rejected elements partition reviewed plants, so Equation 1 becomes Equation 3. Equation 2 becomes $\textstyle \sum _ { i } | \dot { \mathcal { C } } _ { i } ^ { P } | / ( \sum _ { i } | \mathcal { C } _ { i } ^ { P } | + \sum _ { i } | \dot { \mathcal { A } } _ { i } ^ { P } | )$ . No term depends on an independent realization of $g ;$ detectors with diferent outputs are observationally equivalent under the stored record. □

A sensitivity-to-plants measure would require both ledgers and a versioned matching rule M:

$$
\widehat { \mathrm { S e n s } } _ { D \mid P } = \frac { \sum _ { i } \left| M ( \mathcal { D } _ { i } , \mathcal { C } _ { i } ^ { P } ) \right| } { \sum _ { i } | \mathcal { C } _ { i } ^ { P } | } .\tag{5}
$$

It is not computable when $\mathcal { D }$ was never independently preserved.

## 4 Case study and audit method

The case converts synthetic Japanese spoken care reports into Focus, Subjective, Objective, Assessment, Intervention, and Plan fields. Factory logic creates faithful targets and controlled error cards. An internal interface supports human review. We triangulated: (1) a frozen read-only database audit; (2) current final aggregate annotations; (3) rubric and handof contracts; and (4) scorer, read-path, and interface code. A deterministic count sheet reproduces displayed ratios and Wilson intervals but does not independently query or validate the database.

The factory population contains 47 targets with 69 planted cards: 47 labeled fabricated and 22 omitted. The available records preserve no independent detector realization for proposals used in the audited final workflow. We select one current final annotation per target by an explicit rule: prefer approved; otherwise use the latest completed record. This yields 22 reviewed targets, all from one cohort. Other cohorts with plants have no usable final verification under this rule. We also require proposal payloads to contain the expected error-card shape so newer unrelated generic outputs cannot shadow the intended record.

Separate workflows remain separate: 12 candidate-note corrections and 75 spoken-input realism records, each with one independent label per item. Correction concerns target editing; realism concerns audio plausibility and the oral/written boundary. Neither adjudicates the independent detector ledger required by Equations 1–2.

We report counts, ratios, and Wilson 95% intervals. Intervals are descriptive at the displayed item or card grain: they ignore clustering within target, non-random selection, and annotator uncertainty. With only three overlapping targets, we do not estimate agreement.

## 5 What current records establish

Among 22 final reviewed targets, humans disposed of 31 imported proposals: 22 confirmed and 9 rejected. Proposalconfirmation yield is 71.0% (Wilson 95% interval 53.4–

Table 1: Record meaning follows producer and selection mechanism, not variable name.
<table><tr><td>Ledger</td><td>Producer</td><td>Supports</td><td>Does not support alone</td></tr><tr><td>Plant P</td><td>synthetic factory</td><td>intervention coverage and reviewed plant yield</td><td>detector output or natural prevalence</td></tr><tr><td>Detection D</td><td>pinned judge</td><td>detector proposals and support</td><td>correctness without adjudication</td></tr><tr><td>Disposition H</td><td>human reviewer</td><td>source-specific confirmation/rejection</td><td>recall without miss search</td></tr><tr><td>Discovery A⁸</td><td>human reviewer</td><td>submitted concerns beyond source s</td><td>prevalence under partial review</td></tr></table>

83.9%); rejection is 29.0% (16.1–46.6%). These are singlereviewer dispositions conditional on a selected slice. They can diagnose possible mismatch among plants, rubric, and reviewer, but are not population validity or detector precision.

Reviewers added 79 cards, or 3.59 per reviewed target. The submitted final-review ledger contains 101 cards: 22 confirmed imports and 79 additions. Additions comprise 78.2% (69.2–85.2%). This shows that reviewers asserted many concerns beyond displayed plants. It is not a detector false-negative rate because the displayed list was not an independent detector output and miss-search completeness was not established.

Coverage is 22/47 factory targets, or 46.8% (33.3– 60.8%). Unreviewed targets are not negatives. At card grain, 38 of 69 plants lack a disposition. If all 38 would be rejected or all confirmed, overall plant confirmation ranges from 31.9% to 87.0%; this is a deterministic missing-review bound, not a confidence interval. Only 3/22 reviewed targets have two final annotations, an overlap of 13.6% (4.7–33.3%).

If imported plants are mislabeled as detector predictions, a generic scorer receives T P = 22, F P = 9, and F N = 79, emitting 0.710 and 0.218. Table 2 states what these ratios actually identify.

Factory cards are 68.1% fabricated and 31.9% omitted; additions are 93.7% fabricated and 6.3% omitted. The 25.6-point diference in fabricated share is descriptive. Sources have diferent selection mechanisms and cards cluster within targets, so it does not estimate natural prevalence or a significant detector bias.

Separate workflows do not repair the ledger. In 12 correction items, one rater changed at least one field in 8, marked hallucination in 3, and marked a missing highrisk fact in 2. The 75-item realism workflow rates spoken input and includes mixed target cohorts. Pooling either workflow into target correctness or judge calibration would change the construct.

## 6 Implementation failures

The failure is not one arithmetic bug but a chain of locally plausible choices.

Source/name collision. The interface and scorer call proposals judgeCards, while observed final-workflow rows contain imported factory keys. Storage location and variable name do not prove producer identity.

Shape-dependent row shadowing. For two targets, later generic payloads could shadow older error-card payloads when a reader selected the newest row regardless of schema. Selecting by required shape and source before recency recovers the intended rows. This is a read-path defect, not model or reviewer quality.

Forced severity. The human-addition form does not elicit severity and stores every addition as high. Severity is therefore a software default, not a human or clinical judgment.

Vacuous ratios and unexercised fields. The scorer returns 1.0 when a precision or recall denominator is zero and assigns a neutral weight of 1.0 to fields with zero support. Both values can be miscommunicated as perfect performance. Presentation should emit na, support counts, and explicit states for no error, not reviewed, parse failure, and unexercised field.

Construct boundaries. The rubric permits conversion from dialectal or casual speech to standard written clinical Japanese. Verbatim evidence checks can mislabel valid normalization as fabrication. Realism items may also omit record-only facts from speech while paired notes retain them. Omission scoring requires per-fact oral-requirement labels.

## 7 Calibration gate

A calibration-ready system needs four immutable records. Plant ledger: item and content hashes; plant ID; mechanism and code version; intended type, field, and fact; pre- and post-mutation values; policy and seed; and visibility rules. Plants remain hidden during detector inference and blind review.

Detection ledger: detector card IDs; pinned model, prompt, temperature, and code; input hashes; raw structured output and parse status; predicted type, field, evidence, and confidence where meaningful. An empty detector output is an observed outcome, not a missing row.

Table 2: Claim contract for the observed ratios.
<table><tr><td>Ratio</td><td>Identified quantity</td><td>Unsupported inference</td></tr><tr><td>22/31</td><td>reviewed proposal confirmation</td><td>judge precision</td></tr><tr><td>22/101</td><td>confirmed-plant share</td><td>judge recall</td></tr><tr><td>79/101</td><td>human-added share</td><td>detector false-negative rate</td></tr><tr><td>22/47</td><td>observed target coverage</td><td>population accuracy</td></tr><tr><td>3/22</td><td>duplicate-review coverage</td><td>robust agreement</td></tr></table>

Table 3: Implementation mechanisms that can create overstated biomedical capability claims.
<table><tr><td>Risk</td><td>Observed failure mode</td><td>Minimum repair</td></tr><tr><td>Source collision</td><td>plants named judgeCards</td><td>explicit producer and detector-run ID</td></tr><tr><td>Row shadowing</td><td>newer incompatible payload wins</td><td>select schema/source before recency</td></tr><tr><td>Forced severity</td><td>additions stored as high</td><td>elicit or store unelicited</td></tr><tr><td>Vacuous ratio</td><td>zero denominator returns 1.0</td><td>NA plus numerator/denominator</td></tr><tr><td>Unexercised field</td><td>zero support stored as neutral 1.0</td><td>null value and support mask</td></tr><tr><td>Oral/record mismatch</td><td>note-only facts scored against speech</td><td>per-fact source requirement</td></tr></table>

Disposition ledger: for every detector card, confirm/reject/uncertain, anonymized reviewer role, rubric version, evidence, duration, round, and adjudication lineage. Revisions append instead of overwrite.

Discovery ledger: human-added cards stored independently. Match them to plants and detections only after blind review using a versioned hierarchy: exact fact ID; compatible type, field, and evidence; then adjudicated semantic match. Preserve one-to-many and many-to-one relations.

Judge precision or recall should be communicated only when: (i) a pinned independent detection ledger exists; (ii) detection was blind to plant keys; (iii) scored proposals are dispositioned; (iv) additions follow a documented search protocol; (v) items are frozen and disjoint from prompt calibration; and (vi) matching is versioned. If a condition fails, the metric is not identifiable, not zero or one. Each reported score should include producer, selection mechanism, unit, support, matching rule, uncertainty, and an excluded inference.

This gate does not require a large new benchmark before useful work begins. It requires each limited judgment to retain the meaning needed for its intended estimand. A future study can calibrate the protocol on a small disjoint slice, then freeze it, reporting micro-by-card and macroby-target results, target-cluster bootstrap intervals, and stratification by error type, field, register, scenario family, and oral requirement.

## 8 Limitations and ethics

Plants remain valuable for testing failure modes, training targeted filters, and auditing matching logic. The observed 29.0% rejection rate is useful diagnostic feedback, but with one rating per proposal and sparse overlap it signals possible construct mismatch rather than proving a bad intervention. Likewise, 79 additions identify where generation, matching, or definitions deserve attention without establishing that every addition would survive adjudication.

The distinction is most important when expert labels are few. With 22 reviewed targets, a renamed column can dominate the empirical story. Four ledgers let one review support multiple future purposes while preserving conditional meaning: plant confirmation measures selected intervention validity; detector precision measures detector outputs; detector recall additionally depends on a searchand-match process capable of finding misses.

This is one private workflow and a forensic audit, not a prospective benchmark. Its evidence supports an existenceproof claim—that this provenance failure can produce metric-shaped values—rather than an estimate of how often the failure occurs. Reviewed targets are not asserted to be random. Card-level intervals ignore clustering and reviewer uncertainty. Only three targets overlap, additions may not exhaust errors, and historical interface states cannot be replayed. Type mixtures do not estimate natural prevalence. The proposed schema should be tested outside Japanese care handofs.

The underlying content is wholly synthetic and contains no patient recordings or health-record data. Only deidentified aggregate statistics from an existing lawful internal quality-review workflow are reported; reviewer identities and participant outcomes are not analyzed. Synthetic clinical text can still encode implausible care patterns and stereotypes. Terms such as fabricated and omitted refer to a declared research construct, not patient harm. Audio, transcripts, targets, fact checklists, metadata, and provenance may be ofered under controlled academic or commercial access, subject to data-use, security, and applicable third-party terms. Any access package should preserve the four ledgers and must not advertise judge calibration until the gate is met.

Reproducibility artifact. The public artifact at https://github.com/pyingzhu/ rcmlr-four-ledger-artifact-2026 contains the frozen count sheet, a standard-library script that reproduces all displayed ratios, Wilson intervals, source-conditioned shares, missing-review bounds, and implementation constants, an executed audit notebook, a claim crosswalk, and a machine-readable four-ledger schema with a wholly synthetic fixture. It contains no row content, reviewer identity, private locator, or detector realization. It reproduces the paper’s arithmetic but cannot replay the private extraction or repair the missing detection ledger.

## 9 Conclusion

Synthetic errors can make biomedical evaluation cheaper and sharper only if the measurement system remembers who produced each card. Current review confirms 22 of 31 imported proposals and adds 79 further cards, but the resulting 0.710 and 0.218 do not measure an LLM judge because no independent detector realization was preserved for the audited proposals. Four ledgers restore the distinction among intervention design, detector behavior, human disposition, and human discovery. The discipline yields a modest, defensible claim today and a clear path to calibrated evaluation tomorrow.

## References

[1] A. Ben Abacha, W.-W. Yim, G. Michalopoulos, and T. Lin. An investigation of evaluation methods in automatic medical note generation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2575–2588, 2023.

[2] G. H. Chen, S. Chen, Z. Liu, F. Jiang, and B. Wang. Humans or LLMs as the judge? a study on judgement bias. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8301–8327, 2024.

[3] X.-Y. Fu, M. T. R. Laskar, C. Chen, and S. B. Tn. Are large language models reliable judges? a study on the factuality evaluation capabilities of LLMs. In Proceedings of the Third Workshop on Natural Language Generation, Evaluation, and Metrics (GEM), pages 310–316, 2023.

[4] D. M. Howcroft et al. Twenty years of confusion in human evaluation: NLG needs evaluation sheets

and standardised definitions. In Proceedings of the 13th International Conference on Natural Language Generation, pages 169–182, 2020.

[5] D. Li et al. From generation to judgment: Opportunities and challenges of LLM-as-a-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2757–2791, 2025.

[6] M. Li et al. Mind the gap: Static and interactive evaluations of large audio models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8749–8766, 2025.

[7] L. Peng, Y. Gu, C. Dong, Z. Wang, and J. Shang. Text grafting: Near-distribution weak supervision for minority classes in text classification. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3741–3752, 2024.

[8] L. Peng, Z. Wang, and J. Shang. Incubating text classifiers following user instruction with nothing but LLM. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3753–3766, 2024.

[9] A. Savkov, F. Moramarco, A. Papadopoulos Korfiatis, M. Perera, A. Belz, and E. Reiter. Consultation checklists: Standardising the human evaluation of medical note generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 111–120, 2022.

[10] S. Schoch, D. Yang, and Y. Ji. “This is a Problem, Don’t You Agree?” Framing and Bias in Human Evaluation for Natural Language Generation. In Proceedings of the 1st Workshop on Evaluating NLG Evaluation, pages 10–16, 2020.

[11] C. van der Lee, A. Gatt, E. van Miltenburg, S. Wubben, and E. Krahmer. Best practices for the human evaluation of automatically generated text. In Proceedings of the 12th International Conference on Natural Language Generation, pages 355–368, 2019.

[12] W. Wang et al. Train a unified multimodal data quality classifier with synthetic data. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 1972–1986, 2025.

[13] L. Zheng et al. Judging LLM-as-a-judge with MTbench and chatbot arena. In Advances in Neural Information Processing Systems 36, pages 46595–46623, 2023.

[14] K. Zhou, J. M. Giorgi, P. Mani, P. Xu, D. Liang, and C. Tan. From feedback to checklists: Grounded

evaluation of AI-generated clinical notes. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1485–1499, 2025.