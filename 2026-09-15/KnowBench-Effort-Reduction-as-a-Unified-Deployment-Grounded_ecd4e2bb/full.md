# KnowBench: Effort Reduction as a Unified, Deployment-Grounded Benchmark for Clinical AI

Jocelyn Kang Knowtex Inc. jocelyn@knowtex.ai

Caroline Zhang Knowtex Inc. caroline@knowtex.ai

September 2026 Preprint. arXiv: cs.CL (cross-list: cs.HC). Licensed under CC BY 4.0.

## Abstract

Clinical AI systems are evaluated with instruments built for research settings (reference-based similarity metrics and expert rubric panels) that measure resemblance to an artifact rather than reduction of a burden. We introduce KnowBench, pioneered by Knowtex, whose unifying metric is Effort Reduction (ER): the proportion of system-generated clinical work product accepted by the responsible clinician under expert and safety review. ER is defined once and instantiated per task across the administrative workload clinical AI automates: visit notes, diagnosis and billing codes, orders, EHR chart summarization, patient after-visit summaries, and clinical decision support. In every instantiation the construction is identical: the clinician’s review-and-attestation event is the ground truth, every accepted unit is work the system completed, and every correction is residual effort returned to the clinician. KnowBench was co-developed with embedded clinical evaluators to reveal the leading industry deployment standards for clinical AI. The primary contribution of this paper is the benchmark itself: the metric, its degenerate cases, and a reporting protocol under which ER claims are auditable and cross-system comparable. Alongside it we report an initial headline measurement from the documentation instantiation: over one million signed encounters across a production window exceeding six months and thirteen medical specialties, Knowtex’s proprietary fine-tuned clinical foundation models operating inside a closed feedback architecture achieve an aggregate ER of 97.99%, with per-specialty aggregates spanning 96.8–98.9%. This release reports the protocol’s checklist partially, and states which companion statistics are withheld; the benchmark is offered so that this figure, and every figure reported after it, can be held to the same standard.

## 1 Introduction

Ambient clinical documentation has crossed from pilot to infrastructure: AI scribes now draft notes for millions of encounters annually across primary care and specialty medicine [14]. Evaluation has not made the same crossing. The field’s instruments (n-gram and embedding similarity against reference notes, expert rubric panels, curated benchmark sets; Ben Abacha et al. 1, Yim et al. 18) were built for model comparison in research settings, and all share a structural defect when applied to deployed systems: they measure resemblance to an artifact, not reduction of a burden. Documentation burden is the outcome these systems exist to change [11, 4]; no widely adopted metric measures it directly.

The measurement instrument has been present in the workflow all along. Before any clinical artifact is finalized, an accountable clinician reviews the system’s draft and corrects whatever is wrong, missing, or not theirs to attest. This review is an exhaustive expert evaluation performed on every encounter, at production scale, under genuine incentives, at zero marginal measurement cost. Its output is a diff, and that diff is the correct primary evaluation signal: every retained unit is work the system completed; every correction is residual effort returned to the clinician.

We formalize this signal as KnowBench, whose unifying metric is Effort Reduction (ER): the proportion of system-generated clinical work product accepted by the responsible clinician under expert review. The construction is not specific to notes. Wherever a clinical AI system drafts an artifact a clinician must review and own (a visit note, a code set, an order, a chart summary, an after-visit summary, a surfaced recommendation), the review event supplies the ground truth and the acceptance proportion is ER for that task. Knowtex is the first frontier AI lab for healthcare, deploying proprietary fine-tuned clinical foundation models across this full administrative workload and evaluating every capability against the same metric under the same protocol.

This paper defines the metric and its protocol (§2–3), maps its instantiation across the clinical administrative workload (§4), and reports headline measurements from the documentation instantiation, the instantiation with the largest measured population (§5). The claim is deliberately austere: we do not argue the system writes notes clinicians like; we utilize embedded evaluation and measure the work they no longer do.

Relation to prior work. HTER [12] formalized translation quality as required post-editing; ER is its complement, adapted to a domain where the post-editor signs the output into a legal record. Code-completion acceptance rate [19] established retention-under-expert-review as a deployment value metric; ER strengthens the construct with legal attestation stakes and whole-artifact measurement. Note-quality instruments, from PDQI-9 [13] to its LLM-era successor PDSQI-9 [3], formalize the rubric approach ER departs from: expert judgment of quality, rather than measurement of residual work. Human-evaluation studies of consultation-note generation show automatic similarity metrics correlate poorly with the post-editing judgments of clinicians [7], and hallucination and omission in LLM-generated clinical text remain the central reviewed failure modes [10]; both findings motivate a metric anchored to the reviewer’s actual corrections. Deployment studies of ambient scribes measure EHR time, burnout, and productivity [6, 8, 14], including a randomized deployment of the system family measured here, which found productivity and workflow gains in community oncology [15]; these study the downstream outcomes ER proxies, not the per-artifact effort signal itself. The copy-forward and note-bloat literature [9, 17, 16] documents that clinical notes are substantially duplicated material, a dynamic directly relevant to any system, including this one, whose longitudinal loop conditions on prior attested content (§6). The documentation-burden literature [11, 2, 4] defines the outcome ER proxies, and the automation-bias literature [5] defines its principal confound, which the protocol is constructed to expose (§2).

## 2 The Metric

Let g be the system-generated artifact presented to the clinician and f the clinician-finalized artifact. Let $R ( g , f )$ denote the retained content of g in $f$ under an alignment classifying each unit of g as retained, deleted, or rewritten, and each unit of f absent from g as an addition. Effort Reduction for a single encounter is

$$
\operatorname { E R } ( g , f ) = { \frac { | R ( g , f ) | } { | g | } } ,\tag{1}
$$

the proportion of generated content accepted under review. The unit of content is instantiation-specific: tokens for notes and summaries, codes for coding, fields for orders, recommendations for decision support. Aggregate ER is content-weighted rather than encounter-averaged, so long, complex artifacts contribute proportionally to their burden.

ER is a family. ER-text, retention as defined above, is the primary operationalization: cheap and computable from artifacts every records system already stores. ER-time, active review effort measured from interaction-level instrumentation, is the family’s validation instrument: retention and measured effort must covary for ER-text to be interpretable as effort reduction rather than a text statistic, and the divergence pattern between them (high retention with near-zero review time) is the direct detector for rubber-stamp review. ER-cognitive, review and verification load, is the family’s acknowledged horizon and is not directly instrumented.

Three boundaries help to keep the metric honest. ER is not correctness: clinicians can sign erroneous notes [5], so ER is paired with an independent factual-consistency audit on sampled signed notes, conducted on a continuing basis by Knowtex’s clinical quality and safety function. ER is not gameable by brevity: the protocol requires a companion completeness measure (system-authored share of the final artifact) and a constrained variant at a stated completeness floor, so skeletal generation cannot masquerade as performance. ER’s denominator is disciplined: it is computed over clinician-finalized artifacts, with the presentation rate over all generated drafts and the finalization rate over all presented drafts as required companions, so neither selective surfacing nor survivorship can hide in silent exclusions. In the deployment measured here, every generated draft was presented to the clinician and delivered to the EHR (§3).

## 3 The Protocol

An ER claim is complete only if accompanied by: (1) measurement window and encounter count; (2) presentation rate over generated drafts and finalization (signed-note) rate over presented drafts; (3) completeness measure and floor; (4) formatting-edit handling; (5) unit and alignment method; and (6) encounter-level distributional statistics. This six-item checklist is KnowBench’s reporting standard and the minimum for cross-system comparability.

For the documentation instantiation: retention is computed at the token level under longest-commonsubsequence alignment after formatting normalization; edits are classified as content (counted against retention), formatting (reported separately; counted in a strict variant), or additions (counted against completeness, not retention); encounters enter the denominator if and only if a draft was generated, presented, reviewed, and signed, with no exclusions for length, complexity, specialty, or audio quality.

This release reports items (1), (4), and (5), the aggregate and its per-specialty decomposition, and the presentation component of item (2): every draft generated during the measurement window was presented to the reviewing clinician and delivered to the EHR, so the measured population reflects no selective surfacing of high-confidence drafts. The remaining companion statistics, the finalization component of item (2), the completeness measure of item (3), and the encounter-level distribution of item (6), are maintained under the same instrumentation and are withheld from this preprint. We state this plainly because the standard binds its authors first: an ER figure is auditable exactly to the extent its companions are on the table.

## 4 One Metric Across the Clinical Administrative Workload

ER is instantiated per task by specifying g, f, and the unit of content. Every administrative task Knowtex automates admits the construction, because every one terminates in the same event: an accountable clinician reviewing and owning a system-generated draft.

• Visit notes: g the generated draft, f the signed note, units are tokens; validation is instrumented review effort.

• Diagnosis and billing codes: g the proposed code set, f the clinician-confirmed submitted set, units are codes.

• Orders: g the extracted order set (medications, labs, imaging, referrals, follow-up intervals), f the orders as reviewed and entered, units are order fields; tolerance tightens as extraction approaches actuation.

• EHR chart summarization: g the pre-visit chart summary, f the summary as relied upon and amended, units are tokens.

• Patient after-visit summaries: g the generated patient-facing summary, f the version released to the patient, units are tokens.

• Clinical decision support: g the surfaced recommendations, f the set acted on, units are recommendations; alert fatigue is the exact analogue of the rubber-stamp confound, detectable by the same time-divergence signature.

Each instantiation inherits the full protocol: the six-item checklist, denominator discipline, completeness constraints, and instrumented validation. Cross-task aggregate ER, the clinician’s total administrative effort reduction across the encounter, is the benchmark’s headline construct; per-task instantiations are its auditable components.

## 5 Headline Measurements: Documentation

The measured system is Knowtex’s proprietary clinical foundation model family, whose deployment effects on physician productivity and workflow have been evaluated in a randomized multisite study in community oncology [15], operating inside a closed feedback architecture: a generation harness enforcing structural constraints outside the model, per-clinician customization learned from each clinician’s signed notes and edit patterns, and a longitudinal loop in which prior-visit attested content conditions the next draft and every correction returns as training signal. The benchmark metric, the training signal, and the product outcome are the same quantity observed at different timescales.

Across more than one million signed encounters spanning a contiguous production window exceeding six months and thirteen specialties, at multiple anonymized provider organizations, the system achieves an aggregate ER of 97.99%: content-weighted, 97.99% of generated draft content was accepted by the reviewing clinician without content edit and signed into the legal medical record. Every draft generated during the window was presented to the clinician and delivered to the EHR: the drafts entering clinician review were the system’s full output, not a confidence-filtered subset. Formatting-only edits touched 2.3% of generated tokens and are excluded from the primary figure per §3; the strict variant counting them against retention is bounded below at 95.7%.

Per-specialty aggregates span 96.8–98.9% (Table 1), with nephrology, general surgery, and hematology/oncology at the upper end and psychiatry, psychology, and primary care at the lower. Two readings of the band surface. First, its width: roughly two points across thirteen cognitively and structurally dissimilar documentation regimes (procedure-oriented surgical and gastroenterology notes, longitudinal oncology assessments, narrative-heavy behavioral health, broad-spectrum primary care) is narrow by the standards of clinical NLP, where specialty transfer is the standing failure mode; specialty-scoped generation rules and per-clinician customization absorb variation a single generic configuration would surface as dispersion. Second, its shape: the lower end is occupied by the specialties with the widest input distributions per encounter (primary care) and the most narrative, least templated documentation conventions (behavioral health), which is where residual clinician correction should concentrate if the metric is measuring what it claims.

<table><tr><td>Statistic Value</td></tr><tr><td>Signed encounters (denominator) &gt;1,000,000</td></tr><tr><td>Measurement window &gt;6 contiguous months, ending H1 2026</td></tr><tr><td>Specialties 13</td></tr><tr><td>Aggregate ER (content-weighted) 97.99%</td></tr><tr><td>Strict-variant ER (formatting edits counted; lower bound) 95.7%</td></tr><tr><td>Formatting-edit rate (tokens touched) 2.3%</td></tr><tr><td>Aggregate ER by specialty</td></tr><tr><td>Nephrology 98.9%</td></tr><tr><td>General surgery 98.8%</td></tr><tr><td>Hematology/oncology 98.7%</td></tr><tr><td>Orthopedics 98.7%</td></tr><tr><td>Gastroenterology 98.6%</td></tr><tr><td>Physical medicine &amp; rehabilitation 98.3%</td></tr><tr><td>Cardiology 98.1%</td></tr><tr><td>Neurology 98.0%</td></tr><tr><td>Urology 97.9% 97.4%</td></tr><tr><td>Otolaryngology Psychiatry 96.9%</td></tr><tr><td>Psychology 96.8%</td></tr><tr><td>Primary care 96.8%</td></tr><tr><td></td></tr></table>

Table 1: Documentation-instantiation results reported in this release. Strict-variant ER is derived as aggregate ER less the formatting-edit rate and is a lower bound (§3). Remaining checklist companions are withheld from this preprint.

## 6 Limitations

ER is necessary, not sufficient: acceptance is not correctness, and a high-ER system must still demonstrate safety independently; the paired audit bounds but does not eliminate automation bias, and its results are internal to the quality and safety function. The denominator is finalized artifacts. Every generated draft in the window was presented and delivered to the EHR, which rules out selective surfacing; but delivery is not attestation, and the signed-note rate over delivered drafts, the companion that bounds how many delivered drafts were abandoned before signature, is not reported in this release. This release therefore satisfies the reporting checklist partially (§3); by the paper’s own standard, the aggregate figure is auditable only to the extent of the companions published with it. The sharpest version of the remaining gap deserves naming: without the completeness and signed-note-rate companions, a reader cannot distinguish 97.99% retention of substantive clinical content from 97.99% retention of templated content the clinician never intended to touch. The completeness measure exists in the protocol precisely to close that gap, and section-level ER decomposition (history and assessment content versus templated structure) is the discriminating analysis; neither is published here, and the headline figure should be read with exactly that reservation. Relatedly, the longitudinal loop conditions each draft on the prior visit’s attested content, so a portion of measured retention is retention of carried-forward material; the note-bloat literature [9, 16] documents that clinicians retain duplicated content they did not author and do not always read, and ER as reported cannot separate valuable continuity from that dynamic. The measured population is clinicians who adopted and continued using the system, so aggregate ER reflects both system performance and retention dynamics. The 97.99% figure is a property of Knowtex’s model family and closed-loop architecture, not of base models absent those layers; the metric and protocol are system-agnostic, which is their point. Task-specific confounds in the non-documentation instantiations (billing-driven code additions, fatigue-driven alert dismissal) require per-instantiation companion statistics before cross-task ER figures are compared.

## 7 Conclusion

Clinical AI has been evaluated by resemblance, by simulated data, and by opinion; it should be evaluated by the real-time work it removes. KnowBench measures that quantity where it occurs, in the diff between what the system drafted and what the clinician was willing to own, with one construction applied to every administrative artifact of the encounter: notes, codes, orders, chart summaries, after-visit summaries, and surfaced recommendations. Its documentation instantiation, measured over more than one million signed encounters across thirteen specialties, stands at 97.99% aggregate ER within a two-point specialty band. The number is a property of a closed-loop architecture in which the evaluation signal and the training signal are the same measurement; the benchmark and protocol are the property of no one.

Acknowledgments. We thank the clinicians, healthcare organizations and partners whose practice constitutes this measurement, and the Knowtex clinical quality and safety function for the continuous standing audit program.

## References

[1] A. Ben Abacha, W. Yim, Y. Fan, and T. Lin. An empirical study of clinical note generation from doctor-patient encounters. In Proceedings of EACL, pages 2291–2302, 2023.

[2] B. G. Arndt, J. W. Beasley, M. D. Watkinson, et al. Tethered to the EHR: Primary care physician workload assessment using EHR event log data and time-motion observations. Annals of Family Medicine, 15(5):419–426, 2017.

[3] E. Croxford, Y. Gao, N. Pellegrino, et al. Development and validation of the Provider Documentation Summarization Quality Instrument for large language models. JAMIA, 32(6):1050–1060, 2025.

[4] R. L. Gardner, E. Cooper, J. Haskell, et al. Physician stress and burnout: the impact of health information technology. JAMIA, 26(2):106–114, 2019.

[5] K. Goddard, A. Roudsari, and J. C. Wyatt. Automation bias: a systematic review of frequency, effect mediators, and mitigators. JAMIA, 19(1):121–127, 2012.

[6] T. Haberle, C. Cleveland, G. L. Snow, et al. The impact of Nuance DAX ambient listening AI documentation: a cohort study. JAMIA, 31(4):975–979, 2024.

[7] F. Moramarco, A. Papadopoulos Korfiatis, M. Perera, et al. Human evaluation and correlation with automatic metrics in consultation note generation. In Proceedings of ACL, pages 5739–5754, 2022.

[8] K. D. Olson, D. Meeker, M. Troup, et al. Use of ambient AI scribes to reduce administrative burden and professional burnout. JAMA Network Open, 8:e2534976, 2025.

[9] A. Rule, S. Bedrick, M. F. Chiang, and M. R. Hribar. Length and redundancy of outpatient progress notes across a decade at an academic medical center. JAMA Network Open, 4(7):e2115334, 2021.

[10] D. Roustan and F. Bastardot. The clinicians’ guide to large language models: a general perspective with a focus on hallucinations. Interactive Journal ofMedical Research, 14:e59823, 2025.

[11] C. Sinsky, L. Colligan, L. Li, et al. Allocation of physician time in ambulatory practice: a time and motion study in 4 specialties. Annals ofInternal Medicine, 165(11):753–760, 2016.

[12] M. Snover, B. Dorr, R. Schwartz, L. Micciulla, and J. Makhoul. A study of translation edit rate with targeted human annotation. In Proceedings ofAMTA, 2006.

[13] P. D. Stetson, S. Bakken, J. O. Wrenn, and E. L. Siegler. Assessing electronic note quality using the Physician Documentation Quality Instrument (PDQI-9). Applied Clinical Informatics, 3(2):164–174, 2012.

[14] A. A. Tierney, G. Gayre, B. Hoberman, et al. Ambient artificial intelligence scribes to alleviate the burden of clinical documentation. NEJM Catalyst, 5(3), 2024.

[15] N. Toussi, C. Zhang, J. Kang, and E. J. Licitra. Impact of AI scribes on physician productivity and workflow in medical oncology. JCO Oncology Practice, published online February 11, 2026. doi:10.1200/OP-25-00612.

[16] A. Y. Tsou, C. U. Lehmann, J. Michel, R. Solomon, L. Possanza, and T. Gandhi. Safe practices for copy and paste in the EHR: systematic review, recommendations, and novel model for health IT collaboration. Applied Clinical Informatics, 8(1):12–34, 2017.

[17] J. O. Wrenn, D. M. Stein, S. Bakken, and P. D. Stetson. Quantifying clinical narrative redundancy in an electronic health record. JAMIA, 17(1):49–53, 2010.

[18] W. Yim, Y. Fu, A. Ben Abacha, N. Snider, T. Lin, and M. Yetisgen. ACI-BENCH: a novel ambient clinical intelligence dataset for benchmarking automatic visit note generation. Scientific Data, 10:586, 2023.

[19] A. Ziegler, E. Kalliamvakou, X. A. Li, et al. Productivity assessment of neural code completion. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming (MAPS), pages 21–29, 2022.