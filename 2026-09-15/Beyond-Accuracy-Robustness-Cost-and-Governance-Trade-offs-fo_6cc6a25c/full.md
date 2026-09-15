# Beyond Accuracy: Robustness, Cost, and Governance Trade-offs for Vision-Language Models in Templated Document Extraction

Kushal Patel, Pushkal Shrivastava, Mackenzie Lees, Qirui Lu, Bhargobjyoti Saikia, Liying Li, Junlin Jiang John Hancock, 200 Berkeley St, Boston, MA 02116, USA

{kpatel, pxshrivastava, mlees, qirui\_lu, bsaikia, liyingli, junlin\_jiang}@jhancock.com

## Abstract

Vision-language models (VLMs) are increasingly used to extract structured fields from business documents, yet most evaluations report accuracy on clean benchmarks and offer little guidance to practitioners choosing an approach for a given task complexity. We address this gap with a measurement-grounded study and an open-source release. Across eleven systems (three commercial, two reasoning, five opensource VLMs in pretrained and fine-tuned form, and a non-LLM OCR→regex floor) scored on a 750-document held-out pool of synthetic checks, fine-tuning on 3K samples lifts the best open-source VLMs above F1 0.98—above every zero-shot commercial system on this task— while GPT-5 leads the commercial pool on F1 and Claude Sonnet 4.5 collapses on Date. To turn these measurements into actionable choices, we introduce a practitioner-oriented selection framework that maps a task profile (quality, latency, governance, volume) to a recommended approach via filtering and total-cost minimization, illustrated on a hypothetical midvolume document-extraction scenario.<sup>1</sup>

## 1 Introduction

Intelligent document processing (IDP) underpins high-volume operations in finance, healthcare, and insurance. Recent vision-language models (VLMs) promise end-to-end field extraction without explicit OCR (Bai et al., 2025; Marafioti et al., 2025), and large multimodal LLMs (OpenAI, 2023) approach human-level accuracy on clean documents. For a practitioner facing a new extraction task, however, public leaderboards offer little actionable guidance: they report accuracy on curated, well-aligned scans and rarely describe the joint operating conditions— scan-defect robustness, latency budget, deployment governance, and processing volume—that determine which approach is feasible at production.

We focus on structured field extraction from templated business forms, a prevalent and high-volume class of enterprise document processing tasks. Our primary study is on bank checks; we additionally report a cross-industry generalization test on insurance application forms.

Our central question is operational: given a target task with a specified complexity (quality threshold, latency budget, governance requirement, expected volume), which class of approach should a practitioner adopt? We answer it through releasable artifacts and a selection framework grounded in measurement. Because frontier model lineups change on a quarterly cadence, our central contribution is the evaluation-and-selectionframework rather than any fixed vendor ranking: the model pool reported here is a snapshot, and new entrants can be evaluated by re-populating the candidate set M (Section 5) with updated measurements from the released harness. Concretely, we make four contributions.

Contributions. (i) A check-extraction dataset of 3,748 synthetic images with four-field ground truth (Payee, Amount, Date, Bank). (ii) An open-source code release covering data generation, LoRA and full fine-tuning, prompt and API harnesses, and per-sample diagnostics.<sup>2</sup> (iii) An empirical evaluation of eleven systems— three commercial (GPT-4.1 vision, OCR+GPT-4.1, Azure Content Understanding), two reasoning (GPT-5, Claude Sonnet 4.5), five opensource VLMs (Qwen2.5-VL-3B/7B, SmolVLM2- 2.2B/500M/256M, each in pretrained and finetuned form), and a non-LLM OCR→regex floor— with per-field and per-error-mode breakdowns over >45K field-level predictions. (iv) A practitioneroriented selection framework mapping a task profile to a recommended approach via governance/quality/latency filters and total-cost minimization, instantiated on an illustrative mid-volume document-extraction scenario.

## 2 Related Work

VLMs for document understanding. End-toend models such as Donut (Kim et al., 2022) and document-tuned multimodal LLMs eliminate explicit OCR. Recent open-source VLMs (Bai et al., 2025; Marafioti et al., 2025) target on-device inference but have not been systematically evaluated on enterprise document distortions.

Robustness in document AI. Most document benchmarks (FUNSD (Jaume et al., 2019), CORD (Park et al., 2019), RVL-CDIP (Harley et al., 2015)) provide clean scans, and prior OCRrobustness work has tended to address individual preprocessing axes in isolation (e.g., binarization under non-uniform illumination, Michalak and Okarma, 2020) in the pre-VLM era (Cui et al., 2021). The robustness of modern VLMs to compound enterprise distortions—perspective, rotation, and noise—on structured-extraction tasks remains underexplored.

Deployment trade-offs. Cost–accuracy Pareto analyses for LLM APIs (Chen et al., 2024) have addressed text-only inference. Recent work in enterprise document processing has begun to expose the same trade-off space: hybrid OCR+LLM routing across 25 configurations for copy-heavy identity documents (Wang and Shen, 2025), and multiagent orchestration benchmarks on 10 K SEC filings that report cost-per-document and Pareto positions alongside accuracy (Kulkarni and Kulkarni, 2026). Our work extends this analysis to multimodal field extraction and treats governance as a configurable task parameter: a given workload may or may not impose constraints (legal, policy, contractual, or operational) that narrow the set of admissible deployment options, and the framework simply encodes whichever constraints the practitioner specifies.

## 3 Experimental Setup

Task. Models are prompted to return a JSON object with four fields per check: Payee (free-text), Amount (numeric), Date (MM/DD/YYYY), and Bank (categorical: Chase, Citi, BOA). All variants share an identical prompt and schema.

Dataset. We render 3,748 synthetic check images from a structured table of field values using a template-driven pipeline with controlled appearance perturbations. Of these, 2,998 are used for fine-tuning and 750 form the held-out test pool (identical for every model in this study); no document appears in both splits.

Train/test disjointness. Because the dataset is template-driven, leakage is a legitimate concern. We split at the row level of the underlying field table before rendering: the Payee, Amount, and Date values in the 750-document test pool are drawn from rows disjoint from the 2,998-document training table (random partition, seed 42, fixed for all experiments). Bank, being a 3-way categorical, is shared by construction; its near-perfect accuracy across all systems (Section 4.2) should therefore be read as classification on a closed vocabulary rather than evidence of generalisation. Template backgrounds are also held disjoint: the rendering pipeline draws from separate background pools for train and test.

Models. We evaluate three commercial systems— GPT-4.1 (vision-only), OCR+GPT-4.1 (Azure Document Intelligence + GPT-4.1), and Azure Content Understanding (ACU)—two reasoning models— GPT-5 and Claude Sonnet 4.5—five open-source VLMs: Qwen2.5-VL-3B and 7B (LoRA fine-tuned, r=16, α=32, 1.08% / 1.23% trainable parameters), and SmolVLM2-256M, SmolVLM-500M, SmolVLM2-2.2B (full fine-tuning)—and a non-LLM OCR→regex baseline (Azure Document Intelligence Read + hand-written field-specific patterns) that bounds the floor of what a zero-LLMcost pipeline achieves on this task. Open-source models are evaluated both pretrained and after finetuning. Fine-tunes use 2,998 examples, effective batch size four, paged AdamW-8bit; Qwen-3B and the three SmolVLM variants run 3 epochs, with learning rates $2 \times 1 0 ^ { - 4 }$ (Qwen-3B LoRA) and $1 \times 1 0 ^ { - 5 }$ (SmolVLM full fine-tuning). Qwen-7B uses 5 epochs at $1 \times 1 0 ^ { - 4 }$ LoRA learning rate, a longer/gentler schedule that keeps the 7B adapter from truncating long outputs.

Scope of baselines. We do not evaluate purposebuilt structured-extraction encoders (Donut (Kim et al., 2022), LayoutLMv3 (Huang et al., 2022), Pix2Struct (Lee et al., 2023), UDOP (Tang et al., 2023)) in this release: our framework requires candidates that produce structured JSON directly, whereas these models require task-specific decoding heads. Recent 2025 work on document key information extraction (Wang et al., 2025; Yu et al., 2025) benchmarks task-adapted VLMs against other VLMs rather than LayoutLMv3/Donut-era models, reflecting a community shift in what constitutes a competitive extraction system; our candidate set is aligned with this current practice. Incorporating purpose-built encoder-decoder models is future work.

Metrics. We report entity-level micro-F1 with TP/FP/FN aggregated over all four fields, per-field F1, document-level accuracy (all four fields correct), JSON parse success rate, and end-to-end latency. We additionally analyze per-sample debug logs to categorize errors as under-extraction (empty prediction), wrong value, or parsefailure.

## 4 Results

## 4.1 Overall Performance

Table 1 reports headline metrics. All models— commercial, reasoning, and open-source—are scored on the same 750-document held-out PNG test pool.

Within the commercial pool, GPT-5 leads on F1 narrowly above OCR+GPT-4.1 and GPT-4.1 vision but at 2× the latency of the latter, while GPT-4.1 vision has the highest document-level accuracy; ACU is the most balanced across fields (Section 4.2), and Claude Sonnet 4.5 trails, dragged down almost entirely by Date.

Under a fixed adaptation budget of 2,998 training documents (Section 3), the best fine-tuned open-source models exceed every zero-shot commercial system on this schema—Qwen2.5-VL-7B FT reaches 0.985 and Qwen2.5-VL-3B FT 0.983 (within 0.002 of each other; Appendix G confirms this gap is within seed noise), both with zero JSON parse failures, suggesting the 3B adapter already saturates the available signal on this schema. Even SmolVLM2-256M becomes usable with finetuning, moving from 0.000 to 0.726 F1 and from 100% to 30.7% JSON parse failure—operationally fragile but viable for clean-document pilots. The

<table><tr><td>Model</td><td>F1</td><td>DocAcc</td><td>Lat. (s)</td><td>JSON Fail</td></tr><tr><td colspan="5">Commercial &amp; reasoning</td></tr><tr><td>GPT-5 OCR+GPT-4.1</td><td>0.928</td><td>41.7</td><td>26.3</td><td>0.0%</td></tr><tr><td>GPT-4.1 (vision)</td><td>0.917 0.909</td><td>40.7 45.4</td><td>18.1 12.2</td><td>0.0% 0.0%</td></tr><tr><td>ACU</td><td>0.889</td><td>33.1</td><td>12.8</td><td>0.0%</td></tr><tr><td>Claude Sonnet 4.5</td><td>0.857</td><td>14.8</td><td>8.5</td><td>0.0%</td></tr><tr><td>Non-LLM floor OCR+Regex</td><td>0.395</td><td>0.0</td><td>8.6</td><td></td></tr><tr><td colspan="5">Open-source, fine-tuned</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.985</td><td>88.9</td><td>3.53</td><td>0.0%</td></tr><tr><td>Qwen2.5-VL-3B</td><td>0.983</td><td>88.4</td><td>4.18</td><td>0.0%</td></tr><tr><td>SmolVLM2-2.2B</td><td>0.906</td><td>46.7</td><td>1.47</td><td>0.0%</td></tr><tr><td>SmolVLM-500M</td><td>0.854</td><td>45.1</td><td>0.96</td><td>5.3%</td></tr><tr><td>SmolVLM2-256M</td><td>0.726</td><td>22.4</td><td>1.62</td><td>30.7%</td></tr><tr><td colspan="5">Open-source, pretrained (no FT)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.921</td><td>50.7</td><td>2.53</td><td>0.0%</td></tr><tr><td>Qwen2.5-VL-3B</td><td>0.847</td><td>24.5</td><td>4.75</td><td>0.0%</td></tr><tr><td>SmolVLM2-2.2B</td><td>0.504</td><td>0.4</td><td>2.03</td><td>3.3%</td></tr><tr><td>SmolVLM-500M</td><td>0.000</td><td>0.0</td><td>1.80</td><td>100%</td></tr><tr><td>SmolVLM2-256M</td><td>0.000</td><td>0.0</td><td>1.30</td><td>100%</td></tr></table>

Table 1: Headline metrics on the 750-document held-out PNG test pool (identical pool for every model). Finetuned rows use a fixed adaptation budget of 2,998 training documents (LoRA for Qwen, full fine-tuning for SmolVLM; recipes in Section 3). Bold marks the best in column among open-source FT; italic marks the best F1 within the commercial / reasoning block. Open-source latencies are measured on a single NVIDIA A100 80 GB PCIe (bf16, batch size 1, five-document warmup discarded); see Section 4.5 for the A100→H100 sensitivity analysis. The OCR+Regex row is a non-LLM floor baseline (Azure Document Intelligence Read + hand-written field-specific patterns); its outputs are field-keyed dictionaries, not JSON, so the JSON-failure column is not applicable.

non-LLM OCR→regex floor trails every commercial and fine-tuned system (Tables 1–2).

## 4.2 Per-Field Analysis

Table 2 confirms that Date is the hardest field for every commercial system—F1 ranges from 0.468 (Claude) to 0.795 (ACU). Claude Sonnet 4.5’s failure mode is extreme over-extraction: 686 Date false positives against 304 true positives, giving precision 0.31 and recall 0.99. GPT-4.1 (vision) and OCR+GPT-4.1 show the same pattern at lower magnitude (precision 0.65 and 0.63 respectively, recall ≈ 1). Only ACU achieves balanced Date precision/recall (0.733/0.870).

Is Date a prompting artifact? We tested GPT-4.1 with five targeted interventions (higher image detail, a check-parsing persona, orientation cues, an MM/DD/YYYY format instruction, and fewshot rotated / OCR-misread examples). Date accuracy rose by about two percentage points (≈79% to ≈81%), suggesting the failure is not primarily a prompting artifact. JSON-mode was not tested separately: parse validity is not the failure mode.

<table><tr><td>Model</td><td>Payee</td><td>Amount</td><td>Date</td><td>Bank</td></tr><tr><td colspan="5">Commercial &amp; reasoning</td></tr><tr><td>OCR+GPT-4.1</td><td>0.874</td><td>0.994</td><td>0.769</td><td>1.000</td></tr><tr><td>GPT-4.1 (vision)</td><td>0.856</td><td>0.963</td><td>0.791</td><td>1.000</td></tr><tr><td>ACU</td><td>0.780</td><td>0.960</td><td>0.795</td><td>0.988</td></tr><tr><td>GPT-5</td><td>0.870</td><td>0.970</td><td>0.772</td><td>0.999</td></tr><tr><td>Claude S. 4.5</td><td>0.810</td><td>0.874</td><td>0.468</td><td>0.993</td></tr><tr><td colspan="5">Non-LLM floor</td></tr><tr><td>OCR+Regex</td><td>0.305</td><td>0.721</td><td>0.162</td><td>0.264</td></tr><tr><td colspan="5">Open-source, fine-tuned</td></tr><tr><td>Qwen-3B FT</td><td>0.924</td><td>0.984</td><td>0.957</td><td>1.000</td></tr><tr><td>Qwen-7B FT</td><td>0.935</td><td>0.992</td><td>0.960</td><td>1.000</td></tr><tr><td>SmolVLM2-2.2B FT</td><td>0.807</td><td>0.913</td><td>0.886</td><td>0.995</td></tr><tr><td>SmolVLM-500M FT</td><td>0.728</td><td>0.876</td><td>0.855</td><td>0.938</td></tr><tr><td>SmolVLM2-256M FT</td><td>0.587</td><td>0.730</td><td>0.748</td><td>0.818</td></tr></table>

Table 2: Per-field entity-level F1 on the 750-document held-out PNG test pool. Bold = best in column within its block; italic = lowest Date F1 among commercial.

By contrast, fine-tuned Qwen models match or surpass the commercial systems on Date in our pool because they learn to copy the single Date region rather than enumerate every date-like substring on the page.

Conversely, Bank is the easiest field for every generative system at 2B parameters or above (F1 ≥ 0.988)—a small categorical vocabulary (Chase, Citi, BOA) is effectively memorised; only the sub-1B SmolVLMs and the regex floor lag.

GPT-5 is the only model to report two additional fields, AmountInWords (F1 0.957) and CheckNumber (F1 0.967); these are useful when a richer schema is required and influence the cost analysis in Section 6.

## 4.3 Error Taxonomy

A per-sample analysis yields three recurring error categories: (E1) degenerate repetition in sub-500M VLMs (looping arrays that exhaust the token budget, explaining the 30.7% JSON-failure rate of fine-tuned SmolVLM2-256M); (E2) phonetic Payee misspelling in mid-size open-source models on visually-ambiguous character pairs (g/y, s/z, e/i); and (E3) Date over-extraction in commercial LLMs (every date-like substring is emitted; Claude’s worst case is 686 FP against 304 TP). The pattern argues for class-specific post-processing: schema validation and stop-token tuning for sub-500M VLMs, and Date-context disambiguation for commercial LLMs. Appendix E gives quantitative breakdowns and examples.

<table><tr><td>Model</td><td>Pretrained</td><td>Fine-tuned</td><td>∆</td></tr><tr><td>Qwen2.5-VL-3B</td><td>0.847</td><td>0.983</td><td>+16%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.921</td><td>0.985</td><td>+7%</td></tr><tr><td>SmolVLM2-2.2B</td><td>0.504</td><td>0.906</td><td>+79%</td></tr><tr><td>SmolVLM-500M</td><td>0.000</td><td>0.854</td><td></td></tr><tr><td>SmolVLM2-256M</td><td>0.000</td><td>0.726</td><td>一</td></tr></table>

Table 3: Pretrained vs. fine-tuned Micro-F1 on the heldout PNG test pool.

## 4.4 Fine-tuning Impact

Fine-tuning helps every open-source model (Table 3). The largest absolute gains are for the smallest models—SmolVLM-500M and -256M both move from total failure to usable output. Qwen-3B’s relative gain is modest and Qwen-7B’s is even smaller, because both pretrained Qwen models are already strong on this schema. The 3B and 7B finetunes end with only a marginal difference, so the additional 4.6B parameters in the 7B adapter buy very little accuracy on this task.

## 4.5 GPU Sensitivity

Table 1 latency is measured on an NVIDIA A100 80 GB PCIe. Re-running the five fine-tuned models on an NVIDIA H100 NVL under identical software yields a consistent 1.4–1.7× per-document speedup with accuracy unchanged within noise (∆F1 ≤ 0.003); the relative ordering is preserved (Appendix D). The 7B-faster-than-3B reversal at batch 1 is reproducible on both GPUs and attributable to Qwen-7B’s shallower-but-wider decoder (28×3584 vs. 36×2048 layers/hidden); the gap should close at larger batch sizes.

## 4.6 Cross-task and OOD Evaluation

Out-of-distribution checks. We evaluate the four fine-tuned open-source models zeroshot on a public Indian-style check dataset (shivalikasingh/cheques\_sample\_data,

n=400; different rendering pipeline, bank vocabulary, and DD/MM/YY dates). SmolVLM2-2.2B FT, the check-task recommendation, retains 87% of its in-domain F1 (0.788 OOD vs. 0.906 in-domain, Table 4). Qwen-3B FT drops 39 points; the failure is a diagnosable training-data artifact—year 2024 appears in every training record and all 400 OOD predictions are “06/05/2024”. Both sub-1B SmolVLM variants collapse structurally under domain shift.

<table><tr><td>Model</td><td>Checks (F1)</td><td>OOD (F1)</td><td>Insurance (Acc.)</td></tr><tr><td>Qwen2.5-VL-3B FT</td><td>0.983</td><td>0.593</td><td>0.927</td></tr><tr><td>Qwen2.5-VL-7B FT</td><td>0.985</td><td></td><td>0.927</td></tr><tr><td>SmolVLM2-2.2B FT</td><td>0.906</td><td>0.788</td><td>0.292</td></tr><tr><td>SmolVLM-500M FT</td><td>0.854</td><td>0.001</td><td>0.895</td></tr><tr><td>SmolVLM2-256M FT</td><td>0.726</td><td>0.000</td><td>0.145</td></tr><tr><td>GPT-4.1 (vision)</td><td>0.909</td><td></td><td>0.768</td></tr><tr><td>OCR+GPT-4.1</td><td>0.917</td><td>一</td><td>0.757</td></tr></table>

Table 4: Cross-task and OOD evaluation. Checks: 750- doc held-out PNG pool (Table 1). OOD: 400-doc public dataset shivalikasingh/cheques\_sample\_data, evaluated zero-shot after check-task fine-tuning. Insurance: 36-field insurance application forms, 4,000 train / 1,000 test. GPT-4.1 insurance numbers use true↔yes normalization on Y/N outputs (raw: 0.738 and 0.731; see Appendix F). “–” indicates the model was not evaluated on that task. Bold marks the best in-column among fine-tuned open-source.

Insurance-form task. We separately fine-tune the same open-source families on a 36-field insurance application form task (4,000 train / 1,000 test) under the same adaptation budget (Section 3), and compare against zero-shot GPT-4.1. Qwen models tie at field accuracy 0.927, outperforming GPT-4.1 image-only by 15.9 points. A per-field-type breakdown on the four models with exported per-field results (Appendix F) shows this gap concentrates in 9 checkbox / boolean-flag fields, where finetuned open-source models substantially outperform zero-shot GPT-4.1: SmolVLM-500M FT reaches 0.99 checkbox accuracy versus 0.48 for GPT-4.1 image-only, a 50-point difference that alone accounts for essentially the entire aggregate gap. On the 22 structured-text fields the two systems tie within 1 point (≈0.93). SmolVLM2-2.2B FT collapses on this schema (32% coverage, field accuracy 0.292); SmolVLM-500M FT emerges as a small-fast runner-up at 0.895 accuracy. Framework consequence: the coverage-thresholded feasibility set produces a different $m ^ { * }$ per task, as shown in Section 6.

## 5 A Framework for Selecting an Approach

The results in Section 4 show that no single model dominates across the operating axes that matter in practice. We therefore frame approach selection as a constrained optimization problem parameterised by task complexity: a quality target $Q _ { \mathrm { m i n } }$ , latency budget $L _ { \mathrm { m a x } }$ (interpreted as a p95 ceiling), governance requirement $G _ { \mathrm { r e q } } ,$ expected monthly volume

V, and horizon T months.

Let M be the candidate set; each $m \in \mathcal { M }$ is characterised by a set of task-relevant quality metrics $Q ( m ) { = } \{ Q _ { i } ( m ) \} _ { i \in I }$ with componentwise thresholds $Q _ { \mathrm { m i n } } { = } \{ Q _ { i , \mathrm { m i n } } \} _ { i \in I } , { \mathrm { p } } 9 5$ latency $L _ { p 9 5 } ( m )$ , the set of governance constraints it satisfies $G ( m )$ development cost $C _ { \mathrm { d e v } } ( m )$ , per-document inference cost $c _ { \mathrm { i n f } } ( m )$ , and monthly fixed cost $c _ { \mathrm { f i x e d } } ( m )$ Beyond aggregate accuracy, I typically includes structured-output coverage (the fraction of inputs on which the model produces well-formed output; equivalently, one minus the JSON parse failure rate; a special case of the selective-prediction setting (Geifman and El-Yaniv, 2017)), which practitioners can threshold independently of accuracy when parse reliability is operationally critical. The feasible set is

$$
\begin{array} { c } { { \mathcal { M } _ { F } = \{ m : Q _ { i } ( m ) \geq Q _ { i , \mathrm { m i n } } \forall i \in I , } } \\ { { L _ { p 9 5 } ( m ) \leq L _ { \mathrm { m a x } } , G ( m ) \supseteq G _ { \mathrm { r e q } } \} } } \end{array}\tag{1}
$$

and the cost-optimal model minimizes total cost of ownership:

$$
\begin{array} { l } { m ^ { * } = \arg \underset { m \in \mathcal { M } _ { F } } { \operatorname* { m i n } } \left[ C _ { \mathrm { d e v } } ( m ) \right. } \\ { \left. + T ( V c _ { \mathrm { i n f } } ( m ) + c _ { \mathrm { f i x e d } } ( m ) ) \right] . } \end{array}\tag{2}
$$

Eq. (2) is a deliberately simple TCO model for a typical enterprise pilot deployment, with four assumptions: (i) per-document inference cost is linear in volume V (no batching discounts, no tokenlength variance); (ii) fixed cost is linear in horizon T (constant monthly rate, $c _ { \mathrm { f i x e d } } { = } 0$ for managed APIs and equal to the dedicated GPU rental for selfhosted candidates); (iii) $C _ { \mathrm { d e v } }$ is paid once and not amortised across other applications; (iv) self-hosted models run on a dedicated inference instance provisioned continuously for this workload—GPU sharing across workloads is not assumed.

Governance acts as a feasibility filter: $G _ { \mathrm { r e q } }$ is the set of deployment constraints specified for a given workload (these may be technical, contractual, or policy-driven), and a model is feasible only if $G ( m ) { \supseteq } G _ { \mathrm { r e q } }$ . The filter is generic, so practitioners can encode whichever constraints apply to their setting. Among feasible solutions, the breakeven volume between two models m<sub>1</sub>, m<sub>2</sub> is

$$
V ^ { * } = \frac { \Delta C _ { \mathrm { d e v } } + T \Delta c _ { \mathrm { f i x e d } } } { T ( c _ { \mathrm { i n f } } ^ { ( m _ { 2 } ) } - c _ { \mathrm { i n f } } ^ { ( m _ { 1 } ) } ) } ,\tag{3}
$$

where $\Delta C \mathrm { _ d e v } { = } C \mathrm { _ d e v } ( m _ { 1 } ) { - } C \mathrm { _ d e v } ( m _ { 2 } )$ and $\Delta c _ { \mathrm { f i x e d } }$ is defined analogously. If $m _ { 1 }$ carries the higher fixed cost and $m _ { 2 }$ the higher per-document cost (numerator and denominator both positive), then above $V ^ { * }$ the lower-marginal-cost solution $m _ { 1 }$ wins, and below $V ^ { * }$ the lower-fixed-cost solution $m _ { 2 }$ wins.

The framework is intentionally lightweight: practitioners populate $\mathcal { M }$ with measurements obtained from the dataset and code we release (Section 3), specify the four task parameters, and read off the recommendation. Section 6 instantiates this on a concrete scenario.

## 6 Illustrative Case Study: A Hypothetical Mid-Volume Extraction Profile

We instantiate the framework on a hypothetical task profile chosen to exercise all four parameters; it is not based on a production implementation, and no customer documents or internal company metrics were used to construct it. The profile is $V { = } 1 0 0 \mathrm { K }$ documents/month, T=12 months, $Q _ { \mathrm { m i n } } { = } 0 . 8 5 \ : \mathrm { F } 1$ $L _ { \mathrm { m a x } } { = } 5 \ \mathrm { s }$ , and, to illustrate the governance filter, an arbitrary deployment-locality constraint that admits only self-hosted models—a generic example chosen to show how a constraint propagates through the pipeline, not a claim about which constraints any specific workload, organisation, or industry should adopt. The numbers below are illustrative estimates for demonstrating framework application, not a vendor benchmark; practitioners can re-populate Table 5 with the rates and constraints that apply to their own setting.

Cost assumptions. Per-document inference cost ${ \mathit { c } } _ { \mathrm { i n f } }$ for API candidates in Table 5 is derived from published vendor pricing: GPT-4.1 at \$2/M input and \$8/M output (OpenAI, 2025b) (≈1,500 image + 80 output tokens; \$0.004/doc); GPT-5 (OpenAI, 2025a), emitting two extra fields (AmountIn-Words, CheckNumber), \$0.025/doc; Claude Sonnet 4.5 at \$3/\$15 per M (Anthropic, 2025), \$0.006/doc; OCR+GPT-4.1 combines Azure DI Read at \$1.50/1,000 pages (Microsoft Azure, 2025b) with a text GPT-4.1 call (\$0.003/doc); the OCR→regex baseline incurs only the OCR call (\$0.0015/doc); Azure Content Understanding is \$0.010/page (Microsoft Azure, 2025a). Self-hosted candidates are priced from a dedicated, continuously provisioned AWS g5.xlarge (NVIDIA A10G) at \$1.006/hour (Amazon Web Services, 2025) (Eq. (2) assumption iv), which we treat as a fixed monthly cost $c _ { \mathrm { f i x e d } } { = } \$ 73$ 4/month with $c _ { \mathrm { i n f } } { \approx } 0 { : }$ the marginal cost of one additional document on already-provisioned capacity is negligible below saturation, and the case-study volume V=100K docs/month sits well below the singleinstance saturation threshold at target utilisation $\rho _ { \mathrm { m a x } } { = } 0 . 7$ . Self-hosted p95 latencies in Table 5 are projected from A100 measurements to A10G at a 1.7× ratio, drawn from the widely reported 1.5– $1 . 7 \times$ inference slowdown for 3B-7B transformer models at batch size 1 (Baseten, 2025; Modal Labs, 2025; NVIDIA, 2022, 2021); direct A10G measurement is a limitation (see Limitations). Development cost $C _ { \mathrm { d e v } }$ for fine-tuned models is \$6K labelling (the 2,998-document training set in Section 3 at \$2/doc, an authors’ midpoint estimate for layout-aware document annotation between commercial crowd and expert-review rates; see Limitations) + \$7.5K base engineering (50 hr at \$150/hr fully-loaded, an authors’ estimate informed by U.S. Bureau of Labor Statistics (2024) softwaredeveloper wages) + \$1.5–3.5K productionisation overhead; API integrations omit the fine-tune line items. These are pilot-grade illustrative baselines (see Limitations).

Stage 1 (Governance). The illustrative locality constraint admits only self-hosted candidates, removing GPT-5, GPT-4.1, OCR+GPT-4.1, ACU, Claude Sonnet 4.5, and the OCR+Regex pipeline (whose OCR step is also an external API call) from contention.

Stage 2 (Quality + latency). Quality here has two components: F1 with $Q _ { F 1 , \mathrm { { m i n } } } { = } 0 . 8 5$ and coverage with $Q _ { \mathrm { c o v , m i n } } { = } 0 . 9 9$ . Among self-hosted candidates, all four fine-tuned open-source variants meet the F1 threshold; the OCR+Regex baseline at F1 0.395 is eliminated on accuracy regardless of deployment location. Under the A10G latency projection, both Qwen models exceed the 5 s budget (Qwen-7B FT $8 . 2 \mathrm { s } \ \mathrm { p } 9 5$ , Qwen-3B FT 9.9 s p95) and are excluded; SmolVLM2-2.2B FT (3.4 s) and SmolVLM-500M FT (4.1 s) pass. Coverage then excludes SmolVLM-500M FT (0.947 < 0.99, i.e., 5.3% of outputs fail JSON parse and are counted as full misses under our reporting protocol), leaving SmolVLM2-2.2B FT as the sole survivor. On H100-class hardware both Qwen models pass latency, so accelerator choice is itself part of the practitioner’s task profile.

Stage 3 (Cost and result). With only SmolVLM2-2.2B FT surviving governance, latency, and coverage, the framework-derived recommendation is $m ^ { * } { \mathrm { : } }$ =SmolVLM2-2.2B FT at annual cost \$23.8K (V T=1.2M docs/yr; \$15K $C _ { \mathrm { d e v } }$ plus twelve months at \$734/month for one dedicated A10G instance). Were the illustrative locality constraint absent, OCR+GPT-4.1 at \$11.6K/yr would win; GPT-5’s 0.011 F1 lift rarely justifies its 3× price unless its extra fields are required.

<table><tr><td>Model</td><td>Q</td><td>Cov.</td><td> $L _ { p 9 5 }$  (s)</td><td> $C _ { \mathrm { d e v } }$  ($K)</td><td> ${ \mathit { c } } _ { \mathrm { i n f } }$  ($/doc)</td><td>Loc.</td></tr><tr><td>GPT-5</td><td>0.93</td><td>1.00</td><td>40</td><td>5</td><td>0.0250</td><td>API</td></tr><tr><td>OCR+GPT-4.1</td><td>0.92</td><td>1.00</td><td>25</td><td>8</td><td>0.0030</td><td>API</td></tr><tr><td>GPT-4.1 (vision)</td><td>0.91</td><td>1.00</td><td>18</td><td>5</td><td>0.0040</td><td>API</td></tr><tr><td>ACU</td><td>0.89</td><td>1.00</td><td>18</td><td>5</td><td>0.0100</td><td>API</td></tr><tr><td>Claude S. 4.5</td><td>0.86</td><td>1.00</td><td>12</td><td>5</td><td>0.0060</td><td>API</td></tr><tr><td>OCR+Regex</td><td>0.40</td><td>1.00</td><td>12</td><td>6</td><td>0.0015</td><td>API</td></tr><tr><td>Qwen-7B FT</td><td>0.99</td><td>1.00</td><td>8.2</td><td>17</td><td>≈0</td><td>on-prem</td></tr><tr><td>Qwen-3B FT</td><td>0.98</td><td>1.00</td><td>9.9</td><td>15</td><td>≈0</td><td>on-prem</td></tr><tr><td>SmolVLM2-2.2B FT</td><td>0.90</td><td>1.00</td><td>3.4</td><td>15</td><td>≈0</td><td>on-prem</td></tr><tr><td>SmolVLM-500M FT</td><td>0.85</td><td>0.95</td><td>4.1</td><td>15</td><td>≈0</td><td>on-prem</td></tr></table>

Table 5: Case-study parameters. Q is the test-pool Micro-F1 (Table 1); Cov. is structured-output coverage (equivalently, one minus the JSON parse failure rate); $L _ { p 9 5 }$ for self-hosted models is A10G-projected from A100 measurements at $1 . 7 \times$ (Table 7), and for commercial APIs is $\mathrm { a \approx 1 . 5 \times }$ estimate of mean latency (where only mean is logged). Self-hosted candidates carry $c _ { \mathrm { f i x e d } } { = } \$ 7 34 / \mathrm { m o n t h }$ for one dedicated A10G instance (Eq. (2) assumption iv); $c _ { \mathrm { i n f } } { \approx } 0$ on already-provisioned capacity below saturation. Sources: GPT-4.1 pricing (OpenAI, 2025b); GPT-5 dated snapshot (OpenAI, 2025a); Claude Sonnet 4.5 pricing (Anthropic, 2025); Azure Document Intelligence (Microsoft Azure, 2025b) (also the OCR+Regex baseline, which incurs OCR-only cost); Azure Content Understanding (Microsoft Azure, 2025a); AWS g5.xlarge (Amazon Web Services, 2025).

Volume sensitivity. API and self-hosted ranking flips with volume because APIs carry near-zero fixed cost. Eq. (3) with the revised A10G economics gives crossover V<sup>∗</sup>≈439K docs/month for SmolVLM2-2.2B FT vs. OCR+GPT-4.1 over a 12- month horizon; Appendix C reports three volume regimes. At $V { = } 1 \mathbf { M }$ docs/month a single A10G instance is no longer sufficient; the self-hosted deployment scales to N=2 instances (\$32.6K/yr) versus \$44K/yr for OCR+GPT-4.1, a 26% saving, consistent with Chen et al. (2024); below ≈440K docs/month the API pipeline is unambiguously cheaper.<sup>3</sup> The cost-optimal model is therefore a function of the joint $( V , T , G _ { \mathrm { r e q } } )$ profile, not of benchmark accuracy alone.

Framework application to a second task. On the insurance-form task (Section 4.6), applying the same feasibility filters with identical thresholds $( Q _ { F 1 , \mathrm { { m i n } } } { = } 0 . 8 5 , Q _ { \mathrm { { c o v , m i n } } } { = } 0 . 9 9 )$ yields a different survivor set because coverage is task-dependent. SmolVLM2-2.2B FT drops to coverage 0.32 on the 36-field schema (versus 1.00 on checks) and is excluded; SmolVLM-500M FT narrowly fails coverage at 0.988. Qwen models retain full coverage and would be the framework’s recommendation under an appropriately relaxed latency budget for the richer schema. The recommendation is therefore sensitive to task-specific reliability, not just aggregate accuracy.

## 7 Conclusion

Practitioner guidance for vision-language IDP has lagged behind model turnover: public leaderboards report clean-document accuracy, but deployments are decided on the joint axes of quality, latency, governance, and cost. This study reframes approach selection along those axes. Across eleven systems on a 750-document held-out pool, finetuned open-source VLMs match or exceed every zero-shot commercial system on this schema under a fixed adaptation budget (Qwen2.5-VL FT > 0.98 F1), yet commercial APIs remain cheapest at low volume below the ≈440K docs/month breakeven— so the cost-optimal model is a function of the joint operating profile, not of benchmark accuracy. The contribution is the framework, dataset, and harness, released together so practitioners can shift the IDP conversation from “which model is most accurate?” to “which approach is feasible and cheapest under my constraints?” as the model lineup turns over.

## 8 Limitations

Our evaluation uses synthetic checks with a fourfield schema and synthetic 36-field insurance application forms; real production documents have higher template diversity and adversarial scan artifacts (heavy rotation, strong scan noise, multidocument pages) which we do not stress in this study. The synthetic-only design is itself a constraint: real-world financial documents carry PII and contractual restrictions that prevent public release, so a template-driven corpus is the only artefact we can open-source for reproducibility—the rendering pipeline mirrors production checks in layout, fonts, and scan-artefact distribution, but a residual distributional gap remains the principal generalisation caveat. The case-study cost figures are pilot-grade illustrative baselines that vary by region, contract, and time; A10G p95 latencies in Table 5 are projected from A100 measurements at 1.7× using cited industry references, and direct A10G measurement remains future work. Production deployments with elevated review burdens typ ically require 3–10× more engineering for security, observability, and retraining infrastructure, which raises the API-vs-self-hosted break-even volume but preserves the qualitative ranking. We evaluated a single fine-tuning recipe per model family, and the GPU-sensitivity study covers only A100 and H100 at batch size 1; larger batch sizes on any accelerator are extrapolations. Per-tier and percorruption-axis stratification within the synthetic pipeline is not part of this study; the cross-domain OOD evaluation (Section 4.6) provides the primary generalization signal instead. Our candidate set excludes fine-tuned purpose-built layout models (LayoutLMv3, Donut, Pix2Struct, UDOP); a likefor-like comparison against these encoders is a natural extension.

## References

Amazon Web Services. 2025. Amazon EC2 G5 instances (NVIDIA A10G). https://aws.amazon.c om/ec2/instance-types/g5/. Accessed: 2026- 05.

Anthropic. 2025. Claude API pricing. https://plat form.claude.com/docs/en/about-claude/pric ing. Accessed: 2026-05; Claude Sonnet 4.5 input \$3 / output \$15 per million tokens.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie

Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Ming-Hsuan Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923.

Baseten. 2025. NVIDIA A10 vs A100 GPUs for LLM and Stable Diffusion inference. https://www.ba seten.co/blog/nvidia-a10-vs-a100-gpus-f or-llm-and-stable-diffusion-inference/. Accessed 2026-07.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research. Featured Certification.

Lei Cui, Yiheng Xu, Tengchao Lv, and Furu Wei. 2021. Document AI: Benchmarks, models and applications. arXiv preprint arXiv:2111.08609.

Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems 30 (NIPS).

Adam W. Harley, Alex Ufkes, and Konstantinos G. Derpanis. 2015. Evaluation of deep convolutional nets for document image classification and retrieval. In ICDAR.

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. LayoutLMv3: Pre-training for document AI with unified text and image masking. In ACMMM.

Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. 2019. FUNSD: A dataset for form understanding in noisy scanned documents. In International Workshop on Open Services and Toolsfor Document Analysis (OST@ICDAR).

Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. OCR-free document understanding transformer. In European Conference on Computer Vision (ECCV).

Siddhant Kulkarni and Yukta Kulkarni. 2026. Benchmarking multi-agent LLM architectures for financial document processing: A comparative study of orchestration patterns, cost-accuracy tradeoffs and production scaling strategies. arXiv preprint arXiv:2603.22651.

Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. 2023. Pix2Struct: Screenshot parsing as pretraining for visual language understanding. In Proceedings ofthe 40th International Conference on Machine Learning (ICML), volume 202 of PMLR, pages 18893–18912.

Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Manuel Cuenca Jiménez, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. 2025. SmolVLM: Redefining small and efficient multimodal models. In Second Conference on Language Modeling (COLM).

Hubert Michalak and Krzysztof Okarma. 2020. Robust combined binarization method of non-uniformly illuminated document images for alphanumerical character recognition. Sensors, 20(10):2914.

Microsoft Azure. 2025a. Azure AI content understanding pricing. https://azure.microsoft.com/en -us/pricing/details/content-understanding /. Accessed: 2026-05.

Microsoft Azure. 2025b. Azure AI document intelligence pricing. https://azure.microsoft.com/ en-us/pricing/details/ai-document-intelli gence/. Accessed: 2026-05.

Modal Labs. 2025. GPU types for AI workloads. https://modal.com/blog/gpu-types. Smallbatch LLM inference typically memory-bandwidth bound; A100 80GB 2 TB/s vs A10 0.6 TB/s bandwidth; published 2025-01-27, accessed 2026-07.

NVIDIA. 2021. NVIDIA A100 Tensor Core GPU datasheet. https://www.nvidia.com/content /dam/en-zz/Solutions/Data-Center/a100/p df/nvidia-a100-datasheet-us-nvidia-17589 50-r4-web.pdf. 312 TFLOPS FP16/BF16 Tensor (dense), 1,555–2,039 GB/s memory bandwidth depending on 40/80 GB and PCIe/SXM; accessed 2026-07.

NVIDIA. 2022. NVIDIA A10G Tensor Core GPU datasheet. https://d1.awsstatic.com/pro duct-marketing/ec2/NVIDIA\_AWS\_A10G\_Dat aSheet\_FINAL\_02\_17\_2022.pdf. 70 TFLOPS FP16/BF16 Tensor, 600 GB/s memory bandwidth, 24 GB GDDR6; accessed 2026-07.

OpenAI. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

OpenAI. 2025a. GPT-5 model and API pricing (archived snapshot). https://developers.ope nai.com/api/docs/pricing. Accessed: 2026-05; GPT-5 dated snapshot used for the experiments in this paper.

OpenAI. 2025b. OpenAI API pricing. https://open ai.com/api/pricing/. Accessed: 2026-05.

Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. 2019. CORD: A consolidated receipt dataset for post-OCR parsing. In Workshop on Document Intelligence at NeurIPS.

Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. 2023. Unifying vision, text, and layout for universal document processing. In CVPR.

U.S. Bureau of Labor Statistics. 2024. Occupational employment and wage statistics: Software developers (15-1252). Technical report, U.S. Department of Labor. Accessed: 2026-05.

Zilong Wang and Xiaoyu Shen. 2025. Hybrid OCR-LLM framework for enterprise-scale document information extraction under copy-heavy task. arXiv preprint arXiv:2510.10138.

Zining Wang, Tongkun Guan, Pei Fu, Chen Duan, Qianyi Jiang, Zhentao Guo, Shan Guo, Junfeng Luo, and Wei Shen. 2025. Marten: Visual question answering with mask generation for multi-modal document understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Wenwen Yu, Zhibo Yang, Yuliang Liu, and Xiang Bai. 2025. DocThinker: Explainable multimodal large language models with rule-based reinforcement learning for document understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

## A Additional Discussion

Two practitioner-facing patterns emerge. Date is the universal commercial bottleneck: across all five commercial / reasoning systems Date has the lowest per-field F1 (0.47–0.79; Claude’s precision collapses to 0.31), so a downstream date-validation step (constrained format plus business-rule range) is a more robust intervention than additional training data. Reasoning models earn only marginal accuracy: GPT-5’s ≈0.01 F1 lift on the four core fields is dwarfed by ≈ 3× token cost and ≈ 2× latency; its honest value-add is schema coverage (AmountInWords, CheckNumber), so it pays off when those extra fields are required.

## B Reproducibility Details

We release dataset generation code, fine-tuning scripts, prompts, and evaluation harnesses at ht tps://github.com/manulife-ai/beyond-acc uracy-vlm-emnlp2026/. The settings below are constant across all configurations unless noted.

Prompt. A single prompt is used for every VLM and LLM in the study, modulo the image attachment mechanism required by each API:

You are extracting fields from a U.S. business check.

HuggingFaceTB/SmolVLM2-2.2B-Instruct, HuggingFaceTB/SmolVLM-500M-Instruct, HuggingFaceTB/SmolVLM2-256M-Instruct.

Return a JSON object with exactly these keys: “Payee” (string), “Amount” (number, no currency symbol), “Date” (string MM/DD/YYYY), “Bank” (one of: Chase, Citi, BOA).

If a field is unreadable, return an empty string for that key.

Return ONLY the JSON object, no prose, no markdown.

For reasoning models we additionally prepend the system message “Think briefly, then answer.” which we found stabilises the JSON output without measurably changing accuracy.

Decoding. All open-source VLMs and commercial / reasoning APIs are run with temperature = 0.0, top\_p = 1.0, and max\_new\_tokens = 1024, set to avoid long-output truncation on multi-field responses. Stop tokens are the model defaults; we do not impose additional JSON delimiters.

JSON failure handling. Every raw model output is passed through a three-stage parser: (i) strip markdown fences (“‘json ... “‘); (ii) extract the outermost balanced {...}; (iii) json.loads. If all three stages fail the sample is recorded as JSON\_FAIL and counted as a full miss on every field (4 false negatives), which is the worst case for that model and is what the JSON-failure column in Table 1 measures. We do not perform any retry, schema- repair, or constrained-decoding pass; the numbers reported are first-try outputs.

Fine-tuning. Qwen2.5-VL-3B/7B are fine-tuned with LoRA (r=16, α=32, dropout = 0.05, target modules: all attention projections and the MLP gate/up/down); SmolVLM2 variants are full finetuned. All runs use the AdamW-8bit optimiser, effective batch size 4 (per-device 1, gradient accumulation 4), cosine schedule with 3% warmup, weight decay 0.01, and bf16 mixed precision on a single A100 80 GB. Seed 42 throughout.

Compute. Each Qwen fine-tune completes in 6– 14 GPU-hours on the A100; SmolVLM fine-tunes complete in 2–5 GPU-hours. Per-document inference latency in Table 1 is measured on an NVIDIA A100 80 GB PCIe under a Databricks runtime 18.2 image (PyTorch 2.9 / CUDA 12.9, bf16, batch size 1, five-document warmup discarded). Section 4.5 reports the same fine-tuned models rebenchmarked on an NVIDIA H100 NVL under the same software stack for the GPU-sensitivity analysis.

Model versions. Open-source checkpoints are pulled from Hugging-Face: Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct,

Commercial APIs used the following dated snapshots: GPT-4.1 (gpt-4.1-2025-04-14) (OpenAI, 2025b), GPT-5 (gpt-5-2025-08-07) (OpenAI, 2025a), Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) (Anthropic, 2025), Azure Content Understanding (Microsoft Azure, 2025a), and Azure Document Intelligence (Microsoft Azure, 2025b) for the OCR+GPT-4.1 pipeline. Specific request timestamps are logged and released with the harness.

## C Volume-Sensitivity Table

<table><tr><td>Model</td><td>5K/mo</td><td>100K/mo</td><td>1M/mo</td></tr><tr><td>OCR+GPT-4.1</td><td>$8.2K</td><td>$11.6K</td><td>$44.0K</td></tr><tr><td>Qwen-3B FT</td><td></td><td></td><td></td></tr><tr><td>SmolVLM2-2.2B FT</td><td>$23.8K</td><td>$23.8K</td><td>$32.6K</td></tr></table>

Table 6: Annual cost (\$K) at three volumes (T=12) under the revised A10G-dedicated deployment. Bold marks the cost-optimal feasible option when no governance constraint is applied. Qwen-3B FT is excluded on A10G-projected latency across all volumes; on H100- class hardware it becomes feasible and can be re-costed with the corresponding hourly rate. Referenced from Section 6, “Volume sensitivity”.

## D GPU Sensitivity Details

Table 7 reports per-document latency and peak memory on NVIDIA A100 80 GB PCIe and NVIDIA H100 NVL for all five fine-tuned opensource models, measured under identical software (PyTorch 2.9 / CUDA 12.9, bf16, batch size 1) on the 750-document held-out pool. Accuracy is unchanged across GPUs (∆F1 ≤ 0.003), so the relative ordering is stable; practitioners on H100-class hardware can rescale the case-study cost terms approximately linearly by GPU-hour price.

## E Error Taxonomy Details

Quantitative breakdown of the three error categories summarised in Section 4:

(E1) Degenerate repetition (SmolVLM2-256M, pretrained SmolVLM-500M): the model emits looping arrays such as [123.0, 123.0, 123.0, ...] that exhaust the token budget without closing the JSON object. Accounts for $\ge ~ 9 9 \%$ of pretrained sub-500M outputs and persists in finetuned 256M as the 30.7% JSON parse-failure rate reported in Table 1.

<table><tr><td>Model</td><td>Mem mean</td><td>A100 80 GB</td><td> $\mathsf { p } 9 5$ </td><td>H100 NVL mean</td><td>p95 H/A</td></tr><tr><td>SmolVLM2-256M</td><td>(GB) 0.5</td><td>(s)</td><td>(s)</td><td>(s)</td><td>(s) speedup</td></tr><tr><td>SmolVLM-500M</td><td>0.9</td><td>1.62 0.96</td><td>2.31 2.42</td><td>1.10 1.47 0.70</td><td> $1 . 4 7 \times$  1.38×</td></tr><tr><td>SmolVLM2-2.2B</td><td>4.2</td><td>1.47</td><td>1.96</td><td>1.51 1.19</td><td>1.59×</td></tr><tr><td>Qwen2.5-VL-3B</td><td>7.1</td><td>4.18</td><td>5.79</td><td>0.93 2.50 3.43</td><td>1.68×</td></tr><tr><td>Qwen2.5-VL-7B</td><td>15.5</td><td>3.53</td><td>4.79</td><td>2.14 2.89</td><td>1.65×</td></tr></table>

Table 7: Per-document inference latency on A100 80 GB PCIe vs. H100 NVL (bf16, batch size 1, 750- document held-out pool). Mem is peak GPU memory during inference (GPU-independent). H/A speedup is A100 mean / H100 mean.

(E2) Phonetic Payee misspelling (Qwen-3B/SmolVLM2-2.2B): Mark Gwazy vs. Mark Gray; Zachary Villages vs. Zachary Villegas. Errors cluster on visually-ambiguous character pairs (g/y, s/z, e/i).

(E3) Date over-extraction (Claude Sonnet 4.5, GPT-4.1 vision, OCR+GPT-4.1): the model extracts every date-like substring on the check. Claude is the worst case (686 FP against 304 TP); GPT-4.1 vision (337 FP) and OCR+GPT-4.1 (360 FP) are milder but qualitatively identical.

## F Per-Field-Type Analysis (Insurance Forms)

Field taxonomy on the 36-field insurance schema, derived from ground-truth value cardinality on the 1,000-document test set: 9 checkbox / boolean-flag fields (citizenship\_status, sex, occupation\_status and six other Y/N flags), 22 structured-text fields (names, addresses, phone, dates, IDs, SSN, salaries, states, countries), and 5 free-form fields (two bankruptcy\_details lines, job\_duties\_description, occupation\_status\_other\_description, green\_card\_or\_visa\_type).

The per-field-type breakdown below is reported for SmolVLM-500M FT, SmolVLM2-256M FT, GPT-4.1 (vision), and OCR+GPT-4.1. Overall accuracies for Qwen2.5-VL-3B FT, Qwen2.5-VL-7B FT, and SmolVLM2-2.2B FT appear in Table 4.

<table><tr><td>Model</td><td>Ckbox (n=9)</td><td>Text (n=22)</td><td>Free (n=5)</td><td>Overall (n=36)</td></tr><tr><td>SmolVLM-500M FT</td><td>0.987</td><td>0.929</td><td>0.583</td><td>0.895</td></tr><tr><td>SmolVLM2-256M FT</td><td>0.160</td><td>0.151</td><td>0.093</td><td>0.145</td></tr><tr><td>GPT-4.1 (vision)</td><td>0.476</td><td>0.934</td><td>0.560</td><td>0.768</td></tr><tr><td>OCR+GPT-4.1</td><td>0.398</td><td>0.947</td><td>0.569</td><td>0.757</td></tr></table>

Table 8: Per-field-type mean accuracy on the insuranceform test set (n=1,000). Categories are 9 boolean-flag checkboxes, 22 structured-text fields, and 5 free-form fields (see paragraph above). Overall accuracy matches the Insurance column of Table 4 up to true↔yes normalization on Y/N outputs, which is applied to GPT-4.1 rows (GPT-4.1 vision 0.738→0.768 and OCR+GPT-4.1 0.731→0.757); fine-tuned SmolVLM rows are unchanged. Bold marks the best in column.

## G Seed-study Variance (Check Task)

To quantify the sensitivity of the fine-tuning outcomes to random seed, we run a 3-seed Monte-Carlo cross-validation study on the check task. Seeds 42, 1337, and 2718 each draw an independent grouped 80/20 split from the pooled 3,748- image dataset. The grouped split assigns whole (Payee, Amount, Date) components to one side of the split, guaranteeing zero group overlap between train and test. Each seed’s train and test sets preserve the paper’s 10/30/25/20/15% Clean-through-Extreme tier mixture within 0.2 percentage points. All five open-source models are trained under the recipes of Section 3 on each seed’s 2,999-document train set and scored on that seed’s 749-document test set. Because the test set is resampled per seed, the aggregate F1 values in Table 9 use a different pool than the fixed 750-document evaluation reported in Table 1 and should be read as a variance study rather than a substitute for that evaluation.

<table><tr><td>Model</td><td>FT F1 (mean ± std) ∆ vs. pretrained</td></tr><tr><td>SmolVLM2-256M</td><td> $0 . 7 1 9 \pm 0 . 0 1 1$  +0.719</td></tr><tr><td>SmolVLM-500M</td><td> $0 . 8 3 2 \pm 0 . 0 0 8$  +0.832</td></tr><tr><td>SmolVLM2-2.2B</td><td> $0 . 9 0 3 \pm 0 . 0 0 2$  +0.396</td></tr><tr><td>Qwen2.5-VL-3B</td><td> $\mathbf { 0 . 9 8 3 \pm 0 . 0 0 2 }$  +0.141</td></tr><tr><td>Qwen2.5-VL-7B</td><td> $\mathbf { 0 . 9 8 5 \pm 0 . 0 0 2 }$  +0.068</td></tr></table>

Table 9: Seed-study aggregate F1 across three Monte-Carlo splits (n=749 per seed). The Qwen-3B versus Qwen-7B 0.002 gap is within seed noise; sub-billion models are noticeably more seed-sensitive than models at 2B parameters or above. Pretrained F1 is 0.000 for both sub-billion SmolVLMs (structural JSON collapse), 0.507 for SmolVLM2-2.2B, 0.842 for Qwen-3B, and 0.917 for Qwen-7B, all with std $\leq 0 . 0 0 3$

Per-field variance. Table 10 reports the same three seeds broken out by field. Payee is the universal ceiling across model sizes; Bank saturates at 1.000 for both Qwen fine-tunes, consistent with the closed 3-way Bank vocabulary noted in Section 3.
<table><tr><td>Model</td><td>Payee</td><td>Amount</td><td>Date</td><td>Bank</td></tr><tr><td>SmolVLM2-256M</td><td> $0 . 5 8 5 \pm 0 . 0 1 5$ </td><td> $0 . 7 0 6 \pm 0 . 0 2 2$ </td><td> $0 . 7 4 4 \pm 0 . 0 1 0$ </td><td> $0 . 8 1 8 \pm 0 . 0 1 1$ </td></tr><tr><td>SmolVLM-500M</td><td> $0 . 6 9 5 \pm 0 . 0 2 1$ </td><td> $0 . 8 5 4 \pm 0 . 0 1 0$ </td><td> $0 . 8 4 2 \pm 0 . 0 0 6$ </td><td> $0 . 9 1 6 \pm 0 . 0 0 6$ </td></tr><tr><td>SmolVLM2-2.2B</td><td> $0 . 8 0 0 \pm 0 . 0 0 7$ </td><td> $0 . 9 1 0 \pm 0 . 0 1 0$ </td><td> $0 . 8 8 7 \pm 0 . 0 0 4$ </td><td> $0 . 9 9 7 \pm 0 . 0 0 0$ </td></tr><tr><td>Qwen2.5-VL-3B</td><td> $0 . 9 5 9 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 7 7 \pm 0 . 0 0 5$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Qwen2.5-VL-7B</td><td> $0 . 9 6 5 \pm 0 . 0 0 4$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 7 9 \pm 0 . 0 0 2$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

Table 10: Per-field entity-level F1, mean ± std across three Monte-Carlo seeds.

Framework re-derivation. Applying the feasibility set of Eq. (1) with $Q _ { F 1 , \mathrm { { m i n } } } { = } 0 . 8 5$ and $Q _ { \mathrm { c o v , m i n } } { = } 0 . 9 9$ to the seed-study means gives $\mathcal { M } _ { F } { = } \{ \mathrm { S m o l V L M 2 - } 2 . 2 \mathrm { B } \ F \mathrm { T } , \mathrm { Q w e n } { - } 3 \mathrm { B } \ F \mathrm { T } , \mathrm { Q w e n } { - } 7 \mathrm { B } \ F \mathrm { T } \}$ Under the illustrative case-study profile of Section 6 the cheapest survivor remains SmolVLM2- 2.2B FT, matching the $m ^ { * }$ derived from the main results. SmolVLM-500M FT fails both accuracy (F1 $0 . 8 3 2 < 0 . 8 5 )$ and coverage $( 0 . 9 5 1 < 0 . 9 9 )$ under this resampled protocol, so its exclusion does not depend on any single threshold.