# NPCBench: A Clinical Apprenticeship Benchmark for Multimodal Care-Pathway Reasoning in Nasopharyngeal Carcinoma

Pengkai Wang<sup>1,2∗</sup> Wei-Wei Zhang<sup>3,4∗</sup> Yan Li<sup>3,4∗</sup> Min Tang<sup>5</sup> Zhitian Hou<sup>1,2</sup> Zeyu Liu<sup>2</sup> Guanghao Zhu<sup>1,2</sup> Yuanyi Wang<sup>1,2</sup> Yanggan Gu<sup>1,2</sup> Wenjun Wang<sup>1,2</sup> Minheng Ni<sup>1,2</sup> Congkai Xie<sup>2</sup> Zhijie Sang<sup>2</sup> Jianmin Wu<sup>1,2</sup> Ying Sun<sup>3,4†</sup> Hongxia Yang<sup>1,2†</sup>

<sup>1</sup>The Hong Kong Polytechnic University <sup>2</sup>infix.ai <sup>3</sup>Sun Yat-sen University <sup>4</sup>Sun Yat-sen University Cancer Center <sup>5</sup>Monash University

§ GitHub Code Hugging Face Data

## Abstract

Current medical benchmarks typically break down clinical competence into isolated factual questions, single-image tasks, or static case vignettes. These tests score individual answers, but reveal less about whether a model can integrate subspecialty knowledge with multimodal evidence and track a patient’s condition throughout a course of care. To close this gap, we introduce NPCBench, a clinical apprenticeship benchmark for nasopharyngeal carcinoma (NPC). NPCBench mirrors specialist training, from guideline and foundational knowledge to case planning, longitudinal review, CarePath integration, and open-ended consultation. It contains 4,777 component and consultation items and a 100-patient CarePath cohort with 417 scored diagnostic and care tasks. The episode-level evaluation tests whether diagnostic and care answers are jointly correct within a care pathway. By linking stepwise tasks to patient-level care trajectories, NPCBench evaluates whether models ground decisions in visual evidence, interpret changes in patient state, and meet clinical requirements in consultation. Across 30 evaluated models, we observe a composition gap: strong performance on individual tasks does not ensure jointly correct diagnostic and care answers for the same patient. Among 21 multimodal models, strict CarePath success reaches at most 23%. Paired controls and physician calibration further examine evidence grounding, longitudinal revision, and consultation judging. Overall, NPCBench broadens the evaluation of medical artificial intelligence beyond static question-answering paradigms to an apprenticeship-oriented assessment of clinical workflows, thereby offering a testbed for evidence-based and temporally coherent reasoning in oncology care.

## 1 Introduction

Medical large language models (LLMs) and multimodal LLMs (MLLMs) now achieve strong results in medical examinations, visual question answering, and broad benchmark suites [1–6]. However, exam-style accuracy offers limited insight into subspecialty care, where clinicians must synthesize disease-specific guidelines, multimodal evidence, and evolving patient states over time. During clinical apprenticeship, physicians do not learn these skills in isolation. Recalling a clinical guideline is only the first step; specialists must also ground staging in diagnostic images, tailor treatment plans to patient constraints, and revise decisions as new responses or toxicities emerge. Evaluating medical AI therefore requires following this same progression from foundational knowledge to longitudinal management. While isolated tasks can pinpoint weaknesses in individual skills, only linked patient episodes can reveal whether these skills succeed together for the same patient [7–9].

![](images/cae1035b6f965fc2cb11056728ea8be8b6465f395e8768041248d21ae61cca88.jpg)  
Figure 1: Overview of NPCBench. (a) NPC care combines pathology, MRI, biomarkers, and clinical records under guideline constraints. (b) The apprenticeship framework spans component skills, longitudinal revision, and CarePath evaluation; O1 assesses consultations. (c) Changing patient states drive reporting of local and joint correctness. CarePath evaluates each decision using the provided clinical context; separate serial controls examine feedback from prior answers.

Evaluating the progression from individual competencies to integrated clinical care requires a disease setting in which multimodal evidence and sequential decisions are closely connected. Nasopharyngeal carcinoma (NPC), prevalent in Southern China and Southeast Asia [10–12], is managed according to staging systems and clinical guidelines [13–15]. Diagnosis, staging, and risk assessment require integrating histopathology, multi-sequence MRI, and plasma EBV DNA [11, 16]. Clinical evidence and management objectives vary across stages of care, with subsequent decisions shaped by both the current patient state and preceding clinical events. During treatment and follow-up, tumor response, treatment-related toxicities, and recurrence or progression may influence management strategies [12, 14, 15]. NPC therefore provides a suitable setting for evaluating both individua specialist competencies and their integration across successive stages of care.

To systematically assess these capabilities, we introduce NPCBench, a clinical apprenticeship benchmark for NPC modeled on the progressive training of specialist physicians. As shown in Fig. 1, it organizes the evaluation into six progressive levels and an open-ended consultation suite. L1 and L2 test guideline recall and application. L3 tests pathology and MRI evidence grounding. L4 tests case planning from a single patient state, L5 longitudinal decision revision, and L6 linked CarePath episodes. O1 evaluates rubric-based specialist consultation. By pairing component tasks with patient trajectories, NPCBench asks a central measurement question: Are diagnostic and care answers jointly correct for the same patient when component scores are high? Models with similar average accuracy can exhibit different patient-level error patterns, or even select the right treatment while misinterpreting the underlying MRI evidence. NPCBench therefore pairs task-level scores with strict patient-level correctness, while holding patients and questions fixed in matched controls to examine how visual evidence, clinical chronology, and answer formats shape model decisions.

Using NPCBench, we investigate four research questions (RQ1–RQ4) across 30 medical and general purpose models. RQ1 examines how well models translate guideline knowledge into case-level planning: guideline recall exceeds 92%, whereas the highest case-planning accuracy is 71.9%. RQ2 probes whether staging predictions are grounded in accurate visual interpretation: complete invasion correctness reaches at most 5.0% on the 100-patient anatomical panel, showing that correct stage labels often lack complete anatomical support. RQ3 examines decision revision and patient-level success: history can correct answers and introduce new errors. In CarePath, Gemini-3.1 achieves 89.9% Care accuracy but 11.0% Strict success; Claude-4.6 has the highest Strict success at 23.0%. RQ4 assesses consultation safety and automated judge reliability: GPT-5.1, the highest-scoring consultation model, triggers major-error criteria in 14.6% of responses under the primary judge, while rubric judges show sensitivity–specificity imbalances against physician annotations.

We summarize the main contributions of this work as follows.

① A clinical apprenticeship evaluation framework. We shift the evaluation focus from static answer correctness to pathway correctness. NPCBench assesses whether models make consistent clinical decisions across patient-state transitions, integrating guideline constraints, multimodal evidence grounding, and longitudinal disease evolution.

② A specialist-curated multimodal and longitudinal dataset. We release 4,777 component and consultation items alongside 100 CarePath episodes with 417 scored diagnostic and care tasks, unifying guideline rules, pathology slides, multi-sequence MRI scans, longitudinal EHR trajectories, and consultation rubrics within a single oncology pathway.

③ Diagnostic dissection of the clinical composition gap. Across 30 models, we combine capability profiling with controlled interventions and physician-calibrated judging to identify where visual grounding separates from treatment decisions, how patient history changes decisions, and where high local scores coexist with patient-level errors.

## 2 Related Work

From exam QA to clinically structured benchmarks. Early evaluations of medical LLMs relied on static, examination-style question–answering benchmarks, including MedQA [17], MedMCQA [18], PubMedQA [19], the medical subsets of MMLU [20], MultiMedQA [2], MedBench [21], CMB [22], and CMExam [23]. These benchmarks capture medical knowledge and local reasoning, but they do not test whether a model can accumulate patient evidence, review decisions, and preserve coherence across the care pathway. Recent broad suites such as MedHELM [8], MedXpertQA [24], and AnesSuite’s AnesBench [25] move the evaluation toward clinically meaningful capabilities and specialty-aware reasoning. NPCBench follows this direction and makes the clinical apprenticeship ladder explicit: it measures component skills alongside longitudinal decisions and episode-level care correctness.

(a)  
![](images/3e19fa69a42f9ef6d0f830b79f3167ab6e6cf7ce6b0686170dc1b1405129f861.jpg)

![](images/231a90c8765efb6ff65db22820ea4f4a8af85ac2d06c063016ebeca79f67d896.jpg)

![](images/86eaa5f3bede04b22f058df0c6517ef8b6c6bf2bfda47b2d173b5825e5dc64d1.jpg)

(d)  
![](images/c011009c1950bff436291d44359c121cad86ce15405d6ce00ed13079e98f25c0.jpg)

(e)  
![](images/464b4f6cc7752b598d8ef670bea83fe38e7b1b958ddd2cc147c65115a5acc6dd.jpg)

(f)  
![](images/a5cbff23b1375d3ef091722fce221437ace5f127709ad2cdec69482721fc80c4.jpg)

(g)  
![](images/48b82c7f55c8a827ccb94d0ba97af91f624e4c0614bb9bc265d104e3b124be3b.jpg)

(h)  
![](images/0569db4c351e9042eb914d0284a05a35f6e037b9810c834904e09479e3cbb32e.jpg)  
Figure 2: Task composition of NPCBench. Evaluation items are distributed across six progressive levels (L1–L6) and the open-ended consultation suite (O1). Hatched slices each account for less than 5% of their subset; L6 shows its 217 care decisions. More clinical category names are defined in Appendix A.

Multimodal, longitudinal, and rubric-based clinical evaluation. Medical MLLM benchmarks have expanded from image QA datasets such as VQA-RAD [26], SLAKE [27], and PathVQA [28] to broader evaluations of multimodality, trustworthiness, endoscopy, clinical reasoning, and sequence grounding [29–33]. In parallel, HealthBench [34], LiveMedBench [35], LLMEval-Med [36], MedDialogRubrics [37], and ClinConsensus [38] use expert-validated rubrics or temporally grounded cases to evaluate open-ended clinical responses. Workflow-oriented resources such as MedJourney [39], MedAgentBench [40], MedAgentBoard [41], and MTBBench [42] further shift evaluation from static snapshots toward state transitions. NPCBench brings these directions into a single disease setting: pathology and MRI evidence grounding, patient-level longitudinal revision, strict CarePath episodes, and criterion-level consultation rubrics. Appendix B gives a fuller comparison with recent medical benchmark families.

## 3 NPCBench: A Clinical Apprenticeship Benchmark

## 3.1 Overview

NPCBench organizes the clinical apprenticeship framework into a progressive capability ladder (L1–L6) and cross-level consultation rubrics (O1). The ladder spans guideline memory, atomic application, multimodal evidence grounding, case-level planning, longitudinal decision revision, and linked care episodes. Lower levels (L1–L4) assess individual capability components; L5–L6 examine longitudinal decisions and joint diagnostic–care correctness within patients. O1 complements these tasks by assessing free-text consultation quality. These tasks assess whether specialist decisions follow guidelines, use multimodal evidence, and remain consistent with a patient’s clinical course.

Four design principles guide task construction and clinical review: (i) Competence isolation requires each level in the progressive capability ladder (L1–L6) to measure a distinct, clinically meaningful component of specialist competence. (ii) Evidence-boundary control fixes the available evidence at the intended clinical decision point, removing future treatment, outcomes, and other answer-leaking cues. (iii) Clinical determinacy requires reference decisions supported by guidelines or expert consensus, given the provided clinical evidence. (iv)Auditable open-ended scoring uses case-specific consultation rubrics (O1) to assess clinical justification, evidence grounding, conditional reasoning, and safety. All patient materials are de-identified, and NPC oncologists verified component validity, decision determinacy, and the fidelity of patient-state transitions. Figure 2 shows the composition of each subset, and Fig. 3 illustrates the pipeline for constructing L1–L5 items.

## 3.2 Construction and Validation

Expert-curated clinical materials are converted into standardized evaluation items and reference answers. L1–L2 were derived from CSCO and NCCN guidelines; L3 from pathology and multisequence MRI images; and L4–L6 and O1 from de-identified real-world patient records, covering case level planning, longitudinal decision-making, full CarePath evaluation, and open-ended consultations with scoring rubrics.

Multidisciplinary reviewers in oncology, pathology, and radiology assessed clinical validity, evidence sufficiency, and reference standards. For L1–L2, experts verified guideline consistency and answer determinacy; for L3, they validated pathology labels, MRI sequence and anatomical structure identification, invasion status, and T/N staging against imaging evidence. For L4, they assessed case sufficiency and appropriateness of reference decisions given patient-specific clinical states; for L5–L6, they verified evidence from decision-points, reference actions, and transitions from clinical state; and for O1, they reviewed core clinical requirements, supporting evidence, conditional recommendations, and major/minor error criteria to ensure clinically valid and auditable scoring.

## 3.3 Evaluation Levels and Scoring Units

L1.G-Mem: Guideline memory. The 250 expert-reviewed questions assess NPC guideline knowledge: 153 have both CSCO and NCCN support, 80 CSCO support only, and 17 NCCN support only.

![](images/5cbb8e1889a122e4de55551dd093bc535f4491199c3de24408ffcee87743d6ef.jpg)  
Figure 3: Construction of the L1–L5 component tasks. Guideline questions, image tasks, and clinical checkpoints undergo specialist review of the available evidence and reference answers.

They probe treatment indications, staging implications, recommendation boundaries, surveillance, and contraindications. Item accuracy is the primary endpoint.

L2.A-App: Atomic guideline application. The 185 short-vignette questions cover initial definitive management, recurrence, metastatic/systemic management, and toxicity or contraindications. The main score averages accuracy over these four capability groups, giving each group equal weight.

L3.MM-EG: Multimodal evidence grounding. Pathology contains 69 overview cases (39 NPCpositive and 30 negative). MRI comprises 200 sequence-recognition questions, 400 T-stage cases, and 400 N-stage cases, with patient-disjoint T and N cohorts. Within the 400 T-stage cases, a 100-patient anatomical panel provides 2,700 structure cards covering 18 base structures, evaluating both structure identification and tumor invasion assessment. Overall, we report pathology, MRI sequence, structure-identification, T-stage, and N-stage accuracy separately, with additional metrics for invasion accuracy, joint structure–invasion correctness, and evidence-grounded T-stage success.

L4.C-Plan: Case-level planning. The 165 questions cover 84 patient groups and provide clinical snapshots of stage, treatment history, response, organ function, and other relevant constraints. Plausible alternative plans test whether a model identifies the next action appropriate to that specific state. Patient-macro accuracy weights patients equally.

L5.L-Rev: Longitudinal revision. The 305 questions cover 100 trajectories with one to five scored checkpoints at treatment, reassessment, toxicity, recurrence, or follow-up. Each checkpoint requires interpreting new findings in the context of earlier treatment and response. We report turn-micro accuracy and all-turn patient success. The primary task supplies full history in one prompt; matchedhistory and serial-interaction controls are evaluated separately.

L6.CP-Epi: CarePath episodes. The 100-patient CarePath cohort comprises 417 tasks: 100 pathology assessments, 100 MRI T-stage assessments, and 217 care decisions. Clinical records remain patient-linked; care-decision selection uses model-informed difficulty enrichment to prioritize clinical decisions that challenge models (Appendix C.3.5). Strict requires correct pathology, MRI T-stage, and every care answer; structure predictions provide auxiliary diagnostics. Each care question is answered independently with clinical context, incorporating the model’s pathology/T predictions and reference N/M/overall stages where required.

O1.C-Rub: ConsultRubrics. The 103 consultation items span 42 patients and 1,726 rubric criteria: 1,069 positive requirements and 657 negative criteria. Satisfied positive requirements add weighted credit; triggered major and minor errors deduct two and one points, respectively. The sum is floored at zero after item-level aggregation, normalized to 0–100, and averaged over all 103 item scores. The primary judge and physician calibration are described below.

## 4 Experimental Evaluation

## 4.1 Experimental Setup

Evaluation Models. We evaluated 30 medical and general-purpose models, including HuatuoGPTo1 [43], Lingshu [4], Qwen, GPT-OSS [44, 45], and models from the GPT and Gemini families [46, 47]. While all 30 models complete the text-based tasks (L1, L2, L4, L5 and O1), the 21 multimodal models also complete the image-grounded tasks L3 and L6 (full model inventory in Appendix C.1).

Evaluation protocol. All models follow standardized task templates and scoring rules, with modelspecific image processing and generation settings detailed in Appendix C and full prompts in Appendix E. For O1 calibration, automated judges and physicians evaluate identical response sets within each panel. All locally hosted generators and judges run on four NVIDIA A800-SXM4-80GB GPUs.

Patient-level Metrics. On CarePath episodes, we report both decision-level accuracy (Care) and strict patient-level success (Strict), which requires correct pathology, MRI T stage, and all selected care answers for the same patient. For a given model, let $p = 1 , \ldots , P$ index patients and $S _ { p }$ denote the nonempty set of scored care decisions for patient $p ,$ with $n _ { p } = | S _ { p } |$ and $\begin{array} { r } { N = \sum _ { p = 1 } ^ { P } n _ { p } } \end{array}$ . Let $c _ { p j }$ $d _ { p }$ and $t _ { p } \in \{ 0 , 1 \}$ } indicate whether the care response $j$ , the pathological label, and the T-stage MRI label are correct, respectively. With P patients and N total decisions, the percentage endpoints are

$$
\mathrm { C a r e } = \frac { 1 0 0 } { N } \sum _ { p = 1 } ^ { P } \sum _ { j \in { \cal S } _ { p } } c _ { p j } , \qquad \mathrm { S t r i c t } = \frac { 1 0 0 } { P } \sum _ { p = 1 } ^ { P } d _ { p } t _ { p } \prod _ { j \in { \cal S } _ { p } } c _ { p j } .\tag{1}
$$

Appendix C.2.1 defines the remaining task metrics and 95% confidence intervals, and Appendix C.5.1 details the consultation scoring.

## 4.2 Guideline Knowledge and Case Planning

Table 1 summarizes model performance across all benchmark tasks. Addressing RQ1, we examine how foundational guideline competence translates into case-level planning. Although Gemini-3.1-Pro-Preview leads the guideline recall with 94.0% and the atomic application with 92.5%, its precision drops to 71.9% in L4, followed by Claude-Sonnet-4.6 and Gemini-2.5-Pro. In addition, models with nearly identical L1 and L2 scores—such as GPT-5.1 and DeepSeek-V3.1—diverge sharply on L4 planning (63.5 vs. 51.8%). Although these subsets differ in item composition and weighting, this divergence shows that multi-factor case planning must be evaluated directly rather than inferred from guideline accuracy. Furthermore, a paired comparison on 71 physician-audited L1 items reveals that even high guideline scores partly rely on multiple-choice cues: removing answer options while holding the target knowledge fixed lowers accuracy by 7.0 percentage points (pp) for Gemini-3.1-Pro-Preview, 12.7 pp for GPT-4.1, and 15.5 pp for GPT-5.1 shown as in Figure. 7i.

## 4.3 Evidence Grounding and Longitudinal Revision

Stage accuracy and anatomical grounding reveal different weaknesses. For RQ2, we first assess whether models can assign report-derived T and N categories across separate 400-patient cohorts. The highest T-stage accuracy is 43.5% across 400 patients, compared to a baseline of 37.5% in the majority-class. N-stage accuracy reaches 42.8% across a separate 400 patients, whose N1 majority baseline is 40.0%. However, overall accuracy masks differences in performance across different stages: Qwen3.6-27B achieves the highest overall N accuracy of 42.8% but only 31.9% balanced accuracy, whereas Gemini-3.1 maintains consistent performance across stages. Thus the total stage accuracy alone can therefore obscure class-imbalance shortcuts, a pattern that persists across T-staging image formats (Table A11) and on an external 53-patient cohort from a second center (Table A12).

A 100-case anatomical panel tests whether stage predictions rest on valid visual evidence. Direct T-stage accuracy reaches 43.0%, yet complete 18-structure invasion and evidence-grounded T-stage success (requiring correct labels and final stage) both fall below 5.0% (Table A15; Fig. 7a). Correct stage labels thus frequently coexist with flawed anatomical assessments. To isolate this breakdown, we test a fixed staging rule across three evidence conditions (Fig. 7f): reference findings (Ref.), predicted invasion with known anatomy (Known), and fully predicted evidence (Pred.). With reference findings, both a deterministic rule and a fixed GPT-OSS-120B stager reach 99.0% accuracy (Tables A17 and A18), confirming that downstream staging logic is sound. However, the accuracy of Gemini-3.1 drops to 49.0% (Known) and 36.0% (Pred.). Even with correct anatomical localization, flawed invasion recognition remains the primary bottleneck. A separate 25-case static vs. cine MRI comparison yields an inconclusive T-stage difference (section C.4).

Table 1: Performance across NPCBench tasks. Accuracy and success rates are percentages; C-Rub. is a 0–100 score under GPT-OSS-120B (103 consultations). A-App./C-Plan use group/patient macro averages. T-Stg./N-Stg. each cover 400 cases. L-Rev. reports Turn accuracy and All-turn success; CP-Epi. reports Care accuracy and Strict joint pathology/T/care success (100 patients, 217 care decisions). Bold/underline mark the two highest distinct displayed values. †: models used in CarePath construction. –: unavailable interface.
<table><tr><td colspan="3"></td><td colspan="3">MM-EG</td><td rowspan="2"></td><td colspan="3">L-Rev.</td><td colspan="3">CP-Epi.</td></tr><tr><td>Model</td><td>G-Mem. A-App. Path. T-Stg.</td><td></td><td></td><td></td><td>N-Stg. Seq.</td><td>Struct.</td><td></td><td>C-Plan Turn All-turn</td><td></td><td></td><td>Care Strict C-Rub.</td></tr><tr><td colspan="10">MEDICAL-SPECIFIC LLMS</td><td></td><td></td><td></td><td></td></tr><tr><td>HuatuoGPT-o1-72B</td><td>88.4</td><td>73.9</td><td></td><td></td><td></td><td></td><td></td><td>46.9</td><td>63.9</td><td>26.0</td><td></td><td></td><td>10.1</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>77.2</td><td>52.9</td><td></td><td></td><td></td><td></td><td></td><td>43.2</td><td>55.1</td><td>20.0</td><td></td><td>一</td><td>5.3</td></tr><tr><td>Baichuan-M2-32B</td><td>22.8</td><td>2.4</td><td></td><td></td><td></td><td></td><td></td><td>5.3</td><td>66.2</td><td>31.0</td><td></td><td>一</td><td>38.8</td></tr><tr><td>Fleming-R1-32B</td><td>85.6</td><td>37.6</td><td>一</td><td></td><td></td><td></td><td></td><td>30.3</td><td>75.7</td><td>40.0</td><td>一</td><td>一</td><td>28.6</td></tr><tr><td>Fleming-R1-7B</td><td>84.4</td><td>60.6</td><td></td><td></td><td></td><td></td><td></td><td>37.2</td><td>59.3</td><td>23.0</td><td></td><td></td><td>24.8</td></tr><tr><td colspan="10">MEDICAL-SPECIFIC MLLMs</td><td colspan="3"></td></tr><tr><td>HealthGPT-Pro-4B</td><td>80.0</td><td>59.0</td><td>56.5</td><td>38.5</td><td>20.0</td><td>54.0</td><td>9.4</td><td>50.8</td><td>26.2</td><td>4.0</td><td>19.8</td><td>0.0</td><td>14.7</td></tr><tr><td>HealthGPT-Pro-8B†</td><td>81.2</td><td>61.2</td><td>56.5</td><td>39.5</td><td>20.3</td><td>50.0</td><td>8.6</td><td>48.7</td><td>38.7</td><td>9.0</td><td>20.3</td><td>0.0</td><td>18.8</td></tr><tr><td>Hulu-Med-4B</td><td>78.0</td><td>56.9</td><td>56.5</td><td>34.5</td><td>19.8</td><td>30.0</td><td>5.6</td><td>53.1</td><td>59.3</td><td>26.0</td><td>39.6</td><td>0.0</td><td>8.7</td></tr><tr><td>Hulu-Med-7B</td><td>82.4</td><td>56.8</td><td>56.5</td><td>12.5</td><td>20.5</td><td>58.0</td><td>7.5</td><td>48.3</td><td>47.2</td><td>16.0</td><td>39.6</td><td>0.0</td><td>10.7</td></tr><tr><td>Hulu-Med-32B</td><td>86.0</td><td>74.5</td><td>60.9</td><td>29.0</td><td>20.0</td><td>75.5</td><td>21.8</td><td>53.4</td><td>70.2</td><td>37.0</td><td>47.5</td><td>10.0</td><td>7.6</td></tr><tr><td>Lingshu-7B</td><td>78.4</td><td>60.3</td><td>56.5</td><td>37.5</td><td>3.8</td><td>47.0</td><td>9.3</td><td>42.6</td><td>49.2</td><td>14.0</td><td>44.2</td><td>9.0</td><td>3.2</td></tr><tr><td>Lingshu-32B</td><td>85.6</td><td>61.0</td><td>59.4</td><td>37.5</td><td>20.0</td><td>75.0</td><td>11.4</td><td>53.3</td><td>70.5</td><td>37.0</td><td>55.3</td><td>2.0</td><td>9.0</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-30B-A3B</td><td>83.2</td><td>66.2</td><td>一</td><td></td><td></td><td>OPEN-SOURCE LLMs</td><td></td><td>51.3</td><td>63.3</td><td>27.0</td><td>一</td><td></td><td>26.8</td></tr><tr><td>GPT-OSS-20B</td><td>74.4</td><td>58.3</td><td>一</td><td></td><td>一</td><td></td><td></td><td>32.1</td><td>48.9</td><td>13.0</td><td>一</td><td>一</td><td>28.0</td></tr><tr><td>GPT-OSS-120B</td><td>86.0</td><td>75.4</td><td></td><td></td><td></td><td></td><td></td><td>49.4</td><td>60.3</td><td>23.0</td><td></td><td></td><td>44.9</td></tr><tr><td>DeepSeek-V3.1</td><td>90.0</td><td>81.0</td><td></td><td></td><td></td><td></td><td></td><td>51.8</td><td>74.4</td><td>42.0</td><td></td><td>1</td><td>28.2</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-8B-Instruct†</td><td>83.6</td><td>61.5</td><td>56.5</td><td>37.5</td><td>20.0</td><td>OPEN-SOURCE MLLMS 29.5</td><td>4.7</td><td>52.3</td><td>69.8</td><td>38.0</td><td>52.5</td><td>2.0</td><td>25.6</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct†</td><td>86.0</td><td>67.3</td><td>56.5</td><td>37.5</td><td>21.8</td><td>40.0</td><td>36.2</td><td>52.3</td><td>68.5</td><td>36.0</td><td>47.5</td><td>6.0</td><td>29.1</td></tr><tr><td>Qwen3.5-9B</td><td>85.2</td><td>67.8</td><td>72.5</td><td>26.3</td><td>20.3</td><td>65.5</td><td>38.9</td><td>58.0</td><td>60.7</td><td>24.0</td><td>53.0</td><td>1.0</td><td>34.7</td></tr><tr><td>Qwen3.5-27B†</td><td>89.2</td><td>75.8</td><td>71.0</td><td>37.0</td><td>27.3</td><td>67.0</td><td>47.6</td><td>61.0</td><td>76.4</td><td>46.0</td><td>62.7</td><td>1.0</td><td>46.8</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>86.4</td><td>75.2</td><td>66.7</td><td>35.5</td><td>21.8</td><td>65.0</td><td>50.3</td><td>54.5</td><td>72.8</td><td>41.0</td><td>63.1</td><td>1.0</td><td>41.4</td></tr><tr><td>Qwen3.6-27B</td><td>87.6</td><td>71.4</td><td>71.0</td><td>37.5</td><td>42.8</td><td>75.5</td><td>41.4</td><td>50.3</td><td>79.7</td><td>53.0</td><td>65.9</td><td>5.0</td><td>47.2</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>85.2</td><td>67.9</td><td>66.7</td><td>43.5</td><td>20.3</td><td>61.5</td><td>44.8</td><td>46.8</td><td>76.1</td><td>49.0</td><td></td><td></td><td></td></tr><tr><td>InternVL3.5-38B</td><td>88.0</td><td>69.7</td><td>59.4 36.8</td><td>20.0</td><td>45.5</td><td></td><td>3.7</td><td>52.0</td><td>58.4</td><td>21.0</td><td>61.8 47.5</td><td>5.0 0.0</td><td>42.4 9.5</td></tr><tr><td colspan="10"></td><td colspan="3"></td><td></td></tr><tr><td></td><td>90.4</td><td></td><td></td><td></td><td>PROPRIETARY (M)LLMS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>58.4</td></tr><tr><td>GPT-5.1 GPT-4.1</td><td>90.0</td><td>80.9 77.6</td><td>62.3 71.0</td><td>27.3 39.8</td><td>32.8 22.0</td><td>47.0 48.5</td><td>26.3 28.3</td><td>63.5 54.4</td><td>75.7 78.7</td><td>45.0 45.0</td><td>71.0 61.8</td><td>2.0 2.0</td></table>

Longitudinal decision accuracy does not ensure patient-level success. For RQ3, we first examine L5 performance across 305 decisions from 100 patients. Gemini-3.1-Pro-Preview achieves the highest turn accuracy (87.9%) and all-turn patient success (71.0%), followed by Claude-Sonnet-4.6 (85.2% and 63.0%, respectively; Fig. 4b). While turn accuracy measures individual decisions in isolation, all-turn success is substantially lower across models because it depends on how errors distribute across trajectories—any single misstep invalidates an entire patient episode.

![](images/b52c752c95763b7e6d5c6d163e0059f56b42b1cf2ee92047c34f2fd1e41e1329.jpg)  
Figure 4: Staging, longitudinal decisions, and patient-level correctness. (a), L3 T/N accuracy on separate 400-patient cohorts. (b), L5 decision accuracy (Turn) and all-turn success: 305 decisions, 100 patients. (c), L6 Care accuracy and All-care success: 217 decisions, 100 patients. (d), Pathology (Path.) and MRI T accuracy; majority-class baselines are 99% and 58%, respectively. (e), Joint pathology/T correctness (Path.+T), All-care, and their intersection (Strict). All-turn and All-care require every decision correct. Panels (c–e) use the same 100 patients; gray curves show other models.

In paired controlled runs across the same 305 questions and 100 patients, we compare full chronological history against a snapshot that strips time markers and regroups the same records by content (Table A25). Full history yields higher turn-accuracy point estimates for 14 of 20 models, yet the patient-bootstrap 95% confidence intervals span zero for 19 models; DeepSeek-V3.1 gains 4.9 pp and Gemini-3.1 gains 3.3 pp. Net gains can hide offsetting errors: GPT-5.1 is correct on 22 questions only with full history and 19 only with the snapshot. Thus a 1.0 pp turn-accuracy gain coexists with a 1.0 pp decline in all-turn success. Three of 20 models show opposite directions for these endpoints (Fig. 7b,g). Correcting one turn may leave other errors, whereas a new error can invalidate an otherwise correct episode.

To separate record order from temporal information, an order-only control shuffles complete lines while retaining all time markers. Nineteen of 20 models have confidence intervals spanning zero, with Gemini-3.1 as the sole exception (+3.0 pp; 95% CI: 0.6–5.7 pp). Tables A25 and A26 report the paired results. A separate conversation-retention comparison involving 87 patients also yields an interval spanning zero.

## 4.4 Consultation Quality and Physician Calibration

For RQ4, we assess whether consultations meet case requirements and avoid clinical errors. GPT-5.1 has the highest ConsultRubrics score with 58.4, although the primary GPT-OSS-120B judge identifies major and minor errors in 14.6% and 40.8% of its responses. Because sound clinical advice and unsafe recommendations frequently coexist within the same free-text answer, aggregate rubric scores must be interpreted alongside explicit error rates and topic breakdowns (Fig. 5a,c,d). Because the 103 consultations span 42 underlying cases, weighting the 42 cases equally rather than individual questions shifts model scores by only −2.0 to +0.1 points and leaves GPT-5.1 highest, as shown in Fig. 5b.

Table 2: Physician–judge agreement. Agreement (%) and Cohen’s κ are measured against physician annotations. Targeted Audit (n = 548) and Expanded Audit (n = 1,013) denote the two calibration sets; n counts criterion-level labels.
<table><tr><td rowspan="2">Judge</td><td colspan="2">Targeted AuditExpanded Audit  $( n = 5 4 8 )$ </td><td colspan="2"> $( n = 1 , 0 1 3 )$ </td></tr><tr><td>Agree.</td><td></td><td>κ Agree.</td><td>κ</td></tr><tr><td>GPT-4.1</td><td>77.7</td><td>0.553</td><td>77.6</td><td>0.547</td></tr><tr><td>GPT-OSS-120B</td><td>81.4</td><td>0.610</td><td>70.5</td><td>0.387</td></tr><tr><td>Qwen3-30B</td><td>73.2</td><td>0.472</td><td>76.9</td><td>0.538</td></tr><tr><td>Qwen3.6-27B</td><td>79.4</td><td>0.578</td><td>77.8</td><td>0.544</td></tr></table>

To assess judge reliability and sensitivity to calibration-sample composition, we conduct a Targeted Audit (548 criterion labels) and an Expanded Audit (1,013 labels from 54 responses across nine models on six cases; Table 2). GPT-OSS-120B achieves the highest agreement in the Targeted Audit (81.4%, κ = 0.610), whereas Qwen3.6-27B leads the Expanded Audit (77.8%, κ = 0.544), indicating differences in judge performance across calibration samples.

In the Expanded Audit, the primary GPT-OSS-120B judge agrees with physicians on 70.5% of criteria (case-resampled 95% CI: 66.2–77.7%), with 51.8% sensitivity and 85.8% specificity (Fig. A7). A positive label means that a criterion holds, including the presence of an error. Overlooking a satisfied requirement withholds credit, whereas overlooking an error avoids a deduction. Figure 7e shows recognition of physician-positive labels (sensitivity) and negative labels (specificity); panel j compares agreement by criterion type.

Gemini-3.1 GPT-5.1  
![](images/fa33f2c253edd52cde326c209727c720a8f75a29194278c284bfb66fe448cdad.jpg)

![](images/e4596c5448c33f227a6cec5fd3be4084f94944c5a4250268f84aac1ed42977e1.jpg)

![](images/a72fb2dbfca67867af1d8a3a2a626923fd7ec747090b2ff0a1af0904801cbcaa.jpg)

![](images/876926be93c70c9d7b52ef51c98857b51028607d17daaae8b1b49694062c4680.jpg)  
Figure 5: Consultation quality under the GPT-OSS-120B judge. (a), Major-error rate and C-Rub. score; major errors contribute to score deductions. (b), Item-mean and case-mean scores for the same 103 consultations from 42 cases. (c), Clinical tags: adverse events (AE), diagnosis (Dx), elderly (Eld), pediatric (Ped), treatment (Tx), and workup (Wk), with two to 43 questions per tag. (d), Criterion-level rates: core answers, supporting evidence, and conditional requirements satisfied; major and minor error criteria not triggered. These error-avoidance rates differ from the response-level major-error rate in (a).

## 4.5 Integrating Evidence and Decisions within Patients

For RQ3, all-care correctness is $\begin{array} { r } { C _ { p } = \prod _ { j \in S _ { p } } c _ { p j } } \end{array}$ and joint pathology–MRI T-stage correctness is $F _ { p } ~ = ~ d _ { p } t _ { p }$ (Eq. (1)). In the CarePath cohort, Gemini-3.1 achieves the highest Care accuracy (195/217 decisions, 89.9%) and All-care success (83.0%), yet its Strict success is only 11.0%. In contrast, Claude-4.6 achieves lower Care (80.6%) and All-care (64.0%) but the highest Strict success (23.0%; exact 95% CI: 15.2–32.5%; Fig. 4c–e). Thus care accuracy alone does not identify which models most often answer all diagnostic and care questions correctly for a patient.

Joint correctness identifies the patients behind this gap. For Gemini-3.1, 72 of 100 patients have all care decisions correct despite at least one incorrect pathology or MRI T-stage label; the corresponding number is 41 for Claude-4.6. Conversely, Hulu-Med-32B has 38 patients with both diagnostic labels correct but at least one incorrect care decision (Fig. 6; Table A20). Care answers and diagnostic labels can therefore disagree in correctness.

To further examine whether diagnostic evidence affects care decisions, we compare predicted and reference pathology, MRI T-stage, and invasion findings in 25 CarePath patients. Fourteen local models answer the same 146 care questions under both conditions, with other clinical context fixed. Patient-

![](images/4c116e2e268f0f2ba1a60b3b1885a2055b7e6515c72a8164a368e886bfdce049.jpg)

(b) Diagnostic–care agreement  
![](images/43ad0d4a87f3c8ecf222c8c08c63aacbda650350380a8bbe8924d19a12ba1e10.jpg)  
Figure 6: Diagnostic and care correctness in 100 patients. (a), All-care versus Strict for 21 models; diamonds mark the four construction models. (b), Joint outcomes for four models. Care means all care answers are correct; Image requires correct pathology and T labels; Both equals Strict.

averaged care accuracy, excluding diagnostic scores, changes by $- 3 . 2 \ \mathrm { t o \ + 4 . 1 \ p p }$ , with 12 of 14 intervals including zero (Fig. 7d; Table A23). These mixed changes show no consistent care benefit from reference evidence under the supplied clinical context.

Care and All-care outcomes also differ across episode groups. Nine of 21 models have lower decision-weighted Care but higher patient-weighted All-care in the 75 assigned-pathology episodes

(c) L5 by depth

Gemini-3.1 GPT-5.1 Claude-4.6 Qwen3.6-27B Other models than in the 25 patient-matched episodes. Claude-4.6 scores 76.5% versus 84.3% for Care, but 68.0% versus 52.0% for All-care (Table A21). The groups contribute 102 and 115 decisions, respectively. Figure 7c,h further examines performance by decision count: 1–2/3/4–5 for L5 and 1/2/3–6 for CarePath, as patient-level success requires every scored decision to be correct.

Class imbalance also matters: constant T3 predictions score 58% in CarePath, above the best model’s 52%. The highest balanced T accuracy is 30.5%, versus 25% for constant predictions (Table A19; Fig. A4). L3 therefore complements CarePath’s joint endpoint with stage-discrimination and grounding tests.

(g) Turn vs patient  
![](images/8bef59b70850e0c4427f53f80da45e1995facdd1e9745e94c556d6fe8679fb47.jpg)

![](images/d44cc80f79107abacd79e386856c2d146e0fc204ea654dd941509bf7b91bc253.jpg)

![](images/c4097800b66c149dd5d55cc18a1e54c28b83214520c7317733fa544922625747.jpg)

![](images/6a42ad9e315e9a88e13ed192b777aebf5d5e0075d5f1abce3a197e713b159be0.jpg)

(e) Judge Se/Sp  
![](images/f1c366b2d33e4402de72ed6ce2ccccbfd41b607f4ba2aaeff1713abaaa90da94.jpg)

(f) Evidence to T  
![](images/36f272a32157ba1d630e6138f9c1ace980fe33fc7e5352f7caaa7f382dc10a6f.jpg)

![](images/f07b50d7bb64ab5dc66cb1f9d3b14231fc597c1cea4220b29af7b570068c2269.jpg)

![](images/21a51212d93d695a8aaf98778913ead9d55f632605b2c64350d19660448ec209.jpg)

(i) Answer format  
![](images/ed400f0d3d27183df89ca6b35331848140569dae6c2737184d8fe30a9cad19ad.jpg)

(j) Agreement  
![](images/0d3cf0d443a6eef8dba68a23f2ea31f97d9b7d5ae771b237f4e477d2901c3a73.jpg)  
Figure 7: Evidence, decision changes, and evaluation conditions. (a),(f), Evidence-grounded T success (Grounded T) and fixed-rule staging on 100 patients. (b),(g), Corrections, regressions, and gains from snapshot to full history. (c),(h), L5 all-turn and L6 Strict success by decision count (n: patients). (d), Care change with reference pathology, MRI T stage, and invasion findings. (i), MCQ versus short-answer accuracy on 71 matched questions; gray denotes GPT-4.1. (e),(j), Judge sensitivity/specificity (Se/Sp) and agreement on requirements (Req.), major errors (Maj.), and minor errors (Min.). Points represent models or judges; OSS-120B is GPT-OSS-120B. Changes are percentage points (pp); bars show 95% patient-cluster (d) or response-cluster (e,j) intervals.

## 5 Conclusion

NPCBench links foundational guideline knowledge and multimodal image interpretation to longitudinal decision-making, joint patient-level correctness, and specialist consultation in NPC care. Evaluating 30 models reveals that local accuracy rarely guarantees coherent patient management: stage predictions frequently lack complete anatomical support, longitudinal history corrects prior mistakes while introducing new errors, and high-scoring consultations still trigger critical safety violations. On CarePath episodes, accurate treatment choices routinely coexist with flawed image interpretations, capping strict patient-level success at just 23.0% despite substantially higher decisionlevel accuracy. These findings establish evidence–decision consistency as a central benchmark target: meaningful clinical progress requires models to ground interpretations in valid evidence and sustain accurate decisions across the entire course of care for the same patient.

Limitations. Our evaluation is constrained by primarily single-center records, assigned pathology in 75 model-selected CarePath episodes, score sensitivity to class balance and episode length, and the six-case Expanded Audit. Future more multi-center cohorts with fully patient-matched modalities and prospective studies are therefore needed to establish whether benchmark gains improve clinical care.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grants 82441027 and 82441026; the Fundamental Research Funds for Sun Yat-sen University under Grant PTQN202603; the Hong Kong RGC (TRS: T41-517/25-N; GRF: 15228325); ITC RAISe+ (RAI/24/1/086A); and the Research Institute for Generative Artificial Intelligence at PolyU. We also thank Deepexi Technology Co., Ltd. for its support.

Competing interests. The authors declare no competing interests.

## References

[1] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.

[2] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question answering with large language models. Nature medicine, 31(3):943–950, 2025.

[3] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36: 28541–28564, 2023.

[4] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. arXiv preprint arXiv:2506.07044, 2025.

[5] Zeyu Liu, Zhitian Hou, Guanghao Zhu, Zhijie Sang, Congkai Xie, and Hongxia Yang. Infimed: Low-resource medical mllms with advancing understanding and reasoning. arXiv preprint arXiv:2505.23867, 2025.

[6] Pengkai Wang, Pengwei Liu, Qi Zuo, Zhijie Sang, Congkai Xie, and Hongxia Yang. Infimedorbit: Aligning llms on open-ended complex tasks via rubric-based incremental training. arXiv preprint arXiv:2510.15859, 2025.

[7] Ahmed Alaa, Thomas Hartvigsen, Niloufar Golchini, Shiladitya Dutta, Frances Dean, Inioluwa Deborah Raji, and Travis Zack. Position: Medical large language model benchmarks should prioritize construct validity. In Forty-second International Conference on Machine Learning Position Paper Track, 2025.

[8] Suhana Bedi, Hejie Cui, Miguel Fuentes, Alyssa Unell, Michael Wornow, Juan M Banda, Nikesh Kotecha, Timothy Keyes, Yifan Mai, Mert Oez, et al. Holistic evaluation of large language models for medical tasks with medhelm. Nature Medicine, pages 1–9, 2026.

[9] Wei-Wei Zhang, Pengkai Wang, Yan Li, Zhijie Sang, Guan-Qun Zhou, Jun-Yan Li, Shu-Han Zhou, Pei-Wen Liu, Ting-Xi Tang, Min Tang, et al. Patient-specific rubric-based evaluation of large language models across longitudinal cancer care. arXiv preprint arXiv:2610.35847, 2026.

[10] Freddie Bray, Mathieu Laversanne, Hyuna Sung, Jacques Ferlay, Rebecca L Siegel, Isabelle Soerjomataram, and Ahmedin Jemal. Global cancer statistics 2022: Globocan estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: a cancerjournalfor clinicians, 74(3):229–263, 2024.

[11] Yu-Pei Chen, Anthony TC Chan, Quynh-Thu Le, Pierre Blanchard, Ying Sun, and Jun Ma. Nasopharyngeal carcinoma. The Lancet, 394(10192):64–80, 2019.

[12] Kenneth CW Wong, Edwin P Hui, Kwok-Wai Lo, Wai Kei Jacky Lam, David Johnson, Lili Li, Qian Tao, Kwan Chee Allen Chan, Ka-Fai To, Ann D King, et al. Nasopharyngeal carcinoma: an evolving paradigm. Nature reviews Clinical oncology, 18(11):679–695, 2021.

[13] Jian-Ji Pan, Hai-Qiang Mai, Wai Tong Ng, Chao-Su Hu, Jin-Gao Li, Xiao-Zhong Chen, James CH Chow, Edwin Wong, Victor Lee, Ling-Yu Ma, et al. Ninth version of the ajcc and uicc nasopharyngeal cancer tnm staging classification. JAMA oncology, 10(12):1627–1635, 2024.

[14] NCCN Clinical Practice Guidelines in Oncology: Head and Neck Cancers. National Comprehensive Cancer Network, 2025. URL https://www.nccn.org/guidelines.

[15] Chinese Society of Clinical Oncology. CSCO Guidelines for Diagnosis and Treatment of Nasopharyngeal Carcinoma. People’s Medical Publishing House, Beijing, 2024.

[16] KC Allen Chan, John KS Woo, Ann King, Benny CY Zee, WK Jacky Lam, Stephen L Chan, Sam WI Chu, Constance Mak, Irene OL Tse, Samantha YM Leung, et al. Analysis of plasma epstein–barr virus dna to screen for nasopharyngeal cancer. New England Journal of Medicine, 377(6):513–522, 2017.

[17] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, 2021.

[18] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, pages 248–260. PMLR, 2022.

[19] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 2567–2577, 2019.

[20] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

[21] Yan Cai, Linlin Wang, Ye Wang, Gerard De Melo, Ya Zhang, Yanfeng Wang, and Liang He. Medbench: A large-scale chinese benchmark for evaluating medical large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 17709–17717, 2024.

[22] Xidong Wang, Guiming Chen, Song Dingjie, Zhang Zhiyi, Zhihong Chen, Qingying Xiao, Junying Chen, Feng Jiang, Jianquan Li, Xiang Wan, et al. Cmb: A comprehensive medical benchmark in chinese. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6184–6205, 2024.

[23] Junling Liu, Peilin Zhou, Yining Hua, Dading Chong, Zhongyu Tian, Andrew Liu, Helin Wang, Chenyu You, Zhenhua Guo, Lei Zhu, et al. Benchmarking large language models on cmexam-a comprehensive chinese medical exam dataset. Advances in Neural Information Processing Systems, 36:52430–52452, 2023.

[24] Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. Medxpertqa: Benchmarking expert-level medical reasoning and understanding. arXiv preprint arXiv:2501.18362, 2025.

[25] Xiang Feng, Wentao Jiang, Zengmao Wang, Yong Luo, Pingbo Xu, Baosheng Yu, Hua Jin, and Jing Zhang. Anessuite: A comprehensive benchmark and dataset suite for anesthesiology reasoning in llms. arXiv preprint arXiv:2504.02404, 2025.

[26] Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific data, 5(1): 180251, 2018.

[27] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semanticallylabeled knowledge-enhanced dataset for medical visual question answering. In 2021 IEEE 18th international symposium on biomedical imaging (ISBI), pages 1650–1654. IEEE, 2021.

[28] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020.

[29] Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. Omnimed vqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22170–22183, 2024.

[30] Shengyuan Liu, Boyun Zheng, Wenting Chen, Zhihao Peng, Zhenfei Yin, Jing Shao, Jiancong Hu, and Yixuan Yuan. EndoBench: A comprehensive evaluation of multi-modal large language models for endoscopy analysis. arXiv preprint arXiv:2505.23601, 2025. URL https://arxiv. org/abs/2505.23601.

[31] Jingkun Yue, Siqi Zhang, Zinan Jia, Huihuan Xu, Zongbo Han, Xiaohong Liu, and Guangyu Wang. Medsg-bench: A benchmark for medical image sequences grounding. arXiv preprint arXiv:2505.11852, 2025.

[32] Tianhong Zhou, Yin Xu, Yingtao Zhu, Chuxi Xiao, Haiyang Bian, Lei Wei, and Xuegong Zhang. Drvd-bench: Do vision-language models reason like human doctors in medical image diagnosis? arXiv preprint arXiv:2505.24173, 2025.

[33] Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, et al. Cares: A comprehensive benchmark of trustworthiness in medical vision language models. Advances in Neural Information Processing Systems, 37: 140334–140365, 2024.

[34] Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

[35] Zhiling Yan, Dingjie Song, Zhe Fang, Yisheng Ji, Xiang Li, Quanzheng Li, and Lichao Sun. Livemedbench: A contamination-free medical benchmark for llms with automated rubric evaluation. arXiv preprint arXiv:2602.10367, 2026.

[36] Ming Zhang, Yujiong Shen, Zelin Li, Huayu Sha, Binze Hu, Yuhui Wang, Chenhao Huang, Shichun Liu, Jingqi Tong, Changhao Jiang, et al. Llmeval-med: a real-world clinical benchmark for medical llms with physician validation. arXiv preprint arXiv:2506.04078, 2025.

[37] Lecheng Gong, Weimin Fang, Ting Yang, Dongjie Tao, Chunxiao Guo, Peng Wei, Bo Xie, Jinqun Guan, Zixiao Chen, Fang Shi, et al. Meddialogrubrics: A comprehensive benchmark and evaluation framework for multi-turn medical consultations in large language models. arXiv preprint arXiv:2601.03023, 2026.

[38] Xiang Zheng, Han Li, Wenjie Luo, Weiqi Zhai, Yiyuan Li, Chuanmiao Yan, Tianyi Tang, Yubo Ma, Kexin Yang, Dayiheng Liu, et al. Clinconsensus: A consensus-based benchmark for evaluating chinese medical llms across difficulty levels. arXiv preprint arXiv:2603.02097v1, 2026. URL https://arxiv.org/abs/2603.02097v1.

[39] Xian Wu, Yutian Zhao, Yunyan Zhang, Jiageng Wu, Zhihong Zhu, Yingying Zhang, Yi Ouyang, Ziheng Zhang, Huimin Wang, Zhenxi Lin, et al. Medjourney: Benchmark and evaluation of large language models over patient clinical journey. Advances in Neural Information Processing Systems, 37:87621–87646, 2024.

[40] Yixing Jiang, Kameron C Black, Gloria Geng, Danny Park, James Zou, Andrew Y Ng, and Jonathan H Chen. MedAgentBench: a virtual EHR environment to benchmark medical LLM agents. Nejm Ai, 2(9):AIdbp2500144, 2025.

[41] Yinghao Zhu, Ziyi He, Haoran Hu, Xiaochen Zheng, Xichen Zhang, Zixiang Wang, Junyi Gao, Liantao Ma, and Lequan Yu. Medagentboard: Benchmarking multi-agent collaboration with conventional methods for diverse medical tasks. arXiv preprint arXiv:2505.12371, 2025.

[42] Kiril Vasilev, Alexandre Misrahi, Eeshaan Jain, Phil F Cheng, Petros Liakopoulos, Olivier Michielin, Michael Moor, and Charlotte Bunne. Mtbbench: A multimodal sequential clinical decision-making benchmark in oncology. arXiv preprint arXiv:2511.20490, 2025.

[43] Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. Huatuogpt-o1, towards medical complex reasoning with llms. arXiv preprint arXiv:2412.18925, 2024.

[44] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[45] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

[46] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[47] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[48] Pengcheng Chen, Jin Ye, Guoan Wang, Yanjun Li, Zhongying Deng, Wei Li, Tianbin Li, Haodong Duan, Ziyan Huang, Yanzhou Su, et al. Gmai-mmbench: A comprehensive multimodal evaluation benchmark towards general medical ai. Advances in Neural Information Processing Systems, 37:94327–94427, 2024.

[49] Yang Zhou, Tan L Faith, Yanyu Xu, Sicong Leng, Xinxing Xu, Yong Liu, and Rick S Goh. Benchx: A unified benchmark framework for medical vision-language pretraining on chest x-rays. Advances in Neural Information Processing Systems, 37:6625–6647, 2024.

[50] Yuli Wang, Jian Peng, Yuwei Dai, Craig Jones, Haris Sair, Jinglai Shen, Nicolas Loizou, Jing Wu, Wen-Chi Hsu, Maliha Imami, et al. Enhancing vision-language models for medical imaging: bridging the 3d gap with innovative slice selection. Advances in Neural Information Processing Systems, 37:99947–99964, 2024.

[51] Jing Hao, Yuxuan Fan, Yanpeng Sun, Kaixin Guo, Lizhuo Lin, Jinrong Yang, Qi Yong H Ai, Lun M Wong, Hao Tang, and Kuo Feng Hung. Towards better dental ai: A multimodal benchmark and instruction dataset for panoramic x-ray analysis. arXiv preprint arXiv:2509.09254, 2025.

[52] Juntao Jiang, Jiangning Zhang, Yali Bi, Jinsheng Bai, Weixuan Liu, Weiwei Jin, Zhucun Xue, Yong Liu, Xiaobin Hu, and Shuicheng Yan. M3cotbench: Benchmark chain-of-thought of mllms in medical image understanding. arXiv preprint arXiv:2601.08758, 2026.

[53] Zonghai Yao, Benlu Wang, Yifan Zhang, Junda Wang, Iris Xia, Zhipeng Tang, Shuo Han, Feiyun Ouyang, Zhichao Yang, Arman Cohan, et al. Medical thinking with multiple images. arXiv preprint arXiv:2604.16506, 2026.

[54] Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming Li, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 599–621, 2024.

[55] Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik S Chan, Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghassemi, Cynthia Breazeal, and Hae W Park. Mdagents: An adaptive collaboration of llms for medical decision-making. Advances in Neural Information Processing Systems, 37:79410–79452, 2024.

[56] Xiangru Tang, Daniel Shao, Jiwoong Sohn, Jiapeng Chen, Jiayi Zhang, Jinyu Xiang, Fang Wu, Yilun Zhao, Chenglin Wu, Wenqi Shi, et al. Medagentsbench: Benchmarking thinking models and agent frameworks for complex medical reasoning. arXiv preprint arXiv:2503.07459, 2025.

[57] Dyke Ferber, Omar SM El Nahhas, Georg Wölflein, Isabella C Wiest, Jan Clusmann, Marie-Elisabeth Leßmann, Sebastian Foersch, Jacqueline Lammert, Maximilian Tschochohei, Dirk Jäger, et al. Development and validation of an autonomous artificial intelligence agent for clinical decision-making in oncology. Nature cancer, 6(8):1337–1349, 2025.

[58] Samuel Schmidgall, Rojin Ziaei, Carl Harris, Ji Woong Kim, Eduardo Pontes Reis, Jeffrey Jopling, and Michael Moor. AgentClinic: a multimodal benchmark for tool-using clinical AI agents. npj Digital Medicine, 2026.

[59] Luyang Luo, Sung Eun Kim, Xiaoman Zhang, Julius M Kernbach, Roshan Kenia, Julian N Acosta, Larry A Nathanson, Adrian D Haimovich, Adam Rodman, Ethan Goh, et al. A clinical environment simulator for dynamic ai evaluation. Nature medicine, pages 1–8, 2026.

## Appendix Contents

A Data Curation and Statistics 16   
A.1 Task scope and composition 17   
A.2 MRI cohorts and patient coverage 18   
A.3 CarePath episodes and consultation criteria . 18   
B Detailed Related Work 19   
C Evaluation Protocol and Additional Results . 20   
C.1 Evaluation models 21   
C.2 Task definitions and scoring 22   
C.3 Extended results and diagnostics 24   
C.3.1 Longitudinal decisions 25   
C.3.2 MRI T and N staging 27   
C.3.3 Anatomical evidence 27   
C.3.4 Staging ablations 30   
C.3.5 CarePath patient outcome 32   
C.4 Controlled input comparisons 35   
C.5 Consultation scoring and results 39   
D Clinical Review and Judge Calibration 41   
E Evaluation Prompts 44   
F Illustrated Task Inputs 49

## A Data Curation and Statistics

The following sections describe the clinical sources, question construction, patient coverage, and reference annotations.

Clinical data and ethics. The retrospective use of clinical data was approved under IRB protocol B2026-465-01. Patient materials follow the hospital’s de-identification procedure. Direct identifiers and unnecessary location details are removed, and dates are expressed relative to the patient’s clinical course where appropriate. L4 retains clinical decision dates needed to identify the applicable guideline edition. Age and sex are retained when needed for reasoning. NPC oncology specialists review clinical validity, reference answers, supporting evidence, and transitions between care decisions.

Task construction. L1 and L2 translate guideline knowledge and decision thresholds into questions with adjudicated answers. L4 combines diagnosis, stage, treatment history, response, toxicity, and patient-specific constraints into fixed-state planning cases. L5 organizes longitudinal records around clinical decision checkpoints, retaining the cumulative evidence available at each point and removing subsequent orders or outcomes that reveal the target decision. L6 links pathology, MRI, and selected care decisions within patient episodes. O1 uses consultation scenarios with separate criteria for core conclusions, supporting evidence, conditional reasoning, and clinically important errors. L1 retains item-level guideline references and supporting evidence. All 350 L2/L4 items include their construction sources and reviewed answer rationales. Source records distinguish the guideline edition stated in a question from references supporting its answer: the former is available to the model, whereas the latter explains the reference decision. Recorded edition constraints occur in two L2 and four L4 questions and are included in the model inputs; edition and section references are available for a subset of items.

Clinical review and answer format. The evaluated sets contain 250 L1, 185 L2, and 165 L4 questions. Reviewers checked the evidence available at the decision point, the applicable guideline, and whether the options supported a determinate answer. L2 includes one explicit multiple-select item and L4 includes two; the other items are single choice. Multiple-select answers must match the complete reference set. The question specifies this requirement, and selecting only part of the set receives no credit. Reviewed rationales accompany all 350 L2/L4 items.

Construction examples. Figure 3 shows shortened examples of the released task formats. The examples retain the answer labels and decision-relevant information; full inputs and reviewed rationales accompany the dataset.

Data and code availability. De-identified benchmark inputs, multimodal assets, reference annotations, and supplementary control datasets are publicly available on Hugging Face; evaluation code and inference scripts are available on GitHub. The released tasks include the CarePath cohort with 100 patients and 217 scored care decisions. Author-owned code is licensed under MIT and original data and annotations under CC BY 4.0; third-party materials retain their own terms.

## A.1 Task Scope and Composition

Table A1: Benchmark scope, data handling, and intended use.
<table><tr><td>Item</td><td>Description</td></tr><tr><td>Measurement target</td><td>Guideline application, image interpretation, selected care decisions, and consultation quality in NPC; component scores and joint patient correctness are reported separately.</td></tr><tr><td>Intended use</td><td>Research benchmarking, model comparison, clinical error analysis, and evaluation-method development.</td></tr><tr><td>Out-of-scope use</td><td>Direct diagnosis, treatment recommendation, triage, or patient-specific clinical decision-making without physician supervision.</td></tr><tr><td>Data boundary</td><td>Guideline-derived questions, expert-authored vignettes, de-identified patient records, pathology/MRI evidence, longitudinal checkpoints, selected CarePath decisions, and consult rubrics.</td></tr><tr><td>Privacy handling</td><td>Names, contact details, addresses, and institution-specific identifiers are removed. Dates use relative clinical time where appropriate; L4 retains dates needed to determine the applicable guideline edition. Clinically relevant age, sex, and time intervals are preserved.</td></tr><tr><td>Evaluation controls</td><td>Shared prompts, adjudicated reference answers, final-answer extraction, O1 judge comparisons, and specialist review of answer clarity.</td></tr><tr><td>Known limitations</td><td>Disease-specific retrospective data, 2D MRI cards, assigned pathology specimens in 75 CarePath episodes, variable scoring depth, and dependence on the automated consultation judge.</td></tr></table>

Evaluation cohorts. The L6 cohort contains 100 patients and 417 scored tasks: 100 pathology assessments, 100 MRI T-stage assessments, and 217 care decisions. Image-evidence replacement and serial-interaction analyses use 146 care questions for 25 patients within this cohort, as described in section C.4. These analyses use a fixed 146-question set, while the main benchmark scores 217 selected care decisions across all 100 patients.

The L1 guideline-memory set covers CSCO-only, NCCN-only, and dual-source questions. Among the 250 L1 items, 153 are supported by both CSCO and NCCN sources, 80 are CSCO-only, and 17 are NCCN-only. The L2 atomic-application set is organized into four capability families: initial definitive management (84 items), post-treatment recurrence (30), metastatic/systemic management (44), and toxicity or contraindication management (27).

Task categories. Figure 2 uses the four L2 scoring groups. O1 contains 43 treatment, 35 adverseevent, 11 diagnosis, 12 special-population, and two workup questions; the special-population group comprises four elderly and eight pediatric items. The figure and score breakdown therefore refer to the same item tags. The figure abbreviates recurrent/metastatic disease as Rec./met., recurrence as Recur., radiotherapy as RT, treatment as tx, progression as Progress., planning/execution as Plan/exec., metastatic disease as Metast., toxicity/supportive care as Tox./supp., and special populations as Special pop. EBV denotes Epstein–Barr virus; M0 denotes no distant metastasis.

The L4 case-level planning set covers initial treatment (16 items), disease progression (43), regimen switching (100), and toxicity management (6), with patient-level aggregation over 84 patients.

Table A2: Task scale, source materials, and reference annotation.
<table><tr><td>Subset</td><td>Evaluation unit</td><td>Scale</td><td>Source material</td><td>Reference annotation</td></tr><tr><td>L1</td><td>Guideline-memory MCQ</td><td>250 questions</td><td>NPC guidelines</td><td>Expert-validated guideline knowledge</td></tr><tr><td>L2</td><td>Short-vignette MCQ</td><td>185 questions</td><td>Clinical vignettes</td><td>Specialist-reviewed answers</td></tr><tr><td>L3</td><td>Pathology/MRI multimodal tasks</td><td>questions; 2,700 evaluated MRI cards; 400 T-stage cases; 400 N-stage cases; 200 MRI sequence</td><td>69 pathology overview Pathology and MRI images</td><td>Expert-validated pathology, imaging, and staging labels</td></tr><tr><td>L4</td><td>Case-level MCQ</td><td>questions 165 questions over 84 patients</td><td>Clinical vignettes</td><td>Specialist-reviewed plans</td></tr><tr><td>L5</td><td>Sequential patient-turn MCQ</td><td>100 real patients; 305 questions</td><td>De-identified longitudinal Physician-adjudicated records</td><td>decision turns</td></tr><tr><td>L6</td><td>Selected CarePath decisions</td><td>100 patients; 417 scored tasks (100 pathology, 100 MRI T,</td><td>Clinical episodes</td><td>Adjudicated care decisions; 75 pathology proxies</td></tr><tr><td>01</td><td>Open-ended consult rubric item</td><td>217 care) 103 questions across 42 Clinical consultation cases</td><td>scenarios</td><td>Structured scoring anchors</td></tr></table>

## A.2 MRI Cohorts and Patient Coverage

The T-stage cohort contains 400 patients, with T1/T2/T3/T4 counts of 50/50/150/150. The patientdisjoint N-stage cohort contains 400 patients, with N0/N1/N2/N3 counts of 80/160/80/80. Both use eighth-edition report-derived staging labels. The T cohort includes 374 M0, 14 M1, and 12 patients with unspecified M status; all 400 N-stage patients are M0. The 100-patient anatomical panel is contained within the T-stage cohort. The 200 sequence-recognition patients overlap the T and N cohorts in 67 and 59 patients, respectively, giving 874 unique patients across the main L3 MRI tasks. Pathology is counted separately as 69 specimens. Patient coverage is calculated from the union of the linked MRI records.

T-stage examinations span March 2021 to June 2024, as do N-stage examinations. The external MRI examinations were acquired at Nanfang Hospital from January 2023 to December 2024. Fifty-three examinations have a definite report-derived T label for scoring; two without a definitive reference are excluded. The external class counts are 4/8/31/10 for T1/T2/T3/T4.

L2 contains 47 clinical source cases and 138 constructed scenarios. L4, L5, L6, and O1 contain 84, 100, 100, and 42 task-level patient records, respectively. These task-level counts include cross-task overlap and therefore do not define a unique benchmark-wide patient total. Clinical records used in the EHR tasks span 2008–2025; these are source-record dates, not a common follow-up interval.

## A.3 CarePath Episodes and Consultation Criteria

Table A3 separates the two sources of CarePath cohort episodes. The corresponding clinical review is described in Appendix D; Table A4 reports the number of decisions per patient.

Table A3: Membership and evidence linkage within the CarePath cohort. All 100 patients have matched MRI and clinical records; the image-label comparison uses 146 fixed questions for the 25 patients with matched pathology.
<table><tr><td>Episode source</td><td>Patients</td><td>Care nodes</td><td>Pathology evidence</td><td>Treatment-selection coverage</td></tr><tr><td>Matched pathology</td><td>25</td><td></td><td>115 Patient-matched specimens</td><td>Includes selected initial-treatment decisions</td></tr><tr><td>Assigned pathology</td><td>75</td><td></td><td>102 Assigned specimen proxies</td><td>No scored initial-treatment-selection question</td></tr><tr><td>CarePath cohort</td><td>100</td><td></td><td>217 25 matched; 75 proxy</td><td>Six treatment items are excluded from scoring</td></tr></table>

Table A4: Scoring depth and the weight of each patient group. Depth is the number of scored decisions for one patient. Patient-level success weights patients equally; turn/node accuracy weights the scored decisions equally. L5 and the CarePath cohort are separate evaluation cohorts.
<table><tr><td rowspan="2">Depth</td><td colspan="2">L5</td><td colspan="3">CarePath cohort</td></tr><tr><td>Patients</td><td>Turns</td><td>Patients</td><td>Care nodes</td><td>Node share (%)</td></tr><tr><td>1</td><td>13</td><td>13</td><td>52</td><td>52</td><td>24.0</td></tr><tr><td>2</td><td>19</td><td>38</td><td>23</td><td>46</td><td>21.2</td></tr><tr><td>3</td><td>31</td><td>93</td><td>3</td><td>9</td><td>4.1</td></tr><tr><td>4</td><td>24</td><td>96</td><td>7</td><td>28</td><td>12.9</td></tr><tr><td>5</td><td>13</td><td>65</td><td>8</td><td>40</td><td>18.4</td></tr><tr><td>6</td><td>0</td><td>0</td><td>7</td><td>42</td><td>19.4</td></tr><tr><td>Total</td><td>100</td><td>305</td><td>100</td><td>217</td><td>100.0</td></tr></table>

Table A5: Consultation questions and rubric criteria by clinical group. Positive requirements comprise core answers, supporting evidence, and conditional recommendations. The 1,069 positive and 657 error criteria sum to 1,726.
<table><tr><td>Clinical group</td><td>Questions</td><td>Core</td><td>Evidence</td><td>Conditional</td><td>Major</td><td>Minor</td></tr><tr><td>Treatment</td><td>43</td><td>182</td><td>171</td><td>152</td><td>140</td><td>148</td></tr><tr><td>Adverse-event management</td><td>35</td><td>127</td><td>105</td><td>69</td><td>104</td><td>92</td></tr><tr><td>Diagnosis</td><td>11</td><td>31</td><td>51</td><td>22</td><td>52</td><td>37</td></tr><tr><td>Special populations</td><td>12</td><td>53</td><td>49</td><td>36</td><td>36</td><td>34</td></tr><tr><td>Workup</td><td>2</td><td>8</td><td>10</td><td>3</td><td>5</td><td>9</td></tr><tr><td>Total</td><td>103</td><td>401</td><td>386</td><td>282</td><td>337</td><td>320</td></tr></table>

## B Detailed Related Work

Static medical QA and broad medical suites. Examination-style benchmarks such as MedQA, MedMCQA, PubMedQA, the medical subsets of MMLU, CMB, and CMExam have provided a practical basis for assessing medical knowledge and localized reasoning skills [17–20, 22, 23]. MultiMedQA and MedBench extended this approach by combining multiple medical QA datasets and testing more general medical reasoning across them [2, 21]. More recent benchmarks—including MedHELM, MedXpertQA, and AnesSuite’s AnesBench—emphasize construct validity, specialtyspecific reasoning, and clinical abilities alongside exam-style performance [8, 24, 25]. NPCBench focuses on NPC and reports component performance and joint patient correctness separately.

Multimodal medical evidence grounding. Medical MLLM evaluation has progressed beyond image-question datasets like VQA-RAD, SLAKE, and PathVQA toward benchmarks with broader modality coverage and greater clinical specialization [26–28]. Frameworks such as OmniMedVQA, GMAI-MMBench, BenchX, CARES, and BrainMD assess dimensions including the breadth of medical imaging modalities, model reliability, and cross-modality generalization [29, 33, 48–50]. More domain-specific benchmarks—EndoBench, DrVD-Bench, MedSG-Bench, MMOral, M3CoT-Bench, and MedThinkVQA—target fine-grained image understanding, temporal or sequence-level grounding, and reasoning across multiple images [30–32, 51–53]. NPCBench examines these components in NPC, where pathology underpins diagnosis, MRI anatomy delineates local spread, sequence recognition shapes image interpretation, and assessment of local invasion guides staging and treatment intensity.

Rubric-based and real-world clinical response evaluation. Open-ended clinical evaluation is increasingly grounded in explicit scoring rubrics and physician-validated criteria. HealthBench assesses multi-turn health-related dialogues using item-specific rubrics; LiveMedBench focuses on temporally separated clinical cases; LLMEval-Med incorporates physician adjudication combined with checklist-guided evaluation; MedDialogRubrics and ClinConsensus extend rubric-based assess ment frameworks to consultation and longitudinal care scenarios [34–38]. O1 ConsultRubrics uses the same criterion-level approach for NPC consultation, in which positive requirements capture key clinical conclusions, supporting patient evidence, and conditions that change the recommendation, whereas negative criteria differentiate between major and minor safety-critical errors.

Workflow, longitudinal, and agentic clinical benchmarks. Workflow evaluations test information gathering, interaction, and decisions across multiple steps. MedJourney structures evaluation around the patient journey, whereas MedAgents, MDAgents, MedAgentsBench, and MedAgentBoard focus on collaborative problem-solving, tool integration, and multi-step clinical task execution [39, 41, 54– 56]. In oncology, MTBBench emulates multidisciplinary tumor-board decision-making, and recent work on autonomous oncology agents investigates specialized workflows for treatment decision support [42, 57]. AgentClinic evaluates tool use and information gathering in simulated clinical encounters, while MedAgentBench tests tasks in a FHIR-compatible electronic health record [40, 58]. Clinical-environment simulators additionally expose models to explicit state transitions [59]. NPCBench evaluates related questions within NPC: L5 tests decisions under different representations of longitudinal records, and L6 checks image labels and selected care answers for the same patient. Its primary care evaluation supplies clinical context without feeding previous care answers back, whereas separate serial controls test conversation retention. This distinction separates joint patient correctness from autonomous interaction with a clinical environment.

## C Evaluation Protocol and Additional Results

The protocol specifies each task’s inputs, model interface, and scoring rule. L1/L2/L4/L5 use closedchoice answers; L3 uses diagnostic labels and structured image findings; L6 combines pathology and T-stage labels with care decisions; and O1 uses criterion-level judgments of free-text consultations. Table A6 summarizes the inputs and scoring populations for the linked and matched analyses.

Table A6: Inputs and endpoints in the linked and matched evaluations. The CarePath cohort contains 417 scored tasks, including 217 care decisions. Image-evidence replacement and serialinteraction analyses evaluate 146 care questions for 25 patients within this cohort.
<table><tr><td>Evaluation</td><td>Scored units</td><td>Evidence and history</td><td>Endpoint</td></tr><tr><td>L3 direct staging</td><td>400 T-stage and 400 N-stage patients</td><td>Structure cards or same-plane sequence groups; AJCC/UICC edition 8</td><td>T and N accuracy; anatomical diagnostics on 100 T cases</td></tr><tr><td>CarePath cohort</td><td>100 patients; 217 care decisions</td><td>AJCC edition 9 MRI; predicted pathology and T-stage labels and supplied clinical context; no prior care answers</td><td>Decision-weighted Care; patient-level Strict</td></tr><tr><td>Image-evidence control</td><td>25 patients; 146 care decisions</td><td>Predicted versus reference image findings; other clinical context fixed</td><td>Paired change in patient-macro care accuracy</td></tr><tr><td>Serial control</td><td>25 patients; 146 care decisions</td><td>Reference image labels; persistent versus stateless interaction</td><td>Paired change in all-care correctness</td></tr><tr><td>L5 history control</td><td>100 patients; 305 turns</td><td>Full history versus a snapshot with historical time markers removed and records regrouped</td><td>Turn-accuracy change and all-turn correctness</td></tr></table>

Computing resources and generation settings. Local experiments used institutional NVIDIA A800-SXM4-80GB servers with up to eight GPUs; the multimodal experiments also used a four-GPU server. The local environment used CUDA 12.8, PyTorch 2.10.0, vLLM 0.19.1, Transformers 5.6.2, and Accelerate 1.13.0. Models were run through vLLM or model-specific multimodal interfaces, with parallelism selected according to model size. Task-specific settings specify the model interface, image processing, sampling parameters, and output limits. CarePath evaluation uses maximum output lengths of 1,024 tokens for local models and 8,192 for API models. Missing or incomplete final answers receive zero credit.

For L5 primary and paired history evaluations, GPT-OSS-20B allows 16,384 output tokens and HealthGPT-Pro-4B 32,768. Fleming-R1-7B allows up to 16,384 within its 32,768-token context; each matched history/snapshot/shuffled triplet uses the same output allowance.

O1 uses output limits of 2,048 tokens for the 23 local generators and 16,384 tokens for the seven API generators. Its primary judge uses temperature 0 and top-p 1, with a 2,048-token output limit and an 8,192-token retry for incomplete criterion judgments. All 30 models have valid judgments for all 1,726 criteria. The released evaluation code specifies the per-model settings.

Answer extraction. Closed-answer tasks use a common final-answer rule. An explicit final boxed or labelled answer takes precedence over options discussed in intermediate reasoning; a standalone option is accepted when unambiguous. Missing or conflicting final answers are scored incorrect. Multimodal tasks extract the requested diagnostic and anatomical fields. O1 averages all 103 itemlevel rubric scores for every model.

MRI inputs and generation. T staging uses two image presentations: 100 patients have 27 structure-centred cards per patient, and 300 have ordered groups of physically matched MRI planes. N staging uses the latter presentation for all 400 patients. Within an axial group, T1, T2, and enhanced T1 images are stacked vertically; a coronal group pairs enhanced fat-suppressed T1 with precontrast non-fat-suppressed T1. Each slice retains a 512 × 512 display tile. Axial and coronal groups are 512 × 1644 and 512 × 1096 lossless PNGs, respectively. A request contains one patient’s ordered groups, with 44–79 groups per case. The T range extends from the suprasellar cistern to inferior C3; the N range extends from the skull base to the inferior sternoclavicular joint. Clinicians reviewed the selected anatomical coverage and report labels.

Qwen, Lingshu, HealthGPT, and Hulu-Med-4B use a 128,000-token context and an 8,192-token output budget for the plane-group tasks. Hulu-Med-4B uses spatial merge 2 with a 98,304 visualtoken budget; Hulu-Med-7B/32B use merge 4, a 22,000 visual-token budget, and their 32,768-token context. Hulu images enter the processor as height–width–channel arrays. InternVL3.5 uses its 40,960-token context, dynamic cropping with at most four patches per group, and 33–35 uniformly sampled groups. Sampling allocates images proportionally between axial and coronal groups and retains both endpoints of each group.

API inputs preserve the same plane-group pixels: GPT-5.1 uses lossless WebP and the other services use lossless PNG. GPT-5.1 uses snapshot gpt-5.1-2025-11-13 with reasoning effort set to none. GPT-4.1 receives at most 50 groups, sampled proportionally within each anatomical group with both endpoints retained; the other API models receive all groups. Output limits are 4,096 tokens for GPT-4.1 and Claude, 8,192 for GPT-5.1 and Gemini Flash, and 16,384 for Gemini Pro. Empty Claude answers are retried at most three times; incomplete structured answers use 8,192 output tokens. Answers outside the requested categories are scored incorrect. Table A11 separates T accuracy by presentation, so the combined score can be interpreted alongside its two constituent cohorts. CarePath cohort uses its own ten-page MRI input and ninth-edition prompt.

## C.1 Evaluation Models

Table A7 lists all 30 evaluated models. The nine text-only models cover the five text-based subsets; the 21 image-capable models additionally complete pathology, MRI, and CarePath evaluation. The inventory spans medical and general-purpose models, with local inference for open-weight models and API access for hosted services. The four models used in the CarePath cohort construction are marked throughout the results.

Table A7: Evaluated models and task interfaces. Text models cover L1/L2/L4/L5/O1; imagecapable models additionally cover L3 and L6. † marks the four models used during CarePath cohort construction.
<table><tr><td>Model</td><td>Input interface</td><td>Inference</td></tr><tr><td>Medical-Specific LLMs</td><td></td><td></td></tr><tr><td>HuatuoGPT-o1-72B</td><td>Text</td><td>Local</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>Text</td><td>Local</td></tr><tr><td>Baichuan-M2-32B</td><td>Text</td><td>Local</td></tr><tr><td>Fleming-R1-32B</td><td>Text</td><td>Local</td></tr><tr><td>Fleming-R1-7B</td><td>Text</td><td>Local</td></tr><tr><td>Medical-Specific MLLMs</td><td></td><td></td></tr><tr><td>HealthGPT-Pro-4B</td><td>Text and image</td><td>Local</td></tr><tr><td>HealthGPT-Pro-8B†</td><td>Text and image</td><td>Local</td></tr><tr><td>Hulu-Med-4B</td><td>Text and image</td><td>Local</td></tr><tr><td>Hulu-Med-7B</td><td>Text and image</td><td>Local</td></tr><tr><td>Hulu-Med-32B</td><td>Text and image</td><td>Local</td></tr><tr><td>Lingshu-7B</td><td>Text and image</td><td>Local</td></tr><tr><td>Lingshu-32B</td><td>Text and image</td><td>Local</td></tr><tr><td>Open-Source LLMs</td><td></td><td></td></tr><tr><td>Qwen3-30B-A3B</td><td>Text</td><td>Local</td></tr><tr><td>GPT-OSS-20B</td><td>Text</td><td>Local</td></tr><tr><td>GPT-OSS-120B</td><td>Text</td><td>Local</td></tr><tr><td>DeepSeek-V3.1</td><td>Text</td><td>API</td></tr><tr><td>Open-Source MLLMs</td><td></td><td></td></tr><tr><td>Qwen3-VL-8B-Instruct†</td><td>Text and image</td><td>Local</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct†</td><td>Text and image</td><td>Local</td></tr><tr><td>Qwen3.5-9B</td><td>Text and image</td><td>Local</td></tr><tr><td>Qwen3.5-27B†</td><td>Text and image</td><td>Local</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>Text and image</td><td>Local</td></tr><tr><td>Qwen3.6-27B</td><td>Text and image</td><td>Local</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>Text and image</td><td>Local</td></tr><tr><td>InternVL3.5-38B</td><td>Text and image</td><td>Local</td></tr><tr><td>Proprietary (M)LLMs</td><td></td><td></td></tr><tr><td>GPT-5.1</td><td>Text and image</td><td>API</td></tr><tr><td>GPT-4.1</td><td>Text and image</td><td>API</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>Text and image</td><td></td></tr><tr><td></td><td></td><td>API</td></tr><tr><td>Gemini-2.5-Pro</td><td>Text and image</td><td>API</td></tr><tr><td>Gemini-2.5-Flash</td><td>Text and image</td><td>API</td></tr><tr><td>Claude-Sonnet-4.6</td><td>Text and image</td><td>API</td></tr></table>

## C.2 Task Definitions and Scoring

## C.2.1 Notation and Aggregation

All definitions apply to one evaluated model on one stated cohort; the model index is suppressed. An indicator 1{E} equals one if statement E is true and zero otherwise. A sum counts contributions, a product of binary indicators requires every factor to be one, and |A| denotes the number of elements in a finite set A.

Table A8: Task interfaces and scoring units used in the main NPCBench table. Formal definitions follow in sections C.2.1 and C.5.1.
<table><tr><td>Subset</td><td>Input</td><td>Response</td><td>Scoring unit and endpoint</td></tr><tr><td>L1</td><td>Guideline MCQ</td><td>Option letter</td><td>Item accuracy over 250 questions</td></tr><tr><td>L2</td><td>Compact vignette and MCQ</td><td>Option letter(s)</td><td>Macro accuracy over four capability groups</td></tr><tr><td>L3</td><td>Pathology overview; MRI panels or structure cards</td><td>Diagnostic labels and structured findings</td><td>Pathology: case; T/N stage: patient; sequence: item; structure: card accuracy</td></tr><tr><td>L4</td><td>Single-case history and MCQ Option letter(s)</td><td></td><td>Patient-macro accuracy over 84 patients</td></tr><tr><td>L5</td><td>Longitudinal history through Option letter the current checkpoint</td><td></td><td>Turn accuracy and all-turn patient success; 305 turns, 100 patients</td></tr><tr><td>L6</td><td>Images, clinical context, and Diagnostic labels and</td><td></td><td>Care-node accuracy and strict patient success; 217 nodes, 100 patients</td></tr><tr><td>01</td><td>care questions Case and consultation question</td><td>option letters Free-text consultation</td><td>Mean normalized rubric score over 103 questions</td></tr></table>

Table A9: Notation for closed-answer and patient-level evaluation. Each task uses the same finalized cohort and scoring set for every evaluated model.

```latex
Symbol Meaning and domain
I, $n = | \mathcal { I } |$ Scored item set and its size; $i \in \mathcal { Z }$ indexes one question, image item, or turn.
$y _ { i } , { \hat { y } } _ { i } , \ a _ { i }$ Reference label, parsed model label, and correctness $a _ { i } = \mathbf { 1 } \{ \hat { y } _ { i } = y _ { i } \} \in \{ 0 , 1 \}$
$P ,$ p Number of evaluated patients and patient index $p \in \{ 1 , \ldots , P \}$
$S _ { p } , n _ { p } , N$ Scored L6 care-node set for patient p, its size $\begin{array} { r } { n _ { p } = | S _ { p } | \geq 1 , } \end{array}$ and total scored node $\begin{array} { r } { N = \sum _ { p } n _ { p } } \end{array}$
$c _ { p j }$ Correctness of L6 care node $j \in \mathcal S _ { p }$ for patient p; a binary indicator.
$d _ { p } , \ t _ { p }$ Binary correctness indicators for pathology and MRI T stage; these symbols are not the predicted class
labels.
$C _ { p } , \ F _ { p }$ All-care correctness $\begin{array} { r } { C _ { p } = \prod _ { j \in { \cal S } _ { p } } c _ { p j } } \end{array}$ and joint image-label correctness $F _ { p } = d _ { p } t _ { p } ;$ both are binary.
```

Missing or conflicting extracted answers have correctness zero and remain in the scored denominator. The six L6 treatment questions excluded for insufficient decision evidence are omitted from $S _ { p }$ for every model. The CarePath cohort has $P = 1 0 0$ and $N = 2 1 7$ . The image-label analysis uses 146 care questions for 25 patients within this cohort.

For closed-answer tasks, the item-micro accuracy and group-macro accuracy are

$$
\operatorname { A c c } _ { \mathrm { m i c r o } } = \frac { 1 0 0 } { n } \sum _ { i \in \mathcal { I } } a _ { i } , \qquad \operatorname { A c c } _ { \mathrm { g r o u p } } = \frac { 1 0 0 } { G } \sum _ { g = 1 } ^ { G } \frac { 1 } { | \mathcal { T } _ { g } | } \sum _ { i \in \mathcal { I } _ { g } } a _ { i } .\tag{A1}
$$

Here the $G$ nonempty, disjoint sets $\mathcal { T } _ { g }$ partition I. Micro accuracy weights each scored item equally; group macro weights each group equally. L2 uses $G = 4$ clinical capability groups. For patient aggregation, let $\mathcal { T } _ { p }$ contain the scored items belonging to patient p in the task being evaluated:

$$
\mathrm { A c c } _ { \mathrm { p a t i e n t } } = \frac { 1 0 0 } { P } \sum _ { p = 1 } ^ { P } \frac { \sum _ { i \in \mathcal { T } _ { p } } a _ { i } } { | \mathcal { Z } _ { p } | } , \qquad \mathrm { A l l C o r r e c t } = \frac { 1 0 0 } { P } \sum _ { p = 1 } ^ { P } \prod _ { i \in \mathcal { T } _ { p } } a _ { i } .\tag{A2}
$$

Each $\mathcal { T } _ { p }$ is nonempty. L4 reports patient-macro accuracy over 84 patients; L5 reports turn-micro accuracy and AllCorrect as all-turn success over 100 patients. For L6, $\mathcal { T } _ { p } = \mathcal { S } _ { p }$ and $a _ { i } = c _ { p j }$ under the corresponding node indexing; AllCorrect is called AllCare. Its primary Care and Strict endpoints follow Eq. (1); AllCare omits the pathology and T-stage requirements.

Patient intersections. Let $\mathcal { C } = \{ p : C _ { p } = 1 \}$ and $\mathcal { F } = \{ p : F _ { p } = 1 \}$ be patient sets, and let superscript c denote a complement within $\{ 1 , \ldots , P \}$ . Table A20 reports set cardinalities. Equivalently, the four counts are

$$
n _ { a b } = \sum _ { p = 1 } ^ { P } \mathbf { 1 } \{ C _ { p } = a , ~ F _ { p } = b \} , \quad a , b \in \{ 0 , 1 \} , \qquad \sum _ { a = 0 } ^ { 1 } \sum _ { b = 0 } ^ { 1 } n _ { a b } = P .\tag{A3}
$$

Thus $n _ { 1 1 } = | \mathcal { C } \cap \mathcal { F } | , n _ { 1 0 } = | \mathcal { C } \cap \mathcal { F } ^ { c } | , n _ { 0 1 } = | \mathcal { C } ^ { c } \cap \mathcal { F } |$ , and $n _ { 0 0 } = | \mathcal { C } ^ { c } \cap \mathcal { F } ^ { c } |$ . Strict is $1 0 0 n _ { 1 1 } / P ;$ AllCare is $1 0 0 ( n _ { 1 1 } + n _ { 1 0 } ) / P$ . All four groups refer to the same patients.

## C.2.2 Classification Diagnostics

For a target class $k , \mathrm { ~ T P } _ { k }$ counts correctly predicted members of that class, $\mathrm { F N } _ { k }$ counts missed members, and $\mathrm { F P } _ { k }$ counts other examples predicted as class k. Class recall $R _ { k } .$ , precision $Q _ { k }$ , and secondary $F 1 _ { k }$ are proportions:

$$
R _ { k } = \frac { \mathrm { T P } _ { k } } { \mathrm { T P } _ { k } + \mathrm { F N } _ { k } } , \qquad Q _ { k } = \frac { \mathrm { T P } _ { k } } { \mathrm { T P } _ { k } + \mathrm { F P } _ { k } } , \qquad F 1 _ { k } = \frac { 2 \mathrm { T P } _ { k } } { 2 \mathrm { T P } _ { k } + \mathrm { F P } _ { k } + \mathrm { F N } _ { k } } .\tag{A4}
$$

These ratios require nonzero denominators; undefined diagnostic ratios are unavailable. For K evaluated classes with reference support, balanced accuracy is $1 0 0 K ^ { - 1 } \textstyle \sum _ { k = 1 } ^ { K } R _ { k }$ . T staging uses $K = 4 ( \mathrm { T 1 - T } 4 )$ , so a majority-class predictor has 25% balanced accuracy. Per-class recalls shown as percentages equal $1 0 \dot { 0 } R _ { k } . \ \mathrm { L } 3$ structure-card accuracy and these class-balanced measures have different aggregation units; the main structure column uses card-micro accuracy.

## C.2.3 Paired Effects and Uncertainty

For the binary endpoint in Eq. (A6), index the M matched units by i. Let $a _ { i } ^ { ( 0 ) }$ and $a _ { i } ^ { ( 1 ) }$ be correctness under baseline and intervention, respectively. The counts are $\begin{array} { r } { u = \sum _ { i } \mathbf { 1 } \{ a _ { i } ^ { ( 0 ) } = 0 , a _ { i } ^ { ( 1 ) } = 1 \} } \end{array}$ and $\begin{array} { r } { v = \sum _ { i } \mathbf { 1 } \{ a _ { i } ^ { ( 0 ) } = 1 , a _ { i } ^ { ( 1 ) } = 0 \} } \end{array}$ . The remaining $M - u - v$ units retain the same correctness. Consequently, $\Delta \in [ - 1 \dot { 0 } 0 , 1 0 0 ]$ is a percentage-point difference and $\Phi \in [ 0 , 1 0 0 ]$ is a percentage of correctness flips; changes between two incorrect answers do not contribute to either count.

The image-label comparison instead compares within-patient care accuracies. For condition $k \in$ {0, 1}, define the proportion $\begin{array} { r } { A _ { p } ^ { ( k ) } = n _ { p } ^ { - 1 } \sum _ { j \in S _ { p } } c _ { p j } ^ { ( k ) } } \end{array}$ . With $k = 0$ for predicted image labels and $k = 1$ for reference image labels, its reported effect is

$$
\Delta _ { \mathrm { p a t i e n t } } = \frac { 1 0 0 } { P } \sum _ { p = 1 } ^ { P } \left( A _ { p } ^ { ( 1 ) } - A _ { p } ^ { ( 0 ) } \right) { . }\tag{A5}
$$

Here $P = 2 5$ and $\textstyle \sum _ { p } n _ { p } = 1 4 6$ . The better, worse, and same columns in Table A23 count patients according to the sign of $A _ { p } ^ { ( 1 ) } - A _ { p } ^ { ( 0 ) }$ . The patient-macro effect also incorporates each patient’s change in accuracy.

Confidence intervals. Care accuracy and paired patient comparisons use a patient-cluster bootstrap. Each draw resamples patients with replacement, retaining all of a sampled patient’s scored units and both conditions of a pair. Each draw recomputes the relevant estimator: a ratio of turn totals for micro accuracy or a mean of patient accuracies for patient macro. The 2.5th and 97.5th percentiles form the reported 95% interval. L1 controls resample paired items; physician-calibration intervals resample consultation-response clusters. CarePath cohort strict-success intervals are exact binomial (Clopper–Pearson) intervals over 100 patients. Multi-model intervals are descriptive and unadjusted.

## C.3 Extended Results and Diagnostic Analyses

The family profiles in Fig. A1 show that high guideline scores coexist with low strict patient-level success across model groups. Figure A2 retains the differences between individual models. We then examine longitudinal decisions, anatomical evidence, staging, and patient-level outcomes before comparing controlled changes to the inputs.

![](images/f1061f72b96df455ebc88e6c4669584bebec16fc340190a5644bc66afbdb0982.jpg)  
Figure A1: Family-level capability profile. Family medians (a) and task-wise best scores within each family (b); the latter may come from different models. MM-EG averages five L3 endpoints; L6 uses the CarePath cohort. Per-model results appear in Table 1 and Fig. A2.

## C.3.1 L5 Longitudinal Revision Results

L5 reports turn accuracy and all-turn patient success because the two measures answer different questions. Turn accuracy counts correct next actions, whereas all-turn success requires every scored answer for one patient to be correct. All models use the same boxed single-letter answer interface. The 100 patients have 13/19/31/24/13 trajectories with 1/2/3/4/5 scored turns, respectively; the all-turn requirement therefore ranges from one answer to five. Table A10 reports both endpoints so that differences in individual decisions can be distinguished from differences in complete episodes.

Table A10: Longitudinal decision performance on 100 patients and 305 questions (%). Turn accuracy weights decisions equally; patient macro weights patients equally; all-turn success requires every scored answer for a patient to be correct.
<table><tr><td>Model</td><td>Turn</td><td>Patient macro</td><td>All-turn success</td></tr><tr><td>API models</td><td></td><td></td><td></td></tr><tr><td>DeepSeek-V3.1</td><td>74.4</td><td>74.2</td><td>42.0</td></tr><tr><td>GPT-5.1</td><td>75.7</td><td>73.7</td><td>45.0</td></tr><tr><td>GPT-4.1</td><td>78.7</td><td>77.5</td><td>45.0</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>87.9</td><td>88.6</td><td>71.0</td></tr><tr><td>Gemini-2.5-Pro</td><td>82.0</td><td>81.6</td><td>56.0</td></tr><tr><td>Gemini-2.5-Flash</td><td>69.5</td><td>71.7</td><td>42.0</td></tr><tr><td>Claude-Sonnet-4.6</td><td>85.2</td><td>85.6</td><td>63.0</td></tr><tr><td>Local models</td><td></td><td></td><td></td></tr><tr><td>HuatuoGPT-o1-72B</td><td>63.9</td><td>62.1</td><td>26.0</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>55.1</td><td>53.4</td><td>20.0</td></tr><tr><td>Baichuan-M2-32B</td><td>66.2</td><td>65.5</td><td>31.0</td></tr><tr><td>Fleming-R1-32B</td><td>75.7</td><td>74.1</td><td>40.0</td></tr><tr><td>Fleming-R1-7B</td><td>59.3</td><td>59.3</td><td>23.0</td></tr><tr><td>HealthGPT-Pro-4B</td><td>26.2</td><td>24.9</td><td>4.0</td></tr><tr><td>HealthGPT-Pro-8B</td><td>38.7</td><td>37.3</td><td>9.0</td></tr><tr><td>Hulu-Med-4B</td><td>59.3</td><td>59.1</td><td>26.0</td></tr><tr><td>Hulu-Med-7B</td><td>47.2</td><td>47.1</td><td>16.0</td></tr><tr><td>Hulu-Med-32B</td><td>70.2</td><td>70.4</td><td>37.0</td></tr><tr><td>Lingshu-7B</td><td>49.2</td><td>49.4</td><td>14.0</td></tr><tr><td>Lingshu-32B</td><td>70.5</td><td>70.1</td><td>37.0</td></tr><tr><td>Qwen3-30B-A3B</td><td>63.3</td><td>63.5</td><td>27.0</td></tr><tr><td>GPT-OSS-20B</td><td>48.9</td><td>48.0</td><td>13.0</td></tr><tr><td>GPT-OSS-120B</td><td>60.3</td><td>58.0</td><td>23.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>69.8</td><td>70.0</td><td>38.0</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>68.5</td><td>67.6</td><td>36.0</td></tr><tr><td>Qwen3.5-9B</td><td>60.7</td><td>60.7</td><td>24.0</td></tr><tr><td>Qwen3.5-27B</td><td>76.4</td><td>74.5</td><td>46.0</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>72.8</td><td>72.9</td><td>41.0</td></tr><tr><td>Qwen3.6-27B</td><td>79.7</td><td>79.6</td><td>53.0</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>76.1</td><td>74.6</td><td>49.0</td></tr><tr><td>InternVL3.5-38B</td><td>58.4</td><td>58.4</td><td>21.0</td></tr></table>

Model performance by clinical capability
<table><tr><td rowspan=1 colspan=2>HuatuoGPT-o1-72B</td><td rowspan=1 colspan=1>88.4</td><td rowspan=1 colspan=1>73.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>46.9</td><td rowspan=1 colspan=1>63.9</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>10.1</td></tr><tr><td rowspan=1 colspan=2>HuatuoGPT-o1-7B</td><td rowspan=1 colspan=1>77.2</td><td rowspan=1 colspan=1>52.9</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>43.2</td><td rowspan=1 colspan=1>55.1</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>5.3</td></tr><tr><td rowspan=1 colspan=2>Baichuan-M2-32B</td><td rowspan=1 colspan=1>22.8</td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>5.3</td><td rowspan=1 colspan=1>66.2</td><td rowspan=1 colspan=1>31.0</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>38.8</td></tr><tr><td rowspan=1 colspan=2>Fleming-R1-32B</td><td rowspan=1 colspan=1>85.6</td><td rowspan=1 colspan=1>37.6</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>30.3</td><td rowspan=1 colspan=1>75.7</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>28.6</td></tr><tr><td rowspan=1 colspan=2>Fleming-R1-7B</td><td rowspan=1 colspan=1>84.4</td><td rowspan=1 colspan=1>60.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>37.2</td><td rowspan=1 colspan=1>59.3</td><td rowspan=1 colspan=1>23.0</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>24.8</td></tr><tr><td rowspan=10 colspan=2>HealthGPT-4BHealthGPT-8BHulu-Med-4BHulu-Med-7BHulu-Med-32BLingshu-7BLingshu-32BQwen3-30BGPT-OSS-20BGPT-OSS-120B</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>38.5</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>54.0</td><td rowspan=1 colspan=1>9.4</td><td rowspan=1 colspan=1>50.8</td><td rowspan=1 colspan=1>26.2</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>19.8</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>14.7</td></tr><tr><td rowspan=1 colspan=1>81.2</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>39.5</td><td rowspan=1 colspan=1>20.3</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>8.6</td><td rowspan=1 colspan=1>48.7</td><td rowspan=1 colspan=1>38.7</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>20.3</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>18.8</td></tr><tr><td rowspan=1 colspan=1>78.0</td><td rowspan=1 colspan=1>56.9</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>19.8</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>5.6</td><td rowspan=1 colspan=1>53.1</td><td rowspan=1 colspan=1>59.3</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>39.6</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>8.7</td></tr><tr><td rowspan=1 colspan=1>82.4</td><td rowspan=1 colspan=1>56.8</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=1>20.5</td><td rowspan=1 colspan=1>58.0</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>48.3</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>16.0</td><td rowspan=1 colspan=1>39.6</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>10.7</td></tr><tr><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>74.5</td><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>75.5</td><td rowspan=1 colspan=1>21.8</td><td rowspan=1 colspan=1>53.4</td><td rowspan=1 colspan=1>70.2</td><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>7.6</td></tr><tr><td rowspan=1 colspan=1>78.4</td><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>3.8</td><td rowspan=1 colspan=1>47.0</td><td rowspan=1 colspan=1>9.3</td><td rowspan=1 colspan=1>42.6</td><td rowspan=1 colspan=1>49.2</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>44.2</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>3.2</td></tr><tr><td rowspan=1 colspan=1>85.6</td><td rowspan=1 colspan=1>61.0</td><td rowspan=1 colspan=1>59.4</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>75.0</td><td rowspan=1 colspan=1>11.4</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>70.5</td><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>55.3</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>9.0</td></tr><tr><td rowspan=1 colspan=1>83.2</td><td rowspan=1 colspan=1>66.2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>51.3</td><td rowspan=1 colspan=1>63.3</td><td rowspan=1 colspan=1>27.0</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>26.8</td></tr><tr><td rowspan=1 colspan=1>74.4</td><td rowspan=1 colspan=1>58.3</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>32.1</td><td rowspan=1 colspan=1>48.9</td><td rowspan=1 colspan=1>13.0</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>28.0</td></tr><tr><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>75.4</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>49.4</td><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=1>23.0</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>44.9</td></tr><tr><td rowspan=2 colspan=2>DeepSeek-V3.1</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>74.4</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>28.2</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=1>83.6</td><td rowspan=1 colspan=1>61.5</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>29.5</td><td rowspan=1 colspan=1>4.7</td><td rowspan=1 colspan=1>52.3</td><td rowspan=1 colspan=1>69.8</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>52.5</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>25.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3-VL-30B</td><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>67.3</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>21.8</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>36.2</td><td rowspan=1 colspan=1>52.3</td><td rowspan=1 colspan=1>68.5</td><td rowspan=1 colspan=1>36.0</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>29.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1>85.2</td><td rowspan=1 colspan=1>67.8</td><td rowspan=1 colspan=1>72.5</td><td rowspan=1 colspan=1>26.3</td><td rowspan=1 colspan=1>20.3</td><td rowspan=1 colspan=1>65.5</td><td rowspan=1 colspan=1>38.9</td><td rowspan=1 colspan=1>58.0</td><td rowspan=1 colspan=1>60.7</td><td rowspan=1 colspan=1>24.0</td><td rowspan=1 colspan=1>53.0</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>34.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3.5-27B</td><td rowspan=1 colspan=1>89.2</td><td rowspan=1 colspan=1>75.8</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>67.0</td><td rowspan=1 colspan=1>47.6</td><td rowspan=1 colspan=1>61.0</td><td rowspan=1 colspan=1>76.4</td><td rowspan=1 colspan=1>46.0</td><td rowspan=1 colspan=1>62.7</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>46.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3.5-35B</td><td rowspan=1 colspan=1>86.4</td><td rowspan=1 colspan=1>75.2</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>35.5</td><td rowspan=1 colspan=1>21.8</td><td rowspan=1 colspan=1>65.0</td><td rowspan=1 colspan=1>50.3</td><td rowspan=1 colspan=1>54.5</td><td rowspan=1 colspan=1>72.8</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=1>63.1</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>41.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3.6-27B</td><td rowspan=1 colspan=1>87.6</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>42.8</td><td rowspan=1 colspan=1>75.5</td><td rowspan=1 colspan=1>41.4</td><td rowspan=1 colspan=1>50.3</td><td rowspan=1 colspan=1>79.7</td><td rowspan=1 colspan=1>53.0</td><td rowspan=1 colspan=1>65.9</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>47.2</td></tr><tr><td rowspan=1 colspan=2>Qwen3.6-35B</td><td rowspan=1 colspan=1>85.2</td><td rowspan=1 colspan=1>67.9</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>43.5</td><td rowspan=1 colspan=1>20.3</td><td rowspan=1 colspan=1>61.5</td><td rowspan=1 colspan=1>44.8</td><td rowspan=1 colspan=1>46.8</td><td rowspan=1 colspan=1>76.1</td><td rowspan=1 colspan=1>49.0</td><td rowspan=1 colspan=1>61.8</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>42.4</td></tr><tr><td rowspan=1 colspan=2>InternVL3.5-38B</td><td rowspan=1 colspan=1>88.0</td><td rowspan=1 colspan=1>69.7</td><td rowspan=1 colspan=1>59.4</td><td rowspan=1 colspan=1>36.8</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>45.5</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>58.4</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>9.5</td></tr><tr><td rowspan=1 colspan=2>GPT-5.1</td><td rowspan=1 colspan=1>90.4</td><td rowspan=1 colspan=1>80.9</td><td rowspan=1 colspan=1>62.3</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>32.8</td><td rowspan=1 colspan=1>47.0</td><td rowspan=1 colspan=1>26.3</td><td rowspan=1 colspan=1>63.5</td><td rowspan=1 colspan=1>75.7</td><td rowspan=1 colspan=1>45.0</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>58.4</td></tr><tr><td rowspan=1 colspan=2>GPT-4.1</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>77.6</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=1>39.8</td><td rowspan=1 colspan=1>22.0</td><td rowspan=1 colspan=1>48.5</td><td rowspan=1 colspan=1>28.3</td><td rowspan=1 colspan=1>54.4</td><td rowspan=1 colspan=1>78.7</td><td rowspan=1 colspan=1>45.0</td><td rowspan=1 colspan=1>61.8</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>33.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Gemini-3.1</td><td rowspan=1 colspan=1>94.0</td><td rowspan=1 colspan=1>92.5</td><td rowspan=1 colspan=1>63.8</td><td rowspan=1 colspan=1>31.0</td><td rowspan=1 colspan=1>38.5</td><td rowspan=1 colspan=1>77.0</td><td rowspan=1 colspan=1>72.0</td><td rowspan=1 colspan=1>71.9</td><td rowspan=1 colspan=1>87.9</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=1>89.9</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=1>47.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Gemini-2.5-Pro</td><td rowspan=1 colspan=1>93.2</td><td rowspan=1 colspan=1>85.8</td><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>48.6</td><td rowspan=1 colspan=1>67.0</td><td rowspan=1 colspan=1>82.0</td><td rowspan=1 colspan=1>56.0</td><td rowspan=1 colspan=1>80.6</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>38.5</td></tr><tr><td rowspan=1 colspan=2>Gemini-2.5-Flash</td><td rowspan=1 colspan=1>88.4</td><td rowspan=1 colspan=1>71.8</td><td rowspan=1 colspan=1>65.2</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>25.8</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>59.8</td><td rowspan=1 colspan=1>69.5</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>65.0</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>34.4</td></tr><tr><td rowspan=1 colspan=2>Claude-4.6</td><td rowspan=1 colspan=1>93.2</td><td rowspan=1 colspan=1>86.1</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>33.5</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>57.0</td><td rowspan=1 colspan=1>27.9</td><td rowspan=1 colspan=1>69.2</td><td rowspan=1 colspan=1>85.2</td><td rowspan=1 colspan=1>63.0</td><td rowspan=1 colspan=1>80.6</td><td rowspan=1 colspan=1>23.0</td><td rowspan=1 colspan=1>47.5</td></tr><tr><td rowspan=1 colspan=15>Seq.              L5   15   L6Path.   16G-Mem.A-App.T-Stg.N-Stg.Struct.C-PlanTurnCareStrictC-Rub.All-turn0          25         50          75         100Score (%)</td></tr></table>

Figure A2: Model performance by clinical capability. Scores follow Table 1; gray cells denote unavailable interfaces. L6 uses the CarePath cohort, and L3 uses its static-image cohorts.

## C.3.2 MRI T and N Staging

Table A11 separates card-based and plane-group T results because patient composition and image representation differ. The combined T400 score weights all 400 patients equally; N-stage balanced accuracy weights N0–N3 equally. Predictions outside the requested categories count as incorrect.

Table A11: T-stage accuracy on 400 patients and N-stage accuracy on 400 different patients (%). T-stage results are also shown separately for 100 patients with structure cards and 300 with fivesequence plane groups. N-stage balanced accuracy averages recall over N0–N3. Majority-class baselines are 37.5% for T400 and 40.0% for N400.
<table><tr><td>Model</td><td>T400</td><td>T100 cards</td><td>T300 groups</td><td>N400</td><td>N bal.</td></tr><tr><td>Qwen3-VL-8B</td><td>37.5</td><td>35.0</td><td>38.3</td><td>20.0</td><td>25.0</td></tr><tr><td>Lingshu-7B</td><td>37.5</td><td>35.0</td><td>38.3</td><td>3.8</td><td>4.7</td></tr><tr><td>Qwen3.5-9B</td><td>26.3</td><td>26.0</td><td>26.3</td><td>20.3</td><td>25.3</td></tr><tr><td>Qwen3-VL-30B</td><td>37.5</td><td>35.0</td><td>38.3</td><td>21.8</td><td>27.2</td></tr><tr><td>Lingshu-32B</td><td>37.5</td><td>35.0</td><td>38.3</td><td>20.0</td><td>25.0</td></tr><tr><td>Qwen3.5-27B</td><td>37.0</td><td>34.0</td><td>38.0</td><td>27.3</td><td>33.3</td></tr><tr><td>Qwen3.5-35B</td><td>35.5</td><td>43.0</td><td>33.0</td><td>21.8</td><td>27.0</td></tr><tr><td>Qwen3.6-27B</td><td>37.5</td><td>35.0</td><td>38.3</td><td>42.8</td><td>31.9</td></tr><tr><td>Qwen3.6-35B</td><td>43.5</td><td>35.0</td><td>46.3</td><td>20.3</td><td>25.2</td></tr><tr><td>Hulu-Med-4B</td><td>34.5</td><td>35.0</td><td>34.3</td><td>19.8</td><td>24.5</td></tr><tr><td>HealthGPT-4B</td><td>38.5</td><td>39.0</td><td>38.3</td><td>20.0</td><td>25.0</td></tr><tr><td>HealthGPT-8B</td><td>39.5</td><td>33.0</td><td>41.7</td><td>20.3</td><td>25.3</td></tr><tr><td>Hulu-Med-7B</td><td>12.5</td><td>15.0</td><td>11.7</td><td>20.5</td><td>25.3</td></tr><tr><td>Hulu-Med-32B</td><td>29.0</td><td>16.0</td><td>33.3</td><td>20.0</td><td>25.0</td></tr><tr><td>InternVL3.5-38B</td><td>36.8</td><td>33.0</td><td>38.0</td><td>20.0</td><td>25.0</td></tr><tr><td>GPT-5.1</td><td>27.3</td><td>39.0</td><td>23.3</td><td>32.8</td><td>30.6</td></tr><tr><td>GPT-4.1</td><td>39.8</td><td>30.0</td><td>43.0</td><td>22.0</td><td>27.5</td></tr><tr><td>Gemini-2.5-Flash</td><td>38.0</td><td>34.0</td><td>39.3</td><td>25.8</td><td>29.5</td></tr><tr><td>Gemini-2.5-Pro</td><td>37.5</td><td>33.0</td><td>39.0</td><td>26.0</td><td>28.6</td></tr><tr><td>Gemini-3.1</td><td>31.0</td><td>34.0</td><td>30.0</td><td>38.5</td><td>38.4</td></tr><tr><td>Claude-4.6</td><td>33.5</td><td>33.0</td><td>33.7</td><td>27.3</td><td>28.1</td></tr></table>

Prediction counts distinguish single-category predictions from answers outside the allowed categories. On the 300 plane-group T cases, Qwen3-VL-8B/30B-A3B, Lingshu-32B, and HealthGPT-Pro-4B always predict T4 (115/300 correct), whereas Hulu-Med-7B always predicts T1 (35/300). All five have 25.0% balanced accuracy. On N400, only 46/400 Lingshu-7B answers use a valid category (11.5%), all N3; 15 are correct, and balanced accuracy is 4.7%.

Second-centre MRI assessment. The Nanfang Hospital panel uses clinical T-stage references associated with 53 MRI examinations. Each examination provides ten four-row panels, sampling 30 positions per acquired sequence across the panels. All ten images enter one request with an eighth-edition T-only prompt. Table A12 reports accuracy, balanced accuracy, and per-stage recall. T3 accounts for 31 of 53 cases, giving a majority-class accuracy of 58.5%. Hulu-Med-4B reaches 60.4% accuracy but 28.1% balanced accuracy, whereas Qwen3.5-35B reaches 50.9% and 39.8%, respectively. The class-dependent ordering shows that higher overall accuracy can coincide with poorer recognition of less frequent stages. This panel evaluates MRI staging at a second centre.

## C.3.3 L3 Multimodal Evidence-Grounding Diagnostics

The L3 diagnostics examine three linked questions: whether the marked anatomy is identified, whether its invasion status is correct, and whether the resulting evidence supports the T category. They use the 100-case anatomical panel, with 18 invasion labels and one T category per patient. Structure identification is scored over 2,700 cards covering 18 base structures; nasopharynx cards are excluded from this primary MRI scoring set. The text-stager experiment uses structured findings as input; the CarePath and cine inputs are described in sections C.3.5 and C.4.

Patient-level measures then distinguish partial evidence from complete evidence. Pt. Inv. averages invasion correctness over all 18 structures within each patient and then across patients. Complete Inv.

Table A12: T-stage results at Nanfang Hospital on 53 patients with definite reference labels (T1/T2/T3/T4: 4/8/31/10). Two other evaluated examinations lacked a definite reference T stage and are not scored. Accuracy, balanced accuracy and per-stage recall are percentages; the majority-class baseline is 58.5%.
<table><tr><td>Model</td><td>Accuracy</td><td>Balanced</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td></tr><tr><td>HealthGPT-4B</td><td>17.0</td><td>22.5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>90.0</td></tr><tr><td>HealthGPT-8B</td><td>22.6</td><td>24.9</td><td>0.0</td><td>0.0</td><td>9.7</td><td>90.0</td></tr><tr><td>Qwen3-VL-8B</td><td>18.9</td><td>25.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>Lingshu-7B</td><td>39.6</td><td>22.0</td><td>0.0</td><td>0.0</td><td>58.1</td><td>30.0</td></tr><tr><td>Qwen3.5-9B</td><td>35.8</td><td>26.1</td><td>0.0</td><td>12.5</td><td>41.9</td><td>50.0</td></tr><tr><td>Qwen3-VL-30B</td><td>18.9</td><td>25.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>Lingshu-32B</td><td>37.7</td><td>28.0</td><td>0.0</td><td>0.0</td><td>41.9</td><td>70.0</td></tr><tr><td>Hulu-Med-4B</td><td>60.4</td><td>28.1</td><td>0.0</td><td>12.5</td><td>100.0</td><td>0.0</td></tr><tr><td>Hulu-Med-7B</td><td>18.9</td><td>22.0</td><td>0.0</td><td>75.0</td><td>12.9</td><td>0.0</td></tr><tr><td>Hulu-Med-32B</td><td>17.0</td><td>17.4</td><td>0.0</td><td>0.0</td><td>9.7</td><td>60.0</td></tr><tr><td>Qwen3.5-27B</td><td>18.9</td><td>33.1</td><td>50.0</td><td>12.5</td><td>0.0</td><td>70.0</td></tr><tr><td>Qwen3.5-35B</td><td>50.9</td><td>39.8</td><td>75.0</td><td>0.0</td><td>74.2</td><td>10.0</td></tr><tr><td>Qwen3.6-27B</td><td>17.0</td><td>38.1</td><td>100.0</td><td>12.5</td><td>0.0</td><td>40.0</td></tr><tr><td>Qwen3.6-35B</td><td>24.5</td><td>24.0</td><td>0.0</td><td>0.0</td><td>16.1</td><td>80.0</td></tr><tr><td>InternVL3.5-38B</td><td>28.3</td><td>33.7</td><td>0.0</td><td>25.0</td><td>9.7</td><td>100.0</td></tr><tr><td>GPT-5.1</td><td>13.2</td><td>28.7</td><td>75.0</td><td>0.0</td><td>0.0</td><td>40.0</td></tr><tr><td>GPT-4.1</td><td>18.9</td><td>25.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>Gemini-2.5-Flash</td><td>41.5</td><td>26.2</td><td>0.0</td><td>0.0</td><td>54.8</td><td>50.0</td></tr><tr><td>Gemini-2.5-Pro</td><td>35.8</td><td>30.6</td><td>0.0</td><td>0.0</td><td>32.3</td><td>90.0</td></tr><tr><td>Gemini-3.1</td><td>34.0</td><td>28.7</td><td>0.0</td><td>12.5</td><td>32.3</td><td>70.0</td></tr><tr><td>Claude-4.6</td><td>41.5</td><td>20.1</td><td>0.0</td><td>12.5</td><td>67.7</td><td>0.0</td></tr></table>

requires all 18 labels to be correct; EG T additionally requires the correct final T stage. A model can therefore obtain a correct direct T label while failing complete evidence or EG T. Table A15 reports these endpoints together, and the ablations below examine which errors remain when anatomical identity or invasion labels are supplied by the reference.

Table A13: Anatomical diagnostic panel (100 patients). Interpretation of the L3 MRI staging metrics and ablations. These metrics are reported together because each isolates a different part of the evidence-to-stage chain.
<table><tr><td>Metric</td><td>Scoring target</td><td>Interpretation</td></tr><tr><td>Direct T</td><td>The final t_stage field generated by the evaluated model in the patient-level MRI prompt</td><td>Measures whether the model can output the correct stage label, but does not require the supporting structure-level evidence to be correct</td></tr><tr><td>Complete Inv.</td><td>Whether all 18 base-structure invasion labels are correct for one patient</td><td>Measures whether every scored invasion finding is correct</td></tr><tr><td>EG T</td><td>Complete Inv. plus correct Direct T for the same patient</td><td>Strict evidence-grounded staging success; a model must both extract the evidence and produce the right stage</td></tr><tr><td>Oracle-slot T</td><td>Deterministic staging after placing predicted invasion labels into the gold structure slots</td><td>Isolates invasion recognition while removing structure-localization errors</td></tr><tr><td>Model-slot T</td><td>Deterministic staging after placing predicted invasion labels into the model-predicted structure slots</td><td>Measures the full structure-localization-plus-invasion evidence chain</td></tr><tr><td>LLM-slot T</td><td>Fixed GPT-OSS-120B staging from model-generated structure-card summaries</td><td>Tests whether a stronger text stager can recover T stage from the generated evidence representation</td></tr><tr><td>Gold Rule / Gold LLM-slot</td><td>Deterministic or GPT-OSS-120B staging from gold 18-structure evidence</td><td>Estimates the ceiling imposed by the 18-structure evidence abstraction</td></tr></table>

The patient-level MRI comparison uses the same structured response fields for local and API models.

![](images/119dc82e4034a07f9de7e97ba7a96c1c3b3314d0311482aa725299b9b94ceead.jpg)

![](images/695a4580912eecdbb0045557ba1b631d4df4950fc5e5c6f4bc94e87574128311.jpg)

(c) Evidence-based staging  
![](images/c7955c5f198b61bd88511b7126b89211b87244d7e8d8eaa4d275a644b017af33.jpg)

![](images/ee81a850105fb592d23bd316336211128c06e9d55375c0ae053c0bd64f5053a5.jpg)  
Figure A3: L3 evidence-grounding diagnostics. Panels (a,b,d) use the 100-case anatomical panel; panel (c) uses the GPT-OSS-120B text-stager experiment. (a) Direct T-stage accuracy can diverge from class-balanced T recall. (b) Base-structure localization does not guarantee correct structure-plusinvasion evidence. (c) When gold 18-structure evidence is supplied, a fixed GPT-OSS-120B stager nearly reaches the rule ceiling; when model-generated evidence is supplied, LLM-slot T-stage scores remain much lower. Reference shows gold-evidence LLM-slot accuracy and balanced recall, with no direct-image prediction. (d) Per-stage recall shows uneven recognition of the four T categories.

Table A14: Local L3 component results. Structure and joint card accuracies weight all 2,700 cards equally. Inv. is conditioned on correct structure identification. Pt. Inv. averages correctness over the 18 invasion labels per patient. EG T requires all 18 labels and the T category to be correct. Pathology and sequence match Table 1; all values are percentages.
<table><tr><td>Model</td><td>Path.</td><td>Struct.</td><td>Inv.</td><td>Joint</td><td>Pt. Inv.</td><td>EG T</td><td>Seq.</td></tr><tr><td>HealthGPT-Pro-4B</td><td>56.5</td><td>9.4</td><td>30.4</td><td>2.9</td><td>43.3</td><td>0.0</td><td>54.0</td></tr><tr><td>HealthGPT-Pro-8B</td><td>56.5</td><td>8.6</td><td>41.6</td><td>3.6</td><td>48.4</td><td>0.0</td><td>50.0</td></tr><tr><td>Hulu-Med-4B</td><td>56.5</td><td>5.6</td><td>30.9</td><td>1.7</td><td>47.7</td><td>0.0</td><td>30.0</td></tr><tr><td>Hulu-Med-7B</td><td>56.5</td><td>7.5</td><td>48.0</td><td>3.6</td><td>62.4</td><td>5.0</td><td>58.0</td></tr><tr><td>Hulu-Med-32B</td><td>60.9</td><td>21.8</td><td>46.6</td><td>10.1</td><td>59.8</td><td>3.0</td><td>75.5</td></tr><tr><td>Lingshu-7B</td><td>56.5</td><td>9.3</td><td>6.8</td><td>0.6</td><td>57.9</td><td>4.0</td><td>47.0</td></tr><tr><td>Lingshu-32B</td><td>59.4</td><td>11.4</td><td>50.0</td><td>5.7</td><td>37.6</td><td>0.0</td><td>75.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>56.5</td><td>4.7</td><td>33.1</td><td>1.6</td><td>41.3</td><td>0.0</td><td>29.5</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>56.5</td><td>36.2</td><td>20.6</td><td>7.4</td><td>37.6</td><td>0.0</td><td>40.0</td></tr><tr><td>Qwen3.5-9B</td><td>72.5</td><td>38.9</td><td>36.6</td><td>14.3</td><td>66.0</td><td>2.0</td><td>65.5</td></tr><tr><td>Qwen3.5-27B</td><td>71.0</td><td>47.6</td><td>48.9</td><td>23.3</td><td>53.9</td><td>0.0</td><td>67.0</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>66.7</td><td>50.3</td><td>30.3</td><td>15.3</td><td>66.4</td><td>0.0</td><td>65.0</td></tr><tr><td>Qwen3.6-27B</td><td>71.0</td><td>41.4</td><td>46.2</td><td>19.1</td><td>45.0</td><td>0.0</td><td>75.5</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>66.7</td><td>44.8</td><td>31.4</td><td>14.1</td><td>61.7</td><td>0.0</td><td>61.5</td></tr><tr><td>InternVL3.5-38B</td><td>59.4</td><td>3.7</td><td>45.0</td><td>1.7</td><td>54.7</td><td>0.0</td><td>45.5</td></tr></table>

Table A15: MRI diagnostics on the anatomical panel for 100 patients. Complete Inv. requires all 18 invasion labels to be correct; Direct T requires the correct final T-category field; EG T requires both. Missing or invalid labels remain incorrect in their respective endpoints.
<table><tr><td>Model</td><td>Complete Inv.</td><td>Direct T</td><td>EG T</td></tr><tr><td>HealthGPT-Pro-4B</td><td>0.0</td><td>39.0</td><td>0.0</td></tr><tr><td>HealthGPT-Pro-8B</td><td>0.0</td><td>33.0</td><td>0.0</td></tr><tr><td>Hulu-Med-4B</td><td>0.0</td><td>35.0</td><td>0.0</td></tr><tr><td>Hulu-Med-7B</td><td>5.0</td><td>15.0</td><td>5.0</td></tr><tr><td>Hulu-Med-32B</td><td>3.0</td><td>16.0</td><td>3.0</td></tr><tr><td>Lingshu-7B</td><td>4.0</td><td>35.0</td><td>4.0</td></tr><tr><td>Lingshu-32B</td><td>0.0</td><td>35.0</td><td>0.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>0.0</td><td>35.0</td><td>0.0</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>0.0</td><td>35.0</td><td>0.0</td></tr><tr><td>Qwen3.5-9B</td><td>2.0</td><td>26.0</td><td>2.0</td></tr><tr><td>Qwen3.5-27B</td><td>0.0</td><td>34.0</td><td>0.0</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.0</td><td>43.0</td><td>0.0</td></tr><tr><td>Qwen3.6-27B</td><td>0.0</td><td>35.0</td><td>0.0</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>0.0</td><td>35.0</td><td>0.0</td></tr><tr><td>InternVL3.5-38B</td><td>0.0</td><td>33.0</td><td>0.0</td></tr><tr><td>GPT-5.1</td><td>0.0</td><td>39.0</td><td>0.0</td></tr><tr><td>GPT-4.1</td><td>0.0</td><td>30.0</td><td>0.0</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>3.0</td><td>34.0</td><td>3.0</td></tr><tr><td>Gemini-2.5-Pro</td><td>1.0</td><td>33.0</td><td>1.0</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.0</td><td>34.0</td><td>0.0</td></tr><tr><td>Claude-Sonnet-4.6</td><td>1.0</td><td>33.0</td><td>1.0</td></tr></table>

## C.3.4 L3 MRI Staging Ablations

The staging comparisons examine two possible sources of error: extracting anatomical findings and assigning a stage from those findings. The deterministic rule and GPT-OSS-120B stager assign T categories from structured anatomical findings. When gold 18-structure evidence is supplied, the stager approaches the gold ceiling; when model-generated evidence is supplied, performance remains much lower. Under this fixed structured-evidence interface, the contrast locates a major performance gap in structure and invasion extraction. The class recalls in Table A16 show that several models recognize advanced T categories more often than early categories. For example, GPT-5.1 reaches 39.0 Direct T but 27.9 Balanced T because most correct cases are T4, whereas Gemini-3.1-Pro-Preview has lower micro Direct T but the strongest class-balanced recall among the listed patient-staging runs.

The deterministic structure-only rule assigns T4 when cavernous sinus invasion is present; otherwise T3 when skull-base, pterygoid-base, upper-cervical, or paranasal-sinus structures are invaded; otherwise T2 when the parapharyngeal-space, medial/lateral pterygoid, or longus-capitis labels are positive; otherwise T1. Levator veli palatini and carotid sheath are not used for upstaging in this 18-structure rule. The 99.0 Gold Rule ceiling indicates that one gold T4 case depends on staging evidence outside the scored 18-base-structure set.

Table A16: T-category diagnostics on the anatomical panel. The 100 patients comprise 15 T1, 15 T2, 35 T3, and 35 T4 cases. Balanced T averages the four class recalls. T-bal60 uses a fixed subset of 15 cases per stage, with the same patient membership for all models.
<table><tr><td>Model</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>Direct T</td><td>Bal. T</td><td>T-bal60</td></tr><tr><td>HealthGPT-Pro-4B</td><td>6.7</td><td>6.7</td><td>11.4</td><td>94.3</td><td>39.0</td><td>29.8</td><td>31.7</td></tr><tr><td>HealthGPT-Pro-8B</td><td>0.0</td><td>0.0</td><td>5.7</td><td>88.6</td><td>33.0</td><td>23.6</td><td>25.0</td></tr><tr><td>Hulu-Med-4B</td><td>26.7</td><td>13.3</td><td>8.6</td><td>74.3</td><td>35.0</td><td>30.7</td><td>26.7</td></tr><tr><td>Hulu-Med-7B</td><td>100.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>15.0</td><td>25.0</td><td>25.0</td></tr><tr><td>Hulu-Med-32B</td><td>80.0</td><td>0.0</td><td>2.9</td><td>8.6</td><td>16.0</td><td>22.9</td><td>20.0</td></tr><tr><td>Lingshu-7B</td><td>73.3</td><td>0.0</td><td>0.0</td><td>68.6</td><td>35.0</td><td>35.5</td><td>36.7</td></tr><tr><td>Lingshu-32B</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>35.0</td><td>25.0</td><td>25.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>35.0</td><td>25.0</td><td>25.0</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>35.0</td><td>25.0</td><td>25.0</td></tr><tr><td>Qwen3.5-9B</td><td>26.7</td><td>86.7</td><td>20.0</td><td>5.7</td><td>26.0</td><td>34.8</td><td>40.0</td></tr><tr><td>Qwen3.5-27B</td><td>0.0</td><td>33.3</td><td>31.4</td><td>51.4</td><td>34.0</td><td>29.0</td><td>28.3</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>6.7</td><td>0.0</td><td>57.1</td><td>62.9</td><td>43.0</td><td>31.7</td><td>36.7</td></tr><tr><td>Qwen3.6-27B</td><td>0.0</td><td>13.3</td><td>0.0</td><td>94.3</td><td>35.0</td><td>26.9</td><td>25.0</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>0.0</td><td>0.0</td><td>25.7</td><td>74.3</td><td>35.0</td><td>25.0</td><td>16.7</td></tr><tr><td>InternVL3.5-38B</td><td>0.0</td><td>0.0</td><td>5.7</td><td>88.6</td><td>33.0</td><td>23.6</td><td>20.0</td></tr><tr><td>GPT-5.1</td><td>0.0</td><td>0.0</td><td>20.0</td><td>91.4</td><td>39.0</td><td>27.9</td><td>35.0</td></tr><tr><td>GPT-4.1</td><td>0.0</td><td>0.0</td><td>20.0</td><td>65.7</td><td>30.0</td><td>21.4</td><td>23.3</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>60.0</td><td>26.7</td><td>17.1</td><td>42.9</td><td>34.0</td><td>36.7</td><td>33.3</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.0</td><td>6.7</td><td>14.3</td><td>77.1</td><td>33.0</td><td>24.5</td><td>26.7</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.0</td><td>6.7</td><td>11.4</td><td>82.9</td><td>34.0</td><td>25.2</td><td>26.7</td></tr><tr><td>Claude-Sonnet-4.6</td><td>6.7</td><td>6.7</td><td>65.7</td><td>22.9</td><td>33.0</td><td>25.5</td><td>25.0</td></tr></table>

Table A17: Deterministic evidence-to-stage analysis on the 100-patient anatomical panel. Inv. is conditional on correct anatomy; Joint requires both anatomy and invasion. Oracle-slot places predicted invasion in the reference anatomy slot; Model-slot uses predicted anatomy. Unfilled slots take the fixed zero default, whereas complete-evidence correctness requires every label. Gold Rule measures the ceiling of the 18-structure mapping.
<table><tr><td>Model</td><td>Struct.</td><td>Inv.</td><td>Joint</td><td>Gold Rule</td><td>Oracle-slot</td><td>Model-slot</td><td>Direct T</td></tr><tr><td>HealthGPT-Pro-4B</td><td>9.4</td><td>30.4</td><td>2.9</td><td>99.0</td><td>35.0</td><td>26.0</td><td>39.0</td></tr><tr><td>HealthGPT-Pro-8B</td><td>8.6</td><td>41.6</td><td>3.6</td><td>99.0</td><td>35.0</td><td>23.0</td><td>33.0</td></tr><tr><td>Hulu-Med-4B</td><td>5.6</td><td>30.9</td><td>1.7</td><td>99.0</td><td>35.0</td><td>15.0</td><td>35.0</td></tr><tr><td>Hulu-Med-7B</td><td>7.5</td><td>48.0</td><td>3.6</td><td>99.0</td><td>30.0</td><td>28.0</td><td>15.0</td></tr><tr><td>Hulu-Med-32B</td><td>21.8</td><td>46.6</td><td>10.1</td><td>99.0</td><td>52.0</td><td>35.0</td><td>16.0</td></tr><tr><td>Lingshu-7B</td><td>9.3</td><td>6.8</td><td>0.6</td><td>99.0</td><td>35.0</td><td>33.0</td><td>35.0</td></tr><tr><td>Lingshu-32B</td><td>11.4</td><td>50.0</td><td>5.7</td><td>99.0</td><td>35.0</td><td>35.0</td><td>35.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>4.7</td><td>33.1</td><td>1.6</td><td>99.0</td><td>35.0</td><td>35.0</td><td>35.0</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>36.2</td><td>20.6</td><td>7.4</td><td>99.0</td><td>35.0</td><td>33.0</td><td>35.0</td></tr><tr><td>Qwen3.5-9B</td><td>38.9</td><td>36.6</td><td>14.3</td><td>99.0</td><td>36.0</td><td>39.0</td><td>26.0</td></tr><tr><td>Qwen3.5-27B</td><td>47.6</td><td>48.9</td><td>23.3</td><td>99.0</td><td>47.0</td><td>29.0</td><td>34.0</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>50.3</td><td>30.3</td><td>15.3</td><td>99.0</td><td>37.0</td><td>30.0</td><td>43.0</td></tr><tr><td>Qwen3.6-27B</td><td>41.4</td><td>46.2</td><td>19.1</td><td>99.0</td><td>50.0</td><td>32.0</td><td>35.0</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>44.8</td><td>31.4</td><td>14.1</td><td>99.0</td><td>36.0</td><td>34.0</td><td>35.0</td></tr><tr><td>InternVL3.5-38B</td><td>3.7</td><td>45.0</td><td>1.7</td><td>99.0</td><td>35.0</td><td>15.0</td><td>33.0</td></tr><tr><td>GPT-5.1</td><td>26.3</td><td>27.0</td><td>7.1</td><td>99.0</td><td>34.0</td><td>36.0</td><td>39.0</td></tr><tr><td>GPT-4.1</td><td>28.3</td><td>22.5</td><td>6.4</td><td>99.0</td><td>35.0</td><td>27.0</td><td>30.0</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>72.0</td><td>65.0</td><td>46.8</td><td>99.0</td><td>49.0</td><td>36.0</td><td>34.0</td></tr><tr><td>Gemini-2.5-Pro</td><td>48.6</td><td>47.1</td><td>22.9</td><td>99.0</td><td>39.0</td><td>35.0</td><td>33.0</td></tr><tr><td>Gemini-2.5-Flash</td><td>24.5</td><td>52.3</td><td>12.8</td><td>99.0</td><td>45.0</td><td>32.0</td><td>34.0</td></tr><tr><td>Claude-Sonnet-4.6</td><td>27.9</td><td>39.8</td><td>11.1</td><td>99.0</td><td>32.0</td><td>37.0</td><td>33.0</td></tr></table>

Direct T is the patient-level T-stage answer generated by the evaluated model from the L3 patient prompt. Rule-slot T applies the deterministic structure rule described above to the evidence used in each experiment. The LLM-slot experiment uses a separate set of model-generated evidence summaries as input to GPT-OSS-120B. Its Rule-slot column applies the deterministic rule to those same summaries; Table A17 uses the primary L3 responses. The gap between the 99.0 gold-evidence ceiling and model-evidence LLM-slot scores shows how much the same downstream stager depends on the quality of its structured input.

Table A18: LLM-stager ablation for structure-evidence-based MRI T staging. All 21 models contribute evidence summaries for the same 100 patients. A fixed GPT-OSS-120B stager infers T stage from each model’s anatomical labels, invasion predictions, and aggregated votes. Gold 18-structure evidence yields 99.0 LLM-slot T and 99.3 balanced T from these anatomical findings.
<table><tr><td>Model</td><td>Direct T</td><td>Rule-slot T</td><td>LLM-slot T</td><td>Bal. LLM T</td></tr><tr><td>Lingshu-7B</td><td>35.0</td><td>33.0</td><td>32.0</td><td>23.8</td></tr><tr><td>Lingshu-32B</td><td>35.0</td><td>35.0</td><td>35.0</td><td>25.0</td></tr><tr><td>InternVL3.5-38B</td><td>33.0</td><td>15.0</td><td>15.0</td><td>25.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>35.0</td><td>35.0</td><td>35.0</td><td>25.0</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>35.0</td><td>33.0</td><td>33.0</td><td>23.5</td></tr><tr><td>HealthGPT-Pro-4B</td><td>39.0</td><td>26.0</td><td>26.0</td><td>25.2</td></tr><tr><td>HealthGPT-Pro-8B</td><td>33.0</td><td>23.0</td><td>23.0</td><td>25.0</td></tr><tr><td>Hulu-Med-4B</td><td>35.0</td><td>15.0</td><td>15.0</td><td>25.0</td></tr><tr><td>Hulu-Med-7B</td><td>15.0</td><td>28.0</td><td>29.0</td><td>29.3</td></tr><tr><td>Qwen3.5-9B</td><td>26.0</td><td>39.0</td><td>39.0</td><td>27.9</td></tr><tr><td>Qwen3.5-27B</td><td>34.0</td><td>29.0</td><td>29.0</td><td>20.7</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>43.0</td><td>30.0</td><td>30.0</td><td>21.4</td></tr><tr><td>Qwen3.6-27B</td><td>35.0</td><td>32.0</td><td>32.0</td><td>22.9</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>35.0</td><td>34.0</td><td>34.0</td><td>24.3</td></tr><tr><td>Hulu-Med-32B</td><td>16.0</td><td>35.0</td><td>34.0</td><td>24.3</td></tr><tr><td>GPT-5.1</td><td>39.0</td><td>36.0</td><td>36.0</td><td>25.7</td></tr><tr><td>GPT-4.1</td><td>30.0</td><td>27.0</td><td>27.0</td><td>19.3</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>34.0</td><td>36.0</td><td>36.0</td><td>25.7</td></tr><tr><td>Gemini-2.5-Pro</td><td>33.0</td><td>35.0</td><td>35.0</td><td>25.0</td></tr><tr><td>Gemini-2.5-Flash</td><td>34.0</td><td>32.0</td><td>29.0</td><td>20.7</td></tr><tr><td>Claude-Sonnet-4.6</td><td>33.0</td><td>37.0</td><td>36.0</td><td>27.6</td></tr></table>

## C.3.5 CarePath Evaluation and Cohort Composition

The CarePath cohort contains 100 patients, 811 clinical context entries, and 217 scored care decisions. Each model provides 100 pathology predictions, 100 MRI predictions, and 217 care answers. Table A19 reports component performance and patient-level outcomes; Table A20 gives their intersections. Figure A5 shows how requiring pathology, T stage, and care answers to be correct together narrows the set of successful patients.

Cohort selection and modality linkage. HealthGPT-Pro-8B, Qwen3-VL-8B, Qwen3-VL-30B-A3B, and Qwen3.5-27B participated in cohort and decision selection. They are marked in the results tables, and the other 17 models were not used in cohort construction. The 25 episodes with matched pathology contribute 115 decisions; the 75 with assigned specimens contribute 102. MRI and clinical records are patient-linked throughout. All assigned specimens are labelled NPC-positive; the corresponding episode records identify their specimen source.

Selection used the four construction models’ care answers with gold pathology and MRI evidence and reference history. Twenty-five patients with matched multimodal data were included. Among the other 86 reviewed patients, those with at least one construction-model error were ranked first by total errors across models and checkpoints, then by the number of checkpoints missed by at least two models, checkpoints missed by any model, maximum errors per checkpoint, and mean errors per checkpoint; a fixed source identifier broke remaining ties. The first 75 patients completed the CarePath cohort. Within these 100 patients, checkpoints answered correctly by all four models were

Gemini-3.1 GPT-5.1 Claude-4.6 Qwen3.6-27B removed from scoring. This procedure yields 217 scored decisions, including a 93-decision subset missed by at least two construction models. Selection therefore targets clinically relevant decisions that these models found difficult.

Question review and scoring depth. Six treatment questions are excluded from scoring because the source records do not establish a unique treatment choice at the specified checkpoint. The identified issues include combining ordered and administered medicines and incomplete evidence of radiotherapy. Their exclusion reasons accompany the data, and all 100 patients remain in the scored cohort. The scored set contains 52/23/3/7/8/7 patients with 1/2/3/4/5/6 care decisions, respectively. The 75 assigned-specimen episodes contain no scored initial-treatment-selection question; those questions occur in the matched-pathology group.

Clinical inputs and endpoints. Care prompts incorporate predicted pathology and T-stage labels where specified: 207 reference predicted pathology, and 133 retain supplied N/M/overall-stage information. Prior care answers are not supplied in the primary protocol. Strict success requires correct pathology, correct MRI T stage, and every selected care answer for a patient. Care accuracy weights the 217 decisions equally; all-care correctness and strict success weight the 100 patients equally. MRI structure predictions and the 811 clinical context entries support the evaluation but are not additional strict-success criteria.

![](images/6556c28495eca84e830c39c026ac5b39140068acd26c6396eccf0bd18146a179.jpg)

![](images/791e6c21d41989c8c59a4fe33983f9718660a2d6b36928f041eb41c7b4259c23.jpg)

![](images/078b583fd25288604950b813196062dcabdc23b66eeecc55b70321c5dbb2074b.jpg)  
Always T3: 58% accuracy | Always positive: 99% accuracy

Figure A4: Class balance and scoring depth in the CarePath cohort. T1–T4 counts are 7/12/58/23; pathology labels are 99 positive and one negative. For 1/2/3/4/5/6 scored care decisions, patient counts are 52/23/3/7/8/7. These distributions belong to the selected L6 cohort; L3 has different populations and a different MRI staging edition.  
![](images/d639df1108b1b8af1826423e77b9b8d1f39d450fea2581cbcd8d9a70e4c61c39.jpg)

![](images/afd6ef1c93b671a974c72f30af96359a321d860a60959a6a2621425efbc76281.jpg)

![](images/5154723f8602cf05beec5fd27ed614e7512c1eef2d77336a0976c95b0feec6b7.jpg)  
Figure A5: Local and joint correctness in the CarePath cohort. (a) Care accuracy uses 217 decisions; strict success uses 100 patients. (b) Nested intersections count the same patients: correct pathology, correct pathology and T stage, then all selected care answers as well. Each successive requirement narrows the set of successful patients. (c) Exact binomial 95% intervals for the four representative models. Panels (a,b) include all 17 models not used in selection; the remaining 13 are gray. All 21 model scores appear in Table A19.

Table A19: CarePath cohort outcomes for all 21 models. T and balanced T use 100 patients, care uses 217 scored care decisions, and all-care/strict use 100 patients. Brackets are 95% intervals: patient-cluster bootstrap for care, exact binomial for strict. †: construction models.
<table><tr><td>Model</td><td>T</td><td>Bal. T</td><td>Care [95% CI]</td><td>All-care</td><td>Strict [95% CI]</td></tr><tr><td>HealthGPT-4B</td><td>10.0</td><td>9.3</td><td>19.8 [14.4, 25.8]</td><td>8.0</td><td>0.0 [0.0, 3.6]</td></tr><tr><td>HealthGPT-8B†</td><td>4.0</td><td>14.3</td><td>20.3 [15.3, 25.5]</td><td>5.0</td><td>0.0 [0.0, 3.6]</td></tr><tr><td>Hulu-Med-4B</td><td>8.0</td><td>8.4</td><td>39.6 [32.5, 46.7]</td><td>15.0</td><td>0.0 [0.0, 3.6]</td></tr><tr><td>Hulu-Med-7B</td><td>4.0</td><td>14.3</td><td>39.6 [33.3, 46.2]</td><td>19.0</td><td>0.0 [0.0, 3.6]</td></tr><tr><td>Hulu-Med-32B</td><td>52.0</td><td>26.3</td><td>47.5 [41.3, 53.5]</td><td>19.0</td><td>10.0 [4.9, 17.6]</td></tr><tr><td>Lingshu-7B</td><td>45.0</td><td>26.3</td><td>44.2 [37.3, 50.8]</td><td>16.0</td><td>9.0 [4.2, 16.4]</td></tr><tr><td>Lingshu-32B</td><td>19.0</td><td>27.5</td><td>55.3 [48.8, 61.4]</td><td>24.0</td><td>2.0 [0.2, 7.0]</td></tr><tr><td>Qwen3-VL-8B†</td><td>21.0</td><td>26.0</td><td>52.5 [46.2, 58.6]</td><td>19.0</td><td>2.0 [0.2, 7.0]</td></tr><tr><td>Qwen3-VL-30B†</td><td>23.0</td><td>25.0</td><td>47.5 [40.5, 53.7]</td><td>14.0</td><td>6.0 [2.2, 12.6]</td></tr><tr><td>Qwen3.5-9B</td><td>33.0</td><td>30.1</td><td>53.0 [46.0, 59.3]</td><td>17.0</td><td>1.0 [0.0, 5.4]</td></tr><tr><td>Qwen3.5-27B†</td><td>15.0</td><td>25.4</td><td>62.7 [55.0, 69.3]</td><td>30.0</td><td>1.0 [0.0, 5.4]</td></tr><tr><td>Qwen3.5-35B</td><td>11.0</td><td>27.4</td><td>63.1 [56.0, 69.4]</td><td>27.0</td><td>1.0 [0.0, 5.4]</td></tr><tr><td>Qwen3.6-27B</td><td>18.0</td><td>24.8</td><td>65.9 [58.5, 72.3]</td><td>33.0</td><td>5.0 [1.6, 11.3]</td></tr><tr><td>Qwen3.6-35B</td><td>48.0</td><td>22.0</td><td>61.8 [53.9, 68.6]</td><td>26.0</td><td>5.0 [1.6, 11.3]</td></tr><tr><td>InternVL3.5-38B</td><td>37.0</td><td>30.5</td><td>47.5 [41.5, 53.5]</td><td>24.0</td><td>0.0 [0.0, 3.6]</td></tr><tr><td>GPT-5.1</td><td>39.0</td><td>28.7</td><td>71.0 [64.4, 77.0]</td><td>46.0</td><td>2.0 [0.2, 7.0]</td></tr><tr><td>GPT-4.1</td><td>22.0</td><td>21.9</td><td>61.8 [54.7, 68.0]</td><td>29.0</td><td>2.0 [0.2, 7.0]</td></tr><tr><td>Gemini-3.1</td><td>19.0</td><td>22.4</td><td>89.9 [85.4, 93.9]</td><td>83.0</td><td>11.0 [5.6, 18.8]</td></tr><tr><td>Gemini-2.5-Pro</td><td>26.0</td><td>19.1</td><td>80.6 [75.0, 86.2]</td><td>71.0</td><td>15.0 [8.6, 23.5]</td></tr><tr><td>Gemini-2.5-Flash</td><td>22.0</td><td>22.6</td><td>65.0 [58.4, 71.8]</td><td>47.0</td><td>6.0 [2.2, 12.6]</td></tr><tr><td>Claude-4.6</td><td>40.0</td><td>24.1</td><td>80.6 [75.3, 85.6]</td><td>64.0</td><td>23.0 [15.2, 32.5]</td></tr></table>

Table A20: CarePath cohort patient-level joint states for all 21 models. C: patients with all selected care answers correct; ${ \mathcal { F } } \colon$ patients with both image labels correct. Entries are set cardinalities; complements are within the same 100 patients. The four intersection counts sum to 100. † denotes construction models.
<table><tr><td>Model</td><td>|C|</td><td>|F|</td><td>|C∩F|</td><td> $| \mathcal { C } \cap \mathcal { F } ^ { c } |$ </td><td> $| { \mathcal { C } } ^ { c } \cap { \mathcal { F } } |$ </td><td> $| \mathcal { C } ^ { c } \cap \mathcal { F } ^ { c } |$ </td></tr><tr><td>HealthGPT-4B</td><td>8</td><td>9</td><td>0</td><td>8</td><td>9</td><td>83</td></tr><tr><td>HealthGPT-8B†</td><td>5</td><td>4</td><td>0</td><td>5</td><td>4</td><td>91</td></tr><tr><td>Hulu-Med-4B</td><td>15</td><td>8</td><td>0</td><td>15</td><td>8</td><td>77</td></tr><tr><td>Hulu-Med-7B</td><td>19</td><td>1</td><td>0</td><td>19</td><td>1</td><td>80</td></tr><tr><td>Hulu-Med-32B</td><td>19</td><td>48</td><td>10</td><td>9</td><td>38</td><td>43</td></tr><tr><td>Lingshu-7B</td><td>16</td><td>45</td><td>9</td><td>7</td><td>36</td><td>48</td></tr><tr><td>Lingshu-32B</td><td>24</td><td>15</td><td>2</td><td>22</td><td>13</td><td>63</td></tr><tr><td>Qwen3-VL-8B†</td><td>19</td><td>14</td><td>2</td><td>17</td><td>12</td><td>69</td></tr><tr><td>Qwen3-VL-30B-A3B†</td><td>14</td><td>22</td><td>6</td><td>8</td><td>16</td><td>70</td></tr><tr><td>Qwen3.5-9B</td><td>17</td><td>5</td><td>1</td><td>16</td><td>4</td><td>79</td></tr><tr><td>Qwen3.5-27B†</td><td>30</td><td>8</td><td>1</td><td>29</td><td>7</td><td>63</td></tr><tr><td>Qwen3.5-35B</td><td>27</td><td>5</td><td>1</td><td>26</td><td>4</td><td>69</td></tr><tr><td>Qwen3.6-27B</td><td>33</td><td>10</td><td>5</td><td>28</td><td>5</td><td>62</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>26</td><td>16</td><td>5</td><td>21</td><td>11</td><td>63</td></tr><tr><td>InternVL3.5-38B</td><td>24</td><td>7</td><td>0</td><td>24</td><td>7</td><td>69</td></tr><tr><td>GPT-5.1</td><td>46</td><td>9</td><td>2</td><td>44</td><td>7</td><td>47</td></tr><tr><td>GPT-4.1</td><td>29</td><td>7</td><td>2</td><td>27</td><td>5</td><td>66</td></tr><tr><td>Gemini-3.1</td><td>83</td><td>16</td><td>11</td><td>72</td><td>5</td><td>12</td></tr><tr><td>Gemini-2.5 Pro</td><td>71</td><td>23</td><td>15</td><td>56</td><td>8</td><td>21</td></tr><tr><td>Gemini-2.5 Flash</td><td>47</td><td>20</td><td>6</td><td>41</td><td>14</td><td>39</td></tr><tr><td>Claude-4.6</td><td>64</td><td>38</td><td>23</td><td>41</td><td>15</td><td>21</td></tr></table>

Table A21: CarePath cohort outcomes by pathology source. Matched specimens: 25 patients and 115 care decisions; assigned specimens: 75 patients and 102 decisions. Within each group, columns give care accuracy, all-care correctness, both image labels correct, and strict success (%). † marks construction models.
<table><tr><td rowspan="2">Model</td><td colspan="4">Matched specimens (n = 25)</td><td colspan="4">Assigned specimens (n = 75)</td></tr><tr><td>Care</td><td>All-care</td><td>Image</td><td>Strict</td><td>Care</td><td>All-care</td><td>Image</td><td>Strict</td></tr><tr><td>Claude-4.6</td><td>84.3</td><td>52.0</td><td>40.0</td><td>12.0</td><td>76.5</td><td>68.0</td><td>37.3</td><td>26.7</td></tr><tr><td>GPT-4.1</td><td>75.7</td><td>32.0</td><td>4.0</td><td>0.0</td><td>46.1</td><td>28.0</td><td>8.0</td><td>2.7</td></tr><tr><td>GPT-5.1</td><td>77.4</td><td>32.0</td><td>16.0</td><td>4.0</td><td>63.7</td><td>50.7</td><td>6.7</td><td>1.3</td></tr><tr><td>Gemini-2.5-Flash</td><td>60.0</td><td>8.0</td><td>28.0</td><td>0.0</td><td>70.6</td><td>60.0</td><td>17.3</td><td>8.0</td></tr><tr><td>Gemini-2.5-Pro</td><td>73.9</td><td>28.0</td><td>28.0</td><td>8.0</td><td>88.2</td><td>85.3</td><td>21.3</td><td>17.3</td></tr><tr><td>Gemini-3.1</td><td>83.5</td><td>44.0</td><td>20.0</td><td>0.0</td><td>97.1</td><td>96.0</td><td>14.7</td><td>14.7</td></tr><tr><td>HealthGPT-4B</td><td>10.4</td><td>0.0</td><td>36.0</td><td>0.0</td><td>30.4</td><td>10.7</td><td>0.0</td><td>0.0</td></tr><tr><td>HealthGPT-8B†</td><td>16.5</td><td>0.0</td><td>4.0</td><td>0.0</td><td>24.5</td><td>6.7</td><td>4.0</td><td>0.0</td></tr><tr><td>Hulu-Med-32B</td><td>50.4</td><td>4.0</td><td>24.0</td><td>0.0</td><td>44.1</td><td>24.0</td><td>56.0</td><td>13.3</td></tr><tr><td>Hulu-Med-4B</td><td>40.9</td><td>8.0</td><td>12.0</td><td>0.0</td><td>38.2</td><td>17.3</td><td>6.7</td><td>0.0</td></tr><tr><td>Hulu-Med-7B</td><td>35.7</td><td>4.0</td><td>4.0</td><td>0.0</td><td>44.1</td><td>24.0</td><td>0.0</td><td>0.0</td></tr><tr><td>InternVL3.5-38B</td><td>46.1</td><td>4.0</td><td>28.0</td><td>0.0</td><td>49.0</td><td>30.7</td><td>0.0</td><td>0.0</td></tr><tr><td>Lingshu-32B</td><td>62.6</td><td>12.0</td><td>20.0</td><td>4.0</td><td>47.1</td><td>28.0</td><td>13.3</td><td>1.3</td></tr><tr><td>Lingshu-7B</td><td>48.7</td><td>4.0</td><td>8.0</td><td>0.0</td><td>39.2</td><td>20.0</td><td>57.3</td><td>12.0</td></tr><tr><td>Qwen3-VL-30B†</td><td>55.7</td><td>4.0</td><td>28.0</td><td>0.0</td><td>38.2</td><td>17.3</td><td>20.0</td><td>8.0</td></tr><tr><td>Qwen3-VL-8B†</td><td>60.9</td><td>8.0</td><td>8.0</td><td>0.0</td><td>43.1</td><td>22.7</td><td>16.0</td><td>2.7</td></tr><tr><td>Qwen3.5-27B†</td><td>78.3</td><td>40.0</td><td>4.0</td><td>0.0</td><td>45.1</td><td>26.7</td><td>9.3</td><td>1.3</td></tr><tr><td>Qwen3.5-35B</td><td>78.3</td><td>28.0</td><td>8.0</td><td>4.0</td><td>46.1</td><td>26.7</td><td>4.0</td><td>0.0</td></tr><tr><td>Qwen3.5-9B</td><td>66.1</td><td>16.0</td><td>8.0</td><td>0.0</td><td>38.2</td><td>17.3</td><td>4.0</td><td>1.3</td></tr><tr><td>Qwen3.6-27B</td><td>81.7</td><td>44.0</td><td>0.0</td><td>0.0</td><td>48.0</td><td>29.3</td><td>13.3</td><td>6.7</td></tr><tr><td>Qwen3.6-35B</td><td>79.1</td><td>40.0</td><td>8.0</td><td>8.0</td><td>42.2</td><td>21.3</td><td>18.7</td><td>4.0</td></tr></table>

## C.4 Controlled Input Comparisons

Table A22: Design of diagnostic comparisons. The O1 comparison applies different judges to the same Expanded Audit response set.
<table><tr><td>Control</td><td>Held fixed</td><td>Changed</td><td>Reported endpoint</td></tr><tr><td>L6 image evidence (25 pa- Patients, care decisions, tients)</td><td>supplied clinical context</td><td>Pathology, MRI T stage, and Patient-macro care invaded structures</td><td>difference</td></tr><tr><td>L5 history (100 patients)</td><td>Baseline/current evidence, question, options, reference</td><td>Historical time markers and Turn-accuracy difference record ordering</td><td></td></tr><tr><td>L1 options (71 pairs)</td><td>answer Physician-audited target knowledge</td><td>MCQ versus option-free answer</td><td>Paired item-accuracy difference</td></tr><tr><td>MRI interface (25 patients) Model, patients, matched</td><td>image frames</td><td>Static panel versus cine presentation</td><td>T and structure differences</td></tr><tr><td>O1 Expanded Audit (1,013 Responses, criteria, labels)</td><td>physician labels</td><td>Automated judge</td><td>Agreement and polarity-specific agreement</td></tr></table>

Reference image-evidence comparison. This experiment asks whether incorrect image-derived findings account for errors in care answers. Fourteen local models answer the same 146 care questions for 25 patients within the 100-patient CarePath cohort, using predicted or reference pathology, MRI T stage, and invaded structures. Supplied N/M and overall-stage fields remain fixed. These patients have patient-matched pathology, MRI, and clinical records and contribute 115 decisions to the main benchmark; the paired analysis uses its fixed 146-question set. Additional conditions replace pathology or MRI findings separately or vary the supplied history. Table A23 reports joint evidence replacement with clinical context held fixed, using mean within-patient care accuracy and 10,000 paired patient-bootstrap draws. The endpoint scores care answers alone, so replacing an incorrect image label raises the score only if a care answer also improves.

Table A23: L6 image-evidence replacement: scores and patient changes. Predicted and reference pathology, MRI T stage, and invasion findings are compared on the same 25 selected patients and 146 care decisions. Scores are patient-macro care accuracy (%); differences and paired patient-bootstrap 95% intervals are in pp. B/W/S counts patients with better/worse/unchanged care accuracy under reference inputs. Figure 7d plots baseline accuracy against the paired difference.
<table><tr><td>Row</td><td>Model</td><td>Predicted</td><td>Reference</td><td>Difference [95% CI]</td><td>B/W/S</td></tr><tr><td>1</td><td>HealthGPT-8B</td><td>21.1</td><td>17.9</td><td>-3.2 [-10.5, 3.8]</td><td>8/10/7</td></tr><tr><td>2</td><td>Lingshu-7B</td><td>54.4</td><td>53.0</td><td>-1.4 [-5.0, 1.8]</td><td>2/3/20</td></tr><tr><td>3</td><td>Hulu-Med-4B</td><td>48.0</td><td>46.8</td><td>-1.3 [-3.9, 1.3]</td><td>1/3/21</td></tr><tr><td>4</td><td>HealthGPT-4B</td><td>23.9</td><td>22.9</td><td>-1.0 [-9.2, 8.0]</td><td>7/9/9</td></tr><tr><td>5</td><td>Qwen3.6-27B</td><td>85.5</td><td>84.6</td><td>-0.9 [-4.4, 2.4]</td><td>2/3/20</td></tr><tr><td>6</td><td>Qwen3.6-35B-A3B</td><td>84.2</td><td>84.5</td><td>+0.2 [-2.5, 2.9]</td><td>2/2/21</td></tr><tr><td>7</td><td>Qwen3.5-27B</td><td>82.6</td><td>83.1</td><td>+0.5 [-2.7, 3.8]</td><td>3/3/19</td></tr><tr><td>8</td><td>Lingshu-32B</td><td>68.4</td><td>69.9</td><td>+1.5 [-1.7, 4.9]</td><td>4/2/19</td></tr><tr><td>9</td><td>Qwen3.5-9B</td><td>71.0</td><td>73.5</td><td>+2.5 [-0.5, 5.7]</td><td>4/1/20</td></tr><tr><td>10</td><td>Qwen3.5-35B</td><td>79.0</td><td>81.9</td><td>+2.8 [-1.0, 6.6]</td><td>5/1/19</td></tr><tr><td>11</td><td>Hulu-Med-7B</td><td>36.9</td><td>40.1</td><td>+3.2 [-0.6, 7.0]</td><td>6/2/17</td></tr><tr><td>12</td><td>Hulu-Med-32B</td><td>56.8</td><td>60.2</td><td>+3.4 [0.8, 6.4]</td><td>5/0/20</td></tr><tr><td>13</td><td>Qwen3-VL-30B-A3B</td><td>64.6</td><td>68.5</td><td>+3.8 [-1.0, 8.6]</td><td>6/1/18</td></tr><tr><td>14</td><td>Qwen3-VL-8B</td><td>67.8</td><td>71.8</td><td>+4.1 [0.7, 7.8]</td><td>6/1/18</td></tr></table>

History representation and record order. The history comparisons separate two changes that might affect a clinical decision: removing temporal markers and changing the order in which records are presented. The snapshot removes relative-day and admission markers, then groups the historical records by content. Baseline and current findings, the question, options, and reference answer remain unchanged. Although the records match once temporal markers are ignored, their clinical timing is no longer explicit; this contrast combines loss of timing with reorganization. The order-only condition instead shuffles complete lines within their original sections, retaining every time marker.

Twenty models each answer the same 305 questions for 100 patients in all three conditions. These controls use condition-matched generations, with a full-history baseline evaluated alongside each altered representation. Full-history minus snapshot accuracy is positive for 14 models; only DeepSeek V3.1 has a patient-bootstrap interval wholly above zero (Table A25). In the order-only comparison, 19 intervals include zero, while Gemini-3.1 improves by 3.0 pp (95% CI: 0.6–5.7; Table A26). Both analyses use 10,000 paired patient-bootstrap draws, with intervals unadjusted across models. Figure A6 places the full-history comparison alongside the image-evidence comparison, showing both the direction and uncertainty of each paired difference.

The history-dependent subset contains 118 turns from 69 patients and compares full history with current-state information alone (Table A24). A separate Gemini-3.1 experiment evaluates packedhistory, persistent, and stateless interaction on 292 turns from 87 patients. Persistent and stateless turn accuracy are 82.5% and 83.6%, a difference of −1.0 pp (95% CI: −3.8 to 1.7). The interval includes zero.

Table A24: History-dependent subset: 118 turns from 69 patients. Full history is compared with current-state information alone. Turn accuracy is in percent; differences and paired patient-bootstrap 95% intervals are in pp.
<table><tr><td>Model</td><td>History</td><td>Current state</td><td>Difference [95% CI]</td></tr><tr><td>HealthGPT-8B</td><td>24.6</td><td>28.0</td><td>-3.4 [-9.7, 2.8]</td></tr><tr><td>Lingshu-7B</td><td>62.7</td><td>63.6</td><td>-0.8 [-8.4, 6.4]</td></tr><tr><td>Qwen3-VL-8B</td><td>47.5</td><td>50.8</td><td>-3.4 [-12.4, 5.5]</td></tr><tr><td>Qwen3.5-9B</td><td>46.6</td><td>44.1</td><td>+2.5 [-6.3, 11.7]</td></tr></table>

Paired decision changes. Across M matched turns, let u count incorrect-to-correct changes and v correct-to-incorrect changes. Net accuracy change ∆ in percentage points and correctness-flip rate Φ

## in percent are

$$
\Delta = 1 0 0 \frac { u - v } { M } , \qquad \Phi = 1 0 0 \frac { u + v } { M } .\tag{A6}
$$

For the full-history versus snapshot L5 comparison, M = 305 and each condition covers the same 100 patients. All-turn gain is the difference between the two conditions’ proportions of patients with every scored turn correct; it is not derived from u and v alone.

For L1, MRI T staging, and L5, a changed unit switches between correct and incorrect across the paired conditions. For the 25-case L6 control, a changed patient has different within-patient care accuracy. These counts capture offsetting improvements and regressions. GPT-5.1 changes correctness on 41 of 305 L5 turns (22 corrected, 19 regressed), with a net gain of 1.0 pp and an all-turn success change of −1.0 pp. Under gold pathology and MRI evidence, HealthGPT-8B improves care accuracy for eight patients and worsens it for ten; the remaining seven are unchanged. The L6 net difference averages the magnitudes of the within-patient accuracy changes.

Table A25: L5 full history versus the time-stripped snapshot. The snapshot removes historical time markers and regroups records (305 turns, 100 patients). Accuracy is in percent and differences in pp, with paired patient-bootstrap 95% intervals. +/− count turns corrected/harmed by full history.
<table><tr><td>Model</td><td>History</td><td>Snapshot</td><td>Difference [95% CI]</td><td>+/-</td></tr><tr><td>DeepSeek-V3.1</td><td>71.5</td><td>66.6</td><td>+4.9 [1.4, 8.5]</td><td>22/7</td></tr><tr><td>Fleming-R1-7B</td><td>59.7</td><td>61.3</td><td>-1.6 [-6.5, 3.2]</td><td>21/26</td></tr><tr><td>GPT-4.1</td><td>75.1</td><td>75.7</td><td>-0.7 [-4.1, 2.9]</td><td>14/16</td></tr><tr><td>GPT-5.1</td><td>71.5</td><td>70.5</td><td>+1.0 [-3.2, 5.2]</td><td>22/19</td></tr><tr><td>GPT-OSS-120B</td><td>48.9</td><td>45.6</td><td>+3.3 [-1.6, 8.1]</td><td>29/19</td></tr><tr><td>GPT-OSS-20B</td><td>52.8</td><td>48.2</td><td>+4.6 [-1.4, 10.7]</td><td>46/32</td></tr><tr><td>Gemini-3.1</td><td>89.5</td><td>86.2</td><td>+3.3 [0.0, 6.4]</td><td>18/8</td></tr><tr><td>HealthGPT-4B</td><td>36.4</td><td>40.0</td><td>-3.6 [-8.9, 1.6]</td><td>29/40</td></tr><tr><td>HealthGPT-8B</td><td>12.5</td><td>11.8</td><td>+0.7 [-2.5, 3.9]</td><td>13/11</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>45.6</td><td>46.2</td><td>-0.7 [-4.5, 3.1]</td><td>14/16</td></tr><tr><td>Hulu-Med-32B</td><td>69.5</td><td>68.5</td><td>+1.0 [-1.6, 3.7]</td><td>9/6</td></tr><tr><td>Hulu-Med-4B</td><td>58.0</td><td>56.1</td><td>+2.0 [-1.6, 5.5]</td><td>17/11</td></tr><tr><td>Hulu-Med-7B</td><td>43.9</td><td>41.6</td><td>+2.3 [-1.0, 5.4]</td><td>15/8</td></tr><tr><td>Lingshu-7B</td><td>53.4</td><td>51.1</td><td>+2.3 [-0.3, 5.0]</td><td>12/5</td></tr><tr><td>Qwen3-30B</td><td>60.0</td><td>58.4</td><td>+1.6 [-1.3, 4.4]</td><td>14/9</td></tr><tr><td>Qwen3-VL-8B</td><td>68.2</td><td>66.9</td><td>+1.3 [-1.4, 4.1]</td><td>11/7</td></tr><tr><td>Qwen3.5-27B</td><td>78.4</td><td>78.0</td><td>+0.3 [-2.0, 2.6]</td><td>8/7</td></tr><tr><td>Qwen3.5-35B</td><td>73.8</td><td>71.1</td><td>+2.6 [-0.3, 5.6]</td><td>14/6</td></tr><tr><td>Qwen3.5-9B</td><td>67.9</td><td>68.9</td><td>-1.0 [-4.1, 2.0]</td><td>10/13</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>75.4</td><td>77.7</td><td>-2.3 [-5.1, 0.3]</td><td>6/13</td></tr></table>

A three-condition input example. The accompanying example provides one complete L5 prompt triplet and its record correspondence. Each prompt has the same 46 historical records; 44 carry relative-day or admission markers in the full-history version. Table A27 shows two translated treatment-record excerpts. Drug names, doses, cycle number, and treatment intent are preserved, whereas the snapshot omits the admission/day prefix. The shuffled condition keeps each complete source line and changes its position. Baseline/current findings and all answer options remain identical. Removing the time markers changes the temporal information available for the decision.

Success at different scoring depths. Figure 7c groups L5 patients by 1–2, 3, and 4–5 scored turns (n = 32, 31, 37); panel h groups CarePath cohort patients by 1, 2, and 3–6 scored care decisions (n = 52, 23, 25). For each model and group, success is the number of patients meeting the endpoint divided by the group’s patient count: all-turn correctness for L5 and strict success for the CarePath cohort. The groups differ in both clinical composition and the number of decisions required for complete success.

Serial CarePath interaction. The 25-case serial control uses gold pathology and MRI evidence. For GPT-4.1, GPT-5.1, and Gemini-3.1, persistent-minus-stateless all-care correctness differences are 16, 20, and 8 pp, with paired 95% intervals of [−4,36], [−4,44], and [0,20], respectively. These outcomes assess downstream interaction under corrected pathology and MRI evidence on these 25 selected cases.

Table A26: L5 order-only control on 305 turns from 100 patients. Full lines are shuffled within their sections while retaining time markers. Accuracy is in percent; full-history minus shuffled differences and patient-bootstrap 95% intervals are in pp. +/− count corrected/regressed turns.
<table><tr><td>Model</td><td>History</td><td></td><td>Shuffled Difference [95% CI]</td><td>+/-</td></tr><tr><td>Gemini-3.1</td><td>89.5</td><td>86.6</td><td>+3.0 [0.6, 5.7]</td><td>12/3</td></tr><tr><td>Qwen3.5-27B</td><td>78.4</td><td>79.0</td><td>-0.7 [-2.3, 1.0]</td><td>2/4</td></tr><tr><td>Qwen3.6-35B</td><td>75.4</td><td>76.7</td><td>-1.3 [-3.2, 0.3]</td><td>2/6</td></tr><tr><td>GPT-4.1</td><td>75.1</td><td>77.4</td><td>-2.3 [-5.4, 0.7]</td><td>8/15</td></tr><tr><td>Qwen3.5-35B</td><td>73.8</td><td>74.4</td><td>-0.7 [-3.6, 2.2]</td><td>8/10</td></tr><tr><td>DeepSeek-V3.1</td><td>71.5</td><td>68.9</td><td>+2.6 [-0.3, 5.8]</td><td>16/8</td></tr><tr><td>GPT-5.1</td><td>71.5</td><td>71.8</td><td>-0.3 [-3.6, 2.9]</td><td>16/17</td></tr><tr><td>Hulu-Med-32B</td><td>69.5</td><td>68.5</td><td>+1.0 [-0.7, 2.7]</td><td>5/2</td></tr><tr><td>Qwen3-VL-8B</td><td>68.2</td><td>69.2</td><td>-1.0 [-3.4, 1.4]</td><td>6/9</td></tr><tr><td>Qwen3.5-9B</td><td>67.9</td><td>69.8</td><td>-2.0 [-4.6, 0.3]</td><td>5/11</td></tr><tr><td>Qwen3-30B</td><td>60.0</td><td>61.0</td><td>-1.0 [-3.7, 1.9]</td><td>7/10</td></tr><tr><td>Fleming-R1-7B</td><td>59.7</td><td>61.3</td><td>-1.6 [-6.3, 3.0]</td><td>22/27</td></tr><tr><td>Hulu-Med-4B</td><td>58.0</td><td>59.0</td><td>-1.0 [-3.2, 1.0]</td><td>5/8</td></tr><tr><td>Lingshu-7B</td><td>53.4</td><td>51.8</td><td>+1.6 [-0.6, 3.9]</td><td>9/4</td></tr><tr><td>GPT-OSS-20B</td><td>52.8</td><td>51.8</td><td>+1.0 [-4.9, 6.9]</td><td>38/35</td></tr><tr><td>GPT-OSS-120B</td><td>48.9</td><td>45.9</td><td>+3.0 [-1.7, 7.7]</td><td>32/23</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>45.6</td><td>43.9</td><td>+1.6 [-1.7, 5.2]</td><td>18/13</td></tr><tr><td>Hulu-Med-7B</td><td>43.9</td><td>44.6</td><td>-0.7 [-3.7, 2.3]</td><td>10/12</td></tr><tr><td>HealthGPT-4B</td><td>36.4</td><td>35.7</td><td>+0.7 [-4.1, 5.7]</td><td>39/37</td></tr><tr><td>HealthGPT-8B</td><td>12.5</td><td>12.1</td><td>+0.3 [-2.6, 3.4]</td><td>11/10</td></tr></table>

![](images/d2eea2f1c9aaadab8fa5f12f3f4d6885b35ccce5bd03cfda6056a6bc05934e89.jpg)

![](images/3e5442ec921c57052d48cdb4a4e98346aad2369c7fc45e8b66ebce61f81dc6fc.jpg)  
Figure A6: Paired controls for evidence and history. (a) Gold versus predicted pathology/MRI inputs: difference in patient-macro care accuracy across 14 local models on the 25 selected patients/146 care decisions. (b) Full history versus the time-stripped, regrouped snapshot: difference in turn accuracy across 20 models on 100 patients/305 turns. Dots show paired differences and bars patient-cluster 95% bootstrap intervals; all models in each control are retained.

Table A27: What the L5 input conditions preserve and change. Excerpts from one actual prompt triplet; English translations condense repeated source-field names. Record positions refer to the 46 historical lines, excluding baseline/current findings. Complete original-language prompts and the line correspondence are included in the accompanying example.
<table><tr><td>Condition</td><td>Cisplatin record</td><td>Docetaxel record</td><td>Timing and position</td></tr><tr><td>Full history</td><td>Admission 2, day 26; induction, cycle 2; cisplatin 53.0 mg</td><td>Admission 2, day 26; induction, cycle 2; docetaxel 105.0 mg</td><td>Prefixes retained; records 23 and 24</td></tr><tr><td>Snapshot</td><td>Induction, cycle 2; cisplatin 53.0 mg</td><td>Induction, cycle 2; docetaxel 105.0 mg</td><td>Prefixes removed; records 33 and 42</td></tr><tr><td>Shuffled order</td><td>Admission 2, day 26; induction, cycle 2; cisplatin 53.0 mg</td><td>Admission 2, day 26; induction, cycle 2; docetaxel 105.0 mg</td><td>Full lines retained; records 19 and 29</td></tr></table>

Answer-option sensitivity. Physician review identifies 71 MCQ/very-short-answer pairs with the same target knowledge. Table A28 reports the number correct under each format and the paired difference. This control measures sensitivity to answer options while holding the knowledge target fixed.

Table A28: Physician-audited same-target L1 pairs. MCQ minus option-free accuracy, with paired 95% bootstrap intervals in percentage points. This is an option-cue sensitivity analysis.
<table><tr><td>Model</td><td>MCQ correct</td><td>VSAQ correct</td><td>Difference [95% CI]</td></tr><tr><td>Gemini-3.1</td><td>67/71</td><td>62/71</td><td>7.0 [0.0, 14.1]</td></tr><tr><td>GPT-4.1</td><td>61/71</td><td>52/71</td><td>12.7 [2.8, 22.5]</td></tr><tr><td>GPT-5.1</td><td>65/71</td><td>54/71</td><td>15.5 [5.6, 25.4]</td></tr></table>

MRI presentation. The matched comparison evaluates Qwen3-VL-8B on 25 patients using a static panel, cine composed of the same frames, and an exploratory full-sequence cine condition. T accuracy is 32% in each condition. Same-frame cine minus static accuracy is 0.0 pp (95% CI: −16.0 to 16.0), comprising two wrong-to-right and two right-to-wrong changes. Structure-label micro accuracy changes by 1.3 pp (−9.6 to 11.8). Full-sequence cine changes T accuracy by 0.0 pp and structure accuracy by 2.2 pp (−7.3 to 11.1). The wide intervals leave substantial uncertainty about interface effects. A separate 11-model cine panel evaluates 2,700 card-matched tasks, 200 sequence tasks, and 100 patient-level tasks per model. The supplementary cine dataset provides the videos and their frame correspondence. Primary scoring reports card, sequence, and T-stage accuracy, while auxiliary scoring reports structure and invasion measures. This panel describes cine performance across additional models; the paired static/cine inference rests on the 25-patient comparison above.

## C.5 Consultation Scoring and Results

O1 evaluates 103 consultation responses from each model against 1,726 clinician-defined criteria. Positive requirements measure what an answer should contain; error criteria identify claims or recommendations that warrant deductions.

## C.5.1 Scoring Protocol

Table A29 reports the primary O1 ConsultRubrics results. Each model first generates one free-text consultation response for each of the 103 O1 items. Each response is assessed against its item-specific criteria, yielding 1,726 binary criterion judgments per model. For item i, let $\mathscr { A } _ { i } , \mathcal { E } _ { i } ^ { \mathrm { m a j } }$ , and $\mathcal { E } _ { i } ^ { \mathrm { m i n } }$ be the disjoint sets of positive requirements, major-error criteria, and minor-error criteria. Criterion k has a binary judgment $z _ { i k } \in \{ 0 , 1 \}$ , corresponding to criteria\_met. A positive-criterion judgment of one means the requirement is satisfied; a negative-criterion judgment of one means the error is present. Each positive anchor has clinician-assigned weight $w _ { i k } \ge 0$ in points. The available positive score is $\begin{array} { r } { B _ { i } = \sum _ { k \in \mathcal { A } _ { i } } w _ { i k } = 1 0 } \end{array}$ points for every item.

Each triggered major error deducts two points, and each minor error deducts one point. The raw score $r _ { i } ,$ , normalized score $R _ { i }$ , and primary model score are

$$
\begin{array} { l } { r _ { i } = \displaystyle \operatorname* { m a x } \left\{ 0 , \sum _ { k \in \mathcal { A } _ { i } } w _ { i k } z _ { i k } - 2 \sum _ { k \in \mathcal { E } _ { i } ^ { \operatorname* { m a j } } } z _ { i k } - \sum _ { k \in \mathcal { E } _ { i } ^ { \operatorname* { m i n } } } z _ { i k } \right\} , } \\ { R _ { i } = 1 0 0 \displaystyle \frac { r _ { i } } { B _ { i } } , \qquad \mathrm { O 1 } = \displaystyle \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } R _ { i } . } \end{array}\tag{A7}
$$

Here $\nu$ contains items with valid judgments for every criterion. Thus $r _ { i } \in [ 0 , 1 0 ]$ is measured in points, while $R _ { i }$ and O1 lie in [0, 100]. Deductions accumulate across criteria before the item score is floored at zero. The primary mean weights items equally; a clinical-tag score restricts $\nu$ to items with that tag. All 30 generators have valid judgments for all 103 items, so $\vert  \mathcal { V } \vert = 1 0 3$

Error rates have a different unit from score deductions. For severity $s \in \{ \mathrm { m a j } , \mathrm { m i n } \}$ , let $H _ { i } ^ { s } =$ $\textstyle \sum _ { k \in { \mathcal { E } } _ { i } ^ { s } } z _ { i k }$ count triggered criteria. The item-level error rate is

$$
\mathrm { E r r o r R a t e } _ { s } = \frac { 1 0 0 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \mathbf { 1 } \{ H _ { i } ^ { s } > 0 \} .\tag{A8}
$$

An item with several errors of the same severity contributes once to this percentage, although each error contributes its deduction in Eq. (A7). Figure 5d instead uses criteria as the unit: positive-section rates are the percentage of criteria satisfied, and error-avoidance rates are the percentage of error criteria not triggered. Each criterion counts once in that panel; clinician-assigned weights enter the consultation score in Eq. (A7). In panel (a), r denotes the Pearson correlation across models between major-error rate and O1 score.

The primary judge is GPT-OSS-120B with greedy decoding and medium reasoning effort. Its criterion-specific prompt is shown in section E. All 30 generators use this same judge and scoring rule. Physician calibration separately assesses four judges on shared response–criterion pairs, as described in Appendix D.

## C.5.2 Consultation Scores and Clinical Groups

Table A29 reports the score and the fraction of responses triggering each error severity. GPT-5.1 has the highest mean score, 58.4, but still triggers a major-error criterion in 14.6% of responses. Thus a higher score indicates better coverage and fewer deductions on average, rather than uniformly reliable clinical advice.

The 103 questions belong to 42 cases. The primary mean weights questions equally. Averaging first within each case and then across cases changes model scores by −2.0 to +0.1 points and leaves GPT-5.1 highest, as shown in Figure 5b. This comparison shows how the unequal number of questions per case affects the aggregate score.

Table A29 separates adverse-event management, diagnosis, treatment selection, special populations, and workup. Elderly and pediatric cases are reported separately here and combined in Figure 2h. Group sizes range from two workup questions to 43 treatment questions; these scores describe the included clinical questions rather than equally precise estimates for each area.

Table A29: Consultation scores and error rates under the GPT-OSS-120B judge. Every model uses 103 questions and 1,726 criteria. Major/Minor are percentages of answers with at least one corresponding error. Clinical-group scores cover adverse events (AE; 35 questions), diagnosis (Dx; 11), elderly patients (Eld.; 4), pediatric patients (Ped.; 8), treatment (Tx; 43), and workup (2). All scores lie in 0–100.
<table><tr><td>Model</td><td>Score</td><td>Major</td><td>Minor</td><td>AE</td><td>Dx</td><td>Eld.</td><td>Ped.</td><td>Tx</td><td>Workup</td></tr><tr><td>GPT-5.1</td><td>58.4</td><td>14.6</td><td>40.8</td><td>69.2</td><td>42.6</td><td>85.4</td><td>57.4</td><td>51.2</td><td>60.1</td></tr><tr><td>Gemini-3.1</td><td>47.8</td><td>12.6</td><td>59.2</td><td>58.4</td><td>50.1</td><td>60.5</td><td>35.5</td><td>39.1</td><td>60.1</td></tr><tr><td>Claude-4.6</td><td>47.5</td><td>15.5</td><td>58.3</td><td>52.1</td><td>41.5</td><td>76.7</td><td>49.8</td><td>41.5</td><td>63.3</td></tr><tr><td>Qwen3.6-27B</td><td>47.2</td><td>24.3</td><td>52.4</td><td>61.2</td><td>26.5</td><td>67.5</td><td>39.9</td><td>39.6</td><td>65.1</td></tr><tr><td>Model</td><td>Score</td><td>Major</td><td>Minor</td><td>AE</td><td>Dx</td><td>Eld.</td><td>Ped.</td><td></td><td>Tx Workup</td></tr><tr><td>Qwen3.5-27B</td><td>46.8</td><td>19.4</td><td>50.5</td><td>59.5</td><td>31.7</td><td>61.3</td><td>27.4</td><td>42.2</td><td>56.7</td></tr><tr><td>GPT-OSS-120B</td><td>44.9</td><td>20.4</td><td>49.5</td><td>61.8</td><td>29.5</td><td>51.7</td><td>29.3</td><td>36.6</td><td>59.2</td></tr><tr><td>Qwen3.6-35B</td><td>42.4</td><td>24.3</td><td>56.3</td><td>55.3</td><td>26.9</td><td>88.8</td><td>36.7</td><td>31.9</td><td>60.0</td></tr><tr><td>Qwen3.5-35B</td><td>41.4</td><td>21.4</td><td>61.2</td><td>55.4</td><td>34.7</td><td>60.9</td><td>33.5</td><td>30.5</td><td>63.3</td></tr><tr><td>Baichuan-M2-32B</td><td>38.8</td><td>17.5</td><td>58.3</td><td>53.5</td><td>25.5</td><td>80.0</td><td>35.3</td><td>27.3</td><td>33.3</td></tr><tr><td>Gemini-2.5-Pro</td><td>38.5</td><td>21.4</td><td>67.0</td><td>52.3</td><td>31.8</td><td>70.0</td><td>30.5</td><td>26.9</td><td>50.9</td></tr><tr><td>Qwen3.5-9B</td><td>34.7</td><td>28.2</td><td>65.0</td><td>43.4</td><td>20.0</td><td>58.8</td><td>33.7</td><td>28.0</td><td>62.5</td></tr><tr><td>Gemini-2.5-Flash</td><td>34.4</td><td>26.2</td><td>72.8</td><td>47.9</td><td>28.4</td><td>63.0</td><td>23.2</td><td>24.1</td><td>39.2</td></tr><tr><td>GPT-4.1</td><td>33.3</td><td>25.2</td><td>64.1</td><td>49.1</td><td>24.5</td><td>42.9</td><td>23.2</td><td>22.4</td><td>63.3</td></tr><tr><td>Qwen3-VL-30B</td><td>29.1</td><td>32.0</td><td>68.9</td><td>44.8</td><td>12.4</td><td>54.6</td><td>20.5</td><td>19.4</td><td>40.0</td></tr><tr><td>Fleming-R1-32B</td><td>28.6</td><td>28.2</td><td>68.0</td><td>47.4</td><td>23.8</td><td>45.0</td><td>11.5</td><td>15.0</td><td>53.4</td></tr><tr><td>DeepSeek-V3.1</td><td>28.2</td><td>32.0</td><td>67.0</td><td>40.8</td><td>30.3</td><td>46.7</td><td>14.8</td><td>17.0</td><td>52.6</td></tr><tr><td>GPT-OSS-20B</td><td>28.0</td><td>29.1</td><td>72.8</td><td>50.7</td><td>16.5</td><td>26.3</td><td>13.9</td><td>15.6</td><td>17.5</td></tr><tr><td>Qwen3-30B</td><td>26.8</td><td>35.0</td><td>72.8</td><td>39.4</td><td>17.9</td><td>34.2</td><td>23.1</td><td>18.0</td><td>44.2</td></tr><tr><td>Qwen3-VL-8B</td><td>25.6</td><td>35.0</td><td>78.6</td><td>37.3</td><td>22.9</td><td>41.7</td><td>17.1</td><td>15.1</td><td>61.7</td></tr><tr><td>Fleming-R1-7B</td><td>24.8</td><td>38.8</td><td>74.8</td><td>39.3</td><td>15.0</td><td>69.6</td><td>12.9</td><td>13.6</td><td>21.7</td></tr><tr><td>HealthGPT-8B</td><td>18.8</td><td>36.9</td><td>82.5</td><td>32.0</td><td>24.2</td><td>25.0</td><td>6.4</td><td>8.0</td><td>28.3</td></tr><tr><td>HealthGPT-4B</td><td>14.7</td><td>41.7</td><td>88.3</td><td>27.8</td><td>18.2</td><td>14.2</td><td>2.3</td><td>3.8</td><td>50.9</td></tr><tr><td>Hulu-Med-7B</td><td>10.7</td><td>49.5</td><td>91.3</td><td>21.7</td><td>10.2</td><td>21.3</td><td>9.4</td><td>1.5</td><td>5.0</td></tr><tr><td>HuatuoGPT-o1-72B</td><td>10.1</td><td>43.7</td><td>86.4</td><td>20.1</td><td>17.2</td><td>7.5</td><td>0.5</td><td>2.0</td><td>11.7</td></tr><tr><td>InternVL3.5-38B</td><td>9.5</td><td>48.5</td><td>88.3</td><td>11.7</td><td>16.2</td><td>0.0</td><td>0.6</td><td>6.6</td><td>51.7</td></tr><tr><td>Lingshu-32B</td><td>9.0</td><td>54.4</td><td>89.3</td><td>17.7</td><td>13.0</td><td>10.0</td><td>0.0</td><td>2.6</td><td>5.9</td></tr><tr><td>Hulu-Med-4B</td><td>8.7</td><td>48.5</td><td>90.3</td><td>13.7</td><td>13.5</td><td>31.2</td><td>5.4</td><td>2.1</td><td>5.0</td></tr><tr><td>Hulu-Med-32B</td><td>7.6</td><td>54.4</td><td>93.2</td><td>15.4</td><td>14.7</td><td>0.0</td><td>1.3</td><td>1.3</td><td>10.0</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>5.3</td><td>50.5</td><td>97.1</td><td>10.6</td><td>14.1</td><td>0.0</td><td>0.0</td><td>0.5</td><td>0.0</td></tr><tr><td>Lingshu-7B</td><td>3.2</td><td>59.2</td><td>92.2</td><td>5.9</td><td>9.7</td><td>0.0</td><td>0.0</td><td>0.2</td><td>5.0</td></tr></table>

## D Clinical Review and Judge Calibration

NPC specialists reviewed each item’s clinical question, available evidence, reference answer, and scoring criteria. They checked staging, prior treatment, toxicity, organ function, and guideline assumptions, and standardized drug names, doses, units, and answer options. Future orders and outcomes were removed from inputs when they disclosed the target decision. Question-review outcomes and the six unscored CarePath cohort treatment questions appear in sections A and C.3.5.

## D.1 Clinical Annotation and Adjudication

In routine care, pathology and radiology examinations were initially interpreted by a physician with 1–3 years of experience and finalized by a senior subspecialist with more than five years of experience. Pathology reference labels were taken from finalized reports. For benchmark MRI annotation, another physician independently reviewed the examinations, reports, and staging criteria to assign structure-invasion and T-category labels. Three senior NPC radiation oncologists resolved disagreements by consensus. The independent benchmark review followed routine clinical reporting and preceded consensus adjudication.

Review of clinical decisions. For L2, reviewers checked whether the vignette supplied the facts needed for one guideline decision and whether the answer options supported a determinate choice. Missing variables, leading wording, and inconsistent guideline editions prompted revision. For L4, they assessed whether diagnosis, stage, treatment history, response, and contraindications supported the proposed management plan. The evaluated questions distinguish single-choice and explicit multiple-select items (section A).

For L5, review preserved the order of clinical events after date normalization and restricted each question to evidence available at that checkpoint. Distractors differed in evidence use, treatment timing, toxicity management, or follow-up requirements. For O1, specialists separated essential conclusions, supporting case evidence, and conditional recommendations, then assigned major and minor error penalties according to clinical severity.

## D.2 Physician Calibration Audits

Targeted Audit. For the automated O1 rubric judge, we constructed the Targeted Audit at the same criterion level as the scorer. The packet contained 811 criterion-level labels from 48 modelgenerated responses and was stratified to cover high judge disagreement, large score gaps, clinically safety-sensitive tags, and control cases with expected high or low rubric satisfaction. The physician worksheet marked responses whose completeness required review. Seventeen marked Gemini responses, comprising 263 criterion labels, were excluded because severe truncation prevented adjudication. Marked GPT-5.1 and GPT-OSS-120B responses remained eligible after physician confirmation that their individual criteria could still be judged. This left 548 criterion-level labels for the Targeted Audit agreement analysis. The retained set covers GPT-5.1, Qwen3.6-27B-Instruct, GPT-OSS-120B, and Gemini-3.1-Pro-Preview responses, allowing calibration across both API and local response generators. The 548 labels include 286 from marked responses: 184 GPT-5.1 and 102 GPT-OSS-120B labels. Excluding these responses leaves 262 labels for a sensitivity analysis, which gives the same judge ranking. The Expanded Audit broadens response-generator coverage, as described below.

Qwen3.6-27B criterion judging uses non-thinking mode, temperature 0, top-p 1, and a 768-token output limit. Outputs missing required fields are retried up to four times; three remaining Targeted Audit judgments use JSON-schema-constrained decoding with the same prompts and model.

Expanded Audit. Physicians annotated 1,017 criteria from 54 responses generated by nine models on six cases: 859 labels were collected with generator identity hidden and 158 came from the Targeted Audit. Comparisons use the 1,013 criteria judged by all four models; four criteria without a Qwen3.6-27B judgment are excluded from every judge’s comparison. Agreement is 77.6% for GPT-4.1, 70.5% for GPT-OSS-120B, 76.9% for Qwen3-30B, and 77.8% for Qwen3.6-27B, with Cohen’s κ of 0.547, 0.387, 0.538, and 0.544. The Targeted Audit (548 labels) and Expanded Audit (1,013 labels) are analysed separately because their response compositions differ. The primary O1 judge uses a criterion-specific template with four calibration examples and explicit interpretation of requirements and errors.

## D.3 Agreement Measures and Sensitivity Analyses

Within either calibration set, let $h _ { \ell } , \hat { h } _ { \ell } \in \{ 0 , 1 \}$ be the physician and judge labels for criterion $\ell ,$ and let n be the number of valid paired labels. Define TP, FP, TN, and FN as the counts of $( h _ { \ell } , \hat { h } _ { \ell } ) = ( 1 , 1 ) , ( 0 , 1 ) , ( 0 , 0 ) , ( 1 , 0 )$ , respectively; they sum to n. A physician-positive label means the criterion is true. For a negative error criterion, this means an error is present, not that the response is clinically correct.

Observed agreement $p _ { o } ,$ physician-positive rate $q _ { H }$ , judge-positive rate $q _ { J }$ , chance agreement $p _ { e }$ , and Cohen’s κ are

$$
\begin{array} { c } { { p _ { o } = \displaystyle \frac { \mathrm { T P } + \mathrm { T N } } { n } , \qquad q _ { H } = \displaystyle \frac { \mathrm { T P } + \mathrm { F N } } { n } , \qquad q _ { J } = \displaystyle \frac { \mathrm { T P } + \mathrm { F P } } { n } , } } \\ { { p _ { e } = q _ { H } q _ { J } + ( 1 - q _ { H } ) ( 1 - q _ { J } ) , \qquad \kappa = \displaystyle \frac { p _ { o } - p _ { e } } { 1 - p _ { e } } . } } \end{array}\tag{A9}
$$

Sensitivity is $\mathrm { T P } / ( \mathrm { T P } + \mathrm { F N } )$ , specificity is $\mathrm { T N } / ( \mathrm { T N } + \mathrm { F P } )$ , and balanced agreement is their arithmetic mean. The first conditions on physician-positive labels and the second on physiciannegative labels. These rates and $p _ { o }$ are proportions; percentage agreement is $1 0 0 p _ { o } .$ , as in Table 2. The 262-label sensitivity table reports these measures as proportions, with True rate equal to $q _ { J }$ Cohen’s κ is dimensionless. Each audit is evaluated separately.

Agreement by criterion type. Figure $\mathrm { 7 e , j }$ compares four judges with the 1,013 physician labels in the Expanded Audit, from 54 responses: nine generators each answer six cases. Sensitivity and specificity condition on physician-positive $( n = 4 5 6 )$ and physician-negative $( n = 5 5 7 )$ labels. Type-specific agreement counts matching binary judgments among 599 positive requirements, 225 major-error criteria, or 189 minor-error criteria. The positive requirements comprise core answers, key evidence, and conditional modifiers. These measures describe criterion recognition within consultations.

Table A30: Coverage of the Targeted Audit for O1. Flagged labels retained are included in the final column and are removed only in the 262-label sensitivity analysis.
<table><tr><td>Response model</td><td>Sampled resp.</td><td>Excluded resp.</td><td>Flagged labels retained</td><td>Included labels</td></tr><tr><td>GPT-5.1</td><td>15</td><td>0</td><td>184</td><td>264</td></tr><tr><td>Qwen3.6-27B-Instruct</td><td>8</td><td>0</td><td>0</td><td>136</td></tr><tr><td>GPT-OSS-120B</td><td>7</td><td>0</td><td>102</td><td>132</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>18</td><td>17</td><td>0</td><td>16</td></tr></table>

Table A31: Targeted Audit sensitivity after excluding all flagged responses. Agreement statistics are proportions; κ is dimensionless. Acc. is overall agreement, Bal. acc. averages sensitivity and specificity, and True rate is the fraction of labels marked true by the judge.
<table><tr><td>Judge model</td><td>n</td><td>Acc.</td><td>Bal. acc.</td><td>κ</td><td>Sens.</td><td>Spec.</td><td>True rate</td></tr><tr><td>GPT-4.1</td><td>262</td><td>0.767</td><td>0.772</td><td>0.533</td><td>0.802</td><td>0.742</td><td>0.489</td></tr><tr><td>GPT-OSS-120B</td><td>262</td><td>0.840</td><td>0.826</td><td>0.665</td><td>0.739</td><td>0.914</td><td>0.363</td></tr><tr><td>Qwen3-30B-A3B-Instruct</td><td>262</td><td>0.752</td><td>0.760</td><td>0.506</td><td>0.811</td><td>0.709</td><td>0.511</td></tr><tr><td>Qwen3.6-27B</td><td>262</td><td>0.782</td><td>0.781</td><td>0.558</td><td>0.775</td><td>0.788</td><td>0.450</td></tr></table>

The figure’s intervals resample 54 complete response clusters, retaining all criteria within each response. We use the same 10,000 draws for every judge and endpoint (seed 20261029), with the 2.5th and 97.5th percentiles as interval bounds. These responses share six case inputs, so a separate case-level analysis below retains all nine generators when resampling a case. The two analyses quantify variation between responses and between cases.

(a) Physician agreement  
![](images/5bbae88ac46b1fa3f15622c3422ca73fa93b990bb76ff9e32233783bbc80d4c8.jpg)

![](images/9d3c058014038afe5e0410ae7d41da0db50ffbde35ddb4df859f4de8a54ff5e5.jpg)  
Figure A7: Physician–judge agreement in the Expanded Audit. All panels use the same 1,013 criterion labels from 54 consultation responses to six cases. (a) Overall agreement and 95% responsecluster bootstrap intervals. (b) Agreement conditional on physician-positive and physician-negative labels (sensitivity and specificity); connecting lines pair the two measures for each judge.

Table A32 reports the case-level resampling analysis and a leave-one-case-out check. The former retains all nine generators and every criterion for each sampled case; the latter examines whether agreement depends strongly on one case. These checks quantify sensitivity to the six cases represented in the Expanded Audit.

Table A32: Expanded Audit: physician–judge agreement on 1,013 shared criteria, 54 responses and six cases. Case-cluster 95% intervals resample six complete cases 10,000 times. The last column gives agreement after omitting each case in turn. Values are percentages.
<table><tr><td>Judge</td><td>Agreement</td><td>Case-cluster 95% CI</td><td>Leave-one-case-out range</td></tr><tr><td>GPT-OSS-120B</td><td>70.5</td><td>[66.2, 77.7]</td><td>[68.5, 71.8]</td></tr><tr><td>Qwen3-30B</td><td>76.9</td><td>[72.3, 81.1]</td><td>[75.1, 78.4]</td></tr><tr><td>Qwen3.6-27B</td><td>77.8</td><td>[74.1, 81.6]</td><td>[76.8, 79.1]</td></tr><tr><td>GPT-4.1</td><td>77.6</td><td>[72.7, 82.5]</td><td>[75.9, 79.2]</td></tr></table>

## E Evaluation Prompts

These templates show the task instructions and response formats. Chinese-language instructions are rendered in English. Bracketed fields and ellipses stand for the full case text, options, or anatomical lists supplied in each request; images accompany the multimodal tasks. The O1 judge instructions and four calibration examples are reproduced below.

The L2 and L4 cards show the single-choice format. One L2 item and two L4 items explicitly request multiple selections in the question; these require the complete reference option set, with no credit for missing or extra selections.

## E.1 Guideline and Clinical Choice Tasks

## Prompt A1. L1: Guideline memory

Question

[guideline question]

Options

A. [option A]

B. [option B]

Answer with the option letter from the given choices directly.

Question

[application question]

Options

## Prompt A2. L2: Rule application

[clinical vignette]

A. [option A]

B. [option B]

Context

Answer with the option letter from the given choices directly.

## Prompt A3. L4: Case planning

Context

[case at the current decision

state]

Question

[management question]

Options

A. [option A]

B. [option B]

Answer with the option letter from the given choices directly.

## Prompt A4. L5: Longitudinal revision

Context

[patient history up to this turn] Question

At this decision point, which next action best matches guideline-concordant NPC specialist management?

Options

A. [action A]

B. [action B]

. . .

Output the final answer directly. It must contain exactly one option letter inside \boxed{}, for example \boxed{A}.

## E.2 Multimodal Evidence Grounding

## Prompt A5. L3: Pathology overview

Given the provided whole-slide pathology overview image, determine whether the case is nasopharyngeal carcinoma. Output only 0 or 1.   
0 indicates normal tissue, and 1 indicates nasopharyngeal carcinoma.

## Prompt A6. L3: MRI sequence and plane

You will see one NPC MRI sequence-recognition image composed of a 3×3 panel. All nine images are slices from the same sequence of the same patient. Sequence names and explicit textual cues have been removed. Choose the matching joint sequence-plane class.   
Options   
A. T1\_Ax; B. T2\_Ax; C. T1+C\_Ax; D. T1\_COR; E. T1\_SAG; F. T1+C\_SAG; G. T1+C\_COR\_FS. Output only one option letter inside \boxed{}, for example \boxed{A}.

## Prompt A7. L3: MRI structure card

You will see one NPC MRI structure card. The card is a fixed 4×3 image. Each row corresponds to one MRI sequence, and each column contains one of three consecutive slices selected from the same sequence by maximizing the annotated structure-box area. The yellow rectangle marks the anatomical structure under evaluation.   
Row sequences   
[four sequence names in row order]   
Tasks   
Identify the base anatomical structure marked by the yellow box and output structure\_id. Determine whether the structure is invaded and output invasion.   
Select the structure from [18 base-structure choices]. Left and right structures are collapsed for primary scoring. invasion is 0 (not invaded) or 1 (invaded).   
Response   
Output only one JSON object, without explanations, markdown, or code fences:   
{"structure\_id":"nasal\_passage", "invasion":0}

## Prompt A8. L3: T staging from structure cards (100 patients)

You will see 27 MRI structure cards from the same NPC patient. Each card has a fixed 4×3 layout; the yellow rectangle marks the structure under evaluation.   
Tasks   
Determine invasion status for all 18 base anatomical structures. Integrate this evidence using the eighth edition AJCC/UICC TNM criteria for NPC and output the patient-level T stage.   
structure\_invasion must cover all 18 structures with a value of 0 or 1. t\_stage must be T1, T2, T3, or T4.   
Response   
Output only one JSON object, without explanations, markdown, or code fences:   
{"structure\_invasion":{"nasal\_passage":0, "oropharynx":0, "...":0},   
"t\_stage":"T1"}

## Prompt A9. L3: T staging from aligned sequence groups (300 patients)

You will see MRI image groups from one NPC patient. Axial groups contain T1, T2, and enhanced T1 at matched anatomical planes. Coronal groups contain enhanced fat-suppressed T1 and precontrast non-fat-suppressed T1 at matched planes.

Determine invasion status for all 18 base anatomical structures. Integrate this evidence using the eighth edition AJCC/UICC TNM criteria for NPC and output the patient-level T stage.

The structure list and JSON response fields are the same as in the structure-card T-stage task above. All selected image groups enter one request.

## Prompt A10. L3: N staging from aligned sequence groups (400 patients)

You will see axial and coronal MRI image groups from one NPC patient, with the same sequence arrangement described above. The axial coverage extends from the skull base to the inferior sternoclavicular joint.

Task

Determine the N stage using the eighth edition AJCC/UICC TNM criteria for NPC.

Response

Output only one JSON object containing n\_stage and the supporting imaging evidence. n\_stage must be N0, N1, N2, or N3.

## E.3 CarePath episodes

The pathology and MRI prompts produce diagnostic and staging predictions. Each care question receives the specified image-derived findings and clinical context. Earlier care answers are included only in the separate history-feedback experiments. The care prompt supplies the available clinical context, including N/M/overall-stage information in 133 questions. The clinical context and imageprediction fields jointly define the evidence available for the decision.

## Prompt A11. L6: Pathology classification

Based on the provided pathology image, determine whether the case supports a diagnosis of nasopharyngeal carcinoma. Output only 0 or 1.   
0 means the image does not support NPC; 1 means it supports NPC.

## Prompt A12. L6: MRI staging

You will see MRI panels from the same NPC patient. Each panel combines sequences and planes from   
the same examination.   
Tasks   
Assign 0 or 1 to the invasion status of each candidate structure. Infer the AJCC ninth-edition T stage and   
briefly state the imaging evidence.   
The nasopharynx is recorded as the background primary site and excluded from structure-invasion   
scoring. Evaluate [18 candidate structures].   
Response   
Output only JSON with the following fields:   
{"structure\_invasion":{[structure]:0, ...},   
"involved\_structures":[...], "T\_stage":"T1",   
"evidence":"[brief imaging evidence]"}   
T\_stage must be T1, T2, T3, or T4.

Prompt A13. L6: Care decision   
Information is provided in clinical order. Answer using only the information available at the current   
decision point; do not assume findings absent from the question.   
Prior diagnostic findings   
[predicted pathology; predicted MRI T stage and invaded structures; supplied   
N/M and overall-stage fields, where specified]   
Clinical information   
[available history and current checkpoint]   
Question   
[care decision]   
Options   
A. [action A]; B. [action B]; . . . ; [last option and action]   
Output only one option letter from [valid option letters].

## E.4 Consultation Generation and Judging

The response model sees the case and question. The judge additionally sees the generated answer and one signed criterion. A true judgment earns credit for a positive requirement and incurs a deduction for a negative error criterion.

## Prompt A15. O1: Criterion judge

You are evaluating an answer in a nasopharyngeal carcinoma clinical consultation benchmark. Your job is to look at the case, the clinical question, the assistant response, and one rubric item. Judge whether the assistant response matches the rubric item according to the criterion type shown inside the rubric item.

Return a JSON object with exactly the following fields: "explanation" and "criteria\_met".

• "criteria\_met" must be a boolean.

• "explanation" must be a brief string explaining why the assistant response does or does not satisfy the rubric item. Refer to specific content from the assistant response when possible.

• Judge only the assistant response. Do not give credit for information that appears only in the case or question unless the assistant response also uses or states it.

• First identify the criterion type in the rubric item.

• For a POSITIVE REQUIREMENT, return true only when all required elements are substantially satisfied by the assistant response.

• For a NEGATIVE ERROR CRITERION, return true only if the assistant response actually makes the described mistake, unsafe recommendation, contradiction, or omission. Return false if the assistant avoids the error, gives the correct answer, or merely discusses the issue as a caveat without committing the error.

"criteria\_met": false

## Prompt A15. O1: Criterion judge (continued)

• For a NEGATIVE ERROR CRITERION, a true judgment must be supported by specific offending content in the assistant response. If you cannot identify such content, return false.

<sub>• If a negative criterion is about omission, such as "fails to mention" or "</sub>未说明<sub>", return true only when</sub> the required information is absent from the assistant response.

<sub>• If the rubric item uses phrases such as "such as", "for example", "including", "</sub>包括<sub>", "</sub>例如<sub>", or "</sub>如<sub>",</sub> the assistant response does not need to mention every example, but it must satisfy the core clinical concept.

• Accept clinically equivalent wording, synonyms, abbreviations, and reasonable Chinese-English variants.

• Do not require exact wording from the rubric item.

• Do not give extra credit for content outside the rubric item.

• For a POSITIVE REQUIREMENT, if the response is vague, generic, or only partially satisfies the item, return false.

• If the assistant response contradicts the rubric item, return false.

• If the answer states uncertainty appropriately and the rubric item allows uncertainty or missing information, count that as satisfying the item.

## Calibration example 1

If the rubric item is "states that the adverse event is pruritus" and the assistant response says "the patient currently has skin itching", return:

"explanation": "The response   
identifies the current adverse   
event as skin itching, which is   
clinically equivalent to   
pruritus.",   
"criteria\_met": true   
}

## Calibration example 2

If the rubric item is "states that the severity cannot be determined from the available information" and the assistant response directly assigns CTCAE grade 2 without explaining missing grading information, return: 5

"explanation": "The response gives a specific CTCAE grade but does not state that severity cannot be determined from the available information.",

## Calibration example 3

Negative error criterion present. If the criterion type is NEGATIVE ERROR CRITERION, the rubric item is "incorrectly treats anemia as the main current adverse event", and the assistant response says "the main current adverse event is anemia", return: {

"explanation": "The response   
identifies anemia as the main   
current adverse event, which   
matches the mistake described in   
the rubric item.",   
"criteria\_met": true   
}

## Calibration example 4

If the criterion type is NEGATIVE ERROR CRITERION, the rubric item is "misclassifies cT3N0M0 as T4 or N-positive disease", and the assistant response correctly says "the clinical stage is cT3N0M0 because MRI supports T3 and there is no nodal or distant metastasis", return:

"explanation": "The response   
correctly states cT3N0M0 and does   
not misclassify the case as T4 or   
N-positive disease.",   
"criteria\_met": false   
}  
Final instruction. Return only the JSON object. Do not include markdown fences or any other text.

## F Illustrated Task Inputs

These examples show the three L3 image interfaces and their reference labels. The English summaries identify the requested outputs; complete task prompts appear in section E. Table A27 separately illustrates how longitudinal text inputs change across the three history conditions.

Pathology overview

![](images/ac720944a0937f8ca514b0cf43b203bbc113fd674d4e328d5235b8dcc133dc33.jpg)  
NPC-positive: 1

![](images/a5c3908e59e152974e11f0a95bc4229925297cbb987b8f4e6a77ebd1b54acdcc.jpg)  
Negative control: 0

Task. Determine whether the provided whole-slide overview shows NPC; return 0 (normal tissue) or 1 (NPC).

MRI inputs

![](images/01c80869e08f51361b8371f17bf93eee4281f143b33ed07b2d19ed249ada8257.jpg)

Sequence and plane  
![](images/2811bf9dd247b4623ffcf616b03a27323d33a7c77061b75f27f0fd17ff37579b.jpg)

Identify the structure marked in yellow and determine whether it is invaded. Reference: nasal\_passage; invasion=1.

Identify the joint sequence–plane category from a nine-slice panel with sequence-name cues removed. Reference: E, T1\_SAG (sagittal T1).

Patient-level staging. T staging uses structure cards for 100 patients and aligned sequence groups for 300 patients. Both formats request all 18 base-structure invasion labels and a T1–T4 category under the eighth-edition AJCC/UICC criteria. N staging uses sequence groups for a disjoint cohort of 400 patients, with inferior coverage extending to the sternoclavicular joint. All images selected for a patient enter one request; a single card does not determine the final stage.

## F.1 Reading Model Responses

The following responses illustrate why the scoring targets distinguish anatomical identity from invasion and distinguish an error criterion from a positive requirement.

## Example 1: Anatomical identity and invasion

Evaluation. GPT-5.1, MRI structure-identification experiment.

Structure card. Atlas (C1 vertebra).

Gold structure and invasion. atlas (C1 vertebra); invasion=0

Prediction. longus\_capitis; invasion=0.

Interpretation. The numeric invasion value matches, but refers to the wrong anatomical structure. The response fails the joint structure–invasion target; checking the binary value alone would miss this localization error.

## Example 2: An error criterion that is not triggered

Context. In the physician-review packet, a 60-year-old man with recurrent NPC after definitive chemoradiotherapy had received GP plus camrelizumab and was assessed as T3N1. The question asked for the current treatment strategy and guideline basis.

Response. Gemini-3.1-Pro-Preview proposed “camrelizumab monotherapy maintenance (preferred systemic strategy)”, with additional workup and review of hepatitis B and prior radiation dose.

Error criterion. Choosing local radical treatment, such as re-irradiation or surgery, as the main strategy at this stage instead of systemic treatment.

Interpretation. The response centres systemic treatment, so this criterion is false and incurs no deduction. This judgment concerns one criterion; the remaining requirements and errors must still be assessed before scoring the consultation.