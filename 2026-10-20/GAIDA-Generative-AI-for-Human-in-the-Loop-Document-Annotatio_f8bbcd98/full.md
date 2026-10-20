# GAIDA: Generative AI for Human-in-the-Loop Document Annotation

Islam Mesabah<sup>1,2[0000−0002−2298−2127]</sup>, Ali Elhosary<sup>2[0009−0007−5694−5694]</sup>, Priyabanta Sandulu<sup>1,2[0009−0003−9284−5093]</sup> Darko Obradovic<sup>1,2[0009−0002−2681−5649]</sup> , and Sebastian J. Vollmer<sup>1,2[0000−0003−2831−1401]</sup>

<sup>1</sup> German Research Center for Artificial Intelligence (DFKI), RPTU Kaiserslautern-Landau, Kaiserslautern, Germany islam.mesabah@dfki.de

Abstract. Document annotation is a fundamental yet labor-intensive step in building document information extraction systems. While Optical Character Recognition (OCR) and Large Language Models (LLMs) can automate parts of this process, we argue that annotation systems should be judged not by how completely they remove human efort, but by how efectively they redirect it toward review and correction. We propose a modular human-in-the-loop architecture that integrates OCRbased extraction, LLM-based semantic pre-annotation, and human validation in a single editable workflow, and present GAIDA, a research prototype serving as its reference implementation. GAIDA supports interchangeable OCR engines, LLM backends, and prompt configurations, and enables stage-specific measurement of annotation efort. In an initial single-annotator, stage-specific study on the FUNSD scanned-form and WildReceipt retail-receipt benchmarks, OCR assistance reduced regioncreation and transcription time by up to 80.9%, and reviewing LLM pre-annotations reduced semantic-labeling time by up to 73.7%, despite imperfect pre-annotations of 78.86% and 86.98% Micro F1 on FUNSD and WildReceipt, respectively. Finally, we reuse the same implementation, adapting only OCR and prompt configuration, to build and openly release AraReceipt, a manually annotated dataset of 100 Arabic receipt images labeled with 25 key-information classes. The results indicate that imperfect automated outputs provide practical value when presented as editable suggestions under human control.

Keywords: Generative AI · Large Language Models · Human-in-the-Loop · Document Annotation · Document Information Extraction · Optical Character Recognition · Arabic Information Extraction

## 1 Introduction

Document information extraction transforms semi-structured documents into structured representations for downstream systems. Building high-quality annotated datasets remains a bottleneck [28, 30]: identifying regions, transcribing content, assigning labels, and validating output is slow and prone to inconsistency under complex layouts [13, 12].

OCR and LLMs can restructure, not merely accelerate, this process: OCR localizes and transcribes text automatically, after which an LLM proposes semantic labels for human validation, shifting the annotator’s role from creation to verification [18, 29]. Model accuracy alone does not determine practical value; usefulness also depends on the correction efort created by OCR errors, prompt design, model choice, and the review interface. Studying these factors requires a common modular architecture in which individual components can be exchanged while the surrounding annotation workflow and human-review process remain fixed.

To operationalize this architecture, we developed GAIDA<sup>3</sup>, a research prototype that serves as its reference implementation. GAIDA integrates OCRbased extraction, LLM-based semantic pre-annotation, and human validation within the open-source Label Studio<sup>4</sup> annotation environment. GAIDA rests on the premise that annotation systems should be judged by how efectively they redirect human efort toward review, not by how completely they remove it. We evaluate GAIDA in two complementary document settings: FUNSD [9] represents noisy forms whose labels often depend on relations between regions, whereas WildReceipt [21] represents dense retail receipts with a substantially more fine-grained field schema. We also reuse the same architecture to construct and publicly release AraReceipt<sup>5</sup>, a dataset of 100 Arabic receipt images derived from ReceiptSense [2] and annotated using a 25-class schema. Accordingly, we position this work as a system-and-resource contribution accompanied by an initial single-annotator feasibility study; a controlled multi-annotator evaluation is left to future work.

This paper addresses four research questions:

RQ1: To what extent does OCR assistance reduce region-creation and transcription time, and how much localization/recognition error remains for human correction?

– RQ2: How sensitive is semantic pre-annotation quality to model choice and prompt design under a fixed workflow?

– RQ3: Does reviewing LLM-generated pre-annotations reduce semantic-labeling time relative to manual assignment?

– RQ4: Which components and configurations must be adapted to apply GAIDA to Arabic receipts?

Contributions: (1) a modular human-in-the-loop annotation architecture, together with GAIDA, a research prototype that serves as its reference implementation and supports interchangeable OCR and LLM components; (2) initial stage-specific evidence that OCR assistance reduces region-creation and transcription time, while LLM pre-annotations reduce semantic-labeling time; (3) a sensitivity analysis showing that model choice afects pre-annotation quality far more than prompt elaboration, and that more prescriptive prompting does not consistently help (Sect. 4.4); and (4) AraReceipt, a manually annotated, publicly released Arabic dataset of 100 images and 25 classes, showing architectural reuse with language-specific configuration only.

## 2 Related Work

Document information extraction benchmarks rely on task-specific annotations that may include document regions, transcriptions, semantic entities, and key– value relations [14]. Creating such resources requires clear guidelines, annotator training, agreement assessment, and quality control [10, 5], since inconsistent annotation schemes can undermine benchmark reliability [11]. OCR can preextract text regions and transcriptions [17, 27]; however, recognition and localization errors, as well as choices in representing OCR output, can propagate to downstream processing [23, 3, 19]. This motivates evaluating not only standalone OCR accuracy but also the correction efort left to human annotators.

LLMs increasingly generate structured predictions from OCR output or document images [3, 19]. Recent work incorporates document layout in diferent ways: DocLLM models textual and spatial information jointly and LayIE-LLM studies configuration choices for layout-rich information-extraction pipelines [24, 3]. These studies show that extraction quality depends on input representation, prompt design, annotation schema, and model choice [3, 19]. Model-centered performance scores, however, do not quantify the human efort required to inspect and correct the resulting outputs.

Human-in-the-loop annotation therefore treats machine-generated pre-labels as editable suggestions rather than final labels, retaining human control over correction and approval [25, 6]. Such pre-labels can reduce expert annotation time [6, 25], although incorrect pre-annotations can bias annotator decisions or shift effort from label creation to error detection [25, 16]. Closest to our setting, DocSpiral integrates document processing, annotation interfaces, and iterative model improvement in a human-in-the-spiral workflow, progressively reducing manual efort through repeated retraining [22]. GAIDA is complementary: it holds the assisting models fixed and instead decomposes where human efort is spent—region creation, transcription correction, and semantic labeling—under interchangeable OCR and LLM components. To our knowledge, this stage-specific decomposition of annotation efort has not been reported for OCR–LLM-assisted document annotation.

Multilingual document understanding introduces further challenges from script variation, uneven OCR performance across languages, and limited annotated resources [27, 26, 8]. Multilingual systems and resources include LOFI, a language-, OCR-, and form-independent extraction pipeline evaluated on Korean and Japanese business documents [27], and AMuRD and ReceiptSense, which provide Arabic–English receipt data [1, 2]. These works motivate a complementary system-level question: whether the same human-in-the-loop implementation can be reused across scripts through configuration changes rather than architectural redesign.

![](images/4cf89accc51d6200614303c4ce7f5aacdd8239843f2ed649b84b58483ea8c57c.jpg)  
Fig. 1. The proposed human-in-the-loop annotation workflow as implemented in GAIDA. OCR extracts text regions and transcriptions, an LLM generates preliminary semantic labels, and a human annotator reviews, corrects, and approves each prediction before export.

GAIDA is thus positioned as a research prototype and reference implementation of a modular OCR–LLM–human annotation architecture: rather than a new extraction model or a model-training platform, it provides a common editable workflow for comparing component and prompt configurations, measuring transcription and semantic-labeling efort separately, and examining configurationlevel reuse for Arabic receipts.

## 3 Proposed Architecture and GAIDA Implementation

The proposed architecture follows a human-in-the-loop design in which automated components assist, but do not replace, the human annotator. OCR and LLM inference produce editable pre-annotations that a human inspects, corrects, and approves before export. GAIDA implements this architecture as a modular research prototype, separating OCR processing, semantic labeling, human validation, and export into interchangeable components.

## 3.1 Workflow and Architecture

As shown in Fig. 1, the workflow implemented in GAIDA follows four stages. First, the annotator defines the valid semantic classes and annotation configuration in Label Studio. Second, the selected OCR engine returns text regions, transcriptions, and confidence values. Third, GAIDA constructs a structured prompt containing the recognized text, corresponding text regions, valid label set, and task instructions; the LLM then assigns one semantic class to each region. Finally, the resulting pre-annotation is loaded into Label Studio, where the annotator reviews and corrects the regions, transcriptions, and labels before export. Label Studio provides the annotation interface, while the GAIDA backend coordinates OCR processing, prompt construction, LLM inference, response parsing, and transfer of the resulting annotations. OCR engines need only expose text, coordinates, and confidence; LLM backends need only accept a structured prompt and return machine-readable labels. This lets the same architecture be reused, unmodified, for AraReceipt (Sect. 4.6).

## 3.2 OCR Processing and Normalization

Each OCR engine returns a set of regions $\mathcal { R } = \{ ( b _ { i } , t _ { i } , c _ { i } ) \} _ { i = 1 } ^ { n }$ , where $b _ { i } , t _ { i } .$ and $c _ { i }$ denote the bounding box, transcription, and confidence value of region i, respectively. GAIDA converts engine-specific bounding-box formats into a common image-relative coordinate representation, removes leading and trailing whitespace from recognized text, and discards detections with missing text or invalid coordinates. This normalization allows OCR backends to be exchanged without modifying the downstream prompt-construction or annotation components. It also prevents diferences in engine-specific output formats from introducing spurious variation, while preserving genuine diferences arising from the engines detection and recognition behavior and from document properties such as layout complexity, image quality, and region density (Sect. 4.3). Recognition errors require textual correction, whereas localization errors—including missing, duplicated, split, merged, or incorrectly bounded regions—require structural correction; the two error types are therefore reported separately (Sect. 4.3).

## 3.3 Prompt Construction and Development

GAIDA provides the LLM with the complete OCR output, including each text region $t _ { i }$ and its bounding box $b _ { i }$ , together with the valid label set $\mathcal { V }$ and task instructions $P$ . The model returns a machine-readable mapping that assigns one label $\hat { y } _ { i } \in \mathcal { V }$ to each OCR region. We first compared multiple LLMs on FUNSD using the same rule-based prompt. This prompt defined the valid classes, provided class-specific rules and tie-breaking criteria, and required the model to assign exactly one label to each OCR region. Based on this comparison, we selected the best-performing model for the subsequent prompt experiments.

Using the selected model, we then evaluated three prompt families on FUNSD and WildReceipt. The label-definition prompt clarified the meaning and scope of each class to improve distinctions between semantically similar labels. The decision-order prompt further extended these definitions by introducing a priority sequence for cases in which multiple labels appeared plausible. Finally, the dataset-refined prompt was developed through an exploratory qualitative review of preliminary predictions. Recurring misclassifications and ambiguous label distinctions were identified from the model outputs and translated into additional dataset-specific rules and clarifications. Because this process used errors observed on the evaluated data rather than a held-out development split, the resulting scores represent iterative prompt engineering on these datasets and should not be interpreted as evidence of out-of-sample generalization.

Table 1. Datasets used in the study. Test image counts refer to benchmark test sets; for timing, we used 50 FUNSD and 20 WildReceipt images.
<table><tr><td>Dataset</td><td colspan="4">Document Type Test Images Classes Role in the Study</td></tr><tr><td>FUNSD [9]</td><td>Scanned forms</td><td>50</td><td>4</td><td>OCR evaluation, model selection, prompt devel- opment, and timing.</td></tr><tr><td></td><td>WildReceipt [21] Retail receipts</td><td>472</td><td>25</td><td>OCR, prompt, and timing evaluation under a dense layout and fine-grained</td></tr><tr><td>AraReceipt (Ours)</td><td>Arabic receipts</td><td>100</td><td>25</td><td>schema. Cross-language applica- tion and public Arabic dataset construction.</td></tr></table>

The complete pre-annotation is finally presented for human review: the annotator verifies regions, corrects boundaries and transcriptions, and confirms or changes each label; export happens only after this review, letting us test whether correction is faster than annotating from scratch.

## 4 Experimental Evaluation

We combine model-centered measurements of OCR and semantic-labeling quality with annotator-centered measurements of elapsed time. The timing experiments are preliminary stage-specific comparisons: OCR and semantic labeling were timed independently, rather than as one continuous end-to-end annotation session. RQ1–RQ3 use within-annotator comparisons on FUNSD and WildReceipt; RQ4 uses the construction of AraReceipt (Sect. 4.6).

## 4.1 Experimental Design

We use two English benchmarks (Table 1). FUNSD [9] contains noisy scanned forms with a compact four-class schema (header, question, answer, other), where class assignment often depends on relations between neighboring regions. WildReceipt [21] contains dense retail receipts with a 25-class schema, where classes are more often cued by local lexical or positional signals than by inter-region relations. This distinction is interpreted in Sect. 5.

To isolate each assistance stage, we compare four workflows (Table 2), pairing a manual and an assisted condition for each pipeline stage. In every condition, the annotator worked toward the visible benchmark ground truth and was timed until producing a matching annotation; final quality is thus matched by construction, and the reported times isolate efort rather than quality. The semantic-labeling comparison specifically began both conditions from the same ground-truth OCR regions and transcriptions, isolating the LLM’s contribution from OCR quality.

Table 2. Comparison of the annotation workflows evaluated in this study.
<table><tr><td>Workflow</td><td>OCR LLM</td><td></td><td>System Output</td><td>Primary Human Activity</td></tr><tr><td>manual-OCR</td><td>x</td><td>x</td><td>None</td><td>Create regions/text.</td></tr><tr><td>assisted-OCR</td><td>√</td><td>x</td><td>Regions, transcriptions</td><td>Validate, correct regions/text.</td></tr><tr><td>manual-labeling</td><td>x</td><td>x</td><td>None</td><td>Create labels.</td></tr><tr><td>assisted-labeling</td><td>x</td><td>√</td><td>Labels</td><td>Validate, correct labels.</td></tr></table>

All sessions followed the same fixed and non-randomized order: for each dataset, a single annotator completed manual OCR, assisted OCR, manual labeling, and assisted labeling on the same images, with a one-week gap between the manual and assisted passes. Using the same annotator removed betweenannotator variability, while the one-week interval helped reduce immediate memory and fatigue efects. Because the assisted condition always followed the manual condition on the same images, familiarity efects are possible (Sect. 5).

WildReceipt timing used a random 20-image subset of its 472 test images because of the study’s time constraints, whereas FUNSD timing used all 50 test images. The same selected images were used in the corresponding manual and assisted conditions. Timing was conducted by one annotator, a native Arabic speaker with professional English proficiency. The same annotator completed all English timing sessions and annotated AraReceipt. This controls betweenannotator speed variation but does not remove the possible influence of nonnative English proficiency on the absolute English timing baselines (Sect. 5).

We integrated three OCR engines into GAIDA—Tesseract [20], EasyOCR<sup>6</sup>, and PaddleOCR [4]—to verify that the architecture could support diferent OCR backends without changing the surrounding workflow. Following a preliminary qualitative inspection of their outputs, PaddleOCR was selected for the quantitative experiments (Table 3). Because this screening was not a formal comparative benchmark, we do not report or claim quantitative superiority over Tesseract or EasyOCR. For semantic pre-annotation, we evaluated GPT-OSS 20B, GPT-OSS 120B [15], Llama 3.3 70B [7], and Llama 4 Maverick<sup>7</sup>. All models were hosted locally and served through Ollama using a temperature of 0.0, a maximum output length of 2,048 tokens, and a timeout of 180 seconds. Because temperature was fixed at 0.0, each model–prompt configuration was evaluated once; the reported results therefore do not estimate run-to-run variation. Model comparison was conducted exclusively on FUNSD (Table 5). Llama 4 Maverick was selected for the prompt experiments because it achieved the highest Macro F1 under the shared rule-based prompt. For the semantic-labeling timing experiment, the assisted condition used pre-annotations generated by Llama 4 Maverick with the dataset-refined prompt for the corresponding dataset.

## 4.2 Evaluation Metrics

We evaluate OCR localization using intersection over union, $\mathrm { I o U } ( { \hat { b } } , b ) = | { \hat { b } } \cap \quad$ $b | / | \hat { b } \cup b |$ , between predicted and ground-truth bounding boxes $\hat { b }$ and b. Detection is scored by greedy one-to-one matching at $\tau _ { d } ~ = ~ 0 . 5 \colon$ all pairs with $\mathrm { I o U } \geq 0 . 5$ are ranked by descending IoU and accepted sequentially if neither box is already matched, so a predicted region must overlap a ground-truth region substantially—while tolerating minor boundary diferences—and no prediction receives credit for more than one ground-truth region. Given the accepted matches M and the ground-truth and predicted region sets G and $P ,$ recall is $| M | / | G |$ and precision is $| M | / | P |$ ; Mean IoU, $\textstyle | M | ^ { - 1 } \sum _ { ( g , p ) \in M } \operatorname { I o U } ( g , p )$ , measures boundary quality of detected regions rather than coverage.

Segmentation consistency uses a relaxed threshold of 0.1, which identifies weak overlaps associated with possible splitting or merging and is not used to determine correct detections. With $\begin{array} { r } { c _ { g } = \sum _ { p } { \bf 1 } [ \mathrm { I o U } ( g , p ) \geq 0 . 1 ] } \end{array}$ and $c _ { p } =$ $\begin{array} { r } { \sum _ { q } \mathbf { 1 } [ \mathrm { I o U } ( g , p ) \geq 0 . 1 ] } \end{array}$ counting the predictions overlapping ground-truth region $g$ and the ground-truth regions overlapping prediction $\textstyle p , { \mathrm { S p l i t R a t e } } = \sum _ { g } c _ { g } / | G |$ and MergeRate $= \textstyle \sum _ { p } c _ { p } / | P |$ give the mean overlap counts per ground-truth region and per prediction; both are unbounded above, and values below 1 indicate unmatched regions. We also report Exact1to1 $\begin{array} { r } { = \sum _ { q } { \bf 1 } [ c _ { g } = 1 ] / | G | } \end{array}$ , the proportion of ground-truth regions overlapping exactly one prediction at this threshold, without enforcing bidirectional one-to-one matching. Transcription quality is evaluated by the Character Error Rate, $\mathrm { C E R } = ( S + D + I ) / N$ , with S, D, and I the character substitutions, deletions, and insertions and N the reference length, averaged over matched regions per image and then macro-averaged over images.

Semantic pre-annotation is evaluated on the ground-truth OCR regions and transcriptions using Micro and Macro F1; because each ground-truth region receives a predicted label, these scores measure label agreement independently of OCR localization and transcription errors. Micro F1 aggregates true positives, false positives, and false negatives across labels; Macro F1 is $\begin{array} { r } { \frac { 1 } { | \mathcal { V } | } \sum _ { y \in \mathcal { Y } } \mathrm { F } 1 _ { y } , } \end{array}$ , the unweighted mean of per-label F1. Annotation time is the elapsed time per image to produce an annotation consistent with the ground truth (Sect. 4.1), and time reduction is $( T _ { \mathrm { m a n u a l } } - T _ { \mathrm { a s s i s t e d } } ) / T _ { \mathrm { m a n u a l } } \times 1 0 0 $

## 4.3 RQ1: OCR-Assisted Annotation

Table 3 summarizes PaddleOCR performance across the evaluated datasets. WildReceipt achieves higher detection precision and recall, together with a lower

CER, indicating better localization and transcription than FUNSD. FUNSD’s higher SplitRate (1.18 vs. 0.96) is consistent with more gold regions having multiple loosely-overlapping candidates; WildReceipt’s higher MergeRate (1.08 vs. 1.02) is consistent with adjacent items collapsing into single detections.

Table 3. PaddleOCR results. Detection metrics use one-to-one matching at IoU ≥ 0.5, while overlap metrics use IoU ≥ 0.1 (Sect. 4.2). Arrows indicate the preferred direction.
<table><tr><td>Task Type</td><td>Metric</td><td>FUNSD</td><td>WildReceipt</td></tr><tr><td rowspan="3">Detection and localization Recall ↑</td><td>Precision ↑</td><td>65.00%</td><td>83.88%</td></tr><tr><td></td><td>73.00%</td><td>75.80%</td></tr><tr><td>Mean IoU (matched) ↑</td><td>75.00%</td><td>76.50%</td></tr><tr><td rowspan="3">Overlap structure</td><td>Exact 1-to-1 Ratio ↑</td><td>83.00%</td><td>87.97%</td></tr><tr><td>Split Rate → 1</td><td>1.18</td><td>0.96</td></tr><tr><td>Merge Rate → 1</td><td>1.02</td><td>1.08</td></tr><tr><td>Transcription</td><td>CER↓</td><td>8.60%</td><td>6.70%</td></tr></table>

Recognition errors require textual correction, whereas localization and overlapping errors require regions to be resized, added, removed, split, or merged. As shown in the upper part of Table 4, the assisted-OCR condition substantially reduces this correction efort compared with manual-OCR: from 13:30 to 3:37 for FUNSD (73.2%) and from 8:49 to 1:41 for WildReceipt (80.9%). The longer manual annotation time observed for FUNSD is consistent with its lower detection precision and greater split tendency; however, the efects of region count, text length, layout structure, and session order (Sect. 4.1) were not disentangled.

Table 4. Manual vs. assisted annotation time per image, by pipeline stage (mm:ss), for the workflows defined in Table 2. Times measure correction toward visible benchmark ground truth (Sect. 4.1); n = timed images per dataset.
<table><tr><td>Stage</td><td>Dataset</td><td>Manual Assisted Reduction</td><td></td></tr><tr><td rowspan="2">OCR / transcription</td><td>FUNSD (n = 50) 13:30</td><td>3:37</td><td>73.2%</td></tr><tr><td>WildReceipt (n = 20) 8:49</td><td>1:41</td><td>80.9%</td></tr><tr><td rowspan="2">Semantic labeling</td><td>FUNSD (n = 50)</td><td>1:53 0:36</td><td>68.1%</td></tr><tr><td>WildReceipt (n = 20)</td><td>1:39 0:26</td><td>73.7%</td></tr></table>

RQ1: OCR assistance reduced region-creation and transcription time by 73.2% and 80.9% on FUNSD and WildReceipt, respectively, although localization and recognition errors still required manual correction (Sect. 5 discusses ordering efects).

## 4.4 RQ2: Sensitivity to Model and Prompt Configuration

We first compare the LLMs on FUNSD using a shared rule-based prompt (Table 5). Performance varies widely: Micro F1 ranges from 40.57% (GPT-OSS 20B) to 72.43%, which Llama 3.3 70B and Llama 4 Maverick both reach. Maverick attains the higher Macro F1 (63.77% vs. 62.26%), indicating more balanced performance across classes, and is therefore selected for the prompt experiments.

Table 5. Semantic pre-annotation performance across models on FUNSD using the same rule-based prompt.
<table><tr><td colspan="2">Model Micro F1 (%) Macro F1 (%)</td></tr><tr><td>GPT-OSS 20B</td><td>40.57 39.15</td></tr><tr><td>GPT-OSS 120B</td><td>67.20 56.38</td></tr><tr><td>Llama 3.3 70B</td><td>72.43 62.26</td></tr><tr><td>Llama 4 Maverick</td><td>72.43 63.77</td></tr></table>

Using Maverick, we next vary the prompt (Table 6). On FUNSD, explicit label definitions raise Micro F1 to 74.23%, whereas the decision-order prompt lowers it to 70.80% (62.45% Macro F1); the dataset-refined prompt is strongest at 78.86% Micro and 70.18% Macro F1. On WildReceipt, whose 25-class schema is far more fine-grained than FUNSD’s four classes and whose prompts therefore include more detailed rationales, the three prompt families are nearly indistinguishable: Micro F1 varies by less than 0.6 points (86.42–86.98%) and Macro F1 by less than 0.3 points (77.15–77.41%), with no configuration dominating on both metrics.

Table 6. Semantic pre-annotation performance across prompt configurations using Llama 4 Maverick.
<table><tr><td>Dataset</td><td>Prompt Configuration</td><td>Micro F1 (%) Macro F1 (%)</td></tr><tr><td rowspan="3">FUNSD</td><td>Label-definition prompt</td><td>74.23 64.15</td></tr><tr><td>Decision-order prompt</td><td>70.80 62.45</td></tr><tr><td>Dataset-refined prompt</td><td>78.86 70.18</td></tr><tr><td rowspan="3"></td><td>Label-definition prompt</td><td>86.42 77.15</td></tr><tr><td>WildReceipt Decision-order prompt</td><td>86.60 77.41</td></tr><tr><td>Dataset-refined prompt</td><td>86.98 77.36</td></tr></table>

RQ2: Pre-annotation quality is far more sensitive to model choice than to prompt elaboration. Under an identical prompt, Micro F1 on FUNSD spans 40.57–72.43% across models—a gap no prompt variant approaches. Prompt design is inconsistent by comparison: label definitions help, a fixed decision order hurts on FUNSD, and all prompt families converge on WildReceipt. More prescriptive prompting is therefore not inherently better; only guidance derived from observed errors (the dataset-refined prompt) consistently helped. Because that prompt was refined on the evaluated data itself, its scores reflect iterative engineering rather than out-of-sample generalization.

## 4.5 RQ3: Human-in-the-Loop Eficiency

The lower part of Table 4 reports the initial stage-specific comparison of manuallabeling and assisted-labeling under the configuration specified in Sect. 4.1. Both conditions began from the same ground-truth OCR regions and transcriptions, thereby isolating the contribution of semantic pre-annotation. On FUNSD, the average time per image decreased from 1:53 to 0:36, corresponding to a 68.1% reduction. On WildReceipt, it decreased from 1:39 to 0:26, corresponding to a 73.7% reduction.

Because the annotator corrected each image toward visible ground truth (Sect. 4.1), these reductions quantify the efort to reach a known target rather than independent annotation quality. RQ3: Reviewing LLM pre-annotations reduced semantic-labeling time by 68.1% and 73.7%, despite pre-annotation Micro F1 of only 78.86% and 86.98%—imperfect suggestions suficed to redirect efort from creation to correction (threats to validity in Sect. 5).

## 4.6 RQ4: Cross-Language Workflow Adaptation

To examine cross-language applicability, we constructed AraReceipt from 100 Arabic receipt images selected from ReceiptSense [2]. A native Arabic speaker annotated the images without independent review, so we describe the dataset as manually annotated rather than independently validated (Sect. 5).

The core architecture—annotation interface, normalized data representation, OCR–LLM communication, human-validation workflow, and export—was reused without modification, and the PaddleOCR and Llama 4 Maverick configurations were retained without language-specific fine-tuning. Adaptation was limited to configuration: OCR settings for Arabic and mixed-script text, and prompt revisions for Arabic content, right-to-left writing direction, and language-specific field expressions.

Table 7 summarizes the resulting dataset: 100 images with 4,609 annotated regions (46.1 per image), labeled with 25 key-information classes plus an Ignore category, following the WildReceipt schema. Slightly over half of the regions contain Arabic script and 43.6% are purely numeric (prices, quantities, totals), reflecting the mixed-script character of Arabic retail receipts; key-type classes are markedly rarer than their value counterparts, as Arabic receipts often print values without field labels. AraReceipt is released under the MIT license, matching the terms of ReceiptSense, from which its images are derived.

End-to-end GAIDA-assisted annotation, including OCR review, semanticlabel correction, and final approval, averaged 5:51 per image (≈10 hours in total); since no manual Arabic baseline was timed, this is an assisted cost only. RQ4:

Table 7. AraReceipt statistics. Class counts show the five most and least frequent labels; full per-class counts accompany the dataset release.
<table><tr><td>Property</td><td>Value</td><td>Class</td><td>Regions</td></tr><tr><td>Images</td><td>100</td><td>OTHERS</td><td>1,598</td></tr><tr><td>Annotated regions</td><td>4,609</td><td>PROD PRICE_VALUE</td><td>466</td></tr><tr><td>Regions/image</td><td>46.1 ± 21.6</td><td>PROD_QUANTITY_VALUE</td><td>461</td></tr><tr><td>Arabic-script regions</td><td>52.0%</td><td>PROD_ITEM_VALUE</td><td>402</td></tr><tr><td>Numeric-only regions</td><td>43.6%</td><td>SUBTOTAL_VALUE</td><td>225</td></tr><tr><td>Latin-script regions</td><td>4.0%</td><td></td><td></td></tr><tr><td></td><td></td><td>TAX_KEY</td><td>21</td></tr><tr><td></td><td></td><td>TIPS_KEY</td><td>10</td></tr><tr><td></td><td></td><td>TIPS_VALUE</td><td>9</td></tr><tr><td></td><td></td><td>STORE_NAME_KEY</td><td>1</td></tr></table>

GAIDA transferred to Arabic receipts through configuration-level adaptation alone, with no changes to the architecture or implementation logic, demonstrating cross-language reusability, though not eficiency gains over a manual Arabic baseline.

## 5 Discussion

Across the four questions, a consistent picture emerges. Pre-annotation quality depends on OCR reliability, model choice, and—to a lesser and less consistent degree—prompting strategy (RQ1, RQ2). Reviewing imperfect pre-annotations nonetheless substantially reduced stage-specific annotation time (RQ3), supporting the premise that value comes from redirecting human efort from creation to targeted correction rather than from eliminating it. RQ4 shows the same workflow transfers to Arabic receipts through configuration alone.

The two benchmarks also suggest where each assistance stage pays of. FUNSD with sparse relational structure and noisy scans, incurred the higher manual region-creation cost (13:30 vs. 8:49 per image) and gained most in absolute terms from OCR assistance, while its four relational classes kept manual labeling comparatively slow despite the small schema. WildReceipt, with dense but locallycued fields, was cheaper to transcribe manually yet showed the larger relative gains from both stages. This is consistent with a simple heuristic—layout-noisy, relational documents benefit most from OCR pre-extraction, whereas dense, lexically-cued documents benefit most from label pre-annotation—though our two-dataset, single-annotator design can motivate this hypothesis, not establish it.

## Threats to validity and future work.

– Annotator and ordering. The timing study involved one bilingual annotator in a fixed manual-before-assisted order. The one-week gap between passes and identical target annotations in both conditions mitigate immediate memory and quality-diference efects, but residual familiarity may inflate the assisted-condition gains. Multiple annotators and counterbalanced session orders would separate assistance efects from familiarity and speed variability.

– Stage-specific timing. OCR and semantic labeling were timed independently, so the reported reductions characterize each assistance stage but cannot be combined into an end-to-end workflow estimate. Larger studies should time continuous sessions, cover the full WildReceipt test set, add an AraReceipt manual baseline, and report variance statistics from the per-image logs.

– Evaluation-set tuning. The dataset-refined prompts were developed from errors observed on the evaluated FUNSD and WildReceipt sets; their scores therefore reflect iterative engineering rather than out-of-sample performance. The model comparison and the label-definition and decision-order prompts were fixed before evaluation and are unafected. Held-out development splits would enable generalization claims.

– Dataset validation. AraReceipt was annotated by one native Arabic speaker without independent review. A second annotation pass and inter-annotator agreement analysis would strengthen its validation.

Beyond these remedies, future work will extend the evaluation across languages, OCR engines, and LLM backends, and analyze interaction logs to identify which OCR and labeling errors contribute most to correction efort.

## 6 Conclusion

We proposed a modular human-in-the-loop annotation architecture and introduced GAIDA as its research-prototype reference implementation, separating OCR processing, LLM-based semantic pre-annotation, and human validation into interchangeable components with stage-specific efort measurement. In initial timing experiments on FUNSD and WildReceipt, OCR assistance reduced region-creation and transcription time by up to 80.9%, and reviewing LLMgenerated labels reduced semantic-labeling time by up to 73.7%, despite preannotations reaching only 78.86% and 86.98% Micro F1, respectively. Sensitivity analysis showed model choice mattered far more than prompt elaboration: more prescriptive prompts did not consistently help, and only guidance derived from observed errors did. These results reflect an initial single-annotator, fixed-order, stage-specific study in which labeling was timed against visible ground truth and the dataset-refined prompts were tuned on the evaluated sets (Sect. 5). We further applied GAIDA’s unmodified architecture, adapting only OCR and prompt configuration for script and layout, to build and publicly release AraReceipt, a 100-image, 25-class Arabic receipt dataset. Within these bounds, the results support the central premise of the architecture: pre-annotations need not be accurate to be valuable when they redirect human efort from creation to targeted correction.

## References

1. Abdallah, A., Abdalla, M., Elkasaby, M., Elbendary, Y., Jatowt, A.: Amurd: Annotated arabic-english receipt dataset for key information extraction and classification. arXiv preprint arXiv:2309.09800 (2023)

2. Abdallah, A., Mounis, M., Abdalla, M., Kasem, M.S., Mahmoud, M., Abdelhalim, I., Elkasaby, M., ElBendary, Y., Jatowt, A.: Receiptsense: Beyond traditional ocr–a dataset for receipt understanding. arXiv preprint arXiv:2406.04493 (2024)

3. Colakoglu, G., Solmaz, G., Fürst, J.: Problem solved? information extraction design space for layout-rich documents using llms. Findings of the Association for Computational Linguistics: EMNLP 2025, 17908–17927 (2025)

4. Cui, C., Sun, T., Lin, M., Gao, T., Zhang, Y., Liu, J., Wang, X., Zhang, Z., Zhou, C., Liu, H., et al.: Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595 (2025)

5. Ding, Y., Han, S.C., Lee, J., Hovy, E.: Deep learning based visually rich document content understanding: A survey. Artificial Intelligence Review 59(4), 114 (2026)

6. Goel, A., Gueta, A., Gilon, O., Liu, C., Erell, S., Nguyen, L.H., Hao, X., Jaber, B., Reddy, S., Kartha, R., et al.: Llms accelerate annotation for medical information extraction. In: machine learning for health (ML4H). pp. 82–100. PMLR (2023)

7. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)

8. Heakl, A., Sohail, M.A., Ranjan, M., Elbadry, R., Ahmad, G.S., El-Geish, M., Maher, O., Shen, Z., Khan, F.S., Khan, S.: Kitab-bench: A comprehensive multidomain benchmark for arabic ocr and document understanding. In: Findings of the Association for Computational Linguistics: ACL 2025. pp. 22006–22024 (2025)

9. Jaume, G., Ekenel, H.K., Thiran, J.P.: FUNSD: A dataset for form understanding in noisy scanned documents. In: 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW). pp. 1–6. IEEE (2019)

10. Klie, J.C., Castilho, R.E.d., Gurevych, I.: Analyzing dataset annotation quality management in the wild. Computational Linguistics 50(3), 817–866 (2024)

11. Laatiri, S., Ratnamogan, P., Tang, J., Lam, L., Vanhufel, W., Caspani, F.: Information redundancy and biases in public document information extraction benchmarks. In: Document Analysis and Recognition – ICDAR 2023. pp. 280–294 (2023)

12. Liao, H.W., Klugmann, C., Kondermann, D., Mahmood, R.: Minority reports: Balancing cost and quality in ground truth data annotation. arXiv preprint arXiv:2504.09341 (2025)

13. Ma, Y., Zang, Y., Chen, L., Chen, M., Jiao, Y., Li, X., Lu, X., Liu, Z., Ma, Y., Dong, X., et al.: Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. Advances in Neural Information Processing Systems 37, 95963– 96010 (2024)

14. Naparstek, O., Azulai, O., Shapira, I., Amrani, E., Yaroker, Y., Burshtein, Y., Pony, R., Rubinstein, N., Dahood, F.A., Prince, O., Friedman, I., Auer, C., Livathinos, N., Lysak, M., Nassar, A., Staar, P., Barzelay, U.: KVP10k: A comprehensive dataset for key-value pair extraction in business documents. In: ICDAR. pp. 97– 116 (2024). https://doi.org/10.1007/978-3-031-70533-5\_7

15. OpenAI: gpt-oss-120b & gpt-oss-20b model card. preprint arXiv:2508.10925 (2025)

16. Pangakis, N., Wolken, S.: Keeping humans in the loop: human-centered automated annotation with generative ai. In: Proceedings of the International AAAI Conference on Web and Social Media. vol. 19, pp. 1471–1492 (2025)

17. Saout, T., Lardeux, F., Saubion, F.: An overview of data extraction from invoices. IEEE Access 12, 19872–19886 (2024)

18. Schroeder, H., Roy, D., Kabbara, J.: Just put a human in the loop? investigating LLM-assisted annotation for subjective tasks. In: Findings of the Association for Computational Linguistics: ACL 2025. pp. 25771–25795 (2025)

19. Shen, J., Peiyue, Y., Ghosh, A., Mai, Y., Dahlmeier, D.: Ocr or not? rethinking document information extraction in the mllms era with real-world large-scale datasets. In: Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics. pp. 385–396 (2026)

20. Smith, R.: An overview of the tesseract ocr engine. In: Proc. Ninth Int. Conference on Document Analysis and Recognition (ICDAR). pp. 629–633 (2007)

21. Sun, H., Kuang, Z., Yue, X., Lin, C., Zhang, W.: Spatial dual-modality graph reasoning for key information extraction. CoRR abs/2103.14470 (2021)

22. Sun, Q., Li, S., Bi, T., Huynh, D.Q., Reynolds, M., Luo, Y., Liu, W.: DocSpiral: A platform for integrated assistive document annotation through human-in-thespiral. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations). pp. 267–274 (2025)

23. van Strien, D., Beelen, K., Ardanuy, M.C., Hosseini, K., McGillivray, B., Colavizza, G.: Assessing the impact of ocr quality on downstream nlp tasks. In: Proceedings of the 12th International Conference on Agents and Artificial Intelligence. pp. 484– 496 (2020)

24. Wang, D., Raman, N., Sibue, M., Ma, Z., Babkin, P., Kaur, S., Pei, Y., Nourbakhsh, A., Liu, X.: DocLLM: A layout-aware generative language model for multimodal document understanding. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. pp. 8529–8548 (2024)

25. Wang, X., Kim, H., Rahman, S., Mitra, K., Miao, Z.: Human-llm collaborative annotation through efective verification of LLM labels. In: Proceedings of the CHI Conference on Human Factors in Computing Systems, CHI 2024, Honolulu, HI, USA, May 11-16, 2024. pp. 303:1–303:21 (2024)

26. Yang, Z., Tang, J., Li, Z., Wang, P., Wan, J., Zhong, H., Liu, X., Yang, M., Wang, P., Bai, S., Jin, L., Lin, J.: Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 21744– 21754 (2025)

27. Yoon, C.O., Lee, W., Jang, S., Choi, K., Jung, M., Choi, D.: Language, OCR, form independent (LOFI) pipeline for industrial document information extraction. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track. pp. 1056–1067 (2024)

28. Yu, L., Miao, J., Sun, X., Chen, J., Hauptmann, A.G., Dai, H., Wei, W.: Documentnet: Bridging the data gap in document pre-training. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 707–722 (2023)

29. Zhang, H., Fu, X., Carroll, J.M.: Augmenting image annotation: A human-lmm collaborative framework for eficient object selection and label generation. arXiv preprint arXiv:2503.11096 (2025)

30. Zhou, Y., Wendt, J.B., Potti, N., Xie, J., Tata, S.: Selective labeling: How to radically lower data-labeling costs for document extraction models. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 3847–3860 (2023)