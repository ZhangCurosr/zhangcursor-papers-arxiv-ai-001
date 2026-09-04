# LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening

Muhammad Ashad Kabir

School of Computing, Mathematics and Engineering

Charles Sturt University

NSW, Australia

akabir@csu.edu.au

Sirajam Munira

Department of Computer Science

Rensselaer Polytechnic Institute

NY, USA

munirs@rpi.edu

Abstract—Early screening of chronic kidney disease (CKD) is critical for timely intervention, yet most machine learning (ML) and deep learning (DL) approaches require labeled data and model training, limiting their use in real-world screening settings. This study evaluates the effectiveness of large language models (LLMs) for CKD screening under zero-shot and fewshot in-context learning settings and compares them with traditional ML and DL methods. We propose a framework that uses clinically selected tabular features and structured prompt templates to enable LLM-based inference without task-specific training. LLM performance is evaluated across multiple prompt styles, feature configurations, and data settings, and compared with standard ML, DL, and tabular foundation model (TFM) baselines, and existing CKD screening tools. The results show that LLMs can achieve competitive performance using only a small number of examples, often matching or outperforming traditional approaches in low-data settings. However, their performance remains model-dependent and less stable as input complexity increases. In contrast, ML, DL, and TFM models show more consistent improvement with larger training data. Overall, the findings highlight a trade-off between data efficiency and stability, suggesting that LLMs may serve as a flexible complementary approach for CKD screening when labeled data are limited.

Index Terms—Chronic Kidney Disease, Large Language Model, Few-shot, Zero-shot, Machine Learning, Deep Learning, Low-resource Healthcare.

## I. INTRODUCTION

Chronic kidney disease (CKD) is a progressive condition characterized by sustained loss of kidney function or persistent markers of kidney damage. If undetected, CKD can progress to irreversible renal decline and end-stage renal disease, requiring costly renal replacement therapy [18]. CKD affects approximately 850 million people worldwide, nearly 10% of the global population [3]. The burden is particularly severe in low- and middle-income countries (LMICs), where nearly 80% of individuals with CKD reside and where prevalence and mortality are higher than in high-income settings [4], [25]. In South Asia, community-based studies report high CKD prevalence and low awareness, including studies from

Bangladesh showing prevalence around 22% in rural and periurban populations [24].

Early-stage CKD is often clinically silent, yet timely detection enables interventions such as blood pressure control, diabetes management, and nephroprotective therapy that can slow disease progression [27]. Standard screening depends on laboratory measurements, including serum creatinine, estimated glomerular filtration rate (eGFR), and albumin-tocreatinine ratio (ACR), which require diagnostic infrastructure and trained personnel [26]. These requirements limit population-level screening in rural and resource-constrained settings, where laboratory tests may be unavailable, costly, or inconsistently recorded. Consequently, there is a need for scalable, data-efficient, and context-aware CKD screening methods that can operate under heterogeneous feature availability.

Conventional machine learning (ML) and deep learning (DL) models have shown promise for CKD detection, but most rely on fixed feature spaces, well-curated datasets, and stable deployment conditions [23]. These assumptions are often violated in real-world clinical screening: biomarkers may be missing, feature definitions may differ across sites, and population distributions may shift. Although imputation and feature-masking methods can partially address missingness, they may reduce reliability when early-stage CKD signals are subtle [14]. Prior CKD prediction studies have relied on conventional supervised ML pipelines with fixed feature sets and labeled training data [9]. While effective under controlled evaluation settings, such models are less adaptable when deployed across screening contexts with missing variables, heterogeneous inputs, or shifting population characteristics.

Large language models (LLMs) offer a complementary approach to CKD screening. LLMs can perform zero-shot and few-shot in-context learning [2], reducing their reliance on large labeled datasets. A recent study [8] showed that feature-guided zero-shot prompting can match or improve CKD screening performance relative to full-feature inputs. These capabilities are particularly relevant to CKD screening, where risk factors are heterogeneous and data availability varies across populations and care settings [6]. However, the extent to which LLMs can function as quantitative CKD screening models, beyond their established roles in clinical reasoning and communication, remains underexplored.

![](images/59e08f5c39fe624d25d43bdc97009c9014faffd6f98f2a66e662012072b7f256.jpg)  
Fig. 1: Overview of the LLM4CKD evaluation framework.

Motivated by these gaps, we propose LLM4CKD, a framework for evaluating large language models for CKD detection in low-resource screening contexts (i.e., labeled data or diagnostic resources are limited). Our main contributions are as follows:

• We present a comprehensive applied study of LLMs for low-resource early-stage CKD screening under zero-shot and few-shot in-context learning settings, and benchmark their performance against conventional ML, DL, and tabular foundation-model (TFM) baselines.

• We investigate the clinical plausibility of LLM decision behavior by comparing LLM-induced feature rankings with ML-based feature importance and epidemiological associations, providing insight into whether LLMs rely on clinically meaningful CKD risk factors.

• We compare LLMs with established CKD screening tools and conduct an independent-dataset evaluation to assess performance across datasets.

## A. Related Work and Research Gaps

Existing CKD screening tools and risk scores provide useful community-level assessment mechanisms, but many were developed using cohorts from high-income countries and often emphasize later-stage disease [28]. This limits their applicability in LMICs, where early detection is most needed and where laboratory access, population characteristics, and healthcare pathways differ substantially. Similarly, ML- and DL-based CKD prediction studies typically assume complete or consistently measured feature sets, making them less robust to missing variables and heterogeneous inputs that are common in real-world screening environments [23].

Recent work has explored LLMs across a range of nephrology and CKD-related applications, with most studies focusing on clinical support, knowledge extraction, and patientfacing communication. LLMs have been evaluated for kidney pathology support, nephrology reasoning, biopsy recommendation, and guideline-oriented decision support [29], [31], [32]. Other studies have applied LLMs to CKD knowledgegraph construction and feature interpretation within hybrid prediction pipelines [12], [16], as well as patient education and other kidney-related applications [17], [19]. Reviews similarly highlight their potential for summarization, decision support, education, and knowledge modeling, while noting concerns regarding reliability, consistency, and clinical validation [6], [23]. A recent study evaluated zero-shot CKD screening from structured patient data and showed that ML-selected features generally matched or improved LLM performance relative to full-feature inputs [8].

Despite this progress, several gaps remain. First, the behavior of LLM-based CKD screening beyond zero-shot featureguided inference, particularly under few-shot settings, remains underexplored. Second, to our knowledge, no prior study has systematically compared zero- and few-shot LLMs with conventional ML, DL, and tabular foundation-model baselines for CKD screening. Third, the alignment between LLMderived feature importance, ML-based feature ranking, and known epidemiological risk factors has not been systematically examined. Finally, limited evidence exists on LLM-based CKD screening performance across datasets or in comparison with established CKD risk assessment tools.

## II. METHODOLOGY

Fig. 1 provides an overview of the proposed LLM4CKD framework for CKD screening. The framework comprises five key steps: (i) structured-to-text feature serialization, (ii) prompt construction, (iii) LLM-based inference, (iv) clinically grounded feature-ranking analysis, and (v) performance evaluation, including comparison with established CKD screening tools and independent-dataset evaluation to examine crossdataset performance.

## A. Dataset Preparation

We evaluated LLM4CKD using two datasets: a communitybased Bangladeshi cohort for primary evaluation and an independent hospital-based Indian cohort for cross-dataset evaluation. The primary dataset, referred to as Dataset-1, was collected from the Mirzapur sub-district of Tangail, Bangladesh [24]. It was obtained through age-stratified random sampling of adults aged 18 years and above under the Mirzapur demographic surveillance system. CKD was diagnosed using reduced kidney function, measured by estimated glomerular filtration rate (eGFR), and/or elevated urine albumin-to-creatinine ratio (uACR), according to standard clinical guidelines. To reduce label leakage, direct diagnostic markers used to define CKD status, including eGFR and uACR, were not used as model input features. After removing incomplete records, the final Dataset-1 cohort [9] contained 284 participants, consisting of 112 early-stage CKD cases and 172 non-CKD individuals. The CKD cases corresponded to stages 1–3, consistent with the study objective of lowresource early-stage CKD screening. Dataset-1 contains 24 input variables (Table I) covering socio-demographic characteristics, lifestyle habits, medical history, clinical examination findings, and pathology-test features.

TABLE I: All features and selected features (underlined) retained after harmonization for each dataset.
<table><tr><td>Dataset</td><td>Considered features</td></tr><tr><td>Dataset-1 [9]</td><td>Age, Gender, Sleeping duration, History of hypertension, History of diabetes, Body mass index, Presence of red blood cells in urine, Anemia, Family history of hypertension, Illiterate, Occupation, Marital status, Tobacco smoker, Smokeless tobacco, Heart disease, Stroke, Family history of diabetes, Family history of CKD, Abdominal obesity, Undernutrition, Serum albumin, Hyper-cholesterolemia, HDL cholesterol, Hypertriglyceridemia</td></tr><tr><td>Dataset-2 [22]</td><td>Age, History of hypertension, History of Diabetes, Presence of red blood cells in urine, Anemia, Blood pressure, Specific gravity, Albumin, Sugar, Pus cell, Pus cell clumps, Bacteria, Blood glucose random, Blood urea, Sodium, Potassium, Hemoglobin, Coronary artery disease, Appetite condition, Pedal edema, White blood cell count, Red blood cell count, Packed cell volume</td></tr></table>

For independent-dataset evaluation, we used Dataset-2, the UCI CKD dataset [22]. Dataset-2 was collected from a hospital in India and is widely used as a benchmark for CKD detection research. It contains 400 patient records, including 250 CKD cases and 150 non-CKD controls. CKD stages are not reported in this dataset; therefore, Dataset-2 was used only for independent evaluation of cross-dataset performance, rather than early-stage-specific performance. To reduce label leakage, serum creatinine was excluded from the model inputs because it is a direct kidney-function marker strongly related to CKD diagnosis. The resulting Dataset-2 feature set contained 23 variables (Table I) spanning demographic, clinical, medicalhistory, and non-diagnostic laboratory features.

To ensure consistency across datasets, we performed systematic feature harmonization. Variables representing the same clinical concepts were standardized by unifying feature names and value encodings into a common, semantically interpretable format suitable for LLM inference. This harmonization enabled consistent prompt construction and comparable evaluation across datasets. The complete list of harmonized features considered in this study is presented in Table I.

Feature selection was informed by prior machine learning analysis on Dataset-1 [9]. That analysis applied ten complementary feature selection methods, including recursive feature elimination with multiple ML estimators, to identify a compact subset of clinically meaningful and readily obtainable variables predictive of early-stage CKD. The selected features are underlined in Table I. These selected Dataset-1 features were subsequently mapped to Dataset-2 by identifying variables with equivalent clinical interpretation. Because the datasets did not share identical feature spaces, only Dataset-2 variables matching the selected features from Dataset-1 were retained in the selected-feature setting.

## B. Feature Serialization and Prompt Style

To enable large language models to process structured tabular data, each patient record is converted into a textual representation through feature serialization. This step maps harmonized clinical variables into text while preserving feature names, value encodings, and clinical semantics. The same serialization procedure was applied to both the all-feature and selected-feature settings to ensure comparable evaluation across prompt configurations and datasets.

Following TabLLM [5], we considered two serialization styles: list-based and text-based templates. In the list-based template, each feature is represented as an explicit key– value pair, e.g., History of hypertension: Yes; History of diabetes: No; Anemia: Yes. This format preserves feature boundaries and provides a structured representation of each patient profile. In the text-based template, the same information is expressed as a concise clinical description, e.g., The patient has a history of hypertension. The patient has diabetes. This format produces a more natural narrative while retaining the underlying feature-value information. Using both styles allows us to evaluate whether LLM-based CKD screening is sensitive to prompt serialization format.

## C. Prompt Construction and Templates

Serialized patient records were embedded into structured prompts to evaluate LLM-based inference. We used two prompt styles: instruction-style prompts (Table II), which explicitly specify the binary CKD classification task and expected output, and chat-style prompts (Table III), which present the same task in a conversational format. Both styles used the same harmonized feature representations and were evaluated under all-feature and selected-feature settings.

In the zero-shot setting, prompts contained only the task instruction and the query patient record. In the few-shot setting, labeled example records were added before the query record using the same serialization format. All prompts constrained the output to one of two labels: CKD or Non-CKD.

## D. Large Language Models Used for Inference

We evaluated LLM4CKD using five instruction-tuned LLMs: Gemma-2-9B [21], Llama-3-8B [15], Qwen-3-8B [33], Mistral-7B [7], and GPT-4o-mini [20].

TABLE II: Instruction-style list prompt template used for inference. For the few-shot setting, labeled patient-record examples were inserted before the query patient record.

Task: You are a clinical classification model. Given a   
patient’s structured clinical characteristics, determine   
whether the patient has chronic kidney disease (CKD).   
The target label is binary, where 1 indicates CKD and   
0 indicates non-CKD.   
Input: Each patient record consists of structured clini  
cal features, including demographic variables, medical   
history, laboratory measurements, and other relevant   
clinical attributes.   
Instruction: Respond with exactly one token: 0 or   
1. Do not provide any explanation, justification, or   
additional text.   
/<sub>\*\*</sub> Example #1 <sub>\*\*</sub>/   
/<sub>\*\*</sub> Example #2 <sub>\*\*</sub>/   
/<sub>\*\*</sub> .... <sub>\*\*</sub>/   
/<sub>\*\*</sub> Example #k <sub>\*\*</sub>/   
Now classify the following patient.   
History of diabetes: Yes,   
History of hypertension: No,   
Output:

## TABLE III: Chat-style text prompt template for Llama.

<|begin\_of\_text|>\n   
<|start\_header\_id|>system   
<|end\_header\_id|>\n   
You are a clinical classification model. . . .   
Each patient record consists of . . .   
Respond with exactly one token. . .   
<|eot\_id|>\n   
<|start\_header\_id|>user   
<|end\_header\_id|>\n   
The patient is between 40 and 49 years. The   
patient is female. The patient is overweight.   
The patient has a family history of hypertension.   
The patient has diabetes. The patient has   
hypertension. The patient sleeps less than   
seven hours per day. The patient does not have   
anemia. The patient has red blood cells in   
the urine. .. .   
<|eot\_id|>\n   
<|start\_header\_id|>assistant   
<|end\_header\_id|>\n

The model set includes both open-weight and proprietary LLMs, covering different model families, scales, and deployment settings. Gemma-2, Llama-3, Qwen-3, and Mistral provide compact open-weight baselines, while GPT-4o-mini provides a proprietary API-based baseline.

All models were evaluated using the same harmonized feature representations, all-feature and selected-feature settings, and zero-shot and few-shot protocols. For open-weight models, we evaluated both instruction-based and chat-style prompts. Since GPT-4o-mini is accessed through a chat-completion interface, it was evaluated only with the chat-style prompt. For probabilistic classification, token-level log-probabilities for the two constrained output labels, 0 and 1, were converted into normalized class likelihoods, where 1 denotes CKD, and 0 denotes non-CKD.

## E. Feature Ranking and Explainability

To assess interpretability and clinical plausibility, we analyze how LLMs prioritize CKD risk factors and compare their behavior with a data-driven machine learning baseline. Since LLMs do not provide explicit feature attributions for structured inputs, we use a surrogate ML model trained on the same dataset to approximate feature importance.

Feature importance is computed using SHapley Additive ex-Planations (SHAP), which provides consistent, model-agnostic estimates of each feature’s contribution to prediction. SHAP values are aggregated across samples to obtain global feature rankings, capturing the relative influence of clinical variables in CKD classification. These rankings are compared with ML feature importance and known epidemiological associations. This comparison examines whether LLM-induced feature prioritization aligns with ML-based importance and clinically established CKD risk factors.

## F. Model Evaluation

For zero-shot LLM evaluation, models were evaluated without labeled examples, training, or parameter updates. Since zeroshot inference does not require a training split, each model was evaluated directly on the full dataset.

For few-shot LLM evaluation and supervised baselines, each dataset was partitioned into 80% training and 20% testing using stratified sampling. The 20% held-out test partition was consistently used for final evaluation across all models. Fewshot in-context examples for LLMs and low-data training subsets of size 4, 8, 16, and 32 for supervised ML, DL, and tabular foundation-model (TFM) baselines were sampled only from the 80% training partition.

To account for sampling variability, each configuration was repeated across five random seeds: {0, 1, 32, 42, 1024}. For each seed, the same train–test partitioning and sampled training examples were used across all models to ensure paired comparison. We report mean performance across runs using balanced accuracy and probabilistic metrics, including Brier loss. Statistical significance between model pairs was assessed using paired permutation tests and mixed-effects analysis on per-sample Brier loss across repeated runs.

![](images/ceb7fd98393dc80cbe5dfd2ad48582bea4c20f939102d96825f27d278c04de78.jpg)  
(a) Gemma-2

![](images/3fe888dc11f53a509e27073e15b64948fff1ffc51b4e7fb142edb6b6300ed6ce.jpg)  
(b) Llama-3

All features Selected features  
![](images/a710b606551d54f0a80eca0c6cb8d7365d86f66754a79563f492862be74986f2.jpg)  
(c) Qwen-3

![](images/56104588a84354bb63a332174d48e1648dabd951966efef46c1e3f05157f47e5.jpg)  
(d) Mistral

![](images/c388133410370803a155f66c54fcf37893275f739e742e307bbbd3d508d5b0f5.jpg)  
(e) GPT-4o-mini  
Fig. 2: Zero-shot comparison of all versus selected features and the performance of different prompt styles (instruction vs. chat) and templates (list vs. text) for LLMs. Bars report balanced accuracy, while statistical significance between all- and selected-feature settings is assessed using a paired permutation test on per-sample Brier loss. Significance levels are indicated by asterisks: ${ } ^ { * } p < 0 . 0 5 ,$ $^ { * * } p < 0 . 0 1$ and $^ { * * * } p < 0 . 0 0 1$ . Bold denotes the best-performing prompt style and template.

For LLM inference, open-weight models were executed locally with temperature set to 0 to ensure deterministic decoding. The pretrained weights of Gemma-2-9B<sup>-1</sup>, Llama-3-8B<sup>0</sup>, Qwen-3- 8B<sup>1</sup>, and Mistral-7B<sup>2</sup> were obtained from Hugging Face. GPT-4o-mini was accessed through the OpenAI API. For probabilistic classification, we computed normalized class probabilities from output token log-probabilities over the predefined label tokens corresponding to CKD and non-CKD. These normalized probabilities were used to derive predicted labels, Brier loss, and confidence scores.

For conventional ML baselines, models were implemented using scikit-learn. Unless otherwise specified, default hyperparameters were used, with class\_weight=’balanced’ applied where supported to account for class imbalance. DL and TFM baselines were implemented using their standard public implementations: TabNet was implemented in Py-Torch, NODE was implemented using the official repository,<sup>3</sup> TabPFN was installed using the tabpfn Python package, and SAINT was implemented using the official repository.<sup>4</sup> Modelinternal random states were fixed at 42 where applicable, while data partitioning and subset sampling followed the five evaluation seeds above.

The source code for our experimental pipeline is publicly available.<sup>5</sup> Experiments were conducted on a workstation equipped with an NVIDIA GeForce RTX 3080 16GB GPU.

## III. RESULTS AND DISCUSSION

## A. Zero-Shot Analysis ofFeature Selection and Prompt Design

Fig. 2 compares all-feature and selected-feature inputs under the zero-shot setting across prompt styles and serialization templates for each LLM. No labeled examples, training data, or parameter updates were used in this experiment. Performance is reported using balanced accuracy, while statistical significance between paired all-feature and selected-feature predictions is assessed using a paired permutation test on persample Brier loss over the same samples. This comparison evaluates whether reducing input complexity improves zeroshot LLM-based CKD screening.

Overall, selected features improve or maintain performance for most LLMs, but the effect remains model- and promptdependent. Llama-3 benefits most consistently, with selectedfeature inputs improving balanced accuracy across all prompttemplate combinations. The largest gain occurs in the Chat+Text setting, where balanced accuracy increases from 0.51 to 0.74. Mistral also benefits in most settings, especially under instruction-style prompts, increasing from 0.56 to 0.77 with Instruction+List and from 0.72 to 0.79 with Instruction+Text. However, its Chat+List performance drops from 0.67 to 0.50, indicating that feature selection alone is insufficient without an effective prompt format.

Qwen-3 shows mixed behavior. Selected features improve Instruction+List from 0.62 to 0.78 and Instruction+Text from 0.52 to 0.62, while Chat+List remains at 0.50 and Chat+Text decreases from 0.63 to 0.53. In contrast, the Instruction+Text setting changes only slightly, and the Chat+List setting remains near chance-level balanced accuracy. GPT-4o-mini, evaluated only with chat-style prompts, shows stable behavior: selected features improve Chat+List from 0.70 to 0.77 but slightly reduce Chat+Text from 0.75 to 0.70.

![](images/3c9f3e01250b8c86369d41e240ad9859307c8470ad4fb102857b7d18887c6f30.jpg)  
(a) LLM

![](images/0ec04dbaae7ca18116050aa78752a40bad222bc59178fd6d941510c946996f00.jpg)  
(b) Machine learning

![](images/093d9f51215f39c59df9e848c2ff715e39132de06eb0ebb782dcafb896487458.jpg)  
(c) Deep learning/TFM  
Fig. 3: Balanced accuracy comparison using selected features across (a) LLMs, (b) ML models, and (c) DL/TFM models. LLM performance is evaluated using varying numbers of in-context examples (shots), while supervised ML and DL baselines are trained on corresponding labeled subsets and TabPFN uses them as labeled context samples. Results are averaged over five runs with different random seeds; within each run, identical training/in-context and test samples are used across all models.

Gemma-2 differs from the other models. Selected features reduce balanced accuracy in most prompt-template settings, including Instruction+List, Chat+List, and Chat+Text, while Instruction+Text remains relatively stable. This suggests that Gemma-2 may rely more on broader contextual information from the full feature set, whereas other models often benefit from compact, clinically relevant inputs.

These findings show that selected-feature prompting reduces input complexity and improves zero-shot LLM-based CKD screening, particularly for Llama-3, Mistral, Qwen-3, and GPT-4o-mini. However, the gains are not universal: performance depends jointly on the LLM, prompt style, and serialization template. Instruction-style prompts are generally effective for open-weight models, while chat-style prompting remains necessary for GPT-4o-mini. These results highlight the importance of evaluating feature selection and prompt design together rather than treating them as independent choices.

## B. Low-Data Comparison with ML, DL, and Tabular Foundation Models

Fig. 3 compares LLMs with conventional ML and tabular DL and foundation-model baselines under limited labeled-data settings using the selected-feature set. LLMs are evaluated using zero-shot and few-shot in-context learning, whereas ML and tabular baselines are trained with progressively larger labeled subsets. For each seed, the same selected features, training, and in-context examples, and held-out test samples are used across models.

LLMs achieve competitive balanced accuracy with very few examples, but their performance does not consistently improve as the number of shots increases. Gemma-2, Qwen-3, and GPT-4o-mini improve from zero-shot to four-shot inference, reaching approximately 0.80–0.81 balanced accuracy. In contrast, Llama-3 and Mistral show weaker or declining performance as more examples are added, indicating sensitivity to incontext example composition. GPT-4o-mini is the most stable LLM, maintaining balanced accuracy around 0.80 across shots. Supervised ML models show a more predictable trend: performance generally improves as the number of labeled training samples increases. Random Forest, Logistic Regression, Extra Trees, XGBoost, and MLP improve or remain competitive as the training size grows from 4 to 32 samples. Tabular DL and foundation models show a similar scaling pattern, with TabPFN and NODE achieving the strongest performance at larger training sizes.

Overall, the results show a trade-off between data efficiency and scalability under the selected-feature setting. LLMs can provide strong performance with only a few in-context examples, making them attractive for low-resource CKD screening when labeled data are scarce. However, additional shots do not guarantee improvement. In contrast, supervised ML and tabular models require labeled training data but tend to benefit more consistently as data availability increases.

## C. Pairwise Brier Loss Comparison

Fig. 4 reports pairwise mean Brier loss differences across LLM, ML, and tabular DL and foundation-model groups. For each comparison, the plotted value is ∆ = Loss(Model 1) − Loss(Model 2); therefore, negative values favor Model 1 and positive values favor Model 2. Confidence intervals that do not cross zero indicate statistically significant differences under the mixed-effects model.

Within the LLM group, Qwen-3 shows the strongest overall probabilistic performance, achieving lower Brier loss than Llama-3 and Mistral while remaining competitive with Gemma-2 and GPT-4o-mini. This finding is consistent with Table IV, where Qwen-3 obtains strong Brier loss across shot settings, including 0.154 at four shots, 0.141 at 16 shots, and

![](images/c8b2df3a16ddba88ba705219e17b79f08096846007d8bd65f29fc857efbb78e3.jpg)  
Mean loss difference (a) LLM

![](images/c67e2c93cb89915345f0a485a02606137fb227df2ed8e00e6dcbdd51f65ff184.jpg)  
Mean loss difference (b) Deep Learning

![](images/1507656aa39852478c39049e73f1ccf776be6f514f950d6157adc71d15cc8206.jpg)  
Mean loss difference (d) Best LLM vs DL vs ML  
Mean loss difference (c) Machine Learning  
Fig. 4: Pairwise comparison of mean Brier loss differences across model families: (a) LLMs, (b) DL models, (c) ML models, and (d) best-performing LLMs versus DL versus ML models. Each point denotes the mean loss difference between two models (Model 1 − Model 2), with horizontal error bars showing 95% confidence intervals estimated from a mixedeffects model accounting for repeated stochastic evaluations. The vertical red dashed line indicates zero difference. Negative values indicate that Model 1 outperforms Model 2 (lower mean loss), while positive values indicate the opposite. Differences are considered statistically significant when the confidence interval does not intersect the zero line. Underlined model names denote the best-performing model(s) within their group.

0.146 at 32 shots. Qwen-3 also achieves the highest four-shot balanced accuracy among the top models (0.814) and high sensitivity (0.873), indicating strong early-stage CKD case detection in the extreme low-data regime. GPT-4o-mini and Gemma-2 form a competitive middle tier, while Llama-3 and Mistral generally show higher loss.

Among tabular DL and foundation models, NODE and

TABLE IV: Detailed performance comparison of the top representative LLM, ML, and tabular DL and foundationmodel baselines using selected features. Metrics include balanced accuracy, AUROC, macro-F1, sensitivity, Brier loss, and expected calibration error (ECE). Values are reported as means across five random-seed iterations, with standard deviations as subscripts and bias-corrected and accelerated (BCa) 95% confidence intervals in brackets.
<table><tr><td>Shot Model</td></tr><tr><td>4 Qwen</td><td>0.814±0.025</td><td>B. accuracy (↑) AUROC (↑) 0.867±0.032 0.796±0.024</td><td>F1 (↑)</td><td>Sensitivity (↑) Brier (↓) 0.873±0.067</td><td>0.154±0.0220.095±0.054</td><td>ECE (↓)</td></tr><tr><td>TabPFN</td><td></td><td>[0.763, 0.854] 0.658±0.105</td><td>[0.813, 0.907] [0.747, 0.842] 0.813±0.051 0.656±0.116</td><td></td><td>0.436±0.1860.176±0.0320.096±0.037</td><td>[0.799, 0.925] [0.128, 0.185] [0.071, 0.097]</td></tr><tr><td>NODE</td><td>[0.609, 0.701] 0.768±0.053</td><td></td><td>[0.748, 0.865] [0.604, 0.710] 0.863±0.0360.764±0.043</td><td>[0.357, 0.513] 0.718±0.152</td><td>[0.155, 0.200] [0.064, 0.106] 0.160±0.0170.118±0.026</td><td></td></tr><tr><td></td><td>[0.711, 0.811]</td><td></td><td>[0.810, 0.903] [0.708, 0.813]</td><td>[0.635, 0.790]</td><td></td><td>[0.132, 0.194] [0.068, 0.132]</td></tr><tr><td>MLP</td><td>0.754±0.068</td><td></td><td>0.827±0.061 0.757±0.069</td><td>0.645±0.142</td><td>0.161±0.0300.111±0.041</td><td></td></tr><tr><td>8 Qwen</td><td>[0.712, 0.798] 0.712±0.115</td><td></td><td>[0.766, 0.877] [0.714, 0.806] 0.877±0.033 0.709±0.126</td><td>[0.574, 0.717]</td><td>[0.134, 0.190] [0.071, 0.125]</td><td></td></tr><tr><td>TabPFN</td><td>[0.664, 0.761]</td><td></td><td>[0.829, 0.915] [0.653, 0.767]</td><td>0.509±0.262 [0.418, 0.601]</td><td>0.161±0.033 0.105±0.067</td><td>[0.139, 0.186] [0.072, 0.108]</td></tr><tr><td></td><td>0.751±0.085 [0.696, 0.802]</td><td></td><td>0.847±0.0440.756±0.086</td><td>0.627±0.142</td><td></td><td>0.170±0.027 0.117±0.031</td></tr><tr><td>NODE</td><td>0.753±0.071</td><td>0.868±0.0430.753±0.065</td><td>[0.779, 0.895] [0.703, 0.813]</td><td>[0.539, 0.714] 0.655±0.1840.153±0.0230.095±0.033</td><td></td><td>[0.148, 0.196] [0.069, 0.140]</td></tr><tr><td></td><td>[0.693, 0.803]</td><td>[0.811, 0.909] [0.697, 0.807]</td><td></td><td>[0.570, 0.730]</td><td>[0.130, 0.179] [0.074, 0.097]</td><td></td></tr><tr><td>MLP</td><td>0.752±0.048</td><td>0.829±0.038 0.751±0.041</td><td></td><td>0.682±0.144</td><td>0.171±0.0120.124±0.028</td><td></td></tr><tr><td>16 Qwen</td><td>[0.698, 0.801]</td><td>[0.766, 0.880] [0.699, 0.804]</td><td></td><td>[0.599, 0.763]</td><td>[0.140, 0.208] [0.081, 0.145]</td><td></td></tr><tr><td></td><td>0.782±0.051</td><td></td><td>0.878±0.039 0.781±0.052</td><td>0.736±0.099</td><td>0.141±0.029 0.091±0.014</td><td></td></tr><tr><td>TabPFN</td><td>[0.729, 0.832] 0.762±0.042</td><td>0.859±0.056 0.765±0.037</td><td>[0.828, 0.917] [0.732, 0.834]</td><td>[0.648, 0.816] 0.655±0.123</td><td>0.161±0.030 0.141±0.063</td><td>[0.116, 0.168] [0.072, 0.091]</td></tr><tr><td>NODE</td><td>[0.713, 0.807]</td><td></td><td>[0.803, 0.901] [0.718, 0.814]</td><td>[0.566, 0.734]</td><td>[0.136, 0.189] [0.098, 0.162]</td><td></td></tr><tr><td></td><td>0.784±0.092</td><td>0.872±0.0540.780±0.091</td><td></td><td>0.727±0.193</td><td></td><td>0.146±0.0310.106±0.032</td></tr><tr><td>MLP</td><td>[0.735, 0.828] 0.755±0.056</td><td></td><td>[0.819, 0.912] [0.732, 0.830]</td><td>[0.644, 0.802]</td><td></td><td>[0.124, 0.172] [0.077, 0.111]</td></tr><tr><td></td><td>[0.701, 0.805]</td><td></td><td>0.830±0.0710.751±0.070</td><td>0.727±0.064</td><td></td><td>0.175±0.0440.153±0.034</td></tr><tr><td>32 Qwen</td><td>0.764±0.055</td><td>0.874±0.031 0.770±0.051</td><td>[0.771, 0.875] [0.700, 0.805]</td><td>[0.643, 0.803] 0.636±0.129</td><td></td><td>[0.145, 0.209] [0.111, 0.179]</td></tr><tr><td></td><td>[0.713, 0.810]</td><td>[0.828, 0.912] [0.720, 0.819]</td><td></td><td>[0.543, 0.721]</td><td>[0.122, 0.173] [0.068, 0.096]</td><td>0.146±0.023 0.095±0.029</td></tr><tr><td>TabPFN</td><td>0.820±0.041</td><td></td><td>0.871±0.0460.825±0.036</td><td>0.736±0.099</td><td>0.134±0.0280.105±0.049</td><td></td></tr><tr><td>NODE</td><td>[0.769, 0.865]</td><td></td><td>[0.818, 0.910] [0.779, 0.871]</td><td>[0.643, 0.815]</td><td></td><td>[0.113, 0.158] [0.075, 0.113]</td></tr><tr><td></td><td> $0 . 7 9 7 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td></td><td>0.871±0.0650.801±0.071</td><td>0.709±0.153</td><td></td><td>0.142±0.033 0.097±0.035</td></tr><tr><td></td><td>[0.753, 0.841]</td><td></td><td>[0.818, 0.910] [0.758, 0.847]</td><td>[0.627, 0.785]</td><td></td><td>[0.122, 0.166] [0.075, 0.097]</td></tr><tr><td>MLP</td><td> $0 . 7 7 1 { \scriptstyle \pm 0 . 1 4 5 }$ </td><td>0.814±0.1310.768±0.143</td><td></td><td>0.736±0.202</td><td>0.160±0.0550.140±0.066</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>[0.719, 0.817]</td><td></td><td>[0.757, 0.862] [0.719, 0.817]</td><td>[0.664, 0.803]</td><td>[0.133, 0.192] [0.106, 0.155]</td><td></td></tr></table>

TabPFN provide the strongest performance, with lower loss than SAINT and TabNet in most pairwise comparisons. Table IV further shows that their advantage becomes clearer as the number of labeled training samples increases. At 32 context samples, TabPFN achieves the best overall balanced accuracy among the top models (0.820) and the lowest Brier loss (0.122), while NODE also remains strong with balanced accuracy of 0.797 and Brier loss of 0.142. These results suggest that tabular foundation models benefit more consistently from additional labeled data than LLMs, particularly in probabilistic calibration.

For conventional ML models, the pairwise comparisons show a wider spread. MLP and stronger ensemble models form the leading group, whereas simpler or less stable models, such as LR and DT, tend to show higher loss. The detailed metrics in Table IV show that MLP remains a competitive supervised baseline, but its Brier loss is generally higher than the strongest LLM and tabular foundation-model baselines, ranging from 0.160 to 0.175 across training sizes. This indicates that although MLP achieves competitive classification performance, its probabilistic calibration is weaker than Qwen-3 in low-shot settings and TabPFN at larger training sizes.

The cross-family comparison in Fig. 4d shows that the best LLM, Qwen-3, is competitive with the strongest ML and tabular DL/foundation-model baselines. Table IV clarifies this trend: Qwen-3 is particularly strong when only a few incontext examples are available, whereas TabPFN and NODE become more competitive as labeled training data increase. Overall, selected-feature LLM prompting provides strong lowresource probabilistic performance, while supervised tabular models retain an advantage when sufficient labeled samples are available for training.

TABLE V: Comparison of feature importance rankings and epidemiological associations. The table reports an ML-derived reference ranking from the MLP model and rankings induced by LLM-based surrogate models. For each feature, the related risk ratio (RR) with its 95% CI is shown, along with the estimation type (standard or penalized).
<table><tr><td rowspan="3">Feature</td><td colspan="6">Feature ranking</td><td colspan="2">Epidemiological associations</td></tr><tr><td>ML Reference</td><td colspan="5">LLM-induced surrogate</td><td>95% CI</td><td>RR type</td></tr><tr><td>MLP</td><td>Gemma GPT Llama Mistral Qwen risk (RR)</td><td></td><td></td><td></td><td>Related</td><td></td><td></td></tr><tr><td>Hypertension</td><td>1</td><td>3</td><td>3</td><td>1</td><td>3</td><td>4.03*</td><td>[2.97, 5.48]</td><td>Standard</td></tr><tr><td> $\Delta \varrho { \mathrm { e } } \_ { - } 6 0 + y$ </td><td>2</td><td>1</td><td>5</td><td>3</td><td>4</td><td>3.68*†</td><td>[2.98, 4.55]</td><td>Penalized</td></tr><tr><td>RBC</td><td>3</td><td>4</td><td>6</td><td>9</td><td>6</td><td>2.80*†</td><td>[2.34, 3.34]</td><td>Penalized</td></tr><tr><td>Daily sleep &lt; 7h</td><td>4</td><td>8</td><td>7</td><td></td><td>8</td><td>1.75*</td><td>[1.33, 2.30]</td><td>Standard</td></tr><tr><td>Anemia</td><td>8</td><td>5</td><td>2</td><td></td><td>5</td><td>1.64*</td><td>[1.24, 2.17]</td><td>Standard</td></tr><tr><td>Diabetes</td><td>7</td><td>2</td><td>1</td><td></td><td>1</td><td>1.56*</td><td>[1.16, 2.09]</td><td>Standard</td></tr><tr><td>BMI_Obese</td><td>5</td><td>7</td><td>4</td><td></td><td>2</td><td>1.47*</td><td>[1.10, 1.94]</td><td>Standard</td></tr><tr><td>Family hypertension</td><td>9</td><td>9</td><td>9</td><td></td><td>7</td><td>1.08</td><td>[0.80, 1.45]</td><td>Standard</td></tr><tr><td>Gender_Male</td><td>6</td><td>6</td><td>8</td><td></td><td>9</td><td>0.77</td><td>[0.57, 1.05]</td><td>Standard</td></tr></table>

\* 95% confidence interval excluding 1.0 (p < 0.05)  
† penalized estimate computed due to perfect separation (absence of non-CKD cases)

## D. Feature Ranking and Clinical Plausibility

Table V compares the MLP-based ML reference ranking with LLM-derived feature rankings and epidemiological risk-ratio associations. The ML reference ranking emphasizes clinically established CKD risk factors, placing hypertension, age $\geq 6 0$ and RBC as the top three features. These variables also show strong epidemiological associations, with elevated risk ratios for CKD. This agreement suggests that the MLP reference captures major data-driven and clinically plausible risk patterns in the study cohort.

LLMs show partially overlapping but distinct ranking behavior. Several models assign high importance to diabetes, anemia, and obesity-related features, which are clinically meaningful CKD risk factors but ranked lower by the ML reference. Diabetes is ranked first by Llama-3 and Qwen-3 and second by GPT-4o-mini, while BMI Obese is ranked highest by Gemma-2. Hypertension remains highly ranked by most models, especially Mistral, which assigns it the top rank. In contrast, RBC is emphasized strongly by the ML reference but receives moderate or lower rankings from most LLMs, suggesting that some LLMs may underweight urinalysis-related evidence relative to broader comorbidity patterns.

Fig. 5 further quantifies ranking agreement using Spearman correlation. LLM rankings show moderate to strong agreement with each other, particularly among GPT-4o-mini, Qwen-3, and Llama-3, indicating that LLMs tend to follow a shared feature-prioritization pattern. However, correlations between the ML reference ranking and LLM rankings are generally low, showing that LLMs do not simply reproduce the supervised tabular importance structure. Among the LLMs, Gemma-2 shows the highest agreement with the ML reference, while GPT-4o-mini shows weaker agreement.

![](images/7ed90941d131f71049208958bfa22034ebf4266d99521e6af31acddc7a0d72df.jpg)  
Fig. 5: Spearman rank correlation heatmap comparing featureimportance rankings among large language models and the MLP model. Each cell reports the Spearman correlation coefficient (ρ) between a pair of models; darker colors indicate stronger agreement.

TABLE VI: Comparison of zero-shot LLM performance with established CKD screening tools on the full Dataset-1 cohort. Balanced accuracy (Bal. Acc.) is reported for both the LLMs and tools. Reported values are mean Brier loss differences (LLM−Tool), where negative values indicate better probabilistic performance of the LLM. Statistical significance is assessed using a paired permutation test on per-sample Brier loss.
<table><tr><td>Tool→</td><td></td><td>[SCORED [1] Kshirsagar [11] Thakkinstian [30] Kwon [13] Kearns [10]</td><td></td><td></td><td></td><td></td></tr><tr><td>LLM</td><td>Bal. Acc.</td><td>0.7635</td><td>0.7709</td><td>0.8103  $- 0 . 1 2 0 ^ { \ast \ast * }$ </td><td>0.7749 -0.206***</td><td>0.7829</td></tr><tr><td>Qwen Mistral</td><td>0.7829</td><td> $- 0 . 1 5 0 ^ { * * * }$   $- 0 . 1 4 8 ^ { * * * }$ </td><td> $- 0 . 1 1 1 ^ { \ast \ast \ast }$   $- 0 . 1 1 0 ^ { * * * }$ </td><td> $- 0 . 1 1 9 ^ { \ast \ast \ast }$ </td><td>–0.205***</td><td>-0.116***</td></tr><tr><td>Llama</td><td>0.7925 0.7422</td><td> $- 0 . 0 6 5 ^ { * * }$ </td><td> $- 0 . 0 2 7 ^ { n . s }$ </td><td> $- 0 . 0 3 6 ^ { n . s }$ </td><td> $- 0 . 1 2 2 ^ { * * * } \ - 0 . 0 3 1 ^ { n . s }$ </td><td> $- 0 . 1 1 4 ^ { * * * }$ </td></tr><tr><td>GPT-40</td><td>0.7725</td><td> $- 0 . 0 9 8 ^ { \ast \ast }$ </td><td> $- 0 . 0 5 9 ^ { * }$ </td><td> $- 0 . 0 6 8 ^ { \ast \ast }$ </td><td> $- 0 . 1 5 4 ^ { * * * } - 0 . 0 6 4 ^ { * * }$ </td><td></td></tr><tr><td>Gemma</td><td>0.7299</td><td> $- 0 . 1 3 4 ^ { * * }$ </td><td> $- 0 . 0 9 5 ^ { * * * }$ </td><td> $- 0 . 1 0 4 ^ { * * * }$ </td><td> $- 0 . 1 9 0 ^ { * * * } - 0 . 0 9 9 ^ { * * * }$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Overall, these results indicate that LLMs prioritize clinically meaningful CKD risk factors but often differ from the MLPbased ML reference in how they weight those factors. The ML reference places greater emphasis on cohort-specific predictors such as hypertension, age, and urinary RBC, whereas LLMs more strongly emphasize broadly recognized clinical risk factors such as diabetes, anemia, and obesity. This suggests that LLM-based CKD screening is clinically plausible, but its feature-prioritization pattern is not identical to supervised tabular learning and should be interpreted with caution.

## E. Comparison with CKD Screening Tools

Table VI compares zero-shot LLM performance with established CKD screening tools on the full Dataset-1 cohort. The screening tools achieve balanced accuracy between 0.7635 and 0.8103, with Thakkinstian obtaining the highest balanced accuracy among the rule-based tools. Pairwise values report mean Brier loss differences, defined as $\Delta = \mathrm { L o s s } ( \mathrm { L L M } ) \ -$ Loss(Tool); thus, negative values indicate better probabilistic performance for the LLM.

Qwen-3 and Mistral provide the greatest and most consistent probabilistic improvements over the screening tools. Qwen-3 achieves balanced accuracy of 0.7829, matching or exceeding most tools except Thakkinstian [30], while producing significantly lower Brier loss than all five tools, with differences ranging from −0.111 to −0.206 (p < 0.001). Mistral obtains the highest LLM balanced accuracy of 0.7925 and shows a nearly identical pattern, with significant Brier loss reductions across all tool comparisons. These results suggest that the best zero-shot LLMs can provide competitive classification performance while achieving lower Brier loss than established rule-based tools.

![](images/df6fcb43ef70f88f2bd08df38d9e6851280f622a83468f41c837574b27a55af5.jpg)  
Fig. 6: Independent-dataset evaluation on Dataset-2. (a) Balanced accuracy for all versus selected features; significance is assessed using a paired permutation test on per-sample Brier loss. (b) Few-shot performance across shot settings, comparing LLMs with TabPFN as a supervised tabular baseline. n.s means ‘not significant’ $( p \ge 0 . 0 5 )$ .

GPT-4o-mini and Gemma-2 also improve probabilistic performance over most or all tools, although their balanced accuracy is lower than that of Qwen-3 and Mistral. GPT-4o-mini achieves a balanced accuracy of 0.7725 and significantly reduces Brier loss against all tools, with the largest reduction observed against Kwon [13]. Gemma-2 has a lower balanced accuracy of 0.7299, but still yields significant Brier loss reductions across all comparisons, indicating that classification accuracy and probabilistic quality do not always move together. In contrast, Llama-3 shows weaker and less consistent gains: it significantly outperforms SCORED [1] and Kwon [13] in Brier loss, but its differences against Kshirsagar [11], Thakkinstian [30], and Kearns [10] are small and non-significant.

Taken together, these results show that zero-shot LLMs can be competitive with established CKD screening tools in both classification and probabilistic performance. Qwen-3 and Mistral offer the strongest overall trade-off between balanced accuracy and Brier loss, whereas Llama-3 highlights that this advantage is model-dependent rather than universal across LLMs.

## F. Independent Dataset Evaluation

Fig. 6 evaluates model performance on the independent Dataset-2 cohort. Fig. 6a compares all-feature and selectedfeature zero-shot prompting. Feature selection improves or maintains performance for most models, but the effect is model-dependent. Llama-3 shows the largest gain, increasing from 0.73 to 0.88 balanced accuracy $( p < 0 . 0 1 )$ . Mistral and GPT-4o-mini also improve slightly, from 0.86 to 0.89 and 0.86 to 0.88, respectively, although these differences are not statistically significant. Qwen-3 remains stable, with a small decrease from 0.82 to 0.81 $( p < 0 . 0 1 )$ . In contrast, Gemma-2 decreases from 0.90 to 0.82 $( p < 0 . 0 0 1 )$ , indicating sensitivity to feature reduction under distribution shift.

Fig. 6b examines performance across increasing numbers of in-context examples using selected features, with TabPFN included as a supervised tabular baseline. LLM performance is generally strong on Dataset-2, with balanced accuracy mostly between 0.84 and 0.92. Gemma-2 and Qwen-3 improve with additional examples, reaching approximately 0.92 at higher shots. Mistral and GPT-4o-mini also improve gradually and approach 0.91–0.92 at 32 shots. Llama-3 is less stable, declining at higher shots. TabPFN shows the most consistent scaling behavior, increasing from approximately 0.85 with 4 context samples to 0.94 with 32 samples. Because Dataset-2 is hospital-based and does not report CKD stage, these results provide preliminary evidence of performance across datasets and do not validate early-stage community screening.

## G. Limitations

This study has several limitations. First, the primary cohort is relatively small, which may limit the stability of performance estimates. Although repeated runs and an independent-dataset evaluation were used to reduce this concern, larger and more diverse cohorts are needed for stronger generalization claims. Second, LLM reliability depends on prompt design, feature serialization, model version, and availability of token-level log-probabilities; therefore, results may vary across future model releases and deployment settings. Third, the selected feature subset was derived from prior ML-based analysis and may not be optimal for all populations or clinical contexts. Fourth, the independent dataset differs from the primary cohort in sampling setting, feature availability, and case-control composition; therefore, these results provide only preliminary evidence of cross-dataset performance rather than definitive clinical validation. Finally, we did not directly evaluate robustness to patient-level missing features, heterogeneous inputs, deployment efficiency, clinical safety, or cost; these factors, together with prospective validation, require evaluation before clinical use.

## IV. CONCLUSION

This paper presented LLM4CKD, a comprehensive applied evaluation of large language models for low-resource earlystage CKD screening under zero-shot and few-shot in-context learning. We benchmarked five LLMs against conventional ML, tabular DL, tabular foundation-model baselines, established CKD screening tools, and an independent dataset. The results show that LLMs can provide competitive CKD screening with minimal labeled data, particularly when clinically compact selected-feature prompts are used. Among the evaluated models, Qwen-3-8B, Mistral, and GPT-4o-mini achieved strong balanced accuracy and probabilistic performance, and the strongest zero-shot LLMs improved Brier loss relative to established rule-based screening tools. At the same time, additional in-context examples did not consistently improve performance, indicating that LLM-based screening remains sensitive to model choice, prompt design, and example composition. Supervised ML and tabular models improved more steadily as labeled training data increased, highlighting their continued advantage when sufficient training data and training infrastructure are available.

The feature-ranking analysis showed that LLMs emphasize clinically meaningful CKD risk factors, but their prioritization patterns differ from the MLP-based ML reference, indicating distinct feature prioritization between the two approaches. Evaluation on an independent dataset provided preliminary evidence of performance across a different cohort and data setting, although broader external validation is required. Overall, LLM4CKD shows that LLMs may support low-data CKD screening and probabilistic prediction, while supervised tabular models remain important when local labeled data are available. Future work should evaluate robustness to missing and heterogeneous inputs, computational efficiency, and prospective performance across broader clinical settings.

## REFERENCES

[1] Heejung Bang, Suma Vupputuri, David A Shoham, Philip J Klemmer, Ronald J Falk, Madhu Mazumdar, Debbie Gipson, Romulo E Colindres, and Abhijit V Kshirsagar. Screening for occult renal disease (scored): a simple prediction model for chronic kidney disease. Archives of internal medicine, 167(4):374–381, 2007.

[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33, 2020.

[3] Anna Francis, Meera N Harhay, Albert CM Ong, Sri Lekha Tummalapalli, Alberto Ortiz, Agnes B Fogo, Danilo Fliser, Prabir Roy-Chaudhury, Monica Fontana, Masaomi Nangaku, et al. Chronic kidney disease and the global public health agenda: an international consensus. Nature Reviews Nephrology, 20(7):473–485, 2024.

[4] Cindy George, Amelie Mogueo, Ikechi Okpechi, Justin B Echouffo-Tcheugui, and Andre Pascal Kengne. Chronic kidney disease in lowincome to middle-income countries: the case for increased screening. BMJ global health, 2(2), 2017.

[5] Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. Tabllm: Few-shot classification of tabular data with large language models. In International conference on artificial intelligence and statistics, pages 5549–5581. PMLR, 2023.

[6] Yongzheng Hu, Jianping Liu, and Wei Jiang. Large language models in nephrology: applications and challenges in chronic kidney disease management. Renal failure, 47(1), 2025.

[7] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

[8] Muhammad Ashad Kabir and Sirajam Munira. From many to meaningful: Feature-guided zero-shot chronic kidney disease screening using large language models. In International Conference on Artificial Intelligence in Medicine, pages 38–48. Springer, 2026.

[9] Muhammad Ashad Kabir, Sirajam Munira, Dewan Tasnia Azad, Saleh Mohammed Ikram, Mohammad Habibur Rahman Sarker, and Syed Manzoor Ahmed Hanifi. Community-based early-stage chronic kidney disease screening using explainable machine learning for lowresource settings. International Journal of Medical Informatics, 214:106435, 2026.

[10] Benjamin Kearns, Hugh Gallagher, and Simon de Lusignan. Predicting the prevalence of chronic kidney disease in the english population: a cross-sectional study. BMC nephrology, 14(1):49, 2013.

[11] Abhijit V Kshirsagar, Heejung Bang, Andrew S Bomback, Suma Vupputuri, David A Shoham, Lisa M Kern, Philip J Klemmer, Madhu Mazumdar, and Phyllis A August. A simple algorithm to predict incident kidney disease. Archives ofinternal medicine, 168(22):2466–2473, 2008.

[12] Aditya Kumar, Dilpreet Singh, Mario Cypko, and Oliver Amft. A multi-view validation framework for llm-generated knowledge graphs of chronic kidney disease. International Journal of Computer Assisted Radiology and Surgery, 20(12):2523–2528, 2025.

[13] Keun-Sang Kwon, Heejung Bang, Andrew S Bomback, Dai-ha Koh, Jung-Ho Yum, Ju-Hyung Lee, Sik Lee, Sung K Park, Keun-Young Yoo, Sue K Park, et al. A simple prediction score for kidney disease in the korean population. Nephrology, 17(3):278–284, 2012.

[14] Roderick JA Little and Donald B Rubin. Statistical analysis with missing data. John Wiley & Sons, 2019.

[15] Meta AI. Introducing meta llama 3: The most capable openly available llm to date, 2024. Technical report.

[16] Anand Kunal Mishra and S Prabakeran. Kidney disease prediction using gen-ai approach. In Challenges in Information, Communication and Computing Technology, pages 857–862. CRC Press, 2024.

[17] Ruya Naz, Okan Akacı, Hakan Erdo ¨ gan, and Ayfer Ac¸ıkg ˘ oz. Can large¨ language models provide accurate and quality information to parents regarding chronic kidney diseases? Journal of Evaluation in Clinical Practice, 30(8):1556–1564, 2024.

[18] NKF Patient Education Team. Chronic kidney disease (ckd)—symptoms, causes, treatment—national kidney foundation. https://www.kidney.org/kidney-topics/chronic-kidney-disease-ckd,

[19] Elvira Nurfadhilah, Agung Santosa, and Prabu Kresna Putra. Retrievalaugmented large language models for a chronic kidney disease patient education chatbot. In 2025 International Conference on Computer, Control, Informatics and its Applications (IC3INA). IEEE, 2025.

[20] OpenAI. Gpt-4o mini, 2024. Accessed: 2026-04-16.

[21] Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

[22] L. Rubini, P. Soundarapandian, and P. Eswaran. Chronic Kidney Disease. UCI Machine Learning Repository, 2015. DOI: https://doi.org/10.24432/C5G020.

[23] Charumathi Sabanayagam, Riswana Banu, Cynthia Lim, Yih Chung Tham, Ching-Yu Cheng, Gavin Tan, Elif Ekinci, Bin Sheng, Gareth McKay, Jonathan E Shaw, et al. Artificial intelligence in chronic kidney disease management: a scoping review. Theranostics, 15(10):4566, 2025.

[24] Mohammad Habibur Rahman Sarker, Michiko Moriyama, Harun Ur Rashid, Mohammod Jobayer Chisti, Md Moshiur Rahman, Sumon Kumar Das, Aftab Uddin, Samir Kumar Saha, Shams El Arifeen, Tahmeed Ahmed, et al. Community-based screening to determine the prevalence, health and nutritional status of patients with ckd in rural and peri-urban bangladesh. Therapeutic advances in chronic disease, 12:20406223211035281, 2021.

[25] John W Stanifer, Anthony Muiru, Tazeen H Jafar, and Uptal D Patel. Chronic kidney disease in low-and middle-income countries. Nephrology Dialysis Transplantation, 31(6):868–874, 2016.

[26] Paul E Stevens, Sofia B Ahmed, Juan Jesus Carrero, Bethany Foster, Anna Francis, Rasheeda K Hall, Will G Herrington, Guy Hill, Lesley A Inker, Rumeyza Kazancıo¨ glu, et al. Kdigo 2024 clinical practice˘ guideline for the evaluation and management of chronic kidney disease. Kidney international, 105(4):S117–S314, 2024.

[27] Paul E Stevens, Adeera Levin, and Kidney Disease: Improving Global Outcomes Chronic Kidney Disease Guideline Development Work Group Members\*. Evaluation and management of chronic kidney disease: synopsis of the kidney disease: improving global outcomes 2012 clinical practice guideline. Annals of internal medicine, 158(11):825–830, 2013.

[28] Susanne Stolpe, Bernd Kowall, Denise Zwanziger, Mirjam Frank, Karl-Heinz Joeckel, Raimund Erbel, and Andreas Stang. External validation of six clinical models for prediction of chronic kidney disease in a german population. BMC nephrology, 23(1):272, 2022.

[29] Masooma Zehra Syeda, Syed Usama Khalid Bukhari, Maqbool Hussain, Wajahat Ali Khan, and Syed Sajid Hussain Shah. Llm-based kidney disease diagnostic framework for pathologists. In 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pages 1–4. IEEE, 2024.

[30] Ammarin Thakkinstian, Atiporn Ingsathit, Amnart Chaiprasert, Sasivimol Rattanasiri, Pornpen Sangthawan, Pongsathorn Gojaseni, Kriwiporn

Kiattisunthorn, Leena Ongaiyooth, and Prapaipim Thirakhupt. A simplified clinical prediction score of chronic kidney disease: a cross-sectionalsurvey study. BMC nephrology, 12(1):45, 2011.

[31] Michael Toal, Christopher Hill, Michael Quinn, Ciaran O’Neill, and Alexander P Maxwell. Large language models’ clinical decision-making on when to perform a kidney biopsy: comparative study. Journal of Medical Internet Research, 27:e73603, 2025.

[32] Zoe Unger, Shelly Soffer, Orly Efros, Lili Chan, Eyal Klang, and Girish N Nadkarni. Clinical applications and limitations of large language models in nephrology: a systematic review. Clinical kidney journal, 18(9):sfaf243, 2025.

[33] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.