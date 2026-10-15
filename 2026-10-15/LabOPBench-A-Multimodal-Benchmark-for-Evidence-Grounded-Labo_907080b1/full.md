# LabOPBench: A Multimodal Benchmark for Evidence-Grounded Laboratory Decision Making

Jingyuan Li<sup>1,2</sup>, Yujie Liu<sup>1,3</sup>, Yijie Dai<sup>2</sup>, Xiaoyu Yuan<sup>4</sup>, Nian Ran<sup>5</sup>, Huajie Niu<sup>6</sup>, Ming Zhu<sup>5</sup>, Shuwen Zhao<sup>5</sup>, Chenxi Li<sup>1,7</sup>, Fenghao Lu<sup>8</sup>, Jiahao Li<sup>8</sup>, Xinyu Gu<sup>1</sup>, Zhe Liu<sup>9</sup>, Long Xu<sup>2</sup>, Wenlong Zhang<sup>1</sup>, Bo Zhang<sup>1</sup>, Lei Bai<sup>1</sup>, Peiyu Zhang<sup>4†</sup>, Minting Pan<sup>1†</sup>, Weinan Zhang<sup>1,2†</sup>, Dongzhan Zhou<sup>1†</sup>

<sup>1</sup> Shanghai Artificial Intelligence Laboratory <sup>2</sup> Shanghai Jiao Tong University <sup>3</sup> Fudan University <sup>4</sup> XtalPi <sup>5</sup> Shanghai Institute of Ceramics, Chinese Academy of Sciences <sup>6</sup> Jiaxing University <sup>7</sup> Zhejiang University <sup>8</sup> University of Chinese Academy of Sciences <sup>9</sup> East China University of Science and Technology

johnnylee@sjtu.edu.cn, peiyu.zhang@xtalpi.com {panminting,zhangweinan,zhoudongzhan}@pjlab.org.cn

## Abstract

Laboratory experiments involve more than following predefined protocols; they require adaptive decision-making as the experimental state changes. Existing scientific benchmarks rarely assess such evidence-based decisions in laboratory settings. Here we present LabOPBench, a multimodal benchmark of 756 expert-authored, independently reviewed questions in organic synthesis and materials science. Each item pairs laboratory evidence with procedural context and an open-ended judgment, evaluated using a question-specific scoring rubric. Eight multimodal models obtain overall normalized scores of 28.8-45.8%, leaving substantial room for improvement. On a controlled 100-question subset, the strongest models outperform undergraduate participants on average but remain below the average performance of PhD participants. Expert analysis of responses from the three leading models identifies Evidence Misreading and Evidence-Decision Conflict as the dominant error patterns. These results reveal that current models still struggle to turn multimodal perception into reliable laboratory decision-making.

## 1 Introduction

Large language models (LLMs) and LLM-based agents increasingly support laboratory research through experimental planning, protocol generation, and integration with laboratory instruments and robotic systems (Boiko et al., 2023; Darvish et al., 2025; Häse et al., 2019; Tom et al., 2024; Coley et al., 2019; Burger et al., 2020). Yet laboratory research involves more than executing predefined procedures. As experiments progress, unexpected observations and changes in sample states may require researchers to verify, adjust, or terminate an operation. Evaluating multimodal models for laboratory use requires assessing their ability both to generate scientifically plausible procedures and to make appropriate judgments as experiments evolve.

Laboratory observations cannot be interpreted in isolation. The same visual pattern may indicate different causes and require different actions depending on the experimental context. Although previous studies have highlighted challenges in visual grounding and multimodal chemical reasoning (Tong et al., 2024; Cui et al., 2025), recognizing an observation is only the first step. A model should also determine whether its explanation or proposed action is appropriate for the specific experiment. If the evidence is insufficient, further checks may be needed before drawing conclusions or taking action. Evidence-to-decision consistency describes how well laboratory judgments align with experimental evidence and context.

Existing scientific benchmarks evaluate a broad range of capabilities, including domain knowledge, multimodal reasoning, laboratory safety, and experimental procedures (Laurent et al., 2024; Alampara et al., 2025; Zhou et al., 2026; Zhang et al., 2026). However, these evaluations provide less insight into whether models can use experimental observations to make appropriate laboratory judgments. In practice, recognizing an observation is only part of the task. Models must also consider the experimental context to determine what the observation means and how to respond. This motivates evaluating whether a model can make laboratory judgments supported by the available evidence and experimental context.

We introduce LabOPBench, a multimodal benchmark of 756 expert-authored and independently reviewed questions in organic synthesis and materials science. Constructing such a benchmark requires substantial scientific judgment, as real laboratory materials do not readily translate into welldefined, objectively assessable decision tasks. To address this challenge, 22 contributors with handson laboratory experience developed open-ended questions from real laboratory materials. For each question, they identified decision-relevant evidence, provided the necessary experimental context, and defined judgments supported by the available information. They also considered alternative explanations or actions, ensuring that each question could be answered without relying on missing information or unresolved ambiguity. Experts then developed question-specific scoring rubrics, while two additional reviewers independently assessed each item for scientific validity, evidence sufficiency, and scoring clarity.

Evaluation of eight multimodal models on LabOPBench reveals substantial limitations in evidence-grounded laboratory decision-making. Overall normalized scores range from 28.8% to 45.8%, indicating considerable room for improvement. Performance varies across task categories, with postprocessing showing the lowest crossmodel mean (34.9%), compared with materials science (52.2%) and monitoring (50.8%). Postprocessing tasks present particular challenges, requiring integration of preceding experimental operations and current sample states. Expert analysis further identifies recurring failures to interpret decision-relevant observations or align explanations and actions with the experimental context. Together, these findings suggest that current models still face challenges in interpreting laboratory evidence and using it to make reliable decisions in specific experimental situations.

Overall, the contributions of this paper are:

• We introduce LabOPBench, a multimodal benchmark of 756 expert-authored questions grounded in real laboratory materials from organic synthesis and materials science. It evaluates how models interpret experimental states, identify possible causes and risks, and determine appropriate operations.

• We establish expert-defined, question-specific scoring rubrics to assess whether laboratory judgments are supported by the available evidence and experimental context. These rubrics assess whether answers are supported by the evidence and fit the specific experimental context, rather than merely being scientifically plausible.

• We systematically evaluate eight multimodal models to characterize their performance and limitations in laboratory decision-making. Our findings show that models often misinterpret experimental evidence or make judgments that do not fit the laboratory context.

## 2 Related Work

Scientific knowledge benchmarks. Existing chemistry benchmarks evaluate molecular and reaction reasoning, factual chemical knowledge, quantitative problem solving, and expert-level question answering (Guo et al., 2023; Lu et al., 2024; Huang et al., 2026; Mirza et al., 2025; Runcie et al., 2026; Xie et al., 2025). Recent multimodal benchmarks further incorporate molecular structures, reaction schemes, chemical tables, characterization images, and scientific documents (Li et al., 2025b; Zhou et al., 2025; Lai et al., 2025; Li et al., 2025a; Alampara et al., 2025). More general multimodal evaluations, such as MMMU-Pro, have also emphasized the importance of genuine visual understanding and cross-modal reasoning (Yue et al., 2025). Together, these resources provide important assessments of scientific knowledge, visual interpretation, and multimodal reasoning.

Laboratory procedural benchmarks. Recent benchmarks have increasingly addressed practical scientific research tasks. LAB-Bench evaluates biological research capabilities, including protocol troubleshooting, figure interpretation, and experimental planning (Laurent et al., 2024). MaCBench examines multimodal chemistry and materials tasks spanning data extraction, experimental execution, and results interpretation (Alampara et al., 2025), while MatCha focuses on expert-level understanding of materials characterization images (Lai et al., 2025). LabSafety Bench evaluates laboratory hazard identification, risk assessment, and consequence prediction using multiple-choice questions and open-ended scenarios (Zhou et al., 2026). ChemReason-Bench (Zhang et al., 2026) evaluates verifiable procedural reasoning in organic synthesis through ordering, validation, constrained completion, selection, and explanation tasks. While these benchmarks assess practical laboratory skills and procedural consistency, LabOPBench focuses on open-ended laboratory judgments grounded in multimodal evidence and experimental context. Table 1 summarizes the key differences.

Evidence-Grounded Laboratory Reasoning. Laboratory perception studies have investigated the recognition of vessels, material phases, liquid levels, phase boundaries, turbidity, precipitates, and color changes in experimental settings (Eppel et al., 2020; El-Khawaldeh et al., 2024a,b). Related research has also explored multimodal analysis of laboratory anomalies and experimental states (Lin et al., 2025a,b). These studies demonstrate the importance of extracting and interpreting visual evidence from laboratory environments. However, identifying an experimental phenomenon or abnormal state does not guarantee an appropriate explanation or action. Such judgments must also be consistent with the available evidence and previous experimental steps. LabOPBench focuses on the evidence-to-decision relationship by evaluating open-ended judgments about experimental states, possible causes, and appropriate actions. Through expert-defined reference judgments and questionspecific scoring rubrics, it assesses whether models connect multimodal observations and procedural context to experimentally appropriate conclusions.

## 3 LabOPBench

## 3.1 Task Definition

LabOPBench evaluates whether a multimodal model can transform evidence from a concrete laboratory situation into an experimentally appropriate judgment. Each benchmark item contains an experimental context, one or more evidence sources, and an open-ended question. The evidence may include laboratory photographs, instrument interfaces, spectra, chromatograms, microscopy images, sample states, or experimental records. Given these inputs, a model is expected to identify the relevant experimental state, explain a likely cause or risk when required, and select an appropriate next action. Formally, each item can be represented as

$$
q _ { i } = ( c _ { i } , e _ { i } , p _ { i } , r _ { i } ) ,
$$

where $c _ { i }$ denotes the experimental context, $e _ { i }$ the associated multimodal evidence, $p _ { i }$ the question prompt, and $r _ { i }$ the question-specific scoring rubric. The answer model observes only $( c _ { i } , e _ { i } , p _ { i } )$ . The target judgment may concern the current state of an experiment, a likely cause of an observation, a potential risk, or an appropriate experimental operation. The task therefore requires evidence selection and contextual interpretation rather than isolated object recognition or factual question answering.

## 3.2 Benchmark Composition

LabOPBench contains 756 questions, including 664 questions from organic synthesis and 92 questions from materials science (Fig. 1). The organicsynthesis questions are organized according to the stage of the experimental workflow in which the target judgment arises: preparation (67), monitoring (30), postprocessing (425), and characterization (142). Materials-science questions are retained as a separate major category because their experimental objects, procedures, and measurements do not map directly onto the organic-synthesis taxonomy. The category sizes reflect the assessable situations available in the collected laboratory materials rather than predefined quotas. Each judgment must be grounded in the supplied evidence, experimental context, and scientific knowledge.

## 3.3 Question Authoring and Expert Review

LabOPBench questions were authored by 22 contributors with hands-on experience relevant to the represented laboratory tasks. The contributor pool comprised four PhD researchers, six postdoctoral researchers, eight laboratory specialists, and four researchers at associate-researcher level or above. Twenty contributors had at least 5 years of laboratory experience. Their expertise spanned laboratory workflows, from experimental setup to characterization and troubleshooting. Further details are provided in Appendix B.

Each item was designed from real laboratory materials to assess a well-defined experimental judgment. We view expert authoring as a transformation of original laboratory material into a structured decision instance:

$$
\Phi _ { \mathrm { e x p e r t } } ( M _ { i } ) \longrightarrow B _ { i } = ( C _ { i } , E _ { i } , Q _ { i } , Y _ { i } , \Pi _ { i } , \mathcal { R } _ { i } ) ,
$$

where $M _ { i }$ denotes the original laboratory material; $C _ { i } , E _ { i } , Q _ { i }$ , and $Y _ { i }$ denote the experimental context, supplied evidence, question prompt, and target experimental judgment, respectively. Π<sub>i</sub> denotes the expert-specified evidence-to-decision path, and $\mathcal { R } _ { i }$ denotes the associated scoring rubric. Depending on the item, Π<sub>i</sub> connects evidence to an experimental state or an appropriate action.

<table><tr><td>Benchmark</td><td>Data source</td><td>Format</td><td>Focus</td></tr><tr><td>LAB-Bench (Laurent et al., 2024)</td><td>Literature / protocols</td><td>MCQ</td><td>Biological research</td></tr><tr><td>MMMU-Pro (Yue et al., 2025)</td><td>Academic exams</td><td>MCQ</td><td>Visual reasoning</td></tr><tr><td>MatCha (Lai et al., 2025)</td><td>Published figures</td><td>MCQ</td><td>Materials characterization</td></tr><tr><td>MaCBench (Alampara et al., 2025)</td><td>Figures / lab photos</td><td>MCQ + Numeric</td><td>Data &amp; experiment analysis</td></tr><tr><td>LabSafety Bench (Zhou et al., 2026)</td><td>Safety resources</td><td>MCQ + Open</td><td>Hazards &amp; consequences</td></tr><tr><td>ChemReason-Bench (Zhang et al., 2026)</td><td>Synthesis protocols</td><td>Mixed</td><td>Procedural reasoning</td></tr><tr><td>LabOPBench</td><td>Original lab records</td><td>Open</td><td>Evidence-based decisions</td></tr></table>

Table 1: Comparison of LabOPBench with related benchmarks by data source, format, and focus.

![](images/d883559649a0544c2496685cc743d6a7125019f87a70dd06b1cc70cec156d3f7.jpg)  
Figure 1: Composition of LabOPBench across laboratory task categories. a, Distribution of the 756 benchmark questions across the five major categories. b-e, Distribution of organic-synthesis questions across the subcategories of preparation, postprocessing, monitoring and characterization, respectively.

The scientific content of every item was authored and validated by human experts. Language models were not used to generate the experimental scenario, target judgment, evidence selection, reference answer, acceptable alternatives, or scoring criteria. For each item, contributors recorded the decision-relevant evidence, acceptable alternative explanations or actions, critical errors, and four scoring anchors corresponding to scores of 3, 2, 1, and 0. Questions were required to be answerable from the supplied evidence and relevant scientific knowledge without relying on undisclosed experimental conditions or author-specific information.

DeepSeek-V4-Pro was used only for post hoc linguistic standardization of question prompts and reference answers after the scientific content had been fixed. It was not permitted to change the target judgment, decision-relevant evidence, acceptable answers, scoring anchors, or safety requirements. The original author verified each standardized version before two additional experts independently reviewed the credibility and sufficiency of the evidence, prompt-evidence consistency, scientific correctness, answer determinacy, and scoreability. Substantive disagreements were adjudicated by a third expert, whereas questions with insufficient evidence or unresolved ambiguity were revised or excluded. The question-construction and expert-validation workflow is summarized in Fig. 2. Further details are provided in Appendix A.1.

## 3.4 Evaluation Protocol

We evaluated eight multimodal models on all 756 questions under a unified input and generation protocol. Each model received the question, its minimal experimental context, and the original visual evidence, without access to reference answers, scoring rubrics, or external tools.

![](images/46bc3cc7331fc2615088039f203c8fb36ac9b1c3edc89e3f8306721f2240bcd5.jpg)  
Figure 2: Workflow for constructing and validating a LabOPBench question. Experts draft and review each evidence-grounded item, while an LLM is used only for language standardization. Disagreements are adjudicated before the final question, multimodal evidence, reference options, and scoring anchors are compiled.

Responses were evaluated with question-specific scoring rubrics on a 0-3 scale, with half-point scores permitted. All eight evaluated models produced valid responses for all 756 benchmark questions. GPT-5.5 and Gemini 3.1 Pro each assigned a valid score to every response from every evaluated model; therefore, N = 756 for each modeljudge pair when reporting the full benchmark. The Overall score in Table 2 is calculated across all 756 benchmark questions. A higher score required the response to satisfy the scenario-specific evidence, interpretation, and action requirements of the corresponding rubric level; general plausibility or keyword overlap was insufficient. Holistic scoring was used by default, while explicitly independent subquestions were scored separately and averaged. GPT-5.5 and Gemini 3.1 Pro independently scored each response using the question, candidate response, and reference answer. The judges were blinded to the evaluated model and to each other’s assessments. For a reporting set containing $N _ { j }$ valid responses, the normalized score assigned by judge j was

$$
\mathrm { N S } _ { j } = \frac { 1 } { 3 N _ { j } } \sum _ { i = 1 } ^ { N _ { j } } { s _ { i j } } \times 1 0 0 \% ,
$$

where $s _ { i j }$ is the score assigned to question i. The final model score was the arithmetic mean of the two judge-specific scores. Detailed scoring and aggregation procedures are provided in Appendix D.

We validated automated scoring using a stratified sample of approximately 20% of Claude Opus 4.7 responses from each major category. Two experts independently rescored the sampled responses with the same rubrics, without access to automated scores or each other’s ratings, and agreement was assessed at the category-mean level.

We further used a controlled 100-question organic-synthesis subset for comparison with five undergraduate and five PhD participants, an imagemismatch condition for testing image-question alignment, and expert annotation of errors in responses from the three highest-scoring models. These analyses used the same scoring protocol; detailed procedures are provided in Appendix G.

## 4 Experiments

We evaluated eight multimodal models: Claude Opus 4.7 (Anthropic, 2026), GPT-5.5 (OpenAI, 2026), Gemini 3.1 Pro (Google DeepMind, 2026b), Gemini 3.1 Flash-Lite (Google DeepMind, 2026a), Qwen3.6-Plus (Qwen Team, 2026), MiniMax-M3 (MiniMax, 2026), Grok 4.3 (xAI, 2026), and Intern-S1-Pro (Zou et al., 2026). All eight models were evaluated on the full set of 756 LabOPBench questions using a unified protocol. Each model received the question, experimental context, and multimodal evidence, without access to reference answers, scoring rubrics, or external tools. GPT-5.5 and Gemini 3.1 Pro independently scored the openended responses using expert-defined, questionspecific rubrics. The two normalized judge scores were then averaged. We further compared model and human performance, examined the effects of image mismatch, and analyzed model error patterns. Detailed settings and procedures are provided in Appendices C-I.

## 4.1 Main Results

We first evaluate the eight multimodal models on the complete LabOPBench. Evaluation settings and scoring follow the protocol described in Section 3.4. As shown in Table 2, normalized scores range from 28.8% to 45.8%. Claude Opus 4.7 achieves the highest overall score (45.8%), followed by GPT-5.5 (43.6%) and Gemini 3.1 Pro (41.9%).

Performance varied substantially across the five major categories. The highest cross-model mean scores were obtained in materials science (52.2%) and monitoring (50.8%), followed by characterization (41.7%) and preparation (40.2%). Postprocessing was the most challenging category, with a cross-model mean of 34.9%. Notably, postprocessing was the lowest-scoring category for seven of the eight models, suggesting that its difficulty was not driven by a single model. Even Claude Opus 4.7 achieved only 40.3% in this category. These results suggest greater difficulty with the experimental judgments represented in postprocessing tasks, which often involve interpreting observations in relation to preceding operations and sample states. Claude Opus 4.7 achieved the highest score in four of the five categories, while Gemini 3.1 Pro achieved the highest score in characterization. Figure 3 provides a finer-grained view of performance across the 16 organic-synthesis subcategories.

We also compared automated scores with expert assessments on Claude Opus 4.7 responses, as described in Appendix E. Automated and expert scores showed close agreement at the category level, with differences in mean scores across the five major categories ranging from −0.59 to +1.39 percentage points (Fig. 4).

## 4.2 Comparison with Human Participants

We compared models with human participants on the controlled 100-question organic-synthesis subset (Section 3.4). Undergraduates averaged 31.0%, compared with 45.9% for PhD participants (Fig. 5). Claude Opus 4.7 and GPT-5.5 were the top-performing models, scoring 43.7% and 43.5%.

Seven of the eight models exceeded the undergraduate-group mean, while all eight remained below the PhD-group mean. Notably, Claude Opus 4.7 was only 2.2 percentage points below the PhD average. These results show how model performance compares with that of undergraduate and PhD participants.

## 4.3 Analysis of Visual Evidence

To examine the role of question-specific visual evidence in laboratory judgments, we compared model performance under matched- and mismatched-image conditions. The experiment was conducted on the same controlled 100-question organic-synthesis subset used for human comparison. The image-mismatch procedure preserved the question format and task category while disrupting the correspondence between the images and the experimental situation (Fig. 6).

Image mismatch reduced performance for all eight models. The cross-model mean decreased from 37.5% with matched images to 19.3% with mismatched images, corresponding to an 18.2- percentage-point drop. The largest decreases were observed for GPT-5.5, from 43.5% to 19.5% (24.0 percentage points), Gemini 3.1 Pro, from 39.0% to 17.2% (21.8 percentage points), and Claude Opus 4.7, from 43.7% to 22.7% (21.0 percentage points).

The consistent performance decline across all models indicates that their laboratory judgments were sensitive to the correspondence between visual evidence and the experimental context. However, the magnitude of this sensitivity varied across models, with score reductions ranging from 12.9 to 24.0 percentage points. Notably, several of the strongest models under matched-image conditions also experienced substantial decreases when this correspondence was disrupted. These findings highlight the importance of question-specific visual evidence in experimental judgment.

## 4.4 Analysis of Error Patterns

Finally, we analyzed non-full-score responses from the three top-performing models based on unrounded matched-image scores on the controlled organic-synthesis subset: Claude Opus 4.7, GPT-5.5, and Gemini 3.1 Pro. The taxonomy in Appendix I defines six error types: Evidence Misreading, Evidence-Decision Conflict, Chemical Signal Errors, Common Rule Misuse, Mistaking Normal for Abnormal, and Experimental Method Errors.

Errors were concentrated in Evidence Misreading (40.0%) and Evidence-Decision Conflict (35.8%). Evidence Misreading occurred when models failed to identify, correctly read, or use decision-relevant experimental cues, whereas Evidence-Decision Conflict reflected explanations or recommendations inconsistent with the observed state or underlying cause. These two categories accounted for 75.8% of the error distribution, highlighting two related challenges: identifying the evidence that constrains an experimental judgment and producing conclusions consistent with the specific experimental situation. Common Rule Misuse accounted for a further 7.3%, reflecting the use of common laboratory rules in place of casespecific evidence and interpretation. This suggests that scientifically plausible responses may still lead to incorrect judgments when they overlook case-specific experimental evidence. These expertidentified patterns characterize errors in model outputs rather than directly establishing the internal mechanisms responsible for them.

<table><tr><td>Model</td><td>Preparation</td><td>Monitoring</td><td>Postprocessing</td><td>Characterization</td><td>Materials Science</td><td>Overall</td></tr><tr><td>Intern-S1-Pro</td><td>24.2</td><td>40.0</td><td>25.6</td><td>29.6</td><td>41.7</td><td>28.8</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>41.1</td><td>41.4</td><td>33.1</td><td>37.8</td><td>47.6</td><td>36.8</td></tr><tr><td>Grok 4.3</td><td>35.8</td><td>49.4</td><td>33.5</td><td>37.4</td><td>53.7</td><td>37.5</td></tr><tr><td>MiniMax-M3</td><td>47.3</td><td>54.8</td><td>33.3</td><td>41.2</td><td>55.6</td><td>39.6</td></tr><tr><td>Qwen3.6-Plus</td><td>37.3</td><td>54.8</td><td>37.2</td><td>44.8</td><td>52.8</td><td>41.2</td></tr><tr><td>Gemini 3.1 Pro</td><td>38.0</td><td>48.1</td><td>37.5</td><td>48.0</td><td>53.5</td><td>41.9</td></tr><tr><td>GPT-5.5</td><td>49.0</td><td>52.5</td><td>39.0</td><td>47.1</td><td>52.9</td><td>43.6</td></tr><tr><td>Claude Opus 4.7</td><td>49.1</td><td>65.0</td><td>40.3</td><td>47.4</td><td>59.8</td><td>45.8</td></tr></table>

Table 2: Normalized scores (%) of eight multimodal models across the major LabOPBench categories. Bold and underlined values denote the highest and second-highest scores in each column, respectively.

![](images/81ac6f1d93da5887efd1d4d0557018a9e518f76e33a83b377e7c5764858efd66.jpg)  
Figure 3: Fine-grained performance of eight multimodal models across 16 organic-synthesis subcategories. Cells show the mean normalized scores (%) from two independent judges, with darker colors indicating higher scores. Bold values indicate the highest score in each subcategory, including ties.

Table 3 illustrates two types of failures. In the TLC case, the model misclassified standard spotting as non-standard and attributed the pattern to sample overloading rather than unsuitable solvent polarity, despite insufficient supporting evidence. In the LC-MS case, the model recommended centrifugation or filtration followed by supernatant analysis, excluding the solid fraction required for representative sampling before dissolution. These cases show how plausible explanations or procedures can lead to inappropriate judgments when inconsistent with experimental evidence or objectives. Further details are provided in Appendix I.

## 5 Conclusion

We introduced LabOPBench, a 756-question benchmark for evidence-grounded laboratory decisionmaking, and evaluated eight current multimodal models. Overall normalized scores ranged from 28.8% to 45.8%. On the controlled 100-question subset, the strongest models exceeded the undergraduate mean (31.0%) but remained below the PhD mean (45.9%). Image mismatch reduced the cross-model mean from 37.5% to 19.3%, indicating sensitivity to image-question alignment.

<table><tr><td>Case</td><td>Expert judgment</td><td>Model response</td><td>Failure mechanism</td></tr><tr><td>TLC</td><td>Spotting was standard; the developed pattern indicated unsuitable developing-solvent polarity.</td><td>Judged spotting as non-standard and attributed the pattern to excessive sample loading.</td><td>Evidence Misreading. Misinterpreted spotting evidence and selected an unsupported cause.</td></tr><tr><td>LC-MS sampling</td><td>Collect a representative solid-liquid aliquot while stirring, then dissolve in methanol.</td><td>Proposed centrifugation or filtration, then analysis of only the clear supernatant.</td><td>Evidence-Decision Conflict. Excluded the solid fraction required for representative sampling</td></tr></table>

Table 3: Representative failures in model answers. The cases illustrate misinterpretation of experimental evidence and an operation inconsistent with the sampling objective.

![](images/cbc69ef0b867872b8cf3d80f99fd09d32ee30c0f6a6971d144ec5f9c82d756df.jpg)  
Figure 4: Differences between automated and expert scoring across major laboratory categories. Bars show differences in category-level mean normalized scores (automated minus expert) for a stratified sample of Claude Opus 4.7 responses. Positive values indicate higher automated scores, whereas negative values indicate higher expert scores.

These results highlight a persistent gap between multimodal perception and experimental decisionmaking. Models often fail to correctly identify decision-relevant observations or reliably connect them to experimental states, likely causes, and appropriate actions. Evidence Misreading (40.0%) and Evidence-Decision Conflict (35.8%) were the dominant error types, together accounting for 75.8% of the error distribution.

Future laboratory systems will require representations linking observations to procedural context, recent operations, and expected state transitions. They should also assess alternative interventions and determine when to act, take further measurements, or seek human review. LabOPBench provides a basis for evaluating these capabilities in increasingly adaptive laboratory workflows.

![](images/72cf78f8ec70c4cb043ff09c18bdffc585200d91dd19286ad23811d5afba7bcd.jpg)  
Figure 5: Performance on the controlled 100-question organic-synthesis subset. Bars show normalized scores for eight multimodal models and mean normalized scores for undergraduate and PhD participants.

## 6 Limitations

LabOPBench evaluates evidence-grounded laboratory judgment at a single decision point, given a fixed experimental context and multimodal evidence. It does not cover continuous experimental trajectories, post-action observations, robotic execution, or replanning after failed interventions. Accordingly, the findings should be interpreted as evidence of models’ performance in static laboratory decision-making rather than as a complete evaluation of closed-loop experimental autonomy.

## References

Nawaf Alampara, Mara Schilling-Wilhelmi, Martiño Ríos-García, Indrajeet Mandal, Pranav Khetarpal, Hargun Singh Grover, N. M. Anoop Krishnan, and Kevin Maik Jablonka. 2025. Probing the limitations of multimodal language models for chemistry and

![](images/293bc7373985fab59e47f99e4d9ae286fa2a4d578b78e4c8b0cfc82a24d84f50.jpg)  
Figure 6: Image-mismatch ablation on the 100-question organic-synthesis subset. Connected points show normalized scores with matched and mismatched images. Values in parentheses indicate the score difference (matched minus mismatched) in percentage points.

materials research. Nature Computational Science, 5:952–961.

Anthropic. 2026. Claude Opus 4.7 System Card. System card, Anthropic.

Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. 2023. Autonomous chemical research with large language models. Nature, 624(7992):570– 578.

Benjamin Burger, Phillip M. Maffettone, Vladimir V. Gusev, Catherine M. Aitchison, Yang Bai, Xiaoyan Wang, Xiaobo Li, Ben M. Alston, Buyi Li, Rob Clowes, Nicola Rankin, Brandon Harris, Reiner Sebastian Sprick, and Andrew I. Cooper. 2020. A mobile robotic chemist. Nature, 583(7815):237–241.

Connor W. Coley, Dale A. Thomas, Justin A. M. Lummiss, Jonathan N. Jaworski, Christopher P. Breen, Victor Schultz, Travis Hart, Joshua S. Fishman, Luke Rogers, Hanyu Gao, Robert W. Hicklin, Pieter Plehiers, Joshua Byington, John S. Piotti, William H. Green, A. John Hart, Timothy F. Jamison, and Klavs F. Jensen. 2019. A robotic platform for flow synthesis of organic compounds informed by AI planning. Science, 365(6453):eaax1566.

Yiming Cui, Xin Yao, Yuxuan Qin, Xin Li, Shijin Wang, and Guoping Hu. 2025. Evaluating large language models on multimodal chemistry olympiad exams. Communications Chemistry, 8:402.

Kourosh Darvish, Marta Skreta, Yuchi Zhao, Naruki Yoshikawa, Sagnik Som, Miroslav Bogdanovic, Yang Cao, Han Hao, Haoping Xu, Alán Aspuru-Guzik,

Animesh Garg, and Florian Shkurti. 2025. OR-GANA: A robotic assistant for automated chemistry experimentation and characterization. Matter, 8(2):101897.

Rama El-Khawaldeh, Mason Guy, Finn Bork, Nina Taherimakhsousi, Kris N. Jones, Joel M. Hawkins, Lu Han, Robert P. Pritchard, Blaine A. Cole, Sebastien Monfette, and Jason E. Hein. 2024a. Keeping an “eye” on the experiment: Computer vision for real-time monitoring and control. Chemical Science, 15(4):1271–1282.

Rama El-Khawaldeh, Abhijoy Mandal, Naruki Yoshikawa, Wenyu Zhang, Ryan Corkery, Paloma Prieto, Alán Aspuru-Guzik, Kourosh Darvish, and Jason E. Hein. 2024b. From eyes to cameras: Computer vision for high-throughput liquid–liquid separation. Device, 2(7):100404.

Sagi Eppel, Haoping Xu, Mor Bismuth, and Alan Aspuru-Guzik. 2020. Computer vision for recognition of materials and vessels in chemistry lab settings and the Vector-LabPics data set. ACS Central Science, 6(10):1743–1752.

Google DeepMind. 2026a. Gemini 3.1 Flash-Lite Model Card. Technical report, Google DeepMind.

Google DeepMind. 2026b. Gemini 3.1 Pro Model Card. Technical report, Google DeepMind.

Taicheng Guo, Kehan Guo, Bozhao Nan, Zhenwen Liang, Zhichun Guo, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. 2023. What can large language models do in chemistry? a comprehensive benchmark on eight tasks. In Advances in Neural Information Processing Systems, volume 36, pages 59662–59688.

Florian Häse, Loïc M. Roch, and Alán Aspuru-Guzik. 2019. Next-generation experimentation with selfdriving laboratories. Trends in Chemistry, 1(3):282– 291.

Yuqing Huang, Rongyang Zhang, Xuesong He, Xuyang Zhi, Hao Wang, Nuo Chen, Zongbo Liu, Xin Li, Feiyang Xu, Deguang Liu, Huadong Liang, Yi Li, Jian Cui, Yin Xu, Shijin Wang, Qi Liu, Defu Lian, Guiquan Liu, and Enhong Chen. 2026. ChemEval: A multi-level and fine-grained chemical capability evaluation for large language models. In The Fourteenth International Conference on Learning Representations.

Zhengzhao Lai, Youbin Zheng, Zhenyang Cai, Haonan Lyu, Jingpu Yang, Hong-Qing Liang, Yan Hu, and Benyou Wang. 2025. Can multimodal LLMs see materials clearly? a multimodal benchmark on materials characterization. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, pages 4384–4404, Suzhou, China. Association for Computational Linguistics.

Jon M. Laurent, Joseph D. Janizek, Michael Ruzo, Michaela M. Hinks, Michael J. Hammerling, Siddharth Narayanan, Manvitha Ponnapati, Andrew D. White, and Samuel G. Rodriques. 2024. LAB-Bench: Measuring capabilities of language models for biology research. Preprint, arXiv:2407.10362.

Hanzheng Li, Xi Fang, Yixuan Li, Chaozheng Huang, Junjie Wang, Xi Wang, Hongzhe Bai, Bojun Hao, Shenyu Lin, Huiqi Liang, Linfeng Zhang, and Guolin Ke. 2025a. RxnBench: A multimodal benchmark for evaluating large language models on chemical reaction understanding from scientific literature. arXiv preprint arXiv:2512.23565.

Junxian Li, Di Zhang, Xunzhi Wang, Zeying Hao, Jingdi Lei, Qian Tan, Cai Zhou, Wei Liu, Yaotian Yang, Xinrui Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Wei Li, Mao Su, Shufei Zhang, Wanli Ouyang, Yuqiang Li, and Dongzhan Zhou. 2025b. ChemVLM: Exploring the power of multimodal large language models in chemistry area. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 415–423.

Shiwei Lin, Zesen Chen, Xiaobin Jia, Yi Wang, Chenxu Wang, Shuyuan Zhang, Xiaozhen Ding, Boyuan Du, Weili Ding, and Huaping Liu. 2025a. A visual dataset for anomaly detection in self-driving laboratories. Scientific Data, 12(1):1787.

Shiwei Lin, Chenxu Wang, Xiaozhen Ding, Yi Wang, Boyuan Du, Lei Song, Chenggang Wang, and Huaping Liu. 2025b. A VLM-based method for visual anomaly detection in robotic scientific laboratories. In 2025 10th IEEE International Conference on Advanced Robotics and Mechatronics (ICARM), pages 34–39. IEEE.

Xingyu Lu, He Cao, Zijing Liu, Shengyuan Bai, Leqing Chen, Yuan Yao, Hai-Tao Zheng, and Yu Li. 2024. MoleculeQA: A dataset to evaluate factual accuracy in molecular comprehension. In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 3769–3789, Miami, Florida, USA. Association for Computational Linguistics.

MiniMax. 2026. MiniMax M3: Frontier Coding, 1M Context, Native Multimodality — All in One Model. MiniMax Research.

Adrian Mirza, Nawaf Alampara, Sreekanth Kunchapu, Martiño Ríos-García, Benedict Emoekabu, Aswanth Krishnan, Tanya Gupta, Mara Schilling-Wilhelmi, Macjonathan Okereke, Anagha Aneesh, Mehrdad Asgari, Juliane Eberhardt, Amir Mohammad Elahi, Hani M. Elbeheiry, María Victoria Gil, Christina Glaubitz, Maximilian Greiner, Caroline T. Holick, Tim Hoffmann, and 16 others. 2025. A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists. Nature Chemistry, 17(7):1027– 1034.

OpenAI. 2026. GPT-5.5 System Card. System card, OpenAI.

Qwen Team. 2026. Qwen3.6-Plus: Towards Real World Agents. Qwen technical blog.

Nicholas T. Runcie, Charlotte M. Deane, and Fergus Imrie. 2026. Assessing the chemical intelligence of large language models. Journal of Chemical Information and Modeling, 66(1):216–227.

Gary Tom, Stefan P. Schmid, Sterling G. Baird, Yang Cao, Kourosh Darvish, Han Hao, Stanley Lo, Sergio Pablo-García, Ella M. Rajaonson, Marta Skreta, Naruki Yoshikawa, Samantha Corapi, Gun Deniz Akkoc, Felix Strieth-Kalthoff, Martin Seifrid, and Alán Aspuru-Guzik. 2024. Self-driving laboratories for chemistry and materials science. Chemical Reviews, 124(16):9633–9732.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9568–9578.

xAI. 2026. Grok 4.3. xAI developer documentation.

Jiaqing Xie, Weida Wang, Ben Gao, Zhuo Yang, Haiyuan Wan, Shufei Zhang, Tianfan Fu, and Yuqiang Li. 2025. QCBench: Evaluating large language models on domain-specific quantitative chemistry. Journal of Chemical Information and Modeling, 65(22):12268–12278.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. 2025. MMMU-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, Vienna, Austria. Association for Computational Linguistics.

Jinwei Zhang, Xucheng Liang, Yu Zhang, Ruijie Yu, Xiaokang Yang, Yaohui Jin, and Yanyan Xu. 2026. ChemReason-bench: Benchmarking large language models for procedural reasoning in experimental chemistry. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 33211–33248. Association for Computational Linguistics.

Yitong Zhou, Mingyue Cheng, Qingyang Mao, Yucong Luo, Qi Liu, Yupeng Li, Xiaohan Zhang, Deguang Liu, Xin Li, and Enhong Chen. 2025. Benchmarking multimodal llms on recognition and understanding over chemical tables. arXiv preprint arXiv:2506.11375.

Yujun Zhou, Jingdong Yang, Yue Huang, Kehan Guo, Zoe Emory, Bikram Ghosh, Amita Bedar, Sujay Shekar, Zhenwen Liang, Pin-Yu Chen, Tian Gao, Werner Geyer, Nuno Moniz, Nitesh V. Chawla, and Xiangliang Zhang. 2026. Benchmarking large language models on safety risks in scientific laboratories. Nature Machine Intelligence, 8:20–31.

Yicheng Zou, Dongsheng Zhu, Lin Zhu, et al. 2026. Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale. arXiv preprint arXiv:2603.25040.

## A Appendix

## A.1 Standardized Question-Authoring Protocol

This protocol is intended to guide experts in constructing initial benchmark questions based on authentic chemical laboratory materials. Each question should assess the model’s ability to integrate visual evidence, instrumental information, or experimental observations within a specific experimental context to perform state identification, cause analysis, result interpretation, risk assessment, or operational decision-making.

Questions must not merely assess textbook knowledge detached from experimental scenarios, nor should they depend on information that is not provided in the image, question prompt, or supplied background. Each question must have a clearly defined experimental judgment target and must be supportable by the provided evidence together with relevant domain knowledge. This protocol covers only the construction of the initial question draft. After completion, each question will proceed to independent expert review, revision, and, when necessary, third-party adjudication. Each question is assigned only one primary category to ensure mutual exclusivity of categories in statistical analysis and evaluation. Question materials should satisfy the following requirements:

• Images, instrument outputs, or experimental observations must be directly relevant to the judgment target being assessed.

• Key visual or experimental cues should be reasonably observable and should not depend on difficult-to-discern minor details, unavailable preceding or subsequent context, or the author’s personal memory.

• Image cropping, rotation, de-identification, or resolution adjustment must not alter the original experimental facts or obscure key evidence.

• Images may show normal operations, abnormal states, incorrect settings, instrument outputs, experimental results, or representative process phenomena, but a question must not be constructed merely because an image “looks abnormal.”

## A.1.1 Organic Synthesis

Organic-synthesis questions are organized according to authentic experimental workflows into four modules and sixteen subcategories (Table 4).

<table><tr><td>categories</td><td>Subcategories</td></tr><tr><td>Experiment Preparation</td><td>Experiment_plan; Experiment_Safety</td></tr><tr><td>Experiment Monitor Experiment_Monitor;</td><td></td></tr><tr><td>Experiment</td><td>Experiment_notes Extraction; Evaporation; Column;</td></tr><tr><td>Postprocess</td><td>Recrystallization; Trituration; TLC; Lyophilization</td></tr><tr><td>Experiment</td><td>NMR; LCMS; Meltingpoint; Crystal;</td></tr><tr><td>Characterization</td><td>GCMS</td></tr></table>

Table 4: Organic-synthesis categories and subcategories.

## A.1.2 Materials Science

Materials science questions are uniformly labeled as Materials Science. The question author should briefly describe the experimental scenario in the question metadata, such as materials preparation, testing conditions, microscopic observation, structural or performance characterization, interpretation of instrument outputs, or analysis of experimental abnormalities. Materials-science questions should be organized according to their own experimental scenarios and are not required to use the four-stage labels defined for organic synthesis.

## A.2 Question-Design Principles

Questions should be centered on key evidence contained in the provided image or experimental information. The question author should be able to clearly answer: 1. What key evidence must the model identify? 2. What experimental judgment does this evidence support? 3. Does this judgment require integration with the provided experimental background? 4. Are there other equally reasonable interpretations?

When the available evidence is insufficient for a specific causal attribution, the question should focus on judgments supported by the experimental context, such as identifying the observed state or determining an appropriate verification step. Questions with unresolved ambiguity should be revised or excluded.

## A.2.1 Single Core Judgment Target

Multiple interrelated subquestions are permitted, but they must all serve the same experimental decision chain. For example, “whether the observed phenomenon is sufficient to determine reaction completion—how it should be verified—what should be prioritized for optimization if an abnormality is present” may be formulated as a continuous monitoring-decision question. Multiple independent knowledge questions must not be mechanically combined into a single question. Each question should have one primary experimental judgment target, for example: 1.determining whether an operation is appropriate or poses a risk; 2.determining whether a reaction has reached the expected state; 3.identifying a possible cause of an abnormal observation; 4.selecting an appropriate workup, purification, or characterization operation; 5.interpreting an instrumental or characterization result; 6.proposing an actionable corrective measure that is consistent with the evidence.

## A.2.2 Authentic Experimental Context

The question prompt should provide the minimum sufficient background required for answering, including the necessary experimental stage, object of operation, observed phenomenon, instrument type, or known conditions. Questions should not require the model to guess reagent identities, reaction types, concentrations, temperatures, instrument parameters, or previous operations that have not been provided. Questions should primarily assess judgments made in authentic experimental settings rather than use generalized statements detached from a specific context. For example:

It is not appropriate to ask only, “What causes tailing in TLC?”

Instead, the question should ask, “Based on the provided TLC image, developing system, and sample state, how should the current spot abnormality be interpreted and how should the conditions be adjusted?”

## A.2.3 Evidence Sufficiency and Verification

Color, turbidity, precipitation, foam, liquid-level changes, band morphology, or instrument readings may serve as important cues. However, unless such a phenomenon has sufficient and verifiable specificity within the given experimental system, it must not be used as the sole basis for determining product formation, reaction completion, or yield. When additional verification is relevant to the target judgment, appropriate analytical methods, such as TLC, LCMS, GCMS, NMR, or melting-point analysis, may be considered within the experimental context.

## A.3 Required Content for Each Question

Each expert author should submit the following content. The question should: 1.clearly describe the experimental scenario; 2.specify the judgment task that the model is required to perform; 3.provide the minimum necessary background required to make the judgment; 4.use accurate, actionable, and unambiguous experimental terminology; 5.avoid revealing the correct answer through wording, answer ordering, or obvious hints.

{   
"category": "one primary category",   
"experimental\_stage\_or\_context":   
"experimental stage or materials-science   
experimental context",   
"evidence\_type":   
"operation image, instrument interface,   
spectrum, chromatogram, microscopy   
image, experimental observation, or   
other evidence type",   
"image":   
"corresponding image filename(s), which   
may contain a single image or multiple   
images",   
"target\_decision":   
"the primary experimental judgment target   
assessed by the question"   
}

Each question should include an internally used description of key evidence, including: 1.the most important observable cues in the image or experimental record; 2.the relationship between those cues and the experimental judgment; 3.reasonable alternative explanations or alternative operations that should be accepted; 4.core incorrect judgments that should not be accepted.

## A.3.1 Reference Answer and Tiered Scoring Anchors

The Score 3 / Score 2 / Score 1 / Score 0 anchors should reflect a monotonically decreasing scoring logic (Table 5). Lower-scoring answers should primarily differ through missing key information, incomplete evidence chains, or incorrect core conclusions. In addition to the four scoring anchors, each question should include an internal scoring basis specifying at least: 1.the key judgment(s) that must be included in a full-score answer; 2.acceptable alternative phrasings, verification methods, or corrective measures; 3.key omissions that would result in substantial score deductions; 4.core errors or dangerous recommendations that would result in a score of zero. If multiple scientifically equivalent answer pathways exist, they should be explicitly documented in the scoring basis to avoid treating a single wording or a single operational route as the only correct answer.

<table><tr><td>Anchor</td><td>Score</td><td>Definition</td></tr><tr><td>Score 3 anchor</td><td>3</td><td>Complete, scientifically correct, consistent with the supplied evidence, and covers the core judgment, causal explanation, and necessary operational recommendations required by the question.</td></tr><tr><td>Score 2 anchor</td><td>2</td><td>The core judgment is correct, but key evidence, verification steps, causal explanation, or corrective measures are omitted; no substantive error that would change the conclusion should be present.</td></tr><tr><td>Score 1 anchor</td><td>1</td><td>Identifies some relevant observations or proposes a directionally reasonable recommendation, but does not complete the core judgment, lacks the necessary evidence chain, or uses an approach insufficient to support the conclusion.</td></tr><tr><td>Score 0 anchor</td><td>0</td><td>The core judgment is incorrect, contradicts the supplied evidence, proposes clearly inappropriate or unsafe operations, or is irrelevant to the question.</td></tr></table>

Table 5: Four-level scoring anchors for benchmark questions.

## A.4 Common Reasons for Rejection or Revision

The Score 3 / Score 2 / Score 1 / Score 0 anchors should not contain evaluative statements such as “this answer is incorrect,” “this answer lacks expertise,” or “this option may mislead the model.” Such information should instead be placed in the internal scoring basis. The following situations should be avoided during initial drafting or returned for revision during review: 1.the image is not directly related to the question prompt; 2.a deterministic conclusion is drawn solely from color, appearance, or a single observation beyond what the evidence can support; 3.the question depends on experimental conditions that have not been provided; 4.the visual evidence is insufficient to distinguish correct from incorrect answers; 5.multiple equally reasonable answers exist but are not covered by the scoring basis; 6.the question primarily tests textbook recall rather than using multimodal experimental information; 7.the Score 3 / Score 2 / Score 1 / Score 0 anchors do not exhibit a clear hierarchy of performance; 8.the full-score answer contains irrelevant, unverifiable, or overly specific operational recommendations;

## A.5 Expert Self-Check Checklist

Before submitting an initial draft, confirm each of the following:

• Is the question based on an authentic experimental scenario or authentic experimental data?

• Does the question have only one clearly defined primary experimental judgment target?

• Can the model make the required judgment using only the provided image, question prompt, and necessary background?

• Have the key visual or experimental cues been clearly identified?

• Is the reference answer scientifically correct and consistent with the evidence?

• Are there multiple reasonable answers? If so, have they been documented in the scoring basis?

• Do the Score 3 / Score 2 / Score 1 / Score 0 anchors form a clear and monotonic scoring hierarchy?

• Is the Score 2 anchor “core judgment correct but incomplete,” rather than containing a decisive error?

• Does the Score 0 anchor contain a clear error in the core judgment or operational recommendation?

• Has the question been accurately assigned to one primary category?

• Does the question avoid revealing the correct answer through prompt wording, answer phrasing, or meta-commentary?

• Can the question be independently reviewed by another expert with comparable operational experience?

## A.6 Standard Submission Format

```jinja
{
"category": "Experiment_Monitor",
"experimental_stage_or_context": "reaction
monitoring",
"evidence_type": [
"reaction-image"
],
"image": [
"Experiment_Monitor_001_01.png"
],
"target_decision":
```

```jsonl
"determine whether visual appearance alone
can establish reaction completion and
identify appropriate verification and
troubleshooting steps",
"question": "...",
"key_evidence": [
11
],
"answers": [
{
"label": "Score 3",
"score": 3,
"answer": "..."
},
{
"label": "Score 2",
"score": 2,
"answer": "..."
},
{
"label": "Score 1",
"score": 1,
"answer": "...
},
{
"label": "Score 0",
"score": 0,
"answer": "..."
}
]
}
```

## B Contributor Qualifications and Backgrounds

LabOPBench was constructed and reviewed by 22 contributors with hands-on laboratory experience relevant to the experimental tasks represented in the benchmark. The contributor pool included researchers and laboratory specialists at different career stages, providing experience across reaction setup, experimental monitoring, workup and purification, characterization, and laboratory troubleshooting.

## B.1 Contributor composition

The 22 contributors comprised four PhD researchers, six postdoctoral researchers, eight laboratory specialists, and four associate researchers or above. The information is summarized in Table 6.

<table><tr><td>Position</td><td>Contributors</td></tr><tr><td>PhD researcher Postdoctoral researcher</td><td>4</td></tr><tr><td>Laboratory specialist</td><td>6</td></tr><tr><td>Associate researcher or above</td><td>8</td></tr><tr><td>Total</td><td>4 22</td></tr></table>

Table 6: Contributor composition by position.

All contributors had at least 3 years of laboratory experience. Two contributors had 3–5 years of experience, eight had 5–7 years, four had 7–10 years, and eight had more than 10 years of laboratory experience. Overall, 20 of the 22 contributors had at least 5 years of laboratory experience, 12 had at least 7 years, and 8 had more than 10 years of experience. The information is summarized in Table 7.

<table><tr><td>Laboratory experience</td><td>Contributors</td><td>Proportion</td></tr><tr><td>3-5 years</td><td>2</td><td>9.1%</td></tr><tr><td>5–7 years</td><td>8</td><td>36.4%</td></tr><tr><td>7–10 years</td><td>4</td><td>18.2%</td></tr><tr><td>&gt; 10 years</td><td>8</td><td>36.4%</td></tr><tr><td>Total</td><td>22</td><td>100.0%</td></tr></table>

Table 7: Contributor laboratory experience.

Contributors were selected on the basis of handson laboratory experience relevant to the experimental situations represented in LabOPBench. Collectively, their expertise covered reaction setup, experimental monitoring, workup and purification, characterization, instrument operation, and troubleshooting. This range of experience supported both the construction of laboratory-grounded questions and the independent assessment of their scientific correctness, evidence sufficiency, and experimental relevance.

## C Model Evaluation Setup and Parameters

Eight multimodal models were evaluated using an OpenAI-compatible Chat Completions interface. All benchmark questions were evaluated independently using a common input format and answergeneration instruction. The evaluation framework supplied the question and associated visual evidence to the answer models without reference answers or scoring rubrics. Each benchmark question was evaluated in an independent API request without conversational history. No system message was used. For multimodal questions, a single user message contained the following components in order: one image\_url block for each associated image; the question text; and the common answer-generation instruction.

Image filenames and local image paths appearing in the question text were replaced with neutral references such as “Figure 1” before submission. The question identifier, category, reference answers and question-specific scoring rubric were not provided to the answer model. Images were read from the local image directory, Base64 encoded and transmitted as data URLs in the following format: data:{MIME type};base64,{encoded image}

## C.1 Answer prompt

All eight answer models received the same instruction, presented below in a structured format.

```json
{
"instruction": "Please carefully analyze the
question and images (if any) above, and
provide your answer.",
"requirements": [
"1. If the question contains multiple
sub-questions, address EACH one
individually.",
"2. For each sub-question, identify and
provide MAIN cause and its
corresponding optimal solution. DO NOT
list multiple causes, multiple
solutions, or redundant details for
any single sub-question.",
"3. Your answer should be accurate and
professional.",
"4. Output the answer directly and
concisely.",
"5. If the question asks \"why\", provide
only the core reason together with the
answer, no extra content."
]
}
```

Answer generation was attempted up to ten times for each question, with a fixed 3-s delay after an unsuccessful attempt. No exponential backoff, jitter or explicit request timeout was configured. An empty model response was treated as a failed attempt and retried. If all ten attempts failed, the corresponding output record contained predicted\_output=null and error=true. Failure of an individual question did not terminate evaluation of the remaining benchmark questions, and records for which answer generation ultimately failed were not subsequently scored.

## D Automated Scoring Method and Prompts

Model responses were independently evaluated by two judge models, GPT-5.5 (gpt-5.5) and Gemini 3.1 Pro (gemini-3.1-pro-preview). Each judge evaluated the candidate response against the questionspecific 3-, 2-, 1-, and 0-point rubric levels. The two judge-specific evaluation results were retained separately and subsequently combined to obtain the reported score.

## D.1 Scoring framework

The scoring framework used a 0–3 scale. The default procedure was holistic rubric matching, in which the candidate answer was matched to the single rubric level that it best satisfied. Split scoring was used only when the original question explicitly contained multiple independent subquestions. For explicitly numbered subquestions, split scoring was mandatory. Each subquestion was scored independently, and the resulting subquestion scores were averaged and rounded to the nearest 0.5. For each holistic answer or explicit subquestion, the judge was instructed to evaluate the first substantive claim or first complete procedure that directly answered the question. Introductory material, later alternative explanations, alternative solutions, and redundant details were ignored unless they contradicted the initially scored claim, changed the final recommendation, or introduced a dangerous operation. Scores of 3, 2, 1, and 0 were assigned by matching the response to the scenario-specific rubric rather than by isolated keyword overlap. Half-point scores were permitted under the conditions defined in the scoring prompt: 2.5 when the response fully satisfied the 2-point rubric level and partially satisfied a key differentiating requirement of the 3-point level; 1.5 when the response fully satisfied the 1-point rubric level and partially satisfied a key differentiating requirement of the 2-point level; 0.5 when the response was relevant and experimentally reasonable but did not clearly satisfy any positive rubric level.

## D.2 Complete automated-scoring prompt

Both judge models received the following scoring prompt:

{ instructions": [   
"You are a senior laboratory evaluator in   
experimental sciences.",   
"Your task is to score a model-generated   
experimental response against the provided   
A-D rubric options.",   
"The rubric options are authoritative. Do not   
replace them with your own general laboratory   
judgment."   
],   
"## Original question:": "{question\_section}",   
"## Student answer:": "{answer}",   
"## Ground truth and scoring rubric options:":   
"{options\_text}",   
"Scoring rules": {   
"Step 1 - Decide the scoring mode": [

"Default mode is holistic A-D matching:   
choose the single rubric option whose   
complete answer is best matched by the   
student answer.",   
"Use split scoring only when the original   
question explicitly asks multiple independent   
sub-questions, such as numbered questions   
\"1, 2, 3\" or clearly separate tasks.",   
"If the original question explicitly contains   
numbered sub-questions, split scoring is   
mandatory. Do not use holistic scoring merely   
because the rubric options are written as   
integrated answers.",   
"Do not create scoring units from details   
inside the rubric. Scoring units may only   
come from the original question.",   
"If the question asks for a cause plus a   
solution, or a diagnosis plus an operation,   
treat them as one holistic answer unless the   
question explicitly separates them into   
independent sub-questions."

## ],

"Step 2 - Apply the first-point rule": [ "For each holistic answer or explicit sub-question, score only the first substantive claim or first complete procedure that directly answers it.", "Ignore introductions, background comments, later alternative causes, later alternative solutions, and redundant details.", "Later information may affect the score only if it contradicts the first scored claim, changes the final recommendation, or introduces a dangerous operation."

## ],

"Step 3 - Match to the A-D rubric": [ "Match the first scored content to the rubric option A/B/C/D by scenario-specific meaning, not by isolated keywords.", "Treat each rubric option as a complete condition-action package, not as a menu of independent keywords. The score should reflect the best complete package matched by the answer.", "A score of 3 requires matching the key differentiating details of the 3-point option.", "A higher score requires matching all key differentiating components of that rubric option when the option explicitly contains multiple required components, such as observation or diagnosis, cause, and required operation.", "If the answer matches only one component, such as only the cause or only one operation parameter, do not assign the full score for that rubric option.", "A score of 2 requires matching the key differentiating details of the 2-point option, without satisfying the 3-point differentiators.", "A score of 1 requires matching the 1-point option or a clearly valid low-level scoring element supported by the rubric.", "A score of 0 is required if the answer is irrelevant, opposite to the correct conclusion, unsafe, matches the 0-point option, or provides no score-bearing

information.",

"If an answer is generally scientifically   
plausible but misses the scenario-specific   
key point required by the rubric, it cannot   
receive 2 or 3 points. It may receive at most   
0.5 limited credit.",   
"Do not upgrade a score because the answer   
sounds reasonable in general. The rubric is   
more important than general plausibility.",   
"Do not upgrade a score merely because the   
answer shares a broad goal, general   
mechanism, or laboratory category with a   
higher-score option. It must match the   
higher-score option's specific diagnostic   
premise and required operation.",   
"If the rubric specifies a concrete required   
operation, a generic operation category or a   
different operation with a similar intended   
effect is not enough for full credit unless   
the rubric explicitly allows such   
substitutions.",   
"Visual Evidence Rule:",   
1 - Do not require the answer to restate   
visual details merely because an image is   
present.",   
11 - Treat a visual observation as   
score-bearing only when the question   
explicitly asks for it, or when the rubric   
uses it to distinguish between score levels.",   
11 - If the answer matches the required   
diagnosis and operation but omits a   
non-contradictory visual observation required   
by the target higher-score rubric level, cap   
the score at the next lower full-score   
level.",   
11 For example, a 3-point option is capped   
at 2.0, and a 2-point option is capped at   
1.0.",   
11 - If the answer contradicts a visual   
observation that is central to the rubric   
distinction, do not assign the contradicted   
higher-score level. Score it according to the   
lower rubric level it best matches.",   
11 - If the visual observation is only   
background context and not a differentiating   
rubric component, do not penalize the   
omission.",

"If the rubric's higher-score option depends on a specific physical state or material form, the answer must match that state for full credit. If the answer replaces it with a different physical state or material form, do not assign that higher-score level; score it according to the lower rubric level it best matches, with a maximum of 1.0 if no lower level clearly matches.", "Do not treat different scenario-critical diagnoses as equivalent unless the rubric explicitly says they are equivalent.", "Scenario-critical diagnoses include, but are not limited to: physical phenomena such as oiling out, precipitation, emulsion, crystallization, phase separation, undissolved solid, and dark mother liquor; chromatography and TLC issues such as tailing, overloading, adsorption, solvent-polarity mismatch, decomposition, poor separation, fronting, saturation, and

```csv
baseline drift; spectroscopy and MS
assignments such as solvent peak. impurity
peak, product ion, fragment ion, adduct,
isotope peak, co-elution, wrong molecular
weight. and wrong structural assignment:
safety and operation diagnoses such as air
leakage, water ingress, overheating, wrong
apparatus setup, wrong layer selection, and
unsafe reagent handling.",
"If the first diagnosis identifies a
different scenario-critical problem from the
target rubric, the score for that holistic
answer or sub-question cannot exceed 1.0,
even if later cause or solution details
overlap with a higher-score option."
],
"Step 4 - Half-point rule": [
"Use half-point scores sparingly.",
"Assign 2.5 only when the answer fully
satisfies the 2-point option and partially
satisfies a key differentiating detail of the
3-point option.",
"Assign 1.5 only when the answer fully
satisfies the 1-point option and partially
satisfies a key differentiating detail of the
2-point option.",
"Assign 0.5 only when the answer is relevant
and experimentally reasonable but does not
clearly match any positive rubric option.",
"Do not assign half points merely because the
answer is verbose, partially correct in a
vague way, or scientifically plausible
outside this scenario."
],
"Step 5 - Split scoring, only when explicitly
needed": [
"If split scoring is required, score each
explicit sub-question using Steps 2-4.",
"If one explicit sub-question is not
answered, assign 0 to that sub-question.",
"The final score is the mean of the explicit
sub-question scores, rounded to the nearest
0.5.",
"If split scoring is not required, output the
single holistic score."
],
"Step 6 - Rationale requirements": [
"The reason must state:",
"1. Whether holistic scoring or split scoring
was used, and why.",
"2. The first substantive claim/procedure
that was scored."
"3. The matched rubric option or score
level.",
"4. Why the answer did or did not meet the
higher-score differentiating details.",
"5. Whether any extra information was
ignored, contradicted the scored point, or
introduced safety risk."
]
},
"Output requirements:": [
"1. Output exactly in this format:",
"##JSON Output:",
"{\"score\":2,\"reason\":\"...\"}",
"2. The line after \"##JSON Output:\" must be
```

one single valid JSON object.",   
"3. Do not output markdown code fences.",   
"4. Do not output extra text before or after   
the JSON object."   
],   
"final\_instruction": "The JSON object must   
contain exactly two fields: \"score\" (float)   
and \"reason\" (string), in that order."   
}

## D.3 Normalized-score calculation and judge aggregation

Normalized scores were calculated from questionlevel rubric scores as defined in Section 3.4. For full-benchmark reporting, judge-specific scores were first calculated for each fine-grained subcategory and the Materials Science category. Majorcategory scores were obtained as question-countweighted averages of the corresponding subcategory scores, and overall scores were aggregated across the five major categories using their question counts. Judge-specific scores were rounded to one decimal place at each aggregation level. The final reported score was the arithmetic mean of the two judge-specific scores, rounded to one decimal place. Scores for the controlled 100-question subset were calculated directly over the 100 questions.

## E Expert Validation of Automated Scoring

Automated scoring was validated using responses generated by Claude Opus 4.7. Questions were stratified across the five major categories-Experiment Preparation, Experiment Monitor, Experiment Postprocess, Experiment Characterization, and Materials Science-and approximately 20% of the questions in each major category were randomly sampled. Within each of the four organicsynthesis major categories, sampling preserved as closely as possible the proportional distribution of the corresponding fine-grained subcategories. Materials Science was sampled directly because it was not further subdivided into fine-grained subcategories. Two experts with relevant experimental experience independently rescored all sampled responses. The experts received the same scoring prompt used for the automated judges. During scoring, neither expert was provided with the automated-judge scores or the score assigned by the other expert. All scores were normalized to percentages relative to the maximum score of 3. The mean automated score was calculated as the arithmetic mean of the scores from the GPT-5.5 and Gemini 3.1 Pro judges. The mean expert score was calculated as the arithmetic mean of the two independent expert ratings. Category-level differences were calculated as the mean automated score minus the mean expert score and are reported in percentage points (Table 8). Across the five major categories, the difference between mean automated and mean expert scores ranged from −0.59 to +1.39 percentage points. The dual-judge mean achieved a mean absolute error (MAE) of 0.165 points on the 0-3 scale relative to the two-expert mean, with 98.0% of responses differing by no more than 0.5 points (quadratic-weighted $\kappa = 0 . 9 1 6 1 )$ . The individual MAEs were 0.185 and 0.195 for GPT-5.5 and Gemini 3.1 Pro, respectively. The two experts had an inter-rater MAE of 0.330 points $( \kappa = 0 . 7 5 1 5 )$ .

<table><tr><td>Scoring source</td><td>Preparation</td><td>Monitoring</td><td>Postprocess</td><td>Characterization</td><td>Material Science</td></tr><tr><td>GPT-5.5 judge</td><td>48.72%</td><td>63.89%</td><td>40.20%</td><td>47.62%</td><td>59.26%</td></tr><tr><td>Gemini 3.1 Pro judge</td><td>50.00%</td><td>63.89%</td><td>40.39%</td><td>48.21%</td><td>60.19%</td></tr><tr><td>Automated mean</td><td>49.36%</td><td>63.89%</td><td>40.29%</td><td>47.92%</td><td>59.72%</td></tr><tr><td>Expert rater 1</td><td>48.72%</td><td>61.11%</td><td>40.00%</td><td>48.21%</td><td>58.33%</td></tr><tr><td>Expert rater 2</td><td>47.44%</td><td>63.89%</td><td>40.20%</td><td>48.81%</td><td>59.26%</td></tr><tr><td>Expert mean</td><td>48.08%</td><td>62.50%</td><td>40.10%</td><td>48.51%</td><td>58.80%</td></tr><tr><td>Automated — expert</td><td>+1.28 pp</td><td>+1.39 pp</td><td>+0.19 pp</td><td>-0.59 pp</td><td>+0.92 pp</td></tr></table>

Table 8: Automated and expert scores by major category.

## F Model Performance on subcategories

To provide the complete fine-grained results underlying the major-category performance reported in the main text, model scores were further summarized for each organic-synthesis subcategory and for the materials-science category, as shown in Table 9 to 13. The tables below report the two judge-specific normalized scores for each evaluated model within each fine-grained task, together with the number of questions in each subcategory.

The organic-synthesis benchmark was divided into Experiment Preparation, Experiment Monitoring, Experiment Postprocess, and Experiment Characterization, with the corresponding fine-grained subcategories shown below. Materials Science was retained as a separate major category and was not further subdivided in this analysis.

In each table entry, the value to the left of the slash represents the GPT-5.5 judge score, and the value to the right represents the Gemini 3.1 Pro judge score.

Materials Science (n = 92)
<table><tr><td>Model</td><td>Judge scores</td></tr><tr><td>Claude Opus 4.7</td><td>58.2%/61.4%</td></tr><tr><td>Gemini 3.1 Pro</td><td>51.1%/55.8%</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>46.0%/49.1%</td></tr><tr><td>GPT-5.5</td><td>52.4% /53.3%</td></tr><tr><td>Grok 4.3</td><td>52.0%/55.3%</td></tr><tr><td>Qwen3.6-Plus</td><td>50.2%/55.3%</td></tr><tr><td>MiniMax-M3</td><td>54.0%/57.1%</td></tr><tr><td>Intern-S1-Pro</td><td>41.3%/42.0%</td></tr></table>

Table 9: Model performance for materials-science tasks.

Experiment Preparation (n = 67)
<table><tr><td>Model</td><td>Experiment_ plan (n = 46)</td><td>Experiment_ Safety (n = 21)</td></tr><tr><td>Claude Opus 4.7</td><td>54.7% /54.7%</td><td>33.3%/40.5%</td></tr><tr><td>Gemini 3.1 Pro</td><td>41.7%/43.1%</td><td>29.4%/27.0%</td></tr><tr><td>Gemini 3.1</td><td>44.9% /47.5%</td><td>31.0%/28.6%</td></tr><tr><td>Flash-Lite</td><td></td><td></td></tr><tr><td>GPT-5.5</td><td>50.4% /49.6%</td><td>42.9% /50.8%</td></tr><tr><td>Grok 4.3</td><td>37.3%/39.1%</td><td>27.8%/33.3%</td></tr><tr><td>Qwen3.6-Plus</td><td>40.2%/41.7%</td><td>26.2% /32.5%</td></tr><tr><td>MiniMax-M3</td><td>49.3%/48.6%</td><td>42.9%/44.4%</td></tr><tr><td>Intern-S1-Pro</td><td>25.0% /23.6%</td><td>24.6% /23.0%</td></tr></table>

Table 10: Model performance for experimentpreparation subcategories.

Experiment Monitoring (n = 30)
<table><tr><td>Model</td><td>Experiment_ Monitor (n = 16)</td><td>Experiment_ notes (n = 14)</td></tr><tr><td>Claude Opus 4.7</td><td>44.8% /47.9%</td><td>85.7% /86.9%</td></tr><tr><td>Gemini 3.1 Pro</td><td>32.3% /35.4%</td><td>64.3%/64.3%</td></tr><tr><td>Gemini 3.1</td><td>31.3%/31.3%</td><td>52.4%/53.6%</td></tr><tr><td>Flash-Lite GPT-5.5</td><td>43.8%/41.7%</td><td>61.9%/65.5%</td></tr><tr><td>Grok 4.3</td><td>33.3%/41.7%</td><td>61.9%/64.3%</td></tr><tr><td>Qwen3.6-Plus</td><td>36.5%/36.5%</td><td></td></tr><tr><td>MiniMax-M3</td><td></td><td>76.2%/75.0%</td></tr><tr><td></td><td>44.8% /40.6%</td><td>65.5% /71.4%</td></tr><tr><td>Intern-S1-Pro</td><td>21.9% /26.0%</td><td>58.3% /58.3%</td></tr></table>

Table 11: Model performance for experimentmonitoring subcategories.

Postprocess (n = 425)
<table><tr><td>Model</td><td>Extraction (n = 38)</td><td>Evaporation (n = 46)</td><td>Column (n = 75)</td><td>Recrystallization (n = 16)</td></tr><tr><td>Claude Opus 4.7</td><td>48.2%/49.6%</td><td>48.9%/51.4%</td><td>44.9% /46.2%</td><td>32.3%/33.3%</td></tr><tr><td>Gemini 3.1 Pro</td><td>46.5% /44.3%</td><td>39.1%/40.6%</td><td>39.1%/43.1%</td><td>31.3% /36.5%</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>43.0% /42.1%</td><td>37.0%/37.0%</td><td>29.6% /32.7%</td><td>28.1%/35.4%</td></tr><tr><td>GPT-5.5</td><td>43.4%/46.5%</td><td>50.7%/50.4%</td><td>46.9% /46.7%</td><td>34.4%/38.5%</td></tr><tr><td>Grok 4.3</td><td>46.1%/44.7%</td><td>36.2%/36.6%</td><td>40.4%/41.3%</td><td>33.3% /36.5%</td></tr><tr><td>Qwen3.6-Plus</td><td>46.5% /46.9%</td><td>40.2%/40.2%</td><td>36.7% /42.2%</td><td>22.9% /30.2%</td></tr><tr><td>MiniMax-M3</td><td>43.0%/44.7%</td><td>39.9%/39.9%</td><td>34.4% /36.0%</td><td>34.4% /32.3%</td></tr><tr><td>Intern-S1-Pro</td><td>35.1%/36.8%</td><td>27.9%/23.2%</td><td>26.9% /28.2%</td><td>26.0%/29.2%</td></tr><tr><td>Model</td><td>Trituration (n = 25)</td><td colspan="2">Lyophilization (n = 24)</td><td>TLC (n = 201)</td></tr><tr><td>Claude Opus 4.7</td><td>36.7%/38.0%</td><td colspan="2">44.4%/43.8%</td><td>35.0%/34.9%</td></tr><tr><td>Gemini 3.1 Pro</td><td>24.0%/24.0%</td><td colspan="2">42.4%/45.1%</td><td>35.0% /35.7%</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>22.7% /18.0%</td><td colspan="2">39.6% /39.6%</td><td>32.5%/31.3%</td></tr><tr><td>GPT-5.5</td><td>33.3%/33.3%</td><td colspan="2">46.5%/47.9%</td><td>32.1%/32.3%</td></tr><tr><td>Grok 4.3</td><td>34.0%/33.3%</td><td colspan="2">38.2%/38.9%</td><td>27.8% /26.5%</td></tr><tr><td>Qwen3.6-Plus</td><td>26.7%/28.0%</td><td colspan="2">36.1%/36.8%</td><td>36.0% /36.0%</td></tr><tr><td>MiniMax-M3</td><td>22.0% /15.3%</td><td colspan="2">46.5%/47.2%</td><td>30.1%/28.3%</td></tr><tr><td>Intern-S1-Pro</td><td>18.0%/17.3%</td><td colspan="2">36.1%/36.1%</td><td>23.4% /21.6%</td></tr></table>

Table 12: Model performance for postprocessing subcategories.

Experiment Characterization (n = 142)
<table><tr><td>Model</td><td>NMR (n = 42)</td><td>LCMS (n = 48)</td><td>Meltingpoint  $( n = 8 )$ </td><td>Crystal (n = 29)</td><td>GCMS (n = 15)</td></tr><tr><td>Claude Opus 4.7</td><td>41.3%/41.7%</td><td>57.6% /59.0%</td><td>8.3%/6.3%</td><td>55.7%/55.7%</td><td>34.4% /33.3%</td></tr><tr><td>Gemini 3.1 Pro</td><td>47.2% /48.8%</td><td>54.2% /56.6%</td><td>18.8% /25.0%</td><td>49.4% /54.0%</td><td>31.1%/31.1%</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>38.1%/42.1%</td><td>36.8%/37.2%</td><td>16.7% /14.6%</td><td>40.8% /45.4%</td><td>31.1%/40.0%</td></tr><tr><td>GPT-5.5</td><td>44.4% /48.0%</td><td>54.2% /53.8%</td><td>16.7% /16.7%</td><td>47.1%/50.6%</td><td>38.9% /42.2%</td></tr><tr><td>Grok 4.3</td><td>31.0%/31.3%</td><td>45.5% /50.0%</td><td>12.5% /14.6%</td><td>42.0% /40.8%</td><td>24.4% /28.9%</td></tr><tr><td>Qwen3.6-Plus</td><td>42.1% /44.0%</td><td>55.6% /54.9%</td><td>22.9% /20.8%</td><td>36.8%/37.9%</td><td>42.2%/43.3%</td></tr><tr><td>MiniMax-M3</td><td>36.9% /37.3%</td><td>46.9% /49.3%</td><td>16.7% /16.7%</td><td>36.8%/36.8%</td><td>52.2% /52.2%</td></tr><tr><td>Intern-S1-Pro</td><td>31.0% /29.4%</td><td>29.5%/31.3%</td><td>10.4%/8.3%</td><td>32.8%/29.3%</td><td>35.6%/31.1%</td></tr></table>

Table 13: Model performance for characterization subcategories.

## G 100-question subset and human experiment design

We constructed a subset of 100 questions from the 664 organic-synthesis questions in the benchmark. The subset used fixed, approximately proportional quotas at two taxonomy levels. Within each major category, its quota was distributed across the constituent fine-grained subcategories approximately in proportion to their prevalence in the corresponding source category (Table 14).

## G.1 Human evaluation procedure

Ten participants were included in the human evaluation, comprising five undergraduate participants and five PhD participants. Participants in both groups had hands-on laboratory experience relevant to the experimental situations represented in the subset. Each participant independently completed all 100 questions in the controlled organicsynthesis subset. Participants were provided with the same question text, experimental context and associated visual evidence used for model evaluation and were instructed to answer the specific experimental judgment requested by each question.

Responses were expected to be direct and concise while addressing the required judgment. When a question contained multiple explicitly stated parts, participants were asked to address each part. Participants completed the questions independently, without discussion with other participants. Participants were instructed to answer using only the information supplied with each question together with their own laboratory knowledge and experience. External references, AI-assisted tools and other external resources were not permitted during the evaluation.

## G.2 Human-response scoring

Human answers were matched to the subset by question ID. The controlled-subset JSON served as the sole source of question order, category labels, original question text and item-specific scoring rubrics; the participant spreadsheets supplied only the human responses. This ensured that each human response was associated with exactly the same question-specific rubric used for evaluating model responses.

Each response was scored independently by two automated judges, GPT-5.5 and Gemini 3.1 Pro, using the same scoring program and evaluation prompt applied to model responses. Each judge received the original question text, the candidate response and the complete item-specific A-D rubric. Participant identity and group membership were not included in the judge prompt. Images were not separately provided to the scoring judges, consistent with the model-response scoring procedure. The rubric was treated as authoritative. Each judge assigned a score from 0 to 3, with half-point scores permitted under the predefined scoring rules. For participant $p$ and judge j, the judge-specific normalized score was

$$
S _ { p , j } = 1 0 0 \times \frac { \sum _ { i \in V _ { p , j } } s _ { p , i , j } } { 3 | V _ { p , j } | } ,
$$

where $s _ { p , i , j }$ is the score assigned to question i, and $V _ { p , j }$ is the set of questions with valid judge scores. The reported two-judge score was the arithmetic mean

$$
S _ { p } = \frac { S _ { p , \mathrm { G P T - 5 . 5 } } + S _ { p , \mathrm { G e m i n i } } } { 2 } .
$$

All final human-response files contained valid scores for all 100 questions from both judges; therefore, $| V _ { p , j } | = 1 0 0$ for every participant. Grouplevel results were calculated across five participants per training group and are reported as the mean and sample standard deviation.

Individual participant performance on the controlled 100-question organic-synthesis subset is reported below (Table 15). Each participant’s responses were independently scored by GPT-5.5 and Gemini 3.1 Pro using the same automated scoring procedure applied to model responses. The mean score for each participant was calculated from the two judge-specific normalized scores. Group-level variability was calculated as the sample standard deviation across the five participants in each group. The undergraduate group achieved a mean normalized score of 31.0% (sample SD = 2.7 percentage points), whereas the PhD group achieved a mean normalized score of 45.9% (sample SD = 3.0 percentage points).

## H Mismatched-Image Ablation

A mismatched-image ablation was performed to quantify model sensitivity to image-question alignment on the 100-question organic-synthesis subset (Table 16). The intervention disrupted the correspondence between the visual evidence and the question while retaining question identity, question text, text-encoded experimental context, answergeneration instructions, question-specific scoring rubric and evaluation pipeline. For each question in the matched-image subset, the original image or images were replaced with mismatched images sampled from other benchmark items within the same fine-grained subcategory. The target question’s original image or images were excluded from the replacement set. The number of images per question was kept unchanged, while the question text and all other inputs were preserved. The matched and mismatched datasets contained the same 100 unique question IDs in the same order. Direct question-level comparison confirmed that the category, question text, question-specific 0–3 rubric, reference answer and original subset metadata were identical; only the image array changed, with additional provenance fields added to the mismatched dataset. Before inference, literal image filenames in the question were replaced with neutral labels such as “Figure 1”. For mismatched questions, this sanitization used the original-image references rather than the distractor filenames, preventing the replacement filename from revealing the intervention.

<table><tr><td>Major category</td><td>Full benchmark N</td><td>Subset n</td><td>Fine-grained allocation Full N → subset n</td></tr><tr><td>Preparation</td><td>67</td><td>10</td><td>Experiment_plan: 46 → 7; Experiment_Safety: 21 → 3</td></tr><tr><td>Monitoring</td><td>30</td><td>5</td><td>Experiment_Monitor: 16 → 3; Experiment_notes: 14 → 2</td></tr><tr><td>Postprocessing</td><td>425</td><td>64</td><td>Extraction: 38 → 6; Evaporation: 46 → 7; Column: 75 → 11; TLC: 201 → 30; Recrystallization: 16 → 2; Trituration: 25 → 4; Lyophilization: 24 → 4</td></tr><tr><td>Characterization</td><td>142</td><td>21</td><td>NMR: 42 → 6; LCMS: 48 → 7; Meltingpoint: 8 → 1; Crystal: 29 → 5; GCMS: 15 → 2</td></tr><tr><td>Total</td><td>664</td><td>100</td><td>16 subcategories</td></tr></table>

Table 14: Composition of the controlled 100-question subset.
<table><tr><td>Participant ID</td><td>Group</td><td>GPT-5.5 judge</td><td>Gemini 3.1 Pro judge</td><td>Mean score</td><td>Sample SD</td></tr><tr><td>U01</td><td>Undergraduate</td><td>26.7%</td><td>28.2%</td><td>27.4%</td><td></td></tr><tr><td>U02</td><td>Undergraduate</td><td>29.0%</td><td>31.3%</td><td>30.2%</td><td></td></tr><tr><td>U03</td><td>Undergraduate</td><td>33.8%</td><td>35.0%</td><td>34.4%</td><td></td></tr><tr><td>U04</td><td>Undergraduate</td><td>29.8%</td><td>30.8%</td><td>30.3%</td><td></td></tr><tr><td>U05</td><td>Undergraduate</td><td>32.0%</td><td>33.3%</td><td>32.7%</td><td></td></tr><tr><td>Group mean</td><td>Undergraduate, n = 5</td><td>30.3%</td><td>31.7%</td><td>31.0%</td><td>2.7%</td></tr><tr><td>P01</td><td>PhD</td><td>45.3%</td><td>49.3%</td><td>47.3%</td><td></td></tr><tr><td>P02</td><td>PhD</td><td>44.7%</td><td>48.5%</td><td>46.6%</td><td></td></tr><tr><td>P03</td><td>PhD</td><td>39.5%</td><td>42.0%</td><td>40.8%</td><td></td></tr><tr><td>P04</td><td>PhD</td><td>46.2%</td><td>50.7%</td><td>48.4%</td><td></td></tr><tr><td>P05</td><td>PhD</td><td>44.7%</td><td>48.3%</td><td>46.5%</td><td></td></tr><tr><td>Group mean</td><td> $\mathrm { P h D } , \mathrm { n } = 5$ </td><td>44.1%</td><td>47.8%</td><td>45.9%</td><td>3.0%</td></tr></table>

Table 15: Individual human participant scores.

## H.1 Ablation results

For model (m), condition (c), and judge (j), the judge-specific normalized score was calculated over the 100-question subset as

$$
N S _ { j , m , c } = \frac { \sum _ { i = 1 } ^ { 1 0 0 } s _ { i , j , m , c } } { 3 \times 1 0 0 } \times 1 0 0 \% ,
$$

where $s _ { i , j , m , c }$ is the question-level score assigned by judge (j).

The normalized score was calculated separately for GPT-5.5 and Gemini 3.1 Pro. Each judgespecific normalized score was first rounded to one decimal place. The final score for each model under each image condition was then calculated as the arithmetic mean of the two rounded judge-specific normalized scores:

$$
S _ { m , c } = \frac { N S _ { \mathrm { G P T - 5 . 5 , m , c } } + N S _ { \mathrm { G e m i n i , 3 . 1 , P r o , } m , c } } { 2 } .
$$

Cross-model means were calculated as the arithmetic mean of the eight displayed final model scores under each condition and rounded to one decimal place. For each model, the score reduction was calculated as the difference between the original and perturbed scores and is reported in percentage points.

<table><tr><td>Model</td><td>Matched scores</td><td>Matched mean</td><td>Mismatched scores</td><td>Mismatched mean</td><td>Score drop</td></tr><tr><td>Claude Opus 4.7</td><td>43.2% / 44.2%</td><td>43.7%</td><td>25.0% / 20.3%</td><td>22.7%</td><td>21.0 pp</td></tr><tr><td>Gemini 3.1 Pro</td><td>37.8% / 40.2%</td><td>39.0%</td><td>17.7% / 16.7%</td><td>17.2%</td><td>21.8 pp</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>34.7% / 36.0%</td><td>35.4%</td><td>22.5% / 18.7%</td><td>20.6%</td><td>14.8 pp</td></tr><tr><td>GPT-5.5</td><td>41.7% / 45.2%</td><td>43.5%</td><td>21.0% / 18.0%</td><td>19.5%</td><td>24.0 pp</td></tr><tr><td>Grok 4.3</td><td>35.5% / 33.5%</td><td>34.5%</td><td>22.5% / 20.7%</td><td>21.6%</td><td>12.9 pp</td></tr><tr><td>Qwen3.6-Plus</td><td>38.2% / 39.7%</td><td>39.0%</td><td>20.5% / 18.3%</td><td>19.4%</td><td>19.6 pp</td></tr><tr><td>MiniMax-M3</td><td>37.5% / 38.0%</td><td>37.8%</td><td>21.0% / 18.5%</td><td>19.8%</td><td>18.0 pp</td></tr><tr><td>Intern-S1-Pro</td><td>27.5% / 26.3%</td><td>26.9%</td><td>13.5% / 14.2%</td><td>13.9%</td><td>13.0 pp</td></tr><tr><td>Cross-model mean</td><td>37.0% / 37.9%</td><td>37.5%</td><td>20.5% / 18.2%</td><td>19.3%</td><td>18.2 pp</td></tr></table>

Table 16: The cross-model score drop is the difference between the separately rounded condition means, rather than the mean of model-wise reductions.

![](images/966bc288ff63168634dad6128dfe5d41ccdf46392fef3723fe70b616f5db2b35.jpg)  
Figure 7: Model-balanced shares of assigned error labels across Claude Opus 4.7, GPT-5.5, and Gemini 3.1 Pro on the controlled organic-synthesis subset. Percentages sum to 100%.

$$
\Delta _ { m } = S _ { m , \mathrm { m a t c h e d } } - S _ { m , \mathrm { m i s m a t c h e d } } ,
$$

Values are displayed to one decimal place. Before final display rounding, the matched-image aggregate scores of Gemini 3.1 Pro and Qwen3.6- Plus were 39.00% and 38.95%, respectively. All eight matched and mismatched evaluation batches contained 100 model responses and 100 valid scores from each judge, with no remaining answergeneration errors, scoring errors or missing scores. Cross-model means were calculated separately for the matched- and mismatched-image conditions and rounded to one decimal place. The reported 18.2-percentage-point reduction is defined as the difference between these rounded means (37.5% - 19.3% = 18.2 percentage points). This differs slightly from the mean of the eight model-wise reductions (18.1 percentage points) because of the order of rounding and aggregation.

## I Expert Error Analysis

Two experts independently reviewed all non-fullscore responses from Claude Opus 4.7, GPT-5.5, and Gemini 3.1 Pro on the 100-question organicsynthesis subset, using the corresponding questions, visual evidence, reference answers, and automated scoring rationales. Each expert identified errors in evidence interpretation, causal reasoning, or experimental decisions. Their independent annotations were compared and consolidated into a shared six-category taxonomy, with the annotation rules finalized during this process.

Of the 266 non-full-score responses (83 from Claude Opus 4.7, 89 from GPT-5.5, and 94 from Gemini 3.1 Pro), 159 received at least one error label, yielding 257 label assignments. The taxonomy was non-exhaustive; responses without a clearly identifiable error mechanism remained unlabeled. Table 17 summarizes the error types and their raw counts.

<table><tr><td>Error type</td><td>Definition</td><td>GPT- 5.5</td><td>Gemini 3.1 Pro</td><td>Claude Opus 4.7</td></tr><tr><td>Evidence Misreading</td><td>Misreading or omission of decision-relevant experimental evidence.</td><td>38</td><td>31</td><td>34</td></tr><tr><td>Evidence-Decision Conflict</td><td>A diagnosis or action inconsistent with the experimental evidence or rubric.</td><td>33</td><td>29</td><td>30</td></tr><tr><td>Chemical Signal Errors</td><td>Incorrect interpretation of chemical signals, peaks, species, or quantitative results.</td><td>6</td><td>5</td><td>9</td></tr><tr><td>Common Rule Misuse</td><td>Applying generic laboratory rules in place of case-specific evidence.</td><td>9</td><td>6</td><td>4</td></tr><tr><td>Mistaking Normal for Abnormal</td><td>Treating a normal or transient state as abnormal and recommending unnecessary intervention.</td><td>3</td><td>5</td><td>4</td></tr><tr><td>Experimental Method Errors</td><td>Independent errors in experimental steps, conditions, methods, or execution.</td><td>3</td><td>5</td><td>3</td></tr><tr><td colspan="2">Total assignments</td><td>92</td><td>81</td><td>84</td></tr></table>

Table 17: Definitions and raw counts of expert-assigned error labels across the three models.

Annotation and aggregation rules. Evidence Misreading, Common Rule Misuse, and Evidence– Decision Conflict could co-occur when independently supported. The remaining three error types were mutually exclusive with one another and with these evidence-to-decision categories. Errors in experimental methods arising directly from an incorrect diagnosis were classified as Evidence-Decision Conflict, while chemical signal misassignments were classified as Chemical Signal Errors.

For each error type, its share of label assignments was calculated within each model and then averaged equally across the three models, rather than pooling all 257 assignments. Percentages were rounded to one decimal place using the largestremainder method to sum to 100.0%. These values describe the model-balanced composition of assigned error labels, not response-level error rates.