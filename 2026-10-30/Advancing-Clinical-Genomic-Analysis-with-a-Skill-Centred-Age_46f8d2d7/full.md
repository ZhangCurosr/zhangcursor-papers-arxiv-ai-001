# Advancing Clinical Genomic Analysis with a Skill-Centred Agent

Qiaoyu Zheng<sup>1,2</sup>, Weike Zhao<sup>1,2</sup>, Yanjie Fan<sup>3</sup>, Ye Cao<sup>4</sup>, Mengqi Jiang<sup>5</sup>, Xiao Zhou<sup>1</sup>, Xiaoman Zhang<sup>1</sup>, Pengcheng Qiu<sup>1</sup>, Yongguo Yu<sup>3</sup>, Yanfeng Wang<sup>1</sup>, Ya Zhang<sup>1,†</sup>, Chaoyi Wu<sup>1,†</sup> and Weidi Xie<sup>1,†</sup>

<sup>1</sup>Shanghai Jiao Tong University, Shanghai, China <sup>2</sup>Shanghai AI Laboratory, Shanghai, China

<sup>3</sup>Xinhua Hospital afiliated to Shanghai Jiao Tong University School of Medicine, Shanghai, China

<sup>4</sup>Department of Obstetrics and Gynaecology, Faculty of Medicine, The Chinese University of Hong Kong

<sup>5</sup>Department of Radiation Oncology, Shanghai Chest Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China

<sup>†</sup>Corresponding authors. Email addresses: {three-world, ya\_zhang, wtzxxxwcy02, weidi}@sjtu.edu.cn

Clinical genomic analysis requires experts to integrate evolving guidelines, databases, and computational evidence. As sequencing has become increasingly accessible, this complex and labor-intensive process has become a major bottleneck to the timely use of genomic information in clinical decision-making. To fill this gap, we developed ClinGeX, a large-language-model agent that composes 105 curated executable skills through task-specific retrieval and dependency-aware orchestration. On 223 temporally held-out benchmark cases, ClinGeX achieved 71.7% macro-averaged task-level concordance with expert reference classifications, compared with 45.7% for the strongest comparator. In a three-condition, counterbalanced human-AI collaboration study, 45 clinical genomic professionals assessed 135 clinically derived cases under routine (no AI assistance), DeepSeek-assisted and ClinGeX-assisted conditions. ClinGeX assistance increased raw case-level concordance across criterion-level variant analysis, genomic interpretation in rare disease and molecular oncology decision-making (51.1%, 57.8%, 64.4% versus 15.6%, 31.1%, 20.0% in comparator conditions) and reduced reported interpretation time in all scenarios. During an exploratory one-month real-world non-interventional assessment, 11 experienced professionals submitted feedback on 96 self-selected analysis episodes and reported frequent conclusion adoption and time savings. These findings support executable domain skills as a promising approach to augment specialist genomic interpretation.

## 1 INTRODUCTION

Genomic analysis has become an increasingly central component of modern precision medicine, particularly in rare disease and cancer [1, 2, 3]. Advances in high-throughput sequencing have enabled the systematic characterization of genetic variation at unprecedented scale, facilitating molecular diagnosis, disease classification, patient stratification, and the identification of targeted therapeutic opportunities [4, 5, 2, 3]. Yet, as sequencing has become increasingly accessible, the major bottleneck in genomic medicine has shifted from data generation to the interpretation of genetic variation and the derivation of clinically actionable insights [6, 7, 8, 9], a challenge famously framed as the “\$1,000 genome, \$100,000 analysis” [10].

This bottleneck arises from three challenges: (i) fragmented evidence: relevant knowledge is distributed across heterogeneous and often siloed sources, including variant databases, biomedical literature, clinical guidelines, and patient phenotypes; (ii) expertise-intensive reasoning: assessing variant significance demands substantial domain knowledge and contextual judgment involving disease mechanisms, inheritance patterns, phenotypic context, and evidence strength; and (iii) dynamic analytical workflows: diferent patients and questions entail adaptive combinations of variant annotation, phenotype matching, evidence retrieval, pathogenicity assessment, and result synthesis. As a result, although conventional bioinformatics tools can automate individual steps [11, 12], performing a comprehensive genomic analysis still relies heavily on experts to integrate disparate information and synthesize actionable conclusions, limiting its scalability.

Recently, large language model (LLM)-based agents have emerged as a promising paradigm for automating similar knowledge-intensive clinical tasks by combining reasoning with information retrieval and tool use [13, 14]. Bringing this paradigm to genomic analysis, however, is non-trivial. The field relies on a vast and continuously evolving ecosystem of professional guidelines, databases, and computational tools, each with distinct formats, interfaces, and usage protocols [15, 16]. Systematically translating these heterogeneous resources into standardized, agent-operable capabilities within a unified, scalable “agent harness” is therefore a central design problem [17]. Crucially, such a harness must preserve the flexibility of LLMs to dynamically compose workflows for diverse clinical questions, while ensuring that individual analyses follow established domainspecific procedures and standards.

![](images/f79c592282e2df5937870e924c2adf3c29899e795a49c2bbabdb497ea9ef463d.jpg)  
Figure 1 | Overview of ClinGeX and its evaluation. a. Task-oriented workflow. A clinical-genomics request and any accompanying text, image, PDF or other file types are submitted through the ClinGeX web application. The skill selector identifies the analytical needs, retrieves a task-specific subset of skills and their prerequisites, and passes them to the agent core for iterative planning, execution and evidence synthesis. b. Construction of the skill library. Candidate resources were identified through a broad domain scan, prioritised by LLM-assisted screening, reviewed by clinical and domain experts, and compiled and tested as 105 executable skills: 27 guide, 40 search and 38 analysis skills. Skills are connected by functional-relatedness and directed prerequisite edges. c. Controlled human-AI study. 45 readers evaluated 135 clinically derived cases across criterion-level variant analysis, genomic interpretation in rare disease and molecular oncology decision-making. d. Exploratory one-month deployment, where 11 clinical-genomics professionals submitted structured feedback on 96 completed analysis episodes.

Here, we present ClinGeX, a skill-centred agentic framework designed to support and accelerate clinical genomic analysis across heterogeneous inputs, including free text, pedigree images, clinical documents and sequencing data. ClinGeX comprises 105 specialized skills, each encoding an atomic operation in clinical genomics. These skills span five recurring categories: guideline following, entity mapping, data-source lookup, numerical calculation and model-based prediction. In the deployed system, they are grouped by operational role into three functional classes—guide, search and analysis. Together, they transform heterogeneous genomic resources into standardized capabilities that an LLM agent can combine for diferent clinical questions. Methodologically, we developed the skill library and system workflow through six stages: broad scanning of domain resources; LLM-assisted prioritization and expert selection; iterative skill development and testing; graph-based organization of related skills and their prerequisites; task-specific retrieval and progressive disclosure of skill instructions; and integration of multimodal inputs into an online interface. At inference, this organization enables the system to start from skills matched to a request, add their prerequisites and expose detailed instructions only when needed, limiting irrelevant context while preserving flexible workflows. This process converts established guidelines, databases and analytical tools into executable skills, while allowing ClinGeX to retrieve and compose only the capabilities required for each analysis (Fig. 1b).

To evaluate both the performance and clinical utility of ClinGeX, we conducted two independent studies. We assessed its autonomous analytical performance across 223 cases against OpenAI DeepResearch [18], representing a state-of-the-art commercial general-purpose research system, and Biomni [19], a leading opensource biomedical agent from academia. ClinGeX substantially outperformed both systems, achieving 71.7% concordance with expert consensus, 26.0 percentage points above the strongest comparator; moreover, 74.9% of its reports were judged directly adoptable without substantial revision by senior physicians, exceeding both comparators by more than 40 percentage points. More importantly, to determine whether these capabilities could translate into meaningful support for clinicians, we conducted a randomized multicentre study involving 45 clinical genomic professionals from three medical centres and 135 retrospectively collected real-world cases spanning criterion-level variant analysis, genomic interpretation in rare disease, and molecular oncology decision-making (Fig. 1c). DeepSeek<sup>1</sup> [20] was included as an accessible and representative example of widely used general-purpose LLM assistants. ClinGeX-assisted clinicians achieved significantly higher case-level concordance across all three scenarios, reaching 51.1%, 57.8%, and 64.4%, compared with 22.2%, 28.9%, and 31.1% for clinicians-only and 15.6%, 31.1%, and 20.0% with DeepSeek assistance. ClinGeX assistance also improved evidence identification and reduced the median time for criterion-level variant analysis from 60 to 15 minutes. Together, these findings support ClinGeX as a potential “Co-Clinician” [21, 22], augmenting rather than replacing expert genomic decision-making.

Finally, to preliminarily assess its utility in routine clinical-genomics work, we evaluated ClinGeX over a prespecified one-month period of routine use, during which 11 experienced clinical genomics professionals from multiple medical centers used the system for self-directed, naturally occurring tasks and provided structured feedback on 96 completed analyses (Fig. 1d). Experts reported fully or partially adopting ClinGeX’s conclusions in 94.8% of tasks, while time savings of more than 15 minutes were reported for 77.3% of dificult tasks. Together, these complementary evaluations support the potential of a skill-centred, domain-grounded agent to improve the accuracy, eficiency, and practical scalability of clinical genomic analysis.

## 2 Results

## 2.1 ClinGeX Overview

ClinGeX is an LLM-driven agent for clinical genomic analysis. Through a chatbot interface (Fig. 1a), users can submit free-text requests together with optional multimodal materials, including pedigree images, clinical records and other file types. The system returns a free-text reports accompanied by a concise summary of the analytical workflow, evidence sources and key intermediate outputs to support traceability.

At its core, ClinGeX uses a skill-centred architecture comprising 105 atomic clinical-genomics skills. For each request, a skill selector decomposes the task, retrieves relevant skills and expands the selection using semantic and prerequisite relationships encoded in a skill graph. The agent then executes the resulting task-specific workflow iteratively and synthesizes the accumulated evidence into its final response. Actions and tool outputs are retained in the execution record for review. Further details of the system architecture and development are provided in Methods Sec. 4.1.

We next evaluated ClinGeX across three settings of increasing clinical realism: automated system benchmarks, a controlled human–AI collaboration study, and prospective assessment in real-world clinical-genomics use. Together, these evaluations examined standalone performance, efects on clinician performance, and perceived utility in routine workflows.

## 2.2 System-level Analysis of ClinGeX on Automated Benchmarks

To evaluate ClinGeX at scale, we constructed a benchmark comprising 223 clinical-genomics cases released after the knowledge cutof of the underlying model of the comparators, collected from five expert-curated resources: ClinVar [23] for SNV interpretation (n=79), ClinGen [24, 25] for gene–disease validity assessment (n=66), and CIViC [26, 27], OncoKB [28, 29] and PharmGKB [30] for clinical-actionability classification (n=78). The primary metric was final-answer concordance, defined as agreement with all required components of the reference standard. Shared secondary outcomes included repeat consistency and execution time. Task-specific outcomes assessed pathogenicity discrimination and evidence recovery for SNV interpretation; classification severity and genetic and experimental evidence coverage for gene–disease validity; and complete-answer accuracy and classification severity for clinical actionability. Exact definitions, denominators and scoring procedures are provided in Supplementary Sec. A.4. To assess potential leakage from the reference databases, we audited execution records for direct or indirect retrieval of source information and confirmed that no evaluated execution retrieved such information (Supplementary Table 6).

In Extended Data Fig. 1 and Supplementary Secs. A.5 and A.6, we conducted ablation studies to assess the contribution of the ClinGeX architecture and its robustness across diferent agent cores. Here, we used the final ClinGeX configuration with DeepSeek-V4 as the agent core and compared it with the domainspecialized OpenBioLLM [31] and the latest agents, DeepResearch (OpenAI) [18], SciSpace [32], Biomni [19] and TxAgent [33]. General-purpose LLMs such as GPT and DeepSeek were not benchmarked separately because they underlie the evaluated agents. Configurations are detailed in Supplementary Sec. A.4.2.

## ClinGeX outperforms domain-specialized LLMs and agents across all three tasks.

ClinGeX achieved the highest final concordance in each task (Fig. 2a–c): 89.9% for SNV interpretation, 53.0% for gene–disease validity and 72.3% for clinical actionability. The corresponding best baseline results were 49.4% for DeepResearch, 35.5% for Biomni and 52.3% for TxAgent. Repeat consistency remained high across the three tasks (87.9–93.7%). Mean execution time was 175–197,s, broadly comparable to other agentic systems and shorter than SciSpace (496–727,s).

For SNV interpretation, ClinGeX achieved 100% sensitivity and 92.4% specificity for pathogenicity identification, with 82.3% weighted evidence recall, 84.8% evidence-strength concordance and a 1.3% significant-error rate. In gene–disease validity assessment, it reported usable genetic and experimental evidence scores for 90.9% of cases, compared with 28.8% and 15.2%, respectively, for the strongest comparator. For clinical-actionability classification, complete-answer accuracy was 57.7%, compared with 26.9% for TxAgent, while severe over- and under-classification remained infrequent.

## ClinGeX produces higher-quality and more clinically usable reports in blinded expert review.

We additionally compared the free-text reports produced by ClinGeX, Biomni and DeepResearch with human rating, jointly considering the correctness of key results and the soundness of the reasoning. Within each task, cases were partitioned among three domain experts (nine experts in total), each of whom rated the three shufled and de-identified system reports for their assigned cases. Reports were evaluated on three-level ordinal dimensions covering overall acceptability, required revision, citation accuracy, guideline adherence, evidence use and interpretation, fabricated content and potential clinical risk (Fig. 2d; Supplementary Sec. A.4.7).

The proportion of ClinGeX reports receiving the highest overall-acceptability rating was 88.6% for SNV interpretation, 66.7% for gene–disease validity and 67.9% for clinical actionability. The corresponding proportions were 32.9%, 40.9% and 23.1% for DeepResearch and 34.2%, 43.9% and 17.9% for Biomni. ClinGeX received higher ratings across all assessed dimensions in the first two tasks. In clinical-actionability classification, its clearest advantages were in overall acceptability, guideline adherence, evidence completeness and evidence interpretation, whereas citation accuracy and avoidance of fabrication were similar across systems. These findings suggest that the performance advantage of ClinGeX extends beyond final classification to the quality and usability of its supporting analysis.

a Comparison with other systems on SNV interpretation tasks (n=79)  
![](images/7ef4db37f933a6e364f59f6dbf88740559e1ed556eb653bad76fc210f484b7cb.jpg)

![](images/17145fb8b463030176bfcbe83322195a56e9f9c4b8ce826c08c1393728b1c902.jpg)

![](images/78363069ae449b33ce6a786542d8678e367949ecc0e4bb0224dcc7f057689ad3.jpg)

![](images/a42180704c33d552fd03bf3fe82f59a9be9b31811a3f2cfb84344b5168676fd5.jpg)

![](images/611bb2af330dd6a35e6822a6e2d6185a22c38fdbe3369fc57a8cb222c7817ce4.jpg)

![](images/43390580debb5a452c346de732524ee6e28dc7fb1da293f74b4656b7743ef235.jpg)

![](images/c20c53c07ea0028867ae2bc7a78558152710b55a9d4786e969667812de6d4af7.jpg)

![](images/b0c5d2ad7e02957840e08ff977d4df90b8d3441c1d1466554457120fec21ef93.jpg)

![](images/adf3b33ccb44783654f4d97cbe871239fc6e59657da67e09d89c47751aada022.jpg)

![](images/f4af8afdb9444c202eb596fa5d10dc2b1848a9020005efbfacfdfacc500ef146.jpg)

b Comparison with other systems on gene-disease validity assessment tasks (n=66)  
![](images/7f6fbfc25e0cb0e37e5f68b1cd39f14302628a2d171cdf9a1ec521504264770d.jpg)

![](images/92874a9a0f5aee89171d00375007b7fab918d0dc91f1dfe708e040bd3040c6fa.jpg)

![](images/d76376dd7f74f198810d66b33761cdb085669c999c829077ad0629139fe5ce7d.jpg)

![](images/c61ca73cb9a8fbc7d35ff0a4a0e4d4199e3e259e146f19de033416318c22fa43.jpg)

![](images/2f330fd4437b4850382efe6ba832aab057a9563181a3ec9c0774f795138147aa.jpg)

![](images/190afdb44f6306fece9c4ca89c0ba8a1c6f39af06d063af37fd6d3021a8d8553.jpg)

![](images/1f716a0b0f3fcf15a408a6dc767e816f215ae8fd6ba85e73faa50e395d6e2f58.jpg)

![](images/b032af111870d95a7b1ffe7b101bffabc8186450accaa9ac128ab3f7e29b480c.jpg)

![](images/e9e9118c5b538d6e94399505cdb14001cd2d5fe665e2adf12dde640908f1723d.jpg)

![](images/34115c4022c09d75847ce939d93ef3793dd4e518440bff1f71b2519c5d3a8ed2.jpg)

c Comparison with other systems on clinical-actionability classification tasks (n=78)  
![](images/bfcfd8f74dbcd0f9d3750feec5d38ae9b0e2fb22f93ff0c14e332037480026af.jpg)

![](images/d203f6389f937111303cb9be1fb2badbf08efa6ad504cdbdb0295b55ee13cbc4.jpg)

![](images/58d3b1bab26e02393e142c9d107f7098e6f3776d7f619aef776203d69182b56e.jpg)

![](images/3de1c98b1f7190e94b4c42b245aabbaedf8e97e983e797b5c690684cc16446bd.jpg)

![](images/d0ac8a7488bb2ebb9f79edcae970ef3cc79e50704eb1fd2a99073987dc8a85bd.jpg)

![](images/61e10d1d89268b588f629472eee7ae8e24bbe45b4b2c8f5db91f4ba6b38efa08.jpg)

![](images/f0e8c5a96eff9f1264bbaa16b038d2274edcb3bdf8acee9be4cd8994ae37e748.jpg)

![](images/b54ee00d97b33253381915d88b8d1cdfbf33137215ff50e804fb77a78fe77f46.jpg)

d Experts review on response from ClinGeX and tow other systems (n=223)  
![](images/c6d7596ae15047fa48abc5709b4501cfc3ed392e13f7bafde7baf0fa24ad10e4.jpg)

![](images/27c730104ced2409111b0a4519eda13a008449e477cb0a02a0dc8fe9ba0ca032.jpg)

![](images/3ee54b04ebb97902088acc654bb21a6b710e69e3a2d31fd7b31a3127bfd80b4d.jpg)

![](images/c1c6c71c7d138b82671fa639b014ee8f66ab2c002ed7538909a115475193e11f.jpg)  
Figure 2 | System-level comparison of ClinGeX with biomedical and general research systems. a–c. Performance of OpenBioLLM, OpenAI DeepResearch, SciSpace, Biomni, TxAgent and ClinGeX on SNV interpretation $( \mathrm { a } ; \mathrm { n } = 7 9 )$ , gene-disease validity assessment $( \mathrm { b } ; \mathrm { n } = 6 6 )$ and clinical-actionability classification $( \mathrm { c } ; \mathrm { n } = 7 8 )$ . Final concordance, repeat consistency and execution time are shown for all tasks; other measures are task specific. Arrows indicate the favourable direction and error bars denote 95% confidence intervals. d. Blinded assessment of the 669 reports generated by ClinGeX, Biomni and DeepResearch for the 223 benchmark cases. For each case, the three reports were de-identified, randomly labelled and rated on three-level ordinal dimensions by one of nine domain experts; each reviewer assessed all three systems for their assigned cases. Colours denote the three rating categories.

## ClinGeX can ground its classifications in guideline-consistent analysis.

To more directly illustrate its capability, we selected three cases in which ClinGeX matched the reference classification while Biomni and DeepResearch did not (Fig. 3). In a SNV interpretation case, ClinGeX weighted two complementary functional studies against a discordant computational prediction without over-crediting it, reaching Likely Pathogenic where both comparators reported Pathogenic. In another gene-disease validity case, it partitioned genetic and experimental support into explicit, separately scored criteria, supporting a Limited classification where Biomni and DeepResearch reported Moderate and Definitive, respectively. In a clinical actionability case, it distinguished short-term glucose-response from longer-term HbA1c endpoints and weighed replication across overlapping cohorts, reaching Level 3 where the comparators reported Level 2A and Level 2B. In each case, the figure traces the classification to specific, guideline-grounded analytical steps that the comparators omitted, together illustrating ClinGeX’s capacity for integrative synthesis of heterogeneous evidence and for discriminating among evidentiary sources of difering strength.

![](images/75e9d20294be214b702a3bcf4313e82e5842f802dd0fd9bdf5aa25e8fc82738a.jpg)  
Figure 3 | Selected analytical contributions of ClinGeX across three benchmark tasks. a, Evidence-strength assignment for PIK3CD p.Gly124Asp, integrating clinical and mechanistic findings with discordant computational predictions. b, Explicit partitioning of genetic and experimental evidence for the SLC1A1 –dicarboxylic aminoaciduria relationship. c, Consideration of study dependence, response endpoints and clinical applicability for the GLP1R rs6923761–sitagliptin/vildagliptin association. The centre presents selected reasoning steps condensed from the recorded responses; the right compares the final classifications reported by ClinGeX, Biomni and OpenAI DeepResearch. References are the benchmark classifications. Evidence-point totals summarize the recorded responses.

## 2.3 Evaluation of ClinGeX in the MRMC Human-AI Collaboration Study

We next conducted a three-centre, multi-reader multi-case (MRMC) study to assess whether ClinGeX could improve human performance in three clinical-genomics scenarios: criterion-level variant analysis, genomic interpretation in rare disease and molecular oncology decision-making. Each scenario comprised 45 cases, yielding 135 cases in total. Fifteen clinical genomic professionals with 3–10 years of relevant experience participated in each scenario (45 participants overall), with one scenario evaluated at each of three centres: Xinhua Hospital; the Department of Obstetrics and Gynaecology, Faculty of Medicine, CUHK; and Shangha Chest Hospital. Participants completed cases under three conditions: human-only, DeepSeek-aided and ClinGeX-aided. DeepSeek was selected as an accessible and widely used general-purpose LLM comparator.

A Latin-square-inspired counterbalanced design minimized learning and order efects [34] (Extended Data Fig. 2). Each participant completed nine cases—three per condition—without encountering the same case twice. Each case was evaluated under all three conditions by three diferent participants, and condition order was randomized across participants. Further details are provided in Methods Sec. 4.2.

## ClinGeX assistance is associated with higher case-level concordance.

We first assessed whether ClinGeX could improve participants’ final-answer correctness. We used case-level concordance as the primary metric, defined as the proportion of assessments for which the participant’s final conclusion was adjudicated by the expert panel as substantially correct (Supplementary Sec. A.7.2). ClinGeX assistance produced the highest concordance in all three scenarios (Fig. 4a). For criterion-level variant analysis, observed concordance was 51.1%, compared with 22.2% under human-only assessment and 15.6% with DeepSeek assistance; the corresponding GLMM-standardized diferences were 29.9 percentage points (Holmadjusted $P = 0 . 0 1 3 )$ and 36.8 points $( P = 0 . 0 0 9 )$ , respectively. For genomic interpretation in rare disease, observed concordance was $5 7 . 8 \% .$ , compared with 28.9% and 31.1%, respectively; the GLMM-standardized diferences were 29.6 points $( P = 0 . 0 1 7 )$ and 27.3 points $( P = 0 . 0 1 9 )$ . For molecular oncology decision-making, observed concordance was 64.4%, compared with 31.1% and 20.0%, respectively; the GLMM-standardized diferences were 33.4 points $( P = 0 . 0 0 2 )$ and 44.5 points $( P < 0 . 0 0 1 )$ ). Thus, under the current analysis, ClinGeX assistance approximately doubled concordance relative to routine and DeepSeek-aided assessment.

## ClinGeX assistance improves key components of the supporting analysis.

We next evaluated task-specific intermediate outcomes, including evidence identification and weighting, phenotype interpretation and clinical-actionability assessment. Specifically, criterion-level variant analysis was evaluated using ACMG/AMP evidence recall, unsupported-evidence assignment, evidence-strength concordance, final-class accuracy and directional classification errors; genomic interpretation in rare disease using variant-classification and phenotype-interpretation accuracy and directional errors; and molecular oncology decision-making using panel-adjudicated free-response accuracy and the accuracy and directional errors of clinical-appropriateness judgements (Supplementary Sec. A.7.3).

In criterion-level variant analysis (Fig. 4b), ClinGeX increased model-standardized ACMG/AMP evidence recall by 28.6 (percentage) points relative to human-only assessment and by 27.9 points relative to DeepSeek assistance (both $P < 0 . 0 0 1 )$ . It reduced unsupported evidence assignments by 10.4 $( P = 0 . 0 0 9 )$ and 15.3 points $( P < 0 . 0 0 1 )$ , respectively, and improved evidence-strength concordance by 9.9 $( P = 0 . 0 1 5 )$ and 20.2 points $\left( P < 0 . 0 0 1 \right)$ . Final pathogenicity-classification accuracy was 23.1 points higher than with DeepSeek assistance $( P = 0 . 0 4 0 )$ , and over-classification was reduced by 9.2 and 11.1 points relative to human-only and DeepSeek-aided assessment, respectively (both $P = 0 . 0 3 4 )$ ; the remaining directional contrasts were not statistically significant.

In genomic interpretation in rare disease (Fig. 4c), model-standardized phenotype-interpretation accuracy increased by 23.7 (percentage) points relative to human-only assessment $( P < 0 . 0 0 1 )$ , accompanied by a 12.8-point reduction in under-interpretation $( P = 0 . 0 0 6 )$ . Accuracy was 8.8 points higher than with DeepSeek assistance, although this diference was not statistically significant $( P = 0 . 0 7 1 )$ ). Improvements in final classification were also numerical rather than statistically significant.

In molecular oncology decision-making $\left( \mathrm { F i g . ~ 4 d } \right)$ , model-standardized free-response accuracy increased by 38.3 percentage points relative to DeepSeek assistance $( P = 0 . 0 0 1 )$ and by 22.2 points relative to human-only assessment $\left( P = 0 . 0 3 3 \right)$ . Accuracy in judging the appropriateness of clinical decisions increased by 18.5

Molecular oncology decision-making

a

Criterion-level variant analysis

![](images/12cbf9e3eb3c53cc7d7d2ed86cc767498eadaa4a04b5da6d7e17aae22aa7bacb.jpg)

![](images/0486a8a05a6bf7801410052a7e541db426c264dd8a11cedbdfddce8ecb9ade33.jpg)

![](images/36cc01a041dc784e5ffc4a2ab8ee9c912491148f1e27a1be3f934b22ed972401.jpg)

![](images/5567de343028b92470d8eb058e2880c4fa07a553f003728a44ca80dc4a8e3183.jpg)

b  
![](images/7d9890979ef13efac3b5bb44f0332f35b8ffed1c8e9eaff6b6116cc3dedda11d.jpg)

c  
![](images/5113b58091c11b650a6928f68754712add4397b769cf6afdc90851b6787e5afc.jpg)

d  
![](images/161e4b4e964fa8b521ef0a888a47e6c54c008eee3aacd3abab61f2e7d2b52bb8.jpg)

![](images/110117facda7f81395698ecb2e54225cc1a456650982fad2d643fa741c039312.jpg)  
Figure 4 | Clinical genomic professional performance under unassisted, DeepSeek-assisted and ClinGeX-assisted conditions. a. Case-level concordance with the expert reference standard for criterion-level variant analysis, genomic interpretation in rare disease and molecular oncology decision-making. Each condition comprised 45 assessments completed by 15 participants; fractions indicate observed concordant assessments over the total, whereas bars and error bars show model standardized probabilities and two-sided 95% Wald confidence intervals. Pairwise P values are Wald contrasts from crossed random-intercept logistic generalized linear mixed models (GLMMs), with condition as a fixed efect and reader and case as random intercepts, Holm-corrected across the three pairwise comparisons within each scenario. b–d. Task-specific secondary outcomes for criterion-level variant analysis (b), genomic interpretation in rare disease (c) and molecular oncology decision-making (d). Left, model-standardized favourable probabilities under each condition; right, corresponding absolute between-condition diferences in percentage points with two-sided 95% Wald confidence intervals. For outcomes where lower values indicate better performance, the left panels show one minus the model-standardized event probability and contrast directions are reversed.

and 20.1 points relative to the human-only and DeepSeek-aided conditions, respectively (both $P < 0 . 0 0 1 )$ ; overestimation of appropriateness decreased by 15.1 $( P = 0 . 0 0 1 )$ and 16.8 points $( P < 0 . 0 0 1 )$ , respectively. These findings suggest that the observed gains extended to several intermediate components of evidence assessment and clinical interpretation.

## Confidence–correctness alignment is preserved with ClinGeX assistance.

In Fig. 5a, we examined whether improved concordance came at the cost of poorer uncertainty calibration. Certainty in criterion-level variant analysis was derived from whether participants made an explicit call for every applicable ACMG/AMP evidence item; the other two scenarios used self-reported confidence ratings. Metric definitions are provided in Supplementary Sec. A.7.4.

For criterion-level variant analysis, an assessment was counted as fully certain when the participant assigned an explicit call to every applicable ACMG/AMP evidence item, and as partially certain otherwise; accuracy within a group is the proportion of its assessments adjudicated concordant. Of the 45 assessments per condition, 9 were fully certain under ClinGeX assistance and 8 of these were correct (88.9%), against 21 of the remaining 36 (58.3%); the corresponding figures were 7 of 10 (70.0%) under DeepSeek assistance and 3 of 7 (42.9%) in the human-only condition. Certainty was thus most predictive of correctness under ClinGeX assistance, although the small number of fully certain assessments limits formal comparison.

For genomic interpretation in rare disease and molecular oncology decision-making, confidence–correctness alignment was assessed using point-biserial correlation, ROC–AUC and the Brier score. ClinGeX produced the highest correlation and lowest Brier score in both scenarios ((r=0.343), Brier score 0.246 for genomic interpretation in rare disease; (r=0.212), Brier score 0.161 for molecular oncology decision-making). It also produced the highest AUC in molecular oncology decision-making (0.652); its AUC in genomic interpretation in rare disease was 0.655, marginally below that of DeepSeek (0.671). Overall, the increase in concordance with ClinGeX was not accompanied by poorer confidence–correctness alignment.

## Participants report a favorable experience and shorter completion times with ClinGeX.

Inspired by prior clinical AI evaluations [35], we also evaluated the system from readers’ perspective. Specifically, after completing the test, participants rated their experience with DeepSeek and with ClinGeX on an anonymous five-point Likert-type items covering confidence in the answers provided, willingness to use the system again, satisfaction with the presentation of results, smoothness of operation, and reliance on system output (Fig. 5b); ratings were summarized separately for each scenario, by the 15 participants who had completed it in the MRMC study (n = 15 per scenario, 45 in total). More details are in Methods Sec. 4.2.6. ClinGeX received higher mean ratings than DeepSeek on most dimensions in all three scenarios, the only exceptions being smoothness in genomic interpretation in rare disease and molecular oncology decision-making (Fig. 5c).

We next compared self-reported mean time per question across the human-only, DeepSeek-aided and ClinGeXaided conditions (Fig. 5d). Relative to human-only, ClinGeX shortened the median time per question from 60 to 15 min for criterion-level variant analysis and from 30 to 15 min for genomic interpretation in rare disease; both diferences were statistically significant (P = 0.003 for both). For molecular oncology decision-making, the median time decreased from 15 to 11 min, but the diference did not reach statistical significance (P = 0.078). This smaller efect likely reflects the shorter human-only baseline in this task, together with the fact that participant-estimated times included system response latency.

As an exploratory analysis of spontaneous continued use, we recorded new conversations initiated in the two weeks following test completion, without any instruction or inducement for the 45 participants to continue using ClinGeX. Twenty-nine participants initiated at least one further conversation beyond the three study-related sessions, generating 239 new conversations in total, with a mean of 8.2 additional conversations per continuing user (Fig. 5e). These observations provide preliminary evidence of the system’s acceptability and voluntary reuse beyond the controlled study setting.

a  
Expressed certainty vs correctness: criterion-level variant analysis  
![](images/5fbcd26be3843bebf42c545a55cac7620656510dd7e0b5581d15b81f734b00a5.jpg)  
Genomic interpretation in rare disease

b  
![](images/40d4337816e79ee2b6793f2ebdc78ac604db11b928c631c184a66d0bdf7618c3.jpg)

![](images/2dff3db29efe7a641c28eb36e00871d242765111b18553adc8e05a85ed8377a0.jpg)

0 Anonymous questionnaire (1-5 scores each point)  
![](images/a94033c1ae1fd9a9cf7cf9fcf2b453d4461b01648a6362e8a21d4a1db6aa7f90.jpg)

c  
Analysis on feedback of participants  
![](images/f30d8912c21027735eeb2f1b76c512d5c249c94cb707a086f6db5f1eaef08d07.jpg)

![](images/ffe4fdeb8ee356cdec82bc06ef5aeb63f4d3070906980d15798ff14c630d0e34.jpg)

d  
![](images/71dd2e3c72c48987ba20bf602fbf7457a01c7437e5edbd58e7e1eb3dea2fbbbd.jpg)

Analysis on average time usage of participants  
![](images/00ae9483202a9cab05f471604e6f26e005ee352a3b706431a6a32b3622d7f55d.jpg)

e  
![](images/84ab1b776eb2cbffb4e69748a499b7f581ce899d1e65fffe98dcce553000c790.jpg)

![](images/f3535d9564473cb553529239cc31a8c004d441420355552dd137ff50be1d66fa.jpg)  
Figure 5 | Confidence–correctness alignment, user-reported experience and continued use in the MRMC study. a. Top, final-classification accuracy in criterion-level variant analysis for assessments in which all $\mathrm { A C M G / A M P }$ criteria received an explicit call versus assessments with at least one criterion left undetermined; fractions indicate correct assessments over the total and error bars denote 95% Wilson confidence intervals. Bottom, association between self-reported confidence and correctness for genomic interpretation in rare disease and molecular oncology decision-making, summarised by the point-biserial correlation (r), ROC-AUC and Brier score. b. Design of the 13-item post-study questionnaire administered to the three participant groups (n = 15 per group). c. Mean ± s.d. ratings for five experience dimensions under DeepSeek and ClinGeX assistance. d. Participant-reported mean time per case; dots denote participants, numbers denote medians and P values are from paired two-sided Wilcoxon signed-rank tests with Holm correction. e. New ClinGeX conversations initiated during the 14 days after completion of the assigned tasks, excluding the three study-related sessions.

## 2.4 Prospective Assessment of ClinGeX on Real World Feedbacks

Controlled evaluations establish performance on standardized cases, but do not capture usability across heterogeneous, self-directed clinical-genomics tasks. We therefore conducted an exploratory, prospective taskbased assessment in which experienced professionals submitted self-selected gene- and variant-interpretation tasks over a prespecified one-month period. “Prospective” refers to the forward collection of these evaluation tasks and participant feedback. This was a non-interventional, research-only evaluation, and ClinGeX outputs were not permitted to influence clinical reports, patient management or any other clinical decision.

Eleven senior clinical genomic professionals from five medical centres used ClinGeX in routine practice during a one-month period. The analysis here focused on clinicians’ episode-level feedback. An analysis episode was defined as one complete clinical-genomics objective—for example, interpreting a patient’s genomic sequencing findings or determining whether newly published evidence supported a specific variant- or gene– disease conclusion. An episode could comprise multiple dialogue turns, but its scope and analytical workflow were otherwise unrestricted. After completing an episode, the expert selected the corresponding conversation history and submitted one structured feedback record evaluating ClinGeX’s contribution to the episode as a whole, rather than to an individual prompt or turn. Thus, one feedback submission represented one completed episode; 96 submissions were collected. These records were then used to examine when and how ClinGeX helped in routine practice. For each episode, experts rated its dificulty (simple, moderate or dificult), indicated whether they adopted ClinGeX’s conclusion (fully, partially or not) and, where applicable, how many recommended evidence criteria they adopted, reported the time saved, and rated the help provided in information standardization, numerical analysis, literature analysis, and analytical or procedural guidance from little or no help to very helpful (Fig. 6a). Full definitions of these measures are in Methods Sec. 4.3.

## Experts frequently rate ClinGeX conclusions as adoptable and report time savings.

We analysed the reported contribution of ClinGeX at two direct, episode-level endpoints: the extent to which its conclusion was adopted in the clinician’s final assessment and whether its use accelerated completion of the analysis. Across the 96 feedback records, experts rated the conclusion as fully adoptable in 70 episodes (72.9%), partially adopting it in 21 episodes (21.9%), and not adopting it in 5 episodes (5.2%) (Fig. 6b). They reported saving more than 15 min in 40 episodes (41.7%) and 5–15 min in another 50 episodes (52.1%); only 6 tasks (6.3%) took the same amount of time or longer. Full adoption was reported for 4/5 (80.0%) simple, 51/69 (73.9%) moderate and 15/22 (68.2%) dificult episodes. The corresponding proportions for which more than 15 min was saved were 0/5, 23/69 (33.3%) and 17/22 (77.3%), respectively, showing that the reported time benefit was concentrated in more dificult analyses while most final conclusions were fully adopted.

## Experts adopt most of the evidence items recommended by ClinGeX.

As a more direct measure of assistance in routine variant interpretation, we identified 37 of the 96 feedback records that concerned variant-interpretation episodes for which ClinGeX recommended discrete ACMG/AMP evidence codes and the senior expert recorded how many were ultimately adopted. For each episode, we calculated the ratio of adopted to recommended codes, capped at 1. The mean adoption ratio was 0.83 for simple episodes (n = 2), 0.87 for moderate episodes (n = 30), and 0.80 for dificult episodes (n = 5) (Fig. 6c). Although the small numbers of simple and dificult episodes preclude comparisons between dificulty levels, these results show that experts retained a large proportion of the ACMG/AMP evidence codes proposed by ClinGeX in their final variant interpretations.

## ClinGeX helps experts across multiple dimensions in analysis workflow.

Because ClinGeX was designed around five recurring operations in clinical genomics—guideline following, entity mapping, data-source lookup, numerical calculation and model-based prediction—we next examined the specific aspects of analysis in which clinicians found the system helpful. Experts separately rated its contribution to information standardization, numerical analysis, literature analysis, and analytical or procedural guidance. These four clinician-facing dimensions captured overlapping manifestations of the five skill categories: information standardization primarily reflected entity mapping and structured data retrieval; numerical analysis reflected numerical calculation and model-based prediction; literature analysis reflected data-source lookup; and analytical or procedural guidance reflected guideline following and cross-skill orchestration. Across the 96 episodes, mean helpfulness scores on a 0–2 scale were similar across the four

Evidence-criterion adoption ratio (n=37/96 episodes with numeric)

![](images/cc9cf3d1a98a3cc3b063896382eea926acd7ba003b1f94af432ed47ca07fa902.jpg)

b  
![](images/09b761795ae9ba15f40cece3fb435174df79e25092453450d32db5b10ef97afb.jpg)

![](images/c42aa6d89f811e8604baf66c8e482ff39c190acfdd0ab6d0acf443cf7609e49a.jpg)

c  
![](images/f2b6376868e2872b3d5fb90f468038bbcdb50f30f27299dad2e95efe750d3c97.jpg)

d  
![](images/6c69919149dac6d0411cd54f019835e553235643259c864b1351caad0fd5c72d.jpg)

![](images/1dff69f5d11da5663e076708e18c9e1c2e7aba79fcdffd70864f482a5ca3e5bc.jpg)

![](images/940998c197c29e44f279ba4da2fb288ab6bd65736074e8cdbef133ac2098599e.jpg)  
Figure 6 | Exploratory deployment of ClinGeX in routine clinical-genomics work. a. Eleven experienced clinicalgenomics professionals used ClinGeX for one month and submitted structured feedback on 96 completed analysis episodes, recording task dificulty, conclusion adoption, evidence-criterion adoption where applicable, reported time saved and helpfulness across four workflow dimensions. b, Conclusion adoption and reported time saved, stratified by task dificulty. Stacked bars show all outcome categories; points and intervals show the proportion in the highest category (full adoption or >15 min saved) with 95% confidence intervals. $\mathbf { c } ,$ Ratio of adopted to recommended evidence criteria among 37 episodes with countable recommendations; ratios were capped at 1. Grey points denote episodes and orange points and bars denote bootstrap means and 95% confidence intervals. d, Mean helpfulness scores and rating distributions for information standardisation, numerical analysis, literature analysis and analytical or procedural guidance. Helpfulness was rated from 0 (little or no help) to 2 (very helpful).

dimensions, at 1.29, 1.35, 1.33 and 1.34, respectively (Fig. 6d). Some or substantial help was reported for 90 episodes (93.8%) in information standardization, 91 (94.8%) in numerical analysis, 89 (92.7%) in literature analysis, and 92 (95.8%) in analytical or procedural guidance. Thus, clinicians reported that ClinGeX supported multiple components of genomic interpretation rather than a single analytical function.

Case studies illustrate how ClinGeX contributes to clinical genomics analyses.

Two cases from the prospective deployment illustrate how ClinGeX helped clinicians address ambiguities with consequences for evidence use and clinical interpretation (Fig. 7). In an assessment of LRP6 c.1831C>T (p.Arg611Cys), reported as moderately dificult by the submitting clinician, a ClinVar submission cited a family carrying p.Tyr418His as evidence for the assessed variant. ClinGeX identified this mismatch, helping the clinician avoid attributing that family’s evidence to p.Arg611Cys. The clinician reported full adoption of the conclusion, adoption of four of five recommended evidence criteria, and 5–15 min saved, and explicitly noted that the analysis helped avoid incorrect use of family evidence. In a second case, reported as dificult by the submitting clinician, TP53 c.391A>T (p.Asn131Tyr) was detected at a variant allele fraction of 8–18% in blood from an old man with metastatic prostate cancer. The key question was whether this finding supported a hereditary cancer-risk interpretation. ClinGeX distinguished possible origins, including clonal haematopoiesis, tumour-derived DNA and constitutional mosaicism, separated germline and somatic interpretation frameworks, and proposed confirmation of sample type and additional tissue testing before assigning hereditary risk. The clinician reported full adoption of the conclusion and more than 15 min saved, highlighting the separate germline and somatic scenarios as particularly helpful.

![](images/948b81cfc3c8ebe81543b8d6f21f1730ab7e652ec59be9ca85663979b2184068.jpg)

Figure 7 | Illustration and analysis of two feedback cases. a, Evidence attribution for LRP6 p.Arg611Cys. ClinGeX identified that a family cited in a ClinVar submission carried p.Tyr418His, helping the clinician avoid using misattributed family evidence. The cited paper also contained p.Arg611Cys functional data; the mismatch concerned the family evidence. $\mathbf { b } ,$ Interpretation of a low-allele-fraction TP53 p.Asn131Tyr finding in blood from an older patient with metastatic prostate cancer. ClinGeX helped distinguish possible variant origins, separate germline and somatic interpretations, and identify the confirmation needed before assigning hereditary risk; the origin remained unconfirmed in the available record. For each episode, the left and centre show the clinical context, key judgement and contribution to the clinician’s interpretation. The right-hand column reports the submitting clinician’s dificulty rating, conclusion adoption, time saved, evidence-criterion adoption where reported, helpfulness ratings and condensed feedback. Helpfulness was rated from 0 (little or no help) to 2 (very helpful).

## 3 DISCUSSION

Genomic analysis increasingly informs multiple stages of clinical care, from understanding disease mechanisms and establishing molecular diagnoses to guiding targeted interventions and population-level screening. Its broader adoption, however, is constrained by the specialized expertise required, uneven access to specialist support across centres and the dificulty of translating rapidly evolving knowledge into clinical practice. Contemporary artificial intelligence, particularly large language model (LLM)-based agentic systems, has shown promise in supporting clinical decision-making across several medical settings. Realizing this potential in clinical genomics requires systems that can reliably integrate fragmented and continuously evolving guidelines, massive databases and computational tools. Yet most evaluations remain benchmark- or simulation-based, leaving limited evidence of their efects on clinician performance and usability in workflow-relevant settings.

The central contribution of this work is ClinGeX, a skill-centred agent that adapts general-purpose LLMs into clinically grounded assistants for genomic analysis. Its technical value lies in three respects. First, systematic resource collection and expert curation help reduce the omissions and selection biases associated with ad hoc resource enumeration, and may ofer a template for analogous skill discovery in other clinical domains. Second, semi-automated skill construction and graph-based organization allow the skill library to expand without sacrificing eficient task-specific retrieval. Third, progressive disclosure restricts the active context to what each analytical step requires, helping constrain hallucination. The value of these design choices is supported by the automated benchmark, in which the resulting system outperformed both alternative enhancement strategies and SOTA general-purpose and biomedical agents.

Across the automated benchmarks, human–AI collaboration study, and prospective deployment, we demonstrate the potential practical value of ClinGeX in five complementary aspects. First, it constrains the hallucination that current LLM-based systems cannot reliably avoid, grounding conclusions in executed skills and the evidence they return; the risk this addresses is visible in the DeepSeek-aided arm, where concordance fell below that of unassisted clinicians in two of the three scenarios (15.6% versus 22.2%; 20.0% versus 31.1%). Second, it improves the quality of gene-related clinical analysis and decision-making by junior and middle-grade clinicians, raising case-level concordance with expert consensus from 22.2%, 28.9% and 31.1% to 51.1%, 57.8% and 64.4% across the three scenarios—a meaningful step towards democratizing access to expert-level interpretation. Third, it lowers interpretive burden, shortening the median time per case from 60 to 15 min for criterion-level variant analysis and from 30 to 15 min for genomic interpretation in rare disease, while presenting the complete interpretive path so that clinicians can inspect it and decide whether to adopt or override the system’s conclusion. Fourth, it showed broad workflow-level utility for experienced clinical-genomics professionals during self-directed routine use, as reflected by reported benefits spanning information retrieval and standardization, numerical and literature analysis, and procedural guidance. Fifth, the system can be deployed locally and adapts to a range of open- and closed-source backbone models, which accommodates the difering infrastructural constraints and access policies of individual medical centres.

## Limitations

First of all, although ClinGeX targets clinical genomic analysis broadly, the MRMC study covered only three scenarios—criterion-level variant analysis, genomic interpretation in rare disease and molecular oncology decision-making—selected based on the participating centres’ specialties and routine caseloads. Although the skill library was not tuned to these scenarios, their evaluation does not establish generalizability to other areas of clinical genomic interpretation, such as polygenic risk assessment.

Secondly, the MRMC study was powered to detect the main efect of assistance; with 45 readers and 135 cases, several secondary comparisons showed consistent directions but did not all reach statistical significance. The primary endpoint is a proxy: concordance with expert consensus reflects interpretive quality rather than patient outcomes. Although fit for the present aim of assessing interpretive performance, establishing that improved interpretation translates into changes in clinical management and supports broader clinical use will require a prospective randomized trial with prespecified clinical endpoints.

Finally, the prospective deployment focused on workflow-level outcomes—including conclusion adoption, time savings and perceived helpfulness—rather than prespecified clinical endpoints such as diagnostic yield, changes in management or patient outcomes. It therefore provides evidence of practical utility in routine analysis, while the downstream clinical impact of ClinGeX requires evaluation in larger prospective studies.

## 4 METHODS

## 4.1 Development of ClinGeX

ClinGeX adopts a skill-centred agentic architecture. An agent ‘skill’ [36] is a reusable instruction module, written in structured free text, that encapsulates a task-specific agentic workflow for a particular clinical or research task. Each skill specifies when it should be invoked and provides the procedural knowledge required to complete the task, including what information to seek, which standards and resources to consult, which tools to call, and how to carry out the analysis step by step (a detailed description on skills can be found in Supplementary Sec. A.2.1). By defining these capabilities at an atomic level, a broad and systematicall curated skill library allows the agent to flexibly select, combine, and sequence individual skills to construct task-specific workflows, thereby supporting a wide range of clinical genomic analysis needs.

To build the skill library and integrate it into an LLM-based agent, we developed ClinGeX through a six-stage design framework. Stages 1–4 focused on constructing the skill library by identifying, curating, and formalizing domain knowledge and analytical resources into atomic agent skills. Stage 5 integrated these skills into the agent workflow through inference-time skill retrieval and orchestration, while Stage 6 focused on user interface and interaction design.

## Stage 1. Web-scale scan of domain resources.

To systematically build the skill library, we first conducted a web-scale scan to identify and collect a broad range of clinical-genomics resources for subsequent skill construction. Specifically, we organized the scan into five broad categories of clinical-genomic analysis operations: guideline following, entity mapping, data-source lookup, numerical calculation, and model-based prediction. Together, these categories capture common needs in genomic analysis, including applying clinical guidelines, converting heterogeneous entity identifiers, querying case- or cohort-level evidence from genomic and clinical databases, performing standardized calculations, and predicting the functional efects of genomic variants.

Because resources in these five categories are organized diferently across the web, we did not use a shared search strategy for all categories. Instead, for each category, we first selected an existing broad-coverage collection as the primary source for automated resource extraction, and then manually added other important resources that were not captured. For example, to identify resources for data-source lookup, we extracted all 2,594 database entries from the cumulative alphabetical index of the Nucleic Acids Research Database Issue rather than searching it with a limited set of keywords. Because no single catalogue covers the full range of resources used in clinical genomics, we supplemented these entries with records from bio.tools, re3data.org and FAIRsharing.org and then merged records that referred to the same resource. The other four groups were scanned using the same general principle but with sources suited to each type of resource; the complete sources, search procedures and coverage limits are provided in Supplementary Sec. A.2.2 and all considered websites are listed in Supplementary Table 1.

This stage yielded 195 guidelines, 3,508 entity-mapping tools and services, 41,000 databases and resources for evidence lookup, 53 reproducible calculation methods, and 144 prediction models, forming the initial pool of candidate resources for agent skill construction. Notably, each candidate was represented by its associated free-text documentation; for example, tools, databases, and models were represented by their ‘README’ files, documentation pages, model cards, or related methodological descriptions.

## Stage 2. LLM-assisted resource filtering and expert review.

To further improve the quality of the candidate resources identified in Stage 1, we filtered them in Stage 2 based on their clinical relevance, reliability, and suitability for skill construction. We adopted an LLM–human hybrid filtering pipeline, in which an LLM first evaluated the candidate resources using category-specific criteria, followed by final review and selection by clinical experts.

Similar as Stage 1, the LLM rating pipeline was tailored to each resource category. For each category, we predefined a set of evaluation rubrics and prompted the LLM to assess each candidate against each rubric separately. For the entity mapping, data-source lookup, numerical calculation, and model-based prediction categories, the rubrics were quantitative, with the LLM assigning a numerical score to each rubric (e.g., 0–5). These rubric-level scores were then combined as a weighted sum to produce an overall score from 0 to 100. Based on this score, candidates were assigned to Tier 1 (≥ 75), Tier 2 (55–74), Tier 3 (35–54), or exclusion (< 35), with higher-tier candidates prioritized for subsequent review. In contrast, guideline-following candidates were assessed qualitatively: the LLM provided detailed comments on each predefined rubric without assigning numerical scores or tier levels. We designed this mainly because guidelines are typically high-value resources that warrant careful consideration. Providing a single numerical score or tier could lead subsequent reviewers to rely on that summary judgment. We therefore retained free-text comments, requiring experts to examine the underlying rationale and reach more considered conclusions. In general, the rubrics focused on the authority, currency, verifiability, and accessibility of guidelines; the clinical relevance, reliability, and accessibility of data and mapping resources; and the reproducibility, clinical support, and practical deployability of computational methods and predictive models. The complete rubrics, scoring instructions, and weights for each operation type are provided in Supplementary Table 2.

Eight clinical experts from seven institutions then reviewed the candidates for clinical utility, technical reliability, and functional redundancy, with the LLM-generated assessments provided as supporting material. Importantly, the LLM-derived tiers were used only to prioritize candidates for expert review and did not serve as strict inclusion thresholds. Final inclusion was determined by consensus among the eight clinicians, and lower-tier candidates could still be retained if they addressed important clinical needs not adequately covered by higher-ranked resources. More details are provided in Supplementary Sec. A.2.3.

After filtering, 105 candidate resource documents remained for skill construction: 27 for guideline following, 13 for entity mapping, 30 for data-source lookup, 12 for numerical calculation, and 23 for model-based prediction.

## Stage 3. Skill curation through recursive optimization.

In this stage, we translated the 105 selected resources into standardized, executable agent skills and iteratively refined them through a builder–evaluator loop inspired by recursive self-improvement [37, 38]. We first consolidated the five resource categories described above into three functional skill classes: guide, search, and analysis. Guide skills were derived from clinical guidelines and other normative documents; search skills from resources for entity mapping and data-source lookup, including databases, search services, and their APIs; and analysis skills from numerical methods and predictive machine-learning models.

Each skill was initially compiled by a ‘builder’ agent based on DeepSeek-V4. The builder received the corresponding source materials together with a class-specific construction prompt and generated an initial skill implementation. For guide skills, it translated source documents into stepwise procedures and traceable decision rules. For search skills, it specified how to query the target resource, including accepted inputs, query procedures, and processing of returned records. For analysis skills, it implemented the required calculations or specified how predictive models should be invoked and their outputs interpreted. Across all three classes, each skill was represented by a standardized SKILL.md file defining its scope, inputs, execution workflow, outputs, and limitations, accompanied, where necessary, by executable scripts and supporting references. The complete construction prompts and skill specification are provided in Supplementary Sec. A.2.4.

To verify and iteratively improve each compiled skill, we employed a separate ‘evaluator’ agent. Based on the source materials and the initial skill implementation, the evaluator automatically generated 5–20 skill-specific test cases, each specifying an input, the expected action or procedure, and the expected output or behaviour (representative examples are provided in Supplementary Table 3). The resulting test cases were then executed by a separate execution agent, instantiated as an LLM equipped with the corresponding skill. For each failed case, the evaluator summarized the observed failure in free text and provided the corresponding test output to the builder. The builder then revised the relevant skill instructions, interface specification, or executable code as appropriate, and the updated skill was subsequently re-evaluated using the same procedure. This builder–evaluation–revision cycle continued until all identified failures were resolved or a maximum of five revision cycles was reached, resulting in 105 agent skills. The workflow is illustrated in Extended Data Fig. 4.

## Stage 4. Graph construction for scalable skill organization.

After obtaining the final 105 skills, we organized them into a skill graph to support more eficient skill retrieval and combination. In the graph, each skill was represented as a node, and we exhaustively considered all possible pairwise relationships among the 105 skills, yielding ${ \binom { 1 0 5 } { 2 } } = 5$ , 460 unordered skill pairs. Each pair was then evaluated for two types of relationships: functional relatedness and prerequisite dependence. For each skill pair, an LLM first assessed functional relatedness along five dimensions—purpose similarity, input overlap, substitutability, complementarity, and joint use—assigning each a score from 0 to 4. These scores were summed to form an undirected relatedness edge weight ranging from 0 to 20. Separately, a directed prerequisite edge was added when one skill required another to obtain or interpret its output. The resulting graph contained 28 directed prerequisite edges. Relatedness weights support the retrieval of complementary skills, whereas prerequisite edges preserve required workflow steps. New skills can be incorporated by evaluating only their relationships with existing nodes. Further details are provided in Supplementary Sec. A.3.

## Stage 5. Context management based on progressive disclosure.

Till here, Stages 1–4 established the core of ClinGeX: a skill library organized as a skill graph. During inference, this graph is integrated into the host LLM’s context through task-specific retrieval and progressive plan–execute orchestration.

Given a task query, ClinGeX first invokes an LLM-based intent-recognition agent (Extended Data Fig. 5). The agent decomposes the user’s request into one to three concrete analytical needs. Each need is encoded using Qwen text-embedding-v4 [39], and cosine similarity is computed against the purpose and applicability descriptions of all 105 skills. The 30 highest-ranked skills form the seed set. For each seed, one-hop related skills with an edge weight of $R \geq 8$ are considered, and at most 15 additional skills are retained across all seeds. Three general literature- and web-search skills are included by default, after which prerequisite dependencies are recursively added until closure.

After retrieval, ClinGeX invokes a host agent that loads concise descriptions of all selected skills, each summarizing the skill’s capabilities, usage boundaries, and routing rules, and uses them to formulate an overall workflow plan. The host agent then instantiates sub-agents for individual workflow steps, each of which loads the complete instructions of the corresponding skill into its context and executes the assigned task according to the skill-specific workflow. More details of this stage are provided in Supplementary Sec. A.3.3.

## Stage 6. Multimodal support and online deployment.

In the final stage, we implemented ClinGeX as a multimodal online system. Uploaded files were processed in parallel according to type. PDFs were directly converted into Markdown-formatted text using MinerU [40], whereas images, such as pedigree charts, were passed to Qwen3-VL [41] with a fixed instruction to generate descriptive summaries. The resulting texts were concatenated with the user’s query to form the input context.

We deployed ClinGeX as a web application supporting multiple backbone models (Qwen, DeepSeek, Gemini, GLM, GPT, Kimi, etc.), heterogeneous file uploads, and concurrent queries. As shown in Extended Data Fig. 3, the interface contains four components: a dialogue module for user input and file submission, an execution module that streams intermediate operations and tool results, a result module that presents the final structured analysis, and a user-management module for reviewing previous sessions and conducting follow-up analyses.

## 4.2 Conduct of the MRMC Human-AI Collaboration Study

We conducted a randomized, three-condition human–AI collaboration study following established MRMC design principles [42], with condition order and case-set assignment counterbalanced using a Latin-square-like allocation [43], and administered the study through an online evaluation platform (Supplementary Sec. A.7.1).

## 4.2.1 Task Scenarios

We evaluated ClinGeX in three clinical genomic analysis scenarios, selected to cover criterion-level variant analysis, genomic interpretation in rare disease, and molecular oncology decision-making, detailed as:

• ACMG-based variant pathogenicity classification and analysis (criterion-level variant analysis) Participants evaluated the applicability and evidence strength of each of the 28 ACMG/AMP criteria for a given variant and assigned a final pathogenicity classification. They also provided a written justification for their assessment, along with a confidence rating and the time taken to complete the task. This scenario primarily assessed participants’ ability to evaluate pathogenicity evidence according to the ACMG/AMP framework and integrate the applicable criteria into a final variant classification.

• Genomic interpretation in rare disease through integrated variant and phenotype assessment (genomic interpretation in rare disease). Participants evaluated whether the candidate genes and variants could explain the patient’s phenotype, integrating variant-level evidence with phenotype concordance and inheritance information. They provided an integrated case-level interpretation, together with a written justification, confidence rating, and completion time. This scenario primarily assessed participants ability to synthesize genetic and phenotypic evidence to determine the extent to which a candidate finding explained the patient’s disease.

• Molecular tumor analysis and action recommendation (molecular oncology decision-making). Participants reviewed the clinical and molecular information for a tumour case, proposed the most appropriate next analytical or clinical action, and evaluated the plausibility of a series of case-specific statements. They also reported their confidence and completion time. This scenario assessed the integration of molecular findings with clinical context, the identification of actionable evidence, and the distinction between supported and unsupported clinical recommendations.

Representative cases and complete task examples are provided in Supplementary Sec. A.7.6.

## 4.2.2 Tested Conditions

Each participant completed tasks under the following three assistance conditions:

• Routine condition. Participants completed the assigned tasks according to their routine clinical practice. They were allowed to access the internet and freely consult any literature, databases, or online resources that they would normally use in their routine workflow, but were not permitted to use any AI-based tools.

• DeepSeek-assisted condition. Participants had access to the same resources permitted in the unassisted condition and could additionally use DeepSeek through its oficial web interface.

• ClinGeX-assisted condition. Participants had access to the same resources permitted in the unassisted condition and could additionally use the deployed ClinGeX platform.

The assigned system was the only AI tool permitted in each assisted condition, and all other study procedures were kept consistent across conditions.

## 4.2.3 Data Preparation

In this section, we describe the evaluation data preparation procedures for the three clinical scenarios. For each scenario, we prepared 45 evaluation cases, resulting in a total of 135 cases:

• Criterion-level variant analysis. Five experienced genetic analysts from Xinhua Hospital, each routinely interpreting more than 500 variants per year in clinical practice, contributed 39 patient records involving 45 clinically validated variants. After de-identification, the patient phenotypes, target variants, and supporting literature used in the original clinical interpretation were assembled into 45 evaluation cases.

• Genomic interpretation in rare disease. We obtained 25 de-identified patients with rare diseases from the 2023-2025 curated laboratory quality control cases. An expert panel from CUHK carefully reviewed the clinical information, validated variants, interpretation reports, and corresponding VCF files. Additional clinically reportable variants, including secondary findings and candidate variants with partial phenotypic correlation, were selected from the VCF files and combined with the validated diagnostic variants, producing 45 evaluation cases with distinct candidate-variant sets. Diferent cases derived from the same original case were not assigned to the same participant.

• Molecular oncology decision-making. We retrieved 323 human molecular-oncology case reports published after 2025 from PubMed, comprising 186 digestive-system and 137 thoracic-tumour reports. The reports were processed using a multi-model question-generation and quality-control procedure, with details provided in the Supplementary Sec. A.7.7. Following expert review, 27 digestive-system and 18 thoracic-tumour cases were retained, yielding 45 evaluation cases. The selected cases were manually de-identified, and nonessential clinical details were redacted or reformulated to prevent AI systems from directly locating the corresponding source publications.

The research use of these clinical data was approved by the Ethics Committee of Xinhua Hospital.

## 4.2.4 Participant Recruitment

We enrolled 54 clinical genomic professionals as participants, comprising 45 readers and 9 adjudicators. For each of the three clinical scenarios, 15 readers with relevant domain expertise were recruited. The criterion-level variant analysis cohort consisted of genetic analysts with 3–8 years of experience, recruited through Xinhua Hospital and including analysts from other hospitals undertaking advanced clinical training there. The interpretation working group of genomic interpretation in rare disease comprised clinical geneticists, genomic laboratory analysts, and postdoctoral fellows with 3–10 years of experience from Hong Kong, Guangzhou, and Shandong, recruited through the Department of Obstetrics and Gynaecology, Faculty of Medicine, The Chinese University of Hong Kong. For molecular oncology decision-making, 15 clinicians and resident trainees with approximately 4–10 years of clinical experience were recruited through Shanghai Chest Hospital, including 6 specializing in thoracic oncology and 9 in digestive-system oncology. A three-member expert adjudication panel was additionally recruited from each of the three centres for its corresponding clinical scenario; each panel comprised one senior clinician with over 10 years of experience and two additional members from the same department with over 5 years of experience. All participants gave informed consent for their study responses to be used for research analysis.

## 4.2.5 Study Procedure

Within each clinical scenario, the 45 cases were distributed among 15 readers using a Latin-square-like allocation scheme. Each reader completed nine cases: three in the unassisted condition, three in the DeepSeekassisted condition, and three in the ClinGeX-assisted condition. The cases assigned to an individual reader did not overlap across conditions. Consequently, each of the 45 cases was evaluated three times, once under each condition and by a diferent reader. The order of the three conditions was randomized across readers. For each assigned case, participants reviewed the case description and accompanying materials, completed the scenario-specific assessment, and submitted their final conclusion, written justification, confidence rating, and completion time through the online evaluation platform.

Several measures were used to reduce potential learning, order, fatigue, and information-leakage efects. No reader encountered the same case more than once, and the balanced allocation prevented individual cases from being evaluated exclusively under a particular assistance condition. Participants were required to work independently, to refrain from using any AI tools other than those explicitly permitted under their assigned condition, and to avoid any attempt to identify the source or reference answer of a task. To limit fatigue, the nine cases were completed over a two-week period, and participants were asked to distribute them as evenly as possible across this period. During the study, the expert panel corresponding to each clinical scenario answered procedural questions without providing case-specific guidance. After all assessments had been completed, participant responses were evaluated according to the predefined primary and secondary outcomes. Outcomes requiring clinical judgement were assessed by the corresponding expert panels whereas objective outcomes were calculated from the submitted responses and recorded task data using prespecified criteria (Supplementary Sec. A.7.2, A.7.3).

## 4.2.6 Participant Post-study Survey

After completing all assigned tasks, the 45 readers completed the same anonymous post-study survey, with 15 readers for each clinical scenario. Participants were instructed to complete the survey independently based on their experience with the three study conditions. Survey responses were collected anonymously and de-identified before analysis.

The survey comprised 13 experience-related items. Three open-response items asked participants to estimate their average time per case under the routine, DeepSeek-assisted, and ClinGeX-assisted conditions. These estimates were used for the main time comparison because readers were not required to complete cases in a single uninterrupted session; platform timestamps could therefore include pauses or unrelated clinical work and were not treated as active analysis time. The remaining ten items assessed DeepSeek and ClinGeX separately across five clinician-reported dimensions using the following five-point ordinal scales:

• Confidence after assistance: substantially less confident than when working independently; somewhat disrupted by the system; about the same as when working independently; slightly more confident; or markedly more confident.

• Willingness to use the system again in clinical work: very unwilling; somewhat unwilling; neutral; somewhat willing; or very willing.

• Satisfaction with the presentation of results in a clinical context: very dissatisfied; somewhat dissatisfied; neutral; somewhat satisfied; or very satisfied.

• Smoothness of system operation: not at all smooth; not very smooth; moderately smooth; relatively smooth; or very smooth.

• Reliance on system output when completing the tasks: almost no reliance; reliance on a small part of the output; reliance on approximately half of the output; reliance on most of the output; or complete reliance.

## 4.3 Prospective, Non-interventional Feedback Collection of ClinGeX

Following the retrospective MRMC evaluation on controlled cases, we conducted an exploratory prospective task-based assessment to evaluate ClinGeX’s utility in daily clinical-genomics analysis, inspired by prior evaluations of clinical AI in real-world care [44, 45]. Here, “prospective” denotes the forward collection of participants’ routine clinical or research-level analysis tasks that were comparable in nature to the evaluation tasks. Experienced clinical professionals evaluated ClinGeX on tasks arising from their usual genomic-analysis work and provided structured feedback after completing each task. The system outputs were used solely for research evaluation and were not permitted to inform or alter clinical reports, patient management or any other clinical decision. To capture clinicians’ real-world needs and patterns of use, we did not prescribe a fixed set of cases or tasks, nor did we include a control condition. Accordingly, this study was intended to characterize the task-level utility of our system in a workflow-relevant, non-interventional setting.

## 4.3.1 Study Design and Participant Recruitment

We recruited 11 professionals engaged in clinical variant interpretation, each with more than five years of variant-interpretation experience and responsibility for preparing clinical reports for more than 500 cases per year. Six participants were from the Department of Genetics, Xinhua Hospital; two were from the Department of Obstetrics and Gynaecology, Faculty of Medicine, The Chinese University of Hong Kong; and one participant each was from the Department of Medical Genetics, Shandong Provincial Hospital, Anhui Provincial Hospital, and Shanghai Jiao Tong University School of Medicine. None had participated in the preceding MRMC study.

Professionals used ClinGeX during their routine work over a prespecified one-month period. They were asked to submit feedback on clinical genomic analysis episodes for which they had actually used ClinGeX, including, but not limited to, variant interpretation, evidence retrieval, literature analysis, and case-level analysis. They were explicitly instructed to report their experience objectively, irrespective of whether ClinGeX had been beneficial, neutral, or detrimental to completion of the analysis. All of them gave informed consent for their submitted feedback to be used for research analysis.

## 4.3.2 Feedback Platform and Unit of Evaluation

We developed a platform to collect structured feedback on participants’ routine use of ClinGeX. The platform did not request or extract patient names, identifiers, clinical records, genomic data or other patient-level information. Participants were instructed to follow data-protection policies throughout their use of ClinGeX and submission of feedback, including avoiding the disclosure of potential identifiable patient information.

One feedback submission represented one completed real-world clinical-genomics analysis episode and assessed the contribution of ClinGeX across that episode as a whole. An episode could involve multiple rounds of interaction with ClinGeX; the selected conversation record was used only to associate the feedback with the corresponding episode and analysis process, rather than to define an individual message or dialogue turn as an independent evaluation. A post hoc comparison with platform conversation counts indicated feedback coverage of at least 86.5%, providing some reassurance against selective reporting (Supplementary Table 21).

## 4.3.3 Feedback Measures

The feedback measures were designed by a senior clinical-genomics physician at Xinhua Hospital with reference to a previous prospective clinical AI deployment study [46]. Each submission collected the following items:

• Analysis dificulty: simple, not requiring AI; moderate, for which AI could provide some assistance; or dificult, for which AI could provide substantial assistance.

• Adoption of ClinGeX’s conclusion: fully adopted; partially adopted; or not adopted. For variantassessment episodes, adoption referred to whether and to what extent the participant retained ClinGeX’s proposed final pathogenic, likely pathogenic, variant of uncertain significance, likely benign or benign (P/LP/VUS/LB/B) classification in their final assessment. For other episodes, adoption referred to the principal task conclusion.

• Adoption of recommended evidence criteria: the number of evidence criteria recommended by ClinGeX and the number ultimately adopted by the participant. This item was optional and intended for episodes in which discrete evidence criteria were applicable.

• Time saved with ClinGeX: more than 15 min saved; 5–15 min saved; or no change in completion time or slower completion.

• Help with information standardization: very helpful or prevented an error; provided some help and saved time; or provided little or no help.

• Help with numerical analysis: very helpful and gave numerical retrieving or analysis that the participant might otherwise have missed; provided some help and saved time; or provided little or no help.

• Help with literature analysis: very helpful, identifying relevant literature and providing accurate interpretation; provided some useful interpretation; or provided little or no help.

• Help with analytical or procedural guidance: very helpful and provided a distinct, useful analytical approach; provided some help through a reasonably accurate routine analysis; or provided little or no help.

• Specific contribution to the episode: an optional free-text description of the assistance provided by ClinGeX.

All categorical items were required, whereas the evidence-criterion counts and free-text description were optional. Each submission was stored together with the participant identifier and selected conversation identifier.

## 5 Data availability

The data supporting the findings of this study are available upon reasonable request. Requests for data access should be directed to three-world@sjtu.edu.cn or to any of the corresponding-author email addresses.

## 6 Code availability

Although the source code for ClinGeX is not publicly available at present, an online deployed Chinese–English bilingual version of the system is available at https://chatgeno.internal.onexai.com.cn/ after registration.

## References

[1] Olaf Riess, Marc Sturm, Benita Menden, Alexandra Liebmann, German Demidov, Dennis Witt, Nicolas Casadei, Jakob Admard, Leon Schütz, Stephan Ossowski, Stacie Taylor, Sven Schafer, Christopher Schroeder, Andreas Dufke, and Tobias Haack. Genomes in clinical care. npj Genomic Medicine, 9(1):20, March 2024. 1

[2] 100,000 Genomes Project Pilot Investigators. 100,000 Genomes Pilot on Rare-Disease Diagnosis in Health Care—Preliminary Report. New England Journal of Medicine, 385(20):1868–1880, November 2021. 1

[3] Ahmet Zehir, Ryma Benayed, Ronak H. Shah, Aijazuddin Syed, Sumit Middha, Hyunjae R. Kim, Preethi Srinivasan, Jianjiong Gao, Debyani Chakravarty, Sean M. Devlin, Matthew D. Hellmann, David A. Barron, Alison M. Schram, Meera Hameed, Snjezana Dogan, Dara S. Ross, Jaclyn F. Hechtman, Deborah F. DeLair, JinJuan Yao, Diana L. Mandelker, Donavan T. Cheng, Raghu Chandramohan, Abhinita S. Mohanty, Ryan N. Ptashkin, Gowtham Jayakumaran, Meera Prasad, Mustafa H. Syed, Anoop Balakrishnan Rema, Zhen Y. Liu, Khedoudja Nafa, Laetitia Borsu, Justyna Sadowska, Jacklyn Casanova, Ruben Bacares, Iwona J. Kiecka, Anna Razumova, Julie B. Son, Lisa Stewart, Tessara Baldi, Kerry A. Mullaney, Hikmat Al-Ahmadie, Efsevia Vakiani, Adam A. Abeshouse, Alexander V. Penson, Philip Jonsson, Niedzica Camacho, Matthew T. Chang, Helen H. Won, Benjamin E. Gross, Ritika Kundra, Zachary J. Heins, Hsiao-Wei Chen, Sarah Phillips, Hongxin Zhang, Jiaojiao Wang, Angelica Ochoa, Jonathan Wills, Michael Eubank, Stacy B. Thomas, Stuart M. Gardos, Dalicia N. Reales, Jesse Galle, Robert Durany, Roy Cambria, Wassim Abida, Andrea Cercek, Darren R. Feldman, Mrinal M. Gounder, A. Ari Hakimi, James J. Harding, Gopa Iyer, Yelena Y. Janjigian, Emmet J. Jordan, Ciara M. Kelly, Maeve A. Lowery, Luc G. T. Morris, Antonio M. Omuro, Nitya Raj, Pedram Razavi, Alexander N. Shoushtari, Neerav Shukla, Tara E. Soumerai, Anna M. Varghese, Rona Yaeger, Jonathan Coleman, Bernard Bochner, Gregory J. Riely, Leonard B. Saltz, Howard I. Scher, Paul J. Sabbatini, Mark E. Robson, David S. Klimstra, Barry S. Taylor, Jose Baselga, Nikolaus Schultz, David M. Hyman, Maria E. Arcila, David B. Solit, Marc Ladanyi, and Michael F. Berger. Mutational landscape of metastatic cancer revealed from prospective clinical sequencing of 10,000 patients. Nature Medicine, 23(6):703–713, May 2017. 1

[4] Sara Goodwin, John D. McPherson, and W. Richard McCombie. Coming of age: ten years of nextgeneration sequencing technologies. Nature Reviews Genetics, 17(6):333–351, May 2016. 1

[5] Euan A. Ashley. Towards precision medicine. Nature Reviews Genetics, 17(9):507–522, August 2016. 1

[6] Paul Muir, Shantao Li, Shaoke Lou, Daifeng Wang, Daniel J. Spakowicz, Leonidas Salichos, Jing Zhang, George M. Weinstock, Farren Isaacs, Joel Rozowsky, and Mark Gerstein. The real cost of sequencing: scaling computation to keep pace with data generation. Genome Biology, 17(1):53, March 2016. 1

[7] Andrea Sboner, Xinmeng Jasmine Mu, Dov Greenbaum, Raymond K. Auerbach, and Mark B. Gerstein. The real cost of sequencing: higher than you think! Genome Biology, 12(8):125, August 2011. 1

[8] Edgar A. Rivera-Muñoz, Laura V. Milko, Steven M. Harrison, Danielle R. Azzariti, C. Lisa Kurtz, Kristy Lee, Jessica L. Mester, Meredith A. Weaver, Erin Currey, William Craigen, Charis Eng, Birgit Funke, Madhuri Hegde, Ray E. Hershberger, Rong Mao, Robert D. Steiner, Lisa M. Vincent, Christa L. Martin, Sharon E. Plon, Erin Ramos, Heidi L. Rehm, Michael Watson, and Jonathan S. Berg. ClinGen variant curation expert panel experiences and standardized processes for disease and gene-level specification of the ACMG/AMP guidelines for sequence variant interpretation. Human Mutation, 39(11):1614–1622, October 2018. 1

[9] Sue Richards, Nazneen Aziz, Sherri Bale, David Bick, Soma Das, Julie Gastier-Foster, Wayne W. Grody, Madhuri Hegde, Elaine Lyon, Elaine Spector, Karl Voelkerding, and Heidi L. Rehm. Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the american college of medical genetics and genomics and the association for molecular pathology. Genetics in Medicine, 17(5):405–424, May 2015. 1

[10] Elaine R. Mardis. The \$1,000 genome, the \$100,000 analysis? Genome Medicine, 2(11):84, 2010. 1

[11] William McLaren, Laurent Gil, Sarah E. Hunt, Harpreet Singh Riat, Graham R. S. Ritchie, Anja Thormann, Paul Flicek, and Fiona Cunningham. The Ensembl Variant Efect Predictor. Genome Biology, 17(1):122, June 2016. 1

[12] Damian Smedley, Julius O. B. Jacobsen, Marten Jäger, Sebastian Köhler, Manuel Holtgrewe, Max Schubach, Enrico Siragusa, Tomasz Zemojtel, Orion J. Buske, Nicole L. Washington, William P. Bone, Melissa A. Haendel, and Peter N. Robinson. Next-generation diagnostics and disease-gene discovery with the Exomiser. Nature Protocols, 10(12):2004–2015, November 2015. 1

[13] Xiaolan Chen, Jiayang Xiang, Shanfu Lu, Yexin Liu, Mingguang He, and Danli Shi. Evaluating large language models and agents in healthcare: key challenges in clinical applications. Intelligent Medicine, 5(2):151–163, May 2025. 1

[14] Alex J. Goodell, Simon N. Chu, Dara Rouholiman, and Larry F. Chu. Large language model agents can use tools to perform clinical calculations. npj Digital Medicine, 8(1):163, March 2025. 1

[15] Florian Borchert, Andreas Mock, Aurelie Tomczak, Jonas Hügel, Samer Alkarkoukly, Alexander Knurr, Anna-Lena Volckmar, Albrecht Stenzinger, Peter Schirmacher, Jürgen Debus, Dirk Jäger, Thomas Longerich, Stefan Fröhling, Roland Eils, Nina Bougatf, Ulrich Sax, and Matthieu-P. Schapranow. Knowledge bases and software support for variant interpretation in precision oncology. Briefings in Bioinformatics, 22(6):bbab134, May 2021. 1

[16] Qiao Jin, Yifan Yang, Qingyu Chen, and Zhiyong Lu. GeneGPT: augmenting large language models with domain tools for improved access to biomedical information. Bioinformatics, 40(2):btae075, February 2024. 1

[17] Qianyu Meng, Yanan Wang, Liyi Chen, Wei Wu, Yihang Li, Wenyuan Jiang, Qimeng Wang, Chengqiang Lu, Yan Gao, Yi Wu, and Yao Hu. Agent harness for large language model agents: A survey. 2026. 2

[18] OpenAI. Introducing deep research. https://openai.com/index/introducing-deep-research/, February 2025. Accessed 16 August 2026. 3, 4

[19] Kexin Huang, Serena Zhang, Hanchen Wang, Yuanhao Qu, Yingzhou Lu, Ryan Li, Yusuf Roohani, Lin Qiu, Shiyi Cao, Gavin Li, Junze Zhang, Di Yin, et al. Autonomous biomedical research with an artificial intelligence agent. Science, 2026. 3, 4

[20] Daya Guo, Dejian Yang, Haowei Zhang, et al. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645:633–638, 2025. 3

[21] Tim Schubert, Tim Oosterlinck, Robert D. Stevens, Patrick H. Maxwell, and Mihaela van der Schaar. AI education for clinicians. eClinicalMedicine, 79:102968, January 2025. 3

[22] Jack W. O’Sullivan, Anil Palepu, Khaled Saab, Wei-Hung Weng, Daniel K. Amponsah, Evaline Cheng, Yong Cheng, Emily Chu, Yaanik Desai, Aly Elezaby, Muhammad Fazal, Tasmeen Hussain, Sneha S. Jain, Daniel Seung Kim, Roy Lan, Jiwen Li, Wilson Tang, Natalie Tapaskar, Victoria Parikh, Ryan Sandoval, Gabriella Spencer-Bonilla, Bryan Wu, Kavita Kulkarni, Philip Mansfield, Dale Webster, Juraj Gottweis, Joelle Barral, Mike Schaekermann, Ryutaro Tanno, S. Sara Mahdavi, Vivek Natarajan, Alan Karthikesalingam, Euan Ashley, and Tao Tu. A large language model for complex cardiology care. Nature Medicine, 32(2):616–623, February 2026. 3

[23] Melissa J. Landrum et al. ClinVar: Updates to support classifications of both germline and somatic variants. Nucleic Acids Research, 53(D1):D1313–D1321, 2025. 4

[24] Natasha T. Strande et al. Evaluating the clinical validity of gene–disease associations: An evidence-based framework developed by the clinical genome resource. The American Journal of Human Genetics, 100(6):895–906, 2017. 4

[25] ClinGen Consortium. The clinical genome resource (ClinGen): Advancing genomic knowledge through global curation. Genetics in Medicine, 27(1):101228, 2025. 4

[26] Malachi Grifith et al. CIViC is a community knowledgebase for expert crowdsourcing the clinical interpretation of variants in cancer. Nature Genetics, 49(2):170–174, 2017. 4

[27] Kilannin Krysiak et al. CIViCdb 2022: Evolution of an open-access cancer variant interpretation knowledgebase. Nucleic Acids Research, 51(D1):D1230–D1241, 2023. 4

[28] Debyani Chakravarty et al. OncoKB: A precision oncology knowledge base. JCO Precision Oncology, 1:1–16, 2017. 4

[29] Sarah P. Suehnholz et al. Quantifying the expanding landscape of clinical actionability for patients with cancer. Cancer Discovery, 14(1):49–65, 2024. 4

[30] Michelle Whirl-Carrillo, Rachel Huddart, Li Gong, Katrin Sangkuhl, Caroline F. Thorn, Ryan Whaley, and Teri E. Klein. An evidence-based framework for evaluating pharmacogenomics knowledge for personalized medicine. Clinical Pharmacology & Therapeutics, 110(3):563–572, 2021. 4

[31] Ankit Pal and Malaikannan Sankarasubbu. OpenBioLLMs: Advancing open-source large language models for healthcare and life sciences. https://huggingface.co/aaditya/OpenBioLLM-Llama3-70B, 2024. 4

[32] Trinita Roy, Asheesh Kumar, Daksh Raghuvanshi, Siddhant Jain, Goutham Vignesh, Kartik Shinde, and Rohan Tondulkar. SciSpace Copilot: Empowering researchers through intelligent reading assistance. Proceedings of the AAAI Conference on Artificial Intelligence, 38(21):23826–23828, 2024. 4

[33] Shanghua Gao, Richard Zhu, Zhenglun Kong, Ayush Noori, Xiaorui Su, Curtis Ginder, Theodoros Tsiligkaridis, and Marinka Zitnik. TxAgent: An AI agent for therapeutic reasoning across a universe of tools, 2025. 4

[34] James V Bradley. Complete counterbalancing of immediate sequential efects in a latin square design. Journal of the American Statistical Association, 53(282):525–528, 1958. 7

[35] Khaled Saab, Chunjong Park, Tim Strother, Jan Freyberg, David G. T. Barrett, Yong Cheng, Wei-Hung Weng, David Stutz, Nenad Tomasev, Anil Palepu, Valentin Liévin, Yash Sharma, Roma Ruparel, Abdullah Ahmed, Elahe Vedadi, Kimberly Kanada, Cian Hughes, Yun Liu, Geof Brown, Yang Gao, Sean Li, S. Sara Mahdavi, James Manyika, Katherine Chou, Yossi Matias, Avinatan Hassidim, Dale R. Webster, Joëlle Barral, S. M. Ali Eslami, Pushmeet Kohli, Adam Rodman, Vivek Natarajan, Mike Schaekermann, Tao Tu, Alan Karthikesalingam, and Ryutaro Tanno. Advancing conversational diagnostic AI with multimodal reasoning. Nature Medicine, 32(5):1726–1736, 2026. 9

[36] Anthropic. Equipping agents for the real world with agent skills. Anthropic Engineering, October 2025. 15

[37] Alexander Novikov, Ngân V˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025. 16

[38] Anthropic. When ai builds itself: Our progress toward recursive self-improvement, and its implications. https://www.anthropic.com/institute/recursive- self-improvement, June 2026. Anthropic Institute, accessed September 2, 2026. 16

[39] Yanzhao Zhang, Mingxin Li, Dingkun Long, Zhang Xin, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025. 17

[40] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024. 17

[41] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 17

[42] Nancy A. Obuchowski and Jennifer Bullen. Multireader diagnostic accuracy imaging studies: Fundamentals of design and analysis. Radiology, 303(1):26–34, 2022. 17

[43] René Zeelenberg and Diane Pecher. A method for simultaneously counterbalancing condition order and assignment of stimulus materials to conditions. Behavior Research Methods, 47(1):127–133, 2015. 17

[44] Matthew J. Duggan, Julietta Gervase, Anna Schoenbaum, William Hanson, III Howell, John T., Michael Sheinberg, and Kevin B. Johnson. Clinician experiences with ambient scribe technology to assist with documentation burden and eficiency. JAMA Network Open, 8(2):e2460637, 2025. 20

[45] Xiayin Zhang, Qinyi Li, Yinhao Liang, Chunran Lai, Jiahui Cao, Yangqin Feng, et al. AI framework for multidisease detection via retinal imaging. Nature Medicine, 32:2494–2503, 2026. 20

[46] Jack W. O’Sullivan, Anil Palepu, Khaled Saab, et al. A large language model for complex cardiology care. Nature Medicine, 32:616–623, 2026. 21

e Statistical analysis on base model selection as the agent core  
![](images/398debb54db02dda96694a8303ec7c9dc3ddedb04d4b52221da2f78744b33078.jpg)

a <sub>3-tier</sub> <sub>tasks for system architecture evaluation</sub>  
![](images/3095e75e185a60b6877c3853e91aa980c2bbb201ef54f81cbb59364a142de37b.jpg)

b  
![](images/e119ee466de570d1fd8ead24ed3dc4bdc80ca23a5c6fff1aae8b0e06dbdfd4e4.jpg)  
d Evaluation metrics of base model selection as agent core  
c Comparison on system architecutre design

<table><tr><td>Metric</td><td>Calculation</td><td>Scoring</td><td>Measures</td></tr><tr><td>Absolute ACC 0, 0.5 or 1</td><td>LLM labels atomic claims; deterministic thresholds and critical-error override assign the score</td><td>HYBRID</td><td>MODEL OUTPUT</td></tr><tr><td>Reasoning 0-5</td><td>∑ five binary criteria: evidence, trap handling, logic, completeness and self-consistency</td><td>LLM JUDGE</td><td>MODEL OUTPUT</td></tr><tr><td>three trace statistics</td><td>Agentic execution Tool calls; failed-call rate; exact-duplicate rate computed directly from execution traces</td><td>HARD RULE</td><td>AGENTIC RUN</td></tr><tr><td>Faithfulness 0–1; higher is better</td><td>1 - 0.5 × fabricated-quote rate - 0.5 × unsupported-or-contradicted claim rate</td><td>HYBRID</td><td>MODEL OUTPUT</td></tr><tr><td>Robustness 0–1; three runs</td><td>0.4 × answer consistency + 0.3 × tool-sequence similarity + 0.3 × agentic stability</td><td>HARD RULE</td><td>AGENTIC RUN</td></tr></table>

![](images/a1425412036de283ed13caab2e4b242bba077ffe469c3ff490165f4e3ccb2765.jpg)  
response-level requires agentic executior  
LLM judge is external to the tested model

![](images/d299e425158ab42c04778369152fe780e6e28c64f21818c4123130c03fb40bf4.jpg)

![](images/136d47bc895f77832a77e0805bd7fd79b5f2ef236f94c8dd3896fb3de3fbf3b1.jpg)

![](images/3723eec1fae1e25705f0e9ed79a4331b5c7fbc4fe35c47557041412305532e24.jpg)

![](images/e9fa7df1294b307dc7d3944c4ed70e681ecf373fd400eeafa216134b0f068657.jpg)

![](images/99c5a1ad0042496d576940180573572e4b64f68a26f757bf2d4cfdb1fbd8c72f.jpg)

Extended Data Figure 1 | Evaluation on system architecutre and agent core. ${ \mathbf { a } } ,$ The evidence-based question-answering tasks, stratified into three tiers of reasoning complexity—single-paper comprehension, evidence-strength grading with one planted distractor, and multi-paper clinical synthesis—with a representative item and its scoring focus for each tier. b, Tested system architectures and scoring pipeline: every architecture answers the identical task input, and its response is graded against the item-specific rubric by two independent LLM judges, whose normalised scores are averaged. $\mathbf { c } ,$ Ablation on system component design. Top, average normalised score over all items $( n = 4 2 6 )$ for the plain LLM baseline, the single-component variants and the full ClinGeX built on the same base LLM. Bottom, scores by dificulty tier for items derived from ClinVar $( n = 8 0 )$ ClinGen $( n = 1 3 8 )$ and ClinPGx, CIViC and OncoKB $( n = 2 0 8 )$ . Error bars denote 95% confidence intervals; asterisks indicate comparisons against ClinGeX (ns, not significant; $^ { * } P < 0 . 0 5 ; ^ { * * } P < 0 . 0 1 ; ^ { * * * } P < 0 . 0 0 1 )$ . d, The five metrics used for base-model selection, with their calculation, scoring mode and the aspect of system behaviour each measures. $\mathbf { e } ,$ Radar plots of the five metrics on tasks derived from ClinVar (n = 1,366), ClinGen (n = 1,253) and ClinPGx, CIViC and OncoKB (n = 1,043).

Stud<sub>y</sub> Cohort & Case Allocation 3 independent reader groups, one per clinical scenario -- 45 readers and 135 cases in total  
![](images/e62f7240d7269e1e1594912cce25225c1fac931f97de2f84800d107ac8cf0e8b.jpg)

Stud<sub>y</sub> Desi<sub>g</sub>n  
![](images/f8287e4ab508a39383e854f2929686aada262c4e788c3e357544ae3cd2e9b546.jpg)

Extended Data Figure 2 | Design of the multi-reader, multi-case human–AI collaboration study. Three independent groups of 15 clinical genomic professionals were assigned to variant pathogenicity assessment, gene–disease interpretation or molecular oncology decision-making (45 readers in total). Each scenario comprised 45 cases organized into 15 three-case sets. Each clinician assessed nine distinct cases: three unassisted, three with DeepSeek assistance and three with ClinGeX assistance. Condition-to-case-set assignments were counterbalanced across readers using three sequences (U–D–C, D–C–U and C–U–D). Structured responses were compared with the reference standard by rule-based matching, whereas free-text and borderline items were evaluated by a three-member expert panel blinded to study condition. Outcomes included case-level concordance or accuracy, evidentiary quality, confidence calibration, self-reported completion time and perceived utility. Participants also completed a 13-item post-study survey, and platform logs were used to assess spontaneous continued use of ClinGeX during the 14-day follow-up. R, Routine (no AI assistance); D, DeepSeek-assisted; C, ClinGeX-assisted.

![](images/fa61ca975df0ecc225c7a2ffc9ec0aeb65ca4a25717311df226dffeb5bf6560f.jpg)  
Extended Data Figure 3 | Representative workflow and user interface of ClinGeX. After logging into the system (1), the user enters a clinical-genomics query (2), optionally uploads supporting documents or images (3), and selects the underlying model (4). ClinGeX then identifies and invokes task-relevant skills and tools while displaying analysis progress (5), before returning a structured, evidence-supported response containing the final conclusion, supporting criteria and their interpretation (6). The example shown is an SNV pathogenicity assessment under the ACMG/AMP framework.

Skill compilation through recursive optimization  
![](images/fe522424d444d628775640813aeefd5d564a67b564c8fbeeab161cd17341d375.jpg)

Extended Data Figure 4 | Compilation of clinical-genomics skills through recursive optimization. Each expertreviewed skill prototype specifies its scope, target resource and action interface. A builder agent based on DeepSeek-V4 converts the prototype into an agent-readable skill contract, organizes its reference hierarchy and implements any required executable modules. After an internal consistency check of file paths, documented actions and parameter names, a skill-equipped test agent executes a frozen evaluation suite comprising 5–20 positive, boundary, malformed-input, batch-processing and simulated external-failure cases. The complete execution trace, including prompts, tool inputs and outputs, standard output and error streams, parsed results and test status, is retained for failure analysis. Failed tests are assigned to one of four repair targets: documentation or routing ambiguity, action–schema inconsistency, implementation or parsing error, or an external limitation requiring an explicit fallback. The corresponding documentation, interface, implementation, test or limitation statement is then revised, and the full regression suite is rerun. This cycle continues until no actionable failure remains or a maximum of five iterations is reached. Only skills passing all frozen tests are compiled and added to the deployed skill library (n = 105).

![](images/0c2d92a86a0a7fc92260e9ae09bb65064176d1c6f76f070ca7d48b9937f531ec.jpg)

Extended Data Figure 5 | Task-oriented skill selection in ClinGeX. Each user task is first decomposed by DeepSeek V4 Flash into one to three concise analytical needs. These needs are encoded as 1,024-dimensional vectors using Qwen text embedding-v4 and compared by cosine similarity with precomputed embeddings of the 105 skills. The maximum similarity across the decomposed needs is used to rank skills and identify the top candidate seeds. The seed set is subsequently expanded over SkillNet using thresholded related-skill edges and one-hop breadth-first search, while dependency closure adds any skills required for execution. The union of the initial seeds, graph-expanded skills and required dependencies forms the final task-specific skill subset supplied to the agent. Insets illustrate the organization of the skill library in the embedding space, visualized using t-SNE and HDBSCAN, and in the SkillNet relational graph.

## A Supplementary

## A.1 Main System Prompt of ClinGeX

The following reproduces the complete content of the deployed top-level ClinGeX/SKILL.md, with line wrapping, dash characters and the directory-tree glyphs normalized only for typesetting. It makes the operating stance, routing order, progressive-disclosure policy and path-resolution rules auditable.

## Complete top-level ClinGeX/SKILL.md router prompt

Front matter

name: ClinGeX

description: Clinical interpretation co-pilot for gene-related analysis. Helps clinicians and downstream LLM agents answer questions about variants, genes, drug–gene interactions, and reportable findings by routing to authoritative interpretation guidelines, specialized analysis tools, and external knowledge databases.

## ClinGeX

ClinGeX is an agentic chatbot that assists with literature-related, gene-related clinical interpretation, analysis, and recommendation tasks. It is loaded by an LLM agent that has been handed a clinical-genomics request and must decide whether and how to invoke specialized resources before answering.

This top-level SKILL.md is intentionally short. It is a router. For any concrete decision the agent identifies the relevant sub-skill category and descends into that sub-skill’s own SKILL.md.

Operating stance. ClinGeX is restrained, professional, dialectical, and objective. It does not over-trust any single database, publication, guideline summary, prediction score, or prior model belief. It separates observed facts, authoritative rules, tool-derived results, source assertions, and inferred conclusions, and weighs convergence, conflict, absence of evidence, source quality, and task-specific context before synthesizing an answer.

Progressive disclosure. This bundle never exposes all relevant or irrelevant documents, guidelines, or tool cards at once. Nothing is preloaded. Each document reveals only its own scope and may point to deeper material that becomes visible only when it is read. Because every disclosure step can surface new documents, constraints, or routes, a previously chosen plan may have to change at any time; the agent should revise its path as soon as a newly revealed document or result requires it.

How content is read. A sub-skill’s documentation is retrieved by calling ClinGeX with get\_documentation=True. Local bundle files are read with the helper scripts described in section 4, using absolute paths.

Path convention. Treat a relative path as relative to the directory of the document in which that path appears. A get\_documentation=True response provides documentation\_path, document\_base\_dir, and bundle\_base\_dir; resolve paths found in its content against document\_base\_dir. When a local file is subsequently read, resolve paths found in that file against the directory containing that file. This rule therefore continues to work at every depth of progressive disclosure, including .. references.

Only paths written in this top-level router as bundle-root paths—such as skills/... and scripts/...— are resolved against the absolute <BASE\_DIR> supplied outside this prompt (the same directory returned as bundle\_base\_dir). An already absolute path is used as given. In all cases, normalize the result to an absolute path and verify that it exists before passing it to a filesystem helper. Do not infer a path’s base merely from its prefix; the location of the document containing the path determines its base.

## 1. The three sub-skill categories

Nearly every clinical gene-related task reduces to some combination of a handful of underlying needs: consulting an authoritative guideline, looking up a fact in an external database, normalizing or transforming a piece of notation, computing or scoring something, or running a predictive model. This bundle groups the sub-skills that serve these needs into three categories:

• guide/—authoritative procedural standards: which guideline, which section, which rule applies.

• search/—external knowledge retrieval: current, sourced facts from public databases and literature, never guessed from memory.

• analysis/—computation and prediction: scores, classifications, and structured evidence codes computed from a variant or gene.

All sub-skills live under skills/<category>/<skill\_name>/SKILL.md.

## 2. Routing protocol (follow this every time)

1. Determine which guidelines are potentially relevant to the current task. Explicitly state and list their names, and for each candidate, why it is relevant. If none applies, say so explicitly. Then use get\_documentation to read the SKILL.md of each relevant candidate, and plan what to do next: what further documents must be read and what further sub-skills or tools must be used.

2. Execute step by step. As new results, documents, or comparison against the prior plan dictate, revise the plan and disclose further documents or sub-skills accordingly.

3. Decompose the task into the knowledge points and operational steps it involves, revising the initial plan based on the relevant guide documentation just read.

4. Before invoking any sub-skill for the first time, call ClinGeX with get\_documentation=True (any action value, e.g. action="help") to retrieve that sub-skill’s SKILL.md. This returns the exact list of valid actions, required parameters, supported input formats, and usage examples. Do not guess action names or parameter schemas—read the documentation first. If this documentation action is unavailable, read the sub-skill’s SKILL.md directly from disk using scripts/read\_file.py and its absolute path. Resolve any path revealed by a returned document relative to that document’s own directory, as specified by the path convention above.

5. Invoke the sub-skill using only actions and parameters documented above.

6. After each substantive result, decide whether the plan still fits or whether a newly revealed clue, conflict, or limitation requires another reference, tool, or narrower answer.

7. Synthesize an answer that leads with the direct conclusion, followed by detailed analysis then cites which skill produced which fact, weighs convergent and conflicting evidence, and states residual uncertainty, assumptions, and clinically relevant risk.

If nothing surfaced matches the task, say so rather than guessing.

## 3. Repository layout

ClinGeX/

SKILL.md—this file (router)

scripts/—OS-level helpers (read/write/list/run); see section 4

assets/—reserved, currently empty

skills/guide/<topic>/SKILL.md

skills/search/<source>/SKILL.md

skills/analysis/<tool>/SKILL.md

## 4. OS-level helper scripts

scripts/ provides small, self-contained Python utilities (stdlib only, no third-party imports) that an agent can call when it needs to manipulate the local filesystem or run code as part of fulfilling a task. They are documented in scripts/README.md and are intentionally generic; they are not clinical tools.

• scripts/read\_file.py: A general-purpose text file reader for local files in this bundle. Robust read with encoding fallback, size guard, line slicing, optional structured parse, JSON envelope output.

• scripts/write\_file.py: Atomic write; no-clobber / overwrite / append modes; optional directory creation; dry-run; optional canonical JSON re-serialization.

• scripts/list\_files.py: Recursive listing with glob/regex filters, extension and size/mtime filters, depth limit, symlink/hidden controls, sorted output.

• scripts/run\_python.py: Execute a file or inline snippet via subprocess; argument forwarding, cwd, env passthrough, timeout, captured output, JSON envelope.

Every script accepts –help and emits a structured JSON envelope when –json is passed. Errors always exit non-zero with a JSON error object on standard error.

## 5. Authoring discipline

Sub-skill SKILL.md files under skills/ are the source of truth and must not be modified by users of this router.

## A.2 Details of Constructing the ClinGeX Skill Library

## A.2.1 Skill Introduction

Each Agent Skill is a version-controlled directory organized as shown below. Lightweight metadata in SKILL.md supports discovery, while full instructions and optional resources are loaded only after selection. In ClinGeX, a skill represents a recurrent clinical-genomics operation and specifies the workflow and evidence needed to execute and interpret it safely.

Within this system-specific formulation, we retained two complementary taxonomies. The five construction action types—guideline following, entity mapping, data-source lookup, numerical calculation and model-based prediction—describe the clinical operation from which a prototype was discovered. The three deployment classes—guide, search and analysis—describe how the compiled skill is exposed at run time. The axes therefore need not coincide. For example, vep, mutalyzer3 and variant-validator were discovered as entity-mapping capabilities but are deployed under analysis because they perform executable transformations; conversely, ontology and identifier-mapping services are deployed under search. This mapping recovers the 27/13/30/12/23 construction counts reported in the main Methods while preserving the deployed 27/40/38 guide/search/analysis directory counts.

## A.2.2 Resource Discovery across the Five Capability Categories

Discovery was frozen on 28–30 July 2026 and performed separately for the five action types (Supplementary Table 1). A raw source record, normalized candidate and retained skill prototype were distinct units; records could be merged or one document family could support multiple operations, so successive counts are not simple retention rates.

Category boundaries followed the primary operation: representational conversion was entity mapping, evidence retrieval was data-source lookup, reproducible formulas or decision rules were numerical calculation, and learned outputs were model-based prediction. Conservative record linkage favoured residual duplication over false merging, and the predictor landscape was missense-enriched rather than complete across variant classes.

## A.2.3 Category-Specific LLM Triage and Expert Selection

LLMs were used to structure and prioritize the candidate space, not to make the final inclusion decision. Criterion scores were returned through structured schemas and combined deterministically using the weights in Supplementary Table 2. Numerical screens used the same prespecified tiers: Tier 1, ≥ 75; Tier 2, 55–74; Tier 3, 35–54; and exclusion, < 35. Scores supported ranking within an action type and were not compared across action types.

Maintenance and citation or adoption could not always be established from static catalogue metadata and were flagged for live verification rather than inferred. For data-source-lookup candidates, a score of zero for domain centrality, guideline recognition or programmatic access forced a one-tier downgrade. Guideline documents, calculations and predictors were not forced into the database rubric because database accessibility is not an appropriate proxy for normative authority, mathematical reproducibility or clinical calibration.

Eight clinical and domain experts from seven institutions reviewed the prioritized candidates: Xinhua Hospital; the Department of Obstetrics and Gynaecology, Faculty of Medicine, The Chinese University of Hong Kong; Shanghai Ninth People’s Hospital; Shanghai Sixth People’s Hospital; Fudan University Shanghai Cancer Center; Shanghai Chest Hospital; and Shanghai Jiao Tong University School of Medicine. Reviewers considered clinical utility, compatibility with authoritative standards, complementarity, unmet need, legitimacy and stability of access, documentation, technical reliability and functional redundancy. They could retain a lower-tier candidate that filled an uncovered clinical operation, remove a higher-tier but redundant or non-invocable candidate, or consolidate several candidates into one prototype. This process froze 105 prototypes: 27 guideline-following, 13 entity-mapping, 30 data-source-lookup, 12 numerical-calculation and 23 model-based-prediction skills.

Supplementary Table 1 | Source landscapes, acquisition routes and handof from resource discovery to skill-prototype selection.
<table><tr><td>Action type</td><td>Candidate unit</td><td>Principal sources and acquisition route</td><td></td><td>Discovery output</td><td>Priority set Frozen</td><td>proto- types</td></tr><tr><td>Guideline fol- Normative lowing</td><td>document family</td><td></td><td>doc- Organization-first scan: four federation directories 151 ument or linked yielded 106 unique organizations; 7 target bodies plus mentions; 9 gap-identified groups were searched through official unique institutions; supers- sites and PubMed/PMC; linked procedural documents 195 documents retained as families</td><td></td><td>institutional Authority, 106 currency, ession, accessibil- ity</td><td>27 verifiability,</td></tr><tr><td>Entity ping</td><td></td><td>tion tool</td><td>map- Standard, ontology, GA4GH GKS (11 records), EDAM-filtered bio.tools 3,553 raw records; 31 Tier 1 13 nomenclature ser- (3,256), NAR Web Server Issues 2023–2026 (276), and 3,508 normalized candidates vice or transforma- HPO/Monarch (10); required conversion of clinical- candidates genomics entities or representations rather than evidence</td><td></td><td></td><td>cutoff</td><td></td></tr><tr><td>Data-source lookup</td><td>base service</td><td>or</td><td>Database, repos- Complete harvests of the NAR Database Issue (2,594), 45,364 catalogue 211 Tier 1 30 itory, knowledge bio.tools (33,203), re3data.org (3,516) and FAIRshar- records; search ing.org (6,051); identifiers, names and URLs linked con- linked servatively; interface, access, licence, maintenance and clusters output metadata retained</td><td></td><td>resource</td><td>41,000 clusters</td><td></td></tr><tr><td>culation</td><td>gorithm</td><td></td><td>Numerical cal- Deterministic for- Reverse mapping from 195 normative documents plus 53 distinct candi- 19 Tier 1 12 mula, score, lookup targeted clinical-genomics enumeration; 30 listed meth- date methods table or decision al- ods and 23 additional normative requirements consoli- dated when mathematically equivalent; explicit inputs</td><td></td><td></td><td>methods</td><td></td></tr><tr><td>Model-based prediction</td><td>signal</td><td>Trained model or dbNSFP/ClinGen distinct predictive VEP/ANNOVAR/OpenCRAVAT</td><td>and a reproducible rule required SVI (51 (74), CAGI/ProteinGym (80); duplicate access routes models for the same signal consolidated; target, variant class, calibration, benchmark, deployment, maintenance,</td><td>records), 205 source records; 18 Tier 1 23 and 144 consolidated models</td><td></td><td></td><td></td></tr><tr><td>Total</td><td></td><td></td><td>licence and provenance retained</td><td>Heterogeneous discovery not arithmetically pooled</td><td>units;</td><td></td><td>105</td></tr></table>

Abbreviations: HUGO, Human Genome Organisation; ASHG, American Society of Human Genetics; ESHG, European Society of Human Genetics; IFHGS, International Federation of Human Genetics Societies; GA4GH, Global Alliance for Genomics and Health; GKS, Genomic Knowledge Standards; HPO, Human Phenotype Ontology; SVI, Sequence Variant Interpretation; CAGI, Critical Assessment of Genome Interpretation. Tier assignments prioritized expert review and were not mandatory inclusion thresholds.

The scan was a frozen, structured landscape survey rather than a preregistered systematic review. Coverage difered across action types, conservative record linkage could retain residual duplicates, and most LLM triage scores were not independently repeated. These limitations constrain claims of landscape completeness but do not alter the identity or counts of the final frozen prototypes.

## A.2.4 Details of Skill Initialization and Iterative Optimization

Each operation retained after expert review was initialized as a self-contained skill by a builder agent. The builder was required to examine how the underlying guideline, database, tool, calculation or predictive model could be used across clinical-genomics workflows before defining the skill contract. The common initialization prompt is shown below; source documents, technical documentation and the intended operation were supplied with the prompt for each skill.

Supplementary Table 2 | Category-specific criteria used for candidate triage before expert review.
<table><tr><td>Action type</td><td>Screening input and im- Criteria and weights plementation</td><td></td><td>Output before expert review</td></tr><tr><td></td><td>Guideline following Normative-document text; rather than the generic</td><td>Issuing-body authority, document type, currency, supersession, ci- 195 documents re- metadata and source tation verifiability, accessibility, clinical scope and faithful repre- viewed; no common direct review sentability of the operative rule</td><td>numerical cutoff</td></tr><tr><td>Entity mapping</td><td>sessment per record af- mation ter a 40-record pilot</td><td>Registry metadata; one Same 12 dimensions as data-source lookup, with the target operation 31 Tier 1 candidates structured GPT-5.5 as- defined as nomenclature, coordinate, format or ontology transfor- from 3,508 normalized</td><td>candidates</td></tr><tr><td></td><td>rule-filtered record</td><td>Data-source lookup Registry metadata; one Domain centrality (15), clinical actionability (12), human orienta- 211 Tier 1, 939 structured GPT-5-mini tion (8), guideline recognition (15), subdomain mapping (5), pro- Tier 2, 2,600 Tier 3 assessment per non- grammatic access (10), maintenance (7), access/licensing (6), query and 37,250 excluded definability (4), incremental value (3), authoritative stewardship clusters</td><td></td></tr><tr><td>tion</td><td></td><td>(9), and citation/adoption (6) Numerical calcula- Method description and Explicit guideline requirement (20), official thresholds or weights 19 Tier 1, 25 Tier 2 linked normative provi- (15), deterministic algorithm (20), availability of clinical inputs (12), and 9 Tier 3 methods sions; category-specific independent implementation (8), breadth (10), and consequence of</td><td></td></tr><tr><td>Model-based pre- Model, diction</td><td>structured assessment error (15)</td><td>calibration, Domain centrality (12), guideline-calibrated evidence strength (18), 18 Tier 1, 35 Tier 2, benchmark and de- independent benchmark performance (12), variant-class breadth (6), 88 Tier 3 and 3 ex- ployment metadata; precomputed scores (14), pipeline integration (10), maintenance (6), cluded models category-specific struc- access/licence (5), stewardship (9), and citation/adoption (8)</td><td></td></tr></table>

## Prompt used to initialize a ClinGeX skill

You are responsible for converting the supplied clinical-genomics resource or operation into a reliable, reusable agent skill. Study the source material and technical documentation before writing any files. Do not merely summarize the resource. Determine how it may be used in clinical practice and encode the procedures needed to use it correctly.

1. Characterize the resource. Establish its authoritative or evidentiary status, intended users, supported clinical domains and variant or data types, access method, version or update status, and any licence or authentication constraints. Distinguish normative rules, source assertions, deterministic calculations and model-derived predictions.

2. Enumerate applicable use cases. Consider all clinically plausible uses supported by the source, including direct lookup or calculation, interpretation of returned evidence, batch use, integration with upstream normalization and downstream classification, and use in incomplete or conflicting-evidence settings. Do not invent applications that are not supported by the source.

3. Define boundaries and dependencies. State when the skill should and should not be used, the required inputs and accepted representations, applicability restrictions, prerequisites, expected outputs, uncertainty, known limitations and failure conditions. Identify any guideline or companion skill required to interpret the result safely.

4. Construct the skill package. Create a directory containing a required SKILL.md and, where needed, scripts/, references/ and assets/. The SKILL.md must include valid front matter, concise discovery metadata, scope, use-when and do-not-use conditions, supported actions, required and optional inputs, a stepwise workflow, output interpretation, provenance requirements, limitations, error handling and brief examples. Place long criteria, source excerpts or lookup tables in references/; place reusable templates or static data in assets/.

5. Implement executable actions where appropriate. Scripts must use documented interfaces or reproducible published algorithms, validate inputs, preserve source provenance, return structured outputs, distinguish not\_found from not\_applicable and execution failure, support batch inputs where clinically useful, and never fabricate a result when an external service or required field is unavailable.

6. Prepare verification cases. Define tests before revision that collectively cover every exposed action, representative valid inputs, boundary values, alternative input representations, batch and multi-record responses, non-applicable cases, absent records, malformed inputs and anticipated external-service failures. For each test, specify the input, expected output fields or behaviour and an explicit pass criterion.

Return the complete directory structure and file contents. Ensure that every claimed capability is supported by the supplied source, exposed through the documented interface and testable through the proposed verification cases.

Depending on the resource, initialization produced procedural instructions alone or combined them with executable scripts, indexed references and static assets. A separate test agent then received the compiled SKILL.md and ran 5–20 predefined cases through the same action interface used at run time. Failed cases triggered revision of the instructions, interface, implementation or limitation statement followed by rerunning the complete suite; iteration stopped when no actionable failure remained or after five cycles.

Supplementary Table 3 shows the complete verification suite for one representative analysis skill, REVEL. The suite was selected because it jointly tests external retrieval, variant applicability, alternative coordinate systems, multi-allelic and batch handling, missing results and clinically calibrated interpretation.

Supplementary Table 3 | Complete end-to-end verification suite for the representative REVEL skill.
<table><tr><td>ID</td><td>Action and input</td><td>Prespecified pass criterion</td><td>Observed result</td><td>Verdict</td></tr><tr><td>S1</td><td>Query chr7:g.140453136A&gt;T</td><td>applicable=true; REVEL score &gt; 0.9 Applicable; REVEL score 0.931</td><td></td><td>PASS</td></tr><tr><td>S2</td><td>Query synonymous variant rs1799949</td><td>applicable=false</td><td>Not applicable; no REVEL score returned</td><td>PASS</td></tr><tr><td>S3</td><td>Query chr1:g.999999999G&gt;A</td><td>not_found=true</td><td>Variant absent from dbNSFP; not_found=true</td><td>PASS</td></tr><tr><td>S4</td><td>Query multi-allelic rs113488022</td><td>les</td><td>disambiguated=true; at least two alle- Three alleles returned (A&gt;G, A&gt;T and A&gt;C)</td><td>PASS</td></tr><tr><td>S5</td><td>Batch query: S1 variant plus S3 variant</td><td>not found</td><td>Two results; first applicable, second Two ordered results; score 0.931 for the first PASS and not_found=true for the second</td><td></td></tr><tr><td>S6</td><td>Interpret REVEL score 0.95</td><td>acmg_evidence=PP3_Strong</td><td>PP3 Strong; calibrated threshold ≥ 0.932</td><td>PASS</td></tr><tr><td>S7</td><td>Interpret REVEL score 0.01</td><td>acmg_evidence=BP4_Strong</td><td>BP4 Strong; calibrated threshold ≤ 0.016</td><td>PASS</td></tr><tr><td>S8</td><td>Interpret REVEL score 0.50</td><td>acmg_evidence=Indeterminate</td><td>Indeterminate; no PP3 or BP4 assignment</td><td>PASS</td></tr><tr><td>S9</td><td>Query hg38 chr7:g.140753336A&gt;T</td><td></td><td>applicable=true; REVEL score &gt; 0.9 hg38 coordinate resolved; REVEL score 0.931</td><td>PASS</td></tr></table>

## A.3 Details of Building the ClinGeX Skill Graph

## A.3.1 Semantic Representations

For every skill, the compact Scope and Use when fields were parsed from the three registry documents. Softwrapped continuation lines were joined, and extraction failed if either field was missing. The skill identifier and deployment category were retained as metadata but were not included in the text to be embedded. The exact embedding input was therefore

Scope: <scope text>

Use when: <use-when text>

Each input was encoded with Qwen text-embedding-v4 using its default 1,024-dimensional output. Requests were made in batches of ten, and vectors were stored as a float32 matrix whose row order is fixed by embedding\_index in skills.json. The deployed selector checks that the number of metadata rows equals the first dimension of the matrix before computing similarities. Because query and skill vectors must occupy the same vector space, the online query encoder is pinned to the same model; changing the embedding model requires regeneration of the entire matrix.

The embedding layer is used for task-to-skill seed retrieval, not as the sole definition of graph edges. This distinction is important: cosine proximity captures similarity between a requested need and the declared applicability of a skill, whereas it cannot by itself determine whether two skills substitute for one another, provide independent complementary evidence, or have a hard workflow prerequisite.

## A.3.2 Exhaustive Pairwise Relationship and Dependency Scoring

All ${ \binom { 1 0 5 } { 2 } } = 5$ , 460 unordered skill pairs were evaluated. The judge received the complete SKILL.md bodies of Skill A and Skill B rather than the lossy compact registry records. It assigned five independent integer scores from 0 to 4, with fixed anchors, and one categorical dependency label. The symmetric relationship score was computed deterministically rather than requested from the model:

$$
R _ { A B } = P _ { A B } + I _ { A B } + S _ { A B } + C _ { A B } + J _ { A B } , \qquad R _ { A B } \in [ 0 , 2 0 ] ,\tag{1}
$$

where P is purpose similarity, I is applicable-input overlap, S is substitutability, C is complementary gain from using both, and J is joint use or downstream joint consumption. The dimensions were scored independently: in particular, complementarity is not defined as the inverse of substitutability. The anchors and operational meanings are given in Table 4.

Supplementary Table 4 | Anchors used for pairwise Skill Graph scoring.
<table><tr><td>Dimension</td><td>0</td><td>1</td><td>2</td><td></td><td></td><td>4</td></tr><tr><td>Purpose similarity</td><td>Unrelated questions Distant</td><td>overlap</td><td></td><td>subject Overlapping ques- Largely tion, different angle question, different same question</td><td></td><td>same Essentially the</td></tr><tr><td>Input overlap</td><td>Inputs barely inter- Minor intersection Moderate overlap sect</td><td></td><td></td><td>inputs</td><td></td><td>Mostly overlapping Nearly identical in- put domain</td></tr><tr><td>Substitutability</td><td>Not interchangeable Rarely interchange- Partly interchange- Largely interchange- Fully redundant</td><td>able</td><td>able</td><td>able</td><td></td><td></td></tr><tr><td>Complementarity</td><td>No added value to- Slight added value gether</td><td></td><td>dent value</td><td>fidence/coverage</td><td>sential</td><td>Moderate indepen- Strong gain in con- Joint use close to es-</td></tr><tr><td>Joint usage</td><td>Essentially never co- Occasional used</td><td>occurrence</td><td>co- Common pairing</td><td>downstream</td><td>Frequently merged Near-inseparable in</td><td>workflow</td></tr></table>

The dependency label was one of none, A\_requires\_B, B\_requires\_A or mutual. A direction was assigned only when using the dependent skill without first consulting or invoking the other would make its result unobtainable, incomplete or invalid—either because the other supplies a required input or because it supplies the rule under which the output must be interpreted. Merely helpful context, common co-use or higher confidence was not suficient. Deployment class was hidden as a scoring signal so that same-directory pairs received no automatic advantage.

GPT-5.5 with high reasoning efort performed the scoring through a strict structured-output schema. The model returned the five subscores, dependency enum and a short rationale; Python validated ranges and types and summed the subscores. Calls used 32-way concurrency, exponential retry for transient/API failures (up to five attempts) and a separate structured-output validation retry (up to three attempts). Records were streamed to an append-only resumable JSONL file. The completed artefact contained all 5,460 pairs, no failed or duplicate pair, and no mutual dependency. Mean (s.d.) R was 5.45 (2.99), the median was 5 and the observed range was 0–16. The 28 directed dependencies comprised 19 A\_requires\_B and nine B\_requires\_A labels.

The edge list was compiled into 105 node records. Every node stores (i) requires, its outgoing hard prerequisites; (ii) required\_by, incoming hard dependencies; and (iii) all other 104 skills in related, sorted by decreasing R and then by identifier, with the five subscores and dependency label retained for audit. Seventeen nodes have at least one outgoing prerequisite and 19 have at least one incoming dependency. Relationship bands were defined as strong $( R = 1 5  – 2 0 )$ , moderate $( R = 8 – 1 4 )$ , weak (R = 1–7) and none (R = 0); the deployed recall threshold deliberately includes both strong and moderate relationships.

## A.3.3 Per-Message Seed Retrieval and Graph Expansion

Selection is recomputed once for every user message, including follow-up messages, rather than once for an entire session or once per internal tool call. The selector receives the current message plus at most the previous two question–answer pairs; each prior answer is truncated to 800 characters for this purpose. A fixed

DeepSeek-V4-Flash decomposition step with thinking disabled returns strict JSON containing one to three short, non-overlapping English need statements. Each need must be expressed in the same objective register as a skill’s Scope/Use-when description and may represent phenotype normalization, a distinct variant type, gene–disease mechanism, or a requested classification task.

Each need is embedded independently with the pinned Qwen model. After $L _ { 2 }$ normalization, cosine similarity is calculated between all 105 skill vectors and each need vector. If $E _ { i }$ is the stored vector of skill i and $Q _ { k }$ is need k, the internal seed score is

$$
s _ { i } = \operatorname* { m a x } _ { k } \frac { E _ { i } ^ { \mathsf { T } } Q _ { k } } { \| E _ { i } \| _ { 2 } \| Q _ { k } \| _ { 2 } } .\tag{2}
$$

The maximum, rather than a mean across needs, allows a specialized skill to be retained when it matches one facet of a multi-part request. The top $n = 3 0$ skills by $s _ { i }$ form the seed set; no absolute cosine cutof is applied in the deployed selector.

For each seed, all one-hop related edges with $R \geq x = 8$ are added to a shared candidate pool. If a candidate is linked to more than one seed, only its largest edge score and the corresponding seed are retained. Candidates are ordered by decreasing edge score with identifier as the deterministic tie-breaker, and at most $m = 1 5$ are added. The expansion is intentionally one hop: newly expanded nodes do not themselves generate further similarity candidates. Independently, pubmed-search, pmc-search and tavily-search are always included because literature and oficial-web discovery are broadly useful; these forced skills do not act as expansion seeds. Finally, the outgoing requires list of every recalled node is traversed repeatedly until a fixed point is reached. Hard prerequisites are therefore included regardless of cosine rank, relationship threshold or the 15-skill expansion cap.

In compact form, if S is the top-30 seed set, F the three forced literature skills and $T _ { 1 5 }$ the operator retaining the 15 highest-scoring eligible candidates, the recalled set is

$$
A _ { 0 } = { S \cup F \cup T _ { 1 5 } } \left( \bigcup _ { i \in S } \{ j : R _ { i j } \geq 8 \} \right) , \qquad A = \mathrm { c l o s u r e } _ { \mathrm { r e q u i r e s } } ( A _ { 0 } ) .\tag{3}
$$

## A.4 Details of the Automated Benchmark

## A.4.1 Benchmark Composition and Case Construction

The benchmark used expert-curated records with discrete reference classifications and reconstructable public evidence. Eligible records were released or substantively updated after October 2025. For each case, the source identifier and version date, clinical input, supporting literature, reference answer, and scoring annotations were frozen before system evaluation. As the training-data cutof dates of all evaluated models preceded October 2025, this temporal restriction ensured that the case-specific curation records and final reference classifications were held out from model training.

Supporting PMIDs were deduplicated within each source record and resolved to the local literature collection through PMID and DOI mappings. Available articles were converted to normalized plain text or Markdown while retaining headings and paragraph boundaries. Inclusion was assessed at the case level: the processed collection had to contain the public literature needed to reconstruct the reference answer. The same processed texts were supplied to every system, whereas source classifications, interpretation summaries, evidence scores and other ground-truth fields were stored separately and withheld.

For SNV interpretation, a public-evidence audit reduced 2,185 ClinGen expert-panel records to 1,683; requiring complete literature coverage yielded 300 eligible cases. Stratified sampling produced 79 cases (17 Pathogenic, 17 Likely Pathogenic, 17 Uncertain Significance, 12 Likely Benign and 16 Benign). The hidden reference retained the expert-panel classification, interpretation and criterion-level evidence annotations.

Supplementary Table 5 | Composition of the system-level clinical-genomics benchmark.
<table><tr><td>Task</td><td>Source</td><td>n</td><td>Principal information supplied to the sys- Principal reference outputs tem</td><td></td></tr><tr><td>SNV inter- pretation</td><td>ClinVar/ClinGen expert-panel records</td><td></td><td>79 Variant, disease, mode of inheritance and ACMG/AMP evidence criteria, evidence supporting literature</td><td>strengths, final five-tier pathogenicity clas- sification and interpretation summary</td></tr><tr><td>Gene- disease validity</td><td>ClinGen</td><td></td><td>66 Gene, disease, mode of inheritance, appli- Genetic- and experimental-evidence items cable curation framework and supporting and scores, final validity classification and literature</td><td>evidence summary</td></tr><tr><td>Clinical actionability</td><td>CIViC</td><td></td><td>type and supporting literature</td><td>46 Molecular profile, disease, therapy, pheno- Assertion type, significance, AMP category, assertion direction and clinical assertion</td></tr><tr><td></td><td>OncoKB</td><td></td><td>ing literature</td><td>12 Gene alteration, cancer type and support- OncoKB evidence level, associated drug or regimen, FDA level where applicable and clinical interpretation</td></tr><tr><td></td><td></td><td></td><td>type, specialty population and supporting dence level and interpretation summary</td><td>PharmGKB/ClinPGx 20 Variant or haplotype, gene, drug, pheno- Phenotype category, evidence score, evi-</td></tr><tr><td>Total</td><td>Five resources</td><td>223</td><td>literature</td><td></td></tr></table>

<table><tr><td>Representative complete case: SNV interpretation</td></tr><tr><td>Source record: ClinVar Variation ID 127692, released 5 December 2025. Case input: NM 000314.8(PTEN):c.79+7A&gt;G; PTEN hamartoma tumour syndrome; autosomal dominant inheritance. Evidence supplied: processed full text of PMID 28677221. The study reported that the intronic c.79+7A&gt;G</td></tr><tr><td>substitution had no demonstrable effect on splicing. Required output: a five-tier ACMG/AMP classification, the criteria and effective strengths applied, and a concise evidence-based interpretation. Frozen reference answer: Likely Benign. The PTEN Variant Curation Expert Panel specification (version</td></tr><tr><td>3.1.0) applied BS3 because an RNA or minigene splicing assay demonstrated no splicing impact. The sole criterion marked as met was BS3 at its default Strong strength; PM2, BS1, PP3/BP4, BP7 and the remaining assessed criteria were not met. Reference interpretation: “The NM 000314.8(PTEN):c.79+7A&gt;G variant meets criteria to be classified as</td></tr></table>

For gene–disease validity, up to 15 eligible ClinGen records were sampled per final class (fixed seed 42), with all records retained from smaller classes. The 66 cases comprised 15 Definitive, 10 Strong, 15 Moderate, 15 Limited, 2 No Known Disease Relationship, 6 Disputed and 3 Refuted relationships. Counted genetic and experimental points were summed from the structured ClinGen evidence tables, rounded to four decima places and retained only as hidden reference annotations.

<table><tr><td>Representative complete case: gene-disease validity</td></tr><tr><td>Source record: ClinGen assertion for GRAP-autosomal recessive nonsyndromic genetic hearing loss, evaluated under Gene Curation SOP 11. Case input: gene, GRAP; disease, nonsyndromic genetic hearing loss; mode of inheritance, autosomal recessive.</td></tr><tr><td>Evidence supplied: processed full text of Li et al. (2019; PMID 30610177).</td></tr><tr><td>Genetic evidence: one female proband with congenital profound sensorineural hearing impairment carried homozygous GRAP c.311A&gt;T (p.Gln104Leu). The proband contribution was reduced from the default 1 point to 0.5 points because of consanguinity and homozygosity.</td></tr><tr><td>Representative complete case: clinical-actionability classification</td></tr><tr><td>Source record: CIViC assertion 9, last reviewed 9 March 2026. Case input: molecular profile, ACVR1 G328V; disease, diffuse midline glioma, H3 K27M-mutant; no therapy or phenotype field was applicable to this diagnostic assertion. Evidence supplied: processed full texts of Fontebasso et al. and Buczkowicz et al. (PMIDs 24705250 and 24705254). The first study identified activating ACVR1 mutations in five of 39 paediatric midline high- and a justified clinical assertion. Frozen reference answer: assertion type, Diagnostic; significance, Positive; AMP category, Tier II-Level</td></tr><tr><td>grade astrocytomas and demonstrated increased phospho-SMAD1/5/8 signalling. The second identified ACVR1</td></tr><tr><td>alterations in 12 of 61 diffuse intrinsic pontine gliomas, including five G328V tumours, and showed activation of BMP signalling and enhanced astrocyte growth. Required output: assertion type, significance, AMP/ASCO/CAP category where applicable, assertion direction</td></tr></table>

scolopidia and loss of sensory nerve terminals (2 points); mouse inner-ear immunostaining localized Grap to spiral-ganglion fibres and auditory and utricular hair cells (0.5 points). The reference experimental total was therefore 2.5 points.

Required output: a permitted ClinGen validity category, genetic- and experimental-evidence totals and a concise justification.

Frozen reference answer: Limited; genetic evidence, 0.5 points; experimental evidence, 2.5 points. A single missense variant had been reported in four individuals from two unrelated families within one publication, the disease mechanism remained uncertain and no independent publication had replicated the association. The source panel therefore concluded that the relationship had limited supporting evidence and no convincing contradictory evidence.

Clinical actionability retained the native source ontologies. The 46 CIViC cases comprised 20 Diagnostic, 12 Predictive, 10 Oncogenic, 3 Predisposing and 1 Prognostic assertions (33 with an applicable AMP category); the 12 OncoKB cases comprised four LEVEL\_1, two LEVEL\_2, four LEVEL\_3A, one LEVEL\_4 and one LEVEL\_R2; and the 20 PharmGKB cases comprised one Level 1A, sixteen Level 3 and three Level 4. CIViC Diagnostic assertions were downsampled with seed 42; the other subsets were selected after source-specific eligibility filtering.

Reference-answer verification. Reference answers were inherited from versioned source records and were not model-generated. Three physicians with more than five years of relevant experience reviewed all 223 cases for entity–record correspondence, reference-label and annotation accuracy, literature coverage, ground-truth leakage, anonymization and agreement between the task and scoring schema. Ambiguities were resolved before freezing, and system outputs did not alter the references. Only spelling, capitalization, whitespace and unambiguous aliases were normalized across databases.

## A.4.2 System Inputs, Output Schemas and Execution Protocol

All systems received the same case-level clinical entities, task instructions and processed literature, with no cross-case history. Ground-truth fields were withheld; direct access to the five source databases was prohibited. Individual papers were limited to 80,000 characters and combined literature to 200,000 characters. System configurations are given in Supplementary Table 7.

Because some systems provided their own search or database tools, we additionally used an LLM to audit each complete execution record for substantive case-specific information obtained from the corresponding reference resource, either directly or indirectly through an intermediary. Mere mention of a resource or use of general framework terminology was not counted. Positive records were uncommon for every system and task (Supplementary Table 6).

Supplementary Table 6 | Audit of substantive information acquisition from benchmark reference resources.
<table><tr><td>System</td><td>SNV interpretation: ClinVar/ClinGen</td><td>Gene-disease validity: ClinGen</td><td>Clinical actionability: CIViC/OncoKB/PharmGKB</td></tr><tr><td>OpenBioLLM</td><td>0/79 (0%)</td><td>0/66 (0%)</td><td>0/78 (0%)</td></tr><tr><td>OpenAI DeepResearch</td><td>3/79 (3.8%)</td><td>2/66 (3.0%)</td><td>2/78 (2.6%)</td></tr><tr><td>SciSpace</td><td>1/79 (1.3%)</td><td>1/66 (1.5%)</td><td>3/78 (3.8%)</td></tr><tr><td>Biomni</td><td>4/79 (5.1%)</td><td>1/66 (1.5%)</td><td>1/78 (1.3%)</td></tr><tr><td>TxAgent</td><td>0/79 (0%)</td><td>2/66 (3.0%)</td><td>1/78 (1.3%)</td></tr><tr><td>ClinGeX</td><td>0/79 (0%)</td><td>0/66 (0%)</td><td>0/78 (0%)</td></tr></table>

Values are the number of execution records with direct or indirect acquisition of substantive case-specific information from the corresponding reference resource divided by the number of cases in that task.

Supplementary Table 7 | Systems and underlying language models used in the system-level comparison.
<table><tr><td>System</td><td>Underlying LLM used</td><td>System type</td><td>Benchmark configuration</td></tr><tr><td>OpenBioLLM</td><td>OpenBioLLM-70B (derived from Llama 3 70B)</td><td>Domain-adapted biomedical LLM</td><td>Model response generated without an external agent framework.</td></tr><tr><td>OpenAI DeepResearch</td><td>o3-deep-research</td><td>Commercial deep-research agent</td><td>OpenAI Responses interface with native web search and code interpreter; access to the five ground-truth databases was prohibited.</td></tr><tr><td>SciSpace</td><td>Proprietary LLM backend (not disclosed by provider)</td><td>Commercial literature-research system</td><td>Provider-hosted interface; the backend model could not be independently configured or verified.</td></tr><tr><td>Biomni</td><td>DeepSeek-V4 (deepseek-v4-pro)</td><td>Open-source biomedical code-execution agent</td><td>Biomni A1 with its tool retriever enabled; a new A1 instance was created for each case.</td></tr><tr><td>TxAgent</td><td>DeepSeek-V4 (deepseek-v4-pro)</td><td>Open-source ToolUniverse agent</td><td>Original TxAgent orchestration layer retained; inference backend replaced with the same DeepSeek model used by ClinGeX.</td></tr><tr><td>ClinGeX</td><td>DeepSeek-V4 (deepseek-v4-pro)</td><td>Skill-centred clinical-genomics agent</td><td>Full skill library and orchestration workflow.</td></tr></table>

Archived execution records verified the backends for OpenAI DeepResearch, Biomni, TxAgent and ClinGeX. SciSpace exposed no model identifier and is therefore reported as undisclosed; the archived materials did not specify the OpenBioLLM repository revision or numerical precision.

The common case-facing instruction layer is reproduced below. Framework-specific tool-calling scafolds difered, but clinical fields, literature and required outputs were controlled across systems.

```markdown
Common benchmark instruction and user-prompt template
You are a clinical genomics expert. Use the provided literature and your expert knowledge to
answer accurately. Never fabricate data. When you cite literature, be specific about which
paper and which finding.
==== TASK-SPECIFIC INSTRUCTIONS ==========
{source-specific task description from the frozen task record}
## Case Inputs
- {task-specific clinical entities}
## Literature
The following papers are provided for this case. Read them carefully and use the evidence
```

within, together with your expert knowledge, to reach a well-reasoned conclusion.   
### Paper: {source path}   
{processed full text; per-paper and total-context limits applied}   
## Required Output   
{source-specific final classification, evidence fields and concise justification}

The system-specific execution controls recorded in the archived harnesses are summarized in Supplementary Table 8. Free-text answers were subsequently converted into task-specific structured fields by a fixed extraction pipeline. Missing or unrecognized fields were retained as null values rather than repaired from the reference answer.

Supplementary Table 8 | Execution controls recorded in the system-level benchmark harnesses.
<table><tr><td>System</td><td>Agent or tool budget</td><td>Request and recovery controls</td></tr><tr><td>OpenBioLLM</td><td>Direct model generation; no external agent loop.</td><td>The exact checkpoint revision, numerical precision and generation configuration were not recorded in the archived manifest</td></tr><tr><td>OpenAI DeepResearch</td><td>At most 30 combined native web-search and code- Jobs were polled every 5 s for at most 2,700 s; the interpreter calls per job.</td><td>client timeout was 3,600 s. Transient requests used</td></tr><tr><td></td><td>Provider-managed literature-search workflow.</td><td>up to five attempts, and failed background jobs could be recreated up to six times. Tool budget, timeout and recovery policy were not</td></tr><tr><td>SciSpace Biomni</td><td>Tool retriever enabled; a fresh A1 instance for every LLM request timeout, 180 s.</td><td>exposed by the provider.</td></tr><tr><td>TxAgent</td><td>case; individual executed code blocks limited to 600 s. generated tokens per model call; nominal context exponential backoff.</td><td>At most 50 multistep rounds; maximum 2,048 newly LLM request timeout, 180 s; up to five attempts with</td></tr><tr><td></td><td>ceiling, 90,240 tokens; temperature, 0.3.</td><td>LLM request timeout, 180 s; up to five attempts with</td></tr><tr><td>ClinGeX</td><td>At most 50 agent turns.</td><td>exponential backoff.</td></tr></table>

Final-classification and evidence metrics were computed from the frozen single-output records described below. The repeated-run consistency and per-case wall-clock time reported in the main figure were obtained through a separate repetition-and-timing workflow and were not derived from those single-output metric files.

## A.4.3 Metric Implementation and Denominators

All reported classification and evidence metrics were recomputed from the frozen structured reference data and extracted system outputs by deterministic code. The denominators difer by task and, in two cases, by endpoint; they are summarized in Supplementary Table 9. No LLM judge determined whether a final classification matched the reference.

## A.4.4 SNV-Specific Evaluation Metrics

Final-classification concordance required an exact five-tier match. Pathogenic/Likely Pathogenic and Benign/Likely Benign formed the positive and negative groups for sensitivity and specificity; VUS cases were excluded. A significant error was a direction reversal between these groups and used all 79 cases as its denominator.

For reference criterion set $E _ { i }$ and recovered set $\hat { E } _ { i }$ , micro recall was $\textstyle \sum _ { i } | E _ { i } \cap \hat { E } _ { i } | / \sum _ { i } | E _ { i } |$ and macro recall was the mean case-level recall; the micro value is the weighted evidence recall reported in the main figure. A criterion counted only when asserted as met for the evaluated variant–disease pair. Strength concordance required exact strength agreement among correctly recovered criteria.

Supplementary Table 9 | Implemented metric denominators in GeneBench.
<table><tr><td>Task</td><td>Endpoint</td><td>Implemented denominator</td></tr><tr><td>SNV</td><td>Final concordance and significant-error rate</td><td>All 79 cases; missing or unparsable predictions remain incorrect observations.</td></tr><tr><td>interpretation</td><td>Sensitivity / specificity</td><td>The 34 pathogenic or likely pathogenic references / the 28 benign or likely benign references, respectively; VUS references are not in either binary denominator.</td></tr><tr><td></td><td>Evidence recall/ strength concordance</td><td>All reference criteria pooled, or the 79 case-level recall values; strength uses only correctly recalled criterion codes.</td></tr><tr><td>Gene-disease validity</td><td>Concordance and severity errors</td><td>Cases with a parsable prediction in the seven-category vocabulary  $( N _ { \mathrm { v a l i d } } \leq 6 6 )$  reported separately for each system.</td></tr><tr><td></td><td>Score coverage / score MAE</td><td>Coverage denominator,  $N _ { \mathrm { v a l i d } } ;$  MAE denominator, cases in which the relevant numerical score or interval was stated.</td></tr><tr><td>Clinical</td><td>Final concordance and</td><td>The 65 tier-applicable cases: 33 CIViC, 12 OncoKB and 20 PharmGKB.</td></tr><tr><td>actionability</td><td>severity errors Complete-answer accuracy</td><td>All 78 cases.</td></tr></table>

## A.4.5 Gene–Disease Validity-Specific Evaluation Metrics

The fixed ranks were Refuted = −2, Disputed = −1, No Known Disease Relationship = 0, Limited = 1, Moderate = 2, Strong = 3 and Definitive = 4. Concordance and rank diferences used parsable seven-category outputs; diferences of at least two levels defined primary severe over- or under-classification (three levels in supplementary analyses).

Evidence “coverage” denotes reporting of a usable genetic or experimental score, not item-level recall. Reference scores were capped at 12 and 6 points, respectively. For an extracted interval $[ L _ { i } , H _ { i } ]$ and reference $g _ { i } ,$ , error was max $( 0 , L _ { i } - g _ { i } , g _ { i } - H _ { i } )$ ; MAE was conditional on a stated score, with no imputation.

## A.4.6 Clinical Actionability-Specific Evaluation Metrics

Final-classification concordance used the 65 cases with an applicable tier: CIViC AMP category, OncoKB level or PharmGKB level. Complete-answer accuracy used all 78 cases and required, respectively, CIViC assertion type/significance/direction and applicable AMP category; OncoKB level and exact drug set; or PharmGKB level and a score within 0.5 points of reference.

Severity analyses used the source-specific ranks in Supplementary Table 10. A rank diference of at least two levels defined primary severe over- or under-classification (three levels in supplementary analyses); missing or unmappable labels remained in the fixed 65-case denominator.

Supplementary Table 10 | Unified evidence-strength ranks implemented for clinical-actionability severity analyses.
<table><tr><td>Rank</td><td>CIViC AMP category</td><td>OncoKB level</td><td>PharmGKB level</td></tr><tr><td>6</td><td>Tier I-Level A</td><td>LEVEL_1</td><td>1A</td></tr><tr><td>5</td><td>Tier I-Level B</td><td>LEVEL_2; LEVEL_R1</td><td>1B</td></tr><tr><td>4</td><td>Tier II–Level C</td><td>LEVEL_3A</td><td>2A</td></tr><tr><td>3</td><td>Tier II–Level D</td><td>LEVEL_3B; LEVEL_R2</td><td>2B</td></tr><tr><td>2</td><td>Tier III</td><td>LEVEL_4</td><td>3</td></tr><tr><td>1</td><td>Tier IV</td><td>No response-category equivalent</td><td>4</td></tr></table>

The placement of OncoKB resistance categories is an analysis convention rather than an oficial crosswalk: LEVEL\_R1 was aligned with standard-care evidence at rank 5 and LEVEL\_R2 with emerging clinical evidence at rank 3. This convention was fixed in the evaluator before system comparison.

Rule-based scoring and missing outputs. All matches were determined from parsed outputs and frozen references using rules fixed before comparison; the extraction model only normalized labels, criteria and numerical intervals. Missing SNV and actionability predictions remained incorrect in fixed denominators, whereas unparsable gene–disease predictions were reported separately and excluded from concordance and severity denominators. Supplementary Table 9 gives all endpoint-specific denominators.

## A.4.7 Blinded Clinician Assessment of Report Quality

Nine clinicians rated the reports from ClinGeX, Biomni and DeepResearch for all 223 cases. Six clinicians with more than three years of genetic-analysis experience reviewed SNV or gene–disease reports; three internal-medicine physicians with more than five years of experience reviewed clinical-actionability reports.

Within each task, cases were partitioned among three reviewers (SNV: 27/26/26; gene–disease: 22/22/22;   
actionability: 27/26/25), with no duplicate rating. The assigned reviewer assessed all three reports for a case.   
Report identities were independently permuted to Result A–C for each case and revealed only after export.

Each report received eight ordered three-category ratings without a not-applicable option: six common dimensions (Supplementary Table 11) and two task-specific dimensions (Supplementary Table 12).

Supplementary Table 11 | Clinician-assessment dimensions shared across all three clinical-genomics tasks.
<table><tr><td>Dimension</td><td>Operational assessment</td><td>Ordered response categories</td></tr><tr><td>Overall acceptability</td><td>Whether the report&#x27;s conclusion and supporting interpretation could be accepted Yes; Partly; No for the stated clinical-genomics task after expert review.</td><td></td></tr><tr><td></td><td>Required revision The amount of substantive correction or rewriting required before the report could No revision; Partial be accepted. This was rated separately from acceptability.</td><td>revision; Substantial</td></tr><tr><td>Citation accuracy</td><td>Whether cited publications were identifiable and whether the findings attributed to Largely accurate; them were supported by the cited source.</td><td>revision Partly accurate;</td></tr><tr><td>Guideline adherence</td><td>Whether the reasoning, terminology and final conclusion followed the framework Largely adherent; applicable to the case: ACMG/AMP, ClinGen Gene Curation, CIViC, OncoKB or Partly adherent; Not</td><td>Inaccurate</td></tr><tr><td>Hallucinated</td><td>PharmGKB. Whether the report introduced fabricated or unsupported facts, numerical values, Absent; Partly present;</td><td>adherent</td></tr><tr><td>information Potential clinical risk</td><td>citations, study findings or clinical assertions. Whether errors or unsupported claims in the report could plausibly mislead variant Absent; Partly present classification, gene-disease assessment, therapeutic interpretation or another down- Present</td><td>Present</td></tr></table>

Completion and aggregation. All cases were complete: 669 reports and 5,352 ordinal ratings, with no missing value or imputation. After de-blinding, category counts and proportions were summarized by task, system and dimension. No composite score was defined; inter-rater agreement was not estimated because cases were partitioned rather than multiply rated. These ratings neither altered the references nor entered the automated metrics.

## A.5 Selection of the System Architecture

## A.5.1 Study Objective and Controlled Comparison

This experiment compared the complete ClinGeX architecture with an unenhanced DeepSeek-V4 baseline and five single-strategy enhancements on evidence-grounded question–answering items. DeepSeek-V4, item text and hidden-reference controls were held fixed, so conditions difered only in the enhancement or orchestration layer listed in Supplementary Table 13. Unlike the 223-case end-to-end benchmark, this dataset isolated specific evidence-reasoning operations.

Supplementary Table 12 | Task-specific dimensions in the clinician assessment.
<table><tr><td>Task</td><td>Dimension</td><td>Operational assessment</td><td>Ordered response categories</td></tr><tr><td rowspan="2">SNV interpretation</td><td>Evidence use</td><td>Whether the report selected and applied rel- Largely reasonable; Partly reasonable; Un- evant population, computational, functional, reasonable segregation, allelic and other evidence ap- propriately to the evaluated variant-disease</td><td></td></tr><tr><td>Evidence strength</td><td>pair. ACMG/AMP criteria, including any reasonable strength modification under the applicable expert-panel specification, were appropriate.</td><td>Whether the strengths assigned to Largely reasonable; Partly reasonable; Un-</td></tr><tr><td rowspan="2">Gene-disease validity</td><td>Genetic-evidence analysis</td><td>Whether proband, segregation, case-control Largely reasonable; Partly reasonable; Un- and other genetic evidence was identified, reasonable interpreted and weighted appropriately under the ClinGen framework.</td><td></td></tr><tr><td>Experimental- evidence analysis</td><td>cal function, expression, interaction, model- reasonable system and rescue evidence was identified, interpreted and weighted appropriately un- der the ClinGen framework.</td><td>Whether functional alteration, biochemi- Largely reasonable; Partly reasonable; Un-</td></tr><tr><td rowspan="2">Clinical actionability</td><td>Evidence relevance and completeness</td><td>Whether the selected evidence was relevant Largely relevant and complete; Partly rele- peutic context and whether material evidence needed for the clinical assertion was covered.</td><td>to the specific variant, disease, drug or thera- vant and complete; Irrelevant or incomplete</td></tr><tr><td>Evidence interpretation</td><td>preted correctly with respect to its direction, clinical significance, evidence tier, therapeu- tic association and source-specific conclusion.</td><td>Whether the selected evidence was inter- Largely correct; Partly correct; Incorrect</td></tr></table>

## A.5.2 Implementation of the Seven Framework Conditions

Supplementary Table 13 defines the seven conditions. The RAG condition exposed only the search-facing functions from the ClinGeX library; few-shot used the same three fixed demonstrations for every item; and the multi-agent condition used sequential planner, executor and reviewer roles. BioSkill (https://github.com/ GPTomics/bioSkills) and MedSkill (https://github.com/aipoch/medical-research-skills) were frozen snapshots exposed through the same discovery-and-load wrapper without adding ClinGeX content or graph edges. The full ClinGeX condition retained its native skill selection, dependency expansion and progressive execution workflow. No condition received the hidden reference answer or scoring rubric.

## A.5.3 Benchmark Construction

The benchmark drew from ClinVar, ClinGen, CIViC, OncoKB and ClinPGx. Eighty diversity-oriented anchors were selected deterministically: 10 each from OncoKB and CIViC and 20 each from ClinPGx, ClinVar and ClinGen. Selection favoured distinct genes and, where applicable, literature-supported pathogenic assertions or ClinGen records with both genetic and experimental evidence and non-trivial scoring or mechanistic issues.

DeepSeek-V4 generated items in 240 independent anchor–tier jobs using source- and tier-specific instructions, structured records and linked full text; related records could be retrieved for clinically plausible comparisons or distractors. Retained items had to pose a meaningful clinical operation, have a literature-verifiable answer and require interpretation rather than copying. Schema validation required a source-grounded answer, numeric PMIDs with relevance labels, at least one relevant PMID, a construction rationale and an item-level rubric. The final 426 items are summarized in Supplementary Table 14.

Supplementary Table 13 | System-framework conditions in the controlled enhancement-strategy ablation.
<table><tr><td rowspan=1 colspan=4>Condition             Component added to   Operational configurationDeepSeek-V4</td></tr><tr><td rowspan=7 colspan=4>DeepSeek             None                   The model answered the supplied question directly. It was not given retrievaltools, worked examples, domain-skill instructions or the ability to delegate tosub-agents. This condition is referred to as the unenhanced base-model baseline.DeepSeek + RAG     Search-function bundle All search-related capabilities implemented in the ClinGeX skill collection wereconsolidated into a retrieval bundle and exposed to the model. The conditiondid not receive non-search clinical-analysis skills, the skill graph or multi-agentDeepSeek + fewshot Three worked           Three worked examples were inserted into the model context before the test item.The examples illustrated the expected question-answer interaction, but thiscondition had no external retrieval, domain-skill library or sub-agent delegation.Three DeepSeek-V4 instances were assigned the roles of planner, executor andreviewer. The planner decomposed the item, the executor produced a draft by</td></tr><tr><td rowspan=2 colspan=2></td></tr><tr><td rowspan=1 colspan=1>DeepSeek + fewshot</td><td rowspan=1 colspan=1>Three worked</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>question-answer</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>DeepSeek + MA Planner-executor-</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=2 colspan=2>following the plan, and the reviewer checked the draft and returned revisioninstructions before finalisation.</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>DeepSeek + BioSkill</td><td rowspan=1 colspan=1>GitHub BioSkill</td><td rowspan=2 colspan=2>The open-source BioSkill instructions and associated resources were exposedthrough a common wrapper while retaining DeepSeek-V4 as the core model.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>repository</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=2 colspan=2>The ClinGeX-specific selector, skill graph and skill library were not used.The open-source MedSkill instructions and associated resources were exposed</td></tr><tr><td rowspan=1 colspan=2>DeepSeek + MedSkill GitHub MedSkill</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>repository</td><td rowspan=1 colspan=2>through the same wrapper principle and with the same DeepSeek-V4 core. The</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=2>ClinGeX-specific selector, skill graph and skill library were not used.The final system used its 105 clinical-genomics skills, task decomposition, skill</td></tr><tr><td rowspan=1 colspan=1>ClinGeX</td><td rowspan=1 colspan=1>Complete skill-centred</td></tr><tr><td rowspan=1 colspan=1>(DeepSeek-base)</td><td rowspan=1 colspan=1>architecture</td><td rowspan=2 colspan=2>selection, graph-based recall of related and prerequisite skills, progressive dis-closure of instructions, plan-execute orchestration, evidence-retrieval and code-</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=2>execution tools, and synthesis of the resulting execution trace into a finalresponse. This was the prespecified full-system condition and was not a union</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=4>of the BioSkill and MedSkill repositories.</td></tr></table>

## A.5.4 Reasoning Tiers and Task-Specific Construction Rules

Tiers were defined by the required evidence operation, as specified in Supplementary Table 15. Tier 1 items used no deliberately irrelevant PMID and cited a mean of 1.19 publications. Tier 2 allowed realistic non-supporting evidence when available (43/124 items; mean 2.82 publications). Tier 3 integrated multiple sources (mean 4.93 publications) and comprised 51 easy, 27 medium and 27 hard items according to evidence volume, conflict and irrelevant context.

## A.5.5 Response Generation and Rubric-Based Evaluation

All seven conditions answered the same 426 frozen prompts with DeepSeek-V4 as the common core. Reference answers, construction rationales, PMID relevance labels and rubrics were withheld. GPT-5.5 and Claude-4.8-Opus independently scored each source-masked response against its item-specific rubric; factual errors, omissions, irrelevant evidence, over-claiming and clinically incorrect recommendations reduced credit. The complete shared judge prompt follows.

## Prompt used for rubric-based LLM judging

## System instruction

You are an independent evaluator of answers to evidence-grounded clinical-genomics questions. Evaluate only the candidate response supplied below. Apply the item-specific scoring rubric exactly and use the reference answer as the authoritative description of the evidence available for this item. Do not infer which system produced the response and do not reward or penalise a response because of its presumed model or framework.

The benchmark spans variant interpretation, gene–disease validity assessment and clinical actionability, at three reasoning tiers: single-publication comprehension, evidence selection and strength grading, and multi-source synthesis. Depending on the item, a correct response may need to apply ACMG/AMP criteria, ClinGen evidence concepts, CIViC or OncoKB evidence levels, or PharmGKB/ClinPGx evidence levels. Some items deliberately contain superficially related but non-applicable evidence.

Supplementary Table 14 | Composition of the framework-selection benchmark by source and reasoning tier.
<table><tr><td>Source</td><td>Clinical-genomics domain</td><td>Tier 1</td><td>Tier 2</td><td>Tier 3</td><td>Total</td></tr><tr><td>ClinVar</td><td>Variant interpretation</td><td>39</td><td>22</td><td>19</td><td>80</td></tr><tr><td>ClinGen</td><td>Gene-disease validity</td><td>54</td><td>53</td><td>31</td><td>138</td></tr><tr><td>CIViC</td><td>Clinical actionability</td><td>30</td><td>16</td><td>19</td><td>65</td></tr><tr><td>OncoKB</td><td>Clinical actionability</td><td>29</td><td>14</td><td>14</td><td>57</td></tr><tr><td>ClinPGx</td><td>Clinical actionability</td><td>45</td><td>19</td><td>22</td><td>86</td></tr><tr><td>Total</td><td></td><td>197</td><td>124</td><td>105</td><td>426</td></tr></table>

Supplementary Table 15 | Operational definition of the three reasoning tiers in the framework-selection benchmark.
<table><tr><td>Tier</td><td>General operation</td><td>Representative task-specific operations</td><td>Principal scoring focus</td></tr><tr><td>1</td><td>Single-publication comprehension</td><td>Recover a paper-specific clinical, quantitative, functional, Correct factual elements; segregation or pharmacogenomic finding and interpret it correct direction and in the anchor context.</td><td>magnitude where applicable; source-specific caveats;</td></tr><tr><td>2</td><td>Evidence selection and strength grading</td><td>Map evidence to an ACMG/AMP criterion; justify a Clin- Correct evidence selection; Gen evidence type or score modification; distinguish clini- correct criterion, level or cal from preclinical evidence; grade actionability or repli- weight; justification based cation strength; reject a plausible but non-applicable pub- on study design and context; lication.</td><td>absence of invented details. appropriate handling of a distractor or conflicting</td></tr><tr><td>3</td><td>and decision</td><td>Multi-source synthesis Reconcile multiple submissions or studies; classify a Correct final synthesis or variant; synthesize gene-disease mechanism or lumping- recommendation; splitting evidence; select or sequence a therapy; provide a prioritisation of directly PGx prescribing recommendation with evidence-strength applicable evidence; caveats.</td><td>result. integration across sources; calibrated uncertainty and</td></tr></table>

1. Score the response criterion by criterion. Award full credit only when the required factual element, evidence interpretation and requested conclusion are all present. Award partial credit only when permitted by the rubric.

2. Accept clinically and semantically equivalent wording. Do not require exact phrase matching with the reference answer. Do not reward verbosity, formatting, confidence or writing style.

3. Judge cited evidence in the context specified by the question. Evidence about a diferent variant, gene–disease relationship, tumour type, therapy, phenotype or population does not support the target conclusion merely because it is biologically related.

4. Distinguish evidence identification from evidence weighting. A response that names a relevant study but assigns an incorrect ACMG/AMP criterion, evidence strength, database level or clinical implication has not fully satisfied that rubric element.

5. For multi-source items, assess whether the response integrates the relevant evidence, filters non-applicable evidence, resolves conflicts appropriately and produces the requested classification, synthesis or recommendation with calibrated caveats.

6. Penalise fabricated numerical values, publications, study findings, criteria or clinical recommendations. Also penalise over-claiming beyond the supplied evidence. Do not use outside knowledge to add new requirements or to overturn an item-specific reference answer.

7. Do not deduct the same error twice unless it independently violates two explicitly separate rubric criteria.   
Scores must remain between zero and the maximum score stated in the rubric.   
8. If the candidate response is incomplete, unclear or refuses to answer, award only the points supported by   
explicit content in the response. Do not reconstruct a missing answer on the candidate’s behalf.   
Item   
Item identifier: {{ITEM\_ID}}   
Clinical-genomics domain: {{TASK\_DOMAIN}}   
Reasoning tier: {{TIER}}   
Question   
{{QUESTION}}   
Source-grounded reference answer   
{{REFERENCE\_ANSWER}}   
Item-specific scoring rubric   
{{SCORING\_RUBRIC}}   
Candidate response   
{{CANDIDATE\_RESPONSE}}   
Return only one valid JSON object with the following structure:   
{"item\_id":"...", "maximum\_score":0, "awarded\_score":0,   
"criterion\_assessments":[   
{"criterion":"...",   
"status":"met|partially\_met|not\_met|contradicted",   
"points\_awarded":0, "reason":"..."}}],   
"unsupported\_or\_harmful\_claims":["..."],   
"score\_rationale":"..."}   
The sum of points\_awarded must equal awarded\_score. Use the maximum score defined by the item-specific   
rubric. Use an empty list when there are no unsupported or harmful claims. Keep each reason brief and tie it to   
explicit content in the candidate response.

Let $r _ { i j m }$ be the rubric points awarded to framework m on item i by judge j, and $R _ { i }$ the item maximum. The primary rubric-normalised score was

$$
S _ { m } = 1 0 0 \times \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { 2 } \left( \frac { r _ { i , \mathrm { G P T } , m } } { R _ { i } } + \frac { r _ { i , \mathrm { C l a u d e } , m } } { R _ { i } } \right) .\tag{4}
$$

Each item therefore contributed equally, regardless of rubric length or publication count. The same mean was calculated by reasoning tier and clinical-genomics domain; it measures rubric-defined factual correctness, evidence selection and weighting, completeness and final synthesis rather than exact-match accuracy.

## A.5.6 Framework-Selection Result

The full ClinGeX architecture achieved the highest rubric-normalised score overall and in every reasoning tier (Extended Data Fig. 1c). Its overall score was 83.2%, compared with 72.2% for the unenhanced DeepSeek-V4 baseline and 74.0–78.4% for the five single-strategy conditions. The largest separation occurred for tier 3 synthesis tasks. ClinGeX scored 70.1% on variant-interpretation items, 56.7% on gene–disease-validity items and 84.9% on clinical-actionability items, corresponding to improvements of 15.2, 11.1 and 15.1 percentage points, respectively, over the same base model without an enhancement layer. These controlled comparisons supported selection of the complete skill-centred architecture for the main system evaluation.

## A.6 Selection of the Frontier LLM Used as the Agent Core

## A.6.1 Purpose and Experimental Design

This experiment changed only the LLM core while holding the complete ClinGeX architecture, skills, tools, instructions and 3,662-item input set fixed. It compared the six frontier models in Supplementary Table 16

across variant interpretation (n = 1,366), gene–disease validity (n = 1,253) and clinical actionability (n = 1,043). Candidate cores were excluded from external judging.

Supplementary Table 16 | Frontier language models compared as the ClinGeX agent core.
<table><tr><td>Agent core</td><td>Parameter scale reported for the experiment</td><td>Access category</td><td>Role in the comparison</td></tr><tr><td>DeepSeek-V4</td><td>862B</td><td>Openly deployable</td><td>Candidate selected for the main ClinGeX experi- ments</td></tr><tr><td>Qwen3</td><td>235B</td><td>Openly deployable</td><td>Open-model comparator</td></tr><tr><td>GLM-5</td><td>754B</td><td>Openly deployable</td><td>Open-model comparator</td></tr><tr><td>GPT-5.4-Pro</td><td>Provider not disclosed</td><td>Provider hosted</td><td>Closed-model frontier comparator</td></tr><tr><td>Claude-4.6-Opus</td><td>Provider not disclosed</td><td>Provider hosted</td><td>Closed-model frontier comparator</td></tr><tr><td>Gemini-3-Pro</td><td>Provider not disclosed</td><td>Provider hosted</td><td>Closed-model frontier comparator</td></tr></table>

## A.6.2 Source Records and Construction of the Evidence-Grounded QA Corpus

The six pipelines used ClinVar and ClinGen-ACMG for variant interpretation, ClinGen Gene–Disease Validity for gene–disease assessment, and CIViC, OncoKB and PharmGKB/ClinPGx for clinical actionability. Every item was joined to a processed publication at the finest PMID-linked evidence unit supported by its source schema. Multi-publication narratives were treated only as trust anchors; claims and evidence codes had to be verified in the current paper, preventing cross-paper leakage or attribution to a comparator allele, disease, drug or population.

Source-specific candidate, acceptance and final-inclusion counts are given in Supplementary Table 17. The frozen experiment contained 3,662 items: 1,366 variant-interpretation, 1,253 gene–disease-validity and 1,043 clinical-actionability items.

Supplementary Table 17 | Construction and final inclusion of evidence-grounded QA items for agent-core selection.
<table><tr><td>Domain</td><td>Source</td><td>Candidatesª</td><td>Acceptedb</td><td>Discarded</td><td>Selectedc</td><td>Principal evidence join</td></tr><tr><td>Variant interpretation</td><td>ClinVar</td><td>1,176</td><td>743</td><td>433</td><td></td><td>743 Submission-PMID</td></tr><tr><td></td><td>ClinGen-ACMG</td><td>1,235</td><td>623</td><td>612</td><td></td><td>623 Variant-evidence code-PMID</td></tr><tr><td>Gene-disease validity</td><td>ClinGen</td><td>2,471</td><td>1,440</td><td>1,031</td><td></td><td>1,253 Assertion-evidence row-PMID</td></tr><tr><td>Clinical actionability</td><td>CIViC</td><td>408</td><td>294</td><td>114</td><td></td><td>294 Evidence item-PMID</td></tr><tr><td></td><td>OncoKB</td><td>524</td><td>222</td><td>302</td><td>222</td><td>Association-PMID</td></tr><tr><td></td><td>PharmGKB/ClinPGx</td><td>1,000</td><td>527</td><td>473</td><td>527</td><td>Variant annotation-PMID</td></tr><tr><td>Total</td><td>Six sources</td><td>6,814</td><td>3,849</td><td>2,965</td><td>3,662</td><td></td></tr></table>

<sup>a</sup> Candidate units processed by the adversarial QA-generation pipeline.  
<sup>b</sup> Items accepted by the QA gate.  
<sup>c</sup> Items included in the frozen model-selection experiment.

The 3,849 accepted items are the complete output of the archived adversarial QA-generation batches. The model-selection total is 3,662 because the frozen experiment used 1,253 of the 1,440 accepted ClinGen Gene–Disease Validity items. ClinVar and ClinGen-ACMG were combined for the variant-interpretation radar plot; CIViC, OncoKB and PharmGKB/ClinPGx were combined for the clinical-actionability radar plot.

## A.6.3 Adversarial QA Generation and Quality Control

A source-specific generator produced a publication-grounded question, reference answer, reasoning annotation and supporting quotations. Items had to require interpretation or synthesis, and every quotation had to fuzzy-match the current full text at a similarity of at least 0.80.

Four memoryless answerers (DeepSeek-V4, Qwen3.7-Max, MiniMax-M3 and MiMo-V2.5-Pro) then answered from the same paper. A source-masked Qwen3.7-Plus arbiter from a diferent model family checked ambiguity, source support, overreach and semantic dificulty, returning discriminative acceptance, consensus-hard acceptance or rejection. Rejected units were regenerated for at most three quality attempts. Accepted audit records were retained but hidden from evaluated cores.

## A.6.4 Model-Facing Input, Leakage Control and Common Workflow

Each core received the question and retrieval entry points, not full text in the initial prompt, and used the same ClinGeX skills, tool definitions, 50-turn limit, timeout and retry policy. Runs retained the final answer, status and complete trace, including tool, skill, action, canonical parameters and result metadata.

Construction answerers, arbitration data, rejected attempts, generator rationale and target reasoning point were removed before inference; reference answers, quotations and source annotations were scoring-only. Every (model, item) pair was run independently three times without shared state, retrieved content or tool history.

## A.6.5 Five Evaluation Dimensions

The five dimensions combined external-judge assessment of the response with judge-free analysis of the execution trace. The external judge was not one of the six candidate cores.

Absolute accuracy. Each reference answer was cached as atomic claims. The item score was 1 when all claims were supported, 0.5 when at least half were supported, and 0 otherwise; contradiction of the critical target claim forced a score of 0. Thus, accuracy was a deterministic 0/0.5/1 claim-coverage score rather than string matching.

Reasoning score. Five binary indicators assessed specific evidence use, recognition of the hidden reasoning issue, logical connection of evidence to conclusion, coverage of all requested parts and consistency between rationale and answer. Their sum ranged from 0 to 5.

Agentic ability. Three judge-free trace statistics captured execution efort and reliability: tool-call count $T ,$ failed-call rate $f$ and exact-duplicate rate d, where duplicates shared the same skill, action and canonical parameters. Values were averaged across the three runs and, for multi-source domains, across sources with equal weight. For the radar axis, each statistic was reverse min–max normalized across the six models within a domain and combined as $G _ { m } = ( Z _ { m , T } + Z _ { m , f } + Z _ { m , d } ) / 3$ . Raw statistics were retained because call count alone was not interpreted as quality.

Faithfulness. Direct quotations were fuzzy-matched to retrieved text; other concrete claims were labelled supported, contradicted or unverifiable against retrieved evidence. With fabricated-quotation rate $q _ { i }$ and contradicted-or-unverifiable factual-claim rate $h _ { i } .$ , faithfulness was $H _ { i } = 1 - 0 . 5 q _ { i } - 0 . 5 h _ { i }$ . Errors afecting key values, evidence levels, clinical scope or conclusions were additionally flagged as high severity.

Robustness. Across three independent runs, answer consistency $C _ { i }$ was 1 when all accuracy scores agreed, 0.5 when two agreed and 0 otherwise. Tool-sequence similarity $U _ { i }$ was the mean pairwise Levenshtein similarity of (skill, action) sequences; agentic stability $V _ { i }$ was one minus the mean coeficient of variation of tool, failed and duplicate-call counts, truncated at zero. The composite was $B _ { i } = 0 . 4 C _ { i } + 0 . 3 U _ { i } + 0 . 3 V _ { i }$

## A.6.6 External Judge Prompt

Reference claims were decomposed once per item and reused for every model and run. The source-masked prompt below separately elicits claim correctness, five reasoning indicators and factual-faithfulness labels; candidate models never received the reference-only fields.

External judge prompt for agent-core evaluation

You are an independent evaluator of evidence-grounded clinical-genomics answers. You are not one of the models being compared. Evaluate only against the supplied question, reference answer, hidden item annotations and retrieved source evidence. Do not reward fluency, verbosity, model identity or agreement with unsupported background knowledge. Do not infer that a claim is supported merely because it is plausible. Treat scope changes involving the variant, gene, disease, inheritance mode, population, treatment, comparator, evidence level or direction of efect as substantive errors. Return valid JSON only.   
User message   
<QUESTION>   
{question}   
</QUESTION>   
<REFERENCE\_ANSWER>   
{groundtruth\_answer}   
</REFERENCE\_ANSWER>   
<REFERENCE\_ATOMIC\_CLAIMS>   
{cached atomic claims; mark the claim corresponding to the target reasoning point as critical}   
</REFERENCE\_ATOMIC\_CLAIMS>   
<ITEM\_ANNOTATION>   
Reasoning type: {reasoning\_type}   
Target reasoning point: {target\_reasoning\_point}   
Verified supporting quotations: {supporting\_quotes}   
Source-specific evidence identifiers: {used\_evidence\_ids}   
</ITEM\_ANNOTATION>   
<RETRIEVED\_SOURCE\_EVIDENCE>   
{the publication text and structured evidence actually retrieved during the run}   
</RETRIEVED\_SOURCE\_EVIDENCE>   
<CANDIDATE\_ANSWER>   
{model response}   
</CANDIDATE\_ANSWER> 1. For every reference atomic claim, assign exactly one label: SUPPORTED, CONTRADICTED or MISSING. A claim is SUPPORTED only when the candidate states its clinically material content with the correct entities, direction, scope and qualification. Identify whether a contradicted claim is the critical target claim. Do not assign the final 0/0.5/1 accuracy score; it will be computed deterministically.   
2. Score each reasoning criterion as 0 or 1: specific evidence use; recognition of the annotated trap; logical validity; completeness across requested subparts; and consistency of rationale with final answer. Give a short evidence-based reason for every value.   
3. Extract each concrete non-quoted factual claim in the candidate answer and label it SUPPORTED, CONTRADICTED or UNVERIFIABLE against the retrieved source evidence. Do not re-evaluate direct quotations here; they are checked separately by text matching. Mark an error as high severity if it changes a key value, evidence code or level, clinical scope, direction of efect or final conclusion.   
4. Ignore style unless it causes ambiguity or omission. If the source evidence is insuficient to judge a claim, use UNVERIFIABLE; do not use general knowledge to rescue it.

```jsonl
Return this schema:
{
"atomic_claim_labels": [{"claim_id": "C1", "label": "SUPPORTED|CONTRADICTED|MISSING",
"critical": true|false, "reason": "..."}],
"reasoning": {"evidence_use": {"score": 0|1, "reason": "..."}, "trap_recognition": {...},
"logical_validity": {...}, "completeness": {...}, "self_consistency": {...}},
"factual_claims": [{"claim": "...", "label": "SUPPORTED|CONTRADICTED|UNVERIFIABLE",
"severity": "high|standard", "evidence": "..."}],
"notes": "..."
}
```

The scoring code validated the returned schema, summed the five reasoning indicators, applied the deterministic accuracy thresholds and combined the faithfulness labels with the independent quotation-matching result. The judge was not given the candidate model name.

## A.7 Details of the MRMC Human-AI Collaboration Study

The details below supplement the main study design. The analysis unit was a submitted case assessment;   
reader-facing materials never contained reference answers, scoring annotations or adjudication records.

## A.7.1 Dedicated Online Evaluation Platform

The password-protected platform linked each account to nine fixed task–condition assignments and presented de-identified case materials beside the scenario-specific response form. Reference answers, scoring keys, other readers’ responses and alternate-condition responses to the same case were inaccessible. Readers used no AI in the unassisted condition or consulted the assigned external DeepSeek/ClinGeX interface before entering their own conclusion; AI output was never copied automatically.

Drafts were retained, required fields were checked on submission and oncology readers completed the open response before viewing candidate statements. Completion time was self-reported rather than inferred from browser duration. Scenario fields are listed in Supplementary Table 18; five-level confidence was encoded 1–5 from no confidence to very confident.

Supplementary Table 18 | Reader-facing components of the MRMC evaluation platform.
<table><tr><td>Scenario</td><td>Information presented</td><td>Required reader entries</td></tr><tr><td>Criterion-level variant analysis</td><td>De-identified phenotype, variant in transcript-level notation, associated gene/disease and inheritance context, and case-specific publications when</td><td>For each of the 28 ACMG/AMP criteria: applicable, not applicable or undetermined; effective evidence strength (Very Strong, Strong, Mod- erate or Supporting) when applicable; and an optional criterion-level rationale. The reader then entered one five-tier pathogenicity classifica- tion, a five-point confidence rating, an integrated written justification and completion time.</td></tr><tr><td>Genomic interpretation in rare disease</td><td>De-identified phenotype and a set of candidate variants, including gene, variant description, zygosity, parental origin or phase when available</td><td>For every candidate variant: one five-tier pathogenicity category and one five-category assessment of how well the variant explained the phenotype. The reader additionally entered one case-level confidence rating, an integrated explanation covering the candidate set, and completion time.</td></tr><tr><td>Molecular oncology decision-making</td><td>De-identified longitudinal clinical, pathological, treatment by one case-specific decision or interpretation question</td><td>An open-text answer and its confidence rating, followed by a separate judgement of each of four candidate statements as basically reasonable, and molecular findings, followed partly reasonable or unreasonable, a second confidence rating for the statement judgements, optional explanations and completion time.</td></tr></table>

## A.7.2 Reference Standards and Expert Adjudication

Before analysis, each three-member scenario panel froze a reference record matching the reader fields: final classification, ACMG/AMP criteria and strengths for criterion-level variant analysis; candidate pathogenicity, phenotype explanation and acceptable adjacent levels for genomic interpretation in rare disease; and a core open answer plus graded candidate statements for molecular oncology decision-making. Enumerated fields were compared by prespecified mappings, while free text and borderline cases were resolved by source-masked panel consensus based on clinical equivalence rather than wording or response length.

Primary outcome: case-level overall concordance. For each scenario and condition, concordance was the proportion of the 45 submitted assessments satisfying the complete task-specific rule in Supplementary Table 19; no pooled primary outcome combined the three scenarios.

Supplementary Table 19 | Task-specific definition of primary case-level overall concordance in the MRMC study.
<table><tr><td>Scenario</td><td>Requirements for a concordant case</td><td>Tolerance and discordance rules</td></tr><tr><td>Criterion-level variant analysis</td><td>Exact final five-tier classification; every reference ACMG/AMP Any missing reference criterion, criterion recovered; strength of each recovered reference criterion strength difference exceeding one within one adjacent level on Supporting-Moderate-Strong-Very level or wrong final class was Strong.</td><td>discordant. Material unsupported evidence could also make the complete response discordant.</td></tr><tr><td>Genomic interpretation in rare disease</td><td>Exact pathogenicity category for every clinically material candidate; One adjacent phenotype level was phenotype-explanation category within the accepted range; correct accepted only when the substantive causal combination and inheritance configuration; coherent inte- interpretation was unchanged. grated interpretation.</td><td>Larger differences, an incorrect causal combination or incompatible narrative were discordant.</td></tr><tr><td>Molecular oncology decision-making</td><td>Panel-adjudicated agreement of the open answer with the core con- One adjacent statement level was clusion and a statement profile preserving the reference clinical accepted only when the clinical positions.</td><td>stance was unchanged. A contradictory open answer or endorsement of a materially unsafe or unsupported action was discordant.</td></tr></table>

## A.7.3 Secondary Performance Outcomes

Secondary outcomes isolated the structured reasoning components listed with their units and denominators in Supplementary Table 20; eligible item instances were micro-averaged within condition.

For criterion-level variant analysis, evidence recall was $\textstyle \sum _ { i } | G _ { i } \cap P _ { i } | / \sum _ { i } | G _ { i } |$ , unsupported-evidence assignment was $\textstyle \sum _ { i } | P _ { i } \setminus G _ { i } | / \sum _ { i } | P _ { i } |$ , and strength concordance required an exact strength match among recalled reference criteria. The second metric is labelled the evidence hallucination rate in the figure and code, but denotes an unsupported ACMG/AMP criterion rather than a fabricated citation. Final-class accuracy and directional over-/underestimation used the ordered five-tier pathogenicity scale; the reference contained 176 criterion instances.

For genomic interpretation in rare disease, the 117 candidate variants were evaluated for exact pathogenicity, directional error and phenotype-explanation accuracy. Phenotype accuracy allowed an accepted adjacent level only when the substantive interpretation was preserved; over-/underestimation was measured beyond this accepted range. Primary concordance additionally required the correct causal combination and inheritance interpretation.

For molecular oncology decision-making, free-response accuracy used panel-adjudicated agreement with the core answer across 45 cases. The four statements per case produced 180 judgements; accuracy required the reference clinical position, while over-/underestimation followed the ordered unreasonable–partly reasonable–basically reasonable scale and did not tolerate a supported–unsupported boundary crossing.

## A.7.4 Confidence–Correctness Alignment, Time, Survey and Continued-Use Outcomes

For criterion-level variant analysis, behavioural certainty was full when all 28 ACMG/AMP criteria had an explicit applicable/not-applicable call and partial when any remained undetermined. Group accuracy used exact final pathogenicity classification, not the stricter primary composite; Wilson intervals were descriptive because fully certain groups were small.

For genomic interpretation in rare disease, correctness was primary case-level concordance; for molecular oncology decision-making, it was correctness of the open answer entered immediately before confidence. Five-level confidence was related to correctness using point-biserial correlation and ROC–AUC. Positive correlation and AUC above 0.5 indicate discrimination.

Confidence categories were mapped to probabilities 0.1, 0.3, 0.5, 0.7 and 0.9 for the Brier score, $\begin{array} { r } { N ^ { - 1 } \sum _ { i } ( q _ { i } - Y _ { i } ) ^ { 2 } } \end{array}$ correlation and AUC retained the original ordinal scores. Expected calibration error was descriptive, with no model fitting or threshold optimization.

The main time comparison used each reader’s post-study estimate of mean minutes per case, not browser-open duration or the case-form field. Paired two-sided Wilcoxon tests within scenario were Holm-adjusted, and medians were reported.

The questionnaire comprised three time items and five ordinal dimensions rated separately for DeepSeek and ClinGeX; ratings were summarized by mean and standard deviation. Continued use was the number of new ClinGeX conversations initiated in the 14 days after completion, excluding three assigned study sessions, and was analysed descriptively without hypothesis testing.

## A.7.5 Statistical Analysis of Proportions and Efect Diferences

Supplementary Table 20 | Analysis units and denominators for secondary MRMC performance outcomes.
<table><tr><td rowspan=1 colspan=7>Scenario                Outcome                         Analysis unit per       Denominatorcondition</td></tr><tr><td rowspan=5 colspan=7>Criterion-level variant Evidence recall                   Reference criterion      176 reference criteriaanalysis                                                   instanceEvidence hallucination rate      Submitted applicable  Number called applicable in that conditionPathogenicity accuracyPathogenicity underestimation</td></tr><tr><td rowspan=2 colspan=4>Evidence-strength concordance Corectly recalleNumbe</td><td rowspan=1 colspan=1>criterion instance</td></tr><tr><td rowspan=1 colspan=2>Corectle rctalNumber recalledin hat condlilion</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=3>caseComplete submitted    45</td></tr><tr><td rowspan=1 colspan=1>Comple submittdted</td></tr><tr><td rowspan=1 colspan=4>ratePathogenicity overestimation</td></tr><tr><td rowspan=1 colspan=4>rate</td><td></td><td rowspan=2 colspan=2>case117</td></tr><tr><td rowspan=1 colspan=1>Genom</td><td rowspan=3 colspan=5>Genomic interpretation Pathogenicity accuracyin rare diseasePathogenicity overestimation</td><td rowspan=1 colspan=1>Candidate varia</td></tr><tr><td></td><td rowspan=2 colspan=2>Candidate variant      117</td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=2 colspan=1>ratePathogenicity underestimation</td><td></td><td rowspan=2 colspan=1>Candidate variant</td><td rowspan=2 colspan=1>117</td></tr><tr><td rowspan=1 colspan=3></td><td></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>rate</td><td rowspan=3 colspan=2>Candidate variant</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=2 colspan=1>Phenotype-explanation</td><td></td></tr><tr><td rowspan=1 colspan=1>117</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>accuracy</td><td></td><td rowspan=2 colspan=1>Candidate variant</td><td rowspan=2 colspan=1>117</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>Explanation-degree</td><td></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>overestimation rate</td><td></td><td rowspan=2 colspan=1>Candidate variant</td><td rowspan=2 colspan=1>117</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>Explanation-degree</td><td></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>underestimation rate</td><td></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>45</td></tr><tr><td rowspan=1 colspan=3>Molecular oncology</td><td rowspan=1 colspan=1>Free-response accuracy</td><td></td><td rowspan=1 colspan=1>Complete open answer</td></tr><tr><td rowspan=1 colspan=3>decision-making</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>180 (four per case)</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>Reasonableness-judgement</td><td></td><td rowspan=1 colspan=1>Candidate statement</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>accuracy</td><td></td><td rowspan=2 colspan=1>Candidate statement</td><td rowspan=3 colspan=1>180 (four per case)</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>Reasonableness overestimation</td><td></td></tr><tr><td rowspan=1 colspan=7>rateReasonableness underestimation Candidate statement</td></tr><tr><td rowspan=1 colspan=7>rate</td></tr></table>

Observed numerators and denominators were retained for descriptive reporting. For each scenario and outcome, condition efects were estimated using a binomial logistic GLMM with condition as a fixed efect and crossed random intercepts for reader and case. Case-level binary outcomes were entered as success or failure; item-level outcomes were grouped within each reader–case assessment as cbind(success, total-success), thereby retaining their observed denominators. Accuracy, overestimation and underestimation were analysed as separate binary outcomes. Model-standardized probabilities and absolute between-condition diferences in percentage points were obtained by integrating over the estimated reader- and case-level random-efect distributions, with two-sided 95% Wald confidence intervals. For outcomes where lower values were favourable, figures show one minus the model-standardized event probability and the direction of each contrast was reversed. Pairwise Wald contrasts were Holm-adjusted across the three comparisons within each outcome and scenario; the corresponding two-degree-of-freedom omnibus Wald test was descriptive only. Random slopes were not fitted because each reader contributed only three assessments per condition within a scenario.

## A.7.6 Representative Complete Reader-Facing Tasks

The following examples reproduce the structure and level of information presented to readers. They are shown without participant answers and without the hidden expert scoring record.

## Representative MRMC task 1: criterion-level variant analysis

Clinical vignette. A three-year-old boy presented with pallor, failure to gain weight, intermittent vomiting and fever. Weight and height were 10 kg (−2.66 s.d.) and 85 cm (−2.57 s.d.), respectively. Prenatal ultrasonography at 34 weeks had shown enlarged kidneys. Postnatal imaging showed bilaterally increased renal echogenicity with multiple small cysts. At admission, ultrasonography showed cholelithiasis, mild left calyceal dilatation, bilatera renal cysts and renal-parenchymal abnormalities. Blood pressure was 160/100 mmHg, haemoglobin $5 . 4 \mathrm { g } \mathrm { d l } ^ { - 1 }$ serum creatinine $4 . 5 \mathrm { m g } \mathrm { d l } ^ { - 1 }$ and estimated glomerular filtration rate $7 . 8 \mathrm { { m l } \mathrm { { m i n } ^ { - 1 } 1 . 7 3 \mathrm { { m } ^ { - 2 } } } }$ , with metabolic acidosis. The evaluated variant was heterozygous; biallelic INVS variants are associated with autosomal-recessive nephronophthisis.

## Variant. An anonymized heterozygous INVS variant.

Material available. The vignette, variant and one case-associated publication supplied as a PDF; ordinary literature and database searches were permitted under the condition-specific rules.

Reader task. For each of PVS1, PS1–PS4, PM1–PM6, PP1–PP5, BA1, BS1–BS4 and BP1–BP7, select applicable, not applicable or undetermined; when applicable, select Very Strong, Strong, Moderate or Supporting and optionally state the evidence. Then provide the overall five-tier pathogenicity classification, confidence, an integrated justification and completion time.

## Representative MRMC task 2: genomic interpretation in rare disease

Clinical vignette. A female infant developed progressive heart failure shortly after birth. Cardiac imaging showed chamber enlargement and rapidly worsening systolic function, and she died in early infancy. A sibling had an almost identical early-onset course.

Candidate set.

1. An anonymized heterozygous ASNA1/GET3 variant; parental origin not established.

2. An anonymized heterozygous paternal ASNA1/GET3 variant; part of a possible compound-heterozygous configuration with candidate 3.

3. An anonymized heterozygous maternal ASNA1/GET3 variant; part of the same possible compoundheterozygous configuration.

Reader task. For each candidate, select one pathogenicity category (Benign, Likely Benign, Uncertain Significance, Likely Pathogenic or Pathogenic) and one phenotype-explanation category (fully explains, possibly explains, partly explains, no clear evidence or clearly cannot explain). Then provide one confidence rating, an integrated explanation addressing the candidate combination, phenotype and inheritance pattern, and total completion time.

## Representative MRMC task 3: molecular oncology decision-making

Clinical vignette. A 66-year-old never-smoking woman had a 44-mm left-upper-lobe lung adenocarcinoma with pleural efusion, pleural thickening and a solitary temporal-lobe brain metastasis (cT2bN0M1b, stage IVA). The tumour carried EGFR L858R and had PD-L1 TPS < 1%. After stereotactic treatment of the brain metastasis, first-line osimertinib produced a partial response and prolonged disease control. At 46 months, the lung primary enlarged and became metabolically active while the previous pleural and brain sites remained controlled. The isolated progressing lesion was resected. The specimen contained adenocarcinoma plus new giant-cell and spindle-

<table><tr><td>cell components, with reduced TTF-1, increased vimentin and a diagnosis of pleomorphic carcinoma. EGFR L858R remained detectable and PD-L1 TPS was &gt; 75%. No measurable progressing disease remained after surgery. Candidate statements rated after the open response.</td></tr><tr><td>Open question. How should systemic treatment be adjusted after surgery? 1. Continue osimertinib and add platinum-based chemotherapy because pleomorphic transformation and high</td></tr><tr><td>PD-L1 suggest resistant micrometastatic disease requiring chemotherapy coverage. 2. Continue osimertinib and add a PD-1/PD-L1 inhibitor because pleomorphic transformation and TPS &gt; 75% suggest increased immunogenicity. 3. Stop osimertinib and use PD-1/PD-L1 inhibitor monotherapy because high PD-L1 and pleomorphic</td></tr></table>

## A.7.7 Construction and Quality Control of the Molecular Oncology Tasks

The 323 human molecular-oncology case reports identified in the main Methods—186 digestive-system and 137 thoracic-tumour reports published after 2025—were processed independently before expert selection. GPT-5.5 read each complete report and returned structured fields comprising the clinically decisive or counterintuitive features, their clinical implication, a de-identified vignette, one direct question, a concise core answer and explicit scoring points. Vignettes retained the clinical, pathological, treatment and molecular facts needed to answer the question, but omitted article titles, authors, journals, identifiers and source-revealing wording.

Five independent models—DeepSeek-V4, Claude Sonnet 4.6, GPT-5.4, GLM-5.1 and Qwen3.7—then answered each vignette and question in fresh contexts without access to the report, reference answer or other model responses. Claude Opus 4.8 subsequently assessed whether the reference was source-grounded and clinically defensible; whether the vignette was self-contained and free of answer or publication leakage; whether the question was single, direct and appropriate for a practising oncologist; and whether the simulated responses demonstrated useful discriminatory dificulty. Failed items were regenerated by GPT-5.5 using the arbitrator’s written feedback and rerun through independent answering and arbitration, for at most five generation rounds. Reports without an acceptable task were discarded.

For each accepted open question, the reference conclusion supplied the best-supported of four candidate statements, while incorrect or partially correct model responses were used to construct clinically plausible alternatives. Each statement received a hidden five-point truth grade: grades 4–5 mapped to basically reasonable, grade 3 to partly reasonable, and grades 1–2 to unreasonable. Statement order was randomized. A final validation pass required consistency between the core answer and statement grades, exactly one best-supported statement, defensible distractors and no superficial answer cue from length, tone, certainty or absolute wording. The reader-facing task contained only the vignette, question and statements; source reports, scoring points, truth grades, simulated answers and revision records were retained separately for audit and adjudication.

The automated pipeline produced 36 digestive-system and 26 thoracic-tumour candidates. A three-member oncology panel completed and reviewed every candidate for clinical relevance, objectivity, answer defensibility, dificulty, actionability, leakage risk and the plausibility and grading of all statements. The panel retained 27 digestive-system and 18 thoracic-tumour tasks, yielding the final 45, and could correct the core answer, statement wording or reference grades before the scoring record was frozen. Finally, the selected vignettes were manually de-identified: nonessential details and publication-revealing phrases were removed or reformulated, but no fact required to answer the clinical question was deleted.

## A.8 Feedback Coverage in the Prospective Deployment

To assess the completeness of episode-level feedback, we counted all platform conversations recorded for each participant during the deployment period. A conversation was one platform session and could contain multiple dialogue turns. Feedback was submitted per completed analysis episode, and one episode could span multiple conversations. To preserve participant privacy, we did not inspect conversation content and therefore could not determine whether conversations without associated feedback were outside the study scope or represented eligible episodes for which feedback was not submitted. We therefore treated the ratio of feedback submissions to conversations as a conservative lower-bound proxy for episode-level feedback coverage, rather than as a one-to-one response rate. Overall, 96 feedback submissions were recorded for 111 conversations (86.5%); participant-level ratios ranged from 66.7% to 100% (Supplementary Table 21). This high lower-bound coverage was consistent with participants generally following the requested feedback procedure.

Supplementary Table 21 | Conversation counts and feedback coverage during the prospective deployment.
<table><tr><td>Participant</td><td>Conversations</td><td>Feedback submissions</td><td>Conservative coverage</td></tr><tr><td>User 1</td><td>9</td><td>7</td><td>77.8%</td></tr><tr><td>User 2</td><td>9</td><td>9</td><td>100%</td></tr><tr><td>User 3</td><td>12</td><td>9</td><td>75.0%</td></tr><tr><td>User 4</td><td>12</td><td>11</td><td>91.7%</td></tr><tr><td>User 5</td><td>7</td><td>7</td><td>100%</td></tr><tr><td>User 6</td><td>13</td><td>10</td><td>76.9%</td></tr><tr><td>User 7</td><td>8</td><td>6</td><td>75.0%</td></tr><tr><td>User 8</td><td>11</td><td>10</td><td>90.9%</td></tr><tr><td>User 9</td><td>16</td><td>16</td><td>100%</td></tr><tr><td>User 10</td><td>9</td><td>6</td><td>66.7%</td></tr><tr><td>User 11</td><td>5</td><td>5</td><td>100%</td></tr><tr><td>Total</td><td>111</td><td>96</td><td>86.5%</td></tr></table>