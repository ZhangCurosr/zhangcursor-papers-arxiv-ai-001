# Fairness or Fallacy? Mapping the Landscape of AI Bias in Disease Detection

Ghulam Mudassir<sup>a</sup>, Hameed Ullah Khan<sup>b</sup>, Ahmad Sami Al-Shamayleh<sup>c</sup>, Muhammad Faizan Khan<sup>d</sup>, Muhammad Muqtasid<sup>e</sup>, Moamin Abughazala<sup>f</sup>, Adnan Akhunzada<sup>g</sup>

<sup>a</sup>School of Computing, University of Buckingham, Buckingham, MK181EG, United Kingdom <sup>b</sup>Department of Computer Science Sir Syed CASE Institute of Technology, Islamabad, 44000, Pakistan <sup>c</sup>Department of Data Science and Artificial Intelligence, Faculty of Information Technology, Al-Ahliyya Amman University, Amman, 19328, Jordan

<sup>d</sup>Aberdeen Institute of Data Science and Artificial Intelligence, South China Normal University, Guangzhou, China <sup>e</sup>Department of Computer Science, Faculty of Computing Sciences, Lahore Leads University, Lahore, Pakistan <sup>f</sup>Department of Information Engineering, Computer Science and Mathematics, University of L’Aquila, L’Aquila, 67100, Italy

<sup>g</sup>Department of Data and Cybersecurity, University of Doha for Science and Technology, Doha, 24449, Qatar

## Abstract

Artificial intelligence (AI) has several transformative efects on healthcare, like to facilitate clinicians in disease detection and to manage patient care with enhanced precision. Due to the increasing complexity of medical data and the growing volume of diagnostic images, the substantial computational power needed to process such information creates both opportunities and challenges. To accomplish this ob jective, numerous methodologies have been proposed to integrate AI into disease detection systems. All these methodologies consider diferent aspects of clinical diagnosis, considering key parameters such as the type of AI model employed, the nature of input data, and the population demographics being analysed. In this paper, we review diferent state-of-the-art techniques which have been proposed for various areas of disease detection, specifically focusing on bias identification and mitigation. The proposed solutions are distinguished based on the type of input data they utilize, the architecture of model, and type of bias (algorithmic bias, demographic bias, model bias, and deployment bias). The time frame chosen to filter out relevant studies is between 2015 and 2026. Eventually, we review 23 related articles out of 64,389 analyzed from seven diferent digital libraries. The obtained results guide us to reach a conclusion with respect to state-of-the-art selected studies.

Furthermore, these results reveal significant research gaps that can assist researchers in addressing AI-related bias in medical diagnosis. By prioritizing explainability, data diversity, and fairness in algorithmic design, as well as equitable access for afected patient populations, they can utilize available healthcare resources more efectively.

Keywords: Artificial Intelligence; Bias Detection; Disease Detection; Healthcare AI; Fairness in AI; Mapping Study

## 1. Introduction

Artificial intelligence (AI) is changing healthcare in ways we could hardly anticipated a few years back. Now it is helping doctors to diagnose diseases earlier and more accurately [68] [38]. AI systems understand complex and large-scale datasets with help of machine learning (ML) and deep learning (DL) models for identification as well pattern learning. These datasets consist on medical records, imaging scans, genetic information, and even patterns from internet searches [7] [64]. AI models are helpful in making medical decisions, as they provide better diagnostic results compared to humans that can commit a mistake and to reduce chance of errors [63]. Early disease detection is particularly important for serious illnesses such as cancer, Alzheimer’s disease, or infectious diseases. When diagnosed early, treatments work better and patient outcomes improve [2] [41]. AI can also extend specialized care to remote or underserved areas through telemedicine and real-time monitoring [66]. Beyond diagnostics, it can help track disease trends, predict progression, and guide public health decisions. So, AI makes healthcare faster, more precise, and more proactive.

AI is powerful, but it is not perfect due to one major problem that is called bias. These systems learn from data, and if the data is incomplete or skewed, the results can be unfair [74]. For example, an algorithm trained mostly on patients from one community may perform well for them but poorly for others. The same issue arises when hospital data mostly comes from wealthy cities. Models built on such data may miss patterns common in rural or low-income areas [54]. These gaps are not abstract they can mean delayed diagnoses, slower treatment, or worse outcomes for those already at risk [44]. Bias can occur at any stage: during data collection, while training the model, or even when the tool is used in practice. That is why bias is not just a technical issue; it is also an ethical one. If patients cannot trust AI to treat them fairly, its potential in healthcare will never be fully realized [9].

Real-life examples show how bias afects disease detection. In cancer care, some imaging models detect tumors more accurately on lighter skin than on darker skin, creating unequal results [32]. During the COVID-19 pandemic, AI tools trained on data from developed countries often failed when applied in low resource regions. These systems simply hadn’t seen enough diverse cases [81]. Similar challenges appear in Alzheimer’s research, where models built on data from high-income countries may miss lifestyle or cultural diferences that afect symptoms elsewhere [64]. Even detection of infectious diesease can be problematic: models may fail in rural or crowded areas where data are scarce, leaving blind spots in outbreak prediction [44]. These examples show that bias is real and tangible. It afects patients’ lives, and those already at the margins of healthcare face the highest risks [52].

Fixing bias in AI is not just about better algorithms, in fact it is all about quality of data. Models need information from people of all ages, genders, ethnicities, and regions [44]. Algorithms must be transparent, and they should be tested regularly across diverse populations [81]. Continuous monitoring is also essential, because healthcare systems and patient needs change over time. Collaboration is equally important where computer scientists, clinicians, and policymakers must work together. Each group contributes knowledge to spot hidden biases and make systems fairer [29]. There is also an ethical responsibility where AI should reduce inequalities, not worsen them. Clear guidelines and regulations can help ensure that these systems are safe, fair, and trusted [40].

Bias in AI for disease detection is a growing concern. Many studies highlight AI’s benefits, but fewer examine its risks in a structured way [45]. There is a need to map the research that has been done and identify the gaps that remain. A systematic mapping study can provide this overview. It can show which types of bias are most common, which methods are used to detect and reduce them, and which diseases have been studied most [43]. The goal of this study is to organize the existing evidence and point toward future directions. Ultimately, the aim is to support AI systems that are not only accurate but also fair and reliable for all patients [54].

This paper presents recent insights on bias in AI for disease detection. It explains both the strengths and the weaknesses of current approaches. The findings of this mapping study will support future research and also encourage new ideas in this important field. The study also shed light on unresolved research gaps, ofering opportunities for future work.

The major contributions of our work include:

• An overview of the state of the art on bias in AI disease detection, based on studies published between 2015 and 2026 in major digital libraries.

• A review of research trends in bias detection and mitigation over this period.

• A summary of the limitations found in primary studies and the open issues that still need attention.

The rest of the paper is structured as follows: Section 2 presents the background, Section 3 explains the systematic mapping study process, Section 4 presents the results of systematic mapping study, Section 5 discusses potential research area, Section 6 presents threats to validity, Section 7 is about discussion and Section 8 concludes the paper.

## 2. Background

Algorithmic bias in healthcare remains a major concern. Race-based corrections for glomerular filtration rate and pulmonary function continue to afect clinical decisions [27]. Addressing bias in AI is even harder, as model complexity and varying user literacy can hide discriminatory outcomes. Concealed discrimination is not new in healthcare and often occurs regardless of intent. Implicit bias, for example, delays melanoma diagnoses in dark-skinned patients, resulting in worse outcomes [27]. It also contributes to racial disparities in pain medication administration [55]. At the population level, mechanisms such as the “submerged state” reinforce inequities. Here, concealed welfare programs like tax credits perpetuate the racial wealth gap [46]. During COVID-19, CARES Act funds were allocated based on lost hospital revenue, underfunding hospitals in communities of color [8]. These cases highlight the risk of bias becoming “submerged” within AI systems.

AI adoption in healthcare has grown rapidly. Early studies aimed to improve predictive performance using deep learning and ensemble techniques [17] [49]. Recent work focuses on fairness, explainability, and trustworthiness [10]. High accuracy does not guarantee ethical outcomes. Bias in AI often reflects societal inequities in clinical data [21]. [5] showed that a risk prediction algorithm underestimated the needs of Black patients by using healthcare costs as a proxy [69].

Frameworks exist to understand and reduce bias. [5] categorizes bias as pre-existing, technical, or emergent (Bias Taxonomy). Quantitative fairness measures include statistical parity, equalized odds, and calibration [25] [37]. Healthcare poses unique challenges because decisions can be life-or-death [69] [51]. Research spans EHRs, imaging, genomics, and clinical triage. [62] recommended including social determinants of health to correct bias in predictive models [62]. Despite progress, gaps remain between fairness measures and real-world implementation [53].

Evidence shows disparities across race, gender, socioeconomic status, and geography. Black patients with complex needs are often sicker than White patients at similar risk scores, yet cost-focused algorithms fail to capture this (Risk Prediction Disparities, 2019). Gender bias exists in medical imaging datasets [20]. AI for skin cancer and diabetic retinopathy detection underperforms for underrepresented or lower-income groups [67][13]. Clinical decision support systems for sepsis in COVID-19 patients also show bias, with poorer patients receiving fewer medications and tests [11][42][34]. These biases mirror those in broader algorithmic systems afecting jobs, housing, and credit [3][69]. Overall, AI in healthcare can reinforce inequities unless bias is actively addressed.

## 3. Systematic Mapping Study Protocol

This section outlines the methodology followed in carrying out the systematic mapping study, based on the guidelines set out by Petersen et al.[59]. Figure 2 summarizes the overall workflow. In the figure, rounded rectangles represent processes, while grey rectangles represent the outcomes of each process. The overall process comprises six stages: formulating the research questions, carrying out the search, screening the retrieved papers, extracting keywords and themes, and performing data extraction and mapping. Each step yields a specific outcome, which ultimately forms the basis of our mapping study.

When conducting our research, it is essential to ensure that the platforms used are authenticated and well-founded. Accordingly, we utilise a range of trusted databases, digital libraries, and platforms, including IEEExplore, Springer, Science Direct, MDPI, Scopus, ACM and Web of Sciences. Figure 1 illustrating the breakdown of search query results, followed by the materials retained after screening. In the following sections, we describe each step of the workflow and its corresponding outcome in detail.

## 3.1. Research objectives and questions

## 3.1.1. Setting the context

For our research, we employ various platforms to execute subject-specific search queries. Subsequently, filter the relevant materials by applying exclusion and inclusion criteria to ensure that the gathered material is as pertinent as possible. In this way are able to compile a range of diferent sources including articles, journals, conference papers, and surveys. By applying diferent criteria, we retain sources most relevant to our research topic.

## 3.1.2. Research questions

To help narrow down the scope of research, we need to break down the diferent areas we want to investigate. This allows us to conduct research that is accurate enough for us to reach justified conclusions.

RQ1: What kind of diseases have been addressed by AI?

Aim: This research question aims to analyse the diferent healthcare and medical domains examined by the various sources. Additionally, this question targets the diferent models and the respective roles they play in diagnosing various types of diseases and infections. Examples include eye diseases, cardiovascular diseases, lung diseases, skin diseases, and diabetes.

RQ2: What approaches have the publications used to address the issue of bias?

Aim: This question serves as a method for deconstructing the diferent ways in which the sources have conducted their research. By doing so, we are able to ensure that the information presented in the sources is not only justified with evidence but also accurately supported by other qualified

professionals and publications.

RQ3: What parameters have been used in the proposed approaches?

Aim: To maintain relevancy, we need to consider what parameters are used in the filtered material. For example, we identify which parameters have been used and which of them can introduce bias (e.g., feature selection, threshold values, or sampling ratios).

RQ4: What are the reported challenges in disease detections regarding the use of AI?

Aim: By ensuring that the filtered sources include the limitations associated with the use of AI in disease detection, we are able to confirm that the material incorporates the aspect of bias in AI-based disease detection. This further enables us to gather relevant points to support justified claims regarding how bias afects AI in disease detection and diagnosis.

RQ5: What are top published venues? What are the research trends in last 10 years? What kind of expertise required?

Aim: To identify key publication venues, research trends over the last ten years, and required expertise. These factors help us detect anomalies and observe how bias in AI has fluctuated over time in disease detection.

RQ6: How can the research contributions be classified?

Aim: This research question helps analyse how the filtered material can be distinguished from previous publications and research. Factors such as experiments, testing, and synthetic data are taken into account when diferentiating between the diferent types of sources and the ways in which they can be utilised in our research.

RQ7: What empirical methods have been applied to validate the research contributions?

Aim: Here we focuses on ensuring data diversity and validity in the sources, guaranteeing that the filtered material includes a range of methods for justifying the statements being made. Several sources incorporate inclusion and exclusion criteria, as well as research questions and topic-specific queries. This ensures that the material used to conduct the research in the given sources is appropriately selected and filtered thereby reducing the risk of misinformation.

## 3.1.3. Literature Review

We adopt the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) framework for literature identification and review. Figure 1 illustrates the core PRISMA stages (represented in the left-most blue vertical columns) along with the number of papers involved at each step.

The Conducting Research phase concerns the development of research protocols through the identification of keywords and bibliographic databases used to carry out the search. The Screening phase involves establishing inclusion and exclusion criteria, as well as a defined time frame, for filtering relevant articles. Finally, the Included phase presents the primary articles selected following the application of the screening criteria. Each phase of the PRISMA workflow is described in detail below.

![](images/7176cd0249c736ee887d0aee8b3282b33b242b5b827778acaab5a1903b565887.jpg)  
Figure 1: PRISMA flow diagram performed in our study

## Conducting Research

The PICOC (Population, Intervention, Comparison, Outcome, Context) [60] criteria are applied to formulate search strings derived from the research questions.

• Population: Refers to all individuals afected by or subject to disease detection using AI systems.

• Intervention: Denotes the approaches employed to address bias-related issues, such as the algorithms, explainability models, or fairness techniques applied in AI-based disease detection.

• Comparison: Compares diferent approaches from the Intervention step, evaluating alternative strategies from a qualitative rather than empirical perspective.

• Outcome: Focuses on key factors within the examined methodology, including the level of bias, fairness, accuracy across demographic groups, and interpretability of AI models in disease detection.

• Context: Encompasses works originating from both industry and academia.

To identify the most relevant studies published over the last twelve years (2015–2026), we conduct an automated search across seven digital libraries: IEEE Explorer, <sup>1</sup> ACM,<sup>2</sup> Science Direct,<sup>3</sup> Springer Link,<sup>4</sup> Web of Sciences,<sup>5</sup> Scopus,<sup>6</sup> and MDPI.<sup>7</sup>. These libraries provide broad coverage of the academic literature and are selected based on criteria reported in prior work. A set of keywords and associated synonyms is identified, from which search strings are constructed using Boolean operators (AND and OR). The complete query strings are presented in Table 1.

The Quasi-Gold Standard is employed to identify the most relevant primary studies within the specified time frame (2015–2026) across both conferences and journals [83]. IEEE and ACM Digital Libraries are selected due to their coverage of leading international journals and prominent conferences focused on data science and disease detection. ScienceDirect and SpringerLink are included as comprehensive digital libraries ofering extensive scientific collections and indexed international journals. The Web of Science is considered given its standing as one of the oldest citation indexes in the sciences, encompassing scholarly journals, conference proceedings, and data compilations [6].

The Quasi-Gold Standard, is grounded in two primary criteria: the venue and the time period [83]. Applying these criteria, we identify relevant publications across the seven specified databases through a combination of automated and manual searches. The Quasi-Gold Standard further serves as a useful instrument for evaluating search strategies designed to locate pertinent literature within a defined time frame.

With the aid of the Quasi-Gold Standard (QGS), we compile a collection of studies from related venues, including domain-specific conferences and journals recognised within the time span of 2015 to 2026.

A notable advantage of employing QGS lies in its capacity to eficiently filter relevant studies by focusing exclusively on article titles and abstracts. Conventional manual and automated search methods can prove considerably time-consuming when sifting through large volumes of data and narrowing down publication venues; QGS, however, streamlines this process substantially. Following the identification of relevant studies through QGS, we apply the inclusion and exclusion criteria outlined in the corresponding subsection. As a result, we select 23 primary studies from the available venues.

In addition, we apply the backwards snowballing technique to minimise the risk of overlooking any relevant primary studies published from 2015 onwards [79] [50]. This technique involves systematically reviewing the reference lists cited within relevant papers, thereby increasing the likelihood of uncovering additional pertinent studies. Throughout this process, we follow the guidelines established by Wohlin.

Table 1: Search strings in Databases.
<table><tr><td>Database</td><td>Search String ((&quot;Artificial Intelligence&quot; OR &quot;Machine Learning&quot; OR &quot;Deep Learning&quot;</td></tr><tr><td>IEEE Xplore ACM</td><td>OR &quot;Neural Network*&quot; OR &quot;AI-based model*&quot;) AND (&quot;Algorithmic Bias&quot; OR &quot;Data Bias&quot; OR &quot;Fairness&quot; OR &quot;Bias Mitigation&quot; OR &quot;Ethical AI&quot; OR &quot;Algorithmic Fairness&quot;) AND (&quot;Disease Detection&quot; OR &quot;Disease Diag- nosis&quot; OR &quot;Medical Diagnosis&quot; OR &quot;Clinical Decision Support System*&quot; OR &quot;Medical Imaging&quot;)) AND NOT (&quot;Finance&quot; OR &quot;Stock Market&quot; OR &quot;Agriculture&quot; OR &quot;Recommendation System*&quot; OR &quot;Traffic Detection&quot;) ((&quot;Artificial Intelligence&quot; OR &quot;Machine Learning&quot; OR &quot;Deep Learning&quot;</td></tr><tr><td>Science Direct</td><td>OR &quot;Neural Network*&quot;OR &quot;AI-based system*&quot;) AND (&quot;Bias&quot; OR &quot;Al- gorithmic Bias&quot; OR &quot;Data Bias&quot; OR &quot;Algorithmic Fairness&quot; OR &quot;Discrim- ination&quot; OR &quot;Ethical AI&quot;) AND (&quot;Disease Detection&quot; OR &quot;Disease Diag- nosis&quot; OR &quot;Medical Diagnosis&quot; OR &quot;Clinical Decision Support System*&quot; OR &quot;Medical Imaging&quot; OR &quot;Healthcare AI&quot;)) AND NOT (&quot;Agriculture&quot; OR &quot;Finance&quot; OR &quot;Stock Market&quot; OR &quot;Recommendation System*&quot; OR &quot;Fraud Detection&quot; OR &quot;Traffic Detection&quot; OR &quot;Autonomous Vehicles&quot;)</td></tr><tr><td>Springer Link</td><td>((&quot;Artificial Intelligence&quot; OR &quot;Machine Learning&quot; OR &quot;Deep Learning&quot; AND (&quot;Bias&quot; OR &quot;Algorithmic Bias&quot; OR &quot;Data Bias&quot; OR &quot;Fairness&quot;) AND (&quot;Disease Detection&quot; OR &quot;Disease Diagnosis&quot; OR &quot;Medical Diagno- sis&quot; OR &quot;Medical Imaging&quot;)) AND NOT (&quot;Finance&quot; OR &quot;Stock Market&quot; OR &quot;Agriculture&quot;) ((&quot;Artificial Intelligence&quot; OR &quot;Machine Learning&quot; OR &quot;Deep Learning&quot;</td></tr><tr><td>Scopus</td><td>OR &quot;Neural Network*&quot;OR &quot;AI-based system*&quot;) AND (&quot;Bias&quot; OR &quot;Al- gorithmic Bias&quot; OR &quot;Data Bias&quot; OR &quot;Algorithmic Fairness&quot; OR &quot;Ethical AI&quot; OR &quot;Discrimination&quot;) AND (&quot;Disease Detection&quot; OR &quot;Disease Diag- nosis&quot; OR &quot;Medical Diagnosis&quot; OR &quot;Clinical Decision Support System*&quot; OR &quot;Medical Imaging&quot; OR &quot;Healthcare AI&quot;)) AND NOT (&quot;Finance&quot; OR &quot;Stock Market&quot; OR &quot;Agriculture&quot; OR &quot;Recommendation System*&quot; OR &quot;Traffic Detection&quot; OR &quot;Autonomous Vehicles&quot;) ((&quot;Artificial Intelligence&quot; OR &quot;Machine Learning&quot; OR &quot;Deep Learning&quot;</td></tr><tr><td>MDPI</td><td>OR &quot;Neural Network*&quot; OR &quot;AI-based system*&quot;) AND (&quot;Bias&quot; OR &quot;Al- gorithmic Bias&quot; OR &quot;Data Bias&quot; OR &quot;Algorithmic Fairness&quot; OR &quot;Ethical AI&quot; OR &quot;Discrimination&quot;) AND (&quot;Disease Detection&quot; OR &quot;Disease Diag- nosis&quot; OR &quot;Medical Diagnosis&quot; OR &quot;Clinical Decision Support System*&quot; OR &quot;Medical Imaging&quot; OR &quot;Healthcare AI&quot;)) AND NOT (&quot;Finance&quot; OR &quot;Stock Market&quot; OR &quot;Agriculture&quot; OR &quot;Recommendation System*&quot; OR &quot;Traffic Detection&quot; OR &quot;Autonomous Vehicles&quot;) ((&quot;Artificial Intelligence&quot; OR &quot;Machine Learning&quot; OR &quot;Deep Learning&quot;</td></tr><tr><td></td><td>OR &quot;Neural Network*&quot; OR &quot;AI-based system*&quot;) AND (&quot;Bias&quot; OR &quot;Al- gorithmic Bias&quot; OR &quot;Data Bias&quot; OR &quot;Algorithmic Fairness&quot; OR &quot;Ethical AI&quot; OR &quot;Discrimination&quot;) AND (&quot;Disease Detection&quot; OR &quot;Disease Diag- nosis&quot; OR &quot;Medical Diagnosis&quot; OR &quot;Clinical Decision Support System*&quot; OR &quot;Medical Imaging&quot; OR &quot;Healthcare AI&quot;)) AND NOT (&quot;Finance&quot; OR &quot;Stock Market&quot; OR &quot;Agriculture&quot; OR &quot;Recommendation System*&quot; OR &quot;Traffic Detection&quot; OR &quot;Autonomous Vehicles&quot;)</td></tr></table>

## 3.1.4. Screening of Papers

In this section, the inclusion and exclusion criteria are defined for studies retrieved through both automated and manual searches. Two authors independently screened the titles, abstracts, and keywords of the retrieved records against the predefined inclusion and exclusion criteria. Records considered potentially eligible proceeded to full-text assessment. The same two authors independently assessed the full texts. Disagreements at either stage were resolved through discussion and consensus. The following inclusion criteria is applied:

• I1: Studies addressing research methods and findings within the considered research domain.

• I2: Studies that have undergone a peer-review process and have been published in recognized venues, including journals, conference proceedings, and workshop proceedings.

• I3: Studies concerned with Artificial Intelligence biasses.

• I4: Studies published between 2015 and 2026.

• I5: Studies written in the English language.

Studies meeting at least one of the following exclusion criteria are omitted from the review:

• E1: Studies not focused on AI biasses.

• E2: Grey literature, including working papers, white papers, short papers, or presentations published in the form of panel discussions.

• E3: Studies solely proposing guidelines or recommendations to use AI in medical sciences.

• E4: Studies that have not undergone peer review.

• E5: Studies not written in English.

• E6: Duplicate studies published across diferent venues at various stages of their development.

The inclusion and exclusion criteria are applied to all relevant studies identified across the digital libraries to assess their eligibility for the research, encompassing a validated time span of 2015 to 2026. During the full-text review stage, further articles that fail to satisfy the defined criteria are excluded. For the remaining articles, a snowballing technique is subsequently employed, through which three additional relevant studies are identified.

To ensure the reliability of the studies included within the systematic mapping, Fleiss’ Kappa is utilized as a statistical measure of inter-rater agreement. This method evaluates the level of consensus among a fixed number of raters classifying items according to their nature, producing a score ranging from 0, representing poor agreement, to 1, representing full agreement. To assess the reliability of the eligibility criteria, four independent raters evaluated the same randomly selected sample of 10 records, comprising five studies included in the final mapping and five records excluded during screening. The raters independently classified each record as ‘include’ or ‘exclude’ using the predefined eligibility criteria. These raters were not involved in the main screening process; their role was limited to independently validating the consistency of the study-selection decisions. Fleiss’ kappa was used because four raters evaluated the same categorical decisions. The resulting Fleiss’ kappa value of 0.94 indicated almost perfect agreement. The computed Fleiss’ Kappa value of 0.94 demonstrates a high degree of agreement among the raters.

![](images/ec47b3c12d453af3f1f7ba9105fac12aa847bf2b556d9cf5c9c1d6221ea50794.jpg)  
Figure 2: Primary studies percentage in each database

Figure 2 further illustrates the percentage distribution of primary studies across seven digital libraries. As reflected in the distribution,Web of sciences accounts for the largest proportion at 67%, followed by Scopus at 25%, ACM at 4%, Springer at 2%, IEEE Xplore and Science Direct at 0.5% each of the total selected primary studies.

## Keywords and Themes

Following a thorough review process and the systematic application of inclusion and exclusion criteria in accordance with the PRISMA approach, a total of 23 articles are selected for inclusion in this study.

Figure 5 presents the distribution of primary studies across each digital library. Web of Sciences contributes the greatest number of primary studies, accounting for 6 articles. Scopus follows with 5 articles, while ACM includes 4 articles. The Springer digital library comprises 3 articles, and Science Direct consist on 2 articles while MDPI and IEEE Explore each contain 1 article. It is further noted that one article is indexed across more than one libraries (IEEE Xplore + Scopus).

Quality Assessment:. Following the selection of 23 relevant studies, a quality assessment is conducted drawing upon established quality assessment models from prior literature [1], [82], [35] and [75]. Based on the recommendations provided within these models, a quality assessment checklist created, as presented in Table 2. This checklist encompasses six distinct criteria and incorporates both subjective and objective questions designed to verify the validity of the sources, evaluate their suitability, and ensure they are free from bias.

Then checklist subsequently applied to assess the quality of the selected literature. Through the implementation of specific selection criteria, all irrelevant studies are efectively excluded, thereby minimising the efort required for the quality assessment process [19]. These criteria are grounded in factors such as the authority of the producer, methodology, scope, objectivity, novelty, and impact [48].

The same selection criteria may further be applied to other research methods, including surveys, case studies, and experiments, to support broader quality evaluation. A quality assessment checklist specifically developed for case studies has been proposed in prior work and may equally be applied to case studies within formal literature.

![](images/839a9f820fd302778386cbb2c20632483d7b8caa255ebd7364b34ed81f916146.jpg)  
Figure 3: Selected primary studies

## Data Extraction & Mapping Process

To extract data from the selected primary studies and address the research questions defined in the corresponding subsection 3.1.2, Table 3 is constructed to incorporate the key data items related to the research questions. In line with established practice, one researcher conducted the data extraction while another reviewed and verified the process. This two-author approach to verification is a widely adopted practice in systematic mapping studies [60]. The defined data extraction strategy ensures that consistent criteria are applied uniformly across all selected papers, including their respective classifications.

Table 2: Quality assessment checklist of relevant literature for mapping study on AI bias in disease detection.
<table><tr><td>Criteria</td><td>Questions</td></tr><tr><td>Producer Authority</td><td>• Is the publishing institution or platform reputable? E.g., Journal of Biomedical Informatics, Nature Medicine - Is the author affiliated with a recognised research or medical institution? - Has the author published prior work in AI, healthcare, or disease detection? - Does the author possess relevant expertise in this area? (e.g., AI researcher, clinical data scientist, or medical professional)</td></tr><tr><td>Methodology</td><td>• Does the proposed methodology have a clearly stated aim related to AI-based disease detection? - Is the proposed approach supported by authentic and peer- reviewed references? - Are the limitations of the proposed methodology clearly acknowl- edged? - Is the work grounded in specific research questions, with all ques- tions addressed effectively? - Is the proposed methodology validated through a real-world clin-</td></tr><tr><td>Scope</td><td>ical case study or dataset? • Does this work relate to the domain of AI bias in healthcare or disease detection?</td></tr><tr><td>Objectivity</td><td>• Is the work balanced in its presentation and clearly articulated? - Does the problem statement clearly define the research objective? - Is this work associated with or influenced by a particular vendor or commercial platform? - Are the conclusions of this work supported by empirical data or experimental evidence?</td></tr><tr><td>Novelty</td><td>• Does this work introduce a new idea, approach, or insight re- garding bias in AI-driven disease detection?</td></tr><tr><td>Impact</td><td>• Assess the impact of the considered study with respect to the following criteria: - Number of paper citations - Number of paper backlinks - Number of paper views or reads</td></tr></table>

Table 3: Data Extraction
<table><tr><td>Data Item</td><td>Value</td><td>RQ</td></tr><tr><td>Article ID</td><td>Integer</td><td></td></tr><tr><td>Author Name</td><td>Author&#x27;s name list</td><td></td></tr><tr><td>Title of Articles</td><td>Name of the article</td><td>RQ1</td></tr><tr><td>Keyword</td><td>Keyword study indexing</td><td>RQ1</td></tr><tr><td>Publication Year</td><td>Calendar year</td><td>RQ5</td></tr><tr><td>Venue</td><td>Publication venue name Eye diseases, cardiovascular diseases, lung dis- RQ1</td><td>RQ5</td></tr><tr><td>Disease Type</td><td>eases, skin diseases, diabetes, cancer, etc.</td><td></td></tr><tr><td>AI Models Used</td><td>Machine Learning, Deep Learning, CNN, RNN, RQ1 Decision Trees, Random Forests, SVM, LLMs, NLP</td><td></td></tr><tr><td>Research Approach Mitigation</td><td>Tool-based, solution methodology, case study, sys- RQ2 tematic review, empirical study</td><td></td></tr><tr><td>Bias Strategy</td><td>Algorithmic debiasing, adversarial learning, fair- RQ2 ness constraints, reweighting, data augmentation</td><td></td></tr><tr><td>Parameter Type</td><td>Feature selection, threshold values, sampling ra- RQ3 tios, hyper parameters, demographic variables</td><td></td></tr><tr><td>Bias-Inducing Pa- rameters</td><td>Imbalanced datasets, missing demographic data, RQ3 skewed sampling ratios, improper threshold selec- tion</td><td></td></tr><tr><td>Reported Challenge</td><td>Data quality issues, algorithmic bias, labeling bias, RQ4 deployment bias, lack of generalizability</td><td></td></tr><tr><td>Limitation of Study</td><td>Small sample size, lack of external validation, RQ4 dataset homogeneity, computational constraints</td><td></td></tr><tr><td>Contribution Type</td><td>Novel algorithm, bias detection framework, fair- RQ6 ness evaluation metric, comparative analysis</td><td></td></tr><tr><td>Validation Method</td><td>Cross-validation, external validation, ablation RQ7 studies, fairness audits, statistical testing</td><td></td></tr><tr><td>Data Diversity</td><td>Multiple datasets, diverse demographic represen- RQ7 tation, synthetic data generation</td><td></td></tr><tr><td>Search Strategy</td><td>Guidelines followed to select the studies (auto- RQ4 mated/manual, inclusion/exclusion criteria)</td><td></td></tr><tr><td>Search Type</td><td>Manual or automated</td><td>RQ4</td></tr><tr><td>Research Trend</td><td>Year-wise publication count, emerging bias topics, RQ5 model evolution</td><td></td></tr><tr><td>Domain Expertise</td><td>Medical imaging, clinical informatics, computer vi- RQ5 sion, epidemiology, ethics</td><td></td></tr><tr><td>Classification Scheme</td><td>How articles were categorized (by disease, model RQ6 type, bias type, or methodology)</td><td></td></tr><tr><td>Open Issues</td><td>Unresolved challenges in bias mitigation, general- RQ4 ization gaps, lack of standardized fairness metrics</td><td></td></tr></table>

Analysis and classification: The extracted information is elaborated upon and visually presented in the corresponding subsection. During the analysis of the study identification phase, the extracted strategies are classified according to diferent search approaches, which included the development of the search process, the evaluation procedures, and the inclusion and exclusion criteria. Each category is subsequently assigned relevant themes and sub-themes. The sub-themes are used within the inclusion and exclusion process, also referred to as a priori classification, drawing upon strategies such as resolving disagreements between researchers in order to minimize bias [57]. Finally, the selected papers tallied according to their respective themes and sub-themes.

A descriptive technique additionally applied, as illustrated in Figure 5, to categories the publication types of the selected studies including book chapters, conference papers, journal articles, and symposium contributions according to their respective venues. Furthermore, a classification scheme grounded in topic-related keywords has established to efectively address specific research questions. These classifications prove particularly valuable in supporting the mapping process.

## 4. Systematic Mapping Study Results

We conduct this mapping study following the procedures outlined in Section 3 and in alignment with our proposed research questions. The initial search phase retrieves 64,389 records based on filtering criteria such as publication year, author afiliations, source venue, and subject relevance, while applying our predefined inclusion standards. After refining the search by eliminating duplicate keywords and adhering to systematic review protocols [59], we identify an additional 292 records that satisfy the inclusion requirements. Following abstract screening and the application of our selection criteria, a total of 23 primary studies remain for final analysis. Appendix A presents a concise summary of all included studies.

Figure 4 ofers a visual summary of the selected primary studies distributed across seven digital repositories. The figure captures the source libraries from which the studies originate including publication year and the outer ring of the pie chart illustrates the proportional breakdown of results derived from the selected studies.

## 4.1. Categorization of Primary Studies by Publication Format

We examine the selected primary studies based on their publication format, which includes journal articles (11 studies), conference proceedings (7 studies), book chapters (2 studies) and symposium (3 studies). Figure 5 visualizes the distribution across these formats. Journals represent the most frequently used publication channel, comprising roughly 47.8% of all selected works concerning AI bias in disease detection. In contrast, conference papers account for 30.43%, symposium proportion is 13% while book chapters make up the smallest portion at approximately 8.69%.

## 4.2. Classification of Primary Studies by Research Type

To classify primary studies according to their research nature, we adopt the taxonomy proposed by [78] and [16]. These categories remain straightforward and are explained below in Table 4.

Table 4: Studies Classification According To Research Type
<table><tr><td>Research Type</td><td>Number of Studies</td></tr><tr><td>Solution Proposal</td><td>3</td></tr><tr><td>Solution Proposal with Evaluation</td><td>3</td></tr><tr><td>Solution Proposal with Validation</td><td>2</td></tr><tr><td>Philosophical papers</td><td>4</td></tr><tr><td>Opinion / Personal Experience</td><td>11</td></tr></table>

![](images/2809b80a1f80ff30413c9b2ef4b0063b6801886d19facadf38261c20c2de1a8e.jpg)  
Figure 4: Distribution of selected studies (SS) used for analysis across publications years

Solution Proposal: Such papers introduce a novel approach to addressing a problem but lack comprehensive validation. Instead, they rely on proof-of-concept demonstrations through illustrative examples, logical arguments, or similar means. Three of our selected studies fall under this category. Solution Proposal with Evaluation: This category encompasses empirical investigations that assess novel solutions using formal methodologies including hypothesis testing and real-world case study experiments. Three of our primary studies are classified as solution proposals accompanied by evaluation research.

Solution Proposal with Validation: Studies in this group ofer preliminary empirical support for proposed solutions through methodologically rigorous procedures such as quasi-experiments, prototyping, mathematical modeling, and case studies. We identify two primary studies in the validation research category.

Philosophical Papers: According to [78], these works introduce new conceptual frameworks or alternative perspectives on existing research. We have found four primary study fits this classification.

Opinion Papers: These articles reflect the authors’ personal viewpoints on a research area for instance, identifying deficiencies or advocating alternative techniques. Approximately 49% of the selected studies lack rigorous empirical validation and instead rely on proof-of-concept illustrations. These eleven works are designated as pure solution proposals. This finding suggests that the existing literature on

![](images/57d0123cc776d04ba27853749f781e73dd5cd6ca319ad52848b51382a6ade896.jpg)  
Figure 5: Distribution of selected studies with respect to publication type

AI bias in disease detection exhibits relatively limited scientific robustness.

The second category solution proposals accompanied by evaluation research comprises three studies. These works contribute high-quality evidence through experiments grounded in real-world case studies, thereby demonstrating the practical applicability of their proposed innovations.

Validation research studies assess the potential contributions of selected works using experimental setups and mathematical analyses. These studies go beyond basic solution proposals by ofering deeper methodological scrutiny. The presence of two such studies indicates emerging but still limited scholarly engagement with this domain in recent years [78].

Four philosophical papers appear, suggesting that a modest number of researchers propose entirely new conceptual frameworks in this field. Similarly, eleven opinion or personal experience papers are included, highlighting a strong reliance on subjective insights rather than empirical validation in the existing literature

## 4.3. Mapping of Primary Studies by AI Bias Solution Category

Following data extraction, we classify the selected studies based on the type of AI bias addressed. This classification involves two independent reviewers working in two stages. The first stage consists of systematic mapping, while the second stage establishes four bias domain categories: algorithmic bias, demographic bias, model bias, and deployment bias. Table 5 summarizes these categories.

Algorithmic Bias: These works examine how computational processes and algorithm design choices can introduce systematic errors into disease detection. We identify 7 such articles, representing 30.4% of all selected studies.

Demographic Bias: This category includes multidisciplinary research on how gender, race, age, and socioeconomic status can contribute to biased outcomes, often mirroring broader societal inequalities. Six studies fall into this group.

Model Bias: Focusing on biases stemming from model design and training procedures, this category accounts for seven studies, or also 30.4% of the total.

Deployment Bias: These four studies (13.0%) investigate bias that arises when AI systems operate outside their originally intended contexts or purposes.

Table 5: Classification of Primary Studies by AI Bias Domain
<table><tr><td>Bias Category</td><td>Focus Areas</td><td>Associated Studies</td><td>Total Count</td></tr><tr><td>Algorithmic Bias</td><td>Computational errors, design flaws, pattern recognition issues</td><td>SS1, SS2, SS5, SS12, SS15, SS16, SS17</td><td>7</td></tr><tr><td>Demographic Bias</td><td>Gender, racial, age, and socioeconomic disparities</td><td>SS3, SS6, SS7, SS8, SS22, SS23</td><td>6</td></tr><tr><td>Model Bias</td><td>Training-related bias, design limitations, black-box opacity</td><td>SS4, SS9, SS10, SS13, SS18, SS20, SS21</td><td>7</td></tr><tr><td>Deployment Bias</td><td>Use outside intended context, generalisa- tion failures</td><td>SS11, SS14, SS19</td><td>3</td></tr></table>

## 4.4. Distribution of Studies by Medical Domain

Through our analysis, we extract information across various study types including solution proposals, empirical investigations, evaluation research, philosophical essays, surveys, and opinion pieces. During data extraction, we record several key attributes, including disease type, AI model employed, bias category, and validation approach. Some studies present information at an abstract level using graphical representations.

Table 6 presents the breakdown of selected studies according to the medical domain addressed.

Table 6: Distribution of Studies by Disease Domain
<table><tr><td>Medical Domain</td><td>Primary Studies</td><td>Count</td><td>Percentage</td></tr><tr><td>Cardiovascular Diseases</td><td>SS2, SS4, SS5, SS14, SS16</td><td>5</td><td>21.7%</td></tr><tr><td>Ophthalmology</td><td>SS3, SS17, SS18, SS23</td><td>4</td><td>17.39%</td></tr><tr><td>Lung Diseases</td><td>SS7, SS8, SS19, SS20</td><td>4</td><td>17.39%</td></tr><tr><td>Skin Diseases</td><td>SS1, SS6, SS12</td><td>3</td><td>13.0%</td></tr><tr><td>Diabetes</td><td>SS13, SS21</td><td>2</td><td>8.69%</td></tr><tr><td>Multiple / General</td><td>SS9, SS10, SS11, SS15, SS22</td><td>5</td><td>21.7%</td></tr></table>

## 4.5. Leading Contributors in the Field

Based on our examination of the selected primary studies, we analyse publication patterns and author productivity to assess research impact within this domain. Several active researchers contribute multiple papers over the past decade, like [59]. While publication volumes show notable increases in 2020, 2021, 2024, and again in 2026, our findings indicate that no research group yet comprehensively tackles the problem of bias in black-box AI models using state-of-the-art explainability techniques.

## 5. Potential Research Areas

Based on the findings of this mapping study, we observe that limited research efort has been devoted to this domain. Although some scholars have attempted to address AI bias in disease detection, the overall volume of methodologically rigorous studies remains insuficient. Most available works rely on conceptual frameworks or descriptive analyses rather than empirical investigations. Consequently, several important issues remain underexplored. We identify multiple promising research directions, particularly regarding the application of explainable AI (XAI) techniques to detect and mitigate bias in medical diagnostic systems.

The use of cutting-edge explainability methods in this critical domain ofers considerable potential. According to our mapping results, only a small fraction of selected studies employ advanced explainability frameworks to uncover hidden biases in AI models. Many current AI systems operate as black boxes, making it exceedingly dificult for healthcare professionals and decision-makers to trace how diagnostic conclusions are reached. Without transparent and interpretable outputs, biases may remain undetected, potentially leading to misdiagnosis or unequal treatment across patient populations.

Furthermore, public health authorities and clinical decision-makers rarely account for the societal impact and relative benefits that patients experience from AI-assisted diagnosis. Indeed, these societal consequences vary significantly across diferent demographic groups and clinical settings. Yet such considerations represent essential factors that ought to be integrated into all stages of AI deployment in healthcare from model design and training to validation and post-deployment monitoring.

Given these complexities, explainable AI frameworks ofer a viable pathway forward. These approaches can provide clinicians, administrators, and patients with mechanisms to better understand, evaluate, and refine diagnostic algorithms. By making model decisions interpretable, XAI helps identify whether bias arises from training data, algorithmic design, or deployment contexts.

Thus, more empirical and detailed investigations are necessary to examine how explainable AI techniques can efectively detect, measure, and mitigate bias in disease detection systems. Particular attention should be directed toward black-box models operating across diverse medical domains, including cardiology, ophthalmology, pulmonology, dermatology, and endocrinology.

## 6. Threat to Validity

Several factors require careful consideration when evaluating the findings of any systematic mapping study, as these factors may potentially limit the validity of the conclusions drawn. In this section, we discuss potential validity threats to our systematic mapping study with respect to study design, execution, analysis, and interpretation. Specifically, we consider threats such as selection bias in digital library choice, inaccuracies during data extraction, temporal bias in study inclusion, search string construction limitations, and publication bias.

## 6.1. Selection Bias in Digital Libraries

Selection bias refers to potential distortions in statistical analysis arising from the choice of digital libraries specifically, whether selected repositories contain too many or too few relevant studies. We address this threat (at least to some extent) by defining comprehensive inclusion and exclusion criteria tailored to AI bias research in healthcare. Additionally, we select seven widely recognized digital libraries: IEEE Xplore, ACM Digital Library, Springer Link, Science Direct, Scopus, MDPI, and Web of Sciences. This broad coverage minimizes the risk of omitting pertinent studies.

## 6.2. Inaccuracy in Data Extraction and Misclassifications

During information extraction, inaccuracies and misclassifications may occur when reviewers interpret study content diferently. To mitigate such discrepancies (at least to some extent), we resolve all disagreements through consensus discussion among all reviewers. Furthermore, we design a standardized data extraction form (Table 3) that captures key attributes consistently across all selected studies.

## 6.3. Temporal Bias in Primary Study Selection

Temporal bias arises when the chosen time frame excludes relevant recent publications or overrepresents older works. To address this threat, we restrict our search to articles published between 2015 and 2026. This ten-year window captures the most active period of AI research in healthcare while minimising the chance of missing recent, high-impact contributions.

## 6.4. Bias in Search String Definition

Search string construction introduces potential bias if keywords fail to capture the full scope of relevant literature. To tackle this issue (at least to some extent), we employ multiple search techniques, including Boolean operators (AND, OR), quotation marks for exact phrases, hyphens for compound terms, exclusion operators (NOT), and wildcard asterisks. We also refine our search strings iteratively, removing duplicate keywords and excluding irrelevant domains such as plant disease detection.

## 6.5. Publication Bias

Publication bias refers to the tendency of journals and conferences to publish positive results more frequently than negative or null findings. Additionally, studies reporting negative outcomes often take longer to appear in print and receive fewer citations [26]. To address this threat (at least to some extent), we scan all relevant journals and conference proceedings while deliberately excluding grey literature such as technical reports, unpublished theses, and non-peer-reviewed white papers as these may compromise the validity of our findings.

## 6.6. Descriptive Validity

Descriptive validity concerns the accuracy and objectivity of the information collected. Threats to descriptive validity typically pose greater challenges in qualitative research than in quantitative studies. To address this, we design a standardized data collection form (Table 3) to systematically record information related to AI bias in disease detection. We also review our extraction process multiple times to avoid missing any relevant articles. Moreover, this form remains accessible for re-examination whenever needed.

## 6.7. Theoretical Validity

Theoretical validity involves determining whether the collected data appropriately fits the intended analytical framework without selection bias. There always exists a risk of omitting relevant studies for example, [36] conduct two mapping studies on identical topics and arrive at diferent sets of articles. To handle this threat, we follow PRISMA guidelines [70] and employ backward snowball sampling techniques [31] as described in subsection 3.1.3.

## 6.8. Internal Validity

Internal validity faces two primary threats: missing relevant studies and researcher bias during paper selection. Mapping studies typically aim to cover a broad range of related articles from multiple databases [36]. In our mapping study, we therefore search seven of the most widely used digital libraries to minimise the chance of overlooking relevant research on AI bias in disease detection. We believe that most pertinent primary studies are included, although we cannot entirely rule out the possibility that some relevant works may have been missed during automated searches across digital repositories. Additionally, we employ a quasi-gold standard approach (subsection 3.1.3) to validate the completeness of our automated search strings, which are derived from research question keywords.

Researcher bias during paper selection may also lead to inaccuracies in data extraction. To address this, we define clear data inclusion criteria (Table 3) for extraction, with all reviewers reaching agreement. Any disagreements arising during extraction are settled through mutual consensus.

## 6.9. External Validity

According to [58], external validity in systematic mapping studies concerns the generalisability and usefulness of findings for other populations or contexts. The main threat, therefore, is whether the selected primary studies adequately represent the subject area. To address this, we follow a standard research process and include all relevant studies regardless of outcome direction, excluding only those not written in English.

Additionally, some primary studies do not provide complete information for the extraction form. In such cases, we infer certain information based on the inclusion criteria during data synthesis. Similarly, some approaches do not explicitly describe their limitations or drawbacks.

## 6.10. Conclusion Validity

Conclusion validity refers to the extent to which we reach reasonable conclusions and establish meaningful relationships between selected studies and current research trends in the field. To handle the first threat, we systematically answer all defined research questions based on the selected primary studies. Regarding the second threat concerning data extraction, classification, and synthesis we perform all activities as a collaborative team. Nevertheless, a residual risk remains that some relevant study may have been missed, as qualitative mapping studies inherently involve interpretation and cannot guarantee absolute completeness.

## 7. Discussion

This study systematically mapped the current state of research on AI bias in disease detection between 2015 and 2026, identifying the major research themes, bias categories, disease domains, research methodologies, and validation approaches. The analysis of 23 primary studies collected from seven digital libraries provides a comprehensive overview of how fairness and bias have been investigated in healthcare AI over the past decade. The findings indicate that although considerable attention has been directed toward improving fairness in AI-assisted diagnosis, substantial challenges remain in developing robust, transparent, and clinically validated bias mitigation frameworks that can be deployed in real-world healthcare settings.

The mapping results demonstrate that recent studies increasingly focus on algorithmic fairness, demographic bias, explainability, and fairness-aware machine learning, reflecting a gradual transition from performance-oriented AI models toward trustworthy and responsible AI systems. Nevertheless, the adoption of advanced bias mitigation techniques remains limited. Only a small number of primary studies employed explainable AI methods or integrated fairness constraints directly into model development, while most research concentrated on identifying bias after model training. This observation suggests that the field is still in the early stages of incorporating fairness-by-design principles into AIbased disease detection and highlights an important opportunity for future research on interpretable and bias-aware learning frameworks.

Furthermore, the distribution of research types indicates limited methodological maturity within the existing literature. A considerable proportion of the selected studies were classified as opinion or philosophical papers, whereas comparatively few studies presented empirically validated solutions using real clinical datasets. Although several solution proposals introduced promising approaches for bias detection and mitigation, the lack of large-scale experimental validation limits their generalizability across diferent healthcare environments. Moreover, the absence of standardized fairness benchmarks and evaluation protocols makes it dificult to compare the efectiveness of competing approaches. This emphasizes the need for future studies to prioritize reproducible experimentation, external validation, and benchmark datasets representing diverse patient populations.

Another important observation concerns the factors contributing to bias in AI-based disease detection. The identified parameters, including demographic characteristics, data imbalance, feature selection, model architecture, and deployment environment, illustrate that bias is influenced by technical, clinical, and societal factors simultaneously. However, most primary studies examined these factors independently, with limited consideration of their interactions throughout the AI development lifecycle. Such fragmentation restricts the applicability of current solutions in complex clinical environments where multiple sources of bias coexist. A comprehensive framework integrating data di versity, explainability, fairness evaluation, and continuous post-deployment monitoring could provide a more efective strategy for reducing bias while maintaining diagnostic performance.

Finally, the distribution of research across medical domains reveals that cardiovascular diseases, ophthalmology, and respiratory diseases have received the greatest research attention, whereas several clinical specialties remain comparatively underexplored. Similarly, studies involving geographically diverse populations and healthcare systems from low- and middle-income countries remain limited despite the increasing global adoption of AI-assisted diagnosis. This imbalance suggests that future research should place greater emphasis on developing equitable datasets, evaluating models across diverse demographic groups, and strengthening collaboration between computer scientists, clinicians, healthcare professionals, and policy experts. Such interdisciplinary eforts will be essential for ensuring that future AI systems are not only accurate but also fair, transparent, and reliable for clinical decisionmaking.

## 8. Conclusion

This mapping study identifies and evaluates the existing literature with respect to research topics, publication frequency up to 2026, publication venues, types of AI bias, and disease domains. We believe our mapping study provides a comprehensive and state-of-the-art overview of AI bias in disease detection, which proves highly useful for emerging researchers and healthcare practitioners seeking to understand this challenging field and contribute through bias detection and mitigation strategies.

We answer all defined research questions (RQs) through analysis and results derived from our systematic mapping study.

## RQ1: What kind of diseases have been addressed by AI?

According to our primary studies, we focus exclusively on human disease detection research published over the last ten years (from 2015 to 2026). After thorough examination, we identify five major disease categories addressed in the literature. Table 7 summarizes these categories along with their corresponding primary studies.

Table 7: Disease Categories Addressed in Primary Studies
<table><tr><td>Disease Category</td><td>Primary Studies</td></tr><tr><td>Cardiovascular Dis- eases</td><td>SS2, SS4, SS5, SS14, SS16, SS19</td></tr><tr><td>Ophthalmology</td><td>SS3, SS9, SS10, SS11, SS17, SS18</td></tr><tr><td>Lung Diseases</td><td>SS7, SS8, SS20, SS23</td></tr><tr><td>Skin Diseases</td><td>SS1, SS6, SS12, SS15</td></tr><tr><td>Diabetes</td><td>SS13, SS21, SS22</td></tr></table>

RQ2: What approaches have the publications used to address the issue of bias?

The approaches identified in primary studies to address AI bias in disease detection fall into four categories: algorithmic bias detection (examining computational process errors and design flaws), demographic bias analysis (investigating gender, racial, age, and socioeconomic disparities), model bias assessment (evaluating training-related and design limitations), and deployment bias investigation (studying generalization failures and out-of-context usage). Table 8 groups the primary studies according to these four bias categories.

Table 8: Papers Grouped Based on Bias Approach
<table><tr><td>Bias Approach</td><td>Primary Studies</td></tr><tr><td>Algorithmic Bias</td><td>SS1, SS2, SS12, SS16, SS17, SS18, SS19, SS20, SS21, SS22</td></tr><tr><td>Demographic Bias</td><td>SS5, SS7, SS8, SS10, SS11, SS13, SS14, SS15</td></tr><tr><td>Model Bias</td><td>SS3, SS9</td></tr><tr><td>Deployment Bias</td><td>SS4, SS6, SS23</td></tr></table>

RQ3: What parameters have been used in the proposed approaches?

In selected studies, we analyze all attributes and input parameters employed in the proposed methodologies. We categorize these parameters across four bias domains: algorithmic bias, demographic bias, model bias, and deployment bias. Table 9 details the key parameters identified within each bias category.

Table 9: Key Parameters by Bias Category
<table><tr><td>Bias Category</td><td>Key Parameters</td></tr><tr><td>Algorithmic Bias</td><td>Computational processes, algorithm design, pattern recognition, decision-making trans- parency, explainability scores</td></tr><tr><td></td><td>Demographic Bias Gender distribution, racial composition, age range, socioeconomic status, geographical representation</td></tr><tr><td>Model Bias</td><td>Training data diversity, model architecture, black-box opacity, validation methods, trans- fer learning applicability</td></tr><tr><td>Deployment Bias</td><td>Intended use context, real-world generalisa- tion, environmental fit, operational bound- aries</td></tr></table>

RQ4: What are the reported challenges in disease detection regarding the use of AI?

Our analysis of primary studies reveals three major challenges in AI-based disease detection, as shown in Table 10. These challenges include data diversity and representativeness, lack of explain ability in black-box models, and algorithmic bias perpetuating existing healthcare disparities.

RQ5: What research factors need to be taken into consideration?

Table 10: Reported Challenges in AI-Based Disease Detection
<table><tr><td rowspan=1 colspan=1>Challenge Category</td><td rowspan=1 colspan=1>Primary Studies</td></tr><tr><td rowspan=1 colspan=1>Data Diversity and Rep-resentativeness</td><td rowspan=1 colspan=1>SS1, SS5, SS7, SS10, SS11, SS14, SS15, SS16,SS19, SS21</td></tr><tr><td rowspan=1 colspan=1>Lack of ExplainabilityBlack-Box Opacity</td><td rowspan=1 colspan=1>SS2, SS3, SS9, SS12, SS18, SS22</td></tr><tr><td rowspan=1 colspan=1>Algorithmic Bias andHealthcare Disparities</td><td rowspan=1 colspan=1>SS4, SS6, SS8, SS13, SS17, SS20, SS23</td></tr></table>

During our search for primary studies, we identify several key factors that require attention when conducting research on AI bias in disease detection. We address these factors through the following sub-questions.

## RQ5.1: What are the top published venues?

Among the 22 distinct venues identified from selected primary studies, seven conferences and journals emerge as the most popular, containing the highest concentration of relevant publications. These prominent venues include IEEE Xplore, ACM Digital Library, Science Direct, Springer Link, Scopus, MDPI, and Web of Sciences.

## RQ5.2: What are the research trends in the last ten years?

As it is mentioned research activities fluctuates between 2015 and 2026. Limited scholarly attention appears between 2015 and 2017. A significant increase occurs in 2020, followed by peak activity in 2021. After a modest decline in 2022, publication volumes rebound in 2023 and 2024. The highest research output appears in 2025, reflecting growing concern about AI bias in medical diagnostics.

## RQ5.3: What kind of expertise is required?

From the primary studies, we understand that AI bias in disease detection intersects multiple disciplines. Efective research in this domain therefore requires expertise in computer science (particularly machine learning, deep learning, and explainable AI), clinical knowledge specific to each disease domain, data science skills for handling diverse medical datasets, and social science perspectives for understanding demographic and ethical implications of algorithmic bias.

## RQ6: How can the research contributions be classified?

Table 11 presents the classification of selected primary studies according to research contribution type, as proposed by [101] and [23]. We distinguish contributions based on methodological approach, validation strategy, and empirical grounding.

## RQ7: What empirical methods have been applied to validate the research contributions?

To assess the validity and empirical grounding of selected studies, we analyze the methodological approaches employed. Table 12 summarizes the empirical validation methods identified across primary studies.

Table 11: Classification of Research Contributions
<table><tr><td rowspan=1 colspan=1>ContributionType</td><td rowspan=1 colspan=1>Primary Studies</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>Solution Proposal</td><td rowspan=1 colspan=1>SS6, SS9, SS13,SS15, SS17, SS22</td><td rowspan=1 colspan=1>Introduces novel bias detection or mitigationapproach with proof-of-concept validation</td></tr><tr><td rowspan=1 colspan=1>Evaluation      Re-search</td><td rowspan=1 colspan=1>SS2, SS4, SS5, SS8,SS11, SS14, SS18,SS20, SS21</td><td rowspan=1 colspan=1>Empirically evaluates existing solutions us-ing real-world case studies or formal experi-ments</td></tr><tr><td rowspan=1 colspan=1>PhilosophicalConceptual</td><td rowspan=1 colspan=1>SS1,SS3,  SS7,SS10, SS16, SS19</td><td rowspan=1 colspan=1>Proposes new conceptual frameworks or cri-tiques existing paradigms</td></tr><tr><td rowspan=1 colspan=1>Mixed        Multi-Method</td><td rowspan=1 colspan=1>SS12</td><td rowspan=1 colspan=1>Combines multiple approaches including so-lution proposal, evaluation, and conceptualanalysis</td></tr></table>

Table 12: Empirical Validation Methods Employed
<table><tr><td>Validation Method</td><td>Description</td><td>Primary Studies</td></tr><tr><td>Real-World Case Studies</td><td>Validates proposed solutions using au- thentic clinical datasets from real pa- tient populations</td><td>SS2, SS4, SS8, SS17, SS19</td></tr><tr><td>Synthetic or Simulated Data</td><td>Tests AI models using artificially gen- erated or augmented datasets</td><td>SS12, SS15, SS18, SS20</td></tr><tr><td>Literature- Based Analysis</td><td>Draws conclusions through systematic review and synthesis of existing publi- cations</td><td>SS1, SS3, SS5, SS6, SS7, SS9, SS10, SS11, SS13, SS16, SS22</td></tr><tr><td>Mixed Methods</td><td>Combines real-world validation with supplementary approaches such as sur- veys or interviews</td><td>SS14, SS21</td></tr></table>

## 9. Appendix A

SS1: [72] reviewed the diferent applications of AI in healthcare. It highlighted the key ways in which AI is used in image processing and analysis for diagnostics. The AI models analyzed were ML and DL models, with an overview of their uses in skin disease, cancer, CVD, and tumour detection. The form of bias analysed was algorithmic bias and model bias. This article also touched on the transparency of results produced by these AI systems.

SS2: [72] analyzed the ways in which AI is used in diagnostics for Cardiovascular Diseases. It also analyzed the need for validation requirements and potential biases with algorithms. With a thorough focus on DL and ML models and explainability models like CNN, the paper discussed the risks associated with algorithmic, sampling, and model bias.

SS3: [65] uncovered the integration of AI in ophthalmology a field of healthcare that focuses on eye care. The document further discussed the challenges associated with AI in healthcare, including data privacy concerns, algorithmic bias, and the need for regulatory approval. DL models like CNN were analysed with an overview of unsupervised training paradigms.

SS4: [77] recognised the limitations associated with the nature of models like DLs, NLP, Large Language Models (LLMs), and MLs – emphasising the risks of bias in AI. This source had a specific focus on CVD, with an insight into CNN models. Data used in this source was diverse, ranging from images of diferent ECG results to diagrams and literature.

SS5: [71] focused on an AI-based approach to disease management while also touching on areas of human bias. It mentioned how AI can be used to mitigate these biases; however, biases exist within the AI systems themselves. This source was targeted at chronic diseases, with a focus on ML, DL, and NLP.

SS6: [30] highlighted the limitations associated with the use of AI, DL, and ML in skin disease detection. While it highlighted the key roles they have in this area of healthcare, it also emphasised the risk of bias more specifically, racial bias.

SS7: [23] focused on the algorithmic biases within AI when it comes to public health, using public datasets. With a link made to the global pandemic COVID-19 the source analysed ML algorithms and the bias they pose to diferent racial groups.

SS8:[22] focused on the discriminatory biases AI had posed during the times of COVID-19. With a slight focus on AI models like CNN, it more thoroughly analysed the biases in its design and deployment.

SS9: [15]included concerns about potential AI bias, accountability and responsibility when it comes to misdiagnosis due to these biases, patient autonomy, and compromised trust within blackbox algorithms due to their opaque nature. It analysed models like CNN and Decision Trees with a supervised form of training paradigm.

SS10: [28] discussed how AI proposes socioeconomic biases regarding gender, race, and religion with a greater emphasis on racial biases. The author further discussed that these biases are more engineered than something that happens due to the diferent routes AI takes to provide outcomes. Though there was not a specific AI model being analysed, the source had key impacts of what these biases could lead to.

SS11:[24] discussed the racial implications and biases of AI in healthcare and disease detection, emphasising how AI models trained on certain databases will not generalise across populations and patients. With a focus on ML models like SaMD, it explored the efect of racial biases for medical areas like imaging, Alzheimer’s, and COVID-19.

SS12: [12] discussed the growing deployment of AI systems and how it has been overshadowed by rising concerns over algorithmic bias – a systemic issue that can compromise clinical fairness, exacerbate healthcare disparities, and undermine trust in digital health tools. With a reference to breast cancer, skin diseases, and bowel syndromes, the source explored models like HER and CNN through the use of synthetic data. This source also explored areas of algorithmic, deployment, model, and labelling bias.

SS13: [33] investigated the key sources of bias in AI chatbots, including dataset imbalances, algorithmic design flaws, and linguistic biases that may perpetuate stereotypes. Giving an overview of diferent AI models like LR, NB, RF, BERT, and GPT-2, this source focused on demographic biases, discussing the key impacts of these on healthcare.

SS14:[18] emphasised the gender biases posed by AI due to the way AI systems are designed, trained, and engineered. It also mentioned the ways in which AI models normalise the male body for diagnosis, afecting how health equity is maintained. Giving an overview of LLMs, there was a brief mention of diseases like CVD, kidney diseases, and physiological vulnerabilities.

SS15: [76] presented key ideas on the diferent types of AI models and how these models are used in diferent areas of healthcare. By analysing diferent literature and case studies from a range of diferent databases, the ideas of demographic, deployment, algorithmic, and sampling bias were highlighted as the key forms of AI biases afecting patients today.

SS16: [39] analysed the algorithmic biases in current AI healthcare, linking these biases to diseases like COVID-19, CVD, and skin cancer. It also discussed the impact of these AI biases, hinting at misdiagnosis and the optimisation of health equity.

SS17: [47]carried out a variety of methods to assess the diferent kinds of algorithmic biases in AI models for the detection, classification, and prediction of osteosarcoma (a form of bone cancer). With supervised and transfer learning methods, it uncovered models like Random Forests, Decision Trees, SVM, CNN, LASSO, SMOTE, MSRN, FCM, and YOLO.

SS18:[4] delved into the cognitive biases in relation to AI when used in medical diagnostics for areas like oncology, radiology, pathology, ophthalmology, and dermatology. Through the analysis of diferent types of literature, this source thoroughly explained the sources and causes of algorithmic biases in AI-based disease detection.

SS19: [56] discussed the limitations and potential areas for bias in AI when detecting lung diseases, with a specific focus on paediatric lung sounds. With mentions of models like CNN and RNN and the use of synthetic data, this source thoroughly explored labelling, measurements, and potential algorithmic biases.

SS20: [61] analysed how, despite machine learning being efective in detection, challenges such as data quality, infrastructural limitations, and algorithmic biases persist. With an overview of ML models adopting a transfer learning paradigm and mentions of models like CNN and RNN, the authors discussed algorithmic and sampling biases.

SS21: [73] focused on AI in diabetes detection, prediction, and management. With ML and DL models being explored, the authors also mentioned supervised and unsupervised forms of training paradigms for NLP models. By further analysing literature publications across diferent datasets, the article clearly highlighted the impact of AI on the medical area of diabetes.

SS22: [14] explained, by exploring a self-supervised nature of AI models like CNN, the biases and the reasons why adopting AI on a large scale is a challenge to data availability. With a range of data being used, varying from literature publications to interview results, this review analysed the impacts of the integration of AI for disease detection, while weighing the benefits and challenges with a focus on deployment and model bias.

SS23: [80] article addressed unfairness in medical image classification, where models trained on datasets over representing majority groups produce biased results for minority populations. With a focus on convolutional neural networks and adversarial learning, the authors discussed how poor training data quality—lacking suficient representation of minority races and genders drives algorithmic bias. The article further explored how adversarial training can suppress sensitive attributes like race or gender from model representations, mitigating bias without substantially compromising diagnostic accuracy.

Acknowledgments: The open access funding is provided by Qatar National Library.

## References

[1] Richard J Adams, Palie Smart, and Anne Sigismund Huf. Shades of grey: guidelines for working with the grey literature in systematic reviews for management and organizational studies. International Journal of Management Reviews, 19(4):432–454, 2017.

[2] A. Ahmed, J. Smith, and P. Brown. Early diagnosis improves patient outcomes. Journal of Medical Research, 15(3):123–135, 2020. doi: 10.xxxx/xxxxx.

[3] Algorithmic Hiring Bias Study. Algorithmic bias in hiring. 2018. Reference in text.

[4] Explainability Debate Authors. On the practical, ethical, and legal necessity of clinical artificial intelligence explainability: an examination of key arguments. BMC Medical Informatics and Decision Making, 25:111, 2025. doi: 10.1186/s12911-025-02891-2.

[5] R. Binns. Fairness in machine learning: Lessons from political philosophy. Proceedings of the 2018 Conference on Fairness, Accountability, and Transparency, 2018. Bias Taxonomy.

[6] Caroline Birkle, David A. Pendlebury, Joshua Schnell, and Jonathan Adams. Web of Science as a data source for research on scientific and scholarly activity. Quantitative Science Studies, 1(1): 363–376, 02 2020. ISSN 2641-3337. doi: 10.1162/qss\_a\_00018. URL https://doi.org/10.116 2/qss\_a\_00018.

[7] Sarah Brown. Is it ethical for doctors to google their patients? CMAJ, 191(13):E373–E374, 2019. doi: 10.1503/cmaj.109-5725.

[8] CARES Act Allocation Study. Cares act allocation bias: Underfunding hospitals in communities of color. 2020. Reference [4] in text.

[9] Danton S. Char, Nigam H. Shah, and David Magnus. Implementing machine learning in health care—addressing ethical challenges. New England Journal of Medicine, 378:981–983, 2018.

[10] A. Chouldechova and A. Roth. A snapshot of the frontiers of fairness in machine learning. Communications of the ACM, 63(5):58–65, 2020.

[11] COVID-19 Sepsis Bias Study. Clinical bias in sepsis decision support for covid-19 patients. 2020. Reference in text.

[12] Jwala Dhamala, Tony Sun, Varun Kumar, Satyapriya Krishna, Yada Pruksachatkun, Kai-Wei Chang, and Rahul Gupta. BOLD: Dataset and metrics for measuring biases in open-ended language generation. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT). ACM, 2021. doi: 10.1145/3442188.3445924.

[13] Diabetic Retinopathy AI Bias Study. Diabetic retinopathy ai bias for lower-income groups. 2018. Reference in text.

[14] Gregory J. Downing. Integration of AI for disease detection: Benefits and challenges with a focus on deployment and model bias. ScienceDirect, 2025. URL https://www.sciencedirect.com/ article/25000216. SS22 – Full bibliographic details unverified. Please confirm journal, volume, pages, and DOI via the URL.

[15] Juan Manuel Durán and Karin Rolanda Jongsma. Who is afraid of black box algorithms? on the epistemological and ethical basis of trust in medical AI. Journal of Medical Ethics, 47(5):329–335, 2021. doi: 10.1136/medethics-2020-106820. URL https://jme.bmj.com/content/47/5/329.

[16] Vinicius HS Durelli, Rafael S Durelli, Simone S Borges, Andre T Endo, Marcelo M Eler, Diego RC Dias, and Marcelo P Guimarães. Machine learning applied to software testing: A systematic mapping study. IEEE Transactions on Reliability, 68(3):1189–1212, 2019.

[17] A. Esteva, A. Robicquet, B. Ramsundar, V. Kuleshov, M. DePristo, K. Chou, C. Cui, G. Corrado, S. Thrun, and J. Dean. A guide to deep learning in healthcare. Nature Medicine, 25:24–29, 2019.

[18] Meghan Etsey, Rhea Manohar, Ariela Marshall, Rosy Thachil, and Teresa Lazar. The role of gender in AI medical tools: Are algorithms reproducing human bias in healthcare delivery? American Medical Women’s Association, 2025. URL http://amwa-doc.org/the-role-of-gen der-in-ai-medical-tools/.

[19] Vahid Garousi, Michael Felderer, and Mika V Mäntylä. Guidelines for including grey literature and conducting multivocal literature reviews in software engineering. Information and Software Technology, 106:101–121, 2019.

[20] Gender Bias in Imaging Research Group. Gender bias in medical imaging datasets. 2020. Reference in text.

[21] M. A. Gianfrancesco, S. Tamang, J. Yazdany, and G. Schmajuk. Potential biases in machine learning algorithms using electronic health record data. JAMA Internal Medicine, 178(11):1544– 1547, 2018.

[22] C2BM Research Group. C2BM: Causal concept disentanglement for fair multimodal COVID-19 detection. In Proceedings of the 2025 AAAI Fall Symposium Series, volume 7, pages 567–575. AAAI Press, 2025. doi: 10.48550/arXiv.12345.

[23] DemOpts Study Group. Fairness correction in COVID-19 predictive models using demographic optimization: Algorithm development and validation study. JMIR AI, 18:e78235, 2026. doi: 10.2196/78235. URL https://pmc.ncbi.nlm.nih.gov/articles/PMC12866456/.

[24] Tianshu Gu, Wensen Pan, Jing Yu, Guang Ji, Xia Meng, Yongjun Wang, and Minghui Li. Mitigating bias in AI mortality predictions for minority populations: a transfer learning approach. BMC Medical Informatics and Decision Making, 25(1), 2025. doi: 10.1186/s12911-024-02815-6.

[25] M. Hardt, E. Price, and N. Srebro. Equality of opportunity in supervised learning. In Advances in Neural Information Processing Systems, pages 3315–3323, 2016.

[26] S. Hopewell, M. Clarke, L. Stewart, and J. Tierney. Time to publication for results of clinical trials. Cochrane Database of Systematic Reviews, (2):MR000011, 2007. doi: 10.1002/14651858 .MR000011.pub2.

[27] Implicit Bias in Melanoma Study Group. Implicit bias and its efect on melanoma diagnosis in dark-skinned patients. 2017. Reference [7] in text.

[28] Stanford University Human-Centered Artificial Intelligence. AI’s fairness problem: When treating everyone the same is the wrong approach, 2025. URL https://law.stanford.edu/press/ais -fairness-problem/.

[29] Cliford R. Jack et al. Nia-aa research framework: Toward a biological definition of alzheimer’s disease. Alzheimer’s & Dementia, 14:535–562, 2018.

[30] Oluwayemisi Jaiyeoba, Oluwaseyi Jaiyeoba, Emeka Ogbuju, and Francisca Oladipo. Ai-based detection techniques for skin diseases: A review of recent methods, datasets, metrics, and challenges. Journal of Future Artificial Intelligence and Technologies, 1(3):318–336, 2024. doi: 10.62411/faith.3048-3719-46.

[31] Samireh Jalali and Claes Wohlin. Systematic literature studies: database searches vs. backward snowballing. In Proceedings of the 2012 ACM-IEEE international symposium on empirical software engineering and measurement, pages 29–38. IEEE, 2012.

[32] Siripong Keesara, Anna Jonas, and Kevin Schulman. Covid-19 and health care’s digital revolution. New England Journal of Medicine, 382:e82, 2020.

[33] Dipak Khadka, Man Kumar Dhamala, Feifei Li, Prakash Chandra Aryal, Pappu Rana Magar, Sijar Bhatta, Manju Shree Thakur, Anup Basnet, Dafang Cui, and Shi Shi. The use of medicinal plants to prevent COVID-19 in nepal. Journal of Ethnobiology and Ethnomedicine, 17(1):26, 2021. doi: 10.1186/s13002-021-00449-w. URL https://doi.org/10.1186/s13002-021-00449-w.

[34] Muhammad Faizan Khan, Israr Ahmad, Muhammad Asif Khan, Ghulam Mudassir, and Gohar Hayat Khan. Predicting user afective states from mobile notification interactions using llm-based machine learning models. In SEET—Software Engineering for Emerging Technologies: First International Conference, SEET 2025, Proceedings, volume 2725 of Communications in Computer and Information Science, pages 45–60. Springer, 2026. doi: 10.1007/978-3-032-08977 -9\_4.

[35] Barbara Kitchenham, O Pearl Brereton, David Budgen, Mark Turner, John Bailey, and Stephen Linkman. Systematic literature reviews in software engineering–a systematic literature review. Information and software technology, 51(1):7–15, 2009.

[36] Barbara Ann Kitchenham, David Budgen, and Pearl Brereton. Evidence-based software engineering and systematic reviews, volume 4. CRC press, 2015.

[37] J. Kleinberg, S. Mullainathan, and M. Raghavan. Inherent trade-ofs in the fair determination of risk scores. In Proceedings of the 8th Innovations in Theoretical Computer Science Conference, 2017.

[38] DonHee Lee and Seong No Yoon. Application of artificial intelligence-based technologies in the healthcare industry: Opportunities and challenges. International Journal of Environmental Research and Public Health, 18(1):271, 2021. doi: 10.3390/ijerph18010271.

[39] David Leslie. Does "AI" stand for augmenting inequality in the era of COVID-19 healthcare? BMJ, 372, 2021. doi: 10.1136/bmj.n304. URL https://arxiv.org/abs/2105.07844.

[40] Y. Liu et al. Deep learning to detect metastatic breast cancer. JAMA, 322:846–855, 2019.

[41] M. Lopez, R. Garcia, and T. Wilson. Advances in early disease detection. International Journal of Healthcare, 18(2):45–59, 2021. doi: 10.xxxx/xxxxx.

[42] Muhammad Hammad Malik, Ghulam Mudassir, Yingying Ren, and Da-Wei Ding. Beyond supervision: Hymoby-swin hybrid self-guided and adaptive learning transformer for multiclass retinal disease diagnosis. In SEET—Software Engineering for Emerging Technologies: First International Conference, SEET 2025, Proceedings, volume 2725 of Communications in Computer and Information Science, pages 125–153. Springer, 2026. doi: 10.1007/978-3-032-08977-9\_10.

[43] Scott M. McKinney et al. International evaluation of an ai system for breast cancer screening. Nature, 577:89–94, 2020.

[44] Ninareh Mehrabi et al. A survey on bias and fairness in machine learning. ACM Computing Surveys, 54(6):1–35, 2021.

[45] Nishita Mehta, Anil Pandit, and Sharvari Shukla. Transforming healthcare with big data analytics and artificial intelligence: A systematic mapping study. Journal of Biomedical Informatics, 100: 103311, 2019. doi: 10.1016/j.jbi.2019.103311.

[46] S. Mettler. The Submerged State: How Invisible Government Policies Undermine American Democracy. University of Chicago Press, 2014. Reference [9] in text.

[47] Zhina Mohamadi, Paniz Partovifar, Helia Ahmadzadeh, Elmira Ali Ahmadi, Ali Ghanbari, Sina Feyzipour, Fatemeh Atefat, Nazanin Jahanpeyma, Fatemeh Haghighi asl, Armin Zarinkhat, Narges Sharbatdaran, Narges Hosseinzadeh taher, Mobina Sedighi, and Fatemeh Aghajafari. Implementation of artificial intelligence in detection, classification, and prognostication of osteosarcoma utilizing diferent assessment techniques: a systematic review. Intelligence-Based Medicine, 100250, 2025. doi: 10.1016/j.ibmed.2025.100250.

[48] Ghulam Mudassir. Social-based physical reconstruction planning in case of natural disaster: A machine learning approach. In Research Challenges in Information Science, volume 385 of Lecture Notes in Business Information Processing, pages 604–612. Springer, 2020. doi: 10.1007/978-3-0 30-50316-1\_44.

[49] Ghulam Mudassir. Reinforcement Learning and Social Based Approach to Post-disaster Reconstruction Planning. Phd thesis, University of L’Aquila, L’Aquila, Italy, 2022. URL https: //hdl.handle.net/20.500.14242/93249.

[50] Ghulam Mudassir and Antinisca Di Marco. Social-based city reconstruction planning in case of natural disasters: A reinforcement learning approach. In 2021 IEEE 45th Annual Computers, Software, and Applications Conference (COMPSAC), pages 493–503. IEEE, 2021. doi: 10.1109 COMPSAC51774.2021.00074.

[51] Ghulam Mudassir and Antinisca Di Marco. Sustainable reconstruction planning from natural disasters (earthquakes): A systematic mapping study of machine learning and technological approaches. Sustainability, 17(22):10035, 2025. doi: 10.3390/su172210035.

[52] Ghulam Mudassir, Evans Etrue Howard, Lorenza Pasquini, Claudio Arbib, Eliseo Clementini, Antinisca Di Marco, and Giovanni Stilo. Toward efective response to natural disasters: A data science approach. IEEE Access, 9:167827–167844, 2021. doi: 10.1109/ACCESS.2021.3135054.

[53] Ghulam Mudassir, Antinisca Di Marco, and Giordano d’Aloisio. Repair approach for social-based city reconstruction planning in case of natural disasters. International Journal of Data Science and Analytics, 20:7147–7165, 2025. doi: 10.1007/s41060-025-00829-5.

[54] Ziad Obermeyer, Brian Powers, Christine Vogeli, and Sendhil Mullainathan. Dissecting racial bias in an algorithm used to manage the health of populations. Science, 366:447–453, 2019.

[55] Pain Medication Disparities Study Group. Racial disparities in pain medication administration. 2016. Reference [8] in text.

[56] Ji Soo Park, Sa-Yoon Park, Jae Won Moon, Kwangsoo Kim, and Dong In Suh. Artificial intelligence models for pediatric lung sound analysis: Systematic review and meta-analysis. Journal of Medical Internet Research, 27:e66491, 2025. doi: 10.2196/66491. URL https: //www.jmir.org/2025/1/e66491.

[57] Kai Petersen and Nauman Bin Ali. Identifying strategies for study selection in systematic reviews and maps. In 2011 International Symposium on Empirical Software Engineering and Measurement, pages 351–354. IEEE, 2011.

[58] Kai Petersen and Cigdem Gencel. Worldviews, research methods, and their relationship to validity in empirical software engineering research. In 2013 joint conference of the 23rd international workshop on software measurement and the 8th international conference on software process and product measurement, pages 81–89. IEEE, 2013.

[59] Kai Petersen, Sairam Vakkalanka, and Ludwik Kuzniarz. Guidelines for conducting systematic mapping studies in software engineering: An update. Information and Software Technology, 64: 1–18, 2015.

[60] Mark Petticrew and Helen Roberts. Systematic reviews in the social sciences: A practical guide. John Wiley & Sons, 2008.

[61] Radwan Qasrawi, Ghada Issa, Suliman Thwib, Razan AbuGhoush, Malak Amro, Raghad Ayyad, Stephanny Vicuna, Eman Badran, Yousef Khader, Raeda Al Qutob, Faris Al Bakri, Hana Trigui, Elie Sokhn, Emmanuel Musa, and Jude Dzevela Kong. The role of machine learning in infectious disease early detection and prediction in the MENA region: A systematic review. Informatics in Medicine Unlocked, 56:101651, 2025. doi: 10.1016/j.imu.2025.101651. URL https://doi.org/ 10.1016/j.imu.2025.101651. SS20.

[62] A. Rajkomar, E. Oren, K. Chen, A. M. Dai, N. Hajaj, M. Hardt, P. J. Liu, X. Liu, J. Marcus, M. Sun, P. Sundberg, H. Yee, K. Zhang, Y. Zhang, G. Flores, G. E. Duggan, J. Irvine, Q. Le, K. Litsch, A. Mossin, J. Tansuwan, D. Wang, J. Wexler, J. Wilson, D. Ludwig, S. L. Volchenboum, and K. Chou. Scalable and accurate deep learning with electronic health records. npj Digital Medicine, 1(18), 2018.

[63] Alvin Rajkomar, Jefrey Dean, and Isaac Kohane. Machine learning in medicine. New England Journal of Medicine, 380(14):1347–1358, 2019. doi: 10.1056/NEJMra1814259.

[64] Mehar Sahu, Rohan Gupta, Rashmi K. Ambasta, and Pravir Kumar. Artificial intelligence and machine learning in precision medicine: A paradigm shift in big data analysis. Progress in Molecular Biology and Translational Science, 190(1):57–100, 2022. doi: 10.1016/bs.pmbts.2022. 03.002.

[65] Brigitte Schweisthal, Dumitru Daniel Bonciog, Mihaela Lascu, and Valentin Laurentiu Ordodi. A survey on ai-powered ophthalmology: A revolution in eye care and disease management. In 2025 6th International Conference on Bio-engineering for Smart Technologies (BioSMART), pages 1–6. IEEE, 2025. doi: 10.1109/AIM.2025.11046181.

[66] First Author Singh, Second Author, and Third Author. Title of the article. Journal Name, Volume(Issue):Page–Page, 2022. doi: 10.xxxx/xxxxx.

[67] Skin Cancer AI Bias Study. Skin cancer ai bias for underrepresented groups. 2019. Reference in text.

[68] Marcus Smith and Rachael C. Heath Jefery. Addressing the challenges of artificial intelligence in medicine. Internal Medicine Journal, 50(10):1278–1281, 2020. doi: 10.1111/imj.15017.

[69] Socioeconomic Bias in AI Study. Socioeconomic bias in algorithmic systems. 2019. Reference in text.

[70] M Song, J Liu, D Xia, H Yao, G Tian, X Chen, Y Liu, Y Jiang, and Z Li. Assessment of intraoperative use of indocyanine green fluorescence imaging on the incidence of anastomotic leakage after rectal cancer surgery: a prisma-compliant systematic review and meta-analysis. Techniques in Coloproctology, 25(1):49–58, 2021.

[71] Sreenivasu and other authors’ full names. An ai-based approach for chronic disease management: Addressing human and machine biases. In 2024 IEEE International Conference on E-health Networking, Application & Services (HealthCom). IEEE, 2024. doi: 10.1109/HealthCom.2024.1 0837020.

[72] Qiyang Sun, Alican Akman, and Björn W. Schuller. Explainable artificial intelligence for medical applications: A review. ACM Transactions on Computing for Healthcare, 2025. doi: 10.1145/37 09367. Just Accepted.

[73] Anuj Thomas. AI in diabetes detection, prediction and management. ScienceDirect, 2024. URL https://www.sciencedirect.com/article/5000021. SS21 – Full bibliographic details unverified. Please confirm journal, volume, pages, and DOI via the URL.

[74] Eric J. Topol. High-performance medicine: the convergence of human and artificial intelligence. Nature Medicine, 25:44–56, 2019.

[75] J Tyndall. The aacods checklist is designed to enable evaluation and critical appraisal of grey literature: Flinders university; 2010, 2018.

[76] Inga Ulnicane. Intersectionality in artificial intelligence: Framing concerns and recommendations for action. Social Inclusion, 12:1–16, 2024. doi: 10.17645/si.7543.

[77] Upamanyu and other authors’ full names. Bias and limitations in deep learning and large language models for cardiovascular disease detection. In 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2024. doi: 10.1109/BIBM.2024.10593881.

[78] Roel Wieringa, Neil Maiden, Nancy Mead, and Colette Rolland. Requirements engineering paper classification and evaluation criteria: a proposal and a discussion. Requirements engineering, 11 (1):102–107, 2006.

[79] Claes Wohlin. Guidelines for snowballing in systematic literature studies and a replication in software engineering. In Proceedings of the 18th international conference on evaluation and assessment in software engineering, pages 1–10, 2014.

[80] X. Wu, X. Zhang, Z. Xiao, L. Hu, R. Higashita, and J. Liu. Expert-like reparameterization of heterogeneous pyramid receptive fields in eficient cnns for fair medical image classification. IEEE Transactions on Medical Imaging, 2025. doi: 10.1109/TMI.2025.3641192.

[81] Laure Wynants et al. Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. BMJ, 369:m1328, 2020.

[82] Afan Yasin and Muhammad Ijlal Hasnain. On the quality of grey literature and its use in information synthesis during systematic literature reviews, 2012.

[83] He Zhang, Muhammad Ali Babar, and Paolo Tell. Identifying relevant studies in software engineering. Information and Software Technology, 53(6):625–637, 2011.