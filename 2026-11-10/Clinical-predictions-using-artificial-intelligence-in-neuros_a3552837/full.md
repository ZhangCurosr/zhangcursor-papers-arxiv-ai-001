REVIEW ARTICLE

∂ OPEN ACCESS

Check for updates

# Clinical predictions using artificial intelligence in neurosurgery and its subspecialties: a systematic review and meta-analysis

Prajwal Ghimire<sup>a,b,c</sup> , Aswin Chari<sup>d,e</sup>, Jonathan Shapey<sup>a,b</sup>, and Keyoumars Ashkan<sup>a,b,f</sup>

<sup>a</sup>Department of Neurosurgery, King’s College Hospital NHS Foundation Trust, London, UK; <sup>b</sup>School of Biomedical Engineering & Imaging Sciences, King’s College London, London, UK; <sup>c</sup>King’s Institute of Artificial Intelligence, King’s College London, London, UK; <sup>d</sup>Department of Neurosurgery, Great Ormond Street Hospital for Children NHS Foundation Trust, London, UK; <sup>e</sup>Institute of Child Health, University College London, London, UK; <sup>f</sup>Institute of Psychiatry, Psychology & Neuroscience, King’s College London, London, UK

## ABSTRACT

Introduction: Artificial intelligence (AI) models for clinical prediction are proliferating across neurosurgery, but their aggregate discriminative performance, distribution along the care pathway and methodological quality are unclear. We systematically reviewed and meta-analysed these studies in neurosurgery and its subspecialities along with their risk of bias appraisal.

Methods: We performed a PRISMA search on PubMed and Ovid for peer-reviewed original studies (2015–2026) using AI to predict a clinical outcome in neurosurgical populations. Metrics (primary- AUC; secondary- accuracy, sensitivity, specificity, c-index) were pooled using random effects modelling, overall, by subspeciality and care-pathway stage. Each study was classified as diagnostic, prognostic or forecasting and was assigned an evidence level. Clinical prediction task sub-stratification analysis was performed. Risk-of-bias was assessed using PROBAST. PROSPERO registered: CRD420261436614

ARTICLE HISTORY Received 24 August 2026 Revised 8 October 2026 Accepted 1 November 2026

KEYWORDS   
Neurosurgery; artificial intelligence; clinical   
prediction models; bias; meta-analysis

Results: 1063 studies were included. Across 451 (42.2%) studies reporting AUC, pooled discrimination was 0.86 (95% CI: 0.849–0.862), ranging from 0.84 (spine) to 0.87 (neuro-oncology), with extreme heterogeneity (I<sup>2</sup> 98.6%, s<sup>2</sup>¼0.27; 95% prediction interval: 0.68–0.94). Performance varied by clinical pathway stage, highest for diagnosis, grading and molecular tasks and lowest for post-operative tasks; pre-diagnosis, pre-operative and intra-operative applications were sparse, and treatment models were confined to neuro-oncology. Only 51 studies (4.8%) were using prospective data with external validation, whereas 619 (58%) rested on internal validation alone or no validation. 89% studies were at high risk-of-bias, driven by the analysis domain. Pooled discrimination was unchanged (AUC: 0.86%) when restricted to externally validated or non-high-risk studies.

Conclusion: AI models report high performance across all the neurosurgical subspecialities, concordant across AUC, accuracy, sensitivity and specificity and robust to risk-of-bias stratification. Performance is highest for imaging-anchored diagnostic tasks and lowest at decision-facing post-operative stage. However, the literature is dominated by optimism-prone, internally-validated and poorly calibrated models with extreme heterogeneity. External validation, calibration analysis and reporting guidelines adherence remain prerequisites for clinical adoption.

## 1. Introduction

Clinical prediction is key in neurosurgical decisionmaking: estimating the probability of a molecular tumour subtype, of haemorrhage or rupture, of seizure freedom after epilepsy surgery, or of complications and functional recovery after spinal surgery. Over the past decade, artificial intelligence (AI)

methods have been applied to these tasks at rapidly increasing scale, ranging from logistic regression, tree ensembles, support-vector machines, radiomics pipelines and deep neural networks.<sup>1–4</sup>

Individual studies frequently report excellent discrimination, but enthusiasm has outpaced methodological scrutiny. The wider clinical machine learning (ML) literature is characterised by small samples, internal-only validation, absent calibration and incomplete reporting- features that inflate apparent performance and limit generalisability of these prediction models.<sup>5,6</sup> Whether neurosurgical AI prediction models share these weakness and how performance and bias vary across subspecialities, has not been comprehensively quantified.

We therefore performed a systematic review and meta-analysis of AI prediction studies across the neurosurgical subspecialities. The aims were (i) to map the machine learning methods and clinical prediction tasks in use, (ii) to pool discriminative performance, (iii) to map the application of these tools in the clinical care pathway- from pre-diagnosis to adjuvant/neo-adjuvant treatment; (iv) to attempt classification of models, define clinical translation stage and explore level of evidence (v) to appraise risk of bias and applicability.

## 2. Methods

The review followed the Preferred Reporting Items for Systematic Reviews and Meta-analyses<sup>7,8</sup> and incorporated the developing PRISMA-AI<sup>9</sup> reporting considerations for studies of machine learning models. A PROSPERO registration was performed (CRD420261436614).

## 2.1. Eligibility criteria

We included original peer-reviewed studies that applied AI or ML as part of the methodology to predict a clinical outcome in neurosurgical populations across its subspecialities: neuro-oncology, neuro-vascular, spine, functional/epilepsy, skull-base, paediatrics, general and trauma. Retrospective, prospective and trial designs were eligible.

## 2.2. Exclusion criteria

We excluded reviews, editorials, letters, conference proceedings, non-human studies and studies without a clinically relevant predictive outcome. Title-level review articles were removed during screening.

## 2.3. Search and selection

PubMed and a multi-field OVID (Embase, Medline) databases were searched for the period of 2015–2026 using medical subject headings (MeSH) terms. A low precision high sensitivity search was conducted using subject headings and exploding terms. Records were deduplicated and screened against a neurosurgical relevance and study-type to remove reviews and nonneurosurgical records. All results were reviewed with dual-reviewer adjudication.

## 2.4. Statistical analysis

Study-level discrimination (area-under-the-curve, AUC) was logit-transformed and pooled using random-effects modelling,<sup>10</sup> overall and within each subspeciality. The standard error of each logit-AUC was derived from the reported 95% confidence intervals (CI); where unavailable, it was approximated from sample size using the Hanley-McNeil method.<sup>11</sup> Statistical heterogeneity<sup>10</sup> was summarised with metaanalysis metrics I<sup>2</sup> and s<sup>2</sup>, and 95% prediction intervals were calculated; estimates were back-transformed to the AUC scale. Pre-specified sensitivity analyses restricted pooling to externally validated studies and to studies not rated high-risk of bias.

Secondary performance metrics (accuracy, sensitivity, specificity, c-index) were pooled with the same random-effects model on the logit scale. Standard errors were derived from the reported sample size using the binomial approximation or imputed as the median where sample size was not available. Secondary metrics were additionally pooled by subspeciality and by clinical pathway stage, reporting only strata containing at least five studies.

To explore heterogeneity, subgroup analyses based on subspeciality, clinical pathway stage, primary model family, validation status, level of evidence, overall risk of bias, predicted outcome, study design and source of standard error was performed with pooling within each level and testing level means differences using Cochran’s Q between groups. Random-effects metaregression was fitted with methods-of-moments estimation of the s<sup>2</sup> metric, univariably and in a multivariate analysis accounting subspeciality, validation status, model family, sample size and publication year.

The distribution of performance metric was summarised by threshold band (<0.70, 0.70–0.80, 0.80– 0.90, �0.90) overall, by subspeciality and by clinical pathway stage. Small-study effects were examined with funnel plots and with Egger’s regression test for funnel asymmetry.

Proportion of studies reporting calibration were tabulated.

## 2.5. Risk-of-bias assessment

Risk of bias and applicability were assessed with PROBAST<sup>12</sup> across its four domains (participants, predictors, outcome, analysis) and 21 signalling questions, incorporating PROBAST-AI<sup>13</sup> considerations for ML models. Reporting was assessed against TRIPOD<sup>1</sup> and TRIPOD-AI extension.<sup>15</sup>

## 2.6. Data extraction

Extraction was structured against the CHARMS checklist<sup>16</sup> for systematic reviews of prediction modelling studies covering its domain of source of data, participants, outcome to be predicted, candidate predictors, sample size, missing data, model development, model performance, model evaluation, results and interpretation. Following CHARMS strategy, each study was classified as a diagnostic (estimating the probability that a condition is currently present including grade, subtype, molecular biomarkers from imaging), prognostic (estimating probability of a future event) or a forecasting model (dynamic or time-updated prediction over a temporal set of events).

## 2.7. Stage of clinical translation

Each study was placed on a six-level translation ladder based on CHARMS, TRIPOD-AI<sup>15</sup> and $\mathrm { D E C I D E - A I } ^ { \mathrm { 1 7 } }$ guidelines: T0 development only; T1 externally or temporally validated; T2 a usable artefact released (web-based, application or published nomogram); T3 prospective clinical evaluation of the model; T4 an impact or implementation study; T5 regulatory clearance or documented routine deployment.

## 2.8. Level of evidence

No established evidence hierarchy grades clinical prediction model studies by validation status. We therefore descriptively summarised the studies, informed by the Oxford Centre for Evidence-Based Medicine (OECBM)<sup>18</sup> design with combination of study design and validation status: prospective cohort or trial data with external or temporal validation; prospective or trial design without external validation or retrospective or unspecified design with external validation; internal validation only (resampling, cross-validation or split-sample); no validation reported (apparent performance only). Studies were classified into by TRIPOD<sup>14,15</sup> model-evidence type (Type 1a, development only; Type 1b, development with resampling; Type 2b, non-random split-sample; Type 3, development with validation on separate data; Type 4, validation only of an existing model). Validation status was taken from the PROBAST<sup>12</sup> signalling answers derived from each article.

## 2.9. Clinical pathway-stage classification

Each study was assigned to a single stage according to its title, model and outcome in order to describe the stage in clinical pathway each model operates in: prediagnosis/screening/risk, diagnosis/grading/molecular, pre-operative, operative/intra-operative, post-operative, baseline prognostication, adjuvant/neo-adjuvant treatment and unclassified/multi-stage residual.

## 2.10. Clinical prediction task sub-stratification

Studies were identified within specific clinical predictions tasks where the neurosurgical population, the neurosurgical predictors and the predicted outcome were comparable. Discrimination was pooled with random effects modelling on the logit-AUC scale where at least 5 studies reported an AUC for each subspeciality. Other studies (<5) were not pooled, but AUC, median and range were reported.

## 3. Results

1063 studies met eligibility criteria, spanning 2015– 2026 (Figure 1). The studies were grouped as per their subspeciality: neuro-oncology (n ¼ 470, 44%); spine (n ¼ 184); neuro-vascular (n ¼ 180); functional/epilepsy (n ¼ 83); trauma (n ¼ 82), paediatrics (n ¼ 26), skull-base (n ¼ 24) and general neurosurgery (n ¼ 14) (Table 1). Reported designs were predominantly retrospective. Specific algorithms in the model architecture were identified: logistic regression (n ¼ 421, 40%), deep learning/neural networks (n ¼ 289, 27%), LASSO-regularised regression(n ¼ 268, 25%), random forest (n ¼ 256, 24%); support-vector machines (n ¼ 244, 23%), Cox regression (n ¼ 228, 21%) and gradient boosting (n ¼ 174, 16%); transformer (n ¼ 18, 2%), foundation-model (n ¼ 8, 0.8%) approaches remained rare. Studies commonly compared several algorithms (n ¼ 521, 49% � 2 models) (Figure 2).

## 3.1. Pooled discrimination

Across 451 studies reporting AUC, pooled discrimination was high and similar across all subspecialities, spanning 0.84–0.92: 0.87 (0.86–0.88, n ¼ 172) in neuro-oncology; 0.84 (0.82–0.86, n ¼ 83) in neuro-vascular; 0.84 (0.83–0.85, n ¼ 94) in spine; 0.87 (0.83– 0.91, n ¼ 11) in skull-base; 0.84 (0.73–0.91, n ¼ 27) in functional/epilepsy; 0.85 (0.79–0.90, n ¼ 10) in paediatrics; 0.86 (0.83–0.88, n ¼ 46) in trauma; 0.92 (0.88–0.94, n ¼ 8) in general neurosurgery; overall pooled AUC 0.86 (0.85–0.86) (Table 2 and Figure 3). Heterogeneity was extreme in every analysis (I<sup>2</sup> 76.1-99.1% by subspeciality; 98.6% overall) and the corresponding between-study variance was $\tau ^ { 2 } { = } 0 . 2 7$ implying that the overall 95% prediction interval was wide (0.68–0.94) and that the performance in a new setting is less certain than the CI and $\mathrm { I } ^ { 2 }$ implies. Perstudy forest plots by subspeciality and by clinical pathway stage are plotted in Suppl. Figures S1 and S2. 7.8% of these studies (n ¼ 35) had AUC <0.70 and 39.5% studies (n ¼ 178) reported AUC range of 0.80– 0.90 with neuro-oncology (n ¼ 67), neurovascular (n ¼ 35), spine (n ¼ 42) and skull base (n ¼ 4) subspeciality with 36.4% to 44.7% studies in the AUC range of 0.80–0.90. Distribution of the AUCs by threshold band and subspeciality are demonstrated in Suppl. Table S1.

![](images/d3b3e0300baf2c62b99a45f90253955308f7dad1dc4d4e506108b075c9f3b44b.jpg)  
Figure 1. PRISMA-based search strategy of the systematic review for AI-based clinical prediction tools in neurosurgery. AUC: area under the curve

Table 1. Included studies by subspeciality and AUC reporting.
<table><tr><td>Subspeciality</td><td>Studies, n(%)</td><td>Reporting AUC, n</td></tr><tr><td>Neuro-oncology</td><td>470 (44.2%)</td><td>172</td></tr><tr><td>Spine</td><td>184 (17.3%)</td><td>94</td></tr><tr><td>Neuro-vascular</td><td>180 (16.9%)</td><td>83</td></tr><tr><td>Functional/Epilepsy</td><td>83 (7.8%)</td><td>27</td></tr><tr><td>Trauma</td><td>82 (7.7%)</td><td>46</td></tr><tr><td>Pediatric neurosurgery</td><td>26 (2.5%)</td><td>10</td></tr><tr><td>Skull-base</td><td>24 (2.3%)</td><td>11</td></tr><tr><td>General</td><td>14 (1.3%)</td><td>8</td></tr><tr><td>Total</td><td>1063 (100%)</td><td>451</td></tr></table>

AUC: Area under the curve.

![](images/c614f16c92e252655277c2c259340074ca297c34646e26a5fab48e56a9d24d3c.jpg)  
Figure 2. Machine learning algorithms identified in the studies (a) and primary model family (b) across the studies. K-NN: K nearest neighbour, SVM: support vector machine, ML: machine learning; CNN: convolutional neural network; NN: neura network; LASSO: Least absolute shrinkage and selection operator

Table 2. Random-effects pooled AUC, heterogeneity and prediction intervals.
<table><tr><td>Subspeciality</td><td>n</td><td>Pooled AUC (95% CI)</td><td> $1 ^ { 2 } \ ( \% )$ </td><td> $\tau ^ { 2 }$ </td><td>95% PI</td></tr><tr><td>Neuro-oncology</td><td>172</td><td>0.87 (0.86–0.88)</td><td>95.6</td><td>0.595</td><td>0.59-0.97</td></tr><tr><td>Spine</td><td>94</td><td>0.84 (0.83–0.85)</td><td>99.1</td><td>0.118</td><td>0.72-0.91</td></tr><tr><td>Neuro-vascular</td><td>83</td><td>0.84 (0.82–0.86)</td><td>97.7</td><td>0.415</td><td>0.60-0.95</td></tr><tr><td>Trauma</td><td>46</td><td>0.86 (0.83–0.88)</td><td>97.7</td><td>0.450</td><td>0.60-0.96</td></tr><tr><td>Functional/epilepsy</td><td>27</td><td>0.84 (0.73–0.91)</td><td>98.8</td><td>2.681</td><td>0.14-0.99</td></tr><tr><td>Skull Base</td><td>11</td><td>0.87 (0.83–0.91)</td><td>76.1</td><td>0.262</td><td>0.67-0.96</td></tr><tr><td>Paediatrics</td><td>10</td><td>0.85 (0.79–0.90)</td><td>85.7</td><td>0.364</td><td>0.57-0.96</td></tr><tr><td>General</td><td>8</td><td>0.92 (0.88–0.94)</td><td>90.2</td><td>0.333</td><td>0.71-0.98</td></tr><tr><td>Overall</td><td>451</td><td>0.86 (0.85–0.86)</td><td>98.6</td><td>0.264</td><td>0.68-0.94</td></tr></table>

AUC: area under the curve; CI: confidence interval; PI: prediction intervals; n: number of studies; I<sup>2</sup>, s<sup>2</sup>: statistical heterogeneity metrics.

## 3.2. Prediction across the clinical pathway

All the studies were mapped onto the clinical neurosurgical care pathway (Figure 4). The majority of studies were dedicated to diagnosis and prognostication: 276 studies addressed diagnosis, grading or molecular characterisation and 361 studies for baseline prognostication (total: 60% of studies), followed by the post-operative stage (n ¼ 238), where spine, trauma and functional/epilepsy work is concentrated. Applications at the pathway extremes were sparse: pre-diagnosis/screening (n ¼ 32, predominantly neurovascular aneurysm rupture-risk), pre-operative decision support (n ¼ 25) and intra-operative use (n ¼ 22) and adjuvant/neoadjuvant treatment-response modelling (n ¼ 51) was almost entirely confined to neurooncology (n ¼ 47 out of 51). Median reported discrimination was highest at the imaging-anchored diagnosis/molecular stage (0.90) and lowest at postoperative (0.82) (Suppl. Figure S3, Suppl. Table S2).

## 3.3. Secondary performance metrics

Secondary metrics (Table 3) pooled with the same random-effects model were: accuracy 0.88 (95% CI 0.86–0.90; n ¼ 83), specificity 0.88 (0.84–0.90; n ¼ 64), sensitivity 0.87 (0.85–0.89; n ¼ 57) and c-index 0.80 (0.77–0.83; n ¼ 31)(Suppl. Figure S4a, Suppl. Table S3a). Sub-group analyses (n � 5 per stratum; Suppl. Figure S4b,S4c, Suppl. Table S3b,S3c) reproduced the pathway gradient seen for AUC: accuracy(0.89), sensitivity(0.89) and specificity(0.93) were all highest at the diagnosis/grading/molecular stage and lower at baseline prognostication and post-operatively.

## 3.4. Types of prediction models

694 studies (65%) developed prognostic models, 283 (27%) diagnostic models and only 8 forecasting models. Diagnostic models pooled higher performance than prognostic models (0.89, 95% CI: 0.88–0.90, versus 0.83, 0.83–0.84) consistent with the observation that distinguishing a clinical state that already exists is an easier task compared to predicting a future clinical event. Supplemental Table S4 describes the prediction model types by subspeciality with pooled metrics.

(random-effects; thick line = 95% Cl, thin line = 95% PI)  
![](images/36ba5dafbdac79c873d626bc274b4d78a1aa9fc2e2c660de0e1955ca506ada67.jpg)

Figure 3. Pooled discrimination metric (AUC) by neurosurgical subspeciality. AUC: area under the curve; CI: confidence interval, PI: prediction interval; I<sup>2</sup>: statistical heterogeneity metric  
![](images/9caca9201196b89f06c20eb4282d3f35687c9fc817f7a3778af5b554098d5686.jpg)  
Figure 4. Machine learning prediction studies in neurosurgery by clinical-pathway stage and subspeciality. dx: diagnostic

## 3.5. Level of evidence

By TRIPOD<sup>14,15</sup> study type, 161 studies (15%) reported apparent performance only (Type 1a), 420 (40%) used resampling (Type 1b), 123 (12%) used a random split-sample design (Type 2a), 20 (2%) used a non-random split: temporal, geographic or with randomisation not stated (Type 2b), 336 (32%) validated on separate data (Type 3) and only 2 were validationonly studies of an existing model (Type 4)(Table 4). Level of evidence differed by subspeciality: neuro-vascular had the highest proportion of prospective studies with external/temporal validation (10%) and neurooncology the lowest (2%). Level of evidence tracked risk of bias closely: 100% of no validation studies and 93% of internal validation only studies were a high overall risk of bias, compared with 82% of prospective studies with external or temporal validation. CHARMs model evaluation of validation methods is described in Suppl. Table S5. Only 51 studies (4.8%) reached prospective or trial data with external or temporal validation. 479 (45%) internal validation only and 140 (13.2%) no validation.

Table 3. Pooled secondary performance metrics, overall and by clinical pathway stage.
<table><tr><td>Metric</td><td>Clinical pathway stage / stratum</td><td>n</td><td>Pooled estimate (95% CI)</td><td> $1 ^ { 2 } \ ( \% )$ </td></tr><tr><td>Accuracy</td><td>Overall</td><td>83</td><td>0.88 (0.86–0.90)</td><td>98.6</td></tr><tr><td rowspan="6">Sensitivity</td><td>At diagnosis: dx/grading/molecular</td><td>41</td><td>0.89 (0.86–0.92)</td><td>97.5</td></tr><tr><td>Baseline prognostication</td><td>7</td><td>0.84 (0.73–0.91)</td><td>97.7</td></tr><tr><td>Post-operative</td><td>26</td><td>0.88 (0.84–0.91)</td><td>99.2</td></tr><tr><td>Overali</td><td>57</td><td>0.87 (0.85–0.89)</td><td>98.2</td></tr><tr><td>At diagnosis: dx/grading/molecular</td><td>22</td><td>0.89 (0.83–0.93)</td><td>98.8</td></tr><tr><td>Baseline prognostication</td><td>10</td><td>0.85 (0.80–0.89)</td><td>92.6</td></tr><tr><td rowspan="4">Specificity</td><td>Post-operative</td><td>16</td><td>0.87 (0.84–0.90)</td><td>97</td></tr><tr><td>Overali</td><td>64</td><td>0.88 (0.84–0.90)</td><td>99.2</td></tr><tr><td>At diagnosis: dx/grading/molecular</td><td>24</td><td>0.93 (0.88–0.96)</td><td>99</td></tr><tr><td>Baseline prognostication</td><td>16</td><td>0.81 (0.75–0.86)</td><td>96.2</td></tr><tr><td rowspan="3">c-index</td><td>Post-operative</td><td>16</td><td>0.83 (0.75–0.89)</td><td>98.9</td></tr><tr><td>Overali</td><td>31</td><td>0.80 (0.77–0.83)</td><td>92.7</td></tr><tr><td>Baseline prognostication</td><td>23</td><td>0.79 (0.76–0.83)</td><td>93.9</td></tr></table>

Strata with n � 5; CI: confidence interval; I<sup>2:</sup> statistical heterogeneity metric; n: number of studies.

## 3.6. Stage of clinical translation

Only 4.9% of all the studies were at the stage of clinical use. 679 studies (63.9%) only developed models and went no further; 332 (31.2%) reached external or temporal validation; 34 (3.2%) released a useable artefact such as a web-based application; 15 (1.4%) reported prospective clinical evaluation of the model. No studies reported an impact or implementation study and only 3 (0.3%) studies reported regulatory clearance or documented routine deployment (Suppl. Table S6).

## 3.7. Risk of bias

Overall risk of bias was high in 962 of 1063 studies (90%) and unclear in 101; none reached low overall risk of bias. The analysis domain was the dominant driver (high-960 studies), reflecting absent calibration assessment, reliance on internal or cross-validation without external/temporal validation and uncorrected optimism. The participants domain was comparatively reassuring (low-674 studies), whereas the predictor and outcome domains were frequently unclear owing to incomplete reporting (Figure 5a). Sub-speciality risk of bias by domain remained high in each group (range: 88%-96%)( Figure 5b). Reporting completeness by each study derived from PROBAST-AI signalling answers are plotted in Suppl. Figure S5.

## 3.8. Sensitivity analyses

Pooled discrimination was stable across risk-of-bias strata. When restricted to externally or temporally validate studies (n ¼ 241), pooled AUC was 0.86 (95% CI 0.85–0.86) and excluding high-risk-of-bias studies (n ¼ 71) pooled AUC was 0.86 (0.84–0.89), both essentially identical to the primary estimate of 0.86 (Figure 5c). The high pooled discrimination is therefore not an artefact of the weakest studies; however, the better-conducted subset did not resolve the extreme heterogeneity $( \boldsymbol { \mathrm { I } } ^ { 2 }$ remained > 98%).

## 3.9. Heterogeneity-source and calibration analysis

Extreme heterogeneity was observed in each analysis. The sources of heterogeneity was examined in subgroup analyses (Suppl. Table S7) where subgroup mean differed significantly for 6 of the 9 sources tested: subspeciality (Q ¼ 22.73, df ¼ 7, p ¼ 0.002); clinical pathway stage (Q ¼ 113.21, df ¼ 7, p < 0.001); primary model family (Q ¼ 21.24, df ¼ 5, p < 0.001); validation status (Q ¼ 8.39, df ¼ 2, p ¼ 0.015); predicted outcome (Q ¼ 108.24, df ¼ 4, p < 0.001); study design (Q ¼ 15.53, df ¼ 3, p ¼ 0.001). Overall risk of bias (Q ¼ 65, p ¼ 0.42) and source of standard error (p ¼ 0.19) did not discriminate. In random-effects meta-regression the largest single contributor was subspeciality, explaining 24.9% of $\tau ^ { 2 }$ with validation status explaining 6.2%, sample size 3.6% and risk of bias 1.7%. None of the other sources explained $\tau ^ { 2 } .$ This difference likely arose from large within-stratum heterogeneity with marked variability between strata (range of $\tau ^ { 2 } \colon \ 0 . 0 9  – 1 . 0 4$ across pathway stages). A multivariate model combining subspeciality, validation status, model family, sample size and year explained 11.8% of $\tau ^ { 2 } ~ \left( n = 3 5 6 \right)$ , having a residual $\tau ^ { 2 }$ of 0.209 against a total of 0.264.

Table 4. Level of evidence based on study design with validation and TRIPOD prediction-model study type (total 1063 studies).
<table><tr><td rowspan="2">Category based on design and validation</td><td colspan="3"></td><td rowspan="2">Studies, n (%)</td></tr><tr><td>Studies, n (%) 51 (4.8%)</td><td>Category based on TRIPOD Type 1a</td><td>Definition Development only (apparent</td></tr><tr><td>Prospective cohort or trial data with external/temporal validation</td><td></td><td></td><td>performance)</td><td>161 (15.1%)</td></tr><tr><td>Prospective cohort/trial without external validation, OR retrospective/unspecified design with external</td><td>393 (37%)</td><td>Type 1b</td><td>Development + resampling (internal validation)</td><td>420 (39.5%)</td></tr><tr><td>validation Internal validation only (resampling, cross-validation</td><td>479</td><td>Type 2a</td><td>Random split-sample</td><td>123</td></tr><tr><td>or split-sample) No validation reported</td><td>(45%)</td><td></td><td>development and validation</td><td>(11.6%)</td></tr><tr><td>(apparent performance only)</td><td>140 (13.2%)</td><td>Type 2b</td><td>Non-random split-sample (temporal/geographic or</td><td>20 (1.9%)</td></tr><tr><td></td><td></td><td>Type 3</td><td>randomisation not stated) Development + validation on</td><td>336</td></tr><tr><td></td><td></td><td>Type 4</td><td>separate data Validation only of an existing</td><td>(31.6%)</td></tr><tr><td></td><td></td><td></td><td>model</td><td>2 (0.2%)</td></tr><tr><td></td><td></td><td>unclear</td><td>Validation strategy not ascertainable from text</td><td>1 (0.1%)</td></tr></table>

TRIPOD: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis.

Calibration, on the other hand, was identifiable only in 36.2% studies overall (n ¼ 385) and in 45.5% studies (n ¼ 205) of the 451 studies reporting AUC. Decision-curve analysis appeared in 190 (17.9%) studies. Calibration reporting was sparse and heterogenous.

## 3.10. Small-study effects and publication bias analysis

The AUC values were plotted against its standard error in a funnel plot demonstrating asymmetry with a near-absence of studies reporting discrimination below 0.70 reference (Egger intercept 4.38, 95% CI: 3.61–5.16; p < 0.001; n ¼ 451) (Figure 6). Funnel asymmetry has explanations besides publication bias – small-study effects, heterogeneity in task difficulty and the mathematical dependence between AUC and its variance. The finding is consistent with selective publication and selective outcome reporting of well-performing studies.

## 3.11. Clinical prediction task sub-stratification analysis

Thirty-five different clinical predictions tasks were identified. 9 tasks had >5 studies and were pooled. 16 tasks had studies between 2 and 5 and were described with median AUC and range, not pooled (Suppl. Table S8). Imaging-to-molecular tasks sit as the top performers (IDH mutation AUC 0.91, glioma grade AUC 0.90, MGMT methylation AUC 0.84) and clinical-outcome tasks at the bottom (readmission or reoperation AUC 0.82, seizure freedom after epilepsy surgery AUC 0.73). Glioma survival prediction task was the most numerous with 120 studies with 34 reporting AUC (0.83; 95% CI: 0.80–0.86) with only half reporting calibration. The spread across the tasks (AUC 0.73–0.91) is wider than the spread across subspecialities. Heterogeneity remained relevant for within-task analyses too $( \tau ^ { 2 } { : } 0 . 3 3$ against 0.27 studywide). Relatively homogenous pooled tasks were identified as seizure freedom after epilepsy surgery $( \mathrm { I } ^ { 2 } { = } 2 2 . 2 \% , \ \tau ^ { 2 } { = } 0 . 0 1 )$ and shunt failure after hydrocephalus $( \mathrm { I } ^ { 2 } { = } 7 7 . 2 \% , \tau ^ { 2 } { = } 0 . 1 5 )$

## 4. Discussion

This is the largest corpus on the analysis of neurosurgical AI clinical prediction studies till date.<sup>1–6</sup> Sub-speciality level meta-analysis, sub-stratification prediction task analysis, risk-of-bias-stratified sensitivity analyses with prediction intervals and level of evidence synthesis are strengths of the study demonstrating heterogeneity.

The models reported uniformly high discrimination across all subspecialities (pooled AUC: 0.84–0.87), and this signal was robust to risk-of-bias stratification. Three findings temper the results. First, approximately nine in ten studies were at high overall risk of bias, largely because of the analysis domain. Second, heterogeneity was extreme $( \boldsymbol { \mathrm { I } } ^ { 2 } \sim 9 9 \% )$ , so the pooled estimates summarise a highly dispersed literature rather than a transferable level of performance. Third, most models were internally validated and uncalibrated, so reported AUCs are best read as optimismprone upper bounds.

![](images/856bf072c64150a30d46cfa0b4b531eadeff03519a82c6b539fc8b9106ede082.jpg)

a. PROBAST-Al risk of bias by domain  
![](images/46ca69c07c9dc4ac8e16f74b0c77fcade6752fb4541cafe031848ac858fa880a.jpg)  
c. Risk-of-bias-stratified sensitivity analysis  
Figure 5. Risk-of-bias assessment using PROBAST checklist by domain (a), subspeciality (b)and risk-of-bias-stratified sensitivity analysis (c) for pooled AUC metric. There was overall high risk-of-bias overall dominated by analysis domain which was replicated by subspeciality risk-of-bias analysis with high risk-of-bias in each group ranging from 88% in skull base to 96% in paediatrics. ROB: risk of bias; AUC: area under the curve; PROBAST: Prediction model Risk Of Bias ASsessment Tool

AUC: area-under-the-curve; AUC logit: logit transformation of AUC scores for visualisation; standard error: measure of precision to AUC: area-under-the-curve; AUC logit: logit transformation of AUC scores for visualisation; standard error: measure of precision to detect asymmetry  
![](images/05d268fbd13d0eef7f9de83dd6af732335536d35e607845a1fdd25673b2a37d7.jpg)  
Figure 6. Funnel plot of AUC against its standard error for all AUC-reporting neurosurgical studies (n ¼ 451). Each point is one study’s primary AUC, expressed on the logit scale (lower axis); the upper axis gives the corresponding AUC. The vertical axis is the standard error of the logit AUC, inverted. The red dotted line marks the reference AUC of 0.70 (conventional threshold for acceptable discrimination) about which the dashed pseudo-95% confidence limits and the shaded triangle are centred. The solid navy line is the random effects pooled estimate of AUC 0.86 (95% CI 0.849–0.862; $\tau ^ { 2 } = 0 . 2 6 7 , \ l ^ { 2 } = 9 8 \bar { . } \bar { 6 } \% )$ . Egger’s regression test on the logit scale gave an intercept of 4.38 (95% CI 3.60–5.16), p < 0.001, indicating pronounced funnel asymmetry, with a conspicuous absence of studies reporting discrimination at or below the 0.70 reference.

Placing this study along the clinical neurosurgical care pathway sharpens where the field stands. Effort is heavily front-loaded at diagnosis, molecular characterisation and baseline prognostication especially the imaging-rich tasks where discrimination is highest. On the other hand, the stages closest to an operative decision are unevenly served: post-operative decision support and intraoperative guidance, remain sparse. These gaps and lack of treatment response modelling outside of neuro-oncology mark where clinically useful development is most needed and highest-value prediction (informing whether and how to operate or which therapy to give) is currently least mature.

The secondary metrics extracted independently of AUC and from a different subset of studies, reproduced the same ordering, with the diagnosis, grading and molecular stage with highest score (specificity 0.93) and prognostic/post-operative tasks with a lower score. Convergence across metrics derived from different reported quantities makes it unlikely that the clinical pathway gradient reflects selective AUC reporting alone and supports the interpretation that imaginganchored classification is an easier prediction problem than forecasting a future clinical course.

Level of evidence assessment demonstrates the translation gap: only 5% of studies reach prospective cohort or trial data with external or temporal validation, while 58% rest on internal validation alone or on apparent performance with no validation. Level of evidence and risk of bias were aligned: every internal or no validation study was rated high risk of bias identifying the deficits in these studies. The subspeciality with the strongest discrimination (neuro-oncology) had the lowest proportion of prospective evidence, whereas neurovascular neurosurgery, with more prospective and externally validated work, reported lower pooled AUCs. This inverse relationship shows that apparent performance is inflated by optimism and is a caution against interpreting higher pooled AUC as evidence of greater clinical readiness.

These patterns mirror the broader literature on clinical ML prediction, where the analysis domain is universally the weak point. The recurring, addressable deficiencies were absent calibration assessment (discrimination-only reporting), internal-only validation and inconsistent handling and reporting of missing data. Small number of studies that are externally validated and non-high-risk subsets produced similar high pooled AUCs demonstrating that performance is not being increased solely by the weakest work, but the persistence of heterogeneity shows that aggregate discrimination is not a meaningful guide to the performance of individual models in a new unseen population.

The practical implication is that the neurosurgical field<sup>1–6</sup> should shift emphasis from reporting increasing in-sample AUCs and secondary metrics towards demonstrating external and temporal validation, routine calibration, adequate events-per-variable, transparent missing-data handling and complete reporting of these studies using standardised reporting guidelines (TRIPOD, TRIPOD-AI, PROBAST, PROBAST-AI)<sup>13–15</sup> that are prerequisites for trustworthy clinical deployment.

## 4.1. The translation gap

Across all studies spanning every neurosurgical subspeciality and a decade of publication, 63.9% studies stopped at model development and a further 31.2% at external validation. Only 52 studies (4.9%) produced an artefact a clinician could use or evaluated the model in prospective clinical practice, and none reported an impact or implementation study. The field is generating discrimination statistics at scale but not enough clinical evidence for neurosurgical application. The bottleneck is no longer algorithmic; it is calibration at local population, a usable interface, prospective silent period testing and a design capable of detecting whether neurosurgical decisions and outcomes change.

## 5. Limitations

## 5.1. Studies assessed

Limitations encompassed three main areas.

First, pooling of single AUC scores per study across heterogeneous outcomes, predictors and population is coarse, made visible by the extreme heterogeneity $\mathrm { I } ^ { 2 }$ and $\tau ^ { 2 }$ values. AUC metric alone can be misrepresentative and thus performance assessment should be done based on multiple complementary metrics and task-based sub-stratification analyses. Second, secondary metrics were pooled only from studies that reported a single value for each metric leading to these studies being small in number and may not be representative and paired data were limited to permit a bivariate modelling. Third, in minority of studies that did not have CI and sample size reported, standard errors were imputed, so pooled CI are approximate.

## 5.2. Review process

Title and abstract level exclusion of studies is likely to provide a conservative method in including ML prediction studies for neurosurgery. Publication bias may have affected the range of metrics and studies included in the systematic review especially by exclusion of preprints and materials that have not undergone peer review partially answered by small study effects and publication bias analyses. This is relevant to these studies where the rapid pace of development often outstrips the slower process of peer review, leading to some researchers avoiding the submission to peer-reviewed journals.

## 6. Conclusion

AI based clinical prediction tools in neurosurgery report consistently high discrimination across subspecialities and clinical predictions tasks, but the literature is dominated by internally validated and poorly calibrated models with extreme heterogeneity with majority at high risk of bias. External and temporal validation, calibrating reporting and adherence to reporting guidelines are immediate priorities to translate promising discrimination into trustworthy clinical tools in neurosurgery.

## Author contributions

PG conceived the study, performed search, screening, extraction, analysis and drafting. Co-authors AC, JS, KA contributed to second review, interpretation and critical revision. All authors reviewed and approved the manuscript.

## Ethical statement

All studies were retrieved from published literature. No explicit ethical approval was obtained.

## Disclosure statement

No potential conflict of interest was reported by the author(s).

## Software and AI-use declaration

RStudio, Python/PyCharm and Biorender.com were used to perform statistical analyses and create graphs respectively. Authors declare no use of AI during drafting of the manuscript or any supplementary materials.

## ORCID

Prajwal Ghimire http://orcid.org/0000-0003-2884-956X

## Data availability

Full study level datasets, metrics calculations and PROBAST-AI domain judgements are available as supplemental files.

## References

1. Senders JT, Staples PC, Karhade AV, et al. Machine learning and neurosurgical outcome prediction: a systematic review. World Neurosurg 2018;109:476–86.e1. Epub 2017 Oct 3. PMID: 28986230.

02. Reyes JS, Lohia VN, Almeida T, et al. Artificial intelligence in neurosurgery: a systematic review of applications, model comparisons, and ethical implications. Neurosurg Rev 2025;48:455. PMID: 40439939.

3. Huang Y. From severity scoring to predictive analytics: the emerging role of AI in neurosurgery. Front Neurol 2026;17:1753844. PMID: 42434073; PMCID: PMC13353086.

4. Tangsrivimol JA, Schonfeld E, Zhang M, et al. Artificial intelligence in neurosurgery: a state-of-theart review from past to future. Diagnostics (Basel) 2023;13:2429. PMID: 37510174; PMCID: PMC10378231.

5. Christodoulou E, Ma J, Collins GS, et al. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. J Clin Epidemiol 2019;110:12–22. Epub 2019 Feb 11. PMID: 30763612.

06. Andaur Navarro CL, Damen JAA, Takada T, et al. Risk of bias in studies on prediction models developed using supervised machine learning techniques: systematic review. BMJ 2021;375:n2281. PMID: 34670780; PMCID: PMC8527348.

7. Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ 2021;372:n71. PMID: 33782057; PMCID: PMC8005924.

08. Moher D, Shamseer L, Clarke M, et al. Preferred reporting items for systematic review and meta-

analysis protocols (PRISMA-P) 2015 statement. Syst Rev 2015;4:1. PMID: 25554246; PMCID: PMC4320440.

09. Cacciamani GE, Chu TN, Sanford DI, et al. PRISMA AI reporting guidelines for systematic reviews and meta-analyses on AI in healthcare. Nat Med 2023;29: 14–5.

10. Guolo A, Varin C. Random-effects meta-analysis: the number of studies matters. Stat Methods Med Res 2017;26:1500–18. Epub 2015 May 7. PMID: 25953957.

11. Emir B, Can FE, Kaymaz E, et al. Sample size and power analysis for ROC AUC differences in diagnostic tests: a methodological evaluation of the Obuchowski-McClish and Hanley-McNeil methods. BMC Med Res Methodol 2026;26:43. PMID: 41606502; PMCID: PMC12924612.

12. Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. Ann Intern Med 2019;170:51–8. PMID: 30596875.

13. Collins GS, Dhiman P, Andaur Navarro CL, et al. Protocol for development of a reporting guideline (TRIPOD-AI) and risk of bias tool (PROBAST-AI) for diagnostic and prognostic prediction model studies based on artificial intelligence. BMJ Open 2021;11: e048008. PMID: 34244270; PMCID: PMC8273461.

14. Collins GS, Reitsma JB, Altman DG, Moons KG. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. Ann Intern Med 2015;162:55–63. Erratum in: Ann Intern Med. 2015 Apr 21;162(8):600. doi: PMID: 25560714.

15. Collins GS, Moons KGM, Dhiman P, et al. TRIPODþAI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378. Erratum in: BMJ. 2024 Apr 18;385:q902. doi: PMID: 38626948; PMCID: PMC11019967.

16. Moons KG, de Groot JA, Bouwmeester W, et al. Critical appraisal and data extraction for systematic reviews of prediction modelling studies: the CHARMS checklist. PLoS Med 2014;11:e1001744. Published 2014 Oct 14.

17. Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. BMJ 2022;377:e070904. Published 2022 May 18.

18. OCEBM Levels of Evidence Working Group. The Oxford Levels of Evidence 2. Oxford Centre for evidence-based medicine; version 2.1. Available from: https://www.cebm.ox.ac.uk/resources/levels-of-evidence/ocebm-levels-of-evidence