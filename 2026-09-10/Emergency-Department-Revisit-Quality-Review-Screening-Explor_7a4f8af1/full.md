# Emergency Department Revisit Quality Review Screening: Exploring Human Decision-Making and Artificial Intelligence Support

Preprint: Not yet peer-reviewed

Authors (with numeric identifier for affiliation, city, state in parentheses): Jonathan A. Handler, MD (1, 11), Marlene I. Robles-Granda, BSc, MSc (2), Jacob E. Mefford, PA-C (3), Jeremy S. McGarvey, MS (4), Gregory S. Podolej, MD, MHPE (5, 6), Colleen J. Klein, PhD, APRN, FNP-BC, FAAN (7, 8), Matthew D. Dalstrom, PhD, MPH (9), William F. Bond, MD, MS (10).

Author Affiliation, City, and State: (1) Clinical Intelligence and Advanced Data Lab, OSF HealthCare, Peoria, IL. (2) Digital Innovation Development, OSF HealthCare, Peoria, IL. (3) OnCall, OSF HealthCare, Peoria, IL. (4) Department of Healthcare Analytics, OSF HealthCare, Peoria, IL. (5) Department of Emergency Medicine, University of Illinois College of Medicine at Peoria, Peoria, IL. (6) OSF HealthCare, Peoria, Il. (7) Center for Advanced Practice, OSF HealthCare, Peoria, IL. (8) Saint Anthony College of Nursing, Rockford, IL. (9) Graduate Department, Saint Anthony College of Nursing, Rockford, IL. (10) Department of Emergency Medicine, Thomas Jefferson University's Sidney Kimmel College of Medicine, Philadelphia, PA.

Note: Jonathan Handler is currently at Keylog Solutions LLC, Northbrook, IL (Keylog). During the work, he had the roles at Keylog as noted in the Declaration of Interests. However, at the time this work was performed, it was performed while he was at, performed under the auspices of, and performed in his role with, affiliation #1 above.

## ABSTRACT

Background: Emergency Department (ED) return visits are commonly reviewed for quality assurance, but are often limited (e.g., to revisits within 48-72 hours) to increase actionable finding yield while minimizing chart review burden. Those limitations may lead to missed quality improvement opportunities.

Methods: We conducted an exploratory, retrospective study of randomly selected ED visits to a multihospital health system having an ED revisit within 1-14 days to the same health system. Given only each visit’s primary diagnosis, raters (2-3 clinicians and GPT-4 large language model [LLM]) assessed characteristics of the diagnosis pairs, including the “target”: whether a pair warranted further assessment. Informed by rater response analyses, an algorithm leveraging an LLM-populated knowledge graph (“KGA”) was created to automatically screen for potentially concerning pairs, then preliminarily assessed.

Results: 99 diagnosis pairs were included. GPT-4 responses poorly correlated to clinician raters, rating nearly all (94%) pairs as warranting follow-up (4.4-13.3 times more than clinicians). However, prompt engineering was minimal. Among clinician raters, revisit medical gravity was consistently significantly associated with the target, while a differential diagnosis/complication composite was significantly associated on unadjusted, but not adjusted (though less powered) analysis. The KGA achieved 83-100% positive predictive value for at least one clinician rater determining further assessment was warranted based on the diagnosis pair.

Conclusion: These results can inform next steps for improving screening with LLMs like ChatGPT. Further research is warranted to validate this preliminary work’s finding that the KGA may enable enhancing the scope and yield of screening without substantially increasing reviewer workload.

## INTRODUCTION

Emergency Department (ED) return visit (“bounceback”) reviews have been used for quality improvement, but have been limited by the chart review burden and low yield of actionable findings.<sup>1–3</sup>

Although clinical failures can be found outside a 48-72-hour revisit window,<sup>4</sup> that window is commonly used.<sup>1,2,5</sup> Due to limited resources, even more restrictive criteria may be used. However, at our highest-volume ED, even more narrow restrictions (a 48-hour return window plus revist results in admission) still leads to a large number for review and a reportedly low yield of actionable findings.

To improve clinical quality despite constrained resources, record selection for manual review must have a high positive predictive value (PPV, few false positives). Therefore, we sought to: 1) explore factors associated with determining need for revisit manual review; 2) explore automated revisit screening for manual reviews using a large language model (LLM); and 3) derive and explore an automated revisit screening algorithm (“KGA”) that leverages an LLM-populated knowledge graph.

## METHODS

This work was approved by OSF Research Administration and the University of Illinois College of Medicine Peoria Institutional Review Board-1.

Encounter pairs were randomly selected from all ED encounters in a US multi-hospital health system during 2022. Only the primary encounter diagnosis was considered. Inclusion: age at index visit (“index”) 19-89 years; index disposition not Admit, Send to Labor and Delvery, Transfer, Send to Ancillary Department, Deceased, or null; an ED revisit to the same health system within 1-14 days after the index visit; revisit primary ICD-10 code having differing first three characters from that of the index; not having a Z53.21 (left without being seen) primary

diagnosis for either visit; and patient address mappable to a Rural-Urban Continuum Code<sup>6</sup> (for potential future investigation).

## Reviews

Two reviewers were board certified emergency physicians (WFB, GSP), one (WFB) with quality improvement chart review experience. A third (JEM) was an Urgent Care Advanced Practice Provider.

Reviewers independently assessed diagnosis pairs in Microsoft Excel.<sup>7</sup> No other encounterspecific information was provided. For both visits in the pair, reviewers rated:

• Diagnosis Membership Value (DMV): How much of a “diagnosis” (0-100) is the diagnosis? (e.g., symptom/finding get lower scores)

• Medical Gravity Value (MGV): Given only the diagnosis, how would the reviewer respond to a patient asking, "If I have a milder form of this diagnosis (25th percentile level of badness) how bad is this diagnosis?" (0-100, worse diagnoses get higher scores). Reviewers were to enter -1 if they did not consider the diagnosis “enough of a diagnosis” for “bad” to make sense. This question was repeated for typical (median) and more severe (75<sup>th</sup> percentile) forms of the diagnosis.

## Reviewers also rated:

• Differential Inclusion Value (DIV): The extent (0-100) to which the revisit diagnosis is in the differential of the index diagnosis, with instruction to enter -1 if the revisit diagnosis was not enough of a “diagnosis” for “in the differential” to make sense.

• Complication Inclusion Value (CIV): The extent (1-100) to which the revisit diagnosis is a complication of the index diagnosis, with instruction to enter -1 if rating did not make sense given the diagnosis pair.

• Target Variable: Whether further investigation seemed warranted given only the diagnosis pair and notation that the visits occurred within 14 days of one another.

An LLM (private Microsoft Azure OpenAI GPT-4<sup>8</sup> 2024-02-15-preview, or “GPT-4”) was also queried for these, each as a separate prompt. Minor prompt engineering was iteratively performed to improve response quality, so human and LLM prompt wordings were not exactly the same.

These results informed the development and preliminary evaluation (described in Results) of the KGA. The KGA depends on a knowledge graph of “potentially concerning” diagnosis pairs populated by open-source software (“Darth Vecdor”) in a previously described effort,<sup>9</sup> in part by repeatedly querying an LLM (GPT-4o mini<sup>10</sup>) about diagnoses.

## Statistical methods

Reported preliminary descriptive analyses, analyses of rater percentage of “yes” responses for the target variable, and analyses of KGA performance were performed in Excel.<sup>7</sup>

Other analyses were performed using R v4.5.2<sup>11</sup> and assume a two-sided 5% level of significance. Values of -1 (e.g., indicating that a diagnosis was considered not enough of a diagnosis to provide a rating) were treated as missing, and records with missing values were excluded from the relevant analyses. In Stage 1 and 2 analyses (see Results below), mixed effects logistic regression was used to examine the relationship between the variables of interest and the binary target variable. Random intercepts were included for rater and study ID (for the encounter pair) to account for rater and patient-level clustering effects.

## RESULTS

Filtering criteria correction removed one case, leaving 99 encounter pairs in the dataset.

## Stage 1 Analysis (Three Human Raters)

Raters varied substantially in their responses. For the target variable (further assessment warranted), Rater 1 had 21.2% (n=21) yeses, Rater 2 had 11.1% (n=11), and Rater 3 had 7.1% (n=7). GPT-4 rated nearly all (93.9%, n=93) as yeses.

Based on investigator expertise and preliminary descriptive analysis, we chose DMV, DIV, change (delta) in “Typical” MGV, and days between visits as the following independent variables for predicting the target variable in a multivariate model. Although raters were blinded to intervisit span, it was included since greater days between visits might occur more often with clinically unrelated visits that are not judged as index clinical failures.

Only “Typical” MGV delta and DIV were statistically significantly (Table 1). Odds ratios (ORs) are per percentile change.

## Stage 2 Analysis (Two Human Raters)

We later recognized the revisit diagnosis might represent a complication of the index diagnosis and influence decisions. Two raters (WFB, JEM) assessed the extent to which the revisit diagnosis was a complication of the index diagnosis (Complication Inclusion Value [CIV]).

The investigators suspected that revisit MGV may be more relevant than MGV delta, and preliminary analyses (not reported here) demonstrated revisit and delta MGVs had relatively similar ORs with significant p-values. Therefore, additional analyses used “typical” revisit MGV rather than MGV delta.

We hypothesized that a potential index misdiagnosis (proxied by DIV) or index diagnosis complication on revisit (proxied by CIV) might influence the target and created a “relationship composite” (RC) variable having the larger of CIV or DIV for each diagnosis pair. If either DIV or CIV was -1, the non-negative value was selected. If both DIV and CIV were -1, that pair was excluded from this analysis.

RC was significantly associated with the target in unadjusted analysis (Table 2) but not adjusted analysis (Table 3). However, this may have been due to exclusions leading to a smaller sample size, since the RC unadjusted analysis no longer showed significance at a lower sample size resulting from exclusion of the same records excluded in the adjusted model (Table 3).

## Stage 3 Analysis (Algorithm Performance)

Informed by these results, the KGA was developed, leveraging the LLM-generated KG to select pairs potentially indicative of mis-/delayed diagnosis or an index diagnosis complication (akin to the RC). The graph includes a “Medical Gravity Index” (MGI, range 44-219) for each diagnosis (analogous to the clinician-provided “MGV”). The algorithm further filtered results by revisit MGI at two cutoffs (>=90, >= 100). Returned pairs were labeled “positive” (warranting manual follow-up), all others “negative” (Figure 1).

28/99 visit pairs (28.3%) were flagged by at least one rater as warranting further investigation (“actual positives”). At the lower MGI cutoff, 5/6 were True Positives (“TPs”, 83% PPV). At the higher cutoff, 4/4 were TPs (100% PPV). Among TPs, 4/5 (80%) at the lower cutoff and 4/4 (100%) at the higher cutoff were revisits at 6-7 days that would be missed by 48–72-hour revisit screening.

## LIMITATIONS

As appropriate for a brief research report, this paper had a relatively small sample size from a single (albeit large, multihospital) health system.<sup>15</sup> Rater agreement was also limited. Therefore, follow-up validation is needed..

Future work should explore a broader set of inputs (e.g., revisit disposition). However, these preliminary results showed high precision, and the KGA’s requirement of only a few commonly available inputs may facilitate rapid implementation.

## DISCUSSION

Chart reviews for quality assessment entail high cost and effort, and computer-based automation to screen ED visits for adverse events, has been proposed to help address this.<sup>3</sup> Recent work used machine-learning to identify potential missed diagnostic opportunities for specific conditions.<sup>12</sup> LLM-based differential diagnosis generators have been studied,<sup>13</sup> and others have proposed using symptom-disease pairs to help identify diagnostic error.<sup>14</sup> Our work expands upon these efforts by incorporating all three of: 1) differential diagnosis, 2) potential complications, and 3) medical gravity. Doing so via the LLM-informed KGA aims to get LLM benefits with improved cost, speed, consistency, correctability, and explainability of results. Another strength of this work is that it was not limited to specific conditions, symptoms, or more restrictive return visit criteria.

The revisit MGV was most strongly and consistently associated with the target variable. The RC association was significant in unadjusted but not adjusted analysis. However, the latter had a smaller sample size due to more excluded cases, potentially leading to a false conclusion of insignificance. This is supported by the finding that the RC unadjusted model did not achieve significance with a dataset of only subjects included in the adjusted model.

The strengths of our reported associations are much larger than what might be inferred from their small ORs because the ORs apply to each per-point change on a \~100-point scale.

Most GPT-4 target variable responses were false positives. However, only light prompt engineering was performed, and visit pair prompts were asked separately. Asking them all at once might have allowed for a form of reasoning, potentially improving responses. Newer LLMs are now available and may perform better. Our findings suggest that a different model and/or prompt engineering would be needed to achieve better results.

The KGA shows promise as a high-precision mechanism to expand revisit analysis. Although sensitivity was low (14-18%), the appropriate comparator is the zero sensitivity for capturing cases not reviewed (e.g. revisits after 48-72 hours). For this use, the KGA’s high PPV is the relevant metric.

In summary, this study explored clinician and AI-based decision-making, then used the results to inform the derivation of the KGA algorithm. These results can inform next steps for improving screening with LLMs like ChatGPT. Further research is warranted to validate this preliminary work’s finding that the KGA may enable enhancing the scope (e.g., beyond the traditional 72-hour window) and yield of screening without substantially increasing reviewer workload.

## Acknowledgements

The authors would like to gratefully thank and acknowledge Dr. Lisa Barker; Dr. John A. Vozenilek; Rebecca A. Ebert-Allen; Sarah Soorya; Claushayla M. Nunn; Milind Fougler; and Susan Wolf for their assistance in this research effort.

## Funding

This work was funded and enabled through the generous support of OSF HealthCare Systems' OSF Innovation program.

## Data Availability Statement

The publicly available datasets, the open-source software platform, and the platform’s configurations that were all used in this study have been cited or are cited in cited work. Data used by the open-source platform to generate the knowledge graph have themselves been cited in the platform- and/or configuration-related citations. Some data from this study cannot be publicly shared because they are protected health information, subject to ethical and privacy

policies and regulations, and/or subject to the limitations of the IRB and/or institutional approvals/agreements for this study.

## Declaration of Interests

Jonathan Handler is chief executive officer and a shareholder in Keylog Solutions LLC; has received funding from Pfizer; is a shareholder in other healthcare companies including Whispersom Corporation, EmOpti LLC, HealthLab LLC, and Baxter Healthcare; is a shareholder in several companies that relate to the artificial intelligence space (e.g., Nvidia, Marvel, others); serves in an advisory role to Whispersom Corporation, EmOpti LLC, and HealthLab LLC; has various patents that have been granted or are pending; and related to activities for the respective institutions has received stipend, food, and/or travel from the American Medical Association; For the remaining authors, all declare no relevant interests to disclose. All disclosures are to the best of each author’s knowledge.

## Author Contributions

All authors attest to meeting authorship criteria as follows: Conception and Design (JAH), Interpretation of data (JAH, JSM, CJK, MDD, JEM, WFB), Drafting the article (JAH, JSM, WFB), Acquisition of data (JAH, MIR), Analysis (JAH, GSP, JEM, WFB), Revising the article critically for important intellectual content (JAH, MIR, JSM, CJK, MDD, GSP, JRM), Final approval of the version to be submitted (JAH, MIR, JSM, CJK, MDD, GSP, JEM, WFB).

## Keywords

Quality Improvement, Quality Metrics, Artificial Intelligence, Informatics, Emergency Medicine.

## References

1. Peng P, Davazdahemami B, Delen D, Shapiro J, Manini AF. 288 Predicting Potential Bouncebacks to the Emergency Department: A Machine Learning Approach. Annals of Emergency Medicine. 2019;74(4):S114. doi:10.1016/j.annemergmed.2019.08.246

2. Abualenain J, Frohna WJ, Smith M, et al. The prevalence of quality issues and adverse outcomes among 72-hour return admissions in the emergency department. J Emerg Med. 2013;45(2):281-288. doi:10.1016/j.jemermed.2012.11.012

3. Agatstein K. Chart Review Is Dead; Long Live Chart Review: How Artificial Intelligence Will Make Human Review of Medical Records Obsolete, One Day. Popul Health Manag. 2023;26(6):438-440. doi:10.1089/pop.2023.0227

4. Aaronson E, Jansson P, Wittbold K, Flavin S, Borczuk P. Unscheduled return visits to the emergency department with ICU admission: A trigger tool for diagnostic error. The American Journal of Emergency Medicine. 2020;38(8):1584-1587. doi:10.1016/j.ajem.2019.158430

5. Navanandan N, Schmidt SK, Cabrera N, Topoz I, DiStefano MC, Mistry RD. Seventy-twohour Return Initiative: Improving Emergency Department Discharge to Decrease Returns. Pediatr Qual Saf. 2020;5(5):e342. doi:10.1097/pq9.0000000000000342

6. USDA Economic Research Service. Rural-Urban Continuum Codes. Published online May 2013. Accessed July 21, 2022. https://www.ers.usda.gov/data-products/rural-urbancontinuum-codes/documentation/

7. Microsoft Excel. In: Wikipedia. 2026. Accessed May 5, 2026. https://en.wikipedia.org/w/index.php?title=Microsoft\_Excel&oldid=1351408124

8. Boyd E. Introducing GPT-4 in Azure OpenAI Service. Microsoft Azure Blog. March 21, 2023. Accessed May 18, 2026. https://azure.microsoft.com/en-us/blog/introducing-gpt4-in-azureopenai-service/

9. Handler JA. AI to Help Assess Clinical Quality: Example (potential) Darth Vecdor Use Case + Configs. Zero Effectors. April 14, 2026. Accessed April 20, 2026. https://zeroeffectors.com/2026/04/14/ai-to-help-assess-clinical-quality-example-potentialdarth-vecdor-use-case-configs/

10. GPT-4o mini: advancing cost-efficient intelligence. OpenAI. Accessed May 18, 2026. https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/

11. The Comprehensive R Archive Network. Accessed May 5, 2026. https://cran.r-project.org/

12. Zimolzak AJ, Wei L, Mir U, et al. Machine Learning to Enhance Electronic Detection of Diagnostic Errors. JAMA Netw Open. 2024;7(9):e2431982. doi:10.1001/jamanetworkopen.2024.31982

13. McDuff D, Schaekermann M, Tu T, et al. Towards accurate differential diagnosis with large language models. Nature. 2025;642(8067):451-457. doi:10.1038/s41586-025-08869-4

14. Liberman AL, Newman-Toker DE. Symptom-Disease Pair Analysis of Diagnostic Error (SPADE): a conceptual framework and methodological approach for unearthing misdiagnosis-related harms using big data. BMJ Qual Saf. 2018;27(7):557-566. doi:10.1136/bmjqs-2017-007032

15. Teresi JA, Yu X, Stewart AL, Hays RD. Guidelines for Designing and Evaluating Feasibility Pilot Studies. Med Care. 2022;60(1):95-103. doi:10.1097/MLR.0000000000001664

## Tables

Table 1: Stage 1 Adjusted Logistic Regression for Warrants Follow-up
<table><tr><td rowspan=1 colspan=1>Independent Variable</td><td rowspan=1 colspan=1>Warrants Follow-upOdds Ratios (95% CI)(n=130)</td><td rowspan=1 colspan=1>p</td></tr><tr><td rowspan=1 colspan=1>Index DMV</td><td rowspan=1 colspan=1>0.99 (0.97 – 1.01)</td><td rowspan=1 colspan=1>0.393</td></tr><tr><td rowspan=1 colspan=1>“Typical&quot; MGV Delta</td><td rowspan=1 colspan=1>1.07 (1.03 – 1.11)</td><td rowspan=1 colspan=1>&lt;0.001</td></tr><tr><td rowspan=1 colspan=1>DIV</td><td rowspan=1 colspan=1>1.02 (1.005 – 1.04)</td><td rowspan=1 colspan=1>0.013</td></tr><tr><td rowspan=1 colspan=1>Days Between ED Visits</td><td rowspan=1 colspan=1>0.96 (0.83 – 1.11)</td><td rowspan=1 colspan=1>0.606</td></tr></table>

DMV: Diagnosis Membership Value; MGV: Medical Gravity Value; DIV: Differential (Diagnosis) Inclusion Value; 3 raters (WFB, GSP, JEM), 64 included unique visit pairs, 130 total observations. Records were excluded if any variables were -1 or not answered.

Table 2. Stage 2 Unadjusted (all includable observations) Logistic Regression for Relationship Composite and Warrants Follow-up
<table><tr><td rowspan=1 colspan=1>Independent Variable</td><td rowspan=1 colspan=1>Warrants Follow-upOdds Ratios (95% CI)(n=180)</td><td rowspan=1 colspan=1>p</td></tr><tr><td rowspan=1 colspan=1>Relationship Composite (Greater of DIV or CIV)</td><td rowspan=1 colspan=1>1.01 (1.0006 – 1.02)</td><td rowspan=1 colspan=1>0.038</td></tr></table>

Differential (Diagnosis) Inclusion Value; CIV: Complication Inclusion Value; CI: Confidence Interval. n: number of observations included. Analysis included 2 raters (WFB and JEM), 99 included unique visit pairs, 180 total observations. Records were excluded if any variables were - 1 or not answered.

Table 3. Stage 2 Adjusted and Unadjusted (including only those included in Adjusted) Logistic Regression for Warrants Follow-up
<table><tr><td rowspan=1 colspan=1>Independent Variables</td><td rowspan=1 colspan=1>Warrants Follow-upOdds Ratios (95% CI)(n=135)</td><td rowspan=1 colspan=1>p</td></tr><tr><td rowspan=1 colspan=3>Adjusted</td></tr><tr><td rowspan=1 colspan=1>Revisit “Typical&quot; MGV</td><td rowspan=1 colspan=1>1.08 (1.05 – 1.11)</td><td rowspan=1 colspan=1>&lt;0.001</td></tr><tr><td rowspan=1 colspan=1>Relationship Composite (Greater of DIV or CIV)</td><td rowspan=1 colspan=1>1.01 (0.998 – 1.02)</td><td rowspan=1 colspan=1>0.104</td></tr><tr><td rowspan=1 colspan=3>Unadjusted</td></tr><tr><td rowspan=1 colspan=1>Relationship Composite (Greater of DIV or CIV)</td><td rowspan=1 colspan=1>1.01 (0.9990 – 1.02)</td><td rowspan=1 colspan=1>0.076</td></tr></table>

MGV: Medical Gravity Value; DIV: Differential (Diagnosis) Inclusion Value; CIV: Complication Inclusion Value; CI: Confidence Interval. n: number of observations included. Analysis included 2 raters (WFB and JEM), 75 included unique visit pairs, 135 total observations. Records were excluded if any variables were -1 or not answered.

Figure 1: KG Algorithm Overview  
![](images/e2c19bdbb0c9ec54b64c413854df71279d520f7ab7540e3e3d6fec5557d6a9d2.jpg)  
KG: Knowledge Graph; MGI: Medical Gravity Index in Knowledge Graph. As noted in the text, the Knowledge Graph was populated by open-source software, in part by repeatedly querying an LLM (GPT-4o mini) about diagnoses for the relevant relationships (e.g., complications of each diagnosis).