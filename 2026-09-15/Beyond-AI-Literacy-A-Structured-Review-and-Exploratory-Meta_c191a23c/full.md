# Beyond AI Literacy: A Structured Review and Exploratory Meta-Analysis of Measures for Competent Generative-AI Use

Daniele Verí daniele.veri.pe@gmail.com

9 September 2026

## Abstract

AI-literacy measures now include self-report questionnaires, objective tests, and instruments for critical oversight and reliance. Their diferent targets complicate assessment of competent generative-AI use in work settings. We conducted a structured, seeded evidence review anchored in the 2024 COSMIN-based review of AI-literacy scales, with a targeted update through 17 August 2026. The synthesis covers 24 focal empirical publications plus the prior review and organizes reported measurement content into four domains: knowledge and use, epistemic oversight, reliance calibration, and operational control of tool-using agents. An exploratory meta-analysis of three directly reported subjective-objective correlations, all from one research program, produced a REML pooled r = .055 with a Hartung-Knapp 95% confidence interval of [-.047, .156]. The model uses a combined reported N = 2,765; the largest study has an unresolved discrepancy between its reported correlation and p-value, so its weighting requires caution. Adding a synthetic mean of 12 cross-factor correlations from a fourth study yielded r = .079, 95% CI [-.025, .181]. This composite sensitivity addresses a broader comparison than the direct-efect analysis. The small, concentrated evidence base does not establish a population correlation or validate workplace cutofs, and provides no basis for treating self-ratings as interchangeable with performance scores. Objective instruments such as AICOS-S and GLAT assess foundation knowledge, while other instruments address verification, reliance, trust, and dependency. No validated individual-level instrument in the focal corpus tests the full combination of agent scope, permissions, recovery, state isolation, independent review, and evidence-based closure considered here; some measures cover subsets. We propose a four-layer workplace battery and non-compensatory decision rules as designs for validation. The reviewed evidence motivates separate assessment targets but does not establish the superiority of this battery or its gates.

## 1 Introduction

Competent AI use now includes decisions that earlier AI-literacy frameworks did not need to test. Those frameworks asked whether a person could recognize AI, understand broad mechanisms, use applications, evaluate outputs, and reason about ethics [Long and Magerko, 2020, Ng et al., 2021]. Those domains remain useful, but tool-using assistants add decisions about access, state changes, execution, and evidence.

A person can answer conceptual questions about large language models and still make poor operational decisions. Examples include accepting an unsupported claim because the prose sounds convincing, granting a tool more access than a task requires, allowing execution to continue after the plan has changed, repairing a corrupted session instead of returning to a known-good state, or treating an agent’s completion report as proof that required checks ran. These behaviors sit between AI literacy, critical thinking, reliance, human factors, and professional task competence. Measurement research has addressed each neighboring area in part, but the boundaries between them remain loose.

A persistent methodological split separates self-report from performance measures. Selfreport instruments ask people to judge their own capability; performance tests ask them to answer questions or complete tasks with scorable outcomes. In a systematic review completed in mid-2024, Lintner [2024] identified 16 AI-literacy scales represented by 22 validation or revalidation studies. Thirteen scales were self-report instruments and three were performancebased. Structural validity and internal consistency received the strongest support across the set, while content validity, reliability, construct validity, responsiveness, interpretability, and feasibility were tested less consistently. Cross-cultural validity and measurement error had not been examined for the included scales.

Work published in 2025 and 2026 broadened that measurement base. GLAT introduced a 20-item performance test focused on generative AI [Jin et al., 2025]. AICOS and its 12-item short form extended objective assessment to a heterogeneous adult population and included a Generative AI dimension [Markus et al., 2025, 2026]. A performance-based adult knowledge scale introduced an explicit epistemic-knowledge dimension [Klein-Avraham et al., 2026]. Other teams developed measures for critical thinking during GenAI use [Lau et al., 2026], reliance during problem solving [Hou et al., 2025], AI information practices [Alon and Levkovich, 2026], trust [McGrath et al., 2025], and dependency [Goh et al., 2025]. These instruments create a richer measurement toolkit, but they also make construct selection harder.

This paper asks four questions:

RQ1. Which validated instruments now cover the knowledge, evaluation, and use competencies usually grouped under AI or GenAI literacy?

RQ2. How well do subjective assessments of AI literacy align with objective performance measures when both are collected in the same sample?

RQ3. Which standardized instruments measure epistemic oversight and reliance, rather than literacy alone?

RQ4. Which competencies required to supervise tool-using AI agents remain outside validated individual-level measurement?

We treat the registered systematic review by Lintner [2024] as the historical baseline and update the evidence relevant to the four questions above. The quantitative synthesis is deliberately narrow: it pools only same-sample associations between perceived and objectively demonstrated AI literacy. Other evidence is synthesized by construct because the instruments difer in item format, population, latent structure, and validation target, making pooled reliability or proficiency estimates hard to interpret.

## 2 Conceptual frame

## 2.1 Four measurement domains

We organize the reviewed instruments into four domains, ordered by their proximity to operational work (Figure 1).

Knowledge and use. This domain covers recognition of AI, conceptual and procedural knowledge, application, evaluation, ethics, and GenAI-specific capabilities. It includes both perceived capability measures, such as AILS, SNAIL, and MAILS [Wang et al., 2023, Laupichler et al., 2023, Carolus et al., 2023], and performance tests such as AILIT, GLAT, AICOS, AICOS-S, SAIL4ALL, the Chiu et al. test, and the scale by Klein-Avraham et al. [Hornberger et al., 2023, Jin et al., 2025, Markus et al., 2025, 2026, Soto-Sanfiel et al., 2025, Chiu et al., 2024, Klein-Avraham et al., 2026].

Epistemic oversight. This domain concerns what users do when an AI output makes a claim: checking provenance, testing factual support, separating evidence from inference, noticing missing support, and deciding how much confidence a result deserves. The Critical Thinking in AI Use Scale directly measures verification, epistemic motivation, and reflection [Lau et al., 2026]. AILIS covers information assessment, critique, seeking, retrieval, and source identification [Alon and Levkovich, 2026]. Klein-Avraham et al. place epistemic knowledge inside an objective knowledge framework [Klein-Avraham et al., 2026].

Reliance calibration. This domain concerns how people distribute cognitive work between themselves and an AI system. Hou et al. identify reflective, cautious, thoughtless, and collaborative reliance behaviors during problem solving [Hou et al., 2025]. Trust instruments such as TIAS and S-TIAS measure a related judgment about system trustworthiness and predict willingness to rely, but trust is not itself a competence score [McGrath et al., 2025]. The Generative AI Dependency Scale describes a maladaptive outcome pattern rather than skilled reliance [Goh et al., 2025].

Operational control of tool-using agents. Once an AI system can act, a new set of observable decisions appears. A user may need to bound scope and permissions, review a plan before state changes, recognize when new evidence invalidates the plan, choose between continuing, narrowing, rewinding, restarting, or escalating, keep independent review separate from authoring, isolate parallel state, and close work with execution evidence. Human-agent teaming research ofers behavior-level metrics for preparation, execution, evaluation, adjustment, and team chemistry [Dorneich et al., 2023]. A task-oriented workplace assessment has also shown that realistic scenario performance can reveal applied AI literacy missed by generic tests [Bogart et al., 2025]. Neither source provides a validated individual placement scale for the operational behaviors listed above.

Two recent higher-education frameworks partition the same territory diferently and are worth mapping explicitly. The AI Literacy Heptagon distinguishes seven dimensions, including Integration Skills and Legal and Regulatory Knowledge [Hackl et al., 2026]; the AI and Data Acumen framework crosses seven knowledge dimensions with four proficiency levels [Kennedy and Gupta, 2025]. Our four domains are cut by measurement claim rather than by curricular content, so Integration Skills is distributed across knowledge and use and operational control depending on whether the target is understanding or execution, and Legal and Regulatory Knowledge falls under knowledge and use unless it is assessed through a decision about permitted action, in which case it belongs to operational control. These frameworks organize curricular content and measurement claims in diferent ways; their dimensions need not conflict.

These domains should not be collapsed into one continuum without evidence. A person may know a great deal about AI and still rely on it poorly. Another person may show disciplined verification while lacking detailed technical knowledge. A third may manage a tool-using agent safely through a familiar workflow without being able to explain model architecture. Training placement needs these distinctions.

## 3 Methods

## 3.1 Review design

We used a structured, seeded evidence review. The search began with the studies and instruments identified by the registered PRISMA/COSMIN review of AI-literacy scales by Lintner [2024]. That review searched Scopus and arXiv through 18 June 2024 and provides the most defensible baseline for the pre-2025 literature. We then conducted a targeted update for work published or available through 17 August 2026.

![](images/25a7ce587c4db95c496d5660f449b4a1a17f534d839aa2e0fe7c76416201075e.jpg)  
Figure 1: Four-domain framework used to organize the reviewed measures. The domains are treated as complementary measurement targets rather than interchangeable levels of one proficiency continuum. Operational control of tool-using agents is highlighted because no focal instrument directly covered the full set of scope, permission, recovery, state, review, and closure decisions.

The update searched publisher records and arXiv for combinations of the following concepts: AI literacy, generative AI literacy, objective assessment, performance-based assessment, selfreport, critical thinking, verification, reliance, trust, dependency, information literacy, workplace task assessment, and human-agent teaming. Forward and backward links from the focal papers were inspected when they pointed to measurement development or direct modality comparisons. Publisher versions were preferred over preprints when both were available.

This update was designed to answer the measurement questions above, not to reproduce a de novo database-wide PRISMA search. We therefore make no claim about a new record-count flow diagram or complete capture of every AI-literacy scale published after June 2024. The search terms and selection rules reported here describe the scope of the update; they do not constitute a reproducible database-wide search protocol. Figure 2 summarizes the review and synthesis architecture.

## 3.2 Eligibility and coding

A publication entered the focal synthesis when it met at least one of these conditions:

1. it developed, shortened, revalidated, or normed an AI- or GenAI-literacy measure;

2. it developed a validated measure for critical oversight, reliance, trust, dependency, or AI information practices that bears directly on competent GenAI use;

3. it compared subjective and objective AI-literacy measures in the same sample; or

![](images/7e3a8ec95424bc3e66e814847946504393e1f90b93d4d6d985201c3dc963b2b2.jpg)  
Structured seeded update; no new database-wide PRISMA record-count flow is claimed.  
Figure 2: Review and synthesis architecture. The 2024 PRISMA/COSMIN review supplies the systematic baseline; the present study adds a targeted update and structured coding. The synthesis separates a descriptive construct map of 24 focal empirical publications from a primary analysis of three directly reported subjective-objective correlations and a compositeefect sensitivity analysis.

## 4. it provided a measurement approach for situated workplace AI use or human-agent operational behavior.

For each paper we coded year, instrument, construct, measurement mode, item count, population, sample size, main psychometric evidence, criterion or external evidence, and relevance to the four-domain framework. The focal synthesis contains 24 empirical publications: 22 instrument or assessment reports grouped in Table 1, the paired-modality study by Zhang et al. [2026], and the oversight study by Dhanorkar et al. [2026]. We used the earlier systematic review as the historical anchor. We summarize reported psychometric evidence in the targeted update; we did not conduct a new COSMIN appraisal of these publications.

We treated self-eficacy, trust, reliance, and dependency as adjacent constructs rather than interchangeable labels for competence. T-GASE, for example, states that it assesses perceived capability for text-based GenAI use [Durak et al., 2026]. TIAS and S-TIAS quantify trust and predict intention to rely [McGrath et al., 2025]. These variables describe beliefs and judgments; neither supplies direct evidence that a person can complete a task correctly.

The coverage map represents the author’s interpretive classification of reported instrument content, factors, and assessment tasks. For this descriptive map, Strong means that a named dimension or substantial part of the instrument targets the domain; Partial means that the instrument addresses a subset of the domain or an explicit adjacent construct; Weak means that only a broad use or appraisal facet, or an isolated item, bears on the domain; and None means that we identified no operationalized content for it. These are interpretive categories, not a validated coding scale. We report no independent duplicate coding or inter-rater agreement. Coverage describes the target of measurement, not the quality of its validation, the adequacy of a cutof, or whether the response format demonstrates behavior.

## 3.3 Quantitative synthesis

We restricted the primary quantitative synthesis to directly reported same-sample correlations between subjective AI-literacy ratings and objective AI-literacy performance. Koch et al. [2024] reported $r = . 2 1$ between the MAILS total AI-literacy score and ability estimates from the Hornberger AI-literacy test in $N = 1 2 0$ German-speaking adults. Markus et al. [2025] reported $r = . 0 4$ between the 51-item AICOS objective score and MAILS subjective AI literacy in $N = 5 1 4 .$ Markus et al. [2026] reported $r = . 0 5$ between AICOS-S and the short MAILS, with an overall sample of $N = 2 { , } 1 3 1$ . All three studies come from one research program and use the MAILS family as the subjective comparator. Shared research origin limits the range of evidence; it does not by itself establish participant overlap.

For AICOS-S, the article reports $r = . 0 5$ and $p = . 1 1$ in Section 3.5, alongside the overall $N = 2 { , } 1 3 1$ in Section 2.3 [Markus et al., 2026]. An unadjusted two-sided Pearson test with that N and $r$ gives $p \approx . 0 2 1$ . The article does not specify a correlation-specific sample size or a correction that resolves this discrepancy. We retained the reported r and used the overall N for the variance, without inferring a replacement N or $r$ from the p-value. The resulting weight and combined sample size are conditional on that assumption. We report exclusion of AICOS-S as a sensitivity analysis.

Each direct correlation was transformed as

$$
z _ { i } = \operatorname { a t a n h } ( r _ { i } ) , \qquad v _ { i } = \frac { 1 } { N _ { i } - 3 } .\tag{1}
$$

We fitted a random-efects model in Fisher-z space using restricted maximum likelihood (REML) to estimate between-study variance. With $w _ { i } = 1 / ( v _ { i } + \hat { \tau } ^ { 2 } )$ and $\begin{array} { r } { \hat { \mu } = \sum _ { i } w _ { i } z _ { i } / \sum _ { i } w _ { i } } \end{array}$ , we used the unmodified Hartung-Knapp interval as the primary interval [Knapp and Hartung, 2003, Röver et al., 2015]:

$$
q = \frac { \sum _ { i } w _ { i } ( z _ { i } - \hat { \mu } ) ^ { 2 } } { k - 1 } , \qquad \hat { \mu } \pm t _ { k - 1 , 9 7 5 } \sqrt { \frac { q } { \sum _ { i } w _ { i } } } .\tag{2}
$$

We also report the normal-theory interval, $\hat { \mu } \pm 1 . 9 6 / \sqrt { \textstyle \sum _ { i } w _ { i } }$ . We transformed interval limits back to the correlation scale with tanh. We report between-study variance under both REML and Paule-Mandel, and refit the primary pool using Paule-Mandel as a sensitivity analysis [Paule and Mandel, 1982]. This matters because the estimators can disagree when the evidence base is small. The model-based prediction interval uses $\hat { \mu } \pm t _ { k - 1 , 9 7 5 } \sqrt { \hat { \tau } ^ { 2 } + 1 / \sum _ { i } w _ { i } }$ . At this $k ,$ and especially when REML reaches zero, it provides little information about transfer to a new population. We did not test funnel-plot asymmetry or publication bias.

We treated the study by Zhang et al. [2026] as a separate exploratory extension. In $N = 2 8 8$ teachers, the authors reported 12 correlations between four self-report factors and three objective factors, ranging from .07 to .24, but no total-score correlation. We averaged the Fisher-z transformations of the 12 coeficients and transformed the mean back to $r = . 1 4 6$ . This is an average cross-factor association, including comparisons between diferent content domains. It is not a correlation between total scores and does not estimate the same comparison as the three direct efects.

The 12 coeficients share participants and factors. Because their covariance matrix is unavailable, we approximated the sampling variance of the composite as

$$
v _ { \mathrm { c o m p } } = \frac { 1 } { N - 3 } \cdot \frac { 1 + ( m - 1 ) \rho } { m } ,\tag{3}
$$

where $m = 1 2$ and $\rho$ is an assumed common correlation among component estimates. We used $\rho = 1$ for the four-efect sensitivity, with .5 and .7 as alternatives. The value 1 gives the largest variance within this equal-variance, common-dependence approximation; it does not make the composite equivalent to a direct total-score correlation.

Sensitivity analyses comprise leave-one-out refits of the direct-efect pool, the Paule-Mandel refit, addition of the Zhang composite under the three $\rho$ assumptions, and leave-one-out refits of that expanded pool. A descriptive aggregation of the expanded pool reduces each research program to one inverse-variance-weighted Fisher-z mean, with variance $1 / \textstyle \sum _ { i } ( 1 / v _ { i } )$ within each program, before refitting the model across programs. This leaves two aggregate efects and is not a cluster-robust correction or evidence of independent replication. Appendix A reproduces the efect inputs, component correlations, and sensitivity results.

## 3.4 Limits of efect availability and extraction

The quantitative dataset consists of the efects extracted for this analysis, rather than all eligible efects in the literature. Reporting limitations, incomplete eligibility checks, and unfinished numeric extraction restrict its size. The targeted search identified additional candidate records, including a professional-population comparison, for which this review did not complete extraction. We do not classify those records as studies without reported coeficients.

GLAT illustrates a diferent limitation. Jin et al. [2025] administered an objective GenAI test and a self-report ChatGPT literacy scale to the same 83 participants, then regressed task performance on both with visualization literacy and baseline performance as controls. The reported standardized regression coeficient for self-reported literacy was negative and nonsignificant in that model. The reported regression output does not identify the zero-order correlation between the two literacy scores, so we did not convert it into a meta-analytic efect. Recovery of that efect would require a correlation matrix or additional data from the authors. We also kept self-eficacy comparisons outside the primary pool because self-eficacy is an adjacent construct that would require a separate analysis.

## 3.5 Why reliability coeficients were not meta-analyzed

The instrument set includes dichotomously scored knowledge tests, Likert self-reports, multidimensional batteries, short forms, and behavioral or post-task measures. Some studies report Cronbach’s alpha, others composite reliability, omega, IRT information, test-retest coeficients, factor-level reliability, or several of these. A pooled alpha across that mixture would answer no coherent measurement question. We therefore report reliability evidence at instrument level and reserve quantitative pooling for efects with a shared interpretation.

## 4 Results

## 4.1 Post-2024 instrument development has expanded objective measurement

The mid-2024 review described a field dominated by self-report: 13 self-report scales and three performance-based scales [Lintner, 2024]. The post-2024 studies in the focal update add several new performance measures, expanding the objective side of the measurement literature.

GLAT is a 20-item multiple-choice GenAI literacy test validated with 355 higher-education students. Its reported internal consistency was $\alpha = . 8 0$ and $\omega = . 8 1$ , and a 2PL IRT model showed good structural fit. GLAT scores predicted performance on GenAI-supported tasks better than perceived ChatGPT proficiency [Jin et al., 2025]. This criterion result matters because it ties the test to work performed with a generative system rather than to knowledge alone.

AICOS broadened objective measurement to adults. The 51-item instrument covers Apply AI, Create AI, Detect AI, Ethics AI, Generative AI, and Understand AI. In a sample of 514 German-speaking adults it achieved $\alpha = . 8 3$ and composite reliability of .90. Scores correlated r = .58 with another objective knowledge measure, while the correlation with MAILS subjective AI literacy was $r = . 0 4$ [Markus et al., 2025].

AICOS-S reduces the objective test to 12 items and supplies population norms from 2,131 German-speaking adults. The total score showed $\alpha = . 7 1$ and composite reliability of .83. A unidimensional CFA fit the data well $( \mathrm { C F I { = } } . 9 9 5$ , TLI=.994, RMSEA=.014, SRMR=.036). A six-factor model fit too, but the six short subscales had weak reliabilities, so the total score is the defensible interpretation. The objective convergent correlation was $r = . 5 9$ ; MAILS correlated $r = . 0 5$ [Markus et al., 2026]. The norming framework places the population median at 8 of 12 items and uses raw scores of 4 and 11 as low and high cut points associated with the 15th and 85th percentile thresholds.

Klein-Avraham et al. extend performance assessment toward epistemic knowledge. Their 26-item adult scale crosses three knowledge types, content, procedural, and epistemic, with three domains, technology, user, and society. The scale was validated with 800 internet-using adults in Israel. Epistemic and society-related knowledge were negatively associated with trust in GenAI, suggesting that more informed users did not simply become more trusting [Klein-Avraham et al., 2026].

Other objective tests remain useful in narrower populations. The Hornberger AILIT targets higher education [Hornberger et al., 2023]; Chiu et al. validated 25 multiple-choice items with 2,390 students in grades 7 through 9 using Rasch analysis [Chiu et al., 2024]; SAIL4ALL covers four broad adult knowledge themes and ofers both true/false and Likert response formats, while its multidimensional structure argues against a single overall total [Soto-Sanfiel et al., 2025].

## 4.2 Self-report remains useful when the construct is subjective

The rise of objective tests does not make self-report obsolete. It changes what a self-report score can be claimed to measure.

AILS operationalizes awareness, use, evaluation, and ethics through 12 self-report items [Wang et al., 2023]. SNAIL measures perceived technical understanding, critical appraisal, and practical application [Laupichler et al., 2023]. MAILS adds self-eficacy and self-management to AI-literacy facets and supports modular use [Carolus et al., 2023]; a 10-item short form has received further validation [Koch et al., 2024]. AILST provides a teacher-specific 36-item measure across AI perception, knowledge and skills, application and innovation, and ethics, validated with 604 respondents [Ning et al., 2025]. T-GASE narrows the target to confidence in using text-based GenAI and contains 23 items across Application, Expectations, Ethics, and Evaluation [Durak et al., 2026]. Each can support needs analysis, confidence profiling, or evaluation of perceived change.

The interpretation changes when a self-report score is used as a gate for proficiency. A respondent may be well calibrated, overconfident, or underconfident. The next section quantifies the limited alignment seen in studies that measured both modes.

## 4.3 Exploratory meta-analysis: perceived versus demonstrated AI literacy

Figure 3 separates the three direct efects from the composite-efect sensitivity. The primary pool uses a combined reported $N = 2 { , } 7 6 5$ , subject to the AICOS-S sample-size assumption described in Methods. Cochran’s $Q = 3 . 0 9 4$ on two degrees of freedom yielded $I ^ { 2 } = 3 5 . 4 \%$ REML reached the boundary, $\hat { \tau } ^ { 2 } \approx 0$ , while Paule-Mandel gave $\hat { \tau } ^ { 2 } = . 0 0 3 3$ . With three efects, this disagreement cautions against treating the dispersion estimate as stable.

The primary REML pooled correlation was $r = . 0 5 5$ , with a Hartung-Knapp 95% interval of $[ - . 0 4 7$ , .156] and a normal-theory interval of $\left[ . 0 1 8 , . 0 9 2 \right]$ . Its model-based prediction interval was [-.027, .136]; the REML boundary estimate makes this interval particularly dependent on model assumptions. Refitting with Paule-Mandel gave $r = . 0 7 2$ and a Hartung-Knapp interval of [-.113, .252]. The point estimates suggest limited alignment in these samples, while the intervals include zero.

AICOS-S carries 77.2% of the primary REML weight under the reported overall sample size. Removing it gives $r = . 1 0 7$ , with a Hartung-Knapp interval of [-.747, .828] across the two remaining studies. The other two leave-one-out point estimates are .048 and .106. These refits describe sensitivity to individual studies; their small-k intervals do not establish a stable population relationship.

Adding the Zhang cross-factor composite yields four efects and a combined reported $N =$ $3 , 0 5 3 \colon \ r \ = \ . 0 7 9$ , Hartung-Knapp 95% CI [-.025, .181], normal-theory CI [.021, .136], and model-based prediction interval [-.070, .224]. For this expanded pool, $Q = 5 . 2 9 0 , I ^ { 2 } = 4 3 . 3 \%$ $\widehat { \tau } _ { \mathrm { R E M L } } ^ { 2 } = . 0 0 1 3 .$ , and $\hat { \tau } _ { \mathrm { P M } } ^ { 2 } = . 0 0 \hat { 2 } 5$ . Varying ρ from 1 to .5 changes the point estimate from .079 to .088. This sensitivity concerns the assumed composite variance; it does not resolve the diference in what the composite measures. In the expanded pool, AICOS-S carries 47.5% of the weight, and removing it changes r from .079 to .113. Aggregating the expanded data to two research-program efects gives $r = . 0 8 4$ , with a Hartung-Knapp interval of [-.430, .557]. Appendix A lists the sensitivity results.

![](images/f937f2966d72f767c7be07781e3316bb194a11d1b2848daf4938e2b055532482.jpg)  
Figure 3: Direct subjective-objective correlations and the separate composite sensitivity. Individual direct-efect markers scale with primary REML weight; the displayed AICOS-S N is the reported overall sample size, with the weighting caveat in Methods. Individual intervals use the Fisher-z normal approximation. Diamonds show pooled Hartung-Knapp 95% intervals. Zhang’s square is the mean of 12 dependent cross-factor correlations, with $\rho = 1$ for its approximate variance; it enters only the four-efect sensitivity.

The extracted efects give no basis for treating subjective and objective scores as interchangeable in these samples. They do not establish a proficiency cutof, quantify placement errors, or show that all self-report scales measure the same construct. The workplace proposal below therefore draws on the distinction between measurement targets as well as on this limited association evidence.

The teacher study by Zhang et al. [2026] also examined profiles of mismatch. Its 12 cross-factor correlations ranged from .07 to .24, and latent profile analysis identified patterns of overestimation, underestimation, and alignment. Those profiles provide additional descriptive evidence about self-assessment; they do not turn the cross-factor mean into a total-score correlation.

## 4.4 Critical oversight has become measurable

Objective AI knowledge alone does not capture how a user scrutinizes an answer during use. Two independent instrument-development eforts targeted this distinction within a single year, giving the construct stronger empirical footing than either efort would provide alone.

The Critical Thinking in AI Use Scale measures this distinction directly. Across six studies with N = 1,341, Lau et al. [2026] developed a 13-item measure with Verification, Motivation, and Reflection factors. Criterion testing moved beyond correlations among questionnaires: higher scores predicted more frequent and varied verification strategies, greater accuracy in judging the veracity of claims in a naturalistic GPT-powered fact-checking task, and deeper reflection about responsible AI. The criterion task is what distinguishes this instrument, because it ties a self-report score to observed judgements about specific claims.

CAILS approaches a comparable construct from educational technology. Its 24 items span knowledge-related, operational, critical, and ethical dimensions and were validated with 314 firstyear student teachers after a 57-participant doctoral pilot. Reliability for the four dimensions ranged from $\alpha = . 8 3 8$ to .912 [Ranieri et al., 2025]. The critical dimension is a designed component rather than a by-product, so the claim that critical oversight has only recently become measurable would be too strong. What CAILS does not yet have is criterion evidence against observed verification behaviour, which is precisely where Lau et al. add something. For a layered battery, the two instruments provide diferent evidence: CAILS embeds the construct alongside knowledge and ethics, while the Critical Thinking scale adds a behavioral criterion.

AILIS examines similar behavior from an information-science perspective. Its 39 self-report items cover creation and processing, assessment and critique, seeking and retrieval, and ethics in a sample of 758 adults [Alon and Levkovich, 2026]. Items address missing evidence, factual and logical errors, source identification, currency, attribution, privacy, and choosing when another source or tool is more appropriate. AILIS therefore maps well to information-intensive work, although its score remains a report of one’s own practices.

Klein-Avraham et al. provide an objective counterpart by separating epistemic knowledge from content and procedural knowledge [Klein-Avraham et al., 2026]. Taken together, these measures distinguish knowledge about how AI-generated information should be interpreted from the disposition or reported practice of verifying it.

## 4.5 Reliance, trust, and dependency should remain separate

Hou et al. validated a scale for reliance behaviors during GenAI-assisted problem solving. Exploratory factor analysis used 800 responses, followed by confirmatory analyses on a holdout sample of 730 responses and 1,173 responses from a second problem-solving activity. The four factors were reflective, cautious, thoughtless, and collaborative use; overall internal consistency was α = .84 [Hou et al., 2025]. Because the scale describes behavior during problem solving, it sits closer to competent collaboration than a generic trust score.

McGrath et al. validated the 12-item Trust in Automation Scale for contemporary AI applications and developed a three-item S-TIAS [McGrath et al., 2025]. Both scales responded to manipulations of system trustworthiness, and S-TIAS predicted intention to rely. These properties make S-TIAS useful when trust calibration is the research target. A high or low trust score alone cannot establish good judgment. Appropriate reliance depends on whether trust changes with evidence about system performance and whether the user’s action matches task risk.

The Generative AI Dependency Scale targets a separate construct. Across six studies with 1,333 participants, it showed a three-factor structure covering cognitive preoccupation, negative consequences, and withdrawal, with α = .92 to .93 and test-retest ICC=.87 [Goh et al., 2025]. Dependency was associated with lower task performance and critical thinking. It can serve as an outcome or risk indicator and should remain separate from an AI-literacy total.

## 4.6 The operational-agent measurement gap

The reviewed standardized instruments cover increasingly sophisticated parts of GenAI use, yet they stop before several decisions created by tool-using agents. We found no validated individual-level scale in the focal corpus that directly tests all of the following behaviors:

• selecting the smallest permission and action surface needed for a task;

• recognizing that new evidence invalidates an approved plan and requires re-planning;

• preserving a known-good state and choosing among continuation, narrowing, rewind, restart, and escalation;

• separating maker and reviewer roles so that the same reasoning trace does not anchor both;

• isolating state when several agents work in parallel and planning reconciliation before execution;

• distinguishing an agent’s completion claim from independent execution evidence; and

• closing work with failed, skipped, and unverified checks visible.

Dorneich et al. show that preparation, execution, evaluation, adjustment, and team chemistry can be operationalized as observable human-human and human-agent teaming behaviors [Dorneich et al., 2023]. Their unit of analysis is team behavior in gameplay, not individual workplace proficiency. Bogart et al. move closer to occupational assessment: in a US Navy robotics training context, a scenario task designed to simulate job use was more informative for applied AI literacy than generic tests adopted or developed for the program [Bogart et al., 2025]. That work is currently a preprint and does not supply a general normed scale.

The list draws on the conceptual framework and reports of oversight work. Dhanorkar et al. [2026] interviewed 17 experienced developers and identified four forms of emergent oversight work: a priori control, co-planning, real-time monitoring, and post hoc review. They also document situated review dificulties and practitioner heuristics used to cope with them. Those observations motivate candidate assessment content. They do not validate the full list as an individual competence construct or establish how each behavior should be scored.

Regulatory and technical work provides related context. ISO/IEC FDIS 42105 addresses human oversight of AI systems [ISO/IEC, 2026]. Nannini et al. [2026] examine whether oversight remains commensurate with risk across multi-step agent action chains. Their question concerns the adequacy of system oversight, whereas this review concerns measurement of individual competence. The related concerns help motivate assessment development; they do not provide independent confirmation that no individual instrument exists or remove the selection limits of this review.

Work on delegation supplies vocabulary that the construct should adopt rather than reinvent. South et al. [2025] frame agent delegation around authentication, authorization, and auditability and show how natural-language permissions can be translated into auditable access-control configurations. A separate delegation framework by Tomašev et al. [2026] decomposes delegation into task allocation plus transfer of authority, responsibility, accountability, role and boundary specification, clarity of intent, and mechanisms for establishing trust. These concepts map onto the scope, permission, review, and evidence-based closure behaviors above and keep the proposed construct commensurable with technical work already under way.

The proposed construct concerns performance under operational constraints. We use agentic operational competence as a provisional label for the ability to control authority, state, recovery, review, and evidence while delegating work to a tool-using AI system. The label should be treated as a hypothesis for instrument development, not as an established latent variable.

## 5 Cross-instrument synthesis

Table 1 summarizes the author’s descriptive coverage judgments using the categories defined in Methods. These judgments distinguish content coverage from measurement mode and validation quality. A strong content rating does not imply a demonstrated skill or a validated placement decision.

Table 1: Construct coverage of focal instruments
<table><tr><td>Instrument</td><td>Primary mode</td><td>Knowledge &amp; use</td><td>Epistemic oversight</td><td>Reliance</td><td>Agent operations</td></tr><tr><td>AILS</td><td>Self-report</td><td>Strong</td><td>Partial</td><td>Weak</td><td>None</td></tr><tr><td>SNAIL</td><td>Self-report</td><td>Strong</td><td>Partial</td><td>Weak</td><td>None</td></tr><tr><td>MAILS / short MAILS</td><td>Self-report</td><td>Strong</td><td>Partial</td><td>Partial</td><td>None</td></tr><tr><td>AILIT</td><td>Performance</td><td>Strong</td><td>Partial</td><td>None</td><td>None</td></tr><tr><td>Chiu et al. AI test</td><td>Performance</td><td>Strong</td><td>Partial</td><td>None</td><td>None</td></tr><tr><td>SAIL4ALL</td><td>Performance /Likert</td><td>Strong</td><td>Partial</td><td>None</td><td>None</td></tr><tr><td>GLAT</td><td>Performance</td><td>Strong</td><td>Partial</td><td>Partial</td><td>None</td></tr><tr><td>AICOS / AICOS-S</td><td>Performance</td><td>Strong</td><td>Partial</td><td>None</td><td>None</td></tr><tr><td>Klein-Avraham et al.</td><td>Performance</td><td>Strong</td><td>Strong</td><td>Partial</td><td>None</td></tr><tr><td>T-GASE</td><td>Self-report</td><td>Strong</td><td>Partial</td><td>Partial</td><td>None</td></tr><tr><td>AILST</td><td>Self-report</td><td>Strong</td><td>Partial</td><td>Weak</td><td>None</td></tr><tr><td>FALCON-AI</td><td>Self-report</td><td>Strong</td><td>Strong</td><td>Partial</td><td>Weake</td></tr><tr><td>CT in AI Use</td><td>Self-report + task criterion</td><td>Partial</td><td>Strong</td><td>Partial</td><td>None</td></tr><tr><td>CAILS</td><td>Self-report</td><td>Strong</td><td>Strong</td><td>Partial</td><td>None</td></tr><tr><td>AILIS</td><td>Self-report</td><td>Partial</td><td>Strong</td><td>Partial</td><td>None</td></tr><tr><td>GenAI Reliance</td><td>Post-task self-report</td><td>Partial</td><td>Partial</td><td>Strong</td><td>None</td></tr><tr><td>TIAS / S-TIAS</td><td>Self-report</td><td>None</td><td>None</td><td>Partiala</td><td>None</td></tr><tr><td>GenAI Dependency</td><td>Self-report</td><td>None</td><td>Partial</td><td>Partialb</td><td>None</td></tr><tr><td>Dorneich team metrics</td><td>Observed behavior</td><td>None</td><td>Partial</td><td>Partial</td><td>Partialc</td></tr><tr><td>Bogart task assessment</td><td>Scenario performance</td><td>Strong</td><td>Strong</td><td>Partial</td><td>Partiald</td></tr></table>

Note. The 20 rows group 22 focal instrument and assessment publications; MAILS and AICOS each combine a full-form and a short-form report. Zhang et al. and Dhanorkar et al. supply the other two focal publications. Ratings describe content coverage, not psychometric quality or suitability for certification.  
<sup>a</sup> Trust predicts intention to rely, but trust is not a direct measure of appropriate reliance.  
<sup>b</sup> Dependency represents a maladaptive use pattern, not skilled reliance.  
<sup>c</sup> Measures team behaviors across human-human and human-agent teams rather than individual workplace competence.  
<sup>d</sup> Work-task alignment is strong, but the paper is a preprint and the instrument is context-specific.  
<sup>e</sup> A self-report item refers to customizing an agent. This isolated content supports a weak coverage judgment, not evidence of executing or supervising an agent task.

Several assignments require qualification. GLAT and the Klein-Avraham instrument receive partial reliance coverage because their evaluation or epistemic content bears on reliance judgments; neither directly observes reliance calibration. AILS, SNAIL, and AILST receive weak reliance coverage because their broad appraisal and use facets only touch this target. MAILS selfmanagement, information practices in AILIS, and the critical or ethical dimensions of other instruments address subsets or adjacent constructs. The table notes distinguish trust and dependency from skilled reliance, and team or workplace assessment from the full individual agent-control target.

No instrument in the focal corpus covers the full combination of four domains defined here.

Some instruments cover subsets of operational control, so the final column does not establish the absence of measures for every individual behavior. The map motivates separate assessment targets; it does not establish that combining these instruments improves workplace decisions.

Measurement mode should follow the claim being made. A self-eficacy scale is appropriate when confidence is the target. A performance test is appropriate when demonstrated knowledge is the target. A role-specific scenario is appropriate when the decision concerns execution under constraints. Problems start when one mode is asked to stand in for another.

Advanced use of tool-using AI calls for assessment at the level of decisions and traces. An answer can be scored correct while the process that produced it violated an information boundary, bypassed a check, or changed an unauthorized artifact. Operational assessment therefore needs artifacts such as source locators, permission choices, state transitions, test results, and recovery decisions in addition to a final answer.

## 6 Implications for workplace placement and training evaluation

## 6.1 A proposed layered battery

We propose a four-layer workplace assessment that keeps the measurement targets separate (Figure 4). The synthesis motivates this design but does not test its incremental validity against a single score, another battery, or a multidimensional profile. The following layers are candidate components for local validation, not a validated placement protocol.

Layer 1: objective foundation. AICOS-S is a candidate for a brief adult baseline: it has 12 objectively scored items, includes GenAI content, and has German-speaking population norms [Markus et al., 2026]. GLAT is an alternative for a GenAI-focused assessment in populations resembling its higher-education validation samples, with criterion evidence against GenAIsupported task performance [Jin et al., 2025]. A full AICOS administration covers more items when time permits, although its subscale reliability limits fine-grained diagnosis [Markus et al., 2025]. These properties do not establish workplace pass thresholds.

Layer 2: epistemic oversight. The Critical Thinking in AI Use Scale adds verification, motivation to understand, and reflection, and its criterion task links scores to actual factchecking behavior [Lau et al., 2026]. AILIS is useful when source handling and information work dominate the job [Alon and Levkovich, 2026]. Where demonstrated performance matters, a short verification task should accompany either self-report instrument.

Layer 3: reliance behavior. The GenAI Reliance Behaviors Scale can describe reflective, cautious, thoughtless, and collaborative patterns after a realistic problem-solving task [Hou et al., 2025]. S-TIAS can be added when trust calibration is itself a research question, but it should remain separate from proficiency [McGrath et al., 2025].

Layer 4: situated operational challenge. The final layer should reproduce the decisions that matter in the role. A software task might require inspection before modification, bounded permissions, plan review, test evidence, recovery from a faulty assumption, and independent review. A research task might require source authority, provenance, separation of observation from interpretation, and treatment of participant data. QA and operations tasks would use diferent action surfaces. The task-oriented assessment reported by Bogart et al. [2025] motivates role-aligned scenarios. Song et al. [2026] report initial validation of a faculty self-report instrument in 269 respondents, crossing three literacies with four work domains. Their preprint ofers an example of role-embedded measurement, while Bogart et al. assess scenario performance. Neither establishes a validated general test of the full agent-control behavior set proposed here.

Separate results could help distinguish training needs. A participant with high perceived competence and weak demonstrated knowledge might need foundation work; a participant with strong knowledge and weak verification might need diferent training. Success on questionnaires or knowledge tests would not establish success on an operational challenge involving authority or state. These are illustrative interpretations. The proposed battery has not yet shown that it improves placement accuracy or training outcomes.

![](images/6d529ab041d22d2c4c198295f7a47a3dfc2ff3272faa1565bf8e5f71aac6054d.jpg)  
Intended use after validation: placement, exemption, or targeted training

Figure 4: Proposed layered workplace assessment. The layers separate measurement targets; the gates illustrate a non-compensatory decision design for validation. No study in this synthesis tested this full battery, gate thresholds, or its superiority over alternative scoring approaches. The operational challenge requires role-specific development and validation.

## 6.2 Non-compensatory rules as a design option

A weighted total can allow strength in one domain to ofset weakness in another. A noncompensatory rule can instead require evidence for each task-relevant requirement. For example, a knowledge score would not ofset granting write access during a read-only task, and a high self-rating would not replace successful verification of a claim. These examples express design priorities; the review does not estimate how often either scoring approach would misclassify participants.

Proficiency-level matrices ofer another way to organize assessment, including the dimensions and levels described by Kennedy and Gupta [2025]. A matrix need not be compensatory: assessors can retain separate requirements for each dimension. The choice concerns the aggregation and decision rules, not the use of levels itself. The reviewed studies do not compare matrices, gates, and weighted totals for workplace placement.

A candidate gate design could require a foundation threshold, a verification threshold, and evidence of controlling scope, action, recovery, review, and closure in a situated task. Designers could specify automatic-fail behaviors for safety or integrity failures and score ordinary errors within a rubric. Before using such rules for consequential placement, researchers would need to justify the thresholds, establish scoring consistency, examine false-pass and false-fail decisions, and test whether the rules improve on simpler alternatives.

## 6.3 Candidate operational scenarios and validation needs

Advanced questions should require coordination of several principles. Vocabulary recognition is easy to coach and easy to game. A better scenario changes one premise during execution and asks what the participant does with scope, evidence, and authority. Another presents a known-good checkpoint after a wrong assumption and asks whether to repair, rewind, restart, or escalate. A reviewer scenario can test whether the participant separates fresh evaluation from the original agent’s rationale. Parallel-work scenarios can test state isolation and reconciliation.

Such items are situational judgment tasks. Their validity will depend on job analysis, expert review, scoring consistency, dificulty, and predictive evidence. Standard psychometric work remains necessary before a local challenge can be called a standardized instrument. The studies reviewed here motivate situated assessment, but do not validate this item format or a ready-made item bank for the proposed agent-control construct.

## 7 Limitations

This review has five limits.

First, the update is structured and seeded rather than a de novo systematic database review. Lintner’s 2024 review supplies the systematic baseline; our search extends it through targeted publisher and arXiv retrieval. Newly published scales outside those paths may be missing.

Second, the focal publications span general adults, university students, K-12 students, teachers, and specialized training contexts. Measurement properties do not transfer automatically across these groups. A norm from German-speaking adults cannot be applied as a workplace cutof in another country without local evidence. The primary pool covers German-speaking adults; the expanded sensitivity adds teachers. Employed adults and in-service teachers are represented in this evidence, but the studies do not validate the proposed workplace placement decisions. Extracting further professional-population comparisons would broaden the samples without establishing cross-role transfer on its own.

Third, AI-related item content ages quickly. Questions about model capabilities, prompting, or tool behavior can change dificulty as products change. Stable assessment should emphasize durable principles where possible and maintain a versioned item-review process for tool-specific content.

Fourth, all three primary efects come from one research program using the MAILS family. AICOS-S carries 77.2% of primary REML weight, conditional on using its overall reported sample size despite the unresolved $r / p$ discrepancy. The primary interval includes zero, and the heterogeneity estimators disagree. The expanded sensitivity adds a cross-factor composite that measures a diferent comparison, and its interval also includes zero. No analysis corrects for unreliability, diferences in content coverage, or range restriction. These estimates describe a small, selected evidence base and do not validate score interchangeability or placement rules. Incomplete extraction of additional candidates further limits claims about the wider literature.

Fifth, no instrument in the focal corpus tests the full combination of agent-control behaviors proposed here, although some cover subsets. The four-domain map, layered battery, gate rules, and label agentic operational competence are organizing proposals. The coverage ratings reflect one author’s interpretation, without independent duplicate coding or a new COSMIN appraisal. Researchers would need content validation, scenario development, pilot testing, reliability analysis, criterion evidence, and cross-role comparisons before using the proposed assessment for standardized certification.

## 8 Research agenda

The next measurement step should connect psychometrics to traces of real AI-assisted work.

A first study could develop a situational judgment test from critical incidents collected across software development, QA, operations, research, and information-intensive ofice work. Subject-matter experts would rate candidate responses for safety, efectiveness, and evidence quality. Think-aloud pilots could determine whether items test the intended decision rather than obscure domain knowledge.

A second study could pair the new scenario test with AICOS-S or GLAT, the Critical Thinking in AI Use Scale, and the GenAI Reliance Behaviors Scale. This design would test convergent and discriminant relations among knowledge, epistemic oversight, reliance, and operational control. If the domains are distinct, a multi-factor model should fit better than a single generic “AI competence” factor.

A third study should collect behavioral traces from a sandboxed tool-using agent. Candidate markers include permission requests, plan changes, source locators, checkpoint use, unauthorizedscope attempts, restart or rewind decisions, verification commands, reviewer separation, and completion-report accuracy. These traces could supply criterion evidence that questionnaire-only studies cannot provide.

A fourth study should test transfer. Placement instruments become useful when they predict later behavior: fewer unsupported claims, fewer unauthorized actions, more complete evidence, better recovery, and lower rework on comparable tasks. Thirty- and sixty-day samples of completed work would provide a stronger criterion than immediate post-course satisfaction.

## 9 Conclusion

The reviewed instruments assess diferent components of competent GenAI use. Objective tests such as AICOS-S and GLAT assess demonstrated foundation knowledge; self-reports assess perceptions, confidence, and reported practice. The exploratory primary analysis of three direct subjective-objective correlations yielded r = .055, with a Hartung-Knapp interval that includes zero. All three efects come from one research program, and the largest study’s weighting depends on an unresolved reporting discrepancy. Adding a cross-factor composite from another study gives r = .079, also with an interval that includes zero. These results provide no basis for substituting self-ratings for performance scores, but do not establish a population correlation or a workplace pass threshold.

Other instruments address verification, reflective oversight, reliance, trust, and dependency. No validated individual-level instrument in the focal corpus tests the full combination of scope changes, permission decisions, recovery from faulty state, state isolation, independent review, and evidence-based closure considered here. This is a corpus-bounded finding about combined coverage, not a claim that no measures address its component behaviors.

We propose a workplace battery that keeps objective literacy, epistemic oversight, reliance behavior, and situated operational performance separate. Non-compensatory rules are one candidate approach to using those results. The synthesis motivates these designs; their added value, thresholds, error rates, and transfer across roles remain empirical questions.

## Data availability

This review uses published aggregate statistics and collected no participant-level data. Appendix A reproduces the quantitative inputs and sensitivity results, including the Zhang component correlations. Table 1 reports the descriptive coverage judgments. No separate data files, analysis scripts, or supplementary verification records accompany this arXiv submission.

## References

Lilach Alon and Inbar Levkovich. Information literacy in the age of generative tools: Development and validation of the AI information literacy scale (AILIS). Computers in Human Behavior: Artificial Humans, 7:100254, 2026. doi: 10.1016/j.chbah.2026.100254.

Christopher Bogart, Aparna Warrier, Arav Agarwal, Ross Higashi, Yufan Zhang, Jesse Flot, Jaromir Savelka, Heather Burte, and Majd Sakr. AI literacy assessment revisited: A taskoriented approach aligned with real-world occupations. arXiv preprint arXiv:2511.05475, 2025.

Astrid Carolus, Martin J. Koch, Samantha Straka, Marc Erich Latoschik, and Carolin Wienrich. MAILS – meta AI literacy scale: Development and testing of an AI literacy questionnaire based on well-founded competency models and psychological change- and meta-competencies. Computers in Human Behavior: Artificial Humans, 1(2):100014, 2023. doi: 10.1016/j.chbah. 2023.100014.

Thomas K. F. Chiu, Yifan Chen, King Woon Yau, Ching-sing Chai, Helen Meng, Irwin King, and Yeung Yam. Developing and validating measures for AI literacy tests: From self-reported to objective measures. Computers and Education: Artificial Intelligence, 7:100282, 2024. doi: 10.1016/j.caeai.2024.100282.

Shipi Dhanorkar, Samir Passi, and Mihaela Vorvoreanu. Human oversight of agentic systems in practice: Examining the oversight work, challenges, and heuristics of developers using software agents. arXiv preprint arXiv:2606.05391, 2026.

Michael C. Dorneich, Stephen Gilbert, Rick F. Francis, Mitchell Talyat, and Elmin Didic. Team skill metrics that span human-human and human-agent teams: An initial assessment. Proceedings of the Human Factors and Ergonomics Society Annual Meeting, 67(1):123–130, 2023. doi: 10.1177/21695067231192892.

Gürhan Durak, Serkan Çankaya, and Semiral Öncü. A theory-driven scale for assessing text-based generative AI literacy from a self-eficacy perspective (T-GASE). Education and Information Technologies, 2026. doi: 10.1007/s10639-026-14023-y.

Adalia Y. H. Goh, Andree Hartanto, and Nadyanna M. Majeed. Generative artificial intelligence dependency: Scale development, validation, and its motivational, behavioral, and psychological correlates. Computers in Human Behavior Reports, 20:100845, 2025. doi: 10.1016/j.chbr.2025. 100845.

Veronika Hackl, Alexandra Elena Müller, and Maximilian Sailer. The AI literacy heptagon: A structured approach to AI literacy in higher education. Computers and Education: Artificial Intelligence, 10:100540, 2026. doi: 10.1016/j.caeai.2026.100540.

Marie Hornberger, Arne Bewersdorf, and Claudia Nerdel. What do university students know about artificial intelligence? development and validation of an AI literacy test. Computers and Education: Artificial Intelligence, 5:100165, 2023. doi: 10.1016/j.caeai.2023.100165.

Chenyu Hou, Gaoxia Zhu, Vidya Sudarshan, Fun Siong Lim, and Yew Soon Ong. Measuring undergraduate students’ reliance on generative AI during problem-solving: Scale development and validation. Computers & Education, 234:105329, 2025. doi: 10.1016/j.compedu.2025. 105329.

ISO/IEC. FDIS 42105: Information technology — artificial intelligence — guidance for human oversight of AI systems, 2026. URL https://www.iso.org/standard/86902.html. Final Draft International Standard; stage 50.00 as of 17 August 2026.

Yueqiao Jin, Roberto Martinez-Maldonado, Dragan Gašević, and Lixiang Yan. GLAT: The generative AI literacy assessment test. Computers and Education: Artificial Intelligence, 9: 100436, 2025. doi: 10.1016/j.caeai.2025.100436.

Kathleen Kennedy and Anuj Gupta. AI & data competencies: Scafolding holistic AI literacy in higher education. Thresholds in Education, 48(2):182–201, 2025.

Inbal Klein-Avraham, Rut Ston, Osnat Atias, Ido Roll, and Ayelet Baram-Tsabari. Measuring diferent types and domains of AI knowledge: Developing and validating a performance-based scale. Computers & Education, 247:105573, 2026. doi: 10.1016/j.compedu.2026.105573.

Guido Knapp and Joachim Hartung. Improved tests for a random efects meta-regression with a single covariate. Statistics in Medicine, 22(17):2693–2710, 2003. doi: 10.1002/sim.1482.

Martin J. Koch, Astrid Carolus, Carolin Wienrich, and Marc E. Latoschik. Meta AI literacy scale: Further validation and development of a short version. Heliyon, 10(21):e39686, 2024. doi: 10.1016/j.heliyon.2024.e39686.

Gabriel R. Lau, Wei Yan Low, Louis Tay, Ysabel Thereze Ang Guevarra, Dragan Gašević, and Andree Hartanto. Understanding critical thinking in generative artificial intelligence use: Development, validation, and correlates of the critical thinking in AI use scale. Computers in Human Behavior Reports, 22:101103, 2026. doi: 10.1016/j.chbr.2026.101103.

Matthias Carl Laupichler, Alexandra Aster, Nicolas Haverkamp, and Tobias Raupach. Development of the “scale for the assessment of non-experts’ AI literacy” – an exploratory factor analysis. Computers in Human Behavior Reports, 12:100338, 2023. doi: 10.1016/j.chbr.2023.100338.

Tomáš Lintner. A systematic review of AI literacy scales. npj Science of Learning, 9:50, 2024. doi: 10.1038/s41539-024-00264-4.

Duri Long and Brian Magerko. What is AI literacy? competencies and design considerations. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems, pages 1–16, 2020. doi: 10.1145/3313831.3376727.

André Markus, Astrid Carolus, and Carolin Wienrich. Objective measurement of AI literacy: Development and validation of the AI competency objective scale (AICOS). Computers and Education: Artificial Intelligence, 9:100485, 2025. doi: 10.1016/j.caeai.2025.100485.

André Markus, Astrid Carolus, and Carolin Wienrich. How to measure AI literacy (fast): Development and norming of the AI competence objective scale short version (AICOS-S). Computers in Human Behavior: Artificial Humans, 8:100298, 2026. doi: 10.1016/j.chbah.2026. 100298.

Melanie J. McGrath, Oliver Lack, James Tisch, and Andreas Duenser. Measuring trust in artificial intelligence: validation of an established scale and its short form. Frontiers in Artificial Intelligence, 8:1582880, 2025. doi: 10.3389/frai.2025.1582880.

Luca Nannini, Adam Leon Smith, Michele Joshua Maggini, Enrico Panai, Sandra Feliciano, Aleksandr Tiulkanov, Elena Maran, James Gealy, and Piercosma Bisconti. AI agents under EU law: A compliance architecture for AI providers. arXiv preprint arXiv:2604.04604, 2026.

Davy Tsz Kit Ng, Jac Ka Lok Leung, Kai Wah Samuel Chu, and Maggie Shen Qiao. Conceptualizing AI literacy: An exploratory review. Computers and Education: Artificial Intelligence, 2:100041, 2021. doi: 10.1016/j.caeai.2021.100041.

Yimin Ning, Wenjun Zhang, Dengming Yao, Bowen Fang, Binyan Xu, and Tommy Tanu Wijaya. Development and validation of the artificial intelligence literacy scale for teachers (AILST). Education and Information Technologies, 30(12):17769–17803, 2025. doi: 10.1007/ s10639-025-13347-5.

Robert C. Paule and John Mandel. Consensus values and weighting factors. Journal of Research of the National Bureau of Standards, 87(5):377–385, 1982. doi: 10.6028/jres.087.022.

Maria Ranieri, Gabriele Biagini, and Stefano Cuomo. AI literacy in higher education: A systematic approach to questionnaire development and validation. International Journal of Digital Literacy and Digital Competence, 16(1):1–25, 2025. doi: 10.4018/IJDLDC.388469.

Christian Röver, Guido Knapp, and Tim Friede. Hartung-knapp-sidik-jonkman approach and its modification for random-efects meta-analysis with few studies. BMC Medical Research Methodology, 15:99, 2015. doi: 10.1186/s12874-015-0091-1.

Yukyeong Song, Hyunjoo Moon, Hyewon Yang, and Chris Kilgore. Development and validation of a faculty artificial intelligence literacy and competency (FALCON-AI) scale for higher education. arXiv preprint arXiv:2603.20220, 2026.

María T. Soto-Sanfiel, Ariadna Angulo-Brunet, and Christoph Lutz. The scale of artificial intelligence literacy for all (SAIL4ALL): assessing knowledge of artificial intelligence in all adult populations. Humanities and Social Sciences Communications, 12:1618, 2025. doi: 10.1057/s41599-025-05978-3.

Tobin South, Samuele Marro, Thomas Hardjono, Robert Mahari, Cedric Deslandes Whitney, Dazza Greenwood, Alan Chan, and Alex Pentland. Authenticated delegation and authorized AI agents. arXiv preprint arXiv:2501.09674, 2025. doi: 10.48550/arXiv.2501.09674.

Nenad Tomašev, Matija Franklin, and Simon Osindero. Intelligent AI delegation. arXiv preprint arXiv:2602.11865, 2026. doi: 10.48550/arXiv.2602.11865.

Bingcheng Wang, Pei-Luen Patrick Rau, and Tianyi Yuan. Measuring user competence in using artificial intelligence: validity and reliability of artificial intelligence literacy scale. Behaviour & Information Technology, 42(9):1324–1337, 2023. doi: 10.1080/0144929X.2022.2072768.

Shan Zhang, Ruiwei Xiao, Anthony F. Botelho, Guanze Liao, Thomas K. F. Chiu, John Stamper, and Kenneth R. Koedinger. How to assess AI literacy: Misalignment between self-reported and objective-based measures. In Proceedings of the 16th International Learning Analytics and Knowledge Conference (LAK 2026), 2026. doi: 10.1145/3785022.3785088. arXiv:2601.06101.

## A Quantitative inputs and sensitivity results

Table 2 lists the three direct efects used in the primary analysis and the composite used in the expanded sensitivity. Table 3 reproduces its components. All correlations are uncorrected for measurement error. The reported sample totals assume that each study’s stated N applies to the extracted coeficient; the AICOS-S caveat below requires particular attention.

Table 2: Study-level inputs and analytic roles
<table><tr><td>Study</td><td>N used</td><td>r</td><td>Definition and role</td></tr><tr><td>Koch et al. (2024), Study III</td><td>120</td><td>.210</td><td>MAILS literacy total vs. Hornberger test ability estimate; primary</td></tr><tr><td>Markus et al. (2025), AICOS</td><td>514</td><td>.040</td><td>AICOS total vs. subjective MAILS; primary</td></tr><tr><td>Markus et al. (2026), AICOS-S</td><td>2,131ª</td><td>.050</td><td>AICOS-S total vs. short MAILS; primary</td></tr><tr><td>Zhang et al. (2026)</td><td>288</td><td>.146</td><td>Mean Fisher-z of 12 cross-factor</td></tr><tr><td></td><td></td><td></td><td>correlations; sensitivity only</td></tr></table>

<sup>a</sup> Overall study N, not a separately confirmed pairwise sample size. The source reports $r = . 0 5 , p = . 1 1 ;$ an unadjusted Pearson test with $N = 2 { , } 1 3 1$ gives p ≈ .021. We preserve the reported correlation and disclose the unresolved discrepancy rather than imputing a replacement value. Exclusion sensitivities appear in Table 4.

Table 3: Zhang et al. (2026): component correlations used in the sensitivity composite
<table><tr><td>Self-report factor</td><td>Objective conceptual understanding</td><td>Objective capability evaluation</td><td>Objective practical and ethical use</td></tr><tr><td>Concept</td><td>.24</td><td>.14</td><td>.14</td></tr><tr><td>Ethics</td><td>.23</td><td>.17</td><td>.11</td></tr><tr><td>Evaluate</td><td>.17</td><td>.10</td><td>.07</td></tr><tr><td>Use</td><td>.18</td><td>.10</td><td>.10</td></tr></table>

Note. All 12 coeficients share the same N = 288. Their Fisher-z mean back-transforms to r = .146252. The composite is an average cross-factor association, not a total-score correlation. Its approximate variance is $1 / 2 8 5 = . 0 0 3 5 0 9$ at $\rho = 1 ,$ .002544 at $\rho = . 7 ,$ and .001901 at $\rho = . 5 .$

Table 4: Primary model and sensitivity analyses
<table><tr><td>Analysis</td><td>k</td><td>Pooled r</td><td>Hartung-Knapp 95% CI</td></tr><tr><td>Primary: three direct effects, REML</td><td>3</td><td>.055</td><td>[-.047, .156]</td></tr><tr><td>Primary with Paule-Mandel variance</td><td>3</td><td>.072</td><td>[-.113, .252]</td></tr><tr><td>Direct effects without Koch et al.</td><td>2</td><td>.048</td><td>[-.002, .098]</td></tr><tr><td>Direct effects without AICOS</td><td>2</td><td>.106</td><td>[-.707, .798]</td></tr><tr><td>Direct effects without AICOS-S</td><td>2</td><td>.107</td><td>[-.747, .828]</td></tr><tr><td>Expanded: add Zhang composite,  $\rho = 1$ </td><td>4</td><td>.079</td><td>[-.025, .181]</td></tr><tr><td>Expanded,  $\rho = . 7$ </td><td>4</td><td>.085</td><td>[-.024, .191]</td></tr><tr><td>Expanded,  $\rho = . 5$ </td><td>4</td><td>.088</td><td>[-.022, .196]</td></tr><tr><td>Expanded without Koch et al.</td><td>3</td><td>.058</td><td>[-.032, .147]</td></tr><tr><td>Expanded without AICOS</td><td>3</td><td>.109</td><td>[-.088, .298]</td></tr><tr><td>Expanded without AICOS-S</td><td>3</td><td>.113</td><td>[-.098, .314]</td></tr><tr><td>Expanded without Zhang</td><td>3</td><td>.055</td><td>[-.047, .156]</td></tr><tr><td>Expanded, aggregated by research program</td><td>2</td><td>.084</td><td>[-.430, .557]</td></tr></table>

Note. All models use REML except the Paule-Mandel row. Leave-one-out analyses of the expanded pool use $\rho = 1 .$ . Primary reported $N = 2 { , } 7 6 5 ;$ expanded reported $N = 3 { , } 0 5 3$ . Normal-theory intervals are [.018, .092] for the primary REML model and [.021, .136] for the expanded model. Their model-based prediction intervals are [-.027, .136] and [-.070, .224]. The primary REML heterogeneity estimate is at zero; these intervals should not be treated as reliable coverage statements for new workplace populations. The program aggregation is descriptive, not a cluster-robust adjustment.

## B Proposed interpretive rules for a layered workplace battery

The following design rules motivate the proposed battery. Their use in placement requires validation of the instruments, scoring, and decision thresholds in the target setting:

1. Use an objective test when claiming demonstrated foundation knowledge.

2. Use self-report when confidence, perceived capability, trust, dependency, or reported practice is the intended construct.

3. Add an ecologically grounded verification task when oversight of AI-generated claims matters.

4. Keep trust and reliance separate. Trust can explain a reliance decision without proving that the decision was appropriate.

5. Use a situated performance task for authority, state, recovery, independent review, and closure.

6. Avoid one compensatory total across domains until empirical work demonstrates that such a total has a stable meaning.