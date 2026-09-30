This is the author’s version of the article that has been accepted for publication in IEEE ISSE 2026. The final version will be available at IEEE Xplore (DOI to be added once available).

© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

# Cross-Organizational SysML Model Integration: A Survey of Challenges and AI-Supported Tasks

1<sup>st</sup> Li Zirui

2<sup>nd</sup> Brix Torsten

3<sup>rd</sup> Husung Stephan

Product and Systems Engineering Group

Product and Systems Engineering Group

Technische Universitat Ilmenau¨

Product and Systems Engineering Group

Technische Universitat Ilmenau¨

Ilmenau, Germany

Technische Universitat Ilmenau¨

Ilmenau, Germany

https://orcid.org/0009-0007-7983-7901

torsten.brix@tu-ilmenau.de

Ilmenau, Germany

https://orcid.org/0000-0003-0131-5664

Abstract—Cross-organizational collaboration is widely regarded as a key promise of SysML-based Model-Based Systems Engineering (MBSE), yet practitioners still face persistent challenges when exchanging and integrating system models. In parallel, Large Language Models (LLMs) raise expectations for AI-assisted model understanding and integration, while reliability and required human oversight continue to pose challenges. This paper reports the results of an online questionnaire survey with 29 MBSE stakeholders involved in cross-organizational collaboration. Respondents rated eight predefined integration challenge categories and six AI-supported task types on five-point Likert scales. The results indicate that stakeholders perceive model integration as a multi-dimensional alignment problem across semantics, behavior, traceability, and exchange interoperability. These perceptions vary by organizational role and frequency of integration involvement. AI is rated highly useful for analysis tasks such as semantic structure analysis and inconsistency detection, and respondents predominantly prefer human-in-theloop use with mandatory verification. These findings motivate AI support that enhances, rather than replaces, engineering responsibility in SysML-based integration.

Index Terms—Model-Based Systems Engineering (MBSE), SysML, Cross-organizational Collaboration, Model Integration, Questionnaire Survey, Large Language Models (LLMs), AI assistance

## I. INTRODUCTION

Cross-organizational collaboration in Model-Based Systems Engineering (MBSE) is becoming increasingly important, yet the exchange and integration of SysML-based models across company and tool boundaries remains a challenge due to heterogeneous modeling methods, tool ecosystems, and domainspecific practices among partners [1]–[3].

In this context, System Modeling Language v2 (SysML v2) provides new technical capabilities for collaborative MBSE through formally defined semantics, semantic consistency between textual and graphical representations, and standardized REST API [4], [5]. In parallel, recent work explores Large Language Models (LLMs)-based AI-assistance capabilities, such as model generation, model comprehension, and semantic conflict detection [6]. However, the inherent uncertainty and limited explainability of LLM outputs pose challenges to their adoption in rigorous engineering practice [7], [8].

Empirical studies on the introduction of Systems Engineering in industrial contexts have identified persistent organizational, methodological, and tooling-related barriers that hinder effective adoption [9]. Furthermore, recent research on collaborative MBSE and model integration has made progress through survey-based investigations of MBSE stakeholders perspectives alongside literature-based and conceptual analyses [1], [10]. However, the perspectives of MBSE practitioners on cross-organizational model exchange and integration, as well as the role of emerging technologies such as AI, remain insufficiently explored [11]. Based on this motivation, the following research questions are formulated.

• RQ1: How do MBSE stakeholders perceive the severity of various challenge categories in the cross-organizational exchange and integration of SysML-based system models, and how do these perceptions relate to organizational roles and frequency of involvement in system model integration activities?

• RQ2: How do MBSE stakeholders perceive the usefulness of AI-supported integration tasks, and which human-AI collaboration modes do they prefer?

To address the above research questions, this study designed and conducted a questionnaire survey targeting MBSE stakeholders, with a focus on model exchange and integration in cross-organizational collaboration scenarios. The survey collected responses from professionals representing different roles, including OEMs, suppliers, tool providers or consultants, and research institutions. Respondents provided their assessments using a five-point Likert scale and complemented their responses with open-ended questions to share practical experiences and expectations.

The remainder of this paper reviews related work (Section II), describes the survey design and analysis (Section III), reports the results (Section IV), and discusses implications and limitations (Section V).

## II. STATE OF THE ART

## A. Model-Based Systems Engineering

INCOSE Systems Engineering Vision 2035 highlights a shift toward model-centric digital engineering to address increasing system complexity, tight hardware–software integration [12]. This shift positions models as lifecycle-wide information carriers that can support structured and traceable engineering information exchange across disciplines and organizational boundaries [13].

In line with this vision, MBSE should support the early integration of models, facilitate cross-disciplinary collaboration, and improve development efficiency through model reuse [14]. However, MBSE adoption in many organizations remains limited to local project or product levels, while enterprise-wide or cross-organizational integration is rarely achieved [1].

From the perspective of MBSE system model languages, SysML v1 provides a unified means to represent requirements, structure, behavior, and constraints, and has been widely implemented in many modeling tools [15]. As its application has expanded to multidisciplinary and cross-organizational collaboration scenarios, however, several limitations have become increasingly apparent. On the one hand, SysML v1 is defined as a UML profile, which results in limited semantic precision. This has increased the difficulty of standardized usage and automated analysis [16]. On the other hand, practical MBSE adoption is often characterized by organization- or project-specific methodologies and profile extensions [1], [17]. Differences among collaborating parties in abstraction levels further amplify the challenges of model alignment and reuse [10].

From an integration perspective, the profile-based nature of SysML v1 fosters variability in interpretation and tool-specific representation, which increases the likelihood of semantic drift when models are exchanged between organizations. This drift typically manifests as naming inconsistencies, mismatched abstraction levels, and diverging structural decompositions, thereby increasing manual reconciliation effort during integration. These recurring dimensions were used as a conceptual basis to structure the integration challenge categories assessed in the present survey.

Moreover, although model exchange mechanisms such as XMI formats have been established, existing surveys on model and data exchange indicate that tool-specific XMI implementations and the handling of additional metadata still require complex and context-dependent transformation and maintenance efforts when exchanging models across tools [18].

Against this background, SysML v2 introduces a KerMLbased semantic foundation that provides precise definitions of elements, relationships, and constraints [5]. This supports improved consistency checking, model transformation, and formal analysis in comparison to profile-based approaches. This not only creates the basis for cross-tool transformation, configuration management, and cross-company collaboration in data-space-oriented scenarios, but also opens up new possibilities for integrating AI technologies into MBSE environments [10], [19], [20].

## B. Cross-company collaboration scenario in MBSE

In concrete cross-company collaboration scenarios, the OEM typically performs system decomposition and defines subsystem boundaries within its internal SysML models, resulting in black-box specifications that capture functional, structural, and interface aspects. Subsequently, subsystemrelevant model fragments are extracted from the global system model and delivered to suppliers using agreed exchange formats. Based on these specifications, suppliers carry out detailed design and implementation within their own methodologies and tool-chains, producing model-based representations of the subsystem that are then returned to the OEM for integration into the global system model, verification, and iterative refinement. [2], [3], [21]

Around this typical OEM–supplier workflow, both research and industrial practice have identified several classes of structural challenges. The use of a common modeling language alone is insufficient to ensure consistent understanding across organizational boundaries [22]. Differences in abstraction levels, naming conventions among collaborating organizations can lead to divergent interpretations of the same model [23]. Furthermore, effective collaboration often requires explicit upfront agreements on modeling methods and interface specifications to achieve system integration [24]. This OEM–supplier interaction highlights several structurally distinct integration layers such as model element alignment, structural & behavioral consistency, and technical interoperability of exchange formats and tool infrastructures. Each of these layers may introduce independent sources of integration problems.

## C. AI in MBSE

In recent years, research has increasingly explored the use of AI technologies, particularly LLMs, to support MBSE activities [12]. Proposed approaches include LLM-based assistants leveraging domain ontologies or the SysML metamodel to support requirements formulation, modeling guidance, and consistency checking, as well as the integration of LLMs with system models via the SysML v2 API to enable naturallanguage-driven model querying, modification, and the derivation of modeling elements [6], [7], [25].

In the context of model integration, AI applications can be differentiated into analysis support functions (e.g., model comprehension, semantic comparison, inconsistency detection), generative assistance (e.g., model element generation or transformation), and autonomous integration decisions [26]. The acceptance of these categories may differ depending on accountability structures and contractual responsibility in cross-organizational engineering settings. This differentiation informed the selection and formulation of the six AI-supported task types evaluated in this study, covering primarily analysis and assistive functions as well as more generative forms of support.

While LLMs demonstrate capabilities in generating engineering artifacts, they also exhibit notable limitations, including data quality, semantic deviations, hallucinated content, and a lack of transparency [27], [28]. These observations suggest that, within MBSE contexts, AI is more appropriately positioned as a support tool requiring review rather than as a fully autonomous agent [8], [29].

## III. SURVEY DESIGN AND DATA ANALYSIS METHOD

In general, the state of the art highlights the technical diversity of MBSE approaches and modeling practices in cross-company collaboration scenarios, as well as the growing interest in AI-supported assistance. However, how these heterogeneous MBSE methods, tool ecosystems, and modeling practices and emerging capabilities are reflected in practical model exchange and integration remains unclear. To address this gap, an empirical questionnaire-based investigation was conducted to capture current practices and perceptions in SysML-based system model exchange and integration.

## A. Data collection approach and Survey Design

The study is based on an online questionnaire to collect data. At the beginning of the survey, respondents were informed of the research purpose and anonymity, and the collected data were used for academic research and may be included in anonymized form in research publications.

The questionnaire was structured into three parts. The first part collected background information on respondents’ organizational roles, modeling languages and methodologies in use, and their level of involvement in cross-organizational model exchange and integration activities. The second and third parts addressed model-exchange challenges (RQ1) and AIsupported tasks and collaboration modes (RQ2), respectively.

Survey items were derived from recurring challenge aspects and AI-related themes identified in the literature on MBSE collaboration, model integration, and AI-assisted Systems Engineering. Responses were collected using a combination of five-point Likert-scale items, multiple-choice questions, and a limited number of open-ended questions to capture additional practical insights.

The questionnaire was distributed via academic and industrial Systems Engineering communities primarily between May and July 2025 and yielded 30 responses, with 29 retained after response consistency and relevance checks. Detailed respondent backgrounds are summarized in Section IV-A. The original survey is available at [30].

With regard to validity, the questionnaire items were based on the OEM–supplier collaboration processes and model integration issues identified during the state of the art analysis presented in the section II, which supports content validity. In addition, each category of challenges was accompanied by a brief illustrative example to help ensure a consistent understanding of the questions among respondents. Responses to the open-ended questions further complemented and corroborated the Likert-scale results by providing contextualized explanations and additional insights.

## B. Data Analysis Methods

Generally, descriptive statistics (means, medians, and proportions of high ratings) were computed to characterize overall tendencies and perceived importance for Likert-scale items. To explore differences in perceptions across respondent backgrounds, descriptive subgroup comparisons were conducted based on organizational role and frequency of involvement in model exchange and integration activities. Given the limited sample size and the ordinal data, these analyses focused on descriptive comparisons, examining differences in average ratings and distribution patterns across groups. The results are therefore interpreted in an exploratory manner and are not intended to support statistical inference or causal claims.

Responses to open-ended questions were analyzed using a systematic coding procedure based on Mayring’s qualitative content analysis approach [31]. An initial category framework was derived from the main model integration challenge aspects and AI application types identified in the literature review and applied to a subset of responses through pilot coding. Based on recurring and representative content, additional subcategories were inductively refined. The finalized coding scheme was then applied to all open-ended responses to identify practical challenges, application expectations, and risk concerns that the standardized questionnaire did not fully capture. All qualitative findings are reported in aggregated, anonymized form.

## IV. SURVEY RESULT

This section presents the results of the questionnaire survey. First, it reports MBSE stakeholders’ assessments of the most prominent challenges encountered in SysML-based system model exchange and integration. Next, it presents findings on the perceived usefulness of AI-supported tasks, differences by engagement frequency and organizational role. Finally, it summarizes themes derived from open-ended responses.

## A. Respondent background and modeling context

The survey collected responses from 29 MBSE stakeholders with diverse organizational and functional backgrounds. In terms of organizational perspective, the sample includes 8 respondents with OEM-level system responsibility, 8 component or subsystem suppliers, 11 tool providers or consultants, and 6 academic researchers with several respondents reporting experience in more than one role.

Regarding functional responsibilities, respondents cover a broad range of roles along the Systems Engineering lifecycle, including system architects, requirements engineers, model developers, integration engineers, process or methodology specialists, as well as project managers and coordinators.

With respect to modeling languages, SysML v1.x is used by the large majority of respondents, while approximately half report experience with SysML v2. In addition, several respondents interact with UML, BPMN, ArchiMate, UAF, Capella, and company-specific modeling languages. A similar diversity is observed for modeling methodologies, including well-known approaches [17] (e.g., MagicGrid, HarmonySE, OOSEM, ARCADIA, SPES) and company-specific variants.

Overall, the background data show that the survey captures practitioner perspectives from heterogeneous organizational settings characterized by mixed roles, multi-language toolchains, and methodologically diverse MBSE practices, providing an appropriate context for the subsequent analysis of model integration challenges and attitudes toward AI-assisted integration.

## B. Challenges rating in model exchange and integration

Table I summarizes respondents’ ratings of eight categories of challenges (A–H) encountered during cross-organizational

TABLE II  
TABLE I MODEL INTEGRATION CHALLENGES
<table><tr><td>Catory abel</td><td></td><td>Meian</td><td>4-tio tio</td><td>Mean</td></tr><tr><td>A B</td><td>Naming and Semantic Conflicts Structural and Abstraction Challenges</td><td>4 4</td><td>65.5% 51.7%</td><td>3.97 3.55</td></tr><tr><td>C</td><td>Interface and Port Integration Issues</td><td>3</td><td>34.5%</td><td>3.24</td></tr><tr><td>D</td><td>Behavioral and Logical Misalignment</td><td>4</td><td>58.6%</td><td>3.55</td></tr><tr><td>E</td><td>Methodology and Profile Integration</td><td>3</td><td>35.7%</td><td>3.11</td></tr><tr><td>F</td><td>Traceability Gaps</td><td>4</td><td>58.6%</td><td>3.90</td></tr><tr><td>G</td><td>Format Gaps (e.g., XMI vs JSON)</td><td></td><td>60.7%</td><td></td></tr><tr><td>H</td><td>Incomplete/Incorrect Metadata</td><td>43</td><td>39.3%</td><td>3.68 3.25</td></tr></table>

SysML model exchange and integration. All items were rated using a five-point Likert scale (1 = Not an issue, 5 = Very significant issue). For each challenge, the median, the proportion of high ratings (scores of 4 or 5), and the mean value are reported.

Naming and semantic conflicts (A), behavioral and logical inconsistencies (D), missing traceability relationships (F), and differences in model formats (G) consistently receive high ratings, with medians of 4. For these aspects, roughly 60% of respondents assign scores of 4 or 5, with mean values between 3.55 and 3.97, indicating that they are widely perceived as prominent integration issues. Structural and abstraction challenges (B) also show a median of 4, but with a lower proportion of high rating (51.7%) and a mean of 3.55.

In contrast, interface and port integration issues (C), methodological and SysML profile integration (E) and metadata-related issues (H) show medians of 3. For these aspects, the proportion of high ratings ranges from 34.5% to 39.3%, with mean values approximately between 3.11 and 3.25, reflecting a more dispersed distribution of responses.

Overall, the results suggest that challenges in crossorganizational SysML system model integration are not confined to a single technical aspect, but span multiple aspects including semantics, consistency, formats, and methodologies. However, the degree of consensus among respondents varies substantially across these challenges.

## C. AI support usefulness and oversight preferences

Table II summarizes respondents’ assessments of the potential usefulness of AI across six model exchange and integration–related tasks derived from challenges in last section. All tasks were assessed using a five-point Likert scale (1 = Not useful, 5 = Very useful), and the results are reported in terms of median values, the proportion of high ratings (scores of 4 or 5), and mean scores.

Tasks related to model understanding and semantic analysis received the most consistently high ratings. In particular, semantic structure analysis support (F) and identification of semantic conflicts or inconsistencies (C) both show a median rating of 5, with high-rating proportions above 80%. Similarly, tasks associated with model content comprehension (A) and model element mapping suggestions (B) were also rated highly. Both tasks exhibit high-rating proportions above 75%, with mean scores of 4.28 and 4.21. By contrast, generation of integration workflows and guidance (D) and crossmethodology transformation suggestions (E) are still viewed positively but receive lower overall ratings, with medians of 4 and noticeably lower proportions of high ratings.

AI-SUPPORTED MODEL UNDERSTANDING AND INTEGRATION TASKS
<table><tr><td>Label</td><td>Category</td><td>Median</td><td>4-5 Ratio</td><td>Mean</td></tr><tr><td>A</td><td>Understand model content from textual or structural input</td><td>5</td><td>75.9%</td><td>4.28</td></tr><tr><td>B</td><td>Suggest mappings between model elements</td><td>4</td><td>79.3%</td><td>4.21</td></tr><tr><td>C</td><td>Identify semantic conflicts or inconsistencies</td><td>5</td><td>82.8%</td><td>4.34</td></tr><tr><td>D</td><td>Generate integration workflows and prompts</td><td>4</td><td>65.5%</td><td>3.72</td></tr><tr><td>E</td><td>Suggest cross-methodology</td><td>4</td><td>55.2%</td><td>3.55</td></tr><tr><td>F</td><td>transformations Support semantic structure analysis</td><td>5</td><td>89.7%</td><td>4.45</td></tr></table>

Overall, the results indicate a strong consensus on the potential value of AI for analysis, interpretive, and conflictdetection tasks. However, generative tasks, such as creating integration workflows or cross-methodology transformations are perceived as less beneficial overall, though the assessments remain positive.

Regarding acceptable levels of human oversight in AIassisted tasks, majority of respondents (83%) preferred a semiautomatic mode in which AI provides support with mandatory human verification. A further 17% favored a more conservative setup in which AI is limited to making suggestions, with the primary work remaining manual.

Notably, no respondents selected either of the two extremes: manual workflows without AI support or highly automated modes with optimal review. This indicates a clear preference for human-led integration with AI in an assistive role.

## D. Engagement-Dependent Differences in Challenge Perception

Figure 1 compares respondents’ ratings of the eight integration challenges under different frequency of engagement in model exchange and integration activities. In this survey, ”engagement” refers to the self-reported frequency of involvement in cross-organizational model exchange and integration. 14 respondents who selected ”frequently” or ”occasionally” are grouped as high engagement, while 15 respondents who selected “rarely” or “never” are grouped as low engagement. Ratings are reported as group-level mean values.

Across challenges, respondents with higher involvement tend to assign higher importance to behavioral misalignment (D), format gaps (G) and metadata issues (H). Differences in perceived difficulty for structural abstraction (B) and methodology or profile integration (E) remain small between high- and low-engagement groups. For naming and semantic conflicts (A), Interface and Port Integration Issues (C) and traceability gaps (F), ratings are slightly higher in the lowengagement groups.

![](images/35f7494e0a04ab2ba273c5fa22035dfa995cd60d3f514d2cdf12568385070b52.jpg)  
Fig. 1. Perceived challenge severity by cross-organizational engagement level

Overall, differences between engagement-frequency groups are more evident for challenges related to model dynamic consistency and exchange artifacts. Meanwhile, ratings for most basic modeling challenges remain relatively similar across groups.

## E. Role-Specific Differences in Challenge Perception

Figure 2 compares how different organizational roles (OEMs, suppliers, tool vendors, and researchers) perceive the challenges A–H in collaborative SysML-based system model exchange and integration. Since the organizational role was collected as a multi-select item, some respondents reported experience in more than one role (e.g., OEM and researcher). As a result, the role-based groups partially overlap. The results should therefore be interpreted as ratings of respondents with experience in a given role, rather than as comparisons between mutually exclusive groups. In the final sample, 8 respondents reported OEM, 8 suppliers, 11 tool vendors or consultants, and 6 researchers. The results indicate role-related differences in the perceived importance of several challenge aspects (A–H).

Across all roles, Challenge A receives consistently high ratings, with OEMs assigning relatively higher importance than the other groups, whose evaluations remain closely aligned. In contrast, clearer role-specific results emerge for B and C, where suppliers report higher perceived difficulty than OEMs and researchers, while tool vendors occupy an intermediate position. For D, the ratings are largely comparable across roles.

More pronounced divergence is observed for E, where suppliers again report higher concern than OEMs and tool vendors, with researchers positioned between these groups. A distinct pattern appears for F, which is rated higher by tool vendors than by the other roles. Finally, for G and H, suppliers and tool vendors consistently assign higher importance than OEMs, with researchers again showing intermediate evaluations.

Overall, Figure 2 shows that, while the general rating trends remain similar across all roles, differences between organizational roles primarily manifest at the level of individual challenge aspects (A–H).

## F. Themes from open-ended responses

The open-ended questions were analyzed using a qualitative content analysis approach based on Mayring. The analysis resulted in a set of themes that complement the predefined challenge aspects and AI-related items covered by the closed questions. For clarity, the results are presented according to the three open-ended questions in the questionnaire.

## Additional Challenges in Model Exchange and Integration

Respondents identified several additional challenges that extend beyond the predefined integration aspects. A prominent aspect concerns tool limitations and interoperability, including restricted access to model data, unstable exchange mechanisms, and practical difficulties with standardized formats.

A second theme relates to configuration, change, and consistency management, encompassing issues such as tracking changes across model versions, maintaining consistency between models and related lifecycle artifacts, and handling model evolution across iterations.

Respondents also emphasized process-, organizational-, and governance-related constraints, including late identification of the need for model exchange, insufficient integration of model exchange into information management strategies, and legal or contractual requirements that finally enforce document-based deliverables.

Finally, several responses pointed to granularity and perspective mismatches, where differences in abstraction levels or disciplinary viewpoints complicate integration, even when shared modeling tools are used.

## Promising AI Use Cases for Model Integration

Regarding AI support, respondents highlighted several classes of promising Use Cases. One recurring theme describes AI-assisted model understanding and analysis, including AIbased copilot functionality, natural-language querying of models, and support for detecting semantic conflicts or inconsistencies.

Another frequently mentioned theme focuses on the automation of repetitive and low-level modeling tasks, such as correcting notation errors, reducing manual interaction effort, and supporting rapid model validity checks in large models.

Respondents also pointed to model generation and transformation support as a promising area, including the transformation of informal or textual inputs into SysML models, automated diagram creation or refactoring, and transformations across methodologies or SysML versions.

A further theme concerns traceability and cross-domain integration, where AI is seen as supporting end-to-end traceability, consistency analysis, and impact assessment across models, requirements, and related lifecycle artifacts.

## Additional Comments and Suggestions on AI Assistance

In the final open-ended question, respondents frequently addressed preconditions and risks for AI adoption, emphasizing that AI should not be used to compensate for poor data quality or insufficient modeling discipline, and that fully automated solutions should be approached with caution.

Another theme highlights the role of AI in knowledge structuring and representation, including transforming unstructured information into formal representations and enabling more intuitive access to structured knowledge through naturallanguage interaction.

![](images/874f8d551659d26febbea6194e5085fb801de61297d1a9e300b7f189b4b2ccfd.jpg)  
Fig. 2. Perceived challenge severity across respondent roles

Finally, respondents stressed tool-chain integration and architectural requirements for effective AI assistance, such as the need for standardized API, write-back capabilities into engineering tools, and closer integration of SysML-based MBSE tools with related environments such as ALM, PLM, and Digital Twin platforms.

## V. DISCUSSION

This section interprets the survey findings in relation to existing MBSE and AI-assisted Systems Engineering research. While the preceding sections reported the empirical results of the questionnaire, the following discussion relates these observations to prior work and interprets them in the context of the state of the art. It focuses on explaining why certain integration challenges are perceived as particularly prominent, how these perceptions vary across roles and experience levels, and what the findings imply for the design of AI-assisted model integration in SysML-based MBSE contexts.

## A. Core integration challenges in SysML-based MBSE

The survey results indicate that SysML-based model integration is experienced as a multi-dimensional alignment challenge spanning semantics, behavior, traceability, and exchange formats, rather than a single technical bottleneck. The consistently high ratings of these aspects suggest that integration challenges emerge across multiple layers of the system model simultaneously.

In the responses, issues related to naming and semantics remain particularly prominent. From an interpretative perspective, inconsistent terminology and discipline-specific interpretations may lead to divergent understandings of ostensibly shared model elements, even when SysML is adopted on both sides of OEM and supplier. Structural and abstraction-level mismatches may further compound this problem, as supplier subsystem models often reflect implementation-oriented viewpoints that are difficult to reconcile with OEM-level architectural decompositions. Behavioral misalignment can introduce complexity, especially in iterative integration scenarios, where asynchronous model evolution may result in inconsistencies between expected and realized system behavior. Traceability gaps and format-related issues suggest that integration challenges extend beyond model content to encompass lifecycle information management and exchange infrastructure.

In this context, the more formal semantic foundation of SysML v2 provides an important basis for reducing ambiguity and supporting automated analysis and consistency checking [5]. However, the results indicate that formalized language semantics alone are not enough for cross-organizational collaboration. Explicit alignment mechanisms may help to ensure semantic clarity throughout the collaboration process.

Open-ended responses on format gaps and metadata issues highlight concerns about fragile XMI exchanges and toolspecific transformations, which underline the need for more robust exchange approaches. Standardized API, as proposed in SysML v2, may help to enable controlled access to model content and support continuous integration across organizational boundaries, shifting collaboration toward more sustainable integration workflows.

Finally, the AI-related findings suggest that AI can support cross-organizational integration primarily in an assistive role. MBSE stakeholders see clear value in AI for model understanding, semantic analysis, and conflict detection, while expressing caution toward highly generative or autonomous transformations. This indicates that AI can be positioned as a complementary layer that enhances transparency and reduces cognitive load, embedded within well-defined semantic, organizational, and technical integration frameworks.

## B. Role- and experience-specific perspectives

The analysis of subgroup differences reveals that perceived integration challenges vary with both MBSE stakeholders level of engagement in integration activities and their organizational roles, indicating that integration difficulties are shaped by practical exposure and responsibility within the collaboration workflow.

With respect to engagement frequency, respondents with lower involvement in cross-organizational tasks tend to rate naming and semantic conflicts as well as traceability issues slightly higher than those with frequent engagement. In contrast, highly engaged respondents consistently assign higher importance to challenges related to behavioral alignment, model format, methodological and profile integration, and metadata consistency. Meanwhile, differences related to structural abstraction and interface integration remain comparatively small across engagement groups. One possible explanation is that occasional integration efforts are particularly exposed to terminology and traceability issues during initial alignment, while frequent integration activities bring cumulative experience with behavioral, format, and metadatarelated problems to the foreground.

Role-based comparisons indicate that perceived importance of the different challenge aspects varies with organizational responsibilities. OEM respondents assign the highest importance to naming and semantic conflicts, consistent with their system-level responsibility for consolidating models from multiple suppliers and disciplines. Suppliers exhibit a more evenly distributed concern profile, but report particularly high difficulty with structural and abstraction mismatches as well as methodology and profile integration, reflecting the effort required to align implementation-level models with OEMdefined system architectures and methods. Tool providers and consultants are particularly sensitive to traceability gaps and format-related issues, in line with their role in supporting interoperability across heterogeneous tools and maintaining data continuity during model exchange. Researchers display comparatively balanced ratings across aspects, with slightly elevated attention to semantic, behavioral, and traceability issues, which may reflect exploratory modeling contexts and diverse methodological research. Overall, these results suggest that integration support is likely to be more effective when it is role-aware rather than generic—for instance, by prioritizing semantic harmonization for OEMs, methodology alignment for suppliers, and robust traceability and exchange infrastructures for tool providers and consultants.

## C. Insights for AI-assisted integration

The survey results indicate a broadly positive but differentiated attitude toward the use of AI in SysML-based model exchange and integration, with stakeholders assigning higher value to analysis and supportive tasks such as model comprehension, semantic structure analysis, and inconsistency detection. These results suggest that AI is primarily perceived as a means to enhance engineers’ ability to understand and assess complex, distributed models rather than to replace established integration practices.

In contrast, AI use cases that involve automated generation of integration workflows or suggestions for cross-methodology alignment receive lower and more heterogeneous ratings. This may reflect a cautious stance toward AI interventions that directly influence modeling decisions or methodological alignment, particularly in cross-organizational contexts where responsibility and contractual boundaries are clearly defined.

Preferences regarding human–AI collaboration further clarify this distinction. Most respondents favor human-leading interaction modes, in which AI provides recommendations or partial automation but all outcomes remain subject to mandatory human review. Notably, neither fully manual approaches that exclude AI support nor highly automated modes with minimal human oversight gain support, indicating that MBSE stakeholders seek a balanced integration of AI and human decision while preserving control and responsibility.

## D. Insights from Open-Ended Responses

The open-ended responses suggest that cross-organizational SysML-based system model integration requires coordinated advances in semantics, infrastructure, and tooling. While SysML v2 offers a stronger formal semantic basis, effective collaboration additionally depends on explicit alignment mechanisms.

Persistent concerns regarding format gaps and fragile exchanges highlight the need for API-centric, tool-independent integration infrastructures that replace file-based model transfer and enable controlled access to model data. At the same time, the importance of traceability and consistency challenges indicates that integration support must include lifecycle artifacts and version management.

Finally, the AI-related results point to a clear design boundary: AI is most acceptable as an analysis assistant supporting model understanding, semantic analysis, and traceability, while structurally invasive or generative decisions should remain under human control.

## E. Limitations and future work

This study has several limitations that should be considered when interpreting the results. First, the sample size is relatively small and based on voluntary participation within specific MBSE communities, which may introduce selection bias and limits the extent to which the findings can be generalized. Second, the findings rely on self-reported perceptions measured on ordinal Likert scales, and thus reflect subjective experience rather than objective integration performance. Third, rolebased analyses are affected by overlapping role assignments and uneven group sizes, which constrains the strength of comparative conclusions.

Future work will focus on developing and evaluating concrete model alignment mechanisms for cross-organizational SysML-based integration, addressing semantic, behavioral, and traceability consistency across heterogeneous tools and abstraction levels. In this context, AI-assisted approaches will be explored as supportive mechanisms—for example, to assist semantic alignment, detect inconsistencies, and maintain traceability—while keeping integration decisions under explicit human control.

## VI. CONCLUSION

This paper provides empirical insights into how MBSE stakeholders perceive integration challenges and AI opportunities in SysML-based system model exchange across organizational boundaries. Addressing RQ1, the findings show that integration difficulties arise from semantic, behavioral, traceability, and format-related issues, whose relative importance varies with organizational role and integration experience.

Regarding the usefulness of AI-supported tasks (RQ2), stakeholders attribute high value to AI support for analysis tasks such as model understanding, semantic conflict detection, and traceability analysis, while expressing more cautious attitudes toward generative transformations and highly automated workflows. Open-text responses further highlight design requirements for future integration solutions, including robust standards and API, semantically enriched data infrastructures, and close integration with lifecycle management and Digital Twin environments.

Overall, the paper contributes practitioner-grounded insights to ongoing discussions on SysML v2–based collaboration and AI-assisted Systems Engineering. It supports a view of AI as an embedded, assistive component in MBSE tool-chains that enhances transparency and reduces routine workload under human oversight. The results point toward concrete research directions for building trustworthy, role-sensitive integration support in future MBSE ecosystems.

## REFERENCES

[1] M. Elaasar, A. Hamou-Lhadj, B. Oakes, and M. Hamdaqa, “Model-based systems engineering perspectives: A survey of practitioner experiences and challenges,” in 2025 ACM/IEEE 28th International Conference on Model Driven Engineering Languages and Systems Companion (MODELS-C). IEEE, 2025, pp. 367–376.

[2] M. Sohrt, W. Blinkenberg, and N. H. Mortensen, “Shared product architectures for engineering-to-order buyers and suppliers: Insights from a case study,” Applied Sciences, vol. 15, no. 17, p. 9357, 2025.

[3] prostep ivip, “Recommendation sysml wf/if,” prostep ivip e.V., Tech. Rep. PSI 28, 02 2023, sysML Workflow Forum and Implementor Forum Recommendation. [Online]. Available: https://www.ps-ent-2023. de/fileadmin/prod-download/Recommendation SysML WF-IF.pdf

[4] M. Bajaj, S. Friedenthal, and E. Seidewitz, “Systems modeling language (sysml v2) support for digital engineering,” INSIGHT, vol. 25, no. 1, pp. 19–24, 2022.

[5] N. Jansen, J. Pfeiffe, B. Rumpe, D. Schmalzing, and A. Wortmann, “The language of sysml v2 under the magnifying glass,” The Journal of Object Technology, vol. 21, no. 3, p. 3:1, 2022.

[6] J. K. DeHart, “Leveraging large language models for direct interaction with sysml v2,” INCOSE International Symposium, vol. 34, no. 1, pp. 2168–2185, 2024.

[7] J.-M. Gauthier, E. Jenn, and R. Conejo, “Ontology-driven llm assistance for task-oriented systems engineering,” in Proceedings of the 13th International Conference on Model-Based Software and Systems Engineering - Volume 1: MBSE-AI Integration, INSTICC. SciTePress, 2025, pp. 383–394.

[8] T. G. Topcu, M. Husain, M. Ofsa, and P. Wach, “Trust at your own peril: A mixed methods exploration of the ability of large language models to generate expert–like systems engineering artifacts and a characterization of failure modes,” Systems Engineering, 2025.

[9] L. Bretz, L. Kaiser, and R. Dumitrescu, “An analysis of barriers for the introduction of systems engineering,” Procedia CIRP, vol. 84, pp. 783–789, 2019, 29th CIRP Design Conference 2019, 08-10 May 2019, Povoa de Varzim, Portgal.´

[10] Z. Li, F. Faheem, and S. Husung, “Collaborative model-based systems engineering using dataspaces and sysml v2,” Systems, vol. 12, no. 1, p. 18, 2024.

[11] K. Henderson and A. Salado, “Value and benefits of model-based systems engineering (mbse): Evidence from the literature,” Systems Engineering, vol. 24, no. 1, pp. 51–66, 2021.

[12] Systems Engineering Vision 2035, “Systems engineering vision 2035,” 6/17/2025. [Online]. Available: https://sevisionweb.incose.org/

[13] W. D. Miller, “The future of systems engineering: Realizing the systems engineering vision 2035,” in Transdisciplinarity and the Future of Engineering. IOS Press, 2022, pp. 739–747.

[14] A. Mahboob and S. Husung, “A modelling method for describing and facilitating the reuse of sysml models during design process,” in International Design Conference – DESIGN 2022, 2022, vol. 2.

[15] S. Friedenthal, A. Moore, and R. Steiner, A Practical Guide to SysML: The Systems Modeling Language. Morgan Kaufmann, 2014.

[16] S. Friedenthal, “Future directions for mbse with sysml v2,” in Proceedings of the 11th International Conference on Model-Based Software and Systems Engineering. SCITEPRESS - Science and Technology Publications, 2023, pp. 5–9.

[17] J. A. Estefan et al., “Survey of model-based systems engineering (mbse) methodologies,” Incose MBSE Focus Group, vol. 25, no. 8, pp. 1–12, 2007.

[18] C. Zhou, B. An, B. Yu, and S. Li, “Data exchange for sysml: A review,” in Mechanical Design and Simulation: Exploring Innovations for the Future, ser. Lecture Notes in Mechanical Engineering, D. T. Pham, Y. Lei, and Y. Lou, Eds. Singapore: Springer Nature Singapore, 2025, pp. 835–846.

[19] Z. Li, S. Husung, and H. Wang, “Llm-assisted semantic alignment and integration in collaborative model-based systems engineering using sysml v2,” in 2025 IEEE International Symposium on Systems Engineering (ISSE), 2026.

[20] E. Cibrian, J. Olivert-Iserte, J. Llorens, and J. M.´ Alvarez Rodr<sup>´</sup> ´ıguez, “An agent-based approach for the automatic generation of valid sysmlv2 models in industrial contexts,” Computers in Industry, vol. 172, p. 104350, 2025.

[21] F. Belkadi, M. Messaadia, A. Bernard, and D. Baudry, “Collaboration management framework for oem – suppliers relationships: a trust-based conceptual approach,” Enterprise Information Systems, vol. 11, no. 7, pp. 1018–1042, 2017.

[22] J. Pandolf, “Investigation of model-based systems engineering integration challenges and improvements,” Ph.D. dissertation, Massachusetts Institute of Technology, 2023.

[23] S. Powley and S. Perry, “Taking it all in: Representing multiple systems of interest in a single model,” in INCOSE UK Annual Systems Engineering Conference 2020. INCOSE UK, 2020.

[24] A. Benveniste, B. Caillaud, D. Nickovic, R. Passerone, J.-B. Raclet, P. Reinkemeier, A. Sangiovanni-Vincentelli, W. Damm, T. Henzinger, and K. G. Larsen, “Contracts for systems design: Theory,” Ph.D. dissertation, Inria Rennes Bretagne Atlantique; INRIA, 2015.

[25] I. Ghanawi, M. W. Chami, M. Chami, M. Coric, and N. Abdoun, “Integrating ai with mbse for data extraction from medical standards,” INCOSE International Symposium, vol. 34, no. 1, pp. 1354–1366, 2024.

[26] Z. Li, S. Husung, and H. Wang, “Llm-assisted semantic alignment and integration in collaborative model-based systems engineering using sysml v2,” arXiv preprint arXiv:2508.16181, 2025.

[27] M. U. Hadi, R. Qureshi, A. Shah, M. Irfan, A. Zafar, M. B. Shaikh, N. Akhtar, J. Wu, S. Mirjalili et al., “Large language models: a comprehensive survey of its applications, challenges, limitations, and future prospects,” Authorea preprints, vol. 1, no. 3, pp. 1–26, 2023.

[28] M. Hollender, C. Xu, and R. Tan, “Engineering challenges in industrial ai,” in Proceedings of the IEEE/ACM 3rd International Conference on AI Engineering - Software Engineering for AI. New York, NY, USA: ACM, 2024, pp. 41–42.

[29] P. J. Kulkarni, D. Tissen, R. Bernijazov, and R. Dumitrescu, “Towards automated design: Automatically generating modeling elements with prompt engineering and generative artificial intelligence,” in DS 130: Proceedings ofNordDesign 2024, Reykjavik, Iceland, 12th - 14th August 2024, 2024, pp. 617–625.

[30] Google Forms, “Challenges and ai assistance in model exchange and integration (sysml 2.0 context),” 2026. [Online]. Available: https://forms.gle/jGmEMH8BAGM6dmet8

[31] P. Mayring, “Qualitative content analysis: Demarcation, varieties, developments,” in Forum: Qualitative social research, vol. 20, no. 3. Freie Universitat Berlin, 2019.¨