# Generative artificial intelligence prompting strategies among Chinese university students across academic tasks

Lei Fan<sup>1,\*</sup> and Fangxue Liu<sup>1,2</sup>

<sup>1</sup> Department of Civil Engineering, Xi’an Jiaotong-Liverpool University, Suzhou, 215123, China

<sup>2</sup> Department of Computer Science, University of Liverpool, Liverpool, L69 3BX, UK

\*Corresponding Author (Email: Lei.Fan@xjtlu.edu.cn)

Abstract: Generative artificial intelligence (GenAI) is increasingly embedded in university learning, yet effective human–GenAI interaction depends not only on access to these tools but also on students’ ability to formulate, evaluate, and refine prompts for academic tasks. This study examined university students perceived GenAI prompting capability and reported prompting strategies. Questionnaire data from 234 students were analyzed using descriptive statistics, measurement modelling, scenario comparisons, structural associations, and qualitative coding of optional comments. Students reported using GenAI across diverse academic activities and generally expressed confidence in their prompting capabilities, particularly in providing relevant context and refining prompts following initial outputs. However, prompting strategies showed limited sensitivity to task types: specifying output format was more strongly endorsed in a creative workshop preparation scenario than in a research-oriented assignment, whereas other matched strategies showed no adjusted differences. Students also commonly endorsed both concise instructions and providing all relevant information at the outset, suggesting that effective brevity may depend more on selecting and organizing task-relevant information than on prompt length alone. By contrast, fixed prompt templates received lower endorsement than other prompting strategies. In the adopted factor model, perceived cognition showed the clearest unique association with perceived prompting capability, while open-ended responses highlighted prompt–intent alignment as a common challenge.

Keywords: generative artificial intelligence; prompt engineering; prompting strategies; higher education; human-AI interaction

## 1. Introduction

Generative artificial intelligence (GenAI) has made support for explanation, writing, information seeking, and idea development readily available to university students. However, an accessible conversational interface does not eliminate the intellectual demands of effective human–GenAI interaction. Students must formulate appropriate requests, provide relevant context, and judge whether generated responses are accurate, useful, and suitable for the task. Previous surveys documented perceived academic benefits alongside concerns about accuracy, reliability, and responsible use (Chan & Hu, 2023; Fan, Deng, & Liu, 2025; Fan & Liu, 2026a). How students manage these interactions is therefore an educational issue rather than merely a question of technology adoption.

Prompt engineering refers to the deliberate formulation and iterative refinement of instructions, context, and constraints to guide GenAI toward a task-appropriate response. It includes decisions about defining a task, supplying relevant background or examples, specifying output requirements, evaluating generated responses, and revising prompts when necessary. Prompt patterns provide reusable structures for organizing such interactions (White et al., 2023). In educational settings, however, the value of prompting lies not simply in producing a better response but in whether the interaction supports the purpose of the academic task and enables critical evaluation of the resulting output. Research has linked aspects of prompt quality and artificial intelligence (AI) literacy to the quality of GenAI interactions, but improved generated output should not itself be interpreted as evidence of improved learning (Knoth et al., 2024; Fan, Tang, et al., 2025).

Important limitations remain in the literature. First, much of the survey research on student GenAI use has clarified patterns of adoption, perceived benefits, and concerns but has provided less detail about how students formulate requests during GenAI interaction. Second, empirical studies of prompting have often examined particular tasks or disciplinary settings, including introductory programming and undergraduate history (Nguyen et al., 2024; Woo et al., 2026). Such work has provided valuable evidence about prompting within specific learning activities, but broader evidence is still needed on how students report adapting prompting strategies across contrasting academic tasks.

To address these limitations, this study used a questionnaire to examine students’ perceived prompting capability, strategies and difficulties. The study also investigated how four broader AI-related orientations (i.e., affect, cognition, behavioral intention, and ethics), were associated with perceived prompting capability and scenario-specific prompting strategies. Drawing on multidimensional accounts of AI literacy, these orientations were treated as related but distinct aspects of students’ engagement with GenAI rather than as interchangeable indicators of prompting capability (Long & Magerko, 2020; Ng et al., 2021; Carolus et al., 2023).

Three research questions guided the analysis. RQ1: What patterns characterize students’ perceived prompting capability, and reported prompting difficulties? RQ2: How do students’ prompting strategies differ between the research-oriented assignment and creative workshop preparation scenarios? RQ3: How are affect, perceived cognition, behavioral intention, and ethics associated with perceived prompting capability and scenario-specific prompting strategies, and how is perceived prompting capability associated with strategy use across the two scenarios?

## 2 Literature review

## 2.1 Prompt engineering in higher education

In higher education, prompt engineering can be understood as the deliberate specification and iterative refinement of a task for a GenAI system. A prompt may define the problem, provide relevant context or examples, constrain the expected response, or request subsequent revision. White et al. (2023) organized such decisions into reusable prompt patterns, while Lo’s (2023) CLEAR framework framed effective prompting in terms of being concise, logical, explicit, adaptive, and reflective within an information-literacy context. These frameworks provide useful conceptual vocabularies for describing how users structure interactions with GenAI, but they do not by themselves establish that applying a particular pattern improves learning. Their educational relevance therefore depends on how prompting is aligned with the task and how generated responses are subsequently interpreted, evaluated, and used.

Empirical research indicated that effective prompting involved more than entering an initial instruction. In a study of beginning programmers, Nguyen et al. (2024) found difficulties in articulating intentions, evaluating generated code, and revising prompts when outputs were inadequate. Zamfirescu-Pereira et al. (2023) similarly found that non-AI experts often relied on exploratory and trial-and-error prompting rather than systematically diagnosing why an interaction had failed. Together, these findings identified formulation, contextualization, evaluation, and revision as meaningful components of human–GenAI interaction. They also supported examining how students perceived their capability in these areas and which strategies they reported using, while recognizing that self-reported practices do not establish whether those strategies were implemented effectively.

Research further suggested that prompting practices were shaped by task and disciplinary context. Aruleba et al. (2025) examined computer-science students’ prompting strategies and reflections across programming activities, whereas Woo et al. (2026) investigated changes in GenAI knowledge, self-efficacy, prompt artifacts, and reflections following a prompt-engineering workshop with undergraduate history students. Knoth et al. (2024) likewise examined relationships among AI literacy, prompt-engineering quality, and generated outputs across different tasks. Collectively, this work supported viewing prompting as a contextsensitive practice rather than as a single general technique. At the same time, the concentration of individual studies on particular tasks or disciplinary settings left scope for broader evidence on how students reported adapting strategies across contrasting academic and creative contexts.

## 2.2 Prompting and student learning

The relationship between prompting and learning requires a distinction between the quality of a GenAIassisted product and the learner’s knowledge, transfer, and evaluative judgment. Numerous empirical studies found that students valued GenAI for activities including writing, research, and idea development while also expressing concerns about reliability and ethical use (Chan & Hu, 2023; Fan & Liu, 2026b). Among engineering students, Fan, Deng & Liu (2025) reported strong perceived improvements in learning efficiency, while many respondents perceived little corresponding change in academic performance. These findings helped explain why students considered GenAI useful, but they neither identified which prompting practices produced the perceived benefits nor established learning effects.

Experimental evidence reinforced this distinction. In a randomized study of university students, Fan, Tang, et al. (2025) found that the ChatGPT-supported group achieved greater improvement in essay scores, whereas differences in knowledge gain and transfer were not statistically significant. Improved performance on a GenAI-assisted work therefore should not automatically be interpreted as improved learning. For prompting pedagogy, this distinction suggests value in considering not only the quality of the generated product but also whether students can explain their decisions, evaluate and verify generated information, and justify revisions. These are pedagogical implications of the existing evidence rather than outcomes tested directly in the present survey.

Self-regulated learning provides a useful interpretive lens because its models connect planning, monitoring, control, and reflection (Zimmerman, 2002; Panadero, 2017). Prompt formulation can represent planning, evaluation of a generated response can contribute to monitoring, and prompt revision can function as a control response when an output does not meet the task goal. However, reported revision alone does not demonstrate effective monitoring or self-regulation: students may repeatedly modify prompts without identifying why a response is inadequate. Constructive alignment provides a complementary pedagogical perspective by emphasizing consistency among intended learning outcomes, learning activities, and assessment (Biggs, 1996). Applied to prompting instruction, this perspective suggests that prompt formulation and output evaluation should remain connected to the disciplinary objectives that the academic task is intended to develop.

## 2.3 Contextual strategy use

Prompting is inherently context sensitive because the relevance of providing background, assigning a role, specifying an output format, imposing constraints, or revising an instruction varies with the task goal, information requirements, and expected product. The same strategy may therefore be applicable across contexts while serving different functions or carrying different importance in each. Prompt-pattern research has shown how reusable prompting structures can be adapted across problems and domains, while disciplinary studies have demonstrated that prompting practices are embedded in the requirements of particular learning activities (White et al., 2023; Aruleba et al., 2025; Woo et al., 2026). This literature supports examining strategy use within clearly specified contexts while reserving claims about strategy effectiveness for research that directly observes prompts, generated outputs, and relevant learning or performance outcomes.

## 3 Methodology

## 3.1 Study design and participants

The study used a cross-sectional online questionnaire administered via Wenjuanxing from early July to early September 2026. Convenience sampling was supplemented by purposive recruitment to improve representation across gender, degree level, academic discipline, and geographic regions of China. The study protocol was approved by the University Research Ethics Review Panel of Xi’an Jiaotong-Liverpool University on 8 Jun 2026. The survey was conducted in accordance with applicable ethical guidelines and regulations, and informed consent was obtained from all participants prior to their participation.

Disciplines were grouped into science, technology, engineering, and mathematics (STEM) and humanities and social sciences (HSS). Responses with identical numerical ratings across all rating items or indicating “none tried” in the prompting-method question were excluded, yielding a final analytical sample of 234 respondents. Participant characteristics are summarized in Table 1.

Table 1. Participant characteristics
<table><tr><td>Characteristic</td><td>Category</td><td>Percentage (%)</td></tr><tr><td rowspan="2">Gender</td><td>Male</td><td>43.2</td></tr><tr><td>Female</td><td>56.8</td></tr><tr><td rowspan="3">Degree level</td><td>Bachelor</td><td>47.0</td></tr><tr><td>Postgraduate</td><td>24.4</td></tr><tr><td>Three-year college diploma</td><td>28.6</td></tr><tr><td rowspan="2">Subject category</td><td>STEM</td><td>51.7</td></tr><tr><td>HSS</td><td>48.3</td></tr></table>

## 3.2 Questionnaire and measures

The questionnaire comprised three components: participant background information, GenAI use characteristics, and rating items assessing GenAI-related orientations and prompting strategies. Background and use questions covered frequency of GenAI use, commonly used tools, academic tasks for which GenAI had been used, and prompting methods previously attempted.

The main rating component included 27 items across seven domains. The first four domains contained three items each and assessed affective orientation (Domain 1; D1), perceived cognition (Domain 2; D2), behavioral intention (Domain 3; D3), and ethical orientation (Domain 4; D4). Domain 5 (D5) comprised five items assessing perceived prompting capability. Domains 6 and 7 assessed prompting strategies in two task scenarios, with five items each. Domain 6 (D6) focused on a research-oriented assignment examining the impact of GenAI on university students’ employment, whereas Domain 7 (D7) focused on developing an appealing title and slogan for a workshop on GenAI and the future of education.

Four additional rating items assessed general prompting strategies: using brief instructions, structuring longer prompts, providing detailed information at the outset, and using a fixed prompt template. These items were analyzed separately because they reflected distinct interaction preferences rather than a common latent construct.

All 31 rating items used a five-point Likert-type scale ranging from strongly disagree (1) to strongly agree (5). All closed-ended questions were mandatory. The questionnaire concluded with an optional open-ended question asking respondents to describe difficulties they had experienced when prompting GenAI. The full questionnaire is provided in Appendix A.

## 3.3 Statistical analysis

Descriptive analyses included percentages, means (M), and item-level response distributions. For agreement-based items, endorsement was defined as selecting agree (4) or strongly agree (5). Internal consistency for each of the seven questionnaire domains was assessed using Cronbach’s alpha (α), with 95% confidence intervals based on 5,000 participant-level bootstrap resamples. The Kaiser–Meyer–Olkin (KMO) measure and Bartlett’s test of sphericity (Kaiser & Rice, 1974; Tobias & Carlson, 1969) were used to assess the suitability of the correlation structure for factor analysis. These reliability and factorability diagnostics were treated as supporting evidence rather than as direct evidence of construct validity.

Contextual differences between the research-oriented assignment and creative workshop preparation scenarios were examined using paired-samples t tests for four conceptually matched strategy items. For each comparison, we reported the creative-minus-research mean difference, its unadjusted 95% confidence interval, and the standardized paired effect size dz, calculated as the mean paired difference divided by the standard deviation of the paired differences. The third item in each scenario was excluded because the two items represented substantively different strategies. The Benjamini–Hochberg (BH) procedure was applied across the four tests to control the false discovery rate (Benjamini & Hochberg, 1995).

The measurement structure was examined using a seven-correlated-factor model in which each of the 27 domain indicators loaded only on its predefined factor, with no cross-loadings or correlated indicator residuals. A strict one-factor model was fitted for comparison. Both models used normal-theory maximum likelihood estimation based on the observed covariance matrix, treating the five-category indicators as approximately continuous (Rhemtulla et al., 2012). Model fit was evaluated using the chi-square statistic (χ²), comparative fit index (CFI), Tucker–Lewis index (TLI), root mean square error of approximation (RMSEA) with 90% confidence intervals, standardized root mean square residual (SRMR), standardized factor loadings, and latent-factor correlations. Accordingly, the reported fit indices were conventional maximum-likelihood rather than robust scaled ordinal statistics.

A structural model was then used to examine associations among the seven domains. Perceived prompting capability was regressed on the four GenAI-related orientation domains, while each scenario-specific strategy domain was regressed on the four orientations and perceived prompting capability. This specification yielded 14 structural paths, treated as a separate BH testing family from the four scenario comparisons. Standardized path coefficients were estimated using sandwich standard errors and delta method confidence intervals (Oehlert, 1992). Because this structural specification was covarianceequivalent to the seven-factor measurement model, overall model fit could not establish the direction of the structural relationships and was not interpreted causally. Complete path estimates are reported in Section 4.8.

Responses to the optional open-ended question were analyzed using descriptive content coding, with multiple codes permitted for each response where appropriate. Percentages were calculated only among respondents who provided a comment and were therefore not interpreted as prevalence estimates for the full sample.

## 4 Results and interpretation

## 4.1 Internal consistency and factorability

As shown in Table 2, all seven domains demonstrated high internal consistency, with Cronbach’s alpha ranging from 0.858 to 0.935. The lower bounds of all bootstrap 95% confidence intervals exceeded 0.80, indicating consistently strong reliability. Domain-specific KMO values ranged from 0.731 to 0.894, and the overall KMO for the 27 domain items was 0.964. Bartlett’s test of sphericity was significant, χ²(351) = 5413.80, p < 0.001, indicating that the correlation matrix differed substantially from an identity matrix. Collectively, these results showed that the item correlations were suitable for factor analysis and that the seven domains had strong internal consistency. These diagnostics supported the subsequent measurement analyses but were not treated as evidence of construct validity in themselves.

Table 2 Domain summaries and internal consistency
<table><tr><td>Domain</td><td>Items</td><td>a [95% Confidence Interval]</td><td>KMO</td></tr><tr><td>D1: Affective orientation</td><td>3</td><td>0.908 [0.873, 0.934]</td><td>0.753</td></tr><tr><td>D2: Perceived cognition</td><td>3</td><td>0.866 [0.823, 0.899]</td><td>0.731</td></tr><tr><td>D3: Behavioral intention</td><td>3</td><td>0.861 [0.822, 0.894]</td><td>0.732</td></tr><tr><td>D4: Ethical orientation</td><td>3</td><td>0.858 [0.806, 0.898]</td><td>0.733</td></tr><tr><td>D5: Perceived prompting capability</td><td>5</td><td>0.935 [0.916, 0.949]</td><td>0.894</td></tr><tr><td>D6: Research-oriented assignment strategies</td><td>5</td><td>0.903 [0.872, 0.926]</td><td>0.836</td></tr><tr><td>D7: Creative workshop strategies</td><td>5</td><td>0.890 [0.853, 0.917]</td><td>0.878</td></tr></table>

## 4.2 Academic uses of GenAI

As shown in Figure 1, students used GenAI across a range of academic tasks. The most frequently reported uses were report or paper writing (60.7%), literature searching and reading (59.4%), and report or paper polishing (57.7%). Learning new knowledge or explaining concepts (45.7%), data analysis (42.7%), and translation (40.2%) were also commonly reported. The high proportions for both writing and polishing suggest that students used GenAI extensively for both text generation and revision. Similar patterns have been reported in previous studies (e.g., Chan & Hu, 2023).

![](images/1c124f8dd6ef73b239210514fe4f02cba8c7acdba79f39cfb735dc1e79b05b48.jpg)  
Figure 1. Academic tasks supported by GenAI tools. Multiple selections were allowed.

## 4.3 Orientations toward AI technologies

Figure 2 showed generally positive orientations toward AI technologies. Within affective orientation, interest in learning about AI (67.5%) and eagerness to explore new AI technologies (67.1%) were endorsed more frequently than confidence in completing AI-related tasks (62.0%). Within behavioral intention, continued use received the highest endorsement (75.7%), followed by exploring new application features (67.6%) and discussing AI-related topics with peers (64.1%). These results suggested that willingness to continue using AI was more widespread than confidence or intentions involving more active exploration and peer engagement. The overall positive orientation was broadly consistent with favorable student perceptions of GenAI reported by Chan & Hu (2023).

For perceived cognition, 71.8% of respondents endorsed their ability to use AI to solve practical problems, 67.1% endorsed evaluating the suitability of AI applications, and 58.2% endorsed distinguishing relevant technical concepts. The conceptual-differentiation item also had the largest neutral response share (28.2%), suggesting greater uncertainty in this area than in practical application or evaluation. Ethical orientation was comparatively consistent across items: 69.3% recognized substantial risks associated with misuse, 71.8% supported adherence to ethical and legal standards, and 69.7% affirmed user responsibility. The combined emphasis on practical application, evaluation, and ethical responsibility was consistent with multidimensional accounts of AI literacy, in which understanding, application, evaluation, and responsible use are related but distinct components (Long & Magerko, 2020; Ng et al., 2021).

![](images/1f1775ec80b06ce5e9bfdf4686935c3aea497a890b58c8eb6bf156f1b35746e3.jpg)  
Figure 2. Response distributions for (a) affective orientation, (b) perceived cognition, (c) behavioral intention, and (d) ethical orientation. For each item, the stacked bar and five numerical columns present the same response distribution; this format is also used in subsequent figures of the same type.

## 4.4 Perceived prompting capability

As shown in Figure 3, endorsement with the five prompting-capability items ranged from 66.2% to 72.2%, indicating generally positive self-assessments. Including task-relevant details, such as background, scope, or intended use, was endorsed by 70.9% of respondents, while 67.1% reported confidence in breaking complex tasks into steps.

Prompt refinement after reviewing an initial response received the highest endorsement (72.2%), and 70.5% reported confidence in modifying prompts when outputs contained factual errors or bias. Together, these results suggest that iterative adjustment was a relatively strong aspect of respondents’ perceived prompting capability and that prompting was commonly viewed as an interactive process rather than a one-off instruction. This pattern is consistent with Aruleba et al. (2025), who likewise emphasized refinement and appraisal of generated responses in students’ prompting practices.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>Strongly                                             Stronglydisagree   Disagree    Neutral     Agree      agree</td></tr><tr><td rowspan=1 colspan=1>Include task-relevant detailsto improve GenAl responseaccuracy and relevance</td><td rowspan=1 colspan=1>3.0%</td><td rowspan=1 colspan=1>9.0%</td><td rowspan=1 colspan=1>17.1%</td><td rowspan=1 colspan=1>32.9%</td><td rowspan=1 colspan=1>38.0%</td></tr><tr><td rowspan=1 colspan=1>Break complex tasks intoclear prompts that guidestep-by-step completion</td><td rowspan=1 colspan=1>3.0%</td><td rowspan=1 colspan=1>10.7%</td><td rowspan=1 colspan=1>19.2%</td><td rowspan=1 colspan=1>32.5%</td><td rowspan=1 colspan=1>34.6%</td></tr><tr><td rowspan=1 colspan=1>Set response constraintsto improve GenAlresponses</td><td rowspan=1 colspan=1>3.4%</td><td rowspan=1 colspan=1>8.1%</td><td rowspan=1 colspan=1>22.2%</td><td rowspan=1 colspan=1>34.6%</td><td rowspan=1 colspan=1>31.6%</td></tr><tr><td rowspan=1 colspan=1>Refine prompts afterthe initial GenAl responseto obtain better results</td><td rowspan=1 colspan=1>3.4%</td><td rowspan=1 colspan=1>9.4%</td><td rowspan=1 colspan=1>15.0%</td><td rowspan=1 colspan=1>36.3%</td><td rowspan=1 colspan=1>35.9%</td></tr><tr><td rowspan=1 colspan=1>Modify prompts to help GenAlcorrect factual errorsor biased responses</td><td rowspan=1 colspan=1>2.1%</td><td rowspan=1 colspan=1>10.3%</td><td rowspan=1 colspan=1>17.1%</td><td rowspan=1 colspan=1>33.3%</td><td rowspan=1 colspan=1>37.2%</td></tr></table>

Figure 3. Self-reported GenAI prompting capability in academic tasks

Setting response constraints, such as role, tone, length, or format, received the lowest agreement (66.2%) and the largest neutral share (22.2%). Relative to iterative refinement and provision of task details, respondents therefore appeared somewhat less certain about their ability to specify and combine output constraints. The distributions suggested comparatively greater uncertainty about explicit output control than about refinement or contextualization. However, because these measures captured self-perceived capability rather than directly observed performance, they should be interpreted as indicators of perceived readiness and confidence rather than demonstrated competence.

## 4.5 Prompting strategies across task contexts

In the research-oriented assignment scenario, respondents considered using GenAI to prepare a course assignment on the impact of GenAI on university students’ employment. As shown in Figure 4(a), revising the prompt or adding information after an unsatisfactory response received the highest agreement (74.3%), followed by providing relevant background before asking the main question (68.8%). Requesting a specific output format and step-by-step analysis each received 59.4% agreement, while assigning a topic-relevant research-expert role received 58.2%. Thus, iterative revision and contextual information were more strongly endorsed than more explicitly structured techniques such as role assignment and output formatting.

In the creative workshop preparation scenario, respondents considered developing an appealing title and slogan for a workshop on “GenAI and the Future of Education”. As shown in Figure 4(b), providing event background received the highest agreement (75.6%), followed by revising prompt requirements after an unsatisfactory response (72.7%), specifying the output format (72.2%), stating a desired creative direction or evaluation criteria (70.6%), and assigning an appropriate role (65.8%). Compared with the researchoriented scenario, endorsement was generally higher and more consistent across strategies.

(a) Research-oriented assignment

![](images/8ed5aec530c6993d398d27866dc59d4685ff788120d7d59c06d5060ef737a42e.jpg)  
(b) Creative workshop preparation task

![](images/878e240e52e0484d3133381c846eea9ccdf0fcffe69427369fa461ee79ad7924.jpg)

(c) Matched strategies: scenario means and paired differences  
![](images/493096a34488969eb59b067b37bfe6d936ad4ce0d21c348792b4c32309955551.jpg)  
Figure 4. Scenario-specific GenAI prompting strategies: (a) research-oriented assignment scenario; (b) creative workshop preparation scenario; and (c) scenario means (left) and Creative-minus-Research differences with pointwise 95% confidence intervals (right). p values are BH-adjusted across the four paired comparisons.

Across both scenarios, providing background information and revising prompts were among the most strongly endorsed strategies, suggesting that respondents viewed contextualization and iterative adjustment as broadly applicable across tasks. From a self-regulated learning perspective, iterative revision may reflect monitoring and adjustment in response to task feedback (Panadero, 2017), although the present self-report data did not capture the quality of such monitoring.

Figure 4(c) and Table 3 present the four conceptually matched strategy comparisons. The third item in each scenario was excluded because the two items represented substantively different strategies. After BH adjustment, only output-format specification differed significantly between scenarios. This strategy was reported more strongly in the creative scenario, with a mean difference of 0.26 points on the five-point scale, dz = 0.233, and BH-adjusted p = 0.002. Differences in role assignment, background information, and iterative revision were not statistically significant after adjustment.

The greater emphasis on output-format specification in the creative scenario likely reflected the task’s clearly defined deliverable (i.e., a title and slogan), which made explicit formatting requirements more salient. In contrast, background information and iterative revision showed no significant contextual differences after adjustment, suggesting that respondents regarded these strategies as broadly applicable across both tasks. Overall, the findings indicate that prompting strategies were generally consistent across contexts, with selective adaptation where specific task demands made particular strategies more relevant. However, these results reflect reported preferences in hypothetical scenarios rather than observed prompting behavior in authentic tasks.

Table 3 Matched scenario comparisons. Research denotes the research-oriented assignment scenario; Creative denotes the workshop-title and slogan creation scenario.
<table><tr><td>Strategy</td><td>Research M</td><td>Creative M</td><td>Difference [95% confidence interval]</td><td>d2</td><td>p</td><td>BH- adjusted p</td></tr><tr><td>Role assignment</td><td>3.67</td><td>3.79</td><td>0.12 [0.00, 0.25]</td><td>0.129</td><td>0.050</td><td>0.096</td></tr><tr><td>Output format</td><td>3.66</td><td>3.92</td><td>0.26 [0.11, 0.40]</td><td>0.233</td><td>&lt; 0.001</td><td>0.002</td></tr><tr><td>Background data</td><td>3.87</td><td>3.99</td><td>0.12 [-0.01, 0.25]</td><td>0.118</td><td>0.072</td><td>0.096</td></tr><tr><td>Iterative revision</td><td>4.03</td><td>3.99</td><td>-0.03 [-0.16, 0.09]</td><td>-0.035</td><td>0.589</td><td>0.589</td></tr></table>

## 4.6 General prompting strategies

As shown in Figure 5, endorsement was highest for organizing longer prompts using line breaks, paragraphs, or punctuation (70.6%), followed by providing relevant details at the outset rather than beginning with a short trial instruction (69.2%). Brief and direct instructions were endorsed by 63.7% of respondents, while 59.4% endorsed using a fixed template, such as role–task–output format. The template item also had the largest neutral share (24.4%), suggesting less consistent preference for standardized prompt structures.

The results indicate that conciseness and completeness were not necessarily viewed as competing strategies. Respondents commonly endorsed both brief instructions and the provision of relevant information from the outset, suggesting that effective prompting may depend more on selecting and organizing task-relevant information than on prompt length alone. This interpretation is broadly consistent with Lo’s (2023) CLEAR framework, which emphasizes conciseness alongside explicitness and iterative refinement.

The relatively strong endorsement of organizing longer prompts further suggests that respondents valued structural clarity when more information was needed. In contrast, the lower endorsement of fixed templates points to greater variation in preferences for standardized prompting formats. Overall, the findings indicate diverse prompting preferences rather than a single dominant style.

![](images/8f6c493c98c557d466133fd2012021c106405348b84a19ad0223707639f06306.jpg)  
Figure 5. General prompting strategies in academic tasks

## 4.7. Reported prompting difficulties

Substantive responses to the optional open-ended question were provided by 12.8% of respondents. Among those who commented, the most common themes were prompt–intent alignment (40.0%), promptformulation difficulty (23.3%), output usefulness or presentation (20.0%), iteration or conversational context (16.7%), and factual verification (13.3%), with smaller frequencies across other miscellaneous categories. Because multiple codes could be assigned to each response and only a minority of respondents provided comments, these percentages describe the commenting subset rather than the prevalence of difficulties in the full sample.

Several comments illustrated these challenges. One respondent noted, “Even though I have described things in great detail, it still does not understand what I mean”, while another reported “difficulty defining specialized problems precisely”. A third stated, “After repeated prompts requesting revisions, the GenAI becomes confused, leading to poorer output quality”. These comments suggest that adding detail or repeatedly revising a prompt may not resolve an interaction when the task objective, relationships among requirements, or criteria for an acceptable response remain unclear.

Previous studies have similarly reported difficulties in expressing intent, refining prompts, and evaluating generated responses (Zamfirescu-Pereira et al., 2023; Nguyen et al., 2024; Aruleba et al., 2025). The present findings highlight prompt–intent alignment as a key challenge: students may be willing to revise prompts yet still struggle to translate their goals into instructions that GenAI can interpret accurately. University training should therefore emphasize clarifying task goals and expressing them through explicit requirements, relevant context, constraints, and criteria for acceptable outputs.

## 4.8. Measurement models and structural associations

Table 4 compared two measurement models for the same 27 rating items. In the one-factor model, all items loaded on a single latent dimension, whereas in the seven-correlated-factor model, items loaded on their questionnaire-defined domains (D1–D7) and the seven factors were allowed to correlate. The latter therefore represented the domains as related but distinct rather than as independent competencies.

The seven-correlated-factor model showed substantially better fit than the one-factor model, with $\chi ^ { 2 } =$ 538.95, p < 0.001, CFI = 0.956, TLI = 0.949, RMSEA = 0.058, and SRMR = 0.036, compared with CFI = 0.847, TLI = 0.835, RMSEA = 0.104, and SRMR = 0.053 for the one-factor model. Standardized loadings in the seven-factor model ranged from 0.733 to 0.905. These results supported retaining the questionnairedefined multidimensional structure rather than reducing all 27 items to a single general factor.

However, the latent factors were highly correlated, indicating substantial overlap among the domains. Thus, the better fit of the seven-factor model supported multidimensionality but did not demonstrate complete separation between constructs. Some shared covariance may also have reflected the common self-report format (Podsakoff et al., 2003).

Table 4 Exploratory measurement model comparison
<table><tr><td>Model</td><td>χ2 (df)</td><td>CFI</td><td>TLI</td><td>RMSEA [90% confidence interval]</td><td>SRMR</td></tr><tr><td>One general factor</td><td>1137.25 (324)</td><td>0.847</td><td>0.835</td><td>0.104 [0.097, 0.110]</td><td>0.053</td></tr><tr><td>Seven correlated factors</td><td>538.95 (303)</td><td>0.956</td><td>0.949</td><td>0.058 [0.050, 0.066]</td><td>0.036</td></tr></table>

Table 5 Complete exploratory structural associations
<table><tr><td>Outcome</td><td>Predictor</td><td>β[95% confidence interval]</td><td>p</td><td>BH- adjusted p</td></tr><tr><td rowspan="4">Perceived prompting capability</td><td>Affective orientation</td><td>0.177 [-0.065, 0.419]</td><td>0.152</td><td>0.265</td></tr><tr><td>Perceived cognition</td><td>0.390 [0.161, 0.619]</td><td>&lt; 0.001</td><td>0.012</td></tr><tr><td>Behavioral intention</td><td>0.241 [0.028, 0.455]</td><td>0.027</td><td>0.094</td></tr><tr><td>Ethical orientation</td><td>0.159 [-0.042, 0.360]</td><td>0.121</td><td>0.241</td></tr><tr><td rowspan="5">Research- oriented assignment</td><td>Affective orientation</td><td>0.068 [-0.217, 0.353]</td><td>0.639</td><td>0.639</td></tr><tr><td>Perceived cognition</td><td>0.195 [-0.096, 0.487]</td><td>0.189</td><td>0.265</td></tr><tr><td>Behavioral intention</td><td>0.138 [-0.116, 0.392]</td><td>0.288</td><td>0.365</td></tr><tr><td>Ethical orientation</td><td>0.105 [-0.099, 0.308]</td><td>0.313</td><td>0.365</td></tr><tr><td>Perceived prompting capability</td><td>0.427 [0.084, 0.769]</td><td>0.015</td><td>0.094</td></tr><tr><td rowspan="5">Creative workshop preparation</td><td>Affective orientation</td><td>0.263 [0.004, 0.522]</td><td>0.047</td><td>0.131</td></tr><tr><td>Perceived cognition</td><td>0.160 [-0.075, 0.394]</td><td>0.182</td><td>0.265</td></tr><tr><td>Behavioral intention</td><td>0.073 [-0.160, 0.305]</td><td>0.540</td><td>0.581</td></tr><tr><td>Ethical orientation</td><td>0.220 [0.027, 0.413]</td><td>0.025</td><td>0.094</td></tr><tr><td>Perceived prompting capability</td><td>0.240 [-0.036, 0.515]</td><td>0.088</td><td>0.205</td></tr></table>

Table 5 reports the 14 exploratory structural associations. Because multiple correlated predictors were entered simultaneously, each coefficient represented the unique association with the outcome after accounting for the other predictors in the same equation. Perceived cognition showed the strongest association with perceived prompting capability, $\beta = 0 . { \bar { 3 } } 9 0$ , and was the only path that remained significant after BH adjustment, adjusted $p = 0 . 0 1 2$ . Thus, higher perceived cognition was associated with greater perceived prompting capability after controlling for affective orientation, behavioral intention, and ethical orientation. This finding is consistent with AI literacy frameworks linking understanding and application with the effective and evaluative use of AI technologies (Long & Magerko, 2020; Ng et al., 2021).

Several other positive associations were evident before multiplicity adjustment but did not remain statistically significant. Behavioral intention was associated with perceived prompting capability $( \beta = 0 . 2 4 1$ $p \ = \ 0 . 0 2 7 )$ , while perceived prompting capability was associated with research-oriented assignment strategies $( \beta = 0 . 4 2 7 , p = 0 . 0 1 5 )$ . In the creative workshop scenario, affective orientation $( \beta = 0 . 2 6 3 , p =$ 0.047) and ethical orientation $( \beta = 0 . 2 2 0 , p = 0 . 0 2 5 )$ also showed positive unadjusted associations, but neither met the adjusted significance criterion.

These non-significant paths nevertheless had non-trivial point estimates. Most notably, the association between perceived prompting capability and research-oriented assignment strategies was relatively substantial, but its uncertainty and failure to survive multiplicity adjustment limited evidence for an independent effect. This pattern is plausible when correlated predictors are modeled simultaneously, as shared variance can increase uncertainty in individual coefficients and make their unique contributions difficult to distinguish (Dormann et al., 2013). Adjustment across 14 tests further raised the evidentiary threshold for individual paths (Benjamini & Hochberg, 1995). The small number of adjusted-significant associations should therefore be interpreted as limited evidence for unique conditional relationships, rather than as evidence that the domains were unrelated.

Since the data were cross-sectional and the structural specification was covariance-equivalent to the measurement model, the estimated paths should not be interpreted as directional or causal. Overall, the findings support viewing GenAI-related orientations, perceived prompting capability, and task-specific prompting strategies as closely related but conceptually distinct domains (Woo et al., 2026).

## 5 Discussion

## 5.1 From prompt refinement to evaluative self-regulation

Iterative refinement was among the most strongly endorsed aspects of both perceived prompting capability and scenario-specific strategy preferences, suggesting that students recognized the value of adjusting their interaction with GenAI after an initial response. However, the open-ended comments revealed an important distinction between revising a prompt and diagnosing what needs to be revised. Some respondents described providing extensive detail without obtaining the intended response, whereas others reported that repeated revisions made the interaction less coherent.

These comments point to prompt–intent alignment as an important challenge. Simply adding information or repeatedly modifying a prompt may not improve an interaction when the task goal, relationships among requirements, or criteria for an acceptable response remain unclear. Effective refinement therefore requires users first to identify the source of the mismatch and then decide which aspect of the prompt or interaction should be changed.

This process closely resembles self-regulated learning, particularly cycles of planning, monitoring, evaluation, and adaptation (Zimmerman, 2002; Panadero, 2017). In GenAI-supported learning, students may define the task, evaluate a generated response against explicit criteria, diagnose discrepancies, and make targeted revisions. Zamfirescu-Pereira et al. (2023) similarly showed that nonexpert users can struggle to evaluate and refine prompts systematically. The present findings therefore suggest that willingness to revise prompts should be distinguished from the ability to diagnose why an output does not align with the user's intention.

The structural findings reinforce this interpretation. Perceived cognition was the only orientation domain that retained a statistically supported unique association with perceived prompting capability after correction for multiple testing. Because this domain encompassed understanding AI concepts, applying AI tools to practical problems, and judging their suitability, the result suggests that perceived prompting capability was more closely related to perceived understanding and evaluative application than to positive attitudes or intentions alone. This interpretation is consistent with multidimensional accounts of AI literacy that integrate understanding, application, and critical evaluation (Long & Magerko, 2020; Ng et al., 2021). Nevertheless, because both constructs were self-reported and conceptually related, the association should not be interpreted as directional or causal.

## 5.2 Pedagogical implications

The scenario findings indicate that prompting competence combines transferable principles with tasksensitive adaptation. Providing relevant background and revising an initial prompt were among the most strongly endorsed strategies across both scenarios, suggesting that contextualization and iterative refinement have broad applicability. By contrast, output-format specification was more strongly endorsed for the creative task, where the nature of the required deliverable made format particularly salient. This pattern supports the view of prompting as a context-sensitive practice in which general strategies are adapted to task goals and expected outputs (White et al., 2023; Aruleba et al., 2025; Woo et al., 2026). Training could therefore introduce broadly useful principles while helping students determine when additional specifications concerning format, role, constraints, or evaluation criteria are warranted. The comparatively lower endorsement of fixed prompt templates reinforces this interpretation: templates may provide useful initial scaffolding, but they should not be treated as universally applicable solutions.

Difficulties with prompt–intent alignment also point to a role for assessment in encouraging more deliberate prompting practices. This implication is consistent with research distinguishing successful GenAI-assisted products from the development of learning, transfer, and evaluative judgment (Fan, Tang, et al., 2025). Where GenAI use is permitted, students could be asked to present selected prompts or revisions and briefly explain the reasoning behind important changes. The purpose would not be to reward elaborate or formulaic prompting, but to make visible how students interpreted the task, evaluated generated responses, and adapted their interaction. Such an approach is also consistent with constructive alignment, because evidence of prompting practice would remain tied to the disciplinary learning outcomes that the assessment is intended to develop (Biggs, 1996).

The association analyses provide further support for connecting prompting with broader AI literacy. Perceived cognition was the only orientation domain that retained a significant unique association with perceived prompting capability after adjustment for multiple testing. Because this domain encompassed students’ perceived ability to apply AI to practical problems, evaluate the suitability of AI applications, and distinguish relevant concepts, the finding suggests that prompting capability is more closely related to understanding and evaluative use than to positive attitudes toward AI alone. The association is correlational and should therefore not be interpreted causally, but it nevertheless reinforces the educational importance of developing students’ capacity to make informed judgments about GenAI use. This is particularly relevant given students’ relatively strong intention to continue using AI: an important institutional priority is not simply to facilitate access or encourage adoption, but to help students judge when, why, and how GenAI can be used appropriately in academic work.

These findings suggest that prompting instruction should be embedded within broader AI literacy rather than taught primarily as a set of technical techniques. Effective interaction with GenAI involves not only formulating instructions, but also articulating intentions, judging whether GenAI is appropriate for a task, identifying the information it requires, evaluating generated outputs, and revising prompts when necessary (Nguyen et al., 2024; Zamfirescu-Pereira et al., 2023). Prompting education should therefore develop evaluative judgment alongside operational skill, helping students interpret tasks, define criteria for acceptable responses, assess output quality, and justify revisions. Framed in this way, prompting becomes part of the wider processes of critical judgment and adaptive learning that higher education seeks to develop.

## 5.3 Limitations and future research

Several limitations constrain interpretation of the findings. First, the cross-sectional questionnaire measured perceived capability and reported preferences rather than directly observed prompting performance. Agreement with a strategy therefore did not establish how frequently or effectively students used it, and the two scenarios necessarily simplified authentic academic work.

Second, all principal domains used self-report measures collected within the same questionnaire. Their substantial correlations may partly reflect shared response tendencies or common-method effects (Podsakoff et al., 2003). Conceptual proximity between perceived cognition and perceived prompting capability may also have contributed to their association. Future scale development should further clarify construct boundaries through content validation, cognitive interviewing, and more differentiated item wording (Boateng et al., 2018).

Third, the optional comments were provided by only a small subset of participants and were illustrative rather than representative. Prompt–intent alignment should therefore be regarded as an important qualitative indication for further investigation rather than as the predominant difficulty in the full sample.

Future studies should combine self-report measures with authentic prompting tasks in which prompts, revisions, outputs, verification behavior, and final decisions can be examined directly. Longitudinal or intervention research could determine whether students improve in aligning prompts with intended goals, selecting task-appropriate strategies, diagnosing inadequate outputs, and evaluating generated information. Such work would help distinguish perceived prompting capability from demonstrated competence and clarify whether improved prompting contributes to disciplinary learning and transfer.

Overall, the findings suggest that prompting education should extend beyond fixed formulas. Its central educational value may lie in helping students translate academic intentions into task-appropriate prompts and diagnose mismatches between intended and generated outcomes. In this sense, prompting can provide a context for developing academic judgment and self-regulation rather than merely a means of obtaining better GenAI outputs.

## 6 Conclusion

This study examined students’ perceived GenAI prompting capability and reported prompting strategies. Students generally expressed confidence in providing relevant context and refining prompts after an initial response, whereas specifying response constraints received somewhat weaker endorsement. These findings suggest that contextualization and iterative refinement were prominent aspects of students’ self-perceived prompting capability, while more explicit control of output requirements appeared less consistently established.

The two scenarios further indicated that reported prompting strategies were partly sensitive to task context. Providing background information and revising prompts were strongly endorsed across both tasks, whereas output-format specification was more strongly endorsed in the creative workshop scenario and was the only matched strategy that differed significantly after multiplicity adjustment. This pattern suggests that some strategies may be broadly applicable across tasks, while others are adapted to particular task demands. Similarly, the endorsement of both concise instructions and detailed upfront information suggests that students did not necessarily view brevity and completeness as competing approaches; rather, their preferences appeared to depend on how task-relevant information was selected and organized.

Perceived cognition showed the clearest unique association with perceived prompting capability, linking higher self-assessed understanding, practical application, and evaluation of AI suitability with greater perceived prompting capability. However, the strong correlations among the domains and uncertainty surrounding most other structural paths caution against interpreting these relationships as distinct or causal influences. Instead, AI-related orientations, perceived prompting capability, and task-specific strategy preferences appear to represent closely related but non-interchangeable aspects of students’ engagement with GenAI.

The optional comments, although provided by only a minority of respondents, highlighted prompt–intent alignment as a key challenge. Some students appeared willing to revise prompts but still struggled to translate their intended goals into instructions that GenAI could interpret appropriately. University training should therefore move beyond fixed prompt templates and focus on helping students clarify task goals, specify relevant context and constraints, and diagnose misalignment when outputs do not meet the intended purpose.

## Data availability

Data will be made available on reasonable request to the corresponding author.

## Competing interest

The authors declare no competing interests.

## Ethics approval

The study protocol was approved by the University Research Ethics Review Panel of Xi’an Jiaotong-Liverpool University (XJTLU) on 8 Jun 2026. All procedures were conducted in accordance with applicable ethical guidelines and regulations required by XJTLU Policy on Ethical Conduct in Research.

## Informed consent

Informed consent was obtained from all individual participants for participation in this study. Anonymity of their identification and voluntary participation were assured. No identifiable data of participants were present within the manuscript.

## References

Aruleba, K., Sanusi, I. T., Obaido, G., Ogbuokiri, B., & Mienye, I. D. (2025). Beyond the prompt: Student strategies, ethical reflections, and learning with ChatGPT in computer science. Technology, Knowledge and Learning. https://doi.org/10.1007/s10758-025-09899-7

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerfu approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Biggs, J. (1996). Enhancing teaching through constructive alignment. Higher Education, 32, 347–364. https://doi.org/10.1007/BF00138871

Boateng, G. O., Neilands, T. B., Frongillo, E. A., Melgar-Quiñonez, H. R., & Young, S. L. (2018). Best practices for developing and validating scales for health, social, and behavioral research: A primer. Frontiers in Public Health, 6, 149. https://doi.org/10.3389/fpubh.2018.00149

Carolus, A., Koch, M. J., Straka, S., Latoschik, M. E., & Wienrich, C. (2023). MAILS—Meta AI literacy scale: Development and testing of an AI literacy questionnaire based on well-founded competency models and psychological change- and meta-competencies. Computers in Human Behavior: Artificial Humans, 1(2), 100014. https://doi.org/10.1016/j.chbah.2023.100014

Chan, C. K. Y., & Hu, W. (2023). Students' voices on generative AI: Perceptions, benefits, and challenges in higher education. International Journal of Educational Technology in Higher Education, 20, 43. https://doi.org/10.1186/s41239-023-00411-8

Dormann, C. F., Elith, J., Bacher, S., Buchmann, C., Carl, G., Carré, G., García Márquez, J. R., Gruber, B., Lafourcade, B., Leitão, P. J., Münkemüller, T., McClean, C., Osborne, P. E., Reineking, B., Schröder, B., Skidmore, A. K., Zurell, D., & Lautenbach, S. (2013). Collinearity: A review of methods to deal with it and a simulation study evaluating their performance. Ecography, 36(1), 27– 46. https://doi.org/10.1111/j.1600-0587.2012.07348.x

Fan, L., Deng, K., & Liu, F. (2025). Educational impacts of generative artificial intelligence on learning and performance of engineering students in China. Scientific Reports, 15, 26521. https://doi.org/10.1038/s41598-025-06930-w

Fan, L., & Liu, F. (2026a). The Impact of Generative Artificial Intelligence on Academic Development of Chinese Students in Humanities and Social Sciences. Education Sciences, 16(6), 814. https://doi.org/10.3390/educsci16060814

Fan, L., & Liu, F. (2026b). Impacts of generative AI skill development on academic adaptation and learning outcomes among university students, arXiv. https://doi.org/10.48550/arXiv.2610.30599

Fan, Y., Tang, L., Le, H., Shen, K., Tan, S., Zhao, Y., Shen, Y., Li, X., & Gašević, D. (2025). Beware of metacognitive laziness: Effects of generative artificial intelligence on learning motivation,

processes, and performance. British Journal of Educational Technology, 56(2), 489–530. https://doi.org/10.1111/bjet.13544

Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for assessing discriminant validity in variance-based structural equation modeling. Journal of the Academy of Marketing Science, 43, 115–135. https://doi.org/10.1007/s11747-014-0403-8

Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. Psychometrika, 30(2), 179–185. https://doi.org/10.1007/BF02289447

Hornberger, M., Bewersdorff, A., & Nerdel, C. (2023). What do university students know about Artificial Intelligence? Development and validation of an AI literacy test. Computers and Education: Artificial Intelligence, 5, 100165. https://doi.org/10.1016/j.caeai.2023.100165

Kaiser, H. F., & Rice, J. (1974). Little jiffy, mark IV. Educational and psychological measurement, 34(1), 111-117. https://doi.org/10.1177/001316447403400115

Knoth, N., Tolzin, A., Janson, A., & Leimeister, J. M. (2024). AI literacy and its implications for prompt engineering strategies. Computers and Education: Artificial Intelligence, 6, 100225. https://doi.org/10.1016/j.caeai.2024.100225

Lo, L. S. (2023). The CLEAR path: A framework for enhancing information literacy through prompt engineering. The Journal of Academic Librarianship, 49(4), 102720. https://doi.org/10.1016/j.acalib.2023.102720

Long, D., & Magerko, B. (2020). What is AI literacy? Competencies and design considerations. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems (pp. 1–16). ACM. https://doi.org/10.1145/3313831.3376727

Maxwell, S. E., & Cole, D. A. (2007). Bias in cross-sectional analyses of longitudinal mediation. Psychological Methods, 12(1), 23–44. https://doi.org/10.1037/1082-989X.12.1.23

Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data. Psychological Methods, 17(3), 437–455. https://doi.org/10.1037/a0028085

Ng, D. T. K., Leung, J. K. L., Chu, S. K. W., & Qiao, M. S. (2021). Conceptualizing AI literacy: An exploratory review. Computers and Education: Artificial Intelligence, 2, 100041. https://doi.org/10.1016/j.caeai.2021.100041

Nguyen, S., Babe, H. M., Zi, Y., Guha, A., Anderson, C. J., & Feldman, M. Q. (2024). How beginning programmers and code LLMs (mis)read each other. In Proceedings of the 2024 CHI Conference Human Factors in Computing Systems, Article 651, 1–26. https://doi.org/10.1145/3613904.3642706

Oehlert, G. W. (1992). A note on the delta method. The American Statistician, 46(1), 27-29. https://doi.org/10.1080/00031305.1992.10475842

Panadero, E. (2017). A review of self-regulated learning: Six models and four directions for research. Frontiers in Psychology, 8, 422. https://doi.org/10.3389/fpsyg.2017.00422

Podsakoff, P. M., MacKenzie, S. B., Lee, J.-Y., & Podsakoff, N. P. (2003). Common method biases in behavioral research: A critical review of the literature and recommended remedies. Journal of Applied Psychology, 88(5), 879–903. https://doi.org/10.1037/0021-9010.88.5.879

Rhemtulla, M., Brosseau-Liard, P. É., & Savalei, V. (2012). When can categorical variables be treated as continuous? A comparison of robust continuous and categorical SEM estimation methods under suboptimal conditions. Psychological Methods, 17(3), 354–373. https://doi.org/10.1037/a0029315

Tobias, S., & Carlson, J. E. (1969). Brief report: Bartlett’s test of sphericity and chance findings in factor analysis. Multivariate Behavioral Research, 4(3), 375–377. https://doi.org/10.1207/s15327906mbr0403\_8

White, J., Fu, Q., Hays, S., Sandborn, M., Olea, C., Gilbert, H., Elnashar, A., Spencer-Smith, J., & Schmidt, D. C. (2023). A prompt pattern catalog to enhance prompt engineering with ChatGPT. arXiv. https://doi.org/10.48550/arXiv.2302.11382

Woo, D. J., Wang, D., Yung, T., & Guo, K. (2026). Effects of a prompt engineering intervention on undergraduate students' AI self-efficacy, AI knowledge and prompt engineering ability: A mixed

methods study. British Educational Research Journal, 52(2), 1442–1469. https://doi.org/10.1002/berj.70087

Wu, H., & Estabrook, R. (2016). Identification of confirmatory factor analysis models of different levels of invariance for ordered categorical outcomes. Psychometrika, 81(4), 1014–1045. https://doi.org/10.1007/s11336-016-9506-0

Zamfirescu-Pereira, J. D., Wong, R. Y., Hartmann, B., & Yang, Q. (2023). Why Johnny can't prompt: How non-AI experts try (and fail) to design LLM prompts. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, Article 437, 1–21. https://doi.org/10.1145/3544548.3581388

Zimmerman, B. J. (2002). Becoming a self-regulated learner: An overview. Theory Into Practice, 41(2), 64–70. https://doi.org/10.1207/s15430421tip4102\_2