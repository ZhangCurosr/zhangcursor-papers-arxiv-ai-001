# ISSUELIGHTER: An Issue Type-Aware LLM Agent for Issue Report Validity Assessment

Jiawei Ran, Yuxia Zhang<sup>†</sup>, Ziyang Zhao, Hui Liu

School of Computer Science & Technology,

Beijing Institute of Technology,

Beijing, China

{jiaweiran, yuxiazh, ziyangzhao, Liuhui08}@bit.edu.cn

Abstract—Invalid issue reports are prevalent in open source software (OSS) projects, imposing substantial burdens on maintainers who already face a high risk of burnout. The widespread adoption of large language models (LLMs) in software development has further exacerbated this problem. Prior research has primarily focused on identifying invalid reports for specific issue categories (e.g., bug and wontfix), leaving the broader landscape of OSS issue types largely unexplored and limiting the generalizability of existing approaches. Moreover, existing methods provide little support for explaining or validating their predictions, reducing their practical usefulness for maintainers. With the rise of LLM-based agents, it remains unclear whether a comprehensive understanding of the causes of invalidity across all issue types can improve validity assessment performance while producing well-grounded rationales. To this end, we first manually annotated the validity status of 520 issue reports from three OSS projects, i.e., pandas, scikit-learn, and mypy, resulting in 257 invalid reports and 263 valid ones. Then, we conducted a thematic analysis on these invalid reports and identified 13 causes of invalidity that are closely specific to issue types. Based on these insights, we propose ISSUELIGHTER, an agent-based framework with a type-aware validity assessment strategy that dynamically executes commands. Experimental results show that ISSUELIGHTER consistently and substantially outperforms all baseline methods, with an average F1 improvement of over 20 percentage points. Ablation studies confirm that both the type-aware strategy and the taxonomy of causes of invalidity contribute to the performance gains. Qualitative analysis further demonstrates that ISSUELIGHTER generates rationales grounded in accurately identified causes of invalidity and supported by evidence. Moreover, leveraging these rationales, we challenged maintainers’ original decisions in four disputed cases, and all received approval, highlighting the practical value of our approach.

Index Terms—Open source software, Issue report validity assessment, LLM agent, Issue type

## I. INTRODUCTION

In the maintenance of open source software (OSS), issue reports serve as the primary communication channel between users and development teams [1]. However, a significant number of submitted issue reports are invalid, i.e., the claimed problem or request is itself unfounded or unreasonable, and are ultimately marked as not-planned, etc. While these reports do not lead to any project changes, maintainers still invest substantial time reading, diagnosing, and responding to them. This cost is far from negligible. A study of 279 GitHub projects revealed that developers take an average of five months to close a wontfix issue [2]. Research on deep learning compilers further showed that false-positive reports (accounting for 41.7% of all reports) require comparable handling time to real issue reports [3]. This phenomenon is also prevalent in software developed in companies [4], [5]. Maintainers are already well-known for their burnout in the numerous requests they receive, e.g., issue reports, code reviews, and support requests [6]. Recently, as LLMs make it effortless to generate plausible issue reports, OSS projects are experiencing an increasing influx of AI-generated submissions, many of which require substantial manual verification despite ultimately being invalid or non-actionable [7]<sup>1</sup>. These findings highlight the urgent need for automated approaches to reduce the effort spent on invalid reports.

To address the challenge of invalid reports, the research community has proposed various automated approaches, including feature engineering-based classifiers [8], [9], traditional machine learning methods [2], and deep learning models [10], [11]. Although these approaches achieve promising performance, they only produce a binary prediction (valid or invalid). In practice, however, maintainers also need to understand why an issue is considered invalid to facilitate effective communication with reporters and subsequent issue handling. Existing studies have investigated the causes of invalidity, but almost exclusively for issue reports labeled as bugs [3], [12]–[14] and wontfix [2], [15]. Building on such taxonomies, Huang et al. [3] employed LLMs to identify invalid bug reports and generate corresponding rationales. However, existing studies remain limited to a small subset of issue categories, leaving the causes of invalidity across the broader landscape of OSS issue types largely unexplored. Moreover, their rationales lack supporting evidence for validity assessment, making it difficult for maintainers to verify and act upon the results.

The rise of LLM agent mechanisms offers a new direction for overcoming this limitation. Unlike traditional static approaches, agents can actively invoke external tools, forming a “think-act-observe” loop [16]–[18]. This provides the technical foundation for enabling executable validation and leveraging additional resources. To this end, this paper aims to move the needle by characterizing root causes of all the issue types, then guiding the LLM agent to identify invalid issue reports with precise rationales.

We first empirically investigate the root causes of invalid issue reports across all issue types, with particular attention to type-specific causes. Specifically, we ask RQ1: What are the root causes of invalid issue reports across different issue types? We conducted a thematic analysis on 257 invalid reports from three popular OSS projects (i.e., pandas, scikitlearn, and mypy). We identified 13 causes of invalidity across six issue types, i.e., bug, enhancement, documentation, continuous integration (CI), installation, and spam. Furthermore, we found that different issue types exhibit fundamentally different patterns of causes of invalidity.

Based on the findings from RQ1, we propose ISSUE-LIGHTER, an agent-based framework with a validity assessment strategy tailored to each issue type that dynamically executes commands. Specifically, we ask RQ2: What is the performance of ISSUELIGHTER compared to the baselines? We answer this question by evaluating ISSUELIGHTER and the other baselines on our well-annotated issue reports. The results show that ISSUELIGHTER consistently outperforms baseline methods across three LLMs, achieving 96.2% accuracy and 96.0% F1-score on ChatGPT in particular. We also conducted a qualitative analysis of the rationales given by ISSUELIGHTER. We found that its rationales align closely with the causes of invalidity we annotated in RQ1 and are grounded in empirical evidence rather than textual inference. Furthermore, based on the rationale given by ISSUELIGHTER, we challenged maintainers’ original decisions in four disputed cases, all of which were approved, highlighting the practical value of our approach.

The main contributions of this paper are as follows:

• A comprehensive taxonomy of causes of invalidity across six issue types.

• ISSUELIGHTER, an automated framework for issue report validity assessment that achieves substantial performance gains over baselines.

• A publicly available benchmark of 520 manually annotated issue reports with validity labels and root causes of invalidity.

## II. RELATED WORK

## A. Characterization of Invalid Issue Reports

Understanding the causes of invalid issue reports is key to automated detection. Sun et al. manually analyzed over 600 invalid bug reports and found that testing errors, misunderstandings of functionality, and faults introduced by external libraries were the primary causes [12]. Focusing specifically on non-reproducible reports, Joorabchi et al. identified six categories of root causes, with inter-bug dependencies and environmental differences being the most prevalent [13]. Extending this line of inquiry, Rahman et al. further investigated non-reproducible bug reports and identified 11 key factors contributing to non-reproducibility, including missing information, ambiguous specifications, and third-party defects [14]. Huang et al. studied invalid bug reports in deep learning compilers, revealing that such reports primarily stem from incorrect environment configuration, incorrect usage, and misunderstanding of compiler features or limitations [3].

Regarding issues labeled as wontfix, Wang et al. analyzed 600 wontfix reports from three open-source projects on Bugzilla and identified 12 categories of reasons, including unsupported features, non-real defects, and lack of response [15]. Extending this work to GitHub, Panichella et al. constructed a dual taxonomy of opening and closing motivations, revealing that the most common reasons for wontfix decisions are feature requests deemed unnecessary and erroneous bug reports [2].

These studies establish our foundational understanding of the causes of invalid issue reports and provide the basis for subsequent automated approaches. However, they primarily focus on two categories of invalid reports: bug-type invalid reports and issues labeled as wontfix, leaving the causes of invalidity for other issue types underexplored. To address this gap, we conducted a broader analysis covering invalid reports of all types to pursue a more comprehensive taxonomy of causes of invalidity.

## B. Machine Learning-based Classification Methods

To automate the identification of invalid reports, researchers have explored various machine learning approaches. Zanetti et al. built classifiers based on collaboration networks, extracting features such as centrality measures from the network structure [8]. Fan et al. expanded the feature set to 33 dimensions across five categories, i.e., reporter experience, collaboration network, completeness, readability, and text, significantly improving classification performance using random forests [9]. Panichella et al. employed traditional machine learning classifiers on textual features, achieving F-measures up to 0.93 and demonstrating that effective prediction is possible using only titles and descriptions [2].

With the advent of deep learning, He et al. proposed convolutional neural networks that classify issues based solely on titles and descriptions, eliminating the need for manual feature engineering [10]. Meng and Visser further leveraged transformer-based models such as BERT for validity classification, employing explanation methods such as LIME and SHAP to reveal key features influencing classification decisions [11]. Evaluations of various machine learning techniques in industrial settings have shown that simple logistic regression models can meet practical needs in terms of both accuracy and interpretability [5].

However, these approaches face two limitations. First, they only output binary results without providing natural language rationales. Second, they heavily rely on high-quality training data and struggle to generalize across different projects [19], [20]. These limitations call for a more explainable and generalizable approach.

## C. LLM-based Classification Methods

The emergence of LLMs offers new possibilities for addressing the aforementioned limitations. Unlike traditional machine learning methods, LLMs can not only output classification results but also generate natural language rationales. Recent studies have explored applying LLMs to invalid bug report classification. Huang et al. systematically evaluated four prompting strategies and found that few-shot prompting achieves 73% accuracy while generating explanations deemed accurate in 85.2% of cases, demonstrating the potential of LLMs for both classification and explanation [3]. Concurrently, Dinc¸ and Tuz¨ un conducted a comprehensive com-¨ parison of classical ML, fine-tuned transformers, and LLMs on bug report classification [21]. Their fine-tuned RoBERTa achieved the highest F1, outperforming few-shot LLMs and their Judge LLM.

However, these approaches still remain at the level of static text analysis and lack the capability for validity assessment through execution. Moreover, LLMs are limited by their training cutoff—information updated after training, such as project documentation and source code changes, cannot be covered by built-in model knowledge alone [22].

Meanwhile, LLM-driven agents have demonstrated emerging potential in software engineering tasks, such as program repair [23], code generation [24], and test automation [25]. Torun et al. further envisioned an AI-powered bug tracking framework where LLM agents handle tasks, including validity check [26]. These advances inspire us to explore the application of agents in issue report validity assessment.

## III. DATASET CONSTRUCTION

We describe how we collected and preprocessed the dataset to answer RQ1 and RQ2.

Study Objects: We selected three popular open-source Python projects hosted on GitHub as our data sources: pandas [27], scikit-learn [28], and mypy [29]. The rationale for this selection is threefold. First, these projects have a significant influence on the software engineering community [30], [31], featuring active issue tracking systems that provide rich data with both valid and invalid samples. Second, they follow established issue management practices, with maintainers clearly documenting final resolutions (e.g., closing reasons and associated pull requests), providing valuable practical evidence for establishing reliable ground truth. Third, these projects serve different technical domains: pandas for data analysis, scikit-learn for machine learning, and mypy for static type checking. This allows us to examine whether invalid issue patterns vary across software domains and enhance the generalizability of our results.

Data Collection: We collected issue reports from the GitHub repositories of the three selected projects using the GitHub REST API [32], covering reports that were created since August 8, 2025 and closed by March 8, 2026. This window was chosen to ensure a sufficiently large and recent sample of issue reports for analysis. For each issue report, we extracted its ID, reporter, title, body, creation time, labels, comments, edit history for title and body, and associated pull requests (if they exist).

Data Filtering: To ensure the quality and representativeness of the experimental data, we applied the following filtering criteria:

1) Exclude script-generated issue reports: We excluded reports generated by official project scripts (e.g., CI failure notifications), because these are set by the maintenance team. Only scikit-learn had such a script account (scikit-learn-bot), which submitted 45 reports.

2) Exclude maintainer-created issue reports: Issue reports submitted by project maintainers with the “member” or “collaborator” role in the repository are typically used for tracking project maintenance activities, such as release notes and resolution discussions. There were 49 such reports in pandas, 51 in scikit-learn, and 32 in mypy.

3) Exclude issue reports without feedback: For issue reports with neither comments nor associated pull requests, we cannot reliably determine their validity status. There were 8 such reports in pandas, 8 in scikit-learn, and 4 in mypy.

4) Exclude duplicate issue reports: Issue reports marked as “duplicate” are excluded. This study primarily focuses on examining the issue content to assess its validity. There were 23 such reports in pandas, 9 in scikit-learn, and 36 in mypy.

Data Preprocessing: During the lifecycle of an issue, the title or body may be edited. To ensure that our experiments evaluate the user-submitted content rather than maintainermodified versions, we used the collected edit history to reconstruct the version before any maintainer intervention. This preprocessing avoids potential data leakage issues and ensures fairness in evaluation [33].

Dataset Annotation. After data filtering and preprocessing, we obtained 520 issue reports. Two researchers (the first and third authors), each with four years of software development experience and two years of open-source contribution experience, independently annotated the validity of each report. Based on manual examination of comments and relevant pull requests, each report was labeled as Valid if the core claim of the report was acknowledged by community maintainers, and as Invalid otherwise. The two annotators achieved a Cohen’s kappa of 0.85, indicating substantial agreement [34], [35]. Disagreements were resolved through discussion and, when necessary, arbitration by the second author, who has over ten years of research experience in open-source software maintenance. Table I provides an overview of the constructed dataset.

## IV. RQ1: WHAT ARE THE ROOT CAUSES OF INVALID ISSUE REPORTS ACROSS DIFFERENT ISSUE TYPES?

## A. Method

1) Issue Type Annotation: To investigate whether the causes of invalid reports are correlated with issue types, we first classified the type of each issue. We observed that issue labeling practices vary across projects, and some reports lack labels altogether, making it unreliable to rely solely on existing labels for type classification. Therefore, we followed a unified classification schema based on the commonly used issue types in open-source software development, which typically include bug, enhancement, and documentation [36], [37]. The first and third authors categorized each report into one of the above types by considering the report’s existing labels (when available), title, body, and comments. During this process, we observed that several issue reports fell outside these three types and further identified three additional types: CI, installation, and spam. We define the three new issue types as follows:

TABLE I. Dataset Overview
<table><tr><td>Project</td><td>Total</td><td>Valid</td><td>Invalid</td><td>Invalid Ratio</td></tr><tr><td>pandas</td><td>209</td><td>117</td><td>92</td><td>44.0%</td></tr><tr><td>scikit-learn</td><td>118</td><td>56</td><td>62</td><td>52.5%</td></tr><tr><td>mypy</td><td>193</td><td>90</td><td>103</td><td>53.4%</td></tr><tr><td>Total</td><td>520</td><td>263</td><td>257</td><td>49.4%</td></tr></table>

• CI: issue reports are related to continuous integration systems, such as pipeline failures or CI configuration defects.

• Installation: issue reports are related to the project installation process, such as missing pre-built packages or source code build failures.

• Spam: issue reports that contain no substantive content related to the project and thus cannot be meaningfully classified into any specific issue type. For example, a report may consist solely of a title like “bug” with no description, or contain only auto-generated template text without any user-provided information.

We therefore incorporated these three types into our classification schema, resulting in a final set of six issue types. The two annotators achieved a Cohen’s kappa of 0.88, indicating high consistency [34], [35]. Disagreements were resolved through discussion, with the second author arbitrating when necessary.

2) Identification of Causes of Invalidity: To identify the causes of invalidity, we conducted a thematic analysis on the 257 invalid issue reports, analyzing their titles, bodies, and subsequent comments. Thematic analysis is a qualitative research method used to identify, analyze, and report patterns (themes) within data [38], [39]. We describe the detailed process of how we applied this approach below:

(1) Familiarizing with the data and generating initial codes: The two researchers independently read through all 257 invalid issue reports, including titles, bodies, and comments, to establish a holistic understanding of the data. Through a comprehensive review of the collected data, we extracted relevant phrases from each issue and transformed them into concise labels as initial codes.

(2) Searching for themes: Within our established set of codes, we frequently observed overlapping or converging codes that addressed similar aspects. For instance, we found that codes such as “does not align with the primary goal” and “better suited for a third-party library” both suggest that the request falls outside the project’s scope. We therefore merged them into an initial theme “Out of scope”. Accordingly, we meticulously categorized these codes into groups, assigning each group a theme that summarizes their combined meaning. These themes were designed to capture the core significance of the codes within each group.

(3) Reviewing themes: To refine the relationships among themes, we examined each candidate theme for internal consistency and external heterogeneity. We combined related themes as sub-themes into comprehensive themes, merging, splitting, or reorganizing themes with overlapping meanings or unclear hierarchical relationships.

(4) Defining and naming themes: When the analysis of 20 consecutive new issue reports no longer generated new codes or themes, we recognized that theoretical saturation had been reached and ceased theme organization activities [40], [41]. In this step, we defined and named the final list of themes by confirming their exact meanings and ensuring they were named understandably and accurately.

To mitigate the influence of individual subjectivity on the analysis results, the two researchers adopted an approach involving independent coding followed by cross-validation. In cases of coding disagreements, the two researchers first attempted to reach consensus through discussion. If consensus could not be achieved after thorough discussion, the second author, who is familiar with thematic analysis and the research objectives, was invited to make the final judgment. We conducted six rounds of face-to-face discussion meetings, totaling approximately eight hours.

## B. Results

Through thematic analysis, we identified 13 causes of invalidity. Table II presents these causes along with their detailed explanations and their distribution across the three studied projects. We can see that, despite differences in domain and scale among pandas, scikit-learn, and mypy, the same causes of invalidity appear across all three projects, indicating their potential for cross-project generalization.

A prior study [2] has identified 26 closing reasons for wontfix issues. When compared with these reasons, most find counterparts in our framework, such as user misunderstanding, out of scope, and existing alternative. However, three causes we repeatedly observed are not covered by their taxonomy: adequate documentation, misplaced report, and non-informative. These additions stem from our dataset covering a broader range of issue types, such as documentation and spam.

Further, we analyzed the relationship between the identified causes of invalidity and issue types. The correspondence between them is shown in Figure 1. We can observe that the distribution of causes of invalidity exhibits distinct patterns across different issue types:

Bug type has multiple causes of invalidity, dominated by “User misunderstanding”, followed by “Unable reproduction”, “Misplaced report”, “Out of date”, and “inherent limitation”.

TABLE II. Root Cause Taxonomy and Project Distribution of Invalid Issue Reports
<table><tr><td>Cause</td><td>Explanation</td><td>pd</td><td>sk</td><td>mp</td></tr><tr><td>User misunderstanding</td><td>The issue stems from user misunderstanding of the project&#x27;s expected behavior, such as intentional code behavior, correct documentation, or reasonable project configuration.</td><td>26</td><td></td><td>56</td></tr><tr><td>Unable reproduction</td><td>The issue cannot be reproduced based on the information provided in the report.</td><td>11</td><td></td><td>10</td></tr><tr><td>Misplaced report</td><td>The issue stems from a third-party project, not in the project itself.</td><td>12</td><td></td><td>8</td></tr><tr><td>Out of date</td><td>The issue has been fixed in subsequent versions.</td><td>4</td><td></td><td>8</td></tr><tr><td>Inherent limitation</td><td>Due to inherent constraints in the project&#x27;s technology stack, the issue cannot be resolved.</td><td>1</td><td></td><td>5</td></tr><tr><td>Existing alternative</td><td>The request can already be achieved through other means.</td><td>13</td><td></td><td>2</td></tr><tr><td>Out of scope</td><td>The request does not align with the project or is better implemented as an external project.</td><td>10</td><td>23</td><td>1</td></tr><tr><td>Backward incompatibility</td><td>The requested change will cause backward incompatibility.</td><td>2</td><td></td><td>2</td></tr><tr><td>Poor ROI</td><td>The request is too niche or too complex to implement with limited utility.</td><td>1</td><td></td><td>4</td></tr><tr><td>Adequate documentation</td><td>Current documentation already clearly covers the relevant content.</td><td>2</td><td></td><td>1</td></tr><tr><td>Unsupported version</td><td>The issue stems from attempting to install officially unsupported versions.</td><td>1 1</td><td></td><td>1</td></tr><tr><td>Environment misconfiguration</td><td>The issue stems from user-specific permissions, dependency conflicts, or incorrect environment settings.</td><td></td><td></td><td>3</td></tr><tr><td>Non-informative</td><td>The issue is extremely incomplete, lacking sufficient information to be understood.</td><td>8</td><td></td><td>2</td></tr></table>

Note: pd = pandas, sk = scikit-learn, mp = mypy.

![](images/9cd0544745a023e3af22e998238e1b6294785deb9e13973262f4d93ef669773c.jpg)  
Fig. 1: Correspondence Between Causes of Invalidity and Issue Types

This indicates that bug reports are most often deemed as invalid due to the reporter’s misunderstanding of the expected behavior, or the reproduction example is missing or incorrect, rather than an actual defect in the project itself.

Enhancement type has the most diverse set of causes of invalidity, with the most common being “Existing alternative” and “Out of scope”, followed by “Backward incompatibility” and “Poor return on investment (ROI)”. This suggests that enhancement reports are deemed invalid primarily because the proposed functionality is inconsistent with the project’s core goals, already achievable through other means, or lacks implementation value from a maintenance cost perspective.

Documentation type shows a relatively concentrated pattern, mainly consisting of three categories: “User misunderstanding”, “Adequate documentation”, and “Out of scope”. This indicates that such invalid reports primarily stem from reporters’ misunderstanding of the documentation, overlooking existing documentation, or misalignment with the scope of the documentation.

Installation type includes two causes of invalidity: “Unsupported version” and “Environment misconfiguration”. Such reports are deemed invalid because users either use officially unsupported installation packages or have specific configuration issues in their local environment.

CI type has a single cause of invalidity, namely “User misunderstanding”, with this report stemming from the user’s misunderstanding of the expected behavior of the CI system.

Spam type has a single cause of invalidity, namely “Noninformative”, with such reports being extremely incomplete and exhibiting clear characteristics.

This finding reveals a strong correlation between issue types and causes of invalidity. Specifically, “User misunderstanding” and “Unable reproduction”, which are common in bug reports, rarely appear in enhancement reports; conversely, “Out of scope” and “Existing alternative”, which are common in enhancement reports, do not appear in bug reports.

Summary for RQ1: We identified 13 causes of invalidity from 257 invalid reports, clarifying the specific reasons that lead issue reports to be deemed invalid. The distribution of causes of invalidity varies significantly across issue types, revealing a strong correlation between issue types and these causes.

## V. RQ2: WHAT IS THE PERFORMANCE OF ISSUELIGHTER COMPARED TO THE BASELINES?

Inspired by the findings of RQ1, we thoroughly considered the causes of invalidity and types of issue reports to design an automated approach of issue report validity assessment, noted as ISSUELIGHTER.

## A. Approach Design

Figure 2 shows the overall architecture of ISSUELIGHTER. Our approach takes issue report information, i.e., title, body, and creation time, as input, validates it through an interactive loop with a type-aware validity assessment strategy, and ultimately outputs the validity judgment of the issue report along with its rationale. The detailed design of ISSUELIGHTER is described as follows.

![](images/fdeda1906e5838549a7b52ed15f04b9bb66bebb2e74acf92a38eb660efbc7cfc.jpg)  
Fig. 2: Overall Architecture of ISSUELIGHTER

1) Interactive Validation Loop: Based on the 13 causes of invalidity identified in RQ1, we found that the determination of most causes requires information beyond the issue report itself and varies among issue types. For example, “Unable Reproduction” requires actual code execution to confirm, while “User Misunderstanding” requires consulting official documentation to assess whether the expected behavior is reasonable. These determinations collectively point to a core requirement: acquiring diverse information beyond issue reports. LLM-based agents naturally possess this capability [42]. Based on these observations, we designed ISSUELIGHTER as an LLM-based agent with an interactive validation loop as its core operating mechanism, empowering it with the ability to execute commands.

ISSUELIGHTER operates through a cycle of “analysis and decision → command execution → result collection → continue analysis and decision” [18], with all interaction history accumulated as context for subsequent analysis. ISSUE-LIGHTER follows a structured output format: (1) an analysis of the current state; (2) a status indicator (CONTINUE or END); and (3) the corresponding content. If the status is CONTINUE, the agent outputs executable bash commands; if END, it outputs the final validity judgment along with its rationale. This structured output enables the agent to iteratively gather evidence and autonomously terminate the loop when sufficient information has been collected.

To prevent infinite loops during validation, we set a maximum number of interaction rounds as a safety upper bound. In preliminary experiments on 105 randomly sampled reports (35 from each project), over 95% of validations were completed within 9 interaction rounds. Accordingly, we set the upper bound to 9. When ISSUELIGHTER determines that sufficient evidence has been gathered to reach a conclusion, it autonomously ends the loop; if the maximum number of interaction rounds is reached, it forcibly terminates the loop and renders a judgment based on the evidence collected so far. In our main experiments, the average number of interaction rounds for ChatGPT, Grok, and Qwen is 2.8, 4.1, and 4.4, respectively, with the corresponding 95th percentiles at 6, 8, and 8. This indicates that the 9-round upper bound is sufficient for the vast majority of validations.

To ensure isolation and reproducibility of the validation environment, each validation session runs in an independent Docker container [43]. Our Docker image contains the full source code of the target project and is equipped with Git for version switching, Conda for environment management, pip for dependency installation, and grep for code and documentation retrieval. Since not all issue reports specify the affected code version, we use the GitHub REST API to retrieve the latest main-branch commit preceding each report’s creation time as a fallback version reference. During validation, IS-SUELIGHTER checks out the reported version when available; otherwise, it uses the retrieved commit for creating a Conda environment and installing the required dependencies.

2) Type-Aware Validity Assessment Strategy: Furthermore, the findings of RQ1 motivate us to consider that the information dimensions required for validity assessment vary across issue types. An ideal validity assessment strategy should adopt differentiated approaches tailored to each issue type, rather than a one-size-fits-all approach. Based on this, we further analyzed the determination requirements of each cause of invalidity under each type and designed targeted validation steps for each issue type. ISSUELIGHTER first classifies each issue report into one of six types based on its title and body to select a validity assessment strategy. In case ISSUELIGHTER considers the report to lie between multiple types, it will draw on multiple strategies as needed until sufficient evidence is gathered. The detailed design is as follows.

For bug type, reproducibility is the foundation of validity [44], [45] and ranks among the top causes of invalidity as shown in Figure 1. Therefore, the validity assessment process first attempts to reproduce the issue based on the information provided in the report. If reproduction fails, the issue is judged as “Unable reproduction”. Otherwise, ISSUELIGHTER further verifies whether the issue still exists in the latest version: if it has been fixed, the issue is judged as “Out of date”; if it persists, the agent conducts an in-depth root cause analysis, combining code behavior and documentation to determine whether the issue stems from user misunderstanding of expected behavior, belongs to a third-party project, or is an inherent limitation of the project itself.

For enhancement type, validity assessment centers on assessing the reasonableness and feasibility of the request. Since enhancement requests are typically motivated by a perceived deficiency, the process first verifies whether the claimed deficiency actually exists. If not, the request lacks a basis and requires no further evaluation. Next, ISSUELIGHTER checks whether existing features already provide an alternative solution, as such redundancy would render the request unnecessary. Finally, it evaluates the request’s alignment with the project’s design principles and maintenance considerations, identifying potential issues such as being out of scope, having poor ROI, or introducing backward incompatibility.

For documentation type, the core lies in verifying the correctness and completeness of the documentation. If the documentation involves functional behavior (such as API usage examples or code comments), ISSUELIGHTER executes the relevant code to verify the actual behavior. Otherwise, it directly retrieves the relevant documentation. Through comparison, if the documentation is correct but the user’s expectation deviates, it is judged as “User misunderstanding”. If the documentation accurately covers the relevant content, the issue is judged as “Adequate documentation”. If the documentation contains errors or omissions, further judgment is required to determine whether it falls outside the scope of documentation maintenance.

For CI type, the core lies in identifying the cause of failure. If the report contains a CI log link, ISSUELIGHTER will first extract the content to obtain detailed error information; if not, it relies on the information provided in the report. Then, it analyzes the specific failure cause and consults the project’s CI configuration to comprehensively determine whether the failure stems from user misunderstanding or issues with the CI system itself.

For installation type, the core lies in distinguishing whether the source of the issue is the project itself or the user. If the report is related to pre-built packages, ISSUELIGHTER checks whether the package used in the report falls within the project’s official support scope. If the report is related to the source build, it consults the project’s installation guide to determine whether the user has met the installation requirements, and checks for user-specific issues, such as permissions or dependency conflicts.

For spam type, reports are characterized by extremely incomplete content and are easy to identify. Therefore, ISSUE-LIGHTER only assesses the report content and directly judges it as invalid upon recognition, without further processing.

## B. Experimental Setup

1) Baselines: We compare ISSUELIGHTER against three representative state-of-the-art baselines covering both prompting-based and learning-based approaches. For prompting-based methods, Huang et al. systematically evaluated four strategies, i.e., zero-shot, few-shot, chain-ofthought, and retrieval-augmented generation. Their results showed that enhanced zero-shot achieved the highest recall and F1-score, while few-shot achieved the highest precision and accuracy. Chain-of-thought limited LLM adaptability due to rigid stepwise reasoning, and retrieval-augmented generation suffered from noisy and version-misaligned documentation retrieval [3]. We therefore include only enhanced zero-shot and few-shot as baselines. As a representative learning-based baseline, we include the fine-tuned RoBERTa model proposed by Dinc¸ and Tuz¨ un [21], which reported the best performance¨ among traditional classifiers. These three baselines provide comprehensive coverage of the strongest existing approaches for issue report validity assessment.

The enhanced zero-shot and few-shot baselines adopt the same validity criteria as ISSUELIGHTER. For few-shot prompting, we follow the dynamic selection strategy of Huang et al. [3]. Specifically, for each project, we constructed a balanced seed pool of 26 examples selected from the preliminary experiment set: 13 invalid cases, each representing one of the 13 causes of invalidity identified in RQ1, and 13 valid cases covering issue types. These 78 examples were excluded from the final evaluation set, leaving 442 reports for all testing. For each test report, we randomly sampled one invalid and one valid example from the seed pool as in-context examples. For the fine-tuned RoBERTa baseline, we follow the finetuning setup of Dinc¸ and Tuz¨ un [21]. We collected 2,000 issue¨ reports per project created before August 8, 2025, and labeled each as valid if it was resolved with a merged pull request, and as invalid otherwise. We then used this data to fine-tune RoBERTa for each project. The fine-tuned model was then evaluated on the same test set used for ISSUELIGHTER and the prompting-based baselines.

2) LLM Models: We implement both ISSUELIGHTER and the prompting-based baselines using three mainstream large language models: ChatGPT, Grok, and Qwen. These models represent a diverse set of state-of-the-art LLMs with strong performance on natural language understanding and coderelated tasks. ChatGPT (gpt-5) represents leading closedsource commercial models [46], Grok (grok-4) represents models optimized for reasoning capabilities [47], and Qwen (qwen3-235b-a22b-instruct-2507) represents powerful opensource alternatives [48]. All model versions were released prior to our data collection period (August 8, 2025), ensuring that the models have not been exposed to the evaluation data during training. For all LLM calls in both methods, we set the temperature to 0 to maximize reproducibility by having the model select the most likely token at each step. All other hyperparameters were kept at their API default values [49].

3) Evaluation Metrics: Since our primary goal is to identify invalid issue reports, we treat invalid as positive and valid as negative. For each method, we report accuracy, precision, recall, and F1-score.

## C. Results

We evaluate ISSUELIGHTER from two perspectives: (1) quantitatively assess its classification performance, including overall comparison with the baselines, performance on each issue type, ablation studies to assess the contribution of each component, and cost-runtime analysis; (2) to further examine whether the type-aware validity assessment strategy proved effective, we conducted a qualitative analysis from three aspects: consistency between model-generated rationales and human-annotated causes of invalidity for correctly identified invalid reports, rationale characteristics for correctly identified valid reports, and failure patterns of misclassified cases.

1) Quantitative Evaluation: Table III shows the overall performance comparison between ISSUELIGHTER and the three baseline methods. The experimental results demonstrate that ISSUELIGHTER significantly outperforms all baselines across all evaluation metrics, and this advantage remains consistent across all three LLMs. Taking ChatGPT as an example, ISSUELIGHTER achieves an accuracy of 96.2% and an F1-score of 96.0%, compared to enhanced zero-shot (78.5% accuracy, 74.5% F1) and few-shot (78.3% accuracy, 76.4% F1). On Grok, ISSUELIGHTER achieves an F1 of 86.6%, versus 55.5% and 57.8% for enhanced zero-shot and few-shot, respectively. On Qwen, ISSUELIGHTER achieves an F1 of 83.2%, while enhanced zero-shot and few-shot achieve 56.4% and 64.4%, respectively. The fine-tuned RoBERTa achieves an F1 of 65.9%, lower than ISSUELIGHTER on all three models. These results confirm that our agent-based approach effectively overcomes the limitations of static text analysis, providing a more reliable solution for issue report validity assessment.

TABLE III. Performance Comparison of ISSUELIGHTER and Baseline Methods
<table><tr><td>Model</td><td>Method</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td rowspan="4">ChatGPT</td><td>ISSUELIGHTER</td><td>96.2%</td><td>98.6%</td><td>93.6%</td><td>96.0%</td></tr><tr><td>Enhanced zero-shot</td><td>78.5%</td><td>89.7%</td><td>63.8%</td><td>74.5%</td></tr><tr><td>Few-shot</td><td>78.3%</td><td>84.4%</td><td>69.7%</td><td>76.4%</td></tr><tr><td>ISSUELIGHTER</td><td>88.0%</td><td>96.6%</td><td>78.4%</td><td>86.6%</td></tr><tr><td rowspan="3">Grok</td><td>Enhanced zero-shot</td><td>68.8%</td><td>93.5%</td><td>39.4%</td><td>55.5%</td></tr><tr><td>Few-shot</td><td>68.3%</td><td>84.2%</td><td>44.0%</td><td>57.8%</td></tr><tr><td>ISSUELIGHTER</td><td>85.3%</td><td>95.3%</td><td>73.9%</td><td>83.2%</td></tr><tr><td rowspan="3">Qwen RoBERTa</td><td>Enhanced zero-shot</td><td>68.6%</td><td>89.1%</td><td>41.3%</td><td>56.4%</td></tr><tr><td>Few-shot</td><td>71.7%</td><td>85.0%</td><td>51.8%</td><td>64.4%</td></tr><tr><td></td><td>65.6%</td><td>66.2%</td><td>65.6%</td><td>65.9%</td></tr></table>

We further investigate the performance of ISSUELIGHTER across different issue types on three models. Table IV presents the accuracy and F1-scores for each issue type on Chat-GPT, Grok, and Qwen. The results show that ISSUELIGHTER achieves robust performance on the three most common types, i.e., bug, enhancement, and documentation, across all models. For the remaining types (CI, installation, and spam), despite the limited sample sizes, the results remain promising.

TABLE IV. Performance by Issue Types Across Models
<table><tr><td rowspan="2">Type</td><td rowspan="2">N</td><td colspan="2">ChatGPT</td><td colspan="2">Grok</td><td colspan="2">Qwen</td></tr><tr><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td></tr><tr><td>Bug</td><td>296</td><td>96.6%</td><td>96.9%</td><td>86.5%</td><td>88.5%</td><td>87.2%</td><td>89.1%</td></tr><tr><td>Enh.</td><td>65</td><td>93.8%</td><td>89.5%</td><td>90.8%</td><td>85.0%</td><td>81.5%</td><td>73.9%</td></tr><tr><td>Doc.</td><td>58</td><td>96.6%</td><td>97.7%</td><td>89.7%</td><td>93.0%</td><td>77.6%</td><td>85.4%</td></tr><tr><td>CI</td><td>6</td><td>100%</td><td>100%</td><td>83.3%</td><td>90.9%</td><td>83.3%</td><td>90.9%</td></tr><tr><td>Inst.</td><td>5</td><td>80.0%</td><td>66.7%</td><td>100%</td><td>100%</td><td>80.0%</td><td>66.7%</td></tr><tr><td>Spam</td><td>12</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr></table>

Ablation Study. In RQ1, we have two findings: a taxonomy of 13 causes of invalidity and a strong correlation between these causes and issue types. Based on these, ISSUELIGHTER incorporates two components: the taxonomy as explicit judgment criteria and a type-aware validity assessment strategy designed on the observation that different issue types exhibit distinct patterns of invalidity. To assess the contribution of each component, we designed two ablation variants: (1) w/o

Strategy, which removes the type-aware validity assessment strategy while retaining the taxonomy; and (2) w/o Strategy & Taxonomy, which removes both components, retaining only the agent’s execution capability with a generic definition of validity.

TABLE V. Ablation Study Results
<table><tr><td>Model</td><td>Variant</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td rowspan="2">ChatGPT</td><td>w/o Strategy</td><td>82.5%</td><td>84.4%</td><td>79.4%</td><td>81.8%</td></tr><tr><td>w/o Strategy &amp; Taxonomy</td><td>76.5%</td><td>95.2%</td><td>55.0%</td><td>69.8%</td></tr><tr><td rowspan="2">Grok</td><td>w/o Strategy</td><td>73.7%</td><td>82.3%</td><td>59.6%</td><td>69.1%</td></tr><tr><td>w/o Strategy &amp; Taxonomy</td><td>67.4%</td><td>99.9%</td><td>33.9%</td><td>47.8%</td></tr><tr><td rowspan="2">Qwen</td><td>w/o Strategy</td><td>74.8%</td><td>85.9%</td><td>58.7%</td><td>69.8%</td></tr><tr><td>w/o Strategy &amp; Taxonomy</td><td>70.4%</td><td>93.1%</td><td>43.1%</td><td>58.9%</td></tr></table>

Table V shows that removing only the type-aware validity assessment strategy leads to an average F1-score decline of approximately 14.2 percentage points across the three LLMs. Removing both components results in a larger decline of approximately 22.4 percentage points. This variant achieves high precision but low recall, suggesting that without the taxonomy, the agent lacks clear guidance on what constitutes an invalid report. These results confirm that both the type-aware strategy and the taxonomy contribute to the effectiveness of ISSUELIGHTER.

Cost and Runtime Analysis. We further compared the average API call cost and runtime of each method, as shown in Table VI. In terms of cost, ISSUELIGHTER incurs higher API fees than enhanced zero-shot and few-shot. However, considering the substantial performance gains and the high effort of manual review, the additional cost seems acceptable in practice. Regarding runtime, ISSUELIGHTER takes significantly longer than enhanced zero-shot and few-shot, primarily because it requires multiple rounds of interaction to configure the virtual environment, execute code, and retrieve information. Further analysis reveals that, under the ChatGPT configuration, environment reconstruction takes approximately 70 seconds on average, accounting for about 59% of IS-SUELIGHTER’s total runtime. If issue reporters have already configured the environment locally, ISSUELIGHTER can reuse it directly, reducing the wait to roughly 50 seconds, which is more acceptable for a real-time submission scenario. For maintainers, checking the validity of issue reports is not a latency-critical scenario. Moreover, since validation tasks are independent, multiple agents can be configured to process concurrent submissions in parallel. Notably, both the cost and runtime of the two ablation variants are close to those of the full version, indicating that the taxonomy and type-aware strategy introduce little additional overhead.

2) Qualitative Evaluation: We complemented quantitative metrics with a qualitative analysis of ISSUELIGHTER on Chat-GPT (the best-performing model) from three perspectives to examine whether the type-aware validity assessment strategy proved effective. First, we examined whether ISSUELIGHTER correctly identifies why an issue report is invalid by comparing the generated rationale with what we annotated in

TABLE VI. Average Cost (\$) and Runtime (s) per Issue Report
<table><tr><td rowspan="2">Method</td><td colspan="2">ChatGPT</td><td colspan="2">Grok</td><td colspan="2">Qwen</td></tr><tr><td>cost ($)</td><td>runtime (s)</td><td>cost ($)</td><td>runtime (s)</td><td>cost ($)</td><td>runtime (s)</td></tr><tr><td>ISSUELIGHTER</td><td>0.045</td><td>119.4</td><td>0.042</td><td>96.3</td><td>0.007</td><td>106.4</td></tr><tr><td>Enhanced zero-shot</td><td>0.019</td><td>23.3</td><td>0.005</td><td>5.8</td><td>0.001</td><td>4.6</td></tr><tr><td>Few-shot</td><td>0.021</td><td>23.6</td><td>0.010</td><td>6.0</td><td>0.001</td><td>5.0</td></tr><tr><td>w/o Strategy</td><td>0.044</td><td>110.1</td><td>0.039</td><td>82.4</td><td>0.006</td><td>102.5</td></tr><tr><td>w/o Strategy &amp; Taxonomy</td><td>0.045</td><td>95.1</td><td>0.040</td><td>78.9</td><td>0.006</td><td>98.7</td></tr></table>

RQ1. Second, we analyzed the characteristics of its rationales when ISSUELIGHTER correctly classifies an issue as valid, to investigate whether its judgments are based on findings from the validity assessment process or merely on the original report text. Following the annotation procedure described in Section III, the two annotators achieved Cohen’s kappa of 0.89 and 0.83 for the rationale analyses of invalid and valid reports, respectively. Third, we performed a failure analysis using the same analysis method described in Section IV-A2 to understand the limitations of ISSUELIGHTER.

Alignment with Annotated Causes of Invalidity. We randomly sampled 100 reports that ISSUELIGHTER correctly classified as invalid. We compared the rationales generated by ISSUELIGHTER against the causes of invalidity we annotated in RQ1, categorizing the alignment into three types: (1) exact match, where the core reason in the rationale aligns with the annotated cause; (2) partial match, where the rationale includes the annotated cause but also mentions additional factors; and (3) mismatch, where the rationale identifies a different cause. Among the 100 samples, 69 (69%) were exact matches, 28 (28%) were partial matches, and 3 (3%) were mismatches. This indicates that ISSUELIGHTER’s determinations of invalidity are strongly aligned with the underlying causes, rather than occurring by chance. These findings suggest that the provided rationale can guide users to adjust their issue reports.

Rationale for Correctly Classified Valid Reports. Similarly, we randomly sampled 100 reports that ISSUELIGHTER correctly classified as valid and analyzed the characteristics of its generated rationales. The results show that in 99 cases (99%), the rationales explicitly mentioned key results from the validity assessment process that confirmed the authenticity of the issue. In 54 cases (54%), the rationales further pinpointed the exact location of the issue, such as specific code blocks or documentation sections. These characteristics demonstrate that when judging an issue report as valid, ISSUELIGHTER’s rationales are grounded in empirical evidence, obtaining first-hand information through code execution, documentation retrieval, and other means rather than reasoning over the report text.

Failure Analysis. The 17 cases where ISSUELIGHTER made incorrect validity judgments fell into five categories: key information not retrieved or misinterpreted (7 cases); failure to identify undocumented design decisions (3 cases); lack of practical engineering considerations (3 cases); reproduction failures caused by environmental discrepancies or dependency changes (2 cases); and misattribution of fix responsibility between the project and third-party dependencies (2 cases).

The results indicate that most errors fundamentally stem from limitations in information retrieval and interpretation, which is the key area to improve in the future.

During the failure analysis, we identified four cases where ISSUELIGHTER’s rationale revealed factual evidence that had not been mentioned in the original discussion, such as discrepancies between documentation and actual releases or inconsistencies in code behavior. This evidence supported a validity judgment different from the maintainers’ initial decision. In all cases, ISSUELIGHTER judged the reports valid, while the maintainers had initially closed them as invalid. For each case, we extracted the key evidence from ISSUELIGHTER’s final rationale and reorganized it into a new issue report submitted to the respective GitHub repository. All four cases resulted in positive outcomes: maintainers acknowledged the feedback and ultimately merged pull requests addressing the issues. This further demonstrates that the rationales generated by ISSUELIGHTER can provide sufficiently clear and actionable evidence to assist maintainers in their decision-making.

Summary for RQ2: ISSUELIGHTER consistently outperforms all baseline methods in validity assessment effectiveness. Performance across different issue types remains stable across all LLMs. Ablation studies further confirm that both the taxonomy of causes of invalidity and the typeaware validity assessment strategy contribute to the performance gains. Qualitative analysis confirms that its rationales align closely with annotated causes of invalidity and are grounded in empirical evidence. Across four cases where our assessments of issue report validity disagreed with maintainers, we submitted appeals supported by evidence provided by ISSUELIGHTER; all were accepted, indicating the practical utility of our approach.

## VI. IMPLICATIONS

This section discusses the implications of our study for researchers, maintainers, and reporters.

Implications for Researchers. Our study provides the first investigation of causes of invalidity for all types of issue reports, establishing a taxonomy of 13 causes of invalidity across six issue types. This taxonomy lays a theoretical foundation for future research on invalid report analysis. Researchers can build upon it to conduct more fine-grained cause analysis or to investigate the causes of invalidity in other software ecosystems. Additionally, our manually annotated dataset of 520 issue reports (including annotations of causes of invalidity for 257 invalid reports) can serve as a benchmark for evaluating automated issue report validity assessment techniques, providing empirical data to support future studies. Our failure analysis further reveals five categories of challenges encountered in our approach, such as improving information retrieval robustness and better handling of issues arising from crossproject dependencies, which is a promising research avenue.

Implications for Maintainers. Our study reveals that invalid issue reports are not random noise but follow predictable patterns tied to specific issue types. This insight enables maintainers to move from reactive handling toward proactive prevention. For instance, given that “User misunderstanding” is the most common cause among bug reports, maintainers can improve documentation by clarifying examples or design issue templates that prompt users to confirm whether they have consulted relevant documentation. For enhancement requests, which are often invalid due to “Existing alternative” or “Out of scope”, projects can establish clearer contribution boundaries and provide self-service checklists to help reporters assess alignment with project goals before submission. Furthermore, the ISSUELIGHTER framework demonstrates the feasibility of agent-based automated validity assessment; maintainers can draw on this approach to develop assistive tools that automatically detect common causes of invalidity at submission time or provide reference judgments during issue processing, thereby reducing manual overhead.

Implications for Issue Reporters. Our study reveals that a substantial portion of invalid issue reports stems from reporters’ misunderstanding of project documentation, code, or scope, as well as missing or incorrect critical information. Reporters can increase the likelihood of their issue reports being valid by first carefully consulting relevant documentation to confirm whether the observed behavior aligns with project expectations; second, understanding the project’s core scope and design goals to avoid submitting out-of-scope requests; and third, providing complete and accurate information (e.g., a minimal reproduction example). These practices not only reduce maintainers’ burden of handling invalid reports but also significantly improve the chances that reports will be promptly addressed and resolved. In essence, a high-quality issue report is itself a contribution to the project community, because it saves maintainers time that would otherwise be spent clarifying misunderstandings, allowing them to focus on solving genuine issues.

## VII. THREATS TO VALIDITY

This section discusses potential internal and external threats to the validity of our study.

## A. Internal Validity

Limited Sample Size for Certain Issue Types. In our dataset, some issue types, particularly CI and Installation, have relatively few samples. This limited sample size may affect the comprehensiveness of our analysis of causes of invalidity for these types and could impact the reliability of experimental results for these types. Future work with larger datasets encompassing more projects is needed to further validate our findings on these less common issue types.

Subjectivity. The annotation of issue types and validity, thematic analysis of causes of invalidity, and qualitative evaluation all involve subjective judgment, which may introduce researcher bias. To mitigate this threat, two researchers independently performed all annotation and coding tasks, with disagreements resolved through discussion and arbitration by a third researcher. While these processes do not eliminate subjectivity entirely, they reduce the risk of individual bias.

Randomness of LLM Outputs. The outputs of large language models are inherently stochastic, which may affect the reproducibility of experimental results. To mitigate this effect, we set the temperature parameter to 0 in all experiments, making the model outputs as deterministic as possible [49]. While setting the temperature to 0 does not guarantee identical outputs across multiple runs, it minimizes the impact of randomness on the reproducibility of our experimental results.

Label Criterion Discrepancy for RoBERTa. For the RoBERTa baseline, we followed the resolution-to-label mapping strategy of Dinc¸ and Tuz¨ un [21]. However, because¨ GitHub has no explicit field equivalent to Bugzilla’s resolution, and manually annotating roughly 6,000 training reports is infeasible, we use merged pull requests as a proxy signal for validity. Applying the same rule to our annotated dataset yields only 11 discrepancies (about 2.1%). This introduces small noise that underestimates RoBERTa’s performance. However, the core comparison of this paper is between LLM-based methods; RoBERTa serves only as a reference for traditional learning approaches. Therefore, this minor label variance does not affect the validity of our main conclusions.

Data Overlap. The taxonomy was constructed from all 257 invalid reports, primarily based on maintainers’ comments, whereas validity assessment relies on issue content. Although the two processes involve overlapping reports, they rely on different sources of information. Additionally, 105 reports were used in preliminary experiments to refine the typeaware strategy and determine the interaction round limit. After excluding 78 few-shot examples, 27 reports remained in the final evaluation set. To assess the impact of this overlap, we separately evaluated these 27 reports and the remaining 415 reports using ChatGPT, obtaining accuracy and F1-score of 92.6% and 92.3%, and 96.4% and 96.2%, respectively. These results show no apparent performance inflation on the overlapping reports.

Interaction Round Limit. The maximum number of interaction rounds was set to 9 based on preliminary experiments. In the main experiments, 2, 12, and 12 cases reached this limit for ChatGPT, Grok, and Qwen, respectively, of which 0, 2, and 4 were incorrectly assessed. This fixed limit may constrain the assessment of complex issues, motivating future work on adaptive termination strategies.

## B. External Validity

Project Differences. We selected three mainstream opensource Python projects, i.e., pandas, scikit-learn, and mypy, as our experimental subjects. While these projects have significant influence in the software engineering community and maintain active issue tracking systems, they cannot represent all open-source projects, let alone industrial closed-source projects [50]. First, our selected projects are all mature, largescale projects with active maintainers and well-defined issue management processes. Small or personal projects may have more informal issue tracking, and their criteria for determining invalid issues may differ. Second, projects written in different programming languages (e.g., Java, JavaScript, Go) may exhibit significant variations in issue writing style, maintenance norms, and judgment criteria. Third, commercial closed-source projects may follow entirely different issue management practices that are not publicly observable. Therefore, generalizing our findings to projects in other languages, other scales, other types of software, or industrial environments requires caution. We leave this as an important direction for future work.

Stability of Issue Status. Our dataset consists of issue reports created since August 8, 2025 and closed by March 8, 2026, and we annotated their validity status based on their state at the time of data collection (March 9, 2026). However, the resolution status of an issue is not always final. Some closed issues may be reopened in the future [51]. This means that some of our validity judgments, made based on the state at a single point in time, could potentially be overturned if issues are later reopened and addressed.

## VIII. CONCLUSION

This study advances the understanding and automated identification of invalid issue reports from both empirical and methodological perspectives. On the empirical side, through a thematic analysis of 257 invalid reports, we established the taxonomy of 13 causes of invalidity across six issue types, revealing the strong correlation between causes of invalidity and issue types. On the methodological side, to address the fundamental limitations of existing automated approaches, we propose ISSUELIGHTER, an agent-based framework with a type-aware validity assessment strategy. By dynamically executing commands, ISSUELIGHTER transforms issue validity assessment from a text-based inference paradigm to an evidence-based verification paradigm. Quantitative results across three LLMs show that ISSUELIGHTER consistently outperforms all baselines by a substantial margin. The performance remains robust across different issue types, and ablation studies confirm the contribution of both the typeaware validity assessment strategy and the taxonomy of causes of invalidity. More importantly, qualitative analysis confirms that ISSUELIGHTER’s judgments are built upon a substantive understanding of the issue’s nature. Whether identifying causes of invalidity or verifying valid reports, its rationales consistently reflect deep comprehension rather than shallow textual inference. Furthermore, we leveraged the additional key evidence collected by ISSUELIGHTER in four cases to challenge maintainers’ original decisions, and all were ultimately revised, demonstrating the practical value of our approach.

## IX. DATA AVAILABILITY STATEMENT

The replication package of this paper, including the complete prompts, source code, experimental results, etc., is publicly available with DOI: https://doi.org/10.6084/m9.figshare. 33993610.

## ACKNOWLEDGMENT

This work is supported by the National Natural Science Foundation of China Grants (62572048 and 62232003).

## REFERENCES

[1] D. Bertram, A. Voida, S. Greenberg, and R. Walker, “Communication, collaboration, and bugs: the social nature of issue tracking in small, collocated teams,” in Proceedings of the 2010 ACM conference on Computer supported cooperative work, 2010, pp. 291–300.

[2] S. Panichella, G. Canfora, and A. Di Sorbo, ““won’t we fix this issue?” qualitative characterization and automated identification of wontfix issues on github,” Information and Software Technology, vol. 139, p. 106665, 2021.

[3] L. Huang, Q. Shen, D. Wang, Y. Wu, M. Wang, and J. Chen, “Falsepositive bug reports in deep learning compilers: Stages, root causes, and mitigation,” ACM Transactions on Software Engineering and Methodology, 2025.

[4] P. J. Guo, T. Zimmermann, N. Nagappan, and B. Murphy, “Characterizing and predicting which bugs get fixed: an empirical study of microsoft windows,” in Proceedings of the 32nd ACM/IEEE International Conference on Software Engineering-Volume 1, 2010, pp. 495–504.

[5] M. Laiq, N. B. Ali, J. Borstler, and E. Engstr¨ om, “Industrial adoption¨ of machine learning techniques for early identification of invalid bug reports,” Empirical Software Engineering, vol. 29, no. 5, p. 130, 2024.

[6] N. Raman, M. Cao, Y. Tsvetkov, C. Kastner, and B. Vasilescu,¨ “Stress and burnout in open source: toward finding, understanding, and mitigating unhealthy interactions,” in Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering: New Ideas and Emerging Results, ser. ICSE-NIER ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 57–60. [Online]. Available: https://doi.org/10.1145/3377816.3381732

[7] S. Afroz, C. Miller, T. Menezes, E. Gilmour, A. Sarma, and Z. Feng, ““ai slop is ddosing open source”: Understanding the impact of aigenerated contributions on open source sustainability,” 2026. [Online]. Available: https://arxiv.org/abs/2607.04003

[8] M. S. Zanetti, I. Scholtes, C. J. Tessone, and F. Schweitzer, “Categorizing bugs with social networks: a case study on four open source software communities,” in 2013 35th International Conference on Software Engineering (ICSE). IEEE, 2013, pp. 1032–1041.

[9] Y. Fan, X. Xia, D. Lo, and A. E. Hassan, “Chaff from the wheat: Characterizing and determining valid bug reports,” IEEE transactions on software engineering, vol. 46, no. 5, pp. 495–525, 2018.

[10] J. He, L. Xu, Y. Fan, Z. Xu, M. Yan, and Y. Lei, “Deep learning based valid bug reports determination and explanation,” in 2020 IEEE 31st International Symposium on Software Reliability Engineering (ISSRE). IEEE, 2020, pp. 184–194.

[11] Q. Meng and J. Visser, “Which bug reports are valid and why? using the bert transformer to classify bug reports and explain their validity,” in Proceedings of the 4th European Symposium on Software Engineering, 2023, pp. 52–60.

[12] J. Sun, “Why are bug reports invalid?” in 2011 Fourth IEEE International Conference on Software Testing, Verification and Validation. IEEE, 2011, pp. 407–410.

[13] M. Erfani Joorabchi, M. Mirzaaghaei, and A. Mesbah, “Works for me! characterizing non-reproducible bug reports,” in Proceedings of the 11th working conference on mining software repositories, 2014, pp. 62–71.

[14] M. M. Rahman, F. Khomh, and M. Castelluccio, “Why are some bugs non-reproducible?:–an empirical investigation using data fusion– ,” in 2020 IEEE international conference on software maintenance and evolution (ICSME). IEEE, 2020, pp. 605–616.

[15] Q. Wang, “Why is my bug wontfix?” in 2020 IEEE 2nd International Workshop on Intelligent Bug Fixing (IBF). IEEE, 2020, pp. 45–54.

[16] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in The eleventh international conference on learning representations, 2022.

[17] T. Schick, J. Dwivedi-Yu, R. Dess\`ı, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” Advances in neural information processing systems, vol. 36, pp. 68 539–68 551, 2023.

[18] J. Yang, A. Prabhakar, K. Narasimhan, and S. Yao, “Intercode: Standardizing and benchmarking interactive coding with execution feedback,” Advances in Neural Information Processing Systems, vol. 36, pp. 23 826–23 854, 2023.

[19] T. Zimmermann, N. Nagappan, H. Gall, E. Giger, and B. Murphy, “Cross-project defect prediction: a large scale experiment on data vs. domain vs. process,” in Proceedings of the 7th joint meeting of the European software engineering conference and the ACM SIGSOFT symposium on The foundations of software engineering, 2009, pp. 91– 100.

[20] S. Hosseini, B. Turhan, and D. Gunarathna, “A systematic literature review and meta-analysis on cross project defect prediction,” IEEE Transactions on Software Engineering, vol. 45, no. 2, pp. 111–147, 2017.

[21] E. Dinc¸ and E. Tuz¨ un, “Judge the votes: A system to classify bug reports¨ and give suggestions,” in 2025 2nd IEEE/ACM International Conference on AI-powered Software (AIware). IEEE, 2025, pp. 01–10.

[22] T. Vu, M. Iyyer, X. Wang, N. Constant, J. Wei, J. Wei, C. Tar, Y.- H. Sung, D. Zhou, Q. Le et al., “Freshllms: Refreshing large language models with search engine augmentation,” in Findings ofthe Association for Computational Linguistics: ACL 2024, 2024, pp. 13 697–13 720.

[23] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press, “Swe-agent: Agent-computer interfaces enable automated software engineering,” Advances in Neural Information Processing Systems, vol. 37, pp. 50 528–50 652, 2024.

[24] H. Zhang, W. Cheng, Y. Wu, and W. Hu, “A pair programming framework for code generation via multi-plan exploration and feedbackdriven refinement,” in Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering, 2024, pp. 1319–1331.

[25] M. Taeb, A. Swearngin, E. Schoop, R. Cheng, Y. Jiang, and J. Nichols, “Axnav: Replaying accessibility tests from natural language,” in Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, 2024, pp. 1–16.

[26] U. B. Torun, M. T. Demircan, M. F. Gon, and E. T ¨ uz¨ un, “Past, present,¨ and future of bug tracking in the generative ai era,” ACM Trans. Softw. Eng. Methodol., Apr. 2026, just Accepted. [Online]. Available: https://doi.org/10.1145/3806655

[27] (2026) pandas. [Online]. Available: https://pandas.pydata.org/

[28] (2026) scikit-learn. [Online]. Available: https://scikit-learn.org/

[29] (2026) mypy. [Online]. Available: https://mypy-lang.org/

[30] K. Malone and R. Wolski, “Doing data science on the shoulders of giants: the value of open source software for the data science community,” Harvard Data Science Review, vol. 2, no. 2, 2020.

[31] F. Khan, B. Chen, D. Varro, and S. Mcintosh, “An empirical study of type-related defects in python projects,” IEEE Transactions on Software Engineering, vol. 48, no. 8, pp. 3145–3158, 2021.

[32] (2026) Github rest api documentation. [Online]. Available: https: //docs.github.com/en/rest

[33] F. Tu, J. Zhu, Q. Zheng, and M. Zhou, “Be careful of when: an empirical study on time-related misuse of issue tracking data,” in Proceedings of the 2018 26th ACM Joint Meeting on European Software Engineering

Conference and Symposium on the Foundations of Software Engineering, 2018, pp. 307–318.

[34] J. Cohen, “A coefficient of agreement for nominal scales,” Educational and psychological measurement, vol. 20, no. 1, pp. 37–46, 1960.

[35] J. R. Landis and G. G. Koch, “The measurement of observer agreement for categorical data,” biometrics, pp. 159–174, 1977.

[36] T. F. Bissyande, D. Lo, L. Jiang, L. R´ eveillere, J. Klein, and Y. Le Traon,´ “Got issues? who cares about it? a large scale investigation of issue trackers from github,” in 2013 IEEE 24th international symposium on software reliability engineering (ISSRE). IEEE, 2013, pp. 188–197.

[37] J. Cabot, J. L. C. Izquierdo, V. Cosentino, and B. Rolandi, “Exploring the use of labels to categorize issues in open-source software projects,” in 2015 IEEE 22nd International Conference on Software Analysis, Evolution, and Reengineering (SANER). IEEE, 2015, pp. 550–554.

[38] D. S. Cruzes and T. Dyba, “Recommended steps for thematic synthesis in software engineering,” in 2011 international symposium on empirical software engineering and measurement. IEEE, 2011, pp. 275–284.

[39] A. Castleberry and A. Nolen, “Thematic analysis of qualitative research data: Is it as easy as it sounds?” Currents in pharmacy teaching and learning, vol. 10, no. 6, pp. 807–815, 2018.

[40] G. Guest, A. Bunce, and L. Johnson, “How many interviews are enough? an experiment with data saturation and variability,” Field methods, vol. 18, no. 1, pp. 59–82, 2006.

[41] A. Wutich, M. Beresford, and H. R. Bernard, “Sample sizes for 10 types of qualitative data analysis: an integrative review, empirical guidance, and next steps,” International Journal of Qualitative Methods, vol. 23, p. 16094069241296206, 2024.

[42] Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou et al., “The rise and potential of large language model based agents: A survey,” Science China Information Sciences, vol. 68, no. 2, p. 121101, 2025.

[43] D. Merkel et al., “Docker: lightweight linux containers for consistent development and deployment,” Linux j, vol. 239, no. 2, p. 2, 2014.

[44] T. Zimmermann, R. Premraj, N. Bettenburg, S. Just, A. Schroter, and C. Weiss, “What makes a good bug report?” IEEE Transactions on Software Engineering, vol. 36, no. 5, pp. 618–643, 2010.

[45] W. Jin and A. Orso, “Bugredux: Reproducing field failures for inhouse debugging,” in 2012 34th international conference on software engineering (ICSE). IEEE, 2012, pp. 474–484.

[46] A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh, A. El-Kishky, A. McLaughlin, A. Low, A. Ostrow, A. Ananthram et al., “Openai gpt-5 system card,” arXiv preprint arXiv:2601.03267, 2025.

[47] xAI, “Grok 4,” xAI News, Jul. 2025, accessed: March 8, 2026. [Online]. Available: https://x.ai/news/grok-4

[48] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[49] M. Renze, “The effect of sampling temperature on problem solving in large language models,” in Findings of the association for computational linguistics: EMNLP 2024, 2024, pp. 7346–7356.

[50] E. Kalliamvakou, G. Gousios, K. Blincoe, L. Singer, D. M. German, and D. Damian, “The promises and perils of mining github,” in Proceedings of the 11th working conference on mining software repositories, 2014, pp. 92–101.

[51] E. Shihab, A. Ihara, Y. Kamei, W. M. Ibrahim, M. Ohira, B. Adams, A. E. Hassan, and K.-i. Matsumoto, “Studying re-opened bugs in open source software,” Empirical Software Engineering, vol. 18, no. 5, pp. 1005–1042, 2013.