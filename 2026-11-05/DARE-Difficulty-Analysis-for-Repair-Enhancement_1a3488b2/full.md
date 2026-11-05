# DARE: Difficulty Analysis for Repair Enhancement

Jikwang Lee, Seungwook Jung, Jaechang Nam

Handong Global University, Pohang, South Korea

lucas20606@gmail.com, wjdtmddnr123@gmail.com, jcnam@handong.edu

Abstract—LLM-based coding agents are primarily evaluated by their solve rates. However, this metric does not reveal how difficult the solved tasks are or account for differences in task difficulty across benchmarks. Determining the difficulty of individual tasks by repeatedly running multiple agents is costly, while a predicted difficulty score alone provides limited guidance on what support an agent needs. To address these challenges, we propose DARE (Difficulty Analysis for Repair Enhancement), which integrates repair-task difficulty measurement, prediction, feature-group contribution analysis, and taskspecific repair support. We define continuous task difficulty as the observed failure rate across multiple coding agents. The DARE difficulty model predicts this difficulty using 151 static features extracted from benchmark artifacts. The framework also analyzes the contributions of seven Repair-Task Feature Groups to identify the characteristics most salient to each task and uses this information to select task-specific agent support. We evaluate DARE on four benchmark partitions comprising 2,594 tasks. The difficulty model’s aligned predictions achieve Spearman’s ρ = .476–.621 on held-out partitions, and the most salient Feature Groups vary across tasks and benchmarks. Furthermore, codingagent experiments show improved repair success when support is matched to each task’s salient Feature Group. These findings demonstrate that difficulty prediction and interpretation can lead to practical improvements in automated program repair (APR), while also providing a basis for difficulty-stratified agent evaluation, cross-benchmark comparisons, and identifying areas of repair capability that warrant further improvement.

## I. INTRODUCTION

Evaluating the performance of LLM-based coding agents solely by their overall solve rate can obscure both their ability to solve difficult tasks and how that ability has improved. Coding agents have shown rapid improvements in resolving real-world software issues. For example, reported solve rates on SWE-bench Verified increased from 33.2% in 2024 to 80.9% by early 2026 [1], [2]. However, the overall solve rate alone cannot reveal whether these gains reflect progress on previously unsolved difficult tasks or improvements primarily on easier ones. Moreover, two agents with the same solve rate may solve tasks of different difficulty, and differences in solve rates across benchmarks may reflect differences in taskdifficulty distributions as well as agent capabilities. Therefore, understanding agent performance and progress in automated program repair (APR) requires considering not only how many tasks agents solve, but also how difficult those tasks are.

Knowing the difficulty of individual tasks would enable more informative agent evaluation, but measuring it directly is expensive. Estimating task difficulty requires evaluating each task with multiple agents or conducting substantial manual assessments. A more efficient alternative is to predict difficulty from artifacts already available in a benchmark. Such predictions can support difficulty-stratified performance analysis, comparisons of difficulty distributions across benchmarks, and the selection of evaluation tasks without running a large number of agents on every task.

Prior studies have explored predicting task success or difficulty using task- and agent-related characteristics. However, how to leverage the task characteristics contributing to difficulty predictions to improve actual repair performance remains underexplored [3]–[5].

To go beyond difficulty prediction, we propose DARE (Difficulty Analysis for Repair Enhancement), a framework that combines difficulty prediction and feature-group contribution analysis to guide task-specific repair support. We define task difficulty as the observed failure rate across multiple coding agents. The DARE difficulty model uses a generalized additive model (GAM) [6] to estimate task difficulty from 151 static features extracted from benchmark artifacts. These features are organized into seven Repair-Task Feature Groups that capture different characteristics relevant to software repair. DARE quantifies each group’s contribution to the model’s difficulty prediction, identifies the most salient Feature Group for each task, and uses this information to guide the selection of corresponding repair support.

We evaluate DARE on four benchmark partitions comprising 2,594 tasks through three research questions. RQ1: How accurately does DARE predict repair difficulty? RQ2: How do Feature Group contributions vary with predicted task difficulty? RQ3: Does Feature-Group-guided support improve coding-agent repair success? The difficulty model generalizes to held-out partitions (Spearman ρ = .476–.621), and the most salient Feature Group varies across tasks. Matching support to this group raises a coding agent’s repair success from 54.7% to 79.6% on nine high-difficulty tasks (p = .0039).

Our main contributions are as follows:

• Difficulty prediction across benchmarks. We define repair-task difficulty using observed failure rates across multiple agents and evaluate its prediction on held-out benchmark partitions. We also use shared-agent scale alignment to adjust for differing agent populations.

• Interpretable analysis of repair-task difficulty. We organize 151 static features into seven Repair-Task Feature Groups reflecting different characteristics of software repair and analyze their task-level contributions to difficulty predictions. This reveals differences between tasks that a single difficulty score cannot capture.

• Analysis-guided repair support. We use each task’s most salient Feature Group to guide the selection of taskspecific support and evaluate its impact on coding-agent repair success, testing whether model-based difficulty interpretation is useful for practical repair assistance.

## II. BACKGROUND AND RELATED WORK

## A. Automated Program Repair and Coding-Agent Evaluation

Automated program repair (APR) aims to fix software defects automatically [7]. With LLMs, repair now includes coding agents that explore repositories, modify code, and validate changes to resolve real-world issues [8], [9].

SWE-bench evaluates whether agent-generated patches resolve real GitHub issues [10]. SWE-bench Verified provides human-reviewed tasks [1], while SWE-bench-Live adds newer issues for evaluation over time [11]. Section IV-E describes the benchmark partitions used in this study.

Agent performance is commonly summarized by solve rate, the proportion of tasks resolved successfully. However, two agents with the same solve rate need not solve the same tasks. Aggregate solve rates alone therefore provide limited insight into the difficulty of individual tasks or the relationship between task characteristics and repair outcomes [3]–[5]. This motivates measuring and predicting task-level difficulty as a complement to overall agent evaluation.

## B. Repair-Task Difficulty Prediction

Al-Haque and Johnson [3] analyzed repeated runs of a single Qwen3-Coder–OpenHands configuration on CoderForge-Preview using 54 static features from issues, reference patches, and repositories. They reported AUC .863 for predicting anysuccess (whether a task was solved at least once) and also examined continuous pass-rate prediction and SHAP-based attribution. Their single-agent, single-dataset setup does not show generalization across agent populations or benchmarks.

Krsteski and Meyer [4] predicted the relative ordering of benchmark-standardized item response theory (IRT) difficulty across 17 agentic benchmarks without running agents on target tasks. They showed that response-level AUC can reach .715 even with identical task-difficulty estimates, solely because of agent-ability differences. Thus, response-level AUC alone cannot show that a model captures task-difficulty differences.

Ge et al. [5] combined static task information and agent characteristics in an IRT-based model to estimate task difficulty and agent ability and to predict success probabilities for unobserved task–agent pairs. They evaluated generalization to unseen benchmarks and used their predictions to select evaluation tasks efficiently. Their primary evaluation, however, focused on agent–task success/failure discrimination using AUC-ROC, rather than directly assessing the ranking and absolute error of task-level multi-agent failure rates or analyzing contributions from Repair-Task Feature Groups [5].

DARE is therefore distinguished not simply by predicting task outcomes from static features, but by three aspects of its design. First, it defines continuous repair-task difficulty using the observed failure rates of multiple coding agents. Second, it evaluates both difficulty ordering and absolute prediction error on held-out benchmark partitions. Third, it aggregates contributions from the same additive prediction model into seven Repair-Task Feature Groups to identify candidates for task-specific support (Sections III-A–III-C and IV-D; RQ1– RQ2). RQ3 then evaluates whether support guided by these contributions improves coding-agent repair success. Thus, DARE connects difficulty prediction and Feature Group interpretation to a support stage whose practical effectiveness is evaluated through coding-agent experiments.

## C. Failure Analysis and Repair-Specific Support

Liu et al. [12] analyzed failures in unresolved issues and evaluated an improvement architecture informed by their findings. Ceka et al. [13] examined agent execution traces to identify testing and debugging bottlenecks and discussed design implications. Both identify recurring problems from outcomes or trajectories observed after execution [12], [13].

Other systems improve repair during execution: AutoCodeRover [9] supports code search and fault localization; SpecRover [14] uses specification inference and a reviewer agent; and RepairAgent [15] selects tools based on information and feedback gathered during execution. These approaches focus on the repair process or its tool use [9], [14], [15].

Unlike post-hoc analysis or feedback-driven tool selection, DARE identifies each task’s most salient Feature Group from static benchmark artifacts before the target agent attempts repair (RQ2), selects corresponding assistance, and evaluates its impact (RQ3). Reference-patch-derived features may indirectly inform support selection, but the reference patches are not provided to the agent; Section VI-B discusses the implications and scope of this benchmark-informed setting.

## III. APPROACH

Figure 1 shows the overview of DARE, a framework that connects repair-task difficulty prediction, Feature Group contribution analysis, and task-specific agent support. For each repair task, DARE answers three questions in sequence: How difficult is this task?, Which Feature Group is most salient in the model’s difficulty prediction?, and What type of support could help the agent solve it?

DARE first (1) takes static repair artifacts, such as the issue description, repository history, reference patch, and test artifacts, and (2) extracts features organized into seven Feature Groups: S, SG, L, T, V, C, and INT (Section III-B). To answer the first question, (3) the DARE difficulty model predicts the task’s difficulty scores, also reported as levels L1–L5. To answer the second, (4) DARE aggregates the model’s featurelevel contributions by Feature Group to identify the most salient group. To answer the third, (5) DARE selects the assistance module matching that group, and (6) the coding agent attempts the repair with this module. In the example, Transformation (T) is the most salient group, so the agent receives the T assistance module to improve its repair success. The following sections detail the key steps.

![](images/4dd06be1f2c70752e2da1caeee9418d0e9a8eb1fccf4f27865aa0d6ede0cda49.jpg)  
Fig. 1. Overview of the DARE framework. Static repair artifacts yield task features organized into seven Feature Groups. The DARE difficulty model predicts repair difficulty, while its feature contributions are aggregated by Feature Group to guide the selection of task-specific assistance. The displayed contribution values and selected module are illustrative.

## A. Repair-Task Difficulty

To predict how difficult a repair task is, we first need a measurable notion of difficulty.

Difficulty Definition. Response-based difficulty measures are established in educational assessments (proportion correct) [16], human reliability research (error probability and task complexity) [17], and software repair studies (codingagent outcomes) [3]. Following this approach, we define the repair difficulty of task i as

$$
D _ { i } = { \frac { { \mathrm { F a i l e d ~ a g e n t s ~ o n ~ t a s k ~ } } i } { { \mathrm { A g e n t s ~ t h a t ~ e v a l u a t e d ~ t a s k ~ } } i } } .\tag{1}
$$

Here, $D _ { i } \in [ 0 , 1 ] $ ; larger values indicate higher agent failure rates. Tasks not evaluated by an agent are excluded rather than counted as failures. We use SWE-bench Verified to establish the reference difficulty for cross-population consistency analysis because it provides extensive and complete evaluation data: 125 of its 175 agent configurations evaluated all 500 tasks.

Cross-Population Consistency. A common difficulty measure requires that tasks keep similar relative difficulty across agent populations. We split the 125 complete Verified configurations along four dimensions: provider (12 OpenAI vs. 9 Anthropic, same mini-SWE-agent scaffold), scaffold (41 mini-SWE-agent vs. 84 others), submission era (split at the median submission date), and performance level (split at the median overall solve rate). For each split, we compute each group’s difficulty for all 500 tasks independently and measure the Spearman rank correlation between the two task-aligned vectors (Table I).

All four comparisons agree strongly $( \rho = . 8 0 8 \mathrm { - } . 9 2 1 )$ , even between higher- and lower-performing agents $( \rho = . 8 9 4 )$ . The lowest value, for Provider, partly reflects its small groups, which yield coarser estimates. As agents improve, absolute failure rates decline, much as exam scores rise as students learn, but task ordering stays stable across submission eras $( \rho = . 9 2 1 )$ ; scale alignment adjusts the absolute level (Section IV-C). This indicates a shared task-level difficulty structure across agent populations, supporting multi-agent failure rates as our difficulty measure.

TABLE I  
CROSS-POPULATION CONSISTENCY OF TASK DIFFICULTY ON SWE-BENCH VERIFIED.
<table><tr><td>Population comparison</td><td>Spearman ρ</td></tr><tr><td>Provider</td><td>.808</td></tr><tr><td>Scaffold</td><td>.914</td></tr><tr><td>Submission era</td><td>.921</td></tr><tr><td>Performance level</td><td>.894</td></tr></table>

TABLE II

FIXED ABSOLUTE DIFFICULTY LEVELS.
<table><tr><td>Level</td><td>Difficulty interval</td></tr><tr><td>L1</td><td> $0 \leq \hat { D } _ { i } < 0 . 2$ </td></tr><tr><td>L2</td><td> $0 . 2 \leq \hat { D } _ { i } < 0 . 4$ </td></tr><tr><td>L3</td><td> $0 . 4 \leq \hat { D } _ { i } < 0 . 6$ </td></tr><tr><td>L4</td><td> $0 . 6 \leq \hat { D } _ { i } < 0 . 8$ </td></tr><tr><td>L5</td><td> $0 . 8 \leq \hat { D } _ { i } \leq 1 . 0$ </td></tr></table>

Continuous Difficulty. We keep difficulty continuous through out training and prediction, because dichotomizing continuous variables loses information and statistical power [18]. Training on multiple benchmarks, however, raises two scale problems. First, benchmarks differ in their difficulty distributions, so the same failure rate can be relatively easy in one benchmark and relatively hard in another. We therefore train the model on normalized difficulty: within each benchmark, a task’s failure rate is converted to its relative rank and mapped to a standard normal value. The model’s predictions are then mapped back to the original [0, 1] failure-rate scale using the training benchmarks’ difficulty distribution (Section IV-B). Second, failure rates depend on which agents were evaluated, and training and target benchmarks involve different agent populations. Sharedagent scale alignment uses two anchor agents evaluated on all benchmarks to recalibrate the predictions to the target benchmark’s scale (Section IV-C). The final predictions $\hat { D } _ { i }$ are evaluated against the remaining non-anchor agents, so alignment and evaluation never use the same agents.

Absolute Difficulty Levels. To check whether predicted difficulty agrees with observed failure rates (RQ1) and to identify tasks predicted to be difficult (RQ2), we group the final predictions into levels. Following the observation that four or five categories can limit information loss when grouping is necessary [18], we divide [0, 1] into five equal-width intervals of 0.2, labeled L1–L5 (Table II), as in calibration studies [19]. Unlike benchmark-relative groups such as quintiles, fixed intervals denote the same failure-rate range in every benchmark, allowing comparison on a common scale and assessment of absolute agreement, not only ordering [20]. Levels are used only for interpretation, never for training. Because failure rates depend on the evaluated agents, a level describes difficulty for the target benchmark’s agent population rather than a population-independent task property.

## B. Repair-Task Feature Groups

Task difficulty indicates which tasks are difficult, but not why. Software repair requires understanding the problem, locating relevant code, making changes, and validating them [7]. Prior work has examined bug-report information [21], fault localization [22], code-change size and characteristics [23], test-based validation [24], and links between static task characteristics and coding-agent performance [3]. We organize metrics from GitHub issue descriptions, repository snapshots and Git history, reference patches, and test artifacts into seven Repair-Task Feature Groups reflecting the repair process.

Specification (S) captures issue-description clues for understanding a problem and its requirements. Metrics cover issue length, structure, readability, error messages, stack traces, reproduction steps, and code identifiers. Prior work shows that reproduction steps and stack traces can aid bug resolution [21]. Specification–Solution Gap (SG) measures how directly issue information relates to the changes required for a solution [25]. Its metrics include textual similarity between the issue and reference patch or test artifacts, identifier overlap, solution identifier novelty, and specification coverage. Unlike $S ,$ which characterizes the issue itself, SG explicitly captures the information gap between the problem specification and solution artifacts as a separate group.

Localization (L) characterizes clues for identifying the files and code locations requiring modification. Its metrics include file paths, modules, and symbols mentioned in the issue, matches between stack traces and modified locations, and modified-file ranks in BM25- and TF-IDF-based retrieval results. Prior research has used information retrieval to locate relevant source files from bug reports [22]. Building on this approach, $L$ measures the correspondence between clues provided in the issue and the actual modification locations.

Transformation (T) characterizes the size and structural complexity of code changes required to resolve a repair task. Its metrics capture the scope and distribution of referencepatch changes, modified functions and classes, and changes to function signatures, conditionals, loops, and imports. Prior research has linked code change characteristics, including size and diffusion, to software defect risk [23]. We use these characteristics to represent the extent and structure of code transformations involved in repair.

Validation (V) characterizes the requirements for verifying that a patch resolves an issue without breaking existing functionality. Metrics include FAIL\_TO\_PASS and PASS\_TO\_PASS test composition, test-patch size and distribution, assertions, and fixtures, capturing validation scope and structural complexity. Test suites commonly serve as behavioral oracles in APR [24]. Thus, V measures the scope and structural complexity of a task’s validation artifacts.

Context (C) characterizes the structure and development history of the repository containing the code to be modified, beyond the size of an individual patch. Metrics cover file-path structure, dependency-graph fan-in and fan-out, Git change frequency and recency, and task metadata, reflecting connections to surrounding components and past changes [26], [27]. Interaction (INT) captures combinations of static characteristics across groups, such as specification–solution gap with patch complexity, localization features with repository size, and change history with dependency-graph centrality. These combinations can reflect characteristics missed by individual features; INT denotes static-feature interactions, not interactions between an agent and its tools or environment.

Our contribution is to organize previously studied repairrelated factors into a unified measurement framework comprising seven Feature Groups. Each feature group reflects a distinct aspect of the software repair process. We aggregate feature-level contributions within these groups to characterize the model’s difficulty predictions and identify each task’s most salient Feature Group (Section III-C). Feature extraction and selection are described in Section IV-A.

## C. Difficulty Model and Feature Group Contributions

The DARE difficulty model employs GAM-style additive regression to capture nonlinear relationships between features and difficulty while retaining interpretable feature-level contributions (Section IV-B). For task i with feature vector $\mathbf { x } _ { i } = ( x _ { i 1 } , \dots , x _ { i p } )$ , the additive prediction score is

$$
s _ { i } = \beta _ { 0 } + \sum _ { j = 1 } ^ { p } f _ { j } ( x _ { i j } ) .\tag{2}
$$

Here, i indexes tasks, $j$ indexes features, and $p$ is the total number of features. The term $\boldsymbol { x } _ { i j }$ denotes the value of feature $j$ for task $i , \ \beta _ { 0 }$ is the intercept, and $f _ { j } ( x _ { i j } )$ is the learned contribution of feature $j$ to the prediction score. Positive contributions increase the prediction score, whereas negative contributions decrease it. The final difficulty estimate $\hat { D } _ { i }$ is obtained by transforming this score to the [0, 1] failure-rate scale and applying shared-agent scale alignment (Sections IV-B and IV-C). Feature contributions apply to the score before these transformations, not directly to $\hat { D } _ { i }$

We aggregate the difficulty model’s feature-level contributions within each Feature Group. Let $\mathcal { I } _ { g }$ be the features in group $g .$ . Its contribution to the prediction score for task i is

$$
C _ { i , g } = \sum _ { j \in \mathcal { I } _ { g } } f _ { j } ( x _ { i j } ) .\tag{3}
$$

Each task therefore has both a predicted difficulty $\hat { D } _ { i }$ and a Feature Group contribution profile:

$$
\mathbf { C } _ { i } = ( C _ { i , S } , C _ { i , S G } , C _ { i , L } , C _ { i , T } , C _ { i , V } , C _ { i , C } , C _ { i , I N T } ) .\tag{4}
$$

For example, large contributions from L and V indicate that features related to locating the code to modify and validating the resulting patch contribute strongly to the prediction score. For each group, we subtract its average contribution over the training tasks, so that a positive centered contribution indicates that the group raises a task’s predicted difficulty more than it does for a typical training task. These centered values are model-based attributions, not causal evidence of repair failure.

Even tasks with similar predicted difficulty may have different Feature Group contribution profiles. We define a task’s salient Feature Group as the group with the largest positive centered contribution; a task is labeled None if no group has a positive centered contribution.

## D. Feature-Group-Guided Repair Support

Because tasks with similar predicted difficulty can have different salient groups, DARE uses the salient group, rather than the difficulty score alone, to select task-specific support. Each Feature Group can be paired with an assistance module that targets the repair activity the group characterizes; the module gives the agent additional analysis and feedback without changing the underlying model. For each task, DARE attaches the module for its salient group. Tasks labeled None, or whose salient group has no available module, are attempted without assistance. Modules use only information available to the agent, never reference patches or hidden tests.

For example, when T is the salient group (Fig. 1), the task is expected to require large or structurally complex changes. The T-Module therefore helps the agent plan dependency-aware follow-up changes; after a function signature is changed, it points to call sites that may also need updating. Section V-C1 describes the modules implemented in our evaluation.

Thus, DARE’s contribution lies not in the additive model itself, but in integrating difficulty prediction, Feature Group analysis, and task-specific support within a single framework.

## IV. EXPERIMENTAL SETUP

## A. Feature Extraction and Selection

We extract static features to characterize repair tasks and organize them into the seven Feature Groups (Section III-B). They are computed from issue descriptions, repository snapshots and Git history, reference patches, test artifacts, and task metadata. Repository features are extracted from the state at each task’s base\_commit, using only Git history reachable from that commit. Because our features include information from reference patches and test artifacts, we target difficulty prediction during benchmark construction and evaluation, before large-scale agent evaluation. Accordingly, feature extraction uses no agent trajectories, repair outcomes, or observed difficulty, which also prevents label leakage.

We start with 268 predefined candidate metrics and apply four feature selection steps to reduce redundancy and multicollinearity. First, we remove features with zero variance. Next, we remove highly correlated features within each Feature Group using a Spearman correlation threshold of $| \rho | > 0 . 9 0$ . The 0.90 threshold, previously used for correlationbased feature filtering [28], removes only very strong pairwise associations. We then remove near-exact linear dependencies $( R ^ { 2 } ~ > ~ 1 - 1 0 ^ { - 1 0 } )$ , corresponding to an implied variance inflation factor (VIF) greater than 10<sup>10</sup>, to reduce the risk of numerical instability before VIF-based filtering. Here, 10<sup>−10</sup> serves as a numerical tolerance rather than a statistical significance threshold. Finally, we iteratively remove features with a VIF greater than 10, following a commonly used heuristic for multicollinearity screening [29], [30]. This process produces a final set of 151 features.

We extract features consistently across all four benchmark partitions. The final 151 features are selected using only SWEbench Verified feature values, without difficulty labels or agent outcomes, so no label information leaks even when Verified is held out, and are applied to all four partitions. The selected features comprise 24 S, 17 SG, 15 L, 46 T, 29 V , 15 C, and 5 INT features. During model training and cross-validation, missing-value handling and feature transformations are fitted using only the training data, as described in Section IV-B. The complete feature list, including individual definitions, group assignments, and extraction procedures, is provided in the accompanying replication artifact [31].

## B. Difficulty Model Training

We implement the DARE difficulty model as a regularized Generalized Additive Model (GAM), which provides both difficulty predictions and feature-level contributions (Section III-C). GAM is a widely established interpretable model that captures nonlinear relationships while allowing each feature’s contribution to the prediction to be computed directly [6], [32]. Although other additive models, such as Explainable Boosting Machines (EBMs) and Neural Additive Models (NAMs), offer similar capabilities [33], [34], we choose GAM for its relatively simple structure, controlled model complexity, and clear feature-level contributions. Specifically, we use cubic splines to capture nonlinear relationships and ridge regularization to control model complexity [35]. The additive structure also allows feature contributions to be aggregated within each feature group, enabling the same difficulty model to support both prediction and feature-group analysis.

The model uses 151 features across the seven Feature Groups. To account for differing difficulty distributions across benchmarks, we convert observed difficulty within each source partition to ranks and map them to standard normal quantiles. This places difficulty values from different benchmarks on a common scale during training.

We select model hyperparameters using repository-grouped 5-fold cross-validation repeated 10 times, following the use of repeated five-fold validation in prior software prediction studies [36], [37]. The five-fold setting balances training-set size and validation coverage, while repetition reduces sensitivity to a particular fold partition. Grouping by repository prevents tasks from the same repository from appearing in both training and validation sets within a fold, thereby accounting for dependencies among tasks [38]. Within each cross-validation split, feature preprocessing and difficulty normalization are fitted using only the training data and then applied to the validation data. After hyperparameter selection, the model is retrained on the complete source training data. Its prediction scores are then mapped back to the [0, 1] failure-rate scale using the observed difficulty distribution of the source training data. Shared-agent scale alignment is then applied to obtain the final difficulty predictions. The alignment procedure is described in Section IV-C.

## C. Shared-Agent Scale Alignment

As established in Section III-A, task difficulty can vary depending on the population of coding agents being evaluated. To account for these differences, we apply Shared-Agent Scale Alignment to map predictions from the difficulty model trained on source benchmarks to the failure-rate scale of the target benchmark. This approach builds on logistic recalibration, which adjusts the intercept and slope of a prediction model when applying it to a new population [20], [39].

We use two SWE-agent-based configurations, Claude 3.7 Sonnet and GPT-4o, as anchor agents because their results are available across all four benchmarks. We first generate initial difficulty predictions, $\tilde { D } _ { i }$ , for target tasks using the model trained on the source data, as described in Section IV-B. We then fit a logistic regression using the observed success (1) or failure (0) of each anchor agent, denoted by $Y _ { i a }$

$$
\begin{array} { r l r } & { } & { \mathrm { l o g i t } P ( Y _ { i a } = 1 ) = \alpha + \beta z _ { i } + s _ { a } \delta , } \\ & { } & { z _ { i } = \mathrm { l o g i t } ( 1 - \tilde { D } _ { i } ) . } \end{array}\tag{5}
$$

Here, α adjusts the overall success-rate level, $\beta$ controls the spread of predictions, and δ accounts for performance differences between the two anchor agents. We set $s _ { a } = + 1$ for Claude 3.7 Sonnet and $s _ { a } ~ = ~ - 1$ for GPT-4o. For each held-out benchmark, the three parameters are estimated by maximum likelihood using only the observed anchor-agent outcomes. We constrain $\beta ~ = ~ \exp ( \theta ) ~ > ~ 0$ and clip logit inputs to $[ 1 0 ^ { - 6 } , 1 - 1 0 ^ { - 6 } ]$ for numerical stability. Missing agent outcomes are excluded.

The final difficulty prediction is obtained from the fitted relationship without the agent-specific effect:

$$
\hat { D } _ { i } = 1 - \sigma ( \hat { \alpha } + \hat { \beta } z _ { i } ) ,\tag{6}
$$

where $\sigma$ denotes the sigmoid function. This monotonic transformation adjusts the absolute failure-rate scale without reversing the predicted ordering of tasks. Accordingly, the featuregroup contributions defined in Section III-C are interpreted using the additive prediction scores before alignment.

To separate alignment from evaluation, we do not use the aggregate difficulty labels or non-anchor agent outcomes of the held-out benchmark for model training or alignment. Only the two anchor agents’ observed outcomes are used for alignment, while final performance is evaluated against the failure rates of the remaining non-anchor agents. This allows us to evaluate difficulty predictions against agents not used for alignment.

## D. Evaluation Metrics

We evaluate the final aligned predictions from the DARE difficulty model using three complementary metrics: Spearman rank correlation (ρ), pairwise accuracy (pAcc), and mean absolute error (MAE).

Spearman Rank Correlation (ρ). Spearman $\rho$ measures the rank agreement between predicted and observed task difficulty by capturing monotonic relationships [40]. It has also been used to evaluate difficulty prediction across benchmarks [4].

Pairwise Accuracy (pAcc). pAcc measures the proportion of task pairs correctly ordered by predicted difficulty. Let $D _ { i }$ and $\hat { D } _ { i }$ denote the observed and predicted difficulty of task $i ,$ respectively. For comparable pairs $\mathcal { P } = \{ ( i , j ) \mid i < j , D _ { i } \neq j$ $D _ { j } \}$ , pAcc is defined as

$$
\mathrm { p A c c } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } \mathbf { 1 } \Big [ \big ( \hat { D } _ { i } - \hat { D } _ { j } \big ) \big ( D _ { i } - D _ { j } \big ) > 0 \Big ] .\tag{7}
$$

Pairs tied in observed difficulty are excluded; predicted ties on otherwise comparable pairs are counted as incorrect. Krsteski and Meyer [4] also used within-benchmark pAcc alongside Spearman correlation as a complementary metric. We therefore report both to assess overall ranking agreement and pairwise difficulty discrimination.

Mean Absolute Error (MAE). MAE averages the absolute differences between predicted and observed task difficulty, indicating how closely predictions match observations [41]:

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | D _ { i } - \hat { D } _ { i } | ,\tag{8}
$$

where N is the number of evaluation tasks and $\hat { D } _ { i }$ is the final aligned prediction. Because difficulty is measured as a failure rate on [0, 1], an MAE of 0.15 corresponds to an average error of 15 percentage points.

Together, the three metrics provide complementary assessments: Spearman $\rho$ captures overall difficulty ranking agreement, pAcc measures pairwise discrimination, and MAE quantifies absolute prediction error. We thus evaluate both the relative ordering and the absolute difficulty of repair tasks.

For RQ1, we obtain 95% confidence intervals (CIs) by task-level bootstrap resampling, using 2,000 resamples for Spearman’s $\rho$ and 300 for the costlier pairwise pAcc.

## E. Benchmarks

We use SWE-bench and SWE-bench-Live, two benchmarks built from real-world GitHub issues. Both provide repositorylevel software repair tasks with reference patches and executable tests for evaluating the correctness of proposed fixes [10], [11].

SWE-bench. SWE-bench evaluates coding agents on software repair tasks derived from real GitHub issues and their associated pull requests [10]. The Full test set contains 2,294 tasks collected from 12 Python repositories. For each task, an agent must generate a patch using the issue description and the repository state before the fix, and the patch is evaluated using executable tests [10].

To support different evaluation needs, SWE-bench provides two curated subsets of the Full test set: Lite for lower-cost evaluation and Verified for more reliable task selection. SWEbench Lite is a 300-task subset designed to reduce evaluation costs while covering a range of task difficulties, with an emphasis on relatively self-contained bug fixes [42]. SWEbench Verified is a 500-task subset reviewed by professional developers to reduce evaluation issues caused by ambiguous problem descriptions or inadequate tests [1].

Because Verified and Lite are both subsets of Full, using these datasets directly could introduce overlapping tasks between training and evaluation. To prevent this, we partition SWE-bench by instance\_id into Verified (500 tasks), Lite \ Verified (207 tasks), and Full \ (Verified ∪ Lite) (1,587 tasks), as shown in Table III. These mutually exclusive partitions cover all 2,294 tasks in Full.

SWE-bench-Live. To evaluate DARE on more recently collected real-world repair tasks beyond those in the original SWE-bench, we use SWE-bench-Live Lite, a fixed split of 300 Python tasks collected from GitHub [11], [43]. This split constitutes the fourth evaluation partition in Table III. Unlike SWE-bench-Live’s continuously updated Full split, the fixed Lite split enables reproducible evaluations while keeping repeated agent executions in RQ3 manageable. We use this split to evaluate difficulty prediction in RQ1 and to assess whether support informed by the RQ2 feature-group analysis improves coding-agent repair success in RQ3.

Agent Outcomes and Evaluation Partitions. We construct task difficulty from publicly available agent evaluation results, using official SWE-bench evaluation records and SWE-bench-Live submission records [43], [44]. SWE-bench Verified contains results from 175 agent configurations, of which 125 have outcomes for all 500 tasks. We use these 125 complete configurations to construct the reference difficulty for Verified. For the remaining partitions, we use the observed outcomes from 84 configurations for Lite \ Verified, 24 for Full \ (Verified ∪ Lite), and 20 for SWE-bench-Live Lite. Missing task-agent outcomes are excluded rather than treated as failures.

For RQ1, we use leave-one-dataset-out (LODO) evaluation, training the DARE difficulty model on three partitions and evaluating its aligned predictions on the fourth. Unlike random task splits, LODO evaluates generalization to held-out task sets, following prior cross-dataset and cross-project validation studies [45], [46]. Task overlap between training and evaluation partitions is removed, although some repositories are shared across partitions. The separation of scale alignment and evaluation is described in Section IV-C.

## V. RESULTS

A. RQ1: How Accurately Does DARE Predict Repair Difficulty?

1) Prediction Accuracy on Held-Out Benchmarks: Across four LODO holdouts, the DARE difficulty model consistently predicted relative task-difficulty rankings, showing positive rank correlations and above-chance pairwise discrimination (Table IV). We assess performance using Spearman’s ρ, pairwise accuracy (pAcc), and mean absolute error (MAE); 95% confidence intervals (CIs) are computed as described in Section IV-D. Predictions are aligned using two shared anchor agents and evaluated against non-anchor agents not used for alignment (Section IV-C).

TABLE III  
BENCHMARKS AND LODO EVALUATION PARTITIONS.
<table><tr><td>Benchmark / partition</td><td>#Test</td><td>#Train</td><td>#Agents</td></tr><tr><td>SWE-bench Verified</td><td>500</td><td>2,094</td><td>125</td></tr><tr><td>Lite \ Verified</td><td>207</td><td>2,387</td><td>84</td></tr><tr><td>Full\ (Verified ∪ Lite)</td><td>1,587</td><td>1,007</td><td>24</td></tr><tr><td>SWE-bench-Live Lite</td><td>300</td><td>2,294</td><td>20</td></tr></table>

Note. #Test is the held-out task count; #Train is the count in the other three partitions. #Agents counts public configurations used to construct difficulty, not benchmark-defined totals. RQ1 evaluation excludes alignment anchor agents; RQ3 intervention procedures are described in Section V-C.

TABLE IV  
DIFFICULTY PREDICTION PERFORMANCE UNDER CROSS-PARTITION HOLDOUT EVALUATION.
<table><tr><td>Held-out partition</td><td>ρ [95% CI]</td><td>pAcc [95% CI]</td><td>MAE</td></tr><tr><td>SWE-bench Verified</td><td>.621 [.563,.674]</td><td>.723 [.698,.748]</td><td>.254</td></tr><tr><td>Lite\ Verified</td><td>.541 [.434,.635]</td><td>.694 [.658,.737]</td><td>.197</td></tr><tr><td>Full\(Verified∪Lite)</td><td>.566 [.530,.601]</td><td>.744 [.725,.759]</td><td>.123</td></tr><tr><td>SWE-bench-Live Lite</td><td>.476 [.376,.564]</td><td>.675 [.639,.708]</td><td>.149</td></tr></table>

![](images/fab1f97ac2ffadf9bfb9e30b4cef54a64c525db47c983ce73c505699d1861aba.jpg)  
Fig. 2. Distribution of observed non-anchor failure rates across five fixed predicted absolute-difficulty levels (L1–L5), assigned using final aligned predictions. Boxes indicate the interquartile range (IQR), center lines indicate medians, whiskers extend to the most extreme observations within 1.5×IQR, and n above each level indicates the number of tasks. Outlier markers are omitted for readability.

Spearman’s ρ, pAcc, and MAE capture complementary aspects of prediction quality (Section IV-D): task-difficulty rank agreement, pairwise discrimination, and absolute failure-rate error, respectively. Across four held-out partitions, ρ ranged from .476 to .621, within the moderate correlation range (.40– .69) of Schober et al. [47], although this classification depends on context. All 95% CIs lay above zero, indicating positive rank correlation rather than no association. This consistency across held-out partitions supports the difficulty model’s ability to recover relative task-difficulty rankings.

pAcc=.675–.744 means that the model correctly identified the harder task in about 68–74% of pairs with different observed difficulties. All 95% CIs were above .500, the chance level, supporting above-chance discrimination in every partition. MAE=.123–.254 corresponds to mean absolute failurerate errors of 12.3–25.4 percentage points.

For comparison, Krsteski and Meyer [4] reported mean $\rho { = } . 2 2 5$ across 17 held-out agentic benchmarks; their best estimator ablation achieved $\rho { = } . 2 9 5$ and pAcc=.602. Our model achieved mean $\rho { = } . 5 5 1$ and pAcc=.709 across four partitions. However, different benchmark compositions and difficulty definitions preclude direct comparison. Verified had the highest ρ (.621) but also the highest MAE (.254), showing that stronger rank agreement does not guarantee low absolute prediction error [20], [41], [48].

2) Predicted Absolute Difficulty Levels Correspond to Observed Difficulty: Across all four held-out partitions, higher predicted difficulty levels consistently corresponded to higher median observed non-anchor failure rates (Fig. 2). In Verified, the median increased from 5.6% at L1 through 14.1%, 23.8%, and 47.6% at L2–L4 to 83.1% at L5. The other three partitions showed the same monotonic increase across their populated levels.

The observed failure-rate distributions overlapped across levels. Some partitions also had few or no tasks in the lower levels. Nevertheless, the consistent increase in median failure rates across all four partitions indicates that the DARE difficulty model’s predicted difficulty levels reflect meaningful differences in observed repair difficulty.

RQ1 Answer: Across four held-out partitions, the DARE difficulty model consistently predicted relative task-difficulty rankings, showing positive rank correlations and above-chance pairwise discrimination. Moreover, median observed failure rates increased monotonically across the populated levels of the five fixed predicted absolute-difficulty levels (L1–L5), demonstrating that these levels reflect meaningful differences in observed repair difficulty.

## B. RQ2: How Do Feature-Group Contributions Vary with Predicted Task Difficulty?

RQ1 showed that the DARE difficulty model can predict repair-task difficulty on held-out benchmark partitions. RQ2 examines how the seven feature groups introduced in Section III-B contribute to these predictions. We ask (1) which groups contribute most across tasks and (2) whether the most salient group changes with predicted difficulty. We analyze 2,594 tasks from the four cross-partition holdouts in

TABLE V  
FEATURE-GROUP IMPORTANCE AND SPEARMAN CORRELATION WITH OBSERVED NON-ANCHOR DIFFICULTY.
<table><tr><td>Feature group</td><td>Importance (%) Spearman ρ [95% CI]</td></tr><tr><td>Transformation (T)</td><td>26.9 .412 [.366, .455]</td></tr><tr><td>Validation (V)</td><td>20.1 .338 [.290, .380]</td></tr><tr><td>Specification-Solution Gap (SG)</td><td>13.9 .347 [.298, .393]</td></tr><tr><td>Localization (L)</td><td>13.5 .255 [.208, .301]</td></tr><tr><td>Context (C)</td><td>11.0 .030 [−.024, .082]</td></tr><tr><td>Specification (S)</td><td>10.0 .052 [.001, .105]</td></tr><tr><td>Interaction (INT)</td><td>4.5 .199 [.150, .247]</td></tr></table>

RQ1, using the feature-group contributions defined in Section III-C and the observed non-anchor difficulty defined in Section IV-C.

1) Global Feature-Group Importance: Four feature groups—Transformation (T), Validation (V ), Specification– Solution Gap (SG), and Localization (L)—contributed most overall, but no single group dominated the difficulty model’s predictions (Table V). To quantify their relative importance, we average the absolute centered contribution of each group across tasks within each held-out partition. We then average these values equally across the four partitions and normalize the seven groups to sum to 100%, yielding Feature-Group Importance. Because this measure uses absolute contributions, it captures the magnitude of each group’s contribution to the prediction score regardless of whether that contribution is positive or negative.

To examine the relationship with observed repair outcomes, we also compute Spearman’s rank correlation (ρ) between each group’s signed contribution and the observed failure rate among non-anchor agents. The correlations are likewise averaged equally across the four holdouts, and 95% confidence intervals (CIs) are estimated using 2,000 task-level bootstrap repetitions (Table V).

T had the highest importance (26.9%) and the strongest correlation with observed difficulty (ρ = .412). V (20.1%), SG (13.9%), and L (13.5%) followed, giving these four groups a combined importance of 74.4% (Table V). All four showed positive correlations with observed difficulty in every held-out partition (Supplementary Table S1 in [31]).

Even T, the most influential group, accounted for only 25.1–28.8% of Feature-Group Importance within individual partitions (Supplementary Table S1 in [31]). This indicates that the difficulty model combines contributions from all seven feature groups to reflect different aspects of software repair (Section III-B; Table V). The predictive performance reported in RQ1 was also obtained using the difficulty model with all seven feature groups (Section IV-B; Section V-A).

2) Task-Level Salient Feature Groups: The most salient feature group varied across predicted difficulty levels and benchmark partitions, including among tasks assigned to the highest predicted difficulty level (Fig. 3). For each task, we define the salient feature group as the group with the largest positive centered contribution; tasks with no positive group contribution are labeled None. We then assign tasks to the absolute predicted difficulty levels (L1–L5) defined in Section III-A and compare the distribution of salient groups within each level (Fig. 3).

![](images/875985794a4b5635f64587831773c308b53f181c1f3feaaeb399063d0a26e14e.jpg)  
Fig. 3. Distribution of salient feature groups across predicted absolute difficulty levels (L1–L5). Each cell reports the percentage of tasks in that level assigned to the indicated salient-group category; n denotes the number of tasks at each level. Gray rows indicate levels containing no tasks (Section III-A).

Figure 3 shows that different feature groups can be salient even among tasks in the same predicted difficulty level. At L5, T was the most common salient group in three partitions (37.3–61.7%), whereas SG was most common in Lite\Verified (28.7%). Even where T was most frequent, many tasks had a different salient group.

The leading group also changed across difficulty levels within a benchmark. In SWE-bench Verified, S was most frequent at L3 (24.4%), L at L4 (33.1%), and T at L5 (37.3%) (Fig. 3). These patterns show that tasks with similar predicted difficulty may nevertheless have different salient feature groups.

Such task-level variation provides a basis for selecting support according to each task’s salient group and evaluating its effectiveness in RQ3. However, L2 contained only one task in SWE-bench-Live Lite and eight in Lite\Verified, so we interpret the distributions at these sparsely populated levels with caution.

RQ2 Answer: Four groups (T, V , SG, and L) accounted for 74.4% of total Feature-Group Importance, but no single group dominated the difficulty model’s predictions. The salient group varied across predicted difficulty levels and benchmark partitions.

C. RQ3: Does Feature-Group-Guided Support Improve Coding-Agent Repair Success?

RQ1 evaluated DARE’s ability to predict repair-task difficulty, while RQ2 identified the Feature Groups contributing most strongly to the predicted difficulty of individual tasks. RQ3 investigates whether support modules selected based on these analyses can improve coding-agent repair success.

1) Support Modules: RQ2 showed that T, V, SG, and L contribute most to difficulty predictions (Table V). Among them, we implement modules for T, V , and L, whose repair activities (code change, patch validation, and fault localization) can be supported with existing analysis techniques. We omit SG because we did not identify a sufficiently generalizable support mechanism to directly address this information gap.

T-Module. The T-Module supports large or structurally complex changes. Change planning uses repository-level dependency-aware planning [49] to identify follow-up modification locations using static references (Jedi [50]), type checking (basedpyright [51]), inheritance relationships, and co-change history from Git [27]. Completion checking runs code snippets from the issue and relevant existing tests on the original and modified repository states and reports remaining errors to the agent [52], [53].

V-Module. The V-Module assesses the adequacy of the agent’s validation. Change coverage measures the statement and branch coverage of modified code with coverage.py [54], reports untested locations [55], [56], and runs existing tests that exercise modified functions to detect regressions [57]– [59]. Validation-gap checks flag similar expressions left unchanged elsewhere [60], vary inputs of existing tests [61], and check the consistency of callers and subclasses affected by the change [62].

L-Module. The L-Module supports identifying modification locations. It expands initial candidates through call and dependency relationships [9], traces executions of agent-written reproduction scripts and, depending on the symptom type, maps error messages to their raising locations and traces faulty values to where they are written [63]. These diagnostics direct the agent toward root causes rather than where errors manifest.

All modules use only the issue description and the repository state available to the agent. The complete rules are provided in the replication artifact [31].

2) Experimental Setup and Task Selection: We conduct our experiments using SWE-bench-Live Lite, which contains 300 software repair tasks. To ensure reliable evaluation, we select 86 tasks whose issue descriptions are consistent with the evaluation tests and whose reference patches successfully pass the evaluation. This filtering excludes tasks where failures may result from inconsistencies between issue descriptions and tests or defects in the evaluation environment.

From these tasks, we select three tasks for each of the Transformation (T), Localization (L), and Validation (V) groups, resulting in nine evaluation tasks. Task selection follows two criteria: (1) the corresponding Feature Group has the largest positive centered contribution, as defined in RQ2, and (2) the task is assigned to predicted difficulty level L4 or L5, corresponding to a final aligned predicted failure rate of at least 0.6. The second criterion focuses the evaluation on tasks predicted to be difficult, where additional support may offer greater room for improvement, while aiming to reduce potential ceiling effects.

TABLE VI  
REPAIR SUCCESS WITH AND WITHOUT ASSISTANT MODULE.
<table><tr><td>Feature Group</td><td>Task</td><td>Baseline</td><td>With Module</td></tr><tr><td>L</td><td>matplotlib-29721</td><td>8/25</td><td>21/25</td></tr><tr><td>L</td><td>reflex-4371</td><td>16/25</td><td>19/25</td></tr><tr><td>L</td><td>keras-20443</td><td>16/25</td><td>23/25</td></tr><tr><td>T</td><td>pybamm-4865</td><td>2/25</td><td>13/25</td></tr><tr><td>T</td><td>pylint-10089</td><td>18/25</td><td>19/25</td></tr><tr><td>T</td><td>starlette-2812</td><td>18/25</td><td>22/25</td></tr><tr><td>V</td><td>pybamm-4644</td><td>19/25</td><td>25/25</td></tr><tr><td>V</td><td>conan-17967</td><td>18/25</td><td>20/25</td></tr><tr><td>V</td><td>haystack-8489</td><td>8/25</td><td>17/25</td></tr><tr><td>Total</td><td></td><td>123/225</td><td>179/225</td></tr><tr><td colspan="2">Wilcoxon p-value</td><td colspan="2">0.0039</td></tr></table>

3) Baseline and Repeated Evaluation: We use SWE-agent with Claude Sonnet 4.5 as the baseline configuration. At the time of the experiments, this combination achieved the highest performance on the SWE-bench-Live Lite leaderboard among SWE-agent-based configurations that support modular extensions.

All experiments use the same language model and execution and evaluation settings, with access to future Git history containing reference patches blocked. To account for variability across agent executions, each experimental round consists of five baseline runs and five module-assisted runs on the same task. We repeat this procedure for five rounds, resulting in 25 executions per configuration per task and 450 executions across all nine tasks. Repair success is determined by whether the generated patch passes the SWE-bench-Live Lite evaluation tests.

4) Statistical Analysis: We use the two-sided Wilcoxon signed-rank test to evaluate changes in repair performance. For each task, we pair the numbers of successful executions from the baseline and module-assisted configurations across 25 runs, yielding nine paired observations. We set the significance level to 0.05 and additionally report absolute success-rate improvements in percentage points.

5) Effect of Feature-Group-Assistant Module: Table VI presents the repair success counts for the baseline and moduleassisted configurations, together with the overall Wilcoxon test result.

In Table VI, each row represents an evaluation task. The Baseline column reports the number of successful executions using SWE-agent with Claude Sonnet 4.5, while the With Module column reports the corresponding results after applying the Feature Group-specific support module. Each value indicates the number of successful executions out of 25 runs.

The L, T, and V modules improve repair success rates by 30.7, 21.3, and 22.7 percentage points, respectively. The overall success rate increases from 54.7% (123/225) to 79.6% (179/225), an improvement of 24.9 percentage points, with improvements observed across all nine tasks. The Wilcoxon signed-rank test yields $p = 0 . 0 0 3 9$ , indicating a statistically significant difference at the 0.05 significance level.

RQ3 Answer: DARE shows that applying support modules corresponding to the most salient Feature Groups can significantly improve repair success (p = 0.0039) on the selected tasks.

## VI. DISCUSSION

## A. Implications for Agentic Software Engineering

Hassan et al. [64] emphasized the importance of systematic workflows and structured tool support for advancing Agentic Software Engineering. In line with this direction, DARE demonstrates how Feature Group contributions to difficulty predictions can inform practical agent support. In RQ3, support modules targeting Localization, Transformation, and Validation improved repair success on selected tasks. These findings suggest that task-specific difficulty analysis can provide practical guidance for agent assistance and motivate further investigation across diverse tasks and agent configurations.

## B. Threats to Validity

Difficulty is defined by failure rates, which depend on the evaluated agents; strong cross-population agreement and shared-agent scale alignment mitigate this threat. Reference patches are never given to agents, but patch-derived features indirectly inform support selection, limited to choosing a coarse Feature Group. RQ3 tasks predate the model’s training cutoff and may be memorized; however, baseline and moduleassisted runs use the same model and tasks, so contamination cannot explain their difference. RQ3 compares matched sup port only with no support and uses one agent configuration and nine tasks, so part of the gain may come from the modules themselves. All tasks are Python issues with reference patches; broader validation is needed to generalize these results.

## VII. CONCLUSION

We presented DARE, a framework that predicts repair-task difficulty from static benchmark artifacts, explains each prediction through seven Repair-Task Feature Groups, and uses the most salient group to select task-specific agent support. On four held-out benchmark partitions, the difficulty model recovered task-difficulty rankings (Spearman $\rho = . 4 7 6 \mathrm { - . 6 2 1 ) }$ , and matching support to each task’s salient group raised a coding agent’s repair success from 54.7% to 79.6% on nine high-difficulty tasks with statistical significance $( p = . 0 0 3 9 )$ These results show that interpreting difficulty predictions can guide how agents are assisted. We plan to generalize this predict–interpret–support process to other agentic tasks.

[1] N. Chowdhury, J. Aung, C. J. Shern, O. Jaffe, D. Sherburn, G. Starace, E. Mays, R. Dias, M. Aljubeh, M. Glaese, C. E. Jimenez, J. Yang, L. Ho, T. Patwardhan, K. Liu, and A. Madry, “Introducing SWE-bench Verified,” OpenAI, Aug. 2024. [Online]. Available: https://openai.com/index/introducing-swe-bench-verified/

[2] OpenAI, “Why SWE-bench Verified no longer measures frontier coding capabilities,” OpenAI Research, Feb. 2026, published February 23, 2026. [Online]. Available: https://openai.com/index/ why-we-no-longer-evaluate-swe-bench-verified/

[3] E. Al-Haque and B. Johnson, “What Makes Software Issue Resolution Tasks Difficult for Agents?” in 20th International Symposium on Empirical Software Engineering and Measurement (ESEM 2026), ser. Leibniz International Proceedings in Informatics (LIPIcs), R. Feldt, M. Paasivaara, D. Mendez, S. Wagner, and M. M. n. Baron,´ Eds., vol. 394. Dagstuhl, Germany: Schloss Dagstuhl – Leibniz-Zentrum fur Informatik, 2026, pp. 3:1–3:20. [Online]. Available: https:¨ //drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESEM.2026.3

[4] S. Krsteski and C. Meyer, “Predicting task difficulty without rollouts,” 2026, arXiv preprint arXiv:2608.05797. [Online]. Available: https://arxiv.org/abs/2608.05797

[5] C. Ge, D. Kryvosheieva, D. Fried, U. Girit, and K. Hariharan, “Agent psychometrics: Task-level performance prediction in agentic coding benchmarks,” in Proceedings of the 3rd Conference on Language Modeling (COLM), 2026. [Online]. Available: https: //arxiv.org/abs/2604.00594

[6] T. Hastie and R. Tibshirani, “Generalized additive models,” Statistical Science, vol. 1, no. 3, pp. 297–310, 1986. [Online]. Available: https://doi.org/10.1214/ss/1177013604

[7] L. Gazzola, D. Micucci, and L. Mariani, “Automatic software repair: A survey,” IEEE Transactions on Software Engineering, vol. 45, no. 1, pp. 34–67, 2019.

[8] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. R. Narasimhan, and O. Press, “SWE-agent: Agent-computer interfaces enable automated software engineering,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[9] Y. Zhang, H. Ruan, Z. Fan, and A. Roychoudhury, “Autocoderover: Autonomous program improvement,” in Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, ser. ISSTA 2024. New York, NY, USA: Association for Computing Machinery, 2024, p. 1592–1604. [Online]. Available: https://doi.org/10.1145/3650212.3680384

[10] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan, “SWE-bench: Can language models resolve realworld GitHub issues?” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https: //openreview.net/forum?id=VTF8yNQM66

[11] L. Zhang, S. He, C. Zhang, Y. Kang, B. Li, C. Xie, J. Wang, M. Wang, Y. Huang, S. Fu, E. Nallipogu, Q. Lin, Y. Dang, S. Rajmohan, and D. Zhang, “Swe-bench goes live!” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38, Main Conference. Curran Associates, Inc., 2025. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2025/file/d83c4a745789690f82e86d0ef752ae7c-Paper-Datasets and Benchmarks Track.pdf

[12] S. Liu, F. Liu, L. Li, X. Tan, Y. Zhu, X. Lian, and L. Zhang, “An empirical study on failures in automated issue solving,” arXiv preprint arXiv:2509.13941, 2025.

[13] I. Ceka, H. Mitchell, S. Pujar, L. Buratti, S. Ramji, J. Yang, G. Kaiser, and B. Ray, “Understanding automated program repair agents through the lens of traceability: An empirical study,” Proc. ACM Softw. Eng., vol. 3, no. ISSTA, Oct. 2026. [Online]. Available: https://doi.org/10.1145/3832285

[14] H. Ruan, Y. Zhang, and A. Roychoudhury, “Specrover: Code intent extraction via llms,” in Proceedings of the IEEE/ACM 47th International Conference on Software Engineering, ser. ICSE ’25. IEEE Press, 2025, p. 963–974. [Online]. Available: https: //doi.org/10.1109/ICSE55347.2025.00080

[15] I. Bouzenia, P. Devanbu, and M. Pradel, “Repairagent: An autonomous, llm-based agent for program repair,” in Proceedings of the IEEE/ACM 47th International Conference on Software Engineering, ser. ICSE

’25. IEEE Press, 2025, p. 2188–2200. [Online]. Available: https: //doi.org/10.1109/ICSE55347.2025.00157

[16] Y. H. Kim, B. H. Kim, J. Kim, B. Jung, and S. Bae, “Item difficulty index, discrimination index, and reliability of the 26 health professions licensing examinations in 2023, korea: a psychometric study,” Journal of Educational Evaluation for Health Professions, vol. 21, p. 40, 2024.

[17] L. Podofillini, J. Park, and V. N. Dang, “Measuring the influence of task complexity on human error probability: an empirical evaluation,” Nuclear Engineering and Technology, vol. 45, no. 2, pp. 151–164, 2013.

[18] D. G. Altman and P. Royston, “The cost of dichotomising continuous variables,” BMJ, vol. 332, no. 7549, p. 1080, 2006. [Online]. Available: https://www.bmj.com/content/332/7549/1080.1

[19] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the 34th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 70. PMLR, 2017, pp. 1321–1330. [Online]. Available: https://proceedings.mlr.press/v70/guo17a.html

[20] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, “Calibration: The achilles heel of predictive analytics,” BMC Medicine, vol. 17, no. 1, p. 230, 2019. [Online]. Available: https://doi.org/10.1186/s12916-019-1466-7

[21] N. Bettenburg, S. Just, A. Schroter, C. Weiss, R. Premraj, and¨ T. Zimmermann, “What makes a good bug report?” in Proceedings of the 16th ACM SIGSOFT International Symposium on Foundations of Software Engineering, ser. SIGSOFT ’08/FSE-16. New York, NY, USA: Association for Computing Machinery, 2008, p. 308–318. [Online]. Available: https://doi.org/10.1145/1453101.1453146

[22] J. Zhou, H. Zhang, and D. Lo, “Where should the bugs be fixed? more accurate information retrieval-based bug localization based on bug reports,” in 2012 34th International Conference on Software Engineering (ICSE), 2012, pp. 14–24.

[23] Y. Kamei, E. Shihab, B. Adams, A. E. Hassan, A. Mockus, A. Sinha, and N. Ubayashi, “A large-scale empirical study of just-in-time quality assurance,” IEEE Transactions on Software Engineering, vol. 39, no. 6, pp. 757–773, 2013.

[24] C. Le Goues, T. Nguyen, S. Forrest, and W. Weimer, “Genprog: A generic method for automatic software repair,” IEEE Transactions on Software Engineering, vol. 38, no. 1, pp. 54–72, 2012.

[25] H. Tian, X. Tang, A. Habib, S. Wang, K. Liu, X. Xia, J. Klein, and T. F. BissyandE, “Is this change the answer to that problem? correlating<sup>´</sup> descriptions of bug and code changes for evaluating patch correctness,” in Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, ser. ASE ’22. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3551349.3556914

[26] T. Zimmermann and N. Nagappan, “Predicting defects using network analysis on dependency graphs,” in Proceedings ofthe 30th International Conference on Software Engineering, ser. ICSE ’08. New York, NY, USA: Association for Computing Machinery, 2008, p. 531–540. [Online]. Available: https://doi.org/10.1145/1368088.1368161

[27] T. Zimmermann, P. Weisgerber, S. Diehl, and A. Zeller, “Mining version histories to guide software changes,” in Proceedings of the 26th International Conference on Software Engineering, ser. ICSE ’04. USA: IEEE Computer Society, 2004, p. 563–572.

[28] M. Kuhn, “Building predictive models in r using the caret package,” Journal of Statistical Software, vol. 28, no. 5, pp. 1–26, 2008. [Online]. Available: https://www.jstatsoft.org/article/view/v028i05

[29] C. F. Dormann, J. Elith, S. Bacher, C. Buchmann, G. Carl, G. Carre,´ J. R. G. Marquez, B. Gruber, B. Lafourcade, P. J. Leit´ ao˜ et al., “Collinearity: a review of methods to deal with it and a simulation study evaluating their performance,” Ecography, vol. 36, no. 1, pp. 27– 46, 2013.

[30] R. M. O’brien, “A caution regarding rules of thumb for variance inflation factors,” Quality & quantity, vol. 41, no. 5, pp. 673–690, 2007.

[31] Anonymous, “Replication package for “DARE: Difficulty analysis for repair enhancement”,” Zenodo, 2026. [Online]. Available: https: //doi.org/10.5281/zenodo.23911645

[32] Y. Lou, R. Caruana, and J. Gehrke, “Intelligible models for classification and regression,” in Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD ’12. New York, NY, USA: Association for Computing Machinery, 2012, p. 150–158. [Online]. Available: https://doi.org/10.1145/2339530.2339556

[33] H. Nori, S. Jenkins, P. Koch, and R. Caruana, “InterpretML: A unified framework for machine learning interpretability,” arXiv preprint

arXiv:1909.09223, 2019. [Online]. Available: https://arxiv.org/abs/1909. 09223

[34] R. Agarwal, L. Melnick, N. Frosst, X. Zhang, B. Lengerich, R. Caruana, and G. E. Hinton, “Neural additive models: Interpretable machine learning with neural nets,” in Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds., vol. 34. Curran Associates, Inc., 2021, pp. 4699–4711. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf

[35] A. E. Hoerl and R. W. Kennard, “Ridge regression: Biased estimation for nonorthogonal problems,” Technometrics, vol. 12, no. 1, pp. 55–67, 1970. [Online]. Available: https://doi.org/10.1080/00401706. 1970.10488634

[36] K. Gao, T. M. Khoshgoftaar, and A. Napolitano, “Aggregating data sampling with feature subset selection to address skewed software defect data,” International Journal of Software Engineering and Knowledge Engineering, vol. 25, no. 09n10, pp. 1531–1550, 2015.

[37] A. Ali and C. Gravino, “An empirical comparison of validation methods for software prediction models,” Journal of Software: Evolution and Process, vol. 33, no. 8, p. e2367, 2021. [Online]. Available: https://doi.org/10.1002/smr.2367

[38] D. R. Roberts, V. Bahn, S. Ciuti, M. S. Boyce, J. Elith, G. Guillera-Arroita, S. Hauenstein, J. J. Lahoz-Monfort, B. Schroder, W. Thuiller,¨ D. I. Warton, B. A. Wintle, F. Hartig, and C. F. Dormann, “Crossvalidation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” Ecography, vol. 40, no. 8, pp. 913–929, 2017. [Online]. Available: https://doi.org/10.1111/ecog.02881

[39] E. W. Steyerberg, G. J. J. M. Borsboom, H. C. van Houwelingen, M. J. C. Eijkemans, and J. D. F. Habbema, “Validation and updating of predictive logistic regression models: A study on sample size and shrinkage,” Statistics in Medicine, vol. 23, no. 16, pp. 2567–2586, 2004. [Online]. Available: https://doi.org/10.1002/sim.1844

[40] C. Spearman, “The proof and measurement of association between two things,” The American Journal of Psychology, vol. 15, no. 1, pp. 72–101, 1904. [Online]. Available: https://doi.org/10.2307/1412159

[41] C. J. Willmott and K. Matsuura, “Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance,” Climate Research, vol. 30, no. 1, pp. 79–82, 2005. [Online]. Available: https://doi.org/10.3354/cr030079

[42] SWE-bench Team, “SWE-bench Lite,” Official Benchmark Documentation, accessed: October 2026. [Online]. Available: https://www.swebench.com/lite.html

[43] SWE-bench-Live, “SWE-bench-Live: Official dataset and evaluation repository,” GitHub repository, accessed October 2026. [Online]. Available: https://github.com/microsoft/SWE-bench-Live

[44] SWE-bench Team, “SWE-bench Experiments,” GitHub repository, accessed: October 2026. [Online]. Available: https://github.com/ SWE-bench/experiments

[45] R. Laroca, E. V. Cardoso, D. R. Lucio, V. Estevam, and D. Menotti, “On the cross-dataset generalization in license plate recognition,” in Proceedings ofthe 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2022) - Volume 5: VISAPP, INSTICC. SciTePress, 2022, pp. 166–178.

[46] S. Hosseini, B. Turhan, and D. Gunarathna, “A systematic literature review and meta-analysis on cross project defect prediction,” IEEE Transactions on Software Engineering, vol. 45, no. 2, pp. 111–147, 2019.

[47] P. Schober, C. Boer, and L. A. Schwarte, “Correlation coefficients: Appropriate use and interpretation,” Anesthesia & Analgesia, vol. 126, no. 5, pp. 1763–1768, 2018. [Online]. Available: https: //doi.org/10.1213/ANE.0000000000002864

[48] E. W. Steyerberg, A. J. Vickers, N. R. Cook, T. Gerds, M. Gonen, N. Obuchowski, M. J. Pencina, and M. W. Kattan, “Assessing the performance of prediction models: A framework for traditional and novel measures,” Epidemiology, vol. 21, no. 1, pp. 128–138, 2010. [Online]. Available: https://doi.org/10.1097/EDE.0b013e3181c30fb2

[49] R. Bairi, A. Sonwane, A. Kanade, V. D. C., A. Iyer, S. Parthasarathy, S. Rajamani, B. Ashok, and S. Shet, “Codeplan: Repository-level coding using llms and planning,” Proc. ACM Softw. Eng., vol. 1, no. FSE, Jul. 2024. [Online]. Available: https://doi.org/10.1145/3643757

[50] D. Halter, “Jedi: Python static analysis and refactoring library,” Official Documentation, accessed November 3, 2026. [Online]. Available: https://jedi.readthedocs.io/en/latest/

[51] basedpyright Contributors, “basedpyright: Python type checker,” Official Documentation, accessed November 3, 2026. [Online]. Available: https://docs.basedpyright.com/latest/

[52] T. Ahmed, J. Ganhotra, R. Pan, A. Shinnar, S. Sinha, and M. Hirzel, “Otter: Generating tests from issues to validate SWE patches,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=b0jYs6JOZu

[53] M. Rezaalipour and C. A. Furia, “FauxPy: A fault localization tool for Python,” 2024, arXiv:2404.18596. [Online]. Available: https://arxiv.org/abs/2404.18596

[54] N. Batchelder and Contributors to Coverage.py, “Coverage.py: The code coverage tool for python,” accessed: November 3, 2026. [Online]. Available: https://coverage.readthedocs.io/

[55] N. Mundler, M. N. M¨ uller, J. He, and M. Vechev, “SWT-Bench:¨ Testing and validating real-world bug-fixes with code agents,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 81 857–81 887. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2024/file/94f093b41fc2666376fb1f667fe282f3-Paper-Conference.pdf

[56] diff-cover Contributors, “diff-cover: Automatically find diff lines that need test coverage,” GitHub Repository, accessed November 3, 2026. [Online]. Available: https://github.com/Bachmann1234/diff cover

[57] Y. Chen, T. Ahmed, R. Jabbarvand, and M. Hirzel, “Can old tests do new tricks for resolving swe issues?” Proc. ACM Softw. Eng., vol. 3, no. FSE, Jun. 2026. [Online]. Available: https://doi.org/10.1145/3808148

[58] M. Gligoric, L. Eloussi, and D. Marinov, “Practical regression test selection with dynamic file dependencies,” in Proceedings of the 2015 International Symposium on Software Testing and Analysis, ser. ISSTA 2015. New York, NY, USA: Association for Computing Machinery, 2015, p. 211–222. [Online]. Available: https://doi.org/10.1145/2771783.2771784

[59] pytest-testmon Contributors, “pytest-testmon: Test selection based on code changes,” Official Documentation, accessed November 3, 2026. [Online]. Available: https://www.testmon.org/

[60] M. Poehlmann and E. Juergens, “Revealing missing bug-fixes in code clones in large-scale code bases,” Electronic Communications of the EASST, vol. 60, 2013.

[61] B. Danglot, O. L. Vera-Perez, B. Baudry, and M. Monperrus, “Automatic´ test improvement with dspot: a study with ten mature open-source projects,” Empirical Software Engineering, vol. 24, no. 4, pp. 2603– 2635, 2019.

[62] B. G. Ryder and F. Tip, “Change impact analysis for objectoriented programs,” in Proceedings of the 2001 ACM SIGPLAN-SIGSOFT Workshop on Program Analysis for Software Tools and Engineering, ser. PASTE ’01. New York, NY, USA: Association for Computing Machinery, 2001, p. 46–53. [Online]. Available: https://doi.org/10.1145/379605.379661

[63] M. Sridharan, S. J. Fink, and R. Bodik, “Thin slicing,” in Proceedings of the 28th ACM SIGPLAN Conference on Programming Language Design and Implementation, ser. PLDI ’07. New York, NY, USA: Association for Computing Machinery, 2007, p. 112–122. [Online]. Available: https://doi.org/10.1145/1250734.1250748

[64] A. E. Hassan, H. Li, D. Lin, B. Adams, T.-H. Chen, Y. Kashiwa, and D. Qiu, “Agentic software engineering: Foundational pillars and a research roadmap,” 2025, arXiv:2509.06216. [Online]. Available: https://arxiv.org/abs/2509.06216