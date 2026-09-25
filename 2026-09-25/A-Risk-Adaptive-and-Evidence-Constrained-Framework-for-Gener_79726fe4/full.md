Article

# A Risk-Adaptive and Evidence-Constrained Framework for Generative AI Feedback in Programming Education

Shihao Wang <sup>1,</sup>∗

Department of Education, Practice and Society, UCL Institute of Education, University College London, London, United Kingdom; shihao.wang.24@ucl.ac.uk

Generative artificial intelligence can turn learning analytics into personalized support, but feedback systems must decide when to intervene, which evidence to use, and how much assistance to provide. We developed a risk-adaptive, evidence-constrained framework for introductory programming using 2993 failed-submission states from 215 students. Student-disjoint models predicted persistent failure and related outcomes; four matched feedback conditions were generated for 136 cases; and calibrated risk informed capacity-limited intervention policies. The validation-selected logistic regression model achieved a test precision–recall area under the curve of 0.550 and a receiver operating characteristic area under the curve of 0.681. Broader student histories improved prediction of unmodified resubmission. After standardized repair and evidence gating, 519 of 544 newly generated messages contained all required components. A fixed-threshold sequential policy selected 17.8% of eligible test states and captured 25.2% of observed persistent failures. These findings support an evidence-gated progressive assistance strategy: calibrated risk guides intervention timing, recorded evidence constrains feedback content, and assistance progresses from self-checks to localized hints when warranted. The framework connects prediction, decision-making, and grounded generation while keeping their evaluation outcomes distinct.

## Abstract

Keywords: Generative Artificial Intelligence; Learning Analytics; Personalized Feedback; Programming Education; Deep Learning

Received: Accepted: Published:

Copyright: © 2026 by the author. Submitted to Educ. Sci. for possible open access publication under the terms and conditions of the Creative Commons Attribution (CC BY) license.

## 1. Introduction

Feedback connects current performance with the actions needed for improvement, but its effectiveness depends on content, timing, and design (Wisniewski et al., 2020). These decisions are especially important in introductory programming, where students repeatedly write code, run tests, interpret failures, and revise solutions. Each failed submission creates an opportunity for support, yet the appropriate response depends on the learner’s recent history and available evidence. Effective feedback must therefore determine whom to support, when to intervene, what evidence to use, and how much guidance to provide.

Learning analytics can inform these decisions through behavioral and performance traces. Higher-education research shows that such data can support risk identification and targeted intervention, while dashboards and open learner models can link indicators to reflection and learner action (Hooshyar et al., 2020; Ifenthaler & Yau, 2020; Matcha et al., 2020). Generative AI extends this capability by converting contextual evidence into natural-language scaffolds (Kasneci et al., 2023; Li et al., 2025; Molenaar, 2022). Instructional guardrails are important because support should promote independent reasoning alongside immediate task progress (Bastani et al., 2025).

Programming submissions provide precise evidence from code, tests, and revisions. Recent studies have generated code explanations, novice-oriented error messages, and hints with varying levels of specificity (Leinonen et al., 2023; Lohr et al., 2025; Sarsa et al., 2022; Xiao et al., 2024). Their evaluations demonstrate technical feasibility and identify correctness, completeness, comprehensibility, and repair accuracy as key quality dimensions (Koutcheme et al., 2025). Two design questions remain closely connected: when feedback should be triggered within an attempt sequence and how learner history should regulate its content and specificity.

These questions require prediction and feedback generation to be treated as parts of a single educational process. A risk score is useful only when it informs an appropriate decision under realistic limits on instructor attention or automated support. Likewise, a well-written message is educationally credible only when its claims are grounded in the learner record available at that moment and its specificity matches the required level of assistance. Integrating timing, grounding, and assistance intensity therefore provides a stronger basis for personalized feedback than optimizing prediction accuracy or message fluency separately.

This integration also clarifies how the three intended learning outcomes can be examined. Knowledge mastery and longer-term performance require predictions linked to later task evidence, whereas self-regulated learning requires indicators of monitoring, revision, and strategy use rather than a single correctness score. A unified framework can preserve these distinctions while using the same temporally ordered evidence to determine when feedback is likely to be useful.

Ordered programming traces provide the temporal evidence needed to address these questions (Price et al., 2020). Sequence models can represent evolving learner histories, while interpretable models provide transparent benchmarks. Credible comparison requires student-disjoint evaluation and predictors restricted to information available at the intervention point (Kapoor & Narayanan, 2023). Prediction must then be connected to generation: calibrated risk can guide intervention timing and assistance level, while bounded task and code evidence can ground diagnostic statements and enable source verification (Huang et al., 2025).

The present study develops a risk-adaptive and evidence-constrained framework for this purpose. It analyzes 2993 failed-submission states from 215 students and predicts persistent failure, unmodified resubmission, near-term related-task performance, and 7– 28-day related-task performance. Logistic regression and gradient boosting are compared with multilayer perceptron, gated recurrent unit, Transformer, fusion, and multitask models using student-level partitions. A paired generation experiment represents 136 cases under four contextual conditions while holding the language model constant and varying current-task evidence, recent history, module context, calibrated risk, and assistance intensity. Fixed-capacity prioritization and sequential threshold replay assess intervention timing, while structural and source checks assess generated feedback.

The study is guided by four research questions:

RQ1: How accurately can current-submission evidence and prior learning traces predict near-term persistent failure and the defined operational outcomes under student-disjoint evaluation?

RQ2: How do interpretable tabular baselines and deep sequential models compare in predictive performance, calibration, and the contribution of different evidence groups?

RQ3: To what extent can four progressively contextualized feedback conditions generate structurally complete, actionable, and source-traceable messages?

RQ4: Under fixed-capacity and sequential decision settings, how effectively can calibrated risk estimates identify timely opportunities for progressively adjusted assistance?

This study contributes an auditable link between learner-state prediction and feedback generation. It examines the trade-off between predictive value and model complexity, introduces a paired design for evaluating contextual evidence and assistance intensity, and frames intervention timing as a capacity-aware educational decision. Source matching strengthens diagnostic traceability, while the progressive assistance policy translates calibrated risk into a structured sequence of self-checks, targeted hints, and more explicit guidance. Together, these elements provide a reproducible basis for personalized, explainable, and timely learning support and for evaluating knowledge mastery, self-regulated learning, and longer-term performance.

## 2. Literature Review

## 2.1. Feedback as a Personalized and Timely Learning Process

Feedback supports learning when students can connect it to a current goal and use it in subsequent action. Its effectiveness varies with message content, task characteristics, learner needs, and implementation context (Morris et al., 2021; Wisniewski et al., 2020). Personalization therefore requires both evidence about the learner and a pedagogical rationale for selecting a response. Reviews of digital learning environments show that most systems adapt to current knowledge or observed behavior, while goals, affect, and progress over time are incorporated less consistently (Maier & Klotz, 2022). In programming, meaningful adaptation may vary the location, conceptual depth, or specificity of a hint according to the current attempt and recent revisions.

Timing should likewise reflect the learning situation. Early support can prevent repeated unproductive attempts, whereas premature intervention can disrupt productive struggle. An effective policy should respond to evidence of continuing difficulty rather than impose a uniform delay. Assistance intensity must be regulated for the same reason. Programming studies show that learners respond differently to broad prompts, localized hints, and explicit suggestions (Lohr et al., 2025; Xiao et al., 2024). Evidence from mathematics further suggests that unrestricted generative assistance can improve supported practice while reducing later unassisted performance, whereas instructional guardrails can mitigate this effect (Bastani et al., 2025). Adaptive feedback should therefore coordinate intervention timing with progressively adjusted support.

Feedback content should also preserve an active role for the learner. A broad selfcheck may be sufficient when progress is evident, while repeated failure may justify a concept reminder or localized hint. More explicit guidance should be reserved for sustained difficulty. This progression links personalization with formative purpose: assistance changes with observed need, while each message directs attention toward interpretation, revision, and verification. It also offers a practical basis for evaluating whether support remains useful without revealing more of the solution than the situation requires.

## 2.2. Learning Analytics, Learner Models, and Self-Regulated Learning

Learning analytics organizes interaction traces into indicators of performance, effort, and change over time. Higher-education research shows that these indicators can identify risk, visualize progress, and guide attention toward useful next actions (Banihashem et al., 2022; Ifenthaler & Yau, 2020). Dashboards and open learner models can make such evidence visible and support reflection, particularly when indicators are linked to strategies learners can enact (Hooshyar et al., 2020; Matcha et al., 2020; Paulsen & Lindsay, 2024). Generative AI extends this approach by translating selected indicators into contextualized explanations and scaffolds (Li et al., 2025).

Digital traces describe observable actions rather than internal states. Submission timing, test outcomes, revisions, and inactivity can indicate monitoring or persistence, but their interpretation depends on surrounding events. Trace-based research on selfregulated learning is strongest when indicators are linked to a theoretical construct and a defined temporal window (Du et al., 2023). Accordingly, an unchanged resubmission can represent a specific revision behavior, while broader claims about self-regulation require additional evidence. Similarly, later task performance can operationalize near-term transfer or longer-term success without being equated with complete knowledge mastery.

This distinction also shapes explainability. Learner-state representations should expose the observations supporting a decision rather than only a risk score. Failed checks, revision counts, elapsed time, and prior attempts can justify intervention urgency or assistance level, while claims about specific misconceptions should be grounded in task or code evidence. This approach follows open learner model research by making analytics actionable and inspectable (Hooshyar et al., 2020).

Learning analytics also distinguishes prediction from pedagogical interpretation. A model may identify a high-risk state from weak scores, repeated failures, and limited code change, but the estimate alone does not explain why the learner is struggling. The intervention layer must translate that estimate into a bounded decision: whether to provide support, which recorded evidence to use, and what level of assistance is appropriate. This separation improves auditability and prevents statistical associations from being presented as diagnoses. It also allows different outcome models to inform immediate recovery, revision behavior, related-task performance, and longer-term performance.

## 2.3. Sequential Modeling of Learning Processes

Programming-process data preserve the order of attempts, edits, tests, and task transitions, enabling analysis of how a learner reached a given state (Price et al., 2020). This temporal order matters because identical failed outputs may emerge from different histories and therefore warrant different forms of support. Knowledge tracing research has accordingly progressed from probabilistic approaches toward recurrent, memory-based, graph-based, and attention-based models (Abdelrahman et al., 2023). Gated recurrent units summarize variable-length histories through learned gating mechanisms (Cho et al., 2014), whereas Transformer encoders use self-attention to model dependencies across sequence positions (Vaswani et al., 2017).

The present task extends conventional knowledge tracing by predicting whether an observed programming failure persists across the next two attempts. Recurrent and attention-based models test the value of ordered history, while logistic regression and gradient boosting provide structured-data baselines. Model selection should consider calibration and transparency alongside discrimination because estimated probabilities determine intervention priority. Student-disjoint partitions, training-only preprocessing, and prefixrestricted inputs are therefore essential, as random event splits or future-derived features can substantially inflate performance (Kapoor & Narayanan, 2023). Validation-fitted probability calibration further aligns predicted risk with capacity-aware decisions (Guo et al., 2017).

These model families provide complementary evidence. Tabular baselines show how much can be learned from the current state and summarized history with relatively direct interpretation. GRUs test whether gated recurrence captures meaningful progression across attempts, while Transformers test whether attention across positions improves representations of longer dependencies. Their comparison is informative only when preprocessing, data partitions, outcome definitions, and selection rules remain fixed. Under limited instructional capacity, calibrated probabilities are especially useful because validationselected thresholds can be translated into defined intervention volumes rather than treated as abstract model scores.

## 2.4. Generative AI Feedback in Programming Education

Large language models have extended programming support beyond fixed templates to natural-language explanations and hints. Prior work has demonstrated automated exercise and explanation generation, novice-oriented error messages, multi-level hints, and specified feedback types (Leinonen et al., 2023; Lohr et al., 2025; Sarsa et al., 2022; Xiao et al., 2024). These studies establish technical feasibility, but feedback quality still varies across tasks, prompts, and evaluation criteria.

Evaluation has therefore moved beyond fluency. Recent studies examine correctness, completeness, comprehensibility, and repair accuracy, while also showing that language models remain imperfect judges of generated feedback (Koutcheme et al., 2025). Research on writing feedback likewise reports benefits for revision and selected quality dimensions (Meyer et al., 2024; Steiss et al., 2024). Across higher education, systematic evidence highlights the potential of generative feedback alongside the importance of accuracy, transparency, learner agency, and instructor oversight (Lee & Moore, 2024). Message quality and learning effectiveness should therefore be evaluated as distinct outcomes.

Grounding is especially important when feedback refers to code, tests, or learning history. A plausible response may still contain an unsupported diagnosis, so readability alone cannot prevent hallucination (Huang et al., 2025). An evidence-constrained generator should operate on a bounded learner record, distinguish observations from inferences, recommend a feasible next action, and preserve traceability to quoted sources. Explainability should also match stakeholder needs: feature attribution can support researcher audit (Lundberg & Lee, 2017), whereas learners need concise explanations linked to their next action and instructors may require fuller records of risk, evidence, and uncertainty (Khosravi et al., 2022).

Evaluation design should isolate the effect of contextual information from differences among cases or models. A within-case comparison can hold the task and generator constant while adding recent history, module context, calibrated risk, or an assistance instruction. Structural checks can assess whether responses contain an observation, an actionable step, and a self-check, while source verification can test whether quoted code appears in the supplied record. Human review can further assess correctness, relevance, and pedagogical appropriateness. Together, paired generation and explicit evidence checks clarify how additional context changes feedback and which messages warrant further review.

## 2.5. Synthesis and Research Gap

The literature establishes the core components of adaptive feedback but typically evaluates them in isolation. Feedback research examines timing and assistance; learning analytics provides behavioral evidence; sequential models estimate evolving risk; and generative models express support in natural language. A deployable system must connect these functions by identifying intervention opportunities, selecting time-valid evidence, regulating feedback specificity, and preserving an auditable record of each message.

Emerging work has begun to integrate real-time analytics with generative scaffolding (Li et al., 2025), yet calibration, intervention capacity, grounding, and progressive assistance are rarely examined together. The present framework connects observation, prediction, decision, generation, and evaluation through student-disjoint prediction, calibrated capacity-aware policies, bounded generation contexts, and source checks. This integration supports personalized, explainable, and timely feedback while establishing a clear basis for subsequent evaluation of knowledge mastery, self-regulated learning, transfer, and retention.

This integration also makes each claim testable at the appropriate stage. Predictive validity concerns held-out outcomes and calibration; policy value concerns the precision and coverage of selected intervention opportunities; and feedback quality concerns structure, actionability, and traceability. Keeping these endpoints connected but distinct enables a more rigorous assessment of the overall feedback strategy.

## 3. Methodology

## 3.1. Research Design and Scope

This study used a retrospective computational design combining learner-state prediction, paired feedback generation, and historical-log policy evaluation. The unit of analysis was a failed programming state at which feedback could be considered. The framework comprised five linked stages: (1) construct a time-valid representation from the current submission and prior learning traces; (2) estimate the risk of a defined future outcome; (3) map calibrated risk to an intervention decision under fixed feedback capacity; (4) generate feedback from bounded source evidence; and (5) evaluate prediction, calibration, policy capture, and observable message properties. This structure aligns each research claim with a corresponding evaluation endpoint.

The analyses addressed two complementary components. The predictive component examined whether current-state and sequential features could identify students likely to remain unsuccessful across their next two attempts and predict related behavioral and performance outcomes from the same traces. The generation component examined whether progressively richer, source-bounded context changed feedback structure and traceability while holding the language model and decoding procedure constant. Intervention timing was evaluated by replaying decision rules over recorded trajectories and comparing the opportunities selected by alternative policies.

Figure 1 summarizes the complete framework. Learning evidence progresses through observation, prediction, decision, generation, and evaluation, while the evidence-gated progressive assistance policy links calibrated risk to feedback specificity. Each subsequent attempt contributes new evidence for the next decision, creating a sequential cycle of adaptive support.

## 3.2. Dataset, Participants, and Analytical Cohort

The study used the public ProgFeed dataset, a de-identified record of an introductory programming course delivered in Fall 2025 (UMass ML4Ed, 2026). The repository includes programming submissions, autograder results, problem statements, recorded feedback conditions, and entry and exit surveys. Its documentation states that only students who consented to research use were included and that direct identifiers were removed. The data are released under a CC BY 4.0 license. A fixed repository snapshot was used throughout the analysis.

The consolidated source contained 17,385 graded function–test records from 215 students, representing 6693 submissions and 16,365 function-level submission states. Records sharing student, laboratory, source file, function, and timestamp were aggregated into a single state after confirming one code version and nonduplicated test identifiers. Test scores and maximum scores were summed within each state, and any failed functional check classified the state as unsuccessful. Student code was parsed as text for structural features but was never executed.

![](images/75be4ed26b495a2f40804f5a9225b11788f6944ddc408c06fb54e1adf80522a6.jpg)  
Figure 1. Overall research framework.

A candidate decision state required an unsuccessful attempt with nonempty code, a preceding attempt on the same task, and inclusion in the de-identified research dataset. These criteria produced 2993 candidate states. Students were assigned once to training, validation, or test partitions using a reproducible 60%/20%/20% split, with all records from each student retained in the same partition. Table 1 summarizes the cohort and primary-outcome coverage. The 238 endpoint-censored states were retained in prediction and deployment-oriented records, while supervised performance estimates used the 2755 states with observed outcomes.

Table 1. Student-disjoint partitions and primary-outcome availability.
<table><tr><td>Partition</td><td>Students</td><td>Candidate states</td><td>Known outcome</td><td>Unknown outcome</td><td>Persistent failure</td></tr><tr><td>Training</td><td>129</td><td>1897</td><td>1756</td><td>141</td><td>847</td></tr><tr><td>Validation</td><td>43</td><td>499</td><td>444</td><td>55</td><td>183</td></tr><tr><td>Test</td><td>43</td><td>597</td><td>555</td><td>42</td><td>222</td></tr><tr><td>Total</td><td>215</td><td>2993</td><td>2755</td><td>238</td><td>1252</td></tr></table>

The recorded A–D feedback subset comprised 136 matched cases and 544 messages, each with a human-reviewed overall rating on a 0–100 scale. A separate generation experiment used the same cases to produce 544 F1–F4 messages, evaluated through structural and source checks.

## 3.3. Outcome Construction and Measurement Boundaries

The primary outcome, next-two-attempt persistent failure, was defined from the next two chronological states for the same student and task. The outcome was coded 0 if either attempt passed and 1 if both observed attempts failed. When fewer than two subsequent attempts were available and no pass was observed, the outcome remained missing. Future states were used only to construct outcomes and were never supplied as predictors or generation context.

Three secondary outcomes were derived from later observed records. An unmodified retry indicated that the next same-task submission had the same canonical abstract syntax tree (AST) as the current code. If either version could not be parsed, code normalized for comments and whitespace was compared instead. This variable captures an observable revision behavior related to monitoring and strategy change. For the two taskperformance outcomes, tasks were assigned before model fitting to seven broad content groups: basic input/output and arithmetic, conditional logic, iteration, collections, file input/output, classes and state, and recursion. Near-term related-task performance recorded functional success on the first future distinct task in the same broad group within seven days. Longitudinal related-task performance used the corresponding first observation after 7 days and within 28 days. When no qualifying future task was observed, the label was coded as missing.

Table 2 summarizes the operational definitions and coverage. The two related-task outcomes capture transfer-like and longitudinal performance under ordinary course conditions. Their explicit content and temporal definitions support theory-aware interpretation of trace-derived indicators (Du et al., 2023) and provide observable targets for the computational evaluation.

Table 2. Operational outcomes used in the predictive analyses.
<table><tr><td>Outcome</td><td>Operational definition</td><td>Cases</td><td>Studentsª</td></tr><tr><td>Persistent failure</td><td>Both of the next two same-task attempts failed; a pass in either attempt was coded as nonpersistent.</td><td>2755</td><td>198</td></tr><tr><td>Unmodified retry</td><td>Next same-task code had an identical canon- ical AST, with normalized-text fallback for unparseable code.</td><td>2847</td><td>198</td></tr><tr><td>Near-term related- task performance</td><td>Functional success on the first distinct task in the same broad content group within 7 days.</td><td>950</td><td>147</td></tr><tr><td>Longitudinal</td><td>Functional success on the first distinct same-</td><td>307</td><td>105</td></tr><tr><td>related-task per- formance</td><td>group task after 7 days and within 28 days.</td><td></td><td></td></tr></table>

<sup>a</sup> Student counts are summed across disjoint partitions and may include the same student in more than one outcome row.

## 3.4. Time-Valid Feature Construction

Static predictors described the learner state at the decision time. They included attempt count; current and previous score ratios; score change; numbers of failed and total checks; failures across the three most recent attempts; consecutive failures; elapsed time since the previous attempt; code length; edit fraction; and a syntax-parseability indicator. Ten additional counts captured AST structure: total nodes, calls, conditional statements, for and while loops, returns, binary operations, comparisons, function definitions, and literals. Laboratory, source file, and function identifiers were represented categorically. Student identifiers were used only for partitioning and clustered evaluation and never as model inputs.

Continuous missing values were imputed with training-set medians and paired with explicit missingness indicators. Positively skewed count and time variables were transformed using log(1 + x), after which continuous features were standardized with means and standard deviations estimated from known-outcome training cases. Categorical vocabularies were fitted on the training partition, with unseen validation or test categories mapped to an unknown level before one-hot encoding. AST features were derived through parsing only; parsing failures remained missing and were represented by indicators. No student or autograder program was executed during preprocessing.

The primary sequence representation contained up to 20 states from the same student, laboratory, source file, and function, ending at the decision state. Each timestep comprised eight values—score ratio, failed-check count, total-check count, failure status, attempt index, elapsed time, score change, and consecutive failures—together with eight missingness indicators. Transformations and normalization were fitted only on sequence states within training prefixes. Shorter sequences were padded within batches, with true lengths supplied to the recurrent model or used to construct a Transformer padding mask.

A follow-up across-task sequence examined whether broader student history improved prediction beyond same-task prefixes. It included the latest 20 completed states for the same student across tasks, restricted to timestamps no later than the current state. Each 16-dimensional timestep combined eight normalized numeric variables, seven onehot content-group indicators, and a score-ratio missingness indicator. These analyses are reported as validation-informed extensions of the primary representation.

## 3.5. Predictive Models and Training Procedure

Eight model configurations were compared for the primary outcome. Logistic regression and histogram gradient boosting served as validation-tuned tabular baselines. The deep-learning comparison included a multilayer perceptron (MLP), a sequence-only gated recurrent unit (GRU), a static–sequence fusion GRU, the same fusion model without AST predictors, a fusion Transformer, and a multitask fusion GRU. GRUs use learned gates to retain relevant sequential information (Cho et al., 2014), whereas Transformers model cross-position relations through self-attention (Vaswani et al., 2017). Their inclusion also reflects the broader knowledge-tracing literature, although the present target is observed future failure rather than latent mastery (Abdelrahman et al., 2023).

For the MLP and fusion models, the static vector was projected to 32 units with rectified linear activation and dropout of 0.20. The GRU used a 16-dimensional timestep input and a hidden size of 32. In the Transformer, the timestep vector was projected to 32 dimensions and combined with a learned positional embedding for 20 positions. A single encoder layer used four attention heads, a feed-forward dimension of 64, and dropout of 0.20. Fusion models concatenated the 32-dimensional static and sequential representations, followed by a prediction head with a 32-unit hidden layer, rectified activation, dropout of 0.20, and a sigmoid output. The MLP used only the static representation, while the sequence-only GRU omitted the static branch.

Deep models were trained with AdamW (Loshchilov & Hutter, 2019) and validationbased early stopping on precision–recall area under the curve (PR-AUC). Training was repeated with five random seeds, and predicted probabilities were averaged across runs. The multitask model jointly predicted all eligible outcomes, assigned greater weight to the primary outcome, and masked missing labels from the corresponding secondary losses.

Hyperparameters, early stopping, and model selection used only the validation partition. The test set was reserved for final performance estimation and was not used for model selection. Logistic regression, gradient boosting, and separate fusion-GRU models were also fitted for each secondary outcome using all cases with an observed label for that outcome, including states with missing primary outcomes. For the follow-up acrosstask sequence, a multitask fusion GRU and outcome-specific GRUs were trained using the same optimization procedure. Summary-feature logistic and gradient-boosting baselines used the same across-task history source to separate gains from sequence modeling from gains due solely to broader history.

For probability calibration, Platt scaling was fitted to validation predictions from the selected deep ensemble by applying logistic regression to the logit-transformed ensemble probability (Guo et al., 2017). The calibrator and all decision thresholds were fixed before test-set evaluation. The across-task multitask GRU used a separate validation-fitted calibrator for the risk-adaptive strategy.

## 3.6. Predictive Evaluation and Interpretation

PR-AUC, computed as noninterpolated average precision, was the primary discrimination metric because persistent failure was unevenly distributed. ROC-AUC, Brier score, logarithmic loss, F1 score at a 0.50 probability threshold, and balanced accuracy were also reported. Calibration was assessed using quantile-binned reliability curves. To account for repeated states within learners, 95% confidence intervals were estimated from 1000 student-level bootstrap samples. Paired bootstrap differences used the same resampled students for each model comparison. Student-disjoint evaluation and time-restricted predictors reduced the risk of within-student and temporal leakage (Kapoor & Narayanan, 2023).

Interpretability analyses were separated from learner-facing explanations. Logistic coefficients represented conditional associations in the standardized feature space. For gradient boosting, features were grouped into score and failure, timing and attempts, code edits, AST structure, and task identifiers; each group was permuted 30 times on the test set, and the resulting change in PR-AUC was recorded. Attention weights were not treated as explanations. These analyses supported model audit, whereas learner-facing feedback was grounded in observable task or code evidence linked to the next action.

## 3.7. Paired Feedback Generation and Progressive Assistance

The generation experiment used the same 136 cases across four within-case conditions. F1 supplied the task statement, current submitted code, and recorded failed checks. F2 added the two most recent states and compact history indicators, including changes in failure counts and whether the code had changed. F3 added the complete current module to support checks of dependencies and code locations. F4 combined F2 and F3 with calibrated persistent-failure risk from the across-task multitask GRU and an assistancelevel instruction. Holding the cases and language model constant supported within-case comparisons of these context-and-policy configurations.

Risk-adaptive assistance in F4 used the 25th and 75th percentiles of validation-set calibrated risk. Cases below the lower threshold received a self-check-oriented instruction; middle-risk cases received a concept reminder and one minimal action; high-risk cases received a localized hint and the smallest supported next step. When the standard deviation across the five raw model probabilities exceeded 0.15, the instruction prioritized a verifiable check rather than a single causal diagnosis. This ensemble-disagreement rule served as an operational review heuristic.

All 544 outputs were generated with Qwen2.5-Coder-1.5B-Instruct (Hui et al., 2024). The same model, deterministic decoding procedure, and context limits were used throughout. Each response followed a compact structure containing an evidence-based observation, a feasible next action, a learner self-check, and an optional exact code excerpt. The instruction prohibited complete solutions and claims not supported by the supplied evidence.

Automated checks assessed structural completeness, actionability, self-check content, and whether quoted code matched the supplied source. Outputs with a structural error or unverifiable reference received one standardized repair attempt. Any source reference that remained unverifiable was removed, and unresolved structural cases were flagged for human review.

## 3.8. Intervention Timing and Capacity Evaluation

Two policy evaluations were conducted. The fixed-capacity analysis ranked the same labeled test pool by calibrated deep-model risk, risk from the best validation-selected model, consecutive-failure count, or attempt count. Budgets of 5%, 10%, 20%, 30%, 40%,

50%, 75%, and 100% of eligible states were evaluated. A random-alert baseline was estimated from 1000 draws at each budget. The principal policy measures were precision among selected states and the proportion of observed persistent failures captured. Ties were resolved using case identifiers without outcome information.

The sequential analysis represented a deployable trigger at each candidate state. A risk threshold was fixed at the 80th percentile of calibrated validation risk and applied unchanged to the test sequence. This policy was compared with alerting after every failure and after two consecutive failures. Evaluation included alert coverage, precision, persistent-failure capture, and observed time from the alert state to the next attempt. The 238 states with endpoint-censored primary outcomes still received predictions, while policy-performance denominators used only observed outcomes. The analysis therefore measures intervention prioritization across the recorded trajectories.

## 3.9. Additional Analysis

The source course included test-case feedback, natural-language feedback, and nofeedback conditions. Their associations with persistent failure were examined in the randomized course subset using a binomial generalized linear model with task and learnerstate covariates and student-clustered standard errors. Exit-survey responses were summarized descriptively as learner-reported context. Analyses were conducted in Python 3.10.19 with PyTorch 2.5.1 (Paszke et al., 2019).

## 4. Results

## 4.1. Outcome Availability and Test-Set Composition

The primary outcome was observed for 2755 of 2993 candidate states (92.1%). In the held-out test partition, 555 states from 42 students had an observed primary outcome, including 222 (40.0%) followed by persistent failure across the next two attempts. The remaining 42 test states retained predicted risk as endpoint-censored records outside supervised performance and policy denominators. An unmodified-retry outcome was available for 572 test states (180 positive; 31.5%), near-term related-task performance for 209 states (97 successful; 46.4%), and 7–28-day related-task performance for 67 states from 25 students (41 successful; 61.2%).

## 4.2. Prediction of Near-Term Persistent Failure

Table 3 reports primary test performance in validation PR-AUC order. Logistic regression was the validation-selected overall model (validation PR-AUC = 0.532). On held-out students, it achieved a PR-AUC of 0.550 (student-cluster 95% CI [0.361, 0.676]), ROC-AUC of 0.681 [0.585, 0.741], and Brier score of 0.222. The MLP was the validation-selected deep model (validation PR-AUC = 0.525), with a test PR-AUC of 0.525 [0.352, 0.651], ROC-AUC of 0.670 [0.577, 0.732], and Brier score of 0.224. Overall, logistic regression provided the strongest validation-selected performance with competitive held-out discrimination and calibration.

Several models achieved higher numerical test PR-AUC than their validation rankings suggested. The no-AST fusion GRU yielded the highest test PR-AUC (0.596) and ROC-AUC (0.697), despite a validation PR-AUC of 0.508. This finding is reported as an informative ablation result while retaining validation-based model selection. The multitask GRU achieved a test PR-AUC of 0.565 and ROC-AUC of 0.689, while the sequence-only GRU achieved PR-AUC 0.566 and ROC-AUC 0.628. Figure 2 summarizes the estimates and student-cluster intervals across all eight configurations.

The selected MLP outperformed gradient boosting in the paired student bootstrap, with a PR-AUC difference of 0.082 and a 95% interval of [0.009, 0.132]. Together with the logistic regression results, this comparison highlights the importance of benchmarking neural models against transparent baselines and matching model complexity to the predictive task.

Table 3. Prediction of persistent failure on the student-disjoint test set. Confidence intervals were obtained by resampling students.
<table><tr><td>Model</td><td>Validation PR-AUC</td><td>Test PR-AUC</td><td>PR-AUC 95% CI</td><td>Test ROC-AUC</td><td>Brier</td></tr><tr><td>Logistic regression</td><td>0.532</td><td>0.550</td><td>[0.361, 0.676]</td><td>0.681</td><td>0.222</td></tr><tr><td>MLP</td><td>0.525</td><td>0.525</td><td>[0.352, 0.651]</td><td>0.670</td><td>0.224</td></tr><tr><td>GRU fusion, multitask</td><td>0.524</td><td>0.565</td><td>[0.350, 0.699]</td><td>0.689</td><td>0.230</td></tr><tr><td>Transformer fusion</td><td>0.519</td><td>0.547</td><td>[0.345, 0.669]</td><td>0.668</td><td>0.229</td></tr><tr><td>GRU fusion</td><td>0.518</td><td>0.551</td><td>[0.350, 0.683]</td><td>0.684</td><td>0.223</td></tr><tr><td>Gradient boosting</td><td>0.511</td><td>0.443</td><td>[0.330, 0.547]</td><td>0.575</td><td>0.251</td></tr><tr><td>GRU, sequence only</td><td>0.509</td><td>0.566</td><td>[0.294, 0.686]</td><td>0.628</td><td>0.234</td></tr><tr><td>GRU fusion, no AST</td><td>0.508</td><td>0.596</td><td>[0.347, 0.715]</td><td>0.697</td><td>0.227</td></tr></table>

![](images/67a51187e03f54b9ea52617b59b6ae919bcd561e607a0bbcba96692a0c67de0b.jpg)

![](images/550fbf043f9453df950b777e0e92a820a60615987ae5acd4aca84c3fae6f16b6.jpg)  
Figure 2. Primary model performance on held-out students. Error bars show student-cluster 95% confidence intervals. Models are ordered by validation PR-AUC; test results were not used for model selection.

## 4.3. Feature Groups, Calibration, and Task Heterogeneity

Grouped permutation analysis of the gradient-boosting model showed the largest mean PR-AUC decrease for task identifiers (mean decrease = 0.052, SD = 0.010). Score and failure features produced a decrease of 0.011 (SD = 0.007), followed by AST structure at 0.009 (SD = 0.013) and code-edit features at 0.001 (SD = 0.012). Timing and attempt features yielded a mean PR-AUC change of -0.052 (SD = 0.014), reflecting overlap and interaction with other feature groups. Figure 3 summarizes these group-level results.

The logistic model showed similar task dependence, with its largest absolute coefficients including specific function and laboratory indicators. Among continuous variables, consecutive failures had a positive standardized coefficient (0.429), whereas attempt count had a negative coefficient (-0.358); AST node count was positive (0.348), while AST return count was negative (-0.282). These conditional associations show how the model integrated task context with current learner behavior. Laboratory-level ROC-AUC ranged from 0.175 to 0.720 for the follow-up student-history model and from 0.429 to 0.714 for logistic regression among subgroups with at least 20 states. These results highlight meaningful heterogeneity across task contexts.

## 4.4. Secondary Outcomes and Across-Task Student Histories

Table 4 reports dedicated models trained separately for each secondary outcome. Unmodified retry was the most consistently predictable secondary behavior. Gradient boosting, which achieved the highest validation PR-AUC in the dedicated comparison, obtained a test PR-AUC of 0.599 and ROC-AUC of 0.735. With across-task history, the outcome-specific student GRU reached PR-AUC 0.631 and ROC-AUC 0.764, while the history-summary gradient-boosting model reached PR-AUC 0.659 and ROC-AUC 0.776. These results indicate that broader student history contributed useful information and that both sequential and summary representations captured this behavior.

![](images/9a29fd04fc18a76798eeade2f313ea91a3c5cebe30dfa6ab0be606183f273422.jpg)  
Figure 3. Grouped permutation importance for gradient boosting. Values show changes in test PR-AUC across 30 permutations; negative values indicate improved test performance after permutation.

Table 4. Dedicated model performance for the three secondary outcomes.
<table><tr><td>Outcome</td><td>Model</td><td>Test cases</td><td>Validation PR-AUC</td><td>Test PR-AUC</td><td>Test ROC-AUC</td></tr><tr><td rowspan="3">Unmodified retry</td><td>Logistic regression</td><td>572</td><td>0.395</td><td>0.502</td><td>0.654</td></tr><tr><td>Gradient boosting</td><td>572</td><td>0.445</td><td>0.599</td><td>0.735</td></tr><tr><td>GRU fusion</td><td>572</td><td>0.422</td><td>0.466</td><td>0.650</td></tr><tr><td rowspan="3">Near-term related-task performance</td><td>Logistic regression</td><td>209</td><td>0.564</td><td>0.432</td><td>0.433</td></tr><tr><td>Gradient boosting</td><td>209</td><td>0.538</td><td>0.392</td><td>0.383</td></tr><tr><td>GRU fusion</td><td>209</td><td>0.608</td><td>0.424</td><td>0.303</td></tr><tr><td rowspan="3">Related-task performance at 7-28 days</td><td>Logistic regression</td><td>67</td><td>0.679</td><td>0.825</td><td>0.691</td></tr><tr><td>Gradient boosting</td><td>67</td><td>0.650</td><td>0.820</td><td>0.699</td></tr><tr><td>GRU fusion</td><td>67</td><td>0.681</td><td>0.636</td><td>0.521</td></tr></table>

Near-term related-task performance was the most challenging prediction target. The dedicated fusion GRU achieved validation PR-AUC 0.608 and test ROC-AUC 0.303; logistic regression reached ROC-AUC 0.433, and gradient boosting reached 0.383. The outcomespecific across-task GRU achieved validation PR-AUC 0.718, test PR-AUC 0.423, and ROC-AUC 0.302. These results highlight richer knowledge-component and instructionalcontext representations as an important direction for improving near-term transfer prediction.

The 7–28-day outcome showed promising longitudinal discrimination across 67 test cases from 25 students. Dedicated logistic regression and gradient boosting achieved ROC-AUC values of 0.691 and 0.699, respectively, with student-cluster intervals extending from approximately 0.49 to 0.85. The across-task multitask GRU reached PR-AUC 0.795 and ROC-AUC 0.683, with a ROC interval of [0.466, 0.857]. A history-summary logistic model achieved ROC-AUC 0.710 and PR-AUC 0.822. These estimates support further longitudinal validation with larger samples.

For the primary outcome, the follow-up across-task multitask GRU achieved validation PR-AUC 0.543, test PR-AUC 0.557, and ROC-AUC 0.669. Its PR-AUC differed from logistic regression by 0.008 in the paired student bootstrap, with a 95% interval of [-0.101, 0.086]. Figure 4 compares the shared multitask and outcome-specific models across the three secondary outcomes. The clearest improvement occurred for observable retry behavior, while the knowledge-related outcomes identify priorities for richer measurement.

![](images/7574c00f112da3b9fd6309c4d444cc34f5154b12b31ff47681cc555e5e92e300.jpg)

![](images/97c689ebd76199e9f57414b91b14d8458066c37a44b888c1fd348fd7dd206b43.jpg)

![](images/d5767208c02031880e2b959435e99b2aed55d44c24539851de875797456f8a05.jpg)  
Figure 4. Follow-up across-task student-history results. Bars show test ROC-AUC for the multitask and outcome-specific GRUs; error bars resample students.

## 4.5. Recorded Feedback Scores and Newly Generated Feedback

The recorded A–D feedback conditions each contained 136 human-reviewed scores. Mean scores increased monotonically from A (mean = 63.17, SD = 7.68) through B (71.17, SD = 7.68) and C (78.13, SD = 7.81) to D (86.13, SD = 7.81). Mean feedback lengths were 44.5, 71.1, 62.8, and 89.4 words, respectively. Paired mean differences were 8.0 points for B minus A, 15.0 for C minus A, 23.0 for D minus A, 15.0 for D minus B, and 8.0 for D minus C. These within-case contrasts summarize the recorded human assessments under the adopted rubric, including evidence and history dimensions.

The new generation procedure produced all 544 planned messages. Before repair, structurally valid responses were obtained for 96.3% of F1 and F2 outputs, 91.9% of F3 outputs, and 92.6% of F4 outputs. All required components were present in 77.2%, 89.0%, 90.4%, and 92.6%, respectively. A self-check appeared in 96.3% of F1 and F2 outputs, 91.9% of F3, and 92.6% of F4. Mean rendered length remained similar across conditions (59.5– 60.9 words). No input exceeded the context limit, and one F4 output (0.7%) reached the output limit.

The source-verification gate identified exact code matches in 5 F1, 4 F2, 2 F3, and 5 F4 messages, totaling 16 of 544 outputs (2.9%). A standardized repair pass processed 528 outputs. After repair and deterministic reference gating, 519 outputs (95.4%) contained all required components, while 25 required further structural review. Final condition-specific completeness rates were 97.1% for F1, 94.9% for F2, 92.6% for F3, and 97.1% for F4 (Table 5). The gate removed 484 unverifiable source references so that only exact matches were retained. Figure 5 shows the initial and final checks.

Table 5. Structural validation after one repair pass and deterministic reference gating. Rates are calculated within 136 cases per condition.
<table><tr><td>Condition</td><td>Initial complete</td><td>Repair attempted</td><td>Final complete</td><td>Exact reference</td><td>Reference removed</td><td>Manual review</td></tr><tr><td>F1</td><td>77.2%</td><td>96.3%</td><td>97.1%</td><td>3.7%</td><td>82.4%</td><td>2.9%</td></tr><tr><td>F2</td><td>89.0%</td><td>97.1%</td><td>94.9%</td><td>2.9%</td><td>89.7%</td><td>5.1%</td></tr><tr><td>F3</td><td>90.4%</td><td>98.5%</td><td>92.6%</td><td>1.5%</td><td>90.4%</td><td>7.4%</td></tr><tr><td>F4</td><td>92.6%</td><td>96.3%</td><td>97.1%</td><td>3.7%</td><td>93.4%</td><td>2.9%</td></tr></table>

Risk-based instructions were assigned to all 136 F4 cases: 31 received the self-check level, 80 the concept-hint level, and 25 the localized-hint level. All cases remained below the prespecified ensemble-disagreement threshold. These results confirm that the prediction-to-assistance mapping operated as intended and yielded a complete set of messages for subsequent expert and learner evaluation.

## 4.6. Intervention Capacity and Timing

Figure 6 compares the across-task multitask policy with logistic risk, consecutive failures, and random selection under the same capacity. At a 5% budget (28 alerts), the student-history policy captured 21 of 222 persistent failures, yielding precision 0.750 and capture recall 0.095. Logistic risk captured 16 cases (precision 0.571), while the consecutivefailure rule captured 20 (precision 0.714). At a 10% budget, consecutive failures performed best, capturing 47 cases compared with 38 for the student-history model and 34 for logistic regression.

![](images/42a2d2c7085adbb0cc1c4699f076b406c867ee73842f08a90fd4120ebad9425f.jpg)

![](images/63b60fe936d7c6b8e20d06b003e3b2d2baa9c7afaabf8796f446e3f2c05cc7f6.jpg)

![](images/cdb070f14f598e9bc7e781bb97b9a8160b9b13593b20457a95022df61be2ec6a.jpg)  
Figure 5. Initial and final structural checks for newly generated feedback. Exact source-reference rates remained low, and the gate removed unverifiable references rather than retaining unsupported locations.

At the planned 20% budget (111 alerts), the student-history policy captured 62 persistent failures (precision = 0.559; capture recall = 0.279). Logistic regression captured 66 (precision = 0.595; recall = 0.297), while consecutive failures captured 75 (precision = 0.676; recall = 0.338). The student-history minus logistic differences were -0.036 for precision (95% student-bootstrap interval [-0.287, 0.116]) and -0.018 for capture recall [-0.149, 0.057]. At a 50% budget, capture recall reached 0.667 for student history, 0.685 for logistic regression, and 0.581 for consecutive failures. These profiles show how policy performance varies with alert capacity and the desired balance between precision and capture.

![](images/fbd2b929fb5baf2cdc186040d1912e1f1d9af64892d338894641d4eb06ad126f.jpg)

![](images/441f5c5ef34741fe3fd36f8b9646d44775c27c296bbc682ffc57224d55422671.jpg)  
Figure 6. Fixed-capacity comparison on the labeled test pool. Curves show precision and persistentfailure capture under recorded outcomes.

For the sequential student-history policy, the validation-fixed threshold was 0.500. Applied to the test sequence, it triggered on 17.8% of states, with precision 0.566 and capture recall 0.252. A sensitivity analysis using the validation-selected core deep model yielded a lower threshold of 0.492, 22.0% coverage, precision 0.607, and capture recall 0.333. By comparison, alerting after every failure covered all states at the base precision of 0.400, while the two-consecutive-failures rule covered 97.7% and captured 99.1% of persistent failures. This rule reduced alert burden only marginally because most candidate states already occurred within repeated-failure sequences.

Validation-only calibration compressed the range of student-history probabilities and supported a fixed test-time threshold. Figure 7 shows test reliability bins for the calibrated student-history model and logistic regression, providing a transparent basis for threshold selection.

![](images/7da67063174db7d9d109fbae18b5d4c673972d68fca5815e7b4f71690fa1e22f.jpg)  
Figure 7. Calibration of the across-task student-history model before and after validation-fitted scaling, with logistic regression for comparison.

## 4.7. Historical Course Feedback and Survey Context

The historical analysis of the randomized course subset included 552 candidate states from 124 students. After adjustment for task, attempt count, consecutive failures, and edit fraction, the natural-language condition was associated with lower odds of subsequent persistent failure than the no-feedback condition in the task-fixed-effects model (odds ratio = 0.234, 95% CI [0.091, 0.602]). The estimate was similar with laboratory fixed effects (odds ratio = 0.235 [0.094, 0.588]). The test-case condition yielded an odds ratio of 1.203 [0.639, 2.264] with task fixed effects and 1.166 [0.656, 2.073] with laboratory fixed effects (Figure 8). These adjusted associations complement the predictive and generation analyses and inform evaluation of the newly generated F1–F4 messages.

![](images/d0c9d0f90b4a7559f2a729aef23f5657a4cf422bcefc22df553ba8801d3de435.jpg)  
Figure 8. Adjusted associations for recorded source-course feedback conditions. Intervals use student-cluster standard errors.

Among 37 exit-survey respondents, 31 reported encountering generated feedback. Ratings were tabulated for all 37 respondents: 19 rated its helpfulness as 4 or 5 on the fivepoint scale, 13 selected $^ { 3 , }$ and five selected 1 or 2 (mean = 3.49). Fourteen rated learning attributed to feedback as 4 or 5, 12 selected 3, and 11 selected 1 or 2 (mean = 3.16). These ratings describe the full respondent sample.

Taken together, the results address the four research questions and demonstrate the feasibility of an auditable prediction–generation pipeline. Current and historical traces predicted persistent failure with moderate discrimination, with logistic regression providing a strong and interpretable benchmark. Broader histories improved prediction of observable resubmission behavior, while the 7–28-day analysis revealed a longitudinal performance signal. The generation pipeline produced concise, predominantly complete feedback structures after repair and routed unverifiable code references through explicit quality control. Risk-based policies concentrated persistent failures within restricted alert budgets, establishing an empirical basis for adapting assistance thresholds to instructional capacity.

## 5. Discussion

## 5.1. Principal Findings

This study integrated learning traces, predictive modeling, and evidence-constrained generation into an auditable system for personalized programming feedback. For RQ1, current-state and historical features predicted persistent failure with moderate discrimination under student-disjoint evaluation. The validation-selected logistic model achieved a test PR-AUC of 0.550 and ROC-AUC of 0.681, while unmodified retry and the 7–28- day outcome also showed useful predictive signal. Near-term related-task performance emerged as a distinct target requiring richer knowledge-component and instructionalcontext representations.

For RQ2, the comparison showed a practical balance between performance and parsimony. The validation-selected MLP performed similarly to logistic regression on the primary outcome, while recurrent models added value for observable resubmission behavior. Sequence models are therefore most useful when ordered history contributes stable predictive information, whereas logistic regression remains an efficient and transparent choice when discrimination and calibration are comparable.

For RQ3, standardized repair produced compact and structurally complete feedback in most cases. Source verification served as the central quality-control mechanism: unsupported code references were removed and unresolved outputs were routed for review. The matched four-condition design enables within-case comparison of contextual evidence and assistance instructions, providing a basis for expert and learner evaluation.

For RQ4, risk ranking concentrated persistent failures more effectively than random selection under restricted alert budgets. The sequential policy selected 17.8% of eligible test states and captured 25.2% of observed persistent failures, while a more sensitive configuration increased capture at higher coverage. These results support intervention thresholds that reflect available instructional capacity and the relative costs of missed and unnecessary support.

Together, the findings support a unified interpretation. Predictive models identify where support may be useful, calibrated policies determine when limited attention should be allocated, and evidence constraints govern what feedback may claim. Risk guides allocation, observable evidence grounds the message, and subsequent learner responses inform the next decision. The contribution therefore lies in coordinating prediction, decision, and generation around a defined learning process rather than optimizing prediction accuracy or message quality in isolation.

## 5.2. Predictive Modeling, Measurement, and the Value of Learning Histories

The competitiveness of logistic regression indicates that deep learning adds greatest value when an outcome benefits from temporal representation. Recurrent and attentionbased models can capture evolving interaction histories (Abdelrahman et al., 2023); here, broader histories were particularly informative for resubmission behavior, whereas current score, repeated failure, attempt history, and task identity carried substantial signal for the primary outcome. Comparing transparent and deep models therefore helps align model complexity with the educational target.

Task identifiers showed the largest positive grouped importance in the boosting model, indicating that failure patterns were strongly associated with programming context. This finding supports task-aware calibration and reinforces the use of concrete task, test, and code evidence in learner-facing feedback. Differences across secondary outcomes likewise favor outcome-specific representations. An unmodified retry can motivate a reflective prompt about monitoring and strategy change, whereas later related-task performance requires a broader account of learning opportunities. These distinctions follow recommendations to link behavioral traces to explicit constructs and temporal windows (Du et al., 2023).

The later outcomes further clarify the value of learning history. Recent attempts directly characterize persistence and revision behavior, whereas performance on another task depends on content similarity, intervening instruction, practice opportunities, and time. A single learner representation is therefore unlikely to serve all outcomes equally well. Combining same-task sequences with broader histories provides a principled way to distinguish immediate debugging support from feedback intended to promote transfer or longer-term learning.

Student-disjoint partitions and decision-time prefixes reduced identity and temporal leakage (Kapoor & Narayanan, 2023), while validation-based selection preserved an independent final test. Calibration and capacity curves then translated predicted risk into alert volume, precision, and failure capture. This connection is important because intervention value depends on operational decisions as well as predictive discrimination.

## 5.3. From Risk Scores to Evidence-Gated Progressive Assistance

The findings support an evidence-gated progressive assistance strategy built around four connected decisions. A timing gate combines calibrated risk with available support capacity. An evidence gate restricts feedback to observations available at that moment. An assistance gate adjusts specificity according to risk and recent behavior. A verification gate checks structure, source traceability, and model disagreement before delivery. Each subsequent attempt then provides new evidence for the next decision.

The F1–F4 conditions operationalize this strategy as a progressive sequence. At lower risk, feedback can prompt inspection of a test result or prediction of an output. Intermediate support adds a concept reminder and one minimal action. After repeated unsuccessful revisions, a localized hint can identify a relevant code region and suggest the smallest supported debugging step. These guardrails preserve independent reasoning while allowing assistance to become more specific when needed (Bastani et al., 2025); the self-check component keeps learners responsible for verifying the explanation.

Evidence verification is essential because fluent messages can extend beyond the supplied record (Huang et al., 2025). Code-specific claims therefore require a resolvable source anchor, while messages with conflicting evidence or unresolved structure enter focused review. This design directs instructor attention toward cases where professional judgment is most valuable.

Explainability should reflect its audience. Learners need a concise account of the observable issue, the relevance of the next action, and a way to verify progress. Instructors need access to the supporting trace, calibrated risk, uncertainty, and repair history. This layered design follows educational explainability research, which links explanation form to stakeholder and purpose (Khosravi et al., 2022). Timing can then preserve productive debugging by presenting recorded failures first and offering stronger support when help is requested or continuing difficulty is observed.

The timing policy also preserves learner agency. Immediate intervention is appropriate when repeated evidence indicates an unresolved impasse, while a short learnercontrolled interval can support productive debugging. In practice, the system can first display recorded test evidence, then offer a scaffold on request or when the next attempt shows continuing difficulty. Capacity thresholds keep this sequence manageable by limiting intervention volume while preserving a clear escalation rule.

## 5.4. Contributions to Learning Analytics and Generative Feedback Research

The first contribution is a clear separation among prediction, decision, generation, and learning effects. Learning-analytics systems often present indicators without linking them to actionable strategies (Matcha et al., 2020; Paulsen & Lindsay, 2024). The present framework assigns distinct evidence to each claim: held-out outcomes assess prediction, policy replay assesses prioritization, and structural and provenance checks assess generated messages. This structure avoids treating plausible feedback as evidence of learning and can generalize to other settings that combine trace data with generative support.

The second contribution is the paired contextual design. The same 136 cases were represented under four context-and-policy configurations while the generator remained fixed. Prior studies established that language models can produce programming explanations, error messages, and hints at different levels (Leinonen et al., 2023; Lohr et al., 2025; Sarsa et al., 2022; Xiao et al., 2024). This study extends that work by examining how learner history and calibrated risk enter generation and by producing matched messages suitable for blinded assessment.

The third contribution is capacity-aware intervention design. Fixed-capacity curves and sequential replay show how precision and failure capture change as support expands. Instructors can therefore align thresholds with available review time and the cost of missed support, linking learning analytics to feedback purpose and stakeholder action (Banihashem et al., 2022; Ifenthaler & Yau, 2020). A staged implementation can begin with instructor review of risk, evidence, and escalation levels, followed by selective automation after local validation. This workflow preserves oversight while making personalized support operationally feasible.

The results also suggest a practical evaluation sequence. Institutions can first audit risk estimates and message evidence, then obtain blinded ratings of correctness, helpfulness, specificity, and solution disclosure before learner-facing deployment. Subsequent classroom evaluation can assess knowledge, self-regulated learning, transfer, and retention using measures aligned with each construct. This staged approach connects computational performance with educational outcomes without relying on a single metric to support every claim.

## 6. Research Scope and Future Research

The present study provides course-scale validation of the prediction–decision–generation pipeline through student-disjoint evaluation, time-valid features, paired feedback conditions, and explicit intervention budgets. Its operational outcomes and task-aware findings establish a basis for external validation across subsequent cohorts, institutions, programming languages, and assessment designs. Freezing feature definitions, calibration procedures, and decision thresholds before transfer will enable direct evaluation of model portability.

The next phase can extend this trace-based evaluation with learner-facing evidence. A prospective comparison of usual feedback, fixed evidence-based feedback, and evidence-gated progressive assistance can combine an immediate parallel-item test, a structurally different transfer task, an unassisted delayed test after two to four weeks, and a validated self-regulated-learning measure. Blinded programming-education raters can assess correctness, grounding, actionability, clarity, and solution disclosure, while subgroup calibration, accessibility, alert exposure, and learner autonomy can inform equitable implementation.

A student-level or micro-randomized design can examine repeated intervention opportunities using multilevel models. Preregistering prediction models, thresholds, prompts, outcomes, and moderation analyses will connect the computational framework to direct measures of knowledge, transfer, self-regulation, and retention. This research agenda can establish how the strategy generalizes and which combinations of timing, evidence, and assistance most effectively support learning.

## 7. Conclusions

This study developed and evaluated a framework integrating learning traces, calibrated risk prediction, constrained feedback generation, and intervention timing. Using 2993 failed programming states from 215 students, the framework predicted persistent difficulty, generated four matched forms of contextualized feedback, and evaluated alert policies under limited intervention capacity. Its central principle is that prediction, decision, generation, and learning outcomes require distinct forms of evidence.

The results showed that current performance and prior behavior provided meaningful predictive value. Logistic regression offered a strong interpretable benchmark, while deep and broader-history models contributed outcome-specific information, particularly for observable resubmission behavior. The generation pipeline produced predominantly complete feedback structures after repair, and exact source gating provided a transparent quality-control mechanism. Risk-based policies concentrated persistent-failure cases within restricted alert budgets and supported threshold selection according to instructional capacity.

Based on these findings, the study proposes evidence-gated progressive assistance: calibrated risk informs whether support is triggered, time-valid evidence constrains message content, assistance progresses from self-checks to localized guidance, and selected outputs receive focused review. This strategy provides a practical pathway from learning analytics to personalized, explainable, and timely feedback by aligning model complexity and feedback specificity with observed learner needs.

The study establishes a technical and methodological foundation for prospective educational evaluation. Future work can assess knowledge, self-regulated learning, transfer, and long-term retention through learner deployment, validated measures, blinded feedback assessment, and delayed testing. This framework positions generative AI and learning analytics as an auditable support process that targets assistance to relevant moments while preserving learner reasoning and instructor oversight.

Author Contributions: Conceptualization, S.W.; methodology, S.W.; software, S.W.; validation, S.W.; formal analysis, S.W.; investigation, S.W.; data curation, S.W.; writing—original draft preparation, S.W.; writing—review and editing, S.W.; visualization, S.W.; project administration, S.W. The author has read and agreed to the published version of the manuscript.

Funding: This research received no external funding.

Institutional Review Board Statement: Ethical review was not required for this secondary analysis of publicly available, de-identified data.

Informed Consent Statement: The source dataset documents participants’ consent for research use;   
no new participants were recruited.

Data Availability Statement: The source data are publicly available through the ProgFeed Dataset repository cited in this article (UMass ML4Ed, 2026). The analytical records and results supporting the findings are described in the article. Further inquiries can be directed to the corresponding author.

Conflicts of Interest: The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

Abdelrahman, G., Wang, Q., & Nunes, B. (2023). Knowledge Tracing: A Survey. ACM Computing Surveys, 55(11), 1–37. https:// doi.org/10.1145/3569576.

Banihashem, S. K., Noroozi, O., van Ginkel, S., Macfadyen, L. P., & Biemans, H. J. A. (2022). A Systematic Review of the Role of Learning Analytics in Enhancing Feedback Practices in Higher Education. Educational Research Review, 37, 100489. https:// doi.org/10.1016/j.edurev.2022.100489.

Bastani, H., Bastani, O., Sungu, A., Ge, H., Kabakcı, Ö., & Mariman, R. (2025). Generative AI without Guardrails Can Harm Learning: Evidence from High School Mathematics. Proceedings ofthe National Academy ofSciences, 122(26), e2422633122. https://doi.org 10.1073/pnas.2422633122.

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations Using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings ofthe 2014 conference on empirical methods in natural language processing (pp. 1724–1734). Association for Computational Linguistics. https://doi.org/10.3115/v1/D14-1179.

Du, J., Hew, K. F., & Liu, L. (2023). What Can Online Traces Tell Us about Students’ Self-Regulated Learning? A Systematic Review of Online Trace Data Analysis. Computers & Education, 201, 104828. https://doi.org/10.1016/j.compedu.2023.104828.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. In Proceedings of the 34th international conference on machine learning (Vol. 70, pp. 1321–1330). Available online: https://proceedings.mlr.press/v70/guo1 7a.html (accessed on).

Hooshyar, D., Pedaste, M., Saks, K., Leijen, Ä., Bardone, E., & Wang, M. (2020). Open Learner Models in Supporting Self-Regulated Learning in Higher Education: A Systematic Literature Review. Computers & Education, 154, 103878. https://doi.org/10.1016 j.compedu.2020.103878.

Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B., & Liu, T. (2025). A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Transactions on Information Systems, 43(2), 1–55. https://doi.org/10.1145/3703155.

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Lu, K., Dang, K., Fan, Y., Zhang, Y., Yang, A., Men, R., Huang, F., Zheng, B., Miao, Y., Quan, S., . . . Lin, J. (2024). Qwen2.5-Coder Technical Report. arXiv, arXiv:2409.12186. https://doi.org/10.48550/arXiv.2409.12186

Ifenthaler, D., & Yau, J. Y.-K. (2020). Utilising Learning Analytics to Support Study Success in Higher Education: A Systematic Review. Educational Technology Research and Development, 68(4), 1961–1990. https://doi.org/10.1007/s11423-020-09788-z.

Kapoor, S., & Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. Patterns, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804.

Kasneci, E., Sessler, K., Küchemann, S., Bannert, M., Dementieva, D., Fischer, F., Gasser, U., Groh, G., Günnemann, S., Hüllermeier, E., Krusche, S., Kutyniok, G., Michaeli, T., Nerdel, C., Pfeffer, J., Poquet, O., Sailer, M., Schmidt, A., Seidel, T., . . . Kasneci, G. (2023). ChatGPT for Good? On Opportunities and Challenges of Large Language Models for Education. Learning and Individual Differences, 103, 102274. https://doi.org/10.1016/j.lindif.2023.102274.

Khosravi, H., Buckingham Shum, S., Chen, G., Conati, C., Tsai, Y.-S., Kay, J., Knight, S., Martinez-Maldonado, R., Sadiq, S., & Gaševi´c, D. (2022). Explainable Artificial Intelligence in Education. Computers and Education: Artificial Intelligence, 3, 100074. https:/ doi.org/10.1016/j.caeai.2022.100074.

Koutcheme, C., Dainese, N., Sarsa, S., Hellas, A., Leinonen, J., Ashraf, S., & Denny, P. (2025). Evaluating Language Models for Generating and Judging Programming Feedback. In Proceedings of the 56th acm technical symposium on computer science education (pp. 624–630). https://doi.org/10.1145/3641554.3701791.

Lee, S. S., & Moore, R. L. (2024). Harnessing Generative AI (GenAI) for Automated Feedback in Higher Education: A Systematic Review. Online Learning, 28(3), 82–106. https://doi.org/10.24059/olj.v28i3.4593.

Leinonen, J., Hellas, A., Sarsa, S., Reeves, B., Denny, P., Prather, J., & Becker, B. A. (2023). Using Large Language Models to Enhance Programming Error Messages. In Proceedings of the 54th acm technical symposium on computer science education (pp. 563–569). https://doi.org/10.1145/3545945.3569770

Li, T., Nath, D., Cheng, Y., Fan, Y., Li, X., Rakovi´c, M., Khosravi, H., Swiecki, Z., Tsai, Y.-S., & Gaševi´c, D. (2025). Turning Real-Time Analytics into Adaptive Scaffolds for Self-Regulated Learning Using Generative Artificial Intelligence. In Proceedings ofthe 15th international learning analytics and knowledge conference (pp. 667–679). https://doi.org/10.1145/3706468.3706559.

Lohr, D., Keuning, H., & Kiesler, N. (2025). You’re (Not) My Type—Can LLMs Generate Feedback of Specific Types for Introductory Programming Tasks? Journal of Computer Assisted Learning, 41(1), e13107. https://doi.org/10.1111/jcal.13107.

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. In Proceedings of the 7th international conference on learning representations. https://doi.org/10.48550/arXiv.1711.05101.

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In Advances in neural information processing systems 30 (pp. 4765–4774). Available online: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd2 8b67767-Abstract.html (accessed on).

Maier, U., & Klotz, C. (2022). Personalized Feedback in Digital Learning Environments: Classification Framework and Literature Review. Computers and Education: Artificial Intelligence, 3, 100080. https://doi.org/10.1016/j.caeai.2022.100080.

Matcha, W., Uzir, N. A., Gaševi´c, D., & Pardo, A. (2020). A Systematic Review of Empirical Studies on Learning Analytics Dashboards: A Self-Regulated Learning Perspective. IEEE Transactions on Learning Technologies, 13(2), 226–245. https://doi.org/10.1109/ TLT.2019.2916802.

Meyer, J., Jansen, T., Schiller, R., Liebenow, L. W., Steinbach, M., Horbach, A., & Fleckenstein, J. (2024). Using LLMs to Bring Evidence-Based Feedback into the Classroom: AI-Generated Feedback Increases Secondary Students’ Text Revision, Motivation, and Positive Emotions. Computers and Education: Artificial Intelligence, 6, 100199. https://doi.org/10.1016/j.caeai.2023.100199.

Molenaar, I. (2022). Towards Hybrid Human–AI Learning Technologies. European Journal of Education, 57(4), 632–645. https:// doi.org/10.1111/ejed.12527.

Morris, R., Perry, T., & Wardle, L. (2021). Formative Assessment and Feedback for Learning in Higher Education: A Systematic Review. Review ofEducation, 9(3), e3292. https://doi.org/10.1002/rev3.3292.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., . . . Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in neural information processing systems 32 (pp. 8024– 8035). Available online: https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html (accessed on).

Paulsen, L., & Lindsay, E. (2024). Learning Analytics Dashboards Are Increasingly Becoming about Learning and Not Just Analytics— A Systematic Review. Education and Information Technologies, 29, 14279–14308. https://doi.org/10.1007/s10639-023-12401-4.

Price, T. W., Hovemeyer, D., Rivers, K., Gao, G., Bart, A. C., Kazerouni, A. M., Becker, B. A., Petersen, A., Gusukuma, L., Edwards, S. H., & Babcock, D. (2020). ProgSnap2: A Flexible Format for Programming Process Data. In Proceedings ofthe 2020 acm conference on innovation and technology in computer science education (pp. 356–362). https://doi.org/10.1145/3341525.3387373.

Sarsa, S., Denny, P., Hellas, A., & Leinonen, J. (2022). Automatic Generation of Programming Exercises and Code Explanations Using Large Language Models. In Proceedings of the 2022 acm conference on international computing education research (pp. 27–43). https://doi.org/10.1145/3501385.3543957.

Steiss, J., Tate, T., Graham, S., Cruz, J., Hebert, M., Wang, J., Moon, Y., Tseng, W., Warschauer, M., & Olson, C. B. (2024). Comparing the Quality of Human and ChatGPT Feedback of Students’ Writing. Learning and Instruction, 91, 101894. https://doi.org 10.1016/j.learninstruc.2024.101894.

UMass ML4Ed. (2026). ProgFeed Dataset: LLM-generated feedback in introductory programming. GitHub repository. Available online: https://github.com/umass-ml4ed/progFeed-dataset-public (accessed on). (De-identified CS110 Fall 2025 dataset, CC BY 4.0; snapshot e020c7c013187dba7b4130991eadc539d115526a)

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. In Advances in neural information processing systems 30 (pp. 5998–6008). Available online: https://proceedings.neurips.cc/ paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html (accessed on).

Wisniewski, B., Zierer, K., & Hattie, J. (2020). The Power of Feedback Revisited: A Meta-Analysis of Educational Feedback Research. Frontiers in Psychology, 10, 3087. https://doi.org/10.3389/fpsyg.2019.03087.

Xiao, R., Hou, X., & Stamper, J. (2024). Exploring How Multiple Levels of GPT-Generated Programming Hints Support or Disappoint Novices. In Extended abstracts of the chi conference on human factors in computing systems (pp. 1–10). https://doi.org/10.1145/ 3613905.3650937.

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.