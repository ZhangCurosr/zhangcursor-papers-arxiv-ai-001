# Qiushi Engine on AstaBench E2E-Bench-Hard

Wenhao Li, Shuxing Yang, Fujia Chen, Jincheng Mi,

Yuang Pan, Rui Zhao, Zichen Li, Junyao Wu,

Shenzhan Hong, Yaqi Li, Yize Wang, Kaihao Zhu,

Taowen Deng, Junjie Yang, Hongsheng Chen, Yihao Yang

Qiushi Engine Team, College of Information Science and Electronic Engineering Zhejiang University

木 Corresponding author: yangyihao@zju.edu.cn

September 2026 v2.0

This report analyzes Qiushi Engine v0.8 across all 40 test tasks in AstaBench E2E-Bench-Hard, a benchmark that requires autonomous agents to carry a research question through experimental design, code implementation, actual execution, result analysis, and report delivery. Qiushi Engine is model-configurable; this evaluation selected DeepSeek deepseek-v4pro-preview as the model backend. The oficial AstaBench leaderboard records a score of 0.816 and an average benchmark cost of \$15.209 per task, while the full-precision local recomputation is $8 1 . 5 9 \pm 1 . 8 7 $ Four tasks satisfied every rubric item, yielding a full-task completion rate of 4/40 = 10%—7 percentage points above, and about 3.3 times, the approximately 3% best rate reported for AstaBench’s oficial agents. Across 507 required rubric items, 416 were satisfied (82.1%). Oficial scoring archives and 40 Meta-Trace records show sustained production and verification of reports, code, and experimental artifacts; the principal gaps lie in repeated runs, external dependencies, specified metrics, and ablation studies. The report explains the benchmark, system workflow, aggregate results, representative cases, and limits of interpretation.

## 1 Introduction

## 1.1 The End-to-End Scientific Discovery Challenge

The question of whether LLM-based agents can perform genuine scientific research—not merely answer questions, write code, or retrieve papers, but carry out full research cycles that produce real experimental evidence—has become a central challenge in AI evaluation. Unlike narrow coding benchmarks where a single correct output sufices, end-to-end scientific discovery requires an agent to interpret a research question, design an appropriate experimental methodology, implement working code, execute experiments on real data, analyze results with statistical rigor, and produce a coherent technical report whose claims are supported by the generated code and artifacts.

This challenge is qualitatively diferent from the tasks that existing LLM benchmarks measure. HumanEval, MBPP, and DS-1000 test whether a model can produce executable code from a specification [1, 2, 3]. MMLU and GPQA test broad academic knowledge and dificult scientific question answering [4, 5]. SWE-bench and SUPER move toward realistic repository-level issue resolution, environment setup, and execution of research code [6, 7]. End-to-end scientific discovery sits at the intersection of these capabilities while adding experiment design, evidence generation, result interpretation, and coherent reporting.

The dificulty of this challenge has been confirmed empirically. Studies of frontier AI agents on research-like tasks have found that while agents can often complete individual engineering steps, the overall quality of research outputs remains far below expert standards. The compounding nature of multi-step research means that even moderate per-step failure rates produce very low end-to-end completion rates.

## 1.2 AstaBench and E2E-Bench-Hard

AstaBench [8] is a broad benchmark suite designed to measure scientific-agent capabilities across 11 benchmarks and four categories—Literature Understanding, Code & Execution, Data Analysis, and End-to-End Discovery—with over 2,400 problems in total. The benchmarks are designed to remain reproducible as science progresses, using date-limited access to the scientific literature and standardized computational environments.

Among the 11 benchmarks, E2E-Bench and E2E-Bench-Hard target End-to-End Discovery. Each task requires the agent to complete a full AI/NLP research pipeline, and the final score reflects the average proportion of task-specific rubric items jointly supported by the generated report, code, and artifacts.

Research-agent evaluation now forms a progression of increasingly complete tasks. AgentBench measures long-horizon decision making in interactive environments [9]. MLAgentBench and MLEbench focus on machine-learning experimentation, data preparation, model training, and iterative improvement [10, 11]. CORE-Bench and PaperBench evaluate computational reproducibility and paper-level research replication [12, 13]. ScienceAgentBench tests data-driven scientific tasks grounded in peer-reviewed papers [14], while the AI Scientist demonstrates an automated pipeline spanning ideation, coding, experimentation, paper writing, and simulated review [15]. AstaBench is distinctive within this progression because its unsimplified research prompts and evidence-linked rubrics directly evaluate the complete path from task definition to verifiable research deliverables.

E2E-Bench-Hard is distinguished from E2E-Bench by its task generation method. While E2E-Bench tasks undergo a simplification step to ensure tractability, E2E-Bench-Hard tasks are generated by HypER (Hypothesis-driven Exploration of Research trends), which identifies research trends from highly cited ACL papers, proposes underexplored research directions, refines them using relevant paper excerpts, and sends them through automatic ranking and expert human review—without simplification. The result is a set of 50 tasks (40 test, 10 validation) that are genuinely hard: they typically require 10–15 distinct research steps, and AstaBench reports that even when individual step completion rates are reasonable (up to ∼70%), the probability of completing all required steps in a single task remains near zero for all published agents due to compounding failures.

## 1.3 Scope and Structure of This Report

This report analyzes the performance of Qiushi Engine v0.8 on all 40 E2E-Bench-Hard test tasks, providing a complete evidence chain from the benchmark’s scoring mechanism to the observed per-sample outcomes. The analysis follows a continuous argument:

• Section 2 describes precisely how AstaBench designs and scores E2E-Bench-Hard tasks, including the task structure, rubric mechanism, and what the resulting score measures.

• Section 3 introduces the Qiushi Engine architecture relevant to understanding the observed process signatures.

• Section 4 documents the submission protocol, model identity evidence, oficial cost, and resource usage.

• Section 5 presents the aggregate results and their distributional structure.

• Section 6 is the central analytical section: it connects sample-level rubric outcomes to Qiushi’s recorded workflow, explaining which observed mechanisms accompanied high scores and which experimental or artifact gaps caused score losses, with detailed case studies across the full score range.

• Section 7 places the result in the context of the oficial AstaBench leaderboard, using the displayed oficial score and cost fields.

• Section 8 discusses the implications, limitations, and improvement opportunities.

• The appendices provide the full 40-sample evidence table, rubric inventory, archive contents, and data processing methodology.

## 2 How AstaBench E2E-Bench-Hard Scores Research

Understanding the Qiushi result requires understanding precisely what the score measures. AstaBench’s End-to-End Discovery scoring is not a binary pass/fail, nor a single-number quality rating. It is a structured rubric-item evaluation that examines whether each required facet of a valid research result is supported by the agent’s actual outputs.

## 2.1 Task Structure and Specification

Each E2E-Bench-Hard task provides the agent with a detailed research specification containing the following fields:

• Task name and short description: A concise statement of the research direction, e.g., “Dynamic Commonsense Integration” or “Anticipatory Ensemble for Medication Extraction.”

• Hypothesis to explore: The scientific hypothesis that the experiment should test, framed as a testable claim about the relationship between an independent variable (the proposed method or integration) and a dependent variable (a measurable outcome).

• Independent and dependent variables: Explicit identification of what is being varied and what is being measured.

• Comparison groups: The experimental conditions that should be compared, typically including the proposed method, relevant baselines, and ablation conditions.

• Baseline or control conditions: Specific control systems or configurations that provide reference points for the experimental comparison.

• Measurement method: The metrics, evaluation procedures, and statistical tests required to assess the hypothesis.

The specification is detailed enough to define the experimental scope but not so prescriptive as to reduce the task to code completion—the agent must still make numerous design decisions about implementation architecture, data handling, model selection and configuration, evaluation pipeline design, and reporting structure.

For example, the task “Dynamic Commonsense Integration” (idea\_5) asks the agent to integrate dynamic knowledge selection from ConceptNet and ATOMIC using a context-aware emotional graph attention mechanism, compare it against static integration baselines on a dialogue dataset, and measure F1-scores for emotion classification with statistical significance testing and multiple experimental runs. The task “Anticipatory Ensemble for Medication Extraction” (idea\_16) asks for combining anticipatory prompts with ensemble approaches for clinical medication attribute extraction, comparing against heuristic and chain-of-thought prompts, with specific requirements for GPT-3.5 and LLaMA-2 model integration, precision/recall metrics, and model-size comparison.

## 2.2 Task Generation via HypER

The tasks in E2E-Bench-Hard are generated through HypER, a pipeline that:

1. Identifies research trends by analyzing clusters of highly cited ACL papers.

2. Proposes underexplored research directions within those trends using an LLM.

3. Refines proposed ideas using relevant excerpts from the identified papers.

4. Ranks the refined ideas automatically and submits the top candidates to expert human raters.

5. Retains only those tasks that pass expert review for scientific validity and experimental feasibility.

Critically, E2E-Bench-Hard omits the simplification step that E2E-Bench applies. In E2E-Bench, after HypER generates a task, a separate simplification pass reduces the number of required components and experimental conditions to ensure the task is tractable within moderate compute budgets. E2E-Bench-Hard skips this step, preserving the full complexity of the original research specification. This makes the tasks harder not by requiring more sophisticated algorithms, but by requiring more complete experimental coverage—more comparison conditions, more metrics, more statistical analyses, and more ablation studies.

## 2.3 Rubric Design

Each task comes with a task-specific rubric: a list of criteria, each phrased as a yes/no question about whether a specific research facet was accomplished. These rubrics were manually checked and revised by human annotators to ensure that each item is: (1) necessary for a valid investigation of the stated hypothesis, (2) clearly defined enough to be evaluable from the agent’s outputs, and (3) distinguishable from other items in the rubric.

The rubric items span several categories of research activity:

• Implementation items: Did the agent implement the proposed method, baselines, integration mechanisms, and supporting modules?

• Data items: Did the agent obtain or construct the required datasets and prepare them appropriately?

• Measurement items: Did the agent compute the specified metrics (accuracy, F1, BLEU, BERTScore, etc.)?

• Statistical items: Did the agent perform statistical significance testing, multiple independent runs, or meta-analysis?

• Analysis items: Did the agent conduct ablation studies, sensitivity analyses, error analyses, or qualitative assessments?

• Model-specific items: Did the agent use the specified model architectures or backbone models?

• Reporting items: Did the agent produce documentation, prompt designs, or method descriptions?

The number of rubric items varies across tasks. In the 40 test samples analyzed here, the count of scored rubric items ranges from 8 to 17 (Q1 = 11, median = 12.5, Q3 = 14), with a mean of 12.7 items per sample. Each task also includes optional rubric items (typically 5–8 additional items) that are listed in the task target but excluded from scoring.

## 2.4 Scoring Mechanism

The scoring proceeds in three steps:

Step 1: Artifact extraction. The AstaBench E2E scorer extracts the agent’s three output artifact classes: the generated report (a technical document describing the experiment and results), the generated code (implementation files), and the generated artifacts (datasets, result files, figures, model outputs, or other research products).

Step 2: Per-item evaluation. For each rubric item, an LLM-as-judge (in this evaluation, claude-sonnet-4-6) evaluates whether the criterion is met by examining all three artifact classes together. The judge uses cross-artifact consistency: a claim in the report is not suficient if the code does not implement it or the artifacts do not contain the expected evidence. Conversely, code that implements a feature but is not reflected in the report or artifacts may also fail. The judge produces a binary verdict (1 if met, 0 if not) for each item.

Step 3: Score aggregation. The sample score is the arithmetic mean of the binary item scores. A sample with 12 scored items, of which 10 are met, receives a score of $1 0 / 1 2 = 0 . 8 3 3$ . The task-level score is the arithmetic mean of all 40 sample scores, with standard error computed across samples.

This design has an important consequence for interpretation: a score of 0.80 does not mean “80% quality.” It means that 80% of the required research facets—each representing a concrete implementation, measurement, comparison, or analysis step—were judged as supported by the combined report, code, and artifact evidence. The remaining 20% represent specific, identifiable gaps in the research output.

## 2.5 Why E2E-Bench-Hard Is Dificult

The compounding nature of end-to-end research makes these tasks qualitatively diferent from isolated coding or QA problems. A typical task requires the agent to: (1) download or construct appropriate datasets, (2) implement the proposed method and baselines, (3) run experiments with proper controls, (4) compute specified metrics, (5) perform statistical testing, (6) conduct ablation or sensitivity analyses, (7) produce error analysis or qualitative assessment, and (8) write a coherent report that accurately reflects the code and data evidence. Failure at any step can cascade: if a required model cannot be loaded, multiple downstream rubric items—its evaluation metrics, statistical comparisons, and ablation studies—may all score zero.

AstaBench’s own analysis confirms this compounding dificulty. Table 20 of the AstaBench paper reports that the overall end-to-end task completion rate (all required steps completed successfully) is near zero for all published agents: Asta Panda with Claude Sonnet 4 completes 3% of E2E-Bench Hard tasks fully, Asta CodeScientist completes 3%, and all other agents complete 0%, even when the average per-step completion rate reaches ∼70%. This stark contrast between step-level and task-level success rates underscores the compounding challenge that E2E-Bench-Hard is designed to measure.

## 3 Qiushi Engine Architecture

Qiushi Engine is an autonomous research system with a configurable LLM backend. For this evaluation, the selected backend was DeepSeek deepseek-v4pro-preview. This section describes only the public workflow elements needed to interpret the E2E-Bench-Hard evidence; it is not a complete system specification.

## 3.1 Research Functions

The public workflow can be described through four functions without exposing internal routing details:

• Plan: interpret the task, identify required datasets, models, comparisons, and measurements, and choose an experimental route.

• Build: implement data processing, methods, baselines, metrics, and report-ready artifacts.

• Run and repair: execute experiments, inspect outputs, diagnose failures, and revise code or configuration when needed.

• Verify and communicate: check consistency among claims, code, and generated artifacts, then assemble the final evidence package.

## 3.2 Meta-Trace Memory System

The workflow records each substantive step in the Meta-Trace memory system. Each recorded step contains two components:

• Reasoning record: a summary of the scientific judgment at that step—what was learned, what changed, what evidence was inspected, and what remains unresolved.

• Structured handof: the next task, current state (Process, Repair, Verify, Plan, Review, Reflect, Evaluate, or End), and research phase.

This creates a persistent, inspectable record of the research process. The Meta-Trace serves two functions: it provides long-horizon memory so that later agents can recover decisions and evidence from earlier steps, and it provides the process evidence analyzed in this report.

## 3.3 Research Phases

Work proceeds through three non-linear phases:

Explore builds understanding of the task, reads relevant sources, forms a research plan, and evaluates alternative approaches. In E2E-Bench-Hard tasks, Explore typically involves parsing the task specification, identifying which datasets and models are required, and deciding the implementation architecture.

Execute implements, runs, and verifies experiments. This is where the bulk of the work happens: writing code, running scripts, capturing outputs, diagnosing errors, repairing failures, and iterating until the experimental results are satisfactory.

Express synthesizes results into a final report and artifact bundle. This phase packages the research outputs into the JSON structure that the AstaBench scorer expects (report, code, trace, and artifacts).

Phase transitions are driven by the evidence state, not by a fixed schedule—the workflow can return to Explore after discovering that an initial approach fails, or re-enter Execute to repair a broken experiment during the Express phase. This flexibility is visible in the Meta-Trace state distributions: the 184 Repair states across 40 runs indicate that the system frequently detects and fixes problems during execution.

## 3.4 Digital Workspace

The digital workspace maintains research files—scripts, data, figures, notes, and deliverables—as inspectable assets. Search, history recovery, execution, and verification capabilities are invoked as the task requires. This artifact-centered design matters because the AstaBench scorer evaluates evidence in the report, code, and generated files rather than the plausibility of prose alone.

## 3.5 Why This Architecture Matters for E2E-Bench-Hard

The workflow is directly relevant to the scoring outcome. AstaBench evaluates three artifact classes (report, code, and generated artifacts) and checks whether they support one another. Qiushi’s build, execution, repair, and verification functions produce these evidence classes as part of the same research loop. The final checks ask whether reported claims are supported by actual code and output files, which closely matches the benchmark’s scoring target.

This does not guarantee success. The process can fail to satisfy rubric items when: a required external model is unavailable or cannot be loaded in the execution environment; the task demands more experimental runs than the step budget permits; the implementation misinterprets a technical requirement; or ablation and sensitivity analyses are deprioritized in favor of core implementation under time pressure.

## 4 Submission Protocol and Model Identity

Precise documentation of the model identity and submission path is essential for reproducibility and comparison. This section separates the three distinct model roles that appear in the submission archive.

## 4.1 Three-Role Model Identity

Table 1 Model identity separation in the Qiushi submission archive. Three model identifiers appear in the archive serving fundamentally diferent roles; conflating them would misrepresent the evaluation.
<table><tr><td>Role</td><td>Identifier</td><td>Scope and Evidence</td></tr><tr><td>Qiushi agent model</td><td>deepseek-v4pro- preview</td><td>All 40 test runs. UI label: deepseek-v4-pro (952K ctx, 48K out, premium; text). Confirmed by submitter in test_model_configuration.json. Qiushi&#x27;s model backend is configurable; DeepSeek was</td></tr><tr><td>Inspect replay label</td><td>mockllm/model</td><td>selected for this evaluation. Offline replay wrapper used to package cached outputs into Inspect format. This label appears in replay-mode run objects, not in the archive header metadata. It does not represent the model</td></tr><tr><td>Rubric judge model</td><td>claude-sonnet-4-6</td><td>that generated the original research outputs. Used by the AstaBench E2E scorer to evaluate rubric items. Part of the scoring infrastructure, not the Qiushi agent. This local evaluation used claude-sonnet-4-6 as the rubric judge.</td></tr></table>

Table 1 makes the separation explicit. The distinction matters because the Inspect archive header records the model as deepseek/deepseek-v4pro-preview (the Inspect/task metadata label) while the actual LLM that generated all research outputs was DeepSeek deepseek-v4pro-preview, as confirmed in the submission’s test\_model\_configuration.json. The configuration record also states: “Qiushi Engine is not DeepSeek-based as a product constraint; its model backend is selectable. DeepSeek was the backend selected for this evaluation.”

## 4.2 Submission Path

Qiushi was run through its Web UI, so the original agent sessions were not natively instrumented by Inspect. The scientific outputs (report, code, trace, and artifacts) were exported from each Qiushi session without content edits and scored from the command line using the oficial AstaBench E2E rubric scorer (astabench score). The 40 single-sample scorer logs were then merged into the oficial E2E-Bench-Hard test task shape.

Each sample in the submitted .eval archive contains the original scorer events plus one aggregate, non-scorer ModelEvent representing the actual Qiushi UI input/output token totals. These totals were transcribed from corresponding Qiushi Overview screenshots and represent approximate values (token counts shown with K/M units are rounded). This is an aggregate usage record, not a claim that Inspect captured each native DeepSeek API request.

## 4.3 Oficial Cost and Resource Usage

Table 2 Resource usage summary across all 40 E2E-Bench-Hard test runs.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Official leaderboard cost per task</td><td>$15.209</td></tr><tr><td>Total full LLM tokens (approx.)</td><td>1.376 billion</td></tr><tr><td>Mean tokens per sample</td><td>34.4 million</td></tr><tr><td>Total wall-clock time</td><td>386.25 hours</td></tr><tr><td>Mean duration per sample</td><td>9.66 hours</td></tr><tr><td>Mean recorded workflow steps per sample</td><td>27.65</td></tr><tr><td>Mean turns per sample</td><td>212.43</td></tr><tr><td>Mean tool calls per sample</td><td>223.25</td></tr><tr><td>Mean LLM calls per sample</td><td>505.12</td></tr><tr><td>Samples completed normally</td><td>37 (completed_verified)</td></tr><tr><td>Samples reaching step limit</td><td>3 (step_limit_reached)</td></tr></table>

Table 2 summarizes the oficial benchmark cost and recorded resource usage. Monetary cost is reported using the oficial AstaBench leaderboard value of \$15.209 per task.[16] Token counts, duration, and workflow counts are drawn from the run telemetry.

The UI mode distribution across the 40 runs was: Coding (16 samples), Discovery (12), Submission (11), and Report (1). The scientific research strategist setting was Semi Active for 23 samples and Non Active for 17.

## 4.4 Inspect Evaluation Archive Structure

The submitted .eval file is a 112 MB ZIP-compressed Inspect log containing 45 entries:

• \_journal/start.json—the Inspect session startup log recording evaluation start time and configuration parameters.

• 40 samples/idea\_\*\_epoch\_1.json—each corresponding to one E2E-Bench-Hard test sample’s complete evaluation record. Each sample JSON contains:

– Task definition (name, description, hypothesis, variables, contrasts, measurements)

– Input text (complete task specification)

– Output structure (report text, code file list, artifact file list)

– Scorer events (per-rubric-item judgments and explanation text)

– Usage statistics (token counts and event counts)

• summaries.json—score summaries for all 40 samples.

• reductions.json—score aggregation results (mean, standard error).

• header.json—evaluation metadata: task astabench/e2e\_discovery\_hard\_test, model deepseek/deepseek-v4pro-preview, and scorer score\_rubric.

The model field in header.json records the Inspect/task metadata label deepseek/deepseekv4pro-preview, corresponding to the backend configured in the Qiushi Engine Web UI. The ofline replay-mode run objects use mockllm/model; these are diferent roles.

Each sample’s scorer explanation text is the raw data source for the rubric-item analysis in this report. The scorer produces text in the format: “[MET] Dataset Implementation: The code contains a data loader...” or “[NOT MET] Statistical Testing: No evidence of multiple runs...”. Script 4 (extract\_all\_rubric\_process\_patterns.py) parses these texts via regular expressions to extract rubric item names and binary judgments.

## 4.5 UI Run Mode Analysis

The Qiushi Engine Web UI supports multiple run modes, and the submitter selected the most appropriate mode for each task. The mode distribution across the 40 runs was: Coding (16 samples, 40%), Discovery (12, 30%), Submission (11, 27.5%), Report (1, 2.5%).

Table 3 Score distribution by UI run mode. The four modes show minimal score diferences, indicating that mode selection had limited impact on final scores.
<table><tr><td>UI Mode</td><td>n</td><td>Mean</td><td>Range</td><td>Median</td><td>Description</td></tr><tr><td>Coding</td><td>16</td><td>0.810</td><td>0.54-1.00</td><td>0.83</td><td>Code-implementation-focused mode</td></tr><tr><td>Discovery</td><td>12</td><td>0.809</td><td>0.50-1.00</td><td>0.83</td><td>Scientific-discovery-focused mode</td></tr><tr><td>Submission</td><td>11</td><td>0.822</td><td>0.67-1.00</td><td>0.83</td><td>Final-submission-focused mode</td></tr><tr><td>Report</td><td>1</td><td>0.941</td><td>0.94</td><td>0.94</td><td>Report-writing-focused mode</td></tr></table>

The observed mode means are close for Coding, Discovery, and Submission; Report has only n=1. Because mode selection was not randomized and is confounded with task choice, these descriptive values do not identify an efect of UI mode on rubric coverage.

## 4.6 Evidence Sources and Supported Statements

Table 4 Evidence sources used in this report and the statements each source can support.
<table><tr><td>Source</td><td>Supports</td><td>Does not establish</td></tr><tr><td>summary_stats.json</td><td>Full-precision macro score and standard error</td><td>Consistent with official displayed score</td></tr><tr><td>40 sample JSONs</td><td>Per-item judgments and scorer explanations</td><td>Human-expert correctness of the judge</td></tr><tr><td>test_model_configu ration.json</td><td>Submitted model-configuration identifiers</td><td>Every native API request</td></tr><tr><td>UI telemetry</td><td>Approximate tokens and duration</td><td>Standardized cross-agent cost efficiency</td></tr><tr><td>Meta-Trace files</td><td>Recorded workflow states and transitions</td><td>Causal contribution of any workflow component</td></tr><tr><td>SHA-256 inventory</td><td>File identity and integrity</td><td>Scientific correctness of file contents</td></tr></table>

## 5 Aggregate Results

## 5.1 Overall Score

The locally recomputed AstaBench task score for Qiushi Engine v0.8 on E2E-Bench-Hard is:

$$
\mathrm { S c o r e } = 0 . 8 1 5 9 \pm 0 . 0 1 8 7 \quad ( \mathrm { m e a n } \pm \mathrm { s t a n d a r d } \mathrm { e r r o r } , n = 4 0 )\tag{1}
$$

This score was computed by the pinned oficial astabench score command. Its mean rounds to the 0.816 value now displayed on the oficial leaderboard. The full-precision result is equivalently expressed as $8 1 . 5 9 \pm 1 . 8 7$ on the percentage scale. The median sample score is 0.833, the minimum is 0.500, and the maximum is 1.000, with a sample standard deviation of 0.119.

Beyond the mean score, full-task completion is a stricter complementary measure. Four of the 40 tasks satisfied every required rubric item, giving Qiushi Engine a $4 / 4 0 = 1 0 \%$ perfect-completion rate. The AstaBench paper reports an approximately 3% maximum full-task completion rate for its oficial agents on E2E-Bench-Hard (see §2); against that published reference, Qiushi is 7 percentage points higher, or about 3.3 times the rate. This task-level perfect-completion statistic is distinct from the 82.1% micro-average fulfillment rate across individual rubric items.

Of the 40 samples, 37 reached completed\_verified status (the Qiushi session concluded normally after the required completion checks) and 3 reached step\_limit\_reached (the session exhausted its step budget before the completion protocol finished). Notably, one of the three step-limited samples (idea\_5) still achieved a perfect score of 1.0, demonstrating that reaching the step limit does not necessarily imply incomplete research output—it may simply mean that the verification protocol was interrupted after the research work was already complete.

## 5.2 Score Distribution and Bands

![](images/7ebb81266e56f1f3fb54c6e2be00b754986d5406172c34ff0be1aa1208da7528.jpg)  
Figure 1 Distribution of rubric-item average scores across the 40 E2E-Bench-Hard test samples. The dashed line marks the mean score (0.8159); the shaded band shows ±1 standard error. Scores range from 0.50 to 1.00, with a median of 0.833.

Figure 1 shows the score distribution. We classify the 40 samples into five performance bands:

• All items met (score = 1.0): 4 samples—idea\_5, idea\_22, idea\_29, idea\_42.

• Near-complete $( 0 . 9 0 \leq \mathrm { s c o r e } < 1 . 0 )$ : 8 samples.

• Strong (0.80 ≤ score < 0.90): 12 samples.

• Substantial but incomplete (0.70 ≤ score < 0.80): 12 samples.

• Constrained (score < 0.70): 4 samples—idea\_12, idea\_16, idea\_10, idea\_14.

Score Band Distribution (40 Samples)

![](images/43fdc21e2b987e85356a6b11b097579a544bed854fdd2f4e95439a646c8ea360.jpg)  
Figure 2 Score band distribution. The majority of samples (24/40, 60%) achieved strong or better rubric coverage (≥ 0.80).

The distribution is left-skewed: 60% of samples cluster at or above 0.80, with a small tail of constrained runs (Figure 2). Figure 3 shows the individual sample scores in sorted order, providing a complete view of the score landscape.

![](images/28a1efca954abaf4f4ce89047f394d987414b7e05f00241e73a1791aaf0f5f83.jpg)  
Figure 3 Per-sample rubric-item average scores across all 40 test tasks, sorted in ascending order. Color coding corresponds to the five score bands. The dashed line marks the overall mean.

## 5.3 Score–Resource Relationships

A natural question is whether resource expenditure covaries with score. In this 40-sample archive the Pearson correlations are close to zero: $r = 0 . 0 0 1$ for tokens and $r = - 0 . 0 1 0$ for steps. These descriptive correlations show no observed linear association; the non-random tasks and resource allocation do not support causal interpretation.

![](images/ee5063516a5b567bfba346eedbb7aea7d35b8f833f15a6f8bb069402de8f97ca.jpg)

![](images/c8be2e588d54fd57995292e18a4221e0f11f9aec4f6f94ef38f987d39451265b.jpg)

![](images/e740e6e5ed303e67f80fe70eb7845e291e5f109d99d22e94ef75980cbcd76d38.jpg)  
Figure 4 Score vs. resource expenditure across three dimensions: token budget, process length (agent steps), and wall-clock duration. Green triangles mark the five highest-scoring samples; red triangles mark the five lowest-scoring. Correlations are near zero in all cases, indicating that resource volume did not predict score within this evaluation.

The labeled samples in Figure 4 illustrate this vividly: idea\_22 scored 1.0 with only 15.2M tokens and 20 steps, while idea\_10 scored 0.667 despite consuming 103.2M tokens and 75 steps. These examples show that resource volume alone is insuficient to explain score variation in this archive.

## 5.4 Resource Usage Distributions

![](images/7f944b048f9665d4db581a368dba7f2db1cf1b9fd880178f2caa6c943695eea2.jpg)  
Figure 5 Duration distribution across the 40 samples. The right-skewed distribution contains a small number of long-running tasks. The dashed line marks the mean duration.

Figure 5 shows the duration distribution. The median duration per sample is 5.9 hours (mean 9.7 hours). Most samples complete within a shorter period, while a few samples (idea\_10 at 28.9 hours, idea\_37 at 54.1 hours, idea\_25 at 49.9 hours) take substantially longer, often due to extended repair cycles.

## 5.5 Score Variation Across Task Families

The 40 test tasks span diverse AI/NLP research areas, and their rubric coverage varies systematically by task family (Table 5). This variation is important because it reveals whether Qiushi’s performance depends on the research domain or is broadly consistent.

Table 5 Per-task-family score analysis. Families are sorted by sample count. “Zero/Total” shows zero-scored items out of total scored items across all samples in the family.
<table><tr><td>Task Family</td><td>n</td><td>Mean</td><td>Range</td><td>Zero/ Total</td><td>Mean Steps</td><td>Dominant Zero-Score Pattern</td></tr><tr><td>Reasoning / symbolic problem solving</td><td>10</td><td>0.840</td><td>0.70-1.00</td><td>19/121</td><td>29.3</td><td>Statistical support; repeated runs</td></tr><tr><td>Other NLP/ML research</td><td>8</td><td>0.826</td><td>0.73-0.93</td><td>18/107</td><td>21.0</td><td>Mixed: metrics, repeated runs</td></tr><tr><td>Classification / robustness</td><td>7</td><td>0.828</td><td>0.50-1.00</td><td>15/91</td><td>22.0</td><td>Model requirements; statistics</td></tr><tr><td>Dialogue / retrieval QA</td><td>5</td><td>0.752</td><td>0.67-0.83 16/63</td><td></td><td>49.6</td><td>Core components; metrics; statistics</td></tr><tr><td>Summarization / factual consistency</td><td>5</td><td>0.819</td><td>0.67-1.00 9/59</td><td></td><td>27.8</td><td>Ablation; core module</td></tr><tr><td>Info. extraction / clinical NLP</td><td>3</td><td>0.684</td><td>0.54-0.80 13/42</td><td></td><td>30.3</td><td>External model dependencies</td></tr><tr><td>Agents / decision-making</td><td>1</td><td>1.000</td><td>1.00</td><td>0/10</td><td>20.0</td><td>None</td></tr><tr><td>Security / privacy</td><td>1</td><td>0.929</td><td>0.93</td><td>1/14</td><td>25.0</td><td>Ablation(poisoning rates)</td></tr></table>

Several patterns emerge from Table 5:

Information extraction and clinical NLP has the lowest family mean (0.684) with only 3 samples. This family includes idea\_16 (Anticipatory Ensemble for Medication Extraction, score 0.538), which sufered the most severe cascading dependency failure in the archive. The clinical NLP tasks frequently require specific medical models (GPT-3.5, LLaMA-2) that were unavailable, making this family disproportionately afected by external dependency failures.

Dialogue and retrieval QA has the second-lowest mean (0.752) despite the highest mean step count (49.6). This family includes resource-intensive samples like idea\_1 (55 steps) and idea\_10 (75 steps), where complex multi-component systems required extensive implementation and repair efort, leaving less budget for empirical breadth. The high zero-item ratio (16/63 = 25.4%) reflects this pattern.

Reasoning and symbolic problem solving is the largest family (10 samples) and scores well (mean 0.840), with three of the four perfect-scoring samples belonging to this family. The zero-scored items here are predominantly statistical support and repeated runs—the least severe category of score loss.

Classification and robustness shows the widest score range (0.50–1.00), spanning from idea\_12 (TAPP, the lowest-scoring sample overall) to idea\_5 (Dynamic Commonsense Integration, a perfect score). This within-family variation indicates that the task-specific requirements, not the research domain alone, determine the outcome.

The overall pattern suggests that Qiushi’s performance is broadly consistent across task families, with the primary variation driven by task-specific external dependencies and complexity rather than by research domain. The one exception is information extraction/clinical NLP, where specific model requirements create a structural disadvantage.

## 6 Process Analysis: What Produced High Scores and What Caused Losses

The central analytical contribution of this report is connecting rubric-level outcomes to Qiushi’s recorded research workflow. Since each rubric item corresponds to a specific research action, the archive makes it possible to trace the evidence chain from planning, implementation, execution, and verification to the final score.

## 6.1 Rubric-Item Landscape

Across the 40 test samples, the AstaBench scorer evaluated 507 rubric items in total. Of these, 416 were scored as met and 91 as zero (micro-average fulfillment 416/507 = 82.1%, which is close to but distinct from the oficial macro-average task score of 81.59%). The rubric items span a wide range of research actions: dataset loading and preparation, method implementation, baseline construction, metric computation, statistical testing, ablation studies, error analysis, visualization, documentation, and more.

Rubric Item Outcomes per Sample (Sorted by Score)  
![](images/d6cc5fb8ba8c05f3f0ed91ee75c6781c95944f0fb2aad6b687f82a5bfcdf9674.jpg)  
Figure 6 Stacked bar chart showing the number of met (green) and zero-scored (red) rubric items per sample, sorted by score. Samples on the right have all items met; samples on the left have the most zero-scored items. The total bar height reflects the number of scored rubric items, which varies by task.

Figure 6 shows the rubric item outcomes per sample. The variation in total bar height reflects the diferent number of scored rubric items per task (range: 8–17). The pattern is clear: higher-scoring samples have fewer zero-scored items, but even some of the lower-scoring samples (e.g., idea\_10 with 10/15 items met) satisfied the majority of rubric items, losing points on specific identifiable gaps rather than failing broadly.

## 6.2 Recurring Zero-Scored Dimensions

Figure 7 shows the distribution of zero-scored items by category. The pattern is not uniform—certain types of research actions were systematically more likely to be missed:

Statistical support and repeated runs (16 samples). Many tasks require multiple independent experimental runs with statistical significance testing (e.g., paired t-tests, bootstrap resampling, or computation of confidence intervals). These items demand that the agent not only implement the experiment once but run it multiple times with diferent seeds or conditions and then compute inter-run statistics. When the step budget is limited or the primary implementation consumes most of the available compute, repeated runs and their statistical analysis are often the first items to be sacrificed.

Recurring Zero-Scored Rubric Dimensions Across 40 Test Samples  
![](images/312dcd2540527eb1447a1d4396a03fc9aa1166f0f262b79a55fc38af0d2baa82.jpg)  
Figure 7 Recurring categories of zero-scored rubric dimensions across 40 samples. “Statistical support / repeated runs” was the most commonly missed category (16 samples), followed by “core method component” requirements (11 samples) and “required metrics not completed” (10 samples).

Core method components requiring external resources (11 samples). Some rubric items require implementing specific components that depend on external models, datasets, or APIs. For example, a “Neural Selector” baseline in a retrieval task, or a “Knowledge Graph Integration” module in a verification framework. When the required resource is unavailable in the execution environment or the component’s implementation is too complex to complete within the step budget, the item scores zero.

Explicit metric completion (10 samples). Certain rubric items require specific metrics (e.g., response accuracy on a particular split, F1 with a specific tokenization, or a named evaluation like BERTScore) that were either not computed, produced values the scorer could not confirm in the output artifacts, or were computed but not properly reported in the final deliverables.

Ablation and sensitivity analyses (8 samples). Ablation studies require multiple experimental conditions beyond the core comparison—removing or varying individual components to isolate their contributions. Like repeated runs, they are typically scheduled late in the research process and may be truncated by step or time limits.

Exact model/backbone requirements (3 samples). Some tasks specify particular model architectures (e.g., GPT-3.5, LLaMA-2, GPT-4) that must be used as components of the experiment. When these models cannot be accessed through the available API or execution environment, all downstream items dependent on those models score zero.

Other categories include human or qualitative evaluation (2 samples), generalization or crossdomain testing (2), and computational eficiency analysis (1).

## 6.3 Strength Pattern: Core Implementation and Artifact Production

Qiushi consistently scored well on rubric items involving: dataset acquisition and preparation, core method implementation, baseline system construction, integration mechanisms, primary metric computation, and report/code/artifact bundle production. These are the backbone of each research task—the items that require designing, coding, running, and packaging a working experiment.

This pattern is consistent with an artifact-centered workflow in which method construction, execution, repair, and final consistency checking are connected. The Express phase packages the report, code, trace, and artifacts into the deliverable structure expected by the AstaBench scorer. This workflow maps naturally onto the rubric’s requirement that claims be supported across report, code, and artifact evidence; the association is descriptive and does not isolate a causal architectural efect.

## 6.4 Process Characteristics Across Score Bands

The Meta-Trace state summaries show that the constrained and substantial-but-incomplete score bands contain more Repair and Verify activity on average. This count pattern is consistent with more frequent execution dificulties in the inspected cases, but the counts alone do not identify the cause of score loss.

Interestingly, the all-items-met band does not show the lowest resource usage. Some perfect-scoring samples (notably idea\_42 with 79 steps) required extensive repair cycles to achieve complete rubric coverage, while others (idea\_22 with 20 steps) completed eficiently. This further supports the finding that score is determined by the completeness and correctness of research actions, not by resource volume.

## 6.5 Case Studies: Perfect-Scoring Samples

Table 6 Four perfect-scoring samples with rubric details and process characteristics. All scored rubric items were met in each case.
<table><tr><td>ID</td><td>Task Name</td><td>Score</td><td>Met</td><td>Tot</td><td>Steps</td><td>Tok (M)</td><td>Key Process Feature</td></tr><tr><td>idea_5</td><td>Dynamic Commonsense Integration</td><td>1.00</td><td>12</td><td>12</td><td>40</td><td>56.5</td><td>CUDA determinism fix; paired t-tests; seed sensitivity analysis</td></tr><tr><td>idea_22</td><td>Memory-Aug. Decentral. Decision</td><td>1.00</td><td>10</td><td>10</td><td>20</td><td>15.2</td><td>SHA-256 file verification; causally distinct conditions</td></tr><tr><td>idea_29</td><td>Memory-Enhanced Tree Reasoning</td><td>1.00</td><td>11</td><td>11</td><td>20</td><td>24.1</td><td>Final-deliverable re-verification; artifact inspection</td></tr><tr><td>idea 42</td><td>FactCC-CaPE Integration</td><td>1.00</td><td>16</td><td>16</td><td>79</td><td>105.4</td><td>Dual-dataset eval; matched-alpha controls; 79-step repair cycle</td></tr></table>

The perfect-scoring samples (Table 6) share several process characteristics visible in their Meta-Trace records:

idea\_5 (Dynamic Commonsense Integration, score 1.0, 40 steps). This task required integrating ConceptNet and ATOMIC knowledge sources with a context-aware emotional graph attention mechanism. The final Meta-Trace records that all 26 experimental checks were passed, including a determinism fix for CUDA non-reproducibility, seed-sensitivity analysis across multiple runs, and paired t-tests. The agent identified that the “Dynamic” model showed extreme seed sensitivity (F1 ranging from 0.000 to 0.066 across seeds), reported this as a genuine phenomenon rather than an experimental artifact, and included the NoKnowledge baseline comparison. Despite reaching the step limit (40 steps), all scored rubric items were satisfied.

idea\_22 (Memory-Augmented Decentralized Decision-Making, score 1.0, 20 steps). This task asked the agent to integrate memory-augmented planning with GPT-based decision engines in the Overcooked-AI environment. The Preceptor’s final verification confirmed SHA-256 hashes of all 10 code files and verified that the three experimental conditions (GPT-only, memory-only rule-based, and memory-augmented GPT) were causally distinct in the runner code. The eficient 20-step completion suggests a well-matched task complexity and execution capability.

idea\_42 (FactCC-CaPE Integration, score 1.0, 79 steps). This task recorded 79 steps, the highest step count in the archive. The agent needed to implement FactCC-based factual consistency scoring, CaPE fine-tuning, expert/anti-expert model training, and evaluate on both XSUM and CNN/DM datasets. The matched-alpha control experiment (ensuring that the integrated model’s advantage was not simply due to a better alpha parameter) added complexity. The Meta-Trace shows 25 Repair states—the highest in the archive—indicating extensive debugging and iteration before the full rubric was satisfied.

## 6.6 Case Studies: Medium-Scoring Samples

The medium-scoring samples (0.73–0.87) are particularly informative because they reveal partial success patterns—tasks where the core research was substantially completed but specific dimensions fell short.

idea\_1 (Hierarchical Hybrid QA Integration, score 0.833, 10/12 items met, 55 steps). This task required implementing a hierarchical table processing module for multi-hop QA over the MultiHiertt dataset. The two zero-scored items were “Hybrid Evidence Utilization Algorithm” and “Multiple Experimental Runs.” The agent successfully implemented the dataset, processing modules, baselines, metrics, and statistical comparison, but the specific hybrid evidence algorithm and repeated independent runs were not completed. With 55 steps and 81.1M tokens, this was a resource-intensive run, suggesting the task’s complexity exceeded what could be fully covered within the budget.

idea\_37 (score 0.727, 8/11 items met, 65 steps, 84.9M tokens, 54.1 hours). The zero-scored items were “F1 Score Calculation,” “Reframing Technique Documentation,” and “Coherence Score Thresholds.” Its observable Meta-Trace contains 48 Process, 5 Repair, 4 End, 4 Evaluate, 3 Plan, and 1 Review states, showing that high resource use did not substitute for satisfying the three specific requirements.

idea\_25 (Integrated Model Editing for Consistency, score 0.867, 13/15 items met, 39 steps, 49.9 hours). The second-longest run by duration. Missed items were “Multiple Independent Runs” and “Meta-Analysis”—both statistical support requirements. The core MEMIT implementation, RippleEdits benchmark integration, and all measurement items were met.

idea\_7 (Cross-Document Structured Extraction, score 0.800, 12/15 items met, 45 steps). The three missed items were dataset acquisition, statistical significance testing, and cross-validation. This illustrates a cascade pattern: if a specific dataset cannot be acquired, downstream items dependent on that data (statistical testing, cross-validation) also fail.

idea\_50 (Augmented Graph-Based Schema Encoding, score 0.833, 10/12 items met, 60 steps, step\_limit\_reached). One of the three step-limited samples. Missed items were the SGD-X dataset implementation and repeated runs. Despite reaching the step limit, 83% of rubric items were satisfied, indicating substantial research progress before termination.

The near-complete samples (≥ 0.90) are instructive because they show where the system came within one or two rubric items of a perfect score and what prevented full coverage.

idea\_20 (Syntactic Trigger Inspection, score 0.941, 16/17 items met, 17 steps, 24.0M tokens). This task investigated backdoor attack techniques using syntactic triggers and model inspection. The single zero-scored item was “Varying Poisoning Rates”—an experimental condition the agent did not implement. The Preceptor’s final reasoning noted that the experiment produced a legitimate negative result: the backdoor technique was too weak to demonstrate the hypothesized efect, and the report stated this honestly rather than overclaiming. This case illustrates that near-perfect scores can coexist with scientifically honest negative findings. The Meta-Trace shows only 6 Process and 3 Repair states, indicating smooth implementation.

idea\_28 (Contrastive Bayesian Hybrid Learning, score 0.933, 14/15 items met, 14 steps, 10.5M tokens). This active learning task scored well across implementation, baselines, metrics, and statisti cal testing. The single zero-scored item was “Low-Resource Dataset Implementation.” Completed in only 14 steps and 1.1 hours, this was one of the most eficient runs in the archive. The parsed state counts for this row are unavailable in Appendix D, so the evidence here is the score/resource record rather than a state-pattern reconstruction.

idea\_26 (Integrated Contrastive Learning for Factual Extraction, score 0.929, 13/14 items met, 20 steps). The single missed item was “Multiple Experimental Runs.” Core implementation, baselines, all metrics, and statistical significance testing were completed. This case typifies the “only missing repeated runs” pattern—the most common single-item failure across the 40 samples.

idea\_32 (Dynamic Streaming Evaluation, score 0.923, 12/13 items met) was especially short, with 10 steps, 8.6M tokens, and 1.4 hours. idea\_33 (Adversarial Contextual Embeddings, score 0.923, 12/13 items met) also used few steps—13 steps and 11.9M tokens—but had a longer wall-clock duration of 24.3 hours. For idea\_33, the single missed item was “Coded Hate Speech Detection Analysis”; for idea\_32, it was “Computational Eficiency Measurement.” These samples show that short, high-scoring runs can still miss a task-specific analysis or eficiency metric.

idea\_41 (FastText Membership Inference, score 0.929, 13/14 items met, 25 steps). The single missed item was “Multiple Dataset Validation.” Core FastText embedding extraction, membership inference attack implementation, privacy metrics, and statistical analysis were all complete. The pattern is consistent: when one specific experimental extension (additional dataset, repeated runs, or ablation condition) is not completed, the resulting score loss is small (one item out of 14) but prevents a perfect score.

The near-complete samples share a common profile: core implementation is complete, primary metrics are computed, basic significance analysis is performed, and the only gap is one specific experimental condition, dataset variant, or repeated-run requirement. Within these samples, core implementation items had high observed coverage; remaining gaps often concerned an additional condition, dataset, or repeated run.

Table 7 The four constrained samples (score < 0.70) with zero-scored dimensions and process details.
<table><tr><td>ID</td><td>Task Name</td><td>Score</td><td>Met/ Tot</td><td>Steps</td><td>Tok (M)</td><td>Zero-Scored Items</td></tr><tr><td>idea_12</td><td>TAPP with Explicit Demos</td><td>0.50</td><td>6/12</td><td>13</td><td>10.2</td><td>LLM Selection; Accuracy; F1; Statistical Testing; Replication; Demo Selection Strategy</td></tr><tr><td>idea_16</td><td>Anticipatory Ensemble Med. Extract.</td><td>0.54</td><td>7/13</td><td>38</td><td>52.8</td><td>GPT-3.5 Integration; LLaMA-2 Integration; Repeated Runs; Meta-Analysis; Stat. Testing; Model Size Comparison</td></tr><tr><td>idea_10</td><td>Contextual Feedback Integration</td><td>0.67</td><td>10/15</td><td>75</td><td>103.2</td><td>Response Accuracy; Repeated Runs; Ablation; Trigger Conditions;</td></tr><tr><td>idea__14</td><td>Adaptive FactCC Decoding</td><td>0.67 6/9</td><td></td><td>15</td><td>13.2</td><td>Qualitative Analysis FactCC Eval Module; Ablation; Abstractiveness</td></tr></table>

## 6.7 Case Studies: Constrained Samples

The constrained samples (Table 7) reveal distinct failure modes:

idea\_12 (TAPP with Explicit Demonstrations, score 0.500, 6/12 items met). This task asked the agent to enhance LLM text classification using TAPP with explicit answer demonstrations. The agent implemented TAPP, baseline configurations, both datasets, cross-task demonstrations, and prompt format documentation. However, it did not receive credit for LLM model selection (the task required a specific large model), accuracy and F1 measurement, statistical testing, experimental replication, or demonstration selection strategy. With only 13 steps and 10.2M tokens, this was one of the shortest runs, suggesting that the session may have concluded prematurely—possibly interpreting the task too narrowly or failing to recognize that measurement and statistical requirements were not yet satisfied.

idea\_16 (Anticipatory Ensemble for Medication Extraction, score 0.538, 7/13 items met). This is a classic cascading-dependency failure. The task required integrating both GPT-3.5 and LLaMA-2 as model backends for medication extraction. When these specific models could not be accessed in the execution environment, six downstream rubric items—model integration, repeated runs across models, meta-analysis, statistical testing, and model-size comparison—all scored zero. The core prompting framework, ensemble voting, and precision/recall computation were implemented and scored well, demonstrating that the research design was sound but the external dependency chain was broken.

idea\_10 (Contextual Feedback Integration, score 0.667, 10/15 items met, 75 steps, 103.2M tokens). This was the second-highest resource consumer in the archive. The Meta-Trace shows 22 Repair states—extensive debugging efort. The agent successfully built all three system configurations (baseline, rewrite-only, and integrated), implemented retrieval metrics, BERTScore, and statistical testing. But it did not complete response accuracy metrics, repeated independent runs, ablation study, query-rewriting trigger conditions, or qualitative analysis. The high Repair count suggests that the implementation was dificult (possibly due to environment issues or complex dependencies), and the repair efort consumed the budget before empirical breadth requirements could be addressed.

idea\_14 (Adaptive FactCC Decoding, score 0.667, 6/9 items met). The Meta-Trace reveals an interesting semantic mismatch. The task text loosely described FactCC as extracting “fact triples,” but the actual FactCC tool is a weakly supervised document-sentence factual consistency classifier built on BERT. The agent identified this discrepancy during its Explore phase and correctly implemented FactCC-like functionality, but the scorer judged the “FactCC Evaluation Module Implementation” rubric item as not met, likely because the implementation diverged from the task text’s literal description. The other missed items (ablation study and abstractiveness measurement) reflect the typical empirical-breadth shortfall.

## 6.8 The Three Step-Limited Samples

Three of the 40 samples reached step\_limit\_reached status: idea\_5 (40 steps, score 1.0), idea\_- 50 (60 steps, score 0.833), and idea\_10 (75 steps, score 0.667). These three form a descriptive comparison showing how step limits interact with final scores.

idea\_5 (Dynamic Commonsense Integration, score 1.0, 40 steps) reached the step limit but achieved a perfect score. The Meta-Trace shows the system was still in its End-state verification protocol at step 40—the research work itself was already complete, and the step limit was triggered only because the multi-perspective verification protocol had not completed all rounds. The observed states are 16 Process, 11 Repair, 7 End, 2 Evaluate, 2 Review, 1 Plan, and 1 Verify. Its perfect score shows that all 12 scored items were judged met despite the status label.

idea\_50 (Augmented Graph-Based Schema Encoding, score 0.833, 60 steps) had two unmet items (SGD-X Dataset and Repeated Runs) when the step limit was reached. The observed states are 18 Process, 15 Plan, 10 Repair, 8 Reflect, 6 End, 2 Verify, and 1 Review. The two unmet items were SGD-X dataset implementation and multiple seeds; the counts do not by themselves establish why they remained unmet.

idea\_10 (Contextual Feedback Integration, score 0.667, 75 steps) is the case where the step limit coincided with a low score. With 75 steps, 22 Repair states, and 16 End states, this sample exhibited persistent implementation dificulties—dependency conflicts, import errors, and data format issues arose repeatedly. Unlike idea\_5, the core implementation itself was problematic, and additional steps without a strategy change may not have addressed the recurring problems; the counterfactual efect of more steps is unknown.

The descriptive comparison shows that step\_limit\_reached is not synonymous with a low score: idea\_5 scored 1.0, idea\_50 scored 0.833, and idea\_10 scored 0.667. Because the three tasks difer, the archive cannot isolate a causal efect of the step limit.

## 6.9 The Verification–Breadth Tradeof

A distinctive finding from the Meta-Trace analysis is that higher repair efort is weakly positively correlated with score (r = 0.24 for repair fraction vs. score), rather than negatively correlated as one might expect if repair consumed resources that could otherwise be used for broader experimental coverage.

Table 8 reveals two distinct patterns among high-repair samples:

High-repair complete runs: idea\_42 (25 Repair states, score 1.0) and idea\_5 (11 Repair states, score 1.0) show that high Repair counts are compatible with complete rubric coverage. In idea\_- 42, the Meta-Trace records repairs addressing training failures, dataset format mismatches, and parameter configuration issues for the dual-dataset evaluation. The archive supports an observational reading: the run spent many steps resolving implementation and configuration issues before all 16 rubric items were judged satisfied.

Table 8 Samples with the highest Repair state counts, showing the relationship between repair efort and final score. High repair does not predict low scores—idea\_42 (25 repairs) and idea\_5 (11 repairs) both scored 1.0.
<table><tr><td>ID</td><td>Task Name</td><td>Score</td><td>Repair</td><td>Verify</td><td>End</td><td>Steps</td><td>Outcome</td></tr><tr><td>idea_42</td><td>FactCC-CaPE Integration</td><td>1.00</td><td>25</td><td>1</td><td>5</td><td>79</td><td>All 16 items met despite heavy repair</td></tr><tr><td>idea_10</td><td>Contextual Feedback</td><td>0.67</td><td>22</td><td>0</td><td>16</td><td>75</td><td>5 items missed; budget consumed by repair</td></tr><tr><td>idea_9</td><td>Hybrid Bottleneck- Selector</td><td>0.73</td><td>14</td><td>5</td><td>5</td><td>43</td><td>4 items missed; complex dependencies</td></tr><tr><td>idea_5</td><td>Dynamic Commonsense</td><td>1.00</td><td>11</td><td>1</td><td>7</td><td>40</td><td>All 12 items met; CUDA fix included</td></tr><tr><td>idea_50</td><td>Augmented Graph Schema</td><td>0.83</td><td>10</td><td>2</td><td>6</td><td>60</td><td>2 items missed; step limit reached</td></tr><tr><td>idea_1</td><td>Hierarchical Hybrid QA</td><td>0.83</td><td>8</td><td>6</td><td>4</td><td>55</td><td>2 items missed; high verify effort</td></tr><tr><td>idea_19</td><td>Cooperative Ontological NER</td><td>0.91</td><td>8</td><td>4</td><td>4</td><td>15</td><td>1 item missed; efficient despite repairs</td></tr></table>

High-repair incomplete run: idea\_10 (22 Repair states, score 0.667) represents a contrasting outcome. Here, the repairs were concentrated on making the core system functional (handling dependency conflicts, fixing import errors, resolving data format issues), while empirical breadth requirements remained unmet in the final scored artifacts. The 16 End states (unusually high) show repeated attempts to conclude followed by further unresolved issues.

The quantitative picture is nuanced: repair fraction correlates positively with score (r = 0.24), but repair count does not correlate with the number of zero-scored items (r = −0.06). Repair states occur in both complete and incomplete runs; these observed correlations do not isolate whether repair improved scores or merely co-occurred with task and process diferences.

The archive therefore supports a narrower operational lesson: Repair count alone is not a failure indicator. More targeted distinction between environmental and implementation problems may reduce unproductive cycles, but its efect requires controlled evaluation.

## 6.10 Workflow State Signatures

The aggregate Meta-Trace state distribution across the 40 runs reveals several notable patterns:

Repair as a substantial fraction of work: 184 Repair states constitute 18% of all states, indicating that the system frequently detected and fixed problems during execution. This is a natural consequence of working with real code, real data, and real APIs—runtime errors, dependency conflicts, and unexpected data formats require ongoing repair.

Repeated final checks: 188 End states across 40 runs (mean 4.7 per run) record repeated attempts to bring a run to a verified close. In 32 of 40 runs, the last parsed Meta-Trace state was End. These are system-generated process records and should not be interpreted as independent quality judgments.

Balanced Plan/Verify/Review/Reflect: 79 Plan, 82 Verify, 52 Review, and 36 Reflect states indicate that the system regularly pauses to plan next steps, verify intermediate results, review accumulated evidence, and reconsider its approach. These metacognitive states complement the forward-working Process state and the reactive Repair state.

## 6.11 Task Diversity

![](images/bf4be4165e3dbdb1d55851712e4a9577c9eb26c3dd97e5881f7ed049e2f93a60.jpg)  
Figure 8 Primary task family distribution across the 40 E2E-Bench-Hard test samples. The tasks span a broad range of AI/NLP research areas, with no single family dominating the test set.

The primary-family assignment contains reasoning/symbolic problem solving (10), other NLP/ML research (8), classification/robustness (7), dialogue/retrieval QA (5), summarization/factual consistency (5), information extraction/clinical NLP (3), agents/decision-making (1), and security/privacy (1). These categories are analytical groupings of the 40 tasks rather than benchmark-defined strata.

## 7 Comparison with the Oficial AstaBench Leaderboard

The comparison below uses the oficial leaderboard snapshot accessed on 4 September 2026.[16] Figure 9 shows the score–cost distribution on a linear cost axis. It highlights Qiushi Engine and Ai2’s customized ReAct configurations using Claude Opus 4.7, Claude Opus 4.6, and GPT-5.5.

## 7.1 Score Comparison

The oficial leaderboard records Qiushi Engine at 81.6%, compared with 65.5% for the next-highest displayed entry, Ai2’s customized ReAct configuration using Claude Opus 4.7 (Table 9). The displayed diference is 16.1 percentage points. This is a leaderboard-level comparison of complete system configurations; it does not isolate the contributions of the orchestration system, model backend, tools, or execution environment.

For context, the Faker baseline—which simply prompts an LLM to fabricate research outputs without actually doing the work—scores 25.4 ± 4.5. This means that roughly a quarter of rubric items can be satisfied by plausible-sounding but fabricated text. The Faker comparison is an informative reference for the scoring scale, but the 56.2-point diference combines changes in model, system, tools, and execution; it is not an isolated estimate of an execution efect.

![](images/3dc5fffab0800a8a646b2cc4df1dac994e3af99f6c204eedb33a3c42116503e0.jpg)  
Figure 9 AstaBench E2E-Bench-Hard score versus average cost per task, using the 29 results in the report’s HTML snapshot of 4 September 2026. Labels retain the snapshot’s two-decimal percentages; the table uses rounded leaderboard display values. All labeled Ai2 ReAct entries use Ai2’s enhanced implementation; the dashed curve marks the Pareto frontier.

Table 9 Oficial AstaBench E2E-Bench-Hard leaderboard snapshot including Qiushi Engine. Scores and costs are the values displayed by the leaderboard; score percentages are obtained by multiplying the displayed decimal by 100.
<table><tr><td>Agent</td><td>Model used</td><td>Score</td><td>Cost/task</td><td></td></tr><tr><td>Qiushi Engine</td><td>deepseek-v4pro-preview</td><td>81.6%</td><td>$15.209</td><td rowspan="4"></td></tr><tr><td>ReAct (Ai2 customized)</td><td>claude-opus-4-7</td><td>65.5%</td><td>$11.452</td></tr><tr><td>ReAct (Ai2</td><td>claude-sonnet-4-6</td><td>62.2%</td><td>$5.557</td></tr><tr><td>customized) ReAct (Ai2</td><td>claude-opus-4-6</td><td>60.9%</td><td>$6.053</td></tr><tr><td>customized) Asta v0</td><td>Claude Sonnet 4 (2025-05) (+6)</td><td>56.8%</td><td>$14.487</td><td rowspan="4"></td></tr><tr><td>Asta Panda</td><td>Claude Sonnet 4 (2025-05)</td><td>56.5%</td><td>$14.487</td></tr><tr><td>Asta CodeScientist</td><td>Claude 3.7 Sonnet (2025-02)</td><td>55.8%</td><td>$3.549</td></tr><tr><td>ReAct (Ai2</td><td>GPT-5.5 (2026-04-23)</td><td>38.8%</td><td>$1.971</td></tr><tr><td>customized) ReAct (Ai2 customized)</td><td>GPT-5 (2025-08)</td><td>38.2%</td><td>$0.584</td><td></td></tr></table>

The leaderboard also shows that results vary substantially across model and agent configurations. The ReAct entries shown here are Ai2-customized implementations rather than an unmodified textbook ReAct baseline, so the table names them explicitly to avoid implying a purely generic implementation.

## 7.2 Comparison Caveats

Several caveats apply to this comparison:

Execution environment diferences. Qiushi’s submission category is “Closed source & UI only” with fully custom tools, while the AstaBench agents use the standard Asta Environment with its specific code execution sandbox and tool suite. The environments therefore difer in available packages, compute resources, network access, and API availability.

Oficial cost. The reported Qiushi Engine cost is the oficial leaderboard’s standardized average of \$15.209 per task, as used in Table 9 and Figure 9.

Scorer model. All evaluations (both published agents and Qiushi) use the same LLM-as-judge scorer model (claude-sonnet-4-6), ensuring that the rubric evaluation standard is consistent.

## 8 Discussion

## 8.1 What the Score Means

Let $m _ { i }$ and $n _ { i }$ denote met and scored items for sample i. The oficial result is the macro-average

$$
\bar { s } = \frac { 1 } { 4 0 } \sum _ { i = 1 } ^ { 4 0 } \frac { m _ { i } } { n _ { i } } = 0 . 8 1 5 9 4 0 .
$$

Pooling all items instead gives the distinct micro-average $\textstyle { \sum _ { i } m _ { i } } / { \sum _ { i } n _ { i } } = { 4 1 6 } / { 5 0 7 } = 0 . 8 2 0 5 1 3$ . The macro-average weights each task equally; the micro-average weights tasks with more rubric items more heavily. The rubric items are not subjective quality judgments—they are specific, binary checks on whether concrete research actions (implementing a dataset loader, running statistical tests, producing an ablation study) are supported by the combined report, code, and artifact evidence.

It is important to note what the score does not measure. It does not assess the scientific novelty of the research findings. It does not evaluate whether the conclusions are interesting or surprising. It does not measure the quality of writing or the depth of scientific interpretation. What it measures is research-pipeline completeness: did the agent actually do the work specified in the task, and can that work be verified from the produced artifacts?

Satisfying 82% of such items across diverse, unsimplified research tasks represents substantial coverage of the specified research pipeline. The remaining 18% of missed items concentrate in specific categories (statistical support, external dependencies, and ablation analyses), providing a concrete view of current limitations.

## 8.2 The Role of Verification and Repair

A distinctive feature of the Qiushi process, visible in the Meta-Trace records, is the emphasis on verification and repair. The 184 Repair states and repeated End checks are aligned with the scoring demand for cross-artifact evidence.

Many rubric items require not just that code exists, but that it runs correctly and produces outputs consistent with the report. The recorded verification runs and cross-artifact checks address this requirement. In the perfect-scoring samples, the final Meta-Trace entries describe specific actions: SHA-256 hash checking of code files (idea\_22), determinism testing (idea\_5), matched-alpha control verification (idea\_42), and independent result reconstruction (idea\_20).

The verification overhead is not free—it consumes agent steps that could otherwise be used for additional experiments or analyses. This creates a resource allocation tradeof: more verification increases the reliability of existing results but may leave less budget for expanding experimental coverage (e.g., repeated runs or ablation studies). The zero-scored dimension analysis suggests that this tradeof is currently resolved in favor of verification over breadth, which may explain why the most commonly missed items are empirical breadth requirements rather than core implementation items.

## 8.3 Score-Loss Mechanisms

The analysis identifies four primary score-loss mechanisms:

Resource-allocation pressure. Statistical support, repeated runs, and ablation studies are typically the last items attempted in a research pipeline. When the core implementation and primary metrics consume most of the step budget, these breadth items are truncated. This mechanism explains the most common zero-scored category (16 samples missing statistical support).

External dependency failures. When a task requires specific models (GPT-3.5, LLaMA-2) or specific datasets that are unavailable in the execution environment, a cascade of downstream items—all dependent on the missing resource—score zero simultaneously. This mechanism produces the largest per-sample score drops (e.g., idea\_16 losing 6 items from two missing models).

Technical misinterpretation. In rare cases, the agent’s interpretation of a technical requirement difers from what the scorer expects. The idea\_14 case (FactCC interpreted as a document-sentence classifier rather than a fact-triple extractor) illustrates this mechanism.

Premature conclusion. Some runs (idea\_12 with only 13 steps) appear to conclude before the full rubric is addressed, possibly due to the agent’s assessment that the task is complete when it is not. This mechanism is distinct from resource exhaustion—the budget was available but not used.

Each score-loss mechanism’s relative impact can be quantified from the rubric item data. Resourceallocation pressure afected 16 of 40 samples (40%) but typically caused only 1–2 rubric items to fail per sample, with a moderate impact on the total score (approximately 0.06–0.12 points per sample). External dependency failures afected fewer samples but more severely—idea\_16 lost 6 rubric items from two unavailable models, dropping from a potential ∼1.0 to 0.538. Technical misinterpretation afected only individual samples (idea\_14) but is dificult to prevent systematically. Premature conclusion (idea\_12, only 13 steps) afected the fewest samples but produced the largest per-sample loss. The run ended with six unmet items; the archive cannot determine whether additional steps would have improved the score.

## 8.4 Relationship to Prior Research Agent Evaluations

This evaluation’s results can be understood in the broader context of AI scientific research agent assessment. The AI Scientist demonstrates the technical feasibility of automating a complete research loop [15], while results from ScienceAgentBench, CORE-Bench, and PaperBench show that substantial gaps remain on realistic scientific coding, computational reproduction, and paper-level replication tasks [14, 12, 13]. AstaBench’s E2E-Bench-Hard provides a more granular measurement through rubric-item scoring—rather than simply judging a paper as good or bad, it checks item by item whether specific research actions were completed.

At this granularity, Qiushi Engine’s 82% rubric-item pass rate means that across 40 diverse research tasks, the vast majority of required concrete research actions (implementing methods, constructing baselines, computing metrics, producing artifacts) were successfully executed. The concentration of failures in specific categories (statistical support, external dependencies, ablation) rather than uniform distribution indicates that the system’s capability boundary is sharp rather than difuse.

The Faker baseline (25.4) provides an important reference: approximately one quarter of rubric items can be satisfied by plausible-sounding but fabricated text. The 56.2 percentage-point gap between Qiushi (81.6) and Faker (25.4) marks the scale diference between fabricated outputs and a run that produced code, data processing, experiment execution, and artifacts, while still combining model, system, tool, and execution diferences.

## 8.5 System-Level Insights

This evaluation provides several workflow-design insights:

Value of separating research functions. Keeping planning, method construction, experiment execution, and final checking as explicit functions is consistent with E2E-Bench-Hard’s demand for both working code and cross-artifact evidence. The observed archive supports the presence of this separation, but the evaluation does not isolate its causal contribution to the score.

Repair as a research capability. The 184 Repair states should not be viewed as a system deficiency. In real research, runtime errors, dependency conflicts, data format mismatches, and unexpected behavior are normal engineering realities. The system’s ability to detect these problems and fix them automatically—rather than ignoring or hiding them—is an observable feature of these runs. The weak positive correlation between repair fraction and score (r = 0.24) is descriptive; it does not establish that repair caused higher scores.

Meta-Trace as process evidence. The Meta-Trace memory system serves as runtime longhorizon memory and provides the process evidence analyzed in this report. Its reasoning summaries and structured handofs create an inspectable process–outcome record. This inspectability has independent value in AI system evaluation: it allows evaluators to understand not just what the score was, but how it was produced.

DeepSeek v4 Pro Preview in this configuration. All 40 runs used DeepSeek deepseek-v4propreview. The oficial 81.6% result applies to the complete Qiushi–model–tool configuration and should not be attributed to the model alone. Qiushi Engine’s backend is configurable; DeepSeek was the choice for this evaluation, not a product constraint. Controlled backend comparisons remain future work.

## 8.6 Evidence Synthesis: From Paired Observations to Archive-Level Patterns

The case studies, zero-scored dimension analysis, and process signature data form a multi-layer evidence structure that allows the same results to be examined from diferent angles.

Case observations and global dimension patterns. The loss patterns observed in individual cases are consistent with the archive-wide zero-scored dimension distribution. The idea\_5 vs idea\_10 comparison shows that step\_limit\_reached does not predict score—the global correlation between step count and score $( r = - 0 . 0 1 )$ confirms this sample-level observation across all 40 tasks. The idea\_16 vs idea\_12 comparison distinguishes “external dependency cascade” from “breadth coverage shortfall” as distinct failure structures—the global dimension analysis similarly shows “core method component” (11 samples, often involving external dependencies) and “statistical support” (16 samples, involving breadth shortfall) as two independent principal sources of score loss. The idea\_37 case—65 steps, 84.9M tokens, 54.1 hours, yet three specific items unmet—validates the global finding that token volume does not substitute for specific rubric-item fulfillment $( r = 0 . 0 0 1 )$

Process signatures and scoring outcomes. Process-state data provides descriptive evidence about workflow allocation but not causal evidence about scores. Repair states (184 total) appear in both perfect-score and constrained samples, showing that repair is a normal component of the observed research process rather than a score predictor by itself. End states (188 total, mean $4 . 7 / \mathrm { r u n } )$ record repeated final checks; the mean End count for perfect-score samples (5.3) is slightly higher than in other bands, consistent with but not proving that verification contributed to high coverage.

From sample level to archive level. The macro-average of 0.8159 is an archive-level statistic whose standard error (0.0187) reflects how the substantial single-sample variance $( \sigma = 0 . 1 1 9 )$ is smoothed by 40-sample aggregation. Four perfect-score samples and four constrained samples occupy the distribution tails, but most samples $( 2 4 / 4 0 \geq 0 . 8 0 )$ fall in the “core implementation complete, marginal coverage incomplete” range. This left-skewed, high-concentration distribution shape indicates stable system capability on core research pipelines, with marginal variation driven primarily by task-specific conditions (external dependency availability, rubric-item count and type, task complexity) rather than systematic capability deficits.

Evidence chain completeness. All quantitative statements in this report trace to two classes of raw data in the submission archive: scorer JSON files (containing per-item binary judgments and natural-language explanations) and Meta-Trace files (containing per-step process records). The scoring mechanism in §2 and the evidence-source mapping in §4 show the path from task specification to macro-average. All values used in paired comparisons come from e2e\_hard\_per\_- sample\_rubric\_process\_summary.csv, which was extracted from the raw archive by automated scripts and verified through six consistency checks.

## 8.7 Methodology Considerations

Several aspects of the evaluation methodology merit explicit acknowledgment. The rubric-item scores are produced by an LLM judge (claude-sonnet-4-6), not by human experts; binary scoring does not capture partial completion; the oficial score is a macro-average over sample-level means, which difers from the micro-average item fulfillment rate $( 4 1 6 / 5 0 7 = 8 2 . 1 \% \ \mathrm { v s . \ 8 1 . 6 \% } )$ . The zero-scored dimension categories (statistical support, core components, etc.) are analytical classifications derived from rubric-item keywords and case inspection, not labels produced by the scorer itself; categories can overlap within a single sample. Meta-Trace is a system-generated process record, not an independent observation. UI mode, task family, and resource allocation were not randomly assigned. Execution environments, models, and tools difer across leaderboard entries, so the benchmark comparison reflects complete system configurations rather than isolating any single factor. The Qiushi score and standardized cost are now recorded on the oficial leaderboard.

## 8.8 Opportunities for Improvement

The zero-scored dimension analysis points to concrete improvement opportunities:

1. Early statistical planning. Prioritize statistical testing and repeated runs as integral parts of the experimental design, scheduled before the step budget is consumed by implementation and repair. Rather than treating repeated runs as an optional extension, plan for them from the first Explore step.

2. Dependency pre-checking. Before committing to an experimental design that requires specific external models, verify their availability in the execution environment. If a required model is unavailable, adapt the experimental design (e.g., using an available alternative with appropriate justification) rather than proceeding with a plan that cannot be completed.

3. Rubric-aware scheduling. Since each task comes with an explicit rubric, the Orchestrator could use the rubric items as a checklist during the Execute phase to ensure that all required dimensions are addressed before the session moves to Express.

4. Budget reservation. Reserve a fraction of the step budget for ablation studies and sensitivity analyses, scheduling them before final verification rather than after core implementation.

## 9 Conclusions

Qiushi Engine v0.8, using DeepSeek deepseek-v4pro-preview as the selected model backend, is recorded on the oficial AstaBench E2E-Bench-Hard leaderboard with a score of 0.816 and a standardized cost of \$15.209 per task. The next-highest displayed entry scores 0.655, a leaderboard diference of 16.1 percentage points. In addition, Qiushi perfectly completed 4 of 40 tasks (10%), compared with the approximately 3% best full-task completion rate reported for AstaBench’s oficial agents—a gain of 7 percentage points and roughly a 3.3-fold rate. These comparisons apply to complete system configurations rather than any single component.

The process analysis shows strong coverage of core research implementation—dataset handling, method and baseline construction, primary metric computation, and cross-artifact evidence bundling. Score losses concentrate in specific rubric dimensions: statistical support (16 samples), core method components requiring external resources (11), explicit metric completion (10), and ablation studies (8). These observed gaps suggest concrete improvements in resource allocation and dependency management.

AstaBench E2E-Bench-Hard tests a demanding form of automated research capability: completing unsimplified AI/NLP research cycles from task specification to evidence-supported technical reports. Qiushi’s result provides evidence of broad pipeline coverage under this benchmark, while the rubric analysis also makes clear that scientific novelty, human expert judgment, and causal attribution of system components require separate evaluation.

## A F<sub>u</sub>ll 40<sub>-</sub> S<sub>amp</sub>l<sub>e</sub> E<sub>v</sub>id<sub>ence</sub> T<sub>a</sub>bl<sub>e</sub>

T<sub>a</sub>bl<sub>e</sub> 1 0 li<sub>s</sub>t<sub>s a</sub>ll 40 t<sub>es</sub>t <sub>samp</sub>l<sub>es sor</sub>t<sub>e</sub>d b<sub>y score w</sub>ith <sub>score</sub>d it<sub>em coun</sub>t<sub>s zero</sub>-<sub>score</sub>d di<sub>mens</sub>i<sub>ons</sub> t<sub>as</sub>k f<sub>am</sub>il<sub>y process</sub> l<sub>eng</sub>th <sub>resource usage an</sub>d com<sub>p</sub>letion status <sub>.</sub>

T<sub>a</sub>bl<sub>e</sub> 1 0 All 40 E2E-B<sub>enc</sub>h-H<sub>ar</sub>d t<sub>es</sub>t <sub>samp</sub>l<sub>es</sub> <sub>sor</sub>t<sub>e</sub>d b<sub>y</sub> <sub>score .</sub> Th<sub>e</sub> <sub>unme</sub>t-it<sub>ems</sub> <sub>co</sub>l<sub>umn</sub> li<sub>s</sub>t<sub>s</sub> <sub>up</sub> t<sub>o</sub> t<sub>wo</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>ve</sub> it<sub>ems ;</sub> <sub>gap</sub> <sub>ca</sub>t<sub>egor</sub>i<sub>es</sub> <sub>use</sub> <sub>rea</sub>d<sub>a</sub>bl<sub>e</sub> <sub>s</sub>h<sub>or</sub>t l<sub>a</sub>b <sub>e</sub>l<sub>s .</sub>
<table><tr><td>ID</td><td>Task name</td><td>Score Met/-</td><td>Tot</td><td></td><td></td><td></td><td>Steps Tok(M) Hours Representative unmet items</td><td>Gap categories</td></tr><tr><td>idea_12</td><td>TAPP with Explicit Demonstrations</td><td>0.506/12</td><td></td><td>13</td><td>10.2</td><td>4.5</td><td>Large Language Model Selection; Accuracy Measurement;</td><td>metrics, statistics, ablation/sensitivity, required model</td></tr><tr><td>idea_16</td><td>Anticipatory Ensemble for Medication Extraction</td><td>0.54 7/13</td><td></td><td>38</td><td>52.8</td><td>13.5</td><td>GPT-3.5 Model Integration; LLaMA-2 Model Integration; ...</td><td>statistics, required model, core method</td></tr><tr><td>idea_10</td><td>Contextual Feedback Integration</td><td>0.67 10/15</td><td></td><td>75</td><td>103.2</td><td>28.9</td><td>Response Accuracy Metrics; Multiple Experimental Runs; </td><td>metrics, statistics, ablation/sensitivity, qualitative evaluation</td></tr><tr><td>idea_14</td><td>Adaptive FactCC Decoding</td><td>0.67 6/9</td><td></td><td>15</td><td>13.2</td><td>1.8</td><td>FactCC Evaluation Module Implementation; Ablation metrics, ablation/sensitivity, core Study; ..</td><td>method</td></tr><tr><td>idea_38</td><td>Tree-Logical Integration</td><td>0.70 7/10</td><td></td><td>15</td><td>9.3</td><td>0.9</td><td>Statistical Comparison; Multiple Experimental Runs;</td><td>statistics, required model</td></tr><tr><td>idea_39</td><td>RCI-Peer Review Integration</td><td>0.70 7/10</td><td></td><td>28</td><td>27.8</td><td>16.7</td><td>Secondary Metric: Reasoning Steps; Multiple Independent Runs; ….</td><td>metrics, statistics, qualitative evaluation</td></tr><tr><td>idea_31</td><td>Integrated Rule-Entailment Prototypical Networks</td><td></td><td>0.71 10/14</td><td>21</td><td>32.6</td><td>12.6</td><td>TACRED Dataset Implementation; Attention Mechanism Integration;</td><td>ablation/sensitivity, generalization/scaling, core method</td></tr><tr><td>idea_15</td><td>mT5 Cross-Lingual Integration</td><td>0.73 8/11</td><td></td><td>15</td><td>19.3</td><td>4.5</td><td>X-Fact Dataset Loading; Low-Resource Language Analysis; ...</td><td>statistics</td></tr><tr><td>idea_37</td><td>Reframed Coherent Instructions</td><td>0.73 8/11</td><td></td><td>65</td><td>84.9</td><td>54.1</td><td>F1 Score Calculation; Reframing Technique Documentation; ...</td><td>other</td></tr><tr><td>idea_47</td><td>Universal RandAugment Transformers</td><td>0.73 8/11</td><td></td><td>10</td><td>8.2</td><td>1.2</td><td>Compositional Generalization Evaluation; Multiple Training Runs; ...</td><td>ablation/sensitivity, generalization/scaling</td></tr><tr><td>idea_9</td><td>Hybrid Bottleneck-Selector Integration</td><td></td><td>0.73 11/15</td><td>43</td><td>62.3</td><td>8.1</td><td>Baseline System 2: Neural Selector Only; Query Throughput Measurement; ...</td><td>metrics, statistics, core method</td></tr><tr><td>idea_24</td><td>Entity-Prune Summarization</td><td></td><td>0.75 9/12</td><td>29</td><td>34.5</td><td>8.5</td><td>Dataset Selection and Preparation; Hallucination Rate other Evaluation; ...</td><td></td></tr><tr><td>idea_27</td><td>Adversarial Signal Transformation</td><td>0.75 9/12</td><td></td><td>14</td><td>11.7</td><td>1.8</td><td>Speaker Diversity Testing; Ablation Study; ..</td><td>ablation/sensitivity</td></tr><tr><td>idea_34 idea_23</td><td>AlignScore-NLI Integration</td><td>0.75</td><td>6/8</td><td>12</td><td>9.6</td><td>1.2</td><td>Text Generation Model; AlignScore Implementation</td><td>core method</td></tr><tr><td></td><td>Dynamic Task Weighting in Multi-Task Fact Verification</td><td>0.77</td><td>10/13</td><td>27</td><td>36.6</td><td>8.3</td><td>Health Domain Dataset; Statistical Significance Testing; ...</td><td>statistics</td></tr></table>

(continued from previous page)
<table><tr><td colspan="7" rowspan="1">ID      Task name                       Score Met/- Steps Tok(M) Hours Representative unmet items                      Gap categoriesTot</td></tr><tr><td colspan="2" rowspan="2">idea_35 Constraint-CoT Integration        0.79</td><td colspan="5" rowspan="2">11/14    19    18.6    1.2  Task Completion Time Measurement; Success Rate  metricsMeasurement;.</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="2">idea_7  Cross-Document Structured        0.80Extraction</td><td colspan="1" rowspan="1">12/15</td><td colspan="1" rowspan="1">45</td><td colspan="1" rowspan="1">66.3</td><td colspan="1" rowspan="1">5.9</td><td colspan="1" rowspan="2">Dataset Acquisition; Statistical Significance Testing; .. statistics</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="2">idea_30</td><td colspan="1" rowspan="2">Cluster-Enhanced Dimensional     0.80Reduction</td><td colspan="1" rowspan="1">12/15</td><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1">17.5</td><td colspan="1" rowspan="1">11.6</td><td colspan="1" rowspan="2">Hyperparameter Optimization; Isotropy Measurement; metrics, ablation/sensitivity</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_43</td><td colspan="1" rowspan="1">Integrated Verification and Critic  0.80</td><td colspan="1" rowspan="1">8/10</td><td colspan="1" rowspan="1">26</td><td colspan="1" rowspan="1">32.8</td><td colspan="1" rowspan="1">5.4</td><td colspan="1" rowspan="2">Knowledge Graph Integration; Automated            core methodExperimentation Framework</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Framework</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_1</td><td colspan="1" rowspan="1">Hierarchical Hybrid QA Integration0.83</td><td colspan="1" rowspan="1">10/12</td><td colspan="1" rowspan="1">55</td><td colspan="1" rowspan="2">81.1</td><td colspan="1" rowspan="2">10.9</td><td colspan="1" rowspan="2">Hybrid Evidence Utilization Algorithm; Multiple     statisticsExperimental Runs</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="2">idea_11</td><td colspan="1" rowspan="2">Offline-Continuous Prompt         0.83Optimization</td><td colspan="1" rowspan="1">10/12</td><td colspan="1" rowspan="1">27</td><td colspan="1" rowspan="1">35.3</td><td colspan="1" rowspan="1">7.2</td><td colspan="1" rowspan="2">Secondary Metrics; Containerized Environment       metrics</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="2">idea_46</td><td colspan="1" rowspan="2">CoT-Enhanced Type Annotations  0.83</td><td colspan="1" rowspan="1">10/12</td><td colspan="1" rowspan="1">18</td><td colspan="1" rowspan="1">9.4</td><td colspan="1" rowspan="1">1.6</td><td colspan="1" rowspan="2">Type Correctness Evaluation; Multiple Experimental statisticsRuns</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_50</td><td colspan="1" rowspan="1">Augmented Graph-Based Schema   0.83</td><td colspan="1" rowspan="1">10/12</td><td colspan="1" rowspan="1">60</td><td colspan="1" rowspan="1">79.9</td><td colspan="1" rowspan="1">12.7</td><td colspan="1" rowspan="2">SGD-X Dataset Implementation; Multiple Runs with core methodDifferent Seeds</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Encoding</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_13</td><td colspan="1" rowspan="1">DEBIE-MOMA Integration for     0.85</td><td colspan="1" rowspan="1">11/13</td><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">10.8</td><td colspan="1" rowspan="1">1.3</td><td colspan="1" rowspan="2">DEBIE Platform Implementation; MOMA Framework core methodImplementation</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Bias Mitigation</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_4</td><td colspan="1" rowspan="1">Emotion-Enhanced Multimodal    0.86</td><td colspan="1" rowspan="1">12/14</td><td colspan="1" rowspan="1">28</td><td colspan="1" rowspan="1">49.9</td><td colspan="1" rowspan="1">11.8</td><td colspan="1" rowspan="2">Vision-Language Transformer; Emotion Classification otherAccuracy Evaluation</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">CoT</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_48</td><td colspan="1" rowspan="1">Mix Self-Consistency with          0.86</td><td colspan="1" rowspan="1">12/14</td><td colspan="1" rowspan="1">24</td><td colspan="1" rowspan="1">19.1</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="2">Consistency Rate Measurement; Multiple Experimental metrics, statisticsRuns</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Self-Refinement</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_25</td><td colspan="1" rowspan="1">Integrated Model Editing for       0.87</td><td colspan="1" rowspan="1">13/15</td><td colspan="1" rowspan="1">39</td><td colspan="1" rowspan="1">53.3</td><td colspan="1" rowspan="1">49.9</td><td colspan="1" rowspan="2">Multiple Independent Runs; Meta-Analysis           statistics</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Consistency</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="2">idea_17</td><td colspan="1" rowspan="1">Diversified Causal Reasoning       0.88</td><td colspan="1" rowspan="1">14/16</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">13.3</td><td colspan="1" rowspan="1">6.0</td><td colspan="1" rowspan="2">Multiple Runs for Statistical Confidence; GPT-4 Model statistics, core methodImplementation</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_19</td><td colspan="1" rowspan="1">Cooperative Ontological NER      0.91</td><td colspan="1" rowspan="1">10/11</td><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1">12.7</td><td colspan="1" rowspan="1">1.4</td><td colspan="1" rowspan="1">Standard NER Baseline                               core method</td></tr><tr><td colspan="1" rowspan="1">idea_21</td><td colspan="1" rowspan="1">Integrated Feedback and            0.91</td><td colspan="1" rowspan="1">10/11</td><td colspan="1" rowspan="1">27</td><td colspan="1" rowspan="1">37.9</td><td colspan="1" rowspan="1">8.3</td><td colspan="1" rowspan="2">Multiple Experimental Runs                          statistics</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Self-Correction</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_32</td><td colspan="1" rowspan="1">Dynamic Streaming Evaluation     0.92</td><td colspan="1" rowspan="1">12/13</td><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">8.6</td><td colspan="1" rowspan="1">1.4</td><td colspan="1" rowspan="1">Computational Efficiency Measurement               metrics, efficiency/cost</td></tr><tr><td colspan="1" rowspan="1">idea_33</td><td colspan="1" rowspan="1">Adversarial Contextual             0.92</td><td colspan="1" rowspan="1">12/13</td><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">11.9</td><td colspan="1" rowspan="1">24.3</td><td colspan="1" rowspan="2">Coded Hate Speech Detection Analysis               other</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Embeddings</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_26</td><td colspan="1" rowspan="1">Integrated Contrastive Learning for0.93</td><td colspan="1" rowspan="1">13/14</td><td colspan="1" rowspan="1">20</td><td colspan="1" rowspan="1">28.0</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="2">11.6 Multiple Experimental Runs                          statistics</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Factual Extraction</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">idea_41</td><td colspan="1" rowspan="1">FastText-Enhanced Membership   0.93</td><td colspan="1" rowspan="1">13/14</td><td colspan="1" rowspan="1">25</td><td colspan="1" rowspan="1">38.2</td><td colspan="1" rowspan="1">5.4</td><td colspan="1" rowspan="2">Multiple Dataset Validation                           other</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Inference</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="10">(continued from previous page)</td></tr><tr><td>ID Task name</td><td></td><td>Score Met/-</td><td>Tot</td><td></td><td></td><td></td><td>Steps Tok(M) Hours Representative unmet items</td><td>Gap categories</td><td></td></tr><tr><td>idea_28</td><td>Contrastive Bayesian Hybrid</td><td>0.93</td><td>14/15</td><td>14</td><td>10.5</td><td>1.1</td><td>Low-Resource Dataset Implementation</td><td>core method</td><td></td></tr><tr><td>idea_20</td><td>Learning Syntactic Trigger Inspection</td><td>0.94</td><td>16/17</td><td>17</td><td>24.0</td><td>3.0</td><td>Varying Poisoning Rates</td><td></td><td>ablation/sensitivity</td></tr><tr><td>idea_5 idea_22</td><td>Dynamic Commonsense Integration</td><td>1.00</td><td>12/12</td><td>40</td><td>56.5</td><td>6.2</td><td></td><td></td><td>none</td></tr><tr><td></td><td>Memory-Augmented Decentralized Decision-Making</td><td>1.00</td><td>10/10</td><td>20</td><td>15.2</td><td>2.5</td><td></td><td>none</td><td></td></tr><tr><td>idea_29</td><td>Memory-Enhanced Tree Reasoning</td><td>1.00</td><td>11/11</td><td>20</td><td>24.1</td><td>3.7</td><td></td><td></td><td>none</td></tr><tr><td>idea_42</td><td>FactCC-CaPE Integration</td><td>1.00</td><td>16/16</td><td>79</td><td>105.4</td><td>21.1</td><td></td><td>none</td><td></td></tr></table>

## B Rubric Dimension Inventory

Across the 40 test samples, the 507 scored rubric items were classified into dimension categories based on the research action they require. Table 11 shows the categories, the number of samples afected by zero-scored items in each category, the primary mechanism that causes failures, and representative examples from the archive.

Table 11 Rubric dimension categories with zero-scored sample counts, primary failure mechanisms, and representative examples.
<table><tr><td rowspan=1 colspan=4>Dimension             Samples Primary Cause     Representative ExamplesCategory</td></tr><tr><td rowspan=3 colspan=3>Statistical support /       16      Resource-allocation  “Multiple Experimental Runs&quot;repeated runs                         pressure                in idea_35; “StatisticalSignificance Testing&quot; inidea_7, idea_30, idea_23</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=3>Core method                11      External               “Neural Selector&quot; in idea_9;component                             dependency or        “KG Integration&quot; in idea_43;implementation      “FactCC Eval Module” inidea_14</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Required metrics not      10</td><td rowspan=1 colspan=1>Implementation</td><td rowspan=2 colspan=1>“Response Accuracy&quot; inidea_10; “Type Correctness&quot;</td><td rowspan=22 colspan=1></td></tr><tr><td rowspan=1 colspan=1>completed</td><td rowspan=1 colspan=1>gaps or metric</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>misinterpretation</td><td rowspan=2 colspan=1>in idea_46; “ConsistencyRate&quot; in idea_48</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Ablation/                    8</td><td rowspan=1 colspan=1>Resource-allocation</td><td rowspan=1 colspan=1>“Ablation Study&quot; in idea_10,</td></tr><tr><td rowspan=1 colspan=1>sensitivity</td><td rowspan=1 colspan=1>pressure; late</td><td rowspan=3 colspan=1>idea_14, idea_27;“Hyperparameter&quot; in idea_27,idea_47</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>scheduling</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Other specific                5</td><td rowspan=1 colspan=1>Task-specific</td><td rowspan=1 colspan=1>“Compositional</td></tr><tr><td rowspan=3 colspan=1>requirements</td><td rowspan=1 colspan=1>constraints</td><td rowspan=3 colspan=1>Generalization&quot; in idea_47;“Hallucination Rate” inidea_24</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Exact model /                3</td><td rowspan=1 colspan=1>External</td><td rowspan=4 colspan=1>“GPT-3.5 Integration&quot; inidea_16; “LLaMA-2Integration&quot; in idea_16;“LLM Base&quot; in idea_38</td></tr><tr><td rowspan=3 colspan=1>backbone</td><td rowspan=1 colspan=1>dependency failure</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Human / qualitative       2</td><td rowspan=1 colspan=1>Inherently difficult</td><td rowspan=1 colspan=1>“Qualitative Analysis&quot; in</td></tr><tr><td rowspan=2 colspan=1>eval</td><td rowspan=1 colspan=1>for automated</td><td rowspan=1 colspan=1>idea 10; “Peer-Review</td></tr><tr><td rowspan=1 colspan=1>agents</td><td rowspan=1 colspan=1>Criteria&quot; in idea_39</td></tr><tr><td rowspan=1 colspan=1>Generalization/             2</td><td rowspan=1 colspan=1>Resource-allocation</td><td rowspan=3 colspan=1>“Cross-Lingual Evaluation&quot;</td></tr><tr><td rowspan=3 colspan=4>scalingEfficiency analysis          1                                  (none in constrained set;budget constraints   1 sample hadefficiency-related optionalitem)</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Low priority under</td></tr></table>

The distribution reveals a clear hierarchy of failure mechanisms. Resource-allocation pressure accounts for the largest share of failures (statistical support, ablation, generalization, eficiency), afecting items that are typically scheduled last in a research pipeline. External dependency failures produce the most severe per-sample impact (idea\_16 losing 6 items from two unavailable models), but afect fewer samples overall. Implementation complexity and metric misinterpretation fall between these extremes.

A notable feature is that no sample scored zero on all implementation items—even the lowest-scoring sample (idea\_12 at 0.50) satisfied 6 of 12 items, including the TAPP framework implementation, both datasets, and the prompt format documentation. This is consistent with high observed coverage of implementation-related items in this archive; the zero-scored items cluster around empirical breadth rather than core execution.

## C Submission Archive Contents

The submission archive (qiushi\_engine\_e2e\_bench\_hard\_test\_ui\_only\_v0.8\_20260824) contains the following files:

• qiushi\_engine\_e2e\_bench\_hard\_test.eval — Zipped Inspect log (112 MB) containing 45 entries: 40 sample JSONs (samples/idea\_\*\_epoch\_1.json), journal start record, header, summaries, and reductions.

• scores.json — Per-sample and aggregate scores from the oficial astabench score command.

• summary\_stats.json — Task-level statistics including the E2E-Bench-Hard test score (0.8159401260504202) and standard error (0.018745023003278058).

• submission.json — Submission metadata: agent name (Qiushi Engine), version (v0.8), actual model (deepseek-v4pro-preview), openness (Closed source & UI only), tool usage (Fully custom).

• test\_model\_configuration.json — Model-configuration record (Table 1).

• test\_ui\_telemetry.json — Per-sample telemetry: session IDs, steps, turns, tools, LLM calls, duration, and token counts.

• source\_log\_inventory.json — Per-sample score, run label, and SHA-256 hashes for packaged results and source eval logs.

• eval\_config.json — Inspect evaluation configuration.

• SHA256SUMS.json, SHA256SUMS.txt — Integrity checksums.

• README.md — Archive documentation with replay methodology.

## D P<sub>er-</sub> S<sub>amp</sub>l<sub>e</sub> P<sub>rocess</sub> Si<sub>gna</sub>t<sub>ure</sub> D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

Table 1 2 provides the per-sample Meta-Trace state counts for all 40 samples enabling detailed process analysis <sub>.</sub> The states are : Process (P forward work) Repair (R fixing problems) End (E final verification) Verify (V checking intermediate results) Plan (Pl making or repairing plans) Review (Rv checking quality) Reflect (Rf reconsidering approach) and Evaluate (Ev comparing choices) <sub>.</sub>

T<sub>a</sub>bl<sub>e</sub> 1 2 P<sub>er</sub>-<sub>samp</sub>l<sub>e</sub> M<sub>e</sub>t<sub>a</sub>-T<sub>race</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>coun</sub>t<sub>s</sub> f<sub>or</sub> <sub>a</sub>ll 40 <sub>samp</sub>l<sub>es</sub> <sub>sor</sub>t<sub>e</sub>d b<sub>y</sub> <sub>score .</sub>
<table><tr><td>ID</td><td>Task name</td><td>Score</td><td>P</td><td>R </td><td>E</td><td>V</td><td>Pl</td><td>Rv</td><td>Rf</td><td>Ev</td></tr><tr><td>idea_12</td><td>TAPP with Explicit Demonstrations</td><td>0.50</td><td>1</td><td>1</td><td>6</td><td>1</td><td>1</td><td>1</td><td>2</td><td>0</td></tr><tr><td>idea_16</td><td>Anticipatory Ensemble for Medication Extraction</td><td>0.54</td><td>6</td><td>3</td><td>9</td><td>11</td><td>9</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_10</td><td>Contextual Feedback Integration</td><td>0.67</td><td>24</td><td>22</td><td>16</td><td>0</td><td>3</td><td>8</td><td>2</td><td>0</td></tr><tr><td>idea_14</td><td>Adaptive FactCC Decoding</td><td>0.67</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_38</td><td>Tree-Logical Integration</td><td>0.70</td><td>7</td><td>0</td><td>4</td><td>4</td><td>1</td><td>1</td><td>2</td><td>0</td></tr><tr><td>idea_39</td><td>RCI-Peer Review Integration</td><td>0.70</td><td>6</td><td>5</td><td>5</td><td>8</td><td>1</td><td>1</td><td>0</td><td>2</td></tr><tr><td>idea_31</td><td>Integrated Rule-Entailment Prototypical Networks</td><td>0.71</td><td>6</td><td>3</td><td>5</td><td>4</td><td>3</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_15</td><td>mT5 Cross-Lingual Integration</td><td>0.73</td><td>4</td><td>3</td><td>4</td><td>0</td><td>2</td><td>1</td><td>1</td><td>0</td></tr><tr><td>idea_37</td><td>Reframed Coherent Instructions</td><td>0.73</td><td>48</td><td>5</td><td>4</td><td>0</td><td>3</td><td>1</td><td>0</td><td>4</td></tr><tr><td>idea_47</td><td>Universal RandAugment Transformers</td><td>0.73</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_9</td><td>Hybrid Bottleneck-Selector Integration</td><td>0.73</td><td>11</td><td>14</td><td>5</td><td>5</td><td>2</td><td>0</td><td>0</td><td>6</td></tr><tr><td>idea_24</td><td>Entity-Prune Summarization</td><td>0.75</td><td>12</td><td>8</td><td>6</td><td>1</td><td>2</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_27</td><td>Adversarial Signal Transformation</td><td>0.75</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_34</td><td>AlignScore-NLI Integration</td><td>0.75</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_23</td><td>Dynamic Task Weighting in Multi-Task Fact Verification</td><td>0.77</td><td>8</td><td>6</td><td>5</td><td>3</td><td>2</td><td>2</td><td>0</td><td>1</td></tr><tr><td>idea_35</td><td>Constraint-CoT Integration</td><td>0.79</td><td>6</td><td>3</td><td>5</td><td>3</td><td>0</td><td>1</td><td>1</td><td>0</td></tr><tr><td>idea_7</td><td>Cross-Document Structured Extraction</td><td>0.80</td><td>20</td><td>3</td><td>5</td><td>7</td><td>5</td><td>1</td><td>4</td><td>0</td></tr><tr><td>idea_30</td><td>Cluster-Enhanced Dimensional Reduction</td><td>0.80</td><td>4</td><td>0</td><td>9</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>idea_43</td><td>Integrated Verification and Critic Framework</td><td>0.80</td><td>16</td><td>0</td><td>5</td><td>0</td><td>2</td><td>2</td><td>0</td><td>1</td></tr><tr><td>idea_1</td><td>Hierarchical Hybrid QA Integration</td><td>0.83</td><td>26</td><td>8</td><td>4</td><td>6</td><td>8</td><td>0</td><td>0</td><td>3</td></tr><tr><td>idea_11</td><td>Offline-Continuous Prompt Optimization</td><td>0.83</td><td>11</td><td>9</td><td>4</td><td>0</td><td>1</td><td>1</td><td>0</td><td>1</td></tr><tr><td>idea_46</td><td>CoT-Enhanced Type Annotations</td><td>0.83</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_50</td><td>Augmented Graph-Based Schema Encoding</td><td>0.83</td><td>18</td><td>10</td><td>6</td><td>2</td><td>15</td><td>1</td><td>8</td><td>0</td></tr><tr><td>idea_13</td><td>DEBIE-MOMA Integration for Bias Mitigation</td><td>0.85</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_4</td><td>Emotion-Enhanced Multimodal CoT</td><td>0.86</td><td>12</td><td>1</td><td>6</td><td>3</td><td>1</td><td>5</td><td>0</td><td>0</td></tr><tr><td>idea_48</td><td>Mix Self-Consistency with Self-Refinement</td><td>0.86</td><td>5</td><td>6</td><td>5</td><td>2</td><td>1</td><td>4</td><td>1</td><td>0</td></tr><tr><td>idea_25</td><td>Integrated Model Editing for Consistency</td><td>0.87</td><td>21</td><td>0</td><td>8</td><td>5</td><td>1</td><td>1</td><td>2</td><td>0</td></tr><tr><td>idea_17</td><td>Diversified Causal Reasoning</td><td>0.88</td><td>2</td><td>3</td><td>5</td><td>1</td><td>1</td><td>2</td><td>2</td><td>0</td></tr></table>

(continued from previous page)
<table><tr><td>ID</td><td>Task name</td><td>Score</td><td>P</td><td>R</td><td>E</td><td>V</td><td>Pl</td><td>Rv</td><td>Rf</td><td>Ev</td></tr><tr><td>idea_19</td><td>Cooperative Ontological NER</td><td>0.91</td><td>4</td><td>8</td><td>4</td><td>4</td><td>1</td><td>3</td><td>3</td><td>2</td></tr><tr><td>idea_21</td><td>Integrated Feedback and Self-Correction</td><td>0.91</td><td>3</td><td>7</td><td>9</td><td>1</td><td>7</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_32</td><td>Dynamic Streaming Evaluation</td><td>0.92</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_33</td><td>Adversarial Contextual Embeddings</td><td>0.92</td><td>0</td><td>1</td><td>5</td><td>1</td><td>2</td><td>2</td><td>1</td><td>0</td></tr><tr><td>idea_26</td><td>Integrated Contrastive Learning for Factual Extraction</td><td>0.93</td><td>12</td><td>0</td><td>5</td><td>0</td><td>1</td><td>2</td><td>0</td><td>0</td></tr><tr><td>idea_41</td><td>FastText-Enhanced Membership Inference</td><td>0.93</td><td>7</td><td>8</td><td>4</td><td>0</td><td>0</td><td>3</td><td>3</td><td>0</td></tr><tr><td>idea_28</td><td>Contrastive Bayesian Hybrid Learning</td><td>0.93</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>idea_20</td><td>Syntactic Trigger Inspection</td><td>0.94</td><td>7</td><td>0</td><td>5</td><td>1</td><td>1</td><td>3</td><td>0</td><td>0</td></tr><tr><td>idea5</td><td>Dynamic Commonsense Integration</td><td>1.00</td><td>16</td><td>11</td><td>7</td><td>1</td><td>1</td><td>2</td><td>0</td><td>2</td></tr><tr><td>idea_22</td><td>Memory-Augmented Decentralized Decision-Making</td><td>1.00</td><td>7</td><td>7</td><td>8</td><td>3</td><td>0</td><td>1</td><td>4</td><td>2</td></tr><tr><td>idea_29</td><td>Memory-Enhanced Tree Reasoning</td><td>1.00</td><td>4</td><td>4</td><td>5</td><td>4</td><td>0</td><td>0</td><td>0</td><td>3</td></tr><tr><td>idea_42</td><td>FactCC-CaPE Integration</td><td>1.00</td><td>41</td><td>25</td><td>5</td><td>1</td><td>1</td><td>2</td><td>0</td><td>4</td></tr></table>

The process signature table should be read as parsed process-state evidence. Some high-scoring runs were short (for example, idea\_32: 10 steps and idea\_33: 13 steps), but rows with all-zero state counts indicate unavailable or unparsed state evidence rather than a zero-state workflow. High-repair runs such as idea\_42 (25 Repair, score 1.0) and idea\_10 (22 Repair, score 0.667) show opposite scored outcomes; the process counts describe workflow allocation and do not by themselves identify causal efects on score.

## E Data Processing Methodology

Figures and tables describing individual runs were generated from the submission archive and the 40 Meta-Trace process records using reproducible Python scripts. The leaderboard comparison and reported monetary cost use the oficial leaderboard snapshot. The processing pipeline consisted of the following steps:

## E.1 Script Descriptions and Verification Chain

The data processing pipeline consists of five Python scripts, each reading from the previous step’s output and producing inputs for the next, forming a complete reproducible evidence chain.

Script 1: aggregate\_e2e\_hard\_submission.py. Input: scores.json, summary\_stats.json, submission.json, test\_model\_configuration.json, test\_ui\_telemetry.json, and source\_l og\_inventory.json. Processing: merges six JSON files into per-sample records and computes aggregate statistics (mean, standard error, median, min/max, total tokens, total duration). Output: e2e\_hard\_40\_run\_summary.csv (40 rows, 16 columns per row) and e2e\_hard\_aggregate\_stats.json. Verification: confirms sample count = 40, score mean exactly matches the oficial value from summary\_stats.json.

Script 2: inspect\_eval\_structure.py. Input: qiushi\_engine\_e2e\_bench\_hard\_test.eval (112 MB zipped Inspect log). Processing: reads all 45 entries via zipfile module (1 journal, 40 samples, summaries, reductions, header), extracts header metadata (task name, model identifier, scorer name), and records each sample’s event-chain structure. Output: eval\_structure\_summary.json. Verification: confirms the header task astabench/e2e\_discovery\_hard\_test, model deepseek/deeps eek-v4pro-preview, and scorer score\_rubric.

Script 3: extract\_e2e\_hard\_evidence\_chain.py. Input: the 40 sample JSONs from the .eval file. Processing: extracts task definitions (name, description, hypothesis, variables, contrasts, measurements), score values, scorer explanation text, event counts, and output package sizes from each sample’s Inspect events; merges with UI telemetry and Meta-Trace process summaries. Output: e2e\_hard\_per\_sample\_evidence\_chain.csv (10.9 MB, 40 rows with complete scorer explanation text and event chains). Verification: extracted scores compared per-sample against source\_log\_inventory.json, confirming zero mismatches.

Script 4: extract\_all\_rubric\_process\_patterns.py. Input: Script 3 output. Processing: parses scorer explanation strings to extract individual rubric items and corresponding binary scores; simultaneously extracts process-state counts (Process, Repair, End, Verify, Plan, Review, Reflect, Evaluate) from Meta-Trace files. Output: e2e\_hard\_rubric\_item\_scores\_long.csv (507 rows in long format) and e2e\_hard\_per\_sample\_rubric\_process\_summary.csv (40 rows). Verification: the

recomputed score mean matches the oficial value to all significant digits (0.8159401260504202);   
total scored items = 507, met items = 416, and zero-scored items = 91.

Script 5: build\_report\_figures\_tables.py. Input: all outputs from Scripts 1–4. Processing: generates 11 analytical figures and 5 structured JSON tables. Verification: all numerical values consistent with source data; image files non-empty.

## E.2 Cross-Step Consistency Verification

To ensure numerical accuracy throughout this report, the data processing chain performs consistency verification at two independent levels:

Level 1: Source data internal consistency. Per-sample scores from scores.json are compared one-by-one against the independent records in source\_log\_inventory.json, confirming exact matches for all 40 samples.

Level 2: Extraction–recomputation consistency. Per-rubric-item scores parsed from scorer explanation text are reaggregated into sample-level scores and compared against oficial scores. The recomputed mean (0.8159401260504202) matches the oficial value to all significant digits.

These checks verify that the reported scores are traceable to the original submission data and agree with the independent recomputation. Token and step counts come from the run telemetry; monetary cost uses the oficial leaderboard’s \$15.209-per-task value.[16]

## References

[1] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, et al. Evaluating large language models trained on code, 2021.

[2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, et al. Program synthesis with large language models, 2021.

[3] Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, et al. DS-1000: A natural and reliable benchmark for data science code generation, 2022.

[4] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2020.

[5] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, et al. GPQA: A graduate-level google-proof Q&A benchmark, 2023.

[6] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues?, 2023.

[7] Ben Bogin, Kejuan Yang, Shashank Gupta, Kyle Richardson, Erin Bransom, et al. SUPER: Evaluating agents on setting up and executing tasks from research repositories, 2024.

[8] Jonathan Bragg, Mike D’Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, et al. Astabench: Rigorous benchmarking of ai agents with a scientific research suite, 2025.

[9] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, et al. AgentBench: Evaluating LLMs as agents, 2023.

[10] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating language agents on machine learning experimentation, 2023.

[11] Jun Shern Chan, Neil Chowdhury, Oliver Jafe, James Aung, Dane Sherburn, et al. MLE-bench: Evaluating machine learning agents on machine learning engineering, 2024.

[12] Zachary S. Siegel, Sayash Kapoor, Nitya Nagdir, Benedikt Stroebl, and Arvind Narayanan. CORE-Bench: Fostering the credibility of published research through a computational reproducibility agent benchmark, 2024.

[13] Giulio Starace, Oliver Jafe, Dane Sherburn, James Aung, Jun Shern Chan, et al. PaperBench: Evaluating AI’s ability to replicate AI research, 2025.

[14] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, et al. ScienceAgentBench: Toward rigorous assessment of language agents for data-driven scientific discovery, 2024.

[15] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. The AI scientist: Towards fully automated open-ended scientific discovery, 2024.

[16] Allen Institute for AI. Astabench leaderboard, 2026. Accessed 2026-09-04.