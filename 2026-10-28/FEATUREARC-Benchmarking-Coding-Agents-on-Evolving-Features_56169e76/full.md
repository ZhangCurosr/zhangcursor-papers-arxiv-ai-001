# FEATUREARC: Benchmarking Coding Agents on Evolving Features

Yunlong Wang<sup>1</sup>, Alyssa Wu<sup>1</sup>, Cuiyun Gao<sup>2</sup>, Jia Feng<sup>1</sup>, Songyuanyi Lu<sup>1</sup>, Zhichao Hu<sup>1</sup> <sup>1</sup>Hunyuan Team, Tencent

<sup>2</sup>The Chinese University of Hong Kong

{joeyolwang, alyssawwu, plusfeng, alinasyylu, frankiehu}@tencent.com, cygao@cse.cuhk.edu.hk <sup>\*</sup>Corresponding authors: Alyssa Wu and Cuiyun Gao.

Abstract—A software feature usually evolves over time: developers extend its first version as new use cases and requirements arrive, and each extension builds on code written earlier. Most feature benchmarks evaluate a single request on a fixed codebase. Feature evolution requires evaluating successive extensions on an agent’s own earlier code and checking whether the feature remains correct. We introduce FEATUREARC to evaluate coding agents on evolving features, together with a pipeline that constructs its tasks from repository histories. The benchmark combines dependencies between successive stages of the same feature, explicit requirements and interface contracts, and executable checks across four programming languages. The pipeline validates fail-to-pass (F2P) tests for new behavior and pass-to-pass (P2P) tests for existing behavior on the reference code before and after each stage. We evaluated 10 coding agents on 73 manually reviewed tasks of 2 to 10 stages. Each agent works in one persistent session and workspace without external test feedback or rollback. The evaluation reveals a gap between first-stage and whole-feature success: pooled success falls from 62.7% at the first stage to 28.5% when the whole feature is tested on the final code. Trajectory analysis identifies inherited defects as a common failure mechanism, present in 72.9% of failed later stages. These earlier errors often remain undetected: in stages following a failure, agents leave the earlier error unnoticed about 95% of the time, and their self-tests rarely revisit earlier stages. Evolving a feature autonomously, without human checks between stages, therefore remains a substantial challenge for current coding agents. At Tencent, the pipeline extends internal evaluations to this setting and guides the development of Hunyuan, Tencent’s family of large language models.

Index Terms—coding agents, benchmark, software evolution, feature development, mining software repositories

## I. INTRODUCTION

Software features evolve as developers extend existing code to meet new requirements [1], [2]. We describe this evolution as a sequence of stages, from the initial implementation to successive extensions. Coding agents now inspect repositories, edit code, and run tests autonomously [3], [4]. On SWE-bench, which asks agents to resolve individual GitHub issues [5], the best model resolved 1.96% at release in 2023; agents now resolve more than 70% of its 500-issue, human-validated Verified subset [6], [7]. In 37% of the 13,000 real sessions collected in SWE-chat, agents wrote virtually all committed code [8]. Developers have also begun delegating successive extensions of a whole feature without intervening checks. Each extension then inherits the agent’s earlier code, including errors the agent has not noticed. We ask: can an agent that implements single requirements well keep a feature correct while extending its own code as requirements evolve?

![](images/7c4e2201d1f4683c00bd8cc416f3cc8abd13f23726421d6c56149e8d2d06473b.jpg)  
Fig. 1. L4 feature implementation.

Evaluating this ability requires agents to sustain a feature while checking their own earlier work. User feedback between stages determines which implementation and verification responsibilities remain with the agent. Similar to the levels of driving automation [17] and of AI-agent autonomy [18], we distinguish four levels of delegation for feature implementation: at L1 the user tells the agent how to fix a failed stage, at L2 the user points out the failing behavior, at L3 the user only says whether a stage passed, and at L4 the user only gives the next specification and says nothing about earlier stages (Figure 1). L4 feature implementation requires the agent to find earlier errors itself, whereas at L1–L3 the user reports failed stages. Tencent’s internal multi-stage feature evaluations covered only L1–L3, creating a practical need for L4 evaluation of sustained development. An earlier attempt to build L4 tasks by splitting a single feature request into stages produced weak dependencies: the resulting stages behaved largely as independent single-stage tasks, limiting their ability to test successive extensions. L4 therefore needs stages that depend on each other by construction. We recover these stages from repository histories to capture how developers actually extended a feature to meet new requirements, with each extension building on earlier implementations.

TABLE I  
COMPARISON OF FEATURE BENCHMARKS.
<table><tr><td>Benchmark</td><td>Iteration on Single Repository own code feature</td><td></td><td>history</td><td>Automated construction languages</td><td>Multiple</td></tr><tr><td>One feature request per task</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FEA-Bench [9]</td><td>x</td><td>x</td><td>J</td><td>√</td><td>x</td></tr><tr><td>FeatBench [10]</td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>FeatureBench [11]</td><td>x</td><td>x</td><td>△</td><td>√</td><td>x</td></tr><tr><td>Several stages per task</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CCBench [12]</td><td>x</td><td>√</td><td>△</td><td>x</td><td></td></tr><tr><td>SWE-STEPS [13]</td><td>√</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>SlopCodeBench [14]</td><td>√</td><td>△</td><td>x</td><td>x</td><td>x</td></tr><tr><td>EvoCode-Bench [15]</td><td>√</td><td>△</td><td>x</td><td>x</td><td>√</td></tr><tr><td>EvoWeb [16]</td><td>√</td><td>△</td><td>x</td><td>√</td><td>x</td></tr><tr><td>FEATUREARC (ours)</td><td></td><td></td><td>1</td><td>J</td><td>L</td></tr></table>

Existing feature benchmarks address parts of this setting, but leave a gap in evaluating same-feature evolution from repository histories at L4. As shown in Table I, most pose a single feature request on a fixed starting state [9]–[11], so there is no later stage for an error to affect. Benchmarks with several stages evaluate different development settings: CCBench [12] gives all stages in one instruction; SWE-STEPS [13] returns test feedback after each stage and chains pull requests that need not belong to one feature; and SlopCodeBench [14], EvoCode-Bench [15], and EvoWeb [16] use stages written by hand or generated from similar products, which do not follow how a feature actually grew in a repository. Among the benchmarks compared in Table I, FEATUREARC is the only one that combines historical same-feature dependencies, sequential requests, and continued evaluation on the agent’s own code without hidden-test feedback. Ticks, triangles, and crosses indicate full, partial, and absent support; automated construction covers tasks, environments, and tests, with human review allowed.

We address this gap with FEATUREARC and an automated pipeline for constructing its tasks. The benchmark has three complementary properties. First, historical samefeature dependencies require agents to extend an existing implementation: the pipeline recovers successive stages of one user-visible feature from repository histories. Second, requirements-grounded verification makes each extension explicit and testable: a requirements document and interface contract specify the expected behavior, while fail-to-pass (F2P) tests and pass-to-pass (P2P) regression tests [19] check new and existing behavior against reference states. A final check evaluates the complete feature on the agent’s resulting code. Third, diverse repository contexts expose agents to feature evolution in Python, TypeScript, Rust, and Go, across seven application domains. We select the evaluation set through stratified sampling, multi-model trajectory screening for task defects, and manual checks (Section III-F). At L4, each agent receives specifications sequentially in a persistent session and workspace; hidden-test results are withheld, and the agent carries its own code forward after every stage.

The results expose three weaknesses of the evaluated coding agents on evolving-feature tasks. First, whole-feature success is less than half of first-stage success: pooled success falls from 62.7% at the first stage to 28.5% on the whole feature, and even the strongest agent, Claude Opus 5.5 [20], completes only 49.3% of the features, although it passes 83.6% of first stages. Second, inherited errors often persist: after a failed stage, agents pass the next stage’s F2P tests in only 22.6% of cases, versus 70.2% after a passed one, whereas breaking earlier behavior affects only 3.3% of features whose new behavior works. Third, agents rarely notice their own earlier failures: in 94.9% of the stages that follow a failed stage, the agent never notices the error. Evolving a feature without human checks between stages therefore remains a substantial challenge for current coding agents.

At Tencent, we use this pipeline to add L4 feature implementation to internal evaluations and assess Hunyuan 4 Preview, a pre-release model in the Hunyuan family [21], alongside external agents. The evaluation distinguishes implementing a new request from detecting errors in the code on which it depends. This distinction gives the Hunyuan team concrete evidence about failures that first-stage scores do not reveal, as discussed in Section VI.

Our contributions are:

• A benchmark and construction pipeline that combine historical same-feature dependencies, requirementsgrounded verification, and diverse repository contexts for L4 feature implementation.

• An empirical study of coding agents on FEATUREARC that quantifies the gap between first-stage and wholefeature success and documents the persistence of inherited errors and limitations in agents’ self-verification.

• An industrial application at Tencent: extending internal evaluation to L4 and diagnosing Hunyuan’s development needs through feature-level outcomes and trajectory evidence.

## II. TASK FORMULATION

We define a task as a sequence of stages and specify how the agent is evaluated under the L4 setting introduced in Section I.

## A. Task Definition

A task follows one user-visible feature through an ordered sequence of n stages, $m _ { 0 } , \ldots , m _ { n - 1 }$ , in one repository. The first stage implements the initial version, and each later stage extends the same feature. We call this sequence a same-feature chain.

In the task illustrated in Figure 1, the first stage, m<sub>0</sub>, adds app.run(), which starts a run and returns its task identifier. The second stage, m<sub>1</sub>, adds app.runAndWait(), which calls app.run() and waits for completion. The third stage, m<sub>2</sub>, extends the feature to accept file input and adds files.upload(). Each stage builds on the agent’s implementation of the preceding requirements.

The task provides an initial repository state $S _ { 0 } ,$ a container image with the required dependencies, and a specification for each stage. Each specification has two parts: a requirements document describing the requested behavior and acceptance criteria, and an interface contract stating the names, signatures, constants, and error messages needed to use that behavior. The agent receives the specifications one at a time and implements each new request on top of its own earlier work.

Hidden fail-to-pass (F2P) tests check the new behavior requested at a stage, and pass-to-pass (P2P) tests check existing behavior [19]. A separate pair of chain-level suites checks the complete feature on the final workspace. Section III explains how we recover the stages, write their specifications, and validate these tests against reference code.

## B. Evaluation Process

Figure 2 shows the L4 evaluation process. One session and workspace persist through the whole task. The agent can inspect its code and run its own checks; reference patches, hidden test suites, and future specifications stay private. When the agent finishes a stage, a hidden acceptance check, called a stage gate, runs its F2P and P2P suites. The stage is the implementation request; the gate checks its result, and both suites must pass for the stage to pass. Failed stages are neither rolled back nor reported to the agent: the next stage starts from the code the agent just produced. After $m _ { n - 1 }$ , a chain gate runs the chain-level F2P and P2P suites on the final workspace to check the complete feature.

## III. BENCHMARK CONSTRUCTION

The construction pipeline turns repository histories into tasks that require an agent to extend the same feature on its own code. Reliable evaluation requires dependent stages, explicit behavior specifications, and executable tests that distinguish the requested new behavior from existing behavior. As shown in Figure 3, the pipeline addresses these requirements in five steps: repository collection and history analysis, evolutionchain mining, reference-state construction, environment and test validation, and specification generation. We then select and manually review the evaluation set.

## A. Step 1: Repository Collection and History Analysis

To recover meaningful feature evolution, we need histories that contain successive extensions and repositories that can support executable evaluation. As shown in Step 1 of Figure 3, we select repositories with at least 30 stars, permissive licenses, offline Linux builds, and code written mainly in Python, TypeScript, Rust, or Go. The star threshold provides a basic indication of community interest; the other conditions support redistribution and repeatable execution. Repository histories provide the feature changes and reference implementations used to construct and validate each task. During evaluation, the agent extends its own earlier code, so each new requirement must be implemented against the workspace produced by its preceding edits.

![](images/ea1356d8caa483b3f8b8135493c9d4fdf07fd3e543e66e83aee85dac1c54caa0.jpg)  
Fig. 2. L4 evaluation process.

Following a feature through file renames and module moves requires evidence beyond file names: changes to shared files suggest a relationship [22], which we examine through the module structure and the intent of the associated commits. We therefore build a module map from imports at the starting checkpoint and update it as we process commits in order. The topology and semantic agents in Figure 3 check complementary evidence: the former checks module structure, and the latter identifies commit intent and associated tests. Their output supplies the historical relationships used in the next step.

## B. Step 2: Evolution-Chain Mining

To evaluate evolution of one feature, we must distinguish its successive extensions from unrelated changes that happen nearby in time or in the same files. We represent candidate relationships in an evidence graph, illustrated in Step 2 of Figure 3. Nodes are commits or small commit groups, and edges connect changes through shared code elements, file history, references, or imports. This representation allows the pipeline to follow related changes even when other development occurs between them.

The four searches shown in the figure explore complementary parts of this graph: global beam search follows promising paths, anchor-guided search starts from identified feature changes, endpoint bridging connects related endpoints, and community-guided search examines groups of related changes. An LLM review agent checks the proposed chains against source code, tests, and diffs. A chain is retained only when every stage extends the same user-visible feature with observable, testable behavior. The review also adds missed extensions, removes unrelated work, and adjusts stage boundaries. The output is the mined same-feature chain used to construct the task.

![](images/b91b0848eb9b089f6e32bc8bf7da299ef7c912d42676ed581b4b5cd313d2c733.jpg)  
Fig. 3. Benchmark construction workflow.

## C. Step 3: Reference-State Construction

To validate the contribution of each stage, we construct paired reference states that differ by the selected extension while retaining the surrounding repository context. As shown in Step 3 of Figure 3, we label historical file patches by their role in the feature, its tests, and necessary dependencies, and assign the selected changes to stage patches. We remove these patches from the final checkpoint to obtain the initial state S<sub>0</sub>, then reapply them in order to obtain $S _ { 1 } , \ldots , S _ { n }$ . Unselected changes remain as shared context. These paired states let the next step test the contribution of each stage directly.

## D. Step 4: Environment and Test Validation

To attribute a failure to the requested behavior, the task must run consistently and its checks must distinguish the code before and after the extension. The environment agent in Step 4 of Figure 3 prepares one offline container image for the chain and repairs it until the reference states build and their tests can be discovered. The test-extraction agent packages historical tests with their required fixtures and helpers, and selects tests according to their observed outcomes on the paired states. F2P tests must fail before the stage and pass afterward; P2P tests must pass on both states. The chainlevel suites undergo the same validation between $S _ { 0 }$ and $S _ { n } ,$ enabling the final check to cover the complete feature across all stages.

## E. Step 5: Specification Generation

To make the expected behavior accessible without revealing its implementation, each stage needs a behavioral description and an explicit interface contract. As shown in Step 5 of Figure 3, a specification agent uses the stage patch and F2P tests to write the requirements document and contract. The former states behavior and acceptance criteria; the latter specifies the names, signatures, constants, and error messages needed to use that behavior. Before publication, the pipeline removes registered agent-generated documents and restores documents identified as revealing later implementations to their pre-feature state. The figure’s two output groups reflect this separation: the agent receives initial code, sequential specifications, and the container image, while reference code, hidden tests, and validation records remain private.

## F. Evaluation-Set Curation

Multi-stage evaluation requires repeated model calls and sequential execution within each feature, with costs incurred for every evaluated agent. Given these economic constraints, we select a high-quality subset of the constructed task pool as our evaluation set through three steps.

Stratified sampling. To cover different languages and evolution lengths, we sample 100 candidate tasks, 25 per language, stratified by chain length. This retains both short and longer feature-evolution chains in the candidate set. A scan for test-used identifiers missing from the specifications identifies descriptions and contracts that need revision before execution.

Task-defect exclusion. To identify task defects, we run candidate tasks with several models and inspect their execution trajectories. We check these trajectories against the specifications, hidden tests, and reference implementations to distinguish task defects from agent errors. This screening excludes seven tasks: five whose chain-level suites require unspecified behavior, one with a defective test suite, and one with a repository hook that intercepts the prompt.

Manual check. To ensure that the selected tasks measure the intended feature-evolution setting, we manually check them for specification quality, alignment between interface contracts and hidden tests, and whether successive stages extend the same feature. The resulting FEATUREARC evaluation set contains 73 tasks with 263 stages from 53 repositories. All reported agent comparisons use this same set.

## G. Benchmark Characteristics

FEATUREARC supports evaluation of evolving-feature implementation through three complementary properties. Figure 4 provides the supporting dataset characteristics.

![](images/570de7fcb52dfdc9c309eafd51bd5813997cd731ab07c66d3f5f2de26702ced3.jpg)

![](images/591e9531246ec86e95729f6a4491d13ae9a0d8101289f1f4591c96ce6d79d94c.jpg)

![](images/b09e8999fb9ad57c5a8249e437a62b29283e73106746da6156e977bb7f50e3d0.jpg)  
Fig. 4. Characteristics of FEATUREARC.

Historical same-feature dependencies. Later stages require agents to revisit earlier work, making inherited implementation decisions part of the task. Every task connects successive stages through shared files or verified call dependencies. As shown in Figure 4d, a later stage shares, on average, 54.3% of its changed files with earlier stages; the median across tasks is 59.1%. Tasks span 2 to 10 stages, with 3.60 on average, providing both short extensions and longer sequences of dependent development.

Requirements-grounded verification. The benchmark combines explicit stage specifications with validated stagelevel and whole-feature checks. These tasks involve substantial changes across a repository: Figure 4c shows median reference changes of 426 source lines and 416 test lines per stage, with whiskers at the 5th and 95th percentiles; Figure 4d shows an average of 7.71 changed source files. The median wholefeature change is 1,453 source lines. Across implementation changes of this scale, paired-state validation and manual checks establish alignment between the executable evaluation and the behavior requested in the stage specifications.

Diverse repository contexts. The tasks cover four languages and seven application domains, assigned per repository by hand, allowing the same evaluation process to examine different development settings. Figure 4b shows a spread across Python, Go, TypeScript, and Rust (21, 20, 18, and 14 tasks, respectively). Figure 4a includes AI agents and LLM tooling as well as cloud and infrastructure, with 17 and 14 tasks, respectively. The shared task structure supports evaluation across varied code, interfaces, and application needs: each agent must extend one feature per task.

## IV. EXPERIMENTAL SETUP

We evaluate coding agents on FEATUREARC using the L4 process in Section II-B. Our study addresses three complementary questions:

• RQ1: How reliably do agents complete evolving features? Stage and chain gates measure implementation success, failure position, and the effect of task length.

• RQ2: How do earlier errors affect later stages? Conditional pass rates and inheritance labels distinguish defects carried forward from regressions introduced by later edits.

• RQ3: How effectively do agents detect and verify their own errors? Awareness labels, self-test outcomes, and specification checks examine whether agents recognize failures and obtain reliable evidence of correctness.

## A. Evaluated Agents

To examine this setting across providers, we evaluate agents built on Claude Opus 5.5 [20], GPT-6 Astra [23], Hunyuan 4 Preview [21], GLM-5.3 [24], Grok 4.7 [25], Qwen 3.8 [26], Gemini 3.8 Flash [27], Doubao 2.1 Pro [28], Kimi K3 [29], and DeepSeek V4 Pro [30]. Hunyuan 4 Preview is the internal pre-release model evaluated in this study; the citation describes its model family. The unit of comparison is the configured coding agent, including its model and harness.

## B. Trajectory Analysis

Gate outcomes locate failures, while trajectories show what the agent checked and whether it recognized an earlier error. We use trajectory analysis to connect these observations for RQ2 and RQ3. Process labels describe behavior visible in the specifications, trajectory, and agent code; diagnosis labels additionally use hidden-test evidence to explain the observed failure.

We use the following labels to describe verification behavior and diagnose failed stages.

• Final check records the strongest check the agent ran: relevant tests, unrelated tests, a manual behavior check, a build only, no check, or no activity. Last test result records whether the last test command passed, failed, collected no tests, was not run, or had an unclear result.

• Criterion targeting records whether the checks exercise all acceptance criteria, some criteria, only implementation details, or no behavior. It measures what the checks exercise; their expected outcomes may still be wrong.

• Stop reason records whether the agent ended after a successful check, ended without verification, dismissed a failure, ignored a failure, or did no work. Dismissal evidence records whether a claim that a failure was unrelated or pre-existing is supported by a baseline test, unverified, or contradicted by the trajectory.

• Prior awareness records whether an agent missed an earlier failure, noticed it without repairing it, or repaired it. It applies to stages following a failed stage.

• Inheritance records whether a failed later stage was caused wholly, partly, or not at all by an earlier defect, together with the stage that introduced that defect.

• F2P cause: fine records the specific reason for an F2P failure, such as a missing symbol, a wrong signature, incorrect logic, or a missed case. F2P cause: grouped combines these labels into interface, other build, and behavioral failures, while keeping timeouts, failures unrelated to the solver, and unknown causes separate.

## C. Implementation Details

Task construction and review. The pipeline implements the five steps in Figure 3 as agents with separate responsibilities for history review, feature-chain review, patch classification, environment preparation, test extraction, and specification writing. Published task manifests record the models and artifacts used at each step. We used GPT-5.5 [31] for Steps 1–3, DeepSeek V4.1 Flash [32] for Step 4, and Claude Sonnet 5 [33] for Step 5. Acceptance of a task depends on executable reference-state validation and review of the task materials.

Agent execution and comparability. Nine evaluated agents run in Claude Code,<sup>1</sup> while GPT-6 Astra uses the native Codex $\mathrm { C L I } . ^ { 2 }$ For GPT-6 Astra, Codex preserves the native tool-call representation and session-resumption mechanism, avoiding the additional adaptation required to run it through Claude Code. Each harness resumes the same session at the next stage. Each integration preserves its harness’s handling of tool calls and conversation state, with context and reasoning controls specific to the provider. Every agent is evaluated under the same task conditions: initial code, stage specifications, hidden acceptance checks, and a four-hour stage budget, with no external failure feedback and no rollback between stages. Our principal comparisons follow each agent from its first stage to the whole feature and compare its behavior after passed and failed stages, keeping its configuration fixed. Cross-agent scores reflect joint model–harness performance, with the two contributions coupled in these comparisons. Tasks execute in offline containers with preinstalled dependencies, and hidden scoring runs separately from the agent’s workspace.

Trajectory annotation. Hunyuan 4 Preview, DeepSeek V4.1 Flash, and GLM-5.3 Flash [24] independently annotate the trajectories using a shared codebook, following the LLMas-judge approach [34]. They assign process labels before accessing gate logs, hidden tests, and reference patches for diagnosis, and cite trajectory evidence for every label. Reported labels use the majority decision; the first author adjudicates items on which all three judges disagree. Analyses requiring a defect’s origin use inherited failures whose originating stage can be identified. To validate the labels, the first author independently annotates a stratified sample without seeing the model labels: passed stages, failed stages with no prior failure, and stages following a failure. Agreement is measured before adjudication, using only applicable and resolved comparisons; dismissal evidence is compared when both judgments identify a dismissal. An automatic classifier groups tool calls into exploration, editing, testing, building, and other actions. Table II summarizes the blind review; Section V reports further validation results.

TABLE II  
BLIND VALIDATION OF TRAJECTORY LABELS.
<table><tr><td>Label</td><td>Agree / n</td><td>%</td><td>95% CI</td></tr><tr><td>Process labels</td><td></td><td></td><td></td></tr><tr><td>Final check</td><td>175/180</td><td>97.2</td><td>93.7–98.8</td></tr><tr><td>Last test result</td><td>163/179</td><td>91.1</td><td>86.0-94.4</td></tr><tr><td>Criterion targeting</td><td>161/180</td><td>89.4</td><td>84.1-93.1</td></tr><tr><td>Stop reason</td><td>165/180</td><td>91.7</td><td>86.7-94.9</td></tr><tr><td>Dismissal evidence</td><td>41/44</td><td>93.2</td><td>81.8–97.7</td></tr><tr><td>Prior awareness</td><td>56/60</td><td>93.3</td><td>84.1-97.4</td></tr><tr><td>Diagnosis labels</td><td></td><td></td><td></td></tr><tr><td>Inheritance</td><td>90/90</td><td>100.0</td><td>95.9-100</td></tr><tr><td>F2P cause: fine</td><td>88/96</td><td>91.7</td><td>84.4-95.7</td></tr><tr><td>F2P cause: grouped</td><td>96/96</td><td>100.0</td><td>96.2-100</td></tr></table>

Specification checks. For RQ3, we inspect cases in which a green self-test conflicts with the hidden F2P result. Eligible stages begin at $m _ { 0 }$ or after a passed stage and have checks labeled as targeting every acceptance criterion. The first author examines the first failing expectation in a random sample, excluding chain runs used for blind label validation. Each expectation is classified by its visible basis: stated in a specification, implied by the specifications, implied by the starting repository, or unsupported by those materials.

## D. Metrics and Statistics

We adopt the following metrics; table abbreviations appear in parentheses.

• Single-stage pass rate $m _ { 0 } { : }$ the fraction of tasks whose first stage passes both F2P and P2P, starting from the initial state $S _ { 0 }$

• Chain-gate pass rate (Gate): the fraction of tasks whose chain gate passes on the final workspace.

• Continuation rate (Cont.): the stage pass rate at $m _ { 1 }$ and later, conditioned on all earlier stages of that chain having passed.

• Stage F2P rate (F2P): the fraction of evaluated stages whose F2P suite passes.

• Prefix depth (Depth): the number of consecutive passed stages before the first failure, divided by chain length.

• Survival at $m _ { k } { : }$ the fraction of tasks with more than k stages whose stages $m _ { 0 } . . m _ { k }$ all pass.

• Chain-level P2P failure (P2P↓): the fraction of tasks whose chain-level P2P suite fails.

• In-chain regression rate: among runs in which every stage passed F2P at completion, the fraction whose final workspace fails chain-level F2P.

• Pure regression rate: among runs whose final workspace passes chain-level F2P, the fraction that fail chain-level P2P.

For comparisons within the same task and stage, we split agents by the condition being compared and retain only taskstage groups containing both conditions. We compute each condition’s F2P pass rate within each group, then average these rates equally across groups. The agents in each group face the same requested stage but retain their own code and conversation histories.

Confidence intervals for agent performance are 95% bootstrap intervals [35] with 4,000 resamples over tasks in FEA-TUREARC.

## V. EXPERIMENTAL RESULTS

We report the results on FEATUREARC under the L4 evaluation process. Tables III and IV report percentages with best values bold; Table III is ordered by Gate. The analysis first examines whole-feature completion (RQ1), then the effect of earlier errors (RQ2), and finally agents’ ability to detect and verify those errors (RQ3).

Trajectory-label validation. Independent human annotations provide a check on the trajectory labels used to explain the evaluation outcomes. Table II reports the blind review of 180 stages, with 60 from each sampling group defined in Section IV-C. Agree/n counts matches with the model majority among applicable comparisons; intervals are 95% Wilson intervals [36]. Agreement is 89.4%–97.2% for process labels and 91.7%–100.0% for diagnosis labels, over their applicable comparisons. For the labels central to RQ2 and RQ3, the table reports 93.3% agreement on prior awareness and 100.0% on inheritance. Across the three model judges, Fleiss’ κ [37] is 0.94, 0.92, 0.91, 0.86, 0.84, and 0.80 for prior awareness, final check, stop reason, last test result, inheritance, and criterion targeting, respectively. The first author adjudicated 132 label items across 121 stages on which the judges all differed. The automatic tool classifier agrees with the trajectory labels on 94.8% of stages for which both identify a final red/green test result.

TABLE III  
PERFORMANCE ON FEATUREARC AT L4.
<table><tr><td rowspan="2">Model</td><td colspan="2">Chain</td><td colspan="3">Stage level</td><td rowspan="2">P2P↓</td></tr><tr><td>mo</td><td>Gate</td><td>Cont.</td><td>F2P</td><td>Depth</td></tr><tr><td>Claude Opus 5.5</td><td>83.6</td><td>49.3</td><td>79.2</td><td>68.4</td><td>64.9</td><td>5.5</td></tr><tr><td> GPT-6 Astra</td><td></td><td>67.1 37.0</td><td>76.1</td><td>55.5</td><td>50.4</td><td>20.5</td></tr><tr><td>Hunyuan 4 Preview</td><td>60.3</td><td>34.2</td><td>75.0</td><td>49.4</td><td>44.1</td><td>8.2</td></tr><tr><td>GLM-5.3</td><td>64.4 31.5</td><td></td><td>72.0</td><td>56.3</td><td>45.8</td><td>19.2</td></tr><tr><td>Ø Grok 4.7</td><td></td><td>60.3 26.0</td><td>70.9</td><td>50.6</td><td>40.9</td><td>11.0</td></tr><tr><td>Qwen 3.8</td><td></td><td>61.6 24.7</td><td>65.4</td><td>49.0</td><td>39.5</td><td>6.8</td></tr><tr><td>Gemini 3.8 Flash</td><td>61.6 23.3</td><td></td><td>66.2</td><td>49.4</td><td>42.0</td><td>5.5</td></tr><tr><td>Doubao 2.1 Pro</td><td>60.3 21.9</td><td></td><td>60.0</td><td>43.0</td><td>36.1</td><td>8.2</td></tr><tr><td>K&#x27; Kimi K3</td><td>56.2 19.2</td><td></td><td>64.5</td><td>42.2</td><td>35.7</td><td>15.1</td></tr><tr><td>DeepSeek V4 Pro</td><td>52.1</td><td>17.8</td><td>61.5</td><td>40.3</td><td>31.6</td><td>11.0</td></tr><tr><td>Pooled</td><td>62.7</td><td>28.5</td><td>70.1</td><td>50.4</td><td>43.1</td><td>11.1</td></tr></table>

## A. RQ1: Completing Evolving Features

To determine whether success on one request carries over to an evolving feature, we compare first-stage and chain-gate performance in Table III. Figures 5 and 6 then relate this gap to stage position and task length. Figure 5 pools agents, using the 57 and 34 tasks containing m<sub>2</sub> and m<sub>3</sub>, respectively, at those stages. Figure 6b compares first-stage (hollow) and chain-gate (filled) scores with task-bootstrap intervals; ∆ is their percentage-point difference.

Finding 1: The chain-gate pass rate falls to less than half of the first-stage pass rate. The first stage starts from $S _ { 0 }$ and is equivalent to a standard single-stage task. Pooled over agents, success drops from 62.7% at the first stage to 28.5% at the chain gate (Table III). Every agent loses 26.0– 38.4 percentage points (Figure 6b), and even Claude Opus 5.5, which leads every metric, fails more than half of the tasks. Rankings stay broadly similar (Spearman $\rho { = } 0 . 8 7$ between $m _ { 0 }$ and the chain gate), but first-stage scores overstate how often agents complete a feature.

Finding 2: Checking only the last stage overstates success. The last-stage gate checks the final extension using its stage-level F2P and P2P suites, whereas the chain gate checks the complete feature on the final workspace. Of the 273 runs that pass the last-stage gate, 69 (25.3%) fail the chain gate. The separate whole-feature check in Figure 2 detects this mismatch: passing the last extension does not establish that the complete feature works on the final code. Recovery is possible but uncommon: among the 525 runs with at least one failed stage, only 14 (2.7%) eventually pass the chain gate, the segment marked “Rec.” in Figure 6a. “Gate fails” marks runs passing every stage gate but failing the chain gate.

Finding 3: Most runs break in the first two stages, and long tasks rarely succeed. Figure 6a shows where runs that never recover first fail. In 36.2% of all runs, the agent already fails m<sub>0</sub>; in another 20.7%, it first fails m<sub>1</sub>, the first stage that builds on its own code. Only 13.2% of runs first fail at a later stage. The declining survival curves in Figure 5a show how these early failures reduce the share of chains that remain correct as requirements accumulate. Success also falls steeply with task length (Figure 5b): the chain gate passes in 44.4% of runs on two-stage tasks but in only 1.5% on tasks with five or more stages; only two runs pass, both by Grok 4.7. Completing these longer features remains difficult at L4.

Answer to RQ1: At L4, agents complete a whole evolving feature far less often than they pass its first stage: pooled success falls from 62.7% to 28.5%. Most runs break at m<sub>0</sub> or m<sub>1</sub>, runs with a failed stage rarely recover fully, and tasks with five or more stages are almost never completed.

## B. RQ2: The Effect of Earlier Errors

To explain failures beyond the first stage, we examine whether later stages build on correct behavior or inherit unresolved defects. Figure 7 compares next-stage success after passed and failed stages, and Table IV reports the trajectory evidence. The agents revisit earlier code to a similar extent as the reference implementations: 52.8%–57.7% of their laterstage changed files were changed earlier, consistent with the dependence shown in Figure 4d. This establishes the setting in which an earlier implementation error can affect a later request.

Finding 4: Inherited defects are common in failed later stages. Next-stage F2P success falls from 70.2% after a passed stage to 22.6% after a failed stage (Figure 7). The gap appears for every agent. Using the comparison within each task and stage defined in Section IV-D, we retain 109 task-stage groups containing both prior-stage outcomes. The gap remains when task-stage groups receive equal weight: the averaged withingroup F2P pass rates are 72.6% after a passed stage and 29.0% after a failed stage. Once a run fails a stage, on average 85.1% of its remaining stages fail as well. The inheritance column of Table IV connects this performance gap to concrete defects: 72.9% of failed later stages involve errors left in earlier stages, with those defects accounting for the failure wholly in 47.1% and partly in 25.9%.

(a) Tasks passing m0 to mk  
(b) Chain gate by length  
![](images/7ce6f04a31a3237cd58815cdf2897c5636bf9dc4fe24e5ee024c276f36f2c21a.jpg)  
Fig. 5. Performance across stages and task lengths.

TABLE IV  
ERROR INHERITANCE AND SELF-VERIFICATION ON FEATUREARC.
<table><tr><td rowspan="2">Model</td><td colspan="4">Self-test</td><td colspan="2">Trajectory labels</td></tr><tr><td>F2P after Green</td><td>Red</td><td>Ends green</td><td>Criterion targeting</td><td>Inheritance</td><td>Prior awareness</td></tr><tr><td> Claude Opus 5.5</td><td>68.6</td><td>68.1</td><td>70.3</td><td>76.4</td><td>73.8</td><td>95.2</td></tr><tr><td>GPT-6 Astra</td><td>54.5</td><td>46.4</td><td>81.0</td><td>82.9</td><td>74.8</td><td>95.6</td></tr><tr><td>Hunyuan 4 Preview</td><td>52.6</td><td>41.3</td><td>65.8</td><td>70.3</td><td>73.7</td><td>90.2</td></tr><tr><td>GLM-5.3</td><td>59.0</td><td>51.0</td><td>77.9</td><td>84.8</td><td>68.2</td><td>94.5</td></tr><tr><td>∅ Grok 4.7</td><td>57.1</td><td>38.9</td><td>70.0</td><td>62.7</td><td>77.6</td><td>99.0</td></tr><tr><td>Qwen 3.8</td><td>51.0</td><td>42.1</td><td>77.6</td><td>78.3</td><td>71.9</td><td>93.2</td></tr><tr><td>Gemini 3.8 Flash</td><td>49.6</td><td>20.0</td><td>97.3</td><td>77.2</td><td>71.9</td><td>97.1</td></tr><tr><td>0 Doubao 2.1 Pro</td><td>47.2</td><td>34.6</td><td>68.4</td><td>70.0</td><td>73.1</td><td>90.4</td></tr><tr><td>K&#x27; Kimi K3</td><td>44.9</td><td>48.0</td><td>67.7</td><td>50.6</td><td>69.7</td><td>98.2</td></tr><tr><td> DeepSeek V4 Pro</td><td>41.5</td><td>32.4</td><td>62.4</td><td>30.4</td><td>75.0</td><td>96.6</td></tr><tr><td>Pooled</td><td>52.7</td><td>45.2</td><td>73.8</td><td>68.4</td><td>72.9</td><td>94.9</td></tr></table>

We also examine how stage position and context relate to success in later stages. Among runs in which all earlier stages passed, the stage-gate pass rates at m<sub>1</sub>, m<sub>2</sub>, and $m _ { 3 + }$ range from 66.6% to 77.7%, with no downward trend. For later stages with available context measurements, the F2P gap between stages with and without detected context compaction is 6.2 percentage points for stages with fewer than 40 tool calls and 5.2 points for those with 40–99 tool calls. Among later stages with fully passed prefixes, F2P success is 75.0% when the cumulative specifications contain 1,500–2,999 words and 68.6% when they contain at least 8,000 words, a difference of 6.4 points. These gaps are smaller than the 47.6-point gap associated with the previous stage’s outcome.

Finding 5: Regression is uncommon after stages pass their tests. Regression, in which later edits break behavior that already worked, is the other way a chain can fail. We measure it in two ways. In-chain regression: every stage passed its own F2P suite, yet the final workspace fails the chain-level F2P suite; it occurs in 7.3% of the runs in which every stage passed F2P. Pure regression: the final workspace passes chain-level F2P but fails chain-level P2P; it occurs in 3.3% of the runs whose final workspace passes chain-level F2P. Inherited defects fail at least one later stage in 52.9% of feature runs, consistent with the inheritance evidence in Table IV. Runs that establish the required behavior usually retain it through the final check. The higher P2P failure rates in Table III (5.5%–20.5%) also include runs whose new feature behavior fails; they answer a broader question than pure regression.

Answer to RQ2: Earlier defects frequently affect later development: next-stage F2P success falls from 70.2% after a passed stage to 22.6% after a failed one, and trajectory analysis identifies inherited defects in 72.9% of failed later stages. Regression after successful implementation is less common under the conditional measures examined here.

## C. RQ3: Detecting and Verifying Errors

To understand why agents continue from defective code, we examine whether they notice earlier failures and whether their own checks provide reliable evidence of correctness. Table IV connects awareness and verification behavior to hidden-test outcomes; lower inheritance and missed-prior-failure rates are marked as better. Figure 8 pools later stages across agents, showing hidden F2P outcomes conditional on the final selftest and previous-stage results.

Finding 6: Agents rarely notice their own earlier errors. For stages following a failure, the prior-awareness label shows that the agent never notices the earlier error in 94.9% of cases (90.2%–99.0% per agent, Table IV); the blind review agrees with these labels on 56 of 60 sampled stages (Table II). Of the 50 stages in which the agent does notice the error, it repairs the error in 34. These repair cases show that correction is possible once an error is recognized; the much larger missed-failure group places the more frequent breakdown at detection.

Tool use changes little after a failed stage. For the nine agents on Claude Code, the median number of tool calls is 4%–18% higher after a failed stage than after a passed one, and the share of calls spent on testing barely changes (15.0% versus 15.3% for Claude Opus 5.5).

Finding 7: A green self-test barely predicts the hidden F2P result, and a red one is usually dismissed. Agents stop when their own tests pass, but a passing self-test says little about the hidden F2P suite. Overall, 73.8% of stages end on a green self-test, yet only 52.7% of these pass F2P, barely above the 50.4% rate over all stages; stages that end red still pass in 45.2% (Table IV). Green/Red in Table IV are conditional F2P rates; Ends green is the share ending green. Red estimates use fewer than 20 stages for Grok 4.7 and Gemini 3.8 Flash. Using the same comparison procedure, the mean within-group F2P pass rates are 52.9% for green final self-tests and 47.9% for red final self-tests across 161 task-stage groups containing both outcomes. As Figure 8 shows, the self-test is least reliable right after a failed stage: of the 713 stages that still end on a green self-test, only 24.5% pass F2P. The share of false greens, green self-tests whose stage fails F2P, rises from 28.7% after a passed stage to 75.5% after a failed one.

![](images/83e67befb44d395d809b4294a748c0be8875e7a1335e05280b70b96ebeeec52b.jpg)  
Fig. 6. Where tasks break.

![](images/7a4914af62feca1964c983c5ae4723280d4461167bd004f3dfb9d6cb10818828.jpg)  
Fig. 7. Next-stage F2P success by the previous stage’s outcome.

A red self-test, when it occurs, is usually dismissed. Of the 367 stages that end red, the agent states that the failure is unrelated or pre-existing in 321 (87.5%). Only 23 of these 321 (7.2%) first ran the same test on the stage’s starting code to confirm that it already failed there. The final self-test result is therefore a weak signal of correctness, and a red result often leads to dismissal.

Finding 8: Self-tests that target every criterion still miss what the specification asks. The criterion-targeting label marks 68.4% of stages, and 30.4%–84.8% per agent, as having checks that exercise all acceptance criteria (Table IV). Targeting all criteria offers limited assurance: these stages pass F2P in 54.9% of cases, against 42.1% when the checks exercise only some criteria. Using the same comparison procedure, the mean within-group F2P pass rates are 50.4% for all criteria and 38.5% for some criteria across 190 task-stage groups containing both categories. Of the stages that target all criteria, 45% still fail F2P, and 62% of these failures follow an earlier failed stage that the agent did not notice.

![](images/c5682cf05dbdd05e5b4847a06b0821e2bcc59012da88aae500e236907cc836d5.jpg)  
Fig. 8. Self-tests and hidden F2P outcomes.

The mismatch also appears at the first stage and after passed stages. Criterion coverage and the correctness of test expectations are distinct: a check can target the requested behavior while encoding an incorrect expected result. In 239 stages, the checks exercise all criteria, the stage begins at m<sub>0</sub> or after a passed stage, the self-test ends green, and F2P still fails; most are wrong logic or a misread specification (54.4%) or a missed case or partial implementation (24.3%). In the 60- stage specification check described in Section IV-C, 34 of the first failing expectations are stated explicitly in a current or earlier specification, 12 follow from these specifications, and 14 follow from the starting repository.

The same gap appears before any test runs. Of the 1,304 stages that fail F2P, 236 (18.1%) fail because the hidden suite does not compile; 228 of these are caused by the agent, and interface mismatches cause 61.4% of those 228: missing symbols or exports, wrong parameter or return types, and duplicate declarations. Of the missing names, 44.1% appear in the current stage’s specification and 30.5% in an earlier stage’s.

Together with the criterion-targeting and green-test columns of Table IV, these cases show why checking an agent’s interpretation of a requirement may still miss the behavior stated in the task. FEATUREARC makes this discrepancy observable through explicit contracts and separate acceptance checks.

Answer to RQ3: Agents leave earlier errors unnoticed in 94.9% of stages following a failure. Their self-tests provide limited assurance: only 52.7% of stages ending on a green self-test pass F2P, and this falls to 24.5% after a failed stage. Even checks labeled as targeting every criterion can miss specified behavior, while final red results are usually dismissed without a baseline check.

## VI. DISCUSSION

## A. Industry Application at Tencent

Connecting observed usage to internal evaluation. In an internal corpus of 8,881 IDE episodes at Tencent, 3,797 of the 8,728 episodes with valid user turns (43.5%) contained follow-up interaction. Model-based annotations of the first follow-up identified constraint additions (30.1%), requirement clarifications (19.7%), and functionality extensions (17.9%). The observed extensions motivate evaluating continued feature development. To reduce user checking between extensions, agents must detect defects in their own earlier code. Tencent’s existing L1–L3 evaluations supply failure feedback, leaving this unaided detection ability untested. We therefore use our pipeline and FEATUREARC to assess Hunyuan 4 Preview alongside external agents at L4: users supply successive requirements, while earlier failures are not reported to the agent.

Diagnosing development needs beyond the first-stage score. Table III shows that Hunyuan 4 Preview passes 60.3% of first stages but completes 34.2% of whole features (95% interval 23.3%–45.2%). Its 75.0% continuation rate indicates that it can often extend a feature when its earlier stages are correct. Figure 7 reveals a different situation after an error: next-stage F2P success falls from 71.6% after a passed stage to 22.5% after a failed one. The prior-awareness evidence in Table IV helps explain this gap: Hunyuan leaves 90.2% of earlier failures unnoticed. This is the lowest missed-failure rate among the evaluated agents; most earlier errors still persist unrecognized during subsequent feature development.

Turning evaluation evidence into development priorities. The pipeline links unsuccessful runs to specifications, code changes, and checks, helping the Hunyuan team distinguish failures to implement new behavior from failures to recognize inherited defects. For example, Hunyuan attributes a final failing self-test to unrelated or pre-existing causes in 44 of 46 stages, but checks that explanation against the starting code in only 5 of those 44 cases. This evidence gives the Hunyuan team concrete failure cases for improving checks of earlier requirements and verification of failure explanations. These capabilities are necessary for supporting continued feature development when users leave intermediate validation to the agent.

## B. Threats to Validity

FEATUREARC may not cover the full range of workflows encountered in real-world feature development. To broaden coverage, it includes four widely used programming languages—Python, TypeScript, Rust, and Go—and seven application domains, with tasks stratified by language and chain length (Figure 4). This breadth, together with tasks mined from repository histories, helps the benchmark reflect agents’ performance on evolving requirements across practical development settings.

## VII. RELATED WORK

Feature and interactive benchmarks. Single-request benchmarks evaluate feature implementation on fixed codebases [9]–[11]. Multi-stage feature [12]–[14], [16] and sequential-development benchmarks [38]–[43] extend evaluation across development stages, with different stage sources, feature scope, and feedback. EvoCode-Bench allows four attempts that stop at the first failure and compares with reference-code starts [15]. Interactive benchmarks add responsive users [44], [45], incremental requirements [46], [47], or clarification [48].

Task construction. Environment-synthesis methods prepare executable repositories [49]–[52], while automated pipelines collect or synthesize individual tasks [53]–[61]. DeepCom mit [38] and WEAVE [62] recover dependencies from repository histories; WEAVE uses them to award partial credit within a task.

Self-verification. Execution feedback supports selfrepair [63], [64], but its effectiveness depends on feedback quality [65]. Weak tests can overestimate correctness [66], and generating checks that expose defects remains difficult [67].

FEATUREARC differs by combining historical same-feature dependencies with continued evaluation on the agent’s own code without intermediate failure feedback (Table I). Validated stage and whole-feature checks measure correctness across extensions, while trajectory analysis examines the persistence and detection of inherited errors.

## VIII. CONCLUSION

We presented FEATUREARC together with a pipeline that constructs its executable feature-evolution tasks from repository histories. The benchmark combines historical samefeature dependencies, requirements-grounded verification, and diverse repository contexts to evaluate agents that extend their own code without external acceptance feedback. Our study reveals a substantial gap between first-stage implementation and whole-feature completion: pooled success falls from 62.7% to 28.5%. Inherited defects are common in failed later stages, and agents rarely notice these errors or expose them through self-tests. At Tencent, the pipeline extends internal evaluation to L4 and provides evidence about Hunyuan’s ability to recognize and recover from errors during feature development. FEATUREARC supports further study of sustained feature correctness as requirements evolve, using dependent stages and executable checks.

[1] M. M. Lehman, “Programs, life cycles, and laws of software evolution,” Proceedings of the IEEE, vol. 68, no. 9, pp. 1060–1076, 1980.

[2] V. T. Rajlich and K. H. Bennett, “A staged model for the software life cycle,” Computer, vol. 33, no. 7, pp. 66–71, 2000.

[3] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press, “SWE-agent: Agent-computer interfaces enable automated software engineering,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 50 528–50 652.

[4] X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, H. H. Tran, F. Li, R. Ma, M. Zheng, B. Qian, Y. Shao, N. Muennighoff, Y. Zhang, B. Hui, J. Lin, R. Brennan, H. Peng, H. Ji, and G. Neubig, “OpenHands: An open platform for AI software developers as generalist agents,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=OJd3ayDDoF

[5] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan, “SWE-bench: Can language models resolve realworld GitHub issues?” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https: //openreview.net/forum?id=VTF8yNQM66

[6] N. Chowdhury, J. Aung, C. J. Shern, O. Jaffe, D. Sherburn, G. Starace, E. Mays, R. Dias, M. Aljubeh, M. Glaese, C. E. Jimenez, J. Yang, L. Ho, T. Patwardhan, K. Liu, and A. Madry, “Introducing SWE-bench Verified,” https://openai.com/index/introducing-swe-bench-verified/, 2024, updated 24 February 2025.

[7] SWE-bench, “SWE-bench leaderboards,” https://www.swebench.com/, 2026, accessed 22 October 2026.

[8] J. Baumann, V. Padmakumar, K. Li, J. Yang, D. Yang, and S. Koyejo, “SWE-chat: Coding agent interactions from real users in the wild,” in Proceedings of the Third Conference on Language Modeling, 2026. [Online]. Available: https://proceedings.colm.cc/paper files/paper/2026/ hash/e5b4633454cb2174779d294ccda02318-Abstract-Conference.html

[9] W. Li, X. Zhang, Z. Guo, S. Mao, W. Luo, G. Peng, Y. Huang, H. Wang, and S. Li, “FEA-Bench: A benchmark for evaluating repository-level code generation for feature implementation,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 17 160–17 176.

[10] H. Chen, C. Li, and J. Li, “FeatBench: Towards more realistic evaluation of feature-level code generation,” 2025, version 2. [Online]. Available: https://arxiv.org/abs/2509.22237v2

[11] Q. Zhou, J. Zhang, H. Wang, R. Hao, J. Wang, M. Han, Y. Yang, S. Wu, F. Pan, L. Fan, D. Tu, and Z. Zhang, “FeatureBench: Benchmarking agentic coding for complex feature development,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://arxiv.org/abs/2602.10975

[12] CodeCrafters, “CCBench: A benchmark for evaluating AI coding agents on real-world tasks within small-scale codebases,” https://github.com/ codecrafters-io/ccbench, 2026, results updated 11 February 2026.

[13] K. N. A. Shastry, G. Senrayan, S. Satapara, P. Panda, and C. Devaguptapu, “Beyond isolated tasks: A framework for evaluating coding agents on sequential software evolution,” 2026, presented at the ICML 2026 Workshop on Deep Learning for Code (DL4C). [Online]. Available: https://arxiv.org/abs/2604.03035

[14] G. Orlanski, D. Roy, A. Yun, C. Shin, A. Gu, A. Ge, D. Adila, N. Roberts, F. Sala, and A. Albarghouthi, “SlopCodeBench: Benchmarking how coding agents degrade over long-horizon iterative tasks,” 2026, version 2. [Online]. Available: https://arxiv.org/abs/2603.24755v2

[15] H. Shen, X. Chen, W. Xu, Y. Ma, L. Chen, and K. Li, “EvoCode-Bench: Evaluating coding agents in multi-turn iterative interactions,” 2026. [Online]. Available: https://arxiv.org/abs/2605.24110

[16] D. Nukrai, B. Alon, N. Levy, I. A. Atad, I. Cohen, I. Zimerman, L. Wolf, E. David, and I. Naiman, “EvoWeb: A benchmark for coding agents on evolving web applications,” 2026, version 1. [Online]. Available: https://arxiv.org/abs/2610.16402v1

[17] SAE International, “Taxonomy and definitions for terms related to driving automation systems for on-road motor vehicles,” SAE International, Standard J3016 202104, 2021.

[18] K. J. K. Feng, D. W. McDonald, and A. X. Zhang, “Levels of autonomy for AI agents,” 2025, version 2. [Online]. Available: https://arxiv.org/abs/2506.12469v2

[19] S. Yoo and M. Harman, “Regression testing minimization, selection and prioritization: A survey,” Software Testing, Verification and Reliability, vol. 22, no. 2, pp. 67–120, 2012.

[20] Anthropic, “Claude Opus 5.5,” 2026, accessed: 2026-10-24. [Online]. Available: https://www.anthropic.com/claude-opus-5-5

[21] Tencent, “Tencent Hunyuan: Product Overview,” 2026, accessed: 2026-10-24. [Online]. Available: https://cloud.tencent.com/document/ product/1729/104753

[22] T. Zimmermann, P. Weißgerber, S. Diehl, and A. Zeller, “Mining version histories to guide software changes,” IEEE Transactions on Software Engineering, vol. 31, no. 6, pp. 429–445, 2005.

[23] OpenAI, “GPT-6 Astra: A New Generation of Intelligence,” 2026, accessed: 2026-10-24. [Online]. Available: https://openai.com/index/ gpt-6-astra/

[24] Z.ai, “GLM-5.3 and GLM-5.3-Flash,” 2026, accessed: 2026-10-24. [Online]. Available: https://github.com/zai-org/GLM-5

[25] xAI, “Introducing Grok 4.7,” 2026, accessed: 2026-10-24. [Online]. Available: https://x.ai/news/grok-4-7

[26] Qwen Team, “Qwen3.8-Max: A New Bar for Coding and Cowork,” 2026, accessed: 2026-10-24. [Online]. Available: https://qwen.ai/blog? id=qwen3.8

[27] Google DeepMind, “Introducing Gemini 3.8 Flash and 3.8 Flash Cyber,” 2026, accessed: 2026-10-24. [Online]. Available: https://blog.google/innovation-and-ai/models-and-research/ gemini-models/3-8-flash-and-3-8-flash-cyber/

[28] ByteDance Seed, “Seed2.1: A Next-Generation Agent for Real-World Productivity,” 2026, accessed: 2026-10-24. [Online]. Available: https://seed.bytedance.com/en/seed2 1

[29] Moonshot AI, “Kimi K3: Open Frontier Intelligence,” 2026, accessed: 2026-10-24. [Online]. Available: https://www.kimi.com/en/blog/kimi-k3

[30] DeepSeek-AI, “DeepSeek-V4-Pro GA Release,” 2026, accessed: 2026-10-24. [Online]. Available: https://api-docs.deepseek.com/news/ news260813/

[31] OpenAI, “Introducing GPT-5.5,” 2026, accessed: 2026-10-24. [Online]. Available: https://openai.com/index/introducing-gpt-5-5/

[32] DeepSeek-AI, “DeepSeek-V4.1-Flash: Smarter, Faster, More Efficient,” 2026, accessed: 2026-10-24. [Online]. Available: https://api-docs. deepseek.com/news/news260910/

[33] Anthropic, “Claude Sonnet 5,” 2026, accessed: 2026-10- 24. [Online]. Available: https://platform.claude.com/docs/en/models/ sonnet-5/overview

[34] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, vol. 36, 2023, pp. 46 595–46 623.

[35] B. Efron, “Bootstrap methods: Another look at the jackknife,” The Annals of Statistics, vol. 7, no. 1, pp. 1–26, 1979.

[36] E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” Journal of the American Statistical Association, vol. 22, no. 158, pp. 209–212, 1927.

[37] J. L. Fleiss, “Measuring nominal scale agreement among many raters,” Psychological Bulletin, vol. 76, no. 5, pp. 378–382, 1971.

[38] G. Deng, Z. Chen, Z. Yu, H. Fan, Y. Liu, Y. Yang, D. Parikh, R. Kannan, L. Cong, M. Wang, Q. Zhang, V. Prasanna, X. Tang, and X. Wang, “SWE-Milestone: Evaluating AI agents on continuous software evolution,” in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 306, 2026, pp. 23 853–23 890. [Online]. Available: https://proceedings.mlr.press/v306/deng26d.html

[39] M. H. Lam, C. Wang, H. Liu, J. Xiao, H. Li, J. Huang, T. Y. Zhuo, and M. R. Lyu, “SWE-Chain: Benchmarking coding agents on chained release-level package upgrades,” 2026. [Online]. Available: https://arxiv.org/abs/2605.14415

[40] Q. Jin, L. Tung, K. Li, Q. Shi, Y. She, H. Jia, H. Zhao, K. Xia, Z. Du, J. Pei, Z. Zhang, Z. Qi, Y. Duan, W. Lee, and Z. Jin, “ChainSWE: Benchmarking coding agents on multi-bug software maintenance,” in Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, 2026. [Online]. Available: https://openreview.net/forum?id=XJaANR7Pw8

[41] T. Le, M. V. T. Thai, D. N. Manh, H. P. Nhat, and N. D. Q. Bui, “SWE-EVO: Benchmarking coding agents in long-horizon software evolution scenarios,” 2026. [Online]. Available: https://arxiv.org/abs/2512.18470

[42] X. Xu, R. Yang, H. Shen, W. Xu, B. Gao, R. Wu, K. Shi, W. Xie, X. Chen, M. Wu, J. Zeng, M. Heinrich, E. Zhang, L. Chen, K. Li, and B. Chang, “RoadmapBench: Evaluating long-horizon agentic software development across version upgrades,” 2026, version 2. [Online]. Available: https://arxiv.org/abs/2605.15846v2

[43] H. Li, Z. Fang, R. Feng, Y. Zhao, J. Liu, P. Gao, H. Ye, D. Lin, Q. Lin, S. Rajmohan, and D. Zhang, “LoopsBench: From harness engineering to loop engineering in coding agent evaluation,” 2026, version 2. [Online]. Available: https://arxiv.org/abs/2608.00267v2

[44] M. Raghavendra, A. Gunjal, A. Sabharwal, and Y. He, “SWE-Interact: Reimagining SWE benchmarks as user-driven long-horizon coding sessions,” 2026, version 1. [Online]. Available: https://arxiv.org/abs/ 2606.30573v1

[45] Y. Wu, Z. Zhao, S. Li, H. H. Lee, J. Zhu, S. Wu, T. Yu, S. Li, L. Zhang, X. Fan, and S. Li, “SWE-Together: Evaluating coding agents in interactive user sessions,” 2026, version 1. [Online]. Available: https://arxiv.org/abs/2606.29957v1

[46] Z. Zhan, S. Gao, R. Hu, and C. Gao, “SR-Eval: Evaluating LLMs on code generation under stepwise requirement refinement,” 2026. [Online]. Available: https://arxiv.org/abs/2509.18808

[47] L. Yan, X. Chen, and X. Zhang, “When the specification emerges: Benchmarking faithfulness loss in long-horizon coding agents,” 2026. [Online]. Available: https://arxiv.org/abs/2603.17104

[48] S. Vijayvargiya, X. Zhou, A. Yerukola, M. Sap, and G. Neubig, “Ambig-SWE: Interactive agents to overcome underspecificity in software engineering,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=X2yzXtH4wp

[49] R. Hu, C. Peng, X. Wang, J. Xu, and C. Gao, “Repo2Run: Automated building executable environment for code repository at scale,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 36 770–36 809.

[50] I. Bouzenia and M. Pradel, “You name it, I run it: An LLM agent to execute tests of arbitrary projects,” Proceedings of the ACM on Software Engineering, vol. 2, no. ISSTA, pp. 1054–1076, 2025.

[51] C. Guo, J. Wu, S. He, Y. Chen, Z. Kuang, S. Fan, B. Chen, S. Bao, J. Liu, H. Wu, Q. Zhu, W. Che, and H. Wang, “MEnvAgent: Scalable polyglot environment construction for verifiable software engineering,” in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 306, 2026, pp. 38 471–38 495. [Online]. Available: https://proceedings.mlr.press/v306/guo26aa.html

[52] A. Eliseeva, A. Kovrigin, I. Kholkin, E. Bogomolov, and Y. Zharov, “EnvBench: A benchmark for automated environment setup,” Deep Learning for Code Workshop at ICLR 2025, https://github.com/ JetBrains-Research/EnvBench, 2025.

[53] I. Badertdinov, A. Golubev, M. Nekrashevich, A. Shevtsov, S. Karasik, A. Andriushchenko, M. Trofimova, D. Litvintseva, and B. Yangel, “SWE-rebench: An automated pipeline for task collection and decontaminated evaluation of software engineering agents,” in Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, vol. 38, 2025, pp. 26 420–26 466.

[54] I. Badertdinov, M. Nekrashevich, A. Shevtsov, and A. Golubev, “SWE-rebench V2: Language-agnostic SWE task collection at scale,” in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 306, 2026, pp. 4820–4843. [Online]. Available: https://proceedings.mlr.press/ v306/badertdinov26a.html

[55] L. Zhang, S. He, C. Zhang, Y. Kang, B. Li, C. Xie, J. Wang, M. Wang, Y. Huang, S. Fu, E. Nallipogu, Q. Lin, Y. Dang, S. Rajmohan, and D. Zhang, “SWE-bench goes live!” in Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, vol. 38, 2025, pp. 163 347–163 372.

[56] L. Wang, L. Ramalho, A. Celestino, P. A. Pham, Y. Liu, U. K. Sinha, A. Portillo, O. Osunwa, and G. Maduekwe, “SWE-Bench++: A framework for the scalable generation of software engineering benchmarks from open-source repositories,” 2025. [Online]. Available: https://arxiv.org/abs/2512.17419

[57] L. Guo, Y. Wang, C. Li, W. Tao, P. Yang, J. Chen, H. Song, D. Tang, and Z. Zheng, “SWE data construction, automatically!” Proceedings of the ACM on Software Engineering, vol. 3, no. FSE, pp. 525–546, 2026.

[58] J. Yang, K. Lieret, C. E. Jimenez, A. Wettig, K. Khandpur, Y. Zhang, B. Hui, O. Press, L. Schmidt, and D. Yang, “SWE-smith: Scaling data for software engineering agents,” in Advances in Neural Information

Processing Systems, Datasets and Benchmarks Track, vol. 38, 2025, pp. 107 402–107 452.

[59] N. Jain, J. Singh, M. Shetty, T. Zhang, L. Zheng, K. Sen, and I. Stoica, “R2E-Gym: Procedural environment generation and hybrid verifiers for scaling open-weights SWE agents,” in Second Conference on Language Modeling, 2025. [Online]. Available: https://openreview.net/forum?id=7evvwwdo3z

[60] J. Pan, X. Wang, G. Neubig, N. Jaitly, H. Ji, A. Suhr, and Y. Zhang, “Training software engineering agents and verifiers with SWE-Gym,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267, 2025, pp. 47 717–47 737. [Online]. Available: https://proceedings.mlr.press/v267/pan25g.html

[61] H. Qi, X. Wang, X. Gao, B. Sang, X. Zhang, M. Ma, P. Gao, Y. Kang, Q. Lin, S. Rajmohan, D. Zhang, and Q. Zhang, “Change2Task: From repository changes to executable coding agent tasks and environments,” 2026. [Online]. Available: https://arxiv.org/abs/2607.28591

[62] W. Li, Z. Xu, Z. Wang, S. Yu, W. Zhu, Y. Luo, J. Xu, Y. Wu, H. Huang, and A. Deng, “WEAVE: Weaving real-world code evolution into dependency-structured tasks for benchmarking and training code agents,” 2026, version 3. [Online]. Available: https://arxiv.org/abs/2610.22039v3

[63] X. Chen, M. Lin, N. Scharli, and D. Zhou, “Teaching large¨ language models to self-debug,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=KuPixIqPiq

[64] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: Language agents with verbal reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 8634–8652.

[65] T. X. Olausson, J. P. Inala, C. Wang, J. Gao, and A. Solar-Lezama, “Is self-repair a silver bullet for code generation?” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=y0GJXRungR

[66] J. Liu, C. S. Xia, Y. Wang, and L. Zhang, “Is your code generated by ChatGPT really correct? Rigorous evaluation of large language models for code generation,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 21 558–21 572.

[67] N. Mundler, M. N. M¨ uller, J. He, and M. Vechev, “SWT-Bench: Testing¨ and validating real-world bug-fixes with code agents,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 81 857– 81 887.