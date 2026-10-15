![](images/61c50349dbce787ab72ec2e020b780e321050ed2ac509d2df2d7f55fe3da85cb.jpg)

# Doc 2 Harness: Benchmarking LLMs on Long-Horizon Harness Engineering of Coding Agents

Yuyao Ge<sup>1</sup>, Shenghua Liu<sup>1,†</sup>, Peizhong Gao<sup>2</sup>, Chendi Ge<sup>2</sup>, Chenxu Liu<sup>3</sup>, Yiwei Wang<sup>4</sup>, Lingrui Mei<sup>1</sup>, Baolong Bi<sup>1</sup>, Jiayu Yao<sup>1</sup>, Jiafeng Guo<sup>1</sup>, Xueqi Cheng<sup>1</sup>

<sup>1</sup>Institute of Computing Technology, Chinese Academy of Sciences <sup>2</sup>Tsinghua University <sup>3</sup>Peking University <sup>4</sup>Southeast University

<sup></sup> geyuyao.com/doc2harness

![](images/8173bf59b12ffa1db758a415f144443d5ff30a9633f727c1e437c6cd9d1764a2.jpg)  
Figure 1: Pass rate of the nineteen models on the Doc2Harness benchmark, ordered from the strongest model on the left to the weakest on the right. Each model is scored on three independently built candidate harnesses, and each bar gives their mean.

## Abstract

Agents driven by large language models are increasingly deployed on longhorizon tasks, where they use tools and modify real workspaces. How well they perform depends as much on the harness around the model as on the model itself. Existing benchmarks measure downstream task success under a fixed harness, leaving harness engineering itself unexplored. We propose Doc2Harness, a two-stage benchmark that evaluates long-horizon harness engineering directly. In Stage 1, a build model operating inside a fixed builder harness receives an interface-only contract skeleton and a development document that consolidates behavior across twelve frontier harnesses, and then constructs a complete candidate harness in an isolated container. In Stage 2, the frozen candidate harness runs in containers separate from the construction environment, one per test case, and is driven end to end by a scripted mock model. We score each test case using four observable channels, namely the requests sent to the model, the files on disk, the process exit code, and the command output; positive controls rule out vacuous passes. Across nineteen frontier models, each evaluated with three independent builds on 328 test cases, GPT-6 Astra achieves the highest pass rate at 53.05%, while the weakest model reaches 27.44%, a gap of roughly twenty-six percentage points. Even the strongest model fails nearly half of the test cases, and failures cluster around how cross-cutting functionality is split across modules. To our knowledge, Doc2Harness provides the first deterministic benchmark for measuring long-horizon harness engineering.

## 1 Introduction

As coding agents driven by large language models (LLMs) become more capable, they are deployed in increasingly complex settings (Ge et al., 2025). The harness therefore matters more: it manages the interaction loop, tool execution, context, permissions, session state, failure recovery, and result verification, and so determines how much of the model’s capability reaches the environment (Mei et al., 2025). At the same time, human supervision of every agent step has become difficult. This has motivated recursive self-improvement, in which an agent modifies the infrastructure that governs its own behavior. An agent executes a task, observes the outcome, modifies its own harness, and then uses the modified harness on the next task, improving itself over time. A model’s ability to generalize this process may ultimately determine the capability of the resulting agent.

Building a harness is a type of software engineering (SWE) task, but it tests a kind of generalization that ordinary SWE tasks do not capture. At the data-source level, the opensource repositories used by SWE benchmarks are mostly mature projects that may have appeared repeatedly in training corpora. Harnesses, by contrast, have appeared only in the past few years and are updated at high frequency, so the resulting evaluation places greater emphasis on generalization. On the evaluation side, existing SWE benchmarks such as SWE-bench Pro (Deng et al., 2026), NL2Repo-Bench (Ding et al., 2026), and Commit0 (Zhao et al., 2025) judge whether the model’s output is acceptable using the interactive unit tests supplied with the task. Such criteria do not transfer directly to harnesses, for two reasons. (1) A single feature can be realized in many ways inside a harness, whereas an ordinary unit test observes behavior through a fixed interface. (2) A harness does not complete a downstream task on its own, so its correctness must be assessed by driving it with a model.

We therefore propose Doc2Harness, a two-stage benchmark for evaluating how well LLMs develop harnesses. In Stage 1, the build model under evaluation operates inside a fixed builder harness and, given a development document and a contract skeleton that declares interfaces only, constructs a complete candidate harness inside an isolated container. In Stage 2, the candidate harness is moved into isolated containers, one per test case, launched and executed end to end as a black box, with a mock model providing responses according to a preset script and each test case running the harness through one complete session. We observe four channels for each test case, namely the requests sent to the model, the files on disk, the process exit code, and the command output, and aggregate the resulting statuses into a pass rate. The build model’s ability to develop a harness is reflected indirectly in the pass rate.

Both inputs are designed to reduce reliance on parametric knowledge that the build model may already possess: the development document is rewritten from the user documentation of twelve frontier harnesses after merging and disambiguation, and the contract skeleton is distilled further from mini-swe-agent (Yang et al., 2024) with all internal implementation removed. For Stage 2, we design several hundred test cases covering a wide range of scenarios. A test case may pass even though the feature under test never ran, a situation we call a vacuous pass; to guard against it we introduce positive controls that verify the observation preconditions. If a positive control fails, the observation precondition is not satisfied, and the test case is recorded as FAIL regardless of its own status.

As shown in Figure 1, we evaluate nineteen frontier models on Doc2Harness, scoring three candidate harnesses per model and ranking them by pass rate. The strongest, GPT-6 Astra, reaches a pass rate of 53.05%, while the weakest reaches 27.44%; this roughly twenty-sixpoint gap shows that the suite is neither saturated nor unable to discriminate among the models in this cohort. Across categories, a model’s rank varies by as many as sixteen positions.

Our contributions are as follows.

• We propose Doc2Harness, which runs the build model under evaluation inside a fixed builder harness, gives it a development document and a contract skeleton that declares interfaces only, asks it to construct a complete candidate harness inside an isolated container, and then evaluates the result with several hundred test cases.

![](images/5b834dc3afbf3c4ec940cd0e4612c6da7014f3b0109dd5c4386d57096fbc0ac2.jpg)  
Stage 2

![](images/671548246c89422065ae024ea124f7b5e0b73effb392d4e9e2f5cb487e230873.jpg)  
Figure 2: Overview of Doc2Harness. Top (Stage 1): the builder, formed from the build model under evaluation and a fixed builder harness (OpenCode), takes the development document and the contract skeleton as input and emits a submission; the contract check verifies form rather than behavior, freezing the deliverable as the candidate harness on success and returning it to the builder on failure. Bottom (Stage 2): the frozen candidate harness and a mock model form the driven harness, which runs the scenario of each test case; effects are read from four channels, namely the requests sent to the model, the files on disk, the process exit code, and the command output, and the positive control gates the status before the assertion decides PASS or FAIL.

• We provide a test suite that exercises a harness as a black box: the candidate harness is driven by a mock model following a preset script, and positive controls help rule out vacuous passes, keeping the evaluation accurate.

• We evaluate nineteen frontier models and parse the source of every scored candidate harness. Higher-scoring candidate harnesses tend to place cross-cutting functionality in fewer modules. A model’s rank also moves by up to sixteen positions across the six main categories.

## 2 Doc2Harness

## 2.1 Task Description

Doc2Harness has two stages, shown in Figure 2.

Stage 1, Construction. The builder combines the build model under evaluation with a fixed builder harness. It reads a development document and, starting from a contract skeleton, constructs a candidate harness. The development document prescribes the form and functionality the candidate harness should possess; the contract skeleton is a framework, simplified and adapted from mini-swe-agent (Yang et al., 2024), that retains only the interfaces and contains no implementation. The builder determines when Stage 1 ends by issuing a submission, without which the build does not count as delivered. At each submission, the contract check verifies that the deliverable compiles and installs, that all required commands are present, that no placeholder stubs remain, and that the three contract interfaces are in place, without running any test case. If the check passes, the build ends and the deliverable is frozen as the candidate harness, never to be modified again; if it fails, the deliverable is returned, and the builder may keep revising and submit again within its quota (§3). Each scored candidate harness represents the state the model submitted as finished, rather than a state reached only when the time limit expired.

Stage 2, Evaluation. Inside containers isolated from the construction environment, one per test case, a mock model drives the candidate harness, and we determine the status of each test case in turn. Some test cases are additionally equipped with positive controls, which test whether an apparent pass reflects the required behavior rather than a failure of the observation mechanism (§2.4). A test case is recorded as PASS only when it passes on its own and every positive control gating it also passes. The build model’s ability to develop a harness is reflected indirectly through the pass rate over test cases. We therefore rank build models rather than candidate harnesses.

Each stage holds one component fixed to reduce confounding. In Stage 1, the builder harness is uniformly the open-source coding agent framework OpenCode (OpenCode Contributors, 2025) rather than harnesses such as Claude Code (Anthropic, 2025) or Codex CLI (OpenAI, 2025), because such harnesses are developed by model vendors alongside their own models and may confer an advantage on the Claude and GPT model families. In Stage 2, the mock model supplies responses according to a preset script, which avoids interference from the instability that a real LLM introduces.

## 2.2 Task Inputs

## 2.2.1 Development Document

The development document is written in three steps, namely collecting a feature inventory, turning that inventory into clauses, and disambiguating so that the clauses are self-consistent. The clause is the document’s basic unit. Most clauses specify the external behavior of one feature, while the consistency rules specify relations across features and chapters.

Collecting a feature inventory. In the first phase we crawl the user documentation of twelve frontier harnesses (Claude Code (Anthropic, 2025), Kimi Code (Moonshot AI, 2026), Gemini CLI (Google, 2025), Qwen Code (Qwen Team, 2025), OpenCode (OpenCode Contributors, 2025), OpenClaw (OpenClaw Foundation, 2025), Codex CLI (OpenAI, 2025), Aider (Gauthier, 2023), OpenHands (Wang et al., 2025), Cline (Cline Bot Inc., 2024), Crush (Charm, 2025), and Cursor (Anysphere, 2023)) and extract candidate features from it. The second phase examines each candidate feature from the first phase in greater detail, filling in invocation form, parameter semantics, and boundary behavior. The outputs of the two phases are merged, deduplicated, and pruned into one inventory of frontier-harness capabilities.

Turning the inventory into clauses. As shown in Table 1, the development document covers the deliverable and runtime interface, built-in tools, top-level commands and subcommands, permission and role modes, global flags, in-session commands and shortcuts, persistent memory, convention files, consistency rules, exit codes, and the API for driving automated evaluation. Each clause is given a syntax, a meaning, and an expected behavior. Confining each clause in this way serves two purposes, (1) verifiability from external behavior and (2) independence from implementation. To make the first purpose operational, we require the effect of every feature to be observable through one of four channels, namely the requests sent to the model, the files on disk, the process exit code, and the command output. To serve the second purpose, apart from the three contracts of the contract skeleton we do not constrain the manner of implementation.

Disambiguating features. The clauses from the previous step come from twelve sources, and to avoid semantic conflicts we use the following two disambiguation rules. (1) We merge commands with overlapping semantics. For example, Claude Code names context compaction /compact whereas Cursor names it /summarize, and the development document adopts /compact. (2) We split distinct concepts that occupy the same name. Take the word plan: in Claude Code it is one of six permission modes and means read-only, whereas in Aider, proposing changes and carrying out edits are handled by two model roles, architect and editor, and Aider’s documentation instead specifies the responsibilities assigned to each role. The development document accordingly assigns the read-only constraint to permissions and responsibility for proposing and applying changes to roles, so that permissions and roles each form their own clause.

Table 1: Chapter structure of the development document.
<table><tr><td>Content</td><td>Principal clause types</td></tr><tr><td>Deliverable and runtime interface</td><td>File structure, entry script</td></tr><tr><td>Built-in tools</td><td>Invocation form and boundary behavior of tools</td></tr><tr><td>Top-level commands and subcommands</td><td>Command syntax and exit codes</td></tr><tr><td>Permission modes, role modes, sandbox and lists</td><td>Five permission modes, five role modes, deny rules</td></tr><tr><td>Global flags</td><td>Semantics of flags, and their effect on requests</td></tr><tr><td>In-session commands and shortcuts</td><td>Command dispatch and terminal interaction</td></tr><tr><td>Persistent memory</td><td>Writing, retrieval, and clearing of memory</td></tr><tr><td>Convention files</td><td>Discovery order and injection method of files</td></tr><tr><td>Consistency rules</td><td>Relational constraints across chapters</td></tr><tr><td>Exit codes</td><td>A closed set of values</td></tr></table>

## 2.2.2 Contract Skeleton

The form of the contract skeleton determines what proportion of the candidate harness the build model writes. To maximize this proportion while keeping the candidate harness observable, we adopt a stripped-down mini-swe-agent (Yang et al., 2024) as the skeleton.

Composition. The contract skeleton consists of five Python source files (roughly seventy lines in total), plus one install script, one entry script, and one project config. Three of its interfaces constitute an immutable external contract, namely (1) the CLI, (2) the headless entry point, and (3) the API for driving automated evaluation. A representative candidate harness runs to approximately 30 Python source files and approximately 16,000 lines, of which roughly 99.6% is written by the build model beyond what the contract skeleton supplies.

Construction environment. Every build takes place in a newly created empty container, into which the contract skeleton and the development document are copied and which has no external network access other than the designated model gateway. Before construction begins we run a preflight check confirming that the container holds no unexpected files and that OpenCode starts normally; failure of either condition aborts the build and produces no result.

## 2.3 Test Case Synthesis

At design time, a test case must satisfy two requirements at once: it must be anchored to some clause of the development document, and it must discriminate between implementations; whether a retained case actually discriminates is settled only after evaluation. As shown in Figure 3, we use three methods in a yield-driven cycle, the first two broadening the coverage of the test cases and the third deepening what they inspect. §2.3.1 draws on the public records of harness users, §2.3.2 on the source of mature harnesses, and §2.3.3 on the existing test suite itself. We move to the next method when the current method’s marginal yield declines. Because the methods draw on different sources, they also use different filtering criteria.

## 2.3.1 Public Issue Mining

This method rests on an assumption about the distribution of defects. LLMs are trained on code written by human developers. Mature harnesses and the harnesses written by build models face the same interface semantics and the same class of boundary conditions. We therefore expect substantial overlap between the defects mature harnesses have exposed over years of evolution and the mistakes made by the harnesses written by build models, which makes public issue records a ready source of candidates.

![](images/6ff698f348cdc0694dad10c96d8cd40545b6e8aa97ed4e35da1aa44bd65a15ed.jpg)  
Figure 3: The three test case synthesis methods.

Collection. We pull every issue from the public repositories of four frontier harnesses (Claude Code, OpenCode, Gemini CLI, and Kimi Code); after filtering out pull requests these number 9,868, 5,438, 4,831, and 946 respectively, 21,083 in total.

Ranking and stopping. More than 20,000 records cannot be reviewed manually in full, so we first rank them by relevance to the development document. Manual review starts at the top of the ranking and stops when the marginal yield falls below the cost of further review.

Clustering. The same defect is often reported repeatedly, worded differently yet all pointing at one feature. We cluster the records we have read by the phenomenon they report, forming one candidate test case per cluster, into whose material the records of that cluster are then merged.

Filtering. Distinct phenomena may still land on the same point of the implementation. Where two clusters anchor to the same clause and the effects they observe are too close, the test cases realized from them would pass and fail together on the same set of implementations, so only the one with more complete material is kept. A remaining candidate is retained only if it can additionally be turned into a deterministic assertion and is not yet covered by the existing test suite.

This method’s net yield is high at the outset and falls as the same sources are mined further, on which basis we turn to the next method.

## 2.3.2 Cross-Harness Comparison

The previous method is limited by the quantity and quality of external records; this method broadens coverage from the source side instead. We compare one candidate harness against the public source of a mature harness and examine which capabilities that candidate harness lacks. We call the mature implementation that serves as the reference the comparison target; examples include Kimi Code and Qwen Code.

Partition. A candidate harness’s source is partitioned into modules according to the features the development document lists, each module corresponding to one group of features in the document.

Comparison. The same partition is applied to the source of the comparison target, which places the two on a common set of functional boundaries, and we enumerate their implementation differences module by module, covering both the features the comparison target handles and the candidate harness does not, and the boundaries on which the two behave differently.

Attribution. A difference does not necessarily stem from a defect; it may merely reflect a different manner of implementation. Every difference is traced back to the development document, and only those derivable from the purpose of some clause count as candidate test cases.

The coverage of this method is capped by the functional surface of the comparison target: once the candidate test cases obtainable from comparison are exhausted, we turn to the next method.

## 2.3.3 Single-Variable Expansion

This method starts from a test case already known to discriminate and changes exactly one thing each time, so as to deepen what the test cases inspect.

Seed selection. We take a test case already shown to discriminate as the seed test case; the defect it exposes is the target of that expansion round.

Expansion. A test case consists of a scenario and an assertion; the scenario is the session through which the mock model drives the candidate harness, and it fixes only the setting and the flow, while the adjudication rests with the assertion. We vary only the scenario, keeping the assertion fixed. Each time we change exactly one variable of the seed scenario to obtain a new candidate, and every candidate must still be anchored to some clause of the development document. We use four variables.

• The tool. A coding agent can often reach the same end through several tools, while a candidate harness may implement only some of them correctly.

• The inputform. The same inspection can be triggered by inputs of several forms; when inspecting path escape, we can replace the input with a symlink to a file, a symlink to a directory, a multi-level symlink, or a relative-path symlink.

• The erroneous parameter. The same coding mistake may appear on different parameters, for instance treating the value 0 as absent.

• The site where the result is returned. The result of the same behavior may appear on standard output, on standard error, or in the error text.

Distance check. Not every modified scenario adds discriminative power beyond the seed, so we assess whether each candidate’s scenario is far enough from the seed scenario. A candidate too close to the seed is discarded; otherwise we retain it as a test case and use it as the next seed, from which the following round of single-variable expansion proceeds.

This method spreads outward from seeds, complementing the two preceding methods. A clause with no anchored test case cannot provide a seed, and such gaps can only be filled by the first two methods; the three are therefore used in alternation.

## 2.4 Positive Control

A test case that passes does not necessarily show that the candidate harness has implemented the requirement in question. Consider path escape: the mock model’s script has the candidate harness read a symlink pointing outside the workspace with its read tool, and the test case asserts that the external file’s content does not appear in the requests sent to the model. That the content is absent may mean the implementation blocked this out-of-bounds read, or it may mean its read tool cannot read any file at all. The assertion holds in both situations; only the former constitutes meeting the requirement, and the latter is a vacuous pass.

To rule out vacuous passes, we design the positive control mechanism. A positive control shares the scenario and the fixture with its test case, where the fixture consists of the files and initial state prepared in the workspace before a run. It changes only the input or access route, selects a case outside the constraint imposed by the clause under test, and asserts an outcome that must occur when the observation precondition holds, that is, when the effect the test case inspects is in fact observable. In the example above, the positive control reuses the same symlink, accesses it through the bash tool instead, and asserts that the external file is indeed readable.

We group test cases that share an observation precondition under one positive control, and a test case with several such preconditions is gated by several controls. If a positive control fails, the precondition does not hold, so a pass by the test cases in its group can only be vacuous and they are recorded as FAIL regardless of their own statuses. Positive controls themselves do not count toward the pass rate.

Given that the assertion is correctly constructed, a vacuous pass has three causes, each ruled out by one class of positive control.

Channel. The tool’s result may never reach the channel being observed at all. A test case observes only one of the four channels; if an implementation’s read tool never places file content into a request, then the content of a file outside the workspace will not appear there either. We therefore read an ordinary file with the read tool and confirm that its content does appear in the requests sent to the model. Failure here means the implementation does not put tool results onto this channel at all, so the absence of the forbidden content does not constitute interception. The path-escape test case does not require a control of its own, because its reverse control, described below, already covers this condition.

Sameform. This rules out failures that would have occurred even without the constraint under test. Reusing the same input form and changing only the route of access, we access the same symlink through the implementation’s bash tool and confirm that it resolves outside the workspace and that the file there is readable (the path-escape clause constrains file tools, and commands executed by bash fall outside its scope). Failure here means the implementation cannot complete even an access unconstrained by the clause, so its miss on the access under test has nothing to do with interception.

Reverse. This rules out the case where the implementation refuses inputs of this kind categorically, on grounds other than this access being out of bounds. An implementation that refuses all symlinks passes the test case above, yet it also refuses symlinks pointing inside the workspace, which are valid in normal use. We therefore replace the input with such a symlink, perform the same read, and confirm that its content still appears in the requests sent to the model. Failure here means the pass did not arise from recognizing an out-of-bounds read.

## 2.5 Statistics

The test suite contains 328 test cases, each accompanied where necessary by positive controls. We partition them into nine mutually disjoint categories according to the content of the development document and the three external contracts of the contract skeleton, as shown in Figure 4.

Coverage of the positive controls. Of the 328 test cases, 130 are gated by at least one positive control (Figure 4b), of which 114 are gated by one, 15 by two, and one by three. The control with the widest scope gates six test cases and confirms that a symlink pointing outside the workspace is in fact readable; the next two gate five each and confirm, respectively, that reading by range works and that a search returns a hit. These three thus check the observation preconditions shared by the most test cases.

## 3 Experimental Setup

Models under evaluation. We evaluate nineteen models drawn from eleven vendors, namely Claude Opus 4.6 (Anthropic, 2026a), Opus 4.7 (Anthropic, 2026b), Opus 4.8 (Anthropic, 2026c), and Opus 5 (Anthropic, 2026d); GPT-5.5 (OpenAI, 2026a), GPT-5.6-sol (OpenAI, 2026b), and GPT-6 Astra (OpenAI, 2026c); Qwen3.8 Max 0902 (Alibaba Cloud,

![](images/27ca2709379222571bbe8abe0943a024adbe859e5e31c58a105d0cb0e896b785.jpg)  
(a) Test cases per category.

![](images/7961ba9d03bad7ff92e457cc53e5a89b4c2ad4d4588c59d095f144db8d1aaecd.jpg)  
(b) The share of test cases gated by at least one positive control.  
Figure 4: File operations and permissions have the highest positive-control coverage, whereas the interactive terminal has the lowest. Artifact checks is marked NA, because the contract check already settles in full what its test cases examine, namely that the deliverable is complete.

2026); Kimi K3 (Kimi Team, 2026); GLM 5.2 (Z.ai, 2026a) and GLM 5.3 (Z.ai, 2026b); DeepSeek V4 Pro (DeepSeek-AI, 2026b) and DeepSeek V4.1 Flash (DeepSeek-AI, 2026a); Hy3 (Tencent Hunyuan, 2026a) and Hy4 preview (Tencent Hunyuan, 2026b); Doubao Seed 2.1 Pro (ByteDance Seed, 2026); MiMo-V2.6-Flash (Xiaomi MiMo, 2026); Grok 4.7 (xAI, 2026); and Step 5 Preview (StepFun, 2026).

Execution environment. Both stages run on the same server, an AMD EPYC 7K83 machine with 48 logical processors and 128 GB of memory.

Configuration. In Stage 1, the builder harness is OpenCode 1.17.0, installed into the build container as a single compiled executable. The builder constructs the candidate harness through OpenCode’s non-interactive entry point opencode run, with the task description supplied once on the command line. Because that entry point advances the session by only one turn per invocation, we resume the same session with --continue, as OpenCode’s official documentation describes, and supply a further prompt whenever no submission has yet been made, up to twenty-four turns or until the wall clock is exhausted. As the session approaches the context limit, the builder harness compacts the history automatically, with the trigger threshold set at eighty percent of the model’s context window. Each build is subject to a two-hour wall-clock limit and to a quota of at most three submissions, and terminates as soon as either is exhausted. In Stage 2, every test case is executed once inside its own container, on its own fixture, subject to a 900-second limit beyond which execution is terminated and the test case recorded as FAIL.

## 4 Results

## 4.1 Main Results

The suite is far from saturated. As shown in Table 2, GPT-6 Astra attains the highest pass rate at 53.05%, while the weakest model, Hy3, reaches 27.44%, a gap of roughly twenty-six percentage points. Even the strongest model fails nearly half of the test cases. The suite is therefore neither saturated on this cohort nor incapable of discriminating among the models.

The top two models differ on boundary scenarios. On the ordinary scenarios Opus 5 leads with 80.68% against 72.46% for GPT-6 Astra, whereas on the boundary scenarios GPT-6 Astra leads with 48.97% and Opus 5 falls to 37.02%. The performance gap between the two models is therefore concentrated in their handling of boundary clauses.

## 4.2 Results by Category

In addition to each category’s average pass rate, we report the gap between its strongest and weakest models, which we call the strong-weak gap.

Table 2: Pass rate of the nineteen models (%), decomposed by evaluation category. The rightmost column is the primary measure: the pass rate over all 328 test cases, reported as the mean ± the standard deviation over the three scored candidate harnesses. $w _ { \mathrm { O t h e r ^ { \prime \prime } } }$ merges artifact checks, startup, and extension discovery, which hold 7, 7, and 1 test cases respectively. Positive controls are excluded from every denominator. Abbreviations in the header are Perm. for permissions, File ops. for file operations, Session for session state, Direct for direct drive, and Interactive for interactive terminal.
<table><tr><td rowspan="2">Model</td><td colspan="7">Perm. File ops. Session Direct Headless Interactive Other</td><td>All</td></tr><tr><td>61</td><td>75</td><td>84</td><td>33</td><td>40</td><td>20</td><td>15</td><td>328</td></tr><tr><td>GPT-6 Astra</td><td>71.6</td><td>46.7</td><td>44.0</td><td>39.4</td><td>45.0</td><td>91.7</td><td>60.0</td><td> ${ \bf 5 } 3 . 0 5 \pm { \bf 0 . 2 5 }$ </td></tr><tr><td>Opus 5</td><td>63.4</td><td>39.6</td><td>38.1</td><td>17.2</td><td>43.3</td><td>91.7</td><td>75.6</td><td> ${ \bf 4 6 . 6 5 \pm 1 . 8 0 }$ </td></tr><tr><td>Qwen3.8 Max</td><td>61.2</td><td>38.2</td><td>43.3</td><td>15.2</td><td>32.5</td><td>93.3</td><td>75.6</td><td> ${ \bf 4 5 . 8 3 \pm 0 . 1 4 }$ </td></tr><tr><td>GPT-5.6-sol</td><td>47.5</td><td>48.9</td><td>29.4</td><td>32.3</td><td>39.2</td><td>83.3</td><td>68.9</td><td> $4 3 . 8 0 \pm 2 . 4 4$ </td></tr><tr><td>Grok 4.7</td><td>54.6</td><td>50.7</td><td>28.2</td><td>22.2</td><td>33.3</td><td>78.3</td><td>75.6</td><td> $4 3 . 5 0 \pm 2 . 4 9$ </td></tr><tr><td>GLM 5.3</td><td>51.4</td><td>41.8</td><td>33.3</td><td>18.2</td><td>31.7</td><td>85.0</td><td>77.8</td><td> ${ \bf 4 2 . 0 7 \pm 1 . 3 9 }$ </td></tr><tr><td>MiMo-V2.6-Flash</td><td>50.3</td><td>40.4</td><td>37.7</td><td>14.1</td><td>33.3</td><td>76.7</td><td>75.6</td><td> ${ \bf 4 1 . 8 7 \pm 3 . 2 9 }$ </td></tr><tr><td>Step 5 Preview</td><td>47.0</td><td>45.8</td><td>34.1</td><td>14.1</td><td>25.8</td><td>76.7</td><td>77.8</td><td> ${ \bf 4 0 . 7 5 \pm 3 . 2 9 }$ </td></tr><tr><td>DeepSeek V4.1 Flash</td><td>49.2</td><td>36.4</td><td>34.5</td><td>14.1</td><td>31.7</td><td>81.7</td><td>80.0</td><td> ${ \bf 4 0 . 2 4 \pm 0 . 8 6 }$ </td></tr><tr><td>Hy4 preview</td><td>44.8</td><td>38.2</td><td>35.7</td><td>10.1</td><td>27.5</td><td>78.3</td><td>77.8</td><td> ${ \bf 3 8 . 9 2 \pm 1 . 4 6 }$ </td></tr><tr><td>Kimi K3</td><td>43.2</td><td>40.9</td><td>25.4</td><td>18.2</td><td>32.5</td><td>86.7</td><td>73.3</td><td> ${ \bf 3 8 . 3 1 \pm 1 . 2 8 }$ </td></tr><tr><td>Opus 4.8</td><td>30.1</td><td>48.4</td><td>33.7</td><td>12.1</td><td>22.5</td><td>83.3</td><td>80.0</td><td> ${ \bf 3 8 . 0 1 \pm 0 . 8 7 }$ </td></tr><tr><td>GPT-5.5</td><td>36.1</td><td>36.4</td><td>25.0</td><td>24.2</td><td>20.8</td><td>76.7</td><td>77.8</td><td> ${ \bf 3 4 . 6 5 \pm 1 . 4 2 }$ </td></tr><tr><td>Doubao Seed 2.1 Pro</td><td>41.5</td><td>36.0</td><td>15.5</td><td>20.2</td><td>18.3</td><td>83.3</td><td>73.3</td><td> ${ \bf 3 2 . 6 2 \pm 1 . 5 1 }$ </td></tr><tr><td>Opus 4.7</td><td>36.1</td><td>29.8</td><td>25.8</td><td>12.1</td><td>16.7</td><td>85.0</td><td>80.0</td><td> ${ \bf 3 2 . 2 2 \pm 1 . 6 6 }$ </td></tr><tr><td>DeepSeek V4 Pro</td><td>29.5</td><td>33.3</td><td>21.0</td><td>23.2</td><td>21.7</td><td>78.3</td><td>80.0</td><td> ${ \bf 3 1 . 9 1 \pm 1 . 2 3 }$ </td></tr><tr><td>Opus 4.6</td><td>20.8</td><td>37.3</td><td>21.8</td><td>25.3</td><td>3.3</td><td>60.0</td><td>80.0</td><td> ${ \bf 2 8 . 2 5 \pm 0 . 1 4 }$ </td></tr><tr><td>GLM 5.2</td><td>23.0</td><td>32.9</td><td>21.8</td><td>12.1</td><td>12.5</td><td>63.3</td><td>80.0</td><td> ${ \pm 7 . 6 4 \pm 1 . 0 1 }$ </td></tr><tr><td>Hy3</td><td>28.4</td><td>31.1</td><td>19.4</td><td>13.1</td><td>9.2</td><td>65.0</td><td>80.0</td><td> ${ 2 7 . 4 4 \pm 0 . 4 3 }$ </td></tr><tr><td>Mean</td><td>43.7</td><td>39.6</td><td>29.9</td><td>18.8</td><td>26.4</td><td>79.9</td><td>76.3</td><td>38.30</td></tr></table>

Table 3: Strong-weak gap and test-case composition by evaluation category. The last three columns split each category’s test cases into floor test cases, which no candidate harness passes, ceiling test cases, which every candidate harness passes, and the remaining discriminating test cases that determine the ranking.

The categories differ sharply in discriminative power. As shown in Table 3, the permissions category provides the greatest separation among models: its average pass rate is 43.7%, its strong-weak gap is 50.8 points, and 51 of its 61 test cases are discriminating. File operations has a comparable average pass rate of 39.6% but a gap of only 20.9 points. The interactiveterminal category averages 79.9% and is approaching saturation. In addition, 56 test cases are failed by every candidate harness, so these cases provide no discrimination within the evaluated cohort; they are concentrated in the direct drive, session state, permissions, and headless categories.

<table><tr><td>Category</td><td>Cases</td><td>Avg. pass</td><td>Strong-weak</td><td>Floor</td><td>Ceiling</td><td>Discriminating</td></tr><tr><td>Session state</td><td>84</td><td>29.9</td><td>28.6</td><td>14</td><td>0</td><td>70</td></tr><tr><td>File operations</td><td>75</td><td>39.6</td><td>20.9</td><td>3</td><td>0</td><td>72</td></tr><tr><td>Permissions</td><td>61</td><td>43.7</td><td>50.8</td><td>10</td><td>0</td><td>51</td></tr><tr><td>Headless</td><td>40</td><td>26.4</td><td>41.7</td><td>9</td><td>0</td><td>31</td></tr><tr><td>Direct drive</td><td>33</td><td>18.8</td><td>29.3</td><td>17</td><td>0</td><td>16</td></tr><tr><td>Interactive terminal</td><td>20</td><td>79.9</td><td>33.3</td><td>0</td><td>7</td><td>13</td></tr><tr><td>Artifact checks</td><td>7</td><td>100.0</td><td>0.0</td><td>0</td><td>7</td><td>0</td></tr><tr><td>Startup</td><td>7</td><td>63.4</td><td>42.9</td><td>2</td><td>2</td><td>3</td></tr><tr><td>Extension discovery</td><td>1</td><td>0.0</td><td>0.0</td><td>1</td><td>0</td><td>0</td></tr><tr><td>Total</td><td>328</td><td>38.30</td><td></td><td>56</td><td>16</td><td>256</td></tr></table>

Table 4: Architecture of the candidate harnesses. Every column is a mean over each model’s three scored builds, except Lines/module, which is mean code lines divided by mean module count, computed before rounding; Largest module is the share of all code lines held by the single largest module.
<table><tr><td>Model</td><td>Pass rate</td><td>Code lines</td><td>Modules</td><td>Lines/module</td><td>Largest module</td></tr><tr><td>GPT-6 Astra</td><td>53.05</td><td>7964</td><td>12.3</td><td>646</td><td>21%</td></tr><tr><td>Opus 5</td><td>46.65</td><td>13845</td><td>33.3</td><td>415</td><td>13%</td></tr><tr><td>Qwen3.8 Max</td><td>45.83</td><td>16681</td><td>40.0</td><td>417</td><td>10%</td></tr><tr><td>GPT-5.6-sol</td><td>43.80</td><td>2745</td><td>14.7</td><td>187</td><td>23%</td></tr><tr><td>Grok 4.7</td><td>43.50</td><td>5625</td><td>13.3</td><td>422</td><td>25%</td></tr><tr><td>GLM 5.3</td><td>42.07</td><td>8186</td><td>21.3</td><td>384</td><td>16%</td></tr><tr><td>MiMo-V2.6-Flash</td><td>41.87</td><td>11063</td><td>23.0</td><td>481</td><td>18%</td></tr><tr><td>Step 5 Preview</td><td>40.75</td><td>7620</td><td>18.7</td><td>408</td><td>16%</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>40.24</td><td>5519</td><td>15.3</td><td>360</td><td>20%</td></tr><tr><td>Hy4 preview</td><td>38.92</td><td>7251</td><td>19.7</td><td>369</td><td>16%</td></tr><tr><td>Kimi K3</td><td>38.31</td><td>5967</td><td>23.7</td><td>252</td><td>14%</td></tr><tr><td>Opus 4.8</td><td>38.01</td><td>5881</td><td>30.3</td><td>194</td><td>13%</td></tr><tr><td>GPT-5.5</td><td>34.65</td><td>1271</td><td>2.7</td><td>477</td><td>78%</td></tr><tr><td>Doubao Seed 2.1 Pro</td><td>32.62</td><td>6761</td><td>41.7</td><td>162</td><td>8%</td></tr><tr><td>Opus 4.7</td><td>32.22</td><td>4355</td><td>18.7</td><td>233</td><td>19%</td></tr><tr><td>DeepSeek V4 Pro</td><td>31.91</td><td>4169</td><td>13.0</td><td>321</td><td>27%</td></tr><tr><td>Opus 4.6</td><td>28.25</td><td>2351</td><td>12.0</td><td>196</td><td>24%</td></tr><tr><td>GLM 5.2</td><td>27.64</td><td>4047</td><td>18.3</td><td>221</td><td>25%</td></tr><tr><td>Hy3</td><td>27.44</td><td>2690</td><td>13.7</td><td>197</td><td>26%</td></tr></table>

A model’s rank varies greatly across categories. Among the six main categories, each of which holds at least twenty test cases, a model’s rank varies by as many as sixteen positions: Grok 4.7 ranks first on file operations but eleventh on session state, Opus 4.6 third on direct drive but nineteenth on permissions, and Hy4 preview fifth on session state but nineteenth on direct drive. GPT-6 Astra leads four of the six categories, while Grok 4.7 and Qwen3.8 Max lead the remaining two. The categories show positive pairwise associations rather than clear trade-offs. All fifteen pairwise correlations between the six main categories are positive, with a mean of +0.521, and the overall pass rate correlates with all six of them, five between 0.69 and 0.94. These results are consistent with a largely one-dimensional capability: models differ mainly in overall capability rather than in which categories they specialize in. Model rankings nevertheless vary substantially across categories.

The test suite is a reliable instrument. The KR-20 coefficient computed over the fifty-seven candidate harnesses across all test cases is 0.913, indicating high internal consistency, as expected if the test cases measure one underlying capability.

## 4.3 Architecture of the Candidate Harnesses

This section analyzes the source code of the fifty-seven scored candidate harnesses. A module is a Python source file under the root of a candidate harness and a code line is a non-blank line; we define a category as having a dedicated module when at least one module filename contains a keyword associated with that category.

Implementation scale is set mainly by the model. As shown in Table 4, per-model mean code size ranges from 1,271 to 16,681 lines, and across the fifty-seven builds the range is 1,062 to 19,208. Most of this variation lies between models rather than among the three builds of a given model: 95.9% of the variance in code lines is between models, and the coefficient of variation across the three builds of a model averages 12.7%, compared with 60.1% across the nineteen model means. The corresponding between-model variance shares are 87.9% for module count, 80.6% for lines per module, and 95.7% for largest-module share; Figure 5 shows the same between-model dominance for code lines. The correlation between code lines and pass rate is 0.614 across the fifty-seven builds, and 0.629 between the model means. Implementation scale is driven mainly by model identity, with comparatively little variation across a model’s three builds.

![](images/3bce12a9fa8ce625226b99bfd759f1b4e326f8cc37f74e07ec3e3f758fafbd33.jpg)  
Figure 5: Code lines against pass rate. Left, one point per model at its mean over the three scored builds, with a least-squares fit and its 95% confidence band. Right, each model’s three scored builds, spanned by a segment with the mean marked, one row per model, ordered by pass rate.

Harnesses with more modules score lower. Controlling for code lines, the partial correlation between module count and pass rate is −0.423, and the leave-one-model-out check keeps it between −0.484 and −0.309; correspondingly, lines per module has a partial correlation of +0.504, holding between 0.312 and 0.649 under the same check. Fragmentation is therefore not free.

## 5 Related Work

## 5.1 Code and Software Engineering Benchmarks

Coding benchmarks have evolved toward tasks involving increasingly large codebases, and fall broadly into two classes: code completion and repository-level generation.

Benchmarks in the completion class ask for local edits to existing code. HumanEval (Chen et al., 2021) established the practice of judging a single function by unit tests, and MBPP (Austin et al., 2021) and EvalPlus (Liu et al., 2023) extended this unit-test-based paradigm to a larger scale; the scope then grew to a whole repository with SWE-bench (Jimenez et al., 2024), which asks the model to resolve a real issue and judges by the tests shipped with the corresponding pull request. Variants of SWE-bench extend this setup while addressing contamination (Deng et al., 2026), evaluating end-to-end performance on freelance tasks (Miserendino et al., 2025), or deriving specifications from documentation changes (Jiang et al., 2026).

Repository-level generation removes the existing implementation and asks for a complete one. The evaluation criterion depends on the available reference implementation. Commit0 (Zhao et al., 2025) and NL2Repo-Bench (Ding et al., 2026) ask models to reconstruct released libraries and judge by the supplied or upstream unit tests. RepoZero (Zhang et al., 2026b) compares output equivalence against a second implementation; ProgramBench (Yang et al., 2026) uses a reference executable to generate end-to-end tests through fuzzing; and Deep-

SWE (Huang et al., 2026b) hand-writes a verifier per task that accepts any implementation of the requested feature set.

## 5.2 Harness Benchmarks

A harness is a class of software whose functional surface can be enumerated feature by feature (de Macedo, 2026; Guo et al., 2026; Barbaste et al., 2026). This makes it a suitable target for a construction benchmark. Most harness benchmarks evaluate existing harnesses. Claw-SWE-Bench (Zheng et al., 2026) fixes the prompt, budget, workspace, and evaluator, and finds that the choice of harness shifts scores by 27.4 points, compared with 29.4 points for the choice of model; Harness-Bench (Yao et al., 2026) isolates configuration-level effects over 106 sandboxed tasks and concludes that ability should be reported at the model–harness configuration level rather than attributed to the base model alone.

Other benchmarks hold the harness fixed and evaluate a single capability of it, such as rule compliance across the surfaces an agent reads (Huang et al., 2026c), skill retrieval under progressive disclosure (Han et al., 2026), component-lifecycle reasoning (Sileo & Kachler, 2026), or safety along a policy ladder and across persistent carriers (Davidovich et al., 2026; Bai et al., 2026; Zhang et al., 2026a). Still others let the given harness change from stage to stage and ask whether the agent keeps pace (Ke et al., 2026).

A further line of work asks a model to produce or revise the harness itself, starting from an existing harness and its execution feedback (Wu et al., 2026; Huang et al., 2026a; Ursekar et al., 2026); because gains are measured on downstream tasks, they cannot be separated from those obtained by simply allowing more test-time search, a confound noted in this literature (Wang et al., 2026). All of these studies infer harness quality from downstream tasks. Such tasks must be prepared in advance, and the ability of the model that drives them also influences the measured score. Nor is output-only scoring sufficient, since a harness can return a correct and harmless answer along a trajectory that reached unauthorized resources, and output-level evaluation cannot see failures of that kind (Liu et al., 2026). None of them adjudicates, clause by clause, whether a harness’s own behavior conforms to a specification.

## 5.3 Position of This Work

Doc2Harness differs from the work above in three respects. First, the target deliverable does not correspond to any existing implementation: the development document consolidates the functionality of twelve frontier harnesses into a self-consistent set of clauses, whereas prior repository-level deliverables are reconstructions of something already published. Second, the candidate harness is driven by a mock model following a preset script rather than by a real model on downstream tasks, so the evaluation is deterministic and what is judged is the candidate harness itself. Third, the candidate harness is executed end to end as a black box through a complete scripted session, with effects read from four channels. This makes constraints of the form “a rejected operation must not take place” adjudicable clause by clause; and because an assertion requiring some effect not to appear may hold simply because the feature under test never ran, such test cases are equipped with positive controls confirming that the observation precondition holds. In sum, Doc2Harness is the first benchmark able to assess deterministically how well a model develops a harness.

## 6 Limitations

The protocol measures a single build. Each build allows up to three submissions and is not an iterative build–evaluate–repair loop. The contract check tests form rather than behavior, so the amount of revision a candidate harness receives depends on when the model submits, and the pass rate of an early submitter is a lower bound rather than an upper bound. The protocol does not measure improvement based on behavioral feedback after the candidate harness is frozen.

A single defect can fail several test cases at once. Some test cases are built around the same feature: the single-variable expansion of §2.3.3 derives several variants from one seed test case, and although the distance check discards candidates too close to their seed, the variants that remain are still correlated. A single defect can therefore cause correlated failures across these variants.

## 7 Conclusion

Doc2Harness is the first benchmark able to assess deterministically how well a model develops a harness. The build model under evaluation, operating inside a fixed builder harness, builds a complete candidate harness from a development document and a contract skeleton that declares interfaces only; a mock model then drives it through a preset script for each test case, and explicit assertions determine behavioral conformance.

Our experiments show that even the strongest model fails nearly half of the test cases, and that the failures are not random but concentrate on one implementation choice: whether the same functionality is split across several modules. We hope Doc2Harness will serve as a benchmark for measuring this class of construction ability.

## References

Alibaba Cloud. Alibaba unveils Qwen3.8-Max: Its largest and most capable flagship model to date. https://www.alibabacloud.com/en/press-room/alibaba-unveils-qwen3-8-m ax, 2026.

Anthropic. Claude Code. https://github.com/anthropics/claude-code, 2025.

Anthropic. Claude Opus 4.6 system card. https://www.anthropic.com/claude-opus-4-6 -system-card, 2026a.

Anthropic. Claude Opus 4.7 system card. https://www.anthropic.com/claude-opus-4-7 -system-card, 2026b.

Anthropic. Claude Opus 4.8 system card. https://www.anthropic.com/claude-opus-4-8 -system-card, 2026c.

Anthropic. Claude Opus 5 system card. https://www.anthropic.com/claude-opus-5-sys tem-card, 2026d.

Anysphere. Cursor. https://cursor.com, 2023.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Yajing Bai, Jinhao Duan, Jie Peng, Xianfeng Wu, Sijia Liu, Song Wang, and Tianlong Chen. HarnessRisk: A lifecycle-oriented benchmark for agent harness safety. arXiv preprint arXiv:2608.17597, 2026.

Paul Barbaste, Tristan Darrigol, Germain Vu, and Tom Wiltberger. Harness Engineering: Anatomy, architecture, and evolution of coding agents – A source-code study of eleven systems. arXiv preprint arXiv:2609.00006, 2026.

ByteDance Seed. Seed2.1: A next-generation agent for real-world productivity. https: //seed.bytedance.com/en/seed2\_1, 2026.

Charm. Crush. https://github.com/charmbracelet/crush, 2025.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Cline Bot Inc. Cline. https://github.com/cline/cline, 2024.

Dotan Davidovich, Yair Amar, Hai Rozencwajg, and Or Hiltch. Permission denied: Policy-graded evaluation of coding agents in hardened environments. arXiv preprint arXiv:2608.02670, 2026.

Sanderson Oliveira de Macedo. What makes a harness a harness: necessary and sufficient conditions for an agent harness. arXiv preprint arXiv:2606.10106, 2026.

DeepSeek-AI. DeepSeek-V4.1-Flash: Smarter, faster, more efficient. https://api-docs.deep seek.com/news/news260910/, 2026a.

DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026b.

Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, Niklas Lauffer, Andrew Park, Chetan Rane, Karmini Sampath, et al. SWE-Bench Pro: Can AI agents solve long-horizon software engineering tasks? In International Conference on Machine Learning, 2026.

Jingzhe Ding, Shengda Long, Changxin Pu, Ge Zhang, Zhou Huan, Hongwan Gao, Xiang Gao, Chao He, Yue Hou, Fei Hu, et al. NL2Repo-Bench: Towards long-horizon repository generation evaluation of coding agents. In International Conference on Machine Learning, 2026.

Paul Gauthier. Aider. https://github.com/Aider-AI/aider, 2023.

Yuyao Ge, Lingrui Mei, Zenghao Duan, Tianhao Li, Yujia Zheng, Yiwei Wang, Lexin Wang, Jiayu Yao, Tianyu Liu, Yujun Cai, Baolong Bi, Fangda Guo, Jiafeng Guo, Shenghua Liu, and Xueqi Cheng. A survey of vibe coding with large language models. arXiv preprint arXiv:2510.12399, 2025.

Google. Gemini CLI. https://github.com/google-gemini/gemini-cli, 2025.

Jianyuan Guo, Zhiwei Hao, Chengcheng Wang, Cheng Fan, Tingzhang Luo, Hongguang Li, Ying Gao, Hefei Mei, Jiankun Peng, Rongjian Xu, et al. From question answering to task completion: A survey on agent system and harness design. arXiv preprint arXiv:2606.20683, 2026.

Jinyi Han, Yuanjian Xu, Ying Liao, Xinyi Wang, Zishang Jiang, Zixiang Di, Fanyang Lu, Zhichao Hu, and Yanghua Xiao. Skill-Use: Can LLMs actually use skills in agentic harnesses? arXiv preprint arXiv:2608.04828, 2026.

Lisheng Huang, Chen Yang, Hao Zhou, Huatong Song, Zongchao Chen, Ran Le, Yang Song, Wayne Xin Zhao, and Tao Zhang. Evo-Bench: Can language models improve agent harness? arXiv preprint arXiv:2608.09096, 2026a.

Wenqi Huang, Charley Lee, Leonard Tng, and Serena Ge. DeepSWE: Measuring frontier coding agents on original, long-horizon engineering tasks. arXiv preprint arXiv:2607.07946, 2026b.

Zining Huang, Haoran Que, Hong Zeng, Ge Zhang, Zuo Wang, Jin Chen, Haodong Wang, Zhongfei Hou, Changxin Pu, Shen Yan, et al. Harness-IF: Evaluating instruction following across instruction surfaces in coding agents. arXiv preprint arXiv:2608.11727, 2026c.

Zhonghao Jiang, Le Deng, Jialun Cao, Michael Pradel, and Zhongxin Liu. Doc2Feat-Bench: Evaluating documentation-driven feature addition. In IEEE/ACM International Conference on Automated Software Engineering, 2026.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024.

Zixuan Ke, Vaidehi Patil, Haizhou Shi, Yang Li, Ye Liu, Sarath Shekkizhar, Anurag Koul, Jiayu Wang, Xuan Phi Nguyen, Semih Yavuz, et al. EvoHarnessBench: Can your agents keep pace with an evolving harness? arXiv preprint arXiv:2609.04280, 2026.

Kimi Team. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Chengzhi Liu, Yichen Guo, Yepeng Liu, Yuzhe Yang, Qianqi Yan, Xuandong Zhao, Wenyue Hua, Sheng Liu, Sharon Li, Yuheng Bu, and Xin Eric Wang. Auditing agent harness safety. arXiv preprint arXiv:2605.14271, 2026.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by ChatGPT really correct? Rigorous evaluation of large language models for code generation. In Advances in Neural Information Processing Systems, 2023.

Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, and Shenghua Liu. A survey of context engineering for large language models. arXiv preprint arXiv:2507.13334, 2025.

Samuel Miserendino, Michele Wang, Tejal Patwardhan, and Johannes Heidecke. SWE-Lancer: Can frontier LLMs earn \$1 million from real-world freelance software engineering? In International Conference on Machine Learning, 2025.

Moonshot AI. Kimi Code. https://github.com/MoonshotAI/kimi-code, 2026.

OpenAI. Codex CLI. https://github.com/openai/codex, 2025.

OpenAI. GPT-5.5 system card. https://deploymentsafety.openai.com/gpt-5-5, 2026a.

OpenAI. GPT-5.6 system card. https://deploymentsafety.openai.com/gpt-5-6, 2026b.

OpenAI. GPT-6 Astra system card. https://deploymentsafety.openai.com/gpt-6-astra, 2026c.

OpenClaw Foundation. OpenClaw. https://github.com/openclaw/openclaw, 2025.

OpenCode Contributors. OpenCode. https://github.com/anomalyco/opencode, 2025.

Qwen Team. Qwen Code. https://github.com/QwenLM/qwen-code, 2025.

Damien Sileo and Dimitri Kachler. CordisBench: Can language models reason about component lifecycles in dynamic agent harnesses? arXiv preprint arXiv:2609.01600, 2026.

StepFun. Step 5 Preview. https://www.stepfun.com/step-5-preview, 2026.

Tencent Hunyuan. Hy3 model card. https://huggingface.co/tencent/Hy3, 2026a.

Tencent Hunyuan. Hy4 preview model card. https://huggingface.co/tencent/Hy4-previ ew, 2026b.

Varun Ursekar, Apaar Shanker, Yash Maurya, Shehab Yasser, Vijay S. Kalmath, Veronica Chatrath, and Yuan Xue. HarnessOpt-Bench: Evaluating LLMs at harness optimization. arXiv preprint arXiv:2608.06301, 2026.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. OpenHands: An open platform for AI software developers as generalist agents. In International Conference on Learning Representations, 2025.

Yike Wang, Huaisheng Zhu, Zhengyu Hu, Yige Yuan, Zhengyu Chen, Shakti Senthil, Hannaneh Hajishirzi, Yulia Tsvetkov, Noah A. Smith, Pradeep Dasigi, and Teng Xiao. Rethinking the evaluation of harness evolution for agents. arXiv preprint arXiv:2607.12227, 2026.

Yuhao Wu, Jingyuan Zhang, Jiajun Shi, Xinping Lei, Qingshui Gu, Yuxuan Zhang, Zexuan Wang, Chen He, Chen Huang, Maojia Song, et al. HarnessDev: Can LLMs create and evolve their own agent harness? arXiv preprint arXiv:2609.01437, 2026.

xAI. Introducing Grok 4.7. https://x.ai/news/grok-4-7, 2026.

Xiaomi MiMo. MiMo-V2.6: Scaling up reinforcement learning for self-improvement. https: //mimo.mi.com/docs/en-US/news/latest/v2-6, 2026.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, 2024.

John Yang, Kilian Lieret, Jeffrey Ma, Parth Thakkar, Dmitrii Pedchenko, Sten Sootla, Emily McMilin, Pengcheng Yin, Rui Hou, Gabriel Synnaeve, et al. ProgramBench: Can language models rebuild programs from scratch? In Advances in Neural Information Processing Systems, 2026.

Yilun Yao, Xinyu Tan, Chao-Hsuan Liu, Yaoming Li, Zhengyang Wang, Wenhan Yu, Zhewen Tan, Yuxuan Tian, Guangxiang Zhao, Lin Sun, et al. Harness-Bench: Measuring harness effects across models in realistic agent workflows. arXiv preprint arXiv:2605.27922, 2026.

Z.ai. GLM-5.2: Built for long-horizon tasks. https://z.ai/blog/glm-5.2, 2026a.

Z.ai. GLM-5.3: Frontier coding with emergent cyber capabilities. https://z.ai/blog/gl m-5.3, 2026b.

Xiao Zhang, Yusheng Wang, Yuhao Fei, Dongyuan Li, Zian Liang, Liuyu Xiang, Hongxun Gu, and Zhaofeng He. HarnessSafe: Evaluating safety across persistent carriers in agent harnesses. arXiv preprint arXiv:2608.06984, 2026a.

Zhaoxi Zhang, Yiming Xu, Jiahui Liang, Weikang Li, Xiaoshuai Chen, Liwei Qian, Xin Pei, Jizhou Huang, Run Sun, and Yunfang Wu. RepoZero: Can LLMs generate a code repository from scratch? In Advances in Neural Information Processing Systems, 2026b.

Wenting Zhao, Nan Jiang, Celine Lee, Justin Chiu, Claire Cardie, Matthias Gallé, and Alexander Rush. Commit0: Library generation from scratch. In International Conference on Learning Representations, 2025.

Mengyu Zheng, Kai Han, Boxun Li, Haiyang Xu, Yuchuan Tian, Wei He, Hang Zhou, Jianyuan Guo, Hailin Hu, Lin Ma, et al. Claw-SWE-Bench: A benchmark for evaluating OpenClaw-style agent harnesses on coding tasks. arXiv preprint arXiv:2606.12344, 2026.