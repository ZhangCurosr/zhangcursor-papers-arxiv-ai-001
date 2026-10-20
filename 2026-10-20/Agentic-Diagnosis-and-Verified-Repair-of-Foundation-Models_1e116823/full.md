EvalRx

# Agentic Diagnosis and Verified Repair of Foundation Models

From model evaluation to statistically verified improvements across text, vision, and audio

EVALVITALS TEAM

Leading contributors Jiaqi Liu<sup>1,2,\*</sup> Ruinan Jin<sup>1,2,\*</sup> Xiaoxiao Li<sup>1,2,\*,†</sup> jiaqiliu@ece.ubc.ca ruinanjin@alumni.ubc.ca xiaoxiao.li@ece.ubc.ca

Core contributors Yitong Tang<sup>1</sup> Beidi Zhao<sup>1,2</sup> Gexin Huang<sup>1,2</sup> Qiong Zhang<sup>3</sup>

<sup>1</sup>The University of British Columbia <sup>2</sup>Vector Institute <sup>3</sup>Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)

Equal contribution. Authors are listed in alphabetical order. <sup>†</sup>Corresponding author: Xiaoxiao Li.

## ABSTRACT

Model evaluation reveals where models fail, but rarely why or whether a proposed fix genuinely works. We introduce EvalRx, an open-source agentic framework for statistically grounded model diagnosis and repair. EvalRx discovers systematic failure modes, explores candidate mechanisms, formulates falsifiable hypotheses, and verifies them by executing frozen, pre-specified tests before using supported diagnoses to guide tiered repair. Candidate repairs are validated against the unchanged model and promoted only when they improve the target behavior without unacceptable regression. Across language, vision-language, and audio-language settings spanning Gemini, Qwen, Gemma, and Nemotron, EvalRx improves performance in all 12 evaluated model family–modality combinations, with gains ranging from +1.4 to +10.9 percentage points (+5.8 points on average). These results show that model evaluation can serve not only to measure failures, but to support an auditable process for diagnosing and repairing them.

![](images/a127c39cffb901f9747794355e7af174788d94f5bf1392242dbd67994887639d.jpg)  
EvalRx automatically boosts the model performance across diverse LLMs, VLMs and ALMs. Code and artifacts: github.com/evalvitals/evalrx · Project overview · Demo

## 1 Introduction

AI agents are becoming the execution layer for software engineering, data analysis, and scientific work. Systems built on foundation models (FMs) already resolve software issues autonomously [17], execute machine-learning engineering pipelines end-to-end [5], and participate in scientific research [12, 24, 26]. However, an agent is only as reliable as its behavior in the particular conditions it encounters. A model can possess the capacity required for a task while systematic, localized errors prevent that capacity from being expressed reliably, under a prompt perturbation, for a visual relation, or on a long-tail entity class. Unlike intrinsic capacity limits, these operational failure modes can be observed, diagnosed, and sometimes repaired.

This creates a deployment-critical gap. Diagnosing failures should work like a routine model health check: localize where errors concentrate, formulate an actionable explanation, and verify that an intervention improves the afected behavior without causing unacceptable regressions. Aggregate benchmarks reveal that a model fails, but not where, why, or whether an apparent fix generalizes. Expert review of individual errors is costly and hard to scale. LLM- or agent-based inspection is scalable, but can produce fluent conclusions unsupported by valid statistical evidence [2, 4]. The problem is structural: a workflow that searches many slices and explanations will find appealing post-hoc patterns that do not survive a pre-registered test with multiplicity control, or repairs that do not survive confirmation on unseen cases.

EvalRx is an open-source agentic system for closed-loop model diagnosis and repair, turning failure analysis into statistically verified model improvements.

It closes the missing evidence-to-repair loop (Figure 1): starting from a fixed labeled evaluation set, it discovers candidate failure modes from case-level evidence, freezes a falsifiable test before adjudication, and tests repairs only for supported mechanisms. Existing tools automate individual pieces of this process: interpretability and auditing agents synthesize probes [21, 25, 34, 35], slice-discovery systems surface plausible weaknesses [3, 6, 7, 10, 11, 16, 20, 27, 42], and propose-and-verify systems provide statistical discipline [15, 43–45]. EvalRx joins these pieces under one decision rule: no diagnosis without statistical support, and no promoted repair without paired benefit and regression control. Each promoted repair is recorded as a verified repair recipe: the supported diagnosis, the

![](images/eb3667cd71cc65e1bc376c30dad63579840f19fd72e7f2fff871db0fb16d96e7.jpg)  
Figure 1. Verification-gated model evolution. A candidate repair is promoted to the next model version only after its improvement is confirmed on held-out data. Otherwise, the current model remains unchanged and the diagnosis-repair cycle continues.

intervention and its tier, and the statistical evidence that justified promotion, packaged so that the fix can be audited and reapplied.

Iterative improvement is therefore a protocol, not a synonym for repeated optimization. EvalRx distinguishes two evolution paths. First, non-parametric evolution (L1-L3) changes the input, surrounding scafold, or inference-time computation while keeping checkpoint weights frozen. Second, parametric evolution (L4) updates trainable parameters through fine-tuning or adapters and creates a new checkpoint. A repair on either path becomes version $v _ { t + 1 }$ only after passing the gate. That accepted version is then health-checked again for residual and newly introduced failures with fresh evidence, while the immutable root version $v _ { 0 }$ remains available to expose cumulative regression. L4

additionally requires explicit human approval.

EvalRx operationalizes model diagnosis and repair as a closed-loop agentic process. Starting from observed model failures, it probes the model to identify systematic failure modes (M1), analyzes candidate explanations with multiplicity-aware statistical tests (M2), formulates explicit and falsifiable hypotheses about the underlying mechanism (M3), and tests these hypotheses on the exploration split (or an optional held-out split) (M4). Supported diagnoses then guide targeted interventions (M5), whose improvements and side efects are evaluated against the unmodified model. By separating exploratory failure discovery, confirmatory diagnosis, and intervention, EvalRx prevents suggestive patterns from being treated as established mechanisms and proposed fixes from being treated as genuine improvements without independent validation.

EvalRx makes three contributions: (1) a modality-agnostic framework that closes the loop from model probing and failure discovery to hypothesis-driven diagnosis, mechanism verification, and targeted repair. (2) a verification-gated improvement procedure spanning non-parametric interventions at L1-L3 and parametric adaptation at L4, where every candidate update is evaluated against both its parent and the original model, and (3) a machine-readable audit trail that captures the data splits, analysis traces, hypotheses, verification results, interventions, version transitions, and stopping decisions needed to reproduce and audit each diagnosis and repair.

## 2 Related Work

Evaluation infrastructure and benchmark ecosystems. Benchmark suites and evaluation frameworks make model comparison more reproducible by standardizing datasets, prompts, metrics, and reporting. Broad, dynamic, and domain-specific evaluations complement one another: together they measure capabilities, changing task distributions, and user-facing preferences. This infrastructure is the measurement layer on which model development depends, but aggregate scores usually do not identify a coherent operational failure mechanism or specify how to repair it. EvalRx starts from the same labeled-evaluation setting, while treating per-case outcomes, traces, and metadata as evidence from which an actionable model-health hypothesis can be developed.

Discovering model failures. Prior work discovers model weaknesses either by actively generating probes or by mining existing evaluation results. Interpretability and auditing agents probe models to characterize mechanisms or elicit hidden behaviors [16, 18, 21, 25, 34, 35], while systems such as EvalTree and ErrorMap extract patterns or taxonomies from benchmark outputs and transcripts [3, 27, 42]. Related slice-discovery methods identify coherent error subpopulations using metadata, embeddings, or language-grounded attributes, primarily for vision classifiers [6, 7, 10, 11, 20, 36]. EvalRx instead starts from a fixed labeled evaluation set for language, vision-language, or audiolanguage models and uses tool-executed analyses of behavioral and internal signals to form open-ended failure hypotheses.

Validating failure hypotheses. Many diagnosis systems evaluate discovered patterns using LLM judgments or heuristic consistency criteria, which do not by themselves provide false-discovery control and can yield unsupported statistical conclusions [2, 4, 23, 46]. Propose-and-verify methods instead test natural-language hypotheses through held-out prediction, synthetic ground truth, or sequential falsification [9, 15, 22, 43–45], but have not been designed specifically for model-failure diagnosis. EvalRx operationalizes each failure hypothesis as an executable test, adjudicates hypotheses under a common verifier, applies BH or e-BH multiplicity correction, and permits inconclusive outcomes [13, 38].

Repairing model failures. Existing repair methods include natural-language patches, targeted data generation, model editing, and direct prompt optimization from performance feedback [1, 14, 29–31, 41]. Some language-grounded vision debuggers connect discovered slices to classifier retraining [6, 11, 20], whereas many LLM-diagnosis systems stop at discovery [3, 16]. In EvalRx, repair is instead gated by statistical validation: supported hypotheses trigger progressively deeper interventions, and only after being compared with an unchanged baseline will the most concise and efective candidate be retained.

Version governance and bounded recursive improvement. Model documentation and risk-management practices make provenance, intended use, and measured limitations inspectable. Self-improving agents add a further requirement: every promoted change can create a new distribution of failures, so a local gain is not suficient evidence of progress. We therefore frame EvalRx as bounded recursive improvement: a repair is promoted only after a baseline comparison and regression check, then the promoted version is re-audited under the same evidence discipline. The resulting audit record links the discovered issue, executable test, hypothesis verdict, intervention tier, version comparison, and promotion or rejection decision. This distinguishes a verification-gated improvement loop from open-ended optimization against self-generated feedback.

Table 1. Where existing approaches stop, and where EvalRx continues. EvalRx connects failure discovery, statistical hypothesis testing, repair selection, and version re-audit in one evidence-gated loop. Examples are representative, not exhaustive.
<table><tr><td>System family &amp; examples</td><td>Starts with</td><td>What it delivers</td><td>What remains external</td></tr><tr><td>Auditing &amp; slice discovery EvalTree, ErrorMap</td><td>Evaluation failures</td><td>Failure patterns or error slices</td><td>Held-out validation, repair, and post-repair audit</td></tr><tr><td rowspan="2">Propose-and-verify D3, VibeCheck, HypoGeniC Prompting, editing &amp; retraining</td><td>Candidate hypotheses</td><td>Tested hypotheses</td><td>Repair and version-level audit</td></tr><tr><td>A supplied target, reward, or task</td><td>Candidate repair</td><td>Failure diagnosis and a common verification rule</td></tr><tr><td>EvalRx</td><td>Labeled evaluation set</td><td>Verified repair, or an explicit inconclusive result</td><td>Nothing: re-audits the promoted version and recurses</td></tr></table>

<sup>∗</sup>Developer agents such as Claude Code can be target-supplied repair baselines, but do not by themselves provide a verified health loop.

![](images/b1fa75e378582c36336dfff2f50b2a1c85a5e63fae54bf2553402e73fdd6137c.jpg)  
Figure 2. Overview of EvalRx. EvalRx organizes model diagnosis and repair into five stages: M1 (Probe) discovers failures and gathers diagnostic evidence. M2 (Explore) analyzes failure structure and identifies candidate explanatory signals. M3 (Diagnose) formulates and freezes falsifiable hypotheses. M4 (Verify) tests these hypotheses with statistical evidence, and M5 (Repair) uses supported mechanisms to guide targeted repairs and verifies their improvement against the unchanged model.

## 3 EvalRx: Statistically Grounded AI Diagnosis and Repair

Figure 2 gives an overview of EvalRx. Starting from a target model and a frozen case batch, five stages (M1–M5) move from failure discovery to verified repair. Table 1 positions this workflow against existing auditing, propose-and-verify, and repair systems.

## 3.1 Settings: Inputs, Outputs, and Data Preprocessing

Given a model $f _ { \theta }$ and a labeled evaluation set

$$
\mathcal { D } _ { 0 } = \{ ( x _ { i } , y _ { i } , m _ { i } ) \} _ { i = 1 } ^ { N } ,
$$

where $x _ { i }$ may contain text, image, audio, or multimodal inputs, $y _ { i }$ is the reference answer, and $m _ { i }$ contains permitted case metadata, EvalRx first executes the model under a fixed inference configuration to establish its baseline behavior. For each case, it records the model response ${ \hat { y } } _ { i } = f _ { \theta } ( x _ { i } )$ and a task-specific correctness indicator $z _ { i } \in \{ 0 , 1 \}$ , producing the frozen baseline case table

$$
\mathcal { D } = \{ ( x _ { i } , y _ { i } , \hat { y } _ { i } , z _ { i } , m _ { i } ) \} _ { i = 1 } ^ { N } .
$$

Before any exploratory analysis, cases are deterministically assigned by identifier to an exploration split $\mathcal { D } _ { E }$ ,an optional sealed verification split $\mathcal { D } _ { H }$ and a confirmation split $\mathcal { D } _ { C }$ when final evaluation must be separated. The split seed, case identifiers, and any stratification variables are persisted in the run manifest, ensuring that subsequent diagnosis and repair decisions can be traced to the exact data partition on which they were made.

Only the exploration split $\mathcal { D } _ { E }$ is available during failure discovery, exploratory analysis, hypothesis construction, M4 adjudication and M5 candidate search. Each diagnostic hypothesis must be specified and frozen before M4 executes it (and before $\mathcal { D } _ { H }$ is opened when enabled). When multiple repair candidates are considered after diagnosis, the test split $\mathcal { D } _ { C }$ can be reserved for the final paired evaluation, preventing repair selection and final assessment from using the same cases. Throughout the process, EvalRx maintains a machine-readable audit trail containing the model version, data partitions, identified failure slice, frozen hypothesis specification, hypothesis verdict, repair trials, paired comparisons, and final promotion or stopping decision.

Table 2. M1 selects analyzers that match the target model and task protocol. These representative analyzer families difer in availability across modalities and model interfaces.
<table><tr><td>Access and modality</td><td>Per-case diagnostic signals</td></tr><tr><td>Black-box, all modalities</td><td>Calibration, consistency under prompt or input contrasts, uncertainty, termination, response-format violations, and counterfactual outcome</td></tr><tr><td>White-box language models</td><td>changes. Token entropy, logits, attention, representation similarity, and logit-lens-style readouts.</td></tr><tr><td>Vision-language and audio-language models</td><td>Grounding, hallucination, attention or attribution over non-text tokens modality ablations, local masking, and temporal or spatial</td></tr><tr><td>Agentic trajectories</td><td>perturbations. Loop detection, ignored-observation analysis, first-error localization, tool-use traces, and counterfactual trajectory tests.</td></tr></table>

## 3.2 M1: Failure Discovery and Protocol-Guided Probing

Firstly, EvalRx converts model behavior into structured, per-case diagnostic signals as shown in Figure 3. It works on individual successes and failures rather than on aggregate accuracy alone. Analyzers are selected according to the input modalities and model capabilities. The library covers black-box behavior (such as calibration, consistency, and termination), white-box signals (such as logits, attention, and representations), and multimodal failures (such as grounding errors and hallucination). Table 2 summarizes the analyzer families. When no built-in analyzer operationalizes a suspected mechanism, a coding agent may synthesize a diagnostic program, validate its syntax, and execute it in an isolated sandbox. These signals form the basis for subsequent statistical analysis.

![](images/7ffbd28dffee6a01aae4a5f10ab23a5a81681cfd503f882f6aeebd6106380e93.jpg)  
Figure 3. M1 turns raw model outcomes into an auditable diagnostic signal table. A normalized INPUT <sup>M2</sup> <sup>EXPLORATORY</sup> <sup>DATA</sup> <sup>ANALYSIS</sup> <sup>→</sup> <sup>M3</sup> <sup>HYPOTHESIS</sup> <sup>+</sup> <sup>CRITIQUE</sup> OUTPUTcase packet is routed to compatible analyzer families, assembled into a typed per-case signal matrix, and <sup>Which</sup> <sup>pattern</sup> <sup>deserves</sup> <sup>a</sup> <sup>test—a</sup>audited for coverage and missingness before M2.

## 3.3 M2-M3: Exploratory Data Analysis and Statistical Attribution with Hypothesis Generation

In this stage, M2 converts M1 measurements into a family of statistical analyses as shown in Figure 4. Depending on the measurement and task, the StatsAnalysisAgent may use signal-label association tests, paired McNemar comparisons with e-values, bootstrap efect comparisons, rank-based tests, or single-rate tests. Every analysis records the statistic, efect estimate, confidence interval when applicable, and inferential evidence. When a run examines multiple signals or candidate modes, EvalRx applies BH or e-BH correction across the declared test family. In this way, EvalRx measures the degree to which each diagnostic signal is associated with model failure. Visualizations<sup>INPUT</sup> <sup>OUTPUT</sup> such as signal distributions, heatmaps, and forest plots are used for interpretation, whereas statistical tests determine which associations are considered reliable.

M3 turns selected exploratory observations from M2 into one to three falsifiable hypothesis<sup>1</sup> <sup>·</sup> <sup>DISCOVER 02</sup> <sup>·</sup> <sup>SELECT 03</sup> <sup>·</sup> <sup>PROBE 04</sup> <sup>·</sup> <sup>ASSEMBLE</sup> contracts. A contract is represented as:

$$
h _ { k } = ( c _ { k } , p _ { k } , a _ { k } , t _ { k } , q _ { k } ) ,
$$

where $c _ { k }$ is a concrete mechanism claim, $p _ { k }$ is an executable target-slice predicate, <sub>003</sub> $a _ { k }$ specifies the expected association and direction, $t _ { k }$ specifies the analyzer, metric, statistical test, and minimum efect, and $q _ { k }$ specifies the required evidence grade.

This representation makes LLM-generated explanations falsifiable. For example, an assertion that a model underuses short acoustic events is not a diagnosis until it specifies which cases require a short event, which controlled perturbation should change the response, the expected direction,case\_id label signal.\* coverage missing\_reason → and the test. M3 therefore separates descriptive failure labels from claims about a mechanism.<sup>provenance</sup>

![](images/415b081727d6d2e3d81efee2fd4ca31d986063825d115377c1b3e7416e5f1ff8.jpg)  
Figure 4. M2-M3 turns observed failure structure into a falsifiable, frozen test contract. M2 INPUT <sup>M4</sup> <sup>·</sup> <sup>FROZEN-CONTRACT</sup> <sup>ADJUDICATION</sup> OUTPUTrecords descriptive candidates; M3 challenges competing explanations and freezes the prediction, control, <sup>What</sup> <sup>survives</sup> <sup>a</sup> <sup>test</sup> <sup>that</sup> <sup>cannot</sup> <sup>be</sup> <sup>edited?</sup>threshold, falsifier, and test design before M4 using them.

## <sup>01</sup> <sup>·</sup> <sup>TWO</sup> <sup>INPUTS 02</sup> <sup>·</sup> <sup>EXECUTE</sup>3.4 M4: Hypothesis Validation

FROM M3 ✗ hypothesis edits STAT e-BH ✓ ✓ SUPPORAfter proposing testable hypotheses, M4 executes statistical test design on $\mathcal { D } _ { E }$ (or optional $\mathcal { D } _ { H } )$ ✗ test substitution(Figure 5). It executes the frozen contract on D<sub>E</sub>or $\mathcal { D } _ { H }$ , applying the pre-specified analyzer in M1, statistical test in M2 and expected direction in M3. A hypothesis is supported only<sup>FROZEN</sup> when the efect has the expected direction, meets the declared minimum efect, survives the specifiedH3 peripheral\_attn <sup>✗</sup> multiplicity procedure, and satisfies the task protocol. It is refuted when the evidence contradicts the prediction and inconclusive otherwise.EVIDENCE ● writes its own code → ● e-BH corre

EvalRx additionally distinguishes observational from intervention-grade evidence. An associ-UNDARY FROZEN FIELDS NEXT ation between a naturally occurring signal and error is observational; a paired strategy contrast, controlled perturbation, or targeted mechanism test can provide intervention-grade evidence.

![](images/e8bdaedd2f6a17ecd9fc5fb5a7b164e363759594a357144669b424329508ca98.jpg)  
Figure 5. M4 admits a mechanism claim through statistical adjudication. Frozen test specifications INPUT <sup>M5</sup> <sup>·</sup> <sup>INTERVENE</sup> <sup>→</sup> <sup>REPAIR</sup> <sup>→</sup> <sup>VERIFY</sup> OUTPUTand case identifiers enter a locked runner; statistical and protocol-consistency gates jointly route each hypothesis to supported, refuted, or inconclusive.

## 3.5 M5: Mechanism Intervention, Repair, and Verification

Lastly, validated mechanisms are converted into interventions using a repair ladder ordered by implementation cost and invasiveness. The current repair ladder organization is shown in Table 3, the cost increases sequentially from L1 to L4.

20% 34% candidate AThe framework begins with the lowest repair tier capable of addressing the validated mechanism winner Band escalates when lower-cost interventions are inefective.

<sup>candidate</sup> <sup>C</sup> evaluator success Repair candidates are search runs on the exploration split $\mathcal { D } _ { E }$ <sup>both</sup> <sup>co</sup>(or optional $\mathcal { D } _ { H } )$ <sup>both</sup> <sup>wrong</sup>. Candidates from the lowest permitted tier are compared with the baseline on the same exploration cases. <sup>DECISION</sup> <sup>→</sup> <sup>→</sup> <sup>→</sup> <sup>→</sup>Then the best candidate and its success rule are frozen. The frozen candidate is then compared <sup>GATE DECISION</sup> <sup>FIELDS NEXT</sup>with the unchanged baseline through paired evaluation on identical cases drawn from the sealed confirmation split ${ \mathcal { D } } _ { C } ( { \mathrm { F i g u r e ~ 6 } } )$ . A repair is accepted only when its net efect satisfies the prespecified improvement criterion and does not introduce unacceptable degradation. An accepted intervention is then persisted, together with its supported diagnosis, tier, and paired evidence, as a verified repair recipe that can be reapplied to new cases or audited later.

Table 3. The repair ladder makes intervention cost and parameter change explicit. L1-L3 are non-parametric evolution because the checkpoint weights remain frozen; L4 is parametric evolution. The tierM4 · ISOLATED CONFIRMATION constrains the intervention space, not repair quality.What survives evidence that discover
<table><tr><td>Tier</td><td>Evolution path</td><td>Space</td><td>Candidate treatment</td></tr><tr><td>L1</td><td>Non-parametric</td><td>Input</td><td>Prompt rewrites, explicit constraints, output formats, and evidence-use instructions.</td></tr><tr><td>L2</td><td>Non-parametric</td><td>Scaffold</td><td>Deterministic multi-call workflows, preprocessing, external tools, conditional execution, and output aggregation around the unchanged model.</td></tr><tr><td>L3</td><td>Non-parametric</td><td>Internals</td><td>Read-only logits or attention, or audited inference-time interventions such as attention reweighting or activation steering, without</td></tr><tr><td>L4</td><td>Parametric</td><td>Parameters</td><td>persistent weight updates. Dataset construction and fine-tuning or adapter updates. These create a new checkpoint and require explicit human execution or approval.</td></tr></table>

![](images/3e31001b29e58f53bb479f9523f1660bd57f1df9525c7b1050dbb1a295464ba9.jpg)  
Figure 6. M5 promotes only repairs that beat the unchanged baseline under a paired test. A supported hypothesis first receives a causal probe; repair search then starts at the least invasive permitted tier, freezes a candidate and its success rule, and compares it with the unchanged baseline on matched cases. The decision is to ship, stop, try the next tier, or return filtered data to M1.

## 3.6 Verified Evolution and Iterative Improvement

EvalRx makes iterative improvement a sequence of auditable version decisions, not unconstrained self-modification. At iteration t, the system begins with the currently promoted version $v _ { t } ,$ , a frozen diagnostic hypothesis $h _ { t } ,$ and a candidate repair $r _ { t }$ . It promotes the candidate only when the diagnosis is supported by its frozen test, the repair improves the afected behavior in a held-out paired comparison, regressions remain within a predeclared bound, and the intervention stays within the approved repair ceiling:

$$
A _ { t } = \mathbf { 1 } \big [ \mathrm { V e r i f y } ( h _ { t } ) \big ] \mathbf { 1 } \big [ \Delta _ { \mathrm { t a r g e t } } ( r _ { t } ; v _ { t } ) > \delta _ { \mathrm { m i n } } \big ] \mathbf { 1 } \big [ R _ { \mathrm { n o n - t a r g e t } } ( r _ { t } ; v _ { t } , v _ { 0 } ) \leq \rho _ { \mathrm { m a x } } \big ] \mathbf { 1 } \big [ \mathrm { t i e r } ( r _ { t } ) \leq L _ { \mathrm { m a x } } \big ] .
$$

Here, $v _ { 0 }$ is the immutable root version and serving configuration, $\delta _ { \mathrm { m i n } }$ is the minimum paired target benefit, $\rho _ { \mathrm { m a x } }$ is the tolerated non-target regression, and $L _ { \mathrm { m a x } }$ is the approved intervention ceiling. Statistical evidence requirements, including significance or e-value criteria where applicable, are part of the verification and paired-benefit predicates.

The gate defines a conservative version transition:

$$
v _ { t + 1 } = \left\{ { \begin{array} { l l } { r _ { t } ( v _ { t } ) , } & { A _ { t } = 1 , } \\ { v _ { t } , } & { A _ { t } = 0 . } \end{array} } \right.
$$

An unsuccessful candidate therefore remains an auditable result rather than silently changing the deployed model. The immutable root $v _ { 0 }$ remains available for every later comparison, so a chain of local gains cannot conceal accumulated regression.

The repair tier determines which kind of model evolution is permitted. L1-L3 are nonparametric: they change prompts, scafolds, tools, or inference-time computation while checkpoint weights remain frozen. L4 is parametric: it creates a separately versioned adapter or checkpoint through fine-tuning. Both paths must pass the same hypothesis, paired-benefit, and regression gates; L4 additionally requires explicit human approval before a new parameterized artifact can be promoted.

Recursion occurs only after a supported promotion and fresh evidence allocation. Once $v _ { t + 1 }$ is promoted, it may enter a new M1-M5 cycle on a newly declared evaluation split or batch. A failed candidate instead records a rejected edge, stops, or escalates within $L _ { \mathrm { m a x } }$ . This is the operational sense in which EvalRx supports iterative model improvement: each iteration preserves the evidence, version lineage, and stopping reason required to audit the next one.

Algorithm 1. A guarded iterative version loop. A version can seed the next cycle only after frozen-contract diagnosis, held-out paired repair evaluation, and the approved intervention ceiling all pass.

INPUT immutable root $v _ { 0 } ,$ , fresh evaluation stream $\{ \mathcal { D } ^ { ( t ) } \}$ , candidate budget $B ,$ tier ceiling $L _ { \mathrm { m a x } } ,$   
and stop policy.   
INITIALIZE $t  0 ,$ promoted version $v _ { t }  v _ { 0 } .$   
1. Diagnose (M1-M2). Probe $v _ { t }$ on exploration cases and record failure modes plus diagnostic   
signals.   
2. Precommit (M3). Freeze executable hypothesis contracts before adjudication.   
3. Verify (M4). Test each contract against its declared efect, direction, and multiplicity family.   
4. Repair and compare (M5). For a supported contract, search at most $B$ candidates from the   
lowest permitted tier; compare target benefit and regressions with $v _ { t }$ and $v _ { 0 }$   
5. Decide. If $A _ { t } = 1$ , record $v _ { t } \stackrel { r _ { t } } { \longrightarrow } v _ { t + 1 }$ , allocate fresh evidence, and return to Step 1. Otherwise   
record the rejected edge and stop or escalate only within $L _ { \mathrm { m a x } }$   
OUTPUT version graph, health-card registry, rejected candidates, and explicit stopping reason.

Algorithm 1 makes the iterative claim testable. The original baseline $v _ { 0 }$ remains available for every later paired comparison, preventing a sequence of local gains from obscuring cumulative regression. A repair must be tested on cases not used to select it; when an iteration consumes its held-out split, the next iteration uses a fresh, pre-declared split or new evaluation batch. The intervention ceiling is carried forward, so the system may search within the non-parametric L1-L3 path but cannot silently cross into parametric L4 evolution. L4 parameter updates, ceiling increases, and deployment decisions require explicit human approval and a separately versioned checkpoint artifact.

The loop halts when no supported failure mode remains under the declared scope, when the next candidate is inefective or regressive, when the intervention ceiling or resource budget is reached, or when a human stops the investigation. Thus, iterative improvement produces a version graph of supported and rejected changes, not an unconstrained self-modifying system. The health-card registry reports each node, edge, evidence split, and stop reason.

## 4 Evaluation Scope and Setup

## 4.1 Task and Model Roster

The evaluation is designed to test whether the same evidence-to-repair protocol transfers across three distinct input modalities: Large Language Models (LLMs), Vision-Language Models (VLMs), and Audio-Language Models (ALMs). Each track uses two to four predeclared datasets and a fixed model roster across both datasets. For visual tasks, we select ChartQA [28], Spatial457 [39] and Chair [32] to diagnose model’s grounding and hallucination problems. As language reasoning tasks, we choose two BBH subtasks (word sorting and causal judgement) [37], GSM8K [8], and HotpotQA [40], covering algorithm execution, causal judgement, mathematical reasoning, and multi-hop question answering. ALMs track targets environmental-sound understanding, general audio reasoning, and music understanding using MMAU [33] and AudioCaps [19] datasets. Dataset versions, scoring rules, split seeds, and serving parameters are frozen in the run manifest before diagnosis begins. As shown in Table 4, we use both closed-source model Gemini and open-weight models including Qwen, Gemma and Nemotron ranging from 2B to 30B parameters as evaluation targets.

Table 4. A three-modality evaluation roster. The table lists the models in benchmark and the role of each LLM, VLM, and ALM track in evaluating EvalRx.
<table><tr><td>Task</td><td>Models</td><td>Role</td></tr><tr><td>LLM</td><td>Gemini-2.5-Flash-Lite, Qwen3.5 2B, Gemma E2B, Nemotron-3-Nano-4B</td><td>Four text datasets, cross-scale and cross-family comparison.</td></tr><tr><td>VLM</td><td>Gemini-2.5-Flash-Lite, Qwen3.5 2B, Gemma E2B; Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8</td><td>Three visual datasets, cross-scale and cross-family comparison.</td></tr><tr><td>ALM</td><td>Gemini-2.5-Flash-Lite, Qwen3-Omni-30B-A3B-Instruct, Gemma E2B, Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8</td><td>Two audio datasets, cross-family comparison.</td></tr></table>

## 4.2 Serving and Inference Configuration

The evaluation targets are the models in Table 4; diagnostic or coding agents are auxiliary pipeline components, not additional models in the comparison. EvalRx records the identity and version of every auxiliary agent, scorer, analyzer, prompt, decoding setting, and serving configuration in the run manifest. This separation makes the unit of a health-card conclusion unambiguous: a specific model revision under a specific serving configuration on a specific dataset.

For the initial pipeline configuration, we use Claude Code with Opus-5 as the base orchestration agent and additional Antigravity with Gemini-3.1-Pro to ensure that the system works with diferent kinds of agents.

## 5 Empirical Evaluation

## 5.1 Quantitative Results

We evaluate EvalRx as a diagnosis-and-repair system rather than as a score-optimization procedure. The quantitative study asks whether verified repairs improve the afected behavior across model families, modalities, and agent backbones.

LLM experiments across families. Figure 7 summarizes the improvement for the LLM track across four benchmarks. Overall, accuracy improved in 14 of the 16 model–task combinations. The average improvement is 4.7 percentage points (pp), although the magnitude of the gains varied substantially across tasks. The largest individual improvements are observed for Gemini-2.5-Flash-Lite on word sorting (+16.0 pp) and GSM8K (+10.8 pp), and Gemma-4-E2B on HotpotQA (+10.0 pp). Moreover, Qwen3.5-2B and Gemma-4-E2B improve consistently across all four benchmarks, demonstrating the general applicability of EvalRx.

![](images/b88aa63339479c1de9382d9f5616919f1a9236275f3b77927748e890fe9ea11a.jpg)  
Figure 7. Verified repairs across the LLM benchmark track. Each panel compares baseline and post-repair performance; labels report the panel-specific change in percentage points.

VLM experiments across families. Figure 8 reports the accuracy of four vision-language models before and after applying EvalRx. Performance improves in 10 of the 12 model–task combinations. The mean improvement across all settings is 7.6%. However, the gains are strongly task-dependent: the average improvements on ChartQA and Chair were 10.8 pp and 10.4 pp, respectively, whereas Spatial457 shows an average gain of only 1.6 pp. Gemini and Qwen improve on all three benchmarks, while Gemma and Nemotron remain unchanged on Spatial457.

![](images/80c74ffde7517debe2658532f58ab88e69a620022c9f3b77bea202b54e9fdc4b.jpg)  
Figure 8. Verified repairs across the VLM benchmark track. The figure compares baseline and post-repair performance using Claude Opus-5 as the base agent.

Table 5. Repair gains across the Gemma-4 scale range. The table reports baseline accuracy and post-repair improvement for each Gemma-4 model-task setting.
<table><tr><td>Model</td><td>BBH (word sorting)</td><td>gsm8k</td><td>MMAU</td><td>AudioCaps-Hallucination</td><td>ChartQA</td><td>Spatial457</td></tr><tr><td>Gemma-4-E2B</td><td> $5 6 . 5 \% \Delta + 3 . 2 \%$ </td><td> $8 7 . 6 \% \Delta + 2 . 0 \%$ </td><td> $3 6 . 3 \% \Delta + 2 0 . 3 \%$ </td><td> $5 6 . 6 \% \Delta \lambda + 0 . 0 \%$ </td><td> $4 8 . 4 \% ~ \Delta  2 1 . 9 \%$ </td><td> $1 8 . 4 \% \ \Delta \mathrm { { + } 0 . 0 \% }$ </td></tr><tr><td>Gemma-4-E4B</td><td> $6 2 . 0 \% ~ \Delta { + } 2 1 . 6 \%$ </td><td> $9 1 . 4 \% \ \Delta \mathrm { { + 0 . 0 \% } }$ </td><td> $6 5 . 3 \% \Delta + 1 2 . 0 \%$ </td><td> $6 1 . 7 \% \Delta + 1 6 . 7 \%$ </td><td> $5 8 . 3 \% \Delta + 2 6 . 0 \%$ </td><td> $1 6 . 7 \% \Delta + 2 2 . 7 \%$ </td></tr><tr><td>Gemma-4-12B</td><td> $8 2 . 4 \% \Delta + 1 1 . 2 \%$ </td><td> $9 3 . 4 \% \Delta \lambda { + } 1 . 2 \%$ </td><td> $6 3 . 0 \% \Delta + 1 6 . 7 \%$ </td><td> $5 4 . 0 \% \Delta + 1 8 . 7 \%$ </td><td> $7 3 . 7 \% \Delta + 1 7 . 3 \%$ </td><td> $1 6 . 3 \% \Delta + 1 9 . 3 \%$ </td></tr></table>

ALM experiments across families. Figure 9 reports the corresponding audio-language results under the same agent configuration. Performance improves in six of the eight model–benchmark combinations. These results suggest that the efect of EvalRx is strongly dependent on both the underlying model and the evaluated capability. In particular, Gemma exhibits a substantial improvement in general audio understanding on MMAU but no improvement in audio hallucination detection. Nemotron shows the opposite pattern, obtaining only a small gain on MMAU but a considerably larger improvement on AudioCaps. This complementary behavior indicates that EvalRx addresses diferent model-specific failure modes rather than providing a uniform improvement in audio-language understanding.

![](images/6ac968045e43ac054a76a8816eac92f12e5a295d386e3b847f21fb7f5a5af552.jpg)  
Figure 9. Verified repairs across the ALM benchmark track. The figure compares baseline and post-repair performance using Claude Opus-5 as the base agent.

Additional agent-backbone experiment. Figure 10 repeats the LLM comparison with Antigravity (Gemini-3.1-Pro) as the base agent. Accuracy improves in 12 of the 16 model–task combinations, with an average gain of 5.1 pp; the largest improvements are observed for Gemma-4-E2B (+16.2 pp) and Nemotron-3-Nano-4B (+13.6 pp) on word sorting, and Qwen3.5-2B on GSM8K (+11.2 pp). Similar to using Claude as the base agent, Antigravity can also efectively repair potential issues in the model in most cases.

![](images/ba2d9922ff5ab7218e8eb4e586b818b08054dac1f9867bb898fd9d7683b001cc.jpg)  
Figure 10. LLM repair outcomes with a second agent backbone. The figure compares baseline and post-repair performance using Antigravity Gemini-3.1-Pro as the base agent.

Efectiveness across model scales within the Gemma-4 family. We evaluate two datasets each for vision-language, text-only language and audio-language tasks using Gemma-4 models at 2B, 4B, and 12B parameters. This setting tests whether the workflow transfers across model scale

rather than being tied to one checkpoint.

As shown in Table 5, EvalRx is efective across all three model sizes, yielding average accuracy gains of 7.9, 16.5, and 14.1 percentage points for the E2B, E4B, and 12B models, respectively. Performance improves for most model–dataset pairs, with three settings remaining unchanged: E4B on GSM8K and E2B on AudioCaps-Hallucination and Spatial457. The gains are particularly substantial on the multimodal benchmarks, reaching up to 26.0 percentage points. The smaller improvements on GSM8K are likely due to its already high baseline accuracy, which leaves limited room for further improvement. Overall, these results suggest that the efectiveness of EvalRx is not tied to a particular model scale and generalizes across models with diferent capacities.

## 5.2 Case Studies

The following traces make the diagnosis-to-repair process inspectable. They report the discovered behavior, hypothesis verdict, the selected repair tier, and the paired case-level outcome. Crucially, the three cases have diferent evidentiary status: ChartQA and BBH word sorting demonstrate mechanism-verified repairs, whereas MMAU demonstrates a measured repair whose mechanism remains unresolved. We retain both outcomes because the system should not turn an accuracy gain into an unsupported causal explanation. Because the content generated using Claude as the base agent has poor readability, we use Antigravity as the base agent in the qualitative study.

![](images/c1df94bfadeb517d998f9a6c00bd2601bf65f3934ef1fe273d62bc2722720439.jpg)  
Figure 11. A complete, mechanism-verified repair loop on ChartQA. For Gemma-4-E2B, the L2 scafold is promoted only after its hypotheses are supported under frozen tests and its paired outcome improves from 49.2% to 79.7%.

ChartQA: a complete evidence-to-repair loop. Figure 11 is the positive end-to-end example. From 26 measurements, the discovery stage forwards nine signals for screening; one survives the Benjamini-Hochberg gate $( \Delta = 0 . 4 4 , \mathrm { C I } \ [ 0 . 2 8 , 0 . 5 9 ] , p = 9 . 7 \mathrm { e } - 7 )$ . The resulting calculation-error and cross-dataset-consistency hypotheses are supported under frozen tests. The repair search then keeps the least invasive successful intervention: the L2 transcribe then refine zoom scafold. On the held-out evaluation, accuracy rises from 49.2% to 79.7%, with 45 cases fixed and six broken. This trace is the intended verification-gated loop: the intervention is promoted together with the evidence that supports its diagnosis and bounds its regressions.

![](images/244b9d203978fd2363363d0aa7b21ee71152d4a05f29b12a834e9b59cf78b212.jpg)  
Figure 12. A mechanism-verified L1 repair on BBH (word sorting). For Nemotron-3-Nano-4B-FP8, the supported strict-copy intervention improves held-out accuracy from 12.0% to 25.6%.

BBH (word sorting): a verified repair with prompt optimization. Figure 12 presents a verified repair. From 44 measurements, the discovery stage forwards 15 signals for screening, of which seven survive the Benjamini-Hochberg gate. The strongest finding is sensitivity to semanticpreserving restatement $( \Delta = 0 . 6 6 , \mathrm { C I } \ [ 0 . 3 8 , 0 . 9 0 ] , p = 1 . 5 \mathrm { e } - 6 )$ . Adjudication supports all three resulting hypotheses: degenerate repetition, prior-driven brittleness, and decoding instability. The repair search then retains the least invasive successful intervention, the L1 strict copy constraint prompt, which directly targets the supported repetition and copy-fidelity failure. On held-out cases, accuracy increases from 12.0% to 25.6%, with 22 cases fixed and five broken.

![](images/d583b8bc433aa9b428c825f5be87203158823b30aeb6017ad6eb37880cd25faa.jpg)  
Figure 13. An empirical repair that remains mechanistically unresolved on MMAU. For Gemma-4-E2B, the L2 scafold improves held-out accuracy from 43.0% to 60.2%, but no mechanism reaches the supported verdict; the result is therefore not presented as a confirmed mechanism repair.

MMAU: an empirical repair pending diagnosis. Figure 13 illustrates the complementary boundary case. The system detects and screens candidate signals, but no hypothesis reaches the supported verdict: two remain inconclusive, and the language-prior hypothesis is refuted. So M5 stage can only take inconclusive hypotheses into consideration. Nevertheless, inspired by inconclusive hypotheses, the L2 verify then commit scafold produces a measured held-out improvement from 43.0% to 60.2%, with 30 cases fixed and eight broken. We therefore report this result as an empirical repair rather than a mechanism-verified closed-loop repair. Its value is operational and diagnostic: it identifies a useful intervention while preserving the speculative mechanism, the rejected alternatives, and the path for a subsequent investigation.

## 6 Discussion and Future Directions

EvalRx treats a low benchmark score as the beginning of an investigation rather than its conclusion. Its unit of analysis is an auditable health card: a reproducible failure slice, a frozen mechanism hypothesis, hypothesis verdict, held-out paired evidence, a repair attempt, and a paired account of benefits and regressions. This makes negative outcomes useful. A candidate may fail because the discovered pattern does not generalize, the mechanism is unsupported, the repair is inefective, or the repair harms previously correct cases; these are scientifically diferent conclusions and should remain distinguishable.

The current study has clear limitations. It does not claim that every failure mode is repairable or that every statistically supported association establishes a complete causal explanation. Public benchmark contamination, finite probe coverage, imperfect grading of free-form outputs, servingstack dependence, and incomplete access to model internals limit the conclusions that can be drawn. The multiple datasets design in each modality provides a regression panel, but it is not a substitute for a comprehensive safety evaluation or deployment assessment.

The release is designed to make these limitations inspectable. It will include case manifests, split definitions, model and serving configurations, prompt and configuration hashes, analyzer traces, frozen hypothesis contracts, raw result records, repair trials, and failed reproductions. These artifacts allow another group to reproduce a reported health card, challenge its mechanism claim, or test a new repair without changing the evaluation contract. A health card is evidence for technica review and monitoring; it is not, by itself, a claim of safety certification or regulatory compliance.

Trade-ofs of an optional validation set. Although a cleaner design would use separate validation split for M4 hypothesis verification and M5 repair candidates selection, we don’t introduce an additional held-out split in the present experiments. The validation sets contain relatively few examples from each discovered failure slice, making it dificult for genuine efects to reach the prespecified evidentiary thresholds. In pilot runs, M4 verification in held-out data already limits the number of supported mechanisms available to guide repair generation because of limited samples. Using the same small validation set for candidate selection would introduce an additional low-power decision: a candidate would first need to outperform alternatives on validation data and then demonstrate improvement again on the final test data. Candidate rankings on such small validation samples can be dominated by sampling variability, even when the candidate pool is unchanged. We therefore select repair candidates using the exploration data and reserve unseen test cases for final confirmation. This choice improves selection power but may favor interventions tailored to the diagnosed examples. With larger validation sets, held-out repair selection would be preferable because it would provide stronger protection against adaptive overfitting.

Toward recursive self-improvement. The next frontier is to turn individual repair recipes into cumulative, evidence-guided model improvement. The present study evaluates at most one promoted repair per model–task setting; sustained improvement across generations remains an empirical question. Building on Algorithm 1, future work will repeat the M1–M5 cycle on each promoted version, using fresh cases for final repair confirmation and paired comparisons against both the parent version and the immutable root v<sub>0</sub>. The key question is whether successive repairs compose: do gains accumulate, plateau as the remaining failures become harder, or interfere with behaviors repaired earlier? These experiments will also examine when non-parametric interventions at L1–L3 should give way to human-approved L4 adaptation, tracking cumulative accuracy, regressions, and inference cost rather than isolated score gains.

A second direction is to make repair recipes reusable across tasks and model versions. A recipe registry could support retrieval and adaptation of previously successful interventions, while requiring fresh evaluation before a transferred repair is accepted. Comparing diagnosis-guided search with direct repair search under matched agent and compute budgets would establish when explicit diagnosis improves repair quality or search eficiency. This would turn the audit trail into a resource for learning which failure signatures predict useful interventions, without assuming that a successful repair establishes its proposed mechanism.

The longer-term ambition is recursive self-improvement of the diagnostic system itself. Beyond repairing a target model, EvalRx could use recorded successes and failures to propose changes to its analyzers, hypothesis-generation policies, and repair-search strategies. Such changes should be evaluated on held-out model–task settings under an externally fixed acceptance protocol, rather than allowing the improving system to redefine its own success criterion. The goal is a system that becomes better at finding and repairing failures over time, with each accepted change remaining traceable to an executable intervention and independently evaluated benefit.

## 7 Conclusion

We introduce EvalRx, an agentic system that turns model evaluation into statistically grounded model diagnosis and repair. By separating exploratory discovery from frozen hypothesis adjudication and held-out paired repair verification, EvalRx turns model errors into measurable signals, statistically grounded hypotheses, and verified repair recipes. By separating exploration from final test and evaluating repairs against an unchanged baseline, it provides a systematic and reproducible path from failure discovery to model improvement. The benchmark tests this workflow across Gemini, Qwen, Gemma, and Nemotron models, with two to four datasets in each modality track. The central outcome is not a larger score alone, but a traceable answer to which behavior failed, what evidence supports the diagnosis, whether the proposed repair generalized beyond the cases that motivated it and which fix level path was taken to repair model while saving costs.

## References

[1] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alex Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. Gepa: Reflective prompt evolution can outperform reinforcement learning. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 8479–8565, 2026.

[2] Samuel GZ Asher, Janet Malzahn, Jessica M Persano, Elliot J Paschal, Andrew CW Myers, and Andrew B Hall. Do claude code and codex p-hack? sycophancy and statistical analysis in large language models. Sycophancy and Statistical Analysis in Large Language Models (February 19, 2026), 2026.

[3] Shir Ashury-Tahan, Yifan Mai, Elron Bandel, Michal Shmueli-Scheuer, and Leshem Choshen. Errormap and erroratlas: Charting the failure landscape of large language models, 2026.

[4] Joachim Baumann, Paul Röttger, Aleksandra Urman, Albert Wendsjö, Flor Miriam Plaza del Arco, Johannes B. Gruber, and Dirk Hovy. Large language model hacking: Quantifying the hidden risks of using llms for text annotation, 2025.

[5] Jun Shern Chan, Neil Chowdhury, Oliver Jafe, James Aung, Dane Sherburn, Evan Mays, et al. MLE-bench: Evaluating machine learning agents on machine learning engineering. In International Conference on Learning Representations (ICLR), 2025.

[6] Muxi Chen, Chenchen Zhao, and Qiang Xu. Hibug2: Eficient and interpretable error slice discovery for comprehensive model debugging. In International Conference on Learning Representations, volume 2025, pages 102566–102590, 2025.

[7] Yeounoh Chung, Tim Kraska, Neoklis Polyzotis, Ki Hyun Tae, and Steven Euijong Whang. Slice finder: Automated data slicing for model validation. In 2019 IEEE 35th International Conference on Data Engineering (ICDE), pages 1550–1553. IEEE, April 2019.

[8] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.

[9] Lisa Dunlap, Krishna Mandal, Trevor Darrell, Jacob Steinhardt, and Joseph E. Gonzalez. VibeCheck: Discover and quantify qualitative diferences in large language models. In International Conference on Learning Representations (ICLR), 2025.

[10] Sabri Eyuboglu, Maya Varma, Khaled Saab, Jean-Benoit Delbrouck, Christopher Lee-Messer, Jared Dunnmon, James Zou, and Christopher Ré. Domino: Discovering systematic errors with cross-modal embeddings. In International Conference on Learning Representations (ICLR), 2022.

[11] Shantanu Ghosh, Rayan Syed, Chenyu Wang, Vaibhav Choudhary, Binxu Li, Clare B Poynton, Shyam Visweswaran, and Kayhan Batmanghelich. LADDER: Language-driven slice discovery and error rectification in vision classifiers. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 22935–22970, Vienna, Austria, July 2025. Association for Computational Linguistics.

[12] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, et al. Accelerating scientific discovery with co-scientist. Nature, pages 1–3, 2026.

[13] Peter Grünwald, Rianne de Heide, and Wouter Koolen. Safe testing. Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(5):1091–1128, March 2024.

[14] Zexue He, Marco Tulio Ribeiro, and Fereshte Khani. Targeted data generation: Finding and fixing model weaknesses. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8506–8520, Toronto, Canada, July 2023. Association for Computational Linguistics.

[15] Kexin Huang, Ying Jin, Ryan Li, Michael Y. Li, Emmanuel Candes, and Jure Leskovec. Automated hypothesis validation with agentic sequential falsifications. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 25372–25437. PMLR, 13–19 Jul 2025.

[16] Yue Huang, Zhengzhe Jiang, Yuchen Ma, Yu Jiang, Xiangqi Wang, Yujun Zhou, Yuexing Hao, Kehan Guo, Pin-Yu Chen, Marzyeh Ghassemi, Stefan Feuerriegel, and Xiangliang Zhang. ProbeLLM: Automating principled diagnosis of LLM failures. In Forty-third International Conference on Machine Learning, 2026.

[17] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR), 2024.

[18] Been Kim, John Hewitt, Neel Nanda, Noah Fiedel, and Oyvind Tafjord. Because we have llms, we can and should pursue agentic interpretability, 2025.

[19] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, 2019.

[20] Younghyun Kim, Sangwoo Mo, Minkyu Kim, Kyungmin Lee, Jaeho Lee, and Jinwoo Shin. Discovering and mitigating visual biases through keyword explanation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11082–11092, June 2024.

[21] Xiang Lisa Li, Neil Chowdhury, Daniel D. Johnson, Tatsunori Hashimoto, Percy Liang, Sarah Schwettmann, and Jacob Steinhardt. Eliciting language model behaviors with investigator agents, 2025.

[22] Haokun Liu, Sicong Huang, Jingyu Hu, Yangqiaoyu Zhou, and Chenhao Tan. Hypobench: Towards systematic and principled benchmarking for hypothesis generation, 2026.

[23] Xiao Liu, Zirui Wu, Xueqing Wu, Pan Lu, Kai-Wei Chang, and Yansong Feng. Are LLMs capable of data-based statistical and causal reasoning? benchmarking advanced quantitative reasoning with data. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of

the Association for Computational Linguistics: ACL 2024, pages 9215–9235, Bangkok, Thailand, August 2024. Association for Computational Linguistics.

[24] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery, 2024.

[25] Cong Lu, Shengran Hu, and Jef Clune. Automated capability discovery via foundation model self-exploration, 2025.

[26] Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. Discoverybench: Towards data-driven discovery with large language models. In International Conference on Learning Representations, volume 2025, pages 4556–4579, 2025.

[27] Akshay Manglik, Apaar Shanker, Kaustubh Deshpande, Jason Qin, Yash Maurya, Veronica Chatrath, Vijay S. Kalmath, Levi Lentz, and Yuan Xue. Insights generator: Systematic corpus-level trace diagnostics for llm agents, 2026.

[28] Ahmed Masry, Jia Qing Tan, Shafiq Joty, Enamul Hoque, et al. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022.

[29] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 17359–17372. Curran Associates, Inc., 2022.

[30] Shikhar Murty, Christopher Manning, Scott Lundberg, and Marco Tulio Ribeiro. Fixing model bugs with natural language patches. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11600–11613, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics.

[31] Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with “gradient descent” and beam search. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7957–7968, Singapore, December 2023. Association for Computational Linguistics.

[32] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 4035–4045, 2018.

[33] Sakshi Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. In International Conference on Learning Representations, volume 2025, pages 84929–84964, 2025.

[34] Sarah Schwettmann, Tamar Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, and Antonio Torralba. Find: A function description benchmark for evaluating interpretability methods. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 75688–75715. Curran Associates, Inc., 2023.

[35] Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, and Antonio Torralba. A multimodal automated interpretability agent. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 44293–44321. PMLR, 21–27 Jul 2024.

[36] Nimit Sohoni, Jared Dunnmon, Geofrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 19339–19352. Curran Associates, Inc., 2020.

[37] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed H Chi, Denny Zhou, et al. Challenging bigbench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051, 2023.

[38] Ruodu Wang and Aaditya Ramdas. False discovery rate control with e-values. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(3):822–852, 2022.

[39] Xingrui Wang, Wufei Ma, Tiezheng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. Spatial457: A diagnostic benchmark for 6d spatial reasoning of large multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[40] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.

[41] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic "diferentiation" via text, 2024.

[42] Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi, and Pang Wei Koh. Evaltree: Profiling language model weaknesses via hierarchical capability trees. In Second Conference on Language Modeling, 2025.

[43] Ruiqi Zhong, Charlie Snell, Dan Klein, and Jacob Steinhardt. Describing diferences between text distributions with natural language. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 27099–27116. PMLR, 17–23 Jul 2022.

[44] Ruiqi Zhong, Peter Zhang, Steve Li, Jinwoo Ahn, Dan Klein, and Jacob Steinhardt. Goal driven discovery of distributional diferences via language descriptions. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 40204–40237. Curran Associates, Inc., 2023.

[45] Yangqiaoyu Zhou, Haokun Liu, Tejes Srivastava, Hongyuan Mei, and Chenhao Tan. Hypothesis generation with large language models. In Lotem Peled-Cohen, Nitay Calderon, Shir Lissak, and Roi Reichart, editors, Proceedings of the 1st Workshop on NLP for Science (NLP4Science), pages 117–139, Miami, FL, USA, November 2024. Association for Computational Linguistics.

[46] Yizhang Zhu, Shiyin Du, Boyan Li, Yuyu Luo, and Nan Tang. Are large language models good statisticians? In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 62697–62731. Curran Associates, Inc., 2024.