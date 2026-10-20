# Beyond Leaderboards: A Method for Application-Specific Capability Profiling of Large Language Models

Oleg Grynets   
EPAM Systems   
McLean, Virginia, USA   
oleg\_grynets@epam.com

Dmytro Koziuberda EPAM Systems Kyiv, Ukraine dmytro\_koziuberda@epam.com

Olena Pochernina EPAM Systems Lviv, Ukraine olena\_pochernina@epam.com

Viktoriia Tsymbal EPAM Systems Lviv, Ukraine viktoriia\_tsymbal@epam.com

Ivan Fedorchuk EPAM Systems Vinnytsia, Ukraine ivan\_fedorchuk@epam.com

Vasyl Lyashkevych EPAM Systems Lviv, Ukraine vasyl\_lyashkevych@epam.com

Dmytro Kostetskyi EPAM Systems Lviv, Ukraine dmytro\_kostetskyi@epam.com

Alona Seletska EPAM Systems Kyiv, Ukraine alona\_seletska@epam.com

Daryna Tukalo   
EPAM Systems   
Kyiv, Ukraine   
daryna\_tukalo@epam.com

Hryhorii Petrenko EPAM Systems Kharkiv, Ukraine hryhorii\_petrenko@epam.com

Andrii Salyk   
EPAM Systems   
Lviv, Ukraine   
andrii\_salyk@epam.com   
Oleh Kaskun   
EPAM Systems   
Lviv, Ukraine   
oleh\_kaskun@epam.com   
Oleksii Ilchuk   
EPAM Systems   
Lviv, Ukraine   
oleksii\_ilchuk@epam.com   
Bohdan Saukh   
EPAM Systems   
Lviv, Ukraine   
bohdan\_saukh@epam.com

Abstract—Public leaderboards rank large language models (LLMs) in general, but an engineer has to choose a model for a specific application, architecture and cost. This paper presents a method for application-specific capability profiling of LLMs and its realization as a containerized testbed. The method treats each application as a unit of extension: an Oracle-to-PostgreSQL migration agent, a documentation-grounded chatbot or an enterprise-support automation agent runs in its own container behind a small contract, brings its own dataset, evaluator and owned attributes, and is onboarded together with its dashboards. Measurements accumulate in a versioned, sparse metamodel of every model. The method separates the model under test from judges and agents, freezes and hashes plans, refuses to compare values from different contexts, diagnoses the instrument before the system, and applies a null-agent check and a noise rule whose operating characteristics are quantified by simulation. In the evaluated automation conditions, model choices produced larger differences on selected observable outcomes than framework choices, although the experimental design does not establish a general causal ranking of these factors. The experiments also show that instruction tuning is deployment-specific and that system-reported success can exceed independently observed success: of 216 migration units checked against a real Oracle database, 108 were demonstrably correct. These measurements establish application-specific observations, not a universal model ranking.

Keywords—large language models, capability profiling, model selection, metamodel, diagnostics, benchmarking, testbed, Grafana, LLM agents, database migration

## I. INTRODUCTION

Choosing a large language model (LLM) for an enterprise application is still, in most teams, an exercise in reading leaderboards and trusting vendor claims. Public benchmarks such as MMLU [1], BIG-bench [2], HELM [3] and Chatbot Arena [4] answer the question “how good is this model in general?”, but an engineer who has to migrate a legacy database, operate a documentation assistant or automate a support queue asks a different question: “how good is this model for this task, in this architecture, at this cost?” The two questions diverge more than is commonly assumed. In our own experiments a configuration change that improved intent classification by 4.6 percentage points on one deployment lowered it by 3.4 points on a second and had no effect on a third, so that a tuning result measured on a weak deployment described the deployment rather than the prompt. Similarly, in an automation agent that reasons over Jira, Confluence and Outlook events, the choice of the model changed the amount of work actually performed (mean state-changing effects per run) between 0.5 and 1.8, while the choice of the agent framework moved the same quantity only between 0.4 and 0.7.

This paper presents a method for application-specific capability profiling of LLMs and the containerised testbed that realises it. The method is designed to make such differences visible, measurable and comparable. It profiles LLMs by running them through reference applications: small but complete prototypes of the application classes that matter to the authors’ organisation, namely an

Oracle-to-PostgreSQL migration agent, a documentation-grounded chatbot and an enterprise-support automation agent. Every application is packaged as a separate container with an evaluation interface. The testbed executes models against datasets and application configurations, records versioned measurements, and exposes them through an analytical interface and through Grafana dashboards. The accumulated measurements form a sparse, versioned metamodel of each LLM: a profile of attributes, each measured in the same way for every model, which can be queried to decide which model should do which task at what cost.

The method is organised in eight steps, from defining an application and onboarding it to deciding on a model and monitoring the profile (Section IV). Its central design decision is that an application is a unit of extension of the testbed, not just a workload. When a new application is added, it brings its own dataset, its own evaluator and its own attribute declarations, and onboarding registers the application in the probe registry, in the metric catalogue and in the dashboards. As a consequence the set of LLM features that the testbed can talk about grows with the set of applications it hosts: a chatbot adds retrieval and citation attributes, an automation agent adds tool-use and side-effect attributes, a migration agent adds behavioural-equivalence attributes. Section V describes the mechanism; Section VIII reports what the first three applications have revealed, together with a simulation of the decision rule and a statistical re-analysis of the reported counts.

## A. Problem Statement and Research Questions

We study the following problem. Given a growing set of LLMs and a growing set of application classes, build an experimental infrastructure that (i) measures every model on every application under controlled and fully recorded conditions, (ii) keeps the measuring instrument separate from the object measured, (iii) refuses to report numbers that were produced under different conditions as if they were comparable, and (iv) makes the result usable for model selection. The work is organised around four research questions.

RQ1. What architecture allows heterogeneous reference applications to be added to an LLM-profiling testbed without modifying its core, while preserving reproducibility and cost transparency?

RQ2. Which measurement and diagnostic principles are necessary so that the numbers produced by such a testbed can be trusted, given that agents, judges and applications are themselves LLM-driven, and how should a failure be diagnosed: in the instrument first, or in the system?

● RQ3. What do the first three reference applications reveal about the dependence of model behaviour on task, configuration and framework?

RQ4. How can the accumulated measurements be turned into a decision aid for model selection, and which claims must it refuse to make?

## B. Contributions

The paper makes six contributions. First, it states a method for application-specific capability profiling in eight steps (Section IV).

Second, it formalises an LLM metamodel as a sparse partial function from attributes to measurements with explicit context and plan identity, and defines the comparability relation, the evidence levels and a capability taxonomy used throughout (Section III).

Third, it presents the architecture of the testbed with its application-container contract, onboarding algorithm, deterministic plan gate, three-way cost separation and dashboard registration, and it provides a reference implementation of the computational core (Section V).

Fourth, it describes three reference applications and their evaluation stands in sufficient detail to be reproduced (Section VI).

Fifth, it states the measurement and diagnostic principles distilled from defects that the authors found in their own stands, including the instrument-first diagnostic ladder, the degenerate-agent check and a three-state noise rule whose operating characteristics are quantified by Monte Carlo simulation (Section VII).

Sixth, it presents the measured results, their statistical re-analysis, and the resulting evidence and capability maps (Sections VIII and IX).

The methodological contribution is the operational coupling of context-aware measurement classes, evidence-constrained verdicts and instrument-first diagnosis with extensible application probes. The measured evidence is preliminary and does not establish a universal ranking of models or a causal ordering of model and framework effects.

The paper distinguishes measurement validity, repeatability, and generalisability. Some comparisons involve small or single-run samples, and some outcome measures lack complete behavioural ground truth; these boundaries are identified in the experiments and Section IX-J. The study produces conditional capability profiles rather than a context-free ranking of evaluated models.

The remainder of the paper is organised as follows. Section II reviews related work. Section III formalises the problem. Section IV states the method. Section V describes the testbed architecture. Section VI presents the reference applications. Section VII states the evaluation and diagnostic methodology. Section VIII reports experiments and Section IX discusses them. Section X concludes.

## II. RELATED WORK

We reviewed 88 scholarly sources, including peer-reviewed publications and preprints, and organised the literature into six areas: general-purpose benchmarking, application-level evaluation, LLM-as-a-judge, benchmark validity and statistical analysis, cost-aware model selection, and engineering practices for observable machine learning systems. Table I summarises how the testbed relates to each line.

## A. General-Purpose Benchmarks and Holistic Evaluation

Large multi-task suites measure broad knowledge and reasoning. MMLU [1] and AGIEval [5] use examination-style questions, BIG-bench [2] collects hundreds of community-contributed tasks, and HELM [3] argues for holistic reporting over many scenarios and seven metric families, including calibration, robustness and efficiency, rather than accuracy alone. Human-preference platforms such as Chatbot Arena [4] and judge-based protocols such as MT-Bench [6] complement static suites with open-ended comparisons. Surveys [7] show that the evaluation literature is dominated by such general-purpose protocols. Behavioural testing in the CheckList style [8] and dynamic benchmark construction [9] address the fragility of static test sets. These works define what to measure in general; the testbed adopts their emphasis on multiple metrics and on explicit scenarios, but replaces the “scenario” with a complete reference application so that the measured quantity is the behaviour of a model inside an architecture.

Two lessons from this line of work shape the testbed more than any specific benchmark. The first is that a single aggregate score hides the structure of model behaviour: HELM reports accuracy next to calibration, robustness, fairness, bias, toxicity and efficiency for the same scenarios precisely so that trade-offs become visible [3], and BIG-bench documents tasks on which scale alone does not produce a smooth improvement [2]. The second is that preference-based and judge-based protocols inherit properties of their instruments. Pairwise voting at scale is informative about open-ended chat quality [4], yet it cannot say whether a model completes a database migration correctly, and a benchmark that ranks models on examination questions says little about their behaviour inside a retrieval pipeline. For these reasons the testbed does not attempt to compete with general leaderboards; it assumes that a team has already shortlisted models from them and asks what happens next.

## B. Application-Level Evaluation: Retrieval, Agents, Code, SQL

Retrieval-augmented generation [10] is evaluated with component-level retrieval metrics on benchmarks such as BEIR [11] and with generation-level criteria such as faithfulness and answer relevance in automated frameworks [12], [13]. RGB [14] diagnoses noise robustness, negative rejection and information integration, and positional effects in long contexts are documented in [15]. Dense retrieval [16] and rank fusion [17] are the building blocks of the retrieval configurations that our chatbot stand compares.

Agent evaluation has moved from single-turn function calling [18], [19] to interactive environments: AgentBench [20], WebArena [21], GAIA [22] and SWE-bench [23] score the end state or the task outcome in realistic environments, and surveys catalogue architectures and evaluation practices [24], [25]. Reasoning-and-acting loops [26], self-reflection [27] and experience-based learning [28] are the mechanisms around which many agents are built, while multi-agent frameworks [29], [30] supply orchestration patterns that we treat as an experimental variable. Our automation stand follows the end-state logic of these benchmarks but adds a distinction that they rarely make explicit: whether a claim of success in the final answer is backed by an observable effect.

For code and data tasks, Spider [31] and BIRD [32] evaluate text-to-SQL by execution, TransCoder [33] and CodeTransOcean [34] address program translation, and EvalPlus [35] shows that weak test suites overstate functional correctness. A study of bugs introduced during LLM-based code translation [36] and two broad surveys of LLMs in software engineering [37], [38] frame the migration problem. Classical text-similarity metrics such as BLEU [39] and chrF [40] are still used as reference-based signals, but our migration results, like [35], indicate that a reference-based or acceptance-based verdict can overstate correctness; execution against the source system is the stronger instrument.

What these application-level works share is that they evaluate a task in a fixed environment. For retrieval, the environment is the corpus and the index; for agents, a web or code environment; for SQL, a database. The application-level testbed generalises the idea in two directions. It treats the environment as one of the things that can be versioned and registered (the dataset version and the configuration are part of every measurement), and it makes the claim–effect distinction explicit: AgentBench and WebArena score the end state, but an agent that reports success without an observable effect is not separately counted in them, whereas in our automation stand the share of such runs is itself an attribute. The same reasoning applies to code translation: the studies of translation bugs [36] show that models introduce subtle semantic differences, and our finding that a clean install proves little is the database-specific form of the EvalPlus lesson that weak tests overstate correctness [35].

## C. LLM-as-a-Judge

Judge models scale open-ended evaluation [6], [41], [42], yet they exhibit position and verbosity effects [43] and favour their own generations [44]. Our stands therefore use judges only for criteria that cannot be computed deterministically, keep them in a separate role from the model under test, and measure their instability directly: in the migration direction a semantic judge changed its verdict on 16.2% of repeated items at temperature 0. Leakage-resistant evaluation of judges on specifications [45] and execution-judged benchmarks [46] are earlier results from the same group that inform the design of the judge roles.

Judge-based evaluation has a further cost dimension that is rarely reported: the judge’s own spend. In the P1 stand, 1,802 combined judge calls cost more than twice as much as the conversion of the same files, which is why the testbed accounts for it separately. Learned judges such as Prometheus [42] and structured prompting such as G-Eval [41] improve agreement with human raters but do not remove the dependence on the instrument, and the evidence of self-preference [44] suggests that the judge should come from a different family than the model under test whenever the comparison is between vendors. In our chatbot stand both the judge and the controller currently use the same model family as one of the compared models, a limitation we record in Section IX-J.

## D. Benchmark Validity, Contamination and Statistics

The testbed reduces contamination exposure through versioned, application-specific test data and isolation of expected outcomes. Synthetic or private datasets alone do not guarantee freedom from data contamination; that remains an evaluation-specific threat to validity.

The statistical literature points in one direction: with few items and few repeats, effects smaller than the noise cannot be established. Dror et al. recommend matching the test to the data and reporting the variability of the evaluation itself [53], and Card et al. show that typical NLP comparisons have too little power to detect the differences they report [54]. Contamination studies add that the same model may have seen the test items, so that a high score can be memory rather than capability [49], [50]. The testbed responds in two ways. Versioned private and synthetic application datasets, together with the separation of ground-truth labels from model inputs, reduce contamination exposure. However, these safeguards do not establish that evaluation data are entirely absent from model training or development histories. For the variability, it declines to publish model numbers until repeats have established the spread, and it applies the coarse noise-floor rule of Section VII-F where a formal test is not applicable.

## E. Model Selection, Routing and Multi-Objective Trade-of s

Cost-aware use of LLMs has been studied as cascades [59] and as quality-aware routing between a cheaper and a stronger model [60]. Selecting among models with several conflicting objectives is a multi-objective problem whose standard solution concept is the Pareto set [61], [62]. The analytical interface of the testbed computes Pareto sets over measured attributes on the server and displays the evidence count behind each claim; our migration agent uses a deterministic complexity score for routing in the spirit of [60], and our token-optimisation study [63] shows that cost and quality must be evaluated jointly.

## F. Engineering Practice for Observable ML Systems

Machine-learning systems accumulate hidden technical debt [64], and rubrics for production readiness explicitly include tests of data, model and infrastructure [65]. A case study of software engineering for ML at scale [66] documents the importance of data management and model evaluation tooling. Programmable guardrails [67] and prompt-programming frameworks [68] address safety and prompt optimisation, respectively. Dashboards for monitoring evolving information systems, including ontology-driven platforms with dashboard-centric analytics, have been investigated by co-author V. Lyashkevych [69]–[73]. The risk perspective on LLM-based development [74], the notion of a software functional state [75] and the orchestration of multi-agent systems with dynamic involvement of new agents [76] inform, respectively, the security attributes, the lifecycle view and the extensibility of the testbed. The metamodel idea itself builds on a unified architecture metamodel for information systems generated by AI [77]. Finally, a comparative analysis of frameworks for the same three application classes [78] was the starting point for the reference applications used here.

## G. Prompt Programming and Reasoning Strategies

The behaviour of a model inside an application depends on how it is prompted. Few-shot prompting [79], chain-of-thought prompting [80] and interleaved reasoning and acting [26] changed what models can do without changing their weights, and declarative frameworks such as DSPy [68] compile pipelines of model calls and optimise their prompts against a metric. The testbed treats the prompt variant as a registered, hashed object, requires a baseline arm next to any compared variant and, in the programme scheme, optimises prompts per probe so that a model is measured under the prompt that suits it and the gain over the baseline prompt is itself recorded. Our measurement that a description rewrite had effects of opposite signs on three deployments is consistent with the prompt-sensitivity literature, and it adds a practical corollary: a prompt optimised against one deployment is a property of the pair, not of the prompt, so the optimisation target must be part of the identity that is stored with the result. DSPy-style optimisation against a metric also inherits the risk that the metric rewards the wrong behaviour; the null-agent check of Section VII-B is the guard that we apply before any loop is pointed at a number.

## H. Specification-Driven Development and Code Regeneration

Specification-driven development treats a specification as the operational artefact consumed by LLM development agents. The authors’ earlier studies formalise a specification as a structure of semantic blocks with dependency relations, evaluate it with execution-judged benchmarks that keep the implementer panel fixed and include a no-specification control arm [46], evaluate the evaluators of specifications under sealed ground truth [45], and show on database migration that cost, validity and semantic preservation must be traded off [63]. The generation hook of the testbed reuses this machinery at a smaller scale: the application specification is the artefact from which an adapter is generated, and the smoke test and the control-arm discipline are the checks that the generated adapter must survive. The unified architecture metamodel [77] offers a representation (JSON, PlantUML and Mermaid views of architectural diagrams) that can serve as the structured context for such generation.

## I. Dashboards and Monitoring of Evolving Systems

Operational dashboards for ML systems are often built last, as a view over logs. The earlier work of co-author V. Lyashkevych on intelligent monitoring of evolving information systems treats the dashboard as a first-class element of the architecture, with ontology-driven domain knowledge integrated into dashboard-centric analytics [69], context-aware selection of decision strategies [70], evolution-aware synthesis of monitoring systems by LLMs [71], genetic optimisation of monitoring models for LLM-modified systems [73] and multi-drift predictive monitoring [72]. We adopt this stance: the dashboards of the testbed are registered with the application at onboarding, not added afterwards. A natural next step, discussed in Section IX-D, is to monitor the metamodel itself for drift.

TABLE I. POSITIONING OF THE TESTBED AGAINST RELATED LINES OF WORK
<table><tr><td rowspan=1 colspan=1>Line of work</td><td rowspan=1 colspan=1>Typical unit</td><td rowspan=1 colspan=1>What the testbed adds</td></tr><tr><td rowspan=1 colspan=1>General benchmarks[1]-[3]</td><td rowspan=1 colspan=1>task item</td><td rowspan=1 colspan=1>complete application asscenario; per-applicationattributes</td></tr><tr><td rowspan=1 colspan=1>Preference platforms [4],[6]</td><td rowspan=1 colspan=1>pairwise vote</td><td rowspan=1 colspan=1>execution-based anddeterministic metrics</td></tr><tr><td rowspan=1 colspan=1>RAG evaluation[12]-[14]</td><td rowspan=1 colspan=1>answer /retrieval</td><td rowspan=1 colspan=1>model as variable under fixedretrieval; failure localisation</td></tr><tr><td rowspan=1 colspan=1>Agent benchmarks [20],[21], [23]</td><td rowspan=1 colspan=1>task outcome</td><td rowspan=1 colspan=1>claims versus observed effects;framework as variable</td></tr><tr><td rowspan=1 colspan=1>Code and SQLbenchmarks [31], [32],[34]</td><td rowspan=1 colspan=1>snippet /query</td><td rowspan=1 colspan=1>execution against sourcesystem; fix ledger</td></tr><tr><td rowspan=1 colspan=1>Validity and statistics[47], [53], [54]</td><td rowspan=1 colspan=1>guidelines</td><td rowspan=1 colspan=1>enforced in code: comparability,noise floor, fail-closed</td></tr><tr><td rowspan=1 colspan=1>Routing and cascades[59], [60]</td><td rowspan=1 colspan=1>query</td><td rowspan=1 colspan=1>measured profile of each modelas routing evidence</td></tr></table>

Entries paraphrase the cited works; the right-hand column states the contribution of the present testbed.

## J. Intelligent Diagnostics, Knowledge Bases and Testing

The diagnostic stance of this paper has an older root in the co-authors’ earlier research on intelligent diagnostics of computer facilities, which treated diagnosing as a knowledge-based task with its own infrastructure: an ontology of the diagnostic subject domain [81], a software tool for building the knowledge base of the diagnosing process [82], retrieval of diagnostic information from microprocessor devices [83], the search for features of diagnostic information and the development of a search service for it [84], [85], criteria for setting correspondence between the terms of different knowledge bases [86], and logic-textual and neural approaches to information search [87]. Four ideas from this line carry over to the profiling of LLMs. First, diagnosis needs an explicit subject-domain vocabulary; in the testbed it is the attribute catalogue, in which every attribute has an owner, a unit and a direction. Second, diagnostic knowledge should be kept as a knowledge base separate from the diagnosed system; the fix ledger of the migration direction and the failure taxonomies of the other directions play this role (Section VII-K). Third, terms must be matched across knowledge bases with explicit criteria; the same quantity appears as recall at K in one probe and as a recall metric in another, and the catalogue must decide when two names denote one attribute. Fourth, retrieval of similar earlier failures is part of diagnosis, which is the practical reason for storing per-unit outcomes and not only aggregates.

Testing under uncertainty has a second root in the quantitative treatment of software risks, where Monte Carlo simulation is used to evaluate the values of risk attributes [88] and where the functional state of software through its life cycle has been formalised [75]. We use the same technique in Section VIII-I to study how often the decision rule of the testbed declares an effect when there is none and how large a real difference must be to be detected. A third relevant observation comes from literature on agent orchestration reviewed in the programme: a subtask graph fixed at design time can raise retry cost, because a failure upstream forces the re-execution of everything downstream, whereas executable control flow that isolates the failing subtask retries only that subtask. The group-level repair loop of the migration agent and the triage of the probe agent follow the same principle of isolating the failure before retrying.

Table II condenses the lessons drawn from the reviewed literature into design decisions. Each row names the decision, the works that motivate it and the mechanism in which it is realised, so that the paper’s architecture can be read as an application of the literature rather than a collection of ad hoc choices.

TABLE II.  
DESIGN LESSONS DRAWN FROM THE REVIEWED LITERATURE
<table><tr><td rowspan=1 colspan=1>Lesson</td><td rowspan=1 colspan=1>Motivating works</td><td rowspan=1 colspan=1>Realisation in the testbed</td></tr><tr><td rowspan=1 colspan=1>report several metric families, not one score</td><td rowspan=1 colspan=1>[2], [3], [7]</td><td rowspan=1 colspan=1>per-application attribute sets; operational set common to all probes</td></tr><tr><td rowspan=1 colspan=1>behavioural tests expose what accuracy hides</td><td rowspan=1 colspan=1>[8], [9]</td><td rowspan=1 colspan=1>behavioural differential against real Oracle; scenario-based chatbot checks</td></tr><tr><td rowspan=1 colspan=1>execution is stronger evidence than text similarity</td><td rowspan=1 colspan=1>[31], [32], [35], [36]</td><td rowspan=1 colspan=1>verification levels; demonstrably correct units; effects of tool calls</td></tr><tr><td rowspan=1 colspan=1>retrieval and generation must be evaluated separately</td><td rowspan=1 colspan=1>[10]-[13]</td><td rowspan=1 colspan=1>retrieval-only runner; six metric groups; causal gating</td></tr><tr><td rowspan=1 colspan=1>noise robustness and negative rejection matter</td><td rowspan=1 colspan=1>[14], [15]</td><td rowspan=1 colspan=1>citation, refusal and empty-retrieval attributes</td></tr><tr><td rowspan=1 colspan=1>agents are scored on end state, not on prose</td><td rowspan=1 colspan=1>[20]-[23]</td><td rowspan=1 colspan=1>frozen world state; expected tool sequences and final states</td></tr><tr><td rowspan=1 colspan=1>tool use is a distinct capability</td><td rowspan=1 colspan=1>[18], [19], [26]</td><td rowspan=1 colspan=1>tool selection, argument validity, hallucinated-call rate</td></tr><tr><td rowspan=1 colspan=1>judges are biased and must be separated</td><td rowspan=1 colspan=1>[6], [43], [44]</td><td rowspan=1 colspan=1>separate roles; judge spend counted apart; judge churn measured</td></tr><tr><td rowspan=1 colspan=1>benchmarks can fail as instruments</td><td rowspan=1 colspan=1>[47], [48]</td><td rowspan=1 colspan=1>null-agent check; evidence levels; fix ledger</td></tr><tr><td rowspan=1 colspan=1>contamination inflates scores</td><td rowspan=1 colspan=1>[49]-[52]</td><td rowspan=1 colspan=1>private or synthetic datasets; sealed ground truth</td></tr><tr><td rowspan=1 colspan=1>significance claims are fragile</td><td rowspan=1 colspan=1>[53], [54]</td><td rowspan=1 colspan=1>noise-floor rule; repeats before publication</td></tr><tr><td rowspan=1 colspan=1>reproducibility needs identity</td><td rowspan=1 colspan=1>[55], [56]</td><td rowspan=1 colspan=1>plan hash; prompt identity; frozen metric libraries</td></tr><tr><td rowspan=1 colspan=1>document the context of every result</td><td rowspan=1 colspan=1>[57], [58]</td><td rowspan=1 colspan=1>dataset version, configuration, warnings stored with each run</td></tr><tr><td rowspan=1 colspan=1>cost and quality are traded off</td><td rowspan=1 colspan=1>[59]-[62]</td><td rowspan=1 colspan=1>Pareto view with evidence counts; cost separation</td></tr><tr><td rowspan=1 colspan=1>ML systems accumulate hidden debt</td><td rowspan=1 colspan=1>[64]-[66]</td><td rowspan=1 colspan=1>tests without network; pinned metric ruler; orphan recovery</td></tr><tr><td rowspan=1 colspan=1>agent frameworks supply orchestration patterns</td><td rowspan=1 colspan=1>[24], [29], [30]</td><td rowspan=1 colspan=1>framework as experimental variable behind one contract</td></tr><tr><td rowspan=1 colspan=1>experience can improve agents, but must be validated</td><td rowspan=1 colspan=1>[27], [28], [68]</td><td rowspan=1 colspan=1>self-learning as hypotheses; approval and immutable snapshots</td></tr><tr><td rowspan=1 colspan=1>guardrails are programmable but measurable</td><td rowspan=1 colspan=1>[67]</td><td rowspan=1 colspan=1>guardrail precision, recall and false-block rate measured offline</td></tr></table>

In summary, existing work provides the measuring instruments and the warnings. What is missing, and what this paper offers, is an extensible infrastructure in which every application brings its own instruments, all measurements share one identity and comparability discipline, and the result is a profile of each model rather than a rank in a single list.

## III. PROBLEM FORMULATION AND REQUIREMENTS

This section fixes the terminology and the formal objects used in the rest of the paper. The notation is summarised in Table III.

## A. Formal Model

Let M be a set of models and A a set of reference applications. An application $a \in A$ is a tuple

$$
a = ( D _ { \mathrm { a } } , K _ { \mathrm { a } } , E _ { \mathrm { a } } , \Sigma _ { \mathrm { a } } ) ,\tag{1}
$$

where $D _ { \mathrm { a } }$ is a family of versioned datasets, $K _ { \mathfrak { a } }$ is a configuration space (retrieval settings, framework, prompt variant, model roles, evaluation method), $E _ { \mathrm { a } }$ is the evaluator that produces measurements, and $\Sigma _ { \mathrm { a } }$ is the set of attributes that the application declares and owns. A probe $p _ { \mathrm { a } }$ exposes the application to the testbed: it names the modes in which the application can be run, the prompt slots that can be varied and the attributes it measures.

A run fixes a model m, an application a, a dataset version $d ,$ a configuration $\kappa \in K _ { \mathrm { a } }$ and a plan identity h, the hash of the frozen plan that produced the run. A measurement is the tuple

For a declared comparison, partition the recorded configuration into the evaluated treatment (for example, the tested model or prompt variant) and an evaluation context κE containing the evaluator and metric-library versions, scoring protocol, fixed decoding and application settings, and any other factors required to be held constant. A plan hash h identifies the canonical execution plan; it does not itself establish equality of evaluation contexts.

$$
\mu = ( m , \sigma , \nu , n , \kappa , d , h ) , \sigma \in \Sigma _ { \mathrm { a } } ,\tag{2}
$$

with value v computed from n samples. The metamodel of model m is the partial function

$$
\Psi ( m ) \colon \sigma \mapsto \{ ( \nu _ { \mathrm { i } } , n _ { \mathrm { i } } , \kappa _ { \mathrm { i } } , d _ { \mathrm { i } } , h _ { \mathrm { i } } ) \} ,
$$

$$
\mathrm { d e f i n e d \ o n l y \ o n \ m e a s u r e d \odot } .\tag{3}
$$

Partiality is essential. An attribute that has not been measured is absent; it is never replaced by zero or by an average, and each view of the metamodel lists the attributes that are missing. Two values of the same attribute are comparable if they were produced in the same context:

$$
\mathsf { V } _ { 1 } \sim \mathsf { V } _ { 2 } \Leftrightarrow \mathsf { O } _ { 1 } = \mathbb { O } _ { 2 } \wedge \mathsf { K } \mathsf { E } ( \mathsf { K } _ { 1 } ) = \mathsf { K } \mathsf { E } ( \mathsf { K } _ { 2 } ) \wedge \mathsf { d } _ { 1 } = \mathsf { d } _ { 2 } . ( 4 )
$$

The analytical interface applies (4) before it draws any comparison and displays a warning when a selection mixes contexts (Section V-H). Model selection is expressed over a set of objectives $O = \{ ( o _ { 1 } , \delta _ { 1 } ) , . . . , ( o _ { \mathrm { k } } , \delta _ { \mathrm { k } } ) \}$ with directions δ ∈ {max, min}. Model $m _ { 1 }$ dominates $m _ { 2 }$ on O if it is at least as good on every objective and strictly better on one, and the recommended set is the non-dominated set:

The treatment dimension is declared in advance: models or prompt variants may differ between arms, but the evaluation context, dataset, scoring protocol and other fixed factors must match. Equality is tested on registered identifiers and evaluator versions; a changed metric library starts a new comparison class.

$$
{ \mathrm { P } } ( O ) = \{ m \in M ^ { \prime } : \lnot \exists m ^ { \prime } \in M ^ { \prime } , m ^ { \prime } \succ _ { 0 } m \} ,\tag{5}
$$

where $M ^ { \prime }$ is the set of models that satisfy the hard constraints of the task (for instance a minimum accuracy threshold) and for which every objective is measured in a comparable context.

TABLE III. NOTATION
<table><tr><td rowspan=1 colspan=1>Symbol</td><td rowspan=1 colspan=1>Meaning</td></tr><tr><td rowspan=1 colspan=1>M,A</td><td rowspan=1 colspan=1>set of models, set of reference applications</td></tr><tr><td rowspan=1 colspan=1> $a = ( D , K , E , \Sigma )$ </td><td rowspan=1 colspan=1>application: datasets, configurations, evaluatorowned attributes</td></tr><tr><td rowspan=1 colspan=1> $\big | \underline { { p _ { \mathrm { a } } } }$ </td><td rowspan=1 colspan=1>probe that exposes application a to the testbed</td></tr><tr><td rowspan=1 colspan=1> $\underline { { \vert \sigma \in \Sigma _ { \mathrm { { a } } } } }$ </td><td rowspan=1 colspan=1>attribute owned by application a</td></tr><tr><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>run context: mode and configuration identifier</td></tr><tr><td rowspan=1 colspan=1>h</td><td rowspan=1 colspan=1>hash of the frozen plan (identity of ameasurement)</td></tr><tr><td rowspan=1 colspan=1>μ</td><td rowspan=1 colspan=1>measurement: model, attribute, value, samplecount, context, dataset, plan</td></tr><tr><td rowspan=1 colspan=1>Ψ(m)</td><td rowspan=1 colspan=1>metamodel of model m (partial function overattributes)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>comparability relation</td></tr><tr><td rowspan=1 colspan=1>P(O)</td><td rowspan=1 colspan=1>Pareto set for objectives O</td></tr><tr><td rowspan=1 colspan=1> $\big \vert C _ { \mathrm { t } } , C _ { \mathrm { i } } , C _ { \mathrm { g } }$ </td><td rowspan=1 colspan=1>spend of tested model, judge and agent model</td></tr></table>

## B. Formal Properties

Three simple properties follow from the definitions and justify the mechanisms of Section V. They are stated with short proofs because the mechanisms rely on them.

Lemma 1 (comparability). On measurements of one attribute, the relation (4) is an equivalence relation. Proof. It is defined by equality of σ, the evaluation-context projection κE(κ), and d; equality is reflexive, symmetric and transitive, and a conjunction of equivalence relations is an equivalence relation. Consequently measurements fall into context classes, and any comparison, ranking or Pareto set is well defined within one class and meaningless across classes. Lemma 2 (plan identity). Let the plan hash be computed over the list of runs after sorting by a total order on runs. Then two plans that contain the same runs in any order have the same hash, and a rerun that replays the stored plan has the hash of the original. Proof. Sorting maps every permutation of a multiset of runs to the same list, and a hash of the same list is the same value. Replay reads the stored row, which is the sorted list. Lemma 3 (non-emptiness). If the set M′ of eligible models is finite and non-empty, the Pareto set P(O) of Equation (5) is non-empty. Proof. Dominance is irreflexive and transitive, hence a strict partial order; a finite non-empty set has a maximal element under a strict partial order, and a maximal element is not dominated.

Lemma 1 applies within a declared comparison design whose treatment and evaluation-context coordinates have been registered in advance. Lemma 3 concerns exact Pareto dominance only. Uncertainty bands are reported as supplementary pairwise annotations and do not replace the transitive dominance relation used to construct P(O); approximate ties need not be transitive.

## C. Evidence Levels and Demonstrable Correctness

The formal model above says what a measurement is; it does not say how much a measurement proves. Experience with the migration direction showed that the same word, “accepted”, can denote evidence of very different strength, so we attach an evidence level λ to every verdict. The levels are ordered by what the instrument could in principle detect (Table IV): a statement by the system itself, a static check of the output, installation or execution in a target environment, comparison with a human reference, and comparison with the behaviour of the real source system. A claim made at level λ is only as strong as the weakest level at which every unit that contributes to it was checked, and a report must state the denominator at each level.

TABLE IV. EVIDENCE LEVELS
<table><tr><td rowspan=1 colspan=1>Level</td><td rowspan=1 colspan=1>Instrument</td><td rowspan=1 colspan=1>Detects</td><td rowspan=1 colspan=1>Misses</td></tr><tr><td rowspan=1 colspan=1>L0</td><td rowspan=1 colspan=1>system&#x27;s own report</td><td rowspan=1 colspan=1>nothingindependent</td><td rowspan=1 colspan=1>everything</td></tr><tr><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>static check of theoutput</td><td rowspan=1 colspan=1>syntax, lint,structure</td><td rowspan=1 colspan=1>unresolved references;semantics</td></tr><tr><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=1>install or execute inthe target</td><td rowspan=1 colspan=1>load and runtimeerrors</td><td rowspan=1 colspan=1>wrong but validbehaviour</td></tr><tr><td rowspan=1 colspan=1>L3</td><td rowspan=1 colspan=1>reference comparison(text, tree, judge)</td><td rowspan=1 colspan=1>divergence from areference</td><td rowspan=1 colspan=1>behaviour equal to adifferent validreference</td></tr><tr><td rowspan=1 colspan=1>L4</td><td rowspan=1 colspan=1>behaviour of the realsource system</td><td rowspan=1 colspan=1>behaviouraldifferences</td><td rowspan=1 colspan=1>inputs not exercised</td></tr></table>

L0–L4 generalise the verification levels of the migration agent (CHECKED\_STATIC, VERIFIED\_INSTALL, VERIFIED\_BEHAVIOR) to the other applications.

Let $\mathrm { U L } 4 \subseteq \mathrm { U }$ contain only units for which the prescribed Oracle differential checks were actually executed. For $\mathrm {  ~ u ~ } \in \mathrm { U L } 4 _ { \mathrm {  ~ \Omega ~ } }$ , let acc(u) indicate the agent’s own acceptance verdict, con(u) a contradiction detected on the exercised L4 cases, and exp(u) a failure that reproduces the source-system error under the recorded test.

Operational limitation: in the underlying migration stand, “demonstrably correct” denotes the following L4-screened operational count, not a proof of semantic equivalence over all possible inputs. A candidate can remain uncontradicted on the exercised Oracle tests while still diverging on untested cases. The count and the tested denominator must therefore accompany every claim.

$$
\begin{array} { c } { { \mathrm { N d c } = | \{ \mathbf { u } \in \mathbf { U L } 4 : [ \mathbf { a c c } ( \mathbf { u } ) \wedge \neg \mathbf { c o n } ( \mathbf { u } ) ] \vee } } \\ { { \lbrack \mathbf { \neg ~ a c c } ( \mathbf { u } ) \wedge \mathbf { e x p } ( \mathbf { u } ) ] \} | . } } \end{array}\tag{6}
$$

so that a unit counts both when it is accepted and not contradicted, and when it fails in exactly the way the source system fails. The definition applies only to UL4 and reports agreement on the executed checks, not exhaustive equivalence; for units that the stand cannot test the verdict is “not testable”, which is a separate category rather than a failure. The equation also shows why the agent’s own count of accepted units can rise while Ndc rises less: some accepted units are contradicted, and some failures are correct.

## D. A Capability Taxonomy for Mapping

The word capability is used loosely in the LLM literature; for a testbed it must be operational. We define a capability as a named behaviour of a model that is observable in at least one reference application, is measured by at least one declared attribute at a stated evidence level, and can differ between models. The capability map of a model is then the set of its capabilities with the measured values and evidence levels, which is exactly the metamodel Ψ(m) read through the taxonomy. Table V lists the ten capabilities that the first three applications make measurable, with the attributes through which they are observed and the largest variation that the reports show. The taxonomy is not exhaustive and is not a claim about how model capabilities are organised internally; it is a bookkeeping device that lets us state which questions the testbed can currently answer.

TABLE V. CAPABILITIES MADE MEASURABLE BY THE REFERENCE APPLICATIONS
<table><tr><td rowspan=1 colspan=1>Capability</td><td rowspan=1 colspan=1>Observed through</td><td rowspan=1 colspan=1>Application</td><td rowspan=1 colspan=1>Level</td><td rowspan=1 colspan=1>Largest variation reported</td></tr><tr><td rowspan=1 colspan=1>C1 behavioural fidelity oftranslation</td><td rowspan=1 colspan=1>demonstrably correct units;defects; faithful raises</td><td rowspan=1 colspan=1>migration</td><td rowspan=1 colspan=1>L4</td><td rowspan=1 colspan=1>108 of 216 units; 25 of 114 accepted wrong or unverifiable</td></tr><tr><td rowspan=1 colspan=1>C2 whole-file outputcompleteness</td><td rowspan=1 colspan=1>refusals; file efficiency;abbreviation of large files</td><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>L3</td><td rowspan=1 colspan=1>about 69 large files abbreviated by one model; refusals8.32%</td></tr><tr><td rowspan=1 colspan=1>C3 grounded answering</td><td rowspan=1 colspan=1>faithfulness; citation validity andcompliance</td><td rowspan=1 colspan=1>chatbot</td><td rowspan=1 colspan=1>L1-L3</td><td rowspan=1 colspan=1>correctness 0.977–0.995 across finalists</td></tr><tr><td rowspan=1 colspan=1>C4 retrieval use</td><td rowspan=1 colspan=1>recall, precision, MRR underfixed retrieval</td><td rowspan=1 colspan=1>chatbot</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>sparse recall 0.083 against 1.000 for dense (retrieval side)</td></tr><tr><td rowspan=1 colspan=1>C5 tool routing</td><td rowspan=1 colspan=1>tool selection; forbidden calls;router model effect</td><td rowspan=1 colspan=1>chatbot,automation</td><td rowspan=1 colspan=1>L1-L2</td><td rowspan=1 colspan=1>4 routing failures of 10 scenarios; 2 fixed by a router change</td></tr><tr><td rowspan=1 colspan=1>C6 tool-argument validity</td><td rowspan=1 colspan=1>argument correctness;hallucinated tool calls</td><td rowspan=1 colspan=1>chatbot,automation</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>two scenarios still failing on arguments after routing fix</td></tr><tr><td rowspan=1 colspan=1>C7 action honesty</td><td rowspan=1 colspan=1>unbacked success claims; effectrate</td><td rowspan=1 colspan=1>automation</td><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=1>0% to 55% of effect-free runs claim success</td></tr><tr><td rowspan=1 colspan=1>C8 instruction robustness</td><td rowspan=1 colspan=1>effect of an instruction rewrite</td><td rowspan=1 colspan=1>intentclassification</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>+4.6, -3.4, 0.0 pp on three deployments</td></tr><tr><td rowspan=1 colspan=1>C9 cost efficiency</td><td rowspan=1 colspan=1>tokens per run; cost per evidence</td><td rowspan=1 colspan=1>all</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>1,953 to 9,465 tokens per automation run</td></tr><tr><td rowspan=1 colspan=1>C10 run-to-run stability</td><td rowspan=1 colspan=1>variance over repeats; judge churn</td><td rowspan=1 colspan=1>chatbot, P1,judges</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>SD 0.013 to 0.037; judge churn 16.2%</td></tr></table>

Evidence levels as in Table IV; values from the experiments of Section VIII. C4 is a property of the retrieval configuration and is listed because the model-side effect is held fixed in those experiments.

Two points about the taxonomy matter for the rest of the paper. First, several capabilities are only visible in combination: a model’s grounded answering (C3) cannot be separated from the retrieval it is given (C4), which is why the chatbot stand uses a retrieval-only runner and holds the retrieval fixed when the model is varied. Second, the evidence level is part of the capability: C1 is measured at L4, but C3 is largely measured at L1 to L3 because no hard oracle exists for the quality of a free-text answer, in the sense of the generalisation caveat of the score-model decision record. Where a capability cannot be verified, we report it as such and do not substitute a judge and call it measured.

## E. Requirements

Starting from the questions in Section I and from the defects described in Section VII, we derived ten requirements. Table VI lists them with the mechanism that satisfies each one and the section in which it is described.

## F. Scope and Assumptions

The testbed measures models, not applications: the application is held fixed and the model is varied, or the application configuration is varied while the model is held fixed. The comparison of application architectures themselves is the subject of a separate study [78]. All model calls go through a single gateway, so that the models are reached in a uniform manner and cost can be attributed to the caller.

The programme that this work belongs to plans two further stages, requirement analysis with specification generation tailored to a chosen model, and specification decomposition with decision making about which model handles which part. They depend on stable profiles and are outside the scope of this paper.

TABLE VI. REQUIREMENTS AND THE MECHANISMS THAT SATISFY THEM
<table><tr><td rowspan=1 colspan=1>ID</td><td rowspan=1 colspan=1>Requirement</td><td rowspan=1 colspan=1>Mechanism</td><td rowspan=1 colspan=1>Sec.</td></tr><tr><td rowspan=1 colspan=1>R1</td><td rowspan=1 colspan=1>Add an application withoutchanging the core</td><td rowspan=1 colspan=1>probe contract; containeradapter</td><td rowspan=1 colspan=1>IV</td></tr><tr><td rowspan=1 colspan=1>R2</td><td rowspan=1 colspan=1>Every attribute has anowner and a unit</td><td rowspan=1 colspan=1>metric catalogue owned byprobes</td><td rowspan=1 colspan=1>IV</td></tr><tr><td rowspan=1 colspan=1>R3</td><td rowspan=1 colspan=1>Reproduce a measurementexactly</td><td rowspan=1 colspan=1>frozen, hashed plans; promptidentity</td><td rowspan=1 colspan=1>IV</td></tr><tr><td rowspan=1 colspan=1>R4</td><td rowspan=1 colspan=1>Tested model, judge andagents costed apart</td><td rowspan=1 colspan=1>three-way spend accounting</td><td rowspan=1 colspan=1>IV</td></tr><tr><td rowspan=1 colspan=1>R5</td><td rowspan=1 colspan=1>No metric is produced by amodel</td><td rowspan=1 colspan=1>agents propose, code computes</td><td rowspan=1 colspan=1>IV, VI</td></tr><tr><td rowspan=1 colspan=1>R6</td><td rowspan=1 colspan=1>Missing is not zero</td><td rowspan=1 colspan=1>partial metamodel; missing list</td><td rowspan=1 colspan=1>III</td></tr><tr><td rowspan=1 colspan=1>R7</td><td rowspan=1 colspan=1>Never compare acrosscontexts</td><td rowspan=1 colspan=1>comparability check in the UI</td><td rowspan=1 colspan=1>IV</td></tr><tr><td rowspan=1 colspan=1>R8</td><td rowspan=1 colspan=1>Outage is not a verdict</td><td rowspan=1 colspan=1>INCONCLUSIVE / absentverdict</td><td rowspan=1 colspan=1>VI</td></tr><tr><td rowspan=1 colspan=1>R9</td><td rowspan=1 colspan=1>Aggregate must beat thedegenerate agent</td><td rowspan=1 colspan=1>null-agent check</td><td rowspan=1 colspan=1>VI</td></tr><tr><td rowspan=1 colspan=1>R10</td><td rowspan=1 colspan=1>Observable by non-authors</td><td rowspan=1 colspan=1>dashboards, registration atonboarding</td><td rowspan=1 colspan=1>IV</td></tr></table>

## IV. THE PROFILING METHOD

The method answers a practical question: given a shortlist of LLMs, which of them is suited to a particular application, in a particular architecture, at what cost, and how sure can the team be of that answer? It does so by turning each application into an instrument and each model into a profile. The method has eight steps, M1 to M8 (Fig. 1, Table VII). The steps are ordered, but the method is iterative: a new application re-enters at M1, a new model at M3, and a diagnosis at M5 that blames the instrument sends the work back to M2 or M4.

![](images/7740730b98c4179cb30dadf32e31275c8ab525c51af6a43581d8a20b1e51ccb5.jpg)  
Dashed: a new application or a new model re-enters the method at M1 or M3.

Fig. 1. The eight steps of the profiling method. Solid arrows: the normal order; the dashed arrow: a new application or a new model re-enters the method.  
TABLE VII. STEPS OF THE METHOD, THEIR OUTPUTS AND THE GUARDS AGAINST KNOWN FAILURES
<table><tr><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>Activity</td><td rowspan=1 colspan=1>Output</td><td rowspan=1 colspan=1>Guard</td><td rowspan=1 colspan=1>Section</td></tr><tr><td rowspan=1 colspan=1>M1Define</td><td rowspan=1 colspan=1>fix the dataset, the configuration space, theevaluator and the owned attributes with unit anddirection</td><td rowspan=1 colspan=1>application specification</td><td rowspan=1 colspan=1>every attribute has an owner, a unit and adirection</td><td rowspan=1 colspan=1>VI</td></tr><tr><td rowspan=1 colspan=1>M2 Addprobe</td><td rowspan=1 colspan=1>package as a container, expose the contract, passthe smoke test, register probe, catalogue entriesand dashboards</td><td rowspan=1 colspan=1>probe identifier;dashboards</td><td rowspan=1 colspan=1>a generated adapter must pass the smoke test</td><td rowspan=1 colspan=1>V-E</td></tr><tr><td rowspan=1 colspan=1>M3 Plan</td><td rowspan=1 colspan=1>agents propose runs; the gate validates,canonicalises, hashes and freezes the plan</td><td rowspan=1 colspan=1>frozen plan with hash</td><td rowspan=1 colspan=1>scope re-applied by code; baseline armrequired</td><td rowspan=1 colspan=1>V-D</td></tr><tr><td rowspan=1 colspan=1>M4Execute</td><td rowspan=1 colspan=1>run under the caller&#x27;s key; code computes themetrics; tested model, judge and agents are costedapart</td><td rowspan=1 colspan=1>measurements withprovenance</td><td rowspan=1 colspan=1>prompt identity; fail-closed reporting</td><td rowspan=1 colspan=1>V-G</td></tr><tr><td rowspan=1 colspan=1>M5Inspect</td><td rowspan=1 colspan=1>test the instrument before the system, then locatethe first failing layer</td><td rowspan=1 colspan=1>diagnosis; verdict from afixed taxonomy</td><td rowspan=1 colspan=1>an outage is not a verdict; the null agent mustlose</td><td rowspan=1 colspan=1>VII-J</td></tr><tr><td rowspan=1 colspan=1>M6Compare</td><td rowspan=1 colspan=1>compare only values of one context class; applythe noise rule and evidence levels</td><td rowspan=1 colspan=1>effect, suggestive, or noresult</td><td rowspan=1 colspan=1>comparability; per-axis noise floor</td><td rowspan=1 colspan=1>VII-F</td></tr><tr><td rowspan=1 colspan=1>M7Decide</td><td rowspan=1 colspan=1>apply hard constraints, then the Pareto set withties and exclusions</td><td rowspan=1 colspan=1>recommendation withevidence counts</td><td rowspan=1 colspan=1>missing is not zero; context warning</td><td rowspan=1 colspan=1>VIII-M</td></tr><tr><td rowspan=1 colspan=1>M8Publish</td><td rowspan=1 colspan=1>show the profile on dashboards; monitor theprofile for drift</td><td rowspan=1 colspan=1>views; alerts</td><td rowspan=1 colspan=1>dashboard variables fix the context class</td><td rowspan=1 colspan=1>V-H</td></tr></table>

Sections give the place where the mechanism behind each step is described.

Three properties distinguish the method from a benchmark run. The unit ofevaluation is the application, not the item. A leaderboard item is a question; a unit here is a complete behaviour in an architecture, such as the migration of a group of Oracle objects, a chatbot turn with its retrieval and tool calls, or the handling of an enterprise event. The artefact is a profile, not a rank. The result of the method is the metamodel of a model: its measured attributes with their contexts, and an explicit list of what was not measured. The guards are part ofthe method. Each step carries a guard that exists because a stand once misled its authors without it (Table VII), and a result produced without its guards is reported with a warning instead of a number. Table VIII contrasts the method with a leaderboard on the dimensions that matter to a team that has to choose.

## A. Re-Entry Rules

What must be repeated when something changes is as important as what to do the first time (Table IX). The rules follow from the comparability relation: anything that changes a component of the context class opens a new class, and results from different classes are never merged.

TABLE VIII. LEADERBOARD VERSUS APPLICATION-SPECIFIC PROFILING
<table><tr><td rowspan=1 colspan=1>Dimension</td><td rowspan=1 colspan=1>Leaderboard</td><td rowspan=1 colspan=1>Profiling method</td></tr><tr><td rowspan=1 colspan=1>unit of evaluation</td><td rowspan=1 colspan=1>item or task</td><td rowspan=1 colspan=1>application in its architecture</td></tr><tr><td rowspan=1 colspan=1>question answered</td><td rowspan=1 colspan=1>how good is themodel in general</td><td rowspan=1 colspan=1>how good is the model for thisapplication, configuration andcost</td></tr><tr><td rowspan=1 colspan=1>artefact</td><td rowspan=1 colspan=1>rank or score</td><td rowspan=1 colspan=1>sparse metamodel withcontexts and missing list</td></tr><tr><td rowspan=1 colspan=1>comparability</td><td rowspan=1 colspan=1>same benchmarkversion</td><td rowspan=1 colspan=1>same attribute, context anddataset version, checked bycode</td></tr><tr><td rowspan=1 colspan=1>variables</td><td rowspan=1 colspan=1>model</td><td rowspan=1 colspan=1>model, prompt variant,framework, retrieval,configuration</td></tr><tr><td rowspan=1 colspan=1>failure handling</td><td rowspan=1 colspan=1>item scored wrong</td><td rowspan=1 colspan=1>failure diagnosed in instrumentor system; outage not a verdict</td></tr><tr><td rowspan=1 colspan=1>uncertainty</td><td rowspan=1 colspan=1>often single number</td><td rowspan=1 colspan=1>per-axis noise floor; evidencecounts; no result&#x27; allowed</td></tr><tr><td rowspan=1 colspan=1>cost</td><td rowspan=1 colspan=1>rarely reported</td><td rowspan=1 colspan=1>three spends, reportedseparately</td></tr><tr><td rowspan=1 colspan=1>extension</td><td rowspan=1 colspan=1>new benchmark</td><td rowspan=1 colspan=1>new application brings newattributes and dashboards</td></tr></table>

RE-ENTRY RULES OF THE METHOD
<table><tr><td rowspan=1 colspan=1>Change</td><td rowspan=1 colspan=1>Re-enter at</td><td rowspan=1 colspan=1>Consequence for existing results</td></tr><tr><td rowspan=1 colspan=1>new model</td><td rowspan=1 colspan=1>M3</td><td rowspan=1 colspan=1>existing results unchanged; newprofile with absent attributes filledby runs</td></tr><tr><td rowspan=1 colspan=1>new application</td><td rowspan=1 colspan=1>M1</td><td rowspan=1 colspan=1>new attributes in the catalogue;existing profiles list them as notmeasured</td></tr><tr><td rowspan=1 colspan=1>new dataset version</td><td rowspan=1 colspan=1>M3</td><td rowspan=1 colspan=1>new context class; no comparisonwith older versions withoutbridging runs</td></tr><tr><td rowspan=1 colspan=1>new configuration orprompt variant</td><td rowspan=1 colspan=1>M3</td><td rowspan=1 colspan=1>new context class; baseline armrequired for a prompt comparison</td></tr><tr><td rowspan=1 colspan=1>metric library upgrade</td><td rowspan=1 colspan=1>M2</td><td rowspan=1 colspan=1>new ruler; old and new results notcomparable</td></tr><tr><td rowspan=1 colspan=1>diagnosis blames theinstrument</td><td rowspan=1 colspan=1>M2 or M4</td><td rowspan=1 colspan=1>affected runs marked inconclusive;fix and rerun</td></tr><tr><td rowspan=1 colspan=1>drift of a hosted modeldetected</td><td rowspan=1 colspan=1>M3</td><td rowspan=1 colspan=1>re-run a frozen plan; compare as anew context class</td></tr></table>

A new model is run on the existing applications and fills the absent attributes of its own profile. A new application adds attributes to the catalogue and, until models are run on it, every profile lists them as not measured. An upgrade of a metric library changes the ruler itself, so the testbed freezes the libraries to the versions that produced the results and treats an upgrade as the opening of a new context class.

## V. TESTBED ARCHITECTURE

This section answers RQ1. We first give the overview and the design principles, then describe the metamodel and the programme scheme from which the testbed originates, the planes of the core, the life cycle of a goal, the application-container contract with its onboarding procedure, the specification-driven generation hook, the accounting of cost and prompts, and finally the dashboards. The core, called MAS Core (metamodel extraction pipeline), was implemented by D. Koziuberda, with the shared agent-integration layer contributed by H. Petrenko; the analytical interface and the three reference applications were built by the programme teams, as indicated in Section VI.

## A. Overview and Design Principles

Figure 2 shows the testbed. Reference applications live in their own containers on the left. MAS Core in the centre receives a goal (a statement of what to measure), plans runs with the help of agents, validates and freezes the plan with deterministic code, executes it through probes and writes measurements to a single store. Models under test, judges and the models that drive the agents are all reached through one LLM gateway, which keeps the cost of every role attributable. The store feeds the analytical interface, which is deterministic and contains no LLM in its analytics pages, and Grafana dashboards that are registered when an application is onboarded.

Five principles, listed in Table X, run through the whole design. They were not postulated in advance; each was introduced after a defect in an earlier stand showed what happens without it (Section VII).

TABLE X. DESIGN PRINCIPLES OF THE TESTBED
<table><tr><td rowspan=1 colspan=1>Principle</td><td rowspan=1 colspan=1>Why it matters</td><td rowspan=1 colspan=1>Where it is enforced</td></tr><tr><td rowspan=1 colspan=1>The instrument isnot the subject</td><td rowspan=1 colspan=1>judge or agent cost anderrors must not becomethe model&#x27;s</td><td rowspan=1 colspan=1>three spends reportedseparately; distinct modelroles</td></tr><tr><td rowspan=1 colspan=1>Agents decide,code computes</td><td rowspan=1 colspan=1>a model must not writeits own score</td><td rowspan=1 colspan=1>aggregator schema has nonumeric field; metrics aredeterministic code</td></tr><tr><td rowspan=1 colspan=1>Reproducibility</td><td rowspan=1 colspan=1>numbers are comparableonly if produced alike</td><td rowspan=1 colspan=1>plans frozen and hashed;prompts identified by hash;one change per experiment</td></tr><tr><td rowspan=1 colspan=1>Missing is notzero</td><td rowspan=1 colspan=1>averaged or zero-filledvalues hide gaps</td><td rowspan=1 colspan=1>partial metamodel; explicitlist of unmeasuredattributes</td></tr><tr><td rowspan=1 colspan=1>An outage is nota verdict</td><td rowspan=1 colspan=1>infrastructure failuresays nothing about themodel</td><td rowspan=1 colspan=1>INCONCLUSIVE status;absent verdict when anagent is unreachable</td></tr></table>

The five design principles in Table X capture the implementation safeguards introduced after faults in the evaluation infrastructure were identified experimentally.

## B. Programme Scheme and Metamodel

The proposed framework supports a three-stage research programme (Figs. 3 and 4). The implementation and experiments in this paper concern Stage 1: constructing a capability metamodel from application-specific probes. The two subsequent stages—task-aware specification generation and model assignment to specification components—are proposed consumers of the profile, not evaluated functions of the present testbed. A task is represented as T = {agents, operations, environments, modules, context, …}, consistent with the architectural metamodel [77].

![](images/ea1c322c690725816907097691d2543a7483f14f575d4b0811de13c048fb4c01.jpg)  
Solid arrows: data and control flow. Dashed arrows: model calls. Tested model, judge and agent model are separate roles.  
Fig. 2. Overview of the testbed: reference-application containers, MAS Core, measurement store, dashboards and the LLM gateway. Tested model, judge and agent model are separate roles.

![](images/5ef4c5f55645a4f610113911e8a5dfcb66faf92077ff9d2b0eb816a04090d242.jpg)  
Fig. 3. Model families and their sparse capability profiles, linked to application-owned probe attributes and operational measurements; T denotes structured task context.

![](images/e61617fc37b9384d0e9a9fbf2926b8f46c4947ea9fbe1e94b060e81f18f0d285.jpg)  
Fig. 4. Application probes and controlled prompt variants produce versioned capability measurements, which are aggregated into model profiles and exposed to analytical queries.

A metamodel is a sparse record per model, grouped by model family, whose attributes are the quantities that probes measure. Table XI lists the attribute groups of the scheme and the probes that own them. The numbering follows the scheme (P1 conversion, P2 migration, P3 chatbot, P4 automation); in the registry of the implementation the probes have identifiers such as p1\_o2pg and p2\_chatbot, which is a naming difference only. Attributes are owned by probes: conversion accuracy is an attribute of P1 and not a universal attribute of every model, and only a small operational set (tokens, cost, latency, completion) is collected identically for every probe. Figure 5 shows the resulting structure.

![](images/9928245fd69b1a763a26efbe0f82705205b7e627b05b280b3df42fc60758f8c0.jpg)  
Unmeasured attributes are absent, never zero. Planned: P5 security, P6 tokens, P7 resources, P8 DevOps.

Fig. 5. Structure of a model’s metamodel. Probe-owned attributes (P1–P4 implemented or integrated; P5–P8 planned) are measured in the same way for every model; the operational set is common.

TABLE XI. ATTRIBUTE GROUPS OF THE METAMODEL AND THEIR OWNERS
<table><tr><td rowspan=1 colspan=1>Attribute group</td><td rowspan=1 colspan=1>Probe</td><td rowspan=1 colspan=1>State</td></tr><tr><td rowspan=1 colspan=1>Conversion accuracy (Oracle,PostgreSQL)</td><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>measures</td></tr><tr><td rowspan=1 colspan=1>Migration accuracy (Oracle,PostgreSQL)</td><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>own stand; integrationplanned</td></tr><tr><td rowspan=1 colspan=1>Knowledge and retrieval, chatbot</td><td rowspan=1 colspan=1>P3</td><td rowspan=1 colspan=1>integrated (p2_chatbot)</td></tr><tr><td rowspan=1 colspan=1>Knowledge and retrieval,automation</td><td rowspan=1 colspan=1>P4</td><td rowspan=1 colspan=1>own stand; integrationplanned</td></tr><tr><td rowspan=1 colspan=1>Security treatment</td><td rowspan=1 colspan=1>P5</td><td rowspan=1 colspan=1>planned</td></tr><tr><td rowspan=1 colspan=1>Token efficiency</td><td rowspan=1 colspan=1>P6</td><td rowspan=1 colspan=1>planned</td></tr><tr><td rowspan=1 colspan=1>Index of required resources</td><td rowspan=1 colspan=1>P7</td><td rowspan=1 colspan=1>planned</td></tr><tr><td rowspan=1 colspan=1>DevOps</td><td rowspan=1 colspan=1>P8</td><td rowspan=1 colspan=1>planned</td></tr><tr><td rowspan=1 colspan=1>Volume, quantity of parameters</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>facts about the model</td></tr><tr><td rowspan=1 colspan=1>Tokens, cost, latency, completion</td><td rowspan=1 colspan=1>all</td><td rowspan=1 colspan=1>operational set</td></tr></table>

Probe integration status and attribute ownership evaluated in October 2026; P5–P8 denote planned extensions.

## C. MAS Core: Planes and Components

MAS Core is organised in four planes (Fig. 6). The control plane is a FastAPI service with endpoints for goals, models, probes, prompts, metamodels and cost; it hosts the registries (models and families, probes, attributes, datasets, techniques, prompt variants), the planner and understanding agents and the deterministic gate. The data plane is a PostgreSQL blackboard with registries, goals, frozen plans, runs, measurements, agent prompts and agent spend, plus a Redis broker with no persistence, no host port and a password. The execution plane consists of Celery workers with two queues, a probe agent that triages failures, a runner that executes the probe and computes attributes deterministically, and an aggregator agent. The fourth plane is the measurement stand of the first probe, P1, which is a copy of the ConversionEvaluator adapted to the probe contract; it has its own deployment and database. The responsibilities of each MAS Core component are distinguished in Table XII.

![](images/54c67b9e1e02b2346b3073c07ea25aabd44254a0e892816907818c03234ecbfb.jpg)  
Solid: the model under test and the judge that scores it. Dashed paths (agents that decide what to measure) never meet it  
Fig. 6. MAS Core architecture as implemented, redrawn from its documentation: control plane, execution plane, data stores and the P1 measurement stand. The agent model (the instrument) and the tested model with its judge (the object of measurement and its ruler) are separate roles.

TABLE XII. COMPONENTS OF MAS CORE AND THEIR RESPONSIBILITIES
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Plane</td><td rowspan=1 colspan=1>Responsibility</td></tr><tr><td rowspan=1 colspan=1>API surface</td><td rowspan=1 colspan=1>control</td><td rowspan=1 colspan=1>goals, models, probes, prompts,metamodel views, cost</td></tr><tr><td rowspan=1 colspan=1>Registries</td><td rowspan=1 colspan=1>control</td><td rowspan=1 colspan=1>models, probes, attributes, datasets,prompt variants</td></tr><tr><td rowspan=1 colspan=1>Planner agent</td><td rowspan=1 colspan=1>control</td><td rowspan=1 colspan=1>reads registries via read-only tools,proposes runs</td></tr><tr><td rowspan=1 colspan=1>Understandingagent</td><td rowspan=1 colspan=1>control</td><td rowspan=1 colspan=1>pins a prompt variant to every slot beforefreezing</td></tr><tr><td rowspan=1 colspan=1>Deterministic gate</td><td rowspan=1 colspan=1>control</td><td rowspan=1 colspan=1>validates, canonicalises, hashes andfreezes the plan</td></tr><tr><td rowspan=1 colspan=1>Blackboard(PostgreSQL)</td><td rowspan=1 colspan=1>data</td><td rowspan=1 colspan=1>single write path into measurements</td></tr><tr><td rowspan=1 colspan=1>Broker (Redis)</td><td rowspan=1 colspan=1>data</td><td rowspan=1 colspan=1>queues; task messages carry the caller&#x27;skey</td></tr><tr><td rowspan=1 colspan=1>Probe agent</td><td rowspan=1 colspan=1>execution</td><td rowspan=1 colspan=1>triage of a failed run: retry, resume orabort</td></tr><tr><td rowspan=1 colspan=1>Runner and probeplug-ins</td><td rowspan=1 colspan=1>execution</td><td rowspan=1 colspan=1>execute cases; compute attributeswithout an LLM</td></tr><tr><td rowspan=1 colspan=1>Aggregator agent</td><td rowspan=1 colspan=1>execution</td><td rowspan=1 colspan=1>judges whether a goal&#x27;s result istrustworthy; no numbers</td></tr><tr><td rowspan=1 colspan=1>P1 stand (5services)</td><td rowspan=1 colspan=1>stand</td><td rowspan=1 colspan=1>conversion, validation, evaluation,annotation, gateway</td></tr></table>

A new probe is a package with a manifest plus registry rows; the core is not edited.

## D. Life Cycle of a Goal

A request is a goal, for example “profile this model on Oracle-to-PostgreSQL conversion with the baseline prompt on a small subset”. Figure 7 and Algorithm 1 show how it is processed.

```csv
Algorithm 1. Goal planning, freezing and settlement.
Input: goal G, caller key k, registries Reg
Output: settled goal with measurements and assessment
1 S ← ModelSession(k) // one client per unit of work
2 draft ← Planner(G, read-only tools over Reg, S)
3 draft ← Understanding(pin prompt variant per slot, S)
4 if dry_run then return preview(draft) // nothing is stored
5 plan ← Gate(draft) // reject on any violation:
undeclared mode; variant of another probe; duplicate run;
budget overrun; comparison without baseline arm
6 re-apply goal scope and dataset version by code
7 plan ← canonicalise(sort runs); h ← SHA-256(plan); freeze(plan, h)
8 for each run in plan enqueue(run, k) // idempotent by run id
9 on run end do request settlement // success and failure alike
10 if infrastructure failure then status ← INCONCLUSIVE
11 if no measurements then settle without a model call
12 A ← Aggregator(results, S) // judgement; unreachable → absent
13 store A beside the goal, never inside a measurement
```

![](images/482b89d4e93e81b870b1720bdeeb6c0e7a3c666faaaf9990c84b484ae48528e9.jpg)  
Fig. 7. From a goal to a settled result: the agents propose, deterministic code disposes, and the plan is frozen and hashed before anything is spent.

Four agents take part, declared in the same way: an AgentSpec names the agent, its system prompt and its response schema; a shared run\_agent call performs the structured model call, converts any vendor failure into a single domain error and optionally records spend and the prompt hash; and a ModelSession holds the caller’s key and builds one model client per unit of work, so that the planner and the understanding agent of one goal share one client. Table XIII summarises the agents and what happens when one of them is unreachable.

TABLE XIII. AGENTS OF MAS CORE
<table><tr><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>Response schema</td><td rowspan=1 colspan=1>If unreachable</td></tr><tr><td rowspan=1 colspan=1>Planner</td><td rowspan=1 colspan=1>PlanDraft</td><td rowspan=1 colspan=1>planning fails; no planstored</td></tr><tr><td rowspan=1 colspan=1>Understanding</td><td rowspan=1 colspan=1>VariantChoice</td><td rowspan=1 colspan=1>planning fails; singlecandidate needs no call</td></tr><tr><td rowspan=1 colspan=1>Probe agent (triage)</td><td rowspan=1 colspan=1>TriageDecision</td><td rowspan=1 colspan=1>decision is abort, nevera guessed retry</td></tr><tr><td rowspan=1 colspan=1>Aggregator</td><td rowspan=1 colspan=1>GoalAssessment (4fields)</td><td rowspan=1 colspan=1>verdict absent, not&quot;no&quot;</td></tr></table>

The aggregator schema has four fields (trustworthy, summary, anomalies, caveats);

The gate deserves emphasis because it is where an LLM in the control flow is made safe. The planner is told to pass the goal’s subset through, but “the agent was told to” is not a control: in an early run a dropped subset would have launched the full 1,802-file dataset, hours of work and an unapproved bill. The scope and the dataset version are therefore re-applied by code after planning.

Table XIV lists the checks. Because LLM planning cannot be reproducible by determinism, reproducibility is obtained by materialisation: a rerun replays the stored plan instead of asking the agent again, and the hash is the identity of a measurement. Plans with identical canonical representations produce the same SHA-256 hash. Matching hashes are treated as plan identity under the assumption of negligible cryptographic collision probability; strict content identity can be verified by comparing the stored canonical plans. Different hashes do not alone imply incomparable conditions, because controlled experimental arms may intentionally differ. Comparability additionally requires the same evaluation protocol, data and controlled context.

TABLE XIV. CHECKS APPLIED BY THE DETERMINISTIC GATE
<table><tr><td>Check</td><td>Rejects</td></tr><tr><td>Mode declared</td><td>a mode the probe does not declare</td></tr><tr><td>Variant ownership</td><td>a prompt variant belonging to another probe</td></tr><tr><td>Uniqueness</td><td>duplicated runs</td></tr><tr><td>Budget</td><td>plans exceeding the stated budget</td></tr><tr><td>Baseline arm</td><td>a prompt comparison without the baseline on the same model</td></tr><tr><td>Scope</td><td>dropped subset or changed dataset version (re-applied)</td></tr></table>

## E. Application Container Contract and Onboarding

Reference applications are not linked into the core. Each one runs as a separate container and is reached over HTTP, so a new application can use its own language, dependencies and data. The chatbot application is the first to follow this pattern end to end: MAS Core reaches the Chatbot Evaluation API through a configured stand address and a shared container network, queries its datasets and configurations through a probe-options endpoint, submits an evaluation and persists the returned measurements. Table XV abstracts this into a contract. The application boundary and its onboarding data flow are illustrated in Fig. 8.

![](images/f546e9a152a7f370ccc9f8c979dfa02d5e069716ac88a5f5901e075ede297100.jpg)  
Onboarding registers the probe, the owned attributes and the dashboards.  
Fig. 8. Application container contract and onboarding. A specification drives the code generator; the container exposes health, options, evaluation and attribute declarations; onboarding registers the probe, the metrics and the dashboards.

Algorithm 2 gives the onboarding procedure. Its two outputs are what turns an application into new knowledge about models: the attributes are added to the metric catalogue with an owner, a unit and a direction, and every model that is subsequently run on the application acquires measured values of them.

TABLE XV. APPLICATION CONTAINER CONTRACT
<table><tr><td rowspan=1 colspan=1>Element</td><td rowspan=1 colspan=1>Content</td></tr><tr><td rowspan=1 colspan=1>Health</td><td rowspan=1 colspan=1>liveness and readiness endpoints (/livez, /readyz)</td></tr><tr><td rowspan=1 colspan=1>Options</td><td rowspan=1 colspan=1>datasets, configurations and modes the applicationoffers (GET probe options)</td></tr><tr><td rowspan=1 colspan=1>Evaluation</td><td rowspan=1 colspan=1>submit an evaluation for model(s), dataset version,configuration; return per-case and aggregatemeasurements</td></tr><tr><td rowspan=1 colspan=1>Attributes</td><td rowspan=1 colspan=1>declaration of owned attributes: name, unit, direction,category</td></tr><tr><td rowspan=1 colspan=1>Provenance</td><td rowspan=1 colspan=1>dataset version, configuration identifier, promptidentity, warnings</td></tr><tr><td rowspan=1 colspan=1>Variables</td><td rowspan=1 colspan=1>stand address, network, dataset, application,framework, model-set and evaluation configurationidentifiers</td></tr></table>

The run parameters observed for the chatbot are app\_config\_id, framework\_config\_id, pinned\_model\_set\_id, dataset\_version and evaluation\_config\_id.

```csv
Input: application container C, specification Spec, datasets D
Output: registered probe, catalogue entries, dashboards
1 if C.adapter missing or variables reference unsupported dataset/instance
2 C.adapter ← Generate(Spec, variables) // specification-driven
3 if not SmokeTest(C) then reject // health, options, one case
4 Σ ← C.attribute_declaration()
5 for each σ ∈ Σ: assert unique name, owner = probe, unit, direction
6 register probe(C.modes, C.prompt_slots, D) in registry
7 add σ ∈ Σ to metric catalogue with owner; bump metamodel schema
8 provision Grafana: data source, per-application dashboard,
panels derived from Σ (direction-aware, unit-aware)
9 link probe to existing models; list missing attributes per model
10 return probe id
```

Generating Grafana dashboard definitions from attribute declarations is part of the onboarding design; successful import into a live Grafana server was not measured in this study.

TABLE XVI. EFFECTS OF ONBOARDING A NEW APPLICATION
<table><tr><td rowspan=1 colspan=1>Registry</td><td rowspan=1 colspan=1>Effect</td></tr><tr><td rowspan=1 colspan=1>Probe registry</td><td rowspan=1 colspan=1>probe with modes, prompt slots and dataset versions</td></tr><tr><td rowspan=1 colspan=1>Metric catalogue</td><td rowspan=1 colspan=1>new attributes with owner, unit and direction</td></tr><tr><td rowspan=1 colspan=1>Model metamodels</td><td rowspan=1 colspan=1>attributes appear as “not measured&quot; until a run completes</td></tr><tr><td rowspan=1 colspan=1>Analytical UI</td><td rowspan=1 colspan=1>use case, metrics and evaluations are listed without UIchanges</td></tr><tr><td rowspan=1 colspan=1>Grafana</td><td rowspan=1 colspan=1>data source and dashboard panels for the new attributes</td></tr></table>

Chatbot observation: the chat.\* metrics appear on the Metrics page, owned by p2\_chatbot, with measurement counts.

The metric catalogue associates each chat.\* measurement with its owning probe, unit, direction, context and count; unmeasured attributes remain explicitly absent rather than taking a numerical value of zero (Table XVI).

## F. Specification-Driven Generation of Agent Code

Each container is parameterised by template variables (Table XV). In most cases a variable only selects among existing assets: another dataset version, another application configuration, another model set. For the remaining cases the testbed relies on the specification-driven techniques developed in the authors’ earlier work, in which a technology-neutral specification is the operational artefact from which code is regenerated [45], [46], [77]. When a variable refers to a dataset or an instance of the reference application for which no adapter exists, for example a new chatbot instance over a different documentation corpus, the generation hook produces or modifies the adapter code from the application specification. The generated adapter must pass the smoke test of Algorithm 2 before it can be registered, and the measurements it produces carry the dataset version and configuration identifier like any other. We stress that this paper evaluates the contract and the measurement machinery; the quality of generated adapters is not evaluated here and is listed among the open items in Section IX-J.

## G. Prompt Identity, Cost Accounting and Credentials

Three properties concern what is recorded about every call. Prompt identity. The inherited P1 stand called its templates directly and recorded nothing about the text sent to the model, so every earlier run was, strictly, unreproducible. Templates are now accepted as source plus the identity under which they were registered; the stand verifies that the source hashes to that identity and binds only variables the mode provides, then records the variant identifier and the hash on the run. Omitting a prompt records the built-in under its own name and hash.

A regression test asserts that the new path renders byte-identically to the old, so earlier results remain comparable. Cost separation. The tested model, the judge and the agents are counted as three numbers that are never summed, because the first is a property of the model and the other two are the price of measuring it (Table XVII).

TABLE XVII. THREE-WAY COST SEPARATION
<table><tr><td rowspan=1 colspan=1>Spend</td><td rowspan=1 colspan=1>Role</td><td rowspan=1 colspan=1>Belongs to</td></tr><tr><td rowspan=1 colspan=1>C{</td><td rowspan=1 colspan=1>model under test</td><td rowspan=1 colspan=1>property of the model</td></tr><tr><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>judge scoring the output</td><td rowspan=1 colspan=1>price of measuring</td></tr><tr><td rowspan=1 colspan=1>Cg</td><td rowspan=1 colspan=1>planner, understanding, triage,aggregator</td><td rowspan=1 colspan=1>price of measuring</td></tr></table>

The three cost components are reported separately; their sum is used only when explicitly labelled as total experimental cost, as in Equation (10), and is not treated as a property of the model under test.

The judge’s spend was found to be a substantial fraction of the tested model’s, and it had been invisible because the stand discarded it. Credentials. No service stores a gateway key. Each request carries the caller’s own key, so every model call is billed to whoever asked for it; the Launch page of the interface states that the key is kept in memory and that the core stores only a fingerprint. The key crosses a process boundary inside the task message and is mitigated operationally (a broker without persistence, host port or default password) but not yet cryptographically.

## H. Dashboards: Grafana Registration and the Analytical Interface

Two complementary analytical views consume the common measurement store. Per-application Grafana panels show observed metric series and execution health, while a deterministic API provides filtered measurement and comparison data. Testbed metrics include conversion scores, judge criteria, static validation, file completion, refusals, latency, tokens and separate cost components. The work implements dashboard definitions from attribute declarations, but does not measure end-to-end automated provisioning effort. The architecture of the resulting measurement and dashboard views is shown in Fig. 9.

![](images/7d35e20a9449b4fd5839be02a91ecace8b7b794ee9850c3ca46ef97ab57545ae.jpg)  
Panels are registered from attribute declarations at onboarding.  
Fig. 9. Measurement store and attribute catalogue determine context-aware dashboard definitions and execution-health views; live deployment of generated dashboard definitions was not evaluated.

The registration rests on a small set of rules that map the declaration of an attribute, its unit, direction and category, to a panel (Table XVIII, Algorithm 3). The dashboard variables are the run context and the dataset version, so that a panel can only ever show values of one context class (Lemma 1); an attribute that is not yet measured is shown as an explicit “not measured” panel instead of an empty graph or a zero. These are design rules of the testbed. Their implementation for a new application, and the effort that it saves, have not been evaluated quantitatively in this work.

Algorithm 3. Provisioning dashboards from attribute declarations.  
Input: probe p, attribute set Σ , data source S   
Output: dashboard B registered in Grafana   
1 create data source S if absent (read-only role on the measurement store)   
2 B ← new dashboard titled by p; variables ← {run context κ, dataset   
version d, model}   
3 for each σ $\ r \in $   
4 panel ← Template(σ.unit, σ.direction, σ.category) // Table of panel   
rules   
5 query ← SELECT value, n, run FROM measurements WHERE   
attribute = σ AND context = \$κ AND dataset = \$d   
6 if no rows then panel ← “not measured”   
7 add panel to B   
8 add cost panel (three spends, reported separately) and run-health panel   
9 register $\bar { B } _ { \mathrm { p } } ;$ return identifier

Implementation status: The dashboard provisioning rules describe the intended onboarding contract. Their fully automatic deployment to a live Grafana instance, and the effort saved relative to manual onboarding, were not evaluated quantitatively.

TABLE XVIII. RULES MAPPING ATTRIBUTE DECLARATIONS TO DASHBOARD PANELS
<table><tr><td rowspan=1 colspan=1>Attribute type</td><td rowspan=1 colspan=1>Panel</td><td rowspan=1 colspan=1>Rule</td></tr><tr><td rowspan=1 colspan=1>ratio or percentage,higher is better</td><td rowspan=1 colspan=1>gauge and per-runtrend</td><td rowspan=1 colspan=1>thresholds from the declaration;direction-aware colour</td></tr><tr><td rowspan=1 colspan=1>count or iterations,lower is better</td><td rowspan=1 colspan=1>stat and trend</td><td rowspan=1 colspan=1>direction reversed</td></tr><tr><td rowspan=1 colspan=1>latency (ms)</td><td rowspan=1 colspan=1>time series; percentilebars</td><td rowspan=1 colspan=1>median and tail</td></tr><tr><td rowspan=1 colspan=1>cost (USD), tokens</td><td rowspan=1 colspan=1>stacked bars by spendrole</td><td rowspan=1 colspan=1>tested, judge and agent costsshown separately</td></tr><tr><td rowspan=1 colspan=1>categorical verdict</td><td rowspan=1 colspan=1>state timeline;distribution</td><td rowspan=1 colspan=1>PASS, FAIL, degraded;ACCEPTED, FAILED,BLOCKED</td></tr><tr><td rowspan=1 colspan=1>text evidence</td><td rowspan=1 colspan=1>link to run report</td><td rowspan=1 colspan=1>text is evidence, not a value</td></tr><tr><td rowspan=1 colspan=1>not measured</td><td rowspan=1 colspan=1>explicit “notmeasured&quot; panel</td><td rowspan=1 colspan=1>never zero, never blank</td></tr><tr><td rowspan=1 colspan=1>cross-model view</td><td rowspan=1 colspan=1>Pareto scatter;comparison table</td><td rowspan=1 colspan=1>only within one context class;evidence count shown</td></tr><tr><td rowspan=1 colspan=1>run health</td><td rowspan=1 colspan=1>warnings, plan hash,prompt hash</td><td rowspan=1 colspan=1>provenance beside every result</td></tr></table>

Design rules of the onboarding step; not evaluated quantitatively.

The analytical interface is a single-page application over a deterministic backend. Scores and comparisons originate from server-side aggregations; client-side computation is limited to presentation. The main interface areas and their analytic functions are defined in Table XIX.

TABLE XIX. AREAS OF THE ANALYTICAL INTERFACE
<table><tr><td rowspan=1 colspan=1>Area</td><td rowspan=1 colspan=1>Content</td><td rowspan=1 colspan=1>Nature</td></tr><tr><td rowspan=1 colspan=1>Explore</td><td rowspan=1 colspan=1>models, use cases, metrics, evaluations;filters; raw JSON parameters</td><td rowspan=1 colspan=1>read-onlycatalogue</td></tr><tr><td rowspan=1 colspan=1>Analytics: Summary</td><td rowspan=1 colspan=1>KPI tiles, leaderboard for a chosenmetric, breakdown charts</td><td rowspan=1 colspan=1>deterministic</td></tr><tr><td rowspan=1 colspan=1>Analytics: Modelperformance</td><td rowspan=1 colspan=1>model passport: gauges per metric,evaluation reports</td><td rowspan=1 colspan=1>deterministic</td></tr><tr><td rowspan=1 colspan=1>Analytics: Comparemodels / evaluations</td><td rowspan=1 colspan=1>per-category scores; per-metric delta;context warning</td><td rowspan=1 colspan=1>deterministic</td></tr><tr><td rowspan=1 colspan=1>Analytics: Pareto</td><td rowspan=1 colspan=1>objectives with direction; frontier withevidence counts</td><td rowspan=1 colspan=1>server-side</td></tr><tr><td rowspan=1 colspan=1>Launch</td><td rowspan=1 colspan=1>choose probe, model(s), dataset,configuration; preview plan; start</td><td rowspan=1 colspan=1>calls the core</td></tr><tr><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>conversational recommendation usingthe analytical API as tools</td><td rowspan=1 colspan=1>LLM-backed</td></tr></table>

Technology: React, Vite, TanStack Query, React Router, Tailwind with Radix primitives, Recharts; a typed API client surfaces network, HTTP and validation errors through shared components.

The conversational recommendation component is architecturally separate from the deterministic measurement and Pareto selection procedures, and its interactive behaviour is not used as experimental evidence. The comparison API enforces the context constraints of Equation (4): measurements generated using different datasets, evaluation protocols or controlled configurations cannot be combined as if directly comparable. Incomplete model usage, missing semantic scores and unavailable cost records are returned with explicit warnings rather than converted to zero-valued measurements.

## I. Data Model and API Surface

The blackboard holds a small number of entities, and every API path maps onto them (Table XX). Registries describe what exists: models and their families, probes, attributes, datasets, techniques and prompt variants. Goals carry the request, the caller’s key fingerprint and the state of settlement. Plans are frozen and keyed by hash. Runs point to a plan, a model, a mode and a prompt variant and own their measurements. Agent prompts and agent spend record the text and the cost of every agent call. Because measurements have a single write path, the metamodel views are projections that can be recomputed, and they do not contain information that the underlying rows lack. Two views deserve attention. The metamodel of a model lists the measured attributes together with an explicit list of what was not measured. The comparison view prevents direct comparisons between measurements obtained under different evaluation contexts, thereby enforcing the comparability constraints defined in Equation (4).

TABLE XX. PRINCIPAL ENTITIES AND INTERFACES
<table><tr><td rowspan=1 colspan=1>Entity or path</td><td rowspan=1 colspan=1>Role</td></tr><tr><td rowspan=1 colspan=1>POST /v1/goals (dry_run option)</td><td rowspan=1 colspan=1>plan a goal; with dry_run only a planpreview is returned and no goal is written</td></tr><tr><td rowspan=1 colspan=1>GET /v1/goals/ {id}, /cost</td><td rowspan=1 colspan=1>state, runs, assessment; three-way cost</td></tr><tr><td rowspan=1 colspan=1>GET /v1/models, /v1/probes,/v1/prompts/{id}</td><td rowspan=1 colspan=1>registries; variant ids are readablecoordinates and may contain slashes</td></tr><tr><td rowspan=1 colspan=1>GET /v1/probes/{id}/options</td><td rowspan=1 colspan=1>datasets and configurations that a probeoffers (chatbot integration)</td></tr><tr><td rowspan=1 colspan=1>GET /v1/metamodels</td><td rowspan=1 colspan=1>per-model profile with an explicit list ofmissing attributes</td></tr><tr><td rowspan=1 colspan=1>analytical API: GET /models/use-cases, /metrics, /evaluations</td><td rowspan=1 colspan=1>raw catalogue with filters and pagination</td></tr><tr><td rowspan=1 colspan=1>analytical API: POST /analytics/*</td><td rowspan=1 colspan=1>breakdown, compare-models.compare-evaluations, Pareto</td></tr><tr><td rowspan=1 colspan=1>Blackboard tables</td><td rowspan=1 colspan=1>registries, goals, frozen plans, runs,measurements, agent prompts, agent spend</td></tr></table>

The listed endpoints describe the implemented control and analytical interfaces.

## J. Operational Properties

Automated verification comprises 93 tests for the MAS Core, 21 for the P1 evaluation service and 232 backend tests for the integrated chatbot evaluation interface. Metric-library versions are pinned for a given evaluation context because changing the scoring implementation changes the measurement instrument.

## VI. REFERENCE APPLICATIONS

The three reference applications provide distinct, implemented evaluation environments: Oracle-to-PostgreSQL translation with differential behavioural verification, documentation-grounded question answering with retrieval and tool-use diagnostics, and enterprise workflow automation with observable side effects. Table XXI summarises their experimental configurations; Sections VI-A–VI-C describe the associated task and measurement contracts.

TABLE XXI. OVERVIEW OF THE REFERENCE APPLICATIONS
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Migration</td><td rowspan=1 colspan=1>Chatbot</td><td rowspan=1 colspan=1>Automation</td></tr><tr><td rowspan=1 colspan=1>Purpose</td><td rowspan=1 colspan=1>Oracle PL/SQL → PostgreSQL</td><td rowspan=1 colspan=1>documentation-grounded assistant</td><td rowspan=1 colspan=1>support tasks over Jira, Confluence, Outlook</td></tr><tr><td rowspan=1 colspan=1>Probe</td><td rowspan=1 colspan=1>P2 (own stand), P1 (conversion)</td><td rowspan=1 colspan=1>P3, id p2_chatbot</td><td rowspan=1 colspan=1>P4 (own stand)</td></tr><tr><td rowspan=1 colspan=1>Orchestration</td><td rowspan=1 colspan=1>LangGraph repair loop</td><td rowspan=1 colspan=1>LangGraph</td><td rowspan=1 colspan=1>4 interchangeable frameworks</td></tr><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>1,448 tasks; 216 units; 1,802 files (P1)</td><td rowspan=1 colspan=1>mvp scenarios v6, 10 scenarios</td><td rowspan=1 colspan=1>pilot.v4, 49 samples</td></tr><tr><td rowspan=1 colspan=1>Ground truth</td><td rowspan=1 colspan=1>real Oracle run; expert references</td><td rowspan=1 colspan=1>expected sources, tools, answers</td><td rowspan=1 colspan=1>expected tool operations, final states</td></tr><tr><td rowspan=1 colspan=1>Judge</td><td rowspan=1 colspan=1>P1 judge; semantic judge in depth study</td><td rowspan=1 colspan=1>semantic judge v1</td><td rowspan=1 colspan=1>none (effects observed)</td></tr><tr><td rowspan=1 colspan=1>Integration</td><td rowspan=1 colspan=1>own stand; MCP and API</td><td rowspan=1 colspan=1>integrated in MAS Core</td><td rowspan=1 colspan=1>own stand</td></tr></table>

## A. Migration Agent: Oracle PL/SQL to PostgreSQL

The migration agent converts a directory of Oracle SQL and PL/SQL files into PostgreSQL files and, for every database object, reports how it was converted, which checks were completed and what could not be verified. The pipeline and routing were designed by A. Seletska, the integrations by O. Ilchuk and the measurement of quality by D. Kostetskyi. It does not connect to an Oracle database; it works only with the files it is given. The migration sequence from parsing to validation and bounded repair is shown in Fig. 10.

![](images/d6fa26fcbc95c625711c1d57f9a7bdd66687d2c24d5d188808fb7e980a65077f.jpg)  
Accepted: PostgreSQL files and report. Failed: repair or escalate STATIC → CHEAP → STRONG  
Fig. 10. Migration pipeline: files are split into migration units, ordered by a dependency graph, routed by a complexity score, generated, validated in PostgreSQL 16, and repaired or escalated on failure.  
Experimental configurations and dataset sizes used in this study.

Decomposition. The parser splits each file into migration units (table, view, sequence, function, procedure, package, trigger, query or data-changing statement) while preserving the original source text. A package remains one unit because its members share state. Each unit carries a parse confidence: full, partial, lexical-only or failed, and a low confidence excludes the deterministic converter. Ordering. A dependency graph places required objects before their users; mutually dependent objects form one migration group (a strongly connected component), and the condensation is a directed acyclic graph that is processed in topological order. A failing group blocks its dependants while independent branches continue. The same decomposition idea underlies the earlier study of task decomposition for specification generation by the same team.

Routing. A deterministic weighted score in [0, 1] estimates the difficulty of each group from source size, control flow, data access, Oracle-specific features, dependencies, cycles, package size and parse quality. Scores up to 0.30 select the STATIC strategy (Ora2Pg and deterministic rules, no model tokens), scores above 0.30 and up to 0.65 select a cheaper model, and higher scores a stronger model (Table XXII). Large packages go directly to the strong model, and low analysis confidence promotes a unit to a stronger strategy. Every routing decision records a reason. The individual feature families of the complexity score are detailed in Table XXIII. The bounded migration, validation, and repair procedure is formalised in Algorithm 4, which defines the iterative correction process, strategy escalation, and termination conditions under predefined execution and retry budgets.

TABLE XXII. MIGRATION STRATEGIES
<table><tr><td rowspan=1 colspan=1>Complexity</td><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>Mechanism</td></tr><tr><td rowspan=1 colspan=1>0-0.30</td><td rowspan=1 colspan=1>STATIC</td><td rowspan=1 colspan=1>Ora2Pg plus exact rules (e.g. NVL →COALESCE); no tokens</td></tr><tr><td rowspan=1 colspan=1>&gt; 0.30 -0.65</td><td rowspan=1 colspan=1>CHEAP</td><td rowspan=1 colspan=1>less expensive configured model</td></tr><tr><td rowspan=1 colspan=1>&gt; 0.65</td><td rowspan=1 colspan=1>STRONG</td><td rowspan=1 colspan=1>more capable model; large packages;escalations</td></tr></table>

Models in the breadth study: gpt-5.4-nano, gpt-5.4-mini, gpt-5; in the depth study: DeepSeek-V4-Flash (cheap) and DeepSeek-V4-Pro (strong).

TABLE XXIII. INPUTS OF THE COMPLEXITY SCORE
<table><tr><td rowspan=1 colspan=1>Feature family</td><td rowspan=1 colspan=1>Role in routing</td></tr><tr><td rowspan=1 colspan=1>source size</td><td rowspan=1 colspan=1>longer sources raise the score</td></tr><tr><td rowspan=1 colspan=1>control flow</td><td rowspan=1 colspan=1>branching and loops raise the score</td></tr><tr><td rowspan=1 colspan=1>data access</td><td rowspan=1 colspan=1>cursors, bulk operations, dynamic SQL</td></tr><tr><td rowspan=1 colspan=1>Oracle-specific features</td><td rowspan=1 colspan=1>constructs without a PostgreSQL equivalent</td></tr><tr><td rowspan=1 colspan=1>dependencies and cycles</td><td rowspan=1 colspan=1>coupling, mutually dependent groups</td></tr><tr><td rowspan=1 colspan=1>package size</td><td rowspan=1 colspan=1>large packages go directly to the strong model</td></tr><tr><td rowspan=1 colspan=1>parse quality</td><td rowspan=1 colspan=1>low confidence moves a unit to a strongerstrategy</td></tr></table>

Weighted formula with YAML-configured weights; not a trained classifier.

Algorithm 4. Bounded migrate–validate–repair loop for one migration group.  
Input: group g, dependency signatures, budget (2 repairs/strategy, 2   
escalations, 5 attempts, 900 s)   
Output: status, verification level, PostgreSQL text   
1 s ← Route(score(g), parse confidence, type, size); sql ← Generate(s, g,   
signatures)   
2 repeat while attempts < 5 and elapsed < 900 s   
3 attempts ← attempts + 1; r ← Validate(sql)   
4 // static → install → structure → fixed scenarios   
5 if r.pass then return ACCEPTED, level(r)   
6 if r.infra\_error then return INCONCLUSIVE   
7 if repairs(s) < 2 then sql ← Repair(sql, r.diagnostics); repairs(s) += 1   
8 else if escalations < 2 and next(s) exists then s ← next(s); sql ←   
Generate(s, g, signatures); escalations += 1   
9 else return FAILED   
10 return FAILED // attempts or time exhausted; dependants become   
BLOCKED

Generation and validation. The model receives the Oracle source, all members of a mutually dependent group, the signatures of dependencies already migrated, the names of missing dependencies and conversion guidance; it returns structured PostgreSQL and notes. It does not choose the order, compute its own complexity or decide whether its output is correct. Validation runs against a PostgreSQL 16 database kept for the run: static analysis, installation, structural checks, generated behaviour scenarios and derivable invariants. Scenarios are created once per group and reused after repairs so that tests cannot drift to fit a candidate. The strongest completed check is recorded as the verification level (Table XXIV), and a group takes the result of its least successful member.

TABLE XXIV. VERIFICATION LEVELS AND UNIT STATUSES
<table><tr><td rowspan=1 colspan=1>Term</td><td rowspan=1 colspan=1>Meaning</td></tr><tr><td rowspan=1 colspan=1>CHECKED STATIC</td><td rowspan=1 colspan=1>static checks passed; object could not be installed</td></tr><tr><td rowspan=1 colspan=1>VERIFIED INSTALL</td><td rowspan=1 colspan=1>PostgreSQL installed or accepted the object</td></tr><tr><td rowspan=1 colspan=1>VERIFIED BEHAVIOR</td><td rowspan=1 colspan=1>installed and generated behaviour checks passed</td></tr><tr><td rowspan=1 colspan=1>ACCEPTED</td><td rowspan=1 colspan=1>passed every check available for the unit</td></tr><tr><td rowspan=1 colspan=1>FAILED</td><td rowspan=1 colspan=1>attempted but still failing</td></tr><tr><td rowspan=1 colspan=1>BLOCKED</td><td rowspan=1 colspan=1>an upstream unit failed; never tried</td></tr><tr><td rowspan=1 colspan=1>INCONCLUSIVE</td><td rowspan=1 colspan=1>infrastructure unavailable; not a verdict on the SQL</td></tr><tr><td rowspan=1 colspan=1>UNSUPPORTED</td><td rowspan=1 colspan=1>could not be attempted</td></tr></table>

Interfaces. The same service layer is exposed through a command line, an HTTP API with asynchronous jobs and an MCP server, so that an orchestrating model can discover tools and follow a run through structured resources. Provider credentials stay in server-side configuration and are never tool arguments. An optional integration generates a repair agent with EvoAgentX in an isolated project that communicates through a small versioned JSON protocol; it only produces candidates, and installation and scenario execution in the main process remain authoritative.

Evaluation corpora. Two corpora answer two questions. The breadth corpus is a grammar-clean subset of paired Oracle fragments and human-written PostgreSQL references: 1,000 pairs drawn with seed 42 produced 1,448 migration tasks. The depth corpus consists of 216 migration units from 34 complete Oracle PL/SQL example files from a public repository, each executed on a real Oracle instance (Oracle 26ai Free in Docker), so that a translation can be compared with what Oracle printed and raised. The first measures how often the agent’s own checks pass and how close the output is to a reference; the second measures whether the translation behaves like Oracle.

## B. Chatbot Agent: Documentation-Grounded Assistant

The chatbot is a developer-support assistant that answers questions over a body of documentation, retrieving passages, calling tools and citing sources. Its evaluation framework measures how well different LLMs and chatbot configurations work for documentation-based question answering and where a failure occurs: in execution, tool routing, retrieval, grounding or answer generation. It runs the real chatbot with selected retrieval and model configurations, scores it against a versioned dataset and saves the evidence. Ground truth is never passed to the chatbot; expected sources, tool behaviour and answer content are used only by the evaluation engine. Its execution and evaluation stages are summarised in Fig. 11.

![](images/4f0ec00715428acc959495be26bc5eff3d1206e795588c9fbbf250e38dd11a72.jpg)  
Fig. 11. Chatbot evaluation workflow. The real chatbot is executed on a versioned sample; deterministic metrics and a semantic judge produce results that feed aggregation and an evaluation agent.

Application, model, evaluation and experiment configurations are stored separately, and every run saves the configuration actually used. Six metric groups evaluate the components separately (Table XXV); two methods are used. Deterministic checks cover execution, tools, retrieval and citation structure. A judge, semantic\_judge\_v1, scores faithfulness, answer correctness, answer relevance and task completion with fixed criteria; its scores support analysis and comparison but do not by themselves decide PASS or FAIL. The judge is an evaluation component built on an existing LLM, not a separately trained model.

TABLE XXV. CHATBOT METRIC GROUPS
<table><tr><td rowspan=1 colspan=1>Group</td><td rowspan=1 colspan=1>Measurements</td><td rowspan=1 colspan=1>Method</td></tr><tr><td rowspan=1 colspan=1>Execution</td><td rowspan=1 colspan=1>execution success, tool execution success</td><td rowspan=1 colspan=1>deterministic</td></tr><tr><td rowspan=1 colspan=1>Agent /tools</td><td rowspan=1 colspan=1>tool selection, tool arguments, forbiddencalls</td><td rowspan=1 colspan=1>deterministic</td></tr><tr><td rowspan=1 colspan=1>Retrieval</td><td rowspan=1 colspan=1>recall@K, precision@K, R-precision,MRR, metadata</td><td rowspan=1 colspan=1>deterministic</td></tr><tr><td rowspan=1 colspan=1>Grounding</td><td rowspan=1 colspan=1>citation compliance, validity,completeness, source match; faithfulness</td><td rowspan=1 colspan=1>deterministic;judge</td></tr><tr><td rowspan=1 colspan=1>Answerquality</td><td rowspan=1 colspan=1>correctness, relevance, task completion</td><td rowspan=1 colspan=1>judge</td></tr><tr><td rowspan=1 colspan=1>Operational</td><td rowspan=1 colspan=1>latency, tokens, retries, iterations, LLMcalls</td><td rowspan=1 colspan=1>deterministic</td></tr></table>

The metric groups are registered as chat.\* attributes in the metamodel, with explicit ownership, measurement context, and provenance.

Execution contract. The execution layer runs one conversation through the existing LangGraph chatbot under a selected configuration and returns a factual result to the evaluation layer. It does not rebuild the chatbot, does not see ground truth and does not compute metrics. A run plan expands every combination of sample, application configuration, model set and repeat, records the outcome of each execution and hands each result across a framework-neutral boundary (Table XXVI). Per-run settings (chat model, router model, decoding parameters, retrieval strategy, Top-K, reranker state, enabled tools) are injected at run time, which was verified in focused tests and by baseline and alternative-model smoke runs; model selection is therefore explicit and not merely recorded as metadata. Each execution starts from a fresh thread and state and uses a dedicated evaluation principal with memory isolation, and 21 execution tests cover, among others, the mechanical rejection of ground-truth fields, the preservation of retrieved chunks after serialisation and the capture of successful and failed tool attempts.

TABLE XXVI. EXECUTION CONTRACT OF THE CHATBOT EVALUATION
<table><tr><td rowspan=1 colspan=1>Object</td><td rowspan=1 colspan=1>Content</td></tr><tr><td rowspan=1 colspan=1>ExecutableSample</td><td rowspan=1 colspan=1>sample id, dataset version, ordered user turns only(no ground truth)</td></tr><tr><td rowspan=1 colspan=1>AppConfiguration</td><td rowspan=1 colspan=1>retrieval strategy, Top-K, reranker state, enabledtools, identifier</td></tr><tr><td rowspan=1 colspan=1>ModelSet</td><td rowspan=1 colspan=1>chat model, router model; optional temperature,top-p, token limit, structured output</td></tr><tr><td rowspan=1 colspan=1>RunPlan</td><td rowspan=1 colspan=1>dataset version, configurations, model sets,repeats, concurrency, failure handling</td></tr><tr><td rowspan=1 colspan=1>ChatbotAdapter</td><td rowspan=1 colspan=1>invokes the chatbot turn by turn; converts runtimedata into a RunResult</td></tr><tr><td rowspan=1 colspan=1>RunResult</td><td rowspan=1 colspan=1>answer, structured citations, ordered tool calls andresults, retrieval evidence, statuses, typed errors</td></tr><tr><td rowspan=1 colspan=1>RunResultConsumer</td><td rowspan=1 colspan=1>boundary through which the result is handed to theevaluation layer</td></tr></table>

Chatbot evaluation execution contract and interfaces.

Preflight and datasets. Before evaluation, the runner verifies readiness of the database, prompt registry, LLM gateway and configured tool services. Experiments use the versioned mvp\_scenarios v6 dataset. Expected answers, sources and tool actions are held in the evaluator, and never supplied to the tested chatbot. Some expected labels have not been independently validated through execution, so corresponding measurements are restricted to their declared evidence level.

The evaluation agent. A stateful LangGraph agent observes the computed metrics, decides on a diagnostic action, updates its state and decides again until it can name the failing layer. It does not change metric values or verdicts. Its five diagnostic tools (Table XXVII) are internal functions and not tools of the chatbot. The agent applies causal gating: it examines execution first, then routing and tools, retrieval, grounding and answer quality, and the first failing layer is the root cause (Fig. 12). For example, if the documentation-search tool was never called, routing is the cause and retrieval, which never ran, is not assigned a separate cause. A deterministic fallback replaces the agent’s controller if the model is unavailable.

![](images/e2808bfad4a9d964c3b4fb9c87261bcaf87170c09e6012b8ef0d56b73eddd7e5.jpg)  
The first failing layer is the root cause; downstream layers receive no separate cause.  
Example: search tool never called, so routing is the cause and retrieval never ran

Fig. 12. Causal gating: layers are examined in dependency order.  
TABLE XXVII. DIAGNOSTIC TOOLS OF THE EVALUATION AGENT
<table><tr><td rowspan=1 colspan=1>Tool</td><td rowspan=1 colspan=1>Analyses</td><td rowspan=1 colspan=1>Example finding</td></tr><tr><td rowspan=1 colspan=1>diagnose_execution</td><td rowspan=1 colspan=1>runtime and tool failures</td><td rowspan=1 colspan=1>error during a tool call</td></tr><tr><td rowspan=1 colspan=1>diagnose_agent_tools</td><td rowspan=1 colspan=1>tool selection andarguments</td><td rowspan=1 colspan=1>required search tool notcalled</td></tr><tr><td rowspan=1 colspan=1>diagnose_retrieval</td><td rowspan=1 colspan=1>sources and ranking</td><td rowspan=1 colspan=1>a required document notfound</td></tr><tr><td rowspan=1 colspan=1>diagnose_grounding</td><td rowspan=1 colspan=1>source support, citations</td><td rowspan=1 colspan=1>one of two source groupscited</td></tr><tr><td rowspan=1 colspan=1>diagnose_answer_quality</td><td rowspan=1 colspan=1>correctness, relevance,completion</td><td rowspan=1 colspan=1>required information missing</td></tr></table>

Algorithm 5. Causal-gated diagnosis with optional targeted re-test.  
```csv
Input: run result R with metrics, policy Π of allowed actions, candidates Cand
Output: diagnosis, optional accepted candidate
1 L ← [execution, tools, retrieval, grounding, answer_quality]; root_cause ←
none
2 for each layer l in L in order
3 if R.passes(l) then continue
4 d ← diagnose_l(R) // action chosen by controller from Π
5 root_cause ← l; evidence ← d; break // downstream layers not assigned
6 if root_cause ≠ none and approved candidate c ∈ Cand exists
7 rerun same scenario with c; compare target metrics with baseline
8 if PASS is restored or a prespecified target-metric improvement is
confirmed then mark c as a candidate for validation
9 return root_cause, evidence, candidate status; do not mutate the live system
```

TABLE XXVIII. MODEL ROLES IN THE CHATBOT EVALUATION
<table><tr><td rowspan=1 colspan=1>Role</td><td rowspan=1 colspan=1>Configuration</td><td rowspan=1 colspan=1>Purpose</td></tr><tr><td rowspan=1 colspan=1>Chat model</td><td rowspan=1 colspan=1>GPT-5.6 Terra (selected)</td><td rowspan=1 colspan=1>generates answers</td></tr><tr><td rowspan=1 colspan=1>Semantic judge</td><td rowspan=1 colspan=1>semantic_judge_v1, GPT-5.6 Solreasoning</td><td rowspan=1 colspan=1>scores semantic quality</td></tr><tr><td rowspan=1 colspan=1>Agent controller</td><td rowspan=1 colspan=1>GPT-5.6 Sol reasoning</td><td rowspan=1 colspan=1>selects the next permitteddiagnostic action</td></tr><tr><td rowspan=1 colspan=1>Router model</td><td rowspan=1 colspan=1>baseline-models (variable)</td><td rowspan=1 colspan=1>decides tool routing insidethe chatbot</td></tr></table>

Judge and controller have separate roles and instructions although both use the same model in the current configuration. Vendor and deployment designations are reproduced as recorded by the gateway; internal aliases are not verified product identifiers. The explicit model-role configuration is documented in Table XXVIII.

Integration. The chatbot evaluation API executes the configured chatbot, collects structured execution evidence and writes measurements into the MAS Core store. The integrated evaluation path passed an end-to-end validation, including deterministic retrieval and tool-call checks. Semantic metrics are emitted only when the specified judge is available, while missing token usage and incomplete citations generate explicit warnings. These integration checks establish the measurement path but do not substitute for application-quality evaluation.

## C. Automation Agent: Enterprise Support Tasks

The automation system handles inbound Jira, Confluence or Outlook events with a single LLM agent that reasons in a loop and acts through fifteen canonical business operations, five per system. Every execution starts from a fresh state, carries no memory from earlier executions, reads knowledge without changing it and is bounded to ten steps. Failures are typed outcomes, never hidden or fabricated. The system has five layers (Fig. 13): entry points; strategy selection; prompt orchestration, model-call interception and tool management; business integrations, which are live systems in production and a frozen world state in evaluation; and infrastructure for configuration, retries, evidence and authentication.

![](images/823affb63ee7ef4791123b712e1ffc46118a2d09ea29da038f82809ad9f6b125.jpg)  
Evaluation mode reads a frozen world state; execution is bounded to ten steps.  
Fig. 13. Layered architecture of the automation agent. In evaluation mode the tool manager reads a frozen world state.

Interchangeable frameworks. A framework-neutral strategy contract lets a factory build the configured implementation once. LangGraph (compiled graph) and LlamaIndex (event-driven workflow) apply the step limit per execution and support model-call retries; AutoGen (group chat) and CrewAI (agent, task, crew) fix the step limit at start-up and do not support retries. All adapters share the same prompts, tools, retry handling and evidence, which keeps results comparable; adding a framework takes one adapter and one registry entry. The framework is thereby a variable of the experiment, in line with the framework-level comparison in [78]. Experimental-design limitation: Model and framework sweeps are analysed separately here. Their association statistics and effect-rate ranges do not estimate an interaction term in a crossed model-by-framework factorial design and should not be interpreted as a general causal superiority of model choice.

Configuration schema. An agent identity specification fixes the framework, model, prompt, chain-of-thought and few-shot choices [79], [80]; a system configuration adds sampling (temperature, maximum tokens), knowledge-base retrieval settings and retry budget. The configuration examples comprise six identities and eleven configurations: five baseline model conditions (GPT-5.4, GPT-5.4 Mini,

Claude Opus 5, Gemini 3.5 Flash, o4-mini) and six controlled variations around GPT-5.4 (temperature 0.3 and 0.7, chain of thought, knowledge base, retry budgets 1 and 2); the candidate models were checked through the gateway for native capability rather than assumed from their names.

Sample model. Running against live Jira, Confluence and Outlook cannot provide an evaluation because live systems mutate between runs, hold no authored ground truth and would produce hundreds of real tickets and emails under a configuration, task and repeat fan-out. The dataset is therefore a closed simulation: the structure matches real payloads so that the model sees production-shaped data, the content is synthetic and the world is frozen. A sample has three layers (Table XXIX); the expected layer is designed backward from the evaluation axes so that every axis is computable and nothing is decorative. Tool names in the expected steps are bound to the operations that the tool manager really exposes (Table XXX); any other name that an agent requests is by definition what the hallucinated-call rate catches. The dataset identifier is immutable after first use, and the same body of tasks is partitionable into learning, validation and locked-test roles without leakage, so that evidence gathered for self-learning cannot inflate scores on the locked test.

TABLE XXIX. LAYERS OF AN AUTOMATION SAMPLE
<table><tr><td rowspan=1 colspan=1>Layer</td><td rowspan=1 colspan=1>Purpose</td><td rowspan=1 colspan=1>Feeds</td></tr><tr><td rowspan=1 colspan=1>task_input</td><td rowspan=1 colspan=1>the event that starts the run(production-shaped, synthetic)</td><td rowspan=1 colspan=1>input</td></tr><tr><td rowspan=1 colspan=1>world_state</td><td rowspan=1 colspan=1>frozen snapshot that tools read duringthe run</td><td rowspan=1 colspan=1>determinism</td></tr><tr><td rowspan=1 colspan=1>expected</td><td rowspan=1 colspan=1>authored ground truth: steps,parameters, valid tools, final state,completion signal</td><td rowspan=1 colspan=1>all correctnessand reasoningaxes</td></tr></table>

Source: dataset schema of the automation direction.

TABLE XXX. CANONICAL OPERATIONS OF THE THREE TOOLS
<table><tr><td rowspan=1 colspan=1>Tool</td><td rowspan=1 colspan=1>Operations</td></tr><tr><td rowspan=1 colspan=1>Jira</td><td rowspan=1 colspan=1>create_issue, update_issue, search_issues, add_comment,get_issue</td></tr><tr><td rowspan=1 colspan=1>Confluence</td><td rowspan=1 colspan=1>create_page, update_page, search_pages, get_page,add comment</td></tr><tr><td rowspan=1 colspan=1>Outlook</td><td rowspan=1 colspan=1>send_email, list_emails, read_email, reply_emaillist_calendar_events</td></tr></table>

Fifteen operations; add\_comment is shared and takes exactly one target (issue key or page id).

Two-phase tool lifecycle. A tool call passes through three events: a request, emitted before validation with the tool name and the arguments exactly as requested; a start, emitted after validation and before dispatch; and a finish, emitted on completion or rejection. A request rejected before dispatch produces a finish without a start. A tool call is counted from a unique agent-requested action, including requests rejected before dispatch, so that an invented tool is counted even if it was never executed; transport retries and event-delivery retries are not new calls. This convention is what makes the hallucinated-call rate and the argument-validity rate well defined across frameworks that differ in step granularity.

Dataset. The pilot dataset, identified as pilot.v4, contains 49 generated task samples across six task classes (jira\_create, confluence\_lookup, email\_compose, incident\_triage, cross\_system\_sync and escalation). Each sample has a production-shaped input, a frozen world state and authored expected tool operations and final states. All 49 samples pass schema validation and the repository tests (35 passing). The dataset is synthetic and has known gaps: fifteen samples require a contact address that no tool can read, so a completed run on them can still mean that no email was sent. The resulting eleven measurable automation axes are defined in Table XXXI.

TABLE XXXI. AUTOMATION EVALUATION AXES
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Measures</td></tr><tr><td rowspan=1 colspan=1>Task completion rate</td><td rowspan=1 colspan=1>share of executions achieving the expectedoutcome</td></tr><tr><td rowspan=1 colspan=1>Tool selection accuracy</td><td rowspan=1 colspan=1>requested tool equals expected tool at theposition</td></tr><tr><td rowspan=1 colspan=1>Tool sequence orderaccuracy</td><td rowspan=1 colspan=1>complete expected sequence matched</td></tr><tr><td rowspan=1 colspan=1>Tool argument validity rate</td><td rowspan=1 colspan=1>structurally valid, contextually appropriatearguments</td></tr><tr><td rowspan=1 colspan=1>Mean agent loop steps</td><td rowspan=1 colspan=1>execution units, framework-specific</td></tr><tr><td rowspan=1 colspan=1>Mean total LLM calls</td><td rowspan=1 colspan=1>model invocation attempts incl. retries</td></tr><tr><td rowspan=1 colspan=1>Mean total tokens</td><td rowspan=1 colspan=1>input and output tokens</td></tr><tr><td rowspan=1 colspan=1>Error retry rate</td><td rowspan=1 colspan=1>executions triggering retries</td></tr><tr><td rowspan=1 colspan=1>Hallucinated tool call rate</td><td rowspan=1 colspan=1>requests naming unavailable tools</td></tr><tr><td rowspan=1 colspan=1>Premature termination rate</td><td rowspan=1 colspan=1>limits, timeouts, incomplete responses</td></tr><tr><td rowspan=1 colspan=1>Unnecessary loop rate</td><td rowspan=1 colspan=1>loops not warranted by the observationspresented (trace analysis)</td></tr></table>

Eleven evaluation axes are specified and listed. Assessments use recorded execution evidence, not inferred reasoning; unavailable values remain explicitly missing.

Self-learning. The system treats execution experience as evidence and generated lessons as hypotheses, in the spirit of experience-based agents [27], [28]. Candidate procedures are generated from failed executions, validated one at a time on separate tasks, and published only after demonstrated improvement and human approval, as immutable knowledge snapshots selected explicitly by later configurations. This mechanism is designed but not part of the measurements reported here.

Self-learning pipeline and its measurement. As suggested by experience-based agent research [27], [28], candidate procedures may be abstracted from successful and failed trajectories, then compared with an unchanged baseline on independent validation tasks. The testbed provides the separate evaluation, control-arm, evidence and noise-floor mechanisms required by such a loop. The generation and long-term retention of learned procedures are future applications of this measurement method; no learning gains are claimed in this study.

Algorithm 6. Validation of one candidate procedure in the self-learning   
loop.   
Input: failed executions F, successful examples S, snapshot $K _ { 0 } ,$ partitions   
(learning, validation, locked test)   
Output: published snapshot K or rejection   
1 c ← Generate(F, S) // contrast failures with successes; narrow procedure   
2 if ¬Relevant(c) ∨ ¬Safe(c) then reject // before any execution   
3 arm ← run validation tasks with $K _ { 0 } ; a r m _ { \mathrm { s y s } } *$ ← run with $K _ { 0 } + c$   
4 Δ ← score(arm<sub>sys</sub>) − score(arm<sub>ctrl</sub>) // per axis; repeats ≥ 5   
5 if repeats are insufficient or |Δ| ≤ range(armsys) + range(armctrl) + τ then   
return “no result”   
6 if Δ does not improve the direction-adjusted target metric, or any outcome   
axis regresses, then reject // compliance never instead of outcome   
7 if human approves then publish $K _ { 1 } = K _ { 0 } + c$ (immutable)   
8 else defer; a revision is a new explicit attempt   
9 never touch the locked-test partition during steps 1–8

Experimentation strategy. Every configuration choice is an explicit experimental variable. Stage 0 screens acting models against applicability thresholds with all else at baseline; Stage 1 sweeps behavioural factors one at a time on survivors; Stage 2 sweeps retrieval only if a knowledge base proves useful; a final bounded stage probes a few interactions. Repeats quantify variation, and an effect smaller than the repeat spread is not reported as an improvement. Table XXXII lists the axes. The applicability thresholds (hallucinated-tool-call rate at most 5%, completion at least 60% on the screening subset) are placeholders pending team agreement, so Stage 0 is not executable yet.

TABLE XXXII. AUTOMATION EXPERIMENT AXES
<table><tr><td rowspan=1 colspan=1>Axis</td><td rowspan=1 colspan=1>Bound field</td><td rowspan=1 colspan=1>Disposition</td></tr><tr><td rowspan=1 colspan=1>Acting model</td><td rowspan=1 colspan=1>identity model id</td><td rowspan=1 colspan=1>swept (primary)</td></tr><tr><td rowspan=1 colspan=1>Chain of thought</td><td rowspan=1 colspan=1>prompt chain-of-thought flag</td><td rowspan=1 colspan=1>swept</td></tr><tr><td rowspan=1 colspan=1>Temperature</td><td rowspan=1 colspan=1>system model temperature {0.0,0.3, 0.7}</td><td rowspan=1 colspan=1>swept</td></tr><tr><td rowspan=1 colspan=1>Knowledge base</td><td rowspan=1 colspan=1>knowledge base enabled</td><td rowspan=1 colspan=1>swept</td></tr><tr><td rowspan=1 colspan=1>Retrieval</td><td rowspan=1 colspan=1>strategy, top-k, max contexttokens</td><td rowspan=1 colspan=1>if knowledgebase helps</td></tr><tr><td rowspan=1 colspan=1>Retry budget</td><td rowspan=1 colspan=1>max retries {0, 1, 2}</td><td rowspan=1 colspan=1>swept</td></tr><tr><td rowspan=1 colspan=1>max_tokens,framework,prompts</td><td rowspan=1 colspan=1>various</td><td rowspan=1 colspan=1>fixed at baseline</td></tr></table>

Representative axes from the experimentation strategy.

## D. What the Applications Contribute to the Metamodel

Taken together, the applications contribute attributes that no general benchmark provides (Table XXXIII). The migration agent contributes behavioural equivalence to a real system, the notion of a demonstrably correct unit and a ledger of fixes; the chatbot contributes retrieval and grounding attributes and the localisation of failures by layer; the automation agent contributes the distinction between claimed and effective action. Each of these is available to every model that is run on the corresponding application, which is the mechanism by which the set of measurable LLM features grows with the set of applications.

TABLE XXXIII. ATTRIBUTES CONTRIBUTED BY EACH APPLICATION
<table><tr><td rowspan=1 colspan=1>Application</td><td rowspan=1 colspan=1>New attributes (examples)</td><td rowspan=1 colspan=1>Question answered</td></tr><tr><td rowspan=1 colspan=1>Migration</td><td rowspan=1 colspan=1>units demonstrably correct; defects;verification level; fix confirmations</td><td rowspan=1 colspan=1>does the output behavelike Oracle?</td></tr><tr><td rowspan=1 colspan=1>Chatbot</td><td rowspan=1 colspan=1>retrieval recall/precision/MRR;citation validity; faithfulness;tool-argument correctness</td><td rowspan=1 colspan=1>where does groundingfail?</td></tr><tr><td rowspan=1 colspan=1>Automation</td><td rowspan=1 colspan=1>effects per run; unbacked successclaims; hallucinated tool calls;premature termination</td><td rowspan=1 colspan=1>does the model act,and does it say sohonestly?</td></tr><tr><td rowspan=1 colspan=1>P1conversion</td><td rowspan=1 colspan=1>NLP aggregate; judge criteria;linter error rate; refusals</td><td rowspan=1 colspan=1>how complete iswhole-file conversion?</td></tr></table>

## VII. EVALUATION AND DIAGNOSTIC METHODOLOGY

This section answers RQ2. Its content is not a list of good intentions but a set of rules that were introduced because a stand misled its authors. We first summarise the defects, then state the rules they led to, and finally the diagnostic procedure that connects them (step M5 of the method).

## A. Defects Found by Running the Stands

Execution of the testbed revealed six infrastructure defects (Table XXXIV), including invalid planning output, omitted dataset restrictions and inconsistent settlement of queued tasks. Independent Oracle-to-PostgreSQL differential checks uncovered evaluator failures rather than migration failures (Table XXXV). Fixing these defects changed the number of units that could be reliably assessed, demonstrating why an evaluation pipeline must be diagnosed before blaming the model.

TABLE XXXIV. DEFECTS FOUND IN MAS CORE BY RUNNING IT
<table><tr><td rowspan=1 colspan=1>Defect</td><td rowspan=1 colspan=1>Consequence</td><td rowspan=1 colspan=1>Resolution</td></tr><tr><td rowspan=1 colspan=1>strict JSON-schemaoutput unusable for openmaps</td><td rowspan=1 colspan=1>no plan could beparsed</td><td rowspan=1 colspan=1>tool-calling instead</td></tr><tr><td rowspan=1 colspan=1>settlement scheduled onceon a countdown</td><td rowspan=1 colspan=1>goal stayed “running&quot;</td><td rowspan=1 colspan=1>run requestssettlement</td></tr><tr><td rowspan=1 colspan=1>orphan recovery sweptqueued runs</td><td rowspan=1 colspan=1>healthy workcancelled</td><td rowspan=1 colspan=1>only “running&quot; rows,on worker start</td></tr><tr><td rowspan=1 colspan=1>registry populated only inAPI process</td><td rowspan=1 colspan=1>worker could not runany probe</td><td rowspan=1 colspan=1>lazy discovery</td></tr><tr><td rowspan=1 colspan=1>dropped subset notenforced</td><td rowspan=1 colspan=1>would run all 1,802files</td><td rowspan=1 colspan=1>scope re-applied bycode</td></tr><tr><td rowspan=1 colspan=1>variant ids contain slashes</td><td rowspan=1 colspan=1>prompt endpoint nevermatched</td><td rowspan=1 colspan=1>fixed</td></tr></table>

Defects found during MAS Core integration tests and execution.

TABLE XXXV. DEFECTS IN THE MIGRATION TEST STAND (NOT IN THE AGENT)
<table><tr><td rowspan=1 colspan=1>Defect</td><td rowspan=1 colspan=1>Effect</td><td rowspan=1 colspan=1>State</td></tr><tr><td rowspan=1 colspan=1>rolled-back install of a unitthat raises on a test input</td><td rowspan=1 colspan=1>about 32 of 41BLOCKED units</td><td rowspan=1 colspan=1>fixed</td></tr><tr><td rowspan=1 colspan=1>unreachable provider scoredas bad translation</td><td rowspan=1 colspan=1>109 FAILED units fromone outage</td><td rowspan=1 colspan=1>fixed; nowINCONCLUSIVE</td></tr><tr><td rowspan=1 colspan=1>test input crashed validationdatabase memory</td><td rowspan=1 colspan=1>2–4 units per run nevertried</td><td rowspan=1 colspan=1>fixed</td></tr><tr><td rowspan=1 colspan=1>dependent shown wrongupstream signature</td><td rowspan=1 colspan=1>3 callers invoked wrongprocedure</td><td rowspan=1 colspan=1>fixed</td></tr><tr><td rowspan=1 colspan=1>COMMIT/ROLLBACKblocks in per-attempttransaction</td><td rowspan=1 colspan=1>9 units not evaluable</td><td rowspan=1 colspan=1>fixed (disposablecopy)</td></tr><tr><td rowspan=1 colspan=1>decomposer mergesstatements in 9 of 34 files</td><td rowspan=1 colspan=1>units not checkable perunit</td><td rowspan=1 colspan=1>open</td></tr></table>

Migration evaluator defects observed during differential validation; an earlier resolver error blocked 21 dependency groups, representing 57% of the affected cascade.

## B. The Degenerate-Agent Check

The most consequential rule is simple: before trusting any aggregate metric, score the agent that does nothing and see whether it wins. Let f be a candidate headline metric, a<sub>∅</sub> the null agent (an empty migration, an empty answer, a run without tool calls) and $a _ { \mathrm { r e f } }$ a reference agent. A metric is admissible as a headline only if

$$
\begin{array} { r } { f ( \boldsymbol { a } _ { \infty } ) + \varepsilon \leq f ( \boldsymbol { a } _ { \mathrm { r e f } } ) , \ \varepsilon > 0 , } \end{array}\tag{7}
$$

and a loop that optimises against f (a prompt-tuning or self-learning loop) must not be pointed at a metric that fails (7). The check caught two different metrics in the migration direction. “PostgreSQL accepted the SQL” rewards wrapping statements in a function body in which nothing is resolved, and mean behavioural fidelity rewards silence, because the similarity of two empty sequences is 1.0 and Oracle itself is silent on 11 of the first 20 curated units. An empty migration outscored the agent on every headline metric that had been published up to that point. The rule has a practical corollary used throughout the testbed: every aggregate is accompanied by the numerator and denominator and by the evidence level at which it was obtained. The four acceptance levels and their observed values are summarised in Table XXXVI.

## C. Tiers of Acceptance and the Behavioural Dif erential

What “accuracy” means for a migration turned out to depend on the tier at which it is measured (Table XXXVI). The reference loads on a clean database for only 35.93% of the units in isolation, 40.26% with dependency closure and

49.50% as a union. The agent’s output loads less often than the reference by 6.08 percentage points $( \mathsf { p } \approx 5 \cdot 1 0 ^ { - 1 1 } )$ . But about 78% of every “ok” verdict proved nothing, because a function body in PostgreSQL is accepted without resolving the references inside it; restricted to verdicts that actually resolved their references, both headline acceptance differences collapse to nulls (−6.15 pp becomes −0.45 pp, p = 0.39). Finally, execution-judged against a captured real Oracle run, no non-trivial exact behavioural match was found on 20 curated units until the fifth prompt fix. A prompt loop pointed at raw acceptance would have learned to hide statements inside function bodies; this is exactly what separated two real prompt versions early on.

TABLE XXXVI. FOUR TIERS OF “ACCURACY” IN THE MIGRATION DIRECTION
<table><tr><td rowspan=1 colspan=1>Tier</td><td rowspan=1 colspan=1>Measures</td><td rowspan=1 colspan=1>Headline</td></tr><tr><td rowspan=1 colspan=1>Corpus filter</td><td rowspan=1 colspan=1>does the reference load</td><td rowspan=1 colspan=1>35.93% / 40.26% / 49.50%</td></tr><tr><td rowspan=1 colspan=1>Candidate vsreference</td><td rowspan=1 colspan=1>does the agent&#x27;s outputload</td><td rowspan=1 colspan=1> $- 6 . 0 8 \mathrm { p p } , \mathrm { p } \approx 5 { \cdot } 1 0 ^ { - 1 1 }$ </td></tr><tr><td rowspan=1 colspan=1>Acceptance</td><td rowspan=1 colspan=1>what “PostgreSQLaccepted it&quot; proves</td><td rowspan=1 colspan=1>≈78% of verdicts provednothing</td></tr><tr><td rowspan=1 colspan=1>Behaviouraldifferential</td><td rowspan=1 colspan=1>does it behave likeOracle</td><td rowspan=1 colspan=1>0 non-trivial exact matcheson 20 units, until fix 5</td></tr></table>

Observed migration acceptance measurements; the three percentages describe reference installation in isolation, with dependency closure and as their union.

In the depth study the verdict is therefore read through three channels (Table XXXVII). The agent’s own status says whether a unit installed and ran. A differential of errors says whether the translation raises an error Oracle does not, or misses one Oracle raises. A per-unit output comparison replays each translated unit in order on PostgreSQL and compares with what Oracle printed. Every FAILED unit and every disagreement between channels is then read by hand, and where a judgement depends on Oracle’s behaviour it is executed on a real Oracle instance rather than assumed. Two consequences follow. A translation that correctly raises the same error Oracle raises must not be scored FAILED; and when the agent was taught Oracle’s SELECT … INTO semantics, ACCEPTED went down, since two translations that had silently returned NULL now raised Oracle’s error.

TABLE XXXVII. READING A MIGRATION VERDICT THROUGH THREE CHANNELS
<table><tr><td rowspan=1 colspan=1>Channel</td><td rowspan=1 colspan=1>Question</td><td rowspan=1 colspan=1>Instrument</td></tr><tr><td rowspan=1 colspan=1>Status</td><td rowspan=1 colspan=1>did it install and run?</td><td rowspan=1 colspan=1>agent&#x27;s own report</td></tr><tr><td rowspan=1 colspan=1>Errors</td><td rowspan=1 colspan=1>extra or missing errors relativeto Oracle?</td><td rowspan=1 colspan=1>behaviour differential tool</td></tr><tr><td rowspan=1 colspan=1>Output</td><td rowspan=1 colspan=1>does each unit print what Oracleprinted?</td><td rowspan=1 colspan=1>per-unit transcript replay</td></tr><tr><td rowspan=1 colspan=1>Manual</td><td rowspan=1 colspan=1>is the judgementOracle-dependent?</td><td rowspan=1 colspan=1>execute on real Oracle</td></tr></table>

Observed differential verification channels used in the migration experiment.

## D. The Score Record, the Source Hierarchy and Control Arms

The programme fixed what a score is in a decision record before the stands were built, and the testbed inherits it. A score is a record, not a number: it carries the axis, the value and its type, the source, the arm, the number of samples, an explicit and non-blank denominator, the run and the requested model identifier and a timestamp (Table XXXVIII). The measurement μ of Equation (2) is a projection of this record. Two details were learned from defects. A denominator that is present is not a denominator that is stated: a whitespace string once passed a length check and produced a record that was technically complete, said nothing and was incomparable with anything. And the requested model identifier, not the one the gateway answered with, keys the record, because the gateway may return a prefixed variant of the same deployment; the returned identifier belongs in the trace as evidence of what actually served the call, which makes a silent fallback visible.

TABLE XXXVIII. FIELDS OF A SCORE RECORD
<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Rule</td></tr><tr><td rowspan=1 colspan=1>axis, value, value_type</td><td rowspan=1 colspan=1>boolean, categorical, numeric or text; text isevidence, never the value</td></tr><tr><td rowspan=1 colspan=1>source</td><td rowspan=1 colspan=1>one of four levels (Table XXXIX)</td></tr><tr><td rowspan=1 colspan=1>arm</td><td rowspan=1 colspan=1>mandatory: system under test or control with thecomponent removed</td></tr><tr><td rowspan=1 colspan=1>n, denominator</td><td rowspan=1 colspan=1>explicit and non-blank; validated on content, notpresence</td></tr><tr><td rowspan=1 colspan=1>run_id, model_id</td><td rowspan=1 colspan=1>model_id is the requested deployment; thereturned id is traced</td></tr><tr><td rowspan=1 colspan=1>timestamp</td><td rowspan=1 colspan=1>time of the measurement</td></tr></table>

Fields established by the implemented measurement schema and evidence policy.

Scores come from four sources, ordered by what they can verify (Table XXXIX). A lower level never overrides a higher one on the same question; where levels disagree the higher level is the verdict and the disagreement is recorded, because it is usually the most informative thing in the run. A judge may triage, prioritise and explain; it may not decide, and it is never an acceptance gate. The rule has a numerical justification in the migration programme that preceded the testbed, which kept LLMs out of its judging loop: executability moved from 72.0% to 97.3% under a specification, while behavioural exact match on the same artefacts was 2 to 6 of 20, and 42 structured probes against a live Oracle instance returned 19 correct results in both the specification and the control arm [46]. A judge reading the generated code would have seen well-formed PostgreSQL in both arms and scored them close; the difference was not in the text.

TABLE XXXIX. SOURCE HIERARCHY OF SCORES
<table><tr><td rowspan=1 colspan=1>Level</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>Use</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>verifiable outcome: execution, differentialcomparison with a reference system,deterministic check against labelledground truth</td><td rowspan=1 colspan=1>decides</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>human annotation</td><td rowspan=1 colspan=1>small-n; only sourcefor intent</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>static programmatic checks (string,structure, schema, compliance)</td><td rowspan=1 colspan=1>objective about theartefact, not itsbehaviour</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>LLM-as-a-judge</td><td rowspan=1 colspan=1>advisory; never anacceptance gate</td></tr></table>

Correspondence to evidence levels: level 1 ≈ L4/L2; level 3 ≈ L1; level 4 ≈ L3 when the judge compares with a reference.

Two further rules complete the model. A score without a control arm is not reportable: one axis scored 100% in every arm and carried no information about the system at all, and where a control arm is impossible the score is labelled “uncontrolled” and cannot support a claim of improvement. And compliance is reported beside outcome, never instead of it: in one experiment adding a specification raised the compliance axis it targeted while the number of invalid declarations rose from 0 in the control to 10, and a composite score would have shipped that release as a win. Finally, offline measurements with a fixed dataset, a control arm and thresholds per axis are the only ones that may gate a change or support a claim, whereas online traces without ground truth detect drift and nominate candidates for the offline set. The price of this discipline is evaluation cost: control arms double generation and thresholds need repeats, which the decision record estimates at two to three times the cost of a naive evaluation, and we accept it.

## E. Fail-Closed Reporting and the Verdict Taxonomy

A metric that cannot be computed is reported as an error, not as zero. In the chatbot smoke run the semantic metrics requested by the dataset were reported as errors because no judge was configured; in the interface a run with incomplete token usage is shown with a warning rather than a zero cost. An infrastructure failure is not a verdict on the model: a network outage once turned into 109 FAILED migration units and is now recorded as INCONCLUSIVE; an unreachable triage agent aborts instead of guessing a retry; an unreachable aggregator yields an absent verdict. Each stage has a distinct vocabulary (Table XL).

TABLE XL. VERDICT TAXONOMY BY STAGE
<table><tr><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Verdicts</td><td rowspan=1 colspan=1>Outage handling</td></tr><tr><td rowspan=1 colspan=1>Migration unit</td><td rowspan=1 colspan=1>ACCEPTED, FAILED, BLOCKED,INCONCLUSIVE, UNSUPPORTED</td><td rowspan=1 colspan=1>INCONCLUSIVE</td></tr><tr><td rowspan=1 colspan=1>Fix cycle</td><td rowspan=1 colspan=1>CONFIRMED, PARTIAL, FAILED,INCONCLUSIVE</td><td rowspan=1 colspan=1>INCONCLUSIVE</td></tr><tr><td rowspan=1 colspan=1>Chatbot run</td><td rowspan=1 colspan=1>PASS, FAIL, degraded</td><td rowspan=1 colspan=1>error, not zero</td></tr><tr><td rowspan=1 colspan=1>Goalassessment</td><td rowspan=1 colspan=1>trustworthy: yes, no, absent</td><td rowspan=1 colspan=1>absent</td></tr><tr><td rowspan=1 colspan=1>Triage</td><td rowspan=1 colspan=1>retry, resume, abort</td><td rowspan=1 colspan=1>abort</td></tr></table>

## F. Noise Floor, Repeats and Comparability

Repeated runs of an identical plan do not return identical numbers. Two identical goals over 20 cases of the P1 stand differed by 0.056 in conversion accuracy; a semantic judge changed its verdict on 16.2% of repeated items at temperature $0 ;$ and in the specification benchmark of the same group arm deltas across identical re-runs had a median spread of 14.4 percentage points and a worst case of 32.7 [46]. A judge that grades identical retrieved context twice disagrees with itself too: on a retrieval-augmented evaluation with identical context in two databases, answer relevancy came out as 0.4923 and 0.4111, a measured noise floor of up to 0.081, which turned an earlier reported gap of 0.834 against 0.519 on six cases into noise. Such variability, a known obstacle to reproducible AI research [55], [56] and to meaningful significance claims [53], [54], has two consequences for the testbed. Numbers from the first P1 test runs were deliberately not published, because a figure in a report tends to be quoted as a statement about the model. And a difference is reported as an effect only if it exceeds the spread of the repeats of that axis:

$$
\mathrm { r e p o r t } ( \Delta ) \Leftrightarrow | \Delta | > s + \tau , \ s =
$$

$$
\mathrm { r a n g e } ( \nu _ { 1 } ) + \mathrm { r a n g e } ( \nu _ { 2 } ) , \ \nu _ { 1 } \sim \nu _ { 2 } ,\tag{8}
$$

Here, Δ denotes the difference between the mean scores of the two experimental arms, range(vᵢ) is the difference between the maximum and minimum scores obtained from independent repetitions of arm i on the same fixed evaluation dataset, and τ is a numerical tolerance introduced to account for floating-point representation. The two arms must satisfy the comparability relation defined in Equation (4), with the investigated treatment being the only intended difference between their experimental configurations.

The criterion is evaluated separately for each metric rather than using a single global threshold. A difference is considered reportable only when its magnitude exceeds the combined within-arm ranges and the numerical tolerance. The sum of the two ranges provides an upper bound on the variation of pairwise differences under arbitrary re-pairings of the repeated measurements. If the observed difference does not exceed this threshold, the coarse rule does not establish an effect. This outcome should not be interpreted as evidence that the two configurations are equivalent.

The need for a metric-specific threshold is illustrated by the observed experiments. A nominal improvement of 18.7 percentage points was not retained because the corresponding repeat spread reached 20.3 percentage points, whereas an improvement of 17.4 percentage points exceeded the 12.3-point spread observed for another metric. In a separate case, the estimated effect changed sign between nominally identical repetitions, from +23.3 to −9.3 percentage points. Pairing stochastic repetitions solely by their positions can also underestimate variability: one observed pairing yielded five identical differences of +2.86 percentage points, while alternative pairings produced a spread of up to 5.71 percentage points.

A zero observed spread must not be interpreted as proof of high precision, particularly when only a few repetitions are available. If the difference and the spread are both zero, no improvement or regression is reported. The numerical tolerance prevents floating-point rounding near the decision boundary from generating a spurious result. Comparisons with fewer than two independent repetitions per arm are classified as unreplicated according to Equation (9). The decision rule is deliberately conservative and does not replace a calibrated statistical test when sufficient independent measurements are available.

The same principle appears in the automation experimentation strategy, where an effect smaller than the repeat spread is not reported as an improvement, and the paired comparisons of the migration depth study use exact tests where a count is available (for example a McNemar test, p = 0.208, for whole-file versus split migration in Section VIII). Equation (8) is deliberately conservative, and the simulation of Section VIII-I shows how conservative: with five repeats per arm it needs a difference of about 5.8 per-run standard deviations to be detected with 80% power, and, because the range of a sample grows with its size, it becomes less sensitive as repeats are added. We therefore use it as the guard against false effects and add a second tier for the cases in which the repeats allow a calibrated test. The verdict has three states: efect when Equation (8) holds; suggestive when it does not hold but a Welch test of the repeats rejects equality at level α = 0.05; and no result otherwise,

verdict = unreplicated if $\mathfrak { r } < 2 ;$ effect $\operatorname { i f } \mathbf { r } \geq 2 \land | \Delta | > \mathbf { s } + \tau ;$

$$
\mathrm { s u g g e s t i v e ~ i f ~ r \geq 2 ~ } \bigwedge | \Delta | \leq \mathsf { s } + \tau \wedge \mathsf { p W } < \alpha ;
$$

$$
\mathrm { n o r e s u l t o t h e r w i s e . }\tag{9}
$$

Here r is the number of repeated independent runs per arm, and pW is the Welch p-value. The “suggestive” category is advisory and cannot gate a change; with small r its calibration must be treated cautiously. A single run is reported as “unreplicated”, not as evidence of an effect, even if its nominal range is zero. The reference implementation described in Section VIII-K illustrates these rules; no single-run claim is promoted to a confirmed improvement.

Algorithm 7. Noise-aware comparison and Pareto selection.  
Input: models M, task constraints H, objectives O with directions,   
metamodels Ψ   
Output: Pareto set, ties, exclusions with reasons   
1 M′ ← ∅; Excl ← ∅   
2 for each m $\in M$   
3 if some $) \in O$ or constraint in H unmeasured then Excl += (m, “not   
measured”); continue   
4 if evaluation contexts differ outside the declared treatment then Excl +=   
(m, “other context”); continue   
5 if violates H then Excl += (m, constraint); continue   
6 M′ += m   
7 P ← exact non-dominated set of M′ using direction-adjusted recorded   
values   
8 for each pair and objective: if both have ≥2 repeats and |Δo| ≤ range1 +   
range2 + τ, annotate “uncertain difference”   
9 if repeats are unavailable, annotate “unreplicated”; never treat uncertainty   
annotations as transitive equalities   
10 return P, uncertainty annotations, exclusions and evidence counts

## G. Gated Fix Cycles

In the migration direction fixes are not tuned against a score, because every score that was tried rewarded the wrong behaviour at some point; an empty translation once outscored the agent. Each fix instead passes through a gated cycle (Fig. 14, Algorithm 8). Evidence comes from a real run: a named Oracle construct, the units it breaks, verified on both databases. The change and three tiers of checks are pre-registered before the run. Tier A is the gate, a deterministic check on the generated SQL, for example that no DATE type survives translation. Tier B, reported only, asks whether the target units improved. Tier C explains every other unit that changed status. A run then yields an honest verdict, and failed fixes are kept in the ledger.

![](images/857465a8b1bdbe7c9d2a91c58ed6c95a5b8c574b6f96752088fc09b9b340d3c8.jpg)  
Tier A: deterministic gate on generated SQL. Tier B: target units, reported. Tier C: every other status change explained. Failed fixes stay in the ledger

Fig. 14. Gated fix cycle: evidence, pre-registration, run, verdict.  
Input: evidence E (construct, affected units, both-database verification)   
Output: verdict recorded in the ledger   
1 pre-register change Δ (prompt section or code), tier A gate, tier B targets,   
tier C scope   
2 commit pre-registration // before the run   
3 run ← full or targeted run with Δ   
4 if not TierA(run) then return FIX\_FAILED   
5 if infra error then return INCONCLUSIVE   
6 report tier B (targets improved?); explain every tier C status change   
7 return verdict ∈ {FIX\_CONFIRMED, FIX\_PARTIAL, FIX\_FAILED}   
from tiers B and C   
8 failed fixes remain in the ledger

The ledger of fixes from FIX-011 to FIX-056 contains 46 cycles: 26 confirmed, 8 partial, 3 failed, 1 inconclusive and 8 from the last two days of the observation period whose Tier A check holds and whose verdict awaits the next full run. One example shows why checking each unit against Oracle matters: a regression caused by an earlier fix (the cheaper model began to keep :name bind placeholders in dynamic SQL, most likely after the NULL-concatenation fix) retained its ACCEPTED status and was visible only by comparing with what Oracle printed.

## H. Cost Model and Costper Evidence

Because the cost of measuring is itself a result, we define it explicitly. For a run of model m on application a the three spends of Table XVII are accumulated separately, and the cost of the measurement is the vector $( C _ { \mathrm { t } } , \bar { C } _ { \mathrm { j } } , C _ { \mathrm { g } } ) ,$ , never its sum. For decision purposes a scalar is sometimes needed; the natural one is the cost per unit of evidence at a chosen level, for example per demonstrably correct unit, Application-specific experimental costs are listed separately in Table XLI.

$$
c _ { \mathrm { d c } } = \left( C _ { \mathrm { t } } + C _ { \mathrm { j } } + C _ { \mathrm { g } } \right) / N _ { \mathrm { d c } } ,\tag{10}
$$

where the sum is taken explicitly and labelled as the cost of the experiment, not a property of the model. With the reported figures, a full production-routed migration run on the 216-unit corpus costs about USD 3 to 4, and 108 units were demonstrably correct, which gives a derived cost of about USD 0.03 to 0.04 per demonstrably correct unit for the experiment as a whole; the P1 first cycle cost about USD 4 for 1,802 files (about USD 0.002 per file) of which the judge accounted for about three quarters. These derived numbers are included to show the form of the computation; they are not comparable across applications, because the evidence levels differ.

TABLE XLI. REPORTED COSTS OF THE EXPERIMENTS
<table><tr><td rowspan=1 colspan=1>Experiment</td><td rowspan=1 colspan=1>Reported cost</td><td rowspan=1 colspan=1>Derived</td></tr><tr><td rowspan=1 colspan=1>P1 first cycle, 1,802 files</td><td rowspan=1 colspan=1>conversion ≈ USD 1.2; judge≈ USD 3; total ≈ USD 4</td><td rowspan=1 colspan=1>≈ USD 0.002 per file;judge ≈ 2.5 ×conversion</td></tr><tr><td rowspan=1 colspan=1>Migration depth, full run,216 units</td><td rowspan=1 colspan=1>≈ USD 3-4</td><td rowspan=1 colspan=1>≈ USD 0.03–0.04 perdemonstrably correctunit (108)</td></tr><tr><td rowspan=1 colspan=1>Migration breadth, 1,448tasks</td><td rowspan=1 colspan=1>10.0 M tokens (agent)</td><td rowspan=1 colspan=1>3.0% fewer tokensthan the cheap-modelbaseline</td></tr><tr><td rowspan=1 colspan=1>Intent classification, onerepeat</td><td rowspan=1 colspan=1>72 model calls; ≈ 37.4 ktokens</td><td rowspan=1 colspan=1>mean 1,385 ms percall</td></tr><tr><td rowspan=1 colspan=1>Shared proxy key (severalusers)</td><td rowspan=1 colspan=1>≈ USD 331 of USD 500rolling 30-day budget on2026-09-23</td><td rowspan=1 colspan=1>quota may decidebefore price</td></tr></table>

Measured costs and arithmetically derived unit costs; approximate totals reflect source rounding.

## I. Statistical Treatment

The stands report counts with denominators and use exact tests where the data are paired counts. The comparison of whole-file against split migration used a McNemar test on the discordant pairs of exact matches $( \mathtt { p } = 0 . 2 0 8 )$ , and the acceptance-tier comparison reported a difference with p ≈ $5 { \cdot } 1 0 ^ { \dot { - } 1 1 }$ that collapsed to ${ \mathrm { p } } = 0 . 3 9$ once unresolved verdicts were removed. Two practices are worth stating. First, an aggregate is accompanied by its numerator and denominator and its evidence level, so that a reader can see that 284 of 1,448 tasks, not 1,330, reached execution in PostgreSQL. Second, results that depend on a single run are labelled as such, and no confidence interval is claimed where the number of repeats is one. The coarse noise-floor rule of Equation (8) is a substitute for a test, not a stronger version of it: it is applied when a repeat spread exists and the number of repeats is too small for a formal test, which is the case for most of the application experiments.

## J. Instrument-First Diagnosis

When a number is surprising, the question is where the cause lies. The measured defects of Tables XXXIV and XXXV show that a stand can mislead for a long time before the model is at fault: a resolver bug held up 57% of a dependency cascade, a network outage turned into 109 failed units, a test input crashed the validation database and left the following units untried. The diagnostic procedure of the method therefore tests the instrument before the system (Fig. 15, Algorithm 9).

![](images/66eafb89ec21fb9a738caf1cfb9bd627ee8056e12e573166aebb0cc09e300aa9.jpg)  
Left: question asked; right: outcome if the answer is yes. A no answer moves down.

Fig. 15. Instrument-first diagnostic ladder: each question has an outcome if the answer is yes; a no answer moves down to the next question.  
Input: result R, run log, metricf, repeats, application layers L   
Output: diagnosis with evidence and action   
1 if outage or provider error in log then return INCONCLUSIVE; rerun   
2 if harness or test-input defect reproduces without the model   
3 then return instrument defect; fix instrument; rerun // e.g. rollback,   
memory crash   
4 if not admissible(f) // null agent must lose, Eq. (7)   
5 then return demotef; do not optimise it   
6 if verdict(R) = no result then return no result; add repeats   
7 for each layer l in L in causal order   
8 if not passes(l) then return root cause l, evidence, candidate fix   
9 return no defect found; record as pass

It asks, in order, whether an outage or provider error explains the result (then the verdict is inconclusive and the units are rerun), whether a harness or test-input defect explains it (then the instrument is fixed and the run repeated), whether the headline metric passes the null-agent check of Equation (7) (then it may be optimised), whether the difference exceeds the noise floor (else it is reported as no result), and only then diagnoses the system itself, layer by layer, in the causal order of Fig. 12. The order is not arbitrary. An instrument defect can masquerade as any system defect, whereas a system defect cannot masquerade as an instrument defect, so ruling out the instrument first is the only ordering that does not confound the two.

## K. Diagnostic Knowledge Base and Symptom Taxonomy

Diagnosis is only repeatable if the knowledge about causes is kept apart from the diagnosed system, which is the lesson of the knowledge-based diagnostics of computer facilities [81], [82]. In the testbed this knowledge takes two forms. The first is the symptom taxonomy of Table XLII: for a symptom that was actually observed, the candidate causes, the evidence that separates them and the resulting action. The second is the fix ledger of the migration direction, an append-only record of 46 cycles in which every entry states a named construct, the units it breaks, the predicates that gate it, the verdict and the cost. A search service over such a base, with explicit criteria for matching the terms of different entries [84]–[86], is what turns a collection of reports into a diagnostic instrument: a new failure is first matched against known symptoms before anyone reads a trace.

## L. Negative Controls

A negative control is an instrument that must produce a null result; if it does not, the instrument is broken. The stands use several (Table XLIII). The null agent tests whether a metric rewards emptiness. A keyword or rule baseline as the control arm of a classifier shows what an instrument reports when the system contributes nothing; in the intent experiment it scored 0.000, which satisfied the control formally and carried almost no information. A no-specification arm shows what the models do without the component under study. And in the exploratory analysis of the Oracle script dataset, a random control column was added to the features of a random-forest model; a real feature is taken to carry signal only if its importance exceeds that of the random column, which confirmed that the core SQL and PL/SQL counts carry genuine predictive weight rather than noise. The same device applies to any attribute whose predictive value is in question: add a random attribute and require it to lose.

TABLE XLII.  
SYMPTOM TAXONOMY OBSERVED IN THE STANDS
<table><tr><td rowspan=1 colspan=1>Application</td><td rowspan=1 colspan=1>Symptom</td><td rowspan=1 colspan=1>Candidate causes</td><td rowspan=1 colspan=1>Separating evidence</td><td rowspan=1 colspan=1>Action</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>low recall, many refusals,short outputs</td><td rowspan=1 colspan=1>truncation by output cap; voluntaryabbreviation</td><td rowspan=1 colspan=1>completion length against the cap(maximum 7,548 of 32,768 tokens)</td><td rowspan=1 colspan=1>report abbreviation as modelbehaviour; add a truncated status</td></tr><tr><td rowspan=1 colspan=1>migration</td><td rowspan=1 colspan=1>FAILED unit</td><td rowspan=1 colspan=1>defect; faithful raise; invalid testinput; harness limitation; no-op</td><td rowspan=1 colspan=1>execute on the real Oracle instance</td><td rowspan=1 colspan=1>reclassify; fix defect or harness</td></tr><tr><td rowspan=1 colspan=1>migration</td><td rowspan=1 colspan=1>BLOCKED unit</td><td rowspan=1 colspan=1>upstream failure; upstream correctbut raising by design</td><td rowspan=1 colspan=1>trace to the upstream and its failure stage</td><td rowspan=1 colspan=1>keep a correct upstream installedfor dependants</td></tr><tr><td rowspan=1 colspan=1>migration</td><td rowspan=1 colspan=1>mass of FAILED after oneevent</td><td rowspan=1 colspan=1>provider outage; model regression</td><td rowspan=1 colspan=1>provider status; token usage of failed calls</td><td rowspan=1 colspan=1>mark INCONCLUSIVE; rerun</td></tr><tr><td rowspan=1 colspan=1>migration</td><td rowspan=1 colspan=1>ACCEPTED that Oraclecontradicts</td><td rowspan=1 colspan=1>silently wrong translation</td><td rowspan=1 colspan=1>per-unit output comparison; second modeldisagrees</td><td rowspan=1 colspan=1>fix construct; do not count ascorrect</td></tr><tr><td rowspan=1 colspan=1>chatbot</td><td rowspan=1 colspan=1>FAIL scenario</td><td rowspan=1 colspan=1>execution; routing; retrieval;grounding; answer quality</td><td rowspan=1 colspan=1>first failing layer in causal order</td><td rowspan=1 colspan=1>targeted re-test with an approvedalternative</td></tr><tr><td rowspan=1 colspan=1>automation</td><td rowspan=1 colspan=1>no effect but successstated</td><td rowspan=1 colspan=1>unbacked claim; candidminimalism</td><td rowspan=1 colspan=1>final answer against recorded effects</td><td rowspan=1 colspan=1>report claim-effect gap</td></tr><tr><td rowspan=1 colspan=1>automation</td><td rowspan=1 colspan=1>tool error on an inventedidentifier</td><td rowspan=1 colspan=1>hallucinated argument</td><td rowspan=1 colspan=1>argument against the frozen world state</td><td rowspan=1 colspan=1>count in hallucinated-call rate</td></tr><tr><td rowspan=1 colspan=1>judge</td><td rowspan=1 colspan=1>different scores onidentical context</td><td rowspan=1 colspan=1>judge noise</td><td rowspan=1 colspan=1>repeat on identical context</td><td rowspan=1 colspan=1>report noise floor; do not gate</td></tr><tr><td rowspan=1 colspan=1>comparison</td><td rowspan=1 colspan=1>two systems returnidentical results</td><td rowspan=1 colspan=1>true equivalence; comparison thatcannot differ</td><td rowspan=1 colspan=1>does the experiment exercise the index orpath under test?</td><td rowspan=1 colspan=1>redo at realistic scale</td></tr></table>

Symptoms are derived from the execution and validation observations analysed in Sections VII and VIII.

TABLE XLIII. NEGATIVE CONTROLS USED IN THE STANDS
<table><tr><td rowspan=1 colspan=1>Control</td><td rowspan=1 colspan=1>What it must show</td><td rowspan=1 colspan=1>Observed</td></tr><tr><td rowspan=1 colspan=1>null agent (emptymigration, empty answer)</td><td rowspan=1 colspan=1>headline metric does notreward emptiness</td><td rowspan=1 colspan=1>empty migration outscoredthe agent on publishedmetrics</td></tr><tr><td rowspan=1 colspan=1>keyword or rule baseline(control arm)</td><td rowspan=1 colspan=1>what the instrumentreports with no system</td><td rowspan=1 colspan=1>0.000 on intent; 100% on anaxis in every arm carried noinformation</td></tr><tr><td rowspan=1 colspan=1>no-specification arm</td><td rowspan=1 colspan=1>effect attributable to thecomponent studied</td><td rowspan=1 colspan=1>executability 72.0% →97.3%; agreement 83.1% →83.8%</td></tr><tr><td rowspan=1 colspan=1>random control feature</td><td rowspan=1 colspan=1>feature importanceexceeds noise</td><td rowspan=1 colspan=1>core SQL and PL/SQLcounts exceeded the randomcolumn</td></tr><tr><td rowspan=1 colspan=1>exact scan against index</td><td rowspan=1 colspan=1>comparison can come outdifferently</td><td rowspan=1 colspan=1>identical rankings wereguaranteed at 31 chunks</td></tr><tr><td rowspan=1 colspan=1>identical repeat</td><td rowspan=1 colspan=1>noise floor of theinstrument</td><td rowspan=1 colspan=1>judge relevancy 0.4923against 0.4111 on identicalcontext</td></tr></table>

Negative controls used in the evaluation protocols; see Sections VII-B, VII-F and VIII-G.

## M. Cross-Model Disagreement as a Defect Detector

A property that is specific to a testbed that runs many models on the same application is that disagreement between models is itself a signal. If two models produce different behaviours on the same unit under comparable evaluation conditions, the disagreement identifies a candidate for further verification using the stronger evidence levels defined in Table IV. Behavioural differences do not necessarily imply that either model is incorrect, since multiple valid outputs may satisfy the same task requirements. In the migration direction the first silently wrong accepted translation was found this way: a second model, tried for cost reasons, disagreed with the baseline on one unit. The testbed therefore stores, for every application context, the per-unit outcomes of every model, so that a disagreement query is a join and not a new experiment. The signal is asymmetric: agreement is weak evidence of correctness, because models share training data and defaults, whereas disagreement is strong evidence that something deserves a look. This asymmetry agrees with the observation that the quality of a unit test suite determines how much an apparently correct generation can be trusted [35], and it motivates the use of verified outcomes (levels 1 and 2 of Table XXXIX) to settle the disagreement.

## N. Separation of Roles and Leakage Hygiene

The model under test is kept apart from the models that judge and decide. In the chatbot direction the judge and the controller have separate roles, ground truth is never passed to the chatbot, and in the specification-evaluation work of the same group ground truth is mechanically sealed from inference and checked with canary and label-permutation tests, with judges compared against trivial baselines such as specification length [45]. These practices follow the recommendations on contamination [49], [51], [52] and on judge bias [43], [44] and are the reason why the testbed prefers deterministic computation wherever one exists. A further consequence concerns cost: because the judge’s spend can be a substantial fraction of the model’s, a cross-vendor judge to control same-vendor self-preference is still an open item of the P1 stand.

## VIII. EXPERIMENTS AND RESULTS

This section answers RQ3 using the recorded experimental measurements from the three application-specific evaluation environments. Dataset versions, configurations, denominators and evidence levels are retained with each result. Because experiment designs differ between applications, cross-application numerical comparisons are not interpreted as a single model ranking. Values for which only one run is available are identified explicitly. The full comparison set and experimental denominators are summarised in Table XLIV.

TABLE XLIV. SUMMARY OF THE EXPERIMENTS REPORTED
<table><tr><td rowspan=1 colspan=1>ID</td><td rowspan=1 colspan=1>Application</td><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>Models / variables</td><td rowspan=1 colspan=1>Runs</td><td rowspan=1 colspan=1>Section</td></tr><tr><td rowspan=1 colspan=1>E1</td><td rowspan=1 colspan=1>P1 conversion</td><td rowspan=1 colspan=1>1,802 Oracle files</td><td rowspan=1 colspan=1>gpt-4.1-nano; judgegpt-4.1-mini</td><td rowspan=1 colspan=1>1 full cycle</td><td rowspan=1 colspan=1>VIII-A</td></tr><tr><td rowspan=1 colspan=1>E2</td><td rowspan=1 colspan=1>Automation</td><td rowspan=1 colspan=1>49 samples (pilot.v4)</td><td rowspan=1 colspan=1>4 frameworks, gpt-4o</td><td rowspan=1 colspan=1>1 per framework</td><td rowspan=1 colspan=1>VIII-B</td></tr><tr><td rowspan=1 colspan=1>E3</td><td rowspan=1 colspan=1>Automation</td><td rowspan=1 colspan=1>49 samples</td><td rowspan=1 colspan=1>6 models, LlamaIndex</td><td rowspan=1 colspan=1>1 per model</td><td rowspan=1 colspan=1>VIII-B</td></tr><tr><td rowspan=1 colspan=1>E4</td><td rowspan=1 colspan=1>Automation (intent)</td><td rowspan=1 colspan=1>40 frozen cases</td><td rowspan=1 colspan=1>3 deployments, 2 arms</td><td rowspan=1 colspan=1>5 repeats</td><td rowspan=1 colspan=1>VIII-C</td></tr><tr><td rowspan=1 colspan=1>E5</td><td rowspan=1 colspan=1>Chatbot</td><td rowspan=1 colspan=1>mvp_scenarios v6; 6 / 5/ 10 scenarios</td><td rowspan=1 colspan=1>retrieval, Top-K,reranker, temperature, 9models</td><td rowspan=1 colspan=1>1; 15 for finalists</td><td rowspan=1 colspan=1>VIII-D</td></tr><tr><td rowspan=1 colspan=1>E6</td><td rowspan=1 colspan=1>Chatbot evaluation agent</td><td rowspan=1 colspan=1>10 scenarios</td><td rowspan=1 colspan=1>v1 diagnostic; v2adaptive</td><td rowspan=1 colspan=1>1 campaign each</td><td rowspan=1 colspan=1>VIII-D</td></tr><tr><td rowspan=1 colspan=1>E7</td><td rowspan=1 colspan=1>Migration breadth</td><td rowspan=1 colspan=1>1,000 pairs, 1,448 tasks</td><td rowspan=1 colspan=1>agent, cheap LLM,rules, whole-file</td><td rowspan=1 colspan=1>1 per configuration</td><td rowspan=1 colspan=1>VIII-F</td></tr><tr><td rowspan=1 colspan=1>E8</td><td rowspan=1 colspan=1>Migration depth</td><td rowspan=1 colspan=1>216 units, 34 files</td><td rowspan=1 colspan=1>production routing; 2prompt versions</td><td rowspan=1 colspan=1>2 full runs</td><td rowspan=1 colspan=1>VIII-F</td></tr><tr><td rowspan=1 colspan=1>E9</td><td rowspan=1 colspan=1>Fix cycles</td><td rowspan=1 colspan=1>FIX-011...056</td><td rowspan=1 colspan=1>prompt sections andcode</td><td rowspan=1 colspan=1>46 cycles</td><td rowspan=1 colspan=1>VIII-F</td></tr></table>

Experiments, sample sizes and evaluation protocols included in this study.

## A. P1 Conversion: First Full Cycle

The P1 stand (Fig. 16) is split into two independently triggered phases: dataset conversion (dataset, prompt, model, database) and conversion evaluation (stored results, validation, metrics, judge, annotators, report). Because evaluation runs entirely from the database, a no-judge evaluation completed in about 7 s with the conversion backend stopped; this property allows the expensive GPU box planned for generation to be powered off afterwards. The GPU service does not exist yet; its slot is filled by the gateway, and the swap boundary is a single environment variable and a documented generation contract.

![](images/72ac533ad3fdaa81a0e6c9030fa934da284acfa56de649f8f028debbfc777dda.jpg)

Phase 2: conversion evaluation (runs from the database)  
![](images/8703a7ef36b1cd427b20e1c340fdabe032911549cd1f3d25f7f049854caf5920.jpg)  
Fig. 16. Two-stage conversion evaluation: generation and subsequent evaluation from stored outputs; the live PostgreSQL logic check is outside the completed P1 experiment.

The first full cycle converted 1,802 of 1,802 files with gpt-4.1-nano in 48 min with zero skips, and the evaluation took 14 min 38 s (Table XLV). The whole cycle cost about USD 4 on the caller’s own key, of which the judge, gpt-4.1-mini, accounted for about USD 3 and the conversion for about USD 1.2. The aggregate NLP score was 0.513 against 0.736 on the previous stand that used chunked retrieval; the linter-based syntax error rate improved from 14.87% to 11.65%, and refusals rose from 2.94% to 8.32%. This is not a regression of the stand but the finding: without chunking, the model does not truncate large files mid-token, it abbreviates them voluntarily. About 69 files with references above 5 KB received under 35% of the expected output; one 59.6 KB reference was answered with 269 bytes and a remark that complete conversion required thorough rewriting. The short outputs parse cleanly, which explains the better syntax figure, and fail recall, efficiency and the refusal detector, so the metrics punish the behaviour fairly. Direct mode therefore measures a capability that the chunked flow could not: honest whole-file conversion. The distribution of evaluator scores for this completed run is shown in Fig. 17.

TABLE XLV. P1 FIRST FULL CYCLE WITH GPT-4.1-NANO
<table><tr><td rowspan=1 colspan=1>Quantity</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Coverage</td><td rowspan=1 colspan=1>1,802 / 1,802 converted, 0 skipped, 48 min</td></tr><tr><td rowspan=1 colspan=1>Tokens, conversion</td><td rowspan=1 colspan=1>4.67 M input, 1.81 M output; about USD 1.2</td></tr><tr><td rowspan=1 colspan=1>Output ceiling hit</td><td rowspan=1 colspan=1>0 files (maximum completion 7,548 tokens)</td></tr><tr><td rowspan=1 colspan=1>NLP vs references</td><td rowspan=1 colspan=1>Recall 0.725; BLEU 0.249; chrF 40.44; aggregate0.513</td></tr><tr><td rowspan=1 colspan=1>Judge (0–10)</td><td rowspan=1 colspan=1>correctness 7.50; functional equivalence 7.36;syntactic validity 9.02; clarity 7.59</td></tr><tr><td rowspan=1 colspan=1>Linter</td><td rowspan=1 colspan=1>NSER 11.65% (210 files, 214 errors); 9,402warnings</td></tr><tr><td rowspan=1 colspan=1>Annotators</td><td rowspan=1 colspan=1>mean file efficiency 63.07; refusals 8.32% (150files)</td></tr><tr><td rowspan=1 colspan=1>Evaluation time</td><td rowspan=1 colspan=1>14 min 38 s; judge cost about USD 3</td></tr></table>

Single run; judge gpt-4.1-mini over all 1,802 items with 0 failures. BLEU [39] and chrF [40] are reference-based.

![](images/ac16e5854feb7b24722d4fb319b9ba1b478becc57c09e9e58053106273af3fab.jpg)  
Fig. 17. Judge scores for the P1 first cycle (single run, 1,802 items).

A reliability incident belongs to the methodology rather than to the model. The default metric set initially contained translation edit rate, whose block-shift edit distance is quadratic in file length; it ran for more than 57 minutes without finishing on the full dataset, held the service’s CPU, and the orchestrator’s one-hour timeout reported a misleading “service unreachable”. The metric was removed from the default set (it remains available on request) and the failure message was rewritten to name the actual cause.

## B. Automation: Frameworks and Models

In E2 the 49 pilot samples were executed once per framework with gpt-4o through the gateway, a ten-step limit, the knowledge base off and no retries. The results (Table XLVI, Fig. 18) cover execution outcomes only, without ground-truth scoring. Completion status does not separate the frameworks: every run returned a final answer, none reached the step limit and no tool call failed. LangGraph, LlamaIndex and AutoGen behave almost identically; they usually perform one read and then answer, and on cross-system synchronisation and most escalation samples they write nothing while many of their answers claim the work was done. CrewAI does the most work, producing the most writes and the fewest unbacked claims, at about four times the token cost, which comes mostly from a larger per-call prompt. LlamaIndex is the cheapest and matches LangGraph’s effects almost exactly.

FRAMEWORK COMPARISON (GPT-4O, 49 SAMPLES, ONE RUN EACH)
<table><tr><td rowspan=1 colspan=1>Framework</td><td rowspan=1 colspan=1>LLMcalls</td><td rowspan=1 colspan=1>Toolcalls</td><td rowspan=1 colspan=1>Tokens</td><td rowspan=1 colspan=1>Effects</td><td rowspan=1 colspan=1>Noeffect</td><td rowspan=1 colspan=1>Unbacked</td></tr><tr><td rowspan=1 colspan=1>LangGraph</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>2,260</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>LlamaIndex</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>2,055</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>AutoGen</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>2,210</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>CrewAI</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>8,942</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>5</td></tr></table>

All 49 runs completed in each framework. “Unbacked”: run produced no effect but the final answer states the work was done (heuristic; may over-count when an answer quotes past events). Means per run.

![](images/d688967a056d3a1b8362c4c62dcb360633d5134153a2c8eea02f37b37ce0f394.jpg)

![](images/a19a6cfb79279affbd8dab3e6f1e1bede211de3fc8bc98bf3ede5d01f06e1e68.jpg)

![](images/acccb877086ee0fac7c4ff61225476f1e5002f0c0946f8183266eaf1e24184d6.jpg)  
Fig. 18. Framework comparison: tokens, effects and unbacked success claims.

In E3 the framework was fixed to LlamaIndex and only the model was varied (Table XLVII, Fig. 19). The models fall into two groups. gpt-4o, gpt-5.5, Haiku 4.5 and Gemini 3.1 Pro do the minimum, with about one tool call and 0.5 effects per run. Sonnet 4.6 and Gemini 2.5 Pro carry out the multi-step synchronisation, escalation and incident-triage tasks. Unbacked success claims vary by model: gpt-4o, Haiku 4.5 and Gemini 3.1 Pro often report work that was not done; gpt-5.5 does as little but mostly says so; Sonnet 4.6 and Gemini 2.5 Pro have none. Sonnet 4.6 is the most thorough and the most expensive at about 9.5 k tokens per run; it also takes actions nobody asked for, such as 13 new Confluence pages, and its three tool errors come from invented page identifiers.

Gemini 2.5 Pro is cheap but sent no email on any email\_compose sample, and its unusually small reported first-call prompt should be checked before its numbers are trusted. Haiku 4.5 costs more than its output suggests, since the Claude deployments report a larger first-call prompt, and it never adds Jira comments.

![](images/2de89552fa6b9112358a6fdc0c39912c57033bea79b106079769cc0c7b6cb4c0.jpg)

![](images/167ac51bf6c88c9088548d9cebde2f751d0120862d3bd44427a472fa708b2f77.jpg)

![](images/32cc3e823cee3e874291bbac70efe3bc7daad8f13b5b58e311dec05c35f4ad48.jpg)  
Fig. 19. Model comparison under one framework: tokens, effects and unbacked claims per run.

TABLE XLVII. MODEL COMPARISON (LLAMAINDEX, 49 SAMPLES, ONE RUN EACH)
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>LLM calls</td><td rowspan=1 colspan=1>Tool calls</td><td rowspan=1 colspan=1>Tool errors</td><td rowspan=1 colspan=1>Tokens</td><td rowspan=1 colspan=1>Effects</td><td rowspan=1 colspan=1>No effect</td><td rowspan=1 colspan=1>Unbacked</td></tr><tr><td rowspan=1 colspan=1>gpt-40</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2,055</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>gpt-5.5</td><td rowspan=1 colspan=1>1.8</td><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1,953</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Sonnet 4.6</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>3.2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>9,465</td><td rowspan=1 colspan=1>1.8</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Haiku 4.5</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>5,384</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>11</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>3,147</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>Gemini 2.5 Pro</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>1.9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2,516</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>0</td></tr></table>

Token counts come from each provider’s own tokenizer and are only roughly comparable across families.

The central observation is the size of the effects. Mean effects per run range from 0.5 to 1.8 across the six models and from 0.4 to 0.7 across the four frameworks. In these separate, unreplicated sweeps, observed outcomes vary more across model conditions than framework conditions; this is not a causal model-by-framework comparison. Under the noise rule (Equation 8) these single-run numbers are descriptive: no repeat spread was measured, so a difference such as 0.5 versus 0.4 is not reported as an effect, and the larger range of 0.5 to 1.8 is also descriptive until run-to-run variability is measured. More importantly, the effect count is not a quality score. Whether the additional writes of Sonnet 4.6 and Gemini 2.5 Pro are correct requires ground-truth scoring, which is pending, and the screening stage cannot run until the applicability thresholds are agreed.

Two derived quantities make the claim–effect structure visible (Table XLVIII, Fig. 20). The efect rate is the share of the 49 runs with at least one state-changing effect, and the unbacked share is the fraction of runs without any effect whose final answer nevertheless states that the work was done. Both are computed here from the reported counts, using the same heuristic as the source and with its caveat that it may over-count when an answer quotes past events from the data. The models separate along two nearly independent axes. Sonnet 4.6 and Gemini 2.5 Pro have both a high effect rate and no unbacked claims; gpt-5.5 has a low effect rate but is almost always candid about it (7.1% unbacked share); gpt-4o, Haiku 4.5, AutoGen, LangGraph and LlamaIndex combine a low effect rate with a share of unbacked claims between 46% and 55%. A single score such as task completion would place the second and third groups together, whereas the plane of Fig. 20 separates the candid minimalist from the confident one. This is the quantitative form of the claim that a system’s own report of success overstates it.

TABLE XLVIII. DERIVED CLAIM–EFFECT MEASURES (49 RUNS PER ARM)
<table><tr><td rowspan=1 colspan=1>Arm</td><td rowspan=1 colspan=1>Effect rate</td><td rowspan=1 colspan=1>Unbacked / noeffect</td><td rowspan=1 colspan=1>Unbackedshare</td></tr><tr><td rowspan=1 colspan=1>LangGraph (gpt-40)</td><td rowspan=1 colspan=1>44.9%</td><td rowspan=1 colspan=1>14 / 27</td><td rowspan=1 colspan=1>51.9%</td></tr><tr><td rowspan=1 colspan=1>LlamaIndex (gpt-40)</td><td rowspan=1 colspan=1>44.9%</td><td rowspan=1 colspan=1>14 /27</td><td rowspan=1 colspan=1>51.9%</td></tr><tr><td rowspan=1 colspan=1>AutoGen (gpt-40)</td><td rowspan=1 colspan=1>40.8%</td><td rowspan=1 colspan=1>16 / 29</td><td rowspan=1 colspan=1>55.2%</td></tr><tr><td rowspan=1 colspan=1>CrewAI (gpt-40)</td><td rowspan=1 colspan=1>57.1%</td><td rowspan=1 colspan=1>5 /21</td><td rowspan=1 colspan=1>23.8%</td></tr><tr><td rowspan=1 colspan=1>gpt-5.5</td><td rowspan=1 colspan=1>42.9%</td><td rowspan=1 colspan=1>2/28</td><td rowspan=1 colspan=1>7.1%</td></tr><tr><td rowspan=1 colspan=1>Claude Sonnet 4.6</td><td rowspan=1 colspan=1>73.5%</td><td rowspan=1 colspan=1>0 / 13</td><td rowspan=1 colspan=1>0.0%</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>51.0%</td><td rowspan=1 colspan=1>11 / 24</td><td rowspan=1 colspan=1>45.8%</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1>44.9%</td><td rowspan=1 colspan=1>9 /27</td><td rowspan=1 colspan=1>33.3%</td></tr><tr><td rowspan=1 colspan=1>Gemini 2.5 Pro</td><td rowspan=1 colspan=1>65.3%</td><td rowspan=1 colspan=1>0 / 17</td><td rowspan=1 colspan=1>0.0%</td></tr></table>

Derived by the authors from the counts of Tables XLVI and XLVII; the gpt-4o/LlamaIndex row is shared by both comparisons.

![](images/7cda8fbcdb5a6b7985a4cf19278ea7c779b6e3d7fe116a85e7b8d2613b99d26b.jpg)  
Fig. 20. Claim–effect plane (derived): share of runs with an effect versus share of effect-free runs whose answer claims success.

## C. Model-Specific Behaviour of Tuned Instructions

A component of the automation direction classifies inbound messages into intents using natural-language intent descriptions held in a rule store. On a frozen 40-case set with five repeats of both arms, the same unedited descriptions scored 89.7% on gpt-4.1-nano with a one-case wobble, 94.3% on DeepSeek-V4-Flash with zero variance, and 97.1% on Claude Haiku 4.5 with zero variance. A description rewrite improved classification accuracy by 4.6 percentage points on the weakest deployment, reduced accuracy by 3.4 percentage points on the intermediate deployment, and produced no observed change on the strongest deployment (Table XLIX, Fig. 21). A confusion between two intents that had generated two description rewrites and a claimed specification gap simply did not occur on the other models.

TABLE XLIX. INTENT CLASSIFICATION ON THREE DEPLOYMENTS
<table><tr><td rowspan=1 colspan=1>Deployment</td><td rowspan=1 colspan=1>Accuracy,unedited</td><td rowspan=1 colspan=1>Variance (5repeats)</td><td rowspan=1 colspan=1>Effect of rewrite</td></tr><tr><td rowspan=1 colspan=1>gpt-4.1-nano</td><td rowspan=1 colspan=1>89.7%</td><td rowspan=1 colspan=1>one-case wobble</td><td rowspan=1 colspan=1>+4.6 pp,reportable</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Flash</td><td rowspan=1 colspan=1>94.3%</td><td rowspan=1 colspan=1>zero</td><td rowspan=1 colspan=1>–3.4 pp, no result</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>97.1%</td><td rowspan=1 colspan=1>zero</td><td rowspan=1 colspan=1>0.0 pp</td></tr></table>

Frozen 40-case set; labels self-authored by the intent author, so the system arm is an upper bound; the keyword control arm scores 0.000 and is a floor, not a rival.

The three deployment conditions and their classification counts are presented in Table XLIX; the same comparison is visualised in Fig. 21.

![](images/50a950f49a00f447e9157967b1fa77f84803e75f4bb9d4891d785edf4424c1c9.jpg)

Effect of the description rewrite, percentage points  
![](images/59a0d61a168ce154206cf47708bfbc4dd313a0c03a79570f58fa9657d895dee2.jpg)  
Fig. 21. Intent classification accuracy and the effect of a description rewrite on three deployments.

The transferable claim is that prompt and description engineering measured on a weak deployment measures the deployment, not the prompt. An edit whose benefit shrinks as the model gets better was a workaround for a model that could not follow the instruction already present, not an improvement of the instruction. The cheapest experiment in the whole exercise, changing the model before changing the words, was the one that had been run last. The same study measured a regex-based guardrail on 29 labelled drafts: precision 0.733, recall 0.688 and a false-block rate of 0.308. This is the one result in the study that does not depend on a model, since the guard uses regular expressions and substring matching, so it reproduces offline for anyone; its misses are evasions by surface (a lowercase IBAN) and its false blocks are shape collisions (an order reference shaped like a card number), and the adversarial set describes the shape of the failure modes, not their frequency. Programmable guardrails of this kind [67] are one of the planned attribute groups (P5).

## D. Chatbot: Controlled Experiments and the Evaluation Agent

The chatbot experiments used the mvp\_scenarios v6 dataset: most controlled comparisons used six scenarios, the final model comparison five scenarios with three repeats, and the agent evaluation all ten scenarios. A retrieval-only runner isolated retrieval from answer generation and tool routing. Table L and Fig. 22 summarise the controlled experiments on retrieval. Dense and hybrid retrieval [16], [17] reached the same recall, precision and MRR, but dense was faster (487 ms against 634 ms) and was selected. Sparse retrieval found almost none of the required sources; its higher precision does not indicate better overall quality. Retrieving more chunks did not improve recall or MRR and reduced precision (Top-K of 4 against 8 and 12), so K = 4 was selected. A reranker raised precision slightly (0.375 to 0.417) but left MRR unchanged and raised latency (the 2,960 ms includes model-loading cold start), so it was left off. With the retrieval fixed, temperature 0.0 gave the highest correctness (0.952) and task completion (0.963) for GPT-5 mini.

TABLE LI. CONTROLLED CHATBOT EXPERIMENTS ON RETRIEVAL AND GENERATION  
![](images/779e382f06bf5c2e62a44d9ed40e02a4bd4ad47dedab64474f8986552bf175f2.jpg)

![](images/1ebfa229f16379afd6f43c1f7913e3c95ba7355081984b4f1acfc745b4553ad2.jpg)  
Fig. 22. Retrieval strategy and Top-K experiments (six scenarios).

Nine models were then compared under the same configuration (dense, K = 4, reranker off): GPT-5.6 Sol, GPT-5.6 Terra, GPT-5 mini, Claude Sonnet 5, Claude Opus 5, Claude Haiku 4.5, Gemini 3.8 Flash, Gemini 3.1 Pro and

DeepSeek V4 Flash. Three finalists with the strongest balance of quality and latency were repeated: five scenarios times three repeats, 15 runs per model (Table LI, Fig. 23). Terra reached the highest mean correctness (0.995) and the lowest variability (standard deviation 0.013); Haiku was faster (9.8 s). The selected chatbot configuration was dense retrieval, Top-K = 4, reranker off and GPT-5.6 Terra.

TABLE L. REPEATED COMPARISON OF THE FINALISTS (15 RUNS PER MODEL)
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Passrate</td><td rowspan=1 colspan=1>Meancorrectness</td><td rowspan=1 colspan=1>Std</td><td rowspan=1 colspan=1>Responsetime</td></tr><tr><td rowspan=1 colspan=1>GPT-5.6 Terra</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>0.995</td><td rowspan=1 colspan=1>0.013</td><td rowspan=1 colspan=1>11.6 s</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>0.989</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>9.8 s</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>0.977</td><td rowspan=1 colspan=1>0.037</td><td rowspan=1 colspan=1>14.0 s</td></tr></table>

Pass rate applies to the five-scenario subset, not to the full agent campaign.

![](images/7134f0ee44691f958eab27bfb7369f838bd1d5485d286fe2bd0624bcbf495778.jpg)

![](images/0e7c073cc282cf34f6b1dfdd3d648998d61e795029739a0790c7b2095c102fe8.jpg)  
Fig. 23. Finalists: correctness with one standard deviation over 15 runs, and the quality–latency plane.

<table><tr><td rowspan=1 colspan=1>Experiment</td><td rowspan=1 colspan=1>Configuration</td><td rowspan=1 colspan=1>Result</td></tr><tr><td rowspan=1 colspan=1>Strategy (K = 4)</td><td rowspan=1 colspan=1>hybrid / dense / sparse</td><td rowspan=1 colspan=1>Recall 1.000 / 1.000 / 0.083; Precision 0.375 / 0.375 / 0.500; MRR 0.917 / 0.917 / 0.167; latency 634 /487 / 3.5 ms</td></tr><tr><td rowspan=1 colspan=1>Top-K (hybrid)</td><td rowspan=1 colspan=1>K = 4 / 8 / 12</td><td rowspan=1 colspan=1>Precision 0.375 / 0.304 / 0.289; Recall 1.000; MRR 0.917</td></tr><tr><td rowspan=1 colspan=1>Reranker</td><td rowspan=1 colspan=1>off / on (K = 4)</td><td rowspan=1 colspan=1>Precision 0.375 / 0.417; MRR 0.917; latency 762 / 2,960 ms</td></tr><tr><td rowspan=1 colspan=1>Temperature (GPT-5 mini)</td><td rowspan=1 colspan=1>0.0 / 0.3 / 0.7</td><td rowspan=1 colspan=1>Correctness 0.952 / 0.943 / 0.933; task completion 0.963 / 0.955 / 0.955; faithfulness 1.000 / 1.000 /0.987</td></tr></table>

Six scenarios per comparison; single run each; dense\_top4 fixed for the temperature experiment.

The observed mean correctness difference between Terra and Haiku is 0.006, smaller than their reported standard deviations (0.013 and 0.017). However, the individual repeats and their ranges are unavailable, so the criterion in Equation (8) cannot be evaluated exactly and a formal “tie” is not established. Table LII therefore gives only an illustrative two-objective Pareto comparison of the reported point estimates (correctness maximised and latency minimised). Model selection on these data is provisional: neither superiority in mean correctness nor superiority in stability has been established by a calibrated paired comparison.

The evaluation agent was first run as a diagnostic campaign (v1) on all ten scenarios with the selected configuration (Table LIII, Fig. 24). Five scenarios passed without unnecessary diagnostic calls. Four failed in the agent-and-tools layer because the required search tool call and its arguments were missing; one failed in grounding, because only one of two required source groups was cited, with answer quality also affected.

TABLE LII.  
PARETO ANALYSIS OF THE CHATBOT FINALISTS
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Correctness ↑</td><td rowspan=1 colspan=1>Time ↓</td><td rowspan=1 colspan=1>Dominated by</td><td rowspan=1 colspan=1>Tie oncorrectness</td></tr><tr><td rowspan=1 colspan=1>GPT-5.6 Terra</td><td rowspan=1 colspan=1>0.995</td><td rowspan=1 colspan=1>11.6 s</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Haiku</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>0.989</td><td rowspan=1 colspan=1>9.8 s</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Terra</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1>0.977</td><td rowspan=1 colspan=1>14.0 s</td><td rowspan=1 colspan=1>Terra, Haiku</td><td rowspan=1 colspan=1></td></tr></table>

Computed from the repeated finalist runs in Table LI. Because individual repeats are unavailable, the exact range-based rule of Equation (8) cannot be applied; the reported standard deviations indicate uncertainty but are not substituted for ranges.

In the first adaptive version the agent had no approved parameter change for the routing failures and tested only

Claude Haiku for the grounding failure, which stayed FAIL: one follow-up experiment and no accepted candidate. In v2 the available parameters were extended with the router model and several candidates per failing run were allowed (up to four). Of eight follow-up experiments four candidates were accepted on target-metric improvement; changing only the router to GPT-5.6 Sol improved the tool metrics in all four routing failures and turned two of them into passes, while the other two stayed FAIL because of tool arguments. The four candidates for the grounding failure were all rejected. Four accepted candidates do not mean four passing scenarios, only two additional runs changed to PASS, and nothing was applied automatically to the live chatbot.

TABLE LIII. EVALUATION-AGENT CAMPAIGNS ON TEN SCENARIOS
<table><tr><td rowspan=1 colspan=1>Campaign</td><td rowspan=1 colspan=1>Outcome</td></tr><tr><td rowspan=1 colspan=1>v1 diagnostic</td><td rowspan=1 colspan=1>5 PASS; 4 FAIL in agent/tools (search call and argumentsmissing); 1 FAIL in grounding → answer quality</td></tr><tr><td rowspan=1 colspan=1>Adaptive v1</td><td rowspan=1 colspan=1>1 follow-up (Haiku for the grounding failure) rejected; 0accepted</td></tr><tr><td rowspan=1 colspan=1>Adaptive v2 baseline</td><td rowspan=1 colspan=1>10 runs: 5 PASS, 5 FAIL</td></tr><tr><td rowspan=1 colspan=1>Adaptive v2 follow-up</td><td rowspan=1 colspan=1>8 experiments; 4 accepted (target metric +0.02 or PASS); 2FAIL → PASS</td></tr><tr><td rowspan=1 colspan=1>Router change</td><td rowspan=1 colspan=1>Sol as router: tool metrics improved in 4 of 4; 2 of 4became PASS</td></tr></table>

In the recorded v2 campaign, candidates were provisionally accepted if the target metric increased by at least 0.02 or the scenario became PASS. These single-scenario decisions were not confirmatory noise-aware tests and were not applied to the live system.

![](images/0b6f86221cea94224435491688c2fc8c173a7c3fffa14df110c446af104f1eb8.jpg)

![](images/46390e8ae0b200a4605ed956acab0995fd311213d0844c6972171b7f64a153b7.jpg)  
Fig. 24. Evaluation-agent campaigns: diagnostic outcome (v1) and adaptive experiments (v2).

## E. Case Study: Reaching a Chatbot Configuration

The chatbot experiments are a complete example of how the testbed workflow turns measurements into a decision, and they also show where that decision rests on weak evidence. The configuration was chosen in six steps, one variable at a time with the others fixed (Fig. 25, Table LIV). At each step we ask whether the evidence meets the noise-floor rule of Equation (8).

DECISIONS OF THE CHATBOT CONFIGURATION AND THE STRENGTH OF THEIR EVIDENCE
<table><tr><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>Candidates</td><td rowspan=1 colspan=1>Driving evidence</td><td rowspan=1 colspan=1>Strength</td></tr><tr><td rowspan=1 colspan=1>Retrieval strategy</td><td rowspan=1 colspan=1>dense, sparse, hybrid</td><td rowspan=1 colspan=1>sparse recall 0.083 vs 1.000; dense and hybrid tie,dense faster</td><td rowspan=1 colspan=1>sparse rejected clearly; dense vs hybrid bylatency</td></tr><tr><td rowspan=1 colspan=1>Top-K</td><td rowspan=1 colspan=1>4,8,12</td><td rowspan=1 colspan=1>precision falls 0.375 → 0.304 → 0.289; recall, MRRtie</td><td rowspan=1 colspan=1>monotone trend; single run</td></tr><tr><td rowspan=1 colspan=1>Reranker</td><td rowspan=1 colspan=1>off, on</td><td rowspan=1 colspan=1>precision +0.042; MRR tie; latency 762 → 2,960 ms</td><td rowspan=1 colspan=1>trade-off; cold start in latency</td></tr><tr><td rowspan=1 colspan=1>Temperature</td><td rowspan=1 colspan=1>0.0, 0.3, 0.7</td><td rowspan=1 colspan=1>correctness 0.952, 0.943, 0.933</td><td rowspan=1 colspan=1>differences 0.009–0.019; no spreadmeasured</td></tr><tr><td rowspan=1 colspan=1>Model screening</td><td rowspan=1 colspan=1>9 models</td><td rowspan=1 colspan=1>correctness, citations, latency; three finalists</td><td rowspan=1 colspan=1>screening, one run</td></tr><tr><td rowspan=1 colspan=1>Repeats</td><td rowspan=1 colspan=1>3 finalists, 15 runs each</td><td rowspan=1 colspan=1>correctness 0.995 ± 0.013, 0.989 ± 0.017, 0.977 ±0.037</td><td rowspan=1 colspan=1>Terra–Haiku tie on mean; variance andlatency decide</td></tr></table>

![](images/9308702ba3ddce8d9279ed66bc5d9b8bd30df0168c77d95cb7f304ca48d98b00.jpg)  
Fig. 25. Decision funnel of the chatbot configuration: one variable per step, with the evidence that drove the decision.  
Based on the chatbot experiments summarised in Tables L and LI.

Two observations follow. Clear-cut decisions exist: sparse retrieval is far worse on recall, and retrieving more chunks monotonically lowers precision without compensating gain. But several steps rest on single runs with differences of a few hundredths, and the final model choice is a variance-and-latency decision between models that tie on mean correctness. This does not invalidate the selection, which is a reasonable engineering choice, but it changes how the result should be reported: as a configuration selected by controlled experiments, with the tied alternatives named, rather than as the best configuration. The testbed’s role here is to keep every step reproducible, to store each step as a context-tagged measurement and to let a reader recompute the comparison with the noise rule.

The evaluation-agent campaigns add a second type of evidence: a diagnosis rather than a score. Of the ten scenarios, s001–s003 and s009–s010 passed; s004–s007 failed in the tool layer; and s008 failed in grounding with answer quality affected (Table LV). This split is the kind of structure that a single pass-rate would hide, and it is also what makes the router experiment interpretable, because it targets exactly the four scenarios that the diagnosis attributed to routing.

Breadth. The first experiment asked whether automatic strategy selection beats a single method (Table LVI, Fig. 26). The agent accepted 1,330 of 1,448 tasks (91.9%), the cheaper model on every task 1,310 (90.5%) and rules alone

205 (14.2%). An accepted task passed every check available for it; most accepted tasks nevertheless stop at static checks, because the fragments do not contain the tables they reference, which is why only 284 of the agent’s tasks were installed or executed.

SCENARIO-LEVEL OUTCOMES OF THE DIAGNOSTICCAMPAIGN
<table><tr><td rowspan=1 colspan=1>Scenarios</td><td rowspan=1 colspan=1>Result</td><td rowspan=1 colspan=1>Diagnosis</td></tr><tr><td rowspan=1 colspan=1>s001-s003,s009-s010</td><td rowspan=1 colspan=1>5 PASS</td><td rowspan=1 colspan=1>none; no unnecessary diagnostic calls</td></tr><tr><td rowspan=1 colspan=1>s004-s007</td><td rowspan=1 colspan=1>4 FAIL</td><td rowspan=1 colspan=1>agent/tools: required search call andarguments missing</td></tr><tr><td rowspan=1 colspan=1>s008</td><td rowspan=1 colspan=1>1 FAIL</td><td rowspan=1 colspan=1>grounding, then answer quality: oneof two source groups cited</td></tr><tr><td rowspan=1 colspan=1>s004, s005 (router →Sol)</td><td rowspan=1 colspan=1>FAIL</td><td rowspan=1 colspan=1>tool metrics improved; tool argumentsstill wrong</td></tr><tr><td rowspan=1 colspan=1>s006, s007 (router →Sol)</td><td rowspan=1 colspan=1>PASS</td><td rowspan=1 colspan=1>tool metrics improved; scenariopasses</td></tr></table>

Single campaign on the selected configuration; nothing applied to the live chatbot.

## F. Migration: Breadth, Depth and the Fix Ledger

The agent produced 125 exact structural matches with the human reference against 86 for the cheaper model alone (+45.3%), and 426 pairs with similarity at least 0.9 against 379. Rules alone solve a small simple slice accurately (112 exact matches, but only 174 comparable pairs); the agent covers the rest. The token saving relative to the cheaper-model baseline is small, 3.0% fewer tokens and 6.9% fewer calls; repairs and the strong model spend most of what the static path saves. The strong model handled 31 fragments, used 987,364 tokens and produced no exact matches, which is not evidence about the model, since those were the hardest fragments, but it is not evidence that the tier paid for itself. Repair and escalation touched 465 fragments, used 64.4% of the tokens and produced 27 exact matches, while first-attempt fragments used 3.07 million tokens and produced 97. Rules had solved 112 pairs exactly; the agent kept the rules path on 72 of them and, on the other 40, called a model and kept the exact match in only 9, losing 31 matches at a cost of 278,215 tokens.

TABLE LVI. MIGRATION BREADTH: STRATEGY SELECTION ON 1,448 TASKS
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>Cheap LLM</td><td rowspan=1 colspan=1>Rules only</td></tr><tr><td rowspan=1 colspan=1>Accepted tasks</td><td rowspan=1 colspan=1>1,330(91.9%)</td><td rowspan=1 colspan=1>1,310(90.5%)</td><td rowspan=1 colspan=1>205 (14.2%)</td></tr><tr><td rowspan=1 colspan=1>Exact structural matches(of 1,000)</td><td rowspan=1 colspan=1>125</td><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=1>112</td></tr><tr><td rowspan=1 colspan=1>Similarity ≥ 0.9 (of 1,000)</td><td rowspan=1 colspan=1>426</td><td rowspan=1 colspan=1>379</td><td rowspan=1 colspan=1>142</td></tr><tr><td rowspan=1 colspan=1>Installed or executed</td><td rowspan=1 colspan=1>284</td><td rowspan=1 colspan=1>275</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>Tokens</td><td rowspan=1 colspan=1>10,020,728</td><td rowspan=1 colspan=1>10,332,904</td><td rowspan=1 colspan=1></td></tr></table>

One run per configuration on the same sample (seed 42). Structural match: same normalised syntax tree as the human reference; not proof of equal behaviour.

A second experiment sent each file whole to the same cheaper model with the same prompts and repair loop, without splitting. It produced 99 exact matches against 86 for the split run, a gap that is not distinguishable (McNemar p = 0.208). Typical similarity went the other way, 0.674 with the split against 0.520 without, with the drop concentrated in 240 unnamed PL/SQL blocks (0.741 versus 0.179); the whole-file run used 1.37 times more tokens and, in 97 files, used an object before creating it. The split, the dependency order and the per-object repair were removed together, so the run does not isolate splitting by itself.

![](images/6ceeb0023fab482a5dd61dec2435c3fa018edfddb305dd669fe868b65fe8923b.jpg)

Exact structural matches (of 1,000 pairs)  
![](images/004e17a83a97f2cebfd8e50c3c6c19ecddb5f13dd38fa21fb114cf878446996b.jpg)  
Installed or executed in PostgreSQL

![](images/f6156af999a78c704e104c9b2cf6080ee4b03605a3be6cedd16f4779bb40877c.jpg)  
Fig. 26. Migration breadth: accepted tasks, exact matches and executed tasks for the three strategies.

Component-level evidence: decomposition. The decomposition stage of the migration agent was itself evaluated before the agent was assembled. Three ways of splitting an Oracle artefact into units were compared on a sample artefact and a 116-file corpus: a static PL/SQL-aware scanner, a grammar-based parser and an LLM splitter (Table LVII). A round trip (split, reassemble, compare with the original) is a cheap deterministic check that a split is information-preserving before any agent is involved. The scanner is the robust default; the grammar-based parser is the most precise but cleanly parses only about 59% of messy real files; the LLM is lossless at corpus scale but limited by token quota and throughput, with boundaries that need structural verification. Splitting a single file is only half the problem, because real dependencies are global (cross-file calls, package specification and body pairs). Lifting the units into one batch-level dependency graph, resolving cross-file references into edges, and using strongly connected components and condensation to handle real cycles gives a dependency-respecting order and parallel phases.

On the 116-file corpus the parser-only run found 1,271 dependencies and 446 internal links, but left 165 units that the grammar could not parse with no dependencies at all. An LLM fallback for exactly those units reduced the number of units without dependencies from 165 to 0, found 496 additional dependencies and raised internal links from 446 to 527 and external links from 825 to 1,182, at the price of about 37 minutes of sequential model calls (Table LVIII, Fig. 27).

TABLE LVII. COMPARISON OF SPLITTING STRATEGIES FOR ORACLE CODE
<table><tr><td rowspan=1 colspan=1>Splitter</td><td rowspan=1 colspan=1>Coverage and fidelity</td><td rowspan=1 colspan=1>Limitation</td></tr><tr><td rowspan=1 colspan=1>static PL/SQL-awarescanner</td><td rowspan=1 colspan=1>100% coverage, lossless, nodependencies</td><td rowspan=1 colspan=1>no semantic structure</td></tr><tr><td rowspan=1 colspan=1>grammar-based parser(ANTLR)</td><td rowspan=1 colspan=1>most precise, byte-exact</td><td rowspan=1 colspan=1>cleanly parses about 59%of messy real files</td></tr><tr><td rowspan=1 colspan=1>LLM splitter</td><td rowspan=1 colspan=1>lossless at corpus scale (86 of86 processed files)</td><td rowspan=1 colspan=1>token quota, throughput;boundaries needverification</td></tr><tr><td rowspan=1 colspan=1>sqlglot</td><td rowspan=1 colspan=1>plain-SQL cross-check only</td><td rowspan=1 colspan=1>not PL/SQL</td></tr></table>

Sample artefact and 116-file Oracle corpus.

TABLE LVIII. DEPENDENCY EXTRACTION ON THE 116-FILE CORPUS
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Parser only</td><td rowspan=1 colspan=1>Parser + LLM fallback</td></tr><tr><td rowspan=1 colspan=1>Dependencies found</td><td rowspan=1 colspan=1>1,271</td><td rowspan=1 colspan=1>1,271 + 496</td></tr><tr><td rowspan=1 colspan=1>Internal links</td><td rowspan=1 colspan=1>446</td><td rowspan=1 colspan=1>527</td></tr><tr><td rowspan=1 colspan=1>External links</td><td rowspan=1 colspan=1>825</td><td rowspan=1 colspan=1>1,182</td></tr><tr><td rowspan=1 colspan=1>Units withoutdependencies</td><td rowspan=1 colspan=1>165</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Added time</td><td rowspan=1 colspan=1>_</td><td rowspan=1 colspan=1>about 37 min (sequential)</td></tr></table>

The corpus contains no cycles; cycle handling was tested on a small dedicated example.

![](images/1b501429a2dbc7c3206c8006bc7f2c81fe6060d42d656be39215953bd9ab7911.jpg)  
Fig. 27. Effect of the LLM fallback on dependency extraction (116-file corpus).

Depth. On 216 migration units executed against a real Oracle instance, two full runs using production routing were compared: migrate/v34 and migrate/v38, with three additional corrections to the evaluation environment (Table LIX, Fig. 28). The agent’s own count of ACCEPTED rose by 21, from 93 to 114. Checked against Oracle, the number of demonstrably correct units, defined as ACCEPTED and not contradicted by Oracle’s output or errors plus FAILED units that raise exactly what Oracle raises, rose by 10, from 98 to 108. Nine of the 93 earlier and 25 of the 114 later ACCEPTED units are wrong or cannot be verified; most of the extra ones had been BLOCKED before and ran for the first time. Real translation defects rose from 43 to 56 because 32 units that used to be BLOCKED are now tried, and some of them are wrong, which is information that was missing before and not a regression. Examples of wrong answers that the agent had accepted: a collection comparison that says “not equal” where Oracle says “equal”; an error handler that prints NULL where Oracle prints −6502-Value=1000; a bulk update that also touches a row Oracle never updates; a caller that invokes the wrong procedure and returns quietly where Oracle raises.

TABLE LIX. MIGRATION DEPTH: 216 UNITS CHECKED AGAINST REAL ORACLE
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Before</td><td rowspan=1 colspan=1>After</td></tr><tr><td rowspan=1 colspan=1>ACCEPTED (agent&#x27;s own report)</td><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1>114</td></tr><tr><td rowspan=1 colspan=1>Demonstrably correct</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>108</td></tr><tr><td rowspan=1 colspan=1>Real translation defects</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>56</td></tr><tr><td rowspan=1 colspan=1>BLOCKED (never attempted)</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>Not testable by the stand as built</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>23</td></tr><tr><td rowspan=1 colspan=1>Other</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>20</td></tr></table>

Runs 20260923-084200-b72cd4ff and 20260923-121730-92079ad3. The ACCEPTED row is a separate, overlapping agent-status indicator. The remaining mutually exclusive evaluation categories sum to 216 units for each run.

What FAILED contained. Because the agent’s own FAILED status cannot distinguish a wrong translation from a correct one that raises the same error as Oracle, every FAILED unit of the production run was read by hand and, where the judgement rested on Oracle’s behaviour, executed on the real Oracle instance (Table LX). Of 61 FAILED units, 27 were real translation defects, 14 were faithful raises, 8 were invalid test inputs that Oracle rejects too, 8 were limitations of the harness and 4 were explained no-ops where the model returned no SQL and said why, for declarations with no PostgreSQL object to create. The distribution of before-and-after unit states is shown in Fig. 28. The migration depth-study status distribution is given in Table LIX.

![](images/f3effa12df93ad89231b9a7ddc684214f0f2d70529011a2110ce1a390225927a.jpg)  
Fig. 28. Depth study: categories before and after the fixes of 2026-09-23.

A first pass that counted every static-stage failure as a defect without reading it had overstated the defect count (32 instead of 27); five units that started as uncertain were all resolved by execution the same day. A similar reading of the BLOCKED units found that about 32 of 41 descended from upstream units that had installed cleanly and failed only at scenario execution; eleven of those upstreams were faithful raises, so that a correct procedure raising NO\_DATA\_FOUND by design was blocking five other units on its own. The expected consequence was stated in advance: BLOCKED falls and FAILED rises, which read naively looks worse and is not.

TABLE LX. RECLASSIFICATION OF THE 61 FAILED UNITS(PRODUCTION ROUTING)
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Units</td><td rowspan=1 colspan=1>Meaning</td></tr><tr><td rowspan=1 colspan=1>Real translation defects</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>wrong behaviour against Oracle</td></tr><tr><td rowspan=1 colspan=1>Faithful raises</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>raises exactly where Oracle raises, e.g.a numeric overflow that Oracle alsoraises</td></tr><tr><td rowspan=1 colspan=1>Invalid test inputs</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>inputs that Oracle rejects too (e.g. aninjection string as a column name)</td></tr><tr><td rowspan=1 colspan=1>Harness limitations</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>COMMIT/ROLLBACK blocks, hiddendynamic-SQL dependency, cursorargument, non-deterministic cursorname</td></tr><tr><td rowspan=1 colspan=1>Explained no-ops</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>no SQL returned; declaration has noPostgreSQL object</td></tr></table>

Method: read the error, the triggering scenario and the Oracle source; execute on real Oracle where the judgement depends on its behaviour.

Two further observations from this reading bear on the testbed. First, a silent-wrong case was caught only because a second model, tried for cost reasons, disagreed with the baseline on one unit and reading both answers showed that the baseline was the wrong one; cross-model disagreement is therefore a cheap defect detector that the testbed gives for free whenever two models are run on the same application and context. Second, a correct fix can introduce wrong acceptances: a fix teaching the model that a package-level variable is mutable state moved eleven consumers from 5 ACCEPTED and 6 FAILED to 10 and 1, yet executing the whole file in order on both databases showed that Oracle’s collection DELETE leaves a gap which the translation closed up, so that five of the new ACCEPTED units were wrong while every block ran without error. The fix stays partial and the count of demonstrably correct units does not include them. Representative fault classes and associated repairs are itemised in Table LXI.

Model choice and cost also had surprises. A cheaper model on the proxy, about four times cheaper per token, could not complete one full run, because its daily token cap was 2 M against 20 M for the default; on a shared key, quota decides before price. Cost reporting itself was wrong until 2026-09-22, because prices came from a public rate card times a guessed factor; they now come from the proxy’s own rates, and two runs in a row matched the proxy’s bill to the cent. A full production-routed run costs about USD 3 to 4. The limit of the current numbers is mostly what the stand can verify rather than what the model can translate: about 30 to 40 units are untested for reasons in the stand, and the estimate for the next run with the fixes already committed is 115 to 120 demonstrably correct units, with a ceiling of about 180 that requires fixing statement boundaries in the decomposer and making the test-input generator produce inputs that Oracle accepts. The disposition of confirmed, partial and failed fixes is also summarised in Fig. 29.

TABLE LXI. EXAMPLES OF FIXES FROM THE LEDGER
<table><tr><td rowspan=1 colspan=1>Fix</td><td rowspan=1 colspan=1>Oracle behaviour taught to the agent</td></tr><tr><td rowspan=1 colspan=1>FIX-037</td><td rowspan=1 colspan=1>single-row SELECT ... INTO raises on 0 or 2+ rows;PostgreSQL needs INTO STRICT</td></tr><tr><td rowspan=1 colspan=1>FIX-041</td><td rowspan=1 colspan=1>DATE carries a time of day: map to TIMESTAMP(0), notDATE</td></tr><tr><td rowspan=1 colspan=1>FIX-048</td><td rowspan=1 colspan=1>∥ treats NULL as an empty string; PostgreSQL&#x27;s makes thewhole string NULL</td></tr><tr><td rowspan=1 colspan=1>FIX-051</td><td rowspan=1 colspan=1>= between collections compares multisets with their ownNULL rules</td></tr><tr><td rowspan=1 colspan=1>FIX-052</td><td rowspan=1 colspan=1>FORALL ... VALUES OF and the total row count acrossiterations</td></tr><tr><td rowspan=1 colspan=1>FIX-054</td><td rowspan=1 colspan=1>repairs a regression: TOO_MANY_ROWS translated asDUP VAL ON INDEX</td></tr><tr><td rowspan=1 colspan=1>FIX-056</td><td rowspan=1 colspan=1>dynamic-SQL :name binds become $1, $2 ... by position</td></tr></table>

Migration repair outcomes and statuses from the executed fix cycles.

![](images/0204468941db967bdec8094baee2fd2639a5331f5216a0afd3d505cb5b185cff.jpg)  
Fig. 29. Verdicts of the 46 fix cycles.

G. Knowledge Retrievalfor Migration: Measurement Pitfalls

The migration agent can retrieve conversion guidance from a vector knowledge base, and the attribute groups of the scheme include “knowledge and retrieval”. A series of retrieval experiments on this component, run by colleagues of the programme, is worth reporting briefly because every one of its corrections is an instance of a principle of Section VII. The first proof of concept compared two databases for vector retrieval and found identical results. On re-examination the experiment never exercised a vector index: at 31 chunks both databases fall back to an exact scan, so identical embeddings with identical cosine distance and a full scan can only produce identical rankings, and the finding was arithmetically guaranteed and not measured. The reported latency difference (0.0109 s against 0.0052 s) measured driver overhead. A comparison that cannot come out differently is not a measurement.

Repeated at a realistic scale, the experiment showed that at matched recall the two systems were equivalent in retrieval quality and query latency (about 1.75 ms median at an approximate-search recall near 0.99), while the open-source extension built its indexes 8 to 226 times faster (Table LXII).

TABLE LXII. VECTOR INDEX BENCHMARK: TWO SYSTEMS AT MATCHED RECALL
<table><tr><td rowspan=1 colspan=1>Measure</td><td rowspan=1 colspan=1>Oracle</td><td rowspan=1 colspan=1>pgvector</td></tr><tr><td rowspan=1 colspan=1>HNSW build time, recall ≈ 0.99</td><td rowspan=1 colspan=1>40.1 s</td><td rowspan=1 colspan=1>4.9s</td></tr><tr><td rowspan=1 colspan=1>IVF build time, recall ≈ 0.955</td><td rowspan=1 colspan=1>249.0 s</td><td rowspan=1 colspan=1>1.1 s</td></tr><tr><td rowspan=1 colspan=1>Query latency p50 (HNSW)</td><td rowspan=1 colspan=1>≈ 1.75 ms</td><td rowspan=1 colspan=1>≈ 1.75 ms</td></tr><tr><td rowspan=1 colspan=1>Default search settings, IVFrecall</td><td rowspan=1 colspan=1>0.9574 (declarativetarget)</td><td rowspan=1 colspan=1>0.5975 (probes = 1)</td></tr><tr><td rowspan=1 colspan=1>Exact search recall@10 /nDCG@10</td><td rowspan=1 colspan=1>0.4575 / 0.3882</td><td rowspan=1 colspan=1>0.4575 / 0.3882</td></tr><tr><td rowspan=1 colspan=1>Exact scan p50</td><td rowspan=1 colspan=1>61.6 ms</td><td rowspan=1 colspan=1>30.5 ms</td></tr></table>

Oracle AI Vector Search compared with pgvector under matched approximate-recall configurations.

An earlier untuned pass had falsely favoured one system, because the other’s defaults silently discarded about 40% of what an exact search would have found (approximate recall 0.5975 at the default probe setting); raising two search parameters closed the gap entirely, so the difference had been the tuning and not the databases. This is a second form of the context rule: a configuration is part of the measurement, and default configurations are not neutral. A third experiment found that fusing a lexical signal into dense retrieval by reciprocal rank fusion [17] made nDCG@10 worse on natural-language queries on both databases (0.3028, 0.3101 and 0.3191 against 0.3882 for dense search alone), and that a native fused index failed to finish building at 86 thousand chunks after more than 90 minutes at full CPU, against 5 seconds for the alternative, because the fused index cannot consume precomputed embeddings. In the same programme a different corpus with identifier-bearing questions favoured hybrid retrieval, and a measured noise floor of the judge (Section VII-F) showed that an earlier reported gap on six cases lay inside the noise. The chatbot, which found dense and hybrid retrieval tied on its own corpus, is consistent with the view that the value of hybrid retrieval depends on the corpus and the queries and must be measured for each application, which is the purpose of the testbed. The measured dense and fused retrieval results appear in Table LXIII.

TABLE LXIII. DENSE VERSUS FUSED RETRIEVAL (NDCG@10)
<table><tr><td rowspan=1 colspan=1>Arm</td><td rowspan=1 colspan=1>nDCG@10</td></tr><tr><td rowspan=1 colspan=1>dense vector search alone</td><td rowspan=1 colspan=1>0.3882</td></tr><tr><td rowspan=1 colspan=1>Oracle Text + RRF</td><td rowspan=1 colspan=1>0.3028</td></tr><tr><td rowspan=1 colspan=1>BM25 + RRF (PostgreSQL)</td><td rowspan=1 colspan=1>0.3101</td></tr><tr><td rowspan=1 colspan=1>tsvector + RRF (PostgreSQL)</td><td rowspan=1 colspan=1>0.3191</td></tr></table>

Natural-language query set; native fused index did not finish building.

## H. Related Measurements from Earlier Studies of the Group

Three earlier studies of the group are relevant for how the testbed treats cost, specifications and judges. Token optimisation. The first concerns cost [63]. Twelve token-optimisation strategies for LLM-based Oracle-to-PostgreSQL migration were evaluated on samples of 10 and 100 Oracle SQL queries with Valid Syntax Rate, Exact Match, Semantic Match, CodeBLEU and Token Efficiency. Mild context pruning preserved semantic quality almost at the baseline level (89.75% against 89.80% Semantic Match on 100 queries). Adaptive routing gave the best practical trade-off, reducing input tokens by 8.72% and output tokens by 5.49% while maintaining 88.40% Semantic Match and raising Token Efficiency by 6.67%. Aggressive schema distillation raised Token Efficiency by 132.22% but lowered Semantic Match by 44.50 percentage points (Fig. 30). Token optimisation is therefore not prompt shortening but a multi-objective problem balancing cost, validity and semantic preservation, which is why the metamodel stores cost and quality attributes side by side and why the analytical interface offers a Pareto view instead of a single score.

![](images/7e7fcea619fb5d83b7bb5a6842e2e791bcb3158645a3be65f57eb3ca55ad6e5c.jpg)  
Fig. 30. Token optimisation: semantic match and token efficiency change for four of the twelve strategies on 100 queries (values from [63]; distillation bar derived as 89.80 − 44.50).

What a specification determines. The second study introduces a formal semantic-block model of a specification and an execution-judged benchmark with a fixed implementer panel, a mandatory no-specification control arm, and PostgreSQL 16 and a live Oracle instance as deterministic judges [46]. On 75 execution-verified units the full specification raised clean-loading output from 72.0% to 97.3%, whereas aggregate cross-implementer agreement changed only from 83.1% to 83.8%, and structured row-level comparison against live Oracle yielded 19 correct results out of 42 probes in both conditions. The five-layer decomposition reduced the mean per-task context by about

71% through dependency closures. Repeated runs on a 25-unit subsample revealed a median arm-delta spread of 14.4 percentage points. The study therefore supports determinacy as a formal concept but not as a standalone empirical quality metric, and it is the empirical origin of the noise-floor and control-arm rules of Section VII-F.

Evaluating the evaluators. The third study builds a leakage-resistant benchmark of 494 labelled source–specification pairs over 19 quality metrics and compares three LLM judges with two trivial baselines on 247 held-out records [45]. The judges reached balanced accuracies of 0.87 (GPT-4o), 0.89 (DeepSeek V4 Pro) and 0.81 (Claude Sonnet 4.5), while a specification-length baseline reached 0.88, and exact paired tests did not separate any judge from that baseline. The judges nevertheless carried information that length does not: the labelled defect category appeared among their findings in 84% to 92% of violated records and the quoted evidence overlapped the labelled span in 92% to 93%. This is the strongest argument we have for the source hierarchy of Table XXXIX: a judge adds typed, evidence-linked findings that help triage, but its binary verdict is not distinguishable from a trivial baseline, so it may explain but not decide. Table LXIV collects the key figures.

TABLE LXIV. KEY FIGURES FROM EARLIER STUDIES OF THE GROUP
<table><tr><td rowspan=1 colspan=1>Study</td><td rowspan=1 colspan=1>Figure</td></tr><tr><td rowspan=1 colspan=1>Token optimisation[[63]</td><td rowspan=1 colspan=1>mild pruning 89.75% vs 89.80% semantic match;adaptive routing -8.72% input tokens at 88.40%;distillation +132.22% efficiency, -44.50 pp</td></tr><tr><td rowspan=1 colspan=1>Specificationbenchmark [46]</td><td rowspan=1 colspan=1>clean-loading 72.0% → 97.3%; agreement 83.1%→ 83.8%; 19/42 live-Oracle probes correct in botharms; context -71%</td></tr><tr><td rowspan=1 colspan=1>Variability floor [46]</td><td rowspan=1 colspan=1>median arm-delta spread 14.4 pp on a 25-unitsubsample</td></tr><tr><td rowspan=1 colspan=1>Judge benchmark[45]</td><td rowspan=1 colspan=1>balanced accuracy 0.87 / 0.89 / 0.81 vs 0.88 forlength baseline; defect category in 84–92% offindings</td></tr></table>

## I. Simulation Study of the Noise Rule

The noise rule of Section VII-F was derived from failures, not from theory, so its operating characteristics had not been studied. We did so by Monte Carlo simulation, the technique used in the authors’ earlier work on the evaluation of attribute values under uncertainty [88]. Two arms are simulated with independent normal repeats of equal standard deviation σ per run, a true difference δ, and r repeats per arm (40,000 trials per cell, seeded). Three decision rules are compared: the range rule of Equation (8), a Welch test at level 0.05, and a fixed global threshold of two reference deviations applied to every axis. Table LXV reports the minimum detectable differences at 80% statistical power and the probability of sign disagreement between independent replications. Table LXVI presents the proportion of Monte Carlo trials in which an effect is declared under different decision rules, while Fig. 31 illustrates their operating characteristics for five repetitions per experimental arm.

Four findings follow. First, the range rule is a very conservative screen: its false-positive rate is 1.2% with three repeats and below 0.01% with five or more. Second, its sensitivity falls as repeats are added, from a minimal detectable difference of 4.65 σ with three repeats to 7.17 σ with ten, because the range of a sample grows with its size while the standard error shrinks; the Welch test goes the other way (3.51, 2.09 and 1.32 σ).

![](images/1732e844bc7303a9954071cc97ae99b07033ac0bd9fb61fad95ba3816413a8f4.jpg)

Minimal detectable difference (units of SD)  
![](images/526b0c4891b30585873b131be6801d1e29fc88288cf45af86d7e05c37269a811.jpg)  
Fig. 31. Operating characteristics of the noise rule (simulation): decision shares against the true difference for five repeats (top), and the minimal difference detected with 80% power against the number of repeats (bottom).

<table><tr><td rowspan=1 colspan=1>Quantity</td><td rowspan=1 colspan=1>r=3</td><td rowspan=1 colspan=1>r=5</td><td rowspan=1 colspan=1>r = 10</td></tr><tr><td rowspan=1 colspan=1>difference with 80% power, range rule (σ)</td><td rowspan=1 colspan=1>4.65</td><td rowspan=1 colspan=1>5.78</td><td rowspan=1 colspan=1>7.17</td></tr><tr><td rowspan=1 colspan=1>difference with 80% power, Welch test (σ)</td><td rowspan=1 colspan=1>3.51</td><td rowspan=1 colspan=1>2.09</td><td rowspan=1 colspan=1>1.32</td></tr><tr><td rowspan=1 colspan=1>sign disagreement of two replications at 0.5 σ</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>33.7%</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>sign disagreement at 1 σ</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10.6%</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>sign disagreement at 2 σ</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.16%</td><td rowspan=1 colspan=1></td></tr></table>

Sign disagreement: probability that two independent replications of the same experiment give differences of opposite sign.

This is why the verdict of Equation (9) has a second tier. Third, a fixed global threshold is not neutral across axes (Table LXXI): with a threshold of two reference deviations, an axis whose own deviation is twice the reference declares a spurious effect in 11.5% of trials, whereas an axis with half the reference deviation never does, and an effect exactly equal to the threshold is detected in half of the trials on every axis. Fourth, small observed differences cannot be trusted in sign: for a true difference of 0.5 σ, two independent replications disagree on the sign in a third of cases, and at 1 σ in one case in ten, which is consistent with the reports of an axis that changed sign between identical re-runs (+23.3, then −9.3 percentage points). The global-threshold comparison is quantified in Table LXVII.

TABLE LXVI. SHARE OF TRIALS DECLARING AN EFFECT (MONTE CARLO, 40,000 TRIALS PER CELL)
<table><tr><td rowspan=1 colspan=1>True difference (units of σ)</td><td rowspan=1 colspan=1>Range, r = 3</td><td rowspan=1 colspan=1>Range, r = 5</td><td rowspan=1 colspan=1>Range, r = 10</td><td rowspan=1 colspan=1> $\mathbf { W e l c h } , \mathbf { r } = 3$ </td><td rowspan=1 colspan=1> $\mathbf { W e l c h , r = 5 }$ </td><td rowspan=1 colspan=1>Welch, r = 10</td></tr><tr><td rowspan=1 colspan=1>0 (no effect)</td><td rowspan=1 colspan=1>0.012</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.044</td><td rowspan=1 colspan=1>0.047</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.049</td><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.121</td><td rowspan=1 colspan=1>0.265</td><td rowspan=1 colspan=1>0.557</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.182</td><td rowspan=1 colspan=1>0.021</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.366</td><td rowspan=1 colspan=1>0.771</td><td rowspan=1 colspan=1>0.989</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.415</td><td rowspan=1 colspan=1>0.110</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1>0.677</td><td rowspan=1 colspan=1>0.981</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.673</td><td rowspan=1 colspan=1>0.330</td><td rowspan=1 colspan=1>0.030</td><td rowspan=1 colspan=1>0.884</td><td rowspan=1 colspan=1>0.999</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.953</td><td rowspan=1 colspan=1>0.837</td><td rowspan=1 colspan=1>0.469</td><td rowspan=1 colspan=1>0.994</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.997</td><td rowspan=1 colspan=1>0.989</td><td rowspan=1 colspan=1>0.929</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td></tr></table>

Row 0 is the false-positive rate. Normal i.i.d. repeats; the range rule is Equation (8) with τ = 0, using the means of the repeats.

TABLE LXVII.  
FIXED GLOBAL THRESHOLD VERSUS PER-AXIS RULES (FIVE REPEATS, THRESHOLD $2 { \scriptstyle \Sigma _ { \scriptscriptstyle \mathrm { R E F } } } )$
<table><tr><td rowspan=1 colspan=1>Axis deviation $( \sigma _ { \mathrm { r e f } } )$ </td><td rowspan=1 colspan=1>True diff.</td><td rowspan=1 colspan=1>Globalthreshold</td><td rowspan=1 colspan=1>Range rule</td><td rowspan=1 colspan=1>Welch test</td></tr><tr><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.043</td></tr><tr><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.498</td><td rowspan=1 colspan=1>0.330</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.045</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.502</td><td rowspan=1 colspan=1>0.020</td><td rowspan=1 colspan=1>0.768</td></tr><tr><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.115</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.043</td></tr><tr><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1>0.264</td></tr></table>

Share of trials declaring an effect; $\sigma _ { \mathrm { r e f } }$ is the reference per-run deviation.

The limits of the simulation are those of its assumptions: normal, independent and identically distributed repeats, equal deviations in both arms, and a single axis. Real axes are bounded rates, repeats of an LLM run may be correlated through shared prompts and caches, and deviations differ between arms. The numbers therefore characterise the rules, not the stands, and they argue for a rule with two tiers and for planning five or more repeats whenever a claim of improvement is intended.

## J. Statistical Re-Analysis ofMeasured Counts

Some comparisons consist of counts without confidence intervals, and several experimental conditions have only one run. We therefore computed Wilson intervals for binomial proportions, tested available paired discordances when recorded, and otherwise report descriptive approximations without implying that an unpaired test validates a paired design. The migration count intervals and test choices are summarised in Table LXVIII. The re-analysis provides descriptive, not paired-inferential, comparisons.

TABLE LXVIII. WILSON 95% INTERVALS AND UNPAIRED FISHER TESTS FOR MIGRATION COUNTS
<table><tr><td rowspan=1 colspan=1>Quantity</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Proportion</td><td rowspan=1 colspan=1>95% interval</td><td rowspan=1 colspan=1>Comparison, Fisher p</td></tr><tr><td rowspan=1 colspan=1>demonstrably correct, before fixes</td><td rowspan=1 colspan=1>98 / 216</td><td rowspan=1 colspan=1>45.4%</td><td rowspan=1 colspan=1>38.9–52.0%</td><td rowspan=1 colspan=1>before vs after: 0.39</td></tr><tr><td rowspan=1 colspan=1>demonstrably correct, after fixes</td><td rowspan=1 colspan=1>108 / 216</td><td rowspan=1 colspan=1>50.0%</td><td rowspan=1 colspan=1>43.4–56.6%</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>accepted but wrong or unverifiable, before</td><td rowspan=1 colspan=1>9 / 93</td><td rowspan=1 colspan=1>9.7%</td><td rowspan=1 colspan=1>5.2-17.4%</td><td rowspan=1 colspan=1>before vs after: 0.023</td></tr><tr><td rowspan=1 colspan=1>accepted but wrong or unverifiable, after</td><td rowspan=1 colspan=1>25 / 114</td><td rowspan=1 colspan=1>21.9%</td><td rowspan=1 colspan=1>15.3-30.4%</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>exact structural matches, agent</td><td rowspan=1 colspan=1>125 / 1000</td><td rowspan=1 colspan=1>12.5%</td><td rowspan=1 colspan=1>10.6-14.7%</td><td rowspan=1 colspan=1>agent vs cheap: 0.0056</td></tr><tr><td rowspan=1 colspan=1>exact structural matches, cheap model</td><td rowspan=1 colspan=1>86 / 1000</td><td rowspan=1 colspan=1>8.6%</td><td rowspan=1 colspan=1>7.0-10.5%</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>accepted tasks, agent</td><td rowspan=1 colspan=1>1330 / 1448</td><td rowspan=1 colspan=1>91.9%</td><td rowspan=1 colspan=1>90.3-93.2%</td><td rowspan=1 colspan=1>agent vs cheap: 0.21</td></tr><tr><td rowspan=1 colspan=1>accepted tasks, cheap model</td><td rowspan=1 colspan=1>1310 / 1448</td><td rowspan=1 colspan=1>90.5%</td><td rowspan=1 colspan=1>88.9–91.9%</td><td rowspan=1 colspan=1></td></tr></table>

Unpaired tests ignore the matched structure of the two runs and are descriptive approximations; conservativeness is not guaranteed. A McNemar test would require unit-level discordant pairs, which are not reported except for whole-file versus split migration (p = 0.208).

The operational count increased from 98 to 108 of the same 216 migration units; the unpaired Fisher calculation (p = 0.39) does not establish either significance or equivalence for these paired data. The share of accepted units subsequently marked wrong or unverifiable increased from 9.7% to 21.9% (unpaired $\mathsf { p } ^ { - } = 0 . 0 2 3 )$ , demonstrating a larger observed discrepancy between agent acceptance and external checking, but a paired significance claim requires the cross-classified unit records. Likewise, 125 versus 86 exact structural matches (unpaired $\mathfrak { p } = 0 . 0 0 5 6 )$ should not be promoted to a paired improvement without the discordant counts. The data support a stronger measurement taxonomy; precise before–after inference awaits the original paired outcomes.

TABLE LXIX. HETEROGENEITY ACROSS FRAMEWORKS AND ACROSS MODELS (AUTOMATION, 49 RUNS PER ARM)
<table><tr><td rowspan=1 colspan=1>Quantity</td><td rowspan=1 colspan=1>Across</td><td rowspan=1 colspan=1>χ² (df)</td><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1>Cramér&#x27;sV</td></tr><tr><td rowspan=1 colspan=1>effect rate</td><td rowspan=1 colspan=1>4 frameworks</td><td rowspan=1 colspan=1>2.95 (3)</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.12</td></tr><tr><td rowspan=1 colspan=1>effect rate</td><td rowspan=1 colspan=1>6 models</td><td rowspan=1 colspan=1>15.87 (5)</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1>0.23</td></tr><tr><td rowspan=1 colspan=1>unbacked claims</td><td rowspan=1 colspan=1>4 frameworks</td><td rowspan=1 colspan=1>7.92 (3)</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.20</td></tr><tr><td rowspan=1 colspan=1>unbacked claims</td><td rowspan=1 colspan=1>6 models</td><td rowspan=1 colspan=1>35.33 (5)</td><td rowspan=1 colspan=1>&lt; 0.0001</td><td rowspan=1 colspan=1>0.35</td></tr></table>

Effect rate: share of runs with at least one effect; unbacked claims: runs without effect that claim success.

The automation re-analysis identifies heterogeneity within each of the two separate sweeps (Table LXIX). The effect-rate test across four frameworks yielded p = 0.40 and Cramér’s V = 0.12, whereas the test across six model conditions yielded p = 0.007 and V = 0.23. Unbacked-claim rates also varied (frameworks V = 0.20, p = 0.048; models V $= \ 0 . 3 5 , \ \mathrm { ~ p ~ < ~ } 0 . 0 0 0 1 )$ . These are within-sweep association statistics, not estimates of causal effects or a directly comparable effect-size decomposition in a crossed factorial design. Illustrative pairwise contrasts include CrewAI versus LlamaIndex effect rates of 57.1% versus 44.9%, Sonnet 4.6 versus gpt-4o of 73.5% versus 44.9%, and gpt-5.5 unbacked claims of 2/49 versus 14/49 for gpt-4o. The results are unreplicated and do not score whether the observed state changes are correct. Uncertainty in the chatbot-finalist ranking is shown in Table LXX.

TABLE LXX. UNCERTAINTY OF THE CHATBOT FINALIST RANKING (15 RUNS PER MODEL, NORMAL APPROXIMATION)
<table><tr><td rowspan=1 colspan=1>Comparison or model</td><td rowspan=1 colspan=1>Welch t</td><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1>P(highestmean)</td><td rowspan=1 colspan=1>P(Pareto)</td></tr><tr><td rowspan=1 colspan=1>Terra vs Haiku</td><td rowspan=1 colspan=1>1.09</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Terra vs Gemini 3.1 Pro</td><td rowspan=1 colspan=1>1.78</td><td rowspan=1 colspan=1>0.093</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Haiku vs Gemini 3.1 Pro</td><td rowspan=1 colspan=1>1.14</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>GPT-5.6 Terra</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.86</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.03</td></tr></table>

Means and standard deviations from Table LI; 200,000 draws of the mean correctness from a normal distribution with the standard error; latency taken as reported, since its spread was not reported.

For the chatbot finalists, none of the pairwise differences in mean correctness reaches the 5% level (Terra against Haiku p = 0.29; Terra against Gemini p = 0.093). Under the stated independent normal approximation, Terra ranks first in approximately 83% of the simulated draws, compared with 13% for Haiku and 3% for Gemini. These are model-dependent ranking probabilities rather than calibrated probabilities of real-world superiority. The selection of Terra is thus a reasonable bet with a stated probability, not a statistically established winner, which is the form in which the testbed reports such decisions.

## K. Reference Implementation and Generated Artefacts

To verify that the formal method is computationally executable, we implemented its core as a Python package, capmap, including typed measurement records, context-aware comparisons, Pareto selection, dashboard specifications and seeded simulations. The code reproduces the deterministic calculations and invariants used here. These verification checks establish implementation consistency, not additional cross-application quality observations.

Generated metamodel. A schema-validated model profile was constructed from six measured attributes across three application probes, with seven catalogue attributes explicitly represented as unmeasured (Listing 1). This distinguishes absent evidence from an observed value of zero.

Listing 1. Excerpt of a generated sparse model profile (measured attributes and explicit missing-value markers; JSON abridged).  
```jsonl
{ "metamodels": { "claude": { "claude-haiku-4-5": {
"attributes": {
"chat.answer_correctness": [ {
"value": 0.989, "n": 15,
"context": "chat_role/dense_top4",
"dataset": "mvp_scenarios@v6",
"plan_hash": "7eaad59c7dfe7c28...c1c2",
"evidence": "L3" } ],
"aut.unbacked_claim_rate": [ {
"value": 0.2245, "n": 49,
"context": "llamaindex/baseline",
"dataset": "pilot.v4", "evidence": "L1" } ],
"aut.intent_accuracy": [ { "value": 0.971, "n": 40, ...
} ],
... },
"not_measured": [
"chat.citation_validity", "chat.faithfulness",
"chat.retrieval_mrr", "o2pg.conversion_accuracy",
"o2pg.judged_quality", "o2pg.refusal_rate",
"ops.cost_usd" ] } } } }
```

Generated dashboards. For the attribute declarations of the four probes (P1 conversion, the chatbot, automation and migration) the provisioner generated one dashboard each (Table LXXI).

TABLE LXXI. DASHBOARDS GENERATED FROM ATTRIBUTE DECLARATIONS
<table><tr><td rowspan=1 colspan=1>Probe</td><td rowspan=1 colspan=1>Attributes</td><td rowspan=1 colspan=1>Panels</td><td rowspan=1 colspan=1>Panel types</td></tr><tr><td rowspan=1 colspan=1>P1 conversion</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>10 gauge, 3 bar chart, 1 timeseries, 1 table</td></tr><tr><td rowspan=1 colspan=1>P3 chatbot(p2 chatbot)</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>20 gauge, 1 stat, 1 time series, 1bar chart, 1 table</td></tr><tr><td rowspan=1 colspan=1>P4 automation</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>8 gauge, 2 stat, 2 bar chart, 1state timeline, 1 table</td></tr><tr><td rowspan=1 colspan=1>P2 migration</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>6 stat, 3 bar chart, 1 statetimeline, 1 table</td></tr></table>

Each dashboard also has three template variables. Text-typed attributes would be omitted.

Every dashboard carries three template variables (context, dataset and model), so that a panel can only show values of one context class; every attribute becomes a panel chosen by its unit (gauge for ratios, stat for counts, time series for latency, bar chart for cost and tokens, state timeline for verdicts), with “not measured” as the no-value text, and two panels common to all dashboards show the spend by role and the run health. The generated documents are valid JSON of the dashboard model; importing them into a running Grafana server and observing live data was not part of this work.

Property checks. The three lemmas of Section III-B and two properties of the noise rule were checked on 2,000 random instances each: the plan hash is invariant under permutation of runs; comparability is reflexive, symmetric and transitive; the Pareto set of a finite non-empty set is non-empty; the noise rule is symmetric under exchange of the arms (verdict equal, difference of opposite sign); and it is invariant under a common shift of both arms. All 10,000 checks passed. They do not replace a proof but would detect an implementation that violated the properties on which the mechanisms rely.

## L. Throughput, Latency and Cost ofthe Testbed Itself

The testbed itself can be evaluated for execution volume, latency and measurement cost (Table LXXII). The P1 evaluation converted 1,802 files with concurrency eight in 48 minutes and scored the resulting outputs in 14 min 38 s. These quantities characterise the tested hardware, gateway and dataset, rather than the throughput of an arbitrary deployment.

Values measured in the evaluation; throughput ratios are arithmetically derived.  
TABLE LXXII. MEASURED THROUGHPUT, LATENCY AND VOLUME
<table><tr><td rowspan=1 colspan=1>Activity</td><td rowspan=1 colspan=1>Figure</td></tr><tr><td rowspan=1 colspan=1>P1 conversion, 1,802 files, concurrency 8</td><td rowspan=1 colspan=1>48 min (≈ 38 files/min, derived)</td></tr><tr><td rowspan=1 colspan=1>P1 evaluation with judge</td><td rowspan=1 colspan=1>14 min 38 s</td></tr><tr><td rowspan=1 colspan=1>P1 evaluation without judge (from database)</td><td rowspan=1 colspan=1>≈7s</td></tr><tr><td rowspan=1 colspan=1>Chatbot response time, finalists</td><td rowspan=1 colspan=1>9.8 s / 11.6 s / 14.0 s</td></tr><tr><td rowspan=1 colspan=1>Intent classification, one repeat of 40 cases</td><td rowspan=1 colspan=1>72 calls; ≈ 37.4 k tokens; 107 s vs17.1 s control</td></tr><tr><td rowspan=1 colspan=1>Decomposition LLM fallback, 165 units</td><td rowspan=1 colspan=1>≈ 37 min, sequential</td></tr><tr><td rowspan=1 colspan=1>Automation run, mean tokens</td><td rowspan=1 colspan=1>1,953 – 9,465 depending on model</td></tr></table>

M. Cross-Application Synthesis: the Evidence Map and the Capability Map

Fig. 32 shows which models have been measured on which application. The matrix is sparse by construction: P1 was measured with one model, automation with six models in one framework, the chatbot with nine models in a screening and three in repeated runs, and the migration agent with the models of its two evidence levels. Sparsity is the reason for the principle that missing is not zero: any ranking that filled the empty cells would be a statement about the filler.

Table LXXIII collects the observations that can be stated per model with the evidence at hand. Each cell is an observation from one application under one configuration; it is not a general property of the model. The comparable statement across applications is structural rather than numerical: the three directions agree that a system’s own report of success overstates it. The automation agents claim work they did not do; of the translations the migration agent accepts, 10 to 22% are wrong or cannot be verified against Oracle; and in the chatbot an accepted configuration change is not the same as a passing scenario.

<table><tr><td rowspan=2 colspan=10>Pl         Migration       Chatbot       Chatbot      Automation     Automation       Intentconversion       agent        9 models       repeats      frameworks      6 models       classifier</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9 models</td><td rowspan=1 colspan=1>repeats</td><td rowspan=1 colspan=2>frameworks</td></tr><tr><td rowspan=1 colspan=1>gpt-4.1-nano</td><td rowspan=1 colspan=1>●</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>gpt-4o</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>gpt-5.5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Claude Sonnet 4.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Gemini 2.5 Pro</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>GPT-5.6 Terra</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>GPT-5.6 Sol</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>GPT-5 mini</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Claude Sonnet 5, Opus 5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Gemini 3.8 Flash</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>DeepSeek V4 Flash</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>DeepSeek V4 Pro</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>gpt-5.4 nano, mini; gpt-5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr></table>

Fig. 32. Evidence coverage of tested model conditions across the reference applications; a missing mark denotes that no measurement exists for the indicated combination.

TABLE LXXIII. CAPABILITY OBSERVATIONS PER MODEL (SINGLE APPLICATION, SINGLE CONFIGURATION)
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Application</td><td rowspan=1 colspan=1>Observation</td><td rowspan=1 colspan=1>Evidence</td></tr><tr><td rowspan=1 colspan=1>gpt-4.1-nano</td><td rowspan=1 colspan=1>P1 conversion</td><td rowspan=1 colspan=1>abbreviates about 69 large files instead of converting them</td><td rowspan=1 colspan=1>full cycle, 1 run</td></tr><tr><td rowspan=1 colspan=1>gpt-4.1-nano</td><td rowspan=1 colspan=1>intent classification</td><td rowspan=1 colspan=1>89.7%; description rewrite +4.6 pp</td><td rowspan=1 colspan=1>5 repeats</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Flash</td><td rowspan=1 colspan=1>intent classification</td><td rowspan=1 colspan=1>94.3%, zero variance; rewrite -3.4 pp</td><td rowspan=1 colspan=1>5 repeats</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>intent classification</td><td rowspan=1 colspan=1>97.1%, zero variance; rewrite 0.0 pp</td><td rowspan=1 colspan=1>5 repeats</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>chatbot</td><td rowspan=1 colspan=1>fastest finalist (9.8 s), correctness 0.989</td><td rowspan=1 colspan=1>15 runs</td></tr><tr><td rowspan=1 colspan=1>GPT-5.6 Terra</td><td rowspan=1 colspan=1>chatbot</td><td rowspan=1 colspan=1>highest correctness 0.995, lowest variance</td><td rowspan=1 colspan=1>15 runs</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1>chatbot</td><td rowspan=1 colspan=1>faithfulness 1.000 (highest of the finalists); slowest finalist</td><td rowspan=1 colspan=1>5 scenarios</td></tr><tr><td rowspan=1 colspan=1>Claude Sonnet 4.6</td><td rowspan=1 colspan=1>automation</td><td rowspan=1 colspan=1>most thorough (1.8 effects) and costly (9.5 k tokens); unrequested actions; no unbackedclaims</td><td rowspan=1 colspan=1>49 samples, 1 run</td></tr><tr><td rowspan=1 colspan=1>gpt-5.5</td><td rowspan=1 colspan=1>automation</td><td rowspan=1 colspan=1>minimal action, but mostly says so (2 unbacked claims)</td><td rowspan=1 colspan=1>49 samples, 1 run</td></tr><tr><td rowspan=1 colspan=1>Gemini 2.5 Pro</td><td rowspan=1 colspan=1>automation</td><td rowspan=1 colspan=1>multi-step tasks done; sent no email on email compose</td><td rowspan=1 colspan=1>49 samples, 1 run</td></tr><tr><td rowspan=1 colspan=1>gpt-4o, Haiku 4.5, Gemini 3.1Pro</td><td rowspan=1 colspan=1>automation</td><td rowspan=1 colspan=1>often report work not done (14, 11, 9 of 49)</td><td rowspan=1 colspan=1>49 samples, 1 run</td></tr></table>

Observations are conditional on the specified application, configuration and evidence level; they are not global model rankings.

TABLE LXXV. SUMMARY OF THE ANSWERS TO THE RESEARCH QUESTIONS

The two aggregate patterns are worth stating separately. First, within the evaluated automation conditions, observed effects and unbacked success claims varied more across model choices than across framework choices; the limited design does not establish a general causal ordering. A tuning result also depended on the deployment (intent classification). Second, the quality measured by a stage-wise instrument differs from the quality claimed by the system: the metamodel keeps both, as the attributes ACCEPTED and demonstrably correct, effects and unbacked claims, execution success and answer correctness.

For a descriptive test of multi-objective selection, Algorithm 7 was applied to the measured automation aggregates in Table XLVII using mean token count and unbacked success claims as two objectives to minimise. The resulting exact Pareto set contains gpt-5.5 and Gemini 2.5 Pro; all other compared conditions are dominated on those two recorded dimensions (Table LXXIV). This computation must not be interpreted as a deployment recommendation because it omits task correctness and relies on one run per condition.

TABLE LXXIV. TWO-OBJECTIVE PARETO ANALYSIS OF AUTOMATION MEASUREMENTS (TOKEN USE AND UNSUPPORTED CLAIMS)
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Tokens</td><td rowspan=1 colspan=1>Unbacked</td><td rowspan=1 colspan=1>Status</td></tr><tr><td rowspan=1 colspan=1>gpt-5.5</td><td rowspan=1 colspan=1>1,953</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Pareto</td></tr><tr><td rowspan=1 colspan=1>Gemini 2.5 Pro</td><td rowspan=1 colspan=1>2,516</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Pareto</td></tr><tr><td rowspan=1 colspan=1>gpt-40</td><td rowspan=1 colspan=1>2,055</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>dominated by gpt-5.5</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1>3,147</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>dominated by gpt-5.5</td></tr><tr><td rowspan=1 colspan=1>Haiku 4.5</td><td rowspan=1 colspan=1>5,384</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>dominated by gpt-5.5</td></tr><tr><td rowspan=1 colspan=1>Sonnet 4.6</td><td rowspan=1 colspan=1>9,465</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>dominated by Gemini 2.5 Pro</td></tr></table>

Single-run aggregates from Table XLVII; no spreads available; effects and correctness not used.

## IX. DISCUSSION

## A. Answers to the Research Questions

RQ1 (architecture). The testbed treats the application as the unit of extension. An application is a container behind a small contract (health, options, evaluation, attribute declaration, provenance), configured by template variables and, when a variable names an unsupported dataset or instance, by a specification-driven generation hook whose output must pass a smoke test. Onboarding registers a probe, adds owned attributes to the metric catalogue and registers dashboards. The chatbot application was integrated by following this pattern without a chatbot-specific frontend: it appeared in the shared Use Cases, Metrics and Evaluations pages with its own chat.\* attributes, and the core was not edited. Reproducibility and cost transparency come from frozen, hashed plans, prompt identity and three separate spends. What we have not shown is the effort required to onboard a fourth application of a very different shape; the placeholder probe of opposite shape that ships with the core keeps the contract honest structurally, but only a second real probe will show it in practice. Table LXXV summarises the answers.

RQ2 (trust and diagnosis). The evaluation design separates model, judge and orchestration roles; deterministic metric code prevents a model from supplying its own score. Frozen plans, dataset identities, instrument checks, negative controls and explicit evidence levels prevent several observed measurement errors. The remaining variability and limitations of the source-system oracle are discussed in Section IX-J.

<table><tr><td rowspan=1 colspan=1>Question</td><td rowspan=1 colspan=1>Answer</td><td rowspan=1 colspan=1>Evidence</td></tr><tr><td rowspan=1 colspan=1>RQ1architecture</td><td rowspan=1 colspan=1>application = container + contract +owned attributes; onboardingregisters probe, metrics, dashboards</td><td rowspan=1 colspan=1>chatbot integratedwithout core edit; Fig.8</td></tr><tr><td rowspan=1 colspan=1>RQ2 trust</td><td rowspan=1 colspan=1>instrument ≠ subject; codecomputes; missing ≠ zero; outage ≠verdict; null-agent check; noisefloor</td><td rowspan=1 colspan=1>defect tables; nullagent beat the agent</td></tr><tr><td rowspan=1 colspan=1>RQ3behaviour</td><td rowspan=1 colspan=1>separate model/framework sweepsshow different heterogeneity;prompt effects aredeployment-specific; acceptanceoverstates correctness</td><td rowspan=1 colspan=1>E2-E9</td></tr><tr><td rowspan=1 colspan=1>RQ4decision</td><td rowspan=1 colspan=1>Pareto selection with evidencecounts and refusal to mix contexts;no ranking claimed</td><td rowspan=1 colspan=1>Algorithm 7; TablesLVI, LXXVIII</td></tr></table>

RQ3 (behaviour). Three preliminary observations emerge. First, model and framework sweeps in the automation stand show different magnitudes of variation in observed side effects and unsupported success claims, without identifying a causal ordering between these factors. Second, rewriting an instruction produced gains, losses or no change across deployments, so prompt effects must be assessed in the specific deployment. Third, reports of completion often exceeded externally observed actions, while migration results also depended on what the measurement stand could test. The chi-square results (model p = 0.007; framework p = 0.40 for effect rate) describe heterogeneity within separate, one-run sweeps; they do not test a model-by-framework interaction or establish semantic task correctness.

RQ4 (decision). The measurement store and deterministic analytical API support task-constrained model filtering, evidence-aware comparisons, Pareto frontiers and the explicit exclusion of models with missing or incompatible measurements. The empirical Pareto calculation in Section VIII-M illustrates these mathematical operations; it is not a universal model ranking or a validated deployment choice.

## B. Implications for Model Selection

Four practical rules follow for teams that select models for applications like ours. (i) Measure the model in the application, in the configuration that will be used; the retrieval settings of the chatbot were chosen by controlled experiments, one variable at a time, and the model was chosen last. (ii) Change the model before changing the words: an instruction tuned on one deployment may do nothing, or harm, on another. (iii) Distrust the system’s own success report; compare it with independent evidence such as effects, Oracle’s output or the frozen world state. (iv) Treat cost as an objective rather than a footnote, including the cost of measuring: the judge’s spend is substantial, a cheaper model can be unusable for lack of quota, and cost reports themselves can be wrong. These rules agree with the observation of HELM that accuracy alone is an incomplete report [3] and with the cost-aware cascades and routers of the model-selection literature [59], [60], but they are drawn from measured defects of our own stands.

## C. Extending the Metamodel: Planned Probes

The scheme names four further attribute groups, P5 to P8, which are not implemented. Table LXXVI sketches what each could measure if onboarded as an application, with the earlier work that would feed it. We include the table to show how the onboarding mechanism would be used; it is a plan and not a result.

TABLE LXXVI. PLANNED PROBES AND CANDIDATE APPLICATIONS
<table><tr><td rowspan=1 colspan=1>Probe</td><td rowspan=1 colspan=1>Attribute group</td><td rowspan=1 colspan=1>Candidate application</td><td rowspan=1 colspan=1>Feeds from</td></tr><tr><td rowspan=1 colspan=1>P5</td><td rowspan=1 colspan=1>securitytreatment</td><td rowspan=1 colspan=1>security-aware generationand guardrail evaluation</td><td rowspan=1 colspan=1>[67], [74]</td></tr><tr><td rowspan=1 colspan=1>P6</td><td rowspan=1 colspan=1>token efficiency</td><td rowspan=1 colspan=1>token-optimisation harnessfor migration</td><td rowspan=1 colspan=1>[63]</td></tr><tr><td rowspan=1 colspan=1>P7</td><td rowspan=1 colspan=1>requiredresources</td><td rowspan=1 colspan=1>latency, memory andcontext-window probe</td><td rowspan=1 colspan=1>operationalset</td></tr><tr><td rowspan=1 colspan=1>P8</td><td rowspan=1 colspan=1>DevOps</td><td rowspan=1 colspan=1>specification-driven DevOpstasks</td><td rowspan=1 colspan=1>[45], [78]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>model drift</td><td rowspan=1 colspan=1>repeat a frozen plan overtime and apply drift detection</td><td rowspan=1 colspan=1>[71], [72]</td></tr></table>

Two pieces of prior work are natural seeds for P5. A study of security knowledge in specification-driven development found that functional behaviour can be specified explicitly while security behaviour stays implicit, so that generated systems satisfy the visible requirements yet miss ownership checks, administrator safety rules or token-misuse rejection; generic guidance such as ASVS improved generated behaviour, and explicit application-specific security modelling improved it further, most visibly in business-logic and administrator-safety categories [74]. And a guardrail harness around specification-driven workflows returns a structured decision (pass, warning or block) with traceable findings, and ships a deterministic evaluation suite that exits non-zero on unexpected decisions or missing family coverage; its precision and recall would be natural P5 attributes, alongside the offline guardrail measurement reported in Section VIII-C.

## D. Towards Steps 2 and 3: Selecting a Modelfor a Task

Steps 2 and 3 of the programme, which are out of scope here, are the intended long-term consumers of the metamodel; stating their interface now clarifies what the metamodel must provide. A task is a structured object T = {agents, operations, environments, modules, context, …}. Step 2 turns requirements into a specification Spec(T, Ψ(m)) that takes the chosen model’s profile into account, for example by adding the safeguards that the profile shows the model needs, and Step 3 splits the specification into parts $T _ { 1 } ,$ T<sub>k</sub> and chooses for each part a model and a configuration. Given a mapping from a task part to the attributes that matter for it, the choice is a constrained multi-objective problem:

$$
\begin{array} { r l r } & { } & { ( \mathrm { m j } , \mathrm { \boldsymbol { \kappa } j } ) \in \mathrm { P } ( \mathrm { O j } ) \mathrm { o v e r } \{ ( \mathrm { m } , \boldsymbol { \kappa } ) \colon [ \Psi ( \mathrm { m } ) ( \sigma ) \geq 6 \sigma \mathrm { i f } \delta \sigma = \mathrm { m a x } ; } \\ & { } & { \Psi ( \mathrm { m } ) ( \sigma ) \leq 6 \sigma \mathrm { i f } \delta \sigma = \mathrm { m i n } ] \mathrm { f o r e v e r y } \sigma \in \Sigma \mathrm { j } \} . \quad ( 1 1 ) } \end{array}
$$

where Σj is the set of attributes that part Tj depends on, θσ are hard thresholds and δσ specifies whether each attribute is maximised or minimised, and the Pareto set P is taken over the objectives Oj of Equation (5) in comparable contexts only. The equation makes the dependencies of the later steps explicit. It needs a semantic bridge from a task description to the attributes that matter, which is item 6 on the programme’s list of questions; it needs thresholds that are agreed rather than placeholders; and it needs the attributes to be measured in a context that resembles the deployment, which is why the application is part of the measurement. The partiality of the metamodel is a feature here: a model for which a required attribute is absent is excluded with a stated reason, exactly as in the exclusion rule in Algorithm 7.

## E. Relation to Earlier Work of the Group

The testbed continues a line of work on specification-driven development and monitoring. The comparative analysis of frameworks for the same three application classes [78] provided the architectures; the unified architecture metamodel [77] supplied the idea that a model or system can be described by a versioned, structured record; the benchmark on what a specification determines [46] and the leakage-resistant evaluation of specification judges [45] supplied the discipline of execution-judged evaluation and of sealed ground truth; the token-optimisation study [63] showed the multi-objective nature of cost. The monitoring publications of co-author V. Lyashkevych [69]–[73] investigate ontology-driven dashboards and drift in evolving information systems, and a natural extension of the testbed is to apply their drift detection to the metamodel itself, so that a change in a hosted model, a gateway or a dataset version is detected as drift rather than discovered by a surprised user. The security and risk perspective [74], [75] is the basis for the planned security attributes (P5), and the orchestration of agents with dynamic involvement of new agents [76] relates to the planner of the core.

## F. Generalisability and Alternative Designs

The probe-per-application design has competitors. A single monolithic harness that runs all applications in one process would have been simpler to start and would have given one language, one dependency set and one test runner. We rejected it because the applications are built by different teams with different stacks (the migration agent depends on a parser generator and a PostgreSQL container, the chatbot on a document index and a graph runtime, the automation agent on four agent frameworks), and because the frameworks themselves conflict: two frameworks that both want to run the steps cannot be assumed to nest. The price of isolation is a network boundary and a contract to maintain. A second alternative is to adopt an existing evaluation library as the definition of a score. We use such libraries as implementations of individual axes, but not as the answer to what a score is, because they supply implementations and not a hierarchy of sources, a significance threshold or a control arm. A third alternative, a single leaderboard, is simply a different product.

Three conditions limit generalisation. The design assumes that an application can expose an evaluation API and declare its attributes; an application whose evaluation requires a human in the loop at every step would need a different probe type. It assumes a single gateway for model access, which is convenient for cost attribution but couples the testbed to the gateway’s deployment catalogue, including its quotas and the identifiers it returns. And it assumes that the programme’s model of an application, a dataset plus a configuration space plus an evaluator, is adequate; applications whose behaviour depends on long-lived state that cannot be reset between runs would violate the fresh-state assumption on which the repeat logic rests.

## G. Lessons Learned

Table LXXVII lists the lessons that the authors consider transferable to teams that build similar infrastructure. Most of them are about instruments, not models.

TABLE LXXVII. TRANSFERABLE LESSONS
<table><tr><td rowspan=1 colspan=1>Lesson</td><td rowspan=1 colspan=1>Origin in this work</td></tr><tr><td rowspan=1 colspan=1>Score the agent that does nothing beforetrusting a metric.</td><td rowspan=1 colspan=1>an empty migrationoutscored the agent on everypublished headline metric</td></tr><tr><td rowspan=1 colspan=1>A verdict is only as strong as the weakestlevel at which it was checked.</td><td rowspan=1 colspan=1>about 78% of &#x27;ok&#x27;verdictsproved nothing; 108 of 216demonstrable</td></tr><tr><td rowspan=1 colspan=1>Read the failures before fixing the model.</td><td rowspan=1 colspan=1>FAILED contained 14faithful raises and 8 invalidinputs</td></tr><tr><td rowspan=1 colspan=1>Expect a correct fix to expose the nextdefect, and to move the headline the wrongway.</td><td rowspan=1 colspan=1>ACCEPTED fell after acorrectness fix; aFIX PARTIAL is usually acorrect fix</td></tr><tr><td rowspan=1 colspan=1>Change the model before changing thewords.</td><td rowspan=1 colspan=1>a rewrite worth +4.6, -3.4and 0.0 points on threedeployments</td></tr><tr><td rowspan=1 colspan=1>Separate the price of measuring from theproperty measured.</td><td rowspan=1 colspan=1>judge spend exceededconversion spend in P1</td></tr><tr><td rowspan=1 colspan=1>An outage must not look like a bad agent.</td><td rowspan=1 colspan=1>109 FAILED units from anetwork outage</td></tr><tr><td rowspan=1 colspan=1>Pin the ruler.</td><td rowspan=1 colspan=1>two linter versions in onestand; metric libraries frozen</td></tr><tr><td rowspan=1 colspan=1>Put the denominator in the record.</td><td rowspan=1 colspan=1>a blank denominator passeda presence check</td></tr><tr><td rowspan=1 colspan=1>Where two models disagree, read the unit.</td><td rowspan=1 colspan=1>first silently wrong acceptedtranslation found bydisagreement</td></tr></table>

Each rule is supported by the measured incidents and verification results described in Sections VII and VIII.

## H. Practical and Ethical Considerations

The application datasets are private or synthetic, so no personal data enter the store; the automation dataset in particular is a closed simulation with synthetic content, which avoids side effects on real tickets and emails. The testbed spends money on every run, and in the experiments the spend was shared by several users of one key: the migration team recorded that the shared key stood at about USD 331 of a USD 500 rolling budget at one point. The design decision that the caller’s key pays for the call, and that no service stores it, is therefore also an ethical decision about who bears the cost of measuring. Human approval gates remain in the loop where a change would reach a live system (publication of a knowledge snapshot) or where spending is unattended (pre-registration sign-off and permission to spend are two explicit gates in the fix cycle). Finally, because the metamodel can be used to decide which model handles which task, it can be misused to over-trust a profile; the partial metamodel, evidence counts and context warnings are the countermeasures built into the interface, and we recommend that downstream consumers keep them.

## I. Threats to Validity and Limitations

The testbed implementation, automated verification and empirical measurements in Sections V–VIII provide the evidence base for the claims in this work. The limitations concern generalisability, statistical sensitivity, unmeasured dimensions, and transfer to new applications rather than the existence of the implemented artifacts. The complete set of threats is tabulated in Table LXXVIII.

Two further limitations concern the design. The caller’s key crosses a process boundary in the task message and is protected operationally, not cryptographically. And ground-truth leakage is controlled mechanically only in the specification-evaluation work [45]; for the chatbot and the automation agent it is controlled by construction (expected answers and tools are used only by the evaluation engine), which is weaker. Finally, public benchmark contamination [49], [50] is not a concern for the application datasets, which are private or synthetic, but it would become one if the testbed were used on public data.

## J. Validated Uses, Boundaries of Inference, and Transfer Conditions

Interpretation principle. The capability profile is evidence about a model operating inside an explicitly configured application and measurement procedure, not a context-free estimate of model intelligence. The testbed can record which application, dataset version, prompt variant, evaluator and deployment produced a value; attach evidence and sample counts; keep unmeasured attributes absent; and expose comparisons under controlled conditions. These functions are implemented and exercised in the reported stands. Their correctness as software mechanisms must not be conflated with the much stronger proposition that any resulting score predicts performance on an unseen production workload. The former is established by the implemented pipeline and its tests; the latter is a transfer claim that needs additional data.

TABLE LXXVIII. THREATS TO VALIDITY AND LIMITATIONS
<table><tr><td rowspan=1 colspan=1>Threat</td><td rowspan=1 colspan=1>Where</td><td rowspan=1 colspan=1>Status / mitigation</td></tr><tr><td rowspan=1 colspan=1>single runs, small samples</td><td rowspan=1 colspan=1>E1-E3, E5, E7</td><td rowspan=1 colspan=1>reported as single run; noise rule applied; repeats planned</td></tr><tr><td rowspan=1 colspan=1>P1 noise floor unmeasured</td><td rowspan=1 colspan=1>P1 conversion</td><td rowspan=1 colspan=1>model numbers withheld: repeated identical plans planned</td></tr><tr><td rowspan=1 colspan=1>automation not scored against ground truth</td><td rowspan=1 colspan=1>E2, E3</td><td rowspan=1 colspan=1>claims and effects reported as such; scoring pending</td></tr><tr><td rowspan=1 colspan=1>applicability thresholds are placeholders</td><td rowspan=1 colspan=1>automation Stage 0</td><td rowspan=1 colspan=1>screening not executable until agreed</td></tr><tr><td rowspan=1 colspan=1>synthetic, partly unreachable dataset</td><td rowspan=1 colspan=1>automation pilot.v4</td><td rowspan=1 colspan=1>15 samples need an unreadable address: known</td></tr><tr><td rowspan=1 colspan=1>self-authored labels; one set</td><td rowspan=1 colspan=1>intent classification</td><td rowspan=1 colspan=1>upper bound; control arm is a floor</td></tr><tr><td rowspan=1 colspan=1>judge instability and self-preference</td><td rowspan=1 colspan=1>chatbot, P1</td><td rowspan=1 colspan=1>16.2% churn at T = 0; judge separated; cross-vendor judge open</td></tr><tr><td rowspan=1 colspan=1>migration headline predates last fixes</td><td rowspan=1 colspan=1>E8</td><td rowspan=1 colspan=1>estimate 115–120 awaits next full run</td></tr><tr><td rowspan=1 colspan=1>token counts not comparable across vendors</td><td rowspan=1 colspan=1>E3</td><td rowspan=1 colspan=1>stated; own tokenizers</td></tr><tr><td rowspan=1 colspan=1>shared quota and key</td><td rowspan=1 colspan=1>E7, E8</td><td rowspan=1 colspan=1>full runs depend on a daily cap</td></tr><tr><td rowspan=1 colspan=1>Grafana registration not evaluated</td><td rowspan=1 colspan=1>Section V-H</td><td rowspan=1 colspan=1>mechanism described; no quantitative result</td></tr><tr><td rowspan=1 colspan=1>generation hook not evaluated</td><td rowspan=1 colspan=1>Section V-F</td><td rowspan=1 colspan=1>contract and smoke test only</td></tr><tr><td rowspan=1 colspan=1>models of different periods and gateways</td><td rowspan=1 colspan=1>all</td><td rowspan=1 colspan=1>context recorded; no cross-experiment ranking</td></tr><tr><td rowspan=1 colspan=1>simulation assumes normal i.i.d. repeats, one axis</td><td rowspan=1 colspan=1>Section VIII-I</td><td rowspan=1 colspan=1>characterises the rules, not the stands; repeats may be correlated</td></tr><tr><td rowspan=1 colspan=1>unpaired tests on paired runs; no multiplicity correction</td><td rowspan=1 colspan=1>Section VIII-J</td><td rowspan=1 colspan=1>descriptive; paired inference unavailable without unit-level records</td></tr><tr><td rowspan=1 colspan=1>generated dashboards not imported into a Grafana server</td><td rowspan=1 colspan=1>Section VIII-K</td><td rowspan=1 colspan=1>validated as JSON only: live evaluation pending</td></tr></table>

The applicability restrictions are derived from the actual measurement designs and are not claims about untested models or applications.

What a model-selection decision can support. For an application represented by a reference probe, an engineer can filter models by measured hard constraints, examine separate quality, effect, latency and cost attributes, and construct a Pareto set over objectives evaluated in a common context (Sections III-A, V-H and VIII-M). The interface also reveals missing attributes, context mismatches and evidence counts; it can decline a comparison instead of silently imputing missing scores. This is a defensible use of the profile as a decision aid or shortlist. The system does not establish that the leading model is universally best, that an unmeasured constraint has been satisfied, or that a Pareto-optimal candidate will remain optimal after changes in the workload, provider or evaluation method. A selection must be revisited whenever the target context differs materially from the recorded context.

Reproducibility versus repeatability. Frozen plans, hashes, dataset identifiers and registered prompts make the intended experiment and its provenance recoverable. Replaying a frozen plan avoids asking an LLM planner to reproduce its earlier choices; it does not force a hosted model, judge, external API or concurrent runtime to return byte-identical outputs. Thus plan-level reproducibility is supported, whereas numerical repeatability is an empirical property to be measured separately for each attribute, model and deployment. The reported Monte Carlo experiments characterise the operating behaviour of the range-based and three-state decision rules under their stated assumptions; they are not a measurement of run-to-run noise in every hosted application. With single runs, observed contrasts are valid descriptive results but cannot carry a claimed confidence interval for a deployment-level mean.

Migration capability and semantic equivalence. The Oracle-to-PostgreSQL direction can decompose and order migration units, route them to conversion strategies, run bounded repairs and verify PostgreSQL execution. Its stronger evidence comes from differential checks against a real Oracle system. The reported 108 demonstrably correct units out of 216 refer to the exercised corpus and the behaviours tested there. They do not imply equivalence on every input, independence from schema state, complete support for the Oracle PL/SQL language or correctness of a complete enterprise migration. In particular, installation alone is weaker than behavioural equivalence; even L4 observations cover only the tested cases and source conditions. A new database estate therefore needs dependency coverage, fixture representativeness, representative input scenarios and re-execution against its own Oracle reference before a production-quality claim is made.

Chatbot capability and the limits of semantic judging. The chatbot probe can isolate execution, tool choice, retrieval, citation structure, grounding and judged answer quality; controlled retrieval experiments and the integrated smoke run demonstrate that these attributes can be produced and persisted. This does not make retrieval recall a property of the language model when the retrieval configuration is varied, nor does a well-formed citation prove that the answer is factually supported. Some expected labels in the scenario dataset remain provisional, and an LLM judge can disagree with itself or favour a related model family. The current semantic scores are therefore advisory unless independently validated against the intended task criteria. A high-stakes application needs a curated held-out set, a documented adjudication protocol, negative and refusal cases, and checks of both retrieved evidence and the final answer.

Automation capability: effects, claims and correctness. The enterprise-support stand uses production-shaped synthetic events and a frozen state to observe tool requests, accepted arguments, actual side effects, missing operations and unsupported success claims.

This permits a grounded comparison of observable action behaviour in the evaluated conditions. However, an action occurring is not the same as the intended business outcome being correct or authorised: an agent can change state and still choose the wrong issue, page, recipient or sequence. The pilot set contains 49 samples and known cases whose required address cannot be retrieved by any available tool. Accordingly, the model-versus-framework differences reported for side effects and unbacked claims are descriptive within separate sweeps; they are not a factorial causal estimate and do not establish end-to-end task accuracy. That stronger claim requires scored final states, matched model-by-framework cells and repeats over independent scenarios.

Evidence levels and instrument dependence. The evidence ladder is an explicit safeguard against treating all PASS labels alike. L0 self-reports, static checks, installation, reference comparisons and differential execution answer different questions. The testbed can report the strongest available verification and its denominator, distinguish a failure of the evaluation infrastructure from a failure of the tested model, and apply the null-agent check before promoting an aggregate to a headline metric. Nonetheless, an evaluator can overlook a defect when its tests are incomplete, and a metric can reward an unintended shortcut even when the evaluation code executes correctly. The operational conclusion is not that the recorded measurements are unreliable, but that every downstream claim should remain bounded by the specific property, test coverage and evidence level actually observed.

Statistical sensitivity and cross-context comparison. The range-based noise guard favours avoiding unsupported effect claims, but its sensitivity depends on the number and spread of repeats; the simulation indicates that simply increasing the number of repeats need not improve its power. The Welch-based second tier provides a suggestive signal under its modelling assumptions and does not replace a prespecified paired analysis where runs share cases. The existing unpaired re-analysis cannot recover unit-level transition counts absent from the reported aggregates, and several comparisons are unadjusted for multiplicity. Likewise, token counts from different providers are not automatically a common unit of computational effort. Cross-model comparisons therefore require an agreed metric definition, fixed dataset and evaluator versions, recorded pricing assumptions, and uncertainty assessment on the same experimental unit.

Software integration and operational boundary. MAS Core, the deterministic gate, the P1 execution path, the shared analytical interface and the chatbot integration have implementation and test evidence in this work. The migration and automation systems also have running evaluation stands, but have not yet been onboarded end to end as ordinary probes in the common core. The dashboard provisioner generates valid dashboard definitions, while automatic import into a live Grafana instance and its onboarding cost are not assessed. The specification-driven adapter-generation hook is described and subject to a smoke-test contract; the general quality of generated adapters has not been evaluated. These distinctions delimit the presently demonstrated level of application extensibility: the architecture and one integrated application show the mechanism, but integration effort and maintenance cost across a broader class of applications remain open empirical questions.

Security, privacy and deployment constraints. Per-caller model credentials, role-separated accounting, isolated evaluation state and explicit approval gates reduce several operational risks, but they do not constitute a comprehensive production security assurance. The caller key still crosses a worker/broker boundary without end-to-end cryptographic protection in the described design. No general guarantee is provided against prompt injection in arbitrary input documents, disclosure through a provider, unauthorised live side effects, or safety failures on untested tools; the security probe P5 remains planned. Availability and economics are also environment-dependent: shared quotas, deployment catalogues, gateway behaviour, pricing changes and service outages can limit scale or invalidate cost comparisons. Production deployment consequently requires its own secrets handling, access control, threat assessment, rate-limit testing and observation of live drift.

Conditions for transfer and future evaluation. The method is best transferred to a new application when (i) a representative, versioned task set and a defensible target outcome can be defined; (ii) the application can expose a resettable or otherwise controlled evaluation boundary; (iii) the probe declares metric ownership, units, directions and evidence sources; (iv) tested models and judges are assigned distinct roles; (v) configuration, software versions and prompt identities are retained; and (vi) the evaluator is challenged with null agents, negative cases and repeated controls. When one of these conditions is absent, the architecture can still collect diagnostic traces, but an evidence-based selection claim should be withheld or explicitly qualified. The next validation step is therefore not to invent a broader universal leaderboard: it is to connect additional real probes, measure onboarding effort and live dashboard operation, score automation ground truth, quantify within-context repeatability and test whether recommendations hold on new datasets and model deployments.

## K. Engineering Status

Table LXXIX lists the implementation status of the components at the time of writing. We include it to avoid ambiguity between what has been built and what is designed.

TABLE LXXIX. IMPLEMENTATION STATUS OF THE COMPONENTS
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Status</td></tr><tr><td rowspan=1 colspan=1>MAS Core: planning, gate, freezing,workers, aggregation</td><td rowspan=1 colspan=1>implemented; verified end to end on P1with gpt-4.1-nano</td></tr><tr><td rowspan=1 colspan=1>P1 conversion probe</td><td rowspan=1 colspan=1>measures for real; noise floor pending</td></tr><tr><td rowspan=1 colspan=1>Chatbot probe p2_chatbot</td><td rowspan=1 colspan=1>integrated; smoke run and interactivelaunch verified</td></tr><tr><td rowspan=1 colspan=1>Migration and automation probes</td><td rowspan=1 colspan=1>own stands; integration as probes planned</td></tr><tr><td rowspan=1 colspan=1>Analytical UI (Explore, Analytics,Launch, Agent)</td><td rowspan=1 colspan=1>implemented; analytics deterministic</td></tr><tr><td rowspan=1 colspan=1>Grafana connection to the metric store</td><td rowspan=1 colspan=1>connected; dashboard ownership separate</td></tr><tr><td rowspan=1 colspan=1>Automatic dashboard registration atonboarding</td><td rowspan=1 colspan=1>provisioner implemented and run on fourprobes; generated JSON not imported intoa live Grafana</td></tr><tr><td rowspan=1 colspan=1>Reference implementation of thecomputational core (capmap)</td><td rowspan=1 colspan=1>implemented; property checks andsimulations reproducible</td></tr><tr><td rowspan=1 colspan=1>Specification-driven generation hook</td><td rowspan=1 colspan=1>designed; not evaluated</td></tr><tr><td rowspan=1 colspan=1>Security, tokens, resources, DevOpsprobes (P5–P8)</td><td rowspan=1 colspan=1>planned</td></tr><tr><td rowspan=1 colspan=1>Steps 2 and 3 of the programme</td><td rowspan=1 colspan=1>planned</td></tr></table>

Measured implementation coverage of the application probes; planned functions are labelled separately.

## X. CONCLUSION AND FUTURE WORK

We presented a method for application-specific capability profiling of LLMs and the containerised testbed that realises it. The method profiles LLMs through reference applications and accumulates the measurements into a versioned, sparse metamodel of each model. The application is the unit of extension: onboarding registers a probe and owned attributes in the metric catalogue and prepares dashboard definitions, so that the set of measurable LLM features grows with the set of applications. The testbed enforces a small set of principles (instrument is not subject, agents decide and code computes, reproducibility by frozen plans, missing is not zero, outage is not a verdict) and two checks, the null-agent check and the noise-floor rule, each justified by a defect in the authors’ own stands. It diagnoses the instrument before the system. A simulation showed that the noise rule is a conservative screen whose sensitivity falls as repeats are added, which led to a three-state verdict with a calibrated second tier.

The three reference applications—a migration agent, a chatbot and an automation agent—showed context-dependent model behaviour, deployment-specific effects of instruction tuning, and gaps between reported success and externally observed outcomes. In the automation conditions tested, variation across models exceeded variation across frameworks on selected measured axes; the current experimental design does not justify generalising this ordering. The first result of the migration agent measured against real Oracle was 108 of 216 demonstrably correct units, with a ceiling limited by what the stand can verify. These findings are offered as the first measured structure of the metamodel, not as a ranking of models. The practical scope and non-guarantees of these measured capabilities, together with the conditions required to transfer them to new applications, are set out in Section IX-K.

Future work follows the open items. The immediate item is to characterise the run-to-run variation of P1 with repeats of one frozen plan, since everything downstream of “which prompt is better” waits on it. Next, the migration and automation stands will be connected as probes, which will prove the probe contract on a second and third real application and allow the evidence map of Fig. 32 to be filled by planned runs instead of by history. The automation results need ground-truth scoring and agreed thresholds for the screening stage. The quality of generated adapters and the effort of onboarding, as well as the dashboard registration, should be evaluated quantitatively, for example by the time and defects of onboarding a fourth application. Finally, with stable profiles, Steps 2 and 3 of the programme can be undertaken: generating specifications tailored to a chosen model, decomposing them and deciding which model and configuration handles each part.

## APPENDIX: ATTRIBUTE CATALOGUE OF THE CHATBOT PROBE

Table LXXX lists the attributes owned by the probe p2\_chatbot as shown on the Metrics page of the interface, with their group and direction. The operational attributes with the prefix ops. are common to all probes. Directions follow the interface: an upward arrow means that higher is better.

TABLE LXXX. ATTRIBUTES OF THE CHATBOT PROBE
<table><tr><td rowspan=1 colspan=1>Attribute</td><td rowspan=1 colspan=1>Group</td><td rowspan=1 colspan=1>Better</td></tr><tr><td rowspan=1 colspan=1>chat.agent_iterations</td><td rowspan=1 colspan=1>performance</td><td rowspan=1 colspan=1>lower</td></tr><tr><td rowspan=1 colspan=1>chat.answer correctness</td><td rowspan=1 colspan=1>quality (judge)</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.answer relevance</td><td rowspan=1 colspan=1>quality (judge)</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.citation completeness</td><td rowspan=1 colspan=1>grounding</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.citation_compliance</td><td rowspan=1 colspan=1>grounding</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.citation_source_match</td><td rowspan=1 colspan=1>grounding</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.citation validity</td><td rowspan=1 colspan=1>grounding</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.faithfulness</td><td rowspan=1 colspan=1>grounding(judge)</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.forbidden tool free rate</td><td rowspan=1 colspan=1>reliability</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.helpfulness</td><td rowspan=1 colspan=1>quality</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.metadata_complete</td><td rowspan=1 colspan=1>retrieval</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.refusal_appropriateness</td><td rowspan=1 colspan=1>quality</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.retrieval_empty_correct</td><td rowspan=1 colspan=1>retrieval</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.retrieval mrr</td><td rowspan=1 colspan=1>retrieval</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.retrieval_precision_at_k,r precision, recall at k</td><td rowspan=1 colspan=1>retrieval</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.task pass_rate</td><td rowspan=1 colspan=1>quality</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.tool argument_correctness,tool_selection_correctness</td><td rowspan=1 colspan=1>quality</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.tool_execution_success</td><td rowspan=1 colspan=1>reliability</td><td rowspan=1 colspan=1>higher</td></tr><tr><td rowspan=1 colspan=1>chat.turn latency_ms</td><td rowspan=1 colspan=1>performance</td><td rowspan=1 colspan=1>lower</td></tr><tr><td rowspan=1 colspan=1>ops.tokens_per_case</td><td rowspan=1 colspan=1>cost</td><td rowspan=1 colspan=1>lower</td></tr><tr><td rowspan=1 colspan=1>ops.completion_rate,ops.format_compliance</td><td rowspan=1 colspan=1>reliability</td><td rowspan=1 colspan=1>higher</td></tr></table>

Attribute names, categories, and optimization directions are defined by the chatbot probe's registered metric catalogue. Operational attributes are shared across application probes.

The metrics that the evaluation phase of the P1 stand collects and stores, and that serve as the data source for the dashboards, are the NLP aggregate and per-metric values (recall, BLEU, chrF), the four judge criteria, the linter error rate and warnings, the file efficiency and refusal rate of the annotators, and the run totals of tokens and cost. An inventory of the stored values was compiled from the code and feeds the discussion of the final metric set, which includes the observation that two linter-derived entries of the selectable NLP list duplicate the validation stage.

## DATA AND CODE AVAILABILITY

Data and code availability—The results are derived from implemented software, recorded application datasets and executed evaluations. Code artefacts, versioned configurations, test datasets and run identifiers are retained by the authors. Public access, licensing and the availability of restricted enterprise or Oracle datasets are subject to the distribution arrangements specified by the submitting authors; independent replication outside those arrangements is not asserted.

## DECLARATION ON GENERATIVE AI

During the preparation of this work, the authors used Generative AI tools for language editing and editorial refinement. The authors reviewed and verified the resulting text and take full responsibility for the content of the publication.

## REFERENCES

[1] D. Hendrycks et al., “Measuring massive multitask language understanding,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[2] A. Srivastava et al., “Beyond the imitation game: Quantifying and extrapolating the capabilities of language models,” Trans. Mach. Learn. Res., 2023.

[3] P. Liang et al., “Holistic evaluation of language models,” Trans. Mach. Learn. Res., 2023.

[4] W.-L. Chiang et al., “Chatbot Arena: An open platform for evaluating LLMs by human preference,” in Proc. Int. Conf. Mach. Learn. (ICML), 2024.

[5] W. Zhong et al., “AGIEval: A human-centric benchmark for evaluating foundation models,” in Findings Assoc. Comput. Linguistics: NAACL, 2024.

[6] L. Zheng et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets Benchmarks Track, 2023.

[7] Y. Chang et al., “A survey on evaluation of large language models,” ACM Trans. Intell. Syst. Technol., vol. 15, no. 3, 2024.

[8] M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh, “Beyond accuracy: Behavioral testing of NLP models with CheckList,” in Proc. Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020.

[9] D. Kiela et al., “Dynabench: Rethinking benchmarking in NLP,” in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics (NAACL-HLT), 2021.

[10] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Proc. NeurIPS, 2020.

[11] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, “BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models,” in Proc. NeurIPS Datasets Benchmarks Track, 2021.

[12] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “RAGAs: Automated evaluation of retrieval augmented generation,” in Proc. EACL (Syst. Demonstrations), 2024.

[13] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “ARES: An automated evaluation framework for retrieval-augmented generation systems,” in Proc. NAACL-HLT, 2024.

[14] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large language models in retrieval-augmented generation,” in Proc. AAAI Conf. Artif. Intell., 2024.

[15] N. F. Liu et al., “Lost in the middle: How language models use long contexts,” Trans. Assoc. Comput. Linguistics, vol. 12, pp. 157–173, 2024.

[16] V. Karpukhin et al., “Dense passage retrieval for open-domain question answering,” in Proc. EMNLP, 2020.

[17] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, “Reciprocal rank fusion outperforms Condorcet and individual rank learning methods,” in Proc. ACM SIGIR Conf., 2009.

[18] T. Schick et al., “Toolformer: Language models can teach themselves to use tools,” in Proc. NeurIPS, 2023.

[19] Y. Qin et al., “ToolLLM: Facilitating large language models to master 16000+ real-world APIs,” in Proc. ICLR, 2024.

[20] X. Liu et al., “AgentBench: Evaluating LLMs as agents,” in Proc. ICLR, 2024.

[21] S. Zhou et al., “WebArena: A realistic web environment for building autonomous agents,” in Proc. ICLR, 2024.

[22] G. Mialon, C. Fourrier, T. Wolf, Y. LeCun, and T. Scialom, “GAIA: A benchmark for general AI assistants,” in Proc. ICLR, 2024.

[23] C. E. Jimenez et al., “SWE-bench: Can language models resolve real-world GitHub issues?” in Proc. ICLR, 2024.

[24] L. Wang et al., “A survey on large language model based autonomous agents,” Front. Comput. Sci., vol. 18, art. 186345, 2024.

[25] G. Mialon et al., “Augmented language models: A survey,” Trans. Mach. Learn. Res., 2023.

[26] S. Yao et al., “ReAct: Synergizing reasoning and acting in language models,” in Proc. ICLR, 2023.

[27] N. Shinn et al., “Reflexion: Language agents with verbal reinforcement learning,” in Proc. NeurIPS, 2023.

[28] A. Zhao et al., “ExpeL: LLM agents are experiential learners,” in Proc. AAAI Conf. Artif. Intell., 2024.

[29] Q. Wu et al., “AutoGen: Enabling next-gen LLM applications via multi-agent conversation,” in Proc. Conf. Lang. Model. (COLM), 2024.

[30] S. Hong et al., “MetaGPT: Meta programming for a multi-agent collaborative framework,” in Proc. ICLR, 2024.

[31] T. Yu et al., “Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task,” in Proc. EMNLP, 2018.

[32] J. Li et al., “Can LLM already serve as a database interface? A big bench for large-scale database grounded text-to-SQLs,” in Proc. NeurIPS Datasets Benchmarks Track, 2023.

[33] M.-A. Lachaux, B. Roziere, L. Chanussot, and G. Lample, “Unsupervised translation of programming languages,” in Proc. NeurIPS, 2020.

[34] W. Yan et al., “CodeTransOcean: A comprehensive multilingual benchmark for code translation,” in Findings Assoc. Comput. Linguistics: EMNLP, 2023.

[35] J. Liu, C. S. Xia, Y. Wang, and L. Zhang, “Is your code generated by ChatGPT really correct? Rigorous evaluation of large language models for code generation,” in Proc. NeurIPS, 2023.

[36] R. Pan et al., “Lost in translation: A study of bugs introduced by large language models while translating code,” in Proc. IEEE/ACM Int. Conf. Softw. Eng. (ICSE), 2024.

[37] X. Hou et al., “Large language models for software engineering: A systematic literature review,” ACM Trans. Softw. Eng. Methodol., vol. 33, no. 8, 2024.

[38] A. Fan et al., “Large language models for software engineering: Survey and open problems,” in Proc. IEEE/ACM Int. Conf. Softw. Eng.: Future Softw. Eng. (ICSE-FoSE), 2023.

[39] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: A method for automatic evaluation of machine translation,” in Proc. ACL, 2002.

[40] M. Popović, “chrF: Character n-gram F-score for automatic MT evaluation,” in Proc. Workshop Stat. Mach. Transl., 2015.

[41] Y. Liu et al., “G-Eval: NLG evaluation using GPT-4 with better human alignment,” in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), 2023.

[42] S. Kim et al., “Prometheus: Inducing fine-grained evaluation capability in language models,” in Proc. ICLR, 2024.

[43] P. Wang et al., “Large language models are not fair evaluators,” in Proc. ACL, 2024.

[44] A. Panickssery, S. R. Bowman, and S. Feng, “LLM evaluators recognize and favor their own generations,” in Proc. NeurIPS, 2024.

[45] O. Grynets, I. Fedorchuk, and V. Lyashkevych, “Leakage-resistant evaluation of AI-generated database specifications in specification-driven migration,” arXiv preprint arXiv:2610.12778, 2026, doi: 10.48550/arXiv.2610.12778.

[46] O. Grynets, D. Kostetskyi, and V. Lyashkevych, “Measuring what a specification determines: A formal semantic-block model and an execution-judged benchmark,” arXiv preprint arXiv:2608.19475, 2026, doi: 10.48550/arXiv.2608.19475.

[47] I. D. Raji, E. M. Bender, A. Paullada, E. Denton, and A. Hanna, “AI and the everything in the whole wide world benchmark,” in Proc. NeurIPS Datasets Benchmarks Track, 2021.

[48] A. Reuel et al., “BetterBench: Assessing AI benchmarks, uncovering issues, and establishing best practices,” in Proc. NeurIPS Datasets Benchmarks Track, 2024.

[49] O. Sainz et al., “NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark,” in Findings Assoc. Comput. Linguistics: EMNLP, 2023.

[50] S. Golchin and M. Surdeanu, “Time travel in LLMs: Tracing data contamination in large language models,” in Proc. ICLR, 2024.

[51] I. Magar and R. Schwartz, “Data contamination: From memorization to exploitation,” in Proc. ACL (Short Papers), 2022.

[52] A. Jacovi, A. Caciularu, O. Goldman, and Y. Goldberg, “Stop uploading test data in plain text: Practical strategies for mitigating data contamination by evaluation benchmarks,” in Proc. EMNLP, 2023.

[53] R. Dror, G. Baumer, S. Shlomov, and R. Reichart, “The hitchhiker’s guide to testing statistical significance in natural language processing,” in Proc. ACL, 2018.

[54] D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, and D. Jurafsky, “With little power comes great responsibility,” in Proc. EMNLP, 2020.

[55] O. E. Gundersen and S. Kjensmo, “State of the art: Reproducibility in artificial intelligence,” in Proc. AAAI Conf. Artif. Intell., 2018.

[56] J. Pineau et al., “Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program),” J. Mach. Learn. Res., vol. 22, 2021.

[57] M. Mitchell et al., “Model cards for model reporting,” in Proc. Conf. Fairness, Accountability, Transparency (FAT\*), 2019.

[58] T. Gebru et al., “Datasheets for datasets,” Commun. ACM, vol. 64, no. 12, pp. 86–92, 2021.

[59] L. Chen, M. Zaharia, and J. Zou, “FrugalGPT: How to use large language models while reducing cost and improving performance,” Trans. Mach. Learn. Res., 2024.

[60] D. Ding et al., “Hybrid LLM: Cost-efficient and quality-aware query routing,” in Proc. ICLR, 2024.

[61] H. T. Kung, F. Luccio, and F. P. Preparata, “On finding the maxima of a set of vectors,” J. ACM, vol. 22, no. 4, pp. 469–476, 1975.

[62] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput., vol. 6, no. 2, pp. 182–197, Apr. 2002.

[63] O. Grynets, D. Babarytskyi, and V. Lyashkevych, “Token optimization strategies for LLM-based Oracle-to-PostgreSQL migration,” arXiv preprint arXiv:2605.28557, 2026, doi: 10.48550/arXiv.2605.28557.

[64] D. Sculley et al., “Hidden technical debt in machine learning systems,” in Proc. NeurIPS, 2015.

[65] E. Breck, S. Cai, E. Nielsen, M. Salib, and D. Sculley, “The ML test score: A rubric for ML production readiness and technical debt reduction,” in Proc. IEEE Int. Conf. Big Data, 2017.

[66] S. Amershi et al., “Software engineering for machine learning: A case study,” in Proc. IEEE/ACM Int. Conf. Softw. Eng.: Softw. Eng. Pract. (ICSE-SEIP), 2019.

[67] T. Rebedea, R. Dinu, M. Sreedhar, C. Parisien, and J. Cohen, “NeMo Guardrails: A toolkit for controllable and safe LLM applications with programmable rails,” in Proc. EMNLP (Syst. Demonstrations), 2023.

[68] O. Khattab et al., “DSPy: Compiling declarative language model calls into self-improving pipelines,” in Proc. ICLR, 2024.

[69] V. Lyashkevych, “An ontology-driven cloud platform for intelligent monitoring with dashboard-centric analytics and domain knowledge integration,” Security Infocommun. Syst. IoT, vol. 4, no. 1, art. 01010, 2026, doi: 10.31861/sisiot2026.1.01010.

[70] V. Lyashkevych, “Intelligent monitoring as an information technology for context-aware decision-making strategy selection,” Ukrainian J. Inf. Technol., vol. 8, no. 1, pp. 39–50, 2026, doi: 10.23939/ujit2026.01.039.

[71] V. Lyashkevych, “Evolution-aware specification-driven synthesis of intelligent monitoring systems by large language models,” Informatics Math. Methods Simulation, vol. 16, no. 3, pp. 424–432, 2026, doi: 10.15276/imms.v16.no3.424.

[72] V. Lyashkevych, “Multi-drift predictive monitoring for evolving information systems,” Comput. Syst. Inf. Technol., no. 2, pp. 171–184, 2026, doi: 10.31891/csit-2026-2-15.

[73] V. Lyashkevych, “Adaptive synthesis of intelligent monitoring models for LLM-modified information systems: A genetic optimisation

approach,” Appl. Comput. Syst., vol. 31, no. 1, pp. 116–126, 2026, doi: 10.2478/acss-2026-0010.

[74] M. Lyashkevych, V. Lyashkevych, and R. Shuvar, “Security and other risks related to LLM-based software development,” Ukrainian J. Inf. Technol., vol. 7, no. 1, pp. 86–96, 2025, doi: 10.23939/ujit2025.01.086.

[75] M. Lyashkevych, V. Lyashkevych, and R. Shuvar, “Definition and formalization of the software functional state concept throughout the development life cycle,” Electron. Inf. Technol., 2025.

[76] V. Kutsan and V. Lyashkevych, “Semantic Kernel usage for orchestration of multi-agent LLM-based systems to solve the tasks which require dynamic involvement of new agents,” in Proc. ISU Conf., 2026.

[77] O. Grynets and V. Lyashkevych, “Unified architecture metamodel of information systems developed by generative AI,” arXiv preprint arXiv:2604.00171, 2026, doi: 10.48550/arXiv.2604.00171.

[78] O. Grynets, O. Pochernina, A. Seletska, D. Tukalo, D. Kostetskyi, I. Fedorchuk, and V. Lyashkevych, “Comparative framework analysis for enterprise generative AI applications: Chatbot, automation, and Oracle-to-PostgreSQL migration,” arXiv preprint arXiv:2609.13577, 2026, doi: 10.48550/arXiv.2609.13577.

[79] T. B. Brown et al., “Language models are few-shot learners,” in Proc. NeurIPS, 2020.

[80] J. Wei et al., “Chain-of-thought prompting elicits reasoning in large language models,” in Proc. NeurIPS, 2022.

[81] O. Olar and M. Liashkevych, “Software ontology subject domain intelligence diagnostics of computer means,” in Proc. 7th IEEE Int. Conf. Intell. Data Acquisition Adv. Comput. Syst. (IDAACS), 2013.

[82] V. Lokazyuk, O. Olar, and V. Lyashkevych, “Software for creating knowledge base of intelligent systems of diagnosing process,” in Advanced Computer Systems and Networks, Lviv, 2009, pp. 140–145.

[83] V. M. Lokazyuk and V. Y. Lyashkevych, “Retrieval information system of diagnostic information of microprocessor device,” Radio-Electron. Comput. Syst., no. 6(18), pp. 103–109, 2006.

[84] V. Lyashkevych and R. Makarchuk, “Features of diagnostic information search for solving tasks of intelligence diagnostics,” Eastern-European J. Enterprise Technol., vol. 6, no. 12, pp. 109–112, 2013.

[85] V. Lyashkevych, “Actuality development searching service for solution tasks of intelligence diagnostic computer means,” in Proc. 6th Int. Conf. Adv. Comput. Syst. Netw. (ACSN), Lviv, 2013.

[86] V. Lyashkevych and O. Olar, “Choice of criteria for setting correspondence between terms in bases of knowledge, Eastern-European J. Enterprise Technol., vol. 6, no. 12, pp. 65–67, 2013.

[87] V. Lyashkevych, O. Olar, and O. Mirosh, “Logic-textual and neuronet approach to search information,” in Proc. IEEE Int. Workshop Intell. Data Acquisition Adv. Comput. Syst. (IDAACS), 2009.

[88] M. Lyashkevych and V. Lyashkevych, “Risks’ attribute values evaluation in software engineering by Monte Carlo simulation,” in Proc. 13th IEEE Int. Conf. Electron. Inf. Technol. (ELIT), 2023.