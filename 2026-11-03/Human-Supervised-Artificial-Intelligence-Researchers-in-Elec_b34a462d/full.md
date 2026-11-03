# Human-Supervised Artificial Intelligence Researchers in Electric Power Systems: Knowledge, Physics, Simulation, and Governance for Large Language Models

Xiangrui Meng Jingcheng Liu Ke Wang Miaopeng Yu Zhengping Lin

China Southern Power Grid Research Institute Co., Ltd. Guangzhou 510663, China

## Abstract

Conclusions in power-system research require a clear evidential basis. Researchers need to understand the data sources, modelling assumptions, and solution procedures to assess the conditions under which a conclusion holds. Large language models (LLMs) can help researchers extract infor mation from documents and translate research requests into code and tool calls, but their outputs still require physical validation. Focusing on human-supervised artificial intelligence (AI) researchers, this review examines power-system LLM applications and related scientific workflows, together with physics-grounded power AI studies, to discuss the conditions for using LLMs in scientific research. In the proposed knowledge–physics–simulation–governance framework, LLMs translate information and coordinate tasks, simulators and validation tools test models, and experts retain authority over critical assumptions, safety boundaries, and final conclusions. Research processes should preserve materials for review so that claims can be traced to their sources, computations can be reproduced, and failed attempts and approvals can be examined. On this basis, a staged development roadmap and requirements for research reporting are proposed. Evaluation should focus on research artifacts and their validation to assess whether LLMs can improve the efficiency of power-system research within clearly defined permissions.

Keywords: Large language models; Electric power systems; Human-supervised artificial intelligence researchers; Artificial intelligence for science; Physics-informed machine learning; Research agents; Retrieval-augmented generation; Trustworthy artificial intelligence

## 1 Introduction

The large-scale integration of renewable generation and energy storage, together with changes in electricity demand and digital control, has made power-system modelling, computation, and result validation more difficult. At the same time, the data, standards, and models used in a study are often scattered across different documents and software systems. Checking a conclusion requires connecting these sources of evidence. Measurements must be consistent with simulation models, while modelling assumptions often come from standards, operating rules, or expert judgement. Researchers therefore need to establish whether the equations, parameters, and software environment are appropriate for the research question.

Artificial intelligence (AI) is widely used for specific power-system tasks. Deep neural networks can forecast electricity demand and renewable generation, estimate operating states from measurements, or identify faults. Graph neural networks (GNNs) emphasize the representation of electrical connections between devices. Reinforcement learning explores control strategies, such as deciding when a battery should charge or discharge within operating limits. Physics-informed learning further incorporates equations and constraints into training. [1–11] These methods generally construct predictors, surrogate models, or controllers for predefined datasets and objectives, with evaluation focused on model performance. A complete research process must also explain how assumptions were formed, how constraints entered the model, and how computational results support their interpretation. Reviews of probabilistic learning and smart-grid AI provide broader accounts of existing tasks and methods. [12–15] Table 1 summarizes the three literature streams used in this review and their roles.

Table 1: Research focus and roles of the three literature streams.
<table><tr><td>Literature stream</td><td>Main research content</td><td>Role in this framework</td></tr><tr><td>Power-system LLM studies</td><td>Review LLM and generative-AI applications in energy and grids [16–18, 29–35]; study modelling, simulation, optimization, digital twins, regulatory compliance, markets, and design [22–27, 36–41]; propose power-agent architectures and controller-design concepts [28, 42]</td><td>Provide direct application evidence to examine which research tasks language models can support and how those tasks connect</td></tr><tr><td>AI-for-science workflows</td><td>Discuss research agents and scientific discovery [43–48]; explore reasoning, hypothesis generation, and preliminary experiments through retrieval, knowledge representation, and tool execution [49–59]</td><td>Provide workflow design references for organizing multistep tasks, handling execution feedback, and arranging human review</td></tr><tr><td>Physics-grounded power AI</td><td>Study state estimation, optimal power flow, and power-model validation through physics-informed learning and topology modelling [1–6]</td><td>Establish validation requirements for equations, network connectivity, constraints, and numerical solution procedures</td></tr></table>

The role of large language models (LLMs) is gradually extending to the organization of scientific work. Researchers can express requirements in natural language, allowing a model to consult documents, generate code, and call simulation or optimization tools. Existing studies have examined this approach in energy systems, including load forecasting and distribution-network dispatch, as well as markets, cybersecurity, and reliability. Digital twins, power electronics, and power-system question answering have also been explored. [16–28] In these studies, LLMs begin to help clarify questions, organize retrieved evidence, and use tool feedback to revise code or interpret results, potentially connecting research steps that were previously relatively separate.

However, introducing LLMs into research also allows errors to propagate through workflows. A fluent answer may still violate circuit laws, overlook equipment limits, or omit post-fault security requirements. Even executable code may solve an optimization problem different from the one intended. Retrieved standards may be inapplicable because they concern a different version, region, or voltage level. If an agent system checks only whether the final task was completed, unreasonable assumptions in intermediate steps can remain hidden. The process through which a language model handles evidence must therefore be examined to determine whether that evidence supports the final power-system conclusion.

In this review, an AI researcher is a system that assists with research tasks within explicit permissions and preserves the evidence underlying the work. Starting from a research request, it can retrieve evidence, formulate candidate assumptions, translate models into scripts, and call simulation tools. After obtaining results, it can help analyze failures, compare alternatives, and prepare a report for expert review. Decisions about modelling choices, safety boundaries, and final conclusions remain with experts. On this basis, this review examines power-system LLM research, considers which workflow designs from artificial intelligence for science (AI-for-science) can be adapted, and draws on physics-grounded power AI to establish the validation requirements those designs must satisfy. Using verifiable research artifacts as the basis for evaluation, it proposes a knowledge-physics-simulation-governance framework and examines the responsibilities and limitations of LLMs at each stage. A staged roadmap and reporting requirements further identify the evidence and controls needed as the scope of agent tasks expands.

## 2 Responsibilities and boundaries of power-system AI researchers

AI-for-science research has begun to involve LLM agents in planning and executing scientific workflows. Related systems have attempted to connect hypothesis generation, experiment or code execution, and result analysis, while recording the process and drafting reports. Tool-augmented reasoning and knowledge graphs have also been used to support these tasks. [44, 45, 48, 50, 52, 53, 55, 56, 60] For power systems, further questions concern how these systems comply with grid physics and who takes responsibility for engineering judgement. PowerAgent has proposed a conceptual architecture and roadmap for the power domain, [28] but their effectiveness still requires validation in specific research tasks. Power-system research may have operational consequences and is subject to industry regulation, so general scientific demonstrations do not establish how much authority an agent should have. Broader reviews discuss scientific uses of foundation models and the design of planning, memory, and tool interfaces, providing references for further research. [47, 61–64]

When applying a general AI research system to power systems, its permissions must first be defined. A complete-looking report does not establish that its conclusions are reliable; the underlying assumptions and models still require domain-specific checks. Circuit equations and equipment limits cannot be relaxed arbitrarily because a problem is difficult to solve, and dynamic models must be appropriate for the disturbance under study. If a workflow uses an invalid operating point, ignores solver warnings, or presents an exploratory run as a conclusion, the study may be invalid even if every step was executed.

The distinction between power flow and dynamic stability analysis illustrates this risk. Power-flow calculations determine steady-state bus voltages and line power flows for given generation, demand, and network connectivity. Dynamic stability analysis examines whether the system can maintain or recover an acceptable state after a disturbance. Power-flow results alone cannot establish post-disturbance dynamic stability. Similar errors include a case whose network connectivity differs from the assumptions, or mixed units and normalization bases. Optimal power flow (OPF) seeks an operating plan that minimizes an objective such as cost subject to network and equipment constraints. Omitting a constraint directly changes the feasible set. Once such errors enter the code, they can propagate through simulation results into reports and engineering judgement.

An AI researcher should act within the scope authorized by researchers. It can organize evidence, prepare model drafts, generate scripts, run solvers, and submit results and failure analyses for review. Researchers define the problem, approve modelling assumptions, and authorize data use. Agents should not independently set safety boundaries, release research claims, or make grid-operating decisions.

These responsibilities must be implemented through system design. Information processing should preserve sources so that parameters and assumptions can be traced to the original materials. Generated models should undergo the corresponding physical checks and simulation tests. During code execution, a sandbox should limit access to data and tools, and steps involving critical judgements should pause for human approval. Responsibility remains with researchers and institutions. Expanding agent autonomy is justified only when the evidence can be reviewed and tool behavior can be controlled.

System maturity also depends on recognizing when execution should not continue. Missing data or unsupported assumptions require additional evidence. An unsuitable tool or an infeasible case requires an explanation and reconsideration of the task. Requests that exceed permissions or involve safety judgements should be referred to a human and, where necessary, refused. These responses help researchers determine which results can withstand scrutiny.

Figure 1 presents the corresponding framework. The LLM translates researchers’ requests into executable tasks and coordinates information exchange between the knowledge, physics, and simulation layers. The knowledge layer supplies modelling evidence, physical models and simulators test candidate solutions, and researchers review critical decisions. Governance spans these activities, limits operational permissions, and preserves the basis for review. The materials retained at each stage together provide the evidence underlying the study, enabling experts to locate the origins of errors without relying solely on the final report.

![](images/5ee20d7db064970a7d092f84087e75c6a29285570978deaec99971a72c5beb91.jpg)  
Figure 1: Knowledge-physics-simulation-governance framework for human-supervised AI researchers in electric power systems.

## 3 Related research and its implications

Starting from the practical needs of power-system research, this review selects power-system LLM studies, AI-for-science workflows, and physics-grounded power AI as its basis for discussion. Table 1 summarizes the content and roles of these three streams. The analysis focuses on how LLMs participate in modelling, tool execution, and result interpretation, and on the conditions under which the methods apply to power systems. Because studies use different tasks, datasets, and test cases, this review compares methods and validation approaches without constructing a unified ranking of model performance or engineering deployment maturity.

The three streams contribute application experience, workflow design methods, and validation requirements to the framework. Power-system LLM studies show how language models enter specific tasks. AI-for-science research offers approaches to organizing multistep work, such as using retrieval and tool feedback to advance a study and recover from execution failures. [65] Establishing whether a generated model follows grid physics requires physics-grounded power AI to identify checks on equations, topology, and numerical solution procedures. Traditional power-AI reviews provide the background of established methods and evaluation. Organizing the literature by these uses helps distinguish the transferability of cross-domain methods from their applicability to power systems.

Existing power-system LLM research covers forecasting, energy management, cybersecurity, markets, digital twins, and question answering. [21, 22, 25, 38, 66–71] In modelling and tool execution, studies have attempted to generate dispatch models or grid cases from natural-language requirements, improve simulation through execution feedback, assess dispatch-regulation compliance through retrievalaugmented generation (RAG), and organize market-agent tasks. [23, 24, 26, 27, 39, 41] In specific applications, building-control studies use LLMs to explain feature-attribution results, [72] reinforcementlearning-assisted optimization [36, 37] and distribution-network reconfiguration [73] focus on generating and improving operating plans, and power-electronics research explores retrieval-assisted design and proposes controller-design agents. [34, 40, 42] Perspectives on grid operation and resilience also discuss directions for further integration. [74–76] These studies inform different aspects of LLM participation in power-system research. Explanatory accuracy, operational feasibility, and the reliability of a complete research process still require separate evaluation.

AI-for-science research offers specific workflow mechanisms for adaptation. Related systems attempt to extract information from literature, use knowledge representations to support hypothesis generation, call tools or execute experiments, and draft reports from the results. [49, 50, 52, 55, 56, 60, 77] Reviews and perspectives further discuss how to organize these operations in scientific and engineering research. [44, 45, 48, 78] Applying these methods in power systems requires additional domain-specific validation and explicit limits on execution permissions. Biomedical and chemistry perspectives provide further design references. [43, 46] Physics-guided foundation models in water science offer a non-LLM example of combining learned and physical models. [79]

Physics-grounded power AI informs validation design even though many of these studies do not use LLMs. State estimation requires correct treatment of measurements and network topology. OPF graph neural networks must represent different node or edge types, while dynamic surrogate models must be tested against disturbance responses. Transient stability assessment and reinforcement learning for grid operation also have their own conditions of applicability. [1–4, 6, 8, 80–90] Together, these studies show that power-system models require task-appropriate tests, including checks on equations and solution procedures and reporting of uncertainty. Stating physical requirements in a prompt cannot replace numerical checks.

Table 1 sets out the research content of the three literature streams, and Figure 2 shows how they jointly support the research framework. Evaluation must further examine the materials retained by a workflow and establish whether they support source tracing, computational reproduction, and review of conclusions.

## 4 Research artifacts and their evaluation

Trustworthy research artifacts are materials and records retained during a study that other researchers can inspect. They should allow others to understand assumptions, rerun computations, and assess whether conclusions hold. Conversations, prompts, and performance tables provide some information, but they must be linked to sources and executable cases and must identify which results have been reviewed. Evaluation of power-system LLMs should therefore examine what the workflow leaves behind and whether those materials help expose errors or refute an original claim.

Research artifacts can be grouped by purpose into evidence artifacts, modelling artifacts, execution records, and governance records. Evidence artifacts retain the original text and context supporting a claim, such as a standard clause or an event report. Modelling artifacts explain how that evidence enters the model, define variables and units, and describe network connectivity, operating assumptions, and uncertainty. Execution records preserve the cases and scripts actually used, together with solver settings and outputs, including convergence logs, residuals, and failed runs. Governance records identify who authorized or approved each step and retain data-access decisions, permission settings, prompt-injection checks, and human interventions.

Review requires connecting these materials. Even a claim supported by the literature must be translated into explicit modelling assumptions before it can be tested numerically. Reproducing an OPF result requires both the formulation and solver settings. Solver logs should be linked to input data to establish whether the computation addressed the original research question, while approval records identify where experts exercised judgement. LLM-assisted computer-aided design (CAD) and design customization demonstrate the use of structured outputs, [91, 92] and research on prompt patterns and non-expert prompting also supports recording how user requirements become outputs. [93, 94] By connecting these materials, researchers can distinguish errors caused by misreading evidence, modelling omissions, execution failures, or governance failures. Table 2 lists the corresponding checks.

![](images/27afd87f18ea16433dbc64efbeaf001631ffc4393fdac16db277f7259c186f01.jpg)  
Figure 2: Contributions of the three literature streams to the research framework. Power-system LLM applications provide direct evidence, cross-domain scientific workflows offer design references, and physics-related research establishes validation requirements. Together, they support the evaluation of research artifacts.

Research artifacts should also facilitate challenge. Reviewers should be able to locate a claim’s source, identify assumptions that could change a result, and select methods that can test the claim. Excluded failed runs and the person who approved the final interpretation should also be recorded explicitly. These details provide a basis for independent review.

Developing from an LLM assistant into an AI researcher means gradually taking on more verifiable research operations. Early assistants primarily explain concepts or draft code. Building on this, a system can extract source-linked evidence and represent assumptions as typed data structures. Connection to simulation tools then enables it to generate cases, read errors, and propose repairs. Planning experiments and comparing multiple runs further requires a complete execution history and negative results. A humansupervised AI researcher coordinates these processes under expert authorization. Its maturity should be assessed by whether the research process can be reconstructed and its conclusions can withstand testing.

The following sections discuss how these materials are produced. The knowledge layer extracts assumptions from original sources, the physics layer translates them into testable models, and the simulation layer executes those models to obtain results. Governance rules span the process and specify permissions and points of human intervention.

Table 2: Categories of research artifacts in LLM-enabled power-system research.
<table><tr><td>Artifact category</td><td>Power-system examples</td><td>Required validation</td><td>Consequences if absent</td></tr><tr><td>Evidence artifacts</td><td>Retain original standard clauses or event reports; link data sources and literature claims to</td><td>Are sources recoverable and versions and access conditions clear? Does their scope support the claim?</td><td>Unsupported statements enter model design</td></tr><tr><td>Modelling artifacts</td><td>their citation context Specify network connectivity and demand assumptions, units and normalization bases, study</td><td>Are variable types and units consistent? Have boundary conditions, applicability, and uncertainty been confirmed?</td><td>Undocumented assumptions change the actual research question</td></tr><tr><td>Execution records</td><td>conditions, and applicable rules Preserve actual cases, scripts, solver settings, and outputs; archive residuals, convergence logs, and failed runs</td><td>Are data and dependency versions available? Are solver tolerances, residuals, and constraint violations consistent</td><td>Computations are hard to rerun or debug and results cannot be tested adequately</td></tr><tr><td>Governance records</td><td>Record data authorization, operational permissions, tool calls, and human interventions; explain rejected assumptions and claim approvals</td><td>with execution status? Are approvers identified? Are credentials and sensitive data controlled, and have prompt-injection checks been performed?</td><td>Responsibility within the workflow is unclear</td></tr></table>

## 5 Evidence extraction and assumption organization in the knowledge layer

Many modelling choices originate in documents. Equipment manuals specify ratings and operating limits; grid codes and operating procedures establish applicable conditions; and protection-setting documents and market rules influence the definition of specific research questions. Incident reports, project documents, and existing papers or patents may also support assumptions. If these materials appear only as background text, readers cannot determine how model parameters and boundaries were established. LLMs can help extract requirements and claims, link them to structured modelling artifacts, and retain their authoritative sources.

Materials-chemistry research has attempted to extract entities and relations from scientific text, [60] while preliminary workflows for scientific facilities combine document retrieval with tool use. [49] These methods can inform the design of a power-system knowledge layer, but their domain applicability still requires validation. For example, after identifying a generator or line in equipment documents, a system can record its connections and operating limits. Events extracted from fault reports can be linked to the corresponding responses. The resulting candidate assumptions must specify parameter meanings, model boundaries, and uncertainty, with information types and sources recorded. Domain-specific language representations such as MatSciBERT help identify technical terms, [95] while knowledge graphs and corpus platforms can organize extraction results. [96–98] Structuring information does not remove the need to assess whether the assumptions are valid in engineering terms.

Preserving a source also requires retaining the conditions that determine its applicability. A standard clause must identify its version and applicable region, and a requirement concerning a voltage level must remain within its original scope. Parameters should be recorded with units, base values, and measurement conditions. A paper’s claim should be identified as arising from an experiment, simulation, review, or speculation. If documents conflict because they concern different equipment classes or periods, the respective conditions should be retained. Forcing them into one definite answer hides ambiguity in the original evidence.

The authority of a source also determines which judgements it can support. A grid code or operating procedure may establish a mandatory boundary, a manufacturer’s manual may specify equipment-use conditions, and a peer-reviewed simulation result holds only under its associated assumptions. An event report records observations that occurred, whereas informal design notes may only propose ideas for testing. Feeding these materials to a model as equivalent text fragments can confuse normative requirements with research assumptions. A RAG workflow should retain these distinctions and explain which modelling operations each source can support.

Retrieved information in power-system RAG should be able to affect subsequent modelling. After retrieving a requirement, the system must check its applicability to the current problem and use it to define variable meanings, parameter ranges, or constraints, adding validation checks where necessary. Retrieval can then both substantiate an existing claim and expose judgements that still require testing. Question answering using domain-specific application programming interfaces (APIs), manufacturing retrieval, enterprise benchmarks, and energy chatbots demonstrate ways to supply task context, [99–102] but whether the information can become an engineering constraint depends on verification of its scope.

Dynamic knowledge graphs have been used to coordinate distributed laboratory workflows and record data provenance, [103] providing a design precedent for evidence organization in power-system research. A knowledge graph can represent relations among equipment, events, and constraints and link them to clauses, datasets, or experiments. Retrieval indexes retain original passages for review, while structured databases store cases, tool calls, and solver settings. LLMs can help query and translate information between these systems. Each transformation should record its sources and any uncertainty it may introduce. The corresponding storage systems retain the evidence so it can be examined independently of the language model’s answers.

Knowledge-graph RAG may fail in contextual understanding, relation mapping, or disambiguation. [104] In power systems, a retrieved document may be outdated or apply to a different region or voltage level. Unit-conversion errors and misread topology relations can also affect model construction. If a provisional inference is passed to a simulator as fact, even numerically normal results may rest on a false premise. Literature-synthesis tools, battery-recycling document agents, and PowerGridQA provide settings for evaluating document processing and question answering, [71, 105, 106] but these evaluations cannot replace checks on executable grid models. Further work must establish whether retrieved information remains valid after it becomes assumptions and constraints.

This raises reliability questions along the path from retrieval to modelling: whether different names for the same device can be matched, whether a clause’s version and scope can be determined, and whether textual conditions and exceptions can be translated fully into model constraints. Checking only whether an answer cites the correct document is unlikely to expose omissions in the transformation. The original text, extracted assumptions, and actual case must be examined together. When sources conflict or conditions are incomplete, the system should preserve ambiguity and request additional information instead of directly generating a unique parameter value. Once an error reaches simulation, distinguishing retrieval errors, assumption-transformation errors, and solver-setting errors also affects repair. Parameter changes must return to their supporting evidence, and changes to research boundaries require renewed review. Evaluation should therefore cover these transformations and test whether a system can locate errors and stop when evidence is insufficient.

## 6 Modelling and validation in the physics layer

A power-system model must satisfy the physical requirements of the question under study. Currents and voltages obey circuit laws, transmitted power is limited by equipment ratings, and disturbance responses depend on dynamic models. An LLM can assist in proposing a candidate model, but accepting it requires the corresponding physical and numerical tests.

Physics-grounded power AI shows that different models require different checks. Physics-informed learning incorporates differential equations, conservation laws, or boundary conditions into training, while topology-aware graph models emphasize electrical connectivity. OPF surrogates require tests of the feasibility of returned solutions. Dynamic surrogates must be tested at different time scales and under different disturbances. State estimation must establish whether measurements are sufficient, explain dependence on topology, and report residuals and uncertainty. [1–3, 6, 81, 85, 86, 90, 107] Reviews of inertia estimation, residual dynamic-parameter models, and Bayesian physics-informed identification further show how model choice and parameter uncertainty affect interpretation. [108–111] Mathematical models generated by LLMs should undergo these checks as well. Physical prompts in the generation process do not establish model validity.

During modelling, an LLM can translate a natural-language problem into variables and constraints, help draft a power-flow or OPF formulation, generate code, and call candidate datasets and solvers. Whether its output is usable depends on subsequent testing. A power-flow solution requires numerical checks, and a stability conclusion requires an appropriate dynamic model. Solver warnings and safety-related interpretations must also be reviewed. Validation must be able to reject generated results; otherwise errors in the model cannot be corrected effectively.

Validation should proceed alongside model execution. Before a run, inputs should be checked for consistency, particularly units and normalization bases, network connectivity, and parameter ranges, with timestamps and data sources confirmed. During execution, code and tool calls should be checked against the intended equations and constraints. After execution, task-specific checks should cover convergence, residuals, and limit violations. Dynamic or protection studies also require analysis of stability margins and protection actions and the handling of solver warnings. Counterfactual checks further change operating conditions, for example by introducing faults, extreme weather, or missing data, to examine whether conclusions remain supported. Market studies can similarly vary market conditions. Learned dynamic components, including PINNSim, plug-and-play simulation, and physics-informed network variants, also require their coupling assumptions and simulation conditions to be specified. [112–115]

Choosing a validator is itself part of modelling judgement. For example, deterministic OPF alone cannot establish resilience to rare events, and interpreting a cyber event requires an explicit threat model. Learning-based OPF and solver warm-start studies address numerical tasks different from those of graph reconfiguration or load-margin prediction, so their checks are not interchangeable. [116–120] Researchers should first identify the operational or economic question behind a claim, determine whether it involves protection actions or interactions between information and physical systems, and then choose suitable validation methods. This prevents validation from drifting away from the original research question.

Each research artifact should state the grounds for accepting or rejecting it. Power-flow studies must check circuit equations, voltage and equipment limits, and report solution accuracy. In addition to checking the returned solution, OPF studies must verify that the objective and constraints are complete. Dynamic studies should describe the disturbance and control assumptions and identify the criteria used to evaluate the response. Checks should be selected for the task, with reasons given when a check does not apply. Solver success establishes only part of what is required; it cannot replace an assessment of whether the model matches the research question.

Validation and interpretation in protection, market, and cybersecurity research also depend on their respective task conditions. Protection studies must identify the devices that act after a fault and the timing of those actions. Market analysis must specify trading and settlement rules. Cybersecurity studies must define what an attacker can access or alter and distinguish attacks on the language-model interface from attacks on grid measurements and control systems. In interpreting results, stability judgements require evidence such as response curves or protection actions. Forecasting analysis should distinguish the effects of weather and calendar factors from those of the model itself, while security analysis should distinguish observed signs from inferences about attacker intent. Studies of state estimation and its uncertainty [121– 123], a review of frequency control [124], and microgrid-control research under communication failures [125] all indicate that measurement conditions, time scales, and communication assumptions affect interpretation. Uncertainty must therefore be explained in relation to applicable conditions, with observations, inferences, and judgements distinguished.

## 7 Execution and experimental feedback in the simulation layer

Simulation tools should be selected for the research question. Steady-state power flow and OPF can use the corresponding network-computation tools, whereas distribution-network analysis or dynamic-response studies require simulators suited to their model characteristics. Optimization tasks also need modelling tools and solvers that execute the objectives and constraints correctly. A workflow should record why a tool was selected, describe the model it actually solves, and explain how outputs are checked. A software name identifies the execution environment but does not establish that the original research question has been tested.

After code generation, execution feedback gives a system the opportunity to detect and correct errors. A tool run provides solution status and result files and may also return warnings, residuals, or infeasibility diagnostics. Software-engineering reviews and scientific-programming studies offer methods for checking generated code, [126–130] but executable code must also be checked against the intended equations and numerical experiments. With simulation tools connected, an agent can run a case, use feedback to identify inconsistencies, propose changes, and rerun the computation. Assumption changes should follow the established approval rules, and repairs should be recorded fully so that subsequent experiments can explain why results changed.

Transformations from information sources to simulation results should be reconstructable. After information extraction from a research request and its source documents, typed assumptions should be formed and then translated into model constraints and tool inputs. When simulation or optimization has run, solver logs and failure records should be retained with the results for human review. Abandoned cases and negative results must be preserved alongside successful runs because they may reveal conditions under which assumptions fail or tools reach their limits of applicability.

Existing power-system studies have demonstrated several parts of this process. Automated dispatch for active distribution networks extracts information from natural-language requests, formulates a problem, and generates code. [23] Tool-augmented grid-model generation allows users to construct cases through natural language. [26] Feedback-driven simulation agents address DALINE and MATPOWER tasks by using retrieval and reasoning to organize tool execution and adjusting operations in response to errors. [24] RAG-based assessment of dispatch regulations emphasizes compliance checking and human review, [27] while market-agent research uses modelling and simulation to generate bidding strategies. [39, 41] Organizing these components into a scientific workflow still requires explicit accounts of how intermediate assumptions were approved and which checks each step passed. Task-completion records can support research judgement only when linked to this evidence.

Failed runs help identify the limits of a workflow. For example, power-flow non-convergence means that a numerical procedure failed to find a solution within its stopping criteria; it does not directly imply physical impossibility. OPF infeasibility concerns whether the stated constraints can be satisfied simultaneously, and solver diagnostics must be interpreted in the specific circumstances. Inconsistent topology, missing data, or unit errors can cause failures. Software-version differences and omitted constraints can also change results. Preserving these records allows researchers to determine whether a problem originates in assumptions, tools, or execution and where human intervention is required. Reporting only successful runs overestimates agent reliability and obscures a model’s valid scope for subsequent research.

Failure classification should guide repair decisions. Syntax errors can usually undergo an initial attempt at automatic repair. Invalid cases require checks on inputs and their assumptions, whereas infeasible models and numerical non-convergence need separate diagnosis. Outputs that violate physical laws require a return to the model and its constraints. Unsupported assumptions require additional evidence, while unsafe tool requests should be refused or referred to a human. Different categories permit different changes. Root-cause analysis in cloud systems and industrial troubleshooting provide references for diagnostic design, [131, 132] and variable extraction from scientific-software documentation can support metamorphic testing. [133] Recording error types and responses helps compare systematically the conditions under which an agent can recover.

A reproducibility package should allow other researchers to reconstruct how a study was executed and see what it actually validated. Recursive reasoning and refinement, constitutive-model discovery, and the experimentally tested Virtual Lab provide different levels of evidence. [51, 54, 57] For power-system simulation, the package should preserve input data and cases, specify parameters, unit conventions, and tool versions, link prompts or task templates to the actual commands or API calls, and record random seeds. Result files should be archived with warnings, failed attempts, and human interventions. Reviewers can then determine which models and solver settings a conclusion depends on, and where prompt changes or human corrections affected the results.

## 8 Collaboration between multimodal evidence and specialist models

Power-system research combines evidence in different forms, including text, measurements, network structures, and images. Demand and generation measurements must be linked to equipment, and event logs must be interpreted with the network structure. Images and simulation outputs may also address the same question. The first step is therefore to establish that these materials refer to the same equipment, time interval, and operating conditions, and to check their assumptions. [134] Once aligned, different data types still require suitable specialist models.

Time-series foundation models and forecasting methods with LLM-like architectures have been applied to load, wind power, building energy use, and multienergy tasks. [20, 67–70, 135, 136] They are candidate methods when samples are limited, local historical data are unavailable, or transfer across settings is needed. Forecasts remain affected by weather, energy-use behavior, grid connectivity, control policies, and market incentives. Good transfer performance does not establish that a model has captured these mechanisms, so forecast provenance and uncertainty must still be specified when forecasts enter simulation or decision analysis. Reviews of renewable-energy, photovoltaic, and Transformer forecasting provide background on baseline methods, [137–139] including clustering, recurrent-convolutional models, and combinations of fuzzy time series. [140–142] Research on temporal and graph Transformers, seasonal-trend models, and feature selection for prosumers further indicates the need to specify forecast horizons, covariates, and evaluation-data splits when comparing results. [143–148] Some LLM forecasting methods also incorporate external factors to estimate generation or demand and explore zero-shot transfer and parameter-efficient adaptation. [149–152] Their forecast quality likewise requires evaluation under the relevant task and data conditions.

For network structures, graph models retain electrical connections and line attributes. GNNs and related methods use these structures for state estimation, stability analysis, or risk assessment. [4, 82, 83, 85, 153– 155] State estimation, for example, infers operating quantities such as bus voltages from measurements. Measurement sufficiency is directly related to network connectivity, and a textual grid description alone can lose relations that determine observability and electrical feasibility. Other observations have their own representation requirements. Synchronized electrical waveforms require sampling, time synchronization, and channel-to-device mappings to be specified. Logs and alarms must be linked to the correct events, and inspection images must identify the observed object. Existing studies use vision foundation models for equipment inspection and image segmentation, [156] photovoltaic graph models for spatiotemporal relations between sites, [157] knowledge graphs for topology identification, [158] and synchronizedwaveform foundation models for electrical time series. [159] These outputs can jointly explain the same phenomenon only when their objects, times, and operating conditions match.

Specialist models need to collaborate within the same research task. An LLM can coordinate their use according to research needs and organize their outputs into an explanation or report. Time-series models supply forecasts, graph models represent network connectivity, physical simulation or constrained optimization tests candidate feasibility, and knowledge graphs and retrieval systems preserve queryable evidence. Digital-twin modelling and energy-system reviews provide precedents for component integration, [160–165] while multimodal knowledge-graph research in scientific discovery offers further references. [166] A modular organization should retain each model’s own validation rules, uncertainty statements, and refusal conditions. Even when a waveform, outage report, and simulation trajectory concern the same event, they have different noise sources, update frequencies, and evidentiary scope. If fusion retains only a single embedding and discards modality-specific checks, those differences become difficult to identify in subsequent interpretation.

This collaboration also requires connecting how the different materials are used. Research questions must be linked to retrieval results and approved assumptions to actual tool runs, while rejected assumptions and failed attempts are retained. Specialist-model interfaces and output provenance should be independently inspectable. The propagation of uncertainty into subsequent computations and expert approval of interpretations should be recorded with the relevant results. When modalities disagree, researchers can then trace the difference to data, assumptions, or models, preventing an LLM from combining incompatible evidence into an apparently consistent conclusion.

## 9 Governance requirements and development roadmap

Governance rules determine what an agent can do during research and how errors can be traced. Retrieving documents, executing code, and changing assumptions require different permissions, which must be distinguished in system design. Each information-processing operation or tool call should leave an inspectable record. Failures, approvals, and uncertainty statements should be linked to the corresponding results. Researchers can then determine which operations were authorized and which changes affected the final claim.

After diagnosis and validation, failure records can also support conclusions with explicit conditions. Non-convergence or constraint violations, for example, can inform analysis of the limitations of a model configuration, but cannot justify unverified feasibility or safety judgements. Incomplete assumptions or unclear applicability require corresponding limits on the conclusions. Explanations of failures and changes to critical assumptions require human review, and researchers should decide whether final conclusions are acceptable.

Governance must address LLM-system risks and grid operational-technology (OT) threats separately. Research on LLM integration discusses security risks and potential safeguards, [18, 19, 167] while other studies examine attacks generated by non-LLM models such as generative adversarial networks (GANs) and autoencoders, [168] or vulnerabilities in grid machine learning and state estimation themselves. [169, 170] LLMs, multimodal models, and other generative models have also been used for anomaly detection, [171–173] but performance in detecting grid threats does not establish the security of a surrounding agent. The AI Scientist demonstrates a research system’s ability to generate and execute code, [52] making execution permissions a separate control requirement. This review recommends task-specific permissions and sandboxed code execution. Credentials should be protected and sensitive data isolated or anonymized. Review should examine prompt injection and data leakage, check the threat assumptions used in the power-system study, and establish that restricted operations can be revoked when necessary.

Some studies indicate that research quality also depends on whether explanations are faithful to the evidence and whether provenance is complete. [174–176] Governance measures can accordingly be matched to operational risk. Formatting a report or extracting candidate variables can be automated while retaining logs. Determining model boundaries, authorizing sensitive-data access, interpreting protection failures, and deciding which conclusions can be published require review by the people responsible for those decisions. Approval points and audit records should identify the decision makers for subsequent examination.

Table 3 divides capability development into five stages. The earlier stages are already observable in existing applications: LLM assistants explain concepts and draft code, retrieval-based question answering attaches references or source links to answers, and tool-connected agents execute code and adjust operations from runtime feedback. Source-linked answers provide a starting point for traceability but still require citation checking and explicit assumptions. Grid-model generation and feedback-driven simulation demonstrate concrete applications of tool execution. [24, 26] The final two stages concern closed-loop experiments and collaboration across research tasks. Related systems have begun exploring these capabilities, [52, 54, 177] but further research is needed to establish how they can sustain experimental planning, handle failures, and support expert judgement in power systems. Capability expansion should proceed together with validation and permission management, preserving reviewable outputs and decision processes and identifying who approves and takes responsibility for critical operations.

Table 3: Development roadmap for human-supervised AI researchers in power systems.
<table><tr><td>Stage</td><td>Main capability</td><td>Trustworthy research artifacts</td><td>Governance measures</td></tr><tr><td>LLM assistant</td><td>Explain concepts from literature and draft code or text for human use</td><td>Notes with sources and uncertainty statements, and draft code</td><td>Human checking before use</td></tr><tr><td>Traceable artifact builder</td><td>Extract source-linked evidence and represent assumptions as typed data structures</td><td>Source-supported claims and variable definitions with types and units</td><td>Citation, schema, and unit checks</td></tr><tr><td>Simulation- connected agent</td><td>Run generated cases and propose repairs from tool errors</td><td>Executable cases with corresponding solver, error, and validation records</td><td>Sandboxed execution and physical validators</td></tr><tr><td>Closed-loop research agent</td><td>Repeat simulations according to an experimental plan, compare results, and record negative cases</td><td>Experimental plans and complete run histories, including failures and comparison reports</td><td>Human approval of critical experimental decisions and research claims</td></tr><tr><td>Human- supervised AI researcher</td><td>Coordinate research tasks under expert authorization and use evidence and execution results in reporting</td><td>Research packages with sources, decision processes, and approval records</td><td>Expert decision authority, accountability, and institutional review</td></tr></table>

Research reporting can follow the evidence artifacts, modelling artifacts, execution records, and governance records in Table 2, explaining the evidence, computation, and review supporting each conclusion. Reports should distinguish candidate claims proposed by an LLM from conclusions that have undergone physical, simulation, experimental, or expert review. They should provide the model and retrieval configurations, prompts, and tool versions needed for reproduction, together with scripts and logs, and describe failed attempts and human corrections. Where licensing or security conditions prevent full release of data or models, reports can provide data structures, conditions of applicability, and logs and execution descriptions available for controlled review. Research efficiency must be evaluated alongside the cost of checking. Given the electricity required for LLM training and inference, [178] reports of task completion should also state the number of model calls, computational resource use, and human correction effort.

Whether LLMs improve power-system research must be established through comparative experiments on specific tasks. Tasks with sufficiently complete documents, data, and simulation interfaces can be used to compare results and workload with and without LLMs, recording computational resource use and human verification time and identifying the stages responsible for any efficiency gains. Experiments can deliberately introduce outdated clauses, unit errors, or missing measurements to observe whether the system detects problems and to trace how a retrieved item changes parameters, constraints, and final conclusions. When execution fails, evaluation should check whether repairs work, whether they change unapproved assumptions, and whether restricted tool requests can be blocked. Multimodal tasks should examine how conflicting evidence is handled and how uncertainty propagates into subsequent computations. Results can then identify which operations are suitable for automation and which changes require researchers to reconsider their judgement.

## 10 Conclusions

LLMs may reduce repetitive work in power-system research, including consulting literature, organizing assumptions, writing code, and coordinating simulations. This review examines power-system LLM applications, related scientific workflows, and physics-grounded power AI and proposes a knowledge-physicssimulation-governance framework. Evaluation should consider the retained research materials and actual validation process to determine whether a system helps researchers review conclusions, analyze failures, and revise judgements, alongside answer quality and task-completion speed.

Within this framework, LLMs translate information and coordinate tasks. Researchers retain authority over problem definition, data authorization, and critical assumptions and remain responsible for safety boundaries and final conclusions. Generated models should undergo task-appropriate equation and constraint checks and be tested by corresponding solvers or simulators. Sources, conditions of applicability, solver logs, negative results, and approval records should be retained in a research package so that efficiency gains rest on reviewable computations and judgements. The staged development roadmap and reporting requirements offer references for gradually expanding task scope and comparing different systems.

This review selects literature to develop a framework and does not provide an exhaustive systematic review or meta-analysis. Existing tasks and benchmarks are fragmented, and evidence from real engineering deployment remains limited. The proposed workflows and evaluation requirements therefore require testing through shared benchmarks, comparative experiments, and independent reproduction. Further research should identify the conditions under which different methods work, when expert intervention is necessary, and whether computational and human verification costs improve in practice.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

No grid-operating data were generated in this study. This review is based on published literature and locally curated bibliographic records.

## References

[1] Q.-H. Ngo, B. L. Nguyen, T. V. Vu, J. Zhang, T. Ngo, Physics-informed graphical neural network for power system state estimation, Applied Energy 358 (2024) 122602. doi:10.1016/j.apenergy.2023.122602.

[2] S. Falas, M. Asprou, C. Konstantinou, M. K. Michael, Robust Power System State Estimation Using Physics-Informed Neural Networks, IEEE Transactions on Industrial Informatics 21 (10) (2025) 8057–8067. doi: 10.1109/tii.2025.3582293.

[3] T. B. Lopez-Garcia, J. A. Domínguez-Navarro, Optimal Power Flow With Physics-Informed Typed Graph Neural Networks, IEEE Transactions on Power Systems 40 (1) (2025) 381–393. doi:10.1109/tpwrs. 2024.3394371.

[4] W. Liao, B. Bak-Jensen, J. Radhakrishna Pillai, Y. Wang, Y. Wang, A Review of Graph Neural Networks and Their Applications in Power Systems, Journal of Modern Power Systems and Clean Energy 10 (2) (2022) 345–360. doi:10.35833/mpce.2021.000058.

[5] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physics-informed machine learning, Nature Reviews Physics 3 (6) (2021) 422–440. doi:10.1038/s42254-021-00314-5.

[6] B. Huang, J. Wang, Applications of Physics-Informed Neural Networks in Power Systems - A Review, IEEE Transactions on Power Systems 38 (1) (2023) 572–588. doi:10.1109/tpwrs.2022.3162473.

[7] A. Perera, P. Kamalaruban, Applications of reinforcement learning in energy systems, Renewable and Sustainable Energy Reviews 137 (2021) 110618. doi:10.1016/j.rser.2020.110618.

[8] Y. Li, C. Yu, M. Shahidehpour, T. Yang, Z. Zeng, T. Chai, Deep Reinforcement Learning for Smart Grid Operations: Algorithms, Applications, and Prospects, Proceedings of the IEEE 111 (9) (2023) 1055–1096. doi:10.1109/jproc.2023.3303358.

[9] M. Khodayar, G. Liu, J. Wang, M. E. Khodayar, Deep learning in power systems research: A review, CSEE Journal of Power and Energy Systems 7 (2) (2021) 209–220. doi:10.17775/cseejpes.2020.02700.

[10] Y. Eren, <sup>˙</sup>I. Küçükdemiral, A comprehensive review on deep learning approaches for short-term load forecasting, Renewable and Sustainable Energy Reviews 189 (2024) 114031. doi:10.1016/j.rser.2023. 114031.

[11] B. She, F. Li, H. Cui, J. Zhang, R. Bo, Fusion of Microgrid Control With Model-Free Reinforcement Learning: Review and Vision, IEEE Transactions on Smart Grid 14 (4) (2023) 3232–3245. doi:10.1109/tsg. 2022.3222323.

[12] T. Ahmad, R. Madonski, D. Zhang, C. Huang, A. Mujeeb, Data-driven probabilistic machine learning in sustainable smart energy/smart energy systems: Key developments, challenges, and future research opportunities in the context of smart grid paradigm, Renewable and Sustainable Energy Reviews 160 (2022) 112128. doi:10.1016/j.rser.2022.112128.

[13] Y. M. Banad, S. S. Sharif, Z. Rezaei, Artificial intelligence and machine learning for smart grids: from foun dational paradigms to emerging technologies with digital twin and large language model-driven intelligence, Energy Conversion and Management: X 28 (2025) 101329. doi:10.1016/j.ecmx.2025.101329.

[14] I. Alhamrouni, N. H. Abdul Kahar, M. Salem, M. Swadi, Y. Zahroui, D. J. Kadhim, F. A. Mohamed, M. Alhuyi Nazari, A Comprehensive Review on the Role of Artificial Intelligence in Power System Stability, Control, and Protection: Insights and Future Directions, Applied Sciences 14 (14) (2024) 6214. doi:10.3390/app14146214.

[15] O. A. Alimi, K. Ouahada, A. M. Abu-Mahfouz, A Review of Machine Learning Approaches to Power System Security and Stability, IEEE Access 8 (2020) 113512–113531. doi:10.1109/access.2020.3003568.

[16] U. Cali, U. Halden, M. Andoni, F. O. Catak, S. Chen, B. Couraud, E. Kantar, S. Knapper, I. Kucukdemiral, H. Kusetogullari, M. Kuzlu, Y. Mousavi, S. Norbu, T. S. Ustun, D. Flynn, Generative AI and LLM applications in renewable energy and smart grids: a systematic review for the sustainable energy transition, Artificial Intelligence Review 59 (7) (2026) 153. doi:10.1007/s10462-026-11545-2.

[17] H. Mirshekali, M. R. Shadi, F. Ghanadi Ladani, H. R. Shaker, A Review of Large Language Models for Energy Systems: Applications, Challenges, and Future Prospects, IEEE Access 13 (2025) 163162–163188. doi:10.1109/access.2025.3610994.

[18] N. Ibrahim, R. Kashef, Exploring the emerging role of large language models in smart grid cybersecurity: a survey of attacks, detection mechanisms, and mitigation strategies, Frontiers in Energy Research 13 (2025) 1531655. doi:10.3389/fenrg.2025.1531655.

[19] J. Ruan, G. Liang, H. Zhao, G. Liu, X. Sun, J. Qiu, Z. Xu, F. Wen, Z. Y. Dong, Applying Large Language Models to Power Systems: Potential Security Threats, IEEE Transactions on Smart Grid 15 (3) (2024) 3333– 3336. doi:10.1109/tsg.2024.3373256.

[20] W. Liao, S. Wang, D. Yang, Z. Yang, J. Fang, C. Rehtanz, F. Porté-Agel, TimeGPT in load forecasting: A large time series model perspective, Applied Energy 379 (2025) 124973. doi:10.1016/j.apenergy. 2024.124973.

[21] Y. Shang, W.-L. Shang, D. Cui, P. Liu, H. Chen, D. Zhang, R. Zhang, C. Xu, Y. Liu, C. Wang, M. Alhazmi, Spatio-temporal data fusion framework based on large language model for enhanced prediction of electric vehicle charging demand in smart grid management, Information Fusion 126 (2026) 103692. doi:10.1016/ j.inffus.2025.103692.

[22] X. Yang, C. Lin, H. Liu, W. Wu, RL2: Reinforce Large Language Model to Assist Safe Reinforcement Learning for Energy Management of Active Distribution Networks, IEEE Transactions on Smart Grid 16 (4) (2025) 3419–3431. doi:10.1109/tsg.2025.3568226.

[23] X. Yang, C. Lin, Y. Yang, Q. Wang, H. Liu, H. Hua, W. Wu, Large Language Model Powered Automated Modeling and Optimization of Active Distribution Network Dispatch Problems, IEEE Transactions on Smart Grid 17 (2) (2026) 952–965. doi:10.1109/tsg.2025.3621438.

[24] M. Jia, Z. Cui, G. Hug, Enhancing LLMs for Power System Simulations: A Feedback-Driven Multi-

Agent Framework, IEEE Transactions on Smart Grid 16 (6) (2025) 5556–5572. doi:10.1109/tsg.2025. 3589114.

[25] S. Ieva, D. Loconte, G. Loseto, M. Ruta, F. Scioscia, D. Marche, M. Notarnicola, A Retrieval-Augmented Generation Approach for Data-Driven Energy Infrastructure Digital Twins, Smart Cities 7 (6) (2024) 3095– 3120. doi:10.3390/smartcities7060121.

[26] K. Deng, Y. Zhou, H. Zeng, Z. Wang, Q. Guo, Power Grid Model Generation Based on the Tool-Augmented Large Language Model, IEEE Transactions on Power Systems 40 (6) (2025) 5487–5490. doi:10.1109/ tpwrs.2025.3609049.

[27] Y. Cheng, H. Zhao, D. Xiang, Z. Zhang, G. Liu, Y. Liu, J. Zhao, X. Cai, Power system operational reliability evaluation with retrieval-augmented generation enhanced large language model, Energy and AI 24 (2026) 100688. doi:10.1016/j.egyai.2026.100688.

[28] Q. Zhang, L. Xie, PowerAgent: A Road Map Toward Agentic Intelligence in Power Systems: Foundation Model, Model Context Protocol, and Workflow, IEEE Power and Energy Magazine 23 (5) (2025) 93–101. doi:10.1109/mpe.2025.3579718.

[29] C. Zhang, J. Zhang, J. Lu, Y. Zhao, Large Language Models Meet Energy Systems: Opportunities, Challenges, and Future Perspectives, Applied Energy 403 (2026) 127076. doi:10.1016/j.apenergy.2025. 127076.

[30] S. Madani, A. Tavasoli, Z. Khoshtarash Astaneh, P.-O. Pineau, Large Language Models Integration in Smart Grids, Energy Reports 14 (2025) 1562–1577. doi:10.1016/j.egyr.2025.06.051.

[31] H. Shi, L. Fang, X. Chen, C. Gu, K. Ma, X. Zhang, Z. Zhang, J. Gu, E. G. Lim, Review of the opportunities and challenges to accelerate mass-scale application of smart grids with large-language models, IET Smart Grid 7 (6) (2024) 737–759. doi:10.1049/stg2.12191.

[32] G. Antonesi, T. Cioara, I. Anghel, V. Michalakopoulos, E. Sarmas, L. Toderean, A systematic review of transformers and large language models in the energy sector: towards agentic digital twins, Applied Energy 401 (2025) 126670. doi:10.1016/j.apenergy.2025.126670.

[33] Erdiwansyah, R. Mamat, Syafrizal, M. F. Ghazali, F. Basrawi, S. Rosdi, Emerging role of generative AI in renewable energy forecasting and system optimization, Sustainable Chemistry for Climate Action 7 (2025) 100099. doi:10.1016/j.scca.2025.100099.

[34] K. A. Ibrahim, P. C.-K. Luk, Z. Luo, S. Y. Ng, L. Harrison, Revolutionizing power electronics design through large language models: Applications and future directions, Computers and Electrical Engineering 123 (2025) 110248. doi:10.1016/j.compeleceng.2025.110248.

[35] M. Ghafari, A. Sami, H. Rezapour, S. Saravani Ghayour, F. Lin, K. Basaran, G. Cristian Lazaroiu, P. Siano, A Comprehensive Review on the Application of Large Language Models (LLMs) in Power Systems, IEEE Access 13 (2025) 209450–209486. doi:10.1109/access.2025.3637226.

[36] Z. Yan, Y. Xu, Real-Time Optimal Power Flow With Linguistic Stipulations: Integrating GPT-Agent and Deep Reinforcement Learning, IEEE Transactions on Power Systems 39 (2) (2024) 4747–4750. doi:10. 1109/tpwrs.2023.3338961.

[37] L. Zhang, D. Yue, G. P. Hancke, C. Dou, L. Yu, Z. Chen, Optimization of Energy and Carbon Emissions in Integrated Energy System Based on Deep Reinforcement Learning Assisted by Large Language Model, IEEE Transactions on Industrial Informatics 21 (10) (2025) 8186–8197. doi:10.1109/tii.2025.3578123.

[38] X. Lu, J. Qiu, Y. Yang, C. Zhang, J. Lin, S. An, Large Language Model-Based Bidding Behavior Agent and Market Sentiment Agent-Assisted Electricity Price Prediction, IEEE Transactions on Energy Markets, Policy and Regulation 3 (2) (2025) 223–235. doi:10.1109/tempr.2024.3518624.

[39] Y. Cheng, W. Liu, Y. Xue, J. Huang, J. Zhao, F. Wen, Leveraging Large Language Model Based Agent for Automated Electricity Market Modelling and Simulation, Journal of Modern Power Systems and Clean Energy 14 (1) (2026) 50–62. doi:10.35833/mpce.2025.000639.

[40] F. Lin, X. Li, W. Lei, J. J. Rodriguez-Andina, J. M. Guerrero, C. Wen, X. Zhang, H. Ma, PE-GPT: A New Paradigm for Power Electronics Design, IEEE Transactions on Industrial Electronics 72 (4) (2025) 3778– 3791. doi:10.1109/tie.2024.3454408

[41] R. Zou, X. Zhou, Y. Cheng, W. Liu, X. Wang, J. Zhao, X. Cai, A Large Language Model-Based Agent for Automated Bidding Strategy Generation in Electricity Markets, in: 2025 IEEE International Conference on Power and Integrated Energy Systems (ICPIES), IEEE, 2025, pp. 475–480. doi:10.1109/icpies65420. 2025.11070004.

[42] J. Liu, C. Cui, P. Lin, P. Hui, A. M. Y. M. Ghias, C. Zhang, A Large Language Model based Multiagent Framework for Goal Oriented Controller Design in Power Electronics, in: 2025 8th International Conference on Energy, Electrical and Power Engineering (CEEPE), IEEE, 2025, pp. 1499–1502. doi: 10.1109/ceepe64987.2025.11034293.

[43] S. Gao, A. Fang, Y. Huang, V. Giunchiglia, A. Noori, J. R. Schwarz, Y. Ektefaie, J. Kondic, M. Zitnik, Empowering biomedical discovery with AI agents, Cell 187 (22) (2024) 6125–6151. doi:10.1016/j. cell.2024.09.022.

[44] Y. Zhang, S. A. Khan, A. Mahmud, H. Yang, A. Lavin, M. Levin, J. Frey, J. Dunnmon, J. Evans, A. Bundy, S. Dzeroski, J. Tegner, H. Zenil, Exploring the role of large language models in the scientific method: from hypothesis to discovery, npj Artificial Intelligence 1 (1) (2025) 14. doi:10.1038/s44387-025-00019-5.

[45] C. K. Reddy, P. Shojaee, Towards Scientific Discovery with Generative AI: Progress, Opportunities, and Challenges, Proceedings of the AAAI Conference on Artificial Intelligence 39 (27) (2025) 28601–28609. doi:10.1609/aaai.v39i27.35084.

[46] M. C. Ramos, C. J. Collison, A. D. White, A review of large language models and autonomous agents in chemistry, Chemical Science 16 (6) (2025) 2514–2572. doi:10.1039/d4sc03921a.

[47] Y. Zhang, X. Chen, B. Jin, S. Wang, S. Ji, W. Wang, J. Han, A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery, in: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, 2024, pp. 8783–8817. doi:10.18653/v1/2024.emnlp-main.498.

[48] T. Zheng, Z. Deng, H. T. Tsang, W. Wang, J. Bai, Z. Wang, Y. Song, From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery, in: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, 2025, pp. 17744–17761. doi:10.18653/v1/2025.emnlp-main.895.

[49] M. H. Prince, H. Chan, A. Vriza, T. Zhou, V. K. Sastry, Y. Luo, M. T. Dearing, R. J. Harder, R. K. Vasudevan, M. J. Cherukara, Opportunities for retrieval and tool augmented large language models in scientific facilities, npj Computational Materials 10 (1) (2024) 251. doi:10.1038/s41524-024-01423-2.

[50] A. M. Bran, S. Cox, O. Schilter, C. Baldassari, A. D. White, P. Schwaller, Augmenting large language models with chemistry tools, Nature Machine Intelligence 6 (5) (2024) 525–535. doi:10.1038/ s42256-024-00832-8.

[51] M. J. Buehler, PRefLexOR: preference-based recursive language modeling for exploratory optimization of reasoning and agentic thinking, npj Artificial Intelligence 1 (1) (2025) 4. doi:10.1038/ s44387-025-00003-z.

[52] C. Lu, C. Lu, R. T. Lange, Y. Yamada, S. Hu, J. Foerster, D. Ha, J. Clune, Towards end-to-end automation of AI research, Nature 651 (2026) 914–919. doi:10.1038/s41586-026-10265-5.

[53] S. Schmidgall, Y. Su, Z. Wang, X. Sun, J. Wu, X. Yu, J. Liu, M. Moor, Z. Liu, E. Barsoum, Agent Laboratory: Using LLM Agents as Research Assistants, arXiv preprint (2025). doi:10.48550/arxiv.2501.04227.

[54] K. Swanson, W. Wu, N. L. Bulaong, J. E. Pak, J. Zou, The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies, Nature 646 (8085) (2025) 716–723. doi:10.1038/s41586-025-09442-9.

[55] A. Ghafarollahi, M. J. Buehler, SciAgents: Automating Scientific Discovery Through Bioinspired Multi-Agent Intelligent Graph Reasoning, Advanced Materials 37 (22) (2025) 2413523. doi:10.1002/adma. 202413523.

[56] Z. Yang, X. Du, J. Li, J. Zheng, S. Poria, E. Cambria, Large Language Models for Automated Opendomain Scientific Hypotheses Discovery, in: Findings of the Association for Computational Linguistics ACL 2024, Association for Computational Linguistics, 2024, pp. 13545–13565. doi:10.18653/v1/2024. findings-acl.804.

[57] M. Tacke, M. Busch, K. Bali, K. Abdolazizi, K. Linka, C. Cyron, R. Aydin, Constitutive scientific generative agent (CSGA): Leveraging large language models for automated constitutive model discovery, Machine Learning for Computational Science and Engineering 1 (1) (2025) 23. doi:10.1007/ s44379-025-00022-2.

[58] X. Tang, T. Hu, M. Ye, D. Shao, X. Yin, S. Ouyang, W. Zhou, P. Lu, Z. Zhang, Y. Zhao, A. Cohan, M. Gerstein, ChemAgent: Self-updating memories in large language models improves chemical reasoning, in: International Conference on Learning Representations, 2025. https://proceedings.iclr.cc/paper\_files/paper/2025/ hash/fa7f64b45970e6a7f8824781e7e01501-Abstract-Conference.html.

[59] X. Tang, W. Xu, Y. Wang, Z. Guo, D. Shao, J. Chen, C. Zhang, Z. Wang, L. Zhang, G. Wan, W. Zhang, L. Bai, Z. Yin, P. Torr, H. Wang, D. Jin, Eigen-Agent: Adaptive Multi-Agent Scientific Reasoning with Monitor-Based RAG, arXiv:2509.21193v2 (2026). https://arxiv.org/abs/2509.21193v2.

[60] J. Dagdelen, A. Dunn, S. Lee, N. Walker, A. S. Rosen, G. Ceder, K. A. Persson, A. Jain, Structured information extraction from scientific text with large language models, Nature Communications 15 (1) (2024) 1418. doi:10.1038/s41467-024-45563-x.

[61] H. Wang, T. Fu, Y. Du, W. Gao, K. Huang, Z. Liu, P. Chandak, S. Liu, P. Van Katwyk, A. Deac, A. Anandkumar, K. Bergen, C. P. Gomes, S. Ho, P. Kohli, J. Lasenby, J. Leskovec, T.-Y. Liu, A. Manrai, D. Marks, B. Ramsundar, L. Song, J. Sun, J. Tang, P. Velickoviˇ c, M. Welling, L. Zhang, C. W. Coley, Y. Ben-´ gio, M. Zitnik, Scientific discovery in the age of artificial intelligence, Nature 620 (7972) (2023) 47–60. doi:10.1038/s41586-023-06221-2.

[62] S. Rossi, M. Rossi, R. R. Mukkamala, J. B. Thatcher, Y. K. Dwivedi, Augmenting research methods with foundation models and generative AI, International Journal of Information Management 77 (2024) 102749. doi:10.1016/j.ijinfomgt.2023.102749.

[63] W. X. Zhao, K. Zhou, J. Li, T. Tang, Z. Dong, Y. Hou, B. Zhang, Y. Min, J. Zhang, P. Liu, X. Wang, Y. Du, C. Yang, Y. Chen, Z. Chen, J. Jiang, R. Ren, Y. Li, X. Tang, Z. Liu, Y. Hu, J.-Y. Nie, J.-R. Wen, A Survey of Large Language Models, Frontiers of Computer Science 20 (12) (2026) 2012627. doi:10.1007/ s11704-026-60308-3.

[64] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, J. Wen, A survey on large language model based autonomous agents, Frontiers of Computer Science 18 (6) (2024) 186345. doi:10.1007/s11704-024-40231-1.

[65] X. Tang, T. Qin, T. Peng, Z. Zhou, D. Shao, T. Du, X. Wei, P. Xia, F. Wu, H. Zhu, G. Zhang, J. Liu, X. Wang, S. Hong, C. Wu, H. Cheng, C. Wang, W. Zhou, Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving, arXiv:2507.06229v5 (2025). https://arxiv.org/abs/2507.06229v5.

[66] A. Zaboli, Y.-H. Kim, J. Hong, An Advanced Generative AI-Based Anomaly Detection in IEC61850-Based Communication Messages in Smart Grids, IEEE Access 13 (2025) 89997–90016. doi:10.1109/access. 2025.3571881.

[67] M. Gao, S. Zhou, W. Gu, Z. Wu, H. Liu, A. Zhou, X. Wang, MMGPT4LF: Leveraging an optimized pretrained GPT-2 model with multi-modal cross-attention for load forecasting, Applied Energy 392 (2025) 125965. doi:10.1016/j.apenergy.2025.125965.

[68] Y. Zhou, M. Wang, Empower Pre-Trained Large Language Models for Building-Level Load Forecasting, IEEE Transactions on Power Systems 40 (5) (2025) 4220–4232. doi:10.1109/tpwrs.2025.3548891.

[69] S. Li, L. Wu, T. Peng, J. Huang, H. Jiang, Y. Xue, J. Zhang, D. W. Gao, RSynLLM: A risk-aware routing mixture-of-experts large language model for multi-energy load forecasting in large-scale distribution networks, Applied Energy 406 (2026) 127227. doi:10.1016/j.apenergy.2025.127227.

[70] G. Zhu, W. Jia, Z. Xing, L. Xiang, A. Hu, R. Hao, CMLLM: A novel cross-modal large language model for wind power forecasting, Energy Conversion and Management 330 (2025) 119673. doi:10.1016/j. enconman.2025.119673.

[71] A. Hannaan, A. Khandakar, M. F. Hossen, M. T. A. Noor, M. Arselene Ayari, N. Ahmed Al-Emadi, S. M. Muyeen, A Structured Question–Answer Dataset for Large Language Models in Power System Dynamics and Operations, IEEE Access 14 (2026) 8391–8407. doi:10.1109/access.2026.3652625.

[72] L. Zhang, Z. Chen, Large language model-based interpretable machine learning control in building energy systems, Energy and Buildings 313 (2024) 114278. doi:10.1016/j.enbuild.2024.114278.

[73] P. Christou, M. Z. Islam, Y. Lin, J. Xiong, LLM4DistReconfig: A Fine-tuned Large Language Model for Power Distribution Network Reconfiguration, in: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Association for Computational Linguistics, 2025, pp. 4136–4155. doi:10.18653/v1/2025.naacl-long.208.

[74] S. L. Choi, R. Jain, C. Feng, P. Emami, H. Zhang, J. Hong, T. Kim, S. Park, F. Ding, M. Baggu, B. Kroposki, Generative AI for Power Grid Operations, Tech. Rep. NREL/TP-5D00-91176, National Renewable Energy Laboratory (2024). doi:10.2172/2477920.

[75] H. F. Hamann, B. Gjorgiev, T. Brunschwiler, L. S. Martins, A. Puech, A. Varbella, J. Weiss, J. Bernabe-Moreno, A. B. Massé, S. L. Choi, I. Foster, B.-M. Hodge, R. Jain, K. Kim, V. Mai, F. Mirallès, M. De Mon-

tigny, O. Ramos-Leaños, H. Suprême, L. Xie, E.-N. S. Youssef, A. Zinflou, A. Belyi, R. J. Bessa, B. P. Bhattarai, J. Schmude, S. Sobolevsky, Foundation models for the electric power grid, Joule 8 (12) (2024) 3245–3258. doi:10.1016/j.joule.2024.11.002.

[76] H. Dui, Q. Zeng, M. Xie, Generative AI-based spatiotemporal resilience, green and low-carbon transformation strategy of smart renewable energy systems, Frontiers of Engineering Management 12 (4) (2025) 1220–1235. doi:10.1007/s42524-025-4147-6

[77] X. Tang, J. Chen, D. Shao, Z. Yu, Y. Cui, W. Wang, F. Wu, Y. Zhuang, W. Shi, Z. Huang, A. Cohan, X. Lin, F. Theis, S. Krishnaswamy, M. Gerstein, AI Scientists for Building Virtual-Cell Models, arXiv:2508.02276v3 (2026). https://arxiv.org/abs/2508.02276v3.

[78] J. Göpfert, J. M. Weinand, P. Kuckertz, D. Stolten, Opportunities for large language models and discourse in engineering design, Energy and AI 17 (2024) 100383. doi:10.1016/j.egyai.2024.100383.

[79] R. Yu, C. Qiu, R. Ladwig, P. Hanson, Y. Xie, X. Jia, Physics-Guided Foundation Model for Scientific Discovery: An Application to Aquatic Science, Proceedings of the AAAI Conference on Artificial Intelligence 39 (27) (2025) 28548–28556. doi:10.1609/aaai.v39i27.35078.

[80] M. Bossart, J. D. Lara, C. Roberts, R. Henriquez-Auba, D. S. Callaway, B.-M. Hodge, Acceleration of Power System Dynamic Simulations Using a Deep Equilibrium Layer and Neural ODE Surrogate, IEEE Transactions on Energy Conversion 40 (4) (2025) 2710–2722. doi:10.1109/tec.2025.3563142.

[81] B. Habib, E. Isufi, W. van Breda, A. Jongepier, J. L. Cremer, Deep Statistical Solver for Distribution System State Estimation, IEEE Transactions on Power Systems 39 (2) (2024) 4039–4050. doi:10.1109/tpwrs. 2023.3290358.

[82] R. R. Hossain, Q. Huang, R. Huang, Graph Convolutional Network-Based Topology Embedded Deep Reinforcement Learning for Voltage Stability Control, IEEE Transactions on Power Systems 36 (5) (2021) 4848–4851. doi:10.1109/tpwrs.2021.3084469.

[83] X. Zhou, X. Guan, D. Sun, H. Jiang, J. Peng, Y. Jin, Y. Zhang, Transient Stability Assessment Based on Gated Graph Neural Network With Imbalanced Data in Internet of Energy, IEEE Internet of Things Journal 9 (12) (2022) 9320–9331. doi:10.1109/jiot.2021.3127895.

[84] Q. Chen, S. Bu, H. Wang, C. Lei, Real-Time Multi-Stability Risk Assessment and Visualization of Power Systems: A Graph Neural Network-Based Method, IEEE Transactions on Power Systems 40 (4) (2025) 2955–2968. doi:10.1109/tpwrs.2024.3524406.

[85] S. Moshtagh, B. Azimian, M. Golgol, A. Pal, Topology-Aware Graph Neural Network-Based State Estimation for PMU-Unobservable Power Systems, IEEE Transactions on Power Systems 40 (6) (2025) 4547–4560. doi:10.1109/tpwrs.2025.3572848.

[86] R. Steven, O. V. Klymenko, M. Short, Machine learning-accelerated distributed optimisation methods for optimal power flow: A review, Renewable and Sustainable Energy Reviews 226 (2026) 116190. doi:10. 1016/j.rser.2025.116190.

[87] C. Shen, K. Zuo, M. Sun, Physics-Augmented Auxiliary Learning for Power System Transient Stability Assessment, IEEE Transactions on Industrial Informatics 21 (9) (2025) 6811–6822. doi:10.1109/tii. 2025.3567400.

[88] H. Wang, Z. Yang, W. Kang, P. Sun, G. Konstantinou, Z. Chen, Physics-Informed Learning Based Wind Farm Two-Machine Aggregation Model for Large-Scale Power System Stability Studies, IEEE Transactions on Power Systems 40 (1) (2025) 879–892. doi:10.1109/tpwrs.2024.3405543.

[89] Z. Wang, F. Teng, Y. Zhou, Q. Guo, H. Sun, Uncertainty-Aware Transient Stability-Constrained Preventive Redispatch: A Distributional Reinforcement Learning Approach, IEEE Transactions on Power Systems 40 (2) (2025) 1295–1308. doi:10.1109/tpwrs.2024.3421971.

[90] R. Nellikkath, S. Chatzivasileiadis, Physics-Informed Neural Networks for AC Optimal Power Flow, Electric Power Systems Research 212 (2022) 108412. doi:10.1016/j.epsr.2022.108412.

[91] A. Daareyni, A. Martikkala, H. Mokhtarian, I. F. Ituarte, Generative AI meets CAD: enhancing engineer ing design to manufacturing processes with large language models, The International Journal of Advanced Manufacturing Technology (2025). doi:10.1007/s00170-025-15830-2.

[92] Z. Jiang, A. Liu, D. Zhang, X. Xu, Y. Dai, Customization and personalization of large language models for engineering design, CIRP Annals 74 (1) (2025) 191–195. doi:10.1016/j.cirp.2025.03.001.

[93] J. White, S. Hays, Q. Fu, J. Spencer-Smith, D. C. Schmidt, ChatGPT Prompt Patterns for Improving Code Quality, Refactoring, Requirements Elicitation, and Software Design, in: Generative AI for Effective Soft-

ware Development, Springer Nature Switzerland, 2024, pp. 71–108. doi:10.1007/978-3-031-55642-5\_ 4.

[94] J. Zamfirescu-Pereira, R. Y. Wong, B. Hartmann, Q. Yang, Why Johnny Can’t Prompt: How Non-AI Experts Try (and Fail) to Design LLM Prompts, in: Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, ACM, 2023, pp. 1–21. doi:10.1145/3544548.3581388.

[95] T. Gupta, M. Zaki, N. M. A. Krishnan, Mausam, MatSciBERT: A materials domain language model for text mining and information extraction, npj Computational Materials 8 (1) (2022) 102. doi:10.1038/ s41524-022-00784-w.

[96] S. Ji, S. Pan, E. Cambria, P. Marttinen, P. S. Yu, A Survey on Knowledge Graphs: Representation, Acquisition, and Applications, IEEE Transactions on Neural Networks and Learning Systems 33 (2) (2022) 494–514. doi:10.1109/tnnls.2021.3070843.

[97] P. W. J. Staar, M. Dolfi, C. Auer, Corpus processing service: A Knowledge Graph platform to perform deep data exploration on corpora, Applied AI Letters 1 (2) (2020) e20. doi:10.1002/ail2.20.

[98] M. d’Aquin, On the role of knowledge graphs in AI-based scientific discovery, Journal of Web Semantics 84 (2025) 100854. doi:10.1016/j.websem.2024.100854.

[99] Q. Jin, Y. Yang, Q. Chen, Z. Lu, GeneGPT: augmenting large language models with domain tools for improved access to biomedical information, Bioinformatics 40 (2) (2024) btae075. doi:10.1093/ bioinformatics/btae075.

[100] Y. Wan, Z. Chen, Y. Liu, C. Chen, M. Packianather, Empowering LLMs by hybrid retrieval-augmented generation for domain-centric Q&A in smart manufacturing, Advanced Engineering Informatics 65 (2025) 103212. doi:10.1016/j.aei.2025.103212.

[101] J. Sequeda, D. Allemang, B. Jacob, A Benchmark to Understand the Role of Knowledge Graphs on Large Language Model’s Accuracy for Question Answering on Enterprise SQL Databases, in: Proceedings of the 7th Joint Workshop on Graph Data Management Experiences & Systems (GRADES) and Network Data Analytics (NDA), ACM, 2024, pp. 1–12. doi:10.1145/3661304.3661901.

[102] G. Gamage, S. Kahawala, N. Mills, D. De Silva, M. Manic, D. Alahakoon, A. Jennings, Augmenting Industrial Chatbots in Energy Systems using ChatGPT Generative AI, in: 2023 IEEE 32nd International Symposium on Industrial Electronics (ISIE), IEEE, 2023, pp. 1–6. doi:10.1109/isie51358.2023.10228101.

[103] J. Bai, S. Mosbach, C. J. Taylor, D. Karan, K. F. Lee, S. D. Rihm, J. Akroyd, A. A. Lapkin, M. Kraft, A dynamic knowledge graph approach to distributed self-driving laboratories, Nature Communications 15 (1) (2024) 462. doi:10.1038/s41467-023-44599-9.

[104] G. Agrawal, T. Kumarage, Z. Alghamdi, H. Liu, Mindful-RAG: A Study of Points of Failure in Retrieval Augmented Generation, in: 2024 2nd International Conference on Foundation and Large Language Models (FLLM), IEEE, 2024, pp. 607–611. doi:10.1109/fllm63129.2024.10852457.

[105] B. Han, T. Susnjak, A. Mathrani, Automating Systematic Literature Reviews with Retrieval-Augmented Generation: A Comprehensive Overview, Applied Sciences 14 (19) (2024) 9103. doi:10.3390/app14199103.

[106] R. Liu, Z. Zou, S. Chen, Y. Liu, J. Wan, Harnessing AI for understanding scientific literature: Innovations and applications of chat-agent system in battery recycling research, Materials Today Energy 49 (2025) 101818. doi:10.1016/j.mtener.2025.101818.

[107] S. M. Parsa, Physics-informed machine learning meets renewable energy systems: A review of advances, challenges, guidelines, and future outlooks, Applied Energy 402 (2025) 126925. doi:10.1016/j. apenergy.2025.126925.

[108] M. Abouyehia, A. Egea-Àlvarez, K. H. Ahmed, Evaluating inertia estimation methods in low-inertia power systems: A comprehensive review with analytic hierarchy process-based ranking, Renewable and Sustainable Energy Reviews 217 (2025) 115794. doi:10.1016/j.rser.2025.115794.

[109] J. Zeng, X. Li, H. Dai, L. Zhang, W. Wang, Z. Zhang, S. Kong, L. Xu, A Residual Physics-Informed Neural Network Approach for Identifying Dynamic Parameters in Swing Equation-Based Power Systems, Energies 18 (11) (2025) 2888. doi:10.3390/en18112888.

[110] S. Stock, J. Stiasny, D. Babazadeh, C. Becker, S. Chatzivasileiadis, Bayesian Physics-Informed Neural Networks for Robust System Identification of Power Systems, in: 2023 IEEE Belgrade PowerTech, IEEE, 2023, pp. 1–6. doi:10.1109/powertech55446.2023.10202692.

[111] S. Stock, D. Babazadeh, C. Becker, S. Chatzivasileiadis, Bayesian Physics-informed Neural Networks for system identification of inverter-dominated power systems, Electric Power Systems Research 235 (2024)

110860. doi:10.1016/j.epsr.2024.110860.

[112] J. Stiasny, B. Zhang, S. Chatzivasileiadis, PINNSim: A simulator for power system dynamics based on Physics-Informed Neural Networks, Electric Power Systems Research 235 (2024) 110796. doi:10.1016/ j.epsr.2024.110796.

[113] I. Ventura Nadal, J. Stiasny, S. Chatzivasileiadis, Physics-Informed Neural Networks: A plug and play integration into power system dynamic simulations, Electric Power Systems Research 248 (2025) 111885. doi:10.1016/j.epsr.2025.111885.

[114] H. Shuai, F. Li, Physics-Informed Kolmogorov-Arnold Networks for Power System Dynamics, IEEE Open Access Journal of Power and Energy 12 (2025) 46–58. doi:10.1109/oajpe.2025.3529928.

[115] M. Mohammadian, K. Baker, F. Fioretto, Gradient-enhanced physics-informed neural networks for power systems operational support, Electric Power Systems Research 223 (2023) 109551. doi:10.1016/j.epsr. 2023.109551.

[116] H. Khaloie, M. Dolányi, J.-F. Toubeau, F. Vallée, Review of machine learning techniques for optimal power flow, Applied Energy 388 (2025) 125637. doi:10.1016/j.apenergy.2025.125637.

[117] C. Li, A. Kies, K. Zhou, M. Schlott, O. E. Sayed, M. Bilousova, H. Stöcker, Optimal Power Flow in a highly renewable power system based on attention neural networks, Applied Energy 359 (2024) 122779. doi:10.1016/j.apenergy.2024.122779.

[118] À. Tudoras-Miravet, E. González-Iakl, O. Gomis-Bellmunt, E. Prieto-Araujo, Physics-Informed Neural Networks for Power Systems Warm-Start Optimization, IEEE Access 12 (2024) 135913–135928. doi: 10.1109/access.2024.3406471.

[119] J. Authier, R. Haider, A. Annaswamy, F. Dörfler, Physics-informed Graph Neural Network for Dynamic Reconfiguration of power systems, Electric Power Systems Research 235 (2024) 110817. doi:10.1016/j. epsr.2024.110817.

[120] M. E. C. Bento, Load Margin Assessment of Power Systems Using Physics-Informed Neural Network with Optimized Parameters, Energies 17 (7) (2024) 1562. doi:10.3390/en17071562.

[121] S. Falas, M. Asprou, C. Konstantinou, M. K. Michael, Physics-Informed Neural Networks for Accelerating Power System State Estimation, in: 2023 IEEE PES Innovative Smart Grid Technologies Europe (ISGT EUROPE), IEEE, 2023, pp. 1–5. doi:10.1109/isgteurope56780.2023.10408467.

[122] E. Cibaku, F. Gama, S. Park, Boosting efficiency in state estimation of power systems by leveraging attention mechanism, Energy and AI 16 (2024) 100369. doi:10.1016/j.egyai.2024.100369.

[123] Y. Raghuvamsi, K. Teeparthi, A review on distribution system state estimation uncertainty issues using deep learning approaches, Renewable and Sustainable Energy Reviews 187 (2023) 113752. doi:10.1016/j. rser.2023.113752.

[124] Y. Zhang, X. Shi, H. Zhang, Y. Cao, V. Terzija, Review on deep learning applications in frequency analysis and control of modern power system, International Journal of Electrical Power & Energy Systems 136 (2022) 107744. doi:10.1016/j.ijepes.2021.107744.

[125] H. Zhou, A. Aral, I. Brandic, M. Erol-Kantarci, Multiagent Bayesian Deep Reinforcement Learning for Microgrid Energy Management Under Communication Failures, IEEE Internet of Things Journal 9 (14) (2022) 11685–11698. doi:10.1109/jiot.2021.3131719.

[126] A. Fan, B. Gokkaya, M. Harman, M. Lyubarskiy, S. Sengupta, S. Yoo, J. M. Zhang, Large Language Models for Software Engineering: Survey and Open Problems, in: 2023 IEEE/ACM International Conference on Software Engineering: Future of Software Engineering (ICSE-FoSE), IEEE, 2023, pp. 31–53. doi:10. 1109/icse-fose59343.2023.00008.

[127] A. Kashefi, T. Mukerji, ChatGPT FOR PROGRAMMING NUMERICAL METHODS, Journal of Machine Learning for Modeling and Computing 4 (2) (2023) 1–74. doi:10.1615/jmachlearnmodelcomput. 2023048492.

[128] Q. Jiang, Z. Gao, G. E. Karniadakis, DeepSeek vs. ChatGPT vs. Claude: A comparative study for scientific computing and scientific machine learning tasks, Theoretical and Applied Mechanics Letters 15 (3) (2025) 100583. doi:10.1016/j.taml.2025.100583.

[129] X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, H. Tran, F. Li, R. Ma, M. Zheng, B. Qian, D. Shao, N. Muennighoff, Y. Zhang, B. Hui, J. Lin, R. Brennan, H. Peng, H. Ji, G. Neubig, OpenHands: An open platform for AI software developers as generalist agents, in: International Conference on Learning Representations, 2025. https://proceedings.iclr.cc/paper\_files/paper/2025/

[130] X. Tang, Y. Liu, Z. Cai, Y. Shao, J. Lu, Y. Zhang, Z. Deng, H. Hu, K. An, R. Huang, S. Si, S. Chen, H. Zhao, L. Chen, Y. Wang, T. Liu, Z. Jiang, B. Chang, Y. Fang, Y. Qin, W. Zhou, Y. Zhao, A. Cohan, M. Gerstein, ML-Bench: Evaluating Large Language Models and Agents for Machine Learning Tasks on Repository-Level Code, arXiv:2311.09835v5 (2024). https://arxiv.org/abs/2311.09835v5.

[131] Z. Wang, Z. Liu, Y. Zhang, A. Zhong, J. Wang, F. Yin, L. Fan, L. Wu, Q. Wen, RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models, in: Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, ACM, 2024, pp. 4966–4974. doi:10.1145/3627673.3680016.

[132] S. Wen, F. Li, W. Zhuang, X. Pan, W. Yu, J. Bao, X. Li, Leveraging large language models for Human-Machine collaborative troubleshooting of complex industrial equipment faults, Advanced Engineering Informatics 65 (2025) 103235. doi:10.1016/j.aei.2025.103235.

[133] C. Tsigkanos, P. Rani, S. Müller, T. Kehrer, Variable Discovery with Large Language Models for Metamorphic Testing of Scientific Software, in: Lecture Notes in Computer Science, Springer Nature Switzerland, 2023, pp. 321–335. doi:10.1007/978-3-031-35995-8\_23.

[134] C. Huang, S. Li, R. Liu, H. Wang, Y. Chen, Large Foundation Models for Power Systems, in: 2024 IEEE Power & Energy Society General Meeting (PESGM), IEEE, 2024, pp. 1–5. doi:10.1109/pesgm51994. 2024.10688670.

[135] M. Meyer, D. Zapata Gonzalez, S. Kaltenpoth, O. Müller, Benchmarking Time Series Foundation Models for Short-Term Household Electricity Load Forecasting, IEEE Access 13 (2025) 218141–218153. doi: 10.1109/access.2025.3648056.

[136] M. Liang, Y. Hu, H. Weng, J. Xi, B. Yin, EnergyGPT: Fine-tuning large language model for multi-energy load forecasting, Renewable Energy 251 (2025) 123313. doi:10.1016/j.renene.2025.123313.

[137] C. Ying, W. Wang, J. Yu, Q. Li, D. Yu, J. Liu, Deep learning for renewable energy forecasting: A taxonomy, and systematic literature review, Journal of Cleaner Production 384 (2023) 135414. doi: 10.1016/j.jclepro.2022.135414.

[138] M. N. Akhter, S. Mekhilef, H. Mokhlis, N. Mohamed Shah, Review on forecasting of photovoltaic power generation based on machine learning and metaheuristic techniques, IET Renewable Power Generation 13 (7) (2019) 1009–1023. doi:10.1049/iet-rpg.2018.5649.

[139] U. Hussan, H. Wang, J. Peng, H. Jiang, H. Rasheed, Transformer-based renewable energy forecasting: A comprehensive review, Renewable and Sustainable Energy Reviews 226 (2026) 116356. doi:10.1016/j. rser.2025.116356.

[140] D. Syed, H. Abu-Rub, A. Ghrayeb, S. S. Refaat, M. Houchati, O. Bouhali, S. Banales, Deep Learning-Based Short-Term Load Forecasting Approach in Smart Grid With Clustering and Consumption Pattern Recognition, IEEE Access 9 (2021) 54992–55008. doi:10.1109/access.2021.3071654.

[141] L. Wu, C. Kong, X. Hao, W. Chen, A Short-Term Load Forecasting Method Based on GRU-CNN Hybrid Neural Network Model, Mathematical Problems in Engineering 2020 (2020) 1–10. doi:10.1155/2020/ 1428104.

[142] Z. Cao, J. Wang, Y. Xia, Combined electricity load-forecasting system based on weighted fuzzy time series and deep neural networks, Engineering Applications of Artificial Intelligence 132 (2024) 108375. doi: 10.1016/j.engappai.2024.108375.

[143] P. M. J. Ganesh, B. M. Sundaram, P. K. Balachandran, M. A. A. M. Zainuri, Syner-Dandelion Optimization Integrated Temporal Recurrent Transformer Network for Accurate Energy Load Forecasting in IoT-Smart Grids, IEEE Internet of Things Journal 12 (20) (2025) 41752–41770. doi:10.1109/jiot.2025.3590389.

[144] B. Chen, W. Yang, B. Yan, K. Zhang, An advanced airport terminal cooling load forecasting model integrating SSA and CNN-Transformer, Energy and Buildings 309 (2024) 114000. doi:10.1016/j.enbuild.2024. 114000.

[145] J. Liu, Y. Fu, Renewable energy forecasting: A self-supervised learning-based transformer variant, Energy 284 (2023) 128730. doi:10.1016/j.energy.2023.128730.

[146] Q. Zhang, J. Chen, G. Xiao, S. He, K. Deng, TransformGraph: A novel short-term electricity net load forecasting model, Energy Reports 9 (2023) 2705–2717. doi:10.1016/j.egyr.2023.01.050.

[147] Z. Niu, X. Han, D. Zhang, Y. Wu, S. Lan, Interpretable wind power forecasting combining seasonal-trend representations learning with temporal fusion transformers architecture, Energy 306 (2024) 132482. doi:

10.1016/j.energy.2024.132482.

[148] Q. Wen, Y. Liu, Feature engineering and selection for prosumer electricity consumption and production forecasting: A comprehensive framework, Applied Energy 381 (2025) 125176. doi:10.1016/j.apenergy. 2024.125176.

[149] P. Li, Z. Hu, Y. Shen, X. Cheng, M. Alhazmi, Short-term electricity load forecasting based on large language models and weighted external factor optimization, Sustainable Energy Technologies and Assessments 82 (2025) 104449. doi:10.1016/j.seta.2025.104449.

[150] J. Xiao, S. Liu, S. Peng, W. Zhang, C. Shan, Z. Xu, Z. Wang, Leveraging Pretrained Large Language Models for Power System Source-Load Forecasting, in: 2025 7th International Conference on Power and Energy Technology (ICPET), IEEE, 2025, pp. 171–175. doi:10.1109/icpet66029.2025.11160348.

[151] J. Li, D. Li, Y. Yang, H. Xi, W. Yu, Y. Xiao, Q. Yang, Zero-shot load forecasting for integrated energy systems: A large language model-based framework with multi-task learning, Neurocomputing 654 (2025) 131288. doi:10.1016/j.neucom.2025.131288.

[152] W. Zhang, S. Liu, M. Li, H. Bao, S. Yuan, X. Cheng, A Method for Load Forecasting of Distribution Transformers Based on Parameter-Efficient Fine-Tuning of Large Language Models in Power System, Processes 13 (9) (2025) 2986. doi:10.3390/pr13092986.

[153] Z. Wang, W. Jiang, J. Xu, Z. Xu, A. Zhou, M. Xu, Grid2Vec: Learning Node Representations of Digital Power Systems for Anomaly Detection, IEEE Transactions on Smart Grid 15 (5) (2024) 5031–5042. doi: 10.1109/tsg.2024.3377223.

[154] Y. Zhang, P. M. Karve, S. Mahadevan, Graph neural networks for power grid operational risk assessment under evolving unit commitment, Applied Energy 380 (2025) 124793. doi:10.1016/j.apenergy.2024. 124793.

[155] J. Lin, M. Tu, H. Hong, C. Lu, W. Song, Spatiotemporal Graph Convolutional Neural Network-Based Forecasting-Aided State Estimation Using Synchrophasors, IEEE Internet of Things Journal 11 (9) (2024) 16171–16183. doi:10.1109/jiot.2024.3351850.

[156] N. Yan, W. Guan, X. Yu, J. Hou, T. Zeng, A Visual Foundation Model of Image Segmentation for Power Systems, in: 2024 IEEE 14th International Conference on CYBER Technology in Automation, Control, and Intelligent Systems (CYBER), IEEE, 2024, pp. 569–574. doi:10.1109/cyber63482.2024.10749618.

[157] A. M. Karimi, Y. Wu, M. Koyuturk, R. H. French, Spatiotemporal Graph Neural Network for Performance Prediction of Photovoltaic Power Systems, Proceedings of the AAAI Conference on Artificial Intelligence 35 (17) (2021) 15323–15330. doi:10.1609/aaai.v35i17.17799.

[158] C. Wang, J. An, G. Mu, Power System Network Topology Identification Based on Knowledge Graph and Graph Neural Network, Frontiers in Energy Research 8 (2021) 613331. doi:10.3389/fenrg.2020. 613331.

[159] L. Tong, X. Wang, Q. Zhao, Grid monitoring with synchro-waveform and AI foundation model technologies, Energy Systems 17 (3) (2026) 1221–1248. doi:10.1007/s12667-025-00726-7.

[160] A. Rasheed, O. San, T. Kvamsdal, Digital Twin: Values, Challenges and Enablers From a Modeling Perspective, IEEE Access 8 (2020) 21980–22012. doi:10.1109/access.2020.2970143.

[161] M. Jafari, A. Kavousi-Fard, T. Chen, M. Karimi, A Review on Digital Twin Technology in Smart Grid, Transportation System and Smart City: Challenges and Future, IEEE Access 11 (2023) 17471–17484. doi: 10.1109/access.2023.3241588.

[162] G. Rahmani-Sane, S. Azad, M. Taghi Ameli, S. Haghani, The Applications of Artificial Intelligence and Digital Twin in Power Systems: An In-Depth Review, IEEE Access 13 (2025) 108573–108608. doi:10. 1109/access.2025.3580340.

[163] N. Mchirgui, N. Quadar, H. Kraiem, A. Lakhssassi, The Applications and Challenges of Digital Twin Technology in Smart Grids: A Comprehensive Review, Applied Sciences 14 (23) (2024) 10933. doi: 10.3390/app142310933.

[164] A. Ranawaka, D. Alahakoon, Y. Sun, K. Hewapathirana, Leveraging the Synergy of Digital Twins and Artificial Intelligence for Sustainable Power Grids: A Scoping Review, Energies 17 (21) (2024) 5342. doi:10.3390/en17215342.

[165] O. Das, M. H. Zafar, F. Sanfilippo, S. Rudra, M. L. Kolhe, Advancements in digital twin technology and machine learning for energy systems: A comprehensive review of applications in smart grids, renewable energy, and electric vehicle optimisation, Energy Conversion and Management: X 24 (2024) 100715. doi:

10.1016/j.ecmx.2024.100715.

[166] V. Lopez, L. Hoang, M. Martinez-Galindo, R. Fernández-Díaz, M. L. Sbodio, R. Ordonez-Hurtado, M. Zayats, N. Mulligan, J. Bettencourt-Silva, Enhancing foundation models for scientific discovery via multimodal knowledge graph representations, Journal of Web Semantics 84 (2025) 100845. doi:10.1016/j.websem. 2024.100845.

[167] R. Haridas, C. Gu, S. Sharma, R. Bhakar, A Conceptual Framework for Applying Large Language Models in Smart Grid Cybersecurity, in: 2025 IEEE 5th International Conference on Sustainable Energy and Future Electric Transportation (SEFET), IEEE, 2025, pp. 1–6. doi:10.1109/sefet65155.2025.11255142.

[168] M. S. Munir, S. Proddatoori, M. Muralidhara, S. Shetty, A Bayesian Belief Network Framework for Protecting Generative AI-driven Attack in Smart Grid Communication, in: 2025 International Conference on Computing, Networking and Communications (ICNC), IEEE, 2025, pp. 168–173. doi:10.1109/icnc64010. 2025.10993907.

[169] Z. Zhang, M. Liu, M. Sun, R. Deng, P. Cheng, D. Niyato, M.-Y. Chow, J. Chen, Vulnerability of Machine Learning Approaches Applied in IoT-Based Smart Grid: A Review, IEEE Internet of Things Journal 11 (11) (2024) 18951–18975. doi:10.1109/jiot.2024.3349381.

[170] C. Chen, Y. Wang, M. Cui, J. Zhao, W. Bi, Y. Chen, X. Zhang, Data-Driven Detection of Stealthy False Data Injection Attack Against Power System State Estimation, IEEE Transactions on Industrial Informatics 18 (12) (2022) 8467–8476. doi:10.1109/tii.2022.3149106.

[171] A. Zaboli, S. L. Choi, T.-J. Song, J. Hong, ChatGPT and Other Large Language Models for Cybersecurity of Smart Grid Applications, in: 2024 IEEE Power & Energy Society General Meeting (PESGM), IEEE, 2024, pp. 1–5. doi:10.1109/pesgm51994.2024.10688863.

[172] A. Zaboli, J. Hong, A. ¸Stefanov, C.-C. Liu, C.-S. Hwang, Large Language Models for Power System Security: A Novel Multi-Modal Approach for Anomaly Detection in Energy Management Systems, IEEE Access 13 (2025) 203558–203585. doi:10.1109/access.2025.3636184.

[173] S. Badakhshan, J. Zhang, Generative AI-Enhanced Real-Time Anomaly Detection in Integrated Energy Systems, IEEE Transactions on Smart Grid 17 (2) (2026) 1549–1560. doi:10.1109/tsg.2025.3630562.

[174] M. R. Shadi, H. Mirshekali, H. R. Shaker, Explainable artificial intelligence for energy systems maintenance: A review on concepts, current techniques, challenges, and prospects, Renewable and Sustainable Energy Reviews 216 (2025) 115668. doi:10.1016/j.rser.2025.115668.

[175] J. Yang, H. Xu, S. Mirzoyan, T. Chen, Z. Liu, W. Ju, L. Liu, M. Zhang, S. Wang, Poisoning scientific knowledge using large language models, Preprint (2023). doi:10.1101/2023.11.06.565928.

[176] Y. K. Dwivedi, N. Kshetri, L. Hughes, E. L. Slade, A. Jeyaraj, A. K. Kar, A. M. Baabdullah, A. Koohang, V. Raghavan, M. Ahuja, H. Albanna, M. A. Albashrawi, A. S. Al-Busaidi, J. Balakrishnan, Y. Barlette, S. Basu, I. Bose, L. Brooks, D. Buhalis, L. Carter, S. Chowdhury, T. Crick, S. W. Cunningham, G. H. Davies, R. M. Davison, R. Dé, D. Dennehy, Y. Duan, R. Dubey, R. Dwivedi, J. S. Edwards, C. Flavián, R. Gauld, V. Grover, M.-C. Hu, M. Janssen, P. Jones, I. Junglas, S. Khorana, S. Kraus, K. R. Larsen, P. Latreille, S. Laumer, F. T. Malik, A. Mardani, M. Mariani, S. Mithas, E. Mogaji, J. H. Nord, S. O’Connor, F. Okumus, M. Pagani, N. Pandey, S. Papagiannidis, I. O. Pappas, N. Pathak, J. Pries-Heje, R. Raman, N. P. Rana, S.- V. Rehm, S. Ribeiro-Navarrete, A. Richter, F. Rowe, S. Sarker, B. C. Stahl, M. K. Tiwari, W. van der Aalst, V. Venkatesh, G. Viglia, M. Wade, P. Walton, J. Wirtz, R. Wright, Opinion Paper: “So what if ChatGPT wrote it?” Multidisciplinary perspectives on opportunities, challenges and implications of generative conversational AI for research, practice and policy, International Journal of Information Management 71 (2023) 102642. doi:10.1016/j.ijinfomgt.2023.102642.

[177] W. Seo, J. Lee, Y. Shao, Q. Zhou, S. Lee, Y. Bu, SPIO: Ensemble and selective strategies via LLMbased multi-agent planning in automated data science, in: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 22681–22703. https: //aclanthology.org/2026.acl-long.1039/.

[178] Z. Ji, M. Jiang, A systematic review of electricity demand for large language models: evaluations, challenges, and solutions, Renewable and Sustainable Energy Reviews 225 (2026) 116159. doi:10.1016/j.rser. 2025.116159.