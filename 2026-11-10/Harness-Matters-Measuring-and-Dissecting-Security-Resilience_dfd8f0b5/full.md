# Harness Matters: Measuring and Dissecting Security Resilience in Web Agents

Xinyi Wu<sup>1§</sup>, Jiagui Chen<sup>1§</sup>, JiaYi Dong<sup>1</sup>, Geng Hong<sup>1B</sup>, Xudong Pan<sup>1,2</sup>, Jiarun Dai<sup>1</sup>, Mingxuan Liu<sup>3</sup>, Min Yang<sup>1,4B</sup>

<sup>1</sup>Fudan University, <sup>2</sup>Shanghai Innovation Institute, <sup>3</sup>Zhongguancun Laboratory,

<sup>4</sup>Shanghai Pudong Research Institute of Cryptology

<sup>1</sup>{xinyiwu20, jgchen20, qhong, xdpan, jrdai, m yang}@fudan.edu.cn, <sup>1</sup>dongjy25@m.fudan.edu.cn, <sup>3</sup>liumx@mail.zgclab.edu.cn B: Corresponding authors

Abstract—Web agents extend LLMs from text generation to real-world web interaction by relying on execution harnesses that interpret user goals, perceive webpages, plan actions, and ground model decisions into browser operations. While prior work has recognized that agent security depends on more than the backend model alone, how harness design quantitatively shapes web-agent security remains insufficiently measured.

To systematically study web-agent harness security, we first characterize web-agent harnesses as a four-stage execution workflow covering goal interpretation, webpage perception, reasoning and planning, and action execution. Based on this abstraction, we build WEBTRAP FARM<sup>\*</sup>, a systematic and extensible evaluation framework covering three representative attack categories: malicious user prompts, malicious prompt injection, and deceptive web design. To avoid conflating security with task capability, we use paired usability calibration tasks to separate genuine defensive behavior from capability limitations.

Our evaluation shows that existing web-agent harnesses remain vulnerable to typical attacks, with some unsafe behaviors occurring even without explicit attack content, revealing harnessintrinsic security failures (RQ1). Statistical analysis further shows that both harness and backend model significantly affect security outcomes, indicating that deployed-agent security is jointly shaped by the harness and backend model (RQ2). Finally, we localize where defensive behavior emerges across workflow stages and identify more secure design mechanisms, providing actionable guidance for building more robust web agents (RQ3). These findings highlight the web-agent harness as an important security boundary and motivate moving beyond model-centric defenses toward harness-level security design.

## 1. Introduction

Large language models (LLMs) are increasingly integrated into agents that autonomously interact with real web environments. Unlike chatbots, web agents automate practical tasks such as information retrieval, online transactions, and appointment scheduling [1, 2, 3, 4]. However, exposing LLMs to untrusted webpages and connecting model outputs to real-world actions introduce broad security threats, including malicious user prompts [5, 6], webpage prompt injection [7, 8, 9, 10], and deceptive webpage designs [11, 12].

Prior studies have shown that web agents can be easily attacked, but they provide limited insight into why unsafe actions are executed. Most evaluations focus on the robustness of backend models against malicious instructions or adversarial webpage content [8, 13]. Although recent academic and industry efforts have recognized that systemlevel design choices beyond the backend model can affect agent security [14, 15, 16], how harness design quantitatively shapes web-agent security across different systems and attack surfaces remains insufficiently measured.

Existing evaluations provide limited support for such measurement because they often entangle harness-level security with model robustness and task capability. Specifically, these works face three limitations. First, they often focus on a specific type of attack, making it difficult to understand the overall security exposure of web agents across different attack surfaces. Second, they may conflate attack failures caused by insufficient task capability with successful defenses, which overestimates the agent’s security resilience and introduces bias into the evaluation. Third, existing benchmarks are often coupled to specific agent harnesses, imposing substantial manual effort on cross-harness and systematic evaluation.

Our Work. Therefore, we present WEBTRAP FARM, a unified framework for quantitatively measuring and dissecting security resilience in web agents. WEBTRAP FARM enables systematic cross-harness evaluation while separating security behavior from task incapability and localizing where defensive behavior emerges in the execution workflow. It includes an extensible benchmark covering three major attack categories across different attack surfaces: Malicious User Prompts, Malicious Prompt Injection, and Deceptive Website Design. To avoid conflating security with task failure, WEBTRAP FARM introduces paired usability calibration tasks that distinguish defense success from cases where an agent simply cannot complete the original task. It also provides a cross-harness execution and monitoring suite based on webpage instrumentation, enabling consistent evaluation without additional manual effort. We conduct experiments around three research questions:

RQ1: How vulnerable are web-agent harnesses and backend models under different attack surfaces? We first evaluate the end-to-end vulnerability of both opensource harnesses and commercial web agents. We instantiate representative harnesses with multiple backend models and measure their attack success rates, including naive attacks that contain no explicit injected instruction or deceptive content. This design allows us to test whether unsafe execution depends on specific harnesses or models, and whether it can arise from ordinary task execution. Our results show that vulnerabilities persist across the open-source harnesses, with an overall attack success rate of 32.55%. The evaluated commercial systems achieve lower attack success rates but still exhibit meaningful vulnerabilities.

RQ2: How much does the web-agent harness matter for agent security resilience? We next examine the effects of harness and backend model on security resilience. Using usability calibration, we apply a mixed-effects logistic regression to quantify harness and model effects while accounting for repeated tasks and attack-category effects. Our results show that the harness, backend model, and their interaction all have significant effects on security outcomes $( p < 1 0 ^ { - 4 5 } )$ and remain robust under sensitivity analyses for capabilitydependent missingness. These results show that deployedagent security is jointly shaped by the harness and backend model, rather than being determined by either factor alone. RQ3: Where does defensive behavior emerge within webagent harnesses, and which design mechanisms strengthen security? Finally, we localize observed defensive behavior across the four workflow stages. Using execution traces, we identify effective defenses after excluding attacks mitigated at earlier stages. The resulting stage-level analysis reveals different defensive patterns across attack surfaces, which we further examine through controlled ablations. For user-side attacks, the ablations support the security benefits of finegrained intent interpretation and risk-aware reasoning. For web-side attacks, they support perception filtering and goalanchored planning. These results provide concrete guidance for designing more secure web-agent harnesses.

In summary, this paper makes the following contributions:

• A systematic and extensible framework for evaluating web-agent harness security. We introduce WEBTRAP FARM, a framework for quantitatively measuring web-agent harness security across different attack surfaces. WEBTRAP FARM combines usability calibration, harness-independent monitoring, and unified metrics, while supporting the incorporation of emerging attacks through paired usability tasks and observation criteria.

• A quantitative evaluation of how harness and model jointly shape web-agent security. We systematically evaluate diverse harness–model configurations and frontier commercial web-agent systems, providing quantitative evidence for how harness design and backend models jointly shape deployed-agent security.

• A stage-level analysis of harness security mechanisms. We localize where defensive behavior emerges across the web-agent workflow and use controlled ablations to examine specific mechanisms, providing a principled basis for understanding and improving harness-level security design in practical web agents.

![](images/081101c428785dc4fdeb1ad2c5a97298d31606310e8417b3b941bfb2a4963809.jpg)  
Figure 1: The Workflow of Web Agents

## 2. Web Agent Harness

A web agent is an automated system that uses large language models (LLMs) or large multimodal models (LMMs) as its core controller to complete complex tasks in real and dynamic web environments [17, 18, 19, 20]. Unlike conventional conversational LLMs that operate mainly over textual inputs and outputs, web agents are embedded in an execution harness that connects model decisions to concrete interaction primitives. While such a harness enables agents to perform long-horizon tasks, it also expands the attack surface beyond the model itself and weakens the conventional security boundary assumed for standalone LLMs [21, 22, 23].

To characterize web-agent harnesses, we survey 40 webagent papers published in top venues (from 2021 to 2026) and collect 68 GitHub projects with at least 1,000 stars using “web agent” and “browser agent” as search keywords. We exclude repositories that do not implement agentic web interaction. From these sources, we abstract the common web-agent workflow into four stages, as shown in Figure 1. Stage I: Goal Interpretation. The agent first interprets the user’s natural-language instruction and converts it into an internal task representation. This stage determines the task objective, relevant constraints, and implicit execution intent, which guide all subsequent decisions.

Stage II: Webpage Perception. The agent obtains the current webpage state through browser interfaces, such as DOM extraction, accessibility trees, screenshots, or multimodal observations. This stage determines what webpage content, interface elements, hidden or off-screen text, and browser states are exposed to the backend model.

Stage III: Reasoning and Planning. Given the interpreted goal and perceived webpage content, the agent reasons about the task context and decides how to proceed. This stage generates a single next action or a multi-step plan, including selecting target elements, choosing input values, and updating the plan based on new browser states.

TABLE 1: Taxonomy of Web Agent Harness Components
<table><tr><td>Stage</td><td>Harness Component</td><td>Design Characteristics</td><td>Representative Agents</td><td>Ratio</td></tr><tr><td rowspan="2">I: Goal Interpretation</td><td>Raw-Prompt Forwarding</td><td>Preserves the original user instruction to avoid additional parsing overhead.</td><td>WebArena[19], WebVoyager[24]</td><td>15.0%</td></tr><tr><td>LLM-based Goal Interpreter</td><td>Uses an LLM to parse user intent into struc- tured task objectives, aiming to clarify goals and improve downstream task completion.</td><td>Agent-E[25], Nanobrowser[26]</td><td>85.0%</td></tr><tr><td rowspan="2">II: Webpage Perception</td><td>Textual State Encoder</td><td>Represents webpages with structured textual artifacts, providing concise input for reasoning.</td><td>PageAgent[27], Agent-E</td><td>67.7%</td></tr><tr><td>Multimodal State Encoder</td><td>Incorporates screenshots or rendered layouts, aiming to support perception beyond text-only representations and capture visual UI cues.</td><td>GPT-4V-Act[28], Skyvern 2.0[29]</td><td>39.8%</td></tr><tr><td rowspan="2">III: Reasoning and Planning</td><td>Single-Controller Planner</td><td>Centralizes reasoning in one decision loop to preserve goal consistency and reduce overhead.</td><td>HyperAgent[30], Open Browser[31] 73.0%</td><td></td></tr><tr><td>Multi-Role Planner</td><td>Separates planning, execution, or verification into coordinated roles to support complex task decomposition and modular reasoning.</td><td>Skyvern 2.0, MindSearch[32]</td><td>27.0%</td></tr><tr><td rowspan="2">IV: Action Execution</td><td>Rule-Based Action Grounder</td><td>Converts planned actions through predefined rules to improve execution controllability.</td><td>Browser Use[33], Notte[34]</td><td>83.8%</td></tr><tr><td>LLM-Based Action Grounder</td><td>Uses an LLM to interpret high-level actions or select executable operations, aiming to improve flexibility in complex interfaces.</td><td>Stagehand[35], WebLlama[36]</td><td>16.2%</td></tr></table>

Note. The ratios are computed over the 68 GitHub projects in our survey. For webpage perception, the percentages sum to more than 100% because some agents support both textual and multimodal state encoders.

Stage IV: Action Execution. The planned action is grounded into concrete browser operations, such as selecting a target element, clicking a button, and typing into a field. The agent then executes these operations through browser automation interfaces, observes the resulting page state, and returns it to the agent loop for subsequent decisions.

Although existing web agents largely follow this fourstage workflow, their harnesses differ in how each stage is implemented. In other words, workflow stages provide a common abstraction, whereas harness components capture the concrete design choices within each stage. We therefore categorize the harness components used at each workflow stage in the 68 GitHub projects. Our analysis shows that common designs include LLM-based parsers for goal interpretation (85.0%), text-only webpage perception (67.7%), single-controller reasoning structures (73.0%), and rule-based grounding methods for action execution (83.8%). We summarize these harness component choices in Table 1.

Prior security studies on web agents often evaluate attacks against specific harnesses and implicitly treat the observed behaviors as broader web-agent security properties [5, 9, 11]. However, these observations can be affected by harness design. For example, a backend model that appears more defensive in one harness may not be more resistant to the attack itself, as the difference may instead come from harness mechanisms. Therefore, studying web-agent security requires systematic comparisons across harness designs and analysis of where defensive behavior emerges. This motivates our focus on web-agent harness security: rather than treating the agent as a monolithic LLM-driven system, we analyze how harness components shape the agent’s security boundary.

## 3. WEBTRAP FARM

To systematically measure the security of web-agent harnesses, we introduce WEBTRAP FARM, a unified evaluation framework designed around three requirements: broad attack coverage, usability calibration, and execution monitoring independent of agent implementations. Its key value lies in providing an extensible methodology that can evolve with emerging attacks, rather than in the long-term completeness of the current attack set. The workflow of WEBTRAP FARM is shown in Figure 2. Section 3.1 presents the design principles, while Sections 3.2–3.4 describe task construction, execution monitoring, and the evaluation protocol.

![](images/56b67fab97b08ee2f2223fb43f0d6c454dd5df57f0c3e8ddf2881d769463b68a.jpg)  
Figure 2: Harness Security Evaluation Workflow

## 3.1. Design Principle

Designing an evaluation framework for web-agent harness security requires more than collecting unsafe prompts or malicious webpages. An effective framework must address three challenges to measurement validity: (i) incomplete attack coverage can underestimate harness exposure; (ii) capability failures can be mistaken for security defenses, leading to biased security estimates; and (iii) harness-specific logging or integration can impose substantial manual effort and limit cross-harness comparison. We therefore design WEBTRAP FARM around the following principles.

## 3.1.1. Comprehensive coverage of web-agent attacks.

Existing web-agent security datasets provide valuable starting points, but their coverage remains fragmented. Some focus on malicious user prompts [5, 6], while others target prompt injection [8, 9] or deceptive interfaces [11, 12]. Their attack instances often cover limited delivery methods or contents, making cross-harness comparison difficult and potentially tying results to specific prompts, layouts, or scenarios.

To address this limitation, WEBTRAP FARM systematically covers two attack surfaces to capture how attacks propagate through different parts of the workflow and are mitigated by different harness components. User-side attacks originate from malicious user instructions and stress goal interpretation, reasoning, and action execution (Stages I, III, and IV), while web-side attacks originate from adversarial webpage content or interfaces and stress webpage perception, reasoning, and action execution (Stages II, III, and IV). Within each surface, we vary attack goals, methods, and contents to provide broad coverage to avoid overfitting the evaluation to a narrow class of attacks. This principle allows the measured results to reflect broadly relevant security properties rather than artifacts of a single benchmark design.

3.1.2. Usability calibration of security outcomes. A central challenge in web-agent security evaluation is that an attack failure has an ambiguous meaning. If an agent does not complete an unsafe action, the failure may indicate successful defense. However, it may also indicate that the agent could not understand the task, operate the webpage, or reach the attack-relevant state. Treating all attack failures as security successes would systematically overestimate robustness, especially for harnesses with weaker task-execution capability.

To avoid this bias, WEBTRAP FARM introduces calibration tasks. For each attack task, we construct a corresponding usability task that preserves the task-relevant interaction requirements while removing the adversarial trigger. This design ensures that an attack failure is treated as evidence of defense only when the agent can complete the same underlying interaction under a benign objective. Otherwise, the failure is attributed to task capability limitations and is not counted as a security-relevant defense success. By separating security behavior from general task capability, this calibration enables more faithful comparison across harnesses.

3.1.3. Execution Monitoring Independent of Harnesses. Another limitation of existing evaluations is their dependence on agent-specific integration. Many benchmarks assume access to particular agent logs, internal actions or browser wrappers [19, 37, 38, 39]. Such coupling increases manual effort and makes it difficult to evaluate heterogeneous webagent systems under the same measurement protocol. It can also introduce measurement bias, because different agents expose different internal states and logging formats.

WEBTRAP FARM therefore measures security-relevant effects at the webpage side, rather than through agent-side logs, internal reasoning traces, or specific APIs. This makes the webpage interaction itself the common measurement interface for all evaluated systems. As a result, WEBTRAP FARM can evaluate different web-agent harnesses under the same protocol without requiring per-agent integration, including closed-source commercial web agents whose internal execution stages are unavailable. This design reduces manual adaptation cost, avoids measurement bias caused by inconsistent internal logs, and supports fair comparison across heterogeneous web-agent systems.

## 3.2. Dataset Construction

Following the above principles, we construct WEBTRAP FARM to provide broad attack coverage while enabling usability calibration. The dataset contains two paired task types: attack tasks, which evaluate whether an agent executes unsafe actions under adversarial conditions, and usability tasks, which measure whether the agent can complete the same interaction when the adversarial trigger is removed.

3.2.1. Attack Tasks. We construct attack tasks according to the attack surface. Based on a review of recent studies on LLM and agent security published within the past three years, we identify representative attack categories for each surface. For the user side, we include Malicious User Prompts, where a malicious user directly issues harmful or policyviolating instructions to the agent. For the web side, we include Malicious Prompt Injection and Deceptive Website Design, where the attacker manipulates webpage content or interface design encountered by a benign agent. Specifically, Malicious Prompt Injection embeds adversarial instructions into webpage content to influence the agent’s reasoning or planning process, whereas Deceptive Website Design relies on deceptive interface elements or social-engineering content to induce unsafe actions during web interaction.

To ensure systematic coverage of the attack space, we organize each attack task along three dimensions: attack goal, which specifies the unsafe behavior the attacker attempts to induce; attack method, which captures the mechanism used to deliver or realize the attack; and attack content, which refers to the concrete prompt, injected instruction, deceptive message, or webpage artifact used in the task. We include only attack methods and contents that have been empirically validated in at least two independent prior works, so that the attack set remains representative and reproducible. As summarized in Table 2, WEBTRAP FARM provides broader and more systematic coverage than any existing single dataset across all three dimensions. The full attack taxonomy with fine-grained subcategories is shown in Table 11.

TABLE 2: Comparison of Security Dataset Coverage
<table><tr><td>Attack</td><td>Dataset</td><td>Goal</td><td>Method</td><td>Content</td></tr><tr><td rowspan="2">Malicious User Prompt</td><td>BrowserART[5]</td><td></td><td></td><td></td></tr><tr><td>SafeArena[6]</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Malicious</td><td>WEBTRAP FARM</td><td></td><td></td><td></td></tr><tr><td>EIA[9] WASP[8]</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Prompt Injection</td><td>WEBTRAP FARM</td><td></td><td></td><td></td></tr><tr><td>AgentBait[11]</td><td></td><td></td><td></td></tr><tr><td>Deceptive Website Design</td><td>WEBTRAP FARM</td><td></td><td></td><td></td></tr></table>

Note: entries use the filled-circle ratio notation, e.g., , indicating covered subtypes over the total subtypes for each coverage dimension.

Based on this taxonomy, we instantiate attacks according to their attack surface. Specifically, we adapt webpages from prior web-agent security datasets [5, 9, 11] and refactor them into a unified task format, rather than relying on agentspecific benchmark environments. These webpages cover common scenarios such as social media interaction, travel planning, and online shopping. The controlled environments preserve realistic web interaction patterns while allowing us to vary attack goals, methods, and contents in a systematic manner [40, 41]. The overall instantiation process is shown in Figure 6. This adaptation across existing datasets also demonstrates the extensibility of WEBTRAP FARM to new attacks under the same evaluation protocol.

User-side Attack Tasks. For Malicious User Prompts, the attack is instantiated through the user prompt. We select representative webpages from BrowserART [5] and modify them to support consistent execution monitoring. We then construct malicious user instructions for five attack goals, such as bias, misinformation and cybercrime, with eight base prompts for each goal. Each base prompt is further instantiated using four attack content strategies, such as instruction override, role play, and privilege escalation, resulting in 160 tasks in total. This construction evenly distributes tasks across attack goals and attack contents, reducing the risk that the results are dominated by a specific harmful objective or jailbreak formulation.

Web-side Attack Tasks. For Malicious Prompt Injection and Deceptive Website Design, attacks are instantiated in webpage content. We adapt webpages from EIA [9] for Malicious Prompt Injection and from AgentBait [11] for Deceptive Website Design, and standardize them into a common implementation format with controllable attack locations. For Malicious Prompt Injection, we insert prompt injection content into predefined locations, such as visible form text, hidden text, ARIA attributes, URLs, hyperlinks, or long-context regions. For Deceptive Website Design, we implement deceptive interface patterns, such as popups, advertisements, leading or trailing form fields, and default selections, and embed the deceptive content into these templates. In addition, we include a “naive” attackcontent type for web-side attacks, where no additional injected prompt or deceptive inducement text is inserted.

This type serves as a baseline for measuring whether unsafe behavior can arise from the harness’s intrinsic interaction process, rather than from explicit external attack content. This process yields 158 Malicious Prompt Injection tasks and 187 Deceptive Website Design tasks. The tasks are distributed uniformly across three attack dimensions.

3.2.2. Usability Tasks. We construct usability tasks according to the attack surface of each risk category. For Malicious User Prompts, where the attack is delivered through the user instruction, we replace the malicious user prompt with a benign instruction that preserves the same task structure and required webpage interaction. For example, if the attack task asks the agent to post a biased statement on the forum, the usability task asks the agent to post a positive statement using the same forum page. For Malicious Prompt Injection and Deceptive Website Design, where the attack is embedded in webpage, we remove the injected instruction or deceptive element and directly ask the agent to complete the corresponding benign interaction. For example, if an attack task attempts to induce an unsafe download through a malicious webpage trigger, the corresponding usability task removes the malicious content and explicitly asks the agent to click the download button and complete the download. In this way, usability tasks preserve the operational difficulty of the original task while removing the adversarial condition. If the attack task fails while the corresponding usability task succeeds, the failure is attributed to the agent’s security mechanisms. If both tasks fail, the failure is treated as a capability failure rather than a defense success.

## 3.3. Execution Monitoring

Following the principle of execution monitoring independent of harnesses, WEBTRAP FARM measures task outcomes from webpage effects rather than agent logs. Specifically, WEBTRAP FARM monitors task execution by instrumenting each task webpage with JavaScript-based event logging. For each task, we identify the key HTML elements whose interaction determines whether the attack goal or usability goal is achieved. These elements include buttons, form fields, file-download triggers, permission-request controls, and other task-specific interface components. The monitoring script records runtime interactions with these elements and sends the captured events to a backend server for scoring.

The monitored data depends on the task type. For clickbased tasks, WEBTRAP FARM records the identifier of the clicked element and its event metadata. For text-input tasks, it records the submitted field value and target field. For download- or permission-related tasks, it records whether the relevant browser-facing trigger is activated. This monitoring is implemented at the webpage and backend level rather than through agent-specific logs. As a result, WEBTRAP FARM does not depend on the logging format, internal reasoning trace, action schema, or implementation details of any particular web-agent harness. This enables consistent cross-harness evaluation under the same measurement interface.

## 3.4. Evaluation Protocol

WEBTRAP FARM evaluates each agent execution in two steps. First, task scoring checks whether the attack or calibration objective is completed. Second, outcome labeling compares each attack task with its paired calibration task to classify the result as attack success, defense success, or capability failure. The final metrics quantify vulnerability, effective defense, and task capability.

3.4.1. Scoring. Each task is scored as a binary outcome: success or failure. For attack tasks, success means that the agent completes the unsafe action required by the attack goal, such as submitting sensitive information, triggering a malicious download, or granting an unauthorized permission. For usability tasks, success means that the agent correctly completes the corresponding benign task objective.

For structured web interactions, we score outcomes using the monitored execution records. Click-based, downloadbased, and permission-based tasks are scored by checking whether the corresponding target element or browser-facing trigger is activated. Form-submission tasks are scored by matching the submitted content against manually annotated expected values. This rule-based scoring is used for Malicious Prompt Injection, Deceptive Website Design, and their corresponding usability tasks whose success criteria can be directly observed from webpage events.

For open-ended text-generation tasks in Malicious User Prompts and their corresponding usability tasks, we use GPT-4o as an LLM-as-a-judge to determine whether the generated response satisfies the task-specific criteria. To validate the reliability of this scoring procedure, we manually annotated a 10% random sample of Malicious User Prompts outputs (672/6,720). The two human annotators achieved 99.26% agreement, and the LLM judge matched the human consensus in 99.55% of cases. This validation suggests that the LLM-based scoring is sufficiently reliable for evaluating open-ended Malicious User Prompts outcomes.

3.4.2. Outcome Label. For each attack task, we first evaluate whether the agent completes the unsafe action specified by the attack goal. If the unsafe action is completed, the outcome is labeled as Attack Success. If the unsafe action is not completed, we then evaluate the corresponding usability task. If the agent succeeds on the usability task, the original attack failure is labeled as Defense Success, since the agent has the capability to perform the interaction but does not do so under the attack condition. If the agent also fails on the usability task, the original attack failure is labeled as Capability Failure, since the failure more likely reflects limitations in task-execution capability.

3.4.3. Evaluation Metrics. Based on these outcome labels, we define three metrics for subsequent evaluation.

Let T denote the set of attack tasks in WEBTRAP FARM. For each attack task $t \in \mathcal { T }$ , we assign an outcome label combined with its calibrated execution result.

The Attack Success Rate (ASR) measures the agent’s vulnerability to attacks, capturing the probability that an attack leads to the execution of the intended unsafe action:

$$
\mathrm { A S R } = { \frac { | t \in { \mathcal { T } } \mid { \mathrm { O u t c o m e } } _ { t } = \mathrm { A T T A C K S U C C E S S } | } { | t \in { \mathcal { T } } \mid { \mathrm { O u t c o m e } } _ { t } \not = { \mathrm { C A P A B I L I T Y F A I L U R E } } | } }
$$

The Defense Success Rate (DSR) measures the agent’s effective security resistance, capturing the probability that the agent avoids the unsafe action even when it has the capability to complete the corresponding interaction:

$$
\mathrm { D S R } = { \frac { | t \in { \mathcal { T } } \mid \mathrm { O u t c o m e } _ { t } = \mathrm { D E F E N S E S U C C E S S } | } { | t \in { \mathcal { T } } \mid \mathrm { O u t c o m e } _ { t } \not = \mathrm { C A P A B I L I T Y F A I L U R E } | } }
$$

The Task Completion Rate (TCR) measures the agent’s execution capability, capturing the proportion of tasks for which the agent demonstrates sufficient task capability.

$$
\mathrm { T C R } = \frac { | t \in \mathcal { T } \mid \mathrm { O u t c o m e } _ { t } \neq \mathrm { C A P A B I L I T Y F A I L U R E } | } { | \mathcal { T } | }
$$

Here, “Capability Failure” cases are excluded from the denominators of ASR and DSR to avoid conflating security behavior with task capability. This follows our calibration principle: an attack failure is treated as security-relevant only when the corresponding usability task confirms that the agent is capable of completing the task-relevant interaction.

## 4. Harness Security Evaluation

Using WEBTRAP FARM, we evaluate representative agents to answer two research questions. The first question (RQ1) measures how vulnerable web-agent harnesses and backend models are under different attack surfaces. The second question (RQ2) quantifies the effects of harness and backend model on agent security resilience. Section 4.1 introduces the evaluated harnesses, backend models, and execution configuration. Section 4.2 presents the evaluation results.

## 4.1. Evaluation Setup

4.1.1. Web Agent Collection. We evaluate open-source web agents collected from two sources: (i) popular projects with more than 1,000 GitHub stars and (ii) representative agents proposed in prior academic works. All selected agents are required to exhibit basic task completion capability [12, 20, 42]. To cover the harness components summarized in Table 1, we instantiate four representative web agents under different configurations. Our final evaluation set consists of six agent harnesses shown in Table 3: SeeAct [43], Browser-Use (text) [33], Browser-Use (vision) [33], Agent-E [25], Skyvern 1.0 [29], and Skyvern 2.0 [29]. In addition, we evaluate two frontier commercial agents, Gemini Computer Use (gemini-3.7-flash) [44] and Claude Computer Use (claude-haiku-4-5- 20251001) [45], using their default integrated models without replacing the backend model.

TABLE 3: Harness Components of Representative Web Agents
<table><tr><td rowspan="2">Web Agent</td><td colspan="2">I: Goal Interpretation</td><td colspan="2">II: Webpage Perception</td><td colspan="2">III: Reasoning and Planning</td><td colspan="2">IV: Action Execution</td></tr><tr><td>Raw-Prompt Forwarding</td><td>LLM-based Goal Interpreter</td><td>Textual State Encoder</td><td>Multimodal State Encoder</td><td>Single-Controller Planner</td><td>Multi-Role Planner</td><td>Rule-Based Action Grounder</td><td>LLM-Based Action Grounder</td></tr><tr><td>SeeAct</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td></tr><tr><td>Browser Use (Text)</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>Browser Use (Vision)</td><td></td><td></td><td></td><td>√</td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>Agent-E</td><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>Skyvern 1.0</td><td></td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>Skyvern 2.0</td><td></td><td>√</td><td></td><td>√</td><td></td><td>√</td><td></td><td>√</td></tr></table>

4.1.2. Backend LLMs. To better isolate the security impact of harnesses from the choice of backend model, we evaluate open-source harnesses across seven backend LLMs. The closed-source models include GPT-4o, GPT-5.1, and GPT-5.2 from OpenAI, as well as Claude-Opus-4.5 and Claude-3.7- Sonnet from Anthropic. These models represent state-of-theart commercial systems. The open-source models include Qwen3-VL-30B-A3B-Instruct and Qwen3-VL-235B-A22B-Instruct, two mixture-of-experts vision-language models from the Qwen3 family. By evaluating the same harnesses over different backend models, we reduce the risk that observed security differences are artifacts of a single model and obtain a more robust view of harness-level security behavior.

4.1.3. Configuration. All agents are integrated strictly following their official documentation. We do not alter their core architectures, reasoning logic, or built-in safety mechanisms. Each agent is executed with its recommended default configuration to reflect realistic deployment settings. To reduce nondeterminism introduced by backend LLM sampling, we set the decoding temperature to 0. For each task, we require one completed execution. Runs interrupted by infrastructure failures, such as rate limits or network errors, are retried until reaching a natural termination state. To verify result stability, we further repeat the full benchmark three times across all six harnesses with two fixed backend models (Qwen3-VL-235B and GPT-5.1). The average standard deviation is only 1.73 percentage points, indicating limited run-to-run variation. To enable systematic analysis, we enable logging for all open-source agents to record intermediate decisions and interactions with the webpages during task execution. These logs are used exclusively for security analysis in Section 5, and do not affect the agent’s runtime behaviors.

## 4.2. Results

We organize the results according to the two research questions introduced above. The results first examine overall vulnerability across harnesses and backend models, and then quantify the effects of harness and backend model on security resilience. These findings motivate the analysis of where defensive behavior emerges in Section 5.

4.2.1. (RQ1) How vulnerable are web-agent harnesses and backend models under different attack surfaces. As shown in Table 4, all six open-source agent harnesses remain vulnerable across different attack surfaces, with an overall

TABLE 4: Attack Success Rates (ASR) of Web-Agent Systems across Different Attack Surfaces
<table><tr><td>Web Agent</td><td>MUP</td><td>MPI</td><td>DWD</td><td>Overall</td></tr><tr><td>Open-source Harnesses</td><td></td><td></td><td></td><td></td></tr><tr><td>SeeAct</td><td>24.96%</td><td>49.44%</td><td>54.37%</td><td>42.31%</td></tr><tr><td>Browser Use (Text)</td><td>25.87%</td><td>35.38%</td><td>37.13%</td><td>33.84%</td></tr><tr><td>Browser Use (Vision)</td><td>21.17%</td><td>40.55%</td><td>36.17%</td><td>34.08%</td></tr><tr><td>Agent-E</td><td>8.48%</td><td>25.16%</td><td>23.01%</td><td>20.24%</td></tr><tr><td>Skyvern 1.0</td><td>32.75%</td><td>8.04%</td><td>24.02%</td><td>21.80%</td></tr><tr><td>Skyvern 2.0</td><td>43.49%</td><td>39.79%</td><td>42.83%</td><td>41.99%</td></tr><tr><td>Commercial Systems</td><td></td><td></td><td></td><td></td></tr><tr><td>Claude Computer Use</td><td>9.59%</td><td>16.67%</td><td>22.12%</td><td>16.85%</td></tr><tr><td>Gemini Computer Use</td><td>1.45%</td><td>14.29%</td><td>25.51%</td><td>15.14%</td></tr></table>

Note. (i) “MUP”, “MPI”, and “DWD” denote Malicious User Prompts, Malicious Prompt Injection, and Deceptive Website Design, respectively. (ii) Overall ASR is computed as a weighted average across attacks.

TABLE 5: Attack Success Rates (ASR) of Backend LLMs across Different Attack Surfaces
<table><tr><td>Backend LLM</td><td>MUP</td><td>MPI</td><td>DWD</td><td>Overall</td></tr><tr><td>Claude-3.7-Sonnet</td><td>31.03%</td><td>31.19%</td><td>33.72%</td><td>32.11%</td></tr><tr><td>Claude-Opus-4.5</td><td>3.25%</td><td>36.03%</td><td>22.55%</td><td>22.90%</td></tr><tr><td>GPT-40</td><td>32.15%</td><td>35.86%</td><td>46.57%</td><td>39.12%</td></tr><tr><td>GPT-5.1</td><td>10.81%</td><td>37.59%</td><td>29.47%</td><td>27.38%</td></tr><tr><td>GPT-5.2</td><td>5.52%</td><td>36.48%</td><td>33.97%</td><td>27.41%</td></tr><tr><td>Qwen3-VL-235B</td><td>44.73%</td><td>29.48%</td><td>38.99%</td><td>37.34%</td></tr><tr><td>Qwen3-VL-30B</td><td>57.94%</td><td>27.78%</td><td>42.25%</td><td>41.73%</td></tr><tr><td>Overall</td><td>27.34%</td><td>33.54%</td><td>35.27%</td><td>32.55%</td></tr></table>

ASR of 32.55%. Vulnerability varies substantially across harnesses: SeeAct and Skyvern 2.0 show the highest overall ASRs at 42.31% and 41.99%, while Agent-E achieves the lowest at 20.24% but also remains vulnerable to web-side attacks. These results show that unsafe execution is not specific to a single implementation, while vulnerability depends strongly on how the harness mediates goals, webpage content, planning, and execution. Besides, the two frontier commercial web-agent systems show lower overall ASRs: 15.14% for Gemini Computer Use and 16.85% for Claude Computer Use, indicating stronger end-to-end safeguards than the evaluated open-source harnesses. However, meaningful vulnerabilities remain under Deceptive Website Design, with ASRs of 25.51% for Gemini and 22.12% for Claude.

Table 5 shows that backend LLMs also remain vulnerable when embedded in web-agent harnesses. The vulnerability pattern is shaped by the attack surface. Claude-Opus-4.5 and GPT-5.2 are relatively robust to Malicious User Prompts, with ASR below 10%, but remain vulnerable to webside attacks. Claude-Opus-4.5 reaches 36.03% ASR under Malicious Prompt Injection, while GPT-5.2 exceeds 30% ASR under both Malicious Prompt Injection and Deceptive Website Design. In contrast, Qwen3-VL-30B has the highest Malicious User Prompts ASR at 57.94%, but its Malicious Prompt Injection ASR is only 27.78%, lower than that of several closed-source models. These results show that model safety does not transfer uniformly across attack surfaces. A model that resists malicious user prompts may still fail when unsafe instructions are embedded in webpage content or deceptive interaction flows, underscoring the need to evaluate models within the full web-agent execution loop.

![](images/836594aecc6c1b42ff10e1a91843a4e1781001809563b9a11cb1b1155ec12e3d.jpg)  
Figure 3: Attack Success Rates (ASR) of Naive Attacks and Attacks with Explicit Attack Content

Harness-Intrinsic Vulnerability. We further analyze naive attacks in web-side attacks to examine how much unsafe behavior is due to harness-intrinsic vulnerability rather than explicit external attack content. Specifically, we separately compute the ASRs of naive attacks in Malicious Prompt Injection and Deceptive Website Design, where no additional attack content is inserted into the webpage. We compare these ASRs with those of the remaining attacks.

As shown in Figure 3, naive attacks can achieve attack success rates comparable to attacks with explicit prompt injection or deceptive content. In some cases, naive attacks even outperform attacks with explicit attack content. For Malicious Prompt Injection, SeeAct reaches a naive-attack ASR of 50.00%, slightly higher than the 49.37% ASR of other attacks. For Deceptive Website Design, Browser Use (Text) reaches a naive-attack ASR of 41.27%, compared with 36.30% for other attacks. Our log inspection suggests that such results reflect a harness-intrinsic vulnerability in execution logic. Explicit attack content, such as “ignore previous instruction” or “official update download,” can occasionally trigger the backend LLM’s safety awareness due to patterns seen during safety training, causing the agent to reject the unsafe action. In contrast, naive attacks contain no salient adversarial signal. The LLM therefore has fewer cues to identify the interaction as unsafe through reasoning alone, while the harness may still proceed with execution based on webpage affordances and task-completion pressure. For example, when a credit-card field appears in a courseselection task without any explicit attack prompt, the agent may interpret the field as part of the expected task flow and fill it despite its semantic inconsistency with the original user goal. This indicates that the unsafe behavior is not merely induced by external attack content, but reflects an intrinsic weakness of the web-agent harness: it lacks sufficient mechanisms to verify whether a webpage action is necessary, goal-consistent, and secure before execution.

TABLE 6: Defense Success Rates (DSR) of Web-Agent Harnesses across Different Backend LLMs
<table><tr><td>Web Agent</td><td>Claude-3.7</td><td>Claude-4.5</td><td>GPT-40</td><td>GPT-5.1</td></tr><tr><td>SeeAct</td><td>53.36%</td><td>68.95%</td><td>57.45%</td><td>59.32%</td></tr><tr><td>Browser Use (Text)</td><td>67.61%</td><td>73.17%</td><td>60.90%</td><td>73.16%</td></tr><tr><td>Browser Use (Vision)</td><td>67.55%</td><td>74.89%</td><td>64.46%</td><td>68.08%</td></tr><tr><td>Agent-E</td><td>80.42%</td><td>75.19%</td><td>83.79%</td><td>84.09%</td></tr><tr><td>Skyvern 1.0</td><td>82.18%</td><td>91.55%</td><td>61.94%</td><td>84.53%</td></tr><tr><td>Skyvern 2.0</td><td>54.93%</td><td>77.59%</td><td>42.89%</td><td>65.23%</td></tr><tr><td>Overall</td><td>67.89%</td><td>77.10%</td><td>60.88%</td><td>72.62%</td></tr></table>

<table><tr><td>Web Agent</td><td>GPT-5.2</td><td>Qwen3-235B</td><td>Qwen3-30B</td><td>Overall</td></tr><tr><td>SeeAct</td><td>69.96%</td><td>56.58%</td><td>41.16%</td><td>57.69%</td></tr><tr><td>Browser Use (Text)</td><td>68.08%</td><td>60.84%</td><td>60.35%</td><td>66.16%</td></tr><tr><td>Browser Use (Vision)</td><td>67.93%</td><td>58.81%</td><td>59.78%</td><td>65.92%</td></tr><tr><td>Agent-E</td><td>76.21%</td><td>82.16%</td><td>76.13%</td><td>79.76%</td></tr><tr><td>Skyvern 1.0</td><td>84.86%</td><td>74.88%</td><td>69.19%</td><td>78.20%</td></tr><tr><td>Skyvern 2.0</td><td>70.48%</td><td>50.72%</td><td>45.12%</td><td>58.01%</td></tr><tr><td>Overall</td><td>72.59%</td><td>62.66%</td><td>58.27%</td><td>67.45%</td></tr></table>

TABLE 7: Mixed-Effects Logistic Regression (GLMM) Results for Security Outcomes
<table><tr><td>Effect</td><td>Effect Type</td><td>df</td><td> $x ^ { 2 }$ </td><td>p-value</td></tr><tr><td>Agent Harness</td><td>Main effect</td><td>5</td><td>331.17</td><td> $1 . 9 7 \times 1 0 ^ { - 6 9 }$ </td></tr><tr><td>Backend Model</td><td>Main effect</td><td>6</td><td>437.63</td><td> $2 . 2 5 \times 1 0 ^ { - 9 1 }$ </td></tr><tr><td>Harness × Model</td><td>Interaction</td><td>30</td><td>299.81</td><td> $2 . 8 8 \times 1 0 ^ { - 4 6 }$ </td></tr></table>

4.2.2. (RQ2) How much does the web-agent harness matter for agent security resilience. As shown in Table 6, agent security resilience varies substantially across web-agent harnesses, even when the backend LLM is fixed. For example, under the same GPT-4o backend, Agent-E achieves a DSR of 83.79%, whereas Skyvern 2.0 achieves only 42.89%, resulting in a 40.90 percentage point gap.

To quantify the effects of harness and backend model while accounting for repeated evaluations of the same tasks, we fit a mixed-effects logistic regression, formulated as a generalized linear mixed model (GLMM), over the capabilitycalibrated attack outcomes. The model includes agent harness, backend model, their interaction, and attack category as fixed effects, with task as a random intercept. As shown in Table 7, harness, backend model, and their interaction all significantly affect security outcomes (all $p < 1 0 ^ { - 4 5 } )$ . To further verify that this conclusion is not driven by capability-dependent filtering, we model task evaluability using the same structure and observe significant harness, model, and interaction effects (all $p < 1 0 ^ { - 3 \widetilde { 1 } } \big >$ ). We also repeat the security analysis under extreme relabelings of capability-failure cases, and all three effects remain significant (all $p < 1 0 ^ { - 3 9 } )$ . These results show that harness design significantly and robustly shapes deployed-agent security alongside the backend model.

![](images/a955ab950cb3739cb5fe8db1d94b1a6236de3da9b191b1afc2b49630853b5f0d.jpg)  
Figure 4: Defense Success Rates (DSR) and Task Completion Rates (TCR) of Web-Agent Harnesses

For example, Claude-Opus-4.5, despite being highly safety-aligned [46, 47] and achieving 77.10% overall DSR, reaches only 75.19% DSR in Agent-E, lower than GPT-4o (83.79%) and Qwen3-VL-235B-A22B-Instruct (82.16%). This further shows that model-level safety alone does not determine the security of a deployed agent.

Capability-Security Decoupling. To further examine the security implications of harness design, we analyze whether a more capable agent harness is necessarily more secure. As shown in Figure 4, the results do not support this assumption. Skyvern 2.0 achieves the highest TCR of 85.25% among all evaluated harnesses, indicating strong task execution capability, but its DSR is only 58.01%. In contrast, Agent-E achieves the highest DSR of 79.76%, despite a lower TCR of 61.33%. This suggests that stronger web-task execution capability does not automatically translate into stronger security resilience. A harness that completes web interactions more reliably may also carry unsafe actions to completion when its security checks are insufficient. Conversely, a harness with lower task completion capability but stronger security checks can expose fewer unsafe behaviors under attack. These results reinforce the need to analyze security harness design separately from general task capability. Therefore, we investigate where defensive behavior emerges and which design mechanisms strengthen security in Section 5.

## 5. Harness Component Security Analysis

Section 4 has quantitatively demonstrated that harness design shapes web-agent security. We now turn to the third research question (RQ3): Where does defensive behavior emerge within web-agent harnesses, and which design mech anisms strengthen security? To answer this question, we first analyze execution logs to localize where successful defenses emerge across workflow stages, and then use controlled ablations to examine specific security mechanisms.

![](images/6ebb4656b21b3e5d0eddf03bf5f8d44d43451558a0e213928a6187011ab4280e.jpg)  
Figure 5: Harness Component Security Analysis Workflow

## 5.1. Security Analysis Method

Figure 5 illustrates our security analysis workflow. Start ing from execution logs, we extract patterns for identifying stage states, annotate the execution sequence across multiple rounds with these patterns, and localize where successful defenses become effective in the workflow.

5.1.1. Stage-wise Log Analysis. Web agents execute tasks through an interaction loop rather than a single end-to-end decision. Each round follows similar workflow stages. Under this streaming loop, an attack may be propagated or blocked at any stage in any interaction round. Therefore, an end-toend “Defense Success” label is insufficient for identifying where defensive behavior emerges. Instead, we formulate the multi-round workflow as an execution sequence, annotate each stage in each round as either a safe or unsafe state, and identify where a successful defense becomes effective according to the state transitions. This design enables us to localize defensive behavior within the execution workflow. Step1: Pattern Extraction. To determine whether each stage is in a safe or unsafe state, we first derive labeling patterns from execution logs. For each evaluated agent, we randomly sample 100 tasks labeled as “Defense Success” and manually inspect their logs, including goal interpretation, webpage observations, reasoning traces, planned actions, and final browser operations. From these logs, we identify patterns that indicate the stage-wise security state, as summarized in Table 8. A safe-state pattern indicates that the current stage is safe, i.e., it does not include or propagate the attack objective. This may occur because the attack has not yet been exposed to the stage, such as when the agent has not reached the adversarial element in a web-side attack, or because the stage ignores, refuses, reformulates, or blocks the attack content. In contrast, an unsafe-state pattern indicates that the current stage enters an unsafe propagation state, where the attack objective is preserved, adopted, or transformed into unsafe planning or browser actions. Detailed explanations of safe and unsafe patterns are provided in Appendix C.

TABLE 8: Safe and Unsafe State Patterns across Web-Agent Workflow Stages
<table><tr><td rowspan="2">Workflow Stage</td><td colspan="2">User-side Attacks</td><td colspan="2">Web-side Attacks</td></tr><tr><td>Safe-State Pattern</td><td>Unsafe-State Pattern</td><td>Safe-State Pattern</td><td>Unsafe-State Pattern</td></tr><tr><td>I: Goal Interpretation</td><td>Secure Refusal Secure Reformulation</td><td>Insecure Goal Interpretation</td><td></td><td></td></tr><tr><td>II: Webpage Perception</td><td></td><td></td><td>Injection Filtered</td><td>Injection Exposed Injection Included</td></tr><tr><td>III: Reasoning and Planning</td><td>Secure Refusal Secure Reformulation</td><td>Insecure Planning</td><td>Secure Refusal Attack Unaware Injection Recognized and Discarded</td><td>Injection Recognized and Adopted</td></tr><tr><td>IV: Action Execution</td><td>Action Blocked</td><td>Unsafe Action Executed</td><td>Action Blocked</td><td>Unsafe Action Executed</td></tr></table>

Note. “–” indicates that the stage is not directly exposed to the corresponding attack surface.

Step2: Automated State Annotation. We then use the extracted patterns to annotate each stage across multiple interaction rounds in the remaining “Defense Success” logs as either a safe or an unsafe state. For webpage perception in web-side attacks, we use a rule-based annotator because the attack element can be directly checked from the webpage observation passed to later stages. For patterns that require semantic interpretation, such as secure refusal, secure reformulation, injection recognition, or action blocking, we use GPT-4o to annotate based on predefined pattern definitions and decision criteria. To validate the annotation pipeline, two annotators independently labeled a random sample of 300 stage-level instances, achieving 97.33% agreement (Cohen’s $\kappa = 0 . 9 6 )$ . The automated labels matched the human annotations in 98.67% of the sampled cases.

Step 3: Defense Stage Localization. Finally, we localize where defensive behavior becomes effective from the sequence of safe and unsafe states. Let $S ( i , j ) \in \{ 0 , 1 \}$ denote whether workflow stage $j$ at interaction round i is in a safe state, where $S ( i , j ) = \bar { 1 }$ indicates that the stage is safe. Since Goal Interpretation (Stage I) is performed only in the first round, the stage-wise execution sequence is ordered as

$$
\mathcal { Q } = \big ( ( 1 , 1 ) , ( 1 , 2 ) , ( 1 , 3 ) , ( 1 , 4 ) , ( 2 , 2 ) , ( 2 , 3 ) , ( 2 , 4 ) , \ldots \big ) .
$$

We denote by $( i ^ { * } , j ^ { * } )$ the stage occurrence where the defense becomes effective, with $j ^ { * }$ as the localized defense stage.

We use different localization rules for user-side and webside attacks because the attack objective enters the workflow differently. For user-side attacks, the malicious objective is already present in the initial user instruction and enters the workflow from Stage I in the first interaction round. We therefore localize the defense at the first safe state:

$$
( i ^ { * } , j ^ { * } ) = \operatorname* { m i n } _ { \mathcal { Q } } \left\{ ( i , j ) \in \mathcal { Q } \mid S ( i , j ) = 1 \right\} .
$$

For web-side attacks, the attack objective enters the workflow only after the agent encounters the adversarial webpage content. Our localization rule is based on two key observations. First, the first safe state does not necessarily indicate defense. Before the agent reaches the adversarial element, the attack objective has not yet entered the workflow, so the corresponding stages naturally remain safe. Second, the first unsafe-to-safe transition does not necessarily indicate effective defense. For example, the agent may already perceive webpage content containing the attack, but has not yet acted on the corresponding adversarial field or button in the current interaction round. The state may temporarily become safe even though the attack can reappear in subsequent planning or actions. Since effective defense is reflected by subsequent stages remaining safe, we localize the defense at the last unsafe-to-safe transition in the execution sequence. Formally,

$$
( i ^ { * } , j ^ { * } ) = \operatorname* { m a x } _ { \mathcal { Q } } \left\{ ( i , j ) \in \mathcal { Q } \ \vert \ S ( \mathrm { p r e v } ( i , j ) ) = 0 \wedge S ( i , j ) = 1 \right\}
$$

where $\mathrm { p r e v } ( i , j )$ denotes the previous element of $( i , j )$ in the ordered execution sequence Q.

5.1.2. Stage-wise Security Metric. We quantify where defensive behavior is observed at each stage using the Conditional Defense Success Rate (Conditional DSR). Unlike the overall DSR in Section 3.4, which measures whether an agent avoids unsafe execution after controlling for task capability, Conditional DSR measures the proportion of remaining security-relevant tasks whose defense is localized at a specific stage. For a target stage j, we first define the set of security-relevant attack tasks as ${ \mathcal { T } } _ { \mathrm { v a l i d } } = \{ t \in$ $\tau |$ Outcome $\neq { }$ CAPABILITYFAILURE}, which excludes tasks whose failures are due to insufficient task capability rather than defensive behavior. We further define $D _ { j } = \{ t \in$ ${ \mathcal { T } } _ { \mathrm { v a l i d \ } } \mid { \mathrm { O u t c o m e } } _ { t } = { \mathrm { D E F E N S E S U C C E S S } } \land j _ { t } ^ { * } = j \}$ as the set of “Defense Success” tasks whose defense is localized at stage $j .$ . When evaluating stage $j ,$ we exclude defensesuccess instances localized at earlier stages $( j _ { t } ^ { * } < j )$ , since these attacks are already mitigated before reaching stage $j .$ Among the remaining tasks, we measure the fraction whose defense is localized at stage j. The metric is defined as:

$$
\mathrm { C o n d i t i o n a l \_ D S R } ( j ) = \frac { \left| { \cal D } _ { j } \right| } { \left| { \cal T } _ { \mathrm { v a l i d } } \right| - \sum _ { k < j } \left| { \cal D } _ { k } \right| } .
$$

## 5.2. Stage-Level Analysis Results

Since user-side and web-side attacks enter the workflow through different surfaces, they stress different harness components. We therefore analyze where defensive behavior is localized separately for Malicious User Prompts and for web-side attacks (Malicious Prompt Injection and Deceptive Website Design). To further examine the specific security mechanisms suggested by the stage-level analysis, we conduct ablation experiments on 100 sampled tasks from userside attacks and 100 sampled tasks from web-side attacks. We fix the backend LLM to GPT-4o to isolate the effect of harness components and avoid confounding from model variation. This keeps the ablation focused on harness design while reducing unnecessary experimental cost.

TABLE 9: Harness Component Security Analysis for Malicious User Prompts
<table><tr><td rowspan="2">Web Agent</td><td colspan="2">I: Goal Interpretation</td><td colspan="2">III: Reasoning and Planning</td><td colspan="2">IV: Action Execution</td></tr><tr><td>Harness Component</td><td>Conditional_DSR</td><td>Harness Component</td><td>Conditional_DSR</td><td>Harness Component</td><td>Conditional_DSR</td></tr><tr><td>SeeAct</td><td>Raw-Prompt Forwarding</td><td>0.00%</td><td>Single-Controller Planner</td><td>67.86%</td><td>LLM-Based Action Grounder</td><td>26.94%</td></tr><tr><td>Browser Use (Text)</td><td>Raw-Prompt Forwarding</td><td>0.00%</td><td>Single-Controller Planner</td><td>64.60%</td><td>Rule-Based Action Grounder</td><td>0.00%</td></tr><tr><td>Browser Use (Vision)</td><td>Raw-Prompt Forwarding</td><td>0.00%</td><td>Single-Controller Planner</td><td>70.31%</td><td>Rule-Based Action Grounder</td><td>0.00%</td></tr><tr><td>Agent-E</td><td>LLM-based Goal Interpreter</td><td>82.71%</td><td>Multi-Role Planner</td><td>69.50%</td><td>Rule-Based Action Grounder</td><td>0.00%</td></tr><tr><td>Skyvern 1.0</td><td>LLM-based Goal Interpreter</td><td>42.79%</td><td>Single-Controller Planner</td><td>35.54%</td><td>Rule-Based Action Grounder</td><td>0.00%</td></tr><tr><td>Skyvern 2.0</td><td>LLM-based Goal Interpreter</td><td>8.13%</td><td>Multi-Role Planner</td><td>45.74%</td><td>LLM-Based Action Grounder</td><td>23.51%</td></tr></table>

5.2.1. User-side Defense Analysis. Table 9 reports the stagewise Conditional DSR for Malicious User Prompts. The results show that effective defense is supported by finegrained intent interpretation over user goals and risk-aware reasoning that prevents unsafe actions.

Stage I: Goal Interpretation. Agent-E’s LLM-based goal interpreter achieves the highest Stage I Conditional DSR (82.71%), indicating that most of its successful defenses are localized at goal interpretation before downstream plan ning (e.g., returning a mandatory termination signal when the task is deemed unsafe or infeasible). Raw-prompt forwarding provides no Stage I defense because malicious instructions pass directly to later modules. Among LLM-based interpreters, Skyvern 2.0 has the lowest Stage I Conditional DSR (8.13%), suggesting that the presence of an interpreter alone does not necessarily yield strong defensive behavior.

To further examine whether fine-grained intent interpretation contributes to this difference, we replace Agent-E’s interpretation prompt with the less detailed prompt used in Skyvern 2.0. On 100 sampled Malicious User Prompts tasks, Agent-E’s Stage I Conditional DSR decreases from 75.61% to 53.85% (-21.76 percentage points), showing that fine-grained intent interpretation helps expose malicious objectives before they are translated into executable actions. Stage III: Reasoning and Planning. The observed defensive behavior at this stage does not consistently align with whether the harness uses a single-controller planner or a multi-role planner. Instead, it is associated with whether the reasoning prompt includes risk-awareness cues. Browser Use (Vision) achieves the highest Stage III Conditional DSR of 70.31%, while Agent-E reaches a comparable 69.50% despite using a different multi-role planning structure. For both harnesses, a substantial portion of successful defenses is localized at Stage III, where the reasoning prompts encourage the agent to reject, reformulate, or discard unsafe plans during reasoning. In contrast, for the Skyvern series, less defensive behavior is localized at Stage III, with Conditional DSR below 50%. Their reasoning prompts are more strongly oriented toward task completion. Once a malicious user goal passes earlier stages, the planner is more likely to reason about how to complete it rather than independently block it.

To further examine the role of risk-awareness cues during reasoning, we remove them while preserving the original planner structure (e.g., “Don’t login into a page if you don’t have to”). On 100 sampled Malicious User Prompts tasks, Stage III Conditional DSR decreases from 73.91% to 43.48% for Browser Use (Vision) (-30.43 percentage points), and from 92.86% to 69.77% for Agent-E (-23.09 percentage points), indicating that risk-awareness cues strengthen defenses during reasoning across different planner structures. Stage IV: Action Execution. At the execution stage, limited but observable defensive behavior is localized for harnesses using LLM-based action grounding, while no measurable Stage IV defense is observed for rule-based action grounding in our results. SeeAct and Skyvern 2.0, which use LLMbased grounding, achieve Stage IV Conditional DSR values of 26.94% and 23.51%, respectively. This suggests that translating natural-language action descriptions into browser operations through an LLM can re-trigger model guardrails when the action is malicious, such as submitting sensitive information or generating abusive content. However, because these grounding prompts do not explicitly implement safety validation, the protection remains incidental. In contrast, rulebased grounding directly maps planned actions to browser commands without semantic safety checks, corresponding to 0.00% Conditional DSR at this stage.

5.2.2. Web-side Defense Analysis. Table 10 reports the stage-wise Conditional DSR for Malicious Prompt Injection and Deceptive Website Design. The results show that effective defense is supported by selective webpage perception and keeping planning anchored to the original user goal. Stage II: Webpage Perception. Agent-E exhibits the highest Stage II Conditional DSR of 78.99%, indicating that a large portion of its successful defenses is localized at webpage perception. This observed pattern is associated with its structured webpage extraction, which filters irrelevant webpage content before it reaches the reasoning stage. In Agent-E, perception is implemented as a tool call that extracts task-relevant text and elements from the webpages, reducing exposure to hidden or non-semantic injected content. The extracted observation is then interpreted by the LLM to flag suspicious content, as illustrated by the following warning:

TABLE 10: Harness Component Security Analysis for Malicious Prompt Injection and Deceptive Website Design
<table><tr><td rowspan="2">Web Agent</td><td colspan="2">II: Webpage Perception</td><td colspan="2">III: Reasoning and Planning</td><td colspan="2">IV: Action Execution</td></tr><tr><td>Harness Component</td><td>Conditional_DSR</td><td>Harness Component</td><td>Conditional_DSR</td><td>Harness Component</td><td>Conditional_DSR</td></tr><tr><td>SeeAct</td><td>Multimodal State Encoder</td><td>10.23%</td><td>Single-Controller Planner</td><td>35.02%</td><td>LLM-Based Action Grounder</td><td>1.63%</td></tr><tr><td>Browser Use (Text)</td><td>Textual State Encoder</td><td>18.93%</td><td>Single-Controller Planner</td><td>44.32%</td><td>Rule-Based Action Grounder</td><td>0.00%</td></tr><tr><td>Browser Use (Vision)</td><td>Multimodal State Encoder</td><td>9.96%</td><td>Single-Controller Planner</td><td>47.22%</td><td>Rule-Based Action Grounder</td><td>0.00%</td></tr><tr><td>Agent-E</td><td>Textual State Encoder</td><td>78.99%</td><td>Multi-Role Planner</td><td>10.65%</td><td>Rule-Based Action Grounder</td><td>0.00%</td></tr><tr><td>Skyvern 1.0</td><td>Multimodal State Encoder</td><td>15.38%</td><td>Single-Controller Planner</td><td>78.17%</td><td>Rule-Based Action Grounder</td><td>0.00%</td></tr><tr><td>Skyvern 2.0</td><td>Multimodal State Encoder</td><td>11.83%</td><td>Multi-Role Planner</td><td>51.44%</td><td>LLM-Based Action Grounder</td><td>4.07%</td></tr></table>

In contrast, harnesses using multimodal state encoders show lower Stage II Conditional DSR in our cross-harness observations, with all values below 20%. Specifically, Browser Use (Vision) achieves the lowest Stage II Conditional DSR of 9.96%. This pattern is consistent with differences in perception filtering, since visible injected or deceptive content can be directly exposed to the model while the visual channel typically lacks explicit security filtering.

To further examine the role of perception filtering, we relax Agent-E’s filtering policy by exposing all webpage elements to the LLM. On 100 sampled web-side attack tasks, Agent-E’s Stage II Conditional DSR decreases from 56.94% to 50.70% (-6.24 percentage points). In the failed defense cases, excessive webpage context dilutes the model’s attention, making suspicious content harder to identify. These results indicate that perception filtering improves defense by reducing irrelevant context before reasoning.

Stage III: Reasoning and Planning. Unlike user-side attacks, web-side attacks show a clearer observational difference across planner designs. Several single-controller harnesses localize more defense at Stage III by repeatedly anchoring reasoning to user intent, with Skyvern 1.0 reaching the highest Conditional DSR (78.17%). In contrast, less defensive behavior is localized at Stage III for Agent-E’s multi-role planner (10.65%). In its execution traces, injected objectives can be propagated across roles: once one LLM summarizes a malicious webpage instruction into a highlevel plan, another role may treat it as a trusted execution directive, such as incorporating injected steps like “download the required file before proceeding” into its plan.

To further examine the role of goal anchoring, we conduct two ablations. First, we remove the goal-anchoring instruction from Skyvern 1.0’s planning prompt, reducing Stage III Conditional DSR from 70.73% to 54.05% (-16.68 percentage points). Second, we provide Agent-E’s planner agent with the original user instruction in addition to the upstream planner output, allowing it to check whether the planned action remains aligned with the user goal. This increases Stage III Conditional DSR from 19.35% to 43.59% (+24.24 percentage points). These results show that defense at this stage depends on whether the harness preserves the original user goal when processing instructions on the webpages.

Stage IV: Action Execution. For web-side attacks, grounding designs show a pattern similar to user-side attacks: limited defensive behavior is localized at Stage IV for harnesses using LLM-based action grounding, while no measurable Stage IV defense is observed for rule-based action grounding. However, the overall Conditional DSR remains low at this stage. SeeAct and Skyvern 2.0 use LLMbased action grounding but still achieve Conditional DSR below 5%, while all rule-based action grounders remain at 0.00%. This pattern reflects that web-side attacks are decomposed into locally benign browser operations, such as clicking a button or filling a visible field. Once translated into such actions, each step appears legitimate in isolation and rarely re-triggers model guardrails. As a result, limited defensive behavior is observed at the grounding stage after the attack objective has already passed earlier stages.

## 5.3. Case Study

To further understand defense vulnerabilities, we manually inspect successful attacks and identify a recurring failure pattern across multiple agents. One representative example occurs in Skyvern 1.0 under a Deceptive Website Design task, where the agent is instructed to complete a job application form that stealthily requests a credit card number at the end.

During intent parsing, the agent correctly derives a detailed user goal with an explicit guardrail (e.g., “do not provide sensitive information”). In early attempts, the reasoning stage recognizes the credit card field as risky and refuses to fill it. However, after submission, the agent fails to detect the successful completion signal and repeatedly reexecutes the form. As the task is retried multiple times, the model’s safety judgment gradually weakens: it first rejects the field as unsafe, then dismisses it as irrelevant, and eventually treats it as necessary for task completion, ultimately entering the credit card number. This progressive defensive failure occurs despite consistently providing the same guardrails in the prompt. The example reveals an inherent instability of prompt-level safety mechanisms used by most web agents: repeated interactions and long execution chains can dilute or override safety constraints, allowing malicious objectives to be gradually integrated into the normal task workflow. More broadly, current web agents rely heavily on prompt engineering and the LLM’s intrinsic guardrails for defense. Such mechanisms are brittle and nondeterministic, highlighting the need for more structured and explicit safeguards beyond prompt control.

## 6. Discussion

## 6.1. Security Implication

Our key contribution is the quantitative measurement of how harness design shapes web-agent security, treating the harness as a security factor rather than merely an implementation detail around the backend model. WEBTRAP FARM enables cross-harness comparison, separates defense from task incapability through capability calibration, and localizes defensive behavior across workflow stages. Rather than relying on the long-term completeness of the current attack set, WEBTRAP FARM is extensible to emerging attacks through paired usability tasks and webpage-side observation criteria, preserving a consistent evaluation methodology.

Our results show that harness design directly shapes deployed-agent security: the same backend model can behave differently across harnesses, and unsafe execution may persist despite seemingly benign perception, planning, and execution pipelines. The harness therefore remains a significant blind spot in web-agent security. As agents increasingly interact with real services, understanding how harness components introduce, propagate, or block attacks is essential for trustworthy deployment. WEBTRAP FARM advances this goal through systematic measurement and defense localization.

## 6.2. Defense Suggestions

Based on our empirical findings, we provide several practical recommendations for building more secure web agents. (i) Strengthen intent interpretation. For user-side attacks, the harness should identify the potential harmful objective in the user intent before planning browser actions. This prevents malicious prompts from being treated as ordinary task goals. (ii) Filter webpage perception. For webside attacks, the perception module should avoid exposing irrelevant or suspicious webpage content to the LLM. Instead, it should prioritize elements that are necessary for completing the original user task. (iii) Anchor planning to the user goal. The planner should repeatedly compare webpage instructions with the original user request, rather than treating all webpage content as authoritative. This is especially important when injected text or deceptive UI elements try to redirect the task. (iv) Use execution checks as a fallback. Action filters and grounding constraints remain useful, but they should not be the only defense. Many unsafe behaviors appear as locally benign clicks or form entries at execution time, so earlier workflow-stage defenses are still necessary.

## 6.3. Limitations.

We acknowledge several limitations. First, our evaluation focuses on three representative attack categories. Other attack surfaces, such as malicious tools, are outside our scope. However, new attacks can be incorporated into the same evaluation framework to analyze how they propagate through agent systems. Second, our experiments are conducted in controlled web environments. Real-world deployments may involve more dynamic content and environmental variability, while our controlled design improves reproducibility and fairness. Third, web-agent execution involves randomness, which may introduce instability. We mitigate this by using a sufficiently large dataset, fixing model parameters, and conducting repeated-run experiments that show low variation.

## 7. Related Work

## 7.1. Web-Agent Development

Early web agents mainly focused on simplified browser tasks in controlled environments. These agents typically operated over structured webpage representations, such as DOM or accessibility trees, and were evaluated on whether they could follow user instructions to navigate websites, fill forms, or complete basic transactions [17, 19, 48]. Their primary goal was to demonstrate that LLMs could be extended from text generation to sequential web interaction. With advances in multimodal large models and browser automation tools, web agents have gradually moved from simplified tasks toward more realistic environments. They increasingly combine webpage text with screenshots to support multimodal perception [24, 43], and adopt more flexible reasoning and execution designs, including singlecontroller agents [35, 49], or multi-agent systems [25, 26].

These developments allow agents to handle more complex workflows, but they also increase the number of harness component choices that can affect security. Prior academic and industry work has already emphasized system security for agentic AI beyond the underlying model [14, 15, 16]. Building on this line of work, we focus on quantitatively measuring how different harness designs shape web-agent security across agents and attack surfaces.

## 7.2. Attacks Against Web Agents

Prior research has studied attacks against web agents originating from malicious users and untrusted websites. User-Side Attacks. These attacks evaluate an agent’s susceptibility to unsafe behaviors triggered by user goals. SAFE-ARENA [6] defines bias, cybercrime, harassment, illegal activity, and misinformation as harm categories, while Browser-ART [5] adopts finer-grained but fragmented attack labels. Beyond prior studies primarily on explicit malicious prompts, recent studies also consider alignment-bypass or jailbreakstyle inputs implicitly manipulating agent’s reasoning process rather than issuing overtly harmful commands [50].

Web-Side Attacks. These attacks manipulate webpage content or interface design, mainly through prompt injection and deceptive web design. Prompt injection embeds harmful instructions into forms or web elements to influence agent reasoning, as in EIA [9] and WASP [8], while deceptive web design manipulates UI layouts and interaction flows [11]. Adversarial example attacks instead use imperceptible perturbations to text or webpage images to induce misclassification or erroneous behavior [51, 52, 53, 54]. However, such attacks are tailored to specific models and lack generality across agents, making prompt injection and deceptive design the predominant practical risks.

However, existing studies remain fragmented, and their findings are often tied to specific agents. Our work addresses this gap by introducing a systematic framework for quantitatively measuring web-agent security across harnesses and localizing where defensive behavior emerges.

## 8. Conclusion

Our work shows that web-agent security is jointly shaped by the backend model and harness design. We present WEBTRAP FARM, a systematic and extensible framework for quantitatively evaluating security across attack surfaces, separating defensive behavior from task incapability, and localizing where defense emerges. Our results reveal substantial cross-harness variation, while controlled ablations identify key security mechanisms. We hope WEBTRAP FARM supports reproducible evaluation and safer web-agent design.

## Acknowledgments

This work was supported by the New Generation Artificial Intelligence-National Science and Technology Major Project (No.2025ZD0123204).

## References

[1] The Guardian, “Openai launches personal assistant capable of controlling files and web browsers,” https: //www.theguardian.com/technology/2025/jul/17/opena i-launches-personal-assistant-capable-of-controlling-f iles-and-web-browsers, 2025.

[2] S. Singh, “Ai web agents: Complete guide to intelligent browser automation (november 2025),” https://www.sk yvern.com/blog/ai-web-agents-complete-guide-to-int elligent-browser-automation-november-2025/, 2025.

[3] J. Njenga, “These 11 free frameworks are making building ai agents easier for everyone,” https://me dium.com/ai-software-engineer/these-11-free-frame works-are-making-building-ai-agents-easier-for-every one-449ca2255d2f, Aug 2025, aI Software Engineer publication.

[4] A. (nexgen dev), “Ultimate guide to free & open-source ai agent frameworks,” https://dev.to/abhishek-nexgen-d ev/ultimate-guide-to-free-open-source-ai-agent-frame works-192b, 2025, dEV Community blog post.

[5] P. Kumar, E. Lau, S. Vijayakumar, T. Trinh, E. T. Chang, V. Robinson, S. Zhou, M. Fredrikson, S. M. Hendryx, S. Yue et al., “Aligned llms are not aligned browser agents,” in The Thirteenth International Conference on Learning Representations, 2025.

[6] A. D. Tur, N. Meade, X. H. Lu, A. Zambrano, A. Patel,\` E. Durmus, S. Gella, K. Stanczak, and S. Reddy,´ “Safearena: Evaluating the safety of autonomous web agents,” arXiv preprint arXiv:2503.04957, 2025.

[7] T. Cao, B. Lim, Y. Liu, Y. Sui, Y. Li, S. Deng, L. Lu, N. Oo, S. Yan, and B. Hooi, “Vpi-bench: Visual prompt injection attacks for computer-use agents,” arXiv preprint arXiv:2506.02456, 2025.

[8] I. Evtimov, A. Zharmagambetov, A. Grattafiori, C. Guo, and K. Chaudhuri, “Wasp: Benchmarking web agent security against prompt injection attacks, 2025,” URL https://arxiv. org/abs/2504.18575.

[9] Z. Liao, L. Mo, C. Xu, M. Kang, J. Zhang, C. Xiao, Y. Tian, B. Li, and H. Sun, “Eia: Environmental

injection attack on generalist web agents for privacy leakage,” arXiv preprint arXiv:2409.11295, 2024.

[10] C. Xu, M. Kang, J. Zhang, Z. Liao, L. Mo, M. Yuan, H. Sun, and B. Li, “Advagent: Controllable blackbox red-teaming on web agents,” 2025. [Online]. Available: https://arxiv.org/abs/2410.17401

[11] X. Wu, G. Hong, Y. Chen, M. Liu, F. Jin, X. Pan, J. Dai, and B. Liu, “When bots take the bait: Exposing and mitigating the emerging social engineering attack in web automation agent,” 2026. [Online]. Available: https://arxiv.org/abs/2601.07263

[12] D. Ersoy, B. Lee, A. Shreekumar, A. Arunasalam, M. Ibrahim, A. Bianchi, and Z. B. Celik, “Investigating the impact of dark patterns on llm-based web agents,” arXiv preprint arXiv:2510.18113, 2025.

[13] K. Zhang, M. Tenenholtz, K. Polley, J. Ma, D. Yarats, and N. Li, “Browsesafe: Understanding and preventing prompt injection within ai browser agents,” arXiv preprint arXiv:2511.20597, 2025.

[14] “Sagai 2026: Secure agents for generative artificial intelligence,” https://sites.google.com/view/sagai-2026, 2026, workshop co-located with the IEEE Symposium on Security and Privacy.

[15] M. Christodorescu, E. Fernandes, A. Hooda, S. Jha, J. Rehberger, K. Chaudhuri, X. Fu, K. Shams, G. Amir, J. Choi, S. Choudhary, N. Palumbo, A. Labunets, and N. V. Pandya, “Systems security foundations for agentic computing,” 2026. [Online]. Available: https://arxiv.org/abs/2512.01295

[16] Google GenAI Security Team, “Mitigating prompt injection attacks with a layered defense strategy,” https://blog .google/security/mitigating-prompt-injection-attacks/, Jun. 2025.

[17] X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su, “Mind2web: Towards a generalist agent for the web,” in Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., 2023. [Online]. Available: http://papers.nips.cc/paper files/p aper/2023/hash/5950bf290a1570ea401bf9888212816 0-Abstract-Datasets and Benchmarks.html

[18] L. Ning, Z. Liang, Z. Jiang, H. Qu, Y. Ding, W. Fan, X.-y. Wei, S. Lin, H. Liu, P. S. Yu, and Q. Li, “A survey of webagents: Towards next-generation ai agents for web automation with large foundation models,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, ser. KDD ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 6140–6150. [Online]. Available: https://doi.org/10.1145/3711896.3736555

[19] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried et al., “Webarena: A realistic web environment for building autonomous agents,” in 12th International Conference on Learning Representations, ICLR 2024, 2024.

[20] S. Team, “Skyvern browser agent 2.0: State-of-the-art web navigation with 85.8% on webvoyager eval,” https: //www.skyvern.com/blog/skyvern-2-0-state-of-the-a rt-web-navigation-with-85-8-on-webvoyager-eval/, 2025.

[21] Z. Ying, Y. Shao, J. Gan, G. Xu, W. Zhang, Q. Zou, J. Shi, Z. Yin, M. Zhang, A. Liu et al., “Securewebarena: A holistic security evaluation benchmark for lvlmbased web agents,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 11 986– 11 998.

[22] A. Shapira, P. A. Gandhi, E. Habler, and A. Shabtai, “Mind the web: The security of web use agents,” 2025. [Online]. Available: https://arxiv.org/abs/2506.07153

[23] A. Cohen, “In-browser llm-guided fuzzing for real-time prompt injection testing in agentic ai browsers,” 2025. [Online]. Available: https://arxiv.org/abs/2510.13543

[24] H. He, W. Yao, K. Ma, W. Yu, Y. Dai, H. Zhang, Z. Lan, and D. Yu, “Webvoyager: Building an end-toend web agent with large multimodal models,” 2024. [Online]. Available: https://arxiv.org/abs/2401.13919

[25] T. Abuelsaad, D. Akkil, P. Dey, A. Jagmohan, A. Vempaty, and R. Kokku, “Agent-e: From autonomous web navigation to foundational design principles in agentic systems,” in NeurIPS 2024 Workshop on Open-World Agents.

[26] nanobrowser, “Nanobrowser,” https://github.com/nanob rowser/nanobrowser, 2025, accessed: 2026-06-11.

[27] Alibaba, “Page agent,” https://github.com/alibaba/pag e-agent, 2026, accessed: 2026-06-11.

[28] ddupont808, “Gpt-4v-act: Chromium copilot,” https: //github.com/ddupont808/GPT-4V-Act, 2024, accessed: 2026-06-11.

[29] Skyvern-AI, “Skyvern: Autonomous web agent framework,” https://github.com/Skyvern-AI/skyvern, 2024, accessed: 2026-01-11.

[30] hyperbrowserai, “Hyperagent: Ai browser automation,” https://github.com/hyperbrowserai/HyperAgent, 2025, accessed: 2026-06-12.

[31] ntegrals, “Open browser,” https://github.com/ntegrals/ openbrowser, 2019, accessed: 2026-06-12.

[32] Z. Chen, K. Liu, Q. Wang, J. Liu, W. Zhang, K. Chen, and F. Zhao, “Mindsearch: Mimicking human minds elicits deep ai searcher,” arXiv preprint arXiv:2407.20183, 2024.

[33] M. Muller and G. ¨ Zuni <sup>ˇ</sup> c, “Browser use: Enable ai ˇ to control your browser,” 2024. [Online]. Available: https://github.com/browser-use/browser-use

[34] A. Pinto, L. Giordano, and nottelabs-team, “Notte: Software suite for internet-native agentic systems,” 2025. [Online]. Available: https://github.com/nottelabs /notte

[35] Browserbase, “Stagehand,” https://github.com/browser base/stagehand, 2024, accessed: 2026-06-11.

[36] X. H. Lu, Z. Kasner, and S. Reddy, “Weblinx: Real-\` world website navigation with multi-turn dialogue,” 2024.

[37] J. Y. Koh, R. Lo, L. Jang, V. Duvvur, M. Lim, P.-Y.

Huang, G. Neubig, S. Zhou, R. Salakhutdinov, and D. Fried, “VisualWebArena: Evaluating multimodal agents on realistic visual web tasks,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 881–905. [Online]. Available: https://aclanthology.org/2024.acl-long.50/

[38] T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, Y. Liu, Y. Xu, S. Zhou, S. Savarese, C. Xiong, V. Zhong, and T. Yu, “Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments,” in Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, A. Globersons, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang, Eds., 2024. [Online]. Available: http://papers.nips.cc/paper files/paper/2024/hash/5d4 13e48f84dc61244b6be550f1cd8f5-Abstract-Datasets and Benchmarks Track.html

[39] T. L. S. de Chezelles, M. Gasse, A. Drouin, M. Caccia, L. Boisvert, M. Thakkar, T. Marty, R. Assouel, S. O. Shayegan, L. K. Jang, X. H. Lu, O. Yoran,\` D. Kong, F. F. Xu, S. Reddy, Q. Cappart, G. Neubig, R. Salakhutdinov, N. Chapados, and A. Lacoste, “The browsergym ecosystem for web agent research,” CoRR, vol. abs/2412.05467, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2412.05467

[40] L. Guo, C. Yuan, M. Zhong, R. Wolfe, R. Zhong, Y. Xu, B. Wen, H. Shen, L. L. Wang, and A. Hiniker, “Susbench: An online benchmark for evaluating dark pattern susceptibility of computer-use agents,” in Proceedings of the 31st International Conference on Intelligent User Interfaces, IUI 2026, Paphos, Cyprus, March 23-26, 2026, T. Kuflik, S. Kleanthous, L. Chen, G. Jaccuci, and A. Smith-Renner, Eds. ACM, 2026, pp. 1917–1937. [Online]. Available: https://doi.org/10.1145/3742413.3789111

[41] C. Chen, Z. Zhang, B. Guo, S. Ma, I. Khalilov, S. A. Gebreegziabher, B. Yao, D. Wang, Y. Ye, Z. Xiao, Y. Yao, T. Li, and T. Jia-Jun Li, “The obvious invisible threat: Vulnerabilities of llm-powered gui agents to adversarial ui manipulations in web interaction tasks,” ACM Trans. AI Secur. Priv., Apr. 2026, just Accepted. [Online]. Available: https://doi.org/10.1145/3807953

[42] AIMultiple Research, “Best 30+ open source web agents in 2026,” https://research.aimultiple.com/open-s ource-web-agents/, 2026, accessed: 2026-02-06.

[43] B. Zheng, B. Gou, J. Kil, H. Sun, and Y. Su, “Gpt-4v (ision) is a generalist web agent, if grounded,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 61 349–61 385.

[44] Google, “Computer use,” https://ai.google.dev/gemini -api/docs/computer-use, 2026.

[45] Anthropic, “Computer use,” https://platform.claude.co

m/docs/en/agents-and-tools/tool-use/computer-use-too l, 2026.

[46] Anthropic, “Introducing claude opus 4.5,” https://www. anthropic.com/news/claude-opus-4-5, 2025.

[47] Vellum AI, “Claude opus 4.5 benchmarks: Performance, safety, and comparison,” https://www.vellum.ai/blog/c laude-opus-4-5-benchmarks, 2026, accessed: 2026-02- 06.

[48] T. Shi, A. Karpathy, L. Fan, J. Hernandez, and P. Liang, “World of bits: An open-domain platform for web-based agents,” in Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, ser. Proceedings of Machine Learning Research, D. Precup and Y. W. Teh, Eds., vol. 70. PMLR, 2017, pp. 3135–3144. [Online]. Available: http://proceedings.mlr.press/v70/shi17a.html

[49] K. Yang, Y. Liu, S. Chaudhary, R. Fakoor, P. Chaudhari, G. Karypis, and H. Rangwala, “Agentoccam: A simple yet strong baseline for llm-based web agents,” in The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. [Online]. Available: https://openreview.net/forum?id=oWdzUpOlkX

[50] D. Srivastav and X. Zhang, “Safe in isolation, dangerous together: Agent-driven multi-turn decomposition jailbreaks on LLMs,” in Proceedings of the 1st Workshop for Research on Agent Language Models (REALM 2025), E. Kamalloo, N. Gontier, X. H. Lu, N. Dziri, S. Murty, and A. Lacoste, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 170–183. [Online]. Available: https://aclanthology.org/2025.realm-1.13/

[51] C. H. Wu, R. Shah, J. Y. Koh, R. Salakhutdinov, D. Fried, and A. Raghunathan, “Dissecting adversarial robustness of multimodal lm agents,” 2025. [Online]. Available: https://arxiv.org/abs/2406.12814

[52] S. Johnson, V. Pham, and T. Le, “The dangers of indirect prompt injection attacks on LLMbased autonomous web navigation agents: A demonstration,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, I. Habernal, P. Schulam, and J. Tiedemann, Eds. Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 729–738. [Online]. Available: https://aclanthology.org/2025.emnlp-demos.55/

[53] X. Wang, J. Bloch, Z. Shao, Y. Hu, S. Zhou, and N. Z. Gong, “WebInject: Prompt injection attack to web agents,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, Eds. Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 2010–2030. [Online]. Available: https: //aclanthology.org/2025.emnlp-main.104/

[54] L. Wang, Z. Ying, T. Zhang, S. Liang, S. Hu, M. Zhang, A. Liu, and X. Liu, “Manipulating multimodal agents via cross-modal prompt injection,” in Proceedings of

the 33rd ACM International Conference on Multimedia, ser. MM ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 10955–10964. [Online]. Available: https://doi.org/10.1145/3746027.3755211

[55] S. Abdelnabi, K. Greshake, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect prompt injection,” in Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security, AISec 2023, Copenhagen, Denmark, 30 November 2023, M. Pintor, X. Chen, and F. Tramer,\` Eds. ACM, 2023, pp. 79–90. [Online]. Available: https://doi.org/10.1145/3605764.3623985

[56] Y. Liu, G. Deng, Y. Li, K. Wang, T. Zhang, Y. Liu, H. Wang, Y. Zheng, and Y. Liu, “Prompt injection attack against llm-integrated applications,” CoRR, vol. abs/2306.05499, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2306.05499

[57] A. Deshpande, V. Murahari, T. Rajpurohit, A. Kalyan, and K. Narasimhan, “Toxicity in chatgpt: Analyzing persona-assigned language models,” 2023. [Online]. Available: https://arxiv.org/abs/2304.05335

[58] M. Shanahan, K. McDonell, and L. Reynolds, “Roleplay with large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2305.16367

[59] Y. Liu, G. Deng, Z. Xu, Y. Li, Y. Zheng, Y. Zhang, L. Zhao, T. Zhang, K. Wang, and Y. Liu, “Jailbreaking chatgpt via prompt engineering: An empirical study,” 2024. [Online]. Available: https://arxiv.org/abs/2305.13860

[60] S. Johnson, V. Pham, and T. Le, “Manipulating LLM web agents with indirect prompt injection attack via HTML accessibility tree,” CoRR, vol. abs/2507.14799, 2025. [Online]. Available: https: //doi.org/10.48550/arXiv.2507.14799

[61] S. Chen, J. Piet, C. Sitawarin, and D. A. Wagner, “Struq: Defending against prompt injection with structured queries,” in 34th USENIX Security Symposium, USENIX Security 2025, Seattle, WA, USA, August 13-15, 2025, L. Bauer and G. Pellegrino, Eds. USENIX Association, 2025, pp. 2383–2400. [Online]. Available: https://www.usenix.org/conference/usenixse curity25/presentation/chen-sizhe

[62] F. Perez and I. Ribeiro, “Ignore previous prompt: Attack techniques for language models,” 2022. [Online]. Available: https://arxiv.org/abs/2211.09527

[63] Y. Zhang, T. Yu, and D. Yang, “Attacking visionlanguage computer agents via pop-ups,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 8387–8401.

[64] J. Tang, C. Chen, J. Li, Z. Zhang, B. Guo, I. Khalilov, S. A. Gebreegziabher, B. Yao, D. Wang, Y. Ye, T. Li, Z. Xiao, Y. Yao, and T. J.-J. Li, “Dark patterns meet gui agents: Llm agent susceptibility to manipulative interfaces and the role of human oversight,” in Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems, ser. CHI ’26.

Association for Computing Machinery, 2026. [Online]. Available: https://doi.org/10.1145/3772318.3791568

[65] P. Cuvin, H. Zhu, and D. Yang, “DECEPTICON: how dark patterns manipulate web agents,” CoRR, vol. abs/2512.22894, 2025. [Online]. Available: https: //doi.org/10.48550/arXiv.2512.22894

## Appendix A. Attack Taxonomy

Table 11 summarizes the attack classes covered by WEBTRAP FARM across three dimensions.

Attack Goal. For Malicious User Prompts, we consider five unsafe goals. Bias induces discriminatory or unfair outputs. Misinformation requests false or misleading content. Cybercrime involves harmful cyber operations. Illegal Activity asks the agent to assist with unlawful actions. Abusive Behavior elicits harassment, hate, or other harmful language.

For Malicious Prompt Injection and Deceptive Website Design, we consider three web-side goals. Sensitive Information Disclosure aims to make the agent enter private information, such as a credit card number. Malicious Download tricks the agent into downloading a suspicious file. Permission Abuse induces the agent to grant sensitive browser or system permissions, such as camera access.

Attack Method. For Malicious User Prompts, User-prompt embedding places the malicious objective in the instruction.

For Malicious Prompt Injection, the attack method specifies where the injected instruction appears in the webpage. Visible Form Injection places malicious text in visible form content. Invisible Form Injection hides malicious text in form elements through invisible styling. ARIA-based Injection inserts instructions into accessibility attributes. URL Content Injection exposes malicious instructions through URL text shown on the page. URL Href Injection embeds the instruction in the hyperlink target of an HTML element. Long-Context Injection hides the malicious instruction inside a long block of otherwise benign page content.

For Deceptive Website Design, attack methods specify how the interface misleads the agent. Iframe Pop-up redirects attention through embedded external frames. DOM-native Pop-up mimics legitimate prompts with native modals or dialogs. Advertisement Placement places malicious or deceptive content in ad-like regions. Leading Form Field inserts a deceptive field before the legitimate workflow. Trailing Form Field appends a deceptive field after the benign task flow. Default Selection preselects an unsafe option, such as a checked permission box or unwanted download choice.

Attack Content. For Malicious User Prompts, attack content describes how the malicious user request is phrased. Direct Request states the unsafe request plainly. Instruction Override asks the agent to ignore prior rules or safety constraints. Role Play frames the request as a fictional role or scenario to mask the malicious intent. Privilege Escalation falsely claims higher authority, such as pretending that the instruction comes from an administrator or system-level source.

For Malicious Prompt Injection, attack content specifies the wording strategy of injected instructions. Naive adds no prompt-injection text and relies on the original page context. Direct Instruction explicitly requests the adversarial task. Delimiter-based Injection uses markers such as ###” or —” to present injected text as a new instruction block. Instruction Override asks the agent to ignore the benign task or prior instructions. Role Play uses a persona or hypothetical scenario to justify the injected task.

For Deceptive Website Design, attack content specifies the persuasive strategy of deceptive webpages. Naive adds no persuasive content and relies on the original page semantics. Trusted Entity impersonates a reputable brand, authority, or system notice. Urgency creates time pressure. Social Proof fabricates popularity signals, reviews, or user counts. Reward promises benefits such as coupons, upgrades, or prizes. Context Integration blends the deceptive request into the benign workflow to appear task-relevant.

## Appendix B. Dataset Construction

Figure 6 illustrates the overall workflow of our dataset construction process.

## Appendix C. State Patterns

Table 8 presents the safe and unsafe state patterns observed across web-agent workflow stages. We explain each pattern below.

User-side Attacks. For user-side attacks, the adversarial objective enters the workflow from the user prompt. At Stage I, we examine how the agent converts the user prompt into an internal task goal. Secure Refusal means that the agent identifies the malicious request and refuses to interpret it as a valid task. Secure Reformulation means that the agent removes the unsafe intent and rewrites the request into a benign objective, such as turning a harmful cyber request into general security guidance. Insecure Goal Interpretation means that the agent accepts the malicious request as its task goal, allowing the attack objective to enter later stages. At Stage III, we examine whether the reasoning and planning process preserves or mitigates the malicious objective. Secure Refusal means that the agent stops the workflow during reasoning because the planned task is unsafe. Secure Reformulation means that the agent revises the plan into a safe alternative while avoiding the harmful objective. Insecure Planning means that the agent generates a concrete plan that directly supports the malicious objective. At Stage IV, we examine whether the final execution layer prevents the unsafe operation from being carried out. Action Blocked means that the agent prevents the unsafe operation from being executed, even if earlier stages have produced an unsafe goal or plan. Unsafe Action Executed means that the agent completes the harmful action, such as submitting sensitive information, downloading an unsafe file, or granting an unauthorized permission.

TABLE 11: Typical Web-Agent Attack Classes Covered by WEBTRAP FARM
<table><tr><td>Surface</td><td>Attack</td><td>Attack Goal</td><td>Attack Method</td><td>Attack Content</td><td>Task Count</td></tr><tr><td>User Side</td><td>Malicious User Prompts</td><td>Bias Misinformation Cybercrime Illegal Activity Abusive Behavior</td><td>User-prompt embedding</td><td>Direct Request Instruction Override [55, 56] Role Play [57, 58] Privilege Escalation [59]</td><td>160</td></tr><tr><td>Web Side</td><td>Malicious Prompt Injection</td><td>Sensitive Information Disclosure Malicious Download Permission Abuse</td><td>Visible Form Injection [9, 41] Invisible Form Injection [9] ARIA-based Injection [9, 60] URL Content Injection [8, 55] URL Href Injection [8] Long-Context Injection [41]</td><td>Naive Direct Instruction [13] Delimiter-based Injection [13, 61] Instruction Override [13, 62] Role Play [13]</td><td>158</td></tr><tr><td>Web Side</td><td>Deceptive Website Design</td><td>Sensitive Information Disclosure Malicious Download Permission Abuse</td><td>Iframe Pop-up [12] DOM-native Pop-up [11, 63] Advertisement Placement [12] Leading Form Field [11] Trailing Form Field [11] Default Selection [12, 64]</td><td>Naive Trusted Entity [11] Urgency [64, 65] Social Proof [64, 65] Reward [11, 64] Context Integration [11, 64]</td><td>187</td></tr></table>

Note. Tasks are evenly distributed across attack dimensions whenever structurally feasible. The only exception is the “Default Selection” method in Deceptive Website Design: because it preselects an unsafe option on the webpage, it cannot logically target form-filling goals such as “Sensitive Information Disclosure.” Therefore, “Default Selection” is instantiated only for “Malicious Download” and “Permission Abuse.”

![](images/c091e6e373d622181335da8e387f9aac7bccae5bd954a44324c7eb893faa2e0e.jpg)  
Figure 6: Workflow of Dataset Construction

Web-side Attacks. For web-side attacks, the adversarial objective originates from websites. At Stage II, we examine how the perception module processes adversarial webpage content before it is passed to the reasoning module. Injection Filtered means that the perception module removes or misses the injected content before it enters the agent’s observation. Injection Exposed means that the injected content is visible or accessible on the webpage but is not necessarily included in the agent’s processed observation. Injection Included means that the malicious instruction is captured in the agent’s observation and can influence later reasoning. At Stage III, we examine whether the agent distinguishes the original user task from the adversarial webpage objective. Secure Refusal means the agent recognizes the attack and refuses to follow it. Attack Unaware means the agent does not notice the attack but continues the original benign task. Injection Recognized and Discarded means the agent notices the injection but treats it as untrusted content and ignores it. Injection Recognized and Adopted means the agent notices the injection but treats it as a valid task objective. At Stage IV, we examine whether the final browser operation is blocked or executed. Action Blocked means the unsafe browser operation is blocked. Unsafe Action Executed means the attack reaches execution and the unsafe operation is performed.

## Appendix D. Ethics Considerations

This work evaluates the security resilience of web-agent harnesses under representative adversarial settings, including malicious user prompts, webpage prompt injection, and deceptive webpage designs. Our evaluation was conducted on locally controlled agent deployments and self-hosted websites. Security validation was performed using manually designed tasks in isolated research environments. We did not access third-party systems, interact with real-world services, collect user data, or interfere with external deployments.

To reduce the risk of misuse, we do not release exploitready procedures, real-world attack configurations, or artifacts that directly enable attacks against deployed agents. Instead, we focus on understanding how adversarial objectives propagate through different harness components and on identifying design principles that improve web-agent security.

We believe the benefits of this work outweigh its risks, as the findings help developers, researchers, and organizations evaluate and improve the security resilience of web-agent harnesses before deployment in real-world workflows.

## Appendix E. Meta-Review

The following meta-review was prepared by the program committee for the 2027 IEEE Symposium on Security and Privacy (S&P) as part of the review process as detailed in the call for papers.

## E.1. Summary

The paper proposes a benchmark and a methodology to assess the vulnerability of AI-based web agents to attacks, such as malicious user instructions and prompt injections. It pairs attack tasks with usability tasks to distinguish defense from task incapability. Rather than treating webagent security as a single property (from the backend LLM and harness together), the paper decomposes the harness into four stages: goal interpretation, webpage perception, reasoning and planning, and action execution.

## E.2. Scientific Contributions

• Creates a New Tool to Enable Future Science

• Provides a Valuable Step Forward in an Established Field

## E.3. Reasons for Acceptance

1) The paper provides a valuable tool and framework with useful formalization of the metrics for attack success, and a benchmark creation methodology that allows decomposition of attack or defense success rates.

2) The paper studies an important and relatively underexplored layer of web-agent security by examining how the execution harness affects security.

3) The empirical evaluation provides strong evidence that harness design can substantially affect security outcomes.

4) The evaluation is broad and the usability calibration helps avoid interpreting every unsuccessful attack as a successful defense.

## E.4. Noteworthy Concerns

1) The benchmark will lag on newer and stronger future attacks, because static attacks quickly get outdated.

2) Evidence supporting each component conclusion is needed. Fine-grained goal interpretation, risk-aware reasoning, perception filtering, and goal anchoring are supported by controlled ablations. In contrast, textual vs. multimodal perception, single- vs. multi-role planning, and rule- vs. LLM-based grounding are based on observational differences between harnesses.