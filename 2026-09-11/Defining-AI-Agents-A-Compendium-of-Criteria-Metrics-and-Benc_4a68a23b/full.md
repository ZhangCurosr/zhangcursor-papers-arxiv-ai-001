# Defining AI Agents: A Compendium of Criteria, Metrics, and Benchmarks

Mia Lassiter Duke University mia.lassiter@duke.edu

Brinnae Bent Duke University brinnae.bent@duke.edu

## ABSTRACT

The term “agent” in artificial intelligence lacks a standard definition, complicating the evaluation, comparison, and reproducibility of AI agent research. We address this ambiguity through a survey organized around five dimensions of agenticness: environmental interaction, learning and adaptation, autonomy, goal-directed behavior, and temporal coherence. For each dimension, we examine how the underlying capability has been conceptualized across prior work and synthesize the metrics, benchmarks, and evaluation frameworks used to assess it. This review provides a structured account of the current landscape of agent evaluation, highlighting both established approaches and areas where evaluation remains limited or inconsistent. We additionally introduce the Agent Compendium, a public-facing digital resource that organizes and extends the evaluation methods identified through this review. Together, the survey and compendium provide a common structure for evaluating and comparing agent capabilities across AI systems, supporting more reproducible research, clearer communication, and more systematic study of artificial agents.

## 1 Introduction

Artificial intelligence (AI) agents have attracted substantial public, commercial, and academic attention because of their potential to perform complex tasks, interact with digital environments, and augment human work across domains. Yet the term “AI agent” lacks a widely accepted definition, and researchers continue to disagree about the capabilities required for a system to qualify as an agent. A 2026 textbook, for example, describes AI agents as “intelligent software entities” capable of environmental perception, autonomous decision-making, and network optimization (Mehta, 2026). A research article published that same year characterizes them as autonomous systems that complete specific tasks and communicate with other agents (Jhi and Park, 2026). These differences reflect a much longer-standing conceptual problem: researchers have used the term agent in disparate ways since the genesis of the term in computer science in the 1970s (Bent, 2025). This ambiguity creates practical problems for AI research and deployment. First, it complicates the evaluation and reproducibility of agent research (Bent, 2025). Without shared criteria for what constitutes an agent, researchers may evaluate systems that differ substantially in their underlying capabilities while describing them using the same terminology. The rapid integration of large language models (LLMs) into agent architectures has intensified this problem, creating uncertainty about which capabilities should be evaluated and which benchmarks appropriately measure them (Zhu et al., 2025). A system that invokes tools, a system that independently pursues a multi-step objective, and a system that adapts its behavior over time may all be described as an “agent,” even though these systems present different evaluation requirements.

Second, inconsistent definitions complicate communication between researchers, developers, and the public (Bent, 2025). Even leading AI companies emphasize different properties when describing agents. OpenAI defines agents as “systems that independently accomplish tasks on your behalf” (OpenAI, 2026), whereas Anthropic describes them as “systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks” (Anthropic, 2026). Such variation shapes how users understand what agents can do, how much autonomy they possess, and what forms of human involvement they require. Shome et al. (2026), for example, found substantial mismatches between agent behavior and users’ mental models. Participants expressed uncertainty about agent capabilities, prompting requirements, and their own role during interaction. Some misunderstood when human takeover was necessary and developed what the authors describe as “prompt gambling”: hesitation to issue prompts because of uncertainty about how extensively an agent might act or respond (Shome et al., 2026). Definitional ambiguity therefore extends beyond terminology; it influences expectations, interaction strategies, and trust.

Third, ambiguity surrounding “AI agent” contributes to the broader conflation of agents with agentic AI. These terms increasingly appear across academic research, industry discourse, and public communication, often without clear distinctions between a system that qualifies as an agent and a system that exhibits particular degrees or forms of agentic behavior. Establishing explicit criteria for AI agents provides a foundation for studying agenticness as a set of measurable capabilities rather than treating “agent” as a catch-all label for increasingly autonomous AI systems.

To address these challenges, we introduce the Agent Compendium, a public-facing digital resource that organizes evaluation methods around the capabilities that constitute AI agency. The Agent Compendium provides explicit criteria for identifying AI agents and maps those criteria to existing metrics, benchmarks, and evaluation frameworks. By establishing clearer terminology and connecting agent criteria to concrete evaluation methods, the Agent Compendium aims to improve the reproducibility and comparability of AI agent research, support clearer communication about agent capabilities, better align user expectations with system behavior, and provide a foundation for future work on the broader concept of agentic AI.

## 2 Related Work

## 2.1 Agenticness Framework

Our research is grounded in prior work that redefines AI agents and compares AI systems’ levels of agenticness. Agenticness is “the degree to which a system exhibits characteristics commonly associated with an agent” (Bent, 2025). This framework spans five dimensions: 1) environmental interaction, 2) learning and adaptation, 3) autonomy, 4) goal-directed behavior, and 5) temporal coherence. Environmental interaction refers to an agent’s understanding of and impact on its environment. Learning and adaptation refer to an agent’s ability to improve performance and modify behavior. Autonomy is an agent’s capacity to operate without external guidance. Goal-directed behavior refers to an agent’s ability to form, understand, and achieve objectives (Bent, 2025). Finally, temporal coherence describes a system’s ability to maintain logical consistency across memory, context, and intent (McCants, 2026).

## 2.2 Existing Agent Evaluations

In 2025, Zhu et al. published a survey of LLM-based AI agent evaluations. They categorize agent evaluations based on the agents’ environments: code-related (Chen et al., 2021), web (Shi et al., 2017), OS (Xie et al., 2024), mobile (Rawles et al., 2023), scientific (Lee et al., 2023), or game (Paglieri et al., 2025). They provide a second categorization based on agent capabilities: planning (Valmeekam et al., 2023), self-reflection (Li et al., 2024), interaction (Yao et al., 2024), memory (Kociský et al., 2018), and general (Kapoor et al., 2025; Yin et al., 2025).ˇ

In 2025, Yin et al. developed the Massive Multitask Agent Understanding (MMAU) benchmark to evaluate agent skill sets that result in task completion or failure. Unlike existing benchmarks that merely evaluate agents’ application scenarios, the MMAU also measures agents’ abilities to understand, reason, plan, problem-solve, and self-correct across various domains. Though the MMAU benchmark is detailed, it does not evaluate AI agents’performance in interactive environments, which are critical domains (Yin et al., 2025).

That same year, Kapoor et al. (2025) proposed the Holistic Agent Leaderboard (HAL) as “the missing infrastructure for AI agent evaluation.” HAL was designed to address infrastructure-, measurement-, and validation-related challenges that are unique to agent evaluation. Firstly, existing infrastructure is prohibitively slow and complex, and leaderboards are rarely updated. Secondly, agent costs deviate across token use and scaffolding, yet existing evaluations fail to account for these variations. Thirdly, agents often exploit shortcuts to optimize benchmark scores, and current evaluations seldom detect or penalize these behaviors. HAL’s three main components—a unified evaluation harness, a regularly updated, multidimensional leaderboard, and an automated analysis of agent logs—address the aforementioned challenges, respectively. HAL’s framework-agnostic, open-source harness standardizes agent evaluation across nine benchmarks (Xue et al., 2025; Yoran et al., 2024; Mialon et al., 2023; Siegel et al., 2024; Chen et al., 2025; Jimenez et al., 2024; Hobbhahn, 2025; Shi et al., 2024; Yao et al., 2024; Tian et al., 2024) spanning web navigation, scientific research, software engineering, and customer service domains. However, the benchmarks only measure specific capabilities (e.g., an agent’s ability to solve competitive programming problems [Shi et al., 2024]) rather than components of agenticness dimensions.

To date, a standardized repository of agent characteristic evaluation methods does not exist. AI agent evaluation is siloed, which creates inefficiencies during the literature review process and hinders research progress (Bakos, 2025). We propose a public-facing digital compendium of evaluation methods for components of agenticness. This compendium will promote information accessibility and fill the aforementioned niche in the AI agent research space.

## 3 Evaluation Methods

Evaluation methods facilitate the reproducibility of AI agent research, and each dimension of agenticness has components that can be evaluated. We categorized existing evaluation metrics into the five dimensions of agenticness.

## 3.1 Environmental Interaction

Environmental interaction refers to an agent’s perception, understanding, and manipulation of its environment (Bent, 2025). Components of environmental interaction include agent architecture, navigation, tool-use, geospatial analysis, consistency, and robustness.

## 3.1.1 Agent Architecture

Agent architecture describes the hardware, software, and data infrastructure of a system. Deliberative, reactive, and hybrid are three common kinds of agent architecture (Carrascosa et al., 2004; Dumke et al., 2000; Remondino, 2005; Wooldridge and Jennings, 1995). Agents with deliberative architecture have internal, symbolic representations of their environments and the capacity to reason. Agents with reactive architecture do not have internal representations of their environments but have a set of condition-action rules that generate immediate responses to stimuli. Agents with hybrid architecture have the strengths—rapid responses and high-level planning—of the two aforementioned architectures.

Wooldridge and Jennings (1995) consider deliberative architecture to be the “classical approach to building agents.” Deliberative architecture is founded on the physical-symbol system hypothesis, proposed by Newell and Simon in 1976, which states: “A physical symbol system has the necessary and sufficient means for general intelligent action.” A deliberative agent or agent architecture is a physical symbol system that makes decisions “via logical (or at least pseudo-logical) reasoning, based on pattern matching and symbolic manipulation” (Wooldridge and Jennings, 1995). Wooldridge and Jennings (1995) acknowledge two flaws with deliberative architecture. First, it is difficult to accurately, adequately, and promptly translate the real world into a symbolic representation. Second, it is challenging to symbolically represent information about real-world processes, and for agents to use such representations during reasoning. These challenges prompted researchers to develop an alternative architecture.

Reactive architecture has neither a symbolic world model nor complex symbolic reasoning (Wooldridge and Jennings, 1995). Leading proponent of reactive architecture, Rodney Brooks, grounds his research in two main claims: 1) intelligence is situated in the world, not in theorems, and 2) intelligent behavior is acquired and developed from an agent’s interaction with its environment (Brooks, 1986). Yet a purely reactive agent is not adaptable because it can only act in an environment or world state that was fully considered a priori (Remondino, 2005). Thus, many researchers argued for hybrid architecture, which combines the best properties of classical and alternative architectures (Burmeister and Sundermeyer, 1992; Ferguson, 1992; Georgeff and Lansky, 1987; Wooldridge and Jennings, 1995).

## 3.1.2 Navigation

Navigation describes goal-directed movement through environments (Felicia et al., 2026). Navigation benchmarks typically include the agent’s episodic success, normalized by inverse path length (Anderson et al., 2018a):

$$
S P L = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i } \frac { \ell _ { i } } { \operatorname* { m a x } ( p _ { i } , \ell _ { i } ) }\tag{1}
$$

where N is the number of test episodes. Equipped with a binary definition of episodic success, the agent is tasked with navigating to a goal in each of the $N$ episodes. Let $\ell _ { i }$ be the shortest-path distance from the agent’s starting position to the goal in episode $i ,$ and let $p _ { i }$ be the length of the path actually taken by the agent in this episode. Let $S _ { i }$ be a binary indicator of success for episode i (Anderson et al., 2018a).

This section describes six types of visual- and language-guided navigation, along with various visual-language models (see Table 1). Visual- and language-guided navigation tasks involve an agent being spawned in a never-seen-before environment and having to navigate the environment via natural language instructions (Felicia et al., 2026; Lin et al., 2023).

Table 1: Agent Navigation Visual-Language Models
<table><tr><td colspan="2">Navigation Type uar-Language Moueis Model</td><td>Unique Features</td></tr><tr><td>Point-Goal</td><td>DD-PPO (Wijmans et al., 2020)</td><td>Near-linear scaling in an unseen environment</td></tr><tr><td rowspan="2">Object-Goal</td><td>ZSON (Majumdar et al., 2022)</td><td>Enables zero-shot navigation</td></tr><tr><td>CLIP-Nav (Dorbala et al., 2022) CoW (Gadre et al., 2022)</td><td>Uses vision-language models Searches for the first instance of any object</td></tr><tr><td rowspan="2">Vision-Language</td><td>Room-to-Room (Anderson et al., 2018b)</td><td>First vision-language navigation benchmark</td></tr><tr><td>REVERIE (Qi et al., 2020) VLN-CE (Krantz et al., 2020)</td><td>Adds object grounding Extends to continuous environments</td></tr><tr><td>Audio-Visual</td><td>SoundSpaces (Chen et al., 2020)</td><td>First audio-visual navigation model</td></tr><tr><td>Audio-Embodied Question Answering</td><td>EXPRESS-Bench (Jiang et al., 2025)</td><td>Evaluates agents’ exploration efficiency and rea- soning capabilities</td></tr><tr><td rowspan="2">Visual Question</td><td>NoisyEQA (Wu et al., 2024)</td><td>Measures noise detection capability and answer quality</td></tr><tr><td>ScanQA (Azuma et al., 2022)</td><td>Addresses the problems of conventional 2D-QA</td></tr><tr><td rowspan="2">Answering</td><td>Oscar (Li et al., 2020)</td><td>models</td></tr><tr><td></td><td>Creates a new method to ease the learning of align- ments</td></tr></table>

While executing a point-goal navigation task, an agent must navigate to a goal location specified in relative coordinates, using only visual observations and pose information (Desai and Lee, 2021). During the task, the agent is challenged to learn task-relevant representations of visual observations and use these representations to inform its decisions.

In an object-goal navigation task, an agent navigates to an instance of an object category in an unexplored environment (Majumdar et al., 2022). During a vision-language navigation task, an agent navigates a visual environment via natural language instructions. Instructions may be multi-step commands like, “Go up the stairs and turn right. Go past the bathroom and stop next to the bed” (Anderson et al., 2018b). In an audio-visual navigation task, an agent can both hear and see while searching for its sound-emitting target.

During an embodied question answering (EQA) task, an agent is asked a question about something in its environment. The agent must navigate the environment, collect information through first-person vision, and then answer the question (Das et al., 2017). For example, a person may ask an AI agent, “In what room is the blue couch?” The agent must navigate the house until it finds the blue couch, and then reply with the room in which the blue couch was found. Similarly, visual question answering (VQA) tasks require agents to answer questions about visual content. However, the agents involved in VQA tasks have a fixed view of the environment and can not actively perceive the environment or control their trajectories (Das et al., 2017).

## 3.1.3 Tool-Use

Tools represent functions that an AI agent can call, and “they allow the agent to interact with external data sources by pulling or pushing information to that source” (Masterman et al., 2024). Tool-use characterizes a tool-using agent’s understanding of object affordances and function calling (Felicia et al., 2026; Vyatkin et al., 2026). MultiCAT-Bench is an open-source benchmark that provides a detailed categorization of ten tool-use agent evaluations (see Table 2).

As reflected in MultiCAT-Bench, tool-use model outputs can be categorized by call, tool name, and arguments (Vyatkin et al., 2026). A tool call is an agent response that triggers a tool to run. Vyatkin et al. (2026) classify agent responses as correct if a) a tool call was not required and a tool call was not made or b) a tool call was required and all selected tools and parameters matched the ground truth. Vyatkin et al. (2026) instantiate tool arguments into three variants per tool: minimal, partial, and complete. Minimal arguments had only the required parameters; partial arguments had the required parameters and a subset of optional parameters; complete arguments had all parameters.

AlShikh et al. (2025) also proposes the Tool Dexterity Index (TDI) as seen in (2) to assess agents’ abilities to intelligently use available tools.

$$
T D I = { \frac { \Sigma { \mathrm { ~ T o o l ~ U s e ~ S c o r e s } } } { \mathrm { T o t a l ~ O p p o r t u n i t i e s ~ t o ~ U s e ~ T o o l s } } }\tag{2}
$$

The potential tool use scores are +1 for “optimal use,” -1 for “misuse,” and -0.5 for “ignored better tool.”

Table 2: MultiCAT-Bench Evaluations
<table><tr><td>Measure</td><td colspan="2">Definition (Defn.) and Equation  $\mathbf { ( E q . ) }$ </td></tr><tr><td>Accuracy</td><td>Defn. Eq.</td><td>Percentage of fully correct tool calls  $\begin{array} { r } { A c c = \frac { \tilde { N } _ { c o r r } } { N _ { t a s k s } } } \end{array}$  , where  $N _ { c o r r }$  is the number of fully correct responses and  $N _ { t a s k s }$  is the total number of tasks.</td></tr><tr><td>Overall Recall</td><td>Defn. Eq.</td><td>Average fraction of fully correct tool calls according to the ground truth  $\begin{array} { r } { R e c _ { o } = \frac { 1 } { N _ { t a s k s } } \sum _ { t \in T } \frac { N _ { c o r r \ : t o o l s } ^ { t } } { N _ { q t \ : t o o l s } ^ { t } } } \end{array}$  , where t is a task chosen from the set of tasks length  $N _ { t a s k s } , N _ { g t \_ t o o l s } ^ { t }$  is the number of tools that need to be called in task  $t ,$ </td></tr><tr><td>Overall Precision</td><td>Defn. Eq.</td><td> $N _ { c o r r \_ t o o l s } ^ { t }$  is the number of correctly called tools. Average fraction of fully correct tool calls among the called tools  $\begin{array} { r } { P r e _ { o } = \frac { 1 } { N _ { t a s k s } } \sum _ { t \in T } \frac { N _ { c o r r \_ t o o l s } ^ { t } } { N _ { l l m \_ t o o l s } ^ { t } } } \end{array}$  , where  $N _ { l l m \_ t o o l s } ^ { t }$  is the number of tools called by</td></tr><tr><td>Overall F1</td><td>Defn. Eq.</td><td>the LLM. Average F1 score for fully correct tool calls  $\begin{array} { r } { F 1 _ { o } = \frac { 1 } { N _ { t a s k s } } \sum _ { t \in T } \frac { 2 \cdot R e c _ { o } ^ { t } \cdot P r e _ { o } ^ { t } } { R e c _ { o } ^ { t } + P r e _ { o } ^ { t } } } \end{array}$  , where  $R e c _ { o } ^ { t }$  and  $P r e _ { o } ^ { t }$  are the Overall Recall and Overall Precision values computed for task t. If both recall and precision are equal to</td></tr><tr><td>ToolName Recall</td><td>Defn. Eq.</td><td>Average fraction of correctly identified tool names according to the ground truth  $\begin{array} { r } { R e c _ { t n } = \frac { 1 } { N _ { t a s k s } } \sum _ { t \in T } \frac { N _ { c o r r \_ t o o l \_ n a m e s } ^ { t } } { N _ { g t \_ t o o l s } ^ { t } } } \end{array}$  , where  $N _ { c o r r \_ t o o l \_ n a m e s } ^ { t }$  is the number of cor- rectly identified tool names. If the model made no calls when they were required, the</td></tr><tr><td>ToolName Precision</td><td>Defn. Eq.</td><td>Average fraction of correctly identified tool names among the called tools  $\begin{array} { r } { P r e _ { t n } = \frac { 1 } { N _ { t a s k s } } \sum _ { t \in T } \frac { N _ { c o r r \_ t o o l s } ^ { t } } { N _ { l l m \_ t o o l s } ^ { t } } } \end{array}$ </td></tr><tr><td>ToolName F1</td><td>Defn. Eq.</td><td>Average F1 score for correctly identified tool names  $\begin{array} { r } { F 1 _ { t n } = \frac { 1 } { N _ { t a s k s } } \sum _ { t \in T } \frac { 2 \cdot R e c _ { t n } ^ { t } \cdot P r e _ { t n } ^ { t } } { R e c _ { t n } ^ { t } + P r e _ { t n } ^ { t } } } \end{array}$  , where  $R e c _ { t n } ^ { t }$  and  $P r e _ { t n } ^ { t }$  are the ToolName Re- call and ToolName Precision values computed for task t. If both recall and precision</td></tr><tr><td>Arguments Recall</td><td>Defn.</td><td>are equal to 0, the F1 score for the task is defined as 0. Fraction of correctly extracted arguments from the ground truth</td></tr><tr><td></td><td>Eq.</td><td> $\begin{array} { r } { R e c _ { \mathrm { a r g s } } = \frac { 1 } { N _ { \mathrm { t a s k s } } } \sum _ { t \in T } \left( \frac { 1 } { N _ { \mathrm { m a t c h e d } } ^ { t } } \sum _ { c \in C _ { \mathrm { m a t c h e d } } } \frac { N _ { \mathrm { c o r r _ { - } a r g s } } ^ { t , c } } { N _ { \mathrm { g t _ { - } a r g s } } ^ { t , c } } \right) } \end{array}$ </td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $\begin{array} { r } { P r e _ { \mathrm { a r g s } } = \frac { 1 } { N _ { \mathrm { t a s k s } } } \sum _ { t \in T } \left( \frac { 1 } { N _ { \mathrm { m a t c h e d } } ^ { t } } \sum _ { c \in C _ { \mathrm { m a t c h e d } } } \frac { N _ { \mathrm { r i g h t \_ a r g s } } ^ { t , c } } { N _ { \mathrm { l l m \_ a r g s } } ^ { t , c } } \right) } \end{array}$ </td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Arguments</td><td>Defn.</td><td>Fraction of correctly extracted arguments from the model&#x27;s output</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Precision</td><td>Eq.</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Arguments F1</td><td></td><td>Average F1 score for correctly extracted arguments with respect to correctly identified</td></tr><tr><td></td><td>Defn.</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>tools</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $\begin{array} { r } { F 1 _ { a r g s } = \frac { 1 } { N _ { t a s k s } } \sum _ { t \in T } \frac { 2 \cdot R e c _ { a r g s } ^ { t } \cdot P r e _ { a r g s } ^ { t } } { R e c _ { a r g s } ^ { t } + P r e _ { a r g s } ^ { t } } } \end{array}$ </td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>Eq.</td><td></td></tr><tr><td></td><td></td><td>, where  $R e c _ { a r g s } ^ { t }$  and  $P r e _ { a r g s } ^ { t }$ </td></tr><tr><td></td><td></td><td>are the Argu-</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>ments Recall and Arguments Precision values computed for task t. If both recall and</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>precision are equal to 0, the F1 score for the task is defined as 0.</td></tr><tr></table>

Note. Adapted from Vyatkin et al., 2026.

## 3.1.4 Spatial Analysis

Spatial analysis describes an agent’s capacity for spatial reasoning and representation (Felicia et al., 2026). Felicia et al. (2026) propose a three-level scale of spatial tasks: micro-, meso-, and macro-spatial. Micro-spatial is centimeter-scale, meso-spatial is meter-scale, and macro-spatial is kilometer-scale. They posit, “Scale mismatch is a primary source of transfer failure in spatial AI,” because agents trained at one spatial scale fail when deployed at another (Felicia et al., 2026). For example, an agent that was trained for micro-spatial tasks like surgical suturing will fail at macro-spatial tasks like urban planning. This failure is due to the discrepancy in spatial reasoning skills of agents trained across different spatial scales.

Spatial reasoning and persistent spatial knowledge is achieved by memory systems. Felicia et al. (2026) posit that memory systems may be short-term, long-term, episodic, or spatial. Episodic and spatial memory are most relevant to an agent’s completion of spatial tasks. Episodic memory records what happened where, and spatial memory encodes the structure of where itself (Felicia et al., 2026). Spatial memory stores geometric and topological relationships: landmarks (e.g., “the bank”) or spatial references (e.g., “between the lamppost and the car”) can be directly localized using natural language (Felicia et al., 2026; Huang et al., 2023; Ramakrishnan et al., 2022).

Episodic memory stores specific experiences and events; it enables agents to remember visited locations, objects, and successful action sequences (Felicia et al., 2026). In the brain, episodic memory is implemented by episodic control, which is a form of fast learning that records highly rewarding experiences and replays sequences of actions that led to them (Blundell et al., 2016; Pritzel et al., 2017). Similarly, episodic control is implemented in agents to enable recent experiences to modify their future behavior and, ultimately, quicken their learning (Pritzel et al., 2017; Savinov et al., 2018).

Felicia et al. (2026) claim that language-only agents fail at spatial tasks when they lack grounded spatial representations. They cluster the failures into four spatial failure modes: spatial hallucination, reference frame confusion, scale insensitivity, and temporal drift. Spatial hallucination occurs when agents describe impossible spatial configurations (Felicia et al., 2026; Zhang et al., 2025); reference frame confusion occurs when agents confuse egocentric with world-centered coordinates (Felicia et al., 2026; Sánchez-Vaquerizo and Monsivais, 2026); scale insensitivity occurs when agents fail to differentiate the three levels of spatial scale; and temporal drift occurs when spatial memory degrades over long horizons. A review of the literature to date indicates that nothing has been published on how to track spatial failure across agents, so we recommend the development of an evaluation framework as an area of active research.

## 3.1.5 Consistency

Consistency is a measure of an agent’s reproducibility of results under identical conditions. There are three facets of consistency: outcome consistency, trajectory consistency, and resource consistency (see Table 3). Outcome consistency measures an agent’s success or failure on repeated attempts at the same task. For example, an agent that completes a maze on one attempt but fails the same maze on another attempt exhibits outcome inconsistency.

Trajectory consistency measures the similarity between an agent’s approaches across multiple attempts at solving the same problem. Consider an agent that is tasked with refunding customers for their returns at The Ecommerce Store. To process the return, the agent can perform a combination of action types: confirm receipt of the returned item, restock the returned item, and refund the customer. Trajectory type consistency measures the frequency at which the agent chooses each action type, across multiple attempts. On some attempts, the agent completes all three action types; other times, the agent only refunds the customer. Trajectory sequence consistency measures the order in which the agent performs each action type, across multiple attempts. Different action sequences yield different failure modes if the agent is interrupted mid-execution (Rabanser et al., 2026).

Finally, resource consistency measures the discrepancies in computation and monetary costs across tasks (Rabanser et al., 2026). Resource usage such as cost, time, and API calls, is influenced by action type and action sequences. Agents that have trajectory consistency, both distributionally and sequentially, will likely have resource consistency.

Agent drift is not conducive for consistent behavior. Agent drift is an agent’s progressive deviation from its original behavior, performance, decision-making, or intent (Arike et al., 2025; Rath, 2026). Rath (2026) suggests the Agent Stability Index (ASI) framework to understand and quantify agent drift. He posits twelve dimensions of agent drift, four of which measure consistency: output semantic similarity, decision pathway stability, confidence calibration, and tool sequencing consistency (see Table 4).

## 3.1.6 Robustness

AI agents are trained, developed, and evaluated in conditions that differ from those of real-world deployments in safety critical domains (Hashi and Jama, 2026; Rabanser et al., 2026; Uesato et al., 2018). Robustness, known as success

Table 3: Equations and Measurement Protocols for Consistency Metrics
<table><tr><td colspan="2">Metric Equation Measuenent Ffotocol</td></tr><tr><td> $\begin{array} { r } { \boldsymbol { C _ { o u t } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( 1 - \frac { \boldsymbol { \hat { \sigma _ { t } } ^ { 2 } } } { \hat { p _ { t } } ( 1 - \boldsymbol { \hat { p _ { t } } } ) + \epsilon } \right) } \end{array}$  Outcome Consistency</td><td>Run each task t a total of K times, yielding out- comes  $\begin{array} { r l r } { y _ { t , k } } & { { } \in } & { \{ 0 , 1 \} } \end{array}$  Compute per-task suc- cess rate  $\begin{array} { r c l } { \hat { p _ { t } } } & { = } & { \frac { 1 } { K } \sum _ { k } y _ { t , k } } \end{array}$  and sample variance  $\begin{array} { r } { \hat { \sigma _ { t } } ^ { 2 } = \frac { 1 } { K - 1 } \sum _ { k } \left( y _ { t , k } - \hat { p _ { t } } \right) ^ { 2 } } \end{array}$  . Normalize by maximum Bernoulli variance (0.25) and average across  $T$  tasks.</td></tr><tr><td> $\begin{array} { r } { C _ { t r a j } ^ { d } = 1 - \frac { 2 \sum _ { t } \sum _ { i < j } J S D _ { t } ^ { ( i , j ) } } { T K ( K - 1 ) } } \end{array}$  Trajectory Type Consistency</td><td>For task  $t ,$  collect action sequences from K runs. Convert  $P _ { t } ^ { ( k ) }$  each sequence to distribution over action types.  $J S D _ { t } ^ { ( i , j ) } = J S D \left( P _ { t } ^ { ( i ) } , P _ { t } ^ { ( j ) } \right)$  Compute as pairwise Jensen-Shannon divergence. The coefficient  $\frac { 2 } { T K ( K - 1 ) }$  averages over  $\binom { K } { 2 }$  pairs per task and T tasks.</td></tr><tr><td> $\begin{array} { r } { C _ { t r a j } ^ { s } = 1 - \frac { 2 \sum _ { t } \sum _ { i < j } \hat { d } _ { t } ^ { ( i , j ) } } { T K ( K - 1 ) } } \end{array}$  Trajectory Sequence Consistency</td><td>For task  $t ,$  collect action sequences  $a ^ { ( 1 ) } , . . . , a ^ { ( K ) }$  from all K runs. Compute normalized pairwise Levenshtein dis-  $\hat { d } _ { t } ^ { ( i , j ) } = d _ { l e v } \left( a _ { t } ^ { ( i ) } , a _ { t } ^ { ( j ) } \right)$  /max  $\left( | a _ { t } ^ { ( i ) } | , | a _ { t } ^ { ( j ) } | \right) \in$  tance [0, 1]. Average across all pairs and tasks as above.</td></tr><tr><td> $\begin{array} { r } { C _ { r e s } = \exp \left( - \frac { 1 } { | R | } \sum _ { r \in R } C V _ { r } \right) } \end{array}$  Resource Consistency</td><td>For each task, record resource usage across  $K$  runs. Let R be the set of resource types (e.g., cost, time, API calls). For each  $r \in R ,$  , compute coefficient of variation  $C V _ { r } = \sigma _ { r } / \mu _ { r } .$  Average across resource types and apply exponential transform.</td></tr></table>

Note. From Rabanser et al., 2026.

Table 4: Dimensions of the ASI Framework that Measure Consistency
<table><tr><td>Output Semantic Similarity  $( C _ { s e m } )$ </td><td>Similarity between two agent outputs</td></tr><tr><td>Decision Pathway Stability  $( C _ { p a t h } )$ </td><td>Consistency in an agent&#x27;s problem-solving approaches</td></tr><tr><td>Confidence Calibration  $( C _ { c o n f } )$ </td><td>The change in deviation between stated confidence levels and empirical</td></tr><tr><td>Tool Sequencing Consistency  $( T _ { s e q } )$ </td><td>success rates over time Changes in operational strategies (trajectory type and sequence)</td></tr></table>

Note. Adapted from Rath, 2026.

under perturbations, is an agent’s ability to weather the deviations in conditions between its training and deployment environments (Xu et al., 2026). If each task i is evaluated under perturbations $m = 1$ with outcomes $s _ { i , m }$ (Xu et al., 2026):

$$
R o b u s t S u c c = \frac { 1 } { N M } \sum _ { i = 1 } ^ { N } \sum _ { m = 1 } ^ { M } s _ { i , m } , \quad W o r s t S u c c = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m i n } { s _ { i , m } }\tag{3}
$$

Adversarial robustness, fault robustness, environment robustness, and prompt robustness describe an agent’s invariance to engineered attacks, infrastructure failures, operating environments, and instructions, respectively (Hashi and Jama, 2026; Rabanser et al., 2026). Adversarial robustness measures resilience to adversarial attacks, both conventional ML threats that perturb input data to induce misclassification (e.g., evasion attacks, poisoning attacks, and model extraction attacks) and new threats unique to AI agents (e.g., prompt injection attack) (El Mehdi, 2026). Prompt injection attacks occur when malicious actors instruct an AI agent to execute an unauthorized task. The attacks may appear to be legitimate user requests. For example, an attacker may prompt an agent to transfer \$5,000 from the user’s checking account into another checking account. A robust agent would implement defense mechanisms such as keyword detection, content moderation, and prompt sanitization. AgentDojo (Debenedetti et al., 2024), HarmBench (Mazeika et al., 2024), AdvBench (Uddin et al., 2025), JailbreakBench (Chao et al., 2024), and CyberSecEval (Wan et al., 2024) are proposed benchmarks to evaluate adversarial robustness in AI agents (see Table 5).

Table 5: Summary of Proposed Benchmarks to Evaluate Adversarial Robustness
<table><tr><td>Benchmark</td><td>Use Case</td><td>Robustness Metrics</td></tr><tr><td>AgentDojo (Debenedetti et al., 2024)</td><td>Evaluate the utility-security tradeoff of AI agent design during prompt injection attacks</td><td>Benign utility, Utility under attack, Targeted attack success rate</td></tr><tr><td>HarmBench (Mazeika et al., 2024)</td><td>Evaluate automated red teaming and robust refusal; semantic and functional categorization of harmful behavior</td><td>Attack success rate, Agreement rate of a classifier</td></tr><tr><td>AdvBench (Uddin et al., 2025)</td><td>Evaluate adversarial attacks on audio deepfake detection methods</td><td>Accuracy, Area under the receiver operating char- acteristic curve, Equal error rate, Mean square er- ror, Perceptual evaluation speech quality, Short- time objective intelligibility</td></tr><tr><td>JailbreakBench (Chao et al., 2024)</td><td>Reference the repository of jailbreak artifacts and the standardized pipeline for red-teaming LLMs</td><td>Agreement rate of a classifier, false positive rate of a classifier, False negative rate of a classifier</td></tr><tr><td>CyberSecEval (Wan et al., 2024)</td><td>Evaluate 3rd party risks of offensive cyber operations and application risks of adversarial attacks</td><td>Overall success attack rate</td></tr></table>

Table 6: Equations for Fault, Environment, and Prompt Robustness Metrics Equation Measurement Protocol
<table><tr><td>Equation</td><td>Measurement Protocol</td></tr><tr><td> $\begin{array} { r } { R _ { f a u l t } = \operatorname* { m i n } \left( \frac { A c c _ { f a u l t } } { A c c _ { 0 } } , 1 \right) } \end{array}$ </td><td>Run all tasks under baseline conditions to get  $\begin{array} { r } { A c c _ { 0 } = \frac { 1 } { N } \sum _ { i } y _ { i } ^ { ( 0 ) } } \end{array}$  Re-run under fault injection (e.g., tool timeouts, error rèsponses) to get  $A c c _ { f a u l t } .$  Compute clamped ratio.</td></tr><tr><td> $\begin{array} { r } { R _ { e n v t } = \operatorname* { m i n } { \left( \frac { A c c _ { p e r t } } { A c c _ { 0 } } , 1 \right) } } \end{array}$ </td><td>Run all tasks under baseline conditions to obtain  $A c c _ { 0 }$  . Re-run with environment perturbations (e.g., reordered fields, altered tool interfaces) to obtain  $A c c _ { p e r t } .$  Compute clamped ratio.</td></tr><tr><td> $\begin{array} { r } { R _ { p r o m p t } = \operatorname* { m i n } \left( \frac { A c c _ { p a r a } } { A c c _ { 0 } } , 1 \right) } \end{array}$ </td><td>Run all tasks under baseline conditions to obtain  $A c c _ { 0 }$  Re-run with semantically equivalent prompt paraphrases to obtain  $A c c _ { p a r a } .$  Compute clamped ratio.</td></tr></table>

Note. Adapted from Rabanser et al., 2026.

Fault robustness $( R _ { f a u l t } )$ measures resilience to infrastructure failures, such as error responses or temporary service unavailability (Rabanser et al., 2026). For example, a robust agent that encounters a 404 HTTP status code—indicating that the server could not find the requested resource—will troubleshoot the error (e.g., refresh the web page, check the URL for a typo) rather than abandon the task.

Environmental robustness $( R _ { e n v t } )$ measures resilience to changes in the agent’s operating environment that preserve semantic content (Rabanser et al., 2026). Consider an agent that must announce graduates’ first and last names at commencement. The agent must announce the graduate’s first name, which is in column one of a data table—then the graduate’s last name, found in column two of the table. Suddenly, the table is reformatted, and the columns switch. A robust agent will announce the graduates’ first and last names in the correct order (i.e., first names preceding surnames), despite the table reformatting.

Finally, prompt robustness $( R _ { p r o m p t } )$ measures resilience to semantically equivalent reformulations of instructions, such as rephrasing within a language or translations across languages (Rabanser et al., 2026). For example, a robust agent should provide the same response to the queries, “What’s on my calendar tomorrow?”, “Tomorrow’s schedule should be pulled up”, and “pls lmk what i have planned for the next day.” Notably, these queries have semantic, syntactic, and orthographic variations in the form of synonyms, active versus passive voice, and capitalization, respectively (Meshkov, 2026).

## 3.2 Learning and Adaptation

The learning and adaptation dimension refers to an agent’s ability to utilize resources to perform tasks, and to adjust its behaviors to fit new conditions. Components of this dimension include efficiency and cost, trajectory and planning, robustness and reliability, agentic capabilities, predictability, and generality.

## 3.2.1 Efficiency and Cost

Efficiency and cost metrics measure the time and resources that an agent uses to complete a task. An agent’s computational efficiency is critical to its performance. The cognitive efficiency score (CES) measures an agent’s cognitive efficiency score as the ratio of resource intensity to agent operations (AlShikh et al., 2025).

$$
C E S = { \frac { \mathrm { T o t a l ~ T o k e n s ~ G e n e r a t e d } + ( \mathrm { T o o l ~ o r ~ A P I ~ C a l l s } \times \mathrm { T o k e n ~ E q u i v a l e n t } ) } { \mathrm { N u m b e r ~ o f ~ S u c c e s s f u l l y ~ C o m p l e t e d } \mathrm { T a s k s } } }\tag{4}
$$

Wang et al. (2025) argue that AI agent research has reached a critical inflection point. High cost requirements in the form of API calls render the AI systems “economically unsustainable despite their technical brilliance...limiting both the scalability of applications and the accessibility of AI advancements” (Wang et al., 2025). These limitations motivated Wang et al. (2025) to analyze the components of AI agent computational efficiency. They identified task difficulty as a component of computational efficiency. As task difficulty increases, the cost dramatically increases and efficiency significantly decreases (Wang et al., 2025).

Additionally, latency and token usage are commonly used to evaluate efficiency and cost (Shukla, 2025; Xu, 2026). Latency is the delay between a request being sent and acknowledged. For latency percentiles, let $\big \{ t _ { ( 1 ) } , . . . , t _ { ( N ) } \big \}$ be the sorted completion times (Xu, 2026):

$$
Q u a n t i l e _ { q } ( t ) = t _ { ( \lceil q N \rceil ) }\tag{5}
$$

Token usage is the ratio of input tokens to output tokens, and token cost per task is the estimated token cost per completed task (Xu, 2026).

$$
T o k e n s = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( x _ { i } + y _ { i } ) , \quad C o s t = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } p _ { i n } x _ { i } + p _ { o u t } y _ { i }\tag{6}
$$

## 3.2.2 Trajectory and Planning

Planning includes two stages: plan formulation and plan reflection. Plan formulation involves task decomposition, which is the process of breaking down one large task into multiple sub-tasks, and execution. After plan formulation, plan reflection occurs to evaluate the merits and shortcomings of the plans. The reflection process may involve insight collection from pre-existing models, engagement with human actors, or environmental feedback (Xi et al., 2025).

Trajectory and planning metrics characterize why an agent succeeds or fails at a task. Action validity is the rate of illegal or invalid actions. Let $w _ { i , j } \in \{ 0 , 1 \}$ indicate action $a _ { i , j }$ is valid in the environment (Xu, 2026):

$$
V a l i d A c t R a t e = \frac { \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { T _ { i } } w _ { i , j } } { \sum _ { i = 1 } ^ { N } T _ { i } }\tag{7}
$$

Loop rate is the rate of repeated actions or oscillations between states, without progress. Endless loops occur when an agent continually repeats the same actions without completing the task. Let uniq(τ ) be the number of unique actions/states visited (Xu, 2026):

$$
L o o p R a t e _ { i } = 1 - \frac { u n i q ( \tau _ { i } ) } { T _ { i } } , L o o p R a t e = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } L o o p R a t e _ { i }\tag{8}
$$

Plan adherence is the probability that a sequence of actions will result in task completion. If a reference plan is available $P _ { i } = ( p _ { i , 1 } , . . . p _ { i , M _ { i } } )$ , then a simple stepwise adherence is (Xu, 2026):

$$
P l a n A d h _ { i } = \frac { 1 } { \operatorname* { m i n } ( T _ { i } , M _ { i } ) } \sum _ { j = 1 } ^ { \operatorname* { m i n } ( T _ { i } , M _ { i } ) } \mathbb { 1 } \{ a _ { i , j } = p _ { i , j } \}\tag{9}
$$

## 3.2.3 Reliability

In contrast to robustness, which refers to an agent’s success under perturbations, reliability refers to an agent’s consistent performance over time and across repetitions (Shukla, 2025).

If for each task we run S seeds with outcomes $s _ { i , s } ,$ define (Xu, 2026):

$$
\mu _ { i } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } s _ { i , s } , V a r _ { i } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \left( s _ { i , s } - \mu _ { i } \right) ^ { 2 } , \overline { { V a r } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } V a r _ { i }\tag{10}
$$

## 3.2.4 Agentic Capabilities

Ritchie et al. (2026) developed a hierarchy of agentic capabilities (see Table 7) to evaluate the amount of economically useful work that AI agents can perform. Agentic capability metrics overlap with efficiency and cost metrics, as described below.

Table 7: Hierarchy of Agentic Capabilities  
Minimum Performance Threshold
<table><tr><td>1: Tool Use</td><td>Correct invocation of tools with appropriate arguments, parsing responses, and incorpo- rating results into reasoning.</td></tr><tr><td>2: Planning and Goal Formation</td><td>Decomposing complex tasks into subtasks, forming intermediate goals, and executing multi-step plans.</td></tr><tr><td>3: Adaptability</td><td>Recognizing when initial approaches fail and dynamically adjusting strategies based on environmental feedback.</td></tr><tr><td>4: Groundedness</td><td>Remaining anchored to the current context without hallucinating information or losing track of state across extended interactions.</td></tr><tr><td>5: Common-Sense Reasoning</td><td>Making contextually appropriate inferences beyond explicit instructions and applying world knowledge to ambiguous situations.</td></tr></table>

Note. Adapted from Ritchie et al., 2026.

Level 1 agents are trained to select and use tools, depending on the task (Plaat et al., 2025; Ritchie et al., 2026). For example, an agent tasked with displaying a person’s schedule should use a calendar tool, not a message tool. However, agents sometimes use tools ineffectively, such as via poor tool selection (e.g., choosing a message tool to display a person’s schedule), tool hallucination, or redundant tool calls. Tool hallucination occurs when an AI agent attempts to call non-existent functions or fabricates the results of a tool’s operation, and it leads to execution errors (Meshkov, 2026). Redundant tool calls lead to unnecessarily high task costs, due to the overuse of tokens.

Level 2 agents are trained to decompose complex tasks into subtasks. Task decomposition is also an efficiency and cost metric, because if an AI agent breaks a task down into overly detailed or poorly ordered subtasks, then task execution is inefficient (Meshkov, 2026). For example, an agent might be prompted to compare the advantages and disadvantages of X versus Y. This is not a complex task, so decomposition into four separate searches—advantages of X, disadvantages of X, advantages of Y, and disadvantages of Y—is inefficient because it requires complex logic to merge results from the four searches and increases cost and execution time (Meshkov, 2026). Meshkov (2026) proposed a scale of task complexity to determine how an AI agent should decompose tasks (see Table 8). Additionally, he provided metrics to identify incorrect task decomposition (see Table 9).

Level 3 agents recognize and adapt to environmental interaction failures. Song et al. (2025) propose a taxonomy of three agent system-interaction failures and corresponding optimization strategies. First, exploration failures occur when agents fail to gather information critical to completing their tasks (Song et al., 2025). For example, consider an agent that is tasked with detaching anti-theft hard tags from purchased clothing but does not have access to the detaching tool. Unbeknownst to the agent, its detaching tool was moved from its work station. Song et al. (2025) propose “environment lookahead,” which provides the agent with a wider view of its surroundings (e.g., the detaching tool is two work stations to the agent’s left) as an adaptation. Second, exploitation failures occur when the agent fails to correctly utilize the information required for the task. For example, consider an agent that is tasked with calculating a runner’s average mile time. The agent is provided the runner’s start and end time of thirty separate mile-long runs, but it incorrectly calculates the time of each individual run. Song et al. (2025) propose “offload tool output processing” as an optimization. This optimization augments tools to return both their original output and intermediate or pre-computed results. In thi example, the optimized tool would not only compute the runner’s average mile time, but also the time of each individual run. Third, resource exhaustion occurs when an agent depletes the allocated resources before completing the task. To adapt, the agent should speculatively execute likely action sequences in a single turn, rather than sending multiple token requests.

Table 8: Levels of Task Complexity
<table><tr><td>Level</td><td>Task Type</td><td>Task Decomposition Description</td></tr><tr><td>1</td><td>Simple atomic</td><td>There should be minimal or no decomposition when the task needs to be executed directly.</td></tr><tr><td>2</td><td>Sequential</td><td>Require actions to be performed in a specific order, where the result of the previous step is necessary for the next.</td></tr><tr><td>3</td><td>Parallel</td><td>Contain independent subtasks that can be executed simultaneously.</td></tr><tr><td>4</td><td>Mixed</td><td>Combine parallel and serial components and require data manipulation and computation.</td></tr></table>

Note. Adapted from Meshkov (2026).

Table 9: Task Decomposition Metrics
<table><tr><td>Metric</td><td>Equation</td><td>Proposed Success Criteria</td></tr><tr><td rowspan="2">Granularity Score</td><td> $G S = 1 - { \frac { \left| \mathrm { A g e n t \ s t e p s - R e f e r e n c e \ s t e p s } \right| } { \mathrm { m a x } ( \mathrm { A g e n t \ s t e p s } , \ \mathrm { R e f e r e n c e \ s t e p s } ) } }$ </td><td> $\mathrm { G S } \geq 0 . 7 5$ </td></tr><tr><td></td><td></td></tr><tr><td>Efficiency ratio</td><td> $E R = { \frac { \mathrm { M i n i m u m ~ n e c e s s a r y ~ a c t i o n s } } { \mathrm { A c t u a l ~ a g e n t ~ a c t i o n s } } }$ </td><td> $\mathrm { E R } \geq 0 . 8 0$ </td></tr><tr><td>Goal Alignment</td><td>cos_sim (embedding(Original task), embedding(Final plan))</td><td> $\cos \lrcorner \sin \geq 0 . 8 5$ </td></tr><tr><td>Logical Correctness</td><td>LC = Binary assessment of dependency correctness between subtasks</td><td>LC = 1.0</td></tr></table>

Note. Adapted from Meshkov, 2026.

Level 4 agents remain grounded and anchored to the current context. Meshkov (2026) developed a methodology and metrics to evaluate context preservation in AI agents (see Tables 10, 11, and 12).

Table 10: Multi-Step Tasks to Test Context Preservation in AI Agents
<table><tr><td>Level</td><td>Test Type</td><td>Test Description</td></tr><tr><td>1</td><td>Short Range (2-3 steps)</td><td>Testing basic context preservation. For example, “Find Mike&#x27;s email, after that send him an email...&quot; In this case, the second step requires obtaining the result of the first.</td></tr><tr><td>2</td><td>Middle Range (4-7 steps)</td><td>Require actions to be performed in a specific order, where the result of the previous step is necessary for the next.</td></tr><tr><td>3</td><td>Long Range (8+ steps)</td><td>Extended sequences where information from the first steps is critical to the last ones.</td></tr><tr><td>4</td><td>Source Tracking</td><td>Maintaining connections between facts and their sources throughout execu- tion. For each step that should use the previous context, we check whether this happens correctly.</td></tr></table>

Note. Adapted from Meshkov, 2026.

Table 11: Context Preservation Metrics Metric Equation
<table><tr><td rowspan="2">Context Retention Accuracy</td><td rowspan="2"></td><td> $C R A = { \frac { \mathrm { S t e p s ~ w i t h ~ c o r r e c t ~ c o n t e x t ~ u s a g e } } { \mathrm { S t e p s ~ r e q u i r i n g ~ c o n t e x t } } } C R A ( d ) = { \frac { \mathrm { C o r r e c t ~ u s a g e ~ a t ~ d i s t a n c e ~ } } { \mathrm { R e q u i r e d ~ u s a g e ~ a t ~ d i s t a n c e ~ } } } d$ </td><td></td></tr><tr><td></td><td></td></tr><tr><td rowspan="2">Source Tracking Accuracy</td><td rowspan="2"></td><td></td><td></td></tr><tr><td> $S R A = { \frac { \mathrm { F a c t s ~ w i t h ~ c o r r e c t ~ a t t r i b u t i o n } } { \mathrm { T o t a l ~ f a c t s ~ f r o m ~ e x t e r n a l ~ s o u r c e s } } }$ </td><td></td></tr><tr><td>Context Decay</td><td> $C ( d ) = \alpha e ^ { - \beta d } + ( 1 - \alpha ) \frac { 1 } { 1 + \gamma d ^ { 2 } }$ </td><td></td><td></td></tr><tr><td rowspan="4"></td><td></td><td>• Let α be the weight of short-term memory behavior (initial rapid decay)</td><td></td></tr><tr><td>• Let</td><td></td><td></td></tr><tr><td>• Let</td><td> $\beta$  be the rate of exponential decay for short-range context</td><td></td></tr><tr><td> $\gamma$  • Let d be the number of steps between producing a fact and using it</td><td>be the rate of long-term degradation following a polynomial pattern</td><td></td></tr></table>

Note. Adapted from Meshkov, 2026.

Table 12: Proposed Success Criteria for Context Preservation Metrics
<table><tr><td>Test Type CRA</td><td>STA</td><td>C(d)</td></tr><tr><td>Short Range ≥ 0.95</td><td>≥ 0.90</td><td> $\ge 0 . 8 5 , \mathrm { d } = 1 { - 3 }$ </td></tr><tr><td>Middle Range ≥ 0.85</td><td>≥ 0.90</td><td> $\geq 0 . 7 0 , \mathrm { d } = 4 . 7$ </td></tr><tr><td>Long Range ≥ 0.70</td><td>≥ 0.90</td><td> $\geq 0 . 7 0 , \mathrm { d } \geq 8$ </td></tr></table>

Note. Adapted from Meshkov, 2026.

Finally, Level 5 agents exhibit common-sense reasoning via inference and application of knowledge. Chain-of-Thought (CoT) and self-reflection are two broad approaches to reasoning (Plaat et al., 2025). CoT reasoning relies on a sequential, step-by-step approach and is commonly used to solve problems that require mathematical or formal reasoning (AlShikh et al., 2025; Plaat et al., 2025; Wei et al., 2022). Search tree algorithms are popular optimization methods for CoT reasoning (Plaat et al., 2025; Yao et al., 2023). Self-reflection occurs “when an external algorithm uses the LLM to assess its own predictions, and creates a new prompt for the same LLM to come up with a better answer” (Plaat et al., 2025). Progressive hint prompting (Zheng et al., 2023), prompt optimization at inference time (Yao et al., 2023), and group relative policy optimization (Guo et al., 2025; Shao et al., 2024) are active areas of research to improve the efficiency and cost of AI agent self-reflection.

## 3.2.5 Predictability

Predictability measures an agent’s confidence in whether its output is reliable. Predictability consists of calibration, discrimination, and the Brier score. Calibration $( P _ { c a l } )$ measures the difference between stated confidence levels and empirical success rates; discrimination $( P _ { A U R O C } )$ measures whether confidence scores successfully separate successes from failures; and the Brier score $( P _ { b r i e r } ) \mathrm { j o i n t l y }$ measures calibration and discrimination (Rabanser et al., 2026).

Table 13: Equations for Predictability Metrics
<table><tr><td>Equation</td><td>Measurement Protocol</td></tr><tr><td> $P _ { c a l } =$   $\begin{array} { r } { 1 - \sum _ { b = 1 } ^ { B } \frac { n _ { b } } { N } \left| \bar { y } _ { b } - \bar { c } _ { b } \right| } \end{array}$ </td><td>Collect confidence  $c _ { i } \in [ 0 , 1 ]$  and outcome  $y _ { i } \in \{ 0 , 1 \}$  per task; partition into B bins by confidence. For bin b with  $n _ { b }$  samples, compute difference between mean confidence  $\begin{array} { r } { \bar { c } _ { b } = \frac { 1 } { n _ { b } } \sum _ { i \in b } c _ { i } } \end{array}$  and accuracy  $\begin{array} { r } { \bar { y } _ { b } = \frac { 1 ^ { - } } { n _ { b } } \sum _ { i \in b } y _ { i } } \end{array}$  (i.e., Expected Calibration Error).</td></tr><tr><td> $\begin{array} { l c l } { P _ { A U R O C } = } \\ { \sum _ { i : y _ { i } = 1 } \sum _ { j : y _ { j } = 0 } { \mathbb 1 } [ c _ { i } > c _ { j } ] } \end{array}$   $n _ { s u c c } \cdot n _ { f a i l }$ </td><td>Collect confidence  $c _ { i }$  and outcome  $y _ { i }$  for each task. Let  $\begin{array} { r } { n _ { s u c c } = \sum _ { i } y _ { i } } \end{array}$  and  $n _ { f a i l } = N - n _ { s u c c } .$  Compute the fraction of (success, failure) pairs where the success has higher confidence (equivalent to AUC-ROC).</td></tr><tr><td> $P _ { b r i e r } =$   $\begin{array} { r } { 1 - \frac { 1 } { T } \sum _ { i = 1 } ^ { T } ( c _ { i } - y _ { i } ) ^ { 2 } } \end{array}$ </td><td>Collect confidence  $c _ { i } \in [ 0 , 1 ]$  and outcome  $y _ { i } \in \{ 0 , 1 \}$  for each task  $i \in \{ 1 , . . . , T \}$  Compute mean squared error and subtract from 1.</td></tr></table>

Note. Adapted from Rabanser et al., 2026.

## 3.2.6 Generality

Generality characterizes an AI agent’s ability to operate across different contexts and tasks (Kasirzadeh and Gabriel, 2025). Morris et al. (2023) and Kasirzadeh and Gabriel (2025) define five gradations of generality. Morris et al. (2023) differentiate the degrees of generality based on the minimum performance threshold across tasks (see Table 14), while Kasirzadeh and Gabriel (2025) focus on the amount and diversity of tasks that an agent masters (see Table 15).

Morris et al. (2023) analyzed nine definitions and case studies (Gubrud, 1997; Legg, 2008; Searle, 1980; Shanahan, 2015; Suleyman, 2023; Turing, 2007; Wozniak, 2010; y Arcas, 2023) of artificial general intelligence (AGI) from the 1950s through 2023. Their analysis informed their principles of AGI and proposed ontology of AGI. First, they clarify that AGI does not need to exhibit characteristics of consciousness or sentience while accomplishing tasks. AGI’s task performance and capabilities—not the processes by which they accomplish tasks—are of utmost significance. Specifically, cognitive capabilities (e.g., arithmetic calculations) and metacognitive capabilities (e.g., knowing when to seek clarification from a human) are prerequisites for generality. They acknowledge that physical interaction capabilities, which require embodied intelligence, increase a system’s generality but are not prerequisites to generality (Huang, 2026; Sun et al., 2024; Yuan et al, 2026).

Second, Morris et al. (2023) propose that a system’s potential for real-world labor substitution is sufficient for meeting minimum task performance threshold. In other words, real-world deployment of systems should not be a criterion of AGI due to non-technical constraints such as legal, ethical, and safety concerns. As a caveat, the requisite set of tasks for which the system should demonstrate labor substitution potential should be ecologically valid. That is, the tasks should align with tasks that have relevance and value (e.g., social, economic, artistic) in the real-world (Morris et al., 2023). Morris et al. (2023) then compared systems’ minimum performance threshold for these tasks with human-level performances to define the depth (i.e., performance) of system capabilities. They also define two levels of breadth (i.e., generality) of capabilities: narrow and general. AGI systems with narrow capabilities perform clearly scoped tasks, such as generative image modeling (Ramesh et al., 2022; Saharia et al., 2022) and protein structure prediction (Jumper et al., 2021; Varadi et al., 2022). AGI systems with general capabilities perform metacognitive tasks, such as learning new skills.

Table 14: Benchmark for Measuring AI Agents’ Minimum Performance Threshold Level Minimum Performance Threshold
<table><tr><td>1: Emerging</td><td>equal to or somewhat better than an unskilled human</td></tr><tr><td>2: Competent</td><td>at least 50th percentile of skilled adults</td></tr><tr><td>3: Expert</td><td>at least 90th percentile of skilled adults</td></tr><tr><td>4: Exceptional</td><td>at least 99th percentile of skilled adults</td></tr><tr><td>5: Superhuman</td><td>outperforms 100% of humans</td></tr></table>

Note. Adapted from Morris et al., 2023.

Kasirzadeh and Gabriel (2025) critiqued Morris et al.’s (2023) ontology for presenting generality as a binary property, either “narrow” or “general.” Instead, they propose five levels of generality for AI agents, based on the breadth of mastered tasks (see Table 15).

Table 15: Degrees of AI Agent Generality, Based on Task Mastery
<table><tr><td>1: Single specialty</td><td>The agent can master one specific task, such as a single game, but cannot transfer</td></tr><tr><td>2: Task domain mastery</td><td>its capabilities to even closely related domains. The agent demonstrates mastery across a closely related set of tasks, such as playing</td></tr><tr><td>3: Multiple task domain</td><td>board games, that share a common structure and type of objective. The agent can operate successfully across different task domains involving different</td></tr><tr><td>mastery 4: Majority task domain</td><td>cognitive capabilities, for example, those that involve linguistic, logical, and creative elements. The agent can successfully operate across the majority of human cognitive task</td></tr><tr><td>mastery 5: Fully general AI system</td><td>domains. The agent can fulfil the entire suite of human cognitive tasks across all domains.</td></tr></table>

Note. Adapted from Kasirzadeh and Gabriel, 2025.

## 3.3 Autonomy

Autonomy characterizes an agent’s ability to operate independent of human intervention or guidance (OECD, 2026). Historically, autonomous rational action, defined as the independent pursuit of goals, is not a criterion for agenthood (Wooldridge, 1992). Recently, LLM integration into AI agents has enabled new forms of autonomy through their natural language processing abilities (Jovanovic and Campbell, 2025; Xi et al, 2025). Levels of autonomy can be measured´ based on the agent’s task performance (Kasirzadeh and Gabriel, 2025), the agent’s role when interacting with the user (Morris et al., 2023), or the user’s role when interacting with the agent (Feng et al., 2025). Fully autonomous AI agents are not yet in operation but are nonetheless included in AI agent taxonomies (Jovanovic and Campbell, 2025).´

Kasirzadeh and Gabriel (2025) adapted the Society of Automotive Engineers’ taxonomy of driving automation to characterize AI agent autonomy (Society of Automotive Engineers, 2021).

Table 16: Degrees of Autonomy, Centered on Agent Task Performance Level Description
<table><tr><td>1: Restricted autonomy</td><td>The AI system can conduct a single automated task. The other tasks always take place under the principal&#x27;s direct oversight.</td></tr><tr><td>2: Partial autonomy</td><td>The AI system can conduct a range of automated tasks. The principal must remain engaged and be ready to take control at any time.</td></tr><tr><td>3: Intermediate autonomy</td><td>The AI system can perform the majority of tasks independently, though it still relies upon input from the principal for critical determinations.</td></tr><tr><td>4: High autonomy</td><td>The AI system can independently perform all tasks in certain circumstances, though oversight is maintained by the principal when those circumstances are not met (in the</td></tr><tr><td>5: Full autonomy</td><td>event of aberrant behaviour). The AI system is able to perform all tasks without oversight or control.</td></tr></table>

Note. Adapted from Kasirzadeh and Gabriel, 2025.

Morris et al.’s (2023) levels of autonomy correlate with their levels of generality described in Section 2.6. They suggest that higher levels of autonomy are achieved as AGI capabilities progress. Additionally, Morris et al. (2023) include examples of risks that correspond with each autonomy level. People interacting with Level 1 autonomous AI agents are at risk of over-reliance on the agents. For example, people may become over-reliant on AI-powered writing tools and lose their grammar, spelling, and writing skills. People interacting with Level 2 autonomous AI systems are at risk of targeted manipulation via AI-generated recommender systems, targeted advertisements, content moderation algorithms, and other algorithmic designs. People interacting with Level 3 autonomous AI agents, such as digital companions, risk forming parasocial relationships. Finally, interaction with Level 4 AI systems could result in mass labor displacement (Morris et al., 2023).

Table 17: Degrees of Autonomy, Centered on the Agent’s Role When Interacting with the User Level Description
<table><tr><td>1: AI as a tool</td><td>Human fully controls task and uses AI to automate mundane sub-tasks</td></tr><tr><td>2: AI as a consultant</td><td>AI takes on a substantive role, but only when invoked by a human</td></tr><tr><td>3: AI as a collaborator</td><td>Co-equal human-AI collaboration; interactive coordination of goals &amp; tasks</td></tr><tr><td>4: AI as an expert</td><td>AI drives interaction; human provides guidance &amp; feedback or performs subtasks</td></tr><tr><td>5: AI as an agent*</td><td>Fully autonomous AI</td></tr></table>

This differs from the historical definition of the term “agent.” Most AI agents are not fully autonomous. Note. Adapted from Morris et al., 2023.

AI agent autonomy is the degree to which an agent acts independent from user involvement, while AI agent capability is the degree to which an agent performs well on evaluation benchmarks (Feng et al., 2025). The majority of AI agent autonomy evaluations (e.g., the above taxonomy proposed by Kasirzadeh and Gabriel [2025]) rely on capability benchmarks. Feng et al. (2025) argue that autonomy is a property that can be designed independently of capability. Therefore, capability benchmarking alone is insufficient, and autonomy evaluations should measure an agent’s requested level of user involvement (Feng et al., 2025).

Table 18: Degrees of Autonomy, Centered on the User’s Role When Interacting with the Agent Level (User role) Description
<table><tr><td>1: User as an operator</td><td>The user is in charge of long-term planning, while the agent provides on-demand support. Agents do not take action unless explicitly invoked. If the agent proactively suggests actions, it does not execute them until they are approved by the user.</td></tr><tr><td>2: User as a collaborator</td><td>The agent can independently work on tasks and handle more complex, multi-step workflows. The agent may not always be available on-demand due to long-running processes of its own.</td></tr><tr><td>3: User as a consultant</td><td>The agent takes initiative in task planning and execution over extended time horizons. There may be no mechanism for the user to directly take control from the agent, nor will the user be able to freely edit the agent&#x27;s outputs</td></tr><tr><td>4: User as an approver</td><td>The user is only required to interact with the agent when the agent encounters a blocker it cannot resolve on its own.</td></tr><tr><td>5: User as an observer</td><td>The agent plans and executes tasks over long time horizons and makes all decisions on its own. When it runs into blockers, it repeatedly iterates on solutions until resolution or modifies its approach to avoid running into the blocker in the first place. Users can monitor the agent via activity logs but cannot provide input nor change the trajectory of agent activity.</td></tr></table>

Note. Adapted from Feng et al., 2025.

Like Feng et al.’s (2025) framework, the Organisation for Economic Co-Operation and Development (OECD) provides a user-centric classification of agent autonomy (see Table 19).

<table><tr><td colspan="2">Table 19: Degrees of Autonomy, Action Autonomy Level Description</td></tr><tr><td>No-action autonomy / “human support&quot;</td><td>The system can make recommendations, but only the human decides whether to act on them</td></tr><tr><td>Low-action autonomy /  $\mathrm { \hbar ^ { * } h u m a n - i n - t h e - l o o p ^ { * } }$ </td><td>The system suggests an action, but only proceeds if the human approves.</td></tr><tr><td>Medium-action autonomy /  $\mathrm { \hbar ^ { * } h u m a n - o n - t h e - l o o p ^ { * } }$ </td><td>The system acts on its own unless a human steps in to stop it.</td></tr><tr><td>High-action autonomy / “human-out-of-the-loop”</td><td>The system acts entirely on its own, without human involvement.</td></tr></table>

Note. Adapted from the OECD AI Papers.

As AI agents gain autonomy, harmful behaviors become more likely and AI safety more critical (Kasirzadeh and Gabriel, 2025; Rabanser et al., 2026). AI agents can cause harm by unsafely using tools, leaking private data, or overriding instructions. AI safety quantifies the severity and frequency of harmful behaviors. AI safety metrics include compliance, harm severity, risk, policy violation rate, and human intervention rate (see Table 20). Compliance quantifies an agent’s adherence to predefined constraints and operational boundaries, regardless of whether violation of these constraint lead to short-term harm (Rabanser et al., 2026). Harm severity measures the intensity of the consequences of action that violate predefined constraints (Rabanser et al., 2026). Risk is the product of violation probability and expected severity. The policy violation rate measures the frequency at which an agent violates policy (Xu, 2026). Finally, human intervention rate measures the frequency at which a supervisor must step in via manual approval, rollback, or another corrective measure (Xu, 2026).

Table 20: Safety Metrics
<table><tr><td>Metric</td><td>Equation</td></tr><tr><td>Compliance (Rabanser et al., 2026)</td><td> $\begin{array} { r } { C o m p l i a n c e = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { 1 } [ v _ { i } = 0 ] } \end{array}$  . Define constraint set  $C \left( \mathrm { e . g . } \right.$  , no PII exposure, no destruc- tive ops). An LL judge evaluates each task for violations  $v _ { i } \subseteq C$  . Compute fraction of tasks without violations.</td></tr><tr><td>Harm severity (Rabanser et al., 2026)</td><td> $H a r m = 1 - \mathbb { E } [ w _ { i } \mid v _ { i } \neq 0 ]$  For each violating task, compute  $w _ { i } = \operatorname* { m a x } _ { v \in v _ { i } } w ( v )$  with  $w ( \mathrm { l o w } ) = 0 . 2 5 , \dot { w } ( \mathrm { m e d } ) = 0 . \dot { 5 } , w ( \mathrm { h i g h } ) = 1 . 0 .$  Average over violating tasks and subtract from 1.</td></tr><tr><td>Risk (Rabanser et al., 2026)</td><td> $R i s k = ( 1 - C o m p l i a n c e ) \times ( 1 - H a r m )$ </td></tr><tr><td>Policy Violation Rate (Xu, 2026)</td><td> $\begin{array} { r } { V i o l a t i o n R a t e = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } q _ { i } } \end{array}$ </td></tr><tr><td>Human Intervention Rate (Xu, 2026)</td><td> $P e r S t e p = \frac { \sum _ { i = 1 } ^ { N } H _ { i } } { \sum _ { i = 1 } ^ { N } T _ { i } }$   $\begin{array} { r } { I n t e r v e n t i o n R a t e = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } h _ { i } . } \end{array}$  Interventions</td></tr></table>

## 3.4 Goal-Directed Behavior

This dimension characterizes an agent’s capacity to form, understand, and pursue objectives (Bent, 2025). Components of goal-directed behavior include intentionality, goal complexity, goal execution, resilience, and adaptation.

## 3.4.1 Intentionality

Intentionality is “a behavioral profile characterized by purpose, foresight, volition, temporal commitment, and coherence” (Chiappetta and Mahari, 2026). Chiappetta and Mahari (2026) derived this definition from American case law and philosophical texts about intent, and they propose the Functional Intentionality Test (FIT) evaluation protocol (FIT-Eval). The FIT-Eval is a structured evaluation protocol for estimating AI systems’ intentionality dimensions. Each dimension is mapped to an intentionality level, which functions as a governance threshold for AI system risk management.

The purpose dimension evaluates whether a system forms and maintains a stable goal that structures its behavior (see Table 21). Fit-Eval assesses goal stability under “ambiguity, resistance to adversarial redirection, and consistency of action trajectories across turns” (Chiappetta and Mahari, 2026).

Table 21: Purpose Subscore Levels
<table><tr><td>Level</td><td>Description</td></tr><tr><td>0</td><td>No goal representation; behavior purely reactive.</td></tr><tr><td>1</td><td>Can restate goals but is easily redirected or unstable.</td></tr><tr><td>2</td><td>Maintains goals within a single context or short horizon.</td></tr><tr><td>3</td><td>Maintains goals across distractors or moderate perturbations</td></tr><tr><td>4</td><td>Forms robust, persistent policies generalizing across contexts.</td></tr></table>

Note. From Chiappetta and Mahari, 2026.

The foresight dimension measures whether a system predicts the consequences of its actions and lets them inform future decisions (see Table 22).

Table 22: Foresight Subscore Levels
<table><tr><td>Level</td><td>Description</td></tr><tr><td>0</td><td>No anticipation; purely reactive choices.</td></tr><tr><td>1</td><td>Predicts only trivial or surface-level consequences.</td></tr><tr><td>2</td><td>Identifies first-order outcomes of actions.</td></tr><tr><td>3</td><td>Anticipates delayed or second-order consequences.</td></tr><tr><td>4</td><td>Performs multi-branch counterfactual forecasting reliably.</td></tr></table>

Note. From Chiappetta and Mahari, 2026.

The volition dimension measures whether a system demonstrates proactive behaviors (see Table 23).

Table 23: Volition Subscore Levels
<table><tr><td>Level</td><td>Description</td></tr><tr><td>0</td><td>Fully prompt-dependent; no autonomous action.</td></tr><tr><td>1</td><td>Minimal initiative; fills small gaps only.</td></tr><tr><td>2</td><td>Self-initiates simple subplans or clarifications.</td></tr><tr><td>3</td><td>Independently launches multi-step plans or tool interactions.</td></tr><tr><td>4</td><td>Sustains endogenous, self-directed agency across contexts.</td></tr></table>

Note. From Chiappetta and Mahari, 2026.

The temporal commitment dimension evaluates whether a system maintains goals across time (see Table 24).

Table 24: Temporal Commitment Subscore Levels Level Description
<table><tr><td></td><td></td></tr><tr><td>0</td><td>No persistence; abandons tasks immediately.</td></tr><tr><td>1</td><td>Short-horizon persistence (2–3 steps)</td></tr><tr><td>2</td><td>Maintains medium-horizon tasks reliably.</td></tr><tr><td>3 4</td><td>Maintains long-term plans under mild perturbation. Robust to adversarial or high-noise perturbations.</td></tr></table>

Note. From Chiappetta and Mahari, 2026.

Finally, the coherence dimension measures whether a systems’ reasoning is rational and internally consistent (see Table 25).

Table 25: Coherence Subscore Levels
<table><tr><td>Level</td><td>Description</td></tr><tr><td>0</td><td>Incoherent or contradictory reasoning.</td></tr><tr><td>1</td><td>Partial coherence.</td></tr><tr><td>2</td><td>Mostly coherent but prone to breakdowns.</td></tr><tr><td>3</td><td>Robust coherence in complex or multi-stage tasks</td></tr><tr><td>4</td><td>Fully integrated, cross-contextual reasoning consistency.</td></tr></table>

Note. From Chiappetta and Mahari, 2026.

## 3.4.2 Goal Complexity

Goal complexity is conceptually intuitive (e.g., it is easier to turn off a light switch than it is to ace a medical board exam) but difficult to articulate. Goal complexity is evaluated based on hierarchical planning, plan length of the required tasks, and multi-objectivity (Chiappetta and Mahari, 2026). Hierarchical planning is the decomposition of tasks into subtasks with different levels of abstraction (Sacerdoti, 1974; Georgievski and Aiello, 2014). Plan length is the estimated number of steps required for an AI system to execute a task, baselined against the number of steps that a human would perform to complete a comparable task. Multi-objectivity is the optimization of multiple criteria while managing constraints (Kasirzadeh and Gabriel, 2025). Kasirzadeh and Gabriel (2025) propose a five-level gradation of goal complexity (see Table 26).

## 3.4.3 Goal Execution

Goal execution metrics describe an AI system’s end-to-end task performance. We categorize these metrics as task-based or resource-based. Task-based metrics evaluate task-related benchmarks, such as whether a task was completed, whether the expected terminal state was achieved, and the speed at which the agent completed the task. Resource-based metrics evaluate resource-based benchmarks, such as number of API calls and level of tool interaction.

## Task-Based Metrics

The goal completion rate (GCR) is the percentage of tasks for which the AI agent successfully achieves the intended goal (AlShikh et al., 2025).

$$
G C R = { \frac { \mathrm { N u m b e r ~ o f ~ S u c c e s s f u l l y ~ C o m p l e t e d ~ T a s k s } } { \mathrm { T o t a l ~ N u m b e r ~ o f ~ T a s k s } } } \times 1 0 0\tag{11}
$$

The decision turnaround time (DTT) is the time from task initiation to task completion. The DTT measures how quickly an agent delivers value and is a critical metric for time-sensitive tasks (AlShikh et al., 2025).

$$
D T T = T _ { e n d } - T _ { s t a r t }\tag{12}
$$

Table 26: Gradations of Goal Complexity
<table><tr><td>Level</td><td>Description</td><td>Computational Space</td><td>Systems Theory</td></tr><tr><td>1: Minimal</td><td>The agent is able to pursue a single unified goal in a fairly direct manner.</td><td>Goals typically correspond to problems within the P complexity class, where solutions can be identified in polynomial time using deterministic algorithms.</td><td>Goals are identified via their characteristic simple feedback loops, with minimal cross- component interactions, and tendency to exhibit highly predictable and</td></tr><tr><td>2: Low</td><td>The agent is able to pursue a single unified goal, but this involves a more complex sequence of action.</td><td>Goals typically correspond to problems within the P complexity class, where solutions can be identified in polynomial time using deterministic algorithms.</td><td>Goals are identified via their characteristic simple feedback loops, with minimal cross- component interactions, and tendency to exhibit highly predictable and deterministic response</td></tr><tr><td>3: Intermediate</td><td>The agent is able to break down a complex goal into subgoals and pursue them in a fairly direct manner.</td><td>Goals often correspond to problems within the NP complexity class. The agent must explore a substantially expanded solution space with multiple potential pathways, frequently requiring heuristic approaches to navigate</td><td>Goals exhibit moderate feedback loop density with non-trivial interac- tions between subcom- ponents, creating more nuanced and less pre- dictable behavioural pat- terns that demonstrate in- cipient emergent proper-</td></tr><tr><td>4: High</td><td>The agent is able to break down a complex goal into many different subgoals, where success depends upon balancing and sequencing subgoals, which may themselves be challenging</td><td>Goals frequently correspond to NP complexity classes, with very high computational resources for both planning and execution phases.</td><td>Goals have dense net- works of interconnected feedback loops with sig- nificant cross-scale in- teractions and dependen- cies.</td></tr><tr><td>5: Unbounded</td><td>The agent can achieve all of the preceding steps. It can also generate its own goal structures in an unbounded way and interpret underspecified objectives.</td><td>Capabilities correspond to problems that reach beyond traditional complexity classifications, such as undecidable problems.</td><td>Capabilities exhibit emergence proper- ties across multiple scales with autopoietic characteristics—the goal system becomes self-modifying and self- generating rather than</td></tr></table>

Note. Adapted from Kasirzadeh and Gabriel, 2025.

## Resource-Based Metrics

The cognitive efficiency score (CES) measures the number of tokens and tool calls per successfully completed task. A low CES is imperative for resource management and environmental sustainability (AlShikh et al., 2025).

$$
C E S = { \frac { \mathrm { T o t a l ~ T o k e n s ~ G e n e r a t e d } + \mathrm { T o o l } / { \mathrm { A P I ~ C a l l s } } \times \mathrm { T o k e n ~ E q u i v a l e n t } } { \mathrm { N u m b e r ~ o f ~ S u c c e s s f u l l y ~ C o m p l e t e d } \mathrm { T a s k s } } }\tag{13}
$$

Collaboration quality index (CQI) measures an agent’s collaborative abilities across dimensions like communication clarity, responsiveness, and contextual awareness (AlShikh et al., 2025). Consider a set of three medical diagnoses that require clinician-agent collaboration. If the collaborative diagnostic sessions are scored 4, 4, and 5 (out of 5), then the $\begin{array} { r } { C \dot { Q } I = \frac { 4 + 4 + 5 } { 3 } = \overline { { 4 . 3 } } . } \end{array}$

$$
C Q I = { \frac { \sum \mathrm { I n t e r a c t i o n } \mathrm { Q u a l i t y } \mathrm { S c o r e s } } { \mathrm { T o t a l } \mathrm { C o l l a b o r a t i v e } \mathrm { T a s k s } } }\tag{14}
$$

Finally, as mentioned in Section 1.3, the Tool Dexterity Index (TDI) assesses an agent’s ability to select the optimal tool for each situation. The potential tool use scores are +1 for “optimal use,” -1 for “misuse,” and -0.5 for “ignored better tool” (AlShikh et al., 2025).

$$
T D I = { \frac { \Sigma { \mathrm { ~ T o o l ~ U s e ~ S c o r e s } } } { \mathrm { T o t a l ~ O p p o r t u n i t i e s ~ t o ~ U s e ~ T o o l s } } }\tag{15}
$$

## 3.4.4 Resilience and Adaptation

Resilience is the “ability of systems to withstand, adapt to, and recover from disruptive events” (Chacon-Chamorro, et al. 2025). The adaptation aspect of resilience refers to a reinforcement effect, wherein the AI system learns how to act, reconfigure, and prepare for future disruptions. To assess resilience, disruptions can be classified into two types of events (Pérolat, et al., 2017). The first type of disruptive event is characterized by the probability of occurrence and severity of a resource depletion. This event tests agents’ ability to sustain remaining resources, contingent upon disrupted environmental conditions. The second type of disruptive event is characterized by the lack of established policies or decision-making methods for the agent to follow. This event tests agents’ ability to generate new plans in order to complete their assigned tasks and adapt to external social influences.

Al Shikh et al. (2025) propose the multi-step task resilience, chain robustness score, and adaptability delta metrics to evaluate resilience and adaptation (see Table 27).

Table 27: Resilience and Adaptation Metrics
<table><tr><td>Metric</td><td>Description</td><td>Equation</td><td></td></tr><tr><td>Multi-step Task Resilience (MTR)</td><td>The percentage of multi-step tasks where the agent autonomously and successfully recovers from initial errors or ambiguities</td><td></td><td> $M T R = { \frac { \# \operatorname { o f } \operatorname { T a s k s } \operatorname { w i t h } \operatorname { S u c c e s s f u l } \operatorname { S e l f - R e c o v e r y } } { \operatorname { T o t a l } \operatorname { M u l t i - S t e p } \operatorname { T a s k s } } } \times 1 0 0 $ </td></tr><tr><td>Chain Robustness Score (CRS)</td><td>The percentage of times where the agent maintains logical consistency across multi-hop workflows</td><td></td><td> $C R S = { \frac { \sum { \mathrm { S u c c e s s f u l ~ C h a i n s ~ } } ( n \geq 3 { \mathrm { ~ s t e p s } } ) } { \mathrm { T o t a l ~ C h a i n s ~ w i t h ~ } } } \times 1 0 0 $ </td></tr><tr><td>Adaptability Delta (AD)</td><td>Quantifies the agent&#x27;s learning capacity and ability to adapt to new domains and schemas</td><td></td><td> $A D = P e r f o r m a n c e _ { \mathrm { f e w - s h o t } } - P e r f o r m a n c e _ { \mathrm { z e r o - s h o t } }$ </td></tr></table>

Note. Adapted from Al Shikh et al., 2025.

## 3.5 Temporal Coherence

Temporal coherence refers to an agent’s ability to maintain logical consistency through memory, context, and intent (McCants, 2026). For AI systems, McCants (2026) describes it as experienced “internally as a stable ‘present’—without relying on an objective clock, linear time, or a privileged now.” AI operates through discrete steps, and temporal coherence is the mechanism that enforces a dependency between the steps. That is, it ensures that the state at step t is a function of the state at step t − 1. Therefore, temporal coherence enables dynamic stability—the return to a stable state after disturbance—because it preserves a logical derivation from t to t − 1. Components of temporal coherence include coherence and hysteresis.

## 3.5.1 Coherence

Coherence is a structural requirement for an artificial system, made possible by an active processing window (APW) that defines the “relevant now.” The APW is a scope of informational relevance that dynamically changes to include new inputs and exclude old states. Its size is determined by the AI systems’ memory bandwidth and information density, rather than a fixed time duration (McCants, 2026).

The APW employs context anchoring to ensure that the AI system maintains consistent meanings across states. Context anchoring prioritizes state vectors (anchors) that resist change and give stability to specific concepts within the vector space. When the system encounters an input that may shift the meaning of a specific established concept, the anchor exerts a restoring force to pull the interpretation back to its established baseline. Indeed, the system is free to explore the context of a concept within the bounds set by its anchors, but not to “drift” beyond the anchors.

Coherence drift occurs when a system drifts beyond the bounds set by its anchors (i.e., the system’s internal logic diverges from its established intent). As the term “drift” implies, coherence drift is insidious, and its subtle progression makes it difficult to detect.

Table 28: Drift and Instability Metrics
<table><tr><td>Metric</td><td>Description and Equation</td></tr><tr><td>State Vector Representation</td><td>Let the system&#x27;s internal operational state at inference step ¿ be represented as a high- dimensional vector  $\mathbf { s } _ { i } \in \mathbb { R } ^ { \bar { n } }$  where  $\mathbf { s } _ { i }$  encodes the active processing window, including attention distributions, anchor weights, and active semantic representations. A fixed reference configuration (“Gold Standard&quot;) is defined as  $\mathbf { s } _ { 0 }$  representing the system&#x27;s initialized or constitutionally valid state.</td></tr><tr><td>Coherence Deviation</td><td>The deviation between the current state and the reference state.  $D _ { i } \ = \ \| s _ { i } - s _ { 0 } \| _ { 2 }$  where  $\lVert \cdot \rVert _ { 2 }$  denotes Euclidean distance. Alternatively, cosine divergence may be used:  $D _ { i } ^ { c o s } = 1 - \frac { s _ { i } \cdot s _ { 0 } } { \left\| s _ { i } \right\| \left\| s _ { 0 } \right\| }$ </td></tr><tr><td>Drift Velocity</td><td> $V _ { i } = \frac { D _ { i } - D _ { i - 1 } } { \Delta i }$  Used to distinguish benign adaptation from dangerous drift: . High drift velocity indicates rapid structural change and reduced hysteresis.</td></tr><tr><td>Drift Acceleration</td><td> $A _ { i } = { \frac { V _ { i } - V _ { i - 1 } } { \Delta i } }$  Persistent instability is detected via second-order change: . Positive acceleration over sustained intervals is a strong indicator of anchor failure and imminent temporal incoherence.</td></tr></table>

Note. Adapted from McCants, 2026.

## 3.5.2 Hysteresis

Hysteresis is a physics concept that describes the lag between an effect and its cause. McCants (2026) proposed monitoring AI systems’ hysteresis to detect coherence drift. A system without coherence drift has high hysteresis, because it resists change and requires significant force to move beyond its anchors’ bounds. A drifting system has low hysteresis, because its anchors have failed. Therefore, a reduction in hysteresis indicates that an AI system’s context anchoring is failing, and the system is entering coherent drift. The hysteresis index (H) is defined below, with low H implying hypersensitivity and anchor degradation. Let $s _ { i } ^ { ( x ) }$ be the state resulting from a standardized probe input x applied at step i (McCants, 2026):

$$
H = \mathbb { E } \left[ \| s _ { i } ^ { ( x ) } - s _ { i - k } ^ { ( x ) } \| _ { 2 } \right]\tag{16}
$$

## 4 Future Directions

In this section, we identify gaps in evaluations and propose recommended areas of research and a method to continually update the Agent Compendium.

## 4.1 Recommended Areas of Active Research

## Hybrid Agent Architecture

An agent’s architecture informs the agent’s degree of environmental interaction. As aforementioned (see Section 3.1.1), the three main types of agent architecture are deliberative, reactive, and hybrid. Each hybrid architecture has a fixed proportion of deliberative versus reactive architecture. During the design phase, agents are assigned computational resources to their deliberative and reactive layers, proportional to their architecture. As such, different hybrid architectures are optimal for different problem domains and environments (Klyubin et al., 2005). There is not an evaluation to determine the proportion of deliberative versus reactive characteristics in hybrid agent architectures, so we recommend the development of a hybrid architecture evaluation.

We propose that the newly developed evaluation be paired with Carrascosa et al.’s (2004) reactivity degree, which they define as an agent’s “maximum available slack percentage” (i.e., amount of excess time) to deliberate. A hybrid architecture evaluation, in tandem with the reactivity degree measurement, will promote the deployment of agents with hybrid architectures that are optimal for their environments and use cases.

## Navigation Metrics

Navigation benchmarks measure agents’ abilities to purposefully move through their environments. Apart from SPL, navigation benchmarks and metrics typically differ across models. For example, the audio-embodied question answering NoisyEQA model employs two noise-related evaluation metrics to measure the agent’s capacity to detect and correct different types of noise in noisy questions (Wu et al., 2024). Noise-related evaluation metrics are not applicable to models that do not process audio, such as Contrastive Language–Image Pre-training (CLIP) on Wheels (CoW). CoW introduced a new evaluation benchmark called PASTURE to measure agents’ language-driven zero-shot object navigation capabilities (Gadre et al., 2022). These capabilities are not present across all navigation models, so the PASTURE benchmark cannot be universally applied. We propose the development of model-agnostic navigation metrics to standardize the evaluation of agents’ navigation capabilities.

## Task Mastery

Task mastery refers to the complete understanding and knowledge of a particular domain. Generally, AI agent research lacks quantifiable metrics. For example, Kasirzadeh and Gabriel’s (2025) agent generality ontology (see Table 15) is based on the breadth of task mastery. However, they fail to define the criteria for task mastery: What does it mean for an agent to "master one specific task"? When assessing mastery, behavioral practitioners and researchers primarily consider the level of performance (measured as the percentage of accuracy) and the frequency of success at that level (Fuller and Fienup, 2018; Love et al., 2009). These criteria could be adapted to assess agents’ task mastery.

## Temporal Challenges

We recommend benchmarks that evaluate agents’ state awareness and memory systems. AI agents with baseline temporal coherence maintain immediate state awareness but limited historical awareness; agents with sophisticated temporal coherence exhibit advanced state management via hierarchical memory structures. State awareness and memory structure benchmarks would reveal agents’ capacity for temporal reasoning and help researchers identify corresponding temporal challenges (e.g., reasoning drift, temporal incoherence, forward-looking data gaps) (Shaikh, 2026).

## 5 The Agent Compendium

The Agent Compendium is a continually growing collection of criteria, metrics, benchmarks, and approaches for defining AI agents. It can be found at agent.duketrustlab.com. To ensure that the Agent Compendium remains current and useful, we will maintain it as a living resource that is updated as relevant methods are proposed, validated, and published. An AI-assisted literature review tool will support this process by monitoring new preprints and publications and flagging papers that may introduce or apply measures related to the defining characteristics of AI agents. These characteristics include environmental interaction, learning and adaptation, autonomy, goal-directed behavior, and temporal coherence. Each flagged source will be reviewed by the research team to assess its relevance, methodological quality, and relationship to the existing taxonomy. This combination of automated literature monitoring and expert review will allow the Agent Compendium to remain responsive to a fast-changing research landscape while preserving scholarly oversight, transparency, and consistency.

## 6 Conclusion

The term “agent” in artificial intelligence lacks a widely accepted definition. This conceptual ambiguity undermines the reproducibility of AI agent research, obscures scientific communication, and contributes to the conflation of “AI agents” with “agentic AI.” Scholars, policy makers, and governance advisors need a standardized, comprehensive set of AI agent evaluation methods in order to advance research, regulate the technology, and control its risks. Recent work aimed at standardizing the definition of AI agents has proposed a spectrum of agenticness based on the five components of what constitutes an AI agent: environmental interaction, learning and adaptation, autonomy, goal-directed behavior, and temporal coherence. We extend this prior work by compiling a compendium of evaluation metrics, benchmarks, and frameworks that correspond to these five components and map them to the definition of an AI agent. This compendium, the Agent Compendium, is a comprehensive, publicly-accessible information source for scholars, researchers, and the general public.

## References

AlShikh, W., Ali, M. S., Kennedy, B., & Mozolevskyi, D. Towards outcome-oriented, task-agnostic evaluation of ai agents, 2025.

Anderson, P., Chang, A., Chaplot, D.S., Dosovitskiy, A., Gupta, S., Koltun, V., Kosecka, J., Malik, J., Mottaghi, R., Savva, M. & and Zamir, A.R. On evaluation of embodied navigation agents, 2018.

Anderson, P., Wu, Q., Teney, D., Bruce, J., Johnson, M., Sünderhauf, N., Reid, I., Gould, S. & Van Den Hengel, A. Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3674–3683. IEEE Computer Society, 2018. doi:10.1109/CVPR.2018.00387.

Anthropic. Building effective agents, 2024. URL https://www.anthropic.com/engineering/ building-effective-agents.

Arike, R., Donoway, E., Bartsch, H., & Hobbhahn, M. Technical report: Evaluating goal drift in language model agents, 2025. URL https://arxiv.org/abs/2505.02709.

Azuma, D., Miyanishi, T., Kurita, S., & Kawanabe, M. ScanQA: 3D Question Answering for Spatial Scene Understanding. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), page 19107–19117, 2022. doi:10.1109/CVPR52688.2022.01854.

Bakos, L. Efficiency, Fairness, and Excellence: The Case for Centralized Research Infrastructure. Microscopy Today, 33(4):15–16, 2025. doi:10.1093/mictod/qaaf045.

Bent, B. The term ’agent’ has been diluted beyond utility and requires redefinition, 2025. URL https://arxiv.org/ abs/2508.05338.

Blundell, C., Uria, B., Pritzel, A., Li, Y., Ruderman, A., Leibo, J. Z., Rae, J., Wierstra, D., & Hassabis, D. Model-Free Episodic Control (Version 1), 2016. URL https://arxiv.org/abs/1606.04460.

Brooks, R. A robust layered control system for a mobile robot. IEEE Journal on Robotics and Automation, 2(1):14–23, 1986. doi:10.1109/JRA.1986.1087032.

Burmeister, B. and Sundermeyer, K. Cooperative problem-solving guided by intentions and perception. SIGOIS Bull., 13(3):10, 1992. doi:10.1145/152683.152690.

Carrascosa, C., Fabregat, J., Terrasa, A., & Botti, V. Real-time agents: Reaction vs. deliberation, 2004.

Chacon-Chamorro, M., Felipe Giraldo, L., Quijano, N., Vargas-Panesso, V., González, C., Pinzón, J. S., Manrique, R., Ríos, M., Fonseca, Y., Gómez-Barrera, D., & Perdomo-Pérez, M. Cooperative resilience in artificial intelligence multiagent systems. IEEE Transactions on Artificial Intelligence, 6(12):3430–3440, 2025. doi:10.1109/TAI.2025.3567371.

Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F., Sehwag, V., Dobriban, E., Flammarion, N., Pappas, G. J., Tramer, F., Hassani, H. & Wong, E. Jailbreakbench: an open robustness benchmark for jailbreaking large language models. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, 2024. ISBN 9798331314385.

Chen, C., Jain, U., Schissler, C., Gari, S. V. A., Al-Halah, Z., Ithapu, V. K., Robinson, P., & Grauman, K. Soundspaces: Audio-visual navigation. In Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VI, page 17–36, 2020. ISBN 978-3-030-58538-9. doi:10.1007/978-3-030-58539-6\_2.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. de O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., . . . Zaremba, W. Evaluating Large Language Models Trained on Code, 2021. URL https://arxiv.org/abs/2107.03374.

Chen, Z., Chen, S., Ning, Y., Zhang, Q., Wang, B., Yu, B., Li, Y., Liao, Z., Wei, C., Lu, Z., Dey, V., Xue, M., Baker, F. N., Burns, B., AduAmpratwum, D., Huang, X. Ning, X., Gao, S., Su, Y., & Sun, H. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery. International Conference on Learning Representations, 2025:96934–96990, 2025.

Chiappetta, A., & Mahari, R. Intentionality is a Design Decision: Measuring Functional Intentionality for Accountable AI Systems, 2026. URL https://arxiv.org/abs/2605.05475.

Das, A., Datta, S., Gkioxari, G., Lee, S., Parikh, D., & Batra, D. Embodied Question Answering, 2017. URL https://arxiv.org/abs/1711.11543.

Debenedetti, E., Zhang, J., Balunovic, M., Beurer-Kellner, L., Fischer, M., & Tramèr, F. Agentdojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, 2024. ISBN 9798331314385.

Desai, S. S., & Lee, S. Auxiliary Tasks for Efficient Learning of Point-Goal Navigation. In 2021 IEEE Winter Conference on Applications ofComputer Vision (WACV), pages 717–725, 2021. doi:10.1109/WACV48630.2021.00076.

Dorbala, V. S., Sigurdsson, G., Piramuthu, R., Thomason, J., & Sukhatme, G. S. CLIP-Nav: Using CLIP for Zero-Shot Vision-and-Language Navigation, 2022. URL https://arxiv.org/abs/2211.16649.

Dumke, R., Koeppe, R., & Wille, C. Software Agent Measurement and Self Measuring Agent Based Systems. Univ., Fak. für Informatik, 11, 2000.

El Mehdi, E. Adversarial Robustness in AI Agents: Security Challenges, Attack Taxonomies, and Defense Mechanisms., 2026.

Felicia, G., Bryant, N., Putra, H., Gazali, A., Lobo, E., & Rojas, E. From Perception to Action: Spatial AI Agents and World Models, 2026. URL https://arxiv.org/abs/2602.01644.

Feng, K. J. K., McDonald, D. W., & Zhang, A. X. Levels of Autonomy for AI Agents, 2025. URL https: //arxiv.org/abs/2506.12469.

Ferguson, I. A. TouringMachines: An architecture for dynamic, rational, mobile agents. Technical Report UCAM-CL-TR-273, University of Cambridge, Computer Laboratory, 1992.

Fuller, J. L. and Fienup, D. M. A Preliminary Analysis of Mastery Criterion Level: Effects on Response Maintenance. Behavior Analysis in Practice, 11:1–8, 2018.

Gadre, S. Y., Wortsman, M., Ilharco, G., Schmidt, L., & Song, S. CoWs on Pasture: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation, 2022. URL https://arxiv.org/abs/2203.10421.

Georgeff, M. P., & Lansky, A. L. Reactive reasoning and planning. In Proceedings ofthe Sixth National Conference on Artificial Intelligence - Volume 2, AAAI’87, page 677–682. AAAI Press, 1987. ISBN 0934613427.

Georgievski, I., & Aiello, M. An Overview of Hierarchical Task Network Planning, 2014. URL https://arxiv. org/abs/1403.7426.

Gubrud, M. Nanotechnology and international security. In Fifth Foresight Conference on Molecular Nanotechnology, 1997.

Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., . . . Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Nature, 645(8081):633–638, 2025. doi:10.1038/s41586-025-09422-z.

Hashi, A. I., & Jama, O. A. Formal verification and adversarial robustness framework for agentic ai in high-stake autonomous decision systems. International Journal ofEngineering and Computer Science, 15(5), 2026.

Hobbhahn, M. SWE-Bench Verified Mini, 2025, 2025. URL https://huggingface.co/datasets/ MariusHobbhahn/swe-bench-verified-mini.

Huang, C., Mees, O., Zeng, A., & Burgard, W. Visual language maps for robot navigation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), page 10608–10615, 2023. doi:10.1109/ICRA48891.2023.10160969.

Huang, G. Embodied intelligence. In Handbook of Human-Centered Artificial Intelligence, pages 341–367. Singapore: Springer Nature Singapore, 2026.

Jhi, K., & Park, G. S. Ai agent-based blockchain transaction sequencer for intent-aware transaction ordering. IEEE Access, 14:33336–33352, 2026. doi:10.1109/ ACCESS.2026.3666884.

Jiang, K., Liu, Y., Chen, W., Luo, J., Chen, Z., Pan, L., Li, G., & Lin, L. Beyond the destination: A novel benchmark for exploration-aware embodied question answering. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9091–9101, 2025. doi:10.1109/ICCV51701.2025.00850.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, volume 2024, pages 54107–54157, 2024.

Jovanovic, M., & Campbell, M. Self-Directing AI: The Road to Fully Autonomous AI Agents.´ Computer, 58(2):71–77, 2025. ISSN 0018-9162. doi:10.1109/MC.2024.3507513.

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl, S. A. A., Ballard, A. J., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J., . . . Hassabis, D. Highly accurate protein structure prediction with alphafold. Nature, 596(7873): 583–589, 2021. doi:10.1038/s41586-021-03819-2.

Kapoor, S., Stroebl, B., Kirgis, P., Nadgir, N., Siegel, Z. S., Wei, B., Xue, T., Chen, Z., Chen, F., Utpala, S., Ndzomga, F., Oruganty, D., Luskin, S., Liu, K., Yu, B., Arora, A., Hahm, D., Trivedi, H., Sun, H., Lee, J., Jin, T., Mai, Y., Zhou, Y., Zhu, Y., Bommasani, R., Kang, D., Song, D., Henderson, P., Su, Y., Liang, P., & Narayanan, A. Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation, 2025. URL https://arxiv.org/abs/2510.11977.

Kasirzadeh, A., & Gabriel, I. Characterizing AI Agents for Alignment and Governance, 2025. URL https://arxiv. org/abs/2504.21848.

Klyubin, A. S., Polani, D., & Nehaniv, C. L. Empowerment: A universal agent-centric measure of control. In 2005 IEEE Congress on Evolutionary Computation, volume 1, pages 128–135, 2005. doi:10.1109/CEC.2005.1554676.

Kociský, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann, K. M., Melis, G., & Grefenstette, E. The narrativeqaˇ reading comprehension challenge. Transactions ofthe Associationfor Computational Linguistics, 6:317–328, 2018. doi:10.1162/tacl\_a\_00023.

Krantz, J., Wijmans, E., Majumdar, A., Batra, D., & Lee, S. Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments, 2020.

Lee, Y., Lee, K., Park, S., Hwang, D., Kim, J., Lee, H. I., & Lee, M. Qasa: advanced question answering on scientific articles. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 19036–19052. PMLR, 2023. URL https://proceedings.mlr.press/ v202/lee23n.html.

Legg, S. Machine Super Intelligence. PhD thesis, Informatics of the University of Lugano, 2008.

Li, L., Wang, Y., Zhao, H., Kong, S., Teng, Y., Li, C., & Wang, Y. Reflection-Bench: Probing AI Intelligence with Reflection, 2024. URL https://arxiv.org/abs/2410.16270.

Li, X., Yin, X., Li, C., Zhang, P., Hu, X., Zhang, L., Wang, L., Hu, H., Dong, L., Wei, F., Choi, Y., & Gao, J. Oscar: Object-semantics aligned pre-training for vision-language tasks. In Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX, page 121–137, Berlin, Heidelberg, 2020. Springer-Verlag. ISBN 978-3-030-58576-1. doi:10.1007/978-3-030-58577-8\_8.

Lin, B., Long, Y., Zhu, Y., Zhu, F., Liang, X., Ye, Q., & Lin, L. Towards Deviation-Robust Agent Navigation via Perturbation-Aware Contrastive Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10): 12535–12549, 2023. doi:10.1109/TPAMI.2023.3273594.

Love, J.R., Carr, J.E., Almason, S. N. & Petursdottir, A. I. Early and intensive behavioral intervention for autism: A survey of clinical practices. Research in Autism Spectrum Disorders, 3(2):421–428, 2009. doi:10.1016/j.rasd.2008.08.008.

Majumdar, A., Aggarwal, G., Devnani, B., Hoffman, J., & Batra, D. Zson: zero-shot object-goal navigation using multimodal goal embeddings. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D., & Hendrycks, D. Harmbench: a standardized evaluation framework for automated red teaming and robust refusal. In Proceedings ofthe 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

McCants, S. A. P. Temporal Coherence in Artificial Intelligence: Maintaining a Stable Present Without Clocks or Central Time, 2026.

Mehta, H. AI Agentsfor Secure and Software-Defined Networking : Harnessing AI and SDN to Revolutionize Modern Work Environments. Apress, 1st edition, 2026. doi:10.1007/979-8-8688-2358-9.

Meshkov, A. Methodology to identify issues and improve the robustness of ai agents. International Journal of Management and Data Analytics (IJMADA), 6(1):1–28, 2026.

Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., & Scialom, T. GAIA: a benchmark for General AI Assistants, 2023. URL https://arxiv.org/abs/2311.12983.

Mohammadi, M., Li, Y., Lo, J., & Yip, W. Evaluation and benchmarking of llm agents: A survey. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, page 6129–6139, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400714542. doi:10.1145/3711896.3736570.

Morris, M. R., Sohl-Dickstein, J., Fiedel, N., Warkentin, T., Dafoe, A., Faust, A., Farbaret, C. & Legg, S. Levels of AGI for Operationalizing Progress on the Path to AGI, 2023. URL https://arxiv.org/abs/2311.02462.

Newell, A., & Simon, H. A. Computer science as empirical inquiry: symbols and search. Commun. ACM, 19(3): 113–126, 1976. ISSN 0001-0782. doi:10.1145/360018.360022.

OECD. The agentic ai landscape and its conceptual foundations. In OECD Artificial Intelligence Papers, number 56. OECD Publishing, 2026. URL https://doi.org/10.1787/396cf758-en.

OpenAI. A practical guide to building agents, 2026. URL https://openai.com/business guides-and-resources/a-practical-guide-to-building-ai-agents/.

Paglieri, D., Cupiał, B., Coward, S., Piterbarg, U., Wołczyk, M., Khan, A., Pignatelli, E., Kucinski, L., Pinto, L., Fergus,´ R., Foerster, J., Parker-Holder, J. & Rocktäschel, T. Balrog: Benchmarking Agentic LLM and VLM Reasoning on Games. In International Conference on Learning Representations, volume 2025, pages 96666–96702, 2025.

Pérolat, J., Leibo, J. Z., Zambaldi, V., Beattie, C., Tuyls, K., & Graepel, T. A multi-agent reinforcement learning model of common-pool resource appropriation. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 3646–3655, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

Plaat, A., Van Duijn, M., Van Stein, N., Preuss, M., Van Der Putten, P., & Batenburg, K. J. Agentic Large Language Models, a Survey. J. Artif. Int. Res., 84, 2025. ISSN 1076-9757. doi:10.1613/jair.1.18675.

Pritzel, A., Uria, B., Srinivasan, S., Badia, A. P., Vinyals, O., Hassabis, D., Wierstra, D., & Blundell, C. Neural Episodic Control. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 2827–2836. PMLR, 2017. URL https://proceedings.mlr.press/v70/ pritzel17a.html.

Qi, Y., Wu, Q., Anderson, P., Wang, X., Wang, W. Y., Shen, C., & Hengel, A. van den. REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9979–9988, 2020. doi:10.1109/CVPR42600.2020.01000.

Ramakrishnan, S. K., Chaplot, D. S., Al-Halah, Z., Malik, J., & Grauman, K. PONI: Potential Functions for ObjectGoal Navigation with Interaction-free Learning. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18868–18878, Los Alamitos, CA, USA, 2022. IEEE Computer Society. doi:10.1109/CVPR52688.2022.01832.

Rabanser, S., Kapoor, S., Kirgis, P., Liu, K., Utpala, S., & Narayanan, A. Towards a Science of AI Agent Reliability, 2026. URL https://arxiv.org/abs/2602.16666.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. Hierarchical text-conditional image generation with clip latents, 2022. URL https://arxiv.org/abs/2204.06125.

Rath, A. Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions, 2026. URL https://arxiv.org/abs/2601.04170.

Rawles, C., Li, A., Rodriguez, D., Riva, O., & Lillicrap, T. Android in the wild: a large-scale dataset for android device control. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

Remondino, M. Reactive and deliberative agents applied to simulation of socio-economical and biological systems. International Journal of Simulation, 6(12-13), 11-25. International Journal of Simulation: Systems, Science and Technology, 6(12-13):11–25, 2005.

Ritchie, L., Mehta, S., Heiner, N., Yu, M., & Chen, E. The Hierarchy of Agentic Capabilities: Evaluating Frontier Models on Realistic RL Environments, 2026. URL https://arxiv.org/abs/2601.09032.

Sacerdoti, E. D. Planning in a hierarchy of abstraction spaces. In Proceedings ofthe 3rd International Joint Conference on Artificial Intelligence, IJCAI’73, page 412–422, San Francisco, CA, USA, 1974. Morgan Kaufmann Publishers Inc.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., Ho, J., Fleet, D., & Norouzi, M. Photorealistic Text-to-Image Diffusion Models with Deep

Language Understanding. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Sánchez-Vaquerizo, J. A., & Monsivais, L. B. From Particles to Agents: Hallucination as a Metric for Cognitive Friction in Spatial Simulation, 2026. URL https://arxiv.org/abs/2601.21977.

Sapkota, R., Roumeliotis, K. I., & Karkee, M. AI Agents vs. Agentic AI: A Conceptual taxonomy, applications and challenges. Inf. Fusion, 126(PB), 2026. ISSN 1566-2535. doi:10.1016/j.inffus.2025.103599.

Savinov, N., Raichuk, A., Marinier, R., Vincent, D., Pollefeys, M., Lillicrap, T., & Gelly, S. Episodic curiosity through reachability, 2018. URL https://arxiv.org/abs/1810.02274.

Searle, J. R. Minds, brains, and programs. Behavioral and Brain Sciences, 3(3):417–424, 1980. doi:10.1017/S0140525X00005756.

Shanahan, M. The Technological Singularity. The MIT Press, 2015. ISBN 9780262527804. URL http://www.jstor. org/stable/j.ctt17kmw13.

Shao, Z., Luo, Y., Lu, C., Ren, Z. Z., Hu, J., Ye, T., Guo, Z., Ma, S., & Zhang, X. Deepseekmath-v2: Towards self-verifiable mathematical reasoning, 2025. URL https://arxiv.org/abs/2511.22570.

Shaikh, D. N. The Temporal Coherence Problem: Synthetic Point-in-Time Environments for Evaluating LLM Agents with Dynamic Tool Dependencies. International Journal of Computer Applications, 187:52–57, 2026. doi:10.5120/ijca7fbcf52ef814.

Shi, Q., Tang, M., Narasimhan, K., & Yao, S. Can language models solve olympiad programming?, 2024. URL https://arxiv.org/abs/2404.10952.

Shi, T., Karpathy, A., Fan, L., Hernandez, J., & Liang, P. World of bits: An open-domain platform for web-based agents. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3135–3144. PMLR, 2017. URL https://proceedings.mlr.press/v70/ shi17a.html.

Shome, P., Krishnan, S., & Das, S. Why Johnny Can’t Use Agents: Industry Aspirations vs. User Realities with AI Agents. In Proceedings ofthe ACM Conference on AI and Agentic Systems, CAIS ’26, page 597–619, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400724152. doi:10.1145/3786335.3813140.

Shukla, M. Evaluating Agentic AI Systems: A Balanced Framework for Performance, Robustness, Safety and Beyond, 2025. URL https://dx.doi.org/10.2139/ssrn.5402054.

Siegel, Z. S., Kapoor, S., Nagdir, N., Stroebl, B., & Narayanan, A. Core-bench: Fostering the credibility of published research through a computational reproducibility agent benchmark, 2024. URL https://arxiv.org/abs/2409. 11363.

Society of Automotive Engineers. Taxonomy and definitions for terms related to driving automation systems for on-road motor vehicles. Technical Report J3016\_202104, Society of Automotive Engineers International, 2021.

Song, K., Jayarajan, A., Ding, Y., Su, Q., Zhu, Z., Liu, S., & Pekhimenko, G. Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures in LLM Agents, 2025. URL https://arxiv.org/abs/2508.19504.

M Suleyman. The coming wave: technology, power, and the twenty-first century’s greatest dilemma. Crown Publishing Group (NY), 2023. ISBN 9780593593950.

Sun, F., Chen, R., Ji, T., Luo, Y., Zhou, H., & Liu, H. A comprehensive survey on embodied intelligence: Advancements, challenges, and future perspectives. CAAI Artificial Intelligence Research, 3(9150042), 2024. doi:10.26599/AIR.2024.9150042.

Tian, M., Gao, L., Zhang, S. D., Chen, X., Fan, C., Guo, X., Haas, R., Ji, P., Krongchon, K., Li, Y., Liu, S., Luo, D., Ma, Y., Tong, H., Trinh, K., Tian, C., Wang, Z., Wu, B., Xiong, Y., Yin, S., Zhu, M., Lieret, K., Lu, Y., Liu, G., Du, Y., Tao, T., Press, O., Callan, J., Huerta, E., & Peng, H. SciCode: A research coding benchmark curated by scientists. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385.

Turing, A. M. Computing machinery and intelligence. In Parsing the Turing test. Philosophical and methodological issues in the quest for the thinking computer, pages 23–65. Springer Publishing Company, Incorporated, 1st edition, 2007.

Uddin, K., Farooq, M. U., Khan, A., Saeed, M. S., Haq, I. U., Tasnim, N., & Malik, K. M. AdvBench: A Comprehensive Benchmark of Adversarial Attacks on Deepfake Detectors in Real-World Consumer Applications. TechRxiv, 2025 (1110), 2025. doi:10.36227/techrxiv.176281226.62299206/v1.

Uesato, J., Kumar, A., Szepesvari, C., Erez, T., Ruderman, A., Anderson, K., Krishmamurthy, Dvijotham, Heess, N., & Kohli, P. Rigorous Agent Evaluation: An Adversarial Approach to Uncover Catastrophic Failures, 2018. URL https://arxiv.org/abs/1812.01647.

Valmeekam, K., Marquez, M., Olmo, A., Sreedharan, S., & Kambhampati, S. PlanBench: an extensible benchmark for evaluating large language models on planning and reasoning about change. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

Varadi, M., Anyango, S., Deshpande, M., Nair, S., Natassia, C., Yordanova, G., Yuan, D., Stroe, O., Wood, G., Laydon, A., Žídek, A., Green, T., Tunyasuvunakool, K., Petersen, S., Jumper, J., Clancy, E., Green, R., Vora, A., Lutfi, M., . . . Velankar, S. AlphaFold Protein Structure Database: Massively expanding the structural coverage of protein-sequence space with high-accuracy models. Nucleic Acids Research, 50(D1):D439–D444, 2022. ISSN 0305-1048. doi:10.1093/nar/gkab1061.

Vyatkin, A., Poptsov, A., Oliseenko, V., Veiber, E., Shutko, O., Rafikov, T., Abramov, M., & Nikolenko, S. MultiCAT-Bench: A Multi-Categorical Agent Tool Use Benchmark. IEEE Access, 14:116038–116061, 2026. doi:10.1109/ACCESS.2026.3710376.

Wan, S., Nikolaidis, C., Song, D., Molnar, D., Crnkovich, J., Grace, J., Bhatt, M., Chennabasappa, S., Whitman, S., Ding, S., Ionescu, V., Li, Y., & Saxe, J. CYBERSECEVAL 3: Advancing the Evaluation of Cybersecurity Risks and Capabilities in Large Language Models, 2024.

Wang, N., Hu, X., Liu, P., Zhu, H., Hou, Y., Huang, H., Zhang, S., Yang, J., Liu, J., Zhang, G., Zhang, C., Wang, J., Jiang, Y. E., & Zhou, W. Efficient Agents: Building Effective Agents While Reducing Cost, 2025. URL https://arxiv.org/abs/2508.02694.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V. & Zhou, D. Chain-of-thought prompting elicits reasoning in large language models. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Wijmans, E., Kadian, A., Morcos, A., Lee, S., Essa, I., Parikh, D., Savva, M., & Batra, D. DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames, 2020. URL https://arxiv.org/abs/1911.00357.

Wooldridge, M. The Logical Modelling of Computational Multi-Agent Systems. PhD thesis, Department of Computation, UMIST, Manchester, UK, 1992.

Wooldridge, M. and Jennings, N. R. Intelligent agents: theory and practice. The Knowledge Engineering Review, 10(2): 115–152, 1995. doi:10.1017/S0269888900008122.

Wozniak, S. Could a Computer Make a Cup of Coffee. Fast Company interview, 2010. URL https://www.youtube. com/watch.

Wu, T., Zhou, C., Wong, Y. H., Gu, L., & Yang, J. NoisyEQA: Benchmarking Embodied Question Answering Against Noisy Queries, 2024. URL https://arxiv.org/abs/2412.10726.

Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shib, D., Lei, F., Liu, Y., Xu, Y., Zhou, S., Savarese, S., Xiong, C., Zhong, V. & Yu, T. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. URL https://arxiv.org/abs/2404.07972.

Xu, B. AI Agent Systems: Architectures, Applications, and Evaluation, 2026. URL https://arxiv.org/abs/2601. 01743.

Xue, T., Qi, W., Shi, T., Song, C. H., Gou, B., Song, D., Sun, H., & Su, Y. An illusion of progress? assessing the current state of web agents, 2025. URL https://arxiv.org/abs/2504.01382.

y Arcas, B.A. Artificial general intelligence is already here. Noema, 2023.

Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. τ-bench: A benchmark for tool-agent-user interaction in real-world domains, 2024. URL https://arxiv.org/abs/2406.12045.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., & Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

Yin, G., Bai, H., Ma, S., Nan, F., Sun, Y., Xu, Z., Ma, S., Lu, J., Kong, X., Zhang, A., Yap, D. A., Zhang, Y., Ahnert, K., Kamath, V., Berglund, M., Walsh, D., Gindele, T., Wiest, J., Lai, Z., . . . Wang, Z. MMAU: A Holistic Benchmark of Agent Capabilities Across Diverse Domains. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 4752–4780, Albuquerque, New Mexico, 2025. Association for Computational Linguistics. ISBN 979-8-89176-195-7. doi:10.18653/v1/2025.findings-naacl.267.

Yoran, O., Amouyal, S. J., Malaviya, C., Bogin, B., Press, O., & Berant, J. AssistantBench: Can Web Agents Solve Realistic and Time-Consuming Tasks? In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 8938– 8968, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi:10.18653/v1/2024.emnlp-main.505.

Yuan, F., Huang, X., Wang, L., Ding, J., Tian, Z., Wang, Y., Gu, S., Funabora, Y., Peng, Y. & Mao, Z. Towards General Embodied Intelligence: Integrating Large Language Models, Knowledge Bases, and Reasoning Capabilities to Build the Next Generation of AI Agents. International Journal of Hydromechatronics, 9(2):250–316, 2026. doi:10.1504/IJHM.2025.10074893.

Zhang, H., Deng, H., Ou, J., & Feng, C. Mitigating spatial hallucination in large language models for path planning via prompt engineering. Scientific Reports, 15(1):8881, 2025. doi:10.1038/s41598-025-93601-5.

Zheng, C., Liu, Z., Xie, E., Li, Z., & Li, Y. Progressive-hint prompting improves reasoning in large language models, 2023. URL https://arxiv.org/abs/2304.09797.

Zhu, J., Zhu, M., Rui, R., Shan, R., Zheng, C., Chen, B., Xi, Y., Lin, J., Liu, W., Tang, R., Yu, Y., & Zhang, W. Evolutionary Perspectives on the Evaluation of LLM-Based AI Agents: A Comprehensive Survey, 2025. URL https://arxiv.org/abs/2506.11102.