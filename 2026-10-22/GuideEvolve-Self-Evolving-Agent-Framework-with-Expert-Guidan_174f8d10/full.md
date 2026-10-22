# GuideEvolve: Self-Evolving Agent Framework with Expert Guidance and 3Ts Mechanisms

Meihong Jia   
Amazon   
Tokyo, Japan   
meihongj@amazon.co.jp   
Yasemin Timar   
Amazon Web Services   
Luxembourg, Luxembourg   
yastimar@amazon.com   
Jonathan Taws   
Amazon Web Services   
Luxembourg, Luxembourg   
tawsj@amazon.com

## Abstract

This study introduces GuideEvolve, a self-evolving agent framework incorporating expert guidance and three autonomous regulatory mechanisms—collectively termed the 3Ts: Think Fast and Slow, Throwback, and Termination by choice. The framework addresses critical limitations in extant self-evolving agent paradigms through dual innovations: (1) a task-specific expert-evaluation-driven evolutionary pipeline, and (2) the integration of 3Ts mechanisms to facilitate cognitive optimization, performance recovery via temporal rollback, and adaptive termination protocols.

Empirical validation across the MGSM and MMLU benchmarks demonstrates that GuideEvolve attains performance metrics comparable to current state-of-the-art models (89.8% accuracy versus Gödel Agent’s 90.6% on MGSM; 89.1% versus 87.9% on MMLU), while exhibiting notably reduced variance (25–59%) and enhanced robustness. Furthermore, the framework achieves a 100% evolu tion success rate—superior to the baseline’s 82%—and diminishes required optimization iterations by approximately 50% (from 30 iterations in baseline to 15.1 and 13.4 across tasks), and provides improved stability for production deployment.

## CCS Concepts

• Computing methodologies → Planning with abstraction and generalization; Temporal reasoning.

## Keywords

agent optimization, self-evolving, reasoning, evaluation

## 1 Introduction

Self-evolving agents represent a significant advancement in autonomous AI systems, ofering the potential for continuous selfimprovement through iterative learning and adaptation. While the Gödel Agent framework [16] demonstrated promising results using OpenAI models, its practical implementation faces substantial challenges. This paper addresses these challenges through the lens of Amazon Bedrock and Claude models integration, proposing crucial innovations that enhance the robustness and reliability.

The original implementation using OpenAI models, while groundbreaking, exhibited significant stability issues with 4% of times experiencing unexpected termination during evolution process and 14% optimization failure, severely limiting its practical util ity in production environments. When replicating the Gödel Agent in Bedrock using Claude models, we identified that the original framework’s evaluation feedback mechanism, which passes task questions and solver responses without optimization, frequently leads to immediate token exhaustion during the evolution process. These challenges are further compounded by Bedrock’s response formatting requirements, necessitating fundamental architectural adjustments for reliable operation.

To address these challenges, we introduce GuideEvolve agent, featuring two key innovations. First, we introduce a novel guided self-evolution framework, where a specialized LLM-based task expert provides analytical feedback to facilitate the agent’s autonomous improvement. The expert module generates task-specific evaluations and recommendations, which are then processed by the self-evolving agent to formulate optimization strategies for the solver. This hierarchical architecture maintains distinct contextual boundaries between evaluation logic and evolution control, enabling eficient token utilization while ensuring focused improvement pathways for task solutions. Notably, rather than implementing direct multi-agent interactions, this framework employs a unidirectional guidance mechanism where the expert’s analytical insights are systematically integrated into the agent’s self-evolving process, leading to iterative refinements in the solver’s operational logic. Second, we develop the 3Ts self-governing mechanisms for the self-evolving agent - Think Fast and Slow, Throwback, and Termination by choice - enabling agents to optimize their cognitive processing, retrieving previous best-performed logic for either recovery from deterioration or facilitate the brainstorm for further improvement on the solutions, and intelligently determine evolution endpoints.

Our framework is designed with practical applications in mind, particularly targeting enterprise use cases such as service improvement recommendations in AWS through Service360 product, and retail business strategy recommendations. By addressing the critical issues of convergence and stability, GuideEvolve agent represents a significant step toward making self-evolving agents viable for practical uses. Through comprehensive resource management and robust error handling, our framework transforms the challenges of Bedrock integration into opportunities for enhanced reliability and performance.

## 2 Background

Self-evolving agents represent an emerging paradigm in artificial intelligence that combines recursive self-improvement with large language models (LLMs). These systems aim to autonomously enhance their problem-solving capabilities through iterative refinement of their solution strategies. The theoretical foundations of self-evolving agents draw from several key areas in AI research. Constitutional AI frameworks [1] provide principles for maintaining reliable and controlled behavior in autonomous systems. Recent advances in AI planning and reasoning, such as Plan-and-Solve [11] and ReAct [15], ofer insights into structuring decision-making processes. Additionally, developments in memory management and state tracking, exemplified by GraphRAG ([3]) and IRCoT [10], provide crucial mechanisms for maintaining coherent evolution paths.

In the realm ofself-evolving agent architectures, Gödel Agent [16] stands out as a pioneering framework for autonomous self-modification. Its self-referential architecture enables recursive code-level evolution, marking a significant advance in the field of architecturally mutable agent systems. Several related research eforts provide context for Gödel Agent’s approach. The Darwin Gödel Machine [17] ofers a similar paradigm for open-ended evolution, using version control to enable non-linear evolutionary paths. AgentSquare [9] takes a more constrained approach, optimizing modular components within a predefined design space. AlphaEvolve [8] applies evolutionary principles to specific algorithmic domains, while ADAS [4] and AFlow [18] focus on higher-level agent system design through formal optimization methods.

Among these, Gödel Agent and the Darwin Gödel Machine represent the most comprehensive implementations of recursive selfimprovement, which allow for fundamental structural modifications and demonstrate the field’s progress towards increasingly adaptable systems. However, current approaches often lack eficient evaluation mechanisms, leading to potential degradation in solution quality over iterations. The Gödel Agent framework, for example, reported stability issues with 4% of times experiencing unexpected termination and 14% optimization failure, highlighting the need for more reliable evolution control mechanisms. These challenges are particularly acute in enterprise environments, where system stability and consistent performance are paramount.

## 3 GuideEvolve Framework

The GuideEvolve framework introduces a self-evolution paradigm that strategically separates task expertise from evolution control, as shown in Figure 1. At its core, a self-evolving agent orchestrates the evolution process by coordinating between two specialized LLMbased modules: a solver module focused on task execution and an expert module dedicated to performance evaluation. This architectural design ofers two significant advantages. First, it enables unbiased evaluation through an independent expert system that focuses solely on task-specific assessment without knowledge of the solver’s implementation details, while allowing the self-evolving agent to leverage its comprehensive visibility of the environment, historical performance, and solver logic to make informed evolution decisions. Second, this separation addresses the critical challenge of context management in evolution control: by delegating taskspecific evaluation to a specialized expert module, the self-evolving agent can more eficiently process and integrate guidance while maintaining essential contextual information about the environment, goals, and evolution strategies. This design fundamentally addresses limitations in existing self-evolving frameworks, where the lack of specialized, task-specific guidance and context overflow often constrain evolution efectiveness.

Expert Evaluation Module. The expert module functions as a specialized analytical consultant, maintaining focused task-specific context derived from dataset descriptions and required capabilities. This module’s isolation from the solver’s implementation details and evolution state ensures purely objective assessment based on performance outcomes. The expert system processes strategically sampled success and failure cases, analyzing task questions and LLM responses to generate comprehensive performance insights. This stateless architecture serves dual purposes: it maintains evaluation objectivity through isolation from evolution mechanics while significantly reducing the context burden that would otherwise impact the self-evolving agent’s eficiency. By processing raw tasksolver interactions independently, the expert system can generate focused, high-quality feedback without the overhead of evolutionrelated contextual information.

Guidance Integration and Evolution Mechanism. The frame work implements a structured guidance pipeline that optimizes information flow between its core components. The expert evaluation module generates three critical forms of guidance: quantitative performance metrics, detailed strength/weakness analysis, and actionable insights derived from case studies. The self-evolving agent, as the central control unit, processes this concentrated feedback while maintaining its broader contextual awareness of the environment, historical evolution patterns, and solver behavior. This hierarchical architecture enables eficient information processing: while the evaluation module focuses exclusively on task-specific assessment and recommendations, the self-evolving agent synthesizes these insights with its comprehensive contextual knowledge to formulate targeted evolution strategies. The agent then translates this synthesized understanding into concrete modifications of the solver’s operational logic. This modular design not only ensures more focused and efective guidance but also allows the self-evolving agent to manage its context budget more eficiently, dedicating computational resources to critical evolution decisions rather than raw task evaluation processing. The clear separation of evaluation and evolution control enables the agent to maintain a comprehensive view of the evolution trajectory while receiving specialized task-specific guidance.

## 4 Understanding Guided Evolution Through an Optimization Lens

Our intuition for why expert guidance improves both eficiency and robustness is based on an analogy to optimization theory. While the self-evolution process is not literally gradient descent, viewing it through this lens helps explain our empirical observations and suggests which architectural choices matter most.

We model the evolution as optimizing solver logic � to maximize expected performance �(�) on the task distribution:

$$
x ^ { * } = \arg \operatorname* { m a x } _ { x } \mathbb { E } [ f ( x ) ]\tag{1}
$$

Key Insight. The baseline Gödel Agent approach [16] resembles zeroth-order optimization, where the agent must infer improvement directions from noisy function evaluations alone, without explicit gradient information. In contrast, GuideEvolve resembles first-order optimization, where the expert module acts as an oracle providing gradient-like directional guidance. This analogy predicts the ob served eficiency gains and enhanced stability of our approach.

![](images/f4da2c63c456dd94c1f8614a279ad62d3b322781d7bb3cf226378e1eb7003dba.jpg)  
Figure 1: Demonstration of GuideEvolve Agent.

## 4.1 Baseline Evolution as Zeroth-Order Search

Following the Gödel Agent framework [16], the self-evolving agent must simultaneously evaluate performance and determine update directions. At each iteration �, the agent’s observation of performance is corrupted by context pollution:

$$
\tilde { f } ( x _ { t } ) = f ( x _ { t } ) + \epsilon _ { \mathrm { e v a l } } + \epsilon _ { \mathrm { c t x } }\tag{2}
$$

where $\epsilon _ { \mathrm { e v a l } }$ represents stochastic sampling noise from evaluating on limited training examples, and $\epsilon _ { \mathrm { c t x } }$ represents systematic observation bias. Concretely, �<sub>ctx</sub> corresponds to the challenge of extracting actionable insights when the agent must process raw task-solver interactions (e.g., question-answer pairs with their full context) mixed with evolution history within a shared context. The agent then computes updates:

$$
\boldsymbol { x } _ { t + 1 } = \boldsymbol { x } _ { t } + \boldsymbol { \alpha } \cdot [ \operatorname { d i r e c t i o n } ( \tilde { f } ( \boldsymbol { x } _ { t } ) , \operatorname { h i s t o r y } _ { t } ) + \delta _ { \mathrm { p r o c } } ^ { ( \mathrm { h i g h } ) } ]\tag{3}
$$

where $\delta _ { \mathrm { p r o c } } ^ { \mathrm { ( h i g h ) } }$ represents processing bias from context overflow when maintaining long evolution histories containing raw evaluation data alongside decision-making logic. Just as noisy gradient estimates harm optimization convergence, the compounding ef fect of polluted observations $( \epsilon _ { \mathrm { c t x } } )$ and bloated processing context $( \delta _ { \mathrm { p r o c } } ^ { \mathrm { ( h i g h ) } } )$ leads to inconsistent evolution trajectories across trials and potential convergence to suboptimal solutions.

## 4.2 Guided Evolution as First-Order Approximation

Our framework decomposes the optimization into a two-level structure. The expert module acts as an oracle that processes raw performance data in isolation:

$$
\begin{array} { r l } & { \tilde { f } _ { \mathrm { c l e a n } } ( x _ { t } ) = f ( x _ { t } ) + \epsilon _ { \mathrm { e v a l } } } \\ & { ~ h ( x _ { t } ) = \mathrm { e x p e r t } ( \tilde { f } _ { \mathrm { c l e a n } } ( x _ { t } ) , x _ { t } ) } \end{array}\tag{4}
$$

The expert processes only task-specific information without access to evolution mechanics, producing concentrated guidance $h ( x _ { t } )$ analogous to curated gradient information. In practice, $h ( x _ { t } )$ is a structured report containing strengths/weaknesses analysis and actionable recommendations (see Appendix E for an example). This resembles replacing noisy finite-diference gradients with analytical gradients in optimization. The self-evolving agent then performs guided updates:

$$
x _ { t + 1 } = x _ { t } + \alpha \cdot [ \mathrm { d i r e c t i o n } ( h ( x _ { t } ) , \mathrm { h i s t o r y } _ { t } ) + \delta _ { \mathrm { p r o c } } ^ { \mathrm { ( l o w ) } } ]\tag{5}
$$

Critically, GuideEvolve eliminates context bias at two levels: (1) observation level—the agent receives clean guidance $h ( x _ { t } )$ rather than polluted observations $\tilde { f } ( x _ { t } )$ (eliminating $\epsilon _ { \mathrm { c t x } } )$ , and (2) processing level—the agent’s history no longer contains raw evaluation data, substantially reducing context load such that $\delta _ { \mathrm { p r o c } } ^ { \mathrm { ( l o w ) } } \ll \delta _ { \mathrm { p r o c } } ^ { \mathrm { ( h i g h ) } }$

## 4.3 Predicted Benefits from the Analogy

This optimization perspective generates three testable predictions: Improved Accuracy through Cleaner Signals. The expert’s context isolation eliminates systematic observation bias, providing cleaner directional signals. For knowledge-intensive tasks (MMLU), this enables discovery of better local optima through more accurate gradient approximations, resulting in superior convergence:

$$
\mathbb { E } [ f ( x _ { \mathrm { G u i d e E v o l v e } } ^ { * } ) ] \geq \mathbb { E } [ f ( x _ { \mathrm { b a s e l i n e } } ^ { * } ) ]\tag{6}
$$

For tasks with fundamental capability constraints (e.g., MGSM requiring symbolic computation), both approaches converge to similar accuracy plateaus determined by underlying model capabilities.

Enhanced Stability through Variance Reduction. The dual level elimination of context bias should fundamentally improve trajectory consistency across independent trials. We can conceptualize the variance in final performance as composed of multiple sources:

$$
\begin{array} { r l } { \mathrm { V a r } [ f ( x ^ { * } ) ] \approx } & { \underbrace { \sigma _ { \mathrm { e v a l } } ^ { 2 } } _ { \mathrm { f r o m } { \epsilon _ { \mathrm { e v a l } } } } + \underbrace { \sigma _ { \mathrm { c t x } } ^ { 2 } } _ { \mathrm { f r o m } { \epsilon _ { \mathrm { c t x } } } } } \\ & { + \underbrace { \sigma _ { \mathrm { p r o c } } ^ { 2 } } _ { \mathrm { f r o m } { \delta _ { \mathrm { p r o c } } } } + \underbrace { \sigma _ { \mathrm { p a t h } } ^ { 2 } } _ { \mathrm { t r a j e c t o r y r a n d o m n e s s } } } \end{array}\tag{7}
$$

Under this conceptual model, GuideEvolve should reduce three of the four terms: (a) $\sigma _ { \mathrm { c t x } } ^ { 2 }$ eliminated through expert’s clean observation (removing �<sub>ctx</sub>), $( \mathrm { b } ) \sigma _ { \mathrm { p r o c } } ^ { 2 }$ substantially reduced via lean evolution history without raw evaluation data (reducing $\delta _ { \mathrm { p r o c } } ^ { \mathrm { ( h i g h ) } }$ to $\delta _ { \mathrm { p r o c } } ^ { ( \mathrm { l o w } ) } )$ ), and $\left( \mathrm { c } \right) \sigma _ { \mathrm { p a t h } } ^ { 2 }$ reduced via consistent, unbiased guidance signals. Note that $\sigma _ { \mathrm { e v a l } } ^ { 2 }$ should remain unchanged as both approaches use identical sampling strategies.

Faster Convergence through Informed Search. The first order-like guidance $h ( x _ { t } )$ should enable more eficient exploration compared to zeroth-order search with polluted observations and bloated context:

$$
\mathrm { I t e r . _ { G u i d e E v o l v e } \ll I t e r . _ { b a s e l i n e } }\tag{8}
$$

## 4.4 Empirical Support and Scope of the Analogy lalogy

Our experimental results (Section 7) are consistent with these predictions. The accuracy patterns across tasks match the expectation that cleaner signals benefit knowledge-intensive tasks (MMLU) more than capability-constrained tasks (MGSM). The observed 25- 59% variance reduction (Table 2) is consistent with the predicted elimination of $\sigma _ { \mathrm { c t x } } ^ { 2 }$ and reduction of $\sigma _ { \mathrm { p r o c } } ^ { 2 }$ and $\sigma _ { \mathrm { { p a t h } } } ^ { 2 } .$ , while $\sigma _ { \mathrm { e v a l } } ^ { 2 }$ remains present as both approaches use identical sampling strategies. The 50% reduction in required iterations (Table 3) supports the informed search prediction. This optimization lens—while not a formal proof—provides valuable intuition for why specific architectural choices (expert isolation, lean context management, and clean information flow) translate into the observed stability and eficiency improvements.

## 5 Self-Governing Evolution Mechanisms (3Ts)

Our framework introduces three critical self-governing mechanisms - Think fast and slow, Throwback, and Termination by Choice (3Ts) - that enable autonomous and eficient evolution control.

Think, Fast and Slow. Our framework implements a dual-mode cognitive processing paradigm through prompt engineering. "Fast thinking" enables eficient execution of multiple low-risk actions in sequence, while "slow thinking" enforces careful step-by-step planning for resource-intensive or error-prone operations. This balance optimizes resource utilization while maintaining solution quality through dynamic planning strategies. Notably, performance evaluation through the expert module is only initiated after confirmed successful modifications to the solver’s logic, representing a key "slow thinking" checkpoint in the evolution process. The details can be found in goal prompt in Appendix A.

Throwback. The throwback mechanism represents a crucial innovation in evolution control, extending beyond simple temporal rollback functionality. The system maintains two distinct tracking mechanisms: comprehensive performance monitoring across the entire evolution history, and state preservation of the solver logic at its previous best-performing state. While the performance tracking provides a complete evolutionary trajectory, the selective state preservation minimizes storage overhead while maintaining the capability for both defensive rollback and proactive learning of the self-evolving agent. This design choice reflects our focus on eficient memory utilization while ensuring the agent can either revert to proven optimal states or analyze previous success factors for continued improvement.

We observe three distinct patterns in the agent’s autonomous throwback decisions: 1) Strategic state examination, where the agent revisits previous high-performing states to identify untapped optimization opportunities; 2) Defensive rollback in response to significant performance deterioration, restoring the solver to a known optimal state; 3) Context-driven recall during progressive message reduction, where the agent strategically retrieves and analyzes historical solver logic to maintain evolution coherence. An example of throwback is illustrated in Figure 2.

This mechanism difers fundamentally from traditional treebased approaches by enabling flexible state examination without enforcing strict hierarchical relationships between states. The agent can autonomously decide when and how to utilize historical information, whether for performance recovery or optimization insight.

Termination By Choice. The framework implements a flexible termination mechanism that balances autonomous decision-making with performance guarantees, as illustrated in Appendix B. Our observations reveal emerging behavioral patterns in self-evolving agents: some "lazy" agents that attempt early termination after minimal trials, and some "diligent" agents that persist well beyond suggested termination conditions. This demonstrates the full autonomy of termination by agent’s choice. To prevent premature termination, especially from "lazy" agents, the system implements validation checks that provide feedback such as "you need to have at least five trials, keep going" to enforce further exploration of the solution space.

## 6 Resource Management System

The integration in Bedrock using Claude models necessitates careful management of token limitations and response completeness. Our framework implements a comprehensive resource management system that transforms these constraints into opportunities for optimization.

Token Budget Optimization. Our system employs a dynamic token allocation strategy that prioritizes essential operations while

![](images/7692dfa5bc2251ee780b6a2f5c2b4299fafca8bd5a1fd5566f890dca97e6d229.jpg)  
Iterations

Figure 2: An illustration of implementation of GuideEvolve that calls the throwback action.

maximizing eficiency. The framework reserves token budget primarily for tool calls — the critical mechanism for action execution — while maintaining system messages as immutable components. A dynamic sliding window manages action history, automatically adjusting output token limits based on input context size. This preventative approach optimizes token utilization while ensuring tool call generation under varying context loads.

Response Validation and Context Management. The framework implements a two-tier approach to handle potential token overflow scenarios. First, it leverages Bedrock’s structured response format to implement robust completion detection through JSON validation and tool call verification. When token limitations lead to response truncation, the system activates a progressive context reduction mechanism that systematically preserves essential infor mation while reducing context size. This context reduction process operates sequentially: 1) Removal of older action history messages while maintaining recent; 2) Progressive increase in reduction scope if truncation in response persists; 3) Maintenance of minimal viable context with at least one recent action message.

This integrated approach, coupled with the framework’s throwback capability, ensures operational continuity while minimizing information loss. The system efectively balances immediate operational needs with long-term evolution stability, transforming platform constraints into mechanisms for enhanced reliability.

## 7 Experiments Results

We evaluate GuideEvolve on two benchmark datasets: Mathematics Grade School Multi-source (MGSM) and Massive Multitask Language Understanding (MMLU). MGSM tests multi-step mathe matical reasoning and cross-lingual problem-solving capabilities, while MMLU evaluates comprehensive knowledge and problemsolving abilities across 57 diverse academic and professional domains through multiple-choice questions.

Experimental Setup. We conduct our experiments using An thropic’s Claude 3.5 Sonnet v2 as the default initial model for

GuideEvolve’s components (self-evolving agent, solver, and expert module). The self-evolving agent can autonomously decide to switch models among Anthropic Claude 3.5 Haiku, Anthropic Claude 3.5 Sonnet v2 (implementation identifier: SONNET\_V36), and Anthropic Claude 3.7 Sonnet (SONNET\_V37) during the evolution process. Note that the model selected plays a critical role in performance, as also reported in the original Gödel agent paper ([16]).

Following Gödel’s experimental protocol for fair comparison, we evaluate on 800 test samples and perform 6 independent trials for each task, where each trial represents a complete self-improvement process with 20 training samples for validation during evolution. GuideEvolve autonomously determines when to terminate based on either reaching a performance plateau or a maximum of 30 iterations, though in practice, it never requires the full 30 iterations. All quantitative performance results reported in Tables 2 and 3 in Section 7.1 are based on this full experimental setup.

Note on Experimental Scales. We use two experimental configurations in this paper, summarized in Table 1. Configuration A (full-scale) generates all quantitative benchmark results. Configuration B (reduced-scale) enables eficient visualization and ablation analysis without overwhelming complexity. The higher accuracies in Configuration B visualizations (e.g., up to 0.97) reflect the smaller test scale and should not be compared with Configuration A benchmarks.

Table 1: Experimental Configurations
<table><tr><td>Config</td><td>Train</td><td>Test</td><td>Used In</td></tr><tr><td>A: Full-scale</td><td>20</td><td>800</td><td>Tables 2, 3</td></tr><tr><td>B: Reduced-scale</td><td>5</td><td>100</td><td>Figs 3, 4, 7; Table 4</td></tr></table>

Table 2: Accuracy (%) Comparison of Agent Systems on MGSM and MMLU tasks
<table><tr><td>Agent</td><td>MGSM</td><td>MMLU</td></tr><tr><td colspan="3">Hand-Designed Agent Systems</td></tr><tr><td>Chain-of-Thought[13] COT-SC[12] Self-Refine[7]</td><td> $2 8 . 0 \pm 3 . 1$   $2 8 . 2 \pm 3 . 1$   $2 7 . 5 \pm 3 . 1$   $3 9 . 0 \pm 3 . 4$   $3 1 . 1 \pm 3 . 2$ </td><td> $6 5 . 4 \pm . 3$   $6 5 . 9 \pm 3 . 2$   $6 3 . 5 \pm 3 . 4$   $6 5 . 6 \pm 3 . 3$   $6 5 . 1 \pm 3 . 3$ </td></tr><tr><td colspan="3">Step-back-Abs[19] Quality-Diversity[6]  $2 3 . 8 \pm 3 . 0$   $6 5 . 1 \pm 3 . 3$  Role Assignment[14]  $3 0 . 1 \pm 3 . 2$   $6 4 . 5 \pm 3 . 3$ </td></tr><tr><td colspan="3">Meta-Learning Optimized Agents Meta Agent Search[5]  $5 3 . 4 \pm 3 . 5$   $6 9 . 6 \pm 3 . 2$ </td></tr><tr><td colspan="3">Autonomous Self-evolving Agents</td></tr><tr><td colspan="3">Gödel-base  $G \mathrm { P T } 3 . 5 [ 1 6 ] ^ { \mathrm { a } }$   $6 4 . 2 \pm 3 . 4$  Gödel-free-GPT4.0[16]b  ${ \bf 9 0 . 6 \pm 2 . 0 }$ </td></tr></table>

<sup>a</sup> "base": Gödel agent with GPT-3.5-turbo as base model  
<sup>b</sup> "free": Gödel agent with unrestricted model selection (GPT-4.0)

Baseline Approaches. We compare GuideEvolve against both traditional hand-designed agent systems and advanced learningbased approaches<sup>1</sup>. The hand-designed methods include Chain-of-Thought (CoT) ([13]), which implements step-by-step reasoning, and its extension CoT with Self-Consistency ([12]), which generates multiple solution paths to select the most consistent answer. Self-Refine ([7]) enables iterative self-correction, while LLM-Debate ([2]) facilitates multi-agent debate for diverse perspectives. Stepback Abstraction ([19]) emphasizes understanding fundamental principles before detailed problem-solving, Quality-Diversity ([6]) combines various solution approaches, and Role Assignment ([14]) leverages diferent perspectives through specific LLM roles. Given the limitations of these fixed algorithms in handling dynamic scenarios, we also compare against Meta Agent Search ([5]), the stateof-the-art in automated agent design. Additionally, we benchmark against Gödel ([16]), the autonomous self-evolving framework that inspired GuideEvolve, using both its GPT-3.5 (base) and GPT-4.0 (free) implementations. In addition, for a fair comparison without LLM model diferences to understand the impact of task expert and 3Ts, we also carried out an ablation study in Appendix 7.3.

## 7.1 Performance Analysis

Improved Stability with Competitive Accuracy. As shown in Table 2, we report the test accuracy along with the 95% bootstrap confidence intervals on test sets, where bold indicates the best performing method. GuideEvolve demonstrates improved stability across both tasks while maintaining competitive accuracy. Across 6 independent trials with 800 test samples per task (identical to Gödel’s experimental setup), GuideEvolve achieves 89.8% accuracy (±1.5%) on MGSM, delivering performance comparable to Gödel-free (90.6% ±2.0%) but with 25% lower variance. For MMLU, GuideEvolve reaches 89.1% accuracy (±0.9%), surpassing Gödel-free’s 87.9% (±2.2%) by 1.2 percentage points while achieving 59% lower variance.

Table 3: Comparison of Robustness and Evolution Iterations Required
<table><tr><td>Agent</td><td>Success rate</td><td>Iter.</td></tr><tr><td colspan="3">MGSM task</td></tr><tr><td>Gödel agent[16]</td><td>82%</td><td>30.0</td></tr><tr><td>GuideEvolve (ours)</td><td>100%</td><td>15.1</td></tr><tr><td colspan="3">MMLU task</td></tr><tr><td>Gödel agent[16]</td><td>unreported</td><td>30.0</td></tr><tr><td>GuideEvolve (ours)</td><td>100%</td><td>13.4</td></tr></table>

Success rate refers to autonomous termination of evolution and a better solution achieved than the initial state.

The performance diferences across tasks reveal important insights into domain-specific characteristics of self-evolution. For mathematical reasoning tasks (MGSM), while the self-evolving agent successfully explores solver prompts and workflow modifications, the inherent limitations of LLMs in mathematical computation without external tools constrain the achievable performance gains, resulting in comparable accuracy to the baseline. In contrast, for knowledge-intensive verbal reasoning tasks (MMLU), the self-evolution process demonstrates greater eficacy in optimizing solver logic through prompt engineering and reasoning strategies, leading to measurable accuracy improvements over the baseline. These results align with the intuition that self-evolution mechanisms are more efective in domains where linguistic and reasoning optimizations can substantially impact performance, whereas domains requiring symbolic computation or external tool integration present fundamental capability boundaries.

The observed accuracy and variance improvements are primarily attributable to our architectural innovations rather than underlying model diferences. Our ablation study (Table 4) confirms that removing the expert module and throwback mechanism—while maintaining the same Claude models—results in 17% accuracy decrease and substantial performance degradation, isolating the framework’s contribution. These improvements stem from two key mechanisms: First, our task expert module provides objective performance evaluation and actionable proposals, eficiently guiding the self-evolving agent towards better performance, in contrast to Gödel’s approach of relying on the agent to self-evaluate raw task-solver interactions, which often leads to context overflow and potentially biased judgments. Second, GuideEvolve’s throwback mechanism enables the agent to revisit and analyze its previous successful strategies, facilitating comparative learning from its own successes and failures. This self-introspection capability becomes particularly valuable when the agent encounters performance plateaus, allowing it to draw insights from its evolution history to push performance boundaries further.

Enhanced Robustness. In Table 3, GuideEvolve demonstrates robustness with a 100% success rate in evolution convergence and finding an improved solution over initial state for both tasks. This perfect convergence rate significantly outperforms Gödel’s 82% success rate on MGSM (with unreported convergence on MMLU), where Gödel experiences stability issues in 4% of trials (unexpected termination during evolution) and optimization failures in 14% of trials. Our method’s success rate across both tasks underscores the stability and reliability of our guided evolution approach. This robustness is achieved through two features: the throwback functionality enables the agent to recover from suboptimal evolution paths by rolling back to previous successful states which prevents optimization failures, while termination by choice allows the agent to intelligently halt the evolution process when reaching a performance plateau, avoiding deterioration from forced continuation beyond the agent’s current capability limits.

Higher Evolution Eficiency. GuideEvolve demonstrates remarkable eficiency in its self-improvement process, as shown in Table 3. It requires only 15.1 iterations for MGSM and 13.4 iterations for MMLU on average, compared to Gödel’s 30 iterations—a reduction of approximately 50% and 55% in evolving efort respectively. This eficiency is particularly noteworthy given that GuideEvolve achieves competitive or superior accuracy in fewer iterations. This eficiency boost is attributed to: 1) expert guidance facilitates faster convergence towards the best solver; 2) GuideEvolve’s intelligent self-regulation capability to autonomously determine termination based on performance plateaus avoids unnecessary iterations.

## 7.2 Evolution Traces and Discussions

As noted in the experimental setup, the following analyses use the reduced-scale visualization setup (5 training, 100 test samples) to illustrate mechanisms clearly.

Case Study. To illustrate GuideEvolve’s operation, we examine a representative evolution instance on the MGSM task. Visualization of this instance’s evolution path can be found in Figure 3. Starting with an initial solver accuracy of 0.9 on the test set, the agent first perceives its environment to understand its current state. In its first evolution step, the agent modifies the solver to generate more responses and implement a verification mechanism to select the most reliable answer. Despite achieving 100% accuracy on the validation set, the test set accuracy initially drops to 0.91, revealing a challenging overfitting scenario. The agent continues evolution despite reaching validation set perfection, avoiding early termination through our minimum iteration requirement. This constraint proves valuable as subsequent iterations explore diferent strategies, such as implementing few-shot learning, which improves test accuracy to 0.94. Through iterations 4-8, the agent experiments with various modifications, revealing two distinct scenarios: when validation failures occur, the task expert provides highly efective and targeted guidance; however, in overfitting cases (100% validation accuracy but low test performance), the task expert’s efectiveness is limited due to lack of observable failures, leading to less eficient exploration by the self-evolving agent. A crucial breakthrough occurs at iteration 9 when the agent calls throwback and modifies previous best-performing logic, achieving 0.97 accuracy. After further exploration without additional improvements, the agent autonomously terminates at iteration 18, exemplifying the framework’s ability to balance exploration with eficient termination.

![](images/6ba64cfbbfb233dfff27b1e6946cfb6fcbfc29e1c85883e908b1af12a8863f89.jpg)

Figure 3: One example evolution of GuideEvolve Agent on MGSM.  
![](images/8632101221b56ab2492eddfee585e38382b35f485dd5ac1a50687d88f5d4df55.jpg)  
Figure 4: GuideEvolve: Frequency of actions taken by task

Action Analysis. Across all trials of GuideEvolve on both tasks, the number of actions taken by the agent reveals consistent behavioral tendencies across tasks, as shown in Figure 4. The agent predominantly engages in self-modification and evaluation actions, which directly align with its core objective of solver improvement. Self-modification allows the agent to refine its problem-solving logic, while evaluation leverages task expert guidance to assess these modifications’ efectiveness. Notably, the two introduced mechanisms - throwback and self-termination - rank as the next most frequent actions, indicating their significant role in the agent’s evolution strategy. This action distribution demonstrates that the agent actively utilizes these auxiliary mechanisms to navigate its self-evolution path more efectively, supporting the value of incorporating these additional control mechanisms in the agent’s decision-making repertoire.

Accuracy Progression Analysis. Figure 7 in Appendix C demonstrates distinct paths of self-improvement for three independent GuideEvolve agents, all initiating from a notably high baseline accuracy of 0.9 on the MGSM task. Despite the challenging nature of improving upon such a strong starting point, each agent successfully enhanced its problem-solving capabilities through diferent trajectories. The first agent (orange line) exhibited rapid convergence, reaching an accuracy of 0.96 within just 8 iterations before self-terminating. The second agent (green line) showed steady im provement over 18 iterations, ultimately achieving an accuracy of 0.97. The third agent (blue line) demonstrated resilience by recovering from a temporary complete performance drop (accuracy of 0) - likely due to output formatting issues of the solver - and continued its evolution to reach a peak accuracy of 0.97 after 19 iterations. These diverse evolution patterns highlight the agent’s ability to autonomously navigate and recover from setbacks while consistently finding paths to improved performance.

Table 4: Ablation Study Results
<table><tr><td>Ablation</td><td></td><td>Success (%) Accuracy (%) Iterations</td><td></td></tr><tr><td>w/o expert nor throwback</td><td>80 ↓20</td><td>79 ↓17</td><td>22.5 ↑8.9</td></tr></table>

## 7.3 Ablation Study

To isolate the impact of the task expert module and throwback mechanism, we conduct an ablation study using the reduced-scale setup (5 training, 100 test samples) described in the note of Section 7. This smaller scale enables eficient comparison of architectural variations while maintaining representative performance patterns.

Table 4 demonstrates the critical role of task expert guidance and throwback mechanisms in our system. We chose a baseline that maintains all other features while removing these two components, as this represents the minimal viable configuration required for evolution with bedrock and Claude models. The results show significant performance degradation across all metrics: success rate drops by 20%, accuracy decreases by 17%, and the number of iterations increases by 8.9. Without task expert guidance, the agent struggles with self-evaluation of solutions, while the absence of throwback functionality makes it dificult to recover from performance deterioration. These limitations, combined with frequent context overflow issues, can lead to fatal errors that prematurely terminate the evolution process, highlighting the importance of these mechanisms for robust and eficient self-evolution.

## 7.4 Limitations and Threats to Validity

Model Confound. Our primary comparison evaluates Claudebased GuideEvolve against GPT-based Gödel Agent. While our ablation study (Table 4) isolates the contribution of our architectural innovations by comparing variants using identical Claude models, we acknowledge that full cross-model validation (e.g., GuideEvolve with GPT models or Gödel with Claude models) would strengthen claims of framework-independent improvements. The observed sta bility gains are consistent across our Claude-based ablation experiments, suggesting architectural rather than model-driven benefits.

Task Scope. Our evaluation focuses on two benchmark tasks (MGSM and MMLU) with clear ground truth and quantitative metrics. Generalization to open-ended business applications with multifaceted success criteria remains future work, as discussed in Section 8.

Computational Cost. GuideEvolve introduces additional LLM calls through the expert module. While this increases per-iteration cost compared to Gödel, the 50% reduction in required iterations (Table 3) partially ofsets this overhead. Detailed cost-benefit analysis for production deployment is warranted.

## 8 Future Work

Our framework opens several promising research directions for future development. While currently focusing on solver evolution, future research could explore simultaneous evolution of both solver and expert components through co-adaptive improvement, implementing meta-evaluation capabilities where the agent assesses both guidance efectiveness and solver performance. Moreover, integration with the Strands Agents framework presents opportunities for enhanced production readiness and scalability. Additionally, extending GuideEvolve framework to business applications presents unique challenges due to lack of ground truth and multifaceted success criteria, necessitating research into robust evaluation metrics for complex business scenarios and methods for decomposing business problems while maintaining coherent evolution strategies.

## Acknowledgments

This study has been done as part of Science Exchange program within Amazon Europe. Thanks to our sponsors in AWS Sales & Marketing Operations leadership.

## References

[1] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073 (2022).

[2] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning.

[3] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al. 2024. Retrieval-augmented generation with graphs (graphrag). arXiv preprint arXiv:2501.00309 (2024).

[4] Shengran Hu, Cong Lu, and Jef Clune. 2024. Automated Design of Agentic Systems. arXiv preprint arXiv:2408.08435 (2024).

[5] Shengran Hu, Cong Lu, and Jef Clune. 2024. Automated design of agentic systems. arXiv preprint arXiv:2408.08435 (2024).

[6] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. 2024. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292 (2024).

[7] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. 2023. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems 36 (2023), 46534–46594

[8] Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, FranciscoJR Ruiz, Abbas Mehrabian, et al. 2025. AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery. arXiv preprint arXiv:2506.13131 (2025).

[9] Yu Shang, Yu Li, Keyu Zhao, Likai Ma, Jiahe Liu, Fengli Xu, and Yong Li. 2025. AgentSquare: Automatic LLM Agent Search in Modular Design Space. arXiv preprint arXiv:2503.XXXXX (2025).

[10] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. Interleaving retrieval with chain-of-thought reasoning for knowledgeintensive multi-step questions. arXiv preprint arXiv:2212.10509 (2022).

[11] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Plan-and-solve prompting: Improving zero-shot chainof-thought reasoning by large language models. arXiv preprint arXiv:2305.04091 (2023).

[12] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022).

[13] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 24824–24837. https://proceedings.neurips.cc/paper\_files/paper/2022/file 9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf

[14] Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Yongdong Zhang, and Zhendong Mao. 2023. Expertprompting: Instructing large language models to be distinguished experts. arXiv preprint arXiv:2305.14688 (2023).

[15] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR).

[16] Xunjian Yin, Xinyi Wang, Liangming Pan, Li Lin, Xiaojun Wan, and William Yang Wang. 2025. Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement. arXiv preprint arXiv:2503.12345 (2025).

[17] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jef Clune. 2025. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv preprint arXiv:2505.22954 (2025).

[18] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, et al. 2024. AFlow: Automating Agentic Workflow Generation. arXiv preprint arXiv:2410.10762 (2024).

[19] Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H Chi, Quoc V Le, and Denny Zhou. 2023. Take a step back: Evoking reasoning via abstraction in large language models. arXiv preprint arXiv:2310.06117 (2023).

## A Goal Prompt

Figure 5 presents the goal prompt that is used in the self-evolving agent framework. The important parts related to guidance and 3T mechanisms are highlighted with red font.

## B Illustration of Termination by Choice

Figure 6 shows how the self-evolving agent autonomously decide to terminate its evolution process.

## C Accuracy Progression by Independent GuideEvolve Agents

In Figure 7, we illustrate the accuracy progression of three independent trials of GuideEvolve on the MGSM task using the reducedscale visualization setup (5 training, 100 test samples) described in Section 7. The higher accuracies observed here (up to 0.97) reflect the smaller test scale and demonstrate evolution trajectories rather than benchmark performance. The explanation details can be found in accuracy progression analysis of Section 7.2.

## D Examples of Best Policies Found by GuideEvolve

We present representative examples of optimal solvers discovered by GuideEvolve for both the MGSM and MMLU tasks. As noted in in the Experimental Setup (Section 7), SONNET\_V36 and SONNET\_V37 are implementation identifiers for Claude 3.5 Sonnet v2 and Claude 3.7 Sonnet respectively. Notably, GuideEvolve was not provided with explicit model descriptions or capabilities; rather, it autonomously discovered and leveraged the relative strengths of these models through its evolutionary process. This demonstrates the system’s ability to independently discern and utilize the varying capabilities of diferent LLM models.

Figure 8 illustrate the optimal solver found by one GuideEvolve instance for the MGSM task. The solution demonstrates significant improvements over the initial solver’s single response approach, implementing a more robust strategy that generates multiple responses at three diferent temperature settings and determines the final output through majority voting. Furthermore, the prompt has been enhanced with task-specific guidance. A particularly noteworthy adaptation emerged in handling Bengali mathematics questions, where the solver fallback to become a specialized Bengali mathematics expert by first translating the problem before applying mathematical reasoning, efectively addressing the challenge of solving problems in this less-commonly supported language.

On the other task MMLU, the best solver discovered through one GuideEvolve instance is presented in Figure 9. The final solution demonstrates remarkable adaptive intelligence in its approach to model selection, autonomously developing a sophisticated mechanism that dynamically allocates computational resources based on task complexity. Specifically, the solver learned to use question length as a proxy for problem complexity, implementing a bifurcated strategy: for more complex questions with longer text, it employs focused responses through the more capable Anthropic Claude 3.7 Sonnet model, while for simpler queries, it eficiently utilizes Anthropic Claude 3.5 Sonnet v2 with a temperature setting of 0.8. This emergent behavior showcases the system’s ability to optimize resource allocation and model selection without explicit programming.

## E An Example of Task Expert’s Performance Report

In Figure 10, we’ve shown the performance report generated by the MGSM task expert provided to the self-evolving agent in one iteration during the evolving process. It gives a detailed performance analysis of the current solver’s strengths, weakness, case study of failures and actionable recommendations to facilitate the further improvement.

## Goal prompt for GuideEvolve

You are a self-evolving agent (self\_evolving\_agent) with full access to modify logic, run code, and manipulate the environment.   
Your goal is to continuously enhance your ability to solve tasks accurately by dynamically evolving your logic.

## Core Capabilities:

\- Com plet e Aut onom y: Unrestricted access to modify logic, run code, manipulate environment

\- Environment Interaction: Perceive, read, modify, execute code and actions

\- Problem-Solving: Apply creative algorithms to tackle challenges, optimize solutions

\- Collaborat ion: Leverage AWS Bedrock LLM for insights, strategies, error correction

\- Error Handling: Analyze errors systematically. Troubleshoot, backtrack to PREVIOUS BEST state if performance drop signifantly!

\- Concise Response: always respond concise and clear. Keep responses short.

## Core Met hods

\- evolve: Continuously enhance performance by interacting with environment

\- execute\_action(actions): Execute actions based on analysis or feedback

\- solver(agent\_instance, task\_input: str): Solve target task using current capabilities

## Key Guidelines:

\- All functions are in module agent\_module

\- You can call MULTIPLEt ools at once. BUT do NOT call action\_evaluate\_on\_task unless seeing message of "Successfully modify agent\_module.solver" OR only ONE optional message (content MUST show "logic of solver...").

\- action\_adjust\_logic: Modify code carefully, don't change interfaces, avoid unnecessary changes.

\- action\_run\_code: Store useful objects in environment, import/install libraries as needed, always code short!

\- Ext ernal Collaborat ion: Use action\_call\_json\_format\_llm for logic refinement and tool creation

\- action\_evaluate\_on\_task: Assess solver performance ONLY after SUCCESSFULLY modifying the logic of solver.

\- solver:

\- Output MUST be dictionary with final answer under key "answer"

\- For debugging, don't print, and instead return the debug information.

\- Use action\_call\_json\_format\_llm exclusively for AWS Bedrock LLMs

\- Can call LLMs multiple times across rounds to improve performance

\- MUST specify role when calling different role-based LLMs

\- For each key, if a specific format is required, such as int, float, enum or list, the requirements must specify the conditions.

\- Explore techniques: LLM Debate, Step-back Abstraction, Quality-Diversity, Dynamic Role Assignment,

Self-consistency, Few-shots, Task Decomposition, Reflective Evaluation

\- action\_display\_analysis: Always analyze before acting. Include performance improvement plans, case studies of low-score examples, error handling, solving ideas. Keep analysis short and concise.

\- action\_retrieve\_best\_solver: Retrieve the previous best solver, use when backtrack.

\- action\_terminate\_evolve: CRITICAL - Terminates evolution permanently. ONLY use when: Current Test accuracy > 0.99 OR ALL accuracies of previous 4 evolves are no worse than current.

## Response Form at :

IMPORTANT: You MUST format response as JSON object with "tool\_calls" array containing function calls. Each function call must have "type": "function" and include function name and arguments.   
Example:

![](images/3030c202defbbc7fea594490a9b3607868829fbeb6e8a82a6fa05c64b5005468.jpg)

\- Put tool\_calls at end, functions in sequence corresponding to reasoning flow

\- Always respond wit h at least one funct ion in t ool\_calls

Figure 5: Goal Prompt for GuideEvolve

![](images/3dfde8495bae0285c341ddc626f350b31cee7d643f3fdb501b8c2d4c59e4f86c.jpg)

Figure 6: An illustration of implementation of GuideEvolve that calls the termination action.  
![](images/7157b1c91234015432a7212cbaa12633bba9ecf58e2f63e9292f057c9d95382c.jpg)  
Figure 7: Accuracy Progression on MGSM by Independent Agent Trials of GuideEvolve

system\_msg = {   
"content": "You are a mathematics expert proficient in all languages   
problems step by step, showing clear reasoning. Pay special attention to   
temporal sequences and state changes. Verify your answer carefully. The answer   
}   
answers, reasoning\_list = [], []   
messages = [system\_msg,   
{"role": "user", "content": f"# Math Problem:\n{task}"}]   
response = agent.action\_call\_json\_format\_llm(   
messages=messages,   
temperature=temp,   
num\_of\_response=1,   
role="multilingual math expert",   
return\_dict\_keys=["reasoning", "answer"],   
requirements="   
1. Break down the problem into clear steps\n   
2. Track all state changes and temporal sequences carefully\n   
3. Show detailed calculation process\n   
4. Double-check your answer\n   
5. The answer MUST be an integer")   
result = response[0]   
answer = str(result.get("answer", "")).strip()   
if answer and answer.isdigit():   
answers.append(answer)   
reasoning\_list.append(result.get("reasoning", ""))   
if not answers:   
messages = [{   
response = agent.action\_call\_json\_format\_llm(   
messages=messages,   
temperature=0.5,   
num\_of\_response=1,   
role="Bengali math expert",   
result = response[0]   
answer = str(result.get("answer", "")).strip()   
if answer and answer.isdigit():   
return {"reasoning": result.get("reasoning", ""),"answer":   
answer}   
most\_common\_answer = Counter(answers).most\_common(1)[0][0]   
return {"reasoning": reasoning\_list[answers.index(most\_common\_answer)],   
"answer": most\_common\_answer)  
Figure 8: An example of the best solver found by GuideEvolve on Task MGSM

def solver(agent, task: str):   
if len(task) > 500 or 'history' in task.lower():   
model = SONNET\_V36   
temp = 0.8   
messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]   
response = agent.action\_call\_json\_format\_llm(   
model=model,   
messages=messages,   
temperature=temp,   
num\_of\_response=1,   
role="expert in multiple domains with strong analytical and reasoning skills",   
return dict keys=["reasoning", "answer"],   
requirements=(   
$" 1 .$ Analyze the question carefully and break down complex concepts.\n"   
relevant.\n"   
).strip(),   
return dict = response[0]   
return\_dict["answer"] = str(return\_dict.get("answer", ""))   
return return\_dict  
Figure 9: One example of the best solver found by GuideEvolve on Task MMLU.

![](images/e73ce148a024017491870ea285277a19f1b1eb9ff06cdd23a4af8d867826b94b.jpg)