# EvoRouter: Cost-Eficient Multi-Agent Routing via Insight-Guided On-Policy Distillation

Yitao Xiao<sup>∗</sup>   
Beijing University of Posts and   
Telecommunications   
Beijing, China   
yitooooom@bupt.edu.cn   
Shaoyong Guo   
Beijing University of Posts and   
Telecommunications   
Beijing, China   
syguo@bupt.edu.cn   
Sitao Xie   
Fudan University, Shanghai   
Innovation Institute   
Shanghai, China   
26113050151@m.fudan.edu.cn   
Xu Zhang<sup>∗</sup>   
Beijing University of Posts and   
Telecommunications   
Beijing, China   
zhangxu2024@bupt.edu.cn   
Yan Sun   
Beijing University of Posts and   
Telecommunications   
Beijing, China   
sunyan79@bupt.edu.cn

Guo Tong Nanyang Technological University Singapore tong.guo@ntu.edu.sg

Ye Cheng<sup>†</sup>   
Beijing University of Posts and   
Telecommunications   
Beijing, China   
yecheng@bupt.edu.cn   
Ziyue Zhu   
Beijing University of Posts and   
Telecommunications   
Beijing, China   
zyzhu@bupt.edu.cn

Haoran Luo<sup>†</sup> Nanyang Technological University Singapore haoran.luo@ntu.edu.sg

## Abstract

Serving web queries at scale with multi-agent systems requires jointly configuring large language models, collaboration modes, agent counts, interaction rounds, and skills. However, the resulting configuration space grows combinatorially, while sparse execution rewards provide limited fine-grained supervision for interdependent planning decisions. We propose EvoRouter, a selfevolving multi-agent routing framework driven by insight-guided on-policy distillation. A dedicated expert periodically synthesizes performance–cost feedback from configuration trajectories across tasks into reusable conditional routing insights about capability requirements, returns from resource scaling, and interactions among configuration choices. A privileged teacher combines these evolving insights with current execution feedback to provide dense token-level supervision for the router’s own planning trajectories through on-policy distillation. Cost-aware reinforcement learning jointly optimizes task performance and computational cost. Together, these training signals internalize execution insights into the routing policy. The updated policy generates new trajectories that support subsequent insight extraction, forming an execution-driven self-improvement loop. We evaluate EvoRouter on 19 in-domain datasets spanning five domains and two out-of-domain benchmarks. EvoRouter achieves an average in-domain accuracy of 68.69%, outperforming the strongest routing baseline by 7.23 percentage points while reducing inference compute cost by 50.95%. Our code is opensourced at https://github.com/CenYangtze/EvoRouter.

## Keywords

LLM Routing, Multi-Agent Systems, Recursive Self-Improvement, On-Policy Distillation, Reinforcement Learning

![](images/89cc7cb3afc9d4cc72bf895c28b1f38a1e8a5d6b2ba7539b9a731d0e37ba6679.jpg)  
How can experience be distilled from configurations and outcomes, then internalized in routers?  
Figure 1: MAS router for web queries.

## 1 Introduction

Agents powered by large language models (LLMs) are rapidly expanding the capabilities of Web services [29, 35], from information seeking to complex reasoning and problem solving [21, 38]. LLM routers have emerged as a key component of agentic infrastructure, selecting suitable models for individual queries to balance response quality and inference cost [4, 12, 36, 47]. With the growing adoption of multi-agent systems (MAS) in agentic web services such as Grok Bot and WorkBuddy, routing must extend beyond model selection to jointly determine model assignments, collaboration modes, agent counts, interaction rounds, and the skills and subtasks assigned to each agent. For service providers handling Web queries across diverse domains and dificulty levels, adaptively selecting MAS configurations to jointly optimize reasoning performance and compute cost has become a central research problem.

![](images/aa792ab4e070f3d8514a2fcb9615b6b382c0662abb0fcab34f447c8d39befb7d.jpg)  
Figure 2: Motivation and performance of EvoRouter. Left: Two challenges and our solutions. Right: EvoRouter’s performance.

Existing MAS routing research primarily improves routing de cisions through two mechanisms. Parametric policy optimization internalizes execution feedback into router parameters through reinforcement learning or feedback-conditioned distillation, learning policies for model assignment and system orchestration [13, 25, 44, 59]. Explicit experience evolution updates retrievable execution records, routing rules, or orchestration skills, allowing accumulated experience to guide subsequent decisions without gradient updates to the router [27, 31, 41, 60]. Connecting experience reuse with policy learning is particularly challenging for joint MAS configuration: the benefits of a model, collaboration mode, or resource allocation depend jointly on task requirements and the surrounding configuration choices. Reusable experience must therefore preserve these conditions of applicability, while efective policy learning must translate the relevant knowledge into the ability to plan new configurations. As illustrated in Figure 1, this raises a central open question: How can configuration experience be distilled into reusable knowledge of conditional dependencies and internalized in the router’s pre-execution planning policy to improve the system-level trade-of between accuracy and cost?

Addressing this question presents two challenges, as highlighted in Figure 2. (i) Discovering reusable routing principles within a complex, multidimensional configuration space. Jointly selecting models, collaboration modes, agent counts, interaction rounds, and skills causes a combinatorial explosion in the configuration space, making exhaustive evaluation computationally prohibitive [48]. Moreover, because configuration benefits depend on both the task and the combination of choices, a single execution trajectory reveals only the outcome under a specific configuration [56]. It provides insuficient evidence to identify necessary, redundant, or suboptimal decisions. The router must therefore move beyond learning from isolated outcomes and analyze execution trajectories across tasks and configurations to uncover conditional configuration patterns that generalize to subsequent tasks. (ii) Translating post-execution experience into fine-grained supervision for pre-execution routing. Existing reinforcement learning approaches to routing derive reward signals solely from highly sparse feedback on final task success and computational cost [7, 30, 61]. However, a complete MAS configuration involves multiple coupled planning decisions. Terminal rewards provide limited guidance on which decisions are efective, which waste resources, and how individual configuration choices should be adjusted. Even when valuable routing insights can be distilled from historical trajectories, they do not automatically translate into efective action selection at specific decision states of the current routing policy. This mismatch between high-dimensional combinatorial decisions and coarse, sparse execution feedback limits the router’s ability to learn eficiently and improve continually from limited execution experience.

To address these challenges, we propose EvoRouter, an adaptive multi-agent routing framework built around an insight-guided selfimprovement loop. To address the first challenge, we conduct preliminary experiments that systematically examine how configuration changes afect reasoning performance and computational cost across tasks, revealing task dependence, diminishing marginal returns, and configuration interdependence. We then design an LLM-based expert that distills routing insights by periodically aggregating configurations, execution outcomes, and costs across tasks. The expert derives conditional insights into task-specific capability bottlenecks, marginal returns from resource scaling, and interactions among configuration choices. These insights guide exploration according to task characteristics, steering the router toward optimal configurations. To address the second challenge, we develop a router training method that combines privileged onpolicy distillation (OPD) with cost-aware reinforcement learning (RL). OPD transforms configuration insights accumulated across trajectories into dense, token-level rewards informed by routing insights, while RL optimizes the performance–cost trade-of using end-to-end execution outcomes. These complementary training signals address the challenge of learning interdependent multi-agent configuration decisions from sparse execution feedback.

Our main contributions are summarized as follows:

1. Conditional routing insights distillation. We identify task dependence, diminishing marginal returns, and interdependence among MAS configuration choices. We distill cross-trajectory performance–cost feedback into configuration insights with explicit applicability conditions to guide exploration.

2. Insight-guided router post-training. We propose EvoRouter, which internalizes conditional insights into the routing policy through OPD and jointly optimizes execution utility with RL, providing fine-grained supervision for routing decisions.

Table 1: Representative learning methods for MAS routing.
<table><tr><td>Method</td><td>Routing scope</td><td>Learning signal</td><td>Training mechanism</td></tr><tr><td>RouteLLM [36]</td><td>Single-model choice</td><td>Pairwise preferences</td><td>Supervised learning</td></tr><tr><td>Router-R1 [61]</td><td>Multi-round calls</td><td>Format, outcome, cost</td><td>Outcome-based RL</td></tr><tr><td>MasRouter [59]</td><td>Modes, roles, models</td><td>Utility-cost reward</td><td>Policy gradients</td></tr><tr><td>DAAO [44]</td><td>Depth, operators, models</td><td>Success, utility, cost</td><td>Difficulty calibration</td></tr><tr><td>MAS-Orchestra [25]</td><td>Holistic MAS design</td><td>Task-level rewards</td><td>Function-calling GRPO</td></tr><tr><td>Skill-MAS [31]</td><td>Meta skill</td><td>Rollout contrasts</td><td>Nonparametric reflection</td></tr><tr><td>RSI-Router [27]</td><td>Subtask models, skills</td><td>Historical quality-cost traces</td><td>Skill update + Pareto selection</td></tr><tr><td>EvoRouter (ours)</td><td>Joint MAS configuration</td><td>Token-level supervision from insights, execution feedback, compute overhead</td><td>SFT + RL + Insight-guided OPD</td></tr></table>

![](images/b50ba495f98f57db349ffbb6b23a8ea3b62d2ea805d35185202b016f215a6ba0.jpg)

![](images/7ed40a6391f1e61ec692f60ed999e7e6291f19af740349a23b22e32d8c7c5e43.jpg)

![](images/48f67519a8e8e340bfba6cbc83cc8ba138ea0432589b8a835c48193d42fd6f57.jpg)

![](images/b7784d8fb5d0522d64f8a552a02f1c640a3a5f4556f2b9ef20c9135cb0f09421.jpg)  
Figure 3: Empirical insights into MAS configuration. (a) Cross-domain sensitivity to diferent resource scaling; (b) Nonmonotonic returns across interaction rounds (left) and agent counts (right). (c) Combinatorial synergy between model composi tions and collaboration modes.

3. Comprehensive empirical validation. We construct a benchmark covering 21 scenarios across seven domains, demonstrating that EvoRouter achieves better performance, lower cost, and greater robustness than representative baselines.

## 2 Related Work

LLM Routing and Multi-Agent Orchestration. LLM routing spans per-query model selection [4, 10–12, 36, 47] and outputbudget allocation [54], sequential model invocation [39, 61], and joint MAS configuration [25, 44, 53, 59, 63]. Beyond team design, routers adapt delegation [7, 13], agent or context selection [32, 62], and execution state [28, 41]. Experience-based approaches retain reusable routing knowledge [31, 50, 60] or co-evolve routing with worker capabilities [19]. RSI-Router[27] evolves subtask assignments and model-specific skills through Pareto selection. EvoRouter targets pre-execution joint configuration over fixed heterogeneous workers, informed by our analysis of task-dependent scaling and configuration interactions.

Post-Training for LLMs. LLM post-training commonly combines supervised fine-tuning (SFT) on demonstrations with prefer ence optimization [37, 40]. Reinforcement learning with verifiable rewards further develops reasoning capabilities from outcome feedback [16]. Complementing reward-based optimization, on-policy distillation provides teacher supervision on student-generated sequences and can reduce the training–inference distribution mismatch associated with ofline distillation [2]. Recent methods enrich this supervision with textual feedback [20], trajectory-derived skills [49, 52], and role-specific coordination feedback [64]. Building on these principles, EvoRouter combines group-relative RL with OPD on student-generated configuration plans, conditioning the teacher on evolving cross-task insights and current execution feedback. The resulting policy operates without privileged feedback at inference.

## 3 Routing Insights and Problem Definition

## 3.1 Routing Insights In Multi-Class Tasks

Eficient MAS routing requires determining which resources to scale, to what extent, and how to combine them. Using $m _ { \alpha }$ (Qwen3.5-35B-A3B) and $m _ { \beta }$ (Qwen3.5-397B-A17B) as representative backbones, we systematically examine how resource selection, scaling scale, and configuration combinations govern task success and compute cost. Figure 3 highlights three key insights.

Where to Scale? Diferent tasks exhibit distinct capability demands, leading to task-dependent benefits from resource scaling. As shown in Figure 3(a), switching from $m _ { \beta } { + } \mathrm { I O }$ to $m _ { \alpha }$ +FullConnected improves the success rate on Multimodal by 13.1 points, while reducing that on QA tasks by 5.0 points. This suggests that scaling decisions should be driven by task demands, prioritizing task-aligned resource configurations (Insight 1).

How Much to Scale? Increasing resource allocation does not necessarily yield proportional gains. Figure 3(b) shows that diferent models and collaboration strategies reach peak performance at diferent interaction rounds or agent counts, beyond which further scaling may increase costs while degrading success rates. Thus, resource scaling exhibits configuration-dependent saturation points, and the allocation scale should be tailored to the underlying model and collaboration strategy (Insight 2).

![](images/5a8a5567c23d3dc7311653fc2499f34f84d33a290aacca66355863393a0ecd15.jpg)  
Figure 4: EvoRouter’s initialization and evolution loop.

What to Combine? The efectiveness of individual resources depends on their interactions within the overall configuration. Figure 3(c) reveals that the relative performance ofmodel compositions can reverse across collaboration strategies: an all-� configuration outperforms an all- $m _ { \alpha }$ configuration under FullConnected, whereas the latter achieves higher success rates under Debate while reducing MAS FLOPs by 79.8%. Therefore, routing insights should capture efective resource combinations and the conditions under which they excel (Insight 3).

Together, these insights guide subsequent insights distillation from three complementary perspectives: task alignment, scaling saturation, and configuration interactions. Building on these find ings, we jointly analyze tasks, complete configurations, execution outcomes, and costs to distill accumulated execution insights into condition-aware routing knowledge.

## 3.2 Problem Definition

We consider cost-aware multi-agent routing, where a router maps a given task $q \sim Q$ to a tailored MAS configuration P drawn from a broad design space spanning models, skills, subtasks, collaboration modes, and agent counts. The router produces a final answer based on the resulting system execution.

Let M, C, and S denote the available pools of models, collaboration modes, and skills, respectively. Given these resources, the router constructs a detailed MAS configuration $\mathcal { P } _ { : }$

$$
\boldsymbol { \mathcal { P } } = \left( c , r , \left\{ ( u _ { j } , m _ { j } , s _ { j } , p _ { j } ) \right\} _ { j = 1 } ^ { n } \right) ,\tag{1}
$$

where $c \in C$ specifies the collaboration mode, � is the number of interaction rounds within the system, and � is the number of agents. Each agent � is defined by its identifier � , model $m _ { j } \in { \mathcal { M } } ,$ , skill $s _ { j } \in S ,$ , and assigned subtask $\boldsymbol { \mathscr { P } } \boldsymbol { j }$

The environment instantiates and executes the MAS, returning an execution outcome $o \sim \mathcal { E } ( \cdot \mid q , \mathcal { P } )$ to the router, where E(· | $q , \mathcal { P } )$ denotes the distribution of execution outcomes conditioned on the task and configuration. The router then generates a final answer �ˆ for task �, forming a complete trajectory �. A task-specific verifier (e.g., sandbox tests for code generation or exact-match evaluation for question answering) evaluates the final answer, with $s ( \tau ) \in \{ 0 , 1 \}$ indicating whether it passes verification.

Let �(�) ≥ 0 denote the execution cost of the routed MAS along trajectory �, and let $B ( q ) > 0$ be a reference budget for task �. We aim to learn a routing policy that jointly optimizes task success and resource eficiency:

$$
\pi ^ { \star } \in \arg \operatorname* { m a x } _ { \pi } \mathbb { E } \underset { \tau \sim p _ { \pi , \mathcal { E } } ( \cdot \vert q , C , S , { M } ) } { \mathbb { E } } \left[ U \left( s ( \tau ) , \frac { C ( \tau ) } { B ( q ) } ; \lambda _ { C } \right) \right] ,\tag{2}
$$

where $\scriptstyle { \mathcal { P } } \pi , { \mathcal { E } }$ denotes the trajectory distribution jointly induced by the routing policy and multi-agent execution, accounting for stochasticity in both policy generation and system execution. The utility function � balances task success against budget-normalized execution cost, with $\lambda _ { C }$ controlling the cost penalty. Its specific form is defined in Section 4.3.

## 4 Method

We parameterize the routing policy as $\pi _ { \theta }$ and initialize it through SFT. We then collect online trajectories across diverse task categories and employ an insight distillation expert to iteratively extract routing insights, which are internalized into the policy through training. This establishes an iterative evolution loop, sustained by reinforcement learning and on-policy distillation in an online environment, as illustrated in Fig. 4.

## 4.1 Evolution Loop

Routing and execution. We construct the router context �, comprising role specifications, task descriptions and available routing resources. The router first analyzes the task and reasons within the <analysis> tag, then generates a structured plan as defined in Eq. (1) within the <plan> tag. A deterministic executor in the environment parses this plan and instantiates the specified MAS.

The MAS then executes the plan in the environment, producing execution outcomes that are returned as observations for the router’s subsequent response. The router analyzes these observations within the <analysis> tag and generates the final answer $\hat { y }$ within the <answer> tag, forming a two-stage trajectory �. Each trajectory records the task description, multi-agent configuration, execution outcomes, and verifier-evaluated final answer.

Sampling and evolution. We deploy the router in an online environment and sample execution trajectories from a training set spanning diverse capability dimensions. Building on the observations from our preliminary experiments (Section 3.1), we develop an insight distillation expert to analyze these trajectories and continually refine routing insights. The expert integrates newly observed routing patterns from recent online trajectories with accumulated insights, capturing task-dependent preferences for resources and subtask decomposition strategies across diferent domains, difi culty levels, and budgets. This process enables routing knowledge to evolve continuously as new execution insights accumulate, as detailed in Section 4.4.

## 4.2 Supervised Initialization

We initialize the routing policy through SFT to establish protocols for task analysis, system planning, and result synthesis. Specifically, a frozen powerful LLM acts as a bootstrap router to collect trajectories across training tasks $\boldsymbol { Q } _ { \mathrm { s f t } }$ under context �. We retain only demonstrations with valid formats and correct answers to construct ${ \mathcal { D } } _ { \operatorname { S F T } } .$ . Letting $\mathcal T ( y )$ index the router-generated tokens in sequence $y ,$ the initialization objective minimizes

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \sum _ { \substack { ( x , y ) \in \mathcal { D } _ { \mathrm { S F T } } } } \sum _ { t \in \mathcal { T } ( y ) } \log \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) .\tag{3}
$$

Prompt contexts and environment feedback are masked during loss computation. The supervised policy initializes subsequent iterative evolution and provides a frozen reference policy $\pi _ { \mathrm { r e f } } .$

## 4.3 Cost-Aware Policy Optimization

Reward curation. For each trajectory �, the task verifier assigns an outcome reward $R _ { \mathrm { p e r f } } = s ( \tau ) \in \{ 0 , 1 \}$ . We measure execution compute $C ( \tau )$ and impose a cost penalty $R _ { \mathrm { c o s t } } = \log _ { 2 } ( 1 { + } C ( \tau ) / B ( q ) )$ calibrated against a task-specific reference budget $B ( q )$ . We then instantiate the utility in Eq. (2) as the following reward function $R ( \tau )$ :

$$
R ( \tau ) = R _ { \mathrm { p e r f } } - \lambda _ { C } R _ { \mathrm { c o s t } } ,\tag{4}
$$

where $\lambda _ { C }$ controls cost sensitivity. To strictly separate the reward ranges of successful and failed rollouts, we further cap the weighted cost penalty at $c _ { \mathrm { m a x } } \in [ 0 , 1 )$ to ensure that successful trajectories receive rewards of at least $1 - c _ { \operatorname* { m a x } } > 0$ , while unsuccessful trajectories receive nonpositive rewards. Details of cost weighting are provided in Appendix A.2.

Group-relative policy updates. For each task �, we sample $G$ rollouts $\tau _ { q , 1 } , \dots , \tau _ { q , G }$ and compute normalized group advantages $A _ { q , i }$ . Let $\mathcal { T } _ { q , i }$ index the router-generated tokens in trajectory $\tau _ { q , i ; }$ conditioned on the preceding interaction history $H _ { q , i , t }$ . Sharing the same token masking as SFT, the policy loss is defined as:

$$
\mathcal { L } _ { \mathrm { { R L } } } ( \theta ) = - \frac { 1 } { Z _ { \mathrm { { R L } } } } \sum _ { q } \sum _ { i = 1 } ^ { G } A _ { q , i } \sum _ { t \in \mathcal { T } _ { q , i } } \log \pi _ { \theta } \left( y _ { q , i , t } \mid H _ { q , i , t } \right) ,\tag{5}
$$

where $Z _ { \mathrm { R L } }$ is the total number of masked response tokens from trajectories with nonzero advantages in the batch. To prevent policy drift, we regularize $\pi _ { \theta }$ against the frozen reference $\pi _ { \mathrm { r e f } }$ via a tokenlevel KL penalty:

$$
\mathcal { L } _ { \mathrm { K L } } ( \theta ) = \frac { 1 } { Z _ { \mathrm { K L } } } \sum _ { q } \sum _ { i = 1 } ^ { G } \sum _ { t \in \mathcal { T } _ { q , i } } \left[ r _ { q , i , t } ( \theta ) - \log r _ { q , i , t } ( \theta ) - 1 \right] ,\tag{6}
$$

where $\begin{array} { r } { r _ { q , i , t } ( \theta ) = \frac { \pi _ { \mathrm { r e f } } ( y _ { q , i , t } | H _ { q , i , t } ) } { \pi _ { \theta } ( y _ { q , i , t } | H _ { q , i , t } ) } } \end{array}$ denotes the reference-to-policy probability ratio, and $\begin{array} { r } { \dot { Z } _ { \mathrm { K L } } = \dot { \sum } _ { \boldsymbol { q } , i } | \mathcal { T } _ { \boldsymbol { q } , i } | } \end{array}$ counts all router response tokens across the batch, including zero-advantage trajectories.

## 4.4 Insight-Guided On-Policy Distillation

Recursive insight consolidation. Guided by the analytical framework in Section 3.1, we initialize an insight pool ${ \mathcal { I } } _ { 0 }$ and employ an expert $\mathcal { F }$ to periodically distill execution observations into generalizable routing rules with explicit applicability conditions only during training. For the �-th insight update using recent rollout batch $\mathcal { B } _ { k }$ , the expert refines the insight set through a three-step procedure: (i) Rule induction: comparing intra-category trajectories across success or failure and cost spectra to derive conditional rules on capability alignment, scaling saturation, and configuration interactions; $( i i )$ Precondition refinement: resolving empirical conflicts within $\mathcal { T } _ { k }$ by tightening boundaries, such as specifying finer-grained task dificulty and domains; and (iii) Capacity pruning: merging redundant entries and enforcing a strict capacity ceiling to curb context inflation. Formally, the update is expressed as:

$$
\begin{array} { r } { { \mathcal { I } } _ { k + 1 } = \mathcal { F } \left( \mathcal { I } _ { k } , \mathcal { B } _ { k } \right) . } \end{array}\tag{7}
$$

Distillation steps always condition on the latest insights $\boldsymbol { \mathcal { I } } .$ . Further details on insight distillation are provided in Appendix A.3.

Insight-conditioned supervision. For each trajectory $\tau _ { i }$ sampled from the student policy, the frozen teacher model $\pi _ { \phi }$ receives additional privileged information $z _ { i } = ( \bar { I } , \mathcal { P } _ { i } , o _ { i } , s ( \tau _ { i } ) )$ , where $o _ { i }$ denotes the MAS execution outcome, while sharing the same context $x _ { i }$ and prefix $y _ { i , < t }$ as the student. Specifically, at each token position � in the routing response, let $p _ { i , t } ^ { \mathrm { S } } = { \pi } _ { \theta } ( { \cdot } \ \vert \ x _ { i } , y _ { i , < t } )$ and $p _ { i , t } ^ { \mathrm { T } } = \pi _ { \phi } ( \cdot \mid x _ { i } , z _ { i } , y _ { i , < t } )$ denote the next-token distributions of the

Table 2: Overall accuracy (%) on in-domain and out-of-domain benchmarks. The two Average columns report sample-weighted accuracy. Best and second-best scores in each column are shown in bold and underline, respectively.
<table><tr><td></td><td colspan="6">In-Domain</td><td colspan="3">Out-of-Domain</td></tr><tr><td>Method</td><td>Math</td><td></td><td>Commonsense Multimodal</td><td>QA</td><td>Knowledge</td><td>Average</td><td>Code</td><td>Reasoning</td><td>Average</td></tr><tr><td colspan="10">Vanilla LLMs</td></tr><tr><td>Qwen3.5-35B-A3B</td><td> $1 6 . 1 5 _ { \pm 0 . 7 7 }$ </td><td> $8 5 . 0 0 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $2 9 . 7 4 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $4 8 . 4 6 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $4 6 . 6 7 _ { \pm 2 . 8 6 }$ </td><td> $4 6 . 2 3 _ { \pm 0 . 1 5 }$ </td><td> $4 2 . 2 9 _ { \pm 1 . 1 4 }$ </td><td> $6 1 . 5 0 { \scriptstyle \pm 2 . 7 8 }$ </td><td> $5 2 . 5 4 { \scriptstyle \pm 0 . 9 6 }$ </td></tr><tr><td>Qwen3.5-122B-A10B</td><td> $2 5 . 3 8 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $8 4 . 6 2 _ { \pm 1 . 3 3 }$ </td><td> $2 7 . 6 9 _ { \pm 1 . 5 4 }$ </td><td> $5 6 . 1 5 _ { \pm 0 . 2 6 }$ </td><td> $4 8 . 2 1 \scriptstyle \pm 0 . 5 1$ </td><td> $5 0 . 2 3 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $4 9 . 1 4 { \scriptstyle \pm 1 . 1 4 }$ </td><td> $6 8 . 0 0 { \scriptstyle \pm 2 . 1 8 }$ </td><td> $5 9 . 2 0 { \scriptstyle \pm 1 . 2 2 }$ </td></tr><tr><td>DeepSeek-V4-Flash</td><td> $4 5 . 7 7 { \scriptstyle \pm 2 . 3 1 }$ </td><td> $7 0 . 7 7 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $6 . 6 7 \pm 1 . 3 6$ </td><td> $4 7 . 9 5 _ { \pm 0 . 7 7 }$ </td><td> $3 1 . 7 9 _ { \pm 2 . 2 4 }$ </td><td> $4 3 . 4 6 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $4 0 . 5 7 _ { \pm 0 . 9 9 }$ </td><td> $5 7 . 5 0 _ { \pm 3 . 4 6 }$ </td><td> $4 9 . 6 0 { \scriptstyle \pm 2 . 3 1 }$ </td></tr><tr><td>Qwen3.5-397B-A17B</td><td> $3 6 . 1 5 _ { \pm 1 . 0 2 }$ </td><td> $8 8 . 8 5 _ { \pm 0 . 6 7 }$ </td><td> $2 9 . 7 4 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $5 7 . 9 5 _ { \pm 0 . 9 2 }$ </td><td> $5 5 . 9 0 _ { \pm 3 . 5 5 }$ </td><td> $5 5 . 2 3 _ { \pm 0 . 7 7 }$ </td><td> $5 0 . 8 6 _ { \pm 0 . 5 7 }$ </td><td> $7 5 . 0 0 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $6 3 . 7 3 { \scriptstyle \pm 0 . 8 0 }$ </td></tr><tr><td>GLM-5</td><td> $3 2 . 3 1 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $8 7 . 3 1 { \scriptstyle \pm 2 . 1 4 }$ </td><td> $2 1 . 0 3 { \scriptstyle \pm 1 . 8 5 }$ </td><td> $5 7 . 1 8 _ { \pm 0 . 4 4 }$ </td><td> $4 4 . 6 2 \substack { \pm 2 . 2 4 }$ </td><td> $5 0 . 9 3 _ { \pm 1 . 0 2 }$ </td><td> $4 8 . 5 7 _ { \pm 0 . 0 0 }$ </td><td> $7 0 . 5 0 { \scriptstyle \pm 2 . 1 8 }$ </td><td> $6 0 . 2 7 _ { \pm 1 . 1 6 }$ </td></tr><tr><td colspan="10">Single-Agent Routing</td></tr><tr><td>RouteLLM [36]</td><td> $2 3 . 8 5 _ { \pm 3 . 0 0 }$ </td><td> $8 5 . 0 0 { \scriptstyle \pm 1 . 3 3 }$ </td><td> $3 7 . 4 4 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $5 9 . 2 3 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $4 4 . 6 2 \substack { \pm 1 . 3 6 }$ </td><td> $5 1 . 8 5 _ { \pm 0 . 5 4 }$ </td><td> $2 5 . 1 4 _ { \pm 1 . 1 4 }$ </td><td> $6 8 . 5 0 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $4 8 . 2 7 _ { \pm 0 . 9 2 }$ </td></tr><tr><td>GraphRouter [12]</td><td> $4 0 . 0 0 { \scriptstyle \pm 1 . 6 8 }$ </td><td> $8 7 . 3 1 { \scriptstyle \pm 2 . 6 6 }$ </td><td> $5 0 . 2 6 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $5 7 . 4 4 _ { \pm 1 . 0 3 }$ </td><td> $5 1 . 7 9 _ { \pm 1 . 8 5 }$ </td><td> $5 8 . 0 0 { \scriptstyle \pm 1 . 4 0 }$ </td><td> $3 3 . 7 1 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $7 3 . 0 0 { \scriptstyle \pm 2 . 6 0 }$ </td><td> $5 4 . 6 6 { \scriptstyle \pm 1 . 1 6 }$ </td></tr><tr><td>ICL-Router [47]</td><td> $3 5 . 0 0 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $8 7 . 6 9 _ { \pm 0 . 6 7 }$ </td><td> $5 0 . 7 7 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $5 4 . 1 0 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $4 6 . 1 5 _ { \pm 2 . 3 5 }$ </td><td> $5 5 . 3 1 _ { \pm 0 . 3 5 }$ </td><td> $3 5 . 4 3 _ { \pm 0 . 0 0 }$ </td><td> $6 7 . 5 0 { \scriptstyle \pm 3 . 6 1 }$ </td><td> $5 2 . 5 3 { \scriptstyle \pm 1 . 9 2 }$ </td></tr><tr><td>RouterDC [4]</td><td> $3 8 . 4 6 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $8 6 . 1 5 _ { \pm 1 . 3 3 }$ </td><td> $4 5 . 6 4 _ { \pm 0 . 5 1 }$ </td><td> $5 6 . 9 2 _ { \pm 0 . 6 8 }$ </td><td> $5 3 . 3 3 { \scriptstyle \pm 3 . 2 0 }$ </td><td> $5 6 . 8 4 _ { \pm 0 . 4 0 }$ </td><td> $2 3 . 4 3 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $7 3 . 0 0 _ { \pm 2 . 1 8 }$ </td><td> $4 9 . 8 7 _ { \pm 0 . 9 6 }$ </td></tr><tr><td colspan="10">Single-Agent Multi-Turn Routing</td></tr><tr><td>Router-R1 [61]</td><td> $4 6 . 5 4 { \scriptstyle \pm 2 . 1 4 }$ </td><td> $5 0 . 3 8 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $1 1 . 7 9 _ { \pm 0 . 5 1 }$ </td><td> $3 7 . 9 5 _ { \pm 0 . 4 4 }$ </td><td> $2 4 . 1 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 6 . 1 5 _ { \pm 0 . 0 8 }$ </td><td> $2 4 . 0 0 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $5 2 . 0 0 _ { \pm 0 . 8 7 }$ </td><td> $3 8 . 9 3 { \scriptstyle \pm 0 . 5 3 }$ </td></tr><tr><td>ReCal [56]</td><td> $4 0 . 3 8 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $8 0 . 0 0 { \scriptstyle \pm 2 . 0 0 }$ </td><td> $2 1 . 5 4 { \scriptstyle \pm 1 . 8 5 }$ </td><td> $3 8 . 7 2 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $3 7 . 9 5 _ { \pm 2 . 3 5 }$ </td><td> $4 4 . 6 2 _ { \pm 0 . 1 3 }$ </td><td> $2 6 . 8 5 _ { \pm 0 . 5 7 }$ </td><td> $5 8 . 5 0 _ { \pm 3 . 4 6 }$ </td><td> $4 3 . 7 3 _ { \pm 1 . 8 7 }$ </td></tr><tr><td>xRouter [39]</td><td> $3 6 . 9 2 \substack { \pm 2 . 5 2 }$ </td><td> $7 1 . 1 5 _ { \pm 4 . 4 4 }$ </td><td> $3 8 . 9 7 _ { \pm 0 . 5 1 }$ </td><td> $3 7 . 6 9 _ { \pm 1 . 1 8 }$ </td><td> $4 6 . 6 7 _ { \pm 4 . 8 9 }$ </td><td> $4 5 . 7 7 _ { \pm 1 . 4 0 }$ </td><td> $4 6 . 2 9 _ { \pm 0 . 9 9 }$ </td><td> $8 1 . 0 0 \substack { \pm 4 . 2 7 }$ </td><td> $6 4 . 8 0 { \scriptstyle \pm 2 . 7 1 }$ </td></tr><tr><td colspan="10">Multi-Agent Routing</td></tr><tr><td>MAS-Orchestra [25]</td><td> $4 6 . 5 4 _ { \pm 2 . 6 6 }$ </td><td> $\underline { { 8 9 . 6 2 } } { \scriptstyle \pm 1 . 3 9 }$ </td><td> $3 1 . 2 8 { \scriptstyle \pm 1 . 8 5 }$ </td><td> $4 1 . 0 3 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $5 8 . 9 7 _ { \pm 2 . 2 4 }$ </td><td> $5 3 . 0 8 _ { \pm 0 . 3 1 }$ </td><td> $4 4 . 5 7 _ { \pm 1 . 1 4 }$ </td><td> $8 6 . 5 0 { \scriptstyle \pm 2 . 1 8 }$ </td><td> $6 6 . 9 3 _ { \pm 1 . 3 3 }$ </td></tr><tr><td>OrchMAS [13]</td><td> $4 1 . 1 5 _ { \pm 2 . 3 4 }$ </td><td> $8 5 . 3 8 \substack { \pm 0 . 7 7 }$ </td><td> $3 1 . 2 8 _ { \pm 0 . 5 1 }$ </td><td> $3 7 . 4 4 _ { \pm 0 . 7 7 }$ </td><td> $3 1 . 7 9 _ { \pm 3 . 0 8 }$ </td><td> $4 6 . 0 0 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $2 6 . 2 8 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $6 8 . 5 0 { \scriptstyle \pm 4 . 2 7 }$ </td><td> $4 8 . 8 0 _ { \pm 2 . 4 0 }$ </td></tr><tr><td>DAAO [44]</td><td> $5 8 . 4 6 { \scriptstyle \pm 3 . 1 5 }$ </td><td> $8 3 . 0 8 _ { \pm 0 . 6 7 }$ </td><td> $3 2 . 3 1 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $5 5 . 9 0 _ { \pm 0 . 4 4 }$ </td><td> $2 3 . 0 8 _ { \pm 0 . 5 1 }$ </td><td> $5 3 . 3 9 _ { \pm 0 . 6 2 }$ </td><td> $2 9 . 1 4 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $7 6 . 5 0 { \scriptstyle \pm 2 . 1 8 }$ </td><td> $5 4 . 4 0 { \scriptstyle \pm 0 . 9 2 }$ </td></tr><tr><td>MasRouter [59]</td><td> $4 8 . 0 8 _ { \pm 3 . 9 0 }$ </td><td> $8 8 . 0 8 { \scriptstyle \pm 2 . 5 2 }$ </td><td> $5 6 . 4 1 { \scriptstyle \pm 1 . 8 5 }$ </td><td> $6 0 . 7 7 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $5 0 . 2 6 { \scriptstyle \pm 1 . 0 3 }$ </td><td> $6 1 . 4 6 { \scriptstyle \pm 1 . 5 0 }$ </td><td> $3 4 . 8 6 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $8 4 . 5 0 _ { \pm 2 . 1 8 }$ </td><td> $6 1 . 3 3 { \scriptstyle \pm 1 . 1 6 }$ </td></tr><tr><td>RCR-Router [32]</td><td> $5 7 . 3 1 { \scriptstyle \pm 2 . 0 4 }$ </td><td> $\mathbf { 9 0 . 3 8 _ { \pm 0 . 7 7 } }$ </td><td> $2 9 . 2 3 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $5 5 . 9 0 _ { \pm 1 . 1 8 }$ </td><td> $7 1 . 7 9 _ { \pm 1 . 8 5 }$ </td><td> $6 1 . 4 6 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $4 1 . 1 4 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $8 6 . 5 0 { \scriptstyle \pm 3 . 9 7 }$ </td><td> $6 5 . 3 3 { \scriptstyle \pm 2 . 3 2 }$ </td></tr><tr><td>AgentRouter [62]</td><td> $4 2 . 3 1 { \scriptstyle \pm 2 . 1 4 }$ </td><td> $\mathbf { 9 0 . 3 8 } _ { \pm 2 . 0 4 }$ </td><td> $4 4 . 1 0 { \scriptstyle \pm 1 . 0 3 }$ </td><td> $5 8 . 9 7 _ { \pm 1 . 3 6 }$ </td><td> $5 6 . 4 1 { \scriptstyle \pm 2 . 6 6 }$ </td><td> $5 9 . 3 1 _ { \pm 0 . 2 8 }$ </td><td> $2 8 . 5 7 _ { \pm 0 . 0 0 }$ </td><td> $7 9 . 0 0 _ { \pm 0 . 8 7 }$ </td><td> $5 5 . 4 7 _ { \pm 0 . 4 6 }$ </td></tr><tr><td>Uno-Orchestra [7]</td><td> $3 4 . 2 3 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $8 5 . 0 0 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $3 4 . 3 6 \substack { \pm 0 . 8 9 }$ </td><td> $4 6 . 1 5 _ { \pm 0 . 6 8 }$ </td><td> $5 2 . 8 2 _ { \pm 0 . 0 0 }$ </td><td> $5 0 . 7 7 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $4 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $7 3 . 0 0 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $5 7 . 6 0 _ { \pm 0 . 7 1 }$ </td></tr><tr><td colspan="10">EvoRouter (Ours)</td></tr><tr><td>EvoRouter-4B</td><td> $6 1 . 1 5 _ { \pm 2 . 0 4 }$ </td><td> $8 8 . 4 6 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $6 4 . 1 0 { \scriptstyle \pm 2 . 7 1 }$ </td><td> $6 1 . 0 3 { \scriptstyle \pm 1 . 5 6 }$ </td><td> $7 2 . 3 1 { \scriptstyle \pm 2 . 3 5 }$ </td><td> $6 8 . 6 9 _ { \pm 0 . 9 5 }$ </td><td> $5 2 . 5 7 _ { \pm 0 . 5 7 }$ </td><td> $9 4 . 0 0 { \scriptstyle \pm 1 . 5 0 }$ </td><td> $7 4 . 6 7 _ { \pm 0 . 9 6 }$ </td></tr><tr><td>EvoRouter-8B</td><td> ${ \bf 6 4 . 6 2 } _ { \pm 1 . 0 2 }$ </td><td> $8 9 . 2 3 _ { \pm 0 . 2 2 }$ </td><td> $\underline { { 6 9 . 2 3 } } { \scriptstyle \pm 0 . 5 1 }$ </td><td> $\mathbf { 6 3 . 5 9 } _ { \pm 1 . 0 7 }$ </td><td> $\underline { { 7 3 . 3 3 } } { \scriptstyle \pm 2 . 0 5 }$ </td><td> $7 1 . 2 3 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $\underline { { 5 4 . 2 9 _ { \pm 1 . 4 4 } } }$ </td><td> $\mathbf { 9 6 . 5 0 } _ { \pm 1 . 2 6 }$ </td><td> $\underline { { 7 6 . 8 0 \pm 0 . 6 7 } }$ </td></tr><tr><td> $\mathbf { E v o R o u t e r – L L a M a - 8 B }$ </td><td> $\underline { { 6 3 . 4 6 \substack { \pm 1 . 9 4 } } }$ </td><td> $8 9 . 2 3 _ { \pm 2 . 6 2 }$ </td><td> $\mathbf { 6 9 . 7 4 } \substack { \pm 1 . 8 5 }$ </td><td> $\underline { { 6 2 . 0 5 } } { \scriptstyle \pm 1 . 2 9 }$ </td><td> ${ \bf 7 6 . 9 2 _ { \pm 2 . 0 7 } }$ </td><td> $\underline { { 7 1 . 1 5 } } { \scriptstyle \pm 0 . 8 9 }$ </td><td> ${ \bf 5 9 . 4 3 } _ { \pm 0 . 4 0 }$ </td><td> $\underline { { 9 6 . 0 0 { \scriptstyle \pm 1 . 7 7 } } }$ </td><td> $7 8 . 9 3 \substack { \pm 1 . 1 3 }$ </td></tr></table>

(10)

where � controls the contribution of the analysis tokens.

(9)

where $\lambda _ { D }$ and � are the coeficients for distillation and KL regularization, respectively. The updated policy �<sub>�</sub> continues to generate

$$
\mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \frac { \sum _ { i } \sum _ { t \in \mathcal { T } _ { i } ^ { \mathrm { p l a n } } } D _ { i , t } } { \sum _ { i } | \mathcal { T } _ { i } ^ { \mathrm { p l a n } } | } + \alpha \frac { \sum _ { i } \sum _ { t \in \mathcal { T } _ { i } ^ { \mathrm { a n a } } } D _ { i , t } } { \sum _ { i } | \mathcal { T } _ { i } ^ { \mathrm { a n a } } | } ,
$$

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathcal { L } _ { \mathrm { R L } } ( \theta ) + \lambda _ { D } \mathcal { L } _ { \mathrm { O P D } } ( \theta ) + \beta \mathcal { L } _ { \mathrm { K L } } ( \theta ) , } \end{array}
$$

The final joint training objective is defined as:

student and teacher, respectively. Following MiniLLM [15], we minimize the reverse KL divergence over the vocabulary Ω shared by the teacher and student:

$$
D _ { i , t } = D _ { \mathrm { K L } } ( \boldsymbol { \mathcal { P } } _ { i , t } ^ { \mathrm { S } }  \mathrm { s g } [ \boldsymbol { \mathcal { P } } _ { i , t } ^ { \mathrm { T } } ] ) = \sum _ { \alpha \in \Omega } \boldsymbol { \mathcal { P } } _ { i , t } ^ { \mathrm { S } } ( \boldsymbol { a } ) \log \frac { \boldsymbol { \mathcal { P } } _ { i , t } ^ { \mathrm { S } } ( \boldsymbol { a } ) } { \mathrm { s g } [ \boldsymbol { \mathcal { P } } _ { i , t } ^ { \mathrm { T } } ( \boldsymbol { a } ) ] } ,\tag{8}
$$

Insight internalization and joint optimization. Distillation is applied to the initial routing response of each trajectory. Let $\mathcal { T } _ { i } ^ { \mathrm { a n a } }$ and $\mathcal { T } _ { i } ^ { \mathrm { p l a n } }$ denote the positions of student-generated content tokens within the <analysis> and <plan> tags, respectively, excluding formatting and environment tokens. The OPD objective minimizes their weighted distillation losses:

where $D _ { \mathrm { K L } } ( )$ denotes the KL divergence, and sg[·] denotes the stop-gradient operator.

execution trajectories that inform subsequent insight refinement, closing the recursive feedback loop among execution, insight synthesis, and policy internalization.

## 5 Experiments

## 5.1 Experimental Setup

Datasets and metrics. We evaluate EvoRouter across 21 datasets in seven domains: math (MATH [17], GSM8K [6], Omni-MATH-2 [3], Apex Shortlist [8]), commonsense (ARC-Challenge [5], StrategyQA [14], Social IQA [43]), knowledge (MMLU-Pro [51], GPQA-Diamond [42], SuperGPQA [9]), retrieval (NQ [26], TriviaQA [24], HotpotQA [55], 2WikiMultiHopQA [18], PopQA [33], MuSiQue [46]), multimodal (ChartQAPro [34], MMMU [57], MMMU-Pro [58]), alongside two out-of-domain tasks: code generation (LiveCodeBench [22]) and logical reasoning (BBH-LD [45]). Table 2 reports verifier-based accuracy detailed in Section 3.2. We measure the per-question computational cost of each MAS using the relative FLOPs metric defined in Appendix B.

Routing resources. The available LLM pool comprises Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, DeepSeek-V4-Flash, and GLM-5. The collaboration mode pool includes IO, CoT, Reflection, Chain, FullConnected, and Debate. Each task category is additionally equipped with at least six relevant skills, such as code verification for coding tasks and information synthesis for QA. All routing methods share the same resources and are provided with identical descriptions of available LLMs, collaboration modes, and skills wherever applicable.

![](images/348c229ca3d95427c8f8d4c2847e0d76b39a536e0e7fad9a7988aea086f19dfe.jpg)

![](images/da532d23d00bb1561b3e5ede766146442149f22fb1b921413f9cc8011976842c.jpg)  
Figure 5: Left: The Pareto frontier of the evaluated baselines; Right: Mean total LLM inference cost, including router calls.

Baselines. We compare five categories of methods. (I) Vanilla LLMs: Direct inference using individual models from the LLM pool. (II) Single-Agent Routing: RouteLLM [36], GraphRouter [12], ICL-Router [47], and RouterDC [4]. (III) Single-Agent Multi-Turn Routing: Router-R1 [61], ReCal [56], and xRouter [39]. (IV) Multi-Agent Routing: MAS-Orchestra [25], OrchMAS [13], DAAO [44], MasRouter [59], RCR-Router [32], AgentRouter [62], and Uno-Orchestra [7]. (V) Ours: EvoRouter-4B, EvoRouter-8B, and EvoRouter-LLaMa-8B. Further details on the baselines are provided in Appendix A.4.

## 5.2 Performance Analysis

Accuracy and cross-domain generalization. As shown in Table 2, all EvoRouter variants outperform competing baselines across four of the five in-domain capability categories, achieving average accuracies of 68.69% for EvoRouter-4B and 71.23% for EvoRouter-8B. This advantage extends to unseen Code and Reasoning domains, where EvoRouter achieves sample-weighted accuracies of 74.67% to 78.93%, surpassing the strongest baseline by 7.74 to 12.00 percentage points. These consistent gains highlight the trans ferability of EvoRouter’s learned orchestration strategies across diverse capability domains.

Compute eficiency. The left panel of Figure 5 highlights EvoRouter’s favorable position on the accuracy–cost Pareto frontier. EvoRouter-4B achieves 7.23 percentage points higher accuracy than the strongest baseline RCR-Router while reducing mean MAS cost by 50.95% to 15.28k FLOPs units. Accounting for router overhead, the right panel shows that total inference cost reaches only 18.94k units, remaining 76.5% and 39.2% below MasRouter and RCR-Router, respectively. This confirms that substantial worker-level compute savings translate into end-to-end eficiency gains despite the additional planning overhead.

Table 3: Component ablations of EvoRouter-4B.
<table><tr><td rowspan="2">Training Stage</td><td colspan="2">In-Domain</td><td colspan="2">Out-of-Domain</td></tr><tr><td>Accuracy</td><td>Cost</td><td>Accuracy</td><td>Cost</td></tr><tr><td>EvoRouter-4B</td><td>68.69±0.95</td><td>15280.79±770.31</td><td>74.67±0.96</td><td>25952.47±2673.56</td></tr><tr><td>w/o Insights</td><td>64.54±1.52</td><td>18245.46±977.52</td><td>70.93±2.44</td><td>25453.12±2075.82</td></tr><tr><td>w/o OPD</td><td>63.23±1.31</td><td>16467.21±852.09</td><td>69.87±1.41</td><td>24419.74±2017.43</td></tr><tr><td>w/o RL+OPD</td><td>59.77±0.38</td><td>16789.73±691.28</td><td>63.20±1.16</td><td>30158.45±2555.80</td></tr><tr><td>w/o SFT+RL+OPD</td><td>53.23±0.23</td><td>25472.25±1278.80</td><td>52.80±2.40</td><td>19795.09±1654.15</td></tr></table>

## 5.3 Framework Analysis

Ablation study. Table 3 quantifies the contributions ofRL, OPD, and distilled Insights. Compared with SFT alone, full EvoRouter-4B improves in-domain and out-of-domain accuracy by 8.92 and 11.47 percentage points while reducing MAS execution costs by 9.0% and 13.9%, respectively. Under matched training iterations, OPD without Insights yields only 1.31- and 1.06-point accuracy gains over RL, whereas insight-guided OPD boosts these gains to 5.46 and 4.80 points. This substantial gap highlights the importance of distilling cross-trajectory insights into actionable supervision for efective routing.

Evolution dynamics. Figure 6 illustrates how iterative insightguided policy refinement progressively improves EvoRouter-LLaMa-8B’s performance. Early updates improve success rate by 2.62 points while reducing compute by 16.4%, reflecting more eficient resource selection; Subsequent updates leverage additional compute for further accuracy gains, before later refinements achieve another 1.35-point improvement with 7.4% lower cost. These dynamics demonstrate how insight-guided evolution progressively refines the trade-of between accuracy and computational eficiency. Training loss curves for all three backbones are provided in Appendix A.3.

Sensitivity study. We independently train EvoRouter-4B across $\lambda _ { C } \in \{ 0 , 0 . 0 2 5 , 0 . 0 5 , 0 . 1 , 0 . 2 \}$ . As shown in Figure 7, higher $\lambda _ { C }$ steers the policy toward lower-cost configurations, with varying accuracy sensitivity across domains. Furthermore, Figure 8 reveals that increasing cost penalties progressively shifts the per-question compute distribution toward lower-budget regimes while substantially suppressing the expensive upper tail, ensuring predictable workload expenditure.

![](images/dc12d69a0bec10469887abc0f74d5b16a00a1237b5bdce0c8ad2c10455cd7e44.jpg)  
Figure 6: Training dynamics of EvoRouter-LLaMa-8B.

![](images/3478f667c6264bbfb6dfb191fa7ea52d7435b2fac934e4c099a0575e3e74473a.jpg)  
Figure 7: Sensitivity of EvoRouter-4B to coeficient $\lambda _ { C } .$

![](images/786b572be82775ad235bce4b94a4c3bbe36dc2ea47d4021a0610073501ca61d2.jpg)  
Figure 8: Per-question MAS cost distributions of EvoRouter-4B under diferent cost penalties.

## 5.4 Generalization Analysis

Beyond the two out-of-domain datasets reported in Table 2, we further evaluate zero-shot generalization to unseen routing resources by simultaneously replacing the LLM, collaboration mode, and skill pools while keeping the trained router fixed. Replacement resources are provided in Appendix A.1. As shown in Figure 9, EvoRouter-4B consistently achieves the highest accuracy, outperforming Mas-Router by 3.85 and 7.47 percentage points on in-domain and outof-domain tasks while reducing MAS compute by 6.7% and 18.5%, respectively. Notably, it also surpasses the strongest out-of-domain baseline by 5.9 points with less than half its compute expenditure. Compared with the independently trained EvoRouter-4B w/o Insights variant, the full model achieves 5.77 and 11.47 percentage points higher accuracy, with the larger out-of-domain gain highlighting the role of insight-guided training in transferring orchestration knowledge across unseen resources and task distributions.

![](images/997576c8b576597d6d2ad051c4f5f303e56ccc3260818196110ae8651fde4128.jpg)  
Figure 9: Generalization to unseen routing resources.

## 5.5 Case Study

To better illustrate the routing mechanism of EvoRouter, we present a representative case on mathematical reasoning in Figure 10, demonstrating how it achieves accurate and cost-efective MAS routing. An illustrative example of the distilled routing insights is provided in Figure 11.

## 6 Conclusion

We presented EvoRouter, a framework that learns to jointly configure heterogeneous LLM-based MAS from execution experience. Combining cost-aware reinforcement learning with on-policy distillation, EvoRouter internalizes cross-task configuration insights and trajectory feedback into its routing policy, enabling deployment without privileged feedback. Across 21 datasets spanning seven domains, EvoRouter achieves higher average accuracy at lower computational cost than the evaluated routing baselines in both in-domain and out-of-domain settings. Our current framework is designed for cloud-based agent services. Future work will extend EvoRouter to resource-constrained edge environments through coordinated cloud–edge routing that accounts for device heterogeneity and inter-agent communication overhead.

## References

[1] Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin El-Nouby, Joshua M. Susskind, and Vimal Thilak. 2025. Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models. In Proceedings ofthe 42nd International Conference on Machine Learning, Vol. 267. PMLR, 204–230. https://proceedings.mlr.press/v267/abnar25a.html

[2] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. 2024. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. In International

![](images/c2077f1281d82a89fe9a265ff971df5db7e392416cd19a47a3c50b52c9beb1cb.jpg)

## Figure 10: A practical case of EvoRouter.

Conference on Learning Representations, Vol. 2024. 21246–21263. https://openreview.net/forum?id=3zKtaqxLhW

[3] Marthe Ballon, Andres Algaba, Brecht F. Verbeken, and Vincent Ginis. 2026. Benchmarks Saturate When The Model Gets Smarter Than The Judge. arXiv:2601.19532 https://arxiv.org/abs/2601.19532

[4] Shuhao Chen, Weisen Jiang, Baijiong Lin, James T. Kwok, and Yu Zhang. 2024. RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models. In Advances in Neural Information Processing Systems, Vol. 37. 66305–66328. doi:10.52202/079017-2120

[5] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457 https://arxiv.org/abs/1803.05457

[6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168 https://arxiv.org/abs/2110.14168

[7] Zhiqing Cui, Haotong Xie, Jiahao Yuan, Cheng Yang, Hanqing Wang, Yuxin Wu, Yifan Wu, Siru Zhong, Tao Yu, Yifu Guo, Siyu Zhang, Xinlei Yu, Qibing Ren, and Usman Naseem. 2026. Uno-Orchestra: Parsimonious Agent Routing via Selective Delegation. arXiv:2605.05007 [cs.AI] https://arxiv.org/abs/2605.05007

[8] Jasper Dekoninck, Nikola Jovanović, Tim Gehrunger, Kári Rögnvaldsson, Ivo Petrov, Chenhao Sun, and Martin Vechev. 2026. Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs. In 3rd AIfor Math Workshop (AI4Math) at ICML 2026. https://openreview.net/forum?id=DmPE4byHuN

[9] Xeron Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, et al. 2025. SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines. In Advances in Neural Information Processing Systems, Vol. 38. 125016–125089. doi:10.52202/085713-3766

[10] Reza Farahani, Zoha Azimi, Mario Colosi, and Schahram Dustdar. 2026. LMEdge: QoS-Aware LLM Inference Orchestration on Edge Clusters. In 4th International Workshop on Scalable Compute Continuum (WSCC 2026). https://arxiv.org/abs/2607.17175 Co-located with Euro-Par 2026.

[11] Reza Farahani, Zoha Azimi Ourimi, Mario Colosi, Lauri Lovén, Christian Timmerer, and Schahram Dustdar. 2026. DRLM: Deep Reinforcement Learning-Based LLM Query Orchestration in Edge Environments. In 2026 IEEE Global Communications Conference (GLOBECOM) https://arxiv.org/abs/2609.00442 Accepted; forthcoming.

[12] Tao Feng, Yanzhen Shen, and Jiaxuan You. 2025. GraphRouter: A Graph-based

Router for LLM Selections. In The Thirteenth International Conference on Learning Representations. 26186–26203 https://openreview.net/forum?id=eU39PDsZtT

[13] Yichao Feng, Haoran Luo, Zhenghong Lin, Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh, and Anh Tuan Luu. 2026. OrchMAS: Orchestrated Reasoning with Multi Collaborative Heterogeneous Scientific Expert Structured Agents. arXiv:2603.03005 [cs.AI] https://arxiv.org/abs/2603.03005

[14] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. Transactions of the Association for Computational Linguistics 9 (2021), 346–361. doi:10.1162/tacl\_a\_00370

[15] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2024. MiniLLM: Knowledge Distillation of Large Language Models. In International Conference on Learning Representations, Vol. 2024. 32694–32717. https://openreview.net/forum?id=5h0qf7IBZZ

[16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature 645, 8081 (2025), 633–638. doi:10.1038/s41586-025-09422-z

[17] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring Mathematical Problem Solving With the MATH Dataset. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks, Vol. 1. https://openreview.net/forum?id=7Bywt2mQsCe

[18] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In Proceedings of the 28th International Conference on Computational Linguistics. International Committee on Computational Linguistics, 6609–6625. doi:10.18653/v1/2020.coling-main.580

[19] Jiatan Huang, Zheyuan Zhang, Kaiwen Shi, Yanfang Ye, and Chuxu Zhang. 2026. EvolveRouter: Co-Evolving Routing and Prompt for Multi-Agent Question Answering. arXiv:2604.05149 [cs.CL] https://arxiv.org/abs/2604.05149

[20] Jonas Hübotter, Frederike Lübeck, Lejs Deen Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. 2026. Reinforcement Learning via Self-Distillation. In Proceedings ofthe 43rd International Conference on Machine Learning, Vol. 306. PMLR, 48677–48721. https://proceedings.mlr.press/v306/hubotter26a.html

[21] Adriano Izzi, Jerin George Mathew, Flavia Monti, Donatella Firmani, Francesco Leotta, Federica Mandreoli, and Massimo Mecella. 2025. Data Service Composition in Cyber-Physical Systems Adopting LLMs. In 2025 IEEE International Conference on Web Services (ICWS). IEEE, 625–636. doi:10.1109/ICWS67624.2025.00085

[22] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida I. Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=chfJJYC3iL

[23] Yinsicheng Jiang, Yao Fu, Yeqi Huang, Ping Nie, Zhan Lu, Leyang Xue, Congjie He, Man-Kit Sit, Jilong Xue, Li Dong, Ziming Miao, DaYou Du, Tairan Xu, Kai Zou, Edoardo Maria Ponti, and Luo Mai. 2025. MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems. In Advances in Neural Information Processing Systems, Vol. 38. 90197–90223. doi:10.52202/085713-271

[24] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1601–1611. doi:10.18653/v1/P17-1147

[25] Zixuan Ke, Yifei Ming, Austin Xu, Ryan Chin, Xuan-Phi Nguyen, Prathyusha Jwalapuram, Jiayu Wang, Semih Yavuz, Caiming Xiong, and Shafiq Joty. 2026. MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks. In Proceedings of the 43rd International Conference on Machine Learning, Vol. 306. PMLR, 56527–56570. https://proceedings.mlr.press/v306/ke26c.html

[26] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. Transactions ofthe Association for Computational Linguistics 7 (2019), 452–466. doi:10.1162/tacl\_a\_00276

[27] Hao Li, Hangfan Zhang, Zhiyao Cui, Chunjiang Mu, Yiqun Zhang, Bo Zhang, Danyang Jia, and Shuyue Hu. 2026. RSI-Router: Evolving Subtask-Level LLM Routing and Skills for Cost-Eficient Agents. arXiv:2609.34712 https://arxiv.org/abs/2609.34712

[28] Songyuan Li, Ahmed M. Abdelmoniem, and Shiqiang Wang. 2026. ProgRouter: Online Progress-Guided Orchestration for Multi-Agent LLM Workflows under

Quality-Cost Tradeofs. In Findings of the Association for Computational Linguistics: EMNLP 2026. 3272–3287. https://aclanthology.org/2026.findings-emnlp.156/

[29] Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, and Zhicheng Dou. 2026. DeepAgent: A General Reasoning Agent with Scalable Toolsets. In Proceedings of the ACM Web Conference 2026. Association for Computing Machinery, 2219–2230. doi:10.1145/3774904.3792460

[30] Siteng Liao, Yuzhu Liang, Hengzhong Rao, Xizhao Luo, and Tian Wang. 2026. AlphaRouter: Token-level Routing Between SLM and LLM with Reinforcement Learning and Tree Search. In Proceedings ofthe 43rd International Conference on Machine Learning, Vol. 306. PMLR, 73805–73824. https://proceedings.mlr.press/v306/liao26c.html

[31] Hehai Lin, Qi Yang, and Chengwei Qin. 2026. Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems. In Findings ofthe Association for Computational Linguistics: EMNLP 2026. 33685–33718. https://aclanthology.org/2026.findings-emnlp.1582/

[32] Jun Liu, Pu Zhao, Zhenglun Kong, Wenbin Zhang, Peiyan Dong, Fan Yang, Changdi Yang, Wuzhe Fan, Tianqin Li, Hao Tang, Geng Yuan, Wei Niu, Joannah Nanjekye, Xue Lin, Dong Huang, and Yanzhi Wang. 2026. RCR-Router: Role-Conditioned Context Routing for Resource-Eficient Multi-Agent LLM Systems. In Findings ofthe Association for Computational Linguistics: EMNLP 2026. 8938–8963. https://aclanthology.org/2026.findings-emnlp.425

[33] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating Efectiveness of Parametric and Non-Parametric Memories. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 9802–9822. doi:10.18653/v1/2023.acl-long.546

[34] Ahmed Masry, Mohammed Saidul Islam, Mahir Ahmed, Aayush Bajaj, Firoz Kabir, Aaryaman Kartha, Md Tahmid Rahman Laskar, Mizanur Rahman, Shadikur Rahman, Mehrad Shahmohammadi, Megh Thakkar, Md Rizwan Parvez, Enamul Hoque, and Shafiq Joty. 2025. ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering. In Findings ofthe Association for Computational Linguistics: ACL 2025. 19123–19151. doi:10.18653/v1/2025.findings-acl.978

[35] Yasmin Moslem and John D Kelleher. 2026. Dynamic model routing and cascading for eficient LLM inference: A survey. arXiv preprint arXiv:2603.04445 (2026).

[36] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to Route LLMs from Preference Data. In The Thirteenth International Conference on Learning Representations. 34433–34448. https://openreview.net/forum?id=8sSqNntaMr

[37] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, Vol. 35. 27730–27744. doi:10.52202/068431-2011

[38] Robin D. Pesl, Jerin G. Mathew, Massimo Mecella, and Marco Aiello. 2025. Advanced System Integration: Analyzing OpenAPI Chunking for Retrieval-Augmented Generation. In Advanced Information Systems Engineering (Lecture Notes in Computer Science, Vol. 15702). Springer, 130–148. doi:10.1007/978-3-031-94571-7\_8

[39] Cheng Qian, Zuxin Liu, Shirley Kokane, Akshara Prabhakar, Jielin Qiu, Haolin Chen, Zhiwei Liu, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, and Huan Wang. 2025. xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning. arXiv:2510.08439 [cs.LG] https://arxiv.org/abs/2510.08439

[40] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Advances in Neural Information Processing Systems, Vol. 36. 53728–53741. doi:10.52202/075280-2338

[41] Rakibul Hasan Rajib, Mengxin Zheng, and Qian Lou. 2026. Learning What to Retain: Gated-Memory Routing for Eficient Collaboration in Multi-Agent LLM Systems. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing. 39594–39611. https://aclanthology.org/2026.emnlp-main.1735/

[42] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In First Conference on Language Modeling. https://openreview.net/forum?id=Ti67584b98

[43] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social IQa: Commonsense Reasoning about Social Interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 4463–4473. doi:10.18653/v1/D19-1454

[44] Jinwei Su, Qizhen Lan, Yinghui Xia, Lifan Sun, Weiyou Tian, Tianyu Shi, and Lewei He. 2026. Dificulty-Aware Agentic Orchestration for Query-Specific

Multi-Agent Workflows. In Proceedings of the ACM Web Conference 2026. Association for Computing Machinery, 2060–2070. doi:10.1145/3774904.3792240

[45] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. In Findings of the Association for Computational Linguistics: ACL 2023. 13003–13051. doi:10.18653/v1/2023.findings-acl.824

[46] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions ofthe Association for Computational Linguistics 10 (2022), 539–554. doi:10.1162/tacl\_a\_00475

[47] Chenxu Wang, Hao Li, Yiqun Zhang, Linyao Chen, Jianhao Chen, Ping Jian, Qiaosheng Zhang, and Shuyue Hu. 2026. ICL-Router: In-Context Learned Model Representations for LLM Routing. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 33413–33421. doi:10.1609/aaai.v40i39.40628

[48] Fali Wang, Hui Liu, Zhenwei Dai, Jingying Zeng, Zhiwei Zhang, Zongyu Wu, Chen Luo, Zhen Li, Xianfeng Tang, Qi He, and Suhang Wang. 2025. AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks. In Advances in Neural Information Processing Systems, Vol. 38. 98396–98433. doi:10.52202/085713-3288

[49] Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, and Honggang Qi. 2026. Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents. In The Fortieth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=ZeThM3k7x0 Accepted; forthcoming.

[50] Jiayu Wang, Yifei Ming, Zixuan Ke, Shafiq Joty, Aws Albarghouthi, and Frederic Sala. 2026. SkillOrchestra: Learning to Route Agents via Skill Transfer. In The Fortieth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=5B5fOqwuhG Accepted; forthcoming.

[51] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. In Advances in Neural Information Processing Systems, Vol. 37. 95266–95290. doi:10.52202/079017-3018

[52] Jinyang Wu, Shuo Yang, Zhengxi Lu, Fan Zhang, Yuhao Shen, Lang Feng, Haoran Luo, Zheng Lian, Shuai Zhang, Zhengqi Wen, and Jianhua Tao. 2026 SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning. arXiv:2607.14777 https://arxiv.org/abs/2607.14777

[53] Yitao Xiao, Shaoyong Guo, Guoming Yang, Qingnan Wang, Yinlin Ren, Xuesong Qiu, and Qi Feng. 2026. RouterHGC: Optimized Router for LLM-based Multi-Agent Systems via Heterogeneous Graph Contrastive Learning. In Findings ofthe Association for Computational Linguistics: ACL 2026. 31759–31789. doi:10.18653/v1/2026.findings-acl.1589

[54] Jiaqi Xue, Qian Lou, Jiarong Xing, and Heng Huang. 2026. R2-Router: A New Paradigm for LLM Routing with Reasoning. In Proceedings ofthe 43rd International Conference on Machine Learning, Vol. 306. PMLR, 143255–143268. https://proceedings.mlr.press/v306/xue26h.html

[55] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing. 2369–2380. doi:10.18653/v1/D18-1259

[56] Qihang Yu, Hanwen Tong, Zhengqi Zhang, Bo Zheng, Feng Wei, Shengyu Zhang, Zemin Liu, and Fei Wu. 2026. ReCal: Reward Calibration for RL-based LLM Routing. arXiv:2606.12479 [cs.LG] https://arxiv.org/abs/2606.12479

[57] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2024. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 9556–9567. doi:10.1109/CVPR52733.2024.00913

[58] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. 2025. MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 15134–15186. doi:10.18653/v1/2025.acl-long.736

[59] Yanwei Yue, Guibin Zhang, Boyang Liu, Guancheng Wan, Kun Wang, Dawei Cheng, and Yiyan Qi. 2025. MasRouter: Learning to Route LLMs for Multi-Agent Systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 15549–15572. doi:10.18653/v1/2025.acl-long.757

[60] Guibin Zhang, Haiyang Yu, Kaiming Yang, Bingli Wu, Fei Huang, Yongbin Li, and Shuicheng Yan. 2026. EvoRoute: Experience-Driven Self-Routing LLM Agent Systems. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 38213–38225

doi:10.18653/v1/2026.acl-long.1771

[61] Haozhen Zhang, Tao Feng, and Jiaxuan You. 2025. Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning. In Advances in Neural Information Processing Systems, Vol. 38. 141233–141265. doi:10.52202/085713-4719

[62] Zheyuan Zhang, Kaiwen Shi, Zhengqing Yuan, Zehong Wang, Tianyi Ma, Keerthiram Murugesan, Vincent Galassi, Chuxu Zhang, and Yanfang Ye. 2026. AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 788–809. doi:10.18653/v1/2026.acl-long.33

[63] Di Zhao, Longhui Ma, Siwei Wang, Miao Wang, and Yi Kong. 2026. SC-MAS: Constructing Cost-Eficient Multi-Agent Systems with Edge-Level Heterogeneous Collaboration. arXiv:2601.09434 https://arxiv.org/abs/2601.09434

[64] Qiyong Zhong, Mao Zheng, Mingyang Song, Houcheng Jiang, Jiajie Su, Huwei Ji, Li Zhang, and Junfeng Fang. 2026. MAS-OPD: On-Policy Distillation for Multi-agent Systems. arXiv:2609.34234 https://arxiv.org/abs/2609.34234

## A Experimental Details

## A.1 Resource

All LLMs in the pool use temperature 1.0 and maximum output length 1,024 tokens (4,096 for code tasks). For multimodal tasks, models lacking multimodal support receive a fixed placeholder declaring image invisibility, forcing them to answer solely based on text. In generalization experiments (Section 5.4), the model pool comprises Qwen3.8-Flash, Kimi-K2.8-Preview, and Qwen3.8-Max; the collaboration mode pool comprises Self-Consistency and Hierarchical; and the skill pool contains skills unused elsewhere.

Details of the collaboration modes are as follows:

• IO: A single agent answers in a single turn.

• CoT: A single agent reasons step-by-step across multiple turns before delivering the final answer.

• Reflection: A single agent iteratively reflects on its previous response to generate a revised answer.

• Self-Consistency: A single agent is invoked multiple times independently. A subsequent invocation then inspects all candidate answers, evaluates consistency, resolves discrepancies, and aggregates the final result.

• Chain: Agents are activated sequentially, each receiving its predecessor’s output. When an agent exhausts its allocated reasoning turns, the next agent is triggered.

• FullConnected: All agents are activated concurrently, receiving peers’ responses from the prior turn. Final-turn responses are routed to the router to synthesize the answer.

• Debate: All agents run concurrently, critiquing peers or rebutting critiques in subsequent turns. Final-turn responses are routed to the router for answer synthesis.

• Hierarchical: The initial agent plans task decomposition and scheduling, intermediate agents execute independently in parallel, and the final agent synthesizes findings. Turn limits govern reasoning turns per agent.

Certain collaboration modes impose cross-configuration constraints. For example, if the router pairs CoT with two agents, the MAS plan is rejected (as CoT admits at most one agent) and marked incorrect.

## A.2 Dataset

We construct a dataset comprising 12,800 questions across five domains, as described in Section 5.1. The dataset is partitioned into 10,500 training, 1,000 development, and 1,300 test questions. For out-of-domain evaluation, we use 175 questions from the Live-CodeBench [22] and 200 from the BBH-LD [45].

For each training task domain, we set a shared budget in our cost units. We calibrate it from pre-collected successful traces by computing the 25th percentile of trace costs for each question, taking the median across questions in the domain, and adding a 15% margin, with a minimum budget of 512. Budgets vary across domains (as well as trace collections); for example, the budget for the math tasks is 12,374.00, and 5,722.02 for commonsense tasks.

## A.3 Training

We evaluate three router backbones: Qwen3-4B, Qwen3-8B, and Llama-3.1-8B, yielding EvoRouter-4B, EvoRouter-8B, and EvoRouter-LLaMa-8B, respectively, through our iterative training framework. All experiments run on 8 NVIDIA A100 80GB GPUs. Task sets across SFT, RL, and evaluation are mutually disjoint.

For supervised initialization, DeepSeek-V4-Pro serves as a bootstrap router to collect trajectories, yielding 2,000 filtered instances to fine-tune each backbone for 2 epochs. In subsequent training, we use a batch size of 8 with 16 rollouts per question and a learning rate of $3 \times 1 0 ^ { - 6 }$ , setting $\beta ~ = ~ 0 . 0 0 1 , ~ c _ { \operatorname* { m a x } } ~ = ~ 0 . 1 , ~ \lambda _ { D } ~ = ~ 0 . 0 3$ , and � = 0.25. For on-policy distillation, Qwen3-32B and Llama-3.1-70B serve as teachers for the Qwen and Llama backbones, respectively. DeepSeek-V4-Pro also acts as the insight distillation expert, leveraging its context window and reasoning capabilities to analyze dozens of trajectories simultaneously, but does not directly supervise the student. Finally, we cap insight capacity at $\vert \mathcal { I } \vert \le 8$ and total length at 6,144 characters to limit context overhead.

## Example Insights from EvoRouter’s Evolution Loop

Task dependency. Missing or ambiguous facts need models with retrieval or stronger knowledge; . . .

Diminishing marginal returns. For simple arithmetic, accuracy saturates with one capable worker and one round; . . . Combinatorial dependency. Use Chain for chart calculations, pairing a multimodal extraction worker with a numerical solver consuming verified readings; . . .

Figure 11: Example insights distilled during EvoRouter’s evolution.

## A.4 Evaluation

Across all evaluations, we report accuracy averaged over three independent runs, rounding the mean number of correct predictions to the nearest integer before normalizing by the total number of test instances. For baselines without native model selection, we fix the backbone to GLM-5, the model with the largest active parameter in our pool, to ensure strong reasoning capability across tasks. The baselines evaluated in Table 2 are summarized below:

• Vanilla LLMs: Directly use the corresponding LLM to generate responses in a single-turn interaction.

![](images/48ca3bf9a11bb206942382cf5f8f0b07c99a27fd47e05d62a0ca8accc3d83592.jpg)  
Figure 12: Training losses.

• RouteLLM [36]: A classifier-based binary routing method that chooses between GLM-5 and Qwen3.5-35B-A3B for each query based on the predicted win rate of the stronger model against a fixed threshold.

• GraphRouter [12]: Models relations among tasks, queries, and candidate LLMs with a heterogeneous graph, employing a graph neural network to select the suitable model.

• ICL-Router [47]: Constructs capability representations of candidate models from historical performance and routes each query to the model predicted to be the best.

• RouterDC [4]: A dual contrastive learning based routing approach that selects an LLM based on the similarity between the query and candidate capability vectors.

• Router-R1 [61]: An RL-based agentic routing method where the router actively dispatches sub-tasks to candidate models during reasoning and aggregates returned information to synthesize the final answer.

• ReCal [56]: An RL-based agentic routing method following routing behaviors similar to Router-R1 [61], enhancing policy optimization under multi-task and multi-objective settings via reward decomposition and calibration.

• xRouter [39]: A multi-turn routing method that allows the router to answer directly, invoke candidate models, or select an existing model response as the final answer.

• MAS-Orchestra [25]: Encapsulates diverse collaborative reasoning patterns into callable modules coordinated by an orchestrator, using GLM-5 as the LLM backbone.

• OrchMAS [13]: Dynamically instantiates agent roles and task prompts while advancing problem-solving via runtime agent feedback, adopting GLM-5 as the backbone.

• DAAO [44]: Dynamically selects reasoning operators and their LLM backbones based on predicted problem dificulty, constructing an adaptive hierarchical collaborative workflow for each query.

• MasRouter [59]: Employs cascaded controllers to select collaboration modes, agent counts, roles, and LLM backbones, instantiating tailored MAS per query.

• RCR-Router [32]: Allocates shared memory to agents based on task-role relevance, where all agents are backed by GLM-5 and iteratively analyze and answer queries ac cording to a predefined topology.

• AgentRouter [62]: Employs a knowledge graph to characterize relationships among queries, entities, and candidate <sup>71.06%</sup>agents, using a GNN to compute a weighted aggregation of <sub>O</sub>responses across model-strategy combinations.

<sup>69.24% 70.68%</sup> <sub>F</sub><sup>L</sup>• Uno-Orchestra [7]: An agentic orchestration method S<sub>28,250.54</sub>based on SFT and RL that dispatches subtasks to com-M<sup>26,386.29</sup>patible model-primitive pairs, iteratively delegating or <sup>n</sup>20,683.47synthesizing the answer based on execution feedback.

## <sub>63.85%</sub>B Derivation of Inference Computational Cost

<sup>0</sup> <sup>1</sup> <sup>2</sup> <sup>3</sup> <sup>4</sup> <sup>5</sup> <sup>6</sup> <sup>7</sup> <sup>8</sup> <sup>9</sup> <sup>10</sup>Measuring computational overhead via commercial API pricing <sup>Iteration</sup> <sup>Version</sup>often fails to reflect actual resource consumption, as pricing fluctuates across service providers and over time. For clarity and conciseness, we instead model a standard decoder-only Mixture-of-Experts model � with � layers and hidden dimension �, where each layer activates � equally sized routed experts. Let $P _ { \mathrm { s h } }$ denote the parameter count of the always-active backbone matrices—including attention projections, routers, and shared experts—and $P _ { \mathrm { e x p } }$ denote that of a single routed expert. The active matrix parameter count per token is then $A _ { m } = P _ { \mathrm { { s h } } } + L k P _ { \mathrm { { e x p } } }$ , as unselected experts incur no matrix operations [1, 23].

Counting one multiplication and one addition as one FLOP each, parameter matrix operations incur a dominant cost of $2 A _ { m }$ FLOPs per token. Under standard multi-head causal attention, attending to � positions requires an additional 4��� FLOPs for query-key dot products and attention-weighted value aggregation [48]. For a model invocation with input length $n ^ { \mathrm { i n } }$ , output length $n ^ { \mathrm { o u t } }$ , and total sequence length $N = n ^ { \mathrm { { \bar { i } n } } } + n ^ { \mathrm { { o u t } } }$ , let $F _ { m }$ denote the total compu tational cost and ${ \mathcal { R } } _ { m }$ capture all remaining operations. Assuming KV caching within each request and counting causal attention pairs, we obtain:

$$
\begin{array} { c } { F _ { m } \approx \displaystyle 2 A _ { m } N + 4 L D \sum _ { t = 1 } ^ { N } t + \mathcal { R } _ { m } } \\ { = \displaystyle 2 A _ { m } N + 2 L D N ( N + 1 ) + \mathcal { R } _ { m } . } \end{array}\tag{11}
$$

Since parameter matrix operations dominate overall computation (exceeding 95% of total FLOPs in our experiments), we omit both the attention interaction term and ${ \mathcal { R } } _ { m } ,$ , yielding the FLOPs proxy $\widehat F _ { m } = 2 A _ { m } \big ( n ^ { \mathrm { i n } } + n ^ { \mathrm { o u t } } \big )$ . Taking the active parameter count of the smallest model in our LLM pool, Qwen3.5-35B-A3B $( A _ { 0 } = 3 \times 1 0 ^ { 9 } )$ as the reference baseline, we define $2 A _ { 0 }$ FLOPs as one cost unit. Let ${ \cal { T } } _ { W } ( \tau )$ denote the index set of worker calls along trajectory �, where the �-th call invokes model $m _ { j }$ with input length $n _ { j } ^ { \mathrm { i n } }$ , output length $n _ { j } ^ { \mathrm { o u t } }$ , and estimated cost $\widehat { F } _ { j }$ . The normalized trajectory cost is then defined as:

$$
\begin{array} { l } { { \displaystyle C ( \tau ) : = \frac { \sum _ { j \in { \cal I } _ { W } ( \tau ) } \widehat F _ { j } } { 2 A _ { 0 } } } } \\ { { \displaystyle \quad = \sum _ { j \in { \cal I } _ { W } ( \tau ) } \frac { A _ { m _ { j } } } { A _ { 0 } } \big ( n _ { j } ^ { \mathrm { i n } } + n _ { j } ^ { \mathrm { o u t } } \big ) . } } \end{array}\tag{12}
$$

In practice, we approximate $A _ { m }$ using the active parameter counts reported by model providers. For dense models, this equals the total parameter count, since all weights participate in each autoregressive forward pass. Active parameter counts for all evaluated models are summarized in Table 4.

Table 4: Active parameter counts of the models. For dense models (<sup>∗</sup>), the total parameter count is reported. <sup>‡</sup> denotes models used only in Section 5.4.
<table><tr><td>Model</td><td>Act. Params (B)</td><td>Model</td><td>Act. Params (B)</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>3</td><td>DeepSeek-V4-Flash</td><td>13</td></tr><tr><td>Qwen3-4B*</td><td>4</td><td>Qwen3.5-397B-A17B</td><td>17</td></tr><tr><td>Qwen3.8-Flash‡</td><td>6</td><td>Kimi-K2.8-Preview‡</td><td>32</td></tr><tr><td>Qwen3-8B*</td><td>8</td><td>GLM-5</td><td>40</td></tr><tr><td>Llama-3.1-8B*</td><td>8</td><td>Qwen3.8-Max‡</td><td>95</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>10</td><td></td><td></td></tr></table>