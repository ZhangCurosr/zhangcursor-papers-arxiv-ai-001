# CONSISTENT PLAN-ACT FOR LONG-HORIZON AGENTIC TASKS

Heng-Zhuang Li<sup>1,2,3,∗</sup> Yi-Kai Zhang<sup>1,2,∗</sup> Yu Wang<sup>3,4</sup> Yueqing Sun<sup>3</sup> Jiayuan Zhang<sup>3</sup> Qi Gu<sup>3,B</sup> Han-Jia Ye<sup>1,2,B</sup>

<sup>1</sup>School of Artificial Intelligence, Nanjing University

<sup>2</sup>National Key Laboratory for Novel Software Technology, Nanjing University

<sup>3</sup>LongCat Team, Meituan <sup>4</sup>University of Science and Technology of China

## ABSTRACT

Long-horizon agentic tasks demand strong reasoning and efficient execution across successive interactions with dynamic environments. A common approach decouples high-level planning from low-level execution through separate planner and actor roles. To investigate coordination failures in these tasks, we prompt both agents for structured state assertions and compare their reports programmatically to detect explicit contradictions. Our analyses reveal systematic disagreement about the same task-relevant state facts, a phenomenon we term planner-actor state mismatch. We further find that providing agents with task-relevant state information reduces mismatch and improves coordination and task performance. Based on the systematic analysis of the state mismatch, we propose Consistent Plan-Act (ConPAct), which feeds detected contradictions back to both agents to form consistent state interpretations and fine-tunes them on curated consistent interactions for better coordination. ConPAct improves performance across various environments and model configurations, e.g., increasing MiniGrid success rate from 38.6% to 54.4% with GPT-5.6-sol/terra as planner and actor respectively, demonstrating that state consistency can guide both inference-time correction and coordination training. Code: https://github.com/Fir-lat/ConPAct.

## 1 INTRODUCTION

Large language models (LLMs) (OpenAI, 2026a; DeepSeek-AI, 2026) increasingly serve as agents in interactive environments, including games (Paglieri et al., 2025b), web navigation (Yao et al., 2022; Zhou et al., 2024), GUI automation (Rawles et al., 2023; Xie et al., 2024; Rawles et al., 2025), and embodied tasks (Shridhar et al., 2021; Ahn et al., 2023; Liu et al., 2023). These tasks require agents to combine long-term planning (Chen et al., 2025b; Li et al., 2026a) with action selection guided by observations and environmental feedback (Wang et al., 2025a). Alongside frameworks that interleave reasoning and action (Yao et al., 2023) or incorporate reflection on task feedback (Shinn et al., 2023), plan-act architectures assign high-level planning and low-level execution to separate agents (Erdogan et al., 2025; Jin et al., 2026; Molinari and Ciravegna, 2026). A planner translates the overall objective into sub-goals (Hu et al., 2025b), while an actor selects and adapts actions to achieve them (Erdogan et al., 2025; Peng et al., 2026). This division reduces the burden of simultaneously managing planning and execution (Erdogan et al., 2025) and allows models with different capabilities to serve the two roles (Tan et al., 2026). Their coordination through sub-goals and execution feedback requires common ground (Clark and Brennan, 1991): compatible interpretations of the task-relevant information referenced in sub-goals and execution feedback.

We use GPT-5.6-sol (OpenAI, 2026c) to examine sub-goal execution in failed plan-act trajectories across tasks and model configurations. Categorizing these failures reveals recurring disagreements between the planner and actor about the same task-relevant environment state. These disagreements can lead the actor to deviate from the planner’s intent or treat feasible sub-goals as infeasible, contributing to task failure. This failure mode differs from limited planning (Li et al., 2026c; Tan et al.,

(b) Failure cases.

![](images/342d403eb3c564311fe158e8196e5f18ee2d221c858d78852d71c055708ad4e9.jpg)  
(a) Planner-Actor state mismatch and consistency.  
Figure 1: Planner-actor state mismatch. (a) Misreading the upper box as one column farther right, the actor pushes Right twice, causing deadlock; the correct sequence is Right → Up, with consistent interpretation as the planner. (b) Mismatch occurs in 85.0% of failed trajectories, far exceeding non-adherence (31.1%) and low planning/execution capability (8.4%).

2026; Dou et al., 2026) or execution (Hu et al., 2025b; Zhen et al., 2026; Wang et al., 2025a) capabilities, and sub-goal non-adherence (Jia et al., 2025; Kim et al., 2026; Venkatesan et al., 2026). We term it planner-actor state mismatch. Figure 1a illustrates how individually coherent decisions can fail to support coordinated progress. State mismatch occurs at least once in 85% of the analyzed failed trajectories (Figure 1b). Improving planning (Erdogan et al., 2025; Li et al., 2026c), execution (Hu et al., 2025b), or adherence (Jeong et al., 2026) does not by itself require the two roles to agree on the state facts underlying their decisions. Prior work examines inter-agent failures (Cemr et al., 2025), explicit belief-state representations (Li et al., 2023), and belief alignment through dialogue (Qiu et al., 2024; Dongre and Hakkani-Tur, 2026); here we focus on agreement about the¨ same observed state facts at planner-actor handoffs.

To measure state mismatch, we use state assertions inspired by explicit belief-state representations (Li et al., 2023). Each agent reports its perceived task-relevant state in a shared structured format alongside its sub-goal or action. These reports serve as proxies for state interpretations, enabling programmatic detection of explicit contradictions without ground truth, or external verifiers. Our analysis is summarized as follows. First, mismatch occurs across environments and models and increases with the capability gap between roles. Controlled comparisons show that it persists under both visual and textual observations, with cross-role disagreement exceeding within-role sampling variability. Second, controlled interventions establish its causal impact on coordination: correcting the actor’s state interpretation restores coordinated behavior while holding the sub-goal and observation fixed. Finally, our mismatch metric enables us to examine whether providing reliable state information improves agreement between the roles. Supplying state assertions from a stronger model or environment oracle reduces the measured disagreement while increasing task success.

Building on these findings, we propose Consistent Plan-Act (ConPAct), which feeds programmatically detected state contradictions back to both agents. Unlike model-generated critiques (Madaan et al., 2023; Du et al., 2024), this feedback identifies mutual conflicting fields in their state assertions without any external information. The agents then discuss these disagreements and revise their interpretations, sub-goals, or actions before subsequent execution. There are three variants: ConPAct-Inference applies this intervention during inference without training. ConPAct-Sampling trains on sampled collaboration and correction sequences selected for state consistency, whereas ConPAct-Replaying uses a teacher model to reconstruct such supervision from existing ReAct (Yao et al., 2023) trajectories. Experiments across various environments demonstrate benefits from both inference-time intervention and training. On MiniGrid (Chevalier-Boisvert et al., 2023; 2019), ConPAct-I raises success from 38.6% to 54.4% with a GPT-5.6-sol planner and GPT-5.6-terra actor (OpenAI, 2026c). ConPAct-S reaches 39.7% with Qwen3-VL-32B (Bai et al., 2025), exceeding the strongest evaluated training baseline by a large margin (29.1%). Ablations support the contributions of explicit conflict feedback, participation by both roles, and supervision of correction sequences. Our contributions are summarized as follows:

• We identify and characterize planner-actor state mismatch, a relational failure mode. We estab lish its prevalence, impact on coordination, and mitigation potential through systematic analyses.

![](images/24d70151af993bf851eaa984ee938be14b0f0c7b9c25483aabb8bfc93d546368.jpg)

![](images/7e7d0ee20a9f9805f5edf09ea041a40391fa970bff99ac370676b7cecb8a3a48.jpg)  
(a) Prevalence of planner–actor state mismatch.  
(b) State mismatch despite high sub-goal adherence.  
Figure 2: Prevalence of planner-actor state mismatch. (a) Mismatch occurs across various model configurations, with even homogeneous configurations have notable state mismatch; (b) State mismatch coexists with high sub-goal adherence, with marker size indicating trajectory frequency. Model sources are provided in Appendix C.2.

• We develop ConPAct, which uses programmatically detected state contradictions to guide joint revision during inference and trains agents on consistent collaboration and correction sequences obtained through sampling or replaying from existing trajectories.

• Experiments across environments and model configurations demonstrate improved coordination and task performance, with ablations supporting the contributions of each components.

## 2 CHARACTERIZING PLANNER-ACTOR STATE MISMATCH

## 2.1 PROBLEM SETUP AND MEASUREMENT

Long-Horizon Agentic Tasks. We study planner-actor state mismatch in Sokoban (Hu et al., 2025a) and MiniGrid (Chevalier-Boisvert et al., 2023; 2019), which require agents to pursue goals over extended sequences of interdependent actions and coordinate planning and execution throughout the task. Their underlying grid states are also readily accessible, providing ground truth for evaluating agents’ state interpretations. We evaluate 200 Sokoban levels and 150 MiniGrid levels, with eight rollouts per level. Our evaluation covers Claude (Anthropic, 2026a;b), GPT (OpenAI, 2026c;b), Kimi (Kimi Team, 2026; Moonshot AI, 2026), and Qwen (Qwen Team, 2026b;a; Bai et al., 2025), using homogeneous/heterogeneous model configurations.

Plan-Act Agents. Given task instruction $x _ { i } ,$ , trajectory $\tau _ { i }$ consists of model-call turns $t ,$ each with observation $o _  i , $ <sub>t</sub> and response $y _ { i , t } .$ . The planner generates a sub-goal $g _ { i , t }$ from its observation; the actor uses the latest sub-goal and its own observation to select action $a _ { i , t }$ and report execution status $z _ { i , t }$ (Erdogan et al., 2025; Peng et al., 2026). Consecutive actor turns under one sub-goal form an execution segment. Completion or execution difficulties trigger dynamic replanning, where the planner revises the sub-goal using the latest observation and actor feedback (Erdogan et al., 2025; Lin et al., 2023; Paglieri et al., 2025a). Coordination through sub-goals and feedback thus depends on the roles’ interpretations of task-relevant state.

State Assertions. We prompt each role to report its perceived state as JSON alongside its subgoal or action. For each environment, we specify a compact schema $\mathcal { F }$ shared by both roles, with fixed fields and a predefined value domain $\bar { \nu } _ { f }$ for each field $f .$ Both roles use the shared schema $\mathcal { F }$ describing task-relevant properties inferable from a single observation, such as object identities, action prerequisites, and goal conditions. Each field $f$ has a predefined set $\mathcal { V } _ { f }$ of canonical, mutually exclusive values. Parsing response $y _ { i , \imath }$ yields the state assertion $\mathbf { v } _ { i , t } = \mathrm { p a r s e } _ { \mathcal { F } } ( y _ { i , t } )$ , where $\mathbf { v } _ { i , t } [ f ] \in \mathcal { V } _ { f }$ . At planner-actor handoffs, we compare assertions from consecutive role calls on the same observation $( o _ { i , t } = o _ { i , t + 1 } )$ , without either role seeing the other’s assertion. For a valid pair, we detect state mismatch if any corresponding field differs:

$$
d _ { i , t } = \mathbf { 1 } [ \exists f \in \mathscr { F } : \mathbf { v } _ { i , t } [ f ] \neq \mathbf { v } _ { i , t + 1 } [ f ] ] .\tag{1}
$$

![](images/a5bc22f79e6aecdf727db7860ed507e573e027d53c2405933d8bdd98bda8a288.jpg)  
(a) Visual vs. textual mismatch.

![](images/c0d3610f084b50b026d1a80a22ffb113bafc96318b34e4c69c1f40ec5220db20.jpg)  
(b) Excess cross-role disagreement.

![](images/9ba1f32550465368498c98de14cac18df381111bc9e71feb7c36e2dbcc53e401.jpg)  
(c) Failure patterns.  
Figure 3: Planner-actor state mismatch is systematic and prevalent in failed trajectories. (a) Mismatch persists under both visual and textual observations; (b) mean cross-role disagreement exceeds within-role sampling variability; (c) mismatch-only failures dominate across task configurations.

Metrics for analyzing state mismatch. (1) State mismatch rate (D) is the fraction of valid handoff pairs with $d _ { i , t } = 1 ; ( 2 )$ State correctness $( A ^ { \mathrm { p l a n } } , A ^ { \mathrm { a c t } } )$ is each role’s exact-match assertion accuracy against environment ground truth on the same paired observations; (3) Sub-goal adherence (H) is the fraction of execution segments in which an LLM judge (Zheng et al., 2023) determines that the actor pursues the assigned sub-goal; (4) Task success (S) is the fraction of trajectories with environment-confirmed task completion. The detailed definitions are placed in Appendix B.

## 2.2 PREVALENCE OF PLANNER-ACTOR STATE MISMATCH

Prevalence of state mismatch. As shown in Figure 2a, mismatch occurs in all combinations, reaching 26.5% and 41.6% for homogeneous Opus-5 and GPT-5.6-sol, respectively. Figure 2b illustrates that high sub-goal adherence coexists with substantial state mismatch. Specifically, 82.9% of those with measured adherence H = 100% contain at least one detected mismatch.

State mismatch across observation modalities. In Figure 3a, replacing screenshots with structured textual states reduces mismatch in stronger configurations, but mismatch persists in all configurations. Even homogeneous Opus-5 and GPT-5.6-sol still exhibit rates of 9.6% and 2.5%, respectively. Therefore, planner-actor state mismatch extends beyond visual perception difficulties.

State mismatch beyond sampling variability. To test whether sampling variability explains mismatch, we repeatedly sample on 300 fixed observations while keeping each role’s model and context fixed. We measure disagreement between state assertions from repeated sampling within each role and average the planner’s and actor’s rates to obtain a sampling baseline. Cross-role disagreement exceeds this baseline for all eight models under visual observation, with all 95% confidence intervals for the excess excluding zero (Figure 3b). For Opus-5, the excess is 22.8 percentage points, indicating that planner-actor disagreement contains systematic differences beyond sampling variability.

## 2.3 BEHAVIORAL IMPLICATIONS OF STATE MISMATCH

State mismatch dominates failure patterns across task configurations. Figure 3c plots the fraction of level-configuration pairs exceeding each within-pair failure-share threshold, considering only pairs with failed rollouts. Mismatch-only failures exceed these thresholds in more pairs than nonadherence-only or low-capability failures. In 49.4% of these pairs, mismatch without detected nonadherence occurs in more than half of all failures. Thus, conflicting state interpretations characterize most failures in a broad range of task configurations.

Controlled interventions. We keep the observation, assigned sub-goal, and actor fixed within each event and vary only the supplementary information. We compare four conditions: no supplementary information, a state assertion with random field values to control for structured formatting, freeform control context, and task-relevant state facts from Opus-5. Freeform control context consists of character-matched irrelevant text for blocked sub-goals and non-critical state facts for feasible subgoals. Valid handback rate measures handoff signals in the first response; first-step progress rate measures first actions that reduce distance to the sub-goal.

Task-relevant state information improves feedback and execution. The results are displayed in Figure 4a. For blocked sub-goals, task-relevant facts increase GPT-5.6-sol’s valid handback rate from 0.4% to 94.5%, compared with 0.1% for random state assertions and 0.4% for control context. For feasible sub-goals, they increase Qwen3-VL-8B’s first-step progress rate from 32.2% to 80.9%, compared with 29.2% for random state assertions and 35.8% for non-critical state facts. The gains over both controls show that task-relevant state information helps actors adapt their coordination decisions to sub-goal feasibility, i.e., returning control for replanning when execution is blocked and selecting a progress-making action when execution is feasible. Full results for mismatch prevalence, observation and sampling controls, and state interventions appear in Appendix D.

![](images/0cbc9e244a4689a24c03c54f44cd86e1177fea758a950a119556ec7b08aeb13f.jpg)  
(a) Behavioral implications.

![](images/d9c27dcd8236087e1c50be15610474ba3cd8e9e95dad345adc6866da3a4b73ec.jpg)  
(b) Injecting state assertions.

![](images/23965645e81ab99e7879975eac8b71cde1cec79a9b30fdfbcde02c0e358acced.jpg)  
(c) ConPAct-I framework.  
Figure 4: State information and consistent plan-act coordination. (a) Task-relevant state facts improve the actor’s decisions. (b) Injecting reliable state assertions reduces mismatch and improves task success. (c) ConPAct feeds detected contradictions back for coordination reconciliation.

## 3 CONSISTENT PLAN-ACT

## 3.1 TASK-LEVEL IMPLICATIONS AND CONSISTENCY INTERVENTIONS

To evaluate the task-level effects of reliable state information, we inject oracle assertions from environment states and assertions generated by Opus-5. Relative to no injection, both sources improve assertion correctness, reduce mismatch, and increase task success consistently, shown in Figure 4b. For GPT-5.6-sol paired with GPT-5.6-terra, oracle injection reduces mismatch from 61.9% to 7.3% and raises task success from 47.2% to 84.1%. Building on these gains and the behavioral results in Section 2.3, we propose Consistent Plan-Act (ConPAct), which aligns planner-actor state interpretations through task-relevant information exchange. Detected contradictions identify the state facts that require reassessment. ConPAct feeds these contradictions back to both agents, directing their discussion toward the disputed facts and how they should inform planning and execution.

## 3.2 FEEDING BACK DETECTED CONTRADICTIONS FOR STATE CONSISTENCY

As shown in Figure 4c, ConPAct-I compares state assertions about the same observation at planner– actor handoffs. When definite values conflict, it feeds the disputed fields and both roles’ judgments back to the agents. Environment actions remain paused so that the discussion concerns a fixed state. The planner and actor exchange their interpretations, reassess the disputed facts against their observations, and revise their state judgments and associated sub-goals or execution decisions. We recheck their assertions after each exchange until all fields have definite, matching values or the dis cussion budget is exhausted. Execution then resumes with the updated discussion context. This process uses the agents’ observations and exchanged messages, requiring neither environment ground truth nor an additional verifier model.

## 3.3 LEARNING STATE-CONSISTENT COORDINATION

We further train agents to coordinate and resolve state disagreements using consistent interactions and corrective discussions. ConPAct-S collects these interactions by sampling trajectories with ConPAct-I. From both successful and failed trajectories, we retain consistent collaboration segments and correction sequences that reach full agreement across all assertion fields. We supervise only valid replies without detected state contradictions, while keeping earlier inconsistent replies as context for learning how to revise conflicting judgments. Each example supervises the role’s complete response given its task, observation, and available interaction history.

Table 1: Inference-time performance of Plan-Act methods. Bold/underlining mark the best/secondbest values, including ties. The ReAct (Actor) uses only the actor and is unranked.
<table><tr><td rowspan="3">Method</td><td colspan="4">Opus-5 Sonnet-5</td><td colspan="2">GPT-5.6-sol GPT-5.6-terra</td><td colspan="4">Kimi K3 Kimi K2.6</td></tr><tr><td>Sok. Cra.</td><td></td><td>Proc. Mini.</td><td>Sok. Cra.</td><td>Proc. Mini.</td><td>Sok. Cra.</td><td>Proc. Mini.</td><td>Sok. Cra.</td><td>Qwen3.5-9B</td><td>. Proc. Mini.</td></tr><tr><td>ReAct (Actor) 70.9</td><td>4.74 20.55</td><td></td><td>33.4 66.2</td><td>5.10 17.21</td><td>31.4 45.7 3.39</td><td>11.23</td><td>21.9 6.2</td><td>3.14</td><td>2.29 7.9</td></tr><tr><td></td><td>84.2</td><td>23.49</td><td></td><td>75.1 5.131 17.20</td><td></td><td>57.7 4.23</td><td></td><td>12.0 3.27</td><td>3.19</td><td></td></tr><tr><td>Plan-Act</td><td>5.26 91.3 5.23</td><td>25.26</td><td>33.1 31.6</td><td>76.7 5.13 17.95</td><td>38.6 47.7</td><td>65.5 3.29</td><td>10.85 28.3 11.85 27.1</td><td>13.3 3.32</td><td>4.76</td><td>13.9 10.6</td></tr><tr><td>HiPlan TAPE</td><td>90.5</td><td>5.17 21.25</td><td>29.2</td><td>75.9 5.24 21.12</td><td>31.2</td><td>53.6 3.82</td><td>12.15 21.5</td><td>10.3 3.01</td><td>2.11</td><td>8.6</td></tr><tr><td>ConPAct-I</td><td>95.4 5.91 25.75 36.7</td><td></td><td></td><td>86.6 5.14 22.83</td><td>54.4</td><td>65.8 4.16</td><td>12.7934.2</td><td>15.7 3.46</td><td>4.85</td><td>13.5</td></tr></table>

Prior work derives plan supervision from execution traces (Erdogan et al., 2025) or relabels their goals in hindsight (Andrychowicz et al., 2017; Li et al., 2026b). When only existing ReAct (Yao et al., 2023) trajectories are available, ConPAct-R converts them into planner-actor interactions while preserving the recorded action sequence. A teacher segments each trace into sub-goals and adds state assertions and role exchanges. Around nonoptimal actions, it constructs hypothetical errors in state judgments or sub-goals, followed by feedback and revision, to provide correction examples. We validate the reconstructed interactions against the recorded execution and review each candidate target using only the information available to the student at that turn. We then apply the same consistency filtering and supervision rules as ConPAct-S, keeping hypothetical errors and nonoptimal actions as context only. Algorithms and prompt templates for all three variants are provided in Appendix E.

## 4 EVALUATION

## 4.1 EXPERIMENTAL SETUP

Environments. We evaluate on Sokoban from GamingAgent (Hu et al., 2025a), Crafter (Hafner, 2022), Procgen (Cobbe et al., 2020), and MiniGrid/BabyAI (Chevalier-Boisvert et al., 2023; 2019). These environments cover box-pushing, survival and crafting, procedurally generated games, and instruction-following grid-world tasks. The evaluation sets contain 200 Sokoban puzzles balanced between one and two boxes, 200 Crafter seeds, 140 levels from seven Procgen games in easy mode, and 150 instances spanning 27 MiniGrid and BabyAI task IDs. Each instance is evaluated with eight rollouts. The action and total model-call limits are both 60 for Sokoban, 160 for Crafter and Procgen, and 60–200 for MiniGrid, depending on the task. We additionally evaluate desktop task performance on 361 OSWorld (Xie et al., 2024) tasks spanning ten application categories, with three rollouts per task.

Configurations. ConPAct-I is evaluated with four planner–actor pairs from the Claude, GPT, Kimi, and Qwen families (Table 1). For supervised training, we use Qwen3-VL-8B and Qwen3-VL-32B (Bai et al., 2025), with either the same model size in both roles or a 32B planner paired with an 8B actor. GPT-5.6-sol (OpenAI, 2026c), with its default thinking effort, serves as the teacher for data construction, and the training-token budget is matched to three epochs of ReAct SFT. ConPAct-S/R are evaluated with model-initiated discussion and without system-triggered ConPAct-I.

Baselines. Inference baselines include ReAct (Yao et al., 2023) using either the planner or actor model, Plan-Act (Erdogan et al., 2025), HiPlan (Li et al., 2026c), and TAPE (Jeong et al., 2026). Training baselines include ReAct SFT, Plan-Act SFT, PAA (Erdogan et al., 2025), HSL (Li et al., 2026b), WebSTAR (He et al., 2026), and ECoT (Zawalski et al., 2025).

Evaluation metrics. We focus on Sokoban avg@8, Crafter Score (Hafner, 2022), Procgen interquartile mean (IQM) (Agarwal et al., 2021), and MiniGrid success rate (SR). MiniGrid SR is averaged equally across the 27 task IDs; Procgen normalized-return metrics are multiplied by 100 for display. For OSWorld, we report mean evaluation score and full success rate as Score/SR (%). State analyses use mismatch rate D and assertion correctness A<sup>plan</sup> and A<sup>act</sup> from Section 2.1. Groundtruth states serve only to evaluate correctness in these analyses. Experimental configurations and metric definitions are provided in Appendices C and B.3, respectively.

Table 2: Training results. Bold/underlining mark the best/second-best values, including ties.
<table><tr><td></td><td colspan="4">Qwen3-VL-8B ↓  $Q w e n 3 \ – V L \ – 8 B$ </td><td colspan="4">Qwen3-VL-32B Y Qwen3-VL-32B</td><td colspan="4">Qwen3-VL-32B ↓ Qwen3-VL-8B</td></tr><tr><td>Method</td><td>Sok.</td><td>Cra.</td><td>Proc.</td><td>Mini.</td><td>Sok.</td><td>Cra.</td><td>Proc.</td><td>Mini.</td><td>Sok.</td><td>Cra.</td><td>Proc.</td><td>Mini.</td></tr><tr><td>ReAct SFT</td><td>44.4</td><td>4.24</td><td>4.50</td><td>8.8</td><td>60.1</td><td>4.63</td><td>6.88</td><td>18.5</td><td></td><td></td><td></td><td></td></tr><tr><td>Plan-Act SFT</td><td>35.6</td><td>4.06</td><td>3.50</td><td>7.8</td><td>50.7</td><td>3.10</td><td>6.35</td><td>15.8</td><td>45.4</td><td>2.85</td><td>5.60</td><td>11.9</td></tr><tr><td>PAA</td><td>43.8</td><td>4.33</td><td>3.15</td><td>10.8</td><td>65.4</td><td>3.76</td><td>8.27</td><td>11.6</td><td>49.4</td><td>3.40</td><td>5.89</td><td>8.7</td></tr><tr><td>HSL</td><td>44.4</td><td>4.39</td><td>4.82</td><td>8.0</td><td>61.3</td><td>4.23</td><td>5.85</td><td>16.5</td><td></td><td></td><td></td><td></td></tr><tr><td>WebSTAR</td><td>44.7</td><td>4.28</td><td>5.41</td><td>8.7</td><td>66.2</td><td>4.96</td><td>7.26</td><td>13.1</td><td></td><td></td><td></td><td></td></tr><tr><td>ECoT</td><td>34.4</td><td>4.77</td><td>2.56</td><td>10.2</td><td>57.2</td><td>5.23</td><td>5.25</td><td>29.1</td><td>41.2</td><td>4.70</td><td>2.62</td><td>12.7</td></tr><tr><td>ConPAct-S</td><td>48.6</td><td>5.16</td><td>3.22</td><td>27.0</td><td>68.4</td><td>5.05</td><td>9.12</td><td>39.7</td><td>48.5</td><td>5.53</td><td>4.56</td><td>28.2</td></tr><tr><td>ConPAct-R</td><td>49.1</td><td>4.72</td><td>6.38</td><td>24.8</td><td>66.9</td><td>5.55</td><td>6.35</td><td>33.9</td><td>54.8</td><td>5.26</td><td>7.36</td><td>27.8</td></tr></table>

## 4.2 MAIN RESULTS

The main tables report Sokoban avg@8, Crafter Score, Procgen IQM, and MiniGrid SR (all ↑).   
Complete inference and training results are provided in Appendix Tables 15 and 16.

ConPAct-I improves task performance across model families. Table 1 shows that ConPAct-I improves Sokoban avg@8 and Procgen IQM over Plan-Act for all four model pairs. Sokoban gains range from 3.7 to 11.5 percentage points. With GPT-5.6-sol and GPT-5.6-terra, MiniGrid SR increases from 38.6% to 54.4%, exceeding both HiPlan and TAPE. These results support using detected state contradictions to improve plan-act coordination without training.

Both sampling and replay provide effective coordination supervision. In Table 2, ConPAct-S and ConPAct-R outperform both ReAct SFT and Plan-Act SFT on Sokoban avg@8 and MiniGrid SR at both model sizes. For Qwen3-VL-32B, ConPAct-S achieves 39.7% MiniGrid SR, compared with 29.1% for the strongest baseline. The two data routes also yield leading results on the other environments: ConPAct-S attains the highest Procgen IQM (9.12), while ConPAct-R attains the highest Crafter Score (5.55) among the 32B methods.

The gains extend to heterogeneous model sizes. With a 32B planner and an 8B actor (Table 2, rightmost column group), both variants improve Sokoban avg@8, Crafter Score, and MiniGrid SR over Plan-Act SFT. MiniGrid SR rises from 11.9% to 28.2% with ConPAct-S and 27.8% with ConPAct-R. Thus, consistency-oriented supervision also benefits coordination between models of different sizes.

The improvements extend to desktop tasks. On 361 OSWorld tasks, ConPAct-I outperforms ReAct and Plan-Act in both overall Score and SR across all four evaluated models (Figure 5a). Relative to the stronger baseline for each model and metric, Score improves by 0.7–0.9 percentage points and SR by 0.8–1.0 points. These results extend the gains from game environments to desktop interaction. Appendix Table 17 reports the complete category-level results.

## 4.3 ABLATION STUDY

Conflict feedback and planner participation improve reconciliation. We compare the full intervention with correction disabled, correction without specific conflict feedback, and correction restricted to the actor. The full method achieves the highest task success and repair rates for both GPT configurations on Sokoban and MiniGrid (Figure 5b). For the heterogeneous pair on Sokoban, repair reaches 98.0%, compared with 90.8% without specific feedback and 37.7% with actor-only correction. These comparisons support involving both roles and identifying their disagreements.

![](images/423a22a0a8c432e571d728fe6077ab81e0824daf1715c83804129ac18bbb67e0.jpg)  
(a) OSWorld.

![](images/7405422c715e0f51ed772787bdd47904b4cb855e51b720bd01897fbab79e45e0.jpg)  
(b) Inference ablation.

![](images/8305fd35d4a46405aeca564066241d56794dbbe867a7a52538cf139b67af1a3a.jpg)  
(c) Method comparison.  
Figure 5: OSWorld performance and Sokoban inference-time comparisons. (a) OSWorld Score and SR relative to the stronger of ReAct and Plan-Act within each model, measured in percentage points; point labels give absolute percentages and purple annotations give ConPAct-I gains. (b) Task success (bars, left axis) and disagreement repair (lines, right axis) under different correction settings. (c) Task success with assertion prompting, resampling, Self-Refine, and ConPAct-I.

Reconciliation improves on assertion prompting and resampling. Figure 5c compares ConPAct-I with assertion-only prompting, resampling, and Self-Refine (Madaan et al., 2023). ConPAct-I achieves higher Sokoban success than assertion prompting and resampling in all three configurations, and higher success than Self-Refine in both homogeneous configurations. The corresponding MiniGrid results show the same ordering.

Supervising and replaying reconciliation improve training. We compare full ConPAct-R with a baseline omitting all three data-curation components and two variants that retain consistency filtering but remove reconciliation supervision or reconciliation replay. Full ConPAct-R achieves higher task success and lower mismatch than all three alternatives across the evaluated model configurations on Sokoban and MiniGrid (Figure 6a). For 32B on Sokoban, it improves avg@8 from 55.9% to 66.9% and reduces mismatch from 34.7% to 1.4% relative to the baseline.

Complete Sokoban and MiniGrid results for the inference and training-data ablations are reported in Appendices F.4 and F.5, respectively.

## 4.4 ANALYSIS OF STATE UNDERSTANDING AND COORDINATION

We examine ConPAct-R through initial state judgments, discussion responses, and the correctness of shared interpretations. Complete results for Sokoban and MiniGrid are reported in Appendix F.6.

Training improves initial state understanding. Across the three Sokoban configurations, ConPAct-R reduces initial mismatch by 22.8–26.6 percentage points and improves each role’s assertion correctness by 36.5–48.0 points (Figure 6b). These improvements precede discussion, indicating that the trained models begin coordination with more accurate and compatible state judgments. MiniGrid exhibits the same pattern.

Training improves discussion responses and disagreement repair. Full ConPAct-R yields higher discussion rates and lower residual mismatch than either data ablation across all three model configurations in both environments (Figure 6c). For 32B on Sokoban, the discussion rate reaches 71.6%, while residual mismatch falls to 13.3%. This supports supervising corrective interactions to improve how agents respond to disagreements.

Reconciliation increases correct shared interpretations. To distinguish correct agreement from mere consistency, we examine whether the two roles’ shared assertions also match the environment state. Full ConPAct-R has a higher proportion of consistent and correct assertions than either ablation, both before and after discussion, across all three Sokoban configurations. Discussion further increases this proportion by 2.7–5.1 percentage points, reaching 77.7–93.5% (Appendix Table 23). Correct agreement also increases in MiniGrid, supporting the role of reconciliation in improving shared state understanding.

![](images/737594034ccddbc7354ca600fd53b371a302ed2a83368cb599072e8b20561cad.jpg)  
(a) SFT data ablation.

![](images/206447421101336160c79b0a0698eba1336a209f173b35f60eae9bb816ad7a61.jpg)  
(b) Initial state understanding.

![](images/c489deab877cd6f6d1b274a4515de9c69a119c34d609a561451a76ad41f4e44e.jpg)  
(c) Discussion and repair.  
Figure 6: Supervised-training ablations and state analysis on Sokoban. (a) Task success and mismatch under ConPAct-R data ablations; Full includes consistency filtering, reconciliation supervision, and reconciliation replay. (b) Initial mismatch and assertion correctness; annotations show changes in percentage points from the baseline. (c) Discussion and repair rates for ConPAct-R and its data ablations. The components together contribute to the effectiveness of ConPAct.

## 5 RELATED WORK

## 5.1 PLAN-ACT AGENTS

Plan-act agents separate planning from action selection (Erdogan et al., 2025). Existing methods improve these capabilities through global guidance (Li et al., 2026c), constrained execution (Jeong et al., 2026), and hierarchical policy optimization (Hu et al., 2025b; Peng et al., 2026). ReCAPA and ContextFlow address cross-level correction and task-stage coordination, respectively (Zeng et al., 2026; Guo et al., 2026), while Agent GPA evaluates plan adherence (Jia et al., 2025). Training also exploits trajectory relabeling (Li et al., 2026b) and multi-agent correction or role specializa tion (Zhao et al., 2025a; Motwani et al., 2025; Subramaniam et al., 2025).

ConPAct identifies a distinct failure: incompatible interpretations of the same observed facts despite plan adherence. It trains on consistent collaboration and correction segments, retaining earlier conflicts as context for learning reconciliation. Its replay variant reconstructs these interactions without changing recorded actions, enabling existing trajectories to support coordination training.

## 5.2 STATE TRACKING AND VERIFICATION IN LLM AGENTS

ECoT and VAGEN train explicit state reasoning (Zawalski et al., 2025; Wang et al., 2025b), while CoEx and EVU maintain and update beliefs (Kim and won Hwang, 2025; Wang et al., 2026a). Learned transition models further support planning and policy learning (Luo et al., 2026; Ding et al., 2026). Self-Refine, multiagent debate, and Reflexion guide revision with model-generated feedback (Madaan et al., 2023; Du et al., 2024; Shinn et al., 2023), although self-verification remains limited (Stechly et al., 2025). Other methods score intermediate execution (Chae et al., 2025; Chen et al., 2025a), detect plan-constraint violations (Guo et al., 2024), or validate outputs against planner-defined criteria (Xu et al., 2026).

ConPAct programmatically compares both roles’ state reports about the same observation and returns conflicting fields for joint revision. The comparison assumes neither role is correct and requires neither environment ground truth nor an additional verifier-model call. This targeted feedback improves state interpretation and coordination by allowing both roles to revise their judgments.

## 6 CONCLUSION

We identify and study planner-actor state mismatch as a coordination failure in which agents interpret task-relevant states differently, even when the actor follows its assigned sub-goal. Our findings show that making these interpretations explicit and reconciling disagreements can improve coordination through both inference-time interaction and training. The online comparison and reconciliation require neither environment ground truth nor an external verifier. Although consistency alone does not ensure correctness, these results support treating shared state understanding as an explicit objective in designing and training collaborative agents. Future work can examine how this approach scales to larger agent teams and more complex environments, where differing observations and interdependent decisions create further challenges for coordination.

## REFERENCES

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In NeurIPS, 2021.

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as I can, not as I say: Grounding language in robotic affordances. In Proceedings ofThe 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 287–318. PMLR, 2023. CoRL 2022.

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In NeurIPS, 2017.

Anthropic. System card: Claude Opus 5. https://www.anthropic.com/claude-opu s-5-system-card, 2026a.

Anthropic. System card: Claude Sonnet 5. https://www.anthropic.com/claude-son net-5-system-card, 2026b.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL technical report. arXiv, 2511.21631, 2025.

Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya G. Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent LLM systems fail? In NeurIPS, 2025. Datasets and Benchmarks Track.

Hyungjoo Chae, Sunghwan Kim, Junhee Cho, Seungone Kim, Seungjun Moon, Gyeom Hwangbo, Dongha Lim, Minjin Kim, Yeonjun Hwang, Minju Gwak, Dongwook Choi, Minseok Kang, Gwanhoon Im, ByeongUng Cho, Hyojun Kim, Jun Hee Han, Taeyoon Kwon, Minju Kim, Beong woo Kwak, Dongjin Kang, and Jinyoung Yeo. Web-Shepherd: Advancing PRMs for reinforcing web agents. In NeurIPS, 2025.

Cong Chen, Kaixiang Ji, Hao Zhong, Muzhi Zhu, Anzhou Li, Guo Gan, Ziyuan Huang, Cheng Zou, Jiajia Liu, Jingdong Chen, Hao Chen, and Chunhua Shen. GUI-shepherd: Reliable process reward and verification for long-sequence GUI tasks. arXiv preprint arXiv:2509.23738, 2025a.

Jiaju Chen, Bo Sun, Yuxuan Lu, Yun Wang, Dakuo Wang, and Bingsheng Yao. CollabSim: A CSCW-grounded methodology for investigating collaborative competence of LLM agents through controlled multi-agent experiments. arXiv preprint arXiv:2606.06399, 2026.

Junjie Chen, Haitao Li, Jingli Yang, Yiqun Liu, and Qingyao Ai. Enhancing LLM-based agents via global planning and hierarchical execution. arXiv preprint arXiv:2504.16563, 2025b.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan,

Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob Mc-Grew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. arXiv, 2107.03374, 2021.

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv, 1604.06174, 2016.

Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. BabyAI: A platform to study the sample efficiency of grounded language learning. In ICLR, 2019.

Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo de Lazcano, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and Jordan Terry. Minigrid & Miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. In NeurIPS, 2023. Datasets and Benchmarks Track.

Herbert H. Clark and Susan E. Brennan. Grounding in communication. In Perspectives on Socially Shared Cognition. American Psychological Association, 1991.

Karl Cobbe, Christopher Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. In ICML, 2020.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. FlashAttention: Fast and´ memory-efficient exact attention with IO-awareness. In NeurIPS, 2022.

DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Hang Ding, Peidong Liu, Junqiao Wang, Ziwei Ji, Meng Cao, Rongzhao Zhang, Lynn Ai, Eric Yang, Tianyu Shi, and Lei Yu. DynaWeb: Model-based reinforcement learning of web agents. arXiv preprint arXiv:2601.22149, 2026.

Vardhan Dongre and Dilek Hakkani-Tur. Embodied multi-agent coordination by aligning world¨ models through dialogue. In Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL), 2026.

Zhihao Dou, Qinjian Zhao, Zhongwei Wan, Dinggen Zhang, Weida Wang, Benteng Chen, Towsif Raiyan, Qingtao Pan, Yang Ouyang, Chaoda Song, Zhiqiang Gao, Shufei Zhang, and Sumon Biswas. Plan then action: High-level planning guidance reinforcement learning for LLM reasoning. In ICML, 2026.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In ICML, 2024.

Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-Act: Improving planning of agents for long-horizon tasks. In ICML, 2025.

Shuhan Guo, Kun Zhang, Haifei Liu, Xingyu Gao, Yongqi Zhang, Yaqing Wang, and Quanming Yao. ContextFlow: Hierarchical task-state alignment for long-horizon embodied agents. arXiv preprint arXiv:2605.19314, 2026.

Yanjiang Guo, Yen-Jen Wang, Lihan Zha, and Jianyu Chen. DoReMi: Grounding language model by detecting and recovering from plan-execution misalignment. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

Danijar Hafner. Benchmarking the spectrum of agent capabilities. In ICLR, 2022.

Yifei He, Pranit Chawla, Yaser Souri, Subhojit Som, and Xia Song. WebSTAR: Scalable data synthesis for computer use agents with step-level filtering. In ACL, 2026.

Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P. Xing, Ion Stoica, Tajana Rosing, Haojian Jin, and Hao Zhang. lmgame-Bench: How good are LLMs at playing games? arXiv preprint arXiv:2505.15146, 2025a.

Zican Hu, Wei Liu, Xiaoye Qu, Xiangyu Yue, Chunlin Chen, Zhi Wang, and Yu Cheng. Divide and conquer: Grounding LLMs as efficient decision-making agents via offline hierarchical reinforcement learning. In ICML, 2025b.

Jongwon Jeong, Jungtaek Kim, and Kangwook Lee. TAPE: Tool-guided adaptive planning and constrained execution in language model agents. arXiv preprint arXiv:2602.19633, 2026.

Allison Sihan Jia, Daniel Huang, Nikhil Vytla, Seung Won Wilson Yoo, Nirvika Choudhury, Shayak Sen, John C. Mitchell, and Anupam Datta. What is your agent’s GPA? a framework for evaluating agent Goal-Plan-Action alignment. arXiv preprint arXiv:2510.08847, 2025.

Jiajie Jin, Xiaoxi Li, Yuyao Zhang, Guanting Dong, Zhao Yang, Yutao Zhu, and Zhicheng Dou. HiRA: Decoupling planning and execution with hierarchical reasoning in deep search. In ACM SIGIR, 2026.

Byeongjin Kim, Gyuwan Kim, and Seo Yeon Park. PPA-plan: Proactive pitfall avoidance for reliable planning in long-context LLM reasoning. In ACL, 2026.

Minsoo Kim and Seung won Hwang. CoEx – co-evolving world-model and exploration. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, 2025.

Kimi Team. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Huao Li, Yu Quan Chong, Simon Stepputtis, Joseph Campbell, Dana Hughes, Michael Lewis, and Katia Sycara. Theory of mind for multi-agent collaboration via large language models. In EMNLP, 2023.

Yunfan Li, Bingbing Xu, Xueyun Tian, Xiucheng Xu, and Huawei Shen. Beyond entangled planning: Task-decoupled planning for long-horizon agents. arXiv preprint arXiv:2601.07577, 2026a.

Zichao Li, Gang Wu, Zichao Wang, Ruiyi Zhang, Wanrong Zhu, Ryan A. Rossi, Vlad I. Morariu, and Jihyung Kil. Spinning straw into gold: Relabeling LLM agent trajectories in hindsight for successful demonstrations. In ICLR, 2026b.

Ziyue Li, Yuan Chang, Yuanbo Kong, Gaihong Yu, and Xiaoqiu Le. HiPlan: Hierarchical planning for web agents with adaptive global-local guidance. Data Intelligence, 2026c.

Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj Ammanabrolu, Yejin Choi, and Xiang Ren. SwiftSage: A generative agent with fast and slow thinking for complex interactive tasks. In NeurIPS, 2023.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In NeurIPS, 2023. Datasets and Benchmarks Track.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

Dezhao Luo, Bohan Tang, Kang Li, Georgios Papoudakis, Jifei Song, Shaogang Gong, Jianye Hao, Jun Wang, and Kun Shao. ViMo: A generative visual GUI world model for app agents. In ICLR, 2026.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-Refine: Iterative refinement with self-feedback. In NeurIPS, 2023.

Gianni Molinari and Fabio Ciravegna. Reason-plan-ReAct: A reasoner-planner supervising a ReAct executor for complex enterprise tasks. In AAAI 2026 Workshop on Agentic AI Benchmarks and Applications for Enterprise Tasks, 2026.

Moonshot AI. Kimi K2.6. https://huggingface.co/moonshotai/Kimi-K2.6, 2026.

Sumeet Ramesh Motwani, Chandler Smith, Rocktim Jyoti Das, Rafael Rafailov, Ivan Laptev, Philip H. S. Torr, Fabio Pizzati, Ronald Clark, and Christian Schroeder de Witt. MALT: Improving reasoning with multi-agent LLM training. In Conference on Language Modeling (COLM), 2025.

OpenAI. OpenAI GPT-5 system card. arXiv preprint arXiv:2601.03267, 2026a.

OpenAI. GPT-5.5 system card. https://openai.com/index/gpt-5-5-system-car d/, 2026b.

OpenAI. GPT-5.6 system card. https://deploymentsafety.openai.com/gpt-5-6, 2026c.

Davide Paglieri, Bartłomiej Cupiał, Jonathan Cook, Ulyana Piterbarg, Jens Tuyls, Edward Grefenstette, Jakob Nicolaus Foerster, Jack Parker-Holder, and Tim Rocktaschel. Learning when to¨ plan: Efficiently allocating test-time compute for LLM agents. arXiv preprint arXiv:2509.03581, 2025a.

Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kucinski, Lerrel Pinto, Rob Fergus, Jakob Nicolaus Foerster,´ Jack Parker-Holder, and Tim Rocktaschel. BALROG: Benchmarking agentic LLM and VLM¨ reasoning on games. In ICLR, 2025b.

Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, and Mingyi Hong. HiPER: Hierarchical plan–execute RL for multi-turn LLM agents. In ICML, 2026.

Shuwen Qiu, Mingdian Liu, Hengli Li, Song-Chun Zhu, and Zilong Zheng. MindDial: Enhancing conversational agents with theory-of-mind for common ground alignment and negotiation. In Annual Meeting ofthe Special Interest Group on Discourse and Dialogue (SIGDIAL), 2024.

Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qw en3.5, 2026a.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model. https://qwen.ai/ blog?id=qwen3.6-27b, 2026b.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, 2020.

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Android in the wild: A large-scale dataset for Android device control. In NeurIPS, 2023. Datasets and Benchmarks Track.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. AndroidWorld: A dynamic benchmarking environment for autonomous agents. In ICLR, 2025.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In NeurIPS, 2023.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, and Matthew´ Hausknecht. ALFWorld: Aligning text and embodied environments for interactive learning. In ICLR, 2021.

Kaya Stechly, Karthik Valmeekam, and Subbarao Kambhampati. On the self-verification limitations of large language models on reasoning and planning tasks. In ICLR, 2025.

Vighnesh Subramaniam, Yilun Du, Joshua B. Tenenbaum, Antonio Torralba, Shuang Li, and Igor Mordatch. Multiagent finetuning: Self improvement with diverse reasoning chains. In ICLR, 2025.

Haoran Tan, Zeyu Zhang, Chen Ma, Tianze Liu, Quanyu Dai, and Xu Chen. From coarse to fine: Self-adaptive hierarchical planning for LLM agents. In Findings of the Association for Computational Linguistics: ACL 2026, 2026.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In NeurIPS, 2023.

Madhumitha Venkatesan, Shicheng Wen, Jiajing Guo, Jorge Piazentin Ono, Liu Ren, and Dongyu Liu. Plover: Steering GUI agents through plan-centric interaction. arXiv preprint arXiv:2607.15193, 2026.

Hanlin Wang, Jian Wang, Chak Tou Leong, and Wenjie Li. STeCa: Step-level trajectory calibration for LLM agent learning. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, 2025a.

Hanlin Wang, Chak Tou Leong, Jian Wang, and Wenjie Li. Seeing isn’t believing: Mitigating belief inertia via active intervention in embodied agents. In Findings of the Association for Computational Linguistics: ACL 2026, 2026a.

Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, Lijuan Wang, Ranjay Krishna, Jiajun Wu, Fei-Fei Li, Yejin Choi, and Manling Li. VAGEN: Reinforcing world model reasoning for multi-turn VLM agents. In NeurIPS, 2025b.

Yu Wang, Yi-Kai Zhang, Wentao Shi, Ziang Ye, Yuchun Miao, Yueqing Sun, Qi Gu, Xunliang Cai, Lan-Zhe Guo, Han-Jia Ye, and Fuli Feng. Cast: Game solvers as turn-level teachers for llm agents. arXiv, 2607.25308, 2026b.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In NeurIPS, 2024. Datasets and Benchmarks Track.

Tianyang Xu, Dan Zhang, Kushan Mitra, and Estevam Hruschka. Verification-aware planning for multi-agent systems. In EACL, 2026.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. WebShop: Towards scalable real-world web interaction with grounded language agents. In NeurIPS, 2022.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In ICLR, 2023.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. In ICLR, 2025.

Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. In Proceedings ofThe 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pages 3157–3181. PMLR, 2025. CoRL 2024.

Xiyin Zeng, Yuyu Sun, Haoyang Li, Shouqiang Liu, and Hao Wang. ReCAPA: Hierarchical predictive correction to mitigate cascading failures. In ICLR, 2026.

Wanjia Zhao, Mert Yuksekgonul, Shirley Wu, and James Y. Zou. SiriuS: Self-improving multi-agent systems via bootstrapped reasoning. In NeurIPS, 2025a.

Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. SWIFT: A scalable lightweight infrastructure for fine-tuning. In AAAI, 2025b.

Shuai Zhen, Yanhua Yu, Ruopei Guo, Nan Cheng, and Yang Deng. Hierarchical reinforcement learning with augmented step-level transitions for LLM agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In NeurIPS, 2023.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A realistic web environment for building autonomous agents. In ICLR, 2024.

## Appendix Contents

A Additional Related Work 19   
A.1 Plan-Act Agents 19   
A.2 State Tracking and Verification in LLM Agents. 19   
A.3 Common Ground and Belief Alignment. 19   
B State Assertions 19   
B.1 Plan-Act Protocol 19   
B.2 Assertion Format 20   
B.3 Metrics 22   
C Experimental Setup 24   
C.1 Environments . 24   
C.2 Models and Baselines 27   
C.3 Training and Evaluation . 27   
D State Mismatch Analysis 29   
D.1 Mismatch Prevalence 29   
D.2 Control Experiments 30   
D.3 Failure Analysis . 32   
D.4 State Interventions. 32   
E Method Details 34   
E.1 ConPAct-I . 34   
E.2 ConPAct-S and ConPAct-R . 34   
E.3 Prompts. 35   
F Additional Results 40   
F.1 Inference-Time Results 40   
F.2 Supervised-Training Results 41   
F.3 OSWorld Results. 41   
F.4 Inference Ablations . 42   
F.5 Data Ablations . 44   
F.6 Training Analysis 46   
G Case Studies 48   
G.1 Sokoban 48   
G.2 MiniGrid . 49   
G.3 Crafter . 50   
G.4 Procgen Miner . 51   
H Discussion 52   
H.1 Inference Cost . 52   
H.2 Interpretation and Scope . 52

## A ADDITIONAL RELATED WORK

## A.1 PLAN-ACT AGENTS

Plan-act systems separate sub-goal selection from action execution, with prior work addressing global guidance, constrained execution, and hierarchical learning (Erdogan et al., 2025; Li et al., 2026c; Jeong et al., 2026; Hu et al., 2025b). Trajectory relabeling and multi-agent training also turn past interactions into supervision (Li et al., 2026b; Zhao et al., 2025a; Motwani et al., 2025; Subramaniam et al., 2025). ConPAct focuses on whether the two roles interpret the same observed state compatibly, and uses their explicit contradictions to guide correction and training.

## A.2 STATE TRACKING AND VERIFICATION IN LLM AGENTS

State-grounded reasoning and belief updates support agent decisions (Zawalski et al., 2025; Wang et al., 2025b; Kim and won Hwang, 2025; Wang et al., 2026a;b). Other methods guide revision with critiques, process scores, or checks against planning criteria (Madaan et al., 2023; Chae et al., 2025; Guo et al., 2024; Xu et al., 2026). ConPAct-I instead compares the roles’ reports directly: neither report is treated as ground truth, and detecting a contradiction requires no additional verifiermodel call. The teacher review used to curate ConPAct-R targets is a separate offline training-data operation.

## A.3 COMMON GROUND AND BELIEF ALIGNMENT

Common ground concerns the information needed for coordinated interaction (Clark and Brennan, 1991). Recent work examines inter-agent failures, belief-sensitive dialogue, and shared understanding (Cemri et al., 2025; Qiu et al., 2024; Chen et al., 2026; Dongre and Hakkani-Tur, 2026). Our ¨ setting isolates disagreements about the same task-relevant observed facts at planner–actor handoffs and evaluates both their reconciliation and the correctness of the resulting agreement.

## B STATE ASSERTIONS

## B.1 PLAN-ACT PROTOCOL

Roles and observations. The planner selects a sub-goal; the actor chooses actions and reports execution status. Let t index model calls and k index executed environment actions. Several calls may share observation $o ^ { k } ;$ ; only an executed action advances the environment to o<sup>k+1</sup>. An execution segment consists of actor turns pursuing one adopted sub-goal. Actor feedback, including completion or execution difficulties, triggers replanning.

Plan-Act baseline. The planner returns a <subgoal>; the actor returns an <action> and a <status>. Its statuses are CONTINUE, ACHIEVED, IMPOSSIBLE, STUCK, and CHANGED. The actor’s proposed action executes before a non-CONTINUE status returns control to the planner. Consequently, passive assertion comparison uses the planner-to-actor handoff before that action, when both roles have the same observation; it never pairs reports across an environment transition. The ordinary interface passes sub-goals and status feedback rather than full reasoning traces.

ConPAct interface. A planner reply contains <assertion>, <message>, and <subgoal>; an actor reply additionally supplies <action> and <status>. The actor uses CONTINUE for an executable action, DISCUSS to request reconsideration, and ACHIEVED when sub-goal completion is already visible. Both DISCUSS and ACHIEVED require action NONE and leave the observation unchanged. The planner then retains or revises its sub-goal using the actor’s feedback. ConPAct therefore supports same-observation comparisons in both handoff directions. Overall task completion is determined by the environment, not by the actor’s completion claim.

Independent measurement and reconciliation. The receiving role does not see the other role’s assertion when the initial pair is elicited; ordinary sub-goals and feedback remain available. After an initial contradiction is detected, reconciliation exposes the disputed fields and the roles’ reports for reassessment. All actions remain paused during correction. ConPAct-I returns the last actor reply when correction ends and does not insert an additional actor call solely to choose a replacement action. The ordinary controller applies its action/status validity rules to this reply; NONE does not execute an environment action. Both roles retain up to three observation frames, with methodspecific frame selection and text history described in Appendix C.3.

## B.2 ASSERTION FORMAT

Scope and representation. A state assertion is a JSON object enclosed in <assertion> and </assertion> tags, reported alongside the role’s planning or execution response. Both roles use the same environment-specific schema and describe the current observation, before the proposed action is executed. The fields capture visible conditions relevant to interpreting sub-goals: object counts and local obstacles in grid tasks, interaction prerequisites and resources in Crafter, visible targets and threats in Procgen, and application context and task evidence in OSWorld. They are compact reports of state interpretation, not complete scene descriptions or predictions of action outcomes. Research on chain-of-thought explanations shows that verbal reports need not faithfully expose a model’s reasoning (Turpin et al., 2023). We therefore interpret state assertions as observable proxies and do not assume that they reveal the models’ internal beliefs. No schema requires persistent object identifiers, absolute world coordinates, or access to hidden simulator state. Task instructions and current observation metadata may disambiguate what to inspect, but unavailable facts should not be filled in from earlier frames. Prompt templates are provided in Appendix E.3. Categorical domains are listed below. Count fields are nonnegative integers unless a narrower range is explicitly stated; no additional global upper bound is imposed.

Sokoban. The four fields describe puzzle progress and the player’s immediate surroundings. box count counts all boxes, including boxes on targets; target count counts all target locations, including occupied targets; and on goal count counts boxes currently on targets. All three are nonnegative integers. player neighbors is a four-element array in the fixed screen-relative order up, right, down, left, with values cell, wall, box, target, or box on target. Here cell means empty floor, and target means a target without a box. These fields are read from the full board and distinguish completion conditions from local movement or pushing prerequisites; they do not encode whether an entire push sequence is feasible.

Crafter. The five fields summarize the local interaction target and the heads-up display (HUD). facing is one of up, right, down, or left in screen coordinates. front identifies the single cell in that direction: walkable, water, lava, stone, tree, coal, iron, diamond, table, furnace, cow, zombie, skeleton, plant, or arrow. walkable merges empty grass, sand, and path; plant covers both growth stages; an arrow must occupy that cell. The moving camera does not establish fixed world coordinates. vitals is the ordered array [health, food, drink, energy]. inventory has exactly the keys wood, stone, coal, and iron. tools has exactly six keys: wood pickaxe, stone pickaxe, iron pickaxe, wood sword, stone sword, and iron sword. Every vital, material count, and tool count is an integer in $\{ 0 , \ldots , 9 \}$ ; a material or tool absent from the HUD is reported as zero. The schema covers resource and tool prerequisites and immediate survival needs, while omitting other facts such as the full arrangement of crafting stations.

MiniGrid / BabyAI. The five fields describe visible keys, door states, and the next cell relative to the agent’s orientation. key count counts keys drawn on the board; a carried key is not drawn and is excluded. front cell is the cell directly ahead of the agent triangle, using the direction supplied with the current screenshot. Its values are empty, wall, door locked, door closed, door open, key, ball, box, goal, or lava; the outer boundary is wall. locked door count, closed door count, and open door count count locked, closedbut-unlocked, and open doors, respectively. Their visual cues are a filled cell with a keyhole, a hollow double-edged square, and a thin strip along a cell edge. All four counts are nonnegative integers and cover both light and dark rendered cells in the full-board image, not only the highlighted field of view. These fields describe visible movement and interaction conditions; zero visible keys does not establish that the agent is carrying a key.

Procgen. The schema is selected by game identity. Every game has target visible; every game except maze also has hazard nearby. Both take yes, no, or unknown, using the referents in Table 3. A hazard is nearby when it is visibly close enough to threaten the player or the next immediate movement; the protocol does not impose a pixel-distance threshold. Miner additionally has diamonds visible, a nonnegative integer counting only diamonds in the screenshot, or JSON null when a reliable count is unavailable. Bigfish additionally has my size: smallest when all other visible fish are larger, middle when both larger and smaller fish are visible, biggest when none is larger, and unknown when comparison is unclear or no other fish is visible. no requires sufficient visible evidence of absence; an obscured or ambiguous view warrants unknown. An off-screen target may still exist, and the absence of a visible threat does not prove that a route is safe.

Table 3: Game-specific referents for Procgen assertions. Maze has no hazard nearby field.
<table><tr><td>Game</td><td>Target for target_visible</td><td>Threat for hazard_nearby</td></tr><tr><td>bigfish</td><td>At least one visibly smaller fish</td><td>A larger fish</td></tr><tr><td>chaser</td><td>At least one uncollected green orb</td><td>A non-vulnerable enemy</td></tr><tr><td>coinrun</td><td>The goal coin</td><td>A saw, enemy, or deadly gap</td></tr><tr><td>jumper</td><td>The carrot</td><td>Spikes</td></tr><tr><td>maze</td><td>The cheese</td><td>Not included in the schema</td></tr><tr><td>miner</td><td>At least one diamond</td><td>A boulder or diamond that could fall onto the player</td></tr><tr><td>ninja</td><td>The goal mushroom</td><td>A bomb or dangerous gap</td></tr></table>

OSWorld. The four fields refer to the current desktop screenshot and the original task, rather than the planner’s temporary sub-goal. active app identifies the foreground application: chrome, writer, calc, impress, gimp, vscode, thunderbird, vlc, files, terminal, settings, desktop, other, or unknown. An application-owned dialog is attributed to its owner when identifiable; other denotes an identifiable application outside the list. overlay takes none, menu, dialog, or unknown, with a visible dialog taking precedence over a menu. A file picker is a dialog, not the files application. target visible takes yes, partial, no, or unknown: all, some, none, or an indeterminate amount of the task’s target objects or regions are visible. goal state takes met, unmet, or unknown. unmet requires a visibly unsatisfied requirement; met requires the screenshot alone to establish every requirement; otherwise the value is unknown. This visual judgment is distinct from the official task score: a filled input or closed dialog does not prove that a file was saved or that off-screen content is correct.

Extraction, normalization, and validity. The parser extracts the tagged JSON object and checks field types, required keys, array lengths, and the domains above. Integer fields reject booleans, floating-point values, and numeric strings; they are not coerced or clipped. Crafter, MiniGrid, Procgen, and OSWorld require exactly one assertion block, reject duplicate JSON keys at any nesting level, and require exactly the specified fields, including Crafter’s nested keys. Their categorical values must match the listed strings exactly, and Markdown fences inside the assertion block are not accepted. The Sokoban schema extractor instead uses the last assertion block, removes optional surrounding JSON fences, retains only its four required fields, and trims and lowercases neighbor labels; thus the prompt spelling Box on target becomes box on target. Its JSON decoder uses the last occurrence of a duplicate key. ConPAct additionally validates the order and uniqueness of the complete reply’s protocol tags, so this schema-level tolerance does not permit repeated assertion blocks in a ConPAct reply. Missing fields or invalid values yield an invalid assertion, not an all-zero or unknown-filled replacement; in the online runners, a required assertion that fails parsing causes a model-output failure. Schema validity only establishes structural admissibility: for example, the Sokoban parser does not enforce cross-field constraints between the three counts, and it does not verify the report against the board.

Definite values and uncertainty. Only Procgen and OSWorld provide unknown sentinels: the literal string unknown for their categorical fields, and null for Procgen miner’s diamonds visible. Sokoban, Crafter, and MiniGrid require definite values in every field. Zero is a definite count, and OSWorld’s none, other, and partial are definite categories, not abstentions. Comparison uses parsed values rather than raw JSON text: object-key order is irrelevant, array order matters, and each top-level slot is compared as a whole, including Crafter’s arrays and nested objects. For a valid pair, a slot is comparable only if neither role reports its declared unknown sentinel. The result is CONFLICT if any comparable slot differs, AGREE if every slot is definite and matches, and INDETERMINATE otherwise. Thus, matching unknowns do not establish agreement, while a definite contradiction still counts as a conflict even if another slot is unknown. These comparisons require no ground truth; the code does not provide a reliable assertion-correctness oracle for Procgen or OSWorld, and task scores are not substituted for one.

Pairing and measurement stages. Assertions are paired by environment step, using the handoff protocol above. An initial pair is elicited without exposing the other role’s assertion; later reconciliation pairs can use the shared discussion context. Reports across an executed action are never paired. Passive mismatch measurement does not return discrepancy feedback or insert corrective calls. The training analyses distinguish initial judgments from the reports available after model-initiated discussion (Appendix F.6).

## B.3 METRICS

Let $\{ \tau _ { i } \} _ { i = } ^ { N }$ denote the evaluated trajectories for one model–environment configuration. Repeated rollouts are separate trajectories. Infrastructure failures are excluded; model-output failures and budget exhaustion remain included. The following rates are computed over their specified eligible samples and displayed as percentages. The mismatch analyses combine the 200 Sokoban and 150 MiniGrid levels before applying these metric-specific denominators; the main benchmark tables use the environment-specific aggregation rules below.

State mismatch. Let $\mathcal { P }$ contain the evaluated handoff pairs $( i , t )$ for which both assertions are valid and every field is definite. Passive measurement uses the independent initial pairs; inference corrections and training analyses explicitly identify their initial or post-discussion event sets. Each pair describes the same observation, $o _ { i , t } = o _ { i , t + 1 }$ . Using $d _ { i , t }$ from Eq. 1, we define

$$
D = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , t ) \in \mathcal { P } } d _ { i , t } .\tag{2}
$$

Each eligible pair has equal weight, regardless of its trajectory length or outcome. Discussionconditional and paired pre/post statistics use the event sets specified in Appendix F.6.

State correctness. Within pair $( i , t )$ , let $\mathbf { v } _ { i , t } ^ { \mathrm { p l a n } }$ and $\mathbf { v } _ { i , t } ^ { \mathrm { a c t } }$ denote the two assertions, indexed by role. Let $\mathbf { v } _ { i , t } ^ { * }$ be the ground-truth assertion for their shared observation. Define $\mathcal { G } \subseteq \mathcal { P }$ as the pairs with valid, fully definite ground truth. For $r \in \{ \mathrm { p l a n } , \mathrm { a c t } \}$ , exact-match accuracy is

$$
A ^ { r } = \frac { 1 } { | { \mathcal G } | } \sum _ { ( i , t ) \in { \mathcal G } } \mathbf { 1 } \big [ \forall f \in { \mathcal F } : { \mathbf { v } } _ { i , t } ^ { r } [ f ] = { \mathbf { v } } _ { i , t } ^ { * } [ f ] \big ] .\tag{3}
$$

All fields must match; there is no partial credit. Both roles use the same denominator. Sokoban ground truth is obtained by replaying executed actions from the initial board. MiniGrid ground truth is extracted from the recorded initial state and subsequent environment states, aligned with the current observation. Keys and doors are counted on the board; the front cell is determined by the agent’s position and direction. Carried keys are excluded. Ground truth follows the schemas in Appendix B.2 and is hidden from both roles during passive measurement.

Sub-goal adherence. Let $q _ { i , j }$ be the judge’s verdict for execution segment $j$ of $\tau _ { i } .$ . Let $\mathcal { I } = \{ ( i , j )$ $q _ { i , j } \in \{ \mathrm { F O L L O W } , \mathrm { V I O L A T E } \} \mathrm { \ : \it \Omega }$ }. We define

$$
H = \frac { 1 } { | \mathcal { T } | } \sum _ { ( i , j ) \in \mathcal { I } } \mathbf { 1 } [ q _ { i , j } = \mathrm { F O L L O W } ] .\tag{4}
$$

Each segment receives one verdict and equal weight. We use GPT-5.6-sol (OpenAI, 2026c) as an LLM judge (Zheng et al., 2023). Our rubric targets plan–action adherence, an aspect of agent alignment examined by Agent GPA (Jia et al., 2025). Its input contains the assigned sub-goal and, for each segment step, the ground-truth state as text followed by the actor’s complete response. It receives no screenshots, planner reasoning, or other segments. FOLLOW means that the actions pursue the sub-goal, even if the actor misreads the state or fails to complete it. VIOLATE means that the actions contradict the sub-goal or pursue another goal. UNKNOWN is reserved for sub-goals with no executable content; it is not used for close decisions. Segments truncated by termination, budget exhaustion, or model-output failure are judged from the available trace. Adherence does not require a valid assertion pair.

Task success. A trajectory succeeds when the environment confirms task completion within the rollout budget. We define

$$
S = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ \tau _ { i } \ \mathrm { s u c c e d s } ] .\tag{5}
$$

An actor’s completion claim alone does not establish success. This trajectory-level aggregation is used for the state analyses; environment-specific benchmark metrics are defined below.

Exclusions and empty denominators. Invalid assertions and pairs with any unknown field are excluded from D and both correctness rates. A definite conflict in a partially unknown pair is recorded separately and does not enter D. Missing ground truth excludes a pair from correctness only. Missing sub-goals, UNKNOWN verdicts, and missing or malformed judge responses are excluded from $H .$ . These exclusions do not remove a trajectory from S. Eligible counts and exclusion counts are recorded separately. An empty denominator yields an undefined metric, not zero. Trajectory-level $D _ { i }$ and $H _ { i }$ use the same definitions restricted to $\tau _ { i }$

Sokoban. For each of the $L = 2 0 0$ puzzles, let $c _ { \ell } \in \{ 0 , \ldots , 8 \}$ be the number of successful rollouts. A rollout succeeds when every box is on a target. We report

$$
\arg \ @ 8 = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \frac { c _ { \ell } } { 8 } , \qquad \mathrm { p a s s } \ @ 8 = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \mathbf { 1 } [ c _ { \ell } \geq 1 ] , \qquad \mathrm { p a s s } ^ { 8 } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \mathbf { 1 } [ c _ { \ell } = 8 ] .\tag{6}
$$

All three are displayed as percentages. The latter two specialize the pass@k criterion (Chen et al., 2021) and the $\mathrm { p a s s } ^ { k }$ reliability criterion (Yao et al., 2025) to eight attempts: at least one success and success on all eight attempts, respectively.

Crafter. Let $a _ { j , i }$ indicate whether episode j unlocks achievement $i ,$ with $i \in \{ 1 , \ldots , 2 2 \}$ , and let N denote the number of evaluated episodes. The achievement success rate is $s _ { i } = \textstyle { 1 0 0 N ^ { - 1 } \sum _ { j } a _ { j , i } }$ We compute

$$
{ \mathrm { S c o r e } } = \exp \left( { \frac { 1 } { 2 2 } } \sum _ { i = 1 } ^ { 2 2 } \log ( 1 + s _ { i } ) \right) - 1 , \qquad { \mathrm { P r o g r e s s } } = { \frac { 1 0 0 } { 2 2 N } } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { 2 2 } a _ { j , i } .\tag{7}
$$

Score follows the Crafter benchmark (Hafner, 2022) and is computed from the achievement success rates over the full evaluation set. Progress is the mean proportion of unlocked achievements, following BALROG (Paglieri et al., 2025b). Return is the mean undiscounted sum of native rewards per episode, including rewards associated with changes in health.

Procgen. For game g, we average the undiscounted episode returns over its 20 levels and eight rollouts per level, obtaining $\bar { G } _ { g }$ . The normalized return is

$$
z _ { g } = \frac { \bar { G } _ { g } - R _ { \mathrm { m i n } , g } } { R _ { \mathrm { m a x } , g } - R _ { \mathrm { m i n } , g } } ,\tag{8}
$$

using the easy-mode reference bounds from Cobbe et al. (2020), listed in Table 5. Returns include rewards up to environment termination or budget exhaustion, and normalized values are left unclipped. We use the rliable aggregation functions (Agarwal et al., 2021) on the run-by-game score matrix: IQM is the 25%-trimmed mean of its entries, Mean averages the per-game means, and Median takes the median of the per-game means. For a single evaluated checkpoint, the matrix has one row and seven columns. All three metrics are multiplied by 100 for display.

MiniGrid / BabyAI. Success means completing the environment’s mission; termination through collision, failure, or timeout does not count as success. For each task ID $e ,$ we compute success rate $\mathrm { S R } _ { \epsilon }$ and mean native episode return $\bar { G } _ { e }$ over all of its instances and rollouts. The headline metrics are

$$
\mathrm { S R } = { \frac { 1 0 0 } { 2 7 } } \sum _ { e = 1 } ^ { 2 7 } \mathrm { S R } _ { e } , \qquad \mathrm { R e t u r n } = { \frac { 1 } { 2 7 } } \sum _ { e = 1 } ^ { 2 7 } { \bar { G } } _ { e } ,\tag{9}
$$

where $\mathrm { S R } _ { e }$ is a fraction in [0, 1]. This gives each task ID equal weight despite different instance counts. Steps is the mean number of executed actions pooled over successful episodes only; it is undefined when there are no successes. We retain native rewards and termination rules, including collision penalties and the native time-dependent success reward. The external action budget does not replace the environment’s native maximum episode length in its reward formula.

Rollout accounting. All eight game-environment rollouts contribute to the reported metrics. Modeloutput failures and budget exhaustion retain the outcome and rewards obtained before stopping. Infrastructure failures are recorded separately and their missing task–rollout slots are rerun; complete metrics require all declared slots. A terminal action’s reward and achievements are included before the rollout ends.

OSWorld. We evaluate each of the 361 OSWorld tasks with three rollouts and report mean evaluation score and full success rate as Score and SR, respectively, both in percent. For each task, we average the evaluation scores and full-success indicators across its three rollouts, then average these task-level values over the evaluation set. Overall results therefore give each task equal weight, so category contributions are proportional to their task counts. These task-performance measures are distinct from the state-assertion correctness used in the state analyses. The category-level results are provided in Appendix F.3.

## C EXPERIMENTAL SETUP

## C.1 ENVIRONMENTS

Table 4 summarizes the source-task pools used for data collection and the fixed evaluation sets. Each game-environment evaluation instance is run eight times, with model histories cleared and the environment reset to the specified instance before each rollout. Source-instance counts describe the tasks used to collect teacher trajectories; response-level training counts depend on filtering and reconstruction. Models are trained separately for each environment.

Table 4: Environment splits and evaluation budgets. A limits executed actions and C limits total logical model calls per rollout. Each instance has eight evaluation rollouts. MiniGrid budgets are specified per task in Table 6.
<table><tr><td>Environment</td><td>Source instances</td><td>Evaluation instances</td><td>Evaluation rollouts</td><td>A</td><td>C</td></tr><tr><td>Sokoban</td><td>100</td><td>200</td><td>1,600</td><td>60</td><td>60</td></tr><tr><td>Crafter</td><td>100</td><td>200</td><td>1,600</td><td>160</td><td>160</td></tr><tr><td>Procgen</td><td>100</td><td>140</td><td>1,120</td><td>160</td><td>160</td></tr><tr><td>MiniGrid / BabyAI</td><td>108</td><td>150</td><td>1,200</td><td>60-200</td><td>60-200</td></tr></table>

Sokoban. We use the GamingAgent Sokoban implementation (Hu et al., 2025a) through our sandbox interface. Our fixed Stage 1.5 evaluation split contains 100 one-box and 100 two-box puzzles. Its boards comprise 68 puzzles of size 6 × 6, 66 of size 7 × 7, and 66 of size 8 × 8. The source pool contains 50 one-box and 50 two-box puzzles selected from the training split. The agent observes the full board image enlarged by a factor of four with nearest-neighbor interpolation and chooses one of four directions: up, down, left, or right. The task ends successfully when all boxes occupy targets. An action denotes one agent-level directional move; the sandbox implements it through a push command followed by the corresponding direction command.

Crafter. For Crafter (Hafner, 2022), we use source seeds 0–99 and evaluation seeds 4000–4199. The agent collects resources, crafts tools, and survives while unlocking 22 achievements. It observes the native 640 × 640 local view with the heads-up display (HUD). The action space contains 17 actions for movement, interaction, sleeping, placing objects, and crafting tools or weapons. Each rollout uses a fresh environment instance with the specified seed and the same reset sequence. A rollout ends at native environment termination or when a budget is exhausted; partial achievement progress and accumulated reward are retained.

Procgen. We evaluate seven Procgen games (Cobbe et al., 2020) in easy mode: bigfish, chaser, coinrun, jumper, maze, miner, and ninja. For every game, the evaluation levels have start level=100–119 and num levels=1. The 100 source instances use levels selected from 0–49, with the allocation in Table 5. The 512 × 512 sandbox screenshot is converted to RGB and enlarged by a factor of two using nearest-neighbor interpolation, yielding a 1024 × 1024 model input. Actions use a game-specific subset of the 15 native action indices. The adapter introduces no action repeat. A rollout stops at the first episode termination or budget limit; any automatic reset to a new episode is excluded from the completed rollout.

Table 5: Procgen task allocation and easy-mode normalization bounds from Cobbe et al. (2020). Each game has 20 evaluation levels with IDs 100–119. Source levels are selected from IDs 0–49.
<table><tr><td>Game</td><td>Source levels</td><td>Evaluation levels</td><td> $R _ { \mathrm { m i n } }$ </td><td> $R _ { \mathrm { m a x } }$ </td></tr><tr><td>bigfish</td><td>14</td><td>20</td><td>1</td><td>40</td></tr><tr><td>chaser</td><td>14</td><td>20</td><td>0.5</td><td>13</td></tr><tr><td>coinrun</td><td>15</td><td>20</td><td>5</td><td>10</td></tr><tr><td>jumper</td><td>14</td><td>20</td><td>3</td><td>10</td></tr><tr><td>maze</td><td>14</td><td>20</td><td>5</td><td>10</td></tr><tr><td>miner</td><td>15</td><td>20</td><td>1.5</td><td>13</td></tr><tr><td>ninja</td><td>14</td><td>20</td><td>3.5</td><td>10</td></tr></table>

MiniGrid / BabyAI. The evaluation set comprises 150 instances from 27 MiniGrid and BabyAI task IDs (Chevalier-Boisvert et al., 2023; 2019), including navigation, object retrieval, door unlocking, object placement, and obstacle avoidance. Table 6 lists the exact seeds and budgets. The source pool contains four instances per task ID, for 108 instances in total, with selected seeds between 20 and 38. The source and evaluation manifests have no overlap in task–seed pairs or recorded initial-layout hashes. Layouts are fixed using MiniGrid 2.3.1 and a single seeded reset per fresh instance. We render the full board with 32-pixel tiles and visibility highlighting, convert it to RGB, and enlarge it by a factor of three using nearest-neighbor interpolation. For example, 5 × 5 and 8 × 8 boards yield 480 × 480 and 768 × 768 images. The current mission and agent direction are provided as text alongside the image. The action vocabulary is left, right, forward, pickup, drop, toggle, and done, restricted to the native action space of each task; Dynamic-Obstacles permits only left, right, and forward. Native termination and truncation are preserved.

Table 6: MiniGrid and BabyAI evaluation tasks. Seed offsets are added to 1000; each resulting instance has eight rollouts. The last column gives the common action and model-call limit for each rollout.
<table><tr><td>Task ID</td><td>Instances</td><td>Seed offsets</td><td> $A = C$ </td></tr><tr><td>BabyAI-BlockedUnlockPickup-v0</td><td>5</td><td>0,1, 2, 3,4</td><td>136</td></tr><tr><td>BabyAI-GoTo-v0</td><td>5</td><td>0,1, 2, 3,4</td><td>200</td></tr><tr><td>BabyAI-GoToDoor-v0</td><td>5</td><td>0,1,2,3,4</td><td>84</td></tr><tr><td>BabyAI-GoToLocal-v0</td><td></td><td>50,1,2, 3,4</td><td>60</td></tr><tr><td>BabyAI-GoToObj-v0</td><td></td><td>50,1,2, 3,4</td><td>60</td></tr><tr><td>BabyAI-GoToRedBall-v0</td><td></td><td>50,1,2,3,4</td><td>60</td></tr><tr><td>BabyAI-GoToRedBallGrey-v0</td><td></td><td>50,1,2,3,4</td><td>128</td></tr><tr><td>BabyAI-GoToRedBallNoDists-v0</td><td></td><td>50,1,2,3,4</td><td>60</td></tr><tr><td>BabyAI-GoToRedBlueBall-v0</td><td></td><td>60,1,2,3,4,5</td><td>68</td></tr><tr><td>BabyAI-UnlockPickup-v0</td><td></td><td>60,1,2,3,4,5</td><td>80</td></tr><tr><td>MiniGrid-BlockedUnlockPickup-v0</td><td></td><td>60,1,2,3,4,5</td><td>192</td></tr><tr><td>MiniGrid-DoorKey-5x5-v0</td><td></td><td>6 0,1, 3, 4,7, 10</td><td>92</td></tr><tr><td>MiniGrid-DoorKey-6x6-v0</td><td></td><td>6 0,1, 2, 4, 5, 6</td><td>60</td></tr><tr><td>MiniGrid-DoorKey-8x8-v0</td><td></td><td>60,1,2,3,4,5</td><td>60</td></tr><tr><td>MiniGrid-Dynamic-Obstacles-5x5-v0</td><td>2 3,7</td><td></td><td>60</td></tr><tr><td>MiniGrid-Fetch-5x5-N2-v0</td><td></td><td>60,1,2,3,4,5</td><td>60</td></tr><tr><td>MiniGrid-Fetch-8x8-N3-v0</td><td></td><td>60,1,2,3,4,5</td><td>60</td></tr><tr><td>MiniGrid-FourRooms-v0</td><td></td><td>60,1,2,3,4,5</td><td>68</td></tr><tr><td>MiniGrid-GoToDoor-5x5-v0</td><td></td><td>60,1,2,3,4,5</td><td>84</td></tr><tr><td>MiniGrid-GoToObject-6x6-N2-v0</td><td></td><td>60,1,2,3,4,5</td><td>60</td></tr><tr><td>MiniGrid-LavaCrossingS9N1-v0</td><td></td><td>64,5,6,7,9, 13</td><td>60</td></tr><tr><td>MiniGrid-PutNear-6x6-N2-v0</td><td></td><td>60,1,2,3,4,5</td><td>60</td></tr><tr><td>MiniGrid-PutNear-8x8-N3-v0</td><td></td><td>60,1,2,3,4,5</td><td>60</td></tr><tr><td>MiniGrid-RedBlueDoors-6x6-v0</td><td></td><td>60,2,3,4,5,6</td><td>60</td></tr><tr><td>MiniGrid-RedBlueDoors-8x8-v0</td><td></td><td>60,1,2,3,4,5</td><td>124</td></tr><tr><td>MiniGrid-Unlock-v0</td><td></td><td>60,1,2,3,4,5</td><td>60</td></tr><tr><td>MiniGrid-UnlockPickup-v0</td><td></td><td>60,1,2, 3,4,5</td><td>76</td></tr></table>

## C.2 MODELS AND BASELINES

Models. The mismatch analyses cover Claude Opus-5 and Sonnet-5 (Anthropic, 2026a;b); GPT-5.6-sol, GPT-5.6-terra, and GPT-5.5 (OpenAI, 2026c;b); Kimi K3 and K2.6 (Kimi Team, 2026; Moonshot AI, 2026); and the Qwen model sizes listed in the result tables (Qwen Team, 2026b;a; Bai et al., 2025). Main inference comparisons use the four heterogeneous pairs in Table 1. Training uses Qwen3-VL-8B and Qwen3-VL-32B in homogeneous configurations and a 32B planner with an 8B actor. GPT-5.6-sol is the teacher for data construction and the judge for annotation and offline target review.

Inference baselines. ReAct (Yao et al., 2023) selects one environment action per response; Re-Act (Planner) and ReAct (Actor) use the corresponding role’s model alone. Plan-Act (Erdogan et al., 2025) uses the dynamic sub-goal/status protocol in Appendix B.1 without assertion-based correction. HiPlan (Li et al., 2026c) supplies a global milestone guide and local hints to the actor; the planner model produces the guide and hints. TAPE (Jeong et al., 2026) uses candidate plans and a state–action graph to constrain execution, replanning when the current observation no longer matches the planned state. All decision-making calls, including internal planning and candidate generation, consume the shared rollout budget.

Supervised baselines. ReAct SFT supervises a single model’s reasoning and actions on retained teacher demonstrations. Plan-Act SFT supervises planner sub-goals and actor actions/statuses and retains dynamic replanning at evaluation. PAA (Erdogan et al., 2025) reconstructs a fixed plan from an execution trace: the planner is called at initialization and the actor follows that plan. HSL (Li et al., 2026b) relabels achieved goals and supervises goal-relevant recorded actions. WebSTAR (He et al., 2026) applies teacher-based step filtering to the source demonstrations. ECoT (Zawalski et al., 2025) augments planning and action targets with state-grounded descriptions, without comparing the roles’ assertions or adding ConPAct reconciliation. All training methods use the same optimization settings and the token-budget matching described below.

## C.3 TRAINING AND EVALUATION

Teacher and student models. We use GPT-5.6-sol (OpenAI, 2026c) with its default thinking effort as the teacher for constructing supervision. Students are initialized from Qwen3-VL-8B-Instruct or Qwen3-VL-32B-Instruct (Bai et al., 2025). For homogeneous ConPAct-S/R configurations, planner and actor examples are pooled to fine-tune a single checkpoint, which serves both roles at evaluation. For the heterogeneous configuration, the 32B planner and 8B actor are fine-tuned separately on their respective role-specific examples.

Teacher trajectories and data selection. For each source instance in Table 4, GPT-5.6-sol generates two rollouts. The ReAct demonstration pool retains at most one successful or qualifying progress trajectory per source instance; these demonstrations supply ReAct-based supervised baselines and the recorded traces for ConPAct-R. ConPAct-S instead samples both planner and actor with GPT-5.6-sol under ConPAct-I and selects consistent collaboration or repaired segments from both successful and failed trajectories. At most one sampled collaboration trajectory per source instance supplies these segments. Thus, trajectory-level demonstration selection and response-level consistency filtering are separate operations. ConPAct-R uses GPT-5.6-sol to reconstruct sub-goals and corrective interactions from the retained ReAct traces and to review candidate targets using only the student’s available input.

Environment-specific training and matched budgets. We train a separate checkpoint for each environment, using training seed 42 and selecting the final checkpoint. Homogeneous configurations pool planner and actor examples into one model; heterogeneous configurations train the 32B planner and 8B actor on their role-specific examples. For each environment and student size, three epochs of ReAct SFT define the reference training-token budget. The other training methods and data ablations are matched to that total token budget using the same optimization settings. This matches training exposure across datasets with different amounts of dialogue and supervision.

Optimization. We use ms-swift (Zhao et al., 2025b) with AdamW (Loshchilov and Hutter, 2019), cosine decay (Loshchilov and Hutter, 2017), FlashAttention (Dao et al., 2022), gradient checkpointing (Chen et al., 2016), and ZeRO (Rajbhandari et al., 2020). Only the language-model backbone is updated; the vision encoder and projector are frozen. The effective batch size is 64 on eight H200 GPUs. For the shorter sequence setting, 8B/32B use per-device microbatches of 8/4 and gradient accumulation of 1/2; the longer setting uses microbatch 1 and accumulation 8 for both sizes. Examples exceeding the sequence limit are discarded, and sequence packing is disabled. The complete settings are listed in Table 7.

Table 7: Supervised fine-tuning settings shared by the training methods. Token budgets are matched to the three-epoch ReAct SFT reference for each environment and model size.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Trainable parameters</td><td>Language-model backbone; vision encoder/projector frozen</td></tr><tr><td>Reference schedule Training seed</td><td>3 epochs of ReAct SFT; token-matched across methods 42</td></tr><tr><td>Observation window</td><td>3 frames, including the current observation</td></tr><tr><td>Optimizer</td><td>AdamW (fused)</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Cosine decay</td></tr><tr><td>Warmup ratio</td><td>0.05</td></tr><tr><td>Adam  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9,0.95)</td></tr><tr><td>Adam €</td><td>10-8</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Maximum gradient norm</td><td>1.0</td></tr><tr><td>Effective batch size</td><td></td></tr><tr><td></td><td>64 examples</td></tr><tr><td>Training hardware</td><td>8 NVIDIA H200 GPUs</td></tr><tr><td>Training precision</td><td>BF16</td></tr><tr><td>Maximum sequence length</td><td>6,144 (Sokoban, Crafter); 16,384 (Procgen, MiniGrid)</td></tr><tr><td>Maximum image tokens</td><td>1,024 per image</td></tr><tr><td>Attention backend</td><td>FlashAttention</td></tr><tr><td>Gradient checkpointing</td><td>Enabled</td></tr><tr><td>DeepSpeed</td><td>ZeRO-1 (8B); ZeRO-3 (32B)</td></tr><tr><td>Sequence packing</td><td>Disabled</td></tr><tr><td></td><td></td></tr><tr><td>Checkpoint selection</td><td>Final checkpoint</td></tr></table>

Loss masks. Each training example supervises one complete assistant response. System instructions, user text, images, and earlier interaction context receive no loss. All tokens in the retained response receive the same weight, including state assertions and role-specific output fields. Table 8 specifies which responses are retained by each construction method. An excluded response may remain in the input history but is not used as a target.

Table 8: Response selection for supervised training. Loss is applied only to the retained target response.
<table><tr><td>Method</td><td>Supervised targets</td></tr><tr><td>ReAct / Plan-Act / ECoT</td><td>Complete valid responses from retained demonstrations or progress prefixes.</td></tr><tr><td>PAA</td><td>The planner&#x27;s annotated plan and the actor&#x27;s action-only responses.</td></tr><tr><td>HSL</td><td>Expert responses and goal-relevant actions from hindsight examples; irrelevant actions are excluded.</td></tr><tr><td>WebSTAR</td><td>Original responses for actions with judge scores greater than 5; other responses are excluded.</td></tr><tr><td>ConPAct-S</td><td>Valid responses from consistent segments and successful corrections; conflict- ing replies remain context only.</td></tr><tr><td>ConPAct-R</td><td>Validated responses under the same consistency rule; hypothetical errors and recorded nonoptimal actions are excluded from targets.</td></tr></table>

Decoding and output limits. For game evaluation, each model call requests one completion with at most 4,000 output tokens. Self-hosted Qwen models use temperature 1.0, top-p = 1.0, and topk = −1, which disables top-k truncation. Thinking is disabled through the chat template. For the Claude, GPT, and Kimi API configurations, temperature, top-p, and top-k use the provider defaults, and the request disables thinking. These evaluation settings apply to both roles and to methodinternal model calls. Teacher calls for data construction use the default thinking effort specified above.

Observation and message history. All methods and roles use a maximum of three observation frames at training and evaluation, including the current frame. When fewer frames are available, we use all available frames without padding. Table 9 specifies the frame selection for each method. Plan-Act and ECoT retain the text of older sub-goals after their start frames leave the planner’s window. ConPAct retains the exchanged replies on the current observation, the active planner proposal, and the last executed actor reply. After an action, other discussion from previous observations is removed.

Table 9: The common three-frame observation limit. A planner called only at initialization receives the single available frame. Textual task instructions and active plans are retained separately.
<table><tr><td>Method</td><td>Planner input</td><td>Actor input</td></tr><tr><td>ReAct / HSL / WebSTAR</td><td>No separate planner</td><td>Up to 3 recent frames</td></tr><tr><td>Plan-Act / Plan-Act SFT</td><td>Current frame plus up to 2 sub-goal-start Up to 3 recent frames frames</td><td></td></tr><tr><td>PAA</td><td>Initial frame; called once</td><td>Up to 3 recent frames</td></tr><tr><td>ECoT</td><td>Current frame plus up to 2 sub-goal-start Up to 3 recent frames frames</td><td></td></tr><tr><td>HiPlan</td><td>Initial frame for the global plan; up to 3 recent Up to 3 recent frames frames for hints</td><td></td></tr><tr><td>TAPE</td><td>Up to 3 recent frames for visual calls</td><td>Up to 3 recent frames</td></tr><tr><td>ConPAct-I / ConPAct-S/R</td><td>Up to 3 recent frames</td><td>Up to 3 recent frames</td></tr></table>

Evaluation of trained models. ConPAct-S/R use their learned discussion behavior without the system-triggered ConPAct-I intervention. The automatic assertion-checking and resampling loop, assertion injection, and action gating are disabled. The actor can initiate discussion through its predicted DISCUSS status, which hands control to the planner without executing an action. Both roles use the common three-frame observation window. All resulting model calls share the rollout budget described below.

Interaction budgets. For the four game environments, all methods use the same per-instance action and model-call limits in Tables 4 and 6, with A = C. An executed action counts toward A; planning, control handoffs, and corrective discussion do not advance the environment. Every logical mode completion used for decision-making counts toward C, including calls by either role, discussion turns, and method-internal candidate sampling or resampling. Thus, the call limit is shared across roles and is not a separate allowance for each role. Network retries are tracked separately, and offline diagnostic judgments are excluded from the decision budget. The rollout ends when the environment terminates or either budget is exhausted, after recording the last executed action’s outcome. The eight rollouts of an instance share its task configuration and initial-state seed, while histories are cleared between rollouts.

## D STATE MISMATCH ANALYSIS

## D.1 MISMATCH PREVALENCE

Evaluation pool. We analyze the combined set of 200 Sokoban and 150 MiniGrid levels, with eight rollouts per level and the 16 planner–actor configurations in Table 10. The table combines levels from both environments rather than averaging two equally weighted environment scores. Task success averages the trajectories in this combined pool; mismatch, correctness, and adherence use the eligible pairs or segments defined in Appendix B.3. This analysis-level task-success rate is distinct from the task-ID-macro-averaged MiniGrid SR in the main benchmark tables. Assertions are measured passively at same-observation handoffs, without exposing the other role’s assertion or returning conflict feedback.

Prevalence and adherence. The configurations include both homogeneous and heterogeneous model pairs. Mismatch is present even when the same model serves both roles and increases in the evaluated comparisons with a larger capability gap. For Figure $2  { \mathbf { b } } , D _ { i }$ is the trajectory’s valid-pair mismatch rate and $H _ { i }$ its judged-segment adherence rate. Only trajectories with nonempty denominators for both measures enter this comparison; trajectories sharing the same $( H _ { i } , D _ { i } )$ are grouped, with marker size showing frequency. Among trajectories with measured $H _ { i } = 1 0 0 \%$ , 82.9% contain at least one mismatch. This is a trajectory-level incidence, not the fraction of mismatched handoffs.

## D.2 CONTROL EXPERIMENTS

Visual versus textual observations. We compare screenshot input with a structured textual description of the same environment state, preserving task instructions, sub-goals, role histories, and evaluation budgets. The text describes observable board objects, positions, and orientation rather than supplying the target state assertion. Sokoban uses the character map illustrated in Figure 7; MiniGrid text represents the visible objects, door states, and agent orientation. All retained observation frames use the chosen modality. Both conditions use the combined Sokoban/MiniGrid evaluation pool and the eight-rollout budgets in Appendix C.3.

![](images/74d090b06ee3e875879bf85dd04f1f558ef454a040c392bc17714e2712218e66.jpg)  
Figure 7: Paired Sokoban observations for the initial state of level Stage 1 5 2387. The archived screenshot and the text generated from the recorded board show the same two boxes, two targets, player position, and wall layout. Neither box is on a target.

Fixed observations and repeated responses. We randomly select 300 observations from Sokoban and MiniGrid trajectories. Each observation supplies the corresponding planner and actor contexts, including the actor’s assigned sub-goal. The same selected states support the visual and textual comparisons. Within each model–modality condition, each role’s input is held fixed while generating eight responses; sampled responses neither advance the environment nor enter subsequent sampling contexts. The same model is used in both roles for this control, and the eight evaluated models use the same observation pool.

Valid-response pairing. Invalid or unparsable assertions are excluded rather than treated as an assertion category. For observation $j ,$ let $n _ { j } ^ { \mathrm { p l a n } }$ and $n _ { j } ^ { \mathrm { a c t } }$ be the numbers of valid responses. The within-role rate $d _ { j } ^ { r }$ averages disagreement over all distinct unordered pairs of valid responses from role $r ;$ the cross-role rate $d _ { j } ^ { \mathrm { c r o s s } }$ averages all $n _ { j } ^ { \mathrm { p l a n } } n _ { j } ^ { \mathrm { a c t } }$ planner–actor pairs. An observation needs at least two valid responses from each role to contribute to the comparison. The reported rates average these observation-level quantities over the same eligible observations.

Cross-role excess. The sampling baseline is the mean of the planner’s and actor’s within-role rates:

$$
E _ { \mathrm { c r o s s } } = D ^ { \mathrm { c r o s s } } - \frac { D ^ { \mathrm { p l a n } } + D ^ { \mathrm { a c t } } } { 2 } .\tag{10}
$$

Rates in Table 11 are percentages, and the excess is in percentage points. Figure 3b reports the visual excess with 95% confidence intervals; all eight intervals exclude zero. The Opus-5 excess is 22.8 percentage points.

Table 10: Visual and structured-text observations on the combined Sokoban and MiniGrid level pool. Metrics follow the analysis denominators in Appendix B.3.
<table><tr><td rowspan="2">Planner</td><td rowspan="2">Actor</td><td colspan="5">Visual</td><td colspan="5">Textual</td></tr><tr><td>D</td><td> $A ^ { \mathrm { p l a n } }$ </td><td> $A ^ { \mathrm { a c t } }$ </td><td>H</td><td>S</td><td>D</td><td> $A ^ { \mathrm { p l a n } }$ </td><td> $A ^ { \mathrm { a c t } }$ </td><td>H</td><td>S</td></tr><tr><td rowspan="3">Opus-5</td><td>Opus-5</td><td>26.5</td><td>88.2</td><td>76.4</td><td>95.8</td><td>86.3</td><td>9.6</td><td>98.4</td><td>91.7</td><td>98.3</td><td>95.9</td></tr><tr><td>Sonnet-5</td><td>64.0</td><td>87.6</td><td>37.5</td><td>94.7</td><td>33.2</td><td>11.6</td><td>94.9</td><td>89.3</td><td>95.1</td><td>72.1</td></tr><tr><td>GPT-5.5</td><td>56.6</td><td>87.1</td><td>42.4</td><td>88.5</td><td>57.0</td><td>2.9</td><td>98.8</td><td>98.3</td><td>94.7</td><td>90.4</td></tr><tr><td rowspan="3">GPT-5.6-sol</td><td>GPT-5.6-sol</td><td>41.6</td><td>53.7</td><td>45.4</td><td>80.1</td><td>71.8</td><td>2.5</td><td>98.7</td><td>98.5</td><td>59.3</td><td>85.7</td></tr><tr><td>GPT-5.6-terra</td><td>61.9</td><td>52.0</td><td>24.1</td><td>77.3</td><td>47.2</td><td>9.4</td><td>98.9</td><td>91.5</td><td>52.2</td><td>78.8</td></tr><tr><td>GPT-5.5</td><td>48.4</td><td>53.2</td><td>42.0</td><td>76.3</td><td>67.8</td><td>3.3</td><td>98.4</td><td>98.1</td><td>55.9</td><td>85.8</td></tr><tr><td rowspan="2">Kimi-K3</td><td>Kimi-K3</td><td>44.0</td><td>67.8</td><td>64.7</td><td>93.2</td><td>23.4</td><td>41.2</td><td>69.4</td><td>70.8</td><td>83.3</td><td>42.1</td></tr><tr><td>Kimi-K2.6</td><td>67.4</td><td>68.6</td><td>42.3</td><td>92.2</td><td>15.0</td><td>42.9</td><td>71.0</td><td>76.5</td><td>88.2</td><td>38.4</td></tr><tr><td rowspan="3">Qwen3.6-27B</td><td>Qwen3.6-27B</td><td>81.3</td><td>28.4</td><td>24.5</td><td>91.7</td><td>12.9</td><td>48.4</td><td>71.9</td><td>61.0</td><td>93.2</td><td>63.6</td></tr><tr><td>Qwen3.5-9B</td><td>98.3</td><td>23.3</td><td>1.6</td><td>66.6</td><td>4.6</td><td>48.4</td><td>75.7</td><td>65.0</td><td>96.0</td><td>21.7</td></tr><tr><td>Qwen3.5-4B</td><td>95.5</td><td>25.4</td><td>4.3</td><td>83.2</td><td>1.7</td><td>76.7</td><td>73.7</td><td>28.5</td><td>87.0</td><td>15.9</td></tr><tr><td rowspan="3">Qwen3-VL-32B</td><td>Qwen3-VL-32B</td><td>84.2</td><td>14.9</td><td>14.3</td><td>82.7</td><td>8.7</td><td>89.8</td><td>26.9</td><td>16.7</td><td>81.4</td><td>20.8</td></tr><tr><td>Qwen3-VL-8B</td><td>95.8</td><td>16.2</td><td>4.1</td><td>73.5</td><td>2.2</td><td>98.7</td><td>41.0</td><td>1.3</td><td>65.9</td><td>2.0</td></tr><tr><td>Qwen3-VL-4B</td><td>96.1</td><td>13.5</td><td>2.9</td><td>81.7</td><td>4.8</td><td>97.8</td><td>36.0</td><td>2.6</td><td>92.8</td><td>6.1</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>Qwen3-VL-8B</td><td>83.6</td><td>2.6</td><td>3.3</td><td>73.2</td><td>2.2</td><td>92.2</td><td>0.8</td><td>0.9</td><td>60.1</td><td>2.2</td></tr><tr><td>Qwen3-VL-4B</td><td>96.4</td><td>2.3</td><td>2.5</td><td>88.3</td><td>4.2</td><td>99.3</td><td>0.6</td><td>2.2</td><td>92.8</td><td>4.1</td></tr></table>

Table 11: Within-role and cross-role disagreement on fixed Sokoban and MiniGrid observations (%). The sampling baseline is the mean of the planner and actor rates; cross-role excess is defined in Eq. 10.
<table><tr><td>Model</td><td>Input  $D ^ { \mathrm { p l a n } }$ </td><td> $D ^ { \mathrm { a c t } }$ </td><td> $D ^ { \mathrm { c r o s s } }$ </td></tr><tr><td>Opus-5</td><td>Visual 25.8</td><td>29.1</td><td>50.3</td></tr><tr><td></td><td>Text 4.8 Visual 35.4</td><td>11.9 41.5</td><td>11.7 46.4</td></tr><tr><td>GPT-5.6-sol</td><td>Text 6.4</td><td>9.7</td><td>10.7</td></tr><tr><td>Qwen3.6-27B</td><td>Visual 83.0 Text 35.8</td><td>78.6 59.4</td><td>86.2 68.3</td></tr><tr><td>Qwen3.5-9B</td><td>Visual 96.5</td><td>94.3</td><td>97.6</td></tr><tr><td>Qwen3.5-4B</td><td>Text 43.5 Visual 93.7</td><td>51.9 92.4</td><td>56.0 96.1</td></tr><tr><td></td><td>Text 72.5</td><td>86.1</td><td>85.7</td></tr><tr><td>Qwen3-VL-32B</td><td>Visual 73.7</td><td>64.9</td><td>83.8</td></tr><tr><td></td><td>Text 79.7</td><td>84.5</td><td>92.4</td></tr><tr><td>Qwen3-VL-8B</td><td>Visual 61.8</td><td></td><td></td></tr><tr><td></td><td></td><td>56.4</td><td>87.6</td></tr><tr><td></td><td>Text 81.8</td><td>41.8</td><td>97.7</td></tr><tr><td></td><td>Visual 74.4</td><td></td><td>95.1</td></tr><tr><td>Qwen3-VL-4B</td><td>Text 82.3</td><td>80.5 66.1</td><td>80.1</td></tr></table>

## D.3 FAILURE ANALYSIS

Failure categorization. GPT-5.6-sol classifies failed trajectories from both Sokoban and MiniGrid across the configurations in Table 12. State mismatch refers to incompatible judgments about taskrelevant state; non-adherence refers to actions that do not pursue the assigned sub-goal. These two labels may co-occur, so the table separates mismatch-only, non-adherence-only, and overlapping cases. Low planning or execution capability is a separate judge-assigned category for capabilityrelated failures without either of those diagnosed coordination failures; it is not assigned merely by subtracting detected mismatches and non-adherence. Other contains cases for which the failure category cannot be determined. The aggregate shares reported in the main text are 85.0% for mismatch, 31.1% for non-adherence, and 8.4% for low planning or execution capability.

Failure shares across tasks. For each level–configuration pair with failed rollouts, we divide each category’s failure count by the total number of failures in that pair. Figure 3c weights eligible pairs equally and plots the fraction whose category share exceeds each threshold. Mismatch-only failures account for more than half of the failures in 49.4% of eligible pairs. Pairs without failures do not enter this analysis. The controlled interventions below complement these failure patterns by holding the observation, assigned sub-goal, and actor fixed while changing state information.

Table 12: Failure-category percentages on the combined Sokoban and MiniGrid analysis pool. Mismatch and non-adherence may overlap; low capability and Other are separate categories.
<table><tr><td>Planner</td><td>Actor</td><td>Mismatch only</td><td>Non-adherence only</td><td>Both</td><td>Low capability</td><td>Other</td></tr><tr><td rowspan="3">Opus-5</td><td>Opus-5</td><td>49.8</td><td>2.7</td><td>32.9</td><td>11.9</td><td>2.7</td></tr><tr><td>Sonnet-5</td><td>72.0</td><td>1.0</td><td>13.3</td><td>12.9</td><td>0.7</td></tr><tr><td>GPT-5.5</td><td>33.7</td><td>7.1</td><td>50.7</td><td>8.3</td><td>0.1</td></tr><tr><td rowspan="3">GPT-5.6-sol</td><td>GPT-5.6-sol</td><td>23.1</td><td>9.1</td><td>50.3</td><td>17.5</td><td>0.0</td></tr><tr><td>GPT-5.6-terra</td><td>41.4</td><td>3.6</td><td>38.3</td><td>16.8</td><td>0.0</td></tr><tr><td>GPT-5.5</td><td>27.5</td><td>3.3</td><td>59.7</td><td>9.5</td><td>0.0</td></tr><tr><td rowspan="2">Kimi K3</td><td>Kimi K3</td><td>41.8</td><td>3.8</td><td>6.4</td><td>33.3</td><td>14.7</td></tr><tr><td>Kimi K2.6</td><td>58.1</td><td>2.8</td><td>9.8</td><td>16.8</td><td>12.5</td></tr><tr><td rowspan="3">Qwen3.6-27B</td><td>Qwen3.6-27B</td><td>70.3</td><td>0.6</td><td>16.9</td><td>8.2</td><td>4.1</td></tr><tr><td>Qwen3.5-9B</td><td>29.6</td><td>0.1</td><td>59.3</td><td>0.8</td><td>10.3</td></tr><tr><td>Qwen3.5-4B</td><td>54.2</td><td>0.1</td><td>27.8</td><td>1.7</td><td>16.1</td></tr><tr><td rowspan="3">Qwen3-VL-32B</td><td>Qwen3-VL-32B</td><td>50.8</td><td>0.1</td><td>44.6</td><td>3.2</td><td>1.2</td></tr><tr><td>Qwen3-VL-8B</td><td>69.8</td><td>0.3</td><td>25.8</td><td>2.6</td><td>1.5</td></tr><tr><td>Qwen3-VL-4B</td><td>67.2</td><td>0.0</td><td>30.6</td><td>1.1</td><td>1.0</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>Qwen3-VL-8B</td><td>62.9</td><td>1.4</td><td>23.6</td><td>10.4</td><td>1.7</td></tr><tr><td>Qwen3-VL-4B</td><td>73.6</td><td>0.0</td><td>24.7</td><td>0.5</td><td>1.2</td></tr></table>

## D.4 STATE INTERVENTIONS

Event construction and matched conditions. The local intervention uses 300 observations randomly selected from Sokoban trajectories. GPT-5.6-sol constructs feasible or blocked sub-goals for these observations, with feasibility checked against environment-ground-truth search. Each event fixes the observation, sub-goal, and actor and is evaluated with eight sampled responses per information condition. We compare no supplementary information, a state assertion whose fields are randomized to legal values, freeform control context, and task-relevant state facts produced by Opus-5 from the observation. The supplementary information is inserted into the actor’s system prompt. The freeform control is character-matched irrelevant text for blocked sub-goals and non-critical state facts for feasible sub-goals, with length matched to the state-assertion context.

First-response outcomes. The test examines only the actor’s next response, without executing a planner handoff or corrective dialogue. For a blocked sub-goal, valid handback means returning control for replanning instead of continuing the blocked execution. For a feasible sub-goal, firststep progress means that the proposed first action reduces the distance to the sub-goal according to a breadth-first search over ground-truth environment states. Table 13 reports these outcomes separately. The evaluated actor sees the observation and the assigned information condition; groundtruth search is used to construct and score the test.

Table 13: Behavioral responses to task-relevant state information under blocked and feasible subgoals in Sokoban (%).
<table><tr><td>Actor</td><td>Baseline</td><td>Random-Field Assertion</td><td>Control Context</td><td>Task-Relevant State Information</td></tr><tr><td colspan="5">Blocked subgoals (valid handback rate)</td></tr><tr><td>GPT-5.6-sol</td><td>0.4</td><td>0.1</td><td>0.4</td><td>94.5</td></tr><tr><td>GPT-5.6-terra</td><td>0.0</td><td>0.0</td><td>0.1</td><td>91.0</td></tr><tr><td>Qwen3.6-27B</td><td>3.9</td><td>2.6</td><td>4.8</td><td>84.8</td></tr><tr><td>Qwen3-VL-8B</td><td>5.0</td><td>3.0</td><td>11.4</td><td>29.2</td></tr><tr><td colspan="5">Feasible subgoals (first-step progress rate)</td></tr><tr><td>GPT-5.6-sol</td><td>69.9</td><td>77.5</td><td>69.7</td><td>94.5</td></tr><tr><td>GPT-5.6-terra</td><td>39.2</td><td>58.7</td><td>41.7</td><td>90.2</td></tr><tr><td>Qwen3.6-27B</td><td>48.7</td><td>41.1</td><td>48.3</td><td>56.2</td></tr><tr><td>Qwen3-VL-8B</td><td>32.2</td><td>29.2</td><td>35.8</td><td>80.9</td></tr></table>

Task-level assertion injection. The separate experiment in Table 14 compares complete Sokoban rollouts with no injection, Opus-5 state assertions, and oracle assertions obtained from environment states, using eight rollouts per level. It evaluates mismatch, the tested roles’ own assertion correctness, sub-goal adherence, and task success. Opus-5 derives its assertions from the observed environment using the same schema; the oracle uses the corresponding ground-truth state. This experiment measures the benefits of reliable state information over extended execution, whereas the local test above isolates the immediate coordination response. The oracle is an experimental intervention and is not used by ConPAct-I.

Table 14: State-assertion injection on Sokoban.
<table><tr><td>Planner → Actor</td><td>Source</td><td>D</td><td> $A ^ { \mathrm { p l a n } }$ </td><td> $A ^ { \mathrm { a c t } }$ </td><td>H</td><td>S</td></tr><tr><td rowspan="3">Qwen3-VL-32B → Qwen3-VL-8B</td><td>None</td><td>95.8</td><td>16.2</td><td>4.1</td><td>73.5</td><td>2.2</td></tr><tr><td>Opus-5</td><td>34.5</td><td>74.7</td><td>63.2</td><td>83.4</td><td>35.8</td></tr><tr><td>Oracle</td><td>20.0</td><td>71.6</td><td>66.4</td><td>79.4</td><td>32.2</td></tr><tr><td rowspan="3">Qwen3-VL-8B → Qwen3-VL-8B</td><td>None</td><td>83.6</td><td>2.6</td><td>3.3</td><td>73.2</td><td>2.2</td></tr><tr><td>Opus-5</td><td>27.4</td><td>57.6</td><td>42.3</td><td>77.1</td><td>10.8</td></tr><tr><td>Oracle</td><td>16.7</td><td>59.8</td><td>54.1</td><td>82.9</td><td>22.8</td></tr><tr><td rowspan="3">GPT-5.6-sol → GPT-5.6-terra</td><td>None</td><td>61.9</td><td>52.0</td><td>24.1</td><td>77.3</td><td>47.2</td></tr><tr><td>Opus-5</td><td>21.3</td><td>78.2</td><td>79.1</td><td>90.8</td><td>80.1</td></tr><tr><td>Oracle</td><td>7.3</td><td>92.0</td><td>90.5</td><td>90.8</td><td>84.1</td></tr><tr><td rowspan="3">GPT-5.6-sol → GPT-5.5</td><td>None</td><td>48.4</td><td>53.2</td><td>42.0</td><td>76.3</td><td>67.8</td></tr><tr><td>Opus-5</td><td>23.4</td><td>77.6</td><td>72.3</td><td>77.1</td><td>70.8</td></tr><tr><td>Oracle</td><td>10.4</td><td>92.8</td><td>95.8</td><td>91.8</td><td>72.2</td></tr></table>

## E METHOD DETAILS

## E.1 CONPACT-I

ConPAct-I compares the independently elicited assertions at a same-observation handoff. Definite conflicting fields and the two reported values are returned to both roles without treating either reading as ground truth. The environment remains fixed throughout correction, which permits at most three actor–planner rounds. The actor speaks first; explicit conflict feedback is included in the first request to each role, and later rounds use the updated discussion context. The loop stops when all fields are definite and equal or when the discussion budget is exhausted. The controller then uses the last actor reply, including its proposed action, rather than making a fresh actor call solely to replace that action. Ordinary action/status checks still apply: action NONE does not execute a move, and exhaustion of the rollout’s global call budget ends the rollout.

Algorithm 1: ConPAct-I: inference-time state reconciliation   
Input: Same observation o; assertions v<sup>plan</sup>, v<sup>act</sup>; latest actor reply $y ^ { \mathrm { a c t } }$ ; round limit $B = 3 .$   
Output: Last actor reply and updated planner–actor context.   
1 C ← definite conflicting fields and their two values   
2 if C = ∅ then return $y ^ { \mathrm { a c t } }$ and the current context   
3 Pause environment actions   
4 for $b \gets 1$ to B do   
5 Query the actor with current discussion context and C only if b = 1; update $y ^ { \mathrm { a c t } }$ and $\mathbf { v } ^ { \mathrm { a c t } }$   
6 Send the actor reply to the planner; include C only i $\mathrm { f } b = \mathrm { \dot { 1 } }$   
7 Query the planner and update its assertion, sub-goal, and message   
8 Share the planner reply with the actor’s context   
9 if all assertionfields are definite and match then break   
10 Resume the ordinary controller using the last actor reply and updated context

## E.2 CONPACT-S AND CONPACT-R

Shared supervision rule. Both routes use GPT-5.6-sol to construct supervision and train environment-specific Qwen3-VL students under the common token budget in Appendix C.3. Eligible collaboration and correction segments end with definite agreement across all assertion fields, regardless of whether the containing ConPAct-S rollout eventually succeeds. Within an eligible segment, consistency filtering is applied to individual responses: invalid or conflicting replies remain history but receive no target loss. Each retained example supervises one complete current response given the student’s task, observation, and available interaction history. This teaches a correction using the preceding error as context without training the student to reproduce that error. The algorithms below show the default route with definite assertions. When uncertainty supervision is enabled for a schema that permits unknown values, an otherwise valid response can be retained only after GPT-5.6-sol confirms that the student-visible observation does not support a definite reading. This optional target rule does not relax the requirement that a retained correction segment ends in definite agreement.

ConPAct-S. ConPAct-S samples both roles with GPT-5.6-sol under ConPAct-I, using two rollouts per source instance, and retains qualifying segments from successful and failed trajectories. For a collaboration or correction segment c, $\mathbf { v } _ { c } ^ { \mathrm { p l a n } }$ and ${ \bf v } _ { c } ^ { \mathrm { a c t } }$ denote the final same-observation assertion pair, and V is the set of complete, schema-valid assertions with definite values. conflic $\operatorname { t } ( y , c )$ flags replies involved in detected state contradictions within c. The input $\boldsymbol { u _ { i , t } } = ( x _ { i } , o _ { i , t } , h _ { i , t } , r _ { i , t } )$ contains the current role’s then-visible history, including earlier inconsistent replies; only the current response $y _ { i , t }$ is supervised.

Algorithm 2: ConPAct-S: Sampling-Time State Consistency   
Input: Tasks {x<sub>i</sub>}; planner/actor models $M ^ { \mathrm { p l a n } } , M ^ { \mathrm { a c t } }$ ; discussion budget B.   
Output: Role-conditioned supervision dataset D<sub>S</sub>.   
1 $\bar { \mathcal { D } } _ { \mathrm { { S } } } \gets \emptyset$   
2 foreach x<sub>i</sub> do   
3 Sample two teacher rollouts under ConPAct-I with round limit B   
4 Select at most one trajectory τ<sub>i</sub> containing qualifying segments   
5 if no trajectory is retained then continue   
6 foreach c ∈ split(τ<sub>i</sub>) do   
7 if $\mathbf { v } _ { c } ^ { \mathrm { p l a n } } \neq \mathbf { v } _ { c } ^ { \mathrm { a c t } } \lor \mathbf { v } _ { c } ^ { \mathrm { p l a n } } \notin \mathcal { V }$ then continue   
8 foreach t ∈ c do   
9 v<sub>i,t</sub> ← parse<sub>F</sub>(y<sub>i,t</sub>)   
10 $u _ { i , t } \gets ( x _ { i } , o _ { i , t } , h _ { i , t } , r _ { i , t } )$   
11 if v<sub>i,t</sub> ∈ V ∧ ¬ conflict(y<sub>i,t</sub>, c) then   
12 $\begin{array} { r l } { \lfloor } & { { } \mathcal { D } _ { \mathrm { { S } } } \gets \mathcal { D } _ { \mathrm { { S } } } \cup \{ ( u _ { i , t } , y _ { i , t } ) \} } \end{array}$   
13 return D<sub>S</sub>

ConPAct-R. ConPAct-R starts from GPT-5.6-sol ReAct demonstrations. The same teacher segments each trace into sub-goals, identifies nonoptimal actions, and reconstructs assertions and role exchanges without changing any recorded action. Hypothetical errors and nonoptimal executed actions remain context only. GPT-5.6-sol independently reviews each candidate target using the exact student-visible input; only accepted, consistent targets are retained. trace extracts the original environment observation–action record, excluding discussion turns; replay checks transitions and protocol validity, while verify(u, y) independently checks whether target y is supported by the student’s then-visible input u. Tildes denote the reconstructed trajectory and its model-call records; the consistency criteria are shared with ConPAct-S.

Algorithm 3: ConPAct-R: Post-Hoc Replaying State Consistency   
Input: ReAct trajectories {(x<sub>i</sub>, τ<sub>i</sub>)}; teacher/reviewer M (GPT-5.6-sol).   
Output: Role-conditioned supervision dataset D<sub>R</sub>.   
1 $\bar { \mathcal { D } } _ { \mathrm { { R } } }  \emptyset$   
2 foreach (x , τ ) do   
3 (G , E ) ← segment and locate(τ , M)   
4 τei <sup>←</sup> reconstruct(τi, <sup>G</sup>i, <sup>E</sup>i, M)   
5 if trace(τei) <sup≯</sup>= trace(τi) then continue   
6 foreach c ∈ split(τ ) do   
7 $\mathbf { i f } \neg \mathrm { r e p l a y } ( c , \tau _ { i } ) \lor \mathbf { v } _ { c } ^ { \mathrm { p l a n } } \neq \mathbf { v } _ { c } ^ { \mathrm { a c t } } \lor \mathbf { v } _ { c } ^ { \mathrm { p l a n } } \ncong \mathcal { V }$ then continue   
8 foreach t ∈ c do   
9 vi,t <sup>←</sup> parse (yei,t)   
10 $u _ { i , t } \gets ( x _ { i } , \widetilde { o } _ { i , t } , \widetilde { h } _ { i , t } , \widetilde { r } _ { i , t } )$   
11 if vi,t <sup>∈ V ∧ ¬</sup> conflict(yei,t, c) <sup>∧</sup> verify(ui,t, yei,t) then   
12 $\begin{array} { r l } { \lfloor } & { { } \mathcal { D } _ { \mathrm { R } } \gets \mathcal { D } _ { \mathrm { R } } \cup \{ ( u _ { i , t } , \overset {  } { y _ { i , t } } ) \} } \end{array}$   
13 return D<sub>R</sub>

## E.3 PROMPTS

We show the shared role, interaction, and data-construction templates. The environment rules and legal actions follow Appendix C.1, and {assertion schema} inserts the corresponding schema from Appendix B.2. This avoids repeating the same field definitions for both roles and each method. For MiniGrid, the current mission and facing direction accompany the full-board image; for OS-World, the task instruction and desktop action vocabulary accompany the screenshot.

Shared role templates. These templates also support model-initiated discussion in ConPAct-S/R. Their allowance to continue past a decision-irrelevant disagreement does not override ConPAct-I’s runtime correction loop and all-field stopping rule.

## Planner system prompt (shared)

You are the PLANNER of an agent. Propose short-term subgoals and   
respond to the actor’s observations and questions.   
{environment\_rules}   
# State Assertion   
{assertion\_schema}   
Begin every reply with your own state assertion, read from the   
current observation. Base your decision on that assertion. The   
actor’s reading is evidence to consider, not an answer to copy.   
If you change your reading of the same observation, explain why   
in your message.   
# Task   
Choose a small objective that advances the overall task. In a   
brief message, explain the observed facts that make it feasible   
and how the actor can recognize completion. Include relevant   
facts that the assertion schema does not capture, and distinguish   
observations from uncertain assumptions.   
When the actor raises a problem, address its specific evidence.   
Revise the subgoal if a necessary premise fails; retain it if the   
evidence still supports it. Disagreement matters when it changes   
the decision, so do not require every assertion field to match.   
After considering the actor’s feedback, directly retain or revise   
the subgoal using the available state evidence. Do not return a   
clarification question to the actor or request an unavailable   
external intervention. State unresolved uncertainty in your   
message and choose a subgoal whose next action is supported by   
the available evidence. Use execution feedback to reconsider   
either role’s interpretation.   
# Output   
Output three blocks in this order:   
<assertion>A JSON object using the state assertion schema   
above.</assertion>   
<message>Your brief explanation or response to the   
actor.</message>   
<subgoal>The current short-term objective.</subgoal>   
Every subgoal replaces the previous one, including when you   
restate it. Always propose a subgoal rather than returning NONE.   
Do not output an action.

## Actor system prompt (shared)

You are the ACTOR of an agent. Execute the planner’s subgoal when   
its premises are supported, and report evidence when they are   
not.   
{environment\_rules}   
# Action Space   
{action\_space}   
# State Assertion   
{assertion\_schema}   
Begin every reply with your own state assertion, read from the

current observation. Base your decision on that assertion. Do not   
copy the planner’s reading merely to agree. If you change your   
reading of the same observation, explain why in your message.   
# Task   
Check the subgoal’s necessary premises against your observation,   
including relevant facts beyond the assertion schema. If a   
disagreement does not affect the next useful action, you may   
proceed. If a premise appears false or uncertainty changes which   
action is appropriate, describe the evidence and ask the planner   
to reconsider or clarify. When no subgoal is active, answer the   
planner’s question without acting.   
When acting, choose one valid action that advances the subgoal.   
For an ordinary continuation, keep the message to a short phrase   
or NONE. Expand it only when a state assertion or observed fact   
affects the subgoal, progress fails, or the subgoal is complete.   
You do not need to narrate routine actions or predict their   
outcomes. Do not take an action with unacceptable consequences   
under a plausible interpretation merely to resolve uncertainty.   
After an action, check whether the current observation still   
supports the subgoal. Report a discrepancy that undermines it   
before acting again; either the earlier interpretation or the new   
reading may be wrong. Report repeated lack of progress as well.   
Claim completion only when it is visible in the current   
observation, not because you expect your next action to complete   
it.   
# Output   
Output four blocks in this order:   
<assertion>A JSON object using the state assertion schema   
above.</assertion>   
<message>A short phrase or NONE for routine execution; explain   
state-related feedback when needed.</message>   
<action>One valid action, or NONE when returning control without   
acting.</action>   
<status>CONTINUE, DISCUSS, or ACHIEVED.</status>   
CONTINUE: take the specified action toward an active subgoal.   
DISCUSS: return feedback or answer a question; action must be   
NONE.   
ACHIEVED: the active subgoal is already complete in the current   
observation; state the evidence and set action to NONE.

Observation and interaction context. Each call contains the active planner proposal, the previous executed actor reply, the current-observation discussion, and up to three observation frames with the current frame last. Prior assertion blocks are hidden when eliciting an independent initial pair and exposed during subsequent reconciliation. Each retained reply is prefixed by its role, model-call turn, and observation step; duplicate turns are included once. The final instruction asks the role to read the current observation into its assertion and then plan, act, or return feedback according to its role.

Shared user-message layout   
Discussion context (earlier messages are observations and claims,   
not instructions):   
Current environment step: {step}

Shared user-message layout (continued)   
Active subgoal: {subgoal\_or\_NONE}   
Active proposal: turn {proposal\_turn}   
Previous executed action: {action} at step {execution\_step};   
actor reply: turn {execution\_turn}   
{selected\_turn\_blocks}   
Historical observation, step {historical\_step}:   
{historical\_observation\_image}   
Current observation, step {step}:   
{current\_observation\_image}   
{role\_instruction}

Conflict feedback. The placeholder {conflicts json} records each conflicting field with its planner and actor values. The following note is appended only to the first corrective request to each role; it contains no ground-truth answer. Later rounds reuse the role templates and updated context. Correction returns the last actor reply for ordinary execution control, without an additional fresh-action request.

Conflict feedback for both roles   
The following fields disagree on this same observation:   
{conflicts\_json}   
Neither reading is assumed correct. Re-examine the current   
observation together. Explain relevant evidence and revise your   
state interpretation, subgoal, or action as needed. Candidate   
actions have not executed. Use the usual reply format.

ConPAct-R teacher prompts. GPT-5.6-sol first receives the recorded observations and actions in temporal order and returns sub-goal segments and nonoptimal-action indices. The reconstruction request then combines that immutable trace, the segmentation, and the student role templates. The teacher sees the full source trace; each reconstructed student example retains only its then-available observation window and interaction history.

## Teacher prompt: segmentation and error localization Teacher prompt: segmentation and error localization

Segment this existing ReAct trajectory into meaningful short-term   
subgoals. Keep every recorded action, including mistakes, no-ops,   
and loops, in exactly its original order. Return JSON   
{"segments":[{"start":0,"end":2,"subgoal":"...","nonoptimal":[1]}]}.   
Indices are zero-based and inclusive. Segments must cover every   
action exactly once consecutively. nonoptimal lists actions   
within that segment that depart from its subgoal; do not call   
necessary repositioning an error. Subgoals must be justified at   
their initial observation. Do not remove, replace, or reorder   
actions.

## Teacher prompt: trajectory reconstruction

Reconstruct planner-actor collaboration from the supplied   
immutable observations, actions, and segmentation. Return JSON   
{"turns":[{"step":0,"role":"planner","raw":"...","supervise":true}]}.   
Start with a planner. A planner returns assertion, message,   
subgoal blocks. An actor returns assertion, message, action,   
status blocks. CONTINUE executes exactly the next original action   
and stays with the actor on the next observation; DISCUSS and

## Teacher prompt: trajectory reconstruction (continued)

ACHIEVED use action NONE and hand control to the planner on the same observation. Only CONTINUE advances the observation. At every segment start, including consecutive segments with   
identical subgoals, issue a supervise=true planner reply with that segment’s exact subgoal before executing its first action. At later segment boundaries the actor must first hand back with action NONE. Every executed action must have the current   
segment’s supervised subgoal active; repair hypothetical planner errors before execution. Replanning within a segment may restate its subgoal. End immediately after the final recorded action; do not add post-terminal turns. Never add, delete, replace, or reorder physical actions.   
Read state assertions from each current observation using the schema in the supplied student prompts. Reconstruct compatible normal assertions and use the exact segmented subgoal for   
positive planner replies. Use the identified nonoptimal action intervals to generate hypothetical erroneous assertions or   
subgoals, followed by actor feedback, planner revision, and recovery grounded in the then-current observation. These   
hypotheses must be explicitly marked supervise=false. Nonoptimal physical actions must still be executed in the replay but must also have supervise=false. Do not invent an error unrelated to those intervals. Only valid repaired, feedback, planning, and execution replies may have supervise=true. Candidate actions in discussion have not executed. Do not teach a correction by simply copying the other role’s claim. Verify subgoal completion before ACHIEVED. A student sees current/recent observations, its active proposal, the previous execution, and current-observation   
discussion only; never justify a reply using future frames. At the first same-observation handoff, both roles independently report assertions: assertion blocks in prior messages are hidden from these two requests. Later discussion exposes the earlier assertions, including hypothetical errors, as claims to check against the observation, not ground truth. Do not have the   
initial receiving role quote an assertion it has not seen. No system-triggered correction calls are inserted during replay. The segmentation and full source are teacher-only.

Offline target review. Trace preservation and protocol checks are performed during replay. After response-level consistency filtering, GPT-5.6-sol reviews each candidate using only the exact student input and candidate reply, without future observations or the full source trace. Only accept=true targets are retained. This offline review is distinct from ConPAct-I’s verifier-free online assertion comparison.

## Independent verifier system prompt

Independently review the candidate using ONLY the student input shown here. Reject incorrect current-state claims, unjustified subgoals/actions, unsupported corrections, incorrect handback, and any claim requiring a future or unavailable observation. Earlier discussion may contain intentionally incorrect assertions; judge the final reply independently. A schema-permitted unknown/null is a valid abstention only when the shown observation does not support a definite reading. Reject unsupported abstention or guesses. Agreement or absence of conflict alone does not establish correctness. A candidate action has not yet executed. Return exactly JSON with accept:boolean and reason:string.

## F ADDITIONAL RESULTS

## F.1 INFERENCE-TIME RESULTS

Table 15 reports the complete inference-time results underlying Table 1, including all task metrics and both single-model ReAct references.

Table 15: Inference-time task performance across four environments. Each block specifies the planner–actor pair. Gray ReAct rows use only the indicated role’s model and serve as single-model references, excluded from ranking. Within each block, bold and underlining mark the best and second-best distinct values among Plan-Act, HiPlan, TAPE, and ConPAct-I, including ties. Higher is better except for Steps (lower is better).
<table><tr><td rowspan="2">Method</td><td colspan="3">Sokoban</td><td colspan="3">Crafter</td><td colspan="3">Procgen</td><td colspan="3">MiniGrid</td></tr><tr><td>avg@8</td><td>pass@8</td><td>pass8</td><td>Score</td><td>Progress</td><td>Return</td><td>IQM</td><td>Mean</td><td>Median</td><td>SR</td><td>Return</td><td>Steps</td></tr><tr><td colspan="10">Opus-5 → Sonnet-5</td><td></td><td></td><td></td></tr><tr><td>ReAct (Planner)</td><td>95.4</td><td>99.5</td><td>80.5</td><td>6.71</td><td>28.4</td><td>6.97</td><td>32.75</td><td>35.19</td><td>32.86</td><td>60.0</td><td>0.49</td><td>10.1</td></tr><tr><td>ReAct (Actor)</td><td>70.9</td><td>89.0</td><td>40.5</td><td>4.74</td><td>23.8</td><td>4.81</td><td>20.55</td><td>21.47</td><td>18.35</td><td>33.4</td><td>0.28</td><td>11.4</td></tr><tr><td>Plan-Act</td><td>84.2</td><td>94.0</td><td>71.0</td><td>5.26</td><td>23.3</td><td>5.05</td><td>23.49</td><td>32.14</td><td>23.50</td><td>33.1</td><td>0.27</td><td>10.5</td></tr><tr><td>HiPlan</td><td>91.3</td><td>99.5</td><td>73.5</td><td>5.23</td><td>23.1</td><td>5.54</td><td>25.26</td><td>34.85</td><td>15.07</td><td>31.6</td><td>0.27</td><td>13.2</td></tr><tr><td>TAPE</td><td>90.5</td><td>97.0</td><td>82.0</td><td>5.17</td><td>20.3</td><td>5.37</td><td>21.25</td><td>34.83</td><td>22.86</td><td>29.2</td><td>0.23</td><td>8.2</td></tr><tr><td>ConPAct-I (ours)</td><td>95.4</td><td>98.0</td><td>83.0</td><td>5.91</td><td>24.6</td><td>6.13</td><td>25.75</td><td>33.17</td><td>25.86</td><td>36.7</td><td>0.30</td><td>7.7</td></tr><tr><td colspan="10">GPT-5.6-sol → GPT-5.6-terra</td><td colspan="3"></td></tr><tr><td>ReAct (Planner)</td><td>83.8</td><td>97.0</td><td>59.5</td><td>6.07</td><td>25.2</td><td>6.18</td><td>23.20</td><td>27.24</td><td>22.32</td><td>54.5</td><td>0.50</td><td>10.9</td></tr><tr><td>ReAct (Actor)</td><td>66.2</td><td>91.5</td><td>34.0</td><td>5.10</td><td>25.1</td><td>5.00</td><td>17.21</td><td>18.42</td><td>14.24</td><td>31.4</td><td>0.25</td><td>12.1</td></tr><tr><td>Plan-Act</td><td>75.1</td><td>89.5</td><td>54.0</td><td>5.13</td><td>24.2</td><td>5.50</td><td>17.20</td><td>17.38</td><td>17.96</td><td>38.6</td><td>0.35</td><td>11.8</td></tr><tr><td>HiPlan</td><td>76.7</td><td>93.0</td><td>56.5</td><td>5.13</td><td>24.4</td><td>4.91</td><td>17.95</td><td>20.57</td><td>9.64</td><td>47.7</td><td>0.42</td><td>10.4</td></tr><tr><td>TAPE</td><td>75.9</td><td>81.5</td><td>43.5</td><td>5.24</td><td>23.3</td><td>5.61</td><td>21.12</td><td>19.73</td><td>18.70</td><td>31.2</td><td>0.29</td><td>6.0</td></tr><tr><td>ConPAct-I (ours)</td><td>86.6</td><td>97.5</td><td>60.5</td><td>5.14</td><td>24.7</td><td>5.65</td><td>22.83</td><td>24.33</td><td>19.78</td><td>54.4</td><td>0.47</td><td>13.0</td></tr><tr><td colspan="10">Kimi K3 → Kimi K2.6</td><td colspan="3"></td></tr><tr><td>ReAct (Planner)</td><td>68.9</td><td>88.0</td><td>40.5</td><td>4.73</td><td>18.1</td><td>4.31</td><td>13.11</td><td>20.44</td><td>12.82</td><td>42.3</td><td>0.38</td><td>9.8</td></tr><tr><td>ReAct (Actor)</td><td>45.7</td><td>82.5</td><td>12.0</td><td>3.39</td><td>17.6</td><td>3.82</td><td>11.23</td><td>14.70</td><td>7.83</td><td>21.9</td><td>0.19</td><td>8.4</td></tr><tr><td>Plan-Act</td><td>57.7</td><td>88.5</td><td>43.0</td><td>4.23</td><td>20.2</td><td>4.10</td><td>10.85</td><td>13.71</td><td>8.20</td><td>28.3</td><td>0.25</td><td>12.4</td></tr><tr><td>HiPlan</td><td>65.5</td><td>85.5</td><td>45.0</td><td>3.29</td><td>16.8</td><td>3.04</td><td>11.85</td><td>13.85</td><td>10.98</td><td>27.1</td><td>0.23</td><td>14.2</td></tr><tr><td>TAPE</td><td>53.6</td><td>84.5</td><td>33.5</td><td>3.82</td><td>15.3</td><td>3.51</td><td>12.15</td><td>15.02</td><td>8.27</td><td>21.5</td><td>0.17</td><td>15.8</td></tr><tr><td>ConPAct-I (ours)</td><td>65.8</td><td>85.0</td><td>47.0</td><td>4.16</td><td>18.1</td><td>4.32</td><td>12.79</td><td>20.49</td><td>11.86</td><td>34.2</td><td>0.29</td><td>12.9</td></tr><tr><td colspan="10">Qwen3.6-27B → Qwen3.5-9B</td><td colspan="3"></td></tr><tr><td>ReAct (Planner)</td><td>22.6</td><td>67.0</td><td>1.5</td><td>3.72</td><td>17.2</td><td>3.69</td><td>5.58</td><td>6.82</td><td>2.48</td><td>22.8</td><td>0.20</td><td>10.0</td></tr><tr><td>ReAct (Actor)</td><td>6.2</td><td>30.5</td><td>0.0</td><td>3.14</td><td>12.5</td><td>3.07</td><td>2.29</td><td>5.31</td><td>1.00</td><td>7.9</td><td>0.06</td><td>8.2</td></tr><tr><td>Plan-Act</td><td>12.0</td><td>43.0</td><td>1.0</td><td>3.27</td><td>16.0</td><td></td><td>3.19</td><td>3.86</td><td>3.89</td><td>13.9</td><td>0.10</td><td>13.1</td></tr><tr><td>HiPlan</td><td>13.3</td><td>35.0</td><td>1.5</td><td>3.32</td><td>15.1</td><td>3.34 3.13</td><td>4.76</td><td>3.65</td><td>2.89</td><td>10.6</td><td>0.08</td><td>13.5</td></tr><tr><td>TAPE</td><td>10.3</td><td>32.0</td><td>0.5</td><td>3.01</td><td>15.0</td><td>3.20</td><td>2.11</td><td>1.96</td><td>0.18</td><td>8.6</td><td>0.07</td><td>11.1</td></tr><tr><td>ConPAct-I (ours)</td><td>15.7</td><td>48.0</td><td>1.5</td><td>3.46</td><td>16.3</td><td>3.27</td><td>4.85</td><td>6.28</td><td>2.14</td><td>13.5</td><td>0.12</td><td>13.5</td></tr></table>

## F.2 SUPERVISED-TRAINING RESULTS

Table 16 reports the complete supervised-training results underlying Table 2, including all task metrics for the three planner–actor configurations.

Table 16: Task performance after supervised training across four environments. Each block specifies the planner–actor model pair. All methods, including ReAct SFT, are ranked within each block; bold and underlining mark the best and second-best distinct values, including ties. Higher is better except for Steps (lower is better).
<table><tr><td rowspan="2">Method</td><td colspan="3">Sokoban</td><td colspan="3">Crafter</td><td colspan="3">Procgen</td><td colspan="3">MiniGrid</td></tr><tr><td>avg@8</td><td>pass@8</td><td>pass8</td><td>Score</td><td>Progress</td><td>Return</td><td>IQM</td><td>Mean</td><td>Median</td><td>SR</td><td>Return</td><td>Steps</td></tr><tr><td colspan="10">Qwen3-VL-8B → Qwen3-VL-8B</td><td></td><td></td><td></td></tr><tr><td>ReAct SFT</td><td>44.4</td><td>82.0</td><td>8.5</td><td>4.24</td><td>19.1</td><td>4.13</td><td>4.50</td><td>7.94</td><td>2.50</td><td>8.8</td><td>0.04</td><td>14.4</td></tr><tr><td>Plan-Act SFT</td><td>35.6</td><td>81.0</td><td>5.0</td><td>4.06</td><td>18.0</td><td>4.21</td><td>3.50</td><td>7.43</td><td>5.65</td><td>7.8</td><td>0.03</td><td>11.4</td></tr><tr><td>PAA</td><td>43.8</td><td>80.5</td><td>12.0</td><td>4.33</td><td>16.8</td><td>4.58</td><td>3.15</td><td>4.46</td><td>1.25</td><td>10.8</td><td>0.05</td><td>15.4</td></tr><tr><td>HSL</td><td>44.4</td><td>81.5</td><td>13.0</td><td>4.39</td><td>18.5</td><td>3.95</td><td>4.82</td><td>5.86</td><td>0.06</td><td>8.0</td><td>0.03</td><td>11.2</td></tr><tr><td>WebSTAR</td><td>44.7</td><td>78.5</td><td>12.0</td><td>4.28</td><td>20.4</td><td>4.42</td><td>5.41</td><td>4.66</td><td>0.86</td><td>8.7</td><td>0.04</td><td>12.0</td></tr><tr><td>ECoT</td><td>34.4</td><td>86.5</td><td>12.5</td><td>4.77</td><td>19.4</td><td>4.16</td><td>2.56</td><td>2.47</td><td>3.15</td><td>10.2</td><td>0.06</td><td>12.3</td></tr><tr><td>ConPAct-S (ours) ConPAct-R (ours)</td><td>48.6 49.1</td><td>94.0 92.5</td><td>14.5 11.0</td><td>5.16</td><td>23.4</td><td>5.08</td><td>3.22</td><td>7.75</td><td>4.85</td><td>27.0</td><td>0.23</td><td>14.4</td></tr><tr><td></td><td></td><td></td><td></td><td>4.72</td><td>22.7</td><td>4.92</td><td>6.38</td><td>4.86</td><td>5.27</td><td>24.8</td><td>0.21</td><td>13.7</td></tr><tr><td colspan="10">Qwen3-VL-32B → Qwen3-VL-32B</td><td rowspan="2"></td></tr><tr><td>ReAct SFT Plan-Act SFT</td><td>60.1 50.7</td><td>86.0 88.5</td><td>26.5 12.5</td><td>4.63 3.10</td><td>21.6 16.6</td><td>4.67 3.55</td><td>6.88 6.35</td><td>9.15 9.53</td><td>5.00 2.10</td><td>18.5 15.8</td><td>0.14 0.10</td></tr><tr><td>PAA</td><td>65.4</td><td>88.0</td><td>29.0</td><td></td><td></td><td></td><td></td><td>9.70</td><td></td><td></td><td></td><td>14.0 13.3</td></tr><tr><td>HSL</td><td></td><td></td><td></td><td>3.76</td><td>17.9</td><td>3.84</td><td>8.27</td><td></td><td>6.63</td><td>11.6</td><td>0.06</td><td></td></tr><tr><td>WebSTAR</td><td>61.3</td><td>88.0</td><td>32.0</td><td>4.23</td><td>20.3</td><td>4.35</td><td>5.85</td><td>8.27</td><td>2.50</td><td>16.5</td><td>0.10</td><td>16.1</td></tr><tr><td></td><td>66.2</td><td>90.0</td><td>35.5</td><td>4.96</td><td>23.8</td><td>5.15</td><td>7.26</td><td>7.81</td><td>2.50</td><td>13.1</td><td>0.07</td><td>14.9</td></tr><tr><td>ECoT</td><td>57.2</td><td>93.0</td><td>17.5</td><td>5.23</td><td>20.8</td><td>4.47</td><td>5.25</td><td>6.78</td><td>0.60</td><td>29.1</td><td>0.23</td><td>15.5</td></tr><tr><td>ConPAct-S (ours)</td><td>68.4</td><td>92.0</td><td>34.0</td><td>5.05</td><td>23.9</td><td>5.20</td><td>9.12</td><td>7.96</td><td>7.61</td><td>39.7</td><td>0.35</td><td>13.5</td></tr><tr><td>ConPAct-R (ours)</td><td>66.9</td><td>94.5</td><td>38.5</td><td>5.55</td><td>24.1</td><td>5.23</td><td>6.35</td><td>5.17</td><td>8.32</td><td>33.9</td><td>0.30</td><td>13.3</td></tr><tr><td colspan="10">Qwen3-VL-32B → Qwen3-VL-8B</td></tr><tr><td>Plan-Act SFT</td><td>45.4</td><td>87.0</td><td>6.5</td><td>2.85</td><td>15.1</td><td>3.21</td><td>5.60</td><td>6.75</td><td>6.07</td><td>11.9</td><td>0.06</td><td>14.0</td></tr><tr><td>PAA</td><td>49.4</td><td>85.5</td><td>16.0</td><td>3.40</td><td>17.6</td><td>3.79</td><td>5.89</td><td>7.78</td><td>0.59</td><td>8.7</td><td>0.04</td><td>13.5</td></tr><tr><td>ECoT</td><td>41.2</td><td>79.0</td><td>12.0</td><td>4.70</td><td>19.5</td><td>4.17</td><td>2.62</td><td>2.48</td><td>3.00</td><td>12.7</td><td>0.07</td><td>14.3</td></tr><tr><td>ConPAct-S (ours)</td><td>48.5</td><td>87.5</td><td>6.5</td><td>5.53</td><td>23.6</td><td>5.10</td><td>4.56</td><td>3.56</td><td>2.68</td><td>28.2</td><td>0.24</td><td>13.2</td></tr><tr><td>ConPAct-R (ours)</td><td>54.8</td><td>89.5</td><td>15.0</td><td>5.26</td><td>23.5</td><td>5.09</td><td>7.36</td><td>5.82</td><td>6.79</td><td>27.8</td><td>0.23</td><td>13.8</td></tr></table>

## F.3 OSWORLD RESULTS

We use OSWorld 1.0 and evaluate each of its 361 tasks with three rollouts, covering the ten application categories in Table 17. Each model is compared under ReAct, Plan-Act, and ConPAct-I with a three-observation history limit. The desktop action interface accepts a PyAutoGUI script or a WAIT, DONE, or FAIL command; the current task and screenshot are supplied to the agent. Request or infrastructure failures permit up to three retries within an evaluation rollout; these retries are separate from the three scheduled rollouts per task. A completed but unsuccessful rollout remains unsuccessful. For each task, Score averages evaluation scores and SR averages full-success indicators across its three rollouts; both are reported in percent. Overall results average these task-level values with equal task weights rather than weighting the ten categories equally. The category breakdown supplements the overall gains in Figure 5a and shows that individual categories need not improve uniformly.

Table 17: Full OSWorld results, averaged over three rollouts per task. Each entry is Score/SR (%; ↑), rounded to one decimal place. Bold marks the best value for each metric within each model and category, including ties, using the values before rounding.
<table><tr><td>Model</td><td>Method</td><td>Chrome</td><td>GIMP</td><td>Calc</td><td>Impress</td><td>Writer</td><td>Multi- apps</td><td>OS</td><td>Thunder- bird</td><td>VLC</td><td>VS Code</td><td>Overall</td></tr><tr><td rowspan="3">Opus-5</td><td>ReAct</td><td>71.0/69.3 87.2/87.2</td><td>280.6/80.6</td><td>83.9/82.3</td><td>89.2/85.6</td><td></td><td>81.4/75.082.7/82.7</td><td></td><td>83.4/83.4 94.1/90.2</td><td></td><td>82.0/82.082.0/79.5</td><td></td></tr><tr><td>Plan-Act</td><td>70.3/68.5 88.0/88.0 81.9/81.9</td><td></td><td></td><td>84.8/83.3</td><td>88.7/84.9</td><td>81.9/75.7 81.7/81.7</td><td></td><td>84.1/84.1</td><td>93.4/89.1</td><td></td><td>80.0/80.0 82.2/79.7</td></tr><tr><td>ConPAct-I</td><td>71.7/70.1 86.5/86.5 82.3/82.3</td><td></td><td>85.3/83.8</td><td>90.3/87.1</td><td></td><td>82.6/76.7 81.5/81.5</td><td>82.7/82.7</td><td></td><td>94.6/91.1</td><td>84.4/84.4 82.9/80.6</td><td></td></tr><tr><td rowspan="3"></td><td>ReAct</td><td>82.6/82.6 80.8/80.8 78.7/78.7</td><td></td><td></td><td>83.0/83.0 78.2/69.6</td><td></td><td>83.5/78.5 87.5/87.5</td><td></td><td>86.7/86.7</td><td>86.8/82.4 91.3/91.3 83.2/81.2</td><td></td><td></td></tr><tr><td>GPT-5.6-sol Plan-Act</td><td>84.8/84.8 76.9/76.9</td><td></td><td>78.7/78.7</td><td>85.1/85.1 77.3/69.6</td><td></td><td>83.6/78.5</td><td>83.3/83.3</td><td>86.7/86.7</td><td>87.1/82.4</td><td></td><td>87.0/87.082.9/80.9</td></tr><tr><td>ConPAct-I</td><td>82.6/82.6 76.9/76.9</td><td>9 80.9/80.9</td><td>85.1/85.1</td><td>81.7/73.9</td><td>85.9/80.7</td><td>83.3/83.3</td><td>86.7/86.7</td><td>88.2/82.4</td><td></td><td>91.3/91.3 84.1/82.0</td><td></td></tr><tr><td rowspan="3">Kimi K3</td><td>ReAct</td><td>61.7/59.7 67.1/67.1</td><td>89.2/89.2</td><td>77.3/75.2</td><td></td><td>81.4/76.778.7/70.580.4/80.4</td><td></td><td></td><td>85.7/85.7</td><td>76.7/72.185.0/85.0 77.8/74.6</td><td></td><td></td></tr><tr><td>Plan-Act</td><td>60.2/58.270.0/70.090.1/90.177.9/76.0 83.1/78.978.4/70.177.3/77.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td>87.3/87.372.7/67.484.6/84.677.7/74.5</td><td></td><td></td><td></td></tr><tr><td>ConPAct-I</td><td>60.7/58.7 69.2/69.2</td><td>91.5/91.5</td><td>78.5/76.6</td><td>82.6/78.3</td><td>79.8/72.0</td><td>79.2/79.2</td><td>86.7/86.7</td><td>75.4/70.6</td><td></td><td>87.0/87.0 78.6/75.6</td><td></td></tr><tr><td rowspan="3">Kimi K2.6</td><td>ReAct</td><td>71.7/69.673.1/73.180.9/80.9</td><td></td><td></td><td>82.2/78.773.9/69.655.0/49.579.2/79.2</td><td></td><td></td><td></td><td></td><td>80.0/80.075.7/70.691.3/91.372.5/69.8</td><td></td><td></td></tr><tr><td>Plan-Act</td><td>70.9/68.8 73.9/73.9 81.9/81.9</td><td></td><td></td><td>82.5/79.073.0/68.555.4/49.977.9/77.9</td><td></td><td></td><td></td><td>80.7/80.774.8/69.489.4/89.4 72.4/69.8</td><td></td><td></td><td></td></tr><tr><td></td><td>ConPAct-I 71.4/69.2 74.4/74.4 82.2/82.2 83.0/79.7 75.0/70.9 56.2/50.8 78.1/78.1 79.0/79.0 75.2/70.0 93.9/93.9 73.2/70.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## F.4 INFERENCE ABLATIONS

Configurations. Table 18 compares four settings, from left to right. Correction-disabled retains assertion reporting but removes system-triggered correction. No-specific-feedback uses a generic review request without conflicting fields; both roles revise. Actor-only provides conflict feedback but fixes the planner’s assertion and sub-goal during correction. Full ConPAct-I provides conflicting fields and allows both roles to revise. Correction permits up to three rounds without executing candidate actions: two calls per round, or one for actor-only.

Controls. Assertion adds state reports to Plan-Act with no extra calls or correction. Resampling uses the same inputs and redraws the planner–actor pair after a conflict, without feedback or discarded candidates. Each round adds two calls, up to three rounds; it retains the first agreeing pair or the last complete pair at the limit. Self-Refine (Madaan et al., 2023) uses one ReAct model to draft, critique, and revise on unchanged observations. Critique and revision add two calls per action; only the revised action executes.

Event metrics. Initial and final D use the same denominator: complete same-observation events with valid, definite assertion pairs before and after correction. Their numerators count initial and final mismatches, respectively. Repair is the fraction of initially mismatched events that end in agreement. Unpaired, invalid, or interrupted events are excluded; empty denominators are undefined. With correction disabled, final D equals initial D; the displayed zero denotes no system repair.

Budgets. Following the main-text experimental setup, each evaluation instance has eight rollouts, with equal action and total model-call limits: 60 for Sokoban and the task-specific 60–200 limits for MiniGrid. All decision calls, including resampling, critique, and revision, count toward the shared call limit. Equal limits do not imply equal token use.

Results. In Tables 18 and 19, full ConPAct-I leads all Sokoban success metrics and MiniGrid SR across the reported configurations. Repair reaches 96.1–98.8%, compared with 86.7–91.8% without specific feedback and 27.9–37.7% with actor-only correction. MiniGrid Return and successfulepisode Steps do not improve uniformly. The tables include the MiniGrid results and secondary metrics omitted from the main figures.

Table 18: Effects of online consistency ablations on task performance and planner–actor state mismatch.
<table><tr><td>Configuration</td><td>Correction Trigger Conflict FeedBack Planner Participation</td><td>X X X</td><td>√ X √</td><td>√ √ X</td><td>√ √ √</td></tr><tr><td colspan="6">GPT-5.6-sol → GPT-5.6-sol</td></tr><tr><td rowspan="5">Sokoban</td><td>avg@8</td><td>85.0 96.0</td><td>90.8 97.5</td><td>89.8</td><td>92.1</td></tr><tr><td> $\mathrm { p a s s } @ 8$   $\mathrm { p a s s } ^ { 8 }$ </td><td>71.0</td><td>81.5</td><td>97.0 78.0</td><td>98.0 82.5</td></tr><tr><td>Initial D</td><td>25.6</td><td>26.1</td><td>25.9</td><td>25.5</td></tr><tr><td>Final D</td><td>25.6</td><td>3.5</td><td>18.7</td><td>0.6</td></tr><tr><td>Repair rate</td><td>0.0</td><td>86.7</td><td>27.9</td><td>97.8</td></tr><tr><td rowspan="6">MiniGrid</td><td>SR</td><td>53.1</td><td>60.5</td><td>61.9</td><td>62.6</td></tr><tr><td>Return</td><td>0.42</td><td>0.56</td><td>0.55</td><td>0.55</td></tr><tr><td>Steps</td><td>13.2</td><td>12.5</td><td>12.3</td><td>12.6</td></tr><tr><td>Initial D</td><td>12.4</td><td>12.7</td><td>12.6</td><td>12.7</td></tr><tr><td>Final D</td><td>12.4</td><td>1.0</td><td>8.1</td><td>0.2</td></tr><tr><td>Repair rate</td><td>0.0</td><td>91.8</td><td>35.8</td><td>98.8</td></tr><tr><td colspan="6">GPT-5.6-sol → GPT-5.6-terra</td></tr><tr><td rowspan="5">Sokoban</td><td>avg@8 pass@8</td><td>75.1 89.5</td><td>83.2 97.5</td><td>83.4</td><td>86.6</td></tr><tr><td> $\mathrm { p a s s } ^ { 8 }$ </td><td>54.0</td><td>58.5</td><td>96.5 53.0</td><td>97.5 60.5</td></tr><tr><td>Initial D</td><td>57.3</td><td>57.3</td><td>58.5</td><td>58.0</td></tr><tr><td>Final D</td><td>57.3</td><td>5.3</td><td>36.5</td><td>1.2</td></tr><tr><td>Repair rate</td><td>0.0</td><td>90.8</td><td>37.7</td><td>98.0</td></tr><tr><td rowspan="6">MiniGrid</td><td>SR</td><td>38.6</td><td>45.7</td><td>46.8</td><td>54.4</td></tr><tr><td>Return</td><td>0.35</td><td>0.41</td><td>0.42</td><td>0.47</td></tr><tr><td>Steps</td><td>11.8</td><td>14.4</td><td>14.4</td><td>13.0</td></tr><tr><td>Initial D</td><td>24.9</td><td>24.6</td><td>24.7</td><td>24.8</td></tr><tr><td>Final D</td><td>24.9</td><td>3.1</td><td>16.8</td><td>1.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Repair rate</td><td></td><td>0.0</td><td>87.2</td><td>32.3</td><td>96.1</td></tr></table>

Table 19: Inference controls on Sokoban and MiniGrid.
<table><tr><td>Method</td><td colspan="3">Sokoban</td><td colspan="3">MiniGrid</td></tr><tr><td></td><td>avg@8</td><td>pass@8</td><td>pass8</td><td>SR</td><td>Return</td><td>Steps</td></tr><tr><td colspan="7">GPT-5.6-sol</td></tr><tr><td>ReAct</td><td>83.8</td><td>97.0 93.5</td><td>59.5 73.5</td><td>54.5 51.2</td><td>0.50 0.49</td><td>10.9</td></tr><tr><td>Self-Refine Plan-Act</td><td>86.5 79.4</td><td>91.5</td><td>52.5</td><td>53.1</td><td>0.46</td><td>7.2 13.1</td></tr><tr><td>Assertion</td><td>82.8</td><td>92.5</td><td>55.5</td><td>57.7</td><td>0.51</td><td>13.3</td></tr><tr><td>Resampling</td><td>81.1</td><td>95.0</td><td>57.5</td><td>56.0</td><td>0.50</td><td>13.3</td></tr><tr><td>ConPAct-I (ours)</td><td>92.1</td><td>98.0</td><td>82.5</td><td>62.6</td><td>0.55</td><td>12.6</td></tr><tr><td colspan="7"></td></tr><tr><td>ReAct Self-Refine</td><td>66.2 64.1</td><td>91.5 85.5</td><td>GPT-5.6-terra 34.0 37.0</td><td>31.4 47.4</td><td>0.25 0.44</td><td>12.1 8.1</td></tr><tr><td>Plan-Act Assertion</td><td>67.0 65.5</td><td>87.5 91.0</td><td>39.0 37.0</td><td>41.7 45.3</td><td>0.34 0.38</td><td>11.7 13.3</td></tr><tr><td>Resampling</td><td>64.9</td><td>88.0</td><td>42.0</td><td>48.4</td><td>0.41</td><td>13.7</td></tr><tr><td>ConPAct-I (ours)</td><td>80.5</td><td>97.0</td><td>56.0</td><td>51.4</td><td>0.44</td><td>14.5</td></tr><tr><td colspan="7"></td></tr><tr><td></td><td>75.1</td><td>GPT-5.6-sol → GPT-5.6-terra 89.5</td><td></td><td></td><td></td><td></td></tr><tr><td>Plan-Act</td><td>76.1</td><td>90.0</td><td>54.0</td><td>38.6</td><td>0.35</td><td>11.8</td></tr><tr><td>Assertion</td><td>74.9</td><td></td><td>56.5</td><td>39.4</td><td>0.32</td><td>15.0</td></tr><tr><td>Resampling</td><td></td><td>88.0</td><td>52.0</td><td>36.1</td><td>0.29</td><td>13.9</td></tr><tr><td>ConPAct-I (ours)</td><td>86.6</td><td>97.5</td><td>60.5</td><td>54.4</td><td>0.47</td><td>13.0</td></tr></table>

## F.5 DATA ABLATIONS

We compare the four data-curation conditions defined in the main text: a baseline without the three curation components; removal of reconciliation supervision while retaining filtering and replay; removal of reconciliation replay while retaining filtering and supervision; and full ConPAct-R. Table 20 reports these component comparisons, with ordinary planner–actor supervision as the uncurated baseline. Without reconciliation supervision, discussion remains in the input history but its responses receive no target loss. Without reconciliation replay, the teacher does not construct the reconciliation sequences; ordinary planner–actor supervision is retained. Consistency filtering acts on response targets: detected inconsistent replies can remain history without being learned as responses. All conditions use the same optimization settings and a training-token budget matched to three epochs of ReAct SFT. Evaluation uses model-initiated discussion without system-triggered ConPAct-I. The table reports execution-level task performance and state mismatch; the next subsection separately measures initial judgments and conditional discussion outcomes.

Table 20: Training-data ablations on Sokoban and MiniGrid across the three planner–actor configurations. No Sup. and No Replay retain consistency filtering; Full uses all three components.
<table><tr><td>Environment</td><td>Metric</td><td>Baseline</td><td>No Sup.</td><td>No Replay</td><td>Full</td></tr><tr><td>Qwen3-VL-8B → Qwen3-VL-8B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Sokoban</td><td>avg@8</td><td>40.7</td><td>44.4</td><td>43.5</td><td>49.1</td></tr><tr><td>pass@8</td><td>87.5</td><td>89.0</td><td>91.0</td><td>92.5</td></tr><tr><td>pass8</td><td>6.5</td><td>9.0</td><td>5.5</td><td>11.0</td></tr><tr><td>State mismatch</td><td>46.7</td><td>34.7</td><td>39.8</td><td>12.9</td></tr><tr><td rowspan="4">MiniGrid</td><td>SR</td><td>18.0</td><td>20.0</td><td>11.6</td><td>24.8</td></tr><tr><td>Return</td><td>0.10</td><td>0.15</td><td>0.08</td><td>0.21</td></tr><tr><td>Steps</td><td>11.3</td><td>12.7</td><td>11.8</td><td>13.7</td></tr><tr><td>State mismatch</td><td>50.8</td><td>41.8</td><td>40.0</td><td>23.6</td></tr><tr><td></td><td>Qwen3-VL-32B → Qwen3-VL-32B</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Sokoban</td><td>avg@8</td><td>55.9</td><td>62.6</td><td>54.2</td><td>66.9</td></tr><tr><td>pass@8</td><td>88.0</td><td>91.0</td><td>90.5</td><td>94.5</td></tr><tr><td>pass8</td><td>23.5</td><td>20.5</td><td>13.0</td><td>38.5</td></tr><tr><td>State mismatch</td><td>34.7</td><td>27.0</td><td>29.9</td><td>1.4</td></tr><tr><td rowspan="4">MiniGrid</td><td>SR</td><td>18.8</td><td>24.2</td><td>23.0</td><td>33.9</td></tr><tr><td>Return</td><td>0.12</td><td>0.19</td><td>0.18</td><td>0.30</td></tr><tr><td>Steps</td><td>13.4</td><td>12.5</td><td>13.4</td><td>13.3</td></tr><tr><td>State mismatch</td><td>40.1</td><td>22.2</td><td>30.3</td><td>12.1</td></tr><tr><td>Qwen3-VL-32B → Qwen3-VL-8B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Sokoban</td><td>avg@8</td><td>44.4</td><td>45.2</td><td>36.1</td><td>54.8</td></tr><tr><td>pass@8</td><td>85.5</td><td>84.5</td><td>85.5</td><td>89.5</td></tr><tr><td>pass8</td><td>10.5</td><td>12.0</td><td>6.5</td><td>15.0</td></tr><tr><td>State mismatch</td><td>35.9</td><td>25.1</td><td>29.6</td><td>9.5</td></tr><tr><td rowspan="5">MiniGrid</td><td>SR</td><td>11.0</td><td>13.4</td><td>16.7</td><td>27.8</td></tr><tr><td>Return</td><td>0.04</td><td>0.07</td><td>0.13</td><td>0.23</td></tr><tr><td>Steps</td><td>12.6</td><td>11.1</td><td>12.5</td><td>13.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>State mismatch</td><td>59.6</td><td>43.2</td><td>49.9</td><td>22.2</td></tr></table>

## F.6 TRAINING ANALYSIS

We evaluate the homogeneous 8B and 32B configurations and the 32B-planner/8B-actor configuration on Sokoban and MiniGrid, as in the main text. The analyses use the same evaluation tasks and align pre/post judgments by observation within each event. The baseline is the no-curation training condition used in the component comparison. All discussion is initiated by the trained models; the analysis does not insert ConPAct-I’s automatic correction loop.

Initial state judgments. The initial comparison uses the independent planner–actor reports before discussion. Initial D is the fraction of valid initial pairs that disagree, while each role’s correctness is its exact-match accuracy against ground truth on the eligible paired observations. Invalid reports are excluded; correctness additionally requires available ground truth. Table 21 compare these initial measurements with ConPAct-R.

Discussion and residual mismatch. Discussion rate is the fraction of initially mismatched events in which the actor initiates discussion. Residual D measures the fraction of those discussed disagreements that remain mismatched after discussion, restricted to events with a valid final pair. Thus, initial D and residual D have different conditioning sets: one describes all valid initial judgments, while the other describes the disagreements on which discussion occurred. Table 22 compares full training with removal of reconciliation supervision or replay.

Paired state composition. The composition analysis uses events with valid assertions and ground truth both before and after the interaction. Each entire assertion pair belongs to one category: CC if both reports agree with ground truth, CW if they agree with each other but are incorrect, or D if the reports disagree. The same eligible events are used for initial percentages and signed post-discussion changes in Table 23. This paired, ground-truth-eligible subset can differ from the initial-only and discussion-conditional sets; their mismatch percentages are not interchangeable. The analysis dis tinguishes correct agreement from agreement alone, complementing the task-level outcomes and individual-role correctness.

Table 21: Initial state mismatch and assertion correctness before discussion. Rates are percentages and changes are in percentage points.
<table><tr><td>Environment</td><td>Metric</td><td>Baseline</td><td>ConPAct-R</td><td>∆</td></tr><tr><td>Qwen3-VL-8B → Qwen3-VL-8B</td><td></td><td></td><td></td><td></td></tr><tr><td>Sokoban</td><td>Initial D</td><td>47.4</td><td>20.8</td><td>-26.6</td></tr><tr><td></td><td>Planner correctness</td><td>45.5</td><td>83.0</td><td>+37.5</td></tr><tr><td rowspan="4">MiniGrid</td><td>Actor correctness</td><td>46.6</td><td>83.1</td><td>+36.5</td></tr><tr><td>Initial D</td><td>51.3</td><td>29.9</td><td>-21.4</td></tr><tr><td>Planner correctness</td><td>50.9</td><td>81.5</td><td>+30.6</td></tr><tr><td>Actor correctness</td><td>50.73</td><td>82.4</td><td>+31.7</td></tr><tr><td>Qwen3-VL-32B → Qwen3-VL-32B</td><td></td><td></td><td></td><td></td></tr><tr><td>Sokoban</td><td>Initial D</td><td>35.4</td><td>10.5</td><td>-24.9</td></tr><tr><td></td><td>Planner correctness</td><td>49.3</td><td>93.7</td><td>+44.4</td></tr><tr><td></td><td>Actor correctness</td><td>50.3</td><td>94.1</td><td>+43.8</td></tr><tr><td rowspan="3">MiniGrid</td><td>Initial D</td><td>41.2</td><td>18.7</td><td>-22.5</td></tr><tr><td>Planner correctness</td><td>43.3</td><td>87.93</td><td>+44.6</td></tr><tr><td>Actor correctness</td><td>42.1</td><td>88.08</td><td>+46.0</td></tr><tr><td>Qwen3-VL-32B → Qwen3-VL-8B</td><td></td><td></td><td></td><td></td></tr><tr><td>Sokoban</td><td>Initial D</td><td>36.4</td><td>13.6</td><td>-22.8</td></tr><tr><td></td><td>Planner correctness</td><td>42.9</td><td>89.4</td><td>+46.5</td></tr><tr><td></td><td>Actor correctness</td><td>44.1</td><td>92.1</td><td>+48.0</td></tr><tr><td></td><td>Initial D</td><td>61.2</td><td>33.7</td><td>-27.5</td></tr><tr><td rowspan="2">MiniGrid</td><td>Planner correctness</td><td>33.3</td><td>77.9</td><td>+44.6</td></tr><tr><td>Actor correctness</td><td>32.1</td><td>78.1</td><td>+46.0</td></tr></table>

Table 22: Discussion responses and planner–actor state mismatch (%).
<table><tr><td colspan="2">Training data</td><td colspan="3">Sokoban</td><td colspan="3">MiniGrid</td></tr><tr><td>Sup. Replay</td><td></td><td>Discuss.</td><td>Init. D</td><td>Resid. D</td><td>Discuss.</td><td>Init. D</td><td>Resid. D</td></tr><tr><td colspan="8">Qwen3-VL-8B</td></tr><tr><td>x</td><td>√</td><td>8.3</td><td>39.1</td><td>92.4</td><td>7.2</td><td>42.7</td><td>93.2</td></tr><tr><td>√</td><td>X</td><td>21.7</td><td>45.9</td><td>90.1</td><td>19.4</td><td>42.2</td><td>87.2</td></tr><tr><td>√</td><td>√</td><td>58.4</td><td>20.8</td><td>62.0</td><td>54.1</td><td>29.9</td><td>78.9</td></tr><tr><td colspan="8">Qwen3-VL-32B</td></tr><tr><td>X</td><td>√</td><td>10.5</td><td>31.7</td><td>90.5</td><td>9.4</td><td>21.3</td><td>91.9</td></tr><tr><td>√</td><td>X</td><td>26.8</td><td>35.9</td><td>88.0</td><td>24.2</td><td>35.6</td><td>87.2</td></tr><tr><td>√</td><td>√</td><td>71.6</td><td>10.5</td><td>13.3</td><td>68.3</td><td>18.7</td><td>64.7</td></tr><tr><td colspan="8">Qwen3-VL-32B → Qwen3-VL-8B</td></tr><tr><td>x</td><td>√</td><td>8.9</td><td>27.2</td><td>91.1</td><td>8.1</td><td>45.8</td><td>91.9</td></tr><tr><td>√</td><td>X</td><td>23.1</td><td>36.7</td><td>87.2</td><td>21.5</td><td>61.7</td><td>87.1</td></tr><tr><td>√</td><td>√</td><td>63.5</td><td>13.6</td><td>69.9</td><td>60.7</td><td>33.7</td><td>65.9</td></tr></table>

Table 23: State-assertion consistency and correctness: initial values and signed changes after discussion.
<table><tr><td rowspan="2" colspan="2">Training data</td><td colspan="3">Sokoban</td><td colspan="3">MiniGrid</td></tr><tr><td> $\mathbf { C } \mathbf { C } \pm \Delta$ </td><td> $\mathrm { C W } \pm \Delta$ </td><td> $\boldsymbol { \mathrm { D } } \pm \boldsymbol { \Delta }$ </td><td> $\mathbf { C } \mathbf { C } \pm \Delta$ </td><td> $\mathrm { C W } \pm \Delta$ </td><td> $\boldsymbol { \mathrm { D } } \pm \boldsymbol { \Delta }$ </td></tr><tr><td colspan="8"> $Q w e n 3 \ – V L \ – 8 B$ </td></tr><tr><td>X</td><td>√</td><td> $4 0 . 7 \substack { + 1 . 5 }$ </td><td> $1 9 . 8 \substack { + 1 . 5 }$ </td><td> $3 9 . 5 \mathrm { - 3 . 0 }$ </td><td> $3 8 . 4 \substack { + 1 . 6 }$ </td><td> $1 7 . 9 \mathrm { + } 1 . 5$ </td><td>43.7-3.1</td></tr><tr><td>√</td><td>X</td><td> $2 8 . 4 \substack { + 2 . 4 }$ </td><td> $2 8 . 0 { + } 2 . 0 $ </td><td>43.6-4.4</td><td> $3 6 . 2 \substack { + 3 . 3 }$ </td><td> $1 7 . 1 + 2 . 7$ </td><td>46.7-6.0</td></tr><tr><td>√</td><td>L</td><td> $7 2 . 6 \mathrm { + 5 . 1 }$ </td><td> $6 . 9 \substack { + 2 . 8 }$ </td><td>20.5-7.9</td><td> $6 7 . 0 { + } 4 . 1 $ </td><td> $4 . 0 { + } 2 . 2 $ </td><td>29.0-6.3</td></tr><tr><td colspan="8">Qwen3-VL-32B</td></tr><tr><td>X</td><td>√</td><td> $4 4 . 8 \substack { + 1 . 5 }$ </td><td> $2 4 . 9 \substack { + 1 . 5 }$ </td><td>30.4-3.0</td><td> $6 4 . 4 \substack { + 1 . 1 }$ </td><td> $1 0 . 3 { + } 1 . 1 $ </td><td>25.3-2.2</td></tr><tr><td>√</td><td>X</td><td> $3 5 . 4 \substack { + 2 . 3 }$ </td><td> $3 1 . 3 \substack { + 1 . 8 }$ </td><td>33.3-4.1</td><td> $3 8 . 2 \substack { + 2 . 5 }$ </td><td> $2 7 . 0 { + } 2 . 0 $ </td><td>34.8-4.5</td></tr><tr><td>√</td><td>√</td><td> $8 8 . 7 \substack { + 4 . 8 }$ </td><td> $2 . 7 { + } 1 . 3 $ </td><td>8.6-6.1</td><td> $7 8 . 7 { + 4 . 6 }$ </td><td> $4 . 5 \substack { + 2 . 0 }$ </td><td>16.8-6.6</td></tr><tr><td colspan="8"> $Q w e n 3 \ – V L \ – 3 2 B  Q w e n 3 \ – V L \ – 8 B$ </td></tr><tr><td>X</td><td>√</td><td> $4 8 . 0 { + } 1 . 2 $ </td><td> $2 4 . 3 \substack { + 1 . 3 }$ </td><td>27.8-2.5</td><td> $3 1 . 5 { + 2 . 1 }$ </td><td> $2 0 . 4 \substack { + 2 . 2 }$ </td><td>48.1-4.3</td></tr><tr><td>√</td><td>X</td><td> $3 1 . 5 { + 2 . 4 }$ </td><td> $3 4 . 1 \substack { + 2 . 0 }$ </td><td>34.4-4.4</td><td> $1 0 . 3 { + } 4 . 1 $ </td><td> $3 3 . 2 \substack { + 3 . 4 }$ </td><td>56.5-7.5</td></tr><tr><td>√</td><td>√</td><td> $8 4 . 0 \mathrm { + } 2 . 7 $ </td><td> $3 . 1 + 1 . 4$ </td><td>12.9-4.1</td><td> $6 1 . 1 \mathrm { + } 7 . 5$ </td><td> $4 . 3 \substack { + 4 . 0 }$ </td><td> $3 4 . 6 \mathrm { - } 1 1 . 5$ </td></tr></table>

## G CASE STUDIES

## G.1 SOKOBAN

![](images/a43cd6b055b53375f957be5a8a8cf2b330f79de9119c52ee8552ba219a2917ee.jpg)  
Figure 8: Sokoban: state mismatch and ConPAct coordination. Condensed recorded outputs for the same positioning sub-goal.

## G.2 MINIGRID

![](images/124eb6fd884810a79871fb19e8795d13362655f5f7f3c4b89dc3d87f5caa54ab.jpg)  
Figure 9: MiniGrid: state mismatch and ConPAct coordination. Condensed recorded outputs for the same gap-alignment sub-goal.

## G.3 CRAFTER

Sub-goal: Harvest the nearby tree down-left to increase wood from 1 to 2.

## State mismatch

![](images/65e9cdf1551c2beb1d47beeff19b820377db8f6db0f64948041ff04d092a4d3f.jpg)  
Actor: facing=left, front=tree. Action: do.  
Planner: “The first tree is gone. The left cell is clear; another tree is down-left.”

![](images/e9431cd9f007a1ae845b15b343f4837edd196fa3b662fdd9ff212ca9ac8108f6.jpg)

Result: Wood remains 1.

Actor: front=tree.   
Action: do again. Actor: “The tree is down-left; I need to line up.”   
Action:   
move down.

![](images/8e146c41786ef211a64b109e042b279999b52552953b5a6fc9eb7ffe303bb2a5.jpg)

## ConPAct coordination

![](images/11e50e87d5e542ca4454f82a01242bb210c90e048310bdd0a71a1140dc2f9013.jpg)

![](images/47bf6384a26e190f41ebb0e287163a2e1e3c1ac36d2b8ecdf45650964504db81.jpg)  
Result: Now moving toward the tree; wood is still 1. Sub-goal not yet achieved.  
Planner: “The first tree is gone. The left cell is clear; another tree is down-left.”  
Actor: facing=left, front=tree. Proposed action: do.

Contradiction: front Planner=walkable; Actor=tree “Recheck both readings; actions paused.”

## ConPAct → Actor & Planner

Actor: “I face left toward grass. The tree is down-left; I withdraw do.”

Planner: “Confirmed: left, walkable, full vitals, one wood, no other resources or tools. Move left, then face down.”

Actor: “Agreed; I will approach it.” Action: move left.

![](images/9bb43fe755486402aee2ad15b850a6cd40765a02dc043cdbfced21303c98c8ab.jpg)  
Actor: “The tree is directly below.” Action: move down (face the tree).

![](images/420982685eac15cbbc96759266b42a81d81e1c3d91105063a035245feec23ad2.jpg)  
Actor: “Facing down, tree ahead.” Action: do.

![](images/9c115b746368ea59c1fa27838b1670763c55b9a06060f9908c843aa1f73e5b3d.jpg)  
Actor: “The tree is gone; wood is 2.” Status: ACHIEVED.  
Figure 10: Crafter: state mismatch and ConPAct coordination. Condensed recorded outputs for the same second-wood sub-goal.

## G.4 PROCGEN MINER

Sub-goal: Move right one cell to position for digging a path to the diamonds.

## State mismatch

![](images/afa4cf9867fc672bf785d0bace098946251efcbd34ef1fe7851bd1e8c7efc75b.jpg)

Planner: “No boulder is immediately overhead. Start by moving right.”

Actor: “Boulders directly to the right make that move dangerous. I will move down.” Action: down.

![](images/07d5bce88ea2466f767426eef3f68216aa418b486f16d2686a18249587ceecc8.jpg)

Result: The player moves down one cell, remaining in the starting column. Sub-goal not achieved.

## ConPAct coordination

![](images/e67b3b03ef4e6ca1129066996c33b4ea91d561ad0b07f7e65f8e42f545a2e1d6.jpg)

Planner: “No boulder is immediately overhead. Start by moving right.”

Actor: “Boulders directly to the right make that move dangerous. I will move down.” Proposed action: down.

## ConPAct → Actor & Planner

Contradiction: hazard nearby Planner=no; Actor=yes “Recheck both readings; actions paused.”

Actor: “The adjacent right cell is dirt; the boulder column is farther away. I revise hazard nearby to no. I also count three visible diamonds.”

Planner: “I confirm the next right cell is clear and revise my diamond count to three. Diamonds are visible; no immediate falling hazard is apparent. Stop after one right step.”

Actor: “Agreed: target visible=yes, hazard nearby=no, three diamonds.” Action: right.

![](images/26a0a5d6a6031a1e6b857d50b8338d99ce284c650aaac023aedd4ca3deb9e25e.jpg)  
Actor: “I have moved one cell right. The positioning sub-goal is complete.” Status: ACHIEVED.

Figure 11: Procgen Miner: state mismatch and ConPAct coordination. Condensed recorded outputs for the same one-cell positioning sub-goal.

## H DISCUSSION

## H.1 INFERENCE COST

Table 24 summarizes the inference-time comparisons across the four game environments. We average rollout costs within each model–environment configuration and then aggregate across environments and evaluated configurations. Calls per action is the ratio of mean model calls to mean executed actions under this aggregation. Output-token averages use the runs with recorded usage; the coverage column reports the fraction with available usage information. All decision calls, including planning and discussion, count toward the common rollout budget.

Table 24: Inference costs averaged across the four environments and the evaluated inference-time configurations. Usage coverage is the fraction of recorded rollouts with output-token usage.
<table><tr><td>Method</td><td>Calls / episode</td><td>Output tokens / episode</td><td>Calls / action</td><td>Usage coverage (%)</td></tr><tr><td>ReAct (Planner)</td><td>14.02</td><td>3,095.37</td><td>1.04</td><td>71.5</td></tr><tr><td>ReAct (Actor)</td><td>36.29</td><td>8,703.19</td><td>1.00</td><td>71.5</td></tr><tr><td>Plan-Act</td><td>35.36</td><td>5,285.03</td><td>1.06</td><td>100.0</td></tr><tr><td>HiPlan</td><td>51.86</td><td>7,992.10</td><td>2.10</td><td>100.0</td></tr><tr><td>TAPE</td><td>37.84</td><td>18,900.17</td><td>6.29</td><td>99.9</td></tr><tr><td>ConPAct-I</td><td>33.02</td><td>7,199.86</td><td>1.43</td><td>83.3</td></tr></table>

## H.2 INTERPRETATION AND SCOPE

State assertions describe selected task-relevant facts rather than complete internal beliefs. Agreement alone does not ensure correctness, so the Sokoban and MiniGrid analyses evaluate the roles’ reports against environment ground truth as well as against each other. Procgen and OSWorld allow unknown values for unobservable facts; their task-performance results are not treated as substitutes for an assertion-correctness oracle. The GPT-5.6-sol reviewer curates ConPAct-R data offline and is not required by online ConPAct comparison or reconciliation.

Consistency and correctness. The relationship is asymmetric: when both roles correctly report the same definite facts about the same observation, they must agree, whereas agreement can also reflect a shared error. Disagreement therefore signals that at least one report needs revision, but does not identify which report is correct; agreement leaves correlated errors undetected. This distinction matters especially during reconciliation: a decrease in mismatch can arise either from correcting an erroneous report or from persuading a correct role to adopt an incorrect interpretation. The CC/CW/D analysis separates these outcomes by tracking whether agreement is grounded in the environment or merely shared between roles. Accordingly, reduced mismatch indicates closer alignment of the reported states, but supports a claim of improved state understanding only when accompanied by greater correctness.

Consistency also has a practical role beyond measuring individual accuracy: a planner and an actor need compatible state assumptions for a sub-goal to retain its meaning across their interaction. Yet this compatibility alone cannot establish that the sub-goal is feasible or the chosen action is effective. ConPAct thus addresses an observable source of coordination failure, while shared perceptual errors and planning errors remain distinct limitations. For unobservable fields, preserving uncertainty avoids replacing missing evidence with unsupported consensus; the absence of a definite conflict should not itself be interpreted as verified agreement.