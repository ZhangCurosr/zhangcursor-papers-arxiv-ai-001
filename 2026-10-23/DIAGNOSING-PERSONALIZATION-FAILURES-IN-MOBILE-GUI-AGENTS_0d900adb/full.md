# DIAGNOSING PERSONALIZATION FAILURES IN MOBILE GUI AGENTS

Zhixin Lin<sup>1,2</sup>, Dongliang Xu<sup>1,\*</sup>, Jungang Li<sup>3,4</sup>, Shidong Pan<sup>5</sup>, Jun Feng<sup>6</sup>, Yanbin Sun<sup>7</sup>, Jing Li<sup>8</sup>, Liang Liu<sup>2,†</sup>, Yue Yao<sup>1,\*</sup>

<sup>1</sup>Shandong University, <sup>2</sup>VIVO AI Lab, <sup>3</sup>The Hong Kong University of Science and Technology (GZ)

<sup>4</sup>The Hong Kong University of Science and Technology, <sup>5</sup>Indendpendent Researcher

<sup>6</sup>Institute of Automation, Chinese Academy of Sciences, <sup>7</sup>Guangzhou University

<sup>8</sup>University of Technology Sydney,

<sup>†</sup>Project lead. <sup>\*</sup>Corresponding authors.

## ABSTRACT

Unlike conventional GUI agents, personalized GUI agents require user preferences to guide a sequential execution policy over changing GUI states. This creates two challenges for the agentic design: 1) Determining HOW each preference should alter the agent’s execution strategy. 2) Determining WHEN each preference should take effect along the interaction trajectory. However, existing benchmarks primarily evaluate overall task success only, making it difficult to evaluate agents’ reasoning ability to determine HOW and WHEN preferences should affect execution. To address this, we introduce PrefDiag-Bench with two key components: 1) a categorization of user preferences into five types, characterizing HOW they affect agent execution; and 2) a stage-wise diagnostic framework with 1,664 probes, tracing WHEN preferences should take effect along the interaction trajectory and where failures occur. Experimentally, we evaluate eleven GUI agent systems, including end-to-end agents and agentic workflows, under two forms of user context: explicit user profiles and behavioral memory logs. We find while existing GUI agents achieve relatively high task completion rates on nonpersonalized tasks, their performance remains substantially lower on preferencepersonalized tasks. Even the best-performing model, Seed-2.0-Pro, achieves only 47.17% on process preferences under explicit user profiles. Moreover, agents struggle particularly with Ranking and Process preferences, indicating substantial room for improvement. Further analysis reveals two key capability bottlenecks behind these failures: selecting relevant evidence from user history, and recognizing when user preferences should take effect during execution. Controlled interventions that explicitly provide relevant evidence improve end-to-end task success by up to 7.06%. We hope PrefDiag-Bench can provide a more systematic and comprehensive evaluation of GUI agents’ personalization capabilities, moving beyond simply measuring overall task success. All code and dataset are available at https://github.com/Zhixin-L/PrefDiag-Bench.

## 1 INTRODUCTION

Recent mobile GUI agents powered by large language models (LLMs) have made substantial progress in GUI perception, task planning, and task execution in real-world applications, bringing them closer to personalized assistants. In such a personalization setting, simply completing a user request is no longer sufficient; agents also are required to draw on users’ past behavior, habits, and preferences to make decisions that meet individual needs during execution. For example, for the same base task “order a milk tea according to my usual preferences and add it to the cart” the agent may need to select the preferred drink options, satisfy a price constraint, prioritize a store with faster delivery, or compare multiple candidates. Therefore, the same GUI task may impose different execution requirements under different user preferences.

![](images/679bdf8cbad85e4339c8e383b85f5ad3194a4b1dde09a51762a0a4b80580ea09.jpg)  
Figure 1: Motivation for PrefDiag-Bench. Top: Different user preferences impose distinct execution requirements on mobile GUI agents, affecting desired outcomes, feasible choices, candidate priorities, execution procedures, and user interaction. Bottom: The interaction example illustrates that knowing a task-relevant preference is insufficient: the agent must also recognize the appropriate GUI state in which to apply it. PrefDiag-Bench complements end-to-end evaluation with stage-wise diagnostic probes across information processing, contextual reasoning, and action formulation to identify capability gaps in preference application.

The example above highlights two challenges for personalized GUI agents. First, an agent is required to determine HOW each preference should alter execution: it may change the desired outcome, restrict valid choices, change candidate priorities, or constrain the execution process or interaction behavior. Second, the agent is required to determine WHEN a preference should take effect along the interaction trajectory, based on the current task and GUI state. Figure 1 illustrates these two complementary challenges: preferences impose distinct execution requirements, while a task-relevant preference should only be applied when its trigger state is reached. These challenges require evaluation beyond task completion. A benchmark should assess whether an agent can determine HOW a preference should alter execution and WHEN it should take effect, while also identifying where failures arise along the path from user context to GUI actions. Existing evaluations centered on task-level success or preference-sensitive progress provide limited information about this process.

To address these challenges from a data-centric perspective, we introduce PrefDiag-Bench, a finegrained diagnostic benchmark for personalized GUI agents. PrefDiag-Bench contains two complementary components corresponding to the challenges above. First, while the benchmark spans diverse application domains, including shopping, workflow, and transportation, its main evaluation axis is the primary effect of each preference on agent execution. We organize user preferences into five execution-oriented types, i.e., Outcome, Constraint, Ranking, Process, and Interaction—characterizing HOW they affect agent execution, and construct corresponding end-to-end tasks to compare agents under different execution requirements. Second, we develop a stage-wise diagnostic framework that covers three high-level phases, i.e., information processing, contextual reasoning, and action formulation, and further divides them into five diagnostic stages. The framework probes the capabilities required to apply preferences from user context to GUI actions, including recognizing WHEN a preference should take effect, and identifies capability bottlenecks arise during personalized execution. Based on this design, PrefDiag-Bench contains 318 end-to-end GUI tasks and 1,664 associated diagnostic probes. Each end-to-end task is paired with a set of diagnostic probes built around the same underlying preference instance. This paired design allows us not only to evaluate whether an agent completes the task while satisfying the user’s preference in an interactive GUI environment, but also to connect end-to-end performance with specific diagnostic stages. In addition, we construct explicit user profiles and behavioral memory logs as two user-context representations at different levels of abstraction, enabling us to study how the form of user context affects personalized execution.

Experimentally, we systematically evaluate 11 existing popular personalized GUI agentic designs, including nine end-to-end agents and two agentic workflows, M3A and T3A, using explicit user profiles and behavioral memory logs. Our results indicate three main findings. First, performance varies substantially across the five preference types, showing that collapsing performance across preference types into a single aggregate score can obscure important capability differences. Second, explicit user profiles do not consistently outperform behavioral memory logs, suggesting that compact preference summaries are not always more useful than logs that retain concrete behavioral evidence. Third, selecting relevant evidence from user history and recognizing when preferences should take effect emerge as the two main bottlenecks. Controlled interventions on a representative subset of systems further show that explicit evidence guidance improves end-to-end task success by up to 7.06 percentage points, demonstrating that evidence selection has a measurable impact on personalized execution and provides a concrete target for system improvement.

## 2 RELATED WORK

## 2.1 MOBILE GUI AGENTS AND THEIR PERSONALIZED EVALUATION

Recent mobile GUI agents have made substantial progress in GUI perception, task planning, and multi-step task execution (Liu et al., 2025a; Nguyen et al., 2025; Hu et al., 2025). Existing systems include agent frameworks that organize task execution through explicit reasoning and control modules, such as AppAgent (Zhang et al., 2025a), M3A (Rawles et al.), and Mobile-Agent-v3 (Ye et al., 2025), as well as end-to-end agents that directly generate reasoning traces and actions from GUI observations, such as UI-TARS (Qin et al., 2025) and GUI-Owl (Ye et al., 2025). As the GUI capabilities of these systems continue to improve, recent studies have begun to examine how agents can use user history and preferences to adapt their behavior during task execution. PersonalAlign (Lyu et al., 2026) uses long-term user records to recover preferences omitted from vague instructions; PSPA-Bench (Nie et al., 2026) decomposes tasks to evaluate preference-sensitive progress; KnowU-Bench (Chen et al., 2026) uses behavioral logs to assess preference inference, user clarification, and proactive decision-making; and Persona2Web (Kim et al., 2026) studies contextual reasoning from long-term user histories in web environments. These studies broaden personalized agent evaluation from complementary directions. However, they do not jointly examine how different preferences affect GUI execution or which capabilities become bottlenecks when preferences are applied from user context to GUI actions. PrefDiag-Bench complements this line of work with an execution-oriented and stage-wise diagnostic perspective.

## 2.2 PERSONALIZATION USER CONTEXT AND MEMORY

Personalization relies on memory as a key source of user context, allowing agents to retain and reuse information from past interactions (Zhang et al., 2025b; Liu et al., 2025b). Memory systems such as MemoryBank (Zhong et al., 2024) and Mem0 (Chhikara et al., 2025) extract, update, and retrieve user information from ongoing interactions to support long-term personalization. Corresponding benchmarks examine this process from different perspectives. PersonaMem (Jiang et al., 2025) evaluates whether models can track evolving user profiles from multi-session interaction histories and generate personalized responses accordingly. AlpsBench (Xiao et al., 2026) decomposes personalized memory management into information extraction, updating, retrieval, and utilization. In mobile GUI environments, MemGUI-Bench (Liu et al., 2026) focuses on memory retention and use across steps, applications, and sessions.However, correctly inferring or retrieving user information does not necessarily mean that an agent can apply user preferences correctly in a GUI task. Personalized GUI agents must further connect user context with the current task and GUI state, and translate relevant preferences into concrete behaviors such as candidate selection, ranking, process control, or user interaction. PrefDiag-Bench complements these studies by comparing explicit user profiles and behavioral memory logs in GUI tasks and probing capability gaps from user-context processing to action formulation.

Table 1: Comparison with existing personalization benchmarks. EUP and BML denote explicit user profiles and behavioral memory logs, respectively. Personalization reasoning explicitly evaluates how preferences alter execution requirements and when they take effect as complementary dimensions. Specific diagnosis pairs individual task instances with dedicated capability probes.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Environment</td><td colspan="2">User Context</td><td rowspan="2">Personalization Reasoning</td><td rowspan="2">E2E GUI Tasks</td><td rowspan="2">Specific Diagnosis</td></tr><tr><td>EUP</td><td>BML</td></tr><tr><td>AlpsBench (Xiao et al., 2026)</td><td>Dialogue</td><td>√</td><td></td><td>X</td><td>X</td><td>X</td></tr><tr><td>AMemGym (Jiayang et al., 2026)</td><td>Dialogue</td><td>×</td><td></td><td>X</td><td>×</td><td>√</td></tr><tr><td>Persona2Web (Kim et al., 2026)</td><td>Web</td><td></td><td></td><td>X</td><td>√</td><td>X</td></tr><tr><td>PSPA-Bench (Nie et al., 2026)</td><td>Mobile</td><td></td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td>PersonalAlign (Lyu et al., 2026)</td><td>Mobile</td><td>×</td><td>√</td><td>X</td><td>×</td><td>×</td></tr><tr><td>KnowU-Bench (Chen et al., 2026)</td><td>Mobile</td><td>X</td><td>√</td><td>X</td><td>V</td><td>X</td></tr><tr><td>PrefDiag-Bench</td><td>Mobile</td><td>√</td><td>√</td><td>√</td><td>7</td><td>L</td></tr></table>

## 3 BENCHMARK DESIGN

PrefDiag-Bench evaluates personalized mobile GUI agents at both the task and diagnostic levels. It contains 318 end-to-end GUI tasks, including 53 Basic and 265 preference-conditioned instances and 1,664 diagnostic probes, where each end-to-end task is linked to probes constructed from the same underlying preference instance.

## 3.1 PREFERENCE TAXONOMY

User preferences concern different application domains and semantic content. For mobile GUI agents, however, a more direct distinction is the type of decision that each preference changes during task execution. We therefore use preferences’ primary effects on execution as our main evaluation axis, classifying each preference by the part of a personalized task’s success criterion that it directly changes. This yields five types, illustrated in Figure 1.

Specifically, as shown in the Figure 1, Outcome preference specifies the desired outcome or final choice, such as selecting “no sugar” and “less ice” in a drink-ordering task. A Constraint preference sets conditions that a valid outcome or allowed action must satisfy, such as keeping the drink price at or below 20\$. A Ranking preference specifies priorities among valid candidates, rather than determining whether a candidate satisfies a hard constraint. For example, when several stores offer the target drink, the agent should prioritize the store with the shortest delivery time and choose the cheaper store when delivery times are equal. A Process preference specifies how a task should be carried out, such as comparing the price of the same product across multiple platforms before making a choice. The final GUI state alone is insufficient to verify this requirement: the execution trajectory must also show that the prescribed procedure was followed. An Interaction preference specifies whether, when, and how the agent should interact with the user. For example, after finding a suitable item, the agent should request confirmation and add it to the cart only after the user agrees. Although both types affect the interaction trajectory, Process constrains the task procedure, whereas Interaction constrains communication with the user. A task may therefore reach the expected final state yet still violate its preference requirements if a required comparison or confirmation was skipped during execution.

## 3.2 BENCHMARK OVERVIEW AND STATISTICS

PrefDiag-Bench evaluates agents’ ability to apply user preferences across diverse mobile tasks. It comprises 53 task families, each with one Basic task and five preference-conditioned variants, yield ing 318 instances (53 Basic and 265 personalized). As shown in Figure 2(a), the tasks span seven application domains: communication, office, commerce, navigation, system settings, information retrieval, and social media. Within the diagnostic set, preference atoms also cover nine semantic categories, including workflow, user control, time and context, resource limits, and safety and privacy (Figure 2(c)). These categories describe what preferences concern, complementing the taxonomy that distinguishes how they alter agent behavior.

![](images/5df5cc6dd047dd30b8b66d4c9679824659d91764e5d75c5dc8b3ee01a45a9fc6.jpg)  
(a) Task type distribution

![](images/65f28f48a6021ad4db52a73a53715de15ecb573bfca709e88790e6b715da6a24.jpg)  
(b) Diagnostic probes composition

![](images/ff537fe64f61ef3ea0e16794530d1e696c223fcb29dc70252cdbd892b235165f.jpg)  
(c) Preference atom semantic distribution  
Figure 2: Statistics of PrefDiag-Bench. (a) Composition of end-to-end tasks across application domains. (b) Distribution of diagnostic probes across the five diagnostic stages. (c) Distribution of fine-grained preference atoms, showing the semantic diversity of user preferences covered by the benchmark.

Formally, we define a personalized instance as $\mathcal { T } _ { i } = ( I _ { i } , C _ { i } , E _ { i } ^ { 0 } , S _ { i } )$ , where $I _ { i }$ is the instruction, $C _ { i }$ is the user context conveying preference $p _ { i } , E _ { i } ^ { 0 }$ is the initial environment state, and $S _ { i }$ is a deterministic success checker. The agent is required to identify the relevant preference requirements from $C _ { i }$ and complete the task while satisfying all of them at the appropriate stages of execution. Execution produces a trajectory $\tau _ { i } = ( o _ { 1 } , a _ { 1 } , \dots , o _ { T } , a _ { T } )$ of observations and actions. The checker evaluates the final state $E _ { i } ^ { T }$ and relevant action or user–agent interaction records, returning $S _ { i } ( E _ { i } ^ { T } , \tau _ { i } ) \ \in$ {0, 1}. A result of 1 requires satisfying both the task goal and all preference-specific requirements, including any prescribed procedure or confirmation.

To examine the capabilities underlying these task outcomes, we include 1,664 diagnostic probes associated with 260 preference atoms from 256 tasks with observed failures under behavioral logs (Figure 2(b)). Each atom is the smallest preference requirement that can be diagnosed and applied independently. Across information processing, contextual reasoning, and action formulation, the probes test preference induction from logs (P1), evidence selection for a given preference (P2), task relevance (P3), state-level trigger recognition (P4), and semantic action selection for an active preference (P5). They are evaluated independently, with ground-truth prerequisites supplied when needed; This pairing connects task outcomes with tests of specific capabilities rather than describing personalization only through aggregate task success.

## 3.3 DATA CONSTRUCTION AND ANNOTATION

We construct preference-conditioned variants from selected KnowU-Bench tasks (Chen et al., 2026), inspecting their goals, application settings, and success criteria. Every resulting variant is manually reviewed. The target preference is conveyed through user context rather than restated in the instruction. We construct paired contexts by first drafting explicit user profiles with an LLM, revising and checking them with GPT-5.6 Sol, and reviewing every profile manually. Synthetic behavioral memory logs are then generated from the reviewed profiles, with four to five profile-related entries per collection, and undergo the same model-assisted revision and human review. The underlying preference, instruction, initial environment state, and success checker remain fixed across the two context forms.

We derive task-linked diagnostic probes from execution failures under behavioral logs, retaining cases with valid checker results and deduplicating them across agents by task. Independently diagnosable preference requirements are separated into atoms. Probe questions, options, and reference annotations are drafted with LLM assistance and revised by humans. For P4 and P5, we use atoms with clear trigger boundaries and usable reference trajectories. P5 reference actions are extracted from successful rollouts of the corresponding tasks and represented as semantic operations. Human reviewers answer all multiple-choice probes.

Table 2: End-to-end Task Success Rates (TSR, %) on PrefDiag-Bench. Basic reports traditional non-personalized task success rates averaged over the 53 base tasks. The remaining columns report personalization task success rates across five preference types under explicit user profiles and behavioral memory logs, together with their averages. Constr. and Interact. denote Constraint and Interaction, respectively. Within each category, systems are sorted in ascending order by the mean of the two AVG scores. The best and second-best results in each column are highlighted across all systems.
<table><tr><td rowspan="2">System</td><td rowspan="2">Basic</td><td colspan="6">Explicit User Profiles</td><td colspan="6">Behavioral Memory Logs</td></tr><tr><td>Outcome</td><td></td><td>Constr. Ranking</td><td>Process</td><td>Interact.</td><td>AVG</td><td>Outcome</td><td>Constr.</td><td>Ranking</td><td>Process</td><td>Interact.</td><td>AVG</td></tr><tr><td colspan="10">Agentic workflows</td><td></td><td></td><td></td><td></td></tr><tr><td>T3A</td><td>41.51</td><td>37.74</td><td>30.19</td><td>13.21</td><td>15.09</td><td>28.30</td><td>24.91</td><td>43.40</td><td>30.19</td><td>22.64</td><td>18.87</td><td>32.08</td><td>29.44</td></tr><tr><td>M3A</td><td>54.72</td><td>47.17</td><td>50.94</td><td>33.96</td><td>16.98</td><td>39.62</td><td>37.73</td><td>28.30</td><td>20.75</td><td>16.98</td><td>9.43</td><td>16.98</td><td>18.49</td></tr><tr><td colspan="10">End-to-end agents</td><td colspan="3"></td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>30.19</td><td>20.75</td><td>11.32</td><td>13.21</td><td>3.77</td><td>9.43</td><td>11.70</td><td>22.64</td><td>20.75</td><td>11.32</td><td>3.77</td><td>1.89</td><td>12.07</td></tr><tr><td>Step-GUI-4B</td><td>37.74</td><td>33.96</td><td>13.21</td><td>15.09</td><td>3.77</td><td>0.00</td><td>13.21</td><td>30.19</td><td>13.21</td><td>16.98</td><td>1.89</td><td>0.00</td><td>12.45</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>35.85</td><td>28.30</td><td>13.21</td><td>16.98</td><td>13.21</td><td>3.77</td><td>15.09</td><td>33.96</td><td>16.98</td><td>15.09</td><td>7.55</td><td>16.98</td><td>18.11</td></tr><tr><td>MAI-UI-8B</td><td>58.49</td><td>45.28</td><td>26.42</td><td>24.53</td><td>13.21</td><td>11.32</td><td>24.15</td><td>56.60</td><td>18.87</td><td>22.64</td><td>13.21</td><td>13.21</td><td>24.91</td></tr><tr><td>GUI-Owl-1.5-8B-Instruct 56.60</td><td></td><td>47.17</td><td>32.08</td><td>18.87</td><td>16.98</td><td>22.64</td><td>27.55</td><td>50.94</td><td>15.09</td><td>26.42</td><td>15.09</td><td>15.09</td><td>24.53</td></tr><tr><td>UI-Venus-1.5-8B</td><td>52.83</td><td>39.62</td><td>35.85</td><td>24.53</td><td>13.21</td><td>26.42</td><td>27.93</td><td>47.17</td><td>28.30</td><td>28.30</td><td>20.75</td><td>22.64</td><td>29.43</td></tr><tr><td>Qwen3.7-Plus</td><td>67.92</td><td>56.60</td><td>56.60</td><td>41.51</td><td>22.64</td><td>45.28</td><td>44.53</td><td>62.26</td><td>49.06</td><td>33.96</td><td>16.98</td><td>47.17</td><td>41.89</td></tr><tr><td>Claude Sonnet 4.6</td><td>86.79</td><td>71.70</td><td>69.81</td><td>58.49</td><td>35.85</td><td>41.51</td><td>55.47</td><td>75.47</td><td>60.38</td><td>47.17</td><td>39.62</td><td>45.28</td><td>53.58</td></tr><tr><td>Seed-2.0-Pro</td><td>81.13</td><td>79.25</td><td>66.04</td><td>56.60</td><td>47.17</td><td>64.15</td><td>62.64</td><td>83.02</td><td>62.26</td><td>45.28</td><td>60.38</td><td>77.36</td><td>65.66</td></tr></table>

## 4 EXPERIMENTS

We organize our experiments around the following five research questions.

RQ1. How well do GUI agents satisfy different preference requirements during task execution?

RQ2. What strengths and weaknesses do agents exhibit across stages of preference application?

RQ3. What relationships emerge between task-level diagnostic results and end-to-end outcomes?

RQ4. What effect does evidence guidance have on end-to-end task success?

RQ5. What changes in performance occur under different memory construction methods?

## 4.1 EXPERIMENTAL SETUP

Evaluated Systems. We evaluate 11 systems. The closed-source agents are Seed-2.0-Pro (Seed, 2026), Claude Sonnet 4.6 (Anthropic, 2026), and Qwen3.7-Plus (Alibaba Cloud, 2026). The open-source agents are MAI-UI-8B (Zhou et al., 2025), Qwen3-VL-8B-Instruct and Qwen3-VL-8B-Thinking (Team, 2025), Step-GUI-4B (GELab-Team & StepFun, 2025), GUI-Owl-1.5-8B-Instruct (Xu et al., 2026), and UI-Venus-1.5-8B (Team et al., 2026). We also evaluate two agentic workflows, M3A and T3A (Rawles et al.), both using GPT-5.4. Open-source models run on a server with eight NVIDIA RTX 4090 GPUs.

Evaluation Protocol. All 11 systems are evaluated on 53 Basic tasks and 265 preferenceconditioned tasks, the latter under both explicit user profiles and behavioral memory logs. All systems share task environments and checkers while retaining their native observation and action interfaces. Instructions, initial states, and checkers remain fixed across profile–log pairs. We report Task Success Rate (TSR) by preference type and context form. Independent P1–P5 probes are applied to the nine directly queried agents; workflows are excluded to avoid assessing only their backbone or introducing workflow-specific probe adaptations. Diagnostic results are aggregated by task, preference, and probe type: a result is correct only if all associated items are answered correctly. We report accuracy per probe type and an overall score across all task-level diagnostic results.

![](images/9c32de8cb7028e4ec9b7053999dc198c4e5253d50f07ac436cd01d20c2072132.jpg)  
Figure 3: The heatmaps show P2 Evidence Selection, P3 Relevance Selection, and P4 Trigger Recognition. Rows represent evaluated systems and columns represent the five preference types. Darker colors indicate higher task-level accuracy.

## 4.2 END-TO-END RESULTS

Table 2 reports TSR on the 53 Basic tasks and the 265 preference-conditioned tasks. For each personalized task, we compare explicit user profiles with behavioral memory logs while keeping the task instruction, initial environment state, and success checker fixed. The AVG columns report the macro-average over the five preference types.

Personalized tasks are more challenging to execute. All 11 systems achieve lower average TSR on preference-conditioned tasks than on Basic tasks under both forms of user context. Seed-2.0- Pro obtains the strongest personalized results, reaching 62.64% with explicit profiles and 65.66% with behavioral logs, compared with 81.13% on Basic. In contrast, Claude Sonnet 4.6 achieves the highest Basic TSR of 86.79%, but is surpassed by Seed-2.0-Pro under both personalized settings. This ranking change shows that strong general GUI execution is important, but is not sufficient for applying user preferences throughout a multi-step interaction trajectory.

Different preference types expose distinct performance patterns. Outcome is the bestperforming type for 10 of the 11 systems under explicit user profiles and for all 11 systems under behavioral memory logs. At the other end, Ranking and Process emerge as recurring difficulties across systems and user-context forms. Process is the lowest-performing type for seven systems under explicit profiles and the lowest or tied-lowest type for eight systems under behavioral memory logs. Under behavioral memory logs, Qwen3.7-Plus drops from 62.26% on Outcome to 16.98% on Process, a gap of 45.28 percentage points. Ranking remains difficult even for the strongest system: Seed-2.0-Pro achieves 83.02% on Outcome but only 45.28% on Ranking under behavioral memory logs, a difference of 37.74 points. These results show that reporting only an overall personalized score would hide major differences in the execution requirements that each system can handle. The repeated difficulty on Ranking and Process is consistent with the additional candidate comparison required by Ranking and the longer execution procedure required by Process.

More explicit user context is not always more effective. The profile–log comparison has no consistent direction across systems. Behavioral logs improve Seed-2.0-Pro from 62.64% to 65.66% and T3A from 24.91% to 29.44%, whereas M3A drops from 37.73% with profiles to 18.49% with logs. This contrast is notable because M3A and T3A use the same GPT-5.4 backbone, suggesting that the usefulness of a context form also depends on how the overall workflow processes user information. This pattern is consistent with the trade-off built into the two representations: profiles directly provide compact preference rules, whereas logs preserve concrete behavioral evidence but require the agent to identify which evidence is relevant. These task-level comparisons reveal performance gaps across execution requirements and context forms, but do not isolate the capabilities underlying preference application. We next examine these capabilities using independent probes constructed from failures under behavioral memory logs.

![](images/cf9a185db34ef6e1b70989a50a19ba3d1fd586558a60ece84ea2e881f980c32a.jpg)  
Percentage (%)  
Figure 4: Joint outcomes between end-to-end task execution and stage-wise diagnostic probes. For each system and diagnostic stage, we partition paired task-probe outcomes into four categories: TT (both pass), TF (the task passes / the probe fails), FT (the task fails / the probe passes), and FF (both fail). The distributions reveal how failures identified by individual diagnostic stages relate to failures in complete task execution.

## 4.3 STAGE-WISE DIAGNOSTIC RESULTS

Figure 3 reports task-level accuracy on P2–P4 for the nine directly evaluated agents. P1 and P5 are omitted from the heatmaps because they are near ceiling for most systems, but are included in the aggregate statistics below.

Evidence selection and trigger recognition are the weakest measured stages. On the failurederived diagnostic set, P2 Evidence Selection and P4 Trigger Recognition are the lowest-scoring stages across systems, averaging 40.06% and 39.69%, respectively, compared with 88.80% for P3 and over 98.9% for P1 and P5. Even Seed-2.0-Pro reaches only 71.48% on P2 and 58.14% on P4. Since both probes provide the target preference, these results highlight the need to improve agents’ ability to identify supporting evidence in user history and recognize when a known preference should take effect.

The Thinking variant stands out among open-source agents. Qwen3-VL-8B-Thinking achieves the highest accuracy among the evaluated open-source agents on both P2 and P4. On P2, it scores 69.53%, compared with 21.88% for Qwen3-VL-8B-Instruct; on P4, the corresponding scores are 45.79% and 25.58%. Its P2 accuracy also approaches the 71.48% achieved by Seed-2.0-Pro and Qwen3.7-Plus, narrowing the gap with the leading closed-source systems in evidence selection.These results suggest that the Thinking variant’s reasoning capabilities contribute to its higher accuracy in evidence selection and trigger recognition.

## 4.4 JOINT ANALYSIS

For each of the nine directly evaluated agents, we pair every task-level diagnostic result with the corresponding end-to-end outcome under behavioral memory logs. Since the diagnostic set is deduplicated across agents, a task included due to one agent’s failure may still be completed by another agent. Figure 4 partitions the eligible paired instances into four joint outcomes: both task and probe pass (TT), the task passes but the probe fails (TF), the task fails but the probe passes (FT), and both task and probe fail (FF).

P2 and P4 failures frequently co-occur with task failures. P2 Evidence Selection and P4 Trig ger Recognition show markedly higher joint-failure proportions than the other diagnostic stages. Qwen3-VL-8B-Instruct fails both the task and its associated probe on 65.5% of P2 pairs and 61.7% of P4 pairs. The corresponding proportions are 68.1% and 51.9% for GUI-Owl-1.5-8B-Instruct, and 82.1% and 66.0% for Step-GUI-4B. These results connect the weaknesses identified by the independent probes to failures on the corresponding GUI tasks, reinforcing evidence selection and trigger recognition as important targets for improvement.

Table 3: Additional analyses on PrefDiag-Bench (TSR, %). (a) Performance across C0–C2 settings. C0 uses the original logs; C1 retains all logs and labels each as supporting or noise for the target preference; C2 retains only supporting logs. (b) End-to-end TSR under behavioral memory logs using different memory modules. Best and second-best results in each column are highlighted with darker and lighter purple shading, respectively.  
(a) Performance across C0–C2 settings.
<table><tr><td>System</td><td>C0</td><td>C1</td><td>C2</td><td>AVG</td></tr><tr><td>Step-GUI-4B</td><td>14.12</td><td>18.82</td><td>18.82</td><td>17.25</td></tr><tr><td>Qwen3-VL-Thinking-8B</td><td>17.65</td><td>16.47</td><td>18.82</td><td>17.65</td></tr><tr><td>Qwen3.7-Plus</td><td>50.59</td><td>56.47</td><td>51.76</td><td>52.94</td></tr><tr><td>Seed-2.0-Pro</td><td>76.47</td><td>83.53</td><td>81.18</td><td>80.39</td></tr></table>

(b) Results with different memory modules.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=5>OutcomeConstraintRankingProcessInteraction</td><td rowspan=1 colspan=1>AVG</td></tr><tr><td rowspan=1 colspan=1>Mem0</td><td rowspan=1 colspan=2>71.70   56.60</td><td rowspan=1 colspan=1>43.40</td><td rowspan=1 colspan=1>50.94</td><td rowspan=1 colspan=1>54.72</td><td rowspan=1 colspan=1>55.47</td></tr><tr><td rowspan=1 colspan=1>LangMem</td><td rowspan=1 colspan=1>60.38</td><td rowspan=1 colspan=1>58.49</td><td rowspan=1 colspan=1>41.51</td><td rowspan=1 colspan=1>43.40</td><td rowspan=1 colspan=1>49.06</td><td rowspan=1 colspan=1>50.57</td></tr><tr><td rowspan=1 colspan=1>Hindsight</td><td rowspan=1 colspan=1>79.25</td><td rowspan=1 colspan=1>54.72</td><td rowspan=1 colspan=1>45.28</td><td rowspan=1 colspan=1>37.74</td><td rowspan=1 colspan=1>60.38</td><td rowspan=1 colspan=1>55.47</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=2>83.02   62.26</td><td rowspan=1 colspan=1>45.28</td><td rowspan=1 colspan=1>60.38</td><td rowspan=1 colspan=1>77.36</td><td rowspan=1 colspan=1>65.66</td></tr></table>

## 4.5 CONTROLLED EVIDENCE INTERVENTION

To test whether P2 Evidence Selection is an actionable bottleneck, we evaluate Seed-2.0-Pro, Qwen3.7-Plus, Qwen3-VL-8B-Thinking, and Step-GUI-4B on a fixed subset of 85 tasks selected from the 265 preference-conditioned tasks. Table 3a compares three conditions: C0 Raw Logs uses the original logs; C1 Oracle Evidence Labels retains all entries and labels each as supporting or noise for the target preference, without changing its text or order; C2 Supporting Logs Only removes all noise entries and retains only supporting logs. C1 tests whether explicitly identifying supporting and noise logs helps execution, while C2 tests whether removing noise is more effective than retaining and labeling it. The task instruction, initial environment state, and success checker remain fixed.

Evidence labeling can outperform noise removal. From C0 to C1, Seed-2.0-Pro improves from 76.47% to 83.53%, a gain of 7.06 percentage points, and Qwen3.7-Plus improves from 50.59% to 56.47%. These gains show that explicitly distinguishing supporting logs from noise can help agents complete personalized tasks. However, replacing labeled logs with supporting-only logs (C2) reduces their TSR to 81.18% and 51.76%, respectively. For these systems, identifying which logs support the preference is therefore more effective than simply discarding the rest. A possible explanation is that labeled noise provides contrastive cues about which historical behaviors should not guide the current decision, whereas deleting these entries removes those cues.

Limited GUI capability may constrain intervention benefits. Qwen3-VL-8B-Thinking achieves 69.53% on P2 and 45.79% on P4, but only 30.19% on Basic tasks and 12.07% on the full preference-conditioned set under behavioral logs (Table 2). On the 85-task intervention subset, its TSR remains low: 17.65% under C0, 16.47% under C1, and 18.82% under C2. This contrast suggests that limitations in GUI execution constrain the gains from evidence guidance, even when evidence selection is relatively strong. Step-GUI-4B also reaches only 18.82% under both C1 and C2. These results suggest that strong GUI execution is a necessary foundation for personalized task success: when an agent’s basic GUI capability is limited, even accurate preference understanding may not translate into successful end-to-end execution.

## 4.6 ROBUSTNESS TO MEMORY CONSTRUCTION METHODS

To test whether the preference-type differences in Section 4.2 depend on our original memory construction, we evaluate Seed-2.0-Pro on all 265 preference-conditioned tasks using alternative memory inputs produced by Mem0 (Chhikara et al., 2025), LangMem (LangChain, 2025), and Hindsight (Latimer et al., 2026) from the same source behavioral records. The task instruction, initial environment state, success checker, and execution settings remain fixed. Table 3b reports TSR by preference type, with the original behavioral memory logs included as OURS.

Preference-type differences persist across memory construction methods. Outcome remains the highest-scoring type, while Ranking and Process are the two lowest-scoring types under all three methods. With Mem0, TSR is 71.70% on Outcome, compared with 43.40% on Ranking and 50.94% on Process. Hindsight shows the same broad contrast, reaching 79.25% on Outcome, 45.28% on Ranking, and 37.74% on Process. Although the absolute scores and the ordering of Ranking and Process change, their relative difficulty persists across the tested methods. This consistency further supports our preference-conditioned task design, as the observed differences across preference types persist under alternative memory construction methods rather than being specific to our original behavioral memory logs.

## 5 CONCLUSION

We introduce PrefDiag-Bench to provide a more systematic and comprehensive view of personalization in mobile GUI agents. By categorizing user preferences into five types and combining this categorization with a stage-wise diagnostic framework, it examines how preferences affect agent execution, when they should take effect, and where capability gaps arise. Our evaluation shows that overall task success alone can obscure distinct strengths and weaknesses across preference types and stages of preference application. Connecting these capability-level assessments with end-to-end outcomes provides a basis for more informative evaluation and targeted interventions. We hope PrefDiag-Bench supports the development of GUI agents that not only understand user preferences but also apply them reliably throughout task execution.

## AI USE STATEMENT

In this work, we used generative AI tools (GPT-5.6 Sol, Extra high) to assist with synthetic data generation and translation, as well as language editing and polishing. AI-generated synthetic data were manually reviewed and revised by the authors before being included in the benchmark. All other AI-assisted content was also reviewed by the authors. The authors take full responsibility for the final content of this work.

## REFERENCES

Alibaba Cloud. Qwen3.7-plus model documentation. https://www.alibabacloud.com/ help/en/model-studio/qwen3-7-plus, 2026.

Anthropic. Claude sonnet 4.6. https://www.anthropic.com/news/ claude-sonnet-4-6, 2026.

Tongbo Chen, Zhengxi Lu, Zhan Xu, Guocheng Shao, Shaohan Zhao, Fei Tang, Yong Du, Kaitao Song, Yizhou Liu, Yuchen Yan, Wenqi Zhang, Xu Tan, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Knowu-bench: Towards interactive, proactive, and personalized mobile agent evaluation, 2026. URL https://arxiv.org/abs/2604.08455.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready AI agents with scalable long-term memory. 2025. URL https://doi.org/ 10.3233/FAIA251160.

GELab-Team and StepFun. Step-gui technical report. 2025. URL https://doi.org/10. 48550/arXiv.2512.15431.

Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, Yuhuai Li, Shengze Xu, Shenzhi Wang, Xinchen Xu, Shuofei Qiao, Zhaokai Wang, Kun Kuang, Tieyong Zeng, Liang Wang, Jiwei Li, Yuchen Eleanor Jiang, Wangchunshu Zhou, Guoyin Wang, Keting Yin, Zhou Zhao, Hongxia Yang, Fan Wu, Shengyu Zhang, and Fei Wu. OS agents: A survey on MLLM-based agents for computer, phone and browser use. Vienna, Austria, July 2025. URL https://aclanthology.org/2025. acl-long.369/.

Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J. Taylor, and Dan Roth. Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale. In Proc. of the Conference on Language Modeling (COLM), 2025. URL https://cogcomp.seas.upenn.edu/papers/ JHCLYCea25.pdf.

Cheng Jiayang, Dongyu Ru, Lin Qiu, Yiyang Li, Xuezhi Cao, Yangqiu Song, and Xunliang Cai. Amemgym: Interactive memory benchmarking for assistants in long-horizon conversations. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 4413–4446, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 0856bc553d3e3b9827e5140d0ad3bf8d-Paper-Conference.pdf.

Serin Kim, Sangam Lee, and Dongha Lee. Persona2web: Benchmarking personalized web agents for contextual reasoning with user history, 2026. URL https://arxiv.org/abs/2602. 17003.

LangChain. Langmem sdk for agent long-term memory. https://www.langchain.com/ blog/langmem-sdk-launch, 2025. Accessed: 2026-09.

Christopher Latimer, Nicolo Boschi, Andrew Neeser, Chris Bartholomew, Gaurav Srivastava, Xuan\` Wang, and Naren Ramakrishnan. Hindsight: Structured agent memory that retains, recalls, and reflects. 2026. URL https://aclanthology.org/2026.acl-demo.27/.

Guangyi Liu, Pengxiang Zhao, Yaozhen Liang, Liang Liu, Yaxuan Guo, Han Xiao, Weifeng Lin, Yuxiang Chai, Yue Han, Shuai Ren, Hao Wang, Xiaoyu Liang, WenHao Wang, Tianze Wu, Zhengxi Lu, Siheng Chen, LiLinghao, Hao Wang, Guanjing Xiong, Yong Liu, and Hongsheng Li. Llm-powered gui agents in phone automation: Surveying progress and prospects, 2025a. URL https://arxiv.org/abs/2504.19838.

Guangyi Liu, Pengxiang Zhao, Yaozhen Liang, Qinyi Luo, Shunye Tang, Yuxiang Chai, Weifeng Lin, Han Xiao, WenHao Wang, Siheng Chen, Zhengxi Lu, Gao Wu, Hao Wang, Liang Liu, and Yong Liu. Memgui-bench: Benchmarking memory of mobile gui agents in dynamic environments, 2026. URL https://arxiv.org/abs/2602.06075.

Jiahong Liu, Zexuan Qiu, Zhongyang Li, Quanyu Dai, Wenhao Yu, Jieming Zhu, Minda Hu, Menglin Yang, Tat-Seng Chua, and Irwin King. A survey of personalized large language models: Progress and future directions, 2025b. URL https://arxiv.org/abs/2502.11528.

Yibo Lyu, Gongwei Chen, Rui Shao, Weili Guan, and Liqiang Nie. PersonalAlign: Hierarchical implicit intent alignment for personalized GUI agent with long-term user-centric records. 2026. URL https://aclanthology.org/2026.acl-long.1669/.

Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, Xintong Li, Jing Shi, Hongjie Chen, Viet Dac Lai, Zhouhang Xie, Sungchul Kim, Ruiyi Zhang, Tong Yu, Mehrab Tanjim, Nesreen K. Ahmed, Puneet Mathur, Seunghyun Yoon, Lina Yao, Branislav Kveton, Jihyung Kil, Thien Huu Nguyen, Trung Bui, Tianyi Zhou, Ryan A. Rossi, and Franck Dernoncourt. GUI agents: A survey. 2025. URL https://aclanthology.org/2025.findings-acl.1158/.

Hongyi Nie, Xunyuan Liu, Yudong Bai, Yaqing Wang, Yang Liu, Quanming Yao, and Zhen Wang. Pspa-bench: A personalized benchmark for smartphone gui agent, 2026. URL https: //arxiv.org/abs/2603.29318.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

Chris Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 01a83bc2f2732a58e6aa731e659e7101-Paper-Conference.pdf.

Bytedance Seed. Seed2.0 model card: Towards intelligence frontier for real-world complexity, 2026. URL https://arxiv.org/abs/2607.00248.

Qwen Team. Qwen3-vl technical report. CoRR, 2025. URL https://doi.org/10.48550/ arXiv.2511.21631.

Venus Team, Changlong Gao, Zhangxuan Gu, Yulin Liu, Xinyu Qiu, Shuheng Shen, Yue Wen, Tianyu Xia, Zhenyu Xu, Zhengwen Zeng, Beitong Zhou, Xingran Zhou, Weizhi Chen, Sunhao Dai, Jingya Dou, Yichen Gong, Yuan Guo, Zhenlin Guo, Feng Li, Qian Li, Jinzhen Lin, Yuqi Zhou, Linchao Zhu, Liang Chen, Zhenyu Guo, Changhua Meng, and Weiqiang Wang. Ui-venus-1.5 technical report, 2026. URL https://arxiv.org/abs/2602.09082.

Jianfei Xiao, Xiang Yu, Chengbing Wang, Wuqiang Zheng, Xinyu Lin, Kaining Liu, Hongxun Ding, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. Alpsbench: An llm personalization benchmark for real-dialogue memorization and preference alignment. 2026. URL http://dx. doi.org/10.1145/3805712.3808634.

Haiyang Xu, Xi Zhang, Haowei Liu, Junyang Wang, Zhaozai Zhu, Shengjie Zhou, Xuhao Hu, Feiyu Gao, Junjie Cao, Zihua Wang, Zhiyuan Chen, Jitong Liao, Qi Zheng, Jiahui Zeng, Ze Xu, Shuai Bai, Junyang Lin, Jingren Zhou, and Ming Yan. Mobile-agent-v3.5: Multi-platform fundamental gui agents, 2026. URL https://arxiv.org/abs/2602.16855.

Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei Huang, Jingren Zhou, and Ming Yan. Mobile-agent-v3: Fundamental agents for gui automation, 2025. URL https://arxiv.org/ abs/2508.15144.

Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. 2025a. URL https://doi. org/10.1145/3706598.3713600.

Zhehao Zhang, Ryan A. Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim, Ruiyi Zhang, Jiuxiang Gu, Tyler Derr, Hongjie Chen, Junda Wu, Xiang Chen, Zichao Wang, Subrata Mitra, Nedim Lipka, Nesreen K. Ahmed, and Yu Wang. Personalization of Large Language Models: A Survey. Transactions on Machine Learning Research, 2025b. URL https://mlanthology.org/tmlr/2025/ zhang2025tmlr-personalization/.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. 2024. URL https://doi.org/10.1609/ aaai.v38i17.29946.

Hanzhang Zhou, Xu Zhang, Panrong Tong, Jianan Zhang, Liangyu Chen, Quyu Kong, Chenglin Cai, Chen Liu, Yue Wang, Jingren Zhou, and Steven Hoi. Mai-ui technical report: Real-world centric foundation gui agents, 2025. URL https://arxiv.org/abs/2512.22047.

## A DATASET DETAILS AND EXAMPLES

This appendix presents the task catalog, paired user contexts, and examples of P1–P5. Task families are identified as PREFD 01–53.

## A.1 TASK CATALOG AND EXAMPLES

Task sources and catalog. Table 4 lists the 53 Basic tasks adapted from KnowU-Bench (Chen et al., 2026), with one row per family. We inspect each task’s instruction, initial state, and completion conditions, adapting candidates, delivery information, messages, calendar events, and local information pages where needed. Each family has five preference-conditioned variants, yielding 318 end-to-end instances. Profiles and behavioral logs provide alternative contexts for the same 265 personalized tasks. The catalog lists the applications used by each Basic task; variants require additional applications, such as for cross-platform comparison.

Table 4: Catalog of the 53 Basic tasks. Each family also contains five preference-conditioned variants. The application column lists the Basic task’s applications, not every application available in its environment.
<table><tr><td>Family ID</td><td>Application(s)</td><td>Domain</td><td>Base task description</td></tr><tr><td>PrefD 01</td><td>Taodian</td><td>Commerce</td><td>Purchase one case of canned sugar-free Coca-Cola, enter the specified recipient details, and submit the order.</td></tr><tr><td>PrefD 02</td><td>Taodian</td><td>Commerce</td><td>Purchase white everyday commuting sneakers in size 42, enter the specified recipient details, and submit the order.</td></tr><tr><td>PrefD 03</td><td>Taodian</td><td>Commerce</td><td>Purchase an Apple MacBook Air with an M-series processor, at least 16 GB RAM, and a price no greater than CNY 10,000.</td></tr><tr><td>PrefD 04</td><td>Taodian</td><td>Commerce</td><td>Remove all clothing with a unit price above CNY 140 from the cart; exclude shoes and preserve all other items.</td></tr><tr><td>PrefD 05</td><td>Taodian</td><td>Commerce</td><td>Remove black clothing and all bodysuits from the cart while leaving other products unchanged.</td></tr><tr><td>PrefD 06</td><td>Chilemei</td><td>Commerce</td><td>Order a light Japanese- or Beijing-style lunch without peanuts, costing at most CNY 35, and complete simulated payment.</td></tr><tr><td>PrefD 07</td><td>Chilemei</td><td>Commerce</td><td>Order kung pao chicken with rice, add the note “no peanuts, less chili, extra rice," and complete simulated payment.</td></tr><tr><td>PrefD 08</td><td>Taodian</td><td>Commerce</td><td>Purchase a sofa and add an order note requesting shipment by SF Express.</td></tr><tr><td>PrefD 09</td><td>Chrome</td><td>Information retrieval</td><td>Visit the AlphaXiv and Hugging Face Papers trending-paper pages and finish on Hugging Face Papers.</td></tr><tr><td>PrefD 10</td><td>Chrome</td><td>Information retrieval</td><td>Search for today's Hangzhou weather, inspect precipitation probability and temperature extremes, and remain on the result page.</td></tr><tr><td>PrefD 11</td><td>Maps</td><td>Navigation</td><td>Plan a sunny-day bicycle commute from Zhejiang University's Zijingang campus to its science park; report the route, distance, and duration.</td></tr><tr><td>PrefD 12</td><td>Maps</td><td>Navigation</td><td>Plan the campus-to-science-park commute in rainy weather, minimizing exposure to rain, and report a</td></tr><tr><td>PrefD 13</td><td>Maps</td><td>Navigation</td><td>travel plan. Plan a taxi trip from the science park to Zijingang campus at 23:30, after bus and metro services have</td></tr><tr><td>PrefD 14</td><td>Maps; Mattermost</td><td>Navigation</td><td>Plan the fastest campus-to-science-park route and post a message in Town Square reporting an approximately 20-minute delay.</td></tr><tr><td>PrefD 15</td><td>Calendar; Mattermost</td><td>Office</td><td>Create a project-discussion meeting for 14:00–15:00 on February 10, 2026, and notify the team-chat channel.</td></tr><tr><td>PrefD 16</td><td>Messages; Calendar</td><td>Office</td><td>Accept Alex's 10:30–11:00 project-sync invitation on February 10, 2026, reply by SMS, and add the meeting to the calendar.</td></tr><tr><td>PrefD 17</td><td>Files; Docreader</td><td>Office</td><td>Find and open the specified Agent Learning via Early Experience PDF in Documents before the Product</td></tr><tr><td>PrefD 18</td><td>Clock</td><td>Office</td><td>Review meeting. Create an 08:30 Saturday-and-Sunday alarm with Bright Morning; retain the default ringtone if that</td></tr><tr><td>PrefD 19</td><td>Clock</td><td>Office</td><td>choice is unavailable. Turn off the existing 07:30 alarm before the weekend.</td></tr><tr><td>PrefD 20</td><td>Settings</td><td>System settings</td><td>Enable Battery Saver at 15% battery and remain on the battery-settings page.</td></tr><tr><td>PrefD 21</td><td>Settings</td><td>System settings</td><td>Mute media volume after Bluetooth headphones disconnect, without changing Bluetooth settings.</td></tr><tr><td>PrefD 22</td><td>Settings</td><td>System settings</td><td>Enable Do Not Disturb or silence the phone for a 09:00–11:30 focused-work session.</td></tr><tr><td>PrefD 23</td><td>Messages; Contacts</td><td>Communication</td><td>Save the sender of the new-number message as Bob, using phone number 5550199.</td></tr><tr><td>PrefD 24</td><td>Messages</td><td>Communication</td><td>Send Mom a birthday SMS containing "Happy Birthday" or its Chinese equivalent.</td></tr><tr><td>PrefD 25</td><td>Messages</td><td>Communication</td><td>Send Mom the specified message requesting CNY 2,000 for living expenses.</td></tr><tr><td>PrefD 26</td><td>Messages; Contacts</td><td>Communication</td><td>Save the urgent Project Office sender as a contact and mark the contact as a favorite.</td></tr><tr><td>PrefD 27</td><td>Messages</td><td>Communication</td><td>Block the specified phishing-message sender without replying or opening the message's link.</td></tr><tr><td>PrefD 28</td><td>Mattermost</td><td>Communication</td><td>Acknowledge the Cluster-A server-error alert in town-square and state that it is being investigated.</td></tr><tr><td>PrefD 29</td><td>Mattermost</td><td>Communication</td><td>Post the specified one-day sick-leave notice in Town Square.</td></tr><tr><td>PrefD 30</td><td>Mail; Files</td><td>Communication</td><td>Email the dean with subject "Weekly Report,"the specified body, and the Weekly_Report.pdf attachment.</td></tr><tr><td>PrefD 31</td><td>Files</td><td>Office</td><td>Sort the Download folder by modification time, newest first.</td></tr><tr><td>PrefD 32</td><td>Gallery; Mail</td><td>Office</td><td>Select the flower photos and email them to Kevin with the specified message.</td></tr><tr><td>PrefD 33</td><td>Gallery;</td><td>Social media</td><td>Publish the HuggingGPT architecture image from the</td></tr><tr><td>PrefD 34</td><td>Mastodon Mastodon;</td><td>Social media</td><td>gallery with a short caption using the logged-in account. Find an AI- or machine-learning-related post and share</td></tr><tr><td>PrefD 35</td><td>Mattermost Mastodon</td><td>Social media</td><td>its content and original link in Town Square. Publish the specified weekend-hiking update with</td></tr><tr><td>PrefD 36</td><td>Gallery</td><td>Office</td><td>public visibility. Delete screenshots older than seven days while</td></tr><tr><td>PrefD 37</td><td>Taodian</td><td>Commerce</td><td>preserving those from the most recent seven days. Remove electronic and electronics-related products</td></tr><tr><td>PrefD 38</td><td>Chrome; Mail</td><td>Information</td><td>from the cart while preserving other categories. Use the local search page to find recent AI news, then</td></tr><tr><td></td><td></td><td>retrieval</td><td>email Kevin an AI-titled message with the required opening and a summary.</td></tr><tr><td>PrefD 40</td><td>Mattermost</td><td>Communication</td><td>Post the specified end-of-workday farewell in town-square.</td></tr><tr><td>PrefD 41</td><td>Settings</td><td>System settings</td><td>Switch the phone to the system dark/night theme while browsing Mastodon late at night.</td></tr><tr><td>PrefD 42</td><td>Calendar; Files; Docreader</td><td>Office</td><td>Find the document specified in the Product Review calendar event and open it for pre-meeting reading.</td></tr><tr><td>PrefD 43</td><td>Taodian</td><td>Commerce</td><td>Purchase one bag of Mingqian Yirgacheffe medium-dark-roast whole coffee beans and submit the order with the specified delivery details.</td></tr><tr><td>PrefD 44</td><td>Chilemei</td><td>Commerce</td><td>Order jasmine green milk tea with 30% sugar, no ice, and oat milk; request a sealed cup and a separate straw,</td></tr><tr><td>PrefD 45</td><td>Maps</td><td>Navigation</td><td>then complete simulated payment. Plan a trip from Aokesi, Hangzhou, to Hangzhou East railway station with two large suitcases and about 65</td></tr><tr><td>PrefD 46</td><td>Messages; Calendar</td><td>Office</td><td>minutes before ticket inspection. Politely decline Lin Na's Saturday 19:00–21:00 dinner invitation because it conflicts with family time; do not add the dinner to the calendar.</td></tr><tr><td>PrefD 47</td><td>Messages; Contacts</td><td>Communication</td><td>Save the parent-meeting notice sender as Duoduo's class teacher and mark the contact as a favorite.</td></tr><tr><td>PrefD 48</td><td>Gallery; Mastodon</td><td>Social media</td><td>Publish the specified photo of the user's pet Juzi sunbathing by a window, with the prescribed caption.</td></tr><tr><td>PrefD 49</td><td>Files</td><td>Office</td><td>Move report.pdf, plan.docx, and budget.xlsx from Download to the work-documents folder; leave</td></tr><tr><td>PrefD 50</td><td>Clock</td><td>Office</td><td>photo.jpg, movie.mp4, and app.apk in place. Create an enabled 07:00 Saturday-and-Sunday alarm using Cesium, the system-default alarm sound.</td></tr><tr><td>PrefD 51</td><td>Chrome</td><td>Information retrieval</td><td>Look up today's Beijing weather and maximum temperature in the morning, and report them to the user.</td></tr><tr><td>PrefD 52</td><td>Contacts; Phone</td><td>Communication</td><td>Find Grandma in Contacts and initiate an after-dinner phone call.</td></tr><tr><td>PrefD 53</td><td>Settings</td><td>System settings</td><td>Open Settings and enable Battery Saver when the phone's battery is running low.</td></tr></table>

Task-family example. Table 5 presents the six instances of PREFD 02. The Basic instruction specifies the shoes and delivery details, whereas the Outcome variant obtains the user’s usual choice and recipient details from context. The remaining variants retain an explicit purchase goal while introducing a budget, store priority, procedure, or confirmation requirement. Each variant is a separate task instance.

Table 5: User inputs for the six instances in PREFD 02, shown as complete English translations with the application names normalized as in Section A.1. In the behavioral-log condition, the preference is replaced by the corresponding logs.
<table><tr><td>Variant</td><td>User Instruction</td><td>Explicit User Preference</td></tr><tr><td>Basic</td><td>Open Taodian, search for a pair of white everyday commuting sneakers, choose size 42, enter Zhang Wei as the recipient, 13800138000 as the phone number, and Aokesi, Dongcheng District, Beijing, as the address, then place the order.</td><td>None.</td></tr><tr><td>Outcome</td><td>Help me buy a pair of the sneakers I usually wear for commuting to work on Taodian; just place the order.</td><td>The user likes simple white sneakers for commuting to and from work, dislikes flashy colors and high-top styles, and wears size 42. The user's name is Zhang Wei, phone number is 13800138000, and default delivery address is Aokesi, Dongcheng District, Beijing.</td></tr><tr><td>Constraint</td><td>Help me search Taodian for a pair of white everyday commuting sneakers, choose size 42, enter Zhang Wei as the recipient, 13800138000 as the phone number, and Aokesi, Dongcheng District, Beijing, as the address, then place the order.</td><td>The user values cost-effectiveness when buying commuting shoes and does not like buying expensive ones; the price of a pair must not exceed CNY 300. If only options above CNY 300 are available, the user must first be told why; an order must not be placed directly.</td></tr><tr><td>Ranking</td><td>Help me search Taodian for a pair of white everyday commuting sneakers, choose size 42, enter Zhang Wei as the recipient, 13800138000 as the phone number, and Aokesi, Dongcheng District, Beijing, as the address, then place the order.</td><td>The user cares strongly about authenticity when buying shoes and prefers brand official flagship stores (official direct sales). If the same shoe is available from both an official flagship store and an ordinary store, the user usually prioritizes the official flagship store and considers an ordinary store only when the official flagship store is out of stock.</td></tr><tr><td>Process</td><td>Help me search for a pair of white everyday commuting sneakers, choose size 42, enter Zhang Wei as the recipient, 13800138000 as the phone number, and Aokesi, Dongcheng District, Beijing, as the address, then place the order.</td><td>The user is conscious about price when buying shoes and dislikes placing an impulsive order on a single platform. The user habitually compares the same item's prices on Taodian and JD before ordering from the cheaper platform.</td></tr><tr><td>Interaction</td><td>Help me search Taodian for a pair of white everyday commuting sneakers, choose size 42, enter Zhang Wei as the recipient, 13800138000 as the phone number, and Aokesi, Dongcheng District, Beijing, as the address, then place the order.</td><td>The user is fairly cautious and dislikes automated actions making decisions on their behalf. For any action that places an order, the user wants to be asked for confirmation beforehand and does not want an order placed directly without confirmation.</td></tr></table>

## A.2 USER-CONTEXT CONSTRUCTION AND ANNOTATION

Figure 5 summarizes the overall construction process. We detail below how explicit user profiles, behavioral memory logs, and their reference annotations are produced.

Profile drafting and review. For each personalized task, we specify the intended preference and its applicable conditions. An LLM drafts an explicit user profile, which GPT-5.6 Sol checks and revises before human reviewers examine every profile. Reviewers check the requested behavior, applicable conditions, and task-specific details against the intended preference. The reviewed profile then serves as the basis for log synthesis.

Behavioral-log synthesis. Using the reviewed profile, an LLM generates synthetic records of past choices, corrections, confirmations, and related activities. Supporting records provide evidence for the target preference, while distractors may share an application, product category, or recipient without establishing the same requirement. Contextual exceptions are retained where present. The logs undergo GPT-5.6 Sol revision and item-by-item human review alongside the profile. Reviewers check that both forms convey compatible requirements and that distractors remain plausible historical records rather than explicitly declaring their irrelevance. The instruction, initial environment state, and success checker remain fixed across the paired context forms.

Reference annotation and validation. We link each preference atom to its supporting log entries and annotate task applicability, trigger conditions, and the expected behavior for diagnostic construction. Reviewers check evidence support against the actual record content. For the confirmation example in Section A.2.1, a recorded ask–confirm–submit sequence supports the preference, whereas store selection or price comparison does not. When inconsistencies are found, the source records are revised and the affected context pairs and probes are rechecked. Automated checks validate identifiers, required fields, and evidence references.

![](images/b3b929c0ba1fc28a70666c2866f9b9d0cae9f5e5bc8353bd3634c56b09dd0805.jpg)  
Figure 5: Construction pipeline of PrefDiag-Bench. Top: Task construction with paired profiles and behavioral logs. Bottom: Failure selection, preference-atom extraction, and diagnostic-probe construction. Both stages combine model-assisted drafting with human revision.

## A.2.1 PAIRED-CONTEXT EXAMPLE: PREFD 02 / INTERACTION

Reference preference. Request confirmation before submitting the shoe order.

## Task instruction.

Help me search Taodian for a pair of white everyday commuting sneakers, choose size 42, enter Zhang Wei as the recipient, 13800138000 as the phone number, and Aokesi, Dongcheng District, Beijing, as the address, then place the order.

## Explicit user profile.

The user is fairly cautious and dislikes automated actions making decisions on their behalf. For any action that places an order, the user wants to be asked for confirmation beforehand and does not want an order placed directly without confirmation.

## Behavioral memory logs.

L1. During a previous automated shoe purchase, the agent asked, “Do you confirm ordering this white pair in size 42?” before clicking the order button. The order was submitted only after the user confirmed.

L2. The user once canceled an automatically placed clothing order and added that future orders should be preceded by a question, rather than submitted directly.

L3. The user bought three pairs of white ankle socks in Taodian for CNY 19.9, with delivery to Aokesi, Dongcheng District, Beijing.

L4. For a shoe-care purchase, the user selected a cleaning product from an ordinary store because the official store did not have the same item in stock.

L5. When buying a backpack, the user checked the same model’s prices in Taodian and Jingdian, then placed the order in Jingdian.

Reference evidence. L1 and L2 support confirmation before order submission: L1 records an ask– confirm–submit sequence, and L2 records a correction after automatic ordering. L3 concerns a

socks purchase, L4 store selection under a stock exception, and L5 cross-platform price comparison. These records do not directly support the confirmation requirement. The record identifiers L1–L5 are presentation labels; the reference evidence is not part of the ordinary behavioral-log input.

## A.3 DIAGNOSTIC PROBE EXAMPLES

The diagnostic set contains 1,664 items: 260 P1, 260 P2, 325 P3, 601 P4, and 218 P5. The examples below correspond to PREFD 02 / Interaction. Each probe is queried independently with the specified inputs, without passing predictions between probes. Shared inputs are printed once and cross-referenced, but supplied in full in each request. Reference answers and explanations are presented separately from model inputs.

## A.3.1 P1: LOG-TO-MEMORY INDUCTION

Example ID. P1–PrefD 02–Interaction.

Input. The five logs L1–L5 in Section A.2.1, supplied in full, together with the Question.

Question. Based only on the memory logs above, which statement best summarizes the user’s stable communication and confirmation preferences when this kind of task is automated?

A. Ask whether the user is satisfied after submitting a clothing or shoe order, and try to remedy it if the user disagrees.

B. Confirm the style only at the start of the search; the automation may decide whether to place the final order on its own.

C. Ask every time the product or size changes; filtering cannot proceed continuously.

D. Search and filtering may proceed continuously, but the user should have a final decision opportunity before a clothing or shoe order is placed.

## Response format. One option ID, A–D.

Reference answer: D. L1 and L2 establish confirmation before order submission, rather than confirmation after ordering, only at search initiation, or at every product/size change.

## A.3.2 P2: EVIDENCE SELECTION

Example ID. P2–PrefD 02–Interaction.

Input. The known preference is: “Search and filtering may proceed continuously, but the user should have a final confirmation opportunity before a shoe order is submitted.”

Question. The scope above only limits where the preference applies and does not itself need to be established from the logs. Which memory logs directly support the preference being verified? Select all that apply.

A. The user once canceled an automatically placed clothing order and added that future orders should be preceded by a question, rather than submitted directly.

B. Several previous personal shopping orders used Zhang Wei, 13800138000, and Aokesi, Dongcheng District, Beijing, as the default delivery details.

C. The user bought three pairs of white ankle socks in Taodian for CNY 19.9, with delivery to Aokesi, Dongcheng District, Beijing.

D. During a previous automated shoe purchase, the agent asked, “Do you confirm ordering this white pair in size 42?” before clicking the order button. The order was submitted only after the user confirmed.

E. For a shoe-care purchase, the user selected a cleaning product from an ordinary store because the official store did not have the same item in stock.

F. When buying a backpack, the user checked the same model’s prices in Taodian and Jingdian, then placed the order in Jingdian.

G. During a shoe purchase, the user first opened a colorful high-top style, then went back and selected white low-top commuting shoes, adding the note “for work; nothing too flashy.”

H. In the user’s most recent personal commuting-shoe purchase, the user chose simple white lowtop sneakers in size 42, with Zhang Wei, 13800138000, and Aokesi, Dongcheng District, Beijing, as the delivery details.

## Response format. A JSON array of selected option IDs in alphabetical order.

Reference answer: ["A", "D"]. A states the confirmation rule through a correction, and D records the corresponding behavior. The other options concern delivery, product choice, store choice, or price comparison. The selected set must match the reference exactly.

## A.3.3 P3: RELEVANCE SELECTION

## Example ID. P3–PrefD 02–Interaction.

Input. The complete task instruction from Section A.2.1 and the preference candidates below.

Question. Which known user preference should affect planning or execution at an appropriate point in the current task? Consider the whole task.

A. None of these preferences should affect the current task.

B. When ordering a personal weekday lunch for delivery, finish choosing food and entering the address first, then summarize the restaurant, dishes, and amount and obtain one confirmation before payment.

C. When purchasing a laptop for the user’s work, present the model, price, delivery location, and other key risk information together before ordering, and submit only after confirmation.

D. When purchasing everyday commuting shoes for the user, search and filtering may proceed continuously, but the user should have a final confirmation opportunity before a shoe order is submitted.

## Response format. One option ID, A–D.

Reference answer: D. D matches the current shoe-purchase task. B and C specify confirmation requirements for lunch and laptop purchases. The question concerns relevance to the task, not the current trigger state.

## A.3.4 P4: TRIGGER RECOGNITION

Example ID. P4–PrefD 02–Interaction–Active.

Input. The complete task instruction from Section A.2.1, the preference below, the prior action history, and the screenshot in Figure 6.

Given preference. Search and filtering may proceed continuously, but the user should have a final confirmation opportunity before a shoe order is submitted.

## Prior action history.

1. Searched for and selected “Classic White Sneakers (Official Direct).”

2. Selected size 42.

3. Entered Zhang Wei, 13800138000, and Aokesi, Dongcheng District, Beijing.

4. Returned to the order-submission page.

Question. Given the current screenshot and the preceding objective action history, which stage is this preference at in the current task state?

A. The relevant decision has already been completed, or the trigger window for this preference has passed.

B. The decision point at which this preference needs to be applied has not yet been reached.

C. The current state is at the trigger point for this preference.

![](images/7aca89cfaac734d9cd3468b4e368c50ed1d4efda91a556e7f4283a4e7697f1b2.jpg)

Response format. One option ID, A–C.

Reference answer: C (active now). In the reference item, order submission is pending after the product and delivery details have been specified, placing the interaction at the final-confirmation point. The screenshot precedes action 31 of the reference trajectory. Only preceding actions are supplied: future actions, model reasoning, and the reference label are excluded. The other classes are not active yet and no longer active.

Figure 6: GUI observation used in the P4 and P5 examples. The screen shows the pending shoe order and the red “Submit order” button. Interface text is translated for the reader and is not additional model input.

## A.3.5 P5: OPERATION MAPPING

Example ID. P5–PrefD 02–Interaction.

Input. The task instruction, preference, prior action history, and screenshot used in the P4 example are supplied again in a separate request. The additional condition states that the preference is active and the relevant decision remains pending; it is not obtained from the model’s P4 prediction.

Question. Which next-step strategy best follows this preference at the current moment?

A. Summarize the current shoe, size 42, and pending order; ask the user to confirm submission and wait for an explicit reply before deciding whether to click “Submit order.”

B. Click “Submit order” directly. The original instruction already requested an order, so no further confirmation is needed.

C. Click “Submit order” first, then tell the user the result on the payment page and ask whether it is acceptable.

D. Tell the user “I am about to place the order,” then click “Submit order” immediately without waiting for a reply.

Response format. One option ID, A–D.

Reference answer: A. A requests confirmation before submission and waits for the user’s reply. B omits confirmation, C asks after submission, and D replaces confirmation with a notice. The item tests a semantic strategy, not screen coordinates or an action-token sequence.

## B EXPERIMENTAL DETAILS

## B.1 ENVIRONMENT AND AGENT CONFIGURATION

Environment and task inputs. We reuse the Android environment and evaluation infrastructure of KnowU-Bench, with task-specific initial data and checkers for PrefDiag-Bench. Each episode starts from its prescribed application state, with agent history reset between tasks. Episodes allow at most 65 environment steps with a one-second inter-step wait, and can end earlier when the agent reports completion or inability to proceed. Open-source models are served on eight NVIDIA RTX 4090 GPUs. The 53 Basic tasks receive no additional user context. Each of the 265 personalized tasks receives either its explicit user profile or its behavioral memory logs. The instruction and context remain available throughout execution, alongside the current observation, adapter-specific history, and any platform reply. Adapters retain native observation and action formats while mapping actions and coordinates to the common environment interface.

Decoding and workflow interfaces. The Qwen, MAI-UI, Step-GUI, GUI-Owl, and UI-Venus adapters use temperature 0. Seed-2.0-Pro uses temperature 0.7, $\mathrm { t o p } { - } p = 0 . 9$ , and a 4,096-token response limit; its adapter retains the three most recent screenshots alongside textual history. Claude’s adapter does not explicitly set temperature. M3A and T3A use GPT-5.4 with temperature 0, a 2,048 token action-response limit, and up to 20 objective action summaries. Both workflows use the same task contexts, step budget, and checkers as the directly evaluated agents.

## B.2 SCORING AND TASK–PROBE ASSOCIATION

End-to-end success. Let $r _ { i } \in [ 0 , 1 ]$ be the native checker score for task instance i. Binary success and TSR are defined as

$$
s _ { i } = { \bf 1 } [ r _ { i } = 1 ] , \qquad \mathrm { T S R } = \frac { 1 0 0 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } s _ { i } ,
$$

where V contains tasks with valid checker results and TSR is reported as a percentage. Checkers inspect application state, database or callback records, and required action and interaction traces. Any accompanying natural-language reason is not used as a separate success classifier. The personalized AVG is the macro-average of the five preference-type TSRs.

Diagnostic accuracy. Each P1–P5 item is answered in an independent request, without passing model predictions between probes. Single-choice items require the reference option; P2 requires an exact match to the complete reference option set. For task–preference instance i and probe type k, let $\mathcal { Q } _ { i , k }$ contain all associated canonical questions, including those from multiple atoms or GUI states. The task-level result is

$$
a _ { i , k } = \prod _ { q \in \mathcal { Q } _ { i , k } } \mathbf { 1 } [ \hat { y } _ { q } = y _ { q } ] ,
$$

where $\hat { y } _ { q }$ and $y _ { q }$ are the predicted and reference answers. Thus, answering two of three P4 items correctly gives a task-level result of 0, not $2 / 3$ . Groups with any missing, truncated, or unparseable response are excluded, and their coverage is recorded.

Paired outcomes. We join diagnostic and end-to-end results by model, task-family ID, and preference type. Each eligible pair contributes once per probe type. TT, TF, FT, and FF denote $( s _ { i } , a _ { i , k } ) \bar { = } ( 1 , 1 ) , ( 1 , \bar { 0 } ) , ( 0 , \bar { 1 } ) , ( 0 , 0 )$ , respectively. Pooled proportions are calculated from valid pair counts, not by averaging preference-wise percentages.

## B.3 EVIDENCE INTERVENTION

Task selection. The fixed subset contains 85 task–preference pairs from 43 families, with 17 pairs per preference type: 15 P2-target pairs, one downstream control, and one retention control. The totals are therefore 75, 5, and 5 pairs, respectively. Selection uses pre-intervention diagnostic and behavioral-log execution records. A P2-target pair requires at least one model that fails both P2 and the task, and at least one model that completes the task. Downstream controls have correct P2 answers but failed execution; retention controls have successful diagnostic and execution outcomes. Every selected pair has an audited supporting-log mapping and at least one distractor. This P2-enriched subset is shared across all four models and all conditions, and selection does not use intervention outcomes.

Input conditions. C0 supplies the behavioral memory logs without evidence labels. C1 preserves the same entries, text, and order, adding the prefix [Relevant evidence] or [Irrelevant context] according to the reference evidence annotation. C2 retains only supporting entries in their original order, without labels. All three conditions are executed anew with the same instruction, initialization, and checker.

## B.4 MEMORY-MODULE CONFIGURATIONS

Mem0, LangMem, and Hindsight process the same frozen synthetic historical episodes for each of the 265 task–preference pairs. Each task is processed independently: we initialize a fresh memory store for the pair, ingest only its associated episodes in chronological order, and discard the store after evaluation. No memory state is shared across tasks, families, or preference variants. The reference profile, preference atom, evidence-role labels, and checker are not supplied to the memory modules. Table 6 lists their operations and configurations. The resulting memory text replaces Seed-2.0-Pro’s behavioral-log context without any reference-based correction of omissions, merging errors, or over-generalizations. Task initialization, interaction settings, the max step, and checkers remain unchanged.

Table 6: Memory construction and retrieval settings. All modules use GPT-5.4-mini for their LLM operations and an isolated store for each task–preference pair. The returned memory text is supplied to the downstream GUI agent.
<table><tr><td>Module</td><td>Memory operation and agent input</td><td>Configuration</td></tr><tr><td>Mem0 mem0ai 2.0.17</td><td>Sequential add (infer=True) extracts and consolidates memories. Scoped get_a11 returns the final records (limit 100); all returned text is used, without task-conditioned search.</td><td>Local Qdrant; bge-smal1-zh-v1.5 embeddings (512 dimensions). Temperature 0, top-p = 0.1, and 2,000 output tokens.</td></tr><tr><td>LangMem 0.0.30</td><td>create_memory_manager updates a memory list episode by episode, carrying the previous list forward.</td><td>Default memory instructions and Memory(content) schema; inserts and updates enabled, deletes disabled. Temperature 0, top-p = 0.1, and 2,000 output tokens.</td></tr><tr><td>Hindsight 0.9.0</td><td>retain ingests episodes into a fresh bank. recal1 uses the current GUI instruction as its query after ingestion. Returned text is supplied in native order.</td><td>bge-m3 embeddings and bge-reranker-v2-m3reranking. Recall budget mid, 4,096 tokens; world experience, and observation memories.</td></tr></table>

## C PREFERENCE APPLICATION IN GUI TASKS

The following examples illustrate how preferences affect GUI decisions. Basic uses the task instruction alone; Personalized adds behavioral memory logs. Instructions and preferences are summarized below. Each figure aligns corresponding decision stages and retains the original step numbers.

## C.1 CHOOSING AN OFFICIAL STORE (PREFD 03, RANKING)

Instruction. Buy an Apple MacBook Air M-series laptop in Taodian with at least 16 GB of memory and a price of at most CNY 10,000. Submit the order using the specified delivery details.

Preference. Prefer an official store, even at a higher price. Use an ordinary store only when the official option is out of stock.

Figure 7 shows two available M2 listings with 16 GB memory and 512 GB storage. Basic selects the ordinary-store listing at CNY 8,299; Personalized selects the official listing at CNY 8,999. Both runs pass their task-specific checkers. The preference prioritizes the official source among products that meet the configuration and price requirements.

![](images/739232ea37fe2837e0a73a189f91ed64fa040ccf88682972f6f593d25ab937db.jpg)  
Figure 7: Official-store preference. Both candidate lists include the ordinary and official options. Panels (b) and (d) show checkout immediately before order submission.

## C.2 STAYING WITHIN BUDGET (PREFD 02, CONSTRAINT)

Instruction. Buy white commuting sneakers in Taodian, select size 42, and submit the order using the specified delivery details.

Preference. Keep the shoe price at or below CNY 300.

In Figure 8, Basic selects a CNY 399 pair from the initial results. Personalized scrolls further and selects a CNY 259 pair. Both orders contain white, size-42 shoes and pass their respective checkers. The budget requirement restricts eligible products and leads to further search before selection.

![](images/63090053fc1a9ba0f3916afa19a125c859e78e6d2f09e9cfa46e7071e488dfa3.jpg)  
Figure 8: Budget preference. Further search yields a pair within CNY 300. Panels (b) and (d) show the selected white, size-42 shoes immediately before order submission.