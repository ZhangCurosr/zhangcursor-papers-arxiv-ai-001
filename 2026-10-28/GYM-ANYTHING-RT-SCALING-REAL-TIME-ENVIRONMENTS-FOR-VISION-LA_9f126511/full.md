# GYM-ANYTHING-RT: SCALING REAL-TIME ENVIRONMENTS FOR VISION-LANGUAGE AGENTS

Pranjal Aggarwal & Sean Welleck

Carnegie Mellon University

{pranjala, swelleck}@cs.cmu.edu

## ABSTRACT

AI agents have made substantial progress in domains where they have a non-trivial amount of time to deliberate between actions, such as mathematics, software engineering, and performing tasks on desktop applications. It is less clear how such models fare in settings that demand real-time interaction, such as playing a video game or operating a physical device. In these settings, the environment state may evolve as the agent is reasoning, or even as the underlying model is performing a forward pass. To gain insight into the real-time interaction capabilities of visionlanguage agents, we introduce GYM-ANYTHING-RT. Our key idea is to construct visual minigames that demand a portfolio of capabilities required for interactive real-time settings. GYM-ANYTHING-RT uses an agentic pipeline to automatically generate such environments based on a taxonomy of six capabilities required for real-time interaction, including the ability to perceive the environment and act under an evolving environment state. A key feature of GYM-ANYTHING-RT is that it allows for controlled comparison of real-time interaction capabilities: we can run the same minigame in real time, or pause or slow down the environment to relax the requirement for real-time actions. We use GYM-ANYTHING-RT to construct 100 real-time minigames and evaluate AI agents on them. We find large capability gaps in real-time interaction. For example, GPT-6 Astra (xHigh) achieves a pass rate of only 9.0% on the real-time minigames, which rises to 75.0% when the environment is paused. We also identify intriguing behaviors by the most powerful frontier agents: for instance, when given the ability to write code, these agents write code that bypasses the need to perceive and act in real time. We publicly release GYM-ANYTHING-RT and the generated minigames to spur further research on AI agents in real-time interactive settings.<sup>1</sup>

## 1 INTRODUCTION

AI agents have made substantial progress in domains where they have time to deliberate between actions, such as mathematics (DeepSeek-AI, 2025), software engineering (Jimenez et al., 2024), and performing tasks on desktop applications (Yuan et al., 2026; Aggarwal et al., 2026; Jang et al., 2026). However, these agents have a key limitation: they are slow, often taking more than ten seconds to choose a single action. This is acceptable when the environment waits for the agent, but not in settings that demand real-time interaction, such as playing a video game or operating a physical device (Zhang et al., 2026; Black et al., 2025). There, the environment keeps changing while the agent reasons, or even while the model performs a forward pass. By the time an action executes, it may no longer be valid, and the agent cannot react to what happened in the meantime.

Existing evaluations shed little light on this limitation. Computer-use benchmarks wait for the agent between actions, so they never test acting under an evolving state (Xie et al., 2024; Yuan et al., 2026; Aggarwal et al., 2026). Video game benchmarks do run in real time. However, frontier models make little progress on them (Zhang et al., 2026; Ying et al., 2026; Ouyang et al., 2026). Furthermore, each game combines many capabilities, such as 3D spatial understanding, temporal reasoning, and learning unfamiliar controls, in a single fixed task. This makes it difficult to tell whether models fail because they are too slow or because they lack some other capabilities. Separating these two causes requires controlled comparisons that vary how much time an agent has to respond on the same task

![](images/bfbdcaceef7931ed0f504930606063dc805b0ba9ddcf3fc10804a8ace9b5c2fb.jpg)  
Figure 1: Examples of real-time minigames generated with GYM-ANYTHING-RT. The 100 minigames in our benchmark cover different combinations of capabilities, with controlled variations in difficulty and interaction.

To address these challenges, we introduce GYM-ANYTHING-RT, a framework for creating environments for studying the real-time interaction capabilities of vision-language agents. Our key idea is to construct visual minigames that demand a portfolio of capabilities required for interactive real-time settings, while avoiding the complexity of complete games. For instance, in Rotating On-Screen Keyboard (Figure 1), the agent must enter a code by clicking keys that keep moving in three-dimensional space between its observation and its action. Furthermore, we build an evaluation stack that controls the clock of each minigame, so we can run the same task live at its natural speed, slow it down to give the agent more time to respond, or pause it between observation and action to remove the need for real-time actions. Since each minigame is a program, we can also modify it, for instance, to remove its temporal dynamics or to increase its difficulty. By comparing an agent’s performance on these variants of the same task, we can tell whether it fails because it cannot act in time as the environment changes, or because the task is hard for other reasons.

Building such minigames by hand is laborious, since each needs its own visuals, dynamics, controls, and verifier. We therefore build an agentic pipeline that generates the minigames automatically. We first survey prior work on embodied agents (Savva et al., 2019; Srivastava et al., 2021; Yang et al., 2025c), video games (Zhang et al., 2026), and computer use (Cheng et al., 2024; Xie et al., 2024) to identify six capabilities required for real-time interaction, such as 3D visual understanding, temporal reasoning, and acting under an evolving environment state, and select minigames that cover the capabilities. We then build on the creator-auditor procedure of Gym-Anything (Aggarwal et al., 2026), in which one coding agent builds an environment and an independent agent audits it. In our setting, a creator agent implements each minigame as a program, and additionally writes an oracle solution that uses the program’s internal state to show that the task is solvable. Independent audit agents then test the minigame and report problems to the creator agent, which fixes them until the minigame behaves as intended. Finally, review agents label the capabilities that each minigame requires. Each program also defines four easier difficulty levels and a simplified interaction mode, which let us evaluate weaker models and study what makes a task hard.

Using this pipeline, we construct a benchmark of 100 real-time minigames and their non-real-time counterparts, and use it to evaluate frontier and open-weight models. We find that current agents perform poorly in real time. GPT-6 Astra (xHigh) solves only 9.0% of the real-time tasks, while it solves 66.0% of their non-real-time counterparts, which we obtain by modifying each program so that it no longer requires real-time interaction. Other models perform even worse: Claude Opus 5.5 (xHigh) and GPT-6 Luna (xHigh) solve only 5.0% and 1.0% of the real-time tasks, and the open-weight Qwen3.8-27B and Qwen3.5-9B solve only 1.0% and 0.0%, respectively. In addition, we find that for frontier models, the primary bottleneck is speed. On the same real-time tasks, slowing the environment by 64× raises Astra’s pass rate from 9.0% to 57.0%, and pausing it raises the pass rate to 75.0%. Since we also control difficulty and the interaction interface, we further identify temporal reasoning and exploration of unfamiliar interfaces as additional sources of difficulty.

Interestingly, we also find thatfrontier agents can partially offset their poor real-time capabilities through their strong coding and reasoning capabilities. Specifically, when we allow Astra to run code, it writes scripts that perceive and act on its behalf between its own responses. For instance, in Rotating On-Screen Keyboard, it locates the moving keys using image template matching and clicks them within 200 ms. As a result, its real-time pass rate rises from 9.0% to 31.0%.

In summary, our contributions are three-fold. First, we introduce GYM-ANYTHING-RT, a framework that uses coding agents to automatically generate real-time visual minigames as programs, with oracle solutions and controlled difficulty and interaction. Its evaluation stack runs each minigame live, slowed down, or paused. Second, using GYM-ANYTHING-RT, we construct a benchmark of 100 real-time minigames and their non-real-time counterparts for studying the real-time interaction capabilities of vision-language agents. Third, through extensive evaluations, we show that current agents perform poorly in real time, that speed is the primary bottleneck for frontier models, and that frontier agents can partially offset their slow responses by writing code.

## 2 METHODOLOGY

Our goal is to develop a suite of minigames that test the real-time interaction capabilities of visionlanguage agents in a controllable way. In particular, we construct minigames in which the time available to respond can be controlled while the underlying task stays the same. Formally, each minigame is an executable environment $\mathcal { E }$ whose program defines its internal state, controls, and clock. At each step, the agent observes a screenshot $o _ { t }$ and issues mouse and keyboard actions $a _ { t }$ while the environment keeps evolving as the agent chooses $a _ { t }$ . We say that a task requires real-time interaction when the agent must use recent visual evidence to choose an action for the state in which it will execute, so that changes during the delay between observation and action can reduce that action’s value (Appendix H).

Each minigame has ten configurations $\mathcal { E } _ { d , i }$ , where $d \in \{ 1 , \ldots , 5 \}$ specifies the difficulty and $i \in$ {simplified, full} specifies the interaction mode. Each configuration can be run with a slowdown s: live execution at its natural speed $( s = 1 )$ , slowed execution $( s > 1 )$ , in which the clock advances at $1 / s$ of its normal rate, and paused execution, in which the environment freezes until the agent acts. In addition, each real-time minigame has a non-real-time counterpart $\bar { \mathcal { E } } ,$ , obtained through a minimal change to its timing (Section 2.3). Since a broken or unsolvable minigame would be counted as an agent failure, every configuration must also be solvable and objectively gradable. We construct these minigames with coding agents, that is, LLM agents that can write and run code and inspect screenshots in a sandbox. All agents are instances of GPT-6 Astra run through the Codex CLI. Each agent takes on one of four roles: survey agents collect candidate minigames, creator agents implement real-time versions of them, audit agents verify them, and review agents calibrate difficulty and annotate capabilities. Figure 2 summarizes this procedure.

## 2.1 TASK DISCOVERY AND SELECTION

We prompt survey agents to search prior work, public demonstrations, visual puzzles, online videos, and social media for challenges that satisfy four criteria: they can be formulated as self-contained minigames, require visual understanding and grounding, require actions through a visual interface, and have objectively verifiable outcomes. These include tasks that involve moving objects, hidden information, physical manipulation, or changes that occur between actions. For every source, the survey agent records its provenance, objective, visual structure, required interactions, temporal behavior, failure and recovery process, and information needed to verify success.

We manually review the first 15 minigames and use them as quality standards for subsequent agent iterations. We select candidates with a traceable source, distinct visual and interaction requirements, an objectively gradable outcome, and meaningful mouse or keyboard actions. A source need not involve real-time interaction itself, since the creator agent develops its mechanic into a real-time task (Section 2.2). We exclude tasks dominated by OCR, static image classification, arithmetic, or question answering, since these tasks do not require the forms of visual interaction that we seek to measure.

![](images/bc7ea234cb2e116d0010e1691950d38ad423b4737ee75dd147e5a16fa38a7232.jpg)  
Figure 2: Overview of our construction method. A creator agent implements selected minigames with controllable difficulty and interaction, constructs oracle solutions, and records evidence of their behavior. An independent auditor checks the implementation and evidence, returning issues for revision until the minigame passes the audit. Review agents calibrate difficulty and annotate capability requirements for controlled model evaluation.

## 2.2 CREATOR-AUDITOR CONSTRUCTION

To construct a real-time minigame E from a source P, we build upon the creator-auditor procedure from Gym-Anything (Aggarwal et al., 2026), in which a creator agent builds each environment and independent audit agents verify it. In our setting, the creator agent writes a program that implements the minigame and exposes a common interface for screenshot observations as well as mouse and keyboard actions. Crucially, we ask the creator agent to turn the task posed by the source into one that requires real-time interaction, as defined above, whether or not the source itself does. Adding decorative motion, a countdown, or faster animation is not sufficient. For example, Rotating On-Screen Keyboard requires the agent to click keys that keep moving between observation and action.

The creator agent implements the generator, interface, dynamics, action handling, and grader of E. It also constructs an oracle solver that uses the privileged state of the environment to determine a valid solution and executes it through the same action space available to the agent. A successful oracle execution therefore demonstrates that the task is solvable and that its action and grading paths agree. The creator agent records the implementation, oracle trajectory, and visual and temporal evidence for these claims (Appendix B.1).

Each implementation is then audited by a fresh audit agent that cannot modify the program. The audit agent inspects the code and evidence, runs the environment when necessary, and checks generation, solvability, rendering, action handling, grading, and recovery from failure. It also checks the real-time requirement itself, searching for shortcuts that would let an agent avoid acting in real time, such as a brake that halts the moving parts or periodic motion that permits a precomputed solution. The audit agent returns issues to the creator agent for revision, and the loop repeats until an audit passes.

## 2.3 NON-REAL-TIME COUNTERPARTS

A low pass rate on a real-time minigame does not by itself show that real-time interaction is the cause, since the underlying task may also be hard. We therefore construct, for each real-time minigame E, a non-real-time counterpart E<sup>¯</sup> that poses the same task without requiring real-time interaction. The creator agent makes a minimal change to the timing of E, while preserving its goal, controls, difficulty, and the information available to the agent. An independent audit agent then checks across all ten configurations that the two versions differ only in their timing. Comparing an agent on E and E<sup>¯</sup> thus lets us separate the effect of the real-time requirement from the difficulty of the underlying task.

## 2.4 CONTROLLED TASK VARIATIONS

To also evaluate weaker models and study what makes a task hard, we construct easier variants of each minigame. We vary the difficulty d and the interaction mode i while holding the other and the random seed fixed, for both E and its counterpart.

A difficulty-review agent first reads the generator, visible interface, grader, verifier, and oracle solver to determine what the original minigame requires. It compares this configuration with an approved set of reference minigames and assigns it to one of five levels. The creator agent preserves the original configuration at that level and constructs the remaining levels around it by changing the amount of information, number of dependencies, speed, precision, dynamics, or simultaneous control required for success. A difficulty-review agent then checks each adjacent pair and rejects any spurious difficulty increase, such as requesting repeated actions or adding ambiguity. The agent also ensures that the same types of perception, memory, decisions, and control are required across difficulty levels. For example, the levels of Rotating On-Screen Keyboard (Figure 3) vary the code length, the number and size of keys, and the keyboard motion.

The creator agent also implements simplified and full interaction for every difficulty level. Simplified interaction exposes proxy controls, while full interaction requires the direct mouse or keyboard manipulation appropriate to the minigame. For example, in Rotating On-Screen Keyboard, simplified interaction lets the agent type the code directly with keyboard actions, while full interaction requires clicking the rotating keys. For a fixed seed and difficulty, the audit agent verifies that both modes generate the same minigame and differ only in the controls used to act on it. Appendices B.2 and B.3 give examples of both variations.

## 2.5 CONTROLLING TIME

To control how much time an agent has to respond, we build an evaluation stack on top of Gym-Anything (Aggarwal et al., 2026) that controls the clock of each minigame. During live execution, the task continues to change from observation through action execution at its natural speed. In Rotating On-Screen Keyboard, for example, a key can move between observation and action, so the agent must account for its position when the click executes. Since any delay between an action and its effect counts against the agent, we also build a direct connection protocol. A central server still launches and manages many sandboxes in parallel, but each agent exchanges actions and screenshots over a direct connection to its own sandbox, bypassing the server. This keeps the median time from sending an action to receiving the resulting screenshot below 50 ms in our timing tests (Appendix E.1). For slowed execution, the stack scales every clock of the minigame, including its timers, animations, and server-side clocks, so that a slowdown of s advances the game by one second during s seconds of real time. Action delivery and screenshot capture still run in real time, so the slowdown changes the time available to respond while the agent’s interaction stays the same. During paused execution, the task advances through an observation window and then freezes on the final frame until the action is executed. All settings use the same minigame, seed, difficulty, interaction mode, and verifier. Appendix E describes the evaluation protocol.

## 2.6 CAPABILITY REVIEW

To relate agent failures to the capabilities a minigame requires, a capability-review agent labels each configuration of both the real-time minigames and their counterparts for 3D visual understanding, temporal reasoning, reasoning and planning, and exploration and interface understanding (see Appendix D for definitions). It reviews the game’s program, interface, verifier, oracle solution, and recorded behavior to identify what a solution using screenshots and mouse or keyboard actions requires. Each label is supported by code or behavioral evidence. A real-time review agent checks the real-time requirement separately for each configuration, using the implementation and recorded behavior (Appendix H).

## 3 GYM-ANYTHING-RT

We use the pipeline from Section 2 to construct GYM-ANYTHING-RT, comprising 100 real-time minigames and their 100 non-real-time counterparts. Each also has four easier difficulty levels and a

![](images/43c635a3ace19c17fa6d032e05f2c8b2fdd160fb660cce28a627a39edaec08a2.jpg)  
(a) The code and keyboard before the first action.

![](images/2402fd2da3e64085eb0f2f6cc30776225d4c36f06d41b1245ffc813875615006.jpg)  
(b) The first key is selected.

![](images/f8ebb8c4a6fd563b8808f7fadec9ec05158b153a5f8647655f436a35ec4d28cc.jpg)  
(c) The keyboard moves before the next click.  
Figure 3: Rotating On-Screen Keyboard. The model must enter a code by clicking keys that move between observation and action.

simplified interaction mode, giving 2,000 configurations in total. We evaluate models using the live, slowed, and paused settings defined in Section 2.5.

## 3.1 CAPABILITY COVERAGE

Table 1 summarizes GYM-ANYTHING-RT. Using the labels from the capability review (Section 2.6), 31% of the real-time minigames require 3D visual understanding at some difficulty level, all of them require temporal reasoning, 96% require reasoning and planning, and 52% require exploration and interface understanding. Their counterparts require the same capabilities at similar rates, except temporal reasoning, which drops to 82% once the realtime requirement is removed. Table 7 lists all minigames and their capability labels.

Table 1: Benchmark statistics.
<table><tr><td>Statistic</td><td>Count</td></tr><tr><td>Real-time minigames</td><td>100</td></tr><tr><td>Non-real-time counterparts</td><td>100</td></tr><tr><td>Difficulty levels</td><td>5</td></tr><tr><td>Interaction modes Configurations</td><td>2</td></tr><tr><td>Real-time</td><td>2,000</td></tr><tr><td>Non-real-time</td><td>1,000 1,000</td></tr></table>

## 3.2 EVALUATION PROTOCOL

We evaluate models by pass rate, time to solve, and inference cost, and compare these metrics across real-time minigames and their counterparts, slowdowns, difficulty levels, interaction modes, and capability requirements. Models receive task instructions and interact through screenshots, mouse actions, and keyboard input, without access to program internals or oracle solutions. Task-specific verifiers determine whether each attempt succeeds. Appendix F describes the models, prompts, budgets, and cost computation.

## 4 RESULTS AND ANALYSIS

## 4.1 EXPERIMENTAL SETUP

We evaluate Qwen3.5-9B, Qwen3.8-27B, GPT-6 Luna with xHigh reasoning effort, Claude Opus 5.5 with xHigh effort, and GPT-6 Astra with low and xHigh reasoning effort. We evaluate models in the live setting with full interaction at the highest difficulty, and Section 4.9 reports results on the easier difficulty levels. Following prior work on computer-use agents (Xie et al., 2024; Aggarwal & Welleck, 2025), models observe the environment only through screenshots and act only through mouse and keyboard actions. Models cannot run code or access the page source or program internals. In the coding experiment, we additionally allow models to run code in an isolated sandbox, where they can request screenshots and execute the same mouse and keyboard actions through an API. Interaction with the environment remains the same. Further experiments vary difficulty, interaction mode, the real-time setting, and whether interface rules are provided.

![](images/074d4830f7e4a7540cc8b2f400bb4d6268c251aa19649fac3108d46fa7122cfa.jpg)  
Figure 4: Pass rates (%) on GYM-ANYTHING-RT at difficulty 5 with full interaction in the live setting. Each model is evaluated on the 100 real-time tasks and their 100 non-real-time counterparts. Even the strongest model performs poorly on tasks requiring real-time interaction.

## 4.2 CURRENT AGENTS PERFORM POORLY IN REAL TIME

Figure 4 compares the pass rates of each model on the real-time tasks and on their non-real-time counterparts (Section 2.5). Even the strongest model, GPT-6 Astra (xHigh), solves only 9.0% of the real-time tasks at the highest difficulty, compared with 66.0% of their non-real-time counterparts. Interestingly, higher reasoning effort helps only when the task waits for the model: from Low to xHigh effort, Astra’s pass rate rises from 58.0% to 66.0% on the non-real-time counterparts, but only from 8.0% to 9.0% on the real-time tasks. Claude Opus 5.5 (xHigh) solves 5.0% of the real-time tasks and 57.0% of their counterparts. GPT-6 Luna (xHigh) performs worse on both, solving 1.0% of the real-time tasks and 17.0% of their counterparts, and the open-weight Qwen3.8-27B and Qwen3.5-9B solve only 1.0% and 0.0% of the real-time tasks. These two models also solve only 2.0% and 1.0% of the counterparts, so their failures are not specific to real time. Since each real-time task and its counterpart share the same minigame, these results show that acting in real time, rather than the underlying task, accounts for most failures of frontier models. In the following sections, we examine why.

## 4.3 SPEED IS THE PRIMARY BOTTLENECK FOR FRONTIER MODELS

To study how response speed limits performance, we use our evaluation stack to slow down the environment while leaving the model’s inference unchanged (Section 2.5). A slowdown of s advances the environment clock at 1/s of its normal rate, giving the model more time to respond before the task changes, while pausing freezes the environment until the model acts. Figure 5 reports the pass rate of GPT-6 Astra on the same 100 real-time tasks at the highest difficulty as we slow the environment. With xHigh reasoning effort, pass rate rises steadily from 9% at normal speed to 17% at 4×, 48% at 16×, and 57% at 64× slowdown, and reaches 75% when the environment is paused. Pausing similarly raises GPT-6 Luna’s pass rate on real-time tasks from 6.0% to 29.8% across all difficulty levels. Since the tasks are unchanged and only the time available to respond grows, these results show that for frontier models, slow responses are the primary reason for failure in real time.

![](images/7a6a59f79ad9de57fed21e6580648e31701253eb7e33276ef28dff361b6670ec.jpg)  
Figure 5: Effect of environment speed on Astra. Pass rates increase as the environment slows, with xHigh remaining below its paused performance at 64× slowdown.

Moreover, the benefit of additional time grows with reasoning effort. At normal speed, Low and xHigh effort perform similarly, 8% and 9%, since additional reasoning also takes additional time. As the environment slows, xHigh pulls ahead, reaching 57% at 64× compared with 42% for Low, and

Table 2: Effect of coding on real-time performance. Astra benefits most from coding on tasks requiring real-time interaction.
<table><tr><td></td><td colspan="2">Real-time</td><td colspan="2">Non-real-time</td></tr><tr><td>Model</td><td></td><td>Coding Non-coding</td><td></td><td>Coding Non-coding</td></tr><tr><td>GPT-6 Astra (Low)</td><td>14.0</td><td>8.0</td><td>63.0</td><td>58.0</td></tr><tr><td>GPT-6 Astra (xHigh)</td><td>31.0</td><td>9.0</td><td>74.0</td><td>66.0</td></tr><tr><td>GPT-6 Luna (xHigh)</td><td>5.0</td><td>1.0</td><td>18.0</td><td>17.0</td></tr></table>

Table 3: Efficiency on the real-time configurations at difficulty 5. All models are slower and less successful than the human author, who is evaluated on 20 randomly sampled tasks.
<table><tr><td>Agent</td><td>Pass rate ↑ (%)</td><td>Avg. time ↓ (min)</td><td>Avg. cost ↓ ($)</td></tr><tr><td>Qwen3.5-9B</td><td>0.0</td><td>75.9</td><td>0.62</td></tr><tr><td>Qwen3.8-27B</td><td>1.0</td><td>63.0</td><td>0.64</td></tr><tr><td>GPT-6 Luna</td><td>1.0</td><td>6.4</td><td>0.07</td></tr><tr><td>Claude Opus 5.5</td><td>5.0</td><td>15.8</td><td>6.40</td></tr><tr><td>GPT-6 Astra (Low)</td><td>8.0</td><td>4.3</td><td>2.76</td></tr><tr><td>GPT-6 Astra (xHigh)</td><td>9.0</td><td>7.2</td><td>4.50</td></tr><tr><td>Human</td><td>45.0</td><td>2.2</td><td></td></tr></table>

75% compared with 58% when paused. Interestingly, even at 64× slowdown, Astra (xHigh) remains 18 points below its paused performance, suggesting that some tasks remain difficult as long as the environment keeps changing at all.

## 4.4 STRONG CODING CAPABILITIES PARTIALLY OFFSET WEAK REAL-TIME CAPABILITIES

Since frontier models are strong coding agents, we next examine whether they can use code to compensate for their slow responses. We allow models to run code in an isolated sandbox, where they can request screenshots and execute the same mouse and keyboard actions through an API (Appendix F.5). The action space is unchanged, but models can now write scripts that observe and act without waiting for the model’s next response. Table 2 reports results at the highest difficulty with full interaction in the live setting. Coding raises the pass rate of Astra (xHigh) on real-time tasks from 9.0% to 31.0%, a 3.4-fold increase, compared with a gain of only 8.0 points on their non-real-time counterparts, from 66.0% to 74.0%. The gain also grows with model capability: Astra (Low) improves from 8.0% to 14.0%, and Luna only from 1.0% to 5.0%.

To understand this gain, we inspect Astra’s trajectories. Astra writes image-processing scripts that perceive and act between its own responses. For instance, in Rotating On-Screen Keyboard, it locates the moving keys using image template matching and clicks them within 200 ms. In Clockwork Clutch Safe, where moving gears must be aligned, it tracks gear markers and times clutch releases using estimates of their motion (Appendix G). In effect, the model moves the real-time part of the task into code, and reasons only about what the code should do. This suggests that similar scripts could help slow models act in other real-time domains, such as video games and physical interfaces. Nonetheless, coding still leaves Astra far below its paused performance of 75.0%, and such scripts are difficult to write when the relevant visual information cannot be extracted reliably or prepared for in advance.

## 4.5 MODELS ARE SLOW AND EXPENSIVE

Real-time applications require agents that are not only accurate, but also fast and affordable. Table 3 reports the pass rate, average time, and average inference cost per attempt on the 100 real-time tasks at the highest difficulty. Astra (xHigh) achieves the highest pass rate, 9.0%, but takes 7.2 minutes and costs \$4.50 per attempt on average. Lowering its reasoning effort to Low reduces the time to 4.3 minutes and the cost to \$2.76, with a similar pass rate of 8.0%, consistent with additional reasoning helping little when the environment does not wait. Claude Opus 5.5 (xHigh) costs the most, \$6.40 per attempt, and takes 15.8 minutes, while solving only 5.0% of the tasks. Luna is far cheaper at \$0.07 per attempt, but takes 6.4 minutes and solves only 1.0% of the tasks. Qwen3.8-27B and Qwen3.5-9B cost \$0.64 and \$0.62 per attempt, but take about ten times longer than Luna, 63.0 and 75.9 minutes, while solving only 1.0% and 0.0% of the tasks. As a case study, we compare these results with those of a human author who, like the evaluated models, sees each minigame for the first time. On 20 randomly sampled tasks, the human solves 9 tasks, or 45.0%, in an average of 2.2 minutes per task (Appendix F.7). This is five times the pass rate of Astra (xHigh), in less than a third of its average time. These results show that progress in real-time interaction requires improving not only the pass rate of models, but also their speed and cost (Abhyankar et al., 2026; Black et al., 2025).

![](images/672bd783a2c0f6657b40d42366e540743629dddd960016ec3c2e9a6d4cf99f13.jpg)  
Figure 6: Effect of pausing on Luna. Pausing improves performance when real-time interaction or temporal reasoning is required. p-values are Holm-adjusted.

![](images/5ac39d67af928c37a829be0b26eaab6c1835a8a71d79537fef9a1109ed54609f.jpg)  
Figure 7: Performance with increasing capability requirements. Luna’s pass rate generally decreases as more of the four task-level capabilities are required.

![](images/ea0c8c0b2c79202b48e9d2c5754bc172b32d217380d37a8d45864bd2580a1095.jpg)  
Figure 8: Effect of providing interface rules. Luna solves more tasks when given the rules, across real-time and non-realtime environments.

![](images/a399eea78c5926c54bf77465ecd4bb20a297acda84afdf97c8ac7bfafe6b8c13.jpg)  
Figure 9: Effect of difficulty and interaction. Under paused execution, simplified controls improve Luna’s performance at higher difficulties on real-time tasks.

## 4.6 PAUSING ALSO HELPS WHEN TEMPORAL REASONING IS REQUIRED

Besides response speed, our controls let us study which other capabilities limit agents. We first examine whether the benefit of pausing is limited to real-time tasks. Figure 6 compares live and paused execution for GPT-6 Luna with full interaction, averaged across all five difficulty levels. Interestingly, pausing also helps on the non-real-time counterparts that require temporal reasoning, raising Luna’s pass rate from 20.4% to 31.4%, while the gain on counterparts without temporal reasoning, from 45.2% to 52.7%, is not statistically significant $( p = 0 . 4 4 3 )$ . Since these counterparts do not require acting in real time, this suggests that models also struggle to reason about scenes that change over time, even when they need not act quickly, identifying temporal reasoning as a further source of difficulty. Moreover, even when paused, Luna solves only 29.8% of the real-time tasks, showing that for weaker models, response speed is only one of several bottlenecks.

## 4.7 PERFORMANCE DECREASES AS MORE CAPABILITIES ARE REQUIRED

Since review agents label the capabilities each configuration requires, we can also relate performance to the demands of a task. Figure 7 reports Luna’s pass rate as a function of the number of required capabilities, using full interaction across all five difficulty levels. In the live setting, pass rate decreases from 37.0% when one capability is required to 7.2% when all four are required, and paused execution shows the same trend, from 55.6% to 23.2%. Since the decline persists when the environment is paused, tasks become harder as capabilities are combined, even without time pressure. However, since capability requirements overlap across tasks, this analysis cannot isolate the effect of a single capability. We therefore examine interface exploration directly through a controlled experiment in Section 4.8.

Table 4: Pass rates (%) of GPT-6 Luna (xHigh) across difficulty levels, with full interaction in the live setting. Gap is the pass rate on the non-real-time counterparts minus the pass rate on the real-time tasks.
<table><tr><td rowspan="2"></td><td colspan="5">Difficulty</td><td rowspan="2">All</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>Real-time</td><td>15.0</td><td>6.0</td><td>4.0</td><td>4.0</td><td>1.0</td><td>6.0</td></tr><tr><td>Non-real-time</td><td>35.0</td><td>33.0</td><td>22.0</td><td>18.0</td><td>17.0</td><td>25.0</td></tr><tr><td>Gap</td><td>20.0</td><td>27.0</td><td>18.0</td><td>14.0</td><td>16.0</td><td>19.0</td></tr></table>

## 4.8 PROVIDING INTERFACE RULES IMPROVES PERFORMANCE

Many minigames require the agent to learn how an unfamiliar interface behaves before it can solve the task. To isolate this demand, we append an environment-specific rulebook describing the controls, action effects, and world rules to Luna’s task prompt (Appendix F.6), and evaluate with full interaction in the live setting on minigames requiring exploration and interface understanding. Figure 8 shows that providing these rules raises Luna’s pass rate from 11.3% to 16.2% across the five difficulty levels. The model must still solve the same minigame and carry out its solution through the same controls, so this gain comes from knowing the rules in advance. This identifies interface exploration as a further source of difficulty: Luna can use interface knowledge that it struggles to acquire through interaction.

## 4.9 PERFORMANCE DECREASES WITH DIFFICULTY AND IMPROVES WITH SIMPLIFIED INTERACTION

Finally, we turn to the easier difficulty levels and simplified interaction. Table 4 reports Luna’s pass rates in the live setting with full interaction across all five difficulty levels. While Luna solves only 1.0% of the real-time tasks at the highest difficulty, it solves 15.0% at difficulty 1, so the easier levels leave room to measure progress for weaker models. On their non-real-time counterparts, its pass rate falls from 35.0% to 17.0%. The gap between the two stays between 14.0 and 27.0 points at every difficulty, so acting in real time is a major source of failure even on the easiest tasks. To study difficulty without this effect, Figure 9 reports Luna’s pass rate on real-time tasks across the five difficulty levels under paused execution. Averaged across both interaction modes, pass rate decreases from 53.0% at difficulty 1 to 15.5% at difficulty 5, showing that the difficulty levels provide a controlled range of challenge. Simplified interaction raises the overall pass rate from 29.8% to 33.2%, with larger gains at higher difficulty. Since both modes realize the same minigame, this gain shows that some failures arise from operating the full controls rather than from solving the task. However, the decline with difficulty persists under simplified interaction, showing that the challenges introduced at higher difficulty levels are not only about the controls.

## 5 RELATED WORK

Real-time interaction. Acting in a changing environment requires accounting for the delay between observing it and acting on it. Robotics methods handle this delay through asynchronous execution or by predicting the state at execution time (Black et al., 2025; Shukor et al., 2025; Tang et al., 2025), and reinforcement learning studies decision making under action and observation delays (Ramstedt & Pal, 2019; Derman et al., 2021). For computer-use agents, OSWorld-Human measures latency and finds that agents take 2.7 to 4.3 times more steps than necessary (Abhyankar et al., 2026). Game benchmarks either pause the game during inference (Ouyang et al., 2026) or compare paused and real-time play (Zhang et al., 2026; Park et al., 2025), and Real-Time Reasoning Gym adds time pressure to text-based games (Wen et al., 2026). However, these comparisons cover only a few visual games or text-based environments. GYM-ANYTHING-RT pairs each of 100 real-time minigames with a non-real-time counterpart and controls the clock of each minigame, so that we can vary how much time an agent has to respond on the same task.

Benchmarks for interactive visual agents. Benchmarks for vision-language agents cover websites (Zhou et al., 2024; Koh et al., 2024), desktop and mobile applications (Xie et al., 2024; Bonatti et al., 2024; Yuan et al., 2026; Rawles et al., 2024), professional software (Aggarwal & Welleck, 2025;

Aggarwal et al., 2026), games (Zhang et al., 2026; Paglieri et al., 2025; Hu et al., 2025a; Zheng et al., 2025a; Ying et al., 2026; Ouyang et al., 2026; ARC Prize Foundation, 2026), and embodied simulators (Savva et al., 2019; Shridhar et al., 2020; Srivastava et al., 2021; Yang et al., 2025c). However, most of these environments wait for the agent between actions, and each task requires many capabilities at once, so low performance does not reveal which capability limits the agent. Prior works address the latter by mapping tasks to capabilities (Wu et al., 2023), evaluating perception, reasoning, and decision making separately (Wang et al., 2025c; Xu et al., 2026; Li et al., 2024), or varying one condition of a task, such as difficulty, the initial state, or the interface (Davydova et al., 2025; Zheng et al., 2025b; Zhao et al., 2025; Liu et al., 2026). GYM-ANYTHING-RT instead varies the real-time requirement, difficulty, and interaction interface of the same minigame.

Capability-specific evaluation of VLMs. Many benchmarks test individual capabilities of VLMs through questions about images or videos, such as 3D spatial understanding (Yang et al., 2025a; Wang et al., 2025b; Ma et al., 2024; Jia et al., 2025), temporal reasoning (Liu et al., 2024; Shangguan et al., 2024; Zhang et al., 2024), GUI grounding (Cheng et al., 2024; Li et al., 2025), and abstract reasoning (Chia et al., 2024; Ghosal et al., 2024; Song et al., 2025). However, the model answers a question about a fixed input, without acting on what it perceives. iVISPAR and VisGym evaluate visual reasoning through interaction (Mayer et al., 2025; Wang et al., 2026b). GYM-ANYTHING-RT further requires agents to perceive and act while the minigame keeps changing.

Automated environment construction. Recent works use coding agents and LLMs to construct computer-use environments (Aggarwal et al., 2026; Wang et al., 2026a; Cao et al., 2026), text-based environments (Zhang et al., 2025), and environments with tasks of varying difficulty (Hu et al., 2025b; Faldor et al., 2024), while procedural generators vary the complexity of task instances (Cobbe et al., 2019; Samvelyan et al., 2021; Chevalier-Boisvert et al., 2019; Stojanovski et al., 2025; Chen et al., 2025). However, none of these produce visual minigames whose timing, difficulty, and interaction interface can be varied while preserving the underlying task. We extend the creator and audit agents of Gym-Anything (Aggarwal et al., 2026) to construct such minigames, each with a non-real-time counterpart and an oracle solution. Appendix I provides a more detailed related work.

## 6 CONCLUSION

In this work, we introduced GYM-ANYTHING-RT, a scalable framework for creating environments for studying the real-time interaction capabilities of vision-language agents. In GYM-ANYTHING-RT, coding agents implement and audit each environment as a visual minigame program with an oracle solution, and our evaluation stack controls the clock of each minigame. As a result, we can vary how much time an agent has to respond, whether the task requires real-time interaction, its difficulty, and its interaction interface on the same minigame. Using GYM-ANYTHING-RT, we constructed a benchmark of 100 real-time minigames and showed that current agents perform poorly in real time, with the strongest agent, GPT-6 Astra (xHigh), solving only 9.0% of the real-time tasks compared with 66.0% of their non-real-time counterparts. We further showed that speed is the primary bottleneck for frontier models, and that temporal reasoning and exploration of unfamiliar interfaces are additional sources of difficulty. Interestingly, we also find that frontier agents can partially offset their poor real-time capabilities by writing code that perceives and acts on their behalf.

More broadly, our results suggest that progress in real-time interaction requires agents that are not only capable but also fast. We hope that GYM-ANYTHING-RT and the released minigames provide a foundation for future work on real-time agents, including faster models, agents that move the time-critical parts of a task into code, and training on real-time environments.

## ACKNOWLEDGEMENTS

We thank Daytona for their generous support in sandbox compute credits. We thank Jing Yu Koh, Lawrence Jang, Seungone Kim, Zeyu Zheng, and other members of the L3 Lab for feedback and helpful discussions. Pranjal Aggarwal is supported by a SoftBank Group-Arm Fellowship. This work is partially supported by the National Science Foundation under Grant No. DMS-2502281.

## REFERENCES

Reyna Abhyankar, Qi Qi, and Yiying Zhang. OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents, 2026. URL https://arxiv.org/abs/2506.16042.

Pranjal Aggarwal and Sean Welleck. Programming with Pixels: Can Computer-Use Agents do Software Engineering?, 2025. URL https://arxiv.org/abs/2502.18525.

Pranjal Aggarwal, Graham Neubig, and Sean Welleck. Gym-Anything: Turn any Software into an Agent Environment, 2026. URL https://arxiv.org/abs/2604.06126.

ARC Prize Foundation. ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence, 2026.

Ruairidh M. Battleday, Kai Sandbrink, Jimi Cullen-Drohan, Zihan Yan, Timothy Muller, Clare Maguire, Ales Kubicek, Fraser Greenlee-Scott, Sukrit Sumant, Tri Dao, Jurgen Schmidhuber,¨ Michal Valko, Joshua Tenenbaum, Thomas L. Griffiths, Zeb Kurth-Nelson, and James C. R. Whittington. DiG-bench: Discovery in Games, 2026. URL https://arxiv.org/abs/ 2608.12593.

Kevin Black, Manuel Y. Galliker, and Sergey Levine. Real-Time Execution of Action Chunking Flow Policies, 2025. URL https://arxiv.org/abs/2506.07339.

Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, Lawrence Jang, and Zack Hui. Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale, 2024.

Yuan Cao, Dezhi Ran, Mengzhou Wu, Yuzhe Guo, Xin Chen, Ang Li, Gang Cao, Gong Zhi, Hao Yu, Linyi Li, Wei Yang, and Tao Xie. GUI-GENESIS: Automated Synthesis of Efficient Environments with Verifiable Rewards for GUI Agent Post-Training, 2026. URL https://arxiv.org/ abs/2602.14093.

Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Qiying Yu, Xuefeng Li, Jiaze Chen, Hao Zhou, and Mingxuan Wang. Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles, 2025. URL https: //arxiv.org/abs/2505.19914.

Yuxi Chen, Haoyu Zhai, Chenkai Wang, Rui Yang, Lingming Zhang, Gang Wang, and Huan Zhang. CAPTCHA Solving for Native GUI Agents: Automated Reasoning-Action Data Generation and Self-Corrective Training. In International Conference on Machine Learning (ICML), 2026. arXiv:2603.23559.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9313–9332, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.505. URL https://aclanthology.org/2024.acl-long.505/.

Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. In International Conference on Learning Representations (ICLR), 2019.

Yew Ken Chia, Vernon Toh Yan Han, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns, 2024. URL https://arxiv.org/abs/2403.13315.

Karl Cobbe, Christopher Hesse, Jacob Hilton, and John Schulman. Leveraging Procedural Generation to Benchmark Reinforcement Learning, 2019. URL https://arxiv.org/abs/1912. 01588.

Mariya Davydova, Daniel Jeffries, Patrick Barker, Arturo Marquez Flores, and Sin ´ ead Ryan. OSUni-´ verse: Benchmark for Multimodal GUI-navigation AI Agents, 2025.

DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948, 2025.

Gelei Deng, Haoran Ou, Yi Liu, Jie Zhang, Tianwei Zhang, and Yang Liu. Oedipus: LLM-enchanced Reasoning CAPTCHA Solver, 2024. URL https://arxiv.org/abs/2405.07496.

Esther Derman, Gal Dalal, and Shie Mannor. Acting in Delayed Environments with Non-Stationary Markov Policies. In International Conference on Learning Representations (ICLR), 2021. arXiv:2101.11992.

Maxence Faldor, Jenny Zhang, Antoine Cully, and Jeff Clune. OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code, 2024. URL https://arxiv.org/abs/2405.15568.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal Large Language Models Can See but Not Perceive, 2024. URL https://arxiv.org/abs/2404.12390.

Deepanway Ghosal, Vernon Toh Yan Han, Chia Yew Ken, and Soujanya Poria. Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning, 2024. URL https://arxiv.org/abs/2403.03864.

Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P. Xing, Ion Stoica, Tajana Rosing, Haojian Jin, and Hao Zhang. lmgame-Bench: How Good are LLMs at Playing Games?, 2025a.

Mengkang Hu, Pu Zhao, Can Xu, Qingfeng Sun, Jianguang Lou, Qingwei Lin, Ping Luo, and Saravan Rajmohan. AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2025b.

Lawrence Keunho Jang, Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Odysseys: Benchmarking Web Agents on Realistic Long Horizon Tasks, 2026. URL https://arxiv.org/ abs/2604.24964.

Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models, 2025. URL https://arxiv.org/abs/2506.03135.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? In The Twelfth International Conference on Learning Representations, 2024.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks. In Proceedings ofthe Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2024. arXiv:2401.13649.

Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use, 2025. URL https://arxiv.org/abs/2504.07981.

Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Li Erran Li, Ruohan Zhang, Weiyu Liu, Percy Liang, Li Fei-Fei, Jiayuan Mao, and Jiajun Wu. Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2024. arXiv:2410.07166.

Enqi Liu, Liyuan Pan, Zhi Gao, Yan Yang, Chenrui Shi, Yang Liu, Jingrong Wu, and Qing Li. Benchmarking and Improving GUI Agents in High-Dynamic Environments, 2026.

Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. TempCompass: Do Video LLMs Really Understand Videos? In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 8731–8772, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.517. URL https://aclanthology.org/ 2024.findings-acl.517/.

Yaxin Luo, Zhaoyi Li, Jiacheng Liu, Jiacheng Cui, Xiaohan Zhao, and Zhiqiang Shen. Open CaptchaWorld: A Comprehensive Web-based Platform for Testing and Benchmarking Multimodal LLM Agents, 2025. URL https://arxiv.org/abs/2505.24878.

Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso M de Melo, and Alan Yuille. 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark, 2024. URL https://arxiv.org/abs/2412.07825.

Julius Mayer, Mohamad Ballout, Serwan Jassim, Farbod Nosrat Nezami, and Elia Bruni. iVISPAR – An Interactive Visual-Spatial Reasoning Benchmark for VLMs, 2025.

Mingyu Ouyang, Siyuan Hu, Kevin Qinghong Lin, Hwee Tou Ng, and Mike Zheng Shou. GameWorld: Towards Standardized and Verifiable Evaluation of Multimodal Game Agents, 2026. URL https: //arxiv.org/abs/2604.07429.

Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kucinski, Lerrel Pinto, Rob Fergus, Jakob Nicolaus Foerster,´ Jack Parker-Holder, and Tim Rocktaschel. BALROG: Benchmarking Agentic LLM and VLM¨ Reasoning On Games. In International Conference on Learning Representations (ICLR), 2025. arXiv:2411.13543.

Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training Software Engineering Agents and Verifiers with SWE-Gym. In Proceedings of the International Conference on Machine Learning (ICML), 2025.

Dongmin Park, Minkyu Kim, Beongjun Choi, Junhyuck Kim, Keon Lee, Jonghyun Lee, Inkyu Park, Byeong-Uk Lee, Jaeyoung Hwang, Jaewoo Ahn, et al. Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games, 2025.

Simon Ramstedt and Christopher Pal. Real-Time Reinforcement Learning. In Advances in Neural Information Processing Systems (NeurIPS), 2019. arXiv:1911.04448.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents, 2024.

Yufan Ren, Konstantinos Tertikas, Shalini Maiti, Junlin Han, Tong Zhang, Sabine Susstrunk, and ¨ Filippos Kokkinos. VGRP-Bench: Visual Grid Reasoning Puzzle Benchmark for Large Vision-Language Models, 2025. URL https://arxiv.org/abs/2503.23064.

Mikayel Samvelyan, Robert Kirk, Vitaly Kurin, Jack Parker-Holder, Minqi Jiang, Eric Hambro, Fabio Petroni, Heinrich Kuttler, Edward Grefenstette, and Tim Rockt¨ aschel. MiniHack the Planet: A¨ Sandbox for Open-Ended Reinforcement Learning Research. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A Platform for Embodied AI Research, 2019. URL https://arxiv.org/abs/1904.01201.

Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models, 2024. URL https://arxiv.org/abs/2410.23266.

Jiajun Shi, Jian Yang, Jiaheng Liu, Xingyuan Bu, Jiangjie Chen, Junting Zhou, Kaijing Ma, Zhoufutu Wen, Bingli Wang, Yancheng He, Liang Song, Hualei Zhu, Shilong Li, Xingjian Wang, Wei Zhang, Ruibin Yuan, Yifan Yao, Wenjun Yang, Yunli Wang, Siyuan Fang, Siyu Yuan, Qianyu He, Xiangru Tang, Yingshui Tan, Wangchunshu Zhou, Zhaoxiang Zhang, Zhoujun Li, Wenhao Huang, and Ge Zhang. KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation, 2025. URL https://arxiv.org/abs/2505.14552.

Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics, 2025. URL https://arxiv.org/abs/2506.01844.

Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. VisualPuzzles: Decoupling Multimodal Reasoning Evaluation from Domain Knowledge, 2025. URL https: //arxiv.org/abs/2504.10342.

Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Mart´ın-Mart´ın, Fei Xia, Kent Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, C. Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, and Li Fei-Fei. BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments, 2021. URL https://arxiv.org/abs/2108. 03332.

Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, and Andreas Kopf. REASONING GYM: Reasoning Environments for Reinforcement Learning¨ with Verifiable Rewards. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Jiaming Tang, Yufei Sun, Yilong Zhao, Shang Yang, Yujun Lin, Zhuoyang Zhang, James Hou, Yao Lu, Zhijian Liu, and Song Han. VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference, 2025. URL https://arxiv.org/abs/2512.01031.

Xiwen Teoh, Yun Lin, Siqi Li, Ruofan Liu, Avi Sollomoni, Yaniv Harel, and Jin Song Dong. Are CAPTCHAs Still Bot-hard? Generalized Visual CAPTCHA Solving with Agentic Vision Language Model. In 34th USENIX Security Symposium (USENIX Security 25), 2025.

Bowen Wang, Dunjie Lu, Junli Wang, Tianyi Bai, Shixuan Liu, Zhipeng Zhang, Haiquan Wang, Hao Hu, Tianbao Xie, Shuai Bai, Dayiheng Liu, Que Shen, Junyang Lin, and Tao Yu. CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents, 2026a. URL https://arxiv.org/abs/2605.25624.

Junyu Wang, Changjia Zhu, Yuanbo Zhou, Lingyao Li, Xu He, Mingkui Wei, and Junjie Xiong. COGNITION: From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers, 2025a. URL https://arxiv.org/abs/2512.02318. USENIX Security 2026.

Qineng Wang, Baiqiao Yin, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Jiajun Wu, Li Fei-Fei, and Manling Li. MindCube: Spatial Mental Modeling from Limited Views, 2025b. URL https://arxiv.org/abs/2506.21458.

Xinyu Wang, Bohan Zhuang, and Qi Wu. Are Large Vision Language Models Good Game Players? In International Conference on Learning Representations (ICLR), 2025c. arXiv:2503.02358.

Zirui Wang, Junyi Zhang, Jiaxin Ge, Long Lian, Letian Fu, Lisa Dunlap, Ken Goldberg, XuDong Wang, Ion Stoica, David M. Chan, Sewon Min, and Joseph E. Gonzalez. VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents, 2026b. URL https://arxiv. org/abs/2601.16973.

Yule Wen, Yixin Ye, Yanzhe Zhang, Diyi Yang, and Hao Zhu. Real-Time Reasoning Agents in Evolving Environments. In International Conference on Learning Representations (ICLR), 2026. arXiv:2511.04898.

Yue Wu, Xuan Tang, Tom M. Mitchell, and Yuanzhi Li. SmartPlay: A Benchmark for LLMs as Intelligent Agents, 2023.

Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, and Yiren Song. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks, 2025. URL https://arxiv.org/abs/2506.05982.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments, 2024. URL https://arxiv. org/abs/2404.07972.

Zelai Xu, Zhexuan Xu, Xiangmin Yi, Huining Yuan, Mo Guang, Kaiwen Long, Xinlei Chen, Yi Wu, Chao Yu, and Yu Wang. VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. arXiv:2506.02387.

Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10632–10643, June 2025a.

John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. SWE-smith: Scaling Data for Software Engineering Agents, 2025b. URL https://arxiv.org/abs/2504.21798.

Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents, 2025c. URL https://arxiv.org/abs/2502.09560.

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, and Christopher Clark. Holodeck: Language Guided Generation of 3D Embodied AI Environments. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Lance Ying, Ryan Truong, Prafull Sharma, Kaiya Ivy Zhao, Nathan Cloos, Kelsey R. Allen, Thomas L. Griffiths, Katherine M. Collins, Jose Hern´ andez-Orallo, Phillip Isola, Samuel J. Gershman, and´ Joshua B. Tenenbaum. AI Gamestore: Scalable, Open-Ended Evaluation of Machine General Intelligence with Human Games, 2026. URL https://arxiv.org/abs/2602.17594.

Mengqi Yuan, Zilong Zhou, Xinzhuang Xiong, et al. OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks, 2026.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9556–9567, June 2024.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15134–15186, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.736. URL https://aclanthology. org/2025.acl-long.736/.

Alex L. Zhang, Thomas L. Griffiths, Karthik R. Narasimhan, and Ofir Press. VideoGameBench: Can Vision-Language Models complete popular video games?, 2026. URL https://arxiv.org/ abs/2505.18134.

Jianrui Zhang, Mu Cai, and Yong Jae Lee. Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos, 2024. URL https://arxiv.org/abs/2410.02763.

Jiayi Zhang, Yiran Peng, Fanqi Kong, Cheng Yang, Yifan Wu, Zhaoyang Yu, Jinyu Xiang, Jianhao Ruan, Jinlin Wang, Maojia Song, HongZhang Liu, Xiangru Tang, Bang Liu, Chenglin Wu, and Yuyu Luo. AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning, 2025. URL https://arxiv.org/abs/2511.19304.

Henry Hengyuan Zhao, Kaiming Yang, Wendi Yu, Difei Gao, and Mike Zheng Shou. WorldGUI: An Interactive Benchmark for Desktop GUI Automation from Any Starting Point, 2025.

Xiangxi Zheng, Linjie Li, Zhengyuan Yang, Ping Yu, Alex Jinpeng Wang, Rui Yan, Yuan Yao, and Lijuan Wang. V-MAGE: A Game Evaluation Framework for Assessing Vision-Centric Capabilities in Multimodal Large Language Models, 2025a.

Xinyue Zheng, Haowei Lin, Kaichen He, Zihao Wang, Zilong Zheng, and Yitao Liang. MCU: An Evaluation Framework for Open-Ended Game Agents. In International Conference on Machine Learning (ICML), 2025b. arXiv:2310.08367.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations (ICLR), 2024. arXiv:2307.13854.

## APPENDIX CONTENTS

A Limitations 18   
B Minigame Construction 19   
B.1 Creator and Audit Agents . 19   
B.2 Difficulty Levels 19   
B.3 Interaction Modes 20   
B.4 Real-Time Revisions 20   
B.5 Non-Real-Time Counterparts 20   
C Minigame List 21   
D Capability Annotation Criteria 21   
E Real-Time Evaluation Protocol 21   
E.1 Action and Screenshot Latency 21   
F Evaluation Details 23   
F.1 Models and Harnesses 23   
F.2 Action Space and Prompt . 23   
F.3 Budgets and Outcomes 23   
F.4 Time and Cost . 23   
F.5 Coding Protocol . 25   
F.6 Interface Rulebooks . 25   
F.7 Human Baseline . 25   
G Examples of Model-Written Controllers 25   
H Real-Time Definition and Review Prompts 26   
H.1 Definition 26   
H.2 Review Prompts . 27   
I Extended Related Work 32

## A LIMITATIONS

We focus on evaluating existing agents, although our programmatic task generators and automatic verifiers also provide a basis for training. Studying whether training on these minigames improves real-time interaction is an important direction for future work.

Our minigames capture specific demands of real-time environments to enable controlled comparisons across capabilities, difficulty, and interaction. Whether improvements on these tasks transfer to complete games and physical interfaces remains an open question.

Finally, GYM-ANYTHING-RT covers a selected set of capabilities and task mechanics. Our creation– audit framework provides a way to extend this coverage with additional minigames and capability combinations.

## B MINIGAME CONSTRUCTION

This appendix details the construction procedure summarized in Section 2. All agents in the construction pipeline use GPT-6 Astra.

## B.1 CREATOR AND AUDIT AGENTS

The creator agent receives one selected minigame together with its survey records and source links. It also inspects previously approved minigames as working examples of the environment structure. The creator implements the environment specification, task, generator, browser interface, server grader, exported verifier, oracle solver, difficulty and interaction controls, real-time settings, and tests. It may create or edit only the files of its own minigame and cannot modify the shared runtime, evaluator, or clock. The creator runs all ten difficulty and interaction configurations in the browser in the live and paused settings and checks generation, grading, result export, verification, failure, and retry. It records screenshots, recordings, and command outputs as evidence, together with a solution video in which the oracle solver completes the task in the browser.

A fresh audit agent then inspects the environment and its evidence without modifying any files. It checks the following items and ends its report with a verdict of PASS or REVISION REQUIRED.

(a) Difficulty baseline. The generated task is assigned to the appropriate level by comparison with approved reference minigames, without relying on the environment name or the creator’s description.

(b) Difficulty variations. Levels 1 to 5 increase in what the agent must perceive, decide, remember, or control. Adding repetitive steps is not sufficient.

(c) Interaction. For the same seed and difficulty, simplified and full interaction use the same minigame, information, goal, timing, physics, tolerances, and success condition. Full interaction requires the intended mouse or keyboard operations on the minigame.

(d) Real time. In live execution, the environment continues while the model produces an action. In paused execution, it remains frozen until the action arrives, and the action then executes with the intended behavior. The real-time configurations satisfy the definition in Appendix H, and no mechanism lets the agent avoid acting in real time, such as a brake that halts moving parts or repeating motion that permits a precomputed solution.

(e) End-to-end behavior. All configurations load, remain playable, produce the correct grading result, export the result, handle failure, and allow another attempt.

(f) Evidence. The audit agent ignores comments and written claims that the running environment does not support. Screenshots, recordings, frame sequences, exported results, and browser behavior serve as evidence, and missing visual evidence is a severe issue.

The audit agent also checks that the creator changed only files within its permitted scope. The creator revises the implementation in response to the report, and a new audit agent repeats the audit.

## B.2 DIFFICULTY LEVELS

The five difficulty levels, from very easy (level 1) to very hard (level 5), are defined in the same way for every minigame in GYM-ANYTHING-RT. A level describes the exact configuration the agent receives, and the original configuration of a minigame can belong to any level. A difficulty-review agent assigns levels by comparison with approved reference minigames, including Isometric Voxel Extraction Mine (level 1), Cursor-Controlled Constellation Hunt (level 2), Gyroscopic Tilt Board (level 3), Blind Dice Courier (level 4), and Exact-Change Candy Cascade (level 5).

To construct the other levels, the creator changes parameters that alter the problem the agent must solve. Depending on the minigame, these include the information available to the agent, visual ambiguity, the number of variables handled together, dependencies between decisions, planning depth, motion speed and timing windows, action precision, and recovery options. More steps, rounds, or waiting do not by themselves make a configuration harder. For every pair of adjacent levels, the creator must state what changes in the agent’s decision or control problem. Table 5 shows the five levels of Rotating On-Screen Keyboard.

Table 5: Difficulty levels of Rotating On-Screen Keyboard. The keyboard rotates in three dimensions with irregularly timed motion that never repeats within an attempt.
<table><tr><td>Level</td><td>Code length</td><td>Keys</td><td>Keyboard motion</td></tr><tr><td>1</td><td>3</td><td>12 large keys</td><td>Gentle, begins after the second character</td></tr><tr><td>2</td><td>4</td><td>20 keys</td><td>Begins after the first character</td></tr><tr><td>3</td><td>5</td><td>26 keys</td><td>Faster, begins after the first character</td></tr><tr><td>4</td><td>5</td><td>32 keys</td><td>Faster, changes direction more often</td></tr><tr><td>5</td><td>8</td><td>32 smaller keys</td><td>Fast, sharply irregular, starts immediately</td></tr></table>

Table 6: Examples of simplified and full interaction.
<table><tr><td>Minigame</td><td>Simplified</td><td>Full</td></tr><tr><td>Rotating On-Screen Key- board</td><td>Type the code on the physical key- board.</td><td>Click each on-screen key while it moves.</td></tr><tr><td>Clockwork Clutch Safe</td><td>Click a labeled release or engage but- ton beneath each seal.</td><td>Drag each seal&#x27;s clutch lever between its coupled and free positions.</td></tr><tr><td>Parallel Grillmaster</td><td>Select a food, then click buttons to start cooking and move it to the tray.</td><td>Drag each food from the order to the grill and then to the serving tray.</td></tr><tr><td>LIDAR Blacksite</td><td>Hold labeled movement and turning buttons, and use buttons to scan, pick up, and verify.</td><td>Hold movement keys, drag the view to turn, click the scene to scan or pick up the beacon, and walk through the exit gate.</td></tr></table>

## B.3 INTERACTION MODES

Section 2.4 defines the two interaction modes. For a fixed seed and difficulty, both modes generate the same world, information, goal, and action effects. The grader rejects actions submitted through the controls of the other mode. Table 6 shows examples.

## B.4 REAL-TIME REVISIONS

We revised minigames whose earlier versions did not require real-time interaction or allowed an agent to avoid it. For 64 minigames, a creator agent developed a new real-time minigame from the same source, as described in Section 2.2. In 35 of these, the conversion removed the need for exploration, and a further revision restored it while keeping the goal, mechanic, and real-time execution. We also revised 23 minigames that already had real-time configurations, for instance to remove a brake that halts moving parts or to replace repeating motion that permits a precomputed solution. The remaining 13 minigames keep their original implementation, and GYM-ANYTHING-RT uses the revised versions of all others.

## B.5 NON-REAL-TIME COUNTERPARTS

Section 2.3 describes how we construct a non-real-time counterpart for each real-time minigame. The counterpart keeps the generator, goal, controls, difficulty parameters, available information, and grader of its real-time minigame, and changes only how the environment evolves over time. For example, in Rotating On-Screen Keyboard, the keyboard keeps its initial layout while the agent decides. In Clockwork Clutch Safe, each drive segment holds until the agent operates a clutch, and in Parallel Grillmaster, each food settles at its golden point once cooking starts. An independent audit agent compares the two versions across all ten configurations and returns any difference other than timing to the creator for repair. For the 35 minigames revised to restore exploration, we rebuilt the counterparts from the revised versions.

## C MINIGAME LIST

Table 7 lists the 100 real-time minigames in GYM-ANYTHING-RT and the capabilities that each minigame and its non-real-time counterpart require.

## D CAPABILITY ANNOTATION CRITERIA

GYM-ANYTHING-RT studies six capabilities. Review agents annotate the four capabilities below. The remaining two, interaction and real-time interaction, are varied through the interaction mode and the real-time setting. Appendix H defines when a configuration requires real-time interaction.

The authors define the following criteria for capability annotation. Review agents apply them to the requirements of a solution using screenshots and visible mouse or keyboard controls.

Visual understanding. We classify visual understanding as 2D or 3D according to the spatial understanding required to solve the task. Decorative perspective alone is insufficient for a 3D label.

Temporal reasoning. A configuration requires this capability when every general screenshot-only solution must interpret change, motion, order, or duration, remember relevant earlier state that is no longer visible, or control when and for how long an action remains active. The temporal relationship must affect action choice or execution. A simple visible sequence, idle animation, or automatic transition to a static result does not establish this requirement.

Reasoning and planning. Solving requires inferring constraints or choosing actions whose consequences affect later decisions. The annotation identifies the relevant inference or action dependency. Basic visual recognition alone does not establish this requirement.

Exploration and interface understanding. The agent must interact to reveal relevant information or learn how the interface behaves before solving the task. Routine feedback, an automatic transition that supplies the next required information, or following explicit controls does not establish this requirement.

Review agents inspect the implementation and recorded behavior for each difficulty and interaction mode. Each annotation is supported by code or behavioral evidence.

## E REAL-TIME EVALUATION PROTOCOL

The shared runtime controls the execution schedule, and the task clock begins when the environment starts. In live execution, an observation request returns a current frame while the task clock continues. Paused execution advances the task through an observation window and returns a chronological sequence of frames. The clock then stops on the final frame until the action executes and resumes for the next observation window.

The observation window and number of frames are set per minigame and are the same for its non-real-time counterpart. The window ranges from 80 to 800 ms with three to six frames.

## E.1 ACTION AND SCREENSHOT LATENCY

Direct connection protocol. Our action and screenshot path builds on a fast input and output path that we previously developed for a single agent controlling a single sandbox. That design assumes a one-to-one pairing between agent and sandbox. Gym-Anything (Aggarwal et al., 2026), in contrast, runs many sandboxes behind a central server that launches them and forwards requests to them, and forwarding every action and screenshot through this server adds latency. We therefore separate the two roles. The central server still allocates sandboxes and manages their lifecycle, but actions and screenshots bypass it. The agent’s client runs on the same machine as its sandbox and keeps a persistent direct HTTP connection to an input service inside the sandbox, which executes each action and returns its acknowledgement over this connection. Screenshots are captured natively inside the

Table 7: Minigames in GYM-ANYTHING-RT. Capability labels: 3D visual understanding (3D), temporal reasoning (T), reasoning and planning (R), and exploration and interface understanding (E). A capability is marked when it is required at some difficulty level. •: required by the real-time minigame and its non-real-time counterpart. ◦: required only by the real-time minigame. ⋄: required only by the counterpart.
<table><tr><td>Minigame</td><td>3D T</td><td>R</td><td>E Minigame</td><td></td><td>3D</td><td>T R</td><td>E</td></tr><tr><td>After Hours at the Reliquary</td><td></td><td></td><td></td><td>Impossible Ecology</td><td></td><td></td><td></td></tr><tr><td>Anthill Front</td><td></td><td></td><td></td><td>Impossible Panorama</td><td></td><td></td><td></td></tr><tr><td>Apothecary Dead Reckoning</td><td></td><td></td><td></td><td>Isometric Voxel Extraction Mine</td><td></td><td></td><td></td></tr><tr><td>Ballast Lantern</td><td></td><td></td><td></td><td>Lampwrights Program</td><td></td><td></td><td></td></tr><tr><td>Bandaged Rose Window</td><td></td><td></td><td></td><td>Lantern Loft</td><td></td><td></td><td></td></tr><tr><td>Blind Corridor Oscilloscope</td><td></td><td></td><td></td><td>Lanternfin Dive</td><td></td><td></td><td></td></tr><tr><td>Blind Dice Courier</td><td></td><td></td><td></td><td>Lanternwing Roundup</td><td></td><td></td><td></td></tr><tr><td>Bomb Manual From Hell</td><td></td><td></td><td></td><td>Letter Rapids</td><td></td><td></td><td></td></tr><tr><td>Branch Repair</td><td></td><td></td><td></td><td>LIDAR Blacksite</td><td></td><td></td><td></td></tr><tr><td>Bureaucratic Signature Trap</td><td></td><td></td><td></td><td>Live Control-Flow Wiring Lab</td><td></td><td></td><td></td></tr><tr><td>Cell Gatekeeper</td><td></td><td></td><td></td><td>Living Scaffold</td><td></td><td></td><td></td></tr><tr><td>Charter of the Nine Cantons</td><td></td><td></td><td></td><td>Magnetic-Stripe Purgatory</td><td></td><td></td><td></td></tr><tr><td>Circle Limit Twist</td><td></td><td></td><td></td><td>Motion-Only Ghost Jigsaw</td><td></td><td></td><td></td></tr><tr><td>Clockbeat Catacomb</td><td></td><td></td><td></td><td>Museum of Lost Gestures</td><td></td><td></td><td></td></tr><tr><td>Clockwork Clutch Safe</td><td></td><td></td><td></td><td>Occlusion Shell Swindle</td><td></td><td></td><td></td></tr><tr><td>Clockwork Doppelgänger Customs</td><td></td><td></td><td></td><td>Parallax / Inertial Jigsaw Alignment</td><td></td><td></td><td></td></tr><tr><td>Cloudpost Circuit</td><td></td><td></td><td></td><td>Parallax Orchard</td><td></td><td></td><td></td></tr><tr><td>Cloudstep Caddie</td><td></td><td></td><td></td><td>Parallel Grillmaster</td><td></td><td></td><td></td></tr><tr><td>Collision Chimes</td><td></td><td></td><td></td><td>Pearl Lattice</td><td></td><td></td><td></td></tr><tr><td>Concertina Courier</td><td></td><td></td><td></td><td>Photograph Eats the Room</td><td></td><td></td><td></td></tr><tr><td>Consent Gauntlet</td><td></td><td></td><td></td><td>Pocket Locksmith</td><td></td><td></td><td></td></tr><tr><td>Consequences Boss</td><td></td><td></td><td></td><td>Polarized Palimpsest</td><td></td><td></td><td></td></tr><tr><td>Coordinates by Another Name</td><td></td><td></td><td></td><td>Polycube Parcel</td><td></td><td></td><td></td></tr><tr><td>Courtesy Junction</td><td></td><td></td><td></td><td>Polyrhythm Customs</td><td></td><td></td><td></td></tr><tr><td>Crackglaze Crossing</td><td></td><td></td><td>0</td><td>Popup Exorcist</td><td></td><td></td><td></td></tr><tr><td>CRAFTCHA: Alchemy Bench</td><td></td><td></td><td></td><td>Rayglass Vault</td><td></td><td></td><td></td></tr><tr><td>Crash-Deadline Hovercar</td><td></td><td></td><td></td><td>Recursive Dollhouse Smuggling</td><td></td><td></td><td></td></tr><tr><td>Crater Walker</td><td></td><td></td><td></td><td>Reflow Vitrine</td><td></td><td></td><td></td></tr><tr><td>Cursor-Controlled Constellation Hunt</td><td></td><td></td><td></td><td>Reload Interruption</td><td></td><td></td><td></td></tr><tr><td>Dead Man&#x27;s Switch</td><td></td><td></td><td></td><td>Reveal to Identify</td><td></td><td></td><td></td></tr><tr><td>Downsky Causeway</td><td></td><td></td><td></td><td>Ribbon Switchboard</td><td></td><td></td><td></td></tr><tr><td>Dual-Projection Sculpture Rig</td><td></td><td></td><td></td><td>Rising Causeway</td><td></td><td></td><td></td></tr><tr><td>Elastic Membrane Sorter</td><td></td><td></td><td></td><td>Rorschach / Subjective Prompt With A Fixed Rubric</td><td></td><td></td><td></td></tr><tr><td>Elbow Engine</td><td></td><td></td><td></td><td>Rotate The Wrong Thing Upright</td><td></td><td></td><td></td></tr><tr><td>Ember Anvil</td><td>O</td><td></td><td></td><td>Rotating On-Screen Keyboard</td><td>O</td><td></td><td></td></tr><tr><td>Facet Lantern</td><td></td><td></td><td></td><td>Shadow Crime Lab</td><td></td><td></td><td></td></tr><tr><td>Firewatch Fold</td><td></td><td></td><td></td><td>Slot-Reel Character Capture</td><td></td><td></td><td></td></tr><tr><td>First Change Memory Five-Second Rule</td><td></td><td></td><td></td><td>Surveyors Toybox</td><td></td><td></td><td></td></tr><tr><td>Five-System Verification Reactor</td><td></td><td></td><td></td><td>The Flat Prisoner</td><td></td><td></td><td></td></tr><tr><td>Flip-Gate Cascade</td><td></td><td></td><td></td><td>The Last Carbon Isles</td><td></td><td></td><td></td></tr><tr><td>Fluke Census</td><td></td><td></td><td></td><td>The Unlabeled Drawer</td><td></td><td></td><td></td></tr><tr><td>Forced-Perspective Moving Day</td><td></td><td></td><td></td><td>The Unwatched Wing</td><td></td><td></td><td></td></tr><tr><td>Four-Pane Pilgrimage</td><td></td><td></td><td></td><td>Three-Camera Claw Machine</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Tiny FPS Customs</td><td></td><td></td><td></td></tr><tr><td>Four-Tab Robot Handshake</td><td></td><td></td><td></td><td>Tomographic Baggage Surgery</td><td></td><td></td><td></td></tr><tr><td>Funeral Ritual</td><td></td><td></td><td></td><td>Top-Face Dice Arithmetic</td><td></td><td></td><td></td></tr><tr><td>Gyroscopic Tilt Board</td><td></td><td></td><td></td><td>Two-Lamp Dyeworks</td><td></td><td></td><td></td></tr><tr><td>Hearthlift Courier</td><td></td><td></td><td></td><td>Wonky Text Under Hostile Rendering</td><td></td><td></td><td></td></tr><tr><td>Hologram Silhouette Foundry Horizon Relay</td><td>0</td><td>C</td><td></td><td>Wrong Number Zero-G Cable Autopsy</td><td></td><td></td><td></td></tr><tr></table>

sandbox as full-resolution JPEG images, encoded once, and returned without resizing or re-encoding.   
A single request can execute an action and return the subsequent screenshot.

Measurement. We measure latency from immediately before the HTTP request until its complete response, including the image, has arrived. Model inference and subsequent image loading and decoding are outside this measurement.

In isolated timing tests with one client, we measured 150 sequential requests per condition after five warm-up requests. Median latencies were 39.2 ms for Rotating On-Screen Keyboard, 42.5 ms for an idle LIDAR Blacksite scene, and 46.8 ms while issuing LIDAR scan clicks. The corresponding 95th percentiles were 52.1, 55.3, and 70.7 ms.

## F EVALUATION DETAILS

This appendix details the evaluation protocol summarized in Section 3.2.

## F.1 MODELS AND HARNESSES

We evaluate GPT-6 Astra with low and xHigh reasoning effort and GPT-6 Luna with xHigh reasoning effort through the Codex CLI. Each GPT model receives a single computer tool. We disable the shell, code execution, web search, browser, and built-in computer-use features. Claude Opus 5.5 with xHigh effort runs through Claude Code with all built-in tools disabled, and receives the same computer tool through an MCP server. Qwen3.5-9B and Qwen3.8-27B are served through the Tinker API with temperature 1.0, top-p 0.95, top-k 20, and at most 2,048 output tokens per response. They use their native computer-use function format with coordinates on a 1000 × 1000 grid, and their context contains the last 100 turns with at most 20 screenshots.

## F.2 ACTION SPACE AND PROMPT

For the GPT models, each call to the computer tool contains an ordered list of mouse and keyboard actions and returns a screenshot after the actions complete. The actions include moving, clicking, dragging, scrolling, typing, pressing keys, holding and releasing keys and mouse buttons, and waiting. The model instructions and the text after each task description both instruct all models to use only the visible controls of the task webpage, without writing code or inspecting the page. Figure 10 shows the complete prompt for Rotating On-Screen Keyboard in the live setting. In the paused setting, the paragraph following Temporal mode is replaced by the text in Figure 11.

## F.3 BUDGETS AND OUTCOMES

Each attempt allows at most 100 steps. A step is one request that contains one or more actions in order. Requests that only return a screenshot do not consume steps. Each attempt has a time limit of six hours, except that attempts of Astra (xHigh) on the real-time tasks at 1×, 4×, and 16× slowdown and in the paused setting have a limit of one hour. Attempts that reach the time limit count as failures. We exclude and rerun attempts that end with a provider or infrastructure error. The task verifier determines whether each attempt succeeds.

## F.4 TIME AND COST

Time is the wall-clock duration of an attempt from the moment the minigame’s clock starts to the end of its episode. It excludes the time spent setting up the environment. Cost is the API-equivalent cost of the recorded token usage,

$$
\mathrm { c o s t } = \frac { \left( n _ { \mathrm { i n p u t } } - n _ { \mathrm { c a c h e d } } \right) p _ { \mathrm { i n p u t } } + n _ { \mathrm { c a c h e d } } p _ { \mathrm { c a c h e d } } + n _ { \mathrm { o u t p u t } } p _ { \mathrm { o u t p u t } } } { 1 0 ^ { 6 } } ,
$$

where reasoning tokens are counted once as output tokens. Table 8 lists the prices. For GPT-6 Astra and Luna, we use OpenAI API prices,<sup>2</sup> and requests with more than 272K input tokens are charged twice the input and cached-input prices and 1.5 times the output price. For the Qwen models, we use OpenRouter prices as of October 2026.<sup>3</sup> OpenRouter lists no discounted price for cached input tokens of Qwen3.5-9B. We therefore charge its cached input tokens at the regular input price. For Claude Opus 5.5, we use the cost that Claude Code reports for each session, which applies Anthropic API prices to the recorded token usage.

![](images/3f0e1006d1f95615c15c7c47ec28bd58197ac6c0b78147c27a83d382d3e8c3ab.jpg)  
Figure 10: Prompt for Rotating On-Screen Keyboard in the live setting.

The task clock is paused while you reason and while inputs are delivered.   
Each screenshot request or step response advances one configured observation   
window, returning its frames in chronological order. The task then freezes on   
the final frame, and the next input acts on that state. A wait sleeps in wall   
time but does not advance the paused task clock. To hold a key or mouse button   
while the task advances, send down in one request and up in a later request.   
Additional screenshot requests advance additional windows while the input   
stays held. Down, wait, up in a single batch does NOT advance paused task time   
between down and up.  
Figure 11: Timing instructions in the paused setting.

Table 8: Prices used to compute cost, in USD per million tokens.
<table><tr><td>Model</td><td>Input</td><td>Cached input</td><td>Output</td></tr><tr><td>Qwen3.5-9B</td><td>0.17</td><td></td><td>0.25</td></tr><tr><td>Qwen3.8-27B</td><td>0.20</td><td>0.09</td><td>2.50</td></tr><tr><td>GPT-6 Luna</td><td>0.10</td><td>0.01</td><td>0.50</td></tr><tr><td>GPT-6 Astra</td><td>10.00</td><td>1.00</td><td>50.00</td></tr></table>

## F.5 CODING PROTOCOL

In the coding experiment, the model runs in the Codex CLI with its standard tools inside an isolated sandbox. It receives an HTTP endpoint that accepts the same actions and returns screenshots, together with a short Python client. The model can write programs that process screenshots and issue actions through this endpoint. Access to the task machine, page source, and browser developer tools remains prohibited.

## F.6 INTERFACE RULEBOOKS

For the experiment in Section 4.8, each rulebook describes a minigame’s controls, action effects, fixed world rules, feedback, submission, and recovery. It excludes generated answers, target coordinates, hidden state, and seed-specific layouts or solutions. The rulebook is appended to the task prompt, and all other settings remain unchanged.

## F.7 HUMAN BASELINE

One of the authors played 20 tasks randomly sampled from the 100 real-time tasks at difficulty 5 with full interaction in the live setting, seeing each minigame for the first time. Each task used the same seed that the models received. The author solved 9 of the 20 tasks (45.0%), with an average time of 132.7 seconds, or 2.2 minutes, per task.

## G EXAMPLES OF MODEL-WRITTEN CONTROLLERS

We examine saved trajectories from Astra’s coding evaluation at difficulty 5 with full interaction in the live setting. In Rotating On-Screen Keyboard, the model uses perspective transforms and image templates to locate requested keys, converts their positions to screen coordinates, and clicks them. In Wrong Number, it fits curves to waveform pixels from screenshots, estimates the phase error, and adjusts a slider. In Clockwork Clutch Safe, it tracks the marker on each gear, estimates the marker’s angular speed from recent screenshots, and releases each clutch when the marker is predicted to reach its target angle. All three attempts pass the task verifier.

Image-based control also appears in failed attempts. In Gyroscopic Tilt Board, Astra detects the ball and direction indicator, estimates velocity, and moves the tilt control along a sequence of waypoints, but the attempt fails. These examples show how scripts can process observations and issue actions between model responses.

## H REAL-TIME DEFINITION AND REVIEW PROMPTS

## H.1 DEFINITION

The real-time label is assigned to a fixed difficulty and interaction configuration. Let $H _ { t }$ denote the visible interaction history through task time $t , I _ { t }$ the agent’s own prior actions and their task times, and $W _ { t } ^ { w }$ the visible window from $t - w$ to t. Let u denote the terminal task outcome, with less task-active continuation time after delivery of the next action used only to break ties between equal outcomes. Expectations use visible task information and average over future randomness. Private generator or page state is unavailable to the policy.

A grouped control mode $D$ contains every occurrence, across seeds, rounds, and attempts, of the same outcome-affecting control mode. Let $C _ { \Delta } ( D )$ contain the histories in $D$ from which success remains possible after a delay $\Delta$ with no new action. Only histories that become unsalvageable after this delay are excluded. Let $A _ { \Delta } ^ { * } ( H _ { t } )$ be the set of actions maximizing expected u when delivered at $t + \Delta$ and followed by optimal continuation. Let $H _ { t } ^ { \tau }$ be the history after τ with no new action, and $Q ( H _ { t } , a )$ the expected u when a is the next action and is followed by optimal continuation.

A configuration satisfies the real-time definition if and only if there exist a grouped control mode $D ,$ , a delay $\Delta > 0$ , and a bounded visual window w such that $C _ { \Delta } ( D )$ is nonempty and all three conditions hold:

(i) No policy using only the agent’s own action history and task times selects an optimal delayed action for every viable history:

$$
\exists q : I _ { t } \mapsto a \quad \mathrm { s u c h t h a t } \quad \forall H _ { t } \in C _ { \Delta } ( D ) , q ( I _ { t } ) \in A _ { \Delta } ^ { * } ( H _ { t } ) .
$$

(ii) A policy using that history and the recent visual window can select an optimal delayed action for every viable history:

$$
\exists g : ( I _ { t } , W _ { t } ^ { w } ) \mapsto a \quad { \mathrm { s u c h t h a t } } \quad \forall H _ { t } \in C _ { \Delta } ( D ) , g ( I _ { t } , W _ { t } ^ { w } ) \in A _ { \Delta } ^ { * } ( H _ { t } ) .
$$

(iii) For some $H _ { t } \in C _ { \Delta } ( D )$ , action a, and $0 < \tau < \Delta$ , the environment evolves autonomously to $H _ { t } ^ { \tau }$ , success remains possible, and the action loses value or advantage over no action:

$$
\begin{array} { c } { Q ( H _ { t } ^ { \tau } , a ) < Q ( H _ { t } , a ) \qquad \mathrm { o r } } \\ { Q ( H _ { t } ^ { \tau } , a ) - Q ( H _ { t } ^ { \tau } , \mathrm { n o - o p } ) < Q ( H _ { t } , a ) - Q ( H _ { t } , \mathrm { n o - o p } ) . } \end{array}
$$

An unavailable or ineffective action has the value of no action. The strict loss in (iii) must arise from non-clock task-state evolution. Less remaining deadline, later completion alone, a fixed wait that enables an action, and a fixed animation whose result remains available do not satisfy this condition.

All three conditions must be witnessed by the same visible autonomous process and the same $D _ { : }$ $\Delta ,$ and $w .$ . The evidence within $W _ { t } ^ { w }$ must support prediction of the process’s effect at $t + \Delta$ and determine the delayed-optimal action in (i) and (ii). The evidence can occur across frames or in an accumulated trace, chart, or trajectory within one screenshot. Unrelated animation and future randomness independent of the visible window cannot supply this evidence.

The witness delay $\Delta$ is chosen on the time scale of the task behavior. It begins within an active control opportunity and may extend beyond the point where that opportunity degrades or closes. It is independent of the current model provider’s latency. The configured observation window and frame count bound w. A stationary policy over the visibly evidenced dynamic state is allowed.

Pre-run exception. For the configuration-level label, a configuration is classified as not real-time if at least 50% of its generated instances admit a successful pre-run solution. After preparation, such a solution starts the autonomous outcome phase and sends no further outcome-affecting actions until that phase ends. Final submission or certification is administrative and is excluded from this action count. Reviewers establish this threshold from source evidence or a reproducible seed probe. If the threshold is unresolved and could change a positive mathematical classification, the reviewer returns unresolved. Failure of any mathematical condition already establishes a negative classification.

## H.2 REVIEW PROMPTS

The following are the prompt templates used to review real-time labels. The first template assigns one configuration to a reviewer. The second assigns the configurations of one minigame to a reviewer, with a separate judgment for each configuration. Both use the same definition and pre-run exception.

## H.2.1 ONE CONFIGURATION PER REVIEW

```markdown
# Independent real-time classification prompt
Prompt version: `selected-100-pending-v1`
Reuses the historical real-time pilot prompt, adding the settled pre-run threshold and
explicit report destination.
Frozen definition source: `docs/real-time.md
Frozen definition SHA-256: `885cd87d0c59d5017b673a2b8696f251e31eedf0c8e388e32d47101625bcaf09`
Use the configuration and SOURCE_ROOT, CASE_INDEX, and OUTPUT_PATH values supplied by the
launcher. Read source only from that frozen SOURCE_ROOT, not the current checkout. Paths below
are relative to SOURCE_ROOT.
```

You are an independent reviewer classifying exactly one Gym-Anything-RT configuration.

\## Configuration

- Public environment name: \`{public\_environment\_name}   
- Environment ID: \`{environment\_id}\`   
- Difficulty: \`{difficulty}\`   
- Interaction mode: \`{interaction\_mode}

Decide whether this exact configuration is real-time under the frozen mathematical definition below.

Do not use an existing classification, capability annotation, audit result, result matrix, or another reviewer's answer. Do not infer the answer from the environment name or task description.

Read \`AGENTS.md\` and \`docs/real-time.md\` completely. Then read the complete relevant implementation:

1. the exact difficulty and interaction profile in \`controls.json\`;

3. the visible browser implementation and relevant CSS;

4. the grader and verifier; and

5. the solver, when present.

Study a normal screenshot-only solution using visible task controls. Private state and scripted solvers are implementation evidence, not information available to the policy.

```markdown
## Frozen mathematical definition
```

<table><tr><td>`text</td><td></td><td></td><td></td><td></td></tr><tr><td>H_t</td><td>visible interaction history through task time t</td><td></td><td></td><td></td></tr><tr><td>I_t</td><td>the agent&#x27;s own prior actions and task times in H_t</td><td></td><td></td><td></td></tr><tr><td>W_t^w</td><td>the visible window from t-w through t</td><td></td><td></td><td></td></tr><tr><td>u</td><td>terminal task outcome, with less task-active continuation</td><td></td><td></td><td></td></tr><tr><td></td><td>time after delivery of the next action used only to break</td><td></td><td></td><td></td></tr><tr><td>D</td><td>ties between equal outcomes one grouped control mode</td><td></td><td></td><td></td></tr><tr><td>C_Delta(D)</td><td></td><td></td><td>histories in D from which success remains possible after Delta</td><td></td></tr><tr><td></td><td>of no new action</td><td></td><td></td><td></td></tr><tr><td>A*_Delta(H_t)</td><td></td><td></td><td>actions maximising expected u when delivered at t+Delta,</td><td></td></tr><tr><td></td><td></td><td>followed by optimal continuation</td><td></td><td></td></tr><tr><td>H_t^tau</td><td></td><td>history tau later if the agent supplies no new action</td><td></td><td></td></tr><tr><td>Q(H_t, a)</td><td></td><td>expected u if a is the next action, followed by optimal continuation</td><td></td><td></td></tr></table>

Expectations use only visible task information and average over future randomness; private generator or page state is not available to the policy. For a fixed difficulty and interaction configuration, \`D\` contains every occurrence across seeds, rounds, and attempts of the same outcome-affecting control mode. It is not a hand-picked early interval. \`C\_Delta(D)\` removes only histories whose outcome is already unsalvageable after the delay.

A fixed difficulty and interaction configuration of \`E\` is real-time iff there are a grouped control mode \`D\`, a delay \`Delta > 0\`, and a bounded visual window \`w\` such that \`C\_Delta(D)\` is nonempty and:

```markdown
```text
(i) there is no q : I_t -> a
with q(I_t) in A _Delta(H_t) for every H_t in C_Delta(D)
(ii) there is a g : (I_t, W_tˆw) -> a
with g(I_t, W_tˆw) in A<sub>*</sub>_Delta(H_t)
for every H_t in C_Delta(D)
(iii) for some H_t in C_Delta(D), action a, and 0 < tau < Delta,
the environment changes autonomously from H_t to H_tˆtau,
success remains possible from H_tˆtau, and
Q(H_tˆtau, a) < Q(H_t, a)
or
Q(H_tˆtau, a) - Q(H_tˆtau, no-op)
< Q(H_t, a) - Q(H_t, no-op).
```

An unavailable or ineffective action has the value of \`no-op\`. The strict loss in clause (iii) must be caused by non-clock task-state evolution, not merely by less remaining deadline or later completion. An action becoming enabled or more effective after a fixed wait does not satisfy the clause. A fixed animation that later enables controls or permanently archives its result is not an expiring opportunity.

The three clauses must be witnessed by the same phenomenon. The autonomous evolution in clause (iii) must be visibly evidenced within \`W\_tˆw\`, and that evidence must be what makes the delayed-optimal action depend on observation in clauses (i) and (ii). The window must support prediction of the evolution's effect at \`t+Delta\`; unrelated idle animation or future randomness independent of the window cannot supply clause (iii).

The evidence may appear across consecutive frames or as an accumulated visible trace, chart, or trajectory in one screenshot. A single screenshot is allowed when it visibly contains the relevant temporal evidence. This does not admit a static puzzle next to an unrelated animation.

\`Delta\` is a reported witness horizon on the time scale of the task behavior, not the current provider's response latency. It must begin inside a nontrivial active control opportunity and may extend across the point at which that opportunity gets worse or closes. The configured \`observation\_window\_ms\` and \`frames\_per\_observation\` bound \`w\`; they do not force every environment to need every captured frame.

The conjunction has these required consequences:

\- Clause (i) rules out a fixed schedule or a policy driven only by the agent's own action clock.

\- Clause (ii) requires a bounded recent visual window to be sufficient. This excludes unobservable randomness and information that must be recalled only from outside the window. - Clause (iii) requires a useful action opportunity to get worse inside the delay because the task state changes. This excludes fixed waits, minimum-duration holds solvable from the agent's own clock, and information that remains actionable indefinitely.

\- A stationary visual policy is allowed. It can map the currently evidenced changing state to an action without using absolute task time.

\- All three clauses must use one common \`D\`, \`Delta\`, \`w\`, and visible autonomous process.

## ## Pre-run threshold

For the configuration-level catalog label, a configuration is Not real-time when at least 50% of its generated instances admit a successful pre-run solution: after preparation, the solution starts the autonomous outcome phase and sends no more outcome-affecting actions until that phase ends. Final submission or certification is administrative and does not count as an action during the run.

Record whether the threshold is met, with source-structural evidence or an actual reproducible seed probe. A failed existing solver does not prove that no pre-run solution exists. Do not invent a percentage. If the threshold cannot be determined and it could change a mathematical Yes into No, return unresolved. A configuration that already fails the mathematical definition is No regardless of an unresolved pre-run rate.

## ## Review procedure

1. State the controls available in this exact difficulty and interaction mode.   
2. Describe every period in which the task changes without a new agent action.   
3. State whether an outcome-affecting control remains available during each such period.   
4. Enumerate the legitimate grouped control modes \`D\`.   
5. Try to construct an observation-free policy \`q\` using only \`I\_t\`.   
6. Try to construct a bounded-window policy \`g\` using only \`I\_t\` and \`W\_tˆw\`.   
7. Check whether the same visible autonomous process causes an available action to become   
unavailable or strictly less effective within some \`Delta\`.   
8. For \`yes\`, give one concrete common witness: \`D\`, \`Delta\`, \`w\`, the visible process, why \`q\`   
fails, why \`g\` succeeds, and the expiring or degrading action.   
9. For \`no\`, identify the failed clause and explain why no legitimate choice of \`D\`, \`Delta\`,   
and \`w\` repairs it, or establish that the pre-run threshold overrides a mathematical Yes.   
10. Return \`unresolved\` instead of guessing when the complete source does not determine the   
answer.

Return exactly one JSON object:

```jsonl
```json
{
"case_index": "{case_index}",
"public_environment_name": "{public_environment_name}",
"environment_id": "{environment_id}",
"difficulty": "{difficulty}",
"interaction_mode": "{interaction_mode}",
"label": "yes | no | unresolved",
"controls_available": [],
"autonomous_processes": [],
"grouped_control_modes": [],
"delta_ms": null,
"window_ms": null,
"clause_i": {"holds": true, "reason": ""},
"clause_ii": {"holds": true, "reason": ""},
"clause_iii": {"holds": true, "reason": ""},
"common_visible_process": "",
"pre_run": {
"threshold": 0.5,
"at_least_half": null,
"rate": null,
"basis": "source_structural | seeded_probe | unresolved | not_applicable",
"reason": "",
"seed_probe": null
},
"files_read_completely": [],
"source_evidence": [
{"path": "", "lines": "", "claim": ""}
],
"uncertainties": []
}
```

Write the JSON object, with numeric case\_index and difficulty, to the assigned OUTPUT\_PATH using apply\_patch. This one report is the only permitted persistent output. Do not edit puzzle sources, definitions, historical audits, or another report. Do not inspect other first-pass reviews, existing classification/capability files, evaluation results, or previous conversations. The audit manifest is parent-only; the launcher gives you your exact assignment.

You may run read-only source/math probes when needed, with PYTHONDONTWRITEBYTECODE=1 and without modifying source. For a seed probe, include seeds, numerator/denominator, and reproducible probe code in pre\_run.seed\_probe. No browser or desktop access, external websites/services, nested agents, or empirical CUA evaluation. A source-review report may be unresolved; do not guess to finish.

files\_read\_completely and source\_evidence must use repository-relative paths. Read each complete implementation file needed by the assigned mechanic, including shared helpers actually used. If a shared file is long, read it in successive ranges to EOF; keyword searches do not substitute for full reads. Record concrete code evidence for the exact difficulty and interaction. Do not make changes to obtain a preferred label.

## H.2.2 CONFIGURATIONS GROUPED BY MINIGAME

\# Per-game real-time classification prompt

Prompt version: \`selected-100-per-game-v2   
Preserves the first-phase mathematical definition, pre-run threshold, and evidence   
requirements. Changes only the assignment unit: one reviewer reads one game once and judges   
its requested difficulty/interaction configurations together.   
Frozen definition source: \`docs/real-time.md\`

Frozen definition SHA-256: \`885cd87d0c59d5017b673a2b8696f251e31eedf0c8e388e32d47101625bcaf09

Use the game assignment and SOURCE\_ROOT, CASE\_INDEX, CONFIGURATIONS, and OUTPUT\_PATH values supplied by the launcher. Read source only from that frozen SOURCE\_ROOT, not the current checkout. Paths below are relative to SOURCE\_ROOT.

You are an independent reviewer classifying one Gym-Anything-RT game across its difficulty and interaction configurations in a single review.

## ## Assignment

The launcher supplies the exact public environment name, environment ID, game CASE\_INDEX, and CONFIGURATIONS list. Each configuration has its own numeric case\_index, difficulty, and interaction\_mode. Normally the list contains all ten configurations: five difficulties crossed with Full and Simplified. A partially completed game may supply only its unfinished configurations. Read all ten active profiles together to understand their differences, but return judgments only for the supplied CONFIGURATIONS. Do not inspect or infer the already completed judgments.

Read each shared implementation file once, then apply the definition separately to every requested configuration. Explicitly check mode-specific inputs and level-dependent behavior; do not copy one answer across the game without checking those differences. These are multiple configuration judgments from one fresh game-level reviewer, not independent reviewer contexts per configuration.

Do not use an existing classification, capability annotation, audit result, result matrix, or another reviewer's answer. Do not infer the answer from the environment name or task description.

Read \`AGENTS.md\` and \`docs/real-time.md\` completely. Then read the complete relevant implementation:

1. the exact difficulty and interaction profile in \`controls.json\`;

2. the generator;

3. the visible browser implementation and relevant CSS;

4. the grader and verifier; and

5. the solver, when present.

Study a normal screenshot-only solution using visible task controls. Private state and scripted solvers are implementation evidence, not information available to the policy.

## ## Frozen mathematical definition

\`\`\`text   
H\_t visible interaction history through task time t   
I\_t the agent's own prior actions and task times in H\_t   
W\_tˆw the visible window from t-w through t   
u terminal task outcome, with less task-active continuation   
time after delivery of the next action used only to break   
ties between equal outcomes   
D one grouped control mode   
C\_Delta(D) histories in D from which success remains possible after Delta   
of no new action   
A \_Delta(H\_t) actions maximising expected u when delivered at t+Delta,   
followed by optimal continuation   
H\_tˆtau history tau later if the agent supplies no new action   
Q(H\_t, a) expected u if a is the next action, followed by optimal continuation

Expectations use only visible task information and average over future randomness; private generator or page state is not available to the policy. For a fixed difficulty and interaction configuration, \`D\` contains every occurrence across seeds, rounds, and attempts of the same outcome-affecting control mode. It is not a hand-picked early interval. \`C\_Delta(D)\` removes only histories whose outcome is already unsalvageable after the delay.

A fixed difficulty and interaction configuration of \`E\` is real-time iff there are a grouped control mode \`D\`, a delay \`Delta > 0\`, and a bounded visual window \`w\` such that \`C\_Delta(D)\` is nonempty and:

\`\`\`text   
(i) there is no q : I\_t -> a   
with q(I\_t) in A<sub>\*</sub>\_Delta(H\_t) for every H\_t in C\_Delta(D)   
(ii) there is a g : (I\_t, W\_tˆw) -> a   
with g(I\_t, W\_tˆw) in A<sub>\*</sub>\_Delta(H\_t)   
for every H\_t in C\_Delta(D)   
(iii) for some H\_t in C\_Delta(D), action a, and 0 < tau < Delta,   
the environment changes autonomously from H\_t to H\_tˆtau,

success remains possible from H\_tˆtau, and

Q(H\_tˆtau, a) < Q(H\_t, a)

An unavailable or ineffective action has the value of \`no-op\`. The strict loss in clause (iii) must be caused by non-clock task-state evolution, not merely by less remaining deadline or later completion. An action becoming enabled or more effective after a fixed wait does not satisfy the clause. A fixed animation that later enables controls or permanently archives its result is not an expiring opportunity.

The three clauses must be witnessed by the same phenomenon. The autonomous evolution in clause (iii) must be visibly evidenced within \`W\_tˆw\`, and that evidence must be what makes the delayed-optimal action depend on observation in clauses (i) and (ii). The window must support prediction of the evolution's effect at \`t+Delta\`; unrelated idle animation or future randomness independent of the window cannot supply clause (iii).

The evidence may appear across consecutive frames or as an accumulated visible trace, chart, or trajectory in one screenshot. A single screenshot is allowed when it visibly contains the relevant temporal evidence. This does not admit a static puzzle next to an unrelated animation.

\`Delta\` is a reported witness horizon on the time scale of the task behavior, not the current provider's response latency. It must begin inside a nontrivial active control opportunity and may extend across the point at which that opportunity gets worse or closes. The configured \`observation\_window\_ms\` and \`frames\_per\_observation\` bound \`w\`; they do not force every environment to need every captured frame.

The conjunction has these required consequences:

\- Clause (i) rules out a fixed schedule or a policy driven only by the agent's own action clock.

\- Clause (ii) requires a bounded recent visual window to be sufficient. This excludes unobservable randomness and information that must be recalled only from outside the window.

\- Clause (iii) requires a useful action opportunity to get worse inside the delay because the task state changes. This excludes fixed waits, minimum-duration holds solvable from the agent's own clock, and information that remains actionable indefinitely.

\- A stationary visual policy is allowed. It can map the currently evidenced changing state to an action without using absolute task time.

All three clauses must use one common \`D\`, \`Delta\`, \`w\`, and visible autonomous process.

## ## Pre-run threshold

For the configuration-level catalog label, a configuration is Not real-time when at least 50% of its generated instances admit a successful pre-run solution: after preparation, the solution starts the autonomous outcome phase and sends no more outcome-affecting actions until that phase ends. Final submission or certification is administrative and does not count as an action during the run.

Record whether the threshold is met, with source-structural evidence or an actual reproducible seed probe. A failed existing solver does not prove that no pre-run solution exists. Do not invent a percentage. If the threshold cannot be determined and it could change a mathematical Yes into No, return unresolved. A configuration that already fails the mathematical definition is No regardless of an unresolved pre-run rate.

## ## Review procedure

1. State the controls available in this exact difficulty and interaction mode.

2. Describe every period in which the task changes without a new agent action.

3. State whether an outcome-affecting control remains available during each such period.

4. Enumerate the legitimate grouped control modes \`D\`.

5. Try to construct an observation-free policy \`q\` using only \`I\_t\`.

6. Try to construct a bounded-window policy \`g\` using only \`I\_t\` and \`W\_tˆw\`.

7. Check whether the same visible autonomous process causes an available action to become unavailable or strictly less effective within some \`Delta\`.

8. For \`yes\`, give one concrete common witness: \`D\`, \`Delta\`, \`w\`, the visible process, why \`q\` fails, why \`g\` succeeds, and the expiring or degrading action. fails, why gsucceeds, and the expiring or degrading action.

9. For \`no\`, identify the failed clause and explain why no legitimate choice of \`D\`, \`Delta\`, and \`w\` repairs it, or establish that the pre-run threshold overrides a mathematical Yes. 10. Return \`unresolved\` instead of guessing when the complete source does not determine the answer.

Return exactly one game-level JSON object with numeric game case\_index, environment\_id, public\_environment\_name, and configuration\_reports. configuration\_reports is an array containing exactly one object for each supplied configuration, with no duplicates or omitted or extra configurations. Each array member uses this schema (case\_index is the configuration case\_index, not the game case\_index):

```jsonl
```json
{
"case_index": "{case_index}",
"public_environment_name": "{public_environment_name}",
"environment_id": "{environment_id}",
"difficulty": "{difficulty}",
"interaction_mode": "{interaction_mode}",
"label": "yes | no | unresolved",
"controls_available": [],
"autonomous_processes": [],
"grouped_control_modes": [],
"delta_ms": null,
"window_ms": null,
"clause_i": {"holds": true, "reason": ""},
"clause_ii": {"holds": true, "reason": ""},
"clause_iii": {"holds": true, "reason": ""},
"common_visible_process": "",
"pre_run": {
"threshold": 0.5,
"at_least_half": null,
"rate": null,
"basis": "source_structural | seeded_probe | unresolved | not_applicable",
"reason": "",
"seed_probe": null
},
"files_read_completely": [],
"source_evidence": [
{"path": "", "lines": "", "claim": ""}
],
"uncertainties": []
}
```

Write the single game-level JSON object to the assigned OUTPUT\_PATH using apply\_patch. Every configuration\_reports member must contain numeric case\_index and difficulty. The game-level object must identify the assigned game and contain the complete requested configuration set. This one game report is the only permitted persistent output. Do not edit puzzle sources, definitions, historical audits, or another report. Do not inspect other first-pass reviews, existing classification/capability files, evaluation results, or previous conversations. The audit manifest is parent-only; the launcher gives you your exact assignment.

You may run read-only source/math probes when needed, with PYTHONDONTWRITEBYTECODE=1 and without modifying source. For a seed probe, include seeds, numerator/denominator, and reproducible probe code in pre\_run.seed\_probe. No browser or desktop access, external websites/services, nested agents, or empirical CUA evaluation. A source-review report may be unresolved; do not guess to finish.

files\_read\_completely and source\_evidence must use repository-relative paths. Read each complete implementation file needed by the assigned mechanic, including shared helpers actually used. If a shared file is long, read it in successive ranges to EOF; keyword searches do not substitute for full reads. Record concrete code evidence for the exact difficulty and interaction. Do not make changes to obtain a preferred label.

Before finishing, load your JSON and check the game identity, exact requested configuration coverage, numeric IDs, label/clause consistency, and all cited file paths and line bounds. Shared evidence may be reused where it applies, but every configuration must remain independently understandable. A structural check is not evidence that the substantive classification is correct. Do not read other reviewers' reports or the audit manifest.

## I EXTENDED RELATED WORK

This appendix expands on the related work discussion in Section 5.

Benchmarks for interactive visual agents. Existing benchmarks for vision-language agents span web, desktop, mobile, game, and embodied environments. Web benchmarks such as WebArena and VisualWebArena provide self-hosted websites with realistic tasks (Zhou et al., 2024; Koh et al., 2024), while desktop benchmarks such as OSWorld and Windows Agent Arena evaluate open-ended tasks across operating system applications (Xie et al., 2024; Bonatti et al., 2024). OSWorld 2.0 further extends this setting to 108 long-horizon workflows, where the best agent completes only 20.6% of tasks (Yuan et al., 2026). On mobile, AndroidWorld provides tasks across 20 Android apps (Rawles et al., 2024), and Programming with Pixels and Gym-Anything extend computer-use evaluation to software engineering and to 200 professional software applications (Aggarwal & Welleck, 2025; Aggarwal et al., 2026). Game benchmarks evaluate the same models on complete games, ranging from classic video games (Zhang et al., 2026) and reinforcement learning environments such as NetHack (Paglieri et al., 2025) to platformers, puzzle games, and browser games (Hu et al., 2025a; Zheng et al., 2025a; Ying et al., 2026; Ouyang et al., 2026). ARC-AGI-3 evaluates exploration and goal inference in turn-based abstract games, where humans solve all environments while frontier models solve less than 1% (ARC Prize Foundation, 2026). Embodied benchmarks similarly evaluate agents in simulated homes and physical environments (Savva et al., 2019; Shridhar et al., 2020; Srivastava et al., 2021; Yang et al., 2025c). However, since each task in these benchmarks requires many capabilities at once, low performance does not identify which capability limits the agent.

Several works attempt to attribute performance to individual capabilities. SmartPlay maps each of its six games to a subset of nine agent capabilities (Wu et al., 2023), LVLM-Playground evaluates perception, question answering, rule following, and end-to-end play separately on the same board games (Wang et al., 2025c), and Embodied Agent Interface decomposes embodied decision making into separate modules, arguing that final success rates cannot pinpoint which ability is missing (Li et al., 2024). VS-Bench similarly reports perception, strategic reasoning, and decision making separately (Xu et al., 2026). Other benchmarks vary conditions of the same task, such as difficulty tiers (Davydova et al., 2025; Zheng et al., 2025b), initial states (Zhao et al., 2025), or interface changes between actions (Liu et al., 2026). However, these works either operate through text or turn-based interfaces, or vary a single condition without controlling the capabilities a task requires. In contrast, GYM-ANYTHING-RT annotates the capabilities required by each configuration, and varies difficulty, interaction interface, and real-time setting on the same minigame.

Capability-specific evaluation of VLMs. A large body of work evaluates individual capabilities of VLMs through question answering. For 3D spatial understanding, VSI-Bench, MindCube, 3DSR-Bench, and OmniSpatial evaluate spatial relations, viewpoint changes, and mental models of scenes (Yang et al., 2025a; Wang et al., 2025b; Ma et al., 2024; Jia et al., 2025), while BLINK shows that perception tasks that humans solve easily remain difficult for multimodal models (Fu et al., 2024). For temporal reasoning, TempCompass, TOMATO, and Vinoground evaluate speed, direction, and event order in videos (Liu et al., 2024; Shangguan et al., 2024; Zhang et al., 2024), with TOMATO showing that many questions in earlier video benchmarks can be answered from a single frame. For GUI grounding, SeeClick and ScreenSpot-Pro evaluate whether models can locate screen elements from instructions (Cheng et al., 2024; Li et al., 2025). Finally, PuzzleVQA, AlgoPuzzleVQA, and VisualPuzzles evaluate abstract and algorithmic reasoning over images (Chia et al., 2024; Ghosal et al., 2024; Song et al., 2025), and MMMU and MMMU-Pro evaluate multidisciplinary reasoning over images and text (Yue et al., 2024; 2025). However, in all of these benchmarks the model answers a question about a static input, without having to act on its understanding. Closer to our work, iVISPAR evaluates the spatial reasoning of VLM agents on a sliding-tile puzzle presented in 3D, 2D, and text (Mayer et al., 2025), and VisGym provides 17 interactive environments with configurable difficulty and input representation (Wang et al., 2026b). GYM-ANYTHING-RT complements these works by covering temporal reasoning, exploration and interface understanding, and real-time interaction, and by annotating the capabilities required by each configuration.

Real-time interaction. Acting in dynamic environments requires accounting for the delay between observing the environment and executing an action. In robotics, real-time action chunking and asynchronous inference execute previously planned actions while the next chunk is computed (Black et al., 2025; Shukor et al., 2025), and VLASH conditions the policy on the state predicted at execution time (Tang et al., 2025). Reinforcement learning has similarly studied decision making under action and observation delays (Ramstedt & Pal, 2019; Derman et al., 2021). For computer-use agents, OSWorld-Human measures end-to-end latency and finds that agents take 2.7 to 4.3 times more steps than necessary (Abhyankar et al., 2026). Game benchmarks have taken different approaches to latency. GameWorld pauses the game during inference and reports a separate real-time study with two models (Ouyang et al., 2026), VideoGameBench runs games in real time and additionally provides a Lite version that pauses the emulator while the agent is thinking (Zhang et al., 2026), and Orak compares paused and real-time play on StarCraft II (Park et al., 2025). Real-Time Reasoning Gym advances text-based games after a fixed number of generated tokens, comparing reactive and planning agents under increasing time pressure (Wen et al., 2026). Our work complements these efforts by evaluating the same minigame and seed in the live, slowed, and paused settings, and by pairing each real-time minigame with a non-real-time counterpart.

Automated environment construction. A growing body of work uses coding agents and LLMs to construct environments for agents. Gym-Anything uses a creation agent and an independent audit agent to convert 200 software applications into computer-use environments (Aggarwal et al., 2026). CUA-Gym separates an agent that constructs environment states from one that writes reward functions (Wang et al., 2026a), and GUI-GENESIS converts real applications into lightweight web environments with executable rewards (Cao et al., 2026). AutoEnv generates 36 text-based environments with 358 levels (Zhang et al., 2025), while AgentGen and OMNI-EPIC generate environments together with tasks of varying difficulty (Hu et al., 2025b; Faldor et al., 2024). Similar pipelines construct software engineering tasks from code repositories (Pan et al., 2025; Yang et al., 2025b) and 3D scenes for embodied agents (Yang et al., 2024). Procedural generation further provides task instances with adjustable complexity, including Procgen games (Cobbe et al., 2019), MiniHack and BabyAI (Samvelyan et al., 2021; Chevalier-Boisvert et al., 2019), and verifiable reasoning problems in Reasoning Gym and Enigmata (Stojanovski et al., 2025; Chen et al., 2025). However, these approaches construct software, text, or game environments, and do not produce interactive visual minigames whose difficulty and interaction interface can be varied while preserving the underlying task. We extend the creator-auditor approach to such minigames, where each minigame has a non-real-time counterpart, five difficulty levels, two interaction modes, and an oracle solution.

CAPTCHAs and visual puzzles. Many of the seed minigames in GYM-ANYTHING-RT are derived from CAPTCHAs and puzzle games. CAPTCHA research has studied whether automated solvers can pass challenges designed to distinguish humans from bots. Recent works solve reasoning and visual CAPTCHAs with LLMs and VLMs (Deng et al., 2024; Teoh et al., 2025), analyze which types of challenges remain difficult for these models (Wang et al., 2025a; Wu et al., 2025), and train GUI agents on interactive CAPTCHAs (Chen et al., 2026). Open CaptchaWorld evaluates web agents on 20 types of interactive CAPTCHAs, where the best agent achieves 40.0% compared to 93.3% for humans (Luo et al., 2025). Beyond CAPTCHAs, KORGym, DiG-bench, and VGRP-Bench evaluate multi-turn reasoning in text, game, and grid puzzles (Shi et al., 2025; Battleday et al., 2026; Ren et al., 2025). However, these works primarily target security or reasoning in isolation. In contrast, GYM-ANYTHING-RT uses such challenges as seed minigames, and adds controlled capability requirements, difficulty levels, interaction interfaces, and real-time settings.