# From Reasoning to Action: Benchmarking Frontier AI Models on LIBERO-10 Robot Manipulation Tasks

May Espinola<sup>\*</sup>, Qifan Zhang<sup>\*</sup>, Yu Xiang <sup>\*</sup>Equal contribution

Intelligent Robotics and Vision Lab Department of Computer Science, The University of Texas at Dallas

October 2026 • Version 1.0 •

Project: <sub>https:</sub>//<sub>irvlutd.github.io</sub>/<sub>LLM-LIBERO</sub>/

Code: <sub>https:</sub>//<sub>github.com</sub>/<sub>IRVLUTD</sub>/<sub>LLM\_LIBERO</sub>

## ABSTRACT

Recent advances in frontier AI models have demonstrated remarkable capabilities in visual understanding, reasoning, planning, and tool use, raising the possibility of using general-purpose AI models for robot manip ulation without task-specific training. However, their manipulation capabilities have not been systematically evaluated on standardized robotics benchmarks. In this report, we present a unified closed-loop manipulation framework for evaluating frontier AI models on the LIBERO-10 benchmark. Our framework enables models to control a robot directly through visual observations, robot proprioception, and structured tool calls, primarily by specifying target end-efector poses and gripper commands, without task-specific fine-tuning or pretrained manipulation skills. We evaluate six frontier AI models, including GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna, Claude Opus 5, Claude Opus 5.5, and Gemini-4 Argon, across three reasoning efort levels, resulting in 18 configurations and 1,800 evaluation episodes. Our results show that GPT-6 Astra achieves the highest success rate of 44%, followed by Gemini-4 Argon at 43%. Increasing reasoning efort generally improves manipulation performance, but the benefits vary across models and come at additional inference cost. We also observe substantial diferences across tasks, with models performing relatively well on object placement tasks but struggling with precise spatial arrangements and complex multi-step interactions. These findings demonstrate both the potential and limitations of frontier AI models for robot manipulation, highlighting the challenges of translating reasoning capabilities into reliable physical actions. We release our evaluation framework and experimental results to support reproducible benchmarking and future research on reasoning-based robot manipulation.

## 1 Introduction

Recent advances in frontier artificial intelligence (AI) models have demonstrated remarkable capabilities in visual understanding, reasoning, planning, and tool use [1]. In particular, recent demonstrations of GPT-6 Astra have shown impressive robot manipulation behaviors, including interpreting visual observations, generating robot actions, and correcting mistakes through closed-loop interaction [2, 3, 4]. NVIDIA’s Astranomy report [5] further demonstrates the potential of frontier AI models for robot manipulation through diferent control configurations, including direct action generation, agentic tool use, and supervision of pretrained robot policies. These results suggest that general-purpose AI models may be capable of performing robot manipulation without task-specific policy training.

Despite these promising demonstrations, an important question remains: How well do frontier AI models perform on established robot manipulation benchmarks? Existing evaluations often use diferent tasks, robot platforms, and control interfaces, making systematic comparisons dificult. Connecting frontier AI models to standardized benchmarks is essential for understanding their capabilities, identifying limitations, and comparing their performance with conventional robot learning approaches.

Motivated by this question, we develop a unified manipulation pipeline to evaluate frontier AI models on LIBERO-10 (also known as LIBERO-Long) [6], a subset of LIBERO-100 containing ten held-out, long-horizon manipulation tasks. LIBERO-10 has been widely adopted for evaluating robot learning methods, including generalist robot policies and vision-language-action models such as Octo [7], OpenVLA [8], π<sub>0.5</sub> [9], UniVLA [10] and UniAct [11]. Built upon the open-source Inspect Robots framework [12], our pipeline enables frontier AI models to interact with the simulation environment through visual observations and structured tool calls, allowing closed-loop manipulation without task-specific training or fine-tuning. We evaluate multiple frontier AI models, including GPT and Claude models, under a common protocol, exploring diferent reasoning modes to investigate the impact of reasoning efort on manipulation performance. Beyond task success rates, we analyze API calls, inference latency, token usage, and monetary cost to understand the tradeofs between reasoning capabilities, manipulation performance, and eficiency.

Our work provides a standardized evaluation of frontier AI models on robot manipulation tasks, ofering insights into their capabilities and limitations. By releasing our code and evaluation results, we provide reproducible, ready-to-use baselines for future research, reducing the need to repeatedly evaluate these computationally expensive models and lowering evaluation costs.

## 2 Closed-Loop Manipulation with Frontier AI Models

We evaluate frontier AI models, including GPT-6, Claude Opus, and Gemini, directly as closed-loop robot manipulation policies, without robot-specific training, demonstrations, or pretrained skill libraries. Unlike approaches that use AI models to generate robot code or orchestrate predefined skills [13, 14, 15, 16], our pipeline focuses on evaluating the models’ intrinsic visual understanding, spatial reasoning, and action planning capabilities. Given raw camera observations, the models must identify objects, estimate their 3D positions, and plan robot motions through a minimal action interface, without relying on external perception, planning, or manipulation skills. This design allows us to assess how efectively frontier AI models can translate their own reasoning into robot actions.

![](images/b48bc576f0ec9d9a778a2f3f5f0c72990d4e0039ea99d826b1109023a4a159fc.jpg)  
Figure 1: Overview of our closed-loop manipulation framework using frontier AI models.

Figure 1 illustrates our closed-loop manipulation pipeline. At each interaction step, the model receives a fixed system prompt, the task instruction, the interaction history, and the latest observation, including camera images and robot state information (Sec. 2.1). Based on these inputs, the model generates a single tool call: move\_to specifies a target end-efector pose and gripper command, while done and give\_up terminate the episode (Sec. 2.2). For each move\_to call, an action adapter translates the target into a sequence of low-level robot commands, which are executed step by step in the LIBERO simulator. Task success is checked after every control step. If the task remains incomplete after the action sequence finishes, updated observations and execution feedback are returned to the model for the next decision (Sec. 2.3). This process continues until the task succeeds, the model terminates the episode, or the evaluation limits are reached. All experiments are conducted in a zero-shot setting, using a task-agnostic system prompt without task-specific hints or demonstrations.

## 2.1 Observations and Prompt Context

Observations. Each observation contains two 256 × 256 RGB images, from the fixed third-person (agentview) camera and the wrist camera, together with the robot state. The robot state has 12 values: the end-efector position (x, y, z) in meters in the world frame, its orientation in the 6D rotation representation (the first two columns of the rotation matrix), the current gripper command, and the two measured finger positions. The gripper command only records whether the model last asked the gripper to open or close; the measured finger positions show where the fingers actually stopped, so the model can tell whether it closed on an object or on nothing. Each observation is sent as one message: the robot state as text, followed by the two labeled camera images. The mode never sees object poses or the success signal.

System prompt. The system prompt, reproduced verbatim in Figure 2, has two parts. The first part is a generic agent instruction that is independent of the robot and the task: it explains the tool-call interface, requires exactly one tool call per turn with a short note on every motion, tells the model when to call done or give\_up, and states the budget of 80 LLM calls. The second part, appended under Embodiment notes, contains the environment guidelines that we wrote for LIBERO. They describe the robot, the coordinate frame, the rotation representation, the gripper behavior, and the available sensing, followed by eight general manipulation rules. The same prompt is used for all ten tasks and all models; only the embodiment name changes with the task. The task instruction is not part of the system prompt. It is LIBERO’s original natural-language task description, which the model receives as the first user message and again at the top of every observation message. The instructions of all ten tasks are listed in Figure 5.

![](images/59fbb9c4334efb6de44206d35adf007177f55ac97ba89084bdaf7dca373a25a0.jpg)  
Figure 2: Exact system prompt. The first paragraph is the generic agent instruction; the text after Embodiment notes is our environment guidelines. Only the embodiment name difers across tasks.

Interaction history. Each LLM call is stateless, so every call resends the fixed context (the system prompt and the task instruction), followed by the interaction history accumulated so far. Text and images in this history are treated diferently. The text history is complete: it contains the text part of every observation since the start of the episode (the task instruction and the 12 robot-state values), the model’s own tool calls and notes, and the execution feedback of each call. Images, in contrast, are kept only for the two most recent observations (the previous and current observations in Figure 1), i.e., at most four images per call; the images of all earlier observations are replaced by a short text placeholder. Because each call adds one observation, the length of the history is bounded by the budget of 80 LLM calls.

move\_to(targets, note)   
Move to absolute Cartesian end-effector targets (meters for positions, radians for rotations, per the   
dimension labels). The motion is a straight line interpolated at a fixed safe speed and the result reports   
its step count. Unnamed dimensions hold their current value. Coordinates are absolute in the embodiment’s   
declared frame; on multi-arm embodiments each arm uses its own base frame and axes may differ between arms   
depending on mounting. Rotation dimensions are absolute targets measured relative to the trial’s start   
orientation (0 means the start orientation) and interpolate linearly without wrapping, so prefer   
intermediate values for large rotations. Per-dimension bounds: x: [-0.6, 0.6], y: [-0.6, 0.6], z: [0.42,   
1.6], r00: [-1, 1], r10: [-1, 1], r20: [-1, 1], r01: [-1, 1], r11: [-1, 1], r21: [-1, 1], gripper: [0, 1].   
- targets (required): Map of dimension name to value. Valid names: x, y, z, r00, r10, r20, r01, r11, r21,   
gripper   
- note (required): What you observe right now in the observation (images, if any, and state), and why you   
chose this motion. The user reads these notes live and in the saved transcript to follow what you see and   
what you decide. Write for them, in one or two plain sentences.   
done(summary, hindsight)   
Declare the task finished. The trial ends; a scorer judges success.   
- summary (required)   
- hindsight (required): see below   
give\_up(reason, hindsight)   
Stop trying; the task cannot be completed. The trial ends.   
- reason (required)   
- hindsight (required): What do you know now that you wish you had known at the start of this episode?   
Concrete, transferable facts about this rig, task, or embodiment (camera mounting and extrinsics, table and   
base geometry, gripper axis and offsets, controller behavior, metric scale), written as advice to a future   
agent attempting the same task. Say ’none’ if nothing qualifies.  
Figure 3: Tool definitions given to the model. The descriptions are reproduced verbatim; they are sent as JSON function schemas with every request.

## 2.2 Action Interface and Execution

Tools. The model acts through three tools, one tool call per turn. Their full definitions are given in Figure 3, and Figure 4 shows one example of each. The tool calls are generated through the native function-calling interface of each model’s API: every request includes the name, description, and argument format of each tool, and the model replies with a structured call that names one tool and gives its arguments in JSON, which the framework then executes. For instance, in one task-0 episode, Claude Opus 5.5 (high efort) returned the following move\_to call, which sets all ten dimensions to rotate the gripper by 90<sup>◦</sup> about the vertical axis while moving over a fallen can:

{"type": "tool\_use",   
"name": "move\_to",   
"input": {   
"targets": {"x": -0.08, "y": 0.115, "z": 0.55,   
"r00": -1.0, "r10": 0.0, "r20": 0.0,   
"r01": 0.0, "r11": 1.0, "r21": 0.0,   
"gripper": 0},   
"note": "The alphabet soup is lying on its side on the table between   
the basket and the milk carton, with its axis along world y.   
I’m finishing the 90 deg yaw so the fingers close along x,   
across the can, and moving over it at about x=-0.08, y=0.115   
while lowering to check alignment."   
}}

Agent view

![](images/89069794523f56cc45a55390888c6383fa1259161873a60adfd5fed1ddfa2adb.jpg)

![](images/b3f27d5acaf3b34bac117d82eeae98e52e427d4d6fca14aac494df3317275472.jpg)

![](images/4e274e910e003f4bd333a64831615be03814c6689cf20362fa35a1c540207365.jpg)  
Wrist view

![](images/3a5308189ef6b3d55eb456cf23a9ebce0634aec3de267349ee61f1216ca3667f.jpg)  
Move\_to

![](images/a71b055d96865a0f30f422d84d78c694b9dc6ca546c4cb79defaaf80ab9fbadc.jpg)  
Done

![](images/269d4c3bef7cc745cf38118b5b4078d2c363b557b5a6994492dda91aea1ad567.jpg)  
Give\_up  
Figure 4: Examples of the three tool calls in task 0 (“put both the alphabet soup and the tomato sauce in the basket”), taken from Claude Opus 5.5 (low efort). Each column shows the agentview (top) and wrist (bottom) images on which the model issued the call. move\_to: holding a can over the basket, the model calls move\_to to move the end efector to an absolute target pose and lower the can into the basket. done: the model believes the task is complete, but it has placed the wrong objects in the basket (the tomato sauce can and a ketchup bottle, while the alphabet soup remains on the table), so the episode fails. give\_up: the arm knocks the basket onto its side; the model judges that the task can no longer be completed and gives up, so the episode fails.

• move\_to(targets, note) moves the end efector to an absolute target pose (Figure 4, left). In each call, the model outputs two arguments. targets gives target values for any subset of the ten action dimensions: the position $( x , y , z )$ in meters in the world frame, the six entries of the 6D rotation representation, and the gripper command. note is a short explanation of what the model observes and why it chose the motion. The tool description tells the model that the end efector moves along a straight line at a fixed safe speed, that unspecified dimensions keep their current value, and that each dimension is bounded: $x , y \in [ - 0 . 6 , 0 . 6 ] \mathrm { m } , z \in [ 0 . 4 2 , 1 . 6 ] \mathrm { m }$ rotation entries in [−1, 1], and the gripper command in [0, 1].

• done(summary, hindsight) lets the model end the episode when it believes the task is complete (Figure 4, middle). No action is executed, and the episode ends immediately. A done call is never scored as a success, for the following reason. LIBERO’s success checker runs after every control step, and as soon as the task is complete it ends the episode as a success (Sec. 2.3). The model is therefore asked for its next call only when the checker has just reported that the task is not yet complete. Since done does not move the robot, the task is still incomplete when the model calls it, so the episode counts as a failure. In other words, a done call means that the model believes it has finished when it has not.

We still provide done because the model never sees the success checker and must decide by itself when the task is finished. Without done, a model that believes it has finished would have no way to stop and would keep acting. The frequency of done calls therefore indicates how often a model mistakes an unfinished task for a finished one.

Algorithm 1 One closed-loop episode. o: observation (agentview and wrist images plus the 12-value robot state); H: prompt context and interaction history sent to the model; c: the model’s tool call; a<sub>1:n</sub>: interpolated end-efector actions; t: control-step counter. The success checker evaluates the simulator state after every control step.

1: o ← Reset(); t ← 0   
2: H ← [system prompt, goal, o]   
3: for k = 1, . . . , 80 do ▷ LLM call budget   
4: c ← LLM(H) ▷ one tool call   
5: if c is done or give\_up then   
6: return Failure ▷ no action executed   
7: end if   
8: a<sub>1:n</sub> ← Interpolate(current pose, c.targets)   
9: for i = 1, . . . , n do   
10: o ← Step(a<sub>i</sub>); t ← t + 1   
11: if SuccessChecker() then   
12: return Success   
13: end if   
14: if t = 900 then ▷ step limit   
15: return Failure   
16: end if   
17: end for   
18: H ← H + [c, feedback(n), o] ▷ incomplete and call finished   
19: end for   
20: return Failure ▷ call budget exhausted

• give\_up(reason, hindsight) declares that the model believes the task cannot be completed, for example after repeated failed grasps or when an object or the container has been knocked out of reach (Figure 4, right); reason explains why. As with done, no action is executed and the episode ends immediately as a failure. The framework also issues give\_up on the model’s behalf when the budget of 80 LLM calls is exhausted.

Hindsight. Both done and give\_up also require a hindsight field, in which the model states what it wishes it had known at the start of the episode, such as camera placement, controller behavior, or metric scale, written as advice for a future attempt. These notes are logged for analysis but are never fed back to the model, so every episode remains zero-shot.

Execution of move\_to. The adapter (Figure 1) interpolates a straight-line trajectory from the current end-efector pose to the target at a fixed safe speed, producing a sequence of absolute end-efector actions at 20 Hz (about eleven actions per call on average). The number of interpolated steps is returned to the model as the tool result and kept in the interaction history. The actions are then executed one at a time: each 10-D action is converted to LIBERO’s 7-D absolute pose command by orthonormalizing the 6D rotation and converting it to axis-angle form, and LIBERO’s operational space controller (OSC) drives the Franka Panda arm for one control step.

## 2.3 Termination and Evaluation

Per-step success check and the interaction loop. After every control step, LIBERO’s success checker evaluates the task’s goal predicates on the current simulator state. If they are satisfied, the episode ends immediately as a success, even if actions of the current call remain. Otherwise, execution continues with the next action of the same call. When all actions of the call have been executed and the task is still incomplete, the new camera images, robot state, and execution feedback are appended to the history, and the model is queried for the next call. Algorithm 1 summarizes one episode.

![](images/7ba756704597ee207d1d8c0fc532a465542cab86ceda86bd0bedb471dd51e4c2.jpg)

![](images/93d21bdb4bb7b93b161d5f04328994d20b984419d0b84ba9226c7e017e1624f2.jpg)  
(a) Task 0. Put both the alphabet soup and the tomato sauce in the basket.

![](images/e1e263c5b426ab34625a6c542089ae24016d91d910a60b7a11b0f324d8c05c6b.jpg)

![](images/83741d214b504da8a9d9ae93674db0c4baf89a205cbd73fec38a72714b57fb9e.jpg)  
(b) Task 1. Put both the cream cheese box and the butter in the basket.

![](images/4b77f9b167a9a574df9372b3496be63f42bbec3ac677458e717a8f034ef13839.jpg)

![](images/b0b678d874de538b08e811084bbbbf565f6614098810e25b11c1a289bb4ec572.jpg)  
(c) Task 2. Turn on the stove and put the moka pot on it.

![](images/00873539cb61ab94b14e64187956d507c1fabd546a150092e2d1ffa197b70104.jpg)

![](images/678d3660d4f8ca65f8720130f86956eb82283c2898f08ce1b08cf98b05755d8b.jpg)  
(d) Task 3. Put the black bowl in the bottom drawer of the cabinet and close it.

![](images/93715f8694f97c0965d62b581da43e45e977b2238000f7053748963934cc5f1a.jpg)

![](images/f85840e7619b1e35f2822ae9dd05fdd36039f7379f3504ec8e5aa9fd5ed1b2ef.jpg)

![](images/193eadb6b9e7fe93c66ce996097660c0ffc62578850f0c09783e45e24335e6d0.jpg)  
(e) Task 4. Put the white mug on the left plate and the yellow and white mug on the right plate.

![](images/76ae8989b7fcddcd86a58d880612f42cd16a3b5141a1947c994895084e18961a.jpg)  
(f) Task 5. Pick up the book and place it in the back compartment of the caddy.

![](images/a5225d23d3390ca3b18076cb88926afb040d9736d3293ac6082b17d98bf876be.jpg)

![](images/7c26d65087835ebca9824e3393a20ff2d037c21146fd7d9804aa46ef2b136771.jpg)  
(g) Task 6. Put the white mug on the plate and the chocolate pudding to the right of the plate.

![](images/caf93e94c2f65a3b35dd798fce172bdfb175a9e9acbab5fe35e1b3d5ab393a32.jpg)

![](images/6999f2a6fc85e5ecf2778d1541fc422d358ba29d1a18080db6a681e6451a4456.jpg)  
(h) Task 7. Put both the alphabet soup and the cream cheese box in the basket.

![](images/3821b9160ff35c0e1b8dde143dfff44f5eaf72509859a5bc9cfb0aae78f252cf.jpg)  
(i) Task 8. Put both moka pots on the stove.

![](images/598e94dd82b534c2ba042f50050bc1e847ac9de6ebc1b0de787dcfa56109b6d2.jpg)

![](images/25575b3aa45978655f406b305a68c9a529f765554e42c66ba6c6732ae2d027ae.jpg)  
(j) Task 9. Put the yellow and white mug in the microwave and close it.  
Figure 5: The ten LIBERO-10 tasks at initial state 0. Each panel shows the agentview camera (left) and the wrist camera (right) at the first decision step.

Termination. An episode normally ends in one of five ways:

1. the success checker reports success;

2. the model calls done or $\mathsf { g i v e \_ u p : }$

3. the model has used all 80 LLM calls, in which case the framework issues give\_up on its behalf;

4. the episode reaches 900 control steps; this limit is checked after every step, so it can end an episode in the middle of a call; or

5. the next LLM call would exceed the per-episode spending cap, which guards against runaway API cost.

An episode counts as a success only in case 1, when LIBERO’s success checker reports success; episodes that end in any other way count as failures. We report the success rate, the fraction of episodes that end in success.

## 3 Results

## 3.1 Tasks and Experimental Setup

We evaluate frontier AI models on all ten tasks in the LIBERO-10 benchmark [6], indexed from 0 to 9 and illustrated in Figure 5. The benchmark uses a Franka Emika Panda robotic arm equipped with a parallel-jaw gripper. Each task involves multiple object interactions and requires completing a sequence of manipulation actions. Task success is determined by the corresponding success predicate defined in the original LIBERO benchmark.

Each task provides 50 predefined initial states, of which we select the first ten (indices 0–9) for evaluation, resulting in 100 episodes per model configuration. Before evaluation begins, we execute ten neutral simulation steps with the gripper open to allow the scene to settle. The episode step counter is then reset to zero to ensure that these initialization steps are excluded from the evaluation.

## 3.2 Impact of Reasoning Efort on Task Success and Inference Cost

We investigate the impact of reasoning efort on task success and inference cost across frontier AI models on the LIBERO-10 benchmark. For each model, we evaluate three reasoning efort levels (low, medium, and high), with 100 episodes per configuration. Figure 6 presents the relationship between mean API cost per episode and task success rate.

The results reveal substantial diferences in both performance and cost across models. GPT-6 Astra achieves the highest success rate of 44% at high reasoning efort, followed closely by Gemini-4 Argon at 43%. However, both models incur relatively high inference costs, approximately \$7–8 per episode. In contrast, both GPT-6.1 Sol and Claude Opus 5.5 achieve success rate of 33%, at substantially lower costs of approximately \$1.6–2.0 per episode. GPT-6 Luna achieves zero success across all reasoning efort levels despite its low inference cost. Claude Opus 5 exhibits relatively low success rates, ranging from 6% to 12%, even with substantially higher inference costs.

![](images/547a522f8604c48d54a05d2fc15712ff27ef8b85c04f763f10ad39f5173c6bc5.jpg)  
Figure 6: Task success rate versus average API cost per episode across reasoning efort levels. Results are shown for six frontier AI models evaluated on LIBERO-10, with low, medium, and high reasoning efort. Each configuration is evaluated on 100 episodes. Higher reasoning efort generally improves task success but increases inference cost, with substantial diferences across models.

Increasing reasoning efort generally improves task success, although the magnitude of improvement varies considerably across models. GPT-6 Astra improves from 28% to 44%, while Gemini-4 Argon improves from 34% to 43%, accompanied by a substantial increase in inference cost. GPT-6.1 Sol also benefits from increased reasoning efort, improving from 24% to 33%. Claude Opus 5.5 exhibits an improvement from 9% to 33%. Interestingly, Claude Opus 5 does not benefit consistently from increased reasoning efort, with its success rate decreasing from 12% at medium efort to 6% at high efort.

Overall, these results demonstrate that increased reasoning efort can substantially improve robot manipulation performance, but the benefits are model-dependent and come at additional inference cost. The results also highlight an important trade-of between task success and computational expense. While the most capable reasoning models achieve the highest success rates, more costeficient models can provide competitive manipulation performance at a fraction of the cost.

## 3.3 Model-Level Performance Analysis

Table 1 summarizes the detailed results across 18 model configurations, including task success, failure outcomes, API costs, and API calls. Across 1,800 evaluation episodes, we achieve an overall success rate of 22.2%, requiring approximately 99,000 API calls at a total cost of \$4,936.
<table><tr><td># Model</td><td></td><td>Reasoning Effort</td><td>Success</td><td>Rate</td><td></td><td>Failure Limit reached</td><td>Cost ($)</td><td>Calls</td></tr><tr><td></td><td>1 GPT-6 Astra</td><td>high</td><td>44/100</td><td>44%</td><td>53</td><td>3</td><td>709</td><td>4,936</td></tr><tr><td>2</td><td>Gemini-4 Argon</td><td>high</td><td>43/100</td><td>43%</td><td>16</td><td>41</td><td>784</td><td>5,376</td></tr><tr><td>3</td><td>GPT-6 Astra</td><td>medium</td><td>38/100</td><td>38%</td><td>62</td><td>0</td><td>532</td><td>4,773</td></tr><tr><td>4</td><td>Gemini-4 Argon</td><td>medium</td><td>38/100</td><td>38%</td><td>12</td><td>50</td><td>295</td><td>4,848</td></tr><tr><td>5</td><td>Gemini-4 Argon</td><td>low</td><td>34/100</td><td>34%</td><td>28</td><td>38</td><td>205</td><td>5,567</td></tr><tr><td></td><td>6 Claude Opus 5.5</td><td>high</td><td>33/100</td><td>33%</td><td>57</td><td>10</td><td>180</td><td>6,358</td></tr><tr><td></td><td>7 GPT-6.1 Sol</td><td>high</td><td>33/100</td><td>33%</td><td>65</td><td>2</td><td>164</td><td>5,376</td></tr><tr><td></td><td>8 GPT-6.1 Sol</td><td>medium</td><td>30/100</td><td>30%</td><td>69</td><td>1</td><td>120</td><td>5,155</td></tr><tr><td></td><td>9 GPT-6 Astra</td><td>low</td><td>28/100</td><td>28%</td><td>72</td><td>0</td><td>466</td><td>4,833</td></tr><tr><td>10</td><td>GPT-6.1 Sol</td><td>low</td><td>24/100</td><td>24%</td><td>75</td><td>1</td><td>102</td><td>5,155</td></tr><tr><td>11</td><td>Claude Opus 5.5</td><td>medium</td><td>18/100</td><td>18%</td><td>73</td><td>9</td><td>156</td><td>6,796</td></tr><tr><td>12</td><td>Claude Opus 5</td><td>medium</td><td>12/100</td><td>12%</td><td>79</td><td>9</td><td>309</td><td>6,214</td></tr><tr><td>13</td><td>Claude Opus 5</td><td>low</td><td>10/100</td><td>10%</td><td>89</td><td>1</td><td>207</td><td>6,152</td></tr><tr><td>14</td><td>Claude Opus 5.5</td><td>low</td><td>9/100</td><td>9%</td><td>89</td><td>2</td><td>94.2</td><td>5,945</td></tr><tr><td>15</td><td>Claude Opus 5</td><td>high</td><td>6/100</td><td>6%</td><td>88</td><td>6</td><td>567</td><td>6,442</td></tr><tr><td>16</td><td>GPT-6 Luna</td><td>low</td><td>0/100</td><td>0%</td><td>100</td><td>0</td><td>0.294</td><td>852</td></tr><tr><td>17 GPT-6 Luna</td><td></td><td>medium</td><td>0/100</td><td>0%</td><td>79</td><td>21</td><td>18.4</td><td>6,745</td></tr><tr><td>18 GPT-6 Luna</td><td></td><td>high</td><td>0/100</td><td>0%</td><td>57</td><td>43</td><td>28.7</td><td>7,401</td></tr><tr><td colspan="7">Campaign 400/1800 22.2%</td><td>4,936</td><td>98,924</td></tr></table>

Table 1: Evaluation results across 18 model configurations, ranked by success rate. Each configuration is evaluated on 100 episodes using the same tasks and initial states. Failure counts episodes terminated by the model through done() or give\_up() (968) or by reaching the 900-step horizon (195). Limit reached counts episodes cut of before the model could finish: by the 80 model-call allowance, the per-episode spending cap, or a provider output-token limit or API error. Both are counted as unsuccessful episodes. API costs are estimated based on token usage, pricing, and cache usage.

The results also reveal diferences in execution behavior across models. Some configurations exhibit a large number of stopped episodes, particularly Gemini-4 Argon and GPT-6 Luna. Interestingly, higher reasoning efort does not necessarily increase the number of API calls. For example, GPT-6 Astra requires approximately 4,770–4,940 calls across all three efort levels, despite substantial diferences in inference cost. This suggests that the additional cost of higher reasoning efort primarily comes from more expensive API calls rather than more frequent interactions.

## 3.4 Task-Level Performance Analysis

Table 2 presents the evaluation results for each LIBERO-10 task across all 18 model configurations. Task success varies significantly, ranging from 58% for placing cream cheese and butter in a basket to 0% for two tasks involving multiple object manipulations. Tasks involving placing objects into a basket achieve relatively high success rates (50–58%), while tasks requiring more complex interactions, such as opening and closing drawers or microwaves, are considerably more challenging. The two tasks with zero success involve placing multiple objects at specified locations. These results highlight the dificulty of long-horizon manipulation tasks that require precise spatial reasoning and sequential execution. More challenging tasks also tend to incur higher API costs and require more API calls.

<table><tr><td>ID Task</td><td></td><td>Success</td><td></td><td>Rate Limit Reached Cost ($)</td><td></td><td>Calls</td></tr><tr><td>1</td><td>cream cheese + butter in basket</td><td>105/180</td><td>58%</td><td>14</td><td>312</td><td>8,687</td></tr><tr><td>0</td><td>alphabet soup + tomato sauce</td><td>99/180</td><td>55%</td><td>14</td><td>403</td><td>9,485</td></tr><tr><td>7</td><td>alphabet soup + cream cheese</td><td>90/180</td><td>50%</td><td>18</td><td>344</td><td>9,444</td></tr><tr><td>2</td><td>turn on stove, moka pot on it</td><td>59/180</td><td>33%</td><td>22</td><td>496</td><td>9,830</td></tr><tr><td>8</td><td>both moka pots on the stove</td><td>21/180</td><td>12%</td><td>29</td><td>579</td><td>10,373</td></tr><tr><td>3</td><td>black bowl in drawer, close it</td><td>13/180</td><td>7%</td><td>24</td><td>596</td><td>10,752</td></tr><tr><td>5</td><td>book into back of caddy</td><td>12/180</td><td>7%</td><td>28</td><td>493</td><td>8,818</td></tr><tr><td>9</td><td>mug in microwave, close it</td><td>1/180</td><td>1%</td><td>40</td><td>588</td><td>10,310</td></tr><tr><td>6</td><td>mug on plate, pudding to its right</td><td>0/180</td><td>0%</td><td>21</td><td>515</td><td>10,124</td></tr><tr><td>4</td><td>two mugs onto left + right plates</td><td>0/180</td><td>0%</td><td>27</td><td>611</td><td>11,101</td></tr></table>

Table 2: Results by task, over eighteen configurations. Each task is attempted 180 times: six models × three reasoning levels × ten initial states. Instructions are abbreviated.

## 3.5 Per-Model Task Performance

Tables 3–8 provide detailed task-level results for each model under diferent reasoning efort levels. GPT-6 Astra demonstrates strong performance across multiple tasks, particularly object placement and stove manipulation, with success rates generally improving as reasoning efort increases. Gemini-4 Argon achieves consistently high success rates on the three basket placement tasks, including perfect success on several configurations. GPT-6.1 Sol exhibits similar strengths at a lower inference cost, although its performance on more complex tasks remains limited.

Claude Opus 5.5 shows substantial improvements with increased reasoning efort, particularly on object placement and stove manipulation tasks, whereas Claude Opus 5 exhibits inconsistent performance and even degrades at high reasoning efort. GPT-6 Luna fails to complete any task across all three reasoning levels. Across models, tasks involving precise spatial arrangements and multi-step interactions remain particularly challenging. These results suggest that reasoning efort improves manipulation capabilities for some models, but does not necessarily overcome the challenges of complex physical interactions.

Table 3: GPT-6 Astra
<table><tr><td></td><td></td><td colspan="4">low</td><td colspan="4">medium</td><td colspan="4">high</td></tr><tr><td>ID Task</td><td></td><td>Solved /10</td><td></td><td>$</td><td>Calls</td><td></td><td>Solved /10</td><td>$</td><td>Calls</td><td></td><td>Solved /10</td><td>$</td><td>Calls</td></tr><tr><td>0</td><td>soup + tomato sauce</td><td></td><td>7</td><td>55.6</td><td>561</td><td></td><td></td><td>8 42.3</td><td>446</td><td></td><td>9</td><td>63.5</td><td>464</td></tr><tr><td>1</td><td>cream cheese + butter</td><td></td><td>8</td><td>40.2</td><td>454</td><td></td><td></td><td>1031.2</td><td>374</td><td></td><td>10 35.5</td><td></td><td>357</td></tr><tr><td>2</td><td>stove on, moka pot on it</td><td></td><td>2</td><td>42.0</td><td>440</td><td></td><td></td><td>7 50.9</td><td>468</td><td></td><td>8</td><td>61.9</td><td>448</td></tr><tr><td>3</td><td>bowl in drawer, close it</td><td></td><td>1</td><td>52.4</td><td>496</td><td></td><td></td><td>2 75.0</td><td>581</td><td></td><td>2</td><td>99.1</td><td>600</td></tr><tr><td>4</td><td>two mugs onto two plates</td><td></td><td></td><td>67.7</td><td>602</td><td></td><td></td><td>66.6</td><td>534</td><td></td><td></td><td>91.1</td><td>577</td></tr><tr><td>5</td><td>book into back of caddy</td><td></td><td></td><td>25.0</td><td>322</td><td></td><td></td><td>1 31.3</td><td>314</td><td></td><td></td><td>73.2</td><td>481</td></tr><tr><td>6</td><td>mug on plate, pudding R</td><td></td><td></td><td>46.2</td><td>493</td><td></td><td></td><td>52.6</td><td>491</td><td></td><td></td><td>84.9</td><td>561</td></tr><tr><td>7</td><td>soup + cream cheese</td><td></td><td>8</td><td>33.0</td><td>425</td><td></td><td></td><td>8 41.6</td><td>442</td><td></td><td>9</td><td>44.6</td><td>406</td></tr><tr><td>8</td><td>both moka pots on stove</td><td></td><td>2</td><td>49.8</td><td>509</td><td></td><td></td><td>2 65.8</td><td>538</td><td></td><td>6</td><td>60.4</td><td>462</td></tr><tr><td>9</td><td>mug in microwave, close</td><td></td><td></td><td>54.4</td><td>531</td><td></td><td></td><td>74.5</td><td>585</td><td></td><td></td><td>94.4</td><td>580</td></tr><tr><td>All tasks (of 100)</td><td></td><td></td><td>28</td><td>466</td><td></td><td>4,833</td><td></td><td></td><td>38 532</td><td>4,773</td><td></td><td>44 709</td><td>4,936</td></tr></table>

Table 4: Gemini-4 Argon
<table><tr><td></td><td colspan="4">low</td><td colspan="4">medium</td><td colspan="4">high</td></tr><tr><td>ID Task</td><td></td><td>Solved /10</td><td>$</td><td>Calls</td><td></td><td>Solved /10</td><td>$</td><td>Calls</td><td>Solved /10</td><td></td><td>$</td><td>Calls</td></tr><tr><td>0</td><td>soup + tomato sauce</td><td>10</td><td>8.69</td><td>373</td><td></td><td>10</td><td>21.8</td><td>416</td><td></td><td>10</td><td>37.6</td><td>413</td></tr><tr><td>1</td><td>cream cheese + butter</td><td>8</td><td>13.2</td><td>445</td><td></td><td>9</td><td>17.1</td><td>387</td><td></td><td>10</td><td>22.2</td><td>336</td></tr><tr><td>2</td><td>stove on, moka pot on it</td><td></td><td>6 17.9</td><td></td><td>480</td><td></td><td>4 31.1</td><td>526</td><td></td><td>6</td><td>88.5</td><td>514</td></tr><tr><td>3</td><td>bowl in drawer, close it</td><td></td><td></td><td>22.7</td><td>675</td><td></td><td>4 46.7</td><td>567</td><td></td><td>2</td><td>85.0</td><td>542</td></tr><tr><td>4</td><td>two mugs onto two plates</td><td></td><td>30.1</td><td></td><td>763</td><td></td><td>37.5</td><td>653</td><td></td><td></td><td>80.2</td><td>640</td></tr><tr><td>5</td><td>book into back of caddy</td><td></td><td></td><td>18.6</td><td>501</td><td></td><td>1 25.2</td><td>466</td><td></td><td>2</td><td>131</td><td>709</td></tr><tr><td>6</td><td>mug on plate, pudding R</td><td></td><td></td><td>19.7</td><td>591</td><td></td><td>38.8</td><td>613</td><td></td><td></td><td>62.9</td><td>526</td></tr><tr><td>7</td><td>soup + cream cheese</td><td></td><td>9</td><td>12.0</td><td>414</td><td></td><td>1016.2</td><td>380</td><td></td><td>10</td><td>27.8</td><td>370</td></tr><tr><td>8</td><td>both moka pots on stove</td><td></td><td>1</td><td>29.5</td><td>605</td><td></td><td>45.2</td><td>593</td><td></td><td>2</td><td>126</td><td>644</td></tr><tr><td>9</td><td>mug in microwave, close</td><td></td><td></td><td>32.2</td><td>720</td><td></td><td>15.0</td><td>247</td><td></td><td>1</td><td>123</td><td>682</td></tr><tr><td>All tasks (of 100)</td><td></td><td></td><td>34 205</td><td></td><td>5,567</td><td></td><td>38 295 4,848</td><td></td><td></td><td>43</td><td>784</td><td>5,376</td></tr></table>

Table 5: GPT-6.1 Sol
<table><tr><td></td><td></td><td colspan="4">low</td><td colspan="4">medium</td><td colspan="4">high</td></tr><tr><td>ID Task</td><td></td><td>Solved /10</td><td></td><td>$</td><td>Calls</td><td>Solved /10</td><td></td><td>$</td><td>Calls</td><td>Solved /10</td><td></td><td>$</td><td>Calls</td></tr><tr><td>0</td><td>soup + tomato sauce</td><td></td><td>8</td><td>10.7</td><td>548</td><td></td><td>7</td><td>11.9</td><td>522</td><td></td><td>9</td><td>13.3</td><td>484</td></tr><tr><td>1</td><td>cream cheese + butter</td><td></td><td>7</td><td>8.84</td><td>489</td><td></td><td>5</td><td>10.3</td><td>483</td><td></td><td>9</td><td>8.99</td><td>397</td></tr><tr><td>2</td><td>stove on, moka pot on it</td><td></td><td></td><td>12.6</td><td>574</td><td></td><td>4</td><td>12.8</td><td>538</td><td></td><td>4</td><td>18.2</td><td>551</td></tr><tr><td>3</td><td>bowl in drawer, close it</td><td></td><td>1</td><td>10.6</td><td>513</td><td></td><td></td><td>17.4</td><td>635</td><td></td><td></td><td>19.8</td><td>603</td></tr><tr><td>4</td><td>two mugs onto two plates</td><td></td><td></td><td>12.6</td><td>593</td><td></td><td></td><td>13.8</td><td>560</td><td></td><td></td><td>19.0</td><td>598</td></tr><tr><td>5</td><td>book into back of caddy</td><td></td><td>2</td><td>6.86</td><td>383</td><td></td><td>4</td><td>5.48</td><td>325</td><td></td><td>2</td><td>14.3</td><td>480</td></tr><tr><td>6</td><td>mug on plate, pudding R</td><td></td><td></td><td>10.5</td><td>539</td><td></td><td></td><td>12.6</td><td>523</td><td></td><td></td><td>21.4</td><td>642</td></tr><tr><td>7</td><td>soup + cream cheese</td><td></td><td>6</td><td>10.8</td><td>542</td><td></td><td>8</td><td>8.81</td><td>453</td><td></td><td>5</td><td>11.8</td><td>470</td></tr><tr><td>8</td><td>both moka pots on stove</td><td></td><td></td><td>10.1</td><td>511</td><td></td><td>2</td><td>13.4</td><td>551</td><td></td><td>4</td><td>19.9</td><td>599</td></tr><tr><td>9</td><td>mug in microwave, close</td><td></td><td></td><td>8.44</td><td>463</td><td></td><td></td><td>13.4</td><td>565</td><td></td><td></td><td>17.0</td><td>552</td></tr><tr><td>All tasks (of 100)</td><td></td><td></td><td>24 102</td><td></td><td>5,155</td><td></td><td></td><td>30 120</td><td>5,155</td><td></td><td></td><td>33 164</td><td>5,376</td></tr></table>

Table 6: Claude Opus 5.5
<table><tr><td></td><td></td><td colspan="3">low</td><td></td><td colspan="3">medium</td><td colspan="3">high</td></tr><tr><td>ID Task</td><td></td><td>Solved /10</td><td>$</td><td>Calls</td><td>Solved /10</td><td></td><td>$</td><td>Calls</td><td>Solved /10</td><td>$</td><td>Calls</td></tr><tr><td>0</td><td>soup + tomato sauce</td><td></td><td>1 9.87</td><td>635</td><td></td><td>4</td><td>13.3</td><td>639</td><td></td><td>8 12.1</td><td>508</td></tr><tr><td>1</td><td>cream cheese + butter</td><td></td><td>2 8.31</td><td>544</td><td></td><td>7</td><td>10.6</td><td>544</td><td>9</td><td>12.1</td><td>533</td></tr><tr><td>2</td><td>stove on, moka pot on it</td><td></td><td>9.31</td><td>612</td><td></td><td>3</td><td>14.7</td><td>669</td><td>8</td><td>17.8</td><td>638</td></tr><tr><td>3</td><td>bowl in drawer, close it</td><td></td><td>9.52</td><td>584</td><td></td><td>1</td><td>18.6</td><td>739</td><td></td><td>21.7</td><td>712</td></tr><tr><td>4</td><td>two mugs onto two plates</td><td></td><td>10.7</td><td>632</td><td></td><td></td><td>18.7</td><td>748</td><td></td><td>21.3</td><td>679</td></tr><tr><td>5</td><td>book into back of caddy</td><td></td><td>9.07</td><td>544</td><td></td><td></td><td>16.9</td><td>718</td><td></td><td>20.7</td><td>694</td></tr><tr><td>6</td><td>mug on plate, pudding R</td><td></td><td>7.98</td><td>498</td><td></td><td></td><td>15.6</td><td>665</td><td></td><td>18.5</td><td>604</td></tr><tr><td>7</td><td>soup + cream cheese</td><td></td><td>3 9.53</td><td>617</td><td></td><td>3</td><td>14.1</td><td>669</td><td>8</td><td>16.7</td><td>675</td></tr><tr><td>8</td><td>both moka pots on stove</td><td></td><td>10.3</td><td>669</td><td></td><td></td><td>16.2</td><td>726</td><td></td><td>17.9</td><td>691</td></tr><tr><td>9</td><td>mug in microwave, close</td><td></td><td>9.63</td><td>610</td><td></td><td></td><td>17.7</td><td>679</td><td></td><td>21.3</td><td>624</td></tr><tr><td>All tasks (of 100)</td><td></td><td></td><td></td><td>9 94.2 5,945</td><td></td><td></td><td></td><td>18 156 6,796</td><td></td><td>33 180</td><td>6,358</td></tr></table>

Table 7: Claude Opus 5
<table><tr><td></td><td></td><td colspan="4">low</td><td colspan="4">medium</td><td colspan="4">high</td></tr><tr><td>ID Task</td><td></td><td>Solved /10</td><td></td><td>$</td><td>Calls</td><td>Solved /10</td><td></td><td>$</td><td>Calls</td><td>Solved /10</td><td></td><td>$</td><td>Calls</td></tr><tr><td>0</td><td>soup + tomato sauce</td><td></td><td>4</td><td>18.6</td><td>574</td><td></td><td>3</td><td>29.7</td><td>635</td><td></td><td>1</td><td>48.7</td><td>665</td></tr><tr><td>1</td><td>cream cheese + butter</td><td></td><td>5</td><td>14.8</td><td>525</td><td></td><td>4</td><td>28.7</td><td>637</td><td></td><td>2</td><td>44.7</td><td>667</td></tr><tr><td>2</td><td>stove on, moka pot on it</td><td></td><td></td><td>22.1</td><td>630</td><td></td><td>3</td><td>33.3</td><td>610</td><td></td><td>1</td><td>58.4</td><td>669</td></tr><tr><td>3</td><td>bowl in drawer, close it</td><td></td><td></td><td>21.0</td><td>615</td><td></td><td></td><td>30.0</td><td>651</td><td></td><td></td><td>61.8</td><td>708</td></tr><tr><td>4</td><td>two mugs onto two plates</td><td></td><td></td><td>23.9</td><td>659</td><td></td><td></td><td>38.4</td><td>689</td><td></td><td></td><td>74.4</td><td>666</td></tr><tr><td>5</td><td>book into back of caddy</td><td></td><td></td><td>18.7</td><td>563</td><td></td><td></td><td>24.7</td><td>448</td><td></td><td></td><td>68.7</td><td>575</td></tr><tr><td>6</td><td>mug on plate, pudding R</td><td></td><td></td><td>22.7</td><td>609</td><td></td><td></td><td>34.5</td><td>624</td><td></td><td></td><td>60.7</td><td>581</td></tr><tr><td>7</td><td>soup + cream cheese</td><td></td><td></td><td>22.4</td><td>700</td><td></td><td>2</td><td>28.2</td><td>633</td><td></td><td>1</td><td>41.0</td><td>600</td></tr><tr><td>8</td><td>both moka pots on stove</td><td></td><td>1</td><td>22.9</td><td>636</td><td></td><td></td><td>32.6</td><td>628</td><td></td><td>1</td><td>55.0</td><td>690</td></tr><tr><td>9</td><td>mug in microwave, close</td><td></td><td></td><td>19.5</td><td>641</td><td></td><td></td><td>29.1</td><td>659</td><td></td><td></td><td>54.0</td><td>621</td></tr><tr><td>All tasks (of 100)</td><td></td><td></td><td></td><td>10 207</td><td>6,152</td><td></td><td></td><td>12 309</td><td>6,214</td><td></td><td>6</td><td>567</td><td>6,442</td></tr></table>

Table 8: GPT-6 Luna
<table><tr><td rowspan="2" colspan="2">ID Task</td><td colspan="3">low</td><td colspan="3">medium</td><td colspan="3">high</td></tr><tr><td>Solved /10</td><td>$</td><td>Calls</td><td>Solved /10</td><td>$</td><td>Calls</td><td>Solved /10</td><td>$</td><td>Calls</td></tr><tr><td>0</td><td>soup + tomato sauce</td><td></td><td>0.035</td><td>97</td><td></td><td>2.08</td><td>721</td><td></td><td>3.16</td><td>784</td></tr><tr><td>1</td><td>cream cheese + butter</td><td></td><td>0.022</td><td>67</td><td></td><td>1.94</td><td>714</td><td></td><td>2.82</td><td>734</td></tr><tr><td>2</td><td>stove on, moka pot on it</td><td></td><td>0.030</td><td>85</td><td></td><td>1.76</td><td>632</td><td></td><td>3.15</td><td>746</td></tr><tr><td>3</td><td>bowl in drawer, close it</td><td></td><td>0.029</td><td>84</td><td></td><td>2.00</td><td>719</td><td></td><td>2.65</td><td>728</td></tr><tr><td>4</td><td>two mugs onto two plates</td><td></td><td>0.035</td><td>97</td><td></td><td>1.89</td><td>682</td><td></td><td>2.95</td><td>729</td></tr><tr><td>5</td><td>book into back of caddy</td><td></td><td>0.024</td><td>75</td><td></td><td>1.52</td><td>593</td><td></td><td>2.38</td><td>627</td></tr><tr><td>6</td><td>mug on plate, pudding R</td><td></td><td>0.026</td><td>81</td><td></td><td>2.14</td><td>720</td><td></td><td>3.05</td><td>763</td></tr><tr><td>7</td><td>soup + cream cheese</td><td></td><td>0.040</td><td>104</td><td></td><td>2.12</td><td>764</td><td></td><td>2.95</td><td>780</td></tr><tr><td>8</td><td>both moka pots on stove</td><td></td><td>0.027</td><td>82</td><td></td><td>1.15</td><td>517</td><td></td><td>2.67</td><td>722</td></tr><tr><td>9</td><td>mug in microwave, close</td><td></td><td>0.026</td><td>80</td><td></td><td>1.78</td><td>683</td><td></td><td>2.88</td><td>788</td></tr><tr><td colspan="2">All tasks (of 100)</td><td></td><td>0 0.294</td><td>852</td><td></td><td></td><td>0 18.4 6,745</td><td></td><td>0 28.7 7,401</td><td></td></tr></table>

## 3.6 Failure Analysis Through Model Hindsight

Failure and termination behaviors. Of the 1,800 evaluation episodes, 1,400 fail to satisfy the task success predicate. These end in three ways. The model stops the episode itself in 968 cases, by calling give\_up (681) or done (287). A further 195 are stopped by the 900-step simulator horizon, and the remaining 237 are cut of by the 80-call allowance or the spending cap. The first two groups form the Failure column of Table 1 and the third its Limit reached column.

The two ways a model stops itself are informative in opposite directions. A give\_up call means the model has judged the task unreachable; a done call means it believes the task finished while the success predicate is still unsatisfied, so its count measures how often a model mistakes an incomplete task for a complete one.

Learning from model hindsight. Our framework requires each model to provide a hindsight field when calling done or give\_up, describing what it wishes it had known at the beginning of the episode. These self-reported observations provide valuable insights into the challenges encountered during manipulation, including spatial understanding, robot control, grasping, and task-specific interactions. For example, GPT-6.1 Sol at low reasoning efort provides the following hindsight after a failed episode:

“At low heights, forward reach requests were heavily clamped and caused substantial wrist tilt instead of reaching the target. Check actual pose rather than requested pose; fully closed finger positions repeatedly revealed missed grasps. The wrist camera grasp point is near the top of the image, not its center. The stove control is behind the burner, and straight pushes did not reliably actuate it.”

This example illustrates how the model identifies limitations in motion execution, visual grounding, grasp verification, and task-specific interactions. Although these observations are generated by the models themselves and may not always accurately explain the causes of failure, they provide useful hypotheses for understanding unsuccessful manipulation attempts.

Common challenges identified from hindsight. We analyze the hindsight notes to identify recurring challenges across models. One prominent issue is the discrepancy between commanded and executed robot motions. Specifically, 16% of Gemini-4 Argon’s failure notes, 42% of GPT-6 Astra’s, 47% of GPT-6.1 Sol’s, 56% of GPT-6 Luna’s, 88% of Claude Opus 5.5’s, and 91% of Claude Opus 5’s mention that move\_to commands do not fully reach their targets. Models frequently attribute this behavior to motion clamping and suggest repeating commands or checking the actual robot state.

We further verify this observation by comparing commanded end-efector targets with the positions achieved at the next interaction step. The achieved displacement averages approximately 64–72% of the commanded displacement across these models, with 21–36% of motions covering less than half the requested distance. This confirms that incomplete motion execution is a common challenge across models, although the frequency with which models identify it in hindsight varies substantially.

Implications for improving robot manipulation. The hindsight analysis suggests that successful manipulation requires more than high-level reasoning and action planning. Models must also understand the robot’s embodiment, interpret visual observations accurately, and account for discrepancies between commanded and executed actions. Providing more explicit information about robot kinematics, camera geometry, and controller behavior may help reduce these failures. More broadly, model-generated hindsight ofers a promising source of feedback for improving prompts, action interfaces, and closed-loop manipulation strategies in future work.

## 4 Conclusion

In this report, we presented a systematic evaluation of six frontier AI models on the LIBERO-10 robot manipulation benchmark using a unified closed-loop manipulation framework. Across 18 model configurations and 1,800 evaluation episodes, the best-performing model, GPT-6 Astra, achieved a success rate of 44%, demonstrating the potential of general-purpose AI models to perform robot manipulation without task-specific training or fine-tuning.

Our experiments reveal three key findings. First, increasing reasoning efort generally improves manipulation success, although the benefits vary across models and come at additional inference cost. Second, task dificulty has a substantial impact on performance. While frontier AI models perform reasonably well on object placement tasks, they struggle with precise spatial arrangements and complex multi-step interactions. Third, inference costs vary significantly across models, highlighting the importance of considering both task success and computational eficiency.

Overall, our results demonstrate both the promise and current limitations of frontier AI models for robot manipulation. Despite their strong reasoning capabilities, reliably translating visual observations and high-level reasoning into precise physical actions remains challenging. Future research should explore improved spatial grounding, action representations, and integration with low-level robot control and learned manipulation policies. We release our evaluation framework and experimental results to facilitate reproducible benchmarking and future research on reasoning-based robot manipulation.

## References

[1] Arena AI. Arena leaderboard. https://arena.ai/leaderboard, 2026. Accessed: 2026-10-15.

[2] Wenbo Zhang, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin, Wenhao Chai, Haotian Liang, Zhiyang Dou, et al. An unexpected robot policy: Early evaluations of gpt-6 astra on robodojo and beyond. arXiv preprint arXiv:2609.24170, 2026.

[3] Jiayi Su, Yixin Zheng, Mi Yan, Li Yi, Zhizheng Zhang, and He Wang. GPT 6 Astra as an embodied policy. Technical report and code, 2026. URL https://github.com/ anonymous-report-421/eval-of-gpt-6-astra-as-policy.

[4] Galbot Team, Xuchuan Chen, Xiaoqian Cheng, Yu Deng, Lihe Ding, Shaocong Dong, Xiangjun Gao, Haozhe Jia, Zekai Li, Zhoujian Li, et al. Systematically exploring the capabilities of gpt-6 astra as embodied policies. arXiv preprint arXiv:2609.38537, 2026.

[5] Jenai Xuning Yang, Ashna Khetan, and Shun Zhang. Astranomy: The study of GPT-6 Astra as a native robot policy. https://research.nvidia.com/labs/srl/projects/astranomy/, October 2026. Blog post.

[6] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning, 2023. URL https: //arxiv.org/abs/2306.03310.

[7] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[8] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[9] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π<sub>0.5</sub>: a vision-languageaction model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[10] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025.

[11] Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, and Xianyuan Zhan. Universal actions for enhanced

embodied foundation models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22508–22519. IEEE, 2025.

[12] Robocurve. Inspect robots: The open-source evaluation framework for physical ai, 2026. URL https://github.com/robocurve/inspect-robots.

[13] Wenli Xiao, Jia Xie, Tonghe Zhang, Haotian Lin, Letian Fu, Haoru Xue, Jalen Lu, Yi Yang, Cunxi Dai, Zi Wang, et al. Enpire: Agentic robot policy self-improvement in the real world. arXiv preprint arXiv:2606.19980, 2026.

[14] Runyu Lu, Yubo Wu, Ethan Kou, Letian Fu, Wenli Xiao, Ajay Mandlekar, Yinzhen Xu, Guanya Shi, Ken Goldberg, Ang Chen, et al. Aspire: Agentic/skills discovery for robotics. arXiv preprint arXiv:2607.00272, 2026.

[15] Kaiyuan Chen, Shuangyu Xie, Letian Fu, Justin Yu, William Pacini, Sandeep Bajamahal, Hudson Kim, Jaimyn Drake, Daehwa Kim, Haoru Xue, et al. Gap: A graph-as-policy multiagent self-learning harness for variational automation tasks. arXiv preprint arXiv:2607.05369, 2026.

[16] Yuanbo Yang, Siyi Chen, Jun Gao, and Hao Zhang. ORA: Open robot-use agents. Hao AI Lab blog, October 2026. URL https://haoailab.com/blogs/ora/.