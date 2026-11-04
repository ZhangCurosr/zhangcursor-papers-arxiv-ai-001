# AceAction-5M: Hierarchical Goals for Generalist Action Learning Across 3D Game Worlds

Kailing Li<sup>1,2∗</sup> Ruoxuan Zhang<sup>1,3∗</sup> Zheyuan Zhou<sup>1</sup> Boyang Chu<sup>1</sup> Yingshan Lei<sup>1</sup> Hongxia Xie<sup>3</sup> Liang Du<sup>1</sup> Zixun Sun<sup>1</sup>

<sup>1</sup>Tencent <sup>2</sup>East China Normal University <sup>3</sup>Jilin University

https://zhangdaxia22.github.io/AceAction-5M/

![](images/a0d1d9033bf04c503433f12d42103a2981c2f4f1ba078c3981c15cb25bdb06aa.jpg)  
Figure 1. Overview of the AceAction-5M dataset.

## Abstract

Video games provide a practical setting for collecting vision-language-action (VLA) data, whose acquisition in physical environments is costly. Existing datasets, however, remain limited by restricted game coverage, sparse language supervision, or temporal hierarchies that do not necessarily reflect changes in task objectives. We introduce AceAction, a dataset of human gameplay across multiple 3D gamesfor learning skill composition and subgoal tran-

sitions. AceAction organizes demonstrations around complete tasks, linking overall instructions, ordered subgoals, and local skill instructions to temporally aligned visual observations and keyboard-and-mouse actions. To construct these annotations, we develop a vision–language model pipeline that identifies subgoals from changes in objectives or interaction targets, anchors navigation goals to visually recognizable waypoints, and composes adjacent subgoals into task instructions. The resulting demonstrations cover navigation, reconnaissance, combat, interaction, and recovery, with explicit stage boundaries that connect task progression to fine-grained control. We assess annotation quality through model-based verification and human inspection. An audit of500 trajectories yields a 95.8% pass rate across independent reviewer assessments. AceAction provides a data foundation for studying how agents execute, combine, and switch skills to follow complex instructions in dynamic game environments.

## 1. Introduction

Learning vision-language-action (VLA) models requires demonstrations that connect language goals, visual observations, and executable actions [2, 11, 12, 16]. Collecting such data at scale in the physical world is costly, requiring robotic hardware, human supervision, and repeated environment setup [10]. Video games offer a practical setting for collecting large-scale interaction data: human players can perform diverse tasks in rich virtual environments, while visual observations and keyboard-and-mouse actions can be recorded together [20]. Paired with language annotations, these trajectories support learning to translate task instructions into actions grounded in visual observations. In particular, 3D games demand close coordination between high-level decision-making and low-level control, requiring agents to navigate, scout, engage opponents, and interact with the environment through rapid and precise keyboardand-mouse actions. Under partial observability and continuously changing conditions, agents must continually adapt their decisions and actions to environmental feedback.

Tasks in 3D games often require agents to combine multiple skills across successive stages. For example, an agent asked to approach the building, scout the entrance, engage the enemy, andfall back to cover to recover may fight effectively yet continue chasing an opponent when it should retreat. Successful task completion therefore requires agents to understand how each skill serves the current subgoal and to select, compose, and switch skills as the task progresses. This motivates two complementary data requirements. First, hierarchical annotations linking tasks, subgoals, and skill-level instructions: demonstrations should connect overall objectives to local goals and concrete instructions, making explicit what to achieve and how to act, with temporal boundaries indicating subgoal transitions. Second, large-scale action data covering diverse goals and tasks: extensive keyboard-and-mouse trajectories should span varied scenarios and skill combinations, capturing how navigation, reconnaissance, combat, interaction, and recovery are executed and composed for different objectives. Together, these properties support learning both task structure and the behaviors needed to realize it.

Existing game-interaction datasets exhibit limitations in both behavioral coverage and supervision for task structure. EgoCS-400K [7] provides detailed replay-grounded actions and multi-granularity captions, but remains confined to Counter-Strike and primarily describes local clips and action chains. Open-P2P [23] broadens game coverage, yet its sparse, goal-oriented instructions provide limited supervision for connecting fine-grained skill execution with intermediate and overall objectives. GameHorizon [22] introduces multi-horizon instructions aligned with action recordings, but constructs its instruction hierarchy under predefined duration ranges for operations, goals, and strategies. These constraints may split a sustained subgoal or combine several brief subgoals, making temporal hierarchy an imperfect proxy for task-stage structure. These limitations motivate demonstrations with broader behavioral coverage and explicit, objective-driven stage annotations, linking each task to its constituent skills and marking when execution transitions between subgoals.

To bridge this gap, we introduce AceAction-5M, a dataset that unifies multi-game tasks, fine-grained hierarchical instructions, and temporally aligned keyboard-andmouse trajectories. Each episode pairs a global task description with temporally bounded stage-level subgoals, and aligns local action instructions frame by frame with the keyboard-and-mouse actions that realize them. The resulting trajectories capture how navigation, reconnaissance, combat, interaction, and recovery are composed and switched, providing a data foundation for learning to map complex task instructions to continuous, fine-grained control. Our contributions are as follows:

• A hierarchically annotated, multi-game action dataset. AceAction covers 42 games and 5,027.87 hours of play, with 5,476,222 episodes in which task descriptions, stage-level subgoals, and local action instructions are jointly aligned with frame-level keyboard-and-mouse trajectories.

• An annotation scheme that makes stage structure explicit. We define 43 skill categories spanning navigation, reconnaissance, combat, interaction, and recovery, and mark the start and end of every stage, turning when to switch from an implicit property of the data into an explicit, learnable signal.

## 2. Related Work

## 2.1. Game Data and Benchmarks for World Modeling

Gameplay data supports action-conditioned world modeling by aligning visual observations, actions, and environment states. GameNGen [21] learns video prediction from DOOM trajectories collected by a reinforcement learning agent, while DIAMOND [1] demonstrates interactive world modeling using recorded CS:GO gameplay. Wild-World [13] provides frame-level annotations of skeletons, camera poses, and depth for evaluating action following and state consistency.

EgoCS-400K [7] synchronizes player states, actions, and events extracted from CS:GO/CS2 replays with over 10,000 hours of first-person video. Its multi-granularity captions and video generation prompts link visual changes to replay information, providing language supervision for world modeling. GameVerse [9] provides minute-scale gameplay videos with temporally coherent camera trajectories and event-aligned structured captions, supporting long-horizon interactive world modeling.

GameHorizon [22] combines 5,000 hours of gameplay from 21 games with synchronized keyboard-and-mouse actions and three levels of instructions: operations, goals, and strategies. Its offline benchmark evaluates action understanding, goal decomposition, and cross-horizon consistency through multiple-choice questions, with predicted action sequences matched to answer options for scoring. Our work links subtask instructions to waypoints and temporally aligned keyboard-and-mouse trajectories within multistage tasks, providing supervision for subgoal transitions and continuous control.

## 2.2. Game Action Data for Instruction Following

Large-scale gameplay demonstrations support action policy learning [24, 25]. MineRL [8] provides human demonstrations in Minecraft with automatically annotated states and actions. VPT [3] uses inverse dynamics to infer keyboardand-mouse actions from Minecraft videos, while Nitro-Gen [15] extracts gamepad actions from online videos for cross-game behavior cloning.

Language-based supervision further connects gameplay behaviors to task goals. MineDojo [5] combines an openended task suite specified in natural language with an internet-scale multimodal knowledge base of videos, tutorials, wiki pages, and forum discussions. Open-P2P [23] provides cross-game keyboard-and-mouse trajectories with sparse goal-oriented instructions. SIMA [18] learns to follow language instructions across 3D environments, focusing primarily on short-horizon tasks, while SIMA 2 [4] extends this setting to complex instructions and instruction chaining. STEVE-1 [14] supports text- and visualgoal control in Minecraft and composes multi-step behaviors through sequential subgoal prompts.

AceAction organizes demonstrations around complete multi-stage tasks, with annotations spanning overall goals, stage-level subgoals, and local skill instructions. Each subtask is associated with a waypoint, explicit temporal boundaries, and aligned keyboard-and-mouse trajectories. This structure connects individual skill execution to task progression, providing supervision for action execution and the timing of subgoal transitions.

## 3. Dataset

## 3.1. Data Source

Our dataset comprises two complementary components: directly collected gameplay and publicly sourced gameplay. The directly collected gameplay was recorded from full-time in-house staff who held a Gold rank and consented to participate. These recordings were transcribed in anonymized form during data collection. Together, these components provide approximately 5,180 hours of gameplay recordings paired with keyboard-and-mouse actions.

Directly collected gameplay. We collected gameplay recordings and corresponding keyboard-and-mouse actions from human players. This component includes approximately 1,891 hours of Game for Peace, including combat scenarios, and 261.17 hours of Valorant. These recordings pair gameplay observations with the actions performed by human players.

Publicly sourced gameplay. To broaden coverage across games and tasks, we incorporate 2,103.05 hours of CS:GO and CS2 gameplay from match recordings collected online, 811.50 hours of annotated 3D gameplay from Open-P2P, and 113.25 hours of Minecraft gameplay from VPT. These durations refer to the portions included in our dataset rather than the complete upstream datasets.

Action representation and sampling rates. Across both components, we use a common action representation that encodes keyboard and mouse-button states, together with horizontal and vertical mouse displacement. Our directly collected gameplay is recorded at 16 Hz, corresponding to one action step every 62.5 ms. For publicly sourced gameplay, we CS:GO and CS2 action sequences are 16 Hz, while retaining the original 20 Hz sampling rate for Open-P2P and VPT.

## 3.2. Construction Pipeline

We construct hierarchical annotations from gameplay observations and actions recorded from human players (Figure 2). The pipeline first identifies skill-based subgoals and uses visually recognizable waypoints to construct navigation subgoals. Adjacent subgoals are then composed into overall instructions. Throughout this process, source indices preserve the correspondence between semantic annotations and the original image–action sequences.

Skill-based subgoal annotation. We group every four consecutive actions into an annotation unit with a corresponding visual observation. Each trajectory is processed in consecutive windows of up to 32 units. Given the ordered observations and actions, a vision–language model jointly predicts a skill category, start and end indices, and a subgoa description for each segment. The skill category identifies the type of behavior, while the subgoal description specifies its concrete objective. Multiple subgoals can therefore share the same skill category while referring to different targets or objectives.

![](images/f76774bf30f7424759d4da28bdfd2e13ebaad6c1df1290784d2ff82d79bc6f83.jpg)  
Figure 2. Overview of the AceAction data construction pipeline. Gameplay from direct collection and public datasets is annotated with skill- and waypoint-based subgoals. Adjacent subgoals within each trajectory are composed into task-level instructions, followed by annotation and alignment validation. Two models then verify visual grounding and action consistency, retaining only samples approved by both at every annotation level. A sampled human audit further assesses data quality.

Segmentation follows continuity of purpose. An activity remains within one segment while it serves the same objective. A new segment begins when the primary objective or interaction target changes. To preserve this continuity across window boundaries, each subsequent window receives the previous window’s final observation, skill cat egory, and subgoal description. The model also predicts a continuation flag indicating whether its first segment extends the preceding subgoal. We merge the two segments only when this flag is positive, their skill categories match, and their sequence indices are adjacent. This allows a subgoal to span multiple annotation windows. After normalizing category labels, descriptions, and boundaries, we remove residual airborne segments and map the retained subgoals back to their original image–action sequences using source indices.

Waypoint-based subgoal annotation. Navigation subgoals are constructed around arrival events at visually recognizable locations. We identify these events using a sliding window containing up to eight recent observations. The window advances by one observation at each step, and the model evaluates whether the newest frame shows arrival at a specific location. Its input includes the observations, recorded actions, associated skill annotations, previously accepted waypoint names, and the most recent waypoint. Visual observations provide the evidence for arrival, while actions and skill annotations provide behavioral context.

For each accepted arrival event, the model assigns a descriptive waypoint name and associates it with the source frame. Previously accepted names are reused when the same location is recognized again. Continued activity within an already reached location does not produce repeated arrival annotations. Each waypoint thus identifies a destination and the observation at which it is reached.

We then sample clips of varying lengths that end at the waypoint frame. This creates navigation examples with a shared destination but different starting points and intervening actions. For each clip, the model receives sampled keyframes, including the first and arrival frames, together with the complete action sequence, skill context, and target waypoint name. It generates a navigation subgoal that includes the waypoint name and describes the observed route or movement toward that destination. Although annotation uses sampled keyframes, the resulting subgoal retains the complete image–action sequence.

Instruction composition. We combine adjacent subgoals from the same trajectory in execution order to form an overall instruction. The overall instruction summarizes the task defined by this sequence, while the constituent subgoals retain its intermediate objectives. A composed task can include successive subgoals of the same skill or transitions between different skills. For example, navigation subgoals may lead to a destination where the player subsequently performs an interaction.

For each composed sample, we retain the overall instruction, ordered subgoal descriptions, skill categories, segment boundaries, and corresponding image–action sequence. These boundaries identify the observations and actions associated with each intermediate objective, preserving the relationship between the complete task and its execution stages.

Before export, we check that descriptions are non-empty, skill categories are valid, and segment boundaries are ordered and within the source sequence. We also verify image–action correspondence and confirm that each navigation clip ends at its designated waypoint frame. The resulting samples explicitly connect overall instructions, intermediate subgoals, and executable actions.

## 3.3. Quality Control

We apply a two-stage quality-control procedure consisting of model-based verification and human inspection.

Table 1. Comparison of gameplay datasets in instruction annotations and supervision characteristics.
<table><tr><td>Dataset</td><td>Instructions</td><td>Fine-grained hierarchical instruction annotations</td><td>Long instructions</td><td>Remarks</td></tr><tr><td>Open-P2P [23]</td><td>L</td><td>x</td><td>x</td><td>Sparse goal-oriented instructions without an ex- plicit task-subgoal-skill hierarchy.</td></tr><tr><td>EgoCS-400K [7]</td><td>」</td><td>x</td><td>x</td><td>Restricted to Counter-Strike; replay-grounded actions and multi-granularity captions.</td></tr><tr><td>CS2-10k [19]</td><td>x</td><td>x</td><td>x</td><td>Restricted to Counter-Strike 2; frame-aligned actions without task instructions.</td></tr><tr><td>Ours</td><td>L</td><td>1</td><td>L</td><td>Task-subgoal-skill instructions with explicit stage boundaries and aligned keyboard-mouse actions.</td></tr></table>

Model-based verification. We use Qwen3.8-27B [17] and GLM-5.3-FLASH [6] to review each annotated sample against its gameplay clip and recorded actions. Both models perform bottom-up verification across three annotation levels. First, they check whether each action description matches the visible behavior and recorded controls. Second, they assess whether each subgoal accurately describes the objective supported by its associated observations and actions. Finally, they verify that the overall instruction is consistent with the constituent subgoals and the complete trajectory. At every level, the review checks both visual grounding and action consistency. A sample is retained only when both models approve all annotation levels. A rejection at any level by either model leads to its exclusion.

Human inspection. Following model-based verification, eight human reviewers inspect 500 trajectories, with each trajectory assessed by at least two reviewers. The inspection checks whether action descriptions match the observed operations, subgoals reflect the corresponding execution stages, and overall instructions accurately describe the complete tasks. Reviewers assess these annotations against both the gameplay clips and recorded actions. The pass rate across independent review records is 95.8%, where each record represents one reviewer’s assessment of one trajectory.

## 3.4. Compared with other benchmark

Table 1 compares gameplay datasets in terms of instruction annotations and supervision characteristics. Our dataset emphasizes two aspects for training and evaluating game agents, including vision–language–action (VLA) models. First, trajectory segmentation is grounded in both playerview videos and recorded keyboard–mouse actions. Each annotated segment preserves the temporal correspondence between visual observations, language instructions, and control inputs, providing aligned supervision for instruction following and action prediction. Second, our dataset provides extensive subgoal descriptions and second-scale skill annotations within an explicit task–subgoal–skill hierarchy. Overall task instructions are linked to ordered subgoals and action description, with temporal boundaries marking subgoal transitions. These annotations provide structured textual supervision for reasoning about task decomposition, skill composition, and subgoal transitions, supporting both training and evaluation on multi-stage tasks.

## 4. Conclusion

We introduce AceAction-5M, a dataset of human gameplay across multiple 3D games for training and evaluating game agents and vision–language–action (VLA) models. Combining directly collected and publicly sourced gameplay, AceAction links player-view observations and aligned keyboard–mouse actions through a task–subgoal– skill instruction hierarchy. Our annotation pipeline identifies subgoals from changes in objectives or interaction targets, grounds navigation subgoals in visually recognizable waypoints, and composes adjacent subgoals into multistage task instructions. Explicit temporal boundaries connect overall task objectives to local skill execution and mark subgoal transitions. Model-based verification and human inspection assess consistency among language annotations, visual observations, and actions. AceAction provides structured supervision for studying task decomposition, skill composition, and subgoal transitions in dynamic game environments.

## References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and Franc¸ois Fleuret. Diffusion for world modeling: Visual details matter in atari. In Thirty-eighth Conference on Neural Information Processing Systems. 2

[2] Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace B. Connors, James Darpinian, Karan Dhabalia, Jared Di Carlo, Danny Driess, Michael Robert Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Benjamin Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin Leblanc, Sergey Levine, Adrian Li-Bell, Yao Lu, Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Alexander Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, and Zhiyuan Zhou. π<sup>∗</sup><sub>0.6</sub>: a VLA That Learns From Experience. In Proceedings of Robotics: Science and Systems, Sydney, Australia, 2026. 2

[3] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. Advances in Neural Information Processing Systems, 35:24639–24654, 2022. 3

[4] Adrian Bolton, Alexander Lerchner, Alexandra Cordell, Alexandre Moufarek, Andrew Bolt, Andrew Lampinen, Anna Mitenkova, Arne Olav Hallingstad, Bojan Vujatovic, Bonnie Li, et al. Sima 2: A generalist embodied agent for virtual worlds. arXiv preprint arXiv:2512.04797, 2025. 3

[5] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing Systems, 35: 18343–18362, 2022. 3

[6] GLM-5-Team, :, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chengwei Hu, Chenhui Zhang, Dan Zhang, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun,

Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huanpeng Chu, Jia’ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li, Jingwei Yuan, Jinhua Du, Jinxin Liu, Junkai Zhi, Junwen Duan, Kaiyue Zhou, Kangjian Wei, Ke Wang, Keyun Luo, Laiqiang Zhang, Leigang Sha, Liang Xu, Lindong Wu, Lintao Ding, Lu Chen, Minghao Li, Nianyi Lin, Pan Ta, Qiang Zou, Rongjun Song, Ruiqi Yang, Shangqing Tu, Shangtong Yang, Shaoxiang Wu, Shengyan Zhang, Shi jie Li, Shuang Li, Shuyi Fan, Wei Qin, Wei Tian, Weining Zhang, Wenbo Yu, Wenjie Liang, Xiang Kuang, Xiangmeng Cheng, Xiangyang Li, Xiaoquan Yan, Xiaowei Hu, Xiaoying Ling, Xing Fan, Xingye Xia, Xinyuan Zhang, Xinze Zhang, Xirui Pan, Xu Zou, Xunkai Zhang, Yadi Liu, Yandong Wu, Yanfu Li, Yidong Wang, Yifan Zhu, Yijun Tan, Yilin Zhou, Yiming Pan, Ying Zhang, Yinpei Su, Yipeng Geng, Yong Yan, Yonglin Tan, Yuean Bi, Yuhan Shen, Yuhao Yang, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yurong Wu, Yutao Zhang, Yuxi Duan, Yuxuan Zhang, Zezhen Liu, Zhengtao Jiang, Zhenhe Yan, Zheyu Zhang, Zhixiang Wei, Zhuo Chen, Zhuoer Feng, Zijun Yao, Ziwei Chai, Ziyuan Wang, Zuzhou Zhang, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-5: from vibe coding to agentic engineering, 2026. 5

[7] Rongjin Guo, Dong Liang, Yuhao Liu, Fang Liu, Tianyu Huang, Gerhard P Hancke, and Rynson WH Lau. Egocs-400k: An egocentric gameplay dataset for world models. arXiv preprint arXiv:2606.18180, 2026. 2, 3, 5

[8] William H Guss, Brandon Houghton, Nicholay Topin, Phillip Wang, Cayden Codel, Manuela Veloso, and Ruslan Salakhutdinov. Minerl: A large-scale dataset of minecraft demonstrations. arXiv preprint arXiv:1907.13440, 2019. 3

[9] Kang He, Wenshuo Peng, Chuanhao Li, Zihui Gao, Xiaojie Xu, Zhengyuan Lin, Yuzhe Ding, Donghong Ji, Kaipeng Zhang, and Yongtao Ge. Gameverse: A minute-scale gameplay dataset for long-horizon interactive world modeling. In Advances in Neural Information Processing Systems, 2026. 3

[10] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 2

[11] Kailing Li, Yu Han, Tianwen Qian, Yuqian Fu, Jingyu Gong, Jiangming Shi, and Xiaoling Wang. Groundingvln: Reason ing and acting with grounding for vision-language naviga tion, 2026. 2

[12] Kailing Li, Tianwen Qian, Lijin Yang, Yuqian Fu, Jingyu Gong, Xiaoling Wang, and Liang He. Bridging the 2d-3d gap: A hierarchical semantic-geometric map for vision language navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15243–15252, 2026. 2

[13] Zhen Li, Zian Meng, Shuwei Shi, Wenshuo Peng, Yuwei Wu, Bo Zheng, Chuanhao Li, and Kaipeng Zhang. Wild

world: A large-scale dataset for dynamic world modeling with actions and explicit state toward generative arpg. arXiv preprint arXiv:2603.23497, 2026. 2

[14] Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, and Sheila McIlraith. Steve-1: A generative model for text-tobehavior in minecraft. Advances in Neural Information Processing Systems, 36:69900–69929, 2023. 3

[15] Lo¨ıc Magne, Anas Awadalla, Guanzhi Wang, Yinzhen Xu, Joshua Belofsky, Fengyuan Hu, Joohwan Kim, Ludwig Schmidt, Georgia Gkioxari, Jan Kautz, Yisong Yue, Yejin Choi, Yuke Zhu, and Linxi Fan. Nitrogen: An open foundation model for generalist gaming agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21511–21521, 2026. 3

[16] Yanbo Mao, Jianlong Fu, Ruoxuan Zhang, Hongxia Xie, and Meibao Yao. Beyond success: Refining elegant robot manipulation from mixed-quality data via just-in-time intervention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13508– 13518, 2026. 2

[17] Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, 2026. 5

[18] Maria Abi Raad, Arun Ahuja, Catarina Barros, Frederic Besse, Andrew Bolt, Adrian Bolton, Bethanie Brownfield, Gavin Buttimore, Max Cant, Sarah Chakera, et al. Scaling instructable agents across many simulated worlds. arXiv preprint arXiv:2404.10179, 2024. 3

[19] Reka AI. RekaCS2-10k: A large-scale egocentric counterstrike 2 dataset. Online, https://huggingface. co/datasets/RekaAI/CS2-10k, 2026. Blog post. Dataset: https://huggingface.co/datasets/ RekaAI/CS2-10k. 5

[20] SIMA Team, Maria Abi Raad, Arun Ahuja, Catarina Barros, Frederic Besse, Andrew Bolt, Adrian Bolton, Bethanie Brownfield, Gavin Buttimore, Max Cant, et al. Scaling instructable agents across many simulated worlds. arXiv preprint arXiv:2404.10179, 2024. 2

[21] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines, 2024. 2

[22] Yiran Wang, Xingyilang Yin, Junfu Pu, Guangzhi Wang, Kaifeng Li, Mingyu Ouyang, Huiqiang Sun, Lingen Li, Cheng Cheng, Wangbo Yu, et al. Gamehorizon suite: Multihorizon data and evaluation in gameplay. arXiv preprint arXiv:2609.25001, 2026. 2, 3

[23] Yuguang Yue, Irakli Salia, Samuel Hunt, Chris Green, Wenzhe Shi, and Jonathan J Hunt. Scaling behavior cloning improves causal reasoning: An open model for real-time video game playing. arXiv preprint arXiv:2601.04575, 2026. 2, 3, 5

[24] RUOXUAN ZHANG, ZIQI LIAO, ZHIYU ZHOU, JIAN-YU JIANG-LIN, SIYU WU, YANBO MAO, BIN WEN, HONGXIA XIE, JIANLONG FU, MEIBAO YAO, et al. Social world models: A comprehensive survey of theory of mind and social coordination. ACM Comput. Surv, 1(1), 2026. 3

[25] Zheyuan Zhou, Liang Du, Zixun Sun, Xiaoyu Zhou, Ruimin Ye, Qihao Chen, Yinda Chen, and Lemiao Qiu.

Main-vla: Modeling abstraction of intention and environment for vision-language-action models. arXiv preprint arXiv:2602.02212, 2026. 3