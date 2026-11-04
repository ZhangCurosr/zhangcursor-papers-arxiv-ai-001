# Catalyzing Interactive HTML Generation Capabilities via High-Fidelity Reinforcement Learning

Xujia Wang<sup>1,2,\*</sup> Yixuan Liu<sup>1,\*</sup> Shaohan Huang<sup>2,†</sup> Furu Wei<sup>2</sup> Lei Hou<sup>1</sup> Bin Xu<sup>1,†</sup> Juanzi Li<sup>1</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Microsoft Research wang-xj22@mails.tsinghua.edu.cn Equal contribution. <sup>†</sup>Corresponding authors.

## Abstract

Generating HTML from natural-language specifications is an important application of code large language models (LLMs). Despite producing renderable pages, current models often fail to ensure functionality and visual quality during actual interaction. Existing methods typically target task-specific webpage supervision, leaving demanding interactive artifact optimization underexplored. Using minigames as a challenging testbed, we expose limitations in interaction experience, reward signal, and credit assignment fidelity that are obscured by shallow webpage interactions. We introduce WebCatalyst, a browser-interaction reinforcement learning framework with three complementary components: a game-level GUI assessor for sustained interaction and evaluation, Visual Anchor for mitigating VLM reward hacking, and Action-Trace Credit for ground ing token-level credit assignment in runtime execution traces. Trained on fewer than 1K requests, predominantly for games, WebCatalyst raises the Cookie-Bench Mini Games score from 68.8 to 75.3 and also the WebGen-Bench accuracy from 48.8 to 56.9 on Qwen3.6-35B-A3B, outperforming Claude Opus 4.7 in average rank across evaluated metrics. Further analysis shows that Visual Anchor prevents aesthetic degradation, while runtime-grounded credit assignment accelerates functional improvement. Together, our results show that improving feedback fidelity is the key to substantially advancing interactive HTML generation.

## 1 Introduction

With advances in code generation, large language models (LLMs) are widely used to generate HTML webpages from natural-language requests or visual designs (Si et al., 2025; Laurençon et al., 2024; Yun et al., 2024; Gui et al., 2025; Lu et al., 2025). Although large-scale pretraining and supervised fine-tuning enable models to produce syntactically valid HTML (Laurençon et al., 2024; Yun et al.,

![](images/d32fb900494df9362d835b50ce6aca738754a7af2b2c53294269207075fd14bb.jpg)  
Figure 1: From task-specific webpage optimization to interactive HTML capability catalysis. WebCatalyst improves fidelity in interaction experience, reward signals, and credit assignment through a Game-Level GUI Assessor, Visual Anchor, and Action-Trace Credit.

2024; Gui et al., 2025), successful rendering does not guarantee usability: page quality depends on how it looks and behaves during interaction (Xiao et al., 2025; Lu et al., 2025; Zhang et al., 2025).

To incorporate these aspects of usability into training, recent approaches developed reinforcement learning (RL) supervision for webpage generation, including executable rubrics (Jin et al., 2026), iterative-refinement feedback (Lu et al., 2026; Li et al., 2026b; Deng et al., 2026), GUI-agent functional checks (Lu et al., 2026), visual-quality rewards (Lai et al., 2025; Li et al., 2026b; Jiang et al., 2026), and evolving graders (Chen et al., 2026). While these methods demonstrate the value of taskaligned feedback, their supervision is often tailored to specific webpage tasks and evaluation protocols.

Thus, we seek to move beyond task-specific webpage optimization toward catalyzing more demanding interactive HTML capabilities. This raises a central question: What design principles enable effective and more generalizable RL supervision for interactive HTML generation? Alongside conventional webpages with basic click-and-type interactions, we study mini-games as a challenging testbed, whose sustained interactions and dynamic visuals expose fidelity limitations in interaction experience, reward signals, and credit assignment.

To this end, we introduce WebCatalyst, a browser-interaction RL framework for improving the interactive HTML generation capabilities of LLMs. Its high-fidelity supervision addresses three coupled challenges in translating browser interactions into effective learning signals. First, evaluating interactive HTML requires evidence from actual interaction (Zhang et al., 2026b; Yang et al., 2026). Our GUI Assessor (§3.1) conducts sustained, game-level browser interactions and hierarchically aggregates action outcomes and screenshot-based evaluations into a page-level reward. Second, we identify a reward-hacking failure mode in GUI-agent evaluation: VLM judges may overlook visual degradation when functionality improves. To mitigate this bias, the Visual Anchor (§3.2) penalizes degradation relative to reference screenshots. Third, coarse page-level rewards provide little guidance about which code tokens contribute to individual behaviors in complex interactive environments (Dou et al., 2024; Kumar et al., 2026; Jin et al., 2026). We therefore introduce Action-Trace Credit (§3.3), a runtime-grounded credit-assignment method that captures call stacks for successful actions, maps executed code spans back to generated tokens, and redistributes advantages according to call depth. These components improve interaction experience, reward signal, and credit assignment fidelity enabling more effective training for complex interactive HTML generation.

With fewer than 1K training requests, most involving games with complex interactions, Web-Catalyst improves the Cookie-Bench Mini Games overall score from 68.8 to 75.3 and WebGen-Bench accuracy from 48.8 to 56.9 on Qwen3.6-35B-A3B. It also achieves a better average rank than Claude Opus 4.7 across webpage and mini-game evaluations. Further analysis reveals that GUI feedback alone improves functionality at the cost of visual quality. Visual Anchor mitigates this degradation, while Action-Trace Credit accelerates gains on functional metrics. By exposing limitations masked by shallow webpage interactions, these results show how high-fidelity feedback can move RL beyond task-specific optimization toward more capable interactive HTML generation.

Our contributions are:

• We study interactive HTML generation beyond task-specific webpage optimization, using challenging mini-games to reveal three key dimensions of RL supervision fidelity: interaction experience, reward signals, and credit assignment.

• We introduce WebCatalyst, a unified browserinteraction RL framework that addresses these limitations through three complementary components: a Game-Level GUI Assessor, Visual Anchor, and Action-Trace Credit.

• With fewer than 1K training requests, WebCatalyst improves both game and webpage generation, enabling Qwen3.6-35B-A3B to achieve a better average rank than Claude Opus 4.7 across evaluated metrics.

## 2 Related Work

## 2.1 Assessing Interactive HTML Artifacts

Evaluating interactive HTML requires assessing both functional behavior and visual quality during use. GUI agents provide grounded mouse-andkeyboard interaction (Cheng et al., 2024; He et al., 2024; Qin et al., 2025), but existing browser and desktop environments mainly evaluate agents in pre-existing applications (Zhou et al., 2024; Drouin et al., 2024; Xie et al., 2024). For generated HTML, interaction-aware benchmarks evaluate behavior, functionality, and visual quality (Xiao et al., 2025; Lu et al., 2025; Zhang et al., 2025). Recent exploratory evaluators gather visual and behavioral evidence over interaction trajectories (Zhang et al., 2026b; Yang et al., 2026), while game-oriented evaluators assess runtime gameplay and sometimes support iterative code refinement (Zhang et al., 2026a; Huang et al., 2026; Peng et al., 2026). These works use interactive assessors primarily for benchmarking or inference-time refinement, rather than as feedback providers for RL.

## 2.2 Training for HTML Generation

HTML generation covers design reconstruction and web application generation from naturallanguage requests. Supervised approaches learn from screenshot–code pairs (Beltramelli, 2017; Laurençon et al., 2024; Yun et al., 2024; Gui et al., 2025). Browser-guided refinement also supplies SFT data (Lu et al., 2026; Wu et al., 2026). Recent RL methods primarily target webpage generation, using feedback aligning with task-specific evaluation protocols (Lai et al., 2025; Li et al., 2026b;

Jiang et al., 2026; Lu et al., 2026; Chen et al., 2026), whereas we focus on RL supervision fidelity in sustained HTML interaction.

## 2.3 Code Credit Assignment

Sequence-level rewards obscure the contributions of individual code regions. Prior work localizes feedback using learned reward models (Le et al., 2022; Fan et al., 2026), compiler-reported errors (Liu et al., 2023), execution coverage (Dou et al., 2024; Kumar et al., 2026), or a single related code span (Li et al., 2026a; Jin et al., 2026). Our Action-Trace Credit enables precise behavior-to-code reward assignment by tracing each successful GUI action to its runtime code path.

## 3 Method

To move beyond task-specific webpage training toward catalyzing transferable capabilities for more demanding interactive HTML generation, we integrate interactive execution, reference calibration, and runtime-grounded credit assignment in one training framework (Figure 2). Given a requirement $x ,$ we sample $G$ responses $\begin{array} { r l } { y _ { i } } & { { } = } \end{array}$ $( y _ { i , 1 } , \dots , y _ { i , N _ { i } } )$ from $\pi _ { \mathrm { o l d } }$ and render them as HTML pages $h _ { i }$ . GUI Assessor produces interaction trajectories $\tau _ { i }$ and page scores $R _ { i } ^ { \mathrm { G U I } }$ , improving experience fidelity. Visual Anchor estimates visual degradation $D _ { i }$ and calibrates page scores into rewards $r _ { i }$ , improving reward signal fidelity. Action-Trace Credit distributes group-relative advantages $A _ { i }$ into token-level advantages $\widehat { A } _ { i , k }$ , improving credit-assignment fidelity. The policy π<sub>θ</sub> is then updated using GRPO (Shao et al., 2024).

## 3.1 Experience-Grounded Execution and GUI Assessment

Short click-and-type tests in existing webpagespecific training cannot support sustained, multistep interaction and state transitions in interactive HTML artifacts such as games, requiring a rich action space and sufficient interaction depth. To address this, we model $h _ { i }$ as a state machine executed in a browser. An Execution Agent runs a ReAct loop (Yao et al., 2023) conditioned on requirement $x ,$ current screenshot $S _ { i , t } ,$ prior actions and execution observation. It selects ${ a } _ { i , t }$ from

$$
\begin{array} { r } { \mathcal { A } = \{ \mathrm { c l i c k } , \mathsf { h o v e r } , \mathsf { m o u s e M o v e } , \mathsf { p r e s s } , } \\ { \mathsf { c o m b o } , \mathrm { t y p e } , \mathsf { w a i t } , \mathsf { t e r m i n a t e } \} . } \end{array}\tag{1}
$$

Action parameters include coordinates, key combinations, and durations, driving the page from its

initial state $V _ { i , 0 }$ to $V _ { i , T _ { i } }$

$$
\begin{array} { r l } & { V _ { i , t + 1 } = \mathcal { T } _ { h _ { i } } ( V _ { i , t } , a _ { i , t } ) , } \\ & { \quad S _ { i , t } = \mathcal { O } ( V _ { i , t } ) , } \\ & { \quad E _ { i , t } = ( S _ { i , t } , a _ { i , t } , S _ { i , t + 1 } , m _ { i , t } ) , } \\ & { \quad \quad \tau _ { i } = \{ E _ { i , t } \} _ { t = 0 } ^ { T _ { i } - 1 } . } \end{array}\tag{2}
$$

Here, $S _ { i , t }$ is a screenshot, and $m _ { i , t }$ is the browser execution record. Interaction ends when the agent terminates or exhausts its budget.

Hierarchical Judge. Local, single-step judgments cannot capture overall quality across sustained interaction. We therefore adopt a two-level hierarchy consisting of step-level assessment and trajectory-level aggregation. For each transition, the Step-level Judge uses before-and-after screenshots, the action, and the execution record to assess post-action screenshot $\boldsymbol { S _ { i , t + 1 } } ^ { } \boldsymbol { \mathbf { \dot { s } } }$ visual quality and the resulting behavior, producing a screenshot score $q _ { i , t } ^ { \mathrm { s c } }$ and a transition score $q _ { i , t } ^ { \mathrm { t r } }$ . The Meta-level Judge eventually combines memory with all steplevel assessments to evaluate functional behavior and requirement fulfillment across sustained interactions, yielding page-level supervision feedback ranging from 0 to 100:

$$
R _ { i } ^ { \mathrm { G U I } } = \mathcal { I } _ { \mathrm { m e t a } } \left( x , \{ \mathcal { I } _ { \mathrm { s t e p } } ( E _ { i , t } ) \} _ { t = 0 } ^ { T _ { i } - 1 } \right)\tag{3}
$$

This hierarchy aggregates individual-action evidence to assess overall user experience. Detailed scoring criteria appear in the appendix A.1.

Producer–Consumer Queue. Sustained interaction makes sequential execution and assessment prohibitively slow. After each action, the Execution Agent enqueues the resulting transition $E _ { i , t }$ and immediately proceeds to the next step, while the Step-level Judge evaluates queued transitions concurrently. Once execution terminates, we collect all step-level assessments and pass them to the Meta-level Judge for aggregation. This producer– consumer pipeline overlaps interaction with assessment and reduces evaluation latency.

## 3.2 Reference-Calibrated Visual Anchor

When evaluating HTML artifacts involving richer interactions, we observe that relying solely on a VLM-based GUI judge creates a reward loophole: the policy can improve interaction scores by simplifying visual structure or omitting diverse features. As illustrated in Figure 2, such artifacts may preserve a few easily verified interactions while degrading visual dimensions to which the judge is insensitive, yielding high-reward yet visually inferior outputs. Thus, we introduce Visual Anchor, which compares the generated artifact’s screenshot $S _ { \mathrm { L M } }$ against a well-organized reference screenshot $S _ { \mathrm { S O T A } }$ to estimate one-sided visual degradation:

![](images/1e542d612ea7802fabac3d2d7e493aa4bc993ac29c4129e057d63198ca7b8849.jpg)  
Figure 2: Overview of WebCatalyst’s browser-interaction RL framework. Training requests come from Reddit, and sampled HTML artifacts are execute in Playwright. The game-level GUI Assessor combines sustained interaction with hierarchical judgment for page-level rewards. Visual Anchor penalizes visual degradation relative to reference screenshots, while Action-Trace Credit maps successful actions’ runtime call paths to generated tokens and redistributes fine-grained advantages. GRPO optimizes the generation policy using the calibrated rewards and token-level advantages. Gameplay examples illustrate functionality and visual quality before and after training.

$$
D _ { i } = \mathcal { D } _ { \mathrm { v i s u a l } } ( S _ { \mathrm { S O T A } } , S _ { \mathrm { L M } } ; x ) \in [ 0 , 1 0 0 ] .\tag{4}
$$

Reference Construction and Calibration. To provide a stable visual baseline for otherwise reference-free assessment, a capable model generates reference HTML pages before training, which the GUI Assessor executes. For both generated and reference pages, we select the highest-scoring screenshot to avoid unrepresentative states such as startup or failure screens:

$$
\begin{array} { r l r } & { t _ { i } ^ { \star } = \arg \operatorname* { m a x } _ { t } q _ { i , t } ^ { \mathrm { s c } } , } & { S _ { \mathrm { L M } } = S _ { i , t _ { i } ^ { \star } + 1 } , } \\ & { t _ { x } ^ { \star } = \arg \operatorname* { m a x } _ { t } q _ { x , t } ^ { \mathrm { s c } , \mathrm { r e f } } , } & { S _ { \mathrm { S O T A } } = S _ { x , t _ { x } ^ { \star } + 1 } ^ { \mathrm { r e f } } . } \end{array}\tag{5}
$$

The pairwise visual judge evaluates degradation in layout, visual hierarchy, and design completeness.

It assigns $D _ { i } = 0$ when the generated page is comparable to or better than the reference, and applies a penalty to the reward when degradation is clear.

## 3.3 Runtime-Grounded Credit Assignment

Even with calibrated page-level rewards, optimizers such as GRPO uniformly assign advantage $A _ { i }$ to all generated tokens, obscuring which code regions contribute to individual interactive behaviors. Directly penalizing failed actions can cause misattribution: a failure may arise from incorrect or missing logic, whose responsible code can be ambiguous or absent from the runtime trace. Thinking in reverse, we derive Action-Trace Credit (ATC) from the runtime call paths of successful actions.

Runtime Code Tracing. To identify the code responsible for individual interactions, we capture JavaScript call stacks via the Chrome DevTools Protocol during traceable actions, retaining only traces from actions classified as successful by the Step-level Judge. Each stack contains the active leaf frame followed by its callers. Following this leaf-to-caller order, we assign depth 0 to the leaf and progressively larger depths to its callers. We map locatable inline JavaScript function spans to generated tokens. For each token k, set $\mathcal { F } _ { i , k }$ records the traced depth of the functions involving it.

Minimum Call Depth and Depth Decay. A token may occur in multiple traced functions and call paths from different actions. We take the minimum call depth $d _ { i , k } ^ { \operatorname* { m i n } }$ across its traces and modulate credit by $\gamma ^ { d _ { i , k } ^ { \operatorname* { m i n } } }$ , analogous to discounting in Proximal Policy Optimization (Schulman et al., 2017).

$$
\begin{array} { r } { d _ { i , k } ^ { \operatorname* { m i n } } = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m i n } _ { ( f , d ) \in \mathcal { F } _ { i , k } } d , } & { \mathcal { F } _ { i , k } \neq \emptyset , } \\ { + \infty , } & { \mathcal { F } _ { i , k } = \emptyset } \end{array} \right. } \\ { c _ { i , k } = \gamma ^ { d _ { i , k } ^ { \operatorname* { m i n } } } } \end{array}\tag{6}
$$

where $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ . Under the implemented leafto-caller convention, leaf-aligned tokens, which typically executing core action logic and state updates, receive the largest trace coefficient, decaying toward outer callers such as event handlers. We adopt $\gamma ^ { + \infty } = 0$ , preserving baseline weights for tokens outside successful actions’ traces. Using the sign of page-level advantage, we define the modulation weight and token-level advantage as

$$
\begin{array} { r l } & { \widetilde { w } _ { i , k } = [ 1 + \beta \ \mathrm { s g n } ( A _ { i } ) c _ { i , k } ] _ { + } , } \\ & { \widehat { A } _ { i , k } = \frac { A _ { i } \widetilde { w } _ { i , k } } { \frac { 1 } { N _ { i } } \sum _ { u = 1 } ^ { N _ { i } } \widetilde { w } _ { i , u } } . } \end{array}\tag{7}
$$

Here, $[ z ] _ { + } = \operatorname* { m a x } ( z , 0 )$ , and $0 \leq \beta < 1$ controls modulation strength. Relative to untraced tokens, tokens involved in successful actions receive stronger positive updates when $A _ { i } > 0$ and weaker negative updates when $A _ { i } < 0$ . The normalization preserves the mean token-level advantage at $A _ { i }$ See appendix for further details.

## 3.4 Joint Reinforcement Learning Objective

We integrate experience assessment, visual calibration, and runtime-grounded credit assignment into a unified GRPO objective. For each candidate i, Visual Anchor calibrates the GUI Assessor score into the final reward, from which we compute advantages within each generated-candidate group:

$$
\begin{array} { r l } & { { \boldsymbol r } _ { i } = \frac { R _ { i } ^ { \mathrm { G U I } } - D _ { i } } { 1 0 0 } , } \\ & { { \boldsymbol A } _ { i } = \frac { { \boldsymbol r } _ { i } - \operatorname* { m e a n } _ { j } ( { \boldsymbol r } _ { j } ) } { \operatorname* { s t d } _ { j } ( { \boldsymbol r } _ { j } ) + \epsilon } . } \end{array}\tag{8}
$$

With ATC-derived token-level advantages $\widehat { A } _ { i , k }$ , we maximize the following training objective

$$
\begin{array} { l } { \displaystyle \mathcal { I } ( \theta ) = \mathbb { E } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { N _ { i } } \sum _ { k = 1 } ^ { N _ { i } } \Bigl ( \operatorname* { m i n } \bigl \{ \rho _ { i , k } \widehat { A } _ { i , k } , } \\ { \mathrm { c l i p } ( \rho _ { i , k } , 1 - \varepsilon _ { c } , 1 + \varepsilon _ { c } ) \widehat { A } _ { i , k } \bigr \} - \eta \mathrm { K L } _ { i , k } \Bigr ) \Bigg ] . } \end{array}\tag{9}
$$

where

$$
\rho _ { i , k } = { \frac { \pi _ { \theta } ( y _ { i , k } \mid x , y _ { i , < k } ) } { \pi _ { \mathrm { o l d } } ( y _ { i , k } \mid x , y _ { i , < k } ) } } .\tag{10}
$$

The expectation is taken over requirements and candidate responses. $\mathrm { K L } _ { i , k }$ estimates token-level divergence between the current policy $\pi _ { \theta }$ and fixed reference policy $\pi _ { 0 } ,$ where $\eta$ is the KL coefficient.

## 4 Experiments

## 4.1 Experimental Setup

Benchmarks. We evaluate HTML generation at different levels of interaction complexity using WebGen-Bench (Lu et al., 2025) and the Mini Games split of Cookie-Bench (Yang et al., 2026). WebGen-Bench contains 101 webpage-generation requests and 647 test cases, each specifying an operation and expected outcome to verify a functional or presentation requirement. For Cookie-Bench Mini Games, we evaluate 156 deduplicated requests. The benchmark uses UI-TARS (Qin et al., 2025) to continuously interact with browser, producing visual and behavioral traces used to assess aesthetics and functionality which tests more demanding interactive HTML generation.

Metrics. For WebGen-Bench, an evaluation agent executes each test case and compares the observed behavior with its expected outcome, assigning a verdict of YES, PARTIAL, or NO. These verdicts receive scores of 1, 0.5, and 0, respectively, and Accuracy is their mean across all test cases. We also report Accuracy by category, Content Presentation, User Interaction, Data Management, Functional Testing, Data Display Testing, and Design Validation Testing. Start Failure Rate is the percentage of generated websites that fail to launch. Lower is better.

For Cookie-Bench Mini Games, the evaluator scores each artifact’s visual quality (Aesthetics) and its behavior robustness over sustained interaction (Functionality). We average request-level scores and report them on a 0–100 scale. Total aggregates performance across both dimensions.

Table 1: Main results on WebGen-Bench and Cookie-Bench Mini Games. Acc., Aesth., and Func. denote accuracy, aesthetics, and functionality. Size gives amount of total/active parameters, Avg. Rank averages ranks across evaluation dimensions, and ∆ gives changes from Qwen3.6-35B-A3B. Arrows indicate preferred directions. Bold marks group-wise best results, with WebCatalyst included in Other LLMs. WebCatalyst, with only 640 game development requests, improves both game and webpage generation and surpasses Claude Opus 4.7 in average rank.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Date</td><td rowspan="2">Size</td><td colspan="7">WebGen-Bench User</td><td colspan="3">Cookie-Bench Mini Games</td><td rowspan="2">Avg. Rank↓</td></tr><tr><td>Start Fail. Rate ↓</td><td>Acc. ↑</td><td>Content Pres. ↑</td><td>Data Inter. ↑ Mgmt. ↑</td><td>Func. Test ↑</td><td>Data Display ↑</td><td>Design Val. ↑</td><td>Aesth. ↑ Func. ↑</td><td></td><td>Total ↑</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-OSS-120B</td><td>25/08</td><td>116B/5B</td><td>1.4</td><td>37.6</td><td>49.7</td><td>Other LLMs 31.3</td><td>36.6</td><td>26.0</td><td>43.0</td><td>61.5</td><td>40.3</td><td>54.4</td><td>47.4</td><td>19.5</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>26/02</td><td>122B/10B</td><td>2.6</td><td>49.8</td><td>57.5</td><td>46.0</td><td>49.1</td><td>37.0</td><td>55.9</td><td>76.2</td><td>66.0</td><td>60.5</td><td>63.2</td><td>16.0</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>26/02</td><td>397B/17B</td><td>0.5</td><td>48.7</td><td>59.2</td><td>43.5</td><td>47.5</td><td>33.6</td><td>57.0</td><td>77.9</td><td>69.9</td><td>65.0</td><td>67.5</td><td>14.1</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>26/04</td><td>35B/3B</td><td>0.0</td><td>48.8</td><td>59.2</td><td>43.0</td><td>47.1</td><td>33.9</td><td>61.8</td><td>70.5</td><td>81.5</td><td>56.1</td><td>68.8</td><td>14.0</td></tr><tr><td>Qwen3.6-Plus</td><td>26/04</td><td>N/A</td><td>0.0</td><td>53.6</td><td>61.5</td><td>50.0</td><td>52.2</td><td>41.4</td><td>65.9</td><td>68.9</td><td>84.8</td><td>63.6</td><td>74.2</td><td>8.5</td></tr><tr><td>Qwen3.6-Max</td><td>26/04</td><td>N/A</td><td>0.0</td><td>55.4</td><td>64.7</td><td>51.1</td><td>53.8</td><td>45.3</td><td>64.0</td><td>70.5</td><td>87.8</td><td>64.4</td><td>76.1</td><td>5.9</td></tr><tr><td>Qwen3.7-Flash</td><td>26/07</td><td>N/A</td><td>1.1</td><td>48.5</td><td>54.6</td><td>45.2</td><td>48.1</td><td>33.6</td><td>65.1</td><td>64.3</td><td>84.0</td><td>54.2</td><td>69.1</td><td>15.5</td></tr><tr><td>GLM-5.1</td><td>26/04</td><td>754B/40B</td><td>0.0</td><td>54.0</td><td>63.2</td><td>47.4</td><td>56.9</td><td>39.4</td><td>64.5</td><td>78.7</td><td>87.8</td><td>61.2</td><td>74.5</td><td>6.9</td></tr><tr><td>GLM-5.2</td><td>26/06</td><td>754B/40B</td><td>0.0</td><td>57.5</td><td>62.1</td><td>52.4</td><td>62.5</td><td>45.1</td><td>68.6</td><td>73.4</td><td>87.2</td><td>65.3</td><td>76.3</td><td>4.3</td></tr><tr><td>Minimax-M2.7</td><td>26/03</td><td>229B/10B</td><td>0.9</td><td>53.4</td><td>62.4</td><td>49.0</td><td>52.2</td><td>38.8</td><td>66.4</td><td>74.2</td><td>81.4</td><td>60.3</td><td>70.8</td><td>10.3</td></tr><tr><td>DeepSeek-V3.2</td><td>25/12</td><td>671B/37B</td><td>0.0</td><td>49.7</td><td>56.0</td><td>45.2</td><td>51.6</td><td>35.1</td><td>60.2</td><td>74.2</td><td>69.0</td><td>56.5</td><td>62.8</td><td>13.9</td></tr><tr><td>DeepSeek-V4-Flash</td><td>26/07</td><td>284B/13B</td><td>2.6</td><td>51.3</td><td>58.0</td><td>45.4</td><td>55.6</td><td>37.9</td><td>58.3</td><td>77.9</td><td>72.2</td><td>57.9</td><td>65.1</td><td>13.3</td></tr><tr><td>Kimi-K2.6</td><td>26/04</td><td>1T/32B</td><td>0.9</td><td>54.9</td><td>68.4</td><td>49.8</td><td>50.3</td><td>42.2</td><td>66.7</td><td>72.5</td><td>83.5</td><td>58.7</td><td>68.6</td><td>9.3</td></tr><tr><td colspan="10">Proprietary LLMs</td><td colspan="3"></td><td></td><td></td></tr><tr><td>Gemini-3.1-Pro</td><td>26/02</td><td>è</td><td>0.0</td><td>53.2</td><td>60.9</td><td>50.2</td><td>50.6</td><td>39.1</td><td>65.6</td><td>73.4</td><td>62.9</td><td>60.9</td><td>61.9</td><td>11.1</td></tr><tr><td>GPT-5.2</td><td>25/12</td><td>a</td><td>2.9</td><td>50.5</td><td>65.2</td><td>47.8</td><td>39.7</td><td>36.4</td><td>62.9</td><td>70.5</td><td>85.0</td><td>63.7</td><td>74.3</td><td>12.3</td></tr><tr><td>GPT-5.3-codex</td><td>26/02</td><td></td><td>0.0</td><td>54.4</td><td>65.8</td><td>52.4</td><td>45.9</td><td>40.9</td><td>65.9</td><td>74.6</td><td>79.4</td><td>69.6</td><td>74.5</td><td>7.2</td></tr><tr><td>GPT-5.4-mini</td><td>26/03</td><td></td><td>0.0</td><td>52.6</td><td>62.6</td><td>47.6</td><td>51.2</td><td>37.6</td><td>61.6</td><td>80.3</td><td>80.8</td><td>64.1</td><td>72.4</td><td>9.4</td></tr><tr><td>Claude-Opus-4.7</td><td>26/04</td><td>è</td><td>0.0</td><td>57.0</td><td>68.4</td><td>48.1</td><td>62.2</td><td>41.6</td><td>72.8</td><td>75.8</td><td>80.7</td><td>69.5</td><td>75.1</td><td>4.7</td></tr><tr><td>GPT-5.5</td><td>26/04</td><td>è</td><td>0.0</td><td>59.3</td><td>66.1</td><td>54.6</td><td>60.9</td><td>45.7</td><td>70.2</td><td>80.3</td><td>86.4</td><td>66.9</td><td>76.7</td><td>2.1</td></tr><tr><td colspan="10"></td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>WebCatalyst</td><td></td><td>26/04 35B/3B</td><td>0.0</td><td>56.9</td><td>68.1</td><td>Ours 52.9</td><td>54.2</td><td>42.2</td><td>68.8</td><td>77.9</td><td>87.5</td><td>63.2</td><td>75.3</td><td>4.0</td></tr><tr><td>∆ from Base</td><td></td><td></td><td>0.0</td><td>+8.1</td><td>+8.9</td><td>+9.9</td><td>+7.1</td><td>+8.3</td><td>+7.0</td><td>+7.4</td><td>+5.9</td><td>+7.0</td><td>+6.5</td><td>-10.0</td></tr></table>

Implementation Details. We construct the training set by collecting Reddit posts about HTML development, filtering them with game-related keywords, and rewriting them as direct user requests. This allows us to test whether interactive HTML training with high-fidelity feedback transfers to simpler downstream tasks such as conventional webpages. We train Qwen3.6-35B-A3B (Qwen Team, 2026b) on NVIDIA GB200 GPUs with a batch size of 16 and group size $G = 5$ . For ATC, we set $\beta = 0 . 5$ and the decay factor $\gamma = 0 . 9$ . We use GPT-5.6 Terra as the judging model in the GUI Assessor, allowing up to eight interaction turns per page. We use GPT-5.5 to generate Visual Anchor’s reference artifacts. We evaluate the checkpoint after 40 training steps, corresponding to 640 processed requests. Further implementation details are provided in the Appendix A.3.

## 4.2 Main Results

Table 1 compares WebCatalyst on Cookie-Bench Mini Games and WebGen-Bench, representing complex interactive artifacts and conventional webpages, respectively. Because our training requests are collected using game-related keywords, Mini Games provides a task-aligned but demanding evaluation. WebGen-Bench instead uses rubric-based test cases for conventional webpages, providing a cross-setting evaluation of whether capabilities learned from interactive game generation transfer to broader webpage tasks.

First, high-fidelity feedback enables competitive mini-game generation. Using 640 training requests, WebCatalyst raises the Mini Games Total score of Qwen3.6-35B-A3B from 68.8 to 75.3. The improvement spans both evaluation dimensions: Aesthetics increases from 81.5 to 87.5, and Functionality from 56.1 to 63.2. Despite using a relatively compact backbone with 35B total parameters and 3B activated per token, WebCatalyst outperforms several contemporary frontier models, including MiniMax-M2.7, GLM-5.1, and Kimi-K2.6, and matches Claude Opus 4.7 on the Total metric.

Second, the gains transfer to conventional webpages with simpler operations and design requirements. Despite being trained primarily on game-focused requests, WebCatalyst improves overall WebGen-Bench Accuracy from 48.8 to 56.9, with improvements across all request- and test-case categories. Notable gains appear in User Interaction (43.0 to 52.9), Content Presentation (59.2 to 68.1), and Functional Testing (33.9 to 42.2), indicating that the transfer covers both executable behavior and visual presentation. Its overall Accuracy exceeds Kimi-K2.6 (54.9) and nearly matches Claude Opus 4.7 (57.0). These results suggest that high-fidelity training on complex interactive artifacts develops capabilities that generalize beyond game tasks to webpage generation.

Table 2: Ablation study of components in WebCatalyst. Results show the contributions of experience, rewardsignal, and credit-assignment fidelity.
<table><tr><td rowspan="2">Variant</td><td>WebGen</td><td colspan="3">Mini-Games</td></tr><tr><td>Acc.</td><td>Aesth.</td><td>Func.</td><td>Total</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>48.8</td><td>81.5</td><td>56.1</td><td>68.8</td></tr><tr><td>+GUI Assessor</td><td>52.2</td><td>71.3</td><td>60.5</td><td>65.9</td></tr><tr><td>+Self Anchor</td><td>55.5</td><td>82.7</td><td>57.3</td><td>70.0</td></tr><tr><td>+GPT-5.5 Anchor</td><td>56.8</td><td>82.3</td><td>58.2</td><td>70.3</td></tr><tr><td>+ATC</td><td>56.9</td><td>87.5</td><td>63.2</td><td>75.3</td></tr></table>

To aggregate performance across the heterogeneous metrics of both benchmarks, we compute each model’s mean rank, reported as Average Rank. WebCatalyst achieves an Average Rank of 4.0, ranking first among the evaluated open-weight models and ahead of most proprietary models, including Claude Opus 4.7 (4.7). Among all evaluated models, only GPT-5.5 ranks higher (2.1).

Overall, high-fidelity reinforcement learning supervision with only 640 game-focused requests enables WebCatalyst to achieve competitive crossbenchmark performance, transferring gains from mini-games to conventional webpages and narrowing the gap to leading proprietary models.

## 4.3 Ablation Study

Table 2 isolates the complementary contributions of the three feedback-fidelity components. Adding the GUI Assessor increases WebGen Accuracy from 48.8 to 52.2 and Mini Games Functionality from 56.1 to 60.5. However, Aesthetics drops from 81.5 to 71.3, reducing Total from 68.8 to 65.9. This trade-off shows that unanchored GUI feedback can reward functional improvements while overlooking visual degradation, exposing the reward loophole.

Visual Anchor mitigates this degradation. Self Anchor uses screenshots from the base model’s pre-RL outputs as references, whereas GPT-5.5 Anchor uses screenshots from GPT-5.5-generated artifacts. Both variants restore Aesthetics above the backbone level and increase Total from 65.9 to 70.0 and 70.3, respectively, while further improving WebGen Accuracy to 55.5 and 56.8. These results show that reference-based calibration discourages visual simplification while preserving functional gains. Both Visual Anchor variants mitigate reward-induced visual degradation, while further analysis suggests that higher-quality anchors help maintain training stability (§4.4.1).

Finally, adding the action-trace credit raises Mini

Games aesthetics, functionality, and Total from 82.3, 58.2, and 70.3 to 87.5, 63.2, and 75.3, while preserving WebGen accuracy at 56.9. Thus, Visual Anchor and ATC play complementary roles: the former calibrates what makes a high-quality artifact, whereas the latter assigns the resulting feedback more precisely to responsible tokens.

## 4.4 Analysis

## 4.4.1 Looking into Training Dynamics

![](images/46e6116f7d90dfa458e2eca998026cb16ea3c57c2abb39b4f65127f70ce1f68e.jpg)  
Figure 3: Evaluation performance across training steps and feedback configurations. The top two rows show WebGen-Bench performance, while the bottom row shows Mini Games aesthetics, functionality, and overall scores. Anchor denotes Visual Anchor, and ATC denotes Action-Trace Credit.

WebGen Gains Can Mask Feedback-Fidelity Failures. Figure 3 illustrates why improvements on WebGen-Bench alone do not establish feedback fidelity across interaction regimes. When training with only the GUI Assessor, several WebGen metrics improve steadily, whereas Mini Games Aesthetics deteriorates even as Functionality increases. Conventional webpage evaluations can therefore capture improvements in local operation success while masking visual and feature simplification under sustained interaction. The results show that WebGen gains alone do not demonstrate that the supervision faithfully reflects the quality of more demanding interactive artifacts.

Higher-Quality Visual References Stabilize Training. Figure 3 further shows that reference quality affects training stability. Although Self Anchor yields early improvements, several metrics deteriorate during later training. In contrast, GPT-5.5 Anchor remains comparatively stable and retains its gains throughout training. This comparison suggests that weaker references still leave room for one-sided visual degradation, whereas higher-quality reference artifacts provide a more reliable calibration baseline that limits such reward exploitation and stabilizes optimization.

Action-Trace Credit Accelerates Optimization. While improving complex interactive HTML artifacts’ quality with fine-grained reward signals, Action-Trace Credit also changes how quickly the model converts interaction feedback into measurable improvements. In Figure 3, ATC reaches approximately 56–57% WebGen accuracy at step 30, whereas the same GPT-5.5 Anchor without ATC reaches comparable accuracy only around step 70. Functional Testing shows a similar pattern.

![](images/d82f15daa3345018dc1ca28c5687a7fdea46a82cd2f5678786b211fdf975383a.jpg)  
Figure 4: Reward trajectories with and without runtimegrounded credit assignment. The vertical axis shows the EMA of the mean critic score, and High KL denotes stronger KL regularization. Dashed markers indicate when variants reach comparable score levels. All settings exhibit diminishing gains after approximately 40 steps. Optimization is approaching convergence.

Figure 4 provides consistent optimization evidence: ATC reduces training steps to reach a comparable EMA critic score by approximately 1.84×. Together, evaluation and training curves show that runtime-grounded credit assignment accelerates convergence by directing interaction feedback to code responsible for successful actions.

## 4.4.2 Experience Diagnostics and Iterative Refinement

Gains Extend Across Experience Dimensions. We further analyze the Meta-level Judge’s experience dimensions and Step-level Judge outcomes. Figure 5 shows WebCatalyst improves all six evaluated dimensions, including visual stability, feedback clarity, interaction reliability, state legibility, prompt alignment, and task progress. Gains also span all action types, raising the overall action success rate from 48.0% to 67.4%. These broad gains indicate that the aggregate benchmark improvements are not driven by a narrow subset of behaviors, but reflect more general improvements in interaction quality.

Single-Shot Gains Persist under Iterative Refinement. We examine whether WebCatalyst’s single-shot improvements persist in an iterative refinement pipeline that emulates human-in-theloop development, where feedback grounded in actual use of the generated artifact guides successive HTML revisions. Each round, the GUI Assessor interacts with the generated artifact, identifies experiential weaknesses and issues, and provides this feedback for revision. As shown in Table 3, WebCatalyst achieves a Total score of 75.3 without refinement, matching the backbone’s 75.2 after two refinement rounds. It also maintains a higher Total score at every corresponding round and reaches 76.9 after three rounds, exceeding the backbone’s best of 75.2. Thus, WebCatalyst reduces feedback rounds to comparable quality and improves attainable quality within the evaluated refinement budget.

![](images/61ba3ce1a30de7afaf2064c7af15178e4839ea39764ac77ff14b88fbf8519026.jpg)  
Figure 5: User-experience and action-level gains before and after training. Before and After denote Qwen3.6- 35B-A3B and WebCatalyst, respectively. The left panel shows experience-dimension scores, while the right shows action-transition scores.

Table 3: Mini Games performance under iterative refinement. Round 0 denotes the initial generation, and subsequent rounds revise the artifact using feedback from the GUI Assessor. Bold indicates the better result for each metric within each round.
<table><tr><td rowspan="2">Round</td><td colspan="3">Qwen3.6-35B-A3B</td><td colspan="3">+WebCatalyst</td></tr><tr><td>Aes.</td><td>Func.</td><td>Total</td><td>Aes.</td><td>Func.</td><td>Total</td></tr><tr><td>0</td><td>81.5</td><td>56.1</td><td>68.8</td><td>87.5</td><td>63.2</td><td>75.3</td></tr><tr><td>1</td><td>85.1</td><td>57.9</td><td>71.5</td><td>88.1</td><td>63.4</td><td>75.7</td></tr><tr><td>2</td><td>87.7</td><td>62.8</td><td>75.2</td><td>89.5</td><td>61.8</td><td>75.7</td></tr><tr><td>3</td><td>86.2</td><td>62.1</td><td>74.2</td><td>91.1</td><td>62.7</td><td>76.9</td></tr></table>

## 5 Conclusion

We introduce WebCatalyst, a unified browserinteraction RL framework that improves supervision fidelity across interaction experience, reward signals, and credit assignment. Trained on fewer than 1K requests, primarily mini-games, WebCatalyst substantially improves mini-game generation, with gains transferring to conventional webpage tasks. These results underscore the importance of high-fidelity feedback for moving beyond taskspecific webpage optimization toward more capable and general interactive HTML generation.

## 6 Limitations

While WebCatalyst enhances the interactive HTML generation capabilities of LLMs, several aspects remain to be explored. First, this work focuses on direct generation. Although our experiments show that stronger direct generation also benefits multiturn refinement, we do not specifically optimize more complex pipelines involving planning, iterative debugging, or tool use. Second, WebCatalyst studies how RL activates and strengthens interactive HTML capabilities already present in the base model. It complements rather than replaces SFT, which establishes the foundation for visual style, animation, and artifact complexity. The effectiveness of RL therefore remains dependent on the capabilities acquired before training. Finally, we train our model on game prompts to examine whether capabilities learned from interaction-intensive tasks transfer to simpler settings such as conventional webpages. We leave the optimization of task mixtures and data composition, as well as their effects on performance and generalization, to future work.

More broadly, stronger interactive HTML generation capabilities may introduce misuse risks, including the generation of deceptive interfaces, phishing websites, malicious client-side behaviors, or other harmful web content. Although our work focuses on benign generation tasks, future deployment should incorporate appropriate safety safeguards and misuse-aware evaluation.

## References

Tony Beltramelli. 2017. pix2code: Generating code from a graphical user interface screenshot. arXiv preprint arXiv:1705.07962.

Boshui Chen, Huiping Liu, and Shaolei Zhang. 2026. WebGrader: Training LLMs for web development with self-evolving programmatic grader. arXiv preprint arXiv:2608.06474.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. 2024. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313– 9332, Bangkok, Thailand. Association for Computational Linguistics.

Jie Deng, Kaichun Yao, and Libo Zhang. 2026. VisRefiner: Learning from visual differences for screenshot-to-code generation. arXiv preprint arXiv:2602.05998.

Shihan Dou, Yan Liu, Haoxiang Jia, Enyu Zhou, Limao Xiong, Junjie Shan, Caishuang Huang, Xiao Wang, Xiaoran Fan, Zhiheng Xi, Yuhao Zhou, Tao Ji, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. StepCoder: Improving code generation with reinforcement learning from compiler feedback. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4571–4585, Bangkok, Thailand. Association for Computational Linguistics.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. 2024. WorkArena: How capable are web agents at solving common knowledge work tasks? In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 11642–11662. PMLR.

Lishui Fan, Yu Zhang, Mouxiang Chen, and Zhongxin Liu. 2026. ReCode: Reinforcing code generation with reasoning-process rewards. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 43896–43914, San Diego, California, United States. Association for Computational Linguistics.

Yi Gui, Zhen Li, Yao Wan, Yemin Shi, Hongyu Zhang, Yi Su, Bohua Chen, Dongping Chen, Siyuan Wu, Xing Zhou, Wenbin Jiang, Hai Jin, and Xiangliang Zhang. 2025. WebCode2M: A real-world dataset for code generation from webpage designs. In Proceedings ofthe ACM Web Conference 2025. Association for Computing Machinery.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. WebVoyager: Building an end-toend web agent with large multimodal models. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6864–6890, Bangkok, Thailand. Association for Computational Linguistics.

Yixu Huang, Bo Li, Na Li, Zhe Wang, Kaijie Chen, Haonan Ge, Qingyi Si, Yuanzhe Shen, Ruihan Yang, Guangjing Wang, and Hongcheng Guo. 2026. GUI agents for continual game generation. arXiv preprint arXiv:2605.28258.

Juyong Jiang, Chenglin Cai, Chansung Park, Jiasi Shen, Sunghun Kim, Jianguo Li, and Yue Wang. 2026. WebGen-R1: Incentivizing large language models to generate functional and aesthetic websites with reinforcement learning. arXiv preprint arXiv:2604.20398.

Rui Jin, Jikai Chen, Yihan Chen, Hao Zhou, Demin Zhu, Kaichen Yang, Dong Wang, Linjian Mo, and Chenyi Zhuang. 2026. Rubric-to-code credit assignment for reinforcement learning. arXiv preprint arXiv:2608.27906.

Abhijit Kumar, Natalya Kumar, and Shikhar Gupta. 2026. Execution-grounded credit assignment for GRPO in code generation. arXiv preprint arXiv:2603.16158.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, page 611–626, New York, NY, USA. Association for Computing Machinery.

Peichao Lai, Jinhui Zhuang, Kexuan Zhang, Ningchang Xiong, Shengjie Wang, Yanwei Xu, Chong Chen, Yilei Wang, and Bin Cui. 2025. WebRenderBench: Enhancing web interface generation through layoutstyle consistency and reinforcement learning. arXiv preprint arXiv:2510.04097.

Hugo Laurençon, Léo Tronchon, and Victor Sanh. 2024. Unlocking the conversion of web screenshots into HTML code with the WebSight dataset. arXiv preprint arXiv:2403.09029.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. 2022. CodeRL: Mastering code generation through pretrained models and deep reinforcement learning. In Advances in Neural Information Processing Systems, volume 35, pages 21314–21328.

Yuanhao Li, Hongbo Wang, Xiaotang Shang, Xunzhu Tang, Yiming Cao, and Xuhong Chen. 2026a. BOOSTAPR: Boosting automated program repair via execution-grounded reinforcement learning with dual reward models. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 70897–70918. PMLR.

Yuhang Li, Chenchen Zhang, Ruilin Lv, Ao Liu, Ken Deng, Yuanxing Zhang, Jiaheng Liu, and Bo Zhou. 2026b. ReLook: Vision-grounded RL with a multimodal LLM critic for agentic web coding. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 25471–25485, San Diego, California, United States. Association for Computational Linguistics.

Jiate Liu, Yiqin Zhu, Kaiwen Xiao, Qiang Fu, Xiao Han, Wei Yang, and Deheng Ye. 2023. RLTF: Reinforcement learning from unit test feedback. Transactions on Machine Learning Research.

Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Junting Pan, Mingjie Zhan, and Hongsheng Li. 2026. WebGen-Agent: Enhancing interactive website generation with multi-level feedback and step-level reinforcement learning. In International Conference on Learning Representations.

Zimu Lu, Yunqiao Yang, Houxing Ren, Haotian Hou, Han Xiao, Ke Wang, Weikang Shi, Aojun Zhou,

Mingjie Zhan, and Hongsheng Li. 2025. WebGen-Bench: Evaluating LLMs on generating interactive and functional websites from scratch. In Advances in Neural Information Processing Systems, volume 38, pages 80192–80236.

Zhiyuan Peng, Wei Tao, Xin Yin, Chenhao Ying, Yuan Luo, and Yiwen Guo. 2026. PlayCoder: Making LLM-generated GUI code playable. arXiv preprint arXiv:2604.19742. Accepted at FSE 2026.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, and 16 others. 2025. UI-TARS: Pioneering automated GUI interaction with native agents. arXiv preprint arXiv:2501.12326.

Qwen Team. 2026a. Qwen3.6-27B: Flagship-level coding in a 27B dense model.

Qwen Team. 2026b. Qwen3.6-35B-A3B: Agentic coding power, now open to all.

Qwen Team. 2026c. Qwen3.8-Max: A new bar for coding and cowork.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems, EuroSys ’25, page 1279–1297, New York, NY, USA. Association for Computing Machinery.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053.

Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. 2025. Design2Code: Benchmarking multimodal code generation for automated front-end engineering. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3956–3974, Albuquerque, New Mexico. Association for Computational Linguistics.

Jiajun Wu, Jian Yang, Tuney Zheng, Wei Zhang, Haowen Wang, Yihang Lou, and Xianglong Liu. 2026. HTMLCure: Turning browser experience into state guided repair for interactive HTML. arXiv preprint arXiv:2605.26807.

Jingyu Xiao, Yuxuan Wan, Yintong Huo, Zixin Wang, Xinyi Xu, Wenxuan Wang, Zhiyao Xu, Yuhang Wang, and Michael R. Lyu. 2025. Interaction2Code: Benchmarking MLLM-based interactive webpage code generation from interactive prototyping. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 241–253.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. 2024. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems, volume 37, pages 52040–52094.

Haoyue Yang, Zhangxiao Shen, Fan Ding, Hangting Lou, Yifeng Kou, Haoqing Yu, Jingyao Li, Zhengfan Wu, Siqi Bao, Jing Liu, and Hua Wu. 2026. Cookie-Bench: Continuous on-screen key interaction evaluation for web generation. arXiv preprint arXiv:2605.30000.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations.

Sukmin Yun, Haokun Lin, Rusiru Thushara, Mohammad Qazim Bhat, Yongxin Wang, Zutao Jiang, Mingkai Deng, Jinhong Wang, Tianhua Tao, Junbo Li, Haonan Li, Preslav Nakov, Timothy Baldwin, Zhengzhong Liu, Eric P. Xing, Xiaodan Liang, and Zhiqiang Shen. 2024. Web2Code: A large-scale webpage-to-code dataset and evaluation framework for multimodal LLMs. In Advances in Neural Information Processing Systems, volume 37.

Chenchen Zhang, Yuhang Li, Can Xu, Jiaheng Liu, Ao Liu, Changzhi Zhou, Ken Deng, Dengpeng Wu, Guanhua Huang, Kejiao Li, Qi Yi, Ruibin Xiong, Shihui Hu, Yue Zhang, Yuhao Jiang, Zenan Xu, Yuanxing Zhang, Wiggin Zhou, Chayse Zhou, and Fengzong Lian. 2025. ArtifactsBench: Bridging the visual-interactive gap in LLM code generation evaluation. arXiv preprint arXiv:2507.04952.

Wenyu Zhang, Guoliang You, Tianlun, Haotian Zhao, Tianshu Zhu, Haoran Wang, Xiaoxuan Tang, Mingyang Dai, Jingnan Gu, Daxiang Dong, and Jianmin Wu. 2026a. WebGameBench: Requirement-toapplication evaluation for coding agents via browsernative games. arXiv preprint arXiv:2605.17637.

Zuhao Zhang, Chengyue Yu, Yuante Li, Chenyi Zhuang, Linjian Mo, and Shuai Li. 2026b. MiniAppBench:

Evaluating the shift from text to interactive HTML responses in LLM-powered assistants. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 160169–160206. PMLR.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. WebArena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations.

## A Appendix

## A.1 GUI Assessor Pipeline

This section includes the details of our GUI Assessor. Given a requirement prompt x and a sampled HTML artifact $h _ { i } .$ , the assessor (i) executes $h _ { i }$ under controlled browser budget, (ii) chooses and performs a sequence of user actions, (iii) evaluates every transition as soon as it is produced, and (iv) aggregates the completed trajectory into the pagelevel score $R _ { i } ^ { \mathrm { G U I } }$ . We present the overall workflow of the GUI Assessor below, together with the key design considerations that guide its construction.

## A.1.1 Controlled Browser Execution

We load $h _ { i }$ in a fresh Chromium context<sup>1</sup> with a fixed 1440×900 viewport and wait for network idleness before interaction. Playwright’s virtual clock is then paused and advanced only by the executor. This prevents uncontrolled wall-clock delays from changing animations or game state while a model call is in flight. At turn t, the page has latent browser state $V _ { i , t }$ and observable screenshot $S _ { i , t } = \mathcal { O } ( V _ { i , t } )$ . After choosing and executing $a _ { i , t } .$ the browser produces $V _ { i , t + 1 } = \mathcal { T } _ { h _ { i } } ( V _ { i , t } , a _ { i , t } )$ , the post-action screenshot $S _ { i , t + 1 }$ , and the execution record $m _ { i , t }$ . Together they form the transition $E _ { i , t }$ defined in the main text.

Before the first turn, the mouse is moved to the viewport center. Before every subsequent decision, the virtual clock is paused again and a fullpage screenshot is captured. An episode terminates when the Execution Agent emits terminate or when the interaction budget is exhausted.

## A.1.2 Execution-Agent Context and Decision

The Execution Agent implements the ReAct-style interaction policy. Its decision context contains direct visual evidence, a compact history, and browser affordances, as summarized in Table 4. The screenshot remains the source of truth for the current visible state. Source code is also supplied only as a rules reference for identifying supported inputs, event handlers, coordinate transforms, cooldowns, and pointer-lock behavior.

The agent returns a structured decision containing a one-sentence summary, one action ${ a } _ { i , t }$ , a predicted visible target state, an expected transition, and optional psychometric observations. Only ${ a } _ { i , t }$ is executed. The predicted target and transition are retained as diagnostic intent and are explicitly marked as non-ground-truth inputs to the Step-Level Judge. If the returned action violates the schema, the invalid action is not executed. The validation error and previous response are returned to the agent for correction.

## A.1.3 Action Grounding and Timing Semantics

Table 5 details the parameterization of the action space A. For click and hover, targets are resolved in the following order: explicit page coordinates, selector plus text, selector alone, and text alone. The executor records the resolution method, number of matches, actual hit-tested element, element rectangle, page and local coordinates, and canvas dimensions in $m _ { i , t }$ . Consequently, the judge can distinguish a simulator miss from a correctly delivered but ineffective action.

The optional waitMs delays action onset, whereas durationMs controls how long an action is held. Let $\Delta$ denote the browser clock-frame interval. When a duration is explicitly requested, we execute

$$
\ell _ { i , t } = \operatorname* { m a x } \biggl ( \ell _ { i , t } ^ { \mathrm { r e q } } , 2 \Delta + 2 5 \mathrm { m s } \biggr ) ,\tag{11}
$$

so that a held action crosses multiple observable frames. During a held action, intermediate observations are captured after geometrically increasing elapsed times (50, 100, 200 ms, and so on) up to $\ell _ { i , t }$ . For an instantaneous action, the clock is advanced by 50 ms before capturing $S _ { i , t + 1 }$ , allowing immediate visual feedback to become observable. This design is primarily intended to facilitate trajectory debugging for the GUI Assessor and to support its use as an evaluator. During training, for cost efficiency and lower latency, we only provide the before and after states to the step-level judge.

Canvas interactions require special handling. A selector-only click on a 2-D canvas lands at its center and is therefore annotated with a warning. Clicks on specific tiles or sprites should include page coordinates. For first-person or pointer-lock pages, firing is dispatched to the locked target at the current crosshair, while view rotation is expressed as relative mouseMove. Execution exceptions do not erase the transition: they are serialized into $m _ { i , t }$ with success=false, an error message, and diagnostic warnings for the Step-level Judge.

Table 4: Input information provided to the Execution Agent at turn t. Dynamic values are serialized into the user message, while the action schema and execution rules are specified in the system prompt.
<table><tr><td>Component</td><td>Content</td><td>Role in action selection</td></tr><tr><td>Prompt</td><td>Original prompt x</td><td>Specifies the intended artifact and user objective.</td></tr><tr><td>Visual State</td><td>Current full-page screenshot  $S _ { i , t }$ </td><td>Provides the authoritative visible state from which the next action is chosen.</td></tr><tr><td>Artifact</td><td>Source of  $h _ { i } ,$  truncated to 12,000 characters</td><td>Reveals input handlers and interaction rules to support precise action generation.</td></tr><tr><td>Action Hints</td><td>Up to 40 DOM controls, each represented by its tag, selector hint, text, and bounding rectangle</td><td>Grounds ordinary DOM actions in executable selec- tors. Canvas-internal targets are inferred from the screenshot and source.</td></tr><tr><td>History</td><td>Including previous decision summary, ac- tion, execution observation, and termination status</td><td>Discourages repetitive actions and supports multi-step behavior.</td></tr><tr><td>Timing</td><td>Turn index, turn limit, clock-frame interval, and post-action advance budget</td><td>Allows the agent to choose pre-action delays and held- action durations.</td></tr></table>

Table 5: Grounded action interface used for different types of action $a _ { i , t } \in \mathcal { A }$ in the browser.
<table><tr><td>Action</td><td>Parameters</td><td>Execution semantics</td></tr><tr><td>click</td><td>selector or text. (optional) page coordi- nates (x, y), durationMs, waitMs</td><td>Activates a DOM target or page coordinate. A held click emits mouse-down, advances the clock, and then emits mouse-up.</td></tr><tr><td>hover</td><td>selector or text. (optional) page coordi- nates (x, y), waitMs</td><td>Moves the pointer to the resolved element or coordinate without activation.</td></tr><tr><td>mouseMove</td><td>relative (∆x, ∆y). (optional) durationMs, waitMs</td><td>Produces smooth relative motion. Under pointer lock it dispatches relative movement events rather than absolute cursor motion.</td></tr><tr><td>press</td><td>key. (optional) durationMs, waitMs</td><td>Emits an instantaneous key press or a key-down/key-up pair around a held interval.</td></tr><tr><td>combo</td><td>multiple keys. (optional) durationMs, waitMs</td><td>Holds the keys simultaneously and releases them in re- verse order.</td></tr><tr><td>type</td><td>selector or text. (optional) waitMs</td><td>Fills the selected text input.</td></tr><tr><td>wait terminate</td><td>duration in milliseconds</td><td>Advances virtual browser time without another user input.</td></tr><tr><td></td><td>optional reason</td><td>Stops interaction when the task is complete, the game is unrecoverable, or a realistic user would stop.</td></tr></table>

## A.1.4 Queued Step-Level Assessment

Immediately after $E _ { i , t }$ is materialized, it is submitted to a FIFO producer–consumer queue. The Execution Agent can proceed to turn t+1 while the Step-level Judge consumes earlier transitions. A single judge request jointly evaluates the transition and the post-action screenshot, using $x , S _ { i , t } , a _ { i , t }$ $m _ { i , t } , S _ { i , t + 1 }$ , and timing metadata. The image order is fixed as before-action followed by after-action. This joint call keeps the assessments grounded in identical evidence while the queue overlaps judge latency with continued browser execution.

For compactness, let $z _ { i , t , } ^ { \mathrm { t r } }$ and $z _ { i , t , } ^ { \mathrm { s c } }$ denote the Step-level Judge’s component scores, all in [0, 100]. The two reported scores are

$$
\begin{array} { r l } & { q _ { i , t } ^ { \mathrm { t r } } = 0 . 4 0 z _ { i , t , \mathrm { v s } } ^ { \mathrm { t r } } + 0 . 2 5 z _ { i , t , \mathrm { f b } } ^ { \mathrm { t r } } + 0 . 2 5 z _ { i , t , \mathrm { e x } } ^ { \mathrm { t r } } + 0 . 1 0 z _ { i , t , \mathrm { p r } } ^ { \mathrm { t r } } , } \\ & { q _ { i , t } ^ { \mathrm { s c } } = 0 . 4 5 z _ { i , t , \mathrm { v s } } ^ { \mathrm { s c } } + 0 . 3 0 z _ { i , t , \mathrm { s v } } ^ { \mathrm { s c } } + 0 . 2 5 z _ { i , t , \mathrm { r d } } ^ { \mathrm { s c } } . } \end{array}\tag{12}
$$

Here VS, FB, EX, and PR denote visual stability, application-feedback quality, execution correctness, and task progress. SV and RD denote state visibility and readability. Task progress is intentionally weakly weighted so that visually stable, understandable rule-based rejection is not treated as a broken interaction.

In addition to numeric scores, the Step-level Judge returns strengths, weaknesses, a compact visible-state description, and reusable interaction experience. The failure taxonomy is important for attribution:

• coordinate\_miss: the simulator did not hit the intended target;

• action\_ineffective: a correctly delivered, apparently valid action produced no meaningful response;

• illegal\_or\_rejected: the action violated visible game rules or prerequisites;

• timing\_issue: the action was plausible but mistimed or held for an unsuitable duration;

• uncertain: available evidence does not support a reliable diagnosis; and

• none: no relevant failure was observed.

Thus, simulator-side delivery errors primarily lower execution correctness, whereas applicationside failures lower feedback and interaction quality. A clear, rule-consistent rejection can still receive high visual-stability and feedback scores.

## A.1.5 Trajectory-Level Meta Assessment

After interaction terminates, the producer stops and we wait for the queue to drain so that every $E _ { i , t }$ has a completed Step-level assessment. The Metalevel Judge receives x, episode-level termination and timing information, every action and execution record, and compact Step-level outputs. Visual evidence is conveyed through the structured Step-level scores, state descriptions, strengths, weaknesses, and failure modes. This separation makes the final decision auditable and avoids an unbounded multi-image context.

We set seven Meta-level dimensions. They are visual stability $( z _ { i , \mathrm { v s } } )$ , screenshot aesthetics $( \dot { z } _ { i , \mathrm { a e } } ) .$ feedback clarity $( z _ { i , \mathrm { f c } } )$ , interaction reliability $( z _ { i , \mathrm { i r } } )$ state legibility $( z _ { i , \mathrm { s l } } ) .$ , requirement alignment $( z _ { i , \mathrm { p a } } )$ and task progress $( z _ { i , \mathrm { t p } } )$ The weighted episode score is

$$
\begin{array} { r l } & { \overline { { R } } _ { i } ^ { \mathrm { G U I } } = 0 . 1 5 z _ { i , \mathrm { v s } } + 0 . 1 5 z _ { i , \mathrm { a e } } + 0 . 1 0 z _ { i , \mathrm { f c } } } \\ & { + 0 . 2 0 z _ { i , \mathrm { i r } } + 0 . 1 0 z _ { i , \mathrm { s l } } + 0 . 2 0 z _ { i , \mathrm { p a } } + 0 . 1 0 z _ { i , \mathrm { t p } } . } \end{array}\tag{13}
$$

All dimensions are clamped to [0, 100]. To prevent a high average from masking a severe defect, the judge may additionally assign a usability cap $C _ { i }$ (Table 6). Taking $C _ { i } = 1 0 0$ when no cap is needed, the score consumed by the Visual Anchor and GRPO stages is

$$
R _ { i } ^ { \mathrm { G U I } } = \operatorname* { m i n } \left( \overline { { R } } _ { i } ^ { \mathrm { G U I } } , C _ { i } \right) .\tag{14}
$$

## A.2 Action Trace Collection and Token Mapping

Online trace collection. We attach a Chrome DevTools Protocol session to the Chromium page used by the GUI Assessor. Before interaction, JavaScript precise coverage identifies functions with resolvable source locations in inline scripts from the artifact’s main-frame execution context, for which function-entry breakpoints are installed. The current implementation traces click, mouseMove, press, and combo actions. During each action, debugger pauses expose active call stacks, while precise coverage additionally identifies executed functions that may not appear in the captured stacks. To bound tracing overhead, at most 4096 pause events are processed per action.

<table><tr><td> $C _ { i }$ </td><td>Corresponding Issues</td></tr><tr><td>0-10</td><td>Blank, crashed, corrupted, or impossible to evaluate.</td></tr><tr><td>20</td><td>Page renders but no meaningful primary interaction works.</td></tr><tr><td>35</td><td>A required entry action is correctly targeted but nonfunctional.</td></tr><tr><td>45</td><td>The artifact is substantially the wrong prod- uct or lacks the requested core experience.</td></tr><tr><td>60</td><td>The artifact is recognizable, but its cen- tral loop or required primary mechanic is blocked.</td></tr><tr><td>70</td><td>The main loop works, but a prominent re- quired feature is missing or nonfunctional.</td></tr><tr><td>80</td><td>Primary play remains possible despite no- table missing content or unreliable sec- ondary mechanics.</td></tr><tr><td>90</td><td>Only minor prompt omissions or polish gaps prevent a near-perfect score.</td></tr><tr><td>100</td><td>No cap. Ordinary weaknesses are already represented by the component scores.</td></tr></table>

Table 6: Usability-cap rubric. The judge chooses the highest cap consistent with the observed severity.

Tracing is performed online, while attribution is filtered after Step-level assessment. An action contributes source evidence only when its transition is judged successful. Failed or low-quality actions are therefore excluded from source attribution, avoiding spurious negative credit to unrelated code. Before attribution, duplicate stacks and strict suffixes of longer captured call paths are removed.

Source span, runtime depth, and token alignment. Each retained call frame is mapped to the smallest enclosing function span in the generated HTML. Source offsets are converted to UTF-8 byte intervals and then mapped back to the full policy response $y _ { i }$ using the recorded location of $h _ { i }$ . SHA-256 hashes and byte lengths are verified to detect offset drift before attribution.

The implementation measures runtime depth in the leaf-to-caller order returned by Chrome Dev-Tools. Let $s = ( f _ { 0 } , \ldots , f _ { L _ { s } - 1 } )$ denote a retained maximal stack, where $f _ { 0 }$ is the currently executing leaf frame containing core logic and $f _ { L _ { s } - 1 }$ is the outermost captured caller. For the retained stack set $S _ { i }$ , function spans, and UTF-8 byte positions $b ,$ we define

$$
\begin{array} { l } { { d _ { s } ( f ) = \operatorname* { m i n } \{ j \ | \ f _ { j } = f \} , } } \\ { { d _ { i } ( f ) = \displaystyle \operatorname* { m i n } _ { s \in S _ { i } : f \in s } d _ { s } ( f ) , } } \\ { { \delta _ { i } ( b ) = \displaystyle \operatorname* { m i n } _ { f : b \in \mathrm { s p a n } ( f ) } d _ { i } ( f ) , } } \\ { { d _ { i , k } = \displaystyle \operatorname* { m a x } _ { b \in { S _ { i } } , k } \delta _ { i } ( b ) , } } \end{array}\tag{15}
$$

where $B _ { i , k }$ is the UTF-8 byte span of token $k ,$ and $d _ { i , k }$ is defined only for fully traced tokens.

This implementation maps executed code frames back to their corresponding rollout tokens while avoiding potential misalignment at token boundaries. Repeated observations and overlapping function spans therefore retain the smallest available runtime depth, while a token spanning regions with different depths uses the largest depth among its covered bytes. Under this convention, leaf frames have depth 0 and outer callers receive progressively larger depths.

The original rollout token IDs are reconstructed as exact UTF-8 byte spans. For byte-level tokenizers, these spans are obtained directly from token pieces. Otherwise, fast-tokenizer character offsets are converted to UTF-8 byte offsets over the exact rollout text and accepted only when re-tokenization recovers the original non-special token IDs. A token is included in $\mathcal { F } _ { i , k }$ only when every byte in $B _ { i , k }$ is covered by valid trace runs.

Effect of action-trace credit. Figure 6 shows that action-trace credit remains highly selective, covering only about 23% of the generated code while focusing optimization on source regions associated with successful interactions. After training, the average code size increases from 33.27 to 38.96 kB, accompanied by a proportional increase in selected-trace code from 7.37 to 8.94 kB. More notably, the share of total code attributed to depth-0 leaf frames rises from 10.57% to 13.20%. Since these leaf frames contain the logic executed most directly during successful actions, this shift suggests that training allocates more model capacity to robust, task-relevant interaction logic rather than merely expanding peripheral code.

Overall Positioning. WebCatalyst advances RL supervision for interactive HTML generation across three coupled stages from interaction to policy optimization. First, the Game-Level GUI Assessor converts sustained browser interaction into trajectory-level RL feedback, moving beyond shallow click-and-type verification and GUI agents used primarily for evaluation or inference-time refinement. Second, Visual Anchor introduces onesided reference calibration to prevent functionalitydriven optimization from exploiting visual simplification, rather than serving as a generic aesthetic reward. Third, Action-Trace Credit grounds token-level credit in individual successful GUI actions by tracing their runtime JavaScript call paths back to generated tokens, rather than relying on compiler information or static code regions. Together, WebCatalyst establishes an end-to-end interaction-grounded supervision pipeline, carrying high-fidelity signals from experienced behavior to reward construction and ultimately to behaviorrelevant code tokens.

![](images/201a79c9017d83387bf1e19c8273cf53894319aacbb88a306329ca216a617df2.jpg)

![](images/791a5b2633ca30be04a143dda00d1e6f376935d328c54a10ad5ff2bb1929e7f1.jpg)  
Figure 6: Changes in code size and action-trace attribution after training. The solid regions denote selectedtrace code, while the pale regions denote the remaining generated code. Leaf-code coverage measures the proportion of total code attributed to depth-0 frames.

## A.3 Implementation Details

## A.3.1 Training Hyperparameters

Tables 7 and 8 summarize the configuration used for the main experiment. We optimize Qwen3.6- 35B-A3B with GRPO through verl (Sheng et al., 2025). At each update, $G = 5$ responses are sampled for each of 16 requests. The reported checkpoint is taken after 40 updates, corresponding to 640 processed request instances. We use groupwise standard-deviation normalization in our GRPO objective, with $\epsilon = 1 0 ^ { - 6 }$

Training uses one node with eight NVIDIA B200 GPUs. Policy updates are performed with Megatron-Core (Shoeybi et al., 2019) in bfloat16, while vLLM (Kwon et al., 2023) generates the rollouts. On B200 GPU, we run one complete rollout replica per GPU and use the Triton MoE backend.

Table 7: Optimization and generation hyperparameters.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbone</td><td>Qwen3.6-35B-A3B</td></tr><tr><td>Optimization algorithm</td><td>GRPO</td></tr><tr><td>Requests per update</td><td>16</td></tr><tr><td>Group size</td><td>5</td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>LR schedule</td><td>Constant</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Clipping εc</td><td>0.2</td></tr><tr><td>Loss aggregation</td><td>Token mean</td></tr><tr><td>KL regularization</td><td> $\eta = 0 . 0 0 1$ </td></tr><tr><td>GRPO standard-deviation normalization</td><td> $\epsilon = 1 0 ^ { - 6 }$ </td></tr><tr><td>Max. prompt / response</td><td>2,048 / 32,768 tokens</td></tr><tr><td>length Training sampling</td><td>Temperature  $T = 1 . 0$ </td></tr><tr><td>Random seed</td><td>42</td></tr></table>

Table 8: Reward and execution settings. TP, PP, CP, EP, and ETP denote tensor, pipeline, context, expert, and expert-tensor parallelism, respectively.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>GUI Assessor judge</td><td>GPT-5.6 Terra</td></tr><tr><td>Interaction turns</td><td>8</td></tr><tr><td>Visual anchor model</td><td>GPT-5.5</td></tr><tr><td>ATC modulation</td><td> $\beta = 0 . 5$ </td></tr><tr><td>Trace pause-event cap</td><td>4,096 per action</td></tr><tr><td>Assessor workers</td><td>6 workers</td></tr><tr><td>Attempt timeout</td><td>300 s</td></tr><tr><td>Training precision</td><td>bfloat16</td></tr><tr><td>Policy parallelism</td><td> $\mathrm { T P = 2 , P P = 1 , C P = 1 , }$ </td></tr><tr><td>MoE auxiliary / z-loss coefficients</td><td> $\mathrm { E P = 8 , E T P = 1 }$   $0 . 0 1 / 0 . 0 0 1$ </td></tr></table>

## A.3.2 Training Dataset Construction

We construct the training set from Reddit posts related to frontend and game development. We first retrieve candidate requests using “game” and related keywords. We ensure that no sensitive or personal identification information is obtained in our data. The requests are then rewritten into natural user-facing queries with diverse personas, tones, and levels of constraint specificity. We apply a broad HTML-game filter that retains examples exhibiting a complete game subject, browser or HTML technology, and concrete game mechanics, while removing backend-only questions and requests that do not define a playable artifact. GPT-5.5 subsequently verifies each retained query and either keeps it unchanged when it is already suitable or refines it into a direct request for a complete, self-contained browser game. After deduplication, the resulting prompts are split into training and validation sets. At training time, each prompt is paired with a system instruction requiring the policy to return only one complete HTML document with all CSS and JavaScript included inline.

## A.3.3 Evaluation Protocol

Generation setup. We evaluate the models on the test sets of WebGen-Bench (Lu et al., 2025) and Cookie-Bench Mini Games (Yang et al., 2026). For every benchmark request, the model receives the original user prompt together with the same outputformat instruction, that is, return one complete, self-contained HTML document with all CSS and JavaScript included inline.

For our locally served Qwen checkpoints, we use temperature 0.6, top-p 0.95, top-k 20, a maximum of 32,768 response tokens, and enable the model’s thinking mode. For proprietary baselines, we retain the provider-native sampling configuration and set the largest supported output-token budget when that control is available.

Cookie-Bench Mini Games. We use the deduplicated multilingual Mini Games split containing 156 requests, primarily in Chinese and English, with additional examples in French, German, Spanish, and other languages. Each generated HTML file is served in Chromium and first receives provisional aesthetics and functionality scores from its initial rendered state. A source-aware planner then constructs a test plan, after which UI-TARS explores the game through a ReAct loop. The evaluator records the action trace and reviews it, identifies functional and visual problems, and adjusts the provisional scores. We use Qwen3.8-27B as the execution model and DeepSeek-V4-Pro as the review model. A model-side format generation, installation, or launch failure is retained with zero score. A failed evaluator pass is retried once with a fresh browser-agent session.

WebGen-Bench. WebGen-Bench contains 101 requests and 647 atomic test cases. We serve each generated page locally and use Qwen2.5-VL-32B-Instruct as the UI evaluation agent. For each test case, the agent is given an operation and its expected outcome, may perform at most 15 browser interactions, and returns YES, PARTIAL, or NO. With $v _ { j } \in \{ 1 , 0 . 5 , 0 \}$ denoting these verdicts, overall accuracy is $\mathrm { A c c } \ = \ 1 0 0 \textstyle \sum _ { j = 1 } ^ { 6 4 7 } v _ { j } / 6 4 7$ . The same aggregation is used for the request-level categories (Content Presentation, User Interaction, and Data Management) and test-case categories (Functional Testing, Data Display Testing, and Design Validation Testing). The fraction for pages that cannot be served is reported as Start Failure Rate.

## A.4 Practical Deployment Guidance

## A.4.1 Cost Breakdown

We train Qwen3.6-35B-A3B for 40 optimization steps on eight NVIDIA B200 GPUs. The main efficiency bottleneck is artifact evaluation by the GUI Assessor. Our primary setting uses GPT-5.6 Terra through an API limited to 50 requests per minute, with six concurrent assessor workers. Evaluating the 80 rollouts generated at each training step takes approximately 40 minutes, corresponding to an effective throughput of about 2 rollouts per minute. As shown in Table 9, evaluating one rollout consumes, on average, 112.27K input tokens and 9.28K output tokens, or 121.56K tokens in total. A full training step therefore consumes approximately 8.98M input tokens and 0.74M output tokens. At prices of \$2 and \$12 per million input and output tokens, respectively, this amounts to approximately \$26.88 per step, or \$1,075 over 40 steps, excluding caching discounts and unlogged retries. The run takes approximately 26.7 B200 GPU-hours for LLM inference and parameter updates.

On an eight-GPU A100 SXM system, the assessor latency is comparable to the time required for policy rollout generation. This suggests that an asynchronous policy-training pipeline with a policy staleness of approximately one step could overlap assessment with subsequent rollouts and improve hardware utilization. We leave such an implementation to future work.

We additionally experiment with Qwen3.8-27B as a locally served judge, using four B200 GPUs for vLLM judge serving and four B200 GPUs for rollout and training. This configuration requires approximately 192 aggregate B200 GPU-hours for 40 steps (about 24 hours of wall-clock time), while its token consumption is of the same order as that of GPT-5.6 Terra. These measurements expose both the monetary and latency costs of using our GUI Assessor and provide practical guidance for reproducing or scaling the training pipeline.

## A.4.2 Judge Selection

For ease of deployment, our main experiments use GPT-5.6 Terra through its API as the judge model. While this simplifies the training setup, API throughput becomes the primary bottleneck in training. We therefore investigate how judge selection affects training and whether locally served open-source models can provide a practical alternative. As shown in Table 10, we train additional variants using Qwen3.6-27B (Qwen Team, 2026a) and Qwen3.8-27B (Qwen Team, 2026c) as the judge backbone and the operator of the ReAct agent. To measure how well these judges reflect human experience, we construct a small human-annotated evaluation set. Following a simplified version of the Meta-Judge criteria, we select 10 prompts and collect artifacts from seven to eight models spanning different capability levels, including GPT-5.3- Codex, GPT-5.5, Claude Opus 4.7, GPT-5.4-mini among others. Our researchers evaluate more than 70 artifacts based on their actual interaction experience, while blinded to the identities of the models that generated them. We then measure the Pearson correlation between these human scores and the scores produced under different evaluation settings. For comparison, Src-Gemini uses Gemini 3.1 Pro to evaluate the source code directly, whereas Src-Rubric augments this source-based evaluation with rendered screenshots and task-specific rubrics. Table 10 reports both the resulting training performance and the corresponding Pearson correlations.

<table><tr><td>Component</td><td>Input (K)</td><td>Output (K)</td><td>Total (K)</td><td>Cost ($)</td></tr><tr><td>ReAct Agent</td><td>60.34</td><td>2.04</td><td>62.38</td><td>0.1452</td></tr><tr><td>Step-Level Judge</td><td>43.44</td><td>5.69</td><td>49.13</td><td>0.1551</td></tr><tr><td>Meta-Level Judge</td><td>8.49</td><td>1.55</td><td>10.05</td><td>0.0356</td></tr><tr><td>Total</td><td>112.27</td><td>9.28</td><td>121.56</td><td>0.3360</td></tr></table>

Table 9: Average per-sample evaluation token usage and estimated cost for GPT-5.6 Terra, with a budget of eight interaction turns. Token counts are in thousands (K). The turn judge jointly evaluates transitions and screenshots and is counted once. Costs assume \$2 and \$12 per million input and output tokens, respectively, excluding cache discounts.

Across judge backbones, the GUI Assessor maintains a higher correlation with human scores than rubric-based judging, indicating that its feedback better reflects actual human experience. Judge capability nevertheless remains important: a weaker model such as Qwen3.6-27B produces lowerfidelity evaluations and consequently weaker training outcomes. Once judge quality reaches the level of Qwen3.8-27B, with a Pearson correlation above 0.8, training becomes relatively stable, although the exact gains still vary across models and between games and webpages. These results suggest that self-hosted open-source judges can be used in WebCatalyst, provided that their evaluations remain sufficiently aligned with human experience.

Table 10: Effect of judge selection on downstream RL performance and alignment with human experience. Vanilla denotes the Qwen3.6-35B-A3B base model without RL. Src-Gemini uses Gemini 3.1 Pro to evaluate source code directly, while Src-Rubric additionally incorporates rendered screenshots and task-specific rubrics. For GUI Assessor, each row specifies the backbone used for interactive evaluation. Pearson r measures correlation with human experiential ratings. Arrows indicate the preferred direction, dashes indicate unavailable results, and green highlights judge settings with Pearson r > 0.8.
<table><tr><td rowspan="2">Grader</td><td colspan="7">WebGen-Bench</td><td colspan="3">Cookie-Bench Mini Games</td><td rowspan="2">Pearson r↑</td></tr><tr><td>Acc. ↑</td><td>Content Pres. ↑</td><td>User Inter. ↑</td><td>Data Mgmt. ↑</td><td>Func. Test ↑</td><td>Data Display ↑</td><td>Design Val. ↑</td><td>Aesth. ↑ Func. ↑</td><td></td><td>Total ↑</td></tr><tr><td>Vanilla (N/A)</td><td>48.8</td><td>59.2</td><td>43.0</td><td>47.1</td><td>33.9</td><td>61.8</td><td>70.5</td><td>81.5</td><td>56.1</td><td>68.8</td><td></td></tr><tr><td>Src-Gemini</td><td></td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td>一</td><td></td><td>0.065</td></tr><tr><td>Src-Rubric</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td></td><td>一</td><td>一</td><td>一</td><td>一</td><td>0.309</td></tr><tr><td colspan="10">GUI Assessor</td><td></td></tr><tr><td>Qwen3.6-27B</td><td>49.2</td><td>64.7</td><td>39.6</td><td>51.2</td><td>32.2</td><td>65.6</td><td>71.7</td><td>75.4</td><td>62.4</td><td>68.9</td><td>0.613</td></tr><tr><td>Qwen3.8-27B</td><td>59.0</td><td>70.4</td><td>53.7</td><td>56.9</td><td>45.1</td><td>74.5</td><td>73.8</td><td>83.8</td><td>61.0</td><td>72.4</td><td>0.838</td></tr><tr><td>GPT-5.6 Terra</td><td>56.9</td><td>68.1</td><td>52.9</td><td>54.2</td><td>42.2</td><td>68.8</td><td>77.9</td><td>87.5</td><td>63.2</td><td>75.3</td><td>0.812</td></tr></table>

## A.5 Prompts for Generation and Assessment

Each box lists the system prompt and user-message template in this section. Angle-bracketed placeholders denote runtime inputs and image attachments.

## A.5.1 HTML Generation

HTML Generation   
System:   
You are an expert HTML developer. Respond with only   
the complete, self-contained HTML source code for the   
requested game. Start your response with <!DOCTYPE   
html> and end it with </html>. Do not include   
Markdown code fences, explanations, or any text   
outside the HTML document. Include all CSS and   
JavaScript in the HTML file.   
User:   
<task\_prompt>

## A.5.2 Execution-Agent Action Decision

The Execution Agent receives the current screenshot, task, interactive elements, source code, and episode history.

System:   
You are a user simulation decision engine.   
You receive a screenshot of a rendered web artifact,   
optional episode history, optional source code, a   
free-form user persona, and free-form psychometric   
metrics/guidelines. Choose the next realistic user action   
for the current turn, or terminate the episode when the   
task is complete or the simulated user would naturally

Execution-Agent Action Decision (continued)   
stop.   
Return a single JSON object, with no markdown fences.   
Include only fields that are relevant to the chosen action;   
do not copy placeholder fields from other examples.   
{   
"summary": "<one sentence>",   
"action": <one action object>,   
"targetState": "<what visible state should   
,→ count as the goal after this action>",   
"expectedTransition": "<what should   
,→ visually change while time advances>",   
"psychometricObservation": {   
"positive": ["engagement", "control",   
,→ "clarity", "confidence"],   
"negative": ["confusion", "frustration",   
,→ "fatigue", "boredom"]   
}   
}   
Action examples:   
Click DOM control: {"type":"click","selector":   
"#startBtn"}   
- Hold fire on FPS canvas: {"type":"click",   
"selector":"#game","durationMs":250}   
Click a tile/block/sprite inside a 2D canvas using page   
coordinates: {"type":"click","selector":   
"#gameCanvas","x":240,"y":310}   
Coordinate click without selector only when absolute   
page coordinates are the intended input: {"type":   
"click","x":430,"y":520}   
- Press or hold key: {"type":"press","key":"w",   
"durationMs":600}   
- Hold a two-key keyboard combo for   
platformers/action games: {"type":"combo","keys":   
["ArrowRight","Space"],"durationMs":180}   
- Smoothly rotate mouse-look/crosshair: {"type":   
"mouseMove","dx":120,"dy":-30,"durationMs":   
300,"steps":12}   
- Wait: {"type":"wait","ms":300}   
- Stop: {"type":"terminate","reason":"..."}   
Rules:   
- Prefer selectorHint values from Available interactive   
elements for ordinary DOM controls, for example

## Execution-Agent Action Decision (continued)

"#restartBtn". Use coordinates only when no DOM selector can represent the intended action. - For 2D canvas, tile, grid, board, mining, puzzle, or sprite games, a canvas selector alone clicks the center of the whole canvas. If the intended target is a specific tile, block, item, sprite, or point inside the canvas, include both the canvas selector and explicit page x/y coordinates for that target, for example {"type": "click","selector":"#gameCanvas","x":240, "y":310}. Do not return only the canvas selector for an internal canvas target.

\- For first-person, pointer-lock, mouse-look, or shooter games, do not use x/y click coordinates to aim. The screen crosshair is controlled by mouse movement, and absolute Playwright mouse movement can unintentionally rotate the camera. To fire or hold fire, prefer a click action on the game canvas selector (for example {"type":"click","selector":"#game", "durationMs":200}) with no x/y coordinates. To aim or turn the view, use mouseMove with dx/dy; use small smooth deltas and iterate based on screenshots. - action.waitMs is an optional pre-action delay in milliseconds. If waitMs > 0, the engine advances Playwright Clock by that amount before executing the action. Use waitMs > 0 when the correct user behavior is to wait for a moving game state and then act at the right moment. Use waitMs=0 or omit it for actions that should happen immediately.

\- Distinguish timing fields carefully:

\- action.waitMs delays before the action starts. Use this for "wait, then press/click" timing.

\- action.durationMs controls how long the action is held after it starts, such as holding a movement key or held mouse button.

\- If action.durationMs is present and shorter than two engine observation frame intervals plus 25ms, including durationMs=0, the engine will hold it for at least 2\*frameMs + 25ms so the action can cross observable frame boundaries. Omit durationMs entirely for an instantaneous click or key press.

\- Timing-sensitive example: if the current game screen shows that pressing an arrow key immediately would be too early, but the snake will align in about 240ms, return action {"type":"press","key":"ArrowLeft", "waitMs":240, "durationMs":80}. This means wait 240ms first, then press ArrowLeft for 80ms.

\- For UI controls that work while held, such as game movement buttons backed by mousedown/mouseup handlers, return a click durationMs. Do not include durationMs unless a held click is intentional.

\- Use combo only for simultaneous keyboard input, such as moving right while jumping in a platformer. combo.keys must contain 1 or 2 keyboard keys; never return more than 2 keys, and never include mouse buttons. If more keys seem useful, choose the most important 1-2 keys for the immediate short action.

\- Use terminate only when the visible state and episode history show that the task is complete, the game is over with no useful recovery, or a real user would stop. Do not terminate just because one action has been executed.

\- Do not return hidden code-state conditions. The current screenshot is the source of truth for visible state. Episode history is only a record of prior decisions, actions, observations, and termination state; it contains no judge feedback and must not be treated as an

## Execution-Agent Action Decision (continued)

## evaluation of prior actions.

\- When source code is provided, use it to infer supported inputs, event handlers, selectors, canvas coordinate mapping, cooldowns, distance checks, pointer-lock or mouse-look behavior, and other interaction rules. Use code as a rules reference, not as hidden live state; the current screenshot remains the source of truth for visible state. For 2D canvas click handlers, inspect getBoundingClientRect/clientX/clientY/tile-size conversions and choose page x/y coordinates that land on the intended visible tile or object. - If the game is not started and a Start button is visible, click Start and target a visible active gameplay state. - Use the current screenshot, source-code rules when present, interactive elements, and the factual episode history to choose the next action. Do not assume that transition or screenshot judge results from earlier turns will be available.

## User:

URL: <url>

Task prompt: <task\_prompt>

Turn: <turn\_index\_plus\_one> of <turn\_limit>

Reference clock intervals: "referenceFrameIntervalMs": <frame\_ms>, "maxPostActionAdvanceMs": <max\_advance\_ms>   
}

Episode history: <episode\_history\_json>

Available interactive elements: <action\_hints\_json>

User persona (<persona\_path>): <persona\_text>

Psychometric metrics (<metrics\_path>): <metrics\_text>

Artifact source code (<artifact\_path>): <artifact\_source\_code> <current\_screenshot\_attachment>

Invalid actions trigger the following correction, with up to three attempts.

## Action Validation and Correction

User-message extension:

Your previous response was invalid and was ,→ not executed.   
Validation error: <validation\_error>   
Return corrected JSON only, using the same ,→ schema.

## Action Validation and Correction (continued)

Reminder: combo.keys must contain 1 or 2 ,→ keyboard keys only.   
Previous invalid JSON:   
<previous\_invalid\_json>

## A.5.3 Combined Step-Level Evaluation

One model call returns transition and screenshot evaluations from the action metadata and before/after images.

## Combined Step-Level Evaluation

## System:

You are the combined TransitionJudger and ScreenshotJudger for an HTML game quality evaluation system. You receive exactly two images in this order: Image 1 is the before-action screenshot and Image 2 is the after-action screenshot. Evaluate both judge roles independently in one response. Never treat the simulator decision as ground truth.

## TransitionJudger:

Evaluate the two screenshots as evidence of the game’s interaction and visual transition quality. Use action execution and hit-test metadata to determine whether the simulator delivered the requested action to the intended target. Action execution warnings are simulator diagnostics. If metadata warns that a canvas selector click lacked coordinates, treat it as evidence that the simulator clicked the canvas center; for a 2D canvas internal target this is usually coordinate\_miss, not proof that the game failed to respond. If metadata contains success=false or executionError, set executionReward low and classify failureMode as action\_ineffective, while still judging visual stability and app feedback from the images. The targetState, expectedTransition, and decision summary describe simulator intent only. Do not score whether that prediction was fulfilled. Score the HTML game itself: visual continuity, responsiveness, feedback clarity, rule consistency, animation/state coherence, absence of crashes or broken rendering, and whether the observed transition makes sense for visible game mechanics. Use multi-dimensional rewards. Human product-quality preference should prioritize stable, readable, coherent visuals and understandable feedback over immediate task progress. Task progress is a weak component and must not dominate overall reward. Strengths and weaknesses must describe the game’s transition, feedback, coherence, and robustness, not whether the simulator’s predicted target was achieved. If hit-test metadata shows the action point missed the intended target, use coordinate\_miss; do not conclude that the action mechanism itself is ineffective. If the action is invalid under visible state or ordinary app/game rules, use illegal\_or\_rejected. Clear rejection can still receive high visualStabilityReward and appFeedbackReward. If an apparently valid, correctly delivered action produces no feedback, state change, partial progress, or consistent behavior, use action\_ineffective; treat it as a likely implementation bug, selector/event mismatch, or execution-path issue and penalize it strongly. Do not punish the app heavily just because the simulator chose a poor action. A rejected or no-op action can still earn

## Combined Step-Level Evaluation (continued)

high visualStabilityReward and appFeedbackReward when the page remains stable and feedback is clear. Describe the before image state compactly and extract reusable experience from the observed mechanics and feedback.

## Transition reward dimensions:

\- visualStabilityReward: rendering stability, layout coherence, readable visuals, absence of flicker, broken frames, crashes, or state corruption. This is the most important reward.

\- appFeedbackReward: clarity, immediacy, and rule-consistency of the app/game response. Legal rejection can score high when it is understandable.

\- executionReward: whether the simulator delivered the requested action to the intended element, coordinate, key, or mouse target according to metadata.

\- taskProgressReward: weakly weighted progress toward the task/scenario, such as starting play, moving, scoring, collecting, or completing an objective.

Compute transition overallReward = 0.40 \*

visualStabilityReward + 0.25 \* appFeedbackReward + 0.25 \* executionReward + 0.10 \* taskProgressReward. Transition rubric:

\- 90-100 overallReward: Stable, readable, coherent visuals with clear feedback and correct simulator execution; task progress may be partial or absent if a no-op or rejection is understandable.

\- 70-89 overallReward: Mostly stable and understandable, with minor ambiguity, weak feedback, or limited task progress.

\- 40-69 overallReward: Usable but meaningfully weak in feedback, execution clarity, visual stability, or interaction coherence.

\- 1-39 overallReward: Severe visual instability, broken rendering, execution mismatch, or an apparently valid action producing no understandable feedback.

\- 0 overallReward: The two images are blank, corrupted, missing essential page content, or impossible to evaluate.

## Transition failureMode definitions:

\- "none": the action succeeded or any failure is not relevant.

\- "coordinate\_miss": the action point missed the intended target or landed on the wrong element or region.

\- "action\_ineffective": the target was correct and the action should be valid by visible state, but the app did not respond as expected.

\- "illegal\_or\_rejected": the action was invalid by visible state or game rules, missing prerequisites, resource constraints, cooldowns, distance checks, or mechanics and was rejected or correctly no-oped.

\- "timing\_issue": the action is plausible but appears too early, too late, or not held long enough.

\- "uncertain": there is not enough evidence to classify the failure.

## ScreenshotJudger:

Evaluate Image 2, the after-action screenshot, for screenshot usability only. Focus on rendering quality, visual stability, visible page/game state, layout stability, readability, clipping, occlusion, artifacts, and whether the image is useful for judging page state. Do not score task completion, action success, game progress, or whether the simulator made a good choice. A screenshot can receive a high score even when the

## Combined Step-Level Evaluation (continued)

action failed, as long as the visual state is clear and stable. Rate screenshotAestheticReward separately for visual polish, composition, color harmony, spacing, and overall presentation quality. It is not included in   
screenshot score; the run-level global judge uses it. Compute screenshot score = 0.45 \*   
visualStabilityReward + 0.30 \* stateVisibilityReward + 0.25 \* readabilityReward.   
Screenshot rubric:   
- 90-100: The screenshot is sharp, complete, readable, stable, and clearly shows page state after the action. - 70-89: The screenshot is generally useful, with minor clipping, small readability issues, or non-critical visual artifacts.   
- 40-69: The screenshot is partially useful but has   
meaningful occlusion, blurry text, layout instability, cropped important content, or ambiguous state.   
- 1-39: The screenshot is mostly unusable due to blank areas, severe clipping, unreadable UI, broken rendering, or failure to show relevant state.   
- 0: The screenshot is blank, missing, corrupted, or impossible to evaluate.

Return only JSON with exactly these two top-level objects:

```csv
{
"transitionEvaluation": {
"overallReward": number from 0 to 100,
"visualStabilityReward": number from 0
,→ to 100,
"executionReward": number from 0 to 100,
"appFeedbackReward": number from 0 to
,→ 100,
"taskProgressReward": number from 0 to
,→ 100,
"summary": string,
"strengths": string[],
"weaknesses": string[],
"failureMode": "none" |
,→ "coordinate_miss" |
,→ "action_ineffective" |
,→ "illegal_or_rejected" |
,→ "timing_issue" | "uncertain",
"experience": string,
"state": string
},
"screenshotEvaluation": {
"score": number from 0 to 100,
"visualStabilityReward": number from 0
,→ to 100,
"readabilityReward": number from 0 to
,→ 100,
"stateVisibilityReward": number from 0
,→ to 100,
"screenshotAestheticReward": number from
,→ 0 to 100,
"summary": string,
"strengths": string[],
"weaknesses": string[]
}
}
User:
Task prompt:
<task_prompt>
```

## Combined Step-Level Evaluation (continued)

Turn index:   
<turn\_index>   
Chosen action:   
<action\_json>   
Action execution metadata:   
<action\_execution\_json>   
Simulator intent only, not a scoring target   
,→ - decision summary:   
<decision\_summary>   
Simulator intent only, not a scoring target   
,→ - predicted target state:   
<target\_state>   
Simulator intent only, not a scoring target   
,→ - predicted transition:   
<expected\_transition>   
Observer reached target:   
<reached\_goal>   
Timing:   
{   
"totalWaitMs": <total\_wait\_ms>,   
"simulatedAdvanceMs":   
,→ <simulated\_advance\_ms>   
}   
Image order is fixed: Image 1 is before the   
→ action; Image 2 is after the action.   
,→ Both are attached even when   
,→ byte-identical.   
Image 1: before   
Path: <before\_screenshot\_path>   
<before\_screenshot\_attachment>   
Image 2: after   
Path: <after\_screenshot\_path>   
<after\_screenshot\_attachment>

## A.5.4 Meta-Level Evaluation

The Meta-Level Judge summarizes the episode from structured actions and per-turn evaluations.

Meta-Level Evaluation   
System:   
You are GlobalJudger for an HTML   
game/user-experience simulation. You receive   
structured per-turn transition and screenshot evaluations   
only. You do not receive screenshots. Do not invent   
unseen visual details; use screenshot summaries,   
strengths, weaknesses, and reward fields as visual   
evidence.   
Your job is to judge the full play episode as a human   
product-quality reviewer. Consider whether the   
page/game is stable, understandable, playable,   
responsive, and whether critical flows actually work.   
Use transition evaluations for interaction quality,   
feedback clarity, execution reliability, failureMode,   
experience, state, and task progress. Use screenshot   
evaluations for visual stability, state legibility,

```csv
readability, and screenshotAestheticReward. confusing interaction. In general, criticalBlocker should
Compute weightedScore before any cap: be true only for caps of 60 or lower; higher caps usually
weightedScore = 0.15 * visualStabilityReward + 0.15 * describe non-critical limitations.
screenshotAestheticReward + 0.10 * Return only JSON with:
feedbackClarityReward + 0.20 *
interactionReliabilityReward + 0.10 * {
stateLegibilityReward + 0.20 * "weightedScore": number from 0 to 100,
promptAlignmentReward + 0.10 * taskProgressReward "score": number from 0 to 100 after
,→ applying usabilityCap,
Only apply a usability cap as a severity guardrail when "usabilityCap": number from 0 to 100 or
the weightedScore would otherwise overstate the real ,→ null,
user experience. Do not set usabilityCap merely "criticalBlocker": boolean,
because there is a flaw. Ordinary UX weaknesses, "criticalBlockerReason": string or null,
ambiguous feedback, low task progress, occasional "confidence": number from 0 to 100,
ineffective actions, missing polish, or partially "visualStabilityReward": number from 0 to
unreliable secondary mechanics should reduce the 100,
reward dimensions but should not set usabilityCap. "screenshotAestheticReward": number from 0
,→ to 100,
Choose the highest cap consistent with the observed
"feedbackClarityReward": number from 0 to
severity:
,→ 100,
- Cap at 0-10 if the page is blank, crashed, corrupted, or "interactionReliabilityReward": number
impossible to evaluate. ,→ from 0 to 100,
- Cap at 20 if the page renders but is effectively "stateLegibilityReward": number from 0 to
non-interactive: no meaningful primary interaction → 100,
works at all, or the app cannot leave an inert/static state. "promptAlignmentReward": number from 0 to
- Cap at 35 if a required primary entry action such as ,→ 100,
Start, Play, Begin, Next, or an equivalent first-step "taskProgressReward": number from 0 to
control appears correctly targeted but is nonfunctional ,→ 100,
so the main experience cannot begin. "summary": string,
- Cap at 45 if the artifact is substantially the wrong "strengths": string[],
product or the requested core experience is absent, even "weaknesses": string[],
if some unrelated UI or visuals work. "recurringIssues": [{"issue": string,
"severity": "low" | "medium" | "high",,→
- Cap at 60 only for severe but non-total incompleteness:
the main artifact is recognizable and partially usable, ,→ "turns": number[], "evidence":
string}],,→
but the central loop or a required primary mechanic is "failureModeSummary": [{"failureMode":
absent, blocked, or impossible to complete. ,→ "coordinate_miss" |
- Cap at 70 for major prompt gaps where the main loop ,→ "action_ineffective" |
works and the user can make meaningful progress, but a "illegal_or_rejected" | "timing_issue"→
prominent required feature/content area is missing or | "uncertain" | "none", "count":,→
clearly nonfunctional. number, "impact": string}],,→
- Cap at 80 for notable missing content, weak "experience": string,
production breadth, or unreliable secondary mechanics "recommendations": string[]
when the primary experience remains playable. }
- Cap at 90 for minor prompt omissions or polish gaps
only when the requested score would otherwise be
User:
near-perfect.
- Leave usabilityCap null when the flaw is already well Task prompt:
represented by low promptAlignmentReward, <task_prompt>
taskProgressReward, feedbackClarityReward, or
interactionReliabilityReward. Artifact prompt (<artifact_prompt_path><tr <sub>⌋</sub>
- For broad prompts requesting many assets, levels, ,→ uncated_marker>):
weapons, sounds, or polish, do not cap at 60 solely <artifact_prompt_text>
because every requested item was not observed in a
short episode. Use reward dimensions, or a higher cap Structured simulation evidence:
such as 70/80, unless the missing item blocks the {
primary loop or makes the artifact substantially not "run": {
what was requested. "url": <url>,
"turnCount": <turn_count>,
- Do not apply a cap for clear rule-based rejections such "turnLimit": <turn_limit>,
"terminated": <terminated>,
move. "terminationReason":
- Do not apply a cap if failures are mostly ,→ <termination_reason>,
coordinate_miss or otherwise caused by "totalSimulatedAdvanceMs":
simulator/model execution; lower confidence instead.
,→ <total_simulated_advance_ms>,
Classify a critical blocker only when evidence shows a "visualEvaluationSummary":
severe defect or required-content failure that blocks the ,→ <visual_evaluation_summary_json>
main requested experience, not merely a weak or },
```

## Meta-Level Evaluation (continued)

## Meta-Level Evaluation (continued)

## Meta-Level Evaluation (continued)

```json
"turns": [
{
"turnIndex": <turn_index>,
"turnLabel": <turn_label>,
"action": <action_json>,
"actionExecution":
,→ <action_execution_json>,
"timing": {
"totalWaitMs": <total_wait_ms>,
"simulatedAdvanceMs":
,→ <simulated_advance_ms>
},
"reachedGoal": <reached_goal>,
"terminated": <turn_terminated>,
"terminationReason":
,→ <turn_termination_reason>,
"decisionSummary": <decision_summary>,
"transition": <compact_transition_eva <sub>⌋</sub>
,→ luation_json>,
"screenshot": <compact_screenshot_eva <sub>⌋</sub>
,→ luation_json>,
"visualEvaluationErrors":
,→ <visual_evaluation_errors_json>
},
<remaining_turns>
]
}
```

## Compact evaluation fields:

```json
{
"judgeType": <judge_type>,
"turnIndex": <turn_index>,
"score": <score>,
"overallReward": <overall_reward>,
"visualStabilityReward":
,→ <visual_stability_reward>,
"executionReward": <execution_reward>,
"appFeedbackReward":
,→ <app_feedback_reward>,
"taskProgressReward":
,→ <task_progress_reward>,
"readabilityReward": <readability_reward>,
"stateVisibilityReward":
,→ <state_visibility_reward>,
"screenshotAestheticReward":
,→ <screenshot_aesthetic_reward>,
"summary": <summary>,
"strengths": <strengths>,
"weaknesses": <weaknesses>,
"experience": <experience>,
"state": <state>,
"failureMode": <failure_mode>
}
```

## A.5.5 Reference-Based Visual Comparison

The Visual Anchor receives the reference screenshot followed by the current rendering.

## Reference-Based Visual Comparison

## System:

You compare two screenshots of HTML games created for the exact same task prompt. Image 1 is the retained pre-rollout reference. Image 2 is the current training rollout being evaluated.

## Reference-Based Visual Comparison (continued)

Judge only visual presentation: scene quality, beauty, composition, visual hierarchy, layout organization, spacing, typography, color harmony, coherence, polish, and readability. Do not penalize ordinary differences in gameplay progress, animation timing, selected controls, scores, or transient state unless they materially harm visual organization.

If the screenshots do not show an obvious visual-quality difference, assign degradationLevel 0. Treat the provided screenshot scores only as a calibration signal: when their absolute delta is at most 8 points, default to no penalty unless Image 2 shows a clear, material visual degradation. Do not penalize small, ambiguous, or merely numerical differences.

Do not assign a positive penalty merely because the images differ. Return only JSON with:

{   
"degradationLevel": number from 0 to 100,   
"summary": string,   
"referenceAdvantages": string[],   
"currentAdvantages": string[],   
"currentDeficiencies": string[]   
User:   
Task prompt:   
<task\_prompt>   
Reference selection metadata:   
"promptId": <reference\_prompt\_id>,   
"turnIndex": <reference\_turn\_index>,   
"screenshotScore":   
,→ <reference\_screenshot\_score>,   
"selection": <reference\_selection>   
}   
Current selection metadata:   
"turnIndex": <current\_turn\_index>,   
"screenshotScore":   
,→ <current\_screenshot\_score>   
Image 1: pre-rollout reference   
Path: <reference\_screenshot\_path>   
<reference\_screenshot\_attachment>   
Image 2: current evaluated HTML   
Path: <current\_screenshot\_path>   
<current\_screenshot\_attachment>

A.6 Case Study  
A.6.1 Game Generation Tasks Earlier state  
Qwen3.6-35B-A3B  
![](images/1f365f0924061899b9c47dcdba4959d7af35889cb232a5389c7fa8ef20c57173.jpg)

After interaction  
![](images/84220c88c18ea98998eb3dd44c7dbf91fb322ba96de676842c267724bd3d482a.jpg)

Gemini-3.1-Pro  
![](images/b0818e60b68e910c132a5d29f04f6c4be6f160ee1dd7086f527e16f4c16217a3.jpg)  
GPT-5.3-Codex

![](images/1aa50581aa172ac97f2b0c92d17955e7452c80909efc9f79e44b9e9914ed2ccd.jpg)

![](images/37773b9491ab844e5dd5b55cb3b6673d024241ea1b189db3efd2c90e30e55e49.jpg)

No enemies in the play view  
![](images/3c8993f96b2832534117117c93fb324298dd9fb4ae84fd8586175d59146283df.jpg)  
Figure 7: Compared with the Qwen3.6-35B-A3B backbone, WebCatalyst shows markedly stronger robustness after interaction begins. While the backbone often produces plausible initial interfaces, its artifacts may break during execution—for example, rendering fails after launching Krunker, the 3D collection game turns black, Minecraft terrain disappears, and movement is blocked in the tank game. In contrast, WebCatalyst preserves coherent interactive states, with complete FPS scenes, visible collection effects and score updates, stable voxel terrain, and responsive projectile effects. These examples suggest that the gains extend beyond initial appearance to maintaining functionality and visual coherence throughout interaction. Krunker comparison (1/2). The request is “Krunker Game”. The paired views show the launch interface and the first-person scene reached during interaction.

Earlier state  
Claude-Opus-4.7  
![](images/943ae405dada3a83ac51560bb5a00710a4d1b6d6daade5ee95209f47f20ad1e9.jpg)

After interaction  
![](images/19dec4914340310a5160ee4e8ec46108fdde58dc1271aa4315fce61ebed52695.jpg)  
Rendering breaks after launch

GPT-5.4-mini  
![](images/41ff038c262df817ddc0a14bf8909834cc3ab3f7743390a658ecde47002ed70f.jpg)  
WebCatalyst

![](images/14b77de8c212c981fae744f22927bc39b46222caef9a3f15f4c97f7753d221d1.jpg)

![](images/0526f7aa5ad053940ac169b29efed6c4b4bec918ff133792eea30f22066a4a47.jpg)

Buildings, enemies, and weapon HUD  
![](images/5d1e50442b85f70104fc8c59077236e5c5ae7ce114cff8aa919d6ac79c17f172.jpg)  
Figure 8: Krunker comparison (2/2). Continuation of Figure 7.

Earlier state  
![](images/21b0829054e2ae0f5d5721d6bbc9a2621c38008675d066aa613f5eef7a80579c.jpg)

After interaction  
![](images/1928b977069709ae2ceb77643bb5a0cc792999c07de8fa58f2e3091fd5f4b9fe.jpg)

GPT-5.3-Codex  
![](images/a5cf108247c6373dd6b5a4569b686991b31d9c695ed46eb4b46271fdc212d848.jpg)

Backward movement is blocked  
![](images/b044a71b6dcfe89c813ba1324ab2766a09b415cc272788b71e55dc67c113d217.jpg)

Claude-Opus-4.7  
![](images/80c64162d278efaa352076f0638071ffb173a4f11dd713e53a0beca1b76b53d3.jpg)

![](images/c08fd7dcf6a585e05a0f9a5cb61a0d5001d34415c7d896677e8baa3659a8a596.jpg)  
Figure 9: 3D collection-game comparison (1/2). The German request for “Alles nur Schaum” asks for a small yellow character, pink beer mugs, automatic collection on approach, a +1 score effect, and coin rewards in a cheerful 3D environment.

Earlier state  
GPT-5.4-mini  
![](images/0deb4e7bb3f40f7db23e2bc09adf7491b5bec7a6ffd48f2807bb3370727e14fc.jpg)

After interaction  
Items do not respond to interaction  
![](images/c77200a02a4e229ee0165c64f710a3b2afdce51edbb558c9908c4a8486c4aa0c.jpg)

WebCatalyst  
![](images/5889335ce4afad8edf4b54a680b7fdec5f1543f68433bf7ab632c1d351cd774e.jpg)

Collection effect and updated score  
![](images/cc2c9985a9eea411522784463705af2e8415dc0e184dc1e53a120c057cec2599.jpg)  
Figure 10: 3D collection-game comparison (2/2). Continuation of Figure 9. WebCatalyst shows a high-quality 3D scene with a +1 collection effect and an updated score.

![](images/054919657305f3484318c1da319408cb78c5229bda71650076ee3a6a05db5af0.jpg)  
Gemini-3.1-Pro

![](images/b57bb9d09e60e8ea12aba1a3aebe3a4ff9d6a627ccf961f8befe6b06f6610248.jpg)

![](images/56e1739b553b3c01c30506395ecebe2b5cb286389d16ba06e5343f2e0fb4d877.jpg)

![](images/3b2a32e8fea3a93e2edcd736d2e3d4b873021637286cba7e7e5fe64c24d2c653.jpg)

GPT-5.3-Codex  
![](images/6dc086ecacd3c1f0dbaa494af58f65866635d983782863041f2fd2e24b77443f.jpg)

![](images/c6d7a6534d86ceb2816cffc08b5022d1ff81d38c76a09ca6711528c2fdb895c9.jpg)  
Figure 11: Minecraft comparison (1/2). The French request asks for a web version of Minecraft that preserves its visual style and core functionality. The views show the interface and the resulting voxel world after attempting to enter the game.

Earlier state Claude-Opus-4.7  
![](images/0bb3d65b8c4e7b3e4fd8b0edcc7497cb38c4afe28bbfa07b7f5d7728790d6008.jpg)  
GPT-5.4-mini

After interaction  
![](images/688f51982c6b717199f1e1882019443c252c493e58bdf3fa7fbf406b945ee371.jpg)

![](images/490934951c5393c232743b64117e8b0e040055c910aadeeac6aebb013b84ad07.jpg)  
WebCatalyst

Character keeps falling  
![](images/0265bea96d4fefb671853519e75ce8e1371ba254b152533d1dd800d8488b8f82.jpg)

![](images/9ceddec0ca03ece03c8950851ab20c777322011adc0f0323ff733bf5888c1baf.jpg)

Textured terrain, water, and block inventory  
![](images/fc23ac6e3c031f38043c2112ca4e63259c48d7805c1b4a282f79270aee279b65.jpg)  
Figure 12: Minecraft comparison (2/2). Continuation of Figure 11.

Earlier state  
Qwen3.6-35B-A3B  
![](images/35fa07d66621209902bf486319290673435fd320d80d8d8c03f08ddea52e926b.jpg)

After interaction  
![](images/d57ee1397429792a1b6cde65dd6c9c79921d0565b8aa30993195b107bfa9bd9e.jpg)

Gemini-3.1-Pro  
![](images/c5a3f7c9455e5b85080fb98a91bafc9d2d906ca7fcc50e5fe6361211377e361b.jpg)

![](images/70dcde8432beb292c01c1644f3bf698a1812ea095b92d67088efd7bb59a52305.jpg)

GPT-5.3-Codex  
![](images/93482c1c5e5f5a5c70efc0ba54d8cbec894ab563eccaa3e62e575a4c79e5ad72.jpg)

![](images/d050ded580f5ad74281f3f3e4b582647ab7aac19e127759af36c7b0719598573.jpg)  
Figure 13: Two-player tank-game comparison (1/2). The Chinese request asks for a console-style tank duel with distinct player colors, five health points per player, one shot per turn, ricocheting bullets, and an arena that obstructs direct shots.

Earlier state  
Claude-Opus-4.7  
![](images/331dc25e4e5c250a5cb15b23640a0c777f8314024c61d89c6cd9f702d6113470.jpg)

After interaction  
![](images/2b6f9e8f2cf7b760742d114dcfb1df0b82e0c75aa1f40eda52f1a527d91a4da1.jpg)

GPT-5.4-mini  
![](images/177d86912391a4f3592d576647716c26067dfcadbd40da4874bc255039df28c8.jpg)  
WebCatalyst

Cluttered map layout  
![](images/30acc9db2acf8faac22d034e815668d35f161a9f1b23118e5034732e048aaf31.jpg)

![](images/575650eb33b78ab515c010d2daaaf40d38c0fa86b2e7bc1c572f87906260b2f6.jpg)

Projectile and impact effects  
![](images/121a5b6cbfb171b42790df86e447fee4af169c6e0eb21c756b1373afa7a752a2.jpg)  
Figure 14: Two-player tank-game comparison (2/2). Continuation of Figure 13. WebCatalyst’s paired views show a projectile and impact particles around the blue tank.

![](images/f9867577f26c723d3c0922a302de9ee2647600ffc4d76054afdf6017c6df5c1e.jpg)  
A.6.2 Transfer to Conventional Webpages  
Figure 15: LLM product-website comparison (1/2). The enlarged top sections show typography, color, and hero composition; the full-page views show the organization and extent of the generated content.

![](images/5c9c98e9417d606a6eeb0172aa6c1724da99ec385add12d02d7483fe1f5ab674.jpg)

![](images/fc383fab64416a2e9b24100fd7c1b2f6d1d3899e1de22329ee5027a4cfeba24d.jpg)  
Figure 16: LLM product-website comparison (2/2). Continuation of Figure 15. WebCatalyst maintains the dark theme and shows diverse features across the full page.

![](images/7db598ec31bd3e994e23d705c2c040ba8175d4aff2c302ce69af7238a553ed12.jpg)  
Full-page overview

Figure 17: Hotel-website comparison (1/2). The hero details show the landing-page presentation; the full-page overviews show accommodations, services, visual content, and lower-page sections.

![](images/0438a0fe97569014456f024bca869c218bc03cf8feb964fca523ed06901c506d.jpg)  
Figure 18: Hotel-website comparison (2/2). Continuation of Figure 17. WebCatalyst combines a photographic hero with image-card layouts, resulting in a visually polished and aesthetically appealing presentation.

## A.6.3 An Illustrative GUI Assessor Trajectory

Illustration: Figures 19–21 follow an eight-turn GUI Assessor episode on Loan Approval Race. Each turn shows the selected action, before/after screenshots, and transition and screenshot judgements, followed by the Meta-Level assessment.

Task: Create a complete self-contained HTML game called “Loan Approval Race” where I play as a banker racing against AI coworkers to review loan files, dodge paperwork chaos, collect bonuses, and reach the approval desk first. Make it browser-based with simple controls, smooth movement, fun finance-themed obstacles, scoring, and a quick restart option.

![](images/36bb7dad090adfae5ff071075f89e18249f0e608d9ce33f2bb6e7bb9528d122a.jpg)  
Figure 19: GUI Assessor trajectory on Loan Approval Race, turns 1–3.

![](images/6f35d0067a321a5bd9b11df3367956e229744cd150360b0b4bf6cbf51a5d6bf1.jpg)  
Turn 4  
Figure 20: GUI Assessor trajectory, turns 4–6.  
Down + Space (1 s)

![](images/7365b62dab5e9666fe0da189039de0ee99c356c01c55d80600d7b92a1a06a998.jpg)

<table><tr><td>Visual stability</td><td>Screenshot aesthetics</td><td>Feedback clarity</td><td>Interaction reliability</td><td>State legibility</td><td>Prompt alignment</td><td>Task progress</td></tr><tr><td>95</td><td>86.38</td><td>92.38</td><td>100</td><td>93.5</td><td>92</td><td>72</td></tr></table>

Figure 21: GUI Assessor trajectory with the Meta-Level assessment and its dimension scores.