Codex (Harness) effort differs (≠)

IWC-Bench

# IWC-Bench: Evaluating Web Application Generation from a Software Testing Perspective

Chenxu Liu<sup>1,†</sup>, Zilu Zou<sup>1</sup>, Peizhong Gao<sup>2,1</sup>, Jiawen Tao<sup>3,1</sup>, Zhexin Zhang<sup>2,1</sup>, Guang Chen<sup>1</sup>, Haowei Lin<sup>1</sup>, Ying Zhou<sup>1</sup>, Tianyi Bai<sup>1</sup>, Dolly Deng<sup>1</sup>, Suncong Zheng<sup>1</sup> and Maxm Pan<sup>1</sup>

<sup>1</sup>Hunyuan Team, Tencent, <sup>2</sup>Tsinghua University, <sup>3</sup>Peking University {lewiscxliu, ziluzou, chillgao, jadentao, zhexinzhang, maxluxchen, cloudlin, lilithyzhou, tianabai, dollydeng, congzheng, maxmpan}@tencent.com

Human evaluation provides a direct measure of the quality of LLM-generated web applications. However, fitting human judgments through automated evaluation remains challenging. Static benchmarks can credit functionality that exists in source code but is unreachable at runtime. Interactive benchmarks exercise the application, yet incomplete exploration can cause them to miss implemented functionality and confound application defects with agent execution failures. To address these limitations, we propose IWC-Bench, an interactive benchmark for evaluating web application generation from a software testing perspective. IWC-Bench instruments each generated application and uses code coverage to guide an agent in exploring its functionality through user-simulated interactions. It then abstracts the interaction trace into a state-transition graph and evaluates the application along three dimensions: visual aesthetics, usability, and requirement alignment. By separating exploration from scoring, IWC-Bench collects runtime evidence without constraining exploration to predefined acceptance criteria. IWC-Bench comprises 369 real-world user requirements and 5,088 acceptance criteria. Evaluation of 16 frontier LLMs reveals distinct strengths across the three dimensions, with no model leading on every dimension. On 197 validated sessions sampled from an internal arena, IWC-Bench achieves 85.3% agreement with human preferences, with agreement generally increasing as the score diference between paired applications grows. Further experiments show that coverage guidance improves exploration coverage and the model rankings remain stable when the judge model is replaced.

<table><tr><td>IWC-Benich # Model</td><td>Effort</td><td>Score</td></tr><tr><td>1</td><td>Claude-Opus-5</td><td>xhigh ≠ +1.381</td></tr><tr><td>2</td><td>GPT-5.6-Sol Codex</td><td>xhigh +1.120</td></tr><tr><td>3</td><td>Qwen3.8-Max enabled</td><td>+1.029</td></tr><tr><td>4 Kimi-K3</td><td>max</td><td>+1.028</td></tr><tr><td>5</td><td>Claude-Opus-4.8</td><td>xhigh ≠ +0.818</td></tr><tr><td>6 Grok-4.5</td><td></td><td>high +0.809</td></tr><tr><td>7 GPT-5.5</td><td>Codex</td><td>xhigh +0.523</td></tr><tr><td>8</td><td>Claude-Opus-4.7</td><td>xhigh ≠ -0.206</td></tr><tr><td>9</td><td>GLM-5.2</td><td>max -0.257</td></tr><tr><td>10</td><td>DeepSeek-V4-Flash</td><td>max ≠ -0.290</td></tr><tr><td>11 HY3</td><td>high</td><td>-0.534</td></tr><tr><td>12</td><td>Qwen3.7-Max</td><td>enabled -0.849</td></tr><tr><td>13 GLM-5.1</td><td></td><td>enabled -0.883</td></tr><tr><td>14</td><td>DeepSeek-V4-Pro-Prev</td><td>max ≠ -1.067</td></tr><tr><td>15</td><td>Kimi-K2.7-Code</td><td>enabled -1.249</td></tr><tr><td>16</td><td>MiniMax-M3</td><td>adaptive -1.374</td></tr></table>

![](images/4c133d8db773330c35997020a91709dfbaaacc56040f65eb8ec9e79c95a0c8e3.jpg)  
Score scales: Bench [-1.5, +1.5] (zero tick); Code Arena [1450, 1750]. Bars use separate scales.

Code Arena
<table><tr><td>Effort</td><td>Model</td><td>Global [range]</td></tr><tr><td>max≠</td><td>Claude-Opus-5</td><td>3 [3-5]</td></tr><tr><td>enabled</td><td>Qwen3.8-Max</td><td>4 [3-8]</td></tr><tr><td>max</td><td>Kimi-K3</td><td>5 [3-8]</td></tr><tr><td>xhigh</td><td>GPT-5.6-Sol Codex</td><td>14 [9-16]</td></tr><tr><td>max</td><td>GLM-5.2</td><td>18 [16-21]</td></tr><tr><td>high ≠</td><td>DeepSeek-V4-Flash</td><td>20 [17-23]</td></tr><tr><td>high ≠</td><td>Claude-Opus-4.8</td><td>23 [21-26]</td></tr><tr><td>high</td><td>Grok-4.5</td><td>24 [20-26]</td></tr><tr><td>high ≠</td><td>Claude-Opus-4.7</td><td>26 [22-26]</td></tr><tr><td>enabled</td><td>Qwen3.7-Max</td><td>35 [32-42]</td></tr><tr><td>high</td><td>HY3</td><td>39 [32-44]</td></tr><tr><td>xhigh</td><td>GPT-5.5 Codex</td><td>40 [33-43]</td></tr><tr><td>enabled</td><td>GLM-5.1</td><td>41 [34-44]</td></tr><tr><td>adaptive</td><td>MiniMax-M3</td><td>47 [44-52]</td></tr><tr><td>enabled</td><td>Kimi-K2.7-Code</td><td>49 [47-56]</td></tr><tr><td>high ≠</td><td>DeepSeek-V4-Pro-Prev</td><td>53 [48-58]</td></tr><tr><td>higher</td><td>lower</td><td>same</td></tr></table>

Figure 1 | IWC-Bench and Code Arena rankings for 16 shared models. # gives shared-set rank; Global [range] gives Code Arena’s overall rank and reported range. Blue/red/gray links indicate higher/lower/unchanged shared-set rank. Long dashes mark the two GPT models using Codex on both sides; other models use Claude Code in IWC-Bench and a custom harness in Code Arena. Dots mark difering efort labels. Scores use separate scales; harness/efort can difer.

## 1. Introduction

Web applications run across platforms without installation, making them an accessible medium for LLM agents to deliver interactive functionality to users. Given a natural-language requirement, an agent iteratively invokes tools and generates the source code of a runnable application. However, the evaluation of generated applications is typically highly subjective. Human-preference leaderboards such as Code Arena (LMArena.ai, 2026) obtain user preferences directly by asking annotators to choose between two applications generated by diferent models for the same requirement. However, repeated human evaluation is costly, requiring an automated benchmark to reduce the cost of evaluation while preserving user preferences.

Existing automated benchmarks face two major limitations, making them less likely to align with user preferences. First, static benchmarks do not exercise the generated application. Static benchmarks based on source code or a single rendered screenshot (Sun et al., 2025; Liu et al., 2026a) assess the applications without verifying whether users can reach and use the intended functionalities. For example, a broken button on the entry page can block all downstream functionalities, while the corresponding code may still be present, causing the benchmark to credit functionalities that are unusable in practice. Second, interactive benchmarks struggle to distinguish application defects from agent execution failures. Interactive benchmarks define rubrics, reference answers, or contracts and ask an agent to validate them through interactions (Lu et al., 2026; Wu et al., 2025; Zhang et al., 2026b;a; Meng et al., 2026; Lei et al., 2026). However, a failed check may indicate that the application lacks the target functionality or that the agent failed to locate or trigger it. Errors in GUI interaction and evidence interpretation make these causes dificult to distinguish, introducing uncertainty into the final score.

To address these limitations, in this paper, we propose IWC-Bench<sup>1</sup>, an interactive benchmark for evaluating web application generation from a software testing perspective. IWC-Bench treats each generated application as software under test: it simulates user interactions and explores the application to collect runtime evidence. Code coverage serves two purposes in this process. It guides the exploration toward unexecuted code and quantifies how much of the implementation has been exercised. IWC-Bench separates exploration from scoring so that the exploration process is not constrained by acceptance criteria.

IWC-Bench consists of four stages. (i) Instrumentation. An automated tool instruments the JavaScript code in each generated application to collect runtime code coverage without manual adaptation. (ii) Coverage-guided exploration. An LLM-driven agent explores the application as a software tester. When coverage plateaus, a separate LLM analyzes the uncovered code and provides natural-language guidance for subsequent exploration. This feedback supplements the agent’s own assessment of what remains to be tested. (iii) State abstraction. A deterministic state abstraction approach processes each page and merges pages with equivalent normalized representations into the same state, generating a compact state-transition graph to represent the exploration trace. (iv) Scoring. LLM and agentic judges assess visual aesthetics, usability, and requirement alignment using the state-transition graph and the recorded traces, DOM snapshots, and screenshots.

We evaluate 16 frontier LLMs on IWC-Bench and validate the benchmark against human preferences from pairwise comparison samples. IWC-Bench achieves 85.3% pairwise agreement on 197 validated samples. The model-level results reveal complementary strengths in visual aesthetics, usability, and requirement alignment. Additional experiments show that coverage guidance improves exploration coverage and that model rankings remain stable across judge models. As shown in

Figure 1, IWC-Bench rankings are highly correlated with Code Arena rankings, with a Spearman rank correlation (Spearman, 1987) of $\rho = 0 . 8 7 6$ across the 16 shared models. The two evaluations show broadly consistent capability tiers at shared-set ranks 1–4, 5–9, 10–13, and 14–16. Despite diferences in reasoning efort and harness configurations, this alignment suggests that IWC-Bench captures capability distinctions consistent with the community preferences reflected in Code Arena.

In summary, this paper makes the following main contributions:

• We construct IWC-Bench, a benchmark with 369 real-world user requirements and 5,088 acceptance criteria for evaluating web application generation.

• We design a four-stage evaluation pipeline that separates exploration from scoring, guides exploration with code coverage, and organizes runtime evidence through state abstraction.

• We evaluate 16 frontier LLMs across visual aesthetics, usability, and requirement alignment. Validation shows 85.3% agreement with human preferences, stable rankings across judge models, reliable instrumentation, and improved exploration coverage.

## 2. Related Work

## 2.1. Benchmarks for Web Application Generation

Existing related benchmarks are shown in Table 1. Early static benchmarks formulate web generation as reproducing a reference design and measure pixel-, text-, or element-level similarity to a reference screenshot or implementation (Beltramelli, 2018; Bhathal & Gupta, 2025; Vu et al., 2025; Sun et al., 2025). Real-world requirements, however, often admit multiple valid implementations. WebCoderBench (Liu et al., 2026a) assesses source code and initial renderings without a unique reference implementation. These benchmarks support inexpensive, reproducible evaluation, but do not exercise the application and may credit functionality that is unreachable at runtime.

Interactive benchmarks execute applications in a browser, and score them using test suites, fixed action sequences, or staged screenshots (Xu et al., 2025; Zhu et al., 2025; Zhang et al., 2025), or ask agents to validate checklists, reference answers, contracts, or specifications (Lu et al., 2026; Zhang et al., 2026b; Meng et al., 2026; Zhang et al., 2026a; Wu et al., 2025; Lei et al., 2026). These benchmarks observe a broader range of runtime behavior and provide evidence for failure diagnosis. Predefined targets, nevertheless, constrain the verification scope, leaving behavior outside those targets unassessed. Failed agent-based checks can also reflect either application defects or agent failures.

More recently, Cookie-Bench (Yang et al., 2026) removes checklists, allows the agent to plan interactions from the user request, and scores applications after evidence collection. I-WebGenBench (Dai et al., 2026) applies standardized actions to enumerated components and records DOM changes. However, neither explicitly measures exploration adequacy. IWC-Bench introduces code coverage measurement into the exploration process to quantify the functionalities exercised and guide further exploration, while withholding acceptance criteria from the exploration agent.

## 2.2. Automated GUI Testing

Automated GUI testing exercises applications through GUI actions to gain coverage and reveal failures within a specific time budget (Liu et al., 2026c). Existing work studies state abstraction approaches at diferent granularities (Yandrapally & Mesbah, 2022; Liu et al., 2025b; 2026b) to avoid repeated exploration, while using random-based (Google, 2025), model-based (Mesbah et al., 2012), reinforcement-learning-based (Zheng et al., 2021), and LLM-based (Liu et al., 2024; 2025a) exploration strategies.

Table 1 | Comparison of IWC-Bench with existing benchmarks for web application generation.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Samples Interactive</td><td rowspan="2"></td><td rowspan="2">Exploration</td><td colspan="3">Judgment</td><td rowspan="2">Exploration-Scoring Exploration Separation</td><td rowspan="2">Quantification</td></tr><tr><td>Rules LLM Manual</td><td></td><td></td></tr><tr><td>FullFront (Sun et al., 2025)</td><td>400</td><td>X</td><td>None</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>WebCoderBench (Liu et al., 2026a)</td><td>1,572</td><td>X</td><td>None</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FrontendBench (Zhu et al., 2025)</td><td>148</td><td>√</td><td>Scripted</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Web-Bench (Xu et al., 2025)</td><td>1,000</td><td>√</td><td>Scripted</td><td>√</td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>ArtifactsBench (Zhang et al., 2025)</td><td>1,825</td><td>√</td><td>Scripted</td><td>X</td><td>√</td><td>X</td><td>× &gt;</td><td>X</td></tr><tr><td>WebDev Arena (LMArena.ai, 2026)</td><td>11</td><td>√</td><td>Human</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Design Arena (Design Arena, 2026)</td><td></td><td>√</td><td>Human</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>WebGen-Bench (Lu et al., 2026)</td><td>101</td><td>√</td><td>Checklist-guided</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FronTalk (Wu et al., 2025)</td><td>100</td><td>√</td><td>Checklist/autonomous</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>MiniAppBench (Zhang et al., 2026b)</td><td>500</td><td>√</td><td>Checklist-guided</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>WebCompass (Lei et al., 2026)</td><td>1,526</td><td></td><td>Checklist-guided</td><td>X</td><td>√</td><td>X</td><td>××</td><td></td></tr><tr><td>WebGameBench (Zhang et al., 2026a)</td><td>111</td><td>√</td><td>Specification-guided</td><td>X</td><td>√</td><td>X</td><td></td><td>××</td></tr><tr><td>WebRISE (Meng et al., 2026)</td><td>442</td><td>√</td><td>Contract-guided</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>I-WebGenBench (Dai et al., 2026)</td><td>201</td><td>√</td><td>Enumerative</td><td>√</td><td>√</td><td>X</td><td>△&gt;</td><td>X</td></tr><tr><td>Cookie-Bench (Yang et al., 2026)</td><td>1,000</td><td>√</td><td>Model-autonomous</td><td>X</td><td>√</td><td>X</td><td></td><td>X</td></tr><tr><td>IWC-Bench (ours)</td><td>369</td><td>√</td><td>Coverage-guided</td><td>X</td><td>√</td><td>X</td><td>√</td><td>√</td></tr></table>

Note: △ indicates partial satisfaction.

IWC-Bench adapts these approaches to evaluating LLM-generated web applications. Code coverage provides exploration feedback and quantifies the implementation exercised, leveraging the ideology of LLM-based GUI testing approaches. State abstraction groups near-duplicate pages (Yandrapally et al., 2020) into a compact state-transition graph, as part of the evidence for scoring.

## 3. Dataset

IWC-Bench contains 369 real-world user requirements, each paired with a checklist used exclusively for scoring.

## 3.1. Data Source

We collect our user requirements from production trafic on an enterprise-internal replica of Code Arena (LMArena.ai, 2026). These requirements retain characteristics of real-world usage that are dificult to reproduce synthetically, including colloquial expressions, incomplete specifications, and substantial variation in the expected scope of the application. Following WebCoderBench (Liu et al., 2026a), we first anonymize the requirements. Three annotators then review each requirement and use majority voting to exclude samples that are incomprehensible, depend on unavailable materials, or fall outside the web application scope. We remove lexical duplicates using MinHash (Broder, 1997) and semantic duplicates using MiniLM (Wang et al., 2020). The resulting dataset contains 369 text-only requirements. Our requirements have a mean length of 601 characters, a median of 105, and a range of 7–37,903. This distribution captures the coexistence of brief requests and detailed specifications in production trafic.

Table 2 | Dataset statistics for IWC-Bench.
<table><tr><td colspan="2">Artifact Complexity</td><td rowspan="2"></td><td colspan="2">Expression Style</td><td colspan="2">Application Category</td></tr><tr><td>L1 Highly Simple</td><td>48</td><td>S1 Technical</td><td>81</td><td>Entertainment</td><td>106</td></tr><tr><td>L2</td><td>Simple</td><td>161</td><td>S2 Colloquial</td><td>222</td><td>Scientific Demo</td><td>37</td></tr><tr><td>L3</td><td>Medium</td><td>120</td><td>S3 Role-playing</td><td>15</td><td>Online Office Suite</td><td>33</td></tr><tr><td>L4</td><td>Complex</td><td>27</td><td>S4 Analogy</td><td>51</td><td>Utility Website</td><td>29</td></tr><tr><td>L5</td><td>Highly Complex</td><td>13</td><td></td><td></td><td>Data Visualization</td><td>21</td></tr><tr><td></td><td>Requirement Clarity</td><td></td><td>Acceptance Criteria</td><td></td><td>Online Education</td><td>18</td></tr><tr><td>C1</td><td>Clear</td><td>135</td><td>Functional</td><td>2,706</td><td>AI-powered</td><td>15</td></tr><tr><td>C2</td><td>Intermediate</td><td>134</td><td>Content</td><td>1,447</td><td>Multimedia</td><td>15</td></tr><tr><td>C3</td><td>Vague</td><td>100</td><td>Visual</td><td>935</td><td>12 others</td><td>95</td></tr></table>

## 3.2. Requirement Characterization

Following the annotation protocol of WebCoderBench (Liu et al., 2026a), we characterize each requirement along four dimensions: artifact complexity, expression style, requirement clarity, and application category. An LLM assigns initial labels and provides a rationale for each, after which human annotators review and revise the annotations. Table 2 summarizes the results. Overall, 63.4% of requirements are not fully specified (�2 or �3), and 60.2% use colloquial language (�2). These characteristics require LLMs to interpret user intent from incomplete or informal descriptions. Entertainment and scientific demonstrations are the two largest application categories.

## 3.3. Acceptance Criteria

Real-world user requirements often admit multiple valid implementations. We therefore associate each task with a checklist, including three sets of acceptance criteria, with each acceptance criterion expressing a concise, high-level property that should be observable in the delivered application. The three sets of criteria correspond to three dimensions: functional, including features and interactions; content, including text, data, and specified resources; and visual, including layout, style, and required components. These dimensions jointly determine the requirement-alignment score (detailed in §4.4). Following WebCoderBench (Liu et al., 2026a), we use Claude-Opus-4.8, GPT-5.6-Sol, and Grok-4.5 to generate checklists independently. Claude-Opus-4.8 merges the checklists following the majority voting process, and human experts validate the resulting set. The checklists describe the required observable properties without prescribing implementation details or introducing requirements absent from the original request. The 369 requirements yield 5,088 acceptance criteria: 2,706 functional, 1,447 content, and 935 visual. Each task has 13.8 criteria on average, ranging from 2 to 77 according to the detail of the requirement.

## 4. Evaluation Process

The evaluation workflow of IWC-Bench contains four stages, which are instrumentation, exploration, state abstraction, and scoring (shown in Figure 2). The first three stages collect and organize runtime evidence without access to acceptance criteria, while the scoring stage leverages these criteria for a fair judgment.

![](images/33f91da83621725d3342a8863c03516de1ad5013ccdf0383b561386ae6813e60.jpg)  
Figure 2 | The evaluation workflow of IWC-Bench.

## 4.1. Instrumentation

In order to obtain coverage from various kinds of generated applications, we develop an automated instrumentation tool based on Istanbul (Istanbul Contributors, 2024), a mature and famous instrumentation tool for JavaScript. Our tool supports plain static HTML, Vite, Next, Create React App (CRA), and Astro, with framework-specific procedures applied automatically. With an application as input, our tool first detects its framework according to configuration files. For static frameworks, it runs Istanbul locally to instrument the application; for frameworks that require a build, it injects a plugin during the build and performs instrumentation after the build. After instrumentation, each JavaScript statement is associated with a counter that records every execution of the statement at runtime. During exploration, after each interaction, cumulative coverage and per-step gains can be calculated from the number of covered statement counters and the total number of counters. If instrumentation fails, the application remains in the evaluation and undergoes exploration and scoring in black-box mode. Appendix A.1 describes framework support and deployment procedures; §5.5 evaluates instrumentation reliability. The instrumented application is then deployed in a sandbox on a remote server and is accessible through a URL.

## 4.2. Coverage-Guided Exploration

In order to flexibly control the agent’s context during exploration, we develop an agent based on the AWorld (Yu et al., 2025) framework, interacting with the web applications using Playwright MCP (Microsoft, 2025). The agent reads a simplified DOM tree of the current page at each step to understand its content and retrieves only the tree dif between steps to save context space. The agent is prompted as a software tester to discover available functionalities and test boundary conditions. The agent has no access to source code, acceptance criteria, or reference answers.

During exploration, the harness monitors the coverage and detects a coverage plateau after three consecutive interactions without a coverage increase. It then extracts the uncovered code and invokes a separate LLM to produce natural-language suggestions. These suggestions are injected into the exploration agent’s context to guide further testing. Exploration ends when the agent declares testing complete or exhausts a 100-step budget. To conclude, in its final step, the agent summarizes the functionality, interactions, and observed failures. Coverage thus both quantifies code execution and directs subsequent exploration; §5.6 evaluates its contribution.

## 4.3. State Abstraction

Exploration traces, screenshots, and DOM snapshots can easily exceed the judge’s context window and obscure relevant evidence through repetition. We adopt the state abstraction practice from the area of automated GUI testing (Mesbah et al., 2012; Yandrapally & Mesbah, 2022) to organize observations into a compact state-transition graph. Each snapshot is normalized by removing script and style tags and mutable component attributes, and replacing timestamps and clock strings with fixed placeholders. Pages with identical normalized representations form one state, and user actions define transitions. Although simple and heuristic, this approach is demonstrated to be efective (Liu et al., 2025b). The constructed state-transition graph provides a straightforward and concise understanding of the application, serving as part of the evidence for scoring.

## 4.4. Scoring

IWC-Bench scores visual aesthetics, usability, and requirement alignment separately on a 0–100 scale.   
Appendix A.2 provides procedural details.

Visual aesthetics. This dimension assesses up to five representative screenshots collected during exploration. We filter blank and partially loaded frames using foreground and edge density, then apply farthest-point sampling to select visually diverse screenshots. The judges examine layout, information organization, typography, graphic quality, and polish.

Usability. This dimension assesses the experience of interacting with the application using the complete exploration trace and constructed state-transition graph. The judges examine successfully executed interactions and observed failures, including problems with feedback, navigation, state recovery, and fault tolerance.

Both dimensions use a three-round deliberative protocol: an advocate identifies strengths, a critic identifies defects, and a final judge verifies both assessments against the supplied evidence before scoring. The advocate and critic cite concrete observations without assigning scores. The judge adds seven dimension-specific adjustments to a 0–5 base score and clips the result to [0, 5]. Appendix A.2 specifies these aspects. To reduce sampling variance, the final judge runs independently five times with the first two assessments held fixed. Given scores $q _ { 1 } , \ldots , q _ { 5 } \in [ 0 , 5 ]$ , we discard the highest and lowest scores and map the remaining mean to 0–100:

$$
s = 2 0 \times { \frac { \sum _ { i = 1 } ^ { 5 } q _ { i } - \operatorname* { m a x } _ { i } q _ { i } - \operatorname* { m i n } _ { i } q _ { i } } { 3 } } .
$$

This protocol aims to improve score discrimination and reduce concentration at the scale endpoints;   
Appendix D compares it with single-round scoring.

Requirement alignment. This dimension judges the application behavior against the acceptance criterion. Since the complete set of evidence can overwhelm an LLM, we build a judging agent for this dimension. This agent starts with the state-transition graph to form an overall understanding of the application. It then carefully examines and searches through recorded evidence using a tool set, which supports regular-expression searches over the DOMs and traces with line-numbered results, paginated reading of DOMs and traces, screenshot inspection, and state-transition graph inspection. This retrieval process allows the agent to examine relevant evidence without loading the entire record into one context window. Once this agent determines that it has collected enough evidence for scoring, it gives a binary judgment on whether each acceptance criterion is implemented. Every judgment must cite runtime evidence, such as a trace step or a state’s DOM snapshot and screenshot. The score for this dimension is calculated as the number of criteria judged to be implemented divided by the total number of criteria.

Overall score. Because scores difer in discriminative power across dimensions, we compute z-scores for each dimension to aggregate them while preventing any one dimension from dominating the overall score. Specifically, for model � and dimension �, let $\bar { x } _ { m , j }$ be the mean raw score over generated applications. We compute

$$
z _ { m , j } = \frac { \bar { x } _ { m , j } - \mu _ { j } } { \sigma _ { j } } ,
$$

where $\mu _ { j }$ and $\sigma _ { j }$ are the mean and population standard deviation of these model-level means across the � evaluated LLMs. The mean of the functional, content, and visual alignment �-scores gives requirement alignment. Averaging requirement alignment, aesthetics, and usability yields the overall score, with weights of 1/3 for aesthetics and usability and $1 / 9$ for each alignment sub-dimension. Scores express relative performance within the evaluated model pool.

## 5. Evaluation Results

Our evaluation aims to answer five research questions.

## 5.1. Evaluation Setup

We evaluate 16 frontier LLMs on all 369 real-world user requirements. Each model generates one application per requirement through an agentic coding framework, reflecting practical workflows for web application generation. To make the fullest possible use of model capabilities and reflect practical usage, we integrate the GPT-family LLMs with Codex, and the remaining LLMs with Claude Code. Table 3 lists the reasoning-efort settings; other parameters retain the provider defaults of each model. All applications undergo the same four-stage evaluation pipeline (§4), without model-specific adjustments. Unless otherwise stated, Claude-Opus-4.8 performs both exploration and scoring.

## 5.2. RQ1: Main Results

Table 3 reports dimension-level �-scores and the aggregated main scores across 16 LLMs. Positive scores indicate performance above the mean value of the model pool (§4.4). Appendix E reports detailed raw scores.

No model leads on every dimension. Claude-Opus-5 ranks first overall and leads in usability (+1.842); GPT-5.6-Sol leads in aesthetics by a large margin (+2.220), while Kimi-K3 leads in requirement alignment (+1.371). These complementary strengths would be obscured by an overall ranking alone.

Visual aesthetics and usability capture distinct aspects of application quality. Their correlation is strong at the model level (� = 0.85) but only 0.36 at the application level. GPT-5.6-Sol ranks first in aesthetics and fifth in usability; MiniMax-M3 has near-average aesthetics (−0.285) but the lowest usability (−1.387). Visual polish therefore does not establish reliable interaction, supporting runtime evaluation alongside visual assessment.

Model generations difer under the tested configurations. Within each family represented by multiple generations, the newer generation has a higher overall score, indicating steady progress in the capabilities of each model family.

Table 3 | Results of 16 frontier LLMs on IWC-Bench.
<table><tr><td colspan="6"></td><td colspan="5">Requirement Alignment</td></tr><tr><td>#</td><td>Model</td><td>Har.</td><td>Eff.</td><td>Aesth.</td><td>Usab.</td><td>Func.</td><td>Cont.</td><td>Vis.</td><td>Align.</td><td>Total</td></tr><tr><td>1</td><td>Claude-Opus-5</td><td>CC</td><td>xhigh</td><td>+1.428</td><td>+1.842</td><td>+1.149</td><td>+0.885</td><td>+0.586</td><td>+0.874</td><td>+1.381</td></tr><tr><td>2</td><td>GPT-5.6-Sol</td><td>CX</td><td>xhigh</td><td>+2.220</td><td>+0.931</td><td>+0.669</td><td>+0.179</td><td>-0.225</td><td>+0.208</td><td>+1.120</td></tr><tr><td>3</td><td>Qwen3.8-Max</td><td>CC</td><td>enabled</td><td>+0.805</td><td>+1.167</td><td>+1.064</td><td>+1.162</td><td>+1.118</td><td>+1.115</td><td>+1.029</td></tr><tr><td>4</td><td>Kimi-K3</td><td>CC</td><td>max</td><td>+0.609</td><td>+1.105</td><td>+1.195</td><td>+1.370</td><td>+1.548</td><td>+1.371</td><td>+1.028</td></tr><tr><td>5</td><td>Claude-Opus-4.8</td><td>CC</td><td>xhigh</td><td>+0.185</td><td>+0.925</td><td>+1.461</td><td>+1.221</td><td>+1.352</td><td>+1.345</td><td>+0.818</td></tr><tr><td>6</td><td>Grok-4.5</td><td>CC</td><td>high</td><td>+0.547</td><td>+0.979</td><td>+0.980</td><td>+0.749</td><td>+0.972</td><td>+0.901</td><td>+0.809</td></tr><tr><td>7</td><td>GPT-5.5</td><td>CX</td><td>xhigh</td><td>+0.744</td><td>+0.297</td><td>+0.376</td><td>+0.617</td><td>+0.590</td><td>+0.528</td><td>+0.523</td></tr><tr><td>8</td><td>Claude-Opus-4.7</td><td>CC</td><td>xhigh</td><td>-0.123</td><td>-0.377</td><td>-0.192</td><td>-0.083</td><td>-0.079</td><td>-0.118</td><td>-0.206</td></tr><tr><td>9</td><td>GLM-5.2</td><td>CC</td><td>max</td><td>-0.222</td><td>-0.488</td><td>-0.384</td><td>+0.210</td><td>-0.010</td><td>-0.061</td><td>-0.257</td></tr><tr><td>10</td><td>DeepSeek-V4-Flash</td><td>CC</td><td>max</td><td>-0.274</td><td>-0.255</td><td>-0.419</td><td>-0.300</td><td>-0.302</td><td>-0.340</td><td>-0.290</td></tr><tr><td>11</td><td>HY3</td><td>CC</td><td>high</td><td>-0.620</td><td>-0.454</td><td>-0.613</td><td>-0.661</td><td>-0.312</td><td>-0.528</td><td>-0.534</td></tr><tr><td>12</td><td>Qwen3.7-Max</td><td>CC</td><td>enabled</td><td>-0.961</td><td>-0.916</td><td>-0.645</td><td>-0.996</td><td>-0.368</td><td>-0.670</td><td>-0.849</td></tr><tr><td>13</td><td>GLM-5.1</td><td>CC</td><td>enabled</td><td>-1.089</td><td>-0.962</td><td>-0.538</td><td>-0.589</td><td>-0.669</td><td>-0.599</td><td>-0.883</td></tr><tr><td>14</td><td>DeepSeek-V4-Pro-Prev</td><td>CC</td><td>max</td><td>-1.390</td><td>-1.291</td><td>-0.726</td><td>-0.507</td><td>-0.324</td><td>-0.519</td><td>-1.067</td></tr><tr><td>15</td><td>Kimi-K2.7-Code</td><td>CC</td><td>enabled</td><td>-1.575</td><td>-1.117</td><td>-1.068</td><td>-0.619</td><td>-1.475</td><td>-1.054</td><td>-1.249</td></tr><tr><td>16</td><td>MiniMax-M3</td><td>CC</td><td>adaptive</td><td>-0.285</td><td>-1.387</td><td>-2.308</td><td>-2.640</td><td>-2.404</td><td>-2.450</td><td>-1.374</td></tr></table>

Note: Har.: coding harness (CX: Codex; CC: Claude Code). Ef.: reasoning efort.

Table 4 | Agreement between IWC-Bench and human pairwise preferences.
<table><tr><td rowspan="2">∆z band</td><td colspan="3">Within band</td><td colspan="3">Cumulative  $( \Delta z \ge$  lower bound)</td></tr><tr><td>n</td><td>Agree</td><td>Rate</td><td>n</td><td>Agree</td><td>Rate</td></tr><tr><td>[0, 0.25)</td><td>65</td><td>48</td><td>73.8%</td><td>197</td><td>168</td><td>85.3%</td></tr><tr><td>[0.25, 0.5)</td><td>52</td><td>45</td><td>86.5%</td><td>132</td><td>120</td><td>90.9%</td></tr><tr><td>[0.5, 0.75)</td><td>30</td><td>26</td><td>86.7%</td><td>80</td><td>75</td><td>93.8%</td></tr><tr><td>[0.75, 1.0)</td><td>13</td><td>13</td><td>100.0%</td><td>50</td><td>49</td><td>98.0%</td></tr><tr><td>[1.0, 1.5)</td><td>17</td><td>16</td><td>94.1%</td><td>37</td><td>36</td><td>97.3%</td></tr><tr><td>[1.5, 2.0)</td><td>7</td><td>7</td><td>100.0%</td><td>20</td><td>20</td><td>100.0%</td></tr><tr><td>[2.0, ∞)</td><td>13</td><td>13</td><td>100.0%</td><td>13</td><td>13</td><td>100.0%</td></tr></table>

## 5.3. RQ2: Human Preference Alignment

We sample 200 sessions from our internal replica of Code Arena (LMArena.ai, 2026), independently of the 369 benchmark tasks. Among these sessions, 197 retain valid application pairs. Each session records a human vote (preference) between two applications for the same requirement. In order to eliminate irresponsible votes, six additional annotators review and fix every vote, and after that, two others further review a random 20% subset independently. The validated votes serve as ground truth labels.

To obtain IWC-Bench’s automated votes, for application � and component �, we compute $\boldsymbol { z } _ { a , j } ~ =$ $( x _ { a , j } - \mu _ { j } ^ { H } ) / \sigma _ { j } ^ { H }$ from its raw score $x _ { a , j } ,$ using the component mean $\mu _ { j } ^ { H }$ and standard deviation $\sigma _ { j } ^ { H }$ over all applications in this human-agreement experiment. We aggregate these �-scores using the same process as in §4.4 and compare the resulting pairwise ordering with human preferences. We define Δ� as the absolute diference between paired overall scores and report within-band agreement and cumulative agreement above each threshold.

Table 4 shows that IWC-Bench aligns with human preferences on 168 of 197 pairs, yielding a satisfactory overall agreement rate of 85.3%.

Disagreements are concentrated among pairs with small score diferences. Of the 29 disagreements, 17 occur in the [0, 0.25) band and 7 in [0.25, 0.5). Cumulative agreement exceeds 90% for pairs with $\Delta z \ge 0 . 2 5$ . Larger score diferences are generally associated with higher agreement. Among the 50 pairs with $\Delta z \ge 0 . 7 5 .$ , the only disagreement involves a tradeof between visual quality and functional reliability: one application has a polished interface but crashes on its primary functional path, while the other has a less polished interface and working core functionality. IWC-Bench prefers the latter, whereas the human rater prefers the former. Appendix C presents the applications and examines this disagreement.

## 5.4. RQ3: Judge Robustness

In order to test whether IWC-Bench depends on the capabilities of a particular LLM and to prevent model-family bias, we replace the main judge throughout scoring with Gemini-3.7-Flash, a lower-cost model from a diferent provider, while holding tasks, generated applications, exploration traces, procedures, and prompts fixed. Overall scores have a Pearson correlation (Pearson, 1895) of 0.983 and a Spearman rank correlation (Spearman, 1987) of $\rho = 0 . 9 8 2$ across judges. Of 120 model pairs, 115 retain their relative order (95.8% agreement). Three of the five reversals involve main-judge score diferences below 0.04. Although raw-score levels and spreads change substantially, aggregate rankings remain stable. For example, the model-pool mean usability score increases from 59.7 to 75.3, while the range of model-level aesthetics scores expands from 26.8 to 44.2 points. These changes motivate standardizing dimensional raw scores before aggregation. Appendix B provides the full ranking comparison and raw-score analysis.

## 5.5. RQ4: Instrumentation Reliability

We evaluate instrumentation applicability across all 5,904 applications in the main experiment. Instrumentation fails for only 7 applications, yielding a success rate of 99.88%. The failures have identifiable engineering causes: 4 applications use JSX input forms unsupported by Istanbul, 2 use npmworkspaces monorepos for which the tool cannot locate the Vite project root, and 1 exceeds the sandbox deployment-size limit after instrumentation. We separately check deployment regressions on the 360 deployable applications generated by Claude-Opus-5. All remain deployable after instrumentation. This check establishes deployment reliability on the tested subset. Under the fallback procedure in §4.1, the 7 applications for which instrumentation fails still undergo exploration and scoring in black-box mode, without coverage guidance.

## 5.6. RQ5: Coverage Improvement

Code coverage measures the proportion of the implementation exercised during exploration and helps identify functionality that may remain unobserved. We assess the contribution of coverage guidance by comparing exploration with and without this feedback. To control experimental scale, this ablation uses the 360 deployable applications generated by Claude-Opus-5. The remaining 9 applications fail for generation-side reasons, including missing files, deployment errors, and incorrect port declarations.

Figure 3 compares the coverage distributions under the two settings. Coverage guidance increases median function coverage from 91.9% to 94.3%. The number of tasks below 90% coverage decreases from 139 (39%) to 99 (28%), with a relative reduction of 29%, while the number reaching 100% coverage increases from 38 to 52. The gains are concentrated in the medium-to-high coverage range: the [70%, 90%) interval contains 38 fewer tasks, and the number at or above 90% increases by 40, from 221 to 261. The low-coverage tail changes little, with 17 tasks below 70% coverage without guidance and 15 with guidance.

![](images/5c4ff376619143429873962a87733d0ac4d7f373b36ee58016c06057b7c20854.jpg)  
Figure 3 | Efect of coverage guidance on function coverage during exploration.

Manual inspection shows that applications below 70% coverage typically contain failures that block further exploration. Examples include an entry page that immediately reports an error, an event handler that throws an exception, and a broken critical path that makes downstream functionality unreachable. These observations suggest that application defects contribute substantially to the remaining low coverage. Such failures also prevent real users from reaching the afected functionality, making them relevant to the evaluation of user experience.

In addition, the LLM-generated applications remain simple. Prior GUI-testing studies (Liu et al., 2026c; Gu et al., 2025) report greater exploration dificulty on complex applications (Liu et al., 2025a; Gu et al., 2025); coverage guidance is expected to bring more benefits as model capabilities grow and models generate more complex applications.

## 6. Limitations

LLM judges. The evaluation process largely relies on LLMs. Therefore, judge preferences may afect the results. We validated the reliability of the scores by comparing them with human preferences (§5.3) and verified scoring stability by comparing results after replacing the judge (§5.4).

Model-pool dependence. Standardization makes �-scores dependent on the participating LLMs and unsuitable for direct comparison across pools. Future deployments could use fixed reference means and standard deviations. Raw scores further aid interpretation.

Task scope. The dataset covers single-turn, text-only, front-end-only tasks from one platform, and excludes iterative repair, multimodal input, and back-end integration. Since the evaluation process simulates how real users use webpages, it can be extended to other frameworks and webpages with back-end functionality; considering the engineering cost, we leave this as future work.

Open source. To prevent data leakage and comply with our business confidentiality policy, we do not publicly release the dataset or source code.

## 7. Conclusion

In this paper, we introduced IWC-Bench, which evaluates web application generation from a software testing perspective. Combining coverage-guided exploration, state abstraction, and separated scoring grounded in runtime evidence, IWC-Bench can objectively and comprehensively score generated web applications, thereby reliably evaluating the capability of LLMs to generate web applications. IWC-Bench achieves 85.3% agreement with human preferences, with stable rankings across judge models. Coverage guidance increases the coverage during exploration, while the results across 16 frontier LLMs reveal complementary strengths in aesthetics, usability, and requirement alignment.

## References

Tony Beltramelli. Pix2Code: Generating code from a graphical user interface screenshot. In Proceedings of the ACM SIGCHI Symposium on Engineering Interactive Computing Systems, 2018.

Tanvir Bhathal and Asanshay Gupta. Websight: A vision-first architecture for robust web agents. arXiv preprint arXiv:2508.16987, 2025.

Andrei Z Broder. On the resemblance and containment of documents. In Proceedings of the Compression and Complexity of Sequences, 1997.

Dasen Dai, Biao Wu, Meng Fang, Shuoqi Li, and Wenhao Wang. I-WebGenBench: Evaluating interactivity in LLM-generated scientific web applications. arXiv preprint arXiv:2606.00750, 2026.

Design Arena. Design arena. https://www.designarena.ai/, 2026.

Google. Ui/application exerciser monkey. https://developer.android.com/studio/test/ other-testing-tools/monkey, 2025.

Zhiyu Gu, Chenxu Liu, Guoquan Wu, Yifei Zhang, ChenXi Yang, Zheheng Liang, Wei Chen, and Jun Wei. Deep reinforcement learning for automated web GUI testing. arXiv preprint arXiv:2504.19237, 2025.

Istanbul Contributors. Nyc/istanbul, a javascript test coverage tool. https://istanbul.js.org/, 2024.

Xinping Lei, Xinyu Che, Junqi Xiong, Chenchen Zhang, Yukai Huang, Chenyu Zhou, Haoyang Huang, Minghao Liu, Letian Zhu, Hongyi Ye, et al. WebCompass: Towards multimodal web coding evaluation for code language models. arXiv preprint arXiv:2604.18224, 2026.

Chenxu Liu, Zhiyu Gu, Guoquan Wu, Ying Zhang, Jun Wei, and Tao Xie. Temac: Multi-agent collaboration for automated web GUI testing. arXiv preprint arXiv:2506.00520, 2025a.

Chenxu Liu, Junheng Wang, Wei Yang, Ying Zhang, and Tao Xie. Judge: Efective state abstraction for guiding automated web GUI testing. ACM Transactions on Software Engineering and Methodology, 2025b.

Chenxu Liu, Yingjie Fu, Wei Yang, Ying Zhang, and Tao Xie. Webcoderbench: Benchmarking web application generation with comprehensive and interpretable evaluation metrics. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026a.

Chenxu Liu, Junheng Wang, Zihe Song, Wei Yang, Ying Zhang, and Tao Xie. Booster: Efective and eficient web GUI trace reduction based on multi-level state abstraction and time-guided hierarchical delta debugging. ACM Transactions on Software Engineering and Methodology, 2026b.

Chenxu Liu, Wei Yang, Ying Zhang, and Tao Xie. Understanding automated web GUI testing: An empirical study across exploration strategies and state abstractions. arXiv preprint arXiv:2606.16650, 2026c.

Zhe Liu, Chunyang Chen, Junjie Wang, Mengzhuo Chen, Boyu Wu, Xing Che, Dandan Wang, and Qing Wang. Make LLM a testing expert: Bringing human-like interaction to mobile GUI testing via functionality-aware decisions. In Proceedings of the 46th International Conference on Software Engineering, 2024.

LMArena.ai. Code arena WebDev leaderboard. https://arena.ai/leaderboard/code/webdev, 2026.

Zimu Lu, Yunqiao Yang, Houxing Ren, Haotian Hou, Han Xiao, Ke Wang, Weikang Shi, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. WebGen-Bench: Evaluating LLMs on generating interactive and functional websites from scratch. Advances in Neural Information Processing Systems, 2026.

Yuxin Meng, Yuhan Suo, Junjie Wang, Yuhan Sun, Yiyao Yu, Ruixu Zhang, Ruining Hu, Yubin Wang, Shouwei Ruan, Bin Wang, et al. WebRISE: Requirement-induced state evaluation for MLLMgenerated web artifacts. arXiv preprint arXiv:2606.03220, 2026.

Ali Mesbah, Arie Van Deursen, and Stefan Lenselink. Crawling Ajax-based web applications through dynamic analysis of user interface state changes. ACM Transactions on the Web (TWEB), 2012.

Microsoft. Playwright: Fast and reliable end-to-end testing for modern web apps. https: //playwright.dev/, 2025.

Karl Pearson. Note on regression and inheritance in the case of two parents. proceedings of the royal society of London, 1895.

Charles Spearman. The proof and measurement of association between two things. The American journal of psychology, 1987.

Haoyu Sun, Huichen Will Wang, Jiawei Gu, Linjie Li, and Yu Cheng. FullFront: Benchmarking MLLMs across the full front-end engineering workflow. arXiv preprint arXiv:2505.17399, 2025.

Tung D Vu, Chung Hoang, and Truong-Son Hy. Multimodal graph representation learning for website generation based on visual sketch. arXiv preprint arXiv:2504.18729, 2025.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. MiniLM: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. Advances in Neural Information Processing Systems, 2020.

Xueqing Wu, Zihan Xue, Da Yin, Shuyan Zhou, Kai-Wei Chang, Nanyun Peng, and Yeming Wen. Frontalk: Benchmarking front-end development as conversational code generation with multi-modal feedback. arXiv preprint arXiv:2601.04203, 2025.

Kai Xu, YiWei Mao, XinYi Guan, and ZiLong Feng. Web-Bench: A LLM code benchmark based on web standards and frameworks. arXiv preprint arXiv:2505.07473, 2025.

Rahul Krishna Yandrapally and Ali Mesbah. Fragment-based test generation for web apps. IEEE Transactions on Software Engineering, 2022.

Rahulkrishna Yandrapally, Andrea Stocco, and Ali Mesbah. Near-duplicate detection in web app model inference. In Proceedings of the 42nd international conference on software engineering, 2020.

Haoyue Yang, Zhangxiao Shen, Fan Ding, Hangting Lou, Yifeng Kou, Haoqing Yu, Jingyao Li, Zhengfan Wu, Siqi Bao, Jing Liu, et al. Cookie-Bench: Continuous on-screen key interaction evaluation for web generation. arXiv preprint arXiv:2605.30000, 2026.

Chengyue Yu, Siyuan Lu, Chenyi Zhuang, Dong Wang, Qintong Wu, Zongyue Li, Runsheng Gan, Chunfeng Wang, Siqi Hou, Gaochi Huang, et al. Aworld: Orchestrating the training recipe for agentic AI. arXiv preprint arXiv:2508.20404, 2025.

Chenchen Zhang, Yuhang Li, Can Xu, Jiaheng Liu, Ao Liu, Changzhi Zhou, Ken Deng, Dengpeng Wu, Guanhua Huang, Kejiao Li, et al. ArtifactsBench: Bridging the visual-interactive gap in LLM code generation evaluation. arXiv preprint arXiv:2507.04952, 2025.

Wenyu Zhang, Guoliang You, Haotian Zhao, Tianshu Zhu, Haoran Wang, Xiaoxuan Tang, Mingyang Dai, Jingnan Gu, Daxiang Dong, Jianmin Wu, et al. WebGameBench: Requirement-to-application evaluation for coding agents via browser-native games. arXiv preprint arXiv:2605.17637, 2026a.

Zuhao Zhang, Chengyue Yu, Yuante Li, Chenyi Zhuang, Linjian Mo, and Shuai Li. MiniAppBench: Evaluating the shift from text to interactive HTML responses in LLM-powered assistants. arXiv preprint arXiv:2603.09652, 2026b.

Yan Zheng, Yi Liu, Xiaofei Xie, Yepang Liu, Lei Ma, Jianye Hao, and Yang Liu. Automatic web testing using curiosity-driven reinforcement learning. In Proceedings of the 43rd International Conference on Software Engineering, 2021.

Hongda Zhu, Yiwen Zhang, Bing Zhao, Jingzhe Ding, Siyao Liu, Tong Liu, Dandan Wang, Yanan Liu, and Zhaojian Li. FrontendBench: A benchmark for evaluating LLMs on front-end development via automatic evaluation. arXiv preprint arXiv:2506.13832, 2025.

## Appendices

## Table of Contents:

A Evaluation Details 2   
A.1 Instrumentation and Deployment . 2   
A.2 Scoring Procedures . 2   
B Detailed Judge-Robustness Results 4   
C The High-Margin Human–IWC-Bench Disagreement Case 5   
D Three-Round versus Single-Round Scoring 6   
E Raw Scores 7

## A. Evaluation Details

## A.1. Instrumentation and Deployment

The tool described in §4.1 instruments static and plain Node.js projects directly at the source level. For projects requiring compilation, it injects an instrumentation plugin and processes the generated files after the build. Framework detection selects the corresponding procedure automatically, and the instrumented application is deployed in an isolated sandbox over HTTP.

Instrumentation depends on application toolchains and may afect deployment or runtime behavior. Section 5.5 reports its applicability and deployment checks. Since Istanbul is a time-tested and widely used instrumentation tool, we believe that the instrumentation does not harm the application behavior. The black-box fallback retains applications with instrumentation failures for exploration and scoring without coverage feedback.

## A.2. Scoring Procedures

This section details the evidence requirements, judgment procedures, and scoring aspects used in §4.4.

Deliberation and score aggregation. Visual aesthetics and usability each use three rounds. The advocate identifies strengths, and the critic identifies defects; both must cite concrete observations without assigning scores. The final judge checks their claims against the supplied evidence, discards unsupported claims, and adds relevant observations that both rounds missed. It selects a 0–5 base score from the overall visual impression or interaction experience, then records a signed adjustment and a supporting rationale for each of seven aspects, explicitly indicating when no adjustment is needed. The sum of the base score and adjustments is clipped to [0, 5]. With the advocate and critic assessments held fixed, the final judge runs independently five times. The highest and lowest scores are discarded, and the remaining mean is multiplied by 20 to obtain the 0–100 score.

Visual aesthetics. The seven aspects are first impression, content completeness, layout structure, visual detail, information hierarchy and typography, stylistic consistency, and emotion and trust. Observations must refer to specific locations in the representative screenshots. The assessment distinguishes deliberate simplicity from unfinished or empty content and examines the quality of graphics and layout rather than merely their presence. Factual correctness alone does not establish visual quality, and the judge must not infer failures in unobserved dynamic behavior from static screenshots.

Usability. The seven interaction-oriented aspects are efectiveness of core operations, immediacy of feedback, quality of controls and input, interaction flow and discoverability, state reachability and recovery, avoidance of dead ends and errors together with response consistency, and stability and fault tolerance. Each assessment must cite an exploration-trace step or a state-transition graph node or edge. The judge considers observed responses, navigation, recovery, and failures, without inventing requirements for unobserved features. A path that was not tested is neither a strength nor a weakness; observed failures that block core operations warrant substantial penalties.

Requirement alignment. Functional, content, and visual criteria share one evidence-retrieval process but receive separate satisfaction percentages. Starting from the state-transition graph and state overview, the agent retrieves relevant trace steps, DOM passages, and screenshots as needed. Functional criteria are checked against available features, executed interactions, and resulting state changes; content criteria against the specified text, data, and resources; and visual criteria against the required appearance and components. The agent reports a Boolean decision, supporting evidence, a brief analysis, and a confidence estimate for each criterion. A positive decision requires evidence that the property is present in the running application; required functionality must be reachable and observed to work. Unexecuted source code alone is insuficient, and no partial-credit category is used. Each sub-dimension receives 100 times the fraction of its criteria judged satisfied.

## B. Detailed Judge-Robustness Results

The experiment in §5.4 replaces Claude-Opus-4.8 with Gemini-3.7-Flash while holding the generated applications, exploration traces, scoring procedures, and prompts fixed. We report the complete ranking comparison and changes in raw-score distributions below.

Table 5 shows that model rankings remain stable across the two judges. Overall model scores have a Pearson correlation of 0.983 and a Spearman rank correlation of � = 0.982. Of the 120 model pairs, 115 retain their relative order, corresponding to 95.8% pairwise agreement. The top four ranks and ranks 7, 8, and 11 remain unchanged.

Of the five pairs whose order changes, three have overall-score diferences below 0.04 under the main judge. This pattern is consistent with the concentration of human disagreements among near-tied applications in §5.3. The remaining two reversals involve MiniMax-M3, which rises two positions under Gemini-3.7-Flash. These changes suggest that judge choice has its greatest efect on closely ranked models, with some additional variation near the bottom of the leaderboard.

Raw-score changes vary by dimension; Appendix E reports the underlying means and ranges. Standardization adjusts component location and spread, making the overall ranking stable.

Table 5 | Score and ranking diferences after replacing the judge model. Main: Claude-Opus-4.8; Gem.: Gemini-3.7-Flash. Rk. is the rank under Gemini; Δ is that rank minus the main-judge rank, so negative values indicate improvement.
<table><tr><td colspan="2"></td><td colspan="2">Total</td><td colspan="5"></td><td colspan="2">Total</td><td rowspan="2"></td><td rowspan="2">Rk. Δ</td></tr><tr><td>#</td><td>Model</td><td>Main</td><td>Gem.</td><td>Rk.</td><td>Δ</td><td>#</td><td>Model</td><td>Main</td><td>Gem.</td></tr><tr><td>1</td><td>Claude-Opus-5</td><td>+1.381</td><td>+1.626</td><td>1</td><td>0</td><td>9</td><td>GLM-5.2</td><td>-0.257</td><td>-0.196</td><td>10</td><td>+1</td></tr><tr><td>2</td><td>GPT-5.6-Sol</td><td>+1.120</td><td>+1.352</td><td>2</td><td>0</td><td>10</td><td>DeepSeek-V4-Flash</td><td>-0.290</td><td>-0.063</td><td></td><td>9 -1</td></tr><tr><td>3</td><td>Qwen3.8-Max</td><td>+1.029</td><td>+0.851</td><td>3</td><td>0</td><td>11</td><td>HY3</td><td>-0.534</td><td>-0.741</td><td>11</td><td>0</td></tr><tr><td>4</td><td>Kimi-K3</td><td>+1.028</td><td>+0.769</td><td>4</td><td>0</td><td>12</td><td>Qwen3.7-Max</td><td>-0.849</td><td>-0.931</td><td>13</td><td>+1</td></tr><tr><td>5</td><td>Claude-Opus-4.8</td><td>+0.818</td><td>+0.717</td><td>6 +1</td><td></td><td>13</td><td>GLM-5.1</td><td>-0.883</td><td>-0.889</td><td>12</td><td>-1</td></tr><tr><td>6</td><td>Grok-4.5</td><td>+0.809</td><td>+0.760</td><td>5 -1</td><td></td><td>14</td><td>DeepSeek-V4-Pro-Prev</td><td>-1.067</td><td>-1.146</td><td>15 +1</td><td></td></tr><tr><td>7</td><td>GPT-5.5</td><td>+0.523</td><td>+0.393</td><td>7</td><td>0</td><td>15</td><td>Kimi-K2.7-Code</td><td>-1.249</td><td>-1.358</td><td></td><td>16 +1</td></tr><tr><td>8</td><td>Claude-Opus-4.7</td><td>-0.206</td><td>-0.019</td><td>8 0</td><td></td><td>16</td><td>MiniMax-M3</td><td>-1.374</td><td>-1.125</td><td></td><td>14 -2</td></tr></table>

## C. The High-Margin Human–IWC-Bench Disagreement Case

Figure 4 shows the only disagreement among the 50 human-preference pairs with $\Delta z \ge 0 . 7 5$ . The human rater prefers the artifact in Figure 4(a), which has a more polished and visually coherent page. Runtime testing reveals that its primary gameplay path crashes after entry. The artifact in Figure 4(b) has a sparser interface, but its core Sudoku interaction works. IWC-Bench prefers this artifact because its advantages in usability and requirement alignment outweigh the visual advantage of the first artifact. The case is consistent with diferent tradeofs between aesthetics and functional reliability, although the vote alone does not identify the rater’s weighting.

![](images/5a132f427cfd2ce4475d0e59ee1c1b65cccb7244ad77e3a24dd4f654523942b6.jpg)  
(a) Human-preferred artifact: more visually polished, but the primary gameplay path crashes.

![](images/29ebce1f4bed17ad41b98a92e2b9b5b31e64d6ea4403bdd7c93bc88a63ed1b95.jpg)  
(b) IWC-Bench-preferred artifact: less visually polished, but the core Sudoku interaction works.

Figure 4 | The only human–IWC-Bench disagreement with $\Delta z \ge 1 . 0 $ . The human rater prefers the visually polished artifact in (a); IWC-Bench prefers the artifact in (b), whose core functionality works during runtime testing.

## D. Three-Round versus Single-Round Scoring

We compare application-level scores from the three-round deliberative protocol with those from direct single-round scoring to assess the efect on score distributions. The single-round baseline directly uses the final round prompt of the three-round protocol. Figure 5 presents paired results for visual aesthetics and usability, with the dashed diagonal indicating equal scores. The deliberative protocol lowers mean scores by 15.5 points for aesthetics and 8.5 points for usability, suggesting that advocacy, criticism, and evidence verification lead to more conservative judgments. Artifacts assigned identical or similarly saturated scores in a single round also receive a wider range of scores after deliberation. These results indicate reduced concentration at the scale endpoints and greater diferentiation among artifacts that direct scoring evaluates similarly.

![](images/ff93c3b7435a0c68ed5d11aafc1e242eaf2a1e00ba0f2b803abfb5bef68b5495.jpg)

![](images/41e5086fed8f5e1e88c6fd9abae5e440686850d7c70136fd80b95fdd45caf2c7.jpg)  
Figure 5 | Application-level comparison of direct single-round scoring (horizontal axis) and the threeround deliberative protocol used by IWC-Bench (vertical axis) for visual aesthetics and usability. Each point is one paired evaluation; the dashed line marks equal scores. Here, Δ denotes the three-round score minus the single-round score.

## E. Raw Scores

The main-text rankings aggregate standardized component scores to account for diferences in component dispersion (§4.4). These scores express performance relative to the evaluated model pool and are not directly comparable across studies. To make the underlying score levels interpretable, this appendix reports model-level raw means under Claude-Opus-4.8, the main judge (§5.2), and Gemini-3.7-Flash, the alternative judge in the robustness experiment (§5.4).

Table 6 reports raw scores on a 0–100 scale. For visual aesthetics and usability, five final-judge scores on a 0–5 scale are aggregated using a trimmed mean and multiplied by 20. Rows follow the overall ranking under the main judge, as in Table 3. The two judges evaluate the same artifacts and exploration traces using identical prompts, isolating the efect of judge choice.

Additional correlations highlight complementary aspects of application quality: visual aesthetics and requirement alignment are weakly correlated at the artifact level (� = 0.32), while usability and requirement alignment are strongly associated at the model level (� = 0.88).

Component scales difer substantially: under the main judge, aesthetics spans 26.8 points across models, whereas the alignment sub-dimensions span 7.0–10.4 points. Averaging raw components would therefore give aesthetics greater influence on between-model diferences. Standardization equalizes component dispersion before applying the stated weights. The cross-judge diferences in this table concern score level and spread; ranking robustness is evaluated separately in Table 5.

Table 6 | Raw model-level mean scores under both judges before standardization. All scores use a 0–100 scale. Functional, content, and visual alignment scores are the percentages of acceptance criteria judged satisfied. Rows follow the overall ranking under the main judge. The last two rows report each column’s mean and range (maximum minus minimum) across the 16 models.
<table><tr><td colspan="2"></td><td colspan="5">Main judge (Claude-Opus-4.8)</td><td colspan="5">Gemini-3.7-Flash</td></tr><tr><td>#</td><td></td><td></td><td colspan="3">Alignment</td><td></td><td></td><td colspan="3"></td><td>Alignment</td></tr><tr><td></td><td>Model</td><td>Aesth.</td><td>Usab.</td><td>Func.</td><td>Cont.</td><td>Vis.</td><td>Aesth.</td><td>Usab.</td><td>Func.</td><td>Cont.</td><td>Vis.</td></tr><tr><td>1</td><td>Claude-Opus-5</td><td>53.5</td><td>69.6</td><td>96.7</td><td>95.9</td><td>95.6</td><td>59.5</td><td>89.1</td><td>99.2</td><td>98.9</td><td>99.5</td></tr><tr><td>2</td><td>GPT-5.6-Sol</td><td>59.1</td><td>64.7</td><td>95.4</td><td>94.7</td><td>93.7</td><td>70.3</td><td>82.5</td><td>97.6</td><td>97.9</td><td>98.0</td></tr><tr><td>3</td><td>Qwen3.8-Max</td><td>49.1</td><td>66.0</td><td>96.5</td><td>96.4</td><td>96.9</td><td>50.7</td><td>84.2</td><td>98.1</td><td>97.4</td><td>98.0</td></tr><tr><td>4</td><td>Kimi-K3</td><td>47.7</td><td>65.6</td><td>96.9</td><td>96.8</td><td>97.9</td><td>47.2</td><td>82.7</td><td>97.9</td><td>98.1</td><td>98.6</td></tr><tr><td>5</td><td>Claude-Opus-4.8</td><td>44.7</td><td>64.7</td><td>97.6</td><td>96.5</td><td>97.4</td><td>43.4</td><td>82.1</td><td>98.4</td><td>98.0</td><td>99.4</td></tr><tr><td>6</td><td>Grok-4.5</td><td>47.3</td><td>64.9</td><td>96.3</td><td>95.7</td><td>96.5</td><td>47.4</td><td>83.1</td><td>98.0</td><td>97.4</td><td>98.9</td></tr><tr><td>7</td><td>GPT-5.5</td><td>48.7</td><td>61.3</td><td>94.6</td><td>95.5</td><td>95.6</td><td>49.3</td><td>75.2</td><td>97.2</td><td>97.6</td><td>98.5</td></tr><tr><td>8</td><td>Claude-Opus-4.7</td><td>42.6</td><td>57.7</td><td>93.0</td><td>94.2</td><td>94.0</td><td>41.0</td><td>72.8</td><td>97.1</td><td>97.2</td><td>98.1</td></tr><tr><td>9</td><td>GLM-5.2</td><td>41.9</td><td>57.1</td><td>92.5</td><td>94.8</td><td>94.2</td><td>39.3</td><td>71.7</td><td>96.2</td><td>97.5</td><td>97.4</td></tr><tr><td>10</td><td>DeepSeek-V4-Flash</td><td>41.5</td><td>58.4</td><td>92.4</td><td>93.9</td><td>93.5</td><td>39.4</td><td>75.5</td><td>96.6</td><td>96.8</td><td>97.4</td></tr><tr><td>11</td><td>HY3</td><td>39.0</td><td>57.3</td><td>91.9</td><td>93.2</td><td>93.5</td><td>34.9</td><td>71.5</td><td>94.4</td><td>95.4</td><td>95.8</td></tr><tr><td>12</td><td>Qwen3.7-Max</td><td>36.6</td><td>54.8</td><td>91.8</td><td>92.7</td><td>93.3</td><td>32.0</td><td>67.3</td><td>95.1</td><td>95.1</td><td>96.7</td></tr><tr><td>13</td><td>GLM-5.1</td><td>35.7</td><td>54.6</td><td>92.1</td><td>93.4</td><td>92.6</td><td>29.1</td><td>68.5</td><td>95.8</td><td>95.0</td><td>97.2</td></tr><tr><td>14</td><td>DeepSeek-V4-Pro-Prev</td><td>33.6</td><td>52.8</td><td>91.6</td><td>93.5</td><td>93.4</td><td>27.0</td><td>66.5</td><td>94.8</td><td>95.2</td><td>96.4</td></tr><tr><td>15</td><td>Kimi-K2.7-Code MiniMax-M3</td><td>32.3</td><td>53.7</td><td>90.6</td><td>93.3</td><td>90.7</td><td>26.1</td><td>66.1</td><td>94.9</td><td>94.8</td><td>94.5</td></tr><tr><td>16</td><td></td><td>41.4</td><td>52.3</td><td>87.2</td><td>89.8</td><td>88.5</td><td>40.4</td><td>66.2</td><td>92.8</td><td>94.6</td><td>94.0</td></tr><tr><td colspan="2">pool mean</td><td>43.4</td><td>59.7</td><td>93.6</td><td>94.4</td><td>94.2</td><td>42.3</td><td>75.3</td><td>96.5</td><td>96.7</td><td>97.4</td></tr><tr><td colspan="2">range</td><td>26.8</td><td>17.3</td><td>10.4</td><td>7.0</td><td>9.4</td><td>44.2</td><td>23.0</td><td>6.4</td><td>4.3</td><td>5.5</td></tr></table>