# RobustSGPO: Search-Space Control for Agent Harness Evolution

Zibo Zhao<sup>1,2,†</sup>, Jijun Shi<sup>2</sup>, Mo Zhou<sup>2</sup>, Zhongyuan Wang<sup>2</sup>, Shifu Bie<sup>2</sup>, Yunfei Zhang<sup>2</sup>, Xuanting Zhou<sup>2</sup>, Xiangyu Wu<sup>2</sup>, Bin Liu<sup>2</sup>, Ruiming Tang<sup>2</sup>, Wenwu Ou<sup>2</sup>, Kun Gai<sup>2</sup>

<sup>1</sup>Wuhan University, Wuhan, China <sup>2</sup>Kuaishou Technology, Beijing, China

whubear@whu.edu.cn; {shijijun, zhoumo, wangzhongyuan03, bieshifu03, zhangyunfei05, zhouxuanting05, wuxiangyu06}@kuaishou.com

## Abstract

Semantic-gradient-based prompt optimization (SGPO) improves agent harnesses using execution feedback, but its local update rule leaves the choice of edit scope and operation unresolved. We introduce ROBUSTSGPO, which specifies the requested edit, constructs and checks the patch, and continues search from either the incumbent or retained snapshots. We evaluate permission scheduling, cumulative controls, and task-family transfer in the AgentX brainstorming workflow using 120 tasks, 95 runs, and 7,350 candidate attempts. Periodic 1 → 2 → 3 scheduling exceeds fixed maximum permission by 0.28 test-score points. RobustSGPO increases completion on 30 held-out tasks from 60.0% to 80.0% and improves test quality from 3.77 to 4.14 under a 20-million-token budget. Category retention reduces source-task degradation after a shift, whereas random retention reaches a higher destination endpoint. Search-space control benefits quality through executable edits and alternative starting points, with measurable retention overhead.

## 1 Introduction

This paper builds on the published AgentX work [4], which established an industrial recommendation workflow and a semantic-gradient-based prompt optimization (SGPO) loop for harness evolution. SGPO diagnoses trace failures, edits an agent specification, and admits candidates through paired replay. RobustSGPO extends this foundation with explicit search-space control.

Multi-agent behavior is distributed across instructions, input/output contracts, and agent connections. Restricting edits to one agent limits direct repair of cross-agent failures. Open ing the whole permitted space does not ensure that the model explores it: generation can still concentrate on familiar instruction rewrites. We study how to select edits, make them executable, and retain alternative starting points under a fixed budget.

Our contributions are an analysis of the gap between permitted and valid realized edits; RobustSGPO’s explicit control of edit selection, execution, and retention; and experiments connecting these controls to held-out quality, transfer, and token cost.

## 2 Related Work

Harness evolution. Meta-Harness searches code using prior candidates, scores, and traces [5]; AHE exposes editable components and links changes to predicted outcomes [6]. Self-Harness couples failure mining with minimal proposals and regression validation [15]. These works establish harness evolution beyond prompt rewriting. RobustSGPO studies a narrower question: selecting scope, operation, and targets before generation, then enforcing targets and constructing prescribed add/remove edits.

Structured search and adaptation. AgentFlow searches roles, topology, and protocols through a typed graph DSL with pre-execution checks [7]. HARBOR searches bounded harness configurations with cost-aware Bayesian optimization [11]. HELIX provides typed, source-traceable composition and verified trajectories for model–harness coevolution [2]; HASE jointly evolves solutions, harnesses, and model weights [9]. Adaptive Auto-Harness routes tasks through a harness tree [8], while Evo-Harness compiles experience into reusable skills [12]. Our model and evaluator stay fixed; archived snapshots seed further mutations and must pass incumbent-based admission, rather than serving as tasktime routes.

Prompt evolution and retention. OPRO, Promptbreeder, and TextGrad optimize textual variables [3, 13, 14]; GEPA retains complementary candidates on a Pareto frontier [1]. Our MAP-Elites-inspired archive [10] instead groups complete snapshots by edit scope and operation. The experiments isolate these cumulative controls within AgentX; they do not rank RobustSGPO against the above systems on shared benchmarks. Evaluation covers one brainstorming workflow; fixed tables and code reproduce its curves.

## 3 System Context and Baseline SGPO

## 3.1 AgentX and the Evaluation Workflow

In the AgentX workflow (Figure 1), the Brainstorm Agent generates experimental ideas; its Developing Agent changes code;

its Evaluation Agent analyzes experiments; harness evolution improves these agents [4]. We evaluate the brainstorming workflow, in which proposal orchestration coordinates question, idea, and validation agents. Transfer stays within this workflow.

![](images/9f435423e39c4d5fffa8355ca787b065ffd8232b950544996d8247d0883d66cd.jpg)  
Figure 1: AgentX changes industrial recommendation iteration from a manually handed-of pipeline into an agent-driven closed loop. Brainstorming, development, and evaluation agents consume online feedback and trajectory data, while harness evolution improves each agent over time. Prior-work foundation, reproduced from the AgentX technical report [4].

## 3.2 Trace-to-Update Loop

The report’s SGPO-I updates one target agent �, keeping the model, tools, orchestration, and other agents fixed [4]. It samples traces $\mathcal { T }$ , derives rubrics R and standalone replay tasks, and computes a natural-language loss and semantic gradient:

$$
\ell _ { t , i } , g _ { t , i } = E _ { \mathrm { a g e n t } } ( h _ { t , i } ; \mathcal { T } , \mathcal { R } ) .\tag{1}
$$

The gradient identifies missing requirements, sequencing errors, or broken contracts. The refinement agent generates a

candidate:

$$
h _ { t , i } ^ { \prime } = R _ { \mathrm { a g e n t } } ( h _ { t , i } , g _ { t , i } ) .\tag{2}
$$

The experiment agent replays identical tasks on both versions. With $\Delta J _ { i } = \mathrm { F }$ ReplayScore $( h _ { t , i } ^ { \prime } ) - \mathrm { R e p l a y S c o r e } ( h _ { t , i } )$

$$
h _ { t + 1 , i } : = h _ { t , i } ^ { \prime } \quad \mathrm { i f } \Delta J _ { i } > \epsilon \wedge \mathrm { S a f e } ( \Delta h _ { i } ) .\tag{3}
$$

Otherwise the update is rolled back and its patch, score, and diagnosis enter refinement experience. Rubrics, verifiers, traces, and model paths are protected. Figure 2 reproduces this loop.

![](images/0301c1849bd53cf0f4d837b1f3db21eb5e04101944229beb80ab2fab8fcd607e.jpg)  
Figure 2: Prior-work SGPO-I loop: trace sampling and task construction, semantic-gradient refinement, and paired replay with accept/rollback. Reproduced from the AgentX technical report [4].

## 4 The Search-Space Problem

The nominal space $S _ { \mathrm { n o m } }$ contains permitted edits; the realized space $S _ { \mathrm { r e a l } }$ contains actually generated edits. Validity checks select the efective space:

$$
S _ { \mathrm { e f f } } = \{ \Delta H \in S _ { \mathrm { r e a l } } : C _ { \mathrm { p a t h } } \wedge C _ { \mathrm { s e m a n t i c } } \wedge C _ { \mathrm { b u d g e t } } \wedge C _ { \mathrm { s a f e } } \} .\tag{4}
$$

Efective means valid, not admitted by replay. Broader permissions need not yield more valid edits or higher quality. We use three levels: $\alpha _ { 1 }$ edits one designated agent; $\alpha _ { 2 }$ edits any existing brainstorming agents but preserves their set; $\alpha _ { 3 }$ additionally permits agent addition/removal and routing changes.

E1 compares permission schedules; E2 measures validcategory coverage and structural validity, not all three space sizes. E3 tests retained starting points after a task shift.

## 5 Robust SGPO

RobustSGPO extends AgentX’s SGPO while retaining its diagnosis, proposal, replay, and accept/rollback loop. It makes three decisions explicit: what edit to attempt, how to construct and check it, and which version to continue from. A running AgentX example explains these decisions below; Figure 3 locates them within the original loop.

![](images/3f9e6e7db9a30a96a54c1b3b9aab52404448af02eae619e483a9dd7cac626058.jpg)  
Figure 3: RobustSGPO extends Figure 2 with edit selection, patch construction/checking, and retained search starting points (Sections 5.1– 5.3). All candidates still undergo incumbent-based replay admission.

## 5.1 Choose What to Change This Round

Consider an illustrative AgentX failure: single-idea supplies only a vague evidence claim, but single-validate needs its source and finding. A freely generated repair may merely tell idea to provide more evidence. RobustSGPO instead specifies: change both agents’ output/input require-

Only feasible combinations are included; scope and operation need not be independent. Permissions bound allowed changes; factorization directs realized edits within that bound ary.

The exact-target check compares requested and changed object sets. If the request names both idea and validate, changing only idea or also changing propose is rejected. This filters realized edits for object compliance; it does not establish semantic correctness or task quality.

$$
q ( \Delta h \mid h , g ) = \sum _ { s , o , T } p ( s , o \mid h ) p ( T \mid s , o , h ) q _ { \mathrm { L L M } } ( \Delta h \mid h , g , s , o , T ) .
$$

For additions/removals, predefined code constructs the patch. Given new-agent content, it writes the prescribed file and normalizes an existing name field; removal deletes the designated file. A free-form model could write code to do the same, but a checker would only accept or reject its output. The predefined operation performs the repeated editing work itself, helping requested structural edits pass validity checks. Ordinary instruction edits remain model-generated. This is the Typed Compiler in Figure 3.

## 5.2 Construct and Check the Requested Patch

ments to use matching fields. The model then writes the concrete instructions.

The current program does not automatically update caller instructions: creating an evidence agent does not itself insert it between idea generation and review. Loading, references, invocation on trigger tasks, and task quality are checked sepa

This request selects scope � (a pair), operation � (field alignment), and targets � (idea and validate). The controller chooses feasible categories $c = ( s , o )$ and eligible targets before content generation. Given feedback $g ,$

(5)

rately. Candidate admission retains Equation 3; failures also consume budget.

## 5.3 Continue from Retained Versions

Suppose an incumbent scores 4.2. A local rewrite A scores 4.3 and is accepted; a pair edit B scores 4.1 after fixing the handof but making review too strict. These numbers illustrate the selection rule. RobustSGPO can retain B as its category’s best valid snapshot and later refine its review rule. Any descendant must beat the current incumbent, not merely B.

The archive �[�] stores the highest-validation complete harness snapshot for each retained scope–operation category. Membership follows the requested and verified edit. A seeded reservoir selects at most ten category slots; within retained categories, better candidates replace earlier entries. Search alternates incumbent and archive parents, selecting occupied categories uniformly and falling back to the incumbent when needed. Task shifts trigger rescoring before reuse, charged to search cost. This borrows category retention from qualitydiversity search [10], but categorizes edit actions rather than behavioral niches. Retention changes future search parents, not edit permissions.

## 6 Experimental Results

## 6.1 Tasks, Protocol, and Runs

We use 120 tasks in two equally sized families: local instruction failures and cross-agent handof/structure failures. Grouping by source conversation and underlying requirement precedes a 30/15/15 optimization/validation/test split per family, yielding 60/30/30 tasks. Candidate generation receives optimization traces and static edit errors; validation scores guide selection and retention, while test feedback is excluded from search.

Table 1 summarizes runs; all conditions share the initial harness, model, rubric, and proposal budget. We use five paired seeds and three proposals per round, capped at 120 changed lines and 6,000 added characters per proposal. Invalid proposals and retries consume budget. Three paired replays compare candidates and incumbents on identical tasks and execution seeds. Admission requires a mean validation gain above $\epsilon = 0 . 0 5$ and passing safety checks; at most one best valid candidate is accepted per round, with ties retaining the incumbent. Token costs include generation, failures, replay, judging, and archive operations. Fixed test evaluation is separate from search cost.

Table 1: Runs and candidate attempts. All conditions use five seeds; E3 includes two transfer directions and random retention.
<table><tr><td>Study</td><td>Conditions</td><td>Runs</td><td>Rounds</td><td>Attempts</td></tr><tr><td>E1: Scheduling</td><td>6</td><td>30</td><td>30</td><td>2,700</td></tr><tr><td>E2: Ablations</td><td>5</td><td>25</td><td>30</td><td>2,250</td></tr><tr><td>E3: Transfer</td><td>4×2</td><td>40</td><td>20</td><td>2,400</td></tr><tr><td>Total</td><td></td><td>95</td><td></td><td>7,350</td></tr></table>

## 6.2 E1: Periodic Scheduling Improves Final Quality

We compare fixed permissions, two periodic schedules, and a shufle with ten rounds per permission. All use greedy admission without an archive. In Figure 4, final scores for periodic

1 → 2 → 3, periodic $3  2  1$ , and random scheduling are 4.34, 4.24, and 4.17; fixed $\alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 }$ attain 3.85, 4.07, and 4.06. Fixed �<sub>3</sub> leads early, but periodic $1  2  3$ finishes 0.28 points higher.

![](images/1d4fa12b4410359439ea22fa22f581989eccf395ffe0fb7d09c5eb1492d2e22c.jpg)  
Figure 4: E1 test-score trajectories on a 0–5 scale. Periodic $1  2  3$ has the highest final score; early and final rankings difer.

## 6.3 E2: Cumulative Search-Space Controls

Table 2 adds a requested edit, exact-target rejection, predefined agent additions/removals, and category retention. All configurations use �<sub>3</sub> permission and shared safety/replay checks. Controllers uniformly sample feasible categories and eligible targets. Intermediate controls use matched requests. RobustSGPO alternates incumbent and retained parents; every candidate must beat the current incumbent for admission.

Table 2: Cumulative configurations: each row retains all preceding controls. Dashes denote absent additional controls; safety and paired replay apply to every row.
<table><tr><td>Configuration</td><td></td><td>Edit plan Target enforcement</td><td>Fixed structural editing</td><td>Category archive</td></tr><tr><td>SGPO</td><td></td><td></td><td></td><td></td></tr><tr><td>Planned Search</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Constrained Search</td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>Structured Search</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>RobustSGPO</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Quality and coverage. Figure 5(a) reports round-30 scores of 3.82, 3.94, 4.04, 4.20, and 4.30 in table order: Ro bustSGPO improves on SGPO by 0.48 points. Final efectivecategory coverage in Figure 5(b) is 48%, 46%, 48%, 54%, and 68%, respectively. Providing a request alone does not in crease final efective coverage; prescribed edits and retained starting points expand it further. Coverage is the fraction of the fixed scope–operation vocabulary reached by at least one valid edit; repeats add no coverage. It measures category coverage, not full space size or task quality, and does not separate ungenerated from invalid-only categories.

![](images/4512a25edfd82f0d4e0ff379c020dd3078cc256ce39fc5b23d460b421b215f1e.jpg)

Structural proposals and reuse. Valid/attempted counts in Figure 5(c) are 9/14, 23/50, 22/45, 28/36, and 26/34. Structured Search achieves 77.8% validity versus 48.9% for Constrained Search, reducing unsuccessful structural proposals. Rates are undefined before any structural attempt. Figure 5(d) shows ten occupied categories and 17 cumulative archiveorigin updates that pass incumbent admission. These retained versions supply used starting points; Section 6.5 accounts for their overhead.

![](images/3cdd681ec1b1182309493743ea6f0b41d8e4b1adf6fb875e15b8a0fc70e0b32f.jpg)

![](images/6020e38ed0f93354bd505ef660e013df1f3233310c7bd8bd876ae9d5b77daee7.jpg)

![](images/ceab776c7302a1239fe96992cc563569f695f7e30b3aa0546d0ffe6094bac6cf.jpg)  
Figure 5: E2 search behavior: (a) held-out quality, (b) cumulative efective-category coverage, (c) cumulative structural validity with valid/attempted counts, and (d) occupied archive categories and cumulative admitted archive-origin updates.

## 6.4 E3: Adaptation and Retention After a Task Shift

We evaluate local-to-cross-agent failures and the reverse. Each run spends ten rounds on source tasks and ten on destination tasks, using the respective optimization and validation splits. Retained snapshots are rescored on the destination validation set before reuse. We compare SGPO, Structured Search, RobustSGPO, and RobustSGPO (Random Archive). Random retention uses the same capacity, parent-use frequency, rescore budget, and admission rule, but stores valid snapshots by reservoir sampling rather than category-wise quality.

![](images/d0d33322cf0266277f83f900b115f8203c2c9b16e1b6d4c4cb74bf3f73192262.jpg)

![](images/09cd0882b734cb02eca7a3b1d09652aa584a838621e6fdb8b851b89a7ca7320c.jpg)  
Figure 6: E3 adaptation and retention after the round-ten task shift. Dashed curves use random retention; category retention reduces sourcetask loss, while random retention reaches a higher destination endpoint.

Figure 6 tracks destination tasks on the left and source tasks on the right throughout the run. Mean destination scores over rounds 11–20 are 3.72, 3.67, 3.74, and 3.75; final scores are 4.06, 3.96, 3.98, and 4.18. RobustSGPO improves mean adaptation quality over Structured Search, while random retention obtains the highest destination endpoint.

Final source scores decline by 0.33, 0.31, 0.15, and 0.31 points relative to round ten. Category retention has the smallest source-task loss. Random retention improves destination quality further but loses more source capability. The archive policy therefore changes the adaptation–retention tradeof: category elites benefit retention rather than maximizing the destination endpoint.

Table 3: Held-out task results and equal-token comparison. Local and cross-agent sets contain 15 tasks each. Round-30 scores correspond to Figure 5(a); the budget column uses the last afordable checkpoint within 20 million tokens. Bold marks column bests, including ties.
<table><tr><td>Configuration</td><td>Round 30</td><td>Local</td><td>Cross-agent</td><td>Success</td><td>20M tokens</td></tr><tr><td>SGPO</td><td>3.82</td><td>11/15</td><td>7/15</td><td>60.0%</td><td>3.77</td></tr><tr><td>Planned Search</td><td>3.94</td><td>11/15</td><td>8/15</td><td>63.3%</td><td>3.84</td></tr><tr><td>Constrained Search</td><td>4.04</td><td>12/15</td><td>8/15</td><td>66.7%</td><td>3.99</td></tr><tr><td>Structured Search</td><td>4.20</td><td>12/15</td><td>11/15</td><td>76.7%</td><td>4.10</td></tr><tr><td>RobustSGPO</td><td>4.30</td><td>13/15</td><td>11/15</td><td>80.0%</td><td>4.14</td></tr></table>

## 6.5 Behavioral Outcomes at Equal Token Cost

Table 3 evaluates actual task completion, required artifacts, protected operations, and cross-agent handofs. RobustSGPO completes 24/30 tasks versus SGPO’s 18/30: local completion rises from 11/15 to 13/15, and cross-agent completion from 7/15 to 11/15.

![](images/09e77f16fb52d2ff24d684030e6e8b8c905205f322ff44d0907859dc9bc59ac2.jpg)  
Figure 7: Test quality versus cumulative charged tokens, including all search-side model calls. Dots mark the last afordable checkpoints within 20 million tokens.

Per-round costs in Figure 7 are 0.80, 0.83, 0.78, 0.82, and 0.95 million in table order. Within 20 million tokens, the final afordable rounds are 25, 24, 25, 24, and 21, yielding scores of 3.77, 3.84, 3.99, 4.10, and 4.14. RobustSGPO exceeds SGPO by 0.37 points at equal cost. Its margin over Structured Search shrinks from 0.10 at equal rounds to 0.04 at equal token cost, reflecting archive overhead. Checkpoints are selected by budget, not by test score.

## 7 Conclusion

RobustSGPO decomposes free-form SGPO updates into requested edits, patch construction and checking, and retained search starting points. Periodic permissions improve final quality, predefined add/remove operations improve structural validity, and the full method outperforms free search at equal token cost. Category retention reduces source-task degradation after transfer while adding search overhead. These results support explicit control of edit choices and search starting points in multi-agent harness evolution.

## References

[1] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning, 2025.

[2] Tianyu Fan and Chao Huang. HELIX: Model-Harness Co-evolution for Recursive Self-Improvement, 2026.

[3] Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. Promptbreeder: Self-referential self-improvement via prompt evolution, 2023.

[4] Changxin Lao, Fei Pan, Guozhuang Ma, Han Li, Huihuang Lin, Jijun Shi, et al. AgentX: Towards agentdriven self-iteration of industrial recommender systems, 2026.

[5] Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-Harness: End-to-End Optimization of Model Harnesses, 2026.

[6] Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Zhiheng Xi, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui, and Yu-Gang Jiang. Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses, 2026.

[7] Hanzhi Liu, Chaofan Shou, Xiaonan Liu, Hongbo Wen, Yanju Chen, Ryan Jingyang Fang, and Yu Feng. Synthe-

sizing Multi-Agent Harnesses for Vulnerability Discovery, 2026.

[8] Zewen Liu, Zhan Shi, Yisi Sang, Bing He, Minhua Lin, Tianxin Wei, Dakuo Wang, Benoit Dumoulin, Wei Jin, and Hanqing Lu. Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams, 2026.

[9] Haochen Luo, Yi Huang, Sichun Luo, Fengyuan Liu, Lei Li, Zefa Hu, Junlan Feng, and Qi Liu. Harness-Aware Self-Evolving: Co-Evolving Model Weights, Harness, and Task Solutions, 2026.

[10] Jean-Baptiste Mouret and Jef Clune. Illuminating search spaces by mapping elites, 2015.

[11] Biswa Sengupta and Jinhua Wang. HARBOR: Automated Harness Optimization, 2026.

[12] Tianxin Wei, Zhan Shi, Minhua Lin, Bing He, Zewen Liu, Yisi Sang, Yuanchen Bei, Xuying Ning, Jiaru Zou, Ting-Wei Li, Xiao Lin, Yanjun Zhao, Chi Wang, Benoit Dumoulin, Dakuo Wang, Jingrui He, and Hanqing Lu. Evo-Harness: Context-to-Harness Skill Compilation for Self-Evolving Agents, 2026.

[13] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers, 2023.

[14] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic “diferentiation” via text, 2024.

[15] Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, and Shuyue Hu. Self-Harness: Harnesses That Improve Themselves, 2026.