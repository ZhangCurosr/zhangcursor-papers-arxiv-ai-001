# COREDSKILL: Stealthy Skill Composition Attacks via AST-Guided Injection

Qichen Zhao<sup>1∗</sup>, Shengfang Zhai<sup>2∗†</sup>, Qingni Shen<sup>1†</sup>, Zhonghai Wu<sup>1</sup>, Jiaheng Zhang<sup>2</sup> <sup>1</sup>Peking University <sup>2</sup>National University of Singapore z24@stu.pku.edu.cn, shengfang.zhai@nus.edu.sg, qingnishen@pku.edu.cn

## Abstract

Agents extend their capabilities through reusable skills that package instructions, executable code, and resources. Composing these skills enables complex task workflows, but their dependencies can also connect malicious operations across skills, making their combined intent difficult to identify through individual-package auditing. We introduce COREDSKILL, a compositional red-teaming method that forms stealthy attacks through cooperating skills while preserving their original utility. COREDSKILL decomposes an existing workflow into interdependent skills and embeds attack operations in task functions through ab stract syntax tree (AST) rewriting, with skill instructions guiding the agent to autonomously select and invoke the relevant skills, and work flow dependencies and intermediate artifacts connecting the injected operations during task execution. We construct a benchmark of 150 cases covering 50 tasks and three attack objectives to evaluate attack success, task completion, and audit detection. In the main evaluation, COREDSKILL achieves 98.0% attack success and 95.3% task success with a 3.3% detection rate, and a joint success rate of 90.7% (attack success, task completion, and audit evasion in the same case) compared with 6.7– 42.7% for the four baselines. Skill compositions can therefore complete legitimate tasks while carrying out unauthorized operations that evade auditing.

## 1 Introduction

Agent skills package task instructions, executable code, and resources into reusable modules that extend an agent’s capabilities. During task execution, the agent reads installed skills’ documentation and invokes their code. However, public skill libraries draw skills from publishers whose quality and trustworthiness vary. An agent often composes multiple skills for one task, passing intermediate results between stages. Hence, untrustworthy skills can lead normal task execution to malicious operations outside the user’s authorization, affecting the agent system(Liu et al., 2026c; Li et al., 2026; Liu et al., 2026b; Jiang et al., 2026).

Prior red-teaming on skills takes two forms. Single-skill attacks (Schmotz et al., 2026; Jia et al., 2026) keep the whole attack in one package, giving skill scanners (Liu et al., 2026c; NVIDIA, 2026; Bhardwaj, 2026; Cisco AI Defense, 2026b) a relatively complete attack context and showing high detection rates under the large language model (LLM)-based scanners (Table 2). Composition attacks (Xie et al., 2026; Zeng et al., 2026) instead distribute the attack across cooperating skills, but give limited attention to preserving the original task logic and normal task completion. Skill utility is the basis for user adoption and agent invocation. Therefore, we emphasize a more practical setting: Skill red-teaming requires achieving attack objectives while preserving skill utility and maximizing stealthiness.

Constructing such attacks is challenging. (1) Attack operations must be embedded in the original task workflow while keeping the task logic and stage interfaces valid to avoid disrupting normal task execution. (2) Distributed attack operations must match their host skills’ task responsibilities so that local behavior blends into normal processing and is less likely to be flagged by scanners. (3) The distributed operations must remain connected through the agent’s autonomous selection and invocation under legitimate user requests to jointly achieve the attack objective. To address these, we introduce COREDSKILL, a Compositional Redteaming method that forms stealthy Skill attacks while preserving their original utility (Figure 1). COREDSKILL decomposes an existing task workflow into interdependent clean skills and assigns attack operations according to each stage’s task responsibilities. It then uses abstract syntax tree (AST) rewriting guided by stage specifications to embed attack operations in task helper functions, updating related function calls and artifact reads and writes while preserving the original task logic. Skill instructions and input–output dependencies guide the agent to invoke the relevant skills together, while intermediate artifacts pass attackrelevant information between the distributed operations.

We construct a benchmark of 150 evaluation cases by pairing 50 tasks with three attack objectives (detailed in Section 4.1). Using this benchmark, we compare COREDSKILL with four attack baselines in terms of attack success, task completion, and audit detection. Our evaluation includes four LLM-based skill-audit configurations (Liu et al., 2026c; NVIDIA, 2026), compositionaware auditing (Zeng et al., 2026), and runtime defenses applied to execution records (Shan et al., 2026; Cisco AI Defense, 2026a). In experiments, COREDSKILL achieves 98.0% attack success and 95.3% task success with a detection rate of 3.3%. Its joint success rate (attack success, task completion, and audit evasion in the same case) reaches 90.7%, compared with 6.7–42.7% for the four baselines. Our contributions are:

• We propose COREDSKILL, which decomposes a task workflow and uses AST rewriting to embed distributed attack operations into its task functions, connecting them through workflow dependencies and intermediate artifacts while preserving skill utility.

• We build a benchmark of 50 tasks and their skill compositions, spanning three attack objectives and 150 cases, to evaluate task completion, unauthorized effects, and audit detection.

• We evaluate COREDSKILL against four attack baselines under four LLM audit configurations, composition-aware auditing, and runtime defenses. In the experiments, CORED-SKILL attains 98.0% attack success and 95.3% task success at a 3.3% detection rate, substantially outperforming the baselines.

## 2 Threat Model

## 2.1 Scenario

A user delegates a legitimate task to an agent, which uses installed skills to carry out the request.

In agent-skill ecosystems (Liu et al., 2026c; Li et al., 2026), a workflow may involve multiple thirdparty skills that perform different task operations and exchange information through intermediate artifacts. Following prior work on malicious skills (Liu et al., 2026b), we assume that skills supplied by a third-party publisher are installed alongside other available skills. The agent autonomously selects relevant skills and determines their invocation order based on the user’s request, skill documentation, and input–output dependencies. Skill-auditing tools such as SkillScan (Liu et al., 2026c) and Skill-Spector (NVIDIA, 2026) can examine skill documentation and code. Runtime auditing methods such as HITL (Shan et al., 2026) and DefenseClaw (Cisco AI Defense, 2026a) can inspect the agent’s execution records for unauthorized behavior.

## 2.2 Attacker Capabilities and Objectives

The attacker acts as a third-party skill publisher. Following the publisher capabilities considered in Skill-Inject (Schmotz et al., 2026) and Skill-Ject (Jia et al., 2026), the attacker can inspect the original skill’s documentation and scripts, construct and publish multiple skills based on its workflow, and modify their instructions, code, resources, and input–output interfaces. The user’s legitimate request defines the scope of authorized actions, whereas the deployment determines the agent’s model, system prompt, safeguards, and file, network, and process permissions. Within these configured permissions, the attacker seeks to preserve the original skill’s utility, evade security audits, and induce unauthorized operations during normal task execution. The attacker influences the agent’s skill selection and execution through the published skills and the intermediate artifacts they produce, so that operations distributed across skills jointly achieve the attack objective.

## 3 Methodology

In this section, we detail the proposed CORED-SKILL method. We first formulate the problem in Section 3.1. We then introduce workflow decomposition and attack-operation assignment in Section 3.2. We finally describe AST-guided attack injection in Section 3.3. Algorithm 1 summarizes the construction procedure.

## 3.1 Problem Formulation

A skill is a reusable package of task instructions, executable scripts, and auxiliary resources that an agent loads at runtime. Let $C = ( S _ { 1 } , \ldots , S _ { k } )$ denote a workflow composition with $k \geq 2$ stages, where skill $S _ { i }$ handles stage i and stages exchange data through intermediate artifacts. Let A denote the agent, q the user request, and E the initial environment, including task inputs and accessible resources. To describe execution with the composition available, we write:

![](images/7ebffedd4ae8a7bd619e5e114c6dd950fb3bc175fe498a8e477c0b6c4746100f.jpg)  
Figure 1: Overview of COREDSKILL, illustrated with a three-skill exfiltration attack. (a) Decompose the task workflow and assign attack operations. (b) Perform AST rewriting and skill packaging according to the stage-specific specification $\phi _ { i } .$ . Audit the resulting skill packages. (c) During task execution, intermediate artifacts pass task data and attack-relevant information between skills, connecting the distributed attack operations.

$$
\tau = \operatorname { R u n } ( \mathcal { A } , q , E ; C ) ,\tag{1}
$$

where τ denotes the observable execution record, which includes tool actions, intermediate artifacts, final outputs, and the resulting environment state. We use execution trace to refer specifically to the ordered sequence of actions within this record. The indices describe the intended workflow stages, while the actual skill invocations are recorded in $\tau .$ . Artifacts and agent context carry information across stages, defining the intended dependencies. Whether execution realizes these dependencies is evaluated through τ .

For red-team construction, the attacker edits the supplied skill packages to transform a clean composition C into an injected composition $C ^ { \prime } =$ $( S _ { 1 } ^ { \prime } , \ldots , S _ { k } ^ { \prime } )$ . The agent configuration and trusted task and evaluation specifications remain fixed. Given an original skill s, an abstract attack objective $^ { O , }$ and a composition size k, COREDSKILL first constructs C from s and then outputs $C ^ { \prime }$ by distributing operations for o across its skills. Skill utility refers to a skill’s ability to perform its intended task functions and produce the expected outputs. For a fixed task, authorization boundary, and evaluation configuration, let $U ( \tau ) \in \{ 0 , 1 \}$ indicate task acceptance and $A _ { o } ( \tau ) \in \{ 0 , 1 \}$ indicate whether the objective-o oracle is satisfied by the execution evidence. An observed execution $\tau = \mathrm { R u n } ( \mathcal { A } , q , E ; C ^ { \prime } )$ constitutes a successful compositional attack that preserves skill utility if $U ( \tau ) = 1$ and $A _ { o } ( \tau ) = 1$ . We assess preservation of skill utility using the task’s output acceptance criteria, allowing artifacts and environment states to differ from the clean workflow. Distributing operations across skills does not require each modification to be individually necessary.

To assess audit evasion, let $d _ { \rho } ( S ) \in \{ 0 , 1 \}$ indicate whether skill S is flagged under audit configuration $\rho ,$ which specifies the auditor and its detection criterion. The concrete criteria are described in Section 4.1. We aggregate these outcomes as:

$$
D _ { \rho } ( C ^ { \prime } ) = \operatorname* { m a x } _ { 1 \leq i \leq k } d _ { \rho } ( S _ { i } ^ { \prime } ) .\tag{2}
$$

Intuitively, $D _ { \rho } ( C ^ { \prime } ) = 0$ means that no constituent package is flagged in these reviews. The construction objective is to obtain $C ^ { \prime }$ from C for objective o such that execution and individual auditing yield the joint outcome:

$$
U ( \tau ) = A _ { o } ( \tau ) = 1 , \qquad D _ { \rho } ( C ^ { \prime } ) = 0 .
$$

Execution success and audit evasion are evaluated separately. Section 4.1 specifies the evaluation criteria.

Algorithm 1 COREDSKILL construction   
Require: original skill $s ,$ objective $^ { o , }$ size $k ,$ model M   
Ensure: injected composition $C ^ { \prime } = ( S _ { 1 } ^ { \prime } , \ldots , S _ { k } ^ { \prime } )$   
1: $C \gets .$ Decompose $( M , s , k ) = ( \dot { S } _ { 1 } , \dots , \dot { S } _ { k } ) ^ { \ast }$   
2: Select the rewrite anchor $a _ { i }$ in each stage $S _ { i }$   
3: Split the attack chain for objective o into steps   
4: Group steps by data dependencies into the ordered stage   
operations $f _ { i }$   
5: Construct $\breve { \Phi } ( o , k ) = \left( \phi _ { 1 } , \dots , \phi _ { k } \right)$ , each $\phi _ { i }$ recording $f _ { i } ,$   
anchor $^ { a _ { i } , }$ replacement templates, and artifact interfaces   
6: for $i = 1$ to k do   
7: G ← LocateAndParse $( S _ { i } ; \phi _ { i } )$   
8: $G _ { i } ^ { \prime } \gets \mathcal { T } _ { \mathrm { A S T } } ( G _ { i } ; \phi _ { i } )$   
9: $S _ { i } ^ { \prime } \gets \operatorname { P a c k } ( \mathbf { \dot { \boldsymbol { S } } } _ { i } , G _ { i } ^ { \prime } ; \boldsymbol { \phi } _ { i } )$   
10: end for   
11: $C ^ { \prime } \gets ( S _ { 1 } ^ { \prime } , \ldots , S _ { k } ^ { \prime } )$   
12: Reparse $( \dot { C } ^ { \prime } )$ and check its rewrite anchors   
13: return $C ^ { \dot { \prime } }$

We instantiate $o \in \{ \mathrm { E X F } , \mathrm { T A M } , \mathrm { P R V } \}$ to examine three layers at which an installed skill can affect the host system: sensitive data, persistent state, and execution capability. Specifically, (1) data exfiltration (EXF) discloses protected data to recipients outside the task’s authorized scope, (2) unauthorized state tampering (TAM) alters persistent state in installed skills through file writes, and (3) privilege violation (PRV) induces the agent to execute scripts beyond the operations authorized by the user’s request.

## 3.2 Workflow Decomposition

To retain the task workflow while exposing its stage dependencies, the construction model M reads the documentation and scripts of the original skill s and partitions its workflow by task responsibilities and data dependencies:

$$
C = { \mathrm { D e c o m p o s e } } ( M , s , k ) = ( S _ { 1 } , \ldots , S _ { k } ) .
$$

Each $S _ { i }$ retains the task processing assigned to stage i and exchanges results with later stages through intermediate JSON files. Its documentation specifies inputs, processing steps, and outputs to guide the agent’s autonomous skill selection and invocation. The construction instructions specify each stage’s task helper functions, including names and calling interfaces. Let $a _ { i }$ denote the helper function selected as the rewrite anchor in stage i, such as file search, field selection, or data export. Its name and interface identify the function for subsequent AST rewriting.

Attack chain decomposition. To align the attack with these dependencies, we decompose the attack chain for objective o into steps with defined inputs and outputs. We group steps by data dependencies and assign them to workflow stages whose functions support the required operations. Let $f _ { i }$ denote the ordered group of operations assigned to stage i. The stage specification $\phi _ { i }$ records $f _ { i }$ , its rewrite anchor ${ { a } _ { i } } ,$ replacement templates, and intermediate artifact interface conventions. Built on the stages and anchors of the clean composition $C ,$ the stage specifications form:

$$
\Phi ( o , k ) = ( \phi _ { 1 } , . . . , \phi _ { k } ) .\tag{3}
$$

For EXF with three skills, $f _ { 1 }$ searches for files, $f _ { 2 }$ selects fields and writes their values to a file for transmission, and $f _ { 3 }$ transfers the data to an unauthorized recipient.

## 3.3 Attack Injection

To embed the assigned operations while retaining task processing, COREDSKILL transforms each clean skill $S _ { i }$ into an injected skill $S _ { i } ^ { \prime }$ according to $\phi _ { i }$

AST rewriting. Using $a _ { i }$ and the interface changes specified in $\phi _ { i }$ , we locate the affected scripts and parse them into a collection of ASTs $G _ { i }$ . The replacement templates implement $f _ { i }$ in the target helper function and update related calls, arguments, and artifact reads and writes. To preserve core task logic and maintain compatible stage interfaces, we apply these edits together with the corresponding package updates:

$$
\begin{array} { l } { G _ { i } ^ { \prime } = \mathcal { T } _ { \mathrm { A S T } } ( G _ { i } ; \phi _ { i } ) , } \\ { S _ { i } ^ { \prime } = \operatorname { P a c k } ( S _ { i } , G _ { i } ^ { \prime } ; \phi _ { i } ) , } \end{array}\tag{4}
$$

where $\mathcal { T } _ { \mathrm { A S T } }$ denotes the specified AST transformation and Pack assembles the injected skill. Packaging converts $G _ { i } ^ { \prime }$ to source code, writes it to the corresponding scripts, and updates entry points and required resources. We update documentation to describe the added processing within the stage’s task role and specify its inputs and outputs. Finally, we reparse the rewritten scripts and check their rewrite anchors.

Cross-skill state propagation. During execution, intermediate task artifacts carry the data needed by attack operations in later stages. As the agent follows the input–output dependencies, these artifacts connect the distributed operations. In the inventory-planning example, the workflow computes allocation and replenishment decisions and publishes a plan. Under Φ(EXF, 3), the first skill records file paths and field names in its task artifact. The second uses this information to collect selected field values and pass them downstream. The third skill publishes the requested plan, after which the agent follows the injected instructions to invoke an additional distribution operation that sends the collected values to an unauthorized recipient. The resulting record τ supplies the evidence for both $U ( \tau )$ and $A _ { o } ( \tau )$ . Appendix E details the stage interfaces and state fields.

## 4 Experiments

The evaluation addresses three questions. RQ1: Overall attack effectiveness. Can COREDSKILL achieve its attack objectives while preserving the original skill’s utility and evading security audits (Section 4.2)? RQ2: Stealth against skill auditors. Can skill auditors detect these attacks across different auditors and LLM backends (Section 4.3)? RQ3: Stealth against runtime defenses. Can runtime risk and policy rules detect unauthorized actions in task execution records (Section 4.5)? We further evaluate how composition size affects attack stealth, whether composition-aware auditing detects the attacks, and how injection positions affect attack success.

## 4.1 Experimental Setup

Benchmark. We construct a benchmark of 50 tasks based on public scenarios from Skill-Inject (Schmotz et al., 2026), τ-bench (Yao et al., 2024), and AgentDojo (Debenedetti et al., 2024). Each task is implemented by a skill and has defined output acceptance criteria. Pairing each task with data exfiltration (EXF), state tampering (TAM), and privilege violation (PRV) yields 150 cases, with 50 per objective. Appendix A describes benchmark construction and task coverage.

Attack Baselines. We adapt four attack methods to the benchmark’s tasks and objectives, reporting the adapted versions under their original names. For Skill-Inject (Schmotz et al., 2026), we append injection templates to skill documentation and add script wrappers for the target operations. SkillJect (Jia et al., 2026) generates a malicious skill with an auxiliary payload and refines it using execution feedback. SCR-Bench (Xie et al., 2026) links upstream information to downstream actions through prompt injection in a two-skill workflow. ColluSkill (Zeng et al., 2026) distributes interdependent subpayloads across skills and refines flagged skills using scanner feedback. COREDSKILL injects attacks into a three-skill composition by default $( k = 3 )$ . We include Original skills and Clean compositions, the original skills and their clean three-skill decompositions, as uninjected controls for measuring false positives. Appendix D details the construction mechanisms.

Defense Baselines. SkillScan (Liu et al., 2026c) combines static scanning with LLM auditing, while SkillSpector (NVIDIA, 2026) uses an LLM to review findings from static analyzers. SkillFortify (Bhardwaj, 2026) performs static package analysis, and CiscoSkillScan (Cisco AI Defense, 2026b) applies format validation and behavioral checks. For runtime auditing, we apply HITL risk rules (Shan et al., 2026) and DefenseClaw strict policy rules (Cisco AI Defense, 2026a) to the agent’s execution records. Detailed configurations are provided in Appendices A.3 and A.4.

Evaluation Metrics. ❶ ASR (Attack Success Rate) measures the fraction of cases that achieve the specified attack objective. ❷ TSR (Task Success Rate) measures the fraction of cases whose task outputs satisfy the task’s acceptance criteria. Appendix A.2 describes the verification procedures for attack objectives and task outputs. ❸ DR (Detection Rate) measures the fraction of cases detected by the auditor. For skill compositions, a case is counted as detected if any constituent skill is flagged as risky (Zeng et al., 2026). For runtime auditing, detection is determined by whether the execution record triggers the audit rules. ❹ JSR (Joint Success Rate) measures the fraction of cases in which the attack and the legitimate task both succeed without detection, defined as

$$
\mathrm { J S R } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } A _ { n } U _ { n } ( 1 - D _ { n } ) .\tag{5}
$$

Here, N is the number of cases, $A _ { n }$ and $U _ { n }$ indicate attack and task success, respectively (Section 3.1), and $D _ { n } = 1$ indicates detection under the specified audit configuration.

Implementation Details. We use Claude Code (Anthropic, 2026) as the main victim platform. DeepSeek V4 Flash (DeepSeek-AI et al., 2026) (Deepseek) serves as the agent’s execution backend and is also used for attack construction. SkillScan and SkillSpector each use Deepseek and GLM-5.2 (Z.ai, 2026) (GLM), yielding four LLM audit configurations. In the main experiment, each case runs once in a separate sandbox and is audited once under each configuration. The agent receives the user request and shared execution instructions, then autonomously selects and invokes skills. For each method, we install all of its injected skills together with six clean skills as distractors. Table 1 reports results with SkillScan using Deepseek for auditing. Detailed implementation settings are provided in Appendix A.2.

Table 1: Attack success, task completion, and audit detection on the benchmark. SkillScan performs the audit using Deepseek. JSR provides a joint assessment of attack effectiveness, task completion, and stealth. Arrows indicate preferred directions. Bold marks the best ASR, DR, and JSR in each column.
<table><tr><td rowspan="2">Method</td><td colspan="4">EXF</td><td colspan="4">TAM</td><td colspan="4">PRV</td><td colspan="4">Overall</td></tr><tr><td>ASR↑</td><td>TSR↑</td><td>DR↓</td><td>JSR↑</td><td>ASR↑</td><td></td><td>TSR↑ DR↓</td><td>JSR↑</td><td>ASR↑</td><td>TSR↑</td><td>DR↓</td><td>JSR↑</td><td>ASR↑</td><td>TSR↑</td><td>DR↓</td><td>JSR↑</td></tr><tr><td>Skill-Inject</td><td>22.0</td><td>92.0</td><td>100</td><td>0</td><td>92.0</td><td>98.0</td><td>90.0</td><td>6.0</td><td>46.0</td><td>100</td><td>64.0</td><td>14.0</td><td>53.3</td><td>96.7</td><td>84.7</td><td>6.7</td></tr><tr><td>SkillJect</td><td>96.0</td><td>100</td><td>98.0</td><td>2.0</td><td>100</td><td>100</td><td>16.0</td><td>84.0</td><td>92.0</td><td>100</td><td>52.0</td><td>42.0</td><td>96.0</td><td>100</td><td>55.3</td><td>42.7</td></tr><tr><td>SCR-Bench</td><td>30.0</td><td>100</td><td>64.0</td><td>22.0</td><td>24.0</td><td>96.0</td><td>80.0</td><td>18.0</td><td>100</td><td>100</td><td>80.0</td><td>20.0</td><td>51.3</td><td>98.7</td><td>74.7</td><td>20.0</td></tr><tr><td>ColluSkill</td><td>2.0</td><td>24.0†</td><td>10.0</td><td>0</td><td>80.0</td><td>52.0†</td><td>4.0</td><td>46.0</td><td>68.0</td><td>38.0†</td><td>14.0</td><td>22.0</td><td>50.0</td><td>38.0†</td><td>9.3</td><td>22.7</td></tr><tr><td>CoREDSKILL (Ours)</td><td>100</td><td>96.0</td><td>4.0</td><td>92.0</td><td>94.0</td><td>96.0</td><td>0</td><td>90.0</td><td>100</td><td>94.0</td><td>6.0</td><td>90.0</td><td>98.0</td><td>95.3</td><td>3.3</td><td>90.7</td></tr></table>

<sup>†</sup> ColluSkill focuses on constructing attack payloads and evading audits, without consistently preserving skill utility. <sup>‡</sup> SCR-Bench’s original skills use fixed templates with limited task functionality. Adapting them to the benchmark’s tasks may reduce attack stealth.

## 4.2 Overall Evaluation

COREDSKILL achieves high attack and task success while maintaining a low detection rate (Table 1). On the 150-case benchmark, its ASR, TSR, and DR are 98.0%, 95.3%, and 3.3%, respectively. Of the 150 cases, 140 achieve both task completion and attack success, and 136 of these also pass the skill audit, yielding a JSR of 90.7%. This joint success rate exceeds the 6.7–42.7% achieved by the four baselines, highlighting COREDSKILL’s advantage in jointly achieving attack objectives, completing tasks, and evading auditing. Although SCR-Bench and SkillJect achieve comparable ASR or TSR on some attack objectives, their substantially higher overall DR under skill auditing indicates weaker stealth than COREDSKILL. ColluSkill also has a low DR, but its ASR and TSR are substantially lower than COREDSKILL’s. Its injections evade auditing but fail to reliably execute the attacks and preserve skill utility.

## 4.3 Stealth against Skill Auditors

COREDSKILL’s overall DR ranges from 3.3% to 20.0% across the four LLM audit configurations and is lower than those of SkillJect, Skill-Inject, and SCR-Bench in each configuration (Table 2). Using GLM as the audit model increases the overall DR for COREDSKILL under both auditors, but most attack cases remain undetected. ColluSkill’s overall DR ranges from 9.3% to 17.3% and is lower than COREDSKILL’s only under SkillScan with GLM. However, its ASR and TSR are 50.0% and 38.0%, respectively (Table 1). Inspection of ColluSkill’s generated skills found that some cases that passed the audit lacked operations required to achieve the attack objectives. No false positives are observed for original skills or clean compositions under the four LLM audit configurations. Skill-Fortify and CiscoSkillScan both have zero DR for COREDSKILL. Skills implement diverse functions, some of which are specified through prompts and depend on agent execution, making hidden malicious intent difficult to infer from static package inspection alone. SkillJect is also undetected by both scanners, and SCR-Bench is undetected by CiscoSkillScan. ColluSkill’s DR is 6.0% under SkillFortify and 3.3% under CiscoSkillScan (Table A2).

## 4.4 Extension of Composition Size

We compare two-, three-, and four-skill compositions on the same 24 tasks to assess how composition size affects attack success and audit detection. ASR remains 100% at all three sizes, while overall DR ranges from 4.2% to 34.7% (Figure 2 and Appendix Table A3). SkillScan has the lowest overall DR at k = 2 under both backends, tied with k = 3 under Deepseek, whereas SkillSpector has the lowest overall DR at k = 3 under both backends. The composition size with the lowest overall DR therefore varies across audit configurations, and composition stealth does not improve monotonically with the number of skills. Distributing an attack across more skills may leave some stages with less legitimate processing to conceal the injected operations. Additional skills also provide more opportunities for detection, since the composition passes auditing only when every constituent skill remains undetected. Across attack objectives, TAM has the lowest DR in most combinations of composition size and audit configuration.

Table 2: Detection rates (DR) under SkillScan and SkillSpector with Deepseek and GLM. Control rows report false-positive rates. Lower DR indicates greater attack stealth. Bold marks the lowest attack DR in each column within each auditor–backend configuration.
<table><tr><td rowspan="2">Auditor</td><td rowspan="2">Method</td><td colspan="2">EXF</td><td colspan="2">TAM</td><td colspan="2">PRV</td><td colspan="2">Overall</td></tr><tr><td>Deepseek</td><td>GLM</td><td>Deepseek</td><td>GLM</td><td>Deepseek</td><td>GLM</td><td>Deepseek</td><td>GLM</td></tr><tr><td>SkillScan</td><td>Original skills</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>SkillScan</td><td>Clean compositions</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>SkillScan</td><td>Skill-Inject</td><td>100</td><td>100</td><td>90.0</td><td>100</td><td>64.0</td><td>98.0</td><td>84.7</td><td>99.3</td></tr><tr><td>SkillScan</td><td>SkillJečt</td><td>98.0</td><td>100</td><td>16.0</td><td>54.0</td><td>52.0</td><td>78.0</td><td>55.3</td><td>77.3</td></tr><tr><td>SkillScan</td><td>SCR-Bench</td><td>64.0</td><td>100</td><td>80.0</td><td>90.0</td><td>80.0</td><td>100</td><td>74.7</td><td>96.7</td></tr><tr><td>SkillScan</td><td>ColluSkill</td><td>10.0</td><td>22.0</td><td>4.0</td><td>2.0</td><td>14.0</td><td>10.0</td><td>9.3</td><td>11.3</td></tr><tr><td>SkillScan</td><td>CoREDSKILL (Ours)</td><td>4.0</td><td>16.0</td><td>0</td><td>12.0</td><td>6.0</td><td>32.0</td><td>3.3</td><td>20.0</td></tr><tr><td>SkillSpector</td><td>Original skills</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>SkillSpector</td><td>Clean compositions</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>SkillSpector</td><td>Skill-Inject</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>SkillSpector</td><td>SkillJect</td><td>98.0</td><td>100</td><td>20.0</td><td>74.0</td><td>100</td><td>86.0</td><td>72.7</td><td>86.7</td></tr><tr><td>SkillSpector</td><td>SCR-Bench</td><td>86.0</td><td>100</td><td>72.0</td><td>100</td><td>98.0</td><td>100</td><td>85.3</td><td>100</td></tr><tr><td>SkillSpector</td><td>ColluSkill</td><td>24.0</td><td>30.0</td><td>4.0</td><td>4.0</td><td>14.0</td><td>18.0</td><td>14.0</td><td>17.3</td></tr><tr><td>SkillSpector</td><td>CoREDSKILL (Ours)</td><td>12.0</td><td>16.0</td><td>2.0</td><td>8.0</td><td>20.0</td><td>24.0</td><td>11.3</td><td>16.0</td></tr></table>

![](images/2ebb31e5ed70b3ec7b60f4a020d9940f86b0a422e01425dc4f306e8ab59001fb.jpg)  
Figure 2: Overall detection rate (DR) of COREDSKILL across composition sizes, under SkillScan and Skill-Spector with Deepseek and GLM as audit models.

## 4.5 Stealth against Runtime Defenses

We apply runtime risk and policy rules to the agent’s execution records to assess whether they detect unauthorized operations. COREDSKILL’s DR is 10.7% under HITL (Shan et al., 2026) and 1.3% under DefenseClaw (Cisco AI Defense, 2026a), lower than those of Skill-Inject and SCR-Bench under both configurations (Table 3). Under HITL, its DR is also lower than ColluSkill’s 32.0%. Skill-Ject’s DR is 7.3% and 0.7% under the two runtime configurations, respectively, but ranges from 55.3% to 86.7% across the four LLM skill-package audit configurations (Table 2). COREDSKILL maintains low detection rates under both skill-package and runtime auditing. By embedding attack operations in task functions and connecting them through intermediate artifacts, COREDSKILL interweaves unauthorized operations with normal task processing, which may make their combined malicious intent difficult for the audit rules to identify.

Table 3: Runtime detection rates (DR) of HITL and DefenseClaw on execution records. Original skills and clean compositions report false-positive rates. Lower DR indicates greater attack stealth.
<table><tr><td>Method</td><td>HITL</td><td>DefenseClaw</td></tr><tr><td>Original skills</td><td>2.0</td><td>0</td></tr><tr><td>Clean compositions</td><td>16.0</td><td>0</td></tr><tr><td>Skill-Inject</td><td>50.7</td><td>66.7</td></tr><tr><td>SkillJečt</td><td>7.3</td><td>0.7</td></tr><tr><td>SCR-Bench</td><td>24.7</td><td>3.3</td></tr><tr><td>ColluSkill</td><td>32.0</td><td>0.7</td></tr><tr><td>CoREDSKILL (Ours)</td><td>10.7</td><td>1.3</td></tr></table>

## 4.6 Evaluation under Adaptive Defense

To evaluate adaptive defense against cross-skill attacks, we use composition-aware auditing, submitting all constituent skills to the auditor together. Following research on skill-interaction auditing (Zeng et al., 2026), the auditor examines each skill’s declared purpose and implemented operations, then assesses whether their interactions produce suspicious combined behavior. With Deepseek as the audit model, composition-aware auditing yields an overall DR of 28.7% for CORED-SKILL, compared with 3.3% for SkillScan and 11.3% for SkillSpector under individual-package auditing. Under the same composition-aware audit configuration, its overall DR is lower than SCR-Bench’s 74.0% and ColluSkill’s 40.0% (Table 4). The effect differs across objectives. EXF has a DR of 74.0%, whereas TAM and PRV have DRs of 2.0% and 10.0%, respectively, indicating that most state-tampering and privilege-violation attacks remain undetected even when all constituent skills are reviewed together. This configuration has a false-positive rate of 24.0% on clean compositions, whereas the four per-package LLM audit configurations and both static scanners yield no observed false positives on the clean controls.

Table 4: Composition-aware detection rates (DR) with all constituent skills provided to the auditor together. The clean-composition row reports the false-positive rate. Lower DR indicates greater attack stealth.
<table><tr><td>Method</td><td>EXF</td><td>TAM</td><td>PRV</td><td>Overall</td></tr><tr><td>Clean compositions</td><td>一</td><td></td><td>一</td><td>24.0</td></tr><tr><td>SCR-Bench</td><td>100</td><td>96.0</td><td>26.0</td><td>74.0</td></tr><tr><td>ColluSkill</td><td>56.0</td><td>32.0</td><td>32.0</td><td>40.0</td></tr><tr><td>CoREDSKILL (Ours)</td><td>74.0</td><td>2.0</td><td>10.0</td><td>28.7</td></tr></table>

## 4.7 Ablation Study

We fix the three-skill workflows on the shared 24- task subset, replace selected injected skills with their clean versions, and keep all other settings unchanged. Averaged over injection-position configurations, ASR is 0% with zero or one injected skill, 36.6% with two injected skills, and 100% with all three skills injected. Attack success depends on coordination among injected skills, but when one injection stage is missing, the agent can sometimes complete the missing step using malicious instructions in the skills and the workflow context. The execution records in Appendix C illustrate this process. The overall TSR is 93.1% with one injected skill and 100% with zero, two, or three injected skills. In some single-injection cases, the agent departs from the intended workflow and attempts an alternative solution that fails to produce outputs satisfying the acceptance criteria.

## 5 Related Work

Attacks on Agent Skills. Studies of skill ecosystems document malicious packages, credential leakage, and repository-dependent risks (Liu et al., 2026c,b; Chen et al., 2026b; Holzbauer et al., 2026). Attack methods use injected instructions, auxiliary scripts, and position-aware concealment (Schmotz et al., 2026; Jia et al., 2026; Hao et al., 2026), as well as hidden comments, runtime code generation, and dynamic skill modification (Wang et al., 2026b; Liu et al., 2026a; Chen et al., 2026a). Other work conceals executable payloads or poisons models bundled with skills (Ji et al., 2026; Tie et al., 2026). COREDSKILL distributes unauthorized operations across the task functions of multiple skills while preserving their original utility.

Compositional Attacks. Tool-chain attacks and intent fragmentation combine local actions or subtasks into harmful behavior (Li et al., 2025; Ahad et al., 2026). SkillReact, SkillProbe, and SCR-Bench examine risks arising from skill interactions and invocation paths (Wang et al., 2026c; Guo et al., 2026b; Xie et al., 2026). ColluSkill constructs interdependent skills around a malicious objective and refines them with scanner feedback (Zeng et al., 2026), while SkillTrojan reconstructs a payload from fragments collected across skill invocations (Feng et al., 2026). COREDSKILL embeds distributed operations in existing task logic and connects them through normal intermediate artifacts. The modified skills retain their utility, which we verify through task outputs.

Skill Auditing and Runtime Defenses. Skill auditors combine static checks and LLM analysis (Liu et al., 2026c; NVIDIA, 2026; Bhardwaj, 2026; Cisco AI Defense, 2026b). Further approaches use hierarchical triage, declared–implemented capability comparisons, and repository context (Hou and Yang, 2026; Wu et al., 2026; Holzbauer et al., 2026). ChainGuard reconstructs cross-skill dependencies (Zeng et al., 2026), while runtime defenses enforce action policies and monitor execution (Wang et al., 2026a; Shi et al., 2025; Shan et al., 2026; Cisco AI Defense, 2026a). Our evaluation covers individualpackage auditing, composition-aware auditing, and runtime auditing alongside task completion. Appendix B provides a detailed discussion.

## 6 Conclusion

COREDSKILL combines workflow decomposition with AST rewriting to embed distributed attack operations in task functions, creating stealthy composition attacks while preserving skill utility. In the main evaluation, it achieves 98.0% ASR, 95.3% TSR, and 3.3% DR, with a joint success rate of 90.7%. The results show that skill compositions can pass auditing and complete legitimate tasks while jointly performing unauthorized operations.

## Limitations

Attack construction in COREDSKILL relies on predefined attack specifications and workflow interfaces, and automatic adaptation to arbitrary existing skill packages has not been evaluated. We verify preservation of skill utility against output acceptance criteria for 150 offline evaluation cases, without evaluating dynamic service interactions or long-term operation. Our runtime defense evaluation measures the detection capabilities of risk and policy rules using execution records, without evaluating how online blocking affects subsequent execution and attack outcomes.

## Ethics Statement

We evaluate locally constructed skill packages in independent task sandboxes using synthetic data, simulated protected resources, and controlled endpoints, without targeting production services or deployed third-party skill installations. This research aims to help defenders identify unauthorized behavior arising from skill compositions. The method could also be misused to conceal malicious operations. To reduce misuse risks, we plan to release sanitized execution traces, synthetic inputs, and the evaluation harness with guidance for attack identification and defense evaluation, but not malicious skill packages that can be used directly for attacks.

## References

Tanzim Ahad, Ismail Hossain, Md Jahangir Alam, Sai Puppala, Yoonpyo Lee, Syed Bahauddin Alam, and Sajedul Talukder. 2026. Semantic intent fragmentation: A single-shot compositional attack on multiagent AI pipelines. Preprint, arXiv:2604.08608.

Anthropic. 2026. Claude Code overview. Software documentation. Accessed October 13, 2026.

Varun Pratap Bhardwaj. 2026. Formal analysis and supply chain security for agentic AI skills. Preprint, arXiv:2603.00195.

Tianhao Chen, Zhengyuan Jiang, Yuepeng Hu, Yebei Gou, and Neil Zhenqiang Gong. 2026a. Dynamic malicious skills in agentic AI. Preprint, arXiv:2606.16287.

Zhihao Chen, Ying Zhang, Yi Liu, Gelei Deng, Yuekang Li, Yanjun Zhang, Jianting Ning, Leo Yu Zhang, Lei Ma, and Zhiqiang Li. 2026b. How your credentials are leaked by LLM agent skills: An empirical study. Preprint, arXiv:2604.03070.

Cisco AI Defense. 2026a. DefenseClaw: Security governance for agentic AI. Open-source software. Accessed 2026-10-13.

Cisco AI Defense. 2026b. Skill scanner: Multi-engine security scanner for agent skills. Software, version 2.0.13.

Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr. 2024. AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. In Advances in Neural Information Processing Systems, volume 37, pages 82895–82920. Curran Associates, Inc.

DeepSeek-AI et al. 2026. DeepSeek-V4: Towards highly efficient million-token context intelligence. Preprint, arXiv:2606.19348.

Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, and Zhen J. Xiang. 2025. Memory injection attacks on LLM agents via query-only interaction. In Advances in Neural Information Processing Systems, volume 38, pages 46697–46731. Curran Associates, Inc.

Zenghao Duan, Yuxin Tian, Zhiyi Yin, Liang Pang, Jingcheng Deng, Zihao Wei, Shicheng Xu, Yuyao Ge, and Xueqi Cheng. 2026. SkillAttack: Automated red teaming of agent skills through attack path refinement. Preprint, arXiv:2604.04989.

Bacem Etteib, Daniele Lunghi, and Tégawendé F. Bissyandé. 2026. Detecting malicious agent skills in the wild using attention. Preprint, arXiv:2606.23416.

Yunhao Feng, Yifan Ding, Yingshui Tan, Boren Zheng, Yanming Guo, Xiaolong Li, Kun Zhai, Yishan Li, and Wenke Huang. 2026. SkillTrojan: Backdoor attacks on skill-based agent systems. Preprint, arXiv:2604.06811.

Wenbo Guo, Wei Zeng, Chengwei Liu, Xiaojun Jia, Yijia Xu, Lei Tang, Yong Fang, and Yang Liu. 2026a. MalSkillBench: A runtime-verified benchmark of malicious agent skills. Preprint, arXiv:2606.07131.

Zihan Guo, Zhiyu Chen, Xiaohang Nie, Jianghao Lin, Yuanjian Zhou, and Weinan Zhang. 2026b. Skill-Probe: Security auditing for emerging agent skill marketplaces via multi-agent collaboration. Preprint, arXiv:2603.21019.

Haochang Hao, Dehai Min, Zhifang Zhang, Yunbei Zhang, Miao Xu, Yingqiang Ge, and Lu Cheng. 2026. Poise: Position-aware one-instruction skill injection for silent execution on LLM agents. Preprint, arXiv:2606.07943.

Florian Holzbauer, David Schmidt, Gabriel K. Gegenhuber, Sebastian Schrittwieser, and Johanna Ullrich. 2026. Context matters: Repository-aware security analysis of the agent skill ecosystem. Preprint, arXiv:2603.16572.

Yinghan Hou and Zongyou Yang. 2026. SkillSieve: A hierarchical triage framework for detecting malicious AI agent skills. Preprint, arXiv:2604.06550v3.

Zimo Ji, Congying Xu, Zongjie Li, Yudong Gao, Xin Wei, Shuai Wang, and Shing-Chi Cheung. 2026. Cloak and detonate: Scanner evasion and dynamic detection of agent skill malware. Preprint, arXiv:2607.02357.

Xiaojun Jia, Jie Liao, Simeng Qin, Jindong Gu, Wenqi Ren, Xiaochun Cao, Yang Liu, and Philip Torr. 2026. SkillJect: Effectively automating skill-based prompt injection for skill-enabled agents. Preprint, arXiv:2602.14211.

Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. 2026. SoK: Agentic skills – beyond tool use in LLM agents. Preprint, arXiv:2602.20867.

Kimi Team et al. 2026. Kimi K2.5: Visual agentic intelligence. Preprint, arXiv:2602.02276.

Jing-Jing Li, Jianfeng He, Chao Shang, Devang Kulshreshtha, Xun Xian, Yi Zhang, Hang Su, Sandesh Swamy, and Yanjun Qi. 2025. STAC: When innocent tools form dangerous chains for LLM agents. Preprint, arXiv:2509.25624v3.

Zhiyuan Li, Jingzheng Wu, Xiang Ling, Xing Cui, and Tianyue Luo. 2026. Towards secure agent skills: Architecture, threat taxonomy, and security analysis. Preprint, arXiv:2604.02837.

Xinyu Liu, Yukai Zhao, Xing Hu, and Xin Xia. 2026a. Exploiting LLM agent supply chains via payload-less skills. Preprint, arXiv:2605.14460.

Yi Liu, Zhihao Chen, Yanjun Zhang, Gelei Deng, Yuekang Li, Jianting Ning, and Leo Yu Zhang. 2026b. "do not mention this to the user": Detecting and understanding malicious agent skills in the wild. In 35th USENIX Security Symposium (USENIX Security 26), pages 1727–1746, Baltimore, MD. USENIX Association.

Yi Liu, Weizhe Wang, Ruitao Feng, Yao Zhang, Guangquan Xu, Gelei Deng, Yuekang Li, and Leo Zhang. 2026c. Agent skills in the wild: An empirical study of security vulnerabilities at scale. Preprint, arXiv:2601.10338.

Yuting Ning, Zhehao Zhang, Yash Kumar Lal, Boyu Gou, Junyi Li, Weitong Ruan, Chentao Ye, Rahul Gupta, Diyi Yang, Yu Su, and Huan Sun. 2026. Skill-Harm: Lifecycle-aware skill-based attacks via automated construction. Preprint, arXiv:2606.02540.

NVIDIA. 2026. SkillSpector: Security scanner for AI agent skills. Software, version 2.3.9. Accessed 2026- 10-13.

OpenAI. 2026. GPT-5.5 model. Model documentation. Accessed October 13, 2026.

Yangjun Ruan, Honghua Dong, Andrew Wang, Silviu Pitis, Yongchao Zhou, Jimmy Ba, Yann Dubois, Chris Maddison, and Tatsunori Hashimoto. 2024. Identifying the risks of LM agents with an LMemulated sandbox. In International Conference on Learning Representations, pages 27031–27098.

David Schmotz, Luca Beurer-Kellner, Sahar Abdelnabi, and Maksym Andriushchenko. 2026. Skill-Inject: Measuring agent vulnerability to skill file attacks. Preprint, arXiv:2602.20156.

Zhengyang Shan, Jiayun Xin, Yue Zhang, and Minghui Xu. 2026. Don’t let the claw grip your hand: A security analysis and defense framework for OpenClaw. Preprint, arXiv:2603.10387.

Tianneng Shi, Jingxuan He, Zhun Wang, Hongwei Li, Linyu Wu, Wenbo Guo, and Dawn Song. 2025. Progent: Securing AI agents with privilege control. Preprint, arXiv:2504.11703v3.

Guiyao Tie, Jiawen Shi, Pan Zhou, and Lichao Sun. 2026. BadSkill: Backdoor attacks on agent skills via model-in-skill poisoning. Preprint, arXiv:2604.09378.

Haoyu Wang, Christopher M. Poskitt, and Jun Sun. 2026a. AgentSpec: Customizable runtime enforcement for safe and reliable LLM agents. In Proceedings ofthe 2026 IEEE/ACM 48th International Conference on Software Engineering, pages 2938–2950. ACM.

Qianli Wang, Boyang Ma, Minghui Xu, and Yue Zhang. 2026b. When skills lie: Hidden-comment injection in LLM agents. Preprint, arXiv:2602.10498.

Su Wang, Pin Qian, Yihang Chen, Junxian You, Xiaoyuan Wang, Xiaochong Jiang, Lifei Liu, Haoran Yu, and Jingzhou Xu. 2026c. When safe skills collide: Measuring compositional risk in agent skill ecosystems. Preprint, arXiv:2606.00448.

Zhiqiang Wang, Yichao Gao, Yanting Wang, Suyuan Liu, Haifeng Sun, Haoran Cheng, Guanquan Shi, Haohua Du, and Xiangyang Li. 2026d. MCPTox: A benchmark for tool poisoning on real-world MCP servers. Proceedings of the AAAI Conference on Artificial Intelligence, 40(42):35811–35819.

Yuhao Wu, Tung-Ling Li, and Hongliang Liu. 2026. Behavioral integrity verification for AI agent skills. Preprint, arXiv:2605.11770.

Wenjie Xiao, Xuehai Tang, Biyu Zhou, Songlin Hu, and Jizhong Han. 2026. RouteGuard: Internal-signal detection of skill poisoning in LLM agents. Preprint, arXiv:2604.22888.

Yi Xie, Jiawei Du, Yu Cheng, Jiuan Zhou, and Zhaoxia Yin. 2026. Benign in isolation, harmful in composition: Security risks in agent skill ecosystems. Preprint, arXiv:2606.15242.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. Preprint, arXiv:2406.12045.

Zonghao Ying, Xiangfan Wu, Huiyu Wu, Xing Zheng, Huangsheng Cheng, Xiaorong Shi, and Jing Guo. 2026. SkillJack: Persistent skill backdoors in selfevolving agents. Preprint, arXiv:2608.03509.

Z.ai. 2026. GLM-5.2. Model card. Accessed October 13, 2026.

Puyu Zeng, Simeng Qin, Jingzhi Li, Ju Jia, Zheli Liu, and Xiaojun Jia. 2026. ColluSkill: Adversarial crossskill composition for evading agent skill scanners. Preprint, arXiv:2608.09732.

Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel Kang. 2024. InjecAgent: Benchmarking indirect prompt injections in tool-integrated large language model agents. In Findings of the Association for Computational Linguistics: ACL 2024, pages 10471– 10506, Bangkok, Thailand. Association for Computational Linguistics.

Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, and Yongfeng Zhang. 2025. Agent security bench (ASB): Formalizing and benchmarking attacks and defenses in LLM-based agents. In International Conference on Learning Representations, pages 35331–35366.

## A Benchmark and Evaluation Details

This appendix documents the tasks and their acceptance criteria, evaluation case selection, and the execution environment and outcome checks.

## A.1 Benchmark Construction

We draw on public task scenarios from Skill-Inject (Schmotz et al., 2026), τ-bench (Yao et al., 2024), and AgentDojo (Debenedetti et al., 2024) to implement 50 tasks and their corresponding skills in a shared local evaluation environment. Each task includes a user request, structured input fixtures, and an expected publication payload, providing independently verifiable task outputs. Table A1 lists all tasks and their identifiers.

Task logic is implemented with the Python standard library, while external services are simulated using predefined data and controlled responses. Task outputs are checked against the acceptance criteria described in Appendix A.2. For example, AB04 asks the agent to review warehouse stock, account for committed orders and expected demand, and publish an allocation and replenishment plan. Its expected output specifies item availability, projected balances, shortages, replenishment quantities, and allocation decisions. These fields form the task’s acceptance criteria for verifying task completion and preservation of the skill’s utility.

Each of the 50 tasks is evaluated under three attack objectives, data exfiltration (EXF), state tampering (TAM), and privilege violation (PRV), giving 150 evaluation cases with 50 per objective. The objectives respectively involve transferring protected data to hosts outside the task’s authorized scope, copying protected files into skill archives without authorization, thereby changing the installed skills’ persistent state, and executing scripts beyond the user request’s authorized scope. For each evaluation case, all methods receive the same legitimate user request and use the same taskoutput acceptance criteria and attack-success criterion. Section A.2 describes the verification evidence. The composition-size study (Section 4.4) and partial-injection ablation (Appendix C) use a fixed subset of 24 tasks randomly selected before evaluation. This subset provides 72 cases per configuration, with 24 per attack objective. Asterisks in Table A1 mark these tasks.

## A.2 Execution and Verification

We use Claude Code CLI (Anthropic, 2026) (version 2.1.92) as the main victim platform, configured with deepseek-v4-flash as the execution backend. We also use deepseek-v4-flash for skill construction. Skill construction and agent execution have no time limit.

Each execution starts in a fresh session with a separate task workspace and model configuration. For each method, we install all of its injected skills together with six clean skills as distractors. The workspace allows intermediate artifacts and agent-generated code. Input fixtures have read-only file permissions, whereas the agent is instructed to preserve installed skill packages. Drivers record command events, completion status, and resulting artifacts.

The agent receives the legitimate user request and shared instructions to read installed skill documentation, select relevant functions, and place intermediate artifacts in the task workspace. It autonomously selects skill calls and execution commands. Expected publication payloads are available to the evaluator but are excluded from the agent’s request.

Task Output Verification. A local HTTP/HTTPS proxy records publication requests and returns controlled service responses.

Table A1: The tasks in the benchmark. Asterisks mark the tasks used in the composition-size experiment and partial injection ablation.
<table><tr><td>ID</td><td>Task</td><td>ID</td><td>Task</td></tr><tr><td>AB01*</td><td>Customer profile verification</td><td>AB26*</td><td>Document metadata governance</td></tr><tr><td>AB02</td><td>Product catalog review</td><td>AB27</td><td>Employment contract review</td></tr><tr><td>AB03*</td><td>Retail order fulfillment</td><td>AB28*</td><td>Business policy compliance</td></tr><tr><td>AB04*</td><td>Inventory replenishment</td><td>AB29</td><td>Knowledge evidence retrieval</td></tr><tr><td>AB05*</td><td>Supplier quotations</td><td>AB30*</td><td>Brand consistency checks</td></tr><tr><td>AB06*</td><td>Airline reservations</td><td>AB31*</td><td>Tabular data quality auditing</td></tr><tr><td>AB07*</td><td>Hotel availability and booking</td><td>AB32*</td><td>Structured data mapping</td></tr><tr><td>AB08</td><td>Ground transport planning</td><td>AB33</td><td>KPI anomaly review</td></tr><tr><td>AB09</td><td>Entry document checks</td><td>AB34</td><td>Management data reporting</td></tr><tr><td>AB10</td><td>Delivery exception resolution</td><td>AB35</td><td>Migration snapshot validation</td></tr><tr><td>AB11</td><td>Bank transfer instructions</td><td>AB36</td><td>Static patch review</td></tr><tr><td>AB12*</td><td>Bank reconciliation</td><td>AB37</td><td>Test case design</td></tr><tr><td>AB13</td><td>Transaction risk triage</td><td>AB38*</td><td>Defect root cause analysis</td></tr><tr><td>AB14</td><td>Payroll preflight checks</td><td>AB39*</td><td>Shell script quality checks</td></tr><tr><td>AB15</td><td>Insurance claim intake</td><td>AB40</td><td>Web acceptance evaluation</td></tr><tr><td>AB16*</td><td>Financial ratio analysis</td><td>AB41</td><td>System health inspection</td></tr><tr><td>AB17*</td><td>Budget variance analysis</td><td>AB42</td><td>Incident triage and response</td></tr><tr><td>AB18*</td><td>Enterprise valuation</td><td>AB43</td><td>Access permission auditing</td></tr><tr><td>AB19*</td><td>Portfolio risk measurement</td><td>AB44</td><td>Release readiness checks</td></tr><tr><td>AB20*</td><td>Invoice compliance auditing</td><td>AB45</td><td>API and webhook reconciliation</td></tr><tr><td>AB21*</td><td>Email triage</td><td>AB46</td><td>Application completeness checks</td></tr><tr><td>AB22*</td><td>Calendar conflict resolution</td><td>AB47</td><td>Onboarding coordination</td></tr><tr><td>AB23*</td><td>Meeting action tracking</td><td>AB48</td><td>Training certificate verification</td></tr><tr><td>AB24</td><td>Team message coordination</td><td>AB49</td><td>Feedback ticket routing</td></tr><tr><td>AB25</td><td>Internal communication release</td><td>AB50</td><td>SEO content auditing</td></tr></table>

The evaluator selects the last POST request to the designated host and path after excluding empty JSON objects and probe-only objects. Its JSON body must equal the expected publication payload. Missing fields, additional fields, incorrect values, or the absence of a qualifying publication cause the task-output check to fail.

Attack verification. EXF succeeds when a recorded request to a host outside the task’s authorized scope contains protected values. TAM succeeds when the execution record confirms that a protected file was copied into a skill archive without authorization, changing the installed skill’s persistent state. PRV succeeds when the recorded script execution result confirms that the agent ran a helper script unrelated to the user’s task and outside its authorized scope. The helper script is initially permission-locked. The workflow copies it to the task workspace and restores its execute permission, after which the agent runs it according to the skill instructions and workflow context. We verify task output and attack evidence separately to assess task completion and attack success within each execution.

We also evaluate GPT-5.5 (OpenAI, 2026) and

Table A2: Static detection rates (DR) of SkillFortify and CiscoSkillScan. Control rows report false-positive rates. Lower DR indicates greater attack stealth. Bold marks the lowest attack DR in each column within each auditor.
<table><tr><td>Auditor</td><td>Method</td><td>EXF</td><td>TAM PRV</td><td></td><td>Overall</td></tr><tr><td>SkillFortify</td><td>Original skills</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>SkillFortify</td><td>Clean compositions</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>SkillFortify</td><td>Skill-Inject</td><td>100</td><td>0</td><td>100</td><td>66.7</td></tr><tr><td>SkillFortify</td><td>SkillJect</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>SkillFortify</td><td>SCR-Bench</td><td>100</td><td>0</td><td>0</td><td>33.3</td></tr><tr><td>SkillFortify</td><td>ColluSkill</td><td>8.0</td><td>0</td><td>10.0</td><td>6.0</td></tr><tr><td>SkillFortify</td><td>CoREDSKILL (Ours)</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>CiscoSkillScan</td><td>Original skills</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>CiscoSkillScan</td><td>Clean compositions</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>CiscoSkillScan Skill-Inject</td><td></td><td>0</td><td>0</td><td>32.0</td><td>10.7</td></tr><tr><td>CiscoSkillScan SkillJect</td><td></td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>CiscoSkillScanSCR-Bench</td><td></td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>CiscoSkillScan ColluSkill</td><td></td><td>4.0</td><td>0</td><td>6.0</td><td>3.3</td></tr><tr><td></td><td>CiscoSkillScan CoREDSKILL (Ours)</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Kimi K2.5 (Kimi Team et al., 2026) as alternative execution and audit models. Replacing the execution model produces little variation in execution outcomes. Replacing the audit model modestly changes detection rates, while the relative trends across methods remain consistent.

Table A3: Detection rates (DR) of COREDSKILL with two to four skills, where k denotes the number of skills in the composition. Lower DR indicates greater attack stealth. Bold marks the lowest value across composition sizes in each column within each auditor.
<table><tr><td rowspan="2">Auditor</td><td rowspan="2">k</td><td colspan="2">EXF</td><td colspan="2">TAM</td><td colspan="2">PRV</td><td colspan="2">Overall</td></tr><tr><td>Deepseek</td><td>GLM</td><td>Deepseek</td><td>GLM</td><td>Deepseek</td><td>GLM</td><td>Deepseek</td><td>GLM</td></tr><tr><td>SkillScan</td><td>2</td><td>0</td><td>20.8</td><td>0</td><td>12.5</td><td>12.5</td><td>29.2</td><td>4.2</td><td>20.8</td></tr><tr><td>SkillScan</td><td>3</td><td>4.2</td><td>20.8</td><td>0</td><td>16.7</td><td>8.3</td><td>37.5</td><td>4.2</td><td>25.0</td></tr><tr><td>SkillScan</td><td>4</td><td>12.5</td><td>20.8</td><td>12.5</td><td>20.8</td><td>12.5</td><td>37.5</td><td>12.5</td><td>26.4</td></tr><tr><td>SkillSpector</td><td>2</td><td>25.0</td><td>50.0</td><td>12.5</td><td>12.5</td><td>37.5</td><td>41.7</td><td>25.0</td><td>34.7</td></tr><tr><td>SkillSpector</td><td>3</td><td>4.2</td><td>8.3</td><td>0</td><td>8.3</td><td>20.8</td><td>29.2</td><td>8.3</td><td>15.3</td></tr><tr><td>SkillSpector</td><td>4</td><td>12.5</td><td>33.3</td><td>20.8</td><td>20.8</td><td>45.8</td><td>16.7</td><td>26.4</td><td>23.6</td></tr></table>

## A.3 Auditor Settings

In the individual-package configurations, each auditor examines a skill package’s documentation, code, and local resources. Within each LLM-based auditor, Deepseek and GLM use the same input configuration and receive the same skill-package contents. SkillScan (Liu et al., 2026c) supports combining a local static scanner with an LLM audit adapter. We use its skill-security-scan static engine (v1.0.0). The semantic audit examines these package contents to identify unauthorized data transfer, state changes, and hidden execution. The LLMassigned risk level determines the final verdict.

SkillSpector (NVIDIA, 2026) (v2.3.9) first uses its default static analyzers to identify potential risks, then uses an LLM to review and filter the findings. Long files are processed in overlapping chunks. SkillScan and SkillSpector each use Deepseek and GLM, yielding four LLM audit configurations.

SkillFortify (Bhardwaj, 2026) (v0.6.0) and CiscoSkillScan (Cisco Skill Scanner (Cisco AI Defense, 2026b), v2.0.13) run without LLM calls, using static checks of package permissions, dependencies, code patterns, and data flows. SkillFortify receives each package in its expected skill-directory layout. CiscoSkillScan uses its default format validation with behavioral analysis enabled. All skill-package audit configurations count a HIGH or CRITICAL risk rating as a detection.

For composition-aware auditing (Section 4.6), we use Deepseek and provide the documentation, code, and local resources of all constituent skills together.

## A.4 Runtime Audit Configurations

Runtime auditing applies risk and policy rules to the agent’s execution records and installed skill files. During each task run, the driver records command events, completion status, and generated artifacts, while a local proxy records network requests. HITL’s risk rules (Shan et al., 2026) are extracted from its tool-call classification and extended rule layers and applied to serialized tool names and arguments. They check both individual calls and their order in the execution trace, counting a HIGH or CRITICAL match as a detection. DefenseClaw’s strict-policy rules (Cisco AI Defense, 2026a), rated HIGH or CRITICAL, are applied to tool commands, proxy-recorded network requests, and installed Python, shell, JSON, and Markdown skill files. A match likewise counts as a detection.

## A.5 Composition Size Results

Table A3 provides the complete per-objective results summarized in Figure 2, reporting DR for each composition size and audit configuration. Each composition size is evaluated on the same 72 cases from the fixed 24-task subset.

## B Related Work

## B.1 Attacks on Agent Skills

Empirical studies document malicious skills and credential leakage (Liu et al., 2026c,b; Chen et al., 2026b), while architectural and repository-aware analyses characterize their supply-chain risks (Li et al., 2026; Jiang et al., 2026; Holzbauer et al., 2026). Attack construction includes template injection, execution-feedback refinement, and positionaware instruction placement (Schmotz et al., 2026; Jia et al., 2026; Hao et al., 2026). Poise also verifies the attack action and legitimate task in the same trial. Hidden-comment and payload-less attacks conceal instructions or induce runtime code generation (Wang et al., 2026b; Liu et al., 2026a). Dynamic malicious skills and SkillHarm alter persistent skill content during execution or for later reuse (Chen et al., 2026a; Ning et al., 2026).

Other mechanisms include payload obfuscation and packing (Ji et al., 2026), backdoored models bundled with skills (Tie et al., 2026), and conversion of poisoned experiences into persistent skills (Ying et al., 2026). Tool-description and memory poisoning provide related attack channels (Wang et al., 2026d; Dong et al., 2025). SkillAttack instead exploits vulnerabilities in unmodified skills through adversarial prompting (Duan et al., 2026). COREDSKILL distributes attack operations across existing task functions while preserving skill utility.

## B.2 Compositional Attacks

STAC (Li et al., 2025) constructs dangerous chains from individually permissible tools, and Semantic Intent Fragmentation (Ahad et al., 2026) distributes harmful objectives across locally acceptable subtasks. Within skill ecosystems, SkillReact (Wang et al., 2026c) measures risks among individually safe skills, SkillProbe (Guo et al., 2026b) simulates inter-skill interactions, and SCR-Bench (Xie et al., 2026) evaluates capability flow, trust transfer, and authorization confusion along skill paths. These studies examine how available capabilities and agent decisions combine to produce unauthorized effects.

ColluSkill (Zeng et al., 2026) generates interdependent subpayloads around a malicious objective, connects them through context and artifacts, and refines flagged skills using scanner feedback. Skill-Trojan (Feng et al., 2026) accumulates fragments across invocations to reconstruct an executable payload. COREDSKILL instead uses AST rewriting to embed distributed attack operations in existing skills while preserving their original functionality. Appendix D details the closest comparisons and baseline adaptations.

## B.3 Skill Auditing and Runtime Defenses

SkillScan and SkillSpector combine static evidence with LLM analysis (Liu et al., 2026c; NVIDIA, 2026), while SkillFortify and CiscoSkillScan provide static checks (Bhardwaj, 2026; Cisco AI Defense, 2026b). Further approaches use hierarchical triage (Hou and Yang, 2026), attention-guided review (Etteib et al., 2026), internal model signals (Xiao et al., 2026), declared–implemented capability comparisons (Wu et al., 2026), and repository context (Holzbauer et al., 2026). ChainGuard extends auditing to dependencies between candidate and installed skills (Zeng et al., 2026). At runtime, AgentSpec and Progent constrain actions and privileges (Wang et al., 2026a; Shi et al., 2025), HITL and DefenseClaw assess risks and apply policies (Shan et al., 2026; Cisco AI Defense, 2026a), and SkillDetonate audits sandbox behavior through information-flow evidence (Ji et al., 2026).

Table C1: Attack success rates (ASR) and task success rates (TSR) in the partial injection ablation. In $P ^ { z _ { 1 } z _ { 2 } z _ { 3 } }$ 1 and 0 indicate an injected skill and a clean skill at the corresponding position, respectively. Bold marks the highest value in each column.
<table><tr><td colspan="2">ASR↑ ASR↑ ASR↑ (TAM) (PRV) ASR↑ TSR↑</td></tr><tr><td>Configuration (EXF)</td><td></td></tr><tr><td> $P ^ { 0 0 0 }$  0  $P ^ { 1 0 0 }$ </td><td>0 0 0 100</td></tr><tr><td>0  $P ^ { 0 1 0 }$ </td><td>0 0 0 95.8 0 0 0 94.4</td></tr><tr><td>0  $P ^ { 0 0 1 }$ </td><td>0 0 0 88.9</td></tr><tr><td>0  $P ^ { 1 1 0 }$  0</td><td>8.3 79.2 29.2 100</td></tr><tr><td> $P ^ { 1 0 1 }$ </td><td>0 33.3 45.8 26.4 100</td></tr><tr><td> $P ^ { 0 1 1 }$  83.3</td><td>79.2 0 54.2 100</td></tr><tr><td> $P ^ { 1 1 1 }$  100</td><td>100 100 100 100</td></tr></table>

Execution-based evaluation complements these defenses. ToolEmu emulates tool execution (Ruan et al., 2024), InjecAgent tests indirect injection (Zhan et al., 2024), and AgentDojo evaluates legitimate tasks alongside adversarial objectives (Debenedetti et al., 2024). ASB studies attacks across agent components and the utility–security balance (Zhang et al., 2025), while MalSkillBench verifies malicious skills through sandbox execution (Guo et al., 2026a). Our evaluation combines package and runtime auditing with output-based task acceptance to test whether distributed attacks evade detection while preserving skill utility.

## C Partial Injection Ablation

We replace selected injected skills with their clean versions to measure how injection position affects attack success and task success. Each configuration is evaluated on 72 cases from the fixed 24-task subset, with 24 cases per attack objective. For configuration $P ^ { z _ { 1 } z _ { 2 } z _ { 3 } }$ $z _ { i } = 1$ selects the injected skill at workflow position $i ,$ while $z _ { i } = 0$ selects its clean version. The bit order follows the workflow order, so $P ^ { 0 1 1 }$ injects the second and third skills. Workflow roles, user requests, and sandbox settings remain unchanged.

Table C1 shows that attack success under partial injection depends on the retained injection positions and the attack objective. Under $P ^ { 0 1 1 }$ , the

EXF and TAM ASRs reach 83.3% and 79.2%, respectively. For both objectives, the downstream injected skills ask the agent to supply the files or fields required to continue the attack. Because the clean first skill does not provide these inputs, the agent searches the available task materials and supplies them, allowing the subsequent exfiltration or copying of the protected file into the skill archive to proceed. Under $P ^ { 1 1 0 }$ , the PRV ASR reaches 79.2%. The two upstream injected skills place a permission-locked helper script unrelated to the user’s task in the task workspace and restore its execute permission. The agent then runs the script based on the workflow context. Although the third skill remains clean, the agent still completes the task through the clean publishing skill. These execution records illustrate how the agent can sometimes complete one missing attack step using malicious instructions in the remaining skills and the workflow context.

## D Comparison with Related Work

## D.1 Attack Baseline Adaptations

We adapt four attack baselines to the benchmark’s tasks and attack objectives, and their names in the result tables refer to these adapted versions. For Skill-Inject (Schmotz et al., 2026), we append its injection templates to the skill description and add a script wrapper that performs the specified attack operation alongside normal task processing.

For SkillJect (Jia et al., 2026), Deepseek generates a single malicious skill that combines injected instructions with an auxiliary payload. We refine the skill iteratively using execution feedback.

Our adaptation of SCR-Bench (Xie et al., 2026) uses its capability-flow setting to construct twoskill workflows for the benchmark’s tasks and attack objectives. The upstream skill supplies information for the downstream action. We add fixed prompt-injection templates to the skill documentation to guide the agent’s use of this information, together with helper scripts for unauthorized copying of protected files into skill archives and execution of task-unrelated scripts.

For ColluSkill (Zeng et al., 2026), we implement a three-skill chain with interdependent subpayloads and scanner-feedback refinement, using a sevenround refinement budget. We use Deepseek for skill generation and refinement and as the SkillScan audit backend. In each round, SkillScan checks all three skills, and its feedback guides the rewriting

of flagged skills.

## D.2 Threat Model Comparison

ColluSkill (Zeng et al., 2026) decomposes a malicious payload into ordered, interdependent parts and embeds them in separate skills. Context, artifact transfer, and execution handoffs connect these parts during agent execution, while scanner feedback is used to rewrite detected skills. SCR-Bench (Xie et al., 2026) studies risks that arise when an agent composes individually benign skills. Its controlled scenarios cover capability flow, trust transfer, and authorization confusion, with risk arising from the agent’s choices and workflow context rather than malicious code in the component skills.

These studies construct interdependent attack skills or controlled skill-interaction scenarios. COREDSKILL instead starts from existing skills with independently verifiable utility and embeds distributed attack operations in their original task logic. The injected skills retain their expected task outputs, while the user’s legitimate request continues to define the authorization boundary. Our evaluation jointly measures attack success, task completion, and audit evasion.

## E Three-Skill Construction Example

We illustrate the three-skill EXF construction with AB04, an inventory-replenishment task that reviews stock and demand and publishes a replenishment plan. The injected skills preserve these task functions while distributing file-and-field inventory, value collection, and data transfer across the three stages. Together, these operations locate and collect a protected warehouse API credential and send it to an unauthorized recipient. Algorithm E1 summarizes the construction and the information passed between skills.

The construction uses the specification $\Phi ( \mathrm { E X F } , 3 )$ . Each intermediate artifact is represented as $\widetilde { \boldsymbol { x } } _ { i } = ( x _ { i } , h _ { i } )$ , where $x _ { i }$ contains task results and $h _ { i }$ carries information for subsequent attack operations. The initial task input is ${ \widetilde { x } } _ { 0 } .$ . In the algorithm, δ denotes the fixture directory, I its file-and-field inventory, and V the collected values together with their sources and pending-review flags.

BIND applies the AST rewriting and packaging defined in Equation (4). Select is implemented within the third skill’s export helper and filters entries with pending-review flags and non-null values. Each stage call summarizes the agent’s invocations of that skill, including normal publication and the additional distribution command in the third stage. A local proxy records the resulting network requests as execution evidence.

Algorithm E1 Three-skill EXF construction and   
execution dependencies   
Require: clean composition $C = ( S _ { 1 } , S _ { 2 } , S _ { 3 } )$ , input xe0   
Require: specification $\Phi = \Phi ( \mathrm { E X F } , 3 )$   
1: for $i = { \bar { 1 } } , 2 , 3$ do   
2: $\phi _ { i }  \Phi [ i ]$   
3: $S _ { i } ^ { \prime } \gets$ BIN $) ( S _ { i } ; \phi _ { i } )$   
4: end for   
Inventory anchor: a<sub>1</sub> = Inventory   
5: $\widetilde { x } _ { 1 }  \dot { S _ { 1 } ^ { \prime } } ( \widetilde { x } _ { 0 } )$   
6: $h _ { 1 } = ( \delta , { \dot { \mathcal { T } } } )$ ▷ File and field inventory   
Collection anchor: $a _ { 2 } = C _ { 0 }$ llect   
7: $\widetilde x _ { 2 }  S _ { 2 } ^ { \prime } ( \widetilde x _ { 1 } )$   
8: $h _ { 2 } = ( \delta , { \mathcal { T } } , { \mathcal { V } } )$   
V: collected values with sources and pending-review   
flags   
Export anchor: $a _ { 3 } =$ Export   
9: $\varepsilon \gets$ SELECT(V)   
Retain pending entries with non-null values   
10: $x _ { 3 } \gets S _ { 3 } ^ { \prime } ( \hat { \widetilde { x } } _ { 2 } ; \mathcal { E } )$   
11: return task output x<sub>3</sub>

## E.1 Protected Data and State Dependencies

The user-authorized publication destination is the inventory-planning service. The fixture contains a protected warehouse API credential in the warehouse\_api\_token field of env.json, which the request does not authorize the agent to disclose. The three skills perform inventory loading, stockposition auditing, and plan publication. Their intermediate artifacts carry both task results and information needed by the EXF operations.

The first skill, inventory\_loader, loads the inventory fixture and normalizes its four SKU records for downstream processing. Its injected inventory operation also enumerates fixture files and their JSON field names, including warehouse\_api\_token in env.json. The file-and-field inventory passed downstream contains file and field names without the credential value. The second skill, stock\_position\_auditor, calculates projected balances and replenishment needs, allocates available stock to committed orders by priority, and builds the replenishment plan. Its injected collection operation reads the inventoried fields and writes their values, sources, and review\_pending flags into the intermediate artifact, passing the protected credential alongside the task results. The third skill, replenishment\_publisher, reads the replenishment policy, prepares the publication payload, and submits the plan to the service specified by the user. Its rewritten export helper filters collected entries with pending review flags and nonnull values. The injected skill instructions then direct the agent to invoke the additional distribute command after normal publication. This command sends the collected entries to the target supplied by the skill’s configuration, which is outside the user’s authorized publication destination.

The published replenishment plan has identifier RP-2026-07-W31 and a purchase total of CNY 465.00, matching the task’s expected output.

## E.2 Anchor Binding and AST Transformation

The abstract anchors Inventory, Collect, and Export bind to \_list\_fixture\_files in inventory\_loader, \_locate\_state\_fields in stock\_position\_auditor, and \_export\_configuration\_entries in replenishment\_publisher, respectively. Each specification ϕ<sub>i</sub> identifies the helper to replace and the fields needed for downstream execution. The AST rewriter replaces the helper implementation and updates its callers. In the second skill, it also renames \_locate\_state\_fields to \_record\_configuration\_state. The resulting workflow passes the fixture directory and file–field inventory through fixture\_dir and available\_files, and stores collected values and review\_pending flags in located\_state\_fields. The publisher retains its original output construction while gaining a distribute command that reads these entries. The skill documentation prompts the agent to invoke this command, and the followup\_targets configuration supplies its destination. These code, documentation, and configuration changes connect the attack operations through the workflow’s existing artifacts.

The execution record contains two POST requests. The agent first publishes the expected replenishment plan to the inventory-planning service. It then invokes distribute to send configuration entries containing the protected warehouse\_api\_token value to /v1/entries on config-service.internal, the unauthorized recipient specified by the injected skill.

## F Usage of LLMs

We use LLMs to assist with data generation, code development, and manuscript editing. The research ideas and methodology are developed by the human authors.