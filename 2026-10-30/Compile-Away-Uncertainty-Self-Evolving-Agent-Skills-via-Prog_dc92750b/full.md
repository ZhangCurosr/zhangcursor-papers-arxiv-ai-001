# Compile Away Uncertainty: Self-Evolving Agent Skills via Programmatic Compilation

Chengyu Yang<sup>1</sup>, Yunxiang Zheng<sup>1</sup>, Zongzheng Li<sup>1</sup>, Jiamei Li<sup>1</sup>, Boxun Lan<sup>1</sup>, Xiaoyu Liu<sup>2</sup>,

Xiaoyu Zhang<sup>1</sup>, Xiaohong Chen<sup>1</sup>, Ming Hu<sup>1,\*</sup>

<sup>1</sup>East China Normal University, <sup>2</sup>China University of Geoscience Corresponding author

Abstract—Self-evolving agent skills enable AI agents to accumu late and refine reusable procedural knowledge from interaction experience without updating model parameters. However, since such skills are typically represented as textual documents and executed through LLM reasoning, even previously learned procedures must be repeatedly interpreted and regenerated, introducing avoidable uncertainty into every execution. In contrast, program execution is deterministic and independently verifiable. Intuitively, transforming well-specified operations into executable programs can reduce reliance on LLM reasoning and, consequently, reduce execution uncertainty. Motivated by this intuition, we introduce SKILLCOMPILER, a self-evolving skill framework that integrates programmatic compilation into the skill evolution process. Specifically, SKILLCOMPILER adopts an executability-aware skill compilation strategy that decomposes skills into fine-grained operations, analyzes their executability, and constructs hybrid skill skeletons that retain LLM reasoning for context-dependent operations while compiling well-specified ones into validated programs. To avoid redundant program synthesis and enable executable knowledge to accumulate across evolving skills, we further design a program asset reuse mechanism that organizes validated programs as reusable assets, retrieves existing implementations whenever applicable, and synthesizes and verifies new programs only when necessary. Extensive experiments on five benchmarks and six model families demonstrate that SkillCompiler consistently outperforms state-of-the-art methods, achieving up to 32.33% improvement in task performance.

Index Terms—self-evolving agents, reusable skills, programmatic compilation, execution reliability, tool use

## I. INTRODUCTION

Large language model (LLM) agents have demonstrated strong capabilities in reasoning, planning, tool use, and environment interaction, enabling them to solve increasingly complex tasks, including information retrieval, Web interaction, and interactive problem solving [1–5]. However, agents often need to rediscover similar task-solving procedures when facing new tasks, even when related experience has already been acquired. Agent skills provide a natural way to preserve such experience as reusable procedural knowledge, allowing agents to reuse previously acquired task-solving procedures without updating the underlying language model [6–10]. Consequently, effectively accumulating and reusing skills has become an important direction for improving agent capabilities through interaction experience.

To continuously accumulate and refine reusable skills from interaction experience, existing work has explored different strategies for skill evolution. Existing approaches can be broadly classified into two categories, i.e., skill content evolution and skill execution augmentation. Specifically, skill content evolution improves reusable procedural knowledge through trajectory distillation [9], reflection-based revision [12], and validationguided skill optimization [10]. Skill execution augmentation enhances how acquired skills are applied by reusing learned workflows [16, 17], incorporating external tool capabilities [5], or constructing executable procedural representations [19–21]. These methods enable agents to progressively improve both the knowledge encoded in skills and the way such knowledge is reused during task execution.

Although existing skill evolution methods can continuously accumulate and refine reusable task-solving knowledge, they still face two challenges in reliable and efficient skill reuse, i.e., execution uncertainty and repeated capability construction. Specifically, ❶ executing an evolved skill still requires the agent to repeatedly interpret retained knowledge, reconstruct intermediate decisions, and generate corresponding actions through LLM reasoning. As a result, even with an unchanged skill, repeated executions may follow different reasoning paths and produce different outcomes. ❷ Reusable execution capabil ities discovered during skill evolution are often constructed for individual skills or tasks and are not persistently organized for subsequent reuse, leading to repeated generation and validation of similar procedures across evolving skills. Therefore, how to reduce execution uncertainty while efficiently accumulating and reusing executable knowledge across skill evolution remains an important problem.

Typically, LLM reasoning is inherently probabilistic, which makes the execution of the same skill susceptible to different intermediate decisions and outcomes across repeated runs. In contrast, executable programs encode their inputs, control flow, and state transformations explicitly, so the same operation follows a fixed execution procedure under the same conditions rather than regenerating intermediate decisions at each invocation. Intuitively, when an operation has explicit inputs, well-defined execution logic, and checkable outputs, its execution no longer needs to be repeatedly regenerated through LLM reasoning. Instead, such operations can be realized as executable programs, while LLM reasoning is retained for context-dependent or difficult-to-formalize decisions. Moreover, once these programs are validated, they can be persistently accumulated and reused across evolving skills, avoiding repeated synthesis of the same executable procedures.

Inspired by the above intuition, we propose SKILLCOM-PILER, a self-evolving skill framework that transforms suitable reasoning operations into deterministic executable programs as skills evolve. Specifically, ① SKILLCOMPILER adopts an executability-aware skill compilation strategy to identify which skill operations can be reliably transformed into executable programs while retaining LLM reasoning for context-dependent decisions. ② To avoid redundant program synthesis and enable executable knowledge to accumulate across evolving skills, SKILLCOMPILER integrates a program asset reuse mechanism that organizes validated programs as reusable assets, retrieves compatible implementations whenever available, and synthesizes and verifies new programs only when necessary. In this way, SKILLCOMPILER reduces unnecessary LLM rea soning while preserving the flexibility required for unresolved operations. The main contributions are summarized as follows.

• We propose SKILLCOMPILER, a self-evolving skill framework that progressively transforms suitable skill operations into deterministic executable programs. Its executabilityaware compilation strategy identifies which operations can be reliably programmatized while retaining LLM reasoning for context-dependent decisions.

• We design a program asset reuse mechanism that persistently accumulates, retrieves, and reuses validated executable programs across evolving skills, reducing redundant program synthesis and enabling executable knowledge to continuously grow through interaction.

• We conduct extensive experiments on five benchmarks and six model families. The results show that SKILLCOM-PILER outperforms state-of-the-art methods, achieving up to 32.33% improvement in task performance.

## II. RELATED WORK

## A. Experience Accumulation and Reusable Agent Skills

AI agents can improve subsequent task solving by retaining useful experience from previous interactions without updating model parameters. Reflexion [6] stores verbal feedback from previous attempts to guide future decisions, while ExpeL [7] extracts generalizable insights from collections of task experiences. Agent Workflow Memory [17] induces recurring workflows from trajectories and retrieves them to guide subsequent action generation. Voyager [14] further accumulates executable skills from successful environment interactions and refines them through execution feedback.

## B. Agent Skills Evolution

Existing work can be classified into skill content evolution and skill execution augmentation. Skill content evolution methods improve the quality and transferability of reusable procedural knowledge by refining skill representations from accumulated interaction experience. For example, Trace2Skill [9] distills reusable lessons from trajectories. EmbodiSkill [12] revises skills based on execution evidence. SkillOpt [10] conducts validation-guided optimization over natural-language skill documents. Skill execution augmentation methods enhance how acquired skills are carried out by introducing additional execution structures or capabilities. For example, Agent Workflow Memory [17] reuses learned workflows to guide subsequent action generation. FlowEvo [16] reuses verified workflows as executable skills. AutoTools [5] augments agent execution with external tools. NSI [20], SkillDisCo [21], and Agent Skill Induction [19] construct more directly executable procedural representations from interaction experience.

Different from existing skill evolution methods, SKILLCOM-PILER does not merely attach reusable workflows, tools, or executable artifacts to skill execution. Instead, it explicitly analyzes operation-level executability within an evolving skill and selectively determines which operations should remain under LLM reasoning and which should be transformed into validated programs. To the best of our knowledge, SKILLCOMPILER is the first self-evolving skill framework that jointly enables operation-level selective compilation and persistent reuse of the resulting program assets across skill evolution.

## III. OUR SKILLCOMPILER APPROACH

## A. Overview

Fig. 1 presents the overall framework of SKILLCOMPILER, which consists of five major components, including the task analyzer, programmer, executor, skill optimizer, and program asset manager. Task analyzer decomposes a task into finegrained operations, analyzes their executability, and constructs a hybrid skill skeleton. Programmer realizes programmable operations by retrieving existing implementations or synthesizing new programs and validates them before integration. Executor applies the resulting candidate skill by combining programmatic execution with decisions from the target agent. Skill optimizer analyzes successful and failed execution traces to refine the evolving skill. Program asset manager maintains a program asset bank to support program retrieval, updating, and reuse across skill evolution.

As shown in Fig. 1, the workflow of SKILLCOMPILER mainly consists of three phases, i.e., executability-aware skill compilation, program asset reuse, and skill optimization. The details of each phase are as follows:

• Phase 1: Executability-aware skill compilation. Given a task, the Task Analyzer decomposes it into fine-grained operations, analyzes their executability, and organizes both programmatic and agent-guided operations into a hybrid skill skeleton. For operations selected for programmatic execution, the Programmer obtains validated implementations and incorporates them into the skeleton to form a candidate skill, while the remaining operations continue to rely on LLM reasoning.

• Phase 2: Program asset reuse. Before synthesizing a new implementation, the program asset manager retrieves compatible programs from the Program Asset Bank and reuses them whenever applicable. Newly synthesized and validated programs are persisted as reusable assets for subsequent skills.

• Phase 3: Skill optimization. The executor applies the candidate skill in the target environment and records the resulting execution traces and task feedback. If the skill successfully completes the task, the execution result is returned and the validated skill knowledge is retained for subsequent reuse. Otherwise, the Skill Optimizer analyzes the failed execution trace to identify problematic operations, guidance, or dependencies and generates an updated skill skeleton. Operations affected by the revision are returned to the preceding phases for executability analysis, program retrieval or synthesis, and validation before the updated skill is executed again.

![](images/258760a8d0e6a7c02a9e6dd65202cc0a9f310768e32de15c7dce10a36313119b.jpg)  
Fig. 1. Framework and Overview of SKILLCOMPILER

## B. Executability-Aware Skill Compilation

A skill may contain operations with different execution requirements. Some operations depend on semantic understanding, contextual judgment, or changing observations, while others follow explicit and repeatable procedures. Executing all operations through LLM reasoning therefore introduces unnec essary execution uncertainty, whereas compiling the entire skill into a program may sacrifice the flexibility required for contextdependent decisions. To address this issue, SKILLCOMPILER analyzes executability at the operation level and selectively transforms suitable operations into executable programs.

1) Operation Decomposition.: To identify the appropriate unit for selective compilation, the task analyzer first decomposes the task and relevant skill knowledge into a set of finegrained operations. Each operation represents a locally coherent capability that can be independently invoked and evaluated. We represent an operation $\kappa _ { i }$ using a 6-tuple operation contract as follows:

$$
\kappa _ { i } = \langle I _ { i } , O _ { i } , \mathrm { p r e } _ { i } , \mathrm { p o s t } _ { i } , \mathrm { c h e c k } _ { i } , g _ { i } \rangle ,\tag{1}
$$

where $I _ { i }$ and $O _ { i }$ denote the required inputs and expected outputs, pre specifies the observable preconditions, post describes the expected local effects, check defines how these effects can be verified, and $g _ { i }$ contains reusable execution guidance. Task-specific entities and values are instantiated only when the operation is invoked, allowing the same operation to be reused across different tasks.

2) Operation Analysis.: After operation decomposition, the task analyzer determines whether each operation can be reliably realized through programmatic execution. An operation is considered programmatic when its inputs and outputs can be explicitly represented, its execution procedure can be clearly specified, its preconditions can be observed from the current environment, and its resulting outputs or local effects can be independently verified. In addition, the available tool interfaces should support a bounded implementation of the operation. Operations satisfying these requirements are selected for programmatic execution, while operations involving unresolved inputs, context-dependent decisions, or open-ended semantic reasoning remain agent-guided.

This analysis determines the execution mode of each operation before constructing the skill. In particular, programmatic operations no longer require the target agent to repeatedly generate their internal execution procedures, while agentguided operations retain LLM reasoning to accommodate dynamic observations and contextual decisions. In this way, SKILLCOMPILER selectively moves well-specified operations from probabilistic reasoning to deterministic execution without restricting the flexibility of the entire skill.

3) Skill Skeleton Construction.: Based on the executability analysis, the task analyzer organizes the resulting operations into a hybrid skill skeleton. For a task x, the instantiated skill skeleton is represented as

$$
S _ { x } = \langle \mathcal { O } _ { x } , \sigma _ { x } , \mathcal { E } _ { x } , \beta _ { x } \rangle ,\tag{2}
$$

where ${ \mathcal { O } } _ { x }$ contains the operation occurrences required by the task, $\sigma _ { x }$ specifies their execution order, ${ \mathcal E } _ { x }$ records the data and prerequisite relations among operations, and $\beta _ { x }$ contains the current argument bindings. Data relations connect the outputs of preceding operations to the inputs of subsequent operations, while prerequisite relations specify the local effects that must be established before an operation can be executed.

Both programmatic and agent-guided operations are organized in the same skill skeleton. During construction, SKILLCOMPILERchecks the compatibility between operation interfaces and identifies the sources of required inputs. Task specific values that are unavailable during construction remain unresolved in $\beta _ { x }$ and are instantiated according to actual observations during execution. Therefore, the skill skeleton explicitly describes how deterministic programs and LLM reasoning collaborate to accomplish a task while preserving their respective execution requirements.

4) Programming and Validation.: For each operation selected for programmatic execution, SKILLCOMPILERneeds to obtain a qualified executable realization before incorporating it into the candidate skill. The programmer first attempts to obtain an existing compatible implementation through the program asset manager. When no qualified implementation is available, it synthesizes a new program according to the operation contract, permitted tool interfaces, and available execution evidence.

The synthesized program is validated in an isolated environment before being used for task execution. Specifically, SKILLCOMPILERchecks whether the program can be successfully executed, whether its inputs and outputs conform to $I _ { i }$ and $O _ { i } ,$ and whether the resulting local effects satisfy post according to check . When validation fails, the observed errors and check discrepancies are returned to the Programmer to guide program repair. This synthesis, validation, and repair process continues until a qualified implementation is obtained or the operation remains agent-guided. Only validated programs are incorporated into the skill skeleton to form the candidate skill, ensuring that LLM reasoning is replaced only when the corresponding program can be independently verified.

## C. Program Asset Reuse

Although programmatic execution reduces repeated LLM reasoning, repeatedly synthesizing and validating similar programs can introduce additional overhead during skill evolution. In practice, different tasks may contain operations with the same underlying capability, making their executable implementations reusable across skills. To avoid redundant program synthesis, SKILLCOMPILER maintains validated programs in the Program Asset Bank and preferentially reuses existing implementations whenever compatible operations are encountered.

1) Compatibility-Based Program Retrieval.: For each operation selected for programmatic execution, the program asset manager first retrieves candidate programs according to the semantic description and execution requirements of the operation. Lexical similarity is used to identify potentially reusable implementations, while their applicability is further determined according to the operation contract. Specifically, compatible Skill Skeletons are ranked by lexical similarity, while atomic skills are ranked using lexical Dice similarity, historical utility, and activation status:

$$
R ( a , o ) = \frac { 2 | \mathcal { T } ( o ) \cap \mathcal { T } ( a ) | } { | \mathcal { T } ( o ) | + | \mathcal { T } ( a ) | } + 0 . 1 u ( a ) + 0 . 0 5 \mathbf { 1 } [ \mathrm { a c t i v e } ( a ) ] ,\tag{3}
$$

where $\tau ( \cdot )$ denotes lexical term sets, $u ( a )$ is the historical utility of atomic skill $^ { a , }$ and 1[active(a)] indicates its active status. Candidates are first filtered by capability and contract compatibility before ranking.

Program realizations are located through explicit Skill-Implementation-Program references. A retrieved program can be reused only when its required inputs can be correctly bound, its preconditions are satisfied, the current environment provides the required interfaces, and its existing validation record remains applicable to the current operation. Formally, the eligible programs are

$$
\begin{array} { r l } & { \mathcal { P } _ { \mathrm { v a l i d } } ( a , x ) = \{ p \in \mathcal { P } ( a ) : \mathrm { B i n d } ( p , x ) \wedge \mathrm { P r e } ( p , x ) } \\ & { \qquad \wedge \mathrm { E n v } ( p , x ) \wedge \mathrm { C e r t } ( p , x ) \} , } \end{array}\tag{4}
$$

where ${ \mathcal { P } } ( a )$ contains the program realizations referenced by skill $^ { a , }$ and the predicates check argument binding, preconditions, environment compatibility, and validation applicability, respectively. Therefore, lexical similarity supports candidate discovery, while interface and state compatibility determine whether a retrieved program can be safely invoked.

When multiple implementations correspond to the same capability, SKILLCOMPILER retains their respective applicability conditions and validation evidence. The program satisfying the current operation contract and execution environment is selected for reuse. Automatic invocation requires a unique eligible or preferred implementation; otherwise, selection remains agent guided. If no compatible implementation is available, the operation is passed to the Programmer for program synthesis and validation as described in Section III-B.

## D. Skill Optimization

After constructing the candidate skill, SKILLCOMPILER executes it in the target environment to evaluate whether the current skill can successfully accomplish the task. During execution, the executor invokes validated programs for programmatic operations and delegates the remaining operations to the target agent. The resulting actions, observations, argument bindings, program invocations, local checking results, and final task feedback are recorded as execution traces for subsequent skill updating.

1) Trace-Based Skill Revision.: When task execution fails, the skill optimizer locates the failure by comparing the actual execution of each operation with its operation contract. If an operation produces the expected output and satisfies check , the operation is retained even when the final task fails. Otherwise, SKILLCOMPILER revises the corresponding component according to the observed failure. An unsatisfied or overly broad invocation condition leads to an update of pre . Incorrect information passed between operations leads to changes in argument bindings or dependencies in the skill skeleton. Incomplete execution guidance results in an update of $g _ { i }$ . If a program fails to produce the expected effect specified by post , the corresponding program is sent back for repair and validation. Therefore, skill revision is performed locally according to operation-level execution evidence rather than regenerating the entire skill from the failed task.

2) Skill Merging and Updating.: The revised operations are further compared with existing operations in the skill bank to reduce redundant skill knowledge. SKILLCOMPILER first determines whether two operations describe the same local capability and then checks whether their operation contracts are compatible. Operations with consistent input and output semantics, preconditions, expected effects, and checking requirements are merged into a shared reusable operation, while task-specific entities and values are abstracted through parameterization and argument bindings. Operations with different capability semantics or conflicting contracts remain separate.

Formally, for two operation contracts $\kappa _ { i }$ and $\kappa _ { j }$ , the merging criterion is expressed as

$$
\begin{array} { r l } & { \mathrm { M e r g e a b l e } ( \kappa _ { i } , \kappa _ { j } ) = \mathrm { S a m e C a p } ( \kappa _ { i } , \kappa _ { j } ) } \\ & { \qquad \wedge \ \exists \pi : \bigwedge _ { f \in \{ I , O , \mathrm { p r e } , \mathrm { p o s t } , \mathrm { c h e c k } \} } \mathrm { C o m p a t } \big ( \pi ( f _ { i } ) , f _ { j } \big ) . } \end{array}\tag{5}
$$

Here, SameCap denotes shared capability semantics, π aligns task-specific parameters, and Compat requires semantically consistent contract fields under the alignment. Thus, merging is permitted only when capability identity and contract compatibility are both satisfied.

The resulting operations are reorganized into an updated skill skeleton. If the update only changes agent-guided operations, the revised skeleton can be directly used for the next execution. If the update changes a programmatic operation, its implementation is retrieved again or regenerated when necessary and subsequently validated before reuse. The candidate skill is then executed in the target environment. A successful execution confirms the current skill update with corresponding skill and program resources retained, while another failure produces a new execution trace for the next round of local revision.

## IV. EXPERIMENTS

In this section, we first introduce the experimental setup and evaluate the performance of SKILLCOMPILER against representative baselines. We then investigate its execution reliability and resource consumption through repeated-execution experiments and cost analysis. Finally, ablation studies are conducted to assess the contributions of the key components.

## A. Experimental Setup

Benchmarks and Models. We conduct experiments on five benchmarks, including SearchQA [28], OfficeQA [29], DocVQA [30], LiveMath [31], and ALFWorld [32]. Specifically, SearchQA [28] evaluates open-domain question answering, while OfficeQA [29] and DocVQA [30] focus on document understanding. LiveMath [31] assesses mathematical reasoning, and ALFWorld [32] evaluates long-horizon interactive task solving. Table I summarizes the training, validation, and test partitions of these benchmarks. Furthermore, we consider six large language models, including MiniMax-M3, Gemini-3.8- Flash, Qwen3.8-Flash, DeepSeek-V4.1-Flash, Kimi-K2.6, and GLM-5.3-Flash.

TABLE I  
DATASET PARTITION STATISTICS FOR THE FIVE EVALUATION BENCHMARKS.
<table><tr><td>Benchmark</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>SearchQA [28]</td><td>300</td><td>24</td><td>1,400</td></tr><tr><td>OfficeQA [29]</td><td>120</td><td>24</td><td>102</td></tr><tr><td>DocVQA [30]</td><td>180</td><td>22</td><td>201</td></tr><tr><td>LiveMath [31]</td><td>60</td><td>17</td><td>100</td></tr><tr><td>ALFWorld [32]</td><td>120</td><td>24</td><td>134</td></tr></table>

Baselines. We compare our proposed method SKILLCOM-PILER with seven baselines, including No Skill, Human Skill, Trace2Skill [9], EvoSkill [13], EmbodiSkill [12], GEPA [11], and SkillOpt [10].

• No Skill. The agent performs tasks without any skill guidance, relying solely on the underlying LLM.

• Human Skill. The agent executes tasks using fixed humanwritten skills provided by SkillOpt without evolution.

• Trace2Skill [9]. It distills reusable knowledge from execution trajectories and consolidates trajectory-level lessons into transferable skills.

• EvoSkill [13]. It identifies capability gaps through failure analysis and iteratively generates or refines skills based on validation feedback.

• EmbodiSkill [12]. It employs skill-aware reflection to distinguish skill deficiencies from execution failures, enabling targeted skill revisions.

• GEPA [11]. It optimizes prompts through natural-language reflection on execution feedback and Pareto-based candidate selection.

• SKillOpt [10]. It iteratively optimizes natural-language skills through bounded textual revisions, accepting updates only when validation performance improves.

Evaluation Protocol. All methods use the same task partitions, available tools, benchmark evaluators, and inference settings within each model–benchmark configuration. Main performance and cost results are averaged over three independently seeds (42, 43, and 44). Persistent skill states are selected before testing and remain frozen throughout the test evaluation. For repeated-execution and ablation analyses, we use the seed-42 skill states and execute each test task three times under the same inputs and environment conditions.

Metrics. Task performance is evaluated using exact match (EM) where applicable and benchmark-specific scores otherwise, including success rate (SR) for ALFWorld. Execution reliability is assessed through outcome instability and execution-path consistency, while resource consumption is measured by model token usage and the number of model calls.

## B. Performance Comparison

We evaluate the effectiveness of SKILLCOMPILER across different benchmarks, large language model families, and longhorizon interaction tasks.

1) Performance on different benchmarks: Table II presents the performance comparison across six language model families in terms of EM on four benchmarks. Taking Kimi-K2.6 as an example, SKILLCOMPILER achieves scores of 85.50%, 74.51%, 96.52%, and 76.00% on SearchQA, OfficeQA, DocVQA, and LiveMath, respectively. Compared with the strongest baseline on each benchmark, these results correspond to improvements of 1.52%, 16.99%, 0.50%, and 22.00%, respectively. In particular, the most significant improvements are observed on LiveMath and OfficeQA, indicating that SKILLCOMPILER is effective on mathematical reasoning and document-oriented tasks. In contrast, the relatively modest gains on SearchQA and DocVQA suggest that the benefits of programmatic compilation vary across task characteristics. A similar trend is observed across the remaining model families, with the largest performance gains generally occurring on LiveMath. In particular, SKILLCOMPILER improves the performance of GLM-5.3-Flash on LiveMath by 32.33%, further highlighting its effectiveness on mathematical reasoning tasks. These results suggest that SKILLCOMPILER is advantageous for tasks involving well-specified operations that can be compiled into validated executable programs.

TABLE II  
PERFORMANCE COMPARISON OF SKILLCOMPILER AND BASELINE METHODS ACROSS FOUR BENCHMARKS USING SIX MODELS.
<table><tr><td>Model</td><td>Method</td><td>SearchQA</td><td>OfficeQA</td><td>DocVQA</td><td>LiveMath</td><td>Avg</td></tr><tr><td rowspan="7">MiniMax-M3</td><td>No Skill</td><td>76.21</td><td>52.94</td><td>72.64</td><td>32.00</td><td>58.45</td></tr><tr><td>Human Skill</td><td>76.17</td><td>53.92</td><td>83.91</td><td>33.67</td><td>61.92</td></tr><tr><td>Trace2Skill [9]</td><td>79.43</td><td>60.78</td><td>91.38</td><td>42.00</td><td>68.40</td></tr><tr><td>EvoSkill [13]</td><td>79.24</td><td>62.09</td><td>89.88</td><td>51.33</td><td>70.64</td></tr><tr><td>EmbodiSkill [12]</td><td>77.64</td><td>61.76</td><td>79.77</td><td>34.67</td><td>63.46</td></tr><tr><td>GEPA [11]</td><td>79.57</td><td>59.48</td><td>90.55</td><td>48.67</td><td>69.57</td></tr><tr><td>SkillOpt [10] SKILLCOMPILER</td><td>81.88</td><td>60.46</td><td>89.72</td><td>54.67</td><td>71.68 79.78</td></tr><tr><td></td><td>82.86</td><td>63.72</td><td>94.53</td><td>78.00</td><td></td></tr><tr><td rowspan="7">Gemini-3.8-Flash</td><td>No Skill</td><td>82.83</td><td>50.98</td><td>92.70</td><td>33.00</td><td>64.88</td></tr><tr><td>Human Skill</td><td>83.19</td><td>51.63</td><td>94.86</td><td>33.33</td><td>65.75</td></tr><tr><td>Trace2Skill [9]</td><td>84.67</td><td>60.78</td><td>94.20</td><td>45.33</td><td>71.25</td></tr><tr><td>EvoSkill [13]</td><td>84.74</td><td>57.84</td><td>94.36</td><td>55.67</td><td>73.15</td></tr><tr><td>EmbodiSkill [12]</td><td>82.55</td><td>25.49</td><td>90.71</td><td>33.00</td><td>57.94</td></tr><tr><td>GEPA [11]</td><td>87.05</td><td>66.34</td><td>93.70</td><td>67.33</td><td>78.61</td></tr><tr><td>SkillOpt [10]</td><td>84.29</td><td>55.23</td><td>94.69</td><td>66.67</td><td>75.22</td></tr><tr><td>SKILLCOMPILER</td><td>89.00</td><td>75.49</td><td>96.02</td><td>78.00</td><td>84.63</td></tr><tr><td rowspan="8">Qwen3.8-Flash</td><td>No Skill</td><td>79.10</td><td>37.58</td><td>93.53</td><td>25.67</td><td>58.97</td></tr><tr><td>Human Skill</td><td>79.14</td><td>50.98</td><td>93.86</td><td>26.00</td><td></td></tr><tr><td>Trace2Skill [9]</td><td>82.90</td><td>72.88</td><td>93.70</td><td></td><td>62.50</td></tr><tr><td>EvoSkill [13]</td><td>82.74</td><td>70.59</td><td>93.53</td><td>31.67 51.67</td><td>70.29 74.63</td></tr><tr><td>EmbodiSkill [12]</td><td>83.07</td><td>67.65</td><td>94.69</td><td>14.67</td><td>65.02</td></tr><tr><td>GEPA [11]</td><td>81.21</td><td>68.63</td><td>93.37</td><td>39.00</td><td>70.55</td></tr><tr><td>SkillOpt [10]</td><td>80.21</td><td>68.30</td><td>92.70</td><td>35.00</td><td>69.05</td></tr><tr><td>SKILLCOMPILER</td><td>85.07</td><td>75.49</td><td>96.02</td><td>77.00</td><td>83.40</td></tr><tr><td rowspan="8">DeepSeek-V4.1-Flash</td><td>No Skill</td><td>76.10</td><td>33.33</td><td>74.96</td><td>28.00</td><td>53.10</td></tr><tr><td>Human Skill</td><td>76.21</td><td>41.18</td><td>82.75</td><td>27.33</td><td>56.86</td></tr><tr><td>Trace2Skill [9]</td><td>84.24</td><td>67.65</td><td>88.56</td><td>32.67</td><td>68.28</td></tr><tr><td>EvoSkill [13]</td><td>83.33</td><td>69.28</td><td>84.58</td><td>52.00</td><td>72.30</td></tr><tr><td>EmbodiSkill [12]</td><td>81.36</td><td>51.63</td><td>77.61</td><td>23.67</td><td>58.57</td></tr><tr><td>GEPA [11]</td><td>83.21</td><td>70.26</td><td>87.56</td><td>50.67</td><td>72.93</td></tr><tr><td>SkillOpt [10] SKILLCOMPILER</td><td>82.95</td><td>69.93</td><td>85.57</td><td>30.33</td><td>67.20</td></tr><tr><td></td><td>86.36</td><td>71.24</td><td>91.71</td><td>77.00</td><td>81.58</td></tr><tr><td rowspan="8">Kimi-K2.6</td><td>No Skill</td><td>80.79</td><td>32.03</td><td>94.67</td><td>39.67</td><td>61.79</td></tr><tr><td>Human Skill</td><td>81.05</td><td>32.03</td><td>95.02</td><td>39.33</td><td>61.86</td></tr><tr><td>Trace2Skill [9]</td><td>83.98</td><td>55.56</td><td>95.36</td><td>43.33</td><td>69.56</td></tr><tr><td>EvoSkill [13]</td><td>83.33</td><td>52.61</td><td>95.19</td><td>36.00</td><td>66.78</td></tr><tr><td>EmbodiSkill [12]</td><td>78.98</td><td>57.52</td><td>92.37</td><td>51.67</td><td>70.14</td></tr><tr><td>GEPA [11]</td><td>83.10</td><td>55.56</td><td>96.02</td><td>42.00</td><td>69.17</td></tr><tr><td>SkillOpt [10]</td><td>81.48</td><td>54.25</td><td>95.19</td><td>54.00</td><td>71.23</td></tr><tr><td>SKILLCOMPILER</td><td>85.50</td><td>74.51</td><td>96.52</td><td>76.00</td><td>83.13</td></tr><tr><td rowspan="8">GLM-5.3-Flash</td><td>No Skill</td><td>78.62</td><td>30.07</td><td>81.92</td><td>42.00</td><td>58.15</td></tr><tr><td>Human Skill</td><td>79.45</td><td>31.70</td><td>91.04</td><td>41.00</td><td>60.80</td></tr><tr><td>Trace2Skill [9]</td><td>83.90</td><td>65.69</td><td>94.36</td><td>44.67</td><td>72.16</td></tr><tr><td>EvoSkill [13]</td><td>77.38</td><td>59.80</td><td>90.38</td><td>40.67</td><td>67.06</td></tr><tr><td>EmbodiSkill [12]</td><td>81.02</td><td>59.15</td><td>93.20</td><td>40.00</td><td>68.34</td></tr><tr><td>GEPA [11]</td><td>83.50</td><td>64.05</td><td>93.03</td><td>43.00</td><td>70.90</td></tr><tr><td>SkillOpt [10]</td><td>83.55</td><td>64.38</td><td>93.37</td><td>41.67</td><td>70.74</td></tr><tr><td>SKILLCOMPILER</td><td>85.21</td><td>71.89</td><td>94.53</td><td>77.00</td><td>82.16</td></tr></table>

2) Performance with different model families: To investigate the applicability of SKILLCOMPILER across different foundation models, we evaluate its performance using six language model families. As shown in Table II, SKILLCOMPILER consistently achieves the highest average score across all six models, outperforming the strongest baseline for each model by 6.02– 11.90%. Specifically, the average scores of SKILLCOMPILER range from 79.78% to 84.63%, exhibiting a relatively narrow performance range of 4.85% across different models. The improvement is observed even though the relative performance of competing skill evolution methods changes across models. For example, the strongest overall baseline differs between Gemini-3.8-Flash and Qwen3.8-Flash, whereas SKILLCOM-PILER remains the top-performing method in both settings. This result suggests that the proposed skill representation and execution mechanism can be applied across heterogeneous model families without modifying their parameters. It does not imply identical performance across models; rather, the advantages over the corresponding baselines are consistently positive. These results indicate that SKILLCOMPILER maintains consistent performance advantages across different foundation models, demonstrating its broad applicability without modifying the underlying model parameters.

TABLE IIIPERFORMANCE COMPARISON ON ALFWORLD USINGDEEPSEEK-V4.1-FLASH.
<table><tr><td>Method</td><td>Success Rate (%)</td></tr><tr><td>No Skill</td><td>79.60</td></tr><tr><td>Human Skill</td><td>83.83</td></tr><tr><td>Trace2Skill</td><td>88.31</td></tr><tr><td>EvoSkill</td><td>85.82</td></tr><tr><td>EmbodiSkill</td><td>93.53</td></tr><tr><td>GEPA</td><td>86.82</td></tr><tr><td>SkillOpt</td><td>88.06</td></tr><tr><td>SKILLCOMPILER</td><td>98.26</td></tr></table>

TABLE IV

PERFORMANCE AND EXECUTION UNCERTAINTY COMPARISON ON LIVEMATH USING GEMINI-3.8-FLASH.
<table><tr><td>Method</td><td>EM↑</td><td>EIR@3↓</td></tr><tr><td>No Skill</td><td>28.33</td><td>19.00</td></tr><tr><td>Human Skill</td><td>28.33</td><td>13.00</td></tr><tr><td>Trace2Skill</td><td>40.67</td><td>28.00</td></tr><tr><td>EvoSkill</td><td>55.33</td><td>31.00</td></tr><tr><td>EmbodiSkill</td><td>27.67</td><td>18.00</td></tr><tr><td>GEPA</td><td>58.00</td><td>23.00</td></tr><tr><td>SkillOpt</td><td>64.67</td><td>32.00</td></tr><tr><td>SKILLCOMPILER</td><td>78.67</td><td>11.00</td></tr></table>

3) Performance on long-horizon interaction tasks: To further evaluate the effectiveness of SKILLCOMPILER in longhorizon interactive environments, we conduct experiments on ALFWorld using DeepSeek-V4.1-Flash. Unlike the other four benchmarks, ALFWorld requires agents to perform sequential actions and adapt to changing environment states, where intermediate decisions and execution errors may affect subsequent task progress. As shown in Table III, SKILLCOMPILER achieves the highest success rate of 98.26%, outperforming all competing methods. Specifically, SKILLCOMPILER surpasses the strongest baseline EmbodiSkill by 4.73%, while exceeding SkillOpt and GEPA by 10.20% and 11.44%, respectively. Compared with No Skill, SKILLCOMPILER improves the success rate from 79.60% to 98.26%, corresponding to an improvement of 18.66%. These results demonstrate the effectiveness of SKILLCOMPILER in long-horizon interactive tasks, indicating that its performance advantages extend beyond conventional reasoning and questionanswering tasks to complex long-horizon interaction task.

## C. Uncertainty Analysis

To investigate the execution uncertainty, we execute each task three times using fixed learned skills under identical task inputs, initial environment states, and inference settings. We adopt execution instability rate (EIR@3) to measure the proportion of tasks exhibiting both successful and failed executions. For ALFWorld, we further employ average path similarity (APS@3), success-conditioned path similarity $( \mathsf { A P S } _ { \mathrm { s u c c } } @ 3 )$ and exact path repetition (EPR@3) to evaluate executionpath consistency. Table IV presents the results on LiveMath using Gemini-3.8-Flash. SKILLCOMPILER achieves the highest EM of 78.67% and the lowest EIR of 11.00%. Compared with SkillOpt, which achieves the highest EM among the baselines, SKILLCOMPILER improves EM by 14.00% while reducing EIR from 32.00% to 11.00%, corresponding to a relative reduction of 65.6%. Similarly, SKILLCOMPILER achieves the lowest EIR of 3.0% on ALFWorld, outperforming SkillOpt by 4.5%, as shown in Fig. 2. Regarding execution-path consistency, SKILLCOMPILER achieves APS, $\mathrm { \mathbf { A P S _ { \mathrm { s u c c } } } }$ , and EPR scores of 81.3%, 82.5%, and 44.8%, respectively. These results demonstrate that SKILLCOMPILER effectively reduces execution uncertainty across different task environments while maintaining high task performance.

## D. Cost Analysis

To evaluate the resource efficiency of SKILLCOMPILER, we analyze the learning and inference costs in terms of token consumption and model calls. In particular, learning costs include tokens consumed during skill evolution, while inference costs are measured by the average token consumption and model calls per test task. The results are reported in Tables V and VI. Specifically, on LiveMath, SKILLCOMPILER consumes 2.695M learning tokens, reducing the learning cost by 41.9% and 52.1% compared with SkillOpt and GEPA, respectively. Moreover, SKILLCOMPILER requires only 0.450 model calls per task, the lowest among all competing methods. Although its inference token consumption of 13.984k per task exceeds that of SkillOpt, it remains lower than those of EvoSkill, EmbodiSkill, and GEPA. On ALFWorld, SKILLCOMPILER consumes 47.015k inference tokens per task, representing reductions of 7.6% and 32.7% compared with SkillOpt and GEPA, respectively. Meanwhile, its learning cost of 32.442M tokens is also lower than those of both methods. Compared with No Skill, SKILLCOMPILER reduces model calls by 25.7% while improving the task success rate by 18.66%. These results indicate that SKILLCOMPILER achieves substantial performance improvements without necessarily increasing inference costs. Furthermore, Table VII reports the internal execution and program reuse statistics of SKILLCOMPILER on ALFWorld. The PPC reaches 100.00%, while CPO is at least 48.36%. Meanwhile, CSR, PIS, and PAS achieve 54.85%, 88.84%, and 87.06%, respectively. In particular, the high PAS indicates that a substantial proportion of environment actions are performed within program execution spans. These observations provide additional evidence of extensive programmatic execution, complementing the resource efficiency analysis.

![](images/b4390b7dbdcc08c34502b1636a51e4c326aeed499acd0b8b943e395bc3cfd9a7.jpg)

![](images/0293a706e7fd571c33603a33db7c6c83975d87ed7c18b51dcb5ea983ced8fbb4.jpg)

![](images/f1cceaec2ffb00ce8bbf1d8cf45b83ed5f3a3562081512591074541e6a466616.jpg)

![](images/d185a0363dc077ef04bf59ba8b95c28d5d52505b61c808fd7e8c8c2cfe5924d7.jpg)  
Fig. 2. Execution reliability comparison on ALFWorld using DeepSeek-V4.1-Flash.

TABLE V  
RESOURCE CONSUMPTION COMPARISON ON LIVEMATH USING GEMINI-3.8-FLASH.
<table><tr><td>Method</td><td>Learning (M tokens)</td><td>Test/task (k tokens)</td><td>Calls/task</td></tr><tr><td>No Skill</td><td>0.000</td><td>5.429</td><td>1.007</td></tr><tr><td>Human Skill</td><td>0.000</td><td>5.997</td><td>1.000</td></tr><tr><td>Trace2Skill</td><td>1.117</td><td>8.918</td><td>1.003</td></tr><tr><td>EvoSkill</td><td>1.752</td><td>17.311</td><td>1.060</td></tr><tr><td>EmbodiSkill</td><td>2.988</td><td>19.335</td><td>3.047</td></tr><tr><td>GEPA</td><td>5.631</td><td>17.504</td><td>1.033</td></tr><tr><td>SkillOpt</td><td>4.639</td><td>6.672</td><td>1.007</td></tr><tr><td>SKILLCOMPILER</td><td>2.695</td><td>13.984</td><td>0.450</td></tr></table>

TABLE VI

RESOURCE CONSUMPTION COMPARISON ON ALFWORLD USINGDEEPSEEK-V4.1-FLASH.
<table><tr><td>Method</td><td>Learning (M tokens)</td><td>Test/task (k tokens)</td><td>Calls/task</td></tr><tr><td>No Skill</td><td>0.000</td><td>47.906</td><td>20.020</td></tr><tr><td>Human Skill</td><td>0.000</td><td>27.885</td><td>16.813</td></tr><tr><td>Trace2Skill</td><td>3.212</td><td>68.418</td><td>14.085</td></tr><tr><td>EvoSkill</td><td>3.515</td><td>39.102</td><td>19.420</td></tr><tr><td>SkillOpt</td><td>57.468</td><td>50.906</td><td>14.674</td></tr><tr><td>EmbodiSkill</td><td>10.934</td><td>42.710</td><td>15.321</td></tr><tr><td>GEPA</td><td>52.970</td><td>69.836</td><td>16.279</td></tr><tr><td>SKILLCOMPILER</td><td>32.442</td><td>47.015</td><td>14.868</td></tr></table>

## E. Ablation Study

In this subsection, we assess the contributions of the key components of SKILLCOMPILER through ablation experiments on ALFWorld using DeepSeek-V4.1-Flash. Specifically, we construct three variants by individually removing the Skill Skeleton, Programmer, and Skill Optimization. As shown in Table VIII, the variant without the Skill Skeleton achieves SR of 95.52%, compared with 98.26% for the full method. Meanwhile, its token consumption increases by 2.75 times, and EIR rises from 2.99% to 8.21%, indicating the importance of explicit operation organization for efficient and reliable execution. The most significant performance degradation occurs when the Programmer is disabled, with SR dropping to 53.48%, token consumption increasing to 7.36 times that of the full method, and EIR reaching 19.40%. These results highlight the critical role of programmatic execution in improving task performance and reliability. The variant without Skill Optimization achieves an SR of 95.52%, while its token consumption increases to 2.48 times that of the full method. Furthermore, APS decreases from 81.27% to 61.95%, demonstrating the importance of continuous skill Optimization for execution efficiency and consistency.

TABLE VII  
PROGRAM EXECUTION AND REUSE STATISTICS OF SKILLCOMPILER ON ALFWORLD ACROSS THREE LEARNING SEEDS.
<table><tr><td>Metric</td><td>Definition</td><td>Mean (%)</td></tr><tr><td>PPC</td><td>Persistent Program Coverage (covered / total tasks)</td><td>100.00</td></tr><tr><td>CPO</td><td>Consumed Program Outputs (trace-confirmed lower bound)</td><td>≥ 48.36</td></tr><tr><td>CSR</td><td>Cross-Skill Reuse (multi-skill / used programs)</td><td>54.85</td></tr><tr><td>PIS</td><td>Program Invocation Success (successful / total calls)</td><td>88.84</td></tr><tr><td>PAS</td><td>Program Action Share (program / total actions)</td><td>87.06</td></tr></table>

TABLE VIII  
ABLATION STUDY ON ALFWORLD USING DEEPSEEK-V4.1-FLASH.
<table><tr><td>Condition</td><td>SR ↑</td><td>Test tokens (k/task)↓</td><td>EIR@3↓</td><td>APS@3 ↑</td></tr><tr><td>Full SKILLCOMPILER</td><td>98.26</td><td>41.954</td><td>2.99</td><td>81.27</td></tr><tr><td>w/o Skill Skeleton</td><td>95.52</td><td>115.424</td><td>8.21</td><td>67.49</td></tr><tr><td>w/o Programmer</td><td>53.48</td><td>308.887</td><td>19.40</td><td>54.51</td></tr><tr><td>w/o Skill Optimization</td><td>95.52</td><td>103.858</td><td>4.48</td><td>61.95</td></tr></table>

Overall, these results confirm the complementary contributions of the three components to SKILLCOMPILER’s effectiveness.

## F. Discussion

Empirical Implications. The performance gains vary across benchmarks, with larger improvements on LiveMath and OfficeQA than on SearchQA and DocVQA. This suggests that selective compilation is particularly beneficial when tasks contain recurring operations with explicit inputs, well-defined procedures, and verifiable effects. In contrast, tasks dominated by context-dependent interpretation offer fewer opportunities to replace LLM reasoning with executable procedures.

Limitations. SKILLCOMPILER relies on identifiable operation boundaries, observable preconditions, and independently check able effects. Consequently, ambiguous or context-dependent decisions remain under LLM control, meaning that compilation reduces rather than eliminates execution uncertainty. Moreover, local validation establishes correctness only under the tested inputs and environment states, without guaranteeing generalization to unseen conditions. The benefits of persistent program reuse also depend on contract compatibility and sufficient reuse to amortize synthesis and validation costs. Finally, our evaluation of SKILLCOMPILER focused primarily on controlled benchmark environments, and its effectiveness across a broader range of tasks and in dynamic web environments remains to be investigated.

## V. CONCLUSION

SKILLCOMPILER integrates operation-level compilation into skill evolution. Its executability-aware strategy combines validated programs and context-dependent reasoning within hybrid Skill Skeletons, while persistent program assets support reuse and trace-driven Optimization. Experiments across five benchmarks and six model families show higher mean task performance. Fixed-bank repetitions on LiveMath and ALF-World additionally show fewer outcome reversals. Resource savings vary by baseline, while ALFWorld ablations identify the contributions of programmatic execution, explicit operation dependencies, and Optimization.

## REFERENCES

[1] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations. https://arxiv.org/abs/2210.03629

[2] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2Web: Towards a Generalist Agent for the Web. In Advances in Neural Information Processing Systems, Vol. 36. https://arxiv.org/abs/2306.06070

[3] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations. https://arxiv.org/abs/2307.13854

[4] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 6864–6890. https://doi.org/10.18653/v1/2024.acl-long.371

[5] Zhengliang Shi, Shen Gao, Lingyong Yan, Yue Feng, Xiuyi Chen, Zhumin Chen, Dawei Yin, Suzan Verberne, and Zhaochun Ren. 2025. Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents. In Proceedings of the ACM Web Conference 2025. https: //doi.org/10.1145/3696410.3714825

[6] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems, Vol. 36. https://papers.neurips.cc/paper files/paper/2023/hash/ 1b44b878bb782e6954cd888628510e90-Abstract-Conference.html

[7] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. ExpeL: LLM Agents Are Experiential Learners. Proceedings of the AAAI Conference on Artificial Intelligence 38, 17 (2024), 19632–19642. https://doi.org/10.1609/aaai.v38i17.29936

[8] Boxun Lan, Jiamei Li, Weijian Li, Chengyu Yang, Zhuang Qi, Chengcheng Wan, Xiaohong Chen, and Ming Hu. 2026. FedSkillGit: Federated Skill Evolution via Version Control. arXiv:2610.21348. https: //arxiv.org/abs/2610.21348

[9] Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. 2026. Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills. arXiv:2603.25158. https://arxiv.org/abs/2603.25158

[10] Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. 2026. SkillOpt: Executive Strategy for Self-Evolving Agent Skills. arXiv:2605.23904. https://arxiv.org/abs/2605.23904

[11] Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alex Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. 2026. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. In International Conference on Learning Representations. https://proceedings.iclr.cc/paper files/paper/2026/hash/ 0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html

[12] Ruofei Ju, Xinrui Wang, Xin Ding, Yifan Yang, Hao Wu, Shiqi Jiang, Qianxi Zhang, Hao Wen, Xiangyu Li, Weijun Wang, Kun Li, Yunxin Liu, Haipeng Dai, Wei Wang, and Ting Cao. 2026. EmbodiSkill: Skill

Aware Reflection for Self-Evolving Embodied Agents. arXiv:2605.10332. https://arxiv.org/abs/2605.10332

[13] Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. 2026. EvoSkill: Automated Skill Discovery for Multi-Agent Systems. arXiv:2603.02766, version 1. https://arxiv.org/abs/2603.02766v1

[14] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2024. Voyager: An Open-Ended Embodied Agent with Large Language Models. Transactions on Machine Learning Research. https://openreview.net/forum?id= Mn3Lzp1sMQ

[15] Simon Yu, Gang Li, Weiyan Shi, and Peng Qi. 2026. PolySkill: Learning Generalizable Skills Through Polymorphic Abstraction for Continual Learning. In International Conference on Learning Representations. https://proceedings.iclr.cc/paper files/paper/2026/hash/ e350897ed832f9cad6f2e223e7acad6d-Abstract-Conference.html

[16] Zeyu Ren, Ling Yue, Ran Li, Yishu Wang, Shengxiang Xu, Hanmo Liu, Shaowu Pan, and Shimin Di. 2026. FlowEvo: Self-Evolving Agents through the Co-Evolution of Workflows and Executable Skills. In Third Conference on Language Modeling. https://arxiv.org/abs/2607.21596

[17] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2025. Agent Workflow Memory. In Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, 63897–63911. https: //proceedings.mlr.press/v267/wang25bx.html

[18] Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. 2025. SkillWeaver: Web Agents Can Self-Improve by Discovering and Honing Skills. arXiv:2504.07079. https: //arxiv.org/abs/2504.07079

[19] Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, and Daniel Fried. 2025. Inducing Programmatic Skills for Agentic Tasks. In Second Conference on Language Modeling. https://openreview.net/forum?id= lsAY6fWsog

[20] Jie-Jing Shao, Haiyan Yin, Yueming Lyu, Xingrui Yu, Lan-Zhe Guo, Ivor Tsang, James Kwok, and Yu-Feng Li. 2026. Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks. In Proceedings of the 43rd International Conference on Machine Learning, PMLR 306, 110314–110337. https: //proceedings.mlr.press/v306/shao26i.html

[21] Zhongxin Guo, Danrui Qi, Hanwen Gu, Peng Cheng, and Yongqiang Xiong. 2026. SKILL-DISCO: Distilling and Compiling Agent Traces into Reusable Procedural Skills. arXiv:2606.26669. https://arxiv.org/abs/ 2606.26669

[22] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. PAL: Program-aided Language Models. In Proceedings of the 40th International Conference on Machine Learning. https://arxiv.org/abs/2211.10435

[23] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023. Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. Transactions on Machine Learning Research. https://arxiv.org/abs/2211.12588

[24] Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, and Amir Gholami. 2024. An LLM Compiler for Parallel Function Calling. In Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 24370–24391. https: //proceedings.mlr.press/v235/kim24y.html

[25] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan A, Saiful Haq, Ashutosh Sharma, Thomas Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. 2024. DSPy: Compiling Declarative Language Model Calls into State-ofthe-Art Pipelines. In International Conference on Learning Representations. https://proceedings.iclr.cc/paper files/paper/2024/ hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html

[26] Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, and Shumin Deng. 2026. SkillX: Automatically Constructing Skill Knowledge Bases for Agents. arXiv:2604.04804. https://arxiv.org/abs/2604.04804

[27] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. 2023. Code as Policies: Language Model Programs for Embodied Control. In IEEE International Conference on Robotics and Automation. https://arxiv.org/abs/2209.07753

[28] Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and Kyunghyun Cho. 2017. SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine. arXiv:1704.05179. https: //arxiv.org/abs/1704.05179

[29] Databricks AI Research Team. 2025. Introducing OfficeQA: A Benchmark for End-to-End Grounded Reasoning. Databricks Research, December 9, 2025. https://www.databricks.com/blog/ introducing-officeqa-benchmark-end-to-end-grounded-reasoning

[30] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. DocVQA: A Dataset for VQA on Document Images. In IEEE Winter Conference on Applications of Computer Vision, 2200–2209. https: //arxiv.org/abs/2007.00398

[31] Linyang He, Qiyao Yu, Hanze Dong, Baohao Liao, Xinxing Xu, Micah Goldblum, Jiang Bian, and Nima Mesgarani. 2026. LiveMathematician Bench: A Live Benchmark for Mathematician-Level Reasoning with Proof Sketches. arXiv:2604.01754, version 1. https://arxiv.org/abs/2604. 01754v1

[32] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam´ Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In International Conference on Learning Representations. https://arxiv.org/abs/2010.03768

[33] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. In Advances in Neural Information Process ing Systems, Vol. 36. https://papers.nips.cc/paper files/paper/2023/hash/ d842425e4bf79ba039352da0f658a906-Abstract-Conference.html

[34] Cheng Qian, Chi Han, Yi Fung, Yujia Qin, Zhiyuan Liu, and Heng Ji. 2023. CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models. In Findings of the Association for Computational Linguistics: EMNLP 2023, 6922–6939. https://doi.org/10.18653/v1/2023.findings-emnlp.462

[35] Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. 2024. Large Language Models as Tool Makers. In International Conference on Learning Representations. https://arxiv.org/abs/2305.17126

[36] Gabriel Grand, Lio Wong, Maddy Bowers, Theo X. Olausson, Muxin Liu, Joshua B. Tenenbaum, and Jacob Andreas. 2024. LILO: Learning Interpretable Libraries by Compressing and Documenting Code. In International Conference on Learning Representations. https://proceedings.iclr.cc/paper files/paper/2024/hash/ 819cebb05f993840e8a52d7564c5c282-Abstract-Conference.html

[37] Haiming Wang, Huajian Xin, Chuanyang Zheng, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, Jian Yin, Zhenguo Li, and Xiaodan Liang. 2024. LEGO-Prover: Neural Theorem Proving with Growing Libraries. In International Conference on Learning Representations. https://proceedings.iclr.cc/paper files/paper/2024/hash/ 85dca46374dc0f27b4bb5f265b3d17f0-Abstract-Conference.html

[38] Siyuan Guo, Cheng Deng, Ying Wen, Hechang Chen, Yi Chang, and Jun Wang. 2024. DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning. In Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 16813–16848. https://proceedings.mlr.press/v235/guo24b.html

## APPENDIX

Effectiveness and reporting units. For method $m ,$ learning run s, and task t, let $z _ { m , s , t }$ be the score returned by the benchmark evaluator, expressed on [0, 1] when the native score is reported as a percentage. The per-run score is

$$
S _ { m , s } = { \frac { 1 0 0 } { | T | } } \sum _ { t \in T } z _ { m , s , t } .\tag{6}
$$

EM and native binary task-success metrics have $z \in \{ 0 , 1 \}$ ; any native continuous score keeps its own name and interpretation. The four-benchmark main table reports arithmetic means over independent learning runs. Its Avg column is the unweighted mean of SearchQA, OfficeQA, DocVQA, and LiveMath, and does not pool tasks across domains. ALFWorld success rate is reported separately. Repeated evaluations of one frozen bank form a separate replication axis. Secondary metrics are not converted into binary accuracy by an added threshold.

Repeated outcomes. For a fixed bank and a binary correctness or success signal, let $y _ { t , k } \in \{ 0 , 1 \}$ be the outcome of repetition $k \in \{ 1 , \ldots , K \}$ on task t. The following expressions are rates; percentages multiply them by 100:

$$
\mathrm { R S R @ } K = \frac { 1 } { | T | } \sum _ { t \in T } \prod _ { k = 1 } ^ { K } y _ { t , k } ,\tag{7}
$$

$$
\mathrm { E I R @ } K = \frac { 1 } { | T | } \sum _ { t \in T } \mathbf { 1 } \left[ 0 < \sum _ { k = 1 } ^ { K } y _ { t , k } < K \right] .\tag{8}
$$

Repeated-run EM or SR averages all $| T | K$ binary outcomes. EIR is the fraction of tasks exhibiting correctness flips; RSR is the supplementary fraction solved in every repetition. Both use tasks as their denominator, so the K repetitions are not counted as independent tasks.

Paths. Let $A _ { t , k }$ be the canonical sequence of environment actions. Canonicalization removes method-internal bookkeeping, retains semantically relevant arguments, and expands environment calls inside programs. Restoration-discarded actions are excluded from the accepted path but remain in raw cost. With sequence Levenshtein distance $d ,$ define

$$
\sin ( A , B ) = 1 - { \frac { d ( A , B ) } { \operatorname* { m a x } \{ | A | , | B | , 1 \} } } ,\tag{9}
$$

$$
\mathrm { A P S @ } K = \frac { 1 } { | T | } \sum _ { t \in T } \frac { 2 } { K ( K - 1 ) } \sum _ { i < j } \sin ( A _ { t , i } , A _ { t , j } ) ,\tag{10}
$$

$$
\mathrm { E P R @ } K = \frac { 1 } { | T | } \sum _ { t \in T } \mathbf { 1 } [ A _ { t , 1 } = \cdot \cdot \cdot = A _ { t , K } ] .\tag{11}
$$

These formulas are applied to workloads with a meaningful action sequence. A direct answer with no such sequence does not receive an artificial perfect path score. Success-conditioned APS averages pairs in which both executions succeed, first within each eligible task and then across eligible tasks.

Resource variability. For nonnegative cost quantity $q _ { t , k }$ , let $\bar { q } _ { t }$ and $s _ { t } ( q )$ be its mean and sample standard deviation across repetitions. For $\bar { q } _ { t } > 0$

$$
\mathrm { C V } _ { q } ( t ) = 1 0 0 \frac { s _ { t } ( q ) } { \bar { q } _ { t } } .\tag{12}
$$

All-zero observed quantities have CV zero and are counted separately; missing quantities are excluded with their coverage stated. Mean task-level CV is reported with absolute mean usage. Failed episodes remain in the resource sample.

Program execution and reuse. For seed $s ,$ let $T _ { s }$ denote the test tasks and $K _ { s }$ the frozen persistent Bank. Persistent Program Coverage (PPC) is the fraction of tasks in $T _ { s }$ with at least one actually started invocation of a Program from $K _ { s }$ . A Program mentioned only in a plan, or synthesized temporarily for the current test task, does not contribute to PPC.

Consumed Program Outputs (CPO) reports a conservative, trace-confirmed lower bound on downstream consumption of Program results. A result contributes positive evidence only when its use by a subsequent operation, output binding, or accepted environment effect can be traced; a mere invocation or recorded return is insufficient. Cases without a verifiable consumption link are not inferred to be consumed. The reported lower bound is calculated within each seed before averaging.

Cross-Skill Reuse (CSR) is the fraction of actually used persistent Program versions that are invoked by at least two distinct Skill Skeletons. Multiple task instances of the same Skeleton do not by themselves establish cross-skill reuse. Program Invocation Success (PIS) is the number of invocations recorded as successful divided by all actually started Program invocations; invocation success alone does not establish final task correctness or universal semantic validity.

Program Action Share (PAS) is the ratio of canonical environment-action indices executed inside Program spans to all canonical environment-action indices. Program-internal environment actions are expanded, and the numerator is a union of indices so that nested bookkeeping does not double-count an action. Restoration-discarded actions are excluded from the accepted path but retained in cost accounting. Computationonly Programs with no environment action are captured by Program coverage rather than PAS.

For ALFWorld, the numerator/denominator pairs for CSR across seeds 42, 43, and 44 are 6/11, 7/14, and 6/10; for PIS they are 690/807, 641/713, and 698/766; and for PAS they are 1078/1209, 979/1178, and 1034/1163, respectively. Each reported percentage is calculated per seed and then averaged equally across the three seeds. PPC is 134/134 for each seed; the reported CPO values are lower bounds.

Graph Construction Coverage (GCC) is the fraction of tasks that enter an explicit nontrivial reusable or recomposed Skill Skeleton, excluding the one-node dynamic fallback. An LLM-free task requires no model request throughout the task execution; a Program identifier in the plan does not suffice.

Statistical comparisons. Paired task-level differences use the same physical task identities. A task bootstrap resamples tasks while keeping their repeated executions together; it does not treat three executions of a task as three independent samples. Independent learned banks form a separate replication axis. Binary paired comparisons can use McNemar’s test, with a stated multiplicity correction for a family of contrasts. No significance designation or uncertainty interval is inferred from rounded aggregate means alone.

Source-grounded checks. A reusable program’s test binding identifies the actual public inputs, source operation, and observed result. Required inputs are checked against the declared schema and linked to public fields or executedoperation arguments. Required outputs are linked to source result fields or independently recorded file effects. Hidden answer fields and task-identity lookup tables are not program inputs. An intermediate computation is not reclassified as a complete answer unless the recorded operation supports that output role. For validation, a separate environment instance is reset to the source task and the relevant action prefix is replayed. The program runs through the permitted interface. The check compares required outputs with the source reference and, where applicable, verifies the native operation and its allowed writes. File checks compare the recorded output effect while retaining the original artifact hash. Parameter variations are applied only when the source supplies a valid independent reference and the varied input satisfies the schema. These tests are evidence of local behavior, not exhaustive branch coverage or semantic equivalence on arbitrary inputs.

Side effects and failure handling. Isolated validation avoids changing the active task. Task-local execution uses a restorable snapshot where supported, and successful publication occurs only after the associated output checks. A failed attempt retains its actions, model usage, and error record. A restored-away interval is not positive evidence for later skill extraction. Where restoration is unavailable or fails, subsequent decisions depend on actual state observations; the framework does not assume arbitrary external operations are reversible.

Retrieval and invocation qualification. In the ALFWorld execution protocol, Skill Skeleton and atomic-skill candidates are filtered for capability and contract compatibility before lexical ranking. Atomic skills use the Dice-based score in Eq. (3), including recorded utility and active status; no embedding-based retrieval is used. Program realizations are resolved through explicit Skill–Implementation–Program references rather than independently ranked by vector similarity. Eligibility requires input binding, preconditions, environment compatibility, and applicable validation evidence, as summarized in Eq. (4). When no unique eligible or preferred implementation is identified, execution remains agent-guided.

Asset identity and compatibility. Operations, program realizations, and Skill Skeletons are versioned separately. Operations are consolidated only when they represent the same local capability and have compatible contracts; matching names or input–output types alone is insufficient. Program reuse additionally requires compatibility with the current inputs, preconditions, expected effects, and execution environment. The Program Asset Manager associates each accepted program version with its operation contract, applicable execution conditions, and validation evidence. Changes to the implementation, behavioral contract, or supported execution conditions require validation of the affected realization before admission. A failed realization does not invalidate the operation definition itself. Compatible replacements can serve existing Skill Skeletons if their operation requirements and parameter bindings remain satisfied.

Training and trace-based learning. Training trajectories provide evidence for skill induction and revision. The batchwise evolution process described in Section III allows accepted updates to influence subsequent task execution. The Skill Optimizer analyzes both successful and failed episodes to propose local changes and consolidate compatible skills. Retained training trajectories remain associated with the skill state and environmental observations of their original execution when reused for further revision. Task outcomes and local operation outcomes are distinguished. An unsuccessful task may contain independently checked successful operations that support skill learning. Conversely, overall task success does not establish that every intermediate program executed correctly. Failed calls and discarded or rolled-back segments support diagnosis and revision, while only independently checked, retained execution segments provide positive evidence for reusable operations.

Local validation and candidate selection. Local program validation checks whether a candidate realizes its operation under the tested inputs and execution conditions, using reference evidence independent of the candidate’s claimed outputs. Validation-set performance is used to select among candidate skills or bank snapshots. This selection evaluates their contribution to task performance and does not replace local program validation. The selected persistent skill state is fixed before the corresponding test evaluation.

Independent learning runs. Main task results and cost comparisons use seeds 42, 43, and 44. Each learning-based method constructs a separate bank for each model–benchmark– seed configuration. Task partitions are shared across methods and runs. Main performance and cost metrics are calculated within each independent run and then averaged over the three seeds. Fixed-bank repetition metrics are reported separately. Learning seeds control training order and supported local sampling; they do not imply deterministic responses from a hosted language model. No Skill and Human Skill are evaluated under the same run settings without model-based skill learning. Repeated execution. Stability and ablation analyses use the skill state learned with seed 42. For each method or condition, the same test tasks are executed three times with fixed persistent skills, task inputs, initial environment states, and inference settings. These repetitions examine variation in executing an established skill rather than variation between independently learned banks. The LiveMath analysis uses Gemini-3.8-Flash on 100 tasks, giving 300 task executions. The ALFWorld analysis uses DeepSeek-V4.1-Flash on 134 tasks, giving 402 task executions. Mean EM or SR summarizes execution outcomes, whereas EIR@3 counts tasks with both successful and unsuccessful outcomes. ALFWorld path metrics use canonical environment-action sequences, including actions executed inside programs. The Full ablation condition uses the same skill bank and repeated evaluations as the complete framework in the ALFWorld stability analysis. Metric definitions are given in Appendix A.

Frozen skills and task-local execution. The persistent learned skill and Program Asset Bank state remains fixed throughout test evaluation. Task-local argument grounding, error recovery, and programming permitted by the corresponding method remain confined to the current task execution. Temporary programs and intermediate results are not added to the persistent bank or transferred to later test tasks. Each repeated execution starts from the same selected bank and reset task-local state. Test outcomes are used for reporting performance and execution behavior, not for selecting or revising the bank.

Baselines and textual priors. Human Skill uses the benchmark-specific human-written initial.md released by the SkillOpt authors [10]. This textual skill remains unchanged during testing and repeated execution. Other baselines retain their own initialization procedures, learning algorithms, and knowledge representations under the shared task interface. Within each model–benchmark comparison, methods use the same task resources, native tools, evaluators, and inference settings. The reported SKILLCOMPILER banks are learned through the skill evolution process described in Section III. Ablation scope. The three interventions in Section IV-E are evaluated on ALFWorld with DeepSeek-V4.1-Flash. Each condition learns independently from an empty bank using seed 42, and its intervention applies during both learning and testing.

Without Skill Skeleton disables multi-operation skeleton construction and retrieval while retaining individual operation retrieval, programmatic execution, and trace-driven learning. Without Programmer disables program synthesis, qualification, and invocation, including reuse of stored program realizations, while retaining textual skill learning, Skill Skeletons, and native tools. Without Skill Optimization permits the acquisition of new capabilities and an existing textual operation’s first validated program realization, but prevents subsequent semantic revision or merging of accepted guidance, contracts, skeletons, and program realizations. Initial program validation and bounded repair remain available. All conditions retain runtime grounding and error recovery within their permitted execution mechanisms, with shared task partitions, native tools, and budget limits.

Resource accounting. Learning tokens include trainingtrajectory collection, skill construction and revision, feedback, and validation-based selection. Each model request is counted once, and each bank’s learning investment is counted once before averaging across learning runs. Test tokens include input and output tokens from successful and unsuccessful executions, method-internal model calls, and retries. Calls/task counts actual model HTTP requests. Retries contribute to the cost of their original task execution without increasing the number of evaluated tasks. An execution completed entirely by programs contributes zero model requests. The main cost tables average the per-run measurements from seeds 42, 43, and 44. Ablation costs instead average the three repeated evaluations of each condition’s seed-42 bank. These summaries characterize resource use across independently learned banks and during repeated use of a fixed bank, respectively.

Reproducibility and development history. Experimental records associate each run with its model and inference settings, task partition, learning seed, selected bank version, task outcomes, execution traces, and learning and test resource usage. Reasoning settings, completion limits, and retry policies are held consistent within each model–benchmark comparison. Development included inspection of held-out failure cases; the reported frozen-bank evaluations prohibit persistent updates from test outcomes.

Scope of validation. Programs execute within the permitted benchmark interfaces and task resources under bounded execution. Local validation provides evidence for the checked inputs, environment configurations, and operation effects. Its coverage is limited to those conditions. Task correctness is determined by the benchmark evaluator, while unresolved semantics and context-dependent decisions remain under the target agent’s control.