# Complexity-Aware Evaluation of LLM Comprehension

1<sup>st</sup> Ali Mohammadi Esfahani

Systems and computer Engineering

Carleton University

Ottawa, Canada

alimohammadiesfahani@cmail.carleton.ca

2<sup>nd</sup> Nafiseh Kahani

Systems and computer Engineering Carleton University Ottawa, Canada kahani@sce.carleton.ca

3<sup>rd</sup> Samuel A.Ajila   
Systems and computer Engineering   
Carleton University   
Ottawa, Canada   
samuel.ajila@cunet.carleton.ca

Abstract—Large language models (LLMs) are increasingly used for software engineering tasks that require understanding existing source code, including behavior prediction, function explanation, debugging, and code review. However, aggregate benchmark accuracy can conceal how model reliability changes as source code becomes structurally more complex. This paper presents a complexity-aware framework for evaluating LLM code comprehension using cyclomatic complexity, nesting depth, branching factor, and Halstead volume. We evaluate DeepSeek-Coder-V2 and Llama through two complementary tasks: automatic input–output prediction over 300 Python functions and manually assessed semantic comprehension over a balanced subset of 60 functions. The functions are grouped into Low-, Medium-, and High-complexity bands. DeepSeek-Coder-V2 achieves an overall automatic accuracy of 78.33%, compared with 70.33% for Llama. However, accuracy decreases substantially from Low to High complexity, from 93.52% to 52.78% for DeepSeek-Coder-V2 and from 87.04% to 47.22% for Llama. Incorrect predictions are consistently associated with higher values of all four complexity metrics, and correlation and logisticregression analyses confirm broadly comparable negative associations between structural complexity and correctness. Manual semantic comprehension shows the same degradation pattern, with accuracy decreasing from 100.00% to 75.00% for DeepSeek-Coder-V2 and from 90.00% to 60.00% for Llama. These findings demonstrate that complexity-aware evaluation provides a more diagnostic assessment of LLM code-comprehension reliability than aggregate accuracy alone.

Index Terms—Large language models, code comprehension, software engineering, structural complexity, cyclomatic complexity, Halstead metrics, input–output prediction, semantic comprehension.

## I. INTRODUCTION

Large language models (LLMs) are increasingly used in software engineering tasks that require understanding existing source code [1]–[5]. Beyond code generation, developers use LLMs to explain functions, predict program behavior, reason about edge cases, summarize implementation logic, and support debugging and code review [6]–[8]. These tasks differ from code synthesis because they require the model to interpret the control flow, internal state, and behavior of an existing program. Consequently, code-generation performance alone does not fully characterize the reliability of LLMs in practical software engineering workflows.

Code-comprehension difficulty is not uniform across programs. A function containing a single condition is generally easier to interpret than one with multiple branches, nested loops, interacting variables, and alternative execution paths [9], [10]. As structural complexity increases, the model must track more control-flow decisions, intermediate states, boundary conditions, and symbolic operations. Aggregate accuracy can therefore conceal the conditions under which comprehension begins to fail [11].

Existing benchmarks provide important foundations for evaluating programming-oriented LLMs. HumanEval [12] and MBPP [13] are widely used for code generation and program synthesis, while APPS [14] and CodeContests [15] contain more demanding programming problems. CRUXEval [11] and BigCodeBench [16] extend evaluation toward code reasoning, execution, and realistic function usage. However, benchmarklevel scores are commonly reported without considering the structural complexity distribution of the evaluated functions. A benchmark dominated by simple functions may therefore overestimate model reliability on structurally demanding code.

Prior studies have shown that LLM-generated code can contain semantic, logical, and algorithmic defects even when it appears syntactically plausible [17]. Reinforcement-learningbased prompt optimization has also demonstrated that codegeneration performance depends on how programming tasks are formulated [18], while complexity-aware feedback has connected code complexity with generation success [19]. These studies motivate evaluation beyond aggregate correctness, but they primarily examine generated-code quality and prompt-based improvement. The effect of structural complexity on comprehension of existing source code remains less directly investigated.

This paper addresses this gap through a complexity-aware evaluation method for LLM code comprehension. Each Python function is annotated using cyclomatic complexity, nesting depth, branching factor, and Halstead volume, and is assigned to a Low-, Medium-, or High-complexity band. The method combines two complementary tasks: automatic input–output prediction, in which the model predicts the exact return value of a concrete function call, and manual semantic comprehension, in which the model answers questions about function purpose, intermediate variable roles, or edge-case behavior.

The main contributions of this paper are:

• A complexity-aware method that combines automatic be-

havioral prediction with manual semantic comprehension.

• An analysis of how benchmark composition and structural complexity influence observed LLM comprehension accuracy.

• Statistical evidence relating structural complexity metrics to incorrect predictions and complexity-related degradation across both evaluation tasks.

The study is guided by the following research questions: RQ1: How does the complexity distribution of benchmark sources influence observed LLM comprehension performance? RQ2: Which structural complexity metrics are associated with incorrect LLM comprehension predictions?

RQ3: Does manual semantic comprehension reveal the same complexity-related reliability limitations observed in automatic input–output prediction?

## II. RELATED WORK

Recent research has increasingly moved beyond codegeneration accuracy to examine whether LLMs can reason about the behavior and meaning of existing programs. CRUX-Eval [11] evaluates input and output prediction over short Python functions and shows that strong performance on generation benchmarks does not necessarily transfer to execution reasoning. CodeMind [20] further separates independent execution, dependent execution, and specification reasoning, reporting that models become less reliable when programs contain non-trivial control flow, arithmetic operations, complex data types, or API calls. These studies establish that code reasoning is distinct from code synthesis, but they characterize difficulty mainly through task design rather than explicit structural-complexity bands.

Other benchmarks broaden code comprehension beyond exact output prediction. CRQBench [21] evaluates naturallanguage questions derived from code-review comments and demonstrates that even advanced models can produce incorrect or weakly grounded explanations. LiveCodeBench [22] provides a continuously updated, contamination-resistant evaluation covering generation, execution, self-repair, and testoutput prediction, while BigCodeBench [16] emphasizes realistic function calls, library usage, and more demanding programming instructions. These benchmarks improve realism and coverage, but their primary objective is broad capability assessment rather than determining how comprehension changes as the internal structure of an individual function becomes more complex.

Repository-level benchmarks address a different source of difficulty. RepoBench [23] studies code completion with crossfile context, and LongCodeBench [24] evaluates comprehension and repair under very long context windows. Their results show that retrieval, cross-file dependencies, and context length remain challenging for LLMs. However, contextual scale is different from structural complexity: a model may fail because relevant information is distributed across files, rather than because a single function contains deep nesting, dense branching, or many possible execution paths.

More recent studies examine complexity and reasoning fidelity more directly. RE2-Bench [25] evaluates realistic projects containing nested constructs, complex types, and API interactions, but summarizes difficulty using an Easy/Hard division. Machtle et al. [26] analyze the relationship be-¨ tween model performance and conventional properties such as lexical size, control-flow complexity, and abstract-syntax-tree structure. Xie et al. [27] propose LM-CC, a model-perceived complexity measure, and argue that traditional metrics may not fully capture LLM difficulty once code length is controlled. CoRE [28] additionally shows that a model can predict the correct final result while reasoning incorrectly about intermediate execution states, demonstrating that output-only evaluation may overestimate comprehension.

Prior work shows that LLM code comprehension is affected by execution demands, semantic reasoning, contextual scale, and program complexity. However, existing studies typically report aggregate accuracy, use a single difficulty label or metric, or rely on one evaluation modality. The present study addresses this gap by analyzing cyclomatic complexity, nesting depth, branching factor, and Halstead volume jointly, grouping functions into Low-, Medium-, and High-complexity bands, and combining automatic input–output prediction with manually assessed questions about function purpose, variable roles, and edge-case behavior.

## III. COMPLEXITY-AWARE EVALUATION METHOD

We propose a controlled method for evaluating LLM code comprehension across different levels of structural complexity. Rather than measuring code-generation ability, the method assesses whether a model can infer the behavior and semantics of existing Python functions.

Each benchmark instance is represented as $\begin{array} { r l } { \mathbf { x } } & { { } = } \end{array}$ $( c , q , y , m , t )$ , where c denotes the source function, q denotes a function-call query or semantic question, y denotes the ground-truth answer, m denotes the structural-complexity vector, and t ∈ {AUTOMATIC, MANUAL} identifies the task type. The complexity vector includes cyclomatic complexity [29], nesting depth [30], branching factor [27], and Halstead volume [31]. Given $( c , q )$ , the model produces a prediction yˆ, which is compared with y to determine correctness.

The evaluation combines two complementary tasks. In automatic input–output prediction, the model receives a complete function and a concrete function call and must return the exact output. This task provides deterministic ground truth and supports scalable exact-match evaluation. In manual semantic comprehension, the model answers a question about the function’s purpose, the role of an intermediate variable, or edgecase behavior. This task captures semantic understanding that cannot be assessed through exact output prediction alone.

The evaluated functions are organized into Low, Medium, and High-complexity bands, and the same zero-shot protocol is applied to DeepSeek-Coder-V2 and CodeLlama-7b-Instructhf. The method consists of four stages: dataset construction, structural-complexity annotation, model evaluation using standardized prompts, and complexity-aware analysis across bands, dataset sources, task types, and failure patterns.

TABLE I  
AUTOMATIC BENCHMARK COMPOSITION BY DATASET SOURCE AND COMPLEXITY BAND.
<table><tr><td>Dataset</td><td>Low</td><td>Medium</td><td>High</td><td>Total</td></tr><tr><td>HumanEval</td><td>32</td><td>48</td><td>0</td><td>80</td></tr><tr><td>MBPP-sanitized</td><td>50</td><td>30</td><td>0</td><td>80</td></tr><tr><td>CRUXEval</td><td>26</td><td>20</td><td>0</td><td>46</td></tr><tr><td>LeetCode</td><td>0</td><td>12</td><td>48</td><td>60</td></tr><tr><td>BigCodeBench-Hard</td><td>0</td><td>10</td><td>24</td><td>34</td></tr><tr><td>Total</td><td>108</td><td>120</td><td>72</td><td>300</td></tr></table>

## A. Dataset Construction

The benchmark contains 300 short, self-contained Python functions collected from HumanEval [12], MBPPsanitized [13], CRUXEval [11], LeetCode [32], and BigCodeBench-Hard [16]. Python was selected because it is widely represented in code benchmarks and supports reproducible structural analysis through the built-in ast module. HumanEval, MBPP-sanitized, and CRUXEval mainly contribute Low- and Medium-complexity functions, whereas LeetCode and BigCodeBench-Hard provide most of the High-complexity samples.

Candidate functions were retained only if they were selfcontained, deterministic, executable using the Python standard library, short enough to fit within the prompt, and independent of hidden state, external files, user interaction, randomness, or system-specific behavior. Each selected function was paired with a concrete function call and a ground-truth output obtained through controlled execution. Functions that failed, timed out, produced ambiguous outputs, or required undocumented assumptions were excluded.

A manual semantic-comprehension subset was sampled from the same benchmark. Twenty functions were selected without replacement from each complexity band, producing a balanced set of 60 functions. Each function was paired with one question concerning its purpose, an intermediate variable role, or edge-case behavior. This stratified design supports direct comparison across Low-, Medium-, and High-complexity functions while keeping manual annotation manageable.

Table I summarizes the final benchmark composition.

## B. Automatic Comprehension Tasks

The automatic task evaluates whether an LLM can infer the behavior of an existing Python function. For each instance, the model receives the complete function and a concrete function call and must predict the exact return value. This formulation provides deterministic ground truth and supports objective evaluation at scale.

Input arguments were obtained from the original benchmark test cases when available; otherwise, valid and non-trivial inputs were generated programmatically. The corresponding ground-truth outputs were obtained by executing the functions in a controlled environment.

Each instance was presented using the standardized prompt shown in Figure 1. The same zero-shot prompt was used for both evaluated models, without few-shot examples, Chain-of-Thought instructions, or task-specific hints.

1 You are given the following Python function:   
2   
3 {FUNCTION SOURCE CODE}   
4   
5 What is the return value of the function when called   
as:   
6 {FUNCTION\_NAME}({INPUT\_ARGUMENTS})?   
7   
8 Return only the final output value.   
9 Do not include any explanation, reasoning steps, or   
additional text.

Fig. 1. Prompt template for the automatic input–output comprehension task.

A prediction was marked correct when its returned value matched the ground-truth output after light normalization of semantically irrelevant formatting differences, such as whitespace, quotation style, or Boolean capitalization. Empty responses, incorrect values, and responses containing additional explanations were marked incorrect without partial credit.

## C. Manual Comprehension Tasks

The manual task evaluates semantic understanding that exact input–output prediction cannot fully capture. A balanced subset of 60 functions was sampled from the automatic benchmark, with 20 functions from each complexity band. Each function was paired with one manually written question, producing 20 questions for each of three categories: purpose, which assesses the function’s overall intent; variable role, which examines how an intermediate variable contributes to the computation; and edge case, which evaluates reasoning about boundary or uncommon execution paths.

All instances were presented using the standardized prompt in Figure 2. The same zero-shot template was used for DeepSeek-Coder-V2 and CodeLlama-7b-Instruct-hf.

1 You are given the following Python function:   
2   
3 {FUNCTION SOURCE CODE}   
4   
5 Question:   
{MANUAL QUESTION}   
7   
8 Answer concisely in one or two sentences.

## Fig. 2. Prompt template for the manual semantic comprehension task.

Canonical reference answers identified the essential semantic elements required for correctness. A response was marked correct only if it included all required elements without introducing a contradictory interpretation; otherwise, it was marked incorrect, with no partial credit.

To assess annotation reliability, a randomly selected 20% of the manual responses was independently scored by a second annotator using the same binary rubric. Cohen’s κ was calculated before disagreements were resolved, and the reconciled labels were used in the final analysis.

## IV. STUDY DESIGN

This section describes how the functions were annotated, how the models were evaluated, and how comprehension performance was analyzed across structural complexity levels.

## A. Complexity Annotation

To enable complexity-aware analysis, each function in the benchmark is annotated with structural complexity metrics computed through static analysis. These metrics are selected because they capture complementary properties of code structure that are known to affect program comprehension [33], [34]. In this study, four metrics are used: cyclomatic complexity, nesting depth, branching factor, and Halstead volume.

1) Complexity Metrics: Cyclomatic complexity (CC) [30] measures the number of linearly independent execution paths through a function and is formally defined as:

$$
\begin{array} { r } { C C = E - N + 2 P } \end{array}\tag{1}
$$

where E is the number of edges in the control-flow graph, N is the number of nodes, and P is the number of connected components. In practice, CC is computed by counting decision points such as if, elif, for, while, except, and Boolean operators such as and and or, with a baseline value of 1 for a function with no branching. Higher CC values indicate a larger number of possible execution paths and greater reasoning difficulty.

Nesting depth (ND) [29] measures the maximum level of nested control structures within a function, including nested loops, conditionals, and exception-handling blocks. Deeply nested code requires the model to track multiple simultaneous execution contexts, which increases reasoning difficulty and the likelihood of comprehension failure.

Branching factor (BF) [27] counts the number of decision points in a function, including constructs such as if, for, and while. While CC captures the number of independent execution paths, BF captures how frequently the control flow diverges. This metric therefore reflects the density of local decision-making within the code.

Halstead volume (HV) [31] measures lexical and operational complexity based on the number of operators and operands in the function. It captures a different aspect of complexity from control-flow metrics by reflecting the amount of symbolic information that must be interpreted. Higher HV values indicate that the model must process a larger and more varied set of program tokens, variables, and operations.

For all Python functions, the complexity metrics are computed statically from the parsed source code. Cyclomatic complexity, nesting depth, and branching factor are computed using Python’s built-in ast module. Halstead volume is computed from the operators and operands extracted from the same source representation. The raw metric values are retained for statistical analysis, and the functions are also assigned to categorical complexity bands as described in Section IV-A2.

2) Complexity Bands: Each function was assigned to a Low-, Medium-, or High-complexity band using cyclomatic complexity and nesting depth as the primary criteria. If the two metrics indicated different bands, the higher band was selected to avoid underestimating structural difficulty. Branching factor and Halstead volume were retained for descriptive and metriclevel analyses rather than primary band assignment. Table II summarizes the corresponding ranges.

Both the categorical band and the raw values of all four metrics were retained, enabling band-wise performance comparison and continuous analysis of the relationship between structural complexity and model correctness.

## B. Model Evaluation Protocol

Two code-oriented LLMs were evaluated: DeepSeek-Coder-V2 [35] and CodeLlama-7b-Instruct-hf [36]. The models were selected to compare different code-specialized architectures and to examine whether their comprehension accuracy degrades similarly as structural complexity increases.

Both models were evaluated under identical zero-shot conditions using the prompt templates in Sections III-B and III-C. The temperature was set to 0, and no few-shot examples, Chain-of-Thought instructions, or task-specific hints were provided. Each model was queried once per benchmark instance.

For the automatic task, the model returned only the predicted output of the supplied function call, which was evaluated against the executed ground truth. For the manual task, responses were assessed using the binary semantic rubric defined in Section III-C. Model outputs, reference answers, correctness labels, dataset sources, complexity bands, and raw complexity metrics were retained for aggregate, band-wise, dataset-level, and metric-level analyses.

## C. Evaluation Metric

Model performance is evaluated using binary correctness and reported as accuracy. For each instance, the prediction $\hat { y } _ { i }$ is assigned a value of 1 when it matches the ground-truth answer $y _ { i } .$ , and 0 otherwise:

$$
\mathrm { A c c u r a c y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } mathbf { 1 } ( \hat { y } _ { i } = y _ { i } ) ,\tag{2}
$$

where N is the number of evaluated instances and 1(·) is the indicator function.

For the automatic task, semantically equivalent formatting differences, including whitespace, quotation style, and Boolean capitalization, were normalized before comparison. Incorrect values, empty outputs, and responses containing additional explanations were marked incorrect. For the manual task, a response was marked correct only when it contained all essential elements of the canonical answer without contradiction; no partial credit was assigned.

Accuracy is reported by model, complexity band, dataset source, and task type.

TABLE II
<table><tr><td>Band</td><td>Cyclomatic Complexity</td><td>Nesting Depth</td><td>Branching Factor</td><td>Halstead Volume</td></tr><tr><td>Low</td><td>1-3</td><td>0-1</td><td>0-2</td><td> $\overline { { V < 1 0 0 } }$ </td></tr><tr><td>Medium</td><td>4-6</td><td>2-3</td><td>3-5</td><td> $1 0 0 \leq V < 3 0 0$ </td></tr><tr><td>High</td><td>≥ 7</td><td>≥4</td><td>≥ 6</td><td> $V \geq 3 0 0$ </td></tr></table>

COMPLEXITY BAND DEFINITIONS USED FOR STRUCTURAL CODE ANALYSIS.

## D. Complexity-Aware Analysis

The analysis is conducted at three levels. First, accuracy is compared across Low-, Medium-, and High-complexity bands to determine whether aggregate performance conceals degradation on structurally complex functions. Second, accuracy is reported by dataset source and interpreted together with each source’s complexity distribution. Third, correct and incorrect predictions are compared using the raw values of cyclomatic complexity, nesting depth, branching factor, and Halstead volume.

The joint relationship between structural complexity and automatic prediction correctness is estimated using multivariate logistic regression. Let ${ Y _ { i } } ~ \in ~ \{ 0 , 1 \}$ denote whether the prediction for function i is correct. The model is defined as:

$$
\mathrm { l o g i t } \ \mathrm { P r } ( Y _ { i } = 1 ) = \beta _ { 0 } + \beta _ { 1 } C C _ { i } + \beta _ { 2 } N D _ { i } + \beta _ { 3 } B F _ { i } + \beta _ { 4 } \left( \frac { H V _ { i } } { 1 0 0 } \right)\tag{3}
$$

where $C C _ { i } , N D _ { i } , B F _ { i } .$ , and $H V _ { i }$ denote cyclomatic complexity, nesting depth, branching factor, and Halstead volume, respectively. Halstead volume is divided by 100 because its numerical scale is substantially larger than those of the other metrics. Negative coefficients indicate that increasing complexity is associated with lower odds of a correct prediction, while each coefficient is interpreted jointly with the other included metrics.

For the manual subset, accuracy is compared across complexity bands and question types to determine whether the degradation observed in automatic input–output prediction also appears in semantic comprehension.

## E. Statistical Analysis

Descriptive accuracy is reported by model, complexity band, dataset source, and manual question type. For the automatic task, the mean values of cyclomatic complexity, nesting depth, branching factor, and Halstead volume are also compared between correct and incorrect predictions.

Spearman’s rank correlation is used to examine the monotonic association between each structural metric and automatic prediction correctness. For model responses $Y _ { i } \in \{ 0 , 1 \}$ and complexity metric $m _ { i }$ , the correlation is defined as:

$$
\rho _ { s } = \operatorname { c o r r } \big ( \operatorname { r a n k } ( Y _ { i } ) , \operatorname { r a n k } ( m _ { i } ) \big ) .\tag{4}
$$

A negative value of $\rho _ { s }$ indicates that higher structural complexity is associated with lower correctness. The joint effects of the four metrics are estimated using the multivariate logistic-regression model in Equation 3. Because the predictors may be correlated, the coefficients are interpreted as conditional associations rather than as independent measures of metric importance.

TABLE III  
AUTOMATIC COMPREHENSION ACCURACY BY COMPLEXITY BAND.
<table><tr><td>Model</td><td>Band</td><td>Correct</td><td>Total</td><td>Accuracy</td></tr><tr><td rowspan="3">DeepSeek-Coder-V2</td><td>Low</td><td>101</td><td>108</td><td>93.52%</td></tr><tr><td>Medium</td><td>96</td><td>120</td><td>80.00%</td></tr><tr><td>High</td><td>38</td><td>72</td><td>52.78%</td></tr><tr><td rowspan="3">CodeLlama-7b-Instruct-hf</td><td>Low</td><td>94</td><td>108</td><td>87.04%</td></tr><tr><td>Medium</td><td>83</td><td>120</td><td>69.17%</td></tr><tr><td>High</td><td>34</td><td>72</td><td>47.22%</td></tr></table>

For the manual task, a chi-square test of independence evaluates the association between complexity band and correctness, with Cramer’s´ V reported as the effect-size measure. Annotation reliability is assessed using Cohen’s κ on the independently scored subset before disagreements are reconciled. Statistical significance is assessed at $\alpha = 0 . 0 5$

## V. RESULTS

This section reports the results for the three research questions, covering benchmark composition, metric-level failure patterns, and manual semantic comprehension.

## A. RQ1: Effect of Benchmark Complexity Distribution

The automatic benchmark contains 108 Low-, 120 Medium-, and 72 High-complexity functions. As shown in Table I, HumanEval, MBPP-sanitized, and CRUXEval mainly contribute Low- and Medium-complexity functions, whereas LeetCode and BigCodeBench-Hard provide most of the High-complexity samples.

DeepSeek-Coder-V2 correctly answered 235 of 300 instances, achieving 78.33% overall accuracy. CodeLlama-7b-Instruct-hf correctly answered 211 instances, achieving 70.33%. However, these aggregate values conceal substantial degradation across complexity bands.

As shown in Table III, DeepSeek-Coder-V2 decreases from 93.52% accuracy on Low-complexity functions to 52.78% on High-complexity functions, a decline of 40.74 percentage points. CodeLlama-7b-Instruct-hf follows the same pattern, decreasing from 87.04% to 47.22%, a decline of 39.82 percentage points. DeepSeek-Coder-V2 remains more accurate in every band, but both models become substantially less reliable on structurally complex functions.

Dataset-level results follow the same pattern. Both models achieve their highest accuracies on HumanEval, MBPPsanitized, and CRUXEval, which contain only Low- and

TABLE IV  
AUTOMATIC COMPREHENSION ACCURACY BY DATASET SOURCE.
<table><tr><td>Model</td><td>Dataset</td><td>Correct</td><td>Total</td><td>Accuracy</td></tr><tr><td rowspan="5">DeepSeek-Coder-V2</td><td>HumanEval</td><td>70</td><td>80</td><td>87.50%</td></tr><tr><td>MBPP-sanitized</td><td>71</td><td>80</td><td>88.75%</td></tr><tr><td>CRUXEval</td><td>39</td><td>46</td><td>84.78%</td></tr><tr><td>LeetCode</td><td>39</td><td>60</td><td>65.00%</td></tr><tr><td>BigCodeBench-Hard</td><td>16</td><td>34</td><td>47.06%</td></tr><tr><td rowspan="5">CodeLlama-7b-Instruct-hf</td><td>HumanEval</td><td>65</td><td>80</td><td>81.25%</td></tr><tr><td>MBPP-sanitized</td><td>66</td><td>80</td><td>82.50%</td></tr><tr><td>CRUXEval</td><td>35</td><td>46</td><td>76.09%</td></tr><tr><td>LeetCode</td><td>32</td><td>60</td><td>53.33%</td></tr><tr><td>BigCodeBench-Hard</td><td>13</td><td>34</td><td>38.24%</td></tr></table>

TABLE V

AVERAGE COMPLEXITY METRICS FOR CORRECT AND INCORRECT PREDICTIONS.
<table><tr><td>Model</td><td>Outcome</td><td>CC</td><td>ND</td><td>BF</td><td>HV</td></tr><tr><td rowspan="2">DeepSeek-Coder-V2</td><td>Correct</td><td>3.84</td><td>1.72</td><td>3.11</td><td>142.60</td></tr><tr><td>Incorrect</td><td>7.42</td><td>3.46</td><td>6.89</td><td>318.75</td></tr><tr><td rowspan="2">CodeLlama-7b-Instruct-hf</td><td>Correct</td><td>3.51</td><td>1.58</td><td>2.94</td><td>131.40</td></tr><tr><td>Incorrect</td><td>7.24</td><td>3.32</td><td>6.27</td><td>297.80</td></tr></table>

TABLE VI

SPEARMAN CORRELATIONS BETWEEN STRUCTURAL COMPLEXITY AND AUTOMATIC PREDICTION CORRECTNESS. ALL CORRELATIONS ARE STATISTICALLY SIGNIFICANT AT p < 0.001.
<table><tr><td>Model</td><td>CC</td><td>ND</td><td>BF</td><td>HV</td></tr><tr><td>DeepSeek-Coder-V2</td><td>-0.41</td><td>-0.35</td><td>-0.43</td><td> $- 0 . 3 7$ </td></tr><tr><td>CodeLlama-7b-Instruct-hf</td><td> $- 0 . 3 9$ </td><td> $- 0 . 3 3$ </td><td> $- 0 . 4 0$ </td><td>-0.35</td></tr></table>

Medium-complexity functions. Accuracy is substantially lower on LeetCode and BigCodeBench-Hard, where Highcomplexity functions are concentrated.

These results answer RQ1 by showing that benchmarklevel accuracy depends strongly on the underlying complexity distribution. Aggregate scores from benchmarks dominated by simpler functions can therefore overstate model reliability on structurally demanding code.

## B. RQ2: Structural Characteristics of Incorrect Predictions

RQ2 examines which structural metrics are associated with incorrect comprehension predictions. Table V compares the average cyclomatic complexity (CC), nesting depth (ND), branching factor (BF), and Halstead volume (HV) of correctly and incorrectly answered functions.

For both models, incorrect predictions are associated with higher values of all four metrics. For DeepSeek-Coder-V2, average CC increases from 3.84 to 7.42, ND from 1.72 to 3.46, BF from 3.11 to 6.89, and HV from 142.60 to 318.75. CodeLlama-7b-Instruct-hf shows the same pattern, with CC increasing from 3.51 to 7.24, ND from 1.58 to 3.32, BF from 2.94 to 6.27, and HV from 131.40 to 297.80. These results indicate that failures are concentrated in functions with more execution paths, deeper control structures, denser branching, and greater symbolic content.

As shown in Table VI, all four metrics are negatively correlated with correctness for both models. The coefficients range from −0.35 to −0.43 for DeepSeek-Coder-V2 and from −0.33 to −0.40 for CodeLlama-7b-Instruct-hf. Although branching factor and cyclomatic complexity are numerically the largest associations, the differences are small and their confidence intervals overlap substantially. The results therefore do not support identifying any single metric as a statistically stronger predictor.

These findings answer RQ2 by showing that incorrect predictions are systematically associated with greater structural complexity across all measured dimensions. The four metrics can therefore serve as complementary risk indicators: predictions for structurally complex functions should receive additional verification through execution, testing, or human review.

## C. RQ3: Manual Semantic Comprehension

RQ3 examines whether manual semantic comprehension reveals the same complexity-related reliability limitations observed in automatic input–output prediction. The manual subset evaluates three aspects of semantic understanding: function purpose, intermediate variable roles, and edge-case behavior.

To illustrate the distinction between automatic and manual evaluation, consider the Candy function from the LeetCode subset, shown in figure 3.

```python
def min_candies(ratings):
n = len(ratings)
candies = [1] n
for i in range(1, n):
if ratings[i] > ratings[i - 1]:
candies[i] = candies[i - 1] + 1
for i in range(n - 2, -1, -1):
if ratings[i] > ratings[i + 1]:
candies[i] = max(
candies[i],
candies[i + 1] + 1
)
return sum(candies)
```  
Fig. 3. Example function used for manual semantic comprehension.

This function belongs to the Medium-complexity band, with cyclomatic complexity of 5, nesting depth of 3, branching factor of 4, and Halstead volume of 136.0. In the automatic task, the model predicts the output of a call such as min\_candies([1,0,2]), whose return value is 5. In the manual task, the model may instead be asked: What is the role of the variable candies? A correct answer must explain that the variable stores the allocation for each child and is updated through two directional passes to satisfy the neighboring rating constraints. Thus, the manual task evaluates understanding of the algorithm’s internal state rather than only its final output.

TABLE VII  
DISTRIBUTION OF MANUAL QUESTION TYPES ACROSS COMPLEXITY BANDS.
<table><tr><td>Question Type</td><td>Low</td><td>Medium</td><td>High</td><td>Total</td></tr><tr><td>Purpose</td><td>7</td><td>7</td><td>6</td><td>20</td></tr><tr><td>Variable Role</td><td>7</td><td>6</td><td>7</td><td>20</td></tr><tr><td>Edge Case</td><td>6</td><td>7</td><td>7</td><td>20</td></tr><tr><td>Total</td><td>20</td><td>20</td><td>20</td><td>60</td></tr></table>

TABLE VIII  
MANUAL COMPREHENSION ACCURACY BY QUESTION TYPE.
<table><tr><td>Model</td><td>Question Type</td><td>Correct</td><td>Total</td><td>Acc.</td></tr><tr><td rowspan="3">DeepSeek-Coder-V2</td><td>Purpose</td><td>19</td><td>20</td><td>95.00%</td></tr><tr><td>Variable Role</td><td>18</td><td>20</td><td>90.00%</td></tr><tr><td>Edge Case</td><td>16</td><td>20</td><td>80.00%</td></tr><tr><td rowspan="3">CodeLlama-7b-Instruct-hf</td><td>Purpose</td><td>17</td><td>20</td><td>85.00%</td></tr><tr><td>Variable Role</td><td>16</td><td>20</td><td>80.00%</td></tr><tr><td>Edge Case</td><td>13</td><td>20</td><td>65.00%</td></tr></table>

To examine whether question-type difficulty is confounded with complexity level, Table VII reports the distribution of the three question types across the Low-, Medium-, and High-complexity bands. Each question type is represented by approximately the same number of functions in every band.

Because the question types differ by at most one function within each complexity band, the question-type accuracy comparison is not merely a restatement of the complexity-band effect.

Table VIII shows that purpose questions were the least difficult for both models. DeepSeek-Coder-V2 achieved 95.00% accuracy, compared with 85.00% for CodeLlama-7b-Instructhf. Accuracy decreased for variable-role questions to 90.00% and 80.00%, respectively, and was lowest for edge-case questions at 80.00% and 65.00%.

This ordering reflects increasing semantic demand. Purpose questions can often be answered by recognizing the overall algorithmic pattern. Variable-role questions require tracking how internal state is initialized, updated, and used. Edge-case questions are more difficult because they require reasoning about boundary conditions and less frequently executed paths. The three question types were approximately balanced across the complexity bands, so this ordering is not simply caused by concentrating edge-case questions in the High-complexity group.

As shown in Table IX, manual accuracy declined consistently with structural complexity. DeepSeek-Coder-V2 decreased from 100.00% on Low-complexity functions to 90.00% on Medium-complexity functions and 75.00% on

TABLE IX  
MANUAL COMPREHENSION ACCURACY BY COMPLEXITY BAND.
<table><tr><td>Model</td><td>Band</td><td>Correct</td><td>Total</td><td>Acc.</td></tr><tr><td rowspan="3">DeepSeek-Coder-V2</td><td>Low</td><td>20</td><td>20</td><td>100.00%</td></tr><tr><td>Medium</td><td>18</td><td>20</td><td>90.00%</td></tr><tr><td>High</td><td>15</td><td>20</td><td>75.00%</td></tr><tr><td rowspan="3">CodeLlama-7b-Instruct-hf</td><td>Low</td><td>18</td><td>20</td><td>90.00%</td></tr><tr><td>Medium</td><td>16</td><td>20</td><td>80.00%</td></tr><tr><td>High</td><td>12</td><td>20</td><td>60.00%</td></tr></table>

High-complexity functions. CodeLlama-7b-Instruct-hf followed the same pattern, decreasing from 90.00% to 80.00% and then to 60.00%. The corresponding Low-to-High declines were 25.00 and 30.00 percentage points.

Table X shows that automatic prediction was more sensitive to structural complexity than manual semantic comprehension. Exact output prediction requires precise execution tracing; an error in branch selection, loop simulation, or state tracking directly produces an incorrect answer. Manual questions may sometimes be answered through higher-level semantic recognition, explaining their smaller but still substantial degradation.

These results answer RQ3 by showing that complexityrelated failure is not an artifact of exact-match output evaluation. Structural complexity also reduces the reliability of function-purpose explanations, variable-role interpretation, and edge-case reasoning. It therefore affects both behavioral and semantic code comprehension.

## D. Statistical Significance of Complexity Effects

To estimate the joint association between structural complexity and automatic prediction correctness, we fitted the multivariate logistic regression model defined in Equation 3. The four metrics were entered simultaneously; therefore, each coefficient represents its association with correctness while controlling for the other included metrics. Halstead volume was scaled by 100, so its coefficient and odds ratio correspond to a 100-unit increase.

As shown in Table XI, all coefficients are negative and statistically significant for both models. Thus, increases in cyclomatic complexity, nesting depth, branching factor, and Halstead volume are associated with lower odds of a correct prediction. For DeepSeek-Coder-V2, a one-unit increase in CC is associated with an odds ratio of 0.66, while the corresponding odds ratios for ND and BF are 0.71 and 0.68. A 100-unit increase in HV is associated with an odds ratio of 0.76. CodeLlama-7b-Instruct-hf follows the same pattern, with odds ratios ranging from 0.70 to 0.79.

These results are consistent with the descriptive differences in Table V and the negative Spearman correlations in Table VI. However, the confidence intervals overlap substantially, and the correlation coefficients span a narrow range. Therefore, the results support broadly comparable negative associations across all four metrics rather than identifying one metric as a statistically stronger predictor.

For the manual task, a chi-square test of independence was used to evaluate the association between complexity band and correctness. Table XII reports the results.

TABLE X  
COMPARISON OF AUTOMATIC AND MANUAL COMPREHENSION ACCURACY ACROSS COMPLEXITY BANDS.
<table><tr><td>Model</td><td>Task</td><td>Low</td><td>Medium</td><td>High</td><td>Low-High Drop</td></tr><tr><td>DeepSeek-Coder-V2</td><td>Automatic</td><td>93.52%</td><td>80.00%</td><td>52.78%</td><td>40.74 pp</td></tr><tr><td rowspan="2">CodeLlama-7b-Instruct-hf</td><td rowspan="2">Manual Automatic</td><td>100.00%</td><td>90.00%</td><td>75.00%</td><td>25.00 pp</td></tr><tr><td>87.04%</td><td>69.17%</td><td>47.22%</td><td>39.82 pp</td></tr><tr><td></td><td>Manual</td><td>90.00%</td><td>80.00%</td><td>60.00%</td><td>30.00 pp</td></tr></table>

TABLE XI  
MULTIVARIATE LOGISTIC REGRESSION RESULTS FOR AUTOMATIC PREDICTION CORRECTNESS.
<table><tr><td>Model</td><td>Metric</td><td>Coef.</td><td>OR</td><td>95% CI</td><td>p-value</td></tr><tr><td rowspan="4">DeepSeek-Coder-V2</td><td>CC</td><td>-0.41</td><td>0.66</td><td>[0.57, 0.76]</td><td> $\overline { { < 0 . 0 0 1 } }$ </td></tr><tr><td>ND</td><td>-0.34</td><td>0.71</td><td>[0.60, 0.84]</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>BF</td><td>-0.38</td><td>0.68</td><td>[0.58, 0.79]</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>HV/100</td><td>-0.27</td><td>0.76</td><td>[0.66, 0.88]</td><td>0.002</td></tr><tr><td rowspan="4">CodeLlama-7b-Instruct-hf</td><td>CC</td><td>-0.36</td><td>0.70</td><td>[0.61, 0.80]</td><td>&lt; 0.001</td></tr><tr><td>ND</td><td>-0.31</td><td>0.73</td><td>[0.62, 0.86]</td><td>&lt; 0.001</td></tr><tr><td>BF</td><td>-0.35</td><td>0.70</td><td>[0.60, 0.82]</td><td>&lt; 0.001</td></tr><tr><td>HV/100</td><td>-0.24</td><td>0.79</td><td>[0.68, 0.91]</td><td>0.006</td></tr></table>

TABLE XII  
ASSOCIATION BETWEEN COMPLEXITY BAND AND MANUAL COMPREHENSION CORRECTNESS.
<table><tr><td>Model</td><td> $\overline { { x ^ { 2 } } }$ </td><td>df</td><td>p-value</td><td>Cramér&#x27;s V</td></tr><tr><td>DeepSeek-Coder-V2</td><td>6.15</td><td>2</td><td>0.046</td><td>0.32</td></tr><tr><td>CodeLlama-7b-Instruct-hf</td><td>5.22</td><td>2</td><td>0.074</td><td>0.29</td></tr></table>

For DeepSeek-Coder-V2, complexity band was significantly associated with manual correctness, $\chi ^ { 2 } ( 2 ) = 6 . 1 5 , p = 0 . 0 4 6$ with Cramer’s´ $V = 0 . 3 2$ . CodeLlama-7b-Instruct-hf showed the same downward accuracy pattern, but the association did not reach the conventional significance threshold, $\chi ^ { 2 } ( 2 ) =$ 5.22, $p \ = \ 0 . 0 7 4$ , with Cramer’s ´ $V ~ = ~ 0 . 2 9$ . The weaker statistical evidence should be interpreted in light of the smaller manual sample, which contains only 20 functions per complexity band.

Inter-rater agreement for the independently scored manualresponse subset was strong, with Cohen’s $\kappa ~ = ~ 0 . 8 6 .$ This supports the reliability of the manual correctness labels used in the final analysis.

The results demonstrate that benchmark composition is an important factor in interpreting LLM code-comprehension performance. HumanEval, MBPP-sanitized, and CRUXEval contain only Low- and Medium-complexity functions in the constructed benchmark and consequently produce higher model accuracy. In contrast, LeetCode and BigCodeBench-Hard contain most of the High-complexity functions and expose substantially lower reliability. This indicates that differences between benchmark scores cannot always be attributed only to model capability. They may also reflect differences in the structural-complexity distribution of the evaluated code. Reporting benchmark-level accuracy without this distribution can therefore conceal important reliability limitations.

The metric-level findings provide a complementary interpretation. Cyclomatic complexity, nesting depth, branching

## VI. DISCUSSION AND THREATS TO VALIDITY

factor, and Halstead volume were all negatively associated with prediction correctness. Although cyclomatic complexity and branching factor had numerically large associations, their confidence intervals overlapped with those of the other metrics. The results therefore do not establish a single dominant measure of LLM comprehension difficulty. Instead, the four metrics capture different but related reasoning demands: execution-path diversity, hierarchical control flow, decision density, and symbolic processing load. Their joint use provides a more complete indication of when an LLM response may require verification.

The difference between automatic and manual performance also provides insight into how models process source code. Automatic input–output prediction showed a larger Low-to-High accuracy decline than manual semantic comprehension. Exact prediction requires the model to follow a specific execution path and preserve every intermediate state until the final return value is produced. A single error in loop simulation, conditional evaluation, or variable updating leads to an incorrect answer. Manual questions may sometimes be answered through recognition of the function’s overall structure or algorithmic pattern. Nevertheless, their accuracy also declined with complexity, confirming that the effect extends beyond exact-match scoring to semantic understanding.

These findings have practical implications for LLM-assisted software engineering. Structural metrics can be computed before an LLM is asked to explain, review, or predict the behavior of a function. When the code contains deep nesting, many decision points, or high symbolic complexity, a development tool could request additional test execution, generate a warning, or recommend human review. Complexity-aware verification may therefore provide a lightweight mechanism for identifying situations in which model output should not be accepted without further checking.

Several threats limit the generalizability of the findings. The benchmark contains only Python functions, and languages with different typing, memory, or control-flow characteristics may produce different results. The evaluation includes two codeoriented models under zero-shot, deterministic prompting, so the findings may change with larger models, few-shot prompting, tool use, or explicit reasoning instructions. The Low-, Medium-, and High-complexity thresholds provide a practical operationalization of structural difficulty but do not capture semantic factors such as unfamiliar APIs, recursion, or domain-specific logic. Widely used benchmarks may also occur in model training data. Finally, the manual evaluation contains 60 functions and relies on human judgement, although balanced sampling, a fixed binary rubric, and strong inter-rater agreement reduce this risk.

## VII. CONCLUSION

This paper presented a complexity-aware method for evaluating LLM code comprehension using automatic input– output prediction and manual semantic comprehension. Python functions were annotated with cyclomatic complexity, nesting depth, branching factor, and Halstead volume and grouped into Low-, Medium-, and High-complexity bands. This design provides a more diagnostic view of model reliability than aggregate benchmark accuracy alone.

The results show that both DeepSeek-Coder-V2 and CodeLlama-7b-Instruct-hf become substantially less accurate as structural complexity increases. DeepSeek-Coder-V2 decreased from 93.52% accuracy on Low-complexity functions to 52.78% on High-complexity functions, while CodeLlama-7b-Instruct-hf decreased from 87.04% to 47.22%. Incorrect predictions were associated with higher values of all four complexity metrics, and both correlation and multivariate logisticregression analyses confirmed negative associations between structural complexity and correctness. Because the effect estimates and confidence intervals overlapped, the metrics are best interpreted as complementary indicators of comprehension risk rather than as a strict ranking of predictors.

Manual semantic comprehension exhibited the same degradation pattern. Both models performed best on purpose questions and worst on edge-case questions, while accuracy also declined from Low- to High-complexity functions. These findings indicate that structural complexity affects both exact execution reasoning and higher-level semantic understanding.

Complexity distribution should be reported alongside aggregate accuracy when evaluating LLM code comprehension. In practical software engineering workflows, model explanations, predictions, and review suggestions should receive additional verification when applied to code with deep nesting, many execution paths, dense branching, or high symbolic complexity. Future work should extend this evaluation to other programming languages, larger model families, repositorylevel code, and tool-assisted comprehension settings.

## REFERENCES

[1] A. Fan, B. Gokkaya, M. Harman, M. Lyubarskiy, S. Sengupta, S. Yoo, and J. M. Zhang, “Large language models for software engineering: Survey and open problems,” in 2023 IEEE/ACM International Conference on Software Engineering: Future of Software Engineering (ICSE-FoSE). IEEE, 2023, pp. 31–53.

[2] Q. Zhang, C. Fang, Y. Xie, Y. Zhang, S. Yu, W. Sun, Y. Yang, and Z. Chen, “A survey on large language models for software engineering,” Science China Information Sciences, vol. 69, no. 4, p. 141102, 2026.

[3] Z. Zheng, K. Ning, Y. Wang, J. Zhang, D. Zheng, M. Ye, and J. Chen, “A survey of large language models for code: Evolution, benchmarking, and future trends,” arXiv preprint arXiv:2311.10372, 2023.

[4] S. Lu, D. Guo, S. Ren, J. Huang, A. Svyatkovskiy, A. Blanco, C. Clement, D. Drain, D. Jiang, D. Tang et al., “Codexglue: A machine learning benchmark dataset for code understanding and generation,” arXiv preprint arXiv:2102.04664, 2021.

[5] Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin, T. Liu, D. Jiang et al., “Codebert: A pre-trained model for programming and natural languages,” in Findings of the association for computational linguistics: EMNLP 2020, 2020, pp. 1536–1547.

[6] S. Kabir, D. N. Udo-Imeh, B. Kou, and T. Zhang, “Is stack overflow obsolete? an empirical study of the characteristics of chatgpt answers to stack overflow questions,” in Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, 2024, pp. 1–17.

[7] W. Sun, Y. Miao, Y. Li, H. Zhang, C. Fang, Y. Liu, G. Deng, Y. Liu, and Z. Chen, “Source code summarization in the era of large language models,” in 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). IEEE, 2025, pp. 1882–1894.

[8] S. Ramesh, J. Bose, H. Singh, A. Raghavan, S. R. Chowdhury, G. Sridhara, N. Saini, and R. Britto, “Automated code review using large language models at ericsson: An experience report,” in 2025 IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE, 2025, pp. 602–607.

[9] T. J. McCabe, “A complexity measure,” IEEE Transactions on software Engineering, no. 4, pp. 308–320, 1976.

[10] M. Munoz Bar˜ on, M. Wyrich, and S. Wagner, “An empirical validation´ of cognitive complexity as a measure of source code understandability,” in Proceedings of the 14th ACM/IEEE international symposium on empirical software engineering and measurement (ESEM), 2020, pp. 1–12.

[11] A. Gu, B. Roziere, H. Leather, A. Solar-Lezama, G. Synnaeve, and S. I.\` Wang, “Cruxeval: A benchmark for code reasoning, understanding and execution,” arXiv preprint arXiv:2401.03065, 2024.

[12] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman et al., “Evaluating large language models trained on code,” arXiv preprint arXiv:2107.03374, 2021.

[13] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le et al., “Program synthesis with large language models,” arXiv preprint arXiv:2108.07732, 2021.

[14] D. Hendrycks, S. Basart, S. Kadavath, M. Mazeika, A. Arora, E. Guo, C. Burns, S. Puranik, H. He, D. Song et al., “Measuring coding challenge competence with apps,” arXiv preprint arXiv:2105.09938, 2021.

[15] Y. Li, D. Choi, J. Chung, N. Kushman, J. Schrittwieser, R. Leblond, T. Eccles, J. Keeling, F. Gimeno, A. Dal Lago et al., “Competitionlevel code generation with alphacode,” Science, vol. 378, no. 6624, pp. 1092–1097, 2022.

[16] T. Y. Zhuo, M. C. Vu, J. Chim, H. Hu, W. Yu, R. Widyasari, I. N. B. Yusuf, H. Zhan, J. He, I. Paul et al., “Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 66 602–66 656.

[17] A. M. Esfahani, N. Kahani, and S. A. Ajila, “Understanding defects in generated codes by language models,” in 2024 34th International Conference on Collaborative Advances in Software and COmputiNg (CASCON). IEEE, 2024, pp. 1–10.

[18] A. Mohammadi Esfahani, N. Kahani, and S. A. Ajila, “Prompt optimization for llm code generation via reinforcement learning,” in International Symposium on Search Based Software Engineering. Springer, 2026, pp. 34–48.

[19] M. Sepidband, H. Taherkhani, S. Wang, and H. Hemmati, “Enhancing llm-based code generation with complexity metrics: A feedback-driven approach,” in 2025 IEEE 49th Annual Computers, Software, and Applications Conference (COMPSAC). IEEE, 2025, pp. 1416–1426.

[20] C. Liu, S. Dylan Zhang, A. R. Ibrahimzada, and R. Jabbarvand, “Codemind: A framework to challenge large language models for code reasoning,” arXiv e-prints, pp. arXiv–2402, 2024.

[21] E. Dinella, S. Chandra, and P. Maniatis, “Crqbench: A benchmark of code reasoning questions,” arXiv preprint arXiv:2408.08453, 2024.

[22] N. Jain, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica, “Livecodebench: Holistic and contamination free evaluation of large language models for code,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 58 791– 58 831.

[23] T. Liu, C. Xu, and J. McAuley, “Repobench: Benchmarking repositorylevel code auto-completion systems,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 47 832–47 850.

[24] S. Rando, L. Romani, A. Sampieri, L. Franco, J. Yang, Y. Kyuragi, F. Galasso, and T. Hashimoto, “Longcodebench: Evaluating coding llms at 1m context windows,” arXiv preprint arXiv:2505.07897, 2025.

[25] C. Liu, A. Ghazanfari, Y. Chen, and R. Jabbarvand, “Evaluating code reasoning abilities of large language models under real-world settings,” arXiv preprint arXiv:2512.14917, 2025.

[26] F. Machtle, J.-N. Serr, N. Loose, and T. Eisenbarth, “Beyond accuracy:¨ Characterizing code comprehension capabilities in (large) language models,” in Proceedings of the 7th IEEE/ACM International Workshop on Deep Learning for Testing and Testing for Deep Learning, 2026, pp. 22–26.

[27] C. Xie, Y. Shi, X. Gu, and B. Shen, “Rethinking code complexity through the lens of large language models,” arXiv preprint arXiv:2602.07882, 2026.

[28] J. Gao, Y. Peng, Q. Qiao, C. Zhou, Y. Zhou, S. Zhang, S. Weng, Z. Xing, and X. Ren, “Core: A fine-grained code reasoning benchmark beyond output prediction,” arXiv preprint arXiv:2604.25399, 2026.

[29] T. Hericko and B.ˇ Sumak, “Exploring maintainability index variants<sup>ˇ</sup> for software maintainability measurement in object-oriented systems,” Applied Sciences, vol. 13, no. 5, p. 2972, 2023.

[30] G. A. Campbell, “Cognitive complexity: An overview and evaluation,” in Proceedings of the 2018 international conference on technical debt, 2018, pp. 57–58.

[31] Y. Tashtoush, N. Abu-El-Rub, O. Darwish, S. Al-Eidi, D. Darweesh, and O. Karajeh, “A notional understanding of the relationship between code readability and software complexity,” Information, vol. 14, no. 2, p. 81, 2023.

[32] Y. Xia, W. Shen, Y. Wang, J. K. Liu, H. Sun, S. Wu, J. Hu, and X. Xu, “Leetcodedataset: A temporal dataset for robust evaluation and efficient training of code llms,” arXiv preprint arXiv:2504.14655, 2025.

[33] N. Peitek, S. Apel, C. Parnin, A. Brechmann, and J. Siegmund, “Program comprehension and code complexity metrics: An fmri study,” in 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE). IEEE, 2021, pp. 524–536.

[34] M. Munoz Bar ˜ on, M. Wyrich, and S. Wagner, “An empirical validation´ of cognitive complexity as a measure of source code understandability,” in Proceedings of the 14th ACM/IEEE international symposium on empirical software engineering and measurement (ESEM), 2020, pp. 1–12.

[35] Q. Zhu, D. Guo, Z. Shao, D. Yang, P. Wang, R. Xu, Y. Wu, Y. Li, H. Gao, S. Ma et al., “Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence,” arXiv preprint arXiv:2406.11931, 2024.

[36] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, R. Sauvestre, T. Remez et al., “Code llama: Open foundation models for code,” arXiv preprint arXiv:2308.12950, 2023.