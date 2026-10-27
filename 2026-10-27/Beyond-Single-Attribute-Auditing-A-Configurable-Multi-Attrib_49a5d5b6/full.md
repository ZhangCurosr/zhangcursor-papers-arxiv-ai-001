# Beyond Single-Attribute Auditing: A Configurable Multi-Attribute Benchmark for Intersectional Fairness in LLMs

Zheng Zhang<sup>1,\*</sup> Yifan Wu<sup>1,\*</sup> Yuxiang Zhou<sup>1,\*</sup> Qi Liu<sup>1,†</sup> Qingyang Mao<sup>1</sup> Ning Li<sup>1</sup> Weibo Gao<sup>1</sup> Qiyuan Shi<sup>1</sup> Zhenya Huang<sup>1</sup> Baosheng Yu<sup>2</sup>

<sup>1</sup>State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China <sup>2</sup>Nanyang Technological University

{zhangzheng,wuyifan060830,zyx12120408}@mail.ustc.edu.cn qiliuql@ustc.edu.cn {maoqy0503,ningli03,weibogao}@mail.ustc.edu.cn qshi24@jh.edu huangzhy@ustc.edu.cn baosheng.yu@ntu.edu.sg

△ Warning: This paper contains contents that may be offensive or harmful.

## Abstract

Large language models (LLMs) are increasingly evaluated for fairness, yet most existing benchmarks audit one sensitive attribute at a time. This single-attribute paradigm can miss intersectional harms, where multiple attributes interact non-additively and produce failures that are invisible under marginal evaluation. We introduce ComFair<sup>1</sup>, a configurable benchmark that converts existing fairness data into 479,066 controlled Dual- and Multi-attribute instances and pairs every augmented instance with its task-equivalent source. Across Recognition, Selection, Continuation, and Conversation, this pairing supports a unified audit of performance or bias-score shifts, how often added attributes introduce new errors or mask existing ones, and model-ranking stability. Experiments on ten LLMs show taskand subgroup-dependent sensitivity: discriminative tasks expose both transition directions even when aggregate changes are modest, generative tasks exhibit larger and directionally asymmetric shifts, and rankings vary with attribute composition. ComFair therefore provides a scalable controlled sensitivity audit— rather than a model of the natural joint distribution of identities—for testing the robustness of fairness-benchmark conclusions beyond singleattribute settings.

## 1 Introduction

Large Language Models (LLMs) are increasingly deployed in high-stakes social applications, including hiring and resume screening, educational assessment, content generation, and decision support (Naveed et al., 2025; Chang et al., 2024; Zhao et al., 2026). In such settings, model outputs may affect individuals’ access to opportunities and resources, making fairness evaluation a critical step before deployment. Fairness benchmarks play a central role in this process: they shape how model biases are measured, how models are compared, and how deployment risks are assessed (Gallegos et al., 2024; Li et al., 2023). Therefore, the design of fairness benchmarks directly determines the reliability and practical relevance of fairness conclusions.

![](images/801c2818a3fcae53fbf21c39c140f1094eacd2d9d6278ec7eec6a2b8b36520bf.jpg)  
Figure 1: Overview of single-attribute and intersectional fairness auditing.

Most existing LLM fairness benchmarks evaluate model behavior with respect to a single sensitive attribute, such as gender, race, or age, and analyze fairness outcomes marginally along that attribute (Wang et al., 2025). Intersectional fairness research has established that this marginal view can differ from evaluations of overlapping groups (Foulds et al., 2020; Molina and Loiseau, 2022), and recent LLM studies further demonstrate these differences in stereotype and coreference settings (Ma et al., 2023; Khan et al., 2025).

The unresolved benchmarking question is more specific: does a conclusion obtained from an existing single-attribute item remain stable after another sensitive attribute is introduced while its label, event, and task structure are heldfixed? Comparisons across separately collected datasets cannot isolate this sensitivity because identity composition changes together with wording, domain, or task. A task-preserving pair instead exposes what an aggregate score conceals: the same average change can arise from different mixtures of newly introduced failures and apparently corrected failures. Because discriminative decisions and open-ended generations react differently to added identity cues, this question must also be examined across task paradigms rather than through one aggregate fairness score.

We address this question with ComFair, a configurable, controlled sensitivity audit. Given a source instance x, ComFair minimally introduces additional attribute values to create a taskequivalent counterpart x<sup>′</sup> and retains the link (x, x<sup>′</sup>). It supports Recognition, Selection, Continuation, and Conversation and organizes the evaluation around three questions: RQ1, how do measured outcomes change across tasks and subgroup values; RQ2, how often does augmentation induce Correct→Wrong aggravation versus Wrong→Correct masking; and RQ3, how stable are model rankings from Single to Dual and Multi evaluation? The objective is not to reproduce the natural joint distribution of real-world identities, but to test benchmark conclusions under a controlled attribute intervention.

We instantiate ComFair with 479,066 augmented instances—about 40× the scale of CEB, one of the most comprehensive fairness benchmarks to date (Wang et al., 2025)(Table 1). We evaluate 10 open- and closed-source LLMs. The results reveal task-dependent sensitivity. Recognition and Selection expose both aggravation and masking, including transitions hidden by near-ceiling aggregate performance; generative tasks show larger and more directionally asymmetric score changes; and model rankings vary across tasks and composition depths. These findings characterize when singleattribute benchmark conclusions are fragile, rather than asserting a universal direction of intersectional harm.

Our contributions are summarized as follows:

• Paired benchmark construction. We construct ComFair, linking existing single-attribute instances to 479,066 controlled, task-preserving Dual- and Multi-attribute counterparts across four LLM fairness tasks.

• Evaluation protocol. We propose a unified protocol for intersectional fairness auditing that quantifies how adding sensitive attributes changes model behavior, including performance shifts, aggravation, masking, and ranking stability.

• Cross-task evidence. Experiments on 10 LLMs show that intersectional sensitivity is task- and subgroup-dependent, distinguishing paired transition patterns and ranking changes that a single task or aggregate score cannot reveal.

## 2 Related Work

## 2.1 Fairness in LLMs

As LLMs are increasingly deployed in real-world applications, fairness concerns have become increasingly salient (Jiang et al., 2024; Shang et al., 2024). Early work showed that distributional representations can encode societal biases from training data (Caliskan et al., 2017), and recent surveys organize LLM fairness risks including toxicity, stereotyping, preference bias, and demographic disparities (Chu et al., 2024; Gallegos et al., 2024). Existing benchmarks examine these risks from complementary perspectives: TrustGPT evaluates toxicity across demographic groups (Huang et al., 2023), DecodingTrust and TrustLLM assess broader trustworthiness and fairness-related behaviors (Wang et al., 2023; Huang et al., 2024), and CEB covers four fairness task paradigms (Wang et al., 2025). These benchmarks usually report results for predefined attributes or groups. ComFair asks a complementary question: whether the conclusion for a source item remains stable after a controlled, taskpreserving identity attribute is introduced.

Beyond fairness-specific benchmarks, recent work studies LLM evaluation and deployment across capability, reliability, and educational settings. More broadly, reliable evaluation requires aligning benchmark protocols, metrics, and data with the capability claims they are intended to support (Zhang et al., 2026a). Adaptive testing has been used to measure LLM cognitive ability (Zhuang et al., 2023), while controllable contamination detection targets the reliability of evaluation results (Zhang et al., 2026b); related work also examines multimodal matching, efficient visual reasoning, and explainable decision-making systems (Zhang et al., 2023a; Wang et al., 2026b,a). In education, cognitive diagnosis and generativeagent-based learner-response modeling illustrate how model behavior is evaluated in applicationspecific contexts (Gao et al., 2023, 2025). These studies motivate evaluation protocols that remain sensitive to task structure and deployment context, which is central to our paired design.

<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">#Inst.</td><td colspan="4">Task Coverage</td><td colspan="3">Sensitive-Attribute Support</td></tr><tr><td>Cont.</td><td>Conv.</td><td>Sel.</td><td>Rec.</td><td>Single</td><td>Dual</td><td>Multi</td></tr><tr><td>BBQ (Parrish et al., 2022)</td><td>58,492</td><td></td><td></td><td>√</td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>CrowS-Pairs (Nangia et al., 2020)</td><td>1,508</td><td></td><td></td><td>√</td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>StereoSet (Nadeem et al., 2021)</td><td>16,995</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td></td><td></td></tr><tr><td>WinoGender (Rudinger et al., 2018)</td><td>720</td><td></td><td></td><td>√</td><td></td><td>7</td><td></td><td></td></tr><tr><td>WinoBias (Zhao et al., 2018)</td><td>3,160</td><td></td><td></td><td>√</td><td></td><td>√</td><td></td><td></td></tr><tr><td>BOLD (Dhamala et al., 2021)</td><td>23,679</td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td>RealToxicityPrompts (Gehman et al., 2020)</td><td>100,000</td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td>Adult (Becker and Kohavi, 1996)</td><td>20,000</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td>German Credit (Hofmann, 1994)</td><td>1,000</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td>Jigsaw Toxic Comment (Borkan et al., 150,000 2019)</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td>WinoIdentity (Khan et al., 2025)</td><td>245,700</td><td></td><td></td><td>√</td><td></td><td>√</td><td></td><td></td></tr><tr><td>Intersectional Stereotype Dataset (Ma et al., 2023)</td><td>480</td><td>√</td><td>√</td><td></td><td>一</td><td>√</td><td></td><td></td></tr><tr><td>CEB (Wang et al., 2025)</td><td>11,004</td><td>√</td><td>L</td><td>√</td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>Ours: ComFair</td><td>479,066</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td></tr></table>

Table 1: Coverage of representative LLM fairness benchmarks. Our benchmark unifies fairness evaluation across four task types(Continuation, Conversation, Selection, Recognition) and supports single, dual, multi-way sensitive-attribute assessment.

Fairness Mitigation and Safety. Another line of work focuses on mitigating unfairness when sensitive information is incomplete or difficult to obtain, including fair user modeling (Zhang et al., 2023b), confidence-based randomization with incomplete sensitive labels (Hu et al., 2024b), and multi-objective formulations of universal fairness (Hu et al., 2024a). These approaches aim to improve or constrain model behavior, whereas ComFair evaluates whether fairness conclusions themselves change after controlled attribute composition. Closely related social-risk research studies controllable safeguards against attacks (Hu et al., 2026) and the dynamics of misinformation acceptance and deception in LLM-agent systems (Liu et al., 2024, 2025). Together, these directions emphasize that reliable LLM assessment must account for both behavioral risks and the conditions under which those risks are measured.

## 2.2 Intersectional Fairness

Intersectionality highlights that discrimination can emerge from interacting sensitive attributes rather than from any one attribute in isolation (Crenshaw, 2013; Foulds et al., 2020; Gohar and Cheng, 2023). Prior NLP work documents intersectional effects in representations (Tan and Celis, 2019). More recently, Ma et al. (2023) introduce an intersectional stereotype dataset and analyze stereotype propagation in LLMs, while Khan et al. (2025) extend coreference evaluation to intersectional identities using confidence disparities. These studies establish that intersectional and single-attribute conclusions can differ. ComFair builds on this evidence by linking source items to configurable, taskpreserving counterparts and applying one paired protocol across two discriminative and two generative tasks. This design is related to metamorphic evaluation under controlled input transformations (Reddy et al., 2025), but operates at benchmark scale to test the stability of existing fairness conclusions.

## 3 Preliminaries

## 3.1 Fairness Evaluation

We consider four common task paradigms for LLM fairness evaluation: Continuation (CONT), Conversation (CONV), Selection (SEL), and Recognition (REC). We denote the task set as T = {CONT, CONV, SEL, REC}. These tasks cover both generative settings, where bias may appear in free-form model outputs, and discriminative settings, where bias is evaluated through classification or choice-based decisions.

Task formalization. For each task $\tau \in \ \tau$ we consider an evaluation dataset $\begin{array} { r l } { \mathcal { D } ^ { \tau } } & { { } = } \end{array}$ $\{ ( x _ { i } ^ { \tau } , y _ { i } ^ { \tau } ) \} _ { i = 1 } ^ { n _ { \tau } }$ , where $x _ { i } ^ { \tau } \in \mathcal { X } ^ { \tau }$ is the task input and $y _ { i } ^ { \tau }$ denotes the task-specific annotation, label, or evaluation metadata. Let $A = \{ A _ { 1 } , \ldots , A _ { m } \}$ denote the set of sensitive attributes (e.g., gender, age, race), where each attribute $A _ { j }$ takes values in $\nu _ { j }$ . In standard single-attribute benchmarks, each instance is associated with a designated sensitive attribute $A _ { \mathrm { b a s e } } \in { \mathcal { A } }$ and a group value $a _ { \mathrm { b a s e } } \in \mathcal { V } _ { \mathrm { b a s e } } .$

Given a model $f ,$ we denote its task-specific output by $f _ { \tau } ( x )$ and evaluate it using a task-dependent scoring function $s _ { \tau } ( f _ { \tau } ( x ) , y )$ . For discriminative tasks, $s _ { \tau }$ is typically instantiated as accuracy or F1, where larger values indicate better performance. For generative tasks, $s _ { \tau }$ is typically instantiated as a bias, toxicity, or stereotype score computed by a human or automatic evaluator, where smaller values indicate fairer generations. When cross-task comparison is required, we convert raw task scores into a unified fairness-oriented score so that larger values consistently indicate fairer behavior. This task-specific score serves as the primitive evaluation signal for subsequent aggregation and robustness analyses.

We summarize the input/output spaces and typical scoring rules below.

• Recognition (REC). Input $x \in \mathcal { X } ^ { \mathrm { R E C } }$ is a text snippet, and the annotation $y \in \mathcal { y } ^ { \mathrm { R E C } } =$ $\{ 0 , 1 \}$ indicates whether the snippet contains bias, stereotypes, or harmful expressions. The model outputs $f _ { \mathrm { R E C } } ( x ) \in \{ 0 , 1 \}$ or a score in [0, 1]. The scoring function s<sub>REC</sub> is typically accuracy or F1.

• Selection (SEL). Input $x \in \mathcal { X } ^ { \mathrm { S E L } }$ is a candidate set $\boldsymbol { x } = \{ c _ { 1 } , \ldots , c _ { K } \}$ , and the annotation $y ~ \in ~ { \mathcal { y } } ^ { \mathrm { S E L } } ~ = ~ \{ 1 , \dots , K \}$ indicates the less biased option. The model outputs $f _ { \mathrm { S E L } } ( \boldsymbol { x } ) \in \{ 1 , \ldots , K \}$ . The scoring function $s _ { \mathrm { S E I } }$ <sub>L</sub> is selection accuracy.

• Continuation (CONT). Input $x \in \mathcal { X } ^ { \mathrm { C o N T } }$ is a prefix or context, and the model generates a continuation $f _ { \mathrm { C o N T } } ( x ) \in \mathcal { V } ^ { \mathrm { C o N T } }$ as free-form text. The scoring function $s _ { \mathrm { C O N T } }$ measures bias-related properties of the generated continuation, using human annotation or an automatic evaluator such as an LLM-based judge.

• Conversation (CONV). Input $x \in \mathcal { X } ^ { \mathrm { C o N v } }$ is a dialogue history, and the model outputs a response $f _ { \mathrm { C o N v } } ( x ) \in \mathcal { V } ^ { \mathrm { C o N v } }$ as free-form text. The scoring function $s _ { \mathrm { C O N V } }$ measures fairnessrelated properties of the response, such as stereotypical, biased, or toxic content, using human annotation or an automatic evaluator.

## 3.2 Problem Definition

Existing fairness benchmarks typically evaluate model behavior with respect to a single sensitive attribute. Formally, each example is associated with a designated base attribute $A _ { \mathrm { b a s e } } \in { \mathcal { A } }$ and its value $a _ { \mathrm { b a s e } } \in \mathcal { V } _ { \mathrm { b a s e } }$ . However, real-world social identities often involve multiple attributes simultaneously, making it necessary to evaluate intersectional behavior.

Let $\mathbf { a } = ( a _ { 1 } , \ldots , a _ { m } ) \in \mathcal { V } _ { 1 } \times \cdots \times \mathcal { V } _ { m }$ denote an intersectional assignment over the attribute set $\mathcal { A } = \{ A _ { 1 } , . . . , A _ { m } \}$ . Our goal is to construct augmented instances that incorporate additional sensitive attributes while preserving the original task semantics. Specifically, given a base instance associated with $( A _ { \mathrm { b a s e } } , a _ { \mathrm { b a s e } } )$ , we define a target attribute set $\mathcal { A } _ { \mathrm { t a r } } \supseteq \{ A _ { \mathrm { b a s e } } \}$ and a target assignment $\mathbf { a } _ { \mathrm { t a r } }$ over $\boldsymbol { A } _ { \mathrm { t a r } }$ . We enforce consistency with the base attribute: the coordinate of $\mathbf { a } _ { \mathrm { t a r } }$ corresponding to $A _ { \mathrm { b a s e } }$ equals $a _ { \mathrm { b a s e } }$

For each task $\tau \in \tau$ , we define a task-aware augmentation operator:

$$
g _ { \tau } ( x , ( A _ { \mathrm { b a s e } } , a _ { \mathrm { b a s e } } )  ( A _ { \mathrm { t a r } } , \mathbf { a } _ { \mathrm { t a r } } ) ) = x ^ { \prime } ,\tag{1}
$$

which minimally edits the input x to encode $\mathbf { a } _ { \mathrm { t a r } }$ while preserving the underlying semantics and the task structure required by τ .

## 4 ComFair

In this section, we present a configurable and extensible benchmark framework, ComFair, for intersectional fairness evaluation of LLMs. ComFair supports arbitrary combinations of multiple sensitive attributes and spans four major fairness evaluation paradigms (REC, SEL, CONT, and CONV). The framework proceeds in two steps: we first expand existing single-attribute benchmarks to multiattribute settings via a task-aware augmentation procedure; we then introduce evaluation metrics that quantify how stacking additional sensitive attributes perturbs model behavior.

## 4.1 Benchmark Construction

ComFair constructs paired evaluation instances $( x , x ^ { \prime } )$ from existing single-attribute fairness benchmarks, where x is the original input and $x ^ { \prime }$ is its intersectional counterpart with additional sensitiveattribute values. We first standardize source datasets into four task paradigms—REC, SEL, CONT, and CONV—and then apply task-aware augmentation to inject additional attribute values into target entity mentions. Detailed task-specific construction procedures are provided in Appendix A.1.

We enforce two principles during augmentation. Minimal Perturbation restricts edits to sensitiveattribute modifiers while preserving the original event description, decision logic, evidence, and task formulation. Semantic Non-conflict requires the augmented instance to be natural, plausible, and internally consistent; instances with commonsense violations, logical contradictions, or incompatible descriptors are filtered during quality control. For example, “female employee” may be augmented to “older female employee”, but implausible descriptions such as “child female employee” in an employment context are rejected. We provide two examples in Appendix A.2.

## 4.2 Quality Control

To ensure reliable intersectional augmentation, we apply a two-stage quality-control pipeline. First, GPT-5.2 verifies whether each augmented pair $( x , x ^ { \prime } )$ satisfies Minimal Perturbation and Semantic Non-conflict, and rejects invalid instances with brief rationales. Second, we randomly sample 10% of the retained instances for expert auditing. Two domain experts independently annotate principle violations, including semantic drift, attribute inconsistency, unnatural phrasing, and cue leakage. Disagreements are resolved by a senior expert. The audit yields Cohen’s $\kappa ~ = ~ 0 . 8 9 7$ and a humanconfirmed pass rate of 93.5%, supporting the reliability of the construction pipeline.

## 4.3 Evaluation Protocol

ComFair evaluates models by comparing an original single-attribute pool $D _ { \tau }$ with its intersectionally augmented counterpart $D _ { \tau } ^ { \prime }$ for each task $\tau \in \tau$ Each augmented instance $x ^ { \prime }$ is paired with its original input x, yielding paired data $P _ { \tau } = \{ ( x , x ^ { \prime } ) \}$ This paired structure allows us to assess both postaugmentation fairness and how adding sensitive attributes changes model behavior.

Post-augmentation performance. For discriminative tasks, i.e., REC and SEL, we report a taskspecific metric $M _ { \tau } ( f ; D _ { \tau } ^ { \prime } )$ , such as F1 or accuracy, where larger values indicate better performance. For generative tasks, i.e., CONT and CONV, we use an evaluator to assign a scalar bias score $b _ { \tau } ( f _ { \tau } ( x ) ) \in [ 0 , 1 0 0 ]$ to each model output, where lower values indicate fairer generations. We report the mean bias score:

$$
\begin{array} { r l } & { B _ { \tau } ( f ; D _ { \tau } ^ { \prime } ) = \mathbb { E } _ { x \sim D _ { \tau } ^ { \prime } } \left[ b _ { \tau } ( f _ { \tau } ( x ) ) \right] , } \\ & { \qquad \tau \in \{ \mathrm { C O N T } , \mathrm { C O N V } \} . } \end{array}\tag{2}
$$

Thus, higher is better for $M _ { \tau }$ , whereas lower is better for $B _ { \tau }$

Paired intersectional effects. Beyond aggregate performance on $D _ { \tau } ^ { \prime }$ , we measure how augmentation changes model behavior on paired instances. For discriminative tasks, let $\begin{array} { r l r } { \hat { y } } & { { } = } & { f _ { \tau } ( x ) } \end{array}$ and $\hat { y } ^ { \prime } = f _ { \tau } ( x ^ { \prime } )$ denote model predictions. The taskpreserving pair shares a label $y$ by construction. We compute the following quantities over aligned pairs with valid predictions:

$$
\begin{array} { r l r } & { } & { \Delta _ { \tau } ( f ) = { \cal M } _ { \tau } ( f ; { \cal D } _ { \tau } ^ { \prime } ) - { \cal M } _ { \tau } ( f ; { \cal D } _ { \tau } ) , } \\ & { } & { \mathrm { A g g r } _ { \tau } ( f ) = \underset { ( x , x ^ { \prime } ) \sim { \cal P } _ { \tau } } { \mathrm { P r } } \left( \hat { y } ^ { \prime } \neq y \mid \hat { y } = y \right) , } \\ & { } & { \mathrm { M a s k } _ { \tau } ( f ) = \underset { ( x , x ^ { \prime } ) \sim { \cal P } _ { \tau } } { \mathrm { P r } } \left( \hat { y } ^ { \prime } = y \mid \hat { y } \neq y \right) . } \end{array}\tag{3}
$$

Thus, Aggr divides newly introduced errors by the number of originally correct predictions, whereas Mask<sub>τ</sub> divides corrected errors by the number of originally incorrect predictions. These are conditional flip rates, not rates normalized by all paired samples. We refer to the latter as masking because improvements on augmented inputs may conceal failures already present in the original singleattribute setting.

For generative tasks, we define the pairwise biasscore change as:

$$
\delta _ { \tau } ( f ; x , x ^ { \prime } ) = b _ { \tau } ( f _ { \tau } ( x ^ { \prime } ) ) - b _ { \tau } ( f _ { \tau } ( x ) ) .\tag{4}
$$

Positive values indicate that the augmented input elicits a more biased generation. Writing $\delta = \delta _ { \tau } ( f ; x , x ^ { \prime } )$ for brevity, we then compute:

$$
\begin{array} { r l } & { \qquad \Delta _ { \tau } ( f ) = \mathbb { E } _ { ( x , x ^ { \prime } ) \sim P _ { \tau } } \left[ \delta _ { \tau } ( f ; x , x ^ { \prime } ) \right] , } \\ & { \mathrm { A g g r } _ { \tau } ( f ) = \mathbb { E } \left[ \delta \mid \delta > 0 \right] , } \\ & { \mathrm { M a s k } _ { \tau } ( f ) = \mathbb { E } \left[ - \delta \mid \delta < 0 \right] . } \end{array}\tag{5}
$$

Thus, generative Aggr and Mask<sub>τ</sub> are the conditional mean magnitudes of bias-score increases and decreases, respectively; they are not flip frequencies or averages over all pairs. We report zero when the corresponding conditioning set is empty.

![](images/38b3def20cec12f82297e1a1f700d9d1308cc3bbf10f546f397b4ac82a6aa6f0.jpg)  
Figure 2: ComFair’s task-preserving pipeline for constructing intersectional evaluation pairs.

Ranking stability. To assess whether intersectional augmentation changes model-selection conclusions, we compare model rankings before and after augmentation. For REC and SEL, rankings are induced by $M _ { \tau }$ , where larger values are better; for CONT and CONV, rankings are induced by − $\phantom { } \cdot B _ { \tau }$ , so that higher-ranked models consistently correspond to fairer behavior. Let $r _ { \tau } ^ { \mathrm { o r i g } }$ and $r _ { \tau } ^ { \mathrm { a u g } }$ denote the rankings on the original pool $D _ { \tau }$ and the augmented pool $D _ { \tau } ^ { \prime } .$ , respectively. We measure ranking stability using Spearman’s rank correlation coefficient:

$$
\mathrm { S R C C } _ { \tau } = \mathrm { S p e a r m a n } \left( r _ { \tau } ^ { \mathrm { o r i g } } , r _ { \tau } ^ { \mathrm { a u g } } \right) .\tag{6}
$$

where larger values indicate more stable model rankings under intersectional augmentation.

## 5 Experiments

## 5.1 Experimental Setup

Models. We evaluate ten representative LLMs spanning open- and closed-source model families, including DeepSeek-V3.2-Chat, DeepSeek-V3.2-Reasoner, Qwen3-8B, Qwen3-235B-A22B, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct, Gemini-3-Flash, GPT-5-mini, Claude-3.5-Sonnet, and Claude-3.7-Sonnet.

Data. We use ComFair as constructed in Section 4. Table 2 summarizes the number of instances across four task types and three attributecomposition settings: Single, Dual, and Multi. Detailed attribute-level breakdowns are provided in Appendix A.6.

Metrics and evaluators. We follow the evaluation protocol in Section 4.3. For REC and SEL, we report F1. For CONT and CONV, we report bias scores assigned by a GPT-5.2-based evaluator, where lower scores indicate fairer generations. To validate automatic evaluation, we conduct human evaluation on a sampled subset of CONT instances; GPT-5.2 and human annotators agree on the direction of $\Delta$ in 35/40 cases (87.5%).

<table><tr><td>Task</td><td>Single</td><td>Dual</td><td>Multi</td><td>Aug. total</td></tr><tr><td>Recognition</td><td>11,745</td><td>89,571</td><td>156,479</td><td>246,050</td></tr><tr><td>Selection</td><td>6,272</td><td>43,173</td><td>75,829</td><td>119,002</td></tr><tr><td>Continuation</td><td>2,368</td><td>14,812</td><td>24,983</td><td>39,795</td></tr><tr><td>Conversation</td><td>2,378</td><td>15,867</td><td>58,352</td><td>74,219</td></tr></table>

Table 2: ComFair instances by task and composition depth; Aug. total = Dual + Multi.

## 5.2 Intersectional Evaluation Changes Fairness Conclusions

Tables 3 and 4 address RQ1 for REC and CONT; SEL and CONV appear in Appendix A.5. In these tables, “Base” is the original score and “+Value” is the score after inserting that value into the paired source instances; parentheses report $\Delta =$ augmented − base. The bold Claude-3.5 contrast in Table 3 illustrates subgroup dependence: adding White changes F1 by −0.118, whereas adding Black changes it by +0.170 for the same baseattribute direction. Table 5 reports the matched human evaluation; agreement with GPT-5.2 on the sign of $\Delta$ in 35/40 settings supports directional consistency on this subset, but does not establish evaluator independence.

Robustness checks. Additional rebuttal experiments test evaluator dependence and a lexical-

<table><tr><td rowspan="2">Model</td><td colspan="3">Gender</td><td colspan="3">Race</td></tr><tr><td>Base</td><td>+White (∆)</td><td>+Black (∆)</td><td>Base</td><td>+Male (∆)</td><td>+Female (∆)</td></tr><tr><td>DS-V3.2-C</td><td>0.855</td><td>0.891 (+0.036)</td><td>0.891 (+0.036)</td><td>0.844</td><td>0.826(-0.018)</td><td>0.886(+0.042)</td></tr><tr><td>DS-V3.2-R</td><td>0.626</td><td>0.682(+0.056)</td><td>0.719 (+0.093)</td><td>0.637</td><td>0.632 (-0.005)</td><td>0.710 (+0.073)</td></tr><tr><td>Gem-3F</td><td>0.576</td><td>0.507 (-0.069)</td><td>0.649 (+0.073)</td><td>0.558</td><td>0.603 (+0.045)</td><td>0.596 (+0.038)</td></tr><tr><td>GPT-5-mini</td><td>0.653</td><td>0.700 (+0.047)</td><td>0.801 (+0.148)</td><td>0.686</td><td>0.636 (-0.050)</td><td>0.702 (+0.016)</td></tr><tr><td>Qwen3-8B</td><td>0.891</td><td>0.891 (+0.000)</td><td>0.891 (+0.000)</td><td>0.863</td><td>0.837 (-0.026)</td><td>0.886 (+0.023)</td></tr><tr><td>Qwen3-235B</td><td>0.851</td><td>0.849 (-0.002)</td><td>0.861 (+0.010)</td><td>0.744</td><td>0.711 (-0.033)</td><td>0.822 (+0.078)</td></tr><tr><td>LLaMA-3.1-8B</td><td>0.692</td><td>0.764(+0.072)</td><td>0.751 (+0.059)</td><td>0.749</td><td>0.745 (-0.004)</td><td>0.820 (+0.071)</td></tr><tr><td>LLaMA-3.3-70B</td><td>0.560</td><td>0.562 (+0.002)</td><td>0.626(+0.066)</td><td>0.544</td><td>0.566 (+0.022)</td><td>0.623 (+0.079)</td></tr><tr><td>Claude-3.5</td><td>0.471</td><td>0.353 (-0.118)</td><td>0.641 (+0.170)</td><td>0.664</td><td>0.667 (+0.003)</td><td>0.703 (+0.039)</td></tr><tr><td>Claude-3.7</td><td>0.501</td><td>0.387 (-0.114)</td><td>0.669 (+0.168)</td><td>0.688</td><td>0.656(-0.032)</td><td>0.723 (+0.035)</td></tr></table>

Table 3: Recognition F1 under paired Gender↔Race interventions. Boldface marks the subgroup contrast discussed in text.

<table><tr><td rowspan="2">Model</td><td colspan="3">Gender</td><td colspan="3">Race</td></tr><tr><td>Base</td><td>+White (∆)</td><td>+Black (∆)</td><td>Base</td><td>+Male (∆)</td><td>+Female (∆)</td></tr><tr><td>DS-V3.2-C</td><td>26.71</td><td>36.17(+9.46)</td><td>31.09 (+4.38)</td><td>32.20</td><td>36.67(+4.47)</td><td>32.74(+0.54)</td></tr><tr><td>DS-V3.2-R</td><td>28.84</td><td>37.83 (+8.99)</td><td>31.97(+3.13)</td><td>35.36</td><td>38.50 (+3.14)</td><td>38.77 (+3.41)</td></tr><tr><td>Gem-3F</td><td>25.95</td><td>28.81 (+2.86)</td><td>16.76(-9.19)</td><td>34.47</td><td>37.17(+2.70)</td><td>35.80(+1.33)</td></tr><tr><td>GPT-5-mini</td><td>24.54</td><td>32.06 (+7.52)</td><td>25.30 (+0.76)</td><td>30.86</td><td>37.96(+7.10)</td><td>32.94(+2.08)</td></tr><tr><td>Qwen3-8B</td><td>34.08</td><td>45.51 (+11.43)</td><td>38.62(+4.54)</td><td>37.47</td><td>41.33(+3.86)</td><td>40.49 (+3.02)</td></tr><tr><td>Qwen3-235B</td><td>29.81</td><td>34.57(+4.76)</td><td>26.06(-3.75)</td><td>34.63</td><td>36.58 (+1.95)</td><td>38.11 (+3.48)</td></tr><tr><td>LLaMA-3.1-8B</td><td>16.58</td><td>24.26(+7.68)</td><td>20.13 (+3.55)</td><td>30.61</td><td>32.29(+1.68)</td><td>30.54 (-0.07)</td></tr><tr><td>LLaMA-3.3-70B</td><td>28.96</td><td>38.12(+9.16)</td><td>32.36(+3.40)</td><td>34.54</td><td>38.25 (+3.71)</td><td>37.34(+2.80)</td></tr><tr><td>Claude-3.5</td><td>10.62</td><td>11.66(+1.04)</td><td>8.41 (-2.21)</td><td>14.63</td><td>12.17 (-2.46)</td><td>11.43 (-3.20)</td></tr><tr><td>Claude-3.7</td><td>13.25</td><td>16.20 (+2.95)</td><td>12.29 (-0.96)</td><td>21.49</td><td>30.33 (+8.84)</td><td>20.60 (-0.89)</td></tr></table>

Table 4: Continuation bias scores under GPT-5.2 judging for paired Gender↔Race interventions. Lower is fairer.

<table><tr><td rowspan="2">Model</td><td colspan="3">Gender</td><td colspan="3">Race</td></tr><tr><td>Base</td><td>+White (∆)</td><td>+Black (∆)</td><td>Base</td><td>+Male (∆)</td><td>+Female (∆)</td></tr><tr><td>DS-V3.2-C</td><td>31.42</td><td>40.69(+9.27)</td><td>35.00(+3.58)</td><td>38.90</td><td>39.79(+0.89)</td><td>40.71 (+1.81)</td></tr><tr><td>DS-V3.2-R</td><td>33.13</td><td>36.54(+3.41)</td><td>32.25 (-0.88)</td><td>36.53</td><td>46.65 (+10.12)</td><td>40.71 (+4.18)</td></tr><tr><td>Gem-3F</td><td>28.99</td><td>30.69(+1.70)</td><td>23.41 (-5.58)</td><td>40.59</td><td>41.67(+1.08)</td><td>43.14(+2.55)</td></tr><tr><td>GPT-5-mini</td><td>26.12</td><td>34.15 (+8.03)</td><td>27.75 (+1.63)</td><td>34.22</td><td>38.33 (+4.11)</td><td>36.86(+2.64)</td></tr><tr><td>Qwen3-8B</td><td>33.73</td><td>41.23 (+7.50)</td><td>37.03 (+3.30)</td><td>40.00</td><td>45.88(+5.88)</td><td>43.86(+3.86)</td></tr><tr><td>Qwen3-235B</td><td>32.13</td><td>34.14(+2.01)</td><td>26.52 (-5.61)</td><td>37.37</td><td>35.62 (-1.75)</td><td>39.57(+2.20)</td></tr><tr><td>LLaMA-3.1-8B</td><td>19.96</td><td>25.31 (+5.35)</td><td>20.87(+0.91)</td><td>33.98</td><td>35.21 (+1.23)</td><td>36.86(+2.88)</td></tr><tr><td>LLaMA-3.3-70B</td><td>18.54</td><td>30.38(+11.84)</td><td>22.90 (+4.36)</td><td>31.53</td><td>33.54(+2.01)</td><td>38.29(+6.76)</td></tr><tr><td>Claude-3.5</td><td>13.28</td><td>15.85(+2.57)</td><td>11.30(-1.98)</td><td>18.14</td><td>17.92 (-0.22)</td><td>18.57(+0.43)</td></tr><tr><td>Claude-3.7</td><td>17.69</td><td>18.92(+1.23)</td><td>15.58 (-2.11)</td><td>25.34</td><td>29.17(+3.83)</td><td>26.71 (+1.37)</td></tr></table>

Table 5: Human continuation bias scores for the same paired interventions as Table 4. Lower is fairer.

<table><tr><td>Model</td><td>+White</td><td>+Black</td><td>+Male</td><td>+Female</td></tr><tr><td>DS-V3.2-C</td><td>+4.54</td><td>-2.69</td><td>-3.59</td><td>-3.02</td></tr><tr><td>DS-V3.2-R</td><td>+6.61</td><td>-1.22</td><td>+13.32</td><td>+3.52</td></tr><tr><td>Gem-3F</td><td>+2.47</td><td>-8.19</td><td>+7.06</td><td>+1.95</td></tr><tr><td>GPT-5-mini</td><td>+3.38</td><td>-1.08</td><td>+0.61</td><td>-0.28</td></tr><tr><td>Qwen3-8B</td><td>+4.21</td><td>-1.46</td><td>+2.67</td><td>+3.02</td></tr><tr><td>Qwen3-235B</td><td>+8.70</td><td>-6.52</td><td>+0.61</td><td>-3.41</td></tr></table>

Table 6: Representative Continuation bias-score ∆ values after re-scoring Table 4 outputs with DeepSeek-V4- Pro.

perturbation alternative.

To assess evaluator dependence, DeepSeek-V4- Pro re-scored the same Continuation outputs underlying Table 4. Table 6 gives the representative model-level ∆ values reported during rebuttal. Across all ten models and four injections, DeepSeek-V4-Pro and GPT-5.2 agree on the direction of ∆ in 32/40 settings (80%); agreement is 27/32 for GPT-5.2-positive shifts and 5/8 for GPT-5.2-negative shifts.

We also use a matched placebo to test whether the Selection shifts could be caused by inserting any modifier. For the same base inputs and syntactic position, Table 7 compares White and Black with the non-demographic modifier “standing”, while holding the label and remaining content fixed. This control covers five models. In a fivemodel Selection matched placebo, mean F1 ∆ for White, Black, and the non-demographic modifier “standing” is +0.0247, +0.1020, and −0.0079; the corresponding control-adjusted shifts are +0.0326 and +0.1098. Thus, within this limited control, generic lexical perturbation alone does not explain the subgroup-specific shifts.

<table><tr><td>Model</td><td>+White</td><td></td><td>+Black +Standing</td><td>White — Standing</td><td>Black — Standing</td></tr><tr><td>DS-V3.2-C</td><td>+0.0647</td><td>+0.1741</td><td>-0.0197</td><td>+0.0844</td><td>+0.1938</td></tr><tr><td>DS-V3.2-R</td><td>+0.0344</td><td>+0.1184</td><td>-0.0059</td><td>+0.0403</td><td>+0.1243</td></tr><tr><td>GPT-5-mini</td><td>+0.0159</td><td>+0.2012</td><td>-0.0362</td><td>+0.0521</td><td>+0.2374</td></tr><tr><td>Qwen3-235B</td><td>+0.0109</td><td>+0.0155</td><td>+0.0218</td><td>-0.0109</td><td>-0.0063</td></tr><tr><td>Qwen3-8B</td><td>-0.0023</td><td>+0.0006</td><td>+0.0007</td><td>-0.0030</td><td>-0.0001</td></tr><tr><td>Mean</td><td>+0.0247</td><td>+0.1020</td><td>-0.0079</td><td>+0.0326</td><td>+0.1098</td></tr></table>

Table 7: Selection matched-placebo F1 shifts. The last two columns subtract the non-demographic “standing” shift from each demographic shift.

## 5.3 Aggravation and Masking Reveal Paired Behavioral Shifts

Aggregate score changes do not fully explain how model behavior changes under intersectional augmentation. We therefore analyze the paired effects defined in Section 4.3: bias aggravation and bias masking. Table 8 reports representative results for REC, while Appendix Tables 13–14 provide the corresponding generative-task results. The discriminative tables report conditional flip rates, whereas the generative tables report conditional mean magnitudes of positive and negative bias-score changes.

The results reveal substantial paired-level shifts even when aggregate changes are modest. In discriminative tasks, many models exhibit considerable masking, indicating that added identity cues can correct decisions that were wrong in the original single-attribute setting. In generative tasks, paired score changes are often larger in magnitude and more directionally asymmetric, suggesting that open-ended generation is highly sensitive to intersectional identity conditioning. Together, these patterns show that intersectional evaluation should report not only net score changes but also the direction of paired behavioral shifts.

<table><tr><td rowspan="3">Model</td><td colspan="4">Gender</td><td colspan="4">Race</td></tr><tr><td colspan="2">+Black</td><td colspan="2">+White</td><td colspan="2">+Female</td><td colspan="2">+Male</td></tr><tr><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td></tr><tr><td>Claude-3.5</td><td>33.59%</td><td>43.66%</td><td>43.61%</td><td>14.83%</td><td>18.97%</td><td>29.70%</td><td>22.68%</td><td>28.00%</td></tr><tr><td>Claude-3.7</td><td>33.33%</td><td>46.60%</td><td>51.72%</td><td>20.30%</td><td>17.13%</td><td>28.48%</td><td>23.56%</td><td>29.19%</td></tr><tr><td>DS-V3.2-C</td><td>0.00%</td><td>20.97%</td><td>0.00%</td><td>24.62%</td><td>0.00%</td><td>10.53%</td><td>0.00%</td><td>5.13%</td></tr><tr><td>DS-V3.2-R</td><td>31.78%</td><td>48.76%</td><td>19.84%</td><td>32.26%</td><td>9.16%</td><td>22.69%</td><td>13.28%</td><td>20.49%</td></tr><tr><td>Gem-3F</td><td>35.96%</td><td>44.12%</td><td>24.14%</td><td>13.08%</td><td>14.68%</td><td>17.52%</td><td>9.73%</td><td>16.92%</td></tr><tr><td>GPT-5-mini</td><td>17.14%</td><td>52.22%</td><td>17.48%</td><td>31.87%</td><td>14.29%</td><td>20.25%</td><td>10.87%</td><td>8.97%</td></tr><tr><td>LLaMA-3.1-8B</td><td>29.38%</td><td>52.00%</td><td>27.08%</td><td>51.33%</td><td>18.27%</td><td>48.09%</td><td>26.43%</td><td>44.37%</td></tr><tr><td>LLaMA-3.3-70B</td><td>27.56%</td><td>32.98%</td><td>21.33%</td><td>22.40%</td><td>12.23%</td><td>21.00%</td><td>16.44%</td><td>26.46%</td></tr><tr><td>Qwen3-235B</td><td>4.71%</td><td>13.56%</td><td>5.52%</td><td>20.29%</td><td>6.75%</td><td>27.59%</td><td>14.89%</td><td>20.18%</td></tr><tr><td>Qwen3-8B</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.55%</td><td>0.00%</td></tr></table>

Table 8: Recognition aggravation (Agg.) and masking (Mask.) rates under paired Gender↔Race interventions.

## 5.4 Intersectional Evaluation Alters Model Rankings

Since fairness benchmarks are often used for model comparison, we further examine whether intersectional augmentation changes model rankings. Using the SRCC metric defined in Section 4.3, we compare rankings induced by the original pool $D _ { \tau }$ and the augmented pool $D _ { \tau } ^ { \prime }$ . As shown in Figure 3, rankings often become unstable after adding sensitive attributes, indicating that model-selection conclusions based on single-attribute evaluation may not generalize to intersectional settings. In several settings, multi-attribute augmentation further reduces ranking consistency, suggesting that fairness behavior is not simply compositional across attributes.

![](images/c2b5af3a3ac71b329754eb0104083a4f5718f10470397fa00ccebbaf97940b38.jpg)  
Figure 3: Model-ranking stability under intersectional augmentation, measured by Spearman’s rank correlation coefficient (SRCC).

## 5.5 Case Studies

We qualitatively inspect representative paired examples illustrating aggravation (correct → wrong) in Figure 4 and masking (wrong → correct) in Figure 5, which help explain how intersectional augmentation can exacerbate or conceal biased behaviors in model decisions. This highlights that reliable model evaluation should comprehensively cover diverse combinations of sensitive attributes, thereby providing a fairer and more realistic assessment setting.

## 6 Conclusion

We introduced ComFair, a configurable benchmark for controlled sensitivity auditing of intersectional fairness in LLMs. Its central contribution is a task-preserving pairing: each single-attribute source item is linked to Dual- and Multi-attribute counterparts, enabling outcome shifts, aggravation, masking, and ranking stability to be examined under the same content and task. Across the ten evaluated models, sensitivity is task- and subgroup-dependent. Discriminative tasks reveal both Correct→Wrong and Wrong→Correct transitions, generative scores change more asymmetrically, and rankings vary across composition depths. These results refine—rather than rediscover—the established observation that intersectional and marginal evaluations can differ: they show which benchmark conclusions are unstable under a controlled additional-attribute intervention. Because the instances are synthetic controls and generative scores remain evaluator-dependent, the results should not be read as estimates of real-world identity prevalence, social advantage, or a universal direction of harm. Our practical conclusion is therefore scoped: paired, composition-sensitive diagnostics should accompany marginal scores when fairness benchmarks are used to compare models or characterize failure modes.

![](images/1d7d9ea8199b8ac31b5cbcaadf44851486ccb00d719eada36b49be792bae338b.jpg)

![](images/e1ba75cfc84cf692b1cfda110fff858002c51a50958c34e51f0a18c28488aed2.jpg)  
Figure 5: Masking case (Wrong→Correct).

## Limitations

ComFair is designed as a configurable benchmark for intersectional fairness auditing, but its conclusions are bounded by the selected source datasets, sensitive-attribute schemas, prompting templates, evaluator choices, and the set of LLMs tested. The augmented examples preserve task semantics through automated checks and expert auditing, yet they cannot cover every culturally specific identity expression or deployment context. Results should therefore be interpreted as controlled benchmark evidence rather than as a complete characterization of model behavior in all real-world applications.

## Ethics Statement

This work studies social bias and fairness risks in LLMs, so the benchmark necessarily includes sensitive attributes and examples that may elicit stereotypical or harmful outputs. These materials are intended for controlled auditing, analysis, and mitigation rather than for profiling individuals or supporting downstream decisions about protected groups. Any release of data, prompts, model outputs, or annotations should follow the licenses of the underlying datasets, avoid exposing annotator identities, and clearly warn users about potentially offensive content.

We use only publicly available datasets, follow their anonymization and redistribution terms, and do not infer or attach new identities to individual people. Code and permissible benchmark data are available through the repositories linked in the abstract footnote, subject to the underlying licenses and safeguards for harmful or sensitive content.

## Acknowledgments

This research was supported by grants from the National Natural Science Foundation of China (62525606, 62477044), the Key Technologies R & D Program of Anhui Province (No. 202423k09020039), the support of the Young Elite Scientists Sponsorship Program by CAST (No. 2024QNRC001), and the Fundamental Research Funds for the Central Universities.

## References

Barry Becker and Ronny Kohavi. 1996. Adult. UCI Machine Learning Repository. Dataset.

Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2019. Nuanced metrics for measuring unintended bias with real data for text classification. In Companion Proceedings of The 2019 World Wide Web Conference, WWW ’19, pages 491–500, New York, NY, USA. Association for Computing Machinery.

Aylin Caliskan, Joanna J. Bryson, and Arvind Narayanan. 2017. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183–186.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2024. A survey on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology, 15(3):1–45.

Zhibo Chu, Zichong Wang, and Wenbin Zhang. 2024. Fairness in large language models: A taxonomic survey. ACM SIGKDD Explorations Newsletter, 26(1):34–48.

Kimberlé Crenshaw. 2013. Demarginalizing the intersection of race and sex: A black feminist critique of antidiscrimination doctrine, feminist theory and antiracist politics. In Feminist legal theories, pages 23–51. Routledge.

Jwala Dhamala, Tony Sun, Varun Kumar, Satyapriya Krishna, Yada Pruksachatkun, Kai-Wei Chang, and Rahul Gupta. 2021. BOLD: Dataset and metrics for measuring biases in open-ended language generation. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, pages 862–872, New York, NY, USA. Association for Computing Machinery.

James R. Foulds, Rashidul Islam, Kamrun Naher Keya, and Shimei Pan. 2020. An intersectional definition of fairness. In 2020 IEEE 36th International Conference on Data Engineering (ICDE), pages 1918–1921. IEEE.

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. 2024. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097– 1179.

Weibo Gao, Qi Liu, Linan Yue, Fangzhou Yao, Rui Lv, Zheng Zhang, Hao Wang, and Zhenya Huang. 2025. Agent4Edu: Generating learner response data by generative agents for intelligent education systems. Proceedings of the AAAI Conference on Artificial Intelligence, 39(22):23923–23932.

Weibo Gao, Hao Wang, Qi Liu, Fei Wang, Xin Lin, Linan Yue, Zheng Zhang, Rui Lv, and Shijin Wang. 2023. Leveraging transferable knowledge concept graph embedding for cold-start cognitive diagnosis. In Proceedings ofthe 46th international ACM SIGIR conference on research and development in information retrieval, pages 983–992.

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. 2020. RealToxicityPrompts: Evaluating neural toxic degeneration in language models. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 3356–3369, Online. Association for Computational Linguistics.

Usman Gohar and Lu Cheng. 2023. A survey on intersectional fairness in machine learning: Notions, mitigation, and challenges. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23, pages 6619–6627. International Joint Conferences on Artificial Intelligence Organization. Survey Track.

Hans Hofmann. 1994. Statlog (German Credit Data). UCI Machine Learning Repository. Dataset.

Zirui Hu, Zheng Zhang, Wenjun Feng, and Qi Liu. 2024a. Achieving universal fairness in machine learning: A multi-objective optimization perspective. In International Conference on Knowledge Science, Engineering and Management, pages 164–179. Springer.

Zirui Hu, Zheng Zhang, Qi Liu, Haoyang Bi, Zhenya Huang, Qingyang Mao, Weibo Gao, and Wenjun Feng. 2024b. Mitigating bias with incomplete sensitive labels: A confidence-based randomization framework. In International Conference on Database Systems for Advanced Applications, pages 139–155. Springer.

Zirui Hu, Zheng Zhang, Yingjie Wang, and Dacheng Tao. 2026. SAFER: A controllable safeguard for llms against backdoor attacks. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 14380–14398.

Yue Huang, Lichao Sun, Haoran Wang, Siyuan Wu, Qihui Zhang, Yuan Li, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, Hanchi Sun, Zhengliang Liu, Yixin Liu, Yijue Wang, Zhikun Zhang, Bertie Vidgen, Bhavya Kailkhura, Caiming Xiong, and 52 others. 2024. Position: TrustLLM: Trustworthiness in large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 20166–20270. PMLR.

Yue Huang, Qihui Zhang, Philip S. Yu, and Lichao Sun. 2023. TrustGPT: A benchmark for trustworthy and responsible large language models. arXiv preprint arXiv:2306.11507.

Meng Jiang, Keqin Bao, Jizhi Zhang, Wenjie Wang, Zhengyi Yang, Fuli Feng, and Xiangnan He. 2024. Item-side fairness of large language model-based recommendation system. In Proceedings of the ACM Web Conference 2024, WWW ’24, pages 4717–4726, New York, NY, USA. Association for Computing Machinery.

Falaah Arif Khan, Nivedha Sivakumar, Yinong Oliver Wang, Katherine Metcalf, Cezanne Camacho, Barry-John Theobald, Luca Zappella, and Nicholas Apostoloff. 2025. Investigating intersectional bias in large language models using confidence disparities in coreference resolution. In Proceedings of the Conference on Language Modeling (COLM).

Yingji Li, Mengnan Du, Rui Song, Xin Wang, and Ying Wang. 2023. A survey on fairness in large language models. Computing Research Repository, arXiv:2308.10149.

Yuhan Liu, Xiuying Chen, Xiaoqing Zhang, Xing Gao, Ji Zhang, and Rui Yan. 2024. From skepticism to acceptance: Simulating the attitude dynamics toward fake news. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pages 7886–7894. International Joint Conferences on Artificial Intelligence Organization. Human-Centred AI.

Yuhan Liu, Zirui Song, Juntian Zhang, Xiaoqing Zhang, Xiuying Chen, and Rui Yan. 2025. The stepwise deception: Simulating the evolution from true news to fake news with LLM agents. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 26176–26192, Suzhou, China. Association for Computational Linguistics.

Weicheng Ma, Brian Chiang, Tong Wu, Lili Wang, and Soroush Vosoughi. 2023. Intersectional stereotypes in large language models: Dataset and analysis. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 8589–8597, Singapore. Association for Computational Linguistics.

Mathieu Molina and Patrick Loiseau. 2022. Bounding and approximating intersectional fairness through marginal fairness. In Advances in Neural Information Processing Systems, volume 35, pages 16796–16807. Curran Associates, Inc.

Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. StereoSet: Measuring stereotypical bias in pretrained language models. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5356–5371, Online. Association for Computational Linguistics.

Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman. 2020. CrowS-Pairs: A challenge dataset for measuring social biases in masked language models. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language

Processing (EMNLP), pages 1953–1967, Online. Association for Computational Linguistics.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2025. A comprehensive overview of large language models. ACM Transactions on Intelligent Systems and Technology, 16(5):1–72.

Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel R. Bowman. 2022. BBQ: A hand-built bias benchmark for question answering. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2086–2105, Dublin, Ireland. Association for Computational Linguistics.

Harishwar Reddy, Madhusudan Srinivasan, and Upulee Kanewala. 2025. Metamorphic testing for fairness evaluation in large language models: Identifying intersectional bias in LLaMA and GPT. In 2025 IEEE/ACIS 23rd International Conference on Software Engineering Research, Management and Applications (SERA), pages 239–246. IEEE.

Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. 2018. Gender bias in coreference resolution. In Proceedings of the 2018 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 8–14, New Orleans, Louisiana. Association for Computational Linguistics.

Yu Shang, Chen Gao, Jiansheng Chen, Depeng Jin, and Yong Li. 2024. Improving item-side fairness of multimodal recommendation via modality debiasing. In Proceedings ofthe ACM Web Conference 2024, WWW ’24, pages 4697–4705, New York, NY, USA. Association for Computing Machinery.

Eric Michael Smith, Melissa Hall, Melanie Kambadur, Eleonora Presani, and Adina Williams. 2022. ”I’m sorry to hear that”: Finding new biases in language models with a holistic descriptor dataset. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9180–9211, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Yi Chern Tan and L. Elisa Celis. 2019. Assessing social and intersectional biases in contextualized word representations. In Advances in Neural Information Processing Systems, volume 32, pages 13209–13220. Curran Associates, Inc.

Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, and Bo Li. 2023. DecodingTrust: A comprehensive assessment of trustworthiness in GPT models. In Advances in Neural Information Processing Systems,

volume 36, pages 31232–31339. Curran Associates, Inc.

Chunjiang Wang, Fan Li, Wenbo Hu, Rui Yan, Kun Zhang, and Shaohua Kevin Zhou. 2026a. Concept bottleneck models for explainable decision making: A survey of progress, taxonomy, and future directions. International Joint Conference on Artificial Intelligence (IJCAI).

Song Wang, Peng Wang, Tong Zhou, Yushun Dong, Zhen Tan, and Jundong Li. 2025. CEB: Compositional evaluation benchmark for fairness in large language models. In International Conference on Learning Representations, volume 2025, pages 22627– 22668.

Yubo Wang, Juntian Zhang, Yichen Wu, Yankai Lin, Nils Lukas, and Yuhan Liu. 2026b. Forest before trees: Latent superposition for efficient visual reasoning. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11272–11288.

Kun Zhang, Zhendong Mao, An-An Liu, and Yongdong Zhang. 2023a. Unified adaptive relevance distinguishable attention network for image-text matching. IEEE Transactions on Multimedia, 25:1320–1332.

Zheng Zhang, Zhixiang Guo, Yuan Si, Siyuan Liang, Shunyu Liu, Weibo Gao, Song Wang, Leszek Rutkowski, Giuseppe Valenzise, Ming-Hsuan Yang, Lin William Cong, M. Jamal Deen, Sally Cripps, and Dacheng Tao. 2026a. A survey of world model benchmarks. Preprints.org.

Zheng Zhang, Qi Liu, Hao Jiang, Fei Wang, Yan Zhuang, Le Wu, Weibo Gao, and Enhong Chen. 2023b. FairLISA: Fair user modeling with limited sensitive attributes information. Advances in Neural Information Processing Systems, 36:41432–41450.

Zheng Zhang, Qi Liu, Siyuan Liang, Ning Li, Zirui Hu, Weibo Gao, Rui Li, Zhenya Huang, Leszek Rutkowski, Baosheng Yu, and Dacheng Tao. 2026b. Controllable contamination detection for reliable LLM evaluation with statistical guarantees. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 30122–30143, San Diego, California, United States. Association for Computational Linguistics.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 15–20, New Orleans, Louisiana. Association for Computational Linguistics.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Zican Dong, Yupeng Hou, Beichen Zhang, Yingqian Min, Junjie Zhang, Peiyu Liu, Xiaolei Wang, Yifan

Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, and 4 others. 2026. A survey of large language models. Frontiers ofComputer Science, 20(12):2012627.

Yan Zhuang, Qi Liu, Yuting Ning, Weizhe Huang, Rui Lv, Zhenya Huang, Guanhao Zhao, Zheng Zhang, Qingyang Mao, Shijin Wang, and Enhong Chen. 2023. Efficiently measuring the cognitive ability of LLMs: An adaptive testing perspective. arXiv preprint arXiv:2306.10512.

## A Appendix

## A.1 Task-specific Construction

Recognition and Selection. Following previous studies (Wang et al., 2025), we build classification-style data from BBQ (Parrish et al., 2022), CrowS-Pairs (Nangia et al., 2020), and WinoBias (Zhao et al., 2018). BBQ and WinoBias provide a context and a QA pair with candidate options, and CrowS-Pairs provides paired sentences (stereotypical vs. anti-stereotypical). These datasets naturally instantiate SELECTION, where a model selects the less biased option. To additionally support RECOGNITION, we convert each item into a single sentence text that expresses (or does not express) stereotypical content by combining the question, candidate answer(s), and context into a unified statement. We assign a binary label label = 1 if the statement is biased/stereotypical and label = 0 otherwise, following the original annotations.

For each base instance, we choose an additional attribute dimension and a specific attribute value (e.g., age → “young”). We then use GPT-5.2 to insert the attribute term into the target entity mention in x, under two principles. To avoid confounds, we apply insertion symmetrically: for SELECTION, we inject the same attribute term into both candidate options and any relevant context; for RECOGNITION, we inject it into both the context and the constructed text.

Continuation and Conversation. We build generative-style data on HolisticBias (Smith et al., 2022), which provides seed texts that implicate stereotypes. Following prior practice (Wang et al., 2025), we use GPT-5.2 to rewrite seeds into harmful stereotype prompts, and instantiate two task formats: (i) CONTINUATION, asking the model to continue the prompt; and (ii) CONVERSATION, presenting the prompt as a dialogue turn to elicit a response.

For each base prompt, we inject an additional attribute term into the target entity mention, enforcing the same two principles as above, yielding intersectional prompts for both CONTINUATION and CONVERSATION.

## A.2 Augmentation & Filter Examples

<table><tr><td>Example: Minimal intersectional augmentation</td></tr><tr><td>Original instance (gender specified)</td></tr><tr><td>“The female employee was promoted after years of consistent performance.&quot;</td></tr><tr><td>Augmented instances (add age; semantics preserved)</td></tr><tr><td>“The young female employee was promoted after years of consistent performance.&quot; “The older female employee was promoted after years of consistent performance.&quot;</td></tr></table>

<table><tr><td>Example: Semantic non-conflict</td></tr><tr><td>Filtered instance (violates Semantic Non-conflict) “The child female employee was promoted after years of consistent performance.&quot;</td></tr><tr><td></td></tr><tr><td>Reason. Although the edit follows the insertion rule, it violates commonsense and yields an implausible description, so it is filtered out by QC.</td></tr></table>

## A.3 Experimental Details

All experiments are conducted through API-based model inference interfaces for both open-source and closed-source models. For each evaluated model, we set the decoding temperature to 0 to reduce sampling variability, and leave all other generation or API parameters at their default values unless otherwise specified. Each evaluation instance is run once per model; we do not perform repeated sampling or average over multiple generations.

We set different maximum output lengths according to the task type. For objective discriminative tasks, i.e., Recognition and Selection, we set the maximum number of generated tokens to 256, since the expected output is a short label or option index. For subjective generative tasks, i.e., Continuation and Conversation, we set the maximum number of generated tokens to up to 2,048 to avoid truncating longer free-form responses. The same configuration is applied consistently across all models within each task type.

For Recognition and Selection, model outputs are parsed into discrete labels or option indices according to the task-specific prompt format. For Continuation and Conversation, model responses are scored using the bias-evaluation protocol described in Section 4.3. For API-based models, minor backend nondeterminism may still exist even under temperature-zero decoding; nevertheless, we keep the inference configuration fixed across models and tasks to ensure a consistent comparison.

## A.4 LLM Usage

We used large language models in the experimental pipeline of this work. GPT-5.2 was used for taskspecific prompt construction, controlled attribute injection, automatic quality checking, and automatic bias scoring for generative tasks. All LLM-generated or LLM-modified data were further filtered through the quality-control pipeline described in Section 4.2, including expert auditing on sampled instances. LLM-based writing assistance was used only for minor language polishing and readability improvements. All research ideas, methodological decisions, experimental analyses, claims, and final revisions were made and verified by the authors.

## A.5 Experiment Results

Table 9: Selection F1 under paired Gender↔Race interventions; parentheses report paired ∆.
<table><tr><td rowspan="2">Model</td><td colspan="3">Gender</td><td colspan="3">Race</td></tr><tr><td>Base</td><td>+White (∆)</td><td>+Black (∆)</td><td>Base</td><td>+Male (∆)</td><td>+Female (∆)</td></tr><tr><td>DS-V3.2-C</td><td>0.967</td><td>0.959 (-0.008)</td><td>0.961 (-0.006)</td><td>0.951</td><td>0.941 (-0.010)</td><td>0.956 (+0.005)</td></tr><tr><td>DS-V3.2-R</td><td>0.968</td><td>0.959 (-0.009)</td><td>0.956(-0.012)</td><td>0.970</td><td>0.984(+0.014)</td><td>0.960 (-0.010)</td></tr><tr><td>Gem-3F</td><td>0.983</td><td>0.959 (-0.024)</td><td>0.982 (-0.001)</td><td>0.968</td><td>0.963 (-0.005)</td><td>0.972 (+0.004)</td></tr><tr><td>GPT-5-mini</td><td>0.988</td><td>0.980 (-0.008)</td><td>0.992 (+0.004)</td><td>0.978</td><td>0.988(+0.010)</td><td>0.980 (+0.002)</td></tr><tr><td>Qwen3-8B</td><td>0.920</td><td>0.906 (-0.014)</td><td>0.937 (+0.017)</td><td>0.918</td><td>0.904 (-0.014)</td><td>0.936(+0.018)</td></tr><tr><td>Qwen3-235B</td><td>0.983</td><td>0.980 (-0.003)</td><td>0.970 (-0.013)</td><td>0.992</td><td>0.988 (-0.004)</td><td>0.992 (+0.000)</td></tr><tr><td>LLaMA-3.1-8B</td><td>0.776</td><td>0.800 (+0.024)</td><td>0.797 (+0.021)</td><td>0.788</td><td>0.763 (-0.025)</td><td>0.812 (+0.024)</td></tr><tr><td>LLaMA-3.3-70B</td><td>0.959</td><td>0.940 (-0.019)</td><td>0.963 (+0.004)</td><td>0.974</td><td>0.980 (+0.006)</td><td>0.980 (+0.006)</td></tr><tr><td>Claude-3.5</td><td>0.937</td><td>0.954(+0.017)</td><td>0.964(+0.027)</td><td>0.958</td><td>0.967 (+0.009)</td><td>0.960 (+0.002)</td></tr><tr><td>Claude-3.7</td><td>0.935</td><td>0.954(+0.019)</td><td>0.957 (+0.022)</td><td>0.944</td><td>0.951 (+0.007)</td><td>0.956(+0.012)</td></tr></table>

Table 10: Conversation bias scores under GPT-5.2 judging for paired Gender↔Race interventions. Lower is fairer.
<table><tr><td rowspan="2">Model</td><td colspan="3">Gender</td><td colspan="3">Race</td></tr><tr><td>Base</td><td>+White (∆)</td><td>+Black (∆)</td><td>Base</td><td>+Male (∆)</td><td>+Female (∆)</td></tr><tr><td>DS-V3.2-C</td><td>6.36</td><td>6.07 (-0.29)</td><td>5.64 (-0.72)</td><td>8.71</td><td>9.88(+1.17)</td><td>8.03 (-0.68)</td></tr><tr><td>DS-V3.2-R</td><td>5.62</td><td>5.70 (+0.08)</td><td>5.58 (-0.04)</td><td>9.10</td><td>8.42 (-0.68)</td><td>8.09 (-1.01)</td></tr><tr><td>Gem-3F</td><td>6.89</td><td>6.88 (-0.01)</td><td>6.13 (-0.76)</td><td>8.89</td><td>8.60 (-0.29)</td><td>8.20 (-0.69)</td></tr><tr><td>GPT-5-mini</td><td>6.20</td><td>6.80 (+0.60)</td><td>6.38(+0.18)</td><td>6.92</td><td>8.06(+1.14)</td><td>7.61 (+0.69)</td></tr><tr><td>Qwen3-8B</td><td>5.75</td><td>5.25 (-0.50)</td><td>5.11 (-0.64)</td><td>7.78</td><td>8.15(+0.37)</td><td>5.90 (-1.88)</td></tr><tr><td>Qwen3-235B</td><td>6.72</td><td>7.35 (+0.63)</td><td>7.10(+0.38)</td><td>10.35</td><td>10.36(+0.01)</td><td>10.91 (+0.56)</td></tr><tr><td>LLaMA-3.1-8B</td><td>8.24</td><td>8.33 (+0.09)</td><td>8.97(+0.73)</td><td>10.86</td><td>12.65(+1.79)</td><td>10.46(-0.40)</td></tr><tr><td>LLaMA-3.3-70B</td><td>6.99</td><td>7.14(+0.15)</td><td>7.77(+0.78)</td><td>10.72</td><td>11.44(+0.72)</td><td>10.05 (-0.67)</td></tr><tr><td>Claude-3.5</td><td>3.80</td><td>2.11 (-1.69)</td><td>1.21 (-2.59)</td><td>6.11</td><td>6.01 (-0.10)</td><td>5.44 (-0.67)</td></tr><tr><td>Claude-3.7</td><td>5.94</td><td>6.13 (+0.19)</td><td>5.41 (-0.53)</td><td>7.91</td><td>7.66 (-0.25)</td><td>7.44 (-0.47)</td></tr></table>

Table 11: Human conversation bias scores for the same paired interventions as Table 10. Lower is fairer.
<table><tr><td rowspan="2">Model</td><td colspan="3">Gender</td><td colspan="3">Race</td></tr><tr><td>Base</td><td>+White (∆)</td><td>+Black (∆)</td><td>Base</td><td>+Male (∆)</td><td>+Female (∆)</td></tr><tr><td>DS-V3.2-C</td><td>12.75</td><td>13.23 (+0.48)</td><td>10.92 (-1.83)</td><td>20.34</td><td>20.21 (-0.13)</td><td>18.48 (-1.86)</td></tr><tr><td>DS-V3.2-R</td><td>12.78</td><td>13.18(+0.40)</td><td>11.60 (-1.18)</td><td>20.26</td><td>18.80 (-1.46)</td><td>18.95 (-1.31)</td></tr><tr><td>Gem-3F</td><td>13.15</td><td>12.79 (-0.36)</td><td>12.42 (-0.73)</td><td>17.39</td><td>18.06(+0.67)</td><td>17.03 (-0.36)</td></tr><tr><td>GPT-5-mini</td><td>14.25</td><td>14.27 (+0.02)</td><td>12.31 (-1.94)</td><td>14.51</td><td>17.68(+3.17)</td><td>15.71 (+1.20)</td></tr><tr><td>Qwen3-8B</td><td>6.96</td><td>6.18 (-0.78)</td><td>5.33 (-1.63)</td><td>12.98</td><td>11.92 (-1.06)</td><td>8.31 (-4.67)</td></tr><tr><td>Qwen3-235B</td><td>11.36</td><td>12.00 (+0.64)</td><td>11.55 (+0.19)</td><td>18.31</td><td>17.81 (-0.50)</td><td>18.01 (-0.30)</td></tr><tr><td>LLaMA-3.1-8B</td><td>14.55</td><td>15.24(+0.69)</td><td>13.89 (-0.66)</td><td>18.87</td><td>21.35(+2.48)</td><td>18.48 (-0.39)</td></tr><tr><td>LLaMA-3.3-70B</td><td>12.11</td><td>11.20 (-0.91)</td><td>10.95 (-1.16)</td><td>20.34</td><td>22.13(+1.79)</td><td>19.39 (-0.95)</td></tr><tr><td>Claude-3.5</td><td>6.78</td><td>3.06(-3.72)</td><td>1.67 (-5.11)</td><td>11.19</td><td>11.16(-0.03)</td><td>10.71 (-0.48)</td></tr><tr><td>Claude-3.7</td><td>10.75</td><td>9.59 (-1.16)</td><td>7.67 (-3.08)</td><td>14.25</td><td>14.05 (-0.20)</td><td>11.57 (-2.68)</td></tr></table>

Table 12: Selection aggravation (Agg.) and masking (Mask.) rates under paired Gender↔Race interventions.
<table><tr><td rowspan="3">Model</td><td colspan="4">Gender</td><td colspan="4">Race</td></tr><tr><td colspan="2">+Black</td><td colspan="2">+White</td><td colspan="2">+Female</td><td colspan="2">+Male</td></tr><tr><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td></tr><tr><td>Claude-3.5</td><td>0.35%</td><td>50.00%</td><td>0.36%</td><td>31.58%</td><td>0.66%</td><td>23.53%</td><td>1.32%</td><td>35.71%</td></tr><tr><td>Claude-3.7</td><td>1.06%</td><td>41.18%</td><td>0.71%</td><td>42.11%</td><td>0.33%</td><td>21.05%</td><td>0.00%</td><td>15.00%</td></tr><tr><td>DS-V3.2-C</td><td>1.65%</td><td>28.57%</td><td>0.83%</td><td>11.11%</td><td>1.24%</td><td>11.11%</td><td>1.28%</td><td>25.00%</td></tr><tr><td>DS-V3.2-R</td><td>1.64%</td><td>0.00%</td><td>2.49%</td><td>55.56%</td><td>2.47%</td><td>42.86%</td><td>0.41%</td><td>62.50%</td></tr><tr><td>Gem-3F</td><td>0.82%</td><td>0.00%</td><td>2.05%</td><td>16.67%</td><td>0.41%</td><td>25.00%</td><td>0.83%</td><td>12.50%</td></tr><tr><td>GPT-5-mini</td><td>0.41%</td><td>66.67%</td><td>2.02%</td><td>100.00%</td><td>0.85%</td><td>75.00%</td><td>0.41%</td><td>33.33%</td></tr><tr><td>LLaMA-3.1-8B</td><td>9.75%</td><td>48.44%</td><td>9.36%</td><td>47.69%</td><td>7.54%</td><td>42.42%</td><td>10.71%</td><td>30.30%</td></tr><tr><td>LLaMA-3.3-70B</td><td>0.00%</td><td>9.09%</td><td>1.73%</td><td>0.00%</td><td>0.00%</td><td>40.00%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Qwen3-235B</td><td>1.22%</td><td>0.00%</td><td>0.41%</td><td>0.00%</td><td>0.40%</td><td>50.00%</td><td>0.40%</td><td>0.00%</td></tr><tr><td>Qwen3-8B</td><td>0.87%</td><td>35.00%</td><td>2.60%</td><td>10.53%</td><td>0.43%</td><td>11.76%</td><td>1.77%</td><td>16.67%</td></tr></table>

Table 13: Continuation conditional mean bias-score increases (Agg.) and decreases (Mask.) under GPT-5.2 judging.
<table><tr><td rowspan="2">Model</td><td colspan="4">Gender</td><td colspan="4">Race</td></tr><tr><td colspan="2">+Black</td><td colspan="2">+White</td><td colspan="2">+Female</td><td colspan="2">+Male</td></tr><tr><td></td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td></tr><tr><td>Claude-3.5</td><td>5.72</td><td>12.00</td><td>11.07</td><td>17.12</td><td>5.00</td><td>12.43</td><td>12.00</td><td>13.90</td></tr><tr><td>Claude-3.7</td><td>7.09</td><td>16.53</td><td>9.88</td><td>15.08</td><td>9.56</td><td>12.43</td><td>15.14</td><td>3.33</td></tr><tr><td>DS-V3.2-C</td><td>12.71</td><td>9.30</td><td>11.17</td><td>13.89</td><td>10.64</td><td>11.77</td><td>11.29</td><td>7.00</td></tr><tr><td>DS-V3.2-R</td><td>11.59</td><td>8.55</td><td>11.59</td><td>9.69</td><td>9.05</td><td>6.38</td><td>9.15</td><td>9.00</td></tr><tr><td>Gem-3F</td><td>7.67</td><td>28.50</td><td>11.90</td><td>40.70</td><td>7.13</td><td>8.40</td><td>8.00</td><td>6.50</td></tr><tr><td>GPT-5-mini</td><td>11.66</td><td>21.38</td><td>12.32</td><td>21.88</td><td>9.39</td><td>7.50</td><td>8.88</td><td>5.75</td></tr><tr><td>LLaMA-3.1-8B</td><td>16.53</td><td>17.00</td><td>16.79</td><td>14.36</td><td>22.60</td><td>26.64</td><td>11.89</td><td>8.75</td></tr><tr><td>LLaMA-3.3-70B</td><td>12.64</td><td>11.08</td><td>11.36</td><td>5.71</td><td>11.91</td><td>7.78</td><td>8.12</td><td>6.00</td></tr><tr><td>Qwen3-235B</td><td>10.17</td><td>20.40</td><td>13.00</td><td>29.79</td><td>9.05</td><td>7.57</td><td>9.00</td><td>17.00</td></tr><tr><td>Qwen3-8B</td><td>11.89</td><td>9.43</td><td>14.96</td><td>13.92</td><td>12.47</td><td>8.50</td><td>10.62</td><td>8.00</td></tr></table>

Table 14: Conversation conditional mean bias-score increases (Agg.) and decreases (Mask.) under GPT-5.2 judging.
<table><tr><td rowspan="2">Model</td><td colspan="4">Gender</td><td colspan="4">Race</td></tr><tr><td colspan="2">+Black</td><td colspan="2">+White</td><td colspan="2">+Female</td><td colspan="2">+Male</td></tr><tr><td></td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td><td>Agg.</td><td>Mask.</td></tr><tr><td>Claude-3.5</td><td>5.46</td><td>6.78</td><td>7.79</td><td>6.28</td><td>8.36</td><td>7.72</td><td>9.69</td><td>7.97</td></tr><tr><td>Claude-3.7</td><td>6.33</td><td>6.29</td><td>6.50</td><td>5.15</td><td>7.18</td><td>7.66</td><td>7.28</td><td>6.90</td></tr><tr><td>DS-V3.2-C</td><td>6.49</td><td>7.39</td><td>6.71</td><td>7.64</td><td>7.50</td><td>6.86</td><td>8.22</td><td>7.87</td></tr><tr><td>DS-V3.2-R</td><td>7.32</td><td>6.47</td><td>7.13</td><td>6.19</td><td>7.84</td><td>8.76</td><td>10.11</td><td>9.80</td></tr><tr><td>Gem-3F</td><td>7.24</td><td>7.87</td><td>8.33</td><td>7.08</td><td>8.41</td><td>9.94</td><td>9.16</td><td>9.39</td></tr><tr><td>GPT-5-mini</td><td>7.35</td><td>6.15</td><td>7.11</td><td>5.95</td><td>7.70</td><td>6.22</td><td>7.12</td><td>5.87</td></tr><tr><td>LLaMA-3.1-8B</td><td>11.67</td><td>8.96</td><td>8.99</td><td>9.39</td><td>10.81</td><td>9.94</td><td>13.57</td><td>12.73</td></tr><tr><td>LLaMA-3.3-70B</td><td>10.79</td><td>6.74</td><td>8.49</td><td>7.60</td><td>8.23</td><td>11.05</td><td>10.50</td><td>9.30</td></tr><tr><td>Qwen3-235B</td><td>7.77</td><td>7.31</td><td>8.64</td><td>6.68</td><td>12.55</td><td>10.59</td><td>10.69</td><td>10.43</td></tr><tr><td>Qwen3-8B</td><td>6.74</td><td>7.17</td><td>6.84</td><td>7.25</td><td>6.10</td><td>7.62</td><td>8.15</td><td>7.46</td></tr></table>

## A.6 Data Statistics

Table 15: User-configurable evaluation capabilities across four task types.
<table><tr><td>Task</td><td>Dataset</td><td>Base attribute Abase</td><td>Added attribute set Aadd</td><td>Added values / schema</td><td>#Aug / sample Single / Dual / Multi</td><td>Size (Orig → Aug)</td></tr><tr><td rowspan="8">Recognition</td><td rowspan="8">BBQ WinoBias CrowS-Pairs</td><td>Age Gender</td><td>gender; race age; race</td><td rowspan="8">age: young / adult / old gender: female / male</td><td>1417 / 9160 /4134 1960 / 20804 / 33633</td><td rowspan="8">11745 → 246.05k</td></tr><tr><td>RaceEthnicity</td><td>age; gender</td><td>2468 / 10131 / 6457</td></tr><tr><td>PhysicalAppearance</td><td>age; gender; race</td><td>624 / 6498 / 14305</td></tr><tr><td>Nationality</td><td>age; gender; race</td><td></td></tr><tr><td></td><td></td><td>1184 / 4832 / 14836</td></tr><tr><td>SES</td><td>age; gender; race</td><td>2694 / 25795 / 57442</td></tr><tr><td>DisabilityStatus</td><td>age; gender; race</td><td>599 / 1432 / 3020</td></tr><tr><td>Religion</td><td>age; gender; race age; gender; race</td><td>477 / 6547 / 14991 322 / 4372 / 7661</td></tr><tr><td rowspan="8"></td><td rowspan="8">BBQ</td><td></td><td>gender; race</td><td rowspan="8">age: young / adult / old gender: female / male</td><td></td><td rowspan="8">6272 → 119.00k</td></tr><tr><td>Age Gender</td><td>age; race</td><td>736 / 4094 / 2924</td></tr><tr><td>RaceEthnicity</td><td>age; gender</td><td>1134 / 8185 / 13890</td></tr><tr><td>PhysicalAppearance</td><td>age; gender; race</td><td>1376 / 4453 / 3412</td></tr><tr><td></td><td>age; gender; race</td><td>315/1792 / 4097</td></tr><tr><td>Nationality SES</td><td>age; gender; race</td><td>616 / 4859 / 10358</td></tr><tr><td>DisabilityStatus</td><td>age; gender; race</td><td>1372 / 11346 / 29349 311 / 5635 / 5635</td></tr><tr><td>Religion</td><td>age; gender; race</td><td>240 /1741 / 3872</td></tr><tr><td rowspan="7"></td><td rowspan="7">HolisticBias Continuation</td><td>race</td><td>age; gender; race age; gender</td><td rowspan="7">age: young / adult / old gender: female / male</td><td>172 / 1068 / 2292 827 / 4617 / 4230</td><td rowspan="6">2368 → 39.80k</td><td rowspan="6"></td></tr><tr><td>gender</td><td>age; race</td><td></td></tr><tr><td>age</td><td>gender; race</td><td>race: Asian / White / Black</td></tr><tr><td>religion</td><td>age; gender; race</td><td>Latinx / Middle_Eastern</td></tr><tr><td></td><td>age; gender</td><td>531 / 4265 / 11705</td></tr><tr><td>race</td><td></td><td>801 / 4680 / 4380</td></tr><tr><td>gender</td><td>age; race</td><td></td></tr><tr><td rowspan="4">Conversation</td><td rowspan="4">HolisticBias</td><td>age</td><td>gender; race</td><td>age: young / adult / old gender: female / male race: Asian / White / Black</td><td>636 / 4808 / 8486</td><td rowspan="4">2378 → 74.22k</td></tr><tr><td>religion</td><td>age; gender; race</td><td></td><td>397 / 1569 / 1356</td></tr><tr><td></td><td></td><td>Latinx / Middle_Eastern</td><td>544 /4810 /44130</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

## A.7 Artifact Licenses and Intended Usage

Table 16 summarizes the existing artifacts used in this work, their sources, license or usage terms, and how they are used in our benchmark construction and evaluation pipeline. We use these artifacts for research and evaluation purposes only. The public ComFair release preserves attribution and remains subject to the licenses and redistribution terms of the underlying artifacts.

Table 16: Existing artifacts used in this paper and their license or usage terms.
<table><tr><td>Artifact</td><td>Source</td><td>License / Terms</td><td>Use in this paper</td></tr><tr><td>BBQ</td><td>Parrish et al. (2022)</td><td>CC-BY-4.0</td><td>Source data for Recognition and Selection; we augment QA-style bias examples with additional sensitive attributes.</td></tr><tr><td>CrowS-Pairs</td><td>Nangia et al. (2020)</td><td>CC-BY-SA-4.0</td><td>Source data for Recognition and Selection; we adapt sen- tence pairs and construct intersectional variants.</td></tr><tr><td>WinoBias</td><td>Zhao et al. (2018)</td><td>MIT License</td><td>Source data for discriminative evaluation; we standardize coreference-style examples into our task format.</td></tr><tr><td>HolisticBias</td><td>Smith et al. (2022)</td><td>Dataset: CC-BY-SA-4.0; code: MIT License</td><td>Source data for Continuation and Conversation; we use demographic descriptors and seed texts to build generative prompts.</td></tr><tr><td>ComFair</td><td>This work</td><td>Publicly available; underlying ar- tifact licenses continue to apply</td><td>Benchmark constructed in this paper; used for paired single- attribute and intersectional fairness evaluation.</td></tr></table>

![](images/2b9337bfacaccbeef5e5e152a6a822a750804d04f9e60ddc4c5cb05145158f77.jpg)  
Figure 6: Recognition task prompts organized by pipeline stage.

![](images/23cd6a0cd52fa323e81671ae907d15f204baf97fcfbd7b738a6aee38693468b1.jpg)  
Figure 7: Selection task prompts organized by pipeline stage.

![](images/667bca9174f338adc3241e038a97b3eae6825c5bdb2237f24de791d4a8883da2.jpg)  
Figure 8: Continuation task prompts organized by pipeline stage.

![](images/35e67da52ca190afe6a072ed9eda1b0c97e58584eae0d9a646ca41b37889d721.jpg)  
Figure 9: Conversation task prompts organized by pipeline stage.