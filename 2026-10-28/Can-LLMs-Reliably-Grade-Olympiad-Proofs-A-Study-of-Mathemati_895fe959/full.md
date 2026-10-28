# Can LLMs Reliably Grade Olympiad Proofs? A Study of Mathematical Verification with LLMs

Azim Ospanov<sup>1,2∗</sup> Zijin Feng<sup>1†</sup> Ding Ding<sup>1,3</sup> Chengwu Liu<sup>1</sup> Haoli Bai<sup>1</sup> Jiacheng Sun<sup>1</sup> Lifeng Shang<sup>1</sup> Farzan Farnia<sup>2,†</sup>

<sup>1</sup>Huawei Foundation Model Department

<sup>2</sup>The Chinese University of Hong Kong

<sup>3</sup>The Hong Kong University of Science and Technology

## Abstract

Large Language Models (LLMs) have made remarkable progress on solving competition-level mathematics, yet their ability to verify natural-language mathematical proofs remains relatively underexplored. This gap is especially pronounced for Olympiad-level proofs, which require meticulous assessment of each claim in a long reasoning chain. Current proof verification thus primarily relies on expert inspection, which is costly to scale. In this paper, we present a systematic study of LLM-based verification in this setting. Specifically, we introduce a human-curated dataset of USAMO 2025 candidate solutions with expert grading, and conduct a broad evaluation of existing LLM verifiers across frontier models. We also propose TROJAN, an iterative self-critique pipeline for generating high-fidelity adversarial proofs at scale. Our comprehensive evaluation covers 20 models spanning seven metrics, two prompt styles, three inference methods, and three prompt templates, showing that state-of-the-art models still leave considerable room for improvement. To narrow this gap, we propose MABGrader, a Multi-Armed Bandit framework that reframes grading as arm selection over the discrete score space. MABGrader outperforms the SOTA ProofGrader by 26.4% in Quadratic Weighted Kappa and reduces grading error by 14%. Code and dataset are available at https://github.com/aziksh-ospanov/TROJAN.

## 1 Introduction

Mathematical problem-solving is a core capability of large language models (LLMs). Recent frontier models have demonstrated remarkable proficiency in competition-level mathematics, including the International Mathematical Olympiad (IMO) and the United States of America Mathematical Olympiad (USAMO) [58, 24]. Much of this progress, however, is measured by final-answer correctness, which is a limited proxy for complex mathematical reasoning since a solution may reach the correct answer while containing logical gaps, unjustified claims, or subtle errors [47, 33]. Moreover, many proving problems do not admit a final answer, requiring costly expert verification.

These motivate scalable methods for evaluating natural-language mathematical solutions. In this work, we study the verification of natural-language mathematical solutions. Specifically, an LLM-based verifier receives a mathematical problem, a candidate solution, and optionally a reference solution and a marking scheme; it must then identify reasoning errors and assign a fair grade [33, 4]. Recent work has approached this task by constructing proof-verification benchmarks and designing LLM-based graders: MathArena [4] evaluates LLMs on competition proof problems using expert graders to assess model-generated solutions, ProofAutoGrader [32] constructs IMO-GradingBench and studies automatic grading with reference solutions and guidelines, and ProofGrader [33] analyzes how model choice, grading schemes, reference solutions, and ensembling affect agreement with expert.

![](images/9dd0c17bd94f0802c3776b93d88b3c7a6e70ce208670936dd02daac77459e654.jpg)  
Figure 1: Performance of models on Trojan-Bench(H) , comparing Majority@16 with our MABGr. inference method across 4 metrics. Color and marker size denotes the model and parameter scale, resp. The spread indicates considerable room for improvement, while MABGr. show consistent gains.

Despite this progress, two challenges remain for Olympiad-level mathematical verification. First, existing datasets often collect LLM-generated candidate solutions and rely on human experts to grade them, which is costly, difficult to scale, and provides limited control over the error distribution [4, 33]. This makes it difficult to diagnose which types of mistakes LLM can or cannot detect. Second, verification itself remains difficult. Competition proofs are long, multi-step arguments in which a single mistake or omission can invalidate later conclusions or alter the appropriate partial credit. A verifier must thus go beyond surface plausibility by localizing the failure point and assessing its effect on the final grade. Figure 1 illustrates this gap. Existing approaches, including prompt engineering, PRM/ORM-based selection [29, 57], and LLM-judge ensembles [8, 22], can improve verification performance, but often require careful finetuning and stacking of different models. These limitations motivate diagnostic benchmarks for targeted error analysis and inference methods that improve grading without modifying the underlying base model.

In this work, we present a systematic study of LLM-based verification for Olympiad-level mathematics. We introduce a human-curated benchmark dataset derived from the USAMO 2025, where rubric-aligned adversarial errors, such as imprecise notation and unjustified lemma applications, are deliberately injected into candidate solutions. This design enables targeted evaluation of whether LLM judges can detect specific reasoning flaws and assign appropriate partial credit, while providing greater control over error type and severity than spontaneous LLM-generated mistakes. To scale data generation, we propose an iterative self-critique pipeline called TROJAN (Targeted adveRsarial errOr inJection for mAth evaluatioN) that produces adversarial solutions with human-like flaws. We show that fine-tuning on this synthetic data improves agreement with expert grading. We further propose MABGrader, an agentic Multi-Armed Bandit framework for mathematical verification. Rather than producing a verification judgment in one pass, MABGrader formulates verification as an optimal selection problem over candidate judgments, where each arm corresponds to a possible grade with supporting error analysis. Through iterative sampling and reward optimization, it allocates more computation to uncertain cases and improves grade selection without modifying the underlying base model. Our contributions are summarized as follows:

• We introduce Trojan-Bench(H), a human-curated dataset of error-injected USAMO 2025 solutions for evaluating how LLM verifiers detect subtle reasoning flaws and assign grades.

• We propose TROJAN, a self-critique pipeline for generating high-fidelity adversarial samples, and show that training on the resulting Trojan-Bench(S) improves alignment with experts.

• We conduct a systematic evaluation of 20 models across multiple prompts, inference methods, and metrics, measuring their verification abilities on Olympiad-level mathematics.

• We introduce MABGrader, a Multi-Armed Bandit algorithm that mitigates LLM verification instability and provides a robust alternative to single- and multi-shot grading methods.

## 2 Related Works

Benchmarks for Mathematical Reasoning. The mathematical evaluation includes foundational datasets such as GSM8K [9] and MATH [29], alongside advanced benchmarks like AIME [10], HARDMath2 [46], and CollegeMath [49]. Recent research has focused on competition-level mathematics through specialized benchmarks including Omni-MATH [18], Olympiad-Bench [23], and HARP [63]. Modern frameworks such as MathArena [4] provide model rankings based on challenging problems sampled from fresh Olympiads. Beyond benchmarking, recent breakthroughs have led to systems that participate directly in the International Mathematical Olympiad (IMO) [24, 21, 53]. These advancements demonstrate that models are moving toward achieving performance comparable to top human competitors.

LLM-as-a-Judge in mathematics. LLM-as-a-Judge frameworks utilize large language models to evaluate responses based on metrics such as correctness and fairness [22, 65, 28, 52]. Recent studies on LLM reliability show that while these models achieve high correlation with human experts on standard tasks [48, 17], they remain susceptible to issues such as sycophancy [43]. However, when grading complex mathematical theorems, many researchers continue to rely on human experts, a practice that limits the scalability of such evaluation pipelines [33, 32, 59, 42].

Verification in formal mathematics. Complementing informal evaluation, formal mathematics utilizes proof assistants such as Lean [34], Coq [5], and Isabelle [35] to achieve automated verification. Benchmarks in this domain rely on the underlying language kernel to confirm proof validity, offering an objective alternative to natural language assessment [64, 39, 54, 62]. Recent research also employs compiler feedback to guide LLM generation [38, 30] and integrates informal reasoning steps into formal proofs to enhance verifiability [40, 25, 6].

Industry standards on fair grading. According to [13], in math competitions with binary grading, the agreement is approximately 90.4%. As we expand this to a 0–7 scale, the agreement and correlation will naturally be lower. Although those figures are not directly translatable to our setting, the consensus is that prestigious math Olympiads and new mathematical proofs should have a high rate of agreement among judges and reviewers. The intended gold standard involves multiple graders, a rubric, and a reconciliation process, which is stricter than single-rater "good enough" labeling. For general educational purposes, [60] reports an absolute $Q W K _ { h u m a n , L L M } \ge 0 . 7 0$ as an acceptable metric for automated scoring. This is not a universal rule of thumb, as different domains have different requirements. We use these guidelines as a conservative lower bound for model evaluation. In this work, we systematically examine the current state of LLM grading to clarify the performance of automated judging against human labels.

## 3 Problem Formulation

In this work, we study the verification of natural-language mathematical solutions. Given a mathematical problem p, a candidate solution s˜, and optionally a reference solution Ref and a marking scheme MS, the goal is for an LLM-based verifier to identify reasoning errors and assign a grade that fairly reflects the correctness and completeness of the solution. Following common practice in [4, 32, 33], grades range from 0 to 7.

We consider two verification settings, illustrated in Figure 2. In the Full setting, corresponding to the Ref+MS configuration of [33], the verifier is provided with both the reference solution and the marking scheme specifying point allocations and deductions. In the Blind setting, corresponding to the None configuration of [33], the verifier receives only the problem and candidate solution, without access to either a reference solution or a marking scheme.

## 4 Dataset Construction

To comprehensively evaluate whether LLMs can verify Olympiad-level proofs and assign appropriate grades, in this section we describe the construction of our human-curated Trojan-Bench(H) benchmark and the proposed data generation pipeline TROJAN, which we use to obtain dataset Trojan-Bench(S) .

![](images/cf460cd6428c7c00d47090a1ea75ef0a95a1f1e7d8e1e77f5dbfa518111a9e48.jpg)  
Figure 2: Example verification inputs for USAMO 2025 Problem 1 under the Full and Blind settings. In the Full setting, the verifier receives the problem, reference solution, marking scheme, and candidate solution. In the Blind setting, the verifier receives only the problem and candidate solution.

## 4.1 Human curated dataset

We construct Trojan-Bench(H) to support controlled evaluation of LLM-based proof verification. Unlike existing datasets that collect LLM-generated solutions and annotate them with human grades, we use a backward construction procedure: starting from golden reference solutions, we map proof components to marking-scheme criteria and systematically inject errors that correspond to targeted point deductions. We build our dataset Trojan-Bench(H) from the USAMO 2025 competition set, which contains six proof-based problems across multiple areas of mathematics. Grades range from 0 to 7 according to the marking schemes. We use the USAMO 2025 problems, reference solutions, and marking schemes provided by MathArena [4].

Concretely, we construct candidate solutions by injecting subtle errors into the golden solution, disrupting the proof structure and leading to specific point deductions. For each problem, we randomly sample a target grade and introduce errors designed to match the corresponding markingscheme deductions. Furthermore, when constructing faulty solutions, we applied heavier penalties to fundamental parts of the proof, such as operating in base-2n in Problem 1 or establishing the correct definition and structure of cupcake segments in Problem 6. A logical gap in a fundamental step typically causes a cascade of point losses across other criteria. The 2026 version of the USAMO in [4] follows a similar logic by specifying overarching deductions.

To account for the inherent variance in individual grading, we utilized two independent annotators for the final dataset. We followed the data creation pipeline from [45] to increase the validity and reliability of the final set. However, we omitted the final step of non-expert validation due to the specialized nature of mathematical proofs. The result is a set of 48 human-verified USAMO 2025 candidate solutions with a corresponding human grade and a justification. We provide a detailed description of the creation pipeline and aggregated metrics in the following sections, and a full datasheet can be found in the Appendix Section A.

Table 1: Inter-annotator agreement on our human-curated benchmark Trojan-Bench(H)
<table><tr><td>Metric</td><td>QWK</td><td>Pearson r</td><td>RMSE</td><td>MAE</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td><td> $\mathrm { \Delta W T A _ { \leq 2 } }$ </td><td>Exact Match</td><td>Bias</td></tr><tr><td>Value</td><td>0.76</td><td>0.89</td><td>1.09</td><td>0.62</td><td>0.85</td><td>0.94</td><td>0.58</td><td>±0.08</td></tr></table>

Expert Inter-Rater Agreement. Table 1 provides the agreement metrics for the human-curated dataset following the procedure in [45]. The results show a Quadratic Weighted Kappa (QWK) of 0.76 and a Pearson correlation of 0.89, which exceed the performance of any of the evaluated LLMs and indicate strong expert consensus. Scoring consistency is further reflected in the Within Tolerance Agreement, where 85% of scores fall within 1 point $( W T A _ { \leq 1 } )$ and 94% fall within

![](images/c385a01eef574f46f5435a8af1132a081256bc826e73f9773b3b024cbd86f7cd.jpg)  
Figure 3: Overview of our synthetic data generation pipeline TROJAN. Given a problem, reference solution, marking scheme, and target score profile, the pipeline injects rubric-aligned errors, performs iterative self-critique and refinement, and outputs an adversarial proof with a grade justification.

2 points $( W T A _ { \leq 2 } )$ . Combined with a low systematic bias of ±0.08, these metrics establish the human-curated set as a reliable gold standard for assessing automated grading systems.

## 4.2 Synthetic dataset generation

To scale benchmark construction beyond manual error injection, we introduce TROJAN, an LLMbased pipeline for generating adversarial mathematical proofs. Given a mathematical problem, a reference golden solution, a marking scheme, and a target score profile, TROJAN outputs a proof whose errors are calibrated to be consistent with the specified score profile, together with a structured grade justification. Figure 3 provides an overview of the pipeline.

The score profile specifies the target point allocation across marking-scheme criteria. The generator LLM is given one human-authored error-injection example for one-shot prompting and then analyzes the marking scheme. For criteria assigned full marks, it is allowed to closely paraphrase the corresponding part of the reference solution; for each criterion with points losses, it is tasked with injecting errors calibrated to the target score. The same LLM then performs self-critique using a separate prompt. The goal is to isolate the parts that contain mistakes and produce an explicit self-judgment. Its output is an alignment judgment together with revision guidance. In the refinement stage, the model applies the suggested revisions to the corresponding proof blocks. This critique-refinement loop terminates once the self-critique model accepts the current candidate $n _ { \mathrm { a l i g n m e n t } }$ consecutive rounds; if alignment is broken before then, the streak is discarded and refinement continues with the latest critique. The maximum number of critique-refinement rounds is bounded by $n _ { \mathrm { i t e r } }$

The pipeline is general and can be applied to any problem set with valid reference solutions and marking schemes. We construct a large synthetic dataset Trojan-Bench(S) from four competitions: USAMO 2025, USAMO 2026, IMO 2025, and IMC 2025, with problem statements, reference solutions, with data sourced from the MathArena repository [4]. When evaluating the synthetic data, we treat this grade and justification as ground-truth labels. Experiments on the subset of Trojan-Bench(S) show that it closely correlates with Trojan-Bench(H), suggesting good alignment.

Table 2: Correlation and agreement of the synthetically-generated benchmark Trojan-Bench(S) with expert graders.
<table><tr><td>Metric</td><td>QWK</td><td>Pearson r</td><td>RMSE</td><td>MAE</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td><td> $\mathrm { \Delta W T A _ { \leq 2 } }$ </td><td>Exact Match</td><td>Bias</td></tr><tr><td>Value</td><td>0.78</td><td>0.87</td><td>1.20</td><td>0.77</td><td>0.83</td><td>0.90</td><td>0.67</td><td>0.11</td></tr></table>

Table 3: Inter-annotator agreement on the synthetically-generated benchmark Trojan-Bench(S).
<table><tr><td>Metric QWK</td><td>Pearson r RMSE MAE  $\mathrm { \Delta W T A _ { \leq 1 } }$   $\mathrm { W T A } _ { \leq 2 }$ </td></tr><tr><td>Value 0.78 0.94 0.70</td><td>0.49 1.00 1.00</td></tr><tr><td>Exact Match Bias</td><td>0.51</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>±0.00</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

Audit of Trojan-Bench(S). To further validate the quality of the proposed TROJAN, we performed a manual audit of the Trojan-Bench(S) subset. We sampled 50 problems and annotated using similar procedures as in the construction and audit of Trojan-Bench(H). Each problem was verified by two experts and we present the correlation, error and bias metrics between LLM and human assigned grades in Table 2. We observe that generally the correlation is high and ∼ 83% of problems fall within one grading points. This generally indicates that the proposed rubric-based error injection helps LLMs inject relevant and appropriate synthetic errors into the correct solutions. We also report the inter-annotation agreement in Table 3 between the experts in this audit and show that the consensus was high when grading, with all graded problems falling into $W T A _ { \leq 1 }$ . Combined, this shows robustness and overall high quality of synthetically injected errors via TROJAN.

## 5 Our Approach

To address the difficulty of assigning fair grades to Olympiad-level problems, we implemented a Multi-Armed Bandit (MAB) algorithm for grade selection. Even with access to a mark scheme and a reference solution, producing a grade outright is often hard. Prior work also suggests that evaluating a pre-assigned grade together with its justification is easier than scoring from scratch, since the candidate grade provides a concrete anchor for assessment. [9, 29]. Independently, evidence from the LLM-as-a-judge and educational assessment literatures shows that aggregating multiple independent evaluators of the same candidate reduces individual-judge bias and improves agreement with expert ratings [55, 65, 52, 16].

Motivated by this, we first considered an A/B-style voting scheme that, given a grade range, argues for one grade against another. This approach is token-hungry: its token cost grows linearly with the number of comparisons, and it wastes pulls on clearly sub-optimal arms whose exploration cost is high. We therefore adopt UCB [27, 3], a well-studied bandit algorithm which, to our knowledge, has not previously been applied to mathematical grading. We compare them in the Appendix section N.

At the core of our algorithm is a judgment-defense setup. For each candidate grade $g \in \{ 0 , \ldots , 7 \}$ , we prompt an LLM to justify the grade and record its justification along with a self-reported confidence. In the next phase, the same model, now prompted as a judging oracle, selects arms and assigns rewards. This recovers the classic stochastic bandit setting. After each round, MABGrader aggregates the oracle’s rewards and updates the arm statistics, balancing exploration and exploitation to converge on the grade closest to the underlying true grade.

At round $t ,$ let $N _ { g } ( t )$ denote the number of times arm g has been pulled prior to round t, let $r _ { s } \in [ 0 , 1 ]$ denote the reward observed at pull s, and let

$$
\hat { \mu } _ { g } ( t ) \ = \ \frac { 1 } { N _ { g } ( t ) } \sum _ { s = 1 } ^ { t - 1 } r _ { s } \Im [ g _ { s } = g ] , \qquad N _ { g } ( t ) \ = \ \sum _ { s = 1 } ^ { t - 1 } \Im [ g _ { s } = g ]
$$

denote the empirical mean and pull count of arm g. UCB first plays each arm once, then at every subsequent round selects

$$
g _ { t } \ = \ \arg \operatorname* { m a x } _ { g \in \{ 0 , \ldots , 7 \} } \Big [ \underbrace { \hat { \mu } _ { g } ( t ) } _ { \mathrm { e x p l o i t a t i o n } } + \underbrace { c \sqrt { \frac { 2 \ln t } { N _ { g } ( t ) } } } _ { \mathrm { e x p l o r a t i o n } } \Big ] ,
$$

breaking ties uniformly at random. The default $c = 1$ recovers the classical UCB1 bound of [3], which yields $\mathcal { O } ( \sqrt { K T \ln T } )$ regret for $K = 8$ arms with rewards in [0, 1]; the oracle’s 1-5 fairness Likert is rescaled to [0, 1] via $r = ( \tilde { r } - 1 ) / 4$ . The parameter c is user-controlled and can be adapted to different problems. The design is compatible with any single-arm sampling bandit, including Exp3, ϵ-greedy, and Thompson Sampling. For this study we chose a single, well-established algorithm and leave a comparison with alternatives to future work. We provide more details in the Appendix Section I.

## 6 Experiments

## 6.1 Experimental Setup

Inference Settings. In all experiments, we fix the sampling parameters at $t o p _ { p } ~ = ~ 0 . 9 5$ and temperature t = 0.1 to minimize LLM-induced variance during the grading process. Unless otherwise noted, we employ the "thinking" versions of the evaluated models. For models that support adjustable reasoning effort, the setting was fixed to "high". Maximum token limits vary by model and are detailed in the Appendix. The default testing environment follows the MathArena template [4]. Every run was initialized with a seed of 42. We explicitly indicate where an alternative template was used for comparison.

All multi-sample experiments are performed at @16, where we sample 16 independent attempts from scratch. To evaluate Best-of-N methods, we utilize two Outcome Reward Models (ORMs): Skywork-Reward-Llama-3.1-8Bv0.2 [31] and ArmoRM-Llama3-8Bv0.1 [56]. We also incorporate two Process Reward Models (PRMs): math-shepherd-mistral-7b-prm [57] and RLHFlow/Llama3.1- 8B-PRMDeepseek-Data [14]. For our main MABGrader results we used 16 number of iterations and $c = 0 . 2 5$ guided by our ablation result.

Evaluation Metrics. We use seven metrics to measure alignment between verifier-generated grades and ground truth grades.

$$
\begin{array} { l l } { \mathrm { Q W K } = 1 - \displaystyle \frac { \sum _ { i , j } w _ { i , j } \partial _ { i , j } } { \sum _ { i , j } w _ { i , j } E _ { i , j } } , \quad } & { \mathrm { W T A } _ { \tau } = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ | y _ { i } - \hat { y } _ { i } | \leq \tau \} , \quad } \\ { \mathrm { M A E } = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | y _ { i } - \hat { y } _ { i } | , \quad } & { \mathrm { P e a r s o n } _ { r } = \displaystyle \frac { \sum _ { i } \left( y _ { i } - \bar { y } \right) \left( \hat { y } _ { i } - \bar { y } \right) } { \sqrt { \sum _ { i } \left( y _ { i } - \bar { y } \right) ^ { 2 } } \sqrt { \sum _ { i } \left( \hat { y } _ { i } - \bar { y } \right) ^ { 2 } } } , \quad } & { \mathrm { E M } = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ y _ { i } = \hat { y } _ { i } \} . } \end{array}
$$

Here, n is the number of grading samples; $y _ { \underline { { i } } }$ and $\hat { y } _ { i }$ the ground-truth and verifier-generated grades for the i-th sample, respectively; and y¯ and $\hat { y }$ their corresponding means. QWK measures ordinal agreement while penalizing larger grade discrepancies, with O and E denoting the observed and expected agreement matrices, respectively, and weights $w _ { i , j } = ( i - j ) ^ { 2 } / ( N - \overline { { 1 } } ) ^ { 2 }$ over the $N = 8$ possible grades. RMSE measures overall grading error, WTA<sub>τ</sub> measures the fraction of predictions within tolerance τ , Pearson r measures linear correlation, MAE measures average absolute grading error, and EM measures exact grade agreement. We also report Bias, defined as $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } ( { \widehat { y _ { i } } } - y _ { i } )$ with positive and negative values indicating over- and under-grading, respectively.

Data Generation. For synthetic data generation, we utilized Claude Opus 4.6 (Think) [1]. Because existing literature indicates a high self-validation bias [7], we exclude this model from our evaluation suite. We use a temperature of $t = 0 . 7$ for data generation and refinement, while the critic module employs a more conservative $t = 0 . 3$ to reduce variance in the feedback. We also finetuned two models, Qwen3.5-9B and 27B SFT, on Trojan-Bench(S). The model cards are specified in Appendix.

## 6.2 Analysis of Trojan-Bench(S)

LLM performance on the human-annotated benchmark. We present the performance of 14 models on the human-annotated benchmark using the QWK metric in Figure 4. In this section, we focus primarily on the correlation metric and provide a detailed breakdown of all other metrics in the Appendix Sections D, E and F. Our evaluation spans two scenarios, Blind and Full, and incorporates three sampling methods: CoT@1 (a single inference), MeanCoT@16 (the averaged grade rounded to the nearest integer), and MajorityVote@16 (selecting the grade with the highest frequency).

Consistent with the findings in [33], the Full setting is, on average, a less challenging task than the Blind setting. We observe that models achieve a 0.06 higher absolute correlation and a 24.2% relative improvement in performance when using MajorityVote@16. Interestingly, smaller models such as GPT-5-mini or Qwen3-8B achieve higher correlation when not provided with additional information. Upon further inspection, we found that these models are often unable to follow the gold solution and instead produce hallucinated responses. A deeper case analysis is provided in the Appendix Section J.1. Overall, the models remain below the 0.70 threshold even when supplied with full grading information and instructions, whereas expert annotators achieve a 0.76 agreement rate. Furthermore, LLM performance collapses to a maximum of 0.53 in the Blind setting. This scenario is particularly difficult as the models must both derive a correct solution and grade the attempt based on that derivation. We observe similar trends in RMSE, Pearson correlation, and Bias, with the full suite of results detailed in the Appendix Sections D, E, and F.

![](images/7862d44d2eaf7e32a0ea005f86c9e485a1ad4252e110b74e2c58b89c13a0244e.jpg)

![](images/f42fb4f048189a87f8a9f020a73bf2c053688f27e4adb7c60b34d69ca7fa6b92.jpg)

![](images/f2aa1aa893b323767cc2d9bd60b6e91b7253951fb07f22909a2b81c325eb9a62.jpg)  
Figure 4: Model correlation with human labels using QWK across Full and Blind settings. The bar plots report CoT@1, MeanCoT@16, and MajorityVote@16 for each tested model.  
Figure 5: Grading performance on human and synthetic USAMO 2025 subsets (N = 48). Reported values represent the outcome of MajorityVote@16 across QWK, RMSE, and Bias metrics.

Additionally, we conducted a comparative analysis of the different sampling methods. Because we utilized a lower temperature, the performance gap between single and multi-sampling is relatively small. However, consistent trends remain present in our evaluation. On average, MajorityVote outperforms other sampling techniques, though the primary benefit serves to stabilize the grading output. Specifically, under Full inference, MajorityVote scores 5.46% higher than CoT and 2.36% higher than MeanCoT.

Quality of the Synthetic Dataset. Furthermore, to evaluate the quality of the synthetic dataset, we provide two points of analysis. First, we measured the QWK, RMSE, and Bias on the USAMO 2025 subset of the synthetic data. We kept all sampling parameters consistent across both sets, with each containing 48 samples. We present these results in Figure 5. Our evaluation suggests that on the human-curated benchmark, models achieve a 0.07 lower absolute QWK, a 0.12 higher absolute RMSE, and a 0.04 higher absolute Bias compared to their synthetic counterparts. These results indicate that the human-crafted dataset presents a higher level of difficulty. While most models exhibit similar trends, we observe that DeepSeek-V4-Flash-Thinking shows a significantly lower correlation with synthetic data compared to the human-annotated version. When measuring the correlation of model responses between the expert-labeled and synthetic sets, we observe a Pearson r of 0.89.

Table 4: Comparative performance of base and fine-tuned models on TROJAN synthetic samples.
<table><tr><td></td><td colspan="3">Full Setting</td><td colspan="3">Blind Setting</td></tr><tr><td></td><td>QWK(↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td><td>QWK(↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td></tr><tr><td>Qwen3.5-9B (Base)</td><td>0.02</td><td>4.00</td><td>3.23</td><td>0.06</td><td>2.52</td><td>0.63</td></tr><tr><td>Qwen3.5-9B (SFT)</td><td>0.35</td><td>2.08</td><td>0.27</td><td>0.07</td><td>2.40</td><td>0.39</td></tr><tr><td>Qwen3.5-27B (Base)</td><td>0.36</td><td>2.98</td><td>1.54</td><td>-0.11</td><td>4.12</td><td>-3.40</td></tr><tr><td>Qwen3.5-27B (SFT)</td><td>0.47</td><td>2.10</td><td>0.77</td><td>0.06</td><td>2.50</td><td>0.21</td></tr></table>

Table 5: Sensitivity of QWK, RMSE, and Bias to $n _ { \mathrm { i t e r } }$ and $n _ { \mathrm { a l i g n } } .$
<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">QWK↑</td><td colspan="3">RMSE↓</td><td colspan="3">Bias</td></tr><tr><td>1</td><td>2</td><td>3</td><td>1</td><td>2</td><td>3</td><td>1</td><td>2</td><td>3</td></tr><tr><td rowspan="5">Niter</td><td>1</td><td>0.583</td><td></td><td></td><td>1.414</td><td></td><td></td><td>0.400</td><td></td><td></td></tr><tr><td>3</td><td>0.579</td><td>0.833</td><td>0.776</td><td>1.265</td><td>0.894</td><td>1.049</td><td>0.400</td><td>0.000</td><td>0.100</td></tr><tr><td>5</td><td>0.757</td><td>0.884</td><td>0.818</td><td>0.949</td><td>0.707</td><td>0.894</td><td>0.300</td><td>0.100</td><td>0.000</td></tr><tr><td>7</td><td>0.694</td><td>0.683</td><td>0.717</td><td>1.225</td><td>1.140</td><td>1.342</td><td>0.700</td><td>0.500</td><td>0.600</td></tr><tr><td>9</td><td>0.571</td><td>0.735</td><td>0.622</td><td>1.225</td><td>1.140</td><td>1.304</td><td>-0.300</td><td>0.100</td><td>0.700</td></tr></table>

Fine-Tuning on Trojan-Bench(S). Additionally, we performed SFT on the synthetic data and present our findings in Table 4. We supervised fine-tuned Qwen3.5-9B and 27B models [44] using LLaMA-Factory [66] on the full set of our synthetic data derived from four competition problems (see Section 4.2). The training mixture includes data from both the full and blind verification settings. The data were split into training and validation sets using a 95/5 ratio. The maximum context length was set to 16K tokens, which is sufficient for our task. The 9B model was fully fine-tuned, while the 27B model was fine-tuned with LoRA using rank-8 adapters. Both models were trained for 2 epochs with an effective global batch size of 16, bfloat16 precision, and a cosine learning-rate schedule with 10% warmup. The peak learning rate was 1e-5 for the full fine-tuning run and 1e-4 for the LoRA run. All participating models were run in "non-thinking" mode, which excludes the reasoning phase typically enclosed in <think> tokens.

We observe that after fine-tuning, smaller models demonstrate improved performance across the three primary metrics. Specifically, when trained on synthetic data, models achieve a higher correlation with human grading, a result that is consistent across both small and large model scales. Taken together, these findings suggest that synthetic data provides a high degree of correlation with human-generated labels and that training on this data effectively improves agreement with expert judgment.

TROJAN hyperparameter selection. To choose parameters for our synthetic pipeline generation, we ablated two main parameters. For each pair of parameters we asked human experts to grade 10 outputs and aggregated LLM-assigned score (according to a score profile) and human assigned scores. We report results in Table 5. We selected $n _ { i t e r } = 5$ and $n _ { a l i g . } = 2$ for the TROJAN pipeline as this configuration represents the global optimum across all primary evaluation metrics. This setting achieves the highest correlation with human labels $( \mathrm { Q W K } \bar { = } 0 . 8 8 \bar { 4 } )$ and the lowest error (RMSE = 0.707). While certain alternative configurations provide a zero bias, they do so at the expense of significantly lower correlation and higher error rates.

## 6.3 Performance of MABGrader

Comparison with baselines. Table 6 presents the performance of four models from different providers across the QWK, RMSE, and Bias metrics. To ensure a comprehensive evaluation, we compare MABGrader against three primary baselines: zero-shot CoT@1, MajorityVote@16 using three distinct prompt templates, and Best-of-N selection utilizing four reward models applied to the MathArena template outputs. Our results indicate that MABGrader consistently achieves the highest correlation with human experts, as evidenced by higher QWK scores across all tested models. While MABGrader significantly improves RMSE and Bias for Kimi-K2.5, Gemini-3-Pro, and Qwen3.5-27B, we observe that specific reward models such as RLHFlow and ArmoRM yield slightly better results for DeepSeek-V4-Flash in terms of error magnitude and systematic bias. Overall, the proposed method provides a more stable alignment with human grading across diverse model architectures.

Table 6: Performance on the manually annotated USAMO 2025 dataset in the Full setting, where models are given a reference solution and marking scheme. We report QWK, RMSE, and Bias for four base models: DeepSeek-V4-Flash, Kimi-K2.5, Gemini-3-Pro-Preview, and Qwen3.5-27B.
<table><tr><td rowspan="2"></td><td colspan="4">QWK (↑)</td><td colspan="4">RMSE (↓)</td><td colspan="4">Bias (≈ 0)</td></tr><tr><td>DS V4F</td><td>Kimi K2.5</td><td>Gem. 3P</td><td>Q3.5 27B</td><td>DS- V4F</td><td>Kimi K2.5</td><td>Gem. 3P</td><td>Q3.5 27B</td><td>DS- V4F</td><td>Kimi K2.5</td><td>Gem. 3P</td><td>Q3.5 27B</td></tr><tr><td colspan="10">Chain-of-Thought (CoT@ 1)</td><td></td><td></td><td></td><td></td></tr><tr><td>MathArena [4]</td><td>0.59</td><td>0.20</td><td>0.45</td><td>0.33</td><td>2.28</td><td>3.29</td><td>2.50</td><td>3.37</td><td>-0.69</td><td>2.24</td><td>1.35</td><td>1.73</td></tr><tr><td>ProofGr. [33]</td><td>0.64</td><td>0.24</td><td>0.46</td><td>0.36</td><td>2.15</td><td>3.06</td><td>2.51</td><td>3.10</td><td>-0.81</td><td>2.14</td><td>1.59</td><td>2.02</td></tr><tr><td>ProofAGr. [32]</td><td>0.52</td><td>0.19</td><td>0.42</td><td>0.32</td><td>2.47</td><td>3.47</td><td>2.84</td><td>3.60</td><td>-1.15</td><td>2.86</td><td>1.78</td><td>1.97</td></tr><tr><td colspan="10">Chain-of-Thought with Majority Vote Selection (MajorityVote@ 16)</td><td></td><td></td><td></td><td></td></tr><tr><td>MathArena [4]</td><td>0.64</td><td>0.28</td><td>0.45</td><td>0.30</td><td>2.13</td><td>3.05</td><td>2.50</td><td>3.04</td><td>-0.62</td><td>2.15</td><td>1.38</td><td>1.81</td></tr><tr><td>ProofGr. [33]</td><td>0.66</td><td>0.27</td><td>0.46</td><td>0.41</td><td>2.14</td><td>2.84</td><td>2.41</td><td>2.99</td><td>-0.71</td><td>2.21</td><td>1.33</td><td>1.89</td></tr><tr><td>ProofAGr. [32]</td><td>0.55</td><td>0.26</td><td>0.44</td><td>0.38</td><td>2.40</td><td>3.21</td><td>2.70</td><td>3.38</td><td>-1.06</td><td>2.34</td><td>1.49</td><td>1.93</td></tr><tr><td colspan="10">Chain-of-Thought with Best-of-N Selection (BoN@ 16)</td><td></td><td></td><td></td><td></td></tr><tr><td>Skywork [31]</td><td>0.61</td><td>0.20</td><td>0.41</td><td>0.29</td><td>2.16</td><td>3.33</td><td>2.64</td><td>3.11</td><td>-0.22</td><td>2.31</td><td>1.44</td><td>1.98</td></tr><tr><td>ArmoRM [56]</td><td>0.59</td><td>0.15</td><td>0.41</td><td>0.29</td><td>2.26</td><td>3.46</td><td>2.66</td><td>3.16</td><td>-0.19</td><td>2.35</td><td>1.58</td><td>2.04</td></tr><tr><td>RLHFlow [14]</td><td>0.60</td><td>0.15</td><td>0.50</td><td>0.27</td><td>2.02</td><td>3.19</td><td>2.41</td><td>3.18</td><td>-0.42</td><td>1.63</td><td>1.42</td><td>1.88</td></tr><tr><td>Shepherd [57]</td><td>0.61</td><td>0.12</td><td>0.45</td><td>0.31</td><td>2.14</td><td>3.40</td><td>2.49</td><td>2.95</td><td>-0.10</td><td>2.13</td><td>1.52</td><td>2.04</td></tr><tr><td colspan="10">Proposed Method (MABGrader@ 16)</td><td></td><td></td><td></td><td></td></tr><tr><td>MABGrader</td><td>0.66</td><td>0.35</td><td>0.63</td><td>0.57</td><td>2.06</td><td>2.65</td><td>1.96</td><td>2.15</td><td>-0.60</td><td>0.20</td><td>0.34</td><td>-0.98</td></tr></table>

## 7 Limitations and Future Work

Our evaluation pipeline depends on the availability of both a markscheme and a golden reference solution. This assumption is reasonable for Olympiad-level grading, where they are produced alongside the competition, but it does not hold for most real-world applications such as peer review of new theorems or open-ended scientific verification. Writing a high-quality markscheme without access to a reference solution is itself a difficult task and remains an open problem that our work does not address.

The same dependency limits MABGrader directly. Both the per-arm justifications and the oracle reward are framed against the reference proof, so the algorithm requires a markscheme that maps cleanly onto a golden solution. MABGrader therefore does not apply to the Blind setting, where there is no point of comparison for either the candidate proof or the per-arm justifications. Extending the algorithm to reference-free regimes, for instance by having the grader first synthesize an outline of the expected proof that the oracle can then evaluate against, is a natural direction for future work.

## 8 Conclusion

We conducted a systematic study of LLM-based verification for Olympiad-level mathematics. We introduced a human-curated benchmark derived from USAMO with markscheme-aligned injected errors and expert grades, and used it to evaluate 20 frontier models across prompting and inference settings. Our results show that current LLM verifiers still fall short of expert-level agreement. We proposed TROJAN for scalable adversarial proof generation to support evaluation and SFT and to improve grading alignment. Finally, we introduced MABGrader, a Multi-Armed Bandit method for verification that achieves the best overall performance, outperforming ProofGrader with 26.4% higher QWK and 14% less error. Our work suggests paths toward more reliable automated verification.

## Acknowledgments

The work of Farzan Farnia is supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China, Project 14210725, and is also supported by CUHK Direct Research Grant with CUHK Project No. 4055164. Moreover, the work is partially supported by a grant under 1+1+1 CUHK-CUHK(SZ)-GDSTC Joint Collaboration Fund. Finally, the authors sincerely thank the anonymous reviewers and metareviewer for their helpful feedback and constructive suggestions.

## References

[1] Anthropic. Claude 4.6 opus. https://www.anthropic.com/news/claude-opus-4-6, 2026.

[2] Anthropic. Introducing Claude Sonnet 4.6, February 2026.

[3] Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multi-armed bandit problem. Machine Learning, 2002.

[4] Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovic, and Martin T. Vechev. Matharena: Evaluating llms on uncontaminated math competitions. CoRR, abs/2505.23281, 2025.

[5] Yves Bertot and Pierre Castéran. Interactive theorem proving and program development: Coq’Art: the calculus ofinductive constructions. Springer Science & Business Media, 2013.

[6] Chenrui Cao, Liangcheng Song, Zenan Li, Xinyi Le, Xian Zhang, Hui Xue, and Fan Yang. Reviving dsp for advanced theorem proving in the era of reasoning models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[7] Wei-Lin Chen, Zhepei Wei, Xinyu Zhu, Shi Feng, and Yu Meng. Do llm evaluators prefer themselves for a reason?, 2025.

[8] Zhijun Chen, Xiaodong Lu, Jingzheng Li, Pengpeng Chen, Zhuoran Li, Kai Sun, Yuankai Luo, Qianren Mao, Ming Li, Likang Xiao, Dingqi Yang, Xiao Huang, Yikun Ban, Hailong Sun, and Philip S. Yu. Harnessing multiple large language models: A survey on llm ensemble, 2026.

[9] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.

[10] American Mathematics Competitions and Art of Problem Solving. Aime problems and solutions — 2025. https://artofproblemsolving.com/wiki/index.php/AIME\_Problems\_and\_ Solutions, 2025.

[11] DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, April 2026.

[12] DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangqi Zhou, Fangyun Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoran Wei, Haowei Zhang, Haowen Luo, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, Jialiang Huang, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jingchang Chen, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jun Ran, Junguang Jiang, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexin Huang, Kexing Zhou, Kezhao Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Liang Zhao, Liangsheng Yin, Lihua Guo, Lingxiao Luo, Linwang Ma, Litong Wang, Liyue Zhang, M. S. Di, M. Y Xu, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Panpan Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S. H. Liu,

Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Songyang Zhou, Tao Ni, Tao Yun, Tian Pei, Tian Ye, Tianyuan Yue, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjun Gao, Wentao Zhang, Xi Gao, Xiangwen Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyuan Li, Xu Chen, Xuecheng Su, Xuehai Pan, Xuheng Lin, Xuwei Fu, Y. Q. Wang, Yang Zhang, Yanhong Xu, Yanru Ma, Yao Li, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yiliang Xiong, Ying He, Ying Zhou, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuduan Wang, Yue Gong, Yuhan Wu, Yuheng Zou, Yukun Li, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhiyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao, Bei Feng, Hui Li, J. L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R. J. Chen, R. L. Jin, S. S. Li, Shuang Zhou, Tianyu Sun, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xinnan Song, Xinyi Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, Dongjie Ji, Jian Liang, Jianzhong Guo, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Ruyi Chen, Shangmian Sun, Shaoqing Wu, Shengfeng Ye, T. Wang, W. L. Xiao, Wei An, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Ying Tang, Yukun Zha, Zekai Zhang, Zhe Ju, Zhen Zhang, and Zihua Qu. Deepseek-v3.2: Pushing the frontier of open large language models, 2025.

[13] Jasper Dekoninck, Ivo Petrov, Kristian Minchev, Miroslav Marinov, Maria Drencheva, Lyuba Konova, Milen Milenov Shumanov, Kaloyan Tsvetkov, Nikolay Drenchev, Lazar D. Todorov, Kalina Nikolova, Nikolay Georgiev, Vanesa Kalinkova, Margulan Ismoldayev, Mislav Balunovic, and Martin Vechev. The open proof corpus: A large-scale study of LLM-generated mathematical proofs. In The Fourteenth International Conference on Learning Representations, 2026.

[14] Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

[15] Gheorghe Comanici et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025.

[16] Nancy Falchikov and Judy Goldfinch. Student peer assessment in higher education: A metaanalysis comparing peer and teacher marks. Review of Educational Research, 70(3):287–322, 2000.

[17] Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios Nikolas Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. How to evaluate reward models for RLHF. In The Thirteenth International Conference on Learning Representations, 2025.

[18] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models, October 2024. arXiv:2410.07985.

[19] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets. Communications ofthe ACM, 64(12):86–92, 2021.

[20] Google. A new era of intelligence with gemini 3, November 2025.

[21] Google DeepMind. AI achieves silver-medal standard solving international mathematical olympiad problems. https://deepmind.google/discover/blog/ ai-solves-imo-problems-at-silver-medal-level/, 2024. Accessed: 2025-05- 08.

[22] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on llm-as-a-judge, 2025.

[23] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.

[24] Yichen Huang and Lin F. Yang. Winning gold at IMO 2025 with a model-agnostic verificationand-refinement pipeline. In The 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025, 2025.

[25] Albert Qiaochu Jiang, Sean Welleck, Jin Peng Zhou, Wenda Li, Jiacheng Liu, Mateja Jamnik, Timoth’ee Lacroix, Yuhuai Wu, and Guillaume Lample. Draft, Sketch, and Prove: Guiding formal theorem provers with informal proofs. In International Conference on Learning Representations, 2023.

[26] Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo. Prometheus 2: An open source language model specialized in evaluating other language models, 2024.

[27] T.L Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4–22, 1985.

[28] Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. In Forty-second International Conference on Machine Learning, 2025.

[29] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024.

[30] Yong Lin, Shange Tang, Bohan Lyu, Ziran Yang, Jui-Hui Chung, Haoyu Zhao, Lai Jiang, Yihan Geng, Jiawei Ge, Jingruo Sun, Jiayun Wu, Jiri Gesi, Ximing Lu, David Acuna, Kaiyu Yang, Hongzhou Lin, Yejin Choi, Danqi Chen, Sanjeev Arora, and Chi Jin. Goedel-prover-v2: Scaling formal theorem proving with scaffolded data synthesis and self-correction. CoRR, abs/2508.03613, 2025.

[31] Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. CoRR, abs/2410.18451, 2024.

[32] Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, and Junehyuk Jung. Towards robust mathematical reasoning. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 35418–35442. Association for Computational Linguistics, 2025.

[33] Wenjie Ma, Andrei Cojocaru, Neel Kolhe, Bradley Louie, Robin Said Sharif, Haihan Zhang, Vincent Zhuang, Matei Zaharia, and Sewon Min. Reliable fine-grained evaluation of natural language math proofs. CoRR, abs/2510.13888, 2025.

[34] Leonardo de Moura and Sebastian Ullrich. The lean 4 theorem prover and programming language. In Automated Deduction – CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings, page 625–635, Berlin, Heidelberg, 2021. Springer-Verlag.

[35] Tobias Nipkow, Lawrence C Paulson, and Markus Wenzel. Isabelle/HOL: a proof assistant for higher-order logic, volume 2283. Springer Science & Business Media, 2002.

[36] OpenAI. Gpt-4o system card, 2024.

[37] OpenAI. Openai gpt-5 system card, 2026.

[38] Azim Ospanov, Farzan Farnia, and Roozbeh Yousefzadeh. Apollo: Automated llm and lean collaboration for advanced formal reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[39] Azim Ospanov, Farzan Farnia, and Roozbeh Yousefzadeh. minif2f-lean revisited: Reviewing limitations and charting a path forward. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[40] Azim Ospanov, Zijin Feng, Jiacheng Sun, Haoli Bai, Xin Shen, and Farzan Farnia. Hermes: Towards efficient and verifiable mathematical reasoning in llms, 2025.

[41] Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. OpenWebMath: An open dataset of high-quality mathematical web text. In The Twelfth International Conference on Learning Representations, 2023.

[42] Ivo Petrov, Jasper Dekoninck, Lyuben Baltadzhiev, Maria Drencheva, Kristian Minchev, Mislav Balunovic, Nikola Jovanovic, and Martin Vechev. Proof or bluff? evaluating LLMs on 2025´ USA math olympiad. In 2nd AI for Math Workshop @ ICML 2025, 2025.

[43] Ivo Petrov, Jasper Dekoninck, and Martin Vechev. Brokenmath: A benchmark for sycophancy in theorem proving with llms, 2025.

[44] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.

[45] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

[46] James V. Roggeveen, Erik Y. Wang, Will Flintoft, Peter Donets, Lucy S. Nathwani, Nickholas Gutierrez, David Ettel, Anton Marius Graf, Siddharth Dandavate, Arjun Nageswaran, Raglan Ward, Ava Williamson, Anne Mykland, Kacper K. Migacz, Yijun Wang, Egemen Bostan, Duy Thuc Nguyen, Zhe He, Marc L. Descoteaux, Felix Yeung, Shida Liu, Jorge García Ponce, Luke Zhu, Yuyang Chen, Ekaterina S. Ivshina, Miguel Fernandez, Minjae Kim, Kennan Gumbs, Matthew Scott Tan, Russell Yang, Mai Hoang, David Brown, Isabella A. Silveira, Lavon Sykes, Ahmed Roman, William Fredenberg, Yiming Chen, Lucas Martin, Yixing Tang, Kelly Werker Smith, Hongyu Liao, Logan G. Wilson, Alexander Dazhen Cai, Andrea Elizabeth Biju, and Michael P. Brenner. Hardmath2: A benchmark for applied mathematics built by students as part of a graduate class. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[47] Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning. CoRR, abs/2511.22570, 2025.

[48] Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Yuan Tang, Alejandro Cuadron, Chenguang Wang, Raluca Popa, and Ion Stoica. Judgebench: A benchmark for evaluating LLM-based judges. In The Thirteenth International Conference on Learning Representations, 2025.

[49] Zhengyang Tang, Xingxing Zhang, Benyou Wang, and Furu Wei. Mathscale: Scaling instruction tuning for mathematical reasoning. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024.

[50] Kimi Team. Kimi k2.5: Visual agentic intelligence, 2026.

[51] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026.

[52] Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, and Dieuwke Hupkes. Judging the judges: Evaluating alignment and vulnerabilities in llms-asjudges, 2025.

[53] Trieu Trinh, Yuhuai Wu, Quoc Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 2024.

[54] George Tsoukalas, Jasper Lee, John Jennings, Jimmy Xin, Michelle Ding, Michael Jennings, Amitayush Thakur, and Swarat Chaudhuri. Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition, 2024.

[55] Pat Verga, Sebastian Hofstatter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, and Patrick Lewis. Replacing judges with juries: Evaluating llm generations with a panel of diverse models, 2024.

[56] Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, pages 10582–10592. Association for Computational Linguistics, 2024.

[57] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 9426–9439. Association for Computational Linguistics, 2024.

[58] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.

[59] Sean Welleck, Jiacheng Liu, Ximing Lu, Hannaneh Hajishirzi, and Yejin Choi. Naturalprover: Grounded mathematical proof generation with language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.

[60] Scott Wood, Erin Yao, Lisa Haisfield, and Susan Lottridge. Establishing standards of best practice in automated scoring. Technical report, ACT, Inc., July 2021.

[61] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.

[62] Roozbeh Yousefzadeh, Xuenan Cao, and Azim Ospanov. A lean dataset for international math olympiad: Small steps towards writing math proofs for hard problems, 2025.

[63] Albert S. Yue, Lovish Madaan, Ted Moskovitz, DJ Strouse, and Aaditya K. Singh. Harp: A challenging human-annotated math reasoning benchmark, 2024.

[64] Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. miniF2F: a cross-system benchmark for formal Olympiad-level mathematics, 2022.

[65] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

[66] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics.

## Contents

A Datasheets 18   
B Distribution of injected errors in Trojan-Bench(H) 24   
C Evaluated Model List 24   
D Extended evaluation results 24   
D.1 Trojan-Bench(H) 27   
D.2 Trojan-Bench(S) 31   
D.3 Additional evaluation of BoN methods . 36   
E Extended Bar plots 37   
F Evaluation of LLM correlation with human and other LLM annotators 48   
G Annotation procedure 65   
H Additional details on SFT training 65   
I Hyperparameter c ablation in MABGrader 67   
J Case studies 68   
J.1 Behavior of smaller models on Full and Blind settings . 68   
J.2 Effect of MABGrader on grading procedures 72   
K Inclusion of USAMO 2025 samples in the training set 74   
L Effect of changing the generator model in TROJAN pipeline 74   
M Benchmarking TROJAN SFT pipeline against ProofBench 74   
N Effect of Multi-Armed Bandit algorithm on grading 76   
O Prompt templates 77

## A Datasheets

Following the framework of [19] and [41], Table 7 provides the additional information about our expert-annotated dataset and Table 8 provides additional information about our synthetically generated dataset.

Table 7: Datasheet for our expert annotated dataset
<table><tr><td colspan="1" rowspan="1">Questions</td><td colspan="1" rowspan="1">Answers</td></tr><tr><td colspan="2" rowspan="1">Motivation</td></tr><tr><td colspan="1" rowspan="1">For what purpose was the dataset created?</td><td colspan="1" rowspan="1">The dataset was developed to benchmark theperformance of current Large Language Mod-els in grading and evaluating Olympiad-levelmathematical proofs.</td></tr><tr><td colspan="1" rowspan="1">Who created the dataset and on behalf of whichentity?</td><td colspan="1" rowspan="1">The authors of this paper.</td></tr><tr><td colspan="1" rowspan="1">Who funded the creation of the dataset?</td><td colspan="1" rowspan="1">The company where the authors work.</td></tr><tr><td colspan="1" rowspan="1">Any other comment?</td><td colspan="1" rowspan="1">None.</td></tr><tr><td colspan="2" rowspan="1">Composition</td></tr><tr><td colspan="1" rowspan="1">What do the instances that comprise the datasetrepresent?</td><td colspan="1" rowspan="1">USAMO 2025 problems that consist of prob-lem statements, golden solutions, correspond-ing markschemes, candidate proofs with mi-nor errors and corresponding expert assignedgrades.</td></tr><tr><td colspan="1" rowspan="1">How many instances are there in total?</td><td colspan="1" rowspan="1">48 instances</td></tr><tr><td colspan="1" rowspan="1">Does the dataset contain all possible instancesor is it a sample of instances from a larger set?</td><td colspan="1" rowspan="1">Yes, we present all instances.</td></tr><tr><td colspan="1" rowspan="1">What data does each instance consist of?</td><td colspan="1" rowspan="1">A problem statement, golden solution, corre-sponding markscheme, candidate solution withlogical errors and corresponding expert grades.</td></tr><tr><td colspan="1" rowspan="1">Is there a label or target associated with eachinstance?</td><td colspan="1" rowspan="1">Only human assigned grades.</td></tr><tr><td colspan="1" rowspan="1">Is any information missing from individual in-stances?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Are relationships between individual instancesmade explicit?</td><td colspan="1" rowspan="1">Not applicable.</td></tr><tr><td colspan="1" rowspan="1">Are there recommended data splits?</td><td colspan="1" rowspan="1">No recommended split.</td></tr><tr><td colspan="1" rowspan="1">Are there any errors, sources of noise, or redun-dancies in the dataset?</td><td colspan="1" rowspan="1">No, there are no errors in the proposed datasetto the best of our knowledge. Verification wasdone by two independent annotators.</td></tr><tr><td colspan="1" rowspan="1">Is the dataset self-contained, or does it link toor otherwise rely on external resources?</td><td colspan="1" rowspan="1">The dataset is self-contained.</td></tr><tr><td colspan="1" rowspan="1">Does the dataset contain data that might beconsidered confidential?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Does the dataset contain data that, if vieweddirectly, might be offensive, insulting, threaten-ing, or might otherwise cause anxiety?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="2" rowspan="1">Collection Process</td></tr><tr><td colspan="1" rowspan="1">How was the data associated with each instanceacquired?</td><td colspan="1" rowspan="1">All 48 instances originate from our team andfurther verified using two-step collection de-scribed in [45].</td></tr><tr><td colspan="2" rowspan="1">Continued on next page</td></tr><tr><td colspan="1" rowspan="1">What mechanisms or procedures were used tocollect the data?</td><td colspan="1" rowspan="1">The seed problems were taken from USAMO2025 subset of the MathArena repository [4]Based on corresponding markschemes, we in-jected mistakes that would result in a particularloss of points.</td></tr><tr><td colspan="1" rowspan="1">If the dataset is a sample from a larger set, whatwas the sampling strategy?</td><td colspan="1" rowspan="1">No, there is no larger version.</td></tr><tr><td colspan="1" rowspan="1">Who was involved in the data collection pro-cess and how were they compensated?</td><td colspan="1" rowspan="1">Authors of this work are responsible for datacollection.</td></tr><tr><td colspan="1" rowspan="1">Over what time frame was the data collected?</td><td colspan="1" rowspan="1">The dataset was collected over a four-monthperiod, from January 2026 through April 2026.</td></tr><tr><td colspan="1" rowspan="1">Were any ethical review processes conducted?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Preprocessing/cl</td><td colspan="1" rowspan="1">eaning/labeling</td></tr><tr><td colspan="1" rowspan="1">Was any preprocessing/cleaning/labeling of thedata done?</td><td colspan="1" rowspan="1">Yes, we manually assigned grades to each sam-ple and preprocessed the data to follow Math-Arena [4] format.</td></tr><tr><td colspan="1" rowspan="1">Was the “raw"data saved in addition to thepreprocessed/cleaned/labeled data?</td><td colspan="1" rowspan="1">No, all raw data was converted to preprocessedsamples.</td></tr><tr><td colspan="1" rowspan="1">Is the software that was used to preprocess/-clean/label the data available?</td><td colspan="1" rowspan="1">We used open-source text editors to performannotation.</td></tr><tr><td colspan="1" rowspan="1">Any other comments?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="2" rowspan="1">Uses</td></tr><tr><td colspan="1" rowspan="1">Has the dataset been used for any tasks already?</td><td colspan="1" rowspan="1">We evaluated a large number of LLMs andmeasured their correlation with human labels.Moreover, we tested our proposed MABGraderagent.</td></tr><tr><td colspan="1" rowspan="1">Is there a repository that links to any or allpapers or systems that use the dataset?</td><td colspan="1" rowspan="1">Public GitHub and HuggingFace links will beprovided at a later date.</td></tr><tr><td colspan="1" rowspan="1">What (other) tasks could the dataset be usedfor?</td><td colspan="1" rowspan="1">The dataset may be used for benchmarkingalignment with human grades in LLM-as-a-Judge scenarios.</td></tr><tr><td colspan="1" rowspan="1">Is there anything about the composition of thedataset or the way it was collected and prepro-cessed/cleaned/labeled that might impact futureuses?</td><td colspan="1" rowspan="1">We do not anticipate future issues emergingfrom the proposed benchmark dataset.</td></tr><tr><td colspan="1" rowspan="1">Are there tasks for which the dataset should notbe used?</td><td colspan="1" rowspan="1">The data is primarily for benchmarking, so wediscourage others from training their modelson this dataset.</td></tr><tr><td colspan="1" rowspan="1">Any other comments?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Distri</td><td colspan="1" rowspan="1">bution</td></tr><tr><td colspan="1" rowspan="1">Will the dataset be distributed to third partiesoutside of the entity on behalf of which thedataset was created?</td><td colspan="1" rowspan="1">Yes, we will release the dataset on public plat-forms such as GitHub and HuggingFace at alater date.</td></tr><tr><td colspan="1" rowspan="1">How will the dataset will be distributed?</td><td colspan="1" rowspan="1">GitHub, HuggingFace.</td></tr><tr><td colspan="1" rowspan="1">When will the dataset be distributed?</td><td colspan="1" rowspan="1">The dataset will be distributed along with thecamera-ready version of the submission.</td></tr><tr><td colspan="1" rowspan="1">Will the dataset be distributed under a copyrightor other intellectual property license, and/orunder applicable terms of use?</td><td colspan="1" rowspan="1">Yes, the dataset will be released under the MITlicense.</td></tr><tr><td colspan="2" rowspan="1">Continued on next page</td></tr><tr><td colspan="1" rowspan="1">Have any third parties imposed IP-based orother restrictions on the data associated withthe instances?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Do any export controls or other regulatory re-strictions apply to the dataset or to individualinstances?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Any other comments?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="2" rowspan="1">Maintenance</td></tr><tr><td colspan="1" rowspan="1">Who will be supporting/hosting/maintainingthe dataset?</td><td colspan="1" rowspan="1">The first author of the submission.</td></tr><tr><td colspan="1" rowspan="1">How can the owner/curator/manager of thedataset be contacted?</td><td colspan="1" rowspan="1">The curator of the dataset may be reachedthrough an email or any other public means,such as GitHub profile.</td></tr><tr><td colspan="1" rowspan="1">Is there an erratum?</td><td colspan="1" rowspan="1">At this stage there is no associated erratum.</td></tr><tr><td colspan="1" rowspan="1">Will the dataset be updated?</td><td colspan="1" rowspan="1">Yes, we plan to periodically update the datasetif mistakes are found.</td></tr><tr><td colspan="1" rowspan="1">If the dataset relates to people, are there applica-ble limits on the retention of the data associatedwith the instances?</td><td colspan="1" rowspan="1">Not applicable.</td></tr><tr><td colspan="1" rowspan="1">Will older versions of the dataset continue tobe supported/hosted/maintained?</td><td colspan="1" rowspan="1">Yes.</td></tr><tr><td colspan="1" rowspan="1">If others want to extend/augment/build on/con-tribute to the dataset, is there a mechanism forthem to do so?</td><td colspan="1" rowspan="1">Yes, since we release all accompanied informa-tion, others may use this data for other uses.</td></tr><tr><td colspan="1" rowspan="1">Any other comments?</td><td colspan="1" rowspan="1">No.</td></tr></table>

Table 8: Datasheet for our synthetically generated dataset using TROJAN.
<table><tr><td>Questions</td><td>Answers</td></tr><tr><td colspan="2">Motivation</td></tr><tr><td>For what purpose was the dataset created?</td><td>The dataset was developed to benchmark the performance of current Large Language Mod- els in grading and evaluating Olympiad-level</td></tr><tr><td>Who created the dataset and on behalf of which entity?</td><td>mathematical proofs. The authors of this paper.</td></tr><tr><td>Who funded the creation of the dataset?</td><td>The company where the authors work.</td></tr><tr><td>Any other comment?</td><td>None.</td></tr><tr><td colspan="2">Composition</td></tr><tr><td>What do the instances that comprise the dataset represent?</td><td>USAMO 2025/USAMO 2026/IMO 2025/IMC 2025 problems that consist of problem statements, golden solutions, corresponding markschemes, candidate proofs with minor er-</td></tr><tr><td>How many instances are there in total?</td><td>rors and corresponding expert assigned grades 200 instances</td></tr><tr><td>Does the dataset contain all possible instances or is it a sample of instances from a larger set?</td><td>Yes, we present all instances.</td></tr><tr><td>What data does each instance consist of?</td><td>A problem statement, golden solution, corre- sponding markscheme, candidate solution with logical errors and corresponding expert grades</td></tr><tr><td>Is there a label or target associated with each instance?</td><td>Only LLM assigned target grades as described in the score profile.</td></tr><tr><td>Is any information missing from individual in- stances?</td><td>No.</td></tr><tr><td>Are relationships between individual instances made explicit?</td><td>Not applicable.</td></tr><tr><td>Are there recommended data splits? Are there any errors, sources of noise, or redun- dancies in the dataset?</td><td>No recommended split. No, there are no errors in the proposed dataset to the best of our knowledge. Verification was</td></tr><tr><td>Is the dataset self-contained, or does it link to or otherwise rely on external resources?</td><td>done by a critic LLM, namely Claude-Opus- 4.6-Think. The dataset is self-contained.</td></tr><tr><td>Does the dataset contain data that might be considered confidential?</td><td>No.</td></tr><tr><td>Does the dataset contain data that, if viewed directly, might be offensive, insulting, threaten-</td><td>No.</td></tr><tr><td>ing, or might otherwise cause anxiety? Collection Process</td><td></td></tr><tr><td colspan="2">How was the data associated with each instance All 200 instances originate from our team and acquired? further verified using critique-alignment pro- cess of our TROJAN pipeline. Continued on next page</td></tr><tr><td colspan="1" rowspan="1">What mechanisms or procedures were used tocollect the data?</td><td colspan="1" rowspan="1">The seed problems were taken from USAMO2025/USÁMO 2026/IMO 2025/IMC 2025 sub-sets of the MathArena repository [4]. Basedon corresponding markschemes, LLM injectedmistakes that would result in a particular lossof points as described in a given score profile.</td></tr><tr><td colspan="1" rowspan="1">If the dataset is a sample from a larger set, whatwas the sampling strategy?</td><td colspan="1" rowspan="1">No, there is no larger version.</td></tr><tr><td colspan="1" rowspan="1">Who was involved in the data collection pro-cess and how were they compensated?</td><td colspan="1" rowspan="1">Authors of this work are responsible for datacollection.</td></tr><tr><td colspan="1" rowspan="1">Over what time frame was the data collected?</td><td colspan="1" rowspan="1">The dataset was collected over a four-monthperiod, from January 2026 through April 2026.</td></tr><tr><td colspan="1" rowspan="1">Were any ethical review processes conducted?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Preprocessing/cl</td><td colspan="1" rowspan="1">eaning/labeling</td></tr><tr><td colspan="1" rowspan="1">Was any preprocessing/cleaning/labeling of thedata done?</td><td colspan="1" rowspan="1">Yes, we manually assigned grades to each sam-ple and preprocessed the data to follow Math-Arena [4] format.</td></tr><tr><td colspan="1" rowspan="1">Was the “raw"data saved in addition to thepreprocessed/cleaned/labeled data?</td><td colspan="1" rowspan="1">No, all raw data was converted to preprocessedsamples.</td></tr><tr><td colspan="1" rowspan="1">Is the software that was used to preprocess/-clean/label the data available?</td><td colspan="1" rowspan="1">We used open-source text editors to performannotation.</td></tr><tr><td colspan="1" rowspan="1">Any other comments?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Use</td><td colspan="1" rowspan="1">s</td></tr><tr><td colspan="1" rowspan="1">Has the dataset been used for any tasks already?</td><td colspan="1" rowspan="1">We evaluated a large number of LLMs and mea-sured their correlation with LLM-assigned la-bels. Moreover, we tested our proposed MAB-Grader agent.</td></tr><tr><td colspan="1" rowspan="1">Is there a repository that links to any or allpapers or systems that use the dataset?</td><td colspan="1" rowspan="1">Public GitHub and HuggingFace links will beprovided at a later date.</td></tr><tr><td colspan="1" rowspan="1">What (other) tasks could the dataset be usedfor?</td><td colspan="1" rowspan="1">The dataset may be used for benchmarkingalignment with human grades in LLM-as-a-Judge scenarios.</td></tr><tr><td colspan="1" rowspan="1">Is there anything about the composition of thedataset or the way it was collected and prepro-cessed/cleaned/labeled that might impact futureuses?</td><td colspan="1" rowspan="1">We do not anticipate future issues emergingfrom the proposed benchmark dataset.</td></tr><tr><td colspan="1" rowspan="1">Are there tasks for which the dataset should notbe used?</td><td colspan="1" rowspan="1">The data is for benchmarking and may be usedfor training. At this stage we do not see scenar-ios for which we discourage others to use thisdata.</td></tr><tr><td colspan="1" rowspan="1">Any other comments?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="2" rowspan="1">Distribution</td></tr><tr><td colspan="1" rowspan="1">Will the dataset be distributed to third partiesoutside of the entity on behalf of which thedataset was created?</td><td colspan="1" rowspan="1">Yes, we will release the dataset on public plat-forms such as GitHub and HuggingFace at alater date.</td></tr><tr><td colspan="1" rowspan="1">How will the dataset will be distributed?</td><td colspan="1" rowspan="1">GitHub, HuggingFace.</td></tr><tr><td colspan="1" rowspan="1">When will the dataset be distributed?</td><td colspan="1" rowspan="1">The dataset will be distributed along with thecamera-ready version of the submission</td></tr><tr><td colspan="2" rowspan="1">Continued on next page</td></tr><tr><td colspan="1" rowspan="1">Will the dataset be distributed under a copyrightor other intellectual property license, and/orunder applicable terms of use?</td><td colspan="1" rowspan="1">Yes, the dataset will be released under the MITlicense.</td></tr><tr><td colspan="1" rowspan="1">Have any third parties imposed IP-based orother restrictions on the data associated withthe instances?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Do any export controls or other regulatory re-strictions apply to the dataset or to individualinstances?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="1" rowspan="1">Any other comments?</td><td colspan="1" rowspan="1">No.</td></tr><tr><td colspan="2" rowspan="1">Maintenance</td></tr><tr><td colspan="1" rowspan="1">Who will be supporting/hosting/maintainingthe dataset?</td><td colspan="1" rowspan="1">The first author of the submission.</td></tr><tr><td colspan="1" rowspan="1">How can the owner/curator/manager of thedataset be contacted?</td><td colspan="1" rowspan="1">The curator of the dataset may be reachedthrough an email or any other public means,such as GitHub profile.</td></tr><tr><td colspan="1" rowspan="1">Is there an erratum?</td><td colspan="1" rowspan="1">At this stage there is no associated erratum.</td></tr><tr><td colspan="1" rowspan="1">Will the dataset be updated?</td><td colspan="1" rowspan="1">Yes, we plan to periodically update the datasetif mistakes are found.</td></tr><tr><td colspan="1" rowspan="1">If the dataset relates to people, are there applica-ble limits on the retention of the data associatedwith the instances?</td><td colspan="1" rowspan="1">Not applicable.</td></tr><tr><td colspan="1" rowspan="1">Will older versions of the dataset continue tobe supported/hosted/maintained?</td><td colspan="1" rowspan="1">Yes.</td></tr><tr><td colspan="1" rowspan="1">If others want to extend/augment/build on/con-tribute to the dataset, is there a mechanism forthem to do so?</td><td colspan="1" rowspan="1">Yes, since we release all accompanied informa-tion, others may use this data for other uses.</td></tr><tr><td colspan="1" rowspan="1">Any other comments?</td><td colspan="1" rowspan="1">No.</td></tr></table>

## B Distribution of injected errors in Trojan-Bench(H)

Table 9 shows the distribution of injected errors in the Trojan-Bench(H) dataset. During construction, we apply logical gap errors the most, since they are easier to control and spot right away. Moreover, a logical gap is the widest error category that is difficult to define with one term, so many unjustified claims go under this broad umbrella category. "Misapplied theorem" refers to errors with incorrect preliminaries or even a completely wrong theorem used to prove the statement. Conceptual errors refer to fundamental issues with the proof, i.e., wrong constructions or even interpretations of the problem. Computational slips are very minor mistakes that do not affect the grand scheme of things, but still lead to less rigorous proofs. "Incomplete justification" is a mistake in which the proof is only partially complete and technically correct, but fails to explain all conditions/steps.

Table 9: Distribution of injected error types in Trojan-Bench(H). We report them as absolute counts and percentages.
<table><tr><td>Error Type</td><td>Count</td><td>Percentage</td></tr><tr><td>Logical gap</td><td>27</td><td>35.06%</td></tr><tr><td>Misapplied theorem</td><td>18</td><td>23.38%</td></tr><tr><td>Conceptual error</td><td>12</td><td>15.58%</td></tr><tr><td>Computational slip</td><td>8</td><td>10.39%</td></tr><tr><td>Incomplete justification</td><td>6</td><td>7.79%</td></tr><tr><td>None</td><td>6</td><td>7.79%</td></tr><tr><td>Total</td><td>77</td><td>100.0%</td></tr></table>

## C Evaluated Model List

Table 10 summarises every grader that participates in our evaluation suite. The list spans 20 distinct models from major providers, covering both open-weight and closed-weight families together with their thinking-enabled and non-thinking variants where available. Each row reports which (dataset, prompting style) combinations a given model was run on. The two datasets are USAMO 2025 (the human-annotated benchmark, Trojan-Bench(H)) and the synthetic set (Trojan-Bench(S)), and the two styles are the rubric-equipped Full template and the rubric-free Blind template. The final column lists the maximum completion-token budget observed across attempts, which clarifies the inference-cos regime each model operated in. We observe that models rarely hit the maximum token limit.

In total, the table tracks 52 model×(dataset, style) configurations and serves as an index for the per-model results that follow later in this appendix. Coverage is not uniform across providers. Smaller open-weight reward-style models such as the Prometheus family were evaluated on Trojan-Bench(H) Full only, since they did not produce usable scores in the rubric-free regime. Frontier thinking-enabled models were evaluated more broadly because their scoring behaviour is the primary subject of our analysis.

## D Extended evaluation results

This section reports the full set of correlation and error metrics for every grader in Table 10, broken down by inference style (Full and Blind), dataset (Trojan-Bench(H) and Trojan-Bench(S)), and aggregation method (single CoT, Mean-CoT@16, and MajorityVote@16). The first two tables of each subsection report the mean and standard deviation of every metric across the 16 sampling attempts at the per-attempt level, which characterises the within-grader variance. The remaining six tables report the aggregated grade under the three aggregation methods, which is the quantity each grader exposes to a downstream user.

Table 10: The list of all 20 evaluated models. A ✓ marks the presence of a sampling\_grades.csvproducing run for the indicated (dataset, style) pair. In total 52 model×(dataset, style) runs are reported. The last column reports the maximum completion\_tokens.
<table><tr><td rowspan="2">Model</td><td colspan="2">Trojan-Bench(H)</td><td colspan="2">Trojan-Bench(S)</td><td rowspan="2">Max tokens</td></tr><tr><td>Full</td><td>Blind</td><td>Full</td><td>Blind</td></tr><tr><td>Claude-Sonnet-4.6 [2] (T)</td><td>√</td><td>√</td><td>一</td><td>一</td><td>65,536</td></tr><tr><td>Claude-Sonnet-4.6 [2] (T)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>23,625</td></tr><tr><td>DeepSeek-V3.2 [12] (T)</td><td>√</td><td>√</td><td>√</td><td>一</td><td>8,192</td></tr><tr><td>DeepSeek-V3.2 [12] (T)</td><td>√</td><td>√</td><td>√</td><td>一</td><td>65,536</td></tr><tr><td>DeepSeek-V4-Flash [11] (T)</td><td>√</td><td>√</td><td>一</td><td>一</td><td>65,536</td></tr><tr><td>DeepSeek-V4-Flash [11] (T)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>65,536</td></tr><tr><td>DeepSeek-V4-Pro [11] (T)</td><td>√</td><td>一</td><td>一</td><td>一</td><td>65,536</td></tr><tr><td>Gemini-2.5-Flash [15]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>32,768</td></tr><tr><td>Gemini-2.5-Pro [15] (T)</td><td>√</td><td>一</td><td>一</td><td>一</td><td>65,536</td></tr><tr><td>Gemini-2.5-Pro [15] (T)</td><td>√</td><td>一</td><td>一</td><td>一</td><td>65,536</td></tr><tr><td>Gemini-3-Pro-Prev. [20] (T)</td><td>√</td><td>√</td><td>一</td><td>一</td><td>65,536</td></tr><tr><td>GPT-4o [36]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>4,096</td></tr><tr><td>GPT-5 [37]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>65,536</td></tr><tr><td>GPT-5-Mini [37]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>32,768</td></tr><tr><td>Kimi-K2.5 [50]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>32,768</td></tr><tr><td>Prometheus-V2-7B [26]</td><td>√</td><td>一</td><td>一</td><td>一</td><td>1024</td></tr><tr><td>Prometheus-V2-8×7B [26]</td><td>√</td><td>一</td><td>一</td><td>一</td><td>1024</td></tr><tr><td>Qwen3-8B [61]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>16,384</td></tr><tr><td>Qwen3.5-27B [51]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>32,768</td></tr><tr><td>Qwen3.5-9B [51]</td><td>√</td><td>一</td><td>√</td><td>√</td><td>16,384</td></tr></table>

Table 11: Mean ± standard deviation of Pearson r, Kendall τ, QWK, and MAE across individual CoT sampling attempts per problem. Thinking-enabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $0 . 5 5 4 \pm 0 . 0 2 5$ </td><td> $0 . 4 5 1 \pm 0 . 0 2 6$ </td><td> $0 . 4 4 6 \pm 0 . 0 2 4$ </td><td> $1 . 7 8 5 \pm 0 . 0 7 0$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $0 . 6 0 4 \pm 0 . 0 4 6$ </td><td> $0 . 5 0 6 \pm 0 . 0 4 3$ </td><td> $0 . 5 1 3 \pm 0 . 0 4 3$ </td><td> $1 . 6 2 3 \pm 0 . 0 8 7$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $0 . 4 3 5 \pm 0 . 0 1 1$ </td><td> $0 . 3 4 2 \pm 0 . 0 0 9$ </td><td> $0 . 2 8 9 \pm 0 . 0 0 8$ </td><td> $2 . 2 9 2 \pm 0 . 0 2 5$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $0 . 6 2 2 \pm 0 . 0 6 5$ </td><td> $0 . 5 2 6 \pm 0 . 0 5 8$ </td><td> $0 . 6 1 3 \pm 0 . 0 6 6$ </td><td> $1 . 3 5 7 \pm 0 . 1 3 6$ </td></tr><tr><td>deepseek-v4-flash-nothinking</td><td> $0 . 3 4 7 \pm 0 . 0 1 4$ </td><td> $0 . 2 8 6 \pm 0 . 0 2 0$ </td><td> $0 . 2 2 8 \pm 0 . 0 1 0$ </td><td> $2 . 3 6 3 \pm 0 . 0 3 0$ </td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td> $0 . 6 5 6 \pm 0 . 0 1 2$ </td><td> $0 . 5 4 4 \pm 0 . 0 1 5$ </td><td> $0 . 6 3 1 \pm 0 . 0 2 1$ </td><td> $1 . 3 7 5 \pm 0 . 0 2 9$ </td></tr><tr><td>deepseek-v4-pro</td><td> $0 . 5 9 4 \pm 0 . 0 7 8$ </td><td> $0 . 5 1 0 \pm 0 . 0 6 9$ </td><td> $0 . 5 8 6 \pm 0 . 0 7 8$ </td><td> $1 . 4 6 9 \pm 0 . 1 3 3$ </td></tr><tr><td>Gemini-2.5-Flash</td><td> $0 . 5 8 6 \pm 0 . 0 6 8$ </td><td> $0 . 4 7 8 \pm 0 . 0 5 7$ </td><td> $0 . 5 7 2 \pm 0 . 0 6 7$ </td><td> $1 . 5 1 2 \pm 0 . 1 5 3$ </td></tr><tr><td>Gemini-2.5-Pro (T)</td><td> $0 . 5 5 5 \pm 0 . 0 7 5$ </td><td> $0 . 4 4 9 \pm 0 . 0 7 4$ </td><td> $0 . 4 6 2 \pm 0 . 0 8 3$ </td><td> $1 . 6 0 6 \pm 0 . 1 4 6$ </td></tr><tr><td>Gemini-2.5-Pro (T)</td><td> $0 . 5 9 9 \pm 0 . 0 6 2$ </td><td> $0 . 4 8 4 \pm 0 . 0 5 7$ </td><td> $0 . 5 1 5 \pm 0 . 0 6 3$ </td><td> $1 . 5 0 0 \pm 0 . 1 3 3$ </td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td> $0 . 5 6 5 \pm 0 . 0 3 2$ </td><td> $0 . 4 6 3 \pm 0 . 0 3 5$ </td><td> $0 . 4 7 0 \pm 0 . 0 3 2$ </td><td> $1 . 6 1 5 \pm 0 . 0 7 7$ </td></tr><tr><td>GPT-4o</td><td> $0 . 2 1 6 \pm 0 . 0 2 6$ </td><td> $0 . 1 7 8 \pm 0 . 0 2 0$ </td><td> $0 . 0 6 2 \pm 0 . 0 1 2$ </td><td> $3 . 1 0 1 \pm 0 . 0 3 2$ </td></tr><tr><td>GPT-5</td><td> $0 . 6 2 0 \pm 0 . 0 5 1$ </td><td> $0 . 5 2 2 \pm 0 . 0 5 0$ </td><td> $0 . 5 5 7 \pm 0 . 0 5 6$ </td><td> $1 . 5 0 7 \pm 0 . 1 0 2$ </td></tr><tr><td>GPT-5-Mini</td><td> $0 . 5 1 9 \pm 0 . 0 4 9$ </td><td> $0 . 4 3 6 \pm 0 . 0 4 3$ </td><td> $0 . 4 2 1 \pm 0 . 0 4 6$ </td><td> $1 . 9 1 8 \pm 0 . 1 1 7$ </td></tr><tr><td>Kimi-K2.5</td><td> $0 . 4 1 3 \pm 0 . 0 9 1$ </td><td> $0 . 3 2 8 \pm 0 . 0 7 9$ </td><td> $0 . 2 8 2 \pm 0 . 0 6 7$ </td><td> $2 . 2 3 4 \pm 0 . 1 7 1$ </td></tr><tr><td>Prometheus-V2-7B</td><td> $0 . 0 2 0 \pm 0 . 0 9 7$ </td><td> $0 . 0 0 4 \pm 0 . 0 8 3$ </td><td> $0 . 0 1 0 \pm 0 . 0 6 8$ </td><td> $2 . 8 5 7 \pm 0 . 1 0 7$ </td></tr><tr><td>Prometheus-V2-8×7B</td><td> $0 . 0 6 3 \pm 0 . 1 0 9$ </td><td> $0 . 0 6 0 \pm 0 . 0 7 6$ </td><td> $0 . 0 3 1 \pm 0 . 0 5 3$ </td><td> $2 . 8 9 5 \pm 0 . 1 4 5$ </td></tr><tr><td>Qwen3-8B</td><td> $0 . 1 8 0 \pm 0 . 0 5 3$ </td><td> $0 . 1 8 6 \pm 0 . 0 3 9$ </td><td> $0 . 0 7 6 \pm 0 . 0 2 4$ </td><td> $3 . 0 7 0 \pm 0 . 0 8 0$ </td></tr><tr><td> $\mathbf { Q } \mathrm { w e n } 3 . 5 \substack { - 2 7 \mathbf { B } }$ </td><td> $0 . 3 9 6 \pm 0 . 0 3 6$ </td><td> $0 . 3 1 9 \pm 0 . 0 4 2$ </td><td> $0 . 2 8 7 \pm 0 . 0 4 3$ </td><td> $2 . 2 7 8 \pm 0 . 1 1 2$ </td></tr><tr><td>Qwen3.5-9B</td><td> $0 . 4 2 7 \pm 0 . 0 6 9$ </td><td> $0 . 3 7 3 \pm 0 . 0 5 6$ </td><td> $0 . 3 0 6 \pm 0 . 0 6 4$ </td><td> $2 . 3 7 0 \pm 0 . 1 6 2$ </td></tr></table>

Table 12: Mean ± standard deviation of RMSE, bias, and exact-match accuracy across individual CoT sampling attempts per problem. Thinking-enabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>RMSE</td><td>Bias</td><td>Exact</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $2 . 4 0 8 \pm 0 . 0 5 0$ </td><td> $1 . 3 4 9 \pm 0 . 0 8 5$ </td><td> $0 . 2 2 8 \pm 0 . 0 4 5$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $2 . 2 2 7 \pm 0 . 0 9 6$ </td><td> $1 . 1 7 7 \pm 0 . 1 1 0$ </td><td> $0 . 2 7 7 \pm 0 . 0 3 8$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $3 . 0 6 8 \pm 0 . 0 2 2$ </td><td> $2 . 0 7 3 \pm 0 . 0 3 5$ </td><td> $0 . 2 3 8 \pm 0 . 0 1 1$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $2 . 1 1 7 \pm 0 . 2 0 1$ </td><td> $0 . 2 6 0 \pm 0 . 2 0 7$ </td><td> $0 . 4 1 4 \pm 0 . 0 3 9$ </td></tr><tr><td>deepseek-v4-flash-nothinking</td><td> $3 . 1 5 8 \pm 0 . 0 2 9$ </td><td> $2 . 0 5 8 \pm 0 . 0 3 6$ </td><td> $0 . 2 3 2 \pm 0 . 0 2 2$ </td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td> $2 . 1 1 1 \pm 0 . 0 6 1$ </td><td> $- 0 . 6 1 5 \pm 0 . 1 6 5$ </td><td> $0 . 3 8 0 \pm 0 . 0 2 0$ </td></tr><tr><td>deepseek-v4-pro</td><td> $2 . 2 5 5 \pm 0 . 2 2 5$ </td><td> $- 0 . 0 2 1 \pm 0 . 1 9 5$ </td><td> $0 . 4 0 1 \pm 0 . 0 2 0$ </td></tr><tr><td>Gemini-2.5-Flash</td><td> $2 . 2 1 7 \pm 0 . 1 8 3$ </td><td> $0 . 4 6 1 \pm 0 . 2 0 9$ </td><td> $0 . 3 2 5 \pm 0 . 0 5 7$ </td></tr><tr><td>Gemini-2.5-Pro (T)</td><td> $2 . 3 5 8 \pm 0 . 1 9 9$ </td><td> $1 . 2 4 5 \pm 0 . 2 2 7$ </td><td> $0 . 3 3 9 \pm 0 . 0 3 6$ </td></tr><tr><td>Gemini-2.5-Pro (T)</td><td> $2 . 1 8 6 \pm 0 . 1 4 5$ </td><td> $1 . 0 9 8 \pm 0 . 1 1 1$ </td><td> $0 . 3 3 0 \pm 0 . 0 5 8$ </td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td> $2 . 4 5 4 \pm 0 . 1 0 9$ </td><td> $1 . 3 6 5 \pm 0 . 0 8 8$ </td><td> $0 . 3 5 9 \pm 0 . 0 2 0$ </td></tr><tr><td>GPT-40</td><td> $3 . 8 2 6 \pm 0 . 0 2 5$ </td><td> $3 . 0 6 4 \pm 0 . 0 3 7$ </td><td> $0 . 1 6 6 \pm 0 . 0 1 3$ </td></tr><tr><td>GPT-5</td><td> $2 . 1 1 2 \pm 0 . 1 2 3$ </td><td> $0 . 9 6 0 \pm 0 . 0 8 4$ </td><td> $0 . 2 8 5 \pm 0 . 0 3 4$ </td></tr><tr><td>GPT-5-Mini</td><td> $2 . 5 8 8 \pm 0 . 1 2 6$ </td><td> $1 . 4 5 6 \pm 0 . 1 1 2$ </td><td> $0 . 2 2 7 \pm 0 . 0 3 6$ </td></tr><tr><td>Kimi-K2.5</td><td> $2 . 9 9 8 \pm 0 . 1 7 8$ </td><td> $1 . 9 4 4 \pm 0 . 1 8 9$ </td><td> $0 . 2 5 1 \pm 0 . 0 2 8$ </td></tr><tr><td>Prometheus-V2-7B</td><td> $3 . 5 6 4 \pm 0 . 1 0 3$ </td><td> $1 . 8 9 5 \pm 0 . 2 7 2$ </td><td> $0 . 1 3 8 \pm 0 . 0 3 6$ </td></tr><tr><td>Prometheus-V2-8×7B</td><td> $3 . 5 8 9 \pm 0 . 1 3 1$ </td><td> $2 . 4 6 4 \pm 0 . 1 2 0$ </td><td> $0 . 1 5 0 \pm 0 . 0 4 2$ </td></tr><tr><td>Qwen3-8B</td><td> $3 . 8 2 4 \pm 0 . 0 5 5$ </td><td> $2 . 9 0 8 \pm 0 . 1 1 5$ </td><td> $0 . 1 5 7 \pm 0 . 0 3 0$ </td></tr><tr><td>Qwen3.5-27B</td><td> $3 . 0 7 1 \pm 0 . 1 1 0$ </td><td> $1 . 9 0 3 \pm 0 . 1 5 9$ </td><td> $0 . 2 5 3 \pm 0 . 0 1 7$ </td></tr><tr><td>Qwen3.5-9B</td><td> $3 . 1 4 5 \pm 0 . 1 6 6$ </td><td> $2 . 0 1 2 \pm 0 . 1 8 6$ </td><td> $0 . 2 2 1 \pm 0 . 0 2 9$ </td></tr></table>

Table 13: Mean ± standard deviation of Pearson r, Kendall τ, QWK, and MAE across individual CoT sampling attempts per problem. Thinking-enabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $0 . 3 7 9 \pm 0 . 0 4 3$ </td><td> $0 . 2 9 3 \pm 0 . 0 6 1$ </td><td> $0 . 3 4 9 \pm 0 . 0 4 9$ </td><td> $1 . 7 9 7 \pm 0 . 0 5 7$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $0 . 5 8 4 \pm 0 . 1 0 0$ </td><td> $0 . 4 8 1 \pm 0 . 1 3 1$ </td><td> $0 . 4 9 7 \pm 0 . 1 0 0$ </td><td> $1 . 6 4 6 \pm 0 . 1 8 8$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $0 . 1 5 1 \pm 0 . 1 1 5$ </td><td> $0 . 0 8 9 \pm 0 . 0 7 9$ </td><td> $0 . 0 5 3 \pm 0 . 0 4 5$ </td><td> $3 . 0 1 2 \pm 0 . 0 9 7$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $0 . 5 1 4 \pm 0 . 0 7 2$ </td><td> $0 . 3 9 9 \pm 0 . 0 6 4$ </td><td> $0 . 4 5 6 \pm 0 . 0 6 9$ </td><td> $1 . 6 9 9 \pm 0 . 1 1 1$ </td></tr><tr><td>deepseek-v4-flash-nothinking</td><td> $0 . 1 6 4 \pm 0 . 1 2 5$ </td><td> $0 . 1 1 2 \pm 0 . 1 2 4$ </td><td> $0 . 0 7 1 \pm 0 . 0 5 8$ </td><td> $2 . 7 7 6 \pm 0 . 0 3 6$ </td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td> $0 . 4 6 8 \pm 0 . 0 7 2$ </td><td> $0 . 3 6 1 \pm 0 . 0 5 4$ </td><td> $0 . 3 9 6 \pm 0 . 0 5 8$ </td><td> $2 . 1 4 6 \pm 0 . 1 4 3$ </td></tr><tr><td>Gemini-2.5-Flash</td><td> $0 . 3 3 9 \pm 0 . 1 4 7$ </td><td> $0 . 2 6 5 \pm 0 . 1 1 9$ </td><td> $0 . 3 3 6 \pm 0 . 1 4 9$ </td><td> $1 . 9 1 3 \pm 0 . 1 9 6$ </td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td> $0 . 4 6 4 \pm 0 . 0 3 3$ </td><td> $0 . 3 7 6 \pm 0 . 0 3 3$ </td><td> $0 . 3 6 9 \pm 0 . 0 4 9$ </td><td> $2 . 0 5 7 \pm 0 . 0 9 5$ </td></tr><tr><td>GPT-40</td><td> $0 . 0 6 3 \pm 0 . 0 6 9$ </td><td> $0 . 0 3 1 \pm 0 . 0 5 7$ </td><td> $0 . 0 4 0 \pm 0 . 0 4 3$ </td><td> $2 . 1 3 8 \pm 0 . 0 5 5$ </td></tr><tr><td>GPT-5</td><td> $0 . 4 7 5 \pm 0 . 0 5 1$ </td><td> $0 . 3 5 1 \pm 0 . 0 6 7$ </td><td> $0 . 4 3 5 \pm 0 . 0 5 4$ </td><td> $1 . 6 6 3 \pm 0 . 1 1 0$ </td></tr><tr><td>GPT-5-Mini</td><td> $0 . 4 5 8 \pm 0 . 0 6 5$ </td><td> $0 . 3 3 7 \pm 0 . 0 6 3$ </td><td> $0 . 4 5 5 \pm 0 . 0 6 6$ </td><td> $1 . 7 7 0 \pm 0 . 1 1 4$ </td></tr><tr><td>Kimi-K2.5</td><td> $0 . 4 3 4 \pm 0 . 0 7 0$ </td><td> $0 . 3 1 9 \pm 0 . 0 6 6$ </td><td> $0 . 3 7 2 \pm 0 . 0 7 9$ </td><td> $1 . 8 7 9 \pm 0 . 1 3 4$ </td></tr><tr><td>Qwen3-8B</td><td> $0 . 1 9 6 \pm 0 . 1 1 9$ </td><td> $0 . 1 5 4 \pm 0 . 0 9 3$ </td><td> $0 . 1 7 5 \pm 0 . 1 0 7$ </td><td> $2 . 4 4 2 \pm 0 . 1 7 3$ </td></tr><tr><td>Qwen3.5-27B</td><td> $0 . 4 5 4 \pm 0 . 0 2 2$ </td><td> $0 . 3 5 2 \pm 0 . 0 3 0$ </td><td> $0 . 4 4 8 \pm 0 . 0 2 3$ </td><td> $1 . 7 0 3 \pm 0 . 1 0 7$ </td></tr></table>

Table 14: Mean ± standard deviation of RMSE, bias, and exact-match accuracy across individual CoT sampling attempts per problem. Thinking-enabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>RMSE</td><td>Bias</td><td>Exact</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $2 . 2 9 2 \pm 0 . 0 5 2$ </td><td> $- 0 . 4 2 2 \pm 0 . 0 8 6$ </td><td> $0 . 2 1 4 \pm 0 . 0 4 6$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $2 . 0 0 6 \pm 0 . 2 3 8$ </td><td> $- 0 . 6 0 9 \pm 0 . 4 3 7$ </td><td> $0 . 1 4 6 \pm 0 . 0 5 5$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $3 . 6 8 3 \pm 0 . 1 0 7$ </td><td> $- 2 . 8 3 2 \pm 0 . 1 0 3$ </td><td> $0 . 1 3 3 \pm 0 . 0 2 0$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>2.243 ± 0.128</td><td> $- 0 . 8 7 1 \pm 0 . 1 3 9$ </td><td> $0 . 2 2 7 \pm 0 . 0 4 8$ </td></tr><tr><td>deepseek-v4-flash-nothinking</td><td>3.476 ± 0.067</td><td> $- 2 . 5 0 5 \pm 0 . 0 7 5$ </td><td> $0 . 1 5 6 \pm 0 . 0 3 6$ </td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>2.798 ± 0.151</td><td> $- 1 . 4 0 6 \pm 0 . 0 2 7$ </td><td> $0 . 2 2 9 \pm 0 . 0 1 7$ </td></tr><tr><td>Gemini-2.5-Flash</td><td> $2 . 5 0 1 \pm 0 . 2 5 4$ </td><td> $- 0 . 0 2 8 \pm 0 . 2 0 6$ </td><td> $0 . 2 0 4 \pm 0 . 0 1 9$ </td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td> $2 . 7 7 2 \pm 0 . 1 0 6$ </td><td> $1 . 5 6 8 \pm 0 . 1 8 0$ </td><td> $0 . 2 4 0 \pm 0 . 0 2 7$ </td></tr><tr><td>GPT-4o</td><td> $2 . 5 4 4 \pm 0 . 0 5 3$ </td><td> $0 . 8 0 2 \pm 0 . 0 6 5$ </td><td> $0 . 0 9 8 \pm 0 . 0 1 7$ </td></tr><tr><td>GPT-5</td><td> $2 . 1 9 9 \pm 0 . 1 0 9$ </td><td> $- 0 . 5 9 8 \pm 0 . 1 6 1$ </td><td> $0 . 2 1 1 \pm 0 . 0 5 4$ </td></tr><tr><td>GPT-5-Mini</td><td> $2 . 3 2 7 \pm 0 . 1 3 4$ </td><td> $- 0 . 1 3 6 \pm 0 . 1 3 6$ </td><td> $0 . 2 0 4 \pm 0 . 0 4 2$ </td></tr><tr><td>Kimi-K2.5</td><td> $2 . 4 1 5 \pm 0 . 1 0 7$ </td><td> $- 1 . 0 0 1 \pm 0 . 1 7 7$ </td><td> $0 . 1 8 2 \pm 0 . 0 5 4$ </td></tr><tr><td>Qwen3-8B</td><td> $3 . 1 4 3 \pm 0 . 1 6 7$ </td><td> $1 . 0 4 4 \pm 0 . 2 9 1$ </td><td> $0 . 1 9 6 \pm 0 . 0 4 6$ </td></tr><tr><td>Qwen3.5-27B</td><td> $2 . 2 8 4 \pm 0 . 0 8 1$ </td><td> $- 0 . 1 5 1 \pm 0 . 1 8 8$ </td><td> $0 . 2 6 6 \pm 0 . 0 4 6$ </td></tr></table>

## D.1 Trojan-Bench(H)

Correlation and error patterns on Trojan-Bench(H). Across both Full and Blind templates, the strongest graders on Trojan-Bench(H) are the thinking-enabled variants of frontier models. Under the Full setting, DeepSeek-V4-Flash (Think) leads on most aggregate metrics with mean Pearson r of 0.66, mean QWK of 0.63, and mean MAE of 1.38 across attempts, closely followed by GPT-5 (0.62 Pearson, 0.56 QWK) and DeepSeek-V3.2 (Think) (0.62 Pearson, 0.61 QWK). The non-thinking variants of the same model families lose between 0.05 and 0.20 QWK and pick up systematic over-grading bias, often exceeding +1.0 points. Smaller open-weight graders such as Qwen3-8B, Prometheus-V2-7B, and Prometheus-V2-8×7B, together with GPT-4o, collapse to near-zero or negative correlation with the human reference, with MAE close to or exceeding 3.0, which is the level of a constant "award full marks" baseline.

Effect of aggregation. Mean-CoT@16 and MajorityVote@16 produce small but consistent gains over a single CoT call. Gemini-2.5-Flash improves from QWK 0.59 at CoT@1 to 0.68 at Mean-CoT@16 in the Full setting, and DeepSeek-V4-Flash (Think) improves from 0.64 to 0.69. MajorityVote@16 produces similar gains except on graders whose all samples agree, in which case voting cannot denoise a deterministic bias (a pattern we revisit in the case-study section). The Blind setting shows the same ordering of graders as Full but with a uniform downward shift in correlation, and a sign flip in bias for several frontier models. DeepSeek-V3.2 (Think), for example, moves from +0.27 Full to −0.88 Blind, which reflects the absence of the rubric anchor.

Smaller graders behave differently on Trojan-Bench(H). A non-trivial ranking shift appears for the small graders. Qwen3-8B, GPT-5 Mini, and Kimi-K2.5 reach better QWK and lower MAE under the Blind template than under the Full template. We discuss this counter-intuitive effect in detail in Section J.1. The short version is that the rubric in the Full template acts as a surface-form anchor that biases small graders toward awarding most of the available points on each rubric line.

Table 15: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.54</td><td>0.44</td><td>0.44</td><td>1.79</td><td>2.43</td><td>1.33</td><td>0.56</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.55</td><td>0.45</td><td>0.47</td><td>1.73</td><td>2.40</td><td>1.24</td><td>0.58</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.43</td><td>0.35</td><td>0.29</td><td>2.27</td><td>3.04</td><td>2.02</td><td>0.48</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.60</td><td>0.53</td><td>0.59</td><td>1.35</td><td>2.26</td><td>0.27</td><td>0.71</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.66</td><td>0.54</td><td>0.64</td><td>1.33</td><td>2.08</td><td>-0.50</td><td>0.69</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.62</td><td>0.50</td><td>0.59</td><td>1.54</td><td>2.23</td><td>0.63</td><td>0.58</td></tr><tr><td>Gemini-2.5-Pro (T)</td><td>0.60</td><td>0.49</td><td>0.52</td><td>1.52</td><td>2.20</td><td>1.06</td><td>0.65</td></tr><tr><td>Gemini-2.5-Pro (T)</td><td>0.57</td><td>0.45</td><td>0.49</td><td>1.54</td><td>2.20</td><td>1.04</td><td>0.69</td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td>0.54</td><td>0.44</td><td>0.45</td><td>1.65</td><td>2.50</td><td>1.35</td><td>0.65</td></tr><tr><td>GPT-40</td><td>0.20</td><td>0.17</td><td>0.06</td><td>3.10</td><td>3.83</td><td>3.06</td><td>0.31</td></tr><tr><td>GPT-5</td><td>0.57</td><td>0.47</td><td>0.51</td><td>1.56</td><td>2.22</td><td>0.98</td><td>0.65</td></tr><tr><td>GPT-5-Mini</td><td>0.63</td><td>0.52</td><td>0.53</td><td>1.67</td><td>2.25</td><td>1.25</td><td>0.54</td></tr><tr><td>Kimi-K2.5</td><td>0.33</td><td>0.22</td><td>0.20</td><td>2.37</td><td>3.29</td><td>2.24</td><td>0.50</td></tr><tr><td>Prometheus-V2-7B</td><td>-0.09</td><td>-0.12</td><td>-0.07</td><td>2.92</td><td>3.66</td><td>1.71</td><td>0.35</td></tr><tr><td>Prometheus-V2-8×7B</td><td>0.12</td><td>0.03</td><td>0.06</td><td>3.00</td><td>3.72</td><td>2.63</td><td>0.38</td></tr><tr><td>Qwen3-8B</td><td>0.15</td><td>0.16</td><td>0.07</td><td>3.04</td><td>3.83</td><td>2.83</td><td>0.34</td></tr><tr><td>Qwen3.5-27B</td><td>0.43</td><td>0.36</td><td>0.33</td><td>2.19</td><td>2.99</td><td>1.73</td><td>0.50</td></tr><tr><td>Qwen3.5-9B</td><td>0.30</td><td>0.26</td><td>0.20</td><td>2.58</td><td>3.37</td><td>2.16</td><td>0.39</td></tr></table>

Table 16: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall  $\tau$ </td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.57</td><td>0.47</td><td>0.46</td><td>1.69</td><td>2.35</td><td>1.31</td><td>0.60</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.65</td><td>0.53</td><td>0.55</td><td>1.54</td><td>2.13</td><td>1.17</td><td>0.60</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.44</td><td>0.34</td><td>0.29</td><td>2.29</td><td>3.06</td><td>2.08</td><td>0.48</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.65</td><td>0.53</td><td>0.65</td><td>1.35</td><td>1.93</td><td>0.23</td><td>0.67</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.72</td><td>0.59</td><td>0.69</td><td>1.25</td><td>1.87</td><td>-0.67</td><td>0.69</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.69</td><td>0.57</td><td>0.68</td><td>1.23</td><td>1.76</td><td>0.44</td><td>0.73</td></tr><tr><td>Gemini-2.5-Pro (T)</td><td>0.59</td><td>0.48</td><td>0.48</td><td>1.52</td><td>2.24</td><td>1.23</td><td>0.69</td></tr><tr><td>Gemini-2.5-Pro (T)</td><td>0.64</td><td>0.53</td><td>0.54</td><td>1.42</td><td>2.07</td><td>1.08</td><td>0.67</td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td>0.59</td><td>0.48</td><td>0.49</td><td>1.58</td><td>2.37</td><td>1.33</td><td>0.65</td></tr><tr><td>GPT-40</td><td>0.20</td><td>0.17</td><td>0.06</td><td>3.10</td><td>3.83</td><td>3.06</td><td>0.31</td></tr><tr><td>GPT-5</td><td>0.67</td><td>0.57</td><td>0.59</td><td>1.44</td><td>2.01</td><td>0.98</td><td>0.69</td></tr><tr><td>GPT-5-Mini</td><td>0.55</td><td>0.46</td><td>0.43</td><td>1.88</td><td>2.52</td><td>1.50</td><td>0.52</td></tr><tr><td>Kimi-K2.5</td><td>0.48</td><td>0.36</td><td>0.31</td><td>2.13</td><td>2.83</td><td>1.92</td><td>0.46</td></tr><tr><td>Prometheus-V2-7B</td><td>0.09</td><td>0.06</td><td>0.04</td><td>2.48</td><td>3.07</td><td>1.94</td><td>0.42</td></tr><tr><td>Prometheus-V2-8×7B</td><td>0.13</td><td>0.17</td><td>0.05</td><td>2.73</td><td>3.42</td><td>2.48</td><td>0.33</td></tr><tr><td>Qwen3-8B</td><td>0.20</td><td>0.21</td><td>0.08</td><td>3.08</td><td>3.78</td><td>2.92</td><td>0.29</td></tr><tr><td>Qwen3.5-27B</td><td>0.44</td><td>0.37</td><td>0.32</td><td>2.17</td><td>2.93</td><td>1.83</td><td>0.48</td></tr><tr><td>Qwen3.5-9B</td><td>0.45</td><td>0.36</td><td>0.31</td><td>2.27</td><td>3.05</td><td>2.02</td><td>0.50</td></tr></table>

Table 17: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.58</td><td>0.47</td><td>0.46</td><td>1.75</td><td>2.37</td><td>1.38</td><td>0.58</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.61</td><td>0.53</td><td>0.52</td><td>1.65</td><td>2.25</td><td>1.23</td><td>0.56</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.44</td><td>0.34</td><td>0.29</td><td>2.29</td><td>3.08</td><td>2.08</td><td>0.48</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.66</td><td>0.57</td><td>0.65</td><td>1.27</td><td>2.05</td><td>0.10</td><td>0.67</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.65</td><td>0.54</td><td>0.63</td><td>1.38</td><td>2.13</td><td>-0.54</td><td>0.65</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.67</td><td>0.57</td><td>0.64</td><td>1.38</td><td>2.14</td><td>0.63</td><td>0.69</td></tr><tr><td>Gemini-2.5-Pro (T)</td><td>0.54</td><td>0.44</td><td>0.45</td><td>1.58</td><td>2.43</td><td>1.29</td><td>0.63</td></tr><tr><td>Gemini-2.5-Pro (T)</td><td>0.63</td><td>0.51</td><td>0.55</td><td>1.42</td><td>2.16</td><td>1.13</td><td>0.71</td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td>0.54</td><td>0.44</td><td>0.45</td><td>1.67</td><td>2.50</td><td>1.38</td><td>0.65</td></tr><tr><td>GPT-40</td><td>0.20</td><td>0.17</td><td>0.06</td><td>3.10</td><td>3.83</td><td>3.06</td><td>0.31</td></tr><tr><td>GPT-5</td><td>0.68</td><td>0.59</td><td>0.62</td><td>1.44</td><td>2.01</td><td>0.98</td><td>0.65</td></tr><tr><td>GPT-5-Mini</td><td>0.57</td><td>0.49</td><td>0.48</td><td>1.81</td><td>2.49</td><td>1.35</td><td>0.56</td></tr><tr><td>Kimi-K2.5</td><td>0.45</td><td>0.35</td><td>0.28</td><td>2.27</td><td>3.05</td><td>2.15</td><td>0.46</td></tr><tr><td>Prometheus-V2-7B</td><td>0.13</td><td>0.07</td><td>0.05</td><td>3.23</td><td>4.00</td><td>3.10</td><td>0.31</td></tr><tr><td>Prometheus-V2-8×7B</td><td>0.11</td><td>0.07</td><td>0.04</td><td>3.19</td><td>3.85</td><td>2.98</td><td>0.25</td></tr><tr><td>Qwen3-8B</td><td>0.18</td><td>0.23</td><td>0.06</td><td>3.21</td><td>3.88</td><td>3.04</td><td>0.27</td></tr><tr><td>Qwen3.5-27B</td><td>0.40</td><td>0.33</td><td>0.30</td><td>2.23</td><td>3.04</td><td>1.81</td><td>0.50</td></tr><tr><td>Qwen3.5-9B</td><td>0.37</td><td>0.30</td><td>0.26</td><td>2.46</td><td>3.29</td><td>2.04</td><td>0.44</td></tr></table>

Table 18: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.40</td><td>0.32</td><td>0.38</td><td>1.77</td><td>2.24</td><td>-0.31</td><td>0.44</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.49</td><td>0.34</td><td>0.46</td><td>1.60</td><td>1.97</td><td>0.16</td><td>0.56</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.15</td><td>0.04</td><td>0.06</td><td>2.96</td><td>3.64</td><td>-2.71</td><td>0.29</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.57</td><td>0.50</td><td>0.51</td><td>1.54</td><td>2.12</td><td>-0.88</td><td>0.56</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.40</td><td>0.31</td><td>0.34</td><td>2.25</td><td>2.94</td><td>-1.38</td><td>0.46</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.42</td><td>0.32</td><td>0.43</td><td>1.86</td><td>2.39</td><td>-0.01</td><td>0.46</td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td>0.48</td><td>0.39</td><td>0.40</td><td>2.00</td><td>2.73</td><td>1.46</td><td>0.52</td></tr><tr><td>GPT-40</td><td>0.06</td><td>0.00</td><td>0.04</td><td>2.17</td><td>2.55</td><td>0.79</td><td>0.40</td></tr><tr><td>GPT-5</td><td>0.45</td><td>0.32</td><td>0.39</td><td>1.71</td><td>2.32</td><td>-0.87</td><td>0.56</td></tr><tr><td>GPT-5-Mini</td><td>0.37</td><td>0.23</td><td>0.37</td><td>1.92</td><td>2.49</td><td>-0.29</td><td>0.44</td></tr><tr><td>Kimi-K2.5</td><td>0.41</td><td>0.31</td><td>0.33</td><td>1.92</td><td>2.35</td><td>-0.92</td><td>0.48</td></tr><tr><td>Qwen3-8B</td><td>0.28</td><td>0.21</td><td>0.26</td><td>2.35</td><td>3.13</td><td>0.91</td><td>0.44</td></tr><tr><td>Qwen3.5-27B</td><td>0.44</td><td>0.34</td><td>0.44</td><td>1.69</td><td>2.31</td><td>-0.31</td><td>0.52</td></tr></table>

Table 19: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.39</td><td>0.31</td><td>0.35</td><td>1.77</td><td>2.22</td><td>-0.40</td><td>0.44</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.48</td><td>0.36</td><td>0.40</td><td>1.63</td><td>2.03</td><td>-0.21</td><td>0.50</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.20</td><td>0.12</td><td>0.06</td><td>3.00</td><td>3.63</td><td>-2.83</td><td>0.31</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.61</td><td>0.48</td><td>0.52</td><td>1.52</td><td>2.03</td><td>-0.85</td><td>0.60</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.51</td><td>0.39</td><td>0.42</td><td>2.04</td><td>2.67</td><td>-1.46</td><td>0.50</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.44</td><td>0.33</td><td>0.43</td><td>1.75</td><td>2.29</td><td>-0.25</td><td>0.54</td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td>0.50</td><td>0.42</td><td>0.39</td><td>2.06</td><td>2.63</td><td>1.56</td><td>0.46</td></tr><tr><td>GPT-40</td><td>0.05</td><td>0.01</td><td>0.03</td><td>2.10</td><td>2.54</td><td>0.81</td><td>0.44</td></tr><tr><td>GPT-5</td><td>0.53</td><td>0.40</td><td>0.48</td><td>1.58</td><td>2.08</td><td>-0.63</td><td>0.56</td></tr><tr><td>GPT-5-Mini</td><td>0.53</td><td>0.39</td><td>0.53</td><td>1.63</td><td>2.07</td><td>-0.08</td><td>0.54</td></tr><tr><td>Kimi-K2.5</td><td>0.54</td><td>0.44</td><td>0.43</td><td>1.71</td><td>2.14</td><td>-0.92</td><td>0.48</td></tr><tr><td>Qwen3-8B</td><td>0.33</td><td>0.25</td><td>0.28</td><td>2.08</td><td>2.58</td><td>1.04</td><td>0.46</td></tr><tr><td>Qwen3.5-27B</td><td>0.49</td><td>0.38</td><td>0.48</td><td>1.56</td><td>2.12</td><td>-0.10</td><td>0.56</td></tr></table>

Table 20: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the human-labelled Trojan-Bench(H) dataset without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.46</td><td>0.38</td><td>0.42</td><td>1.67</td><td>2.15</td><td>-0.46</td><td>0.48</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.47</td><td>0.35</td><td>0.40</td><td>1.63</td><td>2.03</td><td>-0.13</td><td>0.54</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.09</td><td>0.06</td><td>0.02</td><td>3.15</td><td>3.83</td><td>-3.02</td><td>0.29</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.59</td><td>0.48</td><td>0.49</td><td>1.58</td><td>2.18</td><td>-1.08</td><td>0.58</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.49</td><td>0.39</td><td>0.42</td><td>2.10</td><td>2.74</td><td>-1.35</td><td>0.48</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.38</td><td>0.29</td><td>0.38</td><td>1.85</td><td>2.56</td><td>-0.06</td><td>0.52</td></tr><tr><td>Gemini-3-Pro-Prev. (T)</td><td>0.42</td><td>0.35</td><td>0.33</td><td>2.17</td><td>2.86</td><td>1.58</td><td>0.46</td></tr><tr><td>GPT-40</td><td>0.05</td><td>0.01</td><td>0.03</td><td>2.08</td><td>2.52</td><td>0.79</td><td>0.46</td></tr><tr><td>GPT-5</td><td>0.50</td><td>0.38</td><td>0.47</td><td>1.60</td><td>2.17</td><td>-0.56</td><td>0.54</td></tr><tr><td>GPT-5-Mini</td><td>0.51</td><td>0.38</td><td>0.51</td><td>1.63</td><td>2.24</td><td>-0.25</td><td>0.56</td></tr><tr><td>Kimi-K2.5</td><td>0.50</td><td>0.41</td><td>0.42</td><td>1.79</td><td>2.29</td><td>-1.00</td><td>0.50</td></tr><tr><td>Qwen3-8B</td><td>0.29</td><td>0.25</td><td>0.25</td><td>2.40</td><td>3.16</td><td>1.27</td><td>0.44</td></tr><tr><td>Qwen3.5-27B</td><td>0.46</td><td>0.36</td><td>0.46</td><td>1.65</td><td>2.26</td><td>-0.31</td><td>0.56</td></tr></table>

## D.2 Trojan-Bench(S)

Correlation and error patterns on Trojan-Bench(S). The synthetic benchmark is constructed so that every candidate solution has a target rubric-aligned grade, which makes the Full setting easier than on Trojan-Bench(H). Under the Full template, Claude-Sonnet-4.6 (Think) achieves Pearson r of 0.81 and QWK of 0.75 at the aggregated level; DeepSeek-V3.2 (Think) reaches 0.81 Pearson and 0.77 QWK with MeanCoT@16; and GPT-5 reaches 0.77 Pearson and 0.69 QWK. Both Mean-CoT and MajorityVote lift aggregate quality by a few points relative to a single CoT call. MAE values cluster around or below 1.0 for the strongest graders, which means the synthetic distribution is concentrated where rubric-following alone is enough to recover the target grade.

Failure modes carry over from Trojan-Bench(H). Two of the failure modes we observe on Trojan-Bench(H) also appear on the synthetic data. First, GPT-4o has near-zero QWK (0.12) despite a non-trivial Pearson correlation (0.53) and a bias of +3.37, which is the signature of a grader that ranks proofs in roughly the right order but inflates every grade to the maximum. Second, the lower-tier graders (Kimi-K2.5, Qwen3.5-27B, GPT-5 Mini) carry a positive bias of about +1.6 to +2.2, again driven by over-crediting partially correct rubric items. Under the Blind template, the systematic over-grading shrinks because the rubric anchor is removed, and most graders move to a small positive or near-zero bias under MajorityVote@16.

Cross-set consistency. The relative ordering of graders is largely preserved between Trojan-Bench(H) and Trojan-Bench(S). The top three graders on the Full template are the same on both datasets, and the bottom three are also the same. The absolute values are higher on the synthetic set because the candidate distribution has lower per-rubric-item ambiguity by construction, but the directional conclusions match. This cross-set consistency is the basis for our claim in the main text that fine-tuning on the synthetic set transfers usefully to the human set.

Table 21: Mean ± standard deviation of Pearson r, Kendall τ, QWK, and MAE across individual CoT sampling attempts per problem. Thinking-enabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $0 . 7 7 2 \pm 0 . 0 6 7$ </td><td> $0 . 6 7 7 \pm 0 . 0 7 9$ </td><td> $0 . 7 1 3 \pm 0 . 0 7 5$ </td><td> $0 . 8 4 3 \pm 0 . 1 5 7$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $0 . 7 2 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 3 0 \pm 0 . 0 0 7$ </td><td> $0 . 4 2 3 \pm 0 . 0 0 7$ </td><td> $2 . 0 0 8 \pm 0 . 0 1 8$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $0 . 7 6 8 \pm 0 . 0 1 9$ </td><td> $0 . 6 5 2 \pm 0 . 0 2 1$ </td><td> $0 . 7 2 4 \pm 0 . 0 1 9$ </td><td> $1 . 0 2 0 \pm 0 . 0 3 9$ </td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td> $0 . 5 8 7 \pm 0 . 0 4 2$ </td><td> $0 . 4 8 8 \pm 0 . 0 3 7$ </td><td> $0 . 5 3 3 \pm 0 . 0 3 6$ </td><td> $1 . 3 8 0 \pm 0 . 0 5 1$ </td></tr><tr><td>GPT-40</td><td> $0 . 5 2 7 \pm 0 . 0 0 7$ </td><td> $0 . 4 5 4 \pm 0 . 0 0 4$ </td><td> $0 . 1 2 0 \pm 0 . 0 0 3$ </td><td> $3 . 3 7 1 \pm 0 . 0 0 9$ </td></tr><tr><td>GPT-5</td><td> $0 . 7 4 5 \pm 0 . 0 1 8$ </td><td> $0 . 6 2 6 \pm 0 . 0 1 7$ </td><td> $0 . 6 5 8 \pm 0 . 0 1 2$ </td><td> $1 . 1 2 1 \pm 0 . 0 3 9$ </td></tr><tr><td>GPT-5-Mini</td><td> $0 . 6 5 7 \pm 0 . 0 0 8$ </td><td> $0 . 5 1 9 \pm 0 . 0 0 7$ </td><td> $0 . 4 4 5 \pm 0 . 0 0 4$ </td><td> $1 . 7 7 9 \pm 0 . 0 1 8$ </td></tr><tr><td>Kimi-K2.5</td><td> $0 . 6 9 2 \pm 0 . 0 2 6$ </td><td> $0 . 5 7 2 \pm 0 . 0 2 4$ </td><td> $0 . 3 6 3 \pm 0 . 0 1 8$ </td><td> $2 . 2 4 6 \pm 0 . 0 3 3$ </td></tr><tr><td>Qwen3.5-27B</td><td> $0 . 6 8 0 \pm 0 . 0 1 1$ </td><td> $0 . 5 7 2 \pm 0 . 0 1 6$ </td><td> $0 . 3 9 5 \pm 0 . 0 0 6$ </td><td> $2 . 1 8 3 \pm 0 . 0 2 5$ </td></tr><tr><td>Qwen3.5-9B</td><td> $0 . 7 3 2 \pm 0 . 0 1 7$ </td><td> $0 . 6 0 4 \pm 0 . 0 1 9$ </td><td> $0 . 5 2 8 \pm 0 . 0 1 4$ </td><td> $1 . 7 3 1 \pm 0 . 0 5 5$ </td></tr></table>

Table 22: Mean ± standard deviation of RMSE, bias, and exact-match accuracy across individual CoT sampling attempts per problem. Thinking-enabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>RMSE</td><td>Bias</td><td>Exact</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $1 . 2 5 6 \pm 0 . 1 8 9$ </td><td> $0 . 6 0 9 \pm 0 . 0 9 3$ </td><td> $0 . 4 3 6 \pm 0 . 0 8 0$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $2 . 3 3 7 \pm 0 . 0 2 0$ </td><td> $1 . 9 7 8 \pm 0 . 0 1 8$ </td><td> $0 . 0 8 0 \pm 0 . 0 0 5$ </td></tr><tr><td>DeepSeek-V3.2 (T)</td><td> $1 . 3 9 8 \pm 0 . 0 5 2$ </td><td> $0 . 0 9 3 \pm 0 . 0 3 7$ </td><td> $0 . 3 1 2 \pm 0 . 0 1 6$ </td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td> $1 . 8 4 8 \pm 0 . 0 6 8$ </td><td> $- 0 . 5 3 0 \pm 0 . 0 4 1$ </td><td> $0 . 2 3 7 \pm 0 . 0 0 6$ </td></tr><tr><td>GPT-4o</td><td> $3 . 6 2 8 \pm 0 . 0 1 0$ </td><td> $3 . 3 7 1 \pm 0 . 0 0 9$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 0$ </td></tr><tr><td>GPT-5</td><td> $1 . 5 4 9 \pm 0 . 0 2 2$ </td><td> $0 . 7 6 9 \pm 0 . 0 3 8$ </td><td> $0 . 3 1 2 \pm 0 . 0 1 9$ </td></tr><tr><td>GPT-5-Mini</td><td> $2 . 2 7 8 \pm 0 . 0 2 2$ </td><td> $1 . 6 6 1 \pm 0 . 0 4 2$ </td><td> $0 . 1 9 5 \pm 0 . 0 0 3$ </td></tr><tr><td>Kimi-K2.5</td><td> $2 . 5 9 2 \pm 0 . 0 4 8$ </td><td> $2 . 2 2 3 \pm 0 . 0 3 6$ </td><td> $0 . 0 7 4 \pm 0 . 0 0 5$ </td></tr><tr><td>Qwen3.5-27B</td><td> $2 . 6 1 1 \pm 0 . 0 1 3$ </td><td> $2 . 1 0 0 \pm 0 . 0 1 4$ </td><td> $0 . 1 2 6 \pm 0 . 0 1 3$ </td></tr><tr><td>Qwen3.5-9B</td><td> $2 . 2 4 2 \pm 0 . 0 1 8$ </td><td> $1 . 4 9 8 \pm 0 . 0 3 0$ </td><td> $0 . 2 1 0 \pm 0 . 0 2 4$ </td></tr></table>

Table 23: Mean ± standard deviation of Pearson r, Kendall τ, QWK, and MAE across individual CoT sampling attempts per problem. Thinking-enabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $0 . 5 0 7 \pm 0 . 0 6 7$ </td><td> $0 . 4 1 1 \pm 0 . 0 5 8$ </td><td> $0 . 4 9 3 \pm 0 . 0 5 8$ </td><td> $1 . 1 8 8 \pm 0 . 0 6 6$ </td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td> $0 . 4 6 4 \pm 0 . 0 4 4$ </td><td> $0 . 3 6 7 \pm 0 . 0 4 4$ </td><td> $0 . 4 3 2 \pm 0 . 0 4 1$ </td><td> $1 . 5 6 1 \pm 0 . 0 5 4$ </td></tr><tr><td>GPT-40</td><td> $0 . 4 2 7 \pm 0 . 0 0 5$ </td><td> $0 . 3 6 8 \pm 0 . 0 1 5$ </td><td> $0 . 1 2 0 \pm 0 . 0 0 3$ </td><td> $3 . 0 3 4 \pm 0 . 0 1 0$ </td></tr><tr><td>GPT-5</td><td> $0 . 4 9 0 \pm 0 . 0 2 4$ </td><td> $0 . 3 9 7 \pm 0 . 0 3 1$ </td><td> $0 . 4 5 9 \pm 0 . 0 2 5$ </td><td> $1 . 3 4 1 \pm 0 . 0 7 9$ </td></tr><tr><td>GPT-5-Mini</td><td> $0 . 5 0 3 \pm 0 . 0 1 3$ </td><td> $0 . 3 9 4 \pm 0 . 0 1 4$ </td><td> $0 . 4 2 3 \pm 0 . 0 1 2$ </td><td> $1 . 5 3 8 \pm 0 . 0 2 9$ </td></tr><tr><td>Kimi-K2.5</td><td> $0 . 4 5 2 \pm 0 . 0 3 5$ </td><td> $0 . 3 5 2 \pm 0 . 0 1 8$ </td><td> $0 . 3 9 1 \pm 0 . 0 3 1$ </td><td> $1 . 5 9 0 \pm 0 . 0 5 2$ </td></tr><tr><td>Qwen3.5-27B</td><td> $0 . 3 7 9 \pm 0 . 0 2 1$ </td><td> $0 . 2 4 5 \pm 0 . 0 2 1$ </td><td> $0 . 3 4 6 \pm 0 . 0 2 0$ </td><td> $1 . 7 8 7 \pm 0 . 0 3 7$ </td></tr><tr><td>Qwen3.5-9B</td><td> $0 . 4 0 9 \pm 0 . 0 9 0$ </td><td> $0 . 3 3 6 \pm 0 . 0 7 1$ </td><td> $0 . 3 1 4 \pm 0 . 0 7 0$ </td><td> $1 . 9 0 0 \pm 0 . 1 3 3$ </td></tr></table>

Table 24: Mean ± standard deviation of RMSE, bias, and exact-match accuracy across individual CoT sampling attempts per problem. Thinking-enabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>RMSE</td><td>Bias</td><td>Exact</td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td> $1 . 5 6 3 \pm 0 . 0 6 5$ </td><td> $0 . 2 6 7 \pm 0 . 2 3 5$ </td><td> $0 . 2 6 4 \pm 0 . 0 2 9$ </td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td> $1 . 9 9 4 \pm 0 . 0 8 2$ </td><td> $- 0 . 3 7 4 \pm 0 . 0 5 8$ </td><td> $0 . 2 0 3 \pm 0 . 0 3 7$ </td></tr><tr><td>GPT-4o</td><td> $3 . 3 3 2 \pm 0 . 0 1 1$ </td><td> $2 . 9 8 6 \pm 0 . 0 1 3$ </td><td> $0 . 0 3 1 \pm 0 . 0 0 3$ </td></tr><tr><td>GPT-5</td><td> $1 . 7 0 9 \pm 0 . 0 6 7$ </td><td> $0 . 5 7 6 \pm 0 . 0 4 3$ </td><td> $0 . 2 1 2 \pm 0 . 0 3 3$ </td></tr><tr><td>GPT-5-Mini</td><td> $1 . 9 4 3 \pm 0 . 0 3 1$ </td><td> $0 . 9 9 0 \pm 0 . 0 4 5$ </td><td> $0 . 1 9 3 \pm 0 . 0 3 0$ </td></tr><tr><td>Kimi-K2.5</td><td> $1 . 9 7 8 \pm 0 . 0 7 6$ </td><td> $0 . 8 9 2 \pm 0 . 0 1 6$ </td><td> $0 . 1 7 5 \pm 0 . 0 1 8$ </td></tr><tr><td>Qwen3.5-27B</td><td> $2 . 2 8 8 \pm 0 . 0 4 7$ </td><td> $- 0 . 0 4 0 \pm 0 . 0 2 4$ </td><td> $0 . 1 9 6 \pm 0 . 0 0 7$ </td></tr><tr><td>Qwen3.5-9B</td><td> $2 . 3 3 6 \pm 0 . 1 4 8$ </td><td> $1 . 2 9 0 \pm 0 . 0 8 3$ </td><td> $0 . 1 3 9 \pm 0 . 0 2 5$ </td></tr></table>

Table 25: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.81</td><td>0.73</td><td>0.75</td><td>0.78</td><td>1.12</td><td>0.61</td><td>0.80</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.72</td><td>0.62</td><td>0.42</td><td>2.01</td><td>2.35</td><td>1.98</td><td>0.36</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.77</td><td>0.65</td><td>0.73</td><td>1.00</td><td>1.37</td><td>0.05</td><td>0.79</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.57</td><td>0.47</td><td>0.52</td><td>1.43</td><td>1.89</td><td>-0.48</td><td>0.63</td></tr><tr><td>GPT-40</td><td>0.53</td><td>0.45</td><td>0.12</td><td>3.37</td><td>3.63</td><td>3.37</td><td>0.10</td></tr><tr><td>GPT-5</td><td>0.73</td><td>0.61</td><td>0.64</td><td>1.18</td><td>1.58</td><td>0.78</td><td>0.67</td></tr><tr><td>GPT-5-Mini</td><td>0.65</td><td>0.51</td><td>0.44</td><td>1.79</td><td>2.30</td><td>1.67</td><td>0.52</td></tr><tr><td>Kimi-K2.5</td><td>0.67</td><td>0.55</td><td>0.35</td><td>2.24</td><td>2.59</td><td>2.20</td><td>0.31</td></tr><tr><td>Qwen3.5-27B</td><td>0.69</td><td>0.59</td><td>0.40</td><td>2.19</td><td>2.60</td><td>2.10</td><td>0.35</td></tr><tr><td>Qwen3.5-9B</td><td>0.75</td><td>0.63</td><td>0.54</td><td>1.75</td><td>2.24</td><td>1.51</td><td>0.52</td></tr></table>

Table 26: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.74</td><td>0.65</td><td>0.66</td><td>0.86</td><td>1.36</td><td>0.70</td><td>0.78</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.73</td><td>0.63</td><td>0.42</td><td>2.01</td><td>2.32</td><td>1.98</td><td>0.35</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.81</td><td>0.69</td><td>0.77</td><td>0.92</td><td>1.23</td><td>0.05</td><td>0.83</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.63</td><td>0.53</td><td>0.59</td><td>1.23</td><td>1.65</td><td>-0.51</td><td>0.69</td></tr><tr><td>GPT-40</td><td>0.53</td><td>0.46</td><td>0.12</td><td>3.37</td><td>3.63</td><td>3.37</td><td>0.10</td></tr><tr><td>GPT-5</td><td>0.77</td><td>0.66</td><td>0.69</td><td>1.02</td><td>1.45</td><td>0.73</td><td>0.75</td></tr><tr><td>GPT-5-Mini</td><td>0.67</td><td>0.53</td><td>0.46</td><td>1.72</td><td>2.20</td><td>1.64</td><td>0.52</td></tr><tr><td>Kimi-K2.5</td><td>0.70</td><td>0.58</td><td>0.36</td><td>2.25</td><td>2.58</td><td>2.23</td><td>0.31</td></tr><tr><td>Qwen3.5-27B</td><td>0.69</td><td>0.58</td><td>0.40</td><td>2.17</td><td>2.58</td><td>2.10</td><td>0.37</td></tr><tr><td>Qwen3.5-9B</td><td>0.73</td><td>0.60</td><td>0.52</td><td>1.70</td><td>2.18</td><td>1.49</td><td>0.52</td></tr></table>

Table 27: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), against a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.72</td><td>0.62</td><td>0.64</td><td>0.90</td><td>1.44</td><td>0.74</td><td>0.76</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.73</td><td>0.63</td><td>0.42</td><td>2.01</td><td>2.33</td><td>1.98</td><td>0.35</td></tr><tr><td>DeepSeek-V3.2 (T)</td><td>0.77</td><td>0.65</td><td>0.72</td><td>1.03</td><td>1.41</td><td>0.08</td><td>0.77</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.59</td><td>0.49</td><td>0.54</td><td>1.36</td><td>1.83</td><td>-0.50</td><td>0.66</td></tr><tr><td>GPT-40</td><td>0.52</td><td>0.45</td><td>0.12</td><td>3.38</td><td>3.64</td><td>3.38</td><td>0.10</td></tr><tr><td>GPT-5</td><td>0.77</td><td>0.65</td><td>0.67</td><td>1.09</td><td>1.52</td><td>0.79</td><td>0.73</td></tr><tr><td>GPT-5-Mini</td><td>0.66</td><td>0.52</td><td>0.45</td><td>1.78</td><td>2.29</td><td>1.67</td><td>0.52</td></tr><tr><td>Kimi-K2.5</td><td>0.70</td><td>0.57</td><td>0.36</td><td>2.26</td><td>2.61</td><td>2.25</td><td>0.30</td></tr><tr><td>Qwen3.5-27B</td><td>0.69</td><td>0.58</td><td>0.40</td><td>2.14</td><td>2.56</td><td>2.07</td><td>0.37</td></tr><tr><td>Qwen3.5-9B</td><td>0.73</td><td>0.60</td><td>0.52</td><td>1.76</td><td>2.25</td><td>1.53</td><td>0.52</td></tr></table>

Table 28: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.49</td><td>0.37</td><td>0.46</td><td>1.16</td><td>1.55</td><td>0.48</td><td>0.71</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.50</td><td>0.41</td><td>0.47</td><td>1.50</td><td>1.90</td><td>-0.36</td><td>0.60</td></tr><tr><td>GPT-40</td><td>0.43</td><td>0.36</td><td>0.12</td><td>3.03</td><td>3.32</td><td>2.98</td><td>0.14</td></tr><tr><td>GPT-5</td><td>0.50</td><td>0.40</td><td>0.46</td><td>1.36</td><td>1.72</td><td>0.62</td><td>0.63</td></tr><tr><td>GPT-5-Mini</td><td>0.51</td><td>0.40</td><td>0.44</td><td>1.52</td><td>1.90</td><td>0.93</td><td>0.56</td></tr><tr><td>Kimi-K2.5</td><td>0.41</td><td>0.33</td><td>0.36</td><td>1.64</td><td>2.05</td><td>0.91</td><td>0.52</td></tr><tr><td>Qwen3.5-27B</td><td>0.35</td><td>0.22</td><td>0.32</td><td>1.84</td><td>2.35</td><td>-0.06</td><td>0.51</td></tr><tr><td>Qwen3.5-9B</td><td>0.29</td><td>0.24</td><td>0.22</td><td>2.06</td><td>2.50</td><td>1.26</td><td>0.40</td></tr></table>

Table 29: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.50</td><td>0.41</td><td>0.50</td><td>1.18</td><td>1.55</td><td>0.09</td><td>0.70</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.48</td><td>0.38</td><td>0.46</td><td>1.47</td><td>1.86</td><td>-0.35</td><td>0.61</td></tr><tr><td>GPT-40</td><td>0.43</td><td>0.38</td><td>0.12</td><td>3.04</td><td>3.33</td><td>2.99</td><td>0.14</td></tr><tr><td>GPT-5</td><td>0.49</td><td>0.41</td><td>0.46</td><td>1.30</td><td>1.68</td><td>0.53</td><td>0.66</td></tr><tr><td>GPT-5-Mini</td><td>0.53</td><td>0.43</td><td>0.45</td><td>1.45</td><td>1.85</td><td>0.99</td><td>0.59</td></tr><tr><td>Kimi-K2.5</td><td>0.52</td><td>0.41</td><td>0.44</td><td>1.45</td><td>1.78</td><td>0.91</td><td>0.58</td></tr><tr><td>Qwen3.5-27B</td><td>0.39</td><td>0.26</td><td>0.36</td><td>1.67</td><td>2.19</td><td>-0.10</td><td>0.55</td></tr><tr><td>Qwen3.5-9B</td><td>0.46</td><td>0.38</td><td>0.36</td><td>1.76</td><td>2.15</td><td>1.25</td><td>0.46</td></tr></table>

Table 30: Performance of language models across multiple correlation and error metrics. Thinkingenabled and non-thinking model variants are denoted by T and T, respectively. Results are evaluated on the synthetically-generated dataset, Trojan-Bench(S), without access to a standard markscheme and golden solution.
<table><tr><td>Model</td><td>Pearson r</td><td>Kendall τ</td><td>QWK</td><td>MAE</td><td>RMSE</td><td>Bias</td><td> $\mathrm { \Delta W T A _ { \leq 1 } }$ </td></tr><tr><td>Claude-Sonnet-4.6 (T)</td><td>0.47</td><td>0.37</td><td>0.47</td><td>1.25</td><td>1.63</td><td>0.07</td><td>0.67</td></tr><tr><td>DeepSeek-V4-Flash (T)</td><td>0.51</td><td>0.41</td><td>0.47</td><td>1.50</td><td>1.91</td><td>-0.34</td><td>0.58</td></tr><tr><td>GPT-40</td><td>0.42</td><td>0.36</td><td>0.12</td><td>3.06</td><td>3.35</td><td>3.01</td><td>0.14</td></tr><tr><td>GPT-5</td><td>0.49</td><td>0.40</td><td>0.46</td><td>1.35</td><td>1.72</td><td>0.57</td><td>0.64</td></tr><tr><td>GPT-5-Mini</td><td>0.53</td><td>0.41</td><td>0.45</td><td>1.51</td><td>1.89</td><td>0.96</td><td>0.55</td></tr><tr><td>Kimi-K2.5</td><td>0.49</td><td>0.39</td><td>0.44</td><td>1.54</td><td>1.89</td><td>0.80</td><td>0.55</td></tr><tr><td>Qwen3.5-27B</td><td>0.36</td><td>0.22</td><td>0.33</td><td>1.83</td><td>2.33</td><td>-0.16</td><td>0.51</td></tr><tr><td>Qwen3.5-9B</td><td>0.40</td><td>0.33</td><td>0.31</td><td>1.92</td><td>2.36</td><td>1.25</td><td>0.44</td></tr></table>

## D.3 Additional evaluation of BoN methods

The table below extends the main-text MABGrader comparison to the synthetic benchmark. Table 31 reports the same baselines (CoT@1, MajorityVote@16 with three templates, BoN@16 with four reward models) on Trojan-Bench(S) Full, and confirms that MABGrader continues to lead on QWK and RMSE for the three graders run in this setting. The same pattern we observe on Trojan-Bench(H) holds here: token-level sampling baselines (MajorityVote, BoN with off-the-shelf reward models) yield essentially flat improvements over a single CoT call, while the prompt-side hypothesis enumeration in MABGrader is what produces the consistent gain across graders.

Table 31: Comparative analysis of model inference performance on synthetically generated data across QWK, RMSE, and Bias metrics for DeepSeek-V4-Flash, Kimi-K2.5, Gemini-3-Pro-Preview, and Qwen3.5-27B. Evaluation is performed with a reference solution and a corresponding marking scheme.
<table><tr><td></td><td colspan="4">QWK (↑)</td><td colspan="4">RMSE (↓)</td><td colspan="4">Bias (≈ 0)</td></tr><tr><td></td><td>DS V4F</td><td>Kimi K2.5</td><td>Gem. 3P</td><td>Q3.5 27B</td><td>DS- V4F</td><td>Kimi K2.5</td><td>Gem. 3P</td><td>Q3.5 27B</td><td>DS- V4F</td><td>Kimi K2.5</td><td>Gem. 3P</td><td>Q3.5 27B</td></tr><tr><td colspan="9">Chain-of-Thought (CoT@ 1)</td><td></td><td></td><td></td><td></td></tr><tr><td>MathArena [4]</td><td>0.52</td><td>0.35</td><td></td><td>0.40</td><td>1.89</td><td>2.59</td><td></td><td>2.60</td><td>-0.48</td><td>2.20</td><td></td><td>2.10</td></tr><tr><td colspan="9">Chain-of-Thought with Majority Vote Selection (MajorityVote@ 16)</td><td></td><td></td><td></td><td></td></tr><tr><td>MathArena [4]</td><td>0.54</td><td>0.36</td><td></td><td>0.41</td><td>1.83</td><td>2.56</td><td></td><td>2.55</td><td>-0.50</td><td>2.25</td><td></td><td>2.20</td></tr><tr><td colspan="9">Chain-of-Thought with Best-of-N Selection (BoN@16)</td><td></td><td></td><td></td><td></td></tr><tr><td>Skywork [31]</td><td>0.51</td><td>0.34</td><td></td><td>0.41</td><td>1.81</td><td>2.61</td><td></td><td>2.67</td><td>-0.52</td><td>2.35</td><td></td><td>2.15</td></tr><tr><td>ArmoRM [56]</td><td>0.55</td><td>0.35</td><td></td><td>0.41</td><td>1.79</td><td>2.67</td><td></td><td>2.64</td><td>-0.43</td><td>2.32</td><td></td><td>2.11</td></tr><tr><td>RLHFlow [14]</td><td>0.53</td><td>0.35</td><td></td><td>0.39</td><td>1.85</td><td>2.62</td><td></td><td>2.61</td><td>-0.41</td><td>2.28</td><td></td><td>2.18</td></tr><tr><td>Shepherd [57]</td><td>0.52</td><td>0.36</td><td></td><td>0.37</td><td>1.89</td><td>2.66</td><td>=</td><td>2.70</td><td>-0.40</td><td>2.30</td><td></td><td>2.20</td></tr><tr><td colspan="9">Proposed Method (MABGrader@ 16)</td><td></td><td></td><td></td><td></td></tr><tr><td>MABGrader</td><td>0.58</td><td>0.49</td><td>=</td><td>0.53</td><td>1.67</td><td>2.24</td><td>1</td><td>1.90</td><td>-0.50</td><td>0.71</td><td>=</td><td>-0.96</td></tr></table>

## E Extended Bar plots

The bar plots in Figures 6 - 17 are a visual companion to the numerical tables of the previous section. Each plot reports a single metric family (correlation, error, or agreement) for one (dataset, style) pair, with three bars per model corresponding to CoT@1, Mean-CoT@16, and MajorityVote@16. We include them so that a reader can compare every grader and every aggregation method at a glance for any single metric, without having to cross-reference the numerical tables.

The plots make the patterns from the previous section visually obvious. Frontier thinking-enabled models cluster in the upper band of the correlation plots and in the lower band of the error plots, whereas smaller graders separate cleanly into a lower-correlation, higher-MAE cluster. The agreement plots (Exact Match and WTA ≤ 1) show that close-but-not-exact agreement with the human reference is achievable for most frontier models, while exact agreement remains low even for the strongest graders. The synthetic-set plots show the same ordering as the human-set plots but with a uniform upward shift in correlation and a downward shift in error, consistent with the cross-set discussion above.

[Full] [Trojan-Bench (H)] Pearson r by Model & Sampling Method  
![](images/4611ee262a8a833b330771b62af5be6927e0b72a12b27a272748ec7e40e0a627.jpg)

[Full] [Trojan-Bench (H)] QWK by Model & Sampling Method  
![](images/1e540fc01c5aa380b6441d80692b2faaf5f2aec705e921531e0abf4ae466dbf2.jpg)

[Full] [Trojan-Bench (H)] Kendall τ by Model & Sampling Method  
![](images/a5583ee6da44387f017ceea074ea7223c5ea4fe165deb3ff40d9ddc7bcf66591.jpg)  
Figure 6: Trojan-Bench(H) (Full): Correlation metrics (Pearson r, QWK, Kendall τ ) by model and sampling method.

[Full] [Troian-Bench (H)] MAE by Model & Sampling Method  
![](images/37a70a37cbac1003c929af7026b1c9acdabec978a757a74c765ae008f3944cbe.jpg)

[Full] [Trojan-Bench (H)] RMSE by Model & Sampling Method  
![](images/081ba236de54e1be0b957c89e36e334d234dfde2601d217b1546e02abf21699a.jpg)

[Full] [Trojan-Bench (H)] Bias by Model & Sampling Method  
![](images/bd8c1531aaa5f54390e6141d29e45c318ea5df31eb0123fecffb8639f6a088a7.jpg)  
Figure 7: Trojan-Bench(H) (Full): Error metrics (MAE, RMSE, Bias) by model and sampling method.

[Full] [Trojan-Bench (H)] Exact Match by Model & Sampling Method  
![](images/3f45844f3c242bd1a9ca79c7c8125ab02515417b0f815ca8d6c7afa234d99609.jpg)

[Full] [Trojan-Bench (H)] WTA (≤1) by Model & Sampling Method  
![](images/a8c7cfe7a0eccd5f5b1d31042fff37f91e5610dc966a9dfaa16ae9baf750c258.jpg)  
Figure 8: Trojan-Bench(H) (Full): Agreement metrics (Exact Match, WTA ≤ 1) by model and sampling method.

[Blind] [Trojan-Bench (H)] Pearson r by Model & Sampling Method  
![](images/4b7483e124a45ccd610a40ec8d36221bbf641905aa34842df343043d658019d3.jpg)

[Blind] [Trojan-Bench (H)] QWK by Model & Sampling Method  
![](images/916b8d3797df9a4db8374bc78d9fe2a30c4f885b743d52564436f93ca72a6541.jpg)

[Blind] [Trojan-Bench (H)] Kendall τ by Model & Sampling Method  
![](images/e2b476c74b205d3de4866640ea920ad79a28478036fc9e14f96a5be017adb672.jpg)  
Figure 9: Trojan-Bench(H) (Blind): Correlation metrics (Pearson r, QWK, Kendall τ) by model and sampling method.

[Blind] [Trojan-Bench (H)] MAE by Model & Sampling Method  
![](images/1274947f24dcfafc89ab4ef562b89d25774e9f902308ade2ae641a82298cb36e.jpg)

[Blind] [Trojan-Bench (H)] RMSE by Model & Sampling Method  
![](images/f7e6fa04f1332bcfbb565e58381a34f6374df67141b46486e1742bde290c74a1.jpg)

[Blind] [Trojan-Bench (H)] Bias by Model & Sampling Method  
![](images/6173ab6038ece9fbe0eedd27457c3f404b62df12a06dc2a356ebe756a01fe2e0.jpg)  
Figure 10: Trojan-Bench(H) (Blind): Error metrics (MAE, RMSE, Bias) by model and sampling method.

[Blind] [Trojan-Bench (H)] Exact Match by Model & Sampling Method  
![](images/49ded77ae99f821bddc9da7736cbcb383ee0e466684068bee8c384fac0693c10.jpg)

[Blind] [Trojan-Bench (H)] WTA (≤1) by Model & Sampling Method  
![](images/e3e3f02c8d2ee23fca474317c7d87265242ff01772d8dbd4a9172cd131c4f2a8.jpg)  
Figure 11: Trojan-Bench(H) (Blind): Agreement metrics (Exact Match, $\mathbf { W T A } \leq 1 )$ by model and sampling method.

[Full] [Trojan-Bench (S)] Pearson r by Model & Sampling Method  
![](images/e001cc7b3852bc392aa526bb6a020f57524961dcc98e83ef7d649710ffd69d7a.jpg)

[Full] [TrJan-Bench (S)] QWK by Model & Sampling Method  
![](images/43e0d5705744226f0f219b36a7e097705a86560145351b8602342148c7818ae1.jpg)

[Full] [Trojan-Bench (S)] Kendall τ by Model & Sampling Method  
![](images/f8d66c73549f344ef2f7caf5b7145dccbe2ec8a9a505a3bda2cb308af9b9ffcf.jpg)  
Figure 12: Trojan-Bench(S) (Full): Correlation metrics (Pearson r, QWK, Kendall τ) by model and sampling method.

[Full] [Trojan-Bench (S)] MAE by Model & Sampling Method  
![](images/632bfc4bec8a7c15d51dea3298112601e0febfed0c4d327272cf1f3266fca09a.jpg)

[Full] [Trojan-Bench (S)] RMSE by Model & Sampling Method  
![](images/bc27ba598638f7a055b2dfa9b01146a45a7fc05cf10dc70d60faa54220644c9e.jpg)

[Full] [Trojan-Bench (S)] Bias by Model & Sampling Method  
![](images/4c9d0f6761cb21ee59720fa2e7857d56b1bcfcb095cf2ed32646f8bfa849740c.jpg)  
Figure 13: Trojan-Bench(S) (Full): Error metrics (MAE, RMSE, Bias) by model and sampling method.

[Full] [Trojan-Bench (S)] Exact Match by Model & Sampling Method  
![](images/77c793b01df646eeb79135bccf66d9d42b9962ef1d02ac77fde90a4b3f4db4fe.jpg)

[Full] [Trojan-Bench (S)] WTA (≤1) by Model & Sampling Method  
![](images/49ef94e3a48daa1130aa7f456c1b24772fb977e3160aba4c54abf14c7ddb0aec.jpg)  
Figure 14: Trojan-Bench(S) (Full): Agreement metrics (Exact Match, $\mathbf { W T A } \leq 1 )$ by model and sampling method.

[Blind] [Trojan-Bench (S)] Pearson r by Model & Sampling Method  
![](images/e1936971733142c8952e43dbcf910afe5386dd535fff8bd63de6d7a815de150a.jpg)

[Blind] [Trojan-Bench (S)] QWK by Model & Sampling Method  
![](images/f2d3f9dfe50ca041e4cce77d1e3d4eb7b46cc3ed18def61bbb2d9dc0fdbac1af.jpg)

[Blind] [Trojan-Bench (S)] Kendall τ by Model & Sampling Method  
![](images/e4076618ad7d47a1b3caf941da9332cc70b1ac276df85f2f87f9dcf215225fe9.jpg)  
Figure 15: Trojan-Bench(S) (Blind): Correlation metrics (Pearson r, QWK, Kendall τ) by model and sampling method.

[Blind] [Trojan-Bench (S)] MAE by Model & Sampling Method  
![](images/7c213adffc04a4a2ee3d9ba387de1e0b28f1b2777558f71c55989e776b39a88e.jpg)

[Blind] [Trojan-Bench (S)] RMSE by Model & Sampling Method  
![](images/66da795b249c3fa264b29826613965e69926e65a96294dd53231277e5e9a77ad.jpg)

[Blind] [Trojan-Bench (S)] Bias by Model & Sampling Method  
![](images/4c8dc042af01ad55171131692081bd311773e8b7714cec6df9687b8acc06a881.jpg)  
Figure 16: Trojan-Bench(S) (Blind): Error metrics (MAE, RMSE, Bias) by model and sampling method.

[Blind] [Trojan-Bench (S)] Exact Match by Model & Sampling Method

![](images/22dddf324078ab28d55d92100ac057cf3052e6a97e386af1de9e6af9008df930.jpg)

[Blind] [Trojan-Bench (S)] WTA (≤1) by Model & Sampling Method  
![](images/86b9081779d39950a76f9169eaac027e502812fdbf2a6cd9410a0dd90d8b710e.jpg)  
Figure 17: Trojan-Bench(S) (Blind): Agreement metrics (Exact Match, $\mathbf { W T A } \leq 1 )$ by model and sampling method.

## F Evaluation of LLM correlation with human and other LLM annotators

The heatmaps in Figures 18–33 report pairwise agreement between every pair of graders in our suite (LLM-vs-LLM cells) together with their agreement with the human reference (LLM-vs-human row and column). Each figure covers one metric and one dataset, with the upper half reporting the Blind template and the lower half reporting the Full template. We include them so that the inter-grader correlation structure is visible at a glance, since a single number per (grader, metric) cell does not capture how strongly graders agree with each other rather than with the reference. All statistics cited in this section are recomputed from the MajorityVote@16 sampling that produced the heatmaps, and the discussion is restricted to the 14 graders run on Trojan-Bench(H) in both templates, and the 8 graders run on Trojan-Bench(S).

The most striking pattern across both datasets is that hiding the markscheme degrades ordinal agreement with the human reference far more than it degrades the magnitude of grading error. On Trojan-Bench(H) the mean Human-vs-LLM Pearson r drops from 0.50 to $0 . 3 9 \left( \Delta = - 0 . 1 1 \right)$ , Kendall τ from 0.42 to 0.30, and QWK from 0.41 to 0.33, yet mean absolute error is essentially unchanged (1.99 vs 2.00) and RMSE actually edges down slightly $( 2 . 7 1  2 . 6 0 )$ . The $T r o j a n { \mathrm { - } } B e n c h ( S )$ set shows the same dissociation more sharply: Pearson r collapses by $0 . 2 1 \ : ( 0 . 6 7  0 . 4 6 )$ and Kendall τ by 0.20, while MAE improves under Blind (1.83 → 1.74). Removing the markscheme does not cause judges to pick wildly different absolute numbers, since they still place most attempts within roughly one point of the reference, but it strips them of the ability to order attempts correctly.

A second pattern is a near-universal sign flip in bias on Trojan-Bench(H), and a much subtler one on Trojan-Bench(S). Under the Full setting the average judge is stricter than the human (mean bias $= - 1 . 4 8 ) ;$ under Blind that average flips to +0.51, a swing of almost two grade points, and 10 of 14 Trojan-Bench(H) judges individually change sign, including $\mathrm { G P T { - } } 5 ( \mathrm { - 0 . 9 8  + 0 . 5 6 } )$ , Claude 4.6 thinking $( - 1 . 2 3  + 0 . 1 3 )$ , DeepSeek v3.2 thinking $( - 0 . 1 0  + 1 . 0 8 )$ , and DeepSeek v3.2 no-thinking $( - 2 . 0 8  + 3 . 0 2 )$ . Only Gemini 3 Pro doubles down on harshness without the rubric $( - 1 . 3 8 \to - 1 . 5 8 )$ , and GPT-4o together with Qwen3-8B remain stricter than the human in both regimes. On Trojan-Bench(S), where the reference is itself an LLM-rubric consensus rather than a human grade, only 1 of 8 judges flips bias sign, which confirms that the flip on Trojan-Bench(H) is a real markscheme-versus-human-style effect rather than an artifact of model calibration. Read together with the previous observation, the markscheme behaves like a penalty list: when it is taken away, judges default to a more generous approach.

The LLM-vs-LLM blocks of the heatmaps reveal that consensus between judges itself depends heavily on the rubric and depends on it more strongly than human alignment does. Under Full, the frontier reasoning graders form a tight high-agreement block, with pairwise Pearson typically above 0.7 and pairwise QWK above 0.6, well above their mean agreement with the human reference. Hiding the rubric pulls this block apart: on Trojan-Bench(H) the mean inter-LLM Pearson r drops from 0.64 to 0.44 and its pairwise standard deviation almost doubles $( 0 . 1 5  0 . 2 8 )$ , with DeepSeek v3.2 no-thinking becoming essentially decorrelated from every other judge $( | r | \leq 0 . 3 0$ , including $r = - 0 . 4 5$ against GPT-4o) and GPT-4o losing all coherence with the strong-reasoning cluster. The Trojan-Bench(S) inter-LLM block weakens less under rubric removal (mean r from 0.63 to 0.54, $\Delta = 0 . 0 9 )$ , and there the gap between LLM-consensus and Human-agreement actually flips sign: with the rubric, LLMs match the synthetic ground truth slightly better than they match each other $( \mathrm { g a p - 0 . 0 4 } )$ , but without it they revert to agreeing with each other more than with the reference (gap +0.09). Per-model rubric gains are also heterogeneous: frontier reasoning models gain large boosts (DeepSeek v3.2 no-thinking +0.35 in r, Gemini 2.5 Flash +0.29, GPT-5 +0.18), whereas Qwen3-8B, Qwen3.5-27B, and Kimi-K2.5 actually lose a small amount of human alignment when given the rubric. Taken together, the markscheme is not a uniform injection of correct information but a structural scaffold whose benefit scales with the judge’s reasoning capacity, and whose absence pushes weaker judges toward random consensus and stronger judges toward systematic leniency. The negative-rubric-gain pattern for small graders is examined further in Section J.1.

[Trojan-Bench (H)] [Pearson r] Blind and Full inference Heatmap across Evaluated Models  
[Blind]  
![](images/42b2bc3a94610c2fe72c5b531150bddac04c0e3a6fd867b7d231e2e9bbc29234.jpg)

[Full]  
![](images/4e1cf00083b771a3392174fcbc57ec246a0ca805d5723c5e4d9e0c7130f718f2.jpg)  
Figure 18: Trojan-Bench(H) : Pearson r blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full).

[Trojan-Bench (H)] [QWK] Blind and Full inference Heatmap across Évaluated Models [Blind]  
![](images/6d0557acc5044add759e7e2767bb716cee2c142c8015826211da6da1bb85f41a.jpg)

[Full]  
![](images/662677943bfac1d1fdb1ef2c39e5f70b916c705583dddba543d056443e8ee797.jpg)  
Figure 19: Trojan-Bench(H) : QWK blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full).

[Trojan-Bench (H)] [Kendall tau] Blind and Full inference Heatmap across Evaluated Models  
[Blind]  
![](images/b6e2dd781e70b6feb11dff01dbe05dce5eb79fa9470812bd2da3f24850059670.jpg)

[Full]  
![](images/aec7c5f92149edbe9d418dc2ccdbbcc7ebfd8627573c741931b43dcbae82f055.jpg)  
Figure 20: Trojan-Bench(H) : Kendall τ blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full).

[Trojan-Bench (H)] [MAE] Blind and Full inference Heatmap across Évaluated Models [Blind]  
![](images/fbf10926e081240238906beae9914aa17ea9eb69a4d9b02a44d990289e0ae776.jpg)

[Full]  
![](images/444e158daa90a5c99306e87a7d7ead307fde998fcf1442545cbd28ddf43bebeb.jpg)  
Figure 21: Trojan-Bench(H) : MAE blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full).

[Trojan-Bench (H)] [RMSE] Blind and Full inference Heatmap across Evaluated Models [Blind]  
![](images/418f7c20c57c12539a82794582d9de46a95aebf066fc37986ca6a06c7db146fd.jpg)

[Full]  
![](images/9bf6a62f9ac3c036f61c2166d4bcff2f46aa0873f32d1a2acff608197fd1b4b6.jpg)  
Figure 22: Trojan-Bench(H) : RMSE blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full).

[Trojan-Bench (H)] [Bias] Blind and Full inference Heatmap across Evaluated Models [Blind]  
![](images/c57b858e69764d0b375d05de867b917c4a8f5c2a23041a94a108a906cc3ed007.jpg)

[Full]  
![](images/296681f497198800d8fd560397ab0700a2032673ea6a1345d4f3176892291b67.jpg)  
Figure 23: Trojan-Bench(H) : Bias blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full).

[Trojan-Bench (H)] [Exact Match] Blind and Full inference Heatmap across Evaluated Models

[Blind]  
![](images/1ad14b1161e1ca0713020b5690810f731647e515b32c05f37d2db59750b6f3bb.jpg)

[Full]  
![](images/184cb3eaebf0db3865430d474a3edf9da7273cd9412470f98e092a24a9b73515.jpg)  
Figure 24: Trojan-Bench(H) : Exact Match blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full).

[Trojan-Bench (H)] [WTA (<=1)] Blind and Full inference Heatmap across Evaluated Models  
[Blind]  
![](images/7b2d9a5972c6a9c8bf7afd773bc6aa8f1b83d345d9e3a8a95ac9123a244028c6.jpg)

[Full]  
![](images/4e084c170e84454bb9f223d444f3219d43a2de78d9e2b972bc4b16f1de8d1a5b.jpg)  
Figure 25: Trojan-Bench(H) : WTA (≤ 1) blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full).

[Trojan-Bench (S)] [Pearson r] Blind and Full inference Heatmap across Evaluated Models [Blind]  
![](images/d1a32491bbea1807a15bd4cfff5482e1615941a33f83e3329a03461c6c5dadd5.jpg)

[Full]  
![](images/b28bc97616b95b166d6e193660d42728e31306077a88ce6c053e7d149d1ac9c7.jpg)  
Figure 26: Trojan-Bench(S) : Pearson r blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full). Note that "human" in this heatmap indicates an LLM assigned grade during data generation with TROJAN and serves as a reference point when evaluating the LLMs.

[Trojan-Bench (S)] [QWK] Blind and Full inference Heatmap across Évaluated Models [Blind]  
![](images/cf22ff6da0df2e03b86987f36bff5875950e613155b38097a223fc18ca578737.jpg)

[Full]  
![](images/c2df435419d1711218b719bbebacedbbda7b1ba233940cb18be5205db19e9989.jpg)  
Figure 27: Trojan-Bench(S) : QWK blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full). Note that "human" in this heatmap indicates an LLM assigned grade during data generation with TROJAN and serves as a reference point when evaluating the LLMs.

[Trojan-Bench (S)] [Kendall tau] Blind and Full inference Heatmap across Evaluated Models [Blind]  
![](images/695e2ad360a6f778c26cc872b10a75078f98cb4fc127db64f82225de6636946a.jpg)

[Full]  
![](images/d2529b86aed863fa788a1fde3f62a814926b3b9f30b0d83605bc22d41296af8d.jpg)  
Figure 28: Trojan-Bench(S) : Kendall τ blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full). Note that "human" in this heatmap indicates an LLM assigned grade during data generation with TROJAN and serves as a reference point when evaluating the LLMs.

[Trojan-Bench (S)] [MAE] Blind and Full inference Heatmap across Evaluated Models [Blind]  
![](images/1d147fa9ca6bdf123ea484d9cfbc093c2e5c526bec66bd87cc477bdd53cc3981.jpg)

[Full]  
![](images/42b1c072ce38bc26e6177024879a3453821239cfd9fcf3587b041067e28054e1.jpg)  
Figure 29: Trojan-Bench(S) : MAE blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full). Note that "human" in this heatmap indicates an LLM assigned grade during data generation with TROJAN and serves as a reference point when evaluating the LLMs.

[Trojan-Bench (S)] [RMSE] Blind and Full inference Heatmap across Évaluated Models [Blind]  
![](images/8aabbb10ba95f667d441ba7192bc990a2288b355e431d498749f27633a69c359.jpg)

[Full]  
![](images/0fa4aab7207124c3313153a84230bbb5b9f6e55d27fa7a8ec09904bdeae8d3f4.jpg)  
Figure 30: Trojan-Bench(S) : RMSE blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full). Note that "human" in this heatmap indicates an LLM assigned grade during data generation with TROJAN and serves as a reference point when evaluating the LLMs.

[Trojan-Bench (S)] [Bias] Blind and Full inference Heatmap across Evaluated Models [Blind]  
![](images/66b8d6a468e76cb56ef9fc117d2705e838adb1bc149b991f788b87efc6cb8521.jpg)

[Full]  
![](images/a3cc04fc043c169409c2316dcd284deee4199a604dd0431dc4812921c42ea5fa.jpg)  
Figure 31: Trojan-Bench(S) : Bias blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full). Note that "human" in this heatmap indicates an LLM assigned grade during data generation with TROJAN and serves as a reference point when evaluating the LLMs.

[Trojan-Bench (S)] [Exact Match] Blind and Full inference Heatmap across Evaluated Models [Blind]  
![](images/da559f520fab1c2e70198f5b7636661807d588666bf076c1c7fc6d935e7cb795.jpg)

[Full]  
![](images/76364c7ef011fb0a91742740339c85a247bc8138e48b547e7c5a19fed33964ef.jpg)  
Figure 32: Trojan-Bench(S) : Exact Match blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full). Note that "human" in this heatmap indicates an LLM assigned grade during data generation with TROJAN and serves as a reference point when evaluating the LLMs.

[Trojan-Bench (S)] [WTA (<=1)] Blind and Full inference Heatmap across Evaluated Models [Blind]  
![](images/0d2d5d8755e36a1c5b83b41231ff182123f900fc456e0682911e1f0a11499064.jpg)

[Full]  
![](images/dba85f1e539564515d6304f7b4ca034b1ec074c1eabb2a7f1266470cbad131e2.jpg)  
Figure 33: Trojan-Bench(S) : WTA (≤ 1) blind and full inference heatmap across evaluated models (Majority Vote; top: Blind, bottom: Full). Note that "human" in this heatmap indicates an LLM assigned grade during data generation with TROJAN and serves as a reference point when evaluating the LLMs.

## G Annotation procedure

Figure 34 shows the custom web-based annotation tool that two domain experts used to grade the 48 candidate solutions of the Trojan-Bench(H) benchmark. The left panel lists the 48 problems with a counter at the top tracking annotation progress. The main panel shows, in order, the problem statement, the official marking scheme broken into criteria and their point values, the golden reference solution, and the candidate solution to grade. The annotator selects an integer score in [0, 7] from a drop-down, writes a free-form justification, and advances to the next problem. The tool stores annotations as JSON, which is the format we used to compute the inter-annotator agreement reported in the main text.

We followed the two-pass procedure of [45]. Each annotator graded all 48 problems independently. The disagreements were inspected jointly, and a single consensus grade was committed for each problem. The expert-vs-expert agreement statistics are reported in the main text and indicate that the human reference is reliable enough to serve as the gold standard for the LLM evaluation that follows.

![](images/7d400bb5c081c1e0c624b38ba3ae8c7601e803840be562c5408f1c2d37ebedc8.jpg)  
Figure 34: Annotation website used by two domain experts to grade USAMO 2025 candidate solutions. The interface presents the problem statement, marking scheme, golden solution, and candidate solution side by side, and stores per-problem scores together with free-form justifications as JSON.

## H Additional details on SFT training

Tables 32 and 33 document the supervised fine-tuning configurations used to produce the syntheticdata-trained variants of Qwen3.5-9B and Qwen3.5-27B reported in the main text. The 9B model is trained with full fine-tuning, the 27B model with LoRA at rank 8, and both share the data mixture (Full and Blind Trojan-Bench(S) samples from four competitions), the optimisation schedule, and the framework (LLaMA-Factory). We include the configurations for reproducibility.

Table 32: Supervised fine-tuning configuration for Qwen3.5-9B.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td colspan="2">Model</td></tr><tr><td>Base model</td><td>Qwen3.5-9B</td></tr><tr><td>Tuning regime</td><td>Full fine-tuning</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td colspan="2">Data</td></tr><tr><td>Training corpus</td><td>Synthetic data from four competitions</td></tr><tr><td>Verification settings</td><td>Full and Blind</td></tr><tr><td>Train/validation split</td><td>95/5</td></tr><tr><td>Maximum context length</td><td>16K tokens</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Peak learning rate</td><td>1e-5</td></tr><tr><td>LR schedule</td><td>Cosine</td></tr><tr><td>Warmup ratio</td><td>10%</td></tr><tr><td colspan="2">Batching &amp; Schedule</td></tr><tr><td>Effective global batch size</td><td>16</td></tr><tr><td>Epochs</td><td>2</td></tr><tr><td colspan="2">System</td></tr><tr><td>Training framework</td><td>LLaMA-Factory</td></tr></table>

Table 33: Supervised fine-tuning configuration for Qwen3.5-27B.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Model</td><td></td></tr><tr><td>Base model</td><td>Qwen3.5-27B</td></tr><tr><td>Tuning regime</td><td>LoRA</td></tr><tr><td>LoRA rank (r)</td><td>8</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Data</td><td></td></tr><tr><td>Training corpus</td><td>Synthetic data from four competitions</td></tr><tr><td>Verification settings</td><td>Full and Blind</td></tr><tr><td>Train/validation split</td><td>95/5</td></tr><tr><td>Maximum context length</td><td>16K tokens</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Peak learning rate</td><td>1e-4</td></tr><tr><td>LR schedule</td><td>Cosine</td></tr><tr><td>Warmup ratio</td><td>10%</td></tr><tr><td>Batching &amp; Schedule</td><td></td></tr><tr><td>Effective global batch size</td><td>16</td></tr><tr><td>Epochs</td><td>2</td></tr><tr><td>System</td><td></td></tr><tr><td>Training framework</td><td>LLaMA-Factory</td></tr></table>

Table 34: Parameter c ablation with DeepSeek-V4-Flash-Thinking base model.
<table><tr><td>UCB exploration c</td><td>QWK</td><td>RMSE</td><td>Bias</td></tr><tr><td> $c = 0 . 1 0$ </td><td>0.641</td><td>2.38</td><td>-1.17</td></tr><tr><td> $c = 0 . 2 5$ </td><td>0.657</td><td>2.06</td><td>-1.04</td></tr><tr><td> $c = 0 . 5 0$ </td><td>0.634</td><td>2.59</td><td>-1.34</td></tr><tr><td> $c = 0 . 7 5$ </td><td>0.613</td><td>2.69</td><td>-1.40</td></tr><tr><td> $c = 1 . 0 0$ </td><td>0.616</td><td>2.46</td><td>-1.47</td></tr></table>

## I Hyperparameter c ablation in MABGrader

Table 34 reports the effect of the UCB exploration coefficient c on MABGrader when the base grader is DeepSeek-V4-Flash-Thinking. We sweep $c \in \{ 0 . 1 0 , 0 . 2 5 , 0 . 5 0 , 0 . 7 5 , 1 . 0 0 \}$ and report QWK, RMSE, and bias on the human-annotated USAMO 2025 benchmark with 16 oracle pulls per problem. The metric profile is non-monotone in c. Setting c = 0.25 achieves the best QWK (0.657), the lowest RMSE (2.06), and a near-best bias (−1.04). Larger values $( c \ge 0 . 5 )$ over-explore weak arms and degrade all three metrics, while $c = 0 . 1 0$ under-explores and converges too eagerly to the first arm whose oracle reward looks plausible. We therefore use $c = 0 . 2 5$ for all main-text MABGrader results, paired with 16 iterations as documented in Section 5.

## J Case studies

## J.1 Behavior of smaller models on Full and Blind settings

A counter-intuitive effect emerges from our Trojan-Bench(H) evaluation, where the three smaller LLMs we study, i.e., Qwen3-8B, GPT-5 Mini, and Kimi-K2.5, correlate better with the human reference under the Blind template (no rubric provided) than under the Full template (problem + golden solution + markscheme). The Blind template strictly removes information from the prompt, so intuitively, the Full template should grade more reliably. Yet on Trojan-Bench(H) every one of the three small graders has lower MAE and higher QWK under Blind than under Full:

<table><tr><td rowspan="2">Grader</td><td colspan="2">Pearson r</td><td colspan="2">Kendall τ</td><td colspan="2">QWK</td><td colspan="2">MAE</td></tr><tr><td>Blind</td><td>Full</td><td>Blind</td><td>Full</td><td>Blind</td><td>Full</td><td>Blind</td><td>Full</td></tr><tr><td>Qwen3-8B</td><td>0.196</td><td>0.180</td><td>0.154</td><td>0.186</td><td>0.175</td><td>0.076</td><td>2.44</td><td>3.07</td></tr><tr><td>GPT-5 Mini</td><td>0.458</td><td>0.519</td><td>0.337</td><td>0.436</td><td>0.455</td><td>0.421</td><td>1.77</td><td>1.92</td></tr><tr><td>Kimi-K2.5</td><td>0.434</td><td>0.412</td><td>0.319</td><td>0.328</td><td>0.372</td><td>0.282</td><td>1.88</td><td>2.23</td></tr></table>

The QWK and MAE gaps are large (for example, +0.13 QWK and −0.36 MAE for Kimi-K2.5). Pearson r and Kendall τ are also higher with an exception of GPT-5 Mini. QWK and MAE penalise the much larger errors the Full template commits on weak (0 to 2 points) answers.

Six concrete cases (two per model). The six text boxes below summarise, for a single (problem, candidate-solution) pair at a time, the verdicts the same grader returns under the Blind and Full templates. Each box reports the mean grade and the per-instance MAE under each template, gives an aggregated per-instance gap, and excerpts representative free-form text from the grader on each side. The cases were selected as the largest Blind-vs-Full instance gaps within each model on the Trojan-Bench(H) dataset.

## Case A. Qwen3-8B on Trojan-Bench(H) P3. Human grade = 0.

Aggregate grading behaviour on this single (problem, candidate-solution) pair.   
Blind (16 attempts): mean grade 0.85, instance MAE against the human reference 0.80.   
Full (16 attempts): mean grade 5.75, instance MAE against the human reference 5.75.   
Per-instance gap: +4.90 pts of inflation under the full template.

Blind grader (representative response). "The solution isfundamentallyflawed. The key error lies in the definition ofthe Bob-graph, which is unrelated to theproblem’s road construction condition. . . . The disk with diameter AB (radius 0.5) cannot contain any other cities due to the 1-unit distance constraint, making the Bob-graph empty and disconnected. The claim that the graph is connected and planar is invalid, and the conclusion that Bob wins is incorrect."

Full grader (representative response). Awards the rubric items for Connectivity, Finiteness of shortest paths, and Planarity, with reasons such as "the construction is correct and induces a graph that is connected" and "the construction is correct and induces a planar graph". Only the Finishing item is denied, on the grounds that the solution "incorrectly concludes that Bob wins."

What this case shows. The same model awards positive credit to three rubric items (connectivity, finiteness, planarity) whose underlying claims it has just declared invalid under the blind template. The model mathces general approach to the markscheme without deeper analysis on the correctness of the response.

## Case B. Qwen3-8B on Trojan-Bench(H) P5. Human grade = 0.

Aggregate grading behaviour on this single (problem, candidate-solution) pair.   
Blind (16 attempts): mean grade 2.23, instance MAE against the human reference 2.23.   
Full (16 attempts): mean grade 6.94, instance MAE against the human reference 6.94. The full template returned a perfect grade in nearly every attempt.   
Per-instance gap: +4.71 pts of inflation under the full template.

## Blind grader (representative response).

"The solution correctly identifies that odd k fail by testing $n = 2 ,$ but the prooffor even k is fundamentally flawed. The key error lies in the induction step where the solution incorrectly shifts the problem to a different sum and fails to address the original requirement that $\begin{array} { r } { \frac { 1 } { n + 1 } \sum _ { i = 0 } ^ { n } { \binom { n } { i } } ^ { k } } \end{array}$ must be an integer. The inductive argument does not establish divisibility by $n + 1 ,$ and the reindexing step is not rigorouslyjustified, rendering the entire solution incorrect."

## Full grader (representative response).

Awards every rubric item its maximum credit. The reduction line accepts that the solution "Uses Chinese Remainder Theorem to reduce the problem to a system of equations modulo $p _ { l } ^ { e _ { l } \prime \prime } $ ; the two remainder lines (for the cases when p does not divide j and when p divides j) are similarly granted on the basis that the solution "represents the term $\textstyle { \binom { n } { i } }$ as a product of fractions $\textstyle { \frac { n - j } { j } }$ andfinds the remainder ofthe terms".

What this case shows. A free-form rejection ("the inductive argument does not establish divisibility by $n + 1 " )$ is overridden by surface-form pattern matching.

## Case C. GPT-5 Mini on Trojan-Bench(H) P6. Human grade = 0.

## Aggregate grading behaviour on this single (problem, candidate-solution) pair.

Blind (16 attempts): mean grade 0.75, instance MAE against the human reference $_ { 0 . 7 5 . }$ Full (16 attempts): mean grade 5.62, instance MAE against the human reference 5.62. Per-instance gap: +4.87 pts of inflation under the full template.

## Blind grader (representative response).

"The submitted solution is incorrect and fails to rigorously prove the claim. The main fatalflaw is the very first modeling step: an edge is defined between person p and cupcake b iff p values b at least 1. . . . Subsequent arguments (application of Hall’s theorem, the entire removal/merging procedure) rely on that incorrect model and contain additional imprecise orfalse statements."

## Full grader (representative response).

Denies the Defining a graph item, since "the reformulation is incorrect". Yet partial credit is still awarded on the Hall’s theorem item ("correctly invokes . . . Hall’s theorem . . . relies on the erroneous graphformulationfrom thefirstpart, sofull credit is not warranted") and on the Reducing the problem item ("essentially correct in idea and form but relies on the earlier mistaken definition").

What this case shows. The full grader explicitly states that the Hall’s-theorem step and the reduction step rely on a definition it has just declared wrong, yet still grants them partial credit. Under Blind setting, model correctly identifies that the proof structure is incorrect from the beginning and abstains from awarding more points to a fundamentally flawed solution.

![](images/b78b32f2c50ef482e2f9a7c23937b2af7a5a96173602bd4b2a2fba591420e00e.jpg)

![](images/72ece630f61493f1d501229ac28d83f1bd26db9e0038504750fc765da1cddf4b.jpg)

![](images/163065dc37dc2146395d29447ff05ba5d5e9670616b688024a44f18248fd85e2.jpg)

Global vs. local scoring. The Blind template asks the model to "read the proof; output one number 0–7", which forces an integrating operation. A fatal flaw in step 2 must reduce the credit step 5 contributes. The Full template asks "for each rubric item, decide if the student earned the points", which factors the task into independent per-line judgements. Small graders are good at local pattern matching, for instance "does the student use the words Hall’s theorem? yes → award the Hall’s theorem point" (Cases C and E). They are weak at compositional reasoning, namely "step 5 uses the bipartite graph from step 1; step 1 was wrong; therefore step 5 is also wrong". The Full template omits the compositional task by design, exposing the small grader’s weakness. Larger LLMs such as Claude Sonnet 4.6, GPT-5, and DeepSeek-v4-Flash propagate the error across rubric items since they consider local and global grading patterns together.

The rubric is a high-precision surface-form matcher. Every rubric line is, by construction, a description of the correct proof’s structure: "Defining a graph", "Hall’s theorem", "Reducing the problem", "Connectivity of construction", "Showing the graph is planar", and so on. A student who reaches for the right tools but applies them incorrectly produces text that contains exactly those keywords, so the surface-form match is near-perfect even when the substance is wrong. In Cases A, C and E the wrong proofs all look like the right proof. They reach for the right theorem (Hall, induction, planarity) and follow the right outline. Since the smaller models are less capable in uncovering the smaller failures in each claim, they inflate the final score. In Blind setting, there is no markscheme to base the reasoning on, so it forces to inspect proofs closer, which inadvertently improves the performance compared to the Full setting.

Reasoning revolves too much around the rubric and not the substance. When the prompt says "this rubric line is worth 3 points; explain whether the student earned them", the small instructiontuned model is biased toward awarding most of the 3. The default behaviour of small instruct-tuned models is to follow the prompt’s expectation. The prompt asks the model to "explain how many of the 3 points the student got", so the model produces a small downward adjustment from the maximum (gives 2 instead of 3) rather than zeroing out. The Blind prompt has no such anchor. The model has to commit to a number from {0, . . . , 7} after reading the proof, and the natural anchor becomes the proof’s actual quality. This is why the Blind grader can output a low value confidently when it sees a wrong proof, whereas the Full grader output for the same proof rarely falls into the lower grades territory, such as in Cases A, C, and E.

In summary, for a frontier grader, the rubric is additional context that informs but does not override compositional judgement. For a small grader, the rubric forces the model’s attention to it, and the LLM behaves more like pattern matching and less like a proof verifier. This effect is not present in larger models, since they verify the proof and match the reasoning together with a markscheme that provides additional context to how to properly distribute the grade. Strong vs. weak compositional ability is exactly what separates the two evaluation scenarios.

## J.2 Effect of MABGrader on grading procedures

We now study how MABGrader corrects the failure modes of vanilla chain-of-thought (CoT) grading on the same proofs. Both methods use the Gemini-3-Pro-Think model, the same problems and student solutions from Trojan-Bench(H), and the same Full-style prompt (problem, marking scheme, sample solution). The only difference is the inference procedure: vanilla CoT corresponds to a single grading call (with 16 attempts for the MajorityVote), whereas MABGrader runs the UCB procedure with c = 0.25, 8 arms (one per candidate grade in {0, . . . , 7}), and 16 oracle pulls. The comparison is over the 48 datapoints.

Aggregate metrics. MABGrader improves every aggregate metric. The biggest shift is in Bias: CoT is consistently optimistic (mean signed error +1.383 grade points), whereas MABGrader is closest to zero (mean signed error +0.340). The QWK lift of +0.171 is the largest improvement we observe from any prompt-engineering or aggregation (MajorityVote, Best-of-N) methods in this paper.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>EM</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=2>RMSE    Bias</td><td rowspan=1 colspan=1>Pearson r</td><td rowspan=1 colspan=1>QWK</td></tr><tr><td rowspan=3 colspan=2>Vanilla CoT (single attempt)         0.362Vanilla CoT (16-shot majority vote)   0.340MABGrader (UCB, c = 0.25)       0.447</td><td rowspan=1 colspan=1>1.660</td><td rowspan=1 colspan=1>2.502</td><td rowspan=1 colspan=1>+1.352</td><td rowspan=1 colspan=1>0.511</td><td rowspan=1 colspan=1>0.449</td></tr><tr><td rowspan=1 colspan=1>1.681</td><td rowspan=1 colspan=1>2.497</td><td rowspan=1 colspan=1>+1.383</td><td rowspan=1 colspan=1>0.515</td><td rowspan=1 colspan=1>0.454</td></tr><tr><td rowspan=1 colspan=1>1.277</td><td rowspan=1 colspan=1>1.957</td><td rowspan=1 colspan=1>+0.340</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.625</td></tr></table>

Two observations follow. First, the 16-shot MajorityVote variant is no better (and slightly worse on EM) than the single-attempt baseline. The 16 CoT samples agree with each other on 38 of 48 datapoints, so voting cannot denoise a structural bias. Second, the gain in instance-level MAE is asymmetric across the human-grade range. Vanilla CoT is monotonically lenient on the lower band of the scale (mean signed error +2.60 at human grade 0, +2.67 at human grade 2) and slightly lenient on the upper band, with perfect proofs graded correctly. MABGrader better matches grades on the lower end (mean signed error drops to +0.40 at human grade 0 and to +1.00 at human grade 2), while introducing a small under-grading of about −1 point at human grades 4, 5, and 6.

Two concrete cases. The two text boxes below illustrate the two failure modes vanilla CoT exhibits on this benchmark. Each box reports the predicted grade and the per-instance MAE for both methods, and excerpts representative free-form text from the model in each role. The grading CoT was compressed for better readability.

Aggregate grading behaviour on this single (problem, candidate-solution) pair.   
Vanilla CoT (16 attempts): mean grade 5.25, MajorityVote 6, instance MAE against the human reference 6.   
MABGrader (UCB, c = 0.25): selected grade 0, instance MAE against the human reference 0.   
Per-instance error reduction under MABGrader : −6 pts.

## Vanilla CoT (representative response).

The grade-0 arm correctly justifies the cascading-failure argument: "because G is defined between n people and m individual cupcakes . . . the application of Hall’s theorem here is meaningless. . . . Applying a valid theorem to an invalid, broken graph earns no points." The grade-6 arm has to invoke an unsupported "the student implicitly corrects their definitional error mid-proof " move, which the source text does not support; the oracle assigns full reward to the grade-0 arm and minimal reward to the high-grade arms.

What this case shows. The vanilla-CoT failure mode is atomistic rubric application: the model sums per-rubric credit independently and ignores the cascading consequence of the foundational error. MABGrader structurally counters this and correctly points out the accumulation of errors. The CoT samples cluster tightly, so any samplingbased ensemble (MajorityVote, Best-of-N) cannot escape the bias.

## Case H. Gemini-3-Pro-Think on Trojan-Bench(H) P5. Human grade = 5.

Aggregate grading behaviour on this single (problem, candidate-solution) pair.   
Vanilla CoT (16 attempts): mean grade 7.00 with zero variance across attempts, instance MAE against the human reference 2.   
MABGrader (UCB, c = 0.25): selected grade 5, instance MAE against the human reference 0.   
Per-instance error reduction under MABGrader : −2 pts.

## Vanilla CoT (representative response).

The CoT samples agree exactly on the maximum grade. One of them notices the off-by-one in the binomial reindexing step but labels it "a minor typographical error . . . a transcription slip rather than a conceptual misunderstanding". Once the issue is labelled a typo, the rubric cell auto-fills with full credit. The spread across attempts is zero, so no sampling-based ensemble can recover the human grade.

MABGrade (representative arm justifications). The grade-7 arm reads the off-by-one as "clearly a transcription slip". The grade-5 arm reads the same fact as "a critical algebraic error . . . the inductive hypothesis cannot be directly applied, leaving a significant logical gap in the proof". The oracle returns full reward for both readings (each is plausible in isolation); UCB tiebreaking returns the lower arm id, which corresponds to the stricter (and correct) human grade.

What this case shows. The vanilla-CoT failure mode here is over-charitable interpretation: a real algebraic error is reframed as a typo, lifting the grade to the maximum. Because the CoT samples agree for the most part, the failure is deterministic, not stochastic, and no sampling ensemble can correct it. MABGrader produces two equally plausible defences and lets the algorithm choose a better grading arm.

Hypothesis enumeration via arms. Vanilla CoT is a free generation: the model decides on the spot what the grade should be, writing a chain of thought that can drift toward a conclusion which is then rationalised against the rubric. The arm-generation phase of MABGrader does the opposite. It runs 8 separate calls, each of which assumes the answer is grade k and produces the strongest possible defence of grade k. Low-grade defences end up surprisingly compelling because they do not have to wave away foundational errors (Case G). The model is now arguing against its own default rather than rationalising it.

Oracle as adversarial verifier. The oracle is a separate model call that scores each arm’s defence on a Likert scale, mapped to a [0, 1] reward. The oracle does not have to commit to a number; it just has to ask whether each defence holds up against the source. That framing is much easier for the model than open-ended grading. In Case H the oracle returns full reward for both the lenient and the strict reading of the off-by-one, since both are individually plausible.

Table 35: SFT results with excluded USAMO 2025 datapoints
<table><tr><td rowspan="2"></td><td colspan="3">Full Setting</td><td colspan="3">Blind Setting</td></tr><tr><td>QWK(↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td><td>QWK (↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td></tr><tr><td>Qwen3.5-9B (Base)</td><td>0.02</td><td>4.00</td><td>3.23</td><td>0.06</td><td>2.52</td><td>0.63</td></tr><tr><td>Qwen3.5-9B (SFT)</td><td>0.23</td><td>2.19</td><td>0.17</td><td>0.08</td><td>2.32</td><td>-0.52</td></tr></table>

UCB tie-breaking is a strict-grading prior. When several arms tie at maximum reward, UCB returns the arm with the lowest id. Because arm ids equal candidate grades, "lowest id" means "lowest grade". When the evidence is ambiguous between "give credit" and "do not give credit", a professional grader defaults to the latter (the proof should remove ambiguity). In Case H this tie-breaker alone is what recovers the correct human grade.

Why self-consistency cannot close the gap. A natural counterpoint is that the gain might be variance reduction and that a simpler "sample CoT more times and apply MajorityVote" strategy would close it. The aggregate table refutes this: the 16-shot MajorityVote variant is slightly worse than a single sample, and the 16 samples agree with each other on the large majority of cells. Token-level sampling on a deterministic prompt with strong prompt-side bias does not produce diverse enough gradings to escape the bias. The diversity in MABGrader comes from the prompt side (eight different "assume answer = k" prompts), not from token-level sampling.

In summary, the dominant LLM grading failure on this benchmark is over-grading bias, not stochastic noise. Variance-reduction methods (sampling and voting) do not address bias because the bias is deterministic.

## K Inclusion of USAMO 2025 samples in the training set

To further validate the effect of TROJAN-generated samples on the quality of judging after supervised fine-tuning (SFT), we conducted an additional experiment where we fine-tuned a base model without USAMO 2025 samples and tested it exclusively on USAMO 2025. This experiment is designed to see whether the synthetic samples can generalize to previously unseen math problems. Table 35 reports results across thre metrics: QWK, RMSE and Bias, and under two settings: Full and Blind. We observe that all metrics improve after finetuning and highlights that the synthetic data may help with generalization on grading problems, even when the test problems come from a completely different competition.

## L Effect of changing the generator model in TROJAN pipeline

To further extend our analysis of TROJAN, we swapped the base Claude Opus 4.6 model for the smaller and less capable DeepSeek V4 Flash. The results are presented in Table 36. To exclude the effects of memorization and overfitting, we excluded the USAMO 2025 samples. Our tests suggest that using both models improves the performance of the underlying weaker 9B model. In Blind settings Flash model exhibits better performance, while Opus is generally better under Full setting evaluation. We emphasize that using both models lead to a bump in performance comapred to the base Qwen-3.5-9B model.

## M Benchmarking TROJAN SFT pipeline against ProofBench

Table 37 presents additional evaluation results on ProofBench [33], a dataset of LLM-generated proofs for competition-level mathematics paired with expert human grade annotations. For this experiment, we fine-tuned two models (Qwen3.5-9B and Qwen3.5-27B) on synthetic samples generated by our TROJAN pipeline, mirroring the setup described in the main text. We strictly excluded any overlap between the ProofBench dataset and our training set. This ensures that performance improvements stem from genuine learned grading capabilities rather than data contamination or overfitting. We report results for both the Full setting (which provides access to the markscheme and golden solution) and the Blind setting (which provides neither).

Table 36: Comparative performance of base and fine-tuned models on TROJAN synthetic samples coming from DeepSeek-V4-Flash (DSV4F) model and larger Claude Opus 4.6 (Opus-4.6). Best results are bolded.
<table><tr><td rowspan="2"></td><td colspan="3">Full Setting</td><td colspan="3">Blind Setting</td></tr><tr><td>QWK(↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td><td>QWK(↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td></tr><tr><td>Qwen3.5-9B (Base)</td><td>0.02</td><td>4.00</td><td>3.23</td><td>0.06</td><td>2.52</td><td>0.63</td></tr><tr><td>Qwen3.5-9B (SFT-Opus-4.6)</td><td>0.35</td><td>2.08</td><td>0.27</td><td>0.07</td><td>2.40</td><td>0.39</td></tr><tr><td>Qwen3.5-9B (SFT-DSV4F)</td><td>0.23</td><td>2.18</td><td>0.27</td><td>0.17</td><td>2.29</td><td>-0.23</td></tr><tr><td>Qwen3.5-27B (Base)</td><td>0.36</td><td>2.98</td><td>1.54</td><td>-0.11</td><td>4.12</td><td>-3.40</td></tr><tr><td>Qwen3.5-27B (SFT-Opus-4.6)</td><td>0.47</td><td>2.10</td><td>0.77</td><td>0.06</td><td>2.50</td><td>0.21</td></tr><tr><td>Qwen3.5-27B (SFT-DSV4F)</td><td>0.40</td><td>1.93</td><td>0.33</td><td>0.06</td><td>2.42</td><td>0.27</td></tr></table>

Table 37: Comparative performance of base and fine-tuned models on TROJAN synthetic samples on ProofBench benchmark. Best results are bolded.
<table><tr><td rowspan="2"></td><td colspan="3">Full Setting</td><td colspan="3">Blind Setting</td></tr><tr><td>QWK (↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td><td>QWK(↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td></tr><tr><td>Qwen3.5-9B (Base)</td><td>0.70</td><td>2.29</td><td>1.11</td><td>0.47</td><td>3.16</td><td>1.98</td></tr><tr><td>Qwen3.5-9B (SFT)</td><td>0.79</td><td>1.50</td><td>0.15</td><td>0.53</td><td>2.32</td><td>0.93</td></tr><tr><td>Qwen3.5-27B (Base)</td><td>0.80</td><td>1.81</td><td>0.41</td><td>0.49</td><td>3.08</td><td>1.04</td></tr><tr><td>Qwen3.5-27B (SFT)</td><td>0.81</td><td>1.44</td><td>-0.12</td><td>0.48</td><td>2.36</td><td>0.28</td></tr></table>

Table 38: Effect of the multi-armed bandit (UCB) algorithm on grading performance. Baselines include a single-model approach and a uniform sampling strategy, where each arm is pulled exactly r times before a final decision is made. Best results are highlighted in bold.
<table><tr><td></td><td>QWK (↑)</td><td>RMSE (↓)</td><td>Bias (≈ 0)</td></tr><tr><td>Qwen3.5-27B (model only)</td><td>0.30</td><td>3.04</td><td>1.81</td></tr><tr><td>Qwen3.5-27B (uniform with r = 1)</td><td>0.40</td><td>2.78</td><td>-1.48</td></tr><tr><td>Qwen3.5-27B (uniform with r = 5)</td><td>0.56</td><td>2.25</td><td>-1.26</td></tr><tr><td>Qwen3.5-27B (UCB)</td><td>0.57</td><td>2.15</td><td>-0.98</td></tr></table>

We observe that despite being fine-tuned exclusively on synthetic data, the models achieve correlation levels with ground-truth labels that match or exceed baselines, alongside a general reduction in overall error and bias. This consistent improvement demonstrates that our proposed markscheme-aligned synthetic error injection is both a robust evaluation proxy and an effective method for generating training data that aligns with expert human judgment. While the distributions of synthetic and naturally occurring errors inherently differ, our results reveal a functional overlap. Because the fine-tuned base model shows improved performance on an entirely independent problem set with a different group of expert annotations, it is clear that TROJAN’s synthetic errors generalize to realworld mistakes and unseen grading tasks. Furthermore, our generation pipeline provides fine-grained control over the exact type and location of injected mistakes, offering a targeted approach to further refine LLM grading capabilities.

## N Effect of Multi-Armed Bandit algorithm on grading

To isolate the performance gains of the Multi-Armed Bandit (MAB) algorithm from the underlying grade justification pipeline, we conducted an ablation study replacing MABGrader’s Upper Confidence Bound (UCB) selection policy with a uniform sampling baseline. In the "Uniform" setting, the model generates a justification for each candidate grade, an oracle assigns a confidence score to each, and the system selects the highest-scoring grade. Unlike MABGrader, which adaptively allocates its computational budget based on continuous oracle feedback, this baseline samples every arm a fixed number of times (e.g., r), entirely removing the UCB exploration-exploitation dynamics. This setting also corresponds to A/B testing. For comparison, we also included a standard baseline where the model directly outputs a grade using MajorityVote@16.

As presented in Table 38, the results indicate that the justification-and-scoring mechanism alone significantly improves baseline performance, and increasing the uniform sampling budget further amplifies these gains. However, the complete MABGrader framework remains superior. By dynamically balancing exploration and exploitation, the UCB algorithm converges on the optimal grade much more reliably under similar budget constraints.

## O Prompt templates

This section reproduces the six prompt templates referenced throughout the paper. The first one is the rubric grading prompt sourced from MathArena [4], which we use as our Full-template grading prompt in the human-set evaluation. We include it here for completeness so that our experimental setup can be reproduced from a single document. The next two are ProofAutoGrader and ProofGrader templates under Full settings.

The remaining three templates implement the three stages of the TROJAN synthetic-data pipeline (Section 4). The error-injection prompt drives the generator that produces a first-pass candidate student solution targeted at a specified score profile and includes a worked USAMO-style example so that the generator has a clear reference for what calibrated student errors look like. The critique prompt drives an independent grader that audits the candidate against the markscheme and outputs a JSON-formatted misalignment report with revision guidance for each rubric criterion. The refinement prompt prompts the LLM again, conditioned on both the candidate solution and the critique above, and produces a minimally-edited revision that is better aligned with the target score profile on each rubric criterion.

## MathArena rubric grading prompt template (Full setting).

## # Instruction

You are an expert mathematician that grades solutions of high-school olympiad-level problems. You will be given a mathematical problem, as well as a grading scheme that you should adhere to. Your task is to accurately grade a solution according to that grading scheme.

\# Problem and Scheme

\## Problem statement

{problem}

\## Grading scheme

{grading\_scheme}

\# Example

Here is how an example grading looks:

\## Example solution:

{sample\_solution}

\## Example grading:

{sample\_grading}

\# Evaluation

Produce a scoring using the same format for the following solution. Reason carefully about the contents of the solution and make sure that all steps of the proof are included and rigorous.

\## Formatting instructions

Follow a JSON-style formatting as the above example, namely:

\- The categories should be in the same order.

\- The awarded number of points for each category should follow after the ’points’ key.

\- The reasoning for each decision should be explained after

the ’desc’ key.

\- Output only the JSON response.

\- JSON response should be wrapped in ‘‘‘json ‘‘‘.

\- Do not output triple slashes or other characters that cause JSON decode errors.

## Here is the solution you should grade:

{solution}

## ProofAutoGrader rubric grading prompt template (Full setting).

You are an expert grader for the International Mathematics Olympiad (IMO). Your task is to evaluate a proposed solution strictly and rigorously. Keep in mind the standards at the IMO are extremely high: only arguments that are logically sound, complete, and precise should be rewarded.

## ### General Scoring Rubric

Scores are assigned on a 0-7 scale. The general guidelines are:

\*\*7 Points (Correct):\*\* The solution is complete, correct, and fully rigorous. If the submission contains incorrect attempts or lines of reasoning but ultimately presents a complete and correct solution, it should still be awarded full points; the presence of earlier, discarded work does not detract from the final correct proof.

\*\*6 Points (Almost Correct):\*\* The solution is almost correct with a sound core argument, but contains minor errors in calculation or small gaps in logic. Missing proofs for major components, unjustified claims, or sketchy arguments are \*\*not\*\* eligible for 6 points.

\*\*1 Point (Partial Progress):\*\* The solution demonstrates substantial progress explicitly mentioned in the grading guidelines. Initial observations, reformulating the problem without making substantive headway, or proving partial results not mentioned in the grading guidelines are generally \*\*not\*\* eligible for this score.

\*\*0 Points (Incorrect):\*\* The solution doesn’t make substantial progress that is a key step in the full solution or is fundamentally flawed. All partial progress without key results or lacking rigor also fall in this category.

\### Input Data and Interpretation

You are provided with the following:

\*\*Problem Statement:\*\* The IMO problem.

\*\*Ground Truth Solution:\*\* A reference solution. Assume this solution is correct.   
It demonstrates one valid approach.

\*\*Specific Grading Guidelines:\*\* Criteria for awarding credit for this specific problem. These guidelines take precedence over the General Scoring Rubric, especially for partial credit.

\*\*Proposed Solution:\*\* The student submission.

\### Evaluation Process

You must follow this structured process:

You must provide your final score in the format ‘<points>N</points>‘. Ensure   
the ‘<points>‘ block is used \*\*only once\*\*, as your answer will be parsed   
based on the first ‘<points> </points>‘ block that appears in your whole response.

\*\*Analyze References:\*\* Meticulously read and understand the problem and Ground Truth Solution check the Specific Grading Guidelines. Identify the key steps for a complete solution and the criteria for partial credit.

\*\*Step-by-Step Verification:\*\* Verify the logical validity and rigor of every step. Identify all flaws, gaps, assumptions, and errors. \*\*Make sure you fully understand every piece of logic behind each step of the proposed solution, you must be careful for solutions that ’pretend’ to be correct.\*\*

\*\*Assess Progress:\*\* Determine the extent of non-trivial progress made.

\*\*Score Determination:\*\* Compare the findings against the Specific Grading Guidelines and the General Rubric to determine the final score.

## ### Output Requirements

\*\*PROBLEM STATEMENT\*\*

{problem}

\*\*GROUND-TRUTH SOLUTION\*\*

{sample\_solution}

\*\*SPECIFIC GRADING GUIDELINES\*\*

{grading\_scheme}

\*\*PROPOSED SOLUTION\*\*

{solution}

Present your detailed thought process and formal justification based on the scoring rubric and grading guidelines, and finally present your final score in the format below.

\- ‘<points>N</points>‘

## ProofGrader rubric grading prompt template (Full setting).

You are an \*\*expert math proof grader\*\*. You are judging the correctness of an LLM-generated proof for a math problem.

\### Input

Your input will consist of: \* \*\*Problem Statement\*\*: A mathematical problem that the proof is attempting to solve.

\* \*\*Reference Solution\*\*: A correct solution or proof provided for reference. This is \*\*not necessarily the only valid solution\*\*. If the problem requires a final numeric or algebraic answer, this section contains the correct answer, which should be the only accepted final answer (though alternative reasoning paths are valid).

\*\*Requirements:\*\*   
- ‘<score>‘ must be an integer in [0, 7].   
- ‘<assessment>‘ must be a \*\*detailed analysis\*\* that explains   
your reasoning step-by-step and provides a clear \*\*rationale for   
the score\*\*. Reference specific claims/lines if present.   
- ‘<errors>‘ must be a list of specific issues (empty if score = 7).   
Example output:   
<score>0</score>   
<assessment>The proof shows a good understanding of the main idea   
but has some unclear reasoning and minor mistakes...</assessment>   
<errors>   
1. specific error 1,   
2. specific error 2,   
</errors>   
\*\*Problem Statement\*\*   
{problem}   
\*\*Reference Solution\*\*   
{sample\_solution}   
\*\*Marking Scheme\*\*   
{grading\_scheme}   
\*\*Proof Solution\*\*   
{solution}

\* \*\*Marking Scheme\*\*: A problem-specific grading rubric (0-7 scale) with checkpoints, zero-credit items, and deductions. You must   
follow this scheme when assigning points. \* \*\*Proof Solution\*\*: The proof that you need to evaluate. This proof may contain errors, omissions, or unclear steps. The proof   
was generated by another language model.

## ### Task

Analyze the proof carefully.

\* Compare the proof against the reference solution and the marking scheme.

\* Award points according to the marking scheme’s checkpoints, zero-credit items, and deductions.

\* Identify logical errors, incorrect steps, or unclear reasoning.

\* Give a score between 0 and 7 with a brief overall assessment.

\* Show clearly how the score was derived:

\* Which checkpoints were earned (with awarded points).

\* Any zero-credit items or deductions applied.

\* How the subtotal leads to the final score (0-7).

## ### Output Format

Respond with \*\*only\*\* well-formed XML using the structure below. Do not include any extra text or Markdown.

## TROJAN error-injection prompt template (one-shot).

You will be provided with:

1. The Problem

2. The Markscheme (Grading Scheme)

3. A Correct Reference Solution

4. A Score Profile specifying exactly how many points the student earns on each rubric criterion

Your objective is to write a complete candidate proof that earns \*\*EXACTLY\*\* the points specified in the score profile on each criterion.

## Rules:

1. \*\*Full-marks criteria\*\*: The relevant proof sections must be mathematically sound and rigorous. You may freely copy or closely paraphrase corresponding parts of the reference solution -- students who understand a technique often reproduce it in very similar language.

2. \*\*Deducted criteria -- error calibration\*\*: For criteria where points are LOST, introduce a \*believable\* student error proportional to the size of the deduction:

\- \*\*Minor deduction\*\* (e.g., losing 1 of 3 points): a small gap -- an off-by-one index, an edge case left unchecked, a minor algebraic slip, a missing quantifier, or an unjustified but plausible assertion.

\- \*\*Moderate deduction\*\* (e.g., losing 2 of 3 points): a significant logical gap -- wrong case analysis, an unverified hypothesis used as if proven, a crucial intermediate step hand-waved or skipped.

\- \*\*Full deduction\*\* (losing all points on a criterion): a fundamental omission or conceptual error -- the entire argument for that part is missing, a theorem is invoked with unmet hypotheses, or the approach simply does not address the criterion.

3. \*\*Error believability\*\*: Every injected error must be one a competent but imperfect student would plausibly make. Ask yourself: "What would a student who understood the main ideas but rushed or overlooked something actually write?" Avoid random deletions or nonsensical mistakes.

4. \*\*Coherence\*\*: The proof must read as a single, unified student attempt -- not a patchwork of correct and incorrect pieces stitched together. Correct and incorrect sections must flow naturally into each other with consistent transitions, notation, and writing style throughout.

5. \*\*Copying policy\*\*: You may copy or closely paraphrase the reference solution, especially for full-marks sections. For deducted sections, base your writing on the reference’s structure but introduce the targeted errors organically. Do NOT egregiously copy-paste the entire reference and merely delete lines to simulate errors.

## ### Worked Example

Below is a complete example demonstrating the expected input-to-output mapping.

<example\_problem>

Let \$k\$ and \$d\$ be positive integers. Prove that there exists a positive integer \$N\$ such that for every odd integer \$n>N\$, the digits in the base-\$2n\$ representation of \$n^k\$ are all greater than \$d\$.

</example\_problem>

<example\_markscheme>   
[   
{   
"desc": "Uses the base-2n representation of \$n^k\$ to find   
a closed form expression of each digit   
\$a\_i=\left\lfloor \frac{n^k}{(2n)^i}\right\rfloor   
mod\ (2n)\$",   
"points": 1,   
"title": "Initial closed form expression for each digit"   
},   
{

```perl
"desc": "Transforms the digit expression into a easily
boundable one by interchanging the floor and
modulo operations.",
"points": 3,
"title": "Interchanging the floor and mod"
},
{
"desc": "Correctly bounds the aforementioned expression,
usually by showing the numerator is divisible by
n, so that
$a_i>=\left\lfloor \frac{n}{2^i}\right\rfloor$.",
"points": 2,
"title": "Bounding the expression"
},
{
"desc": "Chooses a provably sufficiently large $N$. The
point is not awarded if it has not proven that
n^k has at most k digits.",
"points": 1,
"title": "Finishing"
}
]
</example_markscheme>
```

## <example\_correct\_solution>

We claim one such \$N\$ is \$\boxed{N=2^{k-1}(d+1)}\$. Begin by letting \$a\_i\$ be the digit at the \$i\$th position from the right (0-indexed) of \$n^k\$ in base \$2n\$; i.e., \$a\_0\$ is the units digit. Obviously, \$a\_i=0\$ for all \$i\ge k\$, since \$n^k<(2n)^k\$. We WTS \$a\_i>d\$ for each \$i=0,1,\dots,k-1\$, and observe the following: \[ a\_i=\left\lfloor\frac{n^k}{(2n)^i}\right\rfloor\text{ mod }2n

=\left\lfloor\frac{n^{k-i}}{2^i}\right\rfloor\text{ mod }2n =\left\lfloor\frac{n^{k-i}\text{ mod }2^{i+1}n}{2^i}\right \rfloor. \]

Note that the numerator is a positive multiple of \$n\$, since 1) \$k-i>0\implies n\mid\gcd(n^{k-i},2^{i+1}n)\$ and 2) (\$n\$ is odd) \$2^{i+1}n\nmid n^{k-i}\$. Thus,

\[ a\_i=\left\lfloor\frac{n^{k-i}\text{ mod }2^{i+1}n}{2^i}\right \rfloor>d \quad\iff\quad

n^{k-i}\text{ mod }2^{i+1}n\ge n\ge 2^i(d+1), \]

which is true since \$n>N=2^{k-1}(d+1)\$ and \$k-1\ge i\$. \$\square\$ </example\_correct\_solution>

## <example\_score\_profile>

Uses the base-2n representation of \$n^k\$ to find a closed form expression of each digit

\$a\_i=\left\lfloor \frac{n^k}{(2n)^i}\right\rfloor mod\ (2n)\$

Transforms the digit expression into a easily boundable one by interchanging the floor and modulo operations.

3. "Bounding the expression" (2 pts) -> Awarded 2/2 [FULL MARKS] Correctly bounds the aforementioned expression, usually by showing the numerator is divisible by n, so that \$a\_i>=\left\lfloor \frac{n}{2^i}\right\rfloor\$.

4. "Finishing" (1 pts) -> Awarded 0/1 [LOSE 1/1 points] Chooses a provably sufficiently large \$N\$. The point is not awarded if it has not proven that n^k has at most k digits.

## Total score: 6/7

## </example\_score\_profile>

\*\*Why this output works\*\*: Criteria 1-3 earn full marks, so the student closely follows the reference. Criterion 4 loses its point because the student omits the digit-count bound "\$a\_i=0\$ for all \$i\ge k\$ since \$n^k<(2n)^k\$" -- a plausible oversight by a student who focused on the algebraic core and forgot to justify why only \$k\$ digits matter:

## # Candidate Solution

We claim one such \$N\$ is \$\boxed{N=2^{k-1}(d+1)}\$. Begin by letting \$a\_i\$ be the digit at the \$i\$th position from the right (0-indexed) of \$n^k\$ in base \$2n\$; i.e., \$a\_0\$ is the units digit. We WTS \$a\_i>d\$ for each \$i=0,1,\dots,k-1\$, and observe the following:

\$\$ a\_i=\left\lfloor\frac{n^k}{(2n)^i}\right\rfloor\text{ mod }2n =\left\lfloor\frac{n^{k-i}}{2^i}\right\rfloor\text{ mod }2n =\left\lfloor\frac{n^{k-i}\text{ mod }2^{i+1}n}{2^i}\right \rfloor. \$\$

Note that the numerator is a positive multiple of \$n\$, since 1)

\$k-i>0\implies n\mid\gcd(n^{k-i},2^{i+1}n)\$ and 2) (\$n\$ is odd) \$2^{i+1}n\nmid n^{k-i}\$. Thus,

\$\$ a\_i=\left\lfloor\frac{n^{k-i}\text{ mod }2^{i+1}n}{2^i}\right \rfloor>d \quad\iff\quad

n^{k-i}\text{ mod }2^{i+1}n\ge n\ge 2^i(d+1), \$\$

which is true since \$n>N=2^{k-1}(d+1)\$ and \$k-1\ge i\$.

\hfill \$\square\$

## # Grader’s Notes

\- \*\*Finishing (0/1)\*\*: The student never shows that \$n^k\$ has at most \$k\$ digits in base \$2n\$ (i.e., omits "\$a\_i=0\$ for all \$i\ge k\$ since \$n^k<(2n)^k\$"). Without this, the claim that it suffices to check only \$i=0,\dots,k-1\$ is unjustified, so \$N\$ is not proven sufficiently large.

Now generate a candidate student solution for the following:

<problem> {problem} </problem>

<markscheme> {grading\_scheme} </markscheme>

<correct\_solution> {correct\_solution} </correct\_solution>

<score\_profile>{score\_profile}</score\_profile>

Write the candidate proof below. Use standard LaTeX formatting for all mathematics. After the proof, provide a brief "Grader’s Notes" section: for each deducted criterion, state the criterion name, the score awarded, the specific error introduced, and why it warrants that deduction.

## # Candidate Solution

[The student’s complete proof]

## # Grader’s Notes

[For each deducted criterion: criterion name (awarded/max), the specific error, and why it matches the deduction]

## TROJAN critique prompt template.

You are a rigorous mathematics examiner auditing a synthetic student solution. A generator produced a candidate solution designed to earn a specific TARGET score profile. Your task is to independently grade the candidate and verify that a real grader, using the markscheme below, would award exactly the target scores.

You have access to the correct reference solution as the ground truth for mathematical correctness.

<problem> {problem} </problem>

<markscheme> {grading\_scheme}

## </markscheme>

<correct\_solution> {correct\_solution} </correct\_solution>

<target\_score\_profile> {score\_profile} </target\_score\_profile>

<candidate\_solution> {candidate\_solution} </candidate\_solution>

## Instructions:

1. \*\*Grade each criterion independently.\*\* For each rubric item, read the candidate solution and determine the ACTUAL points it would earn based solely on the markscheme and mathematical content. Use the reference solution as the correctness benchmark.

2. \*\*Compare actual vs. target.\*\* Set "aligned" to true only if actual\_score == target\_score for that criterion.

3. \*\*For misaligned criteria, provide specific and actionable revision guidance:\*\*

\- If the candidate earns \*\*MORE\*\* than the target (error too mild or absent): state what is currently correct that should be broken, and suggest a specific, \*believable\* student error -- something a competent student could plausibly make. Do not suggest random deletions.

\- If the candidate earns \*\*LESS\*\* than the target (error too severe): state what is currently wrong and how to partially or fully fix it so the student earns the target partial credit.

4. \*\*Error calibration reference\*\* (use this to judge whether error severity matches the deduction):

\- Minor deduction (e.g., 1 of 3 points lost): small gap -- off-by-one, unchecked edge case, missing quantifier, unjustified assertion.

\- Moderate deduction (e.g., 2 of 3 points lost): significant logical gap -- skipped step, unverified hypothesis, wrong case analysis.

\- Full deduction (all points on a criterion lost): fundamental omission -- entire argument missing, theorem invoked with unmet hypotheses, approach doesn’t address the criterion.

5. \*\*Coherence check.\*\* Also assess whether the proof reads as a natural, unified student attempt. If it feels like a patchwork of correct and incorrect pieces, or if errors seem artificially inserted rather than organically arising, flag this in the revision guidance.

6. \*\*Copying policy.\*\* Close paraphrasing of the reference solution is acceptable for full-marks sections -- do NOT flag similarity to the reference as a problem. Students who master a technique often reproduce it in very similar language.

Output ONLY a JSON object (no markdown fences, no surrounding text) in this exact format:

{   
"criteria": [   
{   
"criterion\_index": 1,   
"title": "criterion title",   
"target\_score": 0,   
"actual\_score": 0,

"aligned": true,   
"feedback": "What the candidate does for this criterion,   
what the target requires, and (if misaligned)   
a specific revision to make."   
}   
],   
"overall\_aligned": true,   
"revision\_guidance": "High-level summary of all revisions needed,   
including any coherence issues. Empty   
string if fully aligned."

## TROJAN refinement prompt template.

You are an expert mathematics professor revising a synthetic   
student solution. A critique found that some rubric criteria do   
NOT earn the intended number of points. Your task is to revise   
the solution so each criterion earns EXACTLY the target score.   
<problem>   
{problem}   
</problem>   
<markscheme>   
{grading\_scheme}   
</markscheme>   
<correct\_solution>   
{correct\_solution}   
</correct\_solution>   
<target\_score\_profile>   
{score\_profile}   
</target\_score\_profile>   
<current\_candidate>   
{candidate\_solution}   
</current\_candidate>   
<critique>   
{critique}   
</critique>   
Revision rules:

1. \*\*Preserve aligned criteria.\*\* For criteria already matching the target score, keep those proof sections as close to unchanged as possible. Do not inadvertently break what already works.

2. \*\*Address each critique point specifically.\*\* For each misaligned criterion, follow the critique’s feedback:

\- If the candidate earns \*\*MORE\*\* than the target: introduce a larger or more fundamental error to bring the score down. The error must be \*believable\* -- something a competent but imperfect student would plausibly write.

\- If the candidate earns \*\*LESS\*\* than the target: soften or partially fix the error so the student earns the intended partial credit. Use the reference solution to restore the relevant argument.

3. \*\*Error calibration:\*\*

\- Minor deduction (e.g., 1 of 3 points lost): small gap -- off-by-one, unchecked edge case, missing quantifier, unjustified assertion.

\- Moderate deduction (e.g., 2 of 3 points lost): significant logical gap -- skipped step, unverified hypothesis, wrong case analysis.

\- Full deduction (all points on a criterion lost): fundamental omission -- entire argument missing, theorem misapplied, approach doesn’t address the criterion.

4. \*\*Coherence.\*\* The revised proof must read as a single, unified student attempt. After changing one section, re-read the surrounding paragraphs to ensure transitions, notation, and writing style remain consistent throughout.

5. \*\*Copying policy.\*\* You may freely copy or closely paraphrase the reference solution for full-marks sections -- students who master a technique often reproduce it in similar language. For deducted sections, base your revision on the reference’s structure but weave in errors organically. Do NOT simply delete lines from the reference to simulate mistakes.

6. \*\*Minimal edits.\*\* Make the smallest changes that fix the misalignments. Rewriting large correct sections increases the risk of introducing unintended new errors or breaking aligned criteria.

Output format:

\# Candidate Solution

[The revised proof]

\# Grader’s Notes

[For each deducted criterion: criterion name (awarded/max), the specific error, and why it matches the deduction]

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: Each claim is supported by empirical evaluation across a broad range of language models.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide limitations and future work analysis of our work in the Appendix. Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not include theoretical results.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Yes, we discuss all parameters for verification, grading, dataset construction, and inference, and provide training model cards for each SFT experiment.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: The codebase is attached with the submission.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Yes, we provide both Datasheet and model cards for evaluation, data collection and training.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report mean and variance of individual grading with 16 independent runs.   
Error bars are reported for each conducted experiment in the Appendix of the work.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We disclose every model accessed via API.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: All sourced data is in accordance with ethical standards.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: Our work discusses fairness in mathematical verification. We do not anticipate negative societal impacts.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: Every benchmark comes with a root source of the problem to direct the readers to original releases. Other than that we do not anticipate misuse risks.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite and explicitly mention every sourced data and templates in the main text and Appendix.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: Yes, every datapoint comes with a marking scheme, candidate solutions and corresponding justifications.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The work does not use crowd sourcing.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The work does not use crowd sourcing.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [Yes]

Justification: We explicitly describe the LLM uses in grading mathemtical problems. Every experiment comes with a detailed explanation of base models, templates and other relevant parameters.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.