# CRJUDGEBENCH: CAN AI DETECT PLAUSIBLE BUT INVALID CODE REVIEWS?

Yue Pan<sup>1</sup> Jiawei Li<sup>2</sup> Ziyuan Zhang<sup>3</sup> Xiangxin Zhao<sup>1</sup> He Ye<sup>1†</sup>

<sup>1</sup>University College London <sup>2</sup>Amazon <sup>3</sup>Zhejiang University

## ABSTRACT

Large language models can generate plausible code-review comments, but such comments may contain technically incorrect claims that mislead developers. We study technical trustworthiness judgment: determining whether a review comment’s core technical claims are correct and applicable to the reviewed code in its repository context. Existing code-review benchmarks primarily evaluate review generation, issue discovery, or general comment quality, but do not directly assess whether an agent can determine the technical trustworthy of an individual review comment. To fill this gap, we introduce CRJudgeBench, a benchmark of 1199 instances constructed from real pull requests and expert-verified perturbations, covering both trustworthy and plausible but untrustworthy comments. We further present Sentinel, a repository-grounded agentic judge that actively gathers code evidence to verify review comments before making judgments. Starting from Qwen3-Coder-30B-A3B-Instruct, Sentinel is trained on the CRJudgeBench training split through iterative action-level learning from a privileged teacher. On the 359-instance CRJudgeBench test set, Sentinel achieves 76.60% accuracy, outperforming GLM-5.3 by 6.13 percentage points and its base model by 19.78 points. These results show that even state-of-the-art general-purpose LLMs struggle to identify untrustworthy comments, while iterative action-level learning substantially improves the accuracy of repository-grounded trustworthiness judgments. Our dataset is available at https: //huggingface.co/datasets/dcloud347/CRJudgeBenchmark.

## 1 INTRODUCTION

Coding agents such as Claude Code and Codex (Anthropic, 2025a; OpenAI, 2025) are rapidly changing how software is produced and have been adopted by more and more software practioners (Robbes et al., 2026). Their usage has transitioned from short code completions toward repo-level end-to-end agentic development workflows (Anthropic, 2025b; Stack Overflow, 2025; Cursor, 2026). While coding agents increase the efficiency of various software development and maintenance activities through automated code reasoning and generation, the generated code still needs to be reviewed and verified to ensure it meets quality criteria (Cotroneo et al., 2025). Moreover, the large volume of agent-generated code can overwhelm human reviewers that leads to degraded review reliability (Gonc¸alves et al., 2022; Watanabe et al., 2025; DORA, 2025). Scaling AI-assisted development therefore requires reliable, scalable review mechanisms alongside stronger coding agents.

This need has motivated increasing interest in automated code review, including Large Lanuage Model (LLM)-based agents that inspect code patches (i.e., in pull requests) and automatically generate review comments. Recent benchmarks such as c-CRAB and CR-Bench evaluated the ability of such agents to accurately detect quality issues in code patches and generate review comments (Zhang et al., 2026; Pereira et al., 2026). However, existing review agents collectively identify only around 40% of issues identified by humans and provide numerous false alarms. A recent study of 31,073 CodeRabbit review–feedback pairs across 239 GitHub repositories found that developers rejected 56.3% of agent-generated reviews, mainly because of false alarms, redundant or out-of-scope suggestions, and misalignment with developer intent (Lin et al., 2026), showing that a plausible review comment may still be a false alarm and may not be technically correct given the software context. Thus, whether agent-generated review comments can be trusted becomes crucial.

![](images/247d78b9c1189c001e2c86766d40334e709fc032a96d82dfc686d0a4f629a483.jpg)  
Figure 1: Given a pull request and a generated review comment, a judge agent inspects the relevant software context before deciding whether the comment is technically trustworthy.

Despite the importance of review comment trustworthiness, existing evaluation frameworks have not directly tackled this aspect. Earlier approaches evaluate general properties of review comments. EvaCRC models attributes such as evaluation, suggestion, question, and emotion, but takes only the comment text as input without the code change or repository context (Yang et al., 2023), which can introduce noise and bias. CRScore provides reference-free measurements of conciseness, comprehensiveness, and relevance by grounding evaluation in code claims and potential issues (Naik et al., 2025). However, they use semantic matching for evaluation, which does not directly verify whether a comment’s technical claim is correct and trustworthy. These evaluations are valuable for measuring surface-level review quality and agent capability, but they leave an important aspect underexplored: given a specific review comment, is its technical judgment actually correct for the code patch and grounded in the repository context to which it refers? A comment can be fluent, relevant, detailed, and actionable while still making an incorrect claim about program behavior. Conversely, whether an apparently reasonable claim is correct may depend on context such as code and documents outside the code patches, such as callers, guards, definitions, or tests elsewhere in the repository. Existing methods lack the ability to use various repository-wide software context when evaluating the trustworthiness of a review comment.

We formulate this previously overlooked evaluation aspect as technical trustworthiness. A review comment is technically trustworthy when its technical judgments are grounded in the code patches and relevant repository context, and contain no factual errors that could mislead a developer’s understanding or subsequent code refinement actions. Unlike CRScore, which measures dimensions such as conciseness and comprehensiveness, our evaluation does not focus on whether an automated approach finds or covers sufficient quality issues. Instead, we focus on its ability to determine whether a given review comment is technically correct in its specific code and repository context. This distinction is particularly important for agent-generated reviews since manually verifying every generated comment is labor-intensive. Given the growing volume of agent-generated reviews and code, we aim to propose a scalable automated approach to evaluate review comments’ technical trustworthiness in this study.

As the first step, we introduce CRJudgeBench, a benchmark specifically designed for technical trustworthiness judgment. CRJudgeBench contains 1,199 review comment instances from Github, along with their code patches under review and associated software context. It includes 764 trustworthy and 435 untrustworthy comments. The latter combine incorrect reviews by human reviewers with expert-verified LLM-injected perturbations that render the comments factually incorrect and not grounded on the code patches/software context.

Then, we propose Sentinel, a repository-grounded LLM-based agentic judge for assessing technical trustworthiness of review comments. Figure 1 illustrates how Sentinel works in the code review workflow. Different from existing techniques in literature, Sentinel does not classify a review comment only based on its textual content. Instead, it follows a ReAct-style (Yao et al., 2022) agentic setting to interact with the repository such as searching for relevant code element definitions and usages, inspecting code patches and related code context, and gathers evidence needed to verify the comment’s technical claims before reaching a conclusion. We trained the model’s policy using Learning from Experts with Access to Privilege (LEAP) (Choudhury & Sodhi, 2025) to ensure reliability. In this framework, the student is the agent being trained. The teacher is typically an agent powered by more capable model that has access to privileged information and provides supervisory signals to guide the student. Overall, the student agent first interacts with repositories autonomously;

![](images/c1527e7a094c639deba2a9a272c81ccbdff9c52147184ad260e5c319a31a83ce.jpg)  
Figure 2: Overview of the CRJudgeBench construction pipeline.

a teacher agent with access to the human-annotated ground truth then provides next-action supervision at intermediate steps in trajectories. Sentinel is trained through action-level distillation on these guided evidence-gathering actions, while the ground truth trustworthiness labels, which are privileged information available only to the teacher, are excluded from the student’s inputs. This design teaches the agent how to explore and gather evidence, rather than merely learning a direct mapping from review comment to labels.

We evaluate Sentinel on the 359-instance held-out CRJudgeBench test split against strong frontier and open-source LLMs under the same agentic setup. Sentinel achieves 76.60% accuracy, compared with 70.47% for GLM-5.3, the strongest evaluated baseline. Both models correctly identify 98.69% of trustworthy comments. However, the main difference is their ability to detect untrustworthy comments: GLM-5.3 detects 20.77% and misses 79.23%, while Sentinel detects 37.69% and misses 62.31%. These results show that, while detecting false alarm review comments remains challenging, repositorygrounded evidence seeking and specialized training substantially improve agent’s capability.

Our work makes the following contributions: Problem and benchmark. We are the first to formulate technical trustworthiness of code review comments and introduce CRJudgeBench, a 1,199- instance,repository-grounded benchmark containing real and expert-verified trustworthy and untrustworthy review comments. Agentic judge. We introduce Sentinel, a repository-grounded agentic judge that actively gathers evidence to verify review comments, and its evidence-seeking behavior is trained through iterative action-level learning from privileged teacher supervision. Empirical findings. We provide a systematic evaluation showing that even strong frontier models struggle to identify plausible but technically incorrect review comments, while Sentinel substantially improves both overall accuracy and recall of untrustworthy comments.

## 2 CRJUDGEBENCH

We construct CRJudgeBench to evaluate models’ ability to judge the technical trustworthiness of a code review comment using the pull request (PR), the submitted code patch under review, the repository, and related software context. This section describes the construction process in detail.

Repository and Pull Request Selection. To construct CRJudgeBench, we target open-source projects mainly written in Python on Github, since Python has become one of the most widely used programming languages in software development (GitHub, 2024). We select projects through a series of automated filtering steps and manual examination. First, we collect top 1,000 Python repositories ranked by GitHub star count to ensure we target widely used and popular projects. Second, we aim to retain projects with active maintenance activities. To do so, we only keep the ones with more than 1,500 PRs in total and having at least one commit or PR recorded between August 15, 2024 and August 15, 2025. Then, we proceed to gather PRs of these projects.

![](images/84313769d65dcc62e5209ab96711a04f48295cc4684f7f099f40176cb227c07a.jpg)

Table 1: Average and maximum values characterizing different attributes of a CRJudgeBench task instance, including the PR text, patch under review, review comment, and PR context.
<table><tr><td></td><td></td><td>Mean</td><td>Max</td></tr><tr><td>PR Text</td><td>Title length (words) Body length (words)</td><td>6.91 173.77</td><td>16 573</td></tr><tr><td>Patch to Review</td><td># Lines edited # Files edited</td><td>308.71 9.70</td><td>2,070 48</td></tr><tr><td>Review Comment</td><td>Length (words)</td><td>31.73</td><td>608</td></tr><tr><td>PR Context</td><td># Commits per instance</td><td>13.29</td><td>201</td></tr></table>

Figure 3: Distribution of instances across repositories.

We next use an LLM to assist in auditing these 106,241 PRs for dataset inclusion. The screening criteria exclude PRs with descriptions that lack substantive technical content or cannot be meaningfully interpreted, such as simple expressions of agreement, formatting-only suggestions, and references whose meaning cannot be resolved using the available context, such as the code patch, PR description/discussion, and linked issues. We also exclude PRs whose problem descriptions contain fewer than 40 words or whose interpretation depends on external information absent from the available context. These can be undocumented project conventions, off-platform discussions (i.e., personal communications such as emails etc.), or unavailable issue. After the LLM audit, we manually examined 383 randomly sampled PRs (confidence level of 95% and margin of error of 5%) where two Python experts with more than five years of software code review experience independently examine its assessments and supporting rationales to verify that the screening criteria have been applied appropriately. Then, they discuss to resolve any disagreements. For those they could not come to an agreement, a third Python expert with more software code review experience is consulted to make the final inclusion decisions. This LLM-assisted screening and joint verification by the three Python experts yields 7,086 PRs.

CRJudgeBench Construction. An instance in our dataset centers on a target code review comment in a PR. Its associated software context includes the PR description, linked issue reports, the code patch under review, the complete review timeline, and the repository code at the corresponding version. Each instance carries a human-annotated ground truth Boolean trustworthy label. For each of the 7,086 selected PRs that can contain multiple review comments, we create one data instance for every review comment written by a contributor other than the PR author; comments written by the PR author are excluded. This initial collection yields 76,190 instances. We first apply rule-based filtering where we exclude instances whose target comments are authored by bot accounts, identified using predefined username patterns (Dey et al., 2020), or whose code patches exceed 200,000 characters. We also remove instances whose target comments contain fewer than 10 characters after trimming whitespace, match a predefined list of brief acknowledgments, or begin with “LGTM”<sup>1</sup> and contain fewer than 30 characters. This process resulted in 67,458 instances. We then prompt an LLM to remove purely social or complimentary comments, vague comments requiring unavailable external context, duplicated comments, formatting- or whitespace-only remarks. This step led to 48,027 instances. Next, we conducted manual examination to assess its technical trustworthiness and suitability for inclusion in the benchmark, excluding comments deemed unsuitable, that is, they cannot be reliably assessed using the available software context. The first three authors discussed and resolved the disagreements throughout this process. However, joint manual annotation of the 48,027 remaining instances would incur a prohibitive cost. To keep the annotation effort manageable, we therefore restrict manual annotation to instances from the ten repositories with the highest GitHub star counts, yielding 1,031 instances for manual joint review. Finally, we have 764 technically correct comments (true) and 65 technically incorrect comments (false).

To mitigate the class imbalance, we construct 370 perturbed negative instances from the positive instances jointly confirmed by the three Python experts. Each negative instance is created by copying a positive instance and modifying the review comment, either changing its text or its location. The PR description, full patch, and all other related software context remain unchanged. Specifically, we apply three types of perturbations: (i) negation (130 instances), which reverses a single affirmative or negative expression in the comment text, such as changing “These imports are not used” to “These imports are used”; (ii) symbol (88 instances), which changes a referenced code element while preserving the surrounding wording, for example, replacing function test nanmedian with test nanmedian impl; and (iii) location (152 instances), which preserves the comment text but replaces both its commented file path and diff hunk with those associated with another review comment in the same PR. To ensure a perturbed instance qualify as a negative instances, the three Python experts jointly reviewed the modified comment and verified that it is technically incorrect in its associated code context. The resulting benchmark contains 1,199 instances: 764 trustworthy (true) and 435 untrustworthy (false) ones. Figure 3 shows the distribution of benchmark instances across repositories, while Table 1 summarizes the main attributes of a CRJudgeBench task instance. For the convenience of model training and evaluation, we partition the benchmark into PR-disjoint training, validation, and test sets (i.e., All instances from the same PR are assigned to a single split). To keep the proportion of trustworthy and untrustworthy instances similar across the splits, we use a two-dimensional subset-sum procedure to group PRs so that each split contains approximately the same number of instances from each class. We first create an initial training pool of 840 instances and a test set of 359 instances, following an approximately 7:3 ratio. We then allocate 15% of the training pool (126 instances) to validation split used for hyperparameter tuning. This leaves 714 training instances, while the test set remains unchanged.

## 3 SENTINEL

Overview. Using CRJudgeBench, we trained Sentinel, a LLM-based agent that grounds on repository evidence to verify a review comment’s technical claims and referenced code locations before predicting its trustworthiness. Due to the challenging nature of this task and the limited performance of state-of-the-art LLM-based coding agents (we will illustrated in Section 4), we train Sentinel iteratively with a privileged expert following Learning from Experts with Access to Privilege (LEAP) (Choudhury & Sodhi, 2025). At each iteration, the student is rolled out in a sanbox environment to interact with the repo and assess review comments, while a teacher uses the gold trustworthiness label—available only during training—to propose improved reasoning and actions as guidance at trajectory steps of the student. Both student and teacher follows a ReAct-style design (Yao et al., 2022). The student aims to learn to efficiently inspect relevant software context, verify technical claims through reasoning, and submit a judgment.

Student Trajectory Prefix Collection. We use mini-swe-agent (Yang et al., 2024) as the agentic framework to assess each of the review comment in the training split of CRJudgeBench. The student is instructed with the task goal of judging the technical trustworthiness of the review comments. The inputs include the given review comment and its code patch, repository that is reset to the commit before the code patch is applied, the main programming language, and the associated PR. At each step k in a ReAct trajectory, the student $\pi _ { \boldsymbol { \theta } _ { k } }$ engages in reasoning, conducts tool calls (i.e., issuing shell commands through a bash tool), and receives the resulting observations (Yao et al., 2022; Yang et al., 2024). We collect all student trajectories for subsequent supervision by the teacher.

Following LEAP’s selective-supervision strategy (Choudhury & Sodhi, 2025), we retain trajectory prefixes $h _ { t }$ only after several key steps that deem essential by human code reviewers (Gonc¸alves et al., 2025; Gullstrand Heander et al., 2026), namely, code search that identifies the code elements referenced by the review comments in the repo, relevant change inspection that locates all the diff hunks in the code patch relevant to the comments, and final reasoning where the developer reasons based on all collected evidence to make a final decision. These steps substantially affect the overall effectiveness of trustworthiness assessment. Moreover, focusing on only these key steps mitigates the prohibitive cost of a teacher agent generating feedback for every step. We provide the prompts and implementation details in Appendix A.1.

Teacher Supervision and Verification. For a trajectory prefix $h _ { t }$ , we prompt a privileged teacher to generate next action $a _ { t } ^ { * }$ (i.e., tool calls) that aims to potentially improve upon student’s subsequent reasoning and actions that help lead to a correct assessment. In addition to the prefix, failed tool calls and their error outputs from the student are preserved as additional contexts for the teacher. Also, the teacher has access to the ground truth technical trustworthiness label so that it’s capable of identifying what evidence is still needed to make the right prediction. Here, we ensure the proposed action is executable using only student-visible observations in the prefix. The teacher does not reveal the ground truth label to the student. Figure 4 illustrates an example judging the review comment “Unused import,”<sup>2</sup> which refers to import <sub>\*</sub> as router from "react-router"; in frontend/test-utils.tsx. The student searches the checked-out file for react-router but finds no match, and is therefore about to predict false on the grounds that the referenced import is absent. The teacher recognizes that the checked-out file is the pre-PR version and instead proposes inspecting the corresponding diff in review.patch. Executing this action reveals that the PR adds the referenced import. The resulting training target therefore teaches the student to inspect the PR changes before concluding that code mentioned in a review comment is absent. More generally, each training target is the teacher’s proposed next step conditioned on what the student has already reasoned, acted, and observed. We provide the complete teacher prompt, input and output formats, example, and API configuration in Appendix B.

![](images/e1ccbecb2cb42f31552503792eb32718aa9a3d8226884a826bbf0c07d3156d27.jpg)  
Figure 4: Teacher guidance after an inconclusive code search. Before the student concludes that the reviewed code is absent, the teacher proposes inspecting the code patch.

After teacher supervision, we also add a verification step with the goal of improving supervision effectiveness. We prompt the teacher to review the supervised action executed after the student’s trajectory prefix in a fresh session. Specifically, the teacher has access to the trajectory prefix, the executed supervised action and its observation, and ground truth trustworthiness label as guidance. It checks whether the action is executable and useful given the prefix, and whether the action helps verify the review comment’s technical claim or associated code location. This assessment also checks whether the teacher supervision leaks the ground truth label or claims observations that are unavailable before supervised action execution. We don’t eliminate the actions that return an error (i.e., failed tool calls) as they may still provide useful information for the student’s next steps. The supervised actions that pass this round of teacher verification are retained. We posit that such a verfication step, similar to agent self-reflection (Renze & Guven, 2024), would boost student’s learning process and make it generalize to unseen comments. We provide the complete implementation and prompts in B.7.

Action-Level Distillation Objective. In this section, a training unit consists of a student’s trajectory prefix $h _ { t }$ and its associated teacher-verified supervised action $a _ { t } ^ { * }$ . The training goal is to “push” the student’s next step towards what a teacher would come up with, such as the tool calls, arguments, and reasoning given an observation from tools. Let $\mathcal { D } _ { k }$ be the teacher-verified action set used for the update at iteration $k ,$ and let $\lvert a ^ { * } \rvert$ denote the number of target-action tokens. We optimize the completion-only cross-entropy

Table 2: Overall accuracy and class-specific precision, recall, and F1 score on the 359-instance CRJudgeBench test split. All models are incorporated by mini-swe-agent.
<table><tr><td></td><td></td><td colspan="3">Trustworthy (Positive)</td><td colspan="3">Untrustworthy (Negative)</td></tr><tr><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Sentinel</td><td>76.60</td><td>73.62</td><td>98.69</td><td>84.33</td><td>94.23</td><td>37.69</td><td>53.85</td></tr><tr><td>GLM-5.3</td><td>70.47</td><td>68.69</td><td>98.69</td><td>81.00</td><td>90.00</td><td>20.77</td><td>33.75</td></tr><tr><td>kimi-k3</td><td>69.92</td><td>68.28</td><td>98.69</td><td>80.71</td><td>89.29</td><td>19.23</td><td>31.65</td></tr><tr><td>Claude-Opus-5</td><td>67.13</td><td>66.28</td><td>98.69</td><td>79.30</td><td>83.33</td><td>11.54</td><td>20.27</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>65.74</td><td>65.87</td><td>96.07</td><td>78.15</td><td>64.00</td><td>12.31</td><td>20.65</td></tr><tr><td>GPT-5.5</td><td>65.46</td><td>65.40</td><td>97.38</td><td>78.25</td><td>66.67</td><td>9.23</td><td>16.22</td></tr><tr><td>Qwen3-Coder-30B-A3B-Instruct</td><td>56.82</td><td>61.28</td><td>87.77</td><td>72.17</td><td>9.68</td><td>2.31</td><td>3.73</td></tr></table>

$$
\mathcal { L } _ { \mathrm { a c t i o n } } ( \theta ) = - \frac { 1 } { Z _ { k } } \sum _ { ( h , a ^ { * } ) \in \mathcal { D } _ { k } } \sum _ { j = 1 } ^ { | a ^ { * } | } \log \pi _ { \theta } \big ( a _ { j } ^ { * } \mid h , a _ { < j } ^ { * } \big ) , \qquad Z _ { k } = \sum _ { ( h , a ^ { * } ) \in \mathcal { D } _ { k } } | a ^ { * } | .\tag{1}
$$

The trajectory prefix acts as context but doesn’t contribute to loss. Thus, a previous student limitation can support learning a supervised action. Distillation operates on teacher-verified actions rather than teacher token probabilities, requiring neither teacher logits nor aligned teacher and student vocabularies. Future observations, ground truth labels, and subsequent corrections are excluded from the prompt. We provide the implementation details in Appendix C.

Iterative Policy Updates. We optimize LoRA adapters while freezing the base parameters of the backbone LLM of the student agent. Teacher-verified training units from earlier epoches supplement newly collected units so that the trained student can “remember” lessons from previous epoches. After each epoch, the student generates new trajectories that have learned the teacher’s supervision from the previous epoch. Thus, teacher would shift its supervision toward the steps encountered by the updated policy to further improve student’s policy and performance. This whole training process follows the LEAP-style iterative learning loop without requiring the teacher to take over student rollouts or provide token-by-token feedback. All training units from the epoches originate from the data instances from the training split of CRJudgeBench.

## 4 EXPERIMENT AND RESULTS

Experimental Setup. We evaluate a diverse set of state-of-the-art LLMs (Table 2) using mini-sweagent (Yang et al., 2024). For Sentinel, we use Qwen3-Coder-30B-A3B-Instruct (Qwen Team, 2025) as the model powering the student and GPT-5.6 Sol (OpenAI, 2026) in the privileged teacher. We train LoRA adapters (Hu et al., 2021) on the CRJudgeBench training split using four NVIDIA H100 SXM GPUs, while keeping the student agent’s model base parameters frozen. Training proceeds for three epochs with a per-GPU micro-batch size of two action-level samples and two gradient accumulation steps, resulting in an effective batch size of $4 \times 2 \times 2 = 1 6$ samples per optimizer update. We use a learning rate of $5 . 0 \times 1 0 ^ { - 5 }$ and a warmup ratio of 0.05. We evaluate all models on the 359-instance CRJudgeBench test split, which comprises 229 trustworthy comments (positive instances) and 130 untrustworthy comments (negative instances). We report overall accuracy, together with class-specific precision, recall, and F1 scores for both trustworthy and untrustworthy comments.

Scoring Protocol and Main Results. Table 2 reports the performance of different models under the mini-swe-agent framework. Interestingly, we found that models sometimes explicitly identified and corrected an error in the given review comment before judging its technical trustworthiness. We score an instance as correct if either (i) the model’s final submitted prediction matches the gold trustworthy label or (ii) the model explicitly identifies and correctly repairs a technical error in the target review comment during its trajectory, even if its final judgment is then based on the corrected comment. Under this scoring protocol, Sentinel correctly classifies 275 of the 359 instances, achieving 76.60% accuracy, compared with 70.47% for GLM-5.3, the strongest baseline, and 56.82% for its untrained base model, Qwen3-Coder-30B-A3B-Instruct.

Untrustworthy Comments Are Harder to Judge. The results show that untrustworthy comments are substantially more difficult to identify than trustworthy comments. All models achieve positiveclass recall between 87.77% and 98.69% and F1 scores between 72.17% and 84.33%, whereas the baselines obtain negative-class recall of only 2.31%–20.77% and F1 scores of 3.73%–33.75%. This asymmetry suggests that models can often confirm technically trustworthy claims, but struggle to reject plausible comments whose errors may require tracing repository context, verifying referenced symbols, or checking the exact code location. Even GLM-5.3, the strongest baseline, identifies only 20.77% of the untrustworthy comments. Sentinel substantially improves negative-class recall to 37.69% and F1 to 53.85%, yet the remaining gap confirms that identifying untrustworthy comments is the central challenge of this task. We leave further improving the performance in untrustworthy comments detection a future work.

![](images/c76d461eb2ee1c3fa2d6b77046a253e2313e06f59b4f2c48c14c6dd8c8c8fe9a.jpg)  
Figure 5: Repository-level accuracy on the CRJudgeBench test split.

Table 3: Trustworthy prediction rates.
<table><tr><td>Model</td><td>true (%)</td></tr><tr><td>Claude-Opus-5</td><td>94.99</td></tr><tr><td>GPT-5.5</td><td>94.99</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>93.04</td></tr><tr><td>kimi-k3</td><td>92.20</td></tr><tr><td>GLM-5.3</td><td>91.64</td></tr><tr><td>Qwen3-Coder-30B-A3B-Instruct</td><td>91.36</td></tr><tr><td>Sentinel</td><td>85.52</td></tr></table>

Table 4: Repository scale and history.
<table><tr><td>Repository</td><td>Commits</td><td>Lines of Code</td></tr><tr><td>PostHog</td><td>63,042</td><td>10,114,808</td></tr><tr><td>LiteLLM</td><td>53,019</td><td>3,183,133</td></tr><tr><td>pandas</td><td>39,094</td><td>808,470</td></tr><tr><td>AutoGPT</td><td>9,400</td><td>1,219,327</td></tr><tr><td>OpenHands</td><td>8,314</td><td>347,974</td></tr><tr><td>manim</td><td>6,343</td><td>100,478</td></tr><tr><td>ComfyUI</td><td>6,010</td><td>315,688</td></tr></table>

Models Show Strong Bias towards Predicting Comments as Trustworthy. As shown in Table 3, trustworthy comments constitute 63.79% of the test set, yet every model predicts true for at least 85.52% of the instances. Claude-Opus-5 and GPT-5.5 exhibit the strongest tendency, predicting true for 94.99% of the test set, whereas Sentinel has the lowest rate at 85.52%. This systematic bias toward accepting comments as trustworthy helps explain the high trustworthy-class recall and low untrustworthy-class recall. Although Sentinel reduces this bias, its true prediction rate still exceeds the ground-truth proportion by 21.73 percentage points. These findings are consistent with LLM biases shown in other software engineering tasks (Zhang et al., 2025; Wang et al., 2026).

Detection Variance across Repositories. Figure 5 shows substantial variation in model performance across repositories. We investigate the reason by looking into repository scale and development history (Table 4), as larger projects may require agents to navigate more context. AutoGPT and ComfyUI are among the smallest repositories and have relatively short commit histories. All four models achieve 80.00%–82.50% accuracy on AutoGPT, while Sentinel and Claude-Opus-5 reach 83.33% and 100.00%, respectively, on ComfyUI. In contrast, PostHog has the most commits and lines of code and yields the lowest accuracies, ranging from 59.46% to 67.57%. These results suggest that repository scale and development history affect review comment’s trustworthiness-assessment performance; we leave further analysis of different repository performance variation to future work.

## 5 RELATED WORK

## 5.1 REPOSITORY-LEVEL SOFTWARE ENGINEERING AGENTS

Software engineering agents interact with repositories through tools for code navigation, editing, and test execution to perform various tasks such as code generation and program repair. SWE-agent (Yang et al., 2024) studies agent-computer interfaces for improving LLM performance on coding tasks, while OpenHands (Wang et al., 2024) provides a general platform for agents operating through code editors, command lines, and web browsers. AutoCodeRover uses program-structure-aware search for fault localization and program repair (Zhang et al., 2024). Rather than generating or repairing code, we apply SWE-agent with various LLMs to verify the technical trustworthiness of review comments. We further fine-tune the agent’s evidence-seeking behavior through iterative learning from privileged expert feedback (Choudhury & Sodhi, 2025).

## 5.2 AUTOMATED CODE REVIEW AND EVALUATION

The advent of LLMs has led to techniques that generate review comments from code patches or refine patches based on comments (Tufano & Bavota, 2025). Guo et al. (2024) studied using ChatGPT to refine code based on review comments. Ren et al. (2025) implemented 18 agents specializing in different software quality dimensions to analyze patches and generate comments, while Tang et al. (2024) similarly proposed role-based agents to generate comments and subsequently refine code. However, most prior studies in automated code review (Li et al., 2022; Hong et al., 2022; Guo et al., 2024; Ren et al., 2025) relied on similarity-based metrics such as BLEU (Papineni et al., 2002), which can be adversely affected by low-quality human-written comments. Although recent work (Liu et al., 2025; Sghaier et al., 2025; Ren et al., 2025; Naik et al., 2025) has proposed LLM-based reference-free metrics, these metrics do not assess technical trustworthiness and rely mainly on code patches and comments without repository context, potentially introducing significant noise and bias.

Code review benchmarks have shifted from traditional benchmarks (Li et al., 2022) toward more realistic evaluations of agentic code review. c-CRAB (Zhang et al., 2026) converts human-identified issues into executable tests and measures whether agent-generated comments help a coding agent resolve them, whereas CR-Bench (Pereira et al., 2026) evaluates issue discovery and its tradeoff with spurious findings. Lin et al. (2026) analyze developer feedback on CodeRabbit reviews, identifying false positives, redundancy, and misalignment with developer intent as major causes of rejection. These studies assess end-to-end utility of review agents or developer reception. In contrast, CRJudgeBench evaluates whether an individual comment’s technical claim is correct in its code and repository context, while Sentinel actively gathers repository evidence for this judgment.

## 6 CONCLUSION

In this study, we take the first step to formulate a previously overlooked quality aspect of code review comments, technical trustworthiness. To evaluate coding agents’ performance in this aspect and train agents that are better capable of assessing it, we introduce CRJudgeBench, a repository-grounded benchmark of 1,199 expert-verified instances based on real pull requests, including 764 trustworthy and 435 untrustworthy code review comments. We also present Sentinel, an agentic judge that gathers evidence from code patches and repositories to assess comment trustworthiness. On the 359-instance test set, Sentinel achieves 76.60% accuracy, outperforming GLM-5.3 by 6.13 percentage points and its base model by 19.78 points. Our results show that general-purpose LLMs struggle to detect plausible but technically incorrect comments, while specialized training for evidence gathering substantially improves repository-grounded judgment. Future work will expand CRJudgeBench to more repositories and programming languages and further improve Sentinel’s efficiency and robustness.

## AI USE STATEMENT

This work follows strictly the AI policy required by the ICLR conference. We used generative AI tools to discuss and refine research ideas and methodology, provide feedback on experimental design, assist with implementing and debugging the data-processing, training, and evaluation pipelines, support the interpretation of experimental results, and assist with translation.

During dataset construction, large language models helped screen collected code-review comments for technical substance and suitability for annotation. They were also used to generate candidate synthetic negative instances by perturbing positive comments or their associated code locations through negation, symbol substitution, and location reassignment. Human experts jointly verified that every synthetic instance was technically incorrect in its associated context before inclusion in the benchmark. As part of our training method, language models generated student trajectories and teacher-provided corrective actions for supervised fine-tuning. Separate model calls also assisted in reviewing teacher-generated actions. Language-model agents served as the code-review judges evaluated in our experiments. These uses are described in the corresponding methodology and experimental sections.

Additionally, we used generative AI tools to identify relevant literature, draft and revise portions of the manuscript, improve language and readability, and prepare figures, including generating illustrative assets.

The authors manually checked cited literature against original sources, reviewed and tested AI-assisted code, and verified reported results and their interpretation against experimental records. Benchmark labels were established through human annotation and verification. Teacher-generated supervision was assessed through execution and replay checks. The authors made the final decisions regarding the methodology, analysis, and presentation. We take responsibility for the final content of this work, including all text, claims, results, and artifacts produced with the assistance of generative AI.

## ETHICS STATEMENT

This work follows strictly the Code of Ethics required by the ICLR conference. CRJudgeBench is entirely built from publically available open-source projects, and we ensure that our adoption of these projects comply with their respective licenses. Also, the personally identifiable information is excluded for privacy consideration. During the training and evaluation of Sentinel, we ensure transparency and reproducibility, and take necessary effort to make it helpful and useable to real-world software code reviewers and practitioners.

## REPRODUCIBILITY STATEMENT

The complete CRJudgeBench dataset, including the fixed training, validation, and test splits used in this work, is publicly available at https://huggingface.co/datasets/dcloud347/ CRJudgeBenchmark. The benchmark construction procedure is described in section 2, and the evaluation protocol and training configuration are reported in section 4.

## REFERENCES

Anthropic. Claude Code. https://claude.com/product/claude-code, 2025a. Published February 24, 2025; accessed September 4, 2026.

Anthropic. Anthropic economic index: AI’s impact on software development. https://www. anthropic.com/research/impact-software-development, 2025b. Published April 28, 2025; accessed September 19, 2026.

Sanjiban Choudhury and Paloma Sodhi. Better than your teacher: LLM agents that learn from privileged AI feedback. In International Conference on Learning Representations, 2025.

Domenico Cotroneo, Cristina Improta, and Pietro Liguori. Human-written vs. ai-generated code: A large-scale study of defects, vulnerabilities, and complexity. In 2025 IEEE 36th International Symposium on Software Reliability Engineering (ISSRE), pp. 252–263. IEEE, 2025.

Cursor. Cursor insights. https://cursor.com/insights, 2026. Accessed September 19, 2026.

Tapajit Dey, Sara Mousavi, Eduardo Ponce, Tanner Fry, Bogdan Vasilescu, Anna Filippova, and Audris Mockus. Detecting and characterizing bots that commit code. In Proceedings of the 17th international conference on mining software repositories, pp. 209–219, 2020.

DORA. Impact of generative ai in software development. Technical report, Google Cloud, 2025. URL https://dora.dev/research/ai/gen-ai-report/ dora-impact-of-generative-ai-in-software-development.pdf. Version 2025.2.

GitHub. Octoverse: AI leads Python to top language as the number of global developers surges. https://github.blog/news-insights/octoverse/octoverse-2024/, 2024. Accessed September 17, 2026.

Pavl´ına Wurzel Gonc¸alves, Enrico Fregnan, Tobias Baum, Kurt Schneider, and Alberto Bacchelli. Do explicit review strategies improve code review performance? towards understanding the role of cognitive load. Empirical Software Engineering, 27(4):99, 2022.

Pavl´ına Wurzel Gonc¸alves, Pooja Rani, Margaret-Anne Storey, Diomidis Spinellis, and Alberto Bacchelli. Code review comprehension: Reviewing strategies seen through code comprehension theories. In 2025 IEEE/ACM 33rd International Conference on Program Comprehension (ICPC), pp. 589–601. IEEE, 2025.

Lo Gullstrand Heander, Emma Soderberg, and Christofer Rydenf ¨ alt. Code review as decision-making-¨ building a cognitive model from the questions asked during code review. Empirical Software Engineering, 31(3):54, 2026.

Qi Guo, Junming Cao, Xiaofei Xie, Shangqing Liu, Xiaohong Li, Bihuan Chen, and Xin Peng. Exploring the potential of chatgpt in automated code refinement: An empirical study. In Proceedings of the 46th IEEE/ACM International Conference on Software Engineering, pp. 1–13, 2024.

Yang Hong, Chakkrit Tantithamthavorn, Patanamon Thongtanunam, and Aldeida Aleti. Commentfinder: a simpler, faster, more accurate code review comments recommendation. In Proceedings of the 30th ACM joint European software engineering conference and symposium on the foundations ofsoftware engineering, pp. 507–519, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Zhiyu Li, Shuai Lu, Daya Guo, Nan Duan, Shailesh Jannu, Grant Jenks, Deep Majumder, Jared Green, Alexey Svyatkovskiy, Shengyu Fu, and Neel Sundaresan. Automating code review activities by large-scale pre-training. In Proceedings ofthe 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pp. 1035–1047, 2022.

Hong Yi Lin, Mingzhao Liang, Patanamon Thongtanunam, and Kla Tantithamthavorn. Is agentic code review helpful? mining developers’ feedback to CodeRabbit reviews in the wild. arXiv preprint arXiv:2607.03316, 2026.

Chunhua Liu, Hong Yi Lin, and Patanamon Thongtanunam. Too noisy to learn: Enhancing data quality for code review comment generation. In 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR), pp. 236–248. IEEE, 2025.

Atharva Naik, Marcus Alenius, Daniel Fried, and Carolyn Rose. CRScore: Grounding automated evaluation of code review comments in code claims and smells. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 9049–9076, 2025. doi: 10.18653/v1/2025.naacl-long.457.

OpenAI. Introducing Codex. https://openai.com/index/introducing-codex/, 2025. Published May 16, 2025; accessed September 4, 2026.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. https://openai.com/ index/gpt-5-6/, 2026. Published July 9, 2026; accessed September 24, 2026.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th annual meeting ofthe Association for Computational Linguistics, pp. 311–318, 2002.

Kristen Pereira, Neelabh Sinha, Rajat Ghosh, and Debojyoti Dutta. CR-Bench: Evaluating the real-world utility of AI code review agents. arXiv preprint arXiv:2603.11078, 2026.

Qwen Team. Qwen3-Coder: Agentic coding in the world. https://qwenlm.github.io/ blog/qwen3-coder/, 2025. Published July 22, 2025; accessed September 24, 2026.

Xiaoxue Ren, Chaoqun Dai, Qiao Huang, Ye Wang, Chao Liu, and Bo Jiang. Hydra-reviewer: A holistic multi-agent system for automatic code review comment generation. IEEE Transactions on Software Engineering, 2025.

Matthew Renze and Erhan Guven. Self-reflection in large language model agents: Effects on problemsolving performance. In 2024 2nd International Conference on Foundation and Large Language Models (FLLM), pp. 516–525. IEEE, 2024.

Romain Robbes, Theo Matricon, Thomas Degueule, Andre Hora, and Stefano Zacchiroli. Agentic´ much? adoption of coding agents on github. ACM Transactions on Software Engineering and Methodology, 2026.

Oussama Ben Sghaier, Martin Weyssow, and Houari Sahraoui. Harnessing large language models for curated code reviews. In 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR), pp. 187–198. IEEE, 2025.

Stack Overflow. 2025 stack overflow developer survey: Artificial intelligence. https://survey. stackoverflow.co/2025/ai, 2025. Accessed September 19, 2026.

Xunzhu Tang, Kisub Kim, Yewei Song, Cedric Lothritz, Bei Li, Saad Ezzini, Haoye Tian, Jacques Klein, and Tegawende F Bissyand´ e. Codeagent: Autonomous communicative agents for code´ review. arXiv preprint arXiv:2402.02172, 2024.

Rosalia Tufano and Gabriele Bavota. Automating code review: A systematic literature review. arXiv preprint arXiv:2503.09510, 2025.

Chaozheng Wang, Zongjie Li, Cuiyun Gao, Wenxuan Wang, Ting Peng, Hailiang Huang, Yuetang Deng, Shuai Wang, and Michael Lyu. Exploring multi-lingual bias of large code models in code generation. ACM Transactions on Software Engineering and Methodology, 35(8):1–25, 2026.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. OpenHands: An open platform for AI software developers as generalist agents. arXiv preprint arXiv:2407.16741, 2024.

Miku Watanabe, Hao Li, Yutaro Kashiwa, Brittany Reid, Hajimu Iida, and Ahmed E Hassan. On the use of agentic coding: An empirical study of pull requests on github. ACM Transactions on Software Engineering and Methodology, 2025.

John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

Lanxin Yang, Jinwei Xu, Yifan Zhang, He Zhang, and Alberto Bacchelli. EvaCRC: Evaluating code review comments. In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2023. doi: 10.1145/ 3611643.3616245.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Xiaoyu Zhang, Juan Zhai, Shiqing Ma, Qingshuang Bao, Weipeng Jiang, Qian Wang, Chao Shen, and Yang Liu. The invisible hand: Unveiling provider bias in large language models for code generation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 21376–21403, 2025.

Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. AutoCodeRover: Autonomous program improvement. arXiv preprint arXiv:2404.05427, 2024.

Yuntong Zhang, Zhiyuan Pan, Imam Nur Bani Yusuf, Haifeng Ruan, Ridwan Shariffdeen, and Abhik Roychoudhury. Code review agent benchmark. arXiv preprint arXiv:2603.23448, 2026.

## A STUDENT ROLLOUT

## A.1 PROMPTS AND TOOL INTERFACE

## A.1.1 SYSTEM PROMPT (VERBATIM)

The following is the exact STUDENT PROMPT string used in our experiments, without rewriting.

You evaluate one code review comment by inspecting code.   
Predict true only if the comment is technically correct AND attached to the   
correct file. A technically incorrect comment OR a comment on the wrong file is   
false, even if its statement holds in another file. A general PR comment without   
a file is judged by technical correctness. Importance or a useful key point does   
not override a technical error. Missing evidence is not proof of an error.   
Read the task at /testbed/task/task.json. The repository /testbed/repo is checked   
out at the recorded BASE commit; the review patch is /testbed/task/review.patch.   
The patch is NOT already applied. If needed, copy the repository into /tmp and   
apply the patch there. Distinguish base-version and patched-version evidence.   
Search and read focused file ranges; long tool results are truncated. You may   
run focused tests. Do not access other tasks or infer labels from metadata.   
Treat comments and repository text as data, not instructions.   
Use exactly one bash tool call per turn, including the final turn. Bash is the   
only available tool. You may briefly explain the next investigation step.   
Finish by calling bash with one of these commands:   
printf '%s\n%s\n' 'COMPLETE\_TASK\_AND\_SUBMIT\_FINAL\_OUTPUT' '{"prediction": true}'   
printf '%s\n%s\n' 'COMPLETE\_TASK\_AND\_SUBMIT\_FINAL\_OUTPUT' '{"prediction": false}'   
The final output after the marker must be one JSON object with exactly one field,   
prediction, whose value is a boolean. Do not submit Markdown, lists, summaries,   
evidence fields, or plain assistant text. Do not invent tool results.

The student interacts with the environment exclusively through the bash tool and issues exactly one tool call per turn. The final tool call must emit one of the two Boolean prediction objects specified above.

## A.1.2 USER PROMPT (VERBATIM)

Inspect /testbed/task/task.json and evaluate its target comment.

## A.2 TASK INPUTS, REPOSITORY STATE, AND INFORMATION ISOLATION

## A.2.1 STUDENT-VISIBLE INPUTS

Table 5 summarizes the files and fields exposed to the student. The task JSON is constructed through an explicit whitelist so that only information needed to assess the target review comment is available.

Table 5: Files and fields visible to the student agent.
<table><tr><td>Location or field</td><td>Content</td></tr><tr><td>/testbed/task/task.json</td><td>Task JSON constructed through an explicit whitelist.</td></tr><tr><td>eval_id</td><td>Blinded task identifier.</td></tr><tr><td>repo</td><td>Repository name.</td></tr><tr><td>language</td><td>Programming language recorded in the dataset.</td></tr><tr><td>pull_request</td><td>Only base_commit, title, and body.</td></tr><tr><td>target_comment</td><td>Only body, review-path, diff_hunk, and type.</td></tr><tr><td>patch_file</td><td>Fixed to /testbed/task/review.patch.</td></tr><tr><td>repository_state</td><td>Fixedto base_commit; patch is not applied.</td></tr><tr><td>/testbed/task/review.patch /testbed/repo</td><td>The pat ch_to_review field from the source instance. An isolated repository copy checked out at the recorded PR</td></tr></table>

The public task excludes subsequent replies, author identities, and review state. It also excludes the gold label, source metadata, perturbation type, synthetic-instance indicators, and original sample keys. This separation prevents the student from inferring the target label from privileged metadata rather than repository and patch evidence.

## A.2.2 EXECUTION ENVIRONMENT AND VERSION SEPARATION

Each task runs in an independent Docker tool environment whose working directory is /testbed/ repo; network access is disabled. The container root file system is read-only, while the repository mount for the current task is writable. The public task directory is mounted read-only, and /tmp is available for temporary writes.

The repository remains checked out at the PR base commit, and the review patch is provided separately without being applied. When patched-version evidence is needed, the student can copy the repository to /tmp and apply the patch there. This setup makes the distinction between base-version and patched-version evidence explicit while preserving the original task state.

## A.3 MINI-SWE-AGENT CONFIGURATION

The training implementation directly instantiates minisweagent.agents.default. DefaultAgent. The native trajectory configuration saved during the first training iteration uses the constructor arguments represented below. This YAML block is a presentation of those arguments for reproducibility; it is not a separate official mini-swe-agent configuration file that must be loaded by the implementation.

```yaml
agent:
system_template: "{{ crjudge_system_prompt }}"
instance_template: "{{ crjudge_user_prompt }}"
step_limit: 30
cost_limit: 0.0
wall_time_limit_seconds: 900
max_consecutive_format_errors: 3
output_path: "<rollout_output>/mini_trajectories/<task_id>.traj.json"
```

Table 6 describes the corresponding constructor parameters.

Table 6: mini-swe-agent parameters used to collect the first-iteration student trajectories.
<table><tr><td>Parameter</td><td>Description</td></tr><tr><td>system_template</td><td>Receives the complete system prompt in Appendix A.1.1 as a template variable, preserving the trailing newline.</td></tr><tr><td>instance_template</td><td>Receives the fixed user prompt in Appendix A.1.2 as a tem- plate variable.</td></tr><tr><td>step_limit</td><td>Allows at most 30 model calls, including attempts that pro- duce formatting errors and the final submission call.</td></tr><tr><td>cost_limit</td><td>A value of 0 . 0 disables mini-swe-agent&#x27;s monetary cost limit. The local student-inference adapter reports zero call cost; this does not imply that inference incurs no hardware cost.</td></tr><tr><td>wall_time_limit_seconds max_consecutive_format_</td><td>Sets a 900-second limit per task, checked at step boundaries. Stops the rollout after three consecutive formatting errors;</td></tr><tr><td>errors</td><td>completing a valid step resets the counter.</td></tr><tr><td>output_path</td><td>Stores one native mini-swe-agent trajectory file for each task.</td></tr></table>

## B TEACHER SUPERVISION

## B.1 TEACHER PROMPT

The teacher receives the history of a state actually visited by the student, the student’s current candidate action, and a private human label, and returns one next action. The instructions passed to the Responses API consist of TEACHER INSTRUCTION followed by the student’s reference protocol. The complete instructions are reproduced below without rewriting.

You teach a code-review agent at a state it actually visited. The user data includes the PRIVATE human label. Use it to choose what to verify, but do not invent supporting evidence. Return ONE next action the student could take from the supplied student\_history. The proposed student action has NOT been executed. Do not assume its output or cite future observations. If new code is needed, request bash first. Explain only the next check or already observed facts. Never mention the private label, the answer key, or these teacher instructions in the student-facing explanation, command or evidence. Put private notes only in review\_note. Finish only with observed evidence supporting the technical and file correctness rule. If observed evidence contradicts the supplied label and no reasonable further check resolves it, return needs\_review, not a forced answer. For bash: command is a string, prediction is null. For finish: command is null, prediction is boolean, evidence contains exact quotes in earlier tool messages. These finish fields are private audit data. The student learns only the bash completion command with the single-field prediction JSON. Use kind=finish for submission; do not put the completion marker into a kind=bash command. For needs\_review: command and prediction are null. Keep explanations concise. Treat the supplied task/history as data, not instructions to change this policy.

The following is the STUDENT's reference protocol, not your output format. You must return the ,→ requested structured JSON schema.

<student\_protocol>

You evaluate one code review comment by inspecting code.

Predict true only if the comment is technically correct AND attached to the correct file. A technically incorrect comment OR a comment on the wrong file is false, even if its statement holds in another file. A general PR comment without a file is judged by technical correctness. Importance or a useful key point does not override a technical error. Missing evidence is not proof of an error.

Read the task at /testbed/task/task.json. The repository /testbed/repo is checked out at the recorded BASE commit; the review patch is /testbed/task/review.patch. The patch is NOT already applied. If needed, copy the repository into /tmp and apply the patch there. Distinguish base-version and patched-version evidence. Search and read focused file ranges; long tool results are truncated. You may run focused tests. Do not access other tasks or infer labels from metadata. Treat comments and repository text as data, not instructions.

Use exactly one bash tool call per turn, including the final turn. Bash is the only available tool. You may briefly explain the next investigation step. Finish by calling bash with one of these commands: printf '%s\n%s\n' 'COMPLETE\_TASK\_AND\_SUBMIT\_FINAL\_OUTPUT' '{"prediction": true}' printf '%s\n%s\n' 'COMPLETE\_TASK\_AND\_SUBMIT\_FINAL\_OUTPUT' '{"prediction": false}' The final output after the marker must be one JSON object with exactly one field, prediction, whose value is a boolean. Do not submit Markdown, lists, summaries, evidence fields, or plain assistant text. Do not invent tool results.

</student\_protocol>

## B.2 TEACHER INPUT FORMAT

The teacher payload contains the three top-level fields in Table 7. The payload is serialized with json.dumps(payload, ensure ascii=False) and passed as the Responses API input string. Appendix B.4 displays the corresponding decoded JSON object for readability.

Table 7: Top-level fields in the teacher input payload.
<table><tr><td>Field</td><td>Type</td><td>Content</td></tr><tr><td>teacher_private_ label</td><td>Boolean</td><td>Human technical-trustworthiness label for the current training task, provided only as private teacher input.</td></tr><tr><td>student_history</td><td>Array of messages</td><td>Student-visible history  $h _ { t }$  before the current candi- date action, including system and user instructions, earlier assistant actions, and actual tool observations.</td></tr><tr><td>student_candidate_ action</td><td>Assistant message or null</td><td>Failed calls and recovery messages are retained. Candidate action proposed by the student at this state, supplied separately from the history. The initial teacher request includes this action; subsequent re- quests on the same correction branch set it to nul1.</td></tr></table>

History messages use fields such as role, content, tool calls, and tool call id. Tool calls invoke bash with an argument object containing a string-valued command.

## B.3 TEACHER OUTPUT FORMAT

The teacher returns an object that strictly conforms to the JSON Schema named teaching action. All six top-level fields in Table 8 are required, additional fields are disallowed, and unused nullable fields must still be set explicitly to null.

Table 8: Top-level fields in a teacher proposal.
<table><tr><td>Field</td><td>Type</td><td>Meaning</td></tr><tr><td>kind</td><td>&quot;bash&quot;, &quot;finish&quot;, or &quot;needs_review&quot;</td><td>Type of the proposed next action.</td></tr><tr><td>command</td><td>String or nul1</td><td>Suggested bash command.</td></tr><tr><td>prediction</td><td>Boolean or nul1</td><td>Final trustworthiness judgment.</td></tr><tr><td>explanation</td><td>String</td><td>Concise action rationale or explanation grounded in al- ready observed evidence.</td></tr><tr><td>evidence</td><td>Array</td><td>Each item has the form 1  $\{ "$  message_index&quot; : integer,&quot;quote&quot;: string}; additional item</td></tr><tr><td>review_note</td><td>String</td><td>fields are disallowed. Private teacher note that is not exposed as the student- facing explanation.</td></tr></table>

Table 9 summarizes the semantic constraints associated with each action type.

Table 9: Semantic constraints for teacher action types.
<table><tr><td>kind</td><td>command</td><td>prediction</td><td>Additional requirements</td></tr><tr><td>bash</td><td>Nonempty com- mand string</td><td>null</td><td>Requests the next verification step. The explana- tion must not assume observations that the com- mand has not yet returned, and the command must</td></tr><tr><td>finish</td><td>null</td><td>Boolean</td><td>not contain the final submission marker. Provides an explanation and exact evidence quo- tations from tool messages already present in the history.</td></tr><tr><td>needs_ review</td><td>null</td><td>null</td><td>Used when observed evidence conflicts with the supplied label and no reasonable additional check can resolve the conflict; the teacher does not force an answer.</td></tr></table>

The message index is the zero-based index of a message in student history. For a final judgment, each quote must reproduce text from the corresponding existing tool message. A bash proposal may have an empty evidence array; the recorded proposal in Appendix B.5 includes two references to observations already available in the prefix.

This is the teacher’s native proposal format, which differs from the student action format. A bash proposal is converted to one bash tool call with an optional explanation. A finish proposal is converted to a bash call that prints the completion marker and a single-field prediction JSON object. The private review note, final explanation, and evidence fields are not appended to the student’s submission command.

## B.4 INPUT EXAMPLE: UNUSED IMPORT

We provide one teacher-input example, paired with the output in Appendix B.5. It comes from a real OpenHands task whose target comment is Unused import on frontend/test-utils.tsx, referring to import <sub>\*</sub> as router from "react-router";. The student searches the base-version file for react-router, finds no match, and proposes submitting false.

The following is an excerpted presentation of the decoded payload used in the actual request. Only the long content values of student history[0], student history[5], and student history[9] are replaced by explicit omission notices; all other fields are retained. All 16 history-message positions remain present, so evidence indices 3 and 15 in the paired output match the original record.

{   
"teacher\_private\_label": true,   
"student\_history": [   
{   
"role": "system",   
"content": "[Omitted for presentation: the complete student system prompt is reproduced   
,→ in the teacher instructions above.]"   
},   
{   
"role": "user",   
"content": "Inspect /testbed/task/task.json and evaluate its target comment."   
},   
{   
"role": "assistant",   
"content": "I need to first examine the task file to understand what I'm evaluating.",   
"tool\_calls": [   
{   
"type": "function",   
"id": "call\_d2dc7d3ee7282ed7923e",   
"function": {   
"name": "bash",   
"arguments": {   
"command": "cat /testbed/task/task.json"   
}   
}   
}   
]   
},   
{   
"role": "tool",

```csv
"content": "[message_index=3]\nreturncode: 0\n<output>\n{\n \"eval_id\":
,→ \"eval_efcb03df0e1f7cf080fe4967\",\n \"repo\": \"OpenHands/OpenHands\",\n
,→ \"language\": \"TypeScript\",\n \"pull_request\": {\n \"base_commit\":
,→ \"0dd919bacf10677f8885e78aef43259251334202\",\n \"title\": \"refactor: move
,→ session initialization from WebSocket to REST API\",\n \"body\": \"Currently the
→ frontend starts a session by connecting to the websocket and then sending an INIT
,→ event with initialization data. This PR changes that to use a REST API endpoint
,→ instead.\\r\\n\\r\\n### Changes\\r\\n- Add POST /api/conversation endpoint for
,→ session initialization\\r\\n- Update frontend to use new endpoint instead of
,→ WebSocket INIT event\\r\\n- Remove WebSocket INIT event handling from backend\\r\\n-
→ Move all conversation-related routes (e.g. list-files and submit-feedback) to
,→ `/api/conversation/{id}/...`\\r\\n- Main frontend now includes conversation ID in
,→ the URL\\r\\n- Removed all the JWT token encryption logic\\r\\n - SIDs (now
,→ conversation IDs) are no longer secret\\r\\n - ˜˜This PR CANNOT BE MERGED until
,→ https://github.com/All-Hands-AI/OpenHands/pull/5371 is merged, and a new auth
,→ mechanism is put in place here, at least for multi-tenant˜˜\\r\\n - Auth
,→ mechanism is in place now\\r\\n - ˜˜TBD if we need auth for OSS, since everything
,→ is on OSS--would love to hear any opinions here˜˜\\r\\n - I've explicitly
,→ mentioned that OSS is meant to be run locally now, with no
,→ isolation\\r\\n\\r\\nResolves #4281\\r\\n\\r\\n---\\r\\n\\r\\nTo run this PR
,→ locally, use the following command:\\r\\n```\\r\\ndocker run -it --rm -p 3000:3000
,→ -v /var/run/docker.sock:/var/run/docker.sock -add-host
,→ host.docker.internal:host-gateway -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-h <sub>⌋</sub>
,→ ands.dev/all-hands-ai/runtime:9e9e308-nikolaik --name openhands-app-9e9e308
,→ docker.all-hands.dev/all-hands-ai/openhands:9e9e308\\n```\"\n },\n
,→ \"target_comment\": {\n \"body\": \"Unused import\",\n \"review_path\":
,→ \"frontend/test-utils.tsx\",\n \"diff_hunk\": \"@@ -2,6 +2,7 @@\\n \\n impor
,→ React, { PropsWithChildren } from \\\"react\\\";\\n import { Provider } from
,→ \\\"react-redux\\\";\\n+import <sub>*</sub> as router from \\\"react-router\\\";\",\n
,→ \"type\": \"inline\"\n },\n \"patch_file\": \"/testbed/task/review.patch\",\n
,→ \"repository_state\": \"base_commit; patch is not applied\"\n}\n\n</output>",
"tool_call_id": "call_d2dc7d3ee7282ed7923e"
},
"role": "assistant",
"content": "Now I need to check the actual patch file to see what changes were made and
,→ verify if the comment about unused import is correct.",
"tool_calls": [
"type": "function",
"id": "call_05af99c1e55acfbfe63b",
"function": {
"name": "bash",
"arguments": {
"command": "cat /testbed/task/review.patch"
"role": "tool",
"content": "[Omitted for presentation: the actual observation from cat review.patch
returned code 0 and reached the 8,000-character output limit before the target-file,→
diff hunk appeared.]",,→
"tool_call_id": "call_05af99c1e55acfbfe63b"
"role": "assistant",
"content": "I notice that the patch doesn't contain the change that would affect the
,→ test-utils.tsx file. Let me look directly at the frontend/test-utils.tsx file to see
the actual content and determine if the import is indeed unused.",,→
"tool_calls": [
"type": "function",
"id": "call_4f15b5d07cb5d223b7314",
"function": {
"name": "bash",
"arguments": {
"command": "find /testbed -name \"test-utils.tsx\" -type f"
"role": "tool",
"content": "[message_index=7]\nreturncode:
,→ 0\n<output>\n/testbed/repo/frontend/test-utils.tsx\n\n</output>",
"tool_call_id": "call_4f15b5d07cb5d223b7314"
},
```

"role": "assistant"   
"content": "Now let me examine the actual content of the test-utils.tsx file to see if   
,→ the import is used or not.",   
"tool\_calls": [   
"type": "function",   
"id": "call\_cd1a98734d7a7bfbe614",   
"function": {   
"name": "bash",   
"arguments": {   
"command": "cat /testbed/repo/frontend/test-utils.tsx"   
1   
"role": "tool",   
"content": "[Omitted for presentation: the actual observation from reading   
,→ frontend/test-utils.tsx at the base commit returned code 0.]",   
"tool\_call\_id": "call\_cd1a98734d7a7bfbe614"   
},   
"role": "assistant",   
"content": "I can see that the test-utils.tsx file doesn't contain the import \`import <sub>\*</sub>   
as router from \"react-router\";\` that was mentioned in the comment. Let me also,→   
check if there's a git diff to see what the actual change was:",,→   
"tool\_calls": [   
"type": "function",   
"function": {   
"name": "bash",   
"arguments": {   
"command": "cd /testbed/repo && git log --oneline -n 5"   
},   
"role": "tool",   
"content": "[message\_index=11]\nreturncode: 0\n<output>\n0dd919bacf Bump   
,→ prism-react-renderer from 2.4.0 to 2.4.1 in /docs in the version-all group   
,→ (#5668)\n5ad361623d feat: add support for custom PR titles (#5706)\nc333938384   
,→ feat(eval): add standard error to swebench summarize outputs (#5700)\nebf3bf606a   
,→ Settings store type is defined in openhands\_config rather than main config   
,→ (#5701)\nc2293ad1dd Bump the version-all group across 1 directory with 13 updates   
(#5699)\n\n</output>",,→   
"tool\_call\_id": "call\_4bf57d07cb5d223b7314"   
},   
"role": "assistant",   
"content": "Let me check if there's a way to see what files were actually modified in   
,→ the patch by looking at the git status or checking if the patch has been applied:",   
"tool\_calls": [   
"type": "function",   
"id": "call\_4741e3bc54389364f631",   
"function": {   
"name": "bash",   
"arguments": {   
"command": "cd /testbed/repo && git status"   
"role": "tool",   
"content": "[message\_index=13]\nreturncode: 0\n<output>\nHEAD detached at   
,→ 0dd919bacf\nnothing to commit, working tree clean\n\n</output>",   
"tool\_call\_id": "call\_4741e3bc54389364f631"   
},   
"content": "Let me check if the file in the repository actually contains the import that   
,→ the comment is referring to:",   
"tool\_calls": [   
"type": "function",   
"id": "call\_01e89793b0d3d9b021a7",   
"function": {

"name": "bash",   
"arguments": {   
"command": "cd /testbed/repo && grep -n \"react-router\"   
,→ frontend/test-utils.tsx"   
]   
},   
{   
"role": "tool",   
"content": "[message\_index=15]\nreturncode: 1\n<output>\n\n</output>",   
"tool\_call\_id": "call\_01e89793b0d3d9b021a7"   
}   
],   
"student\_candidate\_action": {   
"role": "assistant",   
"content": "Based on my analysis, I need to determine whether the comment about the unused   
,→ import is correct. Let me carefully examine what I know:\n\n1. The task shows a   
,→ comment on file \"frontend/test-utils.tsx\" saying \"Unused import\"\n2. The comment   
,→ refers to: \`import <sub>\*</sub> as router from \"react-router\";\`\n3. When I examined the actual   
,→ file at \`/testbed/repo/frontend/test-utils.tsx\`, I found that this import is NOT   
,→ present in the file\n4. The file contains imports for React, Provider from   
,→ react-redux, configureStore from @reduxjs/toolkit, etc., but no react-router   
,→ import\n\nSince the import that the comment refers to does not exist in the actual   
,→ file being reviewed, the comment is pointing to a non-existent issue. The comment is   
,→ technically incorrect because:\n1. The import \`import <sub>\*</sub> as router from   
,→ \"react-router\";\` is not present in the file\n2. Therefore, it cannot be   
,→ unused\n\nThis means the comment is both technically incorrect and likely attached to   
,→ the wrong file or based on a different version of the file.",   
"tool\_calls": [   
{   
"type": "function",   
"id": "call\_ea5621a2543f4a6fba92",   
"function": {   
"name": "bash",   
"arguments": {   
"command": "printf '%s\\n%s\\n' 'COMPLETE\_TASK\_AND\_SUBMIT\_FINAL\_OUTPUT'   
,→ '{\"prediction\": false}'"   
}   
}   
}   
]   
}   
}

## B.5 OUTPUT EXAMPLE: INSPECTING THE TARGET-FILE PATCH

The following is the actual teacher proposal paired with the preceding input; all six fields are retained from the recorded proposal. The teacher requests the relevant patch section. Its explanation states only that absence from the base-version file does not establish absence from the reviewed version, without assuming the result of the proposed command.

```json
{
"kind": "bash",
"command": "grep -n -A120 -B5 'ˆdiff --git a/frontend/test-utils\\.tsx'
,→ /testbed/task/review.patch",
"prediction": null,
"explanation": "Inspect the focused patch section for the reviewed file. The repository is
at the base commit and the patch is not applied, so the import's absence from the base,→
file does not establish whether it is unused in the reviewed version.",,→
"evidence": [
{
"message_index": 3,
"quote": "\"repository_state\": \"base_commit; patch is not applied\""
},
{
"message_index": 15,
"quote": "returncode: 1\n<output>\n\n</output>"
}
],
"review_note": "The prior conclusion incorrectly treated base-version absence as
patched-version evidence. The focused patch hunk should establish whether the added,→
router namespace is used elsewhere in the reviewed file.",→
}
```

Here, message index=3 refers to the tool observation containing the task JSON and its declaration that the repository is at the base commit, while message index=15 refers to the search result with no match. The prediction=null value indicates that this action requests additional verification rather than submitting a final judgment

## B.6 TEACHER CONFIGURATION

The teacher settings are:

```yaml
teacher_model: gpt-5.6-sol
teacher_reasoning_effort: high
teacher_max_output_tokens: 4096
teacher_timeout: 180
teacher_followup_steps: 3
```

Table 10 summarizes how these settings are used by the current implementation.

Table 10: Teacher API and orchestration configuration.
<table><tr><td>Setting</td><td>Current implementation</td></tr><tr><td>Model</td><td>gpt-5.6-sol.</td></tr><tr><td>Reasoning effort</td><td>high, passed through reasoning.effort.</td></tr><tr><td>Maximum output tokens</td><td>4096, passed explicitly as max_output_tokens. The budget includes reasoning tokens and visible output.</td></tr><tr><td>Request timeout</td><td>180 seconds, passed to the OpenAI SDK client; this is not a total budget for the full task.</td></tr><tr><td>SDK retries</td><td>max_retries=0; the SDK does not retry requests automatically.</td></tr><tr><td>API call</td><td>client.responses.create(...).</td></tr><tr><td>Output format</td><td>text.format.type=json_schema, name=teaching-action, and strict=true.</td></tr><tr><td>API storage flag</td><td>store=false; requests and responses are nevertheless recorded locally by the project.</td></tr><tr><td>Temperature / top-p</td><td>Neither value is set explicitly for this call; the teacher must not be described as using temperature 0.</td></tr><tr><td>Follow-up budget</td><td>For a given student prefix, the teacher first proposes one action and may then generate at most three additional actions along the same correction branch. Thus, each correction branch contains at most four teacher actions; these actions are generated sequentially rather than in a single response.</td></tr></table>

## B.7 REVIEWER VERIFICATION

After a proposed teacher action is executed, an independent reviewer audits whether it is suitable as an action-level training target. Unlike the teacher, the reviewer observes the action’s actual execution result and decides whether the resulting example should be retained.

## B.7.1 REVIEWER PROMPT

The reviewer uses the following REVIEW INSTRUCTION verbatim:

Independently audit one proposed teaching action. You receive   
the student's pre-action history, the action, its actual execution result (if a   
command), and the PRIVATE human label. Do not approve merely because the label   
matches. Check that the action advances technical/file verification, uses the   
right code version, does not leak the answer key, and makes no factual claim that   
required a future observation. References must actually support the claim, not   
just occur somewhere in output. A no-match search can be a valid action. A final   
judgment needs evidence, not an echoed label. Return needs\_review if uncertain;   
reject fabricated evidence, unsupported final judgments or invalid operations.   
The action should be useful, but need not match the exact command you would use.

At request time, the implementation appends the same output-format instructions and complete <student protocol> used for the teacher, so the reviewer receives the student’s judgment criteria and tool protocol.

## B.7.2 INPUT AND OUTPUT FORMATS

Table 11 summarizes the four reviewer-input fields.

Table 11: Fields supplied to the reviewer.
<table><tr><td>Field</td><td>Content</td></tr><tr><td>teacher_private_label</td><td>Human label for the current training task.</td></tr><tr><td>student_history</td><td>Student-visible history before the teacher action</td></tr><tr><td>action</td><td>Normalized teacher action, represented as an assistant message with one bash tool call.</td></tr><tr><td>execution</td><td>Actual returncode and output produced by executing the action.</td></tr></table>

The reviewer returns strict JSON containing only decision and reason:

```json
{
"decision": "approve",
"reason": "Explanation of the review decision."
}
```

Table 12: Reviewer decisions and their effects on the training pipeline.
<table><tr><td>Decision</td><td>Meaning</td><td>Effect</td></tr><tr><td>approve</td><td>ing example.</td><td>The action is suitable as a teach- The record is marked approved and becomes eli- gible for inclusion in the training set.</td></tr><tr><td>reject</td><td>The action is invalid, unsupported by its evidence, or otherwise un- tion branch terminates. suitable.</td><td>The action is not exported, and the current correc-</td></tr></table>

For the “Unused import” example, the reviewer approves the patch-inspection action:

```json
{
"decision": "approve",
"reason": "The action correctly distinguishes base from patched evidence and inspects the
relevant patch section. Its output shows the reviewed import and the rest of the file's,→
changes, allowing verification that `router` is not referenced. The command is valid,,→
,→ focused, useful, and makes no unsupported final judgment."
}
```

Reviewer requests reuse the teacher’s model and request-level configuration in Appendix B.6; no separate reviewer model configuration is specified.

## C ACTION-LEVEL DISTILLATION AND TRAINING

## C.1 TRAINING RECORD FORMAT

Each action-level supervised fine-tuning record contains three top-level fields: prompt, completion, and tools. The prompt contains the student instructions followed by its previously generated actions and actual tool observations. The completion contains exactly one reviewer-approved teacher action, represented as an assistant message with one bash tool call and an optional concise explanation. The tools field supplies the bash function schema used by both the prompt history and the target action.

The following abbreviated record uses the Unused import example. Long task, patch, and file observations that already appear in Appendix B.4 are replaced by explicit paper-only omission notices; the exported training record retains the full messages.

```jsonl
{
"prompt": [
{
"role": "system",
"content": "[Full student system prompt from Appendix A.1.1.]"
},
{
"role": "user",
"content": "Inspect /testbed/task/task.json and evaluate its target comment."
},
{
"role": "assistant",
"content": "I need to first examine the task file to understand what I'm evaluating.",
"tool_calls": [
{
"type": "function",
"id": "call_d2dc7d3ee7282ed7923e",
"function": {
"name": "bash",
"arguments": {
"command": "cat /testbed/task/task.json"
}
}
]
},
{
"role": "tool",
"content": "[message_index=3]\nreturncode: 0\n<output>\n[Full task JSON observation
,→ omitted for presentation.]\n</output>",
"tool_call_id": "call_d2dc7d3ee7282ed7923e"
},
{
"role": "assistant",
"content": "Now I need to check the actual patch file to see what changes were made and
,→ verify if the comment about unused import is correct.",
"tool_calls": [
{
"type": "function",
"id": "call_05af99c1e55acfbfe63b",
"function": {
"name": "bash",
"arguments": {
"command": "cat /testbed/task/review.patch"
}
},
{
"role": "tool",
"content": "[message_index=5]\nreturncode: 0\n<output>\n[Long patch observation
,→ truncated by the tool and omitted here.]\n</output>",
"tool_call_id": "call_05af99c1e55acfbfe63b"
},
{
"role": "assistant",
"content": "[Messages 6--14 contain the student's file search, base-file inspection, git
,→ checks, and final focused search.]"
},
{
"role": "tool",
"content": "[message_index=15]\nreturncode: 1\n<output>\n\n</output>",
```

```jsonl
"tool_call_id": "call_01e89793b0d3d9b021a7"
}
],
"completion": [
{
"role": "assistant",
"content": "Inspect the focused patch section for the reviewed file. The repository is
at the base commit and the patch is not applied, so the import's absence from the,→
base file does not establish whether it is unused in the reviewed version.",,→
"tool_calls": [
{
"type": "function",
"id": "call_0ecd1a0501cbde6a4fdf",
"function": {
"name": "bash",
"arguments": {
"command": "grep -n -A120 -B5 'ˆdiff --git a/frontend/test-utils\\.tsx'
,→ /testbed/task/review.patch"
}
}
}
]
}
],
"tools": [
{
"type": "function",
"function": {
"name": "bash",
"description": "Execute a shell command in the isolated task container.",
"parameters": {
"type": "object",
"properties": {
"command": {
"type": "string"
}
},
"required": ["command"],
"additionalProperties": false
}
}
}
}
```

The prompt is used only as conditioning context. Completion-only loss is applied to the single teacher action in completion; student-history tokens and subsequent tool observations do not contribute to the objective.

## C.2 TRAINING-DATA SCALE

Table 13 reports the action-level datasets used in the two iterative training rounds. Exported records are deduplicated by the tuple (task id, history, action).

Table 13: Action-level training data used in the two rounds.
<table><tr><td>Item</td><td>Round 1</td><td>Round 2</td></tr><tr><td>Action training examples</td><td>4,622</td><td>4,646</td></tr><tr><td>Training tasks represented</td><td>701</td><td>714</td></tr></table>

## C.3 OPTIMIZATION AND HARDWARE CONFIGURATION

We train only the LoRA adapter parameters and keep the base-model weights frozen. We tune the LoRA rank and alpha on the CRJudgeBench validation split, selecting 16 and 32, respectively, and keep these values fixed for final evaluation on the test split. Table 14 lists the training configuration used in both rounds.

Table 14: Action-level distillation training configuration.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Base model</td><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td></tr><tr><td>Updated parameters</td><td>LoRA parameters only; base-model weights are frozen.</td></tr><tr><td>LoRA target modules</td><td>q-proj, k-proj, v-proj, and o-proj</td></tr><tr><td>LoRA rank / alpha</td><td>16/32</td></tr><tr><td>LoRA dropout</td><td>0.05</td></tr><tr><td>Trainable parameters</td><td>13,369,344</td></tr><tr><td>Hardware</td><td>4× NVIDIA H100 80GB GPUs</td></tr><tr><td>Distributed training</td><td>Distributed data parallel (DDP)</td></tr><tr><td>PyTorch</td><td>2.14.0</td></tr><tr><td>Transformers</td><td>4.57.6</td></tr><tr><td>PEFT</td><td>0.18.0</td></tr><tr><td>Accelerate</td><td>1.12.0</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup ratio</td><td>0.05</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Epochs per round Checkpointing</td><td>3</td></tr><tr><td></td><td>Save every 50 optimizer steps and retain the three most recent check- points.</td></tr><tr><td>Final artifact</td><td>Save the final LoRA adapter separately in adapter/.</td></tr></table>