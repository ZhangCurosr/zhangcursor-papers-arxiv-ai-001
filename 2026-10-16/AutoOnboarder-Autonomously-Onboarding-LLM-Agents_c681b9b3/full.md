# AutoOnboarder: Autonomously Onboarding LLM Agents

Tanay Arora Research Science Institute, Massachusetts Institute of Technology CliGenAILab, Boston Children’s Hospital

Cong Liu CliGenAILab, Boston Children’s Hospital

August 2026

## Abstract

Agentic large language models have been documented to lose reasoning capability when placed inside a harness, the software that mediates their observations and actions. To reconcile these losses, we introduce AutoOnboarder, an autonomous system that adapts an agent to a task the way a trainer onboards a new employee. It places the agent under the supervision of an intelligent “Onboarder” coding agent, which iteratively tailors its foundation model, prompt, and harness source code to simultaneously improve performance and decrease cost. Like a real onboarding process, the Onboarder is only given tens of examples, and all evaluation data is isolated on a separate virtual machine to prevent leakage. We report the lessons learned from building and testing AutoOnboarder on genomic-sequencing prior authorization form submission. Autonomously onboarded agents are able to match or exceed the performance of a Gemini 3.5 Flash baseline at up to 29× lower costs. However, these agents remain imperfect, and the Onboarding process varies greatly in terms of final performance. We conclude that future Onboarding systems require a robust environment for the Onboarder, including a structured thought interface, (controlled) internet access, and a wide model search space.

## 1 Introduction

## 1.1 The Harness Bottleneck

In today’s world, large language models (LLMs) are increasingly used as agents. Rather than simply answering questions, they autonomously complete tasks by planning and interacting with tools. This interaction is supported by a harness: a program that supplies a model with its context, mediates its access to tools, interprets its outputs as actions, and decides when a task is finished.

Unfortunately, these harnesses are also liabilities. An agent spends a significant part of its limited context interacting with its harness, and that overhead competes with the model’s reasoning [1, 2, 3].

To illustrate this flaw, Nie et al. design a sandbox – a simulation of a real task environment – for whole-exome and whole-genome sequencing prior-authorization form submission [4]. In their work, when frontier models were asked directly whether an artificial patient record contained an error, they flagged errors correctly 91% of the time. The same models, when placed under a browser-use harness inside of their sandbox, nearly universally failed to recognize errors (< 0.5%) and instead submitted deficient forms [4]. The act of interacting with a harness correlated with worse critical thinking capability for these agents. This work focuses on developing a method to automatically adapt an agent and its harness to a task, with the aim of recovering lost reasoning capability.

<table><tr><td></td><td colspan="4">May change</td><td rowspan="2">Cost objective</td></tr><tr><td>Family</td><td></td><td></td><td>Weights Prompt Model</td><td>1 Harness</td></tr><tr><td>Finetuning</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Prompt-adaptation</td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>Interface design</td><td></td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>Agents improving agents</td><td></td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>Onboarding</td><td></td><td>√</td><td>√</td><td>V</td><td>V</td></tr></table>

Table 1: Scopes of agent-adaptation families.

## 1.2 Related Work

Here, adapting an agent means optimizing it to perform best on a target task. Current methods for adapting agents can be divided into four main families (Table 1).

The first family updates model weights directly: finetuning and Low-Rank Adaptation (LoRA) [5] tune model parameters on labeled task data and reinforcement learning approaches tune them based on policy gradients or value-based approaches. The methods in this family require thousands of examples or expensive trajectory sampling, and neither change what the deployed agent has access to during inference.

The second family adapts the prompt while the model and harness remain fixed. Search-based optimizers generate and select candidate instructions (APE [6]), feed previously scored solutions back to an optimizer LLM (OPRO [7]), or edit prompts along natural-language “gradients” (ProTeGi [8], DSPy [9], TextGrad [10]). An experiential branch distills a few of the agent’s own episodes into lessons: self-reflections (Reflexion [11], ExpeL [12], Learn-by-Interact [13]), environment rule manuals (AutoManual [14]), and state-conditioned guidelines (AutoGuide [15]). These approaches adapt agents with only a few examples, but everything outside the prompt and memory remains fixed.

The third family redesigns the harness itself: SWE-agent’s purpose-built agent-computer interface substantially improves software-engineering agents [16], and AgentOccam gains similarly on web tasks by aligning observation and action spaces with the underlying LLM [17]. However, these works redesign interfaces by hand rather than autonomously, which can make adaptation tedious.

The most similar work to ours lets agents improve themselves or other agents. In ADAS, a meta-agent programs new agents in code, including edits to prompts and tools, and archives the best designs [18]; the Darwin Gödel Machine iteratively rewrites its own codebase, keeping changes that improve coding benchmarks [19]. Both optimize a large benchmark score over hundreds of evaluations and iterations, and neither carries an explicit cost objective for the agents it produces.

None of these families matches how a capable human is onboarded onto a task. When a new employee enters a job, they are often assigned to an onboarding course where a trainer walks them through a few representative cases, the varying quirks of the tools, and frameworks on how to think about the task. This process of onboarding uses only tens or hundreds of examples, and adjusts everything the worker can see (as opposed to only the foundation model weights or only the prompt). To our knowledge, no method today fully plays this role for an agent, especially while simultaneously reducing costs.

## 1.3 AutoOnboarder

In this work, we develop AutoOnboarder : an autonomous, agentic system for “onboarding” other agents. The system pairs two agents: an Executor, the agent being adapted to a task, and an

Algorithm 1 The Onboarding loop   
Require: Onboarding set ${ \mathcal { D } } ,$ cost goal $\beta ,$ iteration count T, initial config. c<sub>0</sub>   
1: for $t = 1 , \dots , T$ do   
2: implement one “idea” to one lever of $c _ { t - 1 }$ , giving $c ^ { \prime }$   
3: run the Executor with $c ^ { \prime }$ on all of $\mathcal { D } ;$ record logs and per-case cost   
4: send episodes to the Trainer; receive scores and per-case feedback   
5: $c _ { t } \gets c ^ { \prime }$ or $c _ { t - 1 }$ , at the Onboarder’s judgement   
6: end for   
7: return one kept configuration, nominated for validation

Onboarder, an intelligent coding agent that adapts it. Given a small Onboarding set (only tens of cases), the Onboarder optimizes the Executor and its harness for both cost and performance. To best emulate an onboarding environment, where an employee is meticulously walked through a few representative cases, we give the Onboarder access to virtually everything the Executor sees, while hiding all evaluation (“real-world”) data on a separate virtual machine to avoid data leakage. We hypothesized that Onboarding could recover the critical thinking capability documented by Nie et al. [4], while simultaneously improving cost-efectiveness. Our results are mixed, and we report them as lessons learned rather than as a finished method. In this paper, we contribute (i) the AutoOnboarder system and its two-machine, leakage-controlled evaluation protocol; (ii) a reliability study of the evaluation itself, covering judge agreement and breaking down sources of variance; (iii) an evaluation of the efectiveness of including the harness in Onboarding; (iv) an observational study of generalization to unseen error classes; and (v) practical lessons for building Onboarding systems, drawn from our experimental logs. Because task evaluation is expensive and time-consuming, each experiment is run once and repeated at most three times; throughout this paper, we report results as directional evidence and avoid statistical tests.

## 2 Methodology

## 2.1 The Onboarding Loop

The Onboarder is given a set number of iterations and a per-case cost goal, $\beta ,$ and must produce the best Executor candidate that it can near the goal. For each iteration, the Onboarder is instructed to implement a “single, identifiable idea” in exactly one of its three levers (defined below). We intentionally keep this instruction vague and let the Onboarder interpret it however it likes, so its changes are precisely what it deems necessary. We never provide it with examples of changes it can make, so it must devise all changes by its own rationale. The three levers that the Onboarder has access to include:

• Prompt: the Executor’s task-prompt template, which is wrapped by the harness before being fed into the model. An “idea” here can be any edit to this template.

• Foundation Model: any OpenRouter model identifier. OpenRouter is a unified router to hundreds of diferent LLMs from various providers by name, so an “idea” here likely swaps one model identifier to another (e.g., ‘google/gemini-3.5-flash’ to ‘openai/gpt-5.6-luna’). The Onboarder is provided with a helper script to inspect afordable options.

• Harness: the Executor’s harness source code. For our experiments, the harness refers to the browser-use harness [20], which is installed in the Onboarder’s local directory. The Onboarder can edit anything in this harness (e.g., history logic, system prompts, action definitions). An “idea” here might range from cutting the action registry to programmatically scraping patient records directly into the Executor’s persistent memory.

![](images/f9ca580b4849503adb1039b14d2d073d198e3322fff73b1f7d7f5426c10a1ff7.jpg)  
Figure 1: Heavily simplified visualization of the Executor’s evaluation task.

In each iteration, after making an isolated change, the Onboarder reevaluates the Executor on the Onboarding data; examines its logs, score, and feedback; and chooses to keep or revert its change based on its own judgement (Algorithm 1). At the end of its iterations, the Onboarder is told to nominate exactly one iteration, whichever it determines to be its best, for evaluation. The Onboarder is also given a hard per-case cost ceiling, computed as 1.5β. If any iteration is above this ceiling when evaluated on the Onboarding data, it cannot be nominated for validation. Additionally, to test the Onboarder’s ability to reduce costs autonomously, the initial configuration of the Executor sits far above its cost ceiling.

## 2.2 Evaluation Task

We instantiate AutoOnboarder on whole-exome and whole-genome sequencing prior authorization, which our laboratory previously used as a sandbox [4]. Prior-authorization is an insurer-initiated check of medical necessity, and for sequencing it is a documented bottleneck to rare-disease diagnosis. Roughly one in five requests are denied [21] and physicians report losing about thirteen hours a week to the paperwork [22]. The sandbox pairs a simulated web physician portal with a synthetic electronic health record (EHR) system, and each dataset sample consists of exactly one patient profile in this system. Some patient profiles are deficient, seeded with irreparable errors that should trigger a halt rather than a submission (Appendix A). Given only a patient’s name, the Executor must log in, search for and read the patient’s EHR record, and either complete the form or halt and report any issues (Figure 1). Completing the form requires both transcribing information from EHR record fields and inferring entries from the attached clinical note.

Scoring is approached in a binary manner. A success corresponds to (i) halting on a deficient case and providing the correct reason or (ii) submitting a valid case with perfect form-field accuracy. Any other situation is considered a failure, whether it be halting with an incorrect reason, halting on a valid case, a technical failure, submitting a deficient case, or imperfect form-field accuracy on a valid submission. An Executor that takes 55 steps (LLM calls) without finishing is automatically stopped and the episode is reported as a technical failure. The score, S, of a set of cases is calculated as

$$
s ( \mathrm { c a s e } ) = \left\{ \begin{array} { l l } { 0 } & { f a i l u r e } \\ { 1 } & { s u c c e s s } \end{array} \right. , \qquad \quad S = \frac { 1 } { 2 } \bar { s } _ { \mathrm { \scriptsize ~ v a l i d } } + \frac { 1 } { 2 } \bar { s } _ { \mathrm { \scriptsize ~ d e f i c i e n t } } ,\tag{1}
$$

![](images/77f90af5758a670b37d5ef27c97d78995270a6082bae652843fc57fc8f33cb23.jpg)  
Figure 2: The two-machine architecture.

where $\bar { s } _ { \mathrm { c l a s s } }$ is the mean over that class; the score weights the valid and deficient classes equally. A small LLM is tasked with deciding whether a reason provided is correct or incorrect by comparing it to the case’s ground truth issue description. We also define MFA as the mean form-field accuracy over all valid cases – where valid cases that are not submitted contribute 0% – and CH as the percentage of deficient cases correctly halted.

For this study, we make the underlying dataset of the Hard-to-Halt study [4] more dificult by expanding it from five fixed error classes to 23, spread over three categories: defects in to-betranscribed EHR records, defects in clinical notes that make the requested sequencing illogical, and cross-field contradictions between the clinical note and other EHR record fields. The validation set holds 180 cases, and Onboarding sets are sampled from its distribution. Exact class counts are listed in Appendix A; there are 141 (78.3%) deficient cases and 39 (21.7%) valid cases. In any given experiment, we use a dataset of 200 total cases in a 10/90 split (20 Onboarding, 180 validation). Because the expanded number of error classes is greater than the number of Onboarding samples, this dataset design forces the Onboarder to avoid overfitting the Executor to the Onboarding data.

## 2.3 Information Boundaries

We split the system across two virtual machines to enforce barriers on the Onboarder and prevent information leakage (Figure 2).

Virtual Machine 1 hosts the agents: an Onboarder and an Executor. The Onboarder is a Coding Agent (e.g., Claude Code, OpenAI Codex) that has full control of the three levers, direct JSON-level access to sanitized copies of the Onboarding cases (stripped of all error class information), the Executor’s step-by-step logs, and the Trainer’s feedback. Like a real onboarding trainer, the Onboarder has access to a great deal of information, and it is up to its own reasoning to extract this information without overfitting the Executor. The Onboarder is also given a helper script to inspect the content of the sandbox website.

Virtual Machine 2 holds all ground truth data and exposes two API endpoints: a Trainer and a Validator. The Trainer sends and scores Onboarding batches exclusively and returns a softened version of the score and per-case textual feedback. This softened score, $S _ { \mathrm { s o f t } }$ , computes $s _ { \mathrm { v a l i d } } = \mathrm { M F A }$ rather than $s _ { \mathrm { v a l i d } } =$ (fraction of successes). This design is used because it provides a more clear heuristic for success to the Onboarder; the Onboarder is still told that validation will be binary. The Trainer never reveals the error classes of the dataset (Appendix A), but it does indicate whether a provided reason is correct in its feedback. On the other hand, the Validator sends and scores validation batches exclusively, is called only by the human operator, and returns nothing but the numerical score; per-case feedback stays on VM2.

To prevent the Onboarder from gaming the task and examining hidden validation data, many structural boundaries are in place. It has no internet access, and all communication across machines is in the form of token-authenticated local API calls. The Executor runs each browser-use episode in a locally-staged sandbox that signs all submissions with HMAC, so the Onboarder cannot fabricate submissions. Filesystem permissions restrict the Onboarder’s write access to its three levers and logging files, keep instruction files read-only, and deny anything that would reveal hidden paths or generation code (the data creation pipeline, the staged app). A small set of less severe rules (e.g., never trigger validation scoring, never bypass cost restrictions, never view other experiments’ git branches) is enforced implicitly by a written “contract” presented to the Onboarder, with compliance verified manually post-hoc.

## 3 Experiments

## 3.1 Experimental Configuration

In our experiments, we use Claude Code running Opus 4.8 at max efort [23] as the Onboarder, with a budget of 12 iterations. We use OpenAI’s gpt-oss-120B as the foundation model for the judge that determines correctness of responses. Per-case cost is calculated via the USD charge that OpenRouter reports for the episode’s API calls. Every run starts from a large baseline: Gemini 3.5 Flash with the prompt from Nie et al. [4] and an unmodified harness (\$0.94 per case). We run Onboarding at three per-case cost goals, $\beta \in \{ \mathfrak { H } . 1 , \mathfrak { H } . 0 4 , \mathfrak { H } . 0 1 \}$ . These per-case cost goals are chosen primarily due to cost constraints, though pilot experiments found candidates between \$0.5 and \$0.03 per-case cost. All Onboarding runs were performed in late July and early August, and model identifiers refer to OpenRouter’s catalog at that time.

## 3.2 Measurement Reliability

## 3.2.1 Judge Reliability

The scoring in Equation (1) relies on an LLM judge (gpt-oss-120B at temperature 0.0) to determine whether the reasons for halting are correct or incorrect. To test its reliability, we sample 184 halted episodes that provided a reason: 8 per error class (Appendix A), with 4 coming from baselines and 4 coming from varying Onboarded candidates. We labeled every sampled reason as incorrect or correct independently of the judge, and across all 184 episodes the two sets of verdicts were identical. Furthermore, every single halting reason was stated clearly and verbosely, indicating that the Onboarder is not simply finding a way to game the judge’s verdict.

## 3.2.2 Noise Estimation

In addition, we test how noisy the experiments are and dissect variance into two sources. First, to measure evaluation noise, we perform a single Onboarding run for each cost goal (Section 3.1), and validate each resulting candidate three times. Then, to measure Onboarding noise, we repeat the entire Onboarding run at $\beta = \mathfrak { F } 0 . 0 4$ three times, on an identical Onboarding dataset, and evaluate each resulting candidate once. The former measures the variance of the agentic evaluation task, whereas the latter measures the variance introduced by the Onboarding procedure in addition to that of the evaluation task. In Table 2, we report the spread in S, CH, MFA, and per-case cost for each noise estimation (Section 2.2). We also calculate the number of cases that flip outcome (success vs. failure) between at least two of the three repeats. The Onboarding run at the \$0.01 cost goal was unable to produce a candidate under its 1.5β cost ceiling, so it was not validated and is excluded from the results. In every case except for the second Onboarding repeat, no perfect form submissions occur, so their score equals exactly half of CH; the second Onboarding repeat saw 10/39 perfect form submissions.

<table><tr><td>Cost goal (β)</td><td>Metric</td><td>Run 1</td><td>Run 2</td><td>Run 3</td><td>Std. Dev.</td><td>Range</td><td>Flips</td></tr><tr><td colspan="8">Evaluation Noise</td></tr><tr><td rowspan="4">$0.10</td><td>S</td><td>0.309</td><td>0.294</td><td>0.298</td><td>0.007</td><td>0.014</td><td rowspan="4">19</td></tr><tr><td>MFA</td><td>0.423</td><td>0.487</td><td>0.455</td><td>0.032</td><td>0.064</td></tr><tr><td>CH</td><td>0.617</td><td>0.589</td><td>0.596</td><td>0.015</td><td>0.028</td></tr><tr><td>$/case S</td><td>0.086 0.323</td><td>0.083 0.340</td><td>0.083</td><td>0.002</td><td>0.004 0.018</td></tr><tr><td>$0.04</td><td>MFA CH $/case</td><td>0.406 0.645 0.069</td><td>0.376 0.681 0.066</td><td>0.326 0.564 0.652 0.067</td><td>0.009 0.101 0.019 0.001</td><td>0.188 0.035 0.002</td><td rowspan="2">24</td></tr><tr><td>$0.01</td><td></td><td></td><td>No legal candidate produced.</td><td></td><td></td></tr><tr><td colspan="8">Onboarding Noise</td></tr><tr><td></td><td>S</td><td>0.323</td><td>0.501</td><td>0.199</td><td>0.152</td><td>0.302</td><td rowspan="3">63</td></tr><tr><td>$0.04</td><td>MFA CH  $\$ 123,456$ </td><td>0.406 0.645 0.069</td><td>0.917 0.745 0.032</td><td>0.041 0.397 0.052</td><td>0.440 0.179 0.019</td><td>0.876 0.348 0.037</td></tr></table>

Table 2: Spread of validation metrics (Section 2.2) across runs. Top: three validation repeats at varying cost goals. Bottom: three onboarding repeats at the $\beta = \mathfrak { F } 0 . 0 4$ goal. Std. Dev. is the sample standard deviation calculated from the three repeats. Flips is the number of the 180 cases whose outcome (failure vs. success) is not constant across all three repeats. The Onboarding run at cost goal \$0.01 was unable to produce a cheap-enough candidate.

In the evaluation noise estimation, the produced candidates only vaguely match their cost goals: the $\beta = \mathfrak { H } 0 . 1 0$ run produces a candidate with a per-case cost of \$0.053 during Onboarding and evaluates at a per-case cost of $\$ 0.084\pm \$ 90.004$ , while the $\beta = \mathfrak { F } 0 . 0 4$ run produces a candidate with a per-case cost of \$0.054 during Onboarding and evaluates at a per-case cost of $\$ 0.068\pm \ S 0.002$ Notably, both candidates sit within the $\beta = \mathfrak { F } 0 . 0 4$ run’s $1 . 5 \beta = \mathfrak { H } 0 . 0 6$ ceiling during Onboarding, so we deem $\beta = \mathfrak { F } 0 . 0 4$ the most informative goal and use it in all remaining experiments. Across both cost goals, the sample standard deviation of the score stays below 0.01, though about 20 cases (19 and 24) flip outcome.

The Onboarding noise estimation is far greater than that of the evaluation noise: the sample standard deviation of S across Onboarding repeats (0.152) is roughly 20× its evaluation noise counterpart (0.007/0.009). Figure 3 depicts this divergence more clearly, plotting the Onboarding per-case cost, MFA, and CH of these three runs as a function of iteration. Notably, the cost varies wildly – in the validation results of Table 2, the sample standard deviation of per-case cost (\$0.019) is about half of the cost goal, \$0.04 – and Onboarding-time MFA ranges from 0% to 100%. Between the three Onboarding runs, there are 63 cases that flip outcomes.

![](images/85c8796d025cdf82c198db0883f56f6c403d7f893fb3f870768300c2ae551089.jpg)  
Figure 3: Onboarding learning curves, displaying Per-Case Cost (left), MFA (middle), and CH (right) with respect to Onboarding iteration. Filled circles correspond to the nominated iteration, crosses correspond to iterations that were reverted and discarded, and open circles correspond to iterations that were kept, but not nominated. All reported calculations are performed on the Onboarding data, rather than the validation data. Iteration 3 of Run 3 was stopped by the Onboarder before scoring, so it is not plotted.

## 3.2.3 Flips

In all of these noise estimates, the calculated variance in the score understates how many cases flip outcome. Here, we perform an observational study over all behavior flips: flips in any of the Executor’s behavior. We classify these behavior flips into two categories: ones where only the correctness of a response changes (i.e., going from incorrect halting to correct halting or imperfect submission to perfect submission) and ones where the decision also changes (i.e., going from halting on a valid case to imperfect/perfect submission or submitting a deficient case to incorrect/correct halting). In total, there are 70 and 77 behavior flips in the $\beta = \mathfrak { H } 0 . 1 0$ and $\beta = \mathfrak { H } 0 . 0 4$ evaluation repeats, respectively, and 123 in the Onboarding repeats. Figure 4 plots the percentage of flips by sample class (Appendix A). Seemingly, valid cases and cross-field contradictions induce the most prominent amount of noise. Moreover, on valid cases every flip is a submit/halt decision change – the three Onboarding repeats submit 18, 37, and 2 of the 39 valid cases – so the large MFA spread in Table 2 largely reflects whether a candidate submits forms at all, as opposed to how accurately it fills out forms.

## 3.3 Comparison Across Lever Sets

To measure the impact of incorporating the model and harness into Onboarding, we compare runs that have access to the prompt+model+harness (PMH), only the prompt+model (PM), and only the prompt (P). Because it is infeasible to reduce costs from the baseline Gemini 3.5 Flash candidate solely through edits to the prompt, we initialize the P run from GPT-5.6 Luna, as it achieves a per-case cost of only \$0.027 with the baseline configuration. We compare these three runs against a collection of baseline models, including Gemini 3.5 Flash and smaller models such as GPT-5.6 Luna, GPT-4.1 Nano, and Gemini 3.1 Flash-Lite.

![](images/4a86064fd87377156ba3d6db02181fcb3bcd571c3eb5c184799f12a42184061d.jpg)

Figure 4: Decision flips of Onboarding runs by sample class type (Appendix A). Left (A): Flips in evaluation noise repeats, top bars correspond to $\beta = \$ 0.10$ and bottom bars correspond to $\beta = \mathfrak { F } 0 . 0 4$ Right (B): Flips in Onboarding noise repeats.
<table><tr><td>Candidate</td><td>S</td><td>MFA</td><td>CH</td><td>Perfect subs. (/39)</td><td>Near-perfect subs. (/39)</td><td>$/case</td></tr><tr><td>Gemini 3.5 Flash (stock)</td><td>0.319</td><td>0.024</td><td>0.638</td><td>0</td><td>0</td><td>0.934</td></tr><tr><td>GPT-5.6 Luna (stock)</td><td>0.028</td><td>0.000</td><td>0.057</td><td>0</td><td>0</td><td>0.027</td></tr><tr><td>Gemini 3.1 Flash-Lite (stock)</td><td>0.000</td><td>0.393</td><td>0.000</td><td>0</td><td>0</td><td>0.041</td></tr><tr><td>GPT-4.1 Nano (stock)</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0</td><td>0</td><td>0.042</td></tr><tr><td>PMH</td><td>0.323</td><td>0.406</td><td>0.645</td><td>0</td><td>2</td><td>0.069</td></tr><tr><td>PM</td><td>0.325</td><td>0.773</td><td>0.624</td><td>1</td><td>7</td><td>0.036</td></tr><tr><td>P (cheap init.)</td><td>0.409</td><td>0.657</td><td>0.766</td><td>2</td><td>9</td><td>0.016</td></tr></table>

Table 3: Validation results (Section 2.2) of the Onboarded candidates at $\beta = \mathfrak { F } 0 . 0 4$ and the stock baselines. Near-perfect submissions have exactly 36 of 37 fields correct.

All three Onboarded candidates match or exceed Gemini 3.5 Flash’s score at a small fraction of its cost (Table 3). Furthermore, they improve its MFA from 0.024 to 0.406–0.773. Other cheap baselines perform far worse, with Gemini 3.1 Flash-Lite and GPT-4.1 Nano scoring 0.0 and GPT-5.6 Luna achieving a score of only 0.028.

However, due to the sizable variance in the Onboarding procedure and low sample size, we believe that comparisons between lever configurations (i.e., PMH vs. PM vs. P) cannot be interpreted in a meaningful context. While the P candidate (score 0.409) seemingly performs substantially stronger than the PM (0.325) and PMH (0.323) candidates, it performs worse than the second PMH repeat in Table 2 (0.501) and far better than the third one (0.199).

## 3.4 Generalization Analysis

Finally, we perform an observational study of generalization to unseen error classes, splitting the error classes in the validation set (Appendix A) into three coverage groups: (i) seen, which are represented in the Onboarding dataset; (ii) near transfer, which have a near sibling represented; and (iii) far transfer, which have no sibling represented. There are 29 near transfer cases, 33 far transfer cases, and 79 seen-class cases. We plot correct and incorrect halting among these three coverage groups, along with halting percentage on valid cases, among varying Onboarded candidates at the $\beta = \mathfrak { F } 0 . 0 4$ cost goal and the baselines Gemini 3.5 Flash and GPT-5.6 Luna. Figure 5 shows the results for the three Onboarding repeats of Section 3.2 and the PM and P Onboarding runs of Section 3.3.

![](images/3cf51e5ea6c77e572a93525e428989fccdd54881ee01f6c1fd019269762d77d3.jpg)  
Figure 5: Halting behavior by coverage group (Section 3.4). For each stacked bar, the filled segment is the share halted for the correct reason, and the gray extension adds halts with an incorrect or technical reason, so the full bar is total halting. The separated bar in red reports halts on the 39 valid cases, which should have been submitted.

While the Onboarding process can seemingly reduce the number of false-halts substantially (from 97.4% of valid cases halted by the baseline Gemini 3.5 Flash to 53.8% for the first PMH repeat), we are again unable to draw a meaningful conclusion about clean transfer from the data. By inspection, it seems as though much of the diference in success between error classes is inherent to case dificulty, as both Onboarded candidates and baselines swing widely across coverage groups.

## 4 Discussion

We close with the lessons we drew from building and testing AutoOnboarder. Beyond the validation results above, they draw on the Onboarder’s per-iteration ledgers and session logs.

Onboarding clearly shows promise. Current experiments testify to the potential of AutoOnboarder: the best candidate exceeds the Gemini 3.5 Flash baseline in performance at a 29× lower cost and submits far more valid cases. Our design allows the Onboarder agent to efectively edit every component that shapes an Executor’s behavior, while simultaneously reducing costs. The leakage-controlled two-machine setup is designed to prevent all data leakage, and post-hoc searches of Onboarder session logs and API logs found no attempts to bypass data protection.

However, Onboarding exhibits excessively large variance. Experiments are incredibly noisy – repeat runs’ scores range from 0.199 to 0.501, per-case cost from \$0.032 to \$0.069, and valid submissions from 2 to 37 (of 39). Much of the Onboarding spread seems to trace back to Onboarder’s environment and search space; the next three lessons take up each of these in turn.

The model lever needs a curated search space. At the cost scales we attempt to cut to, OpenRouter models are often unreliable. Many afordable models on OpenRouter, such as Mistral Small, Llama 4 Scout, and Gemini 2.5 Flash, are unable to conform to browser-use’s output schema; others, like Qwen3 VL 235B, fall back to providers that lack a required feature like image input or are heavily rate limited (HTTP 429). Across six runs with access to the model lever, 24 iterations made changes to the foundation model, and 15 of them ended as crashes or discards. Each dead end consumes a full pass over the Onboarding set and one of only 12 iterations, which contributed significantly to divergence between Onboarding runs. Of the three Onboarding repeats at $\beta = \mathfrak { F } 0 . 0 4$ failed model swaps accounted for the early discarded iterations in the two non-optimal runs. We therefore believe a curated, wider model catalog is a necessity for Onboarding. Furthermore, crafting search spaces that incorporate distilled, quantized, or otherwise compressed versions of the same foundation LLMs would likely improve onboarding quality by granting more fine-grained control in the model lever.

The Onboarder needs a structured interface for thought. In our testing, we notice that AutoOnboarder tends to be quite sensitive and unpredictable in regard to the wording of its contract. Many behavioral instructions are completely ignored or taken to extremes. It often picks models by name recognition and ignores the benchmark data from its helper script, declares its task complete or impossible with iterations left unspent, and only loosely meets cost goals (for example, the β = \$0.10 run ended Onboarding with a per-case cost of \$0.053). Furthermore, this obstinate behavior sometimes leads the Onboarder to give up on form submission entirely, and we believe it may be the root of a significant portion of the Onboarding variance.

Another failure mode we noticed in our experimentation is that the harness lever is used sparingly. In all of our experiments, the Onboarder makes tweaks to the harness lever in at most two iterations, even after we explicitly instruct it to use all of its levers. Examining reasoning traces of the Onboarder does not provide a clear picture of why this is the case; we read this as a limitation of our implementation rather than of the value of edits to the harness. Unlike meta-agents such as ADAS [18] and the Darwin Gödel Machine [19], which are given hundreds of iterations to rewrite their harnesses, AutoOnboarder is only given 12. With so few iterations and so strict a cost budget, it is possible that the Onboarder simply did not have enough room to experiment with the harness lever suficiently.

Together, these observations suggest that the Onboarder’s free-form, contract-driven interface is itself a bottleneck. Reconciling these issues likely requires a more robust interface for the Onboarder that provides it more tools and forces it to consolidate its reasoning, much like that designed for Executor agents in SWE-Agent [16] and AgentOccam [17]. Such an interface would enforce the Onboarder’s rules rather than leave them open to its interpretation of a written contract.

The Onboarder requires controlled, not absent, internet access. To keep the system secure, the Onboarder has no network access. This isolation, though, also limits what the Onboarder can teach. Perfect submissions require exact ICD-10 codes (a clinical diagnosis standard) drawn from a relatively wide pool. Without access to the internet, the Onboarder often explicitly acknowledges that it has no way to package an ICD database for the Executor. We therefore believe that much of the residual imperfection may be a missing-tooling problem, rather than one baked into the prompt, model, or harness. However, unrestricted internet access would make it easy for an Onboarder to reward hack, for example by seeking out answers to specific cases instead of teaching general skills. We therefore argue for controlled access: an allowlisted, logged, read-only channel, such as a proxy or curated mirrors of resources like ICD-10 tables and model and provider documentation, that sits inside the existing boundaries of HMAC-signed submissions, filesystem permissions, and validation data isolated on VM2.

The primary operational limitations of this study stem from the time and cost of the experiments, which reduce the number of repeat experiments we can perform: a single case can take over 30 minutes, and evaluating a model like Gemini 3.5 Flash on the 200-case set takes around \$190.

Interestingly, in our experiments, Gemini 3.5 Flash submits only 6 of 180 cases and wrongly withholds 38 of the 39 valid ones, which conflicts with the automation bias claim of Nie et al. [4]. We hypothesize that the diferences lie primarily in closed-model drift, whereby commercial LLM providers silently change LLM behavior over time. This is also a salient focus in our study: as with any research on closed LLMs, it is possible that some findings in this work do not generalize to newer or tweaked models.

## 5 Conclusion

In this work, we set out to develop an autonomous system that would adapt an agent and its harness to a task, thereby recovering the losses in critical thinking skills documented by Nie et al. [4]. Current experiments testify to the potential of AutoOnboarder in reducing Executor agents’ costs while maintaining or improving their scores. However, current experiments are also prone to large variance and subpar performance.

In the future, we plan a more robust system where the Onboarder has an interface that forces more structured thinking and includes all the tools it needs, while also balancing this freedom with data security. We hypothesize that such a system would both reduce the large variance of Onboarding runs and contribute to better Executor performance.

## 6 Key Takeaways

In this work, we show that Onboarding has the potential to adapt AI agents to be more reliable and cost-efective, making them more practical in production. In the future, agentic systems like AutoOnboarder can drive the creation of small, cheap agents that take high-volume busy work, like prior authorization forms, of of human hands. However, this work also demonstrates the dificulties in creating such a system: the same Onboarding procedure, run twice, can produce very diferent agents, and no candidate is yet dependable enough for real clinical use. Moving forward, we need to develop a more repeatable, robust method for holistic agent-optimizing systems that prioritizes consolidated thinking and more closely balances security with access to information.

## Acknowledgements

We thank Mengshu Nie for building the prior-authorization sandbox that this work extends, and the rest of CliGenAILab, especially Dr. Kamalakkannan Ravi for his help in proposing this project. Finally, we thank the Center for Excellence in Education, the Research Science Institute, and the Massachusetts Institute of Technology for making this research possible.

## References

[1] M. Levy, A. Jacoby, et al. Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pages 15339–15353. ACL, 2024.

[2] F. Shi, X. Chen, et al. Large Language Models Can Be Easily Distracted by Irrelevant Context. In Proceedings of the 40th International Conference on Machine Learning, pages 31210–31227. PMLR, 2023.

[3] N. F. Liu, K. Lin, et al. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024.

[4] M. Nie, C. Liu, et al. Hard to Halt: Automation Bias in Agent-Driven Sequencing Prior Authorization Workflows. medRxiv, 2026.

[5] E. J. Hu, Y. Shen, et al. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations, 2022.

[6] Y. Zhou, A. I. Muresanu, et al. Large Language Models are Human-Level Prompt Engineers. In The Eleventh International Conference on Learning Representations, 2023.

[7] C. Yang, X. Wang, et al. Large Language Models as Optimizers. In The Twelfth International Conference on Learning Representations, 2024.

[8] R. Pryzant, D. Iter, et al. Automatic Prompt Optimization with “Gradient Descent” and Beam Search. In The 2023 Conference on Empirical Methods in Natural Language Processing, Dec. 2023.

[9] O. Khattab, A. Singhvi, et al. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. In The Twelfth International Conference on Learning Representations, 2024.

[10] M. Yuksekgonul, F. Bianchi, et al. Optimizing generative AI by backpropagating language model feedback. Nature, 639(8055):609–616, Mar. 2025.

[11] N. Shinn, F. Cassano, et al. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, Nov. 2023.

[12] A. Zhao, D. Huang, et al. ExpeL: LLM agents are experiential learners. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’24/IAAI’24/EAAI’24. AAAI Press, 2024.

[13] H. Su, R. Sun, et al. Learn-by-interact: A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments. In The Thirteenth International Conference on Learning Representations, 2025.

[14] M. Chen, Y. Li, et al. AutoManual: Generating Instruction Manuals by LLM Agents via Interactive Environmental Learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[15] Y. Fu, D.-K. Kim, et al. AutoGuide: Automated Generation and Selection of Context-Aware Guidelines for Large Language Model Agents. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[16] J. Yang, C. Jimenez, et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. In Advances in Neural Information Processing Systems, volume 37, pages 50528– 50652. Curran Associates, Inc., 2024.

[17] K. Yang, Y. Liu, et al. AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents. In The Thirteenth International Conference on Learning Representations, 2025.

[18] S. Hu, C. Lu, et al. Automated Design of Agentic Systems. In The Thirteenth International Conference on Learning Representations, 2025.

[19] J. Zhang, S. Hu, et al. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. In The Fourteenth International Conference on Learning Representations, 2026.

[20] M. Müller and G. Žunič. Browser Use: Enable AI to control your browser, 2024.

[21] H. S. Smith, R. Franciskovich, et al. Outcomes of prior authorization requests for genetic testing in outpatient pediatric genetics clinics. Genetics in Medicine: Oficial Journal of the American College of Medical Genetics, 23(5):950–955, 2021.

[22] American Medical Association. 2021 AMA Prior Authorization (PA) Physician Survey, 2022.

[23] Anthropic. Claude Code: an agentic coding tool, 2025.

## A Error Classes

Each case corresponds to a single name, which is passed to the Executor inside of its prompt. The dataset in this paper expands the dataset of [4] from five deficiency signatures to 23 (Section 2.2). Percentages below are shares of all 180 validation cases. Classes P and Dsec require submission; the 23 deficient classes require withholding and reporting the specific problem. Error classes can be split into data-integrity (T) issues in transcription fields, clinical necessity (C) issues in the clinical note, and cross-field issues (X) that have contradictions between transcribed fields and the clinical note. Clinical notes were produced with OpenAI’s GPT-5.6 Terra with no reasoning. We define siblings as classes in the same subgroup. For example, T3a and T3b are siblings, while T6 has no siblings.

<table><tr><td>Group</td><td>Class: cases (% of dataset)</td><td>Definition</td></tr><tr><td>Valid</td><td>P: 27 (15.0%)</td><td>Valid, well-supported request.</td></tr><tr><td>39 (21.7%)</td><td>Dsec: 12 (6.7%)</td><td>Valid request with unrelated secondary details; should still submit.</td></tr><tr><td>Data-integrity (T)</td><td>T1a: 4 (2.2%); T1b: 5 (2.8%)</td><td>Subscriber listed as a parent is implausibly close in age to the patient, or born after the patient.</td></tr><tr><td>53 (29.4%)</td><td>T2: 6 (3.3%)</td><td>Specimen collection date precedes the patient&#x27;s date of birth.</td></tr><tr><td></td><td>T3a: 6 (3.3%); T3b: 6 (3.3%)</td><td>Two patients share an identical name (and, for T3b, date of birth), making the case ambiguous.</td></tr><tr><td></td><td>T4a: 5 (2.8%); T4b: 5 (2.8%)</td><td>Provider or lab NPI, or provider phone or fax, is not valid.</td></tr><tr><td></td><td></td><td>T5a: 2 (1.1%); T5b: 2 (1.1%); Required element is missing: provider NPI, lab NPI, provider ad- T5c: 2 (1.1%); T5d: 2 (1.1%); dress, provider phone, subscriber date of birth, or collection date,</td></tr><tr><td></td><td>T5e: 2 (1.1%); T5f: 2 (1.1%) T6: 4 (2.2%)</td><td>respectively. Member ID is malformed.</td></tr><tr><td>Clinical necessity (C)</td><td>C1: 13 (7.2%)</td><td>Acquired or environmental issue with no heritable indication.</td></tr><tr><td></td><td>C2: 11 (6.1%)</td><td>Isolated, common, non-syndromic finding.</td></tr><tr><td>64 (35.6%)</td><td>C3: 11 (6.1%)</td><td>Screening or carrier indication in an asymptomatic patient.</td></tr><tr><td></td><td>C4: 11 (6.1%)</td><td>Causative diagnosis already made, so sequencing is redundant.</td></tr><tr><td></td><td>C5a: 9 (5.0%); C5b: 9 (5.0%)</td><td>Sequencing already performed, or prior targeted testing already diagnostic.</td></tr><tr><td>Cross-field (X)</td><td>X1: 8 (4.4%)</td><td>Recorded sex contradicts the sex described in the note.</td></tr><tr><td></td><td>X3: 8 (4.4%)</td><td>Collection date precedes the prior test described in the note.</td></tr><tr><td>24 (13.3%)</td><td>X4: 8 (4.4%)</td><td>Broad sequencing is ordered where a targeted single-gene test is</td></tr><tr><td>Total</td><td></td><td>the indicated next step.</td></tr></table>

Table 4: Validation-set sample classes and prevalence. The set contains 39 valid cases (21.7%) and 141 deficient cases (78.3%); every subclass has at least two cases.