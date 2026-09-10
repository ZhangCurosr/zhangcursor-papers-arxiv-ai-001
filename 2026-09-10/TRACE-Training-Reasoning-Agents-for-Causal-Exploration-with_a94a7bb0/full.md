# TRACE: Training Reasoning Agents for Causal Exploration with Synthesized Rewards

Rui Sun Zhan Shi Bing He Independent Researchers

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has advanced languagemodel reasoning in domains such as mathematics and code, where objective answers are inexpensive to check. Diagnostic reasoning over complex data lacks this advantage: establishing the true cause of an anomaly often requires costly expert investigation and may remain ambiguous after the fact. We ask whether this asymmetry of verification can instead be engineered. We sample an intervention, inject it into a controlled simulator, and generate the observations it would produce. The hidden intervention provides an oracle label and objective reward, while the agent must still investigate noisy, confounded, and distributed evidence.

We instantiate this approach in TRACE, a digital-advertising diagnostic environment with 12 root causes and fine-grained segment attribution. Agents investigate each episode using Python and SQL and must identify both the root cause and, when applicable, the affected segment assignment. On a held-out 235-episode test set, the strongest prompted baseline, Claude Opus 5, reaches 0.686 FullAttr@1. Supervised fine-tuning raises Qwen3.5-35B-A3B from 0.159 to 0.637, and subsequent RL with synthesized rewards reaches 0.757, outperforming all evaluated prompted baselines, including frontier closed-source models and a prompted Qwen3.5-122B-A10B model. The resulting policy also uses substantially fewer tool calls than the prompted 35B base. These results provide evidence that access to a scalable, objective training signal can be a more important constraint than model scale alone. More broadly, simulation-based verification can make otherwise ambiguous diagnostic reasoning tasks amenable to scalable reinforcement learning.

## 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has driven recent progress in LLM reasoning. Work on mathematical reasoning [Guo et al., 2025, Shao et al., 2024] and coding agents [Wei et al., 2026, Luo et al., 2025b] shows that policies optimized against objective verifiers can acquire multistep reasoning and tool-use capabilities. These domains provide a natural asymmetry of verification: checking a solution against a known answer or test suite is considerably easier than producing the solution itself [Wei, 2025]. RLVR turns this asymmetry into a scalable training signal.

Many practical reasoning tasks lack such verifiers. Diagnostic reasoning over complex data is a representative example: determining why a system changed can require expensive expert investigation, and the resulting attribution may remain uncertain even after the fact. Multiple changes can occur concurrently, observations are noisy, and plausible causes may produce similar signatures. LLMas-judge approaches [Zheng et al., 2023, Bai et al., 2022, Lee et al., 2024] do not eliminate this bottleneck because the target itself remains ambiguous, while learned judges can introduce additional opportunities for reward hacking [Gao et al., 2023, Skalse et al., 2022]. The central obstacle to applying RLVR in these settings is therefore access to scalable, objective ground truth.

In this paper, we ask whether the asymmetry of verification can be engineered. Rather than observing data and attempting to determine an unknown cause, we first sample an intervention, inject it into a controlled simulator, and generate the observations that the intervention would produce. The hidden intervention is retained as an oracle label, making the agent’s final attribution inexpensive and deterministic to verify. The agent does not observe this label: it must still investigate noisy, confounded, and distributed evidence to recover the cause. Simulation therefore separates the difficulty of solving a diagnostic task from the difficulty of verifying its answer.

We instantiate this approach in TRACE (Training Reasoning Agents for Causal Exploration), a simulated diagnostic environment for digital advertising. Each TRACE instance presents an agent with a campaign-performance anomaly and a multi-table database. The agent investigates through Python and SQL, iterating between hypotheses and evidence, and ultimately attributes the anomaly to one of the predefined cause types while ruling out realistic confounders. TRACE builds a simulator–oracle–RL pipeline: the simulator injects a hidden intervention, the oracle verifies that the agent-visible data contains sufficient evidence to recover it, and RL uses the resulting hidden label as a synthesized reward. We refer to this setting as RL with synthesized rewards, where hidden intervention labels are converted into objective reward signals.

We evaluate on a held-out 235-episode test set enriched for segment-specific attribution. The strongest prompted baseline, Claude Opus 5, reaches 0.686 FullAttr@1. On Qwen3.5-35B-A3B, SFT raises FullAttr@1 from 0.159 to 0.637, and subsequent RL with synthesized rewards improves it to 0.757, surpassing every evaluated prompted baseline. The post-trained 35B model also substantially outperforms the prompted Qwen3.5-122B-A10B model, which reaches 0.283. This result provides evidence that, for this diagnostic setting, the binding constraint is access to an effective post-training signal—including a scalable, objective reward—rather than model scale alone. The best trained policy also averages 11.73 tool calls per trajectory, compared with 22.05 for the prompted 35B base.

Our contributions are threefold. First, we introduce a simulator–oracle–RL methodology for synthesizing objective reward in domains where natural verifiers are scarce. The key idea is to generate tasks from controlled interventions, so that the same process that creates a difficult reasoning problem also provides an objective answer. Second, we instantiate this methodology in TRACE, a tool-using diagnostic environment with configurable causes, realistic confounders, and multi-table evidence. TRACE supports held-out evaluation and post-training without human attribution labels or an LLM judge. Third, we show that SFT followed by RL with TRACE’s synthesized rewards substantially improves an open-weight 35B reasoning agent, enabling it to outperform frontier closed-source models on diagnostic attribution tasks. We also conduct ablation experiments to study the roles of supervised initialization and RL reward design in producing these gains. Together, these results show how simulation-based verification can make otherwise ambiguous diagnostic reasoning tasks amenable to scalable reinforcement learning.

## 1.1 Related Work

RLVR and reasoning training. RLVR methods optimize policies against verifiable outcomes, including mathematical answer checkers [Shao et al., 2024, Guo et al., 2025], formal proof validators [Kim and Yun, 2026], and compilation or execution tests for code [Wei et al., 2026, Luo et al., 2025b]. Recent work has broadened RLVR by improving the underlying policy optimization algorithms [Shao et al., 2024, Yu et al., 2026, Liu et al., 2025, Zheng et al., 2025] and by extending the training recipe beyond math and code, for example to medical question answering [Chen et al., 2025]. However, these advances still assume that verifiable targets already exist. Our work instead studies how to construct such targets for domains where they are not naturally available.

Synthetic data, simulators, and reward synthesis. Synthetic generation has been widely used to create instruction-following data and reasoning trajectories for supervised fine-tuning [Wang et al., 2023, Luo et al., 2025a, Yu et al., 2024]. In these settings, the generator primarily supplies examples to imitate. A related line of work develops simulated and interactive environments for evaluating tool-using agents in science, web, and machine-learning tasks [Wang et al., 2022, Zhou et al., 2024, Huang et al., 2024]. These environments use interaction to test whether agents can complete tasks. TRACE uses simulation differently: it samples a hidden root cause, generates the diagnostic data it would produce, and uses the sampled cause as the oracle label for reward calculation.

Tool-using data-analysis and diagnostic agents. A growing line of work evaluates whether LLM agents can analyze structured data with external tools. Text-to-SQL benchmarks provide an early testbed for this setting by asking models to translate natural-language questions into executable database queries [Yu et al., 2018, Zhong et al., 2017, Li et al., 2023, Chang et al., 2023]. More recent data-analysis benchmarks move toward multi-step tool use over tables, files, and code execution [Huang et al., 2024, Zhang et al., 2026]. These tasks capture important components of diagnostic investigation, but they generally score query correctness, code outputs, or final reports rather than rootcause attribution under controlled confounders. TRACE targets this gap by constructing diagnostic tasks with simulated interventions, agent-visible evidence, and verifiable rewards.

Causal reasoning and root-cause attribution. A parallel line of work evaluates whether LLMs can answer causal questions, reason over causal graphs, or recover causal structure from data [Jin et al., 2023, 2024, Kiciman et al., 2024, Wang, 2024]. These benchmarks probe causal knowledge and formal causal reasoning, often through static questions, symbolic graphs, or fixed datasets. TRACE studies a different setting: interactive root-cause attribution in a simulated environment where the true cause is known by construction. The agent must gather evidence through SQL and Python, distinguish the sampled cause from realistic confounders, and produce an attribution that can be verified against ground-truth labels.

## 2 The TRACE Environment

This section describes TRACE, a digital advertising environment that implements the simulator– oracle components of our methodology. TRACE has three parts: a stochastic simulator that generates multi-table advertising data with known injected causes, a task interface through which an agent investigates each generated diagnostic instance, and an oracle verifier that certifies solvability and produces reward labels. Figure 1 summarizes the overall workflow. We refer to each generated diagnostic instance as an episode: the observable history of a single campaign over a fixed time window, together with a hidden injected intervention that defines the ground-truth cause.

![](images/7b10fb91f65775e6dee8380f02ebc96156e364e46ddbfceae844ed713b987f6d.jpg)  
Figure 1: Overview of the TRACE workflow.

## 2.1 Preliminaries: Campaign Metrics and Diagnosis

Digital advertising campaigns serve ads to users through online auctions: when a user loads a page with an ad slot, advertisers bid for the slot, and the winning ad is shown as an impression. Under the commonly adopted pay-per-click model, the advertiser is charged only when the user clicks, and a click that leads to a purchase of the advertised product is recorded as an order, or conversion. Campaign performance is usually summarized by metrics such as impressions, click-through rate (CTR, clicks/impressions), conversion rate (CVR, orders/clicks), and cost-per-click (CPC, total advertiser spend/clicks).<sup>1</sup> These metrics are reported at the campaign level and across segments: combinations of categorical attributes such as placement, device, geography or user group.

Segment-level reporting is what makes diagnosis both possible and difficult. The distribution of impressions across segments is shaped by interacting factors, including campaign configuration, competing campaigns, auction dynamics, and user behavior. These factors also produce ordinary day-to-day metric noise. Telling a real performance shift apart from this noise requires resolving three questions. First, the cause is ambiguous: one metric movement can have several explanations (what). Second, the cause is localized: a segment-level effect can be diluted in the campaign aggregate and surface only under the right split (where). Third, the timing is unknown: gradual onsets are easily missed by simple period comparisons (when). Resolving all three at once is the core reasoning pattern TRACE is designed to elicit.

## 2.2 Data Generation Pipeline

For each TRACE episode, the sampled intervention determines three hidden labels: the cause type, the affected segment slice, and the onset timing.

Generation proceeds in three stages: building a campaign population (Stage 1), sampling an interven tion (Stage 2), and simulating the resulting metrics (Stage 3).

Stage 1: Campaign population. We first construct a static universe of campaigns. Each is assigned a vertical category (e.g. electronics, fashion), a bidding strategy (e.g. search-heavy, acquisition), a baseline daily budget, and a baseline impression volume. A campaign is active in a sampled subset of its possible segments. Per-campaign variation in click-through and conversion rates is captured by a quality multiplier.

Stage 2: Intervention sampling. Each episode spans a contiguous window split into two equal parts: a baseline period, in which the campaign runs normally, and an intervention period, in which a sampled cause is active. We draw four components: the injected cause, one of the twelve predefined cause types (§2.3); the driver slice, which specifies the affected segments using one or two attributes, such as placement=TOP\_OF\_SEARCH or geo=US ∧ device=mobile; the signal strength, which sets the effect magnitude; and the onset profile, which determines whether the effect appears immediately, after a delay, or gradually over time.

Stage 3: Metric simulation. The simulator renders daily metrics in three steps. First, it draws the campaign’s total daily impressions from a baseline volume, a seasonality factor, and any volume change induced by active causes. Second, it distributes these impressions across segments. The base distribution reflects the bidding strategy: a search-heavy campaign, for instance, weights search placements more heavily. Active causes then reweight this distribution, shifting impressions between segments. Third, it computes CTR, CVR, and CPC per segment from base rates, adjusted by the campaign’s quality multiplier and by any cause effects applied to the driver slice. Clicks and orders are sampled from these rates, and spend follows from clicks and CPC.

Equation (1) summarizes the simulator. Active causes can affect the generated data through three channels: total impression volume, segment allocation, and per-segment rates.

$$
\begin{array} { r l } { \mathrm { i m p r e s s i o n s : } } & { N _ { i , t } \sim \mathrm { P o i s s o n } ( \mu _ { i } \eta _ { t } V _ { i , t } ) , } \\ { \mathrm { s e g m e n t ~ m i x : } } & { \pi _ { i , t } = \mathrm { N o r m a l i z e } ( \mathbf w _ { i } \odot \mathbf R _ { i , t } ) , } \\ & { \mathbf { N } _ { i , t } \sim \mathrm { D i r i c h l e t - M u l t i n o m i a l } ( N _ { i , t } , \alpha \pi _ { i , t } ) , } \\ { \mathrm { s e g m e n t ~ r a t e s : } } & { \theta _ { i , g , t } ^ { m } = \bar { \theta } _ { i , g } ^ { m } \cdot Q _ { i } ^ { m } \cdot F _ { i , g , t } ^ { m } , m \in \{ \mathrm { C T R , C V R , C P C } \} . } \end{array}\tag{1}
$$

Here i indexes campaigns, t indexes days, and $g$ indexes segments. The total impressions $N _ { i , t }$ are sampled from the campaign baseline volume $\mu _ { i }$ , a seasonality factor $\eta _ { t }$ , and an interventioninduced volume multiplier $\bar { V } _ { i , t }$ . Segment impressions $\mathbf { N } _ { i , t } = ( N _ { i , g , t } ) _ { g }$ are drawn from a multinomial distribution with probabilities $\pi _ { i , t }$ , obtained by reweighting the campaign’s baseline segment weights $\mathbf { w } _ { i }$ by an intervention-induced segment-mix multiplier $\mathbf { R } _ { i , t }$ . The concentration parameter α controls day-to-day variability in segment allocation. Finally, $\theta _ { i , g , t } ^ { m }$ denotes the segment-level rate for metric $m \in \{ \mathrm { C T R } , \mathrm { C V R } , \mathrm { C P C } \}$ , with baseline rate $\bar { \theta } _ { i , g } ^ { m } .$ , campaign quality multiplier $Q _ { i } ^ { m }$ , and interventioninduced rate multiplier $F _ { i , g , t } ^ { m }$

Table 1: Root-cause signatures, grouped by signal level.
<table><tr><td>Cause</td><td>Metric Signature</td><td>Driver Dimension</td></tr><tr><td>Campaign-wide</td><td></td><td></td></tr><tr><td>BID INCREASE</td><td>CPC↑, CTR↓ (mild)</td><td></td></tr><tr><td>BUDGET CAP</td><td>Impr↓, clicks↓, orders↓; rates≈</td><td></td></tr><tr><td>PAGE DEGRADATION</td><td>CVR↓, CTR≈, orders↓</td><td></td></tr><tr><td>OUT OF STOCK</td><td>stock↓, CVR↓, orders↓</td><td></td></tr><tr><td>Segment-mix</td><td></td><td></td></tr><tr><td>PLACEMENT SHIFT</td><td>segment share ↑</td><td>placement</td></tr><tr><td>TARGETING BROADENING</td><td>Impr↑, CTR↓, CVR↓</td><td>audience</td></tr><tr><td>TARGETING NARROWING</td><td>Impr↓, CTR↑, CVR↑</td><td>audience</td></tr><tr><td>Segment-specific</td><td></td><td></td></tr><tr><td>CREATIVE FATIGUE</td><td>CTR↓ (gradual), Impr≈, CVR≈</td><td>placement/ device/geo</td></tr><tr><td>COMPETITIVE PRESSURE</td><td>CPC↑, CTR↓</td><td>geo</td></tr><tr><td>AD QUALITY DROP</td><td>Impr↓, CPC↑, CTR↓</td><td>device</td></tr><tr><td>AUDIENCE SATURATION</td><td>CTR↓ (gradual), CVR↓ (mild)</td><td>audience</td></tr><tr><td>No signal</td><td></td><td></td></tr><tr><td>NO SIGNAL</td><td>all |∆| within noise</td><td></td></tr></table>

## 2.3 Cause Taxonomy

TRACE defines twelve cause types in four categories (Table 1), organized by where the causal signal appears. The categories require different diagnostic strategies, so no fixed query procedure is sufficient.

Campaign-wide causes produce effects visible in aggregate campaign metrics.

Segment-mix causes redistribute impressions across segments while leaving campaign totals roughly unchanged. Their signature is a change in traffic composition across periods.

Segment-specific causes alter rates inside a localized driver slice, leaving other segments unaffected. These effects can be diluted in aggregate metrics and surface only under the right split. This category is the hardest: the agent must both localize the driver slice and match the multi-metric signature that distinguishes one cause from another.

Finally, no-signal episodes inject no cause at all: every fluctuation is stochastic noise or a confounder. These episodes are not trivial because confounders, such as seasonality, can appear regardless of the cause. They test whether an agent abstains when evidence is insufficient, penalizing models that default to plausible-sounding attributions.

Complexity axes. Episode difficulty is controlled by two cause-conditioned axes. Dimensional complexity controls where: the signal’s driver slice may filter on one attribute or, in harder cases, the intersection of two. Temporal complexity controls when: the signal’s onset profiles can be immediate, delayed, or ramping. Together with the cause taxonomy, these axes test whether the agent can identify what changed and where the signal appears under different temporal patterns.

## 2.4 Database and Task Interface

For each episode, the agent receives query access to fact tables that contain daily metrics at the campaign and segment level. The ground-truth labels are stored in separate tables that are used only for verification, scoring and training reward, and they are never exposed to the agent. The full schema is given in Appendix A.

The agent receives a system prompt specifying its role, the candidate cause types, the available tables, and the required final-answer format. The user prompt describes the observed performance change for a specific campaign. The agent interacts through a Python tool backed by a sandboxed SQL engine, with state persisted across calls so that later analyses can build on earlier queries.

The agent’s final answer contains a target cause and supporting evidence. Segment-specific causes additionally require an affected segment slice, such as placement=TOP\_OF\_SEARCH. In harder episodes, the slice may be an intersection of two attributes, such as placement=TOP\_OF\_SEARCH, geo=US. Campaign-wide, segment-mix, and no-signal episodes require no separate segment field.

## 2.5 Solvability and Verification

A useful benchmark here must meet three competing requirements. Episodes must be solvable: the injected signal must be recoverable from the agent-visible data. They must be non-trivial: simple signature matching should not be sufficient. And they must be realistic: metrics should covary and confounders should be present, as in operational data. These requirements are in tension. A signal weak enough to avoid trivial detection may become unrecoverable, while a signal strong enough to guarantee recovery may collapse the task into pattern matching.

TRACE resolves this tension by separating generation from acceptance. The simulator first generates diverse episodes with varying causes, complexity axes, noise, and confounders. An oracle verifier then admits only episodes that remain recoverable from the agent-visible data. The verifier knows the ground-truth cause but reads only the agent-visible data. It does not solve the episode. Instead, it verifies that the injected signal is detectable, stronger than the strongest confounder, and distinguish able from competing causes. Accepted episodes therefore retain realistic ambiguity while providing objective labels for scoring and reward construction.

## 3 RL with Synthesized Rewards

The TRACE environment turns each accepted episode into an agent-visible diagnostic task paired with a hidden oracle label. The task defines the data observed by the agent, while the label records the injected intervention. Comparing a trajectory’s final attribution with this label produces an objective synthesized reward without human annotation or an LLM judge. This setup enables us to train an open-weight diagnostic agent under the same tool-use interface used for evaluation.

## 3.1 Reward Design

We optimize the policy with a GRPO objective [Shao et al., 2024] using stabilization refinements from recent RLVR recipes [Yu et al., 2026]. For each training episode, the policy samples a group of trajectories and receives a synthesized reward computed against the hidden oracle label.

The reward combines three terms: a graded attribution reward, a binary full-attribution reward, and a small formatting reward:

$$
r = w _ { \mathrm { a t t r } } r _ { \mathrm { a t t r } } + w _ { \mathrm { f u l l } } r _ { \mathrm { f u l l } } + w _ { \mathrm { f m t } } r _ { \mathrm { f m t } } .\tag{2}
$$

The weights are nonnegative and sum to one, so $r \in [ 0 , 1 ]$

The attribution reward decomposes into cause correctness and slice specification:

$$
r _ { \mathrm { a t t r } } = { \bf 1 } \{ \hat { c } = c ^ { \star } \} r _ { \mathrm { s l i c e } } ,\tag{3}
$$

where $\hat { c }$ is the predicted cause, $c ^ { \star }$ is the oracle cause, and $r _ { \mathrm { s l i c e } }$ scores slice specification when an affected segment slice is required. The reward term $r _ { \mathrm { a t t r } }$ gates attribution by cause correctness: a wrong cause receives zero attribution reward.

For segment-specific episodes, let $\widehat { Z }$ and $Z ^ { \star }$ denote the predicted and oracle driver slices, represented as sets of dimension–value pairs. We measure their agreement using Jaccard similarity: ${ \cal J } ( \widehat { Z } , Z ^ { \star } ) =$ $\frac { | \widehat { Z } \cap Z ^ { \star } | } { | \widehat { Z } \cup Z ^ { \star } | }$ . The graded slice reward is

$$
r _ { \mathrm { s l i c e } } = \alpha + ( 1 - \alpha ) J ( \widehat { Z } , Z ^ { \star } ) ,\tag{4}
$$

which provides partial credit for identifying some, but not all, of the affected dimensions. For campaign-wide, segment-mix, and no-signal episodes, where no driver slice is required, we set $r _ { \mathrm { s l i c e } } = 1$

The graded attribution reward assigns substantial partial credit to an incomplete multi-dimensional slice. To provide a stronger incentive for recovering the complete driver slice, we introduce a binary full-attribution reward $r _ { \mathrm { f u l l } } \in \{ 0 , 1 \}$ . We set $r _ { \mathrm { f u l l } } = \mathbf { 1 } \{ r _ { \mathrm { a t t r } } = 1 \}$ , which equals one only when the predicted cause is correct and the predicted driver slice exactly matches the oracle slice. Thus, r<sub>attr</sub> provides dense partial credit, whereas $r _ { \mathrm { f u l l } }$ rewards a complete attribution.

The last term $r _ { \mathrm { f m t } } \in \{ 0 , 1 \}$ is a format reward indicating a parseable final answer under the required schema. Parse failures results in $r _ { \mathrm { a t t r } } = 0$ . In our main results, we set $( w _ { \mathrm { a t t r } } , w _ { \mathrm { f u l l } } , w _ { \mathrm { f m t } } ) =$ (0.65, 0.30, 0.05) and $\alpha = 0 . 5$

For reward computation, we parse the final answer to extract the decision fields: the root cause and, when required, the affected segment slice. Evidence is collected for interpretability and error analysis, but it does not affect the reward. This keeps training and evaluation focused on the same attribution target.

## 3.2 Training Setup

Our trained agent is based on Qwen3.5-35B-A3B, a mixture-of-experts open-weight model with 35B total parameters and 3B active parameters. The RL dataset contains 5,000 oracle-verified episodes, split into 4,472 training and 528 validation tasks, with stratification by cause, signal level, and 1D/2D driver slice complexity. The evaluation dataset contains 235 held-out tasks that are disjoint from the training and validation tasks at episode, campaign, and intervention levels.

Training and evaluation use the same multi-turn tool interface. The agent executes Python containing SQL queries over agent-visible fact tables and return a structured final answer for cause attribution.

We compare two RL initializations, the base model and an SFT warm start checkpoint, and study the impact of reward shape and KL regularization with ablation experiments in §4.3. Full optimizer, rollout, tool-execution, hardware, and parallelism details are provided in Appendix B.

## 3.3 Supervised Warm Start

The supervised fine-tuning (SFT) as a warm-up stage before RL teaches the required answer format and multi-turn tool-use pattern, and exposes the model to valid diagnostic trajectories. We construct 1,200 SFT examples by rejection-sampling complete teacher trajectories and retaining only those whose final cause and driver slice match the oracle label. Each example contains the task prompt, tool calls, tool outputs, and final structured answer.

Candidate teacher trajectories are generated by Claude Opus 4.8. For difficult segment-specific episodes, we also provide the teacher model with lightweight hints about the root-cause signature, as a way of improving rejection-sampling efficiency. These hints are not included in the retained task prompts, RL or evaluation prompts.

## 4 Experiments

We use TRACE to evaluate whether synthesized rewards can train a stronger tool-using diagnostic agent than prompting alone. Our experiments ask four questions. Is the held-out TRACE benchmark challenging for strong frontier models? Does RL with synthesized rewards improve full attribution when initialized from either the base model or an SFT warm start? How do reward design and policy initialization affect the gains from RL? Finally, how does post-training change tool-call efficiency?

## 4.1 Experimental Setup

Evaluation dataset. All reported evaluations use the same held-out 235-episode TRACE test dataset. The dataset is deliberately enriched for segment-specific cases: 164 episodes require identifying a driver slice, including 49 whose oracle slice intersects two dimensions. This composition emphasizes exact attribution under fine-grained segmentation rather than only campaign-level cause identification.

Table 2: Performance on the 235-episode held-out TRACE test set under reward-aligned decision parsing. Cause@1 evaluates root-cause identification. FullAttr@k requires the correct cause and, for segment-specific episodes, an exact driver-slice match. Jaccard is mean Jaccard similarity on cause-correct segment-specific trajectories; No-Signal is accuracy on no-signal episodes; Parsed is the decision-extraction rate.
<table><tr><td>Model</td><td>Cause@1</td><td>Jaccard</td><td>FullAttr@1</td><td>FullAttr@5</td><td>No-Signal</td><td>Parsed</td></tr><tr><td>Qwen3.5-35B</td><td>0.184</td><td>0.596</td><td>0.159</td><td>0.370</td><td>0.41</td><td>0.53</td></tr><tr><td>+ SFT</td><td>0.685</td><td>0.958</td><td>0.637</td><td>0.762</td><td>0.76</td><td>1.00</td></tr><tr><td>+RL</td><td>0.471</td><td>0.913</td><td>0.434</td><td>0.711</td><td>0.28</td><td>0.68</td></tr><tr><td>+ SFT → RL</td><td>0.823</td><td>0.928</td><td>0.757</td><td>0.851</td><td>0.49</td><td>1.00</td></tr><tr><td>Qwen3.5-122B</td><td>0.296</td><td>0.842</td><td>0.283</td><td>0.434</td><td>0.78</td><td>0.88</td></tr><tr><td>Claude Opus 5</td><td>0.764</td><td>0.913</td><td>0.686</td><td>0.809</td><td>0.87</td><td>0.99</td></tr><tr><td>GPT-5.6 Sol</td><td>0.635</td><td>0.852</td><td>0.565</td><td>0.719</td><td>0.30</td><td>0.99</td></tr><tr><td>GPT-5.5</td><td>0.581</td><td>0.893</td><td>0.524</td><td>0.643</td><td>0.61</td><td>0.99</td></tr><tr><td>Claude Sonnet 5</td><td>0.472</td><td>0.899</td><td>0.438</td><td>0.607</td><td>0.85</td><td>1.00</td></tr></table>

Models. We evaluate four variants of Qwen3.5-35B-A3B: the base model, an SFT model, an RL model initialized from the base model, and an RL model initialized from the SFT checkpoint. We compare these variants with Qwen3.5-122B-A10B, Claude Opus 5, Claude Sonnet 5, GPT-5.5, and GPT-5.6 Sol baselines.

Metrics. We evaluate diagnostic correctness using the same cause and driver-slice criteria used by the attribution rewards in Section 3.1. Let $C = \mathbf { \bar { 1 } } \{ \hat { c } = c ^ { \star } \}$ denote root-cause correctness. For segment-specific episodes, driver-slice agreement is measured by the Jaccard similarity $J ( \widehat { Z } , Z ^ { \star } )$ A trajectory achievesfull attribution when $r _ { \mathrm { f u l l } } = 1$ : the cause is correct and, when a driver slice is required, the predicted slice exactly matches the oracle slice.

We report Cause@k, the probability that at least one of k sampled trajectories identifies the correct cause, and FullAttr@k, the probability for full attribution. We additionally report mean driver-slice Jaccard similarity conditioned on a correct cause for segment-specific episodes, no-signal accuracy, decision-parse rate, and mean executed tool calls.

## 4.2 Main Results

Table 2 summarizes our main experiment results and shows three major findings.

First, TRACE is hard and unsaturated: the strongest prompted baseline, Claude Opus 5, achieves only 0.686 FullAttr@1, with substantial remaining errors on segment-specific episodes (Section 4.4).

Second, RL with synthesized rewards is additive to SFT. SFT alone reaches 0.637 FullAttr@1, while SFT→RL improves it by 12.0 percentage points to 0.757, surpassing all prompted baselines.

Third, post-training can outweigh prompted model scale. Within the Qwen3.5 family, the posttrained 35B model substantially outperforms the prompted 122B model (0.757 versus 0.283) and also exceeds every evaluated closed-source baseline. This suggests that TRACE rewards a learnable diagnostic procedure that prompting and additional model scale alone do not reliably elicit.

RL initialized directly from the 35B base also improves FullAttr@1 substantially (0.159 → 0.434), but remains below both SFT and SFT→RL. Its decision-parse rate is also only 0.68, compared with 1.00 for both SFT-initialized models, suggesting that the supervised warm start helps the policy produce scoreable decisions as well as improve attribution.

## 4.3 Training Ablations

Table 3 examines the effects of the full-attribution reward, policy initialization, and KL regularization.

Table 3: Training ablations on the held-out TRACE test set. All columns report FullAttr@1, either overall or on the indicated episode subset. The full-attribution condition adds the binary $r _ { \mathrm { f u l l } }$ term to the graded attribution reward.
<table><tr><td>Training condition</td><td>Overall</td><td>1D Slice</td><td>2D Slice</td><td>No Signal</td></tr><tr><td>SFT only</td><td>0.637</td><td>0.71</td><td>0.04</td><td>0.76</td></tr><tr><td>SFT → RL, graded reward</td><td>0.596</td><td>0.67</td><td>0.00</td><td>0.64</td></tr><tr><td> $\mathrm { S F T }  \mathrm { R L } , { \bar { + } }$  full-attribution term</td><td>0.757</td><td>0.92</td><td>0.27</td><td>0.49</td></tr><tr><td> $\mathrm { S F T }  \mathrm { R L } , +$  full-attribution term + KL</td><td>0.724</td><td>0.92</td><td>0.16</td><td>0.30</td></tr><tr><td>RL from base, + full-attribution term</td><td>0.434</td><td>0.51</td><td>0.03</td><td>0.28</td></tr></table>

Table 4: FullAttr@1 by episode type. N is the number of test episodes in each subset. Twodimensional driver slices remain the most difficult attribution setting.
<table><tr><td>Subset</td><td>N</td><td>Base</td><td>SFT</td><td>SFT→RL</td><td>Opus 5</td></tr><tr><td>Campaign-wide causes</td><td>30</td><td>0.46</td><td>0.99</td><td>0.95</td><td>0.95</td></tr><tr><td>Segment-mix causes</td><td>21</td><td>0.48</td><td>0.98</td><td>1.00</td><td>0.98</td></tr><tr><td>Segment-specific, 1D slice</td><td>115</td><td>0.04</td><td>0.71</td><td>0.92</td><td>0.68</td></tr><tr><td>Segment-specific, 2D slice</td><td>49</td><td>0.01</td><td>0.04</td><td>0.27</td><td>0.33</td></tr><tr><td>No-signal episodes</td><td>20</td><td>0.41</td><td>0.76</td><td>0.49</td><td>0.87</td></tr></table>

The full-attribution reward primarily improves difficult slice attribution. With SFT initialization, RL using only the graded attribution reward reaches 0.596 FullAttr@1 and obtains no exact matches on episodes with two-dimensional driver slices. Adding $r _ { \mathrm { f u l l } }$ increases overall FullAttr@1 to 0.757, with gains from 0.67 to 0.92 on one-dimensional slices and from 0.00 to 0.27 on two-dimensional slices. This pattern suggests that an explicit reward for complete attribution is especially important when success requires recovering multiple driver dimensions.

SFT initialization remains important. Under the same reward containing $r _ { \mathrm { f u l l } }$ , RL initialized from the base model reaches 0.434 FullAttr@1, compared with 0.757 when initialized from SFT. Thus, RL improves the base policy, but does not recover the performance of the supervised warm start under the matched training recipe.

KL regularization does not improve the observed result. Adding a KL penalty to the SFT reference reduces FullAttr@1 from 0.757 to 0.724. Performance on two-dimensional slices decreases from 0.27 to 0.16, while one-dimensional performance remains unchanged at 0.92. No-signal accuracy also decreases from 0.49 to 0.30.

## 4.4 Performance Breakdown

Table 4 decomposes FullAttr@1 by cause scope and driver-slice dimensionality. SFT and SFT→RL approach ceiling on campaign-wide and segment-mix causes. Post-training provides its largest gain over SFT on one-dimensional segment-specific episodes, improving FullAttr@1 from 0.71 to 0.92.

Two-dimensional attribution remains the principal challenge. SFT→RL improves FullAttr@1 from 0.04 to 0.27 on these episodes, but still performs below Claude Opus 5 at 0.33. RL also reduces no-signal accuracy from 0.76 after SFT to 0.49, revealing a trade-off between stronger attribution and avoiding false-positive diagnoses.

To examine two-dimensional errors more closely, we analyze all five saved trajectories per episode. Among cause-correct trajectories whose oracle slice contains a device dimension, SFT→RL recovers the correct device–value pair in 66/94 cases, compared with 14/39 for SFT and 12/31 for RL from the base. Thus, the strongest trained model is substantially better at recovering this second dimension, although incomplete slices remain common.

The model also tends to over-attribute changes when no signal is present. SFT→RL predicts NO\_SIGNAL in only 49/100 saved no-signal trajectories, compared with 76/100 after SFT. Another recurring ambiguity occurs between segment-specific causes with similar observable signatures. Of the 46 SFT→RL failures to identify AD\_QUALITY\_DEGRADATION, 24 predict COM-

![](images/17227c3a910014233ff07b6a4f1e584cd6fd89ff1ef9cb09c2454739a4dcf78f.jpg)  
Figure 2: FullAttr@1 versus mean executed tool calls per evaluation trajectory. Each trajectory entry corresponds to one executed Python-tool call, including unsuccessful calls. Blue points denote fine-tuned Qwen3.5-35B variants, and orange points denote prompted baselines. Better performance lies toward the upper left.

PETITIVE\_PRESSURE. These results suggest that further gains require both more reliable multidimensional slice recovery and better calibration among related causes.

## 4.5 Tool-Call Efficiency

Figure 2 compares FullAttr@1 with the mean number of executed tool calls per evaluation trajectory. SFT simultaneously improves attribution and reduces mean tool use relative to the prompted 35B base, from 22.05 to 10.75 calls. RL after SFT increases mean tool use by only 0.98 calls, to 11.73, while improving FullAttr@1 by 12.0 percentage points. In contrast, RL from the base averages 16.04 calls while remaining less accurate than SFT. These results indicate that the post-training gains are not explained by simply making more tool calls. They also show that supervised warm start teaches a more effective investigation-and-stopping procedure, rather than merely encouraging additional exploration.

## 5 Conclusion

We introduced a simulator–oracle–RL approach for training diagnostic-reasoning agents when naturally occurring verifiers are scarce. A controlled simulator samples a hidden intervention, generates the data resulting from that intervention, and retains it as an oracle label. This construction makes the final attribution objectively verifiable without removing the noise, confounders, multi-table evidence, and exploratory tool use that make the diagnostic task difficult. We instantiate this approach in TRACE, a generative digital-advertising environment containing campaign-wide, segment-mix, and segment-specific root causes.

On the held-out TRACE benchmark, the strongest prompted baseline reaches 0.686 FullAttr@1. SFT raises Qwen3.5-35B-A3B from 0.159 to 0.637, and subsequent RL with synthesized rewards further improves it to 0.757, surpassing every evaluated prompted baseline, including Qwen3.5-122B-A10B. This result provides evidence that, for this diagnostic setting, the binding constraint is access to an effective post-training signal—including a scalable, objective reward—rather than model scale alone.

Several directions could extend this approach. Richer agent harnesses could provide explicit planning, memory, adaptive tool selection, and improved stopping mechanisms, allowing the policy to conduct longer and more structured investigations. The simulator–oracle construction could also be expanded to broader intervention families, held-out root causes, and domains such as software operations and data-quality diagnosis, where failures can be injected and their downstream effects observed. Finally, simulator control enables systematic curricula over noise, confounding, evidence availability, and attribution complexity, providing a way to study which diagnostic procedures transfer across environments.

More broadly, our results demonstrate a route to constructing objective training signals for otherwise ambiguous diagnostic tasks: when a domain admits a controllable generative model, the intervention that creates a difficult problem can also provide its verifier.

## References

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Shuaichen Chang, Jun Wang, Mingwen Dong, Lin Pan, Henghui Zhu, Alexander Hanbo Li, Wuwei Lan, Sheng Zhang, Jiarong Jiang, Joseph Lilien, et al. Dr. spider: A diagnostic evaluation benchmark towards text-to-sql robustness. arXiv preprint arXiv:2301.08881, 2023.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, and Benyou Wang. Towards medical complex reasoning with llms through medical verifiable problems. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 14552–14573, 2025.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. Mlagentbench: Evaluating language agents on machine learning experimentation. In International Conference on Machine Learning, pages 20271–20309. PMLR, 2024.

Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, et al. Cladder: Assessing causal reasoning in language models. Advances in Neural Information Processing Systems, 36:31038– 31065, 2023.

Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Mrinmaya Sachan, Rada Mihalcea, Mona Diab, David Ha, et al. Can large language models infer causation from correlation? In International Conference on Learning Representations, volume 2024, pages 28663–28679, 2024.

Emre Kiciman, Robert Ness, Amit Sharma, and Chenhao Tan. Causal reasoning and large language models: Opening a new frontier for causality. Transactions on machine learning research, 2024.

Minsu Kim and Se-Young Yun. Process-verified reinforcement learning for theorem proving via lean. arXiv preprint arXiv:2606.20068, 2026.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Ren Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. In International Conference on Machine Learning, pages 26874–26901. PMLR, 2024.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, et al. Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls. Advances in Neural Information Processing Systems, 36:42330–42357, 2023.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jian-Guang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, Yansong Tang, et al. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. In International Conference on Learning Representations, volume 2025, pages 49573–49609, 2025a.

Michael Luo, Naman Jain, Jaskirat Singh, Sijun Tan, Ansh Patel, Qian Wu, Alpay Ariyak, Chenguang Cai, S. Z. T. Venkat, Shuyan Zhu, Ben Athiwaratkun, Mohit Roongta, Chuxiong Zhang, Lucy E. Li, Raluca Ada Popa, Koushik Sen, and Ion Stoica. DeepSWE: Training a fully open-sourced, state-of-the-art coding agent by scaling RL. Together AI / Agentica technical blog, 2025b. URL https://www.together.ai/blog/deepswe. Technical blog.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460–9471, 2022.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298, 2022.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings ofthe 61st annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 13484–13508, 2023.

Zeyu Wang. Causalbench: A comprehensive benchmark for evaluating causal reasoning capabilities of large language models. In Proceedings of the 10th SIGHAN Workshop on Chinese Language Processing (SIGHAN-10), pages 143–151, 2024.

Jason Wei. Asymmetry of verification and verifier’s law. Blog post, July 2025. URL https:// www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law. Accessed 2026-06-16.

Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, and Sida Wang. Swe-rl: Advancing llm reasoning via reinforcement learning on open software evolution. Advances in Neural Information Processing Systems, 38:78500–78525, 2026.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In International Conference on Learning Representations, volume 2024, pages 45040–45061, 2024.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, et al. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task. In Proceedings ofthe 2018 conference on empirical methods in natural language processing, pages 3911–3921, 2018.

Dan Zhang, Sining Zhoubian, Min Cai, Fengzu Li, Lekang Yang, Wei Wang, Tianjiao Dong, Ziniu Hu, Jie Tang, and Yisong Yue. Datascibench: An llm agent benchmark for data science. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 3685–3728, 2026.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Victor Zhong, Caiming Xiong, and Richard Socher. Seq2sql: Generating structured queries from natural language using reinforcement learning. arXiv preprint arXiv:1709.00103, 2017.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606, 2024.

## A Database and Output Schemas

## A.1 Agent-Visible Fact Tables

Each episode provides access to a DuckDB fact store containing the four tables summarized in Table 5. Rows are keyed by campaign identifiers, and the task prompt specifies the campaign to investigate. The agent can access these tables only through its Python/SQL tool; no ground-truth field is included in the tool-visible database.

Table 5: Agent-visible database schema. All tables include a date field ds and campaign identifiers as appropriate.
<table><tr><td>Table</td><td>Grain</td><td>Fields</td></tr><tr><td>daily_campaign</td><td>Campaign-day</td><td>Campaign identifiers and category; budget, spend, im- pressions, clicks, orders, sales, page views, and brand searches.</td></tr><tr><td>segment_daily</td><td>Segment-day</td><td>Campaign identifiers and category; ad product, target- ing type, match type, placement, price band, audience, device, and geography; delivery, engagement, and con-</td></tr><tr><td>budget_log</td><td>Campaign-day</td><td>version metrics. Advertiser and campaign identifiers, budget, and spend.</td></tr><tr><td>inventory</td><td>Campaign-day</td><td>Campaign identifiers and category, together with the fraction of advertised products that are in stock.</td></tr></table>

## A.2 Hidden Oracle Tables

The generation database additionally contains three oracle-only tables. gt\_episode records the injected root cause, affected segment assignment, intervention strength, temporal pattern, and number of affected dimensions. gt\_evidence stores post-computed evidence associated with the intervention, including its source table, metric, segment, direction, and change relative to the baseline period. gt\_validation records the oracle verifier’s detectability and confounder-elimination checks. These tables are retained for dataset construction and scoring but are excluded when the agent-visible database is created.

The required final answer contains a decision object with a non-empty root\_cause and an optional driver\_segments list, an evidence list, and a textual explanation. For campaign-wide, segment mix, and no-signal episodes, driver\_segments is omitted or null. For segment-specific episodes, it contains one or more dimension–value assignments.

## A.3 Oracle Verification

After simulation, the oracle verifier compares the episode window with an equal-length preceding baseline window. It first checks that the expected signature is present at the appropriate level: campaign metrics for campaign-wide causes, impression-share changes for segment-mix causes, and changes within the injected segment for segment-specific causes. For delayed and ramping interventions, it also tests the second half of the episode window so that a recoverable late-onset signal is not rejected solely because it is diluted in the full-window average. Segment-specific signals must additionally exceed their own baseline temporal variation, using a minimum signal-to-noise ratio of 1.

The verifier then tests each alternative cause against the same agent-visible data. An alternative is eliminated using differences in signal level, source table, affected dimension, primary metric, or companion-metric direction. Episodes are rejected when the injected signal is absent or when a matching alternative cannot be eliminated. No-signal episodes are retained only when observed fluctuations do not realize another cause’s signature. Thus, acceptance uses the hidden intervention to define what must be verified, but all detectability and distinguishability checks operate on the same fact tables available to the agent.

## B Training Details

## B.1 Training Setup

We train Qwen3.5-35B-A3B using a slime fork with Megatron-LM for optimization, SGLang for rollout inference, and Ray for orchestration. RL uses an asynchronous pipeline in which rollout generation for the next step overlaps policy optimization, with updated weights synchronized from Megatron-LM to SGLang after each optimizer step. Training uses bfloat16 parameters with float32 gradient accumulation, softmax, and router computation.

## B.2 Supervised Fine-Tuning

The SFT stage trains on 1,200 oracle-filtered teacher trajectories for three epochs. We use a learning rate of $1 0 ^ { - 5 }$ with cosine decay and apply the language-model loss only to assistant turns under the Qwen3.5 chat template. Each example retains the complete interaction, including prompts, tool calls, tool outputs, and the final structured answer.

## B.3 Reinforcement Learning

Table 6 gives the shared GRPO configuration. Each optimizer step draws 32 prompts and samples eight trajectories per prompt. The agent has a 32,768-token context window, a 30-turn backstop, and a maximum of 2,048 generated tokens per turn. Training rollouts use temperature 1.0 to maintain within-group exploration.

Table 6: Shared hyperparameters for the GRPO runs.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Training set</td><td>4,472 prompts</td></tr><tr><td>Validation set</td><td>528 prompts</td></tr><tr><td>Optimizer steps</td><td>300 (approximately two epochs)</td></tr><tr><td>Prompts per step</td><td>32</td></tr><tr><td>Samples per prompt</td><td>8</td></tr><tr><td>Effective trajectories per step</td><td>256</td></tr><tr><td>Optimizer</td><td>AdamW, β = (0.9, 0.98)</td></tr><tr><td>Learning rate</td><td>10−6, constant</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>PPO clipping range</td><td>[0.8,1.28]</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Entropy coefficient</td><td>0</td></tr><tr><td>Rollout temperature</td><td>1.0</td></tr><tr><td>Maximum turns</td><td>30</td></tr><tr><td>Context length</td><td>32,768 tokens</td></tr><tr><td>Maximum generation per turn</td><td>2,048 tokens</td></tr><tr><td>Validation frequency</td><td>Every 20 optimizer steps</td></tr><tr><td>Validation sampling</td><td>Five trajectories per prompt at temperature 0.6</td></tr></table>

The main RL condition uses the reward weights reported in Section 3.1. The KL ablation adds a penalty to the SFT reference policy with coefficient 0.005, using the low-variance k3 estimator. All RL conditions use the same data split, rollout group size, optimizer settings, and nominal 300-step budget. The validation split is stratified by root cause, signal level, and one- versus two-dimensional segment attribution and is disjoint from both the RL training prompts and SFT demonstrations.

## C Additional Evaluation Details

## C.1 Evaluation Protocol

All models receive the same task specification, candidate-cause definitions, final-answer contract, and Python/SQL tool interface over the agent-visible fact store. The Python state persists across calls within an episode. Tool calls have a 60-second execution timeout, and tool outputs are truncated after 8,000 characters. A final-turn nudge asks the model to return its answer before exhausting the interaction budget.

Open-weight evaluations sample five trajectories per task at temperature 0.6 with a 60-step backstop. API-model evaluations use provider-supported stochastic sampling without an explicit temperature and a 30-turn backstop. The reported Claude and GPT baselines use the xhigh reasoning-effort setting. Claude models receive a maximum output-token budget of 32,768, while GPT models receive 16,384 output tokens. Every reported FullAttr@5 value is computed from five distinct samples per task. The interaction limits do not bind for the trained or frontier models; one Qwen3.5-35B base-model trajectory is truncated.

For n sampled trajectories containing c successful trajectories, Cause@k and FullAttr@k use the standard unbiased estimator

$$
1 - { \frac { { \binom { n - c } { k } } } { \binom { n } { k } } } .\tag{5}
$$

Cause@k treats root-cause correctness as success, whereas FullAttr@k additionally requires an exact segment assignment when the episode is segment-specific. At k = 1, the estimator is the mean single-trajectory success probability over the test tasks.

## C.2 Answer-Parser Sensitivity

For the primary evaluation, we parse each model’s final answer to extract the decision fields needed to assess attribution: a non-empty root cause and any predicted segment assignment. Malformed report only evidence, an omitted explanation, or benign extra JSON keys do not invalidate an otherwise scoreable decision. A malformed segment assignment is treated as absent and therefore cannot earn full attribution on a segment-specific episode. We report the decision-parse rate separately from diagnostic accuracy.

As a sensitivity analysis, we also use an exact-schema parser that requires the complete prescribed output structure. Table 7 reports a paired re-scoring of the same saved trajectories under both parsers. The parser choice has no effect on the SFT or SFT→RL results, but exact-schema parsing underestimates FullAttr@1 when a model produces a semantically scoreable decision with malformed or missing report-only fields.

Table 7: FullAttr@1 obtained by re-scoring the same trajectories with an exact-schema parser and the decision parser used for the main results.
<table><tr><td>Model</td><td>Exact Schema</td><td>Decision Parser</td></tr><tr><td>Qwen3.5-35B</td><td>0.140</td><td>0.159</td></tr><tr><td>+ SFT</td><td>0.637</td><td>0.637</td></tr><tr><td>+ RL</td><td>0.317</td><td>0.434</td></tr><tr><td>+ SFT → RL</td><td>0.757</td><td>0.757</td></tr><tr><td>Claude Opus 5</td><td>0.665</td><td>0.686</td></tr></table>