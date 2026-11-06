# ATTNSR: ATTENTION STEP REWARD FOR LONG-HORIZON AGENTIC RL

Mengsong Wu<sup>1,2,3†</sup> Tianchen Zhou<sup>3,4†</sup> Zhaochen Su<sup>5†</sup> Zhenhua Liu<sup>3‡</sup> Zhe Cui<sup>3\*</sup> Aimin Zhou<sup>1,2\*</sup>

<sup>1</sup>Shanghai Institute of AI for Education, East China Normal University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Hithink Research   
<sup>4</sup>School of Data Science and Engineering, East China Normal University <sup>5</sup>The Hong Kong University of Science and Technology cuizhe@myhexin.com amzhou@cs.ecnu.edu.cn

## ABSTRACT

Long-horizon LLM agents make multi-step decisions, yet outcome-reward-only reinforcement learning provides little direct supervision for individual steps. We introduce ATTNSR, a step-level reward augmentation method that derives reward allocations from policy attention extracted during training replay, without a separately trained reward model or additional rollouts. ATTNSR traces backward through the policy model attention graph to estimate the relevance of earlier steps to the final response. The terminal step, measured relative to a running baseline, determines whether these rewards reinforce or penalize the selected steps. In a controlled Qwen2.5-7B-Instruct study, ATTNSR consistently outperforms the PPO baseline across all evaluated checkpoints on ALFWORLD, raising final held-out success from 91.2% to 97.1%. The ablation experiment shows that the benefit depends on how intermediate rewards are assigned, rather than simple accumulation of step rewards over time. ATTNSR also improves SCIENCEWORLD at every checkpoint while cutting environment interactions by 29.9%. These results support policy attention as a practical prior for step-level reward allocation in long-horizon agentic RL. Code is available at https://github.com/fairyshine/AttnSR.

## 1 INTRODUCTION

Reinforcement learning (RL) with verifiable rewards (Shao et al., 2024; DeepSeek-AI, 2025) is increasingly applied to agents that interleave reasoning and tool use (Putta et al., 2024; Wang et al., 2025b; Yao et al., 2023). In environments such as ALFWORLD, WEBSHOP, and SCIENCEWORLD, agents make sequences of decisions to complete a task (Shridhar et al., 2021; Yao et al., 2022; Wang et al., 2022). Vanilla PPO uses the outcome reward to estimate value targets and policy advantages (Schulman et al., 2017; 2016), but says little about how intermediate actions contribute. Assigning credit to earlier decisions thus remains a core challenge in long-horizon RL (Zhang, 2026).

Step-level rewards address this challenge by providing more localized training signals, but obtaining them can require step annotations, additional environment interactions, or a separately trained evaluator (Lightman et al., 2024; Setlur et al., 2024; Kazemnejad et al., 2025). This motivates using signals already available within the policy model to guide intermediate rewards, without incurring the substantial training and environment-interaction costs.

Attention patterns in the policy model may provide such a signal, since they record how later decisions attend to earlier actions and observations. We hypothesize that these relationships provide a useful prior for allocating intermediate rewards, although attention alone does not establish causal contributions (Jain & Wallace, 2019; Abnar & Zuidema, 2020). The signal can be extracted during the existing forward pass of the policy model over collected trajectories. The method adds little computation and requires neither a separate attribution backward pass nor a learned progress estimator.

Building on this hypothesis, we introduce Attention Step Reward (ATTNSR), which uses the attention of the policy model to identify relevant earlier steps and turns the task outcome into intermediate training signals. Starting from the final step, it traces attention links backward to estimate the relevance of earlier actions and observations. These estimates determine which intermediate steps receive reward, while weaker or less consistent attention reduces their overall magnitude. Outcomes above the running average score yield positive intermediate rewards, and those below it yield negative rewards. These signals can in principle be used directly by step-level RL algorithms or placed at assistant-step endpoints for token-level RL. We study the latter integration in PPO, where the added rewards change value targets and policy advantages while retaining the critic architecture and losses.

We evaluate ATTNSR on ALFWORLD, SCIENCEWORLD, and WEBSHOP using Qwen2.5-7B-Instruct with matched token-level PPO training. ALFWorld success rises from 91.2% to 97.1%, with larger gains on unseen tasks, whereas the positional ablation reaches 92.3%. On ScienceWorld, ATTNSR yields higher official scores at all four reported checkpoints and 29.9% fewer mean environment steps. These results suggest that policy attention can serve as a useful signal for intermediate reward allocation, offering a way to enrich outcome-based supervision without additional rollouts or a separately trained reward model.

Our contributions are summarized below.

• We hypothesize that policy attention provides a relevance prior over interaction steps by indicating how later decisions attend to earlier actions and observations. Based on this hypothesis, we use attention-derived dependencies to distribute outcome information across steps for temporal credit assignment, treating attention as a heuristic relevance signal, not evidence of causal contribution.

• We introduce ATTNSR, a step-level reward augmentation scheme that combines attention-based allocation, confidence gating, and outcome centering. The signals can be used directly as step-level rewards or aligned with assistant-step endpoints for token-level objectives, and we study the latter integration with PPO. It needs neither counterfactual rollouts nor a separately trained reward model.

• We evaluate ATTNSR across three agentic benchmarks with ablation and trajectory analysis to characterize its performance gains and potential boundary conditions across tasks with different interaction lengths. The results are consistent with a stronger benefit on complex, long-horizon tasks, while suggesting more limited gains when trajectories are short.

## 2 PROBLEM FORMULATION

We model each LLM-based agent interaction trajectory as a multi-turn conversation. Episode i starts from a task query and contains K<sub>i</sub> decision steps. Each step $S _ { i , k }$ comprises an assistant message $A _ { i , k }$ and its environment observation $O _ { i , k }$ when present, and the final message may have no observation. Let $\mathcal { T } _ { i , k }$ be the assistant token indices at step k, and $e _ { i , k } \in \mathcal { T } _ { i , k }$ its last trainable token, the step endpoint. We supply PPO with a task outcome reward $R _ { i } \in [ 0 , 1 ]$ . In the canonical runs, $R _ { i }$ is binary success for ALFWORLD, the normalized official score for WEBSHOP, and the normalized final score for SCIENCEWORLD. Format-error trajectories receive $R _ { i } = 0$ . Thus the binary success metric for WEBSHOP and SCIENCEWORLD is an evaluation threshold on the official score, not a replacement for the training reward. Table 16 in Appendix H summarizes the notation.

Our terminal-only PPO baseline aligns rewards with trajectory tokens, indexed by u, placing the outcome at the final assistant endpoint as

$$
r _ { i , u } ^ { \mathrm { P P O } } = R _ { i } { \rlap / { k } } [ u = e _ { i , K _ { i } } ] ,\tag{1}
$$

where $\nVdash [ \cdot ]$ is the indicator function. Observation tokens provide context but are masked from actor and critic losses. TD residuals and GAE compute value-function targets and policy advantages over valid assistant tokens, skipping masked positions (Schulman et al., 2016; 2017). The outcome reward evaluates the whole trajectory without specifying contributions of individual actions.

Let $s _ { i , k }$ denote a signed shaping reward assigned to intermediate step $k < K _ { i }$ . ATTNSR augments the baseline reward sequence as

$$
r _ { i , u } ^ { \mathrm { A t t n S R } } = r _ { i , u } ^ { \mathrm { P P O } } + \sum _ { k = 1 } ^ { K _ { i } - 1 } s _ { i , k } \lvert \mathcal { k } [ u = e _ { i , k } ] .\tag{2}
$$

![](images/d9f5d807cb3b183da8ba061139d5a24a4b122035066cd88a128d507b420d0f07.jpg)  
Figure 1: Overview of ATTNSR. Actor replay supplies attention for a step graph. Backtracking from the terminal step, recency correction, and layer fusion yield an allocation, and confidence and the centered outcome scale it into signed intermediate rewards. The outcome reward is retained. Graph edges link later steps to earlier dependencies, and episode indices are omitted.

This retains $R _ { i }$ at the final endpoint and adds $s _ { i , k }$ at intermediate assistant endpoints. Algorithms that consume step-level rewards can use these signals directly in principle, and Equation 2 specifies their token-level placement for the PPO implementation studied here. Our goal is to derive $s _ { i , k }$ from actor attention and terminal outcomes. Section 3 defines its allocation, confidence gate, and sign.

## 3 ATTNSR: ACTOR ATTENTION FOR STEP CREDIT

ATTNSR constructs intermediate rewards by tracing how later decisions attend to earlier interaction steps. It uses these relationships to allocate rewards, with their magnitude and sign determined by the reliability of the attention signal and the trajectory outcome relative to a running baseline.

## 3.1 WHERE TO ASSIGN CREDIT: A STEP DEPENDENCY GRAPH

For each selected transformer layer l, we construct a directed graph over interaction steps. Its components $G _ { i , \mathrm { A A } } ^ { ( l ) } [ j , k ]$ and $G _ { i , \mathrm { A O } } ^ { ( l ) } [ j , k ]$ measure attention from the assistant message at step j to the assistant and observation messages at step $k ,$ respectively. We sum causal attention probabilities over each historical message key tokens, then average over step j query tokens and heads. Only $k < j$ edges are retained, and missing observations contribute zero AO mass. The combined graph is $G _ { i } ^ { ( l ) } = 0 . 5 G _ { i , \mathrm { A A } } ^ { ( l ) } + 0 . 5 G _ { i , \mathrm { A O } } ^ { ( l ) }$

During actor replay for old log-probabilities, we capture Q/K after rotary position embedding (Su et al., 2024) and before fused attention (Dao et al., 2022), reconstructing the packed tensor/contextparallel sequence. Query blocks use full causal-prefix normalization before message aggregation, and the historical-step submatrix is not renormalized, which preserves its mass relative to other context tokens. Figure 1 illustrates how token attention becomes a step graph for reward construction.

Relevance then flows backward from the terminal step, independently within each layer. Writing $a _ { i , k } ^ { ( l ) } \geq 0$ for the relevance layer l assigns to step k,

$$
a _ { i , K _ { i } } ^ { ( l ) } = 1 , \qquad a _ { i , k } ^ { ( l ) } = \rho \sum _ { j > k } a _ { i , j } ^ { ( l ) } G _ { i } ^ { ( l ) } [ j , k ] ,\tag{3}
$$

where $j \leq K _ { i }$ . Each backward path contributes its edge-weight product with decay $\rho = 0 . 8 5$ per hop, capturing direct and indirect dependencies. This graph-hop decay differs from the temporal-distance correction below. Only $k < K _ { i }$ scores enter reward allocation.

## 3.2 HOW MUCH TO TRUST ATTENTION: CORRECTION AND CONSENSUS

To reduce recency preference, define terminal distance $\Delta _ { i , k } \ : = \ : K _ { i } - k$ and proximity $d _ { i , k } ~ =$ $\exp ( - ( \Delta _ { i , k } - 1 ) / \kappa )$ . For $K _ { i } > 1$ , we standardize log relevance and proximity over intermediate steps as

$$
z _ { i , k } ^ { ( l ) } = \mathrm { n o r m } \Big ( \log \operatorname* { m a x } ( a _ { i , k } ^ { ( l ) } , \epsilon ) \Big ) + \eta \mathrm { n o r m } ( d _ { i , k } ) , \qquad p _ { i , k } ^ { ( l ) } = \frac { \exp ( z _ { i , k } ^ { ( l ) } / T ) } { \sum _ { h < K _ { i } } \exp ( z _ { i , h } ^ { ( l ) } / T ) } .\tag{4}
$$

Here $\epsilon = 1 0 ^ { - 1 2 }$ and $\operatorname { n o r m } ( x ) \ = \ ( x - \operatorname { m e a n } ( x ) ) / \operatorname { s t d } ( x )$ uses population standard deviation, returning zero for a single element or standard deviation at most $1 0 ^ { - 6 }$ . We use $T = 1 , \kappa = 4$ , and $\eta = - 0 . 2 5$ . The negative proximity term reduces allocation to recent steps but does not set the sign.

Because attribution varies across layers (Abnar & Zuidema, 2020), we fuse their independently computed allocations by a geometric mean,

$$
q _ { i , k } = \exp \left( \frac { 1 } { L } \sum _ { l \in \mathcal { L } } \log \operatorname* { m a x } \left( p _ { i , k } ^ { \left( l \right) } , \epsilon \right) \right) , \qquad p _ { i , k } = \frac { q _ { i , k } } { \sum _ { h < K _ { i } } q _ { i , h } } ,\tag{5}
$$

where $L = | \mathcal { L } |$ is the number of selected layers. Geometric fusion favors consistent support across layers, and the normalization runs over intermediate steps.

Confidence controls the strength of the attention-derived reward. For layer l, define the terminal covered mass as

$$
m _ { i } ^ { ( l ) } = \sum _ { k < K _ { i } } \left( G _ { i , \mathrm { A A } } ^ { ( l ) } [ K _ { i } , k ] + G _ { i , \mathrm { A O } } ^ { ( l ) } [ K _ { i } , k ] \right) \in [ 0 , 1 ] .\tag{6}
$$

This uses the summed AA and $\mathrm { A O }$ masses before the equal-weight graph mixture. For $K _ { i } > 2 .$ , define the normalized entropy

$$
H _ { i } ^ { ( l ) } = - \frac { \sum _ { k < K _ { i } } p _ { i , k } ^ { ( l ) } \log \operatorname* { m a x } ( p _ { i , k } ^ { ( l ) } , \epsilon ) } { \log ( K _ { i } - 1 ) } .\tag{7}
$$

For $K _ { i } = 2 $ , the unique intermediate step has allocation one and we set $H _ { i } ^ { ( l ) } = 0$ . The confidence components are

$$
C _ { i , \mathrm { m a s s } } ^ { ( l ) } = \mathrm { c l i p } ( m _ { i } ^ { ( l ) } / 0 . 2 0 , 0 , 1 ) ,\tag{8}
$$

$$
C _ { i , \mathrm { f o c u s } } ^ { ( l ) } = \mathrm { c l i p } ( ( 1 - H _ { i } ^ { ( l ) } ) / 0 . 1 5 , 0 , 1 ) ,\tag{9}
$$

$$
\begin{array} { r } { C _ { i , \mathrm { l a y e r } } = 1 - \operatorname* { m e a n } _ { l } \mathrm { T V } ( p _ { i } ^ { ( l ) } , p _ { i } ) , } \end{array}\tag{10}
$$

$$
C _ { i } = \mathrm { m e a n } _ { l } \left( C _ { i , \mathrm { m a s s } } ^ { ( l ) } C _ { i , \mathrm { f o c u s } } ^ { ( l ) } \right) C _ { i , \mathrm { l a y e r } } \in [ 0 , 1 ] ,\tag{11}
$$

where cl $\operatorname { i p } ( x , 0 , 1 )$ saturates to [0, 1] and $\begin{array} { r } { \mathrm { T V } ( p , q ) = \frac { 1 } { 2 } \sum _ { k } \left| p _ { k } - q _ { k } \right| } \end{array}$ . Covered mass, concentration, and layer agreement measure historical attention, selectivity, and consistency, respectively. The values 0.20 and 0.15 are saturation thresholds. The resulting confidence is a shaping gate, not a calibrated probability of correct attribution.

Zero total intermediate relevance sets that layer focus confidence to zero. For $K _ { i } = 1$ , we skip allocation, set $C _ { i } = 0$ , and retain terminal-only rewards.

## 3.3 WHETHER TO REINFORCE A STEP: A CENTERED BUDGET

Attention indicates computational relevance, not whether a step contributed positively to success. We determine reward sign relative to a running outcome baseline. For collected training batch t with mean outcome reward $M _ { t }$ , we initialize or update the baseline before shaping as

$$
\begin{array} { r } { \bar { R } _ { t } = \left\{ \begin{array} { l l } { M _ { 1 } , } & { t = 1 , } \\ { 0 . 9 9 \bar { R } _ { t - 1 } + 0 . 0 1 M _ { t } , } & { t > 1 , } \end{array} \right. \quad \quad B _ { i } = \beta C _ { i } ( R _ { i } - \bar { R } _ { t } ) , } \end{array}\tag{12}
$$

Algorithm 1 ATTNSR reward construction for one batch   
Require: completed trajectory with endpoints $e _ { i , k }$ and rewards $R _ { i }$ , selected layers ${ \overline { { \mathcal { L } } } } ,$ and the   
previous baseline if initialized   
Ensure: token rewards $r _ { i , u }$ and updated baseline $\bar { R } _ { t }$   
1: capture detached post-RoPE ${ \dot { Q } } ,$ K during actor replay for old log-probabilities   
2: initialize $r _ { i , u } \gets R _ { i } \lvert \lvert \mathscr { k } [ u = e _ { i , K _ { i } } ]$ and $\bar { C _ { i } } \bar {  } 0$ for each turn   
3: compute batch mean $M _ { t } ,$ then initialize or update $\bar { R } _ { t }$ by Equation 12   
4: for each episode i with $K _ { i } > 1$ do   
5: for each layer $l \in \mathcal L$ do   
6: aggregate causal attention into AA/AO step graphs and set $\begin{array} { r } { G _ { i } ^ { ( l ) } \gets \frac { 1 } { 2 } ( G _ { i , \mathrm { A A } } ^ { ( l ) } + G _ { i , \mathrm { A O } } ^ { ( l ) } ) } \end{array}$   
7: backtrack relevance and apply recency correction to obtain $p _ { i } ^ { ( l ) }$ ▷ Eqs. 3–4   
8: compute mass and focus confidence, setting focus to zero if intermediate relevance is   
zero   
9: fuse layer allocations into $p _ { i }$ and compute $C _ { i }$ ▷ Eqs. 5–11   
10: add $\beta \dot { C } _ { i } ( R _ { i } - \bar { R } _ { t } ) p _ { i , k }$ at $u = e _ { i , k }$ for $\mathsf { \bar { k } } < K _ { i }$ ▷ retain outcome reward   
11: return $r _ { i , u } , \bar { R } _ { t }$

where $\beta = 0 . 2 5$ . The detached baseline is shared across the batch. Outcomes above it yield positive budgets, and those below it yield negative budgets. Thus sign reflects relative outcome quality, not binary success. For $k < K _ { i }$ , we assign

$$
s _ { i , k } = B _ { i } p _ { i , k } = \beta C _ { i } ( R _ { i } - \bar { R } _ { t } ) p _ { i , k } ,\tag{13}
$$

at endpoint $e _ { i , k }$ , retaining $R _ { i }$ at $e _ { i , K _ { i } }$ . For $K _ { i } ~ > ~ 1$ , intermediate rewards sum to $B _ { i }$ , and zero confidence yields no shaping. Format-error trajectories retain environment reward $R _ { i } ~ = ~ 0$ and participate in the EMA update. Algorithm 1 summarizes reward construction for one batch.

Since $\begin{array} { r } { C _ { i } , R _ { i } , \bar { R } _ { t } \in [ 0 , 1 ] , \sum _ { k < K _ { i } } | s _ { i , k } | = | B _ { i } | \le \beta . } \end{array}$ , so the upper bound is length-independent, while the realized budget depends on confidence and outcome. The undiscounted shaped return is $R _ { i } + B _ { i }$ so policy invariance is not guaranteed. This is a training heuristic distinct from return-preserving redistribution (Arjona-Medina et al., 2019), and evaluation uses the original task metrics.

## 3.4 COMPLEXITY AND INTEGRATION

Blockwise aggregation into step buckets avoids storing full token attention matrices but retains quadratic query–key arithmetic and selected-layer buffers. It reuses the actor forward pass without additional rollouts or a separately trained reward model. Wall-clock and peak-memory overheads remain unmeasured. Rewards enter PPO before advantage estimation, keeping its critic and losses.

Actor attention is detached when constructing rewards, while policy probabilities remain differentiable during optimization. Given pre-update critic values $V _ { u } ^ { \mathrm { o l d } }$ and unnormalized GAE advantages $\hat { A } _ { u } ,$ , the canonical PPO training signals are

$$
\widetilde { A } _ { u } = \mathrm { s g } \Bigl ( \mathrm { N o r m } ( \hat { A } ) _ { u } \Bigr ) , \qquad y _ { u } = \mathrm { s g } \Bigl ( \hat { A } _ { u } + V _ { u } ^ { \mathrm { o l d } } \Bigr ) ,\tag{14}
$$

where Norm uses the response mask and sg denotes stop-gradient. The actor uses $ { \widetilde { A } } _ { u }$ in its clipped surrogate, and the critic fits returns $y _ { u } ,$ constructed before advantage normalization, using the existing clipped value loss with reference $\check { V } ^ { \mathrm { o l d } }$ . Both signals remain fixed within the batch, and neither loss backpropagates through GAE or reward construction.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Agentic Tasks. We evaluate on ALFWORLD (274 held-out tasks, 140 seen and 134 unseen), WEBSHOP (500 test tasks), and SCIENCEWORLD (1,819 test tasks) (Shridhar et al., 2021; Yao et al., 2022; Wang et al., 2022), covering household interaction, online shopping, and scientific procedures.

![](images/169e78970840698984d79bcf7866a31ae603cab95dc94b6bfc5d71a24d444c3e.jpg)  
Figure 2: Integration of ATTNSR into actor–critic RL. (a) The PPO integration of ATTNSR adds intermediate rewards while retaining the outcome reward. (b) Detached actor attention supplies rewards, and pre-update critic values support GAE and target construction. Normalized advantages guide the actor, while estimated returns supervise the critic as in Equation 14. Targets remain fixed during optimization. Solid arrows carry signals or targets, and dashed arrows denote updates.

ALFWorld official score equals task success rate. WebShop and ScienceWorld additionally provide graded scores for partial progress, reported separately from binary success. Their evaluators mark success at normalized scores of at least 0.99 and 0.999, respectively.

RL Training. We use Qwen2.5-7B-Instruct (Qwen Team, 2024) with a frozen base model and rank-64 LoRA adapters (Hu et al., 2022) for actor and critic. Both learning rates are 5 × 10<sup>−6</sup>, the global batch size is 8, and the reference KL coefficient is 0.02. PPO uses one rollout per prompt and one optimization epoch per batch. Separate policies train on each benchmark training split, excluding held-out tasks. Table 7 in Appendix C reports data sizes and update counts. Figure 2 shows how detached actor attention changes policy advantages and value targets through intermediate rewards, retaining the PPO losses. We evaluate after each of four data epochs. Training uses VeRL (Sheng et al., 2025), Megatron (Shoeybi et al., 2019), and vLLM (Kwon et al., 2023) on 8×H200 GPUs (140 GB each), with a 32,768-token context and no additional per-turn token or turn-count cap. Attention extraction uses L = 6 layers and 128-token query blocks. Appendix C lists all hyperparameters.

Baselines and Controls. Terminal-only PPO matches ATTNSR in model, rollouts, optimizer, and evaluation. Linear Step Reward, described in Section 4.3, controls for positional shaping on ALFWorld. GRPO uses four rollouts per prompt for one epoch. Alongside the 7B GRPO reference, we compare 3B GRPO with an AttnSR token-advantage variant, as detailed in Appendix F.2. PPO/GRPO optimization budgets differ. We additionally compare adaptations of TARL (Tan et al., 2025), AgentPRM (Xi et al., 2025), and FlowTracer (Dong et al., 2026) on ALFWorld after four epochs of Qwen2.5-7B-Instruct LoRA training, using their final checkpoints. Appendix G gives the details.

Evaluation Protocol. We decode greedily once per held-out task and report full-set metrics. The WebShop validation split is reserved for training-time validation. The main comparison uses epoch 4 without selecting the best test-set checkpoint. Format errors violate the required tool-call protocol and receive outcome reward zero without an additional training penalty. For statistical comparisons, exactinput matching retains all 274 ALFWorld and 500 WebShop tasks and 1,488 of 1,819 ScienceWorld tasks, as described in Appendix E. A two-sided exact McNemar test (McNemar, 1947) assesses binary success differences, and a paired bootstrap (Efron, 1979) with 10,000 resamples estimates 95% percentile intervals for success-rate and official-score differences. These intervals quantify sampling uncertainty over the matched held-out tasks, conditional on the trained policies.

Table 1: Task performance with Qwen2.5-7B-Instruct. (a) Main Result across three benchmarks. (b) ALFWorld success. All values are percentages (%).  
(a) Main Result
<table><tr><td></td><td colspan="3">ALFWORLD</td><td colspan="2">WEBSHOP</td><td colspan="2">SCIENCEWORLD</td></tr><tr><td>Method</td><td>Score</td><td>Seen</td><td>Unseen</td><td>Score</td><td>Success</td><td>Score</td><td>Success</td></tr><tr><td>GRPO†</td><td>69.71</td><td>75.00</td><td>64.18</td><td>67.21</td><td>37.80</td><td>30.19</td><td>16.66</td></tr><tr><td>PPO</td><td>91.24</td><td>92.86</td><td>89.55</td><td>70.74</td><td>40.20</td><td>44.53</td><td>28.59</td></tr><tr><td>PPO + AttnSR</td><td>97.08</td><td>96.43</td><td>97.76</td><td>69.71</td><td>41.00</td><td>46.35</td><td>30.90</td></tr></table>

(b) ALFWorld
<table><tr><td>Method</td><td>Success</td></tr><tr><td>Raw</td><td>26.64</td></tr><tr><td>GRPO†</td><td>69.71</td></tr><tr><td>PPO</td><td>91.24</td></tr><tr><td>TARL</td><td>96.72</td></tr><tr><td>AgentPRM</td><td>91.97</td></tr><tr><td>FlowTracer</td><td>95.62</td></tr><tr><td>PPO + AttnSR</td><td>97.08</td></tr></table>

Raw is the instruction-tuned model before RL. <sup>†</sup>GRPO uses epoch 1 with $n = 4$ , and all other trained policies use epoch 4, with n = 1 for PPO and AttnSR. Optimization budgets differ. ALFWorld score equals success, while other tasks report score and success separately. Panel (b) reuses the canonical GRPO, PPO, and AttnSR results, and Appendix G details the other baselines.

## 4.2 MAIN RESULTS

As shown in Table 1, at epoch 4 ATTNSR shows the clearest benefit on ALFWorld, smaller numerical gains on ScienceWorld, and mixed outcomes on WebShop. ALFWorld success rises from 91.24% to 97.08% (266/274), reducing failures from 24 to 8. ScienceWorld official score increases from 44.53% to 46.35%, and success from 28.59% to 30.90%. WebShop success rises from 40.20% to 41.00%, but official score falls from 70.74% to 69.71%.

Additional ALFWorld Baselines. Table 1(b) compares final success with TARL (96.72%), Agent-PRM (91.97%), and FlowTracer (95.62%). ATTNSR attains the highest observed success, 97.08%, although its margin over TARL is only one task (266 versus 265 of 274). These single-run results are descriptive, and the paired tests below concern the canonical PPO/AttnSR comparison.

Statistical Evidence. Tests use exact-input-matched tasks rather than the full-set estimates in Table 1. On ALFWorld, 20 tasks are solved only by ATTNSR and 4 only by PPO, giving $p =$ 0.002 and a 95% gain interval of [+2.55, +9.49] points. Only ALFWorld remains significant after Bonferroni correction across the three success tests. ScienceWorld gains +2.49 points on 1,488 matched tasks, versus +2.31 on the full set, but its unadjusted p = 0.040 does not survive correction. As reported in Appendix D, its score interval and both WebShop intervals include zero.

## 4.3 ABLATION, CONTROL, AND TRANSFER ANALYSIS

Positional Reward Control. Linear Step Reward tests whether a simple temporal allocation can reproduce the ALFWorld gain. It retains the outcome reward, outcome centering, and budget scale 0.25, but replaces attention allocation with increasing temporal weights and removes confidence gating. For $K _ { i } > 1$ -

$$
w _ { i , k } ^ { \mathrm { l i n } } = \frac { k } { \sum _ { j = 1 } ^ { K _ { i } - 1 } j } , \qquad s _ { i , k } ^ { \mathrm { l i n } } = 0 . 2 5 ( R _ { i } - \bar { R } _ { t } ^ { \mathrm { l i n } } ) w _ { i , k } ^ { \mathrm { l i n } } .\tag{15}
$$

Each run maintains its own outcome baseline. The control reaches 92.34% success, gaining 1.10 points over PPO versus 5.84 points for ATTNSR, as shown in Table 2. The control thus recovers less than one fifth of the gain.

Table 5 in Appendix A shows that Linear Step Reward leads at epoch 1 but trails both methods at epochs 2–3 and ATTNSR at epoch 4. Joint changes to allocation, gating, and realized budgets prevent isolating attention contribution or ruling out other dense rewards. Differences are descriptive because per-task outputs were not retained for paired testing.

Table 2: ALFWorld success (%). ∆ is the gain over each block baseline in percentage points.
<table><tr><td>Method</td><td>Succ. (%)</td><td>∆ (pp)</td></tr><tr><td>PPO</td><td>91.24</td><td></td></tr><tr><td>+Linear Step Reward</td><td>92.34</td><td>+1.10</td></tr><tr><td>+AttnSR</td><td>97.08</td><td>+5.84</td></tr><tr><td>PPO Asynchronous</td><td>59.32</td><td></td></tr><tr><td>+AttnSR</td><td>63.99</td><td>+4.67</td></tr><tr><td>Qwen3-4B-Instruct-2507</td><td></td><td></td></tr><tr><td>PPO</td><td>87.96</td><td></td></tr><tr><td>+AttnSR</td><td>91.24</td><td>+3.28</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td></td><td></td></tr><tr><td>PPO</td><td>79.56</td><td></td></tr><tr><td>+AttnSR</td><td>87.23</td><td>+7.66</td></tr><tr><td>GRPO</td><td>12.41</td><td></td></tr><tr><td>+AttnSR</td><td>20.07</td><td>+7.66</td></tr></table>

Table 3: Full-set trajectory statistics. Steps count environment interactions, and Out. is output in thousands of characters, including observations.
<table><tr><td colspan="3"></td><td colspan="3">Environment steps</td></tr><tr><td>Task</td><td>Method</td><td>Mean</td><td>Med.</td><td>P90</td><td>(k)</td></tr><tr><td>ALFWorld</td><td>GRPO†</td><td>36.93</td><td>11.0</td><td>105.0</td><td>33.44</td></tr><tr><td rowspan="3"></td><td>PPO</td><td>19.28</td><td>9.0</td><td>38.7</td><td>16.91</td></tr><tr><td>+AttnSR</td><td>15.51</td><td>8.0</td><td>31.0</td><td>13.32</td></tr><tr><td>GRPO†</td><td>8.46</td><td></td><td></td><td>11.65</td></tr><tr><td rowspan="3">WebShop</td><td>PPO</td><td>5.93</td><td>6.0 5.0</td><td>12.0 9.0</td><td>8.31</td></tr><tr><td>+AttnSR</td><td>8.03</td><td>7.0</td><td>12.0</td><td>11.08</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">ScienceWorld</td><td>GRPO†</td><td>27.19</td><td>13.0</td><td>74.0</td><td>43.93</td></tr><tr><td>PPO</td><td>30.68</td><td>15.0</td><td>80.0</td><td>45.42</td></tr><tr><td>+AttnSR</td><td>21.51</td><td>10.0</td><td>74.0</td><td>31.99</td></tr></table>

<sup>†</sup>Same GRPO reference and checkpoint budgets as Table 1.

![](images/0f7f3e50d053482facd1f583eefb7abc99923d193a3a489f3a517c45a72c1af8.jpg)

![](images/eebef097b9346e5f17136a3b3fbeb61563c654aebfdfdc91436e141fe63262f8.jpg)

![](images/e42e747aa11e265dfd6fb9616d3a0928a8bcd1584e6b68395ef56db6c83662ab.jpg)  
Figure 3: Environment outcome reward during eight-epoch training (learning rate 1e-6), excluding shaping rewards. These traces supplement the four-epoch held-out comparison.

Supplementary Training Configuration. Single-trajectory asynchronous Qwen2.5-7B-Instruct training gains 4.67 points in Table 2. Changed collection, importance weighting, and critic updates preclude component attribution or direct comparison with canonical PPO, as discussed in Appendix F.

Cross-backbone Check. Qwen3-4B-Instruct-2507 gains 3.28 points, and Qwen2.5-3B-Instruct gains 7.66 points, rising from 218 to 239 successes in Table 2. As shown in Figure 4 in the appendix, the 3B model improves on both seen and unseen tasks at all four checkpoints. Both retain the main PPO and shaping hyperparameters, and the 3B model uses relative-depth attention layers.

Extension to GRPO. GRPO uses four rollouts per prompt for one epoch, matching the rollout count of four-epoch PPO with one rollout per prompt, but with fewer policy updates. 3B GRPO reweights group-normalized outcome advantages across action tokens, preserving each trajectory’s mean advantage with α = 0.25, as detailed in Appendix F.2. In Table 2, success rises from 34 to 55 tasks. Format-error rates of 36.86% and 34.67% and single runs limit interpretation.

## 4.4 LEARNING DYNAMICS AND TASK-LEVEL ANALYSIS

Training and Checkpoints. As reported in Appendix A, ATTNSR scores higher at all four ALF-World and ScienceWorld checkpoints, with the largest gaps at epochs 4 and 3, respectively. WebShop gains at epochs 1–3 reverse slightly at epoch 4. The eight-epoch traces with learning rate 1e-6 in Figure 3 supplement these results, but neither establishes an asymptotic advantage.

Generalization and Task Categories. ALFWorld success rises from 92.86% to 96.43% on seen tasks and 89.55% to 97.76% on unseen tasks, as shown in Table 1. Clean, Heat, Cool, and Pick2 gain 6.89–10.87 points, while Pick and Look remain unchanged, as listed in Table 6 in the appendix. Gains on object-processing and two-object tasks are consistent with benefits on dependent actions, but amount to only three to five successes per family. PPO already reaches 100%/96.77% on Pick/Look, so complexity is confounded with headroom.

## 4.5 INFERENCE-TIME INTERACTION ANALYSIS

Table 3 covers 5,186 PPO/AttnSR and 2,593 GRPO trajectories. Steps and output length do not measure reasoning or generated-token compute. The paired analyses in Appendix E cover only the PPO and AttnSR policies.

Overall Interaction Cost. Mean steps fall from 19.28 to 15.51 on ALFWorld (−19.56%) and from 30.68 to 21.51 on ScienceWorld (−29.88%), with higher success. Serialized output falls by 21.22% and 29.56%, respectively. WebShop rises from 5.93 to 8.03 steps without a clear gain.

What Explains Shorter Trajectories? On 268 ScienceWorld tasks solved by both policies, mean steps fall from 11.06 to 7.40. The 246 joint ALFWorld successes instead increase from 11.19 to 11.99 steps, and the 157 joint WebShop successes from 5.11 to 6.19, as shown in Table 12 in the appendix. The aggregate ALFWorld saving accompanies fewer prolonged failures, since the 24 PPO failures average 103.83 steps versus 109.12 for the 8 AttnSR failures. As discussed in Appendix E, earlier unsuccessful stopping may also reduce aggregate length. These outcome-selected comparisons distinguish shorter solutions from fewer failures but do not establish a causal mechanism.

Scope of the Benefit. WebShop has shorter PPO trajectories (5.93 steps versus 19.28/30.68 on ALFWorld/ScienceWorld), relatively close PPO/GRPO scores (70.74%/67.21%), and no detectable AttnSR gain. This suggests less benefit when shorter trajectories reduce delayed-credit difficulties. However, as noted in Table 1, PPO and GRPO optimization budgets differ, so this comparison establishes neither equivalence nor convergence. Policy-dependent lengths also confound horizon and complexity. Greater benefit on complex, long-horizon tasks remains a hypothesis, not evidence that step rewards generally fail on short tasks.

## 5 RELATED WORK

Agentic RL. Language agents interleave reasoning with actions (Yao et al., 2023; Putta et al., 2024), each spanning multiple tokens. This creates a distinction between token-level optimization and step-level credit assignment. POAD decomposes Bellman backups within and across actions (Wen et al., 2024), ArCHer pairs a turn-level critic with a token-level actor (Zhou et al., 2024), and GiGPO groups steps with matching anchor states to estimate critic-free relative advantages (Feng et al., 2025). These methods address the granularity of value or advantage estimation. ATTNSR instead intervenes at the reward interface. It assigns outcome-conditioned rewards to interaction steps, which then enter the target computation of RL algorithms. In our token-level PPO implementation, these rewards are placed at assistant-step endpoints and propagated through GAE (Schulman et al., 2017; 2016), retaining the critic architecture and optimization losses. We evaluate this reward interface in PPO and separately test attention-based token-advantage reweighting in GRPO.

Credit assignment. Long horizons and partial observability complicate credit assignment from terminal outcomes (Zhang, 2026). VinePPO estimates intermediate values through Monte Carlo rollouts (Kazemnejad et al., 2025), AgentPRM learns step-level promise and progress (Xi et al., 2025), TARL uses an LLM judge for turn-level process supervision (Tan et al., 2025), and SWEET-RL trains an evaluator with privileged information (Zhou et al., 2025). Process supervision also uses step annotations (Lightman et al., 2024), estimated progress (Setlur et al., 2024), or trajectory preferences (Liu et al., 2025). Other approaches use counterfactual baselines (Foerster et al., 2018), hindsight estimators (Tan et al., 2026), learned progress in SPA-RL (Wang et al., 2025a), or gradient saliency in ACPO (Yin et al., 2025). FlowTracer derives token credit from answer-targeted information flow on an attention-induced graph (Dong et al., 2026). ATTNSR draws its allocation prior from actor attention, avoiding additional environment rollouts, a separately trained large reward model, or an attribution backward pass. This signal expresses relevance rather than verified causal importance (Jain & Wallace, 2019; Abnar & Zuidema, 2020). Confidence gating moderates its strength, while the centered outcome determines its sign. Unlike return-preserving redistribution in RUDDER (Arjona-Medina et al., 2019), ATTNSR retains the outcome reward and adds intermediate rewards. It thus changes the return without guaranteeing policy invariance, and we evaluate on the original metrics.

## 6 CONCLUSION

ATTNSR allocates outcome-conditioned rewards using actor attention, without extra rollouts or a learned reward model. With PPO, ALFWorld success rises from 91.2% to 97.1%, exceeding the linear control. ScienceWorld shows numerical gains with fewer interactions, while WebShop has no statistically detectable gain. Supplementary experiments show gains under both PPO and a GRPO token-advantage reweighting variant, with one training run per method. Future work will extend the evaluation to longer context lengths and interaction horizons, examining how attention-based credit allocation handles action dependencies spanning more distant steps.

## ETHICS STATEMENT

This work studies training methods for simulated text-based agents. The experiments use benchmark environments and do not involve human subjects or collection of personal data. As with other agent training methods, improved planning and tool-use capability could be transferred to applications with unintended consequences. Deployment should thus include task-specific safeguards and monitoring.

## DATA AVAILABILITY STATEMENT

We use publicly available ALFWorld, WebShop, and ScienceWorld benchmark environments and do not introduce a new dataset. The evaluation splits, checkpoint-level summaries, and derived tables used in this paper are included with the supplementary artifacts.

## REPRODUCIBILITY STATEMENT

The training configuration, reward equations, evaluation splits, and checkpoint-level results are specified in this paper and its appendix. An anonymized repository will provide the reward extraction and allocation implementation, launch configurations, evaluation summaries, experiment record, significance analysis, and cached per-episode inputs. The main experiments use Qwen2.5-7B-Instruct, LoRA rank 64, VERL/Megatron training, vLLM rollout, and the fixed held-out sets of Section 4.1.

## AI USE STATEMENT

Generative AI tools were used to assist with the organization and language editing of this manuscript, as well as with the development, debugging, and refinement of experimental code. The authors reviewed and verified all AI-assisted content, including technical claims, equations, experimental results, implementation details, and code, against the repository artifacts and experimental records. The authors remain fully responsible for the correctness, integrity, and reproducibility of this work.

## REFERENCES

Samira Abnar and Willem H. Zuidema. Quantifying attention flow in transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 4190–4197, 2020. doi: 10.18653/v1/2020.acl-main.385.

Jose A. Arjona-Medina, Michael Gillhofer, Michael Widrich, Thomas Unterthiner, Johannes Brand stetter, and Sepp Hochreiter. Rudder: Return decomposition for delayed rewards. In Advances in Neural Information Processing Systems, volume 32, 2019.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and´ memory-efficient exact attention with io-awareness. In Advances in Neural Information Processing Systems, volume 35, 2022.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Zhichen Dong, Yang Li, Yuhan Sun, Weixun Wang, Yijia Luo, Zinian Peng, Taiheng Ye, Chao Yang, Wenbo Su, Yu Cheng, Bo Zheng, and Junchi Yan. How does reasoning flow? tracing attention-induced information flow for targeted RL in LLMs. arXiv preprint arXiv:2606.10646, 2026. doi: 10.48550/arXiv.2606.10646. URL https://arxiv.org/abs/2606.10646.

Bradley Efron. Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1): 1–26, 1979.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. Advances in Neural Information Processing Systems, 2025.

Jakob N. Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. Counterfactual multi-agent policy gradients. In Proceedings ofthe Thirty-Second AAAI Conference on Artificial Intelligence, volume 32, 2018. doi: 10.1609/aaai.v32i1.11794.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Sarthak Jain and Byron C. Wallace. Attention is not explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 3543–3556, 2019.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Refining credit assignment in rl training of llms. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, 2024.

Xiaoqian Liu, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li, Junge Zhang, and Jianbin Jiao. Online process reward learning for agentic reinforcement learning. arXiv preprint arXiv:2509.19199, 2025.

Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947.

Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. Agent q: Advanced reasoning and learning for autonomous ai agents. arXiv preprint arXiv:2408.07199, 2024.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Li, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems (EuroSys), 2025.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Hui-Ze Tan, Xiao-Wen Yang, Hao Chen, Jie-Jing Shao, Yi Wen, Yuteng Shen, Weihong Luo, Xiku Du, Lan-Zhe Guo, and Yu-Feng Li. Hindsight credit assignment for long-horizon llm agents. arXiv preprint arXiv:2603.08754, 2026.

Weiting Tan, Xinghua Qu, Ming Tu, Meng Ge, Andy T. Liu, Philipp Koehn, and Lu Lu. Processsupervised reinforcement learning for interactive multimodal tool-use agents. arXiv preprint arXiv:2509.14480, 2025. doi: 10.48550/arXiv.2509.14480. URL https://arxiv.org/abs/ 2509.14480.

Hanlin Wang, Chak Tou Leong, Jiashuo Wang, Jian Wang, and Wenjie Li. Spa-rl: Reinforcing llm agents via stepwise progress attribution. arXiv preprint arXiv:2505.20732, 2025a.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Cotˆ e, and Prithviraj Ammanabrolu. Scienceworld: Is ´ your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, 2022. doi: 10.18653/v1/2022.emnlp-main.775.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, et al. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025b.

Muning Wen, Ziyu Wan, Weinan Zhang, Jun Wang, and Ying Wen. Reinforcing language agents via policy optimization with action decomposition. arXiv preprint arXiv:2405.15821, 2024.

Zhiheng Xi, Chenyang Liao, Guanyu Li, Yajie Yang, Wenxiang Chen, Zhihao Zhang, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, Tao Ji, Tao Gui, Qi Zhang, and Xuanjing Huang. AgentPRM: Process reward models for LLM agents via step-wise promise and progress. arXiv preprint arXiv:2511.08325, 2025. doi: 10.48550/arXiv.2511.08325. URL https://arxiv. org/abs/2511.08325.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. In Advances in Neural Information Processing Systems, volume 35, pp. 20744–20757, 2022.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Junxi Yin, Haisen Luo, Zhenyu Li, Yihua Liu, Dan Liu, Zequn Li, and Xiaohang Xu. Pinpointing crucial steps: Attribution-based credit assignment for verifiable reinforcement learning. arXiv preprint arXiv:2510.08899, 2025.

Chenchen Zhang. From reasoning to agentic: Credit assignment in reinforcement learning for large language models. arXiv preprint arXiv:2604.09459, 2026.

Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. Archer: Training language model agents via hierarchical multi-turn rl. In Proceedings of the 41st International Conference on Machine Learning, 2024.

Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar, and Xian Li. Sweet-rl: Training multi-turn llm agents on collaborative reasoning tasks. arXiv preprint arXiv:2503.15478, 2025.

## A FULL CHECKPOINT RESULTS

Table 4: Official scores (%) at four checkpoints from the canonical held-out evaluations. ALFWorld official score equals binary success.
<table><tr><td>Task</td><td>Metric</td><td>Epoch 1</td><td>Epoch 2</td><td>Epoch 3</td><td>Epoch 4</td></tr><tr><td>ALFWORLD PPO</td><td>score (%)</td><td>36.50</td><td>69.34</td><td>89.42</td><td>91.24</td></tr><tr><td>ALFWORLD ATTNSR</td><td>score (%)</td><td>37.59</td><td>70.44</td><td>91.24</td><td>97.08</td></tr><tr><td>WEBSHOP PPO</td><td>score (%)</td><td>63.72</td><td>67.53</td><td>66.71</td><td>70.74</td></tr><tr><td>WEBSHOP ATTNSR</td><td>score (%)</td><td>65.74</td><td>68.35</td><td>69.58</td><td>69.71</td></tr><tr><td>SCIENCEWORLD PPO</td><td>score (%)</td><td>22.89</td><td>31.49</td><td>34.59</td><td>44.53</td></tr><tr><td>SCIENCEWORLD ATTNSR</td><td>score (%)</td><td>27.12</td><td>35.57</td><td>41.15</td><td>46.35</td></tr></table>

Table 5: ALFWorld success (%) across checkpoints for the canonical positional control. All checkpoints use the full 274-task held-out set.
<table><tr><td>Method</td><td>Epoch 1</td><td>Epoch 2</td><td>Epoch 3</td><td>Epoch 4</td></tr><tr><td>PPO</td><td>36.50</td><td>69.34</td><td>89.42</td><td>91.24</td></tr><tr><td>Linear Step Reward</td><td>43.43</td><td>59.85</td><td>79.20</td><td>92.34</td></tr><tr><td>AttnSR</td><td>37.59</td><td>70.44</td><td>91.24</td><td>97.08</td></tr></table>

## B ALFWORLD CATEGORY-LEVEL RESULTS

Table 6: Fine-grained ALFWorld task-family success at epoch 4 on the full held-out set. Columns use the compact family names Pick (pick and place simple), Look (look at object in light), Clean, Heat, Cool, and Pick2 (pick two objects and place). Counts sum to 274 tasks.
<table><tr><td></td><td>Pick</td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>Overall</td></tr><tr><td>n</td><td>59</td><td>31</td><td>58</td><td>39</td><td>46</td><td>41</td><td>274</td></tr><tr><td>PPO (%)</td><td>100.00</td><td>96.77</td><td>86.21</td><td>89.74</td><td>86.96</td><td>87.80</td><td>91.24</td></tr><tr><td>AttnSR (%)</td><td>100.00</td><td>96.77</td><td>93.10</td><td>100.00</td><td>97.83</td><td>95.12</td><td>97.08</td></tr><tr><td>∆(pp)</td><td>0.00</td><td>0.00</td><td>+6.89</td><td>+10.26</td><td>+10.87</td><td>+7.32</td><td>+5.84</td></tr></table>

## C IMPLEMENTATION DETAILS

The selected layers are 8, 10, 12, 14, 16, and 18. We use full causal-prefix softmax attention and do not renormalize within the step submatrix. AA mass targets historical assistant tokens, while AO mass targets the observation paired with the historical step. Step rewards are placed at assistant-step end tokens, and tool observations are excluded from the actor loss. The main implementation uses query blocks of 128 tokens and reconstructs tensor/context-parallel shards before graph aggregation. Format errors are assigned the ordinary outcome reward of zero rather than an extra penalty.

Each benchmark is trained independently from the instruction-tuned checkpoint. Table 7 reports the training manifests and collected-batch updates. Epoch lengths use complete batches of eight, dropping the incomplete final batch when present. Training reward is binary success on ALFWorld, the official score in [0, 1] on WebShop, and the final ScienceWorld score clipped to [0, 100] and divided by 100. The native ScienceWorld step-reward increments are not supplied to PPO. Benchmark-specific expert-action and search bonuses are disabled.

## D PAIRED SIGNIFICANCE ANALYSIS

Table 10 reports the full uncertainty analysis summarized in Section 4.2. Both policies decode greedily on identical held-out tasks, so the final-checkpoint comparison is paired at the task level

Table 7: Canonical training-set sizes and update budgets, per method. One update uses one collected batch of eight trajectories.
<table><tr><td>Benchmark</td><td>Training tasks</td><td>Updates/epoch</td><td>Updates at epoch 4</td></tr><tr><td>ALFWorld</td><td>3,553</td><td>444</td><td>1,776</td></tr><tr><td>WebShop</td><td>10,587</td><td>1,323</td><td>5,292</td></tr><tr><td>ScienceWorld</td><td>3,592</td><td>449</td><td>1,796</td></tr></table>

Table 8: Framework and model configuration for the canonical runs.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Base model</td><td>Qwen2.5-7B-Instruct</td></tr><tr><td>Training framework</td><td>VeRL with Megatron-based training</td></tr><tr><td>Rollout engine</td><td>vLLM</td></tr><tr><td>Hardware</td><td>8×H200 GPUs; 140 GB memory per GPU</td></tr><tr><td>Fine-tuning</td><td>LoRA rank 64; frozen base model</td></tr><tr><td>Trainable modules</td><td>Actor and critic LoRA adapters</td></tr><tr><td>Message format</td><td>Qwen standard chat format with JSON tool calls</td></tr><tr><td>Maximum context</td><td>32,768 tokens</td></tr><tr><td>Additional length limits</td><td>No per-turn token cap or turn-count cap</td></tr><tr><td>Training parallelism</td><td>Megatron TP=4, CP=2</td></tr><tr><td>Rollout parallelism</td><td>Tensor parallelism 4</td></tr></table>

over the exact-input-matched episodes defined in Appendix E. Wins and losses count tasks solved by exactly one policy, and the McNemar test is applied to those discordant pairs only. Bootstrap intervals resample the paired per-task differences 10,000 times.

## E INFERENCE-TRAJECTORY DIAGNOSTICS

Source and measurement. We re-read the six final-checkpoint generations/ openai.jsonl files referenced by the canonical evaluation CSVs. The analysis includes 274 ALFWorld, 500 WebShop, and 1,819 ScienceWorld episodes per method. Recomputed means for official score, success, environment steps, parsed calls, serialized output length, and trace-parser events agree with the evaluation summaries. Per-episode numeric extracts and source-file hashes are retained with the analysis script, with no new training or evaluation.

Table 3 additionally reports the three historical one-epoch GRPO evaluations on the same full-set sizes (2,593 trajectories). We extract their environment steps and serialized output lengths using the same definitions, and verify score, success, format-error rate, natural termination, and mean steps against the original GRPO summaries. Source hashes and numeric extracts are retained separately. These GRPO statistics are excluded from the paired and significance analyses below.

Environment steps use the evaluator env steps field. Tool calls are counted from openai tool calls. They are parseable calls, not necessarily valid or successful environment actions. In these exports, parsed assistant tool-call turns equal the call count and are not reported as an independent measure of reasoning depth. Output length counts characters in the serialized output, including tool-call syntax and observations. We do not interpret it as model-generated token count, internal reasoning length, or inference FLOPs. Medians and P90 values are computed from the episode-level observations, and P90 uses linear interpolation.

Outcome-conditioned and paired comparisons. Table 11 separates successful and unsuccessful episodes using the evaluator binary success flag. These two groups have different task membership under the two policies. The paired analysis in Table 12 additionally requires the same evaluation row, an exact input-text hash match, and an input unique within each benchmark set. This retains all 274 ALFWorld and 500 WebShop pairs. For ScienceWorld, 331 episodes are excluded because inputs differ or are non-unique, leaving 1,488 eligible pairs and 268 joint successes. No matching is inferred from row position alone.

On ALFWorld, the verified pairs include 20 tasks solved only by AttnSR, 4 only by PPO, 246 by both, and 4 by neither. This yields the observed net gain of 16 successes without claiming that every previously solved task remains solved. The corresponding WebShop counts are 48, 44, 157, and 251. Within the restricted ScienceWorld eligible set, they are 173, 136, 268, and 911. These counts should not replace the full-test success totals. Successful-subset comparisons are descriptive and post-selected on outcomes of both policies, not estimates of a causal treatment effect.

Table 9: PPO, evaluation, and AttnSR hyperparameters used for the main results.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Batch size</td><td>8</td></tr><tr><td>PPO update epochs</td><td>1 1</td></tr><tr><td>Rollouts per prompt</td><td></td></tr><tr><td>Advantage estimator</td><td>GAE with trajectory-level agent advantage</td></tr><tr><td>Actor / critic learning rate</td><td> $5 \times 1 0 ^ { - 6 } / \dot { 5 } \times \dot { 1 0 } ^ { - 6 }$ </td></tr><tr><td>Reference-policy KL coefficient</td><td>0.02</td></tr><tr><td>Checkpoint frequency</td><td>Once per data epoch</td></tr><tr><td>Decoding</td><td>Greedy; one rollout per held-out task</td></tr><tr><td>Evaluation sets</td><td>ALFWorld 274; WebShop 500; ScienceWorld 1,819</td></tr><tr><td>Attention layers</td><td>8, 10, 12, 14, 16, 18</td></tr><tr><td>Attention aggregation</td><td>Full causal-prefix softmax; no step-submatrix renormalization; AA/AO = 0.5/0.5</td></tr><tr><td>Query block size</td><td>128 tokens</td></tr><tr><td>Step-reward budget scale</td><td>0.25</td></tr><tr><td>Global reward mean</td><td>First-batch mean initialization; stop-gradient EMA coefficient 0.99</td></tr><tr><td>Hop decay Temporal-bias correction</td><td>0.85 Weight —0.25; distance scale κ = 4</td></tr><tr><td>Target attention mass / focus</td><td>0.20 / 0.15</td></tr><tr><td></td><td></td></tr><tr><td>Reward placement</td><td>Final token of each assistant step; outcome reward on final step</td></tr></table>

Table 10: Paired evaluation-set uncertainty for the final-checkpoint comparison, computed over exact-input-matched episodes. Wins and losses count tasks solved by exactly one policy, and p is the unadjusted two-sided exact McNemar test on those discordant pairs. Intervals are 10,000-resample percentile bootstrap intervals for the mean paired difference (AttnSR − PPO) in points. ALFWorld official score equals binary success. Intervals quantify sampling uncertainty over matched tasks.
<table><tr><td>Task</td><td>Pairs</td><td>Wins</td><td>Losses</td><td>McNemar p</td><td>∆ Success (pp)</td><td>∆ Score (pp)</td></tr><tr><td>ALFWorld</td><td>274</td><td>20</td><td>4</td><td>0.002</td><td>+5.84 [+2.55, +9.49]</td><td>+5.84 [+2.55, +9.49]</td></tr><tr><td>WebShop</td><td>500</td><td>48</td><td>44</td><td>0.755</td><td>+0.80 [-3.00, +4.40]</td><td>-1.04 [-3.36, +1.20]</td></tr><tr><td>ScienceWorld</td><td>1488</td><td>173</td><td>136</td><td>0.040</td><td>+2.49 [+0.20, +4.70]</td><td>+1.85 [-0.37, +4.06]</td></tr></table>

Tool use and termination. Table 13 distinguishes exported trace-parser events from the evaluator terminal format-error flag. Most parser events are unmatched call/response pairs, not malformed JSON. Specifically, ALFWorld PPO has 23 missing-response events, while AttnSR has 8 missingresponse and 4 orphan-response events. ScienceWorld PPO has 426 missing-response and 10 callparsing events, while AttnSR has 245 missing-response events and no call-parsing events. WebShop has zero events under PPO and 3 missing-response events under AttnSR. An episode may contain multiple events, so the event count per episode differs from the fraction of affected episodes. We do not equate these exported trace diagnostics with runtime tool-execution failures.

Natural termination is also not a success indicator. On ScienceWorld its rate increases from 12.15% to 18.42%, so earlier unsuccessful stopping may contribute to the full-set length reduction, which is why we separately report jointly successful trajectories. Exact adjacent-call repetition does not uniformly decrease either, as the pooled ScienceWorld repetition fraction increases from 15.35% to 16.30%. Repeated commands can be necessary in stateful environments and are not labeled invalid or redundant without inspecting the intervening state. On WebShop, the parsed-call totals are 824 search and 2,143 click calls for PPO, versus 1,088 search and 2,925 click calls for AttnSR. This confirms that the higher interaction count involves both search and navigation rather than one tool alone.

## F SCOPE OF SUPPLEMENTARY RUNS

Table 1 includes a one-epoch GRPO reference with four rollouts per prompt. Table 2 reports the canonical positional control, asynchronous and Qwen3 backbone comparisons, and the Qwen2.5-3B

Table 11: Full-set trajectory lengths on each benchmark. Success and failure groups differ across policies and are not matched task subsets. Table 12 provides the restricted paired comparison.
<table><tr><td colspan="3"></td><td rowspan="2">Successful</td><td colspan="2">Unsuccessful</td></tr><tr><td>Task</td><td>Method</td><td>n</td><td>Mean steps</td><td>n Mean steps</td></tr><tr><td rowspan="2">ALFWorld</td><td>PPO</td><td>250</td><td>11.16</td><td>24</td><td>103.83</td></tr><tr><td>AttnSR</td><td>266</td><td>12.69</td><td>8</td><td>109.12</td></tr><tr><td rowspan="2">WebShop</td><td>PPO</td><td>201</td><td>5.22</td><td>299</td><td>6.41</td></tr><tr><td>AttnSR</td><td>205</td><td>6.62</td><td>295</td><td>9.00</td></tr><tr><td rowspan="2">ScienceWorld</td><td>PPO</td><td>520</td><td>11.32</td><td>1299</td><td>38.43</td></tr><tr><td>AttnSR</td><td>562</td><td>8.00</td><td>1257</td><td>27.55</td></tr></table>

Table 12: Environment steps on exact-input-matched episodes where both policies succeed. Eligible pairs require the same evaluation row and identical input text, unique within each benchmark set. All ALFWorld and WebShop episodes qualify, while ScienceWorld retains 1,488 of 1,819 episodes, excluding 331 with different or non-unique input text. Results are descriptive.
<table><tr><td>Task</td><td>Eligible</td><td>Both succeed</td><td>PPO</td><td>AttnSR</td><td>∆ steps</td></tr><tr><td>ALFWorld</td><td>274</td><td>246</td><td>11.19</td><td>11.99</td><td>+0.80</td></tr><tr><td>WebShop</td><td>500</td><td>157</td><td>5.11</td><td>6.19</td><td>+1.08</td></tr><tr><td>ScienceWorld</td><td>1488</td><td>268</td><td>11.06</td><td>7.40</td><td>-3.66</td></tr></table>

PPO and GRPO comparisons. The supplementary results are interpreted within their respective settings and are not pooled with the primary comparison for component attribution.

Asynchronous configuration. Both variants use Qwen2.5-7B-Instruct, rank-64 actor and critic LoRA, batch size 8, actor/critic learning rates of $5 \times 1 0 ^ { - 6 }$ , a 32,768-token context, and four data epochs on ALFWorld. This configuration uses asynchronous collection with a replay queue and rollout-policy token log-probabilities. Direct token importance sampling masks ratios outside [0.7, 6.0]. Observation tokens are skipped in GAE, with actor $\lambda = 1 - 1 / ( 1 . 5 n _ { i } )$ for $n _ { i }$ trainable assistant tokens and critic $\lambda = 1$ . Two critic passes are performed per actor pass, and critic LoRA is restricted to MLP projections, with a trainable value head. The critic is initialized from the instruction-tuned model without additional value pretraining. The AttnSR variant retains the canonical attention layers and shaping hyperparameters. Thus the comparison changes more than collection timing relative to canonical PPO.

Qwen3 configuration. The paired launch configuration uses Qwen3-4B-Instruct-2507 with the same main hyperparameters as the canonical study, namely rank-64 LoRA, actor/critic learning rates of $5 \times 1 0 ^ { - 6 } ;$ , batch and PPO minibatch size 8, one rollout per prompt, one PPO epoch per batch, reference KL coefficient 0.02, and a 32,768-token context. Training uses TP=4 and CP=2, and rollout uses TP=4. AttnSR retains the selected layers and reward settings. Four ALFWorld data epochs give 1,776 updates. Greedy evaluation of the epoch-4 policies on all 274 held-out tasks yields 241 successes for PPO and 250 for AttnSR. This is supplementary evidence across backbones, not a test isolating model size or individual reward components.

## F.1 QWEN2.5-3B PPO CHECKPOINTS

Both policies start from Qwen2.5-3B-Instruct with a frozen base and rank-64 actor/critic LoRA. The main settings are retained, including actor/critic learning rates $5 \times 1 0 ^ { - 6 }$ , batch and minibatch size 8, one rollout per prompt, one PPO epoch per batch, KL coefficient 0.02, and a 32,768-token context. Four data epochs give 1,776 updates, with checkpoints every 444 updates. AttnSR retains the reward budget scale 0.25, confidence targets, EMA, and attention aggregation settings in Table 9. To preserve relative depth in the 36-layer model, each zero-based reference index $l \in \{ 8 , 1 0 , 1 2 , 1 4 , 1 \dot { 6 } , 1 8 \}$ is mapped to $\lfloor 3 5 l / 2 7 + 0 . 5 \rfloor$ , yielding {10, 13, 16, 18, 21, 23}.

Figure 4 reports greedy evaluation of every checkpoint on all 274 held-out tasks. Overall success for PPO/AttnSR is 4.74/22.26%, 33.21/61.31%, 61.68/72.26%, and 79.56/87.23% at epochs 1–4.

Table 13: Tool-trace and termination diagnostics at epoch 4. Parse/ep. counts exported trace-parser events, and Parse% is the fraction of episodes with any such event. Format% and Natural% are the evaluator terminal flags. Repeat% is the pooled fraction of adjacent parsed-call pairs with identical function names and arguments, and repetition need not imply an invalid or redundant action.
<table><tr><td>Task</td><td>Method</td><td>Parse/ep.</td><td>Parse%</td><td>Format%</td><td>Natural%</td><td>Repeat%</td></tr><tr><td rowspan="2">ALFWorld</td><td>PPO</td><td>0.084</td><td>8.39</td><td>0.00</td><td>0.00</td><td>0.04</td></tr><tr><td>AttnSR</td><td>0.044</td><td>3.28</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="2">WebShop</td><td>PPO</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.04</td></tr><tr><td>AttnSR</td><td>0.006</td><td>0.60</td><td>0.00</td><td>0.00</td><td>0.03</td></tr><tr><td rowspan="2">ScienceWorld</td><td>PPO</td><td>0.240</td><td>23.97</td><td>0.60</td><td>12.15</td><td>15.35</td></tr><tr><td>AttnSR</td><td>0.135</td><td>13.47</td><td>0.22</td><td>18.42</td><td>16.30</td></tr></table>

The final checkpoint is reported without best-checkpoint selection. At epoch 4, mean environment steps fall from 32.79 to 22.44, while format errors increase from 0.00% to 2.55%, so higher success does not imply uniform improvement in protocol compliance. Each method has one training run, so the curves do not estimate training-seed variability. The baseline and AttnSR source trees also contain critic/value-head compatibility differences beyond attention reward code, so these runs are a descriptive cross-backbone check rather than an isolated reward-component ablation.

(a) Overall (n = 274)  
![](images/6d55fa8a69f9c1f27cb75b4f12f140c7eb472c4e83a85f1921ede4af0cc66db0.jpg)

(b) Seen (n = 140)  
![](images/0774beb10eb9d9d8fda7fabd00c403c00dde1a3e96573598c2370a9d94a6a7d5.jpg)

(c) Unseen (n = 134)  
![](images/8fe3faee7c0d0c77b015d11e26d7fdd49849e0e7208a4945a733aa657b52facf.jpg)  
Figure 4: Qwen2.5-3B-Instruct held-out ALFWorld success after each of four PPO training epochs, shown overall (274 tasks), seen (140), and unseen (134). Each point is a full-set greedy evaluation from one training run per method. Lines connect checkpoints without smoothing, and no estimate of training-seed uncertainty is available.

## F.2 QWEN2.5-3B GRPO EXTENSION

The two GRPO policies independently start from Qwen2.5-3B-Instruct and train for one data epoch with four rollouts per prompt. Both use a frozen base, LoRA rank/alpha 64/64, actor learning rate $5 \times 1 0 ^ { - 6 }$ , prompt batch/minibatch size 8, actor KL loss coefficient 0.02, and no critic. Training uses $\mathrm { T P } { = } 4 , \mathrm { C P } { = } 2$ , rollout TP=4, and a 32,768-token context with prompt/response limits 7,168/25,600. The final checkpoint is saved at update 444. Evaluation uses greedy decoding on the same 274 tasks, with no training-time attention signal. Four rollouts for one epoch match the nominal trajectory count of one rollout for four PPO epochs, but not the update count, generated tokens, or runtime.

This experiment uses the initial token-advantage reweighting AttnSR variant. Let A be the ordinary GRPO outcome advantage computed from original environment rewards within each prompt group. Intermediate-step weights are broadcast to their action tokens and normalized to mean one over valid tokens with positive weights, accounting for step lengths. Set $\widetilde { w } _ { i , u } = 1$ at other valid action tokens, including the final step, unparsed positions, and zero-weight positions. The actor uses

$$
A _ { i , u } ^ { \prime } = A _ { i } \big [ ( 1 - \alpha ) + \alpha \widetilde { w } _ { i , u } \big ] , \qquad \alpha = 0 . 2 5 .\tag{16}
$$

Observations and padding are masked. This preserves each trajectory’s mean advantage and its sign, while varying its magnitude across steps, and equal-reward groups retain zero advantage. The collector uses the same relative layers and attention allocation as the 3B PPO run. However, positive trajectory confidence is a common factor that cancels during token-weight normalization, and only zero confidence disables redistribution. Thus this variant does not retain PPO’s continuous confidence modulation, outcome EMA, or intermediate rewards and GAE.

Both methods use the same GRPO implementation overlay, with attention reweighting enabled only in the AttnSR arm. Table 2 shows final success of 34/274 for GRPO and 55/274 for GRPO+AttnSR. Seen success rises from 15.00% to 22.14% (21 to 31 of 140 tasks), and unseen success from 9.70% to 17.91% (13 to 24 of 134 tasks). Format-error counts remain high at 101/274 and 95/274. These single-run results are descriptive, with no repeated-seed or paired significance analysis.

## G ADDITIONAL ALFWORLD BASELINES

Our ALFWorld adaptations of TARL (Tan et al., 2025), AgentPRM (Xi et al., 2025), and Flow-Tracer (Dong et al., 2026) are trained independently from Qwen2.5-7B-Instruct with LoRA for four data epochs. We evaluate the final checkpoints at update 1,776 on all 274 held-out ALFWorld tasks (140 seen and 134 unseen), using one greedy rollout per task and a 32,768-token context. The implementation variants recorded in the experiment inventory are TARL-Turn, AgentPRM-Progress, and FlowTracer-PPO, respectively. Table 1(b) retains the canonical PPO and AttnSR results as references, and no paired significance comparison with the three additional methods is claimed.

Tables 14 and 15 report the final-checkpoint details. Success counts are 265, 252, and 262 for TARL, AgentPRM, and FlowTracer. All three have zero terminal format errors, and AgentPRM has eight naturally terminated episodes. These diagnostic fields are distinct from task success and do not indicate whether training completed normally. The small number of failed tasks (9, 22, and 12) makes failure-conditioned step means sensitive to individual long trajectories. Category-level counts and outcome-conditioned lengths are descriptive, not causal evidence about credit assignment.

Text lengths are obtained by retokenizing saved evaluation text with Qwen2TokenizerFast, with add special tokens=False. Full trajectories include tool observations, while assistant-only counts exclude tool replies. These measurements do not recover the original sampled token IDs and should not be interpreted as exact generation costs. All results are from one training run per method, and no estimate of training-seed variability is available.

## H NOTATION

Table 16 lists the symbols of Sections 2–3. Episode index i is omitted where unambiguous.

Table 14: Task outcomes of the additional ALFWorld baselines at the final checkpoint (update 1,776). (a) Successful tasks, with success rates (%) in parentheses. Format and termination columns count episodes. (b) Successful tasks by category, with task counts in the header. Each row of (b) sums to the overall success count in (a).  
(a) Overall success and termination
<table><tr><td>Method</td><td>Overall / 274</td><td>Seen / 140</td><td>Unseen / 134</td><td>Format errors</td><td>Natural term.</td></tr><tr><td>TARL</td><td>265 (96.72)</td><td>135 (96.43)</td><td>130 (97.01)</td><td>0</td><td>0</td></tr><tr><td>AgentPRM</td><td>252 (91.97)</td><td>130 (92.86)</td><td>122 (91.04)</td><td>0</td><td>8</td></tr><tr><td>FlowTracer</td><td>262 (95.62)</td><td>135 (96.43)</td><td>127 (94.78)</td><td>0</td><td>0</td></tr></table>

(b) Successful tasks by category
<table><tr><td>Method</td><td>Look (31)</td><td>Pick (59)</td><td>Clean (58)</td><td>Cool (46)</td><td>Heat (39)</td><td>Pick2 (41)</td></tr><tr><td>TARL</td><td>30</td><td>58</td><td>54</td><td>44</td><td>39</td><td>40</td></tr><tr><td>AgentPRM</td><td>31</td><td>56</td><td>54</td><td>40</td><td>34</td><td>37</td></tr><tr><td>FlowTracer</td><td>30</td><td>57</td><td>54</td><td>45</td><td>36</td><td>40</td></tr></table>

Table 15: Trajectory lengths of the additional ALFWorld baselines at the final checkpoint. (a) Environment steps and mean serialized output length in thousands of characters. Successful and failed subsets differ across policies. (b) Retokenized evaluation-text lengths. Full trajectories include tool observations, while assistant-only text excludes tool replies. These are counts from saved text, not original sampled token IDs or exact generation costs.  
(a) Environment steps and output length
<table><tr><td colspan="5">Environment steps</td><td rowspan="2">Output (k chars)</td></tr><tr><td>Method</td><td>Mean</td><td>Median</td><td>Success mean</td><td>Failure mean</td></tr><tr><td>TARL</td><td>13.70</td><td>8</td><td>10.83</td><td>98.33</td><td>11.77</td></tr><tr><td>AgentPRM</td><td>14.18</td><td>8</td><td>10.33</td><td>58.18</td><td>13.89</td></tr><tr><td>FlowTracer</td><td>19.06</td><td>14</td><td>16.06</td><td>84.42</td><td>17.26</td></tr></table>

(b) Retokenized text length
<table><tr><td rowspan="2">Method</td><td colspan="3">Full trajectory tokens</td><td colspan="3">Assistant-only tokens</td></tr><tr><td>Mean</td><td>Median</td><td>P90</td><td>Mean</td><td>Median</td><td>P90</td></tr><tr><td>TARL</td><td>3,983</td><td>2,656</td><td>7,824</td><td>348</td><td>214</td><td>641</td></tr><tr><td>AgentPRM</td><td>4,596</td><td>2,673</td><td>9,493</td><td>648</td><td>306</td><td>1,303</td></tr><tr><td>FlowTracer</td><td>5,837</td><td>4,203</td><td>10,186</td><td>480</td><td>348</td><td>768</td></tr></table>

Table 16: Notation. Hyperparameter values refer to the canonical AttnSR runs.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $i , t$ </td><td>episode index; collected training-batch index</td></tr><tr><td> $K _ { i }$ </td><td>number of assistant steps in episode i</td></tr><tr><td> $k , j$ </td><td>step indices,  $1 \leq k < \bar { j } \leq K _ { i }$ </td></tr><tr><td> $S _ { i , k }$ </td><td>interaction step: assistant message and observation, when present</td></tr><tr><td> $A _ { i , k } , O _ { i , k }$ </td><td>assistant message; associated environment observation</td></tr><tr><td> $\mathcal { T } _ { i , k }$ </td><td>token indices in the assistant span of step k</td></tr><tr><td> $u$ </td><td>token index into the serialized trajectory</td></tr><tr><td> $e _ { i , k }$ </td><td>endpoint: last trainable token of step k</td></tr><tr><td> $r _ { i , u }$ </td><td>reward written at token u</td></tr><tr><td> $R _ { i } , \bar { R } _ { t }$ </td><td>outcome reward; EMA updated using batch t before shaping</td></tr><tr><td> $M _ { t }$ </td><td>mean outcome reward in batch t; initializes  ${ \bar { R } } _ { 1 } = M _ { 1 }$ </td></tr><tr><td> $l , \mathcal { L } , L$ </td><td>layer index; selected layers;  $L = | \mathcal { L } | = 6$ </td></tr><tr><td> $G _ { i , \mathrm { A A } } ^ { ( l ) } , G _ { i , \mathrm { A O } } ^ { ( l ) }$ </td><td>step attention to historical assistant tokens; to paired observations</td></tr><tr><td> $G _ { i } ^ { ( l ) }$ </td><td> $0 . 5 G _ { i , \mathrm { A A } } ^ { ( l ) } + 0 . 5 G _ { i , \mathrm { A O } } ^ { ( l ) }$  mixed step attention,</td></tr><tr><td> $a _ { i , k } ^ { ( l ) }$ </td><td>backtracked relevance of step k at layer l, defined in Eq. 3</td></tr><tr><td> $\Delta _ { i , k } , d _ { i , k }$ </td><td>steps from k to the terminus,  $K _ { i } - k ;$  proximity feature  $\exp ( - ( \Delta _ { i , k } - 1 ) / \kappa )$ </td></tr><tr><td> $p _ { i , k } ^ { ( i ) } , p _ { i , k }$ </td><td>per-layer allocation; geometric-consensus allocation, defined in Eq. 5</td></tr><tr><td> $m _ { i } ^ { ( l ) } , H _ { i } ^ { ( l ) }$ </td><td>terminal-step covered mass; normalized entropy of  $p _ { i } ^ { ( l ) }$ </td></tr><tr><td> $C _ { i }$ </td><td>confidence gate,  $C _ { i } \in [ 0 , 1 ]$ </td></tr><tr><td> $B _ { i } , s _ { i , k }$ </td><td>signed shaping budget; step reward  $B _ { i } p _ { i , k } .$  , defined in  $\mathrm { E q } . 1 3$ </td></tr><tr><td> $\hat { A } _ { u } , \widetilde { A } _ { u } , y _ { u }$ </td><td>unnormalized advantage; normalized actor advantage; critic target</td></tr><tr><td> $V _ { u } ^ { \mathrm { o l d } } , V _ { \phi }$ </td><td>pre-update critic value; critic network</td></tr><tr><td> $\mathrm { s g , n o r m , T V }$ </td><td>stop-gradient; standardization; total-variation distance</td></tr><tr><td> $\rho = 0 . 8 5$ </td><td>hop decay in the backtracking recurrence</td></tr><tr><td> $\eta = - 0 . 2 5 , \kappa = 4$ </td><td>recency-correction weight and proximity decay</td></tr><tr><td> $T = 1 , \epsilon = 1 0 ^ { - 1 2 }$ </td><td></td></tr><tr><td> $\beta = 0 . 2 5$ </td><td>allocation softmax temperature; lower bound before logarithms budget scale, bounding  $| B _ { i } |$ </td></tr></table>