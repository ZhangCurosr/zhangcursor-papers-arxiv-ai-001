# Depth as a Candidate Source: Portfolio Allocation for Recurrent Test-Time Scaling in Large Language Models

Chaoyuan Hao, Wentao Yue, Yanyan Wu, Boyuan Gao, Tianyou Lai Maolin Wang, Yongxin Ni, Hanwen Du, Youhua Li

## Abstract

Recurrent latent reasoning in large lanugage models (LLMs) scales test-time computation by increasing recurrent depth, naturally framing inference as a problem of deciding when to stop. We identify a previously overlooked limitation of this single-depth view through systematic preliminary analysis: deeper recurrence can lose correct solutions found at shallower depths, while depth-specific success sets remain broadly non-nested even after controlling for marginal accuracy. This suggests that depth determines not only computation cost, but also access to distinct candidate solutions. Motivated by this finding, we propose Depth Portfolio Allocation (DPA), which reformulates recurrent test-time scaling as budgeted allocation across depth-specific candidate sources. DPA supports both fixed portfolios and feedback-guided sequential allocation under a shared inference budget. Experiments on GSM8K, SVAMP, MultiArith, and MBPP show that mixed-depth portfolios expand oracle coverage under matched candidate counts and nominal costs, while feedback-guided allocation generally matches or improves uniform allocation and can reduce computation in high-budget settings. Code and data are avaible at : https://anonymous.4open.science/r/ LLM-latent-B7BC/.

## 1 Introduction

Test-time scaling (TTS) in LLMs improves performance by allocating additional inference-time computation without changing model parameters (Snell et al., 2025). Existing approaches typically spend this compute in token space, for example through longer reasoning traces (Wei et al., 2022), multiple candidate samples (Wang et al., 2023), or verifier-based selection (Cobbe et al., 2021). Recurrent latent reasoning provides a different form of TTS: instead of generating longer explicit traces, the model repeatedly applies recurrent computation to its hidden representations, allowing answers to be decoded after different numbers of recurrent updates (Geiping et al., 2025). The recurrent depth therefore directly controls how much latent computation is spent on a question, making depth selection a natural inference-time decision.

A common view of recurrent test-time scaling is consequently to ask how much recurrent computation to use, or when computation should stop (Geiping et al., 2025; Graves, 2016; Elbayad et al., 2020). This single-trajectory formulation is most natural when additional recurrence progressively refines, or at least preserves, solutions available at shallower depths. Recent work on looped language models, however, shows that performance can peak and then degrade as recurrence continues (Yang et al., 2026), suggesting that deeper computation need not provide a monotonic refinement. If so, selecting a single stopping depth may overlook useful at other depths solutions that emerge elsewhere along the recurrent trajectory.

We examine this issue at the level of individual questions and candidate solutions through the preliminary analysis in Section 3. Across a large set of recurrent reasoning trajectories, different questions favor different depths, deeper recurrence does not always preserve earlier correct answers, and depth-specific success sets remain broadly nonnested even after controlling for marginal accuracy. These findings suggest that depth determines not only computational cost, but also access to partially distinct candidate solutions. Recurrent inference is therefore not only a question of where to stop, but also of how to allocate a finite budget across depth-specific candidate sources.

## Research Question

Does a single stopping depth miss correct solutions available at other depths, and how can a limited budget exploit these complementary sources?

![](images/5e2b724add98888141993df3ce0fda173ed6147aa5b6692e16c915827288bfd1.jpg)  
Figure 1: Motivation and framework. (a) Existing single-depth and sampling strategies; (b) preliminary evidence that deeper recurrence does not subsume shallower solutions; (c) DPA allocates a shared budget across depth-specific candidate sources; (d) feedback-guided allocation chooses a depth or stops. States and sets are schematic.

Motivated by this preliminary diagnosis, we propose Depth Portfolio Allocation (DPA), which treats recurrent depths as candidate sources with different nominal costs and allocates acquisition counts under a shared budget (Figure 1). DPA separates the allocation problem from its optimizer: portfolios may be fixed in advance or chosen sequentially from candidate feedback. In our sequential implementation, a standard PPO policy observes the current candidate set and remaining budget, then decides which depth to query next or whether to stop, allowing the allocation to adapt to the candidates acquired so far. Experiments across multiple datasets (e.g., GSM8K, SVAMP, Multi-Arith, and MBPP) consistently validate the proposed allocation framework in both solution coverage and budget efficiency.

## Our contributions are threefold:

• We identify and empirically establish a previously overlooked issue in single-depth recurrent scaling: deeper recurrence can lose correct shallower-depth solutions, while depthspecific success sets remain systematically non-nested, making a single stopping depth incomplete.

• A new formulation of recurrent test-time scaling as budgeted allocation across depths. Depth Portfolio Allocation (DPA) treats each depth as both a computation cost and a candidate source, and allocates a shared inference budget across multiple depths rather than selecting a single stopping point.

• Extensive evaluation on four datasets. Mixeddepth portfolios consistently expand oracle coverage under matched candidate and nominal-cost budgets, while feedback-guided allocation generally matches or improves uniform allocation and can leave part of the available computation budget unused without sacrificing accuracy.

## 2 Related Work

Test-time scaling and candidate selection. Testtime scaling spends additional inference compute on explicit reasoning, repeated sampling, search, or verification (Wei et al., 2022; Wang et al., 2023; Yao et al., 2023; Cobbe et al., 2021; Lightman et al., 2024; Zhang et al., 2025). Recent work further studies Best-of-N, compute-optimal inference, and diversity-aware sampling, showing that the value of additional compute depends on candidate coverage, problem difficulty, and verification quality (Snell et al., 2025; Wu et al., 2025; Setlur et al., 2025; Huang et al., 2025; Wu et al., 2026). DPA studies a complementary allocation problem: how a fixed budget should be distributed across candidate sources induced by recurrent depth.

Adaptive computation. Adaptive-computation methods learn when to halt or how much depth to allocate, from Adaptive Computation Time and Universal Transformers to PonderNet, depthadaptive Transformers, and early-exit language models (Graves, $2 0 1 6 ;$ Dehghani et al., 2019; Banino et al., 2021; Elbayad et al., 2020; Zhou et al., 2020; Schuster et al., 2022). These methods primarily determine a single amount of computation per input; our setting instead allows multiple depths to contribute candidates under one shared budget.

Recurrent and continuous latent reasoning. Latent reasoning moves computation from explicit token traces into hidden-state refinement through pause tokens, continuous thoughts, and looped or recurrent Transformers (Goyal et al., 2024; Hao et al., 2025; Saunshi et al., 2025; Geiping et al., 2025). Recent work further explores learned or elastic recurrent depth, recurrent stability, and efficient sampling (Zhu et al., 2025; Jeddi et al., 2026; Yang et al., 2026; Geiping et al., 2026). Hidden states can also encode evaluative or reward signals (Wang et al., 2025; Du et al., 2026). We instead ask whether different recurrent depths expose complementary, non-nested candidate sets and how a shared budget should be allocated across them; PPO serves only as a standard optimizer for the sequential policy (Schulman et al., 2017). A concurrent study (Anonymous, 2026) evaluates probability updating under copied and independent evidence using exact Bayesian references. DPA instead studies candidate complementarity across recurrent depths and budgeted inference allocation.

## 3 Preliminary: Diagnosing Single-Depth Recurrent Scaling

## 3.1 Deeper Recurrence Does Not Preserve Correctness

A single-depth view implicitly assumes that additional recurrence progressively refines the current solution. We first compare latent-state changes with answers decoded at different recurrent depths on Huginn (Figure 2).

Latent updates gradually diminish with recurrent depth. Figure 2(a) shows decreasing normalized update magnitudes on MultiArith and adjacent-state cosine similarities approaching one, an empirical trend toward more stable latent states.

Stable representations do not ensure that deeper decoding retains correct answers from shallower depths. The example in Figure 2(b) asks how many shelves are needed for 14 − 2 remaining books, with three books per shelf. The correct answer, 4, appears in samples at $d = 8 .$ , 16 but not at $d = 2 4 , 3 2$ . Thus, additional recurrence need not improve answer quality monotonically, and deeper stopping can miss earlier correct solutions.

![](images/3bb5254ff474518314ae67658d10368167fad6d78654ba16c8f3ff819f06eb92.jpg)

(b) Answers across depths
<table><tr><td rowspan=1 colspan=1>Depth</td><td rowspan=1 colspan=1>rl</td><td rowspan=1 colspan=2>r2     r3</td><td rowspan=1 colspan=2>r4     r5</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>6.67</td><td rowspan=1 colspan=1>6.67</td><td rowspan=1 colspan=1>6.67</td><td rowspan=1 colspan=1>4.67</td><td rowspan=1 colspan=1>4.67</td></tr></table>

Figure 2: Stability does not ensure correctness. (a) Latent updates and cosine similarity; (b) five answers per depth, with correct answers in green.

The preliminary observation motivates studying depth complementarity. Because complete answers can be decoded from different intermediate states, a depth can act as a distinct candidate source rather than only a stopping position. We ask whether several depths cover questions missed by a single depth before turning to budget allocation.

## 3.2 Correct Solutions Are Complementary across Depths

With inputs fixed, we compare answer diversity and success sets over $\mathcal { D } = \{ 8 , 1 0 , \hdots , 3 2 \}$ , examining depth preferences and whether deeper success sets subsume shallower ones.

Answer diversity decreases with depth, but preferred depths remain dispersed. In Figure 3, bars indicate the fraction of questions for which each depth is best, and curves show within-depth answer diversity. On MultiArith and MBPP, answer diversity generally declines with depth, while bestdepth preferences span multiple positions. Candidates become more similar overall, yet different questions still favor different depths, indicating that answers from multiple depths can be useful.

Overlap between success sets. We sample five answers per question at each of the 13 depths $\mathcal { D } = \{ 8 , 1 0 , . . . , 3 2 \}$ on MultiArith and MBPP. Let $y _ { q d r } \in \{ 0 , 1 \}$ indicate whether repeat r for question q at depth d is correct. The success set at depth d is

![](images/60bf42533d6a0c0860ca23b101f019c40c286a808236c226ef77b5882c8eb76d.jpg)  
Figure 3: Depth preferences and answer diversity. Bars show best-depth frequencies; curves show disagreement across repeats at each depth.

Table 1: Mutually non-nested pairs and depth association after controlling for marginal accuracy.
<table><tr><td>Dataset</td><td>Non-nested pairs</td><td> $\Delta J$  per 8 steps (95% CI)</td></tr><tr><td>MultiArith</td><td>77/78</td><td>+3.36 pp [0.87, 5.83]</td></tr><tr><td>MBPP</td><td></td><td>78/78+5.63 pp [3.48, 8.53]</td></tr></table>

$$
C _ { d } = \{ q : \operatorname* { m a x } _ { r \in \{ 1 , \ldots , 5 \} } y _ { q d r } = 1 \} .\tag{1}
$$

Two success sets are mutually non-nested if each contains questions absent from the other. We measure their difference using Jaccard distance, $J ( d _ { i } , d _ { j } ) = 1 - | C _ { d _ { i } } \cap C _ { d _ { j } } | / | C _ { d _ { i } } \cup C _ { d _ { j } } |$ . Both datasets use the same depth grid; Figure 4 reports distances for all depth pairs.

Success sets at different depths are broadly mutually non-nested. Adjacent depths typically have more similar success sets, whereas more distant depths differ more. Table 1 shows that nearly every pair on both tasks contains questions solved by only one of its depths. Deeper success sets therefore do not simply subsume shallower ones: combining depths can contribute different correct solutions to a candidate portfolio.

To examine whether the association between depth separation and success-set differences persists after accounting for marginal accuracy, we regress pairwise Jaccard distance on depth separation, the absolute difference in single-sample marginal accuracy, and mean marginal accuracy:

![](images/253ad12c8b77b763ef10f6225067f85f68cd5f26d6e0859039b3c778d889ee6f.jpg)  
Figure 4: Jaccard distances between depth-specific success sets. Both tasks share a color scale; diagonal entries are zero.

$$
\begin{array} { l } { { J ( d _ { i } , d _ { j } ) = \beta _ { 0 } + \beta _ { 1 } | d _ { i } - d _ { j } | } } \\ { { \displaystyle ~ + \beta _ { 2 } | a _ { i } - a _ { j } | + \beta _ { 3 } \frac { a _ { i } + a _ { j } } { 2 } + \epsilon _ { i j } . } } \end{array}\tag{2}
$$

Here, $a _ { i }$ and $a _ { j }$ are the marginal accuracies at the two depths. We bootstrap questions 5,000 times, recomputing distances and regression coefficients in each replicate. Table 1 summarizes the depth-separation association after controlling for marginal accuracy.

To test whether complementarity is explained solely by question difficulty and marginal accuracy, we perform 5,000 fixed-margin randomizations of the observed correctness matrix, preserving each question’s total successes and each depth–repeat slot’s success count.

Depth-dependent success-set differences exceed the fixed-margin null expectation. On both tasks, the association between depth separation and success-set differences is significantly greater than expected under this fixed-margin null, supporting source-specific complementary information. Appendix E reports the full statistics. Section 5.2 tests whether these sources improve candidate coverage within a shared budget.

## 3.3 From Stopping Depth to Candidate Sources

The preliminary results suggest a different view of recurrent TTS. If deeper computation does not subsume shallower solutions, inference is not only a stopping problem. Under a finite budget, the system must choose depth-specific sources, acquisition counts, and when to stop. This motivates the allocation formulation in the next section.

## Takeaway

Depth is a candidate source as well as a stopping position. Along a gradually stabilizing latent reasoning trajectory, intermediate states can be decoded again and used as answer sources. Complementary success sets therefore motivate allocating a shared budget across depths.

## 4 Depth Portfolio Allocation

## 4.1 Budgeted Allocation across Depths

The preliminary analysis suggests that a single stopping point can miss correct solutions available at other depths. We therefore formulate Depth Portfolio Allocation (DPA): choose candidate sources and acquisition counts under a shared budget to maximize final selection accuracy.

Given a question $q$ and candidate depths $\mathcal { D } _ { : }$ let $x _ { q d r }$ be candidate r at depth $d ,$ and let ${ \textbf { n } } =$ $( n _ { d } ) _ { d \in \mathcal { D } }$ denote the number of candidates acquired from each depth. With nominal per-candidate cost $c _ { d } = d ,$ , the portfolio cost is

$$
C ( \mathbf { n } ) = \sum _ { d \in \mathcal { D } } d n _ { d } \leq B ,\tag{3}
$$

where B is the budget cap, with at most five candidates per depth in sequential experiments. In our implementation, candidate generation takes tens of seconds, while PPO decisions and selection take milliseconds. Although generated output length affects latency, recurrent computation scales approximately linearly with depth under comparable generation settings; we therefore adopt a nominal recurrent-step budget, excluding decoding and lightweight decision overhead.

Let $R _ { \alpha }$ be the candidate set produced by allocation rule α, and let $g ( q , R _ { \alpha } )$ select its final answer. DPA seeks

$$
\begin{array} { r l } & { \alpha ^ { * } = \arg \underset { \alpha \in \mathcal { F } _ { \mathrm { D P A } } } { \mathrm { m a x } } ~ \mathbb { E } [ y ( q , g ( q , R _ { \alpha } ) ) ] } \\ & { \qquad \mathrm { s . t . } ~ C ( \mathbf { n } _ { \alpha } ) \leq B . } \end{array}\tag{4}
$$

Final selection accuracy evaluates the returned answer. We separately use oracle coverage—whether a portfolio contains at least one correct candidate— to diagnose source complementarity. DPA therefore covers both predetermined portfolios and sequential acquisition rules.

## 4.2 Feedback-Guided Sequential Allocation

For sequential DPA, the policy observes the question, candidates acquired so far, per-depth counts, selector feedback, and remaining budget. We write the state as

$$
s _ { t } = ( \phi ( q ) , \psi ( R _ { t } ) , \mathbf { n } _ { t } , B - C _ { t } ) ,\tag{5}
$$

where $\phi ( q )$ represents the question and $\psi ( R _ { t } )$ summarizes the acquired candidates and selector feedback. At each step the policy chooses

$$
a _ { t } \in { \mathcal { D } } \cup \{ { \mathrm { s t o p } } \} .\tag{6}
$$

If $a _ { t } = d ,$ one new candidate is acquired at that depth and the portfolio is updated as

$$
\begin{array} { r l } & { R _ { t + 1 } = R _ { t } \cup \{ x _ { q , d , n _ { t , d } + 1 } \} , } \\ & { n _ { t + 1 , d } = n _ { t , d } + 1 , \qquad C _ { t + 1 } = C _ { t } + d . } \end{array}\tag{7}
$$

A feasibility mask removes depths that exceed the remaining budget or the per-depth call limit; the stop action makes no further model call.

We instantiate this allocation interface with standard PPO (Schulman et al., 2017). On labeled training questions, the reward is the change in correctness of the selector’s current output,

$$
r _ { t } = u _ { t + 1 } - u _ { t } ,\tag{8}
$$

so the undiscounted trajectory return telescopes to final selection correctness. Standard PPO estimates advantages $\hat { A } _ { t }$ and uses

$$
\rho _ { t } ( \theta ) = \frac { \pi _ { \theta } ( a _ { t } \mid s _ { t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { t } \mid s _ { t } ) } ,\tag{9}
$$

with the clipped surrogate objective

$$
\begin{array} { r } { L _ { \mathrm { P P O } } ( \theta ) = \mathbb { E } _ { t } \big [ \operatorname* { m i n } ( \rho _ { t } \hat { A } _ { t } , ~ } \\ { \operatorname { c l i p } ( \rho _ { t } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { t } ) \big ] . } \end{array}\tag{10}
$$

Ground-truth labels are used only for training rewards; deployment decisions use candidate feedback and budget state. PPO is therefore the optimizer for feedback-guided allocation, rather than the core DPA formulation.

Implementation. Our experiments realize each depth action with an unused candidate from a frozen question–depth pool, which isolates allocation and selection from regeneration variance; an online realization would call the recurrent generator after each depth choice. The policy state combines question features, per-depth acquisition counts, remaining budget, selector-score summaries, and projected terminal hidden states from acquired candidates.

![](images/901ef244c99bb22e9e45449ad2aa1ceb646c13ae2b6e4d6caf123ba284ecb217.jpg)  
Budget ceiling B (nominal recurrent steps)  
Figure 5: Accuracy and budget use under shared budget caps. Top: test accuracy for feedback-guided DPA (PPO) and uniform allocation, with static fixed-depth references. Bottom: PPO nominal cost C versus cap B; shading shows unused budget $B - C ,$ , with unused steps annotated at $B = 3 2 0$ . Detailed same-cap comparisons are discussed in Section 5.3.

Selection and termination. After every acquisition, a five-member Borda selector reranks the current subset and returns updated feedback. Depth actions are masked when they violate the remaining budget or the per-depth call limit, and STOP becomes available after the first acquisition. Candidate slots are never reused. Appendix C gives the exact feature construction, network, masks, selector, cache simulation, and optimization settings.

## 5 Experiments

We organize the evaluation around three research questions:

• RQ1: Does depth diversity expand the reachable solution set under controlled candidate counts and costs?

• RQ2: Does feedback-guided allocation improve final accuracy over uniform allocation under the same budget cap?

• RQ3: When can adaptive depth allocation reduce nominal computation rather than simply exhaust the available budget?

## 5.1 Experimental Setup

We first test whether depth portfolios expand oracle coverage, then evaluate selection accuracy and budget use. Experiments cover GSM8K (Cobbe et al., 2021), SVAMP (Patel et al., 2021), Multi-Arith (Roy and Roth, 2015), and MBPP (Austin et al., 2021) with a frozen Huginn backbone. Appendix D gives the splits. Within each comparison, methods share prompts and sampling settings. Controlled experiments fix candidate counts or nominal budgets using Table 2.

Sequential experiments use $\mathcal { D } = \{ 1 6 , 2 4 , 3 2 \}$ , at most five on-demand candidates at each depth, and caps $B \in \{ 4 8 , 8 0 , 1 2 0 , 1 6 0 , 2 0 0 , 2 4 0 , 3 2 0 \}$ . Each DPA/PPO depth action acquires one new candidate before the policy continues or stops using the newly updated candidate feedback. Policies are trained separately for each task. Configurations are selected on validation data, then frozen and evaluated on independent test splits. A Borda selector based on latent reward models (LRMs) returns the final answer. We report selection accuracy, nominal cost used, and budget curves to characterize answer quality and computation usage.

## 5.2 Does Depth Diversity Expand the Reachable Solution Set? (RQ1)

To separate source complementarity from additional sampling or computation, we fix candidate counts or nominal budgets and compare oracle coverage for depth portfolios, a fixed intermediate depth, and repeated shallow sampling (Table 2).

Depth portfolios expand coverage at matched candidate counts and nominal costs. Table 2A compares a mixed-depth portfolio with a fixed intermediate depth under both experimental controls. Higher coverage on both tasks supports a benefit from combining sources beyond increasing the number of candidates or computational expenditure. Fixed-depth coverage fluctuates across recurrent depths, showing that deeper computation need not access a broader solution set. The matchedcount, matched-cost coverage gain also exceeds the fixed-margin null expectation (Appendix E), further supporting source complementarity.

Table 2: RQ1: Oracle coverage (%) under controlled allocations. (A) Four candidates per configuration; the portfolio and $4 \times d _ { 1 6 }$ both cost 64. (B) Matched nominal budgets. Parentheses report gains (pp) over $4 \times d _ { 1 6 }$ in A and the stronger fixed-depth baseline in B.  
A. Fixed candidate count
<table><tr><td>Allocation</td><td>Nominal cost Candidates</td><td></td><td>MBPP</td><td>MultiArith</td></tr><tr><td>4× d8</td><td>32</td><td>4</td><td>21.04%</td><td>82.50%</td></tr><tr><td> $4 \times d _ { 1 6 }$ </td><td>64</td><td>4</td><td>31.72%</td><td>91.25%</td></tr><tr><td> $4 \times d _ { 2 4 }$ </td><td>96</td><td>4</td><td>31.72%</td><td>90.25%</td></tr><tr><td> $4 \times d _ { 3 2 }$ </td><td>128</td><td>4</td><td>30.88%</td><td>82.75%</td></tr><tr><td> $d _ { 8 } + d _ { 1 2 } + d _ { 2 0 } + d _ { 2 4 }$ </td><td>64</td><td>4</td><td>36.23% (+4.51 pp)</td><td>93.75% (+2.50 pp)</td></tr></table>

B. Fixed nominal cost
<table><tr><td>Budget</td><td>Method</td><td>Depth allocation</td><td>Candidates</td><td>GSM8K</td><td>SVAMP</td><td>MBPP</td><td>MultiArith</td></tr><tr><td rowspan="3">48</td><td>Shallow resampling</td><td> $6 \times d _ { 8 }$ </td><td>6</td><td>25.34%</td><td>48.58%</td><td>26.44%</td><td>87.88%</td></tr><tr><td>Fixed intermediate</td><td> $3 \times d _ { 1 6 }$ </td><td>3</td><td>33.04%</td><td>58.05%</td><td>36.04%</td><td>85.96%</td></tr><tr><td>Depth portfolio</td><td> $d _ { 1 4 } + d _ { 1 6 } + d _ { 1 8 }$ </td><td>3</td><td>35.20% (+2.16 pp)</td><td>64.50% (+6.45 pp)</td><td>40.40% (+4.36 pp)</td><td>88.70% (+0.82 pp)</td></tr><tr><td rowspan="3">80</td><td>Shallow resampling</td><td> $1 0 \times d _ { 8 }$ </td><td>10</td><td>31.05%</td><td>51.60%</td><td>28.90%</td><td>94.22%</td></tr><tr><td>Fixed intermediate</td><td> $5 \times d _ { 1 6 }$ </td><td>5</td><td>36.29%</td><td>59.19%</td><td>37.22%</td><td>92.30%</td></tr><tr><td>Depth portfolio</td><td> $d _ { 1 2 } + d _ { 1 4 } + d _ { 1 6 } + d _ { 1 8 } + d _ { 2 0 }$ </td><td>5</td><td>41.70% (+5.41 pp)</td><td>67.00% (+7.81 pp)</td><td>42.80% (+5.58 pp)</td><td>95.70% (+1.48 pp)</td></tr><tr><td rowspan="3">144</td><td>Shallow resampling</td><td> $1 8 \times d _ { 8 }$ </td><td>18</td><td>39.27%</td><td>54.49%</td><td>31.49%</td><td>97.85%</td></tr><tr><td>Fixed intermediate</td><td> $9 \times d _ { 1 6 }$ </td><td>9</td><td>40.30%</td><td>60.70%</td><td>37.90%</td><td>93.40%</td></tr><tr><td>Depth portfolio</td><td> $d _ { 8 } + \cdots + d _ { 2 4 }$ </td><td>9</td><td>49.55% (+9.25 pp)</td><td>68.40% (+7.70 pp)</td><td>43.56% (+5.66 pp)</td><td>97.85% (+0.00 pp)</td></tr></table>

Note: Oracle coverage is the fraction of questions with at least one correct candidate. Nominal cost sums candidate recurrent depths.

Combining complementary sources can cover more correct answers than adding shallow samples. Table 2B compares allocations at equal nominal budgets. Portfolios and the intermediate-depth baseline use the same candidate count, while shallow sampling permits more candidates. Portfolios still achieve higher coverage on GSM8K, SVAMP, and MBPP, indicating that source complementarity affects the solutions reachable within a budget alongside candidate quantity.

Portfolio gains narrow when shallow sampling approaches coverage saturation. Multi-Arith differs from the other tasks: as the budget increases, repeated shallow sampling approaches the portfolio’s coverage and matches it at the highest nominal budget. The value of depth allocation thus depends on which questions a fixed source still misses. When shallow sampling already covers most correct solutions, other depths have less room to contribute further.

Answer to RQ1. Depth diversity expands the reachable solution set beyond candidate quantity alone, but the gain narrows when a strong fixed source approaches coverage saturation.

## Takeaway

Source diversity contributes beyond candidate quantity. At matched nominal costs, complementary depths expand coverage, with smaller gains when a strong fixed source approaches saturation.

## 5.3 Does Adaptive Allocation Improve Accuracy under the Same Budget? (RQ2)

Table 3 gives representative absolute performance, while Figure 5 and Appendix F provide the direct same-cap budget comparison between feedbackguided DPA and DPA/Uniform. Policies are selected on validation data and subsequently evaluated on independent test splits.

Feedback-guided allocation generally improves over Uniform under the same cap. DPA/PPO performs better than Uniform across most budget settings on all four datasets, with the largest gains appearing on MultiArith. Improvements are smaller on the remaining tasks, with only occasional minor regressions.

The absolute gains over single-trajectory inference remain substantial, but they reflect a different inference configuration. At B = 320, DPA/PPO reaches 47.62%, 58.03%, 31.30%, and 92.12% on GSM8K, SVAMP, MBPP, and Multi-

Table 3: Representative DPA results. A: singletrajectory and latent-correction baselines; B: feedbackguided DPA test accuracy at representative caps; C: nominal cost used. Figure 5 and Appendix F give samecap Uniform comparisons.  
A. Single-trajectory and latent-corr accuracy (%)
<table><tr><td>Method</td><td>GSM8K</td><td>SVAMP</td><td>MBPP</td><td>MultiArith</td></tr><tr><td>Base / single</td><td>32.60%</td><td>51.70%</td><td>27.80%</td><td>75.00%</td></tr><tr><td>CoE-R correction</td><td>33.00%</td><td>51.00%</td><td>27.60%</td><td>71.25%</td></tr><tr><td>CoE-C correction</td><td>32.40%</td><td>51.60%</td><td>28.00%</td><td>72.50%</td></tr></table>

B. DPA/PPO selection accuracy (%)

<table><tr><td>Budget cap B</td><td>GSM8K</td><td>SVAMP</td><td>MBPP</td><td>MultiArith</td></tr><tr><td>80</td><td>43.92%</td><td>58.24%</td><td>30.63%</td><td>87.29%</td></tr><tr><td>200</td><td>47.06%</td><td>57.99%</td><td>30.86%</td><td>91.76%</td></tr><tr><td>320</td><td>47.62%</td><td>58.03%</td><td>31.30%</td><td>92.12%</td></tr></table>

C. DPA/PPO nominal cost used (recurrent steps)

<table><tr><td>Budget cap B</td><td>GSM8K</td><td>SVAMP</td><td></td><td>MBPP MultiArith</td></tr><tr><td>80</td><td>80.0</td><td>80.0</td><td>74.3</td><td>77.4</td></tr><tr><td>200</td><td>200.0</td><td>198.5</td><td>180.7</td><td>200.0</td></tr><tr><td>320</td><td>297.5</td><td>312.0</td><td>238.7</td><td>296.2</td></tr></table>

Arith. We therefore use the same-cap Uniform comparison, rather than these single-trajectory baselines, as the main test of adaptive allocation.

Answer to RQ2. Under a shared budget cap, feedback-guided allocation usually matches or improves uniform allocation, with the clearest gains on MultiArith at intermediate budgets.

## 5.4 When Does Depth Allocation Save Computation? (RQ3)

Figure 5 also reports mean nominal cost $C =$ $\textstyle \sum _ { d } d n _ { d }$ and unused budget B − C across the seven budget caps. Static fixed-depth references use twenty independent candidates and therefore are not matched-budget comparisons at every horizontal-axis position. Appendix F reports the complete test results and budget utilization.

Larger budgets do not guarantee monotonic improvement or lower cost. SVAMP and MBPP selection accuracy exhibits local fluctuations. At $B \ = \ 3 2 0$ , PPO uses less nominal computation than Uniform on GSM8K, MBPP, and MultiArith, but slightly more on SVAMP. At smaller budgets, higher accuracy can also entail greater computational expenditure, so a shared budget cap must be distinguished from equal actual cost.

PPO can terminate before exhausting its budget. At larger budgets, the policy often leaves compute unused, suggesting that further acquisition may not help. This effect is most pronounced on MBPP, where PPO uses only 74.6% of the B = 320 budget while slightly improving test accuracy over the $B = 2 4 0$ setting. At $B = 3 2 0$ PPO also achieves a better accuracy–cost tradeoff than Uniform on several tasks. We note that unused budget may arise either from an explicit STOP decision or from infeasible remaining actions, and nominal cost excludes decoding, scoring, and policy-inference overhead.

Static fixed-depth references also show that deeper computation need not yield better final answers. Deeper configurations have no consistent advantage across tasks, agreeing with the earlier depth observations. Their full nominal costs appear in the legend.

Answer to RQ3. Adaptive allocation saves nominal computation mainly when the policy can stop acquiring useful candidates before the cap is exhausted; this is most visible at high budget caps, especially on MBPP. At larger budgets, the policy often leaves part of the available computation remaining unused, indicating that additional candidate acquisition is not always beneficial.

## Takeaway

Allocation gains depend on the task and budget. Feedback-guided allocation usually improves over Uniform under the same cap, while computation savings appear when the policy can stop before exhausting the available budget.

## 6 Conclusion

Recurrent test-time scaling is not only a question of how deep to run. Our preliminary analysis shows that deeper recurrence can lose correct shallowerdepth solutions and that depth-specific success sets are broadly non-nested. Motivated by this finding, we formulate Depth Portfolio Allocation (DPA), treating recurrent depths as complementary candidate sources under a shared inference budget. Experiments across four benchmark datasets show that mixed-depth portfolios expand candidate solution coverage, while feedback-guided sequential allocation generally matches or improves uniform allocation and can reduce unnecessary recurrent computation. Overall, DPA reframes recurrent TTS from single-depth stopping decisions to budgeted allocation across complementary depths.

## Limitations

Our study focuses on a controlled recurrent reasoning setting with a frozen Huginn backbone and four reasoning/code-generation benchmarks. Extending the analysis to additional recurrent architectures, model scales, and task domains would further test the generality of the depth-allocation view. Our sequential experiments use a bounded candidate pool and nominal recurrent depth as the primary computation measure, which provides a clean basis for comparing allocation strategies. Future work may consider richer candidate-generation settings and end-to-end runtime costs, including decoding and selector overhead. Finally, the current analysis emphasizes depth complementarity and budget allocation; broader forms of adaptive recurrent inference may provide additional opportunities beyond the formulations studied here.

## Ethics Statement

This work studies test-time computation strategies using established reasoning and code-generation benchmarks. It does not involve human subjects, private personal data, or real-world decision making about individuals. All experiments are conducted offline using model-generated candidates and benchmark-based evaluation. Ground-truth labels are used only for training rewards and evaluation and are not available to the allocation policy at inference time.

## References

Anonymous. 2026. When does agreement add information? evaluating probability updates in language models. Under review.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program Synthesis with Large Language Models. arXiv preprint arXiv:2108.07732.

Andrea Banino, Jan Balaguer, and Charles Blundell. 2021. PonderNet: Learning to Ponder. arXiv preprint arXiv:2107.05407.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. 2019. Uni-

versal transformers. In International Conference on Learning Representations.

Hanwen Du, Yuxin Dong, and Xia Ning. 2026. Latent thinking optimization: Your latent reasoning language model secretly encodes reward signals in its latent thoughts. In International Conference on Learning Representations.

Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. 2020. Depth-adaptive transformer. In International Conference on Learning Representations.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. 2025. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In Advances in Neural Information Processing Systems, volume 38, pages 41340–41391.

Jonas Geiping, Xinyu Yang, and Guinan Su. 2026. Efficient parallel samplers for recurrent-depth models and their connection to diffusion language models. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 34442–34462. PMLR.

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. 2024. Think before you speak: Training language models with pause tokens. In International Conference on Learning Representations.

Alex Graves. 2016. Adaptive Computation Time for Recurrent Neural Networks. arXiv preprint arXiv:1603.08983.

Shibo Hao, Sainbayar Sukhbaatar, Dijia Su, Xian Li, Zhiting Hu, Jason E. Weston, and Yuandong Tian. 2025. Training large language models to reason in a continuous latent space. In Conference on Language Modeling.

Audrey Huang, Adam Block, Qinghua Liu, Nan Jiang, Akshay Krishnamurthy, and Dylan J. Foster. 2025. Is best-of-n the best of them? coverage, scaling, and optimality in inference-time alignment. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 25075–25126. PMLR.

Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. 2026. Loopformer: Elastic-depth looped transformers for latent reasoning via shortcut modulation. In International Conference on Learning Representations.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s Verify Step by Step. In International Conference on Learning Representations, pages 39578–39601.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 2080–2094, Online. Association for Computational Linguistics.

Subhro Roy and Dan Roth. 2015. Solving general arithmetic word problems. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1743–1752, Lisbon, Portugal. Association for Computational Linguistics.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. 2025. Reasoning with latent thoughts: On the power of looped transformers. In International Conference on Learning Representations.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay, and Donald Metzler. 2022. Confident adaptive language modeling. In Advances in Neural Information Processing Systems, volume 35.

Amrith Setlur, Nived Rajaraman, Sergey Levine, and Aviral Kumar. 2025. Scaling test-time compute without verification or RL is suboptimal. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 54058–54094. PMLR.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2025. Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning. In International Conference on Learning Representations.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations.

Yiming Wang, Pei Zhang, Baosong Yang, Derek F. Wong, and Rui Wang. 2025. Latent space chain-ofembedding enables output-free llm self-evaluation. In International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837.

Chen Henry Wu, Sachin Goyal, and Aditi Raghunathan. 2026. Mode-conditioning unlocks superior test-time

compute scaling. In International Conference on Learning Representations.

Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. 2025. Inference scaling laws: An empirical analysis of compute-optimal inference for LLM problem-solving. In International Conference on Learning Representations.

Xiao-Wen Yang, Ziyu Han, Xi-Hua Zhang, Wen-Da Wei, Jie-Jing Shao, Lan-Zhe Guo, and Yu-Feng Li. 2026. Stabilizing recurrent dynamics for test-time scalable latent reasoning in looped language models. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 144670–144682. PMLR.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, volume 36.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Seyed Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. 2025. Generative verifiers: Reward modeling as next-token prediction. In International Conference on Learning Representations.

Wangchunshu Zhou, Canwen Xu, Tao Ge, Julian McAuley, Ke Xu, and Furu Wei. 2020. BERT Loses Patience: Fast and Robust Inference with Early Exit. In Advances in Neural Information Processing Systems, volume 33, pages 18330–18341.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, et al. 2025. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741.

## A Deeper Recurrence Does Not Preserve Correctness: Additional Diagnostics

This appendix extends the preliminary diagnosis in Section 3 across all four tasks and reports the corresponding statistical details.

Table 4: Answer switches under low updates. Rates and confidence intervals are percentages.
<table><tr><td>Task</td><td> $\tau _ { u }$ </td><td> $\tau _ { v }$ </td><td>Low-update intervals</td><td>Truncated answers</td><td>Switch rate [95% CI]</td><td>Non- truncated rate [95% CI]</td></tr><tr><td>GSM8K</td><td>0.073640.002711</td><td></td><td>323</td><td>56</td><td>25.08 [14.33,</td><td>25.82 [15.04,</td></tr><tr><td>SVAMP</td><td>0.05206 0.001355</td><td></td><td>266</td><td>1</td><td>37.20] 9.40 [0.84,</td><td>38.23] 9.40 [0.84,</td></tr><tr><td></td><td>MultiArith0.049500.001225</td><td></td><td>317</td><td>3</td><td>20.20] 21.14 [11.91,</td><td>20.20] 20.89 [11.65,</td></tr><tr><td>MBPP</td><td>0.030290.000459</td><td></td><td>288</td><td>73</td><td>31.07] 9.72 [3.91, 16.39]</td><td>30.79] 10.33 [4.25, 17.31]</td></tr></table>

Table 5: Correctness transitions under low updates. All low-update intervals form the denominator; values are percentages.
<table><tr><td>Task</td><td>Wrong→correct</td><td>Correct→wrong</td><td>Wrong→other wrong</td></tr><tr><td>GSM8K</td><td>1.24 [0.00, 3.13]</td><td>4.33 [1.57, 7.62]</td><td>19.50 [9.54, 31.38]</td></tr><tr><td>SVAMP</td><td>0.00 [0.00, 0.00]</td><td>0.00 [0.00, 0.00]</td><td>9.40 [0.84, 20.20]</td></tr><tr><td>MultiArith</td><td>5.05 [1.02, 9.79]</td><td>5.68 [1.27, 10.84]</td><td>10.41 [3.54, 18.68]</td></tr><tr><td>MBPP</td><td>1.04 [0.00, 3.25]</td><td>0.35 [0.00, 1.08]</td><td>3.82 [0.67, 8.24]</td></tr></table>

Paired diagnostics and statistical units. Each task has twenty test questions, five repeats, and thirteen depths: 1,300 complete answers and 1,200 adjacent-depth intervals. States are taken at the final prompt token during recurrent computation. Elementwise checks confirm shared-prefix consistency for all four tasks. Subsequent answers are generated at their respective fixed depths. Changes therefore reflect the complete depth-specific generation procedure, without isolating the causal effect of small prompt-boundary state changes.

Update metrics and thresholds. For promptboundary state $h _ { d } ,$ define $u _ { d } = \| h _ { d } - h _ { d - 2 } \| _ { 2 } / ( \| h _ { d - 2 } \| _ { 2 } +$ $1 0 ^ { - 1 2 } )$ and $v _ { d } = 1 - \cos ( h _ { d } , h _ { d - 2 } )$ . Low-update intervals satisfy both $u _ { d } \leq \tau _ { u }$ and $v _ { d } \leq \tau _ { v }$ . Each threshold is the 25th percentile of its metric on independent training questions (Table 4). This condition denotes relatively small updates rather than mathematical convergence.

Intervals and truncation sensitivity. We bootstrap questions with replacement 10,000 times using seed 20260922, retaining within-question depths and repeats. Curves average repeats within each question before averaging questions. Conditional event rates divide event counts by the number of qualifying intervals. The non-truncated analysis requires both adjacent answers to terminate below the generation cap; Table 4 reports its switch rates separately.

Interpretation. Answers can switch under low updates, with task-dependent implications. Figure 7 extends the main-text example. Table 5 separates wrong-to-correct, correct-to-wrong, and wrong-to-different-wrong events. No transitions of the first two types are observed in SVAMP’s lowupdate intervals; other tasks exhibit both repairs and degradations. Two correct programs can also differ in identity, so the three correctness events need not exhaust code switches. Representation stability, answer-identity stability, and correctness stability should therefore be measured separately.

## B Correct Solutions Are Complementary across Depths: Additional Statistics

This appendix complements Section 3.2 with full per-depth statistics and directional rescue results.

Marginal performance and coverage. Table 6 recomputes three metrics from raw correctness tensors. Singlesample accuracy averages five repeats per question; solved counts indicate at least one correct repeat; Oracle@5 divides these counts by pool size. Correlations among repeats remain in the observed pool: Oracle@5 is computed directly from its five candidates rather than approximated by independent Bernoulli draws. Per-depth results are also saved as CSV for checking Figure 4 and the success-set definition.

Directional rescue. Jaccard distance summarizes the overall difference between two depth-specific success sets, but it is symmetric and therefore does not reveal which depth contributes the additional solved questions. Figure 6 complements this analysis with a directional measure. For a reference depth d and an added depth $d _ { j } .$ we define

$$
R _ { i  j } = \frac { | C _ { d _ { j } } \setminus C _ { d _ { i } } | } { Q } ,\tag{11}
$$

where $Q$ is the total number of questions in the evaluation pool. The numerator counts questions that are solved by at least one of the five candidates from $d _ { j }$ but by none of the five candidates from $d _ { i }$ . Thus, $R _ { i \to j }$ measures the fraction of the entire pool uniquely rescued by adding source $d _ { j }$ to source $d _ { i } .$ . Because the measure is directional, $R _ { i \to j }$ and $R _ { j  i }$ need not be equal: one depth may recover many questions missed by another while contributing little in the reverse direction. Diagonal entries are zero by definition. Each cell therefore characterizes source-level complementarity between two complete depth-specific candidate sets, rather than the marginal gain of acquiring one additional candidate under a matched computation budget.

Connection to best-depth preferences. Figure 3 identifies, for each question, the depth with the highest fiverepeat accuracy, with tied maxima split equally and unsolved questions excluded. Preference frequencies indicate which depths are most often favored, Oracle@5 measures per-depth coverage, and directional rescue captures failures of one depth recovered by another. Together, these metrics distinguish depth preference, overall coverage, and complementary coverage. A depth can therefore perform well on average while still missing questions solved elsewhere, motivating portfolio allocation across depths.

Statistical interpretation. Marginal performance and directional rescue jointly characterize depth value. Table 6 shows non-monotonic accuracy and solved counts, while the heatmap identifies omissions recovered by an added source. These statements apply within each pool. The early MultiArith preference pool and confirmatory set pool are analyzed separately; cross-pool differences are not interpreted as depth effects, and code-text disagreement on MBPP is not equated with functional disagreement.

![](images/7fa8f13935f6dff739347fd9411144dcd4e2f45920d18f981279fdd3d51409bc.jpg)

![](images/78afc5c014ebb53dd8cb0a13c3874b47da7f3bcb060f9fe36f9249b98d43d554.jpg)  
Figure 6: Directional rescue across depths. Rows denote the reference depth and columns the added depth. Color shows the fraction of questions solved only by the added depth.

Table 6: Full results at thirteen depths. MA: 80-question MultiArith confirmatory pool; MBPP: 500-question grid pool.
<table><tr><td>d</td><td>MA single (%)</td><td>MA solved questions</td><td>MA Oracle@5 (%)</td><td>MBPP single (%)</td><td>MBPP solved questions</td><td>MBPP Oracle@5 (%)</td></tr><tr><td>8</td><td>50.25</td><td>69</td><td>86.25</td><td>15.16</td><td>109</td><td>21.80</td></tr><tr><td>10</td><td>64.00</td><td>73</td><td>91.25</td><td>22.24</td><td>143</td><td>28.60</td></tr><tr><td>12</td><td>66.00</td><td>74</td><td>92.50</td><td>24.12</td><td>147</td><td>29.40</td></tr><tr><td>14</td><td>69.25</td><td>70</td><td>87.50</td><td>26.36</td><td>158</td><td>31.60</td></tr><tr><td>16</td><td>71.75</td><td>75</td><td>93.75</td><td>27.96</td><td>161</td><td>32.20</td></tr><tr><td>18</td><td>74.25</td><td>72</td><td>90.00</td><td>29.04</td><td>162</td><td>32.40</td></tr><tr><td>20</td><td>73.00</td><td>73</td><td>91.25</td><td>29.52</td><td>161</td><td>32.20</td></tr><tr><td>22</td><td>70.75</td><td>72</td><td>90.00</td><td>28.96</td><td>159</td><td>31.80</td></tr><tr><td>24</td><td>68.50</td><td>74</td><td>92.50</td><td>29.48</td><td>160</td><td>32.00</td></tr><tr><td>26</td><td>70.50</td><td>70</td><td>87.50</td><td>28.52</td><td>156</td><td>31.20</td></tr><tr><td>28</td><td>71.75</td><td>70</td><td>87.50</td><td>28.92</td><td>161</td><td>32.20</td></tr><tr><td>30</td><td>71.50</td><td>71</td><td>88.75</td><td>29.12</td><td>157</td><td>31.40</td></tr><tr><td>32</td><td>70.50</td><td>67</td><td>83.75</td><td>29.00</td><td>156</td><td>31.20</td></tr></table>

## C Feedback-Guided Sequential Allocation: Implementation Details

This appendix provides the implementation details for the feedback-guided allocation rule in Section 4.2. The sequential state, portfolio update, reward, PPO ratio, and clipped objective are defined in the main text and are not repeated here.

Sequential execution and cache simulation. DPA’s decision interface selects a depth or stops within a budget. The cache implementation simulates this interface with frozen candidates: after a depth action, it uniformly draws an unused slot from that depth’s five candidates, then updates feedback and cost. It does not regenerate answers or update Huginn or the LRMs. It reproduces allocation and selection on the given pool; online execution instead calls the generator after each depth action to produce one candidate. The parameters below describe this cache simulation.

State features. Question text is lowercased and split into words, numbers, and symbols. Signed hashing of unigrams and adjacent bigrams yields an $\ell _ { 2 }$ -normalized 256- dimensional vector. The five LRM scores for acquired candidates are standardized per LRM over the current subset, then summarized by per-depth means and maxima. Additional features are counts divided by five, the remaining-budget fraction, and the initial budget divided by 320. Unacquired terminal states are zero-filled; acquired $h _ { T }$ vectors pass through Layer-Norm and a 32-dimensional projection, then are averaged by depth. Correctness labels are used only for training rewards and offline evaluation.

Network. Text, score, and budget branches project to 96, 96, and 32 dimensions with Tanh. The hidden-state branch retains four depth slots of 32 dimensions each; the d = 8 slot remains empty in this analysis. Concatenated features pass through a two-layer Tanh body with widths 192 → 128, producing action logits and a scalar value. Five action slots represent four depths and stopping. Masking disables d = 8, leaving 16, 24, 32, consistent with the three-source formulation in the main text.

Masks and termination. A depth is feasible only if it fits the remaining budget and has been used fewer than five times. The cache implementation forbids stopping with an empty initial set and permits it after the first acquisition. If no depth remains feasible, it terminates and returns the current selector output. Costs, counts, and masks are updated after every acquisition. Slots are never reused, at most fifteen candidates can be acquired per question, and unused budget may remain.

Five-member Borda selector. Each frozen LRM ranks only the current subset, assigning higher ranks to higher scores. The candidate with the highest mean rank across five members is selected. Stable sorting resolves ties within a member; ties in final mean rank return the first candidate in the current index order. Ranks are recomputed whenever the subset changes. Standardized scores enter the policy state, whereas subset ranks determine the final answer.

![](images/7c114d4939a9e020b7524b5b385b9fefb209485a2bb9b58d4924c465e212f344.jpg)  
Figure 7: State updates and answer switches. Shading shows question-bootstrap 95% intervals; dashed lines indicate training-calibrated thresholds.

Table 7: Optimization settings for the PPO implementation.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Parallel environments / horizon</td><td>24 / 22</td></tr><tr><td>Epochs per update / minibatch Discount γ / GAE λ</td><td>4/128 1 / 0.95</td></tr><tr><td>PPO clip €</td><td>0.2</td></tr><tr><td>Value coefficient / max. gradient norm Optimizer</td><td>0.5 / 0.5 Adam</td></tr><tr><td>Updates (script default) Selection / evaluation draws (default)</td><td>160</td></tr></table>

Table 8: Historical tuning configurations.
<table><tr><td>Trial index</td><td>Learning rate</td><td>Entropy coefficient</td><td>Seed</td></tr><tr><td>0</td><td>1e-04</td><td>0.001</td><td>13</td></tr><tr><td>1</td><td>1e-04</td><td>0.001</td><td>29</td></tr><tr><td>2</td><td>1e-04</td><td>0.01</td><td>13</td></tr><tr><td>3</td><td>1e-04</td><td>0.01</td><td>29</td></tr><tr><td>4</td><td>3e-04</td><td>0.001</td><td>13</td></tr><tr><td>5</td><td>3e-04</td><td>0.001</td><td>29</td></tr><tr><td>6</td><td>3e-04</td><td>0.01</td><td>13</td></tr><tr><td>7</td><td>3e-04</td><td>0.01</td><td>29</td></tr></table>

Reward and optimization. Rewards are changes in frozen-selector correctness after acquisition, with no additional cost penalty. Stopping acquires no candidate. Advantages are computed per question trajectory and standardized within the update batch; optimization combines the clipped policy loss, value loss, and entropy term. Table 7 lists the settings. Update counts and evaluation draws are command-lineoverridable defaults. The included historical source preserves their definitions; training reproduction should also record the actual command and checkpoint configuration.

Configuration selection. At each budget, configurations are selected independently by validation accuracy; ties favor lower cost, then the smaller trial index. Selected policies are frozen and evaluated on independent test splits. All eight configurations in Table 8 use acquired candidates’ h<sub>T</sub>.

Table 9: Dataset splits.
<table><tr><td>Dataset</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>GSM8K</td><td>6,473</td><td>500</td><td>1,319</td></tr><tr><td>SVAMP</td><td>700</td><td>150</td><td>150</td></tr><tr><td>MBPP</td><td>369</td><td>90</td><td>496</td></tr><tr><td>MultiArith</td><td>330</td><td>90</td><td>80</td></tr></table>

Table 10: Analysis pools and purposes.
<table><tr><td>Analysis</td><td>Questions</td><td>Depths / repeats</td></tr><tr><td>State diagnostics</td><td>20 per task</td><td>8:2:32 / 5 each</td></tr><tr><td>Depth preferences</td><td>MA 180 / MBPP 500</td><td>8:2:32 / 5 each</td></tr><tr><td>Success sets</td><td>MA 80 / MBPP 500</td><td>8:2:32 / 5 each</td></tr><tr><td>Budget evaluation</td><td>Independent test splits</td><td>16, 24, 32 / up to 5</td></tr></table>

## D Experimental Setup: Data and Evaluation Protocols

This appendix expands the experimental setup in Section 5.1 and records dataset splits, analysis pools, matching rules, and evaluation details.

Dataset splits. Table 9 lists the dataset splits. Math tasks use questions and final numerical answers; MBPP uses programming tasks and test cases. Training data fit allocation policies, validation data select configurations, and independent test splits provide the results in Table 3, Figure 5, and Appendix F. Validation and test data are kept separate; selected configurations are frozen before test evaluation.

Analysis pools. Table 10 gives the scope of each analysis. State diagnostics compare recurrent states at the shared prompt boundary and subsequent answers on twenty test questions per task. Fine-grid analyses use thirteen depths and five repeats per depth. MultiArith depth preferences in Figure 3 use an early 180-question pool; Figure 4 and Appendix B use an 80-question confirmatory pool. Their question scopes and prompt batches differ, so question-level results are kept separate. The 500-question MBPP grid pool is also reported separately from the sequential test split.

Matching inputs, candidates, and labels. Each question uses a fixed prompt template. Depth comparisons within an analysis pool share prompts, questions, and scoring rules. A question identifier, depth, and repeat index uniquely identify each candidate. The fine-grid correctness tensor is $Y \in \{ \overline { { 0 } } , 1 \} ^ { Q \times 1 3 \times 5 }$ , with depths $8 , 1 0 , \ldots , 3 2$ and five generations. Sequential caches retain correctness, frozen LRM scores, and terminal hidden states. These fields are joined by question identifier while preserving depth and repeat axes, ensuring that feedback and labels refer to the same candidates.

Diagnostic prompts and generation. Diagnostics use Huginn-0125 with bfloat16 and greedy fixed-depth decoding: do\_sample=False, at most 512 new tokens, and a total length cap of 4096. Math prompts contain eight examples organized with Question: and Answer:; the full template is included in the diagnostic prompt file. The project’s code-task prompt builder supplies MBPP examples and task descriptions. Seeds derived from the question, repeat index, and token position pair shared-prefix states across depths. Repeats mainly reflect stochastic recurrent states. These generation settings apply specifically to this diagnostic batch.

Diagnostic correctness and answer identity. Math evaluation extracts the number following the last ####, or otherwise the final number; absolute error below 0.01 is correct. Code evaluation removes special tokens and executes task tests with a 5-second CPU limit, 2-GiB memory limit, and 7-second subprocess timeout. Math identity uses normalized numbers. Code identity uses an abstract syntax tree without location attributes, falling back to normalized text if parsing fails. Answer switches can therefore change correctness or occur between answers that are both correct or both wrong.

Metrics and cost. Oracle coverage is the fraction of questions with at least one correct candidate; selection accuracy is the fraction for which the selector returns a correct candidate. They share correctness labels but evaluate a set and a returned answer, respectively. Nominal cost sums acquired candidate depths, $\begin{array} { r } { C \ = \sum _ { d } \dot { d } n _ { d } . } \end{array}$ , with cap B. It excludes decoding length, selector runtime, and policy inference. Split sizes, pool sizes, and repeats are reported within their respective protocols.

## E Does Depth Diversity Expand the Reachable Solution Set? Additional Tests

This appendix reports the fixed-margin tests and static fixeddepth references used to support RQ1 in Section 5.2.

Fixed-margin null. The correctness tensor is flattened into a $Q \times 6 \bar { 5 }$ binary matrix, with questions as rows and depth–repeat slots as columns. Randomization fixes both margins, preserving total successes per question and slot. We run 5,000 randomizations per dataset across five chains, reconstructing success sets and statistics to test whether depth structure exceeds these marginal constraints.

Statistics and correction. Mutual non-nesting counts all 78 depth pairs. The Jaccard slope uses the controlled regression in the main text and is scaled per eight steps. Oracle gains compare the specified matched-cost configurations. One-sided p values use the fraction of null statistics at least as large as observed, with a finite-replicate correction. Holm correction jointly covers six tests.

Non-nesting and randomization. MBPP’s count of non-nested depth pairs does not significantly exceed its null expectation, whereas its adjusted slope and matched-cost gain do. Observed set relationships and fixed-margin tests characterize structure and its statistical interpretation separately;

significance for one statistic does not imply significance for another.

Static fixed-depth references. Table 11 gives static references with twenty candidates per fixed depth, costing 320, 480, and 640, respectively. They form the horizontal lines in Figure 5. These configurations are reported separately from sequential pools with at most five candidates per depth; sequential caps do not represent the static configurations’ actual costs.

Table 11: Accuracy and nominal budgets for twenty candidates at each fixed depth.
<table><tr><td>Method</td><td>B</td><td>GSM8K</td><td>SVAMP</td><td>MBPP</td><td>MultiArith</td></tr><tr><td>Fixed  $d _ { 1 6 } \times 2 0$ </td><td>320</td><td>43.40%</td><td>56.00%</td><td>29.50%</td><td>86.25%</td></tr><tr><td>Fixed  $d _ { 2 4 } \times 2 0$ </td><td>480</td><td>39.80%</td><td>54.90%</td><td>29.30%</td><td>83.75%</td></tr><tr><td>Fixed  $d _ { 3 2 } \times 2 0$ </td><td>640</td><td>37.80%</td><td>53.90%</td><td>29.50%</td><td>85.00%</td></tr></table>

## F Does Adaptive Allocation Improve Accuracy under the Same Budget? Complete Results

This appendix provides the complete same-cap comparison used for RQ2 in Section 5.3.

Table 12: Complete test-budget results. Costs are mean recurrent steps; differences are PPO minus Uniform. Values are rounded independently.
<table><tr><td>Task</td><td>B</td><td>PPO (%)</td><td>PPO cost</td><td>Uniform (%)</td><td>Uniform cost</td><td> $\Delta$  (pp)</td></tr><tr><td rowspan="7">GSM8K</td><td>48</td><td>42.03</td><td>48.0</td><td>41.63</td><td>40.0</td><td>+0.41</td></tr><tr><td>80</td><td>43.92</td><td>80.0</td><td>43.96</td><td>72.0</td><td>-0.04</td></tr><tr><td>120</td><td>45.95</td><td>114.7</td><td>45.57</td><td>112.0</td><td>+0.38</td></tr><tr><td>160</td><td>47.01</td><td>159.3</td><td>46.53</td><td>160.0</td><td>+0.48</td></tr><tr><td>200</td><td>47.06</td><td>200.0</td><td>46.70</td><td>184.0</td><td>+0.35</td></tr><tr><td>240</td><td>47.47</td><td>232.7</td><td>47.07</td><td>232.0</td><td>+0.39</td></tr><tr><td>320</td><td>47.62</td><td>297.5</td><td>47.49</td><td>304.0</td><td>+0.12</td></tr><tr><td rowspan="7">SVAMP</td><td>48</td><td>57.83</td><td>42.3</td><td>57.63</td><td>40.0</td><td>+0.20</td></tr><tr><td>80</td><td>58.24</td><td>80.0</td><td>57.88</td><td>72.0</td><td>+0.36</td></tr><tr><td>120</td><td>58.15</td><td>112.0</td><td>58.11</td><td>112.0</td><td>+0.04</td></tr><tr><td>160</td><td>58.24</td><td>147.5</td><td>57.99</td><td>160.0</td><td>+0.26</td></tr><tr><td>200</td><td>57.99</td><td>198.5</td><td>57.76</td><td>184.0</td><td>+0.24</td></tr><tr><td>240</td><td>58.43</td><td>232.5</td><td>58.03</td><td>232.0</td><td>+0.40</td></tr><tr><td>320</td><td>58.03</td><td>312.0</td><td>57.99</td><td>304.0</td><td>+0.05</td></tr><tr><td rowspan="7">MBPP</td><td>48</td><td>30.50</td><td>33.1</td><td>30.30</td><td>40.0</td><td>+0.20</td></tr><tr><td>80</td><td>30.63</td><td>74.3</td><td>30.16</td><td>72.0</td><td>+0.47</td></tr><tr><td>120</td><td>30.83</td><td>115.3</td><td>30.60</td><td>112.0</td><td>+0.24</td></tr><tr><td>160</td><td>30.99</td><td>152.4</td><td>30.77</td><td>160.0</td><td>+0.22</td></tr><tr><td>200</td><td>30.86</td><td>180.7</td><td>30.61</td><td>184.0</td><td>+0.26</td></tr><tr><td>240</td><td>31.13</td><td>238.7</td><td>30.79</td><td>232.0</td><td>+0.33</td></tr><tr><td>320</td><td>31.30</td><td>238.7</td><td>30.94</td><td>304.0</td><td>+0.36</td></tr><tr><td rowspan="7"></td><td>48</td><td>81.70</td><td>48.0</td><td>80.77</td><td>40.0</td><td>+0.93</td></tr><tr><td>80</td><td>87.29</td><td>77.4</td><td>84.87</td><td>72.0</td><td>+2.43</td></tr><tr><td>120</td><td>90.32</td><td>114.7</td><td>88.42</td><td>112.0</td><td>+1.90</td></tr><tr><td>MultiArith 160</td><td>90.66</td><td>152.6</td><td>89.88</td><td>160.0</td><td>+0.78</td></tr><tr><td>200</td><td>91.76</td><td>200.0</td><td>90.43</td><td>184.0</td><td>+1.34</td></tr><tr><td>240</td><td>91.67</td><td>231.9</td><td>91.25</td><td>232.0</td><td>+0.42</td></tr><tr><td>320</td><td>92.12</td><td>296.2</td><td>92.27</td><td>304.0</td><td>-0.15</td></tr></table>

Evaluation and configurations. These results share the source of Figure 5. Policies are fitted on training data, configurations are selected on validation data, and accuracy and mean cost are evaluated on independent test splits. The seven caps range from 48 to 320. Accuracy, cost, and PPOminus-Uniform differences are rounded independently, so a reported difference can differ by 0.01 percentage points from subtracting the two displayed accuracies.

Statistical interpretation. Budget comparisons report point estimates of selection accuracy, mean nominal cost, and PPO-minus-Uniform differences. These point estimates do not establish statistical significance. A shared budget cap does not imply equal actual expenditure; allocation strategies should be compared in terms of both accuracy and cost.

## G When Does Depth Allocation Save Computation? Budget Utilization

This appendix reports the budget-utilization details used for RQ3 in Section 5.4 and makes the Uniform baseline explicit.

Exact uniform-allocation rule. At each step, Uniform considers the least-used depths, choosing the shallowest affordable one among them. If none is affordable, it stops rather than acquiring from a more-used source. Table 14 gives the shared allocations and costs across tasks, explaining the Uniform costs in Table 12.

Table 13: PPO budget use on independent test splits. Cells report mean nominal recurrent steps C, with budget utilization 100C/B in parentheses. Cap B is the maximum permitted nominal cost, not runtime.
<table><tr><td>B</td><td>GSM8K</td><td>SVAMP</td><td>MBPP</td><td>MultiArith</td></tr><tr><td>48</td><td>48.0 (100.0%)</td><td>42.3 (88.1%)</td><td>33.1 (69.0%)</td><td>48.0 (100.0%)</td></tr><tr><td>80</td><td>80.0 (100.0%)</td><td>80.0 (100.0%)</td><td>74.3 (92.9%)</td><td>77.4 (96.8%)</td></tr><tr><td>120</td><td>114.7 (95.6%)</td><td>112.0 (93.3%)</td><td>115.3 (96.1%)</td><td>114.7 (95.6%)</td></tr><tr><td>160</td><td>159.3 (99.6%)</td><td>147.5 (92.2%)</td><td>152.4 (95.2%)</td><td>152.6 (95.4%)</td></tr><tr><td>200</td><td>200.0 (100.0%)</td><td>198.5 (99.2%)</td><td>180.7 (90.3%)</td><td>200.0 (100.0%)</td></tr><tr><td>240</td><td>232.7 (97.0%)</td><td>232.5 (96.9%)</td><td>238.7 (99.5%)</td><td>231.9 (96.6%)</td></tr><tr><td>320</td><td>297.5 (93.0%)</td><td>312.0 (97.5%)</td><td>238.7 (74.6%)</td><td>296.2 (92.6%)</td></tr></table>

Table 14: Uniform allocations in budget evaluation.
<table><tr><td>B</td><td>n16</td><td>N24</td><td>n32</td><td>Nominal cost</td></tr><tr><td>48</td><td>1</td><td>1</td><td>0</td><td>40</td></tr><tr><td>80</td><td>1</td><td>1</td><td>1</td><td>72</td></tr><tr><td>120</td><td>2</td><td>2</td><td>1</td><td>112</td></tr><tr><td>160</td><td>3</td><td>2</td><td>2</td><td>160</td></tr><tr><td>200</td><td>3</td><td>3</td><td>2</td><td>184</td></tr><tr><td>240</td><td>4</td><td>3</td><td>3</td><td>232</td></tr><tr><td>320</td><td>5</td><td>4</td><td>4</td><td>304</td></tr></table>

## H Reproduction Notes

Reproduction scope. The submission source compiles directly from full\_naacl.tex; all tables are embedded in that file and the reported figures are supplied as PDF assets. Full regeneration of the figures and retraining additionally require the frozen diagnostic tensors, depth grids, PPO summaries, prompt examples, model and LRM checkpoints, launch commands, and software environment, which are maintained separately from this manuscript-only archive.

Data checks. Candidate-level source records for Figure 2(b)’s original example have not been located. Some original controlled-table MultiArith values differ slightly from later recomputed summaries; the original controlled values are retained. Diagnostic, confirmatory-grid, and test-budget summaries are analyzed under their respective protocols.