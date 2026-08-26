# RECURSIVE AGENTIC REASONING

Shengxin Zhang Google lezhend@gmail.com

Xiaomin Wu   
University of Maryland   
fred.xiaomin.wu@gmail.com

Xiyang Wu University of Maryland wuxiyang@umd.edu

Jing Xie University of Maryland jingxie@terpmail.umd.edu

## ABSTRACT

Test-time reasoning methods are usually studied in isolation: iterative refinement, decomposition, and repeated sampling each arrive with their own benchmarks, models, and graders, so their gains are hard to compare. We recast them as recursion operators over an agent’s reasoning trace — GROW (deepen one path), PRUNE (decompose and recompose), and BRANCH (sample alternatives and select) — and evaluate them against a single-pass chain-of-thought baseline in one shared harness with identical prompts, token budgets, and grading code. The study spans five benchmarks and three frontier models, yielding 14 model × benchmark cells, 49,327 graded items, and 151,876 model calls. Under a paired protocol that scores every operator on exactly the items resolved by all operators, BRANCH improves accuracy in all 14 cells (mean +5.98 points) and is the strict best operator in 12; GROW averages +2.18 and is negative in two cells, while PRUNE averages +0.94. The central analytical result is that BRANCH’s advantage is driven not only by marginalization over reasoning paths, but also by truncation recovery. Its gain strongly tracks the baseline’s rate of empty, budget-exhausted outputs (r = 0.72), indicating that repeated sampling often recovers answers that a single pass never emits. This finding weakens the routing hypothesis that motivated the study: at this operator granularity, one method dominates nearly everywhere. We also show that unpaired evaluation and scoring infrastructure failures as model errors are large enough to invert conclusions, and we recommend paired scoring as standard practice for comparative test-time-compute studies.

## 1 INTRODUCTION

A language model given one attempt at a hard problem produces one trace and one answer. Almost every recent advance in test-time reasoning can be read as a way of spending additional inference compute to avoid being stuck with that single trace. The model can extend the trace and revise its own answer (Madaan et al., 2023; Shinn et al., 2023; Shi et al., 2025); it can cut the problem into sub-problems and solve them in sequence (Zhou et al., 2023; Khot et al., 2023); or it can sample many independent traces and aggregate them (Wang et al., 2023; Brown et al., 2024; Chang et al., 2025). Recent work has pushed this space further with learned search controllers (Li, 2025), adaptive compute allocation (Wang et al., 2025; Bilal et al., 2026), and explicitly recursive agents that delegate to copies of themselves (Yang et al., 2026; Gandhi et al., 2026). These are different shapes of computation, and the empirical literature still largely treats them as different research programs.

That separation has a practical cost. Each method is typically introduced on the benchmarks and base models that suit it, with its own answer-extraction and grading code. A practitioner deciding how to spend a fixed inference budget therefore cannot answer a basic question: at equal compute, which of these should I use, and does the answer depend on my model or my task? Reported gains are not comparable, because nothing except the method is held fixed. This concern has only become sharper as recent framework papers argue that test-time scaling should be evaluated as a full inference system, with protocol-matched compute accounting and reproducibility requirements rather than a single scalar budget (Hariri et al., 2026).

This paper takes the position that the three families are better understood as three operators in a single space — recursion over an agent’s reasoning trace — and that the interesting question is empirical and comparative. We define GROW (additive recursion along one path), PRUNE (reductive recursion by decomposition), and BRANCH (recursion by search over sampled alternatives), implement them over a shared primitive, and run all three against a single-pass chain-of-thought baseline (Wei et al., 2022) on five benchmarks and three frontier models. Unlike recent work that trains recursive or policy-guided inference systems directly (Li, 2025; Yang et al., 2026; Gandhi et al., 2026), we ask a narrower but more immediately deployable question: if the base model is held fixed, which operator buys the most accuracy and why? Figure 1 shows the four computation graphs.

Holding everything but the operator fixed turns out to matter in ways we did not anticipate. Two methodological hazards, both invisible in a conventional results table, were individually large enough to reverse conclusions in our study. The first is unpaired evaluation: when long-running jobs against a shared inference endpoint fail on different items for different operators, comparing per-operator accuracy over whatever each run happened to complete silently compares methods on different problem sets. In our data this manufactured a spurious 3-point regression. The second is treating infrastructure failures as model errors, which penalizes exactly the operators that issue the most API calls — that is, the most expensive and often the best ones. We therefore adopt a paired protocol throughout: within each model × benchmark cell, every operator is scored on precisely the set of items that all operators resolved, and unrecoverable transport failures are excluded rather than marked wrong.

Our contributions are:

• A unified operator formulation of additive, reductive, and search-based test-time recursion over a reasoning trace, built on a shared solving primitive that includes an explicit finalization step for budget-exhausted reasoning models (Section 3).

• A controlled 14-cell comparison — 5 benchmarks × 3 models × 4 methods, 49,327 graded items, 151,876 model calls — with identical prompts, budgets, and graders, using a paired scoring protocol (Sections 4–5). BRANCH improves accuracy in all 14 cells and wins 12; GROW and PRUNE are inconsistent.

• A mechanistic account of why BRANCH wins that differs from the standard explanation. Its gain tracks the baseline’s rate of empty, budget-exhausted outputs at $r = 0 . 7 2$ , and it roughly halves that rate; the benefit is substantially recovery of a degenerate failure mode rather than marginalization over reasoning paths (Section 5.2).

• Two negative results. Adaptive operator routing is not supported by the data, because a single operator dominates; and we document how unpaired scoring inverted one of our own findings (Sections 5 and 6).

The rest of the paper is organized as follows. Section 3 formalizes the three operators and the shared solving primitive. Section 4 describes the benchmarks, models, and paired evaluation protocol. Section 5 gives the main comparative result, explains the mechanism behind BRANCH’s gains, and analyzes cost and cross-model effects. Section 6 situates the findings relative to the prior literature and makes the paper’s negative results explicit.

## 2 RELATED WORK

Prompting and decomposition. Chain-of-thought prompting established that eliciting intermediate steps improves reasoning (Wei et al., 2022). Least-to-most prompting (Zhou et al., 2023) and Decomposed Prompting (Khot et al., 2023) break a problem into sub-problems solved in sequence, each conditioned on earlier answers. Our PRUNE operator is a direct descendant of this line, with the decomposition produced by the same model that solves the sub-questions and an adaptive gate that falls back to a single direct solve when the model proposes only one sub-question.

Iterative refinement. Self-Refine (Madaan et al., 2023) and Reflexion (Shinn et al., 2023) improve an output by feeding the model its own critique. More recent work pushes refinement to finer granularity: Socratic Self-Refine decomposes a reasoning trace into verifiable sub-question/subanswer pairs, estimates step-level confidence, and selectively revises the weakest step (Shi et al., 2025). Our GROW operator is a deliberately minimal member of this family: it re-solves with the previous attempt in context and uses answer stability — two consecutive rounds extracting the same normalized answer — as its halting signal, which requires no separate critic and no verbal feedback channel.

![](images/f53d86d856a0066697533f4b3dfe41ca10b432f4235f352511c4e1ad7e06424d.jpg)  
Figure 1: The four inference-time computation graphs compared in this paper. Hollow nodes are the problem, filled nodes are model calls, and the bottom node is the returned answer. COT issues one call. GROW extends a single path and halts when the extracted answer stops changing between rounds. PRUNE decomposes into ordered sub-questions, solves each with the previous answers in context, and recomposes. BRANCH samples N = 5 independent solutions and selects by majority vote over normalized answers, which also discards samples that returned nothing. Call counts are means measured over all runs.

Sampling and search. Self-consistency samples multiple chains and takes a majority vote (Wang et al., 2023). Tree of Thoughts (Yao et al., 2023) and Graph of Thoughts (Besta et al., 2024) expand and prune partial reasoning states with lookahead and backtracking. Recent work spends branching compute more deliberately: Policy-Guided Tree Search learns when to expand, branch, backtrack, or terminate (Li, 2025), while step-level hybrid test-time scaling interleaves verifier-guided selfrefinement with Best-of-N and MCTS inside a single inference routine (Chang et al., 2025). Our BRANCH operator sits at the simple end of this spectrum: it is flat parallel sampling with an unweighted vote, and it is important to state plainly that as an algorithm it is self-consistency. We name it BRANCH for symmetry within the operator space, not to claim novelty; Section 6 discusses what does and does not distinguish our findings from that prior work.

Recursive agentic systems. Recent 2026 work turns recursion itself into the object of study. Recursive Models for Long-Horizon Reasoning formalize self-recursive agents as a route around bounded context and analyze their computational power (Yang et al., 2026), while Recursive Agent Optimization trains agents end-to-end to decide when and how to delegate to recursive copies of themselves (Gandhi et al., 2026). These works are complementary to ours: they study learned recursive policies directly, whereas we compare lightweight black-box recursion operators that can be layered onto existing frontier models without additional training.

Test-time compute scaling. Brown et al. (2024) show that coverage — the fraction of problems solved by at least one sample — scales log-linearly with sample count over four orders of magnitude, while noting that selection methods such as majority voting plateau beyond several hundred samples. Snell et al. (2024) show that the best way to spend test-time compute depends on problem difficulty, and that adaptively allocating it beats a fixed best-of-N policy. Recent work sharpens both allocation and evaluation: Every Rollout Counts studies optimal rollout allocation during search and argues for allocating budget at the reasoning-direction level rather than the solution level (Wang et al., 2025); Bilal et al. (2026) propose verifier-guided adaptive allocation across tools, exploration parameters, and iteration budgets; and Hariri et al. (2026) argue that test-time scaling should be reported as whole inference systems with protocol-matched compute accounting and reproducibility artifacts. This line of work motivates the routing hypothesis we set out to test; our data does not support it at the operator granularity, which we take to be an informative negative result rather than a contradiction, since we vary the operator rather than the sample budget. Verifier-guided selection (Cobbe et al.,

2021; Lightman et al., 2024) is also where much of the recent progress sits, and it is the main axis we do not explore: all of our selection is unweighted voting, which we return to as the clearest limitation of this study.

## 3 METHOD

## 3.1 AGENTIC RECURSION OVER A REASONING TRACE

Let x be a problem and let a model call $M ( p )$ map a prompt to a completion. A single-pass baseline computes $y = \operatorname { e x t r a c t } ( M ( x ) )$ ). We define a recursion operator as a policy that composes several such calls into one answer, and we characterize each operator by the computation graph it induces (Figure 1) and by its halting rule. We call the overall procedure agentic because the model is allowed to issue multiple calls conditioned on its own intermediate products rather than committing to a single trace.

## 3.2 SHARED SOLVING PRIMITIVE

All operators are built on one function, sol $\mathrm { v e } ( x , c )$ , which issues a single call with optional prior context c and returns an extracted answer. It contains one piece of machinery that proved essential and that we consider a contribution in its own right. Contemporary reasoning models emit a hidden deliberation stream before their user-visible content; if the token budget is exhausted middeliberation, the call returns successfully with empty content and a length stop reason. Treating this as a wrong answer — the default in most harnesses — discards an item on which the model may have reasoned correctly for thousands of tokens. When solve detects this condition it re-prompts the model with its own truncated reasoning and asks it to commit to a final answer. Section 5.2 shows that this failure mode is not a corner case: it accounts for a majority of baseline items on our hardest benchmark.

## 3.3 THREE RECURSION OPERATORS

GROW: additive recursion. GROW deepens a single path. Round t calls solve $( x , y _ { t - 1 } )$ , placing the previous attempt in context. It halts when $y _ { t } = y _ { t - 1 }$ under answer normalization, or at a cap of three rounds. The premise is that when a first attempt is under-determined — too little information, too short a chain — another pass with the attempt visible will improve it, and that agreement between consecutive rounds is a usable proxy for convergence.

PRUNE: reductive recursion. PRUNE reduces an over-complex problem. One call decomposes x into an ordered list of sub-questions $q _ { 1 } , \ldots , q _ { k } ; \operatorname { e a c h } q _ { i }$ is solved with the answers to $q _ { < i }$ in context, which matters for multi-hop problems where later hops depend on earlier ones; a final call composes the sub-answers into an answer to x. If the model proposes $k = 1$ , the operator degrades to a single direct solve rather than paying the decomposition overhead.

BRANCH: recursion by search. BRANCH explores alternatives. It draws $N = 5$ independent solutions at temperature 0.7, normalizes each extracted answer to a stable key, and returns the plurality key, recording the agreement ratio. Because empty completions normalize to a distinguished null key that never wins a vote unless every sample is empty, the vote structurally discards budgetexhausted samples.

## 3.4 SHARED CONTROL SIGNALS

The three operators differ in graph shape but share the same style of control signal: GROW halts on answer stability across rounds, PRUNE adapts its width to the model’s own proposed decomposition, and BRANCH records agreement across samples. All three signals are computed from extracted answers using the same normalizer, and none requires a trained verifier, a reward model, or access to ground truth. This is what makes the comparison in Section 5 apples-to-apples: any accuracy difference between operators is attributable to the shape of the recursion, not to a difference in supervision.

Table 1: Benchmarks. n is the number of items evaluated per operator per model. MuSiQue supplies all supporting and distractor paragraphs in context, so it measures multi-hop reasoning rather than retrieval. HLE is restricted to its text-only subset. <sup>†</sup>BBEH replaces each task of BIG-Bench Hard (Suzgun et al., 2023) with a harder counterpart probing the same skill.
<table><tr><td>Benchmark</td><td>Capability probed</td><td>n</td><td>Metric</td></tr><tr><td>MuSiQue (Trivedi et al., 2022)</td><td>Multi-hop composition</td><td>2,417</td><td>EM /F1</td></tr><tr><td>HLE (Phan et al., 2026)</td><td>Expert-level academic</td><td>2,158</td><td>Normalized EM</td></tr><tr><td>BBEH (Kazemi et al., 2025)</td><td>General many-hop†</td><td>200</td><td>Normalized EM</td></tr><tr><td>SuperGPQA (M-A-P Team et al., 2025)</td><td>Graduate knowledge</td><td>300</td><td>Multiple choice</td></tr><tr><td>Omni-MATH (Gao et al., 2025)</td><td>Olympiad mathematics</td><td>300</td><td>Normalized EM</td></tr></table>

## 4 EXPERIMENTAL SETUP

## 4.1 MODELS AND BENCHMARKS

We evaluate three frontier models — DeepSeek-V4-Pro, MiniMax-M3, and Qwen3.6-plus — accessed through a single internal OpenAI-compatible evaluation proxy, so that routing, retry policy, and token accounting are identical across models.<sup>1</sup> All three are reasoning models that emit a hidden deliberation stream, which is what makes the truncation behavior of Section 5.2 relevant.

Benchmarks were selected for headroom: we required that a strong model score well below ceiling, since a saturated benchmark cannot show a difference between test-time methods. An initial round on saturated mathematics sets was discarded for exactly this reason. Table 1 lists the final selection, which spans grounded multi-hop retrieval, expert-level closed-ended questions, general reasoning, graduate domain knowledge, and olympiad mathematics.

Grading is deliberately conservative: multiple-choice items are graded by letter extraction, and freeform items by symbolic and string normalization. For HLE this is a strict lower bound relative to the official LLM judge, so our absolute HLE numbers understate true accuracy. Because the same grader is applied to every operator within a cell, it does not bias the comparisons that this paper is about.

## 4.2 PAIRED EVALUATION PROTOCOL

Runs of this length against a shared endpoint fail intermittently. Across the full study, 709 items terminated in unrecoverable transport failures — read timeouts, connection resets, and provider rate limiting — after seven retries with exponential backoff. These are not model errors, and how they are handled changes conclusions. We adopt three rules.

Deduplicate by item. The runner is resumable and appends retried items, so a raw result file can contain several records for the same item. Naive line-level accounting is badly wrong: one of our files reads as 36.3% accurate by line count and 72.5% after deduplication. We keep one record per item, preferring a successful record over a failed one.

Exclude, do not penalize. An item that never produced a completion is dropped rather than scored as incorrect. Scoring it wrong systematically penalizes operators that issue more calls per item, since their probability of hitting at least one failure is higher — which is to say it penalizes BRANCH and PRUNE for being expensive.

Score on the paired set. Within each model × benchmark cell we intersect the resolved item sets of all four methods and report accuracy on that intersection. Every comparison in this paper is therefore between methods answering the identical questions. Table 2 reports the resulting n per cell; attrition is zero in six of the fourteen cells and below 1% in four more, and Appendix A gives the per-cell breakdown.

Section 6 describes the finding that this protocol overturned.

![](images/ccfa05b435b95860f3f2524de25c7eccdb2a6ff918bcd6aa175b98c8d90455e3.jpg)  
Figure 2: Accuracy change relative to the single-pass COT baseline in each of the 14 model × benchmark cells, sorted by BRANCH gain. Bars right of zero are improvements. BRANCH is positive in every cell; GROW and PRUNE each go negative in two. All three operators are scored on the same paired item set as the baseline within a cell.

Table 2: Accuracy (%) by benchmark, model, and operator, with the change from the COT baseline in parentheses. n is the paired item count (Section 4.2). Bold marks the best operator in each cell. Omni-MATH was not run on Qwen3.6-plus, leaving 14 of 15 possible cells.
<table><tr><td>Benchmark</td><td>Model</td><td>n</td><td>CoT</td><td>GROW</td><td>PRUNE</td><td>BRANCH</td></tr><tr><td>MuSiQue</td><td>DeepSeek-V4-Pro</td><td>2,417</td><td>67.94</td><td>66.69 (−1.25)</td><td>68.43 (+0.49)</td><td>72.36 (+4.42)</td></tr><tr><td>MuSiQue</td><td>MiniMax-M3</td><td>2,398</td><td>73.35</td><td>75.23 (+1.88)</td><td>74.23 (+0.88)</td><td>75.85 (+2.50)</td></tr><tr><td>MuSiQue</td><td>Qwen3.6-plus</td><td>2,395</td><td>77.08</td><td>77.49 (+0.41)</td><td>74.91 (−2.17)</td><td>78.46 (+1.38)</td></tr><tr><td>HLE</td><td>DeepSeek-V4-Pro</td><td>2,152</td><td>13.34</td><td>14.82 (+1.48)</td><td>14.68 (+1.34)</td><td>19.93 (+6.59)</td></tr><tr><td>HLE</td><td>MiniMax-M3</td><td>500</td><td>22.40</td><td>27.80 (+5.40)</td><td>23.80 (+1.40)</td><td>31.00 (+8.60)</td></tr><tr><td>HLE</td><td>Qwen3.6-plus</td><td>305</td><td>16.39</td><td>19.34 (+2.95)</td><td>15.08 (−1.31)</td><td>17.05 (+0.66)</td></tr><tr><td>BBEH</td><td>DeepSeek-V4-Pro</td><td>200</td><td>38.50</td><td>36.00 (−2.50)</td><td>37.50 (-1.00)</td><td>52.00 (+13.50)</td></tr><tr><td>BBEH</td><td>MiniMax-M3</td><td>200</td><td>26.50</td><td>32.50 (+6.00)</td><td>30.50 (+4.00)</td><td>37.00 (+10.50)</td></tr><tr><td>BBEH</td><td>Qwen3.6-plus</td><td>192</td><td>66.67</td><td>68.75 (+2.08)</td><td>68.75 (+2.08)</td><td>72.40 (+5.73)</td></tr><tr><td>SuperGPQA</td><td>DeepSeek-V4-Pro</td><td>300</td><td>58.33</td><td>59.00 (+0.67)</td><td>60.67 (+2.34)</td><td>65.33 (+7.00)</td></tr><tr><td>SuperGPQA</td><td>MiniMax-M3</td><td>299</td><td>58.53</td><td>62.54 (+4.01)</td><td>59.20 (+0.67)</td><td>65.22 (+6.69)</td></tr><tr><td>SuperGPQA</td><td>Qwen3.6-plus</td><td>300</td><td>69.67</td><td>72.00 (+2.33)</td><td>69.67 (+0.00)</td><td>72.00 (+2.33)</td></tr><tr><td>Omni-MATH</td><td>DeepSeek-V4-Pro</td><td>299</td><td>33.11</td><td>33.11 (+0.00)</td><td></td><td></td></tr><tr><td>Omni-MATH</td><td>MiniMax-M3</td><td>282</td><td>30.14</td><td>37.23 (+7.09)</td><td>35.79 (+2.68) 31.91 (+1.77)</td><td>39.46 (+6.35) 37.59 (+7.45)</td></tr></table>

## 5 RESULTS

## 5.1 OVERALL COMPARISON

Figure 2 and Table 2 give the main comparative result. BRANCH improves on the baseline in all 14 cells, with a mean gain of +5.98 points, a median of +6.47, and a range from +0.66 to +13.50. It is the strict best of the three operators in 12 cells, ties for best in one, and is beaten in one. GROW averages +2.18 and is positive in 11 of 14, but it is negative on DeepSeek-V4-Pro for both MuSiQue (−1.25) and BBEH (−2.50), so it cannot be recommended unconditionally. PRUNE averages +0.94 with a range of −2.17 to +4.00 — an effect small enough that, at the sample sizes of our smaller cells, much of it is difficult to distinguish from noise.

![](images/c27a6715b0628f40992a15655ac54615425d5cde2fe7853b4439e9cfd953623e.jpg)

![](images/30bbce395d135ef248b5339f2f2ce9b56b0c1c16b46257a591f3449042503c09.jpg)  
Figure 3: (a) Per-cell BRANCH gain against the baseline’s empty-output rate, with a least-squares fit; the association is strong $( r = 0 . 7 2 , n = 1 4 )$ . The four cells at zero on the horizontal axis are the Qwen3.6-plus cells, which never truncated, and they show the smallest gains. (b) Empty-output rate under the baseline and under BRANCH for the ten cells where truncation occurs at all. BRANCH roughly halves it everywhere.

The two cells BRANCH does not win are both Qwen3.6-plus, and they are instructive rather than contradictory. On SuperGPQA, GROW and BRANCH tie exactly at +2.33. On HLE, GROW leads with +2.95 against BRANCH’s +0.66, in the one cell with the heaviest attrition $( n = 3 0 5$ of 500 attempted). Qwen3.6-plus is also the only model in our study that never produced an empty completion, which Section 5.2 identifies as the precise condition under which BRANCH’s advantage should be expected to shrink.

The routing hypothesis is not supported. This study was designed to test whether the best operator depends on the problem, which would motivate an adaptive router that selects an operator per item. The data argues against that hypothesis at this granularity. BRANCH is best in all five DeepSeek-V4-Pro cells and all five MiniMax-M3 cells; only Qwen3.6-plus splits, two to two, and one of those two is an exact tie. A router trained on these outcomes would learn a nearly constant policy. We regard that negative result as more informative than a router built anyway on the two cells that would support it.

## 5.2 WHY BRANCH WINS: TRUNCATION RECOVERY

Self-consistency is usually explained as marginalizing over reasoning paths: a hard problem admits many routes to one correct answer, so agreement across sampled routes is evidence for that answer (Wang et al., 2023). Our data indicates that in this setting a different mechanism carries much of the gain.

The dominant baseline failure is not an incorrect answer but an absent one. Under a single pass, DeepSeek-V4-Pro returns empty content on 51.2% of HLE items: the model exhausts a 16,000- token budget inside its hidden deliberation stream and never reaches user-visible content. The rate is 34.5% on BBEH and 36.1% on Omni-MATH. Sampling five times converts this from a near-coinflip into a minority event, and because empty answers cannot win a plurality vote, they are discarded automatically.

Figure 3 makes the case quantitatively. Panel (a) shows that a cell’s BRANCH gain is strongly associated with how often the baseline truncated $( r = 0 . 7 2$ over 14 cells). Panel (b) shows the reduction directly: on HLE with DeepSeek-V4-Pro the empty rate falls from 51.2% to 32.2%, on BBEH with MiniMax-M3 from 35.0% to 15.5%, and the rate drops in all ten cells where truncation occurs. The cleanest evidence is the negative case. Qwen3.6-plus never truncated on any benchmark, and it is exactly where BRANCH’s gains are smallest (+0.66, +1.38, +2.33, +5.73) and where the one non-tied loss to GROW occurs.

![](images/cba25d967648569597fc56eba5ed2fd323a9a9400edaaae88993259036367b03.jpg)

![](images/af677c8d12791a86cd086525f4906f01cd5ccc3fb3c3b35d74f4bdfc68c0aa8a.jpg)  
Figure 4: Mean accuracy gain against mean model calls per item, averaged over all 14 cells. Dotted guides mark constant returns per call. BRANCH buys the most accuracy; GROW is marginally more efficient per call but has a far lower ceiling; PRUNE is dominated on both axes.  
Figure 5: HLE accuracy on the 305 items resolved by every model × operator run. Restricting to a shared item set changes the model ranking relative to the unpaired numbers in Table 2, because the 500-item subset attempted by two of the models is easier than the full split.

We are not claiming that marginalization contributes nothing — Qwen3.6-plus still gains from BRANCH on three of four benchmarks with no truncation to recover. The claim is narrower: the two mechanisms are separable, they are conflated in aggregate accuracy numbers, and in a regime of long-reasoning models under finite budgets the recovery term is large. This has a practical consequence: a substantial fraction of the reported benefit of repeated sampling on such models may be obtainable far more cheaply by a targeted finalization pass on truncated generations, of the kind our solve primitive performs.

## 5.3 COMPUTE, EFFICIENCY, AND CROSS-MODEL COMPARISON

Cost. The operators are not compute-comparable, and Figure 4 places them on the accuracyversus-calls plane. GROW averages 2.23 calls per item, PRUNE 4.04, and BRANCH 4.85, against one for the baseline. Dividing mean gain by extra calls, GROW returns +1.77 points per additional call, BRANCH +1.55, and PRUNE +0.31. So BRANCH’s dominance is a dominance in achievable accuracy rather than in efficiency: it wins because it can spend more usefully, not because each call is better spent. PRUNE is dominated on both axes and we see no budget at which we would recommend it.

The obvious inefficiency is that BRANCH uses a fixed N = 5 with no early stopping. Because the agreement ratio is already computed per item, halting once a majority is mathematically decided would cut a large share of the cost with no change in the returned answer. We did not implement this, and it is the first thing we would add.

Cross-model comparison requires the same items. The full HLE split was run only on DeepSeek-V4-Pro; the other two models were run on a 500-item subset. The per-model HLE num bers in Table 2 are therefore not comparable across models, and the direction of the error is not obvious in advance. Figure 5 restricts all three models and all four operators to the 305 items every run resolved. On that shared set, DeepSeek-V4-Pro’s baseline rises from 13.34% to 17.38%, because the 500-item subset is easier than the full split, and MiniMax-M3 leads at 31.80%. Any cross-model claim from the unpaired table would have been wrong about both the gap and the ordering.

## 6 DISCUSSION

## 6.1 WHAT IS AND IS NOT NEW

BRANCH is self-consistency (Wang et al., 2023). GROW is a stripped-down member of the Self Refine family (Madaan et al., 2023), and PRUNE is least-to-most decomposition (Zhou et al., 2023). We claim novelty for none of the three as algorithms. The paper’s contribution is instead the controlled comparison that the field’s method-at-a-time convention makes unavailable, the finding that a large part of repeated sampling’s benefit on long-reasoning models is truncation recovery rather than path marginalization, and a negative result on operator routing. Relative to Brown et al. (2024), who show that majority voting plateaus as sample counts grow into the hundreds, our regime is the opposite extreme — N = 5 — and our finding is about where the early gains come from.

## 6.2 THE CONFOUND THAT INVERTED A RESULT

An earlier analysis of our own data showed Qwen3.6-plus on HLE dropping from 14.20% under the baseline to 11.20% under BRANCH, which reads as a real and interesting regression: an operator that helps everywhere else actively hurting one model. It was an artifact. The BRANCH run lost 172 of 500 items to read timeouts against the proxy, against 131 for the baseline, and unresolved items were being scored as incorrect. On the 305 items all four Qwen runs completed, BRANCH is +0.66. The magnitude of this artifact — roughly 3.6 points, larger than most true effects in Table 2 — is our argument for the protocol in Section 4.2. We suspect unpaired scoring is common in multi-method evaluations run against shared or rate-limited endpoints, and that it is rarely reported.

## 6.3 LIMITATIONS

Selection in BRANCH is an unweighted majority vote. Verifier- or confidence-weighted selection (Cobbe et al., 2021; Lightman et al., 2024) is strictly more expressive, and we record an agreement ratio per item that we never use for weighting; this is the largest gap between our BRANCH and the state of the art. Relatedly, BRANCH is flat parallel sampling, not tree search: it does not expand or prune partial reasoning states as ToT and GoT do (Yao et al., 2023; Besta et al., 2024), so the name describes its position in our operator space rather than its algorithmics. We report no significance tests; a single accuracy figure from a 200-item cell carries a 95% interval of roughly ±7 points, and while paired deltas are considerably tighter, we did not quantify them, so small effects in the BBEH and Omni-MATH cells should be read as provisional. Our HLE grader is a lower bound. Omni-MATH was not run on Qwen3.6-plus. Finally, all three models are reasoning models that emit hidden deliberation, and the truncation mechanism of Section 5.2 may not transfer to models without that behavior — although the Qwen3.6-plus cells offer a partial preview of that regime.

## 6.4 WHY THE PILOT MISLED US

A 50-item pilot preceding the full runs showed GROW at +4, PRUNE at +5.3 on MuSiQue but −2 on HLE, and BRANCH at +8 — a pattern of task-dependent operator effectiveness that motivated the routing hypothesis. At full scale these deltas roughly halved and the PRUNE sign flip disappeared. At n = 50 the standard error alone is near 7 points, comparable to every effect being measured. We report this because pilots of that size remain common when inference is expensive, and ours would have supported a conclusion the full data contradicts.

## 7 CONCLUSION

Treating additive refinement, decomposition, and repeated sampling as three operators in one space, and measuring them under identical conditions, gives a clearer picture than the method-at-a-time literature allows. Across 14 model × benchmark cells, sampling and voting improve accuracy every time, by a mean of six points, while deepening a single path and decomposing into sub-questions are inconsistent and sometimes harmful. The reason sampling wins, however, is not only the one usually given: on long-reasoning models under finite token budgets, much of its benefit is the recovery of answers that a single pass never emitted at all. That points somewhere more specific than “sample more” — namely at the budget-exhaustion failure itself, which is cheaper to attack directly. Our routing hypothesis did not survive contact with the full data, and neither did one of our own intermediate results, which we take as evidence that paired scoring and explicit separation of infrastructure from model failure deserve to be default practice in comparative test-time-compute studies.

## REPRODUCIBILITY STATEMENT

Section 4 specifies models, benchmarks, decoding parameters, and token budgets; Section 4.2 specifies the deduplication, exclusion, and pairing rules used for every number reported. Per-item records — extracted answer, gold answer, round count, agreement ratio, full trace, and latency — were retained for all 49,327 graded items, so every figure in this paper can be recomputed without requerying any model. Appendix A reports per-cell attrition. Figures are regenerated from the released numbers by a single script.

## REFERENCES

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2024. arXiv:2308.09687.

Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Ali Subhan, Hassan Rizwan, Ayesha Mohsin, and Dean F. Hougen. What if we allocate test-time compute adaptively? arXiv preprint arXiv:2602.01070, 2026.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Re, and´ Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Kaiyan Chang, Yonghao Shi, Chenglong Wang, Hang Zhou, Chi Hu, Xiaoqian Liu, Yingfeng Luo, Yuan Ge, Tong Xiao, and JingBo Zhu. Step-level verifier-guided hybrid test-time scaling for large language models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 18462–18477, 2025. doi: 10.18653/v1/2025.emnlp-main. 931.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Apurva Gandhi, Satyaki Chakraborty, Xiangjun Wang, Aviral Kumar, and Graham Neubig. Recursive agent optimization. arXiv preprint arXiv:2605.06639, 2026.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omni-MATH: A universal olympiad level mathematic benchmark for large language models. In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.07985.

Mohsen Hariri, Weicong Chen, Nahal Shahini, Vikash Singh, Kai Ye, Amirhossein Samandar, Debargha Ganguly, Sreehari Sankar, Yanyan Zhang, Shouren Wang, Jerry Peng, Biyao Zhang, Michael Hinczewski, and Vipin Chaudhary. Test-time scaling in reasoning LLMs: Inference regimes, evaluation, and reproducibility. arXiv preprint arXiv:2608.04001, 2026.

Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou, Sanket Vaibhav Mehta, Lalit K. Jain, Virginia Aglietti, Disha Jindal, Peter Chen, Nishanth Dikkala, Gladys Tyen, Xin Liu, Uri Shalit, Silvia Chiappa, Kate Olszewska, Yi Tay, Vinh Q. Tran, Quoc V. Le, and Orhan Firat. BIG-Bench extra hard. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 26473–26501, 2025. doi: 10.18653/v1/2025.acl-long.1285.

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. In International Conference on Learning Representations (ICLR), 2023. arXiv:2210.02406.

Yang Li. Policy guided tree search for enhanced LLM reasoning. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025. arXiv:2502.06813.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations (ICLR), 2024. arXiv:2305.20050.

M-A-P Team, Xinrun Du, Yifan Yao, Kaijing Ma, et al. SuperGPQA: Scaling LLM evaluation across 285 graduate disciplines. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. arXiv:2502.14739.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2303.17651.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, et al. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649:1139–1146, 2026. doi: 10.1038/s41586-025-09962-4. arXiv:2501.14249.

Haizhou Shi, Ye Liu, Bo Pang, Zeyu Leo Liu, Hao Wang, Silvio Savarese, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. SSR: Socratic self-refine for large language model reasoning. arXiv preprint arXiv:2511.10621, 2025.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2303.11366.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,¨ Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, 2023. arXiv:2210.09261.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Compu tational Linguistics, 10:539–554, 2022. doi: 10.1162/tacl a 00475.

Xinglin Wang, Yiwei Li, Shaoxiong Feng, Peiwen Yuan, Yueqi Zhang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, and Kan Li. Every rollout counts: Optimal resource allocation for efficient test-time scaling. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2506.15707.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations (ICLR), 2023. arXiv:2203.11171.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2201.11903.

Chenxiao Yang, Nathan Srebro, and Zhiyuan Li. Recursive models for long-horizon reasoning. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. arXiv:2603.02112.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2305.10601.

Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schu-¨ urmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In International Conference on Learning Representations (ICLR), 2023. arXiv:2205.10625.

## A PER-CELL ATTRITION

Table 3 reports, for each cell, the number of items attempted and the number in the paired set used for scoring. Attrition is the count of items that at least one of the four operators failed to resolve after seven retries with exponential backoff. All failures were transport-level: read timeouts against the proxy (600 s read timeout), connection resets, and provider-side rate limiting at 30 requests per minute for one model.

Table 3: Attempted versus paired item counts. Six of the fourteen cells are unaffected. The HLE Qwen3.6-plus cell is the outlier and is the cell discussed in Section 6.
<table><tr><td>Benchmark</td><td>Model</td><td>Attempted</td><td>Paired n</td><td>Attrition</td></tr><tr><td>MuSiQue</td><td>DeepSeek-V4-Pro</td><td>2,417</td><td>2,417</td><td>0</td></tr><tr><td>MuSiQue</td><td>MiniMax-M3</td><td>2,417</td><td>2,398</td><td>19</td></tr><tr><td>MuSiQue</td><td>Qwen3.6-plus</td><td>2,417</td><td>2,395</td><td>22</td></tr><tr><td>HLE</td><td>DeepSeek-V4-Pro</td><td>2,158</td><td>2,152</td><td>6</td></tr><tr><td>HLE</td><td>MiniMax-M3</td><td>500</td><td>500</td><td>0</td></tr><tr><td>HLE</td><td>Qwen3.6-plus</td><td>500</td><td>305</td><td>195</td></tr><tr><td>BBEH</td><td>DeepSeek-V4-Pro</td><td>200</td><td>200</td><td>0</td></tr><tr><td>BBEH</td><td>MiniMax-M3</td><td>200</td><td>200</td><td>0</td></tr><tr><td>BBEH</td><td>Qwen3.6-plus</td><td>200</td><td>192</td><td>8</td></tr><tr><td>SuperGPQA</td><td>DeepSeek-V4-Pro</td><td>300</td><td>300</td><td>0</td></tr><tr><td>SuperGPQA</td><td>MiniMax-M3</td><td>300</td><td>299</td><td>1</td></tr><tr><td>SuperGPQA</td><td>Qwen3.6-plus</td><td>300</td><td>300</td><td>0</td></tr><tr><td>Omni-MATH</td><td>DeepSeek-V4-Pro</td><td>300</td><td>299</td><td>1</td></tr><tr><td>Omni-MATH</td><td>MiniMax-M3</td><td>300</td><td>282</td><td>18</td></tr></table>

## B OPERATOR PSEUDOCODE

Algorithm 1 The three recursion operators. solve is the shared primitive of Section 3, including the   
finalization retry for budget-exhausted generations. norm is the shared answer normalizer.   
1: function $\overline { { \mathrm { G R O W } ( x , T = 3 ) } }$   
2: y ← solve(x, ∅)   
3: for t = 1 to T − 1 do   
4: y<sub>t</sub> ← solve $( x , y _ { t - 1 } )$   
5: if norm $\cdot ( y _ { t } ) = \mathrm { n o r m } ( y _ { t - 1 } )$ then return y<sub>t</sub> ▷ answer stable   
6: end if   
7: end for   
8: return $y _ { T - 1 }$   
9: end function   
10: function PRUNE(x)   
11: q<sub>1</sub>, . . . , q<sub>k</sub> ← decompose(x)   
12: if $k = 1$ then return solve(x, ∅) ▷ adaptive fallback   
13: end if   
14: for i = 1 to k do   
15: $a _ { i } \gets \mathrm { s o l v e } ( q _ { i } , \{ ( q _ { j } , a _ { j } ) \} _ { j < i } )$ ▷ later hops see earlier answers   
16: end for   
17: return compose $( x , \{ ( q _ { i } , a _ { i } ) \} )$   
18: end function   
19: function BRANCH(x, $N = 5 , \tau = 0 . 7 )$   
20: $s _ { 1 } , \ldots , s _ { N }  N$ independent draws of solve $_ { \mathrm { : } ( x , \emptyset ) }$ at temperature τ   
21: return arg max<sub>k</sub> |{i : norm(s<sub>i</sub>) = k}| ▷ empty answers normalize to a null key   
22: end function