# Agent Benchmarks Stop Ranking Before They Run Out of Tasks

Zihan Dong<sup>1</sup>, Qishi Zhan<sup>2</sup>, and Minxuan Hu<sup>3</sup>

<sup>1</sup> Lenovo Group

puma122707@gmail.com

<sup>2</sup> Marquette University

<sup>3</sup> Cornell University

Abstract. Data-science and ML-engineering agents are evaluated on benchmarks such as MLE-bench, ALE-Bench, SWE-ficiency and Algo-Tune, whose tasks yield a quantity (a Kaggle metric, contest score or speedup). When one stops separating the best models, the usual remedy is a harder version. Re-scoring released runs of eight benchmarks, includ ing SWE-bench Verified and Terminal-Bench for breadth, we find much of this saturation lies in the measurement, not the tasks. Most agent benchmarks reduce a run to one pass/fail bit, which fails in two ways. It is too coarse: neighbouring top-ten agents on SWE-bench Verified difer by less than its tests’ errors, and correcting them reorders neighbours no more than deleting a random sixth of each agent’s passes. It is capped at the reference: ALE-Bench’s top ten, separated by contest score, tie when graded against the median human, although nearly half the runs beat that human. On fixed tasks, a higher bar only moves the problem; graded credit fixes it. We derive a clipped log-ratio score with dated anchors and cost as a separate axis. In a Terminal-Bench re-run, pass/fail-tied models difer over tenfold on the quantity the tests discard.

Keywords: Agent evaluation · Data-science agents · Benchmark saturation · Verifiers

## 1 Introduction

Data-science and ML-engineering agents are benchmarked on Kaggle competitions (MLE-bench [6]), optimization contests (ALE-Bench [15]) and code speedups (SWE-ficiency [23], AlgoTune [30]). Consider ALE-Bench, 40 contests scored by contest performance: its top ten separate into three tiers. Scored pass/fail against the median human, as most agent benchmarks score runs, five models pass 39 of the 40 problems and the top ten become one tier, although nearly half the runs beat that human. On the same tasks, a higher bar does not fix this: one that separates the newest models scores 89% of the earlier ones zero. A single pass/fail bar per task sees one band of ability; graded credit widens it.

The symptom is general: the last flagship model card to report SWE-bench Verified appeared in April 2026 (Fig. 1b), when its top ten solved 76–80% of the tasks: it ran out of ranking, not tasks. Across 57 flagship cards from seven labs, only a third of a card’s agentic results are comparable with the same lab’s previous flagship (same benchmark, version, subset and harness; Fig. 1a).

(a) More benchmarks, fewer comparable numbers (b) SWE benchmarks reported per card  
![](images/92d0f4edf1ad9c851f06e8ff933d6c765ba72d6ac96d1c114f3413b3991246fd.jpg)

![](images/4f7e19f5c672e41eb388bb7c22d4386798ef13d599d89e605341b586c5d83262.jpg)  
Fig. 1. Benchmark turnover in 57 flagship model cards (2024–2026). (a) Agentic benchmark families reported per half-year, and the share of a card’s agentic results comparable with the lab’s previous flagship. (b) Which SWE-style benchmark each card reports.

Turnover has many causes, contamination among them; we use it as motivation only. The usual reading of saturation is that the tasks became too easy, and the usual remedy is a harder version. We find that much of it lies in the measurement. Almost every agent benchmark grades a run against an answer key written for each task (a reference solution, an expected end state, or tests derived from the author’s solution), which returns one bit. The key is also the costliest and least verifiable part of a benchmark: a checker that reliably separates good answers from bad ones is case-specific, slow to write and still often wrong [4,40,10], and OSWorld reports twice as much efort per task on the evaluator as on the task [34]. Where no key can be written, the fallback is an LLM judge, which a single token can fool [39].

The answer key thus fails in two separate ways. Failure A, the bit is too coarse (§3): on three leaderboards, neighbouring agents in the top ten difer by less than a few hundred bits can detect, and on SWE-bench Verified by less than the tests’ errors. Failure B, the bit is capped (§4–6): a run that beats the reference, or passes at a fifth of the cost, scores the same as one that just passes.

Both failures come from one design choice: the key encodes an answer rather than a preference. An answer must be written for every task and stops informing once agents reach it. A preference (“faster is better”) can be scored once per domain, with room above any reference. Reinforcement learning made this switch when agents passed humans [28], as did optimization benchmarking [5,14].

Contributions. (1) On released runs of eight benchmarks and a Terminal-Bench re-run, we measure both failures, locate the lost information above the reference, and report where it does not help (§3–5). (2) We show why, on a fixed task set, a higher pass/fail bar cannot fix a saturated key while graded credit can, and derive the form of the graded score (§6). (3) We derive an evaluation methodology with a hardened, auditable gate, an anchored objective, cost as a separate reported axis, and frozen anchors that keep versions reproducible and comparable, instantiated in a small re-run; LLM judges help write evaluators but do not score (§7). We release code and derived statistics.

## 2 Related Work

Saturation. Saturation of 60 LM benchmarks has been tested against answer format, access, size and age [3], not against how the score is built. Two 2026 analyses argue that hard answer keys become unafordable or impossible once models exceed their authors [12,25]. Evaluation results need error bars [27,24], and top SWE-bench submissions are statistically tied [22]; we find the same for neighbouring agents on two more leaderboards and relate it to verifier error. Item response theory selects tasks of informative dificulty [29], and test equating links versions [19]; we test calibrated per-task bars and link versions through anchors. Surveys of agent evaluation report fragile evaluators, contamination and fast-moving versions [9], and rankings are fragile to the wording of the task [38].

Verifiers and judges. Weak tests let wrong patches pass on SWE-bench [4,37,36], false fails are common [10], and seven of ten audited agent benchmarks have flawed outcome checks [40]. This work makes the bit valid and, in our design, hardens the gate; we ask what the score should measure once it is.

Beyond pass/fail. Some leaderboards report the dollar cost of each agent beside its accuracy [18,17,11]; time horizons express success in human time [20]. Optimization benchmarks score an objective in single domains [15,30,23,6], and NPPC pairs one checker with generated instances [35]. These rules are not new; we show when they matter and why a new version cannot replace them.

## 3 Failure A: The Bit Is Too Coarse to Rank the Frontier

A leaderboard ranks agents only if the gaps between neighbours exceed two errors: statistical error from a finite task set, and verifier error from a key that accepts wrong solutions or rejects right ones. For SWE-bench Verified [16] we use per-instance verdicts for the oficial top 30 (1 Sep 2026), whose passing patches SWE-ABS [36] re-checked with strengthened tests. For Terminal-Bench 2.0 [26] we use per-trial results of 61 submissions, after removing three flagged for cheating. For OSWorld-Verified [34] the top ten, seven with per-task trajectories.

We group agents into tiers, sets of agents that the benchmark cannot tell apart, by walking down the ranking: an agent starts a new tier if it is significantly worse than the current tier’s leader. As that leader is the best observed agent, we test all pairs at once, by a step-down max-T test on paired t-statistics with a null from permuting agent labels within tasks [33,31]. This bounds the chance of any false split at 5% (3–6% for ten equal agents in simulation, against 11–24% for Holm over the walk’s own comparisons).

![](images/bb1e8f307741dffdfbf3e2882c6fd9461d2ba83971e6766e82bac1702e054b61.jpg)  
Fig. 2. The top ten agents on three leaderboards, with 95% confidence intervals. Colours mark tiers; tier 1 cannot be separated from the leader. OSWorld-Verified tiers use the seven agents with per-task data.

Neighbours at the top are statistically tied. In September 2024, the SWE-bench Verified top ten spanned 19.2 points and three tiers; since March 2025 they have formed one or two (today 3.6 points, one tier; two in 59% of task resamples). Today Terminal-Bench 2.0’s top ten form one tier and OSWorld-Verified’s two (Fig. 2). On SWE-bench Verified and Terminal-Bench 2.0, none of the 45 pairs within the top ten difers significantly (Holm over all 45), and on all three leaderboards no neighbouring pair does; on OSWorld-Verified a leading pair stands apart from a tied pack. The test is not weak: it detects about 6.3 points with 80% power and splits ten real entries spaced 5.7 points apart into ten tiers, but the largest gap between SWE-bench neighbours is 1.4 points. Nor is it a ceiling: together the top ten solve 91% of the tasks.

The tests’ errors are larger than the gaps. On SWE-bench Verified, strengthened tests reject 17.7% of the median agent’s passing patches (range 10.8–23.2%). The rate difers between agents beyond sampling noise $( \chi _ { 2 9 } ^ { 2 } = 7 7 . 3$ , permutation $p < 1 0 ^ { - 4 } )$ , with a spread of about 2 points, while neighbours are typically 0.2–0.4 points apart. Correcting the verdicts shrinks the top tier of the top 30 from 23 agents to a leading group of five, so a better key changes who is in it. The 18 agents that leave it stay nearly tied among themselves (1 of 153 pairs separates), and no neighbouring pair separates. The correction reverses 8 of the 18 neighbouring orders with a non-zero gap (Kendall $\tau = 0 . 5 1$ between the two leaderboards), although in 11 of these pairs the two agents’ inflation difers significantly and by more than the gap. Removing the same share of passes at random reverses about as many (8; 95% band 4–11). So the order of neighbours is noise, which any perturbation of this size reorders. False fails, about one in six failed runs in an OSWorld audit [10], add to this error; none are released per agent, and simulated, they separate no neighbouring pair at a common rate (5–25%) and at most 9% of runs at agent-specific rates.

This failure has nothing to do with the cap. Its remedies are a stronger gate, more tasks or trials, more information per task, and tiers instead of ranks.

Table 1. Same runs, scored by each benchmark’s oficial metric and three ways of ours. “Beat ref.”: share of runs above the objective’s reference (median human, expert speedup, human baseliner, reference solver, gold medal). Cells: tiers among the top ten (more is a finer ranking), with S in brackets (lower is less saturated). SWE-ficiency: seven models; 430 of 498 tasks, those with expert speedup ≥ 1.2 (oficial score: all 498). <sup>†</sup>Oficially pass/fail. <sup>∗</sup>Released capped at the reference.
<table><tr><td>Benchmark</td><td>Models</td><td>Tasks</td><td>Beat ref.</td><td>Official</td><td>Answer key</td><td>Capped</td><td>Objective</td></tr><tr><td>ALE-Bench</td><td>145</td><td>40</td><td>47.8%</td><td>3 (0.07)</td><td>1 (1.00)</td><td>1 (0.93)</td><td>3 (0.03)</td></tr><tr><td>SWE-fficiency</td><td>7</td><td>430</td><td>36.3%</td><td>61 (0.95)</td><td>4 (0.10)</td><td>5 (0.01)</td><td>4 (0.02)</td></tr><tr><td>AlgoTune</td><td>18</td><td>154</td><td>70.6%</td><td>62 (0.51)</td><td>2 (0.63)</td><td>2 (0.88)</td><td>3 (0.58)</td></tr><tr><td>METR†</td><td>20</td><td>210</td><td>3.7%</td><td>= key</td><td>3 (0.51)</td><td>3 (0.55)</td><td> $\mathrm { n / a ^ { * } }$ </td></tr><tr><td>MLE-bench†</td><td>27</td><td>74</td><td>10.8%</td><td>= key</td><td>4 (0.10)</td><td>3 (0.52)</td><td>3 (0.73)</td></tr></table>

## 4 Failure B: The Bit Is Capped at the Reference

If a key stops resolving the frontier because it is capped, the same runs scored with room above the cap should resolve it again, where the cap binds. We use every benchmark we found that releases per-run, per-task continuous results; 4% to 71% of their runs beat the reference (Table 1).

Two of these benchmarks oficially grade pass/fail per task: MLE-bench [6] by medals, and METR [20] by binarized success. The other three oficially score a continuous quantity; they let us ask what they would report had they been built as answer keys, as most agent benchmarks are. We score the same released runs three ways. The answer key is the oficial binary verdict where one exists, and otherwise whether the run reaches the reference: the median human contestant on ALE-Bench [15], the expert patch’s speedup on SWE-ficiency [23], and a 1.1-fold speedup over the reference solver on AlgoTune [30]. The capped score gives partial credit up to the reference and none beyond it. The objective is the score of §6: 0 for runs that fail the benchmark’s own validity check or do no better than a baseline, 1 at the same reference, and above 1 for runs that beat it, up to a clip at 3. The baseline is the bottom-quartile human on ALE-Bench, no speedup on SWE-ficiency, a zero speedup on AlgoTune, and the median submission on MLE-bench; on AlgoTune, whose baseline is zero, $\log _ { 2 } ( 1 + \mathrm { s p e e d u p } )$ replaces the log-ratio.

We also report a paired variant of the saturation index of $[ 3 ] , S = \exp ( - R ^ { 2 } )$ , where R is the mean gap between the first and fifth model over the standard deviation of their per-task diferences, times $n ^ { 1 / 4 }$ for n tasks: near 1, the top five are within noise; near 0, clearly separated.

Binarize a benchmark that resolves, and it saturates. ALE-Bench’s oficial score, mean contest performance, separates its top ten into three tiers. Scored as an answer key against the median human, the same runs give one tier (S = 1.00), and the objective restores three (S = 0.03; with 40 tasks, robustly at least two). Across task resamples, the key gives one tier every time and the objective at least two 99.6% of the time. Partial credit below the reference does not help:

![](images/343d994bc007b39ab47ea4da12aab3284cfcfc002db02cc240aef495ef04ee1b.jpg)  
Fig. 3. (a) What each task’s test checks. (b) Re-run: all runs of the three models that pass/fail ties, on four tasks where all pass, as the quantity relative to the reference solution (higher is better: speedup, inverse MSE or keystrokes). Right: best over worst model median. (c) Median tokens per run of the tied Terminal-Bench 2.0 top ten.

the capped score still gives one tier. Keeping the bit below the reference and crediting only the margin above it gives all three.

Where it is not saturated, the margin does not help. On SWE-ficiency, partial credit below the reference helps most, and the oficial score, a harmonic mean in which failed tasks dominate, resolves worst of all (one tier): how a continuous score is aggregated matters as much as whether it is continuous. On METR, few runs beat the reference. On MLE-bench the key already separates the top (S = 0.10), and the heavy tail of Kaggle metrics blurs it (S = 0.73). The margin above the reference carries information mainly where the key has run out.

## 5 Failure B on Terminal-Bench, and the Ignored Cost

What the tests compute. Consider the Terminal-Bench 2.0 task query-optimize: the agent must speed up a slow SQL query. The test runs the agent’s query, measures its runtime, and passes it if the runtime is below a threshold. The runtime is computed and then thrown away. We coded the tests of all 89 Terminal-Bench 2.0 tasks and samples of SWE-bench Verified (30), OSWorld-Verified (30) and τ<sup>2</sup>-bench (20) as T (a quantity compared with a threshold), S (a unique correct end state) or F (other). On Terminal-Bench 2.0, 18 tasks (20%) are T. The other samples contain no T tasks (Fig. 3a; with samples this small, true rates up to 10–15% cannot be excluded). For them our design ofers a hardened gate and a cost meter, but no quality objective unless the task is restated around a quantity, as SWE-ficiency does [23].

A controlled re-run. Released logs record the quantity for only four T tasks, on difering machines, so we re-ran 13 T tasks with tests changed only to print it. The re-run is a small instance of the design of §7: the unchanged test is the gate, the printed quantity the objective, and per-trial dollars and tokens the meter. We ran four models (gpt-5.3-codex, claude-opus-5, gpt-6-astra, gemini-3.1-pro; codex, opus, astra, gemini below), reached through a third-party OpenAI-compatible gateway, three trials per task in the Terminus-2 agent [26] (156 runs, about \$115 at list prices). Each run had its own CPU core. Pass/fail ties opus, astra and gemini and separates codex (within-task permutation tests, Holm).

Same pass, an order of magnitude apart. The tied three difer widely on the quantity (Fig. 3b): on portfolio-optimization all their runs pass, yet opus’s median speedup is 10.9× and the others’ 1.3×, the reference’s level; on pytorch-model-recovery the median MSE spans 0.016 to 0.60. Of all passing runs, 69% beat the reference solution. Ranking models is harder: an anchor-free permutation test (failures at the threshold, standardized log quantity, Holm) separates four of six pairs, pass/fail three, and the one tied pair it separates (opus and gemini, Holm $p = 0 . 0 3 1 )$ is not separated on passing runs alone. With three trials per cell, the robust evidence is per task. Two codex runs passed by reward hacking, more verifier error (§3), and both stood out in the printed quantity; counted as failures, they leave codex below its leaderboard entry on these tasks (stratified $p = 0 . 0 2 5 )$ .

What the key ignores about cost. Over all 89 tasks, the six tied top-ten agents that log tokens difer 7.3× in median tokens per run (Fig. 3c) and, for the four priced at list rates, 5.5× in median cost per run (\$0.44 to \$2.47; every pair significant); the cheapest per run uses the priciest model per token. Where per-run cost is released, cost per accepted run separates pass/fail ties (4 of 8 adjacent mini-SWE-agent pairs on SWE-bench Verified and 4 of 5 priced Terminal-Bench 2.0 pairs in the top pass-rate tier difer at $p < 0 . 0 5 )$ , but under one scafold this mostly reflects price per token and verbosity, not ability.

## 6 Why Graded Credit, Not a Higher Bar, Fixes It

On ALE-Bench a run’s contest score can be compared with any bar, so we can test the usual remedy for a saturated key, a higher bar, without a new version.

Candidates. Let J be the quantity a task measures (runtime, contest score), $J _ { \mathrm { b a s e } }$ a baseline, $J _ { \mathrm { r e f } }$ the human or expert reference, and s the run’s position between them (0 at the baseline, 1 at the reference). Pass/fail bars put the target at the median human (the key), at the contest winner, or where it best separates the models of the time. Continuous scores from prior work: capped min $( s , 1 )$ ; linear (J − $J _ { \mathrm { b a s e } } ) / ( J _ { \mathrm { r e f } } - J _ { \mathrm { b a s e } } )$ , the human-normalized score of reinforcement learning [28]; log-ratio $\log ( J / J _ { \mathrm { b a s e } } ) / \log ( J _ { \mathrm { r e f } } / J _ { \mathrm { b a s e } } )$ , as in systems benchmarking [13]; the logratio clipped at 3; and the percentile among a pool of runs on the task [7,2]. All are floored at 0 and averaged over tasks.

A pass/fail bar sees one band of ability. We split the 141 ALE-Bench models with known or estimated release dates into 46 released by the end of 2025 and 95 released later (Table 2). The median-human bar ties the later top ten. The contest-winner bar separates them into two tiers, but scores 89% of the earlier models zero on every task (agreement with the human contest rating: 0.43). A bar tuned to separate the earlier models puts the later top ten in one tier. Each bar resolves one band of ability; raising it moves the band instead of widening it. This is what a harder bar on the same tasks does; a real new version also adds tasks, which we do not simulate. Nor does varying the bar by task: setting each task’s bar where half the earlier models pass still ties the later top ten. Several bars per task, a graded score in the spirit of polytomous item response models, do resolve them; a continuous score is the limit of that. Bounded scores with room above the reference rank both groups (clipped log-ratio, percentiles), while the linear score and the unclipped log-ratio tie the earlier top ten, which supports the clip. The clipped log-ratio agrees with humans about as well as the key (diference +0.02, 95% CI −0.002 to +0.05) and is more stable across task halves (+0.03, CI excludes 0). On the 32 later models with verifiable dates, the tiers are unchanged (3 vs 1; S = 0.41).

Table 2. ALE-Bench models released by the end of 2025 (earlier) and after (later); 63 of the later models have reconstructed dates. Tiers: among the top ten, with S in brackets; with 40 tasks, counts above two depend on calibration. Human: Kendall τ with the contest rating. Zero: models scoring zero on every task. Split-half: ranking agreement across random task halves.
<table><tr><td rowspan="2">Score</td><td colspan="3">Earlier models (46)</td><td colspan="3">Later models (95)</td><td rowspan="2">Split-half</td></tr><tr><td>Tiers (S)</td><td>Human</td><td>Zero</td><td>Tiers (S)</td><td>Human</td><td>Zero</td></tr><tr><td>Bar: median human (key)</td><td>1 (0.94)</td><td>0.89</td><td>9%</td><td>1 (1.00)</td><td>0.91</td><td>0%</td><td>0.81</td></tr><tr><td>Bar: contest winner</td><td>1 (1.00)</td><td>0.43</td><td>89%</td><td>2 (0.08)</td><td>0.61</td><td>73%</td><td>0.66</td></tr><tr><td>Bar: tuned on earlier</td><td>2 (0.41)</td><td>0.88</td><td>0%</td><td>1 (0.85)</td><td>0.81</td><td>0%</td><td>0.78</td></tr><tr><td>Per-task bars, calibrated</td><td>1 (0.95)</td><td>0.89</td><td>4%</td><td>1 (0.85)</td><td>0.75</td><td>0%</td><td></td></tr><tr><td>Graded (6 bars per task)</td><td>2 (0.94)</td><td>0.92</td><td>0%</td><td>3 (0.15)</td><td>0.95</td><td>0%</td><td></td></tr><tr><td>Capped</td><td>2 (0.75)</td><td>0.88</td><td>0%</td><td>1 (0.93)</td><td>0.86</td><td>0%</td><td>0.79</td></tr><tr><td>Linear</td><td>1 (0.83)</td><td>0.87</td><td>0%</td><td>1 (0.82)</td><td>0.87</td><td>0%</td><td>0.82</td></tr><tr><td>Log-ratio</td><td>1 (0.70)</td><td>0.90</td><td>0%</td><td>3 (0.005)</td><td>0.92</td><td>0%</td><td>0.83</td></tr><tr><td>Log-ratio, clip 3</td><td>2 (0.68)</td><td>0.90</td><td>0%</td><td>3 (0.03)</td><td>0.94</td><td>0%</td><td>0.83</td></tr><tr><td>Percentile, earlier pool</td><td>2 (0.87)</td><td>0.92</td><td>0%</td><td>2 (0.60)</td><td>0.88</td><td>0%</td><td>0.81</td></tr></table>

Choosing among continuous scores. Capped scores saturate with the bar. Percentiles depend on the pool: adding or removing models flips up to a third of pairs. Linear scores are dominated by a few tasks: one ALE-Bench problem’s contest score spans four orders of magnitude, and 79% of the linear score comes from the top 5% of tasks (log-ratio: 11%).

Why the clipped log-ratio. The log form is derived rather than chosen on ALE-Bench. A score of a positive quantity should not depend on its unit, so it is a function f of $x = J / J _ { \mathrm { b a s e } } ,$ , and credit should add under composition, so $f ( x y ) = f ( x ) + f ( y )$ . The only continuous solutions of this Cauchy equation are f(x) = k log x; averaging over tasks then summarizes ratio-scale results meaningfully [13,1], and dividing by $\log ( J _ { \mathrm { r e f } } / J _ { \mathrm { b a s e } } )$ sets k per task so that the reference scores 1. The clip at three reference gaps is a stated convention: it bounds any single task’s share of the score to $3 / n$ for n tasks. Without it, a task whose reference barely beats the baseline dominates: on SWE-ficiency, one with an expert speedup of 1.001 would hold a third of the score. Its value matters little: it binds on 4% of ALE-Bench runs, and from clip 2 to none, agreement with humans stays within 0.90–0.94 in both cohorts and the tiers do not change. A clip that binds often signals a weak reference, to be replaced rather than the clip tuned. We also ablated the form: unclipped log-fold scores, raw or scaled by each task’s anchor-set spread, agree less with humans (later cohort: at most 0.89 against 0.935) and are less stable; a quality–cost composite ranks cost (§7).

![](images/9bf3dba38ded266972c09dd0f9b3bb9590d32c496f73e6fa5843248cb53aa29c.jpg)  
Fig. 4. Left: an answer key merges acceptance, quality and the author’s solution into one per-task checker that emits one bit, with an LLM judge as the fallback. Right: the layered design. Quality is scored once per domain with room above the reference, and cost is metered separately.

We therefore propose the log-ratio, floored at the baseline and clipped at three reference gaps, averaged over tasks, for positive quantities such as runtime or contest score, and the linear score with the same floor and clip for quantities with no natural zero, such as Kaggle metrics. It gives at least the key’s top-ten tiers on SWE-ficiency and AlgoTune, one fewer on MLE-bench.

## 7 A Benchmark Design That Scores Acceptance, Quality and Cost Separately

We separate the roles that an answer key merges (Fig. 4); x is an agent’s final artifact and h its trajectory.

Intent states what the human wants: hard constraints and a direction of preference (“the bracket must carry the load; lighter is better”), never a procedure or a target shape, which would force per-task keys [8]; the bracket intent needs one physics check and one mass objective per domain.

Gate $G ( x ) \in \{ 0 , 1 \}$ is a deterministic, hardened acceptance check (behavioural, physics or safety, anti-tamper), where verifier hardening [40,37,36] belongs.

Objective $s ( x )$ normalizes a raw quantity $J ( x )$ (runtime, mass, cost, contest score) to dated anchors, mapping $J _ { \mathrm { b a s e } }$ to 0 and $J _ { \mathrm { r e f } }$ to 1:

$$
\begin{array} { r } { s ( x ) = G ( x ) \cdot \mathrm { m i n } \Big ( 3 , ~ \mathrm { m a x } \Big ( 0 , ~ \frac { \log ( J ( x ) / J _ { \mathrm { b a s e } } ) } { \log ( J _ { \mathrm { r e f } } / J _ { \mathrm { b a s e } } ) } \Big ) \Big ) } \end{array}
$$

for positive quantities, averaged over tasks (§6). The design requires a strong reference, near the best known solution, and excludes tasks whose reference barely beats the baseline by a margin fixed in advance, as SWE-ficiency does. A clip that binds often flags a weak reference, to be replaced, as on four re-run tasks. The objective is written once per domain; exposing it in the re-run took one or two lines per task.

Meter $c ( h )$ records dollars at a dated price table, tokens, steps and time, reported as cost-to-target [14] and never blended into s.

Anchors $J _ { \mathrm { b a s e } }$ and $J _ { \mathrm { r e f } }$ are frozen with a date, so two versions of a benchmark can share one scale. Runtimes drift with hardware, so the baseline and reference artifacts are re-measured on the machine of the runs and enter only as ratios, as the re-run did. Re-running three reference agents linked an old and a harder new half of ALE-Bench within 1–3 points of the accuracy of re-running every old model, which a moving bar cannot do.

Judges and reporting. LLMs may draft gate and objective code at authoring time, with human audit [32], but do not score at run time, so no judge can be persuaded; this is a design position [39], not a finding. Results are reported lexicographically: the gate rate with statistical tiers, then the mean of s (a run that fails the gate scores 0, so the objective never rewards an invalid run). Cost is a separate axis, reported jointly with quality and its Pareto front. Any claimed lead states the interval of exchange rates λ (folds of quality per fold of cost) over which it holds; holding for every λ is Pareto dominance. A composite would rank cost, not quality: on ALE-Bench its τ with the human rating is −0.35 in the later cohort, and 129 of 137 adjacent pairs flip as λ moves from 0.5 to 2.

## 8 Limitations and Conclusion

Limitations. Per-agent verifier error exists only for SWE-bench Verified, where we take SWE-ABS’s tests as ground truth and can bound false fails only by simulation. The sliding-window result rests on ALE-Bench, with partly reconstructed dates; AlgoTune agrees weakly, and MLE-bench, whose key is not saturated, shows no such efect. The re-run’s aggregate separation of tied models is borderline (three trials). Rubric-graded deliverables such as GDPval have neither a gate nor a scalar objective, and objectives can be gamed [21], so the gate must stay hardened.

Conclusion. Agent benchmarks often saturate in the score, not the tasks: the bit is too coarse to separate leading agents, and capped at its author. Benchmarks should build in an anchored objective with room above the reference and a separate cost meter, so a saturated key ends a version, not the measurement.

## References

1. Aczél, J., Roberts, F.S.: On the possible merging functions. Mathematical Social Sciences 17(3), 205–243 (1989)

2. Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A., Bellemare, M.G.: Deep reinforcement learning at the edge of the statistical precipice. In: Advances in Neural Information Processing Systems (NeurIPS) (2021)

3. Akhtar, M., Reuel, A., Soni, P., et al.: When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation. arXiv:2602.16763 (2026)

4. Aleithan, R., Xue, H., Mohajer, M.M., et al.: SWE-Bench+: Enhanced Coding Benchmark for LLMs. arXiv:2410.06992 (2024)

5. Berthold, T.: Measuring the Impact of Primal Heuristics. Operations Research Letters 41(6), 611–614 (2013)

6. Chan, J.S., Chowdhury, N., Jafe, O., et al.: MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering. In: ICLR 2025 (2025)

7. Dolan, E.D., Moré, J.J.: Benchmarking optimization software with performance profiles. Mathematical Programming 91(2), 201–213 (2002)

8. Dong, Z., Liu, Y., Ma, Z., Zhan, Q., Kong, D., Li, G., Li, K.: CADWorld: Computer-Use Benchmark for Long-Horizon Computer-Aided Design. arXiv:2609.16251 (2026)

9. Dong, Z., Liu, Z., Wang, Z., Li, Y., Ma, Z.: The Evaluation Challenge of Agency: Reliability, Contamination, and Evolution in LLM Agents. TechRxiv, doi:10.36227/techrxiv.177222530.04005985 (2026)

10. Dong, Z., Ma, Z., Wang, Z., Li, Y., Liu, Z., Deng, R., Zhan, Q., Qian, R.: How Benchmarks Mis-Score Computer-Use Agents. arXiv:2607.28367 (2026)

11. Erol, M.H., El, B., Suzgun, M., et al.: Cost-of-Pass: An Economic Framework for Evaluating Language Models. arXiv:2504.13359 (2025)

12. Esposito, M., Zhang, L.: The Benchmark Ceiling: Human Judgment, Evaluation Scarcity, and the Political Economy of AI Capability Measurement. arXiv:2607.01254 (2026)

13. Fleming, P.J., Wallace, J.J.: How not to lie with statistics: The correct way to summarize benchmark results. Communications of the ACM 29(3), 218–221 (1986)

14. Hansen, N., Auger, A., Ros, R., et al.: COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting. Optimization Methods and Software 36(1), 114–144 (2021)

15. Imajuku, Y., Horie, K., Iwata, Y., et al.: ALE-Bench: A Benchmark for Long Horizon Objective-Driven Algorithm Engineering. In: NeurIPS 2025 (Datasets and Benchmarks) (2025)

16. Jimenez, C.E., Yang, J., Wettig, A., et al.: SWE-bench: Can Language Models Resolve Real-World GitHub Issues? In: ICLR 2024 (2024)

17. Kapoor, S., Stroebl, B., Kirgis, P., et al.: Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation. arXiv:2510.11977 (2025)

18. Kapoor, S., Stroebl, B., Siegel, Z.S., et al.: AI Agents That Matter. arXiv:2407.01502 (2024)

19. Kolen, M.J., Brennan, R.L.: Test Equating, Scaling, and Linking: Methods and Practices. Springer, 3 edn. (2014)

20. Kwa, T., West, B., Becker, J., et al.: Measuring AI Ability to Complete Long Software Tasks. In: NeurIPS 2025 (2025)

21. Lange, R.T., Sun, Q., Prasad, A., et al.: Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization. arXiv:2509.14279 (2025)

22. Liu, F., Liu, Y., Sun, R., et al.: Coding Agents Have Converged: Why the SWEbench Leaderboard Can No Longer Order Its Top Entries, and What to Measure Instead. arXiv:2609.17394 (2026)

23. Ma, J.J., Hashemi, M., Yazdanbakhsh, A., et al.: SWE-ficiency: Can Language Models Optimize Real-World Repositories on Real Workloads? In: ICML 2026 (2026)

24. Madaan, L., Singh, A.K., Schaefer, R., et al.: Quantifying variance in evaluation benchmarks. arXiv preprint arXiv:2406.10229 (2024)

25. Marro, S., Yu, J., Malfa, E.L., et al.: Benchmarking at the Edge of Comprehension. arXiv:2602.14307 (2026)

26. Merrill, M.A., Shaw, A.G., Carlini, N., et al.: Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces. In: ICLR 2026 (2026)

27. Miller, E.: Adding error bars to evals: A statistical approach to language model evaluations. arXiv preprint arXiv:2411.00640 (2024)

28. Mnih, V., Kavukcuoglu, K., Silver, D., et al.: Human-Level Control Through Deep Reinforcement Learning. Nature 518(7540), 529–533 (2015)

29. Polo, F.M., Weber, L., Choshen, L., et al.: tinybenchmarks: Evaluating LLMs with fewer examples. In: Proceedings of the 41st International Conference on Machine Learning (ICML) (2024)

30. Press, O., Amos, B., Zhao, H., et al.: AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs? In: NeurIPS 2025 (Datasets and Bench marks) (2025)

31. Romano, J.P., Wolf, M.: Exact and approximate stepdown methods for multiple hypothesis testing. Journal of the American Statistical Association 100(469), 94–108 (2005)

32. Wei, J., Ma, Q., Zhao, Y., et al.: OpenComputer: Verifiable Software Worlds for Computer-Use Agents. arXiv:2605.19769 (2026)

33. Westfall, P.H., Young, S.S.: Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment. Wiley, New York (1993)

34. Xie, T., Zhang, D., Chen, J., et al.: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. In: NeurIPS 2024 (Datasets and Benchmarks) (2024)

35. Yang, C., Wang, R., Jiang, J., et al.: Nondeterministic Polynomial-time Problem Challenge: An Ever-Scaling Reasoning Benchmark for LLMs. Transactions on Machine Learning Research (TMLR) (2025)

36. Yu, B., Cao, Y., Zhang, Y., et al.: SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark. arXiv:2603.00520 (2026)

37. Yu, B., Zhu, Y., He, P., et al.: UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench. In: ACL 2025 (2025)

38. Zhan, Q., Jang, S., Dong, Z., Hu, M., Chen, Z., Qu, T.: When Verifiable Counts Depend on Wording: Auditing Wording Robustness in Instruction Following. arXiv:2610.05278 (2026)

39. Zhao, Y., Liu, H., Yu, D., et al.: One Token to Fool LLM-as-a-Judge. arXiv:2507.08794 (2025)

40. Zhu, Y., Jin, T., Pruksachatkun, Y., et al.: Establishing Best Practices for Building Rigorous Agentic Benchmarks. arXiv:2507.02825 (2025)