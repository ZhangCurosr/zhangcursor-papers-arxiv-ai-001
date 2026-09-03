# frb100-40 After Two Decades: An Optimality Certificate and a Preregistered Search Study

Onur Ugurlu ˘

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Abstract—For more than 20 years, the Model-RB benchmark frb100-40 remained an open challenge; since 2014, its public record had stood at 99 of 100 variables. We give a directly checkable 100-vertex independent set for its 4,000-vertex graph. Together with a verified partition into 100 cliques of size 40, the witness proves that the maximum independent-set size is 100 and the minimum vertex-cover size is 3,900. The stochastic run that found the witness is kept separate from this proof. We evaluated its added pair and triple repair operators in a preregistered campaign comprising 8,668 valid runs. The primary comparison found no detectable acceleration over base ULSA (hazard ratio 0.967, 95% confidence interval 0.915–1.023; p = 0.248), and the factorial ablation reached the same conclusion. On a smaller FRB suite, the group-aware CSP pipeline solved 2,500/2,500 runs, compared with 2,391/2,500 for LibMVC-NuMVC. On frb100-40, full ULSA, base ULSA, and NuMVC each produced 0/56 new certificates. With no events, the planned cross-solver hazard ratios remain unidentified. NuMVC ended with cover size 3,902 in 40 runs and 3,903 in 16. Exhaustive enumeration showed that none of the 108 unique recorded conflict-two states had a strictly improving group-aware CSP neighbor within Hamming radius three. The certificate settles the instance. The experiments characterize the search barrier, and the preregistered comparisons show no heuristic advantage.

Index Terms—constraint satisfaction, maximum independent set, minimum vertex cover, stochastic local search, reproducible computation

## I. INTRODUCTION

H <sup>ARD</sup> <sup>satisfiable</sup> <sup>instances</sup> <sup>provide</sup> <sup>an</sup> <sup>unusually</sup> <sup>di-</sup> rect meeting point between computational-intelligence methods and complexity-motivated benchmark design. Model RB was introduced to generate random constraint-satisfaction problems (CSPs) with growing domains and an analytically located satisfiability transition [1]. Forced-satisfiable instances near that transition can remain difficult for both complete and incomplete search [2]. Their graph encodings, distributed through BHOSLIB [3], have consequently become standard tests for maximum clique, maximum independent set (MIS), minimum vertex cover (MinVC), and coloring algorithms.

Among these instances, frb100-40 is the largest commonly discussed case, representing 100 variables with domain size 40. Rosin’s unweighted stochastic local-search algorithm (ULSA) reached 99 satisfied variables in 2014 [4]. The benchmark history still listed 99 as the latest public record when we checked it on 2 September 2026 [3], and a 2025 paper described the instance as an unsolved 20-year challenge [5]. Accordingly, our deliberately qualified priority statement is that this is the first publicly checkable size-100 assignment known to us.

Once found, a complete assignment is easy to check. It selects one vertex from each of 100 groups and must contain no graph edge. Since each group is a clique, the same file also supplies a matching upper bound. The central result of this paper is consequently a short certificate for a fixed graph hash: a 100-vertex independent set, an independently verified 100-clique partition, and the resulting equalities $\alpha ( G ) = 1 0 0$ and $\tau ( G ) = 3 9 0 0 .$

Finding the witness and explaining the search are different questions. The successful run used periodic pair and triple repairs. One rare trajectory provides insufficient evidence for their utility. We therefore replayed the run exactly and tested the operators under a frozen protocol. The resulting 8,668- run campaign includes the smaller-instance comparison, a factorial ablation, and three matched 56-run validation arms on frb100-40. The preregistered comparisons found no speed advantage for the repairs.

The unsuccessful runs still leave useful structural evidence. Complete enumeration around 108 unique conflict-two assignments shows that none has a strictly improving grouped-CSP neighbor within Hamming radius three. The resulting proposition applies to the archived states. Other states and local-search methods lie outside its scope. It points to a specific barrier: escaping these endpoints requires a neutral or worsening transition, a change involving at least four variables, or a move outside the audited neighborhood.

## II. BACKGROUND AND RELATED WORK

## A. Model RB and the FRB Graph Encoding

A binary Model-RB instance has n variables, a domain of size $d = n ^ { \alpha }$ , and randomly selected forbidden value pairs [1], [2]. The standard graph encoding creates one vertex $( i , a )$ for each variable i and value a. All d values of a variable form a clique; a forbidden tuple between variables i and j adds a cross-group edge. A CSP solution is therefore an independent set containing exactly one vertex from every group.

For frb100-40, $n = 1 0 0$ and $d = 4 0$ , hence the graph contains 4,000 vertices. The supplied DIMACS instance has 572,774 edges. Our parser independently classified 78,000 distinct intra-group edges, exactly $\stackrel { \mathrm { ~ \tiny ~ 1 0 0 ~ } } { \mathrm { 1 0 0 } } { } _ { \stackrel { \mathrm { ~ \tiny ~ 2 ~ } } { } } ^ { 4 0 } \mathrm { ) }$ , and 494,774 crossgroup edges. The intra-group cliques give the immediate upper bound $\alpha ( G ) \leq 1 0 0 .$

## B. Stochastic Local Search for CSP and Vertex Cover

Stochastic local search balances improving steps with mechanisms that traverse plateaus or escape local minima [6]. Plateau traversal is also central to maximum-clique search, where dynamic penalties and swaps can sustain exploration after direct improvement stalls [7]. ULSA is an unweighted, group-aware CSP method designed for random binary instances. It samples a violated constraint and updates one endpoint using vectorized value scores, recency information, and randomized tie breaking [4]. Its representation always maintains one selected value per CSP variable.

General-graph MinVC methods use a different state space. Edge-weighted local search and configuration checking preceded NuMVC, which combined a two-stage exchange with edge weighting and forgetting [8]–[10]. FastVC later reduced per-step complexity for massive graphs [11], while MetaVC automated the configuration of a broader collection of MinVC components [12].

The neighboring MIS literature includes efficient two- and three-vertex exchanges [13], evolutionary search coupled with reductions [14], and exact branch-and-reduce algorithms for vertex cover and independent set [15], [16]. Systematic exploration of larger MinVC neighborhoods has shown that exhaustive k-swap search can remain practical well beyond the smallest radii [17]. This context motivates our repair operators and the radius-three enumeration. It also cautions against attributing a pipeline difference to representation alone because ULSA and NuMVC change both solver and encodingaware state space.

## III. THE CERTIFICATE AND THE SEARCH

## A. A Checkable Proof of Optimality

Let $V _ { 1 } , \ldots , V _ { 1 0 0 }$ be the consecutive 40-vertex groups in $\pm \Upsilon \Upsilon 1 0 0 - 4 0$ . Direct enumeration verifies that $G [ V _ { i } ] \cong K _ { 4 0 }$ for every i and that these groups partition V . The argument follows the certifying-algorithm principle: checking the instance, output, and witness is independent of trusting the program that produced them [18].

Proposition 1. The hashed frb100-40 graph has $\alpha ( G ) =$ 100 and $\tau ( G ) = 3 9 0 0$

Proof. Every independent set intersects each clique $V _ { i }$ in at most one vertex, so $\alpha ( G ) ~ \leq ~ 1 0 0$ . The certificate in Appendix A contains 100 distinct in-range vertices, one from every $V _ { i } ,$ and an exhaustive scan of all 572,774 edges finds zero edges with both endpoints selected. Hence $\alpha ( G ) \geq 1 0 0$ and therefore $\alpha ( G ) = 1 0 0$ . For any graph, the complement of an independent set is a vertex cover and $\alpha ( G ) + \tau ( G ) = | V | ;$ thus $\tau ( G ) = 4 0 0 0 - 1 0 0 = 3 9 0 0$ □

The graph SHA-256 is

$$
\begin{array} { r l } & { 1 \ \mathrm { f } \ 9 4 9 \ 6 8 \mathrm { e } 4 5 0 6 \mathrm { f } \mathrm { a } 9 1 6 7 4 \mathrm { e } 5 7 2 8 \mathrm { f } \mathrm { f } 2 8 0 \mathrm { a } 5 4 } \\ & { \qquad \mathrm { a } 0 0 6 \mathrm { e } 4 4 3 6 3 3 0 2 \mathrm { c } 2 \mathrm { c } 8 3 \mathrm { b } 1 8 8 \mathrm { c } \mathrm { b } 2 \mathrm { c } 8 7 0 0 6 6 . } \end{array}
$$

All remaining certificate, log, replay, source, and build digests are recorded in the accompanying artifact manifest.

$$
B . \ U L S A +
$$

ULSA+ leaves ULSA’s core update unchanged and adds two periodic strict-improvement operators.

![](images/ced17be4f4b8b2f65e6864c387c93a0d8bda543194233bc0d9ebfeeec3a59465.jpg)  
Fig. 1. Best violation count in the successful discovery process. The horizontal axis is symmetric-log scaled to retain the early transient and the long conflicttwo tail.

• Every $2 ^ { 9 } = 5 1 2$ core updates, a currently violated constraint is sampled. For its two endpoint variables, all $d ^ { 2 }$ joint value assignments are evaluated against the current assignment. The best assignment is applied only if it strictly reduces the total number of violations.

• Every $2 ^ { 1 8 } = 2 6 2 , 1 4 4$ updates, two variables are obtained from a sampled violation and a third is sampled from violation endpoints (with a bounded fallback). All $d ^ { 3 }$ joint assignments are evaluated with safe lower-bound pruning and, again, only a strict improvement is accepted.

The pair and triple variants toggle these operators independently; the full variant enables both. The instrumented executable uses the same observation path in every factorial arm. Before the campaign, the full instrumented build was checked bit-for-bit against the production ULSA+ build, and the alldisabled build against Rosin’s ULSA for fixed seeds. Sampled logging preserved the solver’s random-number consumption. The measured instrumentation overhead in the cluster self-test was approximately 0.12%.

## C. Discovery Run and Deterministic Replay

The discovery cohort comprised 56 independent ULSA+ processes. Task 33, seed 164415777, reached zero violations after 695,602,633,182 core updates, as recorded on the bestconflicts 0 iteration line, and produced the certificate in Appendix A. Its recorded interval was 26 July 2026 06:19:37 UTC to 28 July 2026 03:45:02 UTC (45.42 h). The other 55 discovery processes ended their budgets without a certificate.

The replay used the production source, the same seed, and the full budget. Its solver output was byte-identical to the frozen reference archived with the artifact. The replay demonstrates deterministic same-seed reproducibility. Independent stochastic replication would require new seeds. Fig. 1 shows the discovery best-so-far trace.

TABLE I  
PREREGISTERED CAMPAIGN STRUCTURE AND CURRENT STATUS
<table><tr><td>Arm</td><td>Solver/variant</td><td>Purpose</td><td>Budget/run</td><td>Rows</td></tr><tr><td>Discovery</td><td>ULSA+ production</td><td>certificate search</td><td>71.75 h</td><td>56</td></tr><tr><td>C</td><td>instrumented full</td><td>independent validation</td><td>71.75 h</td><td>56</td></tr><tr><td>B1</td><td>instrumented base</td><td>independent validation</td><td>71.75 h</td><td>56</td></tr><tr><td>B2</td><td>LibMVC-NuMVC</td><td>independent validation</td><td>71.75 h</td><td>56</td></tr><tr><td>A1</td><td>full/base/NuMVC</td><td>suite comparison</td><td>600 s</td><td>7,500</td></tr><tr><td>A2</td><td>repair variants</td><td>factorial/diagnostic</td><td>600 s</td><td>1,000</td></tr></table>

## IV. PREREGISTERED EXPERIMENTAL DESIGN

## A. Cohorts, Instances, and Budgets

The manifest and analysis plan were frozen before the validation results were inspected, and the executable materials were retained to support reproducibility [19]. The discovery cohort predates this protocol and is treated as exploratory throughout. Table I separates discovery, primary rare-event validation, suite evaluation, and ablation. C, B1, and B2 share 56 seed identifiers as matched initialization blocks. Their subsequent random streams diverge because the repair operators and the NuMVC implementation consume random numbers differently.

A1 contains five instances from each family frb30-15 through frb59-26. The first four families use 100 matched seeds per instance and the last four use 25. The development instances frb56-25-1 and frb59-26-1 were excluded from the primary A1 comparison. A2 uses the remaining four instances in those two families with 25 seeds per factorial cell. Cadence-sensitivity arms are secondary and were excluded from the primary factorial model.

Jobs ran as one-core processes on TRUBA’s Orfoz partition. Orfoz nodes contain two Intel Xeon Platinum 8480+ processors (112 cores total) and 256 GB RAM [20]; the dispatcher allocated 56 one-core tasks per node. ULSA builds used GCC with -O3 -mbmi -mavx2 -m64 -funroll-loops; the LibMVC wrapper used C++14, -O3, and -march=native. Source, binary, manifest, preregistration, instance, and dispatcher hashes prevented mixing results across builds. Scientific time used CLOCK\_MONOTONIC; NuMVC timing began before graph loading. A success flag was created only after a C verifier confirmed both feasibility and the target independent-set size.

## B. Preregistered Outcomes and Analysis

The preregistered primary hypothesis H1 was that full ULSA+ has a shorter time-to-target distribution than base ULSA on A1 validation instances. The two-sided test used an instance-stratified Cox model [21] with a cluster-robust sandwich covariance for each instance–seed block. The principal effect size was restricted mean survival time (RMST) at the family cutoff; because every A1 CSP run succeeded, RMST equals the observed mean time.

The $_ { \mathbf { A } 2 \mathbf { \Lambda } 2 \mathbf { \Lambda } \times \mathbf { \Lambda } 2 }$ model included pair, triple, and pair-bytriple terms, again using instance-stratified Cox regression with seed-block clustering. Success was to be modeled by clustered logistic regression. All 800 validation observations succeeded, leaving that model unidentifiable. H1, the A2 pair main effect, and the A2 pair-by-triple interaction form a three-test Holm family [22]. Success-rate intervals are exact Clopper–Pearson intervals [23]. RMST uncertainty uses a fixed-seed, 10,000- repetition two-stage bootstrap: instances are resampled first and matched seed blocks second.

TABLE II  
PREREGISTERED COX MODELS
<table><tr><td>Contrast</td><td>Hazard ratio (95% CI)</td><td>Raw p Holm p</td></tr><tr><td>H1: full vs base</td><td>0.967 (0.915–1.023) 0.248 0.728</td><td>0.743</td></tr><tr><td>A2: pair</td><td>0.966 (0.796–1.173)</td><td>1.000</td></tr><tr><td>A2: triple</td><td>0.993 (0.823-1.198)</td><td></td></tr><tr><td>A2: pair×triple</td><td>0.975 (0.762–1.249)</td><td>1.000</td></tr></table>

A1’s comparison with LibMVC-NuMVC [24] is descriptive. Its unit of comparison is the complete operational pipeline, encompassing representation, algorithm, and implementation. NuMVC medians are computed among successful runs and are displayed together with success rates.

## V. RESULTS

## A. Certificate and Corpus Verification

An independent parser confirmed 4,000 vertices, 572,774 edge records, 100 complete 40-cliques, 100 unique certificate vertices, and zero selected conflict edges. The archived discovery log and replay output were also checked against the artifact manifest; the replay was byte-identical to its frozen reference.

Across the frozen validation corpus, 8,668 runs were completed: 7,500 A1, 1,000 A2, and 56 in each of C, B1, and B2. All 8,391 emitted success flags were rechecked by the independent C verifier; no unverified success is included. The B2 archive contains exactly the preregistered 56 seeds, and its run records agree byte for byte across the two retained campaign archives. Every B2 process completed normally after its 258,300-s scientific budget; no outer timeout fired.

The successful discovery seed came from an earlier exploratory cohort, so the validation event count excludes it. Its same-seed replay establishes deterministic reproducibility; the stochastic evidence comes from the independently seeded runs.

## B. Primary and Factorial Tests

Table II summarizes the preregistered Cox-model results. Under the frozen analysis, H1 received no support: the hazard ratio for full ULSA+ relative to base ULSA is 0.967 (95% CI 0.915–1.023; raw $p = 0 . 2 4 8$ , Holm-adjusted $p = 0 . 7 4 3 )$ With success as the event, a ratio above one would favor faster full-variant completion. The RMST estimates are 2.540 s for full and 2.283 s for base, a difference of 0.257 s (two-stage bootstrap 95% CI −0.367 to 1.134 s) and a ratio of 1.112 (95% CI 0.836–1.460).

Repair effects were likewise unsupported in A2: the pair, triple, and interaction estimates are all close to one with wide intervals. Mean completion times for base, pair, triple, and full were 19.291, 18.219, 16.277, and 16.839 s, respectively. The preregistered stratified model gives no evidence that these marginal differences represent repair effects. All 1,000 A2 runs, including the 200 cadence-sensitivity runs, attributed the final successful transition to ULSA’s core update; no pair or triple repair supplied that step.

TABLE III  
A1 PIPELINE RESULTS BY FAMILY
<table><tr><td colspan="4">CSP / full ULSA+ Graph / LibMVC-NuMVC</td></tr><tr><td>Family</td><td>Success</td><td>Med. s Success</td><td>Med. s</td></tr><tr><td>frb30-15</td><td>500/500</td><td>0.008 466/500</td><td>0.033</td></tr><tr><td>frb35-17</td><td>500/500</td><td>0.018 468/500</td><td>0.164</td></tr><tr><td>frb40-19</td><td>500/500</td><td>0.060 494/500</td><td>1.075</td></tr><tr><td>frb45-21</td><td>500/500</td><td>0.186 500/500</td><td>3.215</td></tr><tr><td>frb50-23</td><td>125/125</td><td>0.551 125/125</td><td>25.435</td></tr><tr><td>frb53-24</td><td>125/125</td><td>1.161 92/125</td><td>67.213</td></tr><tr><td>frb56-25</td><td>125/125</td><td>3.148122/125</td><td>52.454</td></tr><tr><td>frb59-26</td><td>125/125</td><td>11.085 124/125</td><td>367.312</td></tr></table>

## C. Pipeline Performance on the Smaller FRB Suite

On the A1 suite, full ULSA+ produced 2,500/2,500 verified solutions (100%; 95% CI 99.85–100%), while the LibMVC implementation of NuMVC produced 2,391/2,500 (95.64%; 95% CI 94.76–96.41%). Table III and Fig. 2 break this down by family. The medians use target time among verified successes and exclude censored NuMVC runs. The descriptive family totals include the two development instances excluded from H1. On frb59-26, for example, the full group-aware pipeline solved 125/125 runs with a successful-run median of 11.085 s; LibMVC-NuMVC solved 124/125 with a successfulrun median target time of 367.312 s.

For a practitioner who begins with the original CSP groups, this difference matters: the group-aware pipeline was more reliable and reached its targets sooner on this suite. The comparison covers complete operational pipelines. The generalgraph solver searches a broader cover space and also differs in its moves, weighting, initialization, and stopping rule. Isolating representation would require the same search logic under both encodings or a factorial solver-by-representation design.

## D. Validation Outcomes on frb100-40

No validation arm produced a new certificate: C, B1, and B2 each returned 0/56. The exact two-sided 95% interval is 0– 6.38% for every arm. The absence of events leaves the planned Cox contrasts and hazard ratios unidentified and provides no basis for equivalence or a ranking of the three solvers on frb100-40.

Those event-free runs still provide useful endpoint information. All 112 C and B1 runs stopped with two violated CSP constraints. NuMVC reported a best cover size of 3,902 in 40 runs and 3,903 in 16, corresponding to objective gaps of +2 and +3 from the certified optimum. It never reported the target, either within or after the scientific budget.

To compare terminal objective quality in Fig. 3, we performed a post-hoc conversion of the C and B1 assignments. Each selected one vertex per group and induced exactly two conflict edges. Removing a minimum vertex cover of those two edges yields a valid independent set of size 99 for five records and 98 for the remaining 107; an all-edge rescan verified every derived set. The C and B1 +1/+2 values are reconstructed from archived assignments. The NuMVC +2/+3 values come directly from its native cover objective. Their visual agreement at +2 is suggestive. Its post-hoc origin and the simultaneous solver changes preclude a confirmatory cross-solver interpretation.

TABLE IV  
EXPLORATORY PAIR-REPAIR ACCEPTANCE BY CONFLICT LEVEL
<table><tr><td>Family</td><td>Conflicts</td><td>Accepted/attempted</td><td>Rate</td></tr><tr><td>frb56-25</td><td>1</td><td>0/8,039</td><td>0%</td></tr><tr><td rowspan="2">frb59-26</td><td>2</td><td>9,989/209,415</td><td>4.770%</td></tr><tr><td>1</td><td>0/33,188</td><td>0%</td></tr><tr><td rowspan="2">frb100-40</td><td>2</td><td>30,992/254,928</td><td>12.157%</td></tr><tr><td>2</td><td>0/9,426,240</td><td>0%</td></tr><tr><td></td><td></td><td>34,762,255/52,693,991</td><td>9.038%</td></tr></table>

## E. Radius-Three Structure of the Recorded Endpoints

Every C and B1 run recorded a terminal assignment with two violated cross-group constraints. The 112 records contained 108 unique assignments. For an assignment $x \in$ $\{ 1 , \ldots , 4 0 \} ^ { 1 0 0 }$ , let $f ( x )$ be its number of violated constraints and $d _ { H }$ the number of variables assigned different values.

Proposition 2 (computer-assisted, recorded-state scope). For every unique recorded conflict-two assignment x and every grouped-CSP assignment y with $1 \le d _ { H } ( x , y ) \le 3 , f ( y ) \ge 2 .$

The audit first recomputed f(x) = 2 from the graph. Complete pair evaluation covered 887,040,000 joint value assignments over the 112 run records. For triples, any variable triple disjoint from all endpoints of the two current violated edges cannot remove either violation and was eliminated safely. The remaining 2,085,040 variable triples required 133,442,560,000 joint assignments; none improved on two violations. A second implementation independently reproduced both counts and the zero-improvement result. Delta accounting agreed with full conflict recounting on 10,000 deterministic spot checks.

Proposition 2 applies only to the audited conflict-two states and to strict improvements within radius three. Equal-cost, worsening, and larger-radius moves remain possible; indeed, the successful discovery trajectory ultimately escaped this objective level.

The operator logs provide a complementary, exploratory view. Table IV uses pair attempts consistently across conflict levels. In each of three families, acceptance was zero at the lowest conflict level reached in these trajectories. Positive acceptance remained one level above. The estimand is conditional acceptance along the observed trajectories; uniform state-space density would require a different sampling design.

The 112 conflict-two records had mean Hamming distance 95.991 (range 89–100) from the particular optimum certificate in Appendix A. Two independent uniform assignments over 40 values differ in expectation on 97.5 of 100 variables. Thus, the terminal records are far from this particular witness despite having objective value two. Their distance to the nearest optimum remains unknown because other optima may exist.

![](images/a9be515d8bc7df07cc0776b03cbb6fac3dbfa882871f623ff24941fa22d94594.jpg)

![](images/9e09911364beb9ecf90dc8a2da6afbb6071cbc9ad2271de902f5d86bfa3bceef.jpg)  
Fig. 2. A1 pipeline comparison by FRB family. (a) Verified success proportions with exact 95% binomial intervals. (b) Median target time among verified successes on a logarithmic scale; the medians exclude censored NuMVC runs. Solver, representation, and implementation vary together; the panels therefore summarize complete-pipeline performance.

![](images/09e8cb545921115eb957bddd0c4918c4cad6421ac473b511bc5b10f483ddc02b.jpg)  
Fig. 3. Endpoint objective gaps in the three preregistered validation arms (n = 56 per arm). No arm produced a verified certificate (0/56 in each; exact two-sided 95% CI 0–6.38%), so the planned hazard ratios remain unidentified. C and B1 gaps are reconstructed post hoc from recorded assignments and independently verified; B2 gaps are NuMVC’s reported cover values minus 3,900.

## VI. DISCUSSION

The certificate is unconditional for the hashed input: anyone can verify its 100 vertices, zero internal edges, and the 100- clique cover. The proof depends only on the witness and clique cover. This separation protects the central result from the low empirical discovery rate.

The preregistered results provide no evidence that pair and triple repairs explain the discovery. They yielded no detectable time-to-target gain over base ULSA, and every final transition in the A2 successes came from the core update. The evidence supports a narrower interpretation: the certificate arose from a rare trajectory generated by a reproducible search system.

![](images/de20bc7454fdb728e75eac926b1eb88b4f08a27ad1a84d1556905fef6b36c76a.jpg)  
Fig. 4. Pair-repair acceptance at each recorded conflict level. Exact counts appear in Table IV. Only frb100-40 received an exhaustive radius-three state audit.

The recorded endpoints say more than the aggregate failure count. At each conflict-two endpoint, a strictly improving move must either change at least four CSP variables at once or pass first through an equal- or higher-conflict assignment. That statement is exact for the archived states. Neutral walks, strategic worsening, large-neighborhood search, and nonlocal proposals are therefore natural mechanisms to test next.

On the smaller suite, the ULSA pipeline performed better. The source of that advantage remains unidentified. The A1 comparison changes the solver, state space, and implementation together, leaving the contributions of the CSP representation, ULSA update, vectorization, initialization, and their interactions unresolved. The result informs pipeline selection; a causal encoding claim would require a controlled design. MetaVC appeared in a prospective phase-2 commitment and remained outside the frozen A1 study. Its addition would require a separately labeled experiment.

Several limitations bound these interpretations. The successful seed came from an exploratory 56-run cohort, making its discovery rate unsuitable as a preregistered estimator; the same-seed replay contributes only deterministic evidence. Each validation arm observed 0/56 successes. This allcensored outcome leaves the planned cross-arm hazard ratios unidentified and provides no basis for solver equivalence.

The radius-three result has a similarly finite scope. The audit uses terminal assignments from 56 matched seeds in each of two arms. After accounting for duplicate states and the matched-seed design, the evidence consists of 108 unique archived states with dependence across records. The proposition applies to this finite set of states and the audited neighborhood.

The landscape analysis was designed after the main campaign. Its radius-three enumeration, endpoint-gap reconstruction, and operator-level cross-family summary were independently checked. These analyses remain exploratory. The concentration of both pipelines near objective gap +2 is descriptive and is confounded by different state spaces and endpoint definitions. Exhaustive neighborhood enumeration was performed only for the recorded frb100-40 CSP states; the smaller-family pattern comes from logged operator attempts.

Execution history introduces one further qualification. C and B1 completed in different calendar blocks; the preregistration had scheduled them for the same week. A required v2.4 dispatcher/preregistration amendment was logged after the campaign began, although solver source and binaries were unchanged. The artifact retains the amended preregistration, final dispatcher, and recorded source and binary hashes. Node and calendar effects remain possible for C versus B1. The stronger A1 and A2 conclusions rely on interleaved matched blocks and instance-stratified analyses.

Finally, reproducibility depends on preserving the complete computational record. The working package contains the certificate, compact results, scripts, and audit reports. A persistent archive must also retain the raw logs, exact source, build environment, manifests, verifier, and hashes. The priority search should be refreshed immediately before submission.

## VII. CONCLUSION

The 100 vertices reported here settle the hashed frb100-40 instance. They contain no internal edge, and the graph’s partition into 100 cliques supplies the matching upper bound. Thus $\alpha ( G ) = 1 0 0$ and τ (G) = 3900 independently of how the witness was found. The successful trajectory is also reproducible: its same-seed replay gives identical solver output.

The search study produced a mixed result. Across 8,668 completed preregistered runs, pair and triple strictimprovement repairs gave no detectable improvement over base ULSA. The group-aware CSP pipeline performed better than LibMVC-NuMVC on the smaller FRB suite, with solver, representation, and implementation varying together.

On frb100-40 itself, all three 56-run validation arms were event-free, so cross-solver hazard ratios and rankings are unavailable. Complete enumeration establishes a finite structural result: 108 archived conflict-two states are closed to every strictly improving grouped-CSP move within radius three. Methods designed for this benchmark should therefore look beyond the audited neighborhood or accept neutral and worsening steps deliberately.

## DATA AND CODE AVAILABILITY

The complete reproducibility artifact contains the hashed graph identifier, the 100-vertex certificate, independent verifiers, discovery and replay logs, the frozen manifest, amended preregistration, final dispatcher, canonical results, analysis scripts, raw campaign logs, and build receipts. It has been deposited at Zenodo under the reserved DOI https://doi.org/ 10.5281/zenodo.22257064; the record will be activated upon acceptance.

## ACKNOWLEDGMENT

The large-scale computational campaign reported in this article was carried out on TRUBA, the national highperformance computing infrastructure operated by TUB<sup>¨</sup> <sup>˙</sup>ITAK ULAKB<sup>˙</sup>IM. The authors thank TRUBA for providing the computational resources. In accordance with the IEEE policy on AI-generated content, the authors disclose that generative AI systems (Anthropic Claude and OpenAI GPT) were used, under author direction, to assist in drafting and revising the manuscript text and in developing and cross-checking the analysis and verification code. All experiments were executed by the authors, and every scientific claim, number, and certificate in this article was verified and approved by the authors, who take full responsibility for its content.

## APPENDIX A INDEPENDENT-SET CERTIFICATE

The following one-based DIMACS vertices form the veri fied independent set:

<table><tr><td rowspan=1 colspan=7>255996142164240248311341392</td></tr><tr><td rowspan=1 colspan=2>613657685757 794</td><td rowspan=1 colspan=1>830847903943992</td><td rowspan=1 colspan=1>1015</td><td rowspan=1 colspan=1>1067</td><td rowspan=1 colspan=1>1105</td><td rowspan=1 colspan=1>1151</td></tr><tr><td rowspan=1 colspan=1>11744 1227 1264</td><td rowspan=1 colspan=1>1314</td><td rowspan=1 colspan=1>13571381 14322 1470</td><td rowspan=1 colspan=1>1499</td><td rowspan=1 colspan=1>1557</td><td rowspan=1 colspan=1>1586</td><td rowspan=1 colspan=1>1618</td></tr><tr><td rowspan=1 colspan=3>16633 1710 1729178618191868 190991952</td><td rowspan=1 colspan=3>196820382076</td><td rowspan=1 colspan=1>2104</td></tr><tr><td rowspan=1 colspan=2>21302174 22342247</td><td rowspan=1 colspan=1>22932333 2371 2408</td><td rowspan=1 colspan=3>244624962536</td><td rowspan=1 colspan=1>2566</td></tr><tr><td rowspan=1 colspan=2>2612266826962733</td><td rowspan=1 colspan=1>2788283228632907</td><td rowspan=1 colspan=2>29562963</td><td rowspan=1 colspan=1>3036</td><td rowspan=1 colspan=1>3080</td></tr><tr><td rowspan=1 colspan=2>3112312131853227</td><td rowspan=1 colspan=1>32803311333333822</td><td rowspan=1 colspan=1>3429</td><td rowspan=1 colspan=1>3463</td><td rowspan=1 colspan=1>3485</td><td rowspan=1 colspan=1>3545</td></tr><tr><td rowspan=1 colspan=2>3563360336803688</td><td rowspan=1 colspan=1>3733376438183873</td><td rowspan=1 colspan=2>38883958</td><td rowspan=1 colspan=1>3987.</td><td rowspan=1 colspan=1></td></tr></table>

## REFERENCES

[1] K. Xu and W. Li, “Exact phase transitions in random constraint satisfaction problems,” Journal of Artificial Intelligence Research, vol. 12, pp. 93–103, 2000. [Online]. Available: https://arxiv.org/abs/cs/ 0004005

[2] K. Xu, F. Boussemart, F. Hemery, and C. Lecoutre, “A simple model to generate hard satisfiable instances,” in Proceedings of the 19th International Joint Conference on Artificial Intelligence, 2005, pp. 337–342. [Online]. Available: https://www.ijcai.org/Proceedings/05/ Papers/0989.pdf

[3] K. Xu, “BHOSLIB: Benchmarks with hidden optimum solutions for graph problems,” Benchmark repository, benchmark history, accessed: 2026-09-02. [Online]. Available: https://rb-bench.github.io/benchmarks/ hidden-opt-graph.html

[4] C. D. Rosin, “Unweighted stochastic local search can be effective for random CSP benchmarks,” arXiv preprint arXiv:1411.7480, 2014.

[5] K. Xu and G. Zhou, “SAT requires exhaustive search,” Frontiers of Computer Science, vol. 19, no. 12, p. 1912405, 2025.

[6] H. H. Hoos and T. Stutzle,¨ Stochastic Local Search: Foundations and Applications. San Francisco, CA, USA: Morgan Kaufmann, 2004.

[7] W. J. Pullan and H. H. Hoos, “Dynamic local search for the maximum clique problem,” Journal ofArtificial Intelligence Research, vol. 25, pp. 159–185, 2006.

[8] S. Cai, K. Su, and Q. Chen, “EWLS: A new local search for minimum vertex cover,” in Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence, 2010, pp. 45–50.

[9] S. Cai, K. Su, and A. Sattar, “Local search with edge weighting and configuration checking heuristics for minimum vertex cover,” Artificial Intelligence, vol. 175, no. 9–10, pp. 1672–1696, 2011.

[10] S. Cai, K. Su, C. Luo, and A. Sattar, “NuMVC: An efficient local search algorithm for minimum vertex cover,” Journal of Artificial Intelligence Research, vol. 46, pp. 687–716, 2013.

[11] S. Cai, “Balance between complexity and quality: Local search for minimum vertex cover in massive graphs,” in Proceedings of the 24th International Joint Conference on Artificial Intelligence, 2015, pp. 747–753. [Online]. Available: https://www.ijcai.org/Proceedings/15/ Papers/111.pdf

[12] C. Luo, H. H. Hoos, S. Cai, Q. Lin, H. Zhang, and D. Zhang, “Local search with efficient automatic configuration for minimum vertex cover,” in Proceedings of the 28th International Joint Conference on Artificial Intelligence, 2019, pp. 1297–1304.

[13] D. V. Andrade, M. G. C. Resende, and R. F. Werneck, “Fast local search for the maximum independent set problem,” Journal of Heuristics, vol. 18, no. 4, pp. 525–547, 2012.

[14] S. Lamm, P. Sanders, C. Schulz, D. Strash, and R. F. Werneck, “Finding near-optimal independent sets at scale,” Journal of Heuristics, vol. 23, no. 4, pp. 207–229, 2017.

[15] T. Akiba and Y. Iwata, “Branch-and-reduce exponential/FPT algorithms in practice: A case study of vertex cover,” Theoretical Computer Science, vol. 609, pp. 211–225, 2016.

[16] M. Xiao and H. Nagamochi, “Exact algorithms for maximum independent set,” Information and Computation, vol. 255, pp. 126–146, 2017.

[17] M. Katzmann and C. Komusiewicz, “Systematic exploration of larger local search neighborhoods for the minimum vertex cover problem,” in Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, 2017, pp. 846–852.

[18] R. M. McConnell, K. Mehlhorn, S. Naher, and P. Schweitzer, “Certifying¨ algorithms,” Computer Science Review, vol. 5, no. 2, pp. 119–161, 2011.

[19] J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Lariviere, A. Beygelzimer,\` F. d’Alche Buc, E. Fox, and H. Larochelle, “Improving reproducibility´ in machine learning research: A report from the NeurIPS 2019 reproducibility program,” Journal of Machine Learning Research, vol. 22, no. 164, pp. 1–20, 2021. [Online]. Available: https: //www.jmlr.org/papers/v22/20-303.html

[20] TUB<sup>¨</sup> <sup>˙</sup>ITAK ULAKB<sup>˙</sup>IM, “TRUBA ARF queue and hardware information,” Official user documentation, accessed: 2026-08- 23. [Online]. Available: https://docs.truba.gov.tr/1-kaynaklar/arf/arf kuyruk bilgisi.htm

[21] D. R. Cox, “Regression models and life-tables,” Journal of the Royal Statistical Society: Series B, vol. 34, no. 2, pp. 187–220, 1972.

[22] S. Holm, “A simple sequentially rejective multiple test procedure,” Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.

[23] C. J. Clopper and E. S. Pearson, “The use of confidence or fiducial limits illustrated in the case of the binomial,” Biometrika, vol. 26, no. 4, pp. 404–413, 1934.

[24] F. Moessbauer, “LibMVC: A header-only library for minimum vertex cover solvers,” Software, commit cbcbd53c434b0de096f26b9e91812439ae35f304, 2022. [Online]. Available: https://github.com/fmoessbauer/LibMVC