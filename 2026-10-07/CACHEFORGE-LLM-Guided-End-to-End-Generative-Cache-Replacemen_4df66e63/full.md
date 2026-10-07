# CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency

Kaushal Mhapsekar, Bita Aslrousta, Brijesh Kumar Bhayana, Paula Contreras,

Azam Ghanbari, Ethan Goodman, Anna Andriiko, Samira Mirbagher Ajorpaz

Department of Electrical and Computer Engineering

North Carolina State University, Raleigh, NC, USA

{ kmhapse, baslrou, bkbhayan, pcontre, aghanba2, ecgoodm2, aandrii, smirbag}@ncsu.edu

Abstract—Modern cache replacement designs saturate because they operate within fixed representational structures, handcrafted and heuristic based feature-engineered predictors, or offline imitation models that cannot generate new decision logic on their own. At the same time, replacement is shaped by the causal interaction of prefetching, thrashing, spatial locality, and access-type behavior, producing an enormous design space that is difficult to traverse manually. Prior approaches typically rely on heuristics, parameter tuning, or imitation of an offline optimal policy, capturing correlations rather than synthesizing new mechanisms. As a result, their performance gains often plateau and they overfit under dynamic workload conditions.

CACHEFORGE is the first framework to evolve cachereplacement policies end-to-end by embedding a large language model inside a governed hardware-aware loop. In each iteration, the LLM proposes new C++ replacement logic, the policy is evaluated under a trace-based CRC-2 ChampSim simulator, and the framework enforces feasibility through reward shaping, structural checks, dynamic mutation, temperature scheduling, and cross-policy crossover. This closed-loop generation-evolution loop specifically designed for cache replacement policy enables the discovery of compact policies that satisfy hardware constraints while exploring algorithmic transformations beyond fixed predictor structures.

Across SPEC CPU2006, CACHEFORGE outperforms all CRC-2 baselines. It improves the total hit rate by 27.36%, 19.69%, 13.72%, 13.15%, 11.83%, and 5.73% over MPPPB, ReD, Hawkeye, SHiP++, LIME, and LRU, respectively. On memory-intensive workloads, it increases IPC by 10.15%, 7.89%, 6.34%, 3.64%, 3.12%, and 2.71% over LRU, MPPPB, LIME, ReD, SHiP++, and Hawkeye.

## I. INTRODUCTION

Off-chip DRAM accesses cost hundreds of cycles; consequently, a small number of exposed LLC misses can determine end-to-end performance. Replacement policies therefore remain a critical optimization point: recent work shows that carefully modifying eviction decisions can still yield doubledigit IPC gains. For example, Mockingjay [18] improves IPC despite worsening MPKI by suppressing prefetch pollution in specific sets, while PACIPV [15] improves IPC by 1.1– 3.3% by retaining lines whose misses would directly stall the core. These results demonstrate that the effectiveness of a replacement policy depends on which misses are exposed to the critical path, not on aggregate hit rate.

Modern processors amplify this effect. Every LLC access interacts with speculation, coherence, prefetching, DRAM scheduling, and instruction-stream behavior [3], [15], [18]. Evicting a line that feeds a dependence chain can introduce a long stall even if total hit rate improves; retaining a line that supports forward progress can eliminate a stall even if MPKI increases elsewhere. Replacement is therefore a causal decision problem influenced by dynamic interactions, rather than a static recency heuristic.

Despite this causal complexity, prior automated designs operate within human-defined representational bounds. The IPV framework [9] searches insertion and promotion vectors, a space of $1 6 ^ { 1 7 } \approx 3 \times 1 0 ^ { 2 0 }$ possibilities. Multiperspective Placement Promotion and Bypass (MPPPB [10]) explores over $1 0 ^ { 1 1 0 }$ combinations across sixteen expert-curated feature families, requiring nearly ten CPU-years of search. PACIPV [15] restricts exploration to 1024 RRIP-derived vectors to make exhaustive search tractable. These efforts demonstrate both the promise and saturation of fixed design spaces: no matter how large the parameter space is, the algorithmic structure is fixed in advance.

Replacement policy design has therefore converged on two classes of approaches. First, manually engineered heuristics such as DIP [17], RRIP [8], SHiP [23], and ReD [5] encode expert insight into compact predictors, but cannot alter their control flow or metadata structures. Second, correlationdriven learning frameworks such as Hawkeye [7], Glider [19], Mockingjay [18] and PARROT [13], imitate desirable behavior from traces, but their eviction logic is frozen and cannot adapt based on new causal feedback. Across both categories, the fundamental limitation is the same: the algorithmic structure of the policy is fixed, and search is confined to parameters within that structure.

This limitation arises because replacement is simultaneously a pattern-recognition problem (predicting reuse, phase changes, and prefetch usefulness) and a causal decisionmaking problem (determining which evictions alter future exposed misses). Prior approaches strengthen one side of this decomposition, pattern recognition or decision heuristics, but none can evolve both. Even IPV and MPPPB, which perform causal evaluation in simulation, are confined to rigid representations whose algorithmic frames cannot change.

This motivates a natural question: what if a replacement policy could evolve its own algorithmic structure, its decision flow, metadata rules, and control logic, while simultaneously adapting its finer-grain behaviors to the patterns it observes? What if policies were not tuned, but grown?

We introduce CACHEFORGE, a framework that answers this question by treating a replacement policy as mutable code. A large language model generates candidate C++ mechanisms whose logic may differ structurally from their predecessors; a trace-based simulator provides causal performance feedback; and a governed evolutionary loop enforces hardware constraints and prevents unbounded growth. Unlike prior work [10], CACHEFORGE is not restricted to a fixed predictor template: both algorithmic structure and pattern-interpreting logic can evolve.

Our experiments show that a naive LLM cannot accomplish this task on its own. The solution is far from obvious: without explicit governance, the model consistently fails in ways that render autonomous evolution infeasible. It hallucinates code, emits logic that does not compile or violates interface contracts, and frequently produces mechanisms that explode in metadata size or ignore feasibility constraints altogether. When seeded with existing policies, it tends to imitate them too literally and loses exploratory novelty; when unseeded, it repeats the same structural mistakes across generations. Without a structured memory hierarchy, it oscillates or reverts to previously invalid patterns; with naive memory, it collapses into a single design lineage and becomes trapped in local minima. Without carefully shaped rewards, it aggressively maximizes hit rate, occasionally at the cost of IPC by protecting prefetch or writeback lines. Without objective constraints, it invents simplistic but unbounded mechanisms that violate hardware budgets. In short, an unguided LLM either diverges, overfits, collapses, or produces policies that are architecturally meaningless, demonstrating that a governed, hardware-aware evolutionary loop is not optional, but necessary for reliable discovery.

CACHEFORGE resolves these issues by embedding the LLM inside a tightly governed, hardware-aware generateevaluate-refine loop. Syntax- and structure-aware prompting ensures that every candidate conforms to the ChampSim interface, while static feasibility checks reject hallucinated or unsafe code before simulation. Storage-budget constraints and metadata accounting prevent unbounded growth, forcing the model to trade capacity against performance. A trace-based evaluator provides the causal feedback that imitation-based methods lack, punishing designs that increase hit rate but lengthen exposed stalls. Reward shaping aligns optimization with IPC and IPC-per-KB rather than raw hit rate, steering generation toward demand-critical behaviors. A three-tier memory system: ephemeral, episodic, and summary, retains useful design lessons without collapsing exploration, while dynamic temperature and mutation scheduling maintain diversity while encouraging novelty and prevent local minima.

Structural crossover between top-performing but dissimilar policies injects genuine novelty, and model switching breaks late-stage stagnation. Together, these mechanisms convert an otherwise unstable generator into a disciplined architectural search engine capable of producing hardware friendly, feasible, and high-performance replacement algorithms.

A further limitation of prior learning-based replacement policies is the complete absence of explainability and the lack of phase- or workload-awareness. PARROT [13] and Glider [19] attempt to interpret their neural predictors by inspecting attention vectors or LSTM activations, and CHiRP [14] analyzes perceptron and Adaline weights to identify correlated features; however, these signals reflect only statistical associations learned offline from fixed training traces. Once training ends, the model’s behavior is frozen: it cannot adapt its interpretation of locality across phases, workloads, or architectural contexts, nor can it reveal the causal reasoning behind its eviction choices. At best, these frameworks expose coarse indicators of correlation, e.g., which PCs were influential in the training distribution, but not the algorithmic principles governing online decision-making. This has significantly limited the use of advanced AI in the field and the trust level needed for implementing these policies in the hardware.

In contrast, CACHEFORGE represents each policy as executable C++ code, making its mechanisms, metadata structures, features, and decision making, explicitly visible and analyzable. Because evolution occurs inside a trace-based simulator, every modification is validated against the current phase behavior and workload dynamics, not a frozen offline trace. Moreover, each evolutionary step is archived in a persistent memory ledger, allowing users to inspect how the policy’s logic changes over time, which structures emerge or vanish, and which design ideas survive selection. This yields the first explainable learning-driven replacement framework: the causal logic of the policy is transparent because it is written in code, and its developmental trajectory is transparent because every mutation, crossover, and rationale is recorded. Prior neural approaches reveal correlations after training; CACHEFORGE reveals the algorithm as it is created.

The remainder of this paper formalizes this approach and demonstrates that governed code-space evolution can synthesize compact, high-performance replacement mechanisms beyond the reach of fixed representations.

Contributions. This paper makes four contributions:

Generative code-space evolution for cache replacement. We introduce CACHEFORGE, the first system that treats a replacement policy as mutable code, allowing the LLM to rewrite control flow, metadata structures, decision rules, and scoring logic, while a trace-based ChampSim evaluator provides causal feedback and enforces feasibility. This moves beyond fixed representations such as IPV vectors, RRIP templates, or MPPPB feature libraries, enabling algorithmic forms no prior framework can express autonomously.

• Autonomous discovery of high-performance and hardware-efficient mechanisms. Governed evolution synthesizes compact policies that surpass all CRC-2 baselines. Across SPEC CPU2006, CACHEFORGE improves IPC by 10.15%, 7.89%, 6.34%, 3.64%, 3.12%, and 2.71% over LRU, MPPPB, LiMe, ReD, SHiP++, and Hawkeye, and raises total hit rate by 27.36%, 19.69%, 13.72%, 13.15%, 11.83%, and 5.73% over MPPPB, ReD, Hawkeye, SHiP++, LiMe, and LRU. It simultaneously reduces metadata overhead by 43% relative to MPPPB and delivers the highest IPC-per-KB efficiency among all evaluated designs.

• Comprehensive analysis of novel policies and sensitivity study. We identify 21 novel design components and several algorithmic hybrids synthesized only through code-space evolution, including SPARC+, TA-DPIP, RSP-MPPP, and SAGE-MPPPB. Sensitivity studies show how memory, temperature, mutation rate, prompt, and seed diversity shape convergence among 10,687 generated policies.

Paper Organization: Section II provides background and related work. Section III presents the CACHEFORGE design. Section IV describes our evaluation methodology. Section V reports experimental results. Section VII provides sensitivity analysis. Section VIII concludes the paper.

## II. BACKGROUND AND RELATED WORKS

a) Classical heuristic policies: Classical LLC replacement mechanisms such as DIP [17], RRIP [8], and SHiP [23] are manually engineered around recency, re-reference interval prediction, or lightweight program-context signatures. These heuristics are robust, interpretable, and low cost, but the decision logic is fixed: their metadata structures, state transitions, and victim-selection rules are predetermined and cannot evolve. ReD [5] inserts a line only after observing a prior reuse and uses PC-based logic to bypass predicted-dead blocks. SHiP++ [25] refines signature-based reuse prediction with improved SHCT training and separate handling of demand and prefetch lines. MPPPB [10] uses a perceptron that combines multiple program features to estimate reuse likelihood for bypass and insertion decisions.

b) Imitation-based learning: Recent approaches attempt to approximate Belady or exploit rich locality structure using offline learning. LIME [21] trains a lightweight demand-access classifier to approximate Belady and uses the prediction to choose insertion depth under SRRIP. Hawkeye [7] constructs per-PC labels using Belady’s MIN algorithm and learns a lightweight classifier that distinguishes “cache-friendly” from “cache-averse” lines. Glider [19] trains deep sequence models to capture long-range reuse patterns and distills them into a compact online predictor. PARROT [13] uses LSTM-based sequence modeling to imitate Belady’s optimal algorithm [2] directly from traces. These methods improve pattern recognition but share two fundamental limitations: (1) their algorithmic structure is frozen after training—no new metadata, control flow, or decision logic can emerge; and (2) they operate only on correlation signals, not the true causal impact of a decision on future execution.

c) Evolutionary frameworks with fixed representations: IPV [9] encodes each policy as an insertion–promotion vector and uses a genetic algorithm to search this space. Although this includes causal evaluation via simulation, the representation restricts behavior to LRU-stack manipulations; IPV cannot invent new mechanisms or metadata behaviors. MPPPB [10] aggregates information from sixteen expert-designed perspectives (PC bits, region signatures, reuse classifications, temporal decay signals, access-type discriminators, and more), exploring over 10<sup>110</sup> possible feature combinations. This search required roughly ten CPU-years and produced strong predictors, but the algorithmic form, feature extraction followed by weighted summation and thresholding is fixed. PACIPV [15] restricts exploration to 1024 RRIP-derived vectors, enabling filtered exhaustive search, but the search is confined to the RRIP abstraction and yields only marginal IPC gains. These frameworks employ causal evaluation but are limited to the behaviors representable within their fixed templates.

d) Foundational LLM Reasoning Frameworks: LLMs have shown promising results when it comes to solving multistep problems, by leveraging techniques such as Chain-of-Thought prompting [22] and zero-shot reasoning [12]. Their reasoning capabilities were further augmented with agent based frameworks like ReAct [24], which combines reasoning with environment-specific actions. Building on these LLM capabilities, a new class of frameworks has emerged that focus on automated optimization and discovery. These frameworks deal with using LLMs to optimize their own prompts, such as APE [26] and PromptWizard [1], automatically generate autonomous agents with frameworks like Autoagents [4], and discovering novel algorithms for producing “as-good-aspossible” solutions to NP-Hard problems (FunSearch) [20] and even finding solutions to various other scientific and mathematical problems (AlphaEvolve) [16]. Recent studies have shown that LLMs are prone to “hallucinations” resulting in production of confident falsehoods [11] and lack causal, domain-specific grounding, often limiting their ability to be applicable to the hard constraints of physical hardware design.

## III. CACHEFORGE

Imitation-based frameworks capture statistical correlations but cannot adapt causally or revise their structure. Causal approaches incorporate simulation feedback but remain limited by fixed, human-designed representations. A more flexible framework is needed, one that supports code generation, causal feedback, hardware-aware objectives, and generalization across workloads.

Prior evaluators also suffer from misaligned objectives. Most optimize hit rate or MPKI, assuming a monotonic relationship with IPC. This assumption breaks down in modern processors: demand and RFO misses stall retirement, whereas prefetch hits, writebacks, and speculative accesses raise hit rate without improving performance. As a result, policies such as SHiP, RRIP, MPPPB, and PACIPV often increase total hits while reducing demand-service efficiency. A mechanism to tune the reward and optimization target using direct feedback is therefore necessary.

![](images/766434bf8e8a403bc46c08eb6c77826b34299e2e59eb31455715a67486fd3b4f.jpg)  
Fig. 1. CACHEFORGE system diagram.

CACHEFORGE addresses this gap by enabling evolutionary exploration of executable replacement-policy code under true hardware constraints and causal feedback.

CACHEFORGE embeds an LLM inside a closed generatorevaluator loop. At each iteration, the generator proposes candidate code, ChampSim evaluates it under identical setups, and a reward guides refinement. The evaluator enforces physical correctness, preventing hallucinated or semantically incorrect code from propagating.

Policy Generator. The Policy Generator (Figure 1 A) connects an LLM to the CRC-2 ChampSim loop to autonomously synthesize C++ replacement policies. Each iteration retrieves top-performing policies, their metadata, and rationales from a policy database. These are encoded with objectives, reward, constraints, and workload summaries into a structured prompt. The LLM produces a policy name, short rationale, C++ implementation, and explicit overhead estimate.

A strict one-shot prompting schema ensures syntactic validity and compatibility with ChampSim, while high-performing policies are reintroduced in subsequent prompts to enable chain-of-thought–style refinement. CACHEFORGE also supports dynamic seeding: generation may begin from example baselines such as RRIP, SHiP, Hawkeye, or MPPPB for guided refinement, or from an empty context for free-form exploration. This flexibility allows the system to alternate naturally between anchored optimization and open-ended innovation.

Simulation & Evaluation. Each generated policy is compiled using the CRC-2 version of ChampSim and executed across representative SPEC CPU2006 workloads. The simulator returns performance metrics such as IPC, per-access-type hit rates and MPKI, and estimated storage overhead, which are automatically parsed and recorded (Figure 1 B). These quantitative results serve as the feedback signal that steers the next design iteration, transforming the system into a self-contained reinforcement loop.

Objective and Reward. Reward translates simulator statistics into a single scalar signal that drives the CACHEFORGE optimization loop. After each generated policy is simulated,

ChampSim reports key cache metrics like IPC, MPKI, total hit rate, and per-access-type hit rates (Load, RFO, Prefetch, and Writeback). These metrics are combined according to the active optimization objective defined by the Optimization Objective Controller (Figure 1 C,D).

CACHEFORGE distinguishes between the reward returned by simulation and the objective embedded in the prompt. This separation enables guided divergence, encouraging innovation rather than premature convergence.

Moreover, the reward serves both as evaluation and as alignment: it penalizes excessive metadata, latency-increasing structures, or useless prefetch retention by instructing the LLM to track storage overhead, constraining designs to remain physically realizable and comparable to existing CRC-2 implementations in both capacity and complexity. Finally, high-reward policies populate the positive memory set; lowreward ones enter negative conditioning.

Memory-Guided Refinement. CACHEFORGE maintains a three-tier memory hierarchy to accumulate and reuse design knowledge. Ephemeral memory captures short-term context from the current round. Episodic memory stores structured experiment outcomes and short textual lessons. Summary memory distills persistent principles, constraints, and “don’ts” to inform future prompts. Before each new iteration, the system retrieves the most relevant memories using vector similarity and prepends them to the prompt, ensuring continuity of reasoning. This mechanism allows CACHEFORGE to avoid redundant exploration and progressively internalize its own design insights (Figure 1 A).

Exploration–Exploitation. CACHEFORGE operates in two phases: exploration, where the system searches broadly for new algorithmic structures, and exploitation, where it refines high-performing designs. The system transitions between these phases based on recent simulation results: lack of improvement triggers exploration, while consistent gains shift the system toward exploitation (Figure 1 C).

To implement these phases, CACHEFORGE uses three lightweight control mechanisms. First, a Dynamic Temperature Controller (DTC) adjusts the model’s sampling temperature (τ ). High temperatures (τ = 0.9–1.0) promote diversity during exploration, whereas lower temperatures (τ = 0.7–0.8) stabilize refinement during exploitation. Empirically, moderate values near $\tau = 0 . 7$ strike an effective balance between novelty and convergence.

TABLE I PROCESSOR AND MEMORY CONFIGURATION.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Processor</td><td>1 core; 4 GHz; 6-wide fetch/decode/execute; 4-wide retire; 256-entry ROB; gshare branch predictor</td></tr><tr><td>L1 I/D-Cache</td><td>32 KB, 64 sets, 8 ways, 4-cycle latency, next-line prefetcher;</td></tr><tr><td>L2 Cache</td><td>256 KB, 512 sets, 8 ways; 12-cycle latency, PC-based stride prefetcher;</td></tr><tr><td>LLC</td><td>2 MB, 2048 sets, 16 ways; 32-cycle latency;</td></tr><tr><td>L1 TLB (I/D)</td><td>4KB; 16 sets, 8 ways</td></tr><tr><td>L2 TLB</td><td>96KB; 128 sets, 12 ways</td></tr></table>

Second, a mutation controller determines how strongly each new policy should deviate from its predecessor. Improvements trigger small refinements (mutation rate 0.2), moderate regressions increase mutation to 0.4, and large regressions raise it to 0.9 or 1.0 to force substantial structural changes. Plateaus induce a moderate mutation rate (0.6) to escape stagnation. These directives guide the degree of permissible algorithmic change each iteration.

Finally, when refinement no longer yields progress, CACHEFORGE can activate a tree-based diversification. The system (i) further refines the current parent policy, (ii) performs crossover between top-performing but structurally different designs, and (iii) introduces entirely new policies using accumulated design lessons. This branching search helps escape local optima and reintroduces structural diversity when linear refinement is insufficient.

Together, temperature adaptation, mutation control, and tree-based diversification maintain broad exploration early in search and support focused, stable convergence once promising design directions emerge.

## IV. METHODOLOGY

Simulator and Workload. We use the CRC2-ChampSim trace-based simulator [6] to evaluate policies for single core configuration with prefetcher both enabled & disabled. Each policy is integrated into the CRC2-ChampSim codebase and evaluated using 28 SPEC 2006 CPU traces. We report hit rate, IPC, Speedup, and IPC-per KB for all workloads and as well as the memory-intensive subset of five workloads (astar, mcf, milc, lbm, omnetpp) with the highest sorted miss rate. We report the hardware overhead and our best IPC-per-KB policy for all 88 simulation points after 200M warmup and 1B instructions, execution.

CACHEFORGE automates the entire code–evaluation pipeline with zero-human intervention through the evolutionary phase. Each iteration consists of four stages: (1) Generation, where the LLM produces a syntactically valid C++ policy; (2) Compilation, which integrates the policy into ChampSim; (3) Execution, where the policy is simulated under each evaluation window; and (4) Evaluation, where key metrics like IPC, total hit rate, per-access-type hit rates, MPKI, and estimated metadata cost are extracted. This study compares the results of 10,687 fully autonomously generated policies.

We compare results for the best policy over 260 iterations of CACHEFORGE evolution. We observed that evolving CACHEFORGE by evaluating the workloads for 100M vs 1B does not change the ranking of the policies but significantly reduces the evaluation overhead for CACHEFORGE per iteration required for evolution. We provide insights on CACHEFORGE’S evolution for 10 M(1 M warmup), 100 M(100 M warmup), and 1 B(200 M warmup) execution per instruction since it affects the total evolutionary run time and explain how it impacts the final hit rate, IPC, IPC improvement Per-KB.

Baseline Replacement Policies. We compare the best CACHEFORGE generated policies with the top-performing CRC2 replacement policies: SHiP++, Hawkeye, Less Is More (LIME), Multiperspective (MPPPB), Reuse Detection (ReD), and LRU, as human-designed seed examples [7], [10], [23]. These perform the impact of each of these policies provided to the CACHEFORGE as seeded runs and report its impact on the evolution of CacheForge quantitatively. We consider MPPPB the state of the art AI-assisted replacement policy generator and compare CACHEFORGE novel feature generation with that.

Language Models Evaluated. We evaluate CACHEFORGE using multiple large language models under identical prompt schemas and hyperparameters: GPT-5, o4-mini, Gemini 2.5 Pro, DeepSeek-Coder 33B, and Qwen 2.5. This comparison isolates architectural effects from model-specific generative behavior.

## V. RESULTS

## A. Guided Evolution for Various Objectives.

This section shows how CACHEFORGE can be used to optimize various objectives. Including metrics with complex dynamic interactions in CPU, such as hit rate and IPC.

Although LLC hit rate is often treated as a proxy for performance, in practice it is a remarkably poor stand-in for what the core actually experiences on its critical path, and a naive attempt to “maximize hit rate” can easily degrade IPC.

The fundamental problem is that hit rate is an undifferentiated aggregate over access types, sets, and phases, whereas stalls are driven by a tiny subset of accesses: long-latency, poorly overlapped demand loads. A replacement policy can raise total hit rate by zealously protecting prefetch and writeback lines, or by retaining demand lines in cold sets, yet simultaneously evict the very lines that matter in hot sets where reuse is dense and MLP is limited.

From a statistical point of view, a prefetch that hits looks identical to a load that hits; from the processor’s point of view, the former is often invisible while the latter may save tens of cycles in a dependence chain. Even worse, the rebalancing of occupancy that improves the global hit rate can reduce memory-level parallelism or increase bank and MSHR contention, making the remaining misses more serialized and more frequently exposed to the retire stage. The result is that we can construct policies with strictly higher hit rate and lower MPKI that nonetheless lengthen critical-path stalls and reduce IPC. In other words, maximizing hit rate optimizes for “how many lines live in the cache”, not for “which misses the core actually pays for”.

## B. Automatically Learning What to Optimize

Our first and most surprising observation is that, once we let CACHEFORGE run long enough, it immediately disproves the naive objective we started with. If we sort the thousands of generated policies by LLC hit rate and then examine their IPC, we often see the opposite of what a human designer would expect: many of the highest-hit-rate policies deliver lower IPC than the baselines, while some of the best-IPC policies live in the middle of the hit-rate distribution (Figure 2).

![](images/62573604cd734ff479b0945ea3fe1df66e357587ad048b394094949a22990db4.jpg)  
Fig. 2. Anti-correlation of IPC and Cache Hit Rate observed.

From a traditional, hand-designed study with a handful of policies, this is easy to miss or dismiss as an anomaly. In CACHEFORGE, the pattern is impossible to ignore, because the generator–evaluator loop continuously produces entire populations of policies that cover a wide swath of the design space. The system effectively runs the thought experiment a human architect would find exhausting: “what happens to IPC if I push hit rate up in every conceivable way?”; and the answer, repeatedly, is that aggregate hit rate is the wrong hill to climb.

IPC. Figure 3 shows that IPC rises steadily over time for CACHEFORGE. only 69 iterations of CacheForge improves IPC to 0.334, surpassing all SOTA baselines with improvements of 10.16% over LRU, 7.89% over MPPPB, 6.34% over LiMe, 3.64% over ReD, 3.12% over SHiP++, and 2.71% over Hawkeye. Across the evaluated workloads, CACHEFORGE achieves substantial performance gains over LRU, improving IPC by 11.42% on lbm, 6.48% on omnetpp, 11.30% on milc, 11.11% on astar, and 16.73% on mcf, with a 10.16% speedup overall (Figure 4).

Hit Rate Figure 5 shows that average Hit rate continuously evolving over 260 iterations of CACHEFORGE starts from 47.68% and goes up to 53.43%, a 4.67× times improvement over MPPPB. This number is 42.18%, 54.15%, 42.26%, 23.31%, and 28.16% over Hawkeye, ReD, SHiP++, LIME, and LRU respectively. A small but important detail is that progress plateaus near iteration 250; replacing the underlying LLM (from o4-mini to Gemini 2.5 Pro) at iteration 265 immediately lifts the hit rate from 52.70% to 53.43%, indicating that the late-stage plateau was model-limited rather than search-limited.

![](images/e5a72371a7e5b2a047bbe42cb84c8a607fb59a588ae58e0e1e5fc8408ec2741b.jpg)

Fig. 3. Optimizing for IPC over iterations.  
![](images/e2ce3d1bcaa34dbc1309ce656ab2fa74f3bb6d2d318d1b3f5a17a2e439f9a715.jpg)  
Fig. 4. Speedup of CACHEFORGE relative to LRU across SPEC workloads, compared against state-of-the-art baselines.

![](images/3e18e0b8ae5277d41ae9258c88686375f94d36e998c9cc6c148f0cb28fdee509.jpg)  
Fig. 5. Cache Hit Rate improvement using 4o-mini+Gemini 2.5 pro. The first vertical line shows the first instance where CACHEFORGE surpasses a state-of-the-art policy and the second vertical line signifies the model change from 4o-mini to Gemini 2.5 pro.

Figure 6 shows how CACHEFORGE progressively improves individual access-type hit rates across iterations. Breaking down the total hit rate reveals distinct microarchitectural trends. For example, Load hit rate, represents demand fetch reuse in cache. CACHEFORGE improves this type of hits as well, from 44.8% to 47.5% a 6.7% improvement over the state-of-the-art baseline. RFO hit rate, captures the locality of store ownership, which CACHEFORGE enhances from 46.6% to 54.0%, a 15.6% improvement.

![](images/633eb20eb02b8c716e8741caffffd5319282d9b890508289442cf1441935d2ac.jpg)  
Fig. 6. Comparison of Load and RFO Hit Rate evolution over iterations. Two blue tones denote the access types, with dashed lines indicating RFO progression.

![](images/5e637915d3bd384e89d2db721bf06313ce1ed9bec3ba03433133068a3f270a15.jpg)

![](images/4c8a8eac9a372ff963dbdb7b4d11d058b511fa4f6963ebe086eb4fa17e16e86a.jpg)  
Fig. 7. Sorted cache hit rates for policies evolved under Hit Rate, Average, and Weighted Average rewards and sorted IPC for policies evolved under IPC, and Average rewards.

Weighted Reward Function. Each generated policy is scored using a composite reward function that aligns the simulator’s raw statistics with architectural performance goals. We evaluate three formulations: (i) a Hit-Rate Reward emphasizing overall cache reuse, (ii) an Average Reward combining hit rate and IPC equally, and (iii) a Weighted Average Reward that biases towards IPC rather than hit rate. Empirically, this design yields the best performing policies, improving total hit rate by 3–5% over CacheForge with IPC alone as the reward and demonstrating that reward shaping provides a powerful means of guiding LLM-driven optimization toward high-value microarchitectural trade-offs (see Figure 7).

Generalization vs. Overfitting evaluation. To measure the generalization capability of CACHEFORGE, we autonomously synthesized policies over 10 M-instruction window using exploration phase for faster evolution andfive training workloads, and evaluated them across all 28 SPEC workloads for 1B instruction. On average, it achieves a 15.52% (5.73%, 11.83%, 27.36%, 13.15%, 13.72%, 19.69% over LRU, Less Is More, MPPPB, SHiP++, Hawkeye, and ReD) improvement in total hit rate over the baseline policies.

Taken together, these results show that CACHEFORGE discovers, through trial and validation on the real simulator, what the metric should have been in the first place. By observing thousands of candidates, CACHEFORGE learns that total hit rate is at best incidental, and at worst misleading, and gradually shifts its behavior toward improving the access types and structural patterns that genuinely raise IPC.

Once the reward is aligned with these causal signals, the system consistently refines policies toward higher Load and RFO hit rates, shorter exposed miss latencies, and better IPC-per-KB efficiency without any offline training or human heuristics.

![](images/dad2a78cc070b4d094b1ebff6ee589bd4c070bc1d547ff17e990c3f65203f8c8.jpg)  
eyeFig. 8. Successful replacement policy hardware overhead budget alignment

![](images/4ff4a34d86fa87d3809405c46c07eddf6d53ef68d073d632b36c67b14df9c3ec.jpg)  
Fig. 9. Storage Overhead (KB) of CacheForge Proposed Policies vs. Baseline Policies

## C. Evolving at Controlled and Low Budget

Hardware Overhead. Figure 8 shows that CACHEFORGE reliably follows storage constraints. When prompted with different storage-budget ranges, the generated policies cluster tightly within the corresponding actual storage bands, indicating that the LLM not only understands the budget specification but produces implementations that remain aligned with it. This confirms that storage limits can be enforced effectively through prompting alone in CacheForge.

<table><tr><td>Novel Design Component</td><td>Objective</td><td>Impact</td><td>CacheForge Capability</td></tr><tr><td>Reinforcement-Learning Q-Value Candidate Ranking</td><td>Learn eviction value online Score all candidates</td><td>Qualitative only High granularity eviction decisions</td><td>Explores unconventional algorithmic spaces Applies ranking-based eviction strategies</td></tr><tr><td>Online Reward Adaptation</td><td>Adjust RL policy using hits/misses</td><td>Adaptive learning</td><td>Experiments with reinforcement learning</td></tr><tr><td>Dead-Block Prediction</td><td>Identify low-reuse lines</td><td></td><td></td></tr><tr><td></td><td></td><td>Improves reuse filtering</td><td>Rediscovers classical dead block mechanisms</td></tr><tr><td>Evicted-Address Filter</td><td>Detect recently-dead lines</td><td>Reduces pollution</td><td>Uses compact deadness filters effectively</td></tr><tr><td>Prefetch Usefulness</td><td>Reduce unused prefetches</td><td>Reduces cache pollution</td><td>Models prefetch awareness</td></tr><tr><td>Region Hotness Tracking</td><td>Identify hot or thrashing regions</td><td>Reduces repeated thrashing</td><td>Infers region-level memory behavior</td></tr><tr><td>Ghost Re-Reference Buffer</td><td>Detect premature evictions</td><td>Corrects eviction mistakes</td><td>ghost-based structures</td></tr><tr><td>Probabilistic Stack-Distance</td><td>Estimate reuse using recency stack</td><td>Improves graded reuse prediction</td><td>Investigates probabilistic reuse modeling</td></tr><tr><td>MRU Promotion Control</td><td>Limit noisy MRU promotions</td><td>Reduces pollution due to bursts</td><td>understand promotion strategies</td></tr><tr><td>Adaptive Bypass Control</td><td>Avoid caching low reuse lines</td><td>Reduces pollution</td><td>Understands bypass rules</td></tr><tr><td>Stride Detection</td><td>Detect stride accesses</td><td>Improves predictable performance</td><td>Recognizes regular temporal locality</td></tr><tr><td>Stream Detection</td><td>Detect streaming behavior</td><td>Reduces streaming pollution</td><td>Identifies spatial locality</td></tr><tr><td>PC Friendliness Scoring</td><td>Differentiate friendly vs. averse PCs</td><td>Improves context-aware reuse</td><td>Learns PC/context-driven locality</td></tr><tr><td>Global Predictor Decay</td><td>Prevent counter saturation</td><td>Maintains long-term stability</td><td>Manages dynamic predictor state</td></tr><tr><td>Weighted Eviction Ranking</td><td>Score MAX-RRPV candidates</td><td>More precise victim selection</td><td></td></tr><tr><td></td><td></td><td></td><td>Constructs multi-factor scoring methods</td></tr></table>

NOVEL DESIGN COMPONENTS EXPLORED BY CACHEFORGE.

![](images/a2f2d5f7a471c697fb66975433901763fecc30cc1092b9844322ee1a8110a5d1.jpg)  
Fig. 10. IPC per KB improvement over Multiperspective.

Figure 9 presents the storage overhead of the proposed CACHEFORGE policy relative to established CRC-2 baselines. The baseline policies require between 16 KB and 31.875 KB of LLC metadata: LRU (16 KB), SHiP++ (20 KB), MPPPB (29.05 KB), LIME (31.20 KB), Hawkeye (31.80 KB), and ReD (31.875 KB). In contrast, CACHEFORGE uses only 12 KB. This represents a substantial reduction, CACHEFORGE requires 43% less storage than MPPPB, the most comparable high-performing baseline, while still delivering higher IPC/KB efficiency (Figure 10).

## VI. NOVEL CACHEFORGE POLICIES.

Table II details over some of the novel design components included in the top policies designed by CacheForge across various experiment configurations. The table also captures some of the existing and new ideas in cache replacement domain that are explored by CacheForge by combining some of these ideas into existing replacement policies to come up with novel policies.

We analyzed 21 features across 491 high performing CacheForge policies and sorted them by impact on cache hit rate (Figure 11). Our studies revealed that designs with certain features outperform others. For example, set sampling, bypassing, stride detectors, and confidence predictors have a higher average impact on cache hit rate than designs that had eviction address filters, scan awareness and pollution awareness components. Subsequently, we briefly analyze some of the top policies generated by CacheForge.

Spatial PC Aware Reuse Correlator+ (SPARC+), the best IPC-oriented policy generated by CacheForge, improves IPC by 10.15% over LRU, across a subset of five high-MPKI, cache-sensitive SPEC workloads. SPARC+ adds a Graph Node Table to track spatial locality, reuse aware victim selection logic, hybrid insertion, and PC-based victim feedback to SHCT table. SPARC+ also applies a light periodic decay of graph score to avoid saturation and to maintain stable behavior over long traces. These combined refinements allow SPARC+ to improve reuse efficiency and latency performance, demonstrating CacheForge’s ability to produce simple yet effective enhancements to a well-established replacement policy.

Thrash-Aware Dual-Pattern Insertion with SHiP Reuse (TA-DPIP), a compact policy is synthesized by CacheForge when it was reconfigured to optimize purely for total cache hit rate. The policy retains the standard 2-bit RRIP victim selection mechanism and augments it with several lightweight, signature-indexed signals that modulate insertion priority: a SHiP-style temporal reuse counter, a stride-regularity counter that strengthens when the current line index repeats the last non-zero stride, a pointer-likeness counter that strengthens when the stride changes, and a small thrash-propensity counter that increases when lines from a signature are evicted without being referenced. These signals allow TA-DPIP to separate predicted-reuse signatures from streaming, pointer-heavy, or thrashing behaviors.

TA-DPIP illustrates that CacheForge can synthesize policies that are both interpretable and hardware-efficient.

Region–StreamPath Multiperspective Predictor (RSP-MPPP), a novel policy synthesized by CacheForge, achieves IPC/KB improvement over MPPPB by 135%, and LRU which has the best baseline IPC/KB by nearly 30%. To achieve this performance, RSP-MPPP, underperforms MPPPB for IPC by 1.54%, while cutting the storage requirement down by 49.37% over MPPPB. On the other hand, it improves IPC over LRU by 1.08% while reducing area by 21.97% over LRU.

![](images/5b4f6f5a676df4e9a7d7eb9e390ddb9990cab61c75235d7a1f691ab51ddce357.jpg)  
Fig. 11. Heatmap showing the impact of novel features explored in CacheForge over Hit Rate.

Table III details on the tuning and novel addition made by CacheForge to the baseline MPPPB policy. Collectively, these modifications reduce sampler and predictor overhead while stabilizing confidence estimation, yielding improved IPC/KB performance at a smaller footprint. Thus, CacheForge’s RSP-MPPP demonstrates the LLM’s ability to identify and tune critical thresholds, gating mechanisms, and feature structures within the Multiperspective framework, improving reuse prediction quality and cache efficiency under strict storage constraints.

Set-Adaptive Gated Ensemble (SAGE-MPPPB) While optimizing for IPC/KB across iterations, CACHEFORGE also eventually improves the IPC performance relative to MPPPB. Using MPPPB as the initial seed, the system ultimately discovers Set-Adaptive Gated Ensemble (SAGE-MPPPB), the first generated policy to surpass MPPPB. SAGE-MPPPB tunes MPPPB to bring down storage requirements by 29.53% while adopting per-set hotness counting to achieve a 0.14% speedup. SAGE-MPPPB effectively improves IPC/KB over Multiperspective by 43.88%, demonstrating LLM’s capability to improve performance while keeping storage budget constrained.

## VII. SENSITIVITY STUDIES

Varying Seed Policy. We further test experiments with and without an explicit seed policy, in which the LLM either begins from an existing human-designed baseline or starts from a null design space. These seed and memory variations quantify the degree of dependency on prior knowledge and the model’s capacity for self-discovery. Figure 12 presents a box plot comparing cache hit rate distributions across 6 policies: Hawkeye, LIME, ReD, LRU, MPPPB, and ShiP++, revealing that ShiP++ achieved the highest average performance (29.5%), followed by MPPPB (25.5%). This pattern suggests that certain policy seeds, such as Multiperspective and Ship++, tend to generate more effective cache strategies overall, whereas Hawkeye-based seeds produced less efficient evolutionary results.

Varying Number of Seed Policy. We further assessed the impact of seed on IPC by varying to the number of input seeds shown in Figure 13. We observed that the amount of context given to the LLM influences its ability to refine prior designs, but not in a linear way: more examples do not always yield better results. The strongest performance occurred when the LLM was given the top ten policies, with the top two performing second best; both outperformed the configuration using the top five.

TABLE III
<table><tr><td rowspan="2">Component</td><td rowspan="2">Description</td><td colspan="5">Multiperspective</td><td colspan="5">CacheForge</td><td rowspan="2">Tuning</td></tr><tr><td>A</td><td>B</td><td>E</td><td>W</td><td>X</td><td>A</td><td>B</td><td>E</td><td>W</td><td>X</td></tr><tr><td colspan="10">Perceptron Features</td><td></td><td></td><td></td></tr><tr><td>F_PC</td><td></td><td>7</td><td>14</td><td>43</td><td>11</td><td>0</td><td>6</td><td>0</td><td>16</td><td>8</td><td>0</td><td>Changed</td></tr><tr><td>F_PC</td><td>PC: Bits B-E of the PC of the Wth</td><td>16</td><td>3</td><td>11</td><td>16</td><td>1</td><td>8</td><td>6</td><td>22</td><td>3</td><td>0</td><td>Changed</td></tr><tr><td>F_PC</td><td>memory access instruction. A is the</td><td>10</td><td>1</td><td>53</td><td>10</td><td>0</td><td>10</td><td>4</td><td>20</td><td>1</td><td>0</td><td>Changed</td></tr><tr><td>F_PC</td><td>recency-stack position beyond which the</td><td>16</td><td>8</td><td>16</td><td>5</td><td>0</td><td>12</td><td>4</td><td>20</td><td>0</td><td>2</td><td>Changed</td></tr><tr><td>F_PC</td><td>block is considered dead.</td><td>17</td><td>6</td><td>20</td><td>0</td><td>1</td><td></td><td></td><td>一</td><td></td><td></td><td>Removed</td></tr><tr><td>F_PC</td><td></td><td>17</td><td>6</td><td>20</td><td>0</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td>Removed</td></tr><tr><td>F_PC</td><td></td><td>17</td><td>6</td><td>20</td><td>14</td><td>1</td><td>一</td><td></td><td></td><td>一</td><td></td><td>Removed</td></tr><tr><td>F_OFF</td><td rowspan="2">Offset bits (B-E) extracted from the block offset of memory access.</td><td>15</td><td></td><td>6</td><td>0</td><td>1</td><td>8</td><td>4</td><td>6</td><td>0</td><td>0</td><td>Changed</td></tr><tr><td>F_OFF</td><td>10</td><td>0</td><td>6</td><td>0</td><td>1</td><td>10</td><td>2 0</td><td>6</td><td>0</td><td></td><td>Changed</td></tr><tr><td>F_OFF</td><td></td><td></td><td></td><td>1</td><td></td><td></td><td>14</td><td></td><td>6</td><td>0</td><td>0</td><td>Added</td></tr><tr><td>F_LM</td><td>Lastmiss bit indicating if last set access 9 missed.</td><td></td><td>0</td><td>0</td><td>0</td><td>0</td><td>10</td><td>2</td><td>6</td><td>0</td><td>0</td><td>Changed</td></tr><tr><td>F_INS</td><td rowspan="3">Insertion bit indicating whether the access is marked as an insertion.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>Changed</td></tr><tr><td>F_INS</td><td>16 8</td><td>0 0</td><td>0 0</td><td>0 0</td><td>1 1</td><td>8 一</td><td>0 1</td><td>0 一</td><td>一</td><td>1</td><td>Removed</td></tr><tr><td>F_INS</td><td>16 17</td><td>2 0</td><td>47 0</td><td>2 0</td><td>0</td><td></td><td></td><td></td><td></td><td></td><td>Removed</td></tr><tr><td>F_INS</td><td></td><td></td><td></td><td></td><td></td><td>1</td><td>一</td><td></td><td>一</td><td>一</td><td>一</td><td>Removed</td></tr><tr><td>F_BURST</td><td>Burst bit indicating cache burst.</td><td>6</td><td>11</td><td>22</td><td>9</td><td>0</td><td>6</td><td>0</td><td>0</td><td>0</td><td>2</td><td>Changed</td></tr><tr><td>F_BIAS</td><td rowspan="2">Bias bit.</td><td>16</td><td></td><td>0</td><td>0</td><td>0</td><td>16</td><td>0</td><td>0</td><td>0</td><td>0</td><td>No Change</td></tr><tr><td>F_TAG</td><td></td><td></td><td></td><td></td><td></td><td>12</td><td>30</td><td>42</td><td>0</td><td>0</td><td>Added</td></tr><tr><td>F_TAG</td><td rowspan="3">Physical address bits (B-E) extracted from the address.</td><td></td><td></td><td></td><td></td><td></td><td>16</td><td>12</td><td>21</td><td>0</td><td>0</td><td>Added</td></tr><tr><td>F_TAG</td><td></td><td></td><td></td><td></td><td></td><td>16</td><td>18</td><td>36</td><td>0</td><td>0</td><td>Added</td></tr><tr><td>Sampler</td><td></td><td colspan="10"></td></tr><tr><td>Number</td><td>Number of samplers</td><td colspan="10">80</td><td>Changed</td></tr><tr><td>Associativity</td><td>Associativity of samplers</td><td colspan="5">18</td><td colspan="5">48 12</td><td>Changed</td></tr><tr><td>Access Type</td><td colspan="10">Logic to differentiate access types Prefetch + Writeback</td><td>Changed</td></tr><tr><td colspan="10">Placement Vector Thresholds</td><td></td></tr><tr><td>plv[i][ji]</td><td colspan="10">Placement Level Vector values -15, 0, 35, 12, 44, 15</td><td>Changed</td></tr><tr><td colspan="10">Implementation Changes and Confidence Values</td></tr><tr><td>Confidence</td><td colspan="10">Confidence values (dt2, theta, theta_2, pro- 256, 109, 135, 82</td><td>Changed</td></tr><tr><td>Thresholds</td><td colspan="10">motion threshold).</td><td></td></tr><tr><td>Predictor Tables Record Types</td><td>Number of predictor tables.</td><td colspan="10">16 tables</td><td>Changed Changed</td></tr><tr><td>Noise-controlled</td><td>Types of accesses recorded.</td><td colspan="10">Load, RFO, Prefetch, Writeback</td><td>Novelty</td></tr><tr><td>XOR</td><td>Applies mask on PC-bit slice</td><td>Does not exist</td><td colspan="10"></td><td>duced</td></tr></table>

DESIGN COMPARISON OF CACHEFORGE VS. MULTIPERSPECTIVE.

![](images/b85d50f58c15b5872c1f7e53c5d8220355a9bd9385f6200e8091e4f6d33e9514.jpg)  
Fig. 12. Seed analysis across various replacement policies.

Impact of LLM Variations. We evaluated CACHEFORGE’s performance across multiple LLMs. Figure 14 shows that

![](images/0035d542e863f120b3930e69157568f21fb0895b2d1ff832de615d5ac8b50b28.jpg)  
Fig. 13. Impact of number of seeds to cross over

Gemini 2.5 Pro and o4-mini yield the most consistent top-tier policies (48–52% hit rate), highlighting the growing benefit of scaling and specialization in LLM-driven microarchitectural co-design.

Memory, Exploration, & Exploitation. CACHEFORGE begins in exploration, where the LLM proposes novel policy structures to surpass state-of-the-art baselines. With memory enabled, prior policies and their embeddings are retrieved to guide this search, helping the model reuse successful ideas and avoid redundant paths. After a fixed number of unsuccessful trials, the system switches to exploitation, refining the best policy found so far; here, lineage memory captures its evolution and supports targeted improvements. Memory has its strongest impact during exploration, enabling broader and more non-redundant idea generation, while exploitation behaves similarly with or without memory.

![](images/34c12d86d9d3ba699a3c488047c56b0a49a7a4c211a4d120eb74ff899e7600ac.jpg)  
Fig. 14. Performance across various LLM models

![](images/97e76f7376f5c9581f6592d28989c01b94853b3f108be1023b0df842b4898a62.jpg)  
Fig. 15. Impact of memory-augmented design.

Figure 15 compares results with and without memory across IPC, memory raises the mean hit rate from 42.16% to 48.27% and increases mean IPC from 0.15 to 0.42. Speedup improves to 2.80×, and even normalized by storage, speedup-per-KB increases by 41.36%. Overall, memory-guided exploration produces higher-quality, more efficient, and more stable replacement policies across the design space.

Temperature. We further sweep the sampling temperature τ to explore the trade-off between exploration diversity and convergence speed, finding $\tau = 0 . 7$ yields optimal results across workloads. See Figure 16,

![](images/120b7bff0d604ff40f9496e7239e7f6e101af853dcb908e9402540b59925c17d.jpg)  
Fig. 16. Impact of temperature.

Mutation Sensitivity Analysis. We performed a three-part ablation to assess how surrounding conditions affect mutationdriven policy generation. Figure 17 compares hit-rate distributions for three mutation strategies. The baseline dynamic mutation rate yields a median hit rate of 41% but shows high variability, indicating unstable policy quality. Adding a candidate pool raises the median to 44%, narrows the interquartile range, and shifts the upper quartile upward, producing more consistent and generally stronger policies. Incorporating dynamic temperature alongside the candidate pool maintains similar variability but lifts the minimum hit rate while slightly lowering the median to 43% and reducing the concentration of top-end policies. Overall, the candidate pool is the most effective enhancement, improving both median quality and stability of generated policies.

![](images/956d36bb121ff9d5c252a788da0e8975054156d85429859dc761abaaae960cfb.jpg)  
Fig. 17. Impact of dynamic mutation rate and temperature.

Impact of Prompt and Non-Expert Users. To evaluate whether CACHEFORGE requires microarchitectural expertise to produce high-quality replacement policies, we conducted a user study with 60 students, 34 undergraduates and 26 graduate students, none of whom had prior exposure to cache design. Each participant interacted with the same LLM-guided workflow used throughout this paper, but was restricted to modifying only the prompt. The goal of this experiment was to isolate how much of CACHEFORGE’s performance depends on the skill of the human operator versus the autonomous generator–evaluator loop. Undergraduates achieved a median hit rate of 49.75% , with 88.2% of participants generating policies strictly better than LRU. Graduate students performed even more uniformly, obtaining a median hit rate of 48.81%, with every participant (26/26) surpassing LRU (Figure 18). These results demonstrate that CACHEFORGE’s performance stems from its iterative generation-evaluation and causal feedback loop, eliminating the need for microarchitectural expertise. Furthermore, the framework provides significant educational value by illustrating how specific components impact replacement policies and CPU performance.

## VIII. CONCLUSION

Prior work in cache optimization is limited by diminishing results, and is stuck within human-defined search spaces. This paper is the first to bridge this divide, using an LLM as a generative engine while simultaneously using a hardware simulator as a governance loop to ground the LLM’s reasoning and enforce practical constraints such as hardware overhead.

![](images/65047506a5ae6d1e66e0e57aa193fc8298dab20fe517280e20ca985a55266b1a.jpg)  
Fig. 18. Cache hit rate distributions for the undergraduate and graduate cohorts. and the LRU baseline.

## REFERENCES

[1] E. Agarwal, R. Magazine, J. Singh, V. Dani, T. Ganu, and A. Nambi, “PromptWizard: Optimizing prompts via task-aware, feedback-driven self-evolution,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 19 974–20 003. [Online]. Available: https://aclanthology.org/2025.findings-acl.1025/

[2] L. A. Belady, “A study of replacement algorithms for a virtual-storage computer,” IBM Systems Journal, vol. 5, no. 2, pp. 78–101, 1966.

[3] R. Bera, K. Kanellopoulos, S. Balachandran, D. Novo, A. Olgun, M. Sadrosadati, and O. Mutlu, “Hermes: Accelerating long-latency load requests via perceptron-based off-chip load prediction,” in Proceedings of the 55th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE/ACM, 2022, pp. 1–18. [Online]. Available: https://doi.org/10.1109/MICRO56248.2022.00015

[4] G. Chen, S. Dong, Y. Shu, G. Zhang, J. Sesay, B. F. Karlsson, J. Fu, and Y. Shi, “Autoagents: A framework for automatic agent generation,” 2024. [Online]. Available: https://arxiv.org/abs/2309.17288

[5] J. Diaz Maag, T. Arnal, P. Iba´nez, J. Llaber˜ ´ıa, and V. Vinals-Yufera,˜ “Red: A reuse detector for content selection in exclusive shared lastlevel caches,” Journal of Parallel and Distributed Computing, vol. 125, 11 2018.

[6] N. Gober, G. Chacon, L. Wang, P. V. Gratz, D. A. Jimenez, E. Teran, S. Pugsley, and J. Kim, “The championship simulator: Architectural simulation for education and competition,” 2022. [Online]. Available: https://arxiv.org/abs/2210.14324

[7] A. Jain and C. Lin, “Back to the future: Leveraging belady’s algorithm for improved cache replacement,” in Proceedings of the 43rd International Symposium on Computer Architecture (ISCA). Seoul, Republic of Korea: IEEE Press, 2016, pp. 78–89. [Online]. Available: https://doi.org/10.1109/ISCA.2016.17

[8] A. Jaleel, K. B. Theobald, S. C. Steely, and J. Emer, “High performance cache replacement using re-reference interval prediction (rrip),” SIGARCH Comput. Archit. News, vol. 38, no. 3, p. 60–71, Jun. 2010. [Online]. Available: https://doi.org/10.1145/1816038.1815971

[9] D. A. Jimenez, “Insertion and promotion for tree-based pseudolru´ last-level caches,” in Proceedings of the 46th Annual IEEE/ACM International Symposium on Microarchitecture, ser. MICRO-46. New York, NY, USA: Association for Computing Machinery, 2013, p. 284–296. [Online]. Available: https://doi.org/10.1145/2540708.2540733

[10] D. A. Jimenez and E. Teran, “Multiperspective reuse prediction,” in´ Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture, ser. MICRO-50 ’17. New York, NY, USA: Association for Computing Machinery, 2017, p. 436–448. [Online]. Available: https://doi.org/10.1145/3123939.3123942

[11] A. T. Kalai, O. Nachum, S. S. Vempala, and E. Zhang, “Why language models hallucinate,” ArXiv, vol. abs/2509.04664, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:281194798

[12] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large language models are zero-shot reasoners,” in Proceedings of the 36th International Conference on Neural Information Processing Systems, ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2022.

[13] E. Z. Liu, M. Hashemi, K. Swersky, P. Ranganathan, and J. Ahn, “An imitation learning approach for cache replacement,” in Proceedings of the 37th International Conference on Machine Learning, ser. ICML’20. JMLR.org, 2020. [Online]. Available: https://dl.acm.org/doi/10.1145/ 3352460.3358319

[14] S. Mirbagher-Ajorpaz, E. Garza, G. Pokam, and D. A. Jimenez, “Chirp:´ Control-flow history reuse prediction,” in 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), 2020, pp. 131– 145.

[15] S. Mostofi, S. Gupta, A. Hassani, K. Tibrewala, E. Teran, P. V. Gratz, and D. A. Jimenez, “Light-weight cache replacement for instruction´ heavy workloads,” in Proceedings of the 52nd Annual International Symposium on Computer Architecture, ser. ISCA ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 1005–1019. [Online]. Available: https://doi.org/10.1145/3695053.3730993

[16] A. Novikov, N. Vu, M. Eisenberger, E. Dupont, P.-S. Huang, A. Z.˜ Wagner, S. Shirobokov, B. Kozlovskii, F. J. R. Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog, “Alphaevolve: A coding agent for scientific and algorithmic discovery,” 2025. [Online]. Available: https://arxiv.org/abs/2506.13131

[17] M. K. Qureshi, A. Jaleel, Y. N. Patt, S. C. Steely, and J. Emer, “Adaptive insertion policies for high performance caching,” SIGARCH Comput. Archit. News, vol. 35, no. 2, p. 381–391, Jun. 2007. [Online]. Available: https://doi.org/10.1145/1273440.1250709

[18] I. Shah, A. Jain, and C. Lin, “Effective mimicry of belady’s min policy,” in 2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2022, pp. 558–572.

[19] Z. Shi, X. Huang, A. Jain, and C. Lin, “Applying deep learning to the cache replacement problem,” in Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, ser. MICRO-52. New York, NY, USA: Association for Computing Machinery, 2019, p. 413–425. [Online]. Available: https://doi.org/10. 1145/3352460.3358319

[20] P. Velickoviˇ c, A. Vitvitskyi, L. Markeeva, B. Ibarz, L. Buesing, M. Ba-´ log, and A. Novikov, “Amplifying human performance in combinatorial competitive programming,” 11 2024.

[21] J. Wang, L. Zhang, R. Panda, and L. K. John, “Less is more : Leveraging belady ’ s algorithm with demand-based learning,” 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:28810368

[22] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2201.11903

[23] C.-J. Wu, A. Jaleel, W. Hasenplaugh, M. Martonosi, S. C. Steely, and J. Emer, “Ship: Signature-based hit predictor for high performance caching,” in 2011 44th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), 2011, pp. 430–441.

[24] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=WE vluYUL-X

[25] V. Young, C. Chou, A. Jaleel, and M. Qureshi, “Ship + + : Enhancing signature-based hit predictor for improved cache performance,” 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:43689204

[26] Y. Zhou, A. I. Muresanu, Z. Han, K. Paster, S. Pitis, H. Chan, and J. Ba, “Large language models are human-level prompt engineers,” in NeurIPS 2022 Foundation Modelsfor Decision Making Workshop, 2022. [Online]. Available: https://openreview.net/forum?id=YdqwNaCLCx