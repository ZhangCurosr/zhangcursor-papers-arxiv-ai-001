# ALBI: LLM-Guided Inductive Logic Programming via Iterative Generate-Test-Constrain Refinement

Aboubaker Seddiq Benterki<sup>1</sup>, Jüri Vain<sup>1</sup>, Gert Kanter<sup>1</sup>, and Leelo Aino Jürgenstein<sup>1</sup>

Tallinn University of Technology (TalTech), Ehitajate tee 5, 19086 Tallinn, Estonia {aboubaker.benterki, leelo.jurgenstein, gert.kanter, juri.vain}@taltech.ee

Abstract. We introduce ALBI (Agent for Learning By Induction), a neuro-symbolic ILP framework that replaces the ASP-based hypothesis generator in the Learning from Failures (LFF) paradigm with a Large Language Model (LLM), while retaining SWI-Prolog as a formal verifier. ALBI implements an iterative generate-test-constrain loop: the LLM proposes candidate hypotheses from structured prompts; Prolog tests each candidate; and diagnostic feedback drives refinement. We further introduce semantic anonymization, a metamorphic testing methodology, to distinguish structural symbolic induction from surface-level pattern matching on predicate names. Experiments on 30 standard ILP benchmarks show that ALBI achieves 100% success across three frontier LLMs (Gemini 3.1 Pro, GPT 5.4 Pro, Claude Opus 4.6), matching Popper while requiring only 1.4–1.7 iterations on average. Under anonymization, all models maintain 100% success with a moderate iteration increase (1.36– 1.67 times), providing evidence of robust symbolic reasoning.

Keywords: Inductive Logic Programming · Large Language Models · Neuro-Symbolic AI · Learning from Failures · Metamorphic Testing

## 1 Introduction

Inductive Logic Programming (ILP) addresses the fundamental challenge of inducing first-order logic theories from examples and background knowledge [20]. Unlike statistical learning methods that produce opaque models, ILP yields human-readable, verifiable logic programs making it particularly suited for domains requiring explainability and formal guarantees. The Learning from Failures (LFF) paradigm [7], implemented in the Popper system, decomposes ILP into three stages: generate a candidate hypothesis, test it against examples, and constrain the search space based on failures. This approach has achieved strong results across diverse domains, but its reliance on ASP solvers for hypothesis generation introduces a fundamental bottleneck: the combinatorial explosion of the search space as background knowledge complexity grows [4].

Recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation and logical reasoning [26]. Gandarela et al. [23] provide the first systematic analysis of LLM capabilities in logic theory induction, revealing that while LLMs show competitive performance on simpler theories, they struggle with long predicate chains and formal consistency. These are precisely the weaknesses that a symbolic verifier can detect and correct, suggesting a natural division of labor: leveraging the LLM for hypothesis generation and the formal logic engine for verification and constraint enforcement.

We introduce ALBI (Agent for Learning By Induction), a framework that operationalizes this insight within the LFF paradigm.<sup>1</sup> ALBI replaces the ASP generator with an LLM, augmented by: (i) a structured prompt architecture incorporating background knowledge, examples, bias constraints, and accumulated failure feedback; (ii) beam search over multiple candidate hypotheses with MDL-based selection [24], mitigating mode collapse in LLM generation [29]; (iii) an incremental locked clause mechanism for learning multi-clause theories; and (iv) a divide-and-conquer optimization phase guided by the Minimum Description Length principle.

A critical concern with LLM-based ILP is whether the model performs structural symbolic induction or merely exploits semantic cues in predicate names. We address this through semantic anonymization [1,2], a metamorphic testing methodology that replaces all meaningful predicate names with abstract identifiers, enabling rigorous evaluation of the system’s symbolic reasoning capability.

Our contributions are:

1. A neuro-symbolic ILP framework (ALBI) that integrates LLMs into the LFF paradigm as heuristic hypothesis generators with formal SWI-Prolog verification.

2. A beam search strategy with structured diagnostic feedback that iteratively constrains the LLM’s generation.

3. A semantic anonymization methodology for metamorphic testing of LLMbased ILP systems.

4. Empirical evaluation on 30 benchmark tasks with semantic and anonymized variants, comparing three frontier LLMs against the ASP-based Popper system.

## 2 Related Work

## 2.1 Inductive Logic Programming

ILP [20] has evolved over three decades [4], from inverse entailment [19] and bottom-up generalization [25] to meta-interpretive learning [21,22] and answer set programming [16,17]. The LFF approach [7] (Popper) learns constraints from test failures; subsequent extensions add divide-and-conquer strategies [3], search pruning [6], rule joining [15], program combination [5], higher-order abstractions [13], and noisy-data handling [14]. Our locked clause mechanism shares the spirit of program combination but uses the LLM as the combination engine.

## 2.2 LLMs for Logical Reasoning and Neuro-Symbolic AI

Chain-of-thought prompting [26] and iterative self-refinement [18] showed that LLMs can perform multi-step reasoning when guided by structured feedback. Gandarela et al. [23] evaluated LLMs on ILP tasks, finding that formal inference feedback can approach ILP system performance on simpler theories but degrades on complex rule dependencies. Our work embeds the LLM within a formal generate-test-constrain loop, providing the symbolic grounding that purely neural approaches lack. More broadly, the integration of neural and symbolic AI is a key direction for robust AI [10,12]. Diferentiable ILP (∂ILP [9]) and neural-abductive approaches [8] learn logic programs via gradient methods but sacrifice search interpretability. ALBI falls within the neural generator + symbolic verifier paradigm, preserving soundness guarantees while leveraging neural semantic heuristics.

## 2.3 Metamorphic Testing

Metamorphic testing [1,2] addresses the oracle problem by defining properties that must hold across related inputs, applied successfully to ML classifiers [28]. We adapt this paradigm: our metamorphic relation is that renaming predicates should not change the system’s ability to find a correct hypothesis, since logical structure is preserved under bijective renaming.

## 3 Background

## 3.1 Inductive Logic Programming

Definition 1 (ILP Task). An ILP task is a tuple $T = \langle B , E ^ { + } , E ^ { - } , \mathcal { H } \rangle$ where B is background knowledge (a definite logic program), $E ^ { + } = \{ e _ { 1 } ^ { + } , \ldots , e _ { m } ^ { + } \}$ and $E ^ { - } ~ = ~ \{ e _ { 1 } ^ { - } , \ldots , e _ { n } ^ { - } \}$ are sets of positive and negative examples, and H is the hypothesis space defined by a language bias. The goal is to find $H \in { \mathcal { H } }$ such that:

– Completeness: $\forall e \in E ^ { + } : B \cup H \models e$

– Consistency: $\forall e \in E ^ { - } : B \cup H \models e$

## 3.2 Learning from Failures

The LFF framework [7] decomposes the ILP task into an iterative loop:

1. Generate: An ASP solver proposes a candidate hypothesis H from H, subject to accumulated constraints.

2. Test: H is evaluated against examples using Prolog. Coverage yields true positives TP, false positives FP, and false negatives FN.

3. Constrain: Based on test results, hypotheses are pruned: Generalisation constraint: if H is too specific $\mathrm { ( F N > 0 , F P = 0 ) }$ , prune hypotheses more specific than H.

– Specialisation constraint: if H is too general $\mathrm { ( F P > 0 ) }$ , prune hypotheses more general than H.

The loop repeats until a complete and consistent H is found or H is exhausted.

## 3.3 Hypothesis Language Bias

The hypothesis space is defined by a bias specification. In Popper, this includes mode declarations head\_pred (the target predicate) and body\_pred (allowed body predicates) with argument directions as well as structural constraints such as max\_vars, max\_body, max\_clauses, and type signatures that tightly bound the ASP search space.

## 4 The ALBI Framework

ALBI replaces the ASP generator in the LFF loop with an LLM while preserving the formal test and constrain stages (Figure 1). The key insight is that the LLM acts as a heuristic generator proposing candidates using semantic understanding and learned patterns while SWI-Prolog [27] serves as the formal oracle, providing rigorous coverage evaluation. Unlike ASP-based generation, which provides completeness guarantees over H, the LLM trades completeness for speed and semantic guidance, navigating the hypothesis space through learned inductive biases rather than exhaustive enumeration. ALBI uses only the mode declarations from the bias, relying on the LLM’s inductive priors and the iterative feedback loop rather than explicit structural bounds to guide hypothesis generation.

## 4.1 Structured Prompt Architecture

The initial prompt establishes the LLM as a deterministic Prolog hypothesis generator and includes five components:

1. System role: Instructions defining the task as ILP hypothesis induction, emphasizing deterministic Prolog syntax and the requirement for sound clauses.

2. Bias constraints: The mode declarations from the task’s bias.pl, specifying the target predicate (head\_pred) and allowed body predicates (body\_pred) with argument directions.

3. Background knowledge: The complete B from bk.pl, included in full.

4. Examples: All positive $( E ^ { + } )$ and negative $\left( E ^ { - } \right)$ examples from exs.pl.

5. Beam request: Generate k = 3 candidates ordered by specificity (simplest, more specific, alternative structure).

Subsequent iterations use a feedback prompt with the failed hypothesis, coverage statistics, sampled counterexamples, and accumulated constraints.

## 4.2 Hypothesis Testing and Scoring

Each candidate hypothesis is loaded into SWI-Prolog alongside B and tested against all examples. For each example e, we check whether $B \cup H \ \ v = \ e$ via Prolog evaluation with a timeout guard.

![](images/ad5431b886ccc3b64aad6a72eb3a8d490885f7ad11c3b3c39402cc60f12820cc.jpg)  
Fig. 1: The ALBI framework architecture.

Definition 2 (Scoring Function). Given hypothesis H with false positives $F P ( H ) = | E ^ { - }$ ∩ covered(H)| and recall $r ( H ) = | E ^ { + } \cap$ covered $( H ) | / | E ^ { + }$ |:

$$
s c o r e ( H ) = \left\{ \begin{array} { l l } { 1 0 0 + \lfloor r ( H ) \times 1 0 0 \rfloor } & { i f F P ( H ) = 0 } \\ { \displaystyle { \frac { F _ { 1 } ( H ) } { 1 + F P ( H ) } } } & { o t h e r w i s e } \end{array} \right.\tag{1}
$$

where $F _ { 1 } ( H )$ is the harmonic mean of precision and recall over the covered examples.

This scoring function prioritizes consistency (zero false positives) over completeness, assigning scores in [100, 200] for consistent hypotheses and penalizing false positives in the denominator. Among tied candidates, we select the shortest hypothesis by clause count, implementing an MDL tie-break [24].

## 4.3 Diagnostic Feedback and Constraint Learning

When no candidate achieves a complete solution, ALBI classifies the failure (TOO\_GENERAL: covers negatives; TOO\_SPECIFIC: misses positives; SYNTAX\_ERROR) and generates a diagnostic prompt with the failed hypothesis, coverage statistics, sampled counterexamples (positives via exponential sampling, negatives uniform), grounded BK facts, and a new beam request. This serves as the LFF constraint analogue: natural-language descriptions guide the LLM away from previously failed hypothesis regions.

## 4.4 Locked Clauses and Partial Cover

When a clause achieves partial coverage (≥ 1 positive, zero false positives), ALBI locks it: covered positives are removed from $E ^ { + }$ , BK is filtered to remaining examples, and a fresh round targets the residuals. This progressively builds multiclause theories, analogous to Combo [5] but using the LLM as the combination engine.

## 4.5 Divide-and-Conquer Optimization

Once a complete hypothesis is found, an optimization phase groups clauses by head predicate and prompts the LLM to produce simpler equivalents (MDL principle [11]), each verified by SWI-Prolog. Three iterations are performed; the simplest correct hypothesis is retained. Algorithm 1 summarises the full-context mode loop.

## 4.6 Semantic Anonymization

To rigorously test whether ALBI performs structural symbolic induction rather than exploiting semantic cues in predicate names (e.g., parent(X,Y) suggests a family relationship), we introduce semantic anonymization:

Definition 3 (Semantic Anonymization). Given an ILP task $T = \langle B , E ^ { + } , E ^ { - } , \mathcal { H } \rangle$ with predicate vocabulary P, the anonymized task $T ^ { \prime } = \langle B ^ { \prime } , E ^ { \prime + } , E ^ { \prime - } , \mathcal { H } ^ { \prime } \rangle$ is obtained by applying a bijective mapping $\sigma : \mathcal { P }  \{ p _ { 1 } , p _ { 2 } , . . . , p _ { | \mathcal { P } | } \}$ to all predicate names in $B , E ^ { + } , E ^ { - }$ , and the bias specification.

Algorithm 1 ALBI Full-Context Mode   
Require: Task $\langle B , E ^ { + } , E ^ { - } , \mathcal { H } \rangle$ , max iterations $T$   
Ensure: Hypothesis $H ^ { * }$ or best partial solution   
1: locked ← ∅; $H ^ { * }  \bot ;$ ; constraints $\gets \emptyset$   
2: prompt ← BuildInitPrompt $( B , E ^ { + } , E ^ { - } , \mathcal { H } )$   
3: for $t = 1$ to T do   
4: $\{ h _ { 1 } , h _ { 2 } , h _ { 3 } \}  \mathrm { L L M ( p r o m p t ) }$   
5: for $i = 1$ to 3 do   
6: $( \mathrm { T P } _ { i } , \mathrm { F N } _ { i } , \mathrm { F P } _ { i } ) \gets \mathrm { T } _ { \mathrm { E S T S W I } } ( B , h _ { i } , E ^ { + } , E ^ { - } )$   
7: $s _ { i } \gets$ score(h<sub>i</sub>) {Eq. 1}   
8: end for   
9: ${ h ^ { * } } \gets$ arg max<sub>i</sub> $s _ { i }$   
10: if $\mathrm { F P } ( h ^ { * } ) = 0$ and $\mathrm { F N } ( h ^ { * } ) = 0$ then   
11: $H ^ { * } \gets$ locked $\cup h ^ { * }$   
12: return Optimize $( H ^ { * } )$   
13: else if $\mathrm { F P } ( h ^ { * } ) = 0$ and $\mathrm { F N } ( h ^ { * } ) \neq \emptyset$ then   
14: locked ← locked ∪ GoodClauses(h<sup>∗</sup>)   
15: $E ^ { + }  E ^ { + } \setminus \mathrm { T P } ( h ^ { * } )$   
16: $B $ FilterB $\check { \zeta } ( B , E ^ { + } )$   
17: prompt ← BuildInitPrompt $\cdot ( B , E ^ { + } , E ^ { - } , \mathcal { H } ,$ constraints)   
18: else   
19: constraints ← constraints ∪ AnalyseFailure $\ d _ { \mathrm { i } } ( h ^ { \ast } )$   
20: prompt ← BuildFeedback(h<sup>∗</sup>, TP, FN, FP)   
21: end if   
22: end for   
23: return locked ∪ $H _ { \mathrm { b e s t } } ^ { \ast }$

The metamorphic relation [2] is: if the system correctly solves $T ,$ then it should also correctly solve $T ^ { \prime }$ , since the logical structure is invariant under bijective predicate renaming. A system whose success rate drops significantly on $T ^ { \prime }$ relative to $T$ relies on surface-level semantic pattern matching rather than structural inductive reasoning. We treat semantic anonymization as an ablation of predicate-name semantics: each task is evaluated twice, once with original names and once with anonymized identifiers. This preserves the logical structure while removing surface semantic cues, allowing us to measure how much ALBI depends on predicate-name meaning.

## 5 Experiments

## 5.1 Experimental Setup

Benchmarks. We evaluate on 30 standard ILP tasks from the Popper benchmark suite<sup>2</sup> [7], spanning five domains: program synthesis (10 tasks), IGGP game playing (9), Zendo spatial reasoning (3), robots (3), and other (5). All tasks use noise-free examples and the standard Popper format (bias.pl, bk.pl, exs.pl). Each task has both a semantic variant (original predicate names) and an anonymized variant (abstract identifiers $p _ { 0 } , p _ { 1 } , \ldots \big )$ . ALBI extracts only mode declarations from bias.pl; Popper’s additional constraints (max\_vars, max\_body, max\_clauses, type restrictions) are not used, making ALBI’s bias strictly weaker.

Table 1: LLM API configurations.
<table><tr><td>Model T</td><td>Reasoning Mode</td><td>Max Out</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.0 Thinking: high</td><td>Default</td></tr><tr><td>Claude Opus 4.6 1.0 Ext. thinking (112K) 128K</td><td></td><td></td></tr><tr><td>GPT 5.4 Pro</td><td>Effort: high</td><td>48K</td></tr></table>

Table 2: Overall performance across 30 semantic benchmark tasks.
<table><tr><td>System Succ. (%) Iter. Time (s) Clauses</td></tr><tr><td>Popper (ASP) 100.0 6.8</td></tr><tr><td>81.8 2.7  $\mathrm { A L B I + G e m i n i 3 . 1 P r o }$  100.0 1.7 333.6 2.8</td></tr><tr><td> $\mathrm { A L B I } + \mathrm { G P T } 5 . 4 \ : \mathrm { P r o }$  100.0 1.4 1631.9 2.8</td></tr><tr><td>ALBI + Claude Opus 4.6</td></tr><tr><td>100.0 1.5 397.7 2.9</td></tr></table>

Models. We evaluate three frontier LLMs (Table 1): Gemini 3.1 Pro (τ=0, thinking level high), Claude Opus 4.6 (τ=1.0, extended thinking with 112K budget), and GPT 5.4 Pro (reasoning efort “high”, 48K output limit). All use structured XML prompts with k=3 beam candidates and failure-specific feedback.

Baseline. We compare against Popper [7] (Clingo-based ASP generation, 1200 s timeout per task). ALBI terminates when a correct hypothesis is found or the maximum iteration count T=20 is reached; beam width k=3 balances diversity against API cost (higher k increases latency with diminishing returns on this benchmark).

Metrics. (i) Success rate: fraction of tasks yielding a complete and consistent hypothesis. (ii) Iterations: number of generate-test cycles. (iii) Wall-clock time: total elapsed time including API latency. (iv) Hypothesis size: clause count and total literal count (MDL proxy).

## 5.2 Main Results

Table 2 summarises the main results. All four systems achieve 100% success across all five domains (program synthesis, IGGP, Zendo, robots, other), solving all 30 tasks. Three diferent frontier LLMs, each using the same ALBI framework, match the completeness of Popper’s exhaustive ASP search on this benchmark suite.

Iteration eficiency. The most striking result is ALBI’s iteration count. All three LLM configurations converge in 1.4–1.7 iterations on average, meaning the vast majority of tasks are solved on the first attempt without feedback. This confirms that frontier LLMs possess strong inductive biases for logic program synthesis when provided with structured prompts containing background knowledge, examples, and bias constraints. Popper, by contrast, iterates through a combinatorial ASP search (mean 6.8 refinement steps across solved tasks), though its per-iteration cost is far lower.

Time–accuracy trade-of. Popper is the fastest system (81.8 s mean), as its search involves no network latency. Among ALBI configurations, Gemini is fastest (333.6 s), followed by Claude (397.7 s), with GPT 5.4 Pro significantly slower (1631.9 s) due to higher API latency and longer generation times. Notably, GPT 5.4 Pro’s higher latency does not reduce its accuracy it achieves 100% success suggesting that the additional computation may reflect more thorough reasoning.

Hypothesis compactness. All systems produce comparably sized hypotheses (2.7– 2.9 clauses on average). However, GPT 5.4 Pro achieves the lowest mean literal count (6.0 vs. 8.2–8.6 for Claude and Gemini), and its optimization phase successfully reduces literals in 12/12 applicable tasks. This suggests GPT 5.4 Pro’s structured reasoning favours more concise logical representations.

The cost structure difers fundamentally between systems: ALBI models cluster at 1–2 iterations with 40–3000 s per task (dominated by API latency), while Popper spreads across up to 40 cheap ASP iterations. ALBI’s per-iteration cost is 10–100 times higher, but it requires 5–20 times fewer iterations. Hypothesis complexity (literal count) is largely independent of iteration count, suggesting that iteration count reflects search dificulty rather than target concept complexity (Figure 3).

## 5.3 Optimization Efectiveness

ALBI’s divide-and-conquer optimization phase (Section 4.5) attempts to simplify the initial hypothesis while preserving coverage. Table 3 reports the optimization rates across models.

Of the 30 tasks, only a subset triggers the D&C optimization phase (tasks where the initial hypothesis has multi-clause structure amenable to simplification); the remaining tasks are solved directly without optimization. GPT 5.4 Pro is uniquely efective: all 12 applicable tasks are successfully simplified (mean reduction of 7.3 literals per task). Inspection of the hypotheses reveals why: during optimization, GPT systematically rewrites list-manipulating clauses using Prolog’s native pattern matching ([H|T]) instead of explicit head/2 and tail/2 predicates, e.g. reducing f(A) :- head(A,B), even(B), tail(A,C), f(C) to f([A|B]) :- even(A), f(B). Across the 10 program synthesis tasks, GPT uses list notation in 9/10 (semantic) and $8 / 1 0$ (anonymized); Gemini uses it in only 1/10, and Claude never adopts it. This notation is unavailable to Popper’s ASP generator and is not declared in the bias, making it a concrete instance of LLMs extending the efective hypothesis language beyond the formal specification. Gemini shows moderate optimization success (4 of 15 applicable), while Claude rarely improves its initial hypotheses (1 of 15), suggesting that Claude’s first-pass solutions are already near-minimal.

Table 3: Optimization efectiveness across models.
<table><tr><td>System</td><td>Applicable Optimized Rate (%) Literals Reduced</td><td></td><td></td></tr><tr><td> $\mathrm { A L B I } + \mathrm { G e m i n i }$ </td><td>15/30</td><td>4/30</td><td>13.3</td></tr><tr><td> $\mathrm { A L B I } + \mathrm { C l a u d e }$ </td><td>15/30</td><td>1/30</td><td>-26 3.3 -7 -87</td></tr><tr><td> $\mathrm { A L B I } + \mathrm { G P T } 5 . 4 \ : \mathrm { P r o }$ </td><td>12/30</td><td>12/30</td><td>40.0</td></tr></table>

Table 4: Efect of semantic anonymization on success rate (%) and average iterations.
<table><tr><td rowspan="3">System</td><td></td><td>Semantic Anonymized</td></tr><tr><td></td><td>Succ. (%) Iter. Succ. (%) Iter.</td></tr><tr><td>100.0 1.7</td><td>100.0 2.6</td></tr><tr><td>ALBI + Gemini  $\mathrm { A L B I + G P T 5 . 4 P r o }$ </td><td>100.0 1.4</td><td>100.0 1.9</td></tr><tr><td> $\mathrm { A L B I } + \mathrm { C l a u d e }$ </td><td>100.0 1.5</td><td>100.0 2.5</td></tr></table>

## 5.4 Semantic Anonymization Results

Table 4 reports the impact of the semantic-anonymization ablation, in which each task is run twice: once with meaningful predicate names and once with anonymized identifiers (Popper excluded its ASP search is invariant to naming). All three models maintain 100% success, providing evidence of structural induction rather than reliance on predicate-name semantics. The iteration multiplier quantifies increased dificulty: Gemini 1.53 times (1.7→2.6), Claude 1.67 times (1.5→2.5), GPT 5.4 Pro 1.36 times (1.4→1.9). GPT 5.4 Pro is the most robust, suggesting its reasoning relies more on structural patterns than surface semantics.

Figure 2 shows that most tasks are solved in a single iteration; the hardest (zendo2, iggp-buttons-next, synthesis-contains) require substantially more iterations under anonymization.

![](images/77ea630d59edf7c491ab13682f42aad6afc8ef2751f780346afd8276426869e3.jpg)  
Fig. 2: Per-task iteration counts (semantic vs. anonymized). Tasks on X-axis; model×condition on Y-axis. Green = single-iteration; darker = harder. Dashed lines separate domains.

## 5.5 Qualitative Analysis

We highlight four illustrative cases.

Case 1: Semantic bias and optimization risk (andersen). Gemini produces the textbook-correct four-rule solution for Andersen’s points-to analysis in one iteration, including the recursive store rule that Popper cannot distinguish from overgeneral alternatives (zero negative examples). Popper’s solution exploits coincidental BK patterns and would fail on larger programs. However, GPT 5.4 Pro’s D&C optimization incorrectly simplifies a correct rule to pass the test set, revealing that literal-count minimization can damage hypotheses when negatives are insuficient.

Case 2: Feedback-driven recovery (1d\_flip). Popper requires 1186 s and 10 refinements; all ALBI models converge in 1–2 iterations, demonstrating that LLM heuristics bypass combinatorial explosion on structurally complex tasks.

Case 3: Constant reasoning under anonymization (iggp-attrition-next-score). With semantic names, Gemini solves this in one iteration (3 rules, 8 literals) by reasoning about action constants. Under anonymization, 4 iterations and 18 literals are needed showing that constant semantics accelerate convergence but are not required for correctness.

Case 4: Built-in primitives (zendo2). ALBI uses inequality (\=) not declared in the bias to produce a 2-rule, 12-literal solution in 4 iterations; Popper requires 26 iterations without access to this primitive. Under anonymization, 11 iterations and 7 rules are needed, illustrating that LLM knowledge of Prolog idioms acts as an implicit bias extension.

![](images/e592ba0c85cd645158d64485eb4f56cbe203eacb35c891b68c95f20155ec2ece.jpg)

![](images/852703493297189a85877239a0e736ee2ae02a29017c775f5f850e5970530ab8.jpg)  
Fig. 3: Wall-clock time and literal count vs. iteration count per system.

## 6 Discussion

ALBI vs. Popper. ALBI’s near-first-shot convergence (1.4–1.7 iterations) demonstrates that LLMs can shortcut the combinatorial search that ASP solvers traverse exhaustively. Crucially, ALBI achieves this with a strictly weaker bias: only mode declarations are provided, whereas Popper additionally relies on structural constraints (max\_vars, max\_body, max\_clauses) and type restrictions. The LLM’s learned inductive priors, combined with iterative feedback, substitute for hand-crafted hypothesis space bounds. Additionally, the LLM can employ builtin Prolog primitives (inequality, negation-as-failure) not declared in the bias, effectively extending the hypothesis language (cf. Case 4 in Section 5.5). However, Popper provides completeness guarantees and is 3–20 times faster in wall-clock time; for production settings where latency and cost matter, Popper remains practical for tasks within its search capacity.

Comparison fairness and the semantic gap. LLMs are pretrained on data that may include ILP-related content, giving ALBI an implicit advantage over Popper’s purely algorithmic search. We view this as a feature of the neuro-symbolic integration: leveraging LLM semantic heuristics while retaining formal verification is the core motivation. Our anonymization experiments control for the strongest form of this concern: predicate names accelerate convergence by 1.4– 1.75 times but 100% success is maintained without them, demonstrating genuine structural induction rather than surface-level pattern matching. The iterative feedback mechanism is the key enabler even when the LLM’s initial hypothesis is wrong on anonymized tasks, structured diagnostic feedback (coverage statistics, counterexamples) allows recovery within 2–3 iterations.

Dificulty analysis. Iteration counts reveal meaningful dificulty gradients under anonymization. The hardest tasks are zendo2 (11–17 iterations vs. 1–7 semantic), iggp-buttons-next (up to 14 vs. 7), and synthesis-contains (8 for Gemini vs. 2). No task exceeds the 20-iteration budget, but the 100% ceiling is partly an artifact of evaluating only on Popper’s own benchmark suite. The iteration data also provides implicit component analysis: under semantic conditions 40% of tasks require >1 iteration; under anonymization this rises to 57% (essential feedback loop). Multi-clause tasks succeed via the locked clause mechanism, and anonymization constitutes a controlled ablation of semantic priors (iterations increase 1.36–1.67 times, success preserved). No task exhausts the T=20 iteration budget, confirming that the feedback mechanism is suficient for recovery on all 30 tasks.

Limitations. Key limitations include: (i) no completeness guarantee the LLM may fail on structures it has not learned; (ii) limited reproducibility across API versions; (iii) 3–20 times higher wall-clock cost than Popper due to API latency; (iv) optimization risk D&C can damage correct hypotheses with insuficient negatives; (v) noise-free assumption integrating MDL-based scoring [14] is planned; (vi) the weaker bias trades optimality guarantees for richer hypothesis spaces, and each LLM call is orders of magnitude more expensive than an ASP iteration.

## 7 Conclusion

We presented ALBI, a neuro-symbolic ILP framework that integrates Large Language Models into the Learning from Failures paradigm. ALBI replaces the ASP generator with an LLM-guided generate-test-constrain loop, retaining SWI-Prolog as a formal verifier to combine the semantic heuristics of neural models with the logical rigor of symbolic reasoning. Experiments on 30 standard ILP benchmark tasks demonstrate that ALBI matches Popper’s 100% success rate across all three LLMs while converging in only 1.4–1.7 iterations on average. Our semantic anonymization methodology provides evidence that ALBI performs robust symbolic induction: all models maintain 100% success under anonymization, with only a moderate increase in iterations (1.36–1.67 times).

Future Work. Key extensions include: batch-based scalability for large tasks, LLM-guided predicate invention, noisy-example support via MDL selection, permodel hyperparameter optimization, a cross-model Logic Robustness Index, and evaluation on broader ILP datasets beyond the Popper suite.

Appendix: Prompt Template Summary. The initial prompt (t=1) contains six components: (i) System role “You are an expert ILP solver; synthesize a Prolog hypothesis that entails all positives, rejects all negatives, and follows Occam’s Razor”; (ii) Strategic guidelines identify the target predicate, enforce variable sharing, generalize constants, choose recursion vs. disjunction; (iii) Bias mode declarations from bias.pl (head\_pred/body\_pred with typed directions); (iv) Background knowledge complete B from bk.pl, included verbatim; (v) Examples all E<sup>+</sup>/E<sup>−</sup> from exs.pl; (vi) Beam request generate k=3 candidates (simplest, more specific, alternative), each delimited by –-CANDIDATE–-. The feedback prompt (t > 1) replaces (iv)–(v) with: the previous best candidate with coverage statistics (TP, FP, FN); a failure classification (TOO\_GENERAL, TOO\_SPECIFIC, or SYNTAX\_ERROR); sampled counterexamples with grounded BK facts; analysis hints (compare BK for positives vs. negatives, find discriminating literals); and accumulated constraint history. The system prompt instructs the LLM to produce a one-sentence pattern analysis before the hypothesis, encouraging chain-of-thought reasoning. Common-mistake warnings (e.g., anonymous head variables, disconnected body variables) are included to prevent frequent errors.

## References

1. Chen, T.Y., Cheung, S.C., Yiu, S.M.: Metamorphic testing: A new approach for generating next test cases. Tech. Rep. HKUST-CS98-01, Hong Kong University of Science and Technology (1998)

2. Chen, T.Y., Kuo, F.C., Liu, H., Poon, P.L., Towey, D., Tse, T.H., Zhou, Z.Q.: Metamorphic testing: A review of challenges and opportunities. ACM Computing Surveys 51(1), 1–27 (2018)

3. Cropper, A.: Learning logic programs through divide, constrain, and conquer. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 6446– 6453 (2022)

4. Cropper, A., Dumančić, S., Evans, R., Muggleton, S.H.: Inductive logic programming at 30. Machine Learning 111(1), 147–172 (2022). https://doi.org/10.1007/ s10994-021-06089-1

5. Cropper, A., Hocquette, C.: Learning logic programs by combining programs. In: Proceedings of the 26th European Conference on Artificial Intelligence (ECAI). Frontiers in Artificial Intelligence and Applications, vol. 372, pp. 501–508 (2023). https://doi.org/10.3233/FAIA230309

6. Cropper, A., Hocquette, C.: Learning logic programs by discovering where not to search. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 6316–6323 (2023)

7. Cropper, A., Morel, R.: Learning programs by learning from failures. Machine Learning 110(4), 801–856 (2021). https://doi.org/10.1007/s10994-020-05934-z

8. Dai, W.Z., Muggleton, S.H.: Abductive knowledge induction from raw data. In: Proceedings of the 30th International Joint Conference on Artificial Intelligence (IJCAI). pp. 1845–1851 (2021). https://doi.org/10.24963/ijcai.2021/254

9. Evans, R., Grefenstette, E.: Learning explanatory rules from noisy data. Journal of Artificial Intelligence Research 61, 1–64 (2018). https://doi.org/10.1613/jair.5477

10. d’Avila Garcez, A., Lamb, L.C.: Neurosymbolic AI: The 3rd wave. Artificial Intelligence Review 56, 12387–12406 (2023). https://doi.org/10.1007/ s10462-023-10448-w

11. Grünwald, P.: The Minimum Description Length Principle. MIT Press (2007)

12. Hamilton, K., Nayak, A., Stamber, B., Longo, L., et al.: Neuro-symbolic artificial intelligence: A survey. Neural Computing and Applications (2024). https://doi. org/10.1007/s00521-024-09960-z

13. Hocquette, C., Dumančić, S., Cropper, A.: Learning logic programs by discovering higher-order abstractions. In: Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI). pp. 3421–3429 (2024)

14. Hocquette, C., Niskanen, A., Järvisalo, M., Cropper, A.: Learning MDL logic programs from noisy data. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38 (2024)

15. Hocquette, C., Niskanen, A., Morel, R., Järvisalo, M., Cropper, A.: Learning big logical rules by joining small rules. In: Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI). pp. 3430–3438 (2024). https://doi. org/10.24963/ijcai.2024/380

16. Law, M., Russo, A., Broda, K.: Inductive learning of answer set programs. In: Proceedings of the 14th European Conference on Logics in Artificial Intelligence (JELIA). LNCS, vol. 8761, pp. 311–325. Springer (2014)

17. Law, M., Russo, A., Broda, K.: The ILASP system for inductive learning of answer set programs. The Association for Logic Programming Newsletter (2020)

18. Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegrefe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al.: Self-refine: Iterative refinement with self-feedback. In: Advances in Neural Information Processing Systems. vol. 36 (2023)

19. Muggleton, S.: Inverse entailment and Progol. New Generation Computing 13(3– 4), 245–286 (1995)

20. Muggleton, S., De Raedt, L.: Inductive logic programming: Theory and methods. Journal of Logic Programming 19–20, 629–679 (1994)

21. Muggleton, S.H., Lin, D., Pahlavi, N., Tamaddoni-Nezhad, A.: Meta-interpretive learning: Application to grammatical inference. Machine Learning 94(1), 25–49 (2014)

22. Muggleton, S.H., Lin, D., Tamaddoni-Nezhad, A.: Meta-interpretive learning of higher-order dyadic datalog: Predicate invention revisited. Machine Learning 100(1), 49–73 (2015)

23. ao Pedro Gandarela de Souza, J., Carvalho, D., Freitas, A.: Inductive learning of logical theories with LLMs: An expressivity-graded analysis. In: Proceedings of the 39th AAAI Conference on Artificial Intelligence. vol. 39 (2025)

24. Rissanen, J.: Modeling by shortest data description. Automatica 14(5), 465–471 (1978)

25. Srinivasan, A.: The Aleph manual. University of Oxford (2001)

26. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., Zhou, D.: Chain-of-thought prompting elicits reasoning in large language models. In: Advances in Neural Information Processing Systems. vol. 35, pp. 24824–24837 (2022)

27. Wielemaker, J., Schrijvers, T., Triska, M., Lager, T.: SWI-Prolog. Theory and Practice of Logic Programming 12(1–2), 67–96 (2012)

28. Xie, X., Ho, J.W.K., Murphy, C., Kaiser, G., Xu, B., Chen, T.Y.: Testing and validating machine learning classifiers by metamorphic testing. Journal of Systems and Software 84(4), 544–558 (2011). https://doi.org/10.1016/j.jss.2010.11.920

29. Zhang, J., Yu, S., Chong, D., Sicilia, A., Tomz, M., Manning, C.D., Shi, W.: Verbalized sampling: How to mitigate mode collapse and unlock LLM diversity. In: International Conference on Learning Representations (ICLR) (2026)