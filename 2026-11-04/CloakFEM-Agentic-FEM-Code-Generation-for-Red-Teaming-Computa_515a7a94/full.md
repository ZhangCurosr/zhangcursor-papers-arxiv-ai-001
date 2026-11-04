# CloakFEM: Agentic FEM Code Generation for Red-Teaming Computational Resource Exhaustion

Li Zheng iD Simin Chen iD

STAR Lab, George Mason University lizheng4@illinois.edu schen68@gmu.edu

## ABSTRACT

Finite element method (FEM) solvers are foundational to scientific and engineering simulation, yet their susceptibility to efficiency attacks remains largely unexplored. An adversary can craft syntactically valid and runnable FEM programs that complete successfully while forcing excessive computational work, enabling computational denial-of-service (DoS) attacks. Finding such inputs is challenging because FEM programs occupy a vast, structured space constrained by numerical, geometric, and constitutive requirements. We introduce CloakFEM, a domain-guided LLM-agent framework for automatically generating runnable adversarial FEM programs that amplify solver computation. CloakFEM combines LLM-guided search with an FEM-specific intermediate representation and a deterministic validity gate that enforces runnable constraints. Across three FEM stacks, 17 problem classes, and 1,068 seeds, CloakFEM discovers adversarial inputs that substantially amplify solver work while remaining valid and solvable. In the most extreme case, solver iterations increase from 43 to 247,367, a 5,752.72× amplification. CloakFEM outperforms five generic black-box baselines and retains an advantage over evolutionary search under a controlled bounded action space. Cross-solver analysis further reveals both problem-intrinsic and solver-specific sources of vulnerability. Our results uncover an underexplored efficiency-attack surface in FEM solvers and show that runnable FEM programs can serve as adversarial inputs.

GitHub: https://github.com/STAR-Computing/CloakFEM

Keywords Code Generation; LLM Agent; Adversarial Attack; FEM Testing

## 1 Introduction

Finite element method (FEM) solvers are essential tools for scientific and engineering simulation, supporting applications such as structural analysis, fluid dynamics, and electromagnetics. Users specify a physical problem through an FEM program, and the solver converts it into numerical systems for approximate solution (Hughes, 2000; Strang and Fix, 2008). As FEM workloads increasingly run in shared and automated computing environments (Stanke et al., 2020), their input-dependent computational behavior introduces a new security concern.

In particular, FEM solvers can be vulnerable to efficiency attacks, in which an adversary crafts a valid FEM program that forces the victim solver to perform excessive computation. Such an input can remain syntactically valid and satisfy numerical and constitutive constraints while inducing expensive solver behavior. Small changes to meshes, physical parameters, boundary conditions, or loads and sources can substantially alter conditioning and convergence, leading to excessive iterations, memory consumption, or runtime (Babuška and Suri, 1992; Saad, 2003; Trefethen and Bau, 1997). In shared clusters or simulation services, adversarial FEM programs can therefore monopolize computational resources, increase operational costs, and delay other users, potentially enabling computational denial-of-service (DoS) attacks (Crosby and Wallach, 2003; Staicu and Pradel, 2018; Davis et al., 2018). Unlike malformed-input attacks, these adversarial programs may remain runnable and complete successfully, making them difficult to distinguish from legitimate but computationally expensive workloads.

This raises a fundamental question about the security of FEM solvers against efficiency attacks:

Can an adversary systematically generate syntactically valid and runnable FEM programs that induce computational resource exhaustion in FEM solvers?

Existing attack techniques are poorly matched to this threat model for three reasons: (1) heterogeneous attack surfaces: FEM solvers use different input languages, APIs, data structures, and configuration formats, making it difficult to define a unified attack space across solver stacks; (2) compileable and runnable constraints: adversarial changes to geometry, meshes, materials, boundary conditions, or loads must preserve syntactic validity and satisfy runnable and numerical constraints, while arbitrary modifications can easily yield invalid or meaningless programs; and (3) limited search guidance under black-box access: unlike attacks on differentiable neural networks, FEM solvers provide no gradients or fine-grained optimization signals. The attacker observes only execution outcomes over a vast, structured program space, making it difficult to identify inputs that induce poor conditioning, slow convergence, or other expensive solver behavior.

To address these challenges, our key insight is to combine solver-independent abstraction, FEM domain knowledge, and LLM-guided black-box search. Based on this insight, we introduce CloakFEM, a domain-guided LLM-agent framework for generating runnable adversarial FEM programs that amplify victim-side computation. First, CloakFEM maps heterogeneous native FEM specifications into a solver-independent intermediate representation (IR) that exposes editable materials, boundary conditions, loads and sources, geometry, and mesh perturbations while protecting solver configurations. Second, it incorporates FEM domain knowledge and deterministic compileablity and runnablity checks to constrain modifications and ensure valid executable programs. Third, an LLM agent reasons over these semantic components and iteratively refines candidate programs using solver execution feedback, steering the search toward increasingly expensive behavior without requiring gradient information.

We evaluate CloakFEM across three FEM stacks, 17 problem classes, and 1,068 configurations against five generic black-box baselines under the same solver-query budget. CloakFEM finds runnable adversarial FEM programs that increase solver work by up to 5,752.72×, from 43 to 247,367 iterations. It outperforms generic black-box methods in the unrestricted setting and retains a smaller advantage over evolutionary search under a matched bounded action space. Cross-solver replay further shows that amplification can stem from both problem-intrinsic numerical difficulty and implementation- or discretization-specific sensitivity. Overall, these results expose an underexplored efficiency-attack surface in FEM software.

Our contributions are summarized as follows:

• Problem Novelty. We identify and systematically study efficiency attacks against FEM solvers, in which syntactically valid and runnable FEM programs induce excessive victim-side computation and potentially enable computational DoS.

• Technique Novelty. We propose CloakFEM, a domain-knowledge-guided LLM-agent framework that combines a solver-independent FEM IR, validity-aware adversarial transformations, and feedback-guided black-box search to generate computationally expensive FEM programs across heterogeneous solver stacks.

![](images/9fedc8d06843193d30522e7fd3771d1029ed971005bb597fe18182a6b9e733e0.jpg)  
Figure 1. Finite-element workflow and representative applications. Images are adapted from User A1 (2010); Howard (2008); SimScale GmbH (2014); Feel++ Project.

• Empirical Evaluation. Across three FEM stacks, 17 problem classes, and 1,068 configurations, CloakFEM exposes substantial computational amplification, reaching up to 5,752.72× solver iterations, and outperforms generic black-box search methods under both unrestricted and controlled comparison settings.

To the best of our knowledge, this is the first large-scale study of LLM-agent-guided efficiency attacks against FEM solvers through runnable adversarial FEM program generation.

## 2 Background & Related Work

## 2.1 Finite Element Solvers and Their Numerical Cost

FEM maps geometry, materials, boundary conditions, and loads through meshing and weakform assembly to a sparse system $A ( x ) u = f ( x )$ and an engineering prediction (Hughes, 2000; Strang and Fix, 2008). Figure 1 locates our attack surface in this workflow. Physical inputs can amplify solver cost between specification and prediction even at similar mesh sizes: direct cost depends on fill, whereas iterative cost depends on the spectrum and preconditioner (George and Liu, 1981; Davis, 2006; Saad, 2003; Trefethen and Bau, 1997). Near incompressibility, coefficient contrast, and mesh distortion can pass recognized checks yet remain numerically difficult (Babuška and Suri, 1992); gate acceptance implies neither tractability nor application realism. Appendix L expands the numerical mechanisms and distinguishes well-posedness from conditioning.

## 2.2 Efficiency Degradation and Algorithmic Complexity Attacks

Algorithmic-complexity attacks induce disproportionate work through legitimate inputs, from hash collisions and ReDoS (Crosby and Wallach, 2003; Staicu and Pradel, 2018; Davis et al., 2018) to prolonged learned inference (Shumailov et al., 2021; Hong et al., 2021; Chen et al., 2022a; Feng et al., 2024; Chen et al., 2023, 2022b; Dong et al., 2025; Shapira et al., 2023). Unlike accuracy attacks on scientific surrogates (Adesoji and Chen, 2022), we target deterministic FEM work while solver controls remain protected. Appendix L gives a detailed comparison with classical algorithmic-complexity and learned-inference slowdown attacks.

## 2.3 Validity-Constrained Agentic Search

Code agents use execution feedback, while FunSearch, AlphaEvolve, and related work place language models inside externally evaluated optimization loops (Chen et al., 2025; Jimenez et al., 2024; Yang et al., 2024b; Wang et al., 2025; Romera-Paredes et al., 2024; Novikov et al., 2025; Yang et al., 2024a; Liu et al., 2024). They complement vector-based derivative-free methods such as CMA-ES (Hansen and Ostermeier, 2001). Here the search object is a heterogeneous FEM specification, every evaluation invokes a solver, and checks and accounting remain deterministic. We test whether semantic access improves evaluator-query efficiency over five black-box baselines; Section 3 details the solver-blind agent design, and Appendix L provides a fuller positioning against code agents and LLM-guided optimization.

![](images/34e037af4e09f3cecce5c2fe1bb99609173f851d515b521d67a88a65c6a84bf3.jpg)  
Figure 2. Threat model.

## 3 Approach

## 3.1 Threat Model

Adversarial objective. The attacker acts as an ordinary user of a shared FEM service and submits a valid input designed to take substantially longer to solve than a benign one. As illustrated in Figure 2, such an input could hold a worker until completion or termination, reducing cluster capacity and delaying benign jobs when deployment controls do not contain its cost. Our experiments establish controlled work amplification rather than production-service resource exhaustion.

Capabilities and knowledge. The attacker has black-box access to the normal submission interface. Starting from a valid seed, it may edit exposed materials, conditions, geometry, and supported mesh-node fields, but not solver methods, tolerances, limits, or mesh resolution (Appendix Table 11). It observes past edits, gate outcomes, cost ratios, and service-visible readings such as iterations, wall time, and degrees of freedom. It does not know the query budget and cannot inspect assembled operators, spectra, factorizations, preconditioner state, or hidden solver configuration.

## 3.2 Problem Formulation

Let S be an FEM solver with fixed configuration, x<sub>0</sub> a valid seed, $\mathcal { R } ( x _ { 0 } )$ its supported edit space, and $C ( S , x )$ the cost measure fixed before search. CloakFEM solves

$$
x ^ { \star } = \arg \operatorname* { m a x } _ { x \in \mathcal { R } ( x _ { 0 } ) \cap \mathcal { X } _ { \mathrm { g a t e } } } \rho ( x ) , \rho ( x ) = \frac { C ( S , x ) } { C ( S , x _ { 0 } ) } .\tag{1}
$$

Here, $\rho$ measures resource amplification relative to the seed under the same solver and cost field.   
We use linear or nonlinear iteration count when available and wall time otherwise.

The feasible set requires an edit to be representable in the IR, contain finite values, preserve valid geometry, satisfy recognized constitutive and problem-specific runnable constraints, and complete with a converged solve. These checks ensure that edited seeds remain compileable, runnable inputs while excluding malformed or unfinished jobs from successful attacks. Appendix E gives the gate definition and the elasticity result.

![](images/204f8a2875ad95465f2fe37185ec3237dd7ecd934a3bfc52aeffc2501e7f7f29.jpg)  
Figure 3. CloakFEM Overview.

## 3.3 CloakFEM Overview

Figure 3 summarizes one CloakFEM iteration: (1) IR interface: a native FEM input is mapped to the shared IR, and an archived state is selected as the parent; (2) Strategy: the Strategy Agent chooses physical edit targets, operations, and qualitative directions; (3) Generation: the Generation Agent instantiates these moves as concrete IR patches, while deterministic pre-validation removes invalid candidates; (4) Selection: the Selection Agent compares the surviving patches and commits one for execution; and (5) Trajectory update: deterministic components reconstruct the native input, run the solver, validate the result, and record the outcome in memory and the archive. The agents operate only on the IR and observed history; native write-back, execution, validation, query accounting, and final ranking remain outside the models. After B queries, CloakFEM returns the accepted input with the largest observed cost ratio.

## 3.4 Detailed Design

IR interface. Native FEM formats express the same physical concepts through different syntax. CloakFEM maps each seed to a solver-independent IR containing its physics label, materials, conditions, supported geometry, and mesh-node edits, while omitting solver configuration. The agents exchange patches over this object rather than native files. Each atomic edit $\left( o , p , v \right)$ applies a set, add, or delete operation o to a typed path p, with value v when required; a patch may coordinate several such edits. Typed paths make each target and resulting diff explicit, and source-specific writers reconstruct the native input after pre-execution checks. The same representation is used throughout proposal, validation, trajectory memory, and write-back, giving different FEM backends a common interface. Appendix Table 11 lists the edit surfaces, and Appendix D gives a complete example.

Strategy Agent. From the IR, archive, and prior outcomes, it proposes up to three prioritized moves specifying targets, operations, directions, and optional parents, but no numeric magnitudes. This separates where to search from how far to edit. Multiple moves allow distinct compileable hypotheses to be considered in the same iteration.It uses trajectory history to avoid failed directions and branch from promising archived states.

Generation Agent. It instantiates each move at several magnitudes as concrete, potentially multi-field IR patches. Deterministic checks reject malformed, unchanged, unsupported, or immediately invalid candidates before solver execution; rejection reasons may guide a retry without consuming a solver query. This magnitude ladder balances exploration and exploitation.

Selection Agent. From the surviving candidates and their concrete diffs, it predicts cost ratios and selects one move–magnitude pair for execution. It may override earlier preferences using accumulated evidence, enabling broader exploration while spending one solver query. This focuses solver computation on the candidate judged most promising at that iteration. Predictions allocate queries; solver execution and the gate determine outcomes. Appendix C provides all three agent prompts.

Trajectory memory and archive. A bounded summary of attempts, target-level outcomes, solver feedback, and provenance supports credit assignment and avoids repeated ineffective edits. The archive retains every gate-accepted state, allowing later moves to branch from diverse valid parents. Together, they balance refinement of strong candidates with exploration of alternative search paths. Implementation details appear in Appendix B.

## 4 Evaluation

## 4.1 Experimental Setup

Datasets. We collect 1,746 public FEM specifications from the Elmer regression suite, PDE-Agent-Bench, and NAFEMS (Malinen and Råback, 2013; Hang et al., 2026; NAFEMS, 1990). Baseline replay and edit-surface checks produce a fixed 1,068-seed evaluation cohort containing 618 Elmer, 444 PDE-Agent, and 6 NAFEMS cases. Appendix F reports provenance, representations, physics coverage, target-solver mappings, and exclusions.

Target solvers. Elmer cases run in native ElmerSolver (Malinen and Råback, 2013); PDE-Agent cases use DOLFINx 0.9, PETSc 3.24, and Gmsh 4.15.2 (Baratta et al., 2023; Balay et al., 2019; Geuzaine and Remacle, 2009); and supported NAFEMS cases use scikit-fem 12.0.2 or the DOLFINx scalar bridge (Gustafsson and McBain, 2020).

Baselines. We compare CloakFEM against uniform random search, an evolutionary strategy, CMA-ES (Hansen and Ostermeier, 2001), and the Nevergrad OnePlusOne and CMA implementations (Rapin and Teytaud, 2018). All methods share the manifest, IR, writers, target solvers, compileablity and runnability policy, and nominal B = 40 solver-query cap .

Implementation. The primary RQ1 campaigns use deepseek-flash and gpt-5.6-luna at temperature 0.7, with the same prompt templates, feedback fields, query budget, requested candidate width k = 2, and at most three Strategy moves per iteration. Framework-controlled random choices and stochastic baselines use replicate seed 0, while the model APIs use their standard sampling behavior without an explicit seed. ElmerGrid, individual solver executions, and seed trajectories are capped at 60, 300, and 1,800 seconds, respectively; incomplete trajectories retain their checkpointed best gate-accepted result. Runs are distributed across independent process shards, with numerical-library threads fixed to one, on two AMD EPYC 7352 processors (96 hardware threads) with 503 GB RAM; target solvers do not use GPUs. Each record stores model-request outcomes, input, cache, and output tokens, and end-to-end trajectory wall time covering model, solver, validation, and orchestration. Appendix F gives additional execution and logging details.

Table 1. Count-based amplification occurs across all three datasets.
<table><tr><td rowspan="2"> $\rho ^ { \star }$ </td><td colspan="3">CloakFEM (DeepSeek)</td><td colspan="3">CloakFEM (GPT)</td></tr><tr><td>Elmer</td><td>PDE-Agent</td><td>NAFEMS</td><td>Elmer</td><td>PDE-Agent</td><td>NAFEMS</td></tr><tr><td>Median</td><td>3.01</td><td>1.54</td><td>8.72</td><td>5.00</td><td>1.93</td><td>11.85</td></tr><tr><td>Arithmetic mean</td><td>24.17</td><td>90.30</td><td>17.88</td><td>39.59</td><td>131.34</td><td>22.77</td></tr><tr><td>Geometric mean</td><td>4.79</td><td>3.32</td><td>7.52</td><td>7.24</td><td>4.92</td><td>8.37</td></tr><tr><td>P75</td><td>12.08</td><td>6.38</td><td>17.22</td><td>17.18</td><td>11.03</td><td>36.47</td></tr><tr><td>P90</td><td>42.33</td><td>24.00</td><td>43.36</td><td>97.04</td><td>171.21</td><td>55.30</td></tr><tr><td>Max</td><td>714.29</td><td>5715.53</td><td>67.93</td><td>714.29</td><td>5752.72</td><td>68.24</td></tr><tr><td> $\operatorname* { P r } ( \rho ^ { \star } > 1 . 5 )$ </td><td>68.9%</td><td>52.3%</td><td>83.3%</td><td>79.1%</td><td>61.3%</td><td>66.7%</td></tr><tr><td> $\operatorname* { P r } ( \rho ^ { \star } > 3 )$ </td><td>50.2%</td><td>32.0%</td><td>66.7%</td><td>63.0%</td><td>39.6%</td><td>66.7%</td></tr><tr><td>n</td><td>235</td><td>444</td><td>6</td><td>235</td><td>444</td><td>6</td></tr></table>

## 4.2 RQ1: How Much Cost Amplification Can CloakFEM Find?

## 4.2.1 RQ1.1: Prevalence and Severity

Metric and evaluation cohort. For a seed $x _ { 0 }$ and a gate-accepted candidate x, amplification is $\rho ( x ) = C ( S , x ) / C ( S , x _ { 0 } )$ , and $\rho ^ { \star }$ is the largest gate-accepted value observed under the nominal cap of $B = 4 0$ solver executions. Linear or nonlinear iteration counts are preferred as deterministic work metrics; wall time is used only when no usable count is available, with one cost field locked per seed before search. The fixed 1,068-seed cohort therefore contains 685 count-based and 383 wall-time cases. Table 1 reports only the count-based cases, with each seed contributing one best finite ratio; Appendix F gives the metric-assignment rule.

Amplification prevalence. Table 1 and Figure 4(a) show a strongly long-tailed distribution across all three datasets. CloakFEM (DeepSeek) reaches median and P90 amplification of 1.89× and $3 1 . 4 1 \times$ , while CloakFEM (GPT) reaches 2.60× and $1 2 7 . 0 1 \times$ . Both configurations find high-cost inputs in every dataset source, with the strongest upper tail on PDE-Agent.

Upper-tail robustness. The largest GPT case raises linear work from 43 to 247,367 iterations $( 5 , 7 5 2 . 7 2 \times )$ , so the maximum is not driven by a one-iteration baseline. Requiring a baseline count of at least three still yields P90 138.33× and maximum 5,752.72× for GPT. The separate wall-time stratum also exhibits a persistent upper tail under repeated measurement. Appendix A.1 reports the wall-time confirmation, denominator sensitivity, dataset-wise variation, incomplete-run accounting, and representative cases.

## 4.2.2 RQ1.2: Search Effectiveness

Comparison with generic search. Under the common 40-query cap on the matched count cohort, CloakFEM (GPT) reaches median amplification 2.62×, P90 156.0×, and a 67.9% rate above 1.5×; evolutionary search reaches 1.14×, 6.47×, and 29.9%, respectively. CloakFEM also reaches $\rho > 2$ after a median of four queries, compared with twelve for evolutionary search. Appendix Table 5 reports all seven methods, the separate wall-time stratum, valid-candidate rates, and incomplete trajectories; Appendix Table 3 confirms the ranking at matched 10- and 20-query prefixes.

Strict-relative comparison. Table 2 gives CloakFEM (GPT) and evolutionary search the same bounded action space on all 444 PDE-Agent seeds: nonzero numeric values retain their sign and stay within 0.5–2× their seed magnitude, zeros and structure remain fixed, $\nu \leq 0 . 4 9$ , and cumulative node displacement is at most 0.05h. On the 411 finite pairs, the geometric mean of $\rho _ { \mathrm { C l o a k F E M } } / \rho _ { \mathrm { e v o l u t i o n a r y } }$ is 1.147 (95% CI [1.114, 1.183]), while CloakFEM’s success rate above 1.5× is 9.5 percentage points higher (95% CI [+6.6, +12.7] pp). Both paired tests reject the null with $p < 1 0 ^ { - 9 }$ . Appendix Tables 6 and 7 additionally report three framework seeds for the evolutionary reference and a deterministic non-LLM boundary-prior baseline.

Table 2. CloakFEM retains an advantage under strict-relative edits.
<table><tr><td>Method</td><td>Finite Median</td><td> $\overline { { \mathrm { P 9 0 } ~ > ~ 1 . 5 } }$ </td></tr><tr><td>Evolutionary</td><td>439/444</td><td>1.017 1.67614.8%</td></tr><tr><td>CloakFEM (GPT) 412/444</td><td></td><td>1.038 3.057 25.2%</td></tr></table>

(b) Paired advantage of CloakFEM (GPT), $n = 4 1 1$
<table><tr><td>Outcome</td><td>Effect [95% CI]</td><td>p</td></tr><tr><td>Geometric mean of paired ratios</td><td>1.147 [1.114, 1.183]</td><td> $4 . 1 7 \times 1 0 ^ { - 2 2 }$ </td></tr><tr><td>Success-rate</td><td>+9.5 pp</td><td></td></tr><tr><td>difference</td><td> $[ + 6 . 6 , + 1 2 . 7 ] \mathrm { p p }$ </td><td> $2 . 1 5 \times 1 0 ^ { - 1 0 }$ </td></tr></table>

Random CMA-ES ngCMA CloakFEM (GPT)   
Evolutionary OnePlusOne CloakFEM (DeepSeek)

![](images/5c09ca0453e394dee354a887fc9efab23a31927b6e8d00b0724acad0158389fa.jpg)  
Figure 4. Observed amplification, query-prefix progress, and input similarity.

Similarity and budget efficiency. At post-hoc overall input similarity $S _ { \mathrm { o v e r a l l } } \geq 0 . 9 5 , 3 8 . 3 \%$ of CloakFEM (GPT) winners exceed $\rho ^ { \star } > 1 . 5$ and 27.1% exceed 3×, compared with 25.7% and 13.5% for evolutionary search. Figure 4(b–c) shows that CloakFEM builds its advantage early in the query budget and preserves it across most similarity thresholds. Similarity is descriptive rather than a search constraint; Appendix H defines its structured-IR components, and Appendix A.1 provides the complete search, cost, and similarity tables.

## 4.3 RQ2: Does the Degradation Transfer Across Solvers?

Transfer protocol. We replay each original seed and its exact saved CloakFEM (DeepSeek) winner in two independent FEM implementations without additional search, parameter adjustment, or model calls. The 100 cases span elasticity, Poisson, convection–diffusion, Stokes, and Navier–Stokes. Appendix Table 9 reports median work amplification within each solver using CG, GMRES, MINRES, or cumulative Newton iterations. Transfer requires both independent implementations to exceed 1.5×; magnitude reproduction additionally requires both ratios to remain within a factor of two of the native result.

Problem-intrinsic versus implementation-specific degradation. Elasticity transfers consistently: 42 of 45 attacks reproduce in both canonical implementations and production ElmerSolver, including all 41 native attacks above 2×. Median amplification remains close across the source implementation, scikit-fem, DOLFINx, and Elmer (10.99×, 10.25×, 10.25×, and 10.69×). Every edit that moves Poisson’s ratio toward the incompressible limit reproduces, identifying volumetric locking as a problem-intrinsic mechanism. Poisson shows a complementary pattern: seven of ten attacks reproduce across the two canonical assemblies, while production Elmer retains the strongest effect for a smaller subset, indicating discretization-dependent sensitivity.

![](images/89ba7a740d389b72a82c7b17bf11173ad8e261cd06d72ebc285c9df2dad9954d.jpg)

![](images/9d16041b63d441b5d4a1bcd020bab9403b54d99de4231ae881dc45e079a2d40e.jpg)

![](images/f61debc3a23b169c97499a2140414d9c47dc4fec7360a83d20f2555ce90fd266.jpg)  
Figure 5. Cross-solver transfer varies by physical system.

![](images/0b32963e93c73a47fdf64d844abdd6b19a2f454821912614aa2799be83b1c949.jpg)

![](images/a3de7dc423b34fdc5bed9928fe82fae760ac5ebbef7eea1055c1b20c99e1ec4c.jpg)

![](images/763ae75a3787c4e6dfcfb9f9ca96ff8b4a968a8e69e0709672458175d40fd749.jpg)

![](images/b42d82e3616e6004b571fd31dff9e65feeef8471f5db38a975ec93bfa66d2e54.jpg)

![](images/c5e163989591c4dcbe1af9dafaf678bfe483f3e9ffbb163011bdb99f768ed5f1.jpg)  
Figure 6. Ablations relative to the complete Full (IR) pipeline.

![](images/38a9e9ffd2f6e54cbcf398fd2ed92c588c4b7d619a198bb7ad35c3b2a0c1c625.jpg)

Transfer across operator classes. The result extends beyond symmetric positive-definite systems. Four of eight convection–diffusion attacks preserve native-scale amplification under GMRES, and 17 of 34 Stokes attacks remain above 1.5× under MINRES on both assemblies. All three Navier–Stokes viscosity attacks transfer under independent Taylor–Hood/Newton implementations, with median amplification 9.00× and 9.50× versus 12.33× natively. Figure 5 visualizes the cross-solver agreement; Appendix A.2 provides the complete replay results, and Appendix I details the numerical mechanisms.

## 4.4 RQ3: Which Search Choices Matter?

## 4.4.1 RQ3.1: Component Ablation

Ablation design. We evaluate fresh CloakFEM (GPT) trajectories on PDE-Agent seeds while fixing the IR edit surface, validity gate, unaffected prompts, and 40-query cap. The complete Strategy–Generation–Selection pipeline and all controls request candidates per move except the explicit k = 1 control. The controls remove or replace the FEM primer, solver feedback, candidate width, learned selection, Strategy Agent scheduling, parent selection, or Generation Agent knowledge.

Most influential components. The zero-knowledge Generation control produces the largest decline in Figure 6: geometric-mean amplification falls from 4.91 to 1.24, P90 from 188.13 to 1.46, and $\operatorname* { P r } ( \rho ^ { \star } > 1 0 0 )$ from 12.0% to 1.1%. Removing solver feedback or learned selection also lowers the geometric mean to 3.52 and 2.70 and P90 to 42.09 and 17.66, respectively. The higher valid-candidate rate without learned selection (85.1% versus 74.9%) further shows that validity alone does not identify high-cost candidates. Appendix A.3 reports all arms, coverage, and effect summaries.

## 4.4.2 RQ3.2: Budget Sensitivity

Figure 4(b) traces the fixed 680-seed matched count cohort through the standard budget $B = 4 0 \colon$ CloakFEM separates from generic search early and carries its best gate-accepted candidate forward at each query prefix, complementing the median query-to-2× result in Appendix Table 5. In a separate run of the complete CloakFEM (GPT) pipeline on all 444 PDE-Agent seeds through $B = 1 0 0$ , the fixed-cohort median best-so-far amplification rises from 2.05 at $B = 4 0$ to 2.42 at B = 100, a 17.7% gain, while P90 rises from 125.89 to 150.93. The improvement is strongest before $B = 8 0 ,$ , after which the upper tail plateaus. Appendix Figure 13 and Table 14 provide the complete curves, intermediate budgets, and coverage.

## 4.5 RQ4: What Mitigation Tradeoffs Appear in the Implemented Cells?

## 4.5.1 RQ4.1: Runtime Limits

Figure 7 compares benign and adversarial iteration counts across 171 CG seeds. Their distributions have an overlap coefficient of 0.62, showing that a single iteration threshold has limited discriminative power. At a 1% false-positive rate, the threshold detects 34.5% of adversarial inputs; increasing detection therefore requires rejecting a larger fraction of legitimate expensive jobs, so a single runtime threshold cannot separate adversarial from benign load. Appendix Figure 9 reports the complete detection curve across false-positive allowances.

![](images/e4b2096210097373855765d384151f15fbd0e100ea44349398d2c55673d7ef97.jpg)  
Figure 7. Benign and adversarial iteration distributions overlap substantially.

## 4.5.2 RQ4.2: Solver-Side Defenses

Defense evaluation. The strict-pair matrix applies each solver configuration to all applicable benign–edited pairs and evaluates three properties together: benign runtime overhead, residual edited-to-benign amplification, and paired completion. Across 535 replays, 526 complete as pairs. Appendix Table 10 reports the complete seven-system matrix; no configuration achieves low residual amplification, low benign overhead, and full completion at once, and Appendix Figure 10 visualizes this tradeoff.

Elasticity defenses. On 45 elasticity seeds, Jacobi is effectively unchanged from CG. AMG completes all pairs but costs 5.01× on benign inputs and leaves 2.12× time amplification. GMRES–ILU lowers residual amplification to 1.26× in time at 10.31× benign overhead, while direct LU completes all pairs with a 0.98× edited-to-benign time ratio and 2.00× benign overhead at the tested scale. This clean time behavior is scale-bounded, not general: the direct factors store a median 11× (P90 16.4×) the original nonzeros, and since sparse-direct fill grows superlinearly, the same defense becomes prohibitive as the problem size grows.

Nonlinear mitigation. Viscosity continuation provides the clearest nonlinear result: on 22 Navier–Stokes seeds it reduces the cumulative-Newton ratio from 1.33× to 1.03× with 22/22 paired completion, at 2.38× benign overhead. Appendix A.4 retains the control systems, scale analysis, and complete replay accounting.

## 5 Conclusion

CloakFEM exposes input-induced solver cost as a resource-exhaustion vulnerability surface in FEM software. Its solver-independent IR and agentic search discover gate-accepted inputs while protecting solver controls, with the strongest count-based case rising from 43 to 247,367 iterations and substantial amplification across the long-tailed cohort. Strict-relative and matched-prefix controls support the search advantage, while cross-solver replay separates problem-intrinsic difficulty from implementation-specific sensitivity. Together, these results establish CloakFEM as a vulnerability-oriented red-teaming framework for identifying costly FEM inputs whose deployment risk depends on concrete access controls, quotas, and service guarantees.

## Statement

## AI use statement

Generative-AI assistants were used for English-language editing and for assistance with code, plotting, and debugging. They were not treated as sources of scientific evidence. The authors designed the methodology, selected and curated the data, executed the experiments, verified the reported results, reviewed all generated text and code, and take responsibility for the final manuscript. The language-model calls made by CloakFEM are part of the evaluated research method rather than manuscript assistance; their configurations, prompts, execution feedback, and usage are reported or logged as experimental artifacts.

## Ethics statement

This work studies input-induced computational resource exhaustion in FEM solvers to expose a robustness and security risk that conventional correctness- and crash-oriented testing can overlook, not to facilitate disruption of deployed services. All experiments were executed in a controlled research environment on public FEM benchmarks; no production service, user account, human subject, or personal data was involved. CloakFEM edits exposed FEM inputs but cannot alter protected solver methods, tolerances, stopping limits, or resource controls. We report acceptance checks, cross-solver attribution, limitations on deployment-level vulnerability claims, and solver-side mitigations in Section 4.5. These safeguards and disclosures are intended to support defensive stress testing and resource-governance design.

## Reproducibility statement

Section 3 specifies the threat model, objective, IR, and agent pipeline, while Section 4.1 reports the datasets, solver versions, baselines, budgets, timeouts, and hardware. Appendices B, E, and F provide the exact search policies, acceptance checks, corpus provenance, and exclusions; Appendix K documents record identity, execution boundaries, randomness, and generated artifacts. Framework-controlled sampling and black-box baselines use replicate seed 0, numerical-library threads are fixed to one, and the language-model APIs run at temperature 0.7 without an explicit sampling seed. Consequently, framework decisions are replayable from the logged state, but end-to-end language-model trajectories are not bitwise deterministic. Unless stated otherwise, each seed–method pair has one trajectory. The artifact retains prompts, configurations, immutable seed identifiers, candidate and validity logs, query accounting, model usage, and the scripts and intermediate data used to regenerate the reported analyses.

## References

Abolaji D. Adesoji and Pin-Yu Chen. Evaluating the adversarial robustness for Fourier neural operators. arXiv preprint arXiv:2204.04259, 2022.

Jacob Austin, Augustus Odena, Maxwell Nye, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Ivo Babuška and Manil Suri. On locking and robustness in the finite element method. SIAM Journal on Numerical Analysis, 29(5):1261–1293, 1992.

Satish Balay et al. PETSc users manual. Technical Report ANL-95/11 Revision 3.12, Argonne National Laboratory, 2019.

Igor A. Baratta, Joseph P. Dean, Jørgen S. Dokken, Michal Habera, Jack S. Hale, Chris N. Richardson, Marie E. Rognes, Matthew W. Scroggs, Nathan Sime, and Garth N. Wells. DOLFINx: The next generation FEniCS problem solving environment. Zenodo preprint, 2023. doi: 10.5281/zenodo.10447666.

Javier Bonet and Richard D. Wood. Nonlinear Continuum Mechanics for Finite Element Analysis. Cambridge University Press, 2 edition, 2008.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Simin Chen, Cong Liu, Mirazul Haque, Zihe Song, and Wei Yang. NMTSloth: Understanding and testing efficiency degradation of neural machine translation systems. In ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE), pages 1148–1160, 2022a.

Simin Chen, Zihe Song, Mirazul Haque, Cong Liu, and Wei Yang. NICGSlowDown: Evaluating the efficiency robustness of neural image caption generation models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15365–15374, 2022b.

Simin Chen, Hanlin Chen, Mirazul Haque, Cong Liu, and Wei Yang. The dark side of dynamic routing neural networks: Towards efficiency backdoor injection. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24585–24594. IEEE, 2023.

Simin Chen, Yiming Chen, Zexin Li, Yifan Jiang, Zhongwei Wan, Yixin He, Dezhi Ran, Tianle Gu, Haizhou Li, Tao Xie, et al. Benchmarking large language models under data contamination: A survey from static to dynamic evaluation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 10091–10109, 2025.

Philippe G. Ciarlet. The Finite Element Method for Elliptic Problems, volume 4 of Studies in Mathematics and Its Applications. North-Holland, 1978.

Scott A. Crosby and Dan S. Wallach. Denial of service via algorithmic complexity attacks. In USENIX Security Symposium, 2003.

James C. Davis, Christy A. Coghlan, Francisco Servant, and Dongyoon Lee. The impact of regular expression denial of service (ReDoS) in practice: An empirical study at the ecosystem scale. In ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE), 2018.

Timothy A. Davis. Direct Methods for Sparse Linear Systems. SIAM, 2006.

Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, et al. An Engorgio prompt makes large language model babble on. In International Conference on Learning Representations (ICLR), 2025.

Feel++ Project. Turek–hron cfd2 velocity field. Feel++ Toolboxes Manual, CFD benchmark image. URL https://feelpp.github.io/toolbox/toolboxes/latest/cfd/TurekHron/index.html. Image: TurekHronCFD2\_velocity.png.

Xiaoning Feng, Xiaohong Han, Simin Chen, and Wei Yang. LLMEffiChecker: Understanding and testing efficiency degradation of large language models. ACM Transactions on Software Engineering and Methodology, 33(7), 2024.

Alan George and Joseph W. H. Liu. Computer Solution of Large Sparse Positive Definite Systems. Prentice-Hall, 1981.

Christophe Geuzaine and Jean-François Remacle. Gmsh: A three-dimensional finite element mesh generator with built-in pre- and post-processing facilities. International Journalfor Numerical Methods in Engineering, 79(11):1309–1331, 2009. doi: 10.1002/nme.2579.

Vivette Girault and Pierre-Arnaud Raviart. Finite Element Methods for Navier–Stokes Equations: Theory and Algorithms, volume 5 of Springer Series in Computational Mathematics. Springer, 1986. doi: 10.1007/ 978-3-642-61623-5.

Tom Gustafsson and G. D. McBain. scikit-fem: A Python package for finite element assembly. Journal of Open Source Software, 5(52):2369, 2020. doi: 10.21105/joss.02369.

Zhen Hang, Yushan Yashengjiang, Junhui Li, Huanshuo Dong, Yang Wei, Zhezheng Hao, Jiangtao Ma, Songlin Bai, Haozhong Kai, Xihang Yue, Gangzong Si, Dongming Jiang, Chao Yao, Zhanhua Hu, Jiangqing Zhang, Pengwei Liu, Yaomin Shen, Xingyu Ren, Lei Liu, Zikang Xu, Han Li, Qingsong Yao, Hande Dong, and Hong Wang. PDEAgent-Bench: A multi-metric, multi-library benchmark for PDE solver generation. arXiv preprint arXiv:2605.09636, 2026. doi: 10.48550/arXiv.2605.09636.

Nikolaus Hansen and Andreas Ostermeier. Completely derandomized self-adaptation in evolution strategies. Evolutionary Computation, 9(2):159–195, 2001.

Sanghyun Hong, Yi˘gitcan Kaya, Ionu¸t-Vlad Modoranu, and Tudor Dumitra¸s. A panda? no, it’s a sloth: Slowdown attacks on adaptive multi-exit neural network inference. In International Conference on Learning Representations (ICLR), 2021.

Matt Howard. Crash-test simulation. Wikimedia Commons image, 2008. URL https://commons. wikimedia.org/wiki/File:Crash-test\_simulation\_(3467719123).jpg. CC BY-SA 2.0.

Thomas J. R. Hughes. The Finite Element Method: Linear Static and Dynamic Finite Element Analysis. Dover Publications, 2000.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR), 2024.

Steven G. Krantz and Harold R. Parks. The Implicit Function Theorem: History, Theory, and Applications. Birkhäuser, 2002.

Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. Evolution of heuristics: Towards efficient automatic algorithm design using large language model. In International Conference on Machine Learning (ICML), 2024.

Mika Malinen and Peter Råback. Elmer finite element solver for multiphysics and multiscale problems. In Multiscale Modelling Methods for Applications in Materials Science, volume 19 of IAS Series, pages 101–113. 2013.

NAFEMS. The standard NAFEMS benchmarks. Technical Report TNSB, Rev. 3, National Agency for Finite Element Methods and Standards, 1990.

Alexander Novikov et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

Jérémy Rapin and Olivier Teytaud. Nevergrad: A gradient-free optimization platform. https://github. com/facebookresearch/nevergrad, 2018.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, et al. Mathematical discoveries from program search with large language models. Nature, 625:468–475, 2024.

Yousef Saad. Iterative Methodsfor Sparse Linear Systems. SIAM, 2nd edition, 2003.

Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, and Asaf Shabtai. Phantom sponges: Exploiting non-maximum suppression to attack deep object detectors. In IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 4560–4569, 2023.

Ilia Shumailov, Yiren Zhao, Daniel Bates, Nicolas Papernot, Robert Mullins, and Ross Anderson. Sponge examples: Energy-latency attacks on neural networks. In IEEE European Symposium on Security and Privacy (EuroS&P), pages 212–231, 2021.

SimScale GmbH. Eigenfrequency analysis of a truss bridge. Wikimedia Commons image, 2014. URL https://commons.wikimedia.org/wiki/File:Eigenfrequency\_Analysis\_of\_a\_Truss\_Bridge.jpg. CC BY-SA 4.0.

Cristian-Alexandru Staicu and Michael Pradel. Freezing the web: A study of ReDoS vulnerabilities in JavaScript-based web servers. In USENIX Security Symposium, 2018.

Joachim Stanke, Martin Unterberg, Daniel Trauth, and Thomas Bergs. Development of a hybrid DLT cloud architecture for the automated use of finite element simulation as a service for fine blanking. The International Journal of Advanced Manufacturing Technology, 108(11–12):3717–3724, 2020. doi: 10.1007/ s00170-020-05567-5.

Gilbert Strang and George J. Fix. An Analysis of the Finite Element Method. Wellesley-Cambridge Press, 2nd edition, 2008.

Lloyd N. Trefethen and David Bau. Numerical Linear Algebra. SIAM, 1997.

User A1. Elmer pump heat equation. Wikimedia Commons image, 2010. URL https://commons. wikimedia.org/wiki/File:Elmer-pump-heatequation.png. CC BY-SA 3.0.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, et al. OpenHands: An open platform for AI software developers as generalist agents. In International Conference on Learning Representations (ICLR), 2025.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. In International Conference on Learning Representations (ICLR), 2024a.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), 2024b.

## Appendix Contents

A Complete Evaluation Results. . .15   
A.1 Complete RQ1 Results . . 15   
A.2 Complete RQ2 Results . . 20   
A.3 Complete RQ3 Results . . 21   
A.4 Complete RQ4 Results . . 22   
B Search and IR Implementation Details . . 24   
C Three-Agent Prompts . . . . . . 25   
D Illustrative IR Edit and Native Write-Back . . . 26   
E Acceptance Gate and Well-Posed PDE Subclasses . . 26   
F Dataset and Target-Solver Details . . . 29   
G Representative FEM Case Visualizations . 31   
H Input-Similarity Measurement . . . 32   
I Numerical Mechanisms of Efficiency Degradation . 33   
J Extended-Budget Sensitivity . . . 34   
K Reproducibility and Reporting Details . 35   
L Extended Background and Related Work . . . 35

## A Complete Evaluation Results

## A.1 Complete RQ1 Results

RQ1.1: Prevalence and severity. For a seed x and a gate-accepted candidate x, amplification is $\rho ( x ) = C ( S , x ) / C ( S , x _ { 0 } )$ , and $\rho ^ { \star }$ is the largest gate-accepted value observed under the nominal cap of $B = 4 0$ solver executions. Linear or nonlinear iteration counts are preferred as deterministic work metrics; wall time is used otherwise, with one effective cost field locked per seed before search. This produces 685 count-based and 383 wall-time ratios across the 1,068-seed cohort. Appendix F specifies the deterministic metric-fallback rule. Table 1 and Figure 4(a) reveal a strongly long-tailed count-based distribution across dataset sources and mark the $1 . 5 \times , 3 \times$ , and 10× thresholds. Both models produce a finite baseline-relative result for every included seed. On count-based seeds, the median and 90th percentile are respectively 1.89× and 31.41× for CloakFEM (DeepSeek), and 2.60× and $1 2 7 . 0 1 \times$ for CloakFEM (GPT). The largest GPT case is not a one-iteration denominator: work increases from 43 to 247,367 linear iterations, or 5,752.72×. On wall-time seeds, the corresponding median and 90th percentile are 1.25× and 11.93× for CloakFEM (DeepSeek), and $1 . 4 2 \times$ and $1 4 . 6 0 \times$ for CloakFEM (GPT); GPT’s median is below ngCMA’s $1 . 4 5 \times { }$ , while both CloakFEM variants have substantially larger P90 values than all five baselines. Table 5 reports this metric stratum separately. Each baseline and candidate is timed once within its trajectory, execution order is not randomized, and the selected winner is not independently remeasured. We therefore treat all wall-time maxima and threshold rates as exploratory rather than confirmed runtime effects. Table 1 contains only count-level summaries and never averages iteration counts with wall time; within each stratum, every seed contributes exactly one ratio from its locked metric.

Coverage and incomplete runs. Table 5 reports runs that stop before using all 40 queries, while Figure 4(a) includes their checkpointed lower bounds. Each solver execution is capped at 300 seconds and each seed trajectory at 1,800 seconds. CloakFEM (DeepSeek) completes all 40 queries on 910 seeds and stops early on 158; CloakFEM (GPT) completes 841 and stops early on 227. An incomplete trajectory retains its checkpointed best gate-accepted result, so its reported $\rho ^ { \star }$ is a best-so-far lower bound rather than the outcome of a completed budget. Conversely, a candidate that reaches the 300-second single-solve limit is not counted as a completed result; for a wall-time seed it establishes only the lower bound $\rho \geq 3 0 0 / C ( S , x _ { 0 } )$ . Because high-cost inputs consume the trajectory limit fastest, dropping incomplete runs would preferentially remove difficult seeds. Retaining them avoids that exclusion but does not make realized budgets equal: the primary table compares methods under the same query cap, not the same completed query count, wall-clock search time, or total compute. The matched-prefix analysis below equalizes completed queries; explicit survival analysis remains future work.

Table 3. Matched completed-query prefixes preserve the main ranking.
<table><tr><td></td><td colspan="3"> $\overline { { B = 1 0 \left( n = 5 8 0 \right) } }$ </td><td colspan="3"> $\overline { { B = 2 0 \left( n = 5 0 8 \right) } }$ </td></tr><tr><td>Method</td><td>Median</td><td>P90</td><td> $\overline { { > 1 . 5 } }$ </td><td>Median</td><td>P90</td><td>&gt; 1.5</td></tr><tr><td>Random</td><td>1.000</td><td>1.228</td><td>5.9%</td><td>1.000</td><td>1.250</td><td>6.5%</td></tr><tr><td>Evolutionary</td><td>1.000</td><td>1.920</td><td>12.8%</td><td>1.087</td><td>5.358</td><td>23.8%</td></tr><tr><td>CMA-ES</td><td>1.000</td><td>1.215</td><td>4.7%</td><td>1.000</td><td>1.656</td><td>10.4%</td></tr><tr><td>OnePlusOne</td><td>1.000</td><td>1.119</td><td>4.3%</td><td>1.000</td><td>1.128</td><td>5.7%</td></tr><tr><td>ngCMA</td><td>1.000</td><td>1.307</td><td>7.6%</td><td>1.000</td><td>1.938</td><td>13.4%</td></tr><tr><td>CloakFEM (DeepSeek)</td><td>1.393</td><td>15.121</td><td>44.7%</td><td>2.000</td><td>20.622</td><td>56.1%</td></tr><tr><td>CloakFEM (GPT)</td><td>1.762</td><td>25.396</td><td>56.2%</td><td>2.500</td><td>24.771</td><td>66.1%</td></tr></table>

Table 4. Proposal-side cost is substantial and model dependent.
<table><tr><td>Method</td><td>Calls</td><td>Input tokens</td><td>Output tokens</td><td>Time/seed</td><td>Total time</td></tr><tr><td>CloakFEM (DeepSeek)</td><td>171</td><td>748,918</td><td>80,022</td><td>9.9 min</td><td>238.1 h</td></tr><tr><td>CloakFEM (GPT)</td><td>124</td><td>468,754</td><td>48,494</td><td>13.3 min</td><td>293.6 h</td></tr></table>

Matched completed-query prefixes. We additionally compare all seven methods on countmetric seeds for which every method completed the stated prefix. At 10 completed queries, the paired cohort contains 580 seeds: CloakFEM (GPT) has median $\rho ^ { \star } = 1 . 7 6 2$ versus 1.000 for evolutionary search, with a +43.4 percentage-point difference in Pr $( \rho ^ { \star } > 1 . 5 )$ . At 20 completed queries, the paired cohort contains 508 seeds and the corresponding values are 2.500 versus 1.087 and +42.3 percentage points. The paired geometric-mean CloakFEM-to-evolutionary ratios are 3.009 and 2.614 at 10 and 20 queries, respectively. These fixed-prefix results show that the primary advantage is not created by comparing different realized query counts, although they do not equalize total solver or model time. Because the 10- and 20-query columns use different matched cohorts, their percentiles need not be monotone across the two columns.

Search-side cost. Across the fixed 1,068-seed cohort, CloakFEM (DeepSeek) uses a median 171 successful model requests, 748,918 input tokens, 80,022 output tokens, and 9.9 minutes of end-to-end trajectory time per seed. CloakFEM (GPT) uses 124 requests, 468,754 input tokens, 48,494 output tokens, and 13.3 minutes per seed. Total recorded trajectory time is 238.1 and 293.6 hours, respectively. Per-seed entries in Table 4 are medians, and time includes model, solver, validation, and orchestration. These values expose substantial proposal-side cost omitted by a solver-query-only budget. The served identifiers deepseek-flash and gpt-5.6-luna came through experiment gateways rather than documented public billable SKUs, so applying a public DeepSeek or OpenAI price would not reproduce the experiment’s charge. The logs also do not separate model latency from solver time. We therefore do not claim superiority under equal wall-clock or monetary cost.

Wall-time confirmation. On a fixed, non-random 50-case Elmer confirmation cohort drawn from saved GPT candidates, we independently execute the baseline and winner five times each, randomizing their order within every repeat and taking the ratio of medians. The selection manifest records the cases but not a probability-sampling design, so this experiment measures repeatability for those cases and is not representative of all 383 wall-time seeds. Figure 8 compares the original single search measurement with the ratio of five-run medians. The original median is 1.358; the confirmed median is 1.200 (seed-bootstrap 95% CI [1.051, 1.576]), with P90 7.177. The confirmed fractions exceeding 1.5× and 3× are 40% and 20%, respectively.

This supports a persistent upper tail in the selected cohort but also shows attenuation after independent measurement; the complete wall-time cohort remains exploratory.

Small-denominator sensitivity. Requiring a baseline count of at least three leaves 658 seeds for each model. The P90 and maximum remain 31.80× and 5,715.53× for CloakFEM (DeepSeek), and 138.33× and 5,752.72× for CloakFEM (GPT). The upper tail is therefore not an artifact of one- or two-iteration baselines.

Variation across seeds. Table 1 preserves the sourcewise view because the datasets exercise different solvers and physics distributions. Both models find high-cost cases in every included source. CloakFEM (GPT) improves the median and upper tail most clearly on Elmer and PDE-Agent, while the most extreme cases occur in PDE-Agent. NAFEMS has only six eligible seeds and is therefore evidence of coverage rather than a stable source-level estimate.

![](images/2655d5f09aea8ffc6eb17ec0bf0d2dac9b6af4f9d4ddb04cce966c1b901637b8.jpg)  
Figure 8. Repeated timing attenuates selected wall-time ratios.

Seed-driven ceiling. The Elmer value 714.29× is not unique to either CloakFEM configuration. The same elmer\_MortarPoisson2Dsum\_ref ceiling is reached by six of the seven evaluated methods—random, evolutionary, OnePlusOne, ngCMA, CloakFEM (DeepSeek), and CloakFEM (GPT)—while CMA-ES remains at 1.0× on that seed. This shared maximum reflects an unusually sensitive seed that several proposal policies can discover; it should not be attributed solely to agentic search.

Representative case studies. To make the discovered degradation concrete, we provide a dedicated three-case visualization in Appendix Figure 11. It materializes the benign and adversarial FEM inputs and reports measured solver work for complementary geometry-, coefficient-, and node-induced attacks. A 2% annulus-radius change raises conjugate-gradient (CG) iterations from 95 to 417 (4.39×); changing only the Helmholtz wave number from 12 to 23.94 raises iterations from 775 to 3,897 (5.03×); and moving 58 nodes of a circular Poisson mesh raises iterations from 370 to 1,162 (3.14×). The geometry and coefficient edits satisfy the bounded-edit control evaluated in RQ1.2. The displayed 58-node candidate comes from the unrestricted search and exceeds that control’s per-node displacement bound; on the same seed, the bounded winner combines center and radius edits with one node movement and reaches only 1.04×. These case studies explain concrete mechanisms, while Tables 1 and 5 provide population-level evidence.

Mesh-size versus conditioning amplification. Protected resolution controls do not imply fixed degrees of freedom when an editable geometry is remeshed: for example, the annulus case above changes from 2,099 to 2,512 nodes. We therefore distinguish remeshing-induced size amplification from fixed-topology coefficient or node-perturbation effects. The former remains within the attacker-controlled physical input surface, but its cost increase cannot be attributed solely to conditioning. Headline cases should be interpreted with their recorded mesh size and absolute work rather than as implementation defects inferred from ratios alone.

RQ1.2: Search effectiveness. The comparison includes both CloakFEM configurations and all baselines from Section 4.1 under the same IR, execution path, acceptance-gate policy, manifest, and 40-query cap. It is not fully action-space matched: the LLM pipeline supports coordinated general structured patches, while the black-box baselines mutate encoded numeric leaves, supported nodes, and available material-structure entries. Results use the intersection of seeds with finite values and are reported separately for count and wall-time metrics in Table 5. Median, P90, and threshold rates summarize the best gate-accepted result per seed; Valid is the fraction of completed candidate solves accepted by the gate, $q _ { > 2 }$ is the median first query reaching $\rho > 2$ among successful runs, and Incomplete counts trajectories ending before $B = 4 0$ . Thus, the unrestricted table compares complete search systems rather than isolating the effect of an LLM proposal policy.

Table 5. Search outcomes under the same nominal 40-query cap.
<table><tr><td>Method</td><td>Median</td><td>P90</td><td>&gt; 1.5</td><td>&gt;3</td><td>Valid (%)</td><td> $9 { > } 2$ </td><td>Incomplete</td></tr><tr><td>Count metrics (n = 680)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>1.00</td><td>1.61</td><td>11.6%</td><td>4.1%</td><td>25.8%</td><td>12</td><td>82</td></tr><tr><td>Evolutionary</td><td>1.14</td><td>6.47</td><td>29.9%</td><td>17.2%</td><td>48.2%</td><td>12</td><td>68</td></tr><tr><td>CMA-ES</td><td>1.00</td><td>1.86</td><td>12.5%</td><td>5.1%</td><td>43.9%</td><td>14</td><td>109</td></tr><tr><td>OnePlusOne</td><td>1.00</td><td>1.81</td><td>12.6%</td><td>5.4%</td><td>63.9%</td><td>20</td><td>119</td></tr><tr><td>ngCMA</td><td>1.00</td><td>2.17</td><td>16.3%</td><td>6.8%</td><td>46.6%</td><td>13</td><td>105</td></tr><tr><td>CloakFEM (DeepSeek)</td><td>1.93</td><td>32.2</td><td>58.7%</td><td>38.8%</td><td>81.1%</td><td>6</td><td>118</td></tr><tr><td>CloakFEM (GPT)</td><td>2.62</td><td>156.0</td><td>67.9%</td><td>48.2%</td><td>75.2%</td><td>4</td><td>176</td></tr><tr><td>Wall-time metric (n = 383)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>1.18</td><td>3.17</td><td>25.6%</td><td>10.7%</td><td>40.2%</td><td>11</td><td>6</td></tr><tr><td>Evolutionary</td><td>1.19</td><td>3.01</td><td>24.3%</td><td>10.2%</td><td>59.1%</td><td>14</td><td>8</td></tr><tr><td>CMA-ES</td><td>1.19</td><td>2.81</td><td>23.2%</td><td>9.7%</td><td>58.8%</td><td>11.5</td><td>8</td></tr><tr><td>OnePlusOne</td><td>1.32</td><td>2.17</td><td>31.6%</td><td>7.3%</td><td>78.6%</td><td>6</td><td>9</td></tr><tr><td>ngCMA</td><td>1.45</td><td>2.44</td><td>42.0%</td><td>8.6%</td><td>68.2%</td><td>10</td><td>7</td></tr><tr><td>CloakFEM (DeepSeek)</td><td>1.25</td><td>12.0</td><td>32.9%</td><td>20.4%</td><td>92.2%</td><td>6</td><td>35</td></tr><tr><td>CloakFEM (GPT)</td><td>1.42</td><td>14.8</td><td>46.5%</td><td>26.6%</td><td>87.9%</td><td>4</td><td>46</td></tr></table>

To separate search effectiveness from access to broad edits, Table 2 gives CloakFEM (GPT) and evolutionary search the same bounded action space on all 444 PDE-Agent seeds. We call this control strict-relative: editable nonzero numeric values retain their sign and remain between one-half and twice the seed magnitude; original zeros remain zero; Poisson’s ratio is capped at 0.49; strings, structure, and topology cannot change; and cumulative displacement per node cannot exceed 0.05h, where h is the seed’s characteristic mesh size. Panel (a) reports methodlevel outcomes, while panel (b) reports paired CloakFEM effects on the 411 seeds with finite results from both methods.

Table 2 shows that both strict-relative medians remain close to one, with CloakFEM (GPT) slightly higher (1.038 versus 1.017) and retaining a substantially heavier upper tail and higher success rate. On the 411 seeds finite for both methods, the geometric mean of the paired amplification ratios $\rho _ { \mathrm { C l o a k F E M } } / \rho _ { \mathrm { e v o l u t i o n a r y } }$ is 1.147 (95% CI [1.114, 1.183], $p = 4 . 1 7 \times 1 0 ^ { - 2 2 } )$ and its probability of exceeding 1.5× is 9.5 percentage points higher (95% $\mathrm { C I } \left[ + 6 . 6 , + 1 2 . 7 \right]$ $p = 2 . 1 5 \times 1 0 ^ { - 1 0 } )$ . The seed-bootstrap intervals, Wilcoxon test for amplification, and exact McNemar test for success all exclude the null for this paired control. This narrows the actionspace concern for CloakFEM (GPT) versus evolutionary search on PDE-Agent, but it does not establish action-space-matched superiority over the other four baselines, for DeepSeek, or on Elmer. It also does not estimate within-seed trajectory variance because each method has one run per seed.

A conservative failure-aware sensitivity assigns $\rho = 1$ whenever a method lacks a finite result and retains all 444 seeds. The geometric mean of the paired ratios remains 1.135 (95% CI [1.105, 1.168], $p = 8 . 8 9 \times 1 0 ^ { - 2 0 } )$ , and the success-rate difference remains +8.8 percentage points (95% CI [+6.1, +11.7] pp, exact McNemar $p = 2 . 1 5 \times 1 0 ^ { - 1 0 } )$ . Thus, the strict-relative advantage is not an artifact of excluding CloakFEM’s additional non-finite runs.

Table 6. Strict-relative evolutionary search is stable across framework seeds on the stratified subset.
<table><tr><td>Trajectory</td><td>Finite</td><td>Median</td><td>P90</td><td> $\overline { { > 1 . 5 } }$ </td><td>Completed  $\overline { { B = 4 0 } }$ </td></tr><tr><td>Seed 0</td><td>79</td><td>1.027</td><td>1.721</td><td>19.0%</td><td>53/79</td></tr><tr><td>Seed 1</td><td>79</td><td>1.017</td><td>1.901</td><td>20.3%</td><td>52/79</td></tr><tr><td>Seed 2</td><td>79</td><td>1.027</td><td>1.792</td><td>15.2%</td><td>52/79</td></tr></table>

Table 7. Unrestricted search on the stratified 80-seed manifest.
<table><tr><td>Method</td><td>Finite</td><td>Median</td><td>P90</td><td>&gt; 1.5</td><td>Completed B = 40</td></tr><tr><td>Evolutionary</td><td>79</td><td>1.095</td><td>6.745</td><td>26.6%</td><td>72/79</td></tr><tr><td>Boundary prior (no LLM)</td><td>79</td><td>1.139</td><td>19.379</td><td>39.2%</td><td>46/79</td></tr><tr><td>CloakFEM (GPT)</td><td>80</td><td>2.058</td><td>125.051</td><td>61.3%</td><td>54/80</td></tr></table>

Table 8. Input-space similarity and amplification of selected winners.
<table><tr><td>Method</td><td>Overall</td><td>Surface</td><td>Physical</td><td> $\overline { { \rho ^ { \star } > 1 . 5 \rho ^ { \star } > 3 } }$ </td></tr><tr><td>Random</td><td>0.997</td><td>1.000</td><td>1.000</td><td>11.6% 3.6%</td></tr><tr><td>Evolutionary</td><td>0.992</td><td>1.000</td><td>0.988</td><td>25.7% 13.5%</td></tr><tr><td>CMA-ES</td><td>0.997</td><td>1.000</td><td>1.000</td><td>14.0% 5.8%</td></tr><tr><td>OnePlusOne</td><td>1.000</td><td>1.000</td><td>1.000</td><td>17.5% 5.4%</td></tr><tr><td>ngCMA</td><td>0.994</td><td>1.000</td><td>1.000</td><td>21.5% 5.6%</td></tr><tr><td>CloakFEM (DeepSeek)</td><td>0.970</td><td>1.000</td><td>0.942</td><td>27.8% 17.9%</td></tr><tr><td>CloakFEM (GPT)</td><td>0.969</td><td>1.000</td><td>0.949</td><td>38.3% 27.1%</td></tr></table>

Framework-level repeatability on a stratified subset. We additionally rerun strict-relative evolutionary search with framework seeds $0 , 1 ,$ and 2 on the same physics-stratified 80-seed manifest; 79 seeds are eligible for this non-LLM baseline. Table 6 shows similar medians, upper tails, and threshold rates across the three independent trajectories. This establishes that the strict-relative evolutionary reference is not dominated by one favorable framework seed.

Deterministic boundary-prior baseline. To test whether the unrestricted advantage can be recovered by textbook boundary seeking alone, we evaluate a non-LLM heuristic on the same stratified manifest. It sweeps one numeric IR coordinate at a time over deterministic log-scale factors from $1 0 ^ { - 6 }$ to $1 0 ^ { 6 }$ and explicitly approaches the isotropic Poisson-ratio boundaries; it has no trajectory memory or coordinated multi-field edits. Table 7 compares it with the saved unrestricted runs on the same manifest. The heuristic is substantially stronger than generic evolutionary search in the upper tail, confirming that domain knowledge explains part of the unrestricted gain. CloakFEM nevertheless retains the larger median, P90, and success rate. This comparison isolates a simple domain prior, not the effect of strict-relative edits.

Input similarity of discovered attacks. Table 8 reports median overall, surface, and physical similarity together with the fraction of the full matched cohort jointly satisfying $S _ { \mathrm { o v e r a l l } } \geq 0 . 9 5$ and each stated amplification threshold. Figure 4(c) instead sweeps the similarity threshold from 0.85 to 1.0. Both use the complete 1,063-seed matched cohort as the denominator. The reference value $S _ { \mathrm { o v e r a l l } } = 0 . 9 5$ is descriptive and was not enforced during search. Appendix H defines the structured-IR projection, component scores, and aggregation procedure.

The black-box baselines have the highest median similarity largely because many winners are unchanged or weakly changed and have $\rho ^ { \star }$ near one. Similarity alone is therefore not evidence of attack quality. ${ \mathrm { A t } } S _ { \mathrm { o v e r a l l } } = 0 . 9 5 , 3 8 . 3 \%$ of CloakFEM (GPT) winners jointly exceed $\rho ^ { \star } > 1 . 5 ,$ compared with 27.8% for CloakFEM (DeepSeek) and 25.7% for evolutionary search. For $\rho ^ { \star } > 3 ,$ the corresponding fractions are 27.1%, 17.9%, and 13.5%. Figure 4(c) shows that this advantage persists across most of the displayed similarity range without similarity influencing the search.

Budget efficiency. The query logs yield $\rho ^ { \star } ( b ) = \mathrm { m a x } _ { t \leq b } \rho ( x _ { t } )$ for $b \in \{ 1 , \dots , 4 0 \}$ . Figure 4(b) uses the same fixed 680-seed count cohort as Table $5 ;$ after a trajectory stops, its checkpointed incumbent is carried forward as a lower bound. It therefore complements the single $q _ { > 2 }$ statistic without changing the cohort as the budget increases.

## A.2 Complete RQ2 Results

Transferability. RQ2 tests whether selected expensive inputs expose numerical difficulty beyond the implementation on which they were found. The replay applies the original seed $x _ { 0 }$ and exact saved CloakFEM (DeepSeek) winner $x ^ { \star }$ to independent FEM solver implementations without additional search, parameter adjustment, or LLM calls. Geometry, mesh, materials, boundary conditions, and loads or sources remain fixed within a canonical boundary-value problem. The 100-case cohort contains 45 linear-elasticity, 10 Poisson, 8 convection–diffusion, 34 Stokes, and 3 Navier–Stokes inputs, covering symmetric positive-definite (SPD), nonsymmetric, symmetric-indefinite saddle-point, and nonlinear saddle-point systems. It is a retrospective, mechanism-oriented cohort: cases must have a saved winner representable by the relevant canonical adapter, and some replay scripts order eligible cases by native amplification before applying a limit. It is therefore not a probability sample of the 1,068 seeds, and the highly unequal physics counts—especially 3 Navier–Stokes and 8 convection–diffusion cases—preclude population-level transfer claims. For a source solver S and target solver $S ^ { \prime } { } _ { , }$ , we compute

$$
\rho _ { S } = \frac { C ( S , x ^ { \star } ) } { C ( S , x _ { 0 } ) } , \qquad \rho _ { S ^ { \prime } } = \frac { C ( S ^ { \prime } , x ^ { \star } ) } { C ( S ^ { \prime } , x _ { 0 } ) } .\tag{2}
$$

The two ratios are normalized within solver and never mix costs from different implementations. Work is conjugate-gradient (CG), generalized minimal residual (GMRES), or minimum residual (MINRES) iterations for the corresponding linear operator, and total Newton iterations across viscosity continuation for Navier–Stokes. Threshold transfer requires both independent target solvers to have $\rho > 1 . 5 ;$ magnitude reproduction additionally requires each amplification to remain within a factor of two of the native ratio. Runs reaching the iteration cap are retained as lower bounds and marked separately.

Problem-intrinsic versus implementation-specific degradation. Table 9 reports replay outcomes and median amplification by physics and target solver; elasticity transfers consistently: 42 of 45 attacks reproduce in both the independent canonical implementations and production ElmerSolver. All 41 stronger attacks with native $\rho > 2$ reproduce in Elmer. The three non-reproducing cases make only node perturbations and have weak native amplification; by contrast, every attack that drives Poisson’s ratio toward the incompressible limit reproduces. Median amplification remains close across the source implementation, scikit-fem, DOLFINx, and Elmer $( 1 0 . 9 9 \times , 1 0 . 2 5 \times , 1 0 . 2 5 \times , \mathrm { a n d } 1 0 . 6 9 \times )$ . This agreement identifies near-incompressible volumetric locking as a problem-intrinsic source of computational degradation rather than an artifact of one code path. Poisson presents a different pattern. Seven of ten attacks reproduce between the source DOLFINx implementation and an independently assembled scikit-fem problem, including the strongest variable-coefficient cases. In Elmer’s Heat solver, three of ten remain above the 1.5× transfer threshold, but only two also satisfy the factor-of-two magnitude criterion; the median falls from 2.26× natively to 1.26×. For example, the variable-coefficient stripe case decreases from 9.49× natively and in scikit-fem to 1.29× in Elmer. These attacks therefore expose discretization- or implementation-specific sensitivity rather than a universally difficult PDE instance.

The expanded operator classes show that this distinction is not confined to SPD systems. Four of eight convection–diffusion attacks transfer under GMRES and preserve native-scale amplification. Seventeen of 34 Stokes attacks remain above 1.5× under MINRES on both assemblies, but only one preserves the native magnitude; the original boundary-value problem therefore contributes strongly to the Stokes amplification. All three Navier–Stokes viscosity attacks transfer under independent Taylor–Hood/Newton implementations, with median amplification 9.00× and 9.50× versus 12.33× natively. This 3/3 result provides case-level cross-implementation evidence for low-viscosity nonlinear difficulty, but the stratum is too small for a stable transfer-rate estimate.

Table 9. Transfer strength depends on the physical system.
<table><tr><td>Physics</td><td>Work</td><td>n</td><td>Native scikit-fem</td><td></td><td>DOLFINx</td><td></td><td>Transfer Magnitude</td></tr><tr><td>Elasticity</td><td>CG</td><td>45</td><td>10.99</td><td>10.25</td><td>10.25</td><td>42/45</td><td>42/45</td></tr><tr><td>Poisson</td><td>CG</td><td>10</td><td>2.26</td><td>2.21</td><td>2.26</td><td>7/10</td><td>7/10</td></tr><tr><td>Convection-diffusion GMRES</td><td></td><td>8</td><td>1.80</td><td>1.51</td><td>1.51</td><td>4/8</td><td>4/8</td></tr><tr><td>Stokes</td><td>MINRES</td><td>34</td><td>5.24</td><td>1.53</td><td>1.70</td><td>17/34</td><td>1/34</td></tr><tr><td>Navier-Stokes</td><td>Newton</td><td>3</td><td>12.33</td><td>9.00</td><td>9.50</td><td>3/3</td><td>3/3</td></tr></table>

Figure 5 plots native source amplification against independent target amplification; marker shape identifies the target solver, the diagonal denotes equality, and the shaded band denotes agreement within 1.5×. The denser Stokes replay is separated into Appendix Figure 12 to preserve point-level readability.

These selected replays show that cross-solver transfer is not binary: CloakFEM finds both problem-intrinsic high-cost inputs that survive major implementation changes and solver- or BVP-specific cases whose threshold or magnitude weakens on replay. Detailed root-cause attribution and its evidentiary limits are reported in Appendix I.

## A.3 Complete RQ3 Results

RQ3.1: Component ablation. RQ3 evaluates CloakFEM (GPT) configurations over all 444 PDE-Agent seeds while fixing the IR edit surface, validity gate, unaffected prompts, and maximum budget B = 40. The reference configuration is a fresh execution of the complete Strategy– Generation–Selection pipeline and is not reused from the RQ1 campaign. The primary RQ1 run, the RQ3 reference run, and all RQ3 controls use the same IR representation without the strict-relative constraint profile; the separate Table 2 control adds that profile on top of IR. The complete reference configuration and all controls request k = 4 candidates per move except the one-candidate-per-move control, which sets k = 1 while retaining the Strategy Agent and its limit of three moves per iteration. Thus, every component comparison uses the fresh k = 4 execution of the complete pipeline as its reference, while the k = 1 arm explicitly tests candidate width. The controls remove the FEM primer, solver feedback, multi-candidate generation, learned candidate selection, or Strategy Agent parent selection; replace the Strategy Agent or Generation Agent with seeded zero-knowledge alternatives; or merge strategy and generation into one proposal call. Every arm contains 444 seed records; the reference run has 441 finite results and 249 trajectories completing all 40 queries, while the controls have between 441 and 443 finite results and between 249 and 313 completed trajectories. Figure 6 compares how each control changes the center, upper tail, and extreme-attack frequency of finite $\rho ^ { \star }$

Why the reported GPT medians differ. All four results use IR, but they summarize distinct declared runs or cohorts. Table 1 reports the primary k = 2 run on 444 PDE-Agent count seeds (median 1.93); Table 5 reports that same run on the 680-seed, seven-method matched count cohort spanning all three sources (2.62). The RQ3 complete-pipeline reference is a separate k = 4 run on PDE-Agent (1.74), used only for component ablation. RQ3.2 is another independent k = 2, B = 100 trajectory and reports its fixed-cohort checkpoint lower bound at B = 40 (2.05).

Because API sampling is unseeded, differences between independent runs also include trajectory variation; these values are not interchangeable estimates from one execution.

The zero-knowledge Generation control has the largest observed decline across all five summaries in Figure 6: arithmetic-mean amplification falls from 127.38 to 10.25, geometric-mean amplification from 4.91 to 1.24, P90 from 188.13 to 1.46, and $\operatorname* { P r } ( \rho ^ { \star } > 1 0 0 )$ from 12.0% to 1.1%. Removing solver feedback or the learned Selection Agent also lowers the geometric mean to 3.52 and 2.70 and the P90 to 42.09 and 17.66, respectively; the configuration without the Selection Agent nevertheless produces a higher valid-candidate fraction (85.1% versus 74.9%), showing that validity alone is insufficient without promising candidate selection. Replacing the Strategy Agent with zero-knowledge move scheduling yields a smaller decline across the displayed upper-tail summaries, while removing the generic FEM primer has little effect. The complete pipeline exceeds the $k = 1$ and fixed-parent controls on all five displayed aggregate summaries. Their slightly higher medians (1.87 and 1.94 versus 1.74) show that the observed benefit of multi-candidate generation and adaptive parent selection lies in stronger tail outcomes rather than a higher median; the geometric mean, arithmetic mean, P90, and extreme-attack rates all favor the complete pipeline. The number of Strategy moves was not independently ablated. Because the default system contains no explicit catalog of attack mechanisms, the primer ablation measures general FEM context rather than mechanism-specific prior knowledge. Every arm has one stochastic trajectory per seed, so these comparisons describe the observed cohort rather than repeat-run stability.

RQ3.2: Budget sensitivity. Figure 4(b) reports comparative efficiency through the standard budget $B = 4 0$ . An extended run applies the full CloakFEM (GPT) pipeline to all 444 PDE-Agent seeds through $B = 1 0 0 _ { \mathit { \mathrm { \Pi } } }$ , with a proportionally increased 4,500-second trajectory limit. Using checkpointed best-so-far values for incomplete trajectories, the fixed-cohort median lower bound rises from 2.05 at B = 40 to 2.42 at $B = 1 0 0 _ { \cdot }$ , a 17.7% increase, while the fraction actually reaching those budgets falls from 74.3% to 56.1%. Search therefore continues to improve beyond the standard budget, but with diminishing returns and many incomplete trajectories. Appendix Figure 13 gives the complete curves, coverage, and intermediate-budget statistics.

## A.4 Complete RQ4 Results

RQ4.1: Runtime limits. Figure 7 shows that a fixed limit bounds individual executions but separates benign and adversarial seeds poorly. Across 171 CG seeds, the benign- and adversarial-iteration distributions have an overlap coefficient of 0.62. Restricting false positives to 1% detects only 34.5% of adversarial inputs, leaving 65.5% below the timeout threshold; detecting more adversarial inputs therefore requires rejecting a larger fraction of legitimate expensive seeds.

![](images/4529c34568930c24a27af0293b48d5329e6f793f5f8d030f50d82438d4acbb7c.jpg)

RQ4.2: Solver-side defenses. To avoid repeating the same elasticity conclusion on different cohorts, one strict-pair system-defense matrix serves as the primary

Figure 9. Timeout detection rate as the false-positive allowance varies.

RQ4.2 result. A strict pair consists of a benign seed and the higher-ρ finite replayable input found by CloakFEM (GPT) or evolutionary search under strict-relative constraints. Four of 444 seeds have no finite replayable result. We retain all system classes rather than thresholding on attack success, so several rows are structure-matched controls rather than strong-attack cohorts: median native amplification is 3.06 for elasticity, 1.15 for biharmonic, 1.03 for Navier–Stokes, and 1.00 for the scalar, Stokes, and transient classes. Benign and edited inputs run consecutively in the same worker, and a defense is effective only if it combines low benign overhead, low residual amplification, and reliable completion of both runs.

Table 10. Observed mitigation, benign overhead, and paired coverage.
<table><tr><td>System class</td><td>Defense</td><td>n</td><td>Paired</td><td>Benign overhead</td><td>Work ratio</td><td>Time ratio</td></tr><tr><td>Biharmonic</td><td>Stage-wise CG</td><td>33</td><td>33/33</td><td>1.00</td><td>1.15</td><td>1.00</td></tr><tr><td>Elasticity</td><td>CG</td><td>45</td><td>45/45</td><td>1.00</td><td>2.91</td><td>2.88</td></tr><tr><td>Elasticity</td><td>CG + Jacobi</td><td>45</td><td>45/45</td><td>1.01</td><td>2.88</td><td>2.88</td></tr><tr><td>Elasticity</td><td>GMRES + ILU</td><td>45</td><td>40/45</td><td>10.31</td><td>1.33</td><td>1.26</td></tr><tr><td>Elasticity</td><td>CG + AMG</td><td>45</td><td>45/45</td><td>5.01</td><td>2.74</td><td>2.12</td></tr><tr><td>Elasticity</td><td>Direct LU</td><td>45</td><td>45/45</td><td>2.00</td><td></td><td>0.98</td></tr><tr><td>Navier-Stokes</td><td>Newton + direct</td><td>22</td><td>22/22</td><td>1.00</td><td>1.33</td><td>1.04</td></tr><tr><td>Navier-Stokes</td><td>Viscosity continuation</td><td>22</td><td>22/22</td><td>2.38</td><td>1.03</td><td>1.02</td></tr><tr><td>Poisson</td><td>CG</td><td>79</td><td>79/79</td><td>1.00</td><td>1.00</td><td>0.99</td></tr><tr><td>Stokes</td><td>MINRES</td><td>45</td><td>45/45</td><td>1.00</td><td>1.00</td><td>0.99</td></tr><tr><td>Linear transient</td><td>Backward Euler + direct 72</td><td></td><td>68/72</td><td>1.00</td><td>1.00</td><td>0.99</td></tr><tr><td>Nonlinear transient Newton + direct</td><td></td><td>37</td><td>37/37</td><td>1.00</td><td>1.00</td><td>0.99</td></tr></table>

![](images/2a29f9ce4fdb5aa3e1edaa551ee4969829410c661532f67be5ce53871ee310dc.jpg)  
Figure 10. Solver defenses trade mitigation for overhead and coverage.

Each system–defense cell applies one solver configuration to all applicable strict pairs. Linear work is the iteration count of a Krylov method—CG, GMRES, or MINRES—and nonlinear work is the cumulative number of Newton iterations. Jacobi denotes diagonal preconditioning, incomplete LU (ILU) approximates an LU factorization, algebraic multigrid (AMG) constructs a hierarchy from the discrete operator, and direct LU factorizes the matrix without Krylov iteration. Viscosity continuation solves a sequence of increasingly difficult Navier–Stokes problems ending at the target viscosity, while Backward Euler is the implicit first-order time integrator used for linear transient controls. In Table 10, n is the selected strict-pair cohort, Paired counts cases in which both runs complete, benign overhead is measured against the control configuration for the same system, and Work and Time are median edited-to-benign ratios over completed pairs.

Across 535 executed replays, 526 complete as benign–edited pairs. The nine unpaired cases comprise five elasticity GMRES–ILU failures and four linear-transient failures. Figure 10 exposes the mitigation–overhead–coverage tradeoff. In the 45-seed elasticity cohort, Jacobi is effectively unchanged from CG. AMG completes all pairs but costs 5.01× on benign inputs and leaves 2.74× work and 2.12× time amplification. GMRES–ILU lowers the observed residual amplification but incurs 10.31× benign overhead and fails on five edited inputs. Direct LU completes all pairs with a 0.98× edited-to-benign time ratio and 2.00× benign overhead at the tested scale.

An earlier 42-case high-amplification elasticity pilot is not retained as a separate table or figure because it tests the same five configurations on a threshold-selected cohort. Its unique scale diagnostic is retained: the median number of nonzeros stored in the direct L + U factors is approximately 11× that of the original matrix (P90 16.4×). Sparse-direct memory and factorization cost can grow superlinearly, and neither cohort covers mesh-size, fill-in, or repeated-reassembly attacks. Direct LU is therefore effective only at the evaluated scale, not a universal defense.

Viscosity continuation provides the clearest nonlinear mitigation: on 22 Navier–Stokes seeds it incurs 2.38× benign overhead but reduces the cumulative-Newton ratio from 1.33× under direct Newton at the target viscosity to 1.03×, with 22/22 paired completion. The Poisson, Stokes, biharmonic, and transient rows are structure-matched controls—reference solver configurations appropriate to those systems—rather than comparisons among alternative defenses. The implemented evaluation covers seven numerical-system classes but excludes steady convection– diffusion and Helmholtz, and it uses strict-relative selected pairs rather than the strongest unrestricted winners. RQ4 is therefore an exploratory study of the implemented defenses, not evidence for a generally effective FEM defense.

## B Search and IR Implementation Details

Table 11 lists the semantic edit surfaces shared across native FEM stacks. An atomic edit is $\left( o , p , v \right)$ , where $o \in \{ \mathrm { s e t } , \mathrm { a d d } , \mathrm { d e l e t e } \}$ is an operation, p is a typed structured path, and v is the new value when required. A candidate may combine several atomic edits. Source-specific writers apply the patch, reconstruct the native input, and restore protected blocks verbatim.

Table 11. Solver-independent IR surfaces. Search can edit physical inputs but cannot modify protected solver methods, tolerances, limits, or resolution controls.

<table><tr><td>IR component</td><td>Representative content</td><td>Search access</td></tr><tr><td>Physics</td><td>Equation or problem-class label</td><td>Read only</td></tr><tr><td>Materials</td><td>Region parameters and constitutive values</td><td>Editable</td></tr><tr><td>Conditions</td><td>Boundary conditions, loads, and sources</td><td>Editable</td></tr><tr><td>Geometry</td><td>Supported parametric dimensions</td><td>Editable</td></tr><tr><td>Mesh edits</td><td>Explicit node displacements</td><td>Editable</td></tr><tr><td>Solver state</td><td>Method, tolerances, limits, and resolution</td><td>No access</td></tr></table>

Archive and parent policy. The archive is initialized with $\left( x _ { 0 } , \rho = 1 \right)$ and retains every gateaccepted execution, including non-improving states; rejected executions remain in memory but do not enter the archive. Each Strategy move may select a different archived parent. An omitted or invalid parent resolves to the highest-ρ state. For baselines without learned parent selection, the framework chooses that state with probability 1 − ε and a uniformly random archived state with probability ε. All reported runs use $\varepsilon = 0 . 1 5 ,$ , which was not tuned by method or seed. The models do not observe ε or the remaining budget.

Agent policies and fallbacks. The Strategy Agent proposes up to three prioritized moves. A deterministic sanitizer removes and records leaked magnitudes while preserving structural indices. For each move, the Generation Agent is instructed to produce between one and k patches at different aggressiveness levels; primary searches request k = 2. Over-complete responses are recorded rather than silently truncated, and realized candidate-set size is logged. Generation retries may use explicit rejection reasons; they consume model calls but no solver queries. If several candidates survive, the Selection Agent predicts their cost ratios and chooses one move–magnitude pair. If only one survives, selection is skipped; an invalid selection falls back to Strategy priority, Generation preference, and aggressiveness.

Bounded trajectory memory. For each edited target, a whole-trajectory digest records attempt count, best gate-accepted ratio, valid-but-inert edits, and gate failures, and lists at most 40 untried numeric IR paths. The loop retains eight recent concrete attempts: Strategy and Generation receive the latest six, whereas Selection receives all eight. Current-parent provenance is truncated to five edits and each archived-state summary to four. Strategy sees archived states as identifiers, ratios, short provenance, and solver-visible readings; full parent objects are supplied to Generation only for the at most three selected moves.

## C Three-Agent Prompts

Prompt construction. The implementation adds the current IR, solver-visible readings, search history, accepted-state frontier, and any gate-rejection feedback to each role prompt at runtime. The prompts also state that agents may edit only exposed physical inputs and supported node positions; solver methods, tolerances, limits, and resolution controls remain protected. The primary configuration supplies general FEM terminology but no catalog of attack mechanisms. The following human-readable prompts summarize the instructions given to the three agents.

## Strategy Agent

You are the Strategy Agent in a search for runnable FEM inputs that increase the compute cost of a fixed solver. Review the current FEM problem, its cost relative to the original seed, solver-visible readings, recent attempts, gate rejections, per-target history, and the accepted-state frontier.

Propose up to three promising next moves in priority order. For each move, identify the parent state, the IR fields or physical regions to edit, the operation to apply, the qualitative direction of change, and a brief rationale. Do not choose numerical magnitudes; the Generation Agent will do that. Do not edit solver settings or unsupported fields. Avoid directions that the history shows to be invalid, exhausted, or unable to change cost.

## Generation Agent

You are the Generation Agent. Convert each move selected by the Strategy Agent into concrete, well-typed patches to that move’s parent FEM input. Use only the specified targets, operations, and qualitative direction.

For each move, generate between one and the requested number of candidates at meaningfully different magnitudes, rank them by preference and aggressiveness, and briefly state the expected effect. Use the current IR, solver-visible readings, recent attempts, per-target history, and rejection feedback when choosing values. Keep every value finite and respect the implemented geometry, constitutive, and problem-specific checks. Do not change solver methods, tolerances, limits, or resolution controls.

## Selection Agent

You are the Selection Agent. Several concrete candidate edits have passed pre-execution validation, but none has been run. Review their parent states, diffs, intended directions, aggressiveness, the primary cost metric, solver-visible readings, and the complete search history.

Estimate the likely cost amplification of each candidate and choose exactly one for the next solver execution. Prefer candidates supported by prior improvements or promising unresolved directions, and avoid edits already shown to be inert, invalid, or exhausted. Select from the supplied candidates exactly as given; do not invent or modify an edit.

## D Illustrative IR Edit and Native Write-Back

Consider a two-dimensional linear-elasticity cantilever seed. Its shared IR identifies a rectangular domain, an isotropic material, a fixed condition on the left boundary, and a downward traction on the right boundary. Table 12 shows a candidate containing four coordinated changes rather than a single scalar edit.

Table 12. A coordinated IR edit preserves protected structure.
<table><tr><td>Editable IR leaf</td><td>Seed</td><td>Candidate</td></tr><tr><td>Geometry height</td><td>0.20</td><td>0.16</td></tr><tr><td>Young&#x27;s modulus E</td><td>10.0</td><td>18.0</td></tr><tr><td>Poisson&#x27;s ratio v</td><td>0.30</td><td>0.47</td></tr><tr><td>Right traction  $t _ { y }$ </td><td>-1.0</td><td>-1.5</td></tr></table>

The candidate is represented by four atomic set operations on geometry.geometry\_params.height, materials[0].params.E, materials[0].params.nu, and boundary\_conditions[1].value[1]. Applying the operations to a copy of the seed preserves the original object and makes the complete candidate diff explicit to the Selection Agent.

The deterministic gate first verifies that all four values are finite. It then checks that the edited rectangle has positive dimensions and can be meshed, followed by the constitutive conditions $E > 0$ and $- 1 < \nu < 1 / 2$ . The fixed and traction boundary types are retained, so the edit changes the physical loading without changing the boundary-condition structure. After these input-decidable checks, the source-specific writer materializes the shorter geometry, assigns the edited material and traction, and executes the configured solve. Element-Jacobian and output-finiteness checks are evaluated on the materialized candidate.

For a structured benchmark, geometry fields drive Gmsh materialization while the material and boundary leaves populate the corresponding variational-form inputs. For an Elmer seed, the writer maps whichever of these leaves are exposed by that seed into its editable .sif/.grd fields. In either case, solver method, preconditioner, tolerances, iteration limits, and resolution controls are recovered from the immutable native seed rather than from model output. The example therefore illustrates both coordinated cross-component editing and the separation between the shared physical IR and source-specific protected solver syntax.

## E Acceptance Gate and Well-Posed PDE Subclasses

Compileablity and runnability are meaningful only relative to a modeled physical system, units, boundary conditions, and an application contract. The heterogeneous dataset does not expose enough common information to prove that property for every candidate. We therefore reserve gate-accepted for the implemented predicate. For the set $\mathcal { X } _ { \mathrm { I R } }$ of candidates representable by the supported IR and executable with the seed’s protected solver configuration, define

$$
\begin{array} { r } { \mathcal { X } _ { \mathrm { g a t e } } : = \mathcal { X } _ { \mathrm { I R } } \cap \mathcal { X } _ { \mathrm { f i n } } \cap \mathcal { X } _ { \mathrm { g e o } } \cap \mathcal { X } _ { \mathrm { c o n s t } } \cap \mathcal { X } _ { \mathrm { s o l v e } } . } \end{array}\tag{3}
$$

Equation 3 is the specification of the implementation, not a mathematical characterization of physical admissibility. The following paragraphs state its checks and then give well-posedness results for scalar elliptic and convection–diffusion problems, isotropic linear elasticity, and Stokes flow under explicit hypotheses.

Finite-data and geometry checks. Let each physical element K be obtained from a reference element $\widehat { K }$ through a map $F _ { K } ( \widehat { x } ; x )$ with Jacobian $J _ { K } = \partial F _ { K } / \partial { \widehat { x } } .$ . Finite element assembly requires finite input and derived quantities, while a valid fixed-topology mesh requires every element map to remain nondegenerate and preserve its reference orientation (Ciarlet, 1978; Hughes, 2000; Strang and Fix, 2008). For a node-perturbed candidate, these requirements give

$$
{ \mathcal { X } } _ { \mathrm { f i n } } = \{ x : { \mathrm { a l l ~ e d i t e d ~ i n p u t s ~ a n d ~ r e c o r d e d ~ s o l v e ~ o u t p u t s ~ a r e ~ f i n i t e } } \} ,\tag{4}
$$

$$
\mathcal { X } _ { \mathrm { g e o } } = \{ \boldsymbol { x } : \operatorname* { d e t } J _ { K } ( \boldsymbol { x } ) \operatorname* { d e t } J _ { K } ( \boldsymbol { x } _ { 0 } ) > 0 \quad \forall K \} .\tag{5}
$$

The product in Equation 5 excludes both det $J _ { K } = 0$ and an orientation flip without assuming that every mesh generator uses the same global sign convention. The implementation applies this sign-preservation test to explicit node perturbations with unchanged connectivity. For parametric geometry, where remeshing changes element identities, the corresponding operational condition is successful mesh generation followed by the bridge’s nondegeneracy checks. These conditions make element integration well defined but do not by themselves guarantee a solvable global system.

Constitutive check. For three-dimensional isotropic linear elasticity, Young’s modulus E and Poisson’s ratio ν induce shear and bulk moduli (Bonet and Wood, 2008; Babuška and Suri, 1992)

$$
\mu = \frac E { 2 ( 1 + \nu ) } , \qquad K = \frac E { 3 ( 1 - 2 \nu ) } .\tag{6}
$$

For a symmetric strain $\varepsilon ,$ with $\begin{array} { r } { \varepsilon _ { \mathrm { d e v } } = \varepsilon - \frac { 1 } { 3 } ( \operatorname { t r } \varepsilon ) I , } \end{array}$ , the strain energy decomposes as

$$
W ( \varepsilon ) = \mu \| \varepsilon _ { \mathrm { d e v } } \| ^ { 2 } + \frac { K } { 2 } \big ( \mathrm { t r } \varepsilon \big ) ^ { 2 } .\tag{7}
$$

The gate checks $E > 0$ and $- 1 < \nu < 1 / 2$ . The same pattern is applied by physics and semantic key across the heterogeneous IR: recognized moduli must be positive; viscosity, conductivity, density, thickness, and analogous coefficients must be nonnegative when zero represents a disabled term; yield stress and contact penalty stiffness must be positive; and unrestricted signed quantities such as sources remain subject only to finite-value checks. This semantic classification avoids incorrectly treating a Stokes viscosity named nu as an elastic Poisson ratio. It does not impose application-specific bounds on loads or prescribed displacements.

Theorem (uniformly elliptic scalar and convection–diffusion problems). Let $\Omega \subset \mathbb { R } ^ { d }$ be a bounded Lipschitz domain, $V = H _ { 0 } ^ { 1 } ( \Omega )$ , and

$$
a _ { s } ( u , v ) = \int _ { \Omega } A \nabla u \cdot \nabla v + ( \beta \cdot \nabla u ) v + c u v \mathrm { d } x .\tag{8}
$$

Assume $A \in L ^ { \infty } ( \Omega ; \mathbb { R } ^ { d \times d } )$ is symmetric and uniformly elliptic, so $\xi ^ { \mathsf { T } } A ( x ) \xi \geq \alpha \| \xi \| ^ { 2 }$ almost everywhere for some $\alpha > 0 ; \beta \in W ^ { 1 , \infty } ( \Omega ; \mathbb { R } ^ { d } ) ;$ ; and $\begin{array} { r } { c - \frac { 1 } { 2 } \nabla \cdot \boldsymbol { \beta } \ge 0 } \end{array}$ almost everywhere. Then, for every $L \ \in \ V ^ { \prime }$ , there is a unique $u \in V$ satisfying $a _ { s } ( u , v ) \ = \ L ( v )$ for all $v \in V ,$ , and $\| u \| _ { H ^ { 1 } } \leq C \| L \| _ { V ^ { \prime } }$

Proof. Integration by parts and the zero trace give $\begin{array} { r } { \int _ { \Omega } ( \boldsymbol { \beta } \cdot \nabla v ) v \mathrm { d } x = - \frac { 1 } { 2 } \int _ { \Omega } ( \nabla \cdot \boldsymbol { \beta } ) v ^ { 2 } } \end{array}$ dx. Hence $a _ { s } ( v , v ) \geq  { \hat { \alpha } } \| \nabla v \| _ { L ^ { 2 } } ^ { 2 } ,$ which is coercive on $H _ { 0 } ^ { 1 } ( \Omega )$ by Poincaré’s inequality. Boundedness of the coefficients makes $a _ { s }$ continuous, so Lax–Milgram yields existence, uniqueness, and stability (Ciarlet, 1978). Setting $\beta = 0$ and $c = 0$ covers Poisson and heterogeneous diffusion; nonzero $\beta$ covers the stated coercive convection–diffusion subclass. □

Theorem (isotropic elasticity: constitutive stability and well-posedness). Let $\Omega \subset \mathbb { R } ^ { 3 }$ be a bounded, connected Lipschitz domain whose boundary is partitioned into a relatively open Dirichlet part $\Gamma _ { D }$ and a traction part $\Gamma _ { N }$ . Let

$$
V = \{ v \in H ^ { 1 } ( \Omega ; \mathbb { R } ^ { 3 } ) : \operatorname { T r } v = 0 \mathrm { o n } \Gamma _ { D } \} .
$$

For isotropic parameters $E \in \mathbb { R }$ and $\nu \in \mathbb { R } \setminus \left\{ - 1 , 1 / 2 \right\}$ , the constitutive energy in Equation 7 is positive for every nonzero symmetric strain if and only if

$$
\begin{array} { r } { E > 0 , \qquad - 1 < \nu < \frac { 1 } { 2 } . } \end{array}
$$

If these equivalent conditions hold and $\left| \Gamma _ { D } \right| > 0 ,$ , then, for every bounded load functional $L \in V ^ { \prime } ,$ the weak displacement problem

$$
\mathrm { f i n d ~ } u \in V \mathrm { ~ s u c h ~ t h a t ~ } a ( u , v ) = L ( v ) \quad \forall v \in V ,\tag{9}
$$

with

$$
a ( u , v ) = \int _ { \Omega } 2 \mu \varepsilon _ { \mathrm { d e v } } ( u ) : \varepsilon _ { \mathrm { d e v } } ( v ) + K \ \mathrm { d i v } u \ \mathrm { d i v } v \mathrm { d } x ,
$$

has a unique solution and $\| u \| _ { H ^ { 1 } } \leq C \| L \| _ { V ^ { \prime } }$ . If instead $\Gamma _ { D } = \emptyset ,$ the same operator is not injective on $H ^ { 1 } ( \Omega ; \mathbb { R } ^ { 3 } )$ because it annihilates every rigid displacement.

Proof. The orthogonal decomposition $\begin{array} { r } { \varepsilon = \varepsilon _ { \mathrm { d e v } } + \frac { 1 } { 3 } ( \operatorname { t r } \varepsilon ) I } \end{array}$ shows that the deviatoric and spherical strain subspaces are independent. Equation 7 is therefore positive for every nonzero symmetric ε if and only if $\mu > 0$ and $K > 0 \mathrm { : }$ : necessity follows by choosing respectively a nonzero traceless strain and a nonzero spherical strain, and sufficiency follows because the two squared terms cannot vanish simultaneously. Substituting Equation 6 gives $\mu > 0$ and $K > 0$ if and only if $E > 0$ and $- 1 < \nu < 1 / 2$

Now assume these conditions and $\left| \Gamma _ { D } \right| > 0 .$ . Orthogonality also gives

$$
a ( v , v ) \geq \operatorname* { m i n } \{ 2 \mu , 3 K \} \| \varepsilon ( v ) \| _ { L ^ { 2 } ( \Omega ) } ^ { 2 } .
$$

Korn’s inequality on V yields $\| v \| _ { H ^ { 1 } } \leq C _ { K } \| \varepsilon ( v ) \| _ { L ^ { 2 } } ,$ so a is coercive on $V ;$ it is continuous by Cauchy–Schwarz. The Lax–Milgram theorem then gives the unique solution of Equation 9 and the stated stability estimate (Ciarlet, 1978). Finally, if $\Gamma _ { D } = \emptyset ,$ , every rigid displacement $r ( x ) =$ $c + Q x$ with $Q ^ { \mathsf { T } } = - Q$ satisfies $\varepsilon ( r ) = 0$ , hence $a ( r , v ) = 0$ for all v. The operator consequently has a rigid-motion kernel and uniqueness fails without a quotient-space or additional constraint. □

Theorem (steady Stokes flow). Let $V = H _ { 0 } ^ { 1 } ( \Omega ; \mathbb { R } ^ { d } ) , Q = L _ { 0 } ^ { 2 } ( \Omega )$ , and let the viscosity satisfy $\eta \in L ^ { \infty } ( \Omega )$ and $\eta ( x ) \geq \eta _ { 0 } > 0$ almost everywhere. If the velocity–pressure spaces satisfy the divergence inf-sup condition, then for every bounded body-force functional there is a unique weak Stokes solution $( u , p ) \in V \times Q ,$ , with pressure normalized to zero mean, and the solution obeys a stability estimate in $\| u \| _ { H ^ { 1 } } + \| p \| _ { L ^ { 2 } }$

Proof. Positive viscosity and Korn’s inequality make the viscous form coercive on the divergence-free kernel. Continuity of the viscous and divergence forms together with the inf-sup condition gives existence, uniqueness, and stability by the mixed variational theorem (Girault and Raviart, 1986). The same conclusion holds for a conforming discrete pair when its discrete inf-sup constant is positive. □

Relation of the theorems to the gate. These guarantees are not inferred from solver convergence. They apply when a candidate preserves the corresponding domain, boundary, coefficient, loading, and discretization hypotheses: uniform ellipticity and the convection–reaction sign condition for scalar problems; a positive-measure Dirichlet boundary together with $E > 0$ and $- 1 < \nu < 1 / 2$ for isotropic elasticity; or positive viscosity and an inf-sup-stable velocity– pressure pair for Stokes flow. They therefore cover several major operator classes in the evaluation, but not every heterogeneous seed. Helmholtz problems additionally require exclusion of resonant frequencies, and nonlinear Navier–Stokes problems require regime-dependent existence or uniqueness assumptions; for those classes we retain the local discrete criterion below rather than claim unconditional global well-posedness.

Local discrete solvability. After assembly, write the discrete residual as $R ( u ; x ) = 0$ and its initial linearization as

$$
A _ { 0 } ( x ) = \frac { \partial R } { \partial u } \bigg | _ { u = u _ { 0 } } .\tag{10}
$$

If R is continuously differentiable and $A _ { 0 } ( x )$ is nonsingular, the implicit function theorem gives a neighborhood in which a locally unique differentiable solution branch exists (Krantz and Parks, 2002), with

$$
\frac { \mathrm { d } u } { \mathrm { d } x } = - A _ { 0 } \big ( x \big ) ^ { - 1 } \frac { \partial R } { \partial x } .\tag{11}
$$

Nonsingularity is thus a sufficient regularity condition for local discrete solvability, not a claim of global uniqueness or easy numerical convergence. When an assembled operator is exposed, the canonical implementation checks LU pivots relative to the operator scale. The evaluated dataset bridges do not expose a common operator or rank diagnostic, so $\mathcal { X } _ { \mathrm { s o l v e } }$ operationally requires the configured solve to report successful convergence and finite outputs. A rejected nonconvergent run therefore records failure of the observable completion criterion; it does not prove that the PDE is mathematically unsolvable.

Composed gate. For execution result $r = S ( x )$ , the implemented decision can be written as

$$
V ( x , r ) = V _ { \mathrm { f i n } } ( x , r ) V _ { \mathrm { g e o } } ( x ) V _ { \mathrm { c o n s t } } ( x ) V _ { \mathrm { s o l v e } } ( r ) , \qquad V ( x , r ) \in \{ 0 , 1 \} .\tag{12}
$$

Input-decidable factors are evaluated before a solver query when possible; mesh- and solvedependent factors are evaluated after commitment. Equation 12 is simply the executable gate corresponding to Equation 3; we make no proposition out of that definitional equivalence.

## F Dataset and Target-Solver Details

The dataset is assembled before search and remains fixed for all methods. Table 13 establishes the evaluation population before the dataset-specific processing and exclusions below. Collected counts all public specifications, while Included counts seeds admitted after confirming a usable baseline and at least one supported IR edit. Admission is independent of attack performance.

Dataset-specific processing. For Elmer, we retain the native regression-suite .sif/.grd inputs and map editable physical fields into the shared IR; every write-back restores the original solver blocks. PDE-Agent records provide structured physical specifications, and their meshes are materialized parametrically with Gmsh before execution in DOLFINx/PETSc. NAFEMS cases use independently reproduced public benchmark specifications and versioned meshes; the supported scalar and elasticity cases execute through the corresponding DOLFINx or scikit-fem bridge. Native Elmer snapshots and all versioned IR records remain immutable during search.

Eligibility accounting. The manifest records execution eligibility rather than claiming that an excluded PDE is mathematically unsolvable. Of the 1,087 Elmer cases, 38 expose neither an editable parameter nor a supported editable node. Another 430 do not yield a usable baseline record in the dataset sweep: 156 are not dispatched because their mesh representation is neither .grd-backed nor recognized by the native-input dispatch path, while 274 are dispatched but fail. Among the latter, 250 terminate without a clean ElmerSolver completion, eight reach the 60-second baseline-sweep timeout, seven lack a usable Mesh DB declaration, three reference a missing native .sif, and six encounter mesh-generation or temporary-filesystem errors. Baseline replay additionally excludes elmer\_MortarPoisson2D\_ref: its initial sweep record reported 5,000 iterations, but clean replay in both model campaigns terminates with BiCGStabl numerical divergence before any attack candidate is evaluated. This outcome-independent eligibility correction leaves 618 Elmer seeds. Of the 645 PDE-Agent cases, 42 transient cases are not dispatched because their time formulation is not recognized by the transient adapter. The other 159 excluded cases are dispatched but fail during translation, mesh construction, or variational-form assembly: 118 require unsupported domain, periodic-mesh, hole, or boundaryregion handling, and 41 contain missing or unhandled fields or forms, including absent alpha, integration domains, vertices, initial conditions, or material parameters. This leaves 444 PDE-Agent seeds. Of the 14 NAFEMS benchmarks, seven use physics unsupported by the available bridges; NAFEMS LE6 reaches the elasticity bridge but its encoded boundary conditions provide no Dirichlet constraint, leaving rigid-body modes and a singular system. The remaining six NAFEMS seeds are included.

Table 13. The fixed manifest contains 1,068 eligible seeds.
<table><tr><td>Dataset</td><td></td><td></td><td>Collected Included Source representa- Target solver tion</td><td></td></tr><tr><td>Elmer</td><td>1,087</td><td></td><td>618 Native .sif/.grd ElmerSolver regression cases</td><td></td></tr><tr><td>PDE-Agent</td><td>645</td><td></td><td>444 Versioned bench- DOLFINx 0.9 + PETSc mark records with 3.24 parametric meshes</td><td></td></tr><tr><td>NAFEMS</td><td>14</td><td></td><td>6 Independently reproduced bench- DOLFINx mark IR and .msh bridge</td><td>scikit-fem 12.0.2 or scalar</td></tr><tr><td>Total</td><td>1,746</td><td>1,068</td><td>meshes Fixed before attack search</td><td></td></tr></table>

The resulting 1,068 seeds cover biharmonic, Burgers, convection–diffusion, diffusion–reaction, large-deformation elasticity, electromagnetic wave, electrostatics, steady and transient heat, Helmholtz, small-strain isotropic elasticity, magnetostatics, Navier–Stokes, Poisson, Stokes, acoustic wave, and a residual “other” class used for supported Elmer regression problems that do not map cleanly to the preceding labels.

Fixed identity and comparability. Each manifest entry fixes the dataset source, seed identifier, physics label, editable IR surface, native writer, target solver, acceptance-gate policy, and requested cost field before any method runs. All search methods therefore operate on the same seed identities and reconstruction path. The LLM configurations and black-box baselines differ in proposal policy, not in the target solver or execution environment.

Metric assignment and run logging. The corrected cohort initially requests linear iterations for 469 seeds, Newton iterations for 411, and wall time for 188. A count whose baseline value is at most one cannot express meaningful amplification, so a deterministic one-time fallback selects another positive finite count or wall time before search and then locks that field. The final cohort has 685 usable count-based ratios and 383 wall-time ratios. Primary statistics use count seeds and wall-time seeds are reported separately. Count baselines below three are flagged in the result artifacts because their ratios are quantized; the small-denominator sensitivity analysis in Appendix A.1 reports P90 and maximum after requiring baseline count ≥ 3. The headline maximum itself has a baseline of 43 and rises to 247,367 iterations (5,752.72×). Within each trajectory, the seed baseline is executed once and cached without consuming the query budget; each committed candidate is executed once, and the final maximum is selected from those observations without a confirmation rerun. Each run records the served model identifier, requests, failures, token and cache usage, realized candidate-set summaries, solver outcomes, and completion status. A complete method-level accounting of solver time, LLM latency, tokens, and monetary cost remains future reporting work.

(a) Geometry-induced degradation: 2% outer-radius change  
![](images/9eed491f1122fa6865a27b77613112a56e342972c2673bc08ef3c5614c4be70a.jpg)

![](images/d05afaeda447fe314f7422200e7861feabc98df320a19be9454bcdd0b3422667.jpg)

![](images/bc08c7aa5765f66b1a9334ac2a9c1d793917b0a04173c10400763d149a98735c.jpg)

(b) Coefficient-induced degradation near a Dirichlet resonance  
![](images/8a9bbb1cf017373813a106c2e1912aedd6c5b88c80b22782c98fbacbf2c0dd8f.jpg)

![](images/4c093ac8d58c6a86ef99f7957124616aafa19c368145cf797d687c4118e3f490.jpg)

![](images/699d16d37d5a719f9a5d18928ab30aff9644b3a433e4f60c430df96b9b0b3456.jpg)  
(c) Node-induced degradation of a circular Poisson mesh

![](images/de09ac16efe3d4450b32f90143caf9128358c8f7ba8704395a6bd47b39d816ae.jpg)

![](images/bdace9ed9608d92bd0d65b9ede10f6fb00f5303f7eeeb445baf49e4b10e31e29.jpg)

![](images/4d94d4b62a960439d5b78f7e1af8882d6a78712fa2f3226b4e19ae938df3dea8.jpg)  
Figure 11. Small FEM input edits can substantially increase solver work.

## G Representative FEM Case Visualizations

Figure 11 presents the three complementary attack surfaces analyzed below. Each case isolates the edited input component and connects it to measured solver work.

Parametric geometry. In pdeagent\_helmholtz\_annulus\_thin\_k\_high, $r _ { \mathrm { i n } } = 0 . 9$ remains fixed while $r _ { \mathrm { o u t } }$ changes from 1.00 to 1.02. With $k = 3 0 ,$ , materials, and boundary conditions fixed, the remeshed domain changes from 2,099 to 2,512 nodes and CG work rises from 95 to 417 iterations (4.39×).

Physical coefficient. In pdeagent\_helmholtz\_k12\_exponential\_mode, geometry, mesh, source, and boundary conditions remain fixed while k changes from 12 to 23.942385. The nearest unitsquare Dirichlet spectral gap decreases from 15.70 to 0.80, and CG work rises from 775 to 3,897 iterations (5.03×). Both this case and the geometry case satisfy the strict-relative bounds used in Table 2.

Mesh nodes. In pdeagent\_poisson\_circle\_exp\_sin, the domain, PDE, forcing, and boundary conditions remain fixed while 58 unique nodes move. Blue outlines and red markers show benign and adversarial positions; the distortion raises CG work from 370 to 1,162 iterations (3.14×). This displayed winner comes from the unrestricted search and exceeds the strict cumulative per-node limit of $0 . 0 5 h = 0 . 0 0 0 3 1 2 5$ . The seed is included in the strict comparison, but its strict winner combines bounded center and radius edits with movement of one node and raises work from 370 to 383 iterations $( 1 . 0 4 \times ) ;$ it is therefore not a strict nodeonly analogue of the displayed case. The main-paper composite can be regenerated with paper/figures/plot\_case\_visualizations.py. The script materializes the benign and adversarial meshes from the stored IR records and uses the recorded iteration counts for the cost panels.

## H Input-Similarity Measurement

Matched descriptive protocol. The analysis compares the original seed x with the saved best attack $x ^ { \star }$ for uniform random search, evolutionary search, CMA-ES, Nevergrad OnePlusOne, Nevergrad CMA, CloakFEM (DeepSeek), and CloakFEM (GPT). It uses the intersection of 1,063 seeds with records from all seven methods and is entirely post hoc: similarity neither constrains the search nor affects which winner is saved. Elmer seeds are reconstructed from the immutable native sif0/grd0 stored with each search record; PDE-Agent and NAFEMS seeds are loaded from the versioned dataset. Both inputs are projected through the same structured IR used by the search implementation. Read-only context such as the materialized node count is excluded.

Leaf and component similarity. Let $P _ { 0 }$ and $P _ { \star }$ be the typed leaf paths in the two structured inputs. Structural similarity is their Jaccard overlap,

$$
S _ { \mathrm { t o p } } ( x _ { 0 } , x ^ { \star } ) = \frac { \left| P _ { 0 } \cap P _ { \star } \right| } { \left| P _ { 0 } \cup P _ { \star } \right| } .\tag{13}
$$

Equal categorical leaves receive score one and unequal categorical leaves score zero. For finite numeric leaves a and b with the same nonzero sign, the symmetric fold-change score is

$$
s _ { \mathrm { n u m } } ( a , b ) = \frac { 1 } { 1 + \log _ { 2 } ( \operatorname* { m a x } \{ | a / b | , | b / a | \} ) } .\tag{14}
$$

An unchanged value scores one, a twofold change scores 0.5, and a tenfold change scores approximately 0.23. Zero-to-nonzero, sign, missing, and type changes score zero. Leaf scores are averaged separately for geometry $( S _ { \mathrm { g e o m } } )$ , materials $( S _ { \mathrm { m a t } } ) _ { \mathrm { . } }$ , boundary conditions $( S _ { \mathrm { b c } } )$ , remaining physical configuration $( S _ { \mathrm { o t h e r } } ) _ { \cdot }$ , and categorical values $( S _ { \mathrm { c a t } } )$

For explicit node perturbations, displacements to the same node are accumulated and normalized by the seed’s characteristic mesh size h. If $d _ { \mathrm { m a x } }$ is the largest resulting nodal displacement, mesh similarity is

$$
S _ { \mathrm { m e s h } } = \exp ( - d _ { \mathrm { m a x } } / h ) .\tag{15}
$$

No node displacement scores one. Cases without an available h retain the other components and treat mesh similarity as missing rather than silently assigning a favorable value.

The reported surface, physical, and overall scores are

$$
S _ { \mathrm { s u r f a c e } } = \mathrm { A M } ( S _ { \mathrm { t o p } } , S _ { \mathrm { g e o m } } , S _ { \mathrm { m e s h } } ) ,\tag{16}
$$

$$
S _ { \mathrm { p h y s i c a l } } = \mathrm { A M } ( S _ { \mathrm { m a t } } , S _ { \mathrm { b c } } , S _ { \mathrm { o t h e r } } ) ,\tag{17}
$$

$$
S _ { \mathrm { o v e r a l l } } = \mathrm { A M } ( S _ { \mathrm { s u r f a c e } } , S _ { \mathrm { p h y s i c a l } } , S _ { \mathrm { c a t } } ) ,\tag{18}
$$

where AM denotes the arithmetic mean after omitting missing components. The decomposition prevents an unchanged mesh from concealing a large material edit, while the final scalar supports a common threshold analysis. This is only an input-space measure: it does not claim that the resulting FEM solution fields are visually or norm-wise similar, that residual or conservation error remains comparable, or that the edited input represents the same application-level task.

## I Numerical Mechanisms of Efficiency Degradation

Figure 12 reports the point-level Stokes replay used in the mechanism analysis below.

We separately analyze why selected gate-accepted inputs become expensive. Candidate mechanisms include strong material contrast, near incompressibility, mesh distortion, low-viscosity nonlinear difficulty, behavior near a limit point, contact-state changes, and sparsegraph fill-in. The discussion below is not an exhaustive taxonomy. It distinguishes direct attribution supported by exact-winner replay from a transfer pattern that does not yet identify a single causal edit.

![](images/4f4b58455bea2a3a4223d8bd6e86376b40b25adb8c7c8f7422403dae3f8bb645.jpg)

Attribution protocol. Depending on target-solver support, we compare condition estimates, Krylov or Newton iterations, linear-solve time, matrix reassembly, and

Figure 12. Stokes transfer is frequent but weak.

factorization fill between the benign seed and the exact saved winner. These diagnostics are used only after search and are not visible to the attacker. We report a diagnostic only when the replay solver exposes and records it; an unavailable spectrum, factorization statistic, or timing component remains missing rather than being inferred from solver completion. An LLM rationale is likewise not treated as mechanism evidence. A direct attribution requires an identified edit, a corresponding change in measured solver work, and reproduction in an independent implementation or a diagnostic that isolates the relevant numerical effect.

Near incompressibility and volumetric locking. The strongest directly supported linear mechanism is the approach to the incompressible limit in elasticity. The transferred winners move Poisson’s ratio toward ν = 0.5 while preserving positive elastic moduli and the other validity constraints. The cross-implementation agreement reported in Section 4.3 therefore supports problem-intrinsic volumetric locking rather than a single implementation defect.

Coefficient contrast and discretization sensitivity. Poisson winners that increase spatial coefficient contrast exhibit the mixed transfer pattern quantified in Section 4.3. The edit therefore exposes real coefficient-induced difficulty, but its magnitude depends on discretization, assembly, and target-solver behavior.

Low-viscosity nonlinear difficulty. All three Navier–Stokes exact winners reduce viscosity by orders of magnitude. Two complete the independent Taylor–Hood/Newton solves with large increases in Newton work; the third reaches the Newton cap in both targets and is retained as a lower-bound transfer. Together with the ratios in Section 4.3, this supports problem-intrinsic low-viscosity nonlinear difficulty.

Node perturbations and mesh effects. The three elasticity attacks that fail to reproduce make only node perturbations and have weak native amplification. Their target ratios fall near one, showing that a saved geometric perturbation can exploit details of one mesh or assembly without creating a generally difficult boundary-value problem. The circular Poisson example in Figure 11(c) separately demonstrates that moving 58 nodes can raise native CG work from 370 to 1,162 iterations, but it is an unrestricted mechanism illustration rather than evidence that node-induced amplification generally transfers. Mesh distortion is therefore a plausible attack surface, while the present cross-solver evidence does not justify a universal mesh-distortion attribution.

Transferred operator classes without a unique attribution. The convection–diffusion and Stokes transfer reported in Figure 5 and Figure 12 establishes that edited boundary-value problems can remain difficult across implementations, yet the replay records do not isolate one common edit or operator diagnostic within either class. We therefore do not assign all winners in these classes to a single mechanism.

Mechanisms not yet established. Contact-state oscillation, behavior near nonlinear limit points, and attack-induced sparse fill remain candidate mechanisms rather than validated findings in the complete exact-winner campaigns. An earlier elasticity defense pilot records direct-factor fill with median 11× and P90 16.4× as many nonzeros as the original matrix, but that diagnostic does not establish that the selected attacks themselves changed sparsity or mesh size. The evaluated search also locks mesh-resolution controls. Consequently, we do not infer fill-in, contact, or limit-point mechanisms from runtime alone; establishing them requires dedicated adapters and directly recorded matrix, active-set, or continuation diagnostics.

## J Extended-Budget Sensitivity

We extend the full CloakFEM (GPT) pipeline from the standard maximum budget B = 40 to B = 100 on the fixed cohort of all 444 PDE-Agent seeds. The per-trajectory limit is increased proportionally from 1,800 to 4,500 seconds. Only 249 seeds complete all 100 queries, and restricting the analysis to them would exclude many slow, high-amplification trajectories. We therefore carry each incomplete trajectory’s checkpointed best-so-far ρ<sup>⋆</sup> forward. Because search is monotone in its incumbent, these fixed-cohort values are conservative lower bounds rather than imputations of unobserved improvements. Figure 13 presents the resulting sensitivity analysis. In Table 14, Reached counts trajectories that actually complete each budget, Coverage gives the corresponding cohort fraction, and the amplification summaries use the carriedforward lower bounds.

![](images/7f1c6c494c1e2d10bc781dcc3bf92847fccad1f03b9e897e0117a65487e7db6d.jpg)

![](images/699c5d0af76a93ee67abac822e6c16bb661f16fba7d4cad8ab7898f97d350fb9.jpg)

![](images/a8ffdb03a2f93671f90c0621f3336b5a27ee5c2513e26b3fc2d8e09d2dc5ca90.jpg)

![](images/d5c67e49cd86a5a7780afce7cdbbd83b50e3970cbe4439062b13ffd25db70772.jpg)  
Figure 13. Larger budgets improve amplification but reduce coverage.

The median lower bound increases by 17.7% from $B = 4 0$ to B = 100, whereas P90 increases by 19.9% and then plateaus after $B = 8 0$ . The arithmetic mean changes little after $B = 4 0$ because it is already dominated by valid count-based attacks with amplification in the thousands. Median and P90 therefore provide more representative typical and upper-tail summaries. The simultaneous decline in coverage is why the extended-budget result is reported as a sensitivity analysis rather than as a direct replacement for the standard-budget comparison.

## K Reproducibility and Reporting Details

Code and reproduction artifacts are available at https://anonymous.4open.science/r/cloakfem-5021/.

Immutable seed and result identity. Every search record stores the dataset source, seed identifier, effective cost field, baseline cost, query count, best valid ratio, selected input, validity outcome, and completion status. Aggregation deduplicates reruns by seed and prefers the finite record with the largest completed query count.

Execution boundary and randomness. The LLM proposes structured edits, while sourcespecific writers, pre-execution checks, solver invocation, post-solve validity, query accounting, checkpointing, and final ranking remain deterministic framework operations. Frameworkcontrolled sampling uses NumPy’s generator with replicate seed 0, including the fixed $\varepsilon = 0 . 1 5$ archive policy and seeded black-box proposals. The model APIs are called at temperature 0.7 with their standard sampling behavior. Numerical-library threads are fixed to one, and the software versions, hardware, and timeout limits are reported in Section 4.1.

Generated artifacts. The main evaluation figures are regenerated by:

paper/figures/plot\_evaluation\_figures.py

The joint similarity analysis uses:

experiments/measure\_attack\_similarity.py

paper/figures/plot\_similarity\_tradeoff.py

Representative cases use:

paper/figures/plot\_case\_visualizations.py

The extended-budget fixed-cohort analysis uses:

experiments/aggregate\_budget100\_sensitivity.py

The corresponding CSV, JSON, JSONL, PNG, and PDF outputs are retained with the experiment results. Stochastic uncertainty is reported across seeds; unless stated otherwise, each seed– method pair has one replicate and should not be interpreted as estimating between-run model variance. Table 6 is the exception: it reports three framework seeds for evolutionary search on a stratified subset, but no successful repeated model trajectories.

## L Extended Background and Related Work

Table 14. Larger budgets yield diminishing returns and lower coverage.
<table><tr><td>B</td><td>Reached</td><td>Coverage</td><td>Median</td><td>Arithmetic mean</td><td>P90</td></tr><tr><td>40</td><td>330/444</td><td>74.3%</td><td>2.05</td><td>130.06</td><td>125.89</td></tr><tr><td>60</td><td>294/444</td><td>66.2%</td><td>2.21</td><td>131.45</td><td>136.87</td></tr><tr><td>80</td><td>269/444</td><td>60.6%</td><td>2.33</td><td>132.39</td><td>150.93</td></tr><tr><td>100</td><td>249/444</td><td>56.1%</td><td>2.42</td><td>132.54</td><td>150.93</td></tr></table>

## L.1 Finite-Element Inputs and Numerical Cost

FEM discretizes a partial differential equation over a mesh into a sparse algebraic system, or, for nonlinear and transient problems, a sequence of such systems (Hughes, 2000; Strang and Fix, 2008). The mesh and element type determine the sparsity pattern, while geometry, material coefficients, constitutive laws, boundary conditions, and loads determine the entries and righthand side. Consequently, two accepted inputs of comparable size can induce very different numerical work.

For sparse direct methods, factorization cost is dominated by fill-in and therefore depends on the sparsity pattern and elimination ordering rather than only on the number of nonzeros (George and Liu, 1981; Davis, 2006). For Krylov methods, convergence depends on the operator spectrum and the effectiveness of the preconditioner; coefficient jumps, anisotropy, and mesh distortion can degrade both (Saad, 2003; Trefethen and Bau, 1997). Nonlinear solves add Newton iterations, line-search backtracking, and step cuts, whose counts depend on constitutive nonlinearities and penalty stiffnesses. Transient solvers additionally couple work to stability limits and adaptive time stepping, while remeshing or adaptive refinement can change the degrees of freedom as a function of the submitted model.

Well-posedness and good conditioning are distinct. Near-incompressible elasticity, for example, is a meaningful physical regime but can produce volumetric locking in standard displacement formulations (Babuška and Suri, 1992); similarly, ordinary heterogeneous materials can create strong coefficient contrast. An accepted model can therefore converge to a valid solution while consuming substantially more work. This distinction motivates our acceptance gate: it rejects malformed, unsupported, or nonconvergent candidates, but does not equate acceptance with easy numerical solution or application-level realism. The distinction is operationally important for shared HPC queues, simulation services, and solver-backed endpoints, where the party specifying a model may not bear its full execution cost (Baratta et al., 2023; Malinen and Råback, 2013; Balay et al., 2019).

## L.2 Efficiency Degradation as an Attack Surface

Algorithmic-complexity attacks established the general pattern of a legitimate input that produces the expected output while forcing disproportionate work. Hash-collision attacks exploit worst-case dictionary behavior (Crosby and Wallach, 2003), and regular-expression denial of service drives backtracking matchers into pathological execution (Staicu and Pradel, 2018; Davis et al., 2018). These attacks differ from request-flooding denial of service: amplification arises from the computational structure of an individual accepted input.

Related attacks on learned systems maximize energy or latency, defeat early-exit mechanisms, delay end-of-sequence generation, or inflate post-processing work (Shumailov et al., 2021; Hong et al., 2021; Chen et al., 2022a; Feng et al., 2024; Dong et al., 2025; Shapira et al., 2023). Their mechanisms are typically data-dependent branches or variable-length computation inside a learned pipeline. CloakFEM instead targets convergence and discretization behavior in deterministic numerical software. The attacker submits one gate-accepted model, cannot alter protected solver controls, and optimizes measured work rather than prediction error. Work on adversarial scientific machine learning attacks learned surrogates such as neural operators and evaluates accuracy degradation (Adesoji and Chen, 2022); our victim, objective, and failure mode are therefore different.

Numerical software also differs from architectures with an explicit maximum number of exits, tokens, or retained detections: solver work depends on the submitted problem and convergence trajectory rather than a single fixed branch count. Cost can rise through several mechanisms, including remeshing and larger systems, degraded conditioning at fixed topology, or slower nonlinear and transient convergence. We report mesh size, absolute work, and solver outcomes where available so that these mechanisms are not conflated with crashes, malformed inputs, or silent accuracy failures.

## L.3 Code Agents and LLM-Guided Optimization

Language models for code progressed from function completion (Chen et al., 2021; Austin et al., 2021) to agents that edit repositories and use execution feedback (Jimenez et al., 2024; Yang et al., 2024b; Wang et al., 2025). A related line places a language model inside an externally evaluated search loop: FunSearch and AlphaEvolve evolve artifacts against deterministic evaluators (Romera-Paredes et al., 2024; Novikov et al., 2025), while OPRO and Evolution of Heuristics use language models as optimizers or heuristic designers (Yang et al., 2024a; Liu et al., 2024). These systems show that semantic proposal generation can complement conventional derivative-free search.

CloakFEM follows this outer-loop pattern but changes both the search object and the objective. Its candidates are typed edits to heterogeneous FEM specifications rather than free-form programs or flat vectors, and each evaluation requires a solver execution. Strategy, generation, and selection agents propose targets, instantiate magnitudes, and choose candidates, while representation checks, native write-back, acceptance decisions, archive updates, and query accounting remain deterministic. The model does not observe assembled matrices, spectra, preconditioner state, or hidden solver configuration. This separation makes candidate edits auditable and allows the system to search semantically without granting direct access to the numerical mechanism being stressed.

The direction of optimization also differs from work that seeks more correct, faster, or betterconverged artifacts: CloakFEM searches for gate-accepted, runnable inputs that increase the cost of an unchanged solver. We therefore compare it with random, evolutionary, CMA-ES, OnePlusOne, and ngCMA search under a common solver-query cap. These baselines are derivative-free proposal policies rather than agents (Hansen and Ostermeier, 2001); the comparison tests whether semantic access improves evaluator-query efficiency, not whether all methods have equal proposal-side compute or identical action spaces.