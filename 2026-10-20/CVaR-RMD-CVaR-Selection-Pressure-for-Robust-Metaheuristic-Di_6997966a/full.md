# CVaR-RMD: CVaR Selection Pressure for Robust Metaheuristic Discovery

Ruodai Cui<sup>1,2</sup> Li Niu<sup>3</sup>\*

<sup>1</sup>Qualcomm Technologies, Inc. <sup>2</sup>The Hong Kong Polytechnic University <sup>3</sup>Department of Computer Science and Engineering, MoE Key Lab of Artificial Intelligence, Shanghai Jiao Tong University ruodcui@connect.polyu.hk ustcnewly@sjtu.edu.cn

## Abstract

Large language models have recently been used to discover heuristics through iterative program search, offering a promising approach to automated algorithm design. However, candidates selected by average performance may degrade across instance families, random seeds, and problem scales because mean-based objectives can obscure rare but severe failures. We propose CVaR-RMD, a risk-sensitive evolutionary framework that uses an LLM to mutate executable heuristic programs and ranks candidates by empirical conditional value-atrisk over heterogeneous evaluation conditions. Across three combinatorial optimization domains, CVaR-based selection improves uppertail performance and reduces failure rates while maintaining competitive mean performance. These gains persist on larger unseen instances and independently curated withindomain benchmarks. Code is available here.

## 1 Introduction

Heuristics and metaheuristics are central to combinatorial, stochastic, black-box, or partially specified optimization problems. Their practical value depends not only on benchmark performance during algorithm discovery but also on whether it persists when evaluation conditions change within the same domain. Such changes include shifts across instance families, problem scales, and random seeds. We call this setting within-domain transfer under benchmark distribution shift: the optimization problem and solution representation remain unchanged, but benchmark conditions differ between discovery and testing. This concern is longstanding in heuristic optimization, where empirical success often depends on interactions among algorithm design, benchmark distributions, and stochastic variation (Talbi, 2009; Hoos and Stützle, 2004; Blum and Roli, 2003).

Automated algorithm design naturally supports studying this challenge because candidates are generated and selected by their observed performance under specified evaluation conditions. However, searching algorithmic choices alone does not ensure within-domain transfer, which depends critically on the evaluation conditions and ranking objective. Hyper-heuristics automate design through search spaces of heuristics or heuristic components, while algorithm configuration and selection evaluate candidate procedures across instance sets to support performance beyond individual discovery instances (Burke et al., 2013; Rice, 1976; Hutter et al., 2011; López-Ibáñez et al., 2016; Ansótegui et al., 2009).

A candidate heuristic may achieve strong average performance yet fail on rare instances, different stochastic seeds, or larger scales. This issue is especially relevant to LLM-generated metaheuristics because generated code may exploit a fixed evaluation protocol, such as a narrow instance generator, few seeds, or a restricted scale range. The outer loop may therefore select heuristics with high expected performance but unstable tails. In evolutionary search, this selection pressure rewards favorable means even when outcome distributions contain severe failures (Bäck, 1996; Eiben and Smith, 2003; Goldberg and Deb, 1991).

Because such failures occur only under adverse subsets of evaluation conditions, mean aggregation may obscure them. A useful selection criterion should therefore emphasize adverse outcomes without allowing a single extreme observation to dominate candidate ranking. CVaR satisfies this requirement by averaging performance over a configurable worst-performing fraction of evaluation outcomes. We therefore study a minimal change to discovery: replacing mean-based fitness with CVaR-based fitness while leaving the outer loop unchanged. Each candidate heuristic is evaluated over conditions spanning instance families, stochastic seeds, and scales. Its fitness uses conditional value-at-risk (CVaR), a coherent risk measure estimating expected loss within the worst fraction of outcomes at a chosen quantile level (Artzner et al., 1999; Rockafellar and Uryasev, 2000, 2002; Pflug, 2000). The resulting CVaR-based Robust Metaheuristic Discovery framework (CVaR-RMD) uses the LLM as a mutation generator to propose heuristic code, while estimated CVaR fitness determines candidate acceptance. As Figure 1 shows, meanbased selection tolerates severe tail failures that CVaR-RMD penalizes. Because selected programs become parents and prompt context, CVaR-based ranking affects both final selection and the subsequent trajectory of LLM-driven program discovery.

![](images/c46ebcfcb23ead21c604247d95726f0d87365699e38c7c37ac2e95d23830b2dd.jpg)  
Figure 1: Motivation for CVaR-RMD. Mean-based fitness in LLM-driven metaheuristic discovery rewards strong average performance while tolerating severe failures on rare instances or seeds. CVaR shifts selection pressure toward worst-quantile behavior, discouraging heuristics that depend on favorable evaluation conditions.

This formulation shifts selection toward worstquantile behavior and discourages candidates dependent on favorable evaluation conditions. It follows robust and distributionally robust optimization, where objectives accounting for adverse outcomes and distributional variation replace or supplement expectation-based criteria (Ben-Tal et al., 2009; Beyer and Sendhoff, 2007; Bertsimas et al., 2011; Rahimian and Mehrotra, 2019; Duchi and Namkoong, 2021). It also aligns with optimization under noise and uncertainty, which uses repeated evaluation and robust criteria to reduce the selection of brittle solutions (Jin and Branke, 2005; Branke et al., 2008; Beyer and Sendhoff, 2007). Our empirical analysis tests whether risk-based selection improves heuristic stability and transfer across evaluation conditions without adding architectural components to the LLM-driven discovery pipeline.

The contributions of this work are as follows:

1. We introduce CVaR-based selection into LLM-guided metaheuristic discovery, using empirical CVaR as the primary fitness criterion to emphasize adverse-quantile behavior across heterogeneous evaluation conditions.

2. We propose CVaR-RMD, an execution-driven, risk-sensitive framework that integrates CVaRbased candidate ranking, accumulated execution outcomes, and tail-case feedback into an evolutionary program-discovery loop for generating robust executable heuristics.

3. We conduct controlled, budget-matched experiments across three combinatorial optimization domains and show that CVaR-RMD improves upper-tail performance and reduces execution failures while maintaining competitive mean performance.

## 2 Related Work

Automated Heuristic Design Automated heuristic design has been studied through hyperheuristics, algorithm selection/configuration, and program synthesis. Hyper-heuristics search over heuristic spaces by selecting existing low-level operators or generating new components that generalize across instances (Burke et al., 2013; Cowling et al., 2001; Ochoa et al., 2012). Algorithm selection and configuration address related questions from a statistical and empirical perspective, using instance features, performance observations, and iterative evaluation to identify robust algorithms or parameterizations under limited budgets (Rice, 1976; Hutter et al., 2009, 2011; López-Ibáñez et al., 2016; Ansótegui et al., 2009). Earlier genetic programming and grammatical evolution methods directly searched over program structures via fitnessbased selection and variation (Koza, 1992; O’Neill and Ryan, 2001; Langdon and Poli, 2002). These works frame heuristic construction as a search problem whose evaluation protocol and selection objective strongly affect generalization.

Recent LLM-based methods extend this paradigm by coupling generative proposal mechanisms with external evaluation. FunSearch evolves and validates LLM-generated code variants using task-specific scoring functions (Romera-Paredes et al., 2024). EoH co-evolves natural-language “thoughts” and executable heuristics (Liu et al., 2024a), while ReEvo uses LLMs as hyper-heuristics with reflective evolution (Ye et al., 2024). LLaMEA evolves LLM-generated metaheuristics for black-box continuous optimization (van Stein and Bäck, 2024), and LLAMBO integrates LLM reasoning into Bayesian optimization pipelines (Liu et al., 2024b). Despite differences in representation, prompting, and search dynamics, these systems typically depend on scalar fitness values from limited empirical evaluations.

Robust and Risk-Aware Optimization Robust optimization seeks solutions that remain effective under uncertainty, emphasizing robustness metrics and the trade-off between nominal performance and stability (Ben-Tal et al., 2009; Beyer and Sendhoff, 2007; Bertsimas et al., 2011). Distributionally robust optimization further optimizes against worst-case distributions within ambiguity sets (Delage and Ye, 2010; Rahimian and Mehrotra, 2019; Duchi and Namkoong, 2021). Among risk measures, conditional value-at-risk (CVaR) captures tail behavior and supports tractable sample-based formulations (Artzner et al., 1999; Rockafellar and Uryasev, 2000, 2002; Pflug, 2000). Such measures are widely used in finance, stochastic programming, reinforcement learning, and decisionmaking (Shapiro et al., 2009; Tamar et al., 2015; Chow et al., 2015). Robustness under noise and uncertainty has also been extensively studied in evolutionary computation (Jin and Branke, 2005; Branke et al., 2008; Beyer and Sendhoff, 2007). These studies motivate objectives that account for unfavorable outcomes rather than only average performance.

Benchmarking, Robustness, and Transfer Evaluation protocols are central to empirical combinatorial optimization because conclusions about algorithm performance depend strongly on the instances and conditions under which algorithms are assessed. Benchmark suites and heterogeneous instance collections are commonly used to evaluate whether algorithms remain effective beyond their development conditions (Johnson and McGeoch, 1997; Ochoa et al., 2012; Reinelt, 1991). In routing and learned combinatorial optimization, prior studies have documented performance degradation when methods are evaluated across problem scales, instance generators, and instance families that differ from those used during development (Vinyals et al., 2015; Bello et al., 2017; Kool et al., 2019; Joshi et al., 2019). These findings motivate evaluation protocols that account not only for average performance but also for robustness under heterogeneous and shifted benchmark conditions. When candidate performance is summarized solely by the empirical mean, the selection process may favor heuristics that perform well on typical instances while remaining vulnerable to rare, difficult, or shifted cases. This concern is particularly relevant to LLM-driven heuristic discovery, where finite empirical evaluations repeatedly determine which generated programs are retained and used to guide subsequent search (Romera-Paredes et al., 2024; Liu et al., 2024a; Ye et al., 2024; van Stein and Bäck, 2024). Robust evaluation and selection criteria are therefore important for discovering heuristics that maintain reliable performance across instance families, random seeds, and problem scales.

![](images/12ec23115e1ff7af283e3792fae83f4db98b8f793995ced6c7622e2fffa1fb08.jpg)  
Figure 2: Architecture of the CVaR-guided program search method. Notations are defined in Sec. 3.

## 3 Method

We seek an executable heuristic program for a family of optimization instances. Let I be the instance space, D the scale-parameter set, and S the randomseed set controlling stochastic execution. An evaluation environment is $e \ = \ ( i , d , s ) \in \mathcal { E }$ , where $i \in \mathcal { I } , d \in \mathcal { D } , s \in \mathcal { S } .$ , and ${ \mathcal { E } } = { \mathcal { I } } \times { \mathcal { D } } \times { \mathcal { S } }$ Environments are sampled from a fixed distribution $P$ over E. A candidate heuristic is a program $h \in \mathcal H$ implementing a prescribed callable interface. Given e, executing h returns a feasible solution $x = h ( i , d , s )$ or a failure state. The scalar loss is $\ell ( h ; e ) \in \mathbb { R }$ , with lower values preferred. On success, $\ell ( h ; e )$ is computed from the task objective and returned solution. If the program fails to parse, compile, satisfy the interface, terminate within the execution limit, or return a feasible solution, the loss is set to a fixed penalty $\ell _ { \mathrm { f a i l } }$

Our objective accounts for the upper tail of the loss distribution rather than its mean alone. For $E \sim P .$ , let $L _ { h } \ = \ \ell ( h ; E )$ be the random loss induced by h. Given tail level $q \in ( 0 , 1 )$ , the population objective is

$$
h ^ { \star } \in \arg \operatorname* { m i n } _ { h \in \mathcal { H } } \mathrm { C V a R } _ { q } ( L _ { h } ) .\tag{1}
$$

Here $\begin{array} { r } { \mathrm { C V a R } _ { q } ( L ) = \operatorname* { i n f } _ { \eta \in \mathbb { R } } \{ \eta + \frac { 1 } { 1 - q } \mathbb { E } [ ( L - \eta ) _ { + } ] \} } \end{array}$ where $( x ) _ { + } = \operatorname* { m a x } ( x , 0 )$ . The instance distribution, callable interface, loss transformation, failure penalty, and evaluation budgets are fixed before search.

The method performs population-based program search with tail-risk scoring. At generation t, a population $\mathcal { P } _ { t } = \{ h _ { t } ^ { ( 1 ) } , . . . , h _ { t } ^ { ( N ) } \}$ is evaluated on sampled environments. Each candidate yields a loss vector converted to empirical CVaR fitness. Candidates are ranked, selected programs are mutated by a language-model operator, and returned source code is parsed, compiled, validated, and evaluated.

The next population is formed by elitist replacement over the union of parents and offspring, as illustrated in Figure 2. The remainder of this section is organized as follows: Section 3.1 defines the evaluation distribution and loss construction, Section 3.2 specifies empirical CVaR scoring, and Section 3.3 describes the evolutionary search loop.

## 3.1 Evaluation Distribution and Loss Construction

The evaluation module maps sampled problem settings to scalar losses. Its inputs are a candidate program $h ,$ batch size $m _ { t }$ , and environment distribution $P ;$ its output is

$$
z _ { t } ( h ) = \big ( z _ { t , 1 } ( h ) , \ldots , z _ { t , m _ { t } } ( h ) \big ) \in \mathbb { R } ^ { m _ { t } } ,
$$

with each entry corresponding to one execution of h on a sampled environment.

Instances are produced by a finite generator set G. A generator $g \in { \mathcal { G } }$ maps a scale value d and instance seed u to $i = g ( d , u )$ . The environment distribution samples $g \sim \pi _ { \mathcal { G } } , d \sim \pi _ { \mathcal { D } } ( \cdot \mid g ) , u \sim \pi _ { \mathcal { U } } ,$ and $s \sim \pi s$ , then sets $i = g ( d , u )$ and $e = ( i , d , s )$ The generator identity, scale value, instance seed, and execution seed are stored with the environment, making each execution reproducible within the protocol.

At generation t, the sampled batch is $B _ { t } \ =$ $\{ e _ { t , 1 } , \ldots , e _ { t , m _ { t } } \}$ . All candidates in the same generation use the same batch, so scores are comparable under identical environments. For $\begin{array} { r l } { \boldsymbol { e } _ { t , j } } & { { } = } \end{array}$ $( i _ { t , j } , d _ { t , j } , s _ { t , j } )$ , the harness calls the fixed interface and obtains a solution object $x _ { t , j }$ or a failure state. A deterministic validator checks whether $x _ { t , j }$ satisfies the feasibility constraints of $i _ { t , j }$ . If the program is valid and the solution feasible, the raw task value is transformed into a scalar loss. For minimization objective $a ( i , x ) , z _ { t , j } ( h ) = a ( i _ { t , j } , x _ { t , j } )$ . For maximization reward $r ( i , x )$ , the equivalent minimization loss is $z _ { t , j } ( h ) = - r ( i _ { t , j } , x _ { t , j } )$

If a reference value $b ( i )$ is available, a normalized minimization loss can be used:

$$
z _ { t , j } ( h ) = \frac { a ( i _ { t , j } , x _ { t , j } ) - b ( i _ { t , j } ) } { \operatorname* { m a x } \{ | b ( i _ { t , j } ) | , \epsilon \} } , \qquad \epsilon > 0 .
$$

Failure handling precedes score computation. If the source code cannot be parsed or compiled, all entries for that candidate in the current batch are set to $\ell _ { \mathrm { f a i l } }$ . If compilation succeeds but one execution fails, only that entry is set to $\ell _ { \mathrm { f a i l } }$ . Thus, the loss vector may contain objective-derived and penalty values. The scoring module receives only this scalar vector, not program text, exception type, or feasibility metadata.

The batch policy controls loss accumulation across generations. With a fixed batch, the same $B$ is used throughout search, and each candidate is scored on identical environments whenever evaluated. With resampling, a new $B _ { t }$ is drawn each generation. In the accumulated-loss variant, the method stores a multiset $Z ( h )$ for every evaluated program. When h is evaluated on a new batch,

$$
Z ( h )  Z ( h ) \cup \{ z _ { t , 1 } ( h ) , \ldots , z _ { t , m _ { t } } ( h ) \} .
$$

The empirical score is recomputed from all entries in $Z ( h )$ . A candidate must contain at least $m _ { \mathrm { m i n } }$ stored losses before entering the elite archive; until then, its score is provisional and used only for within-generation ranking.

## 3.2 Empirical CVaR Fitness

The scoring module maps a loss vector $z ( h ) =$ $\left( z _ { 1 } , \ldots , z _ { m } \right)$ and tail level q to empirical CVaR fitness $\hat { F } _ { q } ( h )$ , along with the empirical tail threshold and identifiers of tail environments.

Let $\alpha = 1 - q$ . The empirical score is the sample analogue of Eq. (1):

$$
\widehat { F } _ { q } ( h ) = \operatorname* { m i n } _ { \eta \in \mathbb { R } } \left\{ \eta + \frac { 1 } { \alpha m } \sum _ { j = 1 } ^ { m } ( z _ { j } - \eta ) _ { + } \right\} .\tag{2}
$$

Since Eq. (2) is piecewise linear in $\eta ,$ an optimizer can be selected from observed losses. Let $z _ { ( 1 ) } \leq$ $\cdots \leq z _ { ( m ) }$ be sorted losses, $r = \alpha m , k = \lceil r \rceil$ and $\rho = k - r$ . The empirical CVaR is

$$
\widehat { F } _ { q } ( h ) = \frac { 1 } { r } \left( \sum _ { a = m - k + 2 } ^ { m } z _ { ( a ) } + ( 1 - \rho ) z _ { ( m - k + 1 ) } \right) ,\tag{3}
$$

where an empty sum is zero. If r is an integer, $\rho = 0$ , and Eq. (3) averages the largest r losses. If r is fractional, losses strictly above the boundary receive unit weight, and the boundary loss receives weight $1 - \rho .$

The empirical value-at-risk threshold is $\widehat { \eta } _ { q } ( h ) =$ $z _ { ( m - k + 1 ) }$ . The tail record stores indices of losses larger than $\widehat { \eta } _ { q } ( h )$ and enough boundary indices with value $\widehat { \eta } _ { q } ( h )$ to match the fractional boundary weight in Eq. (3). For each tail index, the method records generator identity, scale value, instance seed, execution seed, realized loss, and failure flag if present. These records are provided to the mutation operator as diagnostic context, while ranking depends only on $\widehat { F } _ { q } ( h )$ and deterministic tie-breaking keys.

Candidates are ordered by ascending empirical CVaR. For candidates $h _ { a }$ and $h _ { b } , h _ { a }$ precedes $h _ { b }$ if $\widehat { F } _ { q } ( h _ { a } ) < \widehat { F } _ { q } ( h _ { b } )$ . If their CVaR scores match, the empirical mean $\begin{array} { r } { \bar { z } ( h ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } z _ { j } } \end{array}$ is the first tie-breaking key; if means match, max<sub>j</sub> $z _ { j }$ is used. Remaining ties are broken by a deterministic codelevel key, such as creation order or source length. This procedure defines a total order over all scored candidates.

For an accumulated multiset $Z ( h )$ , the same computation is applied after sorting all stored losses. If $Z ( h )$ contains $m _ { h }$ entries, then $r _ { h } =$ $\alpha m _ { h }$ and $k _ { h } = \lceil r _ { h } \rceil$ , and the score is computed from the largest $k _ { h }$ sorted entries with the corresponding fractional boundary weight. A candidate with only penalty losses receives $\widehat F _ { q } ( h ) = \ell _ { \mathrm { f a i l } } .$ . A candidate with mixed successful and failed executions is scored by the sorted mixture, so failures enter the upper tail whenever their penalties are among the largest losses.

## 3.3 Program Mutation and Evolutionary Search

The search module maps scored programs to a new population. Its inputs at generation t are the current population $\mathcal { P } _ { t } .$ stored losses and scores for its members, and the mutation budget; its outputs are $\mathcal { P } _ { t + 1 }$ and an updated archive containing the lowestscoring program observed so far.

Each candidate is stored as source code implementing the required callable interface. The mutation operator receives the parent source code, interface signature, task specification, and optionally the parent’s stored tail records. It returns a complete source-code candidate $\tilde { h } .$ . The returned text is parsed to extract a single program body, compiled in the harness, and checked against the required callable signature. If validation fails, the offspring receives penalty losses for the current batch. If validation succeeds, the offspring is executed on sampled environments and scored by the empirical CVaR computation in Section 3.2.

<table><tr><td>Method</td><td colspan="2">Bin Packing</td><td colspan="2">CVRP</td><td colspan="2">Flow-Shop</td></tr><tr><td></td><td>Mean ↓</td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>Mean ↓</td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>Mean ↓</td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td></tr><tr><td>Handcrafted heuristic</td><td> $7 . 4 1 { \pm } 0 . 0 8$ </td><td> $1 1 . 7 9 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $8 . 1 2 { \pm } 0 . 1 0$ </td><td> $1 2 . 9 6 { \pm } 0 . 2 7 $ </td><td> $6 . 7 6 { \pm } 0 . 0 9$ </td><td> $1 0 . 8 4 { \pm } 0 . 2 1 $ </td></tr><tr><td>Direct LLM sampling</td><td> $7 . 0 3 { \pm } 0 . 1 3$ </td><td> $1 1 . 1 8 { \pm } 0 . 3 1 $ </td><td> $7 . 8 0 { \pm } 0 . 1 5 $ </td><td> $1 2 . 4 7 { \pm } 0 . 3 5 $ </td><td> $6 . 4 9 { \pm } 0 . 1 1$ </td><td> $1 0 . 2 9 { \pm } 0 . 2 8 $ </td></tr><tr><td>FunSearch-style</td><td> $6 . 6 6 { \pm } 0 . 0 9$ </td><td> $1 0 . 5 1 { \pm } 0 . 2 3 $ </td><td> $7 . 3 7 { \pm } 0 . 1 1$ </td><td> $1 1 . 6 2 { \pm } 0 . 2 6 $ </td><td> $6 . 1 6 { \pm } 0 . 0 8$ </td><td> $9 . 7 8 { \pm } 0 . 2 1 $ </td></tr><tr><td>EoH-style</td><td> $6 . 4 2 { \pm } 0 . 0 6 $ </td><td> $1 0 . 0 3 { \pm } 0 . 1 8$ </td><td> $\underline { { 7 . 1 8 \pm 0 . 0 8 } }$ </td><td> $1 1 . 2 1 { \pm } 0 . 2 1 $ </td><td> $5 . 9 9 { \pm } 0 . 0 5 $ </td><td> $9 . 3 9 { \pm } 0 . 1 6 $ </td></tr><tr><td>Mean-RMD</td><td> $6 . 5 6 { \pm } 0 . 0 6$ </td><td> $9 . 8 8 { \pm } 0 . 1 5 $ </td><td> $7 . 2 0 { \pm } 0 . 0 6 $ </td><td> $1 1 . 0 4 { \pm } 0 . 1 7 $ </td><td> ${ \bf 5 . 9 5 \pm 0 . 0 4 }$ </td><td> $9 . 2 6 { \pm } 0 . 1 3 $ </td></tr><tr><td>CVaR-RMD</td><td> ${ \bf 6 . 3 1 \pm 0 . 0 4 }$ </td><td> $\mathbf { 9 . 2 9 } 2 6 . 1 1$ </td><td> ${ \bf 7 . 0 8 \pm 0 . 0 5 }$ </td><td> $\mathbf { i 0 . 4 1 \pm 0 . 1 4 }$ </td><td> $5 . 9 6 { \pm } 0 . 0 4 $ </td><td> $\mathbf { 8 . 8 1 \pm 0 . 1 1 }$ </td></tr></table>

Table 1: Held-out performance across 15 discovery seeds. Values are mean ± standard deviation. Lower is better.
<table><tr><td>Method</td><td>Mean loss ↓</td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td> $\mathrm { M a x } \mathrm { l o s s } \downarrow$ </td><td>Failure rate (%) ↓</td></tr><tr><td>Mean-RMD</td><td> $7 . 5 9 { \pm } 0 . 0 8 $ </td><td> $1 2 . 6 1 { \pm } 0 . 2 4$ </td><td> $1 8 . 3 4 { \pm } 0 . 5 1 $ </td><td> $3 . 5 { \pm } 0 . 5 $ </td></tr><tr><td>CVaR-RMD</td><td> $\mathbf { 7 . 4 8 { \pm } 0 . 0 6 }$ </td><td> $\mathbf { 1 1 . 6 3 \pm 0 . 1 8 }$ </td><td> $\mathbf { 1 6 . 6 1 \pm 0 . 4 2 }$ </td><td> ${ \bf 2 . 0 { \pm } 0 . 3 }$ </td></tr></table>

Table 2: Transfer evaluation on larger held-out instances across 15 discovery seeds.

Parent selection uses the current ranking. Let $R _ { t }$ be the ordered list from sorting $\mathcal { P } _ { t } .$ The parent pool is $\mathcal { A } _ { t } ~ = ~ \{ R _ { t } [ 1 ] , \ldots , R _ { t } [ M ] \}$ with $M \leq N$ . To produce one offspring, parent rank $a \in \{ 1 , \dots , M \}$ is sampled with probability

$$
p _ { a } = \frac { \exp ( - \tau a ) } { \sum _ { b = 1 } ^ { M } \exp ( - \tau b ) } , \qquad \tau \geq 0 ,
$$

where τ controls selection pressure toward higherranked programs. The selected parent $h _ { p } = R _ { t } [ a ]$ is mutated to produce $\tilde { h } .$ . Repeated draws are allowed, so multiple offspring may share a parent within one generation.

After K offspring are produced, the pool is $\mathcal { C } _ { t } =$ $\mathcal { P } _ { t } \cup \mathcal { O } _ { t }$ , where $\mathcal { O } _ { t } = \overline { { \{ h _ { t } ^ { ( 1 ) } , \ldots , \tilde { h } _ { t } ^ { ( \bar { K } ) } \} } }$ . Every member of $\mathcal { C } _ { t }$ has an empirical CVaR score from execution or a penalty score from validation failure. The pool is sorted by the ranking rule, and

$$
\mathcal { P } _ { t + 1 } = \{ C _ { t } [ 1 ] , \dots , C _ { t } [ N ] \} ,
$$

where $C _ { t } [ a ]$ is the a-th ranked candidate in $\mathcal { C } _ { t } .$ . The archive stores the best program, or a bounded set of best programs, under the same ordering. After replacement, the archive is updated by comparing its contents with the highest-ranked member of $\mathcal { P } _ { t + 1 }$

The loop stops when the evaluation, generation, or execution-time budget is exhausted. The returned program is the lowest-scoring archived program under empirical CVaR and deterministic tie-breaking. At deployment, this program is executed directly through the same callable interface on a new environment input; no mutation step or language-model call is performed.

## 4 Experiments

## 4.1 Setup and Protocol

We evaluate whether tail-risk improves the robustness of LLM-discovered metaheuristics compared with mean-based selection. The benchmark suite covers online bin packing, capacitated vehicle routing (CVRP), and permutation flow-shop scheduling, exposing brittleness from online decisions, feasibility constraints, and sequencing interactions. CVaR-RMD and Mean-RMD use the same code interface, validator, mutation operator, population size, generation budget, archive mechanism, and held-out evaluation. They differ only in candidate ranking: empirical CVaR for CVaR-RMD and empirical mean loss for Mean-RMD.

Each method uses 15 independent discovery seeds, $\{ 0 , 1 , \ldots , 1 4 \}$ . After discovery, the best archived program from each seed is frozen and evaluated on shared held-out environments. We report the mean, standard deviation, and standardized effect sizes across discovery seeds. Significance is assessed by paired two-sided bootstrap tests with 10,000 resamples over archived programs and shared test environments. Unless noted otherwise, comparisons use Mean-RMD as the baseline.

The mutation model is GPT-4o with temperature 0.7, top-p 0.95, maximum output length 2048, and up to three retries for malformed code. Prompts contain the task definition, required function signature, parent program, validation constraints, and a compact summary of its high-loss cases. The model must return a complete Python function preserving the interface. Generated code runs in a sandbox with fixed package versions. LLM calls occur only during discovery.

<table><tr><td>Setting</td><td>Mean loss↓</td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>Failure rate (%) ↓</td></tr><tr><td>Batch size 32</td><td> $6 . 5 0 { \pm } 0 . 0 6$ </td><td> $9 . 9 1 { \pm } 0 . 1 7 $ </td><td> $2 . 0 { \pm } 0 . 3 $ </td></tr><tr><td>Batch size 64</td><td> ${ \bf 6 . 4 5 { \pm 0 . 0 4 } }$ </td><td> $9 . 5 0 { \pm } 0 . 1 1$ </td><td> $1 . 4 { \pm } 0 . 2 $ </td></tr><tr><td>Batch size 128</td><td> $6 . 4 3 { \pm } 0 . 0 4$ </td><td> $\mathbf { 9 . 4 2 \pm 0 . 1 0 }$ </td><td> ${ \bf 1 . 3 \pm 0 . 2 }$ </td></tr><tr><td>Failure penalty  $\ell _ { \mathrm { f a i l } } = 2 0$ </td><td> $6 . 4 4 { \pm } 0 . 0 5$ </td><td> $9 . 6 3 { \pm } 0 . 1 3 $ </td><td> $1 . 7 { \pm } 0 . 3 $ </td></tr><tr><td>Failure penalty  $\ell _ { \mathrm { f a i l } } = 3 0$ </td><td> $6 . 4 7 { \pm } 0 . 0 5$ </td><td> $9 . 5 1 { \pm } 0 . 1 2 $ </td><td> ${ \bf 1 . 3 \pm 0 . 2 }$ </td></tr></table>

Table 3: Sensitivity analysis across tasks using 15 discovery seeds.

![](images/27c165a1b1fec556f28b1a5b6d5528c57d935e37b494a503abd542d824bafe7b.jpg)

![](images/aa593a37aef56e3f5737c477c0fc09fb4cd981e5aab1f5416dda818fee970c2e.jpg)

![](images/42c872854c05f855394457f587e9ba9058d5520a3cebc6ae82f878adc21e3fb5.jpg)  
B32/B64/B128: batch size 32/64/128 F20/F30: $\ell _ { \mathrm { f a i l } } = 2 0 / 3 0$  
Figure 3: Sensitivity analysis across tasks using 15 discovery seeds. Error bars denote standard deviations; filled markers indicate the best value for each metric, including tied best values.

Training uses medium-scale instances, whereas held-out testing covers in-distribution and larger transfer instances. Bin packing uses uniform, truncated-normal, and heavy-tailed beta item distributions, with 100–300 discovery items and 400– 800 transfer items. CVRP uses uniform and clustered Euclidean generators, with 50–100 discovery customers and 150–300 transfer customers. Flowshop scheduling uses uniform, correlated, and bottleneck processing-time families, with 20–50 discovery jobs and 80–120 transfer jobs.

At each generation, all candidates are evaluated on a shared batch of 64 environments. Losses accumulate across generations until archived candidates have at least 512 evaluations, stabilizing tail estimates. The population has $N = 2 4$ programs, with $M = 8$ parents eligible for mutation and $K = 1 6$ offspring per generation. Each discovery run lasts 40 generations and yields 640 LLM-generated offspring per seed. The default tail level is $q = 0 . 9 0$ Execution is capped at 2 seconds per training instance and 10 seconds per transfer instance. Invalid code, interface violations, infeasible outputs, exceptions, and timeouts receive $\ell _ { \mathrm { f a i l } } = 2 5$

Losses are normalized by task-specific references: the continuous bin-capacity lower bound for bin packing, Clarke–Wright savings with deterministic 2-opt for CVRP, and NEH for flow-shop scheduling. Lower values are better. We report mean loss, empirical $\mathrm { C V a R _ { 0 . 9 0 } }$ , maximum loss, and failure rate. The Appendix provides per-task breakdowns and sensitivity analyses.

## 4.2 Baselines

We compare four baseline groups. Handcrafted heuristics comprise first-fit and best-fit for bin packing, Clarke–Wright savings and cheapest insertion for CVRP, and NEH with shortest-processing-time dispatching for flow-shop scheduling. Direct LLM sampling draws as many candidate programs as one discovery run without evolutionary replacement. FunSearch-style (Romera-Paredes et al., 2024) and EoH-style (Liu et al., 2024a) variants use the same infrastructure, generators, parsers, validators, budgets, and LLM settings as CVaR-RMD but retain mean-performance selection. Mean-RMD is the closest control, using the same evolutionary loop as CVaR-RMD but ranking candidates by empirical mean loss.

## 4.3 Held-Out Robustness

Table 1 shows that CVaR-RMD achieves the lowest $\mathrm { C V a R _ { 0 . 9 0 } }$ on all three tasks. Relative to Mean-

<table><tr><td>Method</td><td>Mean loss ↓</td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>Max loss ↓</td><td>Failure rate ↓</td></tr><tr><td>Mean-RMD</td><td>6.47</td><td>10.03</td><td>15.72</td><td>2.3%</td></tr><tr><td>Mean+Std-RMD</td><td>6.44</td><td>9.70</td><td>15.12</td><td>1.7%</td></tr><tr><td>Quantile-RMD</td><td>6.49</td><td>9.71</td><td>14.98</td><td>1.6%</td></tr><tr><td>Hard-Mining-RMD</td><td>6.46</td><td>9.66</td><td>14.81</td><td>1.5%</td></tr><tr><td>Max-RMD</td><td>6.61</td><td>9.62</td><td>14.32</td><td>1.3%</td></tr><tr><td>CVaR-RMD</td><td>6.45</td><td>9.50</td><td>14.55</td><td>1.4%</td></tr></table>

Table 4: Comparison with alternative risk-aware selection objectives. Values are equally weighted macro-averages across bin packing, CVRP, and flow-shop scheduling. All methods use matched candidate-generation and programexecution budgets. Lower is better.
<table><tr><td>External benchmark</td><td>Mean-RMD</td><td>CVaR-RMD</td><td>Relative change</td></tr><tr><td>OR-Library bin packing</td><td>12.41</td><td>11.63</td><td>-6.3%</td></tr><tr><td>CVRPLIB</td><td>14.62</td><td>13.31</td><td>-9.0%</td></tr><tr><td>Taillard flow shop</td><td>12.78</td><td>11.94</td><td>-6.6%</td></tr></table>

Table 5: Transfer performance on independently curated public benchmark instances. The metric is $\mathrm { C V a R _ { 0 . 9 0 } }$ , and lower values are better. Programs are frozen before external evaluation.

RMD, it reduces tail loss by 6.0% on bin packing, 5.7% on CVRP, and 4.9% on flow-shop scheduling. Paired bootstrap tests yield $p < 0 . 0 1$ for all tasks, with Cohen’s d values of 1.21, 1.08, and 0.94, respectively. These gains therefore represent a consistent shift in the discovered-program distribution rather than isolated favorable runs.

Mean-loss differences are smaller, as expected from the objective. CVaR-RMD improves mean loss by 3.8% on bin packing and 1.7% on CVRP, while leaving flow-shop mean loss statistically unchanged. Risk-sensitive selection thus mainly suppresses high-loss outcomes without sacrificing average performance. Direct LLM sampling underperforms all evolutionary variants, confirming the value of iterative evaluation and replacement. Among discovery methods, CVaR-RMD provides the strongest upper-tail robustness while remaining competitive in mean loss.

## 4.4 Transfer to Larger Instances

We freeze each discovered program and evaluate it on larger unseen instances with new random seeds. Instance families remain unchanged, but their scale parameters exceed the discovery range.

Table 2 shows that the robustness advantage persists under scale transfer. Relative to Mean-RMD, CVaR-RMD reduces transfer CVaR<sub>0.90</sub> by 7.8%, maximum loss by 9.4%, and failure rate by 1.5 percentage points, with all differences significant at $p < 0 . 0 2$ . The strongest effect occurs on CVRP, where mean-based discovery sometimes produces route-construction logic that becomes infeasible for larger customer sets. Flow-shop scheduling shows smaller average gains but still benefits from lower maximum loss.

## 4.5 Comparison with Alternative Risk-Aware Objectives

Mean-RMD controls the comparison between expectation-based and tail-risk selection but does not show whether CVaR is preferable to other robustness objectives. We therefore evaluate four additional risk-aware variants using the same LLM mutation model, population size, search budget, validation procedure, and program–environment execution budget as CVaR-RMD.

MEAN+STD-RMD ranks candidates by the sum of empirical mean loss and standard deviation. QUANTILE-RMD uses empirical $\mathrm { V a R _ { 0 . 9 0 } }$ , while MAX-RMD uses maximum observed loss. Finally, HARD-MINING-RMD increases the sampling probability of environments that previously produced high losses. These controls test whether the robustness gain is specific to CVaR or reflects risk-sensitive selection more broadly.

Table 4 reports equally weighted macro-averages across the three tasks. All methods use matched search and evaluation budgets. CVaR-RMD achieves the lowest $\mathrm { C V a R _ { 0 . 9 0 } }$ while remaining competitive in mean loss. Max-RMD achieves the lowest maximum loss and failure rate but substantially higher mean loss. The other risk-aware objectives also improve tail performance over Mean-RMD.

These results show that CVaR is not the only effective robustness objective. Max-RMD suppresses the worst observed outcome more aggressively and achieves the lowest failure rate, but with worse mean performance. Under the evaluated protocol, CVaR-RMD provides the best observed mean–tail trade-off. We therefore do not claim that CVaR universally dominates other risk measures. Instead, the results establish CVaR as an effective risk-sensitive alternative when average performance and uppertail robustness both matter.

<table><tr><td>Variant</td><td>Mean loss ↓</td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>Failure rate (%) ↓</td></tr><tr><td>Full CVaR-RMD</td><td> ${ \bf 6 . 4 5 { \pm 0 . 0 4 } }$ </td><td> $\mathbf { 9 . 5 0 { \pm } 0 . 1 1 }$ </td><td> $\mathbf { 1 . 4 \pm 0 . 2 }$ </td></tr><tr><td>Mean ranking</td><td> $6 . 4 7 { \pm } 0 . 0 5$ </td><td> $1 0 . 0 6 { \pm } 0 . 1 5 $ </td><td> $2 . 3 { \pm } 0 . 3 $ </td></tr><tr><td>No accumulated losses</td><td> $6 . 5 3 { \pm } 0 . 0 6$ </td><td> $1 0 . 2 2 { \pm } 0 . 1 7 $ </td><td> $2 . 7 { \pm } 0 . 4 $ </td></tr><tr><td>No tail records in prompt</td><td> $6 . 4 8 { \pm } 0 . 0 5$ </td><td> $9 . 8 1 { \pm } 0 . 1 4$ </td><td> $1 . 9 { \pm } 0 . 3 $ </td></tr><tr><td>Tail level  $q = 0 . 8 0$ </td><td> $6 . 4 8 { \pm } 0 . 0 5$ </td><td> $9 . 8 7 { \pm } 0 . 1 4$ </td><td> $1 . 8 { \pm } 0 . 3 $ </td></tr><tr><td>Tail level  $q = 0 . 9 5$ </td><td> $6 . 5 7 { \pm } 0 . 0 6$ </td><td> $9 . 5 6 { \pm } 0 . 1 3$ </td><td> $1 . 5 { \pm } 0 . 2 $ </td></tr></table>

Table 6: Ablation results averaged over tasks across 15 discovery seeds.

## 4.6 Transfer to Independently Curated Public Benchmarks

To reduce dependence on synthetic discovery generators, we also evaluate the frozen programs on independently curated public benchmark instances. None are used for program discovery, prompt construction, hyperparameter selection, or archive selection.

The external evaluation uses OR-Library instances for bin packing, CVRPLIB instances for capacitated vehicle routing, and Taillard instances for permutation flow-shop scheduling. We apply the same programs frozen after the original discovery runs and report their $\mathrm { C V a R _ { 0 . 9 0 } }$ on these instances.

Table 5 shows that CVaR-RMD reduces $\mathrm { C V a R _ { 0 . 9 0 } }$ on all three external benchmark collections: by 6.3% on OR-Library bin packing, 9.0% on CVRPLIB, and 6.6% on Taillard flow-shop instances. The largest reduction occurs on CVRPLIB, whose independently curated instances expose routing and capacity patterns differing from the synthetic discovery instances.

These results support within-domain transfer to independently curated benchmarks. They do not establish cross-domain transfer, robustness to arbitrary distribution shifts, or operational deployment performance. We therefore restrict the generalization claim to improved upper-tail robustness on unseen scales, instance seeds, and independent withindomain benchmarks under the evaluated protocol.

## 4.7 Ablations and Sensitivity

We ablate components distinguishing CVaR-RMD from the mean-based control while fixing prompts, validators, budgets, populations, and held-out tests.

Table 6 identifies CVaR ranking as the main improvement driver. Replacing it with mean ranking increases $\mathrm { C V a R _ { 0 . 9 0 } }$ by 5.9% and raises failure rate from 1.4% to 2.3%. Removing accumulated losses performs worse because fewer evaluations yield noisier tail estimates. Prompted tail-loss summaries provide a smaller additional benefit by informing mutations about adverse execution modes. Tail-level comparisons reveal the expected tradeoff: $q = 0 . 8 0$ offers less protection against rare failures, whereas $q = 0 . 9 5$ suppresses them more aggressively but weakens mean performance. The default $q = 0 . 9 0$ provides the best observed balance.

Table 3 and Figure 3 show that CVaR-RMD remains stable under moderate changes in evaluation batch size and failure penalty. Smaller batches increase tail-estimation variance, whereas larger batches improve robustness at higher evaluation cost. Increasing the failure penalty further discourages infeasible programs but mildly reduces search flexibility in mean loss. These results support the default configuration as a practical robustness-cost compromise.

## 5 Conclusion

We presented CVaR-RMD, a risk-sensitive framework that replaces mean-based fitness with CVaR selection in LLM-driven metaheuristic discovery. Across three combinatorial optimization tasks, CVaR-RMD improved upper-tail performance and reduced failure rates while maintaining competitive mean performance. These benefits persisted under within-domain transfer to larger unseen instances and independently curated benchmarks. The results show that tail-risk-based selection can produce more reliable heuristics under benchmark distribution shifts without adding architectural components to the discovery pipeline.

## Limitations

The empirical evaluation focuses on three representative combinatorial optimization tasks, and further validation on additional problem classes would help assess the generality of the approach. CVaR-RMD also introduces a robustness hyperparameter through the tail level q, and while a single default value performs consistently well in the experiments, the optimal choice may vary across applications depending on the desired trade-off between average and worst-case performance.

## LLM Usage Statement

During the preparation of this manuscript, large language models (LLM) were used to improve the grammar, wording, and overall readability of the text. They were not used to generate scientific content, conduct data analysis, or draw conclusions. All LLM-assisted revisions were carefully reviewed and approved by the authors.

## Acknowledgements

The work was supported by the National Natural Science Foundation of China (Grant No. 62471287)

## References

Carlos Ansótegui, Meinolf Sellmann, and Kevin Tierney. 2009. A gender-based genetic algorithm for the automatic configuration of algorithms. In Principles and Practice ofConstraint Programming – CP 2009, volume 5732 of Lecture Notes in Computer Science, pages 142–157. Springer.

Philippe Artzner, Freddy Delbaen, Jean-Marc Eber, and David Heath. 1999. Coherent measures of risk. Mathematical Finance, 9(3):203–228.

Thomas Bäck. 1996. Evolutionary Algorithms in Theory and Practice: Evolution Strategies, Evolutionary Programming, Genetic Algorithms. Oxford Univer sity Press.

Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, and Samy Bengio. 2017. Neural combinatorial optimization with reinforcement learning. In Proceedings of the International Conference on Learning Representations Workshop.

Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. 2009. Robust Optimization. Princeton University Press.

Dimitris Bertsimas, David B. Brown, and Constantine Caramanis. 2011. Theory and applications of robust optimization. SIAM Review, 53(3):464–501.

Hans-Georg Beyer and Bernhard Sendhoff. 2007. Robust optimization: A comprehensive survey. Computer Methods in Applied Mechanics and Engineering, 196(33–34):3190–3218.

Christian Blum and Andrea Roli. 2003. Metaheuristics in combinatorial optimization: Overview and conceptual comparison. ACM Computing Surveys, 35(3):268–308.

Jürgen Branke, Kalyanmoy Deb, Kaisa Miettinen, and Roman Słowinski, editors. 2008. ´ Multiobjective Optimization: Interactive and Evolutionary Approaches, volume 5252 of Lecture Notes in Computer Science. Springer.

Edmund K. Burke, Michel Gendreau, Matthew Hyde, Graham Kendall, Gabriela Ochoa, Ender Özcan, and Rong Qu. 2013. Hyper-heuristics: A survey of the state of the art. Journal ofthe Operational Research Society, 64(12):1695–1724.

Yinlam Chow, Aviv Tamar, Shie Mannor, and Marco Pavone. 2015. Risk-sensitive and robust decisionmaking: A CVaR optimization approach. In Advances in Neural Information Processing Systems, volume 28, pages 1522–1530.

Peter Cowling, Graham Kendall, and Eric Soubeiga. 2001. A hyperheuristic approach to scheduling a sales summit. In Practice and Theory of Automated Timetabling III, volume 2079 of Lecture Notes in Computer Science, pages 176–190. Springer.

Erick Delage and Yinyu Ye. 2010. Distributionally robust optimization under moment uncertainty with application to data-driven problems. Operations Research, 58(3):595–612.

John C. Duchi and Hongseok Namkoong. 2021. Learning models with uniform performance via distributionally robust optimization. The Annals ofStatistics, 49(3):1378–1406.

A. E. Eiben and J. E. Smith. 2003. Introduction to Evolutionary Computing. Natural Computing Series. Springer.

David E. Goldberg and Kalyanmoy Deb. 1991. A comparative analysis of selection schemes used in genetic algorithms. In Foundations of Genetic Algorithms, volume 1, pages 69–93. Morgan Kaufmann.

Holger H. Hoos and Thomas Stützle. 2004. Stochastic Local Search: Foundations and Applications. Morgan Kaufmann.

Frank Hutter, Holger H. Hoos, and Kevin Leyton-Brown. 2011. Sequential model-based optimization for general algorithm configuration. In Learning and Intelligent Optimization, Lecture Notes in Computer Science, pages 507–523. Springer.

Frank Hutter, Holger H. Hoos, Kevin Leyton-Brown, and Thomas Stützle. 2009. ParamILS: An automatic algorithm configuration framework. Journal ofArtificial Intelligence Research, 36:267–306.

Yaochu Jin and Jürgen Branke. 2005. Evolutionary optimization in uncertain environments: A survey. IEEE Transactions on Evolutionary Computation, 9(3):303–317.

David S. Johnson and Lyle A. McGeoch. 1997. The traveling salesman problem: A case study in local optimization. In Emile H. L. Aarts and Jan Karel Lenstra, editors, Local Search in Combinatorial Optimization, pages 215–310. Wiley.

Chaitanya K. Joshi, Thomas Laurent, and Xavier Bresson. 2019. An efficient graph convolutional network technique for the travelling salesman problem. arXiv preprint arXiv:1906.01227.

Wouter Kool, Herke van Hoof, and Max Welling. 2019. Attention, learn to solve routing problems! In Proceedings ofthe International Conference on Learning Representations.

John R. Koza. 1992. Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press.

William B. Langdon and Riccardo Poli. 2002. Foundations ofGenetic Programming. Springer.

Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. 2024a. Evolution of heuristics: Towards efficient automatic algorithm design using large language model. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 32201–32223. PMLR.

Tennison Liu, Nicolás Astorga, Nabeel Seedat, and Mihaela van der Schaar. 2024b. Large language models to enhance bayesian optimization. In Proceedings of the International Conference on Learning Representations.

Manuel López-Ibáñez, Jérémie Dubois-Lacoste, Leslie Pérez Cáceres, Thomas Stützle, and Mauro Birattari. 2016. The irace package: Iterated racing for automatic algorithm configuration. Operations Research Perspectives, 3:43–58.

Gabriela Ochoa, Matthew Hyde, Tim Curtois, Jose A. Vazquez-Rodriguez, James Walker, Michel Gendreau, Graham Kendall, Barry McCollum, Andrew J. Parkes, Sanja Petrovic, and Edmund K. Burke. 2012. HyFlex: A benchmark framework for cross-domain heuristic search. In Evolutionary Computation in Combinatorial Optimization, volume 7245 of Lecture Notes in Computer Science, pages 136–147. Springer.

Michael O’Neill and Conor Ryan. 2001. Grammatical evolution. IEEE Transactions on Evolutionary Computation, 5(4):349–358.

Georg Ch. Pflug. 2000. Some remarks on the value-atrisk and the conditional value-at-risk. In Stanislav P. Uryasev, editor, Probabilistic Constrained Optimization: Methodology and Applications, pages 272–281. Kluwer Academic Publishers.

Hamed Rahimian and Sanjay Mehrotra. 2019. Distributionally robust optimization: A review. arXiv preprint arXiv:1908.05659.

Gerhard Reinelt. 1991. TSPLIB – a traveling salesman problem library. ORSA Journal on Computing, 3(4):376–384.

John R. Rice. 1976. The algorithm selection problem. Advances in Computers, 15:65–118.

R. Tyrrell Rockafellar and Stanislav Uryasev. 2000. Optimization of conditional value-at-risk. Journal of Risk, 2(3):21–42.

R. Tyrrell Rockafellar and Stanislav Uryasev. 2002. Conditional value-at-risk for general loss distributions. Journal of Banking and Finance, 26(7):1443– 1471.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. 2024. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475.

Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczynski. 2009.´ Lectures on Stochastic Programming: Modeling and Theory. SIAM.

El-Ghazali Talbi. 2009. Metaheuristics: From Design to Implementation. Wiley Series on Parallel and Distributed Computing. Wiley.

Aviv Tamar, Yonatan Glassner, and Shie Mannor. 2015. Optimizing the CVaR via sampling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29.

Niki van Stein and Thomas Bäck. 2024. LLaMEA: A large language model evolutionary algorithm for automatically generating metaheuristics. IEEE Transactions on Evolutionary Computation.

Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer networks. In Advances in Neural Information Processing Systems, volume 28.

Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, and Guojie Song. 2024. ReEvo: Large language models as hyper-heuristics with reflective evolution. In Advances in Neural Information Processing Systems, volume 37.

## Appendix

This appendix provides supplementary material for CVaR-RMD. Appendix A describes the three benchmark tasks (bin packing, CVRP, flow-shop scheduling), instance distributions, and the heldout evaluation protocol across 15 discovery seeds. Appendix B details the empirical CVaR computation, population management, archive mechanism, and the shared code interface with Mean-RMD. Appendix C reports full tail-risk and meanperformance tables, sensitivity analyses on the CVaR confidence level, and per-seed variance statistics. Appendix D provides the prompt templates used for candidate generation and mutation in the LLM-driven discovery loop. Appendix E illustrates specific discovered metaheuristics where CVaR selection avoids brittle solutions that score well on average but fail on tail instances.

## A Environment Details

## A.1 Datasets and Task Settings

We evaluate LLM-discovered metaheuristics on three combinatorial optimization benchmarks: online bin packing, capacitated vehicle routing, and permutation flow-shop scheduling. These tasks are selected to cover online allocation decisions, hard feasibility constraints, and long-horizon sequencing interactions.

For online bin packing, instances consist of sequentially arriving items that must be assigned to bins of fixed capacity. Discovery uses item sequences with 100–300 items drawn from uniform, truncated-normal, and heavy-tailed beta distributions, while transfer evaluation uses larger sequences with 400–800 items.

For CVRP, each instance contains Euclidean customer locations, customer demands, a depot, and a vehicle-capacity constraint. Discovery uses uniform and clustered generators with 50–100 customers, and held-out transfer evaluation uses 150–300 customers generated with unseen random seeds.

For flow-shop scheduling, each instance specifies processing times for a set of jobs across machines, and the objective is to construct a job permutation with low makespan. Discovery uses 20– 50 jobs from uniform, correlated, and bottleneck processing-time families, and transfer evaluation uses larger instances with 80–120 jobs.

All tasks are formulated as program-synthesis problems in which the LLM proposes a complete

Python heuristic satisfying a fixed function signature. Prompts include the task definition, interface requirements, validation constraints, the parent program, and a compact summary of high-loss cases. No held-out test instances are used for discovery or prompt construction. Hyperparameters are fixed before final evaluation, and the same discovery and evaluation splits are used for CVaR-RMD, Mean-RMD, and all implemented baselines.

## A.2 Preprocessing and Evaluation

Generated programs are parsed as Python functions and checked against the required interface before execution. We reject malformed code, missing functions, invalid return types, infeasible solutions, runtime exceptions, and timeouts. Such cases receive the fixed failure loss $\ell _ { \mathrm { f a i l } } = 2 5$ . Candidate execution is capped at 2 seconds per training instance and 10 seconds per transfer instance.

Losses are normalized by task-specific reference procedures to make results comparable across instance families and scales. Bin-packing losses are normalized by the continuous bin-capacity lower bound, CVRP losses by Clarke–Wright savings with deterministic 2-opt, and flow-shop losses by the NEH heuristic. Lower values indicate better performance.

At each generation, candidates are evaluated on a shared batch of 64 environments. Loss histories are accumulated across generations until archived candidates have at least 512 evaluations, which is used to stabilize empirical tail estimates. CVaR-RMD ranks candidates by empirical $\mathrm { C V a R } _ { q }$ with default tail level $q = 0 . 9 0$ , while Mean-RMD ranks candidates by empirical mean loss. For a loss sample $\{ \ell _ { i } \} _ { i = 1 } ^ { n }$ , empirical $\mathrm { C V a R _ { 0 . 9 0 } }$ is computed as the average loss over the largest 10% of observed losses.

Each method is run with 15 independent discovery seeds, {0, 1, . . . , 14}. After discovery, the best archived program from each seed is frozen and evaluated on common held-out environments. We report mean loss, empirical $\mathrm { C V a R _ { 0 . 9 0 } }$ , maximum loss, and failure rate. Aggregate results are reported as mean, standard deviation, and standardized effect size across discovery seeds. Statistical significance is assessed using paired two-sided bootstrap tests with 10,000 resamples over archived programs and shared test environments.

Baselines include handcrafted heuristics, direct LLM sampling, FunSearch-style discovery, EoHstyle discovery, and Mean-RMD. All discovery baselines use the same generators, parsers, validators, mutation budget, population size, archive mechanism, and held-out evaluation protocol.

## B Implementation Details

All methods are implemented in a common Python infrastructure with fixed package versions and sandboxed execution for generated programs. CVaR-RMD and Mean-RMD share the same evolutionary loop and differ only in the candidate-ranking statistic. The population contains $N = 2 4$ programs, $M = 8$ parents are eligible for mutation at each generation, and each generation produces $K = 1 6$ offspring. Each discovery run lasts 40 generations, yielding 640 LLM-generated offspring per seed and at most 664 evaluated candidate programs per run.

The mutation model is GPT-4o. We use temperature 0.7, top-p 0.95, maximum output length 2048, and up to three retries for malformed code. The model is used only during discovery. Final heldout evaluation executes frozen programs without additional LLM calls.

The default CVaR-RMD configuration uses tail level $q = 0 . 9 0$ , batch size 64, accumulated loss histories, tail-case summaries in prompts, and failure penalty $\ell _ { \mathrm { f a i l } } = 2 5$ . Sensitivity experiments vary the batch size, tail level, and failure penalty while keeping all other components fixed.

Experiments are run on an AMD EPYC 7543 workstation with 256GB RAM. Average wall-clock times for CVaR-RMD are 5.9 hours for bin packing, 8.6 hours for CVRP, and 7.3 hours for flow-shop scheduling. Mean-RMD uses the same number of LLM calls, generated offspring, validation checks, and candidate executions; its runtime is within 4% of CVaR-RMD. The additional cost of CVaR-RMD is limited to sorting accumulated loss vectors for empirical tail-risk estimation.

## C More Experiment Results

## C.1 Per-Task Transfer Results

Table 7 decomposes the large-instance transfer results by task. CVaR-RMD consistently reduces tail loss and maximum loss across all three benchmarks. The largest relative reduction in $\mathrm { C V a R _ { 0 } } .$ .90 appears on CVRP, where larger customer sets expose more feasibility and route-construction failures under mean-based selection.

## C.2 Robustness Across Instance Families

Table 8 reports held-out performance by instance family. The advantage of CVaR-RMD is more pronounced on heavy-tailed, clustered, and bottleneck families, which contain more adverse cases than the corresponding uniform generators. This supports the interpretation that CVaR-based selection primarily improves robustness under difficult distributional conditions.

## C.3 Alternative Tail Levels at Evaluation Time

Table 9 evaluates the frozen programs with different tail levels. Although CVaR-RMD is selected using $q = 0 . 9 0 \AA$ , its advantage remains visible at both milder and more extreme tail levels. The improvement is largest at $q = 0 . 9 5$ , indicating that the selected programs reduce severe adverse outcomes rather than only optimizing the specific training quantile.

## C.4 Discovery Progress

Table 10 reports the held-out performance of the best archived program selected after different discovery budgets. CVaR-RMD improves steadily with more generations and remains better than Mean-RMD in tail loss throughout the later stages of discovery. Mean loss converges more closely, consistent with the main observation that the primary benefit of CVaR-RMD is reduced upper-tail risk.

## C.5 Run-to-Run Stability

Table 11 reports dispersion statistics over discovery seeds. CVaR-RMD reduces the interquartile range and worst-seed loss on all tasks, showing that the improvement is not only due to better average seed performance. The reduction in worst-seed $\mathrm { C V a R _ { 0 . 9 0 } }$ is particularly clear on CVRP.

## C.6 Statistical Comparison

Table 12 provides additional paired comparisons between CVaR-RMD and Mean-RMD. The effect sizes are larger for $\mathrm { C V a R _ { 0 . 9 0 } }$ than for mean loss, matching the intended behavior of the risk-sensitive objective.

## D Prompt Templates

## D.1 Overview

This appendix documents the prompt templates used for LLM-guided program discovery. The prompts are used only during discovery to generate or mutate executable heuristic programs. After discovery, archived programs are frozen and evaluated directly through the prescribed callable interface without further LLM calls. All prompts require the model to return complete Python source code satisfying the task interface; validation, feasibility checking, loss computation, and model comparison are performed by deterministic code rather than by an LLM judge.

<table><tr><td>Task</td><td>Method</td><td> $\mathrm { M e a n } \log \downarrow$ </td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td> $\mathrm { M a x } \mathrm { l o s s } \downarrow$ </td><td>Failure rate (%) ↓</td></tr><tr><td rowspan="2">Bin Packing</td><td>Mean-RMD</td><td> $7 . 1 8 { \pm } 0 . 0 9$ </td><td> $1 1 . 8 2 { \pm } 0 . 2 7 $ </td><td> $1 7 . 4 1 { \pm } 0 . 5 5 $ </td><td> $2 . 8 { \pm } 0 . 5 $ </td></tr><tr><td>CVaR-RMD</td><td> ${ \bf 7 . 0 8 } { \pm } { \bf 0 . 0 7 }$ </td><td> $\mathbf { 1 0 . 9 6 { \pm } 0 . 2 0 }$ </td><td> $\mathbf { 1 5 . 9 2 \pm 0 . 4 4 }$ </td><td> ${ \bf 1 . 7 \pm 0 . 3 }$ </td></tr><tr><td rowspan="2">CVRP</td><td>Mean-RMD</td><td> $8 . 3 4 { \pm } 0 . 1 1$ </td><td> $1 3 . 9 1 { \pm } 0 . 3 1 $ </td><td> $2 0 . 4 6 { \pm } 0 . 6 8$ </td><td> $4 . 9 { \pm } 0 . 7 $ </td></tr><tr><td>CVaR-RMD</td><td> $\mathbf { 8 . 1 8 \pm 0 . 0 9 }$ </td><td> $\mathbf { 1 2 . 6 9 { \pm 0 . 2 4 } }$ </td><td> $\mathbf { 1 8 . 1 2 \pm 0 . 5 3 }$ </td><td> ${ \bf 2 . 8 \pm 0 . 5 }$ </td></tr><tr><td rowspan="2">Flow-Shop</td><td>Mean-RMD</td><td> $7 . 2 5 { \pm } 0 . 0 8 $ </td><td> $1 2 . 1 0 { \pm } 0 . 2 4 $ </td><td> $1 7 . 1 6 { \pm } 0 . 4 9$ </td><td> $2 . 7 { \pm } 0 . 4 $ </td></tr><tr><td>CVaR-RMD</td><td> ${ \bf 7 . 1 9 \pm 0 . 0 7 }$ </td><td> $\mathbf { 1 1 . 2 5 { \pm 0 . 1 9 } }$ </td><td> $\mathbf { 1 5 . 7 8 { \scriptstyle \pm 0 . 4 1 } }$ </td><td> ${ \bf 1 . 6 \pm 0 . 3 }$ </td></tr></table>

Table 7: Per-task transfer results on larger held-out instances across 15 discovery seeds. Values are mean ± standard deviation.
<table><tr><td>Task</td><td>Inst. family</td><td>Method</td><td> $\mathrm { M e a n } \log \downarrow$ </td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>Failure rate (%) ↓</td></tr><tr><td rowspan="5">Bin Packing</td><td>Uniform</td><td>Mean-RMD</td><td> $5 . 9 4 { \pm } 0 . 0 5$ </td><td> $8 . 8 4 { \pm } 0 . 1 4$ </td><td> $1 . 3 { \pm } 0 . 2 $ </td></tr><tr><td>Truncated normal</td><td>CVaR-RMD Mean-RMD</td><td> ${ \bf 5 . 9 0 { \pm } 0 . 0 4 }$   $6 . 2 1 { \pm } 0 . 0 6$ </td><td> $\mathbf { 8 . 4 2 \pm 0 . 1 1 }$   $9 . 4 7 { \pm } 0 . 1 6$ </td><td> ${ \bf 0 . 9 \pm 0 . 2 }$   $1 . 7 { \pm } 0 . 3 $ </td></tr><tr><td></td><td>CVaR-RMD</td><td> ${ \bf 6 . 1 5 \pm 0 . 0 5 }$ </td><td> ${ \bf 8 . 8 9 \pm 0 . 1 3 }$ </td><td> ${ \bf 1 . 1 \pm 0 . 2 }$ </td></tr><tr><td rowspan="2">Heavy-tailed beta</td><td>Mean-RMD</td><td> $6 . 9 2 { \pm } 0 . 0 8$ </td><td> $1 1 . 3 3 { \pm } 0 . 2 2 $ </td><td> $2 . 9 { \pm } 0 . 4 $ </td></tr><tr><td>CVaR-RMD</td><td> ${ \bf 6 . 8 7 \pm 0 . 0 7 }$ </td><td> $\mathbf { 1 0 . 5 5 \pm 0 . 1 7 }$ </td><td> ${ \bf 1 . 8 \pm 0 . 3 }$ </td></tr><tr><td rowspan="4">CVRP</td><td>Uniform</td><td>Mean-RMD</td><td> $6 . 8 2 { \pm } 0 . 0 7$ </td><td> $1 0 . 3 8 { \pm } 0 . 1 7 $ </td><td> $1 . 8 { \pm } 0 . 3 $ </td></tr><tr><td rowspan="2">Clustered</td><td>CVaR-RMD</td><td> ${ \bf 6 . 7 7 \pm 0 . 0 6 }$ </td><td> $\mathbf { 9 . 8 4 \pm 0 . 1 4 }$ </td><td> $\mathbf { 1 . 2 \pm 0 . 2 }$ </td></tr><tr><td>Mean-RMD</td><td> $7 . 4 6 { \pm } 0 . 0 8$ </td><td> $1 1 . 7 0 { \pm } 0 . 2 2 $ </td><td> $3 . 1 { \pm } 0 . 4 $ </td></tr><tr><td></td><td>CVaR-RMD</td><td> ${ \bf 7 . 3 8 \pm 0 . 0 7 }$ </td><td> $\mathbf { 1 0 . 9 4 \pm 0 . 1 8 }$ </td><td> ${ \bf 1 . 9 { \pm } 0 . 3 }$ </td></tr><tr><td rowspan="6">Flow-Shop</td><td rowspan="2">Uniform</td><td>Mean-RMD</td><td> ${ \bf 5 . 6 7 \pm 0 . 0 4 }$ </td><td> $8 . 6 2 { \pm } 0 . 1 2$ </td><td> $1 . 0 { \pm } 0 . 2 $ </td></tr><tr><td>CVaR-RMD</td><td> $5 . 6 8 { \pm } 0 . 0 4$ </td><td> $\mathbf { 8 . 2 8 \pm 0 . 1 0 }$ </td><td> ${ \bf 0 . 7 \pm 0 . 1 }$ </td></tr><tr><td rowspan="2">Correlated</td><td>Mean-RMD</td><td> ${ \bf 5 . 8 9 \pm 0 . 0 5 }$ </td><td> $9 . 1 4 { \pm } 0 . 1 4$ </td><td> $1 . 3 { \pm } 0 . 2 $ </td></tr><tr><td>CVaR-RMD</td><td> $5 . 9 0 { \pm } 0 . 0 4$ </td><td> $\mathbf { 8 . 7 1 \pm 0 . 1 2 }$ </td><td> ${ \bf 0 . 9 \pm 0 . 2 }$ </td></tr><tr><td rowspan="2">Bottleneck</td><td>Mean-RMD</td><td> $6 . 2 9 { \pm } 0 . 0 6$ </td><td> $1 0 . 0 2 { \pm } 0 . 1 8$ </td><td> $2 . 2 { \pm } 0 . 3 $ </td></tr><tr><td>CVaR-RMD</td><td> ${ \bf 6 . 2 7 \pm 0 . 0 5 }$ </td><td> $\mathbf { 9 . 4 4 \pm 0 . 1 5 }$ </td><td> ${ \bf 1 . 5 \pm 0 . 2 }$ </td></tr></table>

Table 8: Held-out robustness by instance family across 15 discovery seeds.

## D.2 Notation

We use the following notation in prompt templates.

• T denotes the optimization task, one of online bin packing, capacitated vehicle routing, or permutation flow-shop scheduling.

• {TASK\_SPECIFICATION} is a naturallanguage specification of T , including the objective, feasibility constraints, input format, and output format.

• {FUNCTION\_SIGNATURE} is the required Python callable signature. Every generated program must implement exactly this interface.

• {VALIDATION\_CONSTRAINTS} lists syntactic, interface, runtime, and feasibility requirements enforced by the sandbox validator.

• {PARENT\_PROGRAM} is the complete source code of the selected parent heuristic.

• {PARENT\_SCORE} is the scalar fitness value used for ranking the parent. In CVaR-RMD this is empirical $\mathrm { C V a R } _ { q } ;$ in mean-based variants it is empirical mean loss.

<table><tr><td>Method</td><td colspan="2">Bin Packing</td><td colspan="2">CVRP</td><td colspan="2">Flow-Shop</td></tr><tr><td></td><td> $\mathrm { C V a R } _ { 0 . 8 0 \downarrow }$ </td><td> $\mathrm { C V a R _ { 0 . 9 5 } \downarrow }$ </td><td> $\mathrm { C V a R } _ { 0 . 8 0 \downarrow }$ </td><td> $\mathrm { C V a R _ { 0 . 9 5 } \downarrow }$ </td><td> $\mathrm { C V a R } _ { 0 . 8 0 \downarrow }$ </td><td> $\mathrm { C V a R _ { 0 . 9 5 } \downarrow }$ </td></tr><tr><td>Mean-RMD</td><td> $8 . 7 1 { \pm } 0 . 1 2$ </td><td> $1 0 . 8 7 { \pm } 0 . 1 9$ </td><td> $9 . 7 4 { \pm } 0 . 1 4$ </td><td> $1 2 . 2 9 { \pm } 0 . 2 3 $ </td><td> $8 . 1 8 { \pm } 0 . 1 1$ </td><td> $1 0 . 2 1 { \pm } 0 . 1 7 $ </td></tr><tr><td>CVaR-RMD</td><td> $\mathbf { 8 . 2 9 \pm 0 . 0 9 }$ </td><td> $\mathbf { 1 0 . 0 6 \pm 0 . 1 5 }$ </td><td> $\mathbf { 9 . 2 5 { \pm 0 . 1 2 } }$ </td><td> $\mathbf { 1 1 . 4 7 \pm 0 . 2 0 }$ </td><td> $\mathbf { 7 . 8 4 \pm 0 . 0 9 }$ </td><td> $\mathbf { 9 . 6 1 } \pm \mathbf { 0 . 1 4 }$ </td></tr></table>

Table 9: Evaluation with alternative CVaR tail levels across 15 discovery seeds. Programs are the same frozen programs used in the main held-out evaluation.
<table><tr><td>Generation</td><td colspan="3">Mean-RMD</td><td colspan="3"> $\mathrm { C V a R – R M D }$ </td></tr><tr><td></td><td>Mean loss ↓</td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>Failure rate (%) ↓</td><td> $\mathrm { M e a n } \log \downarrow$ </td><td> $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>Failure rate (%) ↓</td></tr><tr><td>10</td><td> $6 . 7 8 { \pm } 0 . 0 8$ </td><td> $1 0 . 7 1 { \pm } 0 . 1 9$ </td><td> $3 . 0 { \pm } 0 . 4 $ </td><td> ${ \bf 6 . 7 2 { \pm } 0 . 0 7 }$ </td><td> $\mathbf { 1 0 . 2 9 } { \pm } \mathbf { 0 . 1 7 }$ </td><td> $\mathbf { 2 . 5 { \pm } 0 . 4 }$ </td></tr><tr><td>20</td><td> $6 . 5 9 { \pm } 0 . 0 6 $ </td><td> $1 0 . 3 2 { \pm } 0 . 1 7 $ </td><td>2.6±0.4</td><td> ${ \bf 6 . 5 4 \pm 0 . 0 6 }$ </td><td> $\mathbf { 9 . 8 3 \pm 0 . 1 4 }$ </td><td> ${ \bf 1 . 9 \pm 0 . 3 }$ </td></tr><tr><td>30</td><td> $6 . 5 1 { \pm } 0 . 0 5$ </td><td> $1 0 . 1 2 { \pm } 0 . 1 6$ </td><td>2.4±0.3</td><td> ${ \bf 6 . 4 6 { \pm } 0 . 0 5 }$ </td><td> $\mathbf { 9 . 6 0 { \pm 0 . 1 2 } }$ </td><td> ${ \bf 1 . 6 \pm 0 . 3 }$ </td></tr><tr><td>40</td><td> $6 . 4 7 { \pm } 0 . 0 5$ </td><td> $1 0 . 0 3 { \pm } 0 . 1 5 $ </td><td>2.3±0.3</td><td> ${ \bf 6 . 4 2 \pm 0 . 0 4 }$ </td><td> $\mathbf { 9 . 4 8 \pm 0 . 1 1 }$ </td><td> $\mathbf { 1 . 4 \pm 0 . 2 }$ </td></tr></table>

Table 10: Discovery progress averaged across tasks and 15 discovery seeds.

• {MEAN\_LOSS}, {CVAR\_LOSS}, {MAX\_LOSS}, and {FAILURE\_RATE} are deterministic summaries computed from the parent’s stored losses.

• {TAIL\_LEVEL} is the CVaR tail level $q .$

• {TAIL\_RECORDS} is a compact deterministic summary of high-loss environments for the parent, including generator identity, scale value, instance seed, execution seed, realized loss, and failure flag.

• {REFERENCE\_HEURISTICS} optionally contains names of classical heuristics used only as task context, such as first-fit decreasing, best-fit, Clarke–Wright savings, cheapest insertion, NEH, or shortest-processing-time dispatching.

• {ARCHIVE\_PROGRAMS} denotes previously discovered high-performing programs when a baseline variant exposes archive exemplars.

## D.3 Main Inference Prompt

Purpose. This prompt implements the mutation operator in CVaR-RMD. It receives a parent program, the required task interface, validation constraints, and high-loss diagnostic records. It asks the model to produce a complete mutated Python function that preserves the interface while improving robustness on adverse cases.

Inputs. {TASK\_SPECIFICATION},

{FUNCTION\_SIGNATURE},

{VALIDATION\_CONSTRAINTS},

{PARENT\_PROGRAM}, {PARENT\_SCORE}, {MEAN\_LOSS}, {CVAR\_LOSS}, {MAX\_LOSS}, {FAILURE\_RATE}, {TAIL\_LEVEL}, and {TAIL\_RECORDS}.

Output. A single complete Python function implementing {FUNCTION\_SIGNATURE}. The output must contain code only, with no Markdown fences, no explanatory text, and no additional top-level functions unless they are nested inside the required function.

## D.4 Baseline Prompts

Purpose. Direct LLM sampling uses this prompt to generate candidate programs without evolutionary replacement. The same task interface, validation constraints, model settings, and output requirements are used as in CVaR-RMD, but no parent program or tail records are provided.

Inputs. {TASK\_SPECIFICATION},

{FUNCTION\_SIGNATURE},

{VALIDATION\_CONSTRAINTS}, and optionally {REFERENCE\_HEURISTICS}.

Output. A single complete Python function implementing {FUNCTION\_SIGNATURE}.

Purpose. Mean-RMD uses the same mutation prompt as CVaR-RMD, because the two methods share the code interface, validator, mutation operator, population size, generation budget, archive mechanism, and held-out evaluation. The only algorithmic difference is the ranking score used for selection. In the prompt, {PARENT\_SCORE} is populated with empirical mean loss rather than empirical CVaR.

<table><tr><td>Task</td><td>Method</td><td>Median  $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td><td>IQR↓</td><td>Worst-seed CVaR0.90 ↓</td><td>Best-seed  $\mathrm { C V a R _ { 0 . 9 0 } \downarrow }$ </td></tr><tr><td rowspan="2">Bin Packing</td><td>Mean-RMD</td><td>9.86</td><td>0.21</td><td>10.17</td><td>9.61</td></tr><tr><td>CVaR-RMD</td><td>9.28</td><td>0.15</td><td>9.52</td><td>9.09</td></tr><tr><td rowspan="2">CVRP</td><td>Mean-RMD</td><td>11.02</td><td>0.25</td><td>11.41</td><td>10.76</td></tr><tr><td>CVaR-RMD</td><td>10.40</td><td>0.19</td><td>10.72</td><td>10.18</td></tr><tr><td rowspan="2">Flow-Shop</td><td>Mean-RMD</td><td>9.25</td><td>0.18</td><td>9.52</td><td>9.06</td></tr><tr><td>CVaR-RMD</td><td>8.80</td><td>0.15</td><td>9.05</td><td>8.63</td></tr></table>

Table 11: Run-to-run stability of held-out $\mathrm { C V a R _ { 0 . 9 0 } }$ across 15 discovery seeds.
<table><tr><td>Metric</td><td>Task</td><td>Mean-RMD</td><td>CVaR-RMD</td><td>Relative change</td><td>Cohen&#x27;s d</td></tr><tr><td rowspan="3">Mean loss</td><td>Bin Packing</td><td> $6 . 5 6 { \pm } 0 . 0 6$ </td><td> ${ \bf 6 . 3 1 \pm 0 . 0 4 }$ </td><td>-3.8%</td><td>0.58</td></tr><tr><td>CVRP</td><td> $7 . 2 0 { \pm } 0 . 0 6 $ </td><td> ${ \bf 7 . 0 8 { \pm } 0 . 0 5 }$ </td><td>-1.7%</td><td>0.64</td></tr><tr><td>Flow-Shop</td><td> ${ \bf 5 . 9 5 \pm 0 . 0 4 }$ </td><td> $5 . 9 6 { \pm } 0 . 0 4$ </td><td>+0.2%</td><td>-0.18</td></tr><tr><td rowspan="3"> $\mathrm { C V a R _ { 0 . 9 0 } }$ </td><td>Bin Packing</td><td> $9 . 8 8 { \pm } 0 . 1 5 $ </td><td> ${ \bf 9 . 2 9 { \pm 0 . 1 1 } }$ </td><td>-6.0%</td><td>1.21</td></tr><tr><td>CVRP</td><td> $1 1 . 0 4 { \pm } 0 . 1 7$ </td><td> $\mathbf { 1 0 . 4 1 \pm 0 . 1 4 }$ </td><td>-5.7%</td><td>1.08</td></tr><tr><td>Flow-Shop</td><td> $9 . 2 6 { \pm } 0 . 1 3 $ </td><td> $\mathbf { 8 . 8 1 \pm 0 . 1 1 }$ </td><td>-4.9%</td><td>0.94</td></tr><tr><td rowspan="3">Failure rate</td><td>Bin Packing</td><td> $1 . 9 { \pm } 0 . 3 $ </td><td> ${ \bf 1 . 1 \pm 0 . 2 }$ </td><td>−0.8 pp</td><td>0.82</td></tr><tr><td>CVRP</td><td> $2 . 8 { \pm } 0 . 4 $ </td><td> ${ \bf 1 . 7 \pm 0 . 3 }$ </td><td>-1.1 pp</td><td>0.87</td></tr><tr><td>Flow-Shop</td><td> $2 . 1 { \pm } 0 . 3 $ </td><td> ${ \bf 1 . 3 \pm 0 . 2 }$ </td><td>−0.8 pp</td><td>0.76</td></tr></table>

Table 12: Additional statistical comparison between Mean-RMD and CVaR-RMD across 15 discovery seeds. Relative change is computed with respect to Mean-RMD; pp denotes percentage points.

Inputs. The same inputs as Figure 4, with {PARENT\_SCORE} equal to empirical mean loss.

Output. A single complete Python function implementing {FUNCTION\_SIGNATURE}.

Purpose. FunSearch-style and EoH-style baselines are implemented in the same infrastructure and use mean-performance selection. When archive exemplars are exposed, the prompt asks the model to synthesize a new complete program from high-performing previously discovered programs while preserving the same callable interface.

Inputs. {TASK\_SPECIFICATION},

{FUNCTION\_SIGNATURE},

{VALIDATION\_CONSTRAINTS},

{ARCHIVE\_PROGRAMS},

{MAX\_LOSS}, and {FAILURE\_RATE}.

Output. A single complete Python function implementing {FUNCTION\_SIGNATURE}.

## D.5 Few-shot / Demonstration Prompts

No few-shot natural-language demonstrations are used in the default CVaR-RMD prompt. Candidate source programs may appear as parent programs or archive programs in mutation prompts, but they are executable program inputs rather than labeled few-shot examples. Therefore, the discovery protocol does not include a separate few-shot prompt template.

## D.6 Reasoning or Rationale Prompts

No separate chain-of-thought, rationale-generation, or self-explanation prompt is used. The model is instructed to return code only. Diagnostic information from high-loss environments is computed deterministically by the evaluation harness and inserted into the mutation prompt as {TAIL\_RECORDS}.

## D.7 Data Generation or Pseudo-labeling Prompts

No LLM prompt is used for data generation or pseudo-labeling. Training and held-out environments are generated by deterministic benchmark generators from stored generator identities, scale values, instance seeds, and execution seeds. Feasibility labels, losses, failure flags, and tail records are produced by the sandbox validator and scoring code.

![](images/e195339c9fa0fc2b67b078420bf9cc33f64625d796335874e097ea21b48dde7c.jpg)  
Figure 4: Main CVaR-RMD mutation prompt.

## D.8 Evaluation Prompts

No LLM-as-a-judge prompt is used. Program validation, feasibility checking, objective evaluation, loss normalization, empirical mean computation, empirical CVaR<sub>q</sub> computation, maximumloss computation, and failure-rate computation are deterministic.

## D.9 Ablation Prompt Variants

Purpose. The “no tail records in prompt” ablation removes high-loss diagnostic cases from the mutation prompt while keeping all other prompts, validators, budgets, populations, and held-out tests fixed.

Inputs. {TASK\_SPECIFICATION},   
{FUNCTION\_SIGNATURE},   
{VALIDATION\_CONSTRAINTS},   
{PARENT\_PROGRAM}, {PARENT\_SCORE}, {MEAN\_LOSS}, {CVAR\_LOSS}, {MAX\_LOSS}, {FAILURE\_RATE}, and {TAIL\_LEVEL}.

Output. A single complete Python function implementing {FUNCTION\_SIGNATURE}.

Purpose. Tail-level ablations with q = 0.80 and q = 0.95 use the main mutation prompt without changing the prompt structure. Only the value substituted into {TAIL\_LEVEL} and the deterministic CVaR score used for ranking are changed.

Inputs. The same inputs as Figure 4, with {TAIL\_LEVEL} set to either 0.80 or 0.95.

Output. A single complete Python function implementing {FUNCTION\_SIGNATURE}.

Purpose. The “no accumulated losses” ablation keeps the same mutation prompt as CVaR-RMD, but the statistics inserted into the prompt are computed from the current evaluation batch rather than from the accumulated loss multiset.

Inputs. The same inputs as Figure 4, with performance summaries computed from the nonaccumulated batch losses.

Output. A single complete Python function implementing {FUNCTION\_SIGNATURE}.

![](images/de47920ecf7c377c18abfd25348edde3e901c79f5017f90892374185be98bdb3.jpg)  
Figure 5: Direct LLM sampling prompt.

## D.10 Output Formatting

All LLM code-generation prompts enforce the same output contract:

• The response must contain exactly one complete Python implementation of {FUNCTION\_SIGNATURE}.

• The required function signature must be preserved exactly.

• The response must not contain Markdown code fences, prose explanations, test cases, example calls, or text outside the function definition.

• Helper logic must be placed inside the required function unless the harness explicitly permits additional top-level definitions.

• The function must return a solution object in the task-specific format described in {TASK\_SPECIFICATION}.

• The function must avoid unsupported packages, file or network access, nondeterministic global state, excessive runtime, infeasible outputs, and interface violations.

Malformed outputs are retried up to three times using the same prompt. If parsing, compilation, interface validation, execution, or feasibility checking fails after retries, the candidate receives the fixed failure penalty ℓ<sub>fail</sub> under the deterministic evaluation protocol.

## E Case Study

We provide qualitative examples to illustrate how CVaR-RMD changes the type of heuristics selected by the outer loop. The goal of this case study is not to compare aggregate scores, but to inspect the generated programs and the failure modes that the selection objective encourages or suppresses. In all examples, the LLM is used only as a mutation operator: it proposes executable variants of a fixed heuristic interface, while the evaluator accepts or rejects candidates according to their behavior over heterogeneous instances, stochastic seeds, and problem scales.

Bin packing: avoiding brittle “perfect-fit” behavior. A representative mean-selected binpacking heuristic produced a compact rule that prioritized immediate utilization. Its core decision was to place each item into the bin that would leave the smallest remaining capacity after insertion:

“For each item, scan all open bins and choose the feasible bin with minimum residual capacity. If no bin fits, open a new bin.”

The code was syntactically clean and often behaved well on regular instances where item sizes admitted many near-perfect combinations. However, inspection of its outputs revealed a brittle pattern. On instances containing a mixture of medium-sized and small items, the heuristic repeatedly consumed small residual spaces too early. Later medium items then failed to fit into existing bins, forcing the program to open additional bins. The failure was not caused by invalid solutions; rather, the heuristic made locally attractive placements that created unfavorable residual-capacity profiles.

![](images/65f3a75bd5376abdeb2645365ccc3f7b6a73603e8ee4946ef40d1bf60edbfbc6.jpg)  
Figure 6: Mean-RMD mutation prompt difference. All other fields are identical to Figure 4.

A CVaR-selected mutation preserved the same general structure but introduced an explicit slackaware decision rule:

“Prefer the tightestfeasible bin only when the resulting residual capacity is either very small or remains large enough to accommodate future medium items. Otherwise, choose a slightly looser bin to preserve reusable slack. Break ties by favoring bins whose residual capacities are closer to the current median residual capacity.”

The corresponding generated code maintained a small summary of open-bin residual capacities and computed a penalty for placements that created isolated unusable gaps. Qualitatively, this candidate no longer treated every tight fit as equally desirable.

When faced with an item that could fit into several bins, it sometimes avoided the tightest placement if doing so would create a residual capacity unlikely to be useful later. This behavior is consistent with CVaR-RMD’s selection pressure: candidates that occasionally collapse on adversarial item-size mixtures are less likely to survive, even if their average behavior is strong.

Capacitated vehicle routing: balancing route compactness and residual capacity. For CVRP, a typical mean-selected candidate resembled a nearest-feasible insertion heuristic. Given a partial solution, it selected the next customer primarily by geographic proximity:

“For each route, repeatedly append the nearest unrouted customer that fits the remaining vehicle capacity. When nofeasible customer remains, close the route and start a new one from the depot.”

The resulting routes were easy to interpret: vehicles greedily swept through local neighborhoods. Nevertheless, qualitative inspection showed a recurring tail failure. In clustered instances with uneven demand, the heuristic committed early to visually compact routes but left several high-demand customers until late in construction. These remaining customers were geographically close to existing routes but could no longer be inserted because the routes had insufficient residual capacity. The final solution then contained routes that made long depot returns for isolated customers.

![](images/c1ec042c4329ba32e60ae7f9cbeae77326cfe0310134969af8fa9e20acbd5054.jpg)  
Figure 7: Archive-conditioned prompt used by mean-selection evolutionary baselines.

A CVaR-selected variant modified the insertion priority:

“Score eachfeasible customer by combining distance, demand pressure, and the remaining capacity after insertion. Customers with high demand are promoted when delaying them would make later insertion difficult. If several candidates have similar distance, prefer the one that reduces future capacity fragmentation.”

The generated program did not abandon locality; nearby customers were still preferred. The important difference was that distance was no longer the only organizing principle. The candidate explicitly represented residual capacity as a scarce resource and penalized decisions that would strand large-demand customers. In the produced routes, high-demand customers were usually assigned before the route became nearly full, while smaller customers were used later as flexible fillers. This led to more stable behavior across instance families because the heuristic did not depend on the fortunate ordering of customer demands.

Flow-shop scheduling: replacing a single dispatching signal with bottleneck-aware priorities. In the flow-shop task, a mean-selected heuristic generated job sequences using a simple aggregate processing-time rule:

“Compute a priority score for each job from its total processing time. Schedule jobs with larger total processing time earlier, with a small adjustment for the first machine.”

This heuristic was transparent and often produced plausible schedules. However, its behavior was unstable when processing times were unbalanced across machines. For example, when a subset of jobs had moderate total processing time but was unusually heavy on a downstream machine, the heuristic tended to schedule them too late. The resulting schedule accumulated idle time before the bottleneck machine and then produced a long congested tail.

A CVaR-selected mutation introduced a more structured priority:

“For eachjob, estimate its contribution to early-machine release and downstream bottleneck load. Promote jobs that prevent bottleneck starvation, but delay jobs whose early-machine load would block many short jobs without relieving the bot-

![](images/beddfd9dfe07e8b57538757cc239b0c387d10146c5c70ac95afb0ad26f01825f.jpg)  
Figure 8: Mutation prompt for the no-tail-records ablation.

## tleneck. Recompute priorities after partial sequencing.”

The generated code used normalized machine-wise processing times. It also updated job priorities during construction, so the dispatching rule could react to the partial sequence already built. Qualitatively, the schedules produced by this candidate were less dominated by a single global ordering. Jobs with downstream-heavy profiles were moved earlier when they helped keep the bottleneck machine active, while jobs that merely increased early congestion were postponed. This illustrates how tail-risk selection can favor heuristics that encode safeguards against specific pathological schedules.

Observed mutation pattern. Across tasks, we observed a common qualitative pattern in the programs selected by CVaR-RMD. Mean-based selection often retained concise greedy rules whose outputs looked strong on regular instances but depended heavily on favorable input structure. CVaR-RMD preserved the interpretable greedy backbone while adding small protective mechanisms: slack preservation in bin packing, demand-aware capacity management in CVRP, and bottleneckaware dispatching in flow-shop scheduling. These changes were not arbitrary complexity. They were localized modifications that directly addressed rare but consequential failure modes exposed by evaluating each candidate over varied environments.

This case study supports the central claim of the paper: changing the outer-loop selection pressure can change the qualitative character of LLMgenerated heuristics. CVaR-RMD does not require the language model to understand robustness explicitly. The evaluator repeatedly filters generated code through tail-risk behavior, causing robust design motifs to accumulate over evolutionary search.

![](images/20b4faadb83e676006f50aac53f7737f6396888599ba07b3bbf16bb4c3f179ee.jpg)  
Figure 9: Prompt substitution for tail-level ablations.

![](images/395b893d07524ae9c946a6ae20715df23ca895185d25bbcd3138cdbc37f59b1f.jpg)  
Figure 10: Prompt substitution for the no-accumulated-losses ablation.