# Evolving Towards Better Codes: LLM-Guided Search for High-Distance Binary Linear Codes

Amal Seddas<sup>1</sup>, Vladyslav Shashkov<sup>1</sup>, Maryna Viazovska<sup>1</sup>, Emmanuel Abbe<sup>1</sup> <sup>1</sup>EPFL

## Abstract

Evolutionary program search driven by large language models (LLMs) has produced record-breaking constructions for open problems in combinatorics and beyond. We apply this approach to the longstanding problem of improving the best-known bounds for binary linear codes. Building on the EvoTune evolutionary framework and the ShinkaEvolve codebase, we introduce LINCODEEVOLVE, which evolves code-construction programs against an exact minimum-distance evaluator. A strategy loop combines diversity-driven search and expert supervision: when progress plateaus, new strategies are used to redirect the search. LINCODEEVOLVE discovers seven record-breaking codes, [172, 21, 66], [173, 20, 68], [176, 21, 68], [181, 21, 70], [184, 21, 72], [189, 22, 72] and [200, 21, 77], six of which have concise quasicyclic descriptions. With standard code modification techniques, they improve 22 entries of the tables. Every code is verified by exhaustive enumeration; the results have been checked by the maintainer of the code tables and will be incorporated into them. These results suggest that LLM-guided search can help find improved codes and complement existing methods in coding theory.

## 1 Introduction

We study the construction of binary linear codes with large minimum distance. This is a central problem in coding theory: minimum distance determines a code’s worst-case error-correction capability and enters standard upper bounds on block error probability under maximum-likelihood decoding on the binary symmetric channel [SS06, Eq. (2.4), p. 15]. The Gilbert-Varshamov bound establishes the existence of codes with favorable parameters [Var57], yet explicit constructions can exceed this bound for particular lengths and dimensions, as recorded in CodeTables.<sup>1</sup> Finding such constructions improves known lower bounds and can reveal structure shared by codes with large minimum distance.

A construction that works well at one length and dimension need not work equally well at another. Progress therefore calls for selecting, adapting, and combining algebraic constructions and computational search procedures across parameter pairs. This makes code construction a natural setting for automated program discovery: a proposed construction can be executed, its output evaluated, and the result used to guide the next proposal.

FunSearch [RPBN<sup>+</sup>24] implements this idea by pairing language-model-generated programs with an executable evaluator and a database of earlier candidates. High-scoring programs inform subsequent proposals, allowing the search to refine promising ideas while exploring alternatives. FunSearch produced new cap-set constructions and bin-packing heuristics. AlphaEvolve [N<sup>+</sup>25] extends program evolution from individual functions to codebases with multiple evolved components. Georgiev et al. [GGSTW25] combine this approach with reasoning and proof-assistant systems for mathematical discovery and, in some cases, automated proof. In coding theory, Weindel and Heckel [WH25] use FunSearch to evolve priority functions for greedy constructions of deletion-correcting codes, recovering the conjectured-optimal Varshamov-Tenengolts construction for a single deletion and improving lower bounds for multiple deletions and quaternary edit codes.

We introduce LINCODEEVOLVE, a program-evolution framework for discovering binary linear codes with improved parameters. Building on ShinkaEvolve [LIC25], LINCODEEVOLVEcoordinates agents that propose and implement experiments with specialist search islands focused on different construction strategies. They draw on a shared bank of strategies and verified codes, initialized with expert-written implementations of diverse coding-theoretic constructions. An executable evaluator assesses proposed codes, and verified results guide later proposals. This separation between proposing a construction and evaluating its output is especially important here: determining minimum distance is computationally difficult in general, while improvements to the best-known bound occur in discrete steps.

Our contributions are a framework for systematically evolving linear-code constructions and new codes that improve longstanding bounds. Each search run targets a fixed length dimension pair $( n , k )$ . Within the range $1 \leq k \leq n \leq 2 0 0$ , LINCODEEVOLVEdiscovers seven parent codes with parameters [173, 20, 68], [172, 21, 66], [176, 21, 68], [181, 21, 70], [184, 21, 72], [189, 22, 72], and [200, 21, 77]. Each improves a longstanding lower bound recorded in CodeTables by one to three units. Explicit derivations from these parent codes yield improvements for 22 parameter sets at lengths 171-200 (Section 5.2).

## 2 Background and Related Work

Binary linear codes Consider $k , n \in \mathbb { N } , k \leq n .$ . A binary linear $[ n , k ]$ code C is a k-dimensional subspace of $\mathbb { F } _ { 2 } ^ { n }$ . We will refer to $\textstyle R : = { \frac { k } { n } }$ as the code rate. A full-rank generator matrix $G \in \mathbb { F } _ { 2 } ^ { k \times n }$ maps a message vector $u \in \mathbb { F } _ { 2 } ^ { k }$ to a codeword $c = u G \in { \mathcal { C } }$ . By linearity, the minimum distance $d ( \mathcal { C } )$ is equivalent to the minimum Hamming weight among all nonzero codewords in $\mathcal { C } \mathrm { : }$

$$
d ( \mathcal { C } ) : = \operatorname* { m i n } _ { u \in \mathbb { F } _ { 2 } ^ { k } \backslash \{ 0 \} } \mathrm { w t } ( u G ) = \operatorname* { m i n } _ { u \in \mathbb { F } _ { 2 } ^ { k } \backslash \{ 0 \} } | \{ i \in [ n ] | ( u G ) _ { i } = 1 \} | .
$$

The minimum distance establishes the bounded-distance decoding capability of the code, guaranteeing the correction of up to $\lfloor ( d ( \mathcal { C } ) - 1 ) / 2 \rfloor$ arbitrary errors. The multiplicity of weight-w codewords is denoted by $A _ { w }$ , which defines the weight enumerator of C, i.e. the vector $\{ A _ { w } | w \in [ n ] \}$

Gilbert-Varshamov bound Consider a parameter $\delta \in \left[ 0 , { \frac { 1 } { 2 } } \right)$ . Define the binary entropy function $h _ { 2 } ( x ) =$ $- x \log _ { 2 } ( x ) - ( 1 - x ) \log _ { 2 } ( 1 - x )$ . Then, for any $\epsilon > 0$ and all sufficiently large n, there exists a binary linear code C of length $n ,$ dimension $k ,$ and minimum distance $d ( \mathcal C )$ satisfying

$$
\bullet \ k \geq n ( 1 - h _ { 2 } ( \delta ) - \epsilon )
$$

$$
\cdot \ d ( \mathcal { C } ) \geq n \delta
$$

At a high level, the Gilbert-Varshamov bound provides an achievability guarantee for the tradeoff between the code rate and relative minimum distance: asymptotically, binary linear codes with relative minimum distance at least $\delta$ exist at any rate below $1 - h _ { 2 } ( \delta )$

Record tables and distance certification. For fixed $( n , k )$ , the construction problem is to find a binary linear code with the largest possible minimum distance. Grassl’s tables $[ \mathrm { G r a } 0 6 ] ^ { 2 }$ record the best-known lower and upper bounds on this optimum, together with supporting constructions and references. A construction with certified distance at least $d$ establishes a lower bound; upper bounds limit what any code with those parameters can achieve. These include the Griesmer bound [Gri60] and linear-programming bounds.

Certifying a candidate is computationally challenging: computing minimum distance is NP-hard [Var97]. Exact methods include the Brouwer-Zimmermann algorithm, implemented in Magma and Sagemath [BCP97, Gra06], and, for small $k ,$ enumeration of all $2 ^ { k } - 1$ nonzero messages. Randomized searches based on information-set decoding [Pra62] can find low-weight codewords and therefore upper bounds on a candidate’s minimum distance. However, failing to find a low-weight codeword does not certify its absence. Improving a table’s lower bound consequently requires a rigorous distance guarantee.

Computer search for record codes. Existing searches exploit algebraic structure to make the space of candidate codes more manageable. One major approach searches within quasi-cyclic and quasi-twisted families. This line includes early computational tables [CPW69, GB91], structural results for one-generator codes [ASRC01], and subsequent search algorithms [DH03, Che15b]. A second approach uses secondary constructions to obtain new codes from known ones. Examples include Construction X and XX [SRC72], shortening, puncturing, extension, and subcodes [HP03], as well as augmentation methods that add generator rows to a quasi-cyclic parent [Che15a, ACM17]. A third approach prescribes a symmetry group and searches for codes invariant under its action, for example by solving Diophantine systems $[ \mathrm { B B F ^ { + } 0 6 } ]$ or constructing group codes over semidirect products $[ \mathrm { Y } \mathrm { Z } \mathrm { C } ^ { + } 2 4 ]$ . Together, these approaches motivate searching across construction families, parent codes, and symmetries while retaining rigorous distance verification. Learningbased methods have also been explored for specialized families, including reinforcement learning for linear complementary-dual codes [WMY25]. Our focus is the search for improvements to the general binary linear-code tables.

LLM-driven program evolution. Evolution through Large Models $[ \mathrm { L G J ^ { + } } 2 2 ]$ and Language Model Crossover [MNB<sup>+</sup>23] use an LLM to mutate and recombine programs. FunSearch $[ \mathrm { R P B N ^ { + } } 2 4 ] ,$ AlphaEvolve [N<sup>+</sup>25] and ShinkaEvolve [LIC25] apply the same loop to open mathematical and algorithmic problems, EvoTune $[ \mathrm { S M Q ^ { + } } 2 5 ]$ guides the mutation through Direct Preference Optimization (DPO) given the scores assigned to the programs by the separately-defined reward function; [WWW<sup>+</sup>24] survey the field. LIN-CODEEVOLVE is built on ShinkaEvolve and keeps its loop: sample a parent and a few inspiration programs from an archive split into islands, ask an LLM for a patch or a new program, evaluate it, store it. It also keeps the novelty judge, the meta summarizer, prompt evolution and UCB1-style model selection [ACBF02]. Two things change. The single proposal call becomes a chain of oracle, strategist, contract repair and implementer, with a helper-free invent operator as control. And nothing enters the archive without an exact certificate (Section 4).

Diversity and verified feedback. In our runs the model often handed back the parent matrix unchanged. This matches [GSW26], who show that LLM mutation chains collapse onto a few forms even without selection pressure. Quality-diversity archives [MC15] and island models [AT02] are the basis of our family balanced archive and our islands. Learned constructions [Wag21, CEWW24] and neuro-symbolic provers [TWL<sup>+</sup>24] share our premise that a learned proposer needs an exact checker. Automated research agents [LLL<sup>+</sup>24, HLC24] inspired the split of work among our roles. In LINCODEEVOLVE only exact computations reach a score or the stored evidence; what a program says about itself never changes its score.

## 3 Problem Statement

Fix parameters $( n , k )$ where the table entry is open, meaning the lower bound $d _ { \mathrm { L B } }$ is strictly less than the upper bound $d _ { \mathrm { U B } }$ . The task is to find the generator matrix $G$ which generates the binary linear code $\mathcal { C }$ with $d ( \mathcal { C } ) \geq d _ { \mathrm { L B } } + 1$ , meaning we beat the best-known linear code (BKLC) for these parameters.

To do this, we track two values: the minimum distance $d ,$ and the number of minimum-weight codewords $A _ { d }$ . The latter is called the shell. Between two codes with the same minimum distance, we prefer the one with the smaller shell. To reach $d + 1$ , the amount of weight-≤ d codewords must be 0, i.e. the shell must be equal to 0, making $A _ { d }$ a useful metric. Minimizing the shell given that the minimum distance is d guides the search in the direction of better codes.

## 4 The LINCODEEVOLVE Framework

LINCODEEVOLVE evolves construction programs to improve best-known lower bounds on the minimum distance of binary linear codes. It follows the evolutionary search loop of EvoTune $[ \mathrm { S M Q ^ { + } } 2 5 ]$ : a language model proposes programs, their outputs are evaluated, and archived results guide subsequent proposals. ShinkaEvolve [LIC25] provides the execution infrastructure and the program database, including islands and migration, novelty assessment and prompt evolution. We retain these components. The code-construction components summarised in Table A are built on top of them, and the language model’s parameters remain fixed throughout the search.

Three properties of code construction motivate these additions. First, minimum distance is integer-valued and gives coarse feedback: many candidates share the same distance, so we also compare their numbers of minimum-weight codewords. Second, returning the parent code produces a valid output without advancing the search, so the loop must recognise copies. Third, progress can require a change of construction family, so algebraic tools and evidence about routes already explored must be available when the next proposal is written. At the dimensions we consider, an exact evaluation of one code takes seconds while a model proposal takes minutes, so each generated program should search many candidates during one execution. LINCODEEVOLVE organises this search around five components:

• An agent team. The discover operator separates proposal generation into three roles: an oracle analyses the current code, a strategist compares construction approaches and commits to one in a structured plan, and an implementer writes the corresponding program. A separate invent operator generates programs without direct access to the Toolkit.

• A Toolkit. Generated programs can call hand-written strategies that implement constructions and search procedures from the coding-theory literature, so an expert construction costs a few lines rather than a new implementation, and proposals can select, combine and extend them.

• Evaluation and search feedback. The evaluator computes the full weight distribution of each admissible returned code. The fitness is the lexicographic order on $( d , - A _ { d } )$ : higher minimum distance first, then fewer minimum-weight codewords; it is used for champion selection and parent ranking alike.

• Specialised islands and construction-based initialisation. Each island’s prompts carry one construction family, such as algebraic codes over extension fields or sparse-graph parity-check design, to encourage exploration of different construction families. The initial code is selected from a portfolio of constructions after exact evaluation of the shortlisted candidates.

• Exact verification and human input. Only evaluator-computed results determine a code’s fitness; a program’s self-reported claims are stored but cannot change it. Human notes, tools and certificates enter through shared files that the prompts read, and interventions are made through explicit pause-andresume controls.

The model writes the search program: it chooses a construction approach, decides whether to modify the parent or start again, and specifies how candidates are selected. The framework chooses parents and proposal operators, runs the programs, evaluates their outputs, and updates the archive, retaining one program per distinct row space. Figure 1 shows one generation; the following subsections describe its components.

![](images/72f5472c621eb9b33f9229fb52f3f23ee739c9952ee0261d7411f6f807c859a7.jpg)  
Figure 1: Agent-guided proposals in LINCODEEVOLVE. A parent program and its code are selected from the archive (1). The proposal context holds the code’s exact $( d , A _ { d } )$ , the island’s construction approach, and the results of earlier attempts (2). Discover uses three successive model calls to produce a search program; Invent provides an alternative without direct access to the strategy Toolkit (3). The generated program searches candidate matrices, using Toolkit routines when permitted (4). The evaluator checks the returned matrix and computes the exact weight distribution of valid outputs (5). Evaluation results update the archive and inform later proposals. Researchers supply starting points and mathematical observations through shared files.

## 4.1 Agent-Guided Program Generation

Each proposal receives the parent program and generator matrix, its verified weight distribution, the target parameters and distances, applicable bounds, and the island’s construction family. Previous results, failed attempts, and mathematical observations such as obstruction bounds and expert notes provide additional context.

Discover. Three successive model calls turn this context into a search program (Figure 2). The oracle diagnoses the current code and suggests a move, supported by evidence and a condition for stopping an unsuccessful attempt. The strategist compares construction approaches, checks the parameter arithmetic of proposed parent codes, and selects one; it records the approach, the move, whether to modify the parent or start again, the acceptance rule, and the stopping condition in one structured plan. The implementer translates this plan into a program that can call the strategy Toolkit and available exact tools. The program must restate the plan and follow its candidate-selection rule.

Invent. A separate operator proposes a procedure without seeing the strategy catalogue or calling its strategies. It first describes the representation, the move, the verification step, and the stopping condition, then implements the procedure from lower-level operations. It still receives the parent program and accumulated evidence, which may reflect earlier use of the strategy Toolkit.

Operator selection. Outside the invention island, proposals are sampled from the inherited $d i f f , f u l l ,$ and cross operators, alongside discover and invent. Their probabilities adapt to the parent’s distance and recent outcomes, with additional weight given to invent when the parent is below the distance floor set for the campaign and recent proposals repeatedly return the same code.

Checks and execution. Each model call must return its required fields. If the strategist’s plan is incomplete, it gets one more call to complete it, and the proposal is dropped if it is still incomplete. These checks concern the format of the plan, not whether it is mathematically sound. Before a program runs, it is rejected if it only wraps one Toolkit strategy, if it lacks the required header, or if it re-implements the exhaustive enumeration; the novelty judge from ShinkaEvolve, re-prompted for this task, also rejects programs that hardly differ from the parent. A program that passes runs once on the parent’s matrix, in its own process, under a time limit. Its output counts only if it is a binary $k \times n$ matrix of rank k; the evaluator then enumerates the code and records $( d , A _ { d } )$ . If the program crashes, times out, or returns anything else, it is scored as if it had returned the parent unchanged: it keeps the parent’s score, does not enter the archive, and is no longer used as a parent once any program has reached the distance floor. The archive holds up to 96 programs, one per distinct code. A new code is added while there is room; when the archive is full it replaces an archived program under a rule that keeps small construction families represented . A code already in the archive is not added again.

## 4.2 Fitness

The fitness of a program is the pair $( d , - A _ { d } )$ of the code it returns, maximized in lexicographic order: a larger minimum distance d is preferred, and at equal distance, fewer minimum-weight codewords $A _ { d }$ win. Both fitness values are exact, from an enumeration of the weight distribution using rbouwer zimmeand alh.

We use a strictly smaller $A _ { d }$ not only as a heuristic to guide further search but also because it yields a provably more reliable code. Specifically, the undetected-error probability on the binary symmetric channel with crossover probability $p$ is

$$
P _ { \mathrm { e r r } } ( C ; p ) = \sum _ { i = 1 } ^ { n } A _ { i } p ^ { i } ( 1 - p ) ^ { n - i } ,
$$

which represents the probability that the channel turns a transmitted codeword into another codeword. For small values of $p ,$ this probability is heavily dominated by the leading term $A _ { d } p ^ { d } ( 1 - p ) ^ { n - d }$ , meaning minimizing $A _ { d }$ directly minimizes the likelihood of undetected errors.

![](images/33a3b11b75d345e773807a89de3064ba06a5f27ad52daacd9aa7a3d097964e6d.jpg)  
Figure 2: The Discover and Invent proposal operators. Discover separates proposal generation into diagnosis, planning, and implementation. Invent develops a procedure without direct access to Toolkit. Both operators are subject to proposal checks and a time-limited execution. Valid returned matrices are evaluated exactly; failure handling is described in the text.

## 4.3 Initialization: Generation 0

Our search is initialized with a strong initial program that outputs the linear code with the minimum distance exceeding the Gilbert-Varshamov (GV) bound. To construct this base, we evaluated a diverse portfolio of structured code families, including design-based, quasi-cyclic, and algebraic constructions. Candidate codes are first ranked using computationally inexpensive distance estimates [citation needed], and the most promising candidates undergo an exact evaluation using the Brouwer-Zimmermann algorithm implemented in SageMath. We select the code with the largest minimum distance, using the number of low-weight codewords as a secondary filter.

Table 1 compares the selected seeds against the GV bound and random full-rank matrices, reporting both the best and the median minimum distances across 50 random draws. For instances with $k \leq 3 0$ , exact distances are rigorously verified via full enumeration. This initialization establishes a highly competitive baseline from which the subsequent search phase seeks further improvements.

## 4.4 Specialised Islands

We use ShinkaEvolve’s island model to explore several approaches to code construction in parallel. The search begins with eight populations of programs. Each island receives a short prompt describing a construction approach, such as algebraic codes over extension fields, sparse-graph parity-check design, or quasi-cyclic polynomial matrices. The agents use this perspective when proposing changes to the parent code; strategies of other families remain available to them as fallbacks and for final repair.

Two islands have a different role. The cold-start island constructs codes directly from the target parameters $( n , k )$ , using the current best matrix only as a fallback. The invention island always uses the invent operator, which has no direct access to the Toolkit. These islands allow the search to explore alternatives to both the current construction and the available Toolkit routines.

Programs migrate between islands every six generations, while the best program on each island is retained. If the best score has not improved for six generations, a new island is seeded from an archived program. This arrangement allows promising programs to spread while maintaining separate searches guided by different construction approaches.

## 4.5 The Toolkit

The is where most of the human knowledge enters. It is a library of 87 hand-written functions, each taking the current generator matrix and returning a candidate, so a generated program can invoke any of them in one line and combine them freely. They range from general search heuristics, such as simulated annealing and genetic population search, to algebraic constructions from the coding-theory literature: cyclic and BCH parent codes, Construction X, quasi-cyclic and quasi-twisted matrices, and procedures that repair the minimum-weight codewords of an existing code. Six family cores group these strategies by the kind of construction they belong to; a core selects suitable strategies for its input, runs them under a deadline, and returns the best finalist.

Generated programs can combine these strategies in a search loop, choose when to apply them, and decide which candidates to keep. They can also implement additional operations directly. Programs that call only a single strategy without adding search logic are rejected before execution.

<table><tr><td>cell</td><td>LB</td><td>GV</td><td>random best</td><td>random median d</td><td>gen 0</td><td>lane</td><td>S</td></tr><tr><td>[41, 12]</td><td>14</td><td>11</td><td>(11, 5)</td><td>10</td><td>(12,11)</td><td>construction_core</td><td>214</td></tr><tr><td>[48, 16]</td><td>16</td><td>11</td><td>(11, 3)</td><td>10</td><td>(14, 256)</td><td>qc_core</td><td>321</td></tr><tr><td>[50, 20]</td><td>13</td><td>10</td><td>(10, 6)</td><td>9</td><td>(11, 30)</td><td>qc_core</td><td>413</td></tr><tr><td>[60, 20]</td><td>17</td><td>13</td><td>(13, 1)</td><td>12</td><td>(16, 245)</td><td>qc_core</td><td>307</td></tr></table>

Table 1: Generation-0 against random full-rank matrices, exact $( d , A _ { d } )$ . Random: best of 50 draws, and the median d of the 50. GV: the Gilbert–Varshamov guarantee for $( n , k )$ . the portfolio’s pick and its winning lane.

## 5 Results

## 5.1 Choosing Target Parameters $( n , k )$ : Human Intuition

The framework conducts the search at fixed parameters $( n , k )$ . We chose these parameters outside the evolutionary loop by examining the code tables, the constructions behind their lower bounds, and the cost of computing the minimum distance. We considered open entries, where there is a gap between the lower and the upper bound . A large gap leaves room for improvement, but may reflect a loose upper bound rather than a weak construction. We therefore used it only as an initial filter.

Plateaus and inherited bounds. We looked for lengths over which the recorded distance stays constant as n increases. $\mathrm { A t } k = 2 1$ , the lower bound is 64 throughout 165 $\leq n \leq 1 7 3$ and 68 throughout $1 7 9 \leq n \leq 1 8 3$ Such plateaus in the minimum distance can arise when a known code is padded with zero coordinates, leaving open the possibility of using the additional coordinates more effectively.

Our number theoretic hint. Our human input was a hint about where to point the search. The row $k = 2 1$ is frozen at d = 64 for $1 6 5 \le n \le { 1 7 3 }$ , around $n = 1 7 0 = 2 \cdot 8 5$ . For block length 85, $x ^ { 8 5 } - 1$ splits into small factors $( x + 1 , \Phi _ { 5 }$ and ten octics), so $k = 2 1 = 1 + 4 + 8 + 8$ is a natural dimension for an index-2 quasi-cyclic code of length 170. We therefore suggested a quasi-cyclic core of length 170 as a starting point. Similarly, $1 8 9 = 3 \cdot 6 3$ with $2 2 = 1 + 3 + 6 + 6 + 6$ suggested a starting point for [189, 22].

## 5.2 New Records

LINCODEEVOLVE finds seven binary linear codes that improve the lower bounds recorded in the CodeTables snapshot of 25 September 2026. Table 2 summarizes the constructions and their gains. The first five arise from index-2 quasi-cyclic parent codes of length 170 and block length 85; [189, 22, 72] is an index-3 quasi-cyclic code with block length 63. The [200, 21, 77] code is obtained by puncturing a listed [240, 21, 104] binary Goppa-code construction.

Table 2: Seven constructions improving CodeTables lower bounds in the 25 September 2026 snapshot. The gain is the difference between the new minimum distance and the previous lower bound.
<table><tr><td rowspan=1 colspan=3>Code $[ n , k , d ]$   Previous bound  Gain</td></tr><tr><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>172,21, 66]</td><td rowspan=1 colspan=1>64   +2</td></tr><tr><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>173, 20, 68]</td><td rowspan=1 colspan=1>67   +1</td></tr><tr><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>176, 21, 68]</td><td rowspan=1 colspan=1>66   +2</td></tr><tr><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>181, 21, 70]</td><td rowspan=1 colspan=1>68   +2</td></tr><tr><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>184, 21, 72]</td><td rowspan=1 colspan=1>69   +3</td></tr><tr><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>189, 22, 72]</td><td rowspan=1 colspan=1>71   +1</td></tr><tr><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>200, 21, 77]</td><td rowspan=1 colspan=1>76   +1</td></tr></table>

The constructions have effects beyond their own parameter pairs. Explicit derivations yield improvements in 22 CodeTables entries overall, with gains of up to three in minimum distance.

Figure 3 shows the cluster at lengths 171-191; the [200, 21, 77] construction lies outside its displayed range.

We verified the reported minimum distances using SageMath [The24] and Magma [BCP97]. The appendix and accompanying files provide the generator matrices, weight distributions, derivations, and verification scripts.

Markus Grassl, who maintains CodeTables , independently checked the new constructions and accepted them for inclusion. The live tables have not yet been updated. The improvements are measured against the bounds listed in CodeTables on 24 September 2026.

## 5.3 Structural Advantages of Our Codes

A simpler construction matching the best-known lower bound. LINCODEEVOLVE also finds codes that match existing distance bounds while admitting concise algebraic descriptions. For example, it constructs a binary [50, 25, 10] code, meeting the current best-known lower bound of 10 (the upper bound is 12). The code is pure double circulant: its generator matrix has the form $\left[ I _ { 2 5 } \mid A \right]$ , where A is a $2 5 \times 2 5$ circulant matrix. This example shows that the search can recover structured, readily reproducible constructions even when it does not improve the distance bound.

Comparison with direct prompting of GPT-6 Astra. For $( n , k ) = ( 1 7 2 , 2 1 )$ , LINCODEEVOLVE finds a code with minimum distance 66 built from a quasi-cyclic $[ 1 7 0 , 2 1 , 6 4 ]$ core by adding two coordinates via

![](images/67f4b75cf8367dc3af741f3c7b8b13be56e2e0d09440557a8200e4dc213c6bd5.jpg)  
Figure 3: Codes at lengths 171-191. Cells show minimum distance and gain over CodeTables; outlines identify parent codes.

Construction X. The best code obtained by directly prompting GPT-6 Astra at extra-high reasoning effort in our comparison has minimum distance 65.

At (n, k) = (200, 21), both approaches reach distance 77, but they produce inequivalent codes with different weight distributions. The LINCODEEVOLVE code has 25 minimum-weight codewords, compared with 75 for the directly prompted code. It also has fewer nonzero codewords of weight at most 78 (264 versus 303).

Consequently, its parity extension yields a [201, 21, 78] code with 264 minimum-weight codewords, compared with 303 for the extension of the other code.

Direct prompting might recover these constructions with more attempts. The contribution of LINCODEE-VOLVE is a systematic search process that can be applied across pairs of parameters without creating a new prompt for each target. It preserves verified candidates, uses exact evaluations to guide subsequent proposals, and produces codes whose properties can be independently checked.

## 6 Conclusion

Using LINCODEEVOLVE, we found seven parent codes using this method. The code modification techniques extend these results to 22 parameter pairs. Although we applied LINCODEEVOLVE only to binary codes, the same approach can be used for q-ary codes, given an exact minimum-distance evaluator over $\mathbb { F } _ { q } .$ Since exact evaluation then scales like $q ^ { k }$ rather than $2 ^ { k }$ , the feasible dimensions are correspondingly smaller. The search can also be restricted to structured families to look for the best codes within them.

The time complexity of our method scales exponentially in terms of code dimension. However, for fixed k, longer codes remain inexpensive to evaluate, so LINCODEEVOLVE may extend to larger lengths. A practical obstacle is comparison: the binary tables of CodeTables currently stop at length 256, which makes it harder to tell whether a longer code beats the best known code of the same length and dimension.

## 7 Resource usage

We used a single 80GB H100 on the server of the Mathematical Data Science laboratory of EPFL. Each experiment took 3-4 hours on average using these resources.

## 8 Acknowledgement

We thank Gabriel Jimenez Alejandro Calles for their contributions to adapting the ShinkaEvolve database for LINCODEEVOLVE, and Anja Surina for stimulating discussions on evolutionary search.

## 9 LLM Usage Statement

We used LLM-based chatbots to improve and polish the writing of this paper. Given the nature of our research, we also used LLMs for ideation within the framework, for example, to generate improved prompts based on successful prior results. However, beyond the framework itself, we did not use LLMs to generate ideas or strategies for applying the framework. We also used LLMs to identify potentially relevant citations. All such citations were independently verified by the authors before being included in the literature review. Finally, LLMs were used to assist in drafting portions of the manuscript; these sections were subsequently reviewed, revised, and, where necessary, rewritten by the authors.

## References

[ACBF02] Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2–3):235–256, 2002.

[ACM17] Nuh Aydin, Nicholas Connolly, and John Murphree. New binary linear codes from quasi-cyclic codes and an augmentation algorithm. Applicable Algebra in Engineering, Communication and Computing, 28:339–350, 2017.

[ASRC01] Nuh Aydin, Irfan Siap, and Dijen K. Ray-Chaudhuri. The structure of 1-generator quasi-twisted codes and new linear codes. Designs, Codes and Cryptography, 24(3):313–326, 2001.

[AT02] Enrique Alba and Marco Tomassini. Parallelism and evolutionary algorithms. IEEE Transactions on Evolutionary Computation, 6(5):443–462, 2002.

[BBF<sup>+</sup>06] Anton Betten, Michael Braun, Harald Fripertinger, Adalbert Kerber, Axel Kohnert, and Alfred Wassermann. Error-Correcting Linear Codes: Classification by Isometry and Applications, volume 18 of Algorithms and Computation in Mathematics. Springer, 2006.

[BCP97] Wieb Bosma, John Cannon, and Catherine Playoust. The Magma algebra system I: The user language. Journal ofSymbolic Computation, 24(3–4):235–265, 1997.

[CEWW24] François Charton, Jordan S. Ellenberg, Adam Zsolt Wagner, and Geordie Williamson. Pat ternBoost: Constructions in mathematics with a little help from AI. arXiv preprint, 2024. arXiv:2411.00566.

[Che15a] Eric Zhi Chen. New binary h-generator quasi-cyclic codes by augmentation and new minimum distance bounds. Designs, Codes and Cryptography, 2015.

[Che15b] Eric Zhi Chen. A new iterative computer search algorithm for good quasi-twisted codes. Designs, Codes and Cryptography, 76(2):307–323, 2015.

[CPW69] Chin-Long Chen, W. Wesley Peterson, and Edward J. Weldon. Some results on quasi-cyclic codes. Information and Control, 15(5):407–423, 1969.

[DH03] Rumen Daskalov and Plamen Hristov. New binary one-generator quasi-cyclic codes. IEEE Transactions on Information Theory, 49(11):3001–3005, 2003.

[GB91] T. Aaron Gulliver and Vijay K. Bhargava. Some best rate 1/p and rate (p − 1)/p systematic quasi-cyclic codes. IEEE Transactions on Information Theory, 37(3):552–555, 1991.

[GGSTW25] Bogdan Georgiev, Javier Gómez-Serrano, Terence Tao, and Adam Zsolt Wagner. Mathematical exploration and discovery at scale, 2025.

[Gra06] Markus Grassl. Searching for linear codes with large minimum distance. In Wieb Bosma and John Cannon, editors, Discovering Mathematics with Magma, pages 287–313. Springer, 2006.

[Gri60] James H. Griesmer. A bound for error-correcting codes. IBM Journal of Research and Development, 4(5):532–542, 1960.

[GSW26] Can Gurkan, Forrest Stonedahl, and Uri Wilensky. Mutation without variation: Convergence dynamics in LLM-driven program evolution. arXiv preprint, 2026. arXiv:2606.05408.

[HLC24] Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. arXiv preprint, 2024. arXiv:2408.08435.

[HP03] W. Cary Huffman and Vera Pless. Fundamentals of Error-Correcting Codes. Cambridge University Press, 2003.

[LGJ<sup>+</sup>22] Joel Lehman, Jonathan Gordon, Shawn Jain, Kamal Ndousse, Cathy Yeh, and Kenneth O. Stanley. Evolution through large models. arXiv preprint, 2022. arXiv:2206.08896.

[LIC25] Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. ShinkaEvolve: Towards open-ended and sample-efficient program evolution. arXiv preprint, 2025. arXiv:2509.19349.

[LLL<sup>+</sup>24] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv preprint, 2024. arXiv:2408.06292.

[MC15] Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. arXiv preprint, 2015. arXiv:1504.04909.

[MNB<sup>+</sup>23] Elliot Meyerson, Mark J. Nelson, Herbie Bradley, Adam Gaier, Arash Moradi, Amy K. Hoover, and Joel Lehman. Language model crossover: Variation through few-shot prompting. arXiv preprint, 2023. arXiv:2302.12170.

[N<sup>+</sup>25] Alexander Novikov et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint, 2025. arXiv:2506.13131.

[Pra62] Eugene Prange. The use of information sets in decoding cyclic codes. IRE Transactions on Information Theory, 8(5):5–9, 1962.

[RPBN<sup>+</sup>24] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nature, 625:468–475, 2024.

[SMQ<sup>+</sup>25] Anja Surina, Amin Mansouri, Lars Quaedvlieg, Amal Seddas, Maryna Viazovska, Emmanuel Abbe, and Caglar Gulcehre. Algorithm discovery with llms: Evolutionary search meets reinforcement learning, 2025.

[SRC72] Neil J. A. Sloane, Sudhakar M. Reddy, and Chin-Long Chen. New binary codes. IEEE Transactions on Information Theory, 18(4):503–510, 1972.

[SS06] Igal Sason and Shlomo Shamai. Performance analysis of linear codes under maximumlikelihood decoding: A tutorial. Foundations and Trends in Communications and Information Theory, 3(1–2):1–225, 2006.

[The24] The Sage Developers. SageMath, the Sage Mathematics Software System, 2024. https: //www.sagemath.org.

[TWL<sup>+</sup>24] Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625:476–482, 2024.

[Var57] Rom R. Varshamov. Estimate of the number of signals in error correcting codes. Doklady Akademii Nauk SSSR, 117:739–741, 1957.

[Var97] Alexander Vardy. The intractability of computing the minimum distance of a code. IEEE Transactions on Information Theory, 43(6):1757–1766, 1997.

[Wag21] Adam Zsolt Wagner. Constructions in combinatorics via neural networks. arXiv preprint, 2021. arXiv:2104.14516.

[WH25] Franziska Weindel and Reinhard Heckel. LLM-guided search for deletion-correcting codes. arXiv preprint arXiv:2504.00613, 2025.

[WMY25] Yansheng Wu, Jin Ma, and Shangdong Yang. Linear complementary dual codes constructed from reinforcement learning. Journal ofSystems Science and Complexity, 38:1388–1403, 2025.

[WWW<sup>+</sup>24] Xingyu Wu, Sheng-hao Wu, Jibin Wu, Liang Feng, and Kay Chen Tan. Evolutionary computation in the era of large language model: Survey and roadmap. arXiv preprint, 2024. arXiv:2401.10034.

[YZC<sup>+</sup>24] Cong Yu, Shixin Zhu, Hao Chen, Yang Li, and Xiuyu Zhang. New record-breaking binary linear codes constructed from group codes. arXiv preprint, 2024. arXiv:2412.15551.

## A Explicit constructions

All polynomials are over $\mathbb { F } _ { 2 }$ . Given a block length m, a divisor h of $x ^ { m } - 1$ of degree $k , g = ( x ^ { m } - 1 ) / h$ and polynomials $f _ { 1 } = 1 , f _ { 2 } , \ldots , f _ { \ell }$ , the words

$$
\begin{array} { r } { \left( b g f _ { 1 } , b g f _ { 2 } , . . . , b g f _ { \ell } \right) \mathrm { m o d } ( x ^ { m } - 1 ) , \qquad b = \sum _ { i < k } b _ { i } x ^ { i } , } \end{array}
$$

form a quasi-cyclic code of length mℓ, index ℓ and dimension k (coordinate $j$ of a block is the coefficient of $x ^ { j } )$ . Some codes append tail coordinates after the blocks. These depend only on

$$
r = b { \mathrm { ~ m o d ~ } } ( x ^ { 5 } - 1 ) = ( r _ { 0 } , \ldots , r _ { 4 } ) , \qquad r _ { t } = \sum _ { \substack { i \equiv t { \pmod { 5 } } } } b _ { i } ;
$$

a tail coordinate equals $\textstyle \left. c , r \right. = \sum _ { t } c _ { t } r _ { t }$ for a pattern c, printed as the bit string $c _ { 0 } c _ { 1 } c _ { 2 } c _ { 3 } c _ { 4 }$ . In the generator matrix whose row j is the codeword of $b = x ^ { j } \ ( 0 \leq j < k )$ , its entry in row $j$ is $c _ { j \mathrm { ~ m o d ~ } 5 }$ . The following seven base codes give all improved entries of Table A.

[172, 21, 66]

• m = 85, index 2.

• h is the product of

$$
\begin{array} { l } { { - \ x + 1 ; } } \\ { { - \ x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 ; } } \\ { { - \ x ^ { 8 } + x ^ { 6 } + x ^ { 5 } + x ^ { 4 } + x ^ { 3 } + x + 1 ; } } \\ { { - \ x ^ { 8 } + x ^ { 7 } + x ^ { 5 } + x + 1 . } } \end{array}
$$

$$
\bullet f _ { 1 } = 1 { \mathrm { ~ a n d ~ } } f _ { 2 } = x ^ { 1 3 } + x ^ { 1 1 } + x ^ { 9 } + x ^ { 6 } + x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 .
$$

• Tail: $\left( r _ { 0 } + \ldots + r _ { 4 } , \ r _ { 0 } + \ldots + r _ { 4 } \right)$ , i.e. the value b(1) = b mod (x + 1) repeated twice. This is Construction X with the repetition code [2, 1, 2].

[173, 20, 68]

• m = 85, index 2.

• h is the product of

$$
{ \begin{array} { l } { - \ x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 ; } \\ { - \ x ^ { 8 } + x ^ { 5 } + x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 ; } \\ { - \ x ^ { 8 } + x ^ { 7 } + x ^ { 6 } + x ^ { 5 } + x ^ { 4 } + x ^ { 3 } + 1 . } \end{array} }
$$

$f _ { 1 } = 1 { \mathrm { ~ a n d ~ } } f _ { 2 } = x ^ { 1 5 } + x ^ { 1 1 } + x ^ { 6 } + x ^ { 3 } + x ^ { 2 } + 1 .$

• Tail: 3 coordinates $\langle c _ { i } , r \rangle$ , where $c _ { 1 } , c _ { 2 } , c _ { 3 }$ are 01001, 10100, 01010.

[176, 21, 68]

$m = 8 5 ,$ index 2.

• h is the product of

$$
\begin{array} { l } { { - \ x + 1 ; } } \\ { { - \ x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 ; } } \\ { { - \ x ^ { 8 } + x ^ { 6 } + x ^ { 5 } + x ^ { 4 } + x ^ { 3 } + x + 1 ; } } \\ { { - \ x ^ { 8 } + x ^ { 7 } + x ^ { 5 } + x + 1 . } } \end{array}
$$

$f _ { 1 } = 1$ and $f _ { 2 } = x ^ { 1 3 } + x ^ { 1 1 } + x ^ { 9 } + x ^ { 6 } + x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 .$

• Tail: 6 coordinates $\langle c _ { i } , r \rangle$ , where $c _ { 1 } , \ldots , c _ { 6 }$ are 01111, 11111, 11111, 10010, 11011, 00101.

[181, 21, 70]

$m = 8 5 , \mathrm { i n d e x } 2 .$

• h is the product of

$$
\begin{array} { l } { { - \ x + 1 ; } } \\ { { - \ x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 ; } } \\ { { - \ x ^ { 8 } + x ^ { 5 } + x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 ; } } \\ { { - \ x ^ { 8 } + x ^ { 7 } + x ^ { 3 } + x + 1 . } } \end{array}
$$

$$
f _ { 1 } = 1 { \mathrm { ~ a n d ~ } } f _ { 2 } = x ^ { 1 3 } + x ^ { 1 0 } + x ^ { 7 } + x ^ { 6 } + x ^ { 4 } + x ^ { 3 } + 1 .
$$

• Tail: 11 coordinates $\langle c _ { i } , r \rangle$ , where $c _ { 1 } , \ldots , c _ { 1 1 }$ are 11000, 10100, 11100, 01010, 01110, 10001, 11101, 00011, 10011, 00111, 01111.

[184, 21, 72]

• m = 85, index 2.

• h is the product of

$$
\begin{array} { l } { { - \ x + 1 ; } } \\ { { - \ x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + x + 1 ; } } \\ { { - \ x ^ { 8 } + x ^ { 6 } + x ^ { 5 } + x ^ { 4 } + x ^ { 2 } + x + 1 ; } } \\ { { - \ x ^ { 8 } + x ^ { 7 } + x ^ { 5 } + x ^ { 4 } + x ^ { 3 } + x ^ { 2 } + 1 . } } \end{array}
$$

$$
\bullet f _ { 1 } = 1 { \mathrm { ~ a n d ~ } } f _ { 2 } = x ^ { 1 3 } + x ^ { 9 } + x ^ { 8 } + x ^ { 7 } + x ^ { 6 } + x ^ { 5 } + x ^ { 3 } + x + 1 .
$$

• Tail: 14 coordinates $\langle c _ { i } , r \rangle$ , where $c _ { 1 } , \ldots , c _ { 1 4 }$ are 11001, 01001, 00101, 11101, 11110, 00111, 11100, 01010, 10010, 10001, 11111, 01110, 10011, 00110.

[189, 22, 72]

• m = 63, index 3.

$$
{ \begin{array} { r l } & { \bullet \operatorname { \ h i s \ t h e \ p r o d u c t \ o f } } \\ & { \qquad - \ x + 1 ; } \\ & { \qquad - \ x ^ { 3 } + x ^ { 2 } + 1 ; } \\ & { \qquad - \ x ^ { 6 } + x ^ { 5 } + 1 ; } \\ & { \qquad - \ x ^ { 6 } + x ^ { 5 } + x ^ { 3 } + x ^ { 2 } + 1 ; } \\ & { \qquad - \ x ^ { 6 } + x ^ { 5 } + x ^ { 4 } + x ^ { 2 } + 1 . } \\ & { \qquad + \ x ^ { 1 6 } + x ^ { 1 3 } + x ^ { 1 2 } + x ^ { 9 } + x ^ { 7 } + x ^ { 5 } + x ^ { 3 } + x ^ { 2 } + 1 \operatorname { a n d } \ f _ { 3 } = x ^ { 1 6 } + x ^ { 1 4 } + x ^ { 1 3 } + x ^ { 1 0 } + x ^ { 7 } + x + 1 . } \end{array} }
$$

• No tail.

[200, 21, 77]

• Generator matrix.

$A _ { 7 7 } = 2 5$ , against $A _ { 7 7 } = 7 5$ for the [200, 21, 77] code obtained by direct prompting.

## [50, 25, 10] (double circulant)

$m = 2 5 ,$ index 2.

$h = x ^ { 2 5 } - 1 , \mathrm { { s o } } g = 1 .$

$$
\bullet \ f _ { 1 } = 1 \mathrm { a n d } \ f _ { 2 } = x ^ { 2 3 } + x ^ { 2 2 } + x ^ { 2 1 } + x ^ { 2 0 } + x ^ { 1 9 } + x ^ { 1 8 } + x ^ { 1 6 } + x ^ { 1 5 } + x ^ { 1 4 } + x ^ { 1 2 } + x ^ { 8 } + x ^ { 7 } + x ^ { 6 } + x ^ { 4 } + x ^ { 2 } + x .
$$

• Equivalently, the generator matrix is $[ I _ { 2 5 } \mid A ] ,$ , where A is the circulant matrix whose first row is the coefficient vector of $f _ { 2 }$

$d = 1 0 ,$ , the best known minimum distance for [50, 25], and $A _ { 1 0 } = 1 2 5$

All improved entries.

<table><tr><td>n</td><td>k</td><td> $d _ { \mathrm { o l d } }$ </td><td> $d ^ { \mathrm { u b } }$ </td><td> $A _ { d }$ </td></tr><tr><td>172</td><td>20</td><td>66</td><td>67</td><td>272</td></tr><tr><td>173</td><td>20</td><td>67</td><td>68</td><td>1156</td></tr><tr><td>171</td><td>21</td><td>64</td><td>65</td><td>2040</td></tr><tr><td>172</td><td>21</td><td>64</td><td>66</td><td>2720</td></tr><tr><td>173</td><td>21</td><td>64</td><td>66</td><td>680</td></tr><tr><td>174</td><td>21</td><td>65</td><td>66</td><td>680</td></tr><tr><td>175</td><td>21</td><td>66</td><td>67</td><td>1088</td></tr><tr><td>176</td><td>21</td><td>66</td><td>68</td><td>2992</td></tr><tr><td>177</td><td>21</td><td>66</td><td>68</td><td>2720</td></tr><tr><td>178</td><td>21</td><td>67</td><td>68</td><td>1054</td></tr><tr><td>180</td><td>21</td><td>68</td><td>69</td><td>1428</td></tr><tr><td>181</td><td>21</td><td>68</td><td>70</td><td>3502</td></tr><tr><td>182</td><td>21</td><td>68</td><td>70</td><td>2771</td></tr><tr><td>183</td><td>21</td><td>68</td><td>71</td><td>2108</td></tr><tr><td>184</td><td>21</td><td>69</td><td>72</td><td>9826</td></tr><tr><td>185</td><td>21</td><td>70</td><td>72</td><td>8670</td></tr><tr><td>186</td><td>21</td><td>70</td><td>72</td><td>6188</td></tr><tr><td>187</td><td>21</td><td>71</td><td>72</td><td>6103</td></tr><tr><td>200</td><td>21</td><td>76</td><td>77</td><td>25</td></tr><tr><td>201</td><td>21</td><td>77</td><td>78</td><td>264</td></tr><tr><td>188</td><td>22</td><td>70</td><td>71</td><td>924</td></tr><tr><td>189</td><td>22</td><td>71</td><td>72</td><td>2394</td></tr></table>

Table 3: Improved entries, with the best previously known minimum distance (old), the minimum distance of our code (new), and the number $A _ { d }$ of minimum-weight codewords of the released code.