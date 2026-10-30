# CoToan: Executable Rules and a Representation Contract for Reproducible Game AI Evaluation

Nguyen Ha Thanh

College of Engineering and Computer Science

VinUniversity

Hanoi, Vietnam

Abstract—Game AI results are comparable only when different implementations expose the same decision problem. Yet seemingly minor representation choices can silently change that problem by altering action probabilities or collapsing states with different outcomes. We study this issue through CoToan, an executable specification and representation contract for Vietnamese Mathematical Chess connecting the Game Description Language (GDL) with OpenSpiel. The game provides a useful case because a single capture may have multiple arithmetic justifications, while outcomes can depend on scoring rules and a finite playing horizon. CoToan therefore represents actions by their state transitions rather than by arithmetic witnesses, and treats time and victory configuration as part of task identity. Cross-representation tests show agreement in legal actions, successor states, and terminal outcomes across hundreds of sampled and constructed positions, complemented by exhaustive checks of all 32 operator masks and 512 opposing-material subsets. Controlled diagnostics further show that uniform witness-level sampling alters endpoint-action probabilities, while omitting remaining time or victory settings can merge states with different terminal behavior. These results show that action semantics, time, and victory configuration are not merely implementation details. They are part of the experimental task itself and must be represented explicitly for reproducible game AI.

Index Terms—Vietnamese Mathematical Chess, game description language, OpenSpiel, conformance testing, reproducibility

## I. INTRODUCTION

Vietnamese Mathematical Chess is a Vietnamese arithmetic board game with distinctive movement and capture rules [1]. We use it as a case study for a broader reproducibility problem in game AI: when do different implementations actually define the same decision problem?

The game exposes two representation issues that generalize beyond its particular rules. First, a single capture can have multiple arithmetic justifications even though all of them produce the same state transition. If these justifications are represented as separate actions, an implementation can change the probability of selecting a move without changing the set of reachable board positions. Second, identical board positions can have different terminal behavior under different time or victory settings. An environment that omits these variables can therefore merge states that require different outcomes. These examples illustrate a broader point: implementation choices that appear representational can silently change the task presented to an agent.

A representation contract must therefore specify not only which moves and transitions are legal, but also which distinctions are part of the agent’s decision problem. This matters when comparing search and learning systems. If two environments differ in action semantics, time information or victory configuration, differences in agent behavior cannot be attributed cleanly to the algorithm alone.

CoToan<sup>1</sup> addresses this problem through versioned executable rules, a declarative Game Description Language (GDL) description, an OpenSpiel environment and reproducible conformance diagnostics. The design separates three concerns: the mechanics and settings that define a task, the arithmetic witnesses that explain why an action is legal, and the interface through which an agent observes and acts on that task. Our contributions are:

C1. Executable task definition. We specify arithmetic capture, path blocking, capture-value scoring and ordered terminal conditions as an executable and configurable task definition. The specification explicitly separates invariant movement mechanics from victory settings that alter task semantics.

C2. Cross-representation contract. We define a correspondence between declarative GDL states and actions and the sequential OpenSpiel interface, preserving legal actions, state transitions, time and configuration identity across representations. Arithmetic witnesses are retained as explanatory metadata rather than exposed as additional agent actions.

C3. Conformance evidence and representation diagnostics. We compare a separately written move predicate, a declarative GDL realization and the OpenSpiel interface across sampled, constructed and configuration-varying states. Controlled diagnostics isolate probability distortion under uniform witnesslevel sampling and state aliasing caused by omitted time or victory settings.

We investigate three research questions. RQ1 asks whether the GDL, executable-rule and OpenSpiel representations agree on legal actions, successor states and outcomes across supported configurations. RQ2 asks whether representing arithmetic witnesses as distinct actions changes action sampling even when the resulting state transitions are identical. RQ3 asks whether board occupancy and the player to move are sufficient state variables under finite-horizon and configurable victory rules.

Together, these questions identify three requirements for decision-problem equivalence: consistent state transitions, unambiguous action identity and sufficient state information for adjudication. Our focus is reproducibility of the decision problem presented to agents, rather than reproducibility of training outcomes or playing strength.

## II. THE GAME: BOARD, MOVEMENT AND ARITHMETIC CAPTURE

## A. Board and objective

The board contains nine files and eleven ranks. Each side initially has one piece of each value from 0 through 9, with its zero fixed [1]. Fig. 1 fixes the coordinates: Red starts at the lower edge and moves first by default. The players alternate one move at a time.

![](images/2d8afe22076bd912302dc3c9993c656a69f7671ee09e458775587eaff1bb4183.jpg)  
Fig. 1. Initial arrangement in the research profile. Numerals are piece values; R and B identify the two sides independently of color. The crossed squares e2 and e10 are the fixed zero locations.

Capturing the opponent’s zero wins. A zero can support a capture but cannot move. Other captures add the target’s value to the score and may trigger a configured victory. There is no check, promotion or compulsory capture. The research profile exposes five selectable arithmetic result types: addition, subtraction, multiplication, integer quotient and remainder. All five are enabled by default.

## B. Ordinary movement

A nonzero piece of value a can move between one and a squares in any horizontal, vertical or diagonal direction. Every traversed square, including the destination, must be empty. It cannot turn, jump over another piece or capture an enemy by ordinary sliding movement.

![](images/e23a0068fdcbf47057b90d30c2af2bb6afeea10f31f76dbf3227b05cbeaabc7d.jpg)  
A c4-e6: 2 empty diagonal steps, legal B c4-f7: 3 steps exceed value 2 C c4-e4: d4 is occupied, so blocked  
Fig. 2. Ordinary movement in a local board window. A is legal; B exceeds the piece’s range; C is blocked by the enemy on d4. The dashed extension is forbidden. Remote pieces are omitted.

In Fig. 2, the red 2 can move from c4 to e6 because the two-step diagonal is clear. The destination f7 exceeds its range. The destination e4 is within range but inaccessible because d4 is occupied. These are distinct geometric reasons for rejecting a move.

## C. Arithmetic capture

A capture requires an attacker, an adjacent friendly support and an enemy target on the same straight line. The attacker and support provide the arithmetic operands; the target’s value is not an operand. The result of an enabled arithmetic operation determines the target’s distance from the support, measured in squares along that line. The result must be positive, and all squares between the support and target must be empty. The attacker replaces the target and the support remains stationary.

In Fig. 3A, red 2 on c4 uses red 1 on d4: multiplication gives $2 \times 1 = 2$ , allowing the attacker to capture the target on f4, two squares beyond the support. Fig. 3B shows the resulting board: c4 is empty, d4 retains red 1 and f4 contains red 2. The blue target is removed and the turn changes.

Fig. 3C inserts blue 7 on e4, blocking the capture of f4. The blocker may nevertheless be captured: subtraction gives 2-1=1, reaching e4 one square beyond the support. Fig. 3D illustrates another issue. Both floor(8/3)=2 and 8 mod 3=2 justify c4xf4. They explain the same action and produce the same successor. Section IV makes this distinction part of the action contract, and Section V measures its effect on sampling.

![](images/29ae83c2e71387b354ed3a5a4b8a2e253437f14b83d2833ccbb94105ca9df12d.jpg)  
Fig. 3. Arithmetic capture on rank 4. A and B show a legal move before and after execution. C blocks the target on f4. D gives two arithmetic witnesses for one action. These are constructed local examples; reachability is not assumed.

## D. Scoring and victory settings

The zero is captured by the same procedure: red 2 on e7 with red 1 on e8 can capture blue zero on e10 through multiplication if e9 is empty. This wins regardless of the score.

Scores start at zero and accumulate captured values. A positive threshold enables immediate victory when a player’s score reaches or exceeds it; zero disables that condition. At the default research horizon of 300 plies, score adjudication awards victory to the higher score and a draw to equal scores. One ply is one player’s move. Threshold victory and horizon adjudication are separate settings; enabling a threshold does not remove the finite research limit.

We evaluate terminal conditions in this order: zero capture, threshold victory, horizon adjudication, then no-move loss. The last condition is an explicit environment convention for an otherwise nonterminal player with no action. An unconditional horizon draw is retained as an optional experimental setting, not the default rule. Clocks and an unbounded-duration mode are not modeled. Every configuration has its own task identifier.

## III. RELATED WORK

## A. Game description and interoperability

PGN and its associated FEN specification support chess game and position interchange [2]. Such records do not define the arithmetic transition rules of a different game. General Game Playing instead communicates rules independently of the player [3]. Original GDL describes deterministic, complete-information games; Thielscher’s extension addresses randomness and incomplete information [4]. Our profile requires only the original language, here called GDL-I.

Regular Boardgames offers a regular-language formal ism for board games [5], while Ludii organizes descriptions through compositional ludemes [6]. OpenSpiel supplies algorithm-facing game interfaces and research infrastructure [7]. These systems serve complementary purposes. We choose GDL for explicit relational semantics and OpenSpiel for integration with search and learning code. CoToan builds on this infrastructure with a game-specific profile and a checked correspondence across the declarative and agent-facing interfaces.

## B. Search, learning and reproducible evaluation

Shannon’s formulation of computer chess [8] and Knuth and Moore’s analysis of alpha-beta pruning [9] establish a classical search perspective. Deep Blue demonstrates a highly engineered chess system [10], whereas solving checkers establishes a game-theoretic outcome [11].

UCT provides a bandit-based planning method [12], and the MCTS literature describes how tree and rollout policies influence behavior [13]. AlphaGo combined learned policies and values with search [14]; AlphaGo Zero trained through self-play without human game data [15]; AlphaZero extended a general self-play approach to chess, shogi and Go [16]. MuZero instead plans with a learned model [17]. Together, these approaches motivate an algorithm-independent environment with well-defined legal actions, observations and rewards.

Pardo et al. distinguish task-defining time limits from training cutoffs and explain why omitting remaining time can violate the Markov property [18]. Our horizon diagnostic instantiates this established issue in a reachable Co toan position. It is not a new general result about finite-horizon learning.

Henderson et al. identify reproducibility and variability problems in deep reinforcement learning [19], and Agarwal et al. emphasize uncertainty-aware evaluation [20]. Our work addresses an earlier dependency: agreement on the task itself. Inspired by property-based testing [21], we combine examples with invariants and separately implemented checks. The contribution is a versioned Co toan specification and conformance evidence, not a new general game language or search algorithm.

## IV. EXECUTABLE SPECIFICATION AND REPRESENTATION CONTRACT

## A. State and legal actions

Let $S = ( B , p , n ; \theta )$ , where B records occupancy, p is the controller and n is elapsed plies. Immutable configuration $\theta =$ $( H , T , h , O , f _ { 0 } )$ specifies the horizon, score threshold, horizon rule, operator mask and starting player. A well-formed state has unique piece identities per side, anchored surviving zeros, at least one surviving zero and $0 \leq n \leq H$ . With both zeros present and a positive threshold, both players cannot already have met it: play stops at the first crossing, so such snapshots are rejected. For zero-based file $f$ and rank $r ,$ the square index is $q = 9 ( r - 1 ) + f .$

Using two-dimensional square coordinates, let a nonzero attacker of value a occupy s, and let $d$ be one of eight unit directions. An ordinary move to $s { + L d }$ requires $1 \leq L \leq a$ and an empty path including the destination. A capture requires friendly support of value b at $s + d .$ Let m be the arithmetic result, measured as the distance from the support to the target. A valid capture requires $m \geq 1$ , an enemy target at $s + ( m +$ $1 ) d ,$ and empty squares $s + k d$ for $2 \leq k \leq m$ . The permitted arithmetic results are:

TABLE I  
ARITHMETIC CAPTURE RESULTS MEASURED FROM THE SUPPORT.
<table><tr><td>Operation</td><td>Distance m from sup- port</td><td>Guard</td></tr><tr><td>Addition</td><td>(a + b) mod 10</td><td>Always</td></tr><tr><td>Subtraction</td><td> $a - b$ </td><td> $a > b$ </td></tr><tr><td>Multiplication</td><td>(ab) mod 10</td><td>Always</td></tr><tr><td>Division</td><td> $\lfloor a / b \rfloor$ </td><td> $a > b > 0$ </td></tr><tr><td>Remainder</td><td> $\bar { a } \bmod b$ </td><td> $a > b > 0$ </td></tr></table>

All capture results must satisfy $m \ \geq \ 1 ;$ a zero result would point to the support square itself. The additional step in the target coordinate accounts for the support’s position adjacent to the attacker. Division is floored, not restricted to exact divisibility. Only enabled operators contribute witnesses. These are the normative movement rules of this specification; the historical citations provide attribution, not the operational equations. A transition clears s, replaces the destination, preserves other cells, toggles $p$ and increments n. Ordered adjudication follows Section II-D.

Let $V _ { j } ( B )$ contain player $j ^ { \circ } \mathbf { s }$ surviving values. Player i’s captured-value score is

$$
C _ { i } ( B ) = 4 5 - \sum _ { v \in V _ { 1 - i } ( B ) } v .
$$

The initial sum is $ 4 5 ;$ captures only remove pieces and no rule creates them. Scores therefore follow from occupancy without independent history-dependent counters. Settings determine when these points decide the winner.

## B. Actions are transitions, not arithmetic witnesses

We identify an action by its source and destination. The GDL term is (move sSRC sDST); the OpenSpiel ID is 99\*SRC+DST. The ambient policy space has 9801 IDs, with illegal entries masked. Arithmetic witnesses remain available as explanatory metadata.

This choice takes the quotient of operator-labeled captures by endpoint equality. Within a fixed state, witnesses with the same endpoints share the same attacker, target, support and board update, so collapsing them preserves the successor relation. It does not preserve a policy that samples witnesses uniformly. If an endpoint action a has k(a) witnesses, with $\mathrm { k ( a ) } = 1$ for an ordinary move, the two distributions are:

$$
p _ { \mathrm { m o v e } } ( a ) = \frac { 1 } { | A ( S ) | } , \qquad p _ { \mathrm { w i t n e s s } } ( a ) = \frac { k ( a ) } { \sum _ { b \in A ( S ) } k ( b ) } .
$$

Thus arithmetic multiplicity is a potential policy confound, even when the displayed board transitions remain legal. The endpoint contract prevents an incidental choice of explanation from becoming an extra decision for the agent.

## C. Declarative encoding and one-step correspondence

GDL represents dynamic information with cell, control and ply facts. Static ray, between and first relations encode geometry. Facts tabulate all 90 arithmetic pairs and enabled operators. A finite fold over values 1 through 9 sums missing opposing material using addition-table facts, deriving each score without extra dynamic state. Threshold and score-comparison facts implement the chosen victory settings. No engine-specific arithmetic extension is needed. Candidate actions precede terminal status, avoiding a legal/terminal cycle; domains are finite with no recursion through negation.

GDL uses joint actions, whereas our OpenSpiel environment is sequential. We assign the off-turn role the sole action noop. Let Φ map a well-formed Python state to its GDL facts, and let $J _ { p } ( a )$ pair the controlling player’s action with the other player’s noop. Write $T _ { P }$ and $T _ { G }$ for the respective transition functions.

Proposition 1 (one-step correspondence). Under the specified rules, for fixed θ and a well-formed nonterminal state, controlling-player legal actions correspond bijectively to legal GDL joint actions. For each such action,

$$
\Phi ( T _ { P } ( S , a ) ) = T _ { G } ( \Phi ( S ) , J _ { p } ( a ) ) .
$$

Proof sketch. Rays enumerate the permitted directions and distances. Range, support, enabled-operation and occupancy predicates impose the same movement conditions as the imperative generator. GDL set semantics and endpoint deduplication identify the same actions. Both updates clear the source, replace the destination, preserve other cells, toggle the controller and increment the ply. Induction over the nine score-fold stages yields the same captured-value totals. Both realizations then apply the ordered zero, threshold, horizon and mobility tests. Legality, successor states and terminal outcomes therefore correspond. Utility conversion is $u = 2 g / 1 0 0 - 1$ for GDL goals $g \in \{ 0 , 5 0 , 1 0 0 \}$

The argument concerns the representation definitions. The tests below check their executable realizations, including the generated facts and adapter behavior.

## D. Observation and framework contract

The registered OpenSpiel game, python\_cotoan, accepts all five configuration fields. Player IDs 0 and 1 denote Red and Blue; terminal returns are -1, 0 or 1. The observation contains 20 identity planes, a controller plane, a remaining-budget plane and five configuration scalars, totaling 2183 float32 entries. Colors are absolute; supported horizons 1 through 10000 and the other integer settings are exactly representable.

Time awareness is required by a task-defining horizon [18]; H − n is one sufficient encoding. For a fixed game, configuration can be implicit in the environment. We additionally expose it so pooled agents and datasets can distinguish tasks. Identity planes, controller, budget and configuration recover the specified state and scores. Version-2 snapshots include the complete configuration and reject mismatched imports. OpenSpiel’s game-and-state serialization also preserves history. Structural validation is not a reachability proof.

## E. Reusing search agents

An agent need not interpret GDL at every search node. It can use OpenSpiel’s legal\_actions, child or clone/apply\_action, is\_terminal and returns methods. Heuristics remain agent components; a separate optimized engine can implement the same contract. Shared state-cache keys must distinguish configuration, board, controller and remaining budget. A port preserves behavior only when rules, evaluation conventions and action handling agree.

## V. EVALUATION

## A. Protocol and reproducibility

The conformance experiment uses a ray-based implementation and a separately written exhaustive source/destination predicate. Seed 20260901 produces ten trajectories of up to 30 plies, contributing 300 reachable state samples and 300 transitions. A further 300 structurally valid synthetic positions vary occupancy and piece placement. Samples need not be distinct. Complete move sets, rotation/color-swap behavior and snapshot round trips are compared on all 600 positions.

A second path translates the emitted stratified GDL subset to clingo. At H=12, 70 selected fixtures compare legality and outcomes. One legal action is sampled for each of 66 nonterminal fixtures, and its complete successor fact set is compared. Initial facts and all 90 arithmetic pairs are checked separately. The committed H=300 description is regenerated and its initial state and one successor are checked.

The default configuration uses all operators, no score threshold and score adjudication at H=300. OpenSpiel runs 20 simulation tests at H=12. A separate audit varies threshold, horizon rule, operator mask and starter across nine configurations; it also checks all 32 masks and 512 subsets of Blue’s nonzero material. Additional round trips check histories, observations and configuration mismatch rejection. An eight-simulation MCTS decision and a depth-two alpha-beta decision check usability, not strength. Figures are generated and checked by the engine. The environment is Python 3.12.13, OpenSpiel 2.0.2 and clingo 5.8.2.

## B. RQ1: Cross-representation conformance

The nine-configuration audit compares 210 valid states and one selected successor for each of 158 nonterminal cases across Python, GDL and OpenSpiel, plus 45 simulation tests. Six dual-threshold snapshots are rejected before comparison. These checks are distinct from the 70-state baseline. Exactdepth counts remain 87 and 7457 at depths one and two. A terminal position contributes no leaves at positive depth; the counts are regression fingerprints, not game-tree size estimates.

TABLE II  
CROSS-REPRESENTATION CONFORMANCE RESULTS.
<table><tr><td>Check</td><td>Scope</td><td>Result</td></tr><tr><td>Unit tests</td><td>41 tests</td><td>Passed</td></tr><tr><td>Independent move predicate</td><td>600 positions; 59707 legal-action instances</td><td>No mismatch</td></tr><tr><td>Symmetry and snap- shots</td><td>600 positions</td><td>No mismatch</td></tr><tr><td>GDL legality and out- comes</td><td>70 positions</td><td>No mismatch</td></tr><tr><td>GDL successor facts</td><td>66 transitions</td><td>No mismatch</td></tr><tr><td>Arithmetic facts</td><td>90 value pairs</td><td>No mismatch</td></tr><tr><td>OpenSpiel simulation tests</td><td>20 games, H=12</td><td>Passed</td></tr><tr><td>Configuration audit</td><td>210 states; 158 succes- sors</td><td>No mismatch</td></tr><tr><td>Additional simulation tests</td><td>45 games; 9 settings</td><td>Passed</td></tr><tr><td>Operator-mask</td><td>32 masks</td><td>No mismatch</td></tr><tr><td>comparisons Relational score fold</td><td>512 material subsets</td><td>No mismatch</td></tr><tr><td>Opening exact-depth counts</td><td>Depths 1 and 2</td><td>87; 7457</td></tr></table>

## C. RQ2: Arithmetic witnesses change action sampling

We re-examine the 66 nonterminal public fixtures, assigning one entry to each ordinary action and one entry per valid arithmetic witness to each capture. Six fixtures contain duplicated endpoint actions. Across the fixtures, 6023 endpoint-action instances become 6031 witness-labeled instances. This is a diagnostic set with constructed and sampled positions, not a prevalence estimate for normal play.

The controlled position in Fig. 3D makes the effect explicit. Including the remote pieces recorded in its fixture, it has 43 endpoint actions and 44 witness-labeled actions. Uniform endpoint sampling selects c4xf4 with probability 1/43; uniform witness sampling selects it with probability 2/44=1/22. The board update is identical, but its probability is nearly doubled. Hence matching legal successors alone is insufficient to guarantee a consistent action-sampling contract.

## D. RQ3: Horizon omission aliases reachable states

Starting from the initial position, the legal four-ply cycle a1- a2, i11-i10, a2-a1, i10-i11 restores the board and side to move. Repeating the cycle under H=300 yields two nonterminal states at plies 3 and 299 with identical occupancy, controller and legal actions. Applying i10-i11 to both produces identical boards and controllers again, but the early successor at ply 4 is nonterminal while the successor at ply 300 is a draw.

No capture occurs in this cycle, so both scores remain zero: the late draw follows from equal-score adjudication, not an unconditional draw rule. This instantiates Pardo et al.’s timeawareness issue [18]. A board-and-controller encoding merges states with different terminal transitions; the budget channel separates them.

Configuration matters independently of time. In a constructed position containing all initial pieces except Blue’s 9, Red leads 9-0. At the same limit, score adjudication returns a Red win while the optional draw setting returns a draw. Below the limit, threshold 9 ends the game whereas threshold 0 does not. These configurations share occupancy, controller and ply but not task semantics. The diagnostics establish information requirements, not a measured learning benefit.

## E. Limitations and validity

The specification is normative for the supported configurations, not a survey of every historical or tournament variant. No-move loss and unconditional horizon draw are explicitly identified research conventions. The 2015 sources establish attribution and context rather than the operational equations. GDL checks use our subset translator and clingo, not an independent GGP competition runtime. The correspondence argument is not mechanized, and tests do not exhaust reachable states.

Synthetic fixtures and short-horizon simulations target correctness rather than typical strategic behavior. The sampling and aliasing diagnostics establish concrete representation effects but do not quantify their consequences for learning strength, game balance or educational outcomes. A subsequent agent study should match computational budgets, exchange colors, disclose seeds and report uncertainty [19], [20]. A horizon or rule change must identify a different task rather than be pooled silently into the same benchmark.

## VI. CONCLUSION

CoToan establishes an executable representation contract for Vietnamese Mathematical Chess across GDL and OpenSpiel, with no mismatches found in the evaluated cases. The diagnostics show that matching state transitions alone is insufficient: arithmetic witnesses can alter action-sampling probabilities, while omitted time or victory settings can merge states with different terminal outcomes. Reproducible game AI evaluation therefore requires consistent action identity, sufficient state information and explicit task configuration.

## ACKNOWLEDGMENTS

This work is financially supported by VinUniversity under Grant No CECS\_UNI.2026.DA017.

## REFERENCES

[1] N. D. Do, “Co toan Viet Nam: San pham tri tue mang dam triet li Viet,” VNU Media, interview, December 4, 2015. [Online]. Available: https://vnu.edu.vn/ co-toan-viet-nam-san-pham-tri-tue-mang-dam-triet-li-viet-post18673. html

[2] Interested readers of the Internet newsgroup rec.games.chess, Portable Game Notation Specification and Implementation Guide, 1994, coordinated by Steven J. Edwards. [Online]. Available: https: //www.saremba.de/chessgml/standards/pgn/pgn-complete.htm

[3] M. Genesereth, N. Love, and B. Pell, “General Game Playing: Overview of the AAAI Competition,” AI Magazine, vol. 26, no. 2, pp. 62–72, 2005. [Online]. Available: https://doi.org/10.1609/aimag.v26i2.1813

[4] M. Thielscher, “A General Game Description Language for Incomplete Information Games,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 24, no. 1, 2010, pp. 994–999. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/7647

[5] J. Kowalski, M. Mika, J. Sutowicz, and M. Szykuła, “Regular Boardgames,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 1, 2019, pp. 1699–1706. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/3991

[6] E. Piette, D. J. N. J. Soemers, M. Stephenson, C. F. Sironi, M. H. M. Winands, and C. Browne, “Ludii - The Ludemic General Game System,” in ECAI 2020, Frontiers in Artificial Intelligence and Applications, vol. 325, 2020, pp. 411–418. [Online]. Available: https://doi.org/10.3233/FAIA200120

[7] M. Lanctot et al., “OpenSpiel: A Framework for Reinforcement Learning in Games,” arXiv:1908.09453, 2019, framework report. [Online]. Available: https://arxiv.org/abs/1908.09453

[8] C. E. Shannon, “Programming a Computer for Playing Chess,” The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, vol. 41, no. 314, pp. 256–275, 1950. [Online]. Available: https://doi.org/10.1080/14786445008521796

[9] D. E. Knuth and R. W. Moore, “An Analysis of Alpha-Beta Pruning,” Artificial Intelligence, vol. 6, no. 4, pp. 293–326, 1975. [Online]. Available: https://doi.org/10.1016/0004-3702(75)90019-3

[10] M. Campbell, A. J. Hoane, Jr., and F.-h. Hsu, “Deep Blue,” Artificial Intelligence, vol. 134, no. 1–2, pp. 57–83, 2002. [Online]. Available: https://research.ibm.com/publications/deep-blue

[11] J. Schaeffer, N. Burch, Y. Bjornsson, A. Kishimoto, M. Muller, R. Lake, P. Lu, and S. Sutphen, “Checkers Is Solved,” Science, vol. 317, no. 5844, pp. 1518–1522, 2007. [Online]. Available: https://doi.org/10.1126/science.1144079

[12] L. Kocsis and C. Szepesvari, “Bandit Based Monte-Carlo Planning,” in ECML 2006, Lecture Notes in Computer Science, vol. 4212, 2006, pp. 282–293. [Online]. Available: https://link.springer.com/chapter/10. 1007/11871842\_29

[13] C. B. Browne, E. Powley, D. Whitehouse, S. M. Lucas, P. I. Cowling, P. Rohlfshagen, S. Tavener, D. Perez, S. Samothrakis, and S. Colton, “A Survey of Monte Carlo Tree Search Methods,” IEEE Transactions on Computational Intelligence and AI in Games, vol. 4, no. 1, pp. 1–43, 2012. [Online]. Available: https://doi.org/10.1109/TCIAIG.2012. 2186810

[14] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot et al., “Mastering the game of go with deep neural networks and tree search,” nature, vol. 529, no. 7587, pp. 484–489, 2016.

[15] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton et al., “Mastering the game of go without human knowledge,” nature, vol. 550, no. 7676, pp. 354–359, 2017.

[16] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel et al., “A general reinforcement learning algorithm that masters chess, shogi, and go through self-play,” Science, vol. 362, no. 6419, pp. 1140–1144, 2018.

[17] J. Schrittwieser et al., “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model,” Nature, vol. 588, pp. 604–609, 2020. [Online]. Available: https://www.nature.com/articles/ s41586-020-03051-4

[18] F. Pardo, A. Tavakoli, V. Levdik, and P. Kormushev, “Time Limits in Reinforcement Learning,” in Proceedings of the 35th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 80, 2018, pp. 4045–4054. [Online]. Available: https://proceedings.mlr.press/v80/pardo18a.html

[19] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep Reinforcement Learning That Matters,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018. [Online]. Available: https://ojs.aaai.org/index.php/aaai/article/view/11694

[20] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare, “Deep Reinforcement Learning at the Edge of the Statistical Precipice,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 29 304– 29 320. [Online]. Available: https://proceedings.neurips.cc/paper/2021/ hash/f514cec81cb148559cf475e7426eed5e-Abstract.html

[21] K. Claessen and J. Hughes, “QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs,” in Proceedings of the ACM SIGPLAN International Conference on Functional Programming, 2000, pp. 268–279. [Online]. Available: https://doi.org/10.1145/351240. 351266