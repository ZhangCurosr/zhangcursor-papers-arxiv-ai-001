# DIRECTING TO DIVERSIFY: MULTI-DIRECTIONAL SEARCH FOR AUTOMATIC HEURISTIC DESIGN

Jianghan Zhu<sup>1</sup>, Jianan Zhou<sup>2</sup> , Fei Liu<sup>3,4</sup>, Ni Zhang<sup>1</sup>, Yaoxin Wu<sup>5</sup>, Zhuoyi Lin<sup>6</sup>, Zhiguang Cao<sup>1</sup>

<sup>1</sup>Singapore Management University <sup>2</sup>Nanyang Technological University

<sup>3</sup>University of Zurich <sup>4</sup>ETH Zurich <sup>5</sup>Eindhoven University of Technology <sup>6</sup>A\*STAR

## ABSTRACT

Large language models (LLMs) have demonstrated strong potential to automate heuristic design for NP-hard combinatorial optimization (CO) tasks without human intervention. Recent methods predominantly rely on evolutionary computation (EC) frameworks, where heuristic diversification is mainly induced through EC-like operators or downstream selection, while the direction of heuristic transformation remains largely unharnessed. In this paper, we propose a multi-directional Perturbation–Rewrite Search (PRS) framework for automatic heuristic design that factorizes heuristic transformation through directional search primitives at the generation level. Specifically, PRS operates on heuristic ideas through three complementary directions: expansion toward richer and more expressive designs, compression toward simpler and less specialized designs, and steering for finegrained refinement, together enabling broader and more sustained exploration of diverse regions in the heuristic design space. PRS further organizes candidate exploration within a breadth-first search-based process with adaptive operator control, which ensures thorough candidate utilization and sustained exploration, and maintains strong search dynamics while preventing premature convergence. Extensive empirical results across diverse CO tasks show that PRS consistently discovers high-quality heuristics, demonstrating the effectiveness and generality of explicitly directional exploration design.

## 1 INTRODUCTION

Automatic heuristic design (AHD) aims to automatically discover high-performing heuristic algorithms for a wide range of optimization problems (He et al., 2011; Dahmani et al., 2024; Novikov et al., 2025). Despite its importance, traditional AHD methods, such as genetic programming, remain fundamentally constrained by their reliance on manually designed, problem-specific operators, which limits scalability and restricts the exploration of the heuristic search space. The recent emergence of large language models (LLMs) has opened a promising new direction for AHD (Liu et al., 2024b), enabling automated heuristic generation through LLM-based code synthesis and reasoning.

Most existing LLM-based AHD approaches build upon optimization frameworks (e.g., evolutionary computation (EC)), where heuristics are iteratively improved using LLMs (Romera-Paredes et al., 2024; Liu et al., 2024a; Ye et al., 2024; Zheng et al., 2025; Shi et al., 2026). Maintaining diverse heuristic candidates is important for preventing premature concentration around a narrow family of designs and for preserving access to alternative regions that may become useful at different stages of search. Recent works have attempted to enhance the search process with more structured control, such as diversity-aware population management (Dat et al., 2025) and feature-assisted partitioning of the abstract heuristic search space (Hu & Zhang, 2025). For example, EvoX (Liu et al., 2026) adaptively evolves the search strategy that governs parent selection and candidate variation, dynamically switching among refinement, structural variation, and free-form variation to balance exploitation and diversification according to search progress.

Despite these advances, existing approaches mainly encourage diversification through population management, candidate selection, similarity measures, or search-space partitioning. Such mechanisms regulate which candidates are retained, preferred, or grouped after or around generation, but they do not explicitly structure the generation process itself around different directions of heuristic change. As a result, heuristic diversification is typically an emergent consequence of generic mutation, crossover, reflection, or novelty pressure, rather than a directly controlled search primitive. This leaves an important gap: existing AHD frameworks lack explicit generation primitives for deliberately diversifying heuristic ideas along complementary semantic directions.

In this work, we argue that diversification should be controlled directly at the level of heuristic generation. To this end, we propose Perturbation–Rewrite Search (PRS), a multi-directional search framework that explicitly factorizes heuristic generation into complementary search primitives over the idea space. PRS introduces three perturbation strategies, i.e., expansion, compression, and steering, which deliberately transform a parent heuristic toward richer, simpler, and locally refined designs, respectively. Rather than using diversity only as a population property or selection signal, PRS controls how new heuristic ideas are produced before they are realized as executable implementations. These abstraction-level perturbations are then instantiated through an LLM-based rewriting process that translates the modified heuristic specification into executable code while preserving its intended decision logic. Together, this primitive-level directional diversification enables broader and more sustained coverage of structural regions in the heuristic space.

To further enhance search effectiveness, PRS organizes candidate exploration within a breadth-first search (BFS) framework. By expanding multiple candidates layer by layer, PRS improves candidate utilization while preventing early domination by a small subset of heuristics. This search strategy is coupled with an adaptive exploration control mechanism, enabling sustained exploration and robustness against stagnation.

Our contributions are summarized as follows:

• We introduce Perturbation–Rewrite Search (PRS) for automatic heuristic design, which makes directional heuristic transformation an explicit generation-level search primitive for structured exploration of the heuristic idea space.

• We develop a multi-directional perturbation mechanism with expansion, compression, and steering, which deliberately transforms heuristic ideas toward richer, simpler, and locally refined designs, respectively. PRS further integrates bounded BFS-based search and adaptive operator control to allocate search effort across these complementary directions.

• We comprehensively evaluate PRS across 11 combinatorial optimization problem settings and show that it consistently discovers high-quality heuristics, demonstrating strong effectiveness and generality compared to state-of-the-art LLM-based AHD methods.

## 2 PRELIMINARY

AHD. Let P denote a problem class, and let $S ( p )$ be the feasible solution set for an instance $p \in \mathcal P$ . A heuristic space is defined as H, where each heuristic $h \in \mathcal H$ induces a mapping $h : { \mathcal { P } } \to \bigcup _ { p \in { \mathcal { P } } } S ( p )$ and for any $p \in \mathcal { P } , h ( p ) \in \mathcal { S } ( p )$ . For a given validation set $\mathcal { D } \subseteq \mathcal { P }$ and an objective function $f : { \mathcal { S } }  \mathbb { R } ,$ , the goal of AHD is to find an optimal heuristic $h ^ { \star } \in \mathcal { H }$ that maximizes expected performance g:

$$
\boldsymbol h ^ { \star } = \arg \operatorname* { m a x } _ { \boldsymbol h \in \mathcal H } \boldsymbol g ( \boldsymbol h ) = \arg \operatorname* { m a x } _ { \boldsymbol h \in \mathcal H } \mathbb E _ { \boldsymbol x \sim \mathcal D } f ( \boldsymbol h ( \boldsymbol x ) ) .\tag{1}
$$

LLM-based AHD. LLM-based AHD leverages LLMs to navigate the typically vast heuristic space H in CO. In practice, the designed heuristic often constitutes only a key component of a broader optimization algorithm. For example, in Travelling Salesman Problem (TSP), existing LLM-based AHD methods (Liu et al., 2024a) typically focus on designing step-by-step constructive heuristics that select the next node based on the current partial tour. Beyond constructive frameworks, recent studies have also explored learning heuristics that update weighted distance matrices within the broader Guided Local Search (GLS) algorithm, thereby more effectively guiding local search decisions (Ye et al., 2024; Zheng et al., 2025).

A prominent line of LLM-based AHD methods follows an evolutionary paradigm (Liu et al., 2023; Romera-Paredes et al., 2024; Liu et al., 2024a; Ye et al., 2024). Using EC-style operators such as crossover and mutation, these methods progressively explore the heuristic space, with more recent extensions further strengthening this paradigm. For example, HiFo-Prompt (Chen et al., 2026)

introduces hierarchical prompt-level guidance through hindsight–foresight control and knowledge distillation for faster and more efficient heuristic search.

Despite differences in evolution strategies, these frameworks remain rooted in EC-like search, where search operators typically recombine multiple heuristics in crossover to inherit parental features, together with mutation-style operators to promote diversification. Consequently, the direction of heuristic change is largely uncontrolled by the search mechanism itself: different transformation patterns may emerge during evolution, but they are not explicitly harnessed as distinct search primitives. In contrast, PRS directly structures heuristic generation through complementary transformation directions, enabling multi-directional exploration without relying primarily on recombination of existing structures.

Breadth First Search. Breadth-first search is a fundamental graph traversal strategy that explores states layer by layer by maintaining a frontier of candidate states at each depth. Recent work such as Tree-of-Thoughts (Yao et al., 2023) instantiates BFS in the space of intermediate reasoning trajectories, where each layer corresponds to a single reasoning step and the frontier is pruned to a fixed breadth. In a generic BFS procedure, the search starts from an initial state and expands all states in the current frontier using a predefined expansion operator. The resulting states are then scored, and a subset of top-ranked states is retained as the frontier for the next layer. This process repeats until a stopping criterion, such as a maximum search depth or a computational budget, is reached.

## 3 METHODOLOGY

In this section, we introduce PRS, a multi-directional perturbation–rewrite search framework that makes directional heuristic transformation an explicit search primitive for automatic heuristic design. PRS explores the heuristic space by applying multi-directional perturbations to heuristic ideas and then rewriting the corresponding implementations to realize the modified heuristic as executable candidates. At the search level, PRS maintains a heuristic pool and organizes candidate expansion in a breadth-first manner with selective pruning to enhance candidate utilization. In addition, PRS incorporates an adaptive exploration controller to regulate the exploration–exploitation trade-off and mitigate premature convergence during heuristic search.

PRS Pipeline. The overall pipeline of PRS is illustrated in Fig. 1. Specifically, PRS starts the heuristic search by initializing a heuristic pool, which maintains a set of parent heuristics with a selection mechanism, and terminates after a certain condition. We introduce a novel multi-directional perturbation–rewrite operator that explores the heuristic space through complementary transformation directions. It includes expansion, compression, and steering, which transform heuristic ideas along complementary

![](images/f0c8bd3d005401b8fd4522d6e32a5a65afcbed4dd594d0a7db97df4518cb8172.jpg)  
Figure 1: An illustration of PRS pipeline.

directions in the heuristic design space. These operators are used to expand parent nodes within a BFS framework, as illustrated in Fig 2. Regulated by an adaptive exploration controller, these operators jointly facilitate an effective search for the optimal heuristic $h ^ { * }$ for a given task. After each completed BFS tree, random sampling is applied to the heuristic pool to promote diversity and retain temporarily inferior yet potentially promising candidates, while always preserving the current best heuristic. The complete PRS algorithm and associated prompts are presented in Algorithm 1 and Appendix D.1, respectively. In the following subsections, we describe the perturbation and rewrite process, the BFS-based search structure, and the adaptive exploration controller in detail.

## 3.1 PERTURBATION-REWRITE OPERATOR

PRS introduces a multi-directional perturbation–rewrite process composed of a perturbation step followed by an LLM-based rewrite step. Rather than directly modifying executable heuristics through carefully crafted EC operators, PRS applies directional perturbations to heuristic ideas and subsequently rewrites the corresponding implementations to realize the modified heuristics as executable candidates. This design separates the intended algorithmic modification from its lowlevel implementation details, enabling controlled changes in heuristic structure while preserving the executability and semantic coherence required for black-box heuristic evaluation. We denote $h _ { p }$ as the parent heuristic used for perturbation–rewrite search, and elaborate the perturbation and rewriting strategies below.

Multi-directional Perturbation. The goal of multi-directional perturbation is to explore the heuristic space through complementary transformation directions, thereby overcoming the localized perturbation nature of traditional mutation operators. This design is motivated by the fact that heuristic quality does not vary monotonically with structural complexity: overly simple heuristics may lack expressive power, whereas overly complex ones may become bulky, computationally expensive, or over-fit to in-distributional data. Accordingly, PRS employs three complementary perturbation actions that address different search needs: expansion increases heuristic expressiveness, compression counteracts excessive specialization, and steering provides refinement around promising heuristic structures.

Expansion $( \Phi _ { \theta } ^ { e x p n } )$ . The expansion operator explicitly prompts LLMs to enrich the parent heuristic by introducing additional decision signals, interactions, conditional rules, or adaptive mechanisms, deliberately increasing its structural complexity and expressive capacity. Specifically, expansion instructs LLMs to inject new structural components, such as additional scoring terms or conditional logic. The operator further specifies how these additions affect heuristic decisions rather than simply appending independent components. From an optimization perspective, expansion biases heuristic structures toward more elaborate and expressive programmatic forms, such as deeper control flow or richer operator compositions, thereby increasing its expressive capacity and conceptual richness.

Compression $( \Phi _ { \theta } ^ { c o m p } )$ . Complementary to expansion, PRS employs a compression operator that biases LLMs toward reducing the structural complexity of the current parent heuristic by removing redundant mechanisms or over-specialized components that may hinder out-of-distribution (OOD) generalization (e.g., structures tightly coupled to specific problem sizes or data distributions). Intuitively, compression may disrupt the established advantages of the parent heuristic, as simplification can often lead to short-term performance degradation. However, from a broader perspective, it may mitigate the accumulation of dominant structural priors and excessive complexity, acting as a potential refactoring mechanism in the heuristic space that counteracts over-specialization and promotes more robust heuristic representations.

Steering $( \Phi _ { \theta } ^ { s t e e r } )$ . While expansion and compression enable substantial shifts toward richer or simpler heuristic designs, they are less suited for fine-grained exploitation of an already promising heuristic idea. Steering complements these directional perturbations by performing localized refinements while approximately preserving the overall heuristic structure, allowing PRS to further exploit promising regions without inducing another substantial structural drift.

Together, expansion, compression, and steering constitute a multi-directional perturbation mechanism that enables PRS to explore both richer and more generalized heuristics across complementary regimes. The perturbation process can be formally expressed as

$$
\widetilde { z } \ = \ \Phi _ { \theta } ^ { ( o ) } ( z _ { p } \mid { \mathcal C } ) ,\tag{2}
$$

where $z _ { p }$ denotes the heuristic idea of the parent heuristic, z˜ denotes the perturbed heuristic idea, C contains the task and function descriptions, and $\Phi _ { \theta } ^ { ( o ) } \in \{ \Phi _ { \theta } ^ { \mathrm { s t e e r } } , \Phi _ { \theta } ^ { \mathrm { e x p n } } , \Phi _ { \theta } ^ { \mathrm { c o m p } } \}$ denotes the selected perturbation operator. At each perturbation step, one of these actions is adaptively selected according to its recent search utility.

Rewrite. Following multi-directional perturbation, the perturbed offspring idea specifies the intended algorithmic change but does not directly constitute an executable heuristic. The rewrite step therefore prompts the LLM to realize the offspring idea by rewriting the complete parent implementation, preserving existing mechanisms when consistent with the target idea. PRS independently rewrites the same offspring idea into multiple offspring candidates according to the beam size $\kappa ,$ allowing different executable realizations of the same heuristic perturbation. The rewrite process can be formally expressed as:

$$
h _ { j } ^ { \prime } \sim \Psi _ { \theta } ( h _ { p } , \tilde { z } , o \mid \mathcal C ) , \qquad j = 1 , \dotsc , \kappa ,\tag{3}
$$

where $\Psi _ { \theta }$ denotes the LLM-based rewrite operator, $h _ { p }$ is the parent heuristic implementation, z˜ is the perturbed heuristic idea produced by operator $^ { O , }$ and κ denotes the beam size.

![](images/23a180ed2ed7ea7aae50bffdac4e4abb30dbf6098baab448178d624c6f2bcb15.jpg)  
Figure 2: An overview of BFS-based Heuristic Search. Starting from a root parent heuristic with 1.Each path is a Markov chain: parent = previous nodeits corresponding idea, PRS performs a BFS-style expansion in which new offspring ideas are generated via multi-directional perturbation and then realized as executable code through rewriting. The offspring codes are then evaluated and selectively retained in a fixed-size frontier at each depth. Terminal nodes correspond to pruned or non-extendable offspring. Dark green boxes denote heuristics that outperform the root parent, while light green (or light red) boxes represent worse-than-root offspring that are accepted (or rejected) for inclusion in the next frontier. Each parent heuristic across BFS paths retains opportunity for further expansion, preventing early domination by a single trajectory. The beam size is set to $\kappa = 3$ for illustration.

## 3.2 BFS-BASED HEURISTIC SEARCH

We employ a bounded BFS-style search mechanism in conjunction with the perturbation–rewrite operator, as BFS naturally promotes candidate utilization through layer-by-layer expansion and prevents early domination by a small subset of heuristics. At the root level, one parent heuristic is selected from the current heuristic pool using an ϵ-greedy strategy based on its performance. Let $h ^ { * }$ denote the best heuristic in the current pool ${ \bar { \mathcal { P } } } .$ At the root level of each BFS tree, $h ^ { * }$ is selected as the root parent with probability $1 - \epsilon ,$ while each of the remaining heuristics is selected with probability $\frac { \epsilon } { | \mathcal { P } | - 1 }$ . We set $\epsilon = 0 . 5$ in all experiments to balance exploitation of the current best heuristic and exploration of alternative parents.

As shown in Fig. 2, for each non-terminal parent at the current BFS frontier, one perturbation operator is selected to generate offspring candidates through the perturbation–rewrite process. After all parents in the current frontier have been expanded, the resulting offspring at the same depth are evaluated based on their average performance on the validation dataset. A subset of candidates is then retained as the next frontier through softmax-based sampling conditioned on their performance scores. For a candidate set $\tilde { \mathcal { F } }$ at the current BFS depth, the probability of retaining a candidate $h _ { i }$ is defined as:

$$
\begin{array} { r } { p ( h _ { i } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } h _ { i } = \arg \operatorname* { m a x } _ { h \in \widetilde { \mathcal { F } } } g ( h ) , } \\ { \mathrm { S o f t m a x } ( g ( h _ { i } ) ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{4}
$$

Each retained offspring serves as a parent for the next BFS layer. The BFS expansion terminates when the maximum tree depth is reached or the remaining evaluation budget is exhausted. After each completed BFS tree, the surviving frontier candidates are immediately merged with the current heuristic pool, where the best heuristic is preserved and the remaining pool members are sampled randomly to maintain diversity.

## 3.3 ADAPTIVE EXPLORATION CONTROLLER

The BFS-based heuristic search with the perturbation–rewrite operator constitutes the core of PRS. In this subsection, we introduce an adaptive exploration controller that serves as an auxiliary mechanism to balance exploration and exploitation, and to mitigate search stagnation in local optima.

Adaptive Operator Control. In evolutionary search, the balance between exploration and exploitation critically influences both search breadth and convergence quality. Excessive exploration can lead to random drift and insufficient refinement of promising elites, whereas overly conservative updates or exploitation may cause the search to collapse into a narrow region of the solution space. Previous studies have shown that static control policies are generally suboptimal in different search phases (Crepinšek et al., 2013).<sup>ˇ</sup>

To address this issue, PRS adaptively selects among expansion, compression, and steering using a UCB-based criterion according to their recent search utility. For an operator $o \in { \mathcal { O } } .$ , its selection score at evaluation step t is defined as

$$
U _ { t } ( o ) = \bar { R } _ { t } ( o ) + c _ { t } \sqrt { \frac { \log N _ { t } } { N _ { t } ( o ) } } ,\tag{5}
$$

where $\bar { R } _ { t } ( o )$ denotes the mean reward of operator o within a recent sliding window, $N _ { t } ( o )$ is the number of times o has been selected within the window, and $N _ { t }$ is the total number of recorded operator selections. For each parent expansion, the operator reward is computed as $R _ { t } ( o ) \ = \quad$ max $\begin{array} { r } { \left\{ 0 , \frac { g ( h _ { * } ^ { \prime } ) - g ( h _ { p } ) } { | g ( h _ { p } ) | } \right\} } \end{array}$ , where $h _ { * } ^ { \prime }$ is the best offspring generated from the corresponding parent. Operators that have not yet been observed in the current window are prioritized before applying the UCB criterion. The exploration coefficient $c _ { t }$ controls the trade-off between operator exploration and reward-guided selection, and follows a linear annealing schedule as a function of the evaluation step t. Let $\check { T }$ denote the maximum evaluation step. We define:

$$
c _ { t } = c _ { 0 } + ( c _ { T } - c _ { 0 } ) \frac { t } { T } ,\tag{6}
$$

where $c _ { 0 }$ and $c _ { T }$ are the initial and final exploration coefficients, respectively. This schedule encourages broader exploration of different perturbation directions at the beginning and gradually shifts toward exploiting operators with higher observed utility.

Stagnation-Aware Restart. While adaptive operator control improves the overall trade-off between exploration and exploitation, it cannot guarantee escape from stagnation in local optima (Dat et al., 2025). As the search progresses, repeatedly exploiting successful perturbation directions and promising parents may gradually concentrate the search around a narrow set of heuristic structures, making further improvements increasingly difficult. This behavior arises because the best-performing heuristics can repeatedly attract search effort within their local neighborhoods, leading to the proliferation of structurally similar candidates. As a result, the search may enter a plateau regime, where heuristic diversity diminishes due to premature convergence or inbreeding (Eshelman, 1991).

To address this issue, PRS triggers a stagnation-aware restart when no improvement in the global best heuristic is observed for a predefined number C of heuristic evaluations. Once activated, two coordinated actions are performed: a) Elite Restart, where the heuristic pool is reset to retain only the globally best heuristic discovered so far; and b) Operator Reheating, where the recent operator reward window is reset and the UCB exploration coefficient is temporarily restored to its initial value $c _ { 0 } ,$ , shifting operator selection back toward exploration before gradually returning to reward-guided exploitation. Together, these mechanisms enable the heuristic search to escape the current plateau by reintroducing diversity into the heuristic pool while ensuring smooth convergence through restarting from the elite heuristic. The complete PRS algorithm and associated prompts are presented in Algorithm 1 and Appendix D.1, respectively.

## 4 EXPERIMENTS

This section empirically evaluates PRS on some typical NP-hard COPs. The definitions of these COPs are listed in the Appendix B.1. Importantly, these problems are also applied to a wide range of general frameworks, including step-by-step constructive, guided-local search (GLS) and ant colony optimization (ACO). Details of these general frameworks are included in the Appendix B.2. The metrics used to measure the performance of each method primarily include the average objective and the average gap to the reference value (e.g., best-known solution) of each problem size on the test set. Our evaluated problem setting covers a wide range of problem types and general frameworks: 1) ACO: Capacitated Vehicle Routing Problem (CVRP), Orienteering Problem (OP) (both black-box and white-box), Multiple Knapsack Problem (MKP) (black-box); 2) Step-by-step Construction: Knapsack Problem (KP), Online Bin Packing Problem (OBPP), TSP, Admissible Set Problem (ASP); 3) GLS: TSP, Flow Shop Scheduling Problem (FSSP).

Implementation Setups. To maximize the effectiveness of BFS, we adopt a beam search with beam size κ = 3 and maximum tree depth d = 3, and set the frontier size (i.e., the maximum number of parent nodes for the next layer) to $f s = 3$ . For both validation datasets and testing datasets, we follow the procedures and configurations of the MCTS-AHD (Zheng et al., 2025) and use their random seeds to generate all datasets. In addition, to ensure a fair comparison, we adopt the same seed heuristics as MCTS-AHD. For all problems except OBPP, we set the number of evaluated heuristics to $T = 1 0 0 0$ and constrain all LLM-based AHD methods to a maximum budget of 5,000 LLM queries. The size of the heuristic pool is set to 10 for all problems. Comprehensive hyperparameter settings and experimentation setups can be found in the Appendix E. All experiments were conducted on a machine equipped with a single Intel Core i9-12900KF CPU.

Baselines. We compare PRS with the following baselines. They are mainly composed of four categories: (1) hand-crafted heuristics such as Greedy-Construct (GC) (Rosenkrantz et al., 1977), Best-Fit for OBPP (Kenyon et al., 1996), GPUTA (Gupta, 1971), ACO (Blum, 2005; Dorigo et al., 2006) algorithm for CVRP, OP, MKP, etc. (2) Traditional heuristic solvers, including ILS1 (Stützle, 1998), OR-Tools, etc. (3) Neural Combinatorial Optimization (NCO) methods including POMO (Kwon et al., 2020), DeepACO (Ye et al., 2023), etc. (4) Existing LLM-based AHD methods. They are: FunSearch (Romera-Paredes et al., 2024), EoH (Liu et al., 2024a), ReEvo (Ye et al., 2024), HSEvo (Dat et al., 2025) and MCTS-AHD (Zheng et al., 2025). All methods are run with the same initial seed heuristics and tested on identical validation datasets D. Unless otherwise stated, all LLM-based AHD methods are evaluated using GPT-5-nano and mean performance is based on three independent runs.

## 4.1 COMPARISON ON ACO FRAMEWORK

The ACO framework is a classical metaheuristic mechanism for solving combinatorial optimization problems by simulating pheromone-guided stochastic construction (Dorigo et al., 2006; Kim et al., 2024). By updating the pheromone matrix τ to determine the edge priority and crafting a more complicated heuristic matrix η to guide ants’ transition probabilities and assess the constructed tours, ACO can iteratively drive the solution towards optimal or near-optimal regions. As shown in Table 1, under GPT-5-nano, PRS consistently outperforms all compared LLM-based AHD baselines on the out-of-distribution settings of both CVRP and OP. On the in-distribution settings, PRS also achieves the best performance on CVRP and remains competitive on OP, outperforming EoH and MCTS-AHD. These results suggest that PRS is particularly effective in discovering heuristics with stronger out-of-distribution generalization under the ACO framework.

Table 1: Performance of ACO-based heuristics on CVRP and OP. All methods are evaluated on three test sets of 64 randomly generated instances each, following the setup in Dat et al. (2025); Zheng et al. (2025). Test sets whose scales match the validation sets are underlined. Optimal solutions are approximated using DeepACO with significantly more ants and iterations than those in the baselines. The best-performing method is shaded, and each test set’s overall best result is in bold.
<table><tr><td>Task</td><td colspan="6">CVRP</td><td colspan="6">OP</td></tr><tr><td>N=</td><td colspan="2">N=50</td><td colspan="2">N=100</td><td colspan="2">N=200</td><td colspan="2">N=50</td><td colspan="2">N=100</td><td colspan="2">N=200</td></tr><tr><td>Method</td><td colspan="2">Obj↓</td><td colspan="2">Obj↓</td><td colspan="2">Obj↓</td><td colspan="2">Obj↑ Gap↓</td><td colspan="2">Obj↑ Gap↓</td><td colspan="2">Obj↑ Gap↓</td></tr><tr><td>DeepACO</td><td>8.888</td><td>Gap↓</td><td>14.932</td><td>Gap↓</td><td>27.159</td><td>Gap↓</td><td>19.867</td><td></td><td>36.392</td><td></td><td>63.380</td><td></td></tr><tr><td>ACÓ</td><td>18.282</td><td>105.69%</td><td>30.030</td><td>101.11%</td><td>48.582 78.88%</td><td></td><td>13.570</td><td>31.70%</td><td>24.149</td><td>33.64%</td><td>37.109</td><td>41.45%</td></tr><tr><td colspan="9">LLM-based AHD (GPT-5-nano)</td><td></td><td></td><td></td></tr><tr><td>EOH</td><td>9.320</td><td>4.85%</td><td>16.511</td><td>10.57%</td><td>29.603</td><td>9.00%</td><td>15.162</td><td>23.68%</td><td>29.699</td><td>18.39%</td><td>53.497</td><td>15.59%</td></tr><tr><td>ReEvo</td><td>9.186</td><td>3.35%</td><td>16.020</td><td>7.29%</td><td>28.782</td><td>5.98%</td><td>15.290</td><td>23.04%</td><td>30.820</td><td>15.31%</td><td>54.459</td><td>14.08%</td></tr><tr><td>MCTS-AHD</td><td>9.025</td><td>1.54%</td><td>15.974</td><td>6.98%</td><td>28.572</td><td>5.20%</td><td>15.192</td><td>23.53%</td><td>30.752</td><td>15.50%</td><td>54.718</td><td>13.67%</td></tr><tr><td>Ours</td><td>8.958</td><td>0.79%</td><td>15.576</td><td>4.31%</td><td>28.228</td><td>3.93%</td><td>15.247</td><td>23.25%</td><td>30.924</td><td>15.03%</td><td>55.821</td><td>11.93%</td></tr></table>

## 4.2 COMPARISON ON KP AND OBPP

The step-by-step constructive framework incrementally builds a feasible solution by repeatedly selecting and adding one component at a time, where each decision is guided by a problemspecific heuristic and must satisfy the corresponding feasibility constraints (Asani et al., 2023). For KP, the heuristic scores the remaining items based on their values, weights, and current residual capacity, and iteratively selects the next item until no feasible additions are possible.

Table 2: Online bin packing problem results. The first four problem scales are used during validation, and the remaining are used for held-out test. Notations of problem scales are abbreviated, e.g., 1k\_500 indicates 1,000 items and a fixed bin capacity 500.
<table><tr><td>Test Sets</td><td>1k_100</td><td>1k_500</td><td>5k_100</td><td>5k_500</td><td>10k_100</td><td>10k_500</td><td>AVG</td></tr><tr><td colspan="8">LLM-based AHD (GPT-5-nano)</td></tr><tr><td>Best Fit First Fit</td><td>4.77% 5.02%</td><td>0.25% 0.25%</td><td>4.31% 4.65%</td><td>0.55% 0.55%</td><td>4.05% 4.36%</td><td>0.47% 0.50%</td><td>2.40% 2.56%</td></tr><tr><td>FunSearch</td><td>2.45%</td><td>0.66%</td><td>1.30%</td><td>0.25%</td><td>1.05%</td><td>0.21%</td><td>0.99%</td></tr><tr><td>EoH</td><td>2.69%</td><td>0.25%</td><td>1.63%</td><td>0.53%</td><td>1.47%</td><td>0.45%</td><td>1.17%</td></tr><tr><td>ReEvo</td><td>3.13%</td><td>0.25%</td><td>1.78%</td><td>0.55%</td><td>2.63%</td><td>0.50%</td><td>1.47%</td></tr><tr><td>HSEvo</td><td>2.24%</td><td>0.50%</td><td>1.46%</td><td>0.38%</td><td>1.30%</td><td>0.36%</td><td>1.04%</td></tr><tr><td>MCTS-AHD</td><td>1.92%</td><td>0.58%</td><td>1.08%</td><td>0.38%</td><td>0.88%</td><td>0.37%</td><td>0.87%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PRS (ours)</td><td>1.82%</td><td>0.18%</td><td>0.86%</td><td>0.35%</td><td>0.63%</td><td>0.45%</td><td>0.72%</td></tr></table>

For OBPP, items arrive sequentially and must be assigned immediately; the heuristic maps the incoming item size together with current bin states (e.g., remaining capacities) to priority scores over bins, placing each item into the best feasible bin. Specifically, Online-BPP instances are generated following EoH using the standard Weibull distribution (Castiñeiras et al., 2012). Regarding KP, Table 3 shows that PRS consistently achieves the lowest or near-lowest optimality gaps across all problem sizes and clearly outperforms existing LLM-based AHD baselines under GPT-5- nano. Table 2 further shows that online BPP heuristics designed by PRS have a consistently superior average performance across six test settings.

Table 3: Performance of step-by-step constructive heuristics on KP. Near-optimal objectives are obtained by OR-Tools. The best-performing method is shaded, and each test set’s overall best result is in bold.
<table><tr><td rowspan=1 colspan=4>Knapsack Problem (KP)</td></tr><tr><td rowspan=1 colspan=1>Test Sets</td><td rowspan=1 colspan=1>N=50</td><td rowspan=1 colspan=1>N=100</td><td rowspan=1 colspan=1>N=200</td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Obj.↑ Gap↓</td><td rowspan=1 colspan=1>Obj.↑ Gap↓</td><td rowspan=1 colspan=1>Obj.↑ Gap↓</td></tr><tr><td rowspan=1 colspan=1>OptimalGCPOMO</td><td rowspan=1 colspan=1>20.03719.8301.04%19.612 2.12%</td><td rowspan=1 colspan=1>40.27140.030 0.60%39.6761.48%</td><td rowspan=1 colspan=1>57.44857.1150.58%57.2710.31%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LLM-based AHD</td><td rowspan=1 colspan=1>(GPT-5-nano)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FunSearch</td><td rowspan=2 colspan=1>19.9880.24%19.9940.22%20.010 0.13%</td><td rowspan=2 colspan=1>40.2270.11%40.2330.10%40.2500.05%</td><td rowspan=2 colspan=1>57.3980.09%57.3980.09%57.4220.04%</td></tr><tr><td rowspan=2 colspan=1>EoHMCTS-AHDOurs</td></tr><tr><td rowspan=1 colspan=1>20.023 0.07%</td><td rowspan=1 colspan=1>40.262 0.02%</td><td rowspan=1 colspan=1>57.436 0.02%</td></tr></table>

## 4.3 COMPARISON ON FSSP

GLS acts as a perturbation-and-improvement framework, which iteratively perturbs a current schedule using neighborhood operators (e.g., swap and relocate) and employs a heuristic matrix to guide which parts to penalize or prioritize for better exploration (Arnold & Sörensen, 2019). Table 11 reports relative improvements in the makespan in Taillard instances. PRS exhibits a consistent advantage over other methods in nearly all test sets and on average.

## 5 DISCUSSIONS

Ablations on Mechanisms. We analyze the necessity of each component in PRS. As shown in Table 4, all three perturbation operators contribute to search effectiveness: expansion and compression support complementary transformation directions by encouraging richer and simpler heuristic designs, respectively, while steering provides fine-grained exploitation around promising heuristic structures. Disabling UCB-based operator selection and replacing it with random selection consistently degrades performance, indicating that adaptively allocating search effort among the three operators is beneficial. Disabling the stagnation-aware restart mechanism (C = 10<sup>5</sup>) also weakens performance, further showing the importance of adaptive search control for sustained exploration. Beyond component removal, replacing expansion and compression with EC-style operators (M1&M2) degrades performance within PRS, suggesting that explicitly targeting complementary structural regimes is more effective than the default EC-style transformations in this setting.

Evolution Curves of Different AHD methods. In Fig. 3, we compare the search dynamics of PRS with representative AHD methods on two representative tasks: OP with ACO and OBPP with the constructive framework. Most baselines exhibit rapid early improvements followed by early convergence, indicating a limited ability to sustain effective exploration as the search progresses. In contrast, PRS shows smoother and more continuous performance gains throughout the search. This behavior can be attributed to its multi-directional perturbation mechanism, which actively diversifies heuristic generation and mitigates early homogenization, together with the stagnation-aware controller that restores exploration when progress slows. Empirically, the evolution curves suggest that sustained search effectiveness can be achieved through explicitly controlling diversification through directional search

Table 4: Ablation study of PRS on key mechanisms and operators. Results are reported as mean ± standard deviation over five independent runs.
<table><tr><td>Methods</td><td>CVRP-50↓ KP-100↑</td></tr><tr><td>PRS (default setting)</td><td> $\mathbf { 8 . 9 5 8 3 - 0 . 0 7 2 6 }$   $\mathbf { 4 0 . 2 6 0 3 { \scriptstyle \pm 0 . 0 1 6 5 } }$ </td></tr><tr><td>w/o Expansion w/o Compression w/o Steering</td><td> $9 . 7 4 9 2 { \scriptstyle \pm 0 . 2 8 4 0 }$   $4 0 . 2 3 0 7 { \scriptstyle \pm 0 . 0 0 0 1 }$   $9 . 3 7 3 7 { \scriptstyle \pm 0 . 0 7 0 7 }$   $4 0 . 2 3 2 1 { \scriptstyle \pm 0 . 0 0 1 5 }$ </td></tr><tr><td>w/o UCB Selection</td><td> $9 . 0 7 2 8 { \scriptstyle \pm 0 . 1 2 0 7 }$   $4 0 . 2 5 8 1 { \scriptstyle \pm 0 . 0 7 5 4 }$   $9 . 3 7 8 1 { \scriptstyle \pm 0 . 1 1 4 2 }$   $4 0 . 2 4 8 3 { \scriptstyle \pm 0 . 0 3 3 6 }$ </td></tr><tr><td> $C = 1 0 ^ { 5 }$ </td><td></td></tr><tr><td></td><td> $9 . 1 3 9 9 { \scriptstyle \pm 0 . 1 1 0 6 }$   $4 0 . 2 4 3 2 { \scriptstyle \pm 0 . 0 1 3 7 }$   $\overline { { 4 0 . 2 5 1 6 { \scriptstyle \pm 0 . 0 0 7 1 } } }$ </td></tr><tr><td colspan="2">EC Operators (M1&amp;M2)  $9 . 2 2 7 7 { \scriptstyle \pm 0 . 0 5 7 0 }$ </td></tr></table>

![](images/b262d025e939b2552d57b53f498a3905c666a139a4a54a5cb7658faab89d0e84.jpg)  
(a) OP.

![](images/e958e71d0b90edc7070ffe84faadb9078d173707f63208db421a1407de17b75e.jpg)  
(b) OBPP.  
Figure 3: Evolution curves of the best heuristic in training on two scenarios. Each curve is averaged over five independent runs and the standard deviation is shaded.

primitives without relying on elaborate EC-based operator orchestration.

Additional Experimental Results. We provide additional experiments in Appendix F–H. In particular, we first provide diversity analyses at both the idea and implementation levels to examine whether the proposed directional search primitives indeed broaden heuristic-space exploration. We then include statistical significance tests (p-values), LLM cost analysis, and sensitivity analysis on the key hyperparameters. Importantly, we further examine how different perturbation directions relate to ID and OOD performance, and an analysis on the contributions of each directional search operator on different problem settings. Moreover, we provide further experimental results on the TSP and TSPLib (Reinelt, 1991) setting, black-box settings, and benchmarking across different LLMs.

## 6 CONCLUSION

In this paper, we propose PRS, a multi-directional search framework for automatic heuristic design that makes directional heuristic transformation an explicit generation-level search primitive. By organizing heuristic generation through complementary transformation directions and combining them with breadth-first search and adaptive operator control, PRS actively directs diversification, sustains exploration, and mitigates premature convergence. Extensive experiments across diverse combinatorial optimization tasks demonstrate that PRS consistently discovers high-quality heuristics and outperforms existing LLM-based AHD methods in various settings, emphasizing the importance of exploring directionally diverse regions of the heuristic search space.

## REFERENCES

H. Abgaryan, A. Harutyunyan, and T. Cazenave. LLMs can schedule, 2024. arXiv preprint arXiv:2408.06993.

F. Arnold and K. Sörensen. Knowledge-guided local search for the vehicle routing problem. Computers & Operations Research, 105:32–46, 2019.

E.O. Asani, A.E. Okeyinka, and A.A. Adebiyi. A computation investigation of the impact of convex hull subtour on the nearest neighbour heuristic. In 2023 International Conference on Science, Engineering and Business for Sustainable Development Goals (SEB-SDG), volume 1, pp. 1–7, 2023.

L.C.T. Bezerra, M. López-Ibáñez, and T. Stützle. Automatic component-wise design of multiobjective evolutionary algorithms. IEEE Transactions on Evolutionary Computation, 20(3):403–417, 2015.

C. Blum. Ant colony optimization: Introduction and recent trends. Physics of Life Reviews, 2(4): 353–373, 2005.

J. Brecklinghaus and S. Hougardy. The approximation ratio of the greedy algorithm for the metric traveling salesman problem. Operations Research Letters, 43(3):259–261, 2015.

E.K. Burke, M.R. Hyde, G. Kendall, G. Ochoa, E. Özcan, and J.R. Woodward. A classification of hyper-heuristic approaches: Revisited. In Handbook of Metaheuristics, pp. 453–477. Springer, 2018.

H.G. Campbell, R.A. Dudek, and M.L. Smith. A heuristic algorithm for the n job, m machine sequencing problem. Management Science, 16(10):B–630, 1970.

I. Castiñeiras, M. De Cauwer, and B. O’Sullivan. Weibull-based benchmarks for bin packing. In International Conference on Principles and Practice ofConstraint Programming, pp. 207–222, 2012.

C. Chen, M. Zhong, Y. Fan, J. Shi, and J. Sun. HiFo-Prompt: prompting with hindsight and foresight for LLM-based automatic heuristic design. In International Conference on Learning Representations, 2026.

N. Christofides. Worst-case analysis of a new heuristic for the travelling salesman problem. In Operations Research Forum, volume 3, pp. 20, 2022.

M. Crepinšek, S.-H. Liu, and M. Mernik. Exploration and exploitation in evolutionary algorithms: A<sup>ˇ</sup> survey. ACM Computing Surveys, 45(3):1–33, 2013.

N. Dahmani, I. Sbai, T. Tlili, and S. Krichen. On solving the 2L-CVRP using an adaptive chemical reaction algorithm: Postal transportation real-case. International Journal of System Assurance Engineering and Management, pp. 1–25, 2024.

P.V.T. Dat, L. Doan, and H.T.T. Binh. HSEvo: Elevating automatic heuristic design with diversitydriven harmony search and genetic algorithm using LLMs. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pp. 26931–26938, 2025.

M. Dorigo. Optimization, learning and natural algorithms. PhD thesis, 1992. PhD thesis, Politecnico di Milano.

M. Dorigo, M. Birattari, and T. Stutzle. Ant colony optimization. IEEE Computational Intelligence Magazine, 1(4):28–39, 2006.

G. Duflo, E. Kieffer, M.R. Brust, G. Danoy, and P. Bouvry. A GP hyper-heuristic approach for generating TSP heuristics. In 2019 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), pp. 521–529, 2019.

A.E. Eiben and J. Smith. From evolutionary computation to the evolution of things. Nature, 521 (7553):476–482, 2015.

M. Elhenawy, A. Abutahoun, T.I. Alhadidi, A. Jaber, H.I. Ashqar, S. Jaradat, A. Abdelhay, S. Glaser, and A. Rakotonirainy. Visual reasoning and multi-agent approach in multimodal large language models (MLLMs): Solving TSP and MTSP combinatorial challenges, 2024. arXiv preprint arXiv:2407.00092.

H. Emmons and G. Vairaktarakis. Flow shop scheduling: Theoretical results, algorithms, and applications. Springer Science & Business Media, 2012.

L.J. Eshelman. Preventing premature convergence in genetic algorithms by preventing incest. In Proceedings ofthe Fourth International Conference on Genetic Algorithms, 1991.

V. Fernandez-Viagas and J.M. Framinan. On insertion tie-breaking rules in heuristics for the permutation flowshop scheduling problem. Computers & Operations Research, 45:60–67, 2014.

Á. Fialho. Adaptive operator selection for optimization. PhD thesis, 2010. PhD thesis, Université Paris Sud-Paris XI.

C. Gao, H. Shang, K. Xue, D. Li, and C. Qian. Towards generalizable neural solvers for vehicle routing problems via ensemble with transferrable local policy. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence, 2024. Article 764.

P.-F. Guo, Y.-H. Chen, Y.-D. Tsai, and S.-D. Lin. Towards optimizing with large language models, 2023. arXiv preprint arXiv:2310.05204.

J.N.D. Gupta. A functional heuristic algorithm for the flowshop scheduling problem. Journal ofthe Operational Research Society, 22(1):39–47, 1971.

Q. He, K.L. Head, and J. Ding. Heuristic algorithm for priority traffic signal control. Transportation Research Record, 2259(1):1–7, 2011.

Q. Hu and Q. Zhang. Partition to Evolve: Niching-Enhanced Evolution with LLMs for Automated Algorithm Discovery. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems, 2025.

B. Hudson, Q. Li, M. Malencia, and A. Prorok. Graph neural network guided local search for the traveling salesperson problem. In International Conference on Learning Representations, 2022.

X. Jiang, Y. Wu, M. Li, Z. Cao, and Y. Zhang. Large language models as end-to-end combinatorial optimization solvers, 2025. arXiv preprint arXiv:2509.16865.

M. E. Johnson, L. M. Moore, and D. Ylvisaker. Minimax and maximin distance designs. Journal of Statistical Planning and Inference, 26(2):131–148, 1990. doi: 10.1016/0378-3758(90)90122-B.

L. Jost. Entropy and diversity. Oikos, 113(2):363–375, 2006.

C. Kenyon et al. Best-Fit bin-packing with random order. In Proceedings of SODA, volume 96, pp. 359–364, 1996.

M. Kim, S. Choi, H. Kim, J. Son, J. Park, and Y. Bengio. Ant colony sampling with GFlowNets for combinatorial optimization, 2024. arXiv preprint arXiv:2403.07041.

W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.

Y.-D. Kwon, J. Choo, B. Kim, I. Yoon, Y. Gwon, and S. Min. POMO: Policy optimization with multiple optima for reinforcement learning. Advances in Neural Information Processing Systems, 33:21188–21198, 2020.

W.B. Langdon and R. Poli. Foundations of genetic programming. Springer Science & Business Media, 2013.

Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. Shinkaevolve: Towards open-ended and sample-efficient program evolution. arXiv preprint arXiv:2509.19349, 2025.

Craig A. Layman, D. Albrey Arrington, Carmen G. Montaña, and David M. Post. Can stable isotope ratios provide for community-wide measures of trophic structure? Ecology, 88(1):42–48, 2007. doi: 10.1890/0012-9658(2007)88[42:CSIRPF]2.0.CO;2.

S. Lin and B.W. Kernighan. An effective heuristic algorithm for the traveling-salesman problem. Operations Research, 21(2):498–516, 1973.

F. Liu, X. Tong, M. Yuan, and Q. Zhang. Algorithm evolution using large language model, 2023. arXiv preprint arXiv:2311.15249.

F. Liu, X. Tong, M. Yuan, X. Lin, F. Luo, Z. Wang, Z. Lu, and Q. Zhang. Evolution of heuristics: towards efficient automatic algorithm design using large language model. In International Conference on Machine Learning, 2024a.

F. Liu, Y. Yao, P. Guo, Z. Yang, X. Lin, Z. Zhao, X. Tong, K. Mao, Z. Lu, Z. Wang, et al. A systematic survey on large language models for algorithm design. ACM Computing Surveys, 2024b.

F. Liu, R. Zhang, Z. Xie, R. Sun, K. Li, X. Lin, Z. Wang, Z. Lu, and Q. Zhang. LLM4AD: A platform for algorithm design with large language model, 2024c. arXiv preprint arXiv:2412.17287.

F. Liu, Y. Liu, Q. Zhang, X. Tong, and M. Yuan. EoH-S: Evolution of heuristic set using LLMs for automated heuristic design, 2025. arXiv preprint arXiv:2508.03082.

S. Liu, C. Chen, X. Qu, K. Tang, and Y.-S. Ong. Large language models as evolutionary optimizers. In 2024 IEEE Congress on Evolutionary Computation (CEC), pp. 1–8, 2024d.

Shu Liu, Shubham Agarwal, Monishwaran Maheswaran, Mert Cemri, Qiuyang Mang, Zhifei Li, Ashwin Naren, Ethan Boneh, Audrey Cheng, Alexander Du, Melissa Pan, Kurt Keutzer, Alvin Cheung, Koushik Sen, Alex Dimakis, Matei Zaharia, and Ion Stoica. Evox: Meta-evolution for automated discovery. In Third Conference on Language Modeling, 2026.

Y. Ma, J. Li, Z. Cao, W. Song, L. Zhang, Z. Chen, and J. Tang. Learning to iteratively solve routing problems with dual-aspect collaborative transformer. Advances in Neural Information Processing Systems, 34:11096–11107, 2021.

Max D. Morris and Toby J. Mitchell. Exploratory designs for computational experiments. Journal of Statistical Planning and Inference, 43(3):381–402, 1995. doi: 10.1016/0378-3758(94)00035-T.

M. Nawaz, E.E. Enscore, and I. Ham. A heuristic algorithm for the m-machine, n-job flow-shop sequencing problem. Omega, 11(1):91–95, 1983.

A. Novikov, N. Vu, M. Eisenberger, E. Dupont, P.-S. Huang, A.Z. Wagner, S. Shirobokov, B. Ko-˜ zlovskii, F.J.R. Ruiz, A. Mehrabian, et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery, 2025. arXiv preprint arXiv:2506.13131.

Z. Pan, L. Wang, J. Wang, and J. Lu. Deep reinforcement learning based optimization algorithm for permutation flow-shop scheduling. IEEE Transactions on Emerging Topics in Computational Intelligence, 7(4):983–994, 2021.

N. Pillay and R. Qu. Hyper-heuristics: Theory and applications. Springer, 2018.

E.J.S. Pires, J.A.T. Machado, and P.B. de Moura Oliveira. Diversity study of multi-objective genetic algorithm based on Shannon entropy. In 2014 Sixth World Congress on Nature and Biologically Inspired Computing (NaBIC 2014), pp. 17–22, 2014.

E.J.S. Pires, J.A.T. Machado, and P.B. de Moura Oliveira. Dynamic Shannon performance in a multiobjective particle swarm optimization. Entropy, 21(9):827, 2019.

R. Qu, G. Kendall, and N. Pillay. The general combinatorial optimization problem: Towards automated algorithm design. IEEE Computational Intelligence Magazine, 15(2):14–23, 2020.

G. Reinelt. TSPLIB—A traveling salesman problem library. ORSA Journal on Computing, 3(4): 376–384, 1991.

B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M.P. Kumar, E. Dupont, F.J.R. Ruiz, J.S. Ellenberg, P. Wang, O. Fawzi, et al. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024.

D.J. Rosenkrantz, R.E. Stearns, and P.M. Lewis. An analysis of several heuristics for the traveling salesman problem. SIAM Journal on Computing, 6(3):563–581, 1977.

L. Sengupta, R. Mariescu-Istodor, and P. Fränti. Which local search operator works best for the open-loop TSP? Applied Sciences, 9(19):3985, 2019.

Yiding Shi, Jianan Zhou, Wen Song, Jieyi Bi, Yaoxin Wu, Zhiguang Cao, and Jie Zhang. Generalizable heuristic generation through llms with meta-optimization. In International Conference on Learning Representations, volume 2026, pp. 119314–119349, 2026.

T. Stützle. Applying iterated local search to the permutation flow shop problem. Technical report, 1998. Technical Report AIDA-98-04, FG Intellektik, TU Darmstadt.

E. Taillard. Benchmarks for basic scheduling problems. European Journal of Operational Research, 64(2):278–285, 1993.

J. Tuononen. Analysis of rebuild local search operator for TSP, 2022.

L. Wang and Y. Chen. Diversity based on entropy: A novel evaluation criterion in multi-objective optimization algorithm. International Journal of Intelligent Systems and Applications, 4(10): 113–124, 2012.

X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou. Selfconsistency improves chain of thought reasoning in language models, 2022. arXiv preprint arXiv:2203.11171.

C. Yang, X. Wang, Y. Lu, H. Liu, Q.V. Le, D. Zhou, and X. Chen. Large language models as optimizers. In The Twelfth International Conference on Learning Representations, 2023.

S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36:11809–11822, 2023.

H. Ye, J. Wang, Z. Cao, H. Liang, and Y. Li. DeepACO: Neural-enhanced ant systems for combinatorial optimization. Advances in Neural Information Processing Systems, 36:43706–43728, 2023.

H. Ye, J. Wang, Z. Cao, F. Berto, C. Hua, H. Kim, J. Park, and G. Song. Reevo: Large language models as hyper-heuristics with reflective evolution. Advances in Neural Information Processing Systems, 37:43571–43608, 2024.

R. Zhang and Z. Lu. Rethinking Code Similarity for Automated Algorithm Design with LLMs. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/pdf?id=HIUqeO9OOr.

Z. Zheng, Z. Xie, Z. Wang, and B. Hooi. Monte Carlo tree search for comprehensive exploration in LLM-based automatic heuristic design. In International Conference on Machine Learning, 2025.

## APPENDIX CONTENTS

A Related Work 15   
A.1 Automated Heuristic Design 15   
A.2 LLM-based AHD 15   
A.3 LLM for CO 15   
A.4 Adaptive Control in EC and LLM-based AHD. 16   
B Evaluated Problem Settings 16   
B.1 Definition of Tasks 16   
B.2 Evaluated Framework for Tasks 18   
C Complete Algorithm 20   
D Detailed Methodology 21   
D.1 PRS Prompt . 21   
D.2 Generated Heuristics 27   
E Details of Evaluation Configurations 29   
F More Experimental Results 31   
F.1 Search Dynamic Analysis . 31   
F.2 Comparison on Black-box Settings 35   
F.3 TSP Step-by-step Constructive Framework . 35   
F.4 TSP Heuristic under Guided Local Search Framework . 36   
F.5 Flow Shop Scheduling Problem under Guided Local Search Framework 38   
F.6 Results on TSPLib instances 38   
F.7 Statistical Significance 39   
F.8 Search Strategy Variants 40   
F.9 Experiments with different LLMs 41   
G Cost Analysis 42   
H Sensitivity Analysis 43   
I Example Response for Multi-directional Perturbations 44   
J Limitations 46   
K Baselines & Licenses 46

## A RELATED WORK

## A.1 AUTOMATED HEURISTIC DESIGN

Automated Heuristic Design (AHD), often referred to as hyper-heuristics (Burke et al., 2018), aims to automatically discover high-performing heuristics from a large space of valid heuristic candidates rather than directly solving problem instances. Most AHD methods adopt evolutionary algorithms as the underlying search strategy to iteratively refine heuristics automatically through selection and generation, as well as a series of defined evolutionary operators for combination and adaptation (Pillay & Qu, 2018). Among them, genetic programming (GP) (Langdon & Poli, 2013) has emerged as a prevailing approach for heuristic generation due to its flexibility and interpretability, enabling the synthesis of reusable or instance-specific heuristics via tree-based representations. However, GP-based AHD typically relies on carefully hand-crafted mutation and crossover operators, whose effectiveness is highly sensitive to representation choices and domain knowledge (Duflo et al., 2019). Beyond GP, recent AHD frameworks further categorize heuristic generation into constructive and perturbative paradigms, or integrate multiple operators and stages through component-based unified solvers, with the goal of enhancing generality across diverse optimization problems (Bezerra et al., 2015; Qu et al., 2020). Despite these advances, existing AHD approaches largely depend on manually designed components and expert knowledge (Liu et al., 2025), which remain a major bottleneck for scalability and portability.

## A.2 LLM-BASED AHD

Recent work has introduced large language models into automated heuristic design, leading to a class of population-based LLM-driven AHD methods that largely follow an evolutionary paradigm (Liu et al., 2023; Romera-Paredes et al., 2024; Liu et al., 2024a; Ye et al., 2024; Liu et al., 2024c). In general, these methods iterate over three core steps after initially sampling a set of heuristic functions: selecting mutation or crossover operators together with parent heuristics from the population, prompting LLMs to generate new heuristic implementations—typically in executable code form—and updating the population (or the tree) based on empirical performance evaluated on a validation dataset until a fixed maximum evaluation step is reached. Representative frameworks within this paradigm include FunSearch (Romera-Paredes et al., 2024), which adopts a multi-island evolutionary structure with a fixed prompt strategy for algorithm refinement, EoH (Liu et al., 2024a), which expands the search space through multiple prompt strategies that evolve both intermediate reasoning and code, and ReEvo (Ye et al., 2024), which incorporates short- and long-term reflection mechanisms to guide heuristic evolution. More recently, diversity has received increasing attention as an important factor for sustaining exploration in LLM-based AHD. For example, HSEvo (Dat et al., 2025) explicitly promotes population diversity through diversity-aware search and population management, showing the value of preserving heterogeneous heuristic candidates during evolution.

Building on this direction, recent studies have introduced more explicit mechanisms for measuring diversity and structuring the search space. BehaveSim (Zhang & Lu, 2026) measures algorithmic similarity from execution trajectories and incorporates behavioral diversity into candidate selection. PartEvo (Hu & Zhang, 2025) projects candidate algorithms into feature spaces and constructs semantic niches to allocate search effort across different regions of the abstract language space. ShinkaEvolve (Lange et al., 2025) further promotes exploration through archive-based parent sampling and code-novelty rejection based on embedding similarity. These approaches demonstrate the importance of maintaining diverse candidates during LLM-based algorithm search, but primarily regulate diversity through similarity assessment, population organization, novelty filtering, or search-space partitioning. In contrast, PRS controls diversification directly at the generation level by explicitly selecting how a parent heuristic is transformed before realization, using expansion, compression, and steering as complementary directional perturbation operators.

## A.3 LLM FOR CO

Beyond LLM-based AHD, the use of LLMs for tackling combinatorial optimization (CO) problems has also emerged as a promising direction, and one line of research treats LLMs as optimizers. Starting with OPRO (Yang et al., 2023) that iteratively refines candidate solutions for TSP by conditioning on previously generated solutions and their objective values, the existing LLM-asoptimizer methods (Guo et al., 2023; Liu et al., 2024d; Elhenawy et al., 2024) improve solution quality through repeated in-context prompting, evaluation, and reinjection, forming a closed-loop optimization process. Another closely related direction employs LLMs as end-to-end solvers (LLMas-E2E-solver) that directly map problem descriptions to complete solutions, either via prompting or with additional fine-tuning within task-specific environments (Abgaryan et al., 2024; Jiang et al., 2025). These approaches demonstrate the potential of LLMs to reason over complicated CO structures and to generate feasible solutions without explicit algorithmic decomposition and solver utilization.

## A.4 ADAPTIVE CONTROL IN EC AND LLM-BASED AHD.

Adapting search strategies based on the evolving state has long been a crucial objective in evolutionary computation. Traditional approaches mainly rely on low-level mechanisms such as Adaptive Operator Selection (AOS) (Fialho, 2010) and parameter control (Eiben & Smith, 2015), which adjust mutation rates or operator probabilities in a reactive manner. Such approaches operate through numerical credit assignment mechanisms that bias the search toward recently successful strategies, but they lack the capacity to reason about search dynamics, for example, detecting stagnation or loss of diversity. More recent LLM-based AHD methods have also introduced higher-level mechanisms for adapting search behavior, such as diversity-aware population management and stagnation handling (Dat et al., 2025). These developments motivate adaptive search control beyond fixed operator schedules. Driven by these motivations, PRS combines adaptive operator control with stagnation-aware restart, where operator selection is continuously adjusted according to recent search utility, while prolonged stagnation triggers an elite restart and temporary reheating of exploration.

## B EVALUATED PROBLEM SETTINGS

## B.1 DEFINITION OF TASKS

We evaluate the proposed PRS framework on a diverse set of combinatorial optimization problems, covering routing, packing, scheduling, and combinatorial set construction tasks. Specifically, they are TSP, CVRP, OP, MKP, BPP, KP, ASP and FSSP. For BPP, we only test its online setting (online BPP). In this subsection, we will introduce the mathematical formulation and evaluation protocols of these problems in detail.

Traveling Salesman Problem (TSP). Given a set of n cities with pairwise distances, the Traveling Salesman Problem aims to find the minimum-length tour that visits each city exactly once and returns to the starting city.

$$
\operatorname* { m i n } _ { \pi \in S _ { n } } \sum _ { t = 1 } ^ { n } d _ { \pi _ { t } , \pi _ { t + 1 } } , \quad \pi _ { n + 1 } = \pi _ { 1 } ,\tag{7}
$$

where $d _ { i j }$ is the Euclidean distance between city i and city $j , \pi _ { t }$ is the selected node at step t of a certain valid tour (i.e., a Hamiltonian tour over n cities) and $S _ { n }$ is the solution space of the TSP with n cities. We consider the symmetric TSP formulation and evaluate heuristics on both synthetic instances and real-world benchmarks from TSPLib. Following ReEvo, synthetic TSP instances sample nodes uniformly from the unit square $[ 0 , 1 ] ^ { 2 }$ (Ye et al., 2024).

Capacitated Vehicle Routing Problem (CVRP). CVRP consists of a depot indexed 0 and a set of customers $\{ 1 , \ldots , n \}$ , each customer i has a demand $q _ { i }$ (with $q _ { 0 } = 0 )$ , and all vehicles have an identical capacity Q. The objective is to design a set of closed routes that serve all customers without exceeding vehicle capacities, while minimizing the total travel cost:

$$
\operatorname* { m i n } \sum _ { k } \sum _ { ( i , j ) \in \pi _ { k } } d _ { i j }\tag{8}
$$

subject to

$$
\sum _ { i \in \pi _ { k } } q _ { i } \leq Q , \quad \forall k .\tag{9}
$$

where each sub-route $\pi _ { k }$ starts and ends at the depot.

Orienteering Problem (OP). Given a graph with nodes $\{ 0 , 1 , \ldots , n \}$ , where node 0 denotes the depot and is associated with zero reward $( r _ { 0 } = 0 )$ , each remaining node i has a reward $r _ { i } ,$ and $d _ { i j }$ denotes the travel distance. The goal of OP is to find a route $\pi = ( \pi _ { 1 } , \ldots , \pi _ { k } )$ starting and ending at the depot that maximizes the total reward collected under a travel-length budget $B \colon$

$$
\operatorname* { m a x } _ { \pi } \sum _ { i \in \pi } r _ { i } \quad \mathrm { s . t . } \quad \sum _ { t = 0 } ^ { k } d _ { \pi _ { t } , \pi _ { t + 1 } } \leq B\tag{10}
$$

where $\pi _ { 0 } = \pi _ { k + 1 } = 0$ and each node is visited at most once.

Knapsack Problems (KP and MKP). The Knapsack Problem (KP) aims to select a subset of items to maximize total profit subject to a capacity constraint. Given item values $\{ v _ { i } \}$ , item weights $\{ w _ { i } \}$ and knapsack capacity C, KP aims to

$$
\operatorname* { m a x } _ { x \in \{ 0 , 1 \} ^ { n } } \ \sum _ { i = 1 } ^ { n } v _ { i } x _ { i } \quad { \mathrm { s . t . } } \quad \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } \leq C .\tag{11}
$$

where $x _ { i }$ is a binary indicator of whether the item i is packed into the knapsack or not. We follow AM (Kool et al., 2019) in generating instances and take $\bar { W } = 2 5$ for 100-item and 200-item KP instances and $W = 1 2 . 5$ for 50-item ones.

The Multiple Knapsack Problem (MKP) extends this formulation by introducing multiple resource constraints. We evaluate heuristics under a constructive selection setting, where items are considered sequentially. For the Multiple knapsack problem (MKP), we follow the instance generation procedure in ReEvo (Ye et al., 2024). Specifically, item values and weights are independently sampled from a uniform distribution on [0, 1], and the capacity of each knapsack is set proportionally to the total item weights to ensure non-trivial feasible solutions. Specifically for the knapsack problem, select\_next\_item function takes as input the remaining capacity, a numpy array of item values, and a numpy array of item weights and returns the next item to pick. For both KP and MKP, feasibility is enforced at each step, and the total profit of the selected items serves as the objective value.

Online Bin Packing Problem (Online BPP). In the Online Bin Packing Problem, items arrive sequentially and must be assigned to bins immediately upon arrival without knowledge of future items. Given the bin capacity $\bar { W }$ and an item stream $\{ s _ { t } \}$ , the objective is to minimize the number of bins used. Formally, upon receiving $s _ { t } ,$ the policy must choose a bin index $b _ { t }$ with the highest priority score. We follow FunSearch (Romera-Paredes et al., 2024) and EoH (Liu et al., 2024a) to generate Weibull instances for online BPP.

Flow Shop Scheduling Problems (FSSP). Scheduling problems aim to determine the execution order of operations to optimize the overall schedule length. In the Flow Shop Scheduling Problem (FSSP), n jobs must be processed on m machines, where each job consists of m operations executed in the same predetermined order across all machines. Each machine can process at most one operation at a time, and the objective is to minimize the makespan, i.e., the total schedule length.

Following prior work (Emmons & Vairaktarakis, 2012), we consider a perturbation-based scheduling setting, where the processing order remains consistent during each step, and the machines cannot execute multiple operations simultaneously. During heuristic search, we perform optimization on a set of randomly generated instances $D .$ Each instance in D consists of 50 jobs and between 2 and 20 machines, with processing times sampled independently from a uniform distribution over [0, 1] (Pan et al., 2021). The average makespan over these instances is used as the fitness value for heuristic evaluation.

Admissible Set Problem (ASP). The admissible set problem (ASP) aims to construct the largest possible set of vectors $A \subset \{ 0 , 1 , 2 \} ^ { n }$ , where each vector has exactly w non-zero elements. Formally, ASP can be written as

$$
\operatorname* { m a x } | A |\tag{12}
$$

subject to

$$
\sum _ { i = 1 } ^ { n } \mathbb { I } [ a _ { i } \neq 0 ] = w , \quad \forall a \in A ,\tag{13}
$$

and

$$
\exists i \in \{ 1 , \dots , n \} \mathrm { ~ s . t . ~ } \{ a _ { i } , b _ { i } , c _ { i } \} \in \left\{ \{ 0 , 0 , 1 \} , \{ 0 , 0 , 2 \} , \{ 0 , 1 , 2 \} \right\} , \quad \forall a , b , c \in A , a \neq b \neq c .\tag{14}
$$

Under the step-by-step constructive framework, the heuristic to be designed is a priority function that assigns a score to each candidate vector according to its suitability for being added to the current admissible set. Higher scores indicate higher priority.

## B.2 EVALUATED FRAMEWORK FOR TASKS

We evaluate PRS on eight NP-hard combinatorial optimization problems (COPs) under three widely used algorithmic frameworks for automatic heuristic design (AHD). Specifically, we consider (i) step-by-step construction for TSP, KP, ASP and Online-BPP; (ii) ACO for MKP, CVRP and OP; and (iii) GLS for TSP and FSSP.

Evaluation metric. For minimization problems (TSP, CVRP, Online-BPP, FSSP), we report the relative optimality gap:

$$
\mathrm { G a p } ( \% ) = \frac { f ( h ) - f ^ { \star } } { f ^ { \star } } \times 1 0 0 \% ,\tag{15}
$$

where $f ( h )$ is the objective value returned by a heuristic function h and $f ^ { \star }$ is the optimal reference value (e.g., best-known or a lower bound). For maximization problems (OP, KP, MKP, ASP), we use

$$
\mathrm { G a p } ( \% ) = \frac { f ^ { \star } - f ( h ) } { f ^ { \star } } \times 1 0 0 \% .\tag{16}
$$

where $f ^ { \star }$ is the best-known value or an upper bound.

## B.2.1 STEP-BY-STEP CONSTRUCTIVE FRAMEWORK

Step-by-step construction is a generic framework for combinatorial optimization that incrementally builds a solution by repeatedly selecting and adding the highest-priority candidate until a complete feasible solution is obtained. Nearest-neighbor is a widely used handcrafted step-by-step constructive heuristic for TSP, which selects the next node solely based on its distance to the current node. Analogously, for the knapsack problem, a greedy heuristic is adopted that prioritizes items according to their value-to-weight ratio. We refer to these two manually designed heuristics as the Greedy Construct (GC) baseline in Table 3 and Table 9.

• TSP. Under the step-by-step construction framework, the designed heuristic is a select\_next\_node function that scores all unvisited candidates and greedily appends the best one. For each instance, the heuristic outputs the best next city to select based on the current node, the destination node (starting node), the indices of all unvisited nodes and the symmetric distance matrix.

• KP. In step-by-step construction for KP, the heuristic scores the remaining items given $( v _ { i } , w _ { i } )$ and the remaining capacity, then selects the next item to pack.

• Online BPP. We evaluate heuristics under this strict online setting, where the heuristic determines the bin assignment for each incoming item based solely on current bin states and the current item’s size.

• ASP. In step-by-step construction for ASP, the heuristic scores each candidate vector given the vector itself, the problem dimension n, and the weight parameter w, and then selects high-priority vectors to incrementally build the admissible set while preserving feasibility.

## B.2.2 GLS-BASED PROBLEMS

Guided Local Search (GLS). For GLS-based evaluation, we embed the LLM-designed heuristic into a standard GLS loop (e.g., maintaining a penalty-based edge weights matrix). We use two local operators (i.e., Swap and Relocate) for perturbation and local improvements, and evaluate after each perturbation-local search cycle.

TSP under GLS. We apply GLS to TSP by optimizing an edge-weight matrix that penalizes frequently used edges. Let $c _ { i j }$ be the edge cost between node i and node j and let $p _ { i j } \in \mathbb { Z } _ { \geq 0 }$ be a penalty. The augmented edge cost takes the typical form:

$$
c _ { i j } ^ { \prime } = c _ { i j } + \lambda p _ { i j } ,\tag{17}
$$

where λ controls penalty strength. The AHD target is improving the penalty (utility) heuristic (e.g., how to update $p _ { i j } )$ . We apply LLM-based AHD methods to design the core heuristic for generating knowledge matrices within the knowledge-guided local search (KGLS) framework (Arnold & Sörensen, 2019). For TSP, KGLS maintains a current solution together with the best solution encountered so far. At each iteration, the current solution is first perturbed according to the learned knowledge matrix, followed by the local search using 2-opt and relocate operators (Sengupta et al., 2019; Tuononen, 2022). We use scale-wise GLS budgets summarized in Table 5.

Flow Shop Scheduling Problem (FSSP) under GLS. We evaluate FSSP within a GLS-style perturbation and local improvement procedure, where the evolved heuristic controls (i) which operations to penalize (or prioritize) and (ii) which parts of the schedule to perturb.

## B.2.3 ACO-BASED PROBLEMS

For ACO-based problems, including CVRP and OP, we adopt a standard ant colony optimization (ACO) framework that mimics the nature of ants to find routes between food sources and the colony. In this setting, the LLM-designed component is the heuristic information function, which defines the edge desirability matrix η used by ants during solution construction.

At each construction step, an ant located at node i selects the next node j according to the transition probability

$$
P _ { i j } = \frac { \tau _ { i j } ^ { \alpha } \eta _ { i j } ^ { \beta } } { \sum _ { k \in \mathcal { A } ( i ) } \tau _ { i k } ^ { \alpha } \eta _ { i k } ^ { \beta } } ,\tag{18}
$$

where $\tau _ { i j }$ denotes the pheromone value, $\eta _ { i j }$ is the heuristic information produced by the evolved heuristics, α and β are fixed hyperparameters, and $\boldsymbol { \mathscr { A } } ( i )$ denotes the set of feasible next nodes under the problem-specific constraints (Dorigo, 1992; Dorigo et al., 2006).

The heuristic matrix functions under ACO framework have different semantics for CVRP, OP, and MKP:

• CVRP. Under the ACO framework, ants sequentially construct routes by selecting the next customer based on pheromone trails and LLM-generated heuristic scores.

• OP. Within the ACO framework, ants incrementally build candidate paths by choosing the next site according to the pheromone information and LLM-based heuristics that balance reward and travel budget.

• MKP. For MKP, the ACO framework is applied in a black-box manner, which ants stochastically sample item selections guided by pheromone information and LLM-generated heuristic scores that indicate the desirability of including each item in the solution.

The pheromone update rule and other ACO hyperparameters are kept identical across all methods to ensure a fair comparison. Only the heuristic information function η is replaced by the LLM-designed heuristics generated by PRS or the baseline AHD.

## C COMPLETE ALGORITHM

Algorithm 1 provides the complete pseudo-code for the PRS framework. The full framework includes an initialization stage (up to line 6) and an iterative stage consisting of a BFS-based search process with multi-directional perturbation operators and stagnation-aware mechanisms.

Algorithm 1 PRS: Perturbation–Rewrite Search via BFS and Adaptive Contro   
1: Input: Validation dataset $D ;$ evaluation environment $^ { g ; }$ heuristic pool size $N ;$ initial number   
of sampled heuristics $N _ { \mathrm { i n i t } } ;$ maximum evaluation steps $T ;$ root selection via ϵ-greedy; BFS   
parameters: depth $d ,$ beam size $\kappa ,$ frontier size $f s ,$ softmax temperature $T _ { \mathrm { s e l } } ;$ UCB operator   
selection with exploration coefficients $c _ { 0 }$ and c<sub>T</sub>; stagnation trigger threshold $\mathcal { C }$ and reheating   
horizon $\Delta = 0 . 1 \dot { T }$ ; perturbation operators $\Phi _ { \theta }$ and rewrite operator $\Psi _ { \theta } .$   
2: Get $g _ { 0 } = g ( h _ { \mathrm { s e e d } } )$ where $h _ { \mathrm { s e e d } }$ is the seed heuristic function and set $t \gets 1$   
3: Initialize candidate set $\mathcal { P }  \{ h _ { \mathrm { s e e d } } \} \cup \{ h _ { 1 } , \ldots , h _ { N _ { \mathrm { i n i t } } - 1 } \}$ using $h _ { \mathrm { s e e d } }$ and an initialization   
prompt   
4: Evaluate all newly generated $h \in \mathcal { P }$ using $g$   
5: $h ^ { \star } \gets \arg \operatorname* { m a x } _ { h \in \mathcal { P } } g ( h ) ; \ t \gets | \mathcal { P } |$   
6: $\mathcal { P }  \pi _ { \mathrm { g r e e d y } } ( \mathcal { P } , \overset { \cdot } { N } ) ;$ ; initialize operator reward window $\mathcal { H } ; t ^ { \prime } \gets - \infty$   
7: while $t < T$ do   
8: $c _ { t } ^ { \mathrm { b a s e } }  c _ { 0 } + ( c _ { T } - c _ { 0 } ) \frac { t } { T }$   
9: $\mathbf { i } \mathbf { \check { f } } t - t ^ { \prime } < \Delta$ then   
10: $\begin{array} { r } { \hat { t } \gets \mathrm { c l i p } \Big ( \frac { t - t ^ { \prime } } { \Delta } , 0 , 1 \Big ) } \end{array}$   
11: $c _ { t } \gets c _ { t } ^ { \mathrm { b a s e } } + ( c _ { 0 } - \overset { \cdot } { c } _ { t } ^ { \mathrm { b a s e } } ) ( 1 - \hat { t } )$ $/ /$ operator reheating   
12: else   
13: $c _ { t } \gets c _ { t } ^ { \mathrm { b a s e } }$   
14: end if   
15: $h _ { \mathrm { r o o t } }  \mathrm { E p s i l o n G r e e d y } ( \mathcal { P } , \epsilon )$   
16: $\mathcal { F }  \{ h _ { \mathrm { r o o t } } \}$ // BFS frontier (layer 0)   
17: $s  \emptyset$   
18: for $\ell = 1$ to d do   
19: $\mathcal { C } _ { \ell } \gets \emptyset$   
20: for each parent $h \in \mathcal { F }$ do   
21: Select $\mathbf { \phi } ^ { \mathbf { \hat { \Phi } } } \Phi _ { \theta } \in \{ \Phi _ { \theta } ^ { \mathrm { e x p n } } , \Phi _ { \theta } ^ { \mathrm { c o m p } } , \Phi _ { \theta } ^ { \mathrm { s t e e r } } \}$ using UCB with coefficient $c _ { t }$   
22: $\tilde { z } \gets \Phi _ { \theta } ( z _ { h } \mid \mathsf { \bar { C } } ) ^ { \circ }$ $/ /$ perturbation   
23: $\{ h ^ { ( 1 ) } , \ldots , h ^ { ( \kappa ) } \} \sim \Psi _ { \theta } ( h , \tilde { z } , \Phi _ { \theta } \mid \mathcal { C } )$ // rewrite   
24: Evaluate all $h ^ { ( j ) }$ using $g ; t \gets t + \kappa$   
25: Update $\mathcal { H }$ using the best offspring improvement; $\{ { \mathcal { C } } _ { \ell } , S \} \gets \{ { \mathcal { C } } _ { \ell } , S \} \cup \{ h ^ { ( j ) } \} _ { j }$ :valid   
26: if $\dot { \mathbf { \zeta } } _ { t } \geq T$ then   
27: break   
28: end if   
29: end for   
30: $\mathbf { i f } t \geq T$ or $\mathcal { C } _ { \ell } = \varnothing$ then   
31: break   
32: end if   
33: $\mathcal { F } \sim \mathrm { S a m p l e } ( \mathcal { C } _ { \ell } , p ( h _ { i } ) )$ // According to Eq. (4)   
34: end for   
35: $\mathcal { P }  \pi _ { \mathrm { s e l } } ( \mathcal { P } \cup \mathcal { F } )$ $/ /$ preserve best and randomly retain remaining candidates   
36: Update $h ^ { \star }$ using all evaluated offspring in $s$   
37: if no global-best improvement for $\bar { \boldsymbol { c } }$ evaluations then   
38: $\mathcal { P } \overset { \vartriangle } {  } \{ h ^ { \star } \} ; t ^ { \prime } \gets \dot { t } ;$ clear $\mathcal { H }$ // Elite Pool Reset + Operator Reheating   
39: end if   
40: end while   
41: return Best valid heuristic $h ^ { \star }$ found

Function Contract Prompt   
You must implement the function:   
def select\_next\_item(remaining\_capacity, weights, values):   
return next\_item   
inputs:   
-remaining\_capacity: int   
-weights: np.ndarray   
-values: np.ndarray   
outputs:   
-next\_item: int   
Robustness constraints:   
- The select\_next\_item function takes as input the remaining capacity, a numpy array of   
item values, and a numpy array of item weights, and returns the next item to pick.

## D DETAILED METHODOLOGY

## D.1 PRS PROMPT

PRS uses a shared prompt library across all problem settings. It contains two system prompts that regulate the behavior of the LLM as an idea generator and as a code generator, respectively, as well as seven workflow prompt templates. These templates are used for: (i) describing the seed heuristic as a natural-language idea, (ii) generating initial heuristic ideas, (iii) applying the Expansion operator, (iv) applying the Compression operator, (v) applying the Steering operator, (vi) realizing an initial idea as executable code, and (vii) rewriting a parent implementation to realize a target idea.

Besides those common prompts, each problem setting contains two types of problem-specific prompts: function contract which dictates the format (including the names and types of inputs and outputs) and robustness constraints of designed heuristics, and a task description prompt (i.e., problem description). In this section, we describe their meaning and prompt engineering in detail, where we highlight the uniqueness of the prompt design in red colors. For problem-specific prompt components, we take the problem setting Knapsack Problem in step-by-step constructive framework as an example.

• Task Description and Function Contract: While the seed heuristics are only used once in the heuristic pool initialization prompt, the task description and function contract are employed not only during initialization, but also throughout the perturbation and rewrite stages of the evolutionary process. The robustness constraints in function contract do not add inductive bias that explicitly improves the performance of heuristics. Instead, they ensure the validity and consistency of the designed heuristics (e.g., the return must be of the same shape as the distance matrix). Below, we provide examples of the task description and function contract.

## Task Description Prompt

Solving 0-1 Knapsack Problem (KP) with constructive heuristics. The goal is to incrementally build a feasible solution by selecting items one by one, so as to maximize the total value under a weight (capacity) constraint. Each item can be selected at most once. At each step, the heuristic observes the current remaining capacity and the full list of item weights and values, and decides which item to select next. The selection is irreversible and directly affects future decisions.

• System Prompts: PRS uses two system prompts to separate semantic heuristic reasoning from executable code generation. The system idea prompt instructs the LLM to reason at the level of decision principles, interactions, and hyperparameters that affect behaviors, and to produce a concrete natural language heuristic idea. The system code prompt instructs the LLM to translate a supplied heuristic idea into a complete executable implementation. This separation allows the three directional search primitives to operate directly on explicit heuristic ideas, while code realization is guided by the resulting target idea and the required function interface.

## System Idea Prompt

You are an expert in optimization heuristics and algorithm design. Your task is to reason at the level of heuristic ideas, decision principles, interactions, and search logic. Do NOT output Python code.

\# Describe the heuristic idea concretely enough that another expert can implement it without guessing important decision logic.

\# Clearly explain what information or signals are used, how they affect decisions, how different mechanisms interact, and when the behavior changes.

\# For every behavior-affecting hyperparameter, threshold, schedule, or fallback, state its role and concrete value or rule. Do not leave implementation-critical parameter choices implicit.

\# Focus on algorithmic and heuristic logic rather than Python syntax, variable names, code length, or implementation layout.

\# Avoid cosmetic wording changes and parameter-only tweaks unless they meaningfully change the decision mechanism.

Return only the heuristic idea description.

## System Code Prompt

You are an expert in optimization heuristics and Python implementation. Your task is to translate the supplied heuristic idea into a complete executable Python heuristic.

\# Return only Python code wrapped in ‘python ... ‘.

\# Write all necessary imports inside the function when needed.

\# Do not output explanations, comments, or docstrings.

\# Do NOT use type annotations in the function signature.

• Initialization Prompts: PRS initializes the heuristic pool in two stages. The initial idea prompt generates a distinct heuristic idea from the seed idea, and the initial realization prompt translates it into executable code using the seed implementation as context.

![](images/8aa8ad640aa8f00ab0b1c8a66d4d42ffd409d7e64c119c754a62b28c1c4756c6.jpg)

<table><tr><td>Initial Realization Prompt</td><td></td></tr><tr><td>Function Name: {func_name}</td><td rowspan="3"></td></tr><tr><td>Task: {task_desc}</td></tr><tr><td>Function Format:</td></tr><tr><td>{contract} Target Heuristic Idea:</td><td rowspan="3"></td></tr><tr><td>{target_idea} Seed Implementation for Context: {seed_code}</td></tr><tr><td>Implement the TARGET HEURISTIC IDEA as a complete executable heuristic. The target idea is the semantic specification. Use the seed implementation only as low-level implementation context when useful.</td></tr><tr><td>Implementation Requirement: Faithfully realize the stated decision principles, information signals, interactions, conditions, and adaptive behavior. Do not invent additional heuristic mechanisms that are not implied by the target idea.</td><td rowspan="3"></td></tr><tr><td>When low-level implementation details are unspecified, choose simple implementations that preserve the intended decision logic.</td></tr><tr><td>Output Requirement: Return only valid Python code wrapped in “python ... “.</td></tr><tr><td>Task: {task_desc}</td></tr><tr><td>Current Heuristic Idea: {parent_idea} Perform IDEA COMPRESSION. Simplify the current heuristic at the level of algorithmic</td></tr><tr><td>reasoning while preserving its most useful core principle. Remove, merge, replace, or de- couple mechanisms that are redundant, weak, overly specific, or unnecessarily interacting.</td></tr><tr><td>Compression Requirement: When removing or merging a mechanism, explain how the remaining mechanisms should compensate for its role so that the resulting heuristic remains complete and directly imple- mentable.</td></tr><tr><td>The resulting description must still specify a complete decision rule: what information is retained, how it influences decisions, how the remaining mechanisms interact, and when</td></tr><tr><td>behavior changes. Output Requirement: Compression must reduce conceptual burden rather than merely shorten the description,</td></tr></table>

• Multi-directional Idea Operators: During BFS search, PRS applies one of three idea operators to transform each parent idea into a target idea. Expansion enriches the heuristic with new decision mechanisms, Compression removes or consolidates unnecessary mechanisms, and Steering improves the current strategy through localized modifications. The target idea is subsequently realized by a separate code generation prompt.

![](images/3a6af397c2dc1aa4847b6cf301235b7101400cd003f7940d87064abd061b1f45.jpg)

## Compression Operator Prompt

<table><tr><td>Task:</td></tr><tr><td>{task_desc}</td></tr><tr><td>Current Heuristic Idea: {parent_idea}</td></tr><tr><td>Perform IDEA STEERING. Preserve the core heuristic strategy and its overall conceptual structure, but improve it through localized modifications. Exploit the strengths of the current idea rather than redirecting it toward a fundamentally different heuristic family.</td></tr><tr><td>Steering Requirement: Focus on fine-grained changes such as adjusting how existing signals are weighted or</td></tr><tr><td>combined, modifying thresholds, schedules, or parameter settings, refining conditional logic or activation conditions, changing local interactions between existing mechanisms, or strengthening one specific part without altering the overall strategy.</td></tr><tr><td>Parameter-only changes are allowed only when they induce a meaningful behavioral change.</td></tr><tr><td>Avoid purely cosmetic rewriting.</td></tr><tr><td>Output Requirement:</td></tr><tr><td>Describe the locally steered heuristic concretely enough for direct implementation: state what local mechanism changes, how it affects decisions, and when it is active.</td></tr></table>

## Steering Operator Prompt

<table><tr><td>Heuristic Rewrite Prompt</td></tr><tr><td>Function Name: {func_name}</td></tr><tr><td>Task: {task_desc}</td></tr><tr><td>Function Format: {contract}</td></tr><tr><td>Target Heuristic Idea: {target_idea}</td></tr><tr><td>Operation that Produced the Target Idea: {operation}</td></tr><tr><td>Parent Implementation: {parent_code}</td></tr><tr><td>Rewrite the parent implementation into a complete executable heuristic that faithfully realizes the TARGET HEURISTIC IDEA. The target idea is the semantic specification of the new heuristic; the operation states the</td></tr><tr><td>intended kind of change. Rewrite Requirement: You may add, remove, replace, or reorganize any logic necessary to realize the target idea</td></tr><tr><td>completely. Preserve parent mechanisms only when they remain consistent with the target idea. Do not introduce unrelated mechanisms.</td></tr><tr><td>Faithfully instantiate the decision principles, information signals, interactions, conditional behavior, and adaptive mechanisms described in the target idea. Do not introduce additional heuristic mechanisms that are not implied by the target idea.</td></tr><tr><td>When low-level implementation details are unspecified, choose the simplest implementation consistent with the target idea. Output Requirement:</td></tr></table>

## D.2 GENERATED HEURISTICS

This subsection provides example outputs of PRS-designed heuristics. The problem settings include a step-by-step constructive framework for the online bin packing problem and the ant colony optimization framework for the orienteering problem.

## Evolved Heuristic #1: priority function for OBPP with step-by-step constructive framework

```python
def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:

This heuristic scores feasible bins using residual capacity, balanced
utilization, and protection against premature bin exhaustion. It favors tight
placements and moderate post-placement fill levels while giving additional
preference to near-exact fits and already utilized bins.
import numpy as np
eps = 1e-9
tight_threshold = item * 0.1
remaining = bins_remain_cap - item
capacity_scale = bins_remain_cap.max() if bins_remain_cap.size > 0 else 1.0
normalized_remaining = remaining / capacity_scale
post_fill = 1 - normalized_remaining
desired_fill = 0.5
# Favor small residual capacity and balanced post-placement utilization
residual_score = 1 / (remaining + eps)
fill_gap = post_fill - desired_fill
balance_score = np.exp(-20 * (fill_gap ** 2))
# Discourage opening or repeatedly using nearly empty bins
unused_penalty = np.exp(-10 * (normalized_remaining > (1 - item /
capacity_scale)) * normalized_remaining)
# Reward tight placement, near-exact fit, and previously utilized bins
tight_bonus = np.where(
remaining <= tight_threshold,
1.5 + (tight_threshold - remaining) / tight_threshold,
1.0
)
utilization = 1 - (bins_remain_cap / capacity_scale)
utilization_bonus = 1 + 0.5 * utilization
exact_match = (remaining < eps * 10).astype(float)
exact_bonus = 1 + 0.2 * exact_match
scores = residual_score * balance_score * unused_penalty * tight_bonus *
utilization_bonus * exact_bonus
return scores
```

## Evolved Heuristic #2: Heuristic Matrix for OP with ACO framework

def heuristics(prize: "np.ndarray", distance: "np.ndarray", maxlen: float) ->   
"np.ndarray":   
Edge-scoring heuristic that integrates reward, distance, and local   
neighborhood structure under a route-length constraint. It favors profitable,   
nearby, and well-connected transitions while suppressing moves that are   
difficult to complete feasibly.

## Evolved Heuristic #2 (cont.)

```python
import numpy as np
eps = 1e-9
n_nodes = prize.shape[0]
depot_dist = distance[:, 0]
prize_scale = np.max(prize[1:]) if n_nodes > 1 else 1.0
scaled_prize = prize / (prize_scale + eps)
inverse_dist = np.zeros_like(distance)
valid_dist = (distance > 0)
inverse_dist[valid_dist] = 1.0 / (distance[valid_dist] + eps)
prize_power, dist_power, density_power, connect_power = 7.0, 5.0, 4.0, 3.0
depot_weight, reward_threshold = 2.2, 0.25
local_radius = maxlen / 4.0
high_reward = (scaled_prize >= reward_threshold)
local_mask = (distance <= local_radius) & (distance > 0)
local_mask = local_mask & high_reward[None, :]
local_density = local_mask.sum(axis=1)
density_scale = np.max(local_density) if np.max(local_density) > 0 else 1.0
normalized_density = local_density / density_scale
density_bonus = 1.0 + normalized_density # reward nodes near many high-prize
neighbors
connect_radius = maxlen / 2.0
connect_mask = (distance <= connect_radius) & (distance > 0) &
high_reward[None, :]
node_connectivity = connect_mask.sum(axis=1)
connectivity_scale = np.max(node_connectivity) if np.max(node_connectivity) >
0 else 1.0
normalized_connectivity = node_connectivity / connectivity_scale
weak_node_penalty = np.where(normalized_connectivity < 0.3, 0.4, 1.0)
connectivity_penalty = weak_node_penalty[:, None] * weak_node_penalty[None, :]
# downweight weakly connected nodes
return_length = distance + depot_dist[None, :]
normalized_depot_dist = depot_dist / (maxlen + eps)
route_depth = (return_length + depot_dist[:, None]) / (maxlen + eps)
route_depth = np.clip(1.0 - route_depth, 0.0, 1.0)
infeasible_edges = return_length > maxlen # forbid edges that cannot return
to depot within maxlen
depot_affinity = 1.0 - normalized_depot_dist
depot_affinity = np.clip(depot_affinity, 0.0, 1.0)
depot_bonus = 1.0 + depot_weight * ((depot_affinity[:, None] +
depot_affinity[None, :]) / 2.0)
edge_prize = (scaled_prize[:, None] + scaled_prize[None, :]) / 2.0
ring_index = np.floor(normalized_depot_dist * 3).astype(int)
ring_distance = np.abs(ring_index[:, None] - ring_index[None, :])
ring_penalty = np.where(ring_distance > 1, 0.5, 1.0)
structure_score = ((density_bonus[:, None] * density_bonus[None, :]) **
density_power * (connectivity_penalty ** connect_power))
heuristic_matrix = ((edge_prize ** prize_power) * (inverse_dist ** dist_power)
* structure_score * route_depth * depot_bonus * ring_penalty)
heuristic_matrix[infeasible_edges] = 0.0
np.fill_diagonal(heuristic_matrix, 0.0)
heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=eps, posinf=eps,
neginf=eps)
heuristic_matrix = np.maximum(heuristic_matrix, 0.0)
return heuristic_matrix
```

## Evolved Heuristic #3: Black-Box Heuristic (Edge Scoring) for OP with ACO framework

```python
def heuristics(
node_attr: np.ndarray, edge_attr: np.ndarray, node_constraint: float
) -> np.ndarray:

This heuristic ranks edges using endpoint attributes and inverse edge cost,
with a soft penalty when the associated node-based constraint is exceeded.

n_nodes = len(node_attr)
score_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
for j in range(n_nodes):
if edge_attr[i, j] > 0:
combined_cost = np.sum(edge_attr[i]) + np.sum(edge_attr[:, j])
base_score = (node_attr[i] + node_attr[j]) / edge_attr[i, j]
if combined_cost > node_constraint:
score_matrix[i, j] = base_score * 0.5
else:
score_matrix[i, j] = base_score
else:
score_matrix[i, j] = -1
score_matrix = np.clip(score_matrix, -1, None)
score_matrix = (score_matrix - np.mean(score_matrix)) / (np.std(score_matrix)
+ 1e-10)
return score_matrix
```

## E DETAILS OF EVALUATION CONFIGURATIONS

Hyperparameters for PRS. To initialize the heuristic pool, we set the number of initially sampled heuristics to 30, aligning with ReEvo (Ye et al., 2024). The softmax temperature is fixed to $T _ { s e l } = 1 . 0$ for all tasks. The base temperature of the LLMs for both perturbation and rewrite is 1.0. During initialization, the LLM temperature is further increased by 0.2, to enhance diversity. Our code will be released upon acceptance.

For adaptive operator control, we set the initial and final UCB exploration coefficients to $c _ { 0 } = 2 . 0$ and $c _ { T } = 0 . 5$ , respectively, with a sliding reward window of 50 operator selections. The stagnation threshold and the post-restart operator reheating horizon are set to $C = 0 . 1 T$ and $\Delta = 0 . 1 T$ respectively. We set an evaluation timeout of 60s for each heuristic in PRS following MCTS-AHD (Zheng et al., 2025). For FSSP-GLS, this timeout is applied per instance within each heuristic evaluation.

Use of Large Language Models. For most experiments, we use GPT-5-nano-2025-08-08 under its default reasoning mode to ensure consistency. In some problem settings, we additionally adopt GPT-4.1-nano to examine the adaptability of PRS. We further evaluate both closed-source and opensource LLMs to demonstrate the applicability and flexibility of our framework, with additional results reported in the Appendix F.9. In designing heuristics under the ACO framework for CVRP, PRS uses on average 3.09M input tokens and 1.55M output tokens per run, with 1353 LLM queries and a total wall-clock time of 6h. Detailed runtime, token, and query-cost comparisons are reported in Appendix G.

Generation of Validation Sets D and Testing Setups. We follow the procedures in FunSearch (Romera-Paredes et al., 2024), EoH (Liu et al., 2024a), ReEvo (Ye et al., 2024) and MCTS-AHD to construct all validation datasets D and testing instances. In particular, MCTS-AHD employs a varying-scale validation dataset D for online BPP to facilitate the discovery of heuristics with improved generalization performance on a broader scale of testing instances, following the settings of ELG (Gao et al., 2024). The complete evaluation configurations are summarized as follows:

• ACO Framework. We adopt the configurations (i.e., number of ants and iterations) in ReEvo (Ye et al., 2024). (a) For CVRP (both black- and white-box settings), PRS is validated on a dataset of 10 instances with problem size 50 and tested on three sets of 64 instances each at size 50, 100, and 200. During both validation and testing, the number of ants and the number of iterations are fixed to 30 and 100, respectively. (b) For OP (both black- and white-box), the size of the dataset D is 5 and the validation problem size is also 50. As for evolved heuristics for OP, they are tested with three sets of 64 instances each at size 50, 100, and 200. The number of ants is 20 while the number of iterations is 50 for both validation and testing. (c) For black-box MKP, the validation dataset consists of 5 instances of problem size 100. The testing set consists of two sets of 64 instances each at sizes 100 and 200. The number of ants is 10 and the number of iterations is 50 and 100 for validation and testing, respectively.

• Step-by-step Construction Framework. (a) For KP, PRS is validated on a dataset D of 64 instances with problem size 100. Upon testing, the evolved heuristic is tested on three sets of 1,000 instances each at size 50, 100, 200. The capacity of size 50 is 12.5 while the capacity of size 100 and 200 is 25. (b) For Online BPP, PRS generates a dataset D consisting of 4 instances: a 1,000-item instance with W = 100, a 1,000-item instance with $W = 5 0 0 \mathrm { . }$ , a 5,000-item instance with W = 100, and a 5,000-item instance with $W = 5 0 0$ . The number of problem instances for each test set in Table 2 is 5. (c) For TSP, we use a validation set of 64 TSP instances with N = 50 nodes and four test sets of 1,000 instances each at $N = 5 0$ 100, 200, and 500. (d) For ASP, PRS uses a validation set D consisting of a single instance with $( n , w ) = ( 1 5 , 1 0 )$ . Upon testing, the evolved heuristic is evaluated on four settings, namely (12, 7), (15, 10), (21, 15), and (24, 17), with one instance for each setting.

• GLS Framework. (a) For FSSP, we follow EoH to generate D: a dataset of 64 instances is randomly generated with a fixed number of 50 jobs and the number of machines is randomly chosen from U[2, 20]. Details of the testing instances are provided in Appendix F.5. The maximum number of iterations is 1,000. The per instance timeout is 60s. (b) For TSP, we follow ReEvo to generate D of 64 instances of problem size 200 and keep the same iteration limits and perturbation moves, which are 1200 and 30, respectively. We evaluated the heuristics on a wide range of instances as shown in Table 10. The details of the hyperparameters chosen for GLS are provided in Table 5.

Hyperparameters for Baselines. We use the default hyperparameters reported in the original papers for the baseline AHD methods whenever possible. To ensure a fair comparison, we standardize the maximum number of evaluated heuristics across all methods, using T = 2000 for online BPP and T = 1000 for all remaining tasks. For EoH, we retain its default population sizes of 20 for online BPP and 10 for TSP and FSSP, but do not impose the default generation limit; instead, the search continues until the common evaluation budget is reached.

Table 5: GLS parameter settings for different TSP problem scales. The number of perturbation moves controls the strength of solution diversification, while the iteration limit determines the overall local search budget. Heuristics for TSP under GLS found by all LLM-based AHD frameworks are tested under the same configurations here.
<table><tr><td>Problem Size</td><td>Perturbation Moves</td><td>Iteration Limit</td></tr><tr><td>TSP20</td><td>5</td><td>73</td></tr><tr><td>TSP50</td><td>40</td><td>300</td></tr><tr><td>TSP100</td><td>40</td><td>1800</td></tr><tr><td>TSP200</td><td>40</td><td>1000</td></tr><tr><td>TSP500</td><td>60</td><td>250</td></tr><tr><td>TSP1000</td><td>80</td><td>100</td></tr></table>

![](images/59c3c917cd38b81622d1fd1007f969ddb8368bcce875764690125273b58cd7e9.jpg)  
Figure 4: Diversity analysis (CDI) of EoH, ReEvo, MCTS-AHD, and PRS on OBPP-Constructive and OP-ACO at different search stages (0.4T, 0.8T, and T). Error bars indicate standard deviation over three independent runs.

## F MORE EXPERIMENTAL RESULTS

## F.1 SEARCH DYNAMIC ANALYSIS

## F.1.1 DIVERSITY ANALYSIS

Diversity in evolutionary search provides valuable insights into the exploration strength and the convergence performance of AHD algorithms (Pires et al., 2014). A substantial body of optimization research has investigated the impact of estimating and maintaining diversity within the evolutionary search (Pires et al., 2019; Wang & Chen, 2012; Dat et al., 2025). The evolution curves alone in Section 5 are not sufficient to quantify the diversity within the heuristic search process of our PRS framework. Therefore, we adopt Cumulative Diversity Index (CDI) to quantify diversity, which is the same metric used in previous work, HSEvo (Dat et al., 2025). In the context of heuristic search, CDI reflects how widely heuristic candidates are dispersed in the search space, serving as an implicit signal of system energy, or diversity (Jost, 2006).

To align with the approaches in calculating CDI in HSEvo, we utilized the cumulative archive of the embeddings of the top-20 candidates in every 80 evaluation steps to measure CDI for three other AHD baselines (i.e., EoH, ReEvo, and MCTS-AHD) and our PRS framework. CDI is calculated by constructing a Minimum Spanning Tree (MST) of all embeddings in the archive A with edge lengths measured by Euclidean distance. The MST is then used to assess heuristic diversity. Let $d _ { i }$ denote the length of the i-th edge in the minimum spanning tree (MST) constructed over the archive A, where $i \in \bar { \{ 1 , 2 , \ldots , | A | - 1 \} }$ . The cumulative diversity is computed using the Shannon entropy as

$$
D _ { \mathrm { C D I } } = - \sum _ { i = 1 } ^ { | A | - 1 } p _ { i } \log p _ { i } .\tag{19}
$$

where the corresponding probability distribution of distance $d _ { i }$ is defined as

$$
p _ { i } = \frac { d _ { i } } { \sum _ { j = 1 } ^ { | A | - 1 } d _ { j } } , \quad i = 1 , \ldots , | A | - 1 .\tag{20}
$$

A higher CDI reveals a more diverse dispersion of heuristic structures in the search space, and thus suggests more robust exploration dynamics. We mainly consider three stages to demonstrate the diversity of search in different phases of heuristic search: ${ \scriptstyle \frac { 2 } { 5 } } T , \ { \scriptstyle \frac { 4 } { 5 } } T .$ , and $\check { T }$ within the maximum evaluation steps $T ,$ , corresponding to the early, middle, and final stages of heuristic search $( T = 1 0 0 0$ for OP and $T = 2 0 0 0$ for OBPP).

As shown in Figure 4, all four AHD frameworks exhibit ascending CDI values as the evaluation steps grow, indicating that their cumulative heuristic archives become progressively more diverse during search. Across both OBPP and OP, PRS consistently achieves the highest or near-highest CDI at the middle and final stages and attains the largest final stage diversity among all compared methods.

Compared to other baselines, PRS’s diversity growth is particularly pronounced from 0.8T to T, suggesting that PRS maintains active exploration even in the late stage of the search rather than rapidly collapsing into a narrow set of heuristic structures. This delayed convergence indicates a stronger capability to avoid premature convergence and preserve diverse heuristic candidates throughout the search process.

## F.1.2 IDEA-SPACE DIVERSITY ANALYSIS

CDI measures diversity in the learned representation space of realized heuristic implementations, while mean nearest-neighbor distance (MNND) measures diversity in the semantic idea space before code realization. This distinction is important for PRS because its directional perturbation primitives operate directly on natural language heuristic ideas. Distance-based criteria have long been used to characterize dispersion and space-filling properties in multidimensional spaces (Johnson et al., 1990; Morris & Mitchell, 1995), while mean nearest-neighbor distance provides a direct measure of local packing and redundancy (Layman et al., 2007). In the context of AHD, MNND characterizes how semantically distinct each retained heuristic idea is from its closest alternative in the archive: a larger value indicates fewer locally redundant ideas and broader coverage of distinct conceptual neighborhoods in the heuristic design space. For each distinct heuristic idea $z _ { i }$ in the cumulative archive, we obtain an L2-normalized sentence embedding $\mathbf { e } _ { i }$ using sentence-transformers/all-mpnet-base-v2. The nearest-neighbor semantic distance of $z _ { i }$ is defined using cosine distance, and MNND is computed as

$$
D _ { \mathrm { M N N D } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m i n } _ { j \neq i } \left( 1 - \frac { { \bf e } _ { i } ^ { \top } { \bf e } _ { j } } { \| { \bf e } _ { i } \| _ { 2 } \| { \bf e } _ { j } \| _ { 2 } } \right) ,\tag{21}
$$

where N denotes the number of distinct ideas in the archive. By focusing on each idea’s nearest semantic neighbor, MNND reflects local redundancy and sustained exploration across distinct regions of the heuristic idea space.

We follow the same archive construction protocol as in the CDI analysis, adding the top-20 valid candidates every 80 evaluations to a cumulative archive. Exact duplicate ideas are counted only once for MNND to avoid bias from repeated code realizations. The same treatment is applied to all compared methods. We evaluate PRS against EoH and MCTS-AHD, both of which explicitly maintain natural language heuristic descriptions. To minimize differences caused merely by the wording or style of the thought-generation prompts, we additionally standardize the natural language idea-generation prompt across the three methods while retaining their original search mechanisms. Thus, the comparison primarily reflects differences in how the respective search frameworks explore heuristic ideas rather than differences in description style. Results are averaged over three independent runs, with shaded regions denoting standard deviation.

As shown in Figure 5, PRS generally explores less locally redundant regions of the idea space than EoH and MCTS-AHD across both tasks. On OBPP-Constructive, although MCTS-AHD shows a slightly larger MNND at the earliest checkpoint, PRS subsequently maintains the highest mean MNND and finishes at approximately 0.117, compared with 0.116 for MCTS-AHD and 0.109 for EoH. On OP-ACO, PRS shows a clearer advantage during the early and middle stages, reaching an MNND of approximately 0.120, while EoH and MCTS-AHD remain lower; the gap narrows toward the end, but PRS still retains the highest final mean MNND. These results indicate that the directional perturbation primitives allow PRS to reach distinct semantic neighborhoods earlier and sustain broader idea space exploration throughout the search.

Together with the CDI results, the MNND analysis provides two complementary views of search diversity. CDI shows that PRS maintains broad dispersion among the realized heuristic implementations, whereas MNND shows that this diversity is already present at the level of semantic heuristic ideas. Their consistent trends suggest that the broader exploration of PRS is not merely a consequence of superficial variation in code realization. Instead, the explicit expansion, compression, and steering primitives promote sustained exploration of distinct regions of the heuristic design space before these ideas are instantiated as executable heuristics.

![](images/3c9f47646575d01e747d4a925a3f343976044a21607e6f1537a0e8733231488c.jpg)  
(a) OBPP-Constructive

![](images/ba6788087eecb0acefecb1148fab6c0432f8c12d66fae8a8e2d4ed27c5c5eef5.jpg)  
(b) OP-ACO  
Figure 5: Idea-space diversity measured by mean nearest-neighbor distance (MNND) for PRS, MCTS-AHD, and EoH. Higher values indicate that retained heuristic ideas remain farther from their closest semantic neighbors.

## F.1.3 OPERATOR CONTRIBUTION ANALYSIS

In this section, we further analyze how the three directional perturbation operators complement each other during search and how their contributions vary across different tasks and search stages. We measure the cumulative share of best-so-far improvements triggered by each operator across three representative settings: FSSP-GLS, KP-Constructive, and CVRP-ACO, based on five independent runs for each setting (Figure 6).

1) FSSP-GLS. In FSSP-GLS, compression is the dominant contributor, accounting for approximately 46% of the cumulative improvements, and its contribution continues to increase throughout the search trajectory. Expansion contributes about 32%, with most improvements occurring during the earlier and middle stages, while steering accounts for the remaining 22%. This pattern suggests that both enriching and simplifying heuristic structures are useful in FSSP-GLS, with compression playing a particularly persistent role throughout the search.

2) KP-Constructive. In KP-Constructive, the pattern changes substantially: expansion becomes the largest contributor at approximately 42%, while steering also contributes strongly at around 37%. Compression accounts for a smaller share of approximately 21%, with most of its improvements occurring relatively early in the search. The comparable contributions of expansion and steering indicate that KP benefits not only from exploring richer heuristic structures, but also from fine-grained exploitation around promising regions.

3) CVRP-ACO. In CVRP-ACO, compression again becomes the dominant contributor, accounting for approximately 47% of cumulative improvements, followed by steering at around 37%, whereas expansion contributes about 16%. Compression accumulates improvements across the search trajectory, while steering also remains active across multiple stages. This result shows that effective search in CVRP-ACO relies more strongly on structural simplification and local refinement than on continued structural expansion.

Stage-wise UCB behavior. To further examine the stage-wise behavior of adaptive operator selection, Table 6 reports the stage-wise behavior of the UCB controller. For each operator, we consider its selection rate, the probability of producing at least one improving beam child, and the mean best-child relative improvement. The results show that contribution is not determined by selection frequency alone. Across tasks and search stages, operators with similar selection rates can exhibit markedly different improvement probabilities and improvement magnitudes, indicating that the three perturbation directions provide distinct search utility even when they receive comparable amounts of search effort.

The stage-wise statistics also reveal clear task-dependent differences. In CVRP-ACO, compression maintains consistently strong improvement rates across later stages, which is consistent with its large cumulative contribution. In FSSP-GLS, compression frequently achieves high improvement rates, while steering often produces larger best-child improvements in several stages. For OBPP-

![](images/5b65339484dc786d3fa43b875879b5e546c6270ee7139f1c2f8c2b93d3152fe2.jpg)  
Figure 6: Operator contribution statistics. Each curve represents the cumulative share of best-so-far improvements triggered by each perturbation operator across all runs.

Table 6: Stage-wise behavior of UCB-based operator selection. For each operator, we report the selection rate (Sel.), any-improvement rate (Imp.), and mean best-child relative improvement (Gain), all in %. Statistics are aggregated over five runs.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Evaluation stage</td><td colspan="3">Expansion</td><td colspan="3">Compression</td><td colspan="3">Steering</td></tr><tr><td>Sel.</td><td>Imp.</td><td>Gain</td><td>Sel.</td><td>Imp.</td><td>Gain</td><td>Sel.</td><td>Imp.</td><td>Gain</td></tr><tr><td rowspan="4">OBPP-Constructive</td><td>0-249</td><td>32.42</td><td>35.21</td><td>3.241</td><td>34.25</td><td>26.67</td><td>5.881</td><td>33.33</td><td>41.10</td><td>4.849</td></tr><tr><td>250-499</td><td>32.13</td><td>23.75</td><td>1.314</td><td>34.14</td><td>12.94</td><td>0.678</td><td>33.73</td><td>16.67</td><td>2.947</td></tr><tr><td>500-749</td><td>34.15</td><td>20.24</td><td>3.541</td><td>32.11</td><td>16.46</td><td>2.415</td><td>33.74</td><td>22.89</td><td>1.958</td></tr><tr><td>750-999</td><td>31.43</td><td>16.88</td><td>2.632</td><td>31.02</td><td>10.53</td><td>4.006</td><td>37.55</td><td>23.91</td><td>7.667</td></tr><tr><td rowspan="4">CVRP-ACO</td><td>0-249</td><td>31.36</td><td>66.67</td><td>3.947</td><td>34.09</td><td>66.67</td><td>4.059</td><td>34.55</td><td>53.95</td><td>6.751</td></tr><tr><td>250-499</td><td>32.40</td><td>51.85</td><td>2.867</td><td>34.40</td><td>70.93</td><td>2.171</td><td>33.20</td><td></td><td>45.782.513</td></tr><tr><td>500-749</td><td>30.49</td><td>52.00</td><td>2.220</td><td>36.59</td><td>65.56</td><td>3.554</td><td>32.93</td><td></td><td>43.213.106</td></tr><tr><td>750-999</td><td>33.73</td><td>54.12</td><td>3.244</td><td>34.92</td><td>68.18</td><td>3.872</td><td>31.35</td><td>60.76</td><td>3.269</td></tr><tr><td rowspan="4">FSSP-GLS</td><td>0-249</td><td>33.78</td><td>44.00</td><td>3.329</td><td>31.98</td><td>74.65</td><td>4.735</td><td>34.23</td><td>59.21</td><td>5.476</td></tr><tr><td>250-499</td><td>31.98</td><td>56.96</td><td>3.557</td><td>34.01</td><td>61.90</td><td>3.236</td><td>34.01</td><td></td><td>51.194.500</td></tr><tr><td>500-749</td><td>32.11</td><td></td><td></td><td>46.841.52234.55</td><td>47.06</td><td></td><td>1.984 33.33</td><td></td><td>47.563.667</td></tr><tr><td>750-999</td><td>35.3444.323.30931.33</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>64.101.38833.3346.993.499</td></tr></table>

Constructive, the relative utility of the operators changes more noticeably over time, with steering becoming particularly effective in the final stage. Together with Figure 6, these results show that operator effectiveness varies across both tasks and search stages, supporting adaptive allocation among expansion, compression, and steering rather than a fixed operator preference.

## F.1.4 OPERATOR-WISE GENERALIZATION

The preceding operator contribution analysis shows that expansion, compression, and steering contribute differently across tasks and search stages. We further examine whether the strongest candidates produced by different perturbation operators exhibit different in-distribution (ID) and out-of-distribution (OOD) generalization behavior.

For each independent run, we collect the valid children generated by each perturbation operator and rank them according to validation performance. Since AHD search ultimately prioritizes and retains high-performing candidates, we focus on the strongest outcomes produced by each perturbation direction and evaluate the top five children from each operator on both ID test instances and larger OOD instances.

Table 7 shows a consistent qualitative difference between the strongest outcomes produced by the three perturbation directions. On KP-Constructive, expansion achieves the best ID performance, whereas compression achieves the best OOD performance. Steering remains competitive across both evaluation regimes.

Table 7: Generalization performance of the top-five children generated by each perturbation operator. Children are selected according to validation performance within each run. Lower gap is better.
<table><tr><td>Task</td><td>Operator</td><td>ID gap ↓</td><td>OOD gap ↓</td></tr><tr><td>KP-Constructive</td><td>Expansion</td><td> $\mathbf { 0 . 0 3 \% \pm 0 . 0 1 \% }$ </td><td> $0 . 0 4 \% \pm 0 . 0 2 \%$ </td></tr><tr><td></td><td>Compression</td><td> $0 . 0 6 \% \pm 0 . 0 2 \%$ </td><td> $\mathbf { 0 . 0 3 \% \pm 0 . 0 0 \% }$ </td></tr><tr><td></td><td>Steering</td><td> $0 . 0 5 \% \pm 0 . 0 2 \%$ </td><td> $0 . 0 5 \% \pm 0 . 0 3 \%$ </td></tr><tr><td>CVRP-ACO</td><td>Expansion</td><td> $\mathbf { 0 . 8 9 \% \pm 0 . 5 3 \% }$ </td><td> $6 . 3 1 \% \pm 1 . 5 4 \%$ </td></tr><tr><td></td><td>Compression</td><td> $1 . 4 5 \% \pm 0 . 7 1 \%$ </td><td> $\mathbf { 5 . 9 4 \% \pm 1 . 0 2 \% }$ </td></tr><tr><td></td><td>Steering</td><td> $1 . 2 6 \% \pm 0 . 4 3 \%$ </td><td> $6 . 5 2 \% \pm 0 . 6 3 \%$ </td></tr></table>

A similar pattern appears on CVRP-ACO, where expansion again performs best on ID instances while compression achieves the strongest performance on larger OOD instances. Steering provides competitive intermediate performance without consistently favoring either regime.

These results do not imply that expansion is generally preferable for ID performance or that compression always improves OOD generalization. Rather, they suggest that different perturbation directions can produce strong heuristics under different evaluation conditions.

## F.2 COMPARISON ON BLACK-BOX SETTINGS

We further evaluate PRS under black-box evaluation scenarios, which are inherently harder for heuristic search: candidate programs must remain executable under strict interfaces, and improvements must be discovered purely through outcome-driven trial-and-error, making robust diversification essential. Table 8 shows that PRS remains highly competitive or consistently exceeds other AHD frameworks on black-box tasks. These results suggest that PRS can sustain effective exploration even when supervision is extremely sparse, supporting robust heuristic search in opaque environments where objective evaluations are costly.

## F.3 TSP STEP-BY-STEP CONSTRUCTIVE FRAMEWORK

For TSP, PRS is responsible for designing a function to select the next node to visit based on the node coordinates, the starting point, the distance matrix, and all unvisited nodes. The experiments in Table 9 test the heuristics on four different scales, where PRS consistently achieves the best performance among the compared LLM-based AHD methods under both LLM settings. For each method, we set a 1200s total time limit to ensure a manageable testing time budget and avoid excessive validation cost.

Discussion. Across both LLM settings, PRS consistently outperforms all compared LLM-based AHD methods on TSP50, TSP100, TSP200, and TSP500. In particular, PRS improves upon MCTS-AHD across all four problem sizes under both GPT-4.1-nano and GPT-5-nano, while the advantage remains evident as the problem scale increases. The results also show that PRS benefits consistently from the stronger GPT-5-nano backbone, achieving lower gaps across all four test sizes. Overall, these results demonstrate that the proposed multi-directional search process remains effective across different LLM backbones and generalizes well from the search setting to substantially larger TSP instances.

Table 8: Performance of ACO-based heuristics on black-box CVRP, MKP, and OP. We follow the settings of ReEvo (Ye et al., 2024) in heuristic evolution and run each LLM-based method three times for average performance.
<table><tr><td>Task</td><td>CVRP</td><td>MKP</td><td></td><td>OP</td></tr><tr><td>N= Method</td><td>N=50 N=100 Obj.↓ Obj.↓</td><td>N=100 Obj.↑</td><td>N=200 Obj.↑</td><td>N=50 N=100 Obj.↑ Obj.↑</td></tr><tr><td>ACO DeepACO</td><td>18.282 8.888</td><td>30.030 18.969 14.932 23.091</td><td>38.766 41.988</td><td>13.570 24.149 19.867 36.392</td></tr><tr><td colspan="5">LLM-based AHD (GPT-5-nano)</td></tr><tr><td rowspan="4">EoH ReEvo MCTS-AHD</td><td>9.360 16.700</td><td>22.518</td><td>40.568</td><td>14.776</td></tr><tr><td>9.403 16.693</td><td>23.192</td><td>42.148</td><td>28.627 14.717</td></tr><tr><td>9.551 17.070</td><td>23.185</td><td>42.111</td><td>28.603 28.448</td></tr><tr><td>9.351 16.627</td><td>23.207</td><td>14.815 42.174 14.789</td><td>28.898</td></tr></table>

Table 9: Performance of step-by-step construction heuristics on TSP, averaged over three runs. Bold denotes the best non-optimal objective value at each problem size. For each LLM setting, the best LLM-based AHD result in each column is shaded gray. Gaps are the relative difference to the optimal (or near-optimal) objective value obtained by LKH3.
<table><tr><td colspan="7">Travelling Salesman Problem (TSP)</td></tr><tr><td rowspan="2">Method</td><td>TSP 50</td><td>TSP 100</td><td></td><td>TSP 200</td><td></td><td>TSP 500</td></tr><tr><td>Obj.↓ Gap↓</td><td>Obj.↓ Gap↓</td><td>Obj.↓</td><td>Gap↓</td><td>Obj.↓</td><td>Gap↓</td></tr><tr><td>LKH3 (Lin &amp; Kernighan, 1973)|</td><td>5.680</td><td>7.762</td><td></td><td>10.715</td><td>16.524</td><td></td></tr><tr><td>POMO</td><td>5.697 0.30%</td><td>8.001 3.08%</td><td></td><td>12.897 20.37%</td><td></td><td>20.19022.18%</td></tr><tr><td>GC (Nearest Neighbour)</td><td>6.94422.26%</td><td></td><td>9.75525.67%</td><td>13.502 26.02%</td><td></td><td>20.93826.71%</td></tr><tr><td colspan="7">LLM-based AHD GPT-4.1-nano</td></tr><tr><td>FunSearch</td><td>6.58315.90%</td><td></td><td>9.14017.75%</td><td>12.79819.44%</td><td></td><td>19.985 20.94%</td></tr><tr><td>EoH</td><td>6.34511.71%</td><td></td><td>8.86214.17%</td><td>12.42815.99%</td><td></td><td>19.56818.42%</td></tr><tr><td>MCTS-AHD</td><td>6.33711.57%</td><td></td><td>8.86514.21%</td><td>12.41915.91%</td><td></td><td>19.382 17.29%</td></tr><tr><td>PRS (Ours)</td><td>6.299 10.90%</td><td>8.797 13.33%</td><td></td><td>12.268 14.50%</td><td></td><td>19.092 15.54%</td></tr><tr><td colspan="7">LLM-based AHD GPT-5-nano</td></tr><tr><td>FunSearch</td><td>6.44413.46%</td><td></td><td>9.10417.28%</td><td>12.87420.15%</td><td></td><td>20.02121.16%</td></tr><tr><td>EoH</td><td>6.37712.27%</td><td></td><td>8.75512.80%</td><td>12.56317.25%</td><td></td><td>19.87420.27%</td></tr><tr><td>MCTS-AHD</td><td>6.29910.89%</td><td></td><td>8.69412.00%</td><td>12.36415.39%</td><td></td><td>18.82513.92%</td></tr><tr><td>PRS (Ours)</td><td>6.215 9.42%</td><td></td><td>8.568 10.38%</td><td>12.161 13.50%</td><td></td><td>18.553 12.28%</td></tr></table>

## F.4 TSP HEURISTIC UNDER GUIDED LOCAL SEARCH FRAMEWORK

To evaluate the effectiveness of PRS in designing penalty heuristics within the guided local search (GLS) framework, we follow the general experimental protocol of Ye et al. (Ye et al., 2024). Specifi cally, PRS is employed to generate the key penalty heuristic used to construct the knowledge matrices in KGLS (Arnold & Sörensen, 2019), which enables adaptive refinement of the original manually designed heuristic.

All heuristic generation and evaluation procedures are conducted using GPT-5-nano as the backbone LLM. We evaluate the resulting heuristics on different problem scales of TSP, and compare against representative LLM-based AHD baselines, the original KGLS with handcrafted penalty functions, as well as two neural combinatorial optimization (NCO) methods, namely NeuralOpt (Ma et al., 2021) and GNNGLS (Hudson et al., 2022). NeuralOpt is an NCO method that learns to improve the solutions, while GNNGLS is primarily based on GLS framework. The GLS hyperparameters are scaled with problem size to balance exploration and computational cost. Specifically, the number of perturbation moves and the iteration limit are adaptive to problem sizes, as summarized in Table 5. This adaptive budgeting ensures fair comparison across problem scales while avoiding excessive local search on large instances.

As shown in Table 10, PRS consistently improves the performance of KGLS across different problem scales. In particular, PRS achieves lower optimality gaps on TSP50, TSP100 and TSP500 compared to other LLM-based AHD methods while maintaining competitive or the lowest runtime. Notably, our approach consistently reaches zero gaps on small-scale problems (TSP50 and TSP100) and achieves the lowest gap 1.649% on TSP 500 and the second lowest on TSP 200. Together with Table 9, these findings further demonstrate that our multi-directional perturbation–rewrite framework is effective and adaptable across different heuristic formulations and problem scales.

Pareto analysis of runtime and solution quality. Beyond reporting the final best-performing heuristic in Table 10, we further analyze the quality–efficiency tradeoff of the heuristics discovered by different AHD frameworks on TSP under the GLS framework. Figure 7 visualizes the runtime and tour length of candidate heuristics discovered during search at four problem scales.

![](images/64d29efa5808031b136241122b57ecbf81c7bb13b4a7c8e9aa03b0f6f69fd113.jpg)  
Figure 7: Pareto analysis of runtime and solution quality on TSP under the GLS framework. Each scatter point corresponds to one heuristic candidate evaluated during search, where the x-axis denotes runtime per instance and the y-axis denotes tour length. Results are shown for four problem sizes $( n = 5 0 , 1 0 0 , 2 0 0 , 5 0 0 )$ . The connected frontier for each method highlights the best observed quality– efficiency trade-off among its discovered heuristics. Lower tour length and runtime are preferred.

Table 10: Performance of heuristics on TSP benchmarks with guided local search framework. For a fair comparison, we report the result of the best performing heuristic over three independent runs following ReEvo (Ye et al., 2024). Gap (%) denotes the relative optimality gap to the best-known solution, and Time (s) denotes the average runtime per instance. \* denotes results drawn from ReEvo (Ye et al., 2024).
<table><tr><td rowspan="2">Method</td><td colspan="2">TSP50</td><td colspan="2">TSP100</td><td colspan="2">TSP200</td><td colspan="2">TSP500</td></tr><tr><td>Gap(%)↓</td><td>Time(s)</td><td>Gap(%)↓</td><td>Time(s)</td><td>Gap(%)↓</td><td>Time(s)</td><td>Gap(%)↓</td><td>Time(s)</td></tr><tr><td>NeuOpt*</td><td>0.000</td><td>1.32</td><td>0.027</td><td>2.67</td><td>0.403</td><td>4.81</td><td>一</td><td>一</td></tr><tr><td>GNNGLS*</td><td>0.052</td><td>3.83</td><td>0.705</td><td>6.78</td><td>3.522</td><td>9.92</td><td>一</td><td></td></tr><tr><td>EoH</td><td>0.000</td><td>0.03</td><td>0.390</td><td>0.47</td><td>0.196</td><td>0.90</td><td>1.667</td><td>1.93</td></tr><tr><td>ReEvo</td><td>0.000</td><td>0.04</td><td>0.025</td><td>0.55</td><td>0.338</td><td>1.26</td><td>1.703</td><td>4.56</td></tr><tr><td>MCTS-AHD</td><td>0.000</td><td>0.02</td><td>0.000</td><td>0.47</td><td>0.270</td><td>0.90</td><td>1.680</td><td>1.82</td></tr><tr><td>Ours</td><td>0.000</td><td>0.02</td><td>0.000</td><td>0.46</td><td>0.212</td><td>0.90</td><td>1.649</td><td>1.81</td></tr></table>

Across all scales, PRS shows a consistently favorable Pareto frontier. At smaller scales $( n = 5 0$ and $n = 1 0 0 )$ , the heuristics discovered by PRS achieve lower tour lengths while remaining among the fastest candidates. At larger scales (n = 200 and $n = 5 0 0 )$ , this advantage becomes more evident: PRS maintains a frontier that is competitive or better in solution quality while avoiding the large runtime overhead observed in some baselines, especially ReEvo. This indicates that PRS is able to discover heuristics with a stronger balance between effectiveness and computational efficiency, rather than improving quality merely through more expensive heuristic structures.

The results also provide additional evidence for the discussion in Appendix F.3. PRS tends to discover candidates that remain closer to the efficient part of the quality–runtime tradeoff frontier, especially on medium- and large-scale TSP instances.

## F.5 FLOW SHOP SCHEDULING PROBLEM UNDER GUIDED LOCAL SEARCH FRAMEWORK

We adopt the same guided local search (GLS) framework equipped with two neighborhood operators, namely Swap and Relocate as EoH. Within this framework, PRS is employed to automatically design a heuristic strategy that jointly governs the update of the execution-time matrix and the selection of jobs to be perturbed during the search process. The heuristics generated by PRS are compared against a diverse set of non-AHD methods, including both classical heuristics and learning-based approaches. Specifically, we consider the following competitors: (i) GUPTA (Gupta, 1971), a classical heuristic for the flow-shop scheduling problem (FSSP); (ii) CDS (Campbell et al., 1970), another well-established heuristic for FSSP; (iii) NEH (Nawaz et al., 1983), which is widely regarded as an effective heuristic for flow-shop scheduling; (iv) NEHFF (Fernandez-Viagas & Framinan, 2014), an enhanced variant of NEH; (v) PFSPNet (Pan et al., 2021), a recently proposed deep learning-based solver for flow-shop scheduling; (vi) Local Search (LS), which applies the same neighborhood operators as PRS without heuristic adaptation; (vii) ILS1 (Stützle, 1998), a classical iterated local search method for FSSP; and (viii) ILS2, which follows the same overall framework as our method but relies on manually designed heuristic rules. We take EoH as the LLM-based AHD baseline. All methods are evaluated on the widely used Taillard (Taillard, 1993) benchmark instances. The number of jobs ranges from 50 to 200, while the number of machines varies between 5 and 20. Each reported value is the average relative gap over 10 instances in the corresponding test set.

As shown in Table 11, PRS achieves lower gaps across Taillard instances than EoH in most test sets (e.g., 0.27% vs 0.62% on j100\_m20), outperforming it on average while also demonstrating superior performance against non-AHD baselines, including NEH-based heuristics and iterated local search methods.

## F.6 RESULTS ON TSPLIB INSTANCES

We conduct experiments on the well-known real-world TSP benchmark TSPLib (Reinelt, 1991) to evaluate the effectiveness of the step-by-step constructive heuristics of PRS in generalizing to real-world TSP instances. We consider TSPLib instances with up to 500 cities following MCTS-AHD (Zheng et al., 2025) and report the optimality gap with respect to the best-known solutions.

We compare PRS against manually designed heuristics, including Greedy, as well as the GP-based AHD method GHPP (Duflo et al., 2019). For LLM-based AHD baselines, we include the bestperforming heuristics of EoH and ReEvo. All LLM-designed heuristics are executed three times with different starting nodes and averaged for fair comparison. The results of the instances reported in GHPP within the problem scale [0, 500] are reported in Table 12. As shown in Table 12, the heuristic generated by PRS achieves a lower overall optimality gap than manually designed heuristics and the

Table 11: Results on Taillard (Taillard, 1993) instance sets. The results of non-LLM-based methods are cited from EoH (Liu et al., 2024a). Each entry reports the average relative gap (%) to the known upper bound in Taillard (lower is better). The best result in each row is highlighted in bold. Additionally, among all the LLM-based AHD methods, the best value is shaded.
<table><tr><td>Test Set</td><td>GUPTA</td><td>CDS</td><td>NEH</td><td>NEHFF</td><td>PFSPNet</td><td>LS</td><td>ILS1</td><td>ILS2</td><td>EoH</td><td>Ours</td></tr><tr><td>j50_m5</td><td>12.23</td><td>6.98</td><td>0.57</td><td>0.40</td><td>0.51</td><td>0.32</td><td>0.15</td><td>0.32</td><td>0.00</td><td>0.00</td></tr><tr><td>j50_m10</td><td>20.11</td><td>12.72</td><td>3.47</td><td>3.62</td><td>3.48</td><td>3.33</td><td>1.47</td><td>0.29</td><td>0.21</td><td>0.17</td></tr><tr><td>j50_m20</td><td>22.78</td><td>15.03</td><td>5.48</td><td>5.10</td><td>5.05</td><td>4.67</td><td>2.13</td><td>0.34</td><td>0.72</td><td>0.83</td></tr><tr><td>j100_m5</td><td>5.98</td><td>5.10</td><td>0.39</td><td>0.31</td><td>0.31</td><td>0.28</td><td>0.20</td><td>0.38</td><td>-0.04</td><td>-0.05</td></tr><tr><td>j100_m10</td><td>15.03</td><td>9.36</td><td>2.07</td><td>1.88</td><td>1.72</td><td>1.38</td><td>0.77</td><td>0.34</td><td>0.22</td><td>0.14</td></tr><tr><td>j100_m20</td><td>21.00</td><td>13.55</td><td>3.58</td><td>3.73</td><td>3.56</td><td>3.51</td><td>2.27</td><td>0.43</td><td>0.62</td><td>0.27</td></tr><tr><td>j200_m10</td><td>11.59</td><td>7.22</td><td>0.98</td><td>0.70</td><td>0.82</td><td>0.87</td><td>0.74</td><td>0.54</td><td>0.18</td><td>0.18</td></tr><tr><td>j200_m20</td><td>18.09</td><td>11.89</td><td>2.90</td><td>2.52</td><td>2.49</td><td>2.53</td><td>2.26</td><td>0.59</td><td>0.65</td><td>0.55</td></tr><tr><td>Average</td><td>15.85</td><td>10.23</td><td>2.43</td><td>2.28</td><td>2.24</td><td>2.11</td><td>1.25</td><td>0.40</td><td>0.32</td><td>0.26</td></tr></table>

GP-based AHD baseline, and leads all methods in 8 of the 15 instances. Collectively, these results demonstrate the competitive performance of PRS for designing TSP-Constructive heuristics that generalize well to an out-of-distribution benchmark.

Table 12: Results of the GP-based method GHPP, and LLM-based AHD methods on designing step-by-step constructive heuristics for TSP on TSPLib instances. We report the optimality gap (%) of each instance. The results of GHPP-best, Christofides (Christofides, 2022), Greedy (Brecklinghaus & Hougardy, 2015), Nearest insertion and Nearest-greedy (Rosenkrantz et al., 1977) are taken from (Duflo et al., 2019). LLM-based AHD methods (EoH, ReEvo, and PRS) are each run three times with different starting nodes and averaged. The leading LLM-designed heuristic on each instance is shaded, and the overall best heuristic is shown in bold.
<table><tr><td>Instance</td><td></td><td></td><td>Christofides Greedy Nearest insertion Nearest-greedy [GHPP-best</td><td></td><td></td><td>EoH</td><td>ReEvo</td><td>Ours</td></tr><tr><td>bier127</td><td>13.03</td><td>19.50</td><td>23.05</td><td>23.25</td><td>15.64</td><td>19.60</td><td>10.79</td><td>8.31</td></tr><tr><td>d493</td><td>9.52</td><td>16.68</td><td>20.39</td><td>24.00</td><td>15.58</td><td>17.02</td><td>13.43</td><td>13.15</td></tr><tr><td>ch130</td><td>10.09</td><td>28.40</td><td>19.21</td><td>25.66</td><td>14.77</td><td>12.47</td><td>9.40</td><td>12.24</td></tr><tr><td>eil51</td><td>15.18</td><td>13.03</td><td>16.14</td><td>31.96</td><td>10.20</td><td>18.51</td><td>6.47</td><td>15.20</td></tr><tr><td>kroA150</td><td>13.44</td><td>20.24</td><td>19.09</td><td>26.08</td><td>15.59</td><td>24.61</td><td>11.62</td><td>9.86</td></tr><tr><td>kroB100</td><td>9.82</td><td>16.59</td><td>21.53</td><td>26.26</td><td>14.06</td><td>13.46</td><td>12.20</td><td>10.56</td></tr><tr><td>kroC100</td><td>9.08</td><td>12.94</td><td>24.25</td><td>25.76</td><td>16.22</td><td>16.85</td><td>15.88</td><td>8.16</td></tr><tr><td>lin318</td><td>13.80</td><td>18.75</td><td>24.44</td><td>25.78</td><td>14.30</td><td>23.02</td><td>16.63</td><td>10.95</td></tr><tr><td>pr226</td><td>14.17</td><td>21.44</td><td>28.02</td><td>24.65</td><td>15.50</td><td>12.44</td><td>18.02</td><td>10.00</td></tr><tr><td>pr264</td><td>11.28</td><td>11.89</td><td>34.28</td><td>17.87</td><td>24.00</td><td>21.50</td><td>16.78</td><td>10.97</td></tr><tr><td>pr299</td><td>11.23</td><td>31.42</td><td>25.05</td><td>31.42</td><td>18.24</td><td>20.95</td><td>20.63</td><td>15.06</td></tr><tr><td>rat99</td><td>9.43</td><td>22.30</td><td>21.05</td><td>21.79</td><td>14.09</td><td>21.01</td><td>12.41</td><td>9.18</td></tr><tr><td>ts225</td><td>5.67</td><td>5.38</td><td>19.93</td><td>16.82</td><td>7.71</td><td>17.55</td><td>6.56</td><td>11.33</td></tr><tr><td>fl417</td><td>15.57</td><td>12.64</td><td>25.52</td><td>32.42</td><td>22.72</td><td>24.13</td><td>19.15</td><td>10.59</td></tr><tr><td>pr439</td><td>11.16</td><td>20.08</td><td>24.67</td><td>27.36</td><td>21.36</td><td>26.83</td><td>19.25</td><td>14.52</td></tr></table>

## F.7 STATISTICAL SIGNIFICANCE

In this section, we provide a more rigorous statistical significance analysis on three representative scenarios: CVRP under the ACO framework, KP under the step-by-step constructive framework, and TSP with GLS at N = 200. We conduct 10 independent runs for CVRP-ACO and KP-Constructive, and 5 independent runs for TSP-GLS. For each setting, we report the best objective score for each run, together with their average and standard deviation. We then perform one-sided paired t-tests to compare PRS against the state-of-the-art baseline MCTS-AHD. The results in Table 13 indicate that the observed gains of PRS are statistically significant over MCTS-AHD across all three settings. Moreover, the standard deviations of PRS remain comparable to those of MCTS-AHD, suggesting that its improvements are not obtained at the cost of substantially increased instability.

Table 13: Statistical significance analysis of PRS over multiple independent runs. The reported p-values are from one-sided paired t-tests comparing PRS against MCTS-AHD. For TSP-GLS, the reported objective values denote gap (%).
<table><tr><td>Method</td><td></td><td></td><td></td><td></td><td></td><td>run1 run2 run3 run4 run5 run6 run7 </td><td></td><td>run8 run9 run10</td><td></td><td></td><td></td><td>avg</td><td>std p-value</td></tr><tr><td></td><td colspan="10">CVRP-ACO (↓)</td><td></td><td></td><td></td><td></td></tr><tr><td>MCTS-AHD 8.842 8.957 8.623 8.993 9.012 8.790 9.103 8.929 9.231 9.186 8.967 0.184 PRS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>8.5638.5888.751 8.7858.9898.9498.8358.943 9.0129.0588.847 0.1740.0331</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>KP-Constructive (↑)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PRS</td><td>MCTS-AHD40.24940.25540.26140.25740.25840.25240.25840.25440.26040.25240.2560.004</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>40.26240.25840.26340.25840.26640.25940.26240.25840.26340.26140.2610.0030.001</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MCTS-AHD 0.1850.2160.2380.2080.198</td><td></td><td></td><td></td><td></td><td></td><td>TSP-GLS (↓)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td>0.209 0.020</td><td></td><td></td></tr><tr><td>PRS</td><td></td><td></td><td></td><td></td><td>0.169 0.191 0.187 0.153 0.209</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.182 0.021 0.0439</td></tr></table>

## F.8 SEARCH STRATEGY VARIANTS

In this section, we examine whether the effectiveness of PRS is critically dependent on the use of BFS. To this end, we replace the BFS-based search in PRS with two alternative strategies, namely CoT-SC (Wang et al., 2022) and MCTS, while keeping the perturbation–rewrite operator unchanged. Table 14 shows that PRS-BFS achieves the best overall performance on both KP and CVRP. However, the margins are moderate, and both PRS-MCTS and PRS-CoT-SC remain competitive. This suggests that BFS is a strong default choice in our framework, likely because it provides a favorable balance between exploration coverage, candidate utilization, and search efficiency, but the effectiveness of PRS does not rely solely on BFS.

Table 14: Comparison of search strategy variants within the PRS framework. Results are reported as mean ± standard deviation over three independent runs. For KP, higher is better; for CVRP, lower is better. The maximum number of evaluated heuristics is fixed to $T = 1 0 0 0$ for all variants.
<table><tr><td>Variations</td><td> $\mathbf { K P } \left( n = 1 0 0 \right) \uparrow$ </td><td> $\mathbf { C V R P } \left( n = 5 0 \right) \downarrow$ </td></tr><tr><td>PRS-BFS</td><td> $\mathbf { 4 0 . 2 6 1 8 { \scriptstyle \pm 0 . 0 1 4 5 } }$ </td><td> $\mathbf { 8 . 9 5 8 3 } \pm 0 . 0 7 2 6$ </td></tr><tr><td>PRS-CoT-SC</td><td> $4 0 . 2 5 3 2 { \scriptstyle \pm 0 . 0 4 1 8 }$ </td><td> $9 . 3 1 5 9 { \scriptstyle \pm 0 . 0 5 8 6 }$ </td></tr><tr><td>PRS-MCTS</td><td> $4 0 . 2 5 8 4 { \scriptstyle \pm 0 . 0 1 7 5 }$ </td><td> $9 . 0 7 2 8 { \scriptstyle \pm 0 . 0 9 6 0 }$ </td></tr></table>

## F.9 EXPERIMENTS WITH DIFFERENT LLMS

Tables 15-16 further break down the performance of heuristics generated by different LLM backbones on OBPP and KP. We benchmark PRS on LLM models other than GPT, including a closed-source model Claude-3.7-sonnet and the open-source models, namely DeepSeek-v3 and Llama-3.1-70B-Instruct.

GPT-5-nano achieves the best performance among closed-source models on 1k\_100 and exhibits the smallest variance across both settings, while GPT-4.1-nano obtains a slightly lower gap on 5k\_500. Among open-source models, DeepSeek-v3 performs competitively and achieves the best result on the OOD larger-scale setting (5k\_500), while Llama-3.1-70B-Instruct exhibits slightly larger gaps. Overall, closed-source models tend to produce more stable heuristics on in-the-distribution instances, whereas strong open-source models can still achieve competitive performance on the OOD setting.

Table 15: Benchmarking PRS on different LLMs on OBPP (average gap (%) to the lower bound). Best result among all methods of each problem size is highlighted in bold while the best result among all LLM-based results of each scale is shaded.
<table><tr><td colspan="4">OBPP</td></tr><tr><td rowspan="2">N= Method</td><td>1k_100</td><td colspan="2">5k_500</td></tr><tr><td>Gap (%) ↓ std (%)</td><td>Gap (%) ↓</td><td>std (%)</td></tr><tr><td>Best Fit First Fit</td><td>4.77% 一</td><td>0.55%</td><td>一</td></tr><tr><td></td><td>5.02% Closed Source LLMs</td><td>0.55%</td><td>一</td></tr><tr><td></td><td>2.58% 0.723</td><td>0.32%</td><td>0.125</td></tr><tr><td>GPT-4.1-nano GPT-5-nano</td><td>1.82% 0.123</td><td>0.35%</td><td>0.117</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Claude-3.7-sonnet</td><td>2.25% 0.489</td><td>0.39%</td><td>0.178</td></tr><tr><td></td><td>Open Source LLMs</td><td></td><td></td></tr><tr><td>DeepSeek-v3</td><td>3.40% 0.456</td><td>0.23%</td><td>0.164</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>4.03% 0.361</td><td>0.40%</td><td>0.124</td></tr></table>

On KP, all evaluated LLMs are capable of generating high-performing heuristics, which substantially outperform the constructive neural baseline POMO on both N = 100 and N = 200. In particular, GPT-5-nano and Claude-3.7-sonnet consistently achieve the smallest optimality gaps across both problem sizes, while open-source models such as DeepSeek-v3 and Llama-3.1-70B-Instruct remain slightly behind. These results suggest that the proposed PRS is robust to the choice of LLM backbones, while stronger models primarily contribute to improved stability and marginal gains in solution quality.

Table 16: Benchmarking PRS on different LLMs on KP (average objective and average gap (%) to optimal). Best result among all methods of each problem size is highlighted in bold while the best result among all LLM-based results of each scale is shaded.
<table><tr><td colspan="4">KP</td></tr><tr><td rowspan="2">N= Method</td><td>N=100</td><td></td><td>N=200</td></tr><tr><td>Obj. ↑</td><td>Gap (%)↓</td><td>Obj. ↑ Gap (%) ↓</td></tr><tr><td>Optimal POMO</td><td>40.271 39.676</td><td>1.48%</td><td>57.448</td></tr><tr><td></td><td>Closed Source LLMs</td><td>57.271</td><td>0.31%</td></tr><tr><td></td><td>40.234 0.09%</td><td>57.405</td><td>0.07%</td></tr><tr><td>GPT-4.1-nano GPT-5-nano</td><td>40.262</td><td>0.02%</td><td>0.01%</td></tr><tr><td>Claude-3.7-sonnet</td><td>40.263</td><td>57.436 0.02%</td><td></td></tr><tr><td></td><td></td><td>57.435</td><td>0.02%</td></tr><tr><td></td><td>Open Source LLMs</td><td></td><td></td></tr><tr><td>DeepSeek-v3</td><td>40.232</td><td>0.10% 57.398</td><td>0.09%</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>40.233</td><td>0.09% 57.400</td><td>0.08%</td></tr></table>

## G COST ANALYSIS

This section reports the computational cost of different AHD frameworks. Specifically, for each framework, we measure (i) total wall-clock time, (ii) input and output token consumption, and (iii) the number of LLM queries. These metrics complement solution quality by characterizing the practical efficiency and cost profile of different frameworks under the same evaluation budget.

Table 17: Cost comparison across different AHD frameworks.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>OBPP-Constructiv</td><td rowspan=1 colspan=1>e KP-Construct</td><td rowspan=1 colspan=1>ive CVRP-AC</td><td rowspan=1 colspan=1>O OP-ACO</td></tr><tr><td rowspan=1 colspan=1>ReEvo</td><td rowspan=1 colspan=1>TimeInput TokensOutput Tokens#LLM Queries</td><td rowspan=1 colspan=1>8h2.94M1.02M2869</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>4h3.30M1.65M1433</td><td rowspan=1 colspan=1>5h2.82M1.68M1410</td></tr><tr><td rowspan=1 colspan=1>EoH</td><td rowspan=1 colspan=1>TimeInput TokensOutput Tokens#LLM Queries</td><td rowspan=1 colspan=1>10h2.40M0.90M2082</td><td rowspan=1 colspan=1>9h1.80M0.75M1054</td><td rowspan=1 colspan=1>6h1.83M1.44M1045</td><td rowspan=1 colspan=1>6h2.94M1.26M1063</td></tr><tr><td rowspan=1 colspan=1>MCTS-AHD</td><td rowspan=1 colspan=1>TimeInput TokensOutput Tokens#LLM Queries</td><td rowspan=1 colspan=1>26h3.09M1.05M4012</td><td rowspan=1 colspan=1>19h2.10M1.11M2010</td><td rowspan=1 colspan=1>12h3.06M2.16M2004</td><td rowspan=1 colspan=1>10h3.36M2.10M2012</td></tr><tr><td rowspan=3 colspan=1>PRS</td><td rowspan=3 colspan=1>TimeInput TokensOutput Tokens#LLM Queries</td><td rowspan=3 colspan=1>10h5.00M2.86M2706</td><td rowspan=2 colspan=1>5h3.15M</td><td rowspan=1 colspan=1>6h</td><td rowspan=2 colspan=1>5h3.12M</td></tr><tr><td rowspan=1 colspan=1>3.09M</td></tr><tr><td rowspan=1 colspan=1>1.87M1353</td><td rowspan=1 colspan=1>1.55M1353</td><td rowspan=1 colspan=1>1.60M1353</td></tr></table>

LLM queries and token consumption. PRS consistently uses fewer LLM queries than MCTS-AHD across all four tasks, requiring 2706 versus 4012 queries on OBPP, and 1353 versus 2010, 2004, and 2012 queries on KP, CVRP, and OP, respectively. This reduction is approximately 32-33% across the evaluated settings. Compared with ReEvo and EoH, however, PRS does not uniformly use fewer queries: its query cost is lower than ReEvo on the tasks where both methods are reported, but can exceed that of EoH.

Token consumption follows a different pattern. PRS incurs higher token usage than the lightweight baselines in several settings, reflecting its idea-generation and idea-guided rewrite stages. Nevertheless, its token usage is lower than that of MCTS-AHD on CVRP and OP, while its query count remains substantially lower than MCTS-AHD across all tasks.

Runtime Cost and efficiency. In terms of wall-clock time, PRS is comparable to lightweight evolutionary methods and significantly faster than search-heavy approaches such as MCTS-AHD across evaluated settings, as shown in Table 17. The total computational cost is primarily governed by the maximum number of evaluations T and the per-heuristic evaluation time. While T is consistent for all LLM-based AHD methods, the per-heuristic evaluation time of PRS remains under control, leading to a competitive total runtime among LLM-based AHD methods. This shows that PRS stands as a practical, cost-efficient AHD framework with strong exploration capability.

Runtime factors. The overall runtime is primarily dominated by heuristic evaluation rather than LLM inference, and thus scales with the complexity of the heuristics rather than the branching search structure. Heuristics with expensive operations, nested loops, or inefficient data structures can substantially increase per-evaluation runtime, even under the same evaluation budget. As discussed in Appendix F.4, PRS tends to discover heuristics with favorable runtime profiles, particularly on larger problem scales, as evidenced by the Pareto analysis of solution quality and runtime in Fig. 7.

Search granularity. PRS allocates its LLM budget through bounded BFS expansion. For each selected parent, one idea operator produces a target idea, which is then realized through multiple code rewrites with a fixed beam size. This organization concentrates several candidate realizations around a single idea-level proposal, rather than treating every LLM call as an independent search iteration. As shown in Table 17, this design yields substantially fewer total LLM queries than MCTS-AHD, while retaining a competitive end-to-end runtime.

## H SENSITIVITY ANALYSIS

We analyze the sensitivity of PRS to its main search-control hyperparameters on KP and OBPP under the step-by-step constructive framework. BFS beam size κ and tree depth d jointly control the local expansion breadth and depth; the stagnation and reheating ratio $( C / T \overset { - } { = } \Delta / \overset { - } { T } )$ determines the interval for triggering stagnation-aware restart and the subsequent operator reheating horizon; the softmax temperature $T _ { \mathrm { s e l } }$ governs the selection pressure on the frontier; and the UCB window size |H| controls the amount of recent operator feedback used for adaptive operator selection. For each setting, we report the average optimality gap and token cost in Table 18.

Table 18: Sensitivity analysis of PRS on KP and OBPP under the step-by-step constructive framework. Lower gap is better.
<table><tr><td>Modified Parameter</td><td>Value</td><td colspan="2">KP-Constructive</td><td colspan="2">OBPP-Constructive</td></tr><tr><td></td><td></td><td>Gap</td><td># Token (M)</td><td>Gap</td><td># Token (M)</td></tr><tr><td></td><td> $( \kappa = d = 3 ) , \Delta / T = 0 . 1$   $T _ { \mathrm { s e l } } = 1 . 0 , | \mathcal { H } | = 5 0$ </td><td>0.024%</td><td>4.25</td><td>0.802%</td><td>6.07</td></tr><tr><td>beam/depth</td><td> $( \kappa = d = 4 )$ </td><td>0.026%</td><td>3.89</td><td>0.835%</td><td>5.72</td></tr><tr><td>beam/depth</td><td> $( \kappa = d = 2 )$ </td><td>0.031%</td><td>4.98</td><td>0.913%</td><td>6.35</td></tr><tr><td>Stagnation ratio  $\Delta / T$ </td><td>0.05</td><td>0.027%</td><td>4.13</td><td>0.894%</td><td>6.18</td></tr><tr><td>Stagnation ratio  $\Delta / T$ </td><td>0.20</td><td>0.023%</td><td>4.40</td><td>0.836%</td><td>6.36</td></tr><tr><td>Softmax temperature  $T _ { \mathrm { s e l } }$ </td><td>0.5</td><td>0.021%</td><td>3.91</td><td>0.884%</td><td>5.86</td></tr><tr><td>Softmax temperature  $T _ { \mathrm { s e l } }$ </td><td>2.0</td><td>0.027%</td><td>4.06</td><td>0.821%</td><td>5.94</td></tr><tr><td>UCB window |H|</td><td>100</td><td>0.025%</td><td>4.29</td><td>0.793%</td><td>6.02</td></tr></table>

Beam/depth. The beam/depth setting affects the local expansion breadth and depth of the search. Following the BFS convention in ToT (Yao et al., 2023), we adopt a bounded tree search and set $\kappa = d = 3$ as the default for the BFS-based heuristic search. In Table 18, increasing $( \kappa , d )$ from 3 to 4 yields comparable performance with reduced token cost, suggesting that the default setting already provides sufficient search breadth and that further expansion does not necessarily improve exploration quality within the same evaluation budget T. By contrast, reducing them to 2 leads to a moderate degradation in solution quality together with a noticeable increase in token cost. This indicates that overly small search breadth or depth may hurt search efficiency, while the default setting $( \kappa = d = 3 )$ provides a good balance between exploration and cost.

Stagnation ratio. The stagnation ratio controls both the evaluation interval for triggering stagnationaware restart and the subsequent operator reheating horizon. Changing $\Delta / T$ from 0.1 to 0.05 or 0.2 yields only minor changes in both performance and computational cost. This suggests that the adaptive search-control mechanism is robust across a reasonable range of stagnation settings.

Softmax temperature. Varying the selection temperature leads to only limited performance variation across the two tasks. Although different temperatures can achieve slightly better results on individual settings, both $T _ { \mathrm { s e l } } = 0 . 5$ and $T _ { \mathrm { s e l } } = 2 . 0$ remain competitive with the default $T _ { \mathrm { s e l } } = 1 . 0$ . Meanwhile, token usage remains relatively stable across these settings, showing that the stochastic selection mechanism does not introduce large cost fluctuations. We therefore retain $T _ { \mathrm { s e l } } = 1 . 0$ as a moderate default to balance exploration and exploitation within each frontier in BFS.

UCB window size. Increasing the UCB reward window from $| \mathcal { H } | = 5 0$ to 100 results in only minor changes in both performance and token cost on KP and OBPP. This indicates that adaptive operator selection is not highly sensitive to the exact amount of recent operator feedback retained in the window. Overall, the sensitivity analysis confirms that PRS is robust to moderate variations in its main search-control hyperparameters.

## I EXAMPLE RESPONSE FOR MULTI-DIRECTIONAL PERTURBATIONS

This section presents representative examples showing how parent heuristics are perturbed and rewritten through complementary directional transformations in PRS, as shown in Fig. 8 and 9. For OBPP, lower scores are preferred; for OP, higher scores are better.

```python
Parent Heuristic of OP-ACO (Score: g = 13.560)
def heuristics(prize, distance, maxlen):
import numpy as np
n = len(prize)
epsilon = 1e-8
alpha = 1.0
beta = 2.0
heuristic_matrix = np.zeros((n, n))
for i in range(n):
for j in range(n):
if i == j or distance[i][j] == 0:
heuristic_matrix[i][j] = 0.0
else:
base_ratio = prize[j] / (distance[i][j] + epsilon)
scale_factor = 1 + alpha * (
maxlen / (distance[i][j] + epsilon))**beta
heuristic_matrix[i][j] = base_ratio * scale_factor
row_sums = np.sum(heuristic_matrix, axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
heuristic_matrix = heuristic_matrix / row_sums
return heuristic_matrix
```

Expansion: new reward-aware structures injected

Offspring Heuristic of OP-ACO (Score: g = 14.194)   
def heuristics(prize, distance, maxlen):   
import numpy as np   
n, epsilon = len(prize), 1e-8   
gamma = delta = 1.0   
median\_dist = np.median(distance)   
local\_density = np.zeros(n)   
for j in range(n):   
neighbors = [k for k in range(n) if distance[j][k] <= median\_dist and j != k]   
local\_density[j] = np.mean(prize[neighbors]) if neighbors else prize[j]   
max\_density = max(np.max(local\_density), 1.0)   
local\_density /= max\_density   
heuristic\_matrix = np.zeros((n, n))   
top\_prize\_sum = np.sum(prize[np.argsort(-prize)[:3]])   
total\_prize = np.sum(prize)   
LAF = 1 + delta \* (top\_prize\_sum / total\_prize)   
for i in range(n):   
for j in range(n):   
if i == j or distance[i][j] == 0:   
continue   
ratio = prize[j] / (distance[i][j] + epsilon)   
scale\_factor = 1 + (maxlen / (distance[i][j] + epsilon))\*\*2   
density\_weight = 1 + gamma \* local\_density[j]   
heuristic matrix[il[i] = ratio \* scale factor \* density weight \* LAF   
row\_sums = np.sum(heuristic\_matrix, axis=1, keepdims=True)   
row\_sums[row\_sums == 0] = 1.0   
heuristic\_matrix /= row\_sums   
heuristic\_matrix[heuristic\_matrix < 0] = 0.0   
heuristic\_matrix[\~np.isfinite(heuristic\_matrix)] = 0.0   
return heuristic\_matrix  
Figure 8: Example of an expansion operation on heuristics of OP under the ACO framework. The parent heuristic (top) primarily scores transitions using prize-to-distance preference and a route-length scaling factor. After expansion and LLM-based rewriting, the offspring (bottom) retains this backbone while introducing additional reward-aware structures, including local prize density and global high prize information. These additions enrich the heuristic’s decision logic and improve the validation score from g = 13.560 to g = 14.194 (+4.68%).

Parent Heuristic of OBPP-Constructive (Score: g = 2.170)   
def priority(item, bins\_remain\_cap):   
import numpy as np   
cap = np.array(bins\_remain\_cap, copy=True)   
epsilon = 1e-8   
max\_cap = np.max(cap)   
variance = np.var(cap) / (np.sum(cap) + epsilon)   
avg\_residual = np.mean(cap / (max\_cap + epsilon))   
if avg\_residual > 0.75:   
threshold\_factor = 1.5   
elif avg\_residual < 0.3:   
threshold\_factor = 3.0   
else:   
threshold\_factor = 2.0   
diff = cap - item   
fit\_threshold = threshold\_factor \* item   
near\_fit\_mask = diff <= fit\_threshold   
diff\_range = (np.max(diff) - np.min(diff)) / (item + epsilon)   
disparity\_factor = 1 / (diff\_range + epsilon)   
neighbor\_penalty = np.ones\_like(cap)   
for i in range(len(cap) - 1):   
gap = np.abs(cap[i] - cap[i + 1])   
if gap > 0.1 \* item:   
penalty = 1 - gap / (max\_cap + epsilon)   
neighbor\_penalty[i] \*= penalty   
neighbor\_penalty[i + 1] \*= penalty   
bonus = np.zeros\_like(cap)   
if np.sum(diff <= 1e-3) < 2:   
bonus[diff <= 1.2 \* item] = 5.0   
else:   
bonus[near\_fit\_mask] = 1.0   
scores = (1 / (diff + epsilon) + bonus) \* disparity\_factor   
scores \*= (1 - variance) \* neighbor\_penalty   
return scores

```python
Offspring Heuristic of OBPP-Constructive (Score: g = 1.846)
def priority(item, bins_remain_cap):
import numpy as np
cap = np.array(bins_remain_cap, copy=True)
epsilon = 1e-8
max_cap = np.max(cap)
avg_residual = np.mean(cap / (max_cap + epsilon))
if avg_residual > 0.75:
tight_weight = 2.0
elif avg_residual < 0.3:
tight_weight = 0.5
else:
tight_weight = 1.0
diff = cap - item
near_fit_mask = diff <= 1.5 * item
bonus = np.zeros_like(cap)
bonus[diff <= (1.2 * item if np.sum(near_fit_mask) < 2 else 1.5 * item)] = (
3.0 if np.sum(near_fit_mask) < 2 else 1.0
)
disparity_factor = 1 / (
(np.max(diff) - np.min(diff)) / (item + epsilon) + epsilon
)
neighbor_penalty = np.ones_like(cap)
for i in range(len(cap) - 1):
gap = np.abs(cap[i] - cap[i + 1])
if gap > 0.1 * item:
penalty = 1 - gap / (max_cap + epsilon)
neighbor_penalty[i:i + 2] *= penalty
scores = (1 / (diff + epsilon) + bonus) * disparity_factor
scores *= tight_weight * neighbor_penalty
return scores
```  
Figure 9: Example of a compression operation on heuristics of OBPP under the step-by-step constructive framework. The parent heuristic (top) combines adaptive fit thresholds, variance-based global weighting, disparity control, and neighborhood smoothness. After compression and LLM-based rewriting, the offspring (bottom) retains the core fit, disparity, and smoothness mechanisms while simplifying the adaptive threshold and global weighting logic. The resulting heuristic improves the validation score from g = 2.170 to g = 1.846.

## J LIMITATIONS

A current limitation of PRS is that its adaptive exploration controller still relies on predefined scheduling rules and stagnation thresholds. Although the UCB-based operator selection adapts to recent search outcomes, the overall exploration schedule and restart behavior are manually specified, which may limit the flexibility of search control across different tasks and search stages.

Another limitation is the cost of heuristic evaluation. PRS currently realizes each perturbed heuristic idea into executable code before evaluating its performance, even though not every proposed idea is necessarily worth implementation and expensive downstream evaluation. This issue can become particularly important when heuristic evaluation involves large problem instances or computationally costly optimization environments.

In future work, we aim to address these limitations by (1) developing more automatic and LLM-driven control mechanisms, where operator selection, exploration scheduling, and stagnation response are determined autonomously by the LLM agent rather than through predefined rules, and (2) introducing surrogate models, such as learned world models, to estimate the potential utility of perturbed heuristic ideas before realization. Such models could prioritize promising ideas and filter less useful ones, reducing unnecessary implementation and evaluation costs and improving the overall efficiency of heuristic search.

## K BASELINES & LICENSES

Optimal Solutions. For the results of TSP in Table 9 and Table 10, we obtain reference solutions using the Lin–Kernighan heuristic (Lin & Kernighan, 1973) implemented in the elkai solver, which is a widely used Python implementation of LK-based TSP solvers. For problem sizes up to N = 315, the solver returns proven optimal solutions, while for larger instances it provides high-quality approximate solutions. We report optimality gaps with respect to these reference solutions, following common practice in prior TSP literature. For KP instances, we use Google OR-Tools for the optimal value.

Other Baselines Implemented. We implement POMO (Kwon et al., 2020) and DeepACO (Ye et al., 2023) as representative baselines. To ensure a fair comparison, data augmentation in POMO is disabled in all experiments. For KGLS, we adopt the ACO implementation provided in DeepACO and set the heuristic matrix to the distance matrix. The licenses and URLs of baselines, models, and software are summarized in Table 19.

Table 19: External codebases and datasets used in this work.
<table><tr><td>Resource</td><td>License / Availability</td><td>URL</td></tr><tr><td>LKH3</td><td></td><td>Academic research use http://webhotel4.ruc.dk/~keld/research/LKH-3/</td></tr><tr><td>OR-Tools</td><td>MIT License</td><td>https://developers.google.com/optimization/pack/ knapsack</td></tr><tr><td>POMO</td><td>Available online</td><td>https://github.com/yd-kwon/POMO/tree/master</td></tr><tr><td>DeepACO</td><td>MIT License</td><td>https://github.com/henry-yeh/DeepACO</td></tr><tr><td>FunSearch</td><td>Apache License</td><td>https://github.com/google-deepmind/funsearch</td></tr><tr><td>EoH</td><td>MIT License</td><td>https://github.com/FeiLiu36/EoH/tree/main</td></tr><tr><td>ReEvo</td><td>MIT License</td><td>https://github.com/ai4co/reevo</td></tr><tr><td>HSEvo</td><td>Available online</td><td>https://github.com/datphamvn/HSEvo</td></tr><tr><td></td><td>MCTS-AHD MIT License</td><td>https://github.com/zz1358m/MCTS-AHD-master</td></tr><tr><td>TSPLIB</td><td>Non-commercial use</td><td>http://comopt.ifi.uni-heidelberg.de/software/ TSPLIB95</td></tr></table>