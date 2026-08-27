# SHSP: Structure-Aware Hierarchical Solution Prediction for Mixed-Integer Linear Programming

Zherong Zhang<sup>1,2∗</sup>, Guanlin Li<sup>1,2∗</sup>, Chengrui Gao<sup>1,2</sup>, Haopu Shang<sup>1,2</sup>, Ke Xue<sup>1,2</sup>, Jixiang Lu<sup>3</sup>, Weiyong Yang<sup>3</sup>, Chao Qian<sup>1,2†</sup>

<sup>1</sup>State Key Laboratory of Novel Software Technology, Nanjing University

<sup>2</sup>School of Artificial Intelligence, Nanjing University

<sup>3</sup>State Key Laboratory of Technology and Equipment for Defense Against Power System Operational Risks, Nari Technology Co., Ltd.

## Abstract

Mixed-Integer Linear Programming (MILP) is a fundamental optimization paradigm in combinatorial optimization and has been widely applied across real-world domains. Due to its NP-hard nature, obtaining optimal solutions for large-scale or highly constrained MILP instances remains computationally prohibitive. Learning-based solution prediction has therefore emerged as a promising approach to provide high-quality variable assignment for solver acceleration. However, existing methods typically adopt a one-shot prediction paradigm that predicts the marginal probabilities of all variables simultaneously. As a result, the conditional dependencies among variables are only implicitly captured through message passing, with the burden of modeling the combinatorial structure falling entirely on the representational capacity ofgraph neural networks. To address this limitation, we propose the Structure-Aware Hierarchical Solution Prediction (SHSP) framework that replaces the parallel marginal decoding of one-shot methods with a novel hierarchical conditional decoding mechanism. Specifically, SHSP constructs a variable coupling graph from the constraint structure, decodes variables sequentially along a hierarchy of increasing coupling strength, and conditions each hierarchy on previously predicted assignments. To mitigate error accumulation during the decoding process, SHSP further incorporates a confidence-aware mask-andrepair mechanism to identify and correct unreliable intermediate predictions. We integrate SHSP with multiple learningguided search methods, and evaluate it on four standard MILP benchmarks. Experimental results demonstrate that SHSP significantly outperforms existing one-shot prediction baselines, achieving a 54% average reduction in solution gap. Code is available at: https://github.com/lamda-bbo/SHSP.

## 1 Introduction

Mixed-Integer Linear Programming (MILP) is one of the most widely used modeling paradigms in combinatorial optimization and operations research, with broad applications in various optimization domains (Ma et al. 2019; Li et al. 2024b). Owing to the NP-hardness of MILP (Karp 1972), decades of research eforts have been devoted to developing powerful solvers such as SCIP (Achterberg 2009) and Gurobi (Gurobi Optimization, LLC 2026). Equipped with advanced techniques including cutting planes, primal heuristics, and presolving, these solvers are capable of eficiently handling a wide range of practical instances. However, solving large-scale and highly constrained instances remains computationally expensive, making the improvement ofMILP solving eficiency a fundamental research challenge.

In recent years, machine learning has emerged as a promising paradigm for accelerating MILP solving (Li et al. 2024a), and existing studies generally fall into two categories. The first category learns heuristic policies for key components of branch-and-bound (B&B) solvers, such as branching strategy (Khalil et al. 2016; Gasse et al. 2019; Hou et al. 2026) and cut selection (Tang, Agrawal, and Faenza 2020; Huang et al. 2021; Paulus et al. 2022; M. et al. 2026), typically via imitation learning or reinforcement learning. The second category employs machine learning models such as graph neural networks (GNNs) to directly predict high-quality solutions for MILP instances (Ding et al. 2020; Zeng et al. 2024; Geng et al. 2025). Research in this category advances along two complementary aspects. The first aspect concerns how predictions are exploited to reduce the original problem (Nair et al. 2020; Han et al. 2023; Liu et al. 2025). The second aspect focuses on improving the prediction capacity (Pu et al. 2026; Wang, Li, and Wang 2026). A detailed discussion of related work is provided in Appendix A. Together, these efforts have established learning-based solution prediction as an efective paradigm for accelerating MILP solving.

Despite the remarkable performance of existing solution prediction methods, most of them follow a one-shot prediction paradigm (Han et al. 2023; Liu et al. 2025; Pu et al. 2026), where the marginal probabilities of all decision variables are predicted simultaneously in a single forward pass. However, variables in a MILP are strongly interdependent through shared constraints. The optimal assignment of one variable often hinges on the combinatorial structure formed by the others, to the extent that exactly predicting the complete optimal assignments would amount to solving the MILP itself. Under the one-shot paradigm, such dependencies are captured only implicitly by the GNN encoder, while the decoding stage treats all variables as independent. This mismatch can yield mutually conflicting assignments that mislead, rather than accelerate, the downstream solver.

To address these limitations, we propose Structure-Aware

Hierarchical Solution Prediction for MILP (SHSP), a novel learning framework that explicitly models variable dependencies through hierarchical partitioning and prediction, rather than relying on a one-shot predictor to recover them implicitly. Specifically, SHSP first constructs a weighted variable coupling graph that quantifies pairwise dependencies based on expected constraint violation and coeficient importance. Guided by the coupling scores, variables are predicted in a hierarchical manner: weakly coupled variables are resolved first, while strongly coupled ones are subsequently decoded conditioned on the resulting partial assignments. To further enhance robustness, we introduce a confidencebased mask-and-repair mechanism that mitigates error accumulation during multi-step inference. Finally, the predicted assignments are exploited through a structure-aware fixing strategy that prioritizes strongly coupled variables, whose fixed values propagate through shared constraints and tighten the feasible domains of many others, thereby efectively reducing the search space.

As a superior predictor, SHSP serves as a drop-in replacement for the vanilla GNN-based one-shot predictor and can be seamlessly integrated into diverse learning-guided acceleration frameworks (Nair et al. 2020; Han et al. 2023). To validate the efectiveness of SHSP, we conduct extensive experiments on four widely used MILP benchmarks. Experimental results demonstrate that SHSP consistently outperforms the one-shot prediction baselines across all benchmarks, reducing the average absolute primal gap by up to 100.0%, 38.1%, 94.3%, and 42.1%, respectively. Notably, a variant of SHSP even surpasses the solution quality of fullbudget Gurobi on combinatorial auction instances within a substantially smaller time budget.

## 2 Preliminaries

## 2.1 Mixed-Integer Linear Programming

Mixed-integer linear programming (MILP) seeks to optimize a linear objective function over a feasible region defined by linear constraints, where a subset of decision variables take integer values. Formally, an MILP instance can be expressed as

$$
{ \begin{array} { r l } { \operatorname* { m i n } } & { c ^ { \top } x } \\ { { \mathrm { s . t . } } } & { A x \leq b , } \\ & { l \leq x \leq u , } \\ & { x \in \mathbb { Z } ^ { p } \times \mathbb { R } ^ { n - p } , } \end{array} }
$$

where x denotes the n-dimensional decision vector comprising p integer variables and $n \mathrm { ~ - ~ } p$ continuous variables. The vector $c \in \mathbb { R } ^ { n }$ specifies the objective coeficients, $A \in \mathbb { R } ^ { m \times n }$ is the constraint coeficient matrix, and $b \in \mathbb { R } ^ { m }$ is the corresponding right-hand-side vector. The vectors $l \in ( \mathbb { R } \cup \{ - \infty \} ) ^ { n }$ and $u ^ { \mathbf { \bar { \alpha } } } \in ( \mathbb { R } \cup \{ + \infty \} ) ^ { n }$ impose the lower and upper bounds on the variables, respectively. Following prior works (Han et al. 2023; Liu et al. 2025; Pu et al. 2026), we focus on MILP instances whose integer variables are binary, as general integer variables can be reduced to binary ones via well-established preprocessing techniques (Nair et al. 2020).

An MILP instance can be naturally encoded as a weighted bipartite graph $G = ( W \cup V , E )$ (Gasse et al. 2019), where $\dot { W }$ and $\bar { V }$ denote the sets of constraint nodes and variable nodes, respectively. An edge $( w , v ) \in E$ connects a constraint node $w \in { \dot { W } }$ and a variable node $v \in V$ if and only if the corresponding variable appears with a nonzero coeficient in the constraint. The node features typically encompass objective coeficients, variable bounds, and variable types on the variable side, as well as right-hand-side values and constraint senses on the constraint side, while the edge features carry the corresponding constraint coeficients. More details on the graph modeling are provided in Appendix B.

## 2.2 Predict-and-Search

Predict-and-Search (PaS) (Han et al. 2023) is a two-stage framework that integrates neural solution prediction with mathematical optimization for accelerating MILP solving. Given an MILP instance I, PaS first learns the underlying solution distribution from a collection of optimal or nearoptimal solutions. Specifically, the solution distribution is approximated as $\begin{array} { r } { q ( x | I ) \ = \ \frac { \exp ( - E ( x , I ) ) } { \sum _ { x ^ { \prime } \in S } \exp ( - E ( x ^ { \prime } , I ) ) } } \end{array}$ , where S denotes the collected solution set and the energy function is defined as $E ( x , I ) ~ = ~ c ^ { \top } x$ if x is feasible and +∞ otherwise. Guided by this distribution, PaS trains a graph neural network $p _ { \theta }$ to estimate the marginal probability of each binary variable. Under the variable-wise independence assumption, the joint solution distribution is factorized as $\begin{array} { r } { p _ { \theta } ( x \mid I ) = \prod _ { i = 1 } ^ { \tilde { n } } p _ { \theta } ( x _ { i } \mid I ) } \end{array}$ , thereby reducing the learning task to predicting per-variable marginals. The GNN is trained by minimizing a weighted binary cross-entropy loss over the collected solutions, where each solution is weighted according to $q ( x \mid I )$

$$
\mathcal { L } = - \sum _ { s = 1 } ^ { S } q ( x ^ { ( s ) } \mid I ) \sum _ { i = 1 } ^ { n } \ell ( \hat { p } _ { i } , y _ { s , i } )
$$

where $\hat { p } _ { i } \ = \ p _ { \theta } ( x _ { i } \ = 1 \ | \ I )$ denotes the predicted probability of variable $x _ { i } , y _ { s , i } \in \{ 0 , 1 \}$ denotes the value of variable $x _ { i }$ <sub>i</sub> in solution $x ^ { ( s ) }$ , and $\ell ( \hat { p } _ { i } , y _ { s , i } ) = - y _ { s , i } \log \hat { p } _ { i } -$ $( 1 - y _ { s , i } ) \log ( 1 - \hat { p } _ { i } )$ denotes the binary cross-entropy loss. Based on the predicted marginal probabilities, PaS selects $k _ { 1 }$ variables with the highest marginal probabilities and fixes them to one, while selecting $k _ { 0 }$ variables with the lowest marginal probabilities and fixes them to zero. The resulting partial solution is denoted as $\hat { x } ^ { \left[ P \right] }$ , where $P$ represents the index set of fixed variables.

Rather than hard-fixing these variables as in neural diving (Nair et al. 2020), PaS formulates a trust-region optimization problem that constrains the feasible region to the neighborhood of the predicted partial solution. Formally, the trust region is defined as $\bar { B _ { P } ( \hat { x } ^ { [ P ] } , \Delta ) } = \{ x ^ { [ P ] } \in \bar { \mathbb { R } } ^ { n } \mid$ $\| \hat { x } ^ { [ P ] } - \bar { x } ^ { [ P ] } \| _ { 1 } \leq \Delta \}$ , where $\Delta$ denotes the neighborhood radius. The resulting trust-region problem is formulated as

$$
\begin{array} { r l } { \underset { x } { \operatorname* { m i n } } } & { c ^ { \top } x } \\ { \mathrm { s . t . } } & { A x \leq b , \quad l \leq x \leq u , } \\ & { x ^ { \left[ P \right] } \in B _ { P } ( \hat { x } ^ { \left[ P \right] } , \Delta ) , \quad x \in \mathbb { Z } ^ { p } \times \mathbb { R } ^ { n - p } . } \end{array}
$$

By coupling neural prediction with trust-region-based local optimization, PaS efectively combines the eficiency of learning-based methods and the reliability of mathematical programming solvers. Notably, when $\Delta \stackrel { \cdot } { = } 0$ , PaS degenerates to the hard-fixing scheme of neural diving.

## 3 Methodology

In this section, we present Structure-Aware Hierarchica Solution Prediction for MILP (SHSP). In contrast to existing methods that predict the assignments of all decision variables simultaneously, SHSP explicitly exploits variable coupling information and decodes variable assignments sequentially following a hierarchy of increasing coupling strength, where the prediction at each decoding step is conditioned on the partial assignments of lower hierarchies obtained in preceding steps. The overall framework is illustrated in Figure 1. In the remainder of this section, we first introduce the variable coupling graph, then describe the structure-aware hierarchical predictor, and finally present the corresponding variable fixing strategy.

## 3.1 Variable Coupling Graph

Decision variables are often tightly coupled through shared constraints, and such structural dependencies play a critical role in obtaining high-quality solutions. To explicitly capture these dependencies, we construct a variable coupling graph, which serves as the basis for estimating coupling scores and for determining hierarchical partitioning.

Graph Modeling. We model the coupling structure among variables as an undirected weighted graph, where nodes correspond to variables and edges capture their co-occurrence in constraints. Given linear constraints, we consider all variables when estimating coupling strengths, but retain only those that co-occur with at least one binary variable in some constraints, i.e., those directly connected to binary variables. This makes the coupling graph considerably sparser without severely compromising the prediction accuracy of binary variables. An edge is created between two retained variables if they appear together in at least one constraint, with its weight quantifying the coupling strength implied by the shared constraints, as detailed below.

Edge Weights Computation. For each retained variable pair, we assign an edge weight that reflects the coupling strength implied by the constraints. The weight is determined by two complementary heuristic factors. The first factor, termed the expected violation, measures how strongly a variable pair is restricted by a constraint. Specifically, we treat the two variables as random variables, where binary variables follow independent uniform distributions over {0, 1} and continuous variables follow uniform distributions over their bounds. We then compute the expected violation of the constraint induced by the pair alone. A larger expected violation indicates that the pair is more tightly restricted by the constraint and thus more strongly coupled. The second factor, termed the coeficient importance, characterizes the relative contribution of a variable pair within a constraint. It is computed as the sum of the magnitudes of the two corresponding coeficients, each scaled by the range of its associated variable. To eliminate scale discrepancies across constraints, both factors are normalized by their respective average values over all variable pairs within the same constraint. Note that the above two factors are both computed over all variables, including purely continuous ones.

The local coupling weight of a pair in constraint c is then given by the product of the two normalized factors, and the final edge weight $W _ { i j }$ between variables $z _ { i }$ and $z _ { j }$ is obtained by summing the local weights over all constraints containing both variables. The detailed computation of edge weights is provided in Appendix C. The resulting weighted graph captures the overall structural coupling among variables and serves as the basis for computing the variable coupling score and determining the hierarchical prediction order in structure-aware hierarchical prediction.

## 3.2 Structure-Aware Hierarchical Prediction

The above weighted variable coupling graph provides a quantitative measure of the structural dependency among variables. Building upon this, we design a structure-aware hierarchical prediction framework that decodes variables sequentially from weakly coupled to strongly coupled ones. This framework conditions each decoding step on the assignments fixed in preceding hierarchies, so that strongly coupled variables, whose values are the most sensitive to those of others, are predicted with explicit structural context rather than in isolation as in previous one-shot methods.

Coupling Score-based Hierarchical Partitioning. For each binary variable $z _ { i } ,$ we define its variable coupling score (VCS) as the sum of the weights of its incident edges in the coupling graph, which quantifies the overall strength of its structural dependency on the remaining variables. All variables are sorted in ascending order of VCS and evenly partitioned into K hierarchies $\left\{ H _ { 1 } , H _ { 2 } , \ldots , H _ { K } \right\}$ , such that $H _ { 1 }$ contains the most weakly coupled variables and $H _ { K }$ the most strongly coupled ones. The decoding process then follows this weak-to-strong order. Weakly coupled variables are largely insensitive to the assignments of other variables and can thus be predicted reliably at early stages. Their fixed values in turn serve as conditioning information that anchors the prediction of strongly coupled variables in later hierarchies.

Hierarchical Decoding with Mask-and-Repair. The model decodes the K hierarchies sequentially, and each step is conditioned on the tentative assignments accumulated from all preceding hierarchies. To realize this conditioning, the input feature of every variable node is augmented with a decoding-state triplet $( m _ { i } , \hat { x } _ { i } , c _ { i } )$ , where $m _ { i } \in \{ 0 , 1 \}$ indicates whether variable i should be predicted in the current step, $\hat { x } _ { i }$ denotes its tentative binary assignment, and $c _ { i }$ its associated confidence. For variables that will be decoded in future steps, the triplet is set to a zero placeholder. In addition, to make the network aware of its position in the decoding process, the current step index is encoded via positional encoding (Vaswani et al. 2017), and the resulting step embedding is concatenated with the initial node embeddings and passed through a fusion layer to produce the step-aware node representations. The collection of these state features constitutes the conditioning context $\mathbf { h } _ { < k }$ , and the prediction of the k-th hierarchy is formulated as

![](images/427ee70413c3d98b32901ec6125cee4afbd59fbafe9a46d1f38d5a0bb6177d6d.jpg)  
Figure 1: Overview of the proposed method. Given an original MILP instance, we first construct a variable coupling graph to capture the structural dependencies among decision variables. Based on variable coupling graph, variables are ranked according to their coupling score and divided into multiple hierarchies. Our SHSP sequentially predicts variables following the weak-tostrong hierarchy, with a mask-and-repair mechanism to mitigate error accumulation. Finally, the predicted variable assignments are combined with a structure-aware variable fixing strategy and applied to downstream learning-guided search frameworks such as Predict-and-Search (Han et al. 2023).

$$
\mathbf { p } _ { H _ { k } } = f _ { \theta } ( G , H _ { k } , \mathbf { h } _ { < k } ) ,
$$

where G denotes the bipartite graph and $f _ { \theta }$ the graph neural network. For each variable $i \in H _ { k }$ , the network outputs a probability $p _ { i }$ , from which the binary assignment $\hat { x } _ { i } =$ $\mathbb { I } ( \bar { p } _ { i } > 0 . 5 )$ and the confidence $c _ { i } = 2 | p _ { i } - 0 . 5 |$ are derived.

Since the conditioning context is constructed from model predictions rather than ground-truth assignments, wrong early decisions may propagate and accumulate across hierarchies. To mitigate such error accumulation, we propose a mask-and-repair mechanism. Before being incorporated into the conditioning context, each newly predicted variable is stochastically masked with a probability $p _ { \mathrm { m a s k } } ( i ) \propto 1 - c _ { i }$ that decreases with its confidence. Consequently, highconfidence predictions are likely to be retained as conditioning information, while uncertain predictions tend to be masked, reverting to the undecided placeholder state and being collected into a repair set R. After all K hierarchies have been decoded, a final repair step decodes the variables in R conditioned on the full context of retained assignments, i.e., $\mathbf { p } _ { \mathcal { R } } = f _ { \theta } \left( G , \mathcal { R } , \mathbf { h } _ { \mathcal { V } \backslash \mathcal { R } } \right)$ , and the repaired predictions replace the masked ones to yield the final solution. This iterative masking-and-prediction paradigm shares a similar spirit with the recent work Apollo-MILP (Liu et al. 2025). However, our method difers in two key aspects: (1) it explicitly exploits the variable coupling structure to establish a hierarchy that guides the decoding process; (2) it constitutes an inherently new prediction paradigm that operates entirely within the prediction stage, rather than following Apollo-MILP’s alternation between prediction and solver invocation.

Training Objective. The framework is trained by minimizing a weighted prediction loss aggregated over all hierarchies and the repair step. Given S supervised solutions with objective values $\lbrace o _ { s } \rbrace _ { s = 1 } ^ { S }$ , each solution is assigned a weight $\begin{array} { r } { w _ { s } = \exp ( - o _ { s } / \tau ) / \sum _ { i = 1 } ^ { S } \exp ( - o _ { j } / \tau ) } \end{array}$ , where higher-quality solutions contribute more to the supervision and τ controls the weighting smoothness. The overall training objective is defined as

$$
\begin{array} { l } { \displaystyle \mathcal { L } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sum _ { s = 1 } ^ { S } w _ { s } \sum _ { i \in H _ { k } } \ell \Big ( p _ { i } ^ { ( k ) } , y _ { s , i } \Big ) } \\ { \displaystyle \qquad + \sum _ { s = 1 } ^ { S } w _ { s } \sum _ { i \in \mathcal { R } } \ell \Big ( p _ { i } ^ { ( \mathrm { r e p } ) } , y _ { s , i } \Big ) , } \end{array}
$$

where $p _ { i } ^ { ( k ) }$ and $p _ { i } ^ { ( \mathrm { r e p } ) }$ denote the predicted probabilities of variable i in the k-th hierarchy and the repair step, respectively, $y _ { s , i }$ denotes the label of variable i in solution s, and $\ell ( \cdot , \cdot )$ is the binary cross-entropy loss. Note that when computing the k-th prediction, the preceding k − 1 states are decoded by the model itself, which aligns with the inference procedure and thus mitigates the distributional shift between training and inference. However, fully relying on model-decoded states can hinder convergence, as early predictions are highly inaccurate. We therefore incorporate a teacher forcing strategy that employs ground-truth assignments as conditioning information in the early stage of training, and the proportion of teacher forcing is progressively decayed as training proceeds (See Appendix E.2).

## 3.3 Structure-Aware Variable Fixing Strategy

Existing variable fixing strategies typically select variables solely according to their prediction confidences. However, fixing variables with similar confidence levels may have substantially diferent structural influences on the reduced MILP problem, so treating them uniformly overlooks the valuable structural information.

Motivated by this observation, we propose a structureaware variable fixing strategy that jointly considers the variable coupling structure and the prediction confidences. Specifically, we first identify variables whose variable coupling scores exceed a prescribed threshold as high-coupling variables. Fixing such variables tends to tighten the feasible domains of many others through shared constraints, and thus efectively reduces the search space. The identified variables are subsequently screened according to their prediction confidences, where variables with $p _ { i } \geq \theta _ { 1 }$ are fixed to one, and those with $p _ { i } \leq \theta _ { 0 }$ are fixed to zero. This two-stage screening guarantees that only strongly coupled variables with highly reliable predictions are eligible for fixing.

To align with existing methods and ensure a fair comparison, we further cap the number of fixed variables: the variables selected above are ranked by their prediction probabilities, and at most $k _ { 1 }$ and $k _ { 0 }$ of them are retained for fixing to one and to zero, respectively. Note that $k _ { 1 }$ and $k _ { 0 }$ serve only as upper bounds. If the thresholds yield fewer variables, no additional ones are fixed to reach these bounds. In contrast to confidence-only fixing, the proposed strategy prioritizes variables that are both structurally influential and confidently predicted, thereby enabling downstream optimization frameworks to explicitly exploit structural information for more efective search-space reduction.

## 4 Experiments

## 4.1 Experimental Setup

Datasets We evaluate our framework on four widely adopted MILP benchmarks in this field: Combinatorial Auctions (CA) (Leyton-Brown, Pearson, and Shoham 2000), Workload Appointment (WA) (Gasse et al. 2022), Item Placement (IP) (Gasse et al. 2022), and Set Covering (SC) (Balas and Ho 1980). Following existing studies (Gasse et al. 2019; Han et al. 2023; Liu et al. 2025), we generate SC and CA instances using similar procedures, while the IP and WA benchmarks are obtained from two real-world problems used in NeurIPS ML4CO 2021 competition (Gasse et al.

2022). For each benchmark problem, we use 240 instances for training, 60 instances for validation, and 100 instances for testing. Additional details are provided in Appendix D.

Baselines We compare our method with both traditional MILP solvers and learning-based solution prediction approaches. For traditional optimization methods, we compare our method with Gurobi (Gurobi Optimization, LLC 2026) and SCIP (Achterberg 2009). For learning-based approaches, we consider three representative methods: Neural Diving (ND) (Nair et al. 2020), Predict-and-Search (PaS) (Han et al. 2023), and Apollo-MILP (Liu et al. 2025).

Metrics. We compare the best objective values (OBJ) achieved by diferent methods within a 1000-second time limit on each test instance. Following the evaluation protocol in Predict-and-Search (Han et al. 2023), we use the best solution obtained by a single-threaded Gurobi solver with a 3600- second time limit as the best-known solution (BKS), which serves as an approximation of the optimal value. The absolute primal gap is then calculated as $\mathrm { G a p } _ { a b s } = | \mathrm { O B J - B K S } |$ measuring the distance between the obtained solution and the BKS. It is worth noting that our method achieves better solutions than the Gurobi solver with a 3600-second time limit on certain benchmark problems. In such cases, the objective values found by our method are used to update the BKS.

## 4.2 Main Evaluation

Solving Performance To evaluate the efectiveness of our method, we replace the original GNN predictor and the variable fixing strategy in each framework with our SHSP predictor and structure-aware variable fixing strategy, forming ND+SHSP, PaS+SHSP, and Apollo+SHSP, respectively. We compare the solving performance between our framework and the baselines under a time limit of 1,000 seconds. Although SHSP performs hierarchical multi-step prediction, its additional neural network inference time is less than 0.2 seconds across all benchmark datasets. The graph construction time ranges from 0.26 to 15.23 seconds and is included in the 1,000-second time budget for a fair comparison. Table 1 presents the average objective values and the corresponding average absolute primal gaps across four benchmark datasets, with Gurobi employed as the downstream solver. Similar results obtained with the SCIP solver are reported in Table 2. Detailed results of graph construction time and neural network inference time are provided in Appendix F.3.

Overall, SHSP consistently improves performance across all three learning-guided search frameworks compared with their original one-shot prediction baselines. With PaS, SHSP reduces the absolute primal gap by 99.7% on CA, nearly closing the gap entirely, along with reductions of 38.1%, 24.3%, and 42.1% on WA, IP, and SC, respectively. With Apollo, the gains are similarly substantial: the primal gap is completely eliminated on CA (100.0% reduction) and reduced by 92.9% on IP, with further reductions of 28.6% and 41.7% on WA and SC. When integrated with ND, SHSP also delivers solid improvements, reducing the absolute primal gap by 94.4% on IP and 67.0% on CA, and by 19.2% on WA. Remarkable improvements are also observed when using SCIP as the downstream solver. As shown in Table 2,

![](images/9119b2e6435bb9ccb7351c7ae2e968e81a426b2e1fd69c92d32f4525ed3183e1.jpg)

<table><tr><td rowspan="2">Method</td><td colspan="2">CA ↑ (BKS:98489.24)</td><td colspan="2">WA↓(BKS:706.86)</td><td colspan="2">IP↓(BKS:11.69)</td><td colspan="2">SC↓(BKS:123.30)</td></tr><tr><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td></tr><tr><td>Gurobi (3600s)</td><td>98453.42</td><td>35.82</td><td>706.86</td><td>0.00</td><td>11.69</td><td>0.00</td><td>123.30</td><td>0.00</td></tr><tr><td>Gurobi (1000s)</td><td>97407.51</td><td>1081.73</td><td>707.07</td><td>0.21</td><td>12.16</td><td>0.47</td><td>123.54</td><td>0.24</td></tr><tr><td>ND</td><td>94340.63</td><td>4148.61</td><td>707.12</td><td>0.26</td><td>14.18</td><td>2.49</td><td>123.51</td><td>0.21</td></tr><tr><td>ND+SHSP</td><td>97121.05</td><td>1368.19</td><td>707.07</td><td>0.21</td><td>11.83</td><td>0.14</td><td>123.51</td><td>0.21</td></tr><tr><td>Improvement</td><td>-</td><td>67.02%</td><td>、</td><td>19.23%</td><td>-</td><td>94.38%</td><td>-</td><td>0.00%</td></tr><tr><td>PaS</td><td>97891.74</td><td>597.50</td><td>707.07</td><td>0.21</td><td>12.06</td><td>0.37</td><td>123.49</td><td>0.19</td></tr><tr><td>PaS+SHSP</td><td>98487.72</td><td>1.52</td><td>706.99</td><td>0.13</td><td>11.97</td><td>0.28</td><td>123.41</td><td>0.11</td></tr><tr><td>Improvement</td><td></td><td>99.75%</td><td>一</td><td>38.1%</td><td>-</td><td>24.32%</td><td></td><td>42.11%</td></tr><tr><td>Apollo</td><td>97997.74</td><td>491.50</td><td>707.07</td><td>0.21</td><td>11.97</td><td>0.28</td><td>123.42</td><td>0.12</td></tr><tr><td>Apollo+SHSP</td><td>98489.24</td><td>0.00</td><td>707.01</td><td>0.15</td><td>11.71</td><td>0.02</td><td>123.37</td><td>0.07</td></tr><tr><td>Improvement</td><td>-</td><td>100.00%</td><td>-</td><td>28.6%</td><td>-</td><td>92.86%</td><td>I</td><td>41.67%</td></tr></table>

Table 1: Performance comparison on four benchmark datasets (CA, WA, IP, and SC) using Gurobi under a 1000-second time limit. ↑ indicates that higher values are better, while ↓ indicates that lower values are better. Bold values denote the better results. Improvement denotes the relative reduction in $\mathrm { G a p } _ { \mathrm { a b s } }$ relative to the corresponding downstream solution-prediction framework.  
Gurobi ND+GCN PS+GCN Apollo+GCN ND+SHSP PS+SHSP Apollo+SHSP

Figure 2: The convergence curves of primal gaps as the solving process proceeds. All methods are implemented using Gurobi with a total time budget of 1,000 seconds. For a fair comparison, the graph construction time of SHSP is included in this budget. Results are averaged over 100 testing instances. A lower primal gap indicates faster convergence and better solving performance.

SHSP consistently reduces the absolute primal gap across all three frameworks on all four benchmarks, demonstrating that the proposed method generalizes well across diferent downstream solvers. It is worth noting that Apollo+SHSP reaches the best known solution, surpassing the solution obtained by Gurobi with a 3,600-second time limit on the CA benchmark. This observation suggests that our method can provide a stronger initialization for the downstream solver, not only accelerating the search process but also guiding it toward higher-quality solutions.

Primal Gap as a Function of Runtime Figure 2 presents   
the curves of the average primal gap, defined as ${ \mathrm { G a p } } _ { \mathrm { r e l } } =$   
|OBJ−BKS| , throughout the solving process. Overall, SHSP-|BKS|   
based methods consistently exhibit faster convergence and   
achieves lower primal gaps on the benchmarks than cor  
responding one-shot prediction methods. This behavior is   
attributed to more accurate solution predictions and more   
efective search-space reduction of our method.

## 4.3 Ablation Study

Confidence-Based Mask and Repair Mechanism To evaluate the efectiveness of the proposed mask-and-repair mechanism, we compare three settings: without mask and repair (None), with mask stage only (Mask), and with both mask and repair (Mask+Repair). As shown in Table 3, applying the masking mechanism alone does not consistently improve performance and may even cause slight degradation, since masked variables are simply discarded without correction, resulting in a loss of predictive information. In contrast, coupling masking with the proposed repair strategy consistently yields the best results across all datasets and both prediction frameworks, confirming that the two components are complementary and jointly contribute to the overall performance. More results are provided in Appendix F.1.

Hierarchical Solution Prediction Predictor To evaluate the efectiveness of the proposed hierarchical solution prediction predictor, we compare it with the original GNN predictor under the same structure-aware fixing strategy. As shown in

<table><tr><td rowspan="2">Method</td><td colspan="2">CA ↑ (BKS:98489.24)</td><td colspan="2">WA↓(BKS:706.86)</td><td colspan="2">IP↓(BKS:11.69)</td><td colspan="2">SC↓ (BKS:123.30)</td></tr><tr><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td><td>Obj</td><td> ${ \mathrm { G a p } } _ { a b s }$ </td><td>Obj</td><td> ${ \mathrm { G a p } } _ { a b s }$ </td></tr><tr><td>SCIP (3600s)</td><td>97795.78</td><td>693.46</td><td>708.21</td><td>1.35</td><td>19.06</td><td>7.37</td><td>124.58</td><td>1.28</td></tr><tr><td>SCIP (1000s)</td><td>96932.37</td><td>1556.87</td><td>709.62</td><td>2.76</td><td>22.69</td><td>11.00</td><td>125.81</td><td>2.51</td></tr><tr><td>ND</td><td>94338.51</td><td>4150.73</td><td>707.83</td><td>0.97</td><td>16.33</td><td>4.64</td><td>125.87</td><td>2.57</td></tr><tr><td>ND+SHSP</td><td>97267.46</td><td>1221.78</td><td>707.18</td><td>0.32</td><td>13.70</td><td>2.01</td><td>125.84</td><td>2.54</td></tr><tr><td>Improvement</td><td>-</td><td>70.57%</td><td>-</td><td>67.01%</td><td>-</td><td>56.68%</td><td>-</td><td>1.17%</td></tr><tr><td>PaS</td><td>97223.78</td><td>1265.46</td><td>708.27</td><td>1.41</td><td>19.61</td><td>7.92</td><td>125.82</td><td>2.52</td></tr><tr><td>PaS+SHSP</td><td>97230.19</td><td>1259.05</td><td>708.21</td><td>1.35</td><td>19.44</td><td>7.75</td><td>125.33</td><td>2.03</td></tr><tr><td>Improvement</td><td>-</td><td>0.51%</td><td>-</td><td>4.26%</td><td>-</td><td>2.15%</td><td>-</td><td>19.44%</td></tr><tr><td>Apollo</td><td>96270.42</td><td>2218.82</td><td>708.42</td><td>1.56</td><td>20.48</td><td>8.79</td><td>125.74</td><td>2.44</td></tr><tr><td>Apollo+SHSP</td><td>97192.31</td><td>1296.93</td><td>708.21</td><td>1.35</td><td>18.56</td><td>6.87</td><td>125.45</td><td>2.15</td></tr><tr><td>Improvement</td><td>-</td><td>41.55%</td><td>-</td><td>13.46%</td><td>-</td><td>21.84%</td><td>-</td><td>11.89%</td></tr></table>

Table 2: Performance comparison on four benchmark datasets (CA, WA, IP, and SC) using SCIP under a 1000-second time limit. ↑ indicates that higher values are better, while ↓ indicates that lower values are better. Bold values denote the better results. Improvement denotes the relative reduction in $\mathrm { G a p } _ { \mathrm { a b s } }$ relative to the corresponding downstream solution-prediction framework.

<table><tr><td>Method</td><td>CA↑</td><td>IP↓</td></tr><tr><td>PaS+SHSP (None)</td><td>98380.92</td><td>12.12</td></tr><tr><td>PaS+SHSP (Mask)</td><td>98416.55</td><td>12.17</td></tr><tr><td>PaS+SHSP (Mask+Repair)</td><td>98487.72</td><td>11.97</td></tr><tr><td>Apollo+SHSP (None)</td><td>98385.20</td><td>11.78</td></tr><tr><td>Apollo+SHSP (Mask)</td><td>98344.07</td><td>11.80</td></tr><tr><td>Apollo+SHSP (Mask+Repair)</td><td>98489.24</td><td>11.71</td></tr></table>

Table 3: Ablation study of the proposed confidence-based mask-and-repair strategy on the CA and IP datasets. Bold values denote the better results.

Table 4, replacing original GNN predictor with hierarchical solution prediction predictor consistently improves the solution quality on both the PaS and Apollo frameworks. The improvement suggests that our predictor generates more accurate solution predictions, thereby enabling more efective variable fixing and subsequent problem reduction. Results on other problems are provided in Appendix F.1.

Structure-Aware Fixing Strategies To evaluate the efectiveness of the proposed structure-aware fixing strategy, we compare it with the original fixing strategy while using the same predictor. As shown in Table 4, the proposed structureaware fixing strategy consistently outperforms the normal fixing strategy on both the PaS and Apollo frameworks. The improvement suggests that prioritizing structurally important variables leads to more efective search space reduction, allowing the solver to explore the remaining search space more eficiently and ultimately achieve better solving performance. Results on other problems are provided in Appendix F.1.

<table><tr><td>Method (Fixing Strategy)</td><td>CA↑</td><td> $\mathrm { I P } \downarrow$ </td></tr><tr><td>PaS (Structure-Aware) PaS+SHSP (Original) PaS+SHSP (Structure-Aware)</td><td>97184.33 98061.51</td><td>12.12 12.34 11.97</td></tr><tr><td>Apollo (Structure-Aware)</td><td>98487.72 98015.32</td><td>11.87</td></tr><tr><td>Apollo+SHSP (Original)</td><td>98070.43</td><td>12.08</td></tr><tr><td>Apollo+SHSP (Structure-Aware)</td><td>98489.24</td><td>11.71</td></tr></table>

Table 4: Ablation study of the proposed hierarchical solution prediction and structure-aware fixing strategy on the CA and IP datasets. Bold values denote the better results.

## 5 Conclusion

In this paper, we propose SHSP, a structure-aware hierarchical solution prediction framework for MILP solving. SHSP replaces the parallel marginal decoding with a hierarchical conditional decoding mechanism guided by a variable coupling graph, decoding variables from weakly to strongly coupled ones while conditioning each step on previously predicted assignments. It further incorporates a mask-and-repair mechanism to mitigate error accumulation, and a structureaware fixing strategy to enable more efective search-space reduction in downstream frameworks. Extensive experiments on four standard benchmarks show that SHSP consistently outperforms one-shot baselines across three frameworks and two downstream solvers, achieving a 54% average reduction in the absolute primal gap with negligible inference overhead. In future work, we plan to learn coupling scores in an end-to-end manner and extend to broader problem classes.

## References

Achterberg, T. 2009. SCIP: Solving constraint integer programs. Mathematical Programming Computation, 1(1): 1– 41.

Balas, E.; and Ho, A. 1980. Set Covering Algorithms Using Cutting Planes, Heuristics, and Subgradient Optimization: A Computational Study. Springer.

Chmiela, A.; Khalil, E.; Gleixner, A.; Lodi, A.; and Pokutta, S. 2021. Learning to Schedule Heuristics in Branch and Bound. In Advances in Neural Information Processing Systems 34, 24235–24246. Virtual.

Ding, J.-Y.; Zhang, C.; Shen, L.; Li, S.; Wang, B.; Xu, Y.; and Song, L. 2020. Accelerating Primal Solution Findings for Mixed Integer Programs Based on Solution Prediction. In Proceedings of the 34th AAAI Conference on Artificial Intelligence, 1452–1459. New York, NY.

Gasse, M.; Bowly, S.; Cappart, Q.; Charfreitag, J.; Charlin, L.; Chetelat, D.; Chmiela, A.; Dumouchelle, J.; Gleixner, A.; Kazachkov, A. M.; Khalil, E.; Lichocki, P.; Lodi, A.; Lubin, M.; Maddison, C. J.; Christopher, M.; Papageorgiou, D. J.; Parjadis, A.; Pokutta, S.; Prouvost, A.; Scavuzzo, L.; Zarpellon, G.; Yang, L.; Lai, S.; Wang, A.; Luo, X.; Zhou, X.; Huang, H.; Shao, S.; Zhu, Y.; Zhang, D.; Quan, T.; Cao, Z.; Xu, Y.; Huang, Z.; Zhou, S.; Chen, B.; He, M.; Hao, H.; Zhang, Z.; An, Z.; and Mao, K. 2022. The Machine Learning for Combinatorial Optimization Competition (ML4CO): Results and Insights. In Proceedings of the NeurIPS 2021 Competitions and Demonstrations Track, 220–231. Virtual.

Gasse, M.; Chetelat, D.; Ferroni, N.; Charlin, L.; and Lodi, A. 2019. Exact combinatorial optimization with graph convolutional neural networks. In Advances in Neural Information Processing Systems 33, 15554–15566. Vancouver, Canada.

Geng, Z.; Wang, J.; Li, X.; Zhu, F.; Hao, J.; Li, B.; and Wu, F. 2025. Diferentiable Integer Linear Programming. In Proceedings of the 13th International Conference on Learning Representations. Singapore.

Gleixner, A.; Hendel, G.; Gamrath, G.; Achterberg, T.; Bastubbe, M.; Berthold, T.; Christophel, P.; Jarck, K.; Koch, T.; Linderoth, J.; et al. 2021. MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library. Mathematical Programming Computation, 13(3): 443–490.

Gupta, P.; Gasse, M.; Khalil, E.; Mudigonda, P.; Lodi, A.; and Bengio, Y. 2020. Hybrid Models for Learning to Branch. In Advances in Neural Information Processing Systems 34, 18087–18097. Virtual.

Gupta, P.; Khalil, E. B.; Chetelat, D.; Gasse, M.; Bengio, Y.; Lodi, A.; and Kumar, M. P. 2022. Lookback for Learning to Branch. arXiv:2206.14987.

Gurobi Optimization, LLC. 2026. Gurobi Optimizer Reference Manual.

Han, Q.; Yang, L.; Chen, Q.; Zhou, X.; Zhang, D.; Wang, A.; Sun, R.; and Luo, X. 2023. A GNN-guided predict-andsearch framework for mixed-integer linear programming. In Proceedings of the 11th International Conference on Learning Representations. Kigali, Rwanda.

Hou, Z.; Li, X.; Zhang, Y.; Li, T.; and You, K. 2026. LLM4Branch: Large Language Model for Discovering Eficient Branching Policies of Integer Programs. arXiv:2605.10401.

Huang, T.; Ferber, A.; Zharmagambetov, A.; Tian, Y.; and Dilkina, B. 2024. Contrastive Predict-and-Search for Mixed Integer Linear Programs. In Proceedings of the 41st International Conference on Machine Learning, 19406–19424. Vienna, Austria.

Huang, Z.; Wang, K.; Liu, F.; Zhen, H.-L.; Zhang, W.; Yuan, M.; Hao, J.; Yu, Y.; and Wang, J. 2021. Learning to Select Cuts for Eficient Mixed-Integer Programming. arXiv:2105.13645.

Karp, R. M. 1972. Complexity of Computer Computations. Plenum Press.

Khalil, E. B.; Dilkina, B.; Nemhauser, G. L.; Ahmed, S.; and Shao, Y. 2017. Learning to Run Heuristics in Tree Search. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, 659–666. Melbourne, Australia.

Khalil, E. B.; Le Bodic, P.; Song, L.; Nemhauser, G.; and Dilkina, B. 2016. Learning to branch in mixed integer programming. In Proceedings ofthe 30th AAAI Conference on Artificial Intelligence, 724–731. Phoenix, AZ.

Kuang, Y.; Wang, J.; Liu, H.; Zhu, F.; Li, X.; Zeng, J.; Hao, J.; Li, B.; and Wu, F. 2024. Rethinking Branching on Exact Combinatorial Optimization Solver: The First Deep Symbolic Discovery Framework. In Proceedings of the 12th International Conference on Learning Representations. Vienna, Austria.

Leyton-Brown, K.; Pearson, M.; and Shoham, Y. 2000. Towards a Universal Test Suite for Combinatorial Auction Algorithms. In Proceedings of the 2nd ACM Conference on Electronic Commerce, 66–76. Minneapolis, MN.

Li, X.; Zhu, F.; Zhen, H.-L.; Luo, W.; Lu, M.; Huang, Y.; Fan,Z.; Zhou, Z.; Kuang, Y.; Wang, Z.; Geng, Z.; Li, Y.; Liu, H.;An, Z.; Yang, M.; Li, J.; Wang, J.; Yan, J.; Sun, D.; Zhong,T.; Zhang, Y.; Zeng, J.; Yuan, M.; Hao, J.; Yao, J.; and Mao,K. 2024a. Machine Learning Insides OptVerse AI Solver:Design Principles and Applications. arXiv:2401.05960.

Li, Y.; Wang, W.; Xu, W.; Deng, Y.; and Wu, W. 2024b. Factor graph neural network meets max-sum: A real-time route planning algorithm for massive-scale trips. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems, 1165–1173. Auckland, New Zealand.

Lin, J.; Xu, M.; Xiong, Z.; and Wang, H. 2024. CAMBranch: Contrastive Learning with Augmented MILPs for Branching. In Proceedings ofthe 12th International Conference on Learning Representations. Vienna, Austria.

Ling, H.; Wang, Z.; and Wang, J. 2024. Learning to Stop Cut Generation for Eficient Mixed-Integer Linear Programming. In Proceedings of the 38th AAAI Conference on Artificial Intelligence, 20759–20767. Vancouver, Canada.

Liu, H.; Wang, J.; Geng, Z.; Li, X.; Zong, Y.; Zhu, F.; Hao, J.; and Wu, F. 2025. Apollo-MILP: An alternating predictioncorrection neural solving framework for mixed-integer linear

programming. In Proceedings ofthe 13th International Conference on Learning Representations. Singapore.

M., A.; Tandon, R.; Gupta, A.; KODAMANA, H.; and Ramteke, M. 2026. MIRACLE: Model-free Imitation and Reinforcement Learning for Adaptive Cut-Selection. In Proceedings of the 14th International Conference on Learning Representations. Rio de Janeiro, Brazil.

Ma, K.; Xiao, L.; Zhang, J.; and Li, T. 2019. Accelerating an FPGA-based SAT solver by software and hardware codesign. Chinese Journal ofElectronics, 28(5): 953–961.

Nair, V.; Bartunov, S.; Gimeno, F.; von Glehn, I.; Lichocki, P.; Lobov, I.; O’Donoghue, B.; Sonnerat, N.; Tjandraatmadja, C.; Wang, P.; et al. 2020. Solving Mixed Integer Programs Using Neural Networks. arXiv:2012.13349.

Paulus, M. B.; and Krause, A. 2023. Learning to Dive in Branch and Bound. In Advances in Neural Information Processing Systems 36, 34260–34277. New Orleans, LA.

Paulus, M. B.; Zarpellon, G.; Krause, A.; Charlin, L.; and Maddison, C. J. 2022. Learning to Cut by Looking Ahead: Cutting Plane Selection via Imitation Learning. In Proceedings of the 39th International Conference on Machine Learning, 17584–17600. Baltimore, MD.

Pu, T.; Li, J.; Gao, Y.; Liu, S.; Geng, Z.; Liu, H.; Chen, C.; and Fan, C. 2026. CoCo-MILP: Inter-Variable Contrastive and Intra-Constraint Competitive MILP Solution Prediction. In Proceedings of the 40th AAAI Conference on Artificial Intelligence, 24882–24890. Singapore.

Puigdemont, P.; Skoulakis, S.; Chrysos, G.; and Cevher, V. 2024. Learning to Remove Cuts in Integer Linear Programming. In Proceedings of the 41st International Conference on Machine Learning, 41235–41255. Vienna, Austria.

Scavuzzo, L.; Chen, F.; Chetelat, D.; Gasse, M.; Lodi, A.; Yorke-Smith, N.; and Aardal, K. 2022. Learning to Branch with Tree MDPs. In Advances in Neural Information Processing Systems 35, 18514–18526. New Orleans, LA.

Tang, Y.; Agrawal, S.; and Faenza, Y. 2020. Reinforcement Learning for Integer Programming: Learning to Cut. In Proceedings of the 37th International Conference on Machine Learning, 9367–9376. Virtual.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems 30, 5998–6008. Long Beach, CA.

Wang, J.; Wang, Z.; Li, X.; Kuang, Y.; Shi, Z.; Zhu, F.; Yuan, M.; Zeng, J.; Zhang, Y.; and Wu, F. 2024. Learning to Cut via Hierarchical Sequence/Set Model for Eficient Mixed-Integer Programming. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12): 9697–9713.

Wang, R.; Li, X.; and Wang, M. 2026. MILPnet: A Multi-Scale Architecture with Geometric Feature Sequence Representations for Advancing MILP Problems. In Proceedings ofthe 14th International Conference on Learning Representations. Rio de Janeiro, Brazil.

Wang, Z.; Li, X.; Wang, J.; Kuang, Y.; Yuan, M.; Zeng, J.; Zhang, Y.; and Wu, F. 2023. Learning Cut Selection

for Mixed-Integer Linear Programming via Hierarchical Sequence Model. In Proceedings of the 11th International Conference on Learning Representations. Kigali, Rwanda.

Yoon, T. 2022. Confidence Threshold Neural Diving. arXiv:2202.07506.

Zarpellon, G.; Jo, J.; Lodi, A.; and Bengio, Y. 2021. Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies. In Proceedings ofthe 35th AAAI Conference on Artificial Intelligence, 3931–3939. Virtual.

Zeng, H.; Wang, J.; Das, A.; He, J.; Han, K.; Hu, H.; and Sun, M. 2024. Efective Generation of Feasible Solutions for Integer Programming via Guided Difusion. arXiv:2406.12349.

Zhang, C.; Ouyang, W.; Yuan, H.; Gong, L.; Sun, Y.; Guo, Z.; Dong, Z.; and Yan, J. 2024. Towards Imitation Learning to Branch for MIP: A Hybrid Reinforcement Learning Based Sample Augmentation Approach. In Proceedings ofthe 12th International Conference on Learning Representations. Vienna, Austria.

## A Related Work

## A.1 Learning alongside MILP Solvers

In recent years, machine learning has emerged as a promising paradigm for accelerating MILP solving (Li et al. 2024a), and existing studies generally fall into two categories. The first category learns heuristic policies for key components of branch-and-bound (B&B) solvers, including branching variable selection (Khalil et al. 2016; Gasse et al. 2019; Gupta et al. 2020; Zarpellon et al. 2021; Gupta et al. 2022; Scavuzzo et al. 2022; Lin et al. 2024; Zhang et al. 2024; Kuang et al. 2024), cutting plane selection (Tang, Agrawal, and Faenza 2020; Wang et al. 2023, 2024; Huang et al. 2021; Paulus et al. 2022; Ling, Wang, and Wang 2024; Puigdemont et al. 2024), scheduling of primal heuristics (Khalil et al. 2017; Chmiela et al. 2021). These methods typically leverage imitation learning or reinforcement learning to learn efective heuristic policies from historical solving trajectories or expert policies.

Among all components, branching and cut selection have received the most attention due to their significant impact on solver performance. For branching, Khalil et al. (2016) pioneered the imitation of strong branching decisions via a learned ranking function, and more recently, Hou et al. (2026) leveraged large language models to automatically discover branching policies. For cut selection, Tang, Agrawal, and Faenza (2020) proposed to learn adaptive selection policies through reinforcement learning, while M. et al. (2026) further combined imitation learning and reinforcement learning to obtain memory-eficient policies. Collectively, these studies demonstrate that learning efective policies for individual solver components can substantially enhance the performance of modern MILP solvers.

## A.2 End-to-End Prediction for MILP

The second category employs machine learning models such as graph neural networks (GNNs) to directly predict highquality solutions for MILP instances (Ding et al. 2020; Zeng et al. 2024; Geng et al. 2025). Research in this category advances along two complementary aspects. The first aspect concerns how predictions are exploited to reduce the original problem (Nair et al. 2020; Han et al. 2023; Liu et al. 2025). Nair et al. (2020) proposed to directly fix high-confidence variables to their predicted values, yielding a reduced subproblem, which was subsequently refined in follow-up studies (Yoon 2022; Paulus and Krause 2023). Han et al. (2023) and Huang et al. (2024) relaxed such hard fixing by searching within a trust region centered at the predicted partial assignment, and Liu et al. (2025) further extended this paradigm by alternating between prediction and solver-based correction.

The second aspect focuses on improving the prediction capacity (Pu et al. 2026; Wang, Li, and Wang 2026). Pu et al. (2026) enhanced prediction quality by explicitly modeling inter-variable correlations, while Wang, Li, and Wang (2026) replaced the conventional bipartite graph representation with geometric feature sequences and performed prediction via a multi-scale attention architecture. Together, these eforts have established learning-based solution prediction as an efective paradigm for accelerating MILP solving.

## B Graph Representation Details

The bipartite graph representation used in this paper is based on the representation adopted by Predict-and-Search (Han et al. 2023). To support hierarchical decoding, we further introduce three additional variable features: target indicator, previous prediction, and previous confidence. We list the graph features in Table 5.

<table><tr><td>Index</td><td>Variable Feature Name</td><td>Description</td></tr><tr><td>0</td><td>Objective</td><td>Normalized objective coefficient</td></tr><tr><td>1</td><td>Variable coefficient</td><td>Average variable coefficient in all constraints</td></tr><tr><td>2</td><td>Variable degree</td><td>Degree of the variable node in the bipartite graph representation</td></tr><tr><td>3</td><td>Maximum variable coefficient</td><td>Maximum variable coefficient in all constraints</td></tr><tr><td>4</td><td>Minimum variable coefficient</td><td>Minimum variable coefficient in all constraints</td></tr><tr><td>5</td><td>Variable type</td><td>Whether the variable is an integer variable or not</td></tr><tr><td>6</td><td>Target indicator</td><td>Whether the variable belongs to the current prediction hierarchy</td></tr><tr><td>7</td><td>Previous prediction</td><td>Previously predicted value of the variable</td></tr><tr><td>8</td><td>Previous confidence</td><td>Confidence score of the predicted value</td></tr><tr><td>9-24</td><td>Position embedding</td><td>Binary encoding of the order of appearance for each variable among all variables</td></tr><tr><td>Index</td><td>Constraint Feature Name</td><td>Description</td></tr><tr><td>0</td><td>Constraint coefficient</td><td>Average of all coefficients in the constraint</td></tr><tr><td>1</td><td>Constraint degree</td><td>Degree of constraint nodes</td></tr><tr><td>2</td><td>Bias</td><td>Normalized right-hand side of the constraint</td></tr><tr><td>3</td><td>Sense</td><td>The sense of the constraint</td></tr><tr><td>Index</td><td>Edge Feature Name</td><td>Description</td></tr><tr><td>0</td><td>Coefficient</td><td>Constraint coefficient</td></tr></table>

Table 5: The variable features, constraint features, and edge features used for the graph representation. Newly introduced features are shown in bold.

## C Variable Coupling Graph Details

Expected Violation The expected violation measures how strongly a variable pair is restricted by a constraint. Violation of the variable pair $( z _ { i } , z _ { j } )$ in constraint c is defined as

$$
\Delta \left( \phi _ { i j } ^ { ( c ) } \right) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \bigl ( b _ { c } - \phi _ { i j } ^ { ( c ) } , 0 \bigr ) , } & { \circ _ { c } \mathrm { i s } \ \geq , } \\ { \operatorname* { m a x } \bigl ( \phi _ { i j } ^ { ( c ) } - b _ { c } , 0 \bigr ) , } & { \circ _ { c } \mathrm { i s } \ \leq , } \\ { \left| \phi _ { i j } ^ { ( c ) } - b _ { c } \right| , } & { \circ _ { c } \mathrm { i s } = . } \end{array} \right.
$$

where $\phi _ { i j } ^ { ( c ) } = a _ { c i } z _ { i } + a _ { c j } z _ { j }$ , with $a _ { c i }$ and $\boldsymbol { a } _ { c j }$ denoting the coeficients of variables $z _ { i }$ and $z _ { j }$ in constraint c, respectively.

The expected violation $P _ { i j } ^ { ( c ) } = \mathbb { E } \left[ \Delta \left( \phi _ { i j } ^ { ( c ) } \right) \right]$ is computed as follows. For a binary–binary pair $( z _ { i } , z _ { j } )$ , we assume $z _ { i } , z _ { j } \sim$ Uniform $( \{ 0 , 1 \} )$ ), and the expected violation is computed as

$$
P _ { i j } ^ { \left( c \right) } = \frac { 1 } { 4 } \sum _ { z _ { i } , z _ { j } \in \{ 0 , 1 \} } \Delta \left( \phi _ { i j } ^ { \left( c \right) } \right) .
$$

For a binary–continuous pair $( z _ { i } , z _ { j } )$ , we assume $z _ { i } \sim$ Uniform({0, 1}) and $z _ { j } \sim \mathrm { U n i f o r m } ( \bar { l } _ { j } , u _ { j } )$ , where $l _ { j }$ and $u _ { j }$ denote the lower and upper bounds of $z _ { j }$ . The expected violation is computed as

$$
P _ { i j } ^ { ( c ) } = \frac { 1 } { 2 } \sum _ { z _ { i } \in \{ 0 , 1 \} } \frac { 1 } { u _ { j } - l _ { j } } \int _ { l _ { j } } ^ { u _ { j } } \Delta \left( \phi _ { i j } ^ { ( c ) } \right) d z _ { j } .
$$

To eliminate scale diferences across constraints and measure the relative strength of a variable pair within the constraint, we normalize the expected violation as

$$
\widehat { P } _ { i j } ^ { ( c ) } = \frac { P _ { i j } ^ { ( c ) } } { \frac { 1 } { | E _ { c } | } \displaystyle \sum _ { ( u , v ) \in E _ { c } } P _ { u v } ^ { ( c ) } } ,
$$

where $E _ { c }$ denotes the set of retained variable pairs in constraint c.

Coeficient Importance The coeficient importance characterizes the relative contribution of a variable pair within a constraint. For each variable $z _ { i } ,$ , we define a scale factor $s _ { i }$ according to its variable type, where $s _ { i }$ is set to 1 for binary variables and to $u _ { i } - l _ { i }$ for continuous variables. Then we compute the coeficient importance $r _ { i j } ^ { ( c ) }$ of the variable pair and normalize it within the constraint to obtain $\widehat { r } _ { i j } ^ { ( c ) }$ , which represents the relative importance of a variable pair within the constraint:

$$
r _ { i j } ^ { ( c ) } = \frac { | a _ { c i } | s _ { i } + | a _ { c j } | s _ { j } } { 2 } ,
$$

$$
\widehat { r } _ { i j } ^ { ( c ) } = \frac { r _ { i j } ^ { ( c ) } } { \frac { 1 } { | E _ { c } | } \displaystyle \sum _ { ( u , v ) \in E _ { c } } r _ { u v } ^ { ( c ) } } .
$$

Edge Weight Aggregation. Finally, the local coupling weight of a pair in constraint $c$ is computed as $w _ { i j } ^ { ( c ) } ~ =$ $\widehat { P } _ { i j } ^ { ( c ) } \widehat { r } _ { i j } ^ { ( c ) }$ , and the final edge weight $W _ { i j }$ between variables $z _ { i }$ and $z _ { j }$ is obtained by summing the local weights over all constraints containing both variables.

## D Benchmark Details

## D.1 Benchmarks in Main Evaluation

The CA and SC benchmark instances are generated following the process described by Gasse et al. (2019). Specifically, the CA dataset follows the generation method of Leyton-Brown, Pearson, and Shoham (2000), while the SC dataset is constructed using the algorithm presented by Balas and Ho (1980). The IP and WA instances are obtained from the NeurIPS ML4CO 2021 competition (Gasse et al. 2022). The statistical information is provided in Table 6.

<table><tr><td></td><td>CA</td><td>WA</td><td>IP</td><td>SC</td></tr><tr><td>Constraint Number</td><td>2592</td><td>64318</td><td>195</td><td>3000</td></tr><tr><td>Variable Number</td><td>1500</td><td>61000</td><td>1083</td><td>5000</td></tr><tr><td>Binary Variables</td><td>1500</td><td>1000</td><td>1050</td><td>5000</td></tr><tr><td>Continuous Variables</td><td>0</td><td>60000</td><td>33</td><td>0</td></tr><tr><td>Integer Variables</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 6: Summary statistics of the benchmark datasets used in the main evaluation.

## D.2 Subset of MIPLIB

We use the IIS dataset, which is a subset ofMIPLIB (Gleixner et al. 2021) constructed by Liu et al. (2025), to evaluate the solvers’ ability to handle challenging real-world instances. According to Liu et al. (2025), the instances in the dataset are selected based on their similarity, which is measured using the 100 human-designed features (Gleixner et al. 2021). Instances with presolving times exceeding 300 seconds or those that exceed GPU memory limits during the inference process are discarded. The IIS dataset contains eleven instances, including eight training instances and three testing instances (ramos3, scpj4scip, and scpl4). Detailed statistics of the IIS dataset are reported in Table 7.

<table><tr><td>Instance</td><td>Constraint Number</td><td>Variable Number</td><td>Binary Variables</td></tr><tr><td>ex1010-pi</td><td>1468</td><td>25200</td><td>25200</td></tr><tr><td>fast0507</td><td>507</td><td>63009</td><td>63009</td></tr><tr><td>glass-sc</td><td>6119</td><td>214</td><td>214</td></tr><tr><td>iis-glass-cov</td><td>5375</td><td>214</td><td>214</td></tr><tr><td>iis-hc-cov</td><td>9727</td><td>297</td><td>297</td></tr><tr><td>ramos3</td><td>2187</td><td>2187</td><td>2187</td></tr><tr><td>scpj4scip</td><td>1000</td><td>99947</td><td>99947</td></tr><tr><td>scpk4</td><td>2000</td><td>100000</td><td>100000</td></tr><tr><td>scpl4</td><td>2000</td><td>200000</td><td>200000</td></tr><tr><td>seymour</td><td>4944</td><td>1372</td><td>1372</td></tr><tr><td>v150d30-2hopcds</td><td>7822</td><td>150</td><td>150</td></tr></table>

Table 7: Statistical information of the instances in the IIS dataset. All instances contain only binary variables.

## E Implementation Details

## E.1 SHSP Predictor Details

The predictor used in SHSP is based on the graph neural network architecture adopted in previous learning-based MILP approaches (Han et al. 2023; Liu et al. 2025). To support the hierarchical decoding process in SHSP, we introduce a learnable step embedding, which enables the network to adapt its prediction behavior across diferent decoding steps. Specifically, the initial representation of each variable node is computed as

$$
h _ { v _ { i } } ^ { ( 0 , k ) } = f _ { s } \left( \left[ f _ { v } \left( x _ { i } \right) ; \operatorname { E m b } ( k ) \right] \right) .
$$

Here, $v _ { i }$ denotes the i-th variable node, k is the current decoding step, and $x _ { i }$ is the feature vector of $v _ { i }$ . The function $f _ { v } ( \cdot )$ denotes the variable embedding network, Emb(k) is the learnable embedding corresponding to decoding step k, and $f _ { s } ( \cdot )$ fuses the variable and step embeddings. The resulting representation $h _ { v _ { i } } ^ { ( 0 , k ) }$ is then used as the input to the same bipartite message-passing architecture adopted in previous approaches (Han et al. 2023; Liu et al. 2025).

## E.2 Training Details

During predictor training, we set the initial learning rate to be 0.001 and the number of training epochs to 500. To collect the training data, we run a single-threaded Gurobi (Gurobi Optimization, LLC 2026) on each training and validation instance for 3,600 seconds and record the best 50 solutions.

For SHSP training, we incorporate a teacher forcing strategy that employs ground-truth assignments as conditioning information in the early stage of training, and the teacher forcing ratio $\rho _ { t }$ is progressively decayed as training proceeds. Specifically, with probability $\rho _ { t }$ , we use the ground-truth value as the history input for the next step; otherwise, we use the model prediction. The teacher forcing ratio $\rho _ { t }$ is linearly decayed from $\rho _ { \mathrm { s t a r t } }$ to $\rho _ { \mathrm { e n d } }$ during the first $T _ { \mathrm { d e c a y } }$ epochs:

$$
\rho _ { t } = \rho _ { \mathrm { s t a r t } } + \operatorname* { m i n } \left( 1 , \frac { t } { T _ { \mathrm { d e c a y } } } \right) \left( \rho _ { \mathrm { e n d } } - \rho _ { \mathrm { s t a r t } } \right) ,
$$

where t is the current epoch. During validation and inference, we disable teacher forcing and always use the model predictions to update the conditioning context. In experiments, we set $\rho _ { \mathrm { s t a r t } } = 1 . 0 , \rho _ { \mathrm { e n d } } = 0$ , and $T _ { \mathrm { d e c a y } } = 5 0$

## E.3 Integration with Downstream Learning-based Search Methods

We replace the original GNN predictor and the variable fixing strategy in each framework with our SHSP predictor and structure-aware variable fixing strategy, forming ND+SHSP, PaS+SHSP, and Apollo+SHSP, respectively.

For ND+SHSP and PaS+SHSP, the variable coupling graph is constructed once before neural network inference for each instance. The graph construction time is included in the solver time budget.

For Apollo+SHSP, the variable coupling graph is constructed before the first prediction and is subsequently updated for each reduced subproblem before prediction. Similarly, the time required for graph construction and updates is deducted from the corresponding solver time budget.

## E.4 Inference Details

We conducted all the experiments on a single machine with an AMD EPYC 7513 32-Core Processor and NVIDIA GeForce RTX 4090 GPUs. The inference settings are described as follows.

For the baselines, the hyperparameters $( k _ { 0 } , k _ { 1 } )$ for ND and $( k _ { 0 } , k _ { 1 } , \Delta )$ for PaS are summarized in Table 8. For Apollo-MILP, the number of iterations is set to 4. The hyperparameter settings $( k _ { 0 } ^ { ( i ) } , k _ { 1 } ^ { ( i ) } , \Delta ^ { ( i ) } )$ for each iteration are summarized in Table 9.

<table><tr><td>Dataset</td><td>PaS  $( k _ { 0 } , k _ { 1 } , \Delta )$ </td><td> $\mathrm { N D } \left( k _ { 0 } , k _ { 1 } \right)$ </td></tr><tr><td>CA</td><td>(600, 0, 20)</td><td>(600,0)</td></tr><tr><td>WA</td><td>(0, 500, 10)</td><td>(0,500)</td></tr><tr><td>IP</td><td>(400, 5, 10)</td><td>(400, 5)</td></tr><tr><td>SC</td><td>(2000, 0, 100)</td><td>(2000,0)</td></tr></table>

Table 8: Hyperparameter settings for PaS and ND on diferent benchmark datasets.

For our method, we use the same hyperparameter settings as the corresponding baseline framework. The number of decoding steps is set to 2 for all benchmark datasets except WA, for which it is set to 4. The additional hyperparameter

<table><tr><td></td><td>CA</td><td>WA</td><td>IP</td><td>SC</td></tr><tr><td>Iteration 1</td><td>(400, 0, 60)</td><td>(20, 200, 100)</td><td>(100, 20,50)</td><td>(1000,0,200)</td></tr><tr><td>Iteration 2</td><td>(200, 0,30)</td><td>(10, 100, 50)</td><td>(40, 15, 20)</td><td>(500, 0, 100)</td></tr><tr><td>Iteration 3</td><td>(100, 0, 15)</td><td>(10, 5,5)</td><td>(20, 15, 10)</td><td>(250,0,50)</td></tr><tr><td>Iteration 4</td><td>(50, 0, 10)</td><td>(1, 10, 5)</td><td>(5, 50, 30)</td><td>(10,0,5)</td></tr></table>

Table 9: Hyperparameter configuration of $( k _ { 0 } ^ { ( i ) } , k _ { 1 } ^ { ( i ) } , \Delta ^ { ( i ) } )$ in Apollo-MILP.

settings $( \theta _ { 0 } , \theta _ { 1 } , \eta )$ are reported in Table 10, where η controls the VCS threshold. Specifically, let $z _ { ( 1 ) } , \dotsc , z _ { ( n ) }$ denote the variables sorted in descending order of VCS. We use the VCS of $z _ { ( \lceil \eta n \rceil ) }$ as the threshold.
<table><tr><td>Dataset</td><td>ND+SHSP</td><td>PaS+SHSP</td><td>Apollo+SHSP</td></tr><tr><td>CA</td><td>(0.10, 0.90, 0.5)</td><td>(0.10, 0.90, 0.5)</td><td>(0.04, 0.90, 0.7)</td></tr><tr><td>WA</td><td>(0.10, 0.90, 0.7)</td><td>(0.10, 0.90, 0.7)</td><td>(0.10, 0.95, 0.8)</td></tr><tr><td>IP</td><td>(0.05, 0.95, 0.7)</td><td>(0.10, 0.90, 0.7)</td><td>(0.10, 0.50, 0.5)</td></tr><tr><td>SC</td><td>(0.10, 0.90, 0.5)</td><td>(0.10, 0.90, 0.5)</td><td>(0.10, 0.90, 0.7)</td></tr></table>

Table 10: The additional hyperparameter settings $( \theta _ { 0 } , \theta _ { 1 } , \eta )$ for ND+SHSP, PaS+SHSP and Apollo+SHSP on diferent benchmark datasets.

## F Additional Experimental Results

## F.1 More Ablation Study Results

Confidence-Based Mask and Repair Mechanism We perform an ablation study to evaluate the proposed maskand-repair mechanism. Specifically, we compare three variants: None (without mask or repair), Mask (mask only), and Mask+Repair (mask and repair) on the four benchmark datasets (CA, WA, IP, and SC). The results are shown in Table 11. The observations are consistent with those discussed in the main paper: the two components are complementary and jointly contribute to the overall performance.

Hierarchical Solution Prediction Predictor To evaluate the efectiveness of the proposed hierarchical solution prediction predictor, we compare it with the original GNN predictor under the same structure-aware fixing strategy on the four benchmark datasets (CA, WA, IP, and SC). The results are shown in Table 12, which suggests that our predictor generates more accurate solution predictions, thereby enabling more efective variable fixing and problem reduction.

Fixing Strategies To evaluate the efectiveness of the proposed structure-aware fixing strategy, we compare it with the original fixing strategy while using the same predictor. The results on the four benchmark datasets (CA, WA, IP, and SC) are shown in Table 12. The observations are consistent with those discussed in the main paper: prioritizing structurally important variables leads to more efective search space reduction, allowing the solver to explore the remaining search space more eficiently and achieve better performance.

## F.2 Results on MIPLIB

To further demonstrate the applicability of our method, we conduct experiments on instances from challenging realworld dataset MIPLIB (Gleixner et al. 2021). However, applying ML-based solvers directly to the entire dataset can be dificult because of the heterogeneous nature of the instances in MIPLIB. Therefore, following Liu et al. (2025), we focus on a subset of MIPLIB that contains similar instances. More information on the selected MILP subset, referred to as IIS, is provided in Appendix D.2. We report the solving performance of the solvers in Table 13 and Table 14, where our methods outperform their corresponding baselines, showcasing the potential for real-world applications.

<table><tr><td rowspan="2">Method</td><td colspan="2">CA ↑(BKS:98489.24)</td><td colspan="2">WA ↓(BKS:706.86)</td><td colspan="2">IP↓(BKS:11.69)</td><td colspan="2">SC ↓(BKS:123.30)</td></tr><tr><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td><td>Obj</td><td> ${ \mathrm { G a p } } _ { a b s }$ </td><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td></tr><tr><td>PaS+SHSP (None)</td><td>98380.92</td><td>108.32</td><td>707.06</td><td>0.20</td><td>12.12</td><td>0.43</td><td>123.47</td><td>0.17</td></tr><tr><td>PaS+SHSP (Mask)</td><td>98416.55</td><td>72.69</td><td>707.05</td><td>0.19</td><td>12.17</td><td>0.48</td><td>123.53</td><td>0.23</td></tr><tr><td>PaS+SHSP (Mask+Repair)</td><td>98487.72</td><td>1.52</td><td>706.99</td><td>0.13</td><td>11.97</td><td>0.28</td><td>123.41</td><td>0.11</td></tr><tr><td>Apollo+SHSP (None)</td><td>98385.20</td><td>104.04</td><td>707.10</td><td>0.24</td><td>11.78</td><td>0.09</td><td>123.42</td><td>0.12</td></tr><tr><td>Apollo+SHSP (Mask)</td><td>98344.07</td><td>145.17</td><td>707.01</td><td>0.15</td><td>11.80</td><td>0.11</td><td>123.39</td><td>0.09</td></tr><tr><td>Apollo+SHSP (Mask+Repair)</td><td>98489.24</td><td>0.00</td><td>707.01</td><td>0.15</td><td>11.71</td><td>0.02</td><td>123.37</td><td>0.07</td></tr></table>

Table 11: Detailed ablation study of the proposed confidence-based mask-and-repair strategy. ↑ indicates that higher values are better, while ↓ indicates that lower values are better. Bold values denote the best results, including ties.
<table><tr><td rowspan="2">Method</td><td colspan="2">CA ↑(BKS:98489.24)</td><td colspan="2">WA ↓(BKS:706.86)</td><td colspan="2">IP↓(BKS:11.69)</td><td colspan="2">SC ↓(BKS:123.30)</td></tr><tr><td>Obj</td><td> ${ \mathrm { G a p } } _ { a b s }$ </td><td>Obj</td><td> ${ \mathrm { G a p } } _ { a b s }$ </td><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td><td>Obj</td><td> $\mathrm { G a p } _ { a b s }$ </td></tr><tr><td>PaS (structure-aware fixing)</td><td>97184.33</td><td>1304.91</td><td>707.08</td><td>0.22</td><td>12.12</td><td>0.43</td><td>123.49</td><td>0.19</td></tr><tr><td>PaS+SHSP (original fixing)</td><td>98061.51</td><td>427.73</td><td>707.01</td><td>0.15</td><td>12.34</td><td>0.65</td><td>123.53</td><td>0.23</td></tr><tr><td>PaS+SHSP (structure-aware fixing)</td><td>98487.72</td><td>1.52</td><td>706.99</td><td>0.13</td><td>11.97</td><td>0.28</td><td>123.41</td><td>0.11</td></tr><tr><td>Apollo (structure-aware fixing)</td><td>98015.32</td><td>473.92</td><td>707.09</td><td>0.23</td><td>11.87</td><td>0.18</td><td>123.40</td><td>0.10</td></tr><tr><td>Apollo+SHSP (original fixing)</td><td>98070.43</td><td>418.81</td><td>707.01</td><td>0.15</td><td>12.08</td><td>0.39</td><td>123.37</td><td>0.07</td></tr><tr><td>Apollo+SHSP (structure-aware fixing)</td><td>98489.24</td><td>0.00</td><td>707.01</td><td>0.15</td><td>11.71</td><td>0.02</td><td>123.37</td><td>0.07</td></tr></table>

Table 12: Detailed ablation study of the proposed hierarchical solution prediction predictor and structure-aware fixing strategy. ↑ indicates that higher values are better, while ↓ indicates that lower values are better. Bold values denote the best results, including ties.

<table><tr><td>Method</td><td>Obj↓</td><td> $\operatorname { G a p } _ { \mathrm { a b s } } \downarrow$ </td></tr><tr><td>Gurobi</td><td>211.00</td><td>20.00</td></tr><tr><td>PaS</td><td>209.00</td><td>18.00</td></tr><tr><td>PaS+SHSP</td><td>206.67</td><td>15.67</td></tr><tr><td>Apollo</td><td>214.33</td><td>23.33</td></tr><tr><td>Apollo+SHSP</td><td>208.67</td><td>17.67</td></tr></table>

Table 13: Results on the IIS dataset, a subset of MIPLIB used by Liu et al. (2025). All learning-based methods use Gurobi as the downstream solver, with a solving time limit of 3,600 seconds.

## F.3 Runtime Results

We report network inference time and graph construction time in Table 15. The inference time of the SHSP predictor is only slightly higher than that ofthe original GNN predictor,

<table><tr><td>Instance</td><td>BKS</td><td>Gurobi</td><td>PaS</td><td>Apollo</td><td>PaS+SHSP</td><td>Apollo+SHSP</td></tr><tr><td>ramos3</td><td>186.00</td><td>228.00</td><td>227.00</td><td>242.00</td><td>221.00</td><td>226.00</td></tr><tr><td>scpj4scip</td><td>128.00</td><td>133.00</td><td>131.00</td><td>132.00</td><td>130.00</td><td>131.00</td></tr><tr><td>scpl4</td><td>259.00</td><td>272.00</td><td>269.00</td><td>269.00</td><td>269.00</td><td>269.00</td></tr></table>

Table 14: The best objectives found on each test instance in IIS. BKS denotes the best known objective values reported by MIPLIB (https://miplib.zib.de/index.html).

which remains negligible compared with the overall solver runtime. In addition, the graph construction time ranges from 0.26s to 15.23s per instance, which is also much shorter than the solver runtime.
<table><tr><td>Dataset</td><td>GNN Predictor</td><td>SHSP Predictor</td><td>Graph construction time</td></tr><tr><td>CA</td><td>0.0050s</td><td>0.0146s</td><td>0.82s</td></tr><tr><td>WA</td><td>0.0439s</td><td>0.2197s</td><td>5.48s</td></tr><tr><td>IP</td><td>0.0043s</td><td>0.0131s</td><td>0.26s</td></tr><tr><td>SC</td><td>0.0658s</td><td>0.1953s</td><td>15.23s</td></tr></table>

Table 15: Comparison of the average per-instance inference time and graph construction time (in seconds) on the four benchmark datasets.