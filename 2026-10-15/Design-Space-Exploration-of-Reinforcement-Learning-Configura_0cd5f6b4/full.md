# Design Space Exploration of Reinforcement Learning Configurations for Post-Legalization Placement

Foteini Oikonomou<sup>\*</sup>, Maria Pantazi-Kypraiou<sup>\*</sup>, George Goudroumanis<sup>\*</sup>, Pavlos Stoikos<sup>\*</sup>,

Athanasios Tziouvaras<sup>\*</sup>, Christos Anagnostopoulos<sup>†</sup>, Georgios Karakonstantis<sup>\*</sup>, and George Floros<sup>‡</sup>

Department of Electrical and Computer Engineering, University of Thessaly, Volos, Greece

emails: {fotoikon, mpantazi-, ggeorgios-r, pastoikos, attziouv, gkarakon}@e-ce.uth.gr

<sup>†</sup>School of Computing Science, University of Glasgow, UK, email: christos.anagnostopoulos@glasgow.ac.uk

<sup>‡</sup>Department of Electronic and Electrical Engineering, Trinity College Dublin, Ireland, email: florosg@tcd.ie

Abstract—Graph Neural Networks (GNNs) and Reinforcement Learning (RL) have recently been explored for VLSI placement optimization, but their effectiveness for post-legalization placement refinement remains unclear. This paper presents a systematic design space exploration of a GNN-Proximal Policy Optimization (GNN-PPO) framework across three ISPD2015 benchmarks. We evaluate 18 configurations covering reward functions, GNN architectures, action spaces, candidate-set sizes, and training strategies. We analyse the results we obtain to identify pitfalls and to assess the prospects and opportunities of the RL methods for post-legalization placement. On top of that we propose a design classification approach to guide the design decisions of RL models.

Index Terms—Placement Refinement, Reinforcement Learning, Graph Neural Network, Design Space Exploration, VLSI EDA, ISPD2015

## I. INTRODUCTION

Post-legalization detailed placement refinement seeks to reduce half-perimeter wirelength (HPWL) after global placement and legalization, without violating site constraints. Classical methods such as perturbation moves, local search, and HPWL greedy swaps have been refined over decades and remain highly competitive [1], [2]. Analytical placers such as DREAMPlace [3] and RePlAce [4] achieve near-optimal results via differentiable HPWL optimization. Reinforcement learning (RL) approaches, popularized by macro placement [5], emerge as an alternative approach. However, RL-based standard-cell methods [6], [7] remain underexplored and operate at smaller scale or under simplified assumptions, such as discretized grids.

The direct application of RL to standard-cell refinement is therefore poorly characterized. It is unclear which GNN architecture, reward formulation, or action space is most effective, and whether the RL can scale to large designs. This paper addresses that gap with a controlled, multi-dimensional design space exploration across three ISPD2015 benchmarks. It reports on the obtained results and benchmarks its findings against OpenROAD [1].

The paper makes four main contributions:

• We conduct a systematic design space exploration of an RL method that combines a GNN with PPO for post-legalization placement refinement, evaluating 18 configurations across several design dimensions and over both regular and irregular netlists.

• We uncover pitfalls and design considerations that, if left unchecked, may impede the effectiveness of RL, and we show they are tightly linked to whether the netlist is regular or irregular.

• We identify the prospects and opportunities of RL and show that graph partitioning overcomes some of its limitations where the full-graph reward is noisy or sparse, helping substantially on an irregular design and modestly on a regular one.

• We propose a design classification method that provides practical guidance for choosing the most suitable training strategy.

## II. DESIGN SPACE EXPLORATION FRAMEWORK

In this section we elaborate on the different methods, models and parameters we use in order to perform the envisioned design space exploration.

## A. Problem Formulation

In reinforcement learning, an agent is leveraged that observes an environment, takes some actions and then, receives a reward that resembles the utility of selected actions. In our case, given a legalized placement NET of N standard cells, the agent selects a sequence of single-cell moves to minimize total HPWL. Thus, for each time step t, the RL follows an iterative process which consists the following operations:

1) Observation: NET is fed to a GNN, which outputs the representation of the RL environment.

2) Refinement: Each cell is assigned a score by the GNN. The top-K cells (in terms of that score) formulate candidate set, upon which the agent will choose an action.

3) Action selection: The policy π selects one cell c and one action a.

4) Action implementation: Cell c moves according to action a. We allow temporary overlaps during moves, which we then fix with a final legalization step.

5) Reward generation: A reward $r _ { t }$ is computed and the agent updates its policy via $\pi _ { \theta }$ the PPO technique.

## B. Graph Construction

The netlist is represented as a heterogeneous graph $\mathcal { G } = ( V , E )$ where nodes $\begin{array} { r l r } { v _ { i } } & { { } \in } & { V } \end{array}$ are placeable cells with features $( x _ { i } , y _ { i } , w _ { i } , h _ { i } )$ and edges $( v _ { i } , v _ { j } ) \in E$ for every two-pin net. High-fanout nets (>20 pins), power pins and clock nets are excluded from edge construction to avoid dominating the GNN.

## C. GNN Architecture

We evaluate the effectiveness of three GNN architectures:

• GAT-2L: A standard graph attention model with 2 layers.

• GCN: A spectral convolution architecture implemented in [8].

• GraphSAGE: The GraphSAGE architecture, implemented in [9].

## D. Action spaces

We evaluate three action spaces:

• 1-site move: Move the cell to one adjacent placement site; $a \in \{ \mathrm { N } , \mathrm { S } , \mathrm { E } ,$ W, no move}.

• 3-site move: Move the cell to a position, which is located 3 placement sites away; a ∈ {NNN, SSS, EEE, WWW, no move}.

• Multi-site move: Move the cell to a position, which is located up to 3 placements sites away; $a \in \{ \mathrm { N } ,$ NN, NNN, S, SS, SSS, E, EE, EEE, W, WW, WWW, no move}.

## E. Reward function

In this work, we analyze the effect of different reward structures in order to study their effects, as follows. The first reward structure, named Local is derived as follows:

$$
\begin{array} { r } { L o c a l = \sum _ { n \in \mathcal { N } _ { t } } \left( \mathrm { H P W L } _ { n } ( P _ { t - 1 } ) - \mathrm { H P W L } _ { n } ( P _ { t } ) \right) } \end{array}\tag{1}
$$

where $\mathcal { N } _ { t }$ is the set of the nets incident to the moved cell $c ,$ during the step t.

The second reward structure, named Normalized is derived as follows:

$$
N o r m a l i z e d = \frac { L o c a l } { d }\tag{2}
$$

where d is the total perimeter of the die area.

## F. Graph partitioning

For large or medium designs where the full-graph reward signal is too sparse, we introduce a subregion training technique: For each RL iteration, a random bounding box is generated that covers a fraction (denoted as $f \in [ 0 , 1 ] )$ of the die area and then, the agent operates on this sampled graph partition. Evidently, when $f  0$ the agent has access to a smaller graph segment; whereas when $f  1$ the agent investigates larger graph segments in a single run. In both cases, any nets that remain outside of the subgraph are treated as virtual external nodes and contribute normally to the reward generation, but they are excluded from the action space.

## III. EXPERIMENTAL SETUP

For the design space exploration we leverage the ISPD2015 designs [2], as depicted in Table I. Benchmarks exhibit distinct structural properties that directly affect GNN performance, as discussed in section V. Starting from post-legalized layouts, refinement moves may cause overlaps, which are resolved via a final legalization pass. OpenROAD [1] serves as our baseline.

TABLE I  
THE ISPD2015 BENCHMARKS USED TO PERFORM THE DESIGN SPACE EXPLORATION.
<table><tr><td>Design</td><td>N (cells)</td><td>Baseline HPWL (µm)</td></tr><tr><td>mgc  $\_ { \mathrm { f f t } } _ { \mathrm { r } } \_ { \_ 1 }$ </td><td>32K</td><td>260,172</td></tr><tr><td> $\mathtt { m g c \_ p c i \_ b r i d g e } 3 2 \_ a$ </td><td>~30K</td><td>548,355 B</td></tr><tr><td> $\mathrm { \ m g c \_ m a t r i x \_ m u l t \_ 1 }$ </td><td>155K</td><td>1,518,463</td></tr></table>

We run the RL training process, for each different RL configuration for 400 episodes (800 when we use our graph partitioning technique). For the evaluation process we perform inference, using the trained RL agent over the corresponding benchmark. To evaluate the performance of each RL configuration we use the following metrics:

• Training peak: The episode during training where the agent achieves the best HPWL decrease (Train∆%).

• Inference score: The decrease of HPWL (Eval∆%) that is achieved during inference, using the best performing RL checkpoint.

## IV. EXPERIMENTAL EVALUATION

Table II summarizes the conducted experiments along with the corresponding Train∆% and Eval∆%. For reference, OpenROAD achieved an HPWL reduction of 6.4% for mgc\_fft\_1, 4.3% for mgc\_pci\_bridge32\_a, and 4.6% for mgc\_matrix\_mult\_1. The results we obtain expose some key insights of the RL underlying parameters.

TABLE II  
DESIGN SPACE EXPLORATION CONFIGURATIONS AND RESULTS, GROUPED PER BENCHMARK.
<table><tr><td>Config.</td><td>Arch.</td><td>Action</td><td>Reward</td><td>K</td><td>Partit.</td><td>Train ∆%</td><td>Eval ∆%</td></tr><tr><td colspan="8">mgc_fft_1</td></tr><tr><td>C1</td><td>GAT-2L</td><td>1-site</td><td>Local</td><td>5K</td><td>No</td><td>0.44</td><td>0.20</td></tr><tr><td>C2</td><td>GAT-2L</td><td>1-site</td><td>Normalized</td><td>5K</td><td>No</td><td>0.41</td><td>0.20</td></tr><tr><td>C3</td><td>GCN</td><td>1-site</td><td>Local</td><td>5K</td><td>No</td><td>0.46</td><td>0.21</td></tr><tr><td>C4</td><td>SAGE</td><td>1-site</td><td>Local</td><td>5K</td><td>No</td><td>0.53</td><td>0.20</td></tr><tr><td>C5</td><td>GAT-2L</td><td>3-site</td><td>Local</td><td>5K</td><td>No</td><td>0.26</td><td>0.22</td></tr><tr><td>C7</td><td>SAGE</td><td>1-site</td><td>Local</td><td>5K</td><td> $f { = } 0 . 1 5$ </td><td>0.01</td><td>0.00</td></tr><tr><td colspan="8">mgc_pci_bridge32_a</td></tr><tr><td>C8</td><td>GAT-2L</td><td>1-site</td><td>Local</td><td>5K</td><td>No</td><td>0.36</td><td>0.15</td></tr><tr><td>C9</td><td>SAGE</td><td>1-site</td><td>Local</td><td>5K</td><td>No</td><td>0.00</td><td>0.00</td></tr><tr><td>C10</td><td>GAT-2L</td><td>1-site</td><td>Local</td><td>5K</td><td> $f { = } 0 . 1 5$ </td><td>2.99</td><td>0.15</td></tr><tr><td>C11</td><td>SAGE</td><td>1-site</td><td>Local</td><td>5K</td><td> $f { = } 0 . 1 5$ </td><td>5.85</td><td>0.43</td></tr><tr><td colspan="8">mgc_matrix_mult_1</td></tr><tr><td>C12</td><td>GAT-2L</td><td>1-site</td><td>Local</td><td>5K</td><td>No</td><td>0.01</td><td>0.01</td></tr><tr><td>C13</td><td>GAT-2L</td><td>1-site</td><td>Local</td><td>20K</td><td>No</td><td>0.00</td><td>0.01</td></tr><tr><td>C14</td><td>SAGE</td><td>1-site</td><td>Local</td><td>5K</td><td>No</td><td>0.01</td><td>0.01</td></tr><tr><td>C15</td><td>GAT-2L</td><td>1-site</td><td>Scaled</td><td>5K</td><td>No</td><td>0.00</td><td>0.01</td></tr><tr><td>C16</td><td>GAT-2L</td><td>1-site</td><td>Per-Net</td><td>5K</td><td>No</td><td>0.00</td><td>0.01</td></tr><tr><td>C17</td><td>SAGE</td><td>1-site</td><td>Local</td><td>5K</td><td>f=0.15</td><td>0.03</td><td>0.01</td></tr><tr><td>C18</td><td>SAGE</td><td>1-site</td><td>Local</td><td>5K</td><td>f=0.08</td><td>0.04</td><td>0.03</td></tr></table>

GNN architecture. GraphSAGE achieves some of the best training performance, but exhibits policy collapse in cases where the model is over-trained. In such cases, the reward degrades after and the model’s best checkpoint is not realized in the last training round. On the other hand, GCN converges more slowly yet remains stable throughout the training process.

In terms of inference, all architectures produce comparable results and thus, the training advantage of GraphSAGE is not fully transferred to inference.

Action space. We also tried a Swap option, which exchanges the selected cell with its nearest neighbor, however it is computationally expensive and difficult to converge. Among feasible variants, the 3-site move marginally outperforms 1- site move during inference (0.22 vs 0.20 Eval ∆% for the mgc\_fft\_1 design), though demonstrates small gains.

Reward function. The Local and Normalized rewards achieve comparable results and do not seem to significantly affect the RL training or inference process. A Global reward recomputing the total HPWL each move matches the exact ob jective but is ∼100× slower per episode, so we discarded it as intractable. We have also explored additional reward engineering approaches for the $\mathrm { \ m g c \_ m a t r i x \_ m u l t \_ 1 }$ benchmark in order to increase the RL performance. To this end, two additional rewards are explored, just for this benchmark. The first one, denoted as Per-Net is defined as follows:

$$
\begin{array} { r } { P e r - N e t = \frac { L o c a l } { \frac { 1 } { | \mathcal { N } _ { t } | } \sum _ { n \in \mathcal { N } _ { t } } \mathrm { H P W L } _ { n } ( P _ { t - 1 } ) } } \end{array}\tag{3}
$$

that is, the Local reward normalized by the mean pre-move HPWL of the nets incident to the moved cell, making the reward magnitude scale-invariant across die sizes, and the second one, denoted as Scaled, defined as: $S c a l e d = a * L o c a l .$ , which is a essentially an amplified Local reward, by a factor of $a .$ In our case, we set $a = 6$ . Both of these reward structures fail to deliver a better RL performance, thus highlighting that reward engineering cannot alone solve the RL training challenge.

Graph partitioning. Graph partitioning plays an important role on the overall RL performance, but its benefit is interconnected with the micro-architecture of each design and with the reason why full-graph training struggles. For example, the mgc\_pci\_bridge32\_a benchmark implements a bus circuit, where the bottleneck is a noisy reward signal. Restrict ing the agent to a small bounding box concentrates the local signal and stabilizes the learning process, and partitioning is therefore highly effective here. On the other hand, the $\mathrm { \ m g c \_ m a t r i x \_ m u l t \_ 1 }$ benchmark implements a circuit for matrix multiplication (C=A×B), designed via a regular partial-product array where every cell shares a similar k-hop neighborhood structure. As a result, a policy trained on a single partition can easily generalize to the rest of the design, since the cell topology is repeated in a periodic pattern, which makes even very small partitions effective and cheap to train. However, global placement has already optimized this regular layout, so little is left to improve and partitioning brings only a small gain. Partitioning also fails on mgc\_ $\_ { \ / \mathrm { f } } \mathrm { f t } - 1$ , its butterfly topology relies on long-range connections, so cutting the design into small boxes severs these nets and destroys the local reward signal, causing the policy to collapse.

Comparison with OpenROAD. OpenROAD achieves 4.3% to 6.4% HPWL reduction, above the final inference results of our RL configurations. The reason is largely structural, since OpenROAD evaluates coordinated multi-cell moves against the exact global HPWL, whereas the RL refines one cell at a time from a local signal, so credit assignment (Section V) is harder. The RL still shows promise, as on the irregular mgc\_pci\_bridge32\_a several training checkpoints exceed OpenROAD, so the main limitation is not solution quality but policy stability at inference. Reducing the search space through subregion training improves learning, and our netlist classification exploits this to flag designs that benefit from partitioned optimization.

## V. PITFALLS AND RL DESIGN CONSIDERATIONS

In this section we elaborate on the pitfalls of the RL design phase, which have the capacity to degrade the model perfor mance, if left unaddressed by the designers.

(1) Credit assignment bottleneck. Given the fact that the agent examines only portion of the total cells per episode $( K / N )$ , it has a limited perception of the whole netlist. This phenomenon is apparent in mgc\_matrix\_mult\_1, where the agent covers 3.2% of the total cells per episode, versus 15.5% for mgc\_fft\_1, and 16.9% for mgc\_pci\_bridge32\_a . In this case, the PPO fails to assign a policy to rest of the 96.8% of cells in each episode and thus, the model does not converge. Evidently, higher coverage rates (as in mgc\_pci\_bridge32\_a and mgc\_fft\_1) solve this problem, but require a higher K value. This translates to higher computational complexity and larger training times.

(2) GNN over-smoothing. The micro-architecture of mgc\_matrix\_mult\_1, and more specifically the fact that its cells share a similar neighborhood structure, can lead to oversmoothing. This phenomenon emerges when the GNN produces near-identical embeddings for all cells thus, preventing the policy from distinguishing beneficial moves from harmful ones. In contrast, the wire connections in mgc\_fft\_1 connections create structurally unique multi-hop neighborhoods and result in diverse embeddings. Overall, in designs where several cells share the same neighborhood configuration, GNN seem to decrease the model quality.

(3) Tightly placed cells. The periodic grid structure of a matrix multiplier maps well to a regular silicon layout, which Global placement exploits efficiently. The few remaining misplacements are largely out of reach for single-cell local moves, so the RL finds little to improve even though coordinated multi-cell refiners still reduce HPWL on this design. On the other hand, $\mathrm { m g c \_ f f t } \_ 1$ and mgc\_pci\_bridge32\_a carry more local residual misplacements that single-cell RL moves can exploit.

## VI. PROSPECTS AND OPPORTUNITIES OF RL

Results indicate that the RL can benefit from prior knowledge on each design’s characteristics (e.g. micro-architectural details). This is apparent when the agent is trained using smaller graph partitions, instead of the whole circuit. Then a question arises on how to a extract such information from a given netlist, without requiring to train the RL agent for a large number of steps. One possible solution is to classify the netlist to a set of pre-defined classes, which indicate how the graph should be partitioned. In this works we perform a preliminary assessment of this approach and we devise three distinct classes as follows:

Algorithm 1 Netlist classification   
Require: netlist net; candidate-set size $K ;$ score threshold   
$\theta _ { s } { = } 5 \times 1 0 ^ { - 4 }$ ; SNR threshold $\theta _ { m } { = } 2 . 0 ; \ P { = } 5$ probes of   
$L { = } 5 0 0$ steps   
Ensure: class $\in \{ { \bf A } , { \bf B } , { \bf C } \}$ and partition fraction f   
1: $N \gets$ number of cells in net; $d $ die perimeter   
2: HP $\overline { { \mathrm { W L } } } _ { \mathrm { n e t } } $ baseline HPWL of net $/$ number of nets   
3: $S _ { p } \gets ( \overline { { \mathrm { H P W L } } } _ { \mathrm { n e t } } / d ) \times ( K / N )$ ▷ Eq. (4)   
4: $\mathcal { R }  \emptyset$   
5: for $i \gets 1$ to P do   
6: $r _ { i } \gets$ total reward of an L-step random-policy rollout   
7: $\mathcal { R }  \mathcal { R } \cup \{ r _ { i } \}$ ▷ no training   
8: end for   
9: $R _ { m } \gets | \mathrm { m e a n } ( \mathcal { R } ) | / \mathrm { s t d } ( \mathcal { R } )$ ▷ Eq. (5)   
10: if $R _ { m } < \theta _ { m }$ then   
11: class $ \mathbf { B } ; ~ f  0 . 1 5$ ▷ noisy reward   
12: else if $S _ { p } > \theta _ { s }$ then   
13: class $ \mathrm { A } ; \quad f  1$ ▷ strong, clean signal   
14: else   
15: class $ \mathrm { { C } } ; f  0 . 0 8$ ▷ weak signal   
16: end if   
17: return (class, $f )$

• Class A: The graph should not be partitioned.

• Class B: The graph should be partitioned, with $f = 0 . 1 5$

• Class C: The graph should be partitioned, with $f = 0 . 0 8 .$

Given a netlist net, we assign it a static partition score $S _ { p } { \mathrm { : } }$

$$
S _ { p } = \frac { \overline { { \mathrm { { H P W L } } } } _ { \mathrm { { n e t } } } } { d } \times \frac { K } { N }\tag{4}
$$

where $\overline { { \mathrm { H P W L } } } _ { \mathrm { n e t } }$ is the mean per-net HPWL (baseline HPWL divided by the number of nets), d the die perimeter, K the candidate-set size, and N the number of cells. The first factor is the per-move reward magnitude and the second the fraction of cells acted on per episode.

A high $S _ { p }$ alone does not guarantee a usable signal, since the reward can still be dominated by noise. We thus add a dynamic metric $R _ { m }$ , the reward signal-to-noise ratio over a few short random-policy probes without training:

$$
R _ { m } = \frac { \left| \mu _ { R } \right| } { \sigma _ { R } }\tag{5}
$$

where $\mu _ { R }$ and $\sigma _ { R }$ are the mean and standard deviation of the reward for the episode. A low $R _ { m }$ means noisy rewards and unreliable gradient updates. Algorithm 1 uses both metrics to assign each design to a class (A, B, or C) and its thresholds are calibrated on our three designs and are indicative.

In order to preliminarily evaluate the effectiveness of this approach, we perform an evaluation with the ISPD2015 benchmarks. Table III depicts this evaluation.

Each design lands in a different class, matching its outcome. mgc\_fft\_1 (A) has a high score and a clean reward, so full-graph SAGE is best and forcing a subregion collapses its policy. $\mathrm { m g c \_ p c i \_ b r i d g e } 3 2 \mathrm { \_ } \mathrm { } a \mathrm { \_ } \mathrm { } ( \mathbf { B } )$ has a high score but a noisy reward, full-graph training fails, while subregion training cleans the signal and unlocks the largest gain. mgc\_matrix\_mult\_1 (C) has a low score from limited coverage and embedding collapse, so even subregion training improves it only marginally.

TABLE III  
EVALUATION OF THE PROPOSED DESIGN CLASSIFICATION APPROACH OVER THE ISPD2015 BENCHMARKS.
<table><tr><td>Design</td><td> $S _ { p }$ </td><td> $R _ { m }$ </td><td>Class</td><td>Train  $\Delta \%$ </td><td>Eval ∆%</td></tr><tr><td> $\mathrm { m g c \_ f f t } _ { - } 1$ </td><td> $1 . 1 4 \times 1 0 ^ { - 3 }$ </td><td>9.27</td><td>A</td><td>0.53</td><td>0.20</td></tr><tr><td> $\mathtt { m g c \_ p c i \_ b r i d g e } 3 2 \_ a$ </td><td> $1 . 9 6 \times 1 0 ^ { - 3 }$ </td><td>0.97</td><td>B</td><td>5.85</td><td>0.43</td></tr><tr><td>mgc_matrix_mult_1</td><td> $1 . 4 1 \times 1 0 ^ { - 4 }$ </td><td>4.77</td><td>C</td><td>0.04</td><td>0.03</td></tr></table>

## VII. CONCLUSION

We presented a design space exploration of GNN-PPO for postlegalization placement refinement over 18 configurations across regular and irregular ISPD2015 netlists. Regular designs expose three compounding pitfalls a credit-assignment bottleneck, GNN over-smoothing, and tightly placed cells making subregion training most beneficial on irregular, noise-limited designs and only marginally on the regular mgc\_matrix\_mult\_1, confirming that a netlist’s regularity, reflected in its cell spacing, indirectly indicates how much refinement it can still gain. Guided by this, we proposed a lightweight classification based on the partition score $S _ { p }$ and the reward signal-to-noise ratio $R _ { m }$ that selects full-graph or subregion training from the netlist alone, calibrated on three designs and pending validation on a broader suite.

## REFERENCES

[1] T. Ajayi, V. A. Chhabria, M. Fogac¸a, S. Hashemi, A. Hosny, A. B. Kahng, M. Kim, J. Lee, U. Mallappa, M. Neseem et al., “Toward an open-source digital flow: First learnings from the OpenROAD project,” in Proceedings of the 56th Annual Design Automation Conference, 2019, pp. 1–4.

[2] I. S. Bustany, D. Chinnery, J. R. Shinnerl, and V. Yutsis, “ISPD 2015 benchmarks with fence regions and routing blockages for detailed-routing driven placement,” in Proceedings of the 2015 Symposium on International Symposium on Physical Design, 2015, pp. 157–164.

[3] Y. Lin, S. Dhar, W. Li, H. Ren, B. Khailany, and D. Z. Pan, “DREAMPlace: Deep learning toolkit-enabled GPU acceleration for modern VLSI placement,” in Proceedings of the 56th Annual Design Automation Conference, 2019, pp. 1–6.

[4] C.-K. Cheng, A. B. Kahng, I. Kang, and L. Wang, “RePlAce: Advancing solution quality and routability validation in global placement,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 38, no. 9, pp. 1717–1730, 2019.

[5] A. Mirhoseini, A. Goldie, M. Yazgan, J. W. Jiang, E. Songhori, S. Wang, Y.-J. Lee, E. Johnson, O. Pathak, A. Nova et al., “A graph placement methodology for fast chip design,” Nature, vol. 594, no. 7862, pp. 207–212, 2021.

[6] A. Agnesina, K. Chang, and S. K. Lim, “Parameter optimization of VLSI placement through deep reinforcement learning,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 42, no. 4, pp. 1295–1308, 2022.

[7] H. Ren and M. Fojtik, “Standard cell routing with reinforcement learning and genetic algorithm in advanced technology nodes,” in Proceedings of the 26th Asia and South Pacific Design Automation Conference, 2021, pp. 684–689.

[8] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learning Representations (ICLR), 2017.

[9] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” Advances in Neural Information Processing Systems, vol. 30, 2017.