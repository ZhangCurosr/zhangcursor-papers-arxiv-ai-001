# Fidelity-Aware Scheduling of Quantum Circuits on Multi-QPU Systems

Innocenzo Fulginiti<sup>∗</sup>, Antonio Tudisco<sup>†</sup>, Salvatore Zammuto <sup>∗</sup>, Patrick Hopf <sup>∗</sup> <sup>‡</sup>, Deborah Volpe <sup>§</sup>, Helmut Seidl <sup>∗</sup>, Giovanna Turvani <sup>†</sup>, Robert Wille <sup>∗</sup> <sup>‡</sup>, Christian B. Mendl <sup>∗</sup>, Martin Schulz

<sup>∗</sup>Technical University of Munich, Munich, Germany

<sup>†</sup>Department of Electronics and Telecommunications, Politecnico di Torino, Turin, Italy

<sup>‡</sup>MQSC, Garching near Munich, Germany

<sup>§</sup>Istituto Nazionale di Geofisica e Vulcanologia, Rome, Italy

{antonio.tudisco, giovanna.turvani}@polito.it;

{innocenzo.fulginiti, salvatore.zammuto, patrick.hopf, helmut.seidl, robert.wille, christian.mendl, martin.w.j.schulz}@tum.de; deborah.volpe@ingv.it;

Abstract—High Performance Computing-Quantum Computing (HPCQC) platforms expose multiple Quantum Processing Units (QPUs) that may differ in size, topology, native gates, and noise characteristics. For current noisy devices, errors compound along the compiled circuits quickly, and minimizing them, that is, maximizing the circuits’ execution fidelity, is essential for reliable results. Fidelity depends on the compilation to a specific target device: the same high-level circuit may produce different executables and, therefore, different expected fidelities across QPUs. We present a low-overhead fidelity-aware scheduling framework for multi-QPU systems based on a Graph Neural Network (GNN) that estimates, before compilation, the expected fidelity of each circuit on each available QPU. Then, a tunable scheduler uses these estimates to control the trade-off between execution fidelity and parallelism. Results show that this framework allows for approximating an exhaustive fidelity-based assignment, saving computational resources compared to a brute-force approach that compiles each circuit on every device.

Index Terms—Quantum computing, quantum circuit schedul ing, fidelity maximization, graph neural networks

## I. INTRODUCTION

Quantum computers are becoming part of broader infrastructures, including cloud platforms and High-Performance Computing (HPC) environments [1], [2]. In these settings, users do not necessarily interact with a single isolated quantum processor. Instead, they submit tasks to a shared system that may expose multiple quantum processing units (QPUs). These QPUs may differ in technology, size, connectivity, native gate set, and noise characteristics [1], [3]. This opens opportunities for parallel execution, but also introduces new challenges in workload placement and scheduling [4]–[8]. A natural workload model in such environments is a task composed of multiple circuits. Such a task may contain circuits with different widths, depths, gate compositions, and multi-qubit interaction patterns. This structural diversity makes device selection a circuit-dependent decision: different circuits may be best executed on different QPUs. Indeed, even when different QPUs provide a comparable number of qubits, they may differ in their physical topology, supported native gates, gate error rates, and coherence properties [3]. These hardware characteristics directly affect the resulting compiled circuit. A high-level circuit can lead to substantially different compiled implementations, and, as a consequence, different expected execution fidelities, depending on the selected target device [9]–[11]. This issue is particularly evident in the Noisy Intermediate-Scale Quantum (NISQ) regime, where current quantum devices are affected by gate noise and limited coherence [12]. Therefore, the quality of a circuit execution strongly depends on the accumulated error of the compiled circuit. For this reason, fidelity is a central metric when deciding where a circuit should be executed, particularly in a heterogeneous multi-QPU system, where selecting an unsuitable device can degrade the quality of the result [13]. Fidelity can only be estimated once the circuit has been compiled for a specific device, as it is influenced by the actual gate decomposition and qubit mapping imposed by that hardware [3], [9]. An exhaustive multi-device compilation strategy, in which each circuit is compiled for every available QPU to estimate the expected fidelity using hardware calibration data, introduces substantial overhead. Indeed, circuit compilation is a process that tends to be expensive in terms of runtime costs [3]. Moreover, only one compiled version will ultimately be executed, while all the other versions are generated only to support the assignment decision. On the other hand, optimizing fidelity alone is not always sufficient in a multi-QPU system, as a purely fidelitydriven policy may concentrate circuits on a small subset of devices, leaving other QPUs underutilized.

In this paper, we propose a framework that avoids exhaustive multi-device compilation by determining the most suitable QPU for each circuit before device-specific compilation. The core idea is to exploit a Graph Neural Network (GNN)-based model that, given a high-level quantum circuit, predicts the expected execution fidelity of the post-compilation circuit on each available QPU. These predictions are then used by a tunable scheduler, which assigns circuits to QPUs according to a user-defined setting that controls how strongly the system should prioritize fidelity over parallelism. Once a target device has been selected, the circuit is compiled only for that device, thereby reducing the compilation overhead associated with exhaustive device selection. We evaluate the proposed framework on an emulated multi-QPU environment derived from real IQM superconducting devices. The results show that the GNN predictor estimates post-compilation circuit fidelities with low error, and that the proposed scheduler closely approaches the fidelity of ground-truth-based assignment while avoiding exhaustive compilation across all devices. Moreover, by tuning the fidelity weight, the framework can move from balanced device utilization to fidelity-oriented assignments, exposing a practical trade-off between parallelism and execution quality.

The rest of the paper is organized as follows. Sec. II motivates the need for fidelity-aware and tunable scheduling in multi-QPU systems. Sec. III reviews the relevant literature. Sec. IV presents the proposed framework, including the fidelity prediction model and the tunable scheduling policy. Sec. V describes the emulated multi-QPU environment, the dataset, and the training procedure adopted for the evaluation. Sec. VI reports the obtained results and compares the proposed approach against baseline assignment strategies. Finally, Sec. VII concludes the paper.

## II. MOTIVATION

Consider a heterogeneous High Performance Computing-Quantum Computing (HPCQC) system exposing a set of QPUs with different hardware characteristics. Users submit tasks that consist of batches of k circuits. The goal of the system is to assign these circuits to the available QPUs, preserving high execution fidelity while also exploiting the degree of parallelism requested by the user.

## A. Circuit-Dependent Fidelity

In the NISQ setting, the execution quality of a circuit c on a device d can be quantified in terms of the expected fidelity $f _ { c , d }$ , which can be estimated by accumulating the success probabilities of all operations in the circuit as

$$
f _ { c , d } = \prod _ { g \in G _ { c , d } } \left( 1 - \varepsilon _ { d } ( g , Q _ { g } ) \right)\tag{1}
$$

where $G _ { c , d }$ is the sequence of operations of the circuit c compiled for d, $Q _ { g }$ is the set of qubits on which the gate $g \in G _ { c , d }$ acts, and $\varepsilon _ { d } ( g , Q _ { g } )$ denotes the error rate of gate g acting on $Q _ { g }$ for device d, i.e. the gate error rate for gates and the readout error rate for measurements. Eq. (1) accumulates the calibrated success probabilities of gate and readout operations and, like the Estimated Success Probability (ESP) [13] commonly used for device selection, it neglects idle-time decoherence over the schedule duration, crosstalk between concurrently driven qubits, calibration drift between snapshots, and shot noise. It is therefore an approximation of the fidelity that would be measured on hardware, and is adopted here because it is the quantity that exhaustive multidevice compilation itself computes: the comparison between our predictor and the exhaustive baseline is unaffected by this choice, since both are evaluated against the same target.

Absolute fidelity values should nonetheless be read as relative indicators of device suitability rather than as predictions of measured outcomes.

The expected fidelity of a circuit cannot be estimated from the high-level circuit alone, since it depends on both the device-specific compiled implementation and the hardware characteristics of the target QPU, such as per-qubit and pergate error rates. During the compilation process, the logical circuit must be mapped and routed onto the physical qubits of the selected QPU. Furthermore, quantum gates need to be translated and decomposed into the native gate set supported by that device. As a result, the same high-level circuit can lead to different compiled circuits on different QPUs. A device whose topology matches the circuit connectivity may require fewer routing operations, while another device may introduce additional gates due to limited connectivity or a different native gate set. These device-specific transformations affect gate count and, consequently, accumulated error, which determines the expected execution fidelity. Moreover, even two QPUs sharing the same topology and native gate set can yield substantially different execution fidelities for the same circuit if their noise profiles differ.

## B. Cost of Exhaustive Device Selection

A straightforward method for fidelity-aware assignment is to compile each circuit on each available QPU, evaluate the fidelity of each compiled version, and select the device with the highest fidelity. This strategy provides a strong hardwareaware baseline, since the decision is based on device-specific compiled circuits. However, this approach incurs significant overhead, as quantum circuit compilation is computationally expensive and scales poorly with circuit size, with core subproblems such as qubit mapping and routing known to be NP-hard and often dominating compilation time in practice [3], [10], [11]. Moreover, for a batch of k circuits and D available QPUs, it requires $k \times D$ compilations, although only k compiled circuits are eventually executed. The remaining compiled versions are generated just for the assignment decision. This overhead grows with the number of circuits and the number of available devices. Therefore, reducing this overhead by predicting the expected execution fidelity of each circuit on each QPU before device-specific compilation is particularly beneficial. In this way, the circuit is compiled only for the selected QPU, preserving the idea of fidelity-aware assignment while avoiding compilation attempts for devices that are unlikely to be selected.

## C. Limits of Fidelity-First Scheduling and Parallelism Tradeoffs

The selection of the QPU with the highest predicted fidelity maximizes execution quality at the circuit level, but may not fully exploit a multi-QPU system. If many circuits in a batch are predicted to run best on the same device, a strict fidelityfirst policy assigns most of the workload to that QPU, while other devices remain underutilized. The opposite extreme is also undesirable. A policy that only balances the number of circuits across QPUs, such as round robin, can improve device utilization but ignores device-dependent fidelity. As a result, it may assign circuits to QPUs on which they are expected to execute with substantially lower fidelity. Therefore, fidelity and parallelism represent competing objectives: improving one may reduce the other.

To address this trade-off, our framework exposes a tunable scheduling parameter, i.e., the fidelity weight $w ~ \in ~ [ 0 , 1 ]$ Lower values of w give more importance to parallel execution, while higher values prioritize the predicted fidelity. In the extreme case $\begin{array} { r } { \begin{array} { r c l } { w } & { = } & { 0 , } \end{array} } \end{array}$ , the scheduler optimizes workload distribution across the available QPUs, behaving similarly to a round-robin policy. In the extreme case $w = 1$ , each circuit is assigned to the QPU predicted to provide the highest fidelity. Intermediate values allow the scheduler to trade limited fidelity loss for better distribution across devices.

## III. RELATED WORKS

Scheduling quantum circuits across the QPUs of an HPCQC system based on fidelity estimates is a problem that touches several research areas, including device selection, fidelity estimation, and resource scheduling. However, to the best of our knowledge, none of the existing works provides a complete workflow for assigning a batch of circuits to the available QPUs based on pre-compilation fidelity estimates, while avoiding the overhead of multi-device compilation.

A closely related direction is automatic quantum device selection before compilation. In [14]–[16], it is studied how quantum computers can be selected from the input circuit before executing a full compilation workflow to maximize various metrics, including fidelity. These works share our goal of avoiding expensive brute-force compilation across multiple devices. However, they mainly focus on selecting a target device for an individual circuit, treating the problem as a classification task that returns the device with the highest fidelity. In contrast, our framework proposes a scheduler that exploits a Machine Learning (ML) model to estimate the expected fidelity of each circuit on every available QPU, dispatching circuits according to both fidelity information and QPU workload.

A previous work, [omitted for double blind, paper accepted], presents a GNN-based model that predicts the ESP of a quantum circuit for multiple QPUs. It has been shown that a graph representation of the uncompiled circuit yields more accurate estimates than feature-based regressors. However, that work is confined to the estimation task, missing all aspects related to dispatching the circuit across different QPUs. In addition, the model employed here additionally refines the graph encoder with a richer readout, in which the mean, sum, and maximum of the node embeddings are concatenated rather than reduced through a single pooling function.

ML has also been explored for predicting the reliability of quantum circuits. The work in [17] uses a graph transformer to estimate circuit fidelity from graph representations of quantum circuits. On a different line, [18] uses LSTM networks to model circuit fidelity from tokenized gate sequences. These approaches show that ML models can approximate expensive fidelity estimation procedures and capture non-trivial properties of noisy circuit execution. Our work builds on this general idea, but uses fidelity prediction as part of a resource management pipeline to drive scheduling decisions across heterogeneous QPUs.

Scheduling and resource management have received increasing attention as quantum hardware becomes accessible through shared infrastructures. In [4], resource management for quantum clouds is proposed, considering features like queue times, fidelity trends, and calibration constraints. In [7], the authors present a tool that manages quantum resources through hardware-agnostic execution, error mitigation, multiprogramming, and scheduling. The work in [6] formulates fidelity-aware orchestration across heterogeneous quantum nodes using deep reinforcement learning, balancing execution fidelity, given an already compiled circuit, and execution time. The authors of [5] address multi-device scheduling for variational quantum algorithms by distributing different optimization phases across devices to reduce queueing delays and improve resource usage. These systems address important scheduling and orchestration challenges in quantum HPCQC systems. In contrast, our framework performs device selection before device-specific compilation, predicting post-compilation fidelity directly from the high-level circuit representation rather than from compiled instances across all candidate QPUs.

Other works exploit parallelism by executing multiple circuits concurrently on a single quantum processor, partitioning physical qubits so that multiple programs can run simultaneously while accounting for fidelity and resource utilization [19], [20].

Finally, [21] addresses the execution of large quantum circuits on resource-constrained devices by exploiting distributed quantum circuit cutting for hybrid quantum-classical HPC systems. Therefore, they decompose large circuits into smaller subcircuits that can be distributed and executed across heterogeneous resources.

Overall, our work differs from prior approaches by combining pre-compilation fidelity prediction with tunable multicircuit scheduling in a heterogeneous multi-QPU setting. The main contributions of this paper are:

• an end-to-end framework that integrates pre-compilation fidelity estimation with multi-circuit allocation;

• a tunable scheduling policy that assigns batches of independent circuits to heterogeneous QPUs through a single parameter $w ~ \in ~ [ 0 , 1 ]$ , having round-robin and fidelity maximization as its two extremes;

• an experimental evaluation on an emulated multi-QPU environment derived from real IQM superconducting devices.

## IV. METHODOLOGY

As previously mentioned, this work aims to maximize the overall fidelity of a batch of circuits while improving resource utilization. To this end, we develop a framework comprising three main steps: a pre-processing step that translates the input circuits into a format suitable for our model (Sec. IV-A); an ML model, specifically a GNN, that estimates the fidelity of the circuits on the available QPUs prior to compilation (Sec. IV-B); a scheduler that, based on the model’s predictions, assigns the circuits in the batch to the available QPUs according to the selected policy (Sec. IV-C). An overview of the end-to-end pipeline is shown in Fig. 1.

## A. Pre-processing

Before a quantum circuit can be processed by a GNN model, it must pass through a pre-processing step in which it is converted into a graph representation with numerical attributes, as shown in Fig. 2. Quantum circuits can be naturally translated into a graph formulation as Directed Acyclic Graphs (DAGs), where each node represents a quantum gate or operation and each directed edge encodes a data dependency between operations, reflecting the order in which they must be executed. This representation preserves the structure of the circuit in a form that can be directly exploited by our GNN model. Once the DAG is constructed, each node is mapped to a fixed-length numerical feature vector of 40 elements. Specifically, the first 28 elements encode the quantum operation type in one-hot format, following the QASM3 [22] gate set. The subsequent 6 elements represent the sine and cosine of the rotation angle parameters associated with the gate, to allow encoding the angle parameters and their periodicity into a [−1, 1] interval. The remaining 6 elements capture structural properties of the node, namely the number of qubits involved in the operation, the number of control qubits, the number of angle parameters, a binary flag indicating whether the gate lies on the deviceagnostic critical path (i.e., the longest path in the DAG), the number of predecessors, and the number of successors.

## B. Model Definition

Once the circuit has been translated as a numerical DAG, it is fed to a GNN model. This model consists of a graph encoder, which processes the input graph by aggregating information from neighboring nodes, and an MLP, which predicts the fidelity of the circuit on the target devices.

The graph encoder consists of a set of SAGE convolutional layers [23] organized into two distinct stacks: an initial series of SAGE convolutional layers applied directly to the input graph, and a subsequent set arranged in a residual configuration. SAGE convolutional layers are adopted to enable inductive learning, allowing the model to generalize to unseen quantum circuits. Each convolutional layer is then followed by a normalization and regularization block, consisting of a Graph Normalization layer, a Leaky ReLU activation function, and a dropout layer. This sequence has a triple effect: stabilizes training, introduces non-linearity, and mitigates overfitting.

To fully exploit the directional structure of the DAG, an optional technique known as bidirectional message passing can be employed, in which each convolutional layer is applied in parallel to both the original graph and its reversed counterpart. When enabled, the forward direction captures the natural flow of quantum operations, from inputs to outputs, while the reverse direction allows each node to aggregate information about the downstream consequences of its operation, i.e., how its output influences the gates that follow. Whether to enable this bidirectional aggregation is a hyperparameter of the model.

TABLE I: Hyperparameter Search Space.
<table><tr><td>Hyperparameter</td><td>Range</td><td>Best</td></tr><tr><td>hidden_dim</td><td>[32, 256]</td><td>142</td></tr><tr><td>num_conv_wo_resnet</td><td>[1, 3]</td><td>3</td></tr><tr><td>num_resnet_layers</td><td>[1, 9]</td><td>1</td></tr><tr><td>dropout</td><td>{0.0, 0.1, 0.2}</td><td>0.0</td></tr><tr><td>bidirectional</td><td>{True, False} (), (32), (64), (128), (256), (512), (64, 32), (128, 32),</td><td>True</td></tr><tr><td>mlp_units</td><td>(128, 64), (256, 32), (256, 64), (256, 128), (512, 256), (512, 128), (512, 64), (512, 32), (128, 64, 32), (256, 128, 64), (512, 256, 128)</td><td>(512, 256, 128)</td></tr><tr><td>1r</td><td> $\{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ </td><td>10-3</td></tr></table>

After all convolutional layers, the resulting node embeddings are aggregated into a single graph-level representation through a mean-max-sum pooling readout. This operator concatenates three distinct aggregations of the node embeddings: their element-wise mean, their element-wise maximum, and their element-wise sum. This design aims to exploit all three advantages of the pooling techniques considered: the mean captures the average structural pattern across the graph, the maximum preserves the most prominent features, and the sum provides a size-aware aggregate that retains information about the overall scale of the circuit.

The optimal model was identified by tuning the hyperparameters outlined in Table I.

## C. Scheduling

We now present our fidelity-aware scheduler that assigns circuits in the incoming batch to one of the available QPU backends. The scheduler exploits the per-device fidelity estimates produced by the GNN to make informed dispatch decisions while simultaneously controlling the distribution of workload across devices.

1) Fidelity-aware device selection: We formulate device selection as a sequential assignment over a unified circuit queue shared by all backends.

Let $\mathcal { D } = \{ d _ { 1 } , \ldots , d _ { D } \}$ denote the set of available devices, $\hat { f } _ { c , d } \in [ 0 , 1 ]$ the GNN-predicted fidelity of circuit c on device $d , \ell _ { d }$ the number of circuits already assigned to d in the current dispatch round (load), and $\begin{array} { r } { L = \sum _ { d ^ { \prime } } \ell _ { d ^ { \prime } } } \end{array}$ the total number of circuits dispatched so far. Devices compete on their current load share

$$
s _ { d } = { \left\{ \begin{array} { l l } { \ell _ { d } / L , } & { L > 0 , } \\ { 0 , } & { L = 0 . } \end{array} \right. }\tag{2}
$$

Circuit c is assigned to the device maximizing a composite score:

$$
d ^ { * } ( c ) = \arg \operatorname* { m a x } _ { d \in \mathcal { D } } ~ \boldsymbol { w } \cdot \hat { \boldsymbol f } _ { c , d } - ( 1 - \boldsymbol w ) \cdot \boldsymbol s _ { d }\tag{3}
$$

![](images/e4081a2eedb38b8f012917278dd5b43fbb91baedc717f3cd810858bedbd35352.jpg)  
Fig. 1: End-to-end pipeline of the proposed framework. A user-submitted batch of k high-level quantum circuits is converted into Directed Acyclic Graphs with per-node feature vectors and fed to the GNN fidelity predictor, which produces an estimated fidelity matrix $\hat { F }$ over all $k \times D$ circuit-device pairs. The tunable scheduler, parameterized by the fidelity weight w, routes each circuit to one of the available QPU backends; each assigned circuit is then transpiled against the target device’s native gate set and connectivity, and then executed on the corresponding backend.

![](images/0a8b5468d88593a4ee7c31fa0b6f16a7d3898b3af237b5ce914a78dde1467997.jpg)  
Fig. 2: Conversion from circuit to DAG. A quantum circuit (a) is translated into a DAG (b), whose nodes are gates and measurements and whose edges encode data dependencies; each node is then mapped to the feature vector of (c), shown for a CX gate.

where $w \in [ 0 , 1 ]$ is the fidelity weight. Predicted fidelities already lie in [0, 1] and load shares are fractions, so both terms share a common scale without further normalization. Retaining the raw fidelities is deliberate: it preserves the magnitude of each circuit’s device preference, so that a device holding a 0.001 fidelity edge barely outbids the load term whereas a 0.3 edge dominates it, and circuits whose preferences differ in strength switch device at correspondingly different weights. $\mathbf { A } \mathbf { t } \ w = 0$ the score reduces $\operatorname { t o } - s _ { d } .$ , selecting the least-loaded device and recovering a uniform round-robin distribution; at $w = 1$ it reduces to ${ \hat { f } } _ { c , d } ,$ selecting the device with the highest predicted fidelity.

Interpretation of w: Because the penalty grows with the actual share imbalance rather than with a normalized rank, dispatching a further circuit to an already-loaded device becomes progressively more expensive. Devices therefore fill until the marginal share gap offsets the fidelity gap: at equilibrium, for any two devices a, b,

$$
s _ { a } - s _ { b } \approx { \frac { w } { 1 - w } } \Bigl ( \bar { f } _ { a } - \bar { f } _ { b } \Bigr ) , \qquad \bar { f } _ { d } = { \frac { 1 } { k } } \sum _ { c = 1 } ^ { k } \hat { f } _ { c , d } .\tag{4}
$$

where $\bar { f _ { d } }$ is the mean fidelity of the batch of circuits assigned to the device d. The ratio $w / ( 1 - w )$ thus acts as an explicit exchange rate between fidelity advantage and load share: a device may accumulate $w / ( 1 - w )$ units of additional load share per unit of fidelity advantage. Consequently, circuits do not all change device at the same value of w. Each circuit switches at the weight at which its own fidelity advantage outweighs the load penalty, so the allocation changes gradually over the entire range $w \in ( 0 , 1 )$ , as confirmed experimentally in Sec. VI-B. This change is not equally fast at all weights. The exchange rate $w / ( 1 - w )$ grows slowly for small w and diverges as w approaches 1: circuits with a pronounced device preference are therefore reassigned already at low weights, which is where most of the attainable fidelity gain is obtained, while the remaining load balance is given up only close to w = 1. At w = 1 the load term disappears entirely and the assignment reduces to a pure fidelity argmax. This behavior at the endpoint follows directly from ignoring the load and is not an artifact of the scoring function.

2) Dispatcher integration: The scheduling policy described above is embedded in a Quantum Meta-Scheduler (QMS) that orchestrates the full circuit lifecycle within an HPC allocation. The QMS operates a continuous dispatch loop over a unified circuit queue shared across all available backends. Fig. 3 illustrates the logic of a single dispatch tick.

At each tick, the QMS broker inspects the system state and constructs an immutable snapshot capturing two categories of information: (i) the set of pending circuits in the unified queue, ordered by arrival time, and (ii) the runtime state of each registered backend, including the number of currently executing circuits and the latest hardware calibration data.

This snapshot is passed to the multi-device scheduling policy, which returns a device-to-circuit mapping: for each device, an ordered list of circuit identifiers to dispatch. The contract requires that each circuit appears in at most one device’s list. In the fidelity-aware policy, the iteration over the pending circuits proceeds sequentially: for each circuit, the per-device fidelity estimates are retrieved from the GNN (either pre-computed and attached to the circuit metadata, or obtained via live inference at decision time) and the scoring function of Eq. (3) is applied against the load shares of Eq. (2) accumulated at the moment of each individual assignment. Since the load counter is updated after every assignment, the load penalty adapts within the same tick: as a preferred device accumulates load share, subsequent circuits with weaker device preferences are naturally redistributed to less-loaded backends.

Once the mapping is determined, the QMS dispatches each assigned circuit to its target backend through three sequential steps: (i) transpilation against the device’s native gate set and coupling map, (ii) submission of the transpiled circuit for execution, and (iii) asynchronous polling for results. Because each circuit is transpiled only for the device selected by the policy, the framework avoids the overhead of compiling every circuit for all available QPUs: the core efficiency gain over exhaustive device selection.

This architecture decouples the scheduling logic from the execution infrastructure: any policy conforming to the multidevice interface can be substituted without modifying the dispatch or execution pipeline, enabling systematic comparison of alternative strategies under identical conditions.

## V. EXPERIMENTAL SETTINGS

To verify the effectiveness of the proposed approach, we conduct a twofold analysis. First, we evaluate the predictive performance of the trained GNN model in terms of fidelity estimation accuracy. Second, we assess the practical applicability of the model by integrating it into a quantum circuit scheduling framework, where it is employed to guide the assignment of circuits to the most suitable target devices.

![](images/960efdffe11410e51d83b49ac9ab5ca1ff7969c7a9dc20e577d4c2e4688e614e.jpg)  
Fig. 3: Logic of a single dispatch tick within the QMS. The broker constructs an immutable snapshot of the queue and per-device state, invokes the scheduling policy to produce a device-to-circuit mapping, and dispatches each circuit to its assigned backend for transpilation and execution. The dashed arrow indicates the tick-based cycle.

We evaluate our framework under realistic conditions by emulating the Leibniz-Rechenzentrum (LRZ) quantum computing environment, which provides access to multiple QPUs. Specifically, we consider two IQM superconducting devices: one comprising 53 qubits (EQE1) and the other 20 qubits (QExa20). For each device, the per-gate, per-qubit fidelity values are available and used to characterize the noise profile of the hardware. We selected these two devices because their noise profiles are publicly available, which is essential for the development of this work. Despite sharing the same underlying technology, the two devices exhibit different noise profiles, resulting in diverse fidelity performance across quantum circuits. The proposed fidelity prediction model is not limited to the devices considered in this work, but can be readily extended to any HPCQC system integrating devices with different technologies and gate sets and regenerating the GNN model, as similarly demonstrated in [16], [omitted for double blind, paper accepted].

Device Partitioning: To further stress-test the scheduling capabilities of our framework and to better observe the effects of parallelization across a larger pool of devices, the 53-qubit device is partitioned into two independent sub-devices of 26 and 27 qubits, by performing a horizontal cut through the center of its layout. Seven cross-boundary edges are severed by the cut. Both sub-devices inherit the per-qubit calibration data from the parent device, re-indexed to contiguous qubit labels. This effectively simulates a scenario with three available QPUs and allows us to observe how the scheduler distributes circuits across devices of comparable size but different noise characteristics. Tab. III summarizes the noise characteristics of the three devices in terms of mean, minimum, and maximum gate fidelities for both single- and two-qubit operations, while Fig. 4 illustrates the respective coupling maps.

TABLE II: Number of circuits per category.
<table><tr><td>Category</td><td>#</td><td>Category</td><td>#</td></tr><tr><td>vqe_two_local</td><td>3449</td><td>qpeinexact</td><td>19</td></tr><tr><td>vqe_real_amp</td><td>2861</td><td>wstate</td><td>19</td></tr><tr><td>vqe_su2</td><td>2858</td><td>graphstate</td><td>18</td></tr><tr><td>qaoa</td><td>2706</td><td>bmw_quark_copula</td><td>10</td></tr><tr><td>randomcircuit</td><td>1234</td><td>draper_qft_adder</td><td>10</td></tr><tr><td>qft</td><td>760</td><td>modular_adder</td><td>10</td></tr><tr><td>qnn</td><td>760</td><td>cdkm_ripple_carry_adder</td><td>9</td></tr><tr><td>ae</td><td>758</td><td>full_adder</td><td>9</td></tr><tr><td>qftentangled</td><td>746</td><td>half_adder</td><td>8</td></tr><tr><td>iqpe</td><td>570</td><td>vbe_ripple_carry_adder</td><td>6</td></tr><tr><td>grover</td><td>62</td><td>ghz_dynamic</td><td>3</td></tr><tr><td>hhl</td><td>44</td><td>multiplier</td><td>3</td></tr><tr><td>bmw_quark_cardinality</td><td>19</td><td>qwalk</td><td>3</td></tr><tr><td>bv</td><td>19</td><td>rg_qft_multiplier</td><td>3</td></tr><tr><td>dj</td><td>19</td><td>hrs_cumulative_multiplier</td><td>2</td></tr><tr><td>dynamic_qft</td><td>19</td><td>seven_qubit_steane_code</td><td>1</td></tr><tr><td>ghz</td><td>19</td><td>shors_nine_qubit_code</td><td>1</td></tr><tr><td>qpeexact</td><td>19</td><td></td><td></td></tr></table>

Total (35 categories)

![](images/25399162ba0b7c16dc4c06a8092cc756e60c4d0421927b2f47198dd359600225.jpg)  
(a) EQE1 Top

![](images/280ce65b437cc9511a0c2fa68e7a7a89f021460b092fdd647bc803b4e3173f99.jpg)  
(b) EQE1 Bottom

![](images/9c84293cddef039eaccf1caf3e1c135d2fe6f41b35873dd36582bca584ec7a6e.jpg)  
(c) QExa20  
Fig. 4: Coupling maps of the target devices.

Dataset: For this work, we have adopted a dataset of 17056 circuits drawn from MQT Bench [24], by augmenting the base suite with alternative state-preparation routines, ansatz topologies and oracle targets. Table II shows the number of circuits for each category. For each circuit, device-specific compilation is performed on all target devices using the Qiskit compiler, setting its optimization level to 2. The groundtruth fidelity label $f _ { c , d }$ for each circuit-device pair is then computed from hardware calibration data as shown in Eq. (1). The dataset was partitioned into training, validation, and test sets using stratified sampling ensuring that the distribution of circuit quality is preserved across all splits. Specifically, 70% of the circuits form the training pool, which is further divided 80/20 into the actual training set and the validation set used during hyperparameter search. The remaining 30% constitute the test set, never seen during optimization.

Model’s Hyperparameters: The optimal model is selected through Bayesian optimization performed with Optuna [25] over 50 trials. Each trial consists of training a full model for up to 1000 epochs with early stopping, with the patience set to 50 epochs. The objective function minimized by Optuna is the minimum validation Mean Squared Error (MSE) reached across all epochs of the trial. The corresponding hyperparameters are reported in Tab. I.

TABLE III: Gate fidelity statistics per device.
<table><tr><td></td><td colspan="3">1-qubit fidelity (%)</td><td colspan="3">2-qubit fidelity (%)</td></tr><tr><td>Device</td><td>Mean</td><td>Min</td><td>Max</td><td>Mean</td><td>Min</td><td>Max</td></tr><tr><td>EQE1_Top</td><td>99.92</td><td>99.17</td><td>99.97</td><td>96.23</td><td>81.43</td><td>98.88</td></tr><tr><td>EQE1_Bottom</td><td>99.94</td><td>99.59</td><td>99.98</td><td>96.78</td><td>84.31</td><td>99.13</td></tr><tr><td>QExa20</td><td>99.61</td><td>96.29</td><td>99.89</td><td>98.26</td><td>94.63</td><td>99.16</td></tr></table>

![](images/c36411b814f9033ba5b9e429390a0773c555ed2e98d3085e9d20bcf6609cb3eb.jpg)  
Fig. 5: RMSE of the model on the test set, grouped by target fidelity.

## VI. RESULTS

In this section, we evaluate the effectiveness of the proposed approach along two complementary dimensions: the accuracy of our model in predicting the fidelity, and the ability of the scheduler to dispatch circuits across the available QPUs.

The code is available is publicly available at https://github. com/1nnocenzo/pred-distr-tool.git.

## A. Fidelity Estimation Model

The results of the optimal model on the test set are reported in Tab. IV. The results show that the model is particularly effective in predicting circuit fidelity, with low Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), on this split limited to 1.04% and 2.00%, respectively. To further characterize the model’s predictive behavior, we report the RMSE grouped by target fidelity in Fig. 5. For each group, the RMSE remains below 4%, with maximum errors falling for circuits whose fidelities are in the range [0.1, 0.4]. The model performs particularly well on the QExa20 device, where the RMSE of the fidelity prediction is about 27.5% lower than the overall RMSE.

To further assess the quality of the model, we compare the predicted and target fidelities on the test set for the three target devices, shown in Figure 6. Each panel reports the regression performance for a single device, with the dashed diagonal denoting perfect agreement. Across the full fidelity range, the predictions cluster tightly around the identity line, indicating that the model captures device-specific fidelity trends with little systematic bias. This confirms that the model can accurately predict circuit fidelities spanning the entire range of observed values.

TABLE IV: Regression performance metrics of the GNN model on the test set, reported per device and overall.
<table><tr><td>Device</td><td>MAE</td><td>MSE</td><td>RMSE</td><td> $R ^ { 2 }$ </td></tr><tr><td> $\mathrm { E Q E l _ { T o p } }$ </td><td>0.0105</td><td> $4 . 2 1 6 \times 1 0 ^ { - 4 }$ </td><td>0.0205</td><td>0.9959</td></tr><tr><td> $\mathrm { E Q E l _ { B o t } }$ </td><td>0.0121</td><td> $5 . 6 7 3 \times 1 0 ^ { - 4 }$ </td><td>0.0238</td><td>0.9948</td></tr><tr><td>QExa20</td><td>0.0086</td><td> $2 . 1 1 7 \times 1 0 ^ { - 4 }$ </td><td>0.0145</td><td>0.9977</td></tr><tr><td>Overall</td><td>0.0104</td><td> $\mathbf { 4 . 0 0 2 \times 1 0 ^ { - 4 } }$ </td><td>0.0200</td><td>0.9962</td></tr></table>

![](images/b328b805067275feeff1f27a6eedd086f12137a62b5f1dcc7006687530ae7192.jpg)

![](images/ee5e8c2f2236265e7b2a0a708d4c84a49247b8a33e0bcb9bad95887b593127b6.jpg)

![](images/d57d6a76f41a5950d05ad806735073da92ce25153d6715d00fe1335b0a0afa7f.jpg)  
Fig. 6: Predicted vs. ground-truth fidelity for each circuit–device pair on the test set.

## B. Scheduling

We evaluate the scheduling policy in a simulated multidevice setting, using the ground-truth fidelities of Eq. (1) evaluated on the compiled test circuits and GNN-predicted fidelities obtained from the high-level circuits. Four policies are compared:

• Oracle: assigns each circuit to the device with the highest ground-truth fidelity, disregarding load;

• GT-Weighted: applies Eq. (3) with ground-truth fidelities, providing an upper bound for the predictiondependent policy;

• GNN: applies the same scoring formula with GNNpredicted fidelities, while all metrics are evaluated against ground truth;

• Round-Robin: distributes circuits across devices without fidelity information.

The GNN and GT-Weighted policies are swept over an 11-point grid $w \in \{ 0 . 0 , 0 . 1 , . . . , 1 . 0 \}$ on the full test split of 5117 circuits, dispatched to the three target devices (EQE1 Top, EQE1 Bottom, QExa20). Oracle and Round-Robin are weight-independent by construction and serve as constant references throughout.

1) Parallelism evaluation: The three-device configuration enables concurrent circuit execution on EQE1 Top, EQE1 Bottom, and QExa20. However, the Oracle reveals a structural bias in the dataset: 55.6% of circuits achieve their highest ground-truth fidelity on EQE1 Bottom, with 35.1% on QExa20 and only 9.3% on EQE1 Top (load-balance coefficient of variation $\mathrm { C V } ~ = ~ 0 . 6 9 7 )$ . A purely fidelityoptimal policy would therefore leave EQE1 Top substantially underutilized, reducing the effective parallelism gained by device partitioning.

Figure 7 visualizes how each policy distributes the 5117 circuits across the three backends as the fidelity weight w varies. At $w = 0$ , both the GNN and GT-Weighted policies reduce to a uniform distribution (CV = 0.0003, because 5117 is not divisible by 3), fully exploiting all three backends; the GNN bars at $w = 0$ exactly match the Round-Robin reference, reflecting the round-robin fallback built into the policy. As w increases, circuits with a clear device preference migrate toward their best-fidelity backend, and the share equilibrium of Eq. (4) makes this migration gradual: the GNN reaches $\mathrm { C V } ~ = ~ 0 . 0 3 3$ at $w \ = \ 0 . 3 .$ , 0.069 at $w \ = \ 0 . 5$ and 0.149 at w = 0.7, still distributing 1652, 1982 and 1483 circuits across the three backends at the latter operating point. Only in the immediate neighborhood of $w = 1$ , where the load term of Eq. (3) vanishes, does the allocation collapse onto the Oracle-like profile $\mathrm { ( C V = 0 . 8 8 4 }$ , with 3375 circuits on EQE1 Bottom).

TABLE V: Cross-policy metrics (GNN vs. GT-Weighted) across the fidelity weight sweep, shown on a reduced grid for compactness. r¯: mean regret; Med. r: median regret; $r { = } 0 \colon$ fraction of circuits with zero regret. The latter slightly exceeds the agreement rate because a minority of circuits attain identical ground-truth fidelity on two devices, so a disagreement in device choice costs nothing.
<table><tr><td>w</td><td>Agree (%)</td><td>r</td><td>Med. r</td><td>Max r</td><td>r=0 (%)</td></tr><tr><td>0.00</td><td>100.0</td><td>0.0000</td><td>0.0000</td><td>0.000</td><td>100.0</td></tr><tr><td>0.30</td><td>68.8</td><td>0.0031</td><td>0.0000</td><td>0.2275</td><td>69.0</td></tr><tr><td>0.50</td><td>72.8</td><td>0.0027</td><td>0.0000</td><td>0.2232</td><td>73.2</td></tr><tr><td>0.70</td><td>77.5</td><td>0.0024</td><td>0.0000</td><td>0.2232</td><td>78.0</td></tr><tr><td>1.00</td><td>80.1</td><td>0.0021</td><td>0.0000</td><td>0.3306</td><td>80.1</td></tr></table>

In a practical setting with finite execution capacity, the Oracle’s concentration, which places 2846 of 5117 circuits on a single device while sending only 474 to EQE1 Top, creates a load imbalance that may leave backends idle while circuits queue on the preferred device. The GNN policy at moderate weights approaches oracle-level fidelity while distributing work far more evenly across all three devices, providing a more practical operating point for throughputsensitive HPCQC workloads.

2) Fidelity results: Table V and Figure 8 summarize the fidelity results. Round-Robin establishes a weight-independent baseline at a mean fidelity of 0.3436, and the Oracle an upper bound of 0.3759. Both the GT-Weighted and GNN policies improve monotonically with w across the whole grid, from the Round-Robin endpoint at $w = 0$ to 0.3759 and 0.3737 respectively at $w = 1$ . The GT-Weighted curve recovers the Oracle exactly at $w = 1$ , confirming that the scoring function of Eq. (3) reproduces optimal device assignments given perfect fidelity knowledge, while the GNN tracks it within 0.0034 at every weight.

The practically relevant quantity is the exchange rate between fidelity and load balance along the curve. Round-robin balances the load almost perfectly $\mathrm { ( C V = 0 . 0 0 0 3 ) }$ but only reaches a mean fidelity of 0.3436, while the Oracle attains 0.3759: the entire fidelity margin that any policy can trade load imbalance for lies in this band. Within it, the proposed policy captures most of the available gain early. $\mathrm { { A t } } \ w = 0 . 5$ it reaches 0.3675 (+7.0% over round-robin) with $\mathrm { C V } = 0 . 0 6 9 ,$ and at $w = 0 . 7$ it reaches 0.3703 (+7.8%) with CV = 0.149. Beyond this point the marginal returns diminish sharply: the last 0.9% of mean fidelity (up to 0.3737 at w = 1.0) is bought only by letting the CV grow to 0.884. From w = 0.5 onwards the policy therefore operates in the regime where it retains nearly all of the fidelity advantage over a static assignment while keeping all three devices in use, which is precisely where a fidelity-aware scheduler is preferable to round-robin.

![](images/8e0508eac5c0ed5c1c02f67f4d9b18948ed66d865c795a1250ec06cac786e17e.jpg)  
Fig. 7: Load distribution across devices for each policy, shown on a reduced grid $w \in \{ 0 . 0 , 0 . 3 , 0 . 5 , 0 . 7 , 1 . 0 \}$ for legibility. Round-Robin (red) is weight-independent and stays uniform across the three backends. The Oracle (green) is also weight independent and exposes the structural bias of the dataset, concentrating 2846 circuits on EQE1 Bottom and only 474 on EQE1 Top. The GT-Weighted (purple) and GNN (blue) policies interpolate between these regimes as w grows: at w = 0 the GNN coincides with Round-Robin (round-robin fallback), and the distribution skews progressively toward the Oracle profile as $w \to 1$

![](images/7b387fc450b2022c94ebd0fb8bff29d67f652aa3a90939578dcc8e2d53f95d11.jpg)  
Fig. 8: Overall mean ground-truth fidelity versus the fidelity weight w, over the 11-point sweep. Round-Robin (0.3436) and the Oracle (0.3759) are weight-independent and bound the attainable range, a span of 0.032 on the [0, 1] fidelity scale; the vertical axis is scaled to that range to resolve the policy curves, and absolute differences should be read against it. At $w = 0$ the GNN falls back to round-robin by design, so the two coincide. For $w > 0$ both prediction-aware policies increase monotonically, the GNN remaining within 0.0034 of the GT-Weighted upper bound at every weight.

To isolate the impact of using predicted fidelities in place of ground-truth values, we compare GNN assignments directly against the GT-Weighted policy, which applies the same scoring formula with exact fidelity knowledge. We report two metrics: the device agreement rate (fraction of circuits assigned to the same device by both policies) and the fidelity regret $r ( c ) = f _ { \mathrm { G T - W } } ( c ) - f _ { \mathrm { G N N } } ( c )$ , measuring per-circuit fidelity loss attributable to prediction error. $\mathbf { A } \mathbf { t } \ w = 0 .$ , both policies reduce to round-robin, yielding 100% agreement and zero regret by construction. For $w > 0$ the agreement rate increases monotonically with the fidelity weight, from 68.8% at $w = 0 . 3$ to 80.1% at $w = 1 \colon$ as the fidelity term comes to dominate the score, both policies converge on the same per-circuit ranking, and the residual disagreement isolates the effect of prediction error alone. Mean regret decreases monotonically over the same range, from 0.0031 to 0.0021, with a median of exactly zero throughout, confirming that the large majority of circuits land on the GT-Weighted–optimal device. The residual ∼20% disagreement at $w = 1$ is concentrated on circuits whose inter-device fidelity spreads are narrow, so that a suboptimal device choice is inexpensive: the median regret is zero, the 95th percentile is 0.011, and only 1.15% of the circuits incur a regret above 0.05. The distribution is nonetheless longtailed, and the maximum observed regret reaches 0.331 on an individual circuit, so the aggregate figures should not be read as a per-circuit bound.

3) Scheduling overhead: The central motivation of this work is avoiding the k×D compilations required by exhaustive device selection, so we measure both paths directly. Figure 9 reports the per-circuit wall-clock breakdown, as a function of the two-qubit gate count. Timings were collected on the same test split, excluding QASM parsing from both sides and constructing the pass managers and the model once outside the measurement loop, so that only the per-circuit cost is compared. Of the 5117 test circuits, all are timed except for the Grover instances, which are omitted because their transpiled size exceeds the others by four orders of magnitude and would dominate every aggregate.

Over the whole split, the prediction-based path is faster both in the median and mean with respect to exhaustive compilation. The two additional costs of the fast path, DAG encoding and inference, together amount to 11.57 ms in the median, comparable at $\textit { D } = ~ 3$ to the two compilations it avoids. They are not constant, growing from 6.94 ms at two qubits to 18.27 ms at twenty, but they grow more slowly than the exhaustive cost, and it is this difference in slope that produces a crossover. As Figure 9 shows, the two paths cross at about 35 two-qubit gates: below that the exhaustive route is cheaper, while in the densest bin, at a median of 292 twoqubit gates, exhaustive selection costs 110 ms against 64 ms for the prediction-based path, a 1.72× saving. The same trend holds in the other size measures, with the advantage reaching 1.73× at 20 qubits and 1.65× in the deepest depth quartile. Conversely, on the shallowest quartile, the exhaustive path remains cheaper, and 47% of the circuits in the split are individually faster to compile exhaustively than to predict. The gain scales with $D ,$ since the exhaustive cost is linear in the number of devices while the prediction cost is not, so the threedevice configuration studied here represents a lower bound on the achievable saving.

Training-label generation is itself an exhaustive process and must be accounted for. Compiling and scoring on all three devices the ∼ 12 thousand timed circuits that fall outside the test split costs 904.5 s in total. Against a mean saving of 9.96 ms per dispatched circuit, this one-off cost is amortized after approximately $9 . 1 \times 1 0 ^ { 4 }$ dispatched circuits, beyond which the framework yields a net saving over exhaustive selection at $D = 3 ;$ the break-even point falls as D grows.

Two caveats bound these figures. Inference is measured at batch size one on CPU, matching a per-circuit dispatch decision but not the batched regime in which a meta-scheduler would realistically operate; and the encoding stage is a pure-Python DAG traversal. Both are implementation rather than algorithmic costs, so the reported speedup should be read as a conservative estimate.

## VII. CONCLUSIONS

This work presents a fidelity-aware scheduling framework for heterogeneous multi-QPU systems. At its core, a GNN model exploits the DAG representation of quantum circuits to accurately predict the expected fidelity of each circuit on each target device, prior to compilation. The resulting perdevice fidelity estimates are then fed to a tunable scheduler, which assigns each circuit to a target QPU according to a configurable policy that allows the user to control the trade-off between execution fidelity and workload parallelism across the available devices. The experimental evaluation demonstrates the effectiveness of the proposed approach: the framework achieves a mean batch fidelity improvement over a round-robin baseline, while avoiding the compilation overhead associated with exhaustive multi-device selection.

As future work, a natural next step is to validate the proposed framework on real multi-QPU infrastructures with concrete quantum circuit batches to study its behavior on the application level. Another direction is to evaluate the model across a broader set of quantum technologies and native gate sets, assessing how the learned circuit representations generalize across substantially different hardware platforms.

Device-selection cost: exhaustive vs GNN-predicted  
![](images/de6af5abac3ba2d691d73d23ad1330c0c445ce5eea80bee95450752e7a7a7de4.jpg)

Fig. 9: Median per-circuit wall-clock cost of the two deviceselection strategies against the number of two-qubit gates in the submitted circuit, with inter-quartile bands, on logarithmic axes; circuits are grouped into twelve equal-count bins. Both paths grow with the entangling-gate count, but the exhaustive cost grows faster because it pays the layout and routing effort on all D devices, and the two cross at about 35 two-qubit gates. The encoding and inference overhead (dotted) is the floor of the prediction-based path. The entangling-gate count is used rather than circuit width, total gate count, or depth because it is the quantity that actually drives layout and routing cost: on this split its rank correlation with the exhaustive cost is 0.87, against 0.55 for width, 0.55 for depth and 0.52 for total gate count, and it is the only one of the four under which the binned cost is monotone. Medians are taken within bins, so the separation shown is wider than the aggregate speedup, which is diluted by the overlap of the two distributions.

## ACKNOWLEDGMENT

The work was funded by the Munich Quantum Valley (MQV), which is supported by the Bavarian State Government with funds from the Hightech Agenda Bayern. Furthermore, this research is funded by BMIMI, BMWET, the state of Upper Austria and the State of Tyrol within the COMET module Quantum Algorithm Engineering (FFG Grant no. 923923) managed by Austrian Research Promotion Agency FFG. Furthermore, the authors would like to thank Hossam Ahmed of the Leibniz Supercomputing Center (LRZ) for his invaluable support in providing and interpreting the device data used in this research. The authors acknowledge HPC@PoliTo for providing computational resources. Large Language Models have been used to support the development of the scripts employed in the experimental evaluation and to rephrase parts of the manuscript text. All content has been reviewed and validated by the authors, who take full responsibility for it.

## REFERENCES

[1] Y. Alexeev, D. Bacon, K. R. Brown, R. Calderbank, L. D. Carr, F. T. Chong, B. DeMarco, D. Englund, E. Farhi, B. Fefferman, A. V. Gorshkov, A. Houck, J. Kim, S. Kimmel, M. Lange, S. Lloyd, M. D. Lukin, D. Maslov, P. Maunz, C. Monroe, J. Preskill, M. Roetteler,

M. J. Savage, J. Thompson, M. Troyer, C. Umans, D. Wecker, and M.- H. Yung, “Quantum computer systems for scientific discovery,” PRX Quantum, vol. 2, no. 1, p. 017001, 2021.

[2] T. Beck, A. Baroni, R. Bennink, G. Buchs, E. A. Coello Perez,´ M. Eisenbach, R. Ferreira da Silva, T. S. Humble et al., “Integrating quantum computing resources into scientific HPC ecosystems,” Future Generation Computer Systems, pp. 11–25, 2024.

[3] C. Zhu, X. Wu, Z. Yang, J. Wang, A. Wu, S. Zheng, and X. Wang, “Quantum compiler design for qubit mapping and routing: A crossarchitectural survey of superconducting, trapped-ion, and neutral atom systems,” 2025.

[4] G. S. Ravi, K. N. Smith, P. Murali, and F. T. Chong, “Adaptive job and resource management for the growing quantum cloud,” in 2021 IEEE International Conference on Quantum Computing and Engineering (QCE), 2021, pp. 301–312.

[5] M. Wang, P. Das, and P. J. Nair, “Qoncord: A multi-device job scheduling framework for variational quantum algorithms,” in 2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO), 2024, pp. 735–749.

[6] H. T. Nguyen, M. Usman, and R. Buyya, “QFOR: A fidelity-aware orchestrator for quantum computing environments using deep reinforcement learning,” 2025.

[7] E. Giortamis, F. Romao, N. Tornow, and P. Bhatotia, “QOS: Quantum˜ operating system,” in 19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25). Boston, MA: USENIX Association, 2025, pp. 429–447.

[8] S. Raj, S. Sai, Y. Simmhan, K. Chard, and R. Buyya, “Quantum integrated high-performance computing: Foundations, architectural elements and future directions,” 2026. [Online]. Available: https: //arxiv.org/abs/2604.19814

[9] G. Aleksandrowicz et al., “Qiskit: An open-source framework for quantum computing,” 2019.

[10] A. Molavi, A. Xu, M. Diges, L. Pick, S. Tannu, and A. Albarghouthi, “Qubit mapping and routing via MaxSAT,” in 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), 2022, pp. 1078–1091.

[11] C.-Y. Cheng, C.-Y. Yang, Y.-H. Kuo, R.-C. Wang, H.-C. Cheng, and C.-Y. R. Huang, “Robust qubit mapping algorithm via double-source optimal routing on large quantum circuits,” ACM Transactions on Quantum Computing, vol. 5, no. 3, Sep. 2024. [Online]. Available: https://doi.org/10.1145/3680291

[12] J. Preskill, “Quantum computing in the NISQ era and beyond,” Quantum, vol. 2, p. 79, 2018.

[13] P. Hopf, N. Quetschlich, L. Schulz, and R. Wille, “Improving figures of merit for quantum circuit compilation,” in 2025 Design, Automation & Test in Europe Conference (DATE), 2025, pp. 1–7.

[14] M. Salm, J. Barzen, F. Leymann, and P. Wundrack, “How to select quantum compilers and quantum computers before compilation,” in Proceedings of the 13th International Conference on Cloud Computing and Services Science. SciTePress, 2023, pp. 172–183.

[15] N. Quetschlich, L. Burgholzer, and R. Wille, “MQT Predictor: Automatic device selection with device-specific circuit compilation for quantum computing,” ACM Transactions on Quantum Computing, 2025.

[16] A. Tudisco, D. Volpe, G. Orlandi, and G. Turvani, “Graph neural network-based predictor for optimal quantum hardware selection,” 2025.

[17] H. Wang, P. Liu, J. Cheng, Z. Liang, J. Gu, Z. Li, Y. Ding, W. Jiang, Y. Shi, X. Qian, D. Z. Pan, F. T. Chong, and S. Han, “QuEst: Graph transformer for quantum circuit reliability estimation,” 2022.

[18] Y. Mao, S. Shresthamali, and M. Kondo, “Q-fid: Quantum circuit fidelity improvement with LSTM networks,” Advanced Quantum Technologies, vol. 8, no. 10, p. 2500022, 2025.

[19] L. Liu and X. Dou, “QuCloud: A new qubit mapping mechanism for multi-programming quantum computing in cloud environment,” in 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2021, pp. 167–178.

[20] S. Niu and A. Todri-Sanial, “Enabling multi-programming mechanism for quantum computing in the NISQ era,” Quantum, vol. 7, p. 925, 2023.

[21] M. Tejedor, B. Casas, J. Conejero, A. Cervera-Lierta, and R. M. Badia, “Distributed quantum circuit cutting for hybrid quantum-classical highperformance computing,” 2025.

[22] A. Cross, A. Javadi-Abhari, T. Alexander, N. De Beaudrap, L. S. Bishop, S. Heidel, C. A. Ryan, P. Sivarajah, J. Smolin, J. M. Gambetta, and B. R. Johnson, “OpenQASM 3: A broader and deeper quantum assembly language,” ACM Transactions on Quantum

Computing, vol. 3, no. 3, p. 1–50, 2022. [Online]. Available: http://dx.doi.org/10.1145/3505636

[23] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” 2018. [Online]. Available: https://arxiv.org/ abs/1706.02216

[24] N. Quetschlich, L. Burgholzer, and R. Wille, “MQT Bench: Benchmarking Software and Design Automation Tools for Quantum Computing,” Quantum, vol. 7, p. 1062, 2023, MQT Bench is available at https://mqt-bench.app/.

[25] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” 2019. [Online]. Available: https://arxiv.org/abs/1907.10902