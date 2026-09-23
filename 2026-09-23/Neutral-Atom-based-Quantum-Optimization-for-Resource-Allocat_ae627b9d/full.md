# Neutral-Atom-based Quantum Optimization for Resource Allocation in NOMA Networks

Patatchona Keyela∗ , Remon Polus∗ , Soumaya Cherkaoui∗ , and Ola Ahmad†

∗Department of Computer and Software Engineering, Polytechnique Montréal, Montréal, Québec, H3T 1J4, Canada

Email: {keyela.patatchona, remon.polus, soumaya.cherkaoui}@polymtl.ca

†Thales cortAIx Labs, Montréal, Québec, H3C 0B4, Canada

Email: ola.ahmad@thalesgroup.com

Abstract—In wireless communication networks, many resource optimization problems are nondeterministic polynomialtime hard (NP-hard) due to their combinatorial nature and high computational complexity. Recently, neutral-atom-based quantum computing has emerged as a promising platform for eficiently solving such problems by leveraging quantum superposition and entanglement. However, its application to wireless communication optimization problems remains largely unexplored. In this paper, we investigate the use of neutralatom quantum platforms to solve the maximum access problem (MAP), formulated as a mixed-integer programming task that jointly considers admission control, user clustering, channel assignment, and power allocation in a non-orthogonal multiple access (NOMA)-enabled uplink network. To reduce the computational burden, the MAP is equivalently reformulated as a maximum independent set (MIS) problem in graph theory. This reformulation enables the use of the neutral atom platform based on Rydberg atom arrays, where the MIS problem is naturally encoded into the physical geometry and blockade constraints of the quantum system. Numerical results demonstrate the feasibility and potential of this approach for addressing large-scale wireless resource optimization problems.

Index Terms—Neutral-atom, non-orthogonal multiple access, resource allocation, maximum independent set

## I. Introduction

With the advent of 6G, massive connectivity has become a major usage scenario, requiring connection densities up to a hundred times higher than in 5G [1]. In traditional orthogonal multiple access (OMA) schemes, distinct time, frequency, or code resources are allocated to users to avoid interference. However, the number of users that can be served simultaneously is limited, resulting in constrained system capacity and connectivity [2]. Non-orthogonal multiple access (NOMA) overcomes this by exploiting powerdomain user diversity, allowing multiple users to share the same resources [3]. Signals are superimposed at diferent power levels on the same frequency band, and successive interference cancellation (SIC) at the receiver separates overlapping transmissions [4], improving spectral eficiency, throughput, and user support. Consequently, NOMA is recognized as a key enabler for the Internet of Things (IoT) and the Industry 5.0 era, designed to scale communication networks to support millions of connected devices [5]. In such large-scale deployments, conventional resource allocation and scheduling techniques quickly become inadequate, struggling to eficiently manage the exponentially growing number of users and interactions [6].

Quantum computing ofers eficient solutions to problems intractable for classical methods [7]. Neutral-atom architectures have recently gained attention for their scalability and precise control [8], exploiting the Rydberg blockade, where optically trapped atoms are manipulated between ground |g⟩ and Rydberg |r⟩ states. This enables hardware-eficient encoding of non-deterministic polynomial-time (NP)-hard problems like the maximum independent set (MIS) [9], mapping graph vertices to neutral atoms and enforcing non-adjacency via interatomic interactions [10]. Pasqal demonstrated feasibility by controlling over 1,100 atoms in 2D and 3D arrays with nearly 2,000 optical tweezers [11]. While early results on small graphs are promising, further investigation is needed to assess robustness and scalability [12], [13].

Although quantum computing ofers potential solutions to intractable optimization problems, its application to mobile communications remains largely unexplored [6]. Prior wireless resource allocation work often formulates MIS problems but relies on heuristics [14]–[16]. To our knowledge, neutral-atom-based quantum solutions for MIS in resource allocation and channel assignment have not been studied. Addressing this gap, we investigate the maximum access problem (MAP) in an uplink NOMA network, jointly optimizing admission control, power allocation, and channel assignment. In 6G scenarios with massive device connectivity, the number of users frequently exceeds available resource blocks, making simultaneous support for all users infeasible, even with NOMA [17]. This challenge motivates the design of efective admission control and resource allocation strategies that maximize the number of users served simultaneously. The main contributions of this paper are summarized as follows:

We reformulate the MAP, which involves admission control, user clustering, power control, and channel assignment, into a conflict graph.

The formulated conflict graph is addressed using Pasqal’s Pulser emulator, which is designed to eficiently solve large-scale combinatorial optimization problems through quantum adiabatic evolution.

Numerical results are presented to validate the effectiveness of solving the MIS problem on Pasqal’s neutral-atom emulator, in comparison with a classical optimal solver.

The remainder of this paper is organized as follows. The system model for an uplink NOMA network is presented in Section II. In Section III, the MAP is formulated as a mixed-integer programming problem and subsequently recast as an MIS problem in graph theory. The quantum optimization framework employing neutral-atom arrays to eficiently solve the MIS problem is described in Section IV. Simulation results are presented and discussed in order to evaluate the performance of the proposed quantum approach in Section V. Finally, conclusions are provided and potential directions for future research are discussed in Section VI.

## II. System Model

As shown in Fig. 1, we consider an uplink NOMA network in which a base station (BS) provides wireless coverage to a set of ground users denoted by U, forming L NOMA clusters. Let ${ \cal K } = \{ 1 , 2 , \dots , K \}$ denote the set of uplink channels supported by the BS. We assume perfect time and frequency synchronization among users at the BS, along with an idealized SIC receiver in which interference from previously decoded users is cancelled without residual error. To limit the decoding complexity associated with SIC, each channel is constrained to serve exactly two users simultaneously. Following [1], users within a NOMA cluster are grouped such that the user experiencing weaker channel conditions is allocated a higher transmit power, while the user with stronger channel gain is assigned a lower power level.

Since the number of ground users may exceed the available system capacity, a binary admission variable $a _ { u }$ is introduced, where $a _ { u } \ = \ 1$ indicates that user u is admitted to the network and $a _ { u } \ = \ 0$ otherwise. In addition, a binary channel assignment variable $s _ { u , k }$ is defined as:

$$
s _ { u , k } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ u s e r ~ } u \mathrm { ~ i s ~ a s s i g n e d ~ t o ~ c h a n n e l ~ } k , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

The signal-to-interference-plus-noise ratio (SINR) for user u on channel k is given by

$$
\gamma _ { u , k } = \frac { a _ { u } s _ { u , k } p _ { u } \chi _ { u , k } } { \displaystyle { \sum _ { i \in \mathcal { U } , ~ \chi _ { i , k } < \chi _ { u , k } } s _ { i , k } p _ { i } \chi _ { i , k } + N _ { o } } } ,\tag{2}
$$

where $p _ { u }$ denotes the transmit power of user u, $\chi _ { u , k }$ is the channel power gain from user u to the BS over the $k ^ { \mathrm { t h } }$ uplink channel, and the transmission is impaired by additive white Gaussian noise (AWGN) with zero mean and variance $N _ { 0 }$ . The channel power gain $\chi _ { u , k }$ is given by

$$
\begin{array} { r } { \chi _ { u , k } = | h _ { u , k } | ^ { 2 } d _ { u } ^ { - \eta } , } \end{array}\tag{3}
$$

![](images/2fce3a8cacb3fb29e155b43e49a2db6b0ebb35b5cddb5842d14418fef93b42a4.jpg)  
Fig. 1: An Uplink NOMA system model.

where $h _ { u , k }$ models the small-scale fading and is assumed to be independent and identically distributed (i.i.d.) across users and channels, following a Rayleigh distribution with scale parameter σ. The term $d _ { u }$ denotes the distance between user u and the BS, while η represents the pathloss exponent, which characterizes the rate at which the received signal power attenuates with distance in the considered propagation environment.

Based on Shannon’s theorem, the overall achievable rate for user u is expressed as [18]

$$
R _ { u } = \sum _ { k = 1 } ^ { K } B _ { k } \log _ { 2 } \left( 1 + \gamma _ { u , k } \right) ,\tag{4}
$$

where $B _ { k }$ is the bandwidth allocated per channel.

## III. Problem Formulation

The following optimization problem is formulated to jointly address user admission, power allocation, and subchannel assignment. The objective of this problem is to maximize the number of admitted users while ensuring that each selected user satisfies the minimum data rate requirement, remains within the transmit power limits, and is assigned to at most one frequency channel. The formulation is presented as follows:

$$
\begin{array} { r l r l } { \underset { \mathbf { p } , \mathbf { s } , \mathbf { a } } { \mathrm { m a x i n i z e } } } & { \displaystyle \sum _ { u \in \mathcal { U } } a _ { u } , } \\ { \mathrm { s u b j e c t ~ t o } } & { \displaystyle C _ { 1 } : R _ { u } \geq a _ { u } R _ { \mathrm { m i n } } , } & { \forall u \in \mathcal { U } , } \\ & { \displaystyle C _ { 2 } : 0 \leq p _ { u } \leq P _ { \mathrm { m a x } } , } & { \forall u \in \mathcal { U } , } \\ & { \displaystyle C _ { 3 } : \sum _ { k \in \mathcal { K } } s _ { u , k } \leq 1 , } & { \forall u \in \mathcal { U } , } \\ & { \displaystyle C _ { 4 } : \sum _ { u \in \mathcal { U } } s _ { u , k } = 2 , } & { \forall k \in \mathcal { K } , } \\ & { \displaystyle C _ { 5 } : a _ { u } \in \{ 0 , 1 \} , } & { \forall u \in \mathcal { U } , } \\ & { \displaystyle C _ { 6 } : \upsilon _ { u , k } \in \{ 0 , 1 \} , } & { \forall u \in \mathcal { U } , \forall k \in \mathcal { K } , } \end{array}\tag{5}
$$

where, $\textbf { p } \in \ \mathbb { R } ^ { | \mathcal { U } | }$ denotes the selected transmit power vector, $\bar { \mathbf { S } } \in \{ 0 , 1 \} ^ { | \mathcal { U } | \times | \mathcal { K } | }$ represents the channel assignment matrix, and $\mathbf { a } \in \{ 0 , 1 \} ^ { | \mathcal { U } | }$ corresponds to the admission variable vector. The constraints are interpreted as follows:

$\mathcal { C } _ { 1 } \colon$ The minimum rate $R _ { \mathrm { m i n } }$ is ensured for each admitted user.

C<sub>2</sub>: The transmit power is restricted to $P _ { \mathrm { m a x } }$

$\mathcal { C } _ { 3 } \colon$ Assignment of each user to more than one channel is prohibited.

$\mathcal { C } _ { 4 } \mathrm { : }$ Each channel is required to accommodate exactly two users.

$\mathcal { C } _ { 5 } { \mathrm { - } } \mathcal { C } _ { 6 } \colon$ The decision variables are enforced to be binary.

The problem in (5) is formulated as a mixed-integer programming problem involving discrete variables (a, S) and continuous variables p, making it NP-hard in general [19]. The optimization problem in (5) exhibits key similarities with the MIS problem. Both aim to maximize a target metric—specifically, the number of supported users in (5) and the number of selected vertices in the MIS. Moreover, the MIS constraint that prohibits the simultaneous selection of adjacent vertices corresponds to the NOMA requirement that cluster–channel pairs must be mutually exclusive, i.e., they cannot share users or occupy the same channel. This analogy allows NOMA cluster–channel pairs to be represented as vertices within an MIS-based framework.

## A. Conflict Graph Construction and Cluster Feasibility

Algorithm 1 Conflict Graph Generation   
1: Initialize $\mathcal { G } = ( \mathcal { V } , \mathcal { E } )  ( \emptyset , \emptyset )$   
2: for each channel $k = 1$ to $K$ do   
3: for $( u , \hat { u } ) \in \mathcal { U } \times \mathcal { U }$ do   
4: if {u, uˆ} satisfies $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 2 }$ on ch. k then   
5: Add vertex $v  \{ \{ u , \hat { u } \} , k \}$ to V   
6: end if   
7: end for   
8: end for   
9: for $( v , \hat { v } ) \in \mathcal { V } \times \mathcal { V }$ with $\boldsymbol { v } \neq \boldsymbol { \hat { v } }$ do   
10: if v and vˆ have overlapping users or channel then   
11: Add edge (v, vˆ) to E   
12: end if   
13: end for   
14: return $\mathcal { G } = ( \nu , \mathcal { E } )$

Algorithm 1 constructs a conflict graph $\mathcal { G } ~ = ~ ( \nu , \mathcal { E } )$ ， where each vertex corresponds to a feasible NOMA cluster—consisting of a pair of users—admitted on a given channel without violating constraints $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 2 }$ . Here, V denotes the set of vertices, and E is the set of edges representing admission conflicts arising from shared users or shared channels.

Let Q denote a candidate NOMA cluster operating on channel k. For each user $u \in \mathsf Q$ , the transmit power required to satisfy the minimum rate constraint $R _ { u } ^ { \mathrm { { m i n } } }$ is obtained by inverting the SINR and rate expressions in (2) and (4):

$$
p _ { u } ^ { \mathrm { r e q } } = \left( 2 ^ { \frac { R _ { \mathrm { m i n } } } { B _ { k } } } - 1 \right) \left( \sum _ { \substack { i \in \mathcal { Q } } } \frac { p _ { i } \chi _ { i , k } } { \chi _ { u , k } } + \frac { N _ { o } } { \chi _ { u , k } } \right) .\tag{6}
$$

A NOMA cluster is feasible if, for every user, both the minimum data rate and maximum transmit power constraints must be satisfied:

$$
\big ( R _ { u } \ge R _ { \operatorname* { m i n } } \big ) \wedge \big ( 0 \le p _ { u } \le P _ { \operatorname* { m a x } } \big ) , \quad \forall u \in \mathcal { Q } .\tag{7}
$$

Therefore, each vertex in the conflict graph represents a candidate NOMA cluster–channel pair (Q, k) that satisfies these feasibility conditions. Moreover, an edge exists between two vertices if their corresponding NOMA clusters share at least one user or operate on the same channel, meaning they cannot be admitted simultaneously.

Constructing the conflict graph requires $\mathcal { O } ( | \mathcal { K } | | \mathcal { U } | ^ { 2 } )$ operations to generate the vertices and $\mathcal { O } ( | \mathcal { V } | ^ { 2 } )$ operations to establish the edges. Since edge construction dominates the vertex-generation cost, the worst-case complexity is $\mathcal { O } ( | \mathcal { V } | ^ { 2 } )$ , corresponding to the case where all candidate vertices satisfy the feasibility conditions $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 2 }$ in (7). In practical settings, however, the power and rate constraints in (7) discard many candidate vertices, resulting in a substantially sparser graph and an efective computational cost well below the worst-case bound.

## IV. Neutral-Atom Quantum (NAQ) Optimization Framework

In this section, a neutral-atom quantum (NAQ) optimization approach, proposed to solve the problem formulated in Section III, is presented. A brief overview of Rydberg states is provided, as they enable tunable, distance-dependent interactions between neutral atoms via the Rydberg blockade efect. The process of groundstate preparation through adiabatic evolution is then described, wherein the NAQ platform is evolved from an initial Hamiltonian to a problem-specific Hamiltonian, whose ground state encodes the optimal solution.

## A. Rydberg States and the Rydberg Blockade Efect

Rydberg states are defined as electronic configurations in which an electron is excited to a high principal quantum number n, placing it at a large distance from the atomic nucleus. In these states, the electron is weakly bound, and orbitals that are significantly larger than those of the ground state are formed. This behavior is strongly exhibited by rubidium (Rb) atoms due to their single valence electron, making them particularly suitable for neutral-atom quantum (NAQ) computing platforms. Consequently, Rb has been widely adopted in experimental systems developed by companies such as Pasqal [20].

Excitation to a Rydberg state induces strong van der Waals interactions between atoms [21]. When one atom is excited, these interactions shift the energy levels of neighboring atoms, thereby preventing their simultaneous excitation within a characteristic blockade radius, $R _ { b }$ which is defined as [22]

$$
R _ { b } = \left( \frac { C _ { 6 } } { \hbar \Omega } \right) ^ { \frac { 1 } { 6 } } ,\tag{8}
$$

where $C _ { 6 }$ denotes the van der Waals coeficient, Ω is the Rabi frequency, and ℏ is the reduced Planck constant. Inside this blockade radius, the doubly excited state becomes of-resonant, and laser excitation efectively couples the collective ground state $| g g \rangle$ exclusively to the symmetric singly excited state, expressed as

$$
\left| \psi _ { + } \right. = \frac { 1 } { \sqrt { 2 } } \left( \left| g r \right. + \left| r g \right. \right) .\tag{9}
$$

B. Neutral-Atom-Based Quantum Optimization $\mathrm { A p \mathrm { - } }$ proach

In neutral-atom systems, individual atoms are trapped and manipulated using optical tweezers and tightly focused laser electromagnetic pulses. These applied pulses serve as fundamental tools for encoding and executing quantum operations. In such systems, each atom in this programmable array serves as a qubit. By exciting atoms to Rydberg states, strong dipole-dipole interactions are induced between neighboring atoms, leading to the Rydberg blockade efect. The dynamics of the ensemble of atoms is governed by the following time-dependent Hamiltonian [23]:

$$
H ( t ) = \hbar \Omega ( t ) \sum _ { j } \sigma _ { j } ^ { x } - \hbar \delta ( t ) \sum _ { j } n _ { j } + \sum _ { i \neq j } \frac { C _ { 6 } } { r _ { i j } ^ { 6 } } n _ { i } n _ { j } ,\tag{10}
$$

where $\begin{array} { r } { n _ { j } = \frac { 1 } { 2 } ( 1 + \sigma _ { j } ^ { z } ) } \end{array}$ is the projector onto the Rydbergexcited state and $\sigma _ { j } ^ { \bar { x } }$ and $\sigma _ { j } ^ { z }$ are the Pauli spin operators acting on qubit $j$ . The parameter $\Omega ( t )$ represents the Rabi frequency, which governs the rate of coherent transitions between the ground and excited states, while $\delta ( t )$ denotes the detuning between the laser driving frequency and the atomic resonance. The term $r _ { i j }$ refers to the interatomic distance between atoms i and j. The first two terms in (10) describe atom–field interactions and play a role analogous to transverse and longitudinal magnetic fields in spin models. These terms control the coherent dynamics of each qubit and can be tuned by adjusting the laser’s intensity and frequency. The third term corresponds to the van der Waals interaction between atoms excited simultaneously, which varies with $r _ { i j } ^ { - 6 }$ and results in the Rydberg blockade efect, where the excitation of neighboring atoms within a specific radius is inhibited. By dynamically modulating parameters such as amplitude, Rabi frequency Ω(t), phase, and detuning δ(t) over time, atomic transitions between ground and Rydberg states can be coherently driven.

Based on the Rydberg blockade efect, the geometry of qubit positions can be naturally mapped to a unit disk graph (UDG), wherein each qubit corresponds to a vertex, and an edge exists between two vertices if the

TABLE I: Simulation Parameters
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Symbol</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Number of ground users</td><td rowspan=1 colspan=1>|u|</td><td rowspan=1 colspan=1>20</td></tr><tr><td rowspan=1 colspan=1>Number of channels</td><td rowspan=1 colspan=1> $\overline { { | \mathcal { K } | } }$ </td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>Channel bandwidth</td><td rowspan=1 colspan=1> $B _ { k }$ </td><td rowspan=1 colspan=1>180 kHz</td></tr><tr><td rowspan=1 colspan=1>Rayleigh scale parameter</td><td rowspan=1 colspan=1> $\sigma$ </td><td rowspan=1 colspan=1>√2</td></tr><tr><td rowspan=1 colspan=1>Noise variance</td><td rowspan=1 colspan=1> $N _ { o }$ </td><td rowspan=1 colspan=1>-121 dBm</td></tr><tr><td rowspan=1 colspan=1>Path loss exponent</td><td rowspan=1 colspan=1> $\eta$ </td><td rowspan=1 colspan=1>2.5</td></tr><tr><td rowspan=1 colspan=1>Minimum data rate</td><td rowspan=1 colspan=1> $R _ { \mathrm { m i n } }$ </td><td rowspan=1 colspan=1>1 Mbps</td></tr><tr><td rowspan=1 colspan=1>Maximum transmit power</td><td rowspan=1 colspan=1> $P _ { \mathrm { m a x } }$ </td><td rowspan=1 colspan=1>23 dBm</td></tr><tr><td rowspan=1 colspan=1>Number of iterations</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1000</td></tr></table>

Euclidean distance between the associated qubits is less than $R _ { b }$ . Due to the blockade constraint, pairs of qubits within this radius cannot both occupy the excited state |r⟩. Consequently, finding the ground state of the system Hamiltonian H becomes equivalent to computing the MIS of the induced UDG, representing the largest subset of mutually non-interacting qubits [24].

After the adiabatic evolution is performed on the Pasqal platform, measurements are taken on the resulting quantum state. By executing this process repeatedly, a distribution of bitstrings is generated, where each bitstring encodes a potential solution to the MIS problem. The bitstrings that appear most frequently correspond to the largest independent sets and are therefore indicative of optimal solutions. In the context of the NOMA framework, this approach identifies the maximal set of NOMA cluster– channel pairs that can be admitted simultaneously without violating user or channel constraints. Given that each vertex represents one channel and two users, the number of admitted users is inferred to be twice the number of selected clusters, while the number of occupied channels matches the number of selected clusters, thereby enabling eficient resource allocation and conflict-free operation.

## V. Numerical Results

This section presents numerical results evaluating the quantum-assisted solution on Pasqal’s neutral-atom emulator under various system parameters. Unless stated otherwise, default simulation settings are listed in Table I. In our simulation, users are uniformly distributed within a service cell of radius R and the BS is located at its center point. The network is converted into a conflict graph using Algorithm 1, and the quantum adiabatic protocol is applied on Pasqal’s emulator to obtain the MIS solution, corresponding to the largest set of non-overlapping user pairs assigned to distinct uplink channels. The results are averaged over multiple iterations, illustrating the average number of supported users under varying system conditions.

Fig. 2 illustrates the impact of the path loss exponent, η, on the average number of supported users. As η increases, signal attenuation becomes more severe, which reduces the received signal strength at the BS. Consequently, fewer users are able to meet the required data rate threshold, leading to a decrease in the average number of supported users. Furthermore, we evaluate the performance of the proposed quantum-based approach by comparing it with benchmark schemes, namely the random allocation scheme and the Gurobi-based optimal solver [25]. The results demonstrate that the curves obtained using Pasqal’s emulator exactly match the performance achieved by the Gurobi solver, thereby validating the accuracy of our approach. In addition, both approaches significantly outperform the random allocation scheme by supporting a larger number of users.

![](images/f8165b80305207782236232170945db836d768d6b282d2b679a3e5ebd2378884.jpg)  
Fig. 2: Average number of supported users vs. path loss exponent.

The complexity of the graph is illustrated in Fig. 3. Specifically, Fig. 3a presents the average number of vertices as a function of the path loss exponent, while Fig. 3b depicts the average number of edges. As η increases, reductions in both the vertex and edge counts are observed, indicating that fewer connections among users can be established. This behavior is caused by the increased signal attenuation at higher values of η, which limits the number of links satisfying the quality-of-service constraints. For instance, at a path-loss exponent of 2.5, the resulting conflict graph contains, on average, 1,900 vertices and 495,702 edges across the evaluated realizations. This demonstrates that even a moderately sized network with a limited number of channels and ground users can result in a highly connected conflict graph, highlighting the combinatorial growth of the corresponding MIS problem.

![](images/b27b805aa5f3bb408cede28441b17bb0244e5e95016b0d3c143e9139bb8dee42.jpg)

(a) Average Number of Vertices.  
![](images/c85a3913db680b76fdcc40eaaa6222ab198185c85f8cfdded1c455d4166d5c37.jpg)  
(b) Average Number of Edges.  
Fig. 3: Graph complexity vs. Path loss exponent

Fig. 4 illustrates the average number of supported users as a function of the minimum data rate for diferent values of the Rayleigh fading scale parameter σ. It can be observed that, for all considered values of σ, the average number of supported users decreases as the minimum data rate requirement increases. This behavior is expected since higher data rate demands impose stricter quality-of-service constraints, thereby reducing the number of users that can be simultaneously supported. Moreover, the impact of the fading parameter is clearly evident: larger values of $\sigma \ ( \mathrm { e . g . , } \ \sigma = 5 )$ yield a significantly higher number of supported users compared to smaller values $( \mathrm { e . g . , } \sigma = 0 . 5 )$ This is because increasing σ improves the average channel gain, resulting in higher achievable rates. In contrast, weak signal conditions associated with smaller σ values limit the system’s ability to meet the rate requirements, leading to a rapid decline in the number of supported users.

![](images/2698f6271258df27d05445e21cff87fe857744410339e8a7b2074ae2170e720b.jpg)  
Fig. 4: Average number of supported users vs. minimum data rate.

## VI. Conclusion

In this paper, a neutral-atom quantum optimizationbased approach was introduced to address the MAP in an uplink NOMA network. The problem, originally formulated as a mixed-integer program involving admission control, user clustering, power allocation, and channel assignment, was transformed into an MIS problem. Pasqal’s Pulser emulator, leveraging Rydberg atom arrays, was then employed to solve the MIS problem by naturally embedding the problem structure into the hardware’s physical constraints. Numerical evaluations validated the feasibility and efectiveness of the proposed framework in maximizing user access while satisfying system requirements. These findings highlight the potential of neutralatom platforms for tackling complex wireless resource allocation tasks. Future work will extend this approach to broader classes of optimization problems in large-scale wireless communication systems, including joint resource allocation, interference management, and network slicing.

## VII. Acknowledgement

This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) and Thales Canada Defence & Security (TCDS).

## References

[1] Y. Liu, S. Zhang, X. Mu, Z. Ding, R. Schober, N. Al-Dhahir, E. Hossain, and X. Shen, “Evolution of NOMA Toward Next Generation Multiple Access (NGMA) for 6G,” IEEE J. Sel. Areas Commun., vol. 40, no. 4, pp. 1037–1071, 2022.

[2] Y. Cai, Z. Qin, F. Cui, G. Y. Li, and J. A. McCann, “Modulation and Multiple Access for 5G Networks,” IEEE Commun. Surv. Tutor., vol. 20, no. 1, pp. 629–646, 2018.

[3] Z. Ding, X. Lei, G. K. Karagiannidis, R. Schober, J. Yuan, and V. K. Bhargava, “A Survey on Non-Orthogonal Multiple Access for 5G Networks: Research Challenges and Future Trends,” IEEE J. Sel. Areas Commun., vol. 35, no. 10, pp. 2181–2195, 2017.

[4] S. R. Islam, N. Avazov, O. A. Dobre, and K.-S. Kwak, “Power-Domain Non-Orthogonal Multiple Access (NOMA) in 5G Systems: Potentials and Challenges,” IEEE Commun. Surv. Tutor., vol. 19, no. 2, pp. 721–742, 2016.

[5] A. Ahmed, X. Wang, A. Hawbani, W. Yuan, H. Tabassum, Y. Liu, M. U. F. Qaisar, Z. Ding, N. Al-Dhahir, A. Nallanathan et al., “Unveiling the Potential of NOMA: A Journey to Next-Generation Multiple Access,” IEEE Commun. Surv. Tutor., vol. 27, no. 5, pp. 3099–3164, 2025.

[6] S. Macaluso, G. Geraci, E. F. Combarro, S. Abadal, I. Arapakis, S. Vallecorsa, and E. Alarcon, “Quantum Computing for Large-Scale Network Optimization: Opportunities and Challenges,” IEEE Commun. Mag., vol. 1, no. 64, pp. 116–122, 2026.

[7] W. Zhao, T. Weng, Y. Ruan, Z. Liu, X. Wu, X. Zheng, and N. Kato, “Quantum Computing in Wireless Communications and Networking: A Tutorial-Cum-Survey,” IEEE Commun. Surv. Tutor., vol. 27, no. 4, pp. 2378–2419, 2025.

[8] G. T. Byrd and Y. Ding, “Quantum Computing: Progress and Innovation,” Computer, vol. 56, no. 1, pp. 20–29, 2023.

[9] S. Ebadi, A. Keesling, M. Cain, T. T. Wang, H. Levine, D. Bluvstein, G. Semeghini, A. Omran, J.-G. Liu, R. Samajdar et al., “Quantum Optimization of Maximum Independent Set using Rydberg Atom Arrays,” Science, vol. 376, no. 6598, pp. 1209–1215, 2022.

[10] C. Dalyac, L. Leclerc, L. Vignoli, M. Djellabi, W. d. S. Coelho, B. Ximenez, A. Dareau, D. Dreon, V. E. Elfving, A. Signoles et al., “Graph Algorithms with Neutral Atom Quantum Processors,” Eur. Phys. J. A, vol. 60, no. 9, p. 177, 2024.

[11] G. Pichard, D. Lim, É. Bloch, J. Vaneecloo, L. Bourachot, G.- J. Both, G. Mériaux, S. Dutartre, R. Hostein, J. Paris et al., “Rearrangement of individual atoms in a 2000-site opticaltweezer array at cryogenic temperatures,” Phys. Rev. Appl., vol. 22, no. 2, p. 024073, 2024.

[12] R. Polus, P. Keyela, S. Cherkaoui, and O. Ahmad, “Quantum-Native Resource Allocation for NOMA-Enabled NTNs: A Neutral Atom Approach,” IEEE Open J. Veh. Technol., 2026.

[13] C. Dalyac, L.-P. Henry, M. Kim, J. Ahn, and L. Henriet, “Exploring the impact of graph locality for the resolution of the maximum-independent-set problem with neutral atom devices,” Phys. Rev. A, vol. 108, no. 5, p. 052423, 2023.

[14] A. Khreishah, J. Chakareski, and A. Gharaibeh, “Joint Caching, Routing, and Channel Assignment for Collaborative Small-Cell Cellular Networks,” IEEE J. Sel. Areas Commun., vol. 34, no. 8, pp. 2275–2284, 2016.

[15] D. Zhai and R. Zhang, “Joint Admission Control and Resource Allocation for Multi-Carrier Uplink NOMA Networks,” IEEE Wireless Commun. Lett., vol. 7, no. 6, pp. 922–925, 2018.

[16] C. Guo, B. Liao, L. Huang, P. Zhang, M. Huang, and J. Zhang, “On Proportional Fairness in Power Allocation for Two-Tone Spectrum-Sharing Networks,” IEEE Trans. Veh. Technol., vol. 65, no. 12, pp. 10 090–10 096, 2016.

[17] R. Liu, L. Zhang, R. Y.-N. Li, and M. Di Renzo, “The ITU Vision and Framework for 6G: Scenarios, Capabilities, and Enablers,” IEEE Veh. Technol. Mag., vol. 20, no. 2, pp. 114–122, 2025.

[18] C. E. Shannon, “A Mathematical Theory of Communication,” Bell Syst. Tech. J., vol. 27, no. 3, pp. 379–423, 1948.

[19] S. P. Boyd and L. Vandenberghe, Convex Optimization. Cambridge university press, 2004.

[20] K. Wintersperger, F. Dommert, T. Ehmer, A. Hoursanov, J. Klepsch, W. Mauerer, G. Reuber, T. Strohm, M. Yin, and S. Luber, “Neutral Atom Quantum Computing Hardware: Performance and End-user Perspective,” EPJ Quantum Technol., vol. 10, no. 1, p. 32, 2023.

[21] A. A. Kamenski, N. L. Manakov, S. N. Mokhnenko, V. D. Ovsiannikov, and A. A. Zenischeva, “van der Waals interaction of atoms in circular Rydberg states,” Eur. Phys. J. D, vol. 72, no. 10, p. 174, 2018.

[22] C. Picken, R. Legaie, K. McDonnell, and J. Pritchard, “Entanglement of neutral-atom qubits with long ground-Rydberg coherence times,” Quantum Sci. Technol., vol. 4, no. 1, p. 015011, 2018.

[23] L. Henriet, “Robustness to spontaneous emission of a variational quantum algorithm,” Phys. Rev. A, vol. 101, no. 1, p. 012335, 2020.

[24] C. Vercellino, G. Vitali, P. Viviani, E. Giusto, A. Scionti, A. Scarabosio, O. Terzo, and B. Montrucchio, “Bbq-mis: A Parallel Quantum Algorithm for Graph Coloring Problems,” in 2023 IEEE Int. Conf. Quantum Comput. Eng. (QCE), vol. 2. IEEE, 2023, pp. 141–147.

[25] Gurobi Optimization, LLC, “Gurobi Optimizer Reference Manual,” 2026. [Online]. Available: https://docs.gurobi.com/ current/