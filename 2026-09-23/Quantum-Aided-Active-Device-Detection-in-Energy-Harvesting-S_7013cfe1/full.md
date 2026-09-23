# Quantum-Aided Active Device Detection in Energy-Harvesting Symbiotic Radio Networks

Remon Polus , Deemah H. Tashman , and Soumaya Cherkaoui

∗ Department of Computer and Software Engineering, Polytechnique Montréal, Montréal, Canada Email: {remon.polus, deemah.tashman, soumaya.cherkaoui}@polymtl.ca

Abstract—Massive connectivity in next-generation networks demands energy- and spectrum-eficient solutions for largescale Internet of Things (IoT) deployments. Symbiotic radio (SR) enables passive IoT devices to communicate by backscattering existing cellular transmissions. A key challenge in uplink SR is active device detection (ADD), which directly afects decoding reliability, interference management, and system throughput. We propose an energy-harvesting code-domain non-orthogonal multiple access (NOMA)-SR system in which IoT devices harvest energy from ambient uplink signals and backscatter information using low-density spreading (LDS) codes. To reduce the complexity of ADD, Grover’s quantum search algorithm is employed, providing a quadratic reduction in oracle-query complexity over exhaustive maximum-likelihood (ML) search. Numerical results show that the proposed approach closely approaches ML performance while substantially reducing the number of search iterations, demonstrating its potential for scalable ambient IoT systems.

Index Terms—Symbiotic radio, energy harvesting, nonorthogonal multiple access (NOMA), quantum-inspired algorithms.

## I. Introduction

With Industry 4.0 evolving toward the fifth industrial revolution, the number of Internet of Things (IoT) devices is projected to reach 29 billion worldwide by 2030 [1]. The International Telecommunication Union (ITU) IMT-2030 framework envisions sixth-generation (6G) wireless networks capable of supporting ultra-massive and seamless connectivity. However, the rapid proliferation of IoT devices is expected to exacerbate energy consumption and spectrum scarcity [2], motivating the development of energy- and spectrum-eficient communication technologies. Symbiotic radio (SR) has emerged as a promising paradigm for addressing these challenges [3]. By enabling passive IoT devices, referred to as backscatter devices, to modulate and reflect ambient radio-frequency signals, SR reduces the reliance on active radio-frequency (RF) components while providing additional multipath diversity to the primary communication system [4]. This mutually beneficial coexistence between passive IoT devices and cellular infrastructure makes SR a promising technology for supporting ambient IoT in 6G networks [5].

The integration of energy harvesting (EH) and nonorthogonal multiple access (NOMA) provides a viable framework for enabling large-scale energy-constrained SR networks [6]. EH enables IoT devices to extract energy from ambient sources and store it for subsequent communication and computation, thereby reducing dependence on conventional batteries and associated replacement costs [7]. In parallel, NOMA improves spectral eficiency by allowing multiple devices to concurrently access the same time-frequency resources [8]. In code-domain NOMA (CD-NOMA), users are distinguished through devicespecific non-orthogonal spreading sequences characterized by low cross-correlation, with low-density spreading (LDS) and sparse code multiple access (SCMA) constituting representative schemes [9].

A fundamental challenge in uplink SR networks is active device detection (ADD), where the base station (BS) must identify, in real time, the subset of devices that are actively transmitting [10]. Accurate ADD is essential for subsequent multi-user decoding, interference management, and throughput optimization. To address this issue, Grover’s quantum search algorithm provides a promising approach to reducing the search complexity of ADD by ofering a quadratic speedup for unstructured search problems [11]. However, to the best of the authors’ knowledge, quantum-assisted ADD has not yet been investigated in SR networks, where ADD is further afected by energyharvesting constraints, backscatter channel conditions, and multi-user interference. This paper addresses this gap by developing a quantum-assisted ADD framework for energy-harvesting SR networks. The main contributions are as follows:

We propose an energy-harvesting CD-NOMA-SR system where IoT devices harvest energy from cellular uplink signals and backscatter information to the BS via LDS codes, enabling battery-less massive IoT connectivity.

We apply Grover’s quantum search algorithm to ADD in the proposed uplink SR network under Rayleigh fading and AWGN, demonstrating its potential as a low-complexity detection mechanism for large-scale SR networks.

The remainder of this paper is organized as follows: Section II presents the system model for the energy harvesting-based SR network, Section III details the application of Grover’s algorithm to the ADD process, Section IV presents simulation results and discussion, and Section V concludes the paper and outlines future work.

![](images/a87be90781cae67b466bf971bdb5c3a1cf6dbd17021b400c18f49f7d67d90f99.jpg)  
Fig. 1. System model of an uplink SR energy-harvesting network.

## II. System Model

As illustrated in Fig. 1, we consider an SR system in which a cellular user communicates with the BS while K IoT devices harvest energy from the uplink transmission to support battery-less operation. Once suficiently charged, the IoT devices become active and backscatter their information using distinct LDS codes, which the BS jointly decodes alongside the cellular transmission. The cellular user also benefits, as the backscattered signals create additional propagation paths that improve diversity and reliability at the BS. The frame duration $T$ is divided into an energy-harvesting phase $\tau _ { 0 }$ and a data-transmission phase $T - \tau _ { 0 }$

## A. Energy Harvesting

During $\tau _ { 0 } .$ , the signal received at IoT device k is [12]

$$
x _ { k } ( m N + n ) = \sqrt { p } h _ { k } s ( m N + n ) ,\tag{1}
$$

where $p$ is the cellular transmit power, $h _ { k }$ the channel to device $k , s ( m N + n )$ the $n ^ { \mathrm { t h } }$ chip of the $m ^ { \mathrm { t h } }$ symbol, and $N$ the LDS code length. The power harvested by device k is [13]

$$
p _ { k } = \eta p | h _ { k } | ^ { 2 } ,\tag{2}
$$

with $\eta$ the harvesting eficiency, and the energy accumulated over $\tau _ { 0 }$ is

$$
\begin{array} { r } { E _ { k } = p _ { k } \tau _ { 0 } = \eta \tau _ { 0 } p | h _ { k } | ^ { 2 } . } \end{array}\tag{3}
$$

During $T - \tau _ { 0 }$ , each device consumes

$$
E _ { c } = \frac { T - \tau _ { 0 } } { T _ { b } } N E _ { 0 } + ( T - \tau _ { 0 } ) p _ { c } ,\tag{4}
$$

where the first term is the energy for reflection-coeficient adjustment $\left( E _ { 0 } \right)$ and the second is the circuit consumption $\left( p _ { c } \right)$ , with $T _ { b }$ the backscatter bit duration [12]. Device k activates only if $E _ { k } \ \ge \ E _ { c }$ , giving the binary activation vector $\mathbf { a } = [ a _ { 1 } , \ldots , a _ { K } ] ^ { \mathsf { T } }$ with

$$
a _ { k } = { \left\{ \begin{array} { l l } { 1 , } & { \eta \tau _ { 0 } p | h _ { k } | ^ { 2 } \geq E _ { c } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{5}
$$

After harvesting suficient energy, active devices begin transmitting by backscattering the cellular signal; to avoid mutual interference, each device multiplies its symbol by a unique LDS code.

## B. LDS Codes

Let ${ \bf F } \in \{ 0 , 1 \} ^ { N \times K }$ denote the LDS factor matrix, where N is the number of chips per codeword, K is the number of available codewords, and each codeword has exactly J nonzero chips; $f _ { n , k } = 1$ indicates that chip n is nonzero in codeword $k ,$ and $f _ { n , k } = 0$ otherwise. The total number of available codewords is given by the binomial coeficient

$$
U = { \binom { N } { J } } .\tag{6}
$$

For example, with $N = 4 , J = 2$

$$
U = { \binom { 4 } { 2 } } = { \frac { 4 ! } { 2 ! \left( 4 - 2 \right) ! } } = 6 ,\tag{7}
$$

so the system accommodates up to 6 devices, each assigned a unique codeword, giving the factor matrix

$$
\mathbf { F } = \left[ \begin{array} { l l l l l l } { 1 } & { 1 } & { 1 } & { 0 } & { 0 } & { 0 } \\ { 1 } & { 0 } & { 0 } & { 1 } & { 1 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { 1 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 1 } & { 0 } & { 1 } & { 1 } \end{array} \right] ,\tag{8}
$$

where each column has exactly $J = 2$ nonzero entries and each row has $d _ { f } = 3$ nonzero entries, meaning every chip is shared among 3 devices. The resulting overloading factor is

$$
\lambda = \frac { U } { N } = \frac { 6 } { 4 } = 1 5 0 \% .\tag{9}
$$

## C. Received Signal at the BS

The received signal at the BS comprises both the directlink signal from the cellular user and the backscatter links from all active IoT devices, which can be expressed as

$$
\begin{array} { l } { { \displaystyle y ( m N + n ) = \sqrt { p } h _ { 0 } s ( m N + n ) } } \\ { ~ } \\ { { \displaystyle ~ + \sum _ { k = 1 } ^ { K } \sqrt { p } h _ { k } g _ { k } b _ { k } ( m ) a _ { k } c _ { k } ( n ) s ( m N + n ) } } \\ { { \displaystyle ~ + u ( m N + n ) , } } \end{array}\tag{10}
$$

where $h _ { 0 }$ is the channel coeficient of the direct link from the cellular device to the BS. $g _ { k }$ denotes the channel coeficient of the channel between IoT device k to the BS. We assume that all these channels follow Rayleigh distribution. $b _ { k } ( m )$ is the symbol transmitted by device k and $c _ { k } ( n )$ is the LDS code used by device k. $u ( m N + n ) \sim$ $\mathcal { C N } ( 0 , \sigma ^ { 2 } )$ is the AWGN at the BS, where the total signalto-noise ratio (SNR) can be defined as

$$
\gamma = \frac { p | h _ { 0 } | ^ { 2 } + \sum _ { k = 1 } ^ { K } a _ { k } p | g _ { k } h _ { k } | ^ { 2 } } { \sigma ^ { 2 } } .\tag{11}
$$

Since the direct-link signal is typically much stronger than the backscattered signals, the receiver first decodes $s ( m N + n )$ and removes it using successive interference

cancellation (SIC) [2]. After SIC, the resulting signal is given as

$$
\begin{array} { c } { { \tilde { y } ( m N + n ) = \displaystyle \sum _ { k = 1 } ^ { K } \sqrt { p } g _ { k } h _ { k } a _ { k } c _ { k } ( n ) s ( m N + n ) b _ { k } ( m ) } } \\ { { + u ( m N + n ) . } } \end{array}\tag{12}
$$

III. Grover’s Algorithm for Active IoT Device Detection

At the BS, the goal is to detect the active IoT devices, corresponding to the ADD problem, whose search space grows as $2 ^ { \overset { \smile } { K } }$ , rendering classical exhaustive detection prohibitive for large-scale SR deployments. We instead employ a quantum search framework that exploits superposition to evaluate multiple candidate activity patterns in parallel, reducing the computational burden of ADD [14]. Among quantum search methods, Grover’s algorithm is widely used for finding a target element in an unsorted database of size $B ,$ , ofering a quadratic speedup of $\mathcal { O } ( \sqrt { B } )$ over the classical O(B) [11]. It is built on two main components: the Oracle and the Difuser.

The Oracle identifies the quantum state associated with the desired outcome — in the ADD context, the codeword that best matches the received superposed signal from the active backscatter IoT devices — by marking it via a phase inversion. Using an auxiliary qubit initialized in |−⟩, the Oracle operator $O _ { \nu }$ applies a phase shift of −1 to a state |x⟩ satisfying $Y ( x ) = \nu ,$ leaving all other states unchanged [11]:

$$
O _ { \nu } | x \rangle = { \left\{ \begin{array} { l l } { - | x \rangle , } & { { \mathrm { i f ~ } } Y ( x ) = \nu } \\ { | x \rangle , } & { { \mathrm { i f ~ } } Y ( x ) \neq \nu } \end{array} \right. } .\tag{13}
$$

While the Oracle tags the desired state, the Difuser amplifies it via inversion-about-the-mean, and iterating the two increases the probability of observing the correct solution [11]. The number of Grover iterations follows [13]

$$
\kappa = \lfloor \frac { \pi } { 4 } \sqrt { \frac { B } { L } } \rfloor ,\tag{14}
$$

where $B ~ = ~ 2 ^ { K }$ is the search space size (all activity combinations) and L is the number of feasible solutions.

Prior to encoding, the received signal is discretized to ensure compatibility with binary quantum circuits. Assuming unipolar signaling with symbols in {0, 1}, the received signal y is quantized as [13]

$$
y _ { I } = \operatorname* { m i n } \left( \operatorname* { m a x } \left( 0 , \operatorname { r o u n d } ( y ) \right) , 2 ^ { s } - 1 \right) ,\tag{15}
$$

where s sets the number of quantization levels.

## A. Average Probability of Success

The average probability of success $P _ { s }$ measures the BS’s ability to correctly distinguish active from inactive devices, with higher $P _ { s }$ implying a lower false-positive rate. Letting $\hat { a } _ { k }$ denote the estimated activity of device $k ,$ detection is successful if

$$
r _ { k } = \left\{ { \begin{array} { l l } { 1 , } & { \hat { a } _ { k } = a _ { k } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right. \quad k = 1 , \ldots , K ,\tag{16}
$$

![](images/7cb3b2800aa97134a49003136dc3246384aaf8a43934c69da92f321457b6abdb.jpg)  
Fig. 2. Average probability of success versus the transmit power in dBm (varying K).

and the average success probability over $Z _ { \mathrm { t o t a l } }$ trials is

$$
P _ { s } = \frac { \sum _ { t = 1 } ^ { Z _ { \mathrm { t o t a l } } } \sum _ { k = 1 } ^ { K } r _ { k } ^ { ( t ) } } { K Z _ { \mathrm { t o t a l } } } .\tag{17}
$$

## IV. Numerical Results

This section evaluates the performance of the proposed approach. Since the quantum algorithm is implemented using a classical simulator, the evaluation can be computationally intensive [15]. Therefore, a simplified network architecture is considered to enable practical evaluation. The simulation parameters are summarized in Table I.

TABLE I  
Simulation Parameters
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Symbol</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Bit duration</td><td rowspan=1 colspan=1> $T _ { b }$ </td><td rowspan=1 colspan=1>0.02 ms</td></tr><tr><td rowspan=1 colspan=1>Total time slot</td><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>0.1 s</td></tr><tr><td rowspan=1 colspan=1>Energy harvesting slot</td><td rowspan=1 colspan=1> $\tau _ { 0 }$ </td><td rowspan=1 colspan=1>0.01 s</td></tr><tr><td rowspan=1 colspan=1>Energy required to adjustthe reflection coefficient</td><td rowspan=1 colspan=1> $\overline { { E _ { 0 } } }$ </td><td rowspan=1 colspan=1>10 nJ</td></tr><tr><td rowspan=1 colspan=1>Harvesting efficiency</td><td rowspan=1 colspan=1>η</td><td rowspan=1 colspan=1>0.9</td></tr><tr><td rowspan=1 colspan=1>Noise variance</td><td rowspan=1 colspan=1> $\overline { { \sigma ^ { 2 } } }$ </td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=1>Circuit power</td><td rowspan=1 colspan=1> $p _ { c }$ </td><td rowspan=1 colspan=1>-5 dBm</td></tr><tr><td rowspan=1 colspan=1>Number of Iterations</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { 4 } } }$ </td></tr></table>

Fig. 2 shows the average probability of successful detection versus transmit power from 10 to 30 dBm for classical maximum-likelihood (ML) detection, used as the benchmark, and Grover-based detection with K = 2 and $K \ = \ 6 .$ with $J \ = \ 2 .$ . As transmit power increases, all schemes exhibit improved detection performance due to the increased received SNR at the BS. For $K = 2 ,$ , the

TABLE II  
Computational Complexity: Grover’s Algorithm vs. ML Detection
<table><tr><td> $\overline { { K } }$ </td><td>Search space</td><td>Grover</td><td>ML</td><td>Speedup</td></tr><tr><td>2</td><td>4</td><td>1</td><td>4</td><td>4.0×</td></tr><tr><td>4</td><td>16</td><td>3</td><td>16</td><td>5.3×</td></tr><tr><td>6</td><td>64</td><td>12</td><td>64</td><td>5.3×</td></tr></table>

![](images/4e4e9bf50b25480a27b18f1a6ac5009411d553708d948de7def501721f547033.jpg)  
Fig. 3. Average probability of success versus the transmit power in dBm (varying J).

Grover-based detector closely approaches the ML benchmark while requiring quadratically fewer search iterations. As K increases to 6, the search space expands from $2 ^ { 2 } = 4$ to $2 ^ { 6 } = 6 4$ states, while increased multi-user interference leads to some performance degradation. Nevertheless, the $K = 6$ case maintains the expected improvement with transmit power, demonstrating the applicability of Groverbased detection to larger device sets.

The corresponding search complexity of the detection schemes in Fig. 2 is presented in Table II that compares the number of oracle calls required by the proposed Groverbased detector with the exhaustive comparisons required by the classical ML benchmark for $K = 2 ,$ 4, 6 IoT devices. As K increases, the ML search space grows exponentially as $B ~ = ~ 2 ^ { K }$ , whereas the number of Grover iterations $\begin{array} { r } { \kappa = \left| \frac { \pi } { 4 } \sqrt { B } \right| } \end{array}$ grows as $\mathcal { O } ( \sqrt { B } )$ . For $K = 6$ , Grover requires only 12 oracle calls compared with 64 ML comparisons, corresponding to a 5.3× reduction in search operations. This advantage becomes increasingly significant as the number of IoT devices grows, highlighting the potential of the proposed approach for large-scale SR deployments.

Fig. 3 shows the average probability of successful detection versus transmit power from 10 to 30 dBm for $N = 6 .$ $J = 1$ and J = 3, with $K = 6$ IoT devices. The $J = 3$ configuration consistently outperforms $J = 1$ due to the greater spreading diversity obtained by allocating multiple chips to each device. This increased diversity improves the distinguishability of the received backscatter signals and, consequently, detection reliability. In contrast, $J =$ 1 provides limited diversity, resulting in weaker signal distinguishability despite lower inter-device interference. These results demonstrate that increasing the number of chips per device can improve the performance of Groverbased ADD.

## V. Conclusion

This paper proposed an energy-harvesting CD-NOMA symbiotic radio framework for battery-less IoT connectivity, where IoT devices harvest energy from ambient cellular uplink signals and backscatter information using LDS codes. Grover’s quantum search algorithm is employed at the BS to identify active devices with reduced search complexity. Simulation results show that the proposed detector closely approaches the ML upper bound while substantially reducing the number of search iterations. Future work will investigate implementation on near-term quantum hardware and extensions to multi-user primary networks and larger-scale IoT deployments.

## References

[1] C. De Alwis, A. Kalla, Q.-V. Pham, P. Kumar, K. Dev, W.-J. Hwang, and M. Liyanage, “Survey on 6G Frontiers: Trends, Applications, Requirements, Technologies and Future Research,” IEEE Open J. Commun. Soc., vol. 2, pp. 836–886, 2021.

[2] J. Wang, Z. Zhao, and Y.-C. Liang, “Energy Harvesting-Data Transmission Tradeof in Symbiotic Radios for Ambient IoT,” IEEE Trans. Wireless Commun., vol. 25, pp. 1437–1450, 2026.

[3] R. Long, H. Guo, L. Zhang, and Y.-C. Liang, “Full-Duplex Backscatter Communications in Symbiotic Radio Systems,” IEEE Access, vol. 7, pp. 21 597–21 608, 2019.

[4] S. Mondal, D. Bepari, A. Chandra, K. Singh, C.-P. Li, and Z. Ding, “A Comprehensive Survey on NOMA-Based Backscatter Communication for IoT Applications,” IEEE Internet Things J., vol. 12, no. 12, pp. 18 929–18 953, 2025.

[5] Y.-C. Liang, R. Long, Q. Zhang, and D. Niyato, “Symbiotic Communications: Where Marconi Meets Darwin,” IEEE Wirel. Commun., vol. 29, no. 1, pp. 144–150, 2022.

[6] G. Moloudian, M. Hosseinifard, S. Kumar, R. B. Simorangkir, J. L. Buckley, C. Song, G. Fantoni, and B. O’Flynn, “RF Energy Harvesting Techniques for Battery-Less Wireless Sensing, Industry 4.0, and Internet of Things: A Review,” IEEE Sens. J., vol. 24, no. 5, pp. 5732–5745, 2024.

[7] Y. Kim, L. Liu, K. Takeda, B. Han, V. Vintola, C. Wei, P. Gupta, C. Zhang, Z. Fan, K. Mukkavilli et al., “Challenges and Advances in Ambient IoT within 3GPP,” IEEE Commun. Mag., vol. 64, no. 1, pp. 182–188, 2026.

[8] L. Dai, B. Wang, Z. Ding, Z. Wang, S. Chen, and L. Hanzo, “A Survey of Non-Orthogonal Multiple Access for 5G,” IEEE Commun. Surv. Tutor., vol. 20, no. 3, pp. 2294–2323, 2018.

[9] H. Jafarkhani, H. Maleki, and M. Vaezi, “Modulation and coding for NOMA and RSMA,” Proc. IEEE, vol. 112, no. 9, pp. 1179–1213, 2024.

[10] M. I. Habibie, C. Goursaud, and J. Hamie, “Quantum Minimum Searching Algorithms for Active User Detection in Wireless IoT Networks,” IEEE Internet Things J., vol. 11, no. 12, pp. 22 603– 22 615, 2024.

[11] P. Botsinis, D. Alanis, Z. Babar, H. V. Nguyen, D. Chandra, S. X. Ng, and L. Hanzo, “Quantum Search Algorithms for Wireless Communications,” IEEE Commun. Surv. Tutor., vol. 21, no. 2, pp. 1209–1242, 2019.

[12] Y.-C. Liang, Q. Zhang, E. G. Larsson, and G. Y. Li, “Symbiotic Radio: Cognitive Backscattering Communications for Future Wireless Networks,” IEEE Trans. Cogn. Commun. Netw., vol. 6, no. 4, pp. 1242–1255, 2020.

[13] D. H. Tashman and S. Cherkaoui, “Quantum-Aided Active User Detection for Energy-Eficient CD-NOMA in Cognitive Radio Networks,” in Proc. Int. Wirel. Commun. Mob. Comput. Conf. (IWCMC). IEEE, 2025, pp. 1661–1666.

[14] Y. Li, M. Tian, G. Liu, C. Peng, and L. Jiao, “Quantum Optimization and Quantum Learning: A Survey,” IEEE Access, vol. 8, pp. 23 568–23 593, 2020.

[15] R. Piron and C. Goursaud, “Quantum Annealing for Active User Detection in NOMA Systems,” in Proc. Asilomar Conf. Signals Syst. Comput. IEEE, 2024, pp. 1448–1452.