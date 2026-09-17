# APGEM: Adaptive Policy-Guided Error Mitigation for Quantum Reinforcement Learning on a Real-World CVRP Case Study

Shabir Ahmad Sofi<sup>\*</sup>

Department of Information Technology

NIT Srinagar

Bisma Majid<sup>\*</sup>

shabir@nitsri.ac.in

Department of Information Technology

J&K, India

NIT Srinagar

J&K, India

bismabhat ite006@nitsri.ac.in

Mir Mohammad Yousuf<sup>\*</sup>

Department of Information Technology

NIT Srinagar

J&K, India

yousuf 2022phaite006@nitsri.ac.in

Abstract—Quantum Reinforcement Learning (QRL) represents policies as variational quantum circuits (VQCs), making it attractive for combinatorial optimization such as the Capacitated Vehicle Routing Problem (CVRP). On noisy intermediate-scale quantum (NISQ) hardware, however, decoherence degrades fidelity and destabilizes learning, and conventional error mitigation is applied statically without regard to the learning context. We introduce Adaptive Policy-Guided Error Mitigation (APGEM), a controller that selects among Zero-Noise Extrapolation (ZNE), Probabilistic Error Cancellation (PEC), Clifford Data Regression (CDR), and Readout Error Mitigation (REM) online, driven by a fidelity, entropy, and cost aware utility function and an ε-greedy rule over temporal-difference Q-scores. We evaluate on a realistic urban-logistics testbed, a Delhi-based CVRP over real landmarks with geodesic inter-node costs, exercised across five noise families and four severity levels. On this instance, the QRL agent outperforms constructive heuristics and approaches metaheuristics, while mitigation restores approximation ratios from 0.84–0.87 to 0.92– 0.94 under high noise. The controller shifts from a CDRdominated regime under short training horizons to a balanced deployment across all four techniques under longer horizons, indicating genuine regime-dependent selection. These preliminary results position adaptive, learning-aware mitigation as a practical route to noiseresilient QRL.

Index Terms—Quantum reinforcement learning, error mitigation, NISQ, vehicle routing problem, variational quantum circuits.

## I. INTRODUCTION

The Vehicle Routing Problem (VRP) is a canonical NP-hard combinatorial optimization task underpinning transportation and last-mile logistics [1]–[3]. Efficient routing yields direct savings in cost, fuel, and emissions, so the problem is a persistent driver of efficiency in industry [4], [5]. Classical exact solvers and metaheuristics scale poorly as the number of customers and constraints grows, motivating alternative computational paradigms [6], [7]. Noisy intermediate-scale quantum (NISQ) devices have emerged as one such candidate [8], [9], with variational quantum algorithms particularly suited to near-term hardware [10].

Quantum Reinforcement Learning (QRL) embeds parameterized quantum circuits within reinforcement learning agents, representing the policy as a variational quantum circuit (VQC) that maps environment states to action probabilities while classical optimization updates the parameters [11]. This hybrid structure suits sequential decision problems such as VRP, and recent work applies parameterized quantum policies directly to routing and last-mile delivery [12]. Two obstacles limit QRL in practice. First, circuits executed on NISQ devices are highly susceptible to noise, which lowers fidelity, raises state entropy, and destabilizes learning [13]. Second, established mitigation methods, including Zero-Noise Extrapolation (ZNE), Probabilistic Error Cancellation (PEC), Clifford Data Regression (CDR), and Readout Error Mitigation (REM), are typically chosen once and applied uniformly, without adapting to the circuit or noise characteristics of a given execution [14]–[16]. Learning-based mitigation has begun to close this gap for variational circuits and quantum software [17]–[19], yet it is rarely coupled to the reinforcement signal that drives policy optimization. How noise-aware adaptation should influence policy optimization, and how to select among mitigation strategies in real time from observed performance signals, remains open [20].

We address this gap with Adaptive Policy-Guided Error Mitigation (APGEM). Our contributions are:

1) A hybrid QRL environment instantiated on a realworld CVRP case study, a Delhi last-mile delivery scenario with authentic geographic structure, in which the policy is a VQC trained under noise models that emulate realistic NISQ conditions.

2) The APGEM controller, which dynamically selects among ZNE, PEC, CDR, and REM using an ε-greedy rule guided by a composite utility over fidelity, entropy, and computational cost, turning mitigation from a static pre-processing step into a learning component coupled to reinforcement feedback.

3) An adaptive learning-rate scheme that modulates parameter updates by both reward and quantum execution quality, stabilizing training under noise.

## II. RELATED WORK

Quantum computing for logistics. Quantum and hybrid quantum-classical methods have been explored across supply-chain and routing tasks, motivated by the combinatorial hardness of VRP variants and the economic value of marginal routing improvements [4], [21], [22]. These efforts largely target the optimization formulation itself, for example through annealing or QAOA-style objectives, rather than through learned sequential policies.

Quantum reinforcement learning. QRL replaces classical policy or value networks with variational quantum circuits, exploiting quantum state representation for compact policy encodings [11], [23], [24]. Recent studies apply parameterized quantum policies to routing and last-mile delivery under realistic constraints [12], but typically assume ideal or lightly modeled noise and do not treat error mitigation as part of the learning loop.

Quantum error mitigation. A mature toolbox of mitigation techniques now exists, including ZNE [25], [26], readout correction [15], and calibration-based regression [20], surveyed comprehensively in [27]. More recent work learns mitigation models directly from data for variational circuits and quantum software pipelines [17], [18]. These methods are powerful but are almost always applied as a fixed, pre-selected stage. Our work differs by making the choice of mitigation technique itself an online, reward-coupled decision within a QRL agent.

## III. BACKGROUND

## A. CVRP as a Reinforcement Learning Problem

The CVRP seeks minimum-cost tours for a fleet of capacity-limited vehicles serving customers from a single depot. We cast it as an episodic Markov decision process. The state encodes vehicle positions, remaining loads, partial route lengths, and per-customer visit and remaining-demand indicators. The action set $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } _ { t } )$ comprises unserved customers within remaining capacity plus a return-to-depot action. The step reward combines a distance term with a visit bonus, and a terminal reward or penalty reflects unserved customers.

## B. Quantum Reinforcement Learning Formulation

The policy $\pi _ { \theta }$ is realized by a variational quantum circuit $U ( \theta )$ with trainable parameters $\theta = ( \theta _ { 1 } , \ldots , \theta _ { p } )$ Acting on $n _ { q }$ qubits it prepares

$$
\left. \psi ( \boldsymbol { \theta } ) \right. = U ( \boldsymbol { \theta } ) \left. 0 \right. ^ { \otimes n _ { q } } ,\tag{1}
$$

and a computational-basis measurement yields a bitstring z with probability $\begin{array} { c c l } { P _ { \theta } ( z ) } & { = } & { | \langle z | \psi ( \theta ) \rangle | ^ { 2 } } \end{array}$ . A classical post-processing map $a _ { t } ~ = ~ f ( z , s _ { t } )$ enforces feasibility, so the induced policy is

$$
\pi _ { \boldsymbol { \theta } } ( a _ { t } \mid s _ { t } ) = \sum _ { z \in \mathcal { Z } ( a _ { t } ) } P _ { \boldsymbol { \theta } } ( z ) ,\tag{2}
$$

where $\mathcal { Z } ( a _ { t } )$ is the set of bitstrings mapped to action $a _ { t }$ . Training maximizes the expected return $J ( \theta ) ~ =$ $\mathbb { E } _ { \pi _ { \theta } } [ \sum _ { t = 0 } ^ { T } r _ { t } ]$ . Since the reward encodes negative travel cost, maximizing $J ( \theta )$ is equivalent to minimizing the classical routing cost $\begin{array} { r } { \sum _ { k \in K } \sum _ { ( i , j ) \in \mathrm { r o u t e } _ { k } } c _ { i j } } \end{array}$ . Gradients use the parameter-shift rule,

$$
\frac { \partial } { \partial \theta _ { i } } \langle { \cal O } \rangle _ { \theta } = { \scriptstyle \frac { 1 } { 2 } } \Big ( \langle { \cal O } \rangle _ { \theta _ { i } + \frac { \pi } { 2 } } - \langle { \cal O } \rangle _ { \theta _ { i } - \frac { \pi } { 2 } } \Big ) ,\tag{3}
$$

enabling unbiased stochastic gradient ascent $\theta  \theta +$ $\eta \nabla _ { \theta } J ( \theta )$

## C. Noise Modeling

On near-term devices the circuit executes under a completely positive trace-preserving map ${ \mathcal { N } } ,$ so the prepared state is $\rho ( \theta ) = \mathcal { N } ( U ( \theta ) | 0 \rangle \langle 0 | ^ { \otimes n _ { q } } U ^ { \dag } ( \theta ) )$ with Kraus form $\begin{array} { r } { \mathcal { N } ( \rho ) = \sum _ { \alpha } K _ { \alpha } \rho K _ { \alpha } ^ { \dagger } } \end{array}$ and $\begin{array} { r } { \sum _ { \alpha } K _ { \alpha } ^ { \dagger } K _ { \alpha } = I } \end{array}$ We simulate four canonical channels: the depolarizing channel $\begin{array} { r } { \mathcal { D } _ { p } ( \rho ) = ( 1 - p ) \rho + \frac { p } { 3 } ( X \rho X + Y \rho Y + Z \rho Z ) \mathrm { { : } } } \end{array}$ dephasing $\begin{array} { r } { \mathcal { Z } _ { \lambda } ( \rho ) = ( 1 - \lambda ) \rho + \lambda Z \rho Z ; } \end{array}$ amplitude damping with Kraus operators parameterized by $\gamma ;$ and readout noise modeled by a confusion matrix A with $\tilde { p } = A p$ . Two-qubit gate noise is applied as depolarizing noise on entangling gates.

## D. Error Mitigation Techniques

The mitigation pool spans complementary tradeoffs [27]. ZNE scales the effective noise to levels $\lambda _ { i }$ and extrapolates expectation values to the zeronoise limit, for example by Richardson extrapolation $\begin{array} { r } { \hat { E } ( 0 ) \ = \ \sum _ { i = 1 } ^ { r } c _ { i } E ( \lambda _ { i } ) } \end{array}$ , reducing bias at the cost of added variance [25], [26]. PEC expresses a target gate as a quasi-probability decomposition $\begin{array} { r } { \mathcal { G } = \sum _ { j } \alpha _ { j } \tilde { \mathcal { G } } _ { j } } \end{array}$ and reweights outcomes by $\mathrm { s g n } ( \alpha _ { j } ) \Gamma$ with $\Gamma \doteq \sum _ { j } | \alpha _ { j } |$ giving an unbiased estimate with variance overhead scaling as $\Gamma ^ { 2 }$ . CDR fits a regression $y \approx a x + b$ between noisy and ideal expectation values from near-Clifford proxy circuits and benefits from richer calibration data [20]. REM inverts the confusion matrix, $\hat { p } = A ^ { - 1 } \tilde { p } ,$ to correct measurement statistics [15]. No single method dominates across noise regimes or across the training trajectory, which motivates adaptive selection.

## IV. ADAPTIVE POLICY-GUIDED ERROR MITIGATION

The effectiveness of each mitigation method varies over training: early policies are highly stochastic and tolerate low-cost mitigation, whereas sharpened later policies demand more aggressive correction to preserve fidelity. APGEM frames the choice of mitigation as a learning problem guided by policy diagnostics.

For a policy parameter θ and state s, we track the fidelity $F ( \theta , s )$ of the mitigated state relative to the ideal, the entropy of the policy distribution,

$$
H ( \theta , s ) = - \sum _ { z } \tilde { p } _ { \theta } ( z \mid s ) \log \tilde { p } _ { \theta } ( z \mid s ) ,\tag{4}
$$

the variance of corrected estimates, and the computational cost $C _ { \mathcal { M } }$ . These are aggregated into a scalar utility for each candidate method $\mathcal { M } \mathrm { : }$

$$
U ( \mathcal { M } ; \theta , s ) = w _ { F } F _ { \mathcal { M } } ( \theta , s ) - w _ { H } H _ { \mathcal { M } } ( \theta , s ) - w _ { C } C _ { \mathcal { M } } ,\tag{5}
$$

where $w _ { F } , w _ { H } , w _ { C }$ weight fidelity, entropy, and overhead. The controller maintains Q-scores $Q _ { t } ( \mathcal { M } )$ updated by a temporal-difference rule,

$$
\begin{array} { r } { Q _ { t + 1 } ( \mathcal { M } ) = ( 1 - \alpha ) Q _ { t } ( \mathcal { M } ) + \alpha U ( \mathcal { M } ; \theta _ { t } , s _ { t } ) , } \end{array}\tag{6}
$$

with learning rate $\alpha \in ( 0 , 1 )$ , and selects a method $\varepsilon -$ greedily:

$$
\mathcal { M } _ { t } = \left\{ \begin{array} { l l } { \mathrm { r a n d o m ~ \mathcal { M } } , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } \varepsilon , } \\ { \mathrm { a r g ~ m a x } _ { \mathcal { M } } Q _ { t } ( \mathcal { M } ) , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - \varepsilon . } \end{array} \right.\tag{7}
$$

The selected method yields corrected statistics $\tilde { p } _ { \theta } ^ { \mathcal { M } _ { t } } ( z \mid$ s) and a mitigated policy $\pi _ { \boldsymbol { \theta } } ^ { \mathcal { M } _ { t } }$ used for action sampling

and gradient estimation, so that the policy gradient

$$
\nabla _ { \theta } J ( \theta ; \mathcal { M } _ { t } ) = \mathbb { E } \left[ \sum _ { { t = 0 } } ^ { T } \nabla _ { \theta } \log \pi _ { \theta } ^ { \mathcal { M } _ { t } } ( a _ { t } \mid s _ { t } ) G _ { t } \right]\tag{8}
$$

is informed by adaptively mitigated measurements, with $G _ { t }$ the discounted return-to-go. Algorithm 1 summarizes the controller.

Algorithm 1 APGEM Controller   
1: Input: pool M = {ZNE, PEC, CDR, REM}, rates   
$\alpha , \varepsilon$   
2: Init: $Q ( { \mathcal { M } } ) \gets 0 \forall { \mathcal { M } }$   
3: procedure $\mathbf { A P G E M } ( \theta , s _ { t } )$   
4: for each $\mathcal { M } \in \mathcal { M }$ do   
5: apply M to $\tilde { p } _ { \theta } ( z \mid s _ { t } )$   
6: compute $F _ { \mathcal { M } } , H _ { \mathcal { M } } , C _ { \mathcal { M } }$   
7: $U \gets w _ { F } F _ { \mathcal { M } } - w _ { H } H _ { \mathcal { M } } - w _ { C } C _ { \mathcal { M } }$   
8: $Q ( \mathcal { M } )  ( 1 - \alpha ) Q ( \mathcal { M } ) + \alpha U$   
9: end for   
10: $\boldsymbol { \mathcal { M } } _ { t } \gets$ random w.p. ε, else arg max $\mathcal { M } ^ { Q ( \mathcal { M } ) }$   
11: return $\mathcal { M } _ { t }$   
12: end procedure

To couple update magnitude to execution quality, APGEM further modulates the learning rate by fidelity and entropy diagnostics, as summarized in Algorithm 2.

Algorithm 2 Policy Optimization with Adaptive Learn  
ing Rate   
1: Input: parameters $\theta ,$ base rate $\eta _ { 0 }$ , scale $\beta ,$ discount   
γ, weights $( \kappa , \lambda )$   
2: for episode = 1 to N do   
3: roll out trajectory τ under mitigated policy $\pi _ { \boldsymbol { \theta } } ^ { \mathcal { M } _ { t } }$   
4: compute returns $\begin{array} { r } { G _ { k }  \sum _ { j = k } ^ { | \tau | - 1 } \gamma ^ { j - k } r _ { j } } \end{array}$   
5: $\eta _ { t }  \eta _ { 0 } ( 1 + \kappa F _ { \mathcal { M } } - \lambda H _ { \mathcal { M } } )$   
6: $\theta \gets \theta + \eta _ { t } \nabla _ { \theta } J ( \theta ; \mathcal { M } _ { t } ) ;$ wrap $\theta  \theta$ mod 2π   
7: end for   
8: Output: optimized parameters $\theta ^ { \star }$

![](images/5da367b1f8fa37ac8cd42d897964b9b64e023bf397a8fbe2bbef6a8ecd9bc4d1.jpg)  
Fig. 1: Proposed framework. VRP instance generation feeds an RL environment whose policy is a variational quantum circuit. Execution under noise is corrected online by the APGEM controller, with fidelity and entropy diagnostics guiding policy optimization; solutions are compared against classical baselines.

## V. PERFORMANCE METRICS

We evaluate along three complementary layers. Quantum reliability is captured by state fidelity $F ( \rho _ { \mathrm { n o i s y } } , \rho _ { \mathrm { i d e a l } } )$ , the overlap between noisy and ideal states, and the normalized von Neumann entropy $H ( \rho ) ~ = ~ - \mathrm { { T r } } ( \rho \log _ { 2 } \rho )$ divided by $n _ { q } ,$ , where lower entropy indicates a purer, more reliable execution. Optimization effectiveness is measured by cumulative reward $\begin{array} { r } { \boldsymbol { R } ~ = ~ \sum _ { t } \boldsymbol { r } _ { t } } \end{array}$ , total routing cost $C ,$ and the approximation ratio $\alpha ~ = ~ C _ { \mathrm { q u a n t u m } } / C _ { \mathrm { c l a s s i c a l } }$ , where α near one indicates parity with the classical baseline. Robustness is quantified by the mitigation gain $\Delta M =$ $M _ { \mathrm { m i t i g a t e d } } \mathrm { ~ - ~ } M _ { \mathrm { u n m i t i g a t e d } }$ for a metric M, the selection frequency of each mitigation technique, and the variance of reward, cost, and fidelity across seeds. Together these ensure that the evaluation reflects not only solution quality but also quantum reliability and adaptive resilience.

## VI. EXPERIMENTAL SETUP

We deliberately ground the evaluation in a real urban geography rather than a synthetic point set, so that the routing task reflects the spatial structure of an operational last-mile scenario. Instances use a single depot at Connaught Place (Delhi) with customer nodes drawn from real landmarks; inter-node costs are geodesic great-circle distances, preserving the true metric relationships between locations. Demands are sampled with zero depot demand, and vehicle capacity is set as a function of total demand to ensure feasibility.

The policy ansatz, shown in Fig. 2, applies $R _ { y } ( \theta _ { i } )$ feature-encoding rotations, a controlled-Z entanglement layer, and stacked trainable $R _ { y } ( \phi _ { i } ^ { ( k ) } )$ variational layers, balancing expressivity against circuit depth for NISQ feasibility. Circuits are simulated on the AerSimulator, for which the noiseless reference state is directly accessible; the state fidelity used in APGEM’s utility is therefore computed exactly against this ideal reference at negligible cost. On real NISQ hardware the exact ideal state is not available, so the fidelity term would instead be estimated through hardware-computable surrogates (for example, near-Clifford or mirror-circuit fidelity proxies), or replaced by the entropy and estimatorvariance diagnostics that the controller already tracks; the APGEM selection logic is unchanged in either case. Hardware-side fidelity estimation is left to future work. To emulate NISQ conditions, circuits execute under five noise families (depolarizing, amplitude damping, phase damping, two-qubit gate, and readout) at severity levels {0.01, 0.05, 0.08, 0.10}, yielding a grid of scenarios. Classical baselines are Nearest Neighbor, Savings (Clarke–Wright), Genetic Algorithm, Simulated Annealing, and Tabu Search. Training horizons of 100 and 500 episodes probe initial learning and stabilization.

Implementation details. The CVRP instance comprises a single depot, 8 customer nodes, and a fleet of 3 vehicles, with customer demands sampled randomly. The policy is encoded on 4 qubits: the ansatz applies an $R _ { y }$ feature-encoding layer, a controlled-Z entanglement layer, and a RealAmplitudes variational block of two stacked trainable $R _ { y }$ layers, giving 12 trainable parameters (4 qubits × 3 rotation layers). Circuit parameters are initialized uniformly at random in [0, 2π), and the discount factor is $\gamma = 1$ (episodic). Reported metrics are averaged over multiple random seeds, with the same seeds shared across noise families and mitigation settings to enable paired comparison.

![](images/81ef6f307af417a39d1ff4480f38714ff1b2349f601f8d42b043c950ab320683.jpg)  
Fig. 2: Quantum policy ansatz. Feature-encoding $R _ { y }$ rotations embed the environment state, a controlled-Z layer entangles qubits, and stacked variational $R _ { y }$ layers provide trainable expressivity before measurement drives action selection.

## VII. RESULTS

Table I reports QRL against classical baselines. The agent outperforms constructive heuristics and approaches metaheuristics; against Nearest Neighbor the approximation ratio averages 0.743 with a success rate above 92%, while ratios against metaheuristics lie between 0.83 and 0.91. Extending training to 500 episodes improves ratios uniformly, for example Genetic Algorithm from 0.909 to 0.915 and Tabu Search from 0.890 to 0.896, with reward curves converging 15 to 20% above the unmitigated baseline and approximation ratios stabilizing between 0.92 and 0.94 at lower variance.

Mitigation techniques. Table II summarizes pertechnique approximation ratios. Under the short horizon PEC is strongest at 0.922, though its aggressive cancellation amplifies variance; ZNE, CDR, and REM cluster near 0.903–0.908. Under the long horizon CDR overtakes PEC (0.918 vs. 0.913), consistent with richer calibration data favoring regression-based correction,

TABLE I: QRL vs. classical heuristics (100 vs. 500 episodes). Ratio is QRL/heuristic; success in %.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Cost</td><td colspan="2">100 ep.</td><td colspan="2">500 ep.</td></tr><tr><td>Ratio</td><td>Succ.</td><td>Ratio</td><td>Succ.</td></tr><tr><td>Nearest Neighbor</td><td>108.29</td><td>0.743</td><td>92.2</td><td>0.748</td><td>92.1</td></tr><tr><td>Savings</td><td>90.57</td><td>0.888</td><td>57.6</td><td>0.894</td><td>57.9</td></tr><tr><td>Genetic Algorithm</td><td>88.52</td><td>0.909</td><td>54.7</td><td>0.915</td><td>54.1</td></tr><tr><td>Simulated Annealing</td><td>96.78</td><td>0.831</td><td>70.8</td><td>0.837</td><td>70.7</td></tr><tr><td>Tabu Search</td><td>90.36</td><td>0.890</td><td>57.2</td><td>0.896</td><td>57.6</td></tr></table>

and all techniques yield positive improvement over the unmitigated baseline.

TABLE II: Mitigation technique performance (100 vs. 500 episodes).
<table><tr><td></td><td>Technique Ratio (100)</td><td>Impr. (100)</td><td>Ratio (500)</td><td>Impr. (500)</td></tr><tr><td>ZNE</td><td>0.903</td><td>+0.5%</td><td>0.910</td><td>+0.8%</td></tr><tr><td>PEC</td><td>0.922</td><td>-1.6%</td><td>0.913</td><td>+0.5%</td></tr><tr><td>CDR</td><td>0.907</td><td>+0.1%</td><td>0.918</td><td>+0.9%</td></tr><tr><td>REM</td><td>0.908</td><td>-0.1%</td><td>0.914</td><td>+0.7%</td></tr></table>

Noise resilience. Table III and Fig. 3 report noise sensitivity. Across depolarizing, amplitude damping, dephasing, gate, and readout noise, unmitigated approximation ratios fall to 0.84–0.87 at high noise (0.08– 0.10), whereas mitigation restores them to 0.92–0.94. Depolarizing, amplitude damping, and readout noise are mitigated most effectively; dephasing and gate noise remain partially challenging.

TABLE III: Noise sensitivity and mitigation impact (100 vs. 500 episodes).
<table><tr><td>Noise NM100 M100 NM500 M500</td></tr><tr><td>Depolarizing 0.86 0.93 0.87</td></tr><tr><td>0.94 0.85 0.92 0.86 0.93</td></tr><tr><td>Amplitude Damping Dephasing 0.84 0.91 0.85</td></tr><tr><td>0.92 0.84 0.90 0.85 0.92</td></tr><tr><td>Gate Noise Readout 0.85 0.92 0.86 0.93</td></tr></table>

![](images/e859fa54a1b1fa77594e8bc930288315a4283911cf9a7331969515d76ca9125d.jpg)  
Fig. 3: Noise sensitivity across five families over 500 episodes. Mitigation sustains approximation ratios above 0.92 even at noise level 0.1, well above the 0.84–0.87 unmitigated range.

Adaptive dynamics. Table IV and Fig. 4 report controller behavior, the central finding of this work.

Under the short horizon the controller strongly favors CDR (8,332 selections), a conservative preference for calibration-based correction when data are scarce. Under the long horizon selections balance across ZNE (24,067), PEC (24,485), CDR (19,305), and REM (20,692), with success rates converging. This shift from a single dominant technique to diversified deployment indicates genuine regime-dependent selection and coadaptation between policy learning and mitigation.

TABLE IV: Adaptive mitigation dynamics (usage counts).
<table><tr><td>Tech.</td><td>U100</td><td>S100</td><td>U500</td><td>S500</td></tr><tr><td>ZNE</td><td>3,498</td><td>High</td><td>24,067</td><td>High</td></tr><tr><td>PEC</td><td>3,884</td><td>High</td><td>24,485</td><td>High</td></tr><tr><td>CDR</td><td>8,332</td><td>Mod.</td><td>19,305</td><td>High</td></tr><tr><td>REM</td><td>1,962</td><td>Mod.</td><td>20,692</td><td>High</td></tr></table>

![](images/e4bb932a8ea71c028a2a62bd35d3b2dd270aa57a7f24634c0fb74be6606d57bf.jpg)  
Fig. 4: APGEM selection dynamics over 500 episodes. The controller distributes usage across ZNE, PEC, CDR, and REM as success rates converge, reflecting mature policy and mitigation co-adaptation.

## VIII. DISCUSSION

Two patterns stand out. First, mitigation is not optional under realistic noise: at the highest severities it is the difference between sub-parity performance and approximation ratios near 0.94. Second, the value of any single technique is regime-dependent. PEC leads under short horizons where its unbiased correction matters most, whereas CDR leads under long horizons as accumulated calibration data sharpen its regression. A static choice would lock in one of these regimes and forfeit the other; APGEM instead tracks the transition, which is visible as the shift from CDR-dominated selection to balanced deployment. Grounding the study in a real Delhi last-mile scenario shows the controller operating under the spatial structure of an operational routing task rather than a synthetic distribution, strengthening the practical reading of these results. The evaluation still centers on a single instance, which we treat as a focused case study; cross-instance benchmarking on standard suites is the natural next step and is already framed as future work.

## IX. CONCLUSION

We presented APGEM, an adaptive, learning-aware error mitigation controller for QRL applied to the CVRP. By selecting among ZNE, PEC, CDR, and REM online through a fidelity, entropy, and cost aware utility, APGEM restores approximation ratios under high noise and exhibits regime-dependent selection that matures from a CDR-dominated short horizon to balanced longhorizon deployment. Grounding the study in a real Delhi last-mile scenario demonstrates the controller under the spatial structure of an operational routing task, and these preliminary results support adaptive mitigation as a practical path toward noise-resilient QRL on NISQ hardware. Building on this case study, the framework extends naturally to standard CVRPLIB suites for crossinstance benchmarking, richer VRP variants, qubitefficient encodings for larger instances, and deployment on real quantum hardware.

## REFERENCES

[1] T. Vidal, “Hybrid genetic search for the cvrp: Open-source implementation and swap\* neighborhood,” Computers & Operations Research, vol. 140, p. 105643, 2022.

[2] A. Mor and M. G. Speranza, “Vehicle routing problems over time: a survey,” Annals ofOperations Research, vol. 314, no. 1, pp. 255–275, 2022.

[3] S. Ghosal, C. P. Ho, and W. Wiesemann, “A unifying framework for the capacitated vehicle routing problem under risk and ambiguity,” Operations Research, vol. 72, no. 2, pp. 425– 443, 2024.

[4] S. Sabet and B. Farooq, “Green vehicle routing problem: State of the art and future directions,” IEEE Access, vol. 10, pp. 101 622–101 642, 2022.

[5] J. Los, F. Schulte, M. Gansterer, R. F. Hartl, M. T. Spaan, and R. R. Negenborn, “Large-scale collaborative vehicle routing,” Annals of Operations Research, pp. 1–33, 2022.

[6] J. Ochelska-Mierzejewska, A. Poniszewska-Maranda, and ´ W. Maranda, “Selected genetic algorithms for vehicle routing´ problem solving,” Electronics, vol. 10, no. 24, p. 3147, 2021.

[7] K. Bouanane, M. E. Amrani, and Y. Benadada, “The vehicle routing problem with simultaneous delivery and pickup: a taxonomic survey,” International Journal of Logistics Systems and Management, vol. 41, no. 1-2, pp. 77–119, 2022.

[8] J. Preskill, “Quantum computing 40 years later,” in Feynman Lectures on Computation. CRC Press, 2023, pp. 193–244.

[9] K. Bharti, A. Cervera-Lierta, T. H. Kyaw, T. Haug, S. Alperin-Lea, A. Anand, M. Degroote, H. Heimonen, J. S. Kottmann, T. Menke et al., “Noisy intermediate-scale quantum algorithms,” Reviews of Modern Physics, vol. 94, no. 1, p. 015004, 2022.

[10] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio et al., “Variational quantum algorithms,” Nature Reviews Physics, vol. 3, no. 9, pp. 625–644, 2021.

[11] M. Kolle, T. Witter, T. Rohe, G. Stenzel, P. Altmann, and¨ T. Gabor, “A study on optimization techniques for variational quantum circuits in reinforcement learning,” in 2024 IEEE International Conference on Quantum Software (QSW). IEEE, 2024, pp. 157–167.

[12] F. Moosavi and B. Farooq, “Quantum-efficient reinforcement learning solutions for last-mile on-demand delivery,” in 2025 IEEE International Conference on Quantum Artificial Intelligence (QAI). Naples, Italy: IEEE, 2025, arXiv:2508.09183.

[13] Y. Kim, C. J. Wood, T. J. Yoder, S. T. Merkel, J. M. Gambetta, K. Temme, and A. Kandala, “Scalable error mitigation for noisy quantum circuits produces competitive expectation values,” Nature Physics, vol. 19, no. 5, pp. 752–759, 2023.

[14] E. Pelofske and V. Russo, “Digital zero-noise extrapolation with quantum circuit unoptimization,” arXiv preprint arXiv:2503.06341, 2025.

[15] P. D. Nation, H. Kang, N. Sundaresan, and J. M. Gambetta, “Scalable mitigation of measurement errors on quantum computers,” PRX Quantum, vol. 2, no. 4, p. 040326, 2021.

[16] C. Ding, X.-Y. Xu, S. Zhang, H.-L. Huang, and W.-S. Bao, “Evaluating the resilience of variational quantum algorithms to leakage noise,” Physical Review A, vol. 106, no. 4, p. 042421, 2022.

[17] H. Liao, D. S. Wang, I. Sitdikov, C. Salcedo, A. Seif, and Z. K. Minev, “Machine learning for practical quantum error mitigation,” Nature Machine Intelligence, vol. 6, no. 12, pp. 1478–1486, 2024.

[18] A. Muqeet, S. Ali, T. Yue, and P. Arcaini, “A machine learning-based error mitigation approach for reliable software development on ibm’s quantum computers,” in Companion Proceedings of the 32nd ACM International Conference on the Foundations of Software Engineering (FSE). ACM, 2024, pp. 80–91.

[19] M. M. Yousuf and S. A. Sofi, “A systematic exploration of quantum software engineering in the nisq era: Methods, lifecycle practices, and a taxonomy of challenges,” Neurocomputing, p. 132809, 2026.

[20] P. Czarnik, A. Arrasmith, P. J. Coles, and L. Cincio, “Error mitigation with clifford quantum-circuit data,” Quantum, vol. 5, p. 592, 2021.

[21] H. Fakhravar, “Combining heuristics and exact algorithms: A review,” arXiv preprint arXiv:2202.02799, 2022.

[22] D. Ambrosino and C. Cerrone, “A rich vehicle routing problem for a city logistics problem,” Mathematics, vol. 10, no. 2, p. 191, 2022.

[23] B. Majid, S. A. Sofi, and Z. Jabeen, “Quantum machine learning: a systematic categorization based on learning paradigms, nisq suitability, and fault tolerance,” Quantum Machine Intelligence, vol. 7, no. 1, pp. 1–55, 2025.

[24] P. Lamichhane and D. B. Rawat, “Quantum machine learning: Recent advances, challenges and perspectives,” IEEE Access, 2025.

[25] T. Giurgica-Tiron and et al., “Digital zero noise extrapolation for quantum error mitigation,” PRX Quantum, vol. 2, no. 1, p. 010323, 2021.

[26] A. Mari, N. Shammah, and W. J. Zeng, “Extending quantum probabilistic error cancellation by noise scaling,” Physical Review A, vol. 104, no. 5, p. 052607, 2021.

[27] Z. Cai, R. Babbush, S. C. Benjamin, S. Endo, W. J. Huggins, Y. Li, J. R. McClean, and T. E. O’Brien, “Quantum error mitigation,” Reviews of Modern Physics, vol. 95, no. 4, p. 045005, 2023.