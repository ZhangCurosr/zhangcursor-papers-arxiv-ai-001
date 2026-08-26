# QML for Quantum Sensing under Measurement-Induced Information Loss

Sounak Bhowmik and Himanshu Thapliyal

Department of Electrical and Computer Engineering

Southern Methodist University, Dallas, TX, USA

sbhowmik@smu.edu, hthapliyal@smu.edu

Abstract—Nitrogen-vacancy (NV) centers in diamond can serve as highly sensitive solid-state quantum sensors for highsensitivity magnetometry. However, in the noisy intermediatescale quantum (NISQ) era, extracting reliable information from noisy, finite-shot, and measurement-limited sensing data remains a considerable challenge. Whereas, quantum machine learning (QML) offers a potential path to improve parameter estimation by learning nonlinear relationships between quantum-sensing data and the underlying physical signal. In this work, we investigate the role of QML in magnetic-field estimation within an NV center-inspired magnetometry setting. We formulated magnetic field sensing as a supervised regression task. We compared the performance of several classical machine learning models trained on measurement-based classical data with that of quantum kernel-based models trained on pre-measurement coherent quantum states. Our objective is to isolate the impact of measurement-induced information loss and therefore provide a theoretical upper bound on the sensing performance. The upper bound is achievable only when coherent quantum information is directly available to the learning model. Our results show that QML-based sensing performance improves significantly with coherent quantum-state information, and not much with changes in model complexity or learning paradigm. This observation underscores the importance of learning pipelines that tightly integrate quantum sensors and QML models to enhance magnetic field sensing under realistic constraints.

Index Terms—Nitrogen-vacancy centers, NV center magnetometry, quantum machine learning, quantum sensing, Ramsey interferometry

## I. INTRODUCTION

Quantum sensing [1], [2] has emerged as one of the most promising near-term applications of quantum technology, with the potential to improve precise measurements beyond the limits of many classical sensing platforms. Among the leading solid-state sensing systems, negatively charged nitrogenvacancy (NV) centers in diamond have attracted significant attention. NV centers as sensors offer a combination of atomicscale spatial resolution, optical initialization and readout, microwave spin control, and room-temperature operation [3], [4]. These properties make the NV center a powerful platform for measuring magnetic fields, electric fields, temperature, strain, and other local physical quantities with high sensitivity.

In NV center magnetometry, the physical signal is encoded into the quantum state of the sensor through coherent spin evolution. For example, in Ramsey interferometry, an external magnetic field can produce a phase shift in the NV spin state, which is subsequently estimated from measurement outcomes [5]. Although the underlying quantum state contains rich information about the magnetic field, practical experiments cannot directly access the full coherent state. Instead, they provide finite measurement statistics obtained after projective or optical readout. This distinction is critical. Noise, decoherence, shot noise, and limited sampling can remove or obscure information before it becomes available to a learning algorithm. As a result, the performance of any downstream inference method is constrained not only by the model architecture but also by the information it can preserve in the measured data.

Recent advances in quantum machine learning (QML) have motivated new approaches for signal processing, parameter estimation, and data-driven inference in quantum sensing systems [6], [7]. QML models are often motivated by their ability to represent data in high-dimensional quantum feature spaces and therefore exploit coherent quantum dynamics for learning. However, in realistic sensing workflows, it remains unclear whether QML provides an advantage with only postmeasurement classical data available. Therefore, an important question arises: does the improvement in sensing performance primarily arise from the expressive power of the learning model or from access to coherent quantum information before measurement?

![](images/3855dacbf37075b8bde54b63ab0e193186c594f7b7e94eed4a071bcc3a66203b.jpg)  
Fig. 1: Level diagram of a nitrogen-vacancy center in diamond, where a substitutional nitrogen atom and an adjacent vacancy form a spin defect used for nanoscale magnetometry.  
Without any external magnetic field, the energies of the orbital ground states, ∈ {|0⟩ , |±1⟩}, split, called zero field splitting (ZFS). In the presence of an external magnetic field, $B _ { N V }$ along the quantization axis of the crystal, the degenerate ground states, $| + 1 \rangle$ and $| { \bar { - } } 1 \rangle$ , split up, which is referred to as Zeeman splitting. Once optically excited, the |±1⟩ states decay through metastable singlet states to the ground state, |0⟩, leading to optical initialization and spin-dependent fluorescence to be read optically to estimate the unknown magnetic field.

In this work, we investigate this question through a controlled sensing experiment inspired by NV center magnetometry. We formulate magnetic-field estimation as a supervised regression task and evaluate various classical machine learning models using measurement statistics from Ramsey interferometry experiments. Afterward, we compare these results with a quantum kernel ridge regression model, trained directly on premeasurement coherent quantum states represented as density matrices. This quantum-kernel setting provides a theoretical upper bound on the information available before measurementinduced loss occurs.

Our results aim to demonstrate the bottleneck imposed by measurement-induced information loss on the learningbased quantum sensing task. We observed that the quality and completeness of the available readout data limit the classical models trained on the measured statistics. Even when the underlying sensing dynamics contain stronger field-dependent structure, these models deviate far from the theoretical upper bound. In contrast, when we trained the quantum kernel models operating on coherent density matrices, the estimation performance improved substantially. This observation underscores the impact of measurement-induced information. It suggests that QML’s success in quantum sensing depends largely on preserving, accessing, and effectively encoding the coherent quantum data before readout.

The main contributions of this work are threefold:

1) We formulate an NV center-inspired Ramsey magnetometry experiment as a regression benchmark for studying learning under realistic measurement constraints.

2) We quantify the impact of measurement-induced information loss by comparing the performance of the classical models trained on finite measurement statistics with quantum kernel models trained on coherent quantum state representations.

3) We establish a theoretical upper bound on sensing performance using quantum kernel ridge regression over pre-measurement density matrices generated during the sensing evolution.

The rest of this paper is organized as follows. Section II introduces NV center-inspired magnetometry, the sensing protocol, and the measurement strategy. Section III describes the experimental setup, simulation parameters, and noise model that have been used in this experiment. Section IV depicts how different models learn from the classical and quantum data, highlighting their differences in terms of learning strategy. Section V presents the performance analysis and discusses the effect of measurement-induced information loss. Finally, section VI concludes the paper and outlines future research directions.

## II. BACKGROUND

## A. NV centers in Diamond

Nitrogen-vacancy (NV) centers in diamond are atomicscale point defects formed by a nitrogen atom, adjacent to a vacant carbon lattice site. In its negatively charged state, the NV center has an electronic spin-triplet ground state, $m _ { s } ~ = ~ 0 , + 1 , - 1$ , making it a robust solid-state quantum sensor. As shown in the figure 1, the nitrogen vacancy (NV) axis defines the spin system’s quantization axis. Even in the absence of an external magnetic field, the ground-state energy levels are split by a phenomenon called zero-field splitting (ZFS) [3], [5], separating the |0⟩ state from the degenerate |+1⟩ and |−1⟩ states by approximately 2.87 GHz. When a magnetic field component $B _ { N V }$ is projected along the NV axis, the |±1⟩ states undergo Zeeman splitting [8], producing an energy separation proportional to $2 \gamma B _ { N V }$ , where $\gamma$ is the gyromagnetic ratio of electronic spin. This magnetic-fielddependent splitting is the basis of NV center magnetometry.

Figure 1 further illustrates the optical spin-readout mechanism. Under green laser excitation, the NV center emits red fluorescence. However, the fluorescence intensity depends on the spin state as the |±1⟩ states have a higher probability of decaying through intermediate singlet states before returning to |0⟩. This spin-dependent fluorescence enables optical initialization and readout, while microwave excitation drives transitions between |0⟩ and |±1⟩. By monitoring shifts in the optically detected magnetic resonance spectrum, local magnetic fields can be measured with nanoscale spatial resolution under ambient conditions.

NV magnetometry experiments are usually restricted to two-level systems, formed by $m _ { s } ~ = ~ \{ 0 , - 1 \}$ . This setup is a suitable qubit representation for Ramsey interferometry [Fig. 2(A)] [1], [5].

## B. NV Inspired Quantum Sensing Simulation

To isolate only the field-dependent dynamics of the NV center, we consider only the Zeeman interaction contributing to the effective Hamiltonian [4], [9] of the system under observation: $\mathcal { H } = \hbar \gamma B S _ { z }$ . Where B is the unknown magnetic field, our parameter of interest. Time evolution of this Hamiltonian will produce a reliable simulation of a physical NV center system:

$$
U ( t ) = e ^ { - i \mathcal { H } t } = e ^ { - i \hbar \gamma B t S _ { z } }\tag{1}
$$

For $\begin{array} { r } { S _ { z } = { \frac { 1 } { 2 } } \sigma _ { z } , \mathcal { H } = { \frac { \hbar } { 2 } } B t \sigma _ { z } } \end{array}$ , the unitary evolution can be described as equation 2:

$$
U ( t ) = e ^ { - i \hbar \gamma B t \sigma _ { z } / 2 } = R _ { Z } ( \phi )\tag{2}
$$

for $\phi = \gamma B t$ , considering ℏ = 1. Therefore, our quantum circuit to simulate the Hamiltonian of a two-level NV center model consists of a rotation gate around the Z-axis, with an angle that depends on the magnetic field and time.

However, initially, the basis state gets transcended to a highly sensitive superposition state, with the application of a <sup>π</sup><sub>2</sub> -pulse, essentially a Hadamard gate:

![](images/0a342a5bdfea4c84ebd54ad8925fd987763ed2a203b19fffd555090f0db77067.jpg)  
Fig. 2: Magnetic field estimation using NV center-inspired quantum sensing simulation. (A) Ramsey Interferometry simulation: $\theta _ { i } =$ $\gamma { \bar { B _ { i } } } t _ { k } ; \rho _ { i } ( t _ { k } ) $ is the corresponding density matrix, after the system evolves under a magnetic field $B _ { i }$ for time $t _ { k } ; { \bf { x } } _ { \mathrm { { i } } }$ is the corresponding expectation value of the observable. (B) Regression problem to estimate the unknown magnetic field: $\hat { B } ^ { \rho }$ and $\hat { B } ^ { \langle x \rangle }$ are the estimated magnetic field values based on coherent-state and classical measurement statistics, respectively.

$$
\left. \psi _ { 0 } \right. = H \left. 0 \right. = \frac { \left. 0 \right. + \left. 1 \right. } { \sqrt { 2 } }\tag{3}
$$

Afterward, we evolve this initial sensing state |ψ⟩ under the influence of $U ( t )$

$$
\left. \psi _ { 1 } \right. = e ^ { - i \gamma B t / 2 } \left. \psi _ { 0 } \right.\tag{4}
$$

Finally, we apply a second Hadamard gate to bring $\psi _ { 1 }$ back to the computational basis [Fig. 2(A)].

Then we measure the system and process the data to estimate the parameter of interest, the magnetic field, B. However, before measurement, the magnetic field is encoded in the offdiagonal elements of the density matrix; measurement in the Z-basis irreversibly discards the phase information, leading to irreversible data loss.

$$
\rho ( t _ { k } ) = \frac { 1 } { 2 } \left( \begin{array} { c c } { { 1 } } & { { e ^ { - i \phi } } } \\ { { } } & { { } } \\ { { e ^ { i \phi } } } & { { 1 } } \end{array} \right)\tag{5}
$$

Where $\phi = \gamma B t _ { k }$ , and $t _ { k }$ is the time of evolution.

## III. EXPERIMENTAL SETUP

To examine the true quantum advantage offered by QML, we designed a controlled sensing experiment inspired by NV center magnetometry. Figure 2 shows the experimental pipeline used to estimate the magnetic field from the NV center simulation. We begin by preparing the basis states, transforming into a more sensitive superposition state, and simulating their unitary time evolution under an unknown magnetic field [refer to equation 1]. The resulting data are then collected and used to train the following regression models [refer to Fig. 2(B)]:

• Classical machine learning models on the expectation values $( \{ \langle Z \rangle _ { t _ { 1 } } , \langle Z \rangle _ { t _ { 2 } } , . . . , \langle Z \rangle _ { t _ { M } } \} )$ , and

• Quantum fidelity-kernel ridge regression model on coherent quantum states $( \mathcal { R } _ { i } = \{ \rho _ { i } ( t _ { 1 } ) , \rho _ { i } ( t _ { 2 } ) , \dots , \rho _ { i } ( t _ { M } ) \} )$

To better capture the physical behavior of the NV center, we incorporated a realistic noise model [1], [5], including a phase-damping channel with characteristic time $T _ { 2 } = 1 0 0 ~ \mu s$ Measurement noise is modeled as a 10% classical bit-flip readout error, and shot noise is introduced through binomial sampling of the measurement outcomes. The system evolves at time points $t _ { k } \in \{ 5 , 1 0 , 2 0 , 4 0 , 6 0 \} \ \mu .$ $\mu s$ . The noise model and the parameters describing the sensing dynamics are kept fixed throughout all experiments. Table I shows a list of the parameters used in this experiment.

While recording the dataset, we stored the density matrix prior to measurement, and later used it as the quantum data to train the QML models. In contrast, the post-measurement expectation values are treated as classical data in classical regression models.

In the next section, we shed more light on this distinction while building the learning framework for classical and quantum machine learning models.

## IV. LEARNING FROM CLASSICAL AND QUANTUM DATA

The central objective of this study is to separate the effect of model expressivity from the effect of quantum data accessibility. In this experiment, we constructed two learning pipelines that use the same underlying NV centered-inspired sensing dynamics but differ in the information available to the learner. The classical pipeline receives only measurementderived features after projective readout. In contrast, the quantum pipeline operates on the pre-measurement density matrices generated during the coherent sensing evolution. This distinction allows us to evaluate how much information is lost during measurement and how this loss affects magnetic-field estimation.

TABLE I: Simulation and learning parameters used in the NV center-inspired sensing benchmark.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Magnetic field range</td><td> $B \in [ 0 , 2 ] \ \mu T$ </td></tr><tr><td>Training samples</td><td>800</td></tr><tr><td>Test samples</td><td>100</td></tr><tr><td>Evolution times</td><td>{5, 10, 20, 40, 60} µs</td></tr><tr><td>Random seeds</td><td>3</td></tr><tr><td>Dephasing time</td><td> $T _ { 2 } = 1 0 0 ~ \mu s$ </td></tr><tr><td>Readout error</td><td>10% bit-flip noise</td></tr><tr><td>Shot noise</td><td>Binomial sampling</td></tr><tr><td>Evaluation metric</td><td>RMSE on B</td></tr><tr><td>Classical baselines</td><td>LRR, RBF-KRR, MLP</td></tr><tr><td>Quantum model</td><td>Fidelity-kernel ridge regression</td></tr></table>

RMSE: Root mean squared error, LRR: Linear ridge regression, RBF: Radial basis function, KRR: Kernel ridge regression, MLP: Multilayer perceptron

## A. Classical Learning From Measurement Records

In the classical learning pipeline, each training instance is generated by sampling a magnetic field value $B _ { i }$ between $0 - 2 \mu T$ and running the Ramsey sensing protocol through multiple evolution times $t _ { k } .$ . After each evolution, the resultant state is measured and converted into a finite-shot estimate of an observable expectation value. Therefore, the classical learner does not access the quantum state directly; rather, it only receives a feature vector constructed from measurement statistics.

We consider two classical feature representations. The first is a Z-only feature map, where the input is given by

$$
\mathbf { x } _ { i } ^ { ( Z ) } = \left[ \langle Z \rangle _ { i , t _ { 1 } } , \langle Z \rangle _ { i , t _ { 2 } } , \dots , \langle Z \rangle _ { i , t _ { M } } \right] .\tag{6}
$$

This setting represents a restricted measurement protocol in which only computational-basis readout is available. Since the magnetic field is encoded as a phase during the Ramsey evolution, this feature map loses information about it during the measurement protocol.

The second representation is an XY Z feature map, where measurements are performed in three Pauli bases:

$$
\begin{array} { r } { \mathbf { x } _ { i } ^ { ( X Y Z ) } = [ \{ \langle X \rangle _ { i , t _ { 1 } } , \langle Y \rangle _ { i , t _ { 1 } } , \langle Z \rangle _ { i , t _ { 1 } } \} , \dots , } \\ { \{ \langle X \rangle _ { i , t _ { M } } , \langle Y \rangle _ { i , t _ { M } } , \langle Z \rangle _ { i , t _ { M } } \} ] } \end{array}\tag{7}
$$

This representation provides a slightly richer classical description of the resultant state because it partially captures the coherence through measurements in complementary bases. However, limitations regarding finite-shot sampling, readout error, and the need to estimate quantum-state information from classical measurement outcomes persist.

For the linear classical baseline model, we use ridge regression and solve

$$
\underset { \mathbf { w } } { \operatorname* { m i n } } \sum _ { i = 1 } ^ { N } \left( B _ { i } - \mathbf { w } ^ { T } \mathbf { x } _ { i } \right) ^ { 2 } + \lambda \| \mathbf { w } \| ^ { 2 } ,\tag{8}
$$

where $\mathbf { x } _ { i }$ denotes either the Z-only or XYZ classical feature vector, $B _ { i }$ is the target magnetic field, and λ is the regularization strength. We also evaluate nonlinear classical baselines, including RBF-kernel ridge regression and a multilayer perceptron, to separate measurement-induced information loss from limitations arising from the limited linear model capacity.

## B. Quantum Learning From Pre-Measurement Quantum States

The quantum machine learning pipeline uses the same physical sensing process but assumes access to the full density matrices before final measurement. For each sampled magnetic field $B _ { i }$ , the Ramsey protocol generates a trajectory of quantum states across the selected evolution times (t<sub>k</sub>):

$$
\mathcal { R } _ { i } = \left\{ \rho _ { i } ( t _ { 1 } ) , \rho _ { i } ( t _ { 2 } ) , \textrm { \ldots } , \rho _ { i } ( t _ { M } ) \right\} .\tag{9}
$$

This representation preserves the coherent phase information encoded during sensing within the off-diagonal elements of the density matrix.

To learn from these quantum-state trajectories, we use kernel ridge regression with a fidelity-based quantum kernel. For two sensing trajectories $\mathcal { R } _ { i }$ and $\mathcal { R } _ { j }$ , the kernel is defined as

$$
k ( \mathcal { R } _ { i } , \mathcal { R } _ { j } ) = \frac { 1 } { M } \sum _ { k = 1 } ^ { M } F \left( \rho _ { i } ( t _ { k } ) , \rho _ { j } ( t _ { k } ) \right) ^ { 2 } ,\tag{10}
$$

where $F ( \rho , \sigma )$ is the Uhlmann fidelity [10] between two density matrices:

$$
F ( \rho , \sigma ) = \mathrm { T r } \left[ \sqrt { \sqrt { \rho } \sigma \sqrt { \rho } } \right] .\tag{11}
$$

The predicted magnetic field for a test trajectory $\mathcal { R }$ is then given by

$$
\hat { B } ( \mathcal { R } ) = \sum _ { i = 1 } ^ { N } \alpha _ { i } k ( \mathcal { R } , \mathcal { R } _ { i } ) ,\tag{12}
$$

where the coefficients α are obtained by solving

$$
( K + \lambda I ) \pmb { \alpha } = \mathbf { B } .\tag{13}
$$

Here, K is the Gram matrix with entries $K _ { i j } = k ( \mathcal { R } _ { i } , \mathcal { R } _ { j } )$ and B contains the training magnetic-field values.

![](images/58f29dc373522c9904b1400fb75d8e622055cbbe1c52c47a52cbad343e0fbc09.jpg)  
Fig. 3: NV center-inspired magnetometry regression (mean ± std over seeds): Depicts the theoretical upper bound, and its performance gap with the models learning from the measurement data. The estimation error in the latter is 100 times more. Sensing pipelines that are more closely integrated with the coherent quantum states, or a more sophisticated measurement strategy, can help to close this gap and provide more accurate results.

## C. Oracle and Tomography-Limited Quantum Kernels

We evaluate two forms of the quantum kernel. The first is an oracle kernel computed from the exact pre-measurement density matrices $\rho _ { i } ( t _ { k } )$ . This setting is not intended to represent a directly available experimental measurement pipeline. Instead, it provides an upper bound on the performance that could be achieved if the learner had direct access to coherent quantumstate information before measurement-induced information loss.

The second is a tomography-limited kernel computed from reconstructed density matrices. In this setting, the density matrix at each evolution time is estimated from measured Pauli expectation values:

$$
\hat { \rho } ( t _ { k } ) = \frac { 1 } { 2 } \left( I + \sum _ { \alpha \in \{ X , Y , Z \} } \left. \sigma _ { \alpha } ( t _ { k } ) \right. \sigma _ { \alpha } \right) .\tag{14}
$$

The tomography-based reconstructed state is still constrained by finite-shot sampling, readout noise, and basis-measurement overhead. Therefore, this setting provides a more realistic comparison between the quantum and classical learning methods.

The comparison between classical feature-based learning, tomography-limited quantum kernels, and oracle quantum kernels, therefore, isolates the central learning mechanism studied in this work. If the oracle quantum kernel outperforms the measurement-based models, the gain should signify the value of coherent quantum-state access.

## V. RESULTS

We evaluated the magnetic field regression performance using 800 train and 100 test samples. For each sampled field value $( B _ { i } \in [ 0 . 0 , 2 . 0 ] \mu T )$ , Ramsey data is generated under a realistic dephasing noise model at multiple evolution times. We compared the performances of the following models:

1) Linear ridge regression (LRR) on Z-only, as well as XYZ-measurement data.

2) Nonlinear classical models (RBF kernel ridge regression and multilayer perceptron (MLP)) using XY Zmeasurement data.

3) Quantum kernel ridge regression (Q-KRR, using the Uhlmann fidelity [10], equation 10, using (a) oracle density matrices, ρ, and (b) reconstructed density matrices using tomography, equation.14.

![](images/0122139890cce4102d6798d28352e23ce15be379c1bab8fb60597b5e85ef81f8.jpg)  
Fig. 4: Test RMSE (µT) at 2048 shots $( { \mathrm { m e a n } } \pm { \mathrm { s t d } } ) \colon$ : The RMSE is significantly lower in the theoretical upper bound, which highlights its gap with the measurement-based learning models.

Figure 3 shows the test root-mean squared error (RMSE), produced by the models as a function of total number of shots (mean±std over seeds). We observed that the linear models show monotonic improvement, with XYZ features generally outperforming Z-only features at moderateto-high shot counts. Whereas the nonlinear classical models (RBF/MLP) significantly outperform linear baselines across several regimes, particularly for Z-only features at higher shots, indicating that part of the estimation problem is nonlinear in the measurement statistics.

The oracle quantum kernel (KRR on exact ρ) provides a near-constant, substantially lower error floor $( \sim 7 \times 1 0 ^ { - 4 } \mu T )$ , which represents a theoretical upper bound devoid of any loss due to measurement [Figure 4]. In contrast, the tomographylimited quantum kernel tracks the performance of strong classical XYZ-based models, which demonstrates that realistic measurement constraints significantly reduce the gap between the quantum and classical models.

These results indicate that quantum advantage in learningbased quantum sensing might primarily stem from access to coherent quantum-state information rather than from model expressivity alone.

## VI. CONCLUSION

In this work, we investigated the role of data in quantum machine learning (QML) for quantum sensing via an NV centerinspired magnetometry simulation. We formulated magnetic field estimation as a supervised regression problem and compared classical learning models trained on post-measurement statistics with quantum kernel-based models trained on coherent pre-measurement quantum states. This comparison allowed us to isolate the effect of measurement-induced information loss and examine whether improved sensing performance arises from the learning model itself or from the type of data available to it.

Our results show that lossy measurements fundamentally constrain models trained solely on measured classical data. While classical models can learn useful correlations from finite-shot measurement statistics, their performance is limited once quantum-state information has been collapsed into classical outcomes. In contrast, quantum models operating on coherent quantum-state representations achieve much lower estimation error. They provide a theoretical upper bound on the sensing performance achievable when coherent quantum information is directly accessible for learning. These findings indicate that the potential advantage of QML in quantum sensing is not solely due to increased model complexity. Rather, it is strongly linked to the preservation and use of quantum coherence.

The study also highlights an important future direction for the design of quantum sensing systems. If sensing data are fully measured and converted into classical statistics before learning, a significant portion of the quantum advantage may already be lost. Therefore, future QML-enhanced sensing architectures should integrate sensing, state processing, and learning more closely within the quantum domain. Such quantum-native pipelines will reduce information loss and improve parameter estimation in near-term quantum sensing. In future work, we will extend this analysis to hardware experiments, more complex noise models, and hybrid sensinglearning architectures that can operate under practical device limitations. Our experimental outcomes suggest that the path toward useful QML for quantum sensing depends not only on designing stronger learning models, but also on preserving coherent quantum information in the learning data.

## REFERENCES

[1] C. L. Degen, F. Reinhard, and P. Cappellaro, “Quantum sensing,” Reviews of modern physics, vol. 89, no. 3, p. 035002, 2017.

[2] Z. Zhang and Q. Zhuang, “Distributed quantum sensing,” Quantum Science & Technology, vol. 6, no. 4, p. 043001, 2021.

[3] S. Hong, M. S. Grinolds, L. M. Pham, D. Le Sage, L. Luan, R. L. Walsworth, and A. Yacoby, “Nanoscale magnetometry with nv centers in diamond,” MRS bulletin, vol. 38, no. 2, pp. 155–161, 2013.

[4] R. Schirhagl, K. Chang, M. Loretz, and C. L. Degen, “Nitrogen-vacancy centers in diamond: nanoscale sensors for physics and biology,” Annual review of physical chemistry, vol. 65, no. 1, pp. 83–105, 2014.

[5] L. Rondin, J.-P. Tetienne, T. Hingant, J.-F. Roch, P. Maletinsky, and V. Jacques, “Magnetometry with nitrogen-vacancy defects in diamond,” Reports on progress in physics, vol. 77, no. 5, p. 056503, 2014.

[6] M. Schuld and N. Killoran, “Quantum machine learning in feature hilbert spaces,” Physical review letters, vol. 122, no. 4, p. 040504, 2019.

[7] K. S. Rawat and T. Sharma, “Emerging paradigm of quantum machine learning: Knowledge insights and future prospects,” IEEE Transactions on Artificial Intelligence, 2026.

[8] C. Dean, “Zeeman splitting of nuclear quadrupole resonances,” Phys. Rev., vol. 96, pp. 1053–1059, Nov 1954. [Online]. Available: https://link.aps.org/doi/10.1103/PhysRev.96.1053

[9] M. A. Nielsen and I. L. Chuang, Quantum computation and quantum information. Cambridge university press, 2010.

[10] P. Marian and T. A. Marian, “Uhlmann fidelity between two-mode gaussian states,” Physical Review A—Atomic, Molecular, and Optical Physics, vol. 86, no. 2, p. 022340, 2012.