# Fault-tolerant shallow recurrent decoders for sparse sensing with missing sensor measurements

Matteo Romilio Pasqual<sup>∗</sup>, Matteo Tomasetto<sup>∗∗</sup>, J. Nathan Kutz<sup>†</sup>, and Andrea Manzoni<sup>‡</sup>

∗ Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milano, Italy

∗∗ Department of Mechanical Engineering, Politecnico di Milano, Milano, Italy

<sup>†</sup> Autodesk Research, London, UK and

MOX - Department of Mathematics, Politecnico di Milano, Milano, Italy

Real-world sensing systems are inherently imperfect: sensors may fail, intermittently disconnect, operate with asynchronous sampling rates, and be subject to noise, leading to measurement streams that are irregular, incomplete, and heterogeneous. Nevertheless, sensing strategies for real-time state reconstruction typically assume synchronous and continuously active sensors, a constraint that substantially limits their applicability in operational environments. In this work, we introduce fault-tolerant sparse sensing strategies in the context of shallow recurrent decoders (SHRED) to enable high-dimensional state estimation from sparse, asynchronous, and missing sensor measurements. Specifically, we consider naive imputation strategies, fault-tolerant recurrent temporal encoders, and we introduce four diferent transformer-based masking mechanisms to make SHRED explicitly aware of sensor availability through the binary data-availability mask. Moreover, we further introduce a multi-step decoding training objective for SHRED, resulting in smoother latent trajectories and more robust predictions. Throughout a synthetic benchmark and three real-world problems in hydrodynamic and ocean modeling, we show that (i) transformer-based masking mechanisms outperform standard imputation techniques and fault-tolerant recurrent architectures, (ii) the multi-step decoding training objective enhances reconstruction fidelity and robustness in the presence of heterogeneous missing-data patterns, while (iii) the computational eficiency of SHRED is preserved, with no additional inference-time overhead.

## I. INTRODUCTION

Many real-world dynamical systems evolve in highdimensional state spaces that cannot be directly observed in their entirety [1, 26]. This challenge arises across a wide range of scientific and engineering applications, including fluid and solid mechanics, weather and climate modeling, plasma physics, and structural health monitoring. In these systems, sensors provide the primary in terface between theory and reality, yet their deployment is inevitably limited by cost, reliability, physical inaccessibility, and operational constraints. However, being able to access local measurements is crucial to, e.g., inferring full state dynamics, designing feedback control strategies, or predicting quantities of interest.

Recovering the state of a dynamical system from partial observations enables downstream tasks such as monitoring, forecasting, anomaly detection, and control, while avoiding expensive and impractical full-field measurements. Traditional approaches to state reconstruction are grounded in linear algebra frameworks. The singular value decomposition (SVD) – also known as proper orthogonal decomposition (POD) [4, 6, 31, 35] – provides an optimal low-rank approximation of high dimensional state data through a linear projection onto a low-dimensional basis of spatial functions capturing most of the data variability. After computing the dominant POD modes from the training data, the coeficients of the low-rank representation can be estimated from a limited number of point measurements by solving a linear inverse problem, provided that the sensing matrix is well-conditioned [22, 49]. However, the requirement for a well-conditioned sensing matrix imposes constraints on both the number and placement of sensors, leading to an NP-hard optimization problem for identifying the optimal configuration [12, 29, 30] and potentially limiting its applicability in real-world settings [11, 36]. Moreover, the reconstruction accuracy of this method is bounded by the expressiveness of the POD modes used, so that fine-scales or nonlinear features of the field cannot be recovered entirely when only a few modes are retained.

The advent of scientific machine learning has introduced new data-driven methods for addressing limitations of traditional linear methods [5, 23, 40]. Neural networks and, in particular, shallow decoder networks (SDNs), leverage the universal approximation theorem of feedforward architectures [14] to learn an end-to-end nonlinear mapping from sensor measurements directly to the corresponding high-dimensional state [8, 19, 34, 42], without requiring an explicit basis representation as in POD-based methods. Similarly, the Senseiver [43] employs an attention-based architecture inspired by the Perceiver IO [28] to reconstruct spatio-temporal fields at arbitrary query locations from sparse observations. Once trained on a set of snapshots, these strategies can capture nonlinear spatial correlations in the data that are inaccessible to linear methods and have been shown to outperform linear POD-based architectures in field reconstruction, particularly in ill-posed settings, i.e., with few or poorly placed sensors. Nevertheless, POD-based methods, SDNs, and the Senseiver all share an intrin sic limitation: they operate on static snapshots, treating each time instant independently and discarding the tem poral dynamics of the measurement signal. The tempo ral history of observations, naturally available whenever sensors operate continuously over time, is essential for capturing intrinsic dynamical features, such as temporal correlations between consecutive measurements.

![](images/4f5127cd77073f73ca6ed69d3efcfaf62b230f85f51e90fd49af5d85b0da65c2.jpg)  
FIG. 1. Overview of fault-tolerant shallow recurrent decoders for state reconstruction from sparse sensors under missing sensor measurements. Sensor measurements $\left\{ \mathbf { s } _ { k - L } , \ldots , \mathbf { s } _ { k } \right\}$ with possibly missing entries are padded, lagged, and processed by a fault-tolerant encoder E to produce a latent embedding $\mathbf { z } _ { k } ,$ , which is subsequently decoded into the POD coeficients $\hat { \mathbf { a } } _ { k }$ via the shallow decoder D. The state reconstruction at time $t _ { k }$ is then given by $\hat { \mathbf { u } } _ { k } = \Psi _ { r } \hat { \mathbf { a } } _ { k }$ , where Ψ<sub>r</sub> denotes the (possibly field-specific) r dominant POD modes. The upper panel illustrates the four Transformer-based masking strategies proposed where the binary data-availability mask ${ { \bf { M } } _ { k } }$ is injected at diferent levels of the encoder architecture, namely channel-wise masking, hidden-space masking, MHSA masking, and learned MHSA masking. The training box reports the standard last-step SHRED loss function and the multi-step decoding objective, where the decoder is supervised at intermediate time steps to encourage temporally consistent latent representations across the observation window.

The shallow recurrent decoder (SHRED) framework proposed by Williams et al. [51] addresses this limitation by combining a recurrent temporal encoder with a shallow spatial decoder. The encoder, which is typically modeled via recurrent neural networks such as, e.g., long short-term memory (LSTM) [25, 33] or gated recurrent unit (GRU) [10] networks, processes a sliding window of lagged sensor measurements, compressing their temporal history into a latent representation, which is subsequently mapped to the full field by a SDN. Thanks to the temporal history of sensor readings, SHRED achieves dramatically lower reconstruction errors than classical techniques, even with a very limited number of randomly placed sensors. Diferent SHRED extensions have recently been proposed in the literature. For instance,

Riva et al. [41] take into account an ensemble of SHRED models to cope with noisy measurements. Moreover, the SHRED framework has been extended to parametric systems [46, 47], demonstrating that the temporal history of a handful of sensors sufices to reconstruct and control high-dimensional fields across varying physical and geometrical parameters. Gao et al. [24], instead, introduce a regularization based on sparse identification of nonlinear dynamics (SINDy) [7] to promote interpretable latent dynamics, further improving forecasting performance and explainability in challenging scenarios. More recently, Yermakov et al. employ a Transformer [48] as temporal encoder in the so-called T-SHRED framework [52], thus exploiting the multi-head self-attention mechanism to capture long-range temporal dependencies among lagged sensor measurements.

SHRED has been successfully employed in a wide range of applications, while typically considering only simulated data, synchronous sensors with uninterrupted operation throughout the observation period, and point wise field evaluations as synthetic sensor readings. The deployment of SHRED in real-world environments must account for challenging sensing conditions, including incomplete, noisy, and asynchronous data streams [44]. This results in time-series that are irregular, with different sampling rates, and may contain gaps and missing values [37]. Moreover, in practical settings, sensor measurements do not exactly match direct pointwise evalua tions of the underlying state, but rather provide (possibly indirect) observations that are correlated with the system state through an observation process. Note that discarding all sensors that are not jointly available over the entire observation period severely degrades reconstruction qual ity, particularly in the ultra-sparse-sensor regime. Moreover, this naive strategy becomes infeasible when sensor observation windows do not overlap, as naturally occurs under fully asynchronous sampling. To overcome this limitation, CS-SHRED [15] and LAPIS-SHRED [3] employ, respectively, (possibly expensive) convex minimization problems and latent-space inference to recover missing sensor values and reconstruct spatio-temporal dynamics from corrupted or incomplete observations.

In this work, we consider diferent eficient faulttolerant sparse sensing strategies in the context of SHRED. First, naive imputation strategies are taken into account to infer missing sensor values, at the price of introducing additional estimation errors and potentially biasing the reconstructed field. In particular, standard baselines include mean, median, or last observation carried forward (LOCF) imputation. Rather than imputing missing sensor measurements before feeding the resulting sequence to the SHRED temporal encoder, we take into account recurrent models specifically designed to handle missing data in the context of time-series forecasting, such as GRU-D [9] and its attention-augmented variant attnGRU-D [53]. Finally, we introduce four diferent transformer-based masking mechanisms to conveniently handle missing and asynchronous sensor values in the SHRED temporal encoder. Specifically, we explore both fixed and learnable masking mechanisms applied at the channel level (channel-wise masking), at the hidden space level (hidden-space masking), or at the multi-head selfattention level (fixed or learned MHSA masking). Naive missing-data imputation strategies, fault-tolerant recurrent temporal encoders and the proposed masked shallow recurrent decoders are then compared on a synthetic benchmark and on three real-world problems in hydrodynamic and ocean modeling, where sensor data are not always available or synchronous. This work thus aims at providing practical guidance for the deployment of faulttolerant SHRED models in operational settings characterized by heterogeneous and incomplete sensor coverage.

A further contribution of this work concerns the training objective considered to fit SHRED. The temporal encoders within the original SHRED architecture and its extensions, such as T-SHRED, entail a hidden state for every time step received as input. However, only the last latent representation is passed to the decoder both during training and deployment, leaving the intermediate temporal representations entirely unexploited. This limitation is particularly consequential in the context of heterogeneous missing data: when sensor observations are irregular or partially absent, it is beneficial to leverage the temporal context accumulated across the lookback window to compensate for the missing information. Therefore, rather than relying on a SINDy regularization [24, 52], which requires careful dictionary selection and dedicated optimization routines, we propose a multi-step decoding training strategy. This approach is inspired by the auxiliary classifiers in GoogLeNet [45], which enforce representational regularity at intermediate layers of a deep convolutional neural network by attaching additional supervision at multiple depths. Under the proposed framework, we decode intermediate hidden states of the temporal encoder using the same spatial decoder, and combine the corresponding reconstruction losses into a weighted objective, thus providing supervision across multiple temporal depths. The multi-step decoding training strategy acts as a powerful regularizer, forcing the temporal encoder to build informative latent representations throughout the entire temporal window. By ensuring that the latent trajectory remains well-structured at every step, this strategy leads to improved reconstruction accuracy and enhances the model robustness to missing data. Finally, during inference, the additional decoding heads are discarded, allowing the architecture to reduce to the standard, computationallyeficient SHRED pipeline. An overview of fault-tolerant SHRED architectures, with particular emphasis on the Transformer-based masking strategies proposed, is available in Figure 1.

This work is organized as follows. Section II introduces the sparse sensing problem and reviews different state-of-the-art sensing methods, such as gappy POD, SDNs, and SHRED. Section III presents diferent fault-tolerant sparse sensing strategies in the context of SHRED. Specifically, it presents standard imputation techniques, fault-tolerant recurrent temporal encoders, and introduces four diferent Transformer-based masking mechanisms, namely channel-wise, hidden-space, MHSA, and learned MHSA masking. Moreover, it presents the multi-step decoding training objective, detailing its mo tivation and formulation within the SHRED training pipeline. Section IV shows the numerical results of the diferent fault-tolerant sensing strategies on a synthetic benchmark and three real-world problems in the context of hydrodynamic and ocean modeling. Finally, Section V summarizes the main findings, discusses the practical implications of the proposed contributions, and outlines directions for future work.

## II. SPARSE SENSING

This section introduces the sensor-based state reconstruction problem, recalling diferent sensing methods, ranging from linear strategies to nonlinear and recurrent frameworks.

Let $u ( { \bf x } , t )$ be the high-dimensional state of a dynamical system in the space-time domain $\Omega \times [ 0 , T ]$ , where $\mathbf { x } \in \hat { \Omega } \subset \mathbb { R } ^ { D }$ denotes the spatial coordinate and $i \in [ 0 , T ]$ the temporal variable, with final time $T > 0$ . We consider state evolutions described by means of partial diferential equations (PDEs) in the form

$$
\dot { u } = \mathcal { N } ( u , \mathbf { x } , t ) ,\tag{1}
$$

where $\mathcal { N } ( \cdot )$ is a generic (possibly nonlinear) diferential operator modeling the time derivative of the state ˙u. The governing PDE in Equation (1) is, in general, spatially discretized through, $\mathrm { e . g . }$ , finite elements or finite volumes [13, 31], yielding a finite dimensional discrete state vector ${ \bf u } ( t ) \doteq \dot { \mathbb { R } } ^ { N }$ evolving over time, where N denotes the number of degrees of freedom required to capture the state dynamics. Moreover, a finite dimensional time grid $\{ t _ { 1 } , \ldots , t _ { m } \}$ spanning [0, T] is usually employed to discretize the problem over time, thus resulting in the state vectors $\mathbf { u } _ { k } = \mathbf { u } ( t _ { k } ) \in \mathbb { R } ^ { N }$ for $k = 1 , \ldots , m$

In many practical scenarios, it is impossible to directly capture the full state due to a remarkably high dimension N, or to physical constraints. Instead, the system is often observed through $n \ll N$ sparse sensors, locally monitoring the state evolution at a few locations within the domain Ω over time $t _ { 1 } , \ldots , t _ { m }$ . Formally, the state-to-sensor map is represented by the sensing operator $\mathcal { C } : \mathbb { R } ^ { N }  \mathbb { R } ^ { n }$ , mapping the full state to the corresponding sensor readings. For instance, in the simplest case of pointwise sampling, the sensing operator reduces to a matrix $\mathbf { C } \in \mathbb { R } ^ { n \times N }$ obtained by selecting n rows of the $N \times N$ identity matrix. Note that, in general, C may embody a nonlinear observation process, thus allowing sensors to measure quantities that may difer from those to be reconstructed. The sensor measurements are thus given by

$$
\mathbf { s } _ { k } = { \mathcal { C } } ( \mathbf { u } _ { k } ) + \pmb { \eta } _ { k } \in \mathbb { R } ^ { n } , \quad { \mathrm { f o r ~ } } k = 1 , \dots , m ,\tag{2}
$$

where $\pmb { \eta } _ { k } \in \mathbb { R } ^ { n }$ for $k = 1 , \ldots , m$ denotes measurement noise. The noise components are usually assumed to be independent and identically distributed, as they typically arise independently from each sensor measurement system rather than from correlated physical sources, and are often modeled as Gaussian.

The sparse sensing problem consists of finding a reconstruction mapping $\mathcal { R } : \mathbb { R } ^ { n } \to \mathbb { R } ^ { N }$ that accurately recovers the high-dimensional state $\mathbf { u } _ { k }$ from the noisy, lowdimensional observations $\mathbf { s } _ { k }$ for $k = 1 , \ldots , m$ . In other words, the sparse sensing problem aims at finding the inverse of the sensing operator C. However, this is generally an underdetermined and ill-posed problem, as $n \ll N$ limited and sparse observations do not uniquely determine the underlying high-dimensional state. To address this challenge, several strategies have been developed in the literature, as discussed in the following section.

## A. From linear to nonlinear sensing strategies

Given the severely underdetermined nature of the state reconstruction problem from limited sensor readings, one may leverage the intrinsic low-rank structure of the data. Let us collect m snapshots of the N-dimensional state vector at diferent time instances $t _ { 1 } , \ldots , t _ { m }$ into the socalled snapshot or data matrix

$$
\mathbf { X } = \left[ \mathop { \bf u } _ { 1 } ~ \mathop { \bf u } _ { 2 } ~ \cdot ~ \cdot ~ \mathop { \bf u } _ { m } \right] \in \mathbb { R } ^ { N \times m } .\tag{3}
$$

The singular value decomposition (SVD), commonly referred to as proper orthogonal decomposition (POD) [4, 6], provides the optimal (in a least-squares or statistical sense [18]) data-driven low-rank approximation of the matrix X, and thus constitutes a valuable option for addressing the reconstruction task. Specifically, the POD factorizes the snapshot matrix as

$$
\mathbf { X } = \Psi \Sigma \mathbf { V } ^ { \top } ,\tag{4}
$$

where $\Psi \in \mathbb { R } ^ { N \times N }$ is an orthogonal matrix whose columns contain the spatial modes, $\breve { \Sigma } \in \mathbb { R } ^ { N \times m }$ is a rectangular diagonal matrix containing the non-negative singular values in descending order, and $\mathbf { V } \in \mathbb { R } ^ { m \times m }$ is an orthogonal matrix whose columns contain the temporal modes. Retaining only the first $r \ll N$ columns of Ψ – here denoted as $\Psi _ { r } \in \mathbb { R } ^ { \check { N } \times { r } }$ – yields the optimal r-rank approximation of the state snapshots, that is

$$
\mathbf { u } _ { k } \approx \Psi _ { r } \mathbf { a } _ { k } \quad { \mathrm { f o r } } k = 1 , \ldots , m ,\tag{5}
$$

where $\mathbf { a } _ { k } \in \mathbb { R } ^ { r }$ for $k = 1 , \ldots , m$ are the time-dependent POD coeficients.

Instead of learning the map from sensor to state values, POD is helpful in reducing the sparse sensing problem to the estimation of the POD coeficients ${ \bf a } _ { k }$ from sensor readings $\mathbf { s } _ { k }$ for $k = 1 , \ldots , m$ , while exploiting POD to recover the original full-order state. In the case of a linear sensing operator C, this is typically solved via the gappy POD approach [22, 49], which considers the least-squares problems

$$
\hat { \mathbf { a } } _ { k } = \arg \operatorname* { m i n } _ { \mathbf { a } _ { k } } \| { \bf C } \Psi _ { r } \mathbf { a } _ { k } - \mathbf { s } _ { k } \| ^ { 2 } ,\tag{6}
$$

for $k = 1 , \ldots , m$ , where ∥·∥ denotes the Euclidean norm. The solution of Equation (6) can be computed via the Moore-Penrose pseudo-inverse of the matrix $\begin{array} { r l } { \mathbf { C } \Psi _ { r } } & { { } \in } \end{array}$ $\mathbb { R } ^ { n \times r }$ , thus yielding the high-dimensional state reconstruction

$$
\hat { \mathbf { u } } _ { k } = \Psi _ { r } \hat { \mathbf { a } } _ { k } = \Psi _ { r } ( \mathbf { C } \Psi _ { r } ) ^ { \dagger } \mathbf { s } _ { k } \quad \mathrm { f o r } \ k = 1 , \ldots , m .
$$

Despite its rigorous formulation, gappy POD is inherently constrained by the low-rank approximation induced by POD, since the reconstructed state is restricted to belong to the linear subspace spanned by the r POD modes $\Psi _ { r } .$ , with reconstruction error lower bounded by the POD accuracy. Moreover, the quality of the reconstruction strongly depends on sensor placement, as poor sensor configurations may lead to ill-conditioned or rankdeficient sensing operators, significantly amplifying measurement noise and numerical errors [11, 36].

To overcome the limitations of linear methods, more flexible nonlinear strategies for state reconstruction have been proposed in the scientific machine learning community [5, 23, 40]. Among these strategies, shallow decoder networks (SDNs) [8, 19, 34, 42] replace the linear reconstruction map with a shallow, fully connected neural network, whose weights and biases are learned by minimizing the reconstruction error over the training data. Diferently from linear approaches, SDNs capture nonlinear dependencies between sparse sensor measurements and the underlying state, leading to improved reconstruction performance over linear methods, particularly in highly sparse sensing regimes [19, 42]. Similarly, the Senseiver [43] reconstructs spatio-temporal fields at arbitrary query locations from sparse observations, leveraging an attention-based architecture inspired by the Perceiver IO [28]. However, similarly to gappy-POD, SDNs and the Senseiver operate on static snapshots, treating each time instant independently and ignoring the temporal correlation in the data, which is a core feature of sequential measurements. As a result, they fail to exploit the information contained in the history of observations, which is often crucial for accurately reconstructing dynamical systems, especially in the presence of limited and noisy data [50].

## B. Shallow recurrent decoders

Rather than reconstructing the full state field ${ \mathbf { u } } _ { k } \in \mathbb { R } ^ { N }$ directly from sensor readings $\mathbf { s } _ { k } \in \mathbb { R } ^ { n }$ for $k = 1 , \ldots , m$ shallow recurrent decoders (SHRED) [51] exploit a recurrent temporal encoder to learn a latent representation from a window of $L$ past sensor measurements. The architecture is therefore defined as

$$
\begin{array} { r } { \hat { \mathbf { u } } _ { k } = \mathcal { D } \big ( \mathcal { E } \big ( \mathbf { s } _ { k - L } , \ldots , \mathbf { s } _ { k } ; \mathbf { W } _ { E } \big ) ; \mathbf { W } _ { D } \big ) , \quad \mathrm { f o r } \ k = 1 , \ldots , m } \end{array}\tag{7}
$$

where $\mathcal { E } : \mathbb { R } ^ { n \times ( L + 1 ) }  \mathbb { R } ^ { z }$ is a recurrent temporal encoder parameterized by $\mathbf { W } _ { E } .$ , which embeds the current value and the past L observations of the n sensors into $\mathrm { ~ a ~ } z -$ dimensional latent variable $\mathbf { z } _ { k } \in \mathbb { R } ^ { z }$ , while $\mathcal { D } : \mathbb { R } ^ { z }  \mathbb { R } ^ { N }$ is a shallow decoder parameterized by $\mathbf { W } _ { D }$ , that spatially reconstructs the spatio-temporal fields of interest. Notably, pre-padding is applied – i.e., $\mathbf { s } _ { k } = \mathbf { 0 }$ for $k \leq L$ – to avoid any burn-in period.

To model temporal dynamics, i.e. to build the encoder $\mathcal { E } ,$ , one may rely on recurrent neural networks such as, $\mathrm { e . g . }$ , long short-term memory (LSTM) [25, 33] or gated recurrent unit (GRU) [10] networks. Alternatively,

T-SHRED [52] replaces the recurrent temporal encoder with a Transformer [48]. This design choice leverages the self-attention mechanism to directly model pairwise interactions among lagged sensor measurements, enabling the network to capture long-range temporal dependen cies more efectively than recurrent architectures. The spatial decoder ${ \mathcal { D } } ,$ instead, maps the latent representation to the full spatial field, reconstructing $\mathbf { u } _ { k }$ from the encoded state $\mathbf { z } _ { k }$ for $k = 1 , \ldots , m$ , and it is commonly modeled as a SDN with a progressively widening architecture, even though multiple alternatives are possible, such as convolutional neural networks.

The SHRED model is trained end-to-end by minimiz ing the reconstruction loss

$$
\sum _ { k \in \Xi _ { \mathrm { t r a i n } } } \left\| \mathbf { u } _ { k } - \mathcal { D } ( \mathcal { E } ( \mathbf { s } _ { k - L } , \dots , \mathbf { s } _ { k } ; \mathbf { W } _ { E } ) ; \mathbf { W } _ { D } ) \right\| ^ { 2 } ,\tag{8}
$$

where $\Xi _ { \mathrm { t r a i n } }$ denotes the time indices in the training set. To enhance computational eficiency for large-scale problems, SHRED architectures usually exploit a compressive training strategy [32, 47] where high-dimensional state snapshots are projected onto a $r \ll N$ dimensional subspace through, e.g., POD, so that the decoder estimates only the r POD coeficients rather than the full state, that is

$$
\begin{array} { r } { \hat { \mathbf { u } } _ { k } = \Psi _ { r } \hat { \mathbf { a } } _ { k } = \Psi _ { r } \mathcal { D } ( \mathcal { E } ( \mathbf { s } _ { k - L } , \ldots , \mathbf { s } _ { k } ) ) . } \end{array}\tag{9}
$$

Importantly, as shown in Figure 1, SHRED can be extended to multiple field reconstructions by concatenating the POD coeficient vectors associated with diferent physical fields into a single reduced-order representation to predict. After projecting the state snapshots, SHRED is trained by minimizing the reconstruction error at the reduced level

$$
\begin{array} { r l } { ~ } & { \displaystyle \sum _ { k \in \Xi _ { \mathrm { t r a i n } } } \left\| \boldsymbol { \Psi } _ { r } ^ { \top } \mathbf { u } _ { k } - \mathcal { D } \big ( \mathcal { E } \big ( \mathbf { s } _ { k - L } , \ldots , \mathbf { \setminus } _ { k } ; \mathbf { W } _ { E } \big ) ; \mathbf { W } _ { D } \big ) \right\| ^ { 2 } , } \end{array}\tag{10}
$$

where $\mathbf { a } _ { k } = \Psi _ { r } ^ { \top } \mathbf { u } _ { k }$ for $k \in \Xi _ { \mathrm { t r a i n } }$ are the reduced state snapshots $- \ \mathrm { i . e . }$ , the POD coeficients – in the training set. Once trained, SHRED generalizes to unseen sensor readings at new time instants, enabling reconstruction of the target field from previously unseen sequences of sensor measurements. Empirically, it has been shown that SHRED outperforms SDNs in reconstruction accuracy [51], as the recurrent temporal encoder is capable of exploiting the temporal correlations in the input sequence, leading to more informative latent representations.

## Transformer shallow recurrent decoders

T-SHRED [52] considers a Transformer [48] as temporal encoder $\mathcal { E }$ within SHRED in order to directly model pairwise interactions among lagged sensor measurements through the self-attention mechanism. Unlike recurrent networks, which progressively aggregate past observations into a hidden representation, Transformers establish direct dependencies between all time steps within the observation window, thus enabling the model to effectively capture long-range temporal dependencies.

Given the input sequence of lagged sensor measurements at time $t _ { k }$

$$
\mathbf { S } _ { k } = \left[ \begin{array} { c c c c } { \rule { 0.5 pt } { 2.5 ex } } & { \rule { 0.5 pt } { 2.5 ex } } & { \cdots } & { \rule { 0.5 pt } { 2.5 ex } } \\ { \rule { 0.5 pt } { 2.5 ex } } & { \mathbf { s } _ { k - L + 1 } } & { \cdots } & { \mathbf { s } _ { k } } \\ { \rule { 0.5 pt } { 2.5 ex } } & { \rule { 0.5 pt } { 2.5 ex } } & { \cdots } & { \rule { 0.5 pt } { 2.5 ex } } \end{array} \right] \in \mathbb { R } ^ { n \times ( L + 1 ) } ,
$$

T-SHRED first considers a projection $\begin{array} { r l } { \mathbf { S } _ { k } ^ { \top } \mathbf { W } _ { E } } & { { } \in } \end{array}$ $\mathbb { R } ^ { ( L + 1 ) \times z }$ of the sensor values onto a z-dimensional embedding through the learnable matrix $\mathbf { W } _ { E } \in \mathbb { R } ^ { n \times z }$ , thus obtaining a sequence of $L + 1$ tokens over time. Moreover, sinusoidal positional encoding [48] is added to preserve the temporal ordering of the sequence, that is the zeroindexed matrix $\mathbf { P E } \in \mathbb { R } ^ { ( L + 1 ) \times \dot { z } }$ having entries defined as

$$
\mathbf { P E } _ { j , \ell } = \left\{ { \begin{array} { l l } { \sin \left( { \frac { j } { 1 0 0 0 0 ^ { \ell / z } } } \right) , } & { { \mathrm { i f ~ } } \ell { \mathrm { ~ i s ~ e v e n } } , } \\ { \cos \left( { \frac { j } { 1 0 0 0 0 ^ { ( \ell - 1 ) / z } } } \right) , } & { { \mathrm { i f ~ } } \ell { \mathrm { ~ i s ~ o d d } } , } \end{array} } \right.
$$

for $j = 0 , \ldots , L$ and $\ell = 0 , \dots , z - 1$ . The resulting matrix $\mathbf { X } _ { k } ^ { ( 1 ) } = \mathbf { S } _ { k } ^ { \top } \mathbf { W } _ { E } + \mathbf { P } \mathbf { E } \in \mathbb { R } ^ { ( L + 1 ) \times z }$ is then sequentially processed through B Transformer blocks (with, typically, $B \geq 2 )$ . A Transformer block is a fundamental building unit of the Transformer architecture that processes input representations to capture temporal dependencies between elements at diferent time instants in the input sequence. The core of each block is the multi-head selfattention (MHSA) mechanism, which computes H attention operations – i.e., the so-called attention heads – in parallel, each parameterized by its own learnable weights, thus allowing each head to capture diferent temporal patterns and dependencies. Specifically, for the hth head in the b-th block, the learnable projection matrices $\mathbf { W } _ { Q } ^ { ( b , h ) } , \mathbf { W } _ { K } ^ { ( b , h ) } , \mathbf { W } _ { V } ^ { ( b , h ) } \in \mathbb { R } ^ { z \times d }$ are used to compute the so-called query, key, and value matrices

$$
\begin{array} { r l } & { \mathbf { Q } _ { k } ^ { \left( b , h \right) } = \mathbf { X } _ { k } ^ { \left( b \right) } \mathbf { W } _ { Q } ^ { \left( b , h \right) } \in \mathbb { R } ^ { \left( L + 1 \right) \times d } , } \\ & { \mathbf { K } _ { k } ^ { \left( b , h \right) } = \mathbf { X } _ { k } ^ { \left( b \right) } \mathbf { W } _ { K } ^ { \left( b , h \right) } \in \mathbb { R } ^ { \left( L + 1 \right) \times d } , } \\ & { \mathbf { V } _ { k } ^ { \left( b , h \right) } = \mathbf { X } _ { k } ^ { \left( b \right) } \mathbf { W } _ { V } ^ { \left( b , h \right) } \in \mathbb { R } ^ { \left( L + 1 \right) \times d } } \end{array}
$$

from the block input $\mathbf { X } _ { k } ^ { ( b ) } \in \mathbb { R } ^ { ( L + 1 ) \times z }$ . The attention operation is then defined as

$$
\mathrm { A } ^ { ( h ) } \left( \mathbf { X } _ { k } ^ { ( b ) } \right) = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { k } ^ { ( b , h ) } \mathbf { K } _ { k } ^ { ( b , h ) } } { \sqrt { d } } \right) \mathbf { V } _ { k } ^ { ( b , h ) } ,
$$

where the softmax function is applied row-wise. Specifically, the product $\mathbf { Q } _ { k } ^ { \left( b , h \right) } \mathbf { K } _ { k } ^ { \left( b , h \right) ^ { \top } } \in \mathbb { R } ^ { \left( L + 1 \right) \times \left( L + 1 \right) }$ collects the pairwise dot-product scores between all query and key vectors, which quantify how strongly each pair of time instants is aligned. The softmax function converts these scores into positive attention weights, with each row summing to one. Finally, the attention weights are multiplied by the value matrix $\mathbf { V } _ { k } ^ { ( b , h ) }$ , so that the output at each time instant is a weighted combination of the value vectors, with input-dependent weights determined by the learned query and key projections. The outputs of the H attention heads are concatenated and linearly projected through a learnable matrix $\mathbf { W } _ { O } \in \mathbb { R } ^ { H d \times z }$ , yielding the MHSA representation

$$
\mathrm { M H S A } \left( \mathbf { X } _ { k } ^ { \left( b \right) } \right) = \left[ \mathbb { A } ^ { \left( 1 \right) } \left( \mathbf { X } _ { k } ^ { \left( b \right) } \right) , \ldots , \mathbf { A } ^ { \left( H \right) } \left( \mathbf { X } _ { k } ^ { \left( b \right) } \right) \right] \mathbf { W } _ { O } .
$$

Finally, the Transformer block applies residual connections and layer normalization $[ 2 ] - \mathrm { i . e . }$ , each input feature is standardized to zero mean and unit variance, and learnable per-feature scale and shift parameters are applied to stabilize training – around both the MHSA module and the (position-wise) multi-layer perceptron (MLP), that is

$$
\begin{array} { r } { \tilde { \mathbf { X } } _ { k } ^ { \left( b \right) } = \mathrm { L a y e r N o r m } \left( \mathbf { X } _ { k } ^ { \left( b \right) } + \mathrm { M H S A } \left( \mathbf { X } _ { k } ^ { \left( b \right) } \right) \right) , } \\ { \mathbf { X } _ { k } ^ { \left( b + 1 \right) } = \mathrm { L a y e r N o r m } \left( \tilde { \mathbf { X } } _ { k } ^ { \left( b \right) } + \mathrm { M L P } \left( \tilde { \mathbf { X } } _ { k } ^ { \left( b \right) } \right) \right) . } \end{array}
$$

The output of the last Transformer block $\mathbf { X } _ { k } ^ { ( B + 1 ) } \in$ $\mathbb { R } ^ { ( L + 1 ) \times z }$ can be regarded as a collection of $L + 1$ latent representations over the observation window. To recover the full state field at time $t _ { k } ,$ , its last row ${ \bf z } _ { k } =$ $\left( \mathbf { X } _ { k } ^ { \left( B + 1 \right) } \right) ^ { \top } \mathbf { e } _ { L + 1 } \in \mathbb { R } ^ { z }$ , where $\mathbf { e } _ { L + 1 } \in \mathbb { R } ^ { L + 1 }$ is the $( L { + } 1 ) .$ th one-hot selection vector, has to be upscaled by the spatial decoder D, which is typically modeled as a SDN or a convolutional neural network.

## III. FAULT-TOLERANT SHALLOW RECURRENT DECODERS

In this section, we introduce diferent fault-tolerant sparse sensing strategies in the context of SHRED. In particular, we consider naive imputation techniques to recover missing sensor values, fault-tolerant recurrent temporal encoders, and we present four diferent Transformer-based masking mechanisms that enable SHRED to deal with irregular and asynchronous timeseries data, such as sensor readings with missing values and diferent sampling rates.

## A. Handling missing sensor measurements

The SHRED framework described in Section II B assumes that all the n input sensor measurements are available at every time step within the observation window $\{ t _ { k - L } , \ldots , t _ { k } \}$ Under this assumption, the encoder receives a fully populated input sequence at each reconstruction step. This hypothesis is reasonable when testing the model in synthetic settings, although it is not fully attainable in real-world systems. In practice, sensors are subject to a broad spectrum of failure modes: power outages, communication dropouts, hardware degradation, scheduled maintenance, calibration windows, and environmental interference. These events produce timeseries that are irregular, asynchronous across sensors, and punctuated by gaps at isolated time instants or contiguous intervals with variable duration. Moreover, when multiple heterogeneous sensors are combined into a single network, their temporal availability patterns are rarely aligned, resulting in a measurement matrix with structured, sensor-dependent patterns of missing entries that reflect the operational history of each individual instrument. For instance, Figure 2 shows the spatial coverage and the sensor data availability of 20 moored buoys collecting in situ sea surface temperature (SST) measurements [38], with availability rates ranging from 97.2% to 53.0% and diferent data availability patterns. In this work, we aim at considering fault-tolerant sparse sensing strategies capable of dealing with missing and asynchronous sensor values, such as defective sensors with heterogeneous sampling frequencies.

SST buoy availability  
(a)  
![](images/4324d370323534271592437d951ca4d6c7dafb0507b3ea5aeef5de168e9c127a.jpg)

(b)  
![](images/b1cf4424ceb9ddec1f908d50a1e39bd4714b2803104f39da55a2329e70bc7653.jpg)  
FIG. 2. Sea surface temperature. (a) Spatial coverage of moored buoys collecting in situ sea surface temperature (SST) measurements [38]. (b) Sensor data availability over time of diferent moored buoys [38], covering the period from January 2000 to December 2025, showing availability rates ranging from 97.2% to 53.0% and diferent data availability patterns. The available sensor readings are depicted by blue dots, while missing values are represented by red bars.

To address incomplete, irregular and asynchronous sensor data, one may consider naive imputation strategies, where the mean, the median or the last-value of the available training data are employed to fill missing entries, with no modification to the SHRED architecture. Specifically, whenever the i-th sensor presents a missing value at time $t _ { k } ,$ the mean imputation replaces the i-th entry $s _ { i , k }$ in the vector $\mathbf { s } _ { k }$ with the average of the available sensor observations over the training set, that is

$$
\hat { s } _ { i , k } = \frac { 1 } { | \Xi _ { i } | } \sum _ { j \in \Xi _ { i } } s _ { i , j } ,
$$

where $\Xi _ { i }$ is the subset of training time steps at which sensor i is observed, with cardinality |Ξ |. Similarly, the median imputation considers the median of the available observations in the training set, that is

$$
{ \hat { s } } _ { i , k } = \mathrm { m e d i a n } \{ s _ { i , j } : j \in \Xi _ { i } \} .
$$

Finally, the last observation carried forward (LOCF) imputation replaces missing entries with the most recent available measurement of the same sensor

$$
\hat { s } _ { i , k } = s _ { i , j ^ { * } } , j ^ { * } = \operatorname* { m a x } \{ j < k : j \in \Xi _ { i } \} ,
$$

while mean imputation is employed whenever no past sensor values are available. Thanks to local temporal continuity, LOCF is particularly efective for short, isolated gaps in smooth signals, while its performance degrades for long gaps or rapidly varying signals. All three imputation strategies can be readily combined with any temporal encoder architecture, since they operate entirely as pre-processing steps.

In addition to naive imputation strategies, it is possible to consider SHRED temporal encoders E specifically designed to handle missing data in the context of multivariate time-series forecasting, such as GRU-D [9] and its attention-augmented variant attnGRU-D [53]. GRU D considers learnable temporal decay mechanisms that account for the time elapsed since the last observation of each sensor. In particular, missing input values are imputed by decaying the last observed value toward a feature-specific empirical mean, while the hidden state is simultaneously decayed toward a default state. These decay mechanisms allow the model to learn how rapidly sensor information becomes outdated and enable the elapsed time since the last observation to directly influence the recurrent dynamics. The resulting input and hidden state are then processed by a standard GRU, allowing GRU-D to jointly exploit the observed values, their missingness patterns, and the temporal structure of the data. Instead, attnGRU-D extends GRU-D by incorporating an attention mechanism to selectively weight the recurrent hidden representations across time, thus allowing the model to identify and emphasize the time steps that are most relevant for the prediction task.

## B. Transformer-based masking strategies

Rather than relying on naive imputation techniques and fault-tolerant recurrent encoders, which address data missingness through pre-processing operations or via modified recurrence equations, in this section we incorporate data availability information directly into the Transformer encoder of T-SHRED. Specifically, we inject the binary data-availability mask at diferent levels of the architecture, in order to condition the internal representation of the temporal encoder both on the observed values and on their corresponding availability patterns. Doing so, the model can learn to discount missing entries instead of treating placeholder values as genuine data, which would otherwise bias the reconstruction.

Let the time grid $\{ t _ { 1 } , \ldots , t _ { m } \}$ be the collection of the sampling instants of all the available sensors. In the case of missing and asynchronous sensor measurements, the observed data at reconstruction step k read as follows

$$
\tilde { \mathbf { S } } _ { k } = \mathbf { S } _ { k } \odot \mathbf { M } _ { k } ,\tag{11}
$$

where $\odot$ denotes element-wise multiplication, $\begin{array} { r l } { \mathbf { S } _ { k } } & { { } = } \end{array}$ $[ \mathbf { s } _ { k - L } , \mathbf { \sigma } _ { \cdot } \mathbf { \cdot } \mathbf { \cdot } , \mathbf { s } _ { k } ] \in \mathbb { R } ^ { n \times ( L + 1 ) }$ is the matrix collecting the sequence of sensor values over the past L steps until time $t _ { k }$ for $k = 1 , \ldots , m$ , with possibly missing entries, while $\mathbf { M } _ { k } \ \in \ \{ 0 , 1 \} ^ { n \times ( L + 1 ) }$ is the mask matrix encoding the availability of each sensor over the lag window. Specifically, the mask is a binary matrix where each entry indicates the data availability of sensor i at time $t _ { j } ,$ that is

$$
( \mathbf { M } _ { k } ) _ { i , j } = \left\{ { 1 \atop 0 } \right. \mathrm { ~ i f ~ s e n s o r ~ } i \mathrm { ~ a t ~ t i m e ~ } t _ { k + j - ( L + 1 ) } \mathrm { ~ i s ~ a v a i l a b l e }
$$

for $i = 1 , \ldots , n$ and $j = 1 , \ldots , L + 1$ The pattern of missing entries jointly depends on the sampling rates and the failure processes of the individual sensors, making the mask ${ { \bf { M } } _ { k } }$ generally heterogeneous across sensors and non-stationary over time.

Diferently from the original SHRED implementation tailored to regular and complete data streams, in this section we propose masked shallow recurrent decoders where we efectively inform SHRED on missing sensor values through Transformer-based masking strategies. Specif ically, we take into account the observed data matrix $\tilde { \mathbf { S } } _ { k }$ as input for $k = 1 , \ldots , m$ , and we inject the corresponding binary mask ${ { \bf { M } } _ { k } }$ at diferent levels of the architecture. In the following, we introduce four diferent masking strategies: channel-wise masking, where the mask is concatenated to the input prior to embedding; hidden-space masking, which gates intermediate hidden representations via a learned projection of the mask; MHSA masking, which enforces hard attention suppression of time steps with no measurements available; and learned MHSA masking, which learns head-specific ad ditive attention biases conditioned on the missingness pattern. The resulting Transformer-based temporal encoders $\mathcal { E } : \mathbb { R } ^ { n \times ( L + 1 ) } \times \{ 0 , 1 \} ^ { n \times ( L + 1 ) }  \mathbb { R } ^ { z }$ are thus able to return a latent representation $\mathbf { z } _ { k }$ for $k = 1 , \ldots , m$ aware of the sensor availability, and allow SHRED to generalize to irregular and asynchronous data streams.

Channel-wise masking The first strategy to make the Transformer encoder aware of missing values is to concatenate the binary mask ${ { \bf { M } } _ { k } }$ directly to the observed data $\tilde { \mathbf { S } } _ { k }$ as an additional set of feature channels prior to the input embedding layer, that is

$$
\left[ \tilde { \mathbf { S } } _ { k } \right] \in \mathbb { R } ^ { 2 n \times ( L + 1 ) } .
$$

This efectively doubles the number of input channels from n to $2 n .$ , where the first n rows correspond to the observed sensor values, while the remaining n rows encode the corresponding binary availability indicators. The augmented input is then processed as in the standard T-SHRED framework. Notably, no modification is made to the attention mechanism or any other architectural component, as the mask information is injected exclusively at the input level. As a result, all subsequent layers must implicitly learn how to interpret the additional input information and how to propagate it through the encoder. The channel-wise strategy is therefore simple and architecture-agnostic, but it does not provide an explicit mechanism to prevent missing entries from influencing the attention scores, so that the model must learn to discount them implicitly during training.

Hidden-space masking Instead of providing the binary data-availability mask as additional input, the second masking strategy operates at the level of the hidden representation. We recall that the masked sensor measurements $\tilde { \mathbf { S } } _ { k }$ are embedded through the projection matrix $\mathbf { W } _ { \mathrm { E } } \in \mathbb { R } ^ { n \times z }$ , while positional encoding is added, thus yielding the matrix $\mathbf { X } _ { k } ^ { ( 1 ) } \in \mathbb { R } ^ { ( L + 1 ) \times z }$ At the beginning of each Transformer block, the hidden representation $\mathbf { X } _ { k } ^ { ( b ) } \in \mathbb { R } ^ { ( L + 1 ) \times z }$ for $b = 1 , \dots , B$ is modulated element-wise by a continuous gating signal, that is

$$
\mathbf { X } _ { k } ^ { \left( b \right) } \gets \mathbf { X } _ { k } ^ { \left( b \right) } \odot \sigma \big ( \mathbf { M } _ { k } ^ { \top } \mathbf { W } _ { M } \big ) ,
$$

where $\mathbf { W } _ { M } \ \in \ \mathbb { R } ^ { n \times z }$ is a learnable projection matrix shared across diferent blocks, while $\sigma ( \cdot )$ is the sigmoid function. The gated input is then fed to the Transformer block.

This masking mechanism explicitly amplifies or attenuates each hidden dimension at each time step according to which sensors are available, so that the model can learn to down-weight the hidden features that mainly depend on missing measurements. Moreover, since the mask directly influences the intermediate representations at every Transformer block, it ensures that the data availability information is propagated throughout the depth of the network rather than being confined to the input embedding stage, thus efectively controlling the information flow as it propagates through the model. Note that, since ${ \bf W } _ { M }$ is learned jointly with all the other model parameters, the resulting gating signal is not a fixed binary mask, but a continuous, data-dependent weighting. This allows the model to adaptively determine how much each hidden dimension should be suppressed or preserved at every time step, depending on the observed missingness pattern and its interaction with the hidden state.

MHSA Masking The third masking strategy operates directly within the scaled dot-product attention mechanism, analogously to the masking mechanism considered in the standard Transformer [48], preventing missing values from contributing to the attention computation. Rather than modifying the input or the hidden representations, this approach introduces an additive bias to the attention scores that explicitly excludes time steps with no sensor values available from the softmax computation. Specifically, we consider the modified self-attention operation

$$
\mathrm { A } ^ { ( h ) } ( \mathbf { X } _ { k } ^ { ( b ) } ) = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { k } ^ { ( b , h ) } \mathbf { K } _ { k } ^ { ( b , h ) } } { \sqrt { d } } + \mathbf { B } _ { k } \right) \mathbf { V } _ { k } ^ { ( b , h ) } ,
$$

where the attention bias matrix at the k-th reconstruction step $\mathbf { B } _ { k } \in \mathbb { R } ^ { ( L + 1 ) \times ( L + 1 ) }$ is defined as

$$
( \mathbf { B } _ { k } ) _ { l , j } = { \left\{ \begin{array} { l l } { - \infty } & { { \mathrm { i f ~ } } \sum _ { i = 1 } ^ { n } ( \mathbf { M } _ { k } ) _ { i , j } = 0 } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{12}
$$

for $l , j = 1 , \ldots , L + 1$ . Note that the attention bias matrix is shared across Transformer blocks and attention heads. After softmax normalization, this ensures that the attention weights assigned to time steps with no sensor values available $\begin{array} { r } { ( \mathrm { i . e . , ~ \sum _ { \it i = 1 } ^ { { n } } ( M _ { k } ) _ { \it i , j } \it ~ = ~ } 0 ) } \end{array}$ are exactly zero, independently of the learned projections. Note that one may additionally require that $( \mathbf { B } _ { k } ) _ { l , j } = - \infty$ for all $l < j ,$ thereby recovering the autoregressive structure of standard Transformer models [48].

Importantly, since the mask is derived directly from the binary availability pattern and applied uniformly across all attention heads and Transformer blocks, this strategy imposes a hard structural constraint on information flow within the model. Note that, while time steps with no sensor values available are completely excluded from the attention computation, time instants with partial missingness are not explicitly modeled in this formulation. The MHSA masking thus yields a strict and interpretable mechanism for filtering missing information, but it does not capture finer-grained patterns of partial observability within a timestep.

Learned MHSA Masking The fourth masking strategy extends the MHSA masking by replacing the fixed, rule-based mechanism for filtering missing information with a learnable data-dependent attention bias. Specifi cally, we consider the attention operation

$$
\mathrm { A } ^ { ( h ) } ( \mathbf { X } _ { k } ^ { ( b ) } ) = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { k } ^ { ( b , h ) } \mathbf { K } _ { k } ^ { ( b , h ) } } { \sqrt { d } } + \mathbf { B } _ { k } ^ { ( b , h ) } \right) \mathbf { V } _ { k } ^ { ( b , h ) } ,
$$

where the attention bias $\mathbf { B } _ { k } ^ { ( b , h ) } \in \mathbb { R } ^ { ( L + 1 ) \times ( L + 1 ) }$ for the b-th Transformer block and the h-th attention head is given by a lightweight two-layer MLP with ReLU activation function. In particular, the block-specific feedforward neural network $\varphi ^ { ( b ) } : \mathbb { R } ^ { ( L + 1 ) \times z }  \mathbb { R } ^ { ( L + 1 ) \times ( L + 1 ) \times H }$ returns, after a suitable reshape, the attention biases $\mathbf { B } _ { k } ^ { ( b , h ) } \in \mathbb { R } ^ { ( L + 1 ) \times ( L + 1 ) }$ for $h = 1 , \ldots , H$ , that is

$$
\left[ { \mathbf { B } _ { k } } ^ { ( b , 1 ) } , \ldots , { \mathbf { B } _ { k } } ^ { ( b , H ) } \right] = \varphi ^ { ( b ) } ( \mathbf { M } _ { k } ^ { \top } \mathbf { W } _ { M } ) ,
$$

where $\mathbf { W } _ { M } \in \mathbb { R } ^ { n \times z }$ is a learnable embedding matrix, shared across diferent Transformer blocks.

In contrast to the previous masking strategies, the learned MHSA masking allows each Transformer block and attention head to adjust the attention between pairs of time instants according to the data-availability pattern. This gives the model the flexibility to handle partially observed time steps, and lets diferent heads spe cialize in diferent missingness patterns, at the price of a soft rather than guaranteed exclusion of missing entries from the attention computation. It is important to note that the diferent masking mechanisms proposed are not mutually exclusive, and they can be readily combined in hybrid configurations, thus allowing the model to jointly exploit explicit availability information and more flexible, layer-wise adaptations to missing-data patterns.

## C. Multi-step decoding training strategy

This section introduces the multi-step decoding training strategy designed to enhance latent trajectories and SHRED predictions. The original SHRED implementation, as well as its extensions such as T-SHRED, embeds a sequence of lagged sensor measurements $\mathbf { s } _ { k - L } , \ldots , \mathbf { s } _ { k }$ into a single latent representation $\mathbf { z } _ { k } ~ \in ~ \mathbb { R } ^ { z }$ , which is spatially decoded into the full state variable. However, temporal encoders internally produce a sequence of zdimensional latent representations $\widetilde { \mathbf { z } } _ { k - L } , \ldots , \widetilde { \mathbf { z } } _ { k - 1 } , \mathbf { z } _ { k } \in$ $\mathbb { R } ^ { z }$ over time. Note that, while recurrent temporal encoders such as, e.g., LSTM and GRU recursively update the hidden state over time, Transformer-based encoders aggregate the latent representations in the rows of the output matrix $\mathbf { X } _ { k } ^ { ( B + 1 ) } \in \mathbb { R } ^ { ( L + 1 ) \times z }$ Discarding all the hidden states but the last one may be a limiting factor when dealing with heterogeneous missing data, as temporal context accumulated across the lookback window may compensate for the missing information. Instead, supervising the decoding of intermediate latent states provides a training signal at every time step of the lookback window, which encourages the encoder to retain state information over time and makes the latent trajectory less sensitive to missing measurements at individual steps, which may be helpful in improving reconstruction accuracy and forecasting capabilities.

To mitigate this limitation, we introduce a multistep decoding training strategy, inspired by the auxiliary classifier mechanism introduced in GoogLeNet [45], where auxiliary branches are attached at intermediate depths of the network and produce additional classifica tion predictions during training. Specifically, while training SHRED, we apply the same spatial decoder D to the latent variables $\widetilde { \mathbf { z } } _ { k - L } , \ldots , \widetilde { \mathbf { z } } _ { k - 1 } , \widetilde { \mathbf { z } } _ { k } = \mathbf { z } _ { k }$ , thus obtaining multiple lagged state reconstructions. The encoder and decoder parameters $\mathbf { W } _ { E }$ and $\mathbf { W } _ { D }$ can then be determined by minimizing the training loss

$$
\sum _ { k \in \Xi _ { \mathrm { t r a i n } } } \sum _ { j = 0 } ^ { L } w _ { j } \| \mathbf { u } _ { k - j } - \mathcal { D } ( \tilde { \mathbf { z } } _ { k - j } ) \| ^ { 2 } ,
$$

where the coeficients $w _ { j } \in ( 0 , 1 )$ for $j = 0 , \ldots , L$ balance the contribution of each prediction horizon. In the following, we consider weights in the form $w _ { j } = \alpha ^ { j }$ for $j = 0 , \ldots , \tilde { L }$ and $w _ { j } ~ = ~ 0$ for $j = \tilde { L } + 1 , . . . , L$ , with $0 < \tilde { L } < L$ and $\alpha \in ( 0 , 1 )$ , so as to prioritize accuracy on more recent time instants.

Whenever a compressive training strategy is employed through, e.g., a POD-based projection of the full state fields in the training set, the multi-step decoding loss reads as follows

$$
\sum _ { k \in \Xi _ { \mathrm { t r a i n } } } \sum _ { j = 0 } ^ { L } w _ { j } \| \Psi _ { r } ^ { \top } \mathbf { u } _ { k - j } - \mathcal { D } ( \tilde { \mathbf { z } } _ { k - j } ) \| ^ { 2 } ,
$$

where the spatial decoder aims at reconstructing the lagged POD coeficients starting from the corresponding latent representations, rather than predicting the full state.

It is worth noting that, after training, only the last latent representation $\mathbf { z } _ { k }$ is decoded to obtain the state reconstruction $\hat { \mathbf { u } } _ { k } ,$ thus recovering the standard SHRED pipeline at inference time with no additional computational overhead.

## IV. NUMERICAL RESULTS

This section presents the experimental validation of the proposed fault-tolerant shallow recurrent decoder strategies for handling heterogeneous missing sensor data in four diferent scenarios. In particular, we consider a first synthetic benchmark dealing with shallow water equations solution data, in which sensor malfunctions are artificially introduced. Moreover, we take into account three real-world sparse sensing problems in the context of hydrodynamic and ocean modeling with missing and asynchronous sensor measurements.

Throughout the test cases, we split the temporal snapshots into training, validation and test set, so that we test the performance of fault-tolerant sensing strategies on sensor measurements unseen during training. The test set includes both data within the temporal range covered by the training data, to assess interpolation performance, and beyond the training period, to evaluate extrapolation to future time instants. Moreover, we quantitatively assess the quality of the reconstruction on test snapshots through two standard metrics: the root mean squared error (RMSE)

$$
\sqrt { \frac { 1 } { \vert \Xi _ { \mathrm { t e s t } } \vert } \sum _ { k \in \Xi _ { \mathrm { t e s t } } } \Vert \mathbf { u } _ { k } - \hat { \mathbf { u } } _ { k } \Vert ^ { 2 } } ,
$$

and the range-normalized mean absolute error (NMAE)

$$
\frac { 1 } { \lvert \Xi _ { \mathrm { t e s t } } \rvert } \sum _ { k \in \Xi _ { \mathrm { t e s t } } } \frac { \lvert \lvert { \bf u } _ { k } - \hat { \bf u } _ { k } \rvert \rvert _ { 1 } } { { \bf u } _ { \mathrm { m a x } } - { \bf u } _ { \mathrm { m i n } } } ,
$$

where $\lVert \cdot \rVert _ { 1 }$ denotes the $L ^ { 1 }$ (Manhattan) norm, while $\mathbf { u } _ { \mathrm { m a x } }$ and $\mathbf { u } _ { \mathrm { m i n } }$ are the maximum and minimum state components over the test set, respectively. Note that, when reconstructing multiple physical fields, the error metrics are computed separately for each field and then averaged to obtain the final score. Finally, we consider the same training configuration for all the models and test cases, which takes into account 200 epochs, batch size equal to 32, AdamW optimizer with cosine annealing learning rate scheduler, and multi-step decoding training strategy with $\alpha = 0 . 8$ and $\tilde { L } = 1 6$

## A. Shallow water equations

The first test case serves as a numerical benchmark to test the diferent fault-tolerant strategies proposed in Section III. Specifically, we focus on high-dimensional temporal data provided by The Well dataset [39] corresponding to the solution of shallow water equations on a sphere with earth-like topography and daily/annual periodic external forces. The dataset collects 1008 observations over a year of three high-dimensional fields $( N = 1 3 1 0 7 2 )$ , namely the height deviation of pressure surface from its mean height, and the velocity components with respect to the polar and azimuthal angles. We consider 10 in situ measurement stations placed randomly in the domain, where the sensor readings are directly extracted from the numerical simulation as pointwise evaluations. To simulate diferent sensor behaviors, we consider 2 always-available sensors, 4 sensors with missing values at random time instants, and 4 sensors with gaps over contiguous intervals with variable duration, as visible in Figure 3.

![](images/c1c39bd60406767ba02a256fd92007ff62ad730759fcf339aed9c0e999d5e350.jpg)

![](images/e5b496189a30123e50311021287b7b7e297892c3e1326def84d65d2f16d78887.jpg)

FIG. 3. Shallow water equations. (a) Spatial coverage of simulated water level stations. (b) Sensor data availability over one year of diferent water level stations, showing availability rates ranging from 100% to 70.1% and diferent data availability patterns. The available sensor readings are depicted by blue dots, while missing values are represented by red bars.
<table><tr><td colspan="2"></td><td colspan="3">RMSE</td><td colspan="3">NMAE</td><td>Improvement vs. Baseline</td></tr><tr><td>Type</td><td>Configuration</td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td><td>Test NMAE (%)</td></tr><tr><td>M</td><td>Channel-wise masking</td><td>17.29</td><td>22.00</td><td>22.39</td><td>0.0177</td><td>0.0263</td><td>0.0251</td><td>45.51%</td></tr><tr><td>M</td><td>Channel-wise and MHSA masking</td><td>18.76</td><td>22.76</td><td>25.67</td><td>0.0192</td><td>0.0262</td><td>0.0286</td><td>37.85%</td></tr><tr><td>M</td><td>MHSA masking</td><td>19.55</td><td>23.14</td><td>25.28</td><td>0.0197</td><td>0.0265</td><td>0.0287</td><td>37.58%</td></tr><tr><td>M</td><td>Learned MHSA masking</td><td>18.66</td><td>23.42</td><td>25.45</td><td>0.0193</td><td>0.0271</td><td>0.0289</td><td>37.27%</td></tr><tr><td>M</td><td>Input masking</td><td>18.51</td><td>24.38</td><td>25.18</td><td>0.0190</td><td>0.0281</td><td>0.0296</td><td>35.60%</td></tr><tr><td>M</td><td>Hidden-space masking</td><td>19.28</td><td>25.32</td><td>24.80</td><td>0.0197</td><td>0.0292</td><td>0.0296</td><td>35.57%</td></tr><tr><td>M</td><td>Channel-wise and learned MHSA masking</td><td>19.92</td><td>26.41</td><td>26.42</td><td>0.0203</td><td>0.0308</td><td>0.0310</td><td>32.57%</td></tr><tr><td>G</td><td>GRUD</td><td>23.85</td><td>31.74</td><td>27.41</td><td>0.0243</td><td>0.0349</td><td>0.0314</td><td>31.73%</td></tr><tr><td>G</td><td>Attn-GRUD</td><td>26.88</td><td>36.18</td><td>29.75</td><td>0.0273</td><td>0.0379</td><td>0.0330</td><td>28.25%</td></tr><tr><td>I</td><td>Mean imputation + T-SHRED</td><td>24.39</td><td>29.25</td><td>32.54</td><td>0.0248</td><td>0.0338</td><td>0.0377</td><td>18.06%</td></tr><tr><td>I</td><td>LOCF + T-SHRED</td><td>26.64</td><td>35.38</td><td>34.21</td><td>0.0270</td><td>0.0386</td><td>0.0390</td><td>15.29%</td></tr><tr><td>I</td><td>Median imputation + T-SHRED</td><td>26.35</td><td>35.18</td><td>35.54</td><td>0.0266</td><td>0.0397</td><td>0.0398</td><td>13.52%</td></tr><tr><td>B</td><td>Always-available sensors + SHRED</td><td>37.41</td><td>43.32</td><td>39.72</td><td>0.0360</td><td>0.0463</td><td>0.0437</td><td>4.91%</td></tr><tr><td>B</td><td>Always-available sensors + T-SHRED</td><td>42.55</td><td>42.69</td><td>42.03</td><td>0.0419</td><td>0.0466</td><td>0.0460</td><td>0.00%</td></tr><tr><td></td><td>B: Baselines</td><td>I: Imputation</td><td></td><td>G: GRU-D</td><td></td><td>M: Masking</td><td></td><td></td></tr></table>

FIG. 4. Shallow water equations. Root mean squared errors and normalized mean absolute errors on training, validation and test data achieved by the diferent competing models.

We split the data into training, validation and test set with ratio 80 : 10 : 10, so that the test set includes both months within the temporal range covered by the training data and months beyond the training period. We then employ POD to reduce the dimensions of the fields to reconstruct: 100 POD modes are enough to compress each field, namely the surface elevation and the velocity components, with a residual energy below 0.1%. Moreover, we consider a lag equal to 100 time steps to capture the history of the system and the temporal pattern of the sensor readings.

To solve the reconstruction problem, we consider the diferent fault-tolerant sensing strategies introduced in Section III, such as imputation-based and GRU-D-based strategies, as well as diferent masking strategies in the context of T-SHRED. Moreover, we also take into account SHRED with LSTM-based recurrent temporal encoder and T-SHRED trained exclusively on the 2 alwaysavailable sensor stations. To fairly evaluate the performance of the competing models, we consider the same training setup and architectures for all the competing models. Specifically, we employ recurrent temporal encoders with a single hidden layer having 32 neurons, Transformer-based encoders with 1 block, 4 attention heads, embedding dimension equal to 32, and a MLP with a single hidden layer of size 48. The spatial decoder is instead modeled via a SDN with a single hidden layer having 64 neurons, while dropout with rate 0.1 is applied to prevent overfitting.

![](images/48c5830cf90bbe6c4a7cce8a867eb0ff9173150b9cb43c778b5ea76dec5a5fb5.jpg)  
FIG. 5. Shallow water equations. Ground truth, reconstruction provided by masked SHRED with channel-wise masking and reconstruction error on surface elevation (first column), polar velocity (second column) and azimuthal velocity (third column) at four diferent time instants in the test set.

Figure 4 quantitatively assesses the performance of the diferent baselines and fault-tolerant sensing strategies considered through the RMSE and the NMAE on training, validation and test data, with Transformer-based masking strategies providing the most accurate results. In particular, the best-performing configuration employs channel-wise masking to obtain a NMAE on test data equal to 2.51%, with a relative improvement compared to the T-SHRED baseline fitted on always-available sensors only equal to 45.51%. Imputation- and GRU-Dbased methods, instead, yield higher errors, although they still outperform the baselines. Notably, hyperparameter tuning and repeated training runs with diferent random initializations may lead to slight changes in the ranking of the top-performing models. Nevertheless, masked SHRED strategies systematically outperform imputation- and GRU-D-based methods, although no single masking configuration consistently dominates across all runs. Figure 5, instead, qualitatively compares the reconstructed fields provided by the best-performing model with the ground truth at four diferent time instants in the test set, showing the accuracy of the proposed strategy to predict the spatio-temporal patterns of the three high-dimensional fields of interest, despite the limited and irregular input information.

## B. Southern North Sea hydrodynamics

This section delves into a first real-world application in the context of hydrodynamic modeling. Specifically, we aim at reconstructing the water level and velocity in the Southern North Sea relying solely on sensor readings provided by sparse water level stations within the region of interest. This challenging setting is characterized by spatial heterogeneity, complex coastal geometries, and strong temporal variability driven by tides, wind forcing, and boundary conditions, making it a suitable test case for assessing the robustness of the proposed strategies under heterogeneous sensor coverage. The high-dimensional surface elevation and current velocity fields (N = 25599) from January 2022 to October 2023 are generated by numerically solving the twodimensional shallow water equations with a 30-minute resolution through the MIKE 21 Flow Model FM [17], a two-dimensional, depth-averaged finite element model widely used in coastal and shelf-sea applications. The model employs observational water levels and current velocities data as boundary conditions, as well as wind ve locities and air pressure data within the domain for calibration. Sensor measurements, instead, are provided by 16 in situ measurement stations in the Southern North Sea [17, 20]. The spatial coverage of water level stations is available in panel (a) of Figure 6, while the corresponding data streams are available in panel (b) of the same figure. In particular, some stations exhibit intermittent gaps due to communication failures or acquisition issues, while others experience extended inactivity caused by hardware malfunctions or maintenance. Moreover, sensor readings are afected by missing values and uncertainties typical of operational ocean observing systems, including measurement noise, irregular sampling, communication delays, sensor outages, calibration errors, and local disturbances. It is worth noting that the sensor measurements do not exactly match the simulated field values at the corresponding station locations, although they exhibit a strong correlation [17].

![](images/f8db1d719a06576e9ab0de087c9ebdd99ac0061b8c3f177d958cb3c62c404267.jpg)

FIG. 6. Southern North Sea hydrodynamics. (a) Spatial coverage of in situ water level stations in the Southern North Sea. (b) Sensor data availability over time of diferent in situ water level stations in the Southern North Sea, covering the period from January 2022 to October 2023, showing availability rates ranging from 100% to 28.2% and diferent data availability patterns. The available sensor readings are depicted by blue dots, while missing values are represented by red bars.
<table><tr><td colspan="2"></td><td colspan="3">RMSE</td><td colspan="3">NMAE</td><td rowspan="2">Improvement vs. Baseline</td></tr><tr><td>Type Configuration</td><td></td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>M</td><td>Channel-wise and MHSA masking</td><td>0.0895</td><td>0.0969</td><td>0.0783</td><td>0.0089</td><td>0.0102 0.0086</td><td></td><td>15.59%</td></tr><tr><td>M</td><td>Learned MHSA masking</td><td>0.0883</td><td>0.1018</td><td>0.0777</td><td>0.0088</td><td>0.0105</td><td>0.0086</td><td>15.51%</td></tr><tr><td>M</td><td>Channel-wise and learned MHSA masking</td><td>0.0899</td><td>0.1003</td><td>0.0786</td><td>0.0089</td><td>0.0105</td><td>0.0087</td><td>14.36%</td></tr><tr><td>M</td><td>Hidden-space masking</td><td>0.0937</td><td>0.1012</td><td>0.0796</td><td>0.0093</td><td>0.0105</td><td>0.0088</td><td>14.10%</td></tr><tr><td>G</td><td>GRUD</td><td>0.0944</td><td>0.1019</td><td>0.0819</td><td>0.0091</td><td>0.0104</td><td>0.0088</td><td>13.98%</td></tr><tr><td>I</td><td>LOCF + T-SHRED</td><td>0.0892</td><td>0.0982</td><td>0.0788</td><td>0.0090</td><td>0.0104</td><td>0.0088</td><td>13.87%</td></tr><tr><td>M</td><td>Channel-wise masking</td><td>0.0934</td><td>0.1004</td><td>0.0809</td><td>0.0093</td><td>0.0106</td><td>0.0088</td><td>13.32%</td></tr><tr><td>M I</td><td>MHSA masking</td><td>0.0927</td><td>0.0964</td><td>0.0804</td><td></td><td>0.0093 0.0101</td><td>0.0089</td><td>12.57%</td></tr><tr><td>G</td><td>Median imputation + T-SHRED</td><td>0.0947</td><td>0.0993</td><td>0.0827</td><td>0.0094</td><td>0.0105</td><td>0.0090</td><td>12.01%</td></tr><tr><td>M</td><td>Attn-GRUD</td><td>0.0962</td><td>0.1017</td><td>0.0830 0.0806</td><td>0.0095</td><td>0.0107</td><td>0.0090</td><td>12.00%</td></tr><tr><td>I</td><td>Input masking</td><td>0.0945</td><td>0.0996 0.0996</td><td>0.0853</td><td>0.0095 0.0099</td><td>0.0106</td><td>0.0091</td><td>10.91%</td></tr><tr><td>B</td><td>Mean imputation + T-SHRED Always-available sensors + T-SHRED</td><td>0.0980 0.1112</td><td>0.1188</td><td>0.0918</td><td>0.0102</td><td>0.0107 0.0110</td><td>0.0094 0.0096</td><td>7.33%</td></tr><tr><td>B</td><td>Always-available sensors + SHRED</td><td>0.1160</td><td>0.1264</td><td>0.0950</td><td>0.0110</td><td>0.0121</td><td>0.0102</td><td>5.75%</td></tr><tr><td></td><td></td><td>I: Imputation</td><td></td><td>G: GRU-D</td><td></td><td>■</td><td>M: Masking</td><td>0.00%</td></tr><tr><td colspan="5">B: Baselines</td><td></td><td></td><td></td></tr></table>

FIG. 7. Southern North Sea hydrodynamics. Root mean squared errors and normalized mean absolute errors on training, validation and test data achieved by the diferent competing models.

We here investigate the performance of the diferent competing methods in the ultra-sparse sensing regime where we challenge the models to reconstruct the water level and velocity in the Southern North Sea over time relying solely on 4 in situ measurement stations, namely

ZeebruggeWielingendok, K141, NWA, Dover, showing temporal coverage between 100% and 48.6%, and coeficient of determination $R ^ { 2 }$ between sensor and pointwise simulation values between 0.934 and 0.911. After splitting the available data into training, validation and test with ratio 80 : 10 : 10, so that the test set includes both months within and beyond the temporal range considered during training, we compress the surface elevation and velocity components through POD with, respectively, reduced dimensions equal to 12, 40 and 68 POD modes, ensuring that the residual energy remains below 0.2%.

![](images/884b4577ef464dc1b5e6e1bd3ae223f00ecda3d0658bf1d03551ab46c468f006.jpg)

FIG. 8. Southern North Sea hydrodynamics. Ground truth, reconstruction provided by masked SHRED with channel-wise and MHSA masking and reconstruction error on surface elevation (first column), horizontal velocity (second column) and vertical velocity (third column) at four diferent time instants in the test set.  
![](images/041e9edd2e1d025bfad1528b91cbed289cbdb6c7b45dfadfe50c3e5067e81998.jpg)

FIG. 9. Southern North Sea hydrodynamics. Cumulative root mean squared error (first row) and normalized mean absolute error (second row) for the three output fields over the last two months in the test set, predicted by masked SHRED with channel-wise and MHSA masking and by the T-SHRED baseline trained exclusively on always-available sensors.  
![](images/de2c41489dba8482fc1437e1a6ea94334e6c004d203f92feb964865e876e7487.jpg)  
FIG. 10. Southern North Sea hydrodynamics. Mean and one-standard-deviation intervals of the first four components of the latent variable $\tilde { \mathbf { z } } _ { k }$ over time $k \in \Xi _ { \mathrm { t e s t } }$ of masked SHRED with channel-wise and MHSA masking, evaluated using the diferent lagged sensor measurements in the test set and the last- and multi-step decoding training strategies.

Moreover, we consider sensor values lagged by 24 time steps, which corresponds to a half-day delay.

In addition to GRU-D-based models, classical imputation methods and masking strategies in the context of T-SHRED, we benchmark the performance against SHRED with LSTM as recurrent temporal encoder and T-SHRED trained exclusively on the always-available sensor station, i.e. ZeebruggeWielingendok. To fairly compare the diferent fault-tolerant strategies, we consider the same training setup and architecture for all the methods. In particular, we take into account recurrent temporal encoders with 3 layers and hidden size 32, while Transformer-based encoders consider 3 blocks, 4 attention heads, embedding dimension 32, and a MLP with a single hidden layer with 48 neurons. The spatial decoder is instead modeled via a SDN with two hidden layers having 64 and 96 neurons, respectively.

Figure 7 displays the RMSE and the NMAE on training, validation and test data of the diferent baselines and fault-tolerant sensing strategies considered. Overall, the Transformer-based masking strategies proposed in Section III B are able to provide the most accurate results on test data, i.e. when predicting the solution on time instants unseen during training, both within the training range and when extrapolating in time. The bestperforming configuration, that is masked SHRED with channel-wise and MHSA masking, achieves a NMAE on test data equal to 0.86%, with a relative improvement of 15.59% with respect to the SHRED baseline fitted on the always-available water level station only. Importantly, the limited margin of improvement in the real-world setting is consistent with the higher intrinsic dificulty of the task, the limited number of retained sensors, and the noise inherent in operational measurements, all of which compress the achievable performance range across competing methods. Figure 8 displays, instead, the ground truth and the reconstructed fields at four diferent time instants in the test set provided by the best-performing fault-tolerant strategy when extrapolating in time with respect to the training data. Masked SHRED is able to provide accurate reconstructions of the diferent fields of interest when dealing with sensor values unseen during training. Moreover, by leveraging all the water level stations available, masked SHRED accumulates substantially less error in long-horizon prediction regimes, i.e. over the last two months in the test set, with respect to the T-SHRED baseline, as visible in Figure 9.

The multi-step decoding training strategy proposed in Section III C provides comparable results compared to the classical last-step decoding objective when applied to the diferent sensing strategies. Moreover, it yields smoother and temporally consistent latent trajectories $\tilde { \mathbf { z } } _ { k }$ for $k \in \Xi _ { \mathrm { t e s t } }$ compared to the ones produced by last-step training, as demonstrated by Figure 10 through the mean and standard deviation of the first four components of $\tilde { \mathbf { z } } _ { k }$ computed across diferent lagged sensor values in the test set, with reference to masked SHRED with channel-wise and MHSA masking, thus potentially making latent forecasting more tractable and stable under distributional shift. To quantitatively support this finding, Figure 10 also reports the mean and the standard deviation of the total variation (TV) $\| \tilde { \mathbf { z } } _ { k + 1 } - \tilde { \mathbf { z } } _ { k } \|$ for $k \in \Xi _ { \mathrm { t e s t } } \ \backslash$ {m}. In particular, we observe a 49.6% reduction of the TV between the two training strategies, and a 25.1% reduction when considering the TV computed on the last-step embeddings $\mathbf { z } _ { k }$ for $k \in \Xi _ { \mathrm { t e s t } }$ only.

## C. Øresund Strait hydrodynamics

This section presents another real-world application where, similarly to the test case detailed in Section IV B, we aim at reconstructing the water level and velocity in the Øresund Strait, an area of approximately $2 0 0 0 \mathrm { k m ^ { 2 } }$ between Denmark and Sweden, while relying solely on observational data provided by 12 measurement stations [16, 21]. The high-dimensional target fields $( N = 9 9 8 0 )$ from January 2020 to October 2023, that are the surface elevation and the current velocity solving the two-dimensional shallow water equations with a 30-minute resolution, are provided by the MIKE 21 Flow Model FM [16]. The panel (a) of Figure 11 shows the spatial coverage of the available in situ measurements stations, while panel (b) of the same figure displays the data availability patterns of the diferent stations. For instance, Koege station does not provide measurements for about one month at the beginning of 2022, while the six stations with the lowest temporal coverage are characterized by a halved sampling rate with respect to those with higher temporal coverage. Notably, since no water level station provides a temporal coverage equal to 100%, it is not possible to consider naive SHRED baselines fitted on always-available sensors only. Similarly to the previous test case, the sensor values are not directly ex tracted from the simulated high-dimensional water level data, although they exhibit a strong correlation [16].

After splitting the available data into training, validation and test with ratio $7 0 : 1 5 : 1 5$ we employ POD to compress the surface elevation and velocity components with reduced dimensions equal to, respectively, 15, 100 and 100, so that the residual energy remains below 0.1%. Moreover, we consider a lag equal to 24, which corresponds to a half-day delay. As far as the diferent competing models are concerned, we consider recurrent temporal encoders with 3 hidden layers and hidden size equal to 32, while Transformer-based encoders consider 3 blocks, 4 attention heads, embedding dimension 32, and a MLP with a single hidden layer of dimension 48. The spatial decoder is instead modeled via a SDN with two hidden layers having 64 and 96 neurons, where dropout with rate 0.1 is employed to prevent overfitting.

The error metrics, namely RMSE and NMAE, on training, validation and test set of the diferent competing methods are available in Figure 12. In particular, it is possible to assess the superiority of Transformerbased masking strategies with respect to imputation- and

![](images/12544e0462d158d6e039e60961e9cef1f708008b2e387b268e47277d35459257.jpg)

FIG. 11. Øresund Strait hydrodynamics. (a) spatial coverage of in situ water level stations in the Øresund Strait [16]. (b) Sensor data availability over time of diferent in situ water level stations in the Øresund Strait [16], covering the period from January 2020 to October 2023, showing availability rates ranging from 99.2% to 41.7% and diferent data availability patterns. The available sensor readings are depicted by blue dots, while missing values are represented by red bars.
<table><tr><td colspan="2"></td><td colspan="3">RMSE</td><td colspan="3">NMAE</td><td rowspan="2">Improvement vs. Worst Test NMAE (%)</td></tr><tr><td>Type</td><td>Configuration</td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>M</td><td>Learned MHSA masking</td><td>0.0410</td><td>0.0508</td><td>0.0511</td><td>0.0090 0.0143</td><td>0.0119</td><td></td><td>7.21%</td></tr><tr><td>M</td><td>Hidden-space masking</td><td>0.0413</td><td>0.0527</td><td>0.0522</td><td>0.0091</td><td>0.0147</td><td>0.0120</td><td>6.83%</td></tr><tr><td>M</td><td>Channel-wise and learned MHSA masking</td><td>0.0420</td><td>0.0556</td><td>0.0542</td><td>0.0092</td><td>0.0148</td><td>0.0121</td><td>6.21%</td></tr><tr><td>I</td><td>Median imputation + T-SHRED</td><td>0.0419</td><td>0.0465</td><td>0.0580</td><td>0.0099</td><td>0.0147</td><td>0.0121</td><td>5.68%</td></tr><tr><td>M</td><td>Input masking</td><td>0.0418</td><td>0.0540</td><td>0.0545</td><td>0.0092</td><td>0.0145</td><td>0.0122</td><td>5.24%</td></tr><tr><td>M</td><td>Channel-wise masking</td><td>0.0411</td><td>0.0553</td><td>0.0535</td><td>0.0090</td><td>0.0149</td><td>0.0122</td><td>4.88%</td></tr><tr><td>G M</td><td>Attn-GRUD</td><td>0.0420</td><td>0.0465</td><td>0.0582</td><td>0.0099</td><td>0.0149</td><td>0.0123</td><td>4.52%</td></tr><tr><td>M</td><td>Channel-wise and MHSA masking</td><td>0.0414</td><td>0.0530</td><td>0.0536</td><td>0.0091</td><td>0.0146</td><td>0.0123</td><td>4.45%</td></tr><tr><td>G</td><td>MHSA masking</td><td>0.0423</td><td>0.0567</td><td>0.0542</td><td>0.0093</td><td>0.0154</td><td>0.0124</td><td>3.83%</td></tr><tr><td>I</td><td>GRUD</td><td>0.0409</td><td>0.0469</td><td>0.0596</td><td>0.0097</td><td>0.0150</td><td>0.0125</td><td>3.09%</td></tr><tr><td>I</td><td>Mean imputation + T-SHRED</td><td>0.0423</td><td>0.0488 0.0455</td><td>0.0608</td><td>0.0100</td><td>0.0153</td><td>0.0126</td><td>1.60%</td></tr><tr><td></td><td>LOCF + T-SHRED</td><td>0.0411</td><td></td><td>0.0665</td><td>0.0097</td><td>0.0146</td><td>0.0129</td><td>0.00%</td></tr><tr><td></td><td>B: Baselines</td><td>I: Imputation</td><td></td><td>G: GRU-D</td><td></td><td>■</td><td>M: Masking</td><td></td></tr></table>

FIG. 12. Øresund Strait hydrodynamics. Root mean squared errors and normalized mean absolute errors on training, validation and test data achieved by the diferent competing models.

GRU-D-based techniques, although all strategies achieve comparable and acceptable performance. For example, the best-performing model, that is masked SHRED with learned MHSA masking, provides a NMAE on test data equal to 1.19%. To qualitatively assess the goodness of the predictions provided by the best-performing model, Figure 13 displays the ground truth and the corresponding reconstruction at three diferent time instants in the test set when extrapolating with respect to the training time range.

## D. Sea surface temperature

The last real-world application deals with sea surface temperature data, where the high-resolution snapshots (N = 64800) from January 2000 to December 2025 are obtained by combining daily satellite observations on a regular grid with in situ measurements from diferent ocean-observing platforms, such as ships and buoys, to fill possible gaps in observational coverage and correct sys tematic biases [27]. The goal of this test case is to predict the spatio-temporal high-resolution SST field over the entire globe relying solely on a limited amount of observational data provided by 20 moored buoys monitoring SST within the domain and over the same time horizon. Figure 2, panel (a), shows the spatial coverage of the considered moored buoys, while the panel (b) of the same figure displays the patterns of missing values of the SST buoy data.

![](images/e9256535a6fcd3aebc65686fe83f5a484a8983b10b1558c3f26be396bfb2c654.jpg)  
FIG. 13. Øresund Strait hydrodynamics. Ground truth, reconstruction provided by masked SHRED with learned MHSA masking and reconstruction error on surface elevation (first column), horizontal velocity (second column) and vertical velocity (third column) at three diferent time instants in the test set.

After splitting the available data into training, validation and test with ratio 70 : 15 : 15, we reduced the dimensionality of the high-resolution SST snapshots through POD by considering a reduced dimension equal to 500. Moreover, we employ a lag equal to 40, which corresponds to 40 days of historical data, and we consider 400 epochs for training. Regarding the model architectures, the recurrent temporal encoders are made of one single hidden layer of size 32, while Transformer-based encoders employ 1 block, 4 attention heads, embedding dimension 32, and a MLP with a single hidden layer of dimension 48. The spatial decoder is instead modeled via a SDN with a single hidden layer of size 256, where dropout with rate 0.1 is employed to prevent overfitting.

Figure 14 shows the ground truth and the reconstructed SST fields provided by masked SHRED with channel-wise and learned MHSA masking at four diferent time instants in the test set when extrapolating over time with respect to the training data. Figure 15, instead, compares the proposed masked SHRED strategy with CS-SHRED [15]: while both solutions are able to cope with missing sensor measurements and achieve acceptable results on the test set, masked SHRED is far more eficient both in training (15× of speed up) and during evaluation (16× of speed up), since it does not require computationally expensive convex minimization problems to recover missing values and incurs no additional computational overhead with respect to standard SHRED implementation.

While all competing methods are evaluated using the same hyperparameter values throughout the test cases discussed in Section IV, it is worth noting that additional experiments involving hyperparameter tuning yield sim ilar results and support the same overall conclusions. In particular, masked SHRED systematically outperforms imputation- and GRU-D-based methods, with only minor variations in the ranking of the diferent masking strategies. Indeed, the proposed Transformer-based masking strategies exhibit limited sensitivity to hyperparameter variations, with their efectiveness largely preserved across diferent hyperparameter values.

## V. CONCLUSIONS

This work addresses one of the main challenges in the context of sparse sensing, namely the ability to handle incomplete, heterogeneous, and noisy sensor measurements that characterize operational sensing environments. Indeed, state-of-the-art sparse sensing techniques such as, e.g., shallow recurrent decoders (SHRED), typically consider continuously active sensors with the same sampling rates, that are often directly extracted as pointwise values of the corresponding full-order state. To move toward practical settings, we first introduce a hierarchy of increasingly expressive strategies, ranging from standard imputation baselines to recurrent architectures specifically designed for irregular time-series forecasting. We then propose four Transformer-based masking strategies in order to make the SHRED temporal encoder explicitly aware of sensor availability patterns. Specifically, these strategies inject the binary data-availability mask into the Transformer-based encoder at diferent levels of the architecture, including input concatenation (channelwise masking), hidden-state gating (hidden-space masking), and additive attention biases within the multi-head self-attention mechanism, either fixed or learned (MHSA and learned MHSA masking). Finally, we introduce a multi-step decoding training strategy that enforces structured latent representations throughout the full observation window.

![](images/9e38da0aa855cdb38e573c78986929b2949f6a2ffdd8138d6f3ee29845502f49.jpg)

FIG. 14. Sea surface temperature. Ground truth, reconstruction provided by masked SHRED with channel-wise and learned MHSA masking and reconstruction error on sea surface temperature at four diferent time instants in the test set.
<table><tr><td colspan="2"></td><td colspan="3">NMAE</td><td colspan="2">Time</td><td colspan="2">Time improvement</td></tr><tr><td colspan="2">Type Configuration</td><td>Train</td><td>Val</td><td>Test</td><td>Train (min)</td><td>Inference (s)</td><td>Train</td><td>Inference</td></tr><tr><td>M</td><td>Channel-wise and learned MHSA masking</td><td>0.018</td><td>0.018</td><td>0.018</td><td>37</td><td>4</td><td>x15</td><td>x16</td></tr><tr><td>C CS-SHRED</td><td></td><td>0.018</td><td>0.019</td><td>0.018</td><td>540</td><td>63</td><td></td><td></td></tr><tr><td colspan="8">■ M: Masking C: CS-SHRED</td></tr></table>

FIG. 15. Sea surface temperature. Normalized mean absolute errors on training, validation and test data (first column) and training and inference times (second column) of the diferent competing models.

The diferent state-of-the-art imputation methods and the proposed masked shallow recurrent decoders have been employed on four diferent test cases to assess their performance. In particular, we consider a synthetic benchmark on shallow water equation solution data, where sensors are directly extracted from the corresponding full-order state and sensor malfunctions are simulated for the sake of the analysis. Moreover, we take into account three real-world operational datasets in the context of hydrodynamic and ocean modeling, thus assessing the performance of the diferent proposed strategies on realistic, intermittent and asynchronous sensor values. Across all the test cases considered, the fault-tolerant techniques considered improve the naive SHRED baseline relying solely on the always active sensors. Moreover, masked shallow recurrent decoders consistently provide accurate state reconstructions on new time steps never seen during training, in both interpolation and extrapola tion regimes, yielding lower errors than both imputation baselines and GRU-D-based approaches, with no additional inference-time computational cost. Moreover, the multi-step decoding training scheme improves upon the last-step training baseline in terms of predictive accuracy, and results in substantially smoother latent trajectories, thus potentially making latent forecasting more tractable and stable under distributional shift. Note that, in general, the diferent masking strategies proposed may be combined and there is no universally superior masking solution. However, the hybrid combinations are often ranked among the top-performing configurations across the diferent test cases, suggesting that the injection of the binary data-availability mask at multiple levels of the architecture may be beneficial.

The fault-tolerant shallow recurrent decoders introduced in this work may be extended in multiple directions in future works. For instance, one may consider an ensemble of masked SHRED models to provide robust estimates under noisy and intermittent sensor measurements, similarly to [41]. Moreover, it is possible to consider parametric problems where fault-tolerant SHRED must generalize across varying physical and geometrical parameters, in the direction of SHRED-ROM [47]. In this setting, the fault-tolerant SHRED architecture could additionally support parameter estimation and uncertainty propagation from incomplete sensor measurements to reconstructed fields. Finally, the smoother embeddings produced by the multi-step decoding training strategy may be exploited to accurately forecast the system evolution at the latent level, thus enabling masked SHRED to efectively predict full-order states in advance even when the sensors are not yet available.

## CODE AVAILABILITY

The code can be found in our repository https://github.com/matteopasqual02/ Masked-SHRED.

[1] A. C. Antoulas. Approximation of Large-Scale Dynamical Systems. SIAM, Philadelphia, PA, 2005.

[2] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normal ization. arXiv preprint arXiv:1607.06450, 2016.

[3] Y. Bao, X. Zhang, and J. N. Kutz. LAtent Phase Inference from Short time sequences using SHallow REcurrent Decoders (LAPIS-SHRED). arXiv preprint arXiv:2604.01216, 2026.

[4] P. Benner, S. Gugercin, and K. Willcox. A survey of projection-based model reduction methods for parametric dynamical systems. SIAM review, 57(4):483–531, 2015.

[5] S. L. Brunton et al. Data-driven aerospace engineering: Reframing the industry with machine learning. AIAA Journal, 59:2820–2847, 2021.

[6] S. L. Brunton and J. N. Kutz. Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press, 2019.

[7] S. L. Brunton, J. L. Proctor, and J. N. Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932–3937, 2016.

[8] D. W. Carter, F. De Voogt, R. Soares, and B. Ganapathisubramani. Data-driven sparse reconstruction of flow over a stalled aerofoil using experimental data. Data-Centric Engineering, 2:e5, 2021.

[9] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu. Recurrent neural networks for multivariate time series with missing values. Scientific reports, 8(1):6085, 2018.

[10] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

[11] E. Clark, T. Askham, S. L. Brunton, and J. N. Kutz. Greedy sensor placement with cost constraints. IEEE Sensors Journal, 19(7):2642–2656, 2019.

[12] K. Cohen, S. Siegel, and T. McLaughlin. Sensor placement based on proper orthogonal decomposition modeling of a cylinder wake. In 33rd AIAA Fluid Dynamics Conference and Exhibit. American Institute of Aeronautics and Astronautics, 2003.

[13] R. Courant and D. Hilbert. Methods of mathematical physics: partial diferential equations. John Wiley & Sons, 2008.

[14] G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals, and Systems, 2:303–314, 1989.

## ACKNOWLEDGEMENTS

MT thanks Jesper Sandvig Mariegaard and Freja Høgholm Petersen for the fruitful discussions and valuable insights on hydrodynamic modeling and real-world sensing. AM acknowledges the Project “Reduced Order Modeling and Deep Learning for the real-time approximation of PDEs (DREAM)” (Starting Grant No. FIS00003154), funded by the Italian Science Fund (FIS) - Ministero dell’Universit\`a e della Ricerca and the project FAIR (Future Artificial Intelligence Research), funded by the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, Investment 1.3, Line on Artificial Intelligence).

[15] R. B. da Silva, D. Passos, C. M. Oishi, and J. N. Kutz. CS-SHRED: enhancing SHRED for robust recovery of spatiotemporal dynamics. Proceedings of the Royal Society A Mathematical Physical and Engineering Science, 482(2344), 2026.

[16] DHI. Hydrodynamic model of Øresund: Mike21 model setup, outputs and observation data, Dec. 2024.

[17] DHI. Hydrodynamic model of the southern north sea: Mike21 model setup, outputs and observation data, Mar. 2025.

[18] C. Eckart and G. Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211– 218, sep 1936.

[19] N. B. Erichson, L. Mathelin, Z. Yao, S. L. Brunton, M. W. Mahoney, and J. N. Kutz. Shallow neural networks for fluid flow reconstruction with limited sensors. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 476(2238):20200097, 06 2020.

[20] European Union-Copernicus Marine Service. Atlanticeuropean north west shelf- ocean in-situ near real time observations, 2015.

[21] European Union-Copernicus Marine Service. Baltic seain situ near real time observations, 2015.

[22] R. Everson and L. Sirovich. Karhunen–lo\`eve procedure for gappy data. J. Opt. Soc. Am. A, 12(8):1657–1664, Aug 1995.

[23] T. Freiesleben and C. Molnar. Supervised Machine Learning for Science. 2024.

[24] M. L. Gao, J. P. Williams, and J. N. Kutz. Sparse identification of nonlinear dynamics and koopman operators with shallow recurrent decoder networks. Proceedings of the National Academy of Sciences, 123(16):e2508144123, 2026.

[25] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997.

[26] P. Holmes, J. L. Lumley, G. Berkooz, and C. W. Rowley. Turbulence, coherent structures, dynamical systems and symmetry. Cambridge university press, 2012.

[27] B. Huang, C. Liu, V. Banzon, E. Freeman, G. Graham, B. Hankins, T. Smith, and H.-M. Zhang. Improvements of the daily optimum interpolation sea surface temperature (doisst) version 2.1. Journal of Climate, 34(8):2923–2939, 2021.

[28] A. Jaegle, S. Borgeaud, J.-B. Alayrac, C. Doersch, C. Ionescu, D. Ding, S. Koppula, D. Zoran, A. Brock, E. Shelhamer, O. H´enaf, M. M. Botvinick, A. Zisser-

man, O. Vinyals, and J. Carreira. Perceiver IO: A General Architecture for Structured Inputs & Outputs. arXiv preprint arXiv:2107.14795, 2022.

[29] S. Joshi and S. Boyd. Sensor selection via convex optimization. IEEE Transactions on Signal Processing, 57(2):451–462, 2009.

[30] A. Krause, A. Singh, and C. Guestrin. Near-optimal sensor placements in gaussian processes: Theory, eficient algorithms and empirical studies. Journal of Machine Learning Research, 9:235–284, 2008.

[31] J. N. Kutz. Data-driven modeling & scientific computation: methods for complex systems & big data. Oxford University Press, 2013.

[32] J. N. Kutz, M. Reza, F. Faraji, and A. Knoll. Shallow recurrent decoder for reduced order modeling of plasma dynamics. arXiv preprint arXiv:2405.11955, 2024.

[33] B. Lindemann, T. M¨uller, H. Vietz, N. Jazdi, and M. Weyrich. A survey on long short-term memory networks for time series prediction. Procedia CIRP, 99:650– 655, 2021.

[34] Z. Luo, L. Wang, J. Xu, M. Chen, J. Yuan, and A. C. C. Tan. Flow reconstruction from sparse sensors based on reduced-order autoencoder state estimation. Physics of Fluids, 35(7):075127, 2023.

[35] K. Manohar, B. Brunton, J. Kutz, and S. Brunton. Datadriven sparse sensor placement for reconstruction. IEEE control systems, 38:63–, 05 2018.

[36] K. Manohar, E. Kaiser, S. L. Brunton, and J. N. Kutz. Optimized sampling for multiscale dynamics. Multiscale Modeling & Simulation, 17(1):117–136, 2019.

[37] R. Mitra, S. F. McGough, T. Chakraborti, C. Holmes, R. Copping, N. Hagenbuch, S. Biedermann, J. Noonan, B. Lehmann, A. Shenvi, et al. Learning from data with structured missingness. Nature Machine Intelligence, 5(1):13–23, 2023.

[38] National Data Buoy Center. Historical standard meteorological data. https://www.ndbc.noaa.gov/data/historical/stdmet/, 2026. Accessed September 2026.

[39] R. Ohana, M. McCabe, L. T. Meyer, R. Morel, F. J. Agocs, M. Beneitez, M. Berger, B. Burkhart, S. B. Dalziel, D. B. Fielding, D. Fortunato, J. A. Goldberg, K. Hirashima, Y.-F. Jiang, R. Kerswell, S. Maddu, J. M. Miller, P. Mukhopadhyay, S. S. Nixon, J. Shen, R. Watteaux, B. R.-S. Blancard, F. Rozet, L. H. Parker, M. Cranmer, and S. Ho. The well: a large-scale collection of diverse physics simulations for machine learning. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

[40] A. Quarteroni, P. Gervasio, and F. Regazzoni. Combining physics-based and data-driven models: advancing the frontiers of research with scientific machine learning. Mathematical Models and Methods in Applied Sciences, 35(04):905–1071, 2025.

[41] S. Riva, C. Introini, A. Cammi, and J. N. Kutz. Robust state estimation from partial out-core measurements with shallow recurrent decoder for nuclear reactors. Progress

in Nuclear Energy, 189:105928, 2025.

[42] S. Sahba, C. C. Wilcox, A. McDaniel, B. D. Shafer, S. L. Brunton, and J. N. Kutz. Wavefront sensor fusion via shallow decoder neural networks for aero-optical predictive control. In M. B. North-Morris, K. Creath, and R. Porras-Aguilar, editors, Interferometry XXI, volume 12223, page 1222303. International Society for Optics and Photonics, SPIE, 2022.

[43] J. E. Santos, Z. R. Fox, A. Mohan, D. O’Malley, H. Viswanathan, and N. Lubbers. Development of the senseiver for eficient field reconstruction from sparse observations. Nature Machine Intelligence, 5(11):1317–1325, 2023.

[44] A. B. Sharma, L. Golubchik, and R. Govindan. Sensor faults: Detection methods and prevalence in realworld datasets. ACM Transactions on Sensor Networks, 6(3):23, 2010.

[45] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1–9, 2015.

[46] M. Tomasetto, F. Braghin, J. N. Kutz, and A. Manzoni. Real-time optimal control with shallow recurrent decoder networks. arXiv preprint arXiv:2607.19302, 2026.

[47] M. Tomasetto, J. P. Williams, F. Braghin, A. Manzoni, and J. N. Kutz. Reduced order modeling with shallow recurrent decoder networks. Nature Communications, 16(1):10260, 2025.

[48] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[49] K. Willcox. Unsteady flow sensing and estimation via the gappy proper orthogonal decomposition. Computers & Fluids, 35(2):208–226, 2006.

[50] J. Williams, O. Zahn, and J. N. Kutz. Data-driven sensor placement with shallow decoder networks. arXiv preprint arXiv:2202.05330, 2022.

[51] J. P. Williams, O. Zahn, and J. N. Kutz. Sensing with shallow recurrent decoder networks. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 480(2298):20240054, 09 2024.

[52] A. Yermakov, D. Zoro, L. M. Gao, and J. N. Kutz. T-SHRED: symbolic regression for regularization and model discovery with transformer shallow recurrent decoders. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 384(2317), 2026.

[53] X. Zhang, X. Liang, A. Zhiyuli, S. Zhang, R. Xu, and B. Wu. AT-LSTM: An Attention-based LSTM Model for Financial Time Series Prediction. IOP Conference Series: Materials Science and Engineering, 569(5):052037, jul 2019.