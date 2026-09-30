# Accelerated surrogate dynamics for dynamical, stochastic system evolution

Marco Jochum<sup>1,2</sup> Ioannis Kouroudis<sup>1,2\*</sup> Gohar Ali Siddiqui<sup>1,2</sup> Taher Amine Hamzaoui<sup>1</sup> Manuel Gößwein<sup>1</sup> Prof. Dr. rer. nat. Alessio Gagliardi<sup>1†</sup>

<sup>1</sup> Chair of Simulation of Nanosystems for Energy Conversion, Department of Electrical Engineering, TUM School of Computation, Information and Technology, Atomistic Modeling Center (AMC), Munich Data Science Institute (MDSI), Technical University of Munich, Hans-Piloty-Straße 1, 85748 Garching, Germany

<sup>2</sup> These authors contributed equally

## Abstract

Dynamic simulations are an entrenched way of gaining insight into the evolution of system dynamics. Their computational cost however is often prohibitively high, especially in cases of stochastic frameworks. Machine learning algorithms are especially suited as simulation surrogates. Nevertheless, they face some very distinct limitations. Firstly, the sheer dimensionality of these systems, however, precludes the use of traditional time series models who struggle with high dimensional feature spaces. Additionally, traditional time series focus exclusively on either long or short range effects, causing local or global drift given enough time. In this paper, we propose a framework that addresses those limitations. Our framework combines a Variational Autoencoder, with a convolutional or graph basis that reduces the dimensionality of the system. This latent vector is propagated in time using a Temporal Fusion Transformer model, which includes both long range and short range effect encoding, as well as static covariate support. We test our framework on three distinct cases, to prove its robustness and in all three we have achieved practically identical to the simulation results at a fraction of the time. Further, our framework is flexible enough to be adapted to any new system and provides an inbuilt uncertainty quantification for targeted experiment design.

## 1 Introduction

Phenomena investigation is traditionally undertaken by laborious and meticulously designed experiments. Frequently however, these in themselves are not sufficiently able to isolate the underlying mechanisms that cause the observed dynamics. The systematic understanding of natural phenomena therefore is more deeply achieved through time-consuming physical simulations such as Kinetic Monte Carlo (KMC) [1] and molecular dynamics (MD) [2]. One significant drawback however is that to understand the physical system fully, a high computational and time load is necessary. Traditionally, this issue is mitigated in two ways. The first focuses on developing simulation tools that are more highly parallelizable, thus increasing the hardware utilization and decreasing the simulation time [3, 4, 5]. The second is more mathematical and attempts to approximate simulation elements with coarser and less expensive ones, with a controllable loss of accuracy. This can happen both on the level of domain discretization [6, 7] or on the number of necessary simulations [8, 9, 10]. Nevertheless, a highly promising and emergent third way is the use of Machine Learning to either accelerate or outright substitute the simulation results [11]. The simplest approach is to decrease the number of simulation runs necessary. This can for example take the form of efficiently determining the system parametrization that matches the experimental results[12, 13], or accelerating the trial and error process of optimal system configurations [14, 15]. Further, it can provide a coarse-grained representation of high informational content such as in the case of [16, 17]. Lastly, it can be used in a hybrid way to approximate a computationally expensive simulation step, while keeping the remaining simulation physics driven [18, 19]. Nevertheless, the creation of a complete simulation surrogate has proven a much harder proposition. A good overview of the proposed solutions can be found in [20]. Long Short Term Memory (LSTM) networks were used for dynamics surrogation such as Molecular Dynamics [21, 22] and Kinetic Monte Carlo [23]. Intrinsically however, these architectures focus on short term dynamical trends, neglecting the long term effects. On the obverse, transformers are an algorithm allows the capture for long term dynamics, at the expense of short term information. This has been successfully used in dynamics simulations [24, 25], albeit with the aforementioned limitations. Further, the above mentioned algorithms scale badly, both in accuracy and in computational cost, with increasing dimensionality inputs. Therefore, both LSTMs and transformers suffer significantly when deployed in complex applications where the geometry is high dimensional and the effects both long and short term. Additionally, these algorithms are opaque, providing predictions without any explainability behind them. To address all these issues, we developed a pipeline that simultaneously trains a dimensionality reduction part with the encoding branch of a Variational AutoEncoder (VAE), a time propagator in the form of Temporal Fusion Transformers and a dimensionality reconstruction through the decoding branch of the VAE. The VAE reduces the dimensionality to a manageable degree, the temporal fusion transformer combines LSTMs and transformers for capturing every dynamical scale and the decoder projects the results back to the original dimensionality. We show that this algorithm fusion is able to efficiently and accurately propagate trajectories of diverse simulation methods and cases. Additionally, through the unique variable selection capabilities of the propagator, we are able to assign decipher the importance of static parameters to the phenomenon evolution thus adding a degree of explainability to the pipeline. We deploy our pipeline in 3 diverse dynamical cases to showcase its versatility and robustness under highly different applications. We show that our pipeline can be successfully deployed in both stochastic and deterministic dynamical systems , presented both as space configurations and as molecular graphs.

The first and simplest is the case of spin exchange. This simulation method has been given comparatively little importance in the literature [26] and is meant to showcase the first principles of our algorithm. It nevertheless has some highly interesting applications such as water desalination [27] and hydrogen separation [28].

The second is focused around a kinetic Monte Carlo (kMC) model. This is commonly used to perform ensemble based simulations particularly for solid state electrolytes [29] and organic semi conductors [30]. Few methods have been developed to address the computational limitations of kMC with data-driven approaches and none to our knowledge reconstructs the full stochastic profile of the simulation which our framework provides.

The last case is an atomistic simulation using molecular dynamics (MD). MD has been proven over the years to be a central tool in the study of dynamic behaviour of matter. A major reason for long simulations is sampling rate parity between the different events during the dynamics [31]. This is closely related to the well-known "Sampling Problem" in statistics.

The high diversity of the simulation methods and the excellent results achieved in all is a strong indicator that our pipeline can be integrated across all material science applications and significantly expedite the phenomena investigation.

## 1.1 Methods

In this section, we present the surrogate model architecture. Generally, the main goal of any ML or surrogate model is to be as widely applicable as possible. Therefore, the surrogate is trained to autoregressively model the field of interest the cell at each time step $t , c _ { t } ^ { l } ( x , z )$ for a large set of simulation parameter combinations $l \in \{ P \}$ . Hence, the task of our surrogate model $s$ can be defined as

$$
c _ { t + 1 : t + \tau } ^ { l } ( x , z ) = \mathcal { S } ( c _ { t - k : t } ^ { l } ( x , z ) , \Lambda ^ { l } )\tag{1}
$$

where $\Lambda ^ { l }$ is a vector containing a specific combination l of input parameters to the dynamical simulation. k is the length of the sequence of past inputs used to predict the future sequence of concentrations with length $\tau .$

The idea for our model is to split the task of $s$ into three subtasks and tackle each one with a specific machine-learning algorithm. Figure 1 illustrates the three tasks. We need an encoder E, a propagator ${ \mathcal P } _ { \mathrm { { : } } }$ and finally a decoder D. E and D allow us to go from the physical space to the latent space and vice versa. The propagator $\mathcal { P }$ is specifically chosen such that it considers the sequential nature of our time series data, taking one physical state and propagating it to the next state. Traditional propagators output the next state exclusively in a recurrent manner. Our model however outputs multiple future states simultaneously with one call. This multi-horizon forecasting as this leverages the commonality expected in the subsequent steps. Succinctly the combination of the three components can be seen in fig. 1 and eq. (2) :

$$
c _ { t + 1 : t + \tau } ^ { l } ( x , z ) = \mathcal { D } [ \mathcal { P } ( \mathcal { E } ( c _ { t - k : t } ^ { l } ( x , z ) ) , \Lambda ^ { a } ) ] .\tag{2}
$$

![](images/daec14930c095fd3d3aba4242a70e761d5c31ad7cec593a7e6cfd70236abad7c.jpg)  
Figure 1: Visualization of the entire surrogate model using a multi-horizon forecasting model as the propagator P. $\mathcal { E }$ and D are the encoder/ decoder parts of the $\mathrm { V A E } , c _ { i }$ are the input configurations and $\tilde { c _ { i } }$ are the predicted next configurations. This process is repeated to predict sequences that are longer than the prediction horizon of the multi-horizon model. All the models have a $S ^ { \mu }$ branch that predicts the evolution of the dynamics. Stochastic models have an additional $S ^ { \sigma }$ branch that predicts the stochastic behaviour of the evolution.

## 1.2 Encoder E and Decoder D

For parametrizing our encoder $\mathcal { E }$ and decoder D, we chose a Variational Autoencoder (VAE)[32]. This allows for efficient dimensionality reduction while capturing the essential features of the data and thereby reducing the complexity of the task for the propagator. Additionally, it also functions as an embedding, placing physically similar configurations close to each other.

![](images/6cb8f5be10f910daaf5c717feb4c4d0d8da99e7dbdbed8d316c11af9f276f36b.jpg)  
Figure 2: Schematic representation of the encoding-decoding process. A high dimensional map $c _ { t _ { 0 } } ( x , z )$ at a given time point $t _ { 0 }$ is taken and encoded onto its respective latent space representation $z _ { t _ { 0 } }$ . It can then be decoded by propagating it through the decoder network $\mathcal { D }$ , recovering the high dimensional representation $\tilde { c } _ { t _ { 0 } } ( x , z )$ , which should be as close to the original input as possible.

The VAE is trained separately from the propagator algorithm. This design choice was made as it eases the testing and comparison of different architectures for $\mathcal { P } _ { : }$ , increases the modularity of the model, and finally, gives improved results. The loss, therefore, is geared towards training the model to encode and decode the same configuration as accurately as possible while producing a regularized latent space and can be concretely seen in eq. (3)

$$
\mathcal { L } ( \boldsymbol { c } , \tilde { \boldsymbol { c } } ) = \frac { 1 } { N _ { t o t } } \sum _ { i = 1 } ^ { N _ { t o t } } ( \boldsymbol { c } - \tilde { \boldsymbol { c } } ) ^ { 2 } - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d } ( 1 + \log ( \sigma _ { i } ^ { 2 } ) - \mu _ { i } ^ { 2 } - \sigma _ { i } ^ { 2 } ) .\tag{3}
$$

Here $N _ { t o t }$ is the total number of configurations to encode. These consists of all parameter combinations $l \in \{ P \}$ at all times steps $t \in \{ T \}$ . Two terms make up the total loss term. The first expression optimizes the reconstruction accuracy of the output c˜ with respect to the input c of via the Mean-squared error loss. The second forces the latent space mapping $q _ { x } ( z ) \mu _ { i } , \sigma _ { i }$ to be close to a standard normal $N ( 0 , 1 )$ distribution. In our model, we chose to weigh both loss contributions equally as they are on a similar scale, and this provides good accuracy and regularization.

## 1.3 Temporal Fusion Transformer

The latent space propagator chosen is the Temporal Fusion Transformer(TFT)[33] which was specifically developed for time series forecasting. It takes as input the reduced dimension latent space of the VAE and predicts its evolution over time. If the full configuration is desired, it can be re-decoded via the decoder element of the VAE. This model is superior to models used in similar applications because of its

1. Ability to use static and dynamic metadata i.e., simulation parameters

2. the learning of short- and long-term dependencies

3. A posteriori insight into variable importance

It can accept mixed inputs, specifically static, past, and future covariates that are taken into account in addition to the standard input to make predictions, such as time series with high correlation to the prediction target. Static covariates are quantities that give context to an input sequence but remain unchanged over time, e.g., the values of the conserved properties of our simulation. Past and future covariates serve a similar purpose but are dynamic in time. Past covariates are only known up to the prediction starting point.

Several key features of the TFT architecture allow it to learn the complex relations between these mixed input signals. We will give a short introduction to these main features but the reader is encouraged to read [33] for more details.

Variable Selection Network: Most ML models are black box functions that allow for little to no physical interpretation. Even though TFT is ultimately a highly complex, it allows for relative variable importance weighting by introducing a Variable Selection Block. The color coding in Figure 3 for the Variable Selection blocks indicates that for each type of input, i.e., static, past, and future, an individual block is used. Continuous input variables are linearly transformed into a $d _ { m o d e l }$ dimensional vector before being processed by the Variable selection cell.

While each variable has its respective GRN, the weights are shared across all time steps t. Finally, the processed feature vectors are multiplied with selection weights and summed up.

Static Covariate encoders The static covariate encoders denoted by the orange box in Figure 3 are specifically designed to allow the model to utilize contextual metadata throughout the prediction process. Four unique context vectors ${ \bf c } _ { s } , { \bf c } _ { e } , { \bf c } _ { c }$ and $\mathbf { c } _ { h }$ (orange arrows in Figure 3) are passed into the model for (1) Temporal variable selection $( \mathbf { c } _ { s } )$ , (2) initialization of the LSTM encoders for local processing $( \mathbf { c } _ { c } , \mathbf { c } _ { h } )$ and (3) to enrich the temporal features with static information in the decoder block $( \mathbf { c } _ { e } )$ . GRN cells are used to create these context vectors.

Long Short Term Memory (LSTM) Encoders Short range dependencies are captured by LSTM blocks [34], which share inputs derived from the static covariate encoder and the variable selection. The blocks processing past covariates share weights with each other. Equally, the blocks processing future known covariates share weights as well, as indicated by the colour coding of fig. 3.

Gating Mechanism A Gated Residual Network (GRN) is introduced to allow the weighing of different input variables and switching between linear and non-linear processing. This block processes the primary input and an optional context vector:

LayerNorm is the standard layer normalization introduced in [35].Weights are shared across the layer, in Figure 3 as is indicated by the color coding of the different GRN blocks. Component gating layers are used, which are based on Gated Linear Units (GLUs) [36]. GLUs allow the model to learn how much the GRN should modify the original input. If no context vector is passed the block accepts 0s as inputs and is organically deactivated.

Interpretable Multihead Attention: In [33] a modified version of the Multi-Head attention is presented to improve model introspection. Attention weights in each head depend on the specific value weights in that head $\mathbf { W } _ { V } ^ { h }$ . Hence, they cannot be aggregated and evaluated. Therefore, a modified attention algorithm that shares values across heads is introduced

Now the value weights are shared across all attention heads and combined by a linear mapping. We can interpret this new form of attention as an ensemble of multiple attention heads that can attend to different temporal patterns.

![](images/f4edb1b44da53fc34d690d7a49eea5f3ff4baf21402f224dc7b757f323bb8a64.jpg)  
Figure 3: Temporal Fusion Transformer model, adapted from [33] Static covariates S and time-varying inputs — past observations $( x _ { t - k } . . . x _ { t } )$ and known future covariates $( x _ { t + 1 } . . . x _ { t + \tau _ { m a x } } )$ enter through dedicated Variable Selection Networks (orange), which learn instance-wise feature weights for interpretability. The Static Covariate Encoder (dark orange) produces four context vectors used throughout the network: one for variable selection, two to initialize the LSTM’s cell/hidden state, and one for static enrichment. A sequence-to-sequence LSTM Encoder–Decoder (blue = past, dark red = future) provides local temporal processing, replacing standard positional encoding. Every sub-block is wrapped in a Gate + Add and Norm unit (yellow/green), a GLU-based residual gate that lets the network skip unused components. LSTM outputs are enriched with static context via a Static Enrichment GRN (purple), then passed to Masked Interpretable Multi-Head Attention (teal), which lets each time step attend to relevant past and future points while preserving causal masking. This is followed by a Position-wise Feed-Forward GRN, another gated residual block, and a final FC layer producing quantile forecasts $( \tilde { z } _ { t + \tau } ( 0 . 1 )$ , (0.5), (0.9)) at each horizon step — enabling calibrated uncertainty estimates alongside point predictions.

## 2 Stochastic Methods

## 2.1 Metropolis Monte Carlo Model for ion separation

The Metropolis Monte Carlo model, as described in [37], treats a system as a two-dimensional grid where dipole spins are placed randomly at lattice sites. The The spins are constrained to take one of two values: up (+1) or down (−1). The lattice is defined by dimensions $W \times L$ , and in this example $W = L$ , such that the total number of spins is $L \times L$ . As a result, for a specific sites i, a Hamiltonian is produced

$$
\epsilon _ { i } = - J \sum _ { j } ( s _ { i } s _ { j } - 1 )\tag{2}
$$

where the sum runs over the nearest neighbours of i-the Spin. The coupling constant J characterizes the natural interaction within the Ising Model, with its sign determining the nature of the interaction. A positive J corresponds to a ferromagnetic system, where the spins tend to align parallel to minimize energy. Conversely, a negative J signifies an antiferromagnetic system, where the spins prefer an antiparallel alignment. We will consider a grid of spins where each spin can either be up (+1) or down (-1). The goal is to minimize the magnetic energy of the system by performing spin exchanges based on the energy difference between configurations. The deep learning pipeline is operating similarly, with TFT predicting the energy evolution of a system given its parameters and a recurrent VAE predicting the next configuration given the energy predicted by TFT.

We use this simulation set up to investigate the spin exchange of a 2 dimensional system. We differentiate different systems by altering the following properties of the system.

Table 1: Simulation Parameters, as this is an abstract case the units are irrelevant.
<table><tr><td>Parameter Description</td><td colspan="2"></td></tr><tr><td>J</td><td>Exchange interaction energy between spins</td><td>0.1, 0.5, 1, 2, 3, 5, 6, 7, 8</td></tr><tr><td> $K _ { B T }$ </td><td>Thermal energy (Boltzmann constant multiplied by temperature)</td><td>0.1, 0.5, 0.8, 1, 2, 2.5, 3, 5.</td></tr><tr><td>Ratio</td><td>The probability  $P ( X = 1 )$ </td><td>0.1, 0.25, 0.4, 0.5, 0.75, 0.8</td></tr></table>

To fully describe the phenomenon we need to reconstruct the energy evolution, as well as the convergent energy of each parameter combination. The results are presented in fig. 4 and show a high energy correlation with the ground truth ( lower than 5%) In this instance, the exact configuration of the ions is of little interest as multiple configurations will be equivalent. Nevertheless, a full visualization of the phenomenon is of significant aid to the understanding of it. To this end, we modified the VAE architecture described in fig. 2 in two significant ways. The first added physical insight to the latent space by appending the predicted energy (via TFT) and the spin ratio to it. The loss function, additionally to the reconstruction, included the MSE between the predicted and true energy and ratio of configurations. To further improve the results we modified the architecture to resemble that of a Residual Neural Network (ResNet) [38]. In that architecture the predicted target is not the next step but the difference of the current step to the next. The reconstructed next step is then created by adding the output of the network to the input, resembling the analytical solution of the explicit Euler propagation scheme. As the spin values could only be between 0 and 1, we further processed the end result with a sigmoid. The results of the reconstructed images, along with their energies can be seen in fig. 4.

As is obvious, the method works robustly and well, and the addition of the ResNet provides a consistent foundation for the image reconstruction.

![](images/e41df598c6d7897a5f8a40c3b39625d7d9054cde513707967275413f58a9be05.jpg)  
Figure 4: Snapshots generated through the physical simulation (up) and snapshots generated through our framework (down) with their energies and corresponding timesteps

## 2.2 Kinetic Monte Carlo for Solid State Electrolytes

Even though the Ising model described in section 2.1 provides a good proof of concept, applications of interest require more complex simulation methods. To further test our prediction pipeline, we deployed it to a full kinetic Monte Carlo model applied in the investigation of the boundary layer formation of solid state electrolytes. We used the full configutaion as input to the VAE. The resulting latent space time series provided the time series predicted by TFT, while the simulation paramters provided the static covariates. The decoder of the VAE expanded the predicted latent space to the original configuration.

## 2.3 Kinetic Monte Carlo

The kinetic Monte Carlo method provides a numerical algorithm to model physical phenomena by coarse-graining the dynamics into a set of long-term states. This, in turn, enables, for example, the simulation of battery systems at the device scale. In the case of SSE modeling, this localized state i is represented by the charge carrier distribution at a certain time. The physical knowledge about the process to be modeled is encoded into the transition rates between these localized states. They are implicitly defined a priori either from experiment or from underlying model equations. Implicitly, because often they are dependent on the local environment and therefore need to be computed dynamically during the simulation. Hence, by choosing the rates, we determine which processes are to be included in the model and which processes are coarse-grained into the long-term states. All transitions on smaller time scales are neglected as long as they do not change the long-term state [39]. In the current work we chose to simulate the boundary layer formation in Solid State Electrolytes (SSEs) In general, the mass transport of Li-ions in SSEs can be captured by a thermally activated hopping mechanism between unoccupied vacancies in a crystal lattice. The crystal structure itself consists of immobile anions, cations and vacancies as well as mobile cations. The kMC simulation only considers the transport of mobile $L i ^ { + }$ within a three-dimensional regular grid of vacancies with lattice constant $a _ { \mathrm { L } }$ . Note that the implemented grid does not resemble the actual morphology of the SSE sample but rather must be regarded as a simple lattice gas model. In this framework, SCL formation is caused by the mere redistribution of mobile Li-ions driven by an applied bias potential, $\phi _ { \mathrm { b i a s } }$ . The exact model and approximations taken can be found in [12].

The results can be seen in fig. 5 and validate the robustness of our model.

![](images/4194e0f8f711c7667b91de31d4d3b007168e98bb07724172ce2ca492264d74bd.jpg)

![](images/36a4d1c130d231ddf9d7ab70c83ab04d3275adbe17599f7ec90d3610e3efa8d9.jpg)

![](images/fd99729bd3cfcdd0d03284a4b0327bd2fea570cc7ec5ad71978e48dc464db5b8.jpg)

![](images/7c303894bdff7a6c4eae5224b78515aeb773a5108743f9c5d049d33fea5e8acb.jpg)  
a) =[677, 6.0e+18 cm <sup>3</sup>, 0.5 V]

![](images/f0baa0801c3fad04d79105a73ec2586196ee52f913e08723fc38e6e835b81f81.jpg)  
b) =[1400, 3.0e+18 cm <sup>3</sup>, 0.05 V]

![](images/e77aad079cdc45c1f6c6006fc4a83cf1f0c0c9458dc419e802ebf060554630f4.jpg)  
c) =[1400, 1.0e+19 cm <sup>3</sup>, 0.2 V]  
Figure 5: Predictions of the surrogate model for the mean field of the three different parameter configurations a)-c. The combination of parameters strongly affects the final concentration profile as well as the dynamic process that precedes the steady-state concentration distribution. The time stamps correspond to the following list of time steps: [10, 20, 50, 100, 200, 300]. The represetation shows the evolution of the concentration summed up along the Z axis and averaged over multiple simulation runs.

Additionally, KMC’s main advantage is its stochastic nature. To this end, we also show that our model can reclaim the underline epistemic uncertainty by predicting the standard deviation field, as shown in fig. 6.

![](images/4a2c095488e4977ae7913bb3ea9a80a3d270cb7c6eacf2ce9f2425d39dc238cc.jpg)

![](images/796b49e14c58d4d80578694ee0e7a45320281a9505327818f081ce18b9e7a5b7.jpg)

![](images/84837acf512786fcb71bff6509c21c0186b3062589e8356290481feffdee2f83.jpg)

![](images/4a9b3097efde6ac92d4d141b90d2289a6178294c793d6a7dd957b97b256615c0.jpg)  
a) =[677, 6.0e+18 cm <sup>3</sup>, 0.5 V]

![](images/ab136ce21a01e5669e2fed6e3ec3c9593e1c462463e1dbf84cfa71b4c233f08e.jpg)  
b) =[1400, 3.0e+18 cm <sup>3</sup>, 0.05 V]

![](images/69690ceddc3cb1b889aeeed8293324f0337ccd541e486049a01705f5c1af0d8a.jpg)  
c) =[1400, 1.0e+19 cm <sup>3</sup>, 0.2 V]  
Figure 6: Prediction for the standard deviation fields of the kMC simulation. The represetation shows the evolution of the concentration summed up along the Z axis and averaged over multiple simulation runs. Parameter configurations are the same as in Figure 5.

Furthermore, the training time of the model is in the order of minutes, while the required inference time is measured in seconds. The simulation time however spans days. This adds an additional argument in favour of our framework, namely its high accuracy is compounded by many orders of magnitude increase in efficiency.

## 3 Non Stochastic Methods

## 3.1 Dynamics of Molecular Systems

To design molecular systems in an effective manner, their configurational space needs to be explored in a robust way. Atomistic simulations using density functional theory (DFT) are regularly used in design exploration but this level of theory is limited in time and spatial scalability. Molecular dynamics use computationally efficient empirical equations of motion, parameterized usually using DFT data, to break this scalability barrier by yielding accuracy and the ability to calculate electronic properties. This offers huge possibilities to study dynamics of large complex systems. Nevertheless as the systems under study get larger and more complex, this method also becomes computationally prohibitive. One area of research for the acceleration of dynamics is the consideration of metastable states in which the system can become stuck in a local minima until a rare event leads to an escape. The dynamics inside the metastable basin, although of little consequence to the conformational dynamics, need to be sampled for a large number of steps. As the next step in the development of our framework, we applied it as a surrogate model for dynamics of a molecular system. To this end, we chose a coarse grained model of alanine dipeptide. It has a well defined and widely studied conformational space spanned by the two dihedral angles ϕ and $\psi .$

![](images/10f2df15811169bf044046ddaf0628f5139e5ddd67ba2599c3aee4ffd6c6df83.jpg)  
Figure 7: The system used for surrogate of molecular dynamics. The hydrogens are removed from the MD configurations. The two ramachandran angle ϕ and $\psi$ are indicated.

## 3.2 Latent dynamics

We ignore the hydrogen atoms due to their high frequency vibrations and negligible effects on the conformational dynamics. To produce a latent embedding dynamics, we use a bond-graph architecture inspired by [40] to focus on the topological information. The encoder $\mathcal { E }$ takes as input a graph $\mathcal { G } ^ { E } \in \{ \mathcal { V } ^ { E } , B ^ { E } \}$ that is constructed by the internal coordinates of the molecular conformations. We use bonds as vertices featurized with atomic numbers of the two atoms and the distance between them $\mathcal { V } ^ { E } = \mathbb { B } _ { i } \sim \left( a _ { 0 } , a _ { 1 } , d _ { i } \right)$ where i denotes the dependence on considered frame (time step). Edges represent either angles or torsion angles featurized by the value of the respective angle $\mathcal { B } ^ { E } = \mathbb { A } _ { i } \sim ( 1 , 0 , a _ { i } ) + \mathbb { T } _ { i } \sim ( 0 , 1 , t _ { i } )$ $\mathbb { B } _ { i }$ denotes bonds in frame $i , \mathbb { A } _ { i }$ the angle created by pairs and $\mathbb { T } _ { i }$ by triplets of bonds. The encoder E acts on the graph $\mathcal { G } _ { i } ^ { E }$ by first embedding the scalar features by a set of learnable MLPs (one for each feature) consisting of one hidden layer of length $H _ { e }$ to compute the initial embeddings ${ \mathbf h } _ { a } ^ { 0 }$ for each node a and $\mathbf h _ { b } ^ { 0 }$ for each edge b. The steps of the encoding are then as follows;

1. L message passing steps similar to [41]:

$$
\begin{array} { l } { { \displaystyle \alpha _ { a b } = s o f t m a x ( \frac { ( \mathbf { W } _ { 3 } \mathbf { h } _ { a } ^ { l } ) ^ { T } ) ( \mathbf { W } _ { 4 } \mathbf { h } _ { b } ^ { l } + \mathbf { W } _ { 6 } c _ { a b } } { \sqrt { H _ { e } } } ) } \ ~ } \\ { { \displaystyle ~ \mathbf { m } _ { a } = \sum _ { b \in \mathcal { N } ( + ) \ } \alpha _ { a b } ( \mathbf { W } _ { 2 } \mathbf { h } _ { b } ^ { l } + \mathbf { W } _ { 6 } c _ { a b } ) } \ ~ } \\ { { \displaystyle \beta _ { a } = s i g m o i d ( \mathbf { W } _ { 5 } [ W _ { 1 } \mathbf { h } _ { a } ^ { l } , \mathbf { m } _ { a } , \mathbf { W } _ { 1 } \mathbf { h } _ { a } ^ { l } - \mathbf { m } _ { a } ] ) } \ ~ } \\ { { \displaystyle \mathbf { h } _ { a } ^ { l + 1 } = \beta _ { a } \mathbf { W } _ { 1 } \mathbf { h } _ { a } ^ { l } + ( 1 - \beta _ { a } ) \mathbf { m } _ { a } } } \end{array}
$$

where $W _ { * }$ are learnable parameters, $H _ { e }$ is hidden size of attention heads, [a, b] represent vector concatenation of a and $b , c _ { a b }$ are the edge features of edges a and b and $\mathcal { N } ( a ) = \{ b \ | \ ( a , b ) \in B \}$ ELU nonlinearities and batch normalization is applied between each layer.

2. Pooling is done via a learnable set-to-set mapping using an LSTM layer:

$$
\begin{array} { r l } & { { \bf q } _ { 0 } = [ 0 . . . 0 ] ^ { T } } \\ & { e _ { a , t } = h _ { a } ^ { a } \cdot { \bf q } _ { t } } \\ & { \gamma _ { a , t } = \frac { \exp \left( e _ { a , t } \right) } { \displaystyle \sum _ { b } \sigma _ { x } p \sigma ( c _ { b , t } ) } } \\ & { { \bf r } _ { t } = \sum _ { a = 1 } ^ { N } \gamma _ { a , t } h _ { i } ^ { L } } \\ & { { \bf q } _ { t } ^ { * } = [ { \bf q } _ { t } , { \bf r } _ { t } ] } \\ & { { \bf q } _ { t + 1 } = L S T M \left( q _ { t } ^ { * } \right) } \end{array}
$$

Where · denotes dot product. We perform $T$ aggregations steps

3. Final linear layer, the latent embedding for frame i, $\mathbf { z } _ { i }$ with length $L _ { d }$ is obtained by:

$$
{ \bf z } _ { i } = \Phi ( q _ { T } ^ { * } )
$$

The encoding process of bonds $\mathbb { B } _ { i } ,$ angles $\mathbb { A } _ { i }$ and torsion angles $\mathbb { T } _ { i }$ for frame i can be represented by the following equation

$$
\begin{array} { r } { \mathbf { z } _ { i } = \mathcal { E } ( \mathcal { G } _ { i } ^ { E } ) , \quad \mathcal { G } _ { i } ^ { E } = \mathcal { G } ^ { E } ( \mathbb { B } _ { i } , \mathbb { A } _ { i } , \mathbb { T } _ { i } ) } \end{array}
$$

For the decoder model, we use another graph neural network with nodes encoding atomic species and edges encoding neighbor atoms. We only include time-invariant topological information in this graph which is processed via $L$ message passing steps and combined with the latent space vector $\mathbf { z } _ { i }$ to obtain the time-dependent topological variable (bond lengths, angles and dihedrals). The decoder graph $\mathcal { G } ^ { D } \in \{ \mathcal { V } ^ { D } , B ^ { D } \}$ is obtained with the following components;

$\mathcal { V } ^ { D }$ is a concatenation of time-invariant variables;

1. Scalar features: atomic number, bond degree, number of rings the atom is involved in, implicit valence, formal charge, number of bonded hydrogens

2. Categorical features: chirality, hybridization type, is it in an aromatic ring, is it in a 5-ring, is it in a 6-ring, the name of the residue

Categorical embedding is used for categorical features. The categories are based on the enum structures defined in RDkit library.

$B ^ { D }$ contains all the bond-neighbors in the molecule. In addition we extend the connections between any atoms that can be reached with a maximum of k hops. Thus $B ^ { D } = \{ ( a , b ) ~ | ~ a ~ \in$ $\mathcal { V } \wedge b \in \mathcal { N } ^ { k } ( a ) \}$ }, where $\mathcal { N } ^ { k } ( a )$ represent the up to k-hop neighbors. The edges are featurized with a categorical variable defining the type of connection (single, double, triple, aromatic, virtual) with virtual as the category of more than 2-hop neighbors. The edges between bonded atoms additionally have the equilibrium bond distance in the feature vector which is zero in the other edges.

The decoder D contains the same attention-based message passing steps as E. After L steps, we use the node embeddings $\mathbf { h } _ { a } ^ { L }$ along with the latent space vector $\mathbf { z } _ { i }$ to get the predictions of the time-dependent topological variable;

$$
\begin{array} { r l } & { \quad d _ { a b } ^ { i } = \Gamma _ { b o n d } ( [ \mathbf { h } _ { a } ^ { L } , \mathbf { h } _ { b } ^ { L } , \mathbf { z } _ { i } ] ) \forall ( a , b ) \in \mathbb { B } } \\ & { \quad \phi _ { a b c } ^ { i } = \Gamma _ { a n g l e } ( [ \mathbf { h } _ { a } ^ { L } , \mathbf { h } _ { b } ^ { L } , \mathbf { h } _ { c } ^ { L } , \mathbf { z } _ { i } ] ) \forall ( a , b , c ) \in \mathbb { A } } \\ & { \cos \psi _ { a b d c } ^ { i } = \Gamma _ { t o r _ { c o s } } ( [ \mathbf { h } _ { a } ^ { L } , \mathbf { h } _ { b } ^ { L } , \mathbf { h } _ { c } ^ { L } , \mathbf { h } _ { d } ^ { L } , \mathbf { z } _ { i } ] ) \forall ( a , b , c , d ) \in \mathbb { T } } \\ & { \sin \psi _ { a b c d } ^ { i } = \Gamma _ { t o r _ { s i n } } ( [ \mathbf { h } _ { a } ^ { L } , \mathbf { h } _ { b } ^ { L } , \mathbf { h } _ { c } ^ { L } , \mathbf { h } _ { d } ^ { L } , \mathbf { z } _ { i } ] ) \forall ( a , b , c , d ) \in \mathbb { T } } \end{array}
$$

Note that this encode-decoder architecture can be trained on arbitrary size of graphs and thus can be used to train a single model for multiple molecules.

## 3.3 Dataset

Our dataset is generated via a 100 ns molecular dynamics simulations of alanine dipeptide in water. The coordinates of the atoms of alanine dipeptide is extracted every 100 fs. For the propagator model (TFT), we perform some coarse-graining of the trajectories by taking average positions of the atoms over a window of $l _ { c g }$ steps. This facilitates the model to learn the underlying dynamics instead of trying to predict the thermal vibrations present during MD at finite temperatures. For better performance in predicting sequencing longer than the input sequence length, we sample, we sample multiple long sequences starting from random points in the trajectory and extracting ${ l _ { c g } ( l _ { i n } + l _ { s e q } l _ { o u t } ) }$ frames each time where $l _ { s e q }$ is the length of the sequences we want to train on and $l _ { i n }$ and $l _ { o u t }$ is the input chunk lenght and output chunk length of the TFT model. Each such sequence is a data point and we extract train\_size + validation\_size number of such sequences. The idea to sample from random starting points is to make the model robust against variations in initial points.

## 3.4 Surrogate dynamics

## 3.4.1 Training

The encoder-decoder network is trained first directly on randomly sampled frames from the MD trajectory. Mean square loss is used for training. We found the separate training to perform better than training both the models together. During the training of the propagator, the weights of the encoder-decoder model are frozen. The input to the propagator model is z<sub>i</sub> as past covariates which comes from encoding the trajectory using the encoder. The loss functional is a sum of the error in the latent space prediction and the error in the final predicted topology. The latent space prediction uses gaussian regression with negative loss likelihood as the loss function. The error in the predicted topology is calculated by the mean squarred difference to the true values from MD, same as in the training the encoder-decoder network. Note that the different types of topological variables (bond, angles, torsions) are given different weight in the loss function.

## 3.4.2 Prediction

During prediction, $l _ { i n }$ number of steps are sampled from the MD trajectory as the starting point for the model. The model encoder encodes the model and while the TFT forecasts the future latent space points. Since we use gaussian likelihood regression as the loss function, the latent space points for the decoder must be sampled from a gaussian for which the model outputs the mean $\mu _ { m o d e l }$ and variance $\sigma _ { m o d e l }$ . To replicate the MD dynamics at inference time, instead of $\sigma _ { m o d e l }$ , we use $\sigma _ { m d }$ which is variance directly computed from the latent space embedding of true MD trajectory (without coarse-graining). Additionally, to improve transition dynamics, we perform Independent Component Analysis (ICA) of a preliminary prediction run of 50000 steps of the model and compare with the same analysis done on MD trajectory of same number of steps starting from a random point. The ratio between the mean of the eigenvalues from MD and model is used as a scaling factor for the variance $\sigma _ { m d }$ during prediction. Finally the sampled values are decoded via the decoder and topological values of interest are extracted.

## 3.4.3 Results

In fig. 8 we compare the Free Energy Surface spanned by the two dihedrals $\phi$ and $\psi$ which are the standard natural coordinates of the system. To obtain the FES, we perform kernel density approximation with a gaussian kernel of width 0.2 and bin size of 200. The surrogate model is in good overall agreement with the ground truth values. It is able to also capture the depth of the rate minimum at high $\phi$ values with decent accuracy.

![](images/4f6dd33f40827e68dfdf05e6b92caccdce0e1ca40ed4034539a3776ce14816e8.jpg)  
(a) FES of true dynamics.

![](images/eff85da45d34759a64da5750da9c796af6c0439a74db74206088f08b93f856a9.jpg)  
(b) FES of predicted dynamics.  
Figure 8: Comparison of Free Energy Surface (FES) of the dihedral angles $\phi$ and $\psi$ between true and surrogate dynamics.

To look more closely at the accuracy of the transition statistics of the surrogate vs the simulated dynamics, we create a Markov State Model in the $\phi - \psi$ space. More specifically, to find the states, we start at 200 random points in the $\phi - \psi$ and run minimization algorithm using the BFGS method using the FES as the objective function. Then we cluster the final points using a proximity criterion with threshold of 0.15 radians and label the top $^ 6$ populous clusters and the states (see minima labels in fig. 8). We each point in the trajectory and assign it to the nearest state. Finally we use the number of counts the system transition from one state to other during the whole trajectory to build the transition counts matrix. We can then also calculate the transition probability matrix and the Mean First Passage Time (MFPT) matrix. The last one gives a measure number of step (400 fs for out setup) the simulation takes on average to observe a specific transition.

We show the transition probabilities of the MD dynamics and surrogate model in fig. 9. The surrogate dynamics capture the state transitions probabilities very closely hinting at the model learning a projection of true Boltzmann distribution of the state space of the dynamics.

![](images/e0b50ba35a032d5349685e72b07625f4da1043f4e39f0846222da7f43d0fa236.jpg)  
(a) Transition probabilities of simulation dynamics.

![](images/1f5e079481647447461b18183ef47baa550802e8cf6eb059db3a2c33edb71969.jpg)  
(b) Transition probabilities of surrogate dynamics.  
Figure 9: Comparison of Transition probabilities between simulation and surrogate dyanamics. States labeled in fig. 8

## 4 Efficiency gain and Result physicality

The previous chapters showcase the accuracy of the framework but do not shine sufficient light to the major advantage of our framework, namely its efficiency. Indicatively KMC simulations, would require from days to weeks to converge, while the TFT training required a fraction of that and its deployment an even more insignificant compute. Succintly, the comparative time requirements can be seen in fig. 10 and for inference are without fail multiple orders of magnitude faster.

Execution Times by Task and Method  
![](images/34eb4c6ef40e8366a2e7d3bf13d947eba4f086af5e7486237039134cb6ce611f.jpg)  
Figure 10: Time required for the execution of the simulation, the training of the model and the inference.

Further, it must be noted, that the first two simulations will eventually converge to an equilibrium change that is unchanged. It is of high importance that the surrogate model also reaches a steady unchangeable state. This is crucial as it proves that even if some accuracy loss occurs, the results do not deviate but instead converge, recreating physical stability as well as physical correctness.

## 5 Outlook

In conclusion, we have presented a versatile and robust framework, able to surrogate even complex and stochastic dynamics. Our time evolution model combines both long term predictions, captured by the transformer architecture, as well as local temporal relations, as determined by the LSTM. The model’s ability to incorporate both time independent and time dependent covariates allows the inclusion of physical parameters, as well as time series that are less accurate but much faster to generate and still carry information about the target. Gated Units and Feature analysis naturally quantify the importance of the underlying physical parameters are seamlessly included. The difficulty of the propagation task is alleviated by the use of advanced Autoencoder architectures. Additionally to robust and continuous dimensionality reduction, they also provide full configuration profile visualization which adds to the understanding of the investigated phenomenon. The applicability of our framework was tested on multiple and diverse simulation methodologies and was found to perform well in all. Most impressively, we also measured a high computational efficiency increase, of at least three orders of magnitude, case dependent. We are therefore convinced that our framework can offer significant advantages in the simulation world by both accelerating dynamics investigation and preserving the intrinsic physical uncertainty of the simulations.

## 6 Acknowledgements

I.K. acknowledges funding from the Project ProperPhotoMile, supported under the umbrella of SOLAR-ERA.NET Cofund 2 by The Spanish Ministry of Science and Education and the AEI under the project PCI2020-112185 and CDTI project number IDI-20210171; the Federal Ministry for Economic Affairs and Energy on the basis of a decision by the German Bundestag project number FKZ 03EE1070B and FKZ 03EE1070A and the Israel Ministry of Energy with project number 220-11-031. SOLAR-ERA.NET is supported by the European Commission within the EU Framework Programme for Research and Innovation HORIZON 2020 (Cofund ERA-NET Action, N° 786483).

Further, M.G. acknowledges funding from the European Union’s Horizon 2020 FETOPEN 2018–2020 program “LION-HEARTED” under grant agreement no. 828984.

Finally, A.G. acknowledges financial support from TUM Innovation Network for Artificial Intelligence powered Multifunctional Material Design (ARTEMIS) and funding in the framework of Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC 2089/1 – 390776260 (e-conversion).

## References

[1] Leon Katzenmeier, Manuel Gowein, Alessio Gagliardi, and Aliaksandr S Bandarenka. Modeling of space-charge layers in solid-state electrolytes: a kinetic monte carlo approach and its validation. The Journal ofPhysical Chemistry C, 126(26):10900–10909, 2022.

[2] Scott A Hollingsworth and Ron O Dror. Molecular dynamics simulation for all. Neuron, 99(6):1129–1143, 2018.

[3] Sander Pronk, Szilárd Páll, Roland Schulz, Per Larsson, Pär Bjelkmar, Rossen Apostolov, Michael R Shirts, Jeremy C Smith, Peter M Kasson, David Van Der Spoel, et al. Gromacs 4.5: a high-throughput and highly parallel open source molecular simulation toolkit. Bioinformatics, 29(7):845–854, 2013.

[4] Floris Laporte, Joni Dambre, and Peter Bienstman. Highly parallel simulation and optimization of photonic circuits in time and frequency domain based on the deep-learning framework pytorch. Scientific reports, 9(1):5918, 2019.

[5] Rajive Bagrodia, Richard Meyer, Mineo Takai, Yu-an Chen, Xiang Zeng, Jay Martin, and Ha Yoon Song. Parsec: A parallel simulation environment for complex systems. Computer, 31(10):77–85, 1998.

[6] Eberhard Bänsch. Local mesh refinement in 2 and 3 dimensions. IMPACT of Computing in Science and Engineering, 3(3):181–191, 1991.

[7] Randolph E Bank, Andrew H Sherman, and Alan Weiser. Some refinement algorithms and data structures for regular local mesh refinement. Scientific Computing, Applications of Mathematics and Computing to the Physical Sciences, 1:3–17, 1983.

[8] Valentina Tozzini. Coarse-grained models for proteins. Current opinion in structural biology, 15(2):144–150, 2005.

[9] Sergei Izvekov and Gregory A Voth. A multiscale coarse-graining method for biomolecular systems. The Journal ofPhysical Chemistry B, 109(7):2469–2473, 2005.

[10] Jaehyeok Jin, Alexander J Pak, Aleksander EP Durumeric, Timothy D Loose, and Gregory A Voth. Bottom-up coarse-graining: Principles and perspectives. Journal of chemical theory and computation, 18(10):5759–5791, 2022.

[11] Felix Mayr, Milan Harth, Ioannis Kouroudis, Michael Rinderle, and Alessio Gagliardi. Machine learning and optoelectronic materials discovery: A growing synergy. The Journal of Physical Chemistry Letters, 13(8):1940–1951, 2022.

[12] Ioannis Kouroudis, Manuel Gosswein, and Alessio Gagliardi. Utilizing data-driven optimization to automate the parametrization of kinetic monte carlo models. The Journal ofPhysical Chemistry A, 127(28):5967–5978, 2023.

[13] Kilian Vernickel, Laura Brunner, Georg Hoellthaler, Giuseppe Sansivieri, Christian Hardtlein, Ludwig Trauner, Lukas Bank, Jan Fischer, and Julia Berg. Machine-learning-based approach for parameterizing material flow simulation models. Procedia CIRP, 93:407–412, 2020.

[14] Ioannis Kouroudis, Neel Misciasci, Felix Mayr, Leon Müller, Zhaosu Gu, Alessio Gagliardi, et al. Augur, a flexible and efficient optimization algorithm for identification of optimal adsorption sites. npj Computational Materials, 11(1):1–13, 2025.

[15] Milica Todorovic, Michael U Gutmann, Jukka Corander, and Patrick Rinke. Bayesian inference of atomistic structure in functional materials. Npj computational materials, 5(1):35, 2019.

[16] Gohar Ali Siddiqui, Julia A Stebani, Darren Wragg, Phaedon-Stelios Koutsourelakis, Angela Casini, and Alessio Gagliardi. Application of machine learning algorithms to metadynamics for the elucidation of the binding modes and free energy landscape of drug/target interactions: a case study. Chemistry–A European Journal, 29(62):e202302375, 2023.

[17] Jiang Wang, Simon Olsson, Christoph Wehmeyer, Adrià Pérez, Nicholas E Charron, Gianni De Fabritiis, Frank Noé, and Cecilia Clementi. Machine learning of coarse-grained molecular dynamics force fields. ACS central science, 5(5):755–767, 2019.

[18] Dmitrii Kochkov, Jamie A Smith, Ayya Alieva, Qing Wang, Michael P Brenner, and Stephan Hoyer. Machine learning–accelerated computational fluid dynamics. Proceedings ofthe National Academy ofSciences, 118(21):e2101784118, 2021.

[19] Pascal Friederich, Florian Häse, Jonny Proppe, and Alán Aspuru-Guzik. Machine-learned potentials for next-generation matter simulations. Nature Materials, 20(6):750–761, 2021.

[20] Frank Noé, Alexandre Tkatchenko, Klaus-Robert Müller, and Cecilia Clementi. Machine learning for molecular simulation. Annual review ofphysical chemistry, 71(1):361–390, 2020.

[21] Debby D Wang, Le Ou-Yang, Haoran Xie, Mengxu Zhu, and Hong Yan. Predicting the impacts of mutations on protein-ligand binding affinity based on molecular dynamics simulations and machine learning methods. 2020.

[22] Bipeng Wang, Ludwig Winkler, Yifan Wu, Klaus-Robert Muller, Huziel E Sauceda, and Oleg V Prezhdo. Interpolating nonadiabatic molecular dynamics hamiltonian with bidirectional long shortterm memory networks. The journal ofphysical chemistry letters, 14(31):7092–7099, 2023.

[23] Chi Ho Lee, Silabrata Pahari, Niranjan Sitapure, Mark A Barteau, and Joseph Sang-Il Kwon. Investigating high-performance non-precious transition metal oxide catalysts for nitrogen reduction reaction: a multifaceted dft–kmc–lstm approach. ACS Catalysis, 13(13):8336–8346, 2023.

[24] Max Eissler, Tim Korjakow, Stefan Ganscha, Oliver T Unke, Klaus-Robert Müller, and Stefan Gugler. How simple can you go? an off-the-shelf transformer approach to molecular dynamics. The Journal of Chemical Physics, 164(9), 2026.

[25] Sihao Yuan, Xu Han, Jun Zhang, Zhaoxin Xie, Cheng Fan, Yunlong Xiao, Yi Qin Gao, and Yi Isaac Yang. Generating high-precision force fields for molecular dynamics simulations to study chemical reaction mechanisms using molecular configuration transformer. The Journal ofPhysical Chemistry A, 128(21):4378–4390, 2024.

[26] Tongyu Liu, Katherine R Johnson, Santa Jansone-Popova, and De-en Jiang. Advancing rare-earth separation by machine learning. JACS Au, 2(6):1428–1434, 2022.

[27] Pattarachai Srimuk, Xiao Su, Jeyong Yoon, Doron Aurbach, and Volker Presser. Charge-transfer materials for electrochemical water desalination, ion separation and the recovery of elements. Nature Reviews Materials, 5(7):517–538, 2020.

[28] Nathan W Ockwig and Tina M Nenoff. Membranes for hydrogen separation. Chemical reviews, 107(10):4078–4110, 2007.

[29] Leon Katzenmeier, Manuel Goesswein, Alessio Gagliardi, and Aliaksandr S Bandarenka. Modeling of space-charge layers in solid-state electrolytes: a kinetic monte carlo approach and its validation. The Journal of Physical Chemistry C, 126(26):10900–10909, 2022.

[30] Waldemar Kaiser, Johannes Popp, Michael Rinderle, Tim Albes, and Alessio Gagliardi. Generalized kinetic monte carlo framework for organic electronics. Algorithms, 11(4):37, 2018.

[31] Srinivasan S. Iyengar, H. Bernhard Schlegel, Isaiah Sumner, and Junjie Li. Rare events sampling methods for quantum and classical ab initio molecular dynamics. The Journal of Physical Chemistry A, 128(27):5386–5397, 2024.

[32] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[33] Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal ofForecasting, 37(4):1748– 1764, 2021.

[34] S Hochreiter. Long short-term memory. Neural Computation MIT-Press, 1997.

[35] Jimmy Lei Ba. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[36] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In International conference on machine learning, pages 933–941. PMLR, 2017.

[37] Jacques Kotze. Introduction to monte carlo methods for an ising model of a ferromagnet. arXiv preprint arXiv:0803.0217, 2008.

[38] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[39] Kurt E. Sickafus, E. A. Kotomin, and Blas P. Uberuaga. Radiation effects in solids: [proceedings ofthe NATO Advanced Study Institute on Radiation Effects in Solids Erice, Sicily, Italy, 17-29 July 2004], volume 235 of NATO science series. II, Mathematics, physics and chemistry. Springer, Dordrecht Netherlands, 2007.

[40] Simon Dobers, Hannes Stark, Xiang Fu, Dominique Beaini, and Stephan Günnemann. Latent space simulator for unveiling molecular free energy landscapes and predicting transition dynamics. In NeurIPS 2023 AIfor Science Workshop, 2023.

[41] Yunsheng Shi, Zhengjie Huang, shikun feng, Hui Zhong, Wenjin Wang, and Yu Sun. Masked label prediction: Unified message passing model for semi-supervised classification, 2021.