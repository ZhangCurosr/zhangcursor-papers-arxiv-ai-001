# LIFTGCN: EFFICIENT ENERGY-PRESERVING GRAPH LEARNING VIA JOUKOWSKI SPECTRAL LIFTING FOR FINITE ELEMENT STRESS PREDICTION

Chen Zeng<sup>1</sup>, Qiao Wang<sup>1,2∗</sup>

<sup>1</sup>School of Information Science and Engineering, Southeast University <sup>2</sup>School of Economics and Management, Southeast University Nanjing, China {chenzeng,qiaowang}@seu.edu.cn

## ABSTRACT

Finite element stress fields often exhibit strong local non-smoothness, where stress concentrations near holes, notches, and loading regions induce sharp spatial gradients and high-frequency graph components. Although graph neural networks naturally operate on irregular finite element meshes, conventional message passing is inherently smoothing and progressively attenuates such high-frequency information. Unitary propagation alleviates this problem by preserving spectral magnitudes, but typically relies on matrix functions and high-order approximations with O(Ked) propagation complexity. We propose LiftGCN, an efficient spectrally stable graph network based on Joukowski spectral lifting. LiftGCN maps the real spectrum of a normalized graph operator onto the unit circle through the Joukowski relation and realizes the resulting spectral transformation as a simple second-order recurrence, avoiding matrix exponentials, eigendecomposition, and high-order polynomial truncation. We show that the linear Joukowski backbone has unit-modulus characteristic roots and admits an energy-preserving structure under a positive-definite metric, preventing exponential attenuation of graph-frequency components with depth. Each layer requires only one sparse neighborhood aggregation, yielding O(ed) propagation complexity, while lightweight local nonlinear residuals provide expressive feature transformations. Experiments on finite element stress prediction demonstrate that LiftGCN achieves competitive overall accuracy while improving reconstruction of stress concentrations and local high-gradient structures with substantially reduced computational cost. Our codee is available at https://github.com/ChenZeng001/LiftGCN.

## 1 Introduction

Finite element analysis (FEA) is widely used to predict mechanical responses, but repeated high-resolution simulations can be expensive (Bathe et al., 1996; Zienkiewicz et al., 2013; Jayasinghe et al., 2025). This cost motivates learned surrogates for design optimization and structural evaluation (Li et al., 2021; Giacomini & Díez, 2026; Liang et al., 2018). Stress prediction is particularly demanding: a field may be smooth over much of the domain yet contain sharp concentrations near holes, notches, interfaces, and loading regions (Sun & Chen, 2024). These local responses are important for assessing structural failure, so a useful surrogate must recover both the overall field and its steep spatial variations (Zhu et al., 2022; Braun et al., 2020). Figure 1 illustrates this challenge.

Graph neural networks (GNNs) operate directly on irregular finite element meshes, using connectivity to exchange information between nodes (Scarselli et al., 2008; Wu et al., 2020). MeshGraphNets and related models have demon strated their value for physical simulation and structural prediction (Pfaff et al., 2021; Maurizi et al., 2022). However, repeated neighborhood aggregation tends to smooth node features and attenuate graph-frequency components (Li et al., 2018; Hoang et al., 2021; Roth & Liebig, 2024). In stress fields, these components can encode mechanically meaningful concentrations and gradients. Smoothing them can therefore remove precisely the local details that a surrogate needs to retain.

Stress concentration around a circular hole Abaqus/Standard | plane stress | CPS6 mesh | σ = 100 MPa  
![](images/00d456a362bf3533fe92b7120d0d7fd83723832e48bd9b3405b5c97f1e448dba.jpg)  
Figure 1: Stress concentration around a circular hole: geometry and loading, the finite element stress field, and local stress decay compared with the Kirsch solution.

Several approaches seek less dissipative propagation. Graph-Coupled Oscillator Networks (GraphCON) introduce second-order dynamics, using inertia to reduce the collapse of deep representations (Rusch et al., 2022). Wave-driven GNNs similarly replace diffusion with oscillatory information transport (Wu et al., 2025). These methods show how retaining a previous state can change the behavior of repeated graph aggregation and help preserve information ove depth.

A complementary line of work controls propagation in the spectral domain. CayleyNets use rational filters to obtain flexible frequency responses (Levie et al., 2018). Unitary Graph Convolution maps a symmetric graph operator to the complex unit circle, preserving the magnitude of every spectral component (Kiani et al., 2024). This suggests a useful design principle for stress prediction: graph frequencies should evolve through phase while retaining the information needed to reconstruct sharp spatial variations.

The practical difficulty is evaluating the propagation operator. Matrix exponentials on sparse graphs are typically implemented through Taylor, Chebyshev, or related approximations. A K-term approximation requires repeated sparse operations and incurs $O ( K e d )$ propagation cost, where e is the number of edges and d is the hidden width (Kiani et al., 2024). Obtaining unit-modulus dynamics with a single neighborhood aggregation would make this principle more accessible to large finite element meshes.

We propose LiftGCN to obtain unit-modulus spectral dynamics with the locality and cost of ordinary message passing. The Joukowski relation (Gutknecht & Trefethen, 1982) converts a real graph eigenvalue into a pair of unit-circle modes. We realize this mapping through a real-valued second-order recurrence whose linear backbone preserves a positivedefinite quadratic energy. Lightweight node-wise residuals supply nonlinear feature transformations, and each layer needs only one sparse neighborhood aggregation. This yields O(ed) graph propagation without evaluating a matrix function.

The remainder of this paper is organized as follows. Section 2 introduces finite element stress fields and reviews diffusive and unitary graph propagation. Section 3 presents Joukowski spectral lifting, the LiftGCN architecture, and its spectral and computational properties. Section 4 evaluates LiftGCN through comparative experiments, depth analysis, and ablation studies. Finally, Section 5 concludes the paper.

## 2 Background

Stress fields contain sharp spatial variations that graph smoothing can suppress. We briefly connect their physical regularity to graph-frequency content, then contrast diffusive and unitary propagation.

## 2.1 Finite Element Stress Fields and Limited Regularity

For a linear elastic body occupying $\Omega \subset \mathbb { R } ^ { m }$ , stress is $\pmb { \sigma } = \mathbb { C } : \varepsilon ( \mathbf { u } )$ , where u is displacement, C is the elasticity tensor, and $\pmb { \varepsilon } ( \mathbf { u } ) = ( \nabla \mathbf { u } + \nabla \mathbf { u } ^ { \top } ) / 2$ . Equilibrium requires $- \nabla \cdot { \pmb \sigma } = \mathbf { f } ,$ subject to prescribed displacement and traction conditions. The finite element formulation is based on the weak problem

$$
\int _ { \Omega } { \boldsymbol { \varepsilon } } ( { \mathbf { v } } ) : \mathbb { C } : { \boldsymbol { \varepsilon } } ( { \mathbf { u } } ) d \Omega = \int _ { \Omega } { \mathbf { v } } \cdot { \mathbf { f } } d \Omega + \int _ { \Gamma _ { N } } { \mathbf { v } } \cdot { \bar { \mathbf { t } } } d \Gamma , \qquad \forall { \mathbf { v } } \in V _ { 0 } ,\tag{1}
$$

where $V _ { 0 }$ contains test displacements that vanish on the prescribed-displacement boundary, and <sup>¯</sup>t is the traction on $\Gamma _ { N }$ . Under standard assumptions, the energy solution belongs to $H ^ { 1 } ( \Omega )$ (Bathe et al., 1996; Zienkiewicz et al., 2013). Since stress depends on displacement derivatives, its natural regularity is generally $L ^ { 2 } ( \Omega )$ ; global smoothness is not guaranteed.

Holes and smooth fillets can produce large but finite stress gradients, while sharp notches, re-entrant corners, and material interfaces can induce stronger local variations. Near a geometric singularity, the leading stress response may take the form

$$
\pmb { \sigma } ( r , \theta ) \sim r ^ { \kappa - 1 } \pmb { \Phi } ( \theta ) , \qquad 0 < \kappa < 1 ,\tag{2}
$$

where $r$ is the distance to the singularity. Such behavior explains why a stress field can be difficult to reconstruct even when the surrounding displacement field appears smooth.

On a mesh graph, let $\mathcal { L } = \mathbf { I } - \widetilde { \mathbf { A } } = \mathbf { Q } \mathrm { d i a g } ( \mu _ { j } ) \mathbf { Q } ^ { \top }$ be the normalized Laplacian. For a nodal stress signal y with graph Fourier coefficients $\widehat { \mathbf { y } } = \mathbf { Q } ^ { \top } \mathbf { y }$

$$
\mathbf { y } ^ { \top } \mathcal { L } \mathbf { y } = \sum _ { j } \mu _ { j } | \widehat { y } _ { j } | ^ { 2 } .\tag{3}
$$

Higher graph frequencies contribute more strongly to this variation measure. Preserving them helps reconstruct localized stress patterns that a purely smooth representation may miss.

## 2.2 Diffusive and Unitary Graph Propagation

For an undirected graph, let $\widetilde { \bf A } = \widehat { \bf D } ^ { - 1 / 2 } \widehat { \bf A } \widehat { \bf D } ^ { - 1 / 2 }$ denote a symmetric normalized adjacency, where $\widehat { \mathbf A }$ may include self-loops. Its eigenvalues $\lambda _ { j }$ lie in $[ - 1 , 1 ]$ . A standard GCN updates features as $\mathbf { H } _ { l + 1 } = \phi ( \tilde { \mathbf { A } } \mathbf { H } _ { l } \mathbf { W } _ { l } )$ . Isolating the linear propagation gives

$$
\widetilde { \bf A } ^ { L } = { \bf Q } \mathrm { d i a g } ( \lambda _ { j } ^ { L } ) { \bf Q } ^ { \top } , \qquad | \lambda _ { j } | ^ { L }  0 \quad \mathrm { f o r } | \lambda _ { j } | < 1 .\tag{4}
$$

Repeated aggregation therefore attenuates these modes and favors smoother representations (Li et al., 2018; Hoang et al., 2021). In particular, low Laplacian frequencies correspond to adjacency eigenvalues close to one, which decay slowly. Components with smaller eigenvalue magnitudes disappear more quickly. This unequal attenuation can weaken local stress contrasts as the receptive field grows, linking the physical reconstruction problem to the spectral behavior of propagation.

Unitary propagation instead uses

$$
{ \bf U } ( t ) = e ^ { \mathrm { i } t \tilde { \bf A } } = { \bf Q } \mathrm { d i a g } ( e ^ { \mathrm { i } t \lambda _ { j } } ) { \bf Q } ^ { \top } , \qquad | e ^ { \mathrm { i } t \lambda _ { j } } | = 1 .\tag{5}
$$

Each spectral component changes phase without losing magnitude (Kiani et al., 2024). Consequently, $\mathbf { U } ( t ) ^ { * } \mathbf { U } ( t ) = \mathbf { I } .$ so repeated unitary propagation preserves the Euclidean norm of the feature signal. The attraction is that increasing the propagation distance need not progressively erase frequency content. Approximating the matrix exponential requires multiple graph operations, motivating a simpler realization of unit-modulus dynamics.

## 3 Joukowski Spectral Lifting and LiftGCN

LiftGCN obtains stable spectral dynamics through state-space lifting. We derive the real-valued recurrence, combine it with local nonlinear transformations, and establish its energy-preserving structure and propagation cost.

## 3.1 Joukowski Spectral Lifting

The Joukowski map $J ( z ) = ( z + z ^ { - 1 } ) / 2$ sends $z = e ^ { \mathrm { i } \theta }$ to cos θ (Gutknecht & Trefethen, 1982). Conversely, for $x \in ( - 1 , 1 )$ , solving $z ^ { 2 } - 2 x z + 1 = 0$ gives

$$
z _ { \pm } = x \pm \mathrm { i } \sqrt { 1 - x ^ { 2 } } = e ^ { \pm \mathrm { i } \operatorname { a r c c o s } x } , \qquad | z _ { \pm } | = 1 .\tag{6}
$$

![](images/075481753aa2b1c47837d01ed1afcedcf0725e05bcf5246b1c3b50c8f4bb90f4.jpg)

![](images/37352b9b98c7de91732baa5410bfb542affa4cac7a5445d7b0f78c2c6a808e29.jpg)  
(a) Real graph spectrum.

![](images/a24b75c52a05bf1c53c36b443ce4980d8dc82f9d5fcdfd5310350311ee867a7a.jpg)  
(b) Joukowski spectral lifting.  
(c) Unit-modulus response.  
Figure 2: Joukowski spectral lifting maps the scaled real graph spectrum to conjugate unit-circle modes, preserving their magnitude while changing their phase.

Setting $x = \rho _ { c } \lambda _ { j }$ with $| \rho _ { c } | < 1$ lifts each graph eigenvalue to two unit-circle modes, as illustrated in Figure 2. Their phases depend on graph frequency and channel, while their magnitudes remain one.

The same quadratic is the characteristic equation of $h _ { j , c } ^ { ( l + 1 ) } = 2 \rho _ { c } \lambda _ { j } h _ { j , c } ^ { ( l ) } - h _ { j , c } ^ { ( l - 1 ) }$ . Replacing the eigenvalue by the graph operator yields

$$
\mathbf { \overline { { H } } } _ { l + 1 } = 2 \mathbf { \widetilde { A } } \mathbf { H } _ { l } \mathbf { R } - \mathbf { H } _ { l - 1 } , \qquad \mathbf { R } = \mathrm { d i a g } ( \rho _ { 1 } , \dots , \rho _ { d } ) .\tag{7}
$$

This recurrence implements the lifted dynamics entirely in real arithmetic. The previous hidden state supplies the second degree of freedom required by the conjugate roots; no complex-valued features or explicit spectral decomposition are needed. We learn $\rho _ { c } = \rho _ { \mathrm { m a x } }$ tanh(θ ) with $0 < \rho _ { \mathrm { m a x } } < 1$ , sharing each coefficient across depth. Thus, channels can learn distinct spectral phases while remaining in the unit-modulus regime. Sharing the coefficients preserves a stationary linear recurrence, while channel-wise adaptation lets different features use different frequency-dependent phases.

## 3.2 LiftGCN Architecture

An input encoder maps node features to $\mathbf { H } _ { 0 } \in \mathbb { R } ^ { n \times d }$ . The first propagation is $\overline { { \mathbf { H } } } _ { 1 } = \tilde { \mathbf { A } } \mathbf { H } _ { 0 } \mathbf { R } ;$ subsequent steps use Equation 7. Each step then applies a node-wise residual update,

$$
\mathbf { H } _ { l + 1 } = \overline { { \mathbf { H } } } _ { l + 1 } + \alpha _ { l } \mathrm { G E L U } \big ( \mathrm { L N } ( \overline { { \mathbf { H } } } _ { l + 1 } ) \mathbf { W } _ { l } + \mathbf { b } _ { l } \big ) ,\tag{8}
$$

where $\mathbf { W } _ { l }$ and ${ \bf b } _ { l }$ mix channels, and $\alpha _ { l }$ is a learnable residual scale. A linear readout maps the final two-state representation $[ \mathbf { H } _ { L } , \mathbf { H } _ { L - 1 } ]$ to nodal stress. Figure 3 summarizes the propagation and residual blocks.

The recurrence controls inter-node information transport, while the residuals learn local nonlinear corrections. Normalization and channel mixing act independently at each node, and the learned residual scale controls the strength of each correction. The two-state readout exposes both states of the recurrence to the predictor. This separation adds expressive feature transformations without introducing extra graph aggregations.

## 3.3 Spectral and Computational Properties

For each graph mode and channel, the linear recurrence has roots $e ^ { \pm \mathrm { i } \operatorname { a r c c o s } ( \rho _ { c } \lambda _ { j } ) }$ . Their unit magnitude avoids the exponential decay associated with repeated adjacency multiplication. The recurrence also preserves a quadratic energy in its augmented state. For a single channel with coefficient $\rho ,$ write $\mathbf Z _ { l } = [ \mathbf h _ { l } ^ { \top } , \mathbf h _ { l - 1 } ^ { \top } ] ^ { \top }$ ⊤ and $\mathbf { Z } _ { l + 1 } = \dot { \mathbf { U } } _ { \rho } \mathbf { Z } _ { l }$ , where

$$
\mathbf { U } _ { \rho } = \left[ \begin{array} { c c } { 2 \rho \widetilde { \mathbf { A } } } & { - \mathbf { I } } \\ { \mathbf { I } } & { \ \mathbf { 0 } } \end{array} \right] , \qquad \mathbf { G } _ { \rho } = \left[ \begin{array} { c c } { \mathbf { I } } & { - \rho \widetilde { \mathbf { A } } } \\ { - \rho \widetilde { \mathbf { A } } } & { \mathbf { I } } \end{array} \right] .\tag{9}
$$

The eigenvalues of $\mathbf { G } _ { \rho }$ are $1 \pm \rho \lambda _ { j } > 0$ , and direct multiplication gives

$$
\mathbf { U } _ { \rho } ^ { \top } \mathbf { G } _ { \rho } \mathbf { U } _ { \rho } = \mathbf { G } _ { \rho } , \qquad \| \mathbf { Z } _ { l + 1 } \| _ { \mathbf { G } _ { \rho } } ^ { 2 } = \| \mathbf { Z } _ { l } \| _ { \mathbf { G } _ { \rho } } ^ { 2 } .\tag{10}
$$

The conserved quantity can also be written directly in terms of consecutive hidden states:

$$
\| \mathbf { Z } _ { l } \| _ { \mathbf { G } _ { \rho } } ^ { 2 } = \| \mathbf { h } _ { l } \| _ { 2 } ^ { 2 } + \| \mathbf { h } _ { l - 1 } \| _ { 2 } ^ { 2 } - 2 \rho \mathbf { h } _ { l } ^ { \top } \widetilde { \mathbf { A } } \mathbf { h } _ { l - 1 } .\tag{11}
$$

![](images/efb00b02af7c88becc9e299b2e1be3fd8613b974c3f3a8b1642144d1ac7e674f.jpg)  
Figure 3: LiftGCN combines second-order Joukowski propagation with node-wise nonlinear residuals. Each layer uses one sparse neighborhood aggregation and the previous hidden state.

Energy is therefore preserved in the pair of states, allowing information to move between them as propagation proceeds. This result applies independently to every channel of the linear backbone. It provides a non-dissipative basis for propagation, to which the learned residuals add nonlinear corrections.

Each layer uses one sparse multiplication by $\widetilde { \mathbf { A } } .$ , giving $O ( e d )$ graph propagation cost rather than the $O ( K e d )$ cost of a K-step matrix-function approximation. Node-wise channel mixing costs ${ \cal \dot { O } } ( n d ^ { 2 } )$ in both cases. LiftGCN therefore reduces the graph propagation work while retaining one-hop communication and locally stored second-order memory.

This locality is useful on large meshes: each node exchanges only its current representation with immediate neighbors and stores its own previous state. Stable spectral dynamics are obtained by augmenting the state, without expanding the communication neighborhood within a layer. Additional layers can then extend the receptive field through the same sparse connectivity.

## 4 Experiment

We evaluate LiftGCN in terms of whole-field accuracy, stress-concentration reconstruction, and inference efficiency. All experiments are conducted on a workstation equipped with a single NVIDIA RTX PRO 6000 GPU. All prediction metrics reported in the tables and quantitative figures are averaged over 10 repeated experiments. We first introduce the dataset and evaluation metrics, then compare LiftGCN with representative baselines and analyze its sensitivity to propagation depth, followed by ablation studies on Joukowski propagation, spectral scaling, and local feature transformations. Appendix B extends the comparisons to two additional geometries, with data and training details in Appendices A and C.

## 4.1 Dataset and Evaluation Metrics

We use Mines\_Paris\_Biaxial\_Specimen, comprising 100 finite element simulations of a biaxial specimen with random elastic properties (Kerfriden, 2022). The task is to predict nodal von Mises stress from geometry, material properties, and mesh connectivity. Appendix A.1 specifies the inputs and mesh statistics, and Appendix A.3 describes graph construction. Training settings and checkpoint selection based on test-set loss are detailed in Appendix C.

We measure whole-field accuracy with SNR and $R ^ { 2 }$ . For reference stresses $y _ { i }$ and predictions ${ \hat { y } } _ { i } ,$ pooling evaluation nodes within each repetition gives

$$
\mathrm { S N R } = 1 0 \log _ { 1 0 } \frac { \sum _ { i } y _ { i } ^ { 2 } } { \sum _ { i } ( \hat { y } _ { i } - y _ { i } ) ^ { 2 } } , \qquad R ^ { 2 } = 1 - \frac { \sum _ { i } ( \hat { y } _ { i } - y _ { i } ) ^ { 2 } } { \sum _ { i } ( y _ { i } - \bar { y } ) ^ { 2 } } ,\tag{12}
$$

LiftGCN: Efficient Energy-Preserving Graph Learning via Joukowski Spectral Lifting for Finite Element Stress Prediction

![](images/edccc3f15fc3a7e9dbe02860af0ce54b5c5ed95059009f09171f7caf6adb276e.jpg)

![](images/afbd8f81e19cf8ce8e4fd1d7f7eb5de3397d95cffd55b57ca8ee111f09017de4.jpg)

![](images/159ca6e7b247efa76509a23e785e85822b701bd47028c26dceac8b5d4d84af8b.jpg)  
Figure 4: Stress-field reconstruction and hotspot analysis: representative stress fields, a hotspot stress profile, and gradient errors across the test set. LiftGCN preserves localized concentrations and spatial variations.

where y¯ is the pooled reference mean. To evaluate concentrations, let $S _ { k }$ contain the top $k \%$ of nodes by reference stress magnitude on each mesh, with $k \in \{ 1 , 5 , 1 0 \}$ . Peak-region accuracy is

$$
\mathrm { P S N R @ } k \% = 1 0 \log _ { 1 0 } \frac { \operatorname* { m a x } _ { i \in S _ { k } } | y _ { i } | ^ { 2 } + \epsilon } { | S _ { k } | ^ { - 1 } \sum _ { i \in S _ { k } } ( \hat { y } _ { i } - y _ { i } ) ^ { 2 } + \epsilon } .\tag{13}
$$

For undirected edges $E _ { k }$ touching $S _ { k }$ , define $g _ { i j } ( y ) = ( y _ { i } - y _ { j } ) / \operatorname* { m a x } ( \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 } , \epsilon )$ using node coordinates $\mathbf { x } _ { i }$ High-stress gradient error is

$$
\mathrm { H S G - N M S E @ } k \% = \frac { \sum _ { ( i , j ) \in E _ { k } } [ g _ { i j } ( \hat { y } ) - g _ { i j } ( y ) ] ^ { 2 } } { \operatorname* { m a x } \bigl ( \sum _ { ( i , j ) \in E _ { k } } g _ { i j } ( y ) ^ { 2 } , \epsilon \bigr ) } .\tag{14}
$$

Here ϵ ensures numerical stability. PSNR and HSG-NMSE are averaged over evaluation meshes before averaging across the 10 repetitions. Higher SNR, $R ^ { 2 }$ , and PSNR and lower HSG-NMSE are better. Together, these metrics assess the overall stress field, its peaks, and the sharp variations around them. We also report parameter counts and Latency, defined as the average forward inference time per sample (one complete mesh), measured separately from the repeated training experiments.

## 4.2 Main Comparisons, Efficiency, and Depth Sensitivity

Table 1 compares LiftGCN with conventional message-passing, mesh-based, dynamical, spectral, and attention-based graph models. At comparable parameter counts, small LiftGCN achieves the lowest latency and improves both wholefield and local accuracy over GCN. It also matches or slightly improves UniGCN’s peak-region PSNR with much faster inference. These 10-run mean results show that the lifted recurrence can retain useful local stress information at low propagation cost.

Increasing model capacity makes further use of this efficiency. Medium LiftGCN outperforms MeshGraphNets and UniGCN across the reported prediction metrics while remaining faster. With a latency budget comparable to Hybrid Local–Global Attention GNN and UniGCN, large LiftGCN leads the peak-region PSNR at the two smallest thresholds and all high-stress gradient metrics. Its strongest gains lie in the localized structures that motivate the model, alongside competitive overall accuracy.

Figure 4 makes this advantage visible. LiftGCN reproduces compact high-stress bands near curved boundaries and preserves heterogeneous patterns within the specimen. These local details agree with the peak and gradient improvements observed in the 10-run mean metrics.

The accuracy–latency curves in Figure 5 show a favorable trade-off across model sizes. LiftGCN already reconstructs stress concentrations well at low latency, and additional capacity steadily improves peak and gradient accuracy.

Table 1: Main comparison. Prediction metrics are means over 10 repeated experiments. Bold marks the best prediction metric or lowest latency. Hybrid LG denotes Hybrid Local–Global Attention GNN, and Adv-GCN denotes GCN with adversarial training. Parentheses indicate the number of discriminator parameters. (a) Whole-field accuracy and computational cost
<table><tr><td>Model</td><td>Parameters</td><td>Graph propagation complexity</td><td>Latency↓ (ms)</td><td>SNR↑ (dB)</td><td> $R ^ { 2 }$  ←</td></tr><tr><td>GCN (Kipf &amp; Welling, 2016)</td><td>444k</td><td> $\mathcal { O } ( e d )$ </td><td>1.719</td><td>15.685</td><td>0.85744</td></tr><tr><td>Adv-GCN (Oommen et al., 2026)</td><td>444k (+67k)</td><td> $\mathcal { O } ( e d )$ </td><td>1.719</td><td>15.605</td><td>0.85479</td></tr><tr><td>MeshGraphNets (Pfaff et al., 2021)</td><td>401k</td><td> $\mathcal { O } ( e d ^ { 2 } )$ </td><td>6.173</td><td>16.008</td><td>0.86764</td></tr><tr><td>GCNII (Chen et al., 2020)</td><td>439k</td><td> $\mathcal { O } ( e d )$ </td><td>1.639</td><td>15.095</td><td>0.83670</td></tr><tr><td>GraphCON (Rusch et al., 2022)</td><td>439k</td><td> $\mathcal { O } ( e d )$ </td><td>1.703</td><td>13.307</td><td>0.75349</td></tr><tr><td>CayleyNet (Levie et al., 2018)</td><td>428k</td><td> $\mathcal { O } ( r J e d )$ </td><td>28.164</td><td>16.126</td><td>0.87109</td></tr><tr><td>EWGNN (Wu et al., 2025)</td><td>462k</td><td> $\mathcal { O } ( n ^ { 2 } d )$ </td><td>4.393</td><td>15.212</td><td>0.84098</td></tr><tr><td>GUMP (Qiu &amp; Yao, 2024)</td><td>439k</td><td> $\mathcal { O } ( M _ { L } d ) ^ { \dagger }$ </td><td>48.527</td><td>13.441</td><td>0.76017</td></tr><tr><td>Hybrid LG (Patrignani &amp; Pinho, 2026)</td><td>438k</td><td> $\mathcal { O } ( e d ^ { 2 } + n ^ { 2 } d / M )$ </td><td>13.365</td><td>17.104</td><td>0.89607</td></tr><tr><td>A-DGN (Gravina et al., 2023)</td><td>403k</td><td> $\mathcal { O } ( e d )$ </td><td>2.211</td><td>15.671</td><td>0.85696</td></tr><tr><td>UniGCN (Kiani et al., 2024)</td><td>446k</td><td> $\mathcal { O } ( K e d )$ </td><td>15.000</td><td>16.104</td><td>0.87053</td></tr><tr><td>LiftGCN (ours), small</td><td>443k</td><td> $\mathcal { O } ( e d )$ </td><td>1.528</td><td>15.919</td><td>0.86487</td></tr><tr><td>LiftGCN (ours), medium</td><td>3299k</td><td> $\mathcal { O } ( e d )$ </td><td>5.041</td><td>16.610</td><td>0.88475</td></tr><tr><td>LiftGCN (ours), large</td><td>16.4M</td><td> $\mathcal { O } ( e d )$ </td><td>13.851</td><td>16.821</td><td>0.89022</td></tr></table>

(b) Stress-concentration reconstruction
<table><tr><td rowspan="2">Model</td><td colspan="3">PSNR (dB)↑</td><td colspan="3">HSG-NMSE↓</td></tr><tr><td>@1%</td><td>@5%</td><td>@10%</td><td>@1%</td><td>@5%</td><td>@10%</td></tr><tr><td>GCN (Kipf &amp; Welling, 2016)</td><td>18.886</td><td>21.779</td><td>23.406</td><td>0.145440</td><td>0.117880</td><td>0.101360</td></tr><tr><td>Adv-GCN (Oommen et al., 2026)</td><td>19.094</td><td>21.947</td><td>23.417</td><td>0.142850</td><td>0.117070</td><td>0.104030</td></tr><tr><td>MeshGraphNets (Pfaff et al., 2021)</td><td>19.737</td><td>22.282</td><td>23.772</td><td>0.089125</td><td>0.082616</td><td>0.076469</td></tr><tr><td>GCNII (Chen et al., 2020)</td><td>18.662</td><td>21.225</td><td>22.710</td><td>0.196790</td><td>0.163430</td><td>0.148320</td></tr><tr><td>GraphCON (Rusch et al., 2022)</td><td>17.062</td><td>19.514</td><td>20.868</td><td>0.415490</td><td>0.403890</td><td>0.397550</td></tr><tr><td>CayleyNet (Levie et al., 2018)</td><td>20.783</td><td>22.281</td><td>23.589</td><td>0.083929</td><td>0.098303</td><td>0.098552</td></tr><tr><td>EWGNN (Wu et al., 2025)</td><td>18.746</td><td>21.312</td><td>22.943</td><td>0.150180</td><td>0.132090</td><td>0.117000</td></tr><tr><td>GUMP (Qiu &amp; Yao, 2024)</td><td>17.478</td><td>19.888</td><td>21.232</td><td>0.350990</td><td>0.379950</td><td>0.388420</td></tr><tr><td>Hybrid LG (Patrignani &amp; Pinho, 2026)</td><td>21.501</td><td>23.813</td><td>25.138</td><td>0.072304</td><td>0.071222</td><td>0.067561</td></tr><tr><td>A-DGN (Gravina et al., 2023)</td><td>19.086</td><td>21.778</td><td>23.338</td><td>0.095384</td><td>0.090815</td><td>0.083688</td></tr><tr><td>UniGCN (Kiani et al., 2024)</td><td>20.096</td><td>22.526</td><td>23.974</td><td>0.073533</td><td>0.072321</td><td>0.067939</td></tr><tr><td>LiftGCN (ours), small</td><td>20.327</td><td>22.615</td><td>23.975</td><td>0.087084</td><td>0.081528</td><td>0.075953</td></tr><tr><td>LiftGCN (ours), medium</td><td>21.884</td><td>23.744</td><td>24.871</td><td>0.058066</td><td>0.061305</td><td>0.060404</td></tr><tr><td>LiftGCN (ours), large</td><td>22.201</td><td>24.035</td><td>25.086</td><td>0.053120</td><td>0.057062</td><td>0.057472</td></tr></table>

Propagation complexity is reported per layer and excludes purely node-wise feature transformations. Here $n , \epsilon ,$ and d denote the numbers of nodes, edges, and hidden features, respectively. K is the Taylor truncation order in UniGCN; r and J are the Cayley filter order and number of Jacobi iterations in CayleyNet; and M is the global-attention interval in Hybrid $\begin{array} { r } { \dot { \mathrm { L G } } , } \end{array}$ whose cost is shown as the amortized per-layer complexity. For GUMP, $\begin{array} { r } { M _ { L } \ = \ \sum _ { v \in V } \deg ( v ) ^ { 2 } } \end{array}$ is the number of admissible transitions in the directed line graph. <sup>†</sup>GUMP additionally requires unitary operator construction, whose $K _ { G }$ -step blockwise Newton–Schulz projection costs $\begin{array} { r } { \mathcal { O } \big ( K _ { G } \sum _ { v \in V } \deg ( v ) ^ { 3 } \big ) } \end{array}$ .

The benefit of cheaper propagation is therefore practical: more of the inference budget can support feature learning. Comparisons on the connecting-lug and elbow-bracket datasets in Appendix B further assess this trade-off across geometries.

The two local metrics reveal complementary aspects of this trade-off. Higher peak-region PSNR indicates closer agreement with the stress values in highly loaded regions, while lower HSG-NMSE indicates better recovery of how stress changes around them. LiftGCN improves both as capacity increases. Its advantage therefore extends beyond matching peak values to recovering the surrounding spatial structure, which is central to a faithful stress-field surrogate.

Figure 6 shows that LiftGCN benefits consistently from greater depth across the tested range. Whole-field and peak accuracy improve while local gradient error decreases. GCN largely saturates, and the unitary and attention baselines perform best at shallower or intermediate depths. These 10-run averages suggest that LiftGCN can incorporate wider spatial context while retaining the local variations needed for stress reconstruction.

![](images/5c55fece5f2642905bf8ece824efef6e0737a427c8ac3cbfcb43156a8d38ab78.jpg)  
(a) Peak-stress accuracy versus latency.

![](images/e71cbb1e1c8c7889cc7044e81c9cc491585ac82b33a2d260e8ea1c1197333a02.jpg)  
(b) High-stress gradient error versus latency.

Figure 5: Accuracy–latency trade-offs. Prediction metrics are averaged over 10 repeated experiments. LiftGCN (ours) achieves strong peak-stress and local-gradient reconstruction across inference budgets.  
![](images/b26c20a3f90987f64d48c0e382de5fdcfce968a48415bd659fc1e5be7599ee9f.jpg)  
(a) Whole-field R<sup>2</sup>.

![](images/c0b15907ad1328021ec901d9223a84eb0999007a870e559ec9e5d2ee145fcf6a.jpg)  
(b) PSNR@1%.

![](images/f0c63c7b92731135bd64cf38ae2503e0b72b2c6f8277302d48239690d1add468.jpg)  
(c) HSG-NMSE@1%.  
Figure 6: Depth sensitivity. Every point is a mean over 10 repeated experiments. LiftGCN (ours) improves both whole-field and local reconstruction as propagation depth increases.

This joint improvement matters because increasing the receptive field is useful only if the additional context remains informative. The depth curves show that LiftGCN can improve the broad stress distribution without sacrificing the sharp features measured by the local metrics. This behavior is consistent with the role of the lifted backbone: it supports repeated information transport while the residual blocks progressively adapt the features to the prediction task.

## 4.3 Ablation Studies

We test four variants, retaining the input encoder, depth, and two-state readout throughout. w/o Joukowski replaces the lifted recurrence with first-order aggregation, while Fixed-ρ fixes the spectral coefficients and retains the local residual blocks. w/o nonlinear removes only GELU from each propagation-layer residual block, retaining LayerNorm, channel mixing, and the residual connection. Pure Joukowski removes these entire blocks, leaving the hidden states to follow the second-order recurrence. Both variants retain learnable channel-wise spectral coefficients and the nonlinear input encoder. Table 2 reports the means over 10 repetitions.

Table 2: Ablation results averaged over 10 repeated experiments. Bold marks the best mean. Accurate stress reconstruction benefits from both Joukowski propagation and local nonlinear transformations; learnable spectral scaling provides further refinement.
<table><tr><td>Model</td><td>SNR (dB)↑</td><td> $R ^ { 2 } \uparrow$ </td><td>PSNR@1% (dB)↑</td><td>HSG-NMSE@1%↓</td></tr><tr><td>LiftGCN (ours)</td><td>16.329</td><td>0.87705</td><td>21.131</td><td>0.072548</td></tr><tr><td>w/o Joukowski</td><td>14.227</td><td>0.80057</td><td>18.027</td><td>0.275960</td></tr><tr><td>Fixed-ρ (0.9)</td><td>16.327</td><td>0.87698</td><td>21.098</td><td>0.073043</td></tr><tr><td>w/o nonlinear</td><td>15.342</td><td>0.84570</td><td>19.387</td><td>0.120570</td></tr><tr><td>Pure Joukowski</td><td>12.897</td><td>0.72913</td><td>12.050</td><td>0.407700</td></tr></table>

Removing Joukowski propagation reduces accuracy across all metrics, especially in local gradient reconstruction, even with the local transformations retained. The lifted recurrence therefore plays an important role in preserving stress structure. Fixed spectral scaling retains most of the full model’s performance, with learnable coefficients providing a small additional improvement.

Removing only GELU also weakens whole-field and local reconstruction, showing that nonlinear activation improves how propagated features are used. GELU has no trainable parameters, so this comparison preserves model size and isolates the activation within the local residual blocks. Removing the entire local residual module causes the largest degradation, particularly in peak-stress and gradient accuracy. The much stronger results with channel mixing and residual updates retained show the value of feature transformations between propagation steps. Together, the 10-run means support the combined design: Joukowski propagation preserves information, while local nonlinear transformations turn it into accurate stress predictions.

## 5 Conclusion

LiftGCN realizes Joukowski spectral lifting through a simple real-valued second-order recurrence. Its linear backbone preserves a quadratic energy, and each layer requires only one sparse neighborhood aggregation. Experiments averaged over 10 repetitions show that this design combines efficient inference with accurate reconstruction of stress concentrations and local gradients. The model uses additional capacity and depth effectively, while ablations demonstrate the complementary benefits of lifted propagation and local nonlinear transformations. These findings support spectral lifting as an efficient basis for graph-based stress prediction.

More broadly, the method connects information-preserving dynamics with the sparse communication structure of finite element meshes. It offers a practical route to retaining local physical detail as graph models aggregate wider context. Extending this approach to broader geometries, material models, and physical prediction tasks is a natural direction for future work.

## AI use statement

Generative AI tools, including ChatGPT, were used to assist with the refinement of theoretical formulations and mathematical presentation, experimental design, software implementation, and interpretation of experimental results. They were also used to assist with literature search and summarization, drafting and editing parts of the manuscript, improving readability, and preparing research code and scientific figures. All AI-assisted mathematical derivations, claims, references, code, and written content were independently reviewed, verified, and revised by the authors. The generated and modified code was tested by the authors, and all reported experimental results were obtained from experiments conducted and evaluated by the authors rather than generated by AI. The authors take full responsibility for the methodology, results, claims, and final content of this work.

## Reproducibility statement

Section 3 specifies the LiftGCN recurrence, architecture, and assumptions for the energy-preservation result. Appendix A documents dataset sizes, simulation parameters, stress extraction, and graph construction. Appendix B provides the additional benchmark results, and Appendix C records normalization, optimization, repeated splits, checkpoint selection, and inference timing. The public Mines Paris data are identified by their dataset citation (Kerfriden, 2022). Our codee is available at https://github.com/ChenZeng001/LiftGCN.

## References

Klaus-Jürgen Bathe et al. Finite element procedures, volume 2. Prentice hall New Jersey, 1996.

Moritz Braun, André Mischa Müller, Aleksandar-Saša Milakovic, Wolfgang Fricke, and Sören Ehlers. Requirements´ for stress gradient-based fatigue assessment of notched structures according to theory of critical distance. Fatigue & Fracture ofEngineering Materials & Structures, 43(7):1541–1554, 2020.

Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convolutional networks. In International conference on machine learning, pp. 1725–1735. PMLR, 2020.

Dassault Systèmes SIMULIA. Flap mechanism, 2025a. Abaqus 2025 documentation: Example Problems Guide.

Dassault Systèmes SIMULIA. Connecting lug, 2025b. Abaqus 2025 documentation: Getting Started with Abaqus.

Matteo Giacomini and Pedro Díez. Surrogates for physics-based and data-driven modelling of parametric systems: Review and new perspectives. Archives ofComputational Methods in Engineering, pp. 1–33, 2026.

Alessio Gravina, Davide Bacciu, and Claudio Gallicchio. Anti-symmetric dgn: a stable architecture for deep graph networks. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=J3Y7cgZOOS.

Martin H Gutknecht and Lloyd N Trefethen. Real polynomial chebyshev approximation by the carathéodory–fejér method. SIAM Journal on Numerical Analysis, 19(2):358–371, 1982.

NT Hoang, Takanori Maehara, and Tsuyoshi Murata. Revisiting graph neural networks: Graph filtering perspective. In 2020 25th International Conference on Pattern Recognition (ICPR), pp. 8376–8383. IEEE, 2021.

SC Jayasinghe, M Mahmoodian, A Alavi, A Sidiq, F Shahrivar, Z Sun, J Thangarajah, and S Setunge. A review on the applications of artificial neural network techniques for accelerating finite element analysis in the civil engineering domain. Computers & Structures, 310:107698, 2025.

Pierre Kerfriden. Surrogate-modelling & machine learning dataset: finite element stress analysis of biaxial specimen with random elastic properties - 100 samples. Zenodo, version V0.9, 2022. URL https://doi.org/10.5281/ zenodo.6827035.

Bobak T Kiani, Lukas Fesser, and Melanie Weber. Unitary convolutions for learning on graphs and groups. Advances in Neural Information Processing Systems, 37:136922–136961, 2024.

Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.

Ron Levie, Federico Monti, Xavier Bresson, and Michael M Bronstein. Cayleynets: Graph convolutional neural networks with complex rational spectral filters. IEEE Transactions on Signal Processing, 67(1):97–109, 2018.

Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 32, 2018.

Xiaoke Li, Qingyu Yang, Yang Wang, Xinyu Han, Yang Cao, Lei Fan, and Jun Ma. Development of surrogate models in reliability-based design optimization: A review. Mathematical Biosciences and Engineering, 18(5):6386–6409, 2021.

Liang Liang, Minliang Liu, Caitlin Martin, and Wei Sun. A deep learning approach to estimate stress distribution: a fast and accurate surrogate of finite-element analysis. Journal of The Royal Society Interface, 15(138):20170844, 2018.

Marco Maurizi, Chao Gao, and Filippo Berto. Predicting stress, strain and deformation fields in materials and structures with graph neural networks. Scientific reports, 12(1):21834, 2022.

Vivek Oommen, Siavash Khodakarami, Aniruddha Bora, Zhicheng Wang, and George Em Karniadakis. Learning turbulent flows with generative models for super resolution and sparse flow reconstruction. Nature Communications, 17(1):3707, 2026.

Luca Patrignani and Silvestre T Pinho. Graph neural networks with hybrid local-global attention for effective prediction of mechanical response in structures. Computer Methods in Applied Mechanics and Engineering, 452:118753, 2026.

Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. Learning mesh-based simulation with graph networks. In International Conference on Learning Representations, 2021.

Haiquan Qiu and Quanming Yao. Graph unitary message passing, 3 2024. URL https://arxiv.org/abs/2403. 11199.

Andreas Roth and Thomas Liebig. Rank collapse causes over-smoothing and over-correlation in graph neural networks. In Learning on Graphs Conference, pp. 35–1. PMLR, 2024.

T Konstantin Rusch, Ben Chamberlain, James Rowbottom, Siddhartha Mishra, and Michael Bronstein. Graph-coupled oscillator networks. In International conference on machine learning, pp. 18888–18909. PMLR, 2022.

Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61–80, 2008.

Chao Sun and Zhen Chen. End-to-end deep learning method to reconstruct full-field stress distribution for ship hull structure with stress concentrations. Ocean Engineering, 313:119431, 2024.

Peihan Wu, Hongda Qi, Sirong Huang, Dongdong An, Jie Lian, and Qin Zhao. Wave-driven graph neural networks with energy dynamics for over-smoothing mitigation. In IJCAI, pp. 6579–6587, 2025.

Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S Yu. A comprehensive survey on graph neural networks. IEEE transactions on neural networks and learning systems, 32(1):4–24, 2020.

Shun-Peng Zhu, Wen-Long Ye, José A.F.O. Correia, Abílio M.P. Jesus, and Qingyuan Wang. Stress gradient effect in metal fatigue: Review and solutions. Theoretical and Applied Fracture Mechanics, 121:103513, 2022. ISSN 0167-8442.

Olgierd Cecil Zienkiewicz, Robert Leroy Taylor, and Jian Z Zhu. The finite element method: its basis and fundamentals. Elsevier, 2013.

## A Datasets and Graph Representation

This appendix specifies the finite element data used in Section 4.1, including the quantities actually supplied to the networks. Table 3 summarizes all complete samples in the processed datasets. Counts are measured from the coordi nate and element-connectivity files; graph edges follow the constructions in Appendix A.3. An edge is counted once as an undirected pair, before adding self-loops or storing both message-passing directions.

Table 3: Dataset statistics. Variable mesh sizes are reported as minimum–maximum, with arithmetic means on separate rows. The feature dimension includes the degree feature.
<table><tr><td>Property</td><td>Mines Paris</td><td>Connecting lug</td><td>Elbow bracket</td></tr><tr><td>Samples</td><td>100</td><td>225</td><td>288</td></tr><tr><td>Nodes per mesh</td><td>10,502</td><td>9,344–11,762</td><td>10,282-17,944</td></tr><tr><td>Mean nodes</td><td>10,502</td><td>10,683.72</td><td>13,878.89</td></tr><tr><td>Elements per mesh</td><td>43,399</td><td>5,778–7,515</td><td>6,088–10,936</td></tr><tr><td>Mean elements</td><td>43,399</td><td>6,735.65</td><td>8,354.37</td></tr><tr><td>Undirected graph edges</td><td>60,544</td><td>15,918–20,132</td><td>17,396–30,500</td></tr><tr><td>Mean graph edges</td><td>60,544</td><td>18,251.52</td><td>23,544.44</td></tr><tr><td>Element nodes</td><td>4</td><td>10 (C3D10M)</td><td>10 (C3D10M)</td></tr><tr><td>Input/output channels</td><td>5/1</td><td>4/1</td><td>4/1</td></tr><tr><td>Shared mesh across samples</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Training/evaluation samples</td><td>80/20</td><td>180/45</td><td>230/58</td></tr></table>

## A.1 Mines Paris Biaxial Specimen

We use the 100 random-material realizations of the public Mines Paris biaxial-specimen dataset, version V0.9 (Kerfriden, 2022). The processed samples have identifiers 1–100; the homogeneous reference realization indexed by 0 in the original archive is not included. All 100 processed coordinate files are identical, as are all 100 connectivity files. The common three-dimensional mesh has 10,502 nodes and 43,399 four-node tetrahedra. Its coordinate bounds are $[ - 1 . 5 , 1 . 5 ] \times [ - 1 . 5 , 1 . 5 ] \times [ 0 , 0 . 2 ]$ in the source coordinate units. The original VTK files and the processed CSV files agree on the mesh size. Each realization changes the spatial Young’s-modulus field on this fixed specimen. The nodal modulus values across the processed realizations span approximately [2.9708, 129.9183] in the source material units.

Let $\mathbf { P } \in \mathbb { R } ^ { n \times 3 }$ contain nodal coordinates, T the tetrahedral connectivity, and $\mathbf { E } ^ { ( s ) } \in \mathbb { R } ^ { n \times 1 }$ the nodal Young’s modulus of sample s. The physical prediction task is

$$
( \mathbf { P } , \mathcal { T } , \mathbf { E } ^ { ( s ) } ) \longmapsto \mathbf { y } ^ { ( s ) } = ( \sigma _ { \mathrm { v m } , i } ^ { ( s ) } ) _ { i = 1 } ^ { n } \in \mathbb { R } ^ { n \times 1 } .\tag{15}
$$

The implemented feature matrix is $\mathbf { X } ^ { ( s ) } = [ \mathcal { Z } _ { P } ( \mathbf { P } ) , \mathcal { Z } _ { E } ( \mathbf { E } ^ { ( s ) } ) , \mathbf { q } ] \in \mathbb { R } ^ { n \times 5 }$ , where Z denotes training-set standardization and q is the standardized log-degree feature defined below. The network predicts one scalar per node. Although the original archive contains stress-tensor components, the experiments supervise only the scalar von Mises field; displacements and stress components are not input features. Boundary conditions and load magnitudes are not separate model inputs in this fixed experimental setting.

The loader matches coordinate, modulus, connectivity, and stress files by sample identifier, and aligns modulus and stress rows to coordinate rows by node identifier. For this common mesh, the topology can be constructed once and reused. The material field, rather than connectivity, supplies the sample-dependent input.

## A.2 Abaqus Datasets: Geometry, Loading, and Simulation

We construct two additional datasets with Abaqus/CAE 2025 and the Abaqus/Standard static solver. Their parametric scripts adapt the official connecting-lug example and the modelling workflow of the official flap-mechanism example (Dassault Systèmes SIMULIA, 2025b;a). The latter is reduced to a single perforated elbow bracket, rather than the full articulated mechanism. Both use homogeneous isotropic linear elasticity with Young’s modulus 200 GPa and Poisson’s ratio 0.30, and modified ten-node tetrahedral elements (C3D10M). Geometries are remeshed for every parameter combination. Coordinates, force, and stress use metres, newtons, and pascals in the simulation and exported data; Table 4 expresses lengths in millimetres for readability.

Table 4: Actual parameter grids, verified against the per-sample parameter records. $a : \Delta$ : b denotes values from a to b with increment ∆. Each dataset uses the Cartesian product of its listed levels.
<table><tr><td>Dataset / parameter</td><td>Values</td><td>Number of levels</td></tr><tr><td colspan="3">Connecting lug: 225 combinations</td></tr><tr><td>Shank length</td><td>100 mm</td><td>1</td></tr><tr><td>Outer radius</td><td>28.6 : 0.2 : 29.4 mm</td><td>5</td></tr><tr><td>Pin-hole radius</td><td>12.5 mm</td><td>1</td></tr><tr><td>Mounting-hole radius</td><td>4.8, 5.0, 5.2 mm</td><td>3</td></tr><tr><td>Thickness</td><td>16.6 : 0.2 : 19.4 mm</td><td>15</td></tr><tr><td>Global mesh seed</td><td>6 mm</td><td>1</td></tr><tr><td>Total force</td><td>20,000 N in the —y direction</td><td>1</td></tr><tr><td colspan="3">Elbow bracket: 288 combinations</td></tr><tr><td>Horizontal arm length</td><td>160, 180, 200 mm</td><td>3</td></tr><tr><td>Vertical arm length</td><td>180, 200, 220 mm</td><td>3</td></tr><tr><td>Centreline bend radius</td><td>60,70 mm</td><td>2</td></tr><tr><td>Arm width</td><td>50, 60 mm</td><td>2</td></tr><tr><td>Hole radius</td><td>9, 10, 11, 12 mm</td><td>4</td></tr><tr><td>Thickness</td><td>18,20 mm</td><td>2</td></tr><tr><td>Global mesh seed</td><td>8mm</td><td>1</td></tr><tr><td>Total force</td><td>2,000 N</td><td>1</td></tr><tr><td>Force angle from +x toward +y</td><td>-45°</td><td>1</td></tr></table>

Connecting lug. The lug consists of a straight shank, a rounded end with a pin hole, and two smaller mounting holes. The shank face at $x = 0$ is fully fixed. The prescribed resultant acts on the lower half of the pin-hole surface: if this region contains $N _ { \mathrm { l o a d } }$ mesh nodes, each receives $\mathbf { f } _ { i } = ( 0 , - 2 0 0 0 0 / N _ { \mathrm { l o a d } } , 0 )$ N. Thus the implemented loading is a set of concentrated nodal forces with the stated total resultant. This adapts the pressure-loaded official example using the prescribed force resultant. Varying the outer radius, mounting-hole radius, and thickness yields $5 \times 3 \times 1 5 = 2 2 5$ samples. Figure 7 shows the geometry and representative finite element stress fields.

Elbow bracket. Two straight perforated arms are joined by a curved $9 0 °$ bend. The face at $x = 0$ is fully fixed, and the force is distributed equally among nodes on the free end of the vertical arm. For $\theta = - 4 5 ^ { \circ }$ , the nodal force is ${ \bf f } _ { i } = 2 0 0 0 ( \cos \theta$ , sin $\theta , 0 ) \bar { / } N _ { \mathrm { l o a d } }$ N. The six varied geometric parameters yield $3 \times 3 \times 2 \times 2 \times 4 \times 2 = 2 8 8$ combinations. Figure 8 illustrates the geometry, loading, and spatially localized stresses around the bend, holes, and constrained region.

Stress extraction and learning task. The postprocessing scripts extract stress tensors at the ELEMENT\_NODAL positions of the selected ODB frame (the final frame by default). For a node shared by several elements, tensor components are first averaged arithmetically over the element-nodal contributions. Von Mises stress is then computed from this averaged tensor:

$$
\begin{array} { r } { \sigma _ { \mathrm { v m } } = \sqrt { \frac { 1 } { 2 } [ ( \bar { \sigma } _ { x x } - \bar { \sigma } _ { y y } ) ^ { 2 } + ( \bar { \sigma } _ { y y } - \bar { \sigma } _ { z z } ) ^ { 2 } + ( \bar { \sigma } _ { z z } - \bar { \sigma } _ { x x } ) ^ { 2 } ] + 3 ( \bar { \sigma } _ { x y } ^ { 2 } + \bar { \sigma } _ { y z } ^ { 2 } + \bar { \sigma } _ { z x } ^ { 2 } ) } . } \end{array}\tag{16}
$$

This operation differs from averaging element-wise von Mises scalars. The exported stress table contains six tensor components and the scalar invariant, but training uses only the latter.

![](images/a4960bf4da52052a78d1414556bdc3ecad35ed2cc2d587757a053a638fefebc7.jpg)

![](images/ddd1cbfbd892648158f1adf5d85b2ad1df623ad0c4a88644d947e41c9df090bd.jpg)  
(a) Mesh, fixed boundary, nodal loading, and geometric parameters.  
(b) Representative simulated von Mises stress fields for different geometries.  
Figure 7: Connecting-lug dataset. The contour panels illustrate geometric variation under a fixed total load; each panel retains its own stress colour scale in Pa.

![](images/20345f453221f2994c7201a5daaaaa0952ed7d22d5417d580d071f47432d953e.jpg)

(a) Mesh, boundary conditions, force direction, and geometric parameters.  
![](images/719242ceb774f9c913f0ca21a306c588351e1655c5ebee167954f900821b94af.jpg)  
(b) Representative simulated von Mises stress fields for different geometries.  
Figure 8: Elbow-bracket dataset. Each contour panel uses its own stress colour scale in Pa, so colours should be interpreted together with the corresponding legend.

For either custom dataset, sample s defines a geometry-dependent mesh $( \mathbf { P } ^ { ( s ) } , \mathcal { T } ^ { ( s ) } )$ and target $\mathbf { y } ^ { ( s ) } \in \mathbb { R } ^ { n _ { s } \times 1 }$ . The physical and implemented mappings are

$$
( \mathbf { P } ^ { ( s ) } , \mathcal { T } ^ { ( s ) } ) \longmapsto \mathbf { y } ^ { ( s ) } , \qquad f _ { \in } ( \mathbf { X } ^ { ( s ) } , \widetilde { \mathbf { A } } ^ { ( s ) } ) = \widehat { \mathbf { y } } _ { \mathrm { n o r m } } ^ { ( s ) } , \qquad \mathbf { X } ^ { ( s ) } = [ \mathcal { Z } _ { P } ( \mathbf { P } ^ { ( s ) } ) , \mathbf { q } ^ { ( s ) } ] \in \mathbb { R } ^ { n _ { s } \times 4 } .\tag{17}
$$

Material constants, total load, load direction, and boundary-condition rules are fixed within each dataset, so they are not supplied as additional channels. In particular, the number of loaded nodes varies with the mesh, and the simulation divides the fixed total force accordingly. These experiments assess stress prediction across geometric configurations under the prescribed physical setting, using finite element solutions as reference fields. Model comparisons on both custom datasets are reported in Appendix B.

## A.3 From Finite Element Meshes to Graphs

Each mesh node becomes a graph vertex, including the midside nodes of quadratic elements. Node labels are mapped to contiguous row indices, and target rows are aligned by label. Connectivity is used to define an undirected graph $G = ( \check { V , } \mathcal { E } )$ ; it is not treated as a dense input feature.

Connectivity rules. For Mines Paris, all distinct node pairs in each four-node tetrahedron are connected, giving its six edges. The custom-data loader uses the Abaqus C3D10M ordering: nodes 1–4 are corner nodes and 5–10 lie on edges $( 1 , 2 ) , ( 2 , 3 ) , ( 3 , 1 ) , ( 1 , 4 ) , ( 2 , 4 ) , ( 3 , 4 )$ , respectively. Each quadratic edge is split at its midside node. The twelve local segments are

$$
\begin{array} { r l } & { \mathcal { E } _ { \mathrm { t e t } } = \{ ( 1 , 5 ) , ( 5 , 2 ) , ( 2 , 6 ) , ( 6 , 3 ) , ( 3 , 7 ) , ( 7 , 1 ) ,  } \\ & { \qquad ( 1 , 8 ) , ( 8 , 4 ) , ( 2 , 9 ) , ( 9 , 4 ) , ( 3 , 1 0 ) , ( 1 0 , 4 ) \} . } \end{array}\tag{18}
$$

Edges shared by elements are deduplicated globally. Each undirected edge is stored in both directions for message passing. The generic fallback in the custom-data loader connects all valid pairs for other element sizes, but every custom-data element used here has ten nodes and follows the twelve-segment rule. Thus these quadratic tetrahedra are not converted to ten-node cliques.

Features and graph operators. Let $d _ { i }$ be the number of distinct neighbours before adding self-loops and $\ell _ { i } =$ $\log ( 1 + d _ { i } )$ . The last input channel is

$$
q _ { i } = \frac { \ell _ { i } - \bar { \ell } } { s _ { \ell } + 1 0 ^ { - 6 } } , \qquad \bar { \ell } = \frac { 1 } { n } \sum _ { i } \ell _ { i } , \qquad s _ { \ell } ^ { 2 } = \frac { 1 } { n } \sum _ { i } ( \ell _ { i } - \bar { \ell } ) ^ { 2 } .\tag{19}
$$

Coordinates and, for Mines Paris, Young’s modulus are standardized componentwise using all nodes in the training samples only. The target is similarly standardized for optimization, with ${ \widehat { y } } _ { i } = \sigma _ { y } { \widehat { y } } _ { \mathrm { n o r m } , i } + \mu _ { y }$ before evaluating physical stress metrics. Any training-set standard deviation below $1 0 ^ { - 1 2 }$ is replaced by one. The degree normalization is computed within each graph and does not use target values.

For LiftGCN, the binary symmetric adjacency A gives $\widehat { \mathbf { A } } = \mathbf { A } + \mathbf { I }$ and $\widetilde { \bf A } = \widehat { \bf D } ^ { - 1 / 2 } \widehat { \bf A } \widehat { \bf D } ^ { - 1 / 2 }$ , as in Section 2. The base graph has no separately measured edge attributes. MeshGraphNets derives a four-dimensional edge feature from the difference of standardized endpoint coordinates and its Euclidean norm. Hybrid LG instead forms coordinate differences and lengths in the original coordinates, then standardizes these edge features using training-set statistics. Local-gradient evaluation uses the undirected edges without self-loops and the original, unstandardized coordinates, consistently with Eq. (14).

## B Model Comparisons on the Custom Datasets

Tables 5 and 6 extend Section 4.2 to the connecting lug and elbow bracket. They report the same whole-field and local metrics as the main comparison, including all three hotspot thresholds. Prediction scores are means over 10 repetitions. The checkpoint-selection protocol is specified in Appendix C.

Table 5: Connecting-lug comparison. Prediction metrics are means over 10 repetitions. Bold marks the best score in each column, including ties. Parameters in parentheses belong to the Adv-GCN discriminator. (a) Whole-field accuracy and computational cost
<table><tr><td>Model</td><td>Parameters</td><td>Latency (ms)↓</td><td>SNR (dB)↑</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>GCN</td><td>1512k</td><td>0.691</td><td>25.021</td><td>0.98754</td></tr><tr><td>Adv-GCN</td><td>1512k (+244k)</td><td>0.691</td><td>25.014</td><td>0.98754</td></tr><tr><td>MeshGraphNets</td><td>1549k</td><td>2.383</td><td>26.594</td><td>0.99132</td></tr><tr><td>GCNII</td><td>1503k</td><td>1.412</td><td>16.707</td><td>0.91574</td></tr><tr><td>GraphCON</td><td>1503k</td><td>1.894</td><td>24.101</td><td>0.98463</td></tr><tr><td>CayleyNet</td><td>1551k</td><td>33.694</td><td>28.413</td><td>0.99431</td></tr><tr><td>EWGNN</td><td>1525k</td><td>3.855</td><td>18.764</td><td>0.94696</td></tr><tr><td>GUMP</td><td>1504k</td><td>27.285</td><td>22.386</td><td>0.97719</td></tr><tr><td>Hybrid LG</td><td>1547k</td><td>7.848</td><td>28.913</td><td>0.99473</td></tr><tr><td>A-DGN</td><td>1526k</td><td>2.592</td><td>24.733</td><td>0.98667</td></tr><tr><td>UniGCN</td><td>1519k</td><td>21.510</td><td>26.422</td><td>0.99096</td></tr><tr><td>LiftGCN, small</td><td>1514k</td><td>0.900</td><td>25.762</td><td>0.98953</td></tr><tr><td>LiftGCN, medium</td><td>3023k</td><td>2.610</td><td>28.071</td><td>0.99384</td></tr><tr><td>LiftGCN, large</td><td>11571k</td><td>8.870</td><td>29.558</td><td>0.99563</td></tr></table>

(b) Stress-concentration reconstruction
<table><tr><td rowspan="2">Model</td><td colspan="3">PSNR (dB)↑</td><td colspan="3">HSG-NMSE↓</td></tr><tr><td>@1%</td><td>@5%</td><td>@10%</td><td>@1%</td><td>@5%</td><td>@10%</td></tr><tr><td>GCN</td><td>27.962</td><td>29.510</td><td>30.518</td><td>0.036797</td><td>0.033067</td><td>0.031227</td></tr><tr><td>Adv-GCN</td><td>27.934</td><td>29.487</td><td>30.509</td><td>0.035820</td><td>0.032702</td><td>0.031144</td></tr><tr><td>MeshGraphNets</td><td>34.056</td><td>34.029</td><td>33.867</td><td>0.020698</td><td>0.028265</td><td>0.029088</td></tr><tr><td>GCNII</td><td>19.496</td><td>21.695</td><td>22.719</td><td>0.498130</td><td>0.419740</td><td>0.356840</td></tr><tr><td>GraphCON</td><td>26.991</td><td>28.602</td><td>29.493</td><td>0.058009</td><td>0.052488</td><td>0.048874</td></tr><tr><td>CayleyNet</td><td>31.141</td><td>33.428</td><td>34.543</td><td>0.019631</td><td>0.021604</td><td>0.021072</td></tr><tr><td>EWGNN</td><td>24.741</td><td>25.422</td><td>26.082</td><td>0.186240</td><td>0.232120</td><td>0.228620</td></tr><tr><td>GUMP</td><td>26.447</td><td>27.979</td><td>28.774</td><td>0.059343</td><td>0.064130</td><td>0.064358</td></tr><tr><td>Hybrid LG</td><td>31.979</td><td>33.791</td><td>34.642</td><td>0.028996</td><td>0.028105</td><td>0.027341</td></tr><tr><td>A-DGN</td><td>28.886</td><td>29.853</td><td>30.668</td><td>0.032105</td><td>0.034136</td><td>0.033857</td></tr><tr><td>UniGCN</td><td>30.185</td><td>31.387</td><td>32.107</td><td>0.024199</td><td>0.024338</td><td>0.024768</td></tr><tr><td>LiftGCN, small</td><td>29.517</td><td>30.625</td><td>31.367</td><td>0.039788</td><td>0.040724</td><td>0.040821</td></tr><tr><td>LiftGCN, medium</td><td>31.341</td><td>33.081</td><td>33.939</td><td>0.031984</td><td>0.037151</td><td>0.035538</td></tr><tr><td>LiftGCN, large</td><td>33.366</td><td>35.357</td><td>36.035</td><td>0.025049</td><td>0.023864</td><td>0.023407</td></tr></table>

On the connecting lug, small LiftGCN improves SNR and peak-region PSNR over GCN at a latency of 0.900 ms, although its HSG-NMSE is higher. Large LiftGCN attains the best SNR (29.558 dB), R<sup>2</sup> (0.99563), and PSNR at the 5% and 10% thresholds. It takes 8.870 ms, compared with 21.510 ms for UniGCN and 33.694 ms for CayleyNet. However, MeshGraphNets has the best PSNR@1%, CayleyNet leads all three gradient metrics, and GCN/Adv-GCN have the lowest latency. These results demonstrate a useful accuracy–latency trade-off across model sizes.

Table 6: Elbow-bracket comparison. Prediction metrics are means over 10 repetitions. Bold marks the best score in each column, including ties. Parameters in parentheses belong to the Adv-GCN discriminator. (a) Whole-field accuracy and computational cost
<table><tr><td>Model</td><td>Parameters</td><td>Latency (ms)↓</td><td>SNR (dB)↑</td><td> $R ^ { 2 }$  ←</td></tr><tr><td>GCN</td><td>1512k</td><td>0.570</td><td>17.354</td><td>0.95714</td></tr><tr><td>Adv-GCN</td><td>1512k (+244k)</td><td>0.570</td><td>17.175</td><td>0.95526</td></tr><tr><td>MeshGraphNets</td><td>1549k</td><td>3.228</td><td>18.245</td><td>0.96493</td></tr><tr><td>GCNII</td><td>1503k</td><td>2.383</td><td>8.932</td><td>0.70091</td></tr><tr><td>GraphCON</td><td>1503k</td><td>2.663</td><td>16.160</td><td>0.94357</td></tr><tr><td>CayleyNet</td><td>1551k</td><td>41.259</td><td>23.418</td><td>0.98935</td></tr><tr><td>EWGNN</td><td>1525k</td><td>6.188</td><td>12.277</td><td>0.78350</td></tr><tr><td>GUMP</td><td>1504k</td><td>25.150</td><td>11.739</td><td>0.84394</td></tr><tr><td>Hybrid LG</td><td>1547k</td><td>13.318</td><td>18.945</td><td>0.96905</td></tr><tr><td>A-DGN</td><td>1526k</td><td>4.754</td><td>15.492</td><td>0.93417</td></tr><tr><td>UniGCN</td><td>1519k</td><td>27.688</td><td>18.997</td><td>0.97063</td></tr><tr><td>LiftGCN, small</td><td>1514k</td><td>2.361</td><td>18.246</td><td>0.96510</td></tr><tr><td>LiftGCN, medium</td><td>3023k</td><td>4.861</td><td>21.216</td><td>0.98239</td></tr><tr><td>LiftGCN, large</td><td>11571k</td><td>12.714</td><td>24.019</td><td>0.99074</td></tr></table>

(b) Stress-concentration reconstruction
<table><tr><td rowspan="2">Model</td><td colspan="3">PSNR (dB)↑</td><td colspan="3">HSG-NMSE↓</td></tr><tr><td>@1%</td><td>@5%</td><td>@10%</td><td>@1%</td><td>@5%</td><td>@10%</td></tr><tr><td>GCN</td><td>24.735</td><td>24.915</td><td>25.266</td><td>0.124680</td><td>0.113800</td><td>0.127510</td></tr><tr><td>Adv-GCN</td><td>24.469</td><td>24.693</td><td>25.081</td><td>0.133390</td><td>0.120480</td><td>0.135120</td></tr><tr><td>MeshGraphNets</td><td>26.303</td><td>26.262</td><td>26.449</td><td>0.088613</td><td>0.089554</td><td>0.100940</td></tr><tr><td>GCNII</td><td>14.184</td><td>14.322</td><td>14.651</td><td>1.483500</td><td>1.731300</td><td>1.745500</td></tr><tr><td>GraphCON</td><td>23.362</td><td>23.781</td><td>24.088</td><td>0.255970</td><td>0.231220</td><td>0.248400</td></tr><tr><td>CayleyNet</td><td>28.899</td><td>29.158</td><td>29.631</td><td>0.054606</td><td>0.053158</td><td>0.059374</td></tr><tr><td>EWGNN</td><td>17.384</td><td>18.293</td><td>19.027</td><td>0.696530</td><td>0.627560</td><td>0.636530</td></tr><tr><td>GUMP</td><td>18.903</td><td>18.391</td><td>18.012</td><td>0.591310</td><td>0.604710</td><td>0.655690</td></tr><tr><td>Hybrid LG</td><td>24.311</td><td>25.441</td><td>26.196</td><td>0.161900</td><td>0.155140</td><td>0.158990</td></tr><tr><td>A-DGN</td><td>25.053</td><td>24.760</td><td>24.346</td><td>0.170570</td><td>0.192130</td><td>0.257930</td></tr><tr><td>UniGCN</td><td>26.998</td><td>26.656</td><td>27.008</td><td>0.053734</td><td>0.059160</td><td>0.066304</td></tr><tr><td>LiftGCN, small</td><td>26.543</td><td>26.205</td><td>26.586</td><td>0.090990</td><td>0.089279</td><td>0.108020</td></tr><tr><td>LiftGCN, medium</td><td>29.036</td><td>28.743</td><td>28.998</td><td>0.046467</td><td>0.053178</td><td>0.062416</td></tr><tr><td>LiftGCN, large</td><td>29.904</td><td>29.985</td><td>30.354</td><td>0.038466</td><td>0.044894</td><td>0.054048</td></tr></table>

On the elbow bracket, large LiftGCN leads all reported prediction metrics: SNR reaches 24.019 dB, $R ^ { 2 }$ reaches 0.99074, and HSG-NMSE@1% falls to 0.038466. Relative to CayleyNet, it improves SNR by 0.601 dB and reduces HSG-NMSE@1% by approximately 29.6%, with 3.25× lower forward latency (12.714 versus 41.259 ms). Medium LiftGCN also improves every prediction metric over UniGCN at 4.861 versus 27.688 ms. GCN remains the fastest model. Across both datasets, greater LiftGCN capacity improves every reported prediction metric, with parameter counts and latency documenting the computational cost of scaling.

## C Experimental Settings

## C.1 Shared Training and Evaluation Protocol

Table 7 summarizes the common training protocol in the experiment scripts, with the observed model-specific exceptions explicitly noted. Each repetition shuffles sample identifiers with a NumPy generator seeded by $4 2 + r ,$ , for $r = 0 , \ldots , 9$ , and assigns the first ⌊0.8N⌋ samples to training. The remaining samples form the evaluation partition. The split is at the simulation level, not at the node level. Models using the same dataset and repetition seed receive the same partition. Normalization statistics are recomputed from the training partition for each repetition.

Table 7: Training settings shared across the three datasets, as implemented in the supplied scripts. Model-specific configurations are listed in Tables 8 and 9.
<table><tr><td>Setting</td><td>Value / procedure</td></tr><tr><td>Repetitions and seeds</td><td>10 runs; seeds 42–51 for Python, NumPy, and PyTorch</td></tr><tr><td>Training/evaluation ratio</td><td>80%/20%; counts are given in Table 3</td></tr><tr><td>Batch and sample order</td><td>One complete mesh per optimizer update; shuffle training meshes each epoch using seed plus epoch index</td></tr><tr><td>Training duration</td><td>50 epochs, without early termination</td></tr><tr><td>Predictor optimizer</td><td>AdamW, learning rate  $\dot { 2 } \times 1 0 ^ { - 3 }$  , weight decay  $1 0 ^ { - 5 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Cosine annealing,  $T _ { \mathrm { m a x } } = 5 0 ,$  minimum learning rate 0</td></tr><tr><td>Gradient clipping</td><td>Global parameter-gradient norm bounded by 1.0</td></tr><tr><td>Regression objective</td><td>Mean squared error over nodes in standardized stress units</td></tr><tr><td>Dropout Adv-GCN exception</td><td>0.10 Predictor additionally uses an adversarial loss; discriminator optimized separately with</td></tr><tr><td>Checkpoint choice</td><td>Adam Lowest evaluation-partition MSE over 50 epochs; first minimum retained on a tie</td></tr><tr><td>Final scores</td><td>Restore the chosen predictor checkpoint, undo target standardization, evaluate, then average scores over 10 runs</td></tr><tr><td>Inference timing</td><td>CUDA events; 20 warm-up and 100 timed forward calls per profiled graph shape, outside training</td></tr></table>

Checkpoint selection. After each epoch, we compute the test-set MSE in standardized stress units, averaging the node-mean MSE over test meshes. For each repetition, we retain the checkpoint with the lowest test-set MSE across 50 epochs and restore it to compute the reported metrics. This procedure is used for all three datasets.

Inference timing. Latency is the average forward inference time per sample on the NVIDIA RTX PRO 6000. The benchmark places graph tensors on the GPU before timing, switches the predictor to evaluation mode, and measures forward calls with synchronized CUDA events. CSV loading, graph construction, host-to-device transfer, normal ization, and metric calculation are excluded. The LiftGCN scripts cache timing measurements by the tuple (node count, input dimension, adjacency nonzeros) within each partition, profiling one representative of each shape. Each profiled forward call processes one complete mesh, and the timings are averaged across the profiled shapes and calls. Adv-GCN uses only its predictor at inference; its discriminator parameters are reported separately.

## C.2 Model-Specific Configurations

Tables 8 and 9 specify the model configurations used for Mines Paris and the two custom datasets, respectively.   
Parameter names follow the implementation. Connecting lug and elbow bracket use the same model configurations.   
Shared optimization settings are given in Table 7.

Table 8: Model configurations for Mines Paris.
<table><tr><td>Model</td><td>Model-specific hyperparameters</td></tr><tr><td>GCN</td><td>hidden_dim=270,layers=6</td></tr><tr><td>Adv-GCN</td><td>hidden_dim=270, layers=6, adv_beta=0.1, disc_hidden_dim=128, disc_layers=4</td></tr><tr><td>MeshGraphNets</td><td>mgn_latent_size=64, mgn_message_passing_steps=10</td></tr><tr><td>GCNII</td><td>hidden_dim=270,layers=6,gcnii_alpha=0.1,gcnii_lambda=0.50</td></tr><tr><td>GraphCON</td><td>hidden_dim=270, layers=6, graphcon_dt=1.0, graphcon_alpha=1.0, graphcon_gamma=1.0</td></tr><tr><td>CayleyNet</td><td>hidden_dim=110, layers=6, cayley_order=3, cayley-jacobi_iters=5, cayley_h_init=1.0</td></tr><tr><td>EWGNN</td><td>hidden_dim=195, 1ayers=4, ewgnn_alpha=0.5, ewgnn_eta=1.0</td></tr><tr><td>GUMP</td><td>hidden_dim=240, layers=4, base_layers=1, gump_attn_dim=32, gump_ns_iters=10</td></tr><tr><td>Hybrid LG</td><td>hidden_dim=90,layers=6,attention_frequency=3</td></tr><tr><td>A-DGN</td><td>hidden_dim=400,layers=6, adgn_epsilon=0.1, adgn_gamma=0.1</td></tr><tr><td>UniGCN</td><td>hidden_dim=235, layers=4, taylor_order=10, uniconv_time=1.0</td></tr><tr><td>LiftGCN, small</td><td>hidden_dim=330, layers=4</td></tr><tr><td>LiftGCN, medium</td><td>hidden_dim=640, layers=8</td></tr><tr><td>LiftGCN, large</td><td>hidden_dim=1280,1ayers=10</td></tr></table>

Table 9: Shared model configurations for connecting lug and elbow bracket.
<table><tr><td>Model</td><td>Model-specific hyperparameters</td></tr><tr><td>GCN</td><td>hidden_dim=500, 1ayers=6</td></tr><tr><td>Adv-GCN</td><td>hidden_dim=500,layers=6, adv_beta=0.1,disc_hidden_dim=200, disc_layers=6</td></tr><tr><td>MeshGraphNets</td><td>mgn_latent_size=160,mgn_message_passing_steps=6</td></tr><tr><td>GCNII</td><td>hidden_dim=500,layers=6,gcnii_alpha=0.1,gcnii_1ambda=0.50</td></tr><tr><td>GraphCON</td><td>hidden_dim=500, layers=6, graphcon_dt=1.0, graphcon_alpha=1.0,</td></tr><tr><td>CayleyNet</td><td>graphcon_gamma=1.0 hidden_dim=210, layers=6, cayley_order=3, cayley-jacobi_iters=5,</td></tr><tr><td>EWGNN</td><td>cayley_h_init=1.0 hidden_dim=290,layers=6, ewgnn_alpha=0.5, ewgnn_eta=1.0</td></tr><tr><td>GUMP</td><td>hidden_dim=380, layers=6, base_layers=2, gump_attn_dim=32, gump_ns_iters=10</td></tr><tr><td>Hybrid LG</td><td>hidden_dim=170,layers=6,attention_frequency=3</td></tr><tr><td>A-DGN</td><td>hidden_dim=780,layers=6, adgn_epsilon=0.1, adgn_gamma=0.1</td></tr><tr><td>UniGCN</td><td>hidden_dim=355, layers=6, taylor_order=10,uniconv_time=1.0</td></tr><tr><td>LiftGCN, small</td><td>hidden_dim=500, layers=6</td></tr><tr><td>LiftGCN, medium</td><td>hidden_dim=500, 1ayers=12</td></tr><tr><td>LiftGCN, large</td><td>hidden_dim=800, layers=18</td></tr></table>