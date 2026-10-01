# Cluster Attention Neural Operators for Solving Parametric Partial Diferential Equations

Ming Zhong<sup>1,2,3</sup>, Antonio Colanera<sup>3</sup>, Gianluigi Rozza<sup>3</sup>, Zhenya Yan<sup>4,2,1</sup>

<sup>1</sup>School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences, Beijing 100049, China.

<sup>2</sup>State Key Laboratory of Mathematical Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences, Beijing 100190, China.

<sup>3</sup>International School for Advanced Studies (SISSA), Trieste 34136, Italy.

<sup>4</sup>School of Mathematics and Information Science, Zhongyuan University of Technology, Zhengzhou 450007, China.

Corresponding author: zyyan@mmrc.iss.ac.cn (Z. Yan);

## Abstract

Traditional simulations of parametric partial diferential equations (PDEs) rely on repetitive computations for each parameter, which makes high-fidelity design impractical. Neural operators address this issue by learning solution operators, accelerating parameter-space mapping by orders of magnitude. Recent Transformer-based neural operators attempt to capture global dependencies, but often at the cost of quadratic attention complexity. Transolver resolves this problem by projecting physical states into a reduced slice space for attention computation. Although fast, this projection sacrifices fine spatial information. Moreover, by operating in this reduced space with shared weights across attention heads, it may constrain the model’s flexibility, thereby limiting its capacity to capture complex phenomena. To address these issues, we propose the Cluster Attention Neural Operator (CANO), which reformulates attention via a novel cross-attention mechanism that dynamically clusters queries while preserving full-resolution keys and values. This avoids slice compression loss and removes weight-sharing limits. At the same time, the model remains fast without losing global interactions. Empirically, CANO achieves state-of-the-art performance across canonical PDE benchmarks, covering fluid and solid dynamics (e.g., Navier-Stokes, Airfoil, Plasticity), irregular unstructured geometries (e.g., Pipe

Turbulence, Composites), and long-term temporal rollouts. Across solid deformation and turbulent flow benchmarks, CANO achieves lower errors than baselines and exhibits strong geometric adaptability and temporal consistency.

Keywords: Parametric partial diferential equation, Deep learning, Cluster attention neural operator, Transformers

## 1 Introduction

Partial diferential equations (PDEs) play a pivotal role in modeling physical phenomena and have became ubiquitous across science and engineering [1–5]. While these PDEs provide a theoretical framework for understanding natural phenomena, yet solving these equations numerically is often too slow for practical use [6–10]. Traditional numerical solvers, such as the finite element, finite diference, and spectral methods, typically rely on dense spatial discretizations [6–10]. However, the computational cost of standard methods grows steeply with grid resolution. As a result, high-fidelity simulations are often computationally infeasible. This leaves a significant gap between theoretical models and real-world engineering [11, 12].

Recently, scientific machine learning (SciML) has emerged as a promising alternative, replacing classical iterative solvers with trained neural networks [13–16]. For example, the Deep Ritz method [17] solves PDEs via variational formulations, while Physics-Informed Neural Networks (PINNs) [18–20] penalize PDE residuals in the loss function. Other variants, such as PeRCNN [21], enforce physical laws directly through network architectures. However, these methods only solve one PDE instance at a time. Once the physical parameters or boundary conditions changed, the network must be re-trained from scratch, making it impractical for real-time applications[22, 23].

To address this issues, neural operators (NOs) [24–28] learn mappings between parameter or coeficient fields to solution fields. Instead of solving a single PDE instance, they approximate the underlying operator. Once trained, they provide zero-shot solution predictions for unseen configurations at a fraction of the classical computational cost.

Existing neural operator architectures can be broadly categorized into MLP-based, spectral-based, and transformer-based methods. First, MLP-based neural operators, represented by Deep Operator Networks (DeepONet) [26], are fundamentally grounded in the universal approximation theorem for operators [29]. This framework has been extended through various architectures, including POD-DeepONet [30], PINN-DeepONet [31], MIONet [32], and Hybrid-preconditioned DeepONet [33]. In practice, however, these simple architectures struggle to resolve complex nonlinear behaviors.

Second, spectral-based neural operators, such as the Fourier Neural Operator (FNO) [24, 25, 34, 35], exploit global convolutions in the wavenumbers domain to parameterize integral kernels, achieving high accuracy on structured grids. However, accommodating general unstructured domains significantly increases computational cost or requires lossy interpolations [34, 35].

Third, Transformer-based neural operators [36–41] ofer an alternative way to handle unstructured domains and irregular meshes. A standard self-attention block computes the output Y as [42–44]

$$
\mathbf { Y } = \mathrm { S o f t m a x } \Big ( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } \Big ) \mathbf { V } ,\tag{1}
$$

where Q, K, $\mathbf { V } \in \mathbb { R } ^ { N \times d }$ are the query, key, and value matrices, respectively, obtained from the projection of original features (such as function values or coordinates), where N is the number of tokens and $d$ is the feature dimension. While this formulation captures long-range dependencies, it scales quadratically, $\mathcal { O } ( N ^ { 2 } d )$ , making it impractical for fine physical discretizations. Linear attention variants reduce complexity to $\mathcal { O } ( N d ^ { 2 } )$ through matrix factorization [45–49]:

$$
\mathbf { Y } = \phi ( \mathbf { Q } ) \left( \phi ( \mathbf { K } ) ^ { \top } \mathbf { V } \right) ,\tag{2}
$$

where $\phi ( \cdot )$ denotes a feature mapping. Despite its eficiency, linear attention often sufers from overly smooth attention maps. While hierarchical or convolutional designs [50–54] mitigate this issue, they are strictly tailored for structured grids and do not readily apply to unstructured meshes.

Another line of work tackles complex geometries by introducing latent tokens [55– 57]. A notable example is Transolver [57], which learns a weight matrix to aggregate input features into a compact ”slice space”, performs attention within this reduced representation, and projects back. However, compressing physical points into slices loses local geometric details. In addition, Transolver computes attention only in this slice space and shares weights across attention heads, which limits its ability to capture complex dynamics.

To overcome these challenges, we introduce the Cluster Attention Neural Operator (CANO)<sup>1</sup>. Rather than projecting points into a reduced slice space, CANO computes cross-attention by clustering queries (Q) dynamically, while retaining keys (K) and values (V) in their original coordinate space. By clustering queries dynamically, CANO avoids compression loss in the slice space and removes the need for weight sharing across attention heads. As a result, the model maintains global receptive fields while scaling linearly with the number of points $N _ { ; }$ , yielding strong generalization on complex physical systems.

The main contributions of this work are twofold. First, we introduce Cluster Attention, which clusters queries while keeping keys and values at full resolution. This formulation prevents detail loss and scales linearly with the number of points N. Second, we validate CANO on 12 diverse PDE benchmarks, including complex geometries and long rollout horizons, where it achieves up to 41.67% lower error on solid mechanics and 33.73% on turbulent flows compared to previous models.

The remainder of this paper is organized as follows. Section 2 briefly introduces the neural operator framework and related works. Section 3 presents the proposed CANO architecture. Section 4 reports extensive numerical experimental results on diverse benchmarks and model analysis, followed by concluding remarks in Section 5.

## 2 Related works

In scientific machine learning, neural operators aim to learn the mappings between parameter spaces and corresponding PDE solution spaces [25–28]. Rather than replacing classical solvers, these models act as fast surrogates that accelerate inference.

From a mathematical perspective, we consider a generalized PDE formulated over a bounded spatial domain $\Omega \subset \mathbb { R } ^ { d _ { x } }$ . Consider a physical system parameterized by an input field $a ( x )$ , which can represent coeficients, initial/boundary conditions, or source terms. The governing equation can be written as

$$
\left\{ \begin{array} { r l } { ( L _ { a } u ) ( x ) = f ( x ) , } & { x \in \Omega \subset \mathbb { R } ^ { d _ { x } } , } \\ { u ( x ) = u _ { 0 } ( x ) , } & { x \in \partial \Omega , } \end{array} \right.\tag{3}
$$

where $f ( x )$ denotes the source term. Let A and U be Banach spaces containing the input parameter field a and the target solution $u : \Omega \to \mathbb { R }$ , respectively. Given a parametric diferential operator $\mathcal { L } _ { a } : \mathcal { U } \to \mathcal { U } ^ { * }$ , the goal of operator learning is to approximate the underlying solution operator $\mathcal { G } ^ { \dagger } : \mathcal { A }  \mathcal { U }$ such that $\mathcal { G } ^ { \dagger } ( a ) = u$

Given a training dataset of input–solution pairs $\{ ( \boldsymbol { a } ^ { ( i ) } , \boldsymbol { u } ^ { ( i ) } ) \} _ { i = 1 } ^ { M }$ , we approximate $\mathcal { G } ^ { \dagger }$ using a parameterized neural operator $\mathcal { G } _ { \theta } : \mathcal { A }  \mathcal { U }$ with weights $\theta \in \mathbb { R } ^ { p }$ . The optimal parameters $\theta ^ { * }$ are obtained by minimizing the empirical mean squared error in the Bochner norm [27]. In practice, the risk is approximated by the empirical risk over the training samples:

$$
\operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { p } } \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \Big \| u ^ { ( i ) } - \mathcal { G } _ { \theta } ( a ^ { ( i ) } ) \Big \| _ { \mathcal { U } } ^ { 2 } .\tag{4}
$$

Within this framework, neural operator architectures such as the Graph Kernel $\mathrm { N e t - }$ work (GKN) [61, 62] are closely motivated by Green’s functions. For a linear diferential operator ${ \mathcal { L } } _ { a }$ with Green’s function $G ( x , y )$ satisfying $\mathcal { L } _ { a } G ( x , \cdot ) = \delta _ { x }$ , the solution can be written as an integral convolution:

$$
u ( x ) = \int _ { \Omega } G ( x , y ) f ( y ) d y .\tag{5}
$$

Analogously, iterative neural operators update a continuous hidden representation $v _ { t } ( x )$ through non-local kernel integration:

$$
v _ { t + 1 } ( x ) = \sigma \left( W v _ { t } ( x ) + \int _ { \Omega } \kappa _ { \theta } \big ( x , y , a ( x ) , a ( y ) \big ) v _ { t } ( y ) \nu _ { x } ( d y ) \right) .\tag{6}
$$

Here, an initial lifting layer $P$ maps the local input $( x , a ( x ) )$ to the hidden state $v _ { 0 } ( x )$ = $P ( x , a ( x ) )$ ). Each update layer combines a local linear transformation parameterized by weight matrix W with global kernel aggregation via the parameterized kernel $\kappa _ { \theta }$ followed by an activation function $\sigma$ and integration measure $\nu _ { x }$ . After $T$ iterative updates, a projection layer $Q$ decodes the final hidden state to obtain the solution field $u ( x ) = Q ( v _ { T } ( x ) ) ,$ . This formulation allows the operator to capture both local features and long-range dependencies across the domain.

To realize the operator learning framework described above, three of the most representative architectures are: the Fourier Neural Operator (FNO) [24, 25], the Deep Operator Network (DeepONet) [26], and Transformer-based neural operators [36, 37, 57]. The FNO [24, 25] accelerates non-local integration by assuming a translationinvariant kernel, $\kappa ( x , y ) \ : = \ : \kappa ( x \ : - \ : y )$ . Under this assumption, the integral operator reduces to a spatial convolution, which can be evaluated eficiently in the frequency domain via the convolution theorem:

$$
v _ { t + 1 } ( x ) = \sigma \Big ( \mathcal { F } ^ { - 1 } \big ( \widehat { \kappa } _ { \theta } \cdot \mathcal { F } ( v _ { t } ) \big ) ( x ) + W v _ { t } ( x ) \Big ) ,\tag{7}
$$

where $\mathcal { F }$ and ${ \mathcal { F } } ^ { - 1 }$ denote the forward and inverse Fourier transforms, and $\widehat { \kappa } _ { \theta }$ is a learnable Fourier multiplier. Truncating the spectrum to the lowest $k _ { \mathrm { m a x } }$ modes allows FNO to run in O(N log N) time using the Fast Fourier Transform (FFT). However, because standard FFTs require uniform Cartesian grids and periodic boundaries, standard FNO is dificult to apply directly to complex, unstructured meshes.

An alternative approach is DeepONet [26, 32], which approximates the mapping $G ^ { \dagger }$ based on the universal approximation theorem for operators [29]. DeepONet decomposes the solution into a dot product of a branch network and a trunk network:

$$
G _ { \theta } ( a ) ( x ) = \sum _ { k = 1 } ^ { m } b _ { k } ( a ) t _ { k } ( x ) ,\tag{8}
$$

where $b _ { k } ( a )$ are outputs of the branch network encoding the input parameter function $^ { a , }$ and $t _ { k } ( x )$ are outputs of the trunk network encoding the spatial coordinates $x ,$ m is a pre-specified parameter. While DeepONet ofers significant mathematical generality, its performance is often constrained by the learning dynamics of its underlying multilayer perceptrons (MLPs). Specifically, these architectures exhibit spectral bias [63], meaning they eficiently learn low-frequency macroscopic trends but struggle to resolve the sharp, high-frequency oscillations that are critical for accurate PDE modeling.

To handle unstructured domains and capture complex long-range dependencies, attention mechanisms [42] have been increasingly adopted for PDE operators. Given a set of latent representations $\mathbf { X } \in \mathbb { R } ^ { N \times d }$ , standard transformer blocks project these into query, key, and value matrices:

$$
\mathbf { Q } = \mathbf { X } \mathbf { W } _ { Q } , \mathbf { K } = \mathbf { X } \mathbf { W } _ { K } , \mathbf { V } = \mathbf { X } \mathbf { W } _ { V } ,\tag{9}
$$

where ${ \bf W } _ { Q } , { \bf W } _ { K } , { \bf W } _ { V } \in \mathbb { R } ^ { d \times d }$ . The standard Softmax attention computes the output as displayed in $\operatorname { E q . } \ ( 1 )$ . While this formulation efectively models global pairwise interactions, it requires computing and storing the full $N \times N$ similarity matrix, with a quadratic computational cost of $\mathcal { O } ( N ^ { 2 } d )$ . Linear attention methods [45–48] reduce this complexity via a non-negative feature mapping function $\phi ( \cdot )$ to decompose the kernel. By leveraging the associativity of matrix multiplication, the operation is rewritten as displayed in Eq. (2). This rearrangement evaluates the $d \times d$ context matrix first, avoiding the $N \times N$ attention matrix and reducing the complexity to $\mathcal { O } ( N d ^ { 2 } )$ However, this factorization leads to low-rank, smooth attention maps, which limits the model’s ability to capture high-frequency details.

To avoid both the quadratic cost of softmax attention and the over-smoothing of linear attention, recent models compress physical tokens into a smaller set of latent representations. For example, Transolver [57] uses a physics-aware slicing mechanism to project the mesh into a reduced number of slice tokens. Specifically, given the input physical node features $\mathbf { X } \in \mathbb { R } ^ { N \times d }$ , Transolver learns a normalized assignment matrix $\textbf { \ i } \in \mathbb { R } ^ { N \times M }$ to partition the domain into M slices, where $M \ll N$ . The model aggregates the high-dimensional node features into a compact ”slice space” via a projection:

$$
\mathbf { S } = \mathbf { A } ^ { \top } \mathbf { X } , \qquad \mathbf { S } \in \mathbb { R } ^ { M \times d } .\tag{10}
$$

Standard Softmax attention is then executed exclusively on these M slice tokens to capture global interactions eficiently:

$$
\mathbf { S } ^ { \prime } = \operatorname { S o f t m a x } \Bigl ( \frac { ( \mathbf { S } \mathbf { W } _ { Q } ) ( \mathbf { S } \mathbf { W } _ { K } ) ^ { \top } } { \sqrt { d } } \Bigr ) ( \mathbf { S } \mathbf { W } _ { V } ) ,\tag{11}
$$

where $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V } \in \mathbb { R } ^ { d \times d }$ are the learnable projection matrices. Note that in the implementation of Transolver<sup>2</sup>, these matrices are defined as $\in \mathbb { R } ^ { d _ { \mathrm { h e a d } } \times d _ { \mathrm { h e a d } } }$ , forcing weight sharing across diferent attention heads. Here $d _ { \mathrm { h e a d } }$ denotes the embedding dimension per head, which is typically calculated as $d / H$ with H being the number of attention heads. Finally, a reverse projection (deslicing) is applied to broadcast the updated slice features back to the original N physical nodes:

$$
\mathbf { Y } = \mathbf { A } \mathbf { S } ^ { \prime } , \quad \mathbf { Y } \in \mathbb { R } ^ { N \times d } .\tag{12}
$$

Although this slice-and-broadcast design achieves O(NM d) linear complexity, it has notable limitations.

Specifically, constructing all queries, keys, and values from compressed slices discards local high-frequency details. In addition, Transolver shares attention weights across heads, which prevents individual heads from learning distinct physical patterns. To resolve these issues, our cluster attention mechanism preserves full-resolution keys and values, clustering only the queries dynamically. This allows unconstrained attention heads to capture physical interactions while maintaining linear $\mathcal { O } ( N )$ complexity.

## 3 Cluster Attention Neural Operator (CANO)

As discussed in the previous sections, existing latent compression methods like Transolver cut fine-grained physical structures by projecting the input X into a highly compressed slice space. In such symmetric compression process, all three matrices including Q, K, and V , sufer from slice compression, which limits the model’s performance on complex problems.

To resolve this issue, we propose the Cluster Attention Neural Operator (CANO). The key innovation of CANO is the asymmetric cross-attention mechanism. Instead of aggregating the raw input X, we selectively cluster only the query matrix Q, while preserving the keys K and values V at their full, uncompressed physical resolution. This achieves linear complexity $\mathcal { O } ( N )$ while retaining global context across the domain.

Specifically, given the input representation $\mathbf { X } \in \overset { \sim } { \mathbb { R } } ^ { N \times d }$ , we first project it into the standard query, key, and value spaces as done in Eq. (9). Unlike Transolver that learns a slice matrix from X, CANO derives a dynamic cluster assignment matrix $\mathbf { C } \in \mathbb { R } ^ { N \times K _ { c } }$ from the query matrix Q, where $K _ { c }$ is the predefined number of clusters $( K _ { c } \ll N )$ This matrix maps the N physical nodes to $K _ { c }$ cluster centroids, and it’s computed as

$$
\mathbf { C } = \operatorname { S o f t m a x } \big ( ( \mathbf { X } \mathbf { W } _ { Q } ) \mathbf { W } _ { c } \big ) ,\tag{13}
$$

where the Softmax operator is applied across the cluster dimension, and $\mathbf { W } _ { c } \in \mathbb { R } ^ { d \times K _ { c } }$ is the learnable cluster projection matrix. Using C, we aggregate the query matrix into a compact set of clustered queries $\mathbf { Q } _ { c }$

$$
\begin{array} { r } { \mathbf q _ { c } = \mathbf C ^ { \top } \mathbf q , \quad \mathbf Q _ { c } \in \mathbb { R } ^ { K _ { c } \times d } . } \end{array}\tag{14}
$$

With the clustered queries established, CANO executes a eficient cross-attention mechanism. The compact $\mathbf { Q } _ { c }$ attends to the full, uncompressed key matrix K to extract relevant features from the value matrix V:

$$
\mathbf { Z } _ { c } = \operatorname { S o f t m a x } \Bigl ( \frac { \mathbf { Q } _ { c } \mathbf { K } ^ { \top } } { \sqrt { d } } \Bigr ) \mathbf { V } , \quad \mathbf { Z } _ { c } \in \mathbb { R } ^ { K _ { c } \times d } .\tag{15}
$$

This asymmetric operation is fundamentally diferent from Transolver. By avoiding the compression of K and V, the attention matrix inherently retains physical details.

Finally, to reconstruct the updated representations for each individual physical node, we utilize the exact same assignment matrix C to dispatch (broadcast) the aggregated cluster features $\mathbf { Z } _ { c }$ back to the original spatial resolution:

$$
\mathbf { Y } = \mathbf { C } \mathbf { Z } _ { c } , \quad \mathbf { Y } \in \mathbb { R } ^ { N \times d } .\tag{16}
$$

A comparison of diferent attention mechanisms (including vanilla self-attention, Transolver, and CANO) can be found in Figure 1.

From a computational perspective, the calculation of the cross-attention matrix $\mathbf { Q } _ { c } \mathbf { K } ^ { \top }$ and the subsequent value aggregation require $\mathcal { O } ( N K _ { c } d )$ operations. Since $K _ { c } \ll$ N, CANO elegantly achieves linear complexity with respect to the sequence length N. Furthermore, because its mechanism, CANO inherently supports independent multihead attention computations without the need to share weights across heads, avoiding the detail loss seen in Transolver. The theoretical connection between CANO and generalized linear attention [64] can be seen from Appendix A. The detailed algorithm flow is given in the algorithm 1, where FFN(·) and LayerNorm(·) denote the feedforward network and layer normalization, respectively.

![](images/85c1d3191599b02ff21e8d83da1b8422fbc1c5501b7021c9cd74bb9a24e22b82.jpg)  
Fig. 1 Comparison of the architecture of (a) Vanilla Self-Attention, (b) Transolver, and (c) CANO mechanisms.

## 4 Numerical Experiment Results

In this section, we provide a comprehensive evaluation of the proposed CANO from four aspects. First, we compare CANO with state-of-the-art baselines on standard PDE benchmarks [25, 34]. We then assess its ability to handle complex geometries through experiments on irregular-domain benchmarks [65]. Next, we investigate its temporal stability in challenging long-horizon autoregressive rollout settings [25, 72]. Finally, we analyze the impact of key hyperparameters on the performance of CANO.

To validate the performances of our CANO approach, we compare it against diferent neural PDE solvers. This includes graph neural networks (e.g., GraphSAGE [66]), diferent neural operators, ranging from well known architectures like FNO [25] and DeepONet [26] to their variants (POD-DeepONet [30], Geo-FNO [34], F-FNO [67], U-FNO [68], and U-NO [69]), as well as recent innovations like LSM [70], NORM [65], and HPM [71]. Furthermore, we evaluate against state-of-the-art Transformer-based operators, including Galerkin [36], OFormer [37], FactFormer [38], GNOT [39], ONO [40], HT-Net [52], and Transolver [57]. As the strongest baseline, Transolver serves as our primary point of comparison throughout the experiments.

For most baseline methods, we report the performance values provided in the corresponding original publications or in the comprehensive benchmark of the Transolver study. To enable a controlled comparison of diferent attention mechanisms, we additionally re-implemented the Galerkin Transformer and Transolver within the same training framework adopted for CANO. Table 1 outlines the experimental settings and hyperparameter choices, including cluster numbers $( K _ { c } )$ and transformer configurations, used for each benchmark. These settings have been selected empirically to provide a suitable trade-of between predictive accuracy and computational cost. For Transolver, the MLP expansion ratio $( r _ { \mathrm { m l p } }$ in Table 1) was set to 2 for most benchmarks.

Algorithm 1 Cluster Attention Neural Operator (CANO) for solving PDEs   
Require: Spatial mesh coordinates $\mathbf { x } \in \mathbb { R } ^ { N \times d _ { \mathbf { x } } }$ , input function values $a ( \mathbf { x } ) \in \mathbb { R } ^ { N \times d _ { \mathbf { a } } }$   
ground truth $u _ { \mathrm { g t } } ( \mathbf { x } ) \in \mathbb { R } ^ { N \times d _ { \mathrm { u } } }$   
Ensure: Predicted solution field $\hat { u } ( \mathbf { x } ) \in \mathbb { R } ^ { N \times { d _ { \mathrm { u } } } }$   
1: Initialize network parameters W, base learning rate $\alpha _ { 0 }$ , total epochs $T _ { \mathrm { e p o c h s } } ;$   
2: for epoch = 1 to $T _ { \mathrm { e p o c h s } }$ do   
3: Input encoding:   
4: $\mathbf { h } ^ { ( 0 ) }  \mathrm { E n c o d e r } ( [ \mathbf { x } , a ( \mathbf { x } ) ] ) \in \mathbb { R } ^ { N \times d } ;$   
5: Cluster Attention Blocks:   
6: for $l = 0$ to $L - 1$ do   
7: $\widetilde { \mathbf { h } } ^ { ( l ) }  \mathbf { h } ^ { ( l ) } +$ Cluster-Attention(LayerNorm $( \mathbf { H } ^ { ( l ) } ) )$   
8: $\mathbf { h } ^ { ( l + 1 ) } \gets \widetilde { \mathbf { h } } ^ { ( l ) } + \mathrm { F F N } ( \mathrm { L a y e r N o r m } ( \widetilde { \mathbf { h } } ^ { ( l ) } ) ) ;$   
9: end for   
10: Output decoding:   
11: $\hat { u } ( \mathbf { x } ) \bar { \mathbf { \Psi } } \{ \mathbf { - } \operatorname { D e c o d e r } ( \mathbf { \bar { h } } ^ { ( L ) } ) \in \mathbb { R } ^ { N \times d _ { \mathbf { u } } }$   
12: Loss Computation:   
13: $\mathcal { L } _ { \mathrm { t r a i n } } \gets \frac { \| \hat { u } ( \mathbf { x } ) - u _ { \mathrm { g t } } ( \mathbf { x } ) \| _ { L ^ { 2 } } } { | \mathbf { \Omega } | } ;$   
$\| u _ { \mathrm { g t } } ( \mathbf { x } ) \| _ { L ^ { 2 } }$   
14: Backpropagate to compute parameter gradients $\nabla _ { \mathcal { W } } \mathcal { L } _ { \mathrm { t r a i n } } ;$   
15: Adjust current learning rate: $\alpha _ { \mathrm { e p o c h } }  \mathrm { S c h e d u l e } ( \mathrm { e p o c h } , \alpha _ { 0 } , T _ { \mathrm { e p o c h s } } ) ;$   
16: Update parameters via AdamW:   
17: $\mathcal { W }  \mathrm { A } \mathfrak { e }$ damW(W, ∇<sub>W</sub>L<sub>train</sub>, α<sub>epoch</sub>);   
18: end for   
19: return $\hat { u } ( { \bf x } )$

## 4.1 Standard PDE Benchmarks

To ensure a fair comparison against all baseline methods, we have conducted experiments on six standard PDE benchmarks: Airfoil, Pipe, Plasticity, Navier-Stokes, Darcy flow, and Elasticity, as detailed in Table 2.

This benchmark includes a diverse range of physical systems (e.g., fluid dynamics and solid mechanics), geometry types (regular grids, structured meshes, and point clouds), and input functions. The underlying physical equations and dataset details are introduced as in Appendix B.1.

The quantitative results reported in Table 3 demonstrate that CANO consistently outperforms the considered baselines across all six standard PDE benchmarks. Compared to Transolver, CANO consistently achieves lower relative $L ^ { 2 }$ errors across all cases. The largest improvements occur on the (2+1)-D Plasticity dataset, where the error drops by 41.67%, and on turbulent Navier–Stokes, with a 29.79% reduction. These gains confirm that preserving full-resolution keys and values helps resolve both large solid deformations and turbulent flow dynamics.

Table 1 Unified training and model hyperparameters across benchmarks. All experiments utilize an initial learning rate of $1 0 ^ { - 3 }$ . The model configuration is denoted as $\left( L / H / d / r _ { \mathrm { m l p } } \right)$ , representing layer, number of attention head, embedding dimension and MLP ratio in the transformer blocks.
<table><tr><td>Benchmark</td><td>Loss Function</td><td></td><td>Epochs Batch Cluster Number</td><td> $L / H / d / r _ { \mathrm { m l p } }$ </td></tr><tr><td colspan="6">Standard benchmarks</td></tr><tr><td>Airfoil</td><td>Rel.  $L ^ { 2 }$ </td><td>4</td><td>32</td><td>8 / 8 / 128 / 2</td><td></td></tr><tr><td>Pipe</td><td>Rel.  $L ^ { 2 }$ </td><td>500</td><td>16</td><td>8 / 4 / 128 / 2</td><td></td></tr><tr><td>Plasticity</td><td>Rel.  $L ^ { 2 }$ </td><td>500</td><td>64</td><td>8 / 8 / 128 / 2</td><td></td></tr><tr><td>Navier-Stokes</td><td>Rel.  $L ^ { 2 }$ </td><td>500 2</td><td>32</td><td>8 / 8 / 256 / 1</td><td></td></tr><tr><td>Darcy</td><td>Rel.  $L ^ { 2 } + 0 . 1 L _ { \nabla }$ </td><td>500</td><td>128</td><td>8 / 8 /  128 / 2</td><td></td></tr><tr><td>Elasticity</td><td>Rel.  $L ^ { 2 }$ </td><td>500 1</td><td>128</td><td>8 / 8 / 128 / 2</td><td></td></tr><tr><td colspan="6">Irregular domain benchmarks</td></tr><tr><td>Irregular Darcy</td><td>Rel.  $L ^ { 2 }$ </td><td>2000</td><td>128</td><td>4 / 4 / 64 /4</td><td></td></tr><tr><td>Pipe Turbulence</td><td>Rel.  $L ^ { 2 }$ </td><td>2000</td><td>32</td><td>4 / 4 / 64 /4</td><td></td></tr><tr><td>Composite</td><td>Rel.  $L ^ { 2 }$ </td><td>16 2000 16</td><td>32</td><td>4 /4 /64 / 4</td><td></td></tr><tr><td colspan="6">Long-time rollout</td></tr><tr><td>Navier-Stokes</td><td>Rel.  $L ^ { 2 }$ </td><td>500 2</td><td>32</td><td>8 / 8 / 256 / 1</td><td></td></tr><tr><td>ICP Plasma</td><td>Rel.  $L ^ { 2 }$ </td><td>1000 1</td><td>32</td><td>8 / 8 / 256 / 1</td><td></td></tr><tr><td>Heat Flow</td><td>Rel.  $L ^ { 2 }$ </td><td>1000 1</td><td>16</td><td>8 / 8 / 256 / 1</td><td></td></tr></table>

Figure 2 provides a complementary view of the comparison between CANO and Transolver by showing the distribution of the relative $L ^ { 2 }$ error across test samples. CANO exhibits lower median errors on all six datasets, consistently with the results reported in Table 3. The diference is particularly marked for Plasticity, where the two distributions are clearly separated. In addition, CANO generally shows narrower interquartile ranges and more compact error distributions, indicating improved consistency across diferent test cases.

Figure 3 shows representative CANO predictions and absolute errors for three problems. On both the structured Airfoil mesh and the point-cloud Elasticity dataset, CANO captures the key spatial features with lower prediction errors. The Navier-Stokes example further shows that the model preserves the main flow structures over the autoregressive rollout up to timestep +10, with limited error accumulation.

To further examine the behavior of the proposed attention mechanism, Figure 4 compares the spatial organization learned by CANO and Transolver on the Airfoil benchmark. In particular, we visualize the latent assignment matrices extracted from the final layer of the first attention head: the cluster matrix $\mathbf { C } \in \mathbb { R } ^ { N \times K _ { c } }$ for CANO and the slice matrix $\mathbf { A } \in \mathbb { R } ^ { N \times M }$ for Transolver. In CANO, the assignment matrix C determines how query matrix $Q$ are mapped to query clusters $\displaystyle Q _ { c } ,$ , while in Transolver, the slice matrix A determines the probability that the original token X belongs to the slice token S. As shown in Figure 4(b), these cluster assignments are highly localized, with several clusters concentrating around regions with strong flow gradients—specifically near the shockwave. This confirms that CANO adapts its latent clusters to physical features rather than simply partitioning the domain by geometric proximity. In contrast, the slice assignments in Transolver (Figure 4(c)) are smoother, showing little localized adaptation around flow discontinuities. While both methods capture the overall flow field (Figure 4(a)), CANO yields a significantly lower prediction errors.

Table 2 Overview of standard PDE benchmarks. Details on geometry type, spatial dimension, discretization size, dataset splits, and input features.
<table><tr><td rowspan="2">Test Case</td><td rowspan="2">Geometry</td><td rowspan="2">Dim.</td><td rowspan="2">Resolution</td><td colspan="2">Dataset Split</td><td rowspan="2">Input Type</td></tr><tr><td>Train</td><td>Test</td></tr><tr><td>Airfoil [34]</td><td>Structured Mesh</td><td>2</td><td> $2 2 1 \times 5 1$ </td><td>1000</td><td>200</td><td>Boundary Shape</td></tr><tr><td>Pipe [34]</td><td>Structured Mesh</td><td>2</td><td> $1 2 9 \times 1 2 9$ </td><td>1000</td><td>200</td><td>Domain Shape</td></tr><tr><td>Plasticity [34]</td><td>Structured Mesh</td><td>2+1</td><td> $1 0 1 \times 3 1$ </td><td>900</td><td>80</td><td>External Force</td></tr><tr><td>Navier-Stokes [25]</td><td>Regular Grid</td><td>2+1</td><td> $6 4 \times 6 4$ </td><td>1000</td><td>200</td><td>Previous Vorticity</td></tr><tr><td>Darcy [25]</td><td>Regular Grid</td><td>2</td><td>85 × 85</td><td>1000</td><td>200</td><td>Coefficients</td></tr><tr><td>Elasticity [34]</td><td>Point Cloud</td><td>2</td><td>972</td><td>1000</td><td>200</td><td>Domain Shape</td></tr></table>

For neural operators, a low global error (e.g., relative $L ^ { 2 }$ norm) does not necessarily imply physical fidelity in the predicted fields. To evaluate spectral accuracy across spatial scales, we compute the turbulent kinetic energy (TKE) spectrum of the predicted voticity fields.

To compute the one-dimensional (1D) kinetic energy spatial spectrum $E ( k )$ from a two-dimensional (2D) predicted vorticity field $w ( \mathbf { x } )$ of the Navier-Stokes benchmark, we first obtain its Fourier coeficients in the wavenumber space:

$$
\hat { w } ( \mathbf { k } ) = \mathcal { F } \{ w ( \mathbf { x } ) \} ,\tag{17}
$$

where ${ \bf k } = ( k _ { x } , k _ { y } )$ is the wavenumber vector and $k = | \mathbf { k } | = \sqrt { k _ { x } ^ { 2 } + k _ { y } ^ { 2 } }$ . By applying the incompressibility condition, the 2D spectral kinetic energy density is derived by scaling the vorticity power spectrum by the inverse square of the wavenumber (excluding the zero-frequency mean flow where $| \mathbf { k } | = 0 )$ :

$$
E ( { \bf k } ) = \frac { 1 } { 2 } \frac { | \hat { w } ( { \bf k } ) | ^ { 2 } } { | { \bf k } | ^ { 2 } } .\tag{18}
$$

Table 3 Performance comparison of neural operators on standard benchmarks (CANO vs. baselines). All values represent the relative $L ^ { 2 ^ { \dagger } }$ error. The best results are highlighted in bold, and the second-best are underlined. A slash $\left( { } ^ { 6 6 } / { } ^ { 5 9 } \right)$ denotes benchmarks where the baseline is not applicable. Models marked with <sup>∗</sup> are reimplemented by us for fair comparison, while other results are taken from the original papers or the Transolver study.
<table><tr><td rowspan="2">Model</td><td colspan="3">Structured Mesh</td><td colspan="2">Regular Grid</td><td>Point Cloud</td></tr><tr><td>Airfoil  $( \times 1 0 ^ { - 2 } )$ </td><td>Pipe (×10−2)</td><td>Plasticity  $\left( \times 1 0 ^ { - 2 } \right)$ </td><td>Navier-Stokes  $( \times 1 0 ^ { - 1 } )$ </td><td>Darcy  $( \times 1 0 ^ { - 2 } )$ </td><td>Elasticity  $( \times 1 0 ^ { - 2 } )$ </td></tr><tr><td>Classic models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FNO [25]</td><td>/</td><td>/</td><td>/</td><td>1.56</td><td>1.08</td><td>/</td></tr><tr><td>DeepONet [26]</td><td>3.85</td><td>0.97</td><td>1.35</td><td>2.97</td><td>5.88</td><td>9.65</td></tr><tr><td>U-FNO [68]</td><td>2.69</td><td>0.56</td><td>0.39</td><td>2.23</td><td>1.83</td><td>2.39</td></tr><tr><td>Geo-FNO [34]</td><td>1.38</td><td>0.67</td><td>0.74</td><td>1.56</td><td>1.08</td><td>2.29</td></tr><tr><td>U-NO [69]</td><td>0.78</td><td>1.00</td><td>0.34</td><td>1.71</td><td>1.13</td><td>2.58</td></tr><tr><td>F-FNO [67]</td><td>0.78</td><td>0.70</td><td>0.47</td><td>2.32</td><td>0.77</td><td>2.63</td></tr><tr><td>LSM [70]</td><td>0.59</td><td>0.50</td><td>0.25</td><td>1.54</td><td>0.65</td><td>2.18</td></tr><tr><td colspan="3">Transformer-based models</td><td></td><td></td><td></td><td></td></tr><tr><td>Galerkin* [36]</td><td>0.64</td><td>0.44</td><td>0.17</td><td>1.04</td><td>1.97</td><td>0.84</td></tr><tr><td>HT-Net [52]</td><td>0.65</td><td>0.59</td><td>3.33</td><td>1.85</td><td>0.79</td><td>/</td></tr><tr><td>OFormer [37]</td><td>1.83</td><td>1.68</td><td>0.17</td><td>1.71</td><td>1.24</td><td>1.83</td></tr><tr><td>GNOT [39]</td><td>0.76</td><td>0.47</td><td>3.36</td><td>1.38</td><td>1.05</td><td>0.86</td></tr><tr><td>FactFormer [38]</td><td>0.71</td><td>0.60</td><td>3.12</td><td>1.21</td><td>1.09</td><td>/</td></tr><tr><td>ONO [40]</td><td>0.61</td><td>0.52</td><td>0.48</td><td>1.19</td><td>0.76</td><td>1.18</td></tr><tr><td>Transolver* [57]</td><td>0.51</td><td>0.42</td><td>0.12</td><td>0.94</td><td>0.52</td><td>0.56</td></tr><tr><td>CANO (ours)</td><td>0.43</td><td>0.33</td><td>0.07</td><td>0.66</td><td>0.41</td><td>0.46</td></tr><tr><td>Improvement</td><td>15.69%</td><td>21.43%</td><td>41.67%</td><td>29.79%</td><td>21.15%</td><td>17.86%</td></tr></table>

Finally, the 1D radial energy spectrum is calculated by isotropically summing the discrete 2D energy density over concentric annuli of radius k and bin width ∆k:

$$
\mathbf { E } ( \mathbf { k } ) = \sum _ { k \leq | \mathbf { k } | < k + \Delta k } E ( \mathbf { k } ) ,\tag{19}
$$

which efectively quantifies the distribution of turbulent kinetic energy across varying spatial scales.

Based on this formulation, Figure 5 visualizes the energy spectra E(k) of the ground truth and the CANO predicted fields, for the Navier-Stokes benchmarch, averaged over 200 test samples. The spectral distribution predicted by CANO (dashed red line) matches well with the reference solution (solid blue line) across the entire wavenumber domain k, with the shaded regions (mean ± 3 standard deviation) largely overlapping. Even at small scales where the energy drops significantly, the spectrum predicted by CANO aligns well with the reference data.

![](images/35c74edf1f5bd7f3fddc46f7485b462cd42e1a683508178338a823ede0d26f1c.jpg)  
Fig. 2 Performance and distribution comparison: Relative L<sup>2</sup> error of CANO and Transolver on six standard PDE datasets.

Table 4 Comparison of model parameters between Transolver and CANO across diferent standard cases.
<table><tr><td rowspan="2">Benchmark</td><td colspan="2">Number of Parameters</td></tr><tr><td>Transolver</td><td>CANO (Ours)</td></tr><tr><td>Airfoil</td><td>3.07M</td><td>2.44M</td></tr><tr><td>Pipe</td><td>3.07M</td><td>2.49M</td></tr><tr><td>Plasticity</td><td>3.11M</td><td>2.51M</td></tr><tr><td>Navier-Stokes</td><td>12.28M</td><td>8.12M</td></tr><tr><td>Darcy</td><td>3.09M</td><td>2.42M</td></tr><tr><td>Elasticity</td><td>0.98M</td><td>1.33M</td></tr></table>

To evaluate model eficiency, we compare the parameter counts of CANO and Transolver across all six examples (Table 4). Overall, CANO achieves higher accuracy while maintaining a more compact model size. In five out of the six benchmarks, CANO requires fewer parameters than Transolver. This diference is most noticeable on Navier–Stokes, where CANO reduces the parameter count from 12.28M to 8.12M (a 33.9% reduction). Although CANO uses slightly more parameters on the Elasticity dataset, the overall results show that it is more accurate while using fewer parameters.

![](images/9872058eba5055bbc096410e6b6489c877b1ab63dedc751e9e70d09f471ee9f3.jpg)  
Fig. 3 CANO predictions on representative PDE benchmarks. Steady-state physical fields and absolute errors for (a) Airfoil (structured mesh) and (b) Elasticity (point cloud). CANO accurately captures sharp gradients near complex boundaries. (c) Long-term temporal rollout predictions (from timestep +1 to +10) on the Navier-Stokes dataset. CANO efectively preserves intricate highfrequency eddies and structural integrity over time without severe error accumulation.

## 4.2 Irregular domain benchmarks

We next evaluate CANO on irregular geometries, testing its ability to handle complex physical domains. Table 5 summarizes the three irregular domain PDE examples [65]

# VOOCOVVU VOOUOO VUOUOOVU VUVVツVVE DUVUUTOU UVUVVUUU CUVCOVOO (c) Slice Matrix A VVVVUOVU VUVVVVVU VUVVVOVU VVVVVVVU 15 VUVVVUVU VUVVVUVU

Fig. 4 Visualization of the spatial assignments extracted from the final layer of the first head on the Airfoil dataset. (a) The corresponding ground truth, predictions and absolute errors. (b) The spatial activation maps of CANO’s cluster matrix C, showing allocations around discontinuities (e.g., shockwaves). (c) The slice allocation matrix of Transolver A, displaying relatively smooth regions.

utilized in this section: Irregular Darcy, Pipe Turbulence, and Composite. Unlike the

![](images/518f2bb9a910f47f6e878b6ff5f1bfc6901b7b6a4d7953e931f174afbe8dc9ef.jpg)  
Fig. 5 Comparison of kinetic energy spectra (19) between CANO predictions and ground truth for the Navier-Stokes benchmark. Solid lines show the mean spectrum over 200 test samples; shaded regions indicate the mean ± 3 standard deviation.

Table 5 Overview of irregular domain PDE benchmarks. Details on geometry type, spatial dimension, node discretization, dataset splits, and input features.
<table><tr><td rowspan="2">Test Case</td><td rowspan="2">Geometry</td><td rowspan="2">Dim.</td><td rowspan="2">Nodes (N)</td><td colspan="2">Dataset Split</td><td rowspan="2">Input Type</td></tr><tr><td>Train</td><td>Test</td></tr><tr><td>Irregular Darcy [65]</td><td>Point Cloud</td><td>2</td><td>2290</td><td>1000</td><td>200</td><td>Coefficients</td></tr><tr><td>Pipe Turbulence [65]</td><td>Point Cloud</td><td>2</td><td>2673</td><td>300</td><td>100</td><td>Previous State</td></tr><tr><td>Composite [65]</td><td>Point Cloud</td><td>3</td><td>8232</td><td>400</td><td>100</td><td>Temperature</td></tr></table>

previous benchmarks that mostly use regular grids or structured meshes, all geometries in this task are represented entirely as 2D or 3D point clouds, posing significant challenges. We investigate two 2D fluid problems: the Irregular Darcy flow (2,290 nodes) driven by varying the permeability coeficient field, and the Pipe Turbulence (2,673 nodes) requiring state predictions based on previous one. Furthermore, we test our evaluation to a challenging 3D solid mechanics task using the Composite material benchmark (8,232 nodes) driven by temperature profiles. These benchmarks are useful to validate the model’s capability to process irregular geometries. The underlying physical equations and dataset details are introduced as in Appendix B.2.

Table 6 Performance comparison of neural operators on Irregular Darcy, Pipe Turbulence, and Composite benchmarks (CANO vs. baselines). All values represent the relative $L ^ { 2 }$ error $( \times 1 0 ^ { - 4 } )$ . The best results are highlighted in bold, and the second-best are underlined.
<table><tr><td>Model</td><td>Irregular Darcy</td><td>Pipe Turbulence</td><td>Composite</td></tr><tr><td>GraphSAGE [66]</td><td>6.73</td><td>23.60</td><td>20.90</td></tr><tr><td>DeepONet [26]</td><td>1.36</td><td>9.36</td><td>1.88</td></tr><tr><td>POD-DeepONet [30]</td><td>1.30</td><td>2.59</td><td>1.44</td></tr><tr><td>NORM [65]</td><td>1.05</td><td>1.01</td><td>1.00</td></tr><tr><td>HPM [71]</td><td>0.74</td><td>0.83</td><td>0.93</td></tr><tr><td>Transolver* [57]</td><td>0.89</td><td>1.01</td><td>1.61</td></tr><tr><td>CANO (ours)</td><td>0.71</td><td>0.55</td><td>0.89</td></tr><tr><td>Improvement</td><td>4.05%</td><td>33.73%</td><td>4.30%</td></tr></table>

A quantitative comparison with other baselines across these three cases is presented in Table 6. CANO consistently achieves the lowest relative $L ^ { 2 }$ error, outperforming baseline models ranging from foundational networks (GraphSAGE, DeepONet) to recent operators (NORM, HPM, and Transolver). Specifically, on the Pipe Turbulence dataset, CANO yields a 33.73% error reduction compared to the second-best baseline. Furthermore, while Transolver performs competitively on standard benchmarks, its error increases on these unstructured point clouds, particularly in the 3D Composite task. This observation suggests that the slicing mechanism struggles to efectively compress highly unstructured and non-uniform spatial information. In contrast, CANO’s cross-attention mechanism keeps keys and values at their full spatial resolution, maintaining stable predictive accuracy across diferent geometric discretizations.

Figure 6 presents both the error distributions and the spatial distributions of the predictions. Panel (a) displays raincloud plots comparing CANO and Transolver across all three datasets. CANO shifts the overall error distribution downward and suppresses extreme outliers in Irregular Darcy and Pipe Turbulence datasets. While some outlier errors occur in Composite dataset, the overall average error of CANO is still lower than that of Transolver. Panels (b–d) show qualitative predictions for the 2D irregular Darcy flow, Pipe Turbulence, and 3D Composite material, respectively. The predicted fields match the ground truth closely, and the corresponding absolute error maps confirm that CANO maintains low errors, which highlights its efectiveness on irregular physical geometries.

![](images/500fc20c7880f5c7e34ae98151b78ca3eacc3b17685f2353c5d9348c0fa57588.jpg)

![](images/2df507582cec12dc0743188dfaeb794552a09b9d9b489a2ed6b0ba79d64dc712.jpg)

![](images/fcbcd7d661900eb5412d82dcc3ad7a661581ef9ed18a1c6baf894c9a02d75aeb.jpg)

![](images/6d3761e8b4f3d46b8cc12981c73354d7a7c40155db2b0ca0f481548f33a7cb4a.jpg)

![](images/e59fcf8b1010da0cc90b733fc3507034e3341b3dbaa655f77dad0645ded73b7a.jpg)  
Fig. 6 Performance and qualitative distribution on irregular domains. (a) Raincloud plots comparing the relative $L ^ { 2 }$ error distributions of CANO and Transolver. (b)-(d) Representative visual predictions of CANO for the (b) Irregular Darcy, (c) Pipe Turbulence, and (d) Composite benchmarks.

## 4.3 Long-time rollout benchmarks

In practical physical simulations, neural operators are frequently required to extrapolate system dynamics far beyond the short temporal horizons observed during training. The challenge in such cases is to avoid severe error accumulation, numerical dissipation, and the eventual collapse of physical structures over time.

The autoregressive stability and temporal generalization capabilities of our proposed model are evaluated through the three time-dependent datasets shown in Table 7: Navier-Stokes [25], Inductively Coupled Plasma (ICP) and Heat Transfer [72]. Specifically, the Navier-Stokes dataset models the evolution of chaotic fluid vorticity (ω) on a regular 64 × 64 grid. In contrast, the ICP Plasma and Heat Transfer datasets involve highly nonlinear multi-physics phenomena defined on point clouds, predicting complex state variables such as electron density, electron temperature, and velocity fields.

Table 7 Overview of 2D long-time rollout benchmarks. Details on geometry type, node discretization, training/test window, dataset splits, and input features. Models are trained on short temporal windows $( T _ { \mathrm { o u t } } = 5 )$ but evaluated on significantly longer autoregressive rollouts (up to 15 or 50 steps) to test temporal stability.
<table><tr><td rowspan="2">Test Case</td><td rowspan="2">Geometry</td><td rowspan="2">Nodes</td><td colspan="6">Train Window Test Window  $\left( T _ { \mathbf { o u t } } \right)$  Data</td><td rowspan="2"> $\mathbf { S p l i t }$  Input Type</td></tr><tr><td> $T _ { \mathrm { i n } }$ </td><td> $T _ { \mathrm { o u t } }$ </td><td>Short</td><td>Long</td><td>Train Test</td><td></td></tr><tr><td>Navier-Stokes [25] Regular Grid</td><td></td><td>64 × 64</td><td>5</td><td>5</td><td>5</td><td>15</td><td>1000</td><td>200</td><td>ω</td></tr><tr><td>ICP Plasma [72] Point Cloud</td><td></td><td>3424</td><td>5</td><td>5</td><td>5</td><td>50</td><td>499</td><td>150</td><td> $n _ { e } , T _ { e } , v , T$ </td></tr><tr><td>Heat Transfer [72] Point Cloud 4887 (Avg.) 5</td><td></td><td></td><td></td><td>5</td><td>5</td><td>50</td><td>740</td><td>132</td><td> $u , v$ </td></tr></table>

Table 8 Long-term predictive performance across diferent datasets. We evaluate the relative $\dot { L } ^ { 2 }$ error $( \bar { \times } 1 0 ^ { - 1 } )$ at diferent future time steps (e.g., +5, +15, +50). The best results are highlighted in bold.
<table><tr><td rowspan="2">Model</td><td colspan="2">Navier-Stokes</td><td colspan="2">ICP Plasma</td><td colspan="2">Heat Transfer</td></tr><tr><td>+5</td><td>+15</td><td>+5</td><td>+50</td><td>+5</td><td>+50</td></tr><tr><td>Transolver</td><td>1.21</td><td>3.22</td><td>0.29</td><td>2.11</td><td>0.70</td><td>4.87</td></tr><tr><td>CANO</td><td>0.59</td><td>2.57</td><td>0.23</td><td>1.61</td><td>0.58</td><td>4.19</td></tr></table>

We use random-window training to enhance the model’s temporal stability across rollouts. During the training phase, all models can only see the historical of $T _ { \mathrm { i n } } = 5$ steps, learning to predict the subsequent $T _ { \mathrm { o u t } } = 5$ steps. However, during the inference phase, the models are evaluated on two distinct scenarios: a short rollout matching the training horizon $( T _ { \mathrm { o u t } } = 5 )$ , and a significantly long autoregressive rollout, extrapolating up to $T _ { \mathrm { o u t } } = 1 5$ steps for the Navier-Stokes dataset, and up to $T _ { \mathrm { o u t } } = 5 0$ steps for both the ICP Plasma and Heat Transfer datasets.

Table 8 shows that CANO consistently outperforms Transolver throughout the autoregressive rollout. Although the relative $L ^ { 2 }$ error increases with the prediction horizon for both models, CANO maintains lower errors across all datasets and time horizons, indicating better long-term predictive stability.

Figure 7 presents the corresponding error evolution and statistical distributions. The left column displays the relative $L ^ { 2 }$ error accumulation per step, where the errors for both models increases as the simulation advances. The middle and right columns show the error distributions at the short $( T _ { \mathrm { o u t } } = 5 )$ and long-time horizons. While the error distributions for both models are clustered at $T _ { \mathrm { o u t } } = 5$ , the spread of the distribution increases for both models at the long-time horizons. In these extended rollouts, Transolver shows a more pronounced upper tail in the error distribution than CANO.

![](images/57a408a9c665f1d1f7d7435332a864ad8ff47f962d9a49de6dfc6876d818f804.jpg)  
Fig. 7 Temporal error evolution and distribution analysis for long time rollouts. Panels (a), (b), and (c) correspond to the Navier-Stokes, ICP Plasma, and Heat Transfer datasets, respectively. Left: Per step relative $L ^ { 2 }$ error accumulation curves, showing CANO’s slower error growth rate. Middle and Right: Raincloud plots comparing the error distributions at the short $\overset { \smile } { (  { T _ { \mathrm { o u t } } } } = 5 )$ and long $( T _ { \mathrm { o u t } } = \bar { 1 } 5 \ \mathrm { o r } \ 5 0 )$ rollout stages.

Figures 8 and 9 show the spatial distribution of the absolute error during the autoregressive rollout for the ICP Plasma dataset, considering the electron temperature $T _ { e } ,$ and the Heat Transfer dataset, considering the velocity magnitude $\sqrt { u ^ { 2 } + v ^ { 2 } }$ At the first several prediction steps, CANO and Transolver exhibit comparable error distributions. As the rollout progresses, however, the errors produced by Transolver become increasingly pronounced across the domain, while CANO maintains consistently lower error levels over time.

![](images/b5f5dc740d3382ea08bd96a190c1bc32f75032678e7b953494d65109c1834216.jpg)  
Fig. 8 Temporal error evolution on the ICP Plasma dataset. The visualization displays the absolute error of the predicted electron temperature $T _ { e }$ for Transolver and CANO at timesteps +1, +25, and +50 compared to the ground truth.

![](images/c011f4c5006de97ab094209e9ab8be0624f16bdea2dd609e673adecb75f94602.jpg)  
Fig. 9 Temporal error evolution on the Heat Transfer dataset. The visualization displays the absolute error of the predicted velocity magnitude $( \sqrt { u ^ { 2 } + v ^ { 2 } } )$ for Transolver and CANO at timesteps $+ 1$ +25, and +50 compared to the ground truth.

Table 9 Relative $L ^ { 2 }$ error comparison across diferent training sample sizes. All values for Darcy Flow are scaled by $1 0 ^ { - 2 }$ , and Navier-Stokes by $1 0 ^ { - 1 }$ . Best results are highlighted in bold.
<table><tr><td>Problem</td><td>Training Number</td><td>Transolver</td><td>CANO (ours)</td></tr><tr><td rowspan="5">Darcy Flow</td><td>200</td><td>2.17</td><td>1.80</td></tr><tr><td>400</td><td>0.98</td><td>0.72</td></tr><tr><td>600</td><td>0.71</td><td>0.54</td></tr><tr><td>800</td><td>0.67</td><td>0.46</td></tr><tr><td>1000</td><td>0.52</td><td>0.41</td></tr><tr><td rowspan="5">Navier-Stokes</td><td>200</td><td>3.76</td><td>3.09</td></tr><tr><td>400</td><td>3.14</td><td>1.96</td></tr><tr><td>600</td><td>2.87</td><td>1.01</td></tr><tr><td>800</td><td>2.49</td><td>0.90</td></tr><tr><td>1000</td><td>0.94</td><td>0.66</td></tr></table>

Table 10 Relative $L ^ { 2 }$ errors of CANO with diferent cluster numbers $\left( K _ { c } \right)$ across various PDE datasets. The best results for each dataset are highlighted in bold.
<table><tr><td>Cluster Numeber  $( K _ { c } )$ </td><td>Darcy</td><td>Navier-Stokes</td><td>Pipe</td><td>Airfoil</td><td>Elasticity</td><td>Plasticity</td></tr><tr><td>8</td><td>5.38</td><td>9.05</td><td>3.83</td><td>4.96</td><td>5.08</td><td>0.88</td></tr><tr><td>16</td><td>4.83</td><td>7.67</td><td>3.32</td><td>4.77</td><td>4.78</td><td>0.78</td></tr><tr><td>32</td><td>4.73</td><td>6.58</td><td>3.55</td><td>4.29</td><td>4.72</td><td>0.81</td></tr><tr><td>64</td><td>4.36</td><td>6.36</td><td>3.48</td><td>4.88</td><td>4.81</td><td>0.71</td></tr><tr><td>128</td><td>4.08</td><td>5.60</td><td>3.82</td><td>4.38</td><td>4.62</td><td>0.77</td></tr></table>

## 4.4 Model Analysis

In this section, we analyze how CANO scales with training sample size and evaluate its sensitivity to the number of clusters $( K _ { c } )$

## Training Sample Eficiency

Table 9 evaluates the relative $L ^ { 2 }$ error on the Darcy Flow and Navier-Stokes datasets when the number of training samples is scaled from 200 to 1000. Both evaluated models exhibit a monotonic decrease in error as the data scale expands. Under identically restricted data regimes, CANO consistently maintains lower error bounds compared to the Transolver baseline. Notably, in the most data-scarce scenario (200 samples), CANO limits the error to $1 . 8 0 \times 1 0 ^ { - 2 }$ on Darcy Flow and $3 . 0 9 \times 1 0 ^ { - 1 }$ on Navier-Stokes, yielding a better performance than that of Transolver $( 2 . 1 7 \times 1 0 ^ { - 2 }$ and $3 . 7 6 \times 1 0 ^ { - 1 }$ 2 respectively).

## Sensitivity to Cluster Number

Table 10 shows the efect of the cluster number $K _ { c }$ across six datasets, with $K _ { c }$ varied from 8 to 128. The optimal choice of $K _ { c }$ depends on the specific physical system. For complex or turbulent flows $\left( \mathrm { e . g . } \right.$ ., Navier–Stokes and Darcy Flow), the error systematically decreases with larger $K _ { c } ,$ reaching the lowest value at $K _ { c } = 1 2 8$ . In contrast, for systems governed by localized dynamics (Pipe, Airfoil, and Plasticity), the lowest relative $L ^ { 2 }$ errors occur at moderate configurations $( K _ { c } = 1 6$ , 32, and 64, respectively). This indicates that aligning $K _ { c }$ with the spatial complexity of the PDE helps prevent representational redundancy.

## 5 Conclusions and Discussion

In this work, we introduced the Cluster Attention Neural Operator (CANO) to address the information loss caused by latent space compression. CANO relies on an asymmetric cross-attention mechanism that clusters the query space while keeping keys and at full spatial resolution. This formulation allows the model to preserve fine-scale information while maintaining linear $\mathcal { O } ( N )$ computational complexity. Experiments across 12 PDE benchmarks, including regular and irregular geometries as well as longhorizon autoregressive prediction, show that CANO consistently improves predictive accuracy over the considered state-of-the-art baselines.

Several directions remain open for future work. In the present work, the number of clusters $K _ { c }$ is fixed as a priori. An adaptive strategy that adjusts the cluster number according to solution complexity could provide greater performance. In addition, although CANO retains linear complexity, optimized implementations of the clustering and cross-attention operations will be important for extending the method to large-scale three-dimensional applications [58–60].

The results suggest that asymmetric cluster-based attention provides an efective alternative to symmetric latent compression for neural operator learning, particularly when fine-scale spatial information and long-term predictive stability are important. Overall, CANO ofers a framework for scientific machine learning, enabling accurate and eficient modeling of complex physical systems.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The data that support the findings of this study are available within the article.

## Acknowledgments

This work was partially supported by the National Key R&D Program of China (No. 2024YFA1013101), the National Natural Science Foundation of China (No. 12471242), and the China Scholarship Council.

## A Theoretical Connection to Linear Attention

To motivate CANO mathematically, we analyze existing eficient neural operators through the unified framework of generalized linear attention [64]. To decouple the quadratic Softmax computation, the generalized linear attention framework employs independent feature mappings $\phi ( \cdot )$ and $\psi ( \cdot )$ , and further incorporates an intermediate operation G to process the projected latent representations. The attention mechanism can thus be reformulated as [64]:

$$
\mathbf { Y } = \phi ( \mathbf { Q } ) \circ G \circ \big ( \psi ( \mathbf { K } ) ^ { \top } \mathbf { V } \big ) .\tag{20}
$$

Revisiting Transolver from a Generalized Linear Attention Perspective. Recent analysis reveals that the Physics-Attention mechanism in Transolver [57] can be mathematically unified under this generalized linear attention paradigm. Specifically, Transolver compresses physical nodes into M latent slices using a slice-weight matrix, applies an intermediate self-attention operation $G ( \cdot )$ within this slice space, and subsequently deslices the representations back to the physical domain. In the context of the generalized formulation Eq. (20), Transolver’s feature mappings are constructed as:

$$
\phi _ { \mathrm { T r a n s o l v e r } } ( \mathbf { Q } ) = \mathrm { S o f t m a x } ( \mathbf { X A } ) \in \mathbb { R } ^ { N \times M } ,\tag{21}
$$

$$
\psi _ { \mathrm { T r a n s o l v e r } } ( \mathbf { K } ) ^ { \top } = \mathrm { N o r m } \big ( \mathbf { X } \mathbf { A } \big ) ^ { \top } \in \mathbb { R } ^ { M \times N } ,\tag{22}
$$

where $\mathbf { A } \in \mathbb { R } ^ { d \times M }$ is the learnable weight matrix for slicing.

Notably, ϕ<sub>Transolver</sub> and ψ<sub>Transolver</sub> are derived from the exact same parameter $\mathbf { A } _ { : }$ difering only in their normalization strategies. This architectural choice introduces two structural limitations. First, the forced parameter-sharing between the slicing (ψ) and deslicing (ϕ) mappings constrains the model’s performance (see Figure 4 ). Second, because the intermediate interaction $G ( \cdot )$ operates strictly within the highly compressed slice space, the model’s ability to capture fine-grained dynamics is dificult by the information loss during the geometric compression.

CANO as Dynamic Generalized Linear Attention. Our proposed CANO resolves these limitations. Instead of entirely decoupling the queries and keys through independent linear projections in Eq. (20), CANO dynamically clusters the queries while preserving the full-resolution keys, executing a cross-attention mechanism.

Mathematically, by expanding our dispatching and aggregation equations, the full CANO computation is formulated as:

$$
\mathbf { Y } = \mathbf { C } \operatorname { S o f t m a x } \Bigl ( \frac { ( \mathbf { C } ^ { \top } \mathbf { Q } ) \mathbf { K } ^ { \top } } { \sqrt { d } } \Bigr ) \mathbf { V } .\tag{23}
$$

Then we can establish an exact equivalence to the generalized linear attention given by Eq. (20) by defining the CANO feature mappings as:

$$
\phi _ { \mathrm { C A N O } } ( \mathbf { Q } ) \triangleq \mathbf { Q } _ { c } = \mathbf { C } ^ { \top } ( \mathbf { X } \mathbf { W } _ { Q } ) \in \mathbb { R } ^ { K _ { c } \times d } ,\tag{24}
$$

$$
\psi _ { \mathrm { C A N O } } ( \mathbf { K } ) ^ { \top } \triangleq \mathrm { S o f t m a x } \left( \frac { \mathbf { Q } _ { c } ( \mathbf { X } \mathbf { W } _ { K } ) ^ { \top } } { \sqrt { d } } \right) \in \mathbb { R } ^ { K _ { c } \times N } ,\tag{25}
$$

Finally the CANO output can be written as:

$$
\begin{array} { r } { \mathbf { Y } _ { \mathrm { C A N O } } = \phi _ { \mathrm { C A N O } } ( \mathbf { Q } ) \Big ( \psi _ { \mathrm { C A N O } } ( \mathbf { K } ) ^ { \top } \mathbf { V } \Big ) . } \end{array}\tag{26}
$$

This highlights the main advantage of CANO’s architecture. First, CANO is asymmetric, efectively avoiding the structural issue of Transolver. The left mapping $\phi _ { \mathrm { C A N O } } ( \mathbf { Q } )$ acts as a dynamically computed cluster assignment derived strictly from the query space, while the right mapping ψ<sub>CANO</sub>(K) processes the interaction between the clustered queries $\mathbf { Q } _ { c }$ and the full resolution keys K. More importantly, unlike Transolver which compresses the entire input field, CANO preserves the keys and values in their original full resolution space.

## B Govering Equations

In this part, we present the governing equations for the standard PDE benchmarks and the irregular domain benchmarks.

## B.1 Standard PDE Benchmarks

## Airfoil.

This dataset models the transonic flow over an airfoil. Given the low air viscosity in this scenario, the viscous term is negligible, and the system is governed by the Euler equations (representing mass, momentum, and energy conservation, respectively):

$$
\frac { \partial \rho _ { f } } { \partial t } + \nabla \cdot \left( \rho _ { f } \boldsymbol { U } \right) = 0 ,\tag{27}
$$

$$
\frac { \partial \rho _ { f } \boldsymbol { U } } { \partial t } + \nabla \cdot \left( \rho _ { f } \boldsymbol { U } \boldsymbol { U } + p \boldsymbol { I } \right) = 0 ,\tag{28}
$$

$$
\frac { \partial E } { \partial t } + \nabla \cdot ( ( E + p ) \boldsymbol { U } ) = 0 ,\tag{29}
$$

where $\rho _ { f }$ is the fluid density and $E$ is the total energy. The data is generated on a structured mesh with a resolution of $2 2 1 \times 5 1$ . The model learns to map the locations of these mesh points to the Mach number.

## Pipe.

This benchmark focuses on incompressible flow through a pipe. The governing equations are deduced from the Navier-Stokes relations for Newtonian fluids:

$$
\nabla \cdot { \boldsymbol { U } } = 0 ,\tag{30}
$$

$$
\frac { \partial \boldsymbol { U } } { \partial t } + \boldsymbol { U } \cdot \nabla \boldsymbol { U } = \boldsymbol { f } - \frac { 1 } { \rho } \nabla p + \nu \nabla ^ { 2 } \boldsymbol { U } .\tag{31}
$$

The dataset utilizes a structured mesh with a resolution of $1 2 9 \times 1 2 9$ . We use the mesh structure as input to predict the horizontal fluid velocity within the pipe.

## Plasticity.

Focusing on the plastic forging problem, this dataset simulates a plastic material impacted from above by an arbitrary-shaped die. The governing equation for solid material dynamics is given by the equilibrium equation:

$$
\rho _ { s } \frac { \partial ^ { 2 } \boldsymbol { u } } { \partial t ^ { 2 } } + \nabla \cdot \boldsymbol { \sigma } = 0 ,\tag{32}
$$

where $\rho _ { s }$ represents the solid density, u denotes the displacement vector of the material over time t, and σ is the stress tensor. The input is the die shape recorded on a $1 0 1 \times 3 1$ structured mesh, and the output is the deformation of each mesh point over the future 20 time steps.

## Navier-Stokes.

This dataset simulates incompressible, viscous flow on a unit torus. Since the density of the fluid is constant, the energy conservation is independent, and the fluid dynamics are simplified to the vorticity formulation:

$$
\nabla \cdot { \boldsymbol { U } } = 0 ,\tag{33}
$$

$$
\frac { \partial \omega } { \partial t } + U \cdot \nabla \omega = \nu \nabla ^ { 2 } \omega + f ,\tag{34}
$$

where $U = ( u , v )$ is the velocity vector, ${ \boldsymbol { \omega } } = \nabla \times { \boldsymbol { U } }$ is the vorticity, and the viscosity is set to $\nu = 1 0 ^ { - 5 }$ . We predict the future 10 frames based on the past 10 frames on a $6 4 \times 6 4$ regular grid.

## Darcy Flow.

This benchmark models fluid difusion through porous media, governed by the 2D steady-state elliptic equation:

$$
- \nabla \cdot ( a ( x ) \nabla u ( x ) ) = f ( x ) ,\tag{35}
$$

where $a ( x )$ is the permeability coeficient field. This dataset tests the model’s ability to resolve high-frequency spatial discontinuities in the coeficient field $a ( x )$ on an $8 5 \times 8 5$ regular grid.

## Elasticity.

These benchmarks estimate the inner stress of an incompressible material with an arbitrary central void under external tension. They share the same governing equilibrium equation Eq. (32) as the Plasticity dataset. The model maps the material structure to the inner stress.

## B.2 Irregular domain benchmarks

## Irregular Darcy.

This 2D fluid problem is discretized with 2,290 unstructured nodes. It shares the same 2D steady-state elliptic equation as the standard Darcy flow Eq. (35). However, it challenges the model with highly irregular geometric boundaries. The task maps the varying permeability coeficient field $a ( x )$ to the pressure field $u ( x )$

## Pipe Turbulence.

Discretized with 2,673 nodes, this scenario requires state predictions of incompressible fluid dynamics within an unstructured pipe geometry. The governing equations are the as before $\operatorname { E q . }$ (30). The model is tasked with predicting the future velocity states based on previous observations in the irregular domain.

## Composite.

We scale our evaluation to a challenging 3D solid mechanics task using the Composite material benchmark (8,232 nodes). This dataset models the thermoelastic deformation of a 3D composite material driven by temperature profiles. The underlying physics is governed by the steady-state equilibrium equation coupled with thermal expansion:

$$
\nabla \cdot { \boldsymbol { \sigma } } = 0 , \quad { \mathrm { w h e r e } } \quad { \boldsymbol { \sigma } } = \mathbf { C } : { \boldsymbol { \varepsilon } } - \gamma \Delta T \mathbf { I } .\tag{36}
$$

Here, σ is the stress tensor, C is the stifness tensor, ε is the strain tensor, and $\gamma \Delta T \mathbf { I }$ represents the thermal stress induced by the temperature change $\Delta T .$ . The model learns to map the input 3D temperature profiles directly to the structural stress fields.

## References

[1] A. Friedman, Stochastic Diferential Equations and Applications, Springer, 1975.

[2] M. Braun, M. Golubitsky, Diferential Equations and their Applications, Springer, 1983.

[3] H. L. Smith, An Introduction to Delay Diferential Equations with Applications to the Life Sciences, Springer, 2011.

[4] J. D. Logan, Applied Partial Diferential Equations, Springer, 2014.

[5] G. F. Simmons, Diferential Equations with Applications and Historical Notes, CRC Press, 2016.

[6] A. Jaun, J. Hedin, T. Johnson, Numerical Methods for Partial Diferential Equations, Swedish Netuniversity, 1999.

[7] G. A. Evans, J. M. Blackledge, P. D. Yardley, Numerical Methods for Partial Diferential Equations, Springer, 2000.

[8] J. F. Wendt (Ed.), Computational Fluid Dynamics: An Introduction, Springer, 2008.

[9] E. Tadmor, A review of numerical methods for nonlinear partial diferential equations, Bull. Am. Math. Soc. 49 (4) (2012) 507–554.

[10] W. F. Ames, Numerical Methods for Partial Diferential Equations, Academic Press, 2014.

[11] W. E, J. Han, A. Jentzen, A deep learning-based numerical methods for highdimensional parabolic partial diferential equations and backward stochastic diferential equations, Commun. Math. Stat. 5 (2017) 349-380.

[12] J. Jeon, Coupling of artificial intelligence and traditional numerical methods for accelerated and robust PDE solvers, JMST Adv. 7 (2025) 49-55.

[13] J. Han, A. Jentzen, W. E, Solving high-dimensional partial diferential equations using deep learning, PNAS 115 (34) (2018) 8505-8510.

[14] J. Thiyagalingam, M. Shankar, G. Fox, T. Hey, Scientific machine learning benchmarks, Nat. Rev. Phys. 4 (2022) 413-420.

[15] R. Pestourie, Y. Mroueh, C. Rackauckas, P. Das, S. G. Johnson, Physics-enhanced deep surrogates for partial diferential equations, Nat. Mach. Intell. 5 (2023) 1458-1465.

[16] S. L. Brunton, J. N. Kutz, Promising directions of machine learning for partial diferential equations, Nat. Comput. Sci. 4 (2024) 483-494.

[17] B. Yu, The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems, Comm. Math. Stat. 6 (1) (2018) 1–12.

[18] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations, J. Comput. Phys. 378 (2019) 686–707.

[19] L. Lu, X. Meng, Z. Mao, G. E. Karniadakis, DeepXDE: A deep learning library for solving diferential equations, SIAM Rev. 63 (1) (2021) 208–228.

[20] G.E. Karniadakis, I.G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physicsinformed machine learning, Nat. Rev. Phys. 3 (6) (2021) 422–440.

[21] C. Rao, P. Ren, Q. Wang, O. Buyukozturk, H. Sun, Y. Liu, Encoding physics to learn reaction–difusion processes, Nat. Mach. Intell. 5 (7) (2023) 765–779.

[22] A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, M. W. Mahoney, Characterizing possible failure modes in physics-informed neural networks, Adv. Neural Inform. Process. Syst. 34 (2021) 26548–26560.

[23] S. Wang, X. Yu, P. Perdikaris, When and why PINNs fail to train: A neural tangent kernel perspective, J. Comput. Phys. 449 (2022) 110768.

[24] N. Kovachki, S. Lanthaler, S. Mishra, On universal approximation and error bounds for Fourier neural operators, J. Mach. Learn. Res. 22 (2021) 1–76.

[25] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar, Fourier neural operator for parametric partial diferential equations, arXiv preprint, arXiv:2010.08895 (2021).

[26] L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis, Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Nat. Mach. Intell. 3 (3) (2021) 218–229.

[27] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, A. Anandkumar, Neural operator: Learning maps between function spaces with applications to PDEs, J. Mach. Learn. Res. 24 (2023) 1–97.

[28] S. Lanthaler, Z. Li, A. M. Stuart, Nonlocality and nonlinearity implies universality in operator learning, Constr. Approx. 62 (2025) 261-303.

[29] T. Chen, H. Chen, Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems, IEEE Trans. Neural Netw. 6 (4) (1995) 911-917.

[30] L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, G. E. Karniadakis, A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data, Comput. Methods Appl. Mech. Engrg. 393 (2022) 114778.

[31] S. Wang, H. Wang, P. Perdikaris, Learning the solution operator of parametric partial diferential equations with physics-informed DeepONets, Sci. Adv. 7 (40) (2021) eabi8605.

[32] P. Jin, S. Meng, L. Lu, MIONet: Learning multiple-input operators via tensor product, SIAM J. Sci. Comput. 44 (6) (2022) A3490–A3514.

[33] A. Kopaniˇc´akov´a, G. E. Karniadakis, DeepONet based preconditioning strategies for solving parametric linear systems of equations, SIAM J. Sci. Comput. 47 (1) (2025) C151–C181.

[34] Z. Li, D. Z. Huang, B. Liu, A. Anandkumar, Fourier neural operator with learned deformations for PDEs on general geometries, J. Mach. Learn. Res. 24 (2023) 1–26.

[35] Z. Li, N. Kovachki, C. Choy, B. Li, J. Kossaifi, S. Otta, A. Anandkumar, Geometryinformed neural operator for large-scale 3D PDEs, Adv. Neural Inf. Process. Syst. 36 (2023) 35836–35854.

[36] S. Cao, Choose a transformer: Fourier or Galerkin, Adv. Neural Inf. Process. Syst. 34 (2021) 24924–24940.

[37] Z. Li, K. Meidani, A. B. Farimani, Transformer for partial diferential equations operator learning, arXiv preprint, arXiv:2205.13671 (2022).

[38] Z. Li, D. Shu, A. Barati Farimani, Scalable transformer for PDE surrogate modeling, Adv. Neural Inf. Process. Syst. 36 (2023) 28010–28039.

[39] Z. Hao, Z. Wang, H. Su, C. Ying, Y. Dong, S. Liu, J. Zhu, GNOT: A general neural operator transformer for operator learning, in: Proc. Int. Conf. Mach. Learn. 2023, pp. 12556–12569.

[40] Z. Xiao, Z. Hao, B. Lin, Z. Deng, H. Su, Improved operator learning by orthogonal attention, arXiv preprint, arXiv:2310.12487 (2023).

[41] B. Shih, A. Peyvan, Z. Zhang, G. E. Karniadakis, Transformers as neural operators for solutions of diferential equations with finite regularity, Comput. Methods Appl. Mech. Engrg. 434 (2025) 117560.

[42] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, I. Polosukhin, Attention is all you need, Adv. Neural Inf. Process. Syst. 30 (2017).

[43] J. Devlin, M. W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of deep bidirectional transformers for language understanding, Proc. North Am. Chap. Assoc. Comput. Linguist, 2019, pp. 4171–4186.

[44] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, D. Amodei, Language models are few-shot learners, Adv. Neural Inf. Process. Syst. 33 (2020) 1877–1901.

[45] A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret, Transformers are RNNs: Fast autoregressive transformers with linear attention, Proc. Int. Conf. Mach. Learn. 2020 5156–5165.

[46] S. Wang, B. Z. Li, M. Khabsa, H. Fang, H. Ma, Linformer: Self attention with linear complexity, arXiv preprint, arXiv:2006.04768 (2020).

[47] Z. Shen, M. Zhang, H. Zhao, S. Yi, H. Li, Eficient attention: Attention with linear complexities, Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. 2021, pp. 3531–3539.

[48] D. Han, T. Ye, Y. Han, Z. Xia, S. Pan, P. Wan, G. Huang, Agent attention: On the integration of softmax and linear attention, in: Proc. Eur. Conf. Comput. Vis. 2024, pp. 124–140.

[49] D. Han, Y. Pu, Z. Xia, Y. Han, X. Pan, X. Li, G. Huang, Bridging the divide: Reconsidering softmax and linear attention, Adv. Neural Inf. Process. Syst. 37 (2024) 79221–79245.

[50] O. Ovadia, A. Kahana, P. Stinis, E. Turkel, D. Givoli, G. E. Karniadakis, VITO: Vision transformer-operator, Comput. Methods Appl. Mech. Engrg. 428 (2024) 117109.

[51] S. Wang, J. H. Seidman, S. Sankaran, H. Wang, G. J. Pappas, P. Perdikaris, CVIT: Continuous vision transformer for operator learning, arXiv preprint, arXiv:2405.13998 (2024).

[52] X. Liu, B. Xu, S. Cao, L. Zhang, Mitigating spectral bias for the multiscale operator learning, J. Comput. Phys. 506 (2024) 112944.

[53] A. Dosovitskiy, An image is worth 16x16 words: Transformers for image recognition at scale, arXiv preprint, arXiv:2010.11929 (2020).

[54] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, B. Guo, Swin Transformer: Hierarchical vision transformer using shifted windows, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. 2021, pp. 10012–10022.

[55] T. Wang, C. Wang, Latent neural operator for solving forward and inverse PDE problems, Adv. Neural Inf. Process. Syst. 37 (2024) 33085–33107.

[56] B. Alkin, A. F¨urst, S. Schmid, L. Gruber, M. Holzleitner, J. Brandstetter, Universal physics transformers: A framework for eficiently scaling neural operators, Adv. Neural Inf. Process. Syst. 37 (2024) 25152–25194.

[57] H. Wu, H. Luo, H. Wang, J. Wang, M. Long, Transolver: A fast transformer solver for PDEs on general geometries, arXiv preprint, arXiv:2402.02366 (2024).

[58] H. Luo, H. Wu, H. Zhou, L. Xing, Y. Di, J. Wang, M. Long, Transolver++: An accurate neural solver for partial diferential equations on million-scale geometries, arXiv preprint, arXiv:2502.02414 (2025).

[59] M. A. Nabian, C. Liu, R. Ranade, S. Choudhry, X-meshgraphnet: Scalable multi-scale graph neural networks for physics simulation, arXiv preprint, arXiv:2411.17164 (2024).

[60] M. Bleeker, M. Dorfer, T. Kronlachner, R. Sonnleitner, B. Alkin, J. Brandstetter, Neural computational fluid dynamics: deep learning on high-fidelity automotive aerodynamics simulations, arXiv preprint, arXiv:2502.09692 (2025).

[61] A. Anandkumar, K. Azizzadenesheli, K. Bhattacharya, N. Kovachki, Z. Li, B. Liu, A. Stuart, Neural operator: Graph kernel network for partial diferential equations, 2020, in: ICLR Workshop on Integration of Deep Neural Models and Diferential Equations.

[62] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, A. Stuart, K. Bhattacharya, A. Anandkumar, Multipole graph neural operator for parametric partial diferential equations, Adv. Neural Inf. Process. Syst. 33 (2020) 6755–6766.

[63] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, J.N. Bakas, A. Courville, On the spectral bias of neural networks, in: Int. Conf. Mach. Learn., 2019, pp. 5301–5310.

[64] W. Hu, S. Liu, P. Qiao, Z. Sun, Y. Dou, Transolver is a linear transformer: Revisiting physics attention through the lens of linear attention, Proc. AAAI Conf. Artif. Intell. 40 (1) (2026) 408–416.

[65] G. Chen, X. Liu, Q. Meng, L. Chen, C. Liu, Y. Li, Learning neural operators on Riemannian manifolds, arXiv preprint, arXiv:2302.08166 (2023).

[66] W. Hamilton, Z. Ying, J. Leskovec, Inductive representation learning on large graphs, Adv. Neural Inf. Process. Syst. 30 (2017) 1024–1034.

[67] A. Tran, A. Mathews, L. Xie, C. S. Ong, Factorized Fourier neural operators, arXiv preprint, arXiv:2111.13802 (2021).

[68] G. Wen, Z. Li, K. Azizzadenesheli, A. Anandkumar, S. M. Benson, U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow, Adv. Water Resour. 163 (2022) 104180.

[69] M. A. Rahman, Z. E. Ross, K. Azizzadenesheli, U-NO: U-shaped neural operators, arXiv preprint, arXiv:2204.11127 (2022).

[70] H. Wu, T. Hu, H. Luo, J. Wang, M. Long, Solving high-dimensional PDEs with latent spectral models, arXiv preprint, arXiv:2301.12664 (2023).

[71] X. Yue, Y. Yang, L. Zhu, Holistic physics solver: Learning PDEs in a unified spectral physical space, arXiv preprint, arXiv:2410.11382 (2024).

[72] P. Liu, X. Ren, P. Wang, H. Yuan, Z. Hao, G. Chen, S. Cai, An eficient graph transformer operator for learning physical dynamics with manifolds embedding, arXiv preprint, arXiv:2512.10227 (2025).