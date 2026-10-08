# Activation-Aware Weight Tensorization: A Calibration-Time Preconditioner for Tensor-Network LLM Compression

Alessandro Beatini Sapienza University of Rome alessandro.beatini@uniroma1.it

Marco Maronese Independent Researcher   
marco.maronese95@gmail.com   
ORCID: 0000-0001-5548-5947

Emanuele Rodolà Sapienza University of Rome rodola@di.uniroma1.it

## Abstract

Post-training tensor-network compression replaces Transformer linear layers with Tensor Train (TT) or Tree Tensor Network (TTN) operators, but standard decompositions minimize weightspace Frobenius error rather than functional error under the layer’s activation distribution. We propose Activation-aware Weight Tensorization (AWT), a training-free calibration wrapper that preconditions each weight matrix with a diagonal activation-derived scale before an unchanged TT/TTN solver and deploys the result with only an input-side elementwise rescaling. Across Llama 3.1 8B, Ministral 8B, and Qwen2.5 7B, AWT consistently improves vanilla TT/TTN tensorization at 2–6× compression: under single-operator replacement, AWT closes 12–35% of the WikiText perplexity gap to the dense baseline across the three model families and 2–6× compression settings; while under multi-operator Llama sufix replacement it closes 27–60% across attention-group and all-seven-matrix settings. The gains also transfer to downstream HellaSwag and ARC-Challenge evaluations. We further show that diagonal preconditioning is a robustness–modularity tradeof rather than a diagonal-covariance assumption: a dense fullcovariance oracle wins its own weighted objective in 80/81 cases, yet diagonal AWT gives better held-out functional fidelity in 53/81 cases. Together, these results position AWT as a principled, modular preconditioner for improving functional fidelity in fixed TT/TTN compression pipelines without modifying the decomposition solver.

## 1 Introduction

Large language models (LLMs) are increasingly expensive to serve because dense linear operators dominate memory movement and compute during inference [2, 25, 3, 22]. This has motivated a broad set of post-training compression techniques, including quantization, pruning, distillation, low-rank factorization, and structured tensor decompositions [33]. Tensor-network compression ofers a structured alternative to standard matrix factorizations. A dense matrix $W \in \mathbb { R } ^ { m \times n }$ can be reshaped into a higher-order tensor and represented by a network of low-rank cores: a Tensor Train (TT) or matrix-product operator uses a chain of cores [21, 20], while a Tree Tensor Network (TTN) uses a loop-free hierarchy over tensor modes [23, 7]. Unlike truncated SVD, which imposes a single global rank on a matrix, tensor networks distribute rank budgets across tensor modes and internal edges of a chosen topology. This richer design space has made tensorization an increasingly relevant direction for LLM compression, with recent methods and systems such as CompactifAI, TensorLLM, and Minima demonstrating its use for compressing large neural operators and LLM components [24, 5, 1, 8, 15].

Despite this progress, post-training tensorization is typically activation-blind. Once a tensorization shape, topology, and rank budget are chosen, standard TT/TTN backends approximate the weight matrix by minimizing an isotropic reconstruction objective in weight space. This objective is computationally convenient, but it does not directly capture the error induced by replacing a layer inside a pretrained model.

This observation is well established in post-training quantization. Methods such as SmoothQuant and AWQ show that simple calibration statistics can identify activation channels whose perturbation disproportionately afects model behavior [29, 18]. However, the same principle has not been systematically exploited for post-training tensor-network decomposition. Tensor decompositions introduce a diferent constraint from quantization: the approximation is determined by a fixed tensorization layout, network topology, and internal rank budget rather than by a bit-width. This raises a natural question: can activation statistics be used to make existing tensor-network solvers more function-preserving, without changing the solver itself?

We answer this question with Activation-aware Weight Tensorization (AWT). AWT computes a diagonal scale $D _ { \alpha }$ from cached layer inputs, decomposes the reparameterized matrix $W ^ { \prime } = W D _ { \alpha }$ using the same tensor-network backend, tensorization layout, topology, and rank budget as vanilla tensorization, and deploys the resulting approximation as $\widehat { y } = \widetilde { W } ^ { \prime } ( D _ { \alpha } ^ { - 1 } x )$ . Thus, activation information changes the matrix presented to the decomposition solver, but not the solver itself. The method is intentionally minimal: it transfers the equivalent reparameterization idea from activationaware quantization to post-training tensor-network compression, requiring no retraining, no custom TT/TTN solver, and no change to the tensorized operator apart from a lightweight input-side diagonal scaling. Empirically, we find that this simple preconditioning consistently improves the functional fidelity of tensorized LLM operators. The gains persist across model families, operator types, compression ratios, and tensor-network topologies, and become especially important when multiple operators are compressed jointly. In multi-operator sufix replacement, AWT closes 27–60% of the WikiText perplexity gap across both attention-group and all-seven-matrix settings (Table 2), indicating that activation-aware preconditioning helps control the accumulation of tensorization error inside the model.

Contributions. Our main contributions are: (i) we introduce AWT, a diagonal activation-aware preconditioner for post-training TT/TTN compression that preserves the underlying tensor-network solver and deployment path; (ii) we show that AWT is equivalent to a column-weighted reconstruction proxy (Equation (5)), connecting tensorization error to activation-conditioned functional distortion; (iii) we demonstrate consistent gains over vanilla tensorization in single-operator and multi-operator LLM replacement experiments, with the largest absolute gains arising when compression errors accumulate across operators (Table 2); (iv) we evaluate AWT across Llama 3.1 8B, Ministral 8B, and Qwen2.5 7B, three compression ratios, seven operator types, and TT/TTN backends, including matched-compression SVD and downstream task checks; (v) we compare diagonal AWT with a dense full-covariance oracle and show that the diagonal restriction is a robustness–modularity tradeof rather than a diagonal-covariance assumption.

## 2 Related Work

Low-rank and tensorized neural compression. Low-rank factorization has long been used to compress neural networks, from matrix and filter decompositions for convolutional models to post-compression fine-tuning [4, 12, 16, 14]. Tensor-network parameterizations extend this idea by replacing dense operators with structured multilinear factors. Tensor Train (TT) layers represent a matrix as a chain of low-order cores, building on the TT format from numerical linear algebra [21, 20]; related formats such as Tensor Rings and Block-Term decompositions have also been used for neural compression [32, 17, 27]. These methods provide richer rank allocation than a single global matrix rank, but are typically optimized using weight-space reconstruction objectives.

Tensorization for language models. Tensorized representations have been explored in NLP for large embedding tables and Transformer components, including TT-parameterized embeddings, tensorized Transformer layers, and tensor-compressed models trained or distilled directly [10, 19, 31]. Recent work has moved toward post-training tensorization of pretrained LLMs: TensorGPT studies training-free TT/MPS compression of token embeddings [30]; CompactifAI uses quantum-inspired tensor networks, including MPO/TT-matrix-style decompositions, to compress attention and MLP operators with recovery [24, 5, 1]; TensorLLM tensorizes multi-head attention with Tucker-style structure [8]; and Minima presents a production-oriented pipeline combining sensitivity prediction, Tucker/TT/TR decompositions, healing, and custom kernels [15]. These works establish tensorization as an active LLM-compression direction. AWT is orthogonal: rather than proposing a new topology or recovery procedure, it introduces a calibration-time activation-aware preconditioner before standard TT/TTN decomposition.

Function-aware compression objectives. A central limitation of weight-space compression is that Frobenius reconstruction error can be misaligned with the functional error induced inside a pretrained model. Fisher-weighted and distribution-aware decompositions address this by weighting the objective with curvature, Fisher information, or input covariance statistics [11, 13]. These methods more directly target output distortion, but often require modified optimization, dense preconditioners, or solver-specific updates. AWT instead uses activation statistics only to construct a diagonal equivalent reparameterization, leaving the tensor-network solver unchanged.

Activation-aware post-training compression. Activation-aware scaling has been especially successful in post-training quantization. SmoothQuant applies an equivalent rescaling that transfers quantization dificulty from activation outliers into weights [29], while AWQ uses activation statistics to identify salient channels for low-bit weight-only quantization [18]. Related outlier-suppression methods also use equivalent transformations and clipping to reduce structured activation outliers [28]. AWT brings this activation-aware perspective to tensor-network decomposition, where the approximation budget is not a bit-width but a fixed tensorization shape, topology, and rank budget.

## 3 Background

Tensorizing linear operators. We consider a linear layer $y = W x$ with $W \in \mathbb { R } ^ { m \times n }$ . Tensornetwork compression begins by factorizing the input and output dimensions as $\begin{array} { r } { m = \prod _ { k = 1 } ^ { K } m _ { k } } \end{array}$ and $\begin{array} { r } { n = \prod _ { k = 1 } ^ { K } n _ { k } } \end{array}$ , and reshaping W into an order-2K tensor ${ \mathcal { W } } \in \mathbb { R } ^ { m _ { 1 } \times \cdots \times m _ { K } \times n _ { 1 } \times \cdots \times n _ { K } }$ . The indices associated with $m _ { k }$ correspond to output modes, while those associated with $n _ { k }$ correspond to input modes. A tensor-network approximation then replaces W by a contraction of smaller core tensors whose internal bond dimensions control the parameter budget.

Tensor Train and Tree Tensor Network decompositions. The Tensor Train (TT) format, equivalently a Matrix Product Operator (MPO) for linear maps, represents W as a chain of cores $\{ \mathcal { G } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ , where $\mathcal { G } ^ { ( k ) } \in \mathbb { R } ^ { r _ { k - 1 } \times m _ { k } \times n _ { k } \times r _ { k } }$ and $r _ { 0 } = r _ { K } = 1 \ [ 2 1$ , 20]. Its entries are given by

$$
\mathcal { W } _ { i _ { 1 } , \dots , i _ { K } , j _ { 1 } , \dots , j _ { K } } = \sum _ { \alpha _ { 1 } , \dots , \alpha _ { K - 1 } } \prod _ { k = 1 } ^ { K } \mathcal { G } _ { \alpha _ { k - 1 } , i _ { k } , j _ { k } , \alpha _ { k } } ^ { ( k ) } .\tag{1}
$$

The TT ranks $\{ r _ { k } \}$ are the internal bond dimensions and determine the compression–expressivity trade-of. In practice, TT decompositions are commonly computed by successive truncated SVDs of tensor unfoldings. Tree Tensor Networks (TTNs) generalize this chain topology to a loop-free hierarchical graph over tensor modes [23, 7]. The leaves correspond to the physical modes of W, while internal nodes store latent cores that merge groups of modes. Each internal edge e carries a bond dimension $\chi _ { e } ,$ , often bounded by a maximum rank $\chi .$ Compared with TT, a TTN allocates rank across a hierarchy rather than along a single ordering, yielding a diferent inductive bias and parameter allocation for the same tensorized operator.

Activation-conditioned reconstruction. Standard post-training TT/TTN decompositions approximate $W$ by minimizing a weight-space objective such as $\| W - \widetilde { W } \| _ { F } ^ { 2 }$ . However, replacing a linear layer inside a pretrained model induces output error that depends on the layer input distribution. For $y = W x$ and approximation $\widehat { W }$

$$
\begin{array} { r } { \mathbb { E } \| ( W - \widehat { W } ) x \| _ { 2 } ^ { 2 } = \| ( W - \widehat { W } ) \Sigma _ { x } ^ { 1 / 2 } \| _ { F } ^ { 2 } , \qquad \Sigma _ { x } = \mathbb { E } [ x x ^ { \top } ] . } \end{array}\tag{2}
$$

Thus, directions that are frequently or strongly activated should contribute more to the reconstruction objective. Full covariance-weighted decompositions can directly target this metric, but they generally require modified solvers or dense preconditioners [11, 13]. AWT instead uses an equivalent diagonal reparameterization. For any invertible diagonal matrix D,

$$
W x = ( W D ) ( D ^ { - 1 } x ) ,\tag{3}
$$

so one can rescale the columns of W before decomposition and compensate by the inverse scaling at inference time. This is the same functional-preserving principle used by activation-aware quantization methods such as SmoothQuant and AWQ [29, 18], but applied here to tensor-network decomposition rather than bit-width allocation.

## 4 Activation-aware Weight Tensorization (AWT)

AWT is a calibration-time preconditioning wrapper for tensor-network compression of linear layers. Given a linear map $y = W x$ and a tensor-network backend (TT/TTN) with a fixed tensorization layout, topology, and rank/parameter budget, we compute a diagonal importance scaling $D _ { \alpha }$ from cached input activations and use it to bias the decomposition toward input channels that matter more under typical inputs. Concretely, AWT (i) right-preconditions the weight matrix by scaling its input-channel columns, (ii) tensorizes and decomposes the scaled operator using a standard Frobenius-norm TT/TTN routine, and (iii) deploys the compressed layer without materializing the dense unscaled matrix: the inverse diagonal scaling is applied to the layer input, followed by the TT/TTN operator (Equation (9) and Algorithm 1). As discussed in the background, the expected functional distortion of a compressed linear layer induces a covariance-weighted metric $\| ( W - \widehat { W } ) \Sigma ^ { 1 / 2 } \| _ { F }$ . Directly optimizing this full-metric objective within TT/TTN would require either a solver that handles a non-isotropic least-squares metric or an explicit dense preconditioner. The latter globally mixes input channels, can be poorly aligned with a fixed tensorization shape and rank budget, and would require a dense transform at deployment. AWT instead uses a diagonal preconditioner. This restriction is deliberate: it preserves the original channel layout seen by the tensorization, requires only single-pass gradient-free calibration statistics, and leaves the TT/TTN solver and contraction path unchanged. Thus, AWT trades covariance optimality for modularity, statistical robustness, and compatibility with of-the-shelf tensor-network decomposition.

Key identity: diagonal preconditioning yields a weighted Frobenius proxy. Let $D \succ 0$ be diagonal, define the preconditioned matrix $W ^ { \prime } = W D$ , and let $\widetilde { W } ^ { \prime }$ be any approximation to

$W ^ { \prime }$ returned by an of-the-shelf Frobenius-norm TT/TTN routine after tensorization. Define the efective deployed operator as $\widehat { W } = \widetilde { W } ^ { \prime } D ^ { - 1 }$ . Then

$$
( W - \widehat { W } ) D = W D - \widetilde { W } ^ { \prime } , \qquad \| ( W - \widehat { W } ) D \| _ { F } = \| W D - \widetilde { W } ^ { \prime } \| _ { F } .\tag{4}
$$

Equivalently, AWT minimizes a column-weighted reconstruction proxy,

$$
\| ( W - \widehat { W } ) D \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { n } d _ { i } ^ { 2 } \| W _ { : i } - \widehat { W } _ { : i } \| _ { 2 } ^ { 2 } ,\tag{5}
$$

so columns associated with larger $d _ { i }$ are fit more accurately. This is the precise sense in which AWT injects activation information while reusing an of-the-shelf TT/TTN Frobenius solver.

Diagonal scales from activation importance. Given calibration activations $\{ \mathbf { x } ^ { ( t ) } \} _ { t = 1 } ^ { T }$ for a layer, we compute per-channel importance scores $\{ \mathrm { i m p } _ { i } \} _ { i = 1 } ^ { n }$ and normalize them by their mean $\begin{array} { r } { \overline { { \mathrm { i m p } } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } } \end{array}$ imp . For numerical stability, we clamp the mean as $\overline { { \mathrm { i m p } } } _ { \varepsilon } = \operatorname* { m a x } ( \overline { { \mathrm { i m p } } } , \varepsilon )$ and define scales $s _ { i }$ via

$$
s _ { i } = \operatorname* { m a x } \left( s _ { \mathrm { m i n } } , \left( \frac { \mathrm { i m p } _ { i } } { \mathrm { i m p } _ { \varepsilon } } \right) ^ { \alpha } \right) , \qquad s _ { \mathrm { m i n } } = \operatorname* { m a x } ( \mathtt { m i n } _ { - } \mathtt { s c a l e } , \varepsilon ) .\tag{6}
$$

where $\alpha > 0$ controls the strength of reweighting, ε is a small numerical constant, and $s _ { \mathrm { m i n } }$ prevents large factors in $D _ { \alpha } ^ { - 1 }$ . We set $D _ { \alpha } = \operatorname { d i a g } ( s _ { 1 } , . . . , s _ { n } )$ . In our experiments, we instantiate imp using either MeanAbs, imp<sub>i</sub> MeanAbs $= \mathbb { E } _ { t } [ | x _ { i } ^ { ( t ) } | ]$ , or Std, im $\mathrm { p } _ { i } ^ { \mathrm { S t d } } = \sqrt { \mathrm { V a r } _ { t } ( x _ { i } ^ { ( t ) } ) + \varepsilon }$ , with statistics taken over cached calibration activations. We use MeanAbs and Std because they are single-pass, gradient-free statistics that can be estimated from cached calibration activations, preserving the training-free character of AWT. Gradient- or Fisher-based saliency measures are possible, but would require backpropagation and would change the computational profile of the method. The diagonal restriction should not be interpreted as assuming diagonal activation covariance; rather, it preserves the original channel layout and keeps the preconditioner statistically and computationally lightweight.

Precondition–tensorize–deploy (without materializing $\widehat { W } )$ . For input-channel scaling (right preconditioning), we scale the columns of W and tensorize the scaled operator:

$$
W ^ { \prime } = W D _ { \alpha } ,\tag{7}
$$

apply a tensor-network decomposition to obtain $\widetilde { W } ^ { \prime } \approx W ^ { \prime }$ , and conceptually define

$$
\widehat { W } = \widetilde { W } ^ { \prime } D _ { \alpha } ^ { - 1 } .\tag{8}
$$

In practice, to avoid materializing $\widehat { W }$ (which would be dense), we implement the layer as the equivalent computation

$$
\begin{array} { r } { \widehat { y } = \widehat { W } x = ( \widetilde { W } ^ { \prime } D _ { \alpha } ^ { - 1 } ) x = \widetilde { W } ^ { \prime } ( D _ { \alpha } ^ { - 1 } x ) , } \end{array}\tag{9}
$$

i.e., we apply the inverse diagonal scaling to the layer input and then apply the TT/TTN-compressed operator $\widetilde { W } ^ { \prime }$ . Since $D _ { \alpha }$ is diagonal, this adds only an elementwise per-channel multiply and can be fused with surrounding kernels. This preserves the TT/TTN parameter budget while remaining functionally equivalent to Eq. (8). The algorithm is intentionally modular: any TT/TTN decomposition implementation can be used in the decomposition step, and the only data-aware component is the diagonal scaling computed from a small calibration set.

Algorithm 1 Activation-aware Weight Tensorization (AWT)   
Require: Weight matrix $W \in \mathbb { R } ^ { m \times n }$ , calibration inputs ${ \mathcal { C } } ,$ importance statistic imp(·) (MeanAbs   
or Std), exponent $\alpha > 0 .$ , numerical constant $\varepsilon > 0 ,$ minimum scale min\_scal $\mathsf { e } > 0 ,$ , tensorization   
shape, tensor-network routine Decomp(·) (TT/TTN) with target ranks/budget.   
Ensure: Compressed operator implemented as $( \widetilde { W } ^ { \prime } , D _ { \alpha } ^ { - 1 } )$   
1: Run the model on $\mathcal { C }$ and collect layer input activations $\{ \mathbf { x } ^ { ( t ) } \} _ { t = 1 } ^ { T }$   
2: Compute per-channel importances imp ← imp( $\{ x _ { i } ^ { ( t ) } \} _ { t = 1 } ^ { T } )$ for $i = 1 , \ldots , n .$   
3: Compute mean importance imp $\textstyle  { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n }$ im $\mathrm { p } _ { i }$ and clamp imp ← max $( \overline { { \mathrm { i m p } } } , \varepsilon )$   
4: Set $s _ { \mathrm { m i n } } $ max(min\_scale, ε) and form scales $s _ { i } \gets \operatorname* { m a x } \biggl ( s _ { \mathrm { m i n } } , \biggl ( \frac { \operatorname* { i m p } _ { i } } { \operatorname* { i m p } _ { \varepsilon } } \biggr ) ^ { \alpha } \biggr )$   
5: Form $D _ { \alpha } \gets \mathrm { d i a g } ( s _ { 1 } , \ldots , s _ { n } )$   
6: Precondition (right scaling): $W ^ { \prime } \gets W D _ { \alpha } .$   
7: Tensorize/reshape $W ^ { \prime }$ into $\mathcal { W } ^ { \prime }$ using the specified shape.   
8: Decompose: $\widetilde { \mathcal { W } } ^ { \prime } \gets \mathrm { D E C O M P } ( \mathcal { W } ^ { \prime } )$ (subject to target ranks/budget).   
9: Untensorize $\widetilde { \mathcal { W } } ^ { \prime }$ into a (compressed) operator $\widetilde { W } ^ { \prime }$   
10: Deploy without materializing ${ \widehat { W } } \colon$ implement $\widehat { y } = \widetilde { W } ^ { \prime } ( D _ { \alpha } ^ { - 1 } x )$   
11: return $( \widetilde { W } ^ { \prime } , D _ { \alpha } ^ { - 1 } )$

## 5 Experimental Validation

Models and scope. We evaluate AWT on Llama 3.1 8B, Ministral 8B, and Qwen2.5 7B, considering the seven dense Transformer operators $\left\{ W _ { q } , W _ { k } , W _ { v } , W _ { o } , W _ { u p } , W _ { g a t e } , W _ { d o w n } \right\}$ at $2 / 4 / 6 \times$ compression. Unless stated otherwise, vanilla and AWT use the same tensorization layout, topology, rank budget, calibration set, and TT/TTN backend; only the diagonal preconditioning difers. Our experiments are organized around four questions: (i) whether activation-aware preconditioning improves the local functional fidelity of a fixed TT/TTN approximation; (ii) whether these local improvements translate into model-level behavior under single-operator replacement; (iii) how the gains behave when multiple compressed operators are composed; and (iv) how the diagonal AWT proxy compares with alternative baselines, including matched-compression SVD and dense full-covariance preconditioning.

Calibration data and activation caches. AWT estimates per-channel activation statistics from a small unlabeled calibration set. We draw calibration prompts from a subset of lm-eval tasks (MMLU, HellaSwag, and BoolQ), run the original dense model, and record the input activations to each targeted linear operator using forward hooks. Only the input vectors immediately before multiplication by W are cached; labels and task losses are never used. Calibration examples are kept disjoint from all examples used for metric caches and downstream evaluation. For layer-level evaluation, we construct a separate held-out metric cache of input activations X. This cache is used only to measure local output error and to select among candidate tensorized configurations at a fixed parameter budget. Separating calibration and metric caches avoids selecting tensorized operators on the same activations used to construct the AWT scales.

## 5.1 Compared methods

Tensorization backends. We compare Tensor Train (TT) and two Tree Tensor Network variants (TTN4, TTN8), which difer in tree depth/fanout and therefore impose diferent rank-allocation biases. For a given matrix and compression budget, vanilla tensorization and AWT use the same tensorization shape, mode ordering, topology, and rank budget. This fixed-layout protocol isolates the efect of activation-aware preconditioning rather than jointly optimizing the tensorization search

space.

Activation-aware preconditioning. For each TT/TTN backend, we evaluate Vanilla tensorization with no preconditioning, AWT-MeanAbs with channel scales from $\mathbb { E } [ | x _ { i } | ]$ , and AWT-Std with channel scales from the per-channel standard deviation. All AWT variants use right preconditioning (input-channel scaling) and deploy the compressed operator as $\widetilde { W } ^ { \prime } ( D _ { \alpha } ^ { - 1 } x )$ . We sweep $\alpha \in \{ 0 . 5 , 1 . 0 , 1 . 5 \}$ and use $\alpha = 1 . 0$ as the default unless otherwise specified.

Baselines and oracles. We include matched-compression truncated SVD as a standard matrix low-rank baseline under the same single-operator protocol. We also compare diagonal AWT with a dense full-covariance preconditioner in an ofline operator-level analysis. The full-covariance variant is used as an oracle-style comparison for the richer covariance-weighted objective, while diagonal AWT is the practical plug-in method used in deployment.

Compression matching. To compare methods at equal compression, we sweep TT/TTN rank configurations and select, for each target matrix, candidates satisfying the specified compression ratio. Unless otherwise stated, parameter counts include the tensor-network cores; the additional AWT diagonal scale is one input-channel vector and is negligible relative to the compressed operator, but we report it explicitly in the implementation details. Candidates are ranked using the held-out layer-level functional fidelity metric defined below.

Layer-level metrics. Let $\boldsymbol { X } \in \mathbb { R } ^ { B \times n }$ be a held-out activation cache for a layer with weight matrix $W \in \mathbb { R } ^ { m \times n }$ , and let $Y = X W ^ { \top }$ . For AWT, predictions are computed without materializing $\widehat { W }$ as $\widehat { Y } = ( X D _ { \alpha } ^ { - 1 } ) ( \widetilde { W } ^ { \prime } ) ^ { \top }$ , where $\widetilde { W } ^ { \prime }$ approximates $W ^ { \prime } = W D _ { \alpha }$ . Our primary local metric is activation-conditioned relative output error,

$$
\mathrm { R e l F r o b } _ { o u t } = \frac { \| \widehat { Y } - Y \| _ { F } } { \| Y \| _ { F } + \varepsilon } , \qquad \varepsilon = 1 0 ^ { - 1 2 } .\tag{10}
$$

For comparison, we also report weight-space reconstruction error RelFrob $\ d _ { w } = \| W - \widehat { W } \| _ { F } / ( \| W \| _ { F } +$ $\varepsilon )$ . For the full-covariance oracle analysis only, we use RelFrob<sub>Σ</sub> $= \| ( W - \widehat { W } ) \Sigma _ { X } ^ { 1 / 2 } \| _ { F } / ( \| W \Sigma _ { X } ^ { 1 / 2 } \| _ { F } + \varepsilon )$

## 5.2 Evaluation protocols

Local layer-wise sweep. For each targeted matrix instance, backend, compression ratio, and AWT setting, we sweep admissible TT/TTN rank configurations and compute local metrics on the held-out metric cache. This sweep is performed on activations from the original dense model and is independent of end-to-end benchmark scores. At each parameter budget, we retain the candidate with the lowest $\mathrm { R e l F r o b } _ { \mathrm { o u t } }$ unless otherwise stated. This protocol gives a controlled comparison between vanilla tensorization and AWT under the same tensor-network hypothesis class.

Single-operator replacement. To test whether local improvements translate into model-level behavior, we replace one dense operator at a time with its selected tensorized approximation and evaluate the resulting model. Results are aggregated over blocks, operator types, and backends at each compression ratio. This protocol isolates the efect of compressing a single matrix while preserving the surrounding model.

Multi-operator sufix replacement. Single-operator replacement is a controlled diagnostic, but deployed compression typically composes errors from multiple approximated operators. We therefore also evaluate sufix replacement on Llama 3.1 8B: for the last k Transformer blocks, we jointly replace either the attention group $\{ W _ { q } , W _ { k } , W _ { v } , W _ { o } \}$ or all seven dense operators. This evaluates whether the AWT-vs-vanilla advantage persists when tensorization errors accumulate through multiple compressed operators.

End-to-end metrics. We report WikiText perplexity for both single-operator and multi-operator replacement. We additionally evaluate downstream accuracy on HellaSwag and ARC-Challenge as external task-level checks. These downstream results are used as corroborating evidence rather than as the primary selection criterion; no candidate is selected using downstream task accuracy.

Runtime and deployment overhead. AWT performs calibration and scale construction ofline. At inference time, the only additional operation relative to the underlying tensorized layer is the input-side elementwise multiplication $D _ { \alpha } ^ { - 1 } x$ before the TT/TTN contraction. The tensorized operator and contraction path are unchanged, and the scaling can be fused into surrounding kernels. We therefore separate AWT’s preconditioning overhead from the latency properties of a particular optimized TT/TTN backend.

## 6 Results

Across local, single-operator, and multi-operator evaluations, AWT consistently improves vanilla TT/TTN tensorization at matched parameter budgets. The local efect appears in activationconditioned output error, transfers to WikiText perplexity across three model families, and becomes more pronounced when several operators are compressed jointly. We also find that truncated SVD remains a strong matrix low-rank baseline at matched parameter count, while dense full-covariance preconditioning is not a uniformly better practical substitute for the diagonal AWT proxy.

Local fidelity and metric alignment AWT first improves the quantity it is designed to afect: held-out activation-conditioned output fidelity. At matched $\mathrm { T T / T T N }$ parameter budgets, moderate activation reweighting reduces $\mathrm { R e l F r o b } _ { \mathrm { o u t } }$ relative to vanilla tensorization (Table 1). The optimum for functional fidelity difers from the optimum for weight-space reconstruction: increasing α continues to reduce RelFrob<sub>w</sub>, but the best $\mathrm { R e l F r o b } _ { \mathrm { o u t } }$ occurs around $\alpha = 1$ . This separation supports the view that preserving the action of a compressed layer on real activations is not equivalent to minimizing plain Frobenius error in weight space. The same pattern appears at the model level. Across single-operator replacement trials, changes in activation-conditioned output error are more predictive of changes in perplexity than changes in weight-space error (Table 1, right). We therefore use held-out output fidelity as the candidate-selection metric for end-to-end replacement.

Single-operator replacement across model families We next replace one dense operator at a time with its selected tensorized approximation and evaluate WikiText perplexity. This protocol isolates the efect of compressing one operator while preserving the rest of the model. As shown in Table 2, AWT improves over vanilla tensorization at every tested compression ratio across all three model families. The absolute perplexity changes are small in this controlled setting, but the direction is consistent and the fraction of the dense-to-vanilla gap recovered remains substantial across architectures.

Composition: multi-operator sufix replacement Single-operator replacement tests isolated substitutions; practical compression composes several approximations. We therefore evaluate sufix replacement on Llama 3.1 8B, jointly replacing either the attention group $\{ W _ { q } , W _ { k } , W _ { v } , W _ { o } \}$ or all seven dense operators in the last k Transformer blocks. Table 2 reports $\Delta \mathrm { P P L } = \mathrm { P P L } \mathrm { _ { A W T } } \mathrm { - } \mathrm { P P L } _ { \mathrm { v a n i l l a } } ,$ so negative values favor AWT. AWT improves over vanilla tensorization in every tested sufix setting. The absolute gain grows as more operators are replaced, indicating that activation-aware preconditioning helps control the accumulation of tensorization error through the model.

## 6.1 Baselines, downstream tasks, and deployment overhead

Matched-compression SVD. We include truncated SVD as a strong matrix low-rank reference under the same single-operator protocol. as detailed in Appendix Table A.11., A matched-compression

Table 1: Local functional fidelity and metric alignment. Left: efect of the AWT exponent α at 4× compression on Llama 3.1 8B. AWT reduces activation-conditioned output error, while weight-space error follows a diferent trend. Right: Spearman correlation between changes in local metrics and changes in WikiText perplexity under single-operator replacement (α = 1.0), grouped by operator family. Groups: $\mathrm { Q K } = \{ W _ { q } , W _ { k } \} , \mathrm { V O } = \{ W _ { v } , W _ { o } \} , \mathrm { M L P } = \{ W _ { u p } , W _ { d o w n } , W _ { g a t e } \}$ . Higher correlation means the local metric better predicts model-level degradation.  
AWT exponent ablation at 4×
<table><tr><td rowspan="2">α Stat.</td><td rowspan="2"></td><td colspan="2">Out.</td></tr><tr><td>Val. ∆</td><td>Weight Val. ∆</td></tr><tr><td>0.5 Std</td><td>0.606</td><td>-9.2 0.754</td><td>-1.7</td></tr><tr><td></td><td>0.5 MeanAbs 0.594 -10.9 0.751</td><td></td><td>-2.0</td></tr><tr><td>1.0 Std</td><td></td><td>0.583 -12.6 0.691</td><td>-9.9</td></tr><tr><td></td><td></td><td>1.0 MeanAbs 0.577 -13.5 0.665 -13.2</td><td></td></tr><tr><td>1.5 Std</td><td></td><td>0.589 -11.6 0.526 -31.4</td><td></td></tr><tr><td></td><td></td><td>1.5 MeanAbs 0.592 -11.3 0.455 -40.6</td><td></td></tr></table>

∆ values are percentages.

Metric–PPL rank correlation
<table><tr><td>Model</td><td>Metric</td><td>Overall</td><td>QK</td><td>VO MLP</td></tr><tr><td>Llama 3.1 8B</td><td>∆Out</td><td>0.427</td><td>0.680</td><td>0.219 0.373</td></tr><tr><td></td><td>∆Weight</td><td>0.246</td><td>-0.207</td><td>-0.290 0.265</td></tr><tr><td>Ministral 8B</td><td>∆Out</td><td>0.395</td><td>0.794</td><td>0.245 0.235</td></tr><tr><td></td><td>∆Weight</td><td>0.042</td><td>-0.299 0.438</td><td>-0.072</td></tr><tr><td>Qwen2.5 7B</td><td>∆Out</td><td>0.327</td><td>0.401 0.458</td><td>0.165</td></tr><tr><td></td><td>∆Weight</td><td>-0.049</td><td>-0.213 -0.260</td><td>0.164</td></tr></table>

QK = {W<sub>q</sub>, W<sub>k</sub>}, VO = {W<sub>v</sub>, W<sub>o</sub>}, MLP  
$= \{ W _ { u p } , W _ { d o w n } , W _ { g a t e } \}$ . ∆Out denotes ∆RelFrob<sub>out</sub> and ∆Weight denotes ∆RelFrob<sub>w</sub>.

SVD baseline is stronger than the TT/TTN pipeline under the single-operator protocol, with WikiText PPL 8.71/8.81/8.87 at 2/4/6×, compared to 8.76/8.85/8.93 for AWT and 8.79/8.90/8.97 for vanilla TT/TTN. This clarifies the role of AWT: it improves a fixed tensor-network compression family rather than claiming that TT/TTN universally dominates matrix low-rank compression.

Downstream task validation. As an additional task-level check, we evaluate HellaSwag and ARC-Challenge under single-operator replacement on Llama 3.1 8B. AWT improves over vanilla tensorization at all tested compression ratios on both tasks. The absolute gains are small, as expected in the controlled single-operator setting: HellaSwag improves by 0.0003–0.0008 accuracy points and ARC-Challenge improves by 0.0009–0.0017. We report the full backend-specific downstream breakdowns in the appendix.

Deployment overhead. AWT performs calibration, importance estimation, and scale construction ofline. At inference time, the only extra operation relative to the underlying tensorized layer is the elementwise input scaling $D _ { \alpha } ^ { - 1 }$ x before the TT/TTN contraction. The tensorized operator and contraction path are unchanged, and the scaling can typically be fused into surrounding kernels. We therefore treat optimized TT/TTN latency as a property of the tensorization backend, with AWT adding only a lightweight input-side rescaling.

Diagonal-design analysis. The diagonal preconditioner should not be interpreted as a diagonal-covariance assumption. Activations are strongly non-diagonal: the diagonal covariance mass is only 0.032/0.142/0.212 for Llama/Ministral/Qwen, while the of-diagonal Frobenius ratio is 0.984/0.926/0.885. A dense full-covariance oracle wins its own covariance-weighted objective in 80/81 cases, but diagonal AWT gives lower held-out output error in 53/81 cases. This supports diagonal AWT as a robustness–modularity tradeof; full details are reported in Table A.12.

## 7 Discussion and Conclusion

Our results show that activation-aware preconditioning improves post-training tensor-network compression of LLM linear operators. Across local fidelity, single-operator replacement, and multioperator sufix replacement, AWT consistently improves vanilla TT/TTN tensorization at matched parameter budgets across Llama 3.1 8B, Ministral 8B, and Qwen2.5 7B. The gains become especially pronounced when multiple operators are replaced jointly, suggesting that activation-aware preconditioning helps control accumulated tensorization error.

Table 2: Single-operator and multi-operator WikiText evaluation. Left: single-operator replacement across model families; lower PPL is better. “Gap closed” is computed relative to the dense baseline of the corresponding model. Right: multi-operator sufix replacement on Llama 3.1 8B; entries are $\Delta \mathrm { P P L } = \mathrm { P P L } \mathrm { _ { A W T } } \mathrm { - } \mathrm { P P L } _ { \mathrm { v a n i l l a } } ,$ so negative is better.  
Single-operator replacement
<table><tr><td>Model</td><td>Comp. Dense</td><td></td><td>Vanilla AWT</td><td>Gap closed</td></tr><tr><td rowspan="3">Llama 3.1 8B</td><td>2×</td><td>8.64</td><td>8.79 8.76</td><td>20.4%</td></tr><tr><td>4×</td><td>8.64</td><td>8.90 8.85</td><td>17.1%</td></tr><tr><td>6×</td><td>8.64</td><td>8.97 8.93</td><td>12.2%</td></tr><tr><td rowspan="3">Ministral 8B</td><td>2×</td><td>8.66</td><td>8.74</td><td>8.71 35.3%</td></tr><tr><td>4×</td><td>8.66</td><td>8.81 8.76</td><td>31.3%</td></tr><tr><td>6×</td><td>8.66</td><td>8.86 8.80</td><td>28.6%</td></tr><tr><td rowspan="3">Qwen2.5 7B</td><td>2×</td><td>9.59</td><td>9.76</td><td>9.70 33.5%</td></tr><tr><td>4×</td><td>9.59</td><td>9.85 9.79</td><td>25.8%</td></tr><tr><td>6×</td><td>9.59</td><td>9.91 9.84</td><td>22.2%</td></tr></table>

Multi-operator sufix replacement
<table><tr><td>Scope</td><td>k</td><td>2×</td><td>4×</td><td>6×</td></tr><tr><td>Attention</td><td>1</td><td>-0.41</td><td>-0.47</td><td>-0.46</td></tr><tr><td></td><td>2</td><td>-0.59</td><td>-0.72</td><td>-0.81</td></tr><tr><td></td><td>4</td><td>-0.76</td><td>-1.02</td><td>-1.11</td></tr><tr><td></td><td>8</td><td>-1.40</td><td>-2.48</td><td>-2.75</td></tr><tr><td>All seven</td><td>1</td><td>-1.86</td><td>-2.61</td><td>-3.62</td></tr><tr><td></td><td>2</td><td>-7.74</td><td>-9.21</td><td>-11.71</td></tr></table>

Functional fidelity versus weight reconstruction. Weight-space reconstruction error is not the right proxy for the behavior of a compressed layer inside an LLM. The activation-conditioned output metric is more aligned with perplexity changes than plain Frobenius weight error, especially for attention projections (Table 1). This supports the design of AWT: the preconditioner does not change the tensor-network hypothesis class, but changes how a fixed rank budget is allocated by the solver, biasing reconstruction toward input channels that matter under real activations. It also explains why the best α for functional fidelity need not minimize weight-space error.

Topology- and operator-dependent tensorizability. Tensorizability is not uniform across operators or topologies. TTN backends often outperform TT at the same parameter budget, with the clearest gap on attention projections, particularly $W _ { q }$ and $W _ { k }$ . A plausible explanation is structural alignment: multi-head attention induces head-wise and block-structured subspaces, and a hierarchical tree can merge correlated groups earlier than a one-dimensional TT chain [26, 23, 7, 9]. This remains an empirical hypothesis, since performance depends on tensorization shape, mode ordering, rank allocation, and TTN topology. MLP matrices are generally harder to tensorize at matched compression, consistent with the view that feed-forward blocks implement richer feature mixing and content storage [6]. These diferences suggest that practical tensorization pipelines should allocate ranks and choose topologies in an operator-aware manner.

Relationship to SVD and other compression families. The matched-compression SVD baseline highlights that standard matrix low-rank approximation remains strong at matched parameter count. Tensor networks ofer a diferent structured compression family, with multi-core representations, explicit internal rank budgets, and topology choices that may be useful for deployment and for modeling structured correlations after tensorization. AWT should therefore be viewed as improving this tensor-network family rather than replacing other compression primitives. It is also complementary to quantization, pruning, and matrix factorization: one could apply AWT before tensorizing weights and then further quantize the resulting tensor-network cores.

Diagonal preconditioning as a robustness–modularity tradeof. The diagonal restriction in AWT should not be interpreted as assuming diagonal activation covariance. Our covariance analysis shows the opposite: activations are strongly non-diagonal. Dense full-covariance preconditioning optimizes its intended covariance-weighted objective more directly, but does not reliably improve held-out local output fidelity. Diagonal AWT gives up optimality on that objective, but preserves the original channel layout, avoids dense deployment transforms, reduces the statistical estimation burden, and leaves the TT/TTN solver and contraction path unchanged.

Limitations and future work. Several limitations remain. First, tensorization quality depends on design choices that we keep fixed or only partially sweep, including dimension factorization, mode ordering, rank allocation, and TTN topology. Jointly optimizing these choices with activation-aware preconditioning is a natural next step. Second, our runtime analysis isolates the overhead of AWT itself, an input-side diagonal scaling, but end-to-end latency depends on optimized TT/TTN kernels and hardware-specific implementations. Third, calibration statistics are estimated from a small input set and may not capture all downstream distributions. Finally, while our multi-operator sufix experiments move beyond isolated substitutions, full-model compression requires coordinated choices across layers and operators.

In summary, AWT is a training-free, solver-agnostic mechanism for making post-training tensornetwork compression more activation-aware. By inserting a diagonal equivalent reparameterization before standard TT/TTN decomposition, it improves functional fidelity without retraining, custom solvers, or changes to the tensorized operator, especially when approximation errors accumulate across multiple compressed operators.

## References

[1] Ander Alvarez, Alessandro Genuardi, Nilotpal Sinha, Antonio Tiene, Mikail Okyay, Bakbergen Ryskulov, David Montero, Samuel Mugel, and Román Orús. Scaling laws for energy eficiency of local llms. arXiv preprint arXiv:2512.16531, 2025.

[2] Tom B. Brown et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.

[3] Aakanksha Chowdhery et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

[4] Emily L. Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, and Rob Fergus. Exploiting linear structure within convolutional networks for eficient evaluation. In Advances in Neural Information Processing Systems, 2014.

[5] Damien Fovet, Shashank Chamoli, Sarah Oury, and Srishti Singhal. Accuracy and consumption analysis from a compressed model by compactifai from multiverse computing. arXiv preprint arXiv:2507.08836, 2025.

[6] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Empirical Methods in Natural Language Processing, 2021.

[7] Lars Grasedyck. Hierarchical singular value decomposition of tensors. SIAM Journal on Matrix Analysis and Applications, 2010.

[8] Yuxuan Gu, Wuyang Zhou, Giorgos Iacovides, and Danilo Mandic. Tensorllm: Tensorising multi-head attention for enhanced reasoning and compression in llms. In 2025 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2025.

[9] Wolfgang Hackbusch. A sparse matrix arithmetic based on h-matrices. part i: Introduction to h-matrices. Computing, 1999.

[10] Oleksii Hrinchuk, Valentin Khrulkov, Leila Mirvakhabova, Elena Orlova, and Ivan Oseledets. Tensorized embedding layers for eficient model compression. In Findings of EMNLP, 2019.

[11] Chen-Yu Hsu et al. Language model compression with weighted low-rank factorization. In International Conference on Learning Representations, 2022.

[12] Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. Speeding up convolutional neural networks with low rank expansions. In British Machine Vision Conference, 2014.

[13] Christofer Kalle et al. Distribution-aware tensor decomposition for neural network compression. arXiv preprint, 2025.

[14] Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim Choi, Lu Yang, and Dongjun Shin. Compression of deep convolutional neural networks for fast and low power mobile applications. In International Conference on Learning Representations, 2016.

[15] Sergii Kozyrev and Davyd Maiboroda. A practical tensor-network compression pipeline for production-scale large language models. arXiv preprint arXiv:2602.01613, 2026.

[16] Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan Oseledets, and Victor Lempitsky. Speeding-up convolutional neural networks using fine-tuned cp-decomposition. In International Conference on Learning Representations, 2015.

[17] Yang Li et al. Block-term tensor neural networks. In arXiv preprint, 2017.

[18] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. Awq: Activationaware weight quantization for llm compression and acceleration. In International Conference on Machine Learning, 2024.

[19] Xindian Ma, Peng Zhang, Shuai Zhang, Nan Duan, Yuexian Hou, Ming Zhou, and Dawei Song. A tensorized transformer for language modeling. In Advances in Neural Information Processing Systems, 2019.

[20] Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry Vetrov. Tensorizing neural networks. In Advances in Neural Information Processing Systems, 2015.

[21] Ivan V. Oseledets. Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5): 2295–2317, 2011.

[22] Reiner Pope et al. Eficiently scaling transformer inference. In Proceedings of Machine Learning and Systems, 2023.

[23] Y.-Y. Shi, L.-M. Duan, and Guifré Vidal. Classical simulation of infinite-size quantum lattice systems in the tree tensor network approach. Physical Review A, 74(2):022320, 2006.

[24] Andrei Tomut, Saeed S Jahromi, Abhijoy Sarkar, Uygar Kurt, Sukhbinder Singh, Faysal Ishtiaq, Cesar Muñoz, Prabdeep Singh Bajaj, Ali Elborady, Gianni del Bimbo, et al. Compactifai: extreme compression of large language models using quantum-inspired tensor networks. arXiv preprint arXiv:2401.14109, page 12, 2024.

[25] Hugo Touvron et al. Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[26] Ashish Vaswani et al. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

[27] Wenqi Wang, Yifan Sun, Brian Eriksson, Wenlin Wang, and Vaneet Aggarwal. Wide compression: Tensor ring nets. In IEEE Conference on Computer Vision and Pattern Recognition, 2018.

[28] Xiuying Wei, Ruihao Gong, Yuhang Li, Xianglong Liu, and Fengwei Yu. Outlier suppression: Pushing the limit of low-bit transformer language models. In Advances in Neural Information Processing Systems, 2022.

[29] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and eficient post-training quantization for large language models. In International Conference on Machine Learning, 2023.

[30] Ming Xu et al. Tensorgpt: Eficient compression of gpt models via tensor-train decomposition. arXiv preprint, 2023.

[31] Zhen Yang et al. Quantization-aware training for tensor-compressed transformers. arXiv preprint, 2023.

[32] Qibin Zhao, Guoxu Zhou, Shengli Xie, Liqing Zhang, and Andrzej Cichocki. Tensor ring decomposition. In arXiv preprint arXiv:1606.05535, 2016.

[33] Xun Zhu et al. A survey on model compression for large language models. arXiv preprint, 2023.

## A Additional Experimental Results

This appendix provides additional experimental details and breakdowns omitted from the main text for space: downstream task results, deployment overhead, covariance analysis, cross-model ablations, backend comparisons, and multi-operator replacement tables.

## A.1 Llama per-operator output-error breakdowns

Table A.1: Per-operator activation-conditioned output error (relative Frobenius; lower is better) under TTN depth4 tensorization with α = 1. Entries are reported as mean $\mu$ and standard deviation σ over N = 32 single-operator trials per matrix type.
<table><tr><td></td><td colspan="3">2× compression</td><td colspan="3">4× compression</td><td colspan="3">6× compression</td><td></td></tr><tr><td></td><td>Matrix Vanilla (µ±σ)</td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td> ${ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )$ </td><td>Vanilla (µ±σ)</td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { M e a n A b s } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { V a n i l l a } \ ( \mu \pm \sigma )$ </td><td>Std  $( \mu \pm \sigma )$ </td><td> ${ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )$ </td><td>N</td></tr><tr><td> $W _ { q }$ </td><td> $0 . 1 8 0 \pm 0 . 0 4 7$ </td><td> $0 . 1 4 7 \pm 0 . 0 4 1$ </td><td> $0 . 1 4 7 \pm 0 . 0 4 1$ </td><td> $0 . 2 4 5 \pm 0 . 0 5 6$ </td><td> $0 . 2 0 5 \pm 0 . 0 5 2$ </td><td> $0 . 2 0 5 \pm 0 . 0 5 2$ </td><td> $0 . 2 7 9 \pm 0 . 0 6 0$ </td><td>0.236 ± 0.057</td><td> $0 . 2 3 6 \pm 0 . 0 5 7$ </td><td>32</td></tr><tr><td> $W _ { k }$ </td><td> $0 . 1 9 2 \pm 0 . 0 4 9$ </td><td> $0 . 1 4 7 \pm 0 . 0 3 7$ </td><td> $0 . 1 4 7 \pm 0 . 0 3 7$ </td><td> $0 . 3 0 4 \pm 0 . 0 6 7$ </td><td> $0 . 2 3 0 \pm 0 . 0 5 0$ </td><td> $0 . 2 3 0 \pm 0 . 0 5 0$ </td><td> $0 . 3 5 8 \pm 0 . 0 7 0$ </td><td> $0 . 2 7 4 \pm 0 . 0 5 4$ </td><td> $0 . 2 7 4 \pm 0 . 0 5 3$ </td><td>32</td></tr><tr><td> $W _ { v }$ </td><td> $0 . 5 9 5 \pm 0 . 0 6 5$ </td><td> $0 . 4 7 0 \pm 0 . 0 4 8$ </td><td> $0 . 4 6 7 \pm 0 . 0 4 7$ </td><td> $0 . 7 4 9 \pm 0 . 0 6 1$ </td><td> $0 . 6 2 3 \pm 0 . 0 5 0$ </td><td> $0 . 6 2 1 \pm 0 . 0 4 9$ </td><td> $0 . 8 1 1 \pm 0 . 0 5 4$ </td><td> $0 . 6 9 2 \pm 0 . 0 4 5$ </td><td> $0 . 6 8 9 \pm 0 . 0 4 5$ </td><td>32</td></tr><tr><td> $W _ { o }$ </td><td> $0 . 5 2 6 \pm 0 . 0 8 7$ </td><td> $0 . 3 8 7 \pm 0 . 0 8 9$ </td><td> $0 . 3 8 9 \pm 0 . 0 8 5$ </td><td> $0 . 6 6 5 \pm 0 . 0 9 8$ </td><td> $0 . 5 2 5 \pm 0 . 1 0 7$ </td><td> $0 . 5 2 6 \pm 0 . 1 0 4$ </td><td> $0 . 7 2 2 \pm 0 . 1 0 0$ </td><td> $0 . 5 9 1 \pm 0 . 1 1 3$ </td><td> $0 . 5 9 1 \pm 0 . 1 1 0$ </td><td>32</td></tr><tr><td> $W _ { u p }$ </td><td> $0 . 5 9 7 \pm 0 . 0 7 8$ </td><td> $0 . 5 5 9 \pm 0 . 0 7 5$ </td><td> $0 . 5 6 1 \pm 0 . 0 7 6$ </td><td> $0 . 6 4 6 \pm 0 . 0 8 4$ </td><td> $0 . 6 1 1 \pm 0 . 0 8 1$ </td><td> $0 . 6 1 3 \pm 0 . 0 8 2$ </td><td> $0 . 6 9 8 \pm 0 . 0 9 1$ </td><td> $0 . 6 6 6 \pm 0 . 0 8 7$ </td><td> $0 . 6 6 8 \pm 0 . 0 8 8$ </td><td>32</td></tr><tr><td> $W _ { d o w n }$ </td><td> $0 . 6 7 7 \pm 0 . 0 9 4$ </td><td> $0 . 5 3 5 \pm 0 . 1 5 7$ </td><td> $0 . 5 4 7 \pm 0 . 1 5 8$ </td><td> $0 . 7 3 0 \pm 0 . 0 9 8$ </td><td> $0 . 5 8 7 \pm 0 . 1 7 1$ </td><td> $0 . 6 0 0 \pm 0 . 1 7 2$ </td><td> $0 . 7 8 4 \pm 0 . 1 0 1$ </td><td> $0 . 6 4 4 \pm 0 . 1 8 4$ </td><td> $0 . 6 5 6 \pm 0 . 1 8 6$ </td><td>32</td></tr><tr><td> $W _ { g a t e }$ </td><td> $0 . 3 9 6 \pm 0 . 0 6 1$ </td><td> $0 . 3 7 4 \pm 0 . 0 5 8$ </td><td> $0 . 3 7 6 \pm 0 . 0 5 9$ </td><td> $0 . 4 3 2 \pm 0 . 0 6 7$ </td><td> $0 . 4 1 1 \pm 0 . 0 6 3$ </td><td> $0 . 4 1 3 \pm 0 . 0 6 4$ </td><td> $0 . 4 7 0 \pm 0 . 0 7 3$ </td><td> $0 . 4 5 2 \pm 0 . 0 6 9$ </td><td> $0 . 4 5 4 \pm 0 . 0 7 0$ </td><td>32</td></tr></table>

Table A.2: Per-operator activation-conditioned output error (relative Frobenius; lower is better) under TTN depth8 tensorization with $\alpha = 1$ . Entries are reported as mean $\mu$ and standard deviation σ over N = 32 single-operator trials per matrix type.
<table><tr><td colspan="4">2× compression</td><td colspan="3">4× compression</td><td colspan="4">6× compression</td></tr><tr><td></td><td>Matrix Vanilla (µ±σ) Std (µ±σ)</td><td></td><td> ${ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )$ </td><td>Vanilla (µ±σ)</td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { M e a n A b s } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { V a n i l l a } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td>MeanAbs  $( \mu \pm \sigma )$ </td><td>N</td></tr><tr><td> $W _ { q }$ </td><td> $0 . 1 8 2 \pm 0 . 0 4 7$ </td><td> $0 . 1 4 8 \pm 0 . 0 4 1$ </td><td> $0 . 1 4 8 \pm 0 . 0 4 1$ </td><td> $0 . 2 4 7 \pm 0 . 0 5 7$ </td><td> $0 . 2 0 7 \pm 0 . 0 5 2$ </td><td> $0 . 2 0 7 \pm 0 . 0 5 2$ </td><td> $0 . 2 8 2 \pm 0 . 0 6 0$ </td><td> $0 . 2 3 9 \pm 0 . 0 5 7$ </td><td> $0 . 2 3 9 \pm 0 . 0 5 7$ </td><td>32</td></tr><tr><td> $W _ { k }$ </td><td> $0 . 1 9 7 \pm 0 . 0 5 0$ </td><td> $0 . 1 5 1 \pm 0 . 0 3 8$ </td><td>0.151 ± 0.038</td><td> $0 . 3 0 5 \pm 0 . 0 6 6$ </td><td> $0 . 2 3 3 \pm 0 . 0 5 1$ </td><td> $0 . 2 3 2 \pm 0 . 0 5 0$ </td><td> $0 . 3 5 7 \pm 0 . 0 6 9$ </td><td> $0 . 2 7 5 \pm 0 . 0 5 3$ </td><td> $0 . 2 7 5 \pm 0 . 0 5 3$ </td><td>32</td></tr><tr><td> $W _ { v }$ </td><td> $0 . 6 0 3 \pm 0 . 0 6 5$ </td><td> $0 . 4 7 8 \pm 0 . 0 4 8$ </td><td>0.475 ± 0.047</td><td> $0 . 7 7 0 \pm 0 . 0 6 5$ </td><td> $0 . 6 2 7 \pm 0 . 0 4 8$ </td><td> $0 . 6 2 4 \pm 0 . 0 4 8$ </td><td> $0 . 8 3 3 \pm 0 . 0 6 1$ </td><td> $0 . 6 9 1 \pm 0 . 0 4 5$ </td><td> $0 . 6 8 8 \pm 0 . 0 4 4$ </td><td>32</td></tr><tr><td> $W _ { o }$ </td><td> $0 . 5 3 0 \pm 0 . 0 8 8$ </td><td> $0 . 3 9 0 \pm 0 . 0 9 0$ </td><td> $0 . 3 9 3 \pm 0 . 0 8 6$ </td><td> $0 . 6 7 0 \pm 0 . 0 9 8$ </td><td> $0 . 5 3 0 \pm 0 . 1 0 8$ </td><td> $0 . 5 3 1 \pm 0 . 1 0 5$ </td><td> $0 . 7 2 7 \pm 0 . 1 0 0$ </td><td> $0 . 5 9 7 \pm 0 . 1 1 3$ </td><td> $0 . 5 9 7 \pm 0 . 1 1 1$ </td><td>32</td></tr><tr><td> $W _ { u p }$ </td><td> $0 . 5 9 7 \pm 0 . 0 7 8$ </td><td> $0 . 5 5 9 \pm 0 . 0 7 5$ </td><td> $0 . 5 6 1 \pm 0 . 0 7 6$ </td><td> $0 . 6 4 9 \pm 0 . 0 8 5$ </td><td> $0 . 6 1 3 \pm 0 . 0 8 2$ </td><td> $0 . 6 1 5 \pm 0 . 0 8 2$ </td><td> $0 . 7 0 0 \pm 0 . 0 9 1$ </td><td> $0 . 6 6 8 \pm 0 . 0 8 8$ </td><td> $0 . 6 7 0 \pm 0 . 0 8 8$ </td><td>32</td></tr><tr><td> $W _ { d o w n }$ </td><td> $0 . 6 7 7 \pm 0 . 0 9 4$ </td><td> $0 . 5 3 5 \pm 0 . 1 5 7$ </td><td> $0 . 5 4 7 \pm 0 . 1 5 7$ </td><td> $0 . 7 3 3 \pm 0 . 0 9 9$ </td><td> $0 . 5 9 0 \pm 0 . 1 7 1$ </td><td> $0 . 6 0 3 \pm 0 . 1 7 2$ </td><td> $0 . 7 8 6 \pm 0 . 1 0 2$ </td><td> $0 . 6 4 6 \pm 0 . 1 8 5$ </td><td> $0 . 6 5 8 \pm 0 . 1 8 5$ </td><td>32</td></tr><tr><td> $W _ { g a t e }$ </td><td> $0 . 3 9 6 \pm 0 . 0 6 1$ </td><td> $0 . 3 7 4 \pm 0 . 0 5 8$ </td><td> $0 . 3 7 6 \pm 0 . 0 5 9$ </td><td> $0 . 4 3 3 \pm 0 . 0 6 7$ </td><td> $0 . 4 1 3 \pm 0 . 0 6 3$ </td><td> $0 . 4 1 5 \pm 0 . 0 6 4$ </td><td> $0 . 4 7 1 \pm 0 . 0 7 3$ </td><td> $0 . 4 5 3 \pm 0 . 0 6 9$ </td><td> $0 . 4 5 6 \pm 0 . 0 7 0$ </td><td>32</td></tr></table>

Table A.3: Per-operator activation-conditioned output error (relative Frobenius; lower is better) under TT (cores4) tensorization with $\alpha = 1$ . Entries are reported as mean $\mu$ and standard deviation σ over N = 32 single-operator trials per matrix type.
<table><tr><td rowspan="2"></td><td colspan="3">2× compression</td><td colspan="3">4× compression</td><td colspan="3">6× compression</td><td></td></tr><tr><td></td><td></td><td></td><td>Matrix Vanilla (µ±σ) Std (µ±σ) MeanAbs (µ±σ) Vanilla (µ±σ)</td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td></td><td></td><td></td><td>MeanAbs (µ±σ) Vanilla (µ±σ) Std (µ±σ) MeanAbs (µ±σ) N</td><td></td></tr><tr><td> $W _ { q }$ </td><td> $0 . 3 7 1 \pm 0 . 0 2 0$ </td><td> $0 . 3 0 1 \pm 0 . 0 3 0$ </td><td> $0 . 2 9 3 \pm 0 . 0 3 2$ </td><td> $0 . 5 7 2 \pm 0 . 0 2 3$ </td><td> $0 . 4 7 4 \pm 0 . 0 4 1$ </td><td> $0 . 4 5 7 \pm 0 . 0 4 2$ </td><td> $0 . 6 7 1 \pm 0 . 0 2 4$ </td><td> $0 . 5 6 8 \pm 0 . 0 4 5$ </td><td> $0 . 5 4 7 \pm 0 . 0 4 7$ </td><td>32</td></tr><tr><td> $W _ { k }$ </td><td> $0 . 4 5 0 \pm 0 . 0 2 4$ </td><td> $0 . 3 8 3 \pm 0 . 0 3 7$ </td><td> $0 . 3 6 6 \pm 0 . 0 3 9$ </td><td> $0 . 6 7 6 \pm 0 . 0 2 0$ </td><td> $0 . 6 0 9 \pm 0 . 0 4 5$ </td><td> $0 . 5 8 4 \pm 0 . 0 4 7$ </td><td> $0 . 7 6 9 \pm 0 . 0 1 6$ </td><td> $0 . 7 0 9 \pm 0 . 0 4 6$ </td><td> $0 . 6 8 5 \pm 0 . 0 4 7$ </td><td>32</td></tr><tr><td> $W _ { v }$ </td><td> $0 . 6 4 3 \pm 0 . 0 7 5$ </td><td> $0 . 5 4 1 \pm 0 . 0 2 4$ </td><td> $0 . 5 4 5 \pm 0 . 0 2 5$ </td><td> $0 . 8 1 7 \pm 0 . 0 6 6$ </td><td> $0 . 7 3 3 \pm 0 . 0 2 6$ </td><td> $0 . 7 3 5 \pm 0 . 0 2 7$ </td><td> $0 . 8 7 6 \pm 0 . 0 5 6$ </td><td> $0 . 8 0 7 \pm 0 . 0 2 5$ </td><td> $0 . 8 0 8 \pm 0 . 0 2 5$ </td><td>32</td></tr><tr><td> $W _ { o }$ </td><td> $0 . 6 0 1 \pm 0 . 0 6 5$ </td><td> $0 . 4 7 3 \pm 0 . 1 0 1$ </td><td> $0 . 4 8 4 \pm 0 . 1 0 9$ </td><td> $0 . 7 6 5 \pm 0 . 0 5 4$ </td><td> $0 . 6 5 1 \pm 0 . 1 0 3$ </td><td> $0 . 6 5 6 \pm 0 . 1 1 9$ </td><td> $0 . 8 2 7 \pm 0 . 0 4 7$ </td><td> $0 . 7 2 8 \pm 0 . 1 0 1$ </td><td> $0 . 7 2 8 \pm 0 . 1 1 8$ </td><td>32</td></tr><tr><td> $W _ { u p }$ </td><td> $0 . 5 8 7 \pm 0 . 0 3 7$ </td><td> $0 . 5 6 4 \pm 0 . 0 3 2$ </td><td> $0 . 5 6 7 \pm 0 . 0 3 5$ </td><td> $0 . 7 7 4 \pm 0 . 0 2 8$ </td><td> $0 . 7 5 4 \pm 0 . 0 2 4$ </td><td> $0 . 7 5 3 \pm 0 . 0 2 6$ </td><td> $0 . 8 4 3 \pm 0 . 0 2 1$ </td><td> $0 . 8 2 6 \pm 0 . 0 1 9$ </td><td> $0 . 8 2 3 \pm 0 . 0 2 1$ </td><td>32</td></tr><tr><td> $W _ { d o w n }$ </td><td> $0 . 5 9 4 \pm 0 . 0 4 6$ </td><td> $0 . 5 2 8 \pm 0 . 1 2 8$ </td><td>0.542 ± 0.130</td><td> $0 . 7 7 3 \pm 0 . 0 4 2$ </td><td> $0 . 6 9 6 \pm 0 . 1 6 1$ </td><td> $0 . 7 0 5 \pm 0 . 1 6 3$ </td><td> $0 . 8 3 9 \pm 0 . 0 3 7$ </td><td> $0 . 7 5 8 \pm 0 . 1 7 2$ </td><td> $0 . 7 6 5 \pm 0 . 1 7 5$ </td><td>32</td></tr><tr><td> $W _ { g a t e }$ </td><td> $0 . 4 4 9 \pm 0 . 0 3 4$ </td><td> $0 . 4 2 9 \pm 0 . 0 3 5$ </td><td> $0 . 4 2 8 \pm 0 . 0 3 7$ </td><td> $0 . 6 2 0 \pm 0 . 0 3 8$ </td><td> $0 . 5 9 5 \pm 0 . 0 4 3$ </td><td> $0 . 5 8 9 \pm 0 . 0 4 5$ </td><td> $0 . 6 8 8 \pm 0 . 0 4 1$ </td><td> $0 . 6 6 3 \pm 0 . 0 4 7$ </td><td> $0 . 6 5 4 \pm 0 . 0 4 9$ </td><td>32</td></tr></table>

## A.2 Llama topology and alpha ablations

Table A.4: TT vs. TTN on WikiText at matched compression (parameter budget). Entries are within-backend medians over the same set of single-operator trials (all blocks and matrices), reported separately for each backend (TT core4, TTN depth4, TTN depth8). This table is not pooled across backends; the main text reports pooled results. AWT value is the lower median across AWT methods.
<table><tr><td>Comp.</td><td>Variant</td><td>PPL  $\left( \mathrm { V a n . } \right)$ </td><td> $\Delta$ </td><td>PPL (AWT)</td><td> $\Delta$ </td></tr><tr><td>2×</td><td>TT (core4)</td><td>8.80</td><td>0.00</td><td>8.78</td><td>0.00</td></tr><tr><td>2×</td><td>TTN (depth4)</td><td>8.73</td><td>-0.07</td><td>8.72</td><td>-0.06</td></tr><tr><td>2×</td><td>TTN (depth8)</td><td>8.74</td><td>-0.06</td><td>8.72</td><td>-0.06</td></tr><tr><td>4×</td><td>TT (core4)</td><td>9.01</td><td>0.00</td><td>8.98</td><td>0.00</td></tr><tr><td>4×</td><td>TTN (depth4)</td><td>8.82</td><td>-0.19</td><td>8.79</td><td>-0.19</td></tr><tr><td>4×</td><td>TTN (depth8)</td><td>8.83</td><td>-0.18</td><td>8.79</td><td>-0.19</td></tr><tr><td>6×</td><td>TT (core4)</td><td>9.10</td><td>0.00</td><td>9.07</td><td>0.00</td></tr><tr><td>6×</td><td>TTN (depth4)</td><td>8.87</td><td>-0.23</td><td>8.84</td><td>-0.23</td></tr><tr><td>6×</td><td>TTN (depth8)</td><td>8.89</td><td>-0.21</td><td>8.84</td><td>-0.23</td></tr></table>

Table A.5: Efect of the reweighting exponent α at $2 \times$ compression. AWT consistently reduces activation-conditioned output error (our functional fidelity proxy), while weight-space error exhibits a diferent optimum (continuing to improve at larger α).
<table><tr><td colspan="2"></td><td colspan="2">Output error (median)</td><td colspan="2">Weight error (median)</td></tr><tr><td>α</td><td>Method</td><td>Value</td><td>∆ vs Van.</td><td>Value</td><td>∆ vs Van.</td></tr><tr><td>0.5</td><td>Std</td><td>0.4471</td><td>-15.7%</td><td>0.5985</td><td>-1.0%</td></tr><tr><td>0.5</td><td>MeanAbs</td><td>0.4430</td><td>-16.4%</td><td>0.5953</td><td>-1.6%</td></tr><tr><td>1.0</td><td>Std</td><td>0.4298</td><td>-18.9%</td><td>0.5500</td><td>-9.1%</td></tr><tr><td>1.0</td><td>MeanAbs</td><td>0.4339</td><td>-18.2%</td><td>0.5309</td><td>-12.2%</td></tr><tr><td>1.5</td><td>Std</td><td>0.4385</td><td>-17.3%</td><td>0.3935</td><td>-34.9%</td></tr><tr><td>1.5</td><td>MeanAbs</td><td>0.4466</td><td>-15.8%</td><td>0.3499</td><td>-42.1%</td></tr></table>

Table A.6: Efect of the reweighting exponent α at 6× compression. AWT consistently reduces activation-conditioned output error (our functional fidelity proxy), while weight-space error exhibits a diferent optimum (continuing to improve at larger α).
<table><tr><td colspan="2"></td><td colspan="2">Output error (median)</td><td colspan="2">Weight error (median)</td></tr><tr><td>α</td><td>Method</td><td>Value</td><td>∆ vs Van.</td><td>Value</td><td>∆ vs Van.</td></tr><tr><td>0.5</td><td>Std</td><td>0.6734</td><td>-9.1%</td><td>0.8212</td><td>-1.4%</td></tr><tr><td>0.5</td><td>MeanAbs</td><td>0.6659</td><td>-10.2%</td><td>0.8180</td><td>-1.8%</td></tr><tr><td>1.0</td><td>Std</td><td>0.6518</td><td>-12.1%</td><td>0.7616</td><td>-8.5%</td></tr><tr><td>1.0</td><td>MeanAbs</td><td>0.6477</td><td>-12.6%</td><td>0.7374</td><td>-11.4%</td></tr><tr><td>1.5</td><td>Std</td><td>0.6606</td><td>-10.9%</td><td>0.5973</td><td>-28.3%</td></tr><tr><td>1.5</td><td>MeanAbs</td><td>0.6630</td><td>-10.5%</td><td>0.5169</td><td>-37.9%</td></tr></table>

Table A.7: Per-matrix Spearman correlation between ∆RelFrob\_out and ∆PPL (AWT−Vanilla), $\alpha = 1 . 0$
<table><tr><td>Matrix</td><td>N Spearman  $\rho$ </td></tr><tr><td>Wdown 440</td><td>0.441</td></tr><tr><td>Wgate</td><td>448 0.026</td></tr><tr><td>Wk 570</td><td>0.490</td></tr><tr><td>Wo</td><td>576 0.284</td></tr><tr><td> $\mathrm { W q }$ </td><td>569 0.702</td></tr><tr><td>Wup 448</td><td>0.383</td></tr><tr><td>Wv</td><td>576 0.281</td></tr></table>

Table A.8: Per-matrix Spearman correlation between ∆Weight Error and ∆PPL (AWT−Vanilla), $\alpha = 1 . 0$
<table><tr><td>Matrix</td><td>N Spearman</td></tr><tr><td>Wdown 440</td><td>0.486</td></tr><tr><td>Wgate</td><td>448 0.129</td></tr><tr><td>Wk 570</td><td>-0.178</td></tr><tr><td>Wo</td><td>576 0.034</td></tr><tr><td>Wq</td><td>569 -0.395</td></tr><tr><td>Wup 448</td><td>0.163</td></tr><tr><td>Wv</td><td>576 -0.056</td></tr></table>

## A.3 Downstream evaluation on Llama 3.1 8B

Table A.9: Downstream single-operator evaluation on Llama 3.1 8B. AWT improves over vanilla tensorization at every tested compression on both HellaSwag and ARC-Challenge. Higher is better.
<table><tr><td>Task</td><td>Comp.</td><td>Dense</td><td>Vanilla</td><td>AWT</td><td>Method</td><td> $\mathrm { G a p }$  closed</td></tr><tr><td>HellaSwag</td><td>2×</td><td>0.7931</td><td>0.7891</td><td>0.7894</td><td>Std</td><td>8.8%</td></tr><tr><td></td><td>4×</td><td>0.7931</td><td>0.7867</td><td>0.7873</td><td>Std</td><td>9.4%</td></tr><tr><td></td><td>6×</td><td>0.7931</td><td>0.7849</td><td>0.7857</td><td>Std</td><td>10.3%</td></tr><tr><td>ARC-Challenge</td><td>2×</td><td>0.5512</td><td>0.5486</td><td>0.5495</td><td>Std</td><td>33.3%</td></tr><tr><td></td><td>4×</td><td>0.5512</td><td>0.5461</td><td>0.5469</td><td>Std</td><td>16.7%</td></tr><tr><td></td><td>6×</td><td>0.5512</td><td>0.5435</td><td>0.5452</td><td>Std</td><td>22.2%</td></tr></table>

SVD baseline  
Table A.10: Backend-specific downstream results on Llama 3.1 8B under single-operator replacement. Entries are within-backend pooled medians over the canonical single-operator benchmark rows. ∆ is measured against TT core4 at the same compression when available. Higher is better.
<table><tr><td>Task</td><td>Comp.</td><td>Backend</td><td>Vanilla</td><td>∆</td><td>AWT</td><td>∆</td></tr><tr><td>HellaSwag</td><td>2×</td><td>TTN depth4</td><td>0.7895</td><td>+0.0007</td><td>0.7901</td><td>+0.0013</td></tr><tr><td>HellaSwag</td><td>2×</td><td>TTN depth8</td><td>0.7895</td><td>+0.0007</td><td>0.7900</td><td>+0.0012</td></tr><tr><td>HellaSwag</td><td>4×</td><td>TT core4</td><td>0.7839</td><td>+0.0000</td><td>0.7850</td><td>+0.0000</td></tr><tr><td>HellaSwag</td><td>4×</td><td>TTN depth4</td><td>0.7876</td><td>+0.0038</td><td>0.7882</td><td>+0.0032</td></tr><tr><td>HellaSwag</td><td>4×</td><td>TTN depth8</td><td>0.7875</td><td>+0.0037</td><td>0.7882</td><td>+0.0032</td></tr><tr><td>HellaSwag</td><td>6×</td><td>TT core4</td><td>0.7817</td><td>+0.0000</td><td>0.7826</td><td>+0.0000</td></tr><tr><td>HellaSwag</td><td>6×</td><td>TTN depth4</td><td>0.7861</td><td>+0.0044</td><td>0.7872</td><td>+0.0046</td></tr><tr><td>HellaSwag</td><td>6×</td><td>TTN depth8</td><td>0.7859</td><td>+0.0042</td><td>0.7869</td><td>+0.0043</td></tr><tr><td>ARC-Challenge</td><td>2×</td><td>TTN depth4</td><td>0.5495</td><td>+0.0017</td><td>0.5495</td><td>+0.0009</td></tr><tr><td>ARC-Challenge</td><td>2×</td><td>TTN depth8</td><td>0.5495</td><td>+0.0017</td><td>0.5495</td><td>+0.0009</td></tr><tr><td>ARC-Challenge</td><td>4×</td><td>TT core4</td><td>0.5418</td><td>+0.0000</td><td>0.5435</td><td>+0.0000</td></tr><tr><td>ARC-Challenge</td><td>4×</td><td>TTN depth4</td><td>0.5478</td><td>+0.0060</td><td>0.5486</td><td>+0.0051</td></tr><tr><td>ARC-Challenge</td><td>4×</td><td>TTN depth8</td><td>0.5478</td><td>+0.0060</td><td>0.5478</td><td>+0.0043</td></tr><tr><td>ARC-Challenge</td><td>6×</td><td>TT core4</td><td>0.5384</td><td>+0.0000</td><td>0.5414</td><td>+0.0000</td></tr><tr><td>ARC-Challenge</td><td>6×</td><td>TTN depth4</td><td>0.5461</td><td>+0.0077</td><td>0.5469</td><td>+0.0055</td></tr><tr><td>ARC-Challenge</td><td>6×</td><td>TTN depth8</td><td>0.5461</td><td>+0.0077</td><td>0.5469</td><td>+0.0055</td></tr></table>

## A.4 SVD and dense full-covariance comparisons

Table A.11: Additional baselines and diagonal-design analysis. Left: matched-compression SVD under the Llama 3.1 8B single-operator protocol. Right: diagonal AWT versus dense full-covariance preconditioning.
<table><tr><td>basCnnC</td><td>TT/TTN AWT</td><td>SVD</td></tr><tr><td>Comp.</td><td>8.79 8.76</td><td>8.71</td></tr><tr><td>2× 4×</td><td>8.90 8.85</td><td>8.81</td></tr><tr><td>6×</td><td>8.97 8.93</td><td>8.87</td></tr></table>

Diagonal vs. full covariance
<table><tr><td>Comparison</td><td>Result</td></tr><tr><td>Diagonal covariance mass,  $\mathrm { L } / \mathrm { M } / \mathrm { Q }$ </td><td>0.032 0.142 / 0.212</td></tr><tr><td>Off-diagonal Frobenius ratio,  $\mathrm { L } / \mathrm { M } / \mathrm { Q }$ </td><td>0.984 / 0.926 / 0.885</td></tr><tr><td>Full-cov better on weighted objective</td><td>80 / 81</td></tr><tr><td>Diagonal better on held-out output error</td><td>53 / 81</td></tr></table>

Table A.12: Diagonal AWT versus dense full-covariance preconditioning. The activation covariance is strongly non-diagonal, and full covariance wins on its own weighted objective, but diagonal AWT more often gives lower held-out local output error.
<table><tr><td>Statistic / comparison</td><td>Result</td></tr><tr><td>Diagonal covariance mass, Llama / Ministral / Qwen</td><td>0.0320 / 0.1415 / 0.2122</td></tr><tr><td>Off-diagonal Frobenius ratio, Llama / Ministral / Qwen</td><td>0.9838 / 0.9255 / 0.8850</td></tr><tr><td>Full-cov better on RelFrobΣ objective</td><td>80 / 81 cases</td></tr><tr><td>Full-cov better on held-out  $\mathrm { R e l F r o b } _ { \mathrm { o u t } }$ </td><td>28 / 81 cases</td></tr><tr><td>Diagonal better on held-out  $\mathrm { R e l F r o b } _ { \mathrm { o u t } }$ </td><td>53 / 81 cases</td></tr><tr><td>Llama: full-cov vs diagonal wins on held-out output error</td><td>13 vs 14</td></tr><tr><td>Ministral: full-cov vs diagonal wins on held-out output error</td><td>10 vs 17</td></tr><tr><td>Qwen: full-cov vs diagonal wins on held-out output error</td><td>5 vs 22</td></tr><tr><td>Mean ∆ output error, Llama / Ministral / Qwen</td><td>-0.0284 / +0.6068 / +0.9093</td></tr></table>

$\Delta = \mathrm { R e l F r o b } _ { \mathrm { o u t } } ( \mathrm { f u l l - c o v } ) - \mathrm { R e l F r o b } _ { \mathrm { o u t } }$ (diagonal), so positive values favor the diagonal method.

## A.5 Cross-model single-operator details

Table A.13: Single-operator WikiText perplexity summaries for Ministral 8B and Qwen2.5 7B. These are the per-family counterparts to the pooled cross-model table in the main text. Lower PPL is better.
<table><tr><td>Model</td><td></td><td>Comp. Dense</td><td>Vanilla AWT</td><td></td><td>Method</td><td>Gap closed</td></tr><tr><td>Ministral 8B</td><td>2×</td><td>8.66</td><td>8.74</td><td>8.71</td><td>Std</td><td>35.3%</td></tr><tr><td></td><td>4×</td><td>8.66</td><td>8.81</td><td>8.76</td><td>Std</td><td>31.3%</td></tr><tr><td></td><td>6×</td><td>8.66</td><td>8.86</td><td>8.80</td><td>Std</td><td>28.6%</td></tr><tr><td>Qwen2.5 7B</td><td>2×</td><td>9.59</td><td>9.76</td><td>9.70</td><td>Std</td><td>33.5%</td></tr><tr><td></td><td>4×</td><td>9.59</td><td>9.85</td><td>9.79</td><td>Std</td><td>25.8%</td></tr><tr><td></td><td>6×</td><td>9.59</td><td>9.91</td><td>9.84</td><td>Std</td><td>22.2%</td></tr></table>

Table A.14: Spearman rank correlation between changes in layer-wise metrics and ∆PPL on WikiText for Ministral 8B and Qwen2.5 7B $( \alpha = 1 . 0 )$ . Groups: $\mathrm { Q K } = \{ W _ { q } , W _ { k } \} , \mathrm { V O } = \{ W _ { v } , W _ { o } \}$ , $\mathrm { M L P } { = } \left\{ W _ { u p } , W _ { d o w n } , W _ { g a t e } \right\}$
<table><tr><td>Model</td><td>Metric</td><td>Overall</td><td>QK</td><td>VO</td><td>MLP</td></tr><tr><td rowspan="2">Ministral 8B</td><td> $\Delta \mathrm { R e l F r o b } _ { \mathrm { o u t } }$ </td><td>0.395</td><td>0.794</td><td>0.245</td><td>0.235</td></tr><tr><td> $\Delta \mathrm { R e l F r o b } _ { \mathrm { w } }$ </td><td>0.042</td><td>-0.299</td><td>0.438</td><td>-0.072</td></tr><tr><td rowspan="2">Qwen2.5 7B</td><td> $\Delta \mathrm { R e l F r o b } _ { \mathrm { o u t } }$ </td><td>0.327</td><td>0.401</td><td>0.458</td><td>0.165</td></tr><tr><td> $\Delta \mathrm { R e l F r o b } _ { \mathrm { w } }$ </td><td>-0.049</td><td>-0.213</td><td>-0.260</td><td>0.164</td></tr></table>

## A.6 Cross-model alpha ablations

Table A.15: Alpha ablations for Ministral 8B. Entries report pooled medians over single-operator sweep rows. Lower error is better.
<table><tr><td></td><td></td><td colspan="4">2×</td><td colspan="4">4×</td><td colspan="4">6×</td></tr><tr><td>α</td><td>Method</td><td>Out</td><td></td><td>∆ Weight</td><td>∆</td><td>Out</td><td></td><td>∆ Weight</td><td>∆</td><td>Out</td><td></td><td>∆ Weight</td><td>∆</td></tr><tr><td>0.5 Std</td><td></td><td>0.398-15.1</td><td></td><td>0.585</td><td>+0.1</td><td>0.518-14.7</td><td></td><td>0.736</td><td>-2.7</td><td>0.584-12.2</td><td></td><td>0.805</td><td>-2.4</td></tr><tr><td></td><td>0.5 MeanAbs 0.395 -15.8</td><td></td><td></td><td>0.582</td><td></td><td>-0.4 0.517 -14.9</td><td></td><td>0.734</td><td>-3.0</td><td>0.582 -12.5</td><td></td><td>0.802</td><td>-2.7</td></tr><tr><td>1.0 Std</td><td></td><td>0.376-19.9</td><td></td><td>0.525</td><td>-10.2</td><td>0.500-17.7</td><td></td><td>0.660</td><td>-12.8</td><td>0.562-15.5</td><td></td><td>0.726-12.0</td><td></td></tr><tr><td></td><td>1.0 MeanAbs</td><td>0.377-19.7</td><td></td><td>0.502</td><td>-14.1</td><td>0.500-17.8</td><td></td><td>0.635</td><td>-16.1</td><td>0.564-15.2</td><td></td><td>0.699-15.2</td><td></td></tr><tr><td>1.5 Std</td><td></td><td>0.380-18.9</td><td></td><td>0.365 -37.5 0.505 -16.9</td><td></td><td></td><td></td><td>0.477</td><td>-36.9</td><td>0.570-14.2</td><td></td><td>0.549-33.5</td><td></td></tr><tr><td></td><td>1.5 MeanAbs 0.387 -17.5</td><td></td><td></td><td>0.285 -51.3 0.509 -16.2</td><td></td><td></td><td></td><td>0.379-49.9 0.574-13.6</td><td></td><td></td><td></td><td>0.430-47.8</td><td></td></tr></table>

Vanilla pooled medians: output / weight error = 0.469/0.585 at 2×, 0.608/0.756 at 4×, and 0.665/0.824 at 6×. ∆ values are percentages.

Table A.16: Alpha ablations for Qwen2.5 7B. Entries report pooled medians over single-operator sweep rows. Lower error is better.
<table><tr><td rowspan="2">Method α</td><td rowspan="2"></td><td colspan="4">2×</td><td colspan="4">4×</td><td colspan="4">6×</td></tr><tr><td>Out</td><td></td><td>Δ Weight</td><td>∆</td><td>Out</td><td></td><td>∆ Weight</td><td>∆</td><td>Out</td><td></td><td>Δ Weight</td><td>∆</td></tr><tr><td>0.5 Std</td><td></td><td>0.377-18.6</td><td></td><td>0.594</td><td>+0.8</td><td>0.522 -16.8</td><td></td><td>0.745</td><td>-2.6</td><td></td><td>0.590-15.1</td><td>0.814</td><td>-2.2</td></tr><tr><td></td><td>0.5 MeanAbs 0.375 -19.1</td><td></td><td></td><td>0.588</td><td>-0.1</td><td>0.515-17.9</td><td></td><td>0.740</td><td>-3.3</td><td>0.579-16.7</td><td></td><td>0.808</td><td>-3.0</td></tr><tr><td>1.0 Std</td><td></td><td>0.366-21.1</td><td></td><td>0.487 -17.3 0.501 -20.0</td><td></td><td></td><td></td><td>0.627</td><td>-18.1</td><td>0.568-18.4</td><td></td><td>0.702-15.7</td><td></td></tr><tr><td></td><td>1.0 MeanAbs</td><td>0.366-21.0</td><td></td><td>0.445</td><td>-24.4 0.505-19.4</td><td></td><td></td><td>0.578</td><td>-24.4</td><td>0.568-18.4</td><td></td><td>0.654-21.4</td><td></td></tr><tr><td>1.5 Std</td><td></td><td>0.395-14.8</td><td></td><td>0.271-54.00.520-16.9</td><td></td><td></td><td></td><td>0.364 -52.5 0.582 -16.3</td><td></td><td></td><td></td><td>0.409-50.9</td><td></td></tr><tr><td></td><td>1.5 MeanAbs 0.405 -12.6</td><td></td><td></td><td>0.196-66.7 0.527 -16.0</td><td></td><td></td><td></td><td>0.286-62.7 0.587-15.6</td><td></td><td></td><td></td><td>0.332-60.1</td><td></td></tr></table>

Vanilla pooled medians: output / weight error = 0.463/0.589 at 2×, 0.627/0.765 at 4×, and 0.695/0.833 at 6×. ∆ values are percentages.

## A.7 Cross-model backend comparisons

Table A.17: Backend comparison on WikiText under single-operator replacement for Ministral 8B. Entries are within-backend pooled medians. ∆ is measured against TT core4 at the same compression. Lower PPL is better.
<table><tr><td>Comp.</td><td>Backend</td><td>PPL (Van.)</td><td>∆</td><td>PPL (AWT)</td><td>∆</td></tr><tr><td>2×</td><td>TT core4</td><td>8.75</td><td>+0.00</td><td>8.71</td><td>+0.00</td></tr><tr><td>2×</td><td>TTN depth4</td><td>8.72</td><td>-0.03</td><td>8.70</td><td>-0.01</td></tr><tr><td>2×</td><td>TTN depth8</td><td>8.72</td><td>-0.03</td><td>8.70</td><td>-0.01</td></tr><tr><td>4×</td><td>TT core4</td><td>8.86</td><td>+0.00</td><td>8.79</td><td>+0.00</td></tr><tr><td>4×</td><td>TTN depth4</td><td>8.77</td><td>-0.09</td><td>8.75</td><td>-0.04</td></tr><tr><td>4×</td><td>TTN depth8</td><td>8.78</td><td>-0.08</td><td>8.75</td><td>-0.04</td></tr><tr><td>6×</td><td>TT core4</td><td>8.92</td><td>+0.00</td><td>8.84</td><td>+0.00</td></tr><tr><td>6×</td><td>TTN depth4</td><td>8.81</td><td>-0.11</td><td>8.78</td><td>-0.06</td></tr><tr><td>6×</td><td>TTN depth8</td><td>8.81</td><td>-0.10</td><td>8.78</td><td>-0.06</td></tr></table>

Table A.18: Backend comparison on WikiText under single-operator replacement for Qwen2.5 7B. Entries are within-backend pooled medians. ∆ is measured against TT core4 at the same compression. Lower PPL is better.
<table><tr><td>Comp.</td><td>Backend</td><td>PPL (Van.)</td><td>Δ PPL (AWT)</td><td></td><td>∆</td></tr><tr><td>2×</td><td>TT core4</td><td>9.75</td><td>+0.00</td><td>9.72</td><td>+0.00</td></tr><tr><td>2×</td><td>TTN depth4</td><td>9.76</td><td>+0.01</td><td>9.69</td><td>-0.02</td></tr><tr><td>2×</td><td>TTN depth8</td><td>9.78</td><td>+0.02</td><td>9.70</td><td>-0.02</td></tr><tr><td>4×</td><td>TT core4</td><td>9.88</td><td>+0.00</td><td>9.82</td><td>+0.00</td></tr><tr><td>4×</td><td>TTN depth4</td><td>9.84</td><td>-0.04</td><td>9.75</td><td>-0.07</td></tr><tr><td>4×</td><td>TTN depth8</td><td>9.85</td><td>-0.03</td><td>9.77</td><td>-0.05</td></tr><tr><td>6×</td><td>TT core4</td><td>9.95</td><td>+0.00</td><td>9.89</td><td>+0.00</td></tr><tr><td>6×</td><td>TTN depth4</td><td>9.88</td><td>-0.07</td><td>9.81</td><td>-0.08</td></tr><tr><td>6×</td><td>TTN depth8</td><td>9.91</td><td>-0.04</td><td>9.81</td><td>-0.08</td></tr></table>

## A.8 Multi-operator replacement details

Table A.19: Multi-operator WikiText sufix replacement on Llama 3.1 8B. The attention group replaces $\{ W _ { q } , W _ { k } , W _ { v } , W _ { o } \}$ jointly in the last k blocks. Lower PPL is better; negative ∆PPL favors AWT.
<table><tr><td>k</td><td>Comp.</td><td>Vanilla PPL</td><td>AWT PPL</td><td>∆PPL</td><td>Gap closed</td><td>Finite backends</td></tr><tr><td>1 2×</td><td></td><td>9.33</td><td>8.92</td><td>-0.41</td><td>59.5%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>1</td><td>4×</td><td>9.58</td><td>9.12</td><td>-0.47</td><td>49.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>1</td><td>6×</td><td>9.71</td><td>9.33</td><td>-0.46</td><td>35.3%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>2×</td><td>9.78</td><td>9.19</td><td>-0.59</td><td>51.8%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>4×</td><td>10.28</td><td>9.60</td><td>-0.72</td><td>41.3%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>6×</td><td>10.56</td><td>9.91</td><td>-0.81</td><td>33.9%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>2×</td><td>10.56</td><td>9.80</td><td>-0.76</td><td>39.4%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>4×</td><td>11.93</td><td>10.92</td><td>-1.02</td><td>30.8%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>6×</td><td>12.68</td><td>11.57</td><td>-1.11</td><td>27.4%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>2×</td><td>11.85</td><td>10.45</td><td>-1.40</td><td>43.6%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>4×</td><td>15.55</td><td>13.08</td><td>-2.48</td><td>35.8%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>6×</td><td>17.55</td><td>14.81</td><td>-2.75</td><td>30.8%</td><td>TT core4, TTN depth4, TTN depth8</td></tr></table>

Table A.20: Multi-operator WikiText sufix replacement on Llama 3.1 8B when all seven dense matrices are replaced jointly in the last k blocks. Lower PPL is better; negative ∆PPL favors AWT.
<table><tr><td></td><td>k Comp.</td><td>Vanilla PPL</td><td>AWT PPL</td><td>∆PPL</td><td>Gap closed</td></tr><tr><td>1</td><td>2×</td><td>13.87</td><td>11.99</td><td>-1.86</td><td>36.0%</td></tr><tr><td>1</td><td>4×</td><td>15.54</td><td>12.76</td><td>-2.61</td><td>40.2%</td></tr><tr><td>1</td><td>6×</td><td>17.65</td><td>13.60</td><td>-3.62</td><td>45.0%</td></tr><tr><td>2</td><td>2×</td><td>21.53</td><td>13.79</td><td>-7.74</td><td>60.0%</td></tr><tr><td>2</td><td>4×</td><td>24.47</td><td>15.09</td><td>-9.21</td><td>59.3%</td></tr><tr><td>2</td><td>6×</td><td>30.18</td><td>17.28</td><td>-11.71</td><td>59.9%</td></tr></table>

Table A.21: Multi-operator WikiText sufix replacement on Llama 3.1 8B when the MLP group $\{ W _ { u p } , W _ { g a t e } , W _ { d o w n } \}$ is replaced jointly. Empty rows in the source runs correspond to non-finite PPL. Lower PPL is better; negative ∆PPL favors AWT.
<table><tr><td>k Comp. Vanilla PPL</td></tr><tr><td></td><td>AWT PPL 12.27 12.73 -1.45</td><td>∆PPL -1.27</td><td>Gap closed Finite backends 26.0% TT core4, TTN depth4, TTN depth8</td></tr><tr><td>1 2× 1 4×</td><td>13.54 14.19</td></tr><tr><td>6×</td><td>26.4% TT core4, TTN depth4, TTN depth8 24.9%</td></tr><tr><td>1 14.78 13.25 -1.53 17.65 14.21 -3.43</td><td>TT core4, TTN depth4, TTN depth8 TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2 2× 2 4×</td><td>38.1%</td></tr><tr><td>18.52 14.67 -3.79 39.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>19.78</td><td>15.30 -4.47 40.2% TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2 6× 4 2× 4 4×</td><td>30.38 17.41 -12.97 59.6% TT core4, TTN depth4, TTN depth8 61.8%</td></tr><tr><td>34.02 18.33 -15.69 39.31 19.61</td><td>TTN depth4, TTN depth8 TTN depth4, TTN depth8</td></tr><tr><td>4 6× 8 2× 36.21</td><td>-19.70 64.2% 38.31 +2.09 -7.6% TT core4</td></tr></table>

## A.9 Multi-operator BoolQ details

Table A.22: Multi-operator BoolQ sufix replacement on Llama 3.1 8B. Attention group replaces $\{ W _ { q } , W _ { k } , W _ { v } , W _ { o } \}$ jointly in the last k blocks. Higher is better; positive $\Delta$ favors AWT.
<table><tr><td>k</td><td>Comp.</td><td>Vanilla</td><td>AWT</td><td> $\Delta$ </td><td>Error reduction vs vanilla</td><td>Finite backends</td></tr><tr><td>1</td><td>2×</td><td>0.7900</td><td>0.8000</td><td>+0.0100</td><td>4.8%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>1</td><td>4×</td><td>0.8000</td><td>0.7900</td><td>-0.0100</td><td>-5.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>1</td><td>6×</td><td>0.8000</td><td>0.8000</td><td>+0.0000</td><td>0.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>2×</td><td>0.8000</td><td>0.8100</td><td>+0.0100</td><td>5.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>4×</td><td>0.8100</td><td>0.8300</td><td>+0.0200</td><td>10.5%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>6×</td><td>0.8100</td><td>0.8100</td><td>+0.0000</td><td>0.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>2×</td><td>0.8000</td><td>0.8300</td><td>+0.0200</td><td>15.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>4×</td><td>0.8200</td><td>0.8300</td><td>+0.0100</td><td>5.6%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>6×</td><td>0.8100</td><td>0.8300</td><td>+0.0200</td><td>10.5%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>2×</td><td>0.8100</td><td>0.8300</td><td>+0.0200</td><td>10.5%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>4×</td><td>0.8200</td><td>0.8300</td><td>+0.0100</td><td>5.6%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>6×</td><td>0.8300</td><td>0.8400</td><td>+0.0300</td><td>5.9%</td><td>TT core4, TTN depth4, TTN depth8</td></tr></table>

Table A.23: Multi-operator BoolQ sufix replacement on Llama 3.1 8B when all seven dense matrices are replaced jointly in the last k blocks. Higher is better.
<table><tr><td>k</td><td>Comp. Vanilla</td><td>AWT</td><td>∆ Error reduction vs vanilla</td></tr><tr><td>1 2×</td><td>0.8100</td><td>0.8300 +0.0200</td><td>10.5%</td></tr><tr><td>1 4×</td><td>0.7900</td><td>0.8200 +0.0300</td><td>14.3%</td></tr><tr><td>1 6×</td><td>0.7600</td><td>0.8200 +0.0600</td><td>25.0%</td></tr><tr><td>2 2×</td><td>0.7900</td><td>0.7800 -0.0100</td><td>-4.8%</td></tr><tr><td>2 4×</td><td>0.7800</td><td>0.7900 +0.0100</td><td>4.5%</td></tr><tr><td>2 6×</td><td>0.7900</td><td>0.7800 +0.0000</td><td>-4.8%</td></tr></table>

Table A.24: Multi-operator BoolQ sufix replacement on Llama 3.1 8B when the MLP group $\{ W _ { u p } , W _ { g a t e } , W _ { d o w n } \}$ is replaced jointly. Higher is better.
<table><tr><td>k</td><td>Comp.</td><td>Vanilla</td><td>AWT</td><td> $\Delta$ </td><td>Error reduction vs vanilla</td><td>Finite backends</td></tr><tr><td>1</td><td>2×</td><td>0.8200</td><td>0.8200</td><td>+0.0000</td><td>0.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>1</td><td>4×</td><td>0.8100</td><td>0.8200</td><td>+0.0100</td><td>5.3%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>1</td><td>6×</td><td>0.8100</td><td>0.8300</td><td>+0.0200</td><td>10.5%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>2×</td><td>0.7800</td><td>0.7800</td><td>+0.0000</td><td>0.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>4×</td><td>0.7800</td><td>0.7800</td><td>+0.0000</td><td>0.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>2</td><td>6×</td><td>0.7900</td><td>0.7800</td><td>-0.0100</td><td>-4.8%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>2×</td><td>0.7900</td><td>0.7900</td><td>+0.0000</td><td>0.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>4×</td><td>0.7700</td><td>0.7800</td><td>+0.0100</td><td>4.3%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>4</td><td>6×</td><td>0.7600</td><td>0.7900</td><td>+0.0300</td><td>12.5%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>2×</td><td>0.7000</td><td>0.7100</td><td>+0.0100</td><td>3.3%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>4×</td><td>0.7000</td><td>0.7000</td><td>+0.0000</td><td>0.0%</td><td>TT core4, TTN depth4, TTN depth8</td></tr><tr><td>8</td><td>6×</td><td>0.7000</td><td>0.7200</td><td>+0.0200</td><td>6.7%</td><td>TT core4, TTN depth4, TTN depth8</td></tr></table>

## A.10 Per-operator output-error breakdowns for Ministral and Qwen

Table A.25: Per-operator activation-conditioned output error for Ministral 8B under TTN depth4 tensorization with $\alpha = 1$ . Entries report mean $\mu$ and standard deviation σ over per-block sweep values. Lower is better.
<table><tr><td></td><td colspan="3">2× compression</td><td colspan="3">4× compression</td><td colspan="3">6× compression</td><td></td></tr><tr><td>Matrix</td><td> $\mathrm { V a n i l l a } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td> ${ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )$ </td><td> $\mathrm { V a n i l l a } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { M e a n A b s } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { V a n i l l a } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td>MeanAbs (µ±σ) N</td><td></td></tr><tr><td> $W _ { q }$ </td><td> $0 . 1 9 2 \pm 0 . 0 4 4$ </td><td> $0 . 1 5 8 \pm 0 . 0 3 8$ </td><td> $0 . 1 5 9 \pm 0 . 0 3 8$ </td><td> $0 . 2 5 8 \pm 0 . 0 5 2$ </td><td> $0 . 2 1 7 \pm 0 . 0 4 6$ </td><td> $0 . 2 1 8 \pm 0 . 0 4 6$ </td><td> $0 . 2 9 1 \pm 0 . 0 5 4$ </td><td> $0 . 2 4 8 \pm 0 . 0 4 9$ </td><td> $0 . 2 4 9 \pm 0 . 0 4 9$ </td><td>36</td></tr><tr><td> $W _ { k }$ </td><td> $0 . 1 9 1 \pm 0 . 0 3 7$ </td><td> $0 . 1 5 1 \pm 0 . 0 3 2$ </td><td> $0 . 1 5 2 \pm 0 . 0 3 2$ </td><td> $0 . 2 8 9 \pm 0 . 0 4 4$ </td><td> $0 . 2 3 1 \pm 0 . 0 3 9$ </td><td> $0 . 2 3 3 \pm 0 . 0 3 9$ </td><td> $0 . 3 3 9 \pm 0 . 0 4 4$ </td><td> $0 . 2 7 2 \pm 0 . 0 4 1$ </td><td> $0 . 2 7 5 \pm 0 . 0 4 1$ </td><td>36</td></tr><tr><td> $W _ { v }$ </td><td> $0 . 5 5 8 \pm 0 . 0 8 0$ </td><td> $0 . 4 5 8 \pm 0 . 0 6 1$ </td><td> $0 . 4 5 8 \pm 0 . 0 6 0$ </td><td> $0 . 7 0 8 \pm 0 . 0 8 1$ </td><td> $0 . 6 0 8 \pm 0 . 0 6 8$ </td><td> $0 . 6 0 8 \pm 0 . 0 6 8$ </td><td> $0 . 7 7 0 \pm 0 . 0 7 9$ </td><td> $0 . 6 7 3 \pm 0 . 0 6 7$ </td><td> $0 . 6 7 5 \pm 0 . 0 6 7$ </td><td>36</td></tr><tr><td> $W _ { o }$ </td><td> $0 . 4 9 2 \pm 0 . 1 1 1$ </td><td> $0 . 3 7 5 \pm 0 . 0 9 9$ </td><td> $0 . 3 8 0 \pm 0 . 0 9 7$ </td><td> $0 . 6 2 5 \pm 0 . 1 3 5$ </td><td> $0 . 5 0 6 \pm 0 . 1 2 4$ </td><td> $0 . 5 0 8 \pm 0 . 1 2 2$ </td><td> $0 . 6 8 0 \pm 0 . 1 4 4$ </td><td> $0 . 5 6 9 \pm 0 . 1 3 5$ </td><td> $0 . 5 7 1 \pm 0 . 1 3 3$ </td><td>36</td></tr><tr><td> $W _ { u p }$ </td><td> $0 . 5 5 9 \pm 0 . 0 7 8$ </td><td> $0 . 5 2 0 \pm 0 . 0 7 9$ </td><td>0.522 ± 0.080</td><td> $0 . 6 1 7 \pm 0 . 0 8 7$ </td><td> $0 . 5 7 7 \pm 0 . 0 8 8$ </td><td> $0 . 5 7 9 \pm 0 . 0 8 8$ </td><td> $0 . 6 6 9 \pm 0 . 0 9 4$ </td><td> $0 . 6 2 8 \pm 0 . 0 9 5$ </td><td> $0 . 6 3 1 \pm 0 . 0 9 5$ </td><td>36</td></tr><tr><td> $W _ { d o w n }$ </td><td> $0 . 6 5 5 \pm 0 . 0 9 0$ </td><td> $0 . 5 0 7 \pm 0 . 1 4 9$ </td><td> $0 . 5 2 3 \pm 0 . 1 5 3$ </td><td> $0 . 7 1 6 \pm 0 . 0 9 6$ </td><td> $0 . 5 6 8 \pm 0 . 1 6 6$ </td><td> $0 . 5 8 5 \pm 0 . 1 7 1$ </td><td> $0 . 7 6 9 \pm 0 . 1 0 1$ </td><td> $0 . 6 2 5 \pm 0 . 1 8 0$ </td><td> $0 . 6 4 2 \pm 0 . 1 8 6$ </td><td>36</td></tr><tr><td> $W _ { g a t e }$ </td><td> $0 . 3 4 8 \pm 0 . 0 5 0$ </td><td> $0 . 3 2 8 \pm 0 . 0 5 0$ </td><td> $0 . 3 2 9 \pm 0 . 0 5 0$ </td><td> $0 . 3 8 7 \pm 0 . 0 5 5$ </td><td> $0 . 3 6 6 \pm 0 . 0 5 6$ </td><td> $0 . 3 6 8 \pm 0 . 0 5 6$ </td><td> $0 . 4 2 3 \pm 0 . 0 6 1$ </td><td> $0 . 4 0 3 \pm 0 . 0 6 1$ </td><td> $0 . 4 0 4 \pm 0 . 0 6 1$ </td><td>36</td></tr></table>

Table A.26: Per-operator activation-conditioned output error for Ministral 8B under TTN depth8 tensorization with $\alpha = 1$ . Entries report mean $\mu$ and standard deviation $\sigma$ over per-block sweep values. Lower is better.
<table><tr><td rowspan="2"></td><td colspan="3">2× compression</td><td colspan="3">4× compression</td><td colspan="3">6× compression</td><td></td></tr><tr><td></td><td>Matrix Vanilla (µ±σ) Std (µ±σ)</td><td> ${ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )$ </td><td> $\mathrm { V a n i l l a } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { M e a n A b s } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { V a n i l l a } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td> ${ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )$ </td><td>N</td></tr><tr><td> $W _ { q }$ </td><td> $0 . 1 9 3 \pm 0 . 0 4 4$ </td><td> $0 . 1 6 0 \pm 0 . 0 3 8$ </td><td> $0 . 1 6 0 \pm 0 . 0 3 8$ </td><td> $0 . 2 6 0 \pm 0 . 0 5 2$ </td><td> $0 . 2 2 0 \pm 0 . 0 4 6$ </td><td> $0 . 2 2 0 \pm 0 . 0 4 6$ </td><td> $0 . 2 9 4 \pm 0 . 0 5 4$ </td><td> $0 . 2 5 1 \pm 0 . 0 4 9$ </td><td> $0 . 2 5 2 \pm 0 . 0 4 9$ </td><td>36</td></tr><tr><td> $W _ { k }$ </td><td> $0 . 1 9 5 \pm 0 . 0 3 8$ </td><td>0.155 ± 0.032</td><td> $0 . 1 5 6 \pm 0 . 0 3 2$ </td><td> $0 . 2 9 3 \pm 0 . 0 4 4$ </td><td> $0 . 2 3 5 \pm 0 . 0 3 8$ </td><td> $0 . 2 3 6 \pm 0 . 0 3 9$ </td><td> $0 . 3 3 8 \pm 0 . 0 4 3$ </td><td> $0 . 2 7 4 \pm 0 . 0 3 9$ </td><td> $0 . 2 7 6 \pm 0 . 0 4 0$ </td><td>36</td></tr><tr><td> $W _ { v }$ </td><td> $0 . 5 6 6 \pm 0 . 0 8 0$ </td><td> $0 . 4 6 6 \pm 0 . 0 6 1$ </td><td> $0 . 4 6 5 \pm 0 . 0 6 1$ </td><td> $0 . 7 1 8 \pm 0 . 0 7 9$ </td><td> $0 . 6 1 2 \pm 0 . 0 6 6$ </td><td> $0 . 6 1 3 \pm 0 . 0 6 6$ </td><td> $0 . 7 7 8 \pm 0 . 0 7 7$ </td><td> $0 . 6 7 2 \pm 0 . 0 6 6$ </td><td> $0 . 6 7 3 \pm 0 . 0 6 6$ </td><td>36</td></tr><tr><td> $W _ { o }$ </td><td> $0 . 4 9 5 \pm 0 . 1 1 2$ </td><td> $0 . 3 7 8 \pm 0 . 1 0 0$ </td><td> $0 . 3 8 3 \pm 0 . 0 9 7$ </td><td> $0 . 6 2 9 \pm 0 . 1 3 6$ </td><td> $0 . 5 1 0 \pm 0 . 1 2 5$ </td><td> $0 . 5 1 3 \pm 0 . 1 2 3$ </td><td> $0 . 6 8 5 \pm 0 . 1 4 5$ </td><td> $0 . 5 7 5 \pm 0 . 1 3 6$ </td><td> $0 . 5 7 7 \pm 0 . 1 3 5$ </td><td>36</td></tr><tr><td> $W _ { u p }$ </td><td> $0 . 5 5 9 \pm 0 . 0 7 8$ </td><td> $0 . 5 2 1 \pm 0 . 0 7 9$ </td><td> $0 . 5 2 2 \pm 0 . 0 8 0$ </td><td> $0 . 6 1 8 \pm 0 . 0 8 7$ </td><td> $0 . 5 7 8 \pm 0 . 0 8 8$ </td><td> $0 . 5 8 0 \pm 0 . 0 8 8$ </td><td> $0 . 6 7 0 \pm 0 . 0 9 4$ </td><td> $0 . 6 3 0 \pm 0 . 0 9 5$ </td><td> $0 . 6 3 2 \pm 0 . 0 9 5$ </td><td>36</td></tr><tr><td> $W _ { d o w n }$ </td><td> $0 . 6 5 5 \pm 0 . 0 9 0$ </td><td> $0 . 5 0 7 \pm 0 . 1 4 9$ </td><td> $0 . 5 2 3 \pm 0 . 1 5 3$ </td><td> $0 . 7 1 7 \pm 0 . 0 9 6$ </td><td> $0 . 5 6 9 \pm 0 . 1 6 6$ </td><td> $0 . 5 8 6 \pm 0 . 1 7 1$ </td><td> $0 . 7 7 0 \pm 0 . 1 0 1$ </td><td> $0 . 6 2 6 \pm 0 . 1 8 0$ </td><td> $0 . 6 4 4 \pm 0 . 1 8 6$ </td><td>36</td></tr><tr><td> $W _ { g a t e }$ </td><td> $0 . 3 4 8 \pm 0 . 0 5 0$ </td><td> $0 . 3 2 8 \pm 0 . 0 5 0$ </td><td>0.329 ± 0.050</td><td> $0 . 3 8 8 \pm 0 . 0 5 5$ </td><td> $0 . 3 6 7 \pm 0 . 0 5 6$ </td><td> $0 . 3 6 8 \pm 0 . 0 5 6$ </td><td> $0 . 4 2 4 \pm 0 . 0 6 1$ </td><td> $0 . 4 0 4 \pm 0 . 0 6 1$ </td><td> $0 . 4 0 5 \pm 0 . 0 6 1$ </td><td>36</td></tr></table>

Table A.27: Per-operator activation-conditioned output error for Ministral 8B under TT core4 tensorization with α = 1. Entries report mean µ and standard deviation σ over per-block sweep values. Lower is better.

$$
0 . 3 3 3 \pm 0 . 0 1 9
$$

$$
0 . 2 2 8 \pm 0 . 0 2 4
$$

$$
0 . 3 2 1 \pm 0 . 0 4 4
$$

$$
0 . 2 2 7 \pm 0 . 0 2 4
$$

$$
0 . 2 1 9 \pm 0 . 0 2 2
$$

$$
0 . 2 1 9 \pm 0 . 0 2 2
$$

$$
0 . 4 8 9 \pm 0 . 0 2 4
$$

$$
0 . 3 3 1 \pm 0 . 0 2 8
$$

$$
0 . 5 0 0 \pm 0 . 0 6 5
$$

$$
0 . 3 3 1 \pm 0 . 0 2 9
$$

$$
0 . 3 2 6 \pm 0 . 0 3 1
$$

$$
0 . 5 6 5 \pm 0 . 0 2 9
$$

$$
0 . 3 8 2 \pm 0 . 0 3 2
$$

$$
0 . 3 8 1 \pm 0 . 0 3 2
$$

$$
0 . 4 4 8 \pm 0 . 0 4 3
$$

$$
0 . 6 1 6 \pm 0 . 0 3 9
$$

$$
0 . 3 2 5 \pm 0 . 0 3 1
$$

$$
0 . 7 8 0 \pm 0 . 0 4 6
$$

$$
0 . 5 8 1 \pm 0 . 0 7 1
$$

$$
0 . 3 7 8 \pm 0 . 0 3 7
$$

$$
0 . 3 7 7 \pm 0 . 0 3 6
$$

$$
0 . 4 4 9 \pm 0 . 1 1 9
$$

$$
0 . 5 5 1 \pm 0 . 1 2 4
$$

$$
0 . 6 1 1 \pm 0 . 0 5 7
$$

$$
0 . 4 6 6 \pm 0 . 1 2 1
$$

$$
0 . 6 1 1 \pm 0 . 0 5 7
$$

$$
0 . 6 9 8 \pm 0 . 1 4 9
$$

$$
0 . 8 4 1 \pm 0 . 0 4 5
$$

$$
0 . 6 7 7 \pm 0 . 0 6 1
$$

$$
0 . 6 0 9 \pm 0 . 1 4 4
$$

$$
0 . 6 7 7 \pm 0 . 0 6 1
$$

$$
0 . 4 9 3 \pm 0 . 0 6 1
$$

$$
0 . 5 7 5 \pm 0 . 0 4 3
$$

$$
W _ { u p }
$$

$$
0 . 6 2 0 \pm 0 . 1 4 8
$$

$$
0 . 7 5 3 \pm 0 . 1 5 8
$$

$$
0 . 6 7 7 \pm 0 . 1 5 3
$$

$$
W _ { d o w n }
$$

$$
0 . 7 4 8 \pm 0 . 0 3 8
$$

$$
0 . 6 8 4 \pm 0 . 1 5 7
$$

$$
0 . 5 7 0 \pm 0 . 0 8 0
$$

$$
0 . 4 7 0 \pm 0 . 1 3 9
$$

$$
0 . 6 5 2 \pm 0 . 0 7 2
$$

$$
0 . 4 6 4 \pm 0 . 1 3 7
$$

$$
0 . 3 3 1 \pm 0 . 0 5 3
$$

$$
W _ { g a t e }
$$

$$
0 . 3 3 1 \pm 0 . 0 5 3
$$

$$
0 . 7 2 2 \pm 0 . 1 2 4
$$

$$
0 . 8 1 4 \pm 0 . 0 3 3
$$

$$
0 . 6 2 6 \pm 0 . 1 7 7
$$

$$
0 . 7 1 3 \pm 0 . 0 7 4
$$

$$
0 . 5 3 5 \pm 0 . 0 4 8
$$

$$
0 . 6 2 3 \pm 0 . 1 7 5
$$

$$
0 . 4 4 6 \pm 0 . 0 7 1
$$

$$
0 . 7 1 2 \pm 0 . 0 7 4
$$

$$
0 . 7 7 7 \pm 0 . 1 4 2
$$

$$
0 . 4 4 5 \pm 0 . 0 7 1
$$

$$
0 . 6 8 8 \pm 0 . 1 8 9
$$

$$
0 . 5 9 2 \pm 0 . 0 5 1
$$

$$
0 . 6 8 7 \pm 0 . 1 8 7
$$

$$
0 . 4 9 3 \pm 0 . 0 7 7
$$

$$
0 . 4 9 2 \pm 0 . 0 7 7
$$

Table A.28: Per-operator activation-conditioned output error for Qwen2.5 7B under TTN depth4 tensorization with α = 1. Entries report mean µ and standard deviation σ over per-block sweep values. Lower is better.

$$
{ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )
$$

$$
\mathrm { S t d } \ ( \mu \pm \sigma )
$$

$$
\mathrm { M e a n A b s } \ ( \mu \pm \sigma )
$$

$$
\mathrm { V a n i l l a } \ ( \mu \pm \sigma )
$$

$$
{ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )
$$

$$
0 . 3 3 1 \pm 0 . 0 7 9
$$

$$
0 . 2 2 2 \pm 0 . 0 6 9
$$

$$
0 . 2 2 3 \pm 0 . 0 6 9
$$

$$
0 . 4 4 6 \pm 0 . 0 9 9
$$

$$
0 . 3 5 4 \pm 0 . 0 6 0
$$

$$
0 . 3 1 3 \pm 0 . 0 8 9
$$

$$
0 . 3 1 3 \pm 0 . 0 8 9
$$

$$
0 . 2 0 8 \pm 0 . 0 4 8
$$

$$
0 . 5 0 5 \pm 0 . 1 0 9
$$

$$
0 . 5 6 6 \pm 0 . 0 7 6
$$

$$
0 . 3 6 3 \pm 0 . 0 9 9
$$

$$
0 . 6 6 5 \pm 0 . 0 9 3
$$

$$
0 . 3 4 6 \pm 0 . 0 6 1
$$

$$
0 . 4 6 9 \pm 0 . 1 1 9
$$

$$
0 . 3 6 3 \pm 0 . 0 9 9
$$

$$
0 . 3 4 7 \pm 0 . 0 6 0
$$

$$
0 . 4 6 8 \pm 0 . 1 1 8
$$

$$
0 . 6 5 2 \pm 0 . 0 7 5
$$

$$
0 . 4 1 9 \pm 0 . 0 6 3
$$

$$
0 . 8 2 3 \pm 0 . 0 7 2
$$

$$
0 . 6 4 7 \pm 0 . 0 9 7
$$

$$
0 . 4 7 3 \pm 0 . 0 9 7
$$

$$
0 . 4 1 9 \pm 0 . 0 6 3
$$

$$
0 . 3 7 5 \pm 0 . 0 8 0
$$

$$
0 . 6 4 4 \pm 0 . 1 0 2
$$

$$
0 . 3 7 7 \pm 0 . 0 8 0
$$

$$
W _ { u p }
$$

$$
0 . 8 7 6 \pm 0 . 0 6 0
$$

$$
0 . 7 2 8 \pm 0 . 0 8 2
$$

$$
0 . 6 1 9 \pm 0 . 1 2 1
$$

$$
0 . 6 0 2 \pm 0 . 0 9 7
$$

$$
0 . 4 6 9 \pm 0 . 1 1 9
$$

$$
0 . 5 1 0 \pm 0 . 1 0 6
$$

$$
0 . 7 2 6 \pm 0 . 0 8 4
$$

$$
0 . 5 1 0 \pm 0 . 1 0 6
$$

$$
0 . 4 7 0 \pm 0 . 1 1 9
$$

$$
0 . 5 7 4 \pm 0 . 1 1 9
$$

$$
0 . 4 4 7 \pm 0 . 1 8 4
$$

$$
0 . 6 5 9 \pm 0 . 0 9 8
$$

$$
0 . 2 8 4 \pm 0 . 1 8 4
$$

$$
W _ { g a t e }
$$

$$
0 . 5 2 7 \pm 0 . 1 2 6
$$

$$
0 . 5 2 8 \pm 0 . 1 2 6
$$

$$
0 . 5 7 3 \pm 0 . 1 1 9
$$

$$
0 . 7 1 0 \pm 0 . 1 0 0
$$

$$
0 . 3 1 2 \pm 0 . 1 3 0
$$

$$
0 . 2 5 6 \pm 0 . 1 1 5
$$

$$
0 . 4 9 0 \pm 0 . 2 0 0
$$

$$
0 . 2 5 7 \pm 0 . 1 1 6
$$

$$
0 . 5 8 0 \pm 0 . 1 3 2
$$

$$
0 . 3 4 3 \pm 0 . 1 4 1
$$

$$
0 . 2 8 7 \pm 0 . 1 2 7
$$

$$
0 . 2 8 8 \pm 0 . 1 2 8
$$

$$
0 . 3 7 1 \pm 0 . 1 5 0
$$

$$
0 . 3 6 0 \pm 0 . 2 2 8
$$

$$
0 . 5 8 1 \pm 0 . 1 3 2
$$

$$
0 . 3 1 6 \pm 0 . 1 3 8
$$

$$
0 . 3 1 7 \pm 0 . 1 3 8
$$

$$
\mu
$$

$$
0 . 3 3 3 \pm 0 . 0 8 0
$$

$$
0 . 2 2 4 \pm 0 . 0 7 0
$$

$$
{ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )
$$

$$
\mathrm { V a n i l l a } \ ( \mu \pm \sigma )
$$

$$
0 . 2 2 4 \pm 0 . 0 6 9
$$

$$
\mathrm { S t d } \ ( \mu \pm \sigma )
$$

$$
0 . 2 1 6 \pm 0 . 0 4 8
$$

$$
0 . 3 6 9 \pm 0 . 0 6 0
$$

$$
0 . 4 5 0 \pm 0 . 1 0 0
$$

$$
\mathrm { M e a n A b s } \ ( \mu \pm \sigma )
$$

$$
0 . 3 1 6 \pm 0 . 0 8 9
$$

$$
{ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )
$$

$$
\mathrm { V a n i l l a } \ ( \mu \pm \sigma )
$$

$$
0 . 5 5 4 \pm 0 . 0 7 9
$$

$$
0 . 3 1 7 \pm 0 . 0 9 0
$$

$$
0 . 3 3 8 \pm 0 . 0 5 5
$$

$$
0 . 5 1 1 \pm 0 . 1 1 0
$$

$$
0 . 6 8 0 \pm 0 . 0 9 4
$$

$$
0 . 4 7 6 \pm 0 . 1 2 0
$$

$$
0 . 3 6 8 \pm 0 . 1 0 0
$$

$$
0 . 3 3 8 \pm 0 . 0 5 5
$$

$$
0 . 3 6 8 \pm 0 . 1 0 0
$$

$$
0 . 6 3 7 \pm 0 . 0 7 8
$$

$$
W _ { o }
$$

$$
0 . 3 9 9 \pm 0 . 0 5 8
$$

$$
0 . 3 9 9 \pm 0 . 0 5 8
$$

$$
0 . 8 2 9 \pm 0 . 0 8 5
$$

$$
0 . 4 7 6 \pm 0 . 0 9 7
$$

$$
W _ { u p }
$$

$$
0 . 3 7 8 \pm 0 . 0 8 1
$$

$$
0 . 6 3 8 \pm 0 . 0 9 5
$$

$$
0 . 6 7 7 \pm 0 . 1 1 4
$$

$$
0 . 6 8 1 \pm 0 . 1 1 1
$$

$$
0 . 5 0 3 \pm 0 . 1 1 9
$$

$$
W _ { d o w n }
$$

$$
0 . 6 2 3 \pm 0 . 1 2 2
$$

$$
0 . 5 1 5 \pm 0 . 1 0 7
$$

$$
0 . 5 1 5 \pm 0 . 1 0 7
$$

$$
0 . 4 6 4 \pm 0 . 1 9 0
$$

$$
0 . 6 9 7 \pm 0 . 0 9 2
$$

$$
0 . 2 9 3 \pm 0 . 1 9 0
$$

$$
0 . 6 9 0 \pm 0 . 1 3 3
$$

$$
0 . 5 6 1 \pm 0 . 1 2 4
$$

$$
0 . 5 8 0 \pm 0 . 1 2 0
$$

$$
0 . 5 7 9 \pm 0 . 1 2 0
$$

$$
0 . 3 0 8 \pm 0 . 1 9 5
$$

$$
0 . 5 6 0 \pm 0 . 1 2 4
$$

$$
W _ { g a t e }
$$

$$
0 . 5 0 0 \pm 0 . 2 0 2
$$

$$
0 . 2 8 3 \pm 0 . 1 1 3
$$

$$
0 . 7 6 2 \pm 0 . 0 8 9
$$

$$
0 . 3 2 1 \pm 0 . 2 0 7
$$

$$
0 . 6 2 3 \pm 0 . 1 2 9
$$

$$
0 . 2 8 3 \pm 0 . 1 1 6
$$

$$
0 . 6 2 1 \pm 0 . 1 3 0
$$

$$
0 . 3 3 6 \pm 0 . 2 1 2
$$

$$
0 . 3 8 1 \pm 0 . 1 4 5
$$

$$
0 . 3 1 6 \pm 0 . 1 2 6
$$

$$
0 . 5 4 1 \pm 0 . 2 1 4
$$

$$
0 . 3 1 5 \pm 0 . 1 2 9
$$

$$
0 . 3 5 5 \pm 0 . 2 2 6
$$

$$
0 . 3 6 9 \pm 0 . 2 3 1
$$

$$
0 . 4 2 7 \pm 0 . 1 5 7
$$

$$
0 . 3 5 1 \pm 0 . 1 4 0
$$

$$
0 . 3 5 0 \pm 0 . 1 4 3
$$

Table A.30: Per-operator activation-conditioned output error for Qwen2.5 7B under TT core4 tensorization with α = 1. Entries report mean µ and standard deviation σ over per-block sweep values. Lower is better.
<table><tr><td rowspan="2"></td><td colspan="3">2× compression</td><td colspan="3">4× compression</td><td colspan="3">6× compression</td><td></td></tr><tr><td>Matrix Vanilla (µ±σ)</td><td>Std (µ±σ)</td><td> ${ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )$ </td><td>Vanilla (µ±σ)</td><td> $\mathrm { S t d } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { M e a n A b s } \ ( \mu \pm \sigma )$ </td><td> $\mathrm { V a n i l l a } \ ( \mu \pm \sigma )$ </td><td>Std (µ±σ)</td><td> ${ \mathrm { M e a n A b s ~ } } ( \mu \pm \sigma )$ </td><td>N</td></tr><tr><td>Wq</td><td></td><td>0.487 ± 0.066 0.372 ± 0.070</td><td>0.373 ± 0.072</td><td> $0 . 6 6 8 \pm 0 . 0 6 7$ </td><td>0.535 ± 0.088</td><td> $0 . 5 3 3 \pm 0 . 0 9 0$ </td><td>0.745 ± 0.067 0.613 ± 0.094</td><td></td><td>0.608 ± 0.095</td><td>28</td></tr><tr><td>Wk</td><td> $0 . 4 5 8 \pm 0 . 0 3 4$ </td><td> $0 . 3 6 4 \pm 0 . 0 3 0$ </td><td>0.362 ± 0.030</td><td> $0 . 6 6 8 \pm 0 . 0 3 7$ </td><td> $0 . 5 5 7 \pm 0 . 0 4 0$ </td><td> $0 . 5 5 3 \pm 0 . 0 3 9$ </td><td> $0 . 7 6 0 \pm 0 . 0 3 2$ </td><td> $0 . 6 5 1 \pm 0 . 0 4 2$ </td><td> $0 . 6 4 6 \pm 0 . 0 3 9$ </td><td>28</td></tr><tr><td>Wv</td><td> $0 . 6 0 7 \pm 0 . 0 4 0$ </td><td> $0 . 5 0 1 \pm 0 . 0 4 8$ </td><td> $0 . 5 0 1 \pm 0 . 0 5 0$ </td><td> $0 . 7 9 4 \pm 0 . 0 3 1$ </td><td> $0 . 6 9 6 \pm 0 . 0 5 8$ </td><td> $0 . 6 9 5 \pm 0 . 0 6 0$ </td><td> $0 . 8 6 2 \pm 0 . 0 2 5$ </td><td> $0 . 7 7 5 \pm 0 . 0 6 3$ </td><td> $0 . 7 7 4 \pm 0 . 0 6 4$ </td><td>28</td></tr><tr><td>W。</td><td> $0 . 5 2 9 \pm 0 . 0 9 0$ </td><td> $0 . 4 3 8 \pm 0 . 0 9 2$ </td><td> $0 . 4 4 3 \pm 0 . 0 9 6$ </td><td> $0 . 6 9 0 \pm 0 . 1 1 0$ </td><td> $0 . 6 0 0 \pm 0 . 1 1 9$ </td><td> $0 . 5 9 8 \pm 0 . 1 2 4$ </td><td> $0 . 7 5 6 \pm 0 . 1 1 8$ </td><td> $0 . 6 7 2 \pm 0 . 1 3 1$ </td><td> $0 . 6 6 6 \pm 0 . 1 3 5$ </td><td>28</td></tr><tr><td> $W _ { u p }$ </td><td> $0 . 6 6 1 \pm 0 . 0 9 5$ </td><td> $0 . 4 9 8 \pm 0 . 0 7 9$ </td><td>0.503 ± 0.081</td><td> $0 . 8 3 1 \pm 0 . 0 7 2$ </td><td> $0 . 6 6 8 \pm 0 . 0 9 0$ </td><td> $0 . 6 7 0 \pm 0 . 0 9 5$ </td><td> $0 . 8 8 7 \pm 0 . 0 5 5$ </td><td> $0 . 7 3 4 \pm 0 . 0 9 2$ </td><td>0.734 ± 0.098</td><td>28</td></tr><tr><td> $W _ { d o w n }$ </td><td> $0 . 4 2 9 \pm 0 . 1 1 9$ </td><td> $0 . 3 2 0 \pm 0 . 1 7 7$ </td><td>0.321 ± 0.178</td><td> $0 . 6 0 8 \pm 0 . 1 3 5$ </td><td> $0 . 4 4 4 \pm 0 . 2 3 0$ </td><td> $0 . 4 4 3 \pm 0 . 2 3 2$ </td><td> $0 . 6 8 5 \pm 0 . 1 3 5$ </td><td> $0 . 4 9 5 \pm 0 . 2 4 9$ </td><td>0.493 ± 0.249</td><td>28</td></tr><tr><td> $W _ { g a t e }$ </td><td> $0 . 3 7 5 \pm 0 . 1 1 8$ </td><td> $0 . 2 8 9 \pm 0 . 1 1 5$ </td><td> $0 . 2 9 2 \pm 0 . 1 1 7$ </td><td> $0 . 4 9 9 \pm 0 . 1 5 7$ </td><td> $0 . 3 9 1 \pm 0 . 1 5 1$ </td><td> $0 . 3 9 3 \pm 0 . 1 5 2$ </td><td> $0 . 5 4 6 \pm 0 . 1 7 2$ </td><td> $0 . 4 3 3 \pm 0 . 1 6 5$ </td><td> $0 . 4 3 5 \pm 0 . 1 6 5$ </td><td>28</td></tr></table>