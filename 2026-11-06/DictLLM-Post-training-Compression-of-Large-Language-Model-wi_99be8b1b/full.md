# DictLLM: Post-training Compression of Large Language Model with Dictionary Kernels

Jinho Park Columbia University jp4327@columbia.edu

Se Young Chun Seoul National University sychun@snu.ac.kr

Mingoo Seok Columbia University ms4415@columbia.edu

jp4327.github.io/dictllm/

## Abstract

Efficient deployment of large language models (LLMs) on mobile devices requires balancing performance with compute and memory constraints. Post-training compression is an effective approach to managing this trade-off, while requiring much less computation than in-training counterparts. We propose DictLLM, a dictionarybased compression scheme in which weight matrices within grouped layers are decomposed into a shared dictionary and layer-specific sparse coefficients, directly obtained from a pre-trained model without further training. Specifically, we cast DictLLM as a four-step post-training optimization. First, we obtain shared dictionaries via singular value decomposition (SVD) of input-scaled weight matrices, amortizing rank across grouped layers; second, we sparsify the layer-specific coefficients to the target compression ratio using a dictionary-aware Hessian; and third, we refine the dictionaries given the resulting sparse coefficients. Finally, we aggressively quantize both components using factor-specific Hessian information: input covariance for the dictionaries and the dictionary-aware Hessian for the coefficients. The resulting representation reduces memory footprint by storing a single shared dictionary per layer group, along with sparse layer-specific coefficients. On language modeling and zero-shot reasoning benchmarks, DictLLM consistently outperforms existing post-training low-rank compression methods and pushes the Pareto frontier between model size and performance. DictLLM compresses the original LLaMA-7B model by 5.1× with only a 2.9-point increase in WikiText-2 perplexity, achieving a 3.6× higher compression ratio than the state-of-the-art low-rank method.

## 1 Introduction

Deploying large language models (LLMs) on mobile hardware is fundamentally limited by memory, not computation [1–3]. Their billions of parameters exceed device capacity and dominate energy cost through repeated weight memory transfers. Mobile devices do not have enough memory to store these models, and loading parameters to and from memory consumes significantly more energy than computations required for attention and feedforward (FFN) operations [1].

In light of this, LLM compression has emerged as a promising approach for deploying LLMs on mobile graphics processing units (GPUs) with limited memory. In particular, post-training compression has attracted significant research attention, as it requires far fewer computational resources than its in-training counterparts. Model compression offers two key benefits: (1) it enables deployment of state-of-the-art LLMs on a mobile device, and (2) it alleviates memory-bandwidth bottlenecks, as characterized by the roofline model [3, 4]. While compression also reduces the number of multiply–accumulate (MAC) operations, the dominant performance gains typically arise from reduced memory traffic rather than decreased computation [4, 5].

![](images/92254bcb828e67d156f5fd0f74ee7b20894284d38dbb82ce8ca4b9fd9cd1f425.jpg)  
Figure 1: DictLLM: Dictionary-based compression method. DictLLM resolves the weight memory bottleneck by compressing transformer weights into a shareable dictionary. It compounds four compression steps. ➊ The first step factorizes input-scaled weight matrices of grouped layers into a shared dictionary and layer-specific coefficients. ➋ We then prune the layer-specific coefficients using a dictionary-aware Hessian. ➌ We refit the shared dictionary to the resulting sparse coefficients. ➍ We quantize both the dictionary and the coefficients into NVFP4 using input statistics for the dictionary and dictionary-aware Hessians for the coefficients. The resulting compressed representation enables efficient projections by transferring a small set of shared dictionaries.

Post-training model compression methods have sought to reduce model footprint along three independent axes: low-rank approximation shrinks matrix dimensions, pruning removes individual connections, and quantization lowers per-element bit width. Low-rank approximation uses singular value decomposition (SVD) [6] to factorize weight matrices into smaller ones [7–12]. However, meaningful compression requires the retained rank to be substantially smaller than the original matrix dimensions. Since non-negligible spectral components can remain outside the retained subspace, aggressive low-rank approximation often degrades performance. On the other hand, pruning [13] allows sparsification of weight matrices while preserving their shape [14, 15]. Since neural processing units (NPUs) and GPUs support sparse matrix-matrix multiplication, pruned models can benefit from improved energy efficiency. Orthogonal to these methods, quantization [16–23] reduces the bit width of each element while keeping the number of model parameters constant.

Parameter sharing [24–27] takes a different approach by reusing attention and feedforward weights across layers, enabling deeper models without additional parameters. On the other hand, dictionarybased transformers relax strict sharing by representing weights as combinations of shared atoms and layer-specific coefficients [25, 28]. This idea is closely related to classical dictionary learning, where algorithms such as the Method of Optimal Directions (MOD) [29] and k-SVD [30] reconstruct signals such as images from an overcomplete dictionary and sparse representation coefficients. Although parameter sharing approaches can greatly reduce the number of parameters and memory traffic, they often require training the shared representation from scratch. This in-training requirement of the method makes it difficult to apply directly as a post-training compressor for billion-scale LLMs.

Existing post-training methods [7–12, 14–21] and sharing-based architectures [24–26] therefore exploit complementary axes of redundancy, but they are typically applied in isolation. A more effective post-training compressor should combine these benefits by sharing a reusable structure across layers while preserving layer-specific adaptation. However, stacking these methods is nontrivial since each compression stage changes the weight and activation statistics used by the next stage, and a naïve composition can severely degrade performance.

We propose DictLLM, a post-training compression method that jointly achieves layer sharing, Hessianaware sparsification, and 4-bit block floating-point (FP4) quantization via a single shared dictionary and sparse layer-specific coefficients within grouped layers (Fig. 1). We develop a 3+1 approach that shows shared dictionary-based compression can be constructed via a three-step post-training process: input-scaled factorization, dictionary-aware Hessian coefficient pruning, and dictionary refitting. This distinguishes our approach from prior work that constructs dictionary-based compression from scratch [25] or requires principal component-based analysis and fine-tuning [31]. In the last step, we show that the compressed model remains amenable to FP4 [16] quantization.

We highlight our contributions as follows:

• We propose dictionary-based parameter sharing, which decomposes large weight matrices into shared dictionary kernels and layer-wise sparse representation coefficients. Our zero-shot post-training compression method enables compression of a billion-parameter LLM.

• We combine activation-aware dictionary factorization and dictionary-induced Hessian pruning of the coefficients, then refit the shared dictionary to the sparse representation coefficients, improving reconstruction without changing sparsity or model size.

• We perform Hessian-aware FP4 quantization using the input covariance for the dictionary and the dictionary-based saliency for the coefficients.

• DictLLM reduces the original LLaMA-7B size by 5.1× with only a 2.9-point increase in WikiText-2 perplexity, improving over the state-of-the-art low-rank method by 3.6×.

## 2 Related works

## 2.1 Post-training model compression

Low-rank compression. Low-rank approximation is a natural approach for compressing transformer weights because attention and feedforward layers are dominated by large matrix multiplications. Existing SVD methods reduce each weight matrix into two smaller factors, but aggressive rank truncation often causes substantial degradation in LLMs. SVD-LLM [8] improves low-rank compression by using truncation-aware decomposition and activation information. ASVD [11] and Dobi-SVD [10] also use activation-aware scaling to reduce the error from low-rank approximation.

Pruning. Pruning reduces model size by removing redundant weights while preserving the original matrix shapes. Early pruning methods used magnitude or second-order saliency criteria to identify parameters to remove [13, 32]. For LLMs, SparseGPT [14] introduced an efficient post-training pruning method that uses calibration activations as a Hessian proxy and compensates for pruning errors in a blockwise manner. Wanda [15] showed that combining weight magnitude with activation statistics can provide an effective pruning criterion. These methods directly prune dense weight matrices [33]. In contrast, our method prunes the layer-specific coefficient matrices after dictionary factorization and projects the activation covariance into the coefficient space induced by the shared dictionary.

Low-bit quantization. Quantization reduces memory footprint and bandwidth by representing model weights with fewer bits [18, 19, 21, 34]. Post-training quantization methods, such as GPTQ [19], use calibration activations to compensate for quantization errors without retraining. Recent hardware-oriented formats further push weight precision below INT8 by using floating-point quantization with shared scales. NVFP4 [16] represents groups of FP4 values with micro-scaling factors, enabling efficient low-bit inference on modern accelerators. Microscaling FP4 methods have shown that careful scaling and error compensation are important for closing the gap between theoretical compression and practical accuracy [17]. DictLLM quantizes both the dictionary and the coefficients with NVFP4. It compensates for dictionary quantization errors using the input covariance and for coefficient quantization errors using the dictionary-induced Hessian.

## 2.2 Parameter sharing and dictionary learning

Parameter sharing. Parameter sharing reduces model size by reusing weights across layers or recurrent computation steps. ALBERT [24] shares transformer parameters across layers, showing that substantial parameter reduction is possible without changing the high-level architecture. Deep equilibrium models [35] take a related perspective by replacing explicit layer stacks with a recurrent computation that converges to a fixed point. More recent methods relax strict sharing by adding layer-specific residual or low-rank components [26, 36–39].

Basis Sharing [9] further improves the compression trade-off by sharing basis matrices across layers instead of factorizing each layer independently. DictLLM builds on this direction by recovering shared dictionaries via activation-aware factorization, while further sparsifying the layer-specific coefficients using a dictionary-aware Hessian and quantizing both factors with NVFP4.

![](images/efe448c36b2a35d789e331a813aa11d30b7aafa69e7e12ee0ca484a401a2455f.jpg)  
Figure 2: Dictionary-based forward operation. Each weight matrix $\mathbf { \dot { W } } ^ { \ell }$ is replaced by the product of a shared dictionary $\widetilde { \mathbf Ḋ } ^ { g } Ḍ$ and a layer-specific sparse coefficient matrix $\widetilde { \mathbf { C } } ^ { \ell }$ . The forward pass computes $\mathbf { Y } ^ { \ell } = \mathbf { X } ^ { \ell } \widetilde { \mathbf { D } } ^ { g } \widetilde { \mathbf { C } } ^ { \ell }$

![](images/30a9029cb912fc3cfc5ddd254441f1686602daad69fd459007b715a9096cbef9.jpg)

![](images/afbe0abedaaf6998428fed9a661b5db224e5737db7021c8de5a8194beeb63734.jpg)  
Figure 3: Motivation for staged dictionary refinement. We compare (a) activation reconstruction error and (b) runtime of our factorize– prune–update pipeline against k-SVD on a single LLaMA-7B query projection.

Dictionary learning. Dictionary-based models represent weights or signals as combinations of shared atoms and sparse coefficients. This idea is rooted in classical dictionary learning, where methods such as MOD [29] and k-SVD [30, 40] alternate between sparse coding and dictionary updates to reconstruct signals from an overcomplete basis. DictFormer [25] applies this idea to transformers and substantially reduces model size by sharing dictionary atoms across layers. T-REX [28] further shows that keeping the shared dictionary on chip can reduce external memory access and improve energy efficiency. However, existing dictionary-based transformer approaches typically require training the dictionary representation from scratch, making them difficult to apply to billion-scale pretrained LLMs. DictLLM instead constructs a dictionary representation, prunes coefficients with a dictionary-induced Hessian, and refits the dictionary to the sparse coefficients.

## 3 Proposed methods

## 3.1 Shared dictionary representation

We first describe the shared-dictionary representation and the forward operation. For a layer group g with layers $\ell \in g$ that share a dictionary, we define input dimension $d _ { \mathrm { i n } } .$ , output dimension $d _ { \mathrm { o u t } }$ , and dictionary size k (Fig. 2). A shared dictionary $\widetilde { \mathbf { D } } ^ { g } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times k }$ is reused by every layer in the group, while the coefficient matrix $\widetilde { \mathbf { C } } ^ { \ell } \in \mathbb { R } ^ { k \times d _ { \mathrm { o u t } } }$ of each layer is not shared. The original weight matrix $\mathbf { W } ^ { \ell } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { o u t } } }$ of layer ℓ is approximated column-wise as

$$
\widetilde { \mathbf { W } } _ { : , j } ^ { \ell } = \sum _ { m \in S _ { j } ^ { \ell } } \widetilde { \mathbf { D } } _ { : , m } ^ { g } \widetilde { \mathbf { C } } _ { m , j } ^ { \ell } ,\tag{1}
$$

where $S _ { j } ^ { \ell } \subseteq [ k ]$ denotes the indices of active dictionary atoms. The coefficient sparsity is defined as $p = 1 - | S _ { j } ^ { \ell } | / k$ . Equation 1 can be written in matrix form as $\widetilde { \mathbf { W } } ^ { \ell } = \widetilde { \mathbf { D } } ^ { g } \widetilde { \mathbf { C } } ^ { \ell }$ , where $\widetilde { \mathbf { C } } ^ { \ell }$ is sparse.

## 3.2 Dictionary-based compression objective

We compute the dictionary and sparse coefficients sequentially because joint optimization would require solving a large discrete-continuous problem over shared factors and sparsity masks. DictLLM solves an activation-weighted reconstruction problem,

$$
\operatorname* { m i n } _ { \mathbf { D } ^ { g } , \left\{ \mathbf { C } ^ { \ell } \right\} _ { \ell \in g } } \sum _ { \ell \in g } \left. \mathbf { X } ^ { \ell } \mathbf { W } ^ { \ell } - \mathbf { X } ^ { \ell } \mathbf { D } ^ { g } \mathbf { C } ^ { \ell } \right. _ { F } ^ { 2 }\tag{2}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. Algorithm 1 delineates the decomposition steps. We first decompose grouped model weights into a shared dictionary D<sup>g</sup> and dense layer-specific coefficients $\{ \mathbf { C } ^ { \ell } \} _ { \ell \in g }$ using activation-aware low-rank factorization. Next, we sparsify the coefficients using a dictionary-aware Hessian criterion with the dictionary fixed. The final step refits the shared dictionary based on the resulting coefficients.

Algorithm 1: Dictionary-based compres- Algorithm 2: Dictionary-aware coefficient   
sion pruning   
Input: Pretrained weights $\{ \mathbf { W } _ { t } ^ { \ell } \} _ { \ell = 1 } ^ { L } ,$ Input: Coefficient matrix C, dictionary   
grouped layer index sets ${ \mathcal { G } } ,$ $\mathbf { D } ,$ input covariance $\mathbf { S } ,$ sparsity $p ,$   
calibration set X , sparsity p damping λ, block size B   
Output: Compressed parameters $\{ \widehat { \bf D } , \widehat { \bf C } \}$ Output: Pruned coefficient matrix $\widetilde { \mathbf { C } }$   
1: $\{ \mathbf { S } _ { t } ^ { \ell } \} _ { \ell , t }  \mathrm { C o l l e c t C o v a r i a n c e } ( f _ { \mathbf { W } } , \boldsymbol { \chi } ) ;$ 1: $\mathbf { H } ^ { - 1 }  ( \mathbf { D } ^ { \top } \mathbf { S } \mathbf { D } + \lambda \mathbf { I } ) ^ { - 1 } ;$   
2: foreach weight type t do 2: $\mathbf { d }  \mathrm { d i a g } ( \mathbf { H } ^ { - 1 } ) ;$   
3: foreach layer group $g \in \mathcal G ^ { t }$ do 3: $\begin{array} { r } { \tilde { \mathbf { C } } \gets \mathbf { C } ; } \end{array}$   
4: $\mathcal { W } _ { t } ^ { g }  \{ \mathbf { W } _ { t } ^ { \ell } : \ell \in g \} ;$ 4: for $j \gets 0 , B , 2 B , \ldots$ do   
5: $\begin{array} { r } { \mathbf { S } _ { t } ^ { g }  \sum _ { \ell \in g } \mathbf { S } _ { t } ^ { \ell } ; } \end{array}$ 5: $\mathbf { R } _ { : , j : j + B } $   
6: $k _ { t } ^ { g } \gets \mathrm { R a n k } ( \mathcal { W } _ { t } ^ { g } ) ;$ Salienc $\gamma \mathrm { M a p } ( \widetilde { \mathbf { C } } _ { : , j : j + B } , \mathbf { d } ) ;$   
7: $\mathbf { D } _ { t } ^ { g } , \{ \mathbf { C } _ { t } ^ { \ell } \} _ { \ell \in g } \gets$ 6: for $i \gets 0 \ : { \bf t o } \ : \lfloor p \cdot k \rfloor - 1$ do   
$\begin{array} { r } { \mathrm { F a c t o r i z e } ( \mathcal { W } _ { t } ^ { g } , \mathbf { S } _ { t } ^ { g } , k _ { t } ^ { g } ) ; } \end{array}$ 7: $\mathbf { c } \gets \mathbf { R } _ { i , j : j + B } ;$   
8: $\mathbf { H } _ { t } ^ { \mathbf { C } , g } \gets ( \mathbf { D } _ { t } ^ { g } ) ^ { \top } \mathbf { S } _ { t } ^ { g } \mathbf { D } _ { t } ^ { g } ;$ $/ /$ Least salient rows   
9: foreach $\ell \in g$ do 8: $\mathbf { e }  \widetilde { \mathbf { C } } _ { [ \mathbf { c } , j : j + B ] } \odot \mathbf { d } _ { [ \mathbf { c } ] } ^ { - 1 } ;$   
10: $\begin{array} { r } { \big \lfloor \widetilde { \mathbf { C } } _ { t } ^ { \ell } \gets \mathrm { P r u n e } ( \mathbf { C } _ { t } ^ { \ell } , \mathbf { H } _ { t } ^ { \mathbf { C } , g } , p ) ; } \end{array}$ $/ /$ Corresponding error   
11: $\widetilde { \mathbf { D } } _ { t } ^ { g } \gets \mathrm { R e f i t } ( \mathcal { W } _ { t } ^ { g } , \{ \widetilde { \mathbf { C } } _ { t } ^ { \ell } \} _ { \ell \in g } ) ;$ 9: $\widetilde { \mathbf { C } } _ { : , j : j + B } \gets$   
12: $\widehat {  Ḋ \mathbf Ḋ \Gamma Ḍ Ḍ } _ { t } ^ { g } \gets \mathrm { Q u a n t i z e } ( \widetilde {  Ḋ \mathbf Ḋ \mathbf Ḋ \Gamma Ḍ Ḍ }  ^ { g } , \mathbf S _ { t } ^ { g } ) ;$ $\widetilde { \mathbf { C } } _ { : , j : j + B } - \mathbf { H } _ { : , [ \mathbf { c } ] } ^ { - 1 } \mathrm { d i a g } ( \mathbf { e } ) ;$   
13: 14: foreach $\begin{array} { r } { \mathsf { L } \widehat { \mathbf { C } } _ { t } ^ { \ell } \gets \mathrm { Q u a n t i z e } ( \widetilde { \mathbf { C } } _ { t } ^ { \ell } , \mathbf { H } _ { t } ^ { \mathbf { C } , g } ) ; } \end{array}$ $\ell \in g$ do 10: $\widetilde { \mathbf { C } } _ { : , j : j + B } \gets \widetilde { \mathbf { C } } _ { : , j : j + B } \odot \mathbf { M } _ { : , j : j + B }$ $/ / \mathsf { \nabla { A p p 1 9 } }$ pruning

Motivation for staged dictionary refinement. To motivate our post-training approximation, we compare our method, consisting of activation-aware ${ \mathrm { S V D } } ,$ Hessian coefficient pruning, and postpruning dictionary update, against k-SVD with an undercomplete dictionary $( k ~ < ~ d _ { \mathrm { i n } } )$ on the first layer of LLaMA-7B query projection (Fig. 3). k-SVD is an alternative to our sequential pipeline for learning a sparse dictionary representation from scratch. However, full sparse dictionary learning is expensive at the LLM scale because it requires repeated atom-wise and coefficient updates over a large number of weight columns. We measure the relative activation reconstruction error $\lVert \mathbf { X } \mathbf { W } - \mathbf { X } \widetilde { \mathbf { W } } \rVert _ { F } / \lVert \mathbf { X } \mathbf { W } \rVert _ { F }$ using 256 calibration samples from WikiText-2. Our three-stage pipeline achieves lower reconstruction error than k-SVD while running 3.4× faster. We discuss the formulation of k-SVD in Appendix A1.1.

## 3.3 Dictionary-based compression pipeline: factorize, prune, refit, and quantize

Step 1: Activation-aware dictionary factorization. We first factorize each group of weights into a shared dictionary and layer-specific coefficients using activation-aware SVD. For a layer group $g = \{ \ell _ { 1 } , \ldots , \ell _ { n } \}$ , we first obtain the concatenated weight $\mathbf { W } ^ { g } = [ \mathbf { W } ^ { \ell _ { 1 } } | \mathbf { W } ^ { \ell _ { 2 } } | \cdot \cdot \cdot | \mathbf { W } ^ { \ell _ { n } } ]$ for n weights in $^ { g , }$ similar to [9]. Given calibration examples $x ,$ , we run the original model $f _ { \mathbf { W } }$ to collect each linear module’s input activations $\mathbf { X } ^ { \ell }$ and aggregate it as $\begin{array} { r } { \mathbf { S } ^ { g } = \sum _ { \ell \in g } \mathbf { S } ^ { \ell } = \sum _ { \ell \in g } ( \mathbf { \dot { X } } ^ { \ell } ) ^ { \top } \mathbf { X } ^ { \ell } \in \mathbf { \Xi } } \end{array}$ $\mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { i n } } }$ . This is equivalent to computing the covariance after vertically concatenating the inputs from the grouped layers. We then perform rank-k SVD on the input-scaled concatenated weight $( \mathbf { S } ^ { g } ) ^ { 1 / 2 } \bar { \mathbf { W } } ^ { g } \approx \mathbf { U } _ { k } \bar { \mathbf { \Sigma } } _ { k } ^ { - } \mathbf { V } _ { k } ^ { \top }$ , where $\bar { \mathbf { U } } _ { k } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times k } , \mathbf { \Sigma } _ { k } \in \mathbb { R } ^ { k \times k }$ , and $\mathbf { V } _ { k } \in \mathbb { R } ^ { n d _ { \mathrm { o u t } } \times k }$ The shared dictionary is attained as $ { \mathbf { D } } ^ { g } = (  { \mathbf { S } } ^ { g } ) ^ { - 1 / 2 }  { \mathbf { U } } _ { k }  { \boldsymbol { \Sigma } } _ { k }$ , and $\mathbf { V } _ { k } ^ { \top }$ is equivalent to layer-specific coefficients $\{ \mathbf { C } ^ { \ell } \} _ { \ell \in g }$ . Appendix A1.2 includes details on rank k in relation to the compression ratio.

Step 2: Dictionary-aware coefficient pruning. After dictionary factorization, we prune the layerspecific coefficients $\{ \mathbf { C } ^ { \ell } \} _ { \ell \in g }$ , keeping the shared dictionary D<sup>g</sup> fixed. Unlike SparseGPT [14], which prunes the dense weight $\mathbf { W } ^ { \ell }$ using the input covariance $\mathbf { S } ^ { \ell } = ( \mathbf { X } ^ { \ell } ) ^ { \top } \mathbf { X } ^ { \ell } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { i n } } }$ , DictLLM prunes in the coefficient space defined by the shared dictionary. Pruning the coefficient induces a structured weight perturbation $\Delta \mathbf { W } ^ { \ell } = \mathbf { D } ^ { j } \Delta \mathbf { C } ^ { \ell }$ . The local reconstruction error from coefficient pruning is

$$
\| \mathbf { X } ^ { \ell } \mathbf { D } ^ { g } \Delta \mathbf { C } ^ { \ell } \| _ { F } ^ { 2 } .\tag{3}
$$

This yields the dictionary-aware coefficient Hessian $\mathbf { H } ^ { \mathbf { C } , g } = ( \mathbf { D } ^ { g } ) ^ { \top } \mathbf { S } ^ { g } \mathbf { D } ^ { g } \in \mathbb { R } ^ { k \times k }$ , where ${ \bf S } ^ { g }$ is the grouped input covariance from calibration examples. We derive this coefficient-space Hessian from the local reconstruction objective in Appendix A1.3.

Algorithm 2 summarizes the pruning procedure. $\mathrm { S a l i e n c y M a p } ( \widetilde { \mathbf { C } } _ { : , j : j + B } , \mathbf { d } )$ returns $\mathbf { R } _ { : , j : j + B } \in$ $\mathbb { Z } ^ { k \times B }$ for each block of B columns. Each column of R sorts the k row indices by ascending saliency calculated by $\widetilde { \mathbf { C } } _ { i , j } ^ { 2 } / \mathbf { d } _ { i }$ <sub>i</sub> where $\mathbf { d } \in \mathbb { R } ^ { k }$ . At pruning step i, we find the least salient row index per column for the B columns in the block as $\mathbf { c } \in \mathbb { Z } ^ { B }$ , and compute the error used to compensate the remaining rows as $\mathbf { e } \in \mathbb { R } ^ { B }$ . After $\lfloor p \cdot k \rfloor$ steps, the binary mask M prunes the rows indexed by $\mathbf { R } _ { 0 : \lfloor p \cdot k \rfloor , : } .$ The key efficiency gain comes from operating in the coefficient space. Since $k < d _ { \mathrm { i n } } .$ inverting $\mathbf { H } ^ { \mathbf { C } , g } \in \mathbb { R } ^ { k \times k }$ costs $\mathcal { O } ( k ^ { 3 } )$ rather than $\mathcal { O } ( d _ { \mathrm { i n } } ^ { 3 } )$ of [14, 32]; the inverse is shared across all coefficient columns, reducing each column update to $\ddot { \mathcal { O } } ( k ^ { 2 } )$

Step 3: Updating shared dictionary with sparse coefficients. We refit the dictionary to the sparse coefficients while keeping the coefficient values and sparsity pattern fixed. This is necessary because the shared dictionary $\bar { \mathbf Ḋ } ^ { g }$ is obtained before pruning and is therefore optimized for dense coefficients, not for the sparse support induced by $\widetilde { \mathbf { C } } ^ { \ell }$ . We update the shared dictionary $\mathbf { D } ^ { g } \gets$ $\mathbf { W } ^ { g } ( \widetilde { \mathbf { C } } ^ { g } ) ^ { \top } ( \widetilde { \mathbf { C } } ^ { g } ( \widetilde { \mathbf { C } } ^ { g } ) ^ { \top } + \lambda \mathbf { I } ) ^ { - 1 }$ where $\widetilde { \mathbf { C } } ^ { g } = [ \widetilde { \mathbf { C } } ^ { \ell _ { 1 } } | \cdot \cdot \cdot | \widetilde { \mathbf { C } } ^ { \ell _ { n } } ]$ for a layer group g. This is a solution to a least-squares problem, min<sub>D</sub>g $| | \mathbf { X } ^ { g } ( \mathbf { W } ^ { g } - \mathbf { D } ^ { g } \widetilde { \mathbf { C } } ^ { g } ) | | _ { F } ^ { 2 }$ where $\mathbf { X } ^ { g } = [ ( \mathbf { X } ^ { \ell _ { 1 } } ) ^ { \top } , \ldots , ( \mathbf { X } ^ { \ell _ { n } } ) ^ { \top } ] ^ { \top }$ denotes the vertical concatenation of calibration activations. Hence, $\begin{array} { r } { ( \mathbf { X } ^ { g } ) ^ { \top } \mathbf { X } ^ { g } = \sum _ { \ell \in \mathfrak { q } } ( \mathbf { X } ^ { \ell } ) ^ { \top } \mathbf { X } ^ { \ell } = \mathbf { S } ^ { g } } \end{array}$ This update improves reconstruction under fixed sparse support without changing coefficient sparsity.

Step 4: Hessian-aware NVFP4 quantization. We quantize the dictionary and coefficients using respective Hessians. For the dictionary $\mathbf { D } ^ { g }$ , we group input covariance as the Hessian since the module input is the original activation. For the coefficient $\mathbf { \vec { C } } ^ { \ell }$ , the Hessian proxy is $\mathbf { H } ^ { \mathbf { C } , g } = ( \mathbf { D } ^ { g } ) ^ { \top } \mathbf { S } ^ { g } \mathbf { D } ^ { g }$ since the module input is $\mathbf { X } ^ { \ell } \mathbf { D } ^ { g }$ . Thus, coefficient quantization uses the same dictionary-induced Hessian as coefficient pruning.

With these Hessians, we apply error compensation and quantize the dictionary and coefficients using NVFP4. During coefficient quantization, we preserve the pruning mask by fixing masked entries to zero and compensating for the resulting errors. We denote the NVFP4 quantizer as $\mathcal { Q } _ { \mathrm { N V F P 4 } } ( \cdot )$ and obtain the final compressed dictionary as $\widehat { \mathbf { D } } ^ { g } = \mathcal { Q } _ { \mathrm { N V F P 4 } } ( \mathbf { D } ^ { g } )$ and coefficient $\widehat { \mathbf { C } } ^ { \ell } = \mathcal { Q } _ { \mathrm { N V F P 4 } } ( \widetilde { \mathbf { C } } ^ { \ell } )$ Following the NVFP4 format [16], each block of values is represented by low-precision FP4 together with a shared microscale.

## 4 Experiments

## 4.1 Experimental setup

Models and datasets. We use DictLLM on LLaMA-7B and LLaMA2-7B [41], OPT-6.7B [42], and Mistral-7B [43] using the HuggingFace checkpoint. We evaluate language modeling performance on WikiText-2 [44], Penn TreeBank (PTB) [45], and Colossal Clean Crawled Corpus (C4) [46]. We also evaluate zero-shot reasoning performance on seven benchmarks: OpenBookQA [47], Wino-Grande [48], HellaSwag [49], PIQA [50], MathQA [51], ARC-Easy, and ARC-Challenge [52]. All reasoning evaluations are performed in the zero-shot setting using the LM-Evaluation-Harness [53].

Baselines and metrics. We compare DictLLM against post-training low-rank compression methods. ASVD [11] applies activation-aware scaling before low-rank decomposition. SVD-LLM [8] improves low-rank compression with truncation-aware singular value decomposition. Basis Sharing [9] shares low-rank bases across transformer layers and is the closest baseline to our dictionary factorization. We report perplexity on language modeling tasks and accuracy for reasoning tasks. We measure compression gain based on the number of non-zero entries in the dictionary and coefficient matrices.

Implementation details. We use models with checkpoints from HuggingFace Transformers. We collect calibration statistics from 256 WikiText-2 samples with a sequence length of 2048. All experiments are run on dual NVIDIA RTX PRO 6000 Blackwell GPUs. We apply NVFP4 quantization post hoc to both dictionary and coefficient factors based on QuTLASS [54].

Table 1: Benchmark on language modeling and reasoning tasks. We benchmark DictLLM against low-rank compression baselines on language modeling and commonsense reasoning tasks for LLaMA-7B at different compression ratios. We additionally report the FP4-quantized DictLLM models at the bottom of the table. We measure perplexity on WikiText-2, Penn TreeBank (PTB), and C4, and zero-shot accuracy on OpenBookQA (OB), WinoGrande (WG), HellaSwag (HS), PIQA (PQ), MathQA (MQ), ARC-Easy (AE), and ARC-Challenge (AC).
<table><tr><td rowspan="2">Comp. gain</td><td rowspan="2">Method</td><td colspan="3">Perplexity ↓</td><td colspan="10">Zero-shot accuracy ↑</td></tr><tr><td>WikiText-2</td><td>PTB</td><td>C4</td><td>OB</td><td>WG</td><td></td><td>HS</td><td>PQ</td><td>MQ</td><td>AE</td><td>AC</td><td>Average</td></tr><tr><td>1x</td><td>Original</td><td>5.68</td><td>8.35</td><td>7.34</td><td>0.28</td><td>0.67</td><td></td><td>0.56</td><td>0.78</td><td>0.27</td><td>0.67</td><td>0.38</td><td>0.52</td></tr><tr><td rowspan="4">1.25×</td><td>ASVD [11] SVD-LLM [8]</td><td>11.14 7.94</td><td>16.55 18.05</td><td>15.93 15.93</td><td>0.25 0.22</td><td></td><td>0.64 0.63</td><td>0.41 0.43</td><td>0.68 0.69</td><td>0.24 0.24</td><td>0.53 0.58</td><td>0.27 0.29</td><td>0.43 0.44</td></tr><tr><td>Basis Sharing [9]</td><td>7.74</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>6.89</td><td>17.35</td><td>15.03</td><td>0.28</td><td></td><td>0.66</td><td>0.46</td><td>0.71</td><td>0.25</td><td>0.66</td><td>0.36</td><td>0.48</td></tr><tr><td>DictLLM (Ours)</td><td></td><td>12.27</td><td>11.75</td><td>0.33</td><td>0.67</td><td></td><td>0.50</td><td>0.74</td><td>0.26</td><td>0.69</td><td>0.36</td><td>0.51</td></tr><tr><td rowspan="4">1.43×</td><td>ASVD [11] SVD-LLM [8]</td><td>51 9.56</td><td>70</td><td>41</td><td>0.18</td><td>0.53</td><td></td><td>0.37</td><td>0.65</td><td>0.21</td><td>0.43</td><td>0.25</td><td>0.38</td></tr><tr><td></td><td>9.25</td><td>29.44</td><td>25.11</td><td>0.20</td><td>0.59</td><td></td><td>0.40</td><td>0.65</td><td>0.22</td><td>0.48</td><td>0.26</td><td>0.40</td></tr><tr><td>Basis Sharing [9]</td><td></td><td>29.12</td><td>22.46</td><td>0.27</td><td>0.63</td><td></td><td>0.40</td><td>0.68</td><td>0.24</td><td>0.63</td><td>0.30</td><td>0.45</td></tr><tr><td>DictLLM (Ours)</td><td>7.79</td><td>17.62</td><td>15.20</td><td>0.29</td><td>0.66</td><td></td><td>0.46</td><td>0.71</td><td>0.24</td><td>0.66</td><td>0.35</td><td>0.48</td></tr><tr><td rowspan="4">1.67×</td><td>ASVD [11]</td><td>1407</td><td>3292</td><td>1109</td><td>0.13</td><td></td><td>0.48</td><td>0.26</td><td>0.55</td><td>0.19</td><td>0.28</td><td>0.22</td><td>0.30</td></tr><tr><td>SVD-LLM [8]</td><td>13.11</td><td>63.75</td><td>49.83</td><td>0.19</td><td>0.58</td><td></td><td>0.33</td><td>0.60</td><td>0.21</td><td>0.42</td><td>0.25</td><td>0.37</td></tr><tr><td>Basis Sharing [9]</td><td>12.39</td><td>55.78</td><td>41.28</td><td>0.22</td><td>0.61</td><td></td><td>0.35</td><td>0.62</td><td>0.23</td><td>0.52</td><td>0.27</td><td>0.40</td></tr><tr><td>DictLLM (Ours)</td><td>9.37</td><td>29.75</td><td>23.08</td><td>0.27</td><td></td><td>0.64</td><td>0.40</td><td>0.68</td><td>0.23</td><td>0.63</td><td>0.30</td><td>0.45</td></tr><tr><td rowspan="4">2x</td><td>ASVD [11] SVD-LLM [8]</td><td>15358</td><td>47690</td><td>27925</td><td>0.12</td><td></td><td>0.51</td><td>0.26</td><td>0.52</td><td>0.19</td><td>0.26</td><td>0.22</td><td>0.30</td></tr><tr><td></td><td>23.97</td><td>150.58</td><td>118.57</td><td>0.16</td><td>0.54</td><td></td><td>0.29</td><td>0.56</td><td>0.21</td><td>0.33</td><td>0.23</td><td>0.33</td></tr><tr><td>Basis Sharing [9]</td><td>19.99</td><td>126.35</td><td>88.44</td><td>0.18</td><td>0.57</td><td></td><td>0.31</td><td>0.58</td><td>0.22</td><td>0.42</td><td>0.23</td><td>0.36</td></tr><tr><td>DictLLM (Ours)</td><td>10.59</td><td>35.67</td><td>29.12</td><td>0.24</td><td>0.63</td><td></td><td>0.38</td><td>0.65</td><td>0.23</td><td>0.61</td><td>0.29</td><td>0.43</td></tr><tr><td>5x 6x</td><td>DictLLMFP4</td><td>8.48</td><td>21.40</td><td>17.04</td><td>0.29</td><td>0.65</td><td></td><td>0.44</td><td>0.70</td><td>0.24</td><td>0.64</td><td>0.33</td><td>0.47</td></tr><tr><td></td><td>(Ours)</td><td>10.37</td><td>35.16</td><td>27.07</td><td>0.26</td><td></td><td>0.61</td><td>0.38</td><td>0.66</td><td>0.23</td><td>0.59</td><td>0.28</td><td>0.43</td></tr><tr><td>7x</td><td></td><td>11.92</td><td>43.14</td><td>33.90</td><td>0.24</td><td></td><td>0.59</td><td>0.35</td><td>0.63</td><td>0.22</td><td>0.54</td><td>0.26</td><td>0.40</td></tr></table>

## 4.2 Experimental results

Benchmark on generation and reasoning datasets. We report language modeling perplexity and zero-shot reasoning accuracy in Table 1. DictLLM consistently improves over low-rank baselines [8, 9] across compression ratios from 1.25× to 2×. For example, DictLLM achieves an average zero-shot accuracy of 0.43 at 2× compression, compared with 0.40 for Basis Sharing at 1.67× compression. We also report FP4-quantized variants, DictLLM , at the bottom of Table 1. Notably, DictLL $\mathbf { \mathcal { M } } _ { \mathrm { F P 4 } }$ achieves an average accuracy of 0.40 at 7× compression, outperforming Basis Sharing at 2× compression, which achieves 0.36 average accuracy. We provide the sparsity level to reach the desired compression ratio in Table A1.

Perplexity-model size trade-off. We evaluate the trade-off between WikiText-2 perplexity and model size ratio in Fig. 4. DictLLM achieves a 5.1× compression ratio on the original LLaMA model with only a 2.9-point increase in WikiText-2 perplexity, providing 3.6× higher compression than the state-of-the-art low-rank method. In addition, DictLLM reduces perplexity by 9.4 points compared to Basis Sharing [9] with 2× compression on WikiText-2. DictLLM provides a continuous compression curve by controlling the parameter count via two knobs: dictionary size and coefficient sparsity. As the target model size decreases, perplexity gradually increases, reflecting stronger compression applied to both shared dictionaries and layer-specific

![](images/15ad2547af7a4916439afb37d20948f85b4a204fa33fe26eb00ae4e2cce540dd.jpg)  
Figure 4: Perplexity-model size trade-off. WikiText-2 perplexity of compressed LLaMA-7B across factorization ratios and coefficient sparsities. DictLLM factorizes weights at rate $\rho _ { \mathrm { f a c } }$ and then prunes layer-specific coefficients with 20%– 50% sparsity. We also report their NVFP4 versions.

Table 2: Ablation of pruning target. We ablate the pruning target with sparsity p: the dictionary D, its coefficients C, or both. Perplexity is measured on WikiText-2 using LLaMA-7B, yielding an overall compression gain of 1.43×.
<table><tr><td>Dict. D (p)</td><td>Coef. C (p)</td><td>Perplexity ↓</td></tr><tr><td>Dense (0%)</td><td>Dense (0%)</td><td>9.25</td></tr><tr><td>Dense (0%)</td><td>Sparse (19.4%)</td><td>7.83</td></tr><tr><td>Sparse (35.0%)</td><td>Dense (0%)</td><td>9.04</td></tr><tr><td>Sparse (12.5%)</td><td>Sparse (12.5%)</td><td>7.99</td></tr></table>

Table 3: Choice of pruning method. We prune the coefficients of LLaMA-7B based on magnitude, full Hessian inversion, and Hessian with lazy updates. We measure the perplexity on WikiText-2 with 50% sparsity on a 20% factorized model.
<table><tr><td>Method</td><td>Perplexity ↓</td><td>Time ↓</td></tr><tr><td>Magnitude</td><td>47.43</td><td>44.6 min</td></tr><tr><td>Full inversion</td><td>9.89</td><td>13.49 h</td></tr><tr><td>Lazy update (Ours)</td><td>9.89</td><td>55.6 min</td></tr></table>

coefficients. We first determine the rank k from the factorization compression rate $\rho _ { \mathrm { f a c } } .$ , then apply 20%–50% sparsity to the per-layer coefficients. The final compression gain is therefore determined jointly by $\rho _ { \mathrm { f a c } }$ and the coefficient sparsity p.

Ablation of pruning target. Table 2 examines whether sparsity should be applied to the shared dictionary, the layer-specific coefficients, or both. We control the sparsity of the pruning target to reach a 30% compression ratio. The most interesting finding is that pruning the coefficients gives the lowest perplexity of all combinations. This suggests that the dictionary stores shared information reused across layers, whereas the coefficients exhibit more layer-specific redundancy. Therefore, DictLLM keeps the dictionary dense and applies sparsity only to the coefficients.

Choice of pruning method. We compare three coefficient pruning strategies to isolate the effect of dictionary-aware Hessian compensation (Table 3). Magnitude pruning is fast but ignores activation sensitivity and the dictionary, leading to significant degradation in perplexity. Full Hessian inversion gives much better perplexity by compensating for pruning errors, but its cost is impractical for repeated compression. Our dictionary-aware lazy-update method preserves the same curvature information while avoiding the expensive full inverse update. As shown in Table 3, it matches the perplexity of the full inversion method while being 14.6× faster, with a duration similar to the magnitude-based method.

Ablation of dictionary update. We ablate the effect of updating the dictionary after coefficient pruning (Fig. 5). We keep the sparse coefficient matrices fixed and refit the shared dictionary across a coefficient sparsity range from 20% to 50%. This step improves the reconstruction obtained with the fixed sparse codes without changing the sparsity pattern or the model-size estimate. The dictionary update reduces perplexity compared to the pruned baseline. The interesting part is that perplexity reduction is higher at increased sparsity: 1.04 PPL at 50% sparsity, whereas 0.06 PPL at 20% sparsity. This suggests that coefficient pruning yields a suboptimal dictionary for the remaining active atoms, and that the dictionary refit alleviates some of the resulting increase in perplexity.

![](images/a5ce6dcb5ee4b2f34603b76051cd6f23c5185acc422b9ac642542e321402b882.jpg)  
Figure 5: Ablation of dictionary update. WikiText-2 perplexity of LLaMA-7B across different sparsity levels, with and without dictionary update.

Different quantization format. We investigate how different precision choices for the dictionary and coefficients affect the final compression trade-off (Table 4). The FP16/FP16 baseline corresponds to the factorized model with a 30% reduction in factorizations and 50% coefficient sparsity before quantization. Quantizing only the dictionary to NVFP4 yields a clear storage gain with a small increase in perplexity, suggesting that the shared dictionary is relatively robust to low-bit quantization. When both the dictionary and coefficients are quantized to NVFP4, perplexity remains close to the INT8-coefficient setting while using almost half the precision. Although NVFP4/FP8 mixed quantization exhibits a decent perplexity-size trade-off, our all-NVFP4 setting yields the most compact representation and pushes the model-size frontier further to the left.

Table 4: Different quantization format. We compare quantization choices for the shared dictionary D and layer-specific coefficients C. Perplexity is measured on WikiText-2 using LLaMA-7B, and compression gain denotes the reduction in model size. The FP16/FP16 baseline corresponds to the factorized model with a 30% factorization reduction and 50% coefficient sparsity before quantization.
<table><tr><td>Method</td><td>Dictionary D</td><td>Coefficient C</td><td>PPL↓</td><td>Comp. gain ↑</td></tr><tr><td>Unquantized baseline</td><td>FP16</td><td>FP16</td><td>12.98</td><td>1.00×</td></tr><tr><td>Mixed precision 1</td><td>NVFP4</td><td>FP16</td><td>13.66</td><td>1.61×</td></tr><tr><td>Mixed precision 2</td><td>NVFP4</td><td>FP8</td><td>14.71</td><td>2.60×</td></tr><tr><td>Mixed precision 3</td><td>NVFP4</td><td>INT8</td><td>16.05</td><td>2.60×</td></tr><tr><td>All NVFP4 (Ours)</td><td>NVFP4</td><td>NVFP4</td><td>16.82</td><td>3.56×</td></tr></table>

Table 5: Generalization across model families. Per- Table 6: Effect of LoRA fine-tuning. plexity (↓) of three different LLMs, e.g., LLaMA 2-7B, WikiText-2 perplexity (↓) of LLaMA-7B OPT-6.7B, and Mistral-7B, under 1.25× compression with post-compression LoRA fine-tuning on WikiText-2. across compression gains.
<table><tr><td>Method</td><td>LLaMA2-7B</td><td>OPT-6.7B</td><td>Mistral-7B</td></tr><tr><td>ASVD [11]</td><td>10.10</td><td>82.0</td><td>13.72</td></tr><tr><td>SVD-LLM [8]</td><td>8.5</td><td>16.04</td><td>10.21</td></tr><tr><td>Basis Sharing [9]</td><td>7.70</td><td>11.79</td><td>7.57</td></tr><tr><td>DictLLM (Ours)</td><td>6.93</td><td>11.36</td><td>6.46</td></tr></table>

<table><tr><td>Comp. gain</td><td>Basis Sharing [9]</td><td>DictLLM</td></tr><tr><td>1.25×</td><td>7.09</td><td>6.42</td></tr><tr><td>1.43×</td><td>8.21</td><td>7.54</td></tr><tr><td>1.67×</td><td>10.44</td><td>7.91</td></tr><tr><td>2x</td><td>10.95</td><td>8.71</td></tr></table>

Generalization across model families. Table 5 reports WikiText-2 perplexity at 1.25× compression for OPT-6.7B, LLaMA 2-7B, and Mistral-7B. DictLLM consistently outperforms ASVD, SVD-LLM, and Basis Sharing across all three architectures. These results suggest that dictionary factorization and dictionary-aware Hessian pruning generalize across model families.

Effect of LoRA fine-tuning. Table 6 reports the effect of applying low-rank adaptation (LoRA) fine-tuning [55] after dictionary factorization and coefficient pruning. LoRA fine-tuning consistently recovers a portion of the perplexity lost during compression, reducing WikiText-2 perplexity by up to 1.9 points. The LoRA fine-tuning is more beneficial at higher compression ratios. We report LoRA fine-tuning for both Basis Sharing and DictLLM updating only query and value weights.

![](images/c02fec24798a2f850df38e29c7f9ff49fd6df99cad44fc1dd7a99aa3e44d3904.jpg)  
Figure 6: Effect of calibration size. WikiText-2 perplexity of LLaMA-7B as a function of calibration set size.

Effect of calibration set size. We vary the number of calibration samples from 32 to 256 and measure WikiText-2 perplexity of LLaMA-7B at 1.43× compression (Fig. 6). DictLLM consistently outperforms Basis Sharing across all calibration sizes, while both DictLLM and Basis Sharing improve as more calibration data is available. Notably, DictLLM, using only 32 samples, outperforms Basis Sharing with 256 samples.

## 5 Conclusion

We present DictLLM, a post-training compression method that represents transformer weights with shared dictionary kernels and layer-specific sparse coefficients. DictLLM first recovers shared dictio naries via activation-aware factorization, then prunes the coefficients using a dictionary-aware Hessian that projects the calibration covariance into the coefficient space. We then refit the shared dictionary by using the resulting coefficients. We further apply NVFP4 quantization with Hessian-based error compensation to both dictionary and coefficient factors, reducing storage while maintaining the factorized structure. Comprehensive experiments show that DictLLM improves the perplexity-model size trade-off and preserves performance relative to state-of-the-art low-rank compression methods across language modeling and zero-shot reasoning benchmarks.

## Acknowledgments and disclosure of funding

This work was supported in part by COGNISENSE, one of seven centers in JUMP 2.0, a Semiconductor Research Corporation (SRC) program sponsored by DARPA. The work of SY Chun was supported by IITP grants funded by the Korea government(MSIT) [No.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University) / No.RS-2025-02314125, Effective Human-Machine Teaming With Multimodal Hazy Oracle Models].

## References

[1] Mark Horowitz. 1.1 Computing’s Energy Problem (and what we can do about it). In 2014 IEEE international solid-state circuits conference digest of technical papers (ISSCC), pages 10–14. IEEE, 2014.

[2] Yu-Hsin Chen, Tushar Krishna, Joel S Emer, and Vivienne Sze. Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks. IEEE Journal ofSolid-State Circuits, 52(1): 127–138, 2016.

[3] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: An Insightful Visual Performance Model for Multicore Architectures. Communications ofthe ACM, 52(4):65–76, 2009.

[4] Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. In-Datacenter Performance Analysis of a Tensor Processing Unit. In Proceedings of the 44th annual international symposium on computer architecture, pages 1–12, 2017.

[5] Jaewon Lee, Gain Kim, Jinho Park, and Hyeon-Min Bae. Link bit-error-rate requirement analysis for deep neural network accelerators. In 2021 IEEE International Symposium on Circuits and Systems (ISCAS), pages 1–5. IEEE, 2021.

[6] Carl Eckart and Gale Young. The Approximation of One Matrix by Another of Lower Rank. Psychometrika, 1(3):211–218, 1936.

[7] Changwoo Lee, Soo Min Kwon, Qing Qu, and Hun-Seok Kim. BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inference. Advances in Neural Information Processing Systems, 37:14996–15027, 2024.

[8] Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression. In The Thirteenth International Conference on Learning Representations, 2025.

[9] Jingcun Wang, Yu Guang Chen, Chao Lin, Bing Li, and Grace Li Zhang. Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression. In The Thirteenth International Conference on Learning Representations, 2025.

[10] Qinsi Wang, Jinghan Ke, Masayoshi Tomizuka, Kurt Keutzer, and Chenfeng Xu. Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives. In The Thirteenth International Conference on Learning Representations, 2025.

[11] Zhihang Yuan, Yuzhang Shang, Yue Song, Dawei Yang, Qiang Wu, Yan Yan, and Guangyu Sun. ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models. arXiv preprint arXiv:2312.05821, 2023.

[12] Yu-Chen Lu, Sheng-Feng Yu, Hui-Hsien Weng, Pei-Shuo Wang, Yu-Fang Hu, Liang Hung-Chun, Hung-Yueh Chiang, and Kai-Chiang Wu. SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 24124–24132, 2026.

[13] Song Han, Huizi Mao, and William J Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. arXiv preprint arXiv:1510.00149, 2015.

[14] Elias Frantar and Dan Alistarh. SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot. In International Conference on Machine Learning, pages 10323–10337. PMLR, 2023.

[15] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A Simple and Effective Pruning Approach for Large Language Models. Proceedings ofthe International Conference on Learning Representations, 2024.

[16] NVIDIA. NVIDIA Blackwell Architecture Technical Brief, 2024. URL https://resources.nvidia. com/en-us-blackwell-architecture.

[17] Vage Egiazarian, Roberto L Castro, Denis Kuznedelev, Andrei Panferov, Eldar Kurtic, Shubhra Pandit, Alexandre Marques, Mark Kurtz, Saleh Ashkboos, Torsten Hoefler, et al. Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization. Proceedings ofthe International Conference on Learning Representations, 2026.

[18] Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. QuIP: 2-Bit Quantization of Large Language Models With Guarantees. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[19] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. Proceedings ofthe International Conference on Learning Representations, 2023.

[20] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. In International Conference on Machine Learning, pages 38087–38099. PMLR, 2023.

[21] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. Advances in Neural Information Processing Systems, 37:100213–100240, 2024.

[22] Jinho Park, Jaewon Lee, Gain Kim, and Hyeon-Min Bae. Bin-Specific Quantization in Spectral-Domain Convolutional Neural Network Accelerators. In 2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS), pages 407–410. IEEE, 2022.

[23] Seunghyun Moon, Jinho Park, Chuan-Tung Lin, SeongHwan Cho, Gregory Chen, Phil Knag, Ram Krishnamurthy, and Mingoo Seok. PersASR: A 2.3-µJ/frame 96.8%-accurate TinyML Automatic Speech Recognition Processor with Data Augmentation-Driven Personalization in 16-nm FinFET. In 2026 IEEE Custom Integrated Circuits Conference (CICC), pages 1–4. IEEE, 2026.

[24] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. In International Conference on Learning Representations, 2020.

[25] Qian Lou, Ting Hua, Yen-Chang Hsu, Yilin Shen, and Hongxia Jin. DictFormer: Tiny Transformer with Shared Dictionary. In International Conference on Learning Representations, 2022.

[26] Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster. Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA. Proceedings of the International Conference on Learning Representations, 2025.

[27] Tamir David Hay and Lior Wolf. Dynamic Layer Tying for Parameter-Efficient Transformers. In The Twelfth International Conference on Learning Representations, 2024.

[28] Seunghyun Moon, Mao Li, Gregory K Chen, Phil C Knag, Ram Kumar Krishnamurthy, and Mingoo Seok. T-REX: A 68-to-567µs/Token 0.41-to-3.95 µJ/Token Transformer Accelerator with Reduced External Memory Access and Enhanced Hardware Utilization in 16nm FinFET. In 2025 IEEE International Solid-State Circuits Conference (ISSCC), volume 68, pages 406–408. IEEE, 2025.

[29] Kjersti Engan, Sven Ole Aase, and J Hakon Husoy. Method of optimal directions for frame design. In 1999 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings. ICASSP99 (Cat. No. 99CH36258), volume 5, pages 2443–2446. IEEE, 1999.

[30] Michal Aharon, Michael Elad, and Alfred Bruckstein. K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Transactions on signal processing, 54(11):4311–4322, 2006.

[31] Magauiya Zhussip, Dmitriy Shopkhoev, Ammar Ali, and Stamatios Lefkimmiatis. Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 29260–29268, 2026.

[32] Yann LeCun, John Denker, and Sara Solla. Optimal Brain Damage. Advances in Neural Information Processing Systems, 2, 1989.

[33] Kwanhee Lee, Hyeondo Jang, Dongyeop Lee, Dan Alistarh, and Namhoon Lee. The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM. In The Fourteenth International Conference on Learning Representations, 2026.

[34] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. Proceedings ofmachine learning and systems, 6:87–100, 2024.

[35] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. Deep Equilibrium Models. Advances in Neural Information Processing Systems, 32, 2019.

[36] Yuesen Liao, Zhiwei Li, Binrui Wu, Zihao Cheng, Su Zhao, Shuai Chen, and Weizhong Zhang. Compress Large Language Models via Collaboration Between Learning and Matrix Approximation. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025.

[37] Changhai Zhou, Qian Qiao, Yuhua Zhou, Yuxin Wu, Shichao Weng, Weizhong Zhang, and Cheng Jin. Large Language Model Compression with Global Rank and Sparsity Optimization. In The Fourteenth International Conference on Learning Representations, 2026.

[38] Jialin Zhao, Yingtao Zhang, and Carlo Vittorio Cannistraci. Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models. In Forty-second International Conference on Machine Learning, 2025.

[39] Yi Li, Zhichun Guo, Miao Yin, and Bingzhe Li. LeSTD: LLM Compression via Learning-based Sparse Tensor Decomposition. In The Fourteenth International Conference on Learning Representations, 2026.

[40] Denis Makhov, Dmitriy Shopkhoev, Magauiya Zhussip, Ammar Ali, and Stamatios Lefkimmiatis. CoSpaDi: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning. arXiv preprint arXiv:2509.22075, 2025.

[41] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971, 2023.

[42] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. OPT: Open Pre-trained Transformer Language Models. arXiv preprint arXiv:2205.01068, 2022.

[43] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.

[44] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Models. In International Conference on Learning Representations, 2017.

[45] Mitch Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a Large Annotated Corpus of English: The Penn Treebank. Computational linguistics, 19(2):313–330, 1993.

[46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal ofmachine learning research, 21(140):1–67, 2020.

[47] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In Proceedings ofthe 2018 conference on empirical methods in natural language processing, pages 2381–2391, 2018.

[48] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740, 2020.

[49] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a Machine Really Finish Your Sentence? In Proceedings ofthe 57th annual meeting ofthe associationfor computational linguistics, pages 4791–4800, 2019.

[50] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. PIQA: Reasoning about Physical Commonsense in Natural Language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439, 2020.

[51] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2357–2367, 2019.

[52] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv preprint arXiv:1803.05457, 2018.

[53] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The Language Model Evaluation Harness, 07 2024. URL https://zenodo.org/records/12608602.

[54] Roberto L. Castro and Dan Alistarh. QuTLASS: CUTLASS-Powered Quantized BLAS for Deep Learning. https://github.com/IST-DASLab/qutlass, 2025.

[55] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. LoRA: Low-Rank Adaptation of Large Language Models. Proceedings of the International Conference on Learning Representations, 2022.

[56] Xin Wang, Samiul Alam, Zhongwei Wan, Hui Shen, and Mi Zhang. SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4287–4296, 2025.

[57] Vladimír Macko and Vladimír Boža. MACKO: Sparse Matrix-Vector Multiplication for Low Sparsity. In Forty-third International Conference on Machine Learning, 2025.

[58] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388, 2025.

## Appendix

## Appendix contents

A1 Additional details 14   
A1.1 Additional details on dictionary learning 14   
A1.2 Compression and number of parameters 15   
A1.3 Dictionary-induced coefficient Hessian. 15   
A2 Additional experimental details . 16   
A3 Generalization across model families and scales 17   
A4 Componentwise ablation 18   
A5 Spectral analysis and motivation for cross-layer weight sharing . 18   
A6 Aggressive layer sharing with LoRA fine-tuning. 20   
A7 Comparison with additional baselines 20   
A8 End-to-end decoding throughput and memory 22   
A9 Assets and compute resources 23   
A9.1 Assets 23   
A9.2 23   
A10 Discussions 23

## A1 Additional details

## A1.1 Additional details on dictionary learning

Relation to k-SVD. DictLLM is inspired by sparse dictionary learning, but its objective differs from classical k-SVD [30]. Given training signals $\mathbf { \bar { Y } } = [ \mathbf { y } _ { 1 } , \dots , \mathbf { \bar { y } } _ { N } ]$ , k-SVD learns an overcomplete dictionary $\mathbf { D } \in \mathbb { R } ^ { d \times k }$ with $k > d$ and sparse codes A by solving

$$
\operatorname* { m i n } _ { \mathbf { D } , \mathbf { A } } \left\| \mathbf { Y } - \mathbf { D A } \right\| _ { F } ^ { 2 } \quad \mathrm { s . t . } \quad \left\| \mathbf { a } _ { i } \right\| _ { 0 } \leq s , \forall i .\tag{A1}
$$

Although k-SVD is often used with an overcomplete dictionary, our comparison uses an undercomplete dictionary $( k < d _ { \mathrm { i n } } )$ to match the compression setting. k-SVD is computationally expensive because it updates dictionary atoms and their associated sparse coefficients one at a time by repeatedly forming residuals and solving local rank-one approximation problems.

In contrast, DictLLM does not learn a dictionary from raw data signals. Instead, it starts from pretrained weights and minimizes the activation-space reconstruction error induced by replacing each dense weight with a shared dictionary and sparse coefficients:

$$
\operatorname* { m i n } _ { \mathbf { D } ^ { g } , \{ \mathbf { C } ^ { \ell } \} _ { \ell \in g } } \sum _ { \ell \in g } \big \| \mathbf { X } ^ { \ell } \mathbf { W } ^ { \ell } - \mathbf { X } ^ { \ell } \mathbf { D } ^ { g } \mathbf { C } ^ { \ell } \big \| _ { F } ^ { 2 } \quad \mathrm { s . t . } \quad \| \mathbf { C } _ { : , j } ^ { \ell } \| _ { 0 } \leq ( 1 - p ) k .\tag{A2}
$$

Here, $\mathbf { D } ^ { g } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times k }$ is a compression dictionary whose size k is chosen from the target parameter budget, rather than an overcomplete dictionary. This makes the problem closer to post-training model compression than classical sparse coding. Accordingly, DictLLM uses a scalable staged approximation: activation-aware SVD initializes the dictionary, Hessian-aware pruning chooses the sparse coefficient support, and a MOD-style update refits the dictionary to the fixed sparse coefficients.

Relation to Method of Optimal Directions. The Method of Optimal Directions (MOD) [29] is a classical dictionary-learning algorithm that alternates between estimating sparse codes and updating the dictionary to better reconstruct training signals. Given fixed sparse codes, MOD updates all dictionary atoms at once by solving a least-squares problem. Our refitting step uses the same fixedcode dictionary update idea, but in a post-training model-compression setting. Instead of learning a dictionary from raw signals, we start from pretrained transformer weights, keep the pruned coefficient matrices fixed, and refit the shared dictionary to reduce the reconstruction error of the original weights. Thus, our update is a lightweight post-pruning refinement rather than a full alternating dictionary-learning procedure.

## A1.2 Compression and number of parameters

Modeling number of parameters in LLMs with dictionary and coefficients. Let T denote the set of weight types, such as attention and MLP projection matrices. Assuming that each weight $\mathbf { W } _ { t } ^ { \ell } \in \mathbb { R } ^ { d _ { \mathrm { i n } , t } \times d _ { \mathrm { o u t } , t } ^ { \star } }$ , the total number of parameters in the original model is

$$
N _ { \mathrm { o r i g i n a l } } = \sum _ { \ell = 1 } ^ { L } \sum _ { t \in { \mathcal { T } } } \left| \mathbf { W } _ { t } ^ { \ell } \right| = \sum _ { \ell = 1 } ^ { L } \sum _ { t \in { \mathcal { T } } } d _ { \mathrm { i n } , t } ^ { \ell } d _ { \mathrm { o u t } , t } ^ { \ell } ,\tag{A3}
$$

where $\left| \mathbf { W } _ { t } ^ { \ell } \right|$ denotes the number of scalar parameters in weight matrix $\mathbf { W } _ { t } ^ { \ell }$ . This count includes only the transformer projection matrices compressed by DictLLM. It excludes the tokenizer, embedding tables, normalization parameters, biases, and the language modeling head, which are kept unchanged. DictLLM’s grouped dictionary parameter count can naturally follow as

$$
N _ { \mathrm { D i c t L L M } } = \sum _ { t \in { \cal T } } \sum _ { g \in { \cal G } _ { t } } \left( d _ { \mathrm { i n } , t } ^ { g } k _ { t } ^ { g } + ( 1 - p _ { t } ^ { g } ) k _ { t } ^ { g } \sum _ { \ell \in g } d _ { \mathrm { o u t } , t } ^ { \ell } \right) .\tag{A4}
$$

Compression ratio and sparsity. We first describe how factorization determines the dictionary rank for one group of weights. Consider n matrices with the same shape $d _ { \mathrm { i n } } \times d _ { \mathrm { o u t } }$ that share one dictionary of rank k. Before coefficient pruning, i.e., with sparsity $p = 0 ,$ , the factorization compression ratio is

$$
\rho _ { \mathrm { f a c } } = 1 - { \frac { d _ { \mathrm { i n } } k + n k d _ { \mathrm { o u t } } } { n d _ { \mathrm { i n } } d _ { \mathrm { o u t } } } } .\tag{A5}
$$

Solving for k gives

$$
k = { \frac { n ( 1 - \rho _ { \mathrm { f a c } } ) d _ { \mathrm { i n } } d _ { \mathrm { o u t } } } { d _ { \mathrm { i n } } + n d _ { \mathrm { o u t } } } } .\tag{A6}
$$

For example, the query projection in LLaMA-7B has $d _ { \mathrm { i n } } = d _ { \mathrm { o u t } } = 4 0 9 6$ . With $\rho _ { \mathrm { f a c } } = 2 0 \%$ and a sharing group of $n = 2$ layers, this gives $k = 2 1 8 4$ . In contrast, the down projection has shape $1 1 0 0 8 \times 4 0 9 6$ and is not shared across layers following Basis Sharing, so $n = 1$ and $k = 2 3 8 8$ . We choose the dictionary rank separately for each weight type so that the factorization compression ratio is matched across weight types.

After the rank is fixed, we apply sparsity level p to the coefficients, leaving only ${ \mathfrak { a } } \left( 1 - p \right)$ fraction of coefficient entries nonzero. The resulting final compression ratio is

$$
\rho = 1 - \frac { d _ { \mathrm { i n } } k + k d _ { \mathrm { o u t } } n ( 1 - p ) } { d _ { \mathrm { i n } } d _ { \mathrm { o u t } } n } .\tag{A7}
$$

We apply the same coefficient sparsity across all weight types, so the resulting compression ratio can vary by weight type. For instance, when pruning a LLaMA-7B model factorized with a 20% reduction and layer sharing group size $n = 2$ , we use coefficient sparsity $p = 0 . 1 9 4$ to reach 30% overall compression. Under this setting, the effective compression by weight type is 30.36% for Q, K, and V projections, 33.10% for up and gate projections, 27.78% for output projections, and 24.21% for down projections.

## A1.3 Dictionary-induced coefficient Hessian.

Let $\widetilde { \mathbf { C } } ^ { \ell } = \mathbf { C } ^ { \ell } + \Delta \mathbf { C } ^ { \ell }$ be the pruned coefficient. Since the dictionary is fixed, the induced weight perturbation is $\Delta \mathbf { W } ^ { \ell } = \mathbf { D } ^ { g } \bar { \Delta \mathbf { C } } ^ { \ell }$ . Under the calibration reconstruction objective, the local pruning loss is

$$
\left\| \mathbf { X } ^ { \ell } \Delta \mathbf { W } ^ { \ell } \right\| _ { F } ^ { 2 } = \left\| \mathbf { X } ^ { \ell } \mathbf { D } ^ { g } \Delta \mathbf { C } ^ { \ell } \right\| _ { F } ^ { 2 } = \operatorname { T r } \left( ( \Delta \mathbf { C } ^ { \ell } ) ^ { \top } ( \mathbf { D } ^ { g } ) ^ { \top } ( \mathbf { X } ^ { \ell } ) ^ { \top } \mathbf { X } ^ { \ell } \mathbf { D } ^ { g } \Delta \mathbf { C } ^ { \ell } \right) .\tag{A8}
$$

For a single layer, this derivation gives the coefficient-space Hessian

$$
\mathbf { H } ^ { \mathbf { C } , \ell } = ( \mathbf { D } ^ { g } ) ^ { \top } \mathbf { S } ^ { \ell } \mathbf { D } ^ { g } .\tag{A9}
$$

In practice, we use a shared group-level Hessian proxy $\mathbf { H } ^ { \mathbf { C } , g } = ( \mathbf { D } ^ { g } ) ^ { \top } \mathbf { S } ^ { g } \mathbf { D } ^ { g }$ , where $\begin{array} { r } { \mathbf { S } ^ { g } = \sum _ { \ell \in g } \mathbf { S } ^ { \ell } } \end{array}$ This pooled covariance approximation avoids computing and storing separate Hessian inverses for every layer while preserving activation-aware sensitivity at the group level.

## A2 Additional experimental details

Our comparison excludes language modeling heads and embeddings, following the protocol of prior work; we count the number of non-zero entries in the coefficient matrices. Unless otherwise specified, we use damping $\lambda = 1 0 ^ { - 4 }$ when regularizing covariance and Hessian matrices before inversion or Cholesky decomposition. We use a relative Hessian damping factor of 0.1 for masked quantization. For LM Harness [53] evaluation, we report acc as the accuracy of all reasoning tasks instead of acc\_norm. For PTB [45] evaluation, we strip leading and trailing whitespace from each test sentence and join sentences with double newlines before tokenization, following our original evaluation protocol

Table A1: Factorization and coefficient sparsity settings for LLaMA-7B. $\rho _ { \mathrm { f a c } }$ denotes the parameter reduction from factorization, and sparsity denotes the fraction of pruned coefficient entries.  
Table A2: Coefficient sparsity settings across LLaMA2-7B, OPT-6.7B, and Mistral-7B. All models have a factorization ratio $\rho _ { \mathrm { f a c } } = 1 0 \%$ and corresponding coefficient sparsities to achieve 1.25× compression gain.
<table><tr><td>Compression gain</td><td>Factorization ratio  $\rho _ { \mathrm { f a c } }$ </td><td>Sparsity p</td></tr><tr><td>1.25×</td><td>10%</td><td>17.25%</td></tr><tr><td>1.43×</td><td>20%</td><td>19.40%</td></tr><tr><td>1.67×</td><td>30%</td><td>22.17%</td></tr><tr><td>2x</td><td>30%</td><td>44.38%</td></tr></table>

<table><tr><td>Models</td><td>LLaMA2-7B</td><td>OPT-6.7B</td><td>Mistral-7B</td></tr><tr><td>Sparsity p</td><td>17.25%</td><td>19.42%</td><td>17.66%</td></tr></table>

For the DictLLM results in Table 1, we report the reduction from factorization and coefficient sparsity in Table A1. All compressed models are calibrated on WikiText-2 and evaluated without task-specific fine-tuning. Quantized models DictL $\mathrm { \bf { L M } } _ { \mathrm { { F P 4 } } }$ reported at the bottom of Table 1 have $\rho _ { \mathrm { f a c } } = \{ 2 0 \% , 3 0 \% , 3 0 \% \}$ with sparsity $p = \{ 1 9 . 4 0 \% , 2 2 . 1 \bar { 7 } \% , 4 4 . 3 8 \% \}$ , achieving compression gains of 5.08×, 5.94×, and 7.12×.

In Table 2, the dense-dense setting reaches 1.43× compression using factorization alone. For the remaining settings, we start from a factorized model with $\rho _ { \mathrm { f a c } } = 2 0 \%$ and apply pruning to the dictionary, the coefficients, or both, using the sparsity levels p listed in the table.

In Table 5, we use models factorized with $\rho _ { \mathrm { f a c } } = 1 0 \%$ and sparsify the layer-specific coefficients with sparsity of 17.25% for LLaMA2-7B, 19.42% for OPT-6.7B, and 17.66% for Mistral-7B as described in Table A2. To reach 1.25× compression (or 20% compression rate), the effective compression rate of LLaMA2-7B by weight type is 20.37% for Q, K, and V projections, 23.09% for up and gate projections, 17.77% for output projections, and 14.23% for down projections. On the other hand, OPT-6.7B has per-weight type compression rates of 21.67% for Q, K, and V projections, 25.56% for the up projection, 18.75% for the output projections, and 13.50% for the down projections. Mistral-7B compressed by 1.25× has compression rates of 20.61% for Q, 13.21% for K and V, 17.96% for the output, 23.92% for the gate and up, and 13.54% for the down projections.

For the LoRA fine-tuning results in Table 6, we fine-tune DictLLM models at final compression gains from 1.25× to 2×, using the same factorization and pruning hyperparameters reported in Table 1. We use rank $r = 8$ adapters with a scaling factor $\alpha = 3 2$ , yielding an effective LoRA update scale of $\alpha / r = 4$ . We train only the LoRA adapter parameters while keeping the compressed backbone fixed. Fine-tuning is performed with a batch size of 1, a learning rate of $1 0 ^ { - 4 }$ , and 2 training epochs. These experiments use the factorized and pruned models without dictionary refitting.

In Figure 5, we start from a model with a 20% factorization reduction, apply coefficient pruning with sparsity levels from 20% to 50%, and then apply dictionary refitting to each setting.

For the calibration set length sweep in Figure $^ { 6 , }$ we use the 1.43× compressed DictLLM model with factorization rate $\rho _ { \mathrm { f a c } } = 2 0 \%$ , coefficient sparsity $p = 1 9 . 4 0 \%$ , and no dictionary refitting.

## A3 Generalization across model families and scales

Table A3: Benchmark on language modeling and reasoning tasks with Qwen3-8B. We compare DictLLM against low-rank compression baselines at compression gains from 1.25× to 2×. We report perplexity on WikiText-2 and C4, and zero-shot accuracy on OpenBookQA (OB), WinoGrande (WG), HellaSwag (HS), PIQA (PQ), MathQA (MQ), ARC-Easy (AE), and ARC-Challenge (AC).
<table><tr><td rowspan="2">Comp. gain</td><td rowspan="2">Method</td><td colspan="2">Perplexity↓</td><td colspan="8">Zero-shot accuracy (%) ↑</td></tr><tr><td>WikiText-2</td><td>C4</td><td>OB</td><td>WG</td><td>HS</td><td>PQ</td><td>MQ</td><td>AE</td><td>AC</td><td>Average</td></tr><tr><td>1×</td><td>Original</td><td>7.09</td><td>11.87</td><td>32.20</td><td>72.53</td><td>59.03</td><td>79.16</td><td>55.41</td><td>81.61</td><td>53.24</td><td>61.88</td></tr><tr><td rowspan="3">1.25×</td><td>SVD-LLM [8]</td><td>11.52</td><td>32.16</td><td>29.00</td><td>66.22</td><td>44.16</td><td>71.55</td><td>31.56</td><td>67.51</td><td>38.31</td><td>49.76</td></tr><tr><td>Basis Sharing [9]</td><td>11.14</td><td>29.37</td><td>31.40</td><td>67.56</td><td>45.77</td><td>71.93</td><td>32.26</td><td>67.21</td><td>38.99</td><td>50.73</td></tr><tr><td>DictLLM (Ours)</td><td>9.90</td><td>22.59</td><td>31.00</td><td>70.09</td><td>50.07</td><td>74.86</td><td>35.78</td><td>70.54</td><td>41.89</td><td>53.46</td></tr><tr><td rowspan="3">1.43×</td><td>SVD-LLM [8]</td><td>14.30</td><td>51.96</td><td>26.20</td><td>64.33</td><td>37.91</td><td>65.02</td><td>26.77</td><td>55.51</td><td>28.50</td><td>43.46</td></tr><tr><td>Basis Sharing [9]</td><td>13.29</td><td>44.72</td><td>26.60</td><td>62.98</td><td>39.76</td><td>67.74</td><td>28.68</td><td>58.08</td><td>33.11</td><td>45.28</td></tr><tr><td>DictLLM (Ours)</td><td>11.26</td><td>30.09</td><td>31.60</td><td>67.64</td><td>45.21</td><td>72.36</td><td>32.56</td><td>65.66</td><td>38.82</td><td>50.55</td></tr><tr><td rowspan="3">1.67×</td><td>SVD-LLM [8]</td><td>20.81</td><td>103.87</td><td>20.00</td><td>56.27</td><td>31.77</td><td>60.17</td><td>23.15</td><td>39.02</td><td>20.56</td><td>35.85</td></tr><tr><td>Basis Sharing [9]</td><td>17.56</td><td>84.85</td><td>21.60</td><td>58.25</td><td>33.72</td><td>61.32</td><td>24.02</td><td>47.81</td><td>24.83</td><td>38.79</td></tr><tr><td>DictLLM (Ours)</td><td>13.59</td><td>47.79</td><td>26.40</td><td>63.22</td><td>39.05</td><td>67.03</td><td>27.71</td><td>58.08</td><td>32.94</td><td>44.92</td></tr><tr><td rowspan="3">2x</td><td>SVD-LLM [8]</td><td>35.07</td><td>215.72</td><td>14.40</td><td>55.17</td><td>28.36</td><td>56.47</td><td>22.48</td><td>30.81</td><td>20.56</td><td>32.61</td></tr><tr><td>Basis Sharing [9]</td><td>29.61</td><td>191.90</td><td>15.60</td><td>55.72</td><td>29.25</td><td>58.43</td><td>22.81</td><td>37.16</td><td>21.16</td><td>34.31</td></tr><tr><td>DictLLM (Ours)</td><td>16.58</td><td>73.52</td><td>24.20</td><td>60.14</td><td>35.38</td><td>63.06</td><td>26.37</td><td>54.04</td><td>28.24</td><td>41.63</td></tr></table>

Table A4: Qwen3-1.7B language modeling and reasoning results. Methods are compared at the reported compression gains.
<table><tr><td rowspan="2">Comp. gain</td><td rowspan="2">Method</td><td rowspan="2">Perplexity ↓ WikiText-2</td><td rowspan="2">C4</td><td colspan="8">Zero-shot accuracy (%) ↑</td></tr><tr><td>OB</td><td>WG</td><td>HS</td><td>PQ</td><td>MQ</td><td>AE</td><td>AC</td><td>Average</td></tr><tr><td>1×</td><td>Original</td><td>9.51</td><td>15.34</td><td>29.80</td><td>63.54</td><td>49.29</td><td>75.73</td><td>45.19</td><td>73.74</td><td>41.04</td><td>54.05</td></tr><tr><td rowspan="3">1.25×</td><td>SVD-LLM [8]</td><td>23.20</td><td>96.37</td><td>22.80</td><td>56.83</td><td>34.98</td><td>61.64</td><td>24.15</td><td>44.15</td><td>24.66</td><td>38.46</td></tr><tr><td>Basis Sharing [9]</td><td>22.93</td><td>95.07</td><td>22.60</td><td>56.20</td><td>36.66</td><td>62.02</td><td>26.16</td><td>48.82</td><td>27.13</td><td>39.94</td></tr><tr><td>DictLLM (Ours)</td><td>17.66</td><td>52.80</td><td>25.80</td><td>59.91</td><td>40.63</td><td>67.85</td><td>27.84</td><td>53.37</td><td>28.75</td><td>43.45</td></tr><tr><td rowspan="3">1.43×</td><td>SVD-LLM [8]</td><td>34.60</td><td>190.49</td><td>19.00</td><td>55.01</td><td>31.08</td><td>58.32</td><td>23.75</td><td>38.13</td><td>22.95</td><td>35.46</td></tr><tr><td>Basis Sharing [9]</td><td>33.37</td><td>206.33</td><td>21.00</td><td>55.09</td><td>32.44</td><td>58.60</td><td>23.79</td><td>42.63</td><td>23.38</td><td>36.70</td></tr><tr><td>DictLLM (Ours)</td><td>23.45</td><td>99.27</td><td>21.80</td><td>56.27</td><td>36.39</td><td>62.30</td><td>25.83</td><td>47.52</td><td>25.51</td><td>39.37</td></tr></table>

Table A5: Qwen3-14B language modeling and reasoning results. The improvements over other baselines extend to the largest model evaluated in language modeling and reasoning tasks.
<table><tr><td rowspan="2">Comp. gain</td><td rowspan="2">Method</td><td rowspan="2">Perplexity↓ WikiText-2 C4</td><td rowspan="2"></td><td colspan="8">Zero-shot accuracy (%) ↑</td></tr><tr><td>OB</td><td>WG</td><td>HS</td><td>PQ</td><td>MQ</td><td>AE</td><td>AC</td><td>Average</td></tr><tr><td>1×</td><td>Original</td><td>6.46</td><td>10.95</td><td>35.40</td><td>74.19</td><td>61.75</td><td>80.41</td><td>61.78</td><td>83.33</td><td>55.63</td><td>64.64</td></tr><tr><td rowspan="3">1.25×</td><td>SVD-LLM [8]</td><td>9.49</td><td>27.15</td><td>28.80</td><td>68.98</td><td>47.19</td><td>72.69</td><td>34.34</td><td>67.00</td><td>38.48</td><td>51.07</td></tr><tr><td>Basis Sharing [9]</td><td>9.18</td><td>25.93</td><td>30.20</td><td>70.24</td><td>49.32</td><td>74.32</td><td>36.98</td><td>66.08</td><td>39.33</td><td>52.35</td></tr><tr><td>DictLLM (Ours)</td><td>8.29</td><td>19.69</td><td>32.20</td><td>70.32</td><td>54.48</td><td>76.22</td><td>39.90</td><td>73.02</td><td>43.60</td><td>55.68</td></tr><tr><td rowspan="3">1.43×</td><td>SVD-LLM [8]</td><td>11.51</td><td>44.54</td><td>26.00</td><td>65.11</td><td>39.29</td><td>67.08</td><td>29.95</td><td>62.25</td><td>31.57</td><td>45.89</td></tr><tr><td>Basis Sharing [9]</td><td>10.81</td><td>40.77</td><td>28.80</td><td>68.67</td><td>42.67</td><td>70.24</td><td>32.83</td><td>60.35</td><td>35.07</td><td>48.38</td></tr><tr><td>DictLLM (Ours)</td><td>9.28</td><td>26.46</td><td>30.00</td><td>70.32</td><td>48.87</td><td>73.99</td><td>36.05</td><td>66.08</td><td>38.91</td><td>52.03</td></tr></table>

Tables A3, A4, and A5 compare DictLLM with SVD-LLM [8] and Basis Sharing [9]. We evaluate 1.25× and 1.43× compression for all three scales, and extend the Qwen3-8B sweep to 1.67× and 2×. At every evaluated compression point, DictLLM achieves the lowest WikiText-2 and C4 perplexities and the highest average reasoning accuracy among these compression methods. The results of SVD-LLM [8] and Basis Sharing [9] are reproduced by us.

At 1.43× compression, DictLLM improves average accuracy over Basis Sharing from 36.70% to 39.37% on Qwen3-1.7B, from 45.28% to 50.55% on Qwen3-8B, and from 48.38% to 52.03% on Qwen3-14B. On Qwen3-8B at 2× compression, it reduces WikiText-2 perplexity from 29.61 to 16.58 and increases average accuracy from 34.31% to 41.63% relative to Basis Sharing. These results extend the performance-model size trade-off to a newer model family and multiple parameter scales.

Table A6: Factorization and coefficient sparsity settings for Qwen3 model family. We report the factorization ratio $\rho _ { \mathrm { f a c } }$ and Sparsity p required to achieve a given compression gain for Qwen3-8B, 1.7B, and 14B models.
<table><tr><td>Compression gain</td><td>Factorization ratio  $\rho _ { \mathrm { f a c } }$ </td><td>Qwen3-8B</td><td>Sparsity p Qwen3-1.7B</td><td>Qwen3-14B</td></tr><tr><td>1.25×</td><td>10%</td><td>17.25%</td><td>17.57%</td><td>17.52%</td></tr><tr><td>1.43×</td><td>20%</td><td>19.40%</td><td>19.79%</td><td>19.71%</td></tr><tr><td>1.67×</td><td>30%</td><td>22.17%</td><td></td><td></td></tr><tr><td>2x</td><td>30%</td><td>44.38%</td><td>一</td><td>一</td></tr></table>

We show the factorization ratio and coefficient sparsity to achieve a given compression gain for Qwen3 model families in Table A6

## A4 Componentwise ablation

Table A7: Componentwise ablation of compression stages. ‘F’, $\mathbf { \nabla } ^ { 6 } \mathrm { P } ^ { \prime }$ $ { \mathbf { \hat { R } } }  { \mathbf { \hat { \rho } } }$ , and $\mathbf { \bar { Q } } ^ { \prime }$ respectively denote factorization, Hessian pruning, dictionary refitting, and quantization. A checkmark indicates an active component. We measure language and reasoning performance on LLaMA-7B.
<table><tr><td rowspan="2">Stage</td><td rowspan="2">Comp. gain</td><td rowspan="2">Components F P R Q</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Perplexity↓ Wiki-2</td><td rowspan="2">C4</td><td colspan="8">Zero-shot accuracy ↑</td></tr><tr><td>OB</td><td>WG</td><td>HS</td><td>PQ</td><td>MQ</td><td>AE </td><td>AC</td><td>Avg.</td></tr><tr><td>Original</td><td>1x</td><td></td><td></td><td></td><td></td><td>5.68</td><td>7.34</td><td>0.28</td><td></td><td>0.67 0.56 0.78 0.27</td><td></td><td></td><td>0.67</td><td>0.38</td><td>0.52</td></tr><tr><td>+ Factorization</td><td>1.43×</td><td> $\begin{array} { r l r l } { \surd } & { { } - } & { - } & { { } - } \end{array}$ </td><td></td><td></td><td></td><td>9.25</td><td>22.46</td><td>0.270.630.400.680.24 0.630.300.45</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ Hessian pruning</td><td> $2 \times$ </td><td> $\begin{array}{c}  { / \begin{array} { l l } { \phantom { - } } & {  { / \vphantom { ( } } \sqrt { } } & { - } \end{array}   { \begin{array} { l l } { \phantom { - } } & { - } \end{array} }  } \end{array}$ </td><td></td><td></td><td></td><td>11.53</td><td>33.82</td><td>0.250.62 0.360.640.23 0.570.270.42</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ Dictionary refit</td><td> $2 \times$ </td><td> $\sqrt { \mathrm { ~ \bf ~ \psi ~ } } \sqrt { \mathrm { ~ \bf ~ \psi ~ } } - \mathrm { ~ \bf ~ \psi ~ }$ </td><td></td><td></td><td></td><td>10.59</td><td>29.12</td><td>0.24 0.630.380.650.23 0.61 0.290.43</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ Quantization</td><td> $7 \times$ </td><td> $\sqrt { \mathrm { ~ \bf ~ \psi ~ } } \sqrt { \mathrm { ~ \bf ~ \psi ~ } } \sqrt { \mathrm { ~ \bf ~ \psi ~ } }$ </td><td></td><td></td><td></td><td>11.92</td><td>33.90</td><td>0.240.590.350.630.22 0.54 0.260.40</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table A7 evaluates the successive stages on LLaMA-7B under the same calibration and evaluation protocol. Starting from factorization at $\rho _ { \mathrm { f a c } } = 3 0 \%$ , coefficient-space Hessian pruning at approximately 44% sparsity increases the reported compression gain from 1.43× to 2×, while average reasoning accuracy decreases from 0.45 to 0.42. Dictionary refitting then raises accuracy to 0.43 and reduces WikiText-2 perplexity from 11.53 to 10.59 without changing the dictionary dimensions or coefficient sparsity. C4 perplexity also improves, from 33.82 to 29.12.

Refitting, therefore, provides a quality recovery step without additional parameter storage, although it requires an additional offline update. Subsequent quantization increases the reported compression gain to 7×, with WikiText-2 perplexity of 11.92 and an average accuracy of 0.40. This cumulative comparison exposes the storage–quality trade-off at each stage; it is not a leave-one-out experiment at a fixed final budget.

## A5 Spectral analysis and motivation for cross-layer weight sharing

Motivation for input-scaled cross-layer factorization. We analyze the singular-value spectra of LLaMA-7B projection weights before and after cross-layer concatenation (A1). For each setting, we compare the raw weight spectrum with the input-scaled spectrum computed using the WikiText-2 calibration covariance. The input-scaled spectra accumulate faster, suggesting that fewer components are needed to preserve activation-weighted energy. This supports calibration-aware factorization: while the raw weights may not be strongly low-rank, their action on calibration inputs is concentrated in a smaller set of spectral directions.

While cross-layer SVD has a smaller slope than the single-layer counterpart, it enables the shared dictionary to capture information common across adjacent layers, thereby easing the rank requirement $k = 1 4 3 3  1 9 1 1$ for equal compression rate $\rho _ { \mathrm { f a c } } = 3 0 \%$ . We also observe that the spectrum is more concentrated in the early layers: for example, layer $\ell = 1$ accumulates spectral energy rapidly at low ranks, whereas deeper layers, such as $\ell = 2 4$ , exhibit slower spectral decay.

![](images/def9c2d0954f9f73cdf6d438f3b942d3fb421862872cd120f05d0c3a1613c3b6.jpg)

![](images/a60a7ed113d2daae56c093eaafbf7760752515186e0dec6b77b096a5335b693b.jpg)

(c) Cross-layer  
![](images/fac2009ef77ebe59d433883598c702a2c69880b64e9aba90da6f73e57c4deb6c.jpg)

(d) Input-scaled cross-layer  
![](images/f4218a86b6dc31bf516ac7455fa98effe429590c160617ad9dfd9ea9237ed6b2.jpg)  
Figure A1: Spectral concentration of LLaMA-7B query projection weights. We plot cumulative spectral energy for single-layer SVD (a) with and (b) without input covariance scaling and cross-layer SVD (c) with and (d) without scaling across layers ℓ. Input-scaled spectra rise more rapidly, showing that the energy of input-scaled weights is more concentrated in low-rank components than that of the raw weight spectrum. Vertical dashed lines indicate the selected rank $k \in \{ 1 4 \bar { 3 } 3 , 1 9 1 1 \}$ for the target factorization rate $\rho _ { \mathrm { f a c } } = 3 0 \%$ in single-layer and cross-layer settings.

(a) Cross-layer (K)  
![](images/171cc79e08b2cb83b367eeced6b6eb6d1d7fac2f177802502d6c974712a20488.jpg)

(b) Cross-layer (V)  
![](images/3bada512cd7656c97097aa1a032796d75e5717c41bddd54d0967e12448898a37.jpg)

(c) Single-layer (O)  
![](images/4001c7bd4d68601bda450c290fa7079ed89f0ae06d67c77a77278a8c513ebf8e.jpg)

(d) Cross-layer (Gate)  
![](images/711a530d5070b98dacb7a933d3a23c6c38b114a5a545e80b0662dd2f83753a92.jpg)

(e) Cross-layer (Up)  
![](images/f0d8b13879ee15e385835c90c8441a48759c55d4a1e5a59ee98e4b4f57abd935.jpg)

(f) Single-layer (Down)  
![](images/cc4a997f1f001479df95902643ebd368ac39292eadd05f40a5671ba089c57d60.jpg)  
Figure A2: Spectral concentration across different weight matrices in LLaMA-7B. We plot cumulative spectral energy for the shared projections K, V, gate, and up using cross-layer SVD, and for the private projections O and down using single-layer SVD, following our factorization design. Vertical dashed lines indicate the selected rank k for the target factorization rate $\rho _ { \mathrm { f a c } } = 3 0 \%$ under the corresponding single-layer or cross-layer setting.

Spectral analysis of shared and private projections. We separately analyze the projection families according to our sharing design and report the findings in Fig. A2. For the shared projections K, V, gate, and up, we study the spectrum after cross-layer concatenation, since these matrices share a dictionary across layers. In contrast, O and down are private projections in our model, so we analyze them with single-layer SVD.

![](images/40fbaf17e5d50f5b17aa89c5abe9555be636cde35d2acee03c4931a78e908265.jpg)  
Figure A3: Aggressive layer sharing with LoRA fine-tuning. We increase the dictionary-sharing group size to $n = 4 , 8 , 1 6$ and apply 60% coefficient sparsity while keeping the rank of the shared dictionary fixed, then use LoRA fine-tuning to restore performance.

## A6 Aggressive layer sharing with LoRA fine-tuning.

Fine-tuning allows more agressive layer-sharing. Dictionary sharing reduces parameter count by amortizing a single dictionary across a group of layers, but increasing the sharing group beyond two layers can degrade performance [9, Table 6]. To study a more aggressive compression setting, we also share the output and down projection matrices and increase the sharing group size $\tan { \alpha } = 4 , 8 , 1 6$ We fix the shared dictionary ranks to $k _ { \mathrm { q , k , v , o } } ~ = ~ 2 1 8 4$ $k _ { \mathrm { u p , g a t e } } ~ = ~ 2 7 6 2$ , and $k _ { \mathrm { d o w n } } ~ = ~ 3 7 5 7$ corresponding to the ranks used at $\rho _ { \mathrm { f a c } } = 2 0 \%$ . We then prune the coefficients with sparsity $p = 6 0 \%$ To recover the resulting perplexity degradation, we apply LoRA fine-tuning to all projection matrices using the same hyperparameters as in Table 6. The three settings achieve model-size reductions of 2.9×, 3.5×, and 3.9×, with WikiText-2 perplexities of 10.26, 12.32, and 15.89, respectively.

Table A8: Calibration size for NVFP4 quantization. Post-quantization WikiText-2 perplexity of LLaMA-7B. Using 16 calibration samples provides a strong trade-off between perplexity and runtime.
<table><tr><td>Calibration size</td><td>Perplexity ↓</td></tr><tr><td>16</td><td>9.06</td></tr><tr><td>32</td><td>8.98</td></tr><tr><td>64</td><td>8.90</td></tr><tr><td>128</td><td>8.88</td></tr></table>

Ablation of calibration size on quantization. We apply NVFP4 quantization using subsets of the WikiText-2 training set for calibration. Table A8 shows that 16 calibration sequences provide an efficient accuracy-runtime trade-off. Using 16 sequences achieves a perplexity of 9.06 in slightly over an hour, while increasing the calibration size to 128 takes 8× longer and improves perplexity by only 0.18 points.

## A7 Comparison with additional baselines

Pruning-only baselines at high sparsity. Table A9 compares SparseGPT [14] and Wanda [15] at 80.32% unstructured sparsity with 4-bit DictLLM at a similar reported 5× gain. DictLLM combines factorization at $\rho _ { \mathrm { f a c } } = 2 0 \%$ and approximately 19% coefficient sparsity with quantization. At this aggressive pruning level, SparseGPT and Wanda achieve average accuracies of 29.25% and 29.52%, respectively, whereas DictLLM achieves 46.92%. The language modeling results also favor DictLLM.

Table A9: Comparison with pruning-only methods. SparseGPT and Wanda are evaluated on LLaMA-7B at 80.32% unstructured sparsity. All results are reproduced by us.
<table><tr><td rowspan="2">Comp. gain</td><td rowspan="2">Method</td><td rowspan="2">Perplexity ↓ WikiText-2</td><td rowspan="2">C4</td><td colspan="8">Zero-shot accuracy (%) ↑</td></tr><tr><td>OB</td><td>WG</td><td>HS</td><td>PQ</td><td>MQ</td><td>AE</td><td>AC</td><td>Average</td></tr><tr><td>1x</td><td>Original</td><td>5.68</td><td>7.34</td><td>30.80</td><td>69.38</td><td>56.36</td><td>78.24</td><td>26.57</td><td>74.66</td><td>41.55</td><td>53.94</td></tr><tr><td rowspan="3">5x</td><td>SparseGPT [14]</td><td>86.72</td><td>257.96</td><td>14.00</td><td>48.46</td><td>25.38</td><td>51.41</td><td>18.79</td><td>25.63</td><td>21.08</td><td>29.25</td></tr><tr><td>Wanda [15]</td><td> $2 . 8 \times 1 0 ^ { 3 }$ </td><td> $2 . 8 \times 1 0 ^ { 3 }$ </td><td>11.60</td><td>48.30</td><td>25.78</td><td>53.10</td><td>21.21</td><td>26.35</td><td>20.31</td><td>29.52</td></tr><tr><td>DictLLMFP4</td><td>8.48</td><td>17.04</td><td>29.20</td><td>64.64</td><td>43.67</td><td>70.18</td><td>23.99</td><td>64.02</td><td>32.76</td><td>46.92</td></tr></table>

This comparison illustrates the benefit of distributing compression across factorization, sparsity, and precision, rather than relying solely on pruning to meet the budget.

Table A10: Comparison with quantization-only baselines. We compare methods at 7× compression gain on LLaMA-7B. GPTQ [19] and AWQ [34] use two-bit weights, whereas DictLLM combines factorization and coefficient sparsity with four-bit quantization. All results are reproduced by us.
<table><tr><td rowspan="2">Comp. gain</td><td rowspan="2">Method</td><td colspan="2">Perplexity ↓</td><td colspan="8">Zero-shot accuracy (%) ↑</td></tr><tr><td>WikiText-2</td><td>C4</td><td>OB</td><td>WG</td><td>HS</td><td>PQ</td><td>MQ</td><td>AE</td><td>AC</td><td>Average</td></tr><tr><td>1x</td><td>Original</td><td>5.68</td><td>7.34</td><td>30.80</td><td>69.38</td><td>56.36</td><td>78.24</td><td>26.57</td><td>74.66</td><td>41.55</td><td>53.94</td></tr><tr><td rowspan="3">7x</td><td>GPTQ [19] (2-bit)</td><td>17.59</td><td>36.56</td><td>13.80</td><td>48.38</td><td>30.47</td><td>57.13</td><td>20.84</td><td>32.03</td><td>21.33</td><td>32.00</td></tr><tr><td>AWQ [34] (2-bit)</td><td> $2 . 6 \times 1 0 ^ { 5 }$ </td><td> $2 . 9 \times 1 0 ^ { 5 }$ </td><td>13.60</td><td>49.33</td><td>25.40</td><td>52.83</td><td>21.57</td><td>25.38</td><td>22.35</td><td>30.07</td></tr><tr><td>DictLLMFP4</td><td>11.92</td><td>33.90</td><td>23.60</td><td>59.19</td><td>35.32</td><td>63.17</td><td>22.21</td><td>53.62</td><td>25.60</td><td>40.39</td></tr></table>

Quantization-only baselines at a larger compression gain. Table A10 examines the more aggressive reported 7× budget. NVFP4 supplies approximately a 3.56× reduction, including its scale overhead, which compounds with the approximately 2× gain from factorization and pruning at $\rho _ { \mathrm { f a c } } ~ = ~ 3 0 \%$ and approximately 44% coefficient sparsity. We therefore compare against 2-bit $\mathrm { G P T Q } \left[ 1 9 \right]$ and AWQ [34], rather than 4-bit quantization alone, to approach this larger compression gain.

DictLLM achieves an average accuracy of 40.39%, compared with 32.00% for GPTQ [19] and 30.07% for AWQ [34]. It also has lower WikiText-2 perplexity than either quantization-only baseline.

Table A11: Comparison with low rank methods including SVD-LLM V2. We measure WikiText-2 perplexity at 20% compression.
<table><tr><td>Method</td><td>LLaMA-7B</td><td>OPT-6.7B</td></tr><tr><td>ASVD [11]</td><td>11.14</td><td>82.0</td></tr><tr><td>SVD-LLM [8]</td><td>7.94</td><td>16.04</td></tr><tr><td>SVD-LLM V2 [56]</td><td>7.12</td><td>13.46</td></tr><tr><td>Basis Sharing [9]</td><td>7.74</td><td>11.79</td></tr><tr><td>DictLLM (Ours)</td><td>6.89</td><td>11.36</td></tr></table>

Comparison against SVD-LLM V2. We supplement our comparisons with the results of SVD-LLM V2 [56]. Table A11 reports WikiText-2 perplexities at 20% compression on LLaMA-7B and OPT-6.7B. DictLLM obtains perplexities of 6.89 and 11.36, compared with 7.12 and 13.46 for SVD-LLM V2.

Basis Sharing followed by quantization. We apply 4-bit GPTQ [19] and AWQ [34] to a Basis Sharing model factorized at $\rho _ { \mathrm { f a c } } ~ = ~ 3 0 \%$ . Table A12 compares these combinations with 4-bit DictLLM at the same reported 5× compression gain. DictLLM uses $\rho _ { \mathrm { f a c } } = 2 0 \%$ and approximately 19% coefficient sparsity before quantization. It achieves WikiText-2 perplexity of 8.48, compared with 13.99 for Basis Sharing + GPTQ and 11.38 for Basis Sharing + AWQ. Its average accuracy is 46.92%, compared with 37.80% and 41.22%, respectively. These results show that quantizing

Table A12: Effect of compounding quantization on Basis Sharing. ‘BS’ denotes Basis Sharing [9]. We measure language and reasoning performances on LLaMA-7B. All results are reproduced by us.
<table><tr><td rowspan="2">Comp. gain</td><td rowspan="2">Method</td><td rowspan="2">Perplexity ↓ Wiki-2 C4</td><td rowspan="2">OB</td><td colspan="8">Zero-shot accuracy (%) ↑</td></tr><tr><td></td><td>WG</td><td>HS</td><td>PQ</td><td>MQ</td><td>AE</td><td>AC</td><td>Average</td></tr><tr><td>1x</td><td>Original</td><td>5.68</td><td>7.34</td><td>30.80</td><td>69.38</td><td>56.36</td><td>78.24</td><td>26.57</td><td>74.66</td><td>41.55</td><td>53.94</td></tr><tr><td rowspan="3">5x</td><td>BS [9] + GPTQ [19] (4-bit)</td><td>13.99</td><td>34.01</td><td>24.00</td><td>56.83</td><td>34.72</td><td>57.89</td><td>20.74</td><td>46.93</td><td>23.46</td><td>37.80</td></tr><tr><td>BS [9] + AWQ [34] (4-bit)</td><td>11.38</td><td>30.95</td><td>24.40</td><td>59.91</td><td>36.09</td><td>64.09</td><td>21.94</td><td>55.35</td><td>26.79</td><td>41.22</td></tr><tr><td>DictLLMFP4</td><td>8.48</td><td>17.04</td><td>29.20</td><td>64.64</td><td>43.67</td><td>70.18</td><td>23.99</td><td>64.02</td><td>32.76</td><td>46.92</td></tr></table>

a factorized model alone does not match the quality of the DictLLM configuration at this reported budget.

Table A13: Effect of compounding pruning and quantization on Basis Sharing. $\mathbf { \bar { \rho } } _ { \mathrm { B S } } ,$ denotes Basis Sharing [9]. Magnitude pruning acts on the coefficient matrix. We measure language and reasoning performance on LLaMA-7B. All results are reproduced by us.
<table><tr><td rowspan="2">Comp. gain</td><td rowspan="2">Method</td><td colspan="3">Perplexity ↓</td><td colspan="7">Zero-shot accuracy (%) ↑</td></tr><tr><td>Wiki-2</td><td>C4</td><td>OB</td><td>WG</td><td>HS</td><td>PQ</td><td>MQ</td><td>AE</td><td>AC</td><td>Average</td></tr><tr><td>1x</td><td>Original</td><td>5.68</td><td>7.34</td><td>30.80</td><td>69.38</td><td>56.36</td><td>78.24</td><td>26.57</td><td>74.66</td><td>41.55</td><td>53.94</td></tr><tr><td>1.43×</td><td>BS [9] + magnitude DictLLM</td><td>8.53 7.79</td><td>17.28 15.20</td><td>29.60 29.00</td><td>65.35 66.38</td><td>43.67 45.67</td><td>70.40 71.27</td><td>23.28 24.25</td><td>65.32 65.53</td><td>35.07 35.07</td><td>47.53 48.17</td></tr><tr><td>5x</td><td>BS [9] + magnitude + AWQ [34] (4-bit) DictLLMFP4</td><td>10.39 8.48</td><td>23.50 17.04</td><td>23.80 29.20</td><td>62.98 64.64</td><td>38.64 43.67</td><td>66.05 70.18</td><td>22.28 23.99</td><td>57.32 64.02</td><td>25.51 32.76</td><td>42.37 46.92</td></tr></table>

Basis Sharing followed by magnitude pruning. To compare coefficient pruning strategies, we apply 19.4% coefficient sparsity to a Basis Sharing model with $\rho _ { \mathrm { f a c } } = 2 0 \% ,$ , and optionally apply 4-bit AWQ afterward. As shown in Table A13, DictLLM improves WikiText-2 perplexity from 8.53 to 7.79 and average accuracy from 47.53% to 48.17% before quantization. With quantization, the corresponding values improve from 10.39 to 8.48 and from 42.37% to 46.92%. The unquantized comparison supports the benefit of Hessian-based coefficient pruning together with dictionary refitting over magnitude-only coefficient pruning. The quantized comparison also differs in its quantization method; therefore, it measures the combined compression pipeline rather than isolating pruning alone.

## A8 End-to-end decoding throughput and memory

Table A14: End-to-end decoding on LLaMA-7B. DictLLM uses MACKO for sparse coefficients. The measurement excludes the prompt prefill stage.
<table><tr><td>Configuration</td><td>Weight memory [GB]</td><td>Peak memory [GB]</td><td>Speed [tokens/s]</td></tr><tr><td>Original</td><td>13.48</td><td>13.66</td><td>102.35</td></tr><tr><td>Basis Sharing [9] (dense D, dense C)</td><td>9.59</td><td>9.83</td><td>113.59</td></tr><tr><td>DictLLM (dense D, sparse C)</td><td>7.82</td><td>8.02</td><td>126.81</td></tr></table>

Benchmark setup. We use MACKO [57] to perform sparse-coefficient operations and to measure whole-model decoding, rather than isolated matrix multiplication. All configurations use FP16 precision on an NVIDIA RTX PRO 6000 Blackwell Max-Q GPU with a batch size of 1. Each run uses a 128-token prompt and 100 single-token autoregressive decoding steps. We report results over 10 measured runs following three warm-up runs. Prompt prefill is excluded from the timed interval.

Memory saving and speed-up. Table A14 compares the dense LLaMA-7B model, Basis Sharing with $\rho _ { \mathrm { f a c } } = 3 0 \%$ , and DictLLM with the same factorization rate and approximately 44% coefficient sparsity. DictLLM reaches 126.81 tokens/s, a 1.24× increase over the original model and a 1.12× increase over the factorization-only model. Weight memory decreases from 13.48 to 7.82 GB (1.72× reduction), and peak memory decreases from 13.66 to 8.02 GB (1.70× reduction). These results demonstrate a realized decoding benefit for the tested hardware and sparse backend, beyond the representation-level size estimates.

## A9 Assets and compute resources

## A9.1 Assets

License for WikiText-2. The WikiText-2 [44] dataset is published under the CC-BY-SA-3.0 License, and all data<sup>1</sup> is released under the GNU Free Documentation License family.

License for PTB. The PTB [45] dataset<sup>2</sup> is released without an explicit license; however, the data is released under LDC User Agreement for Non-Members.

License for C4. The C4 [46] dataset<sup>3</sup> is released under the ODC-BY License.

License for LM Harness. The LM Harness framework [53] with code<sup>4</sup> is released under MIT License.

License for LLaMA-7B. The LLaMA-7B model [41] with a variant checkpoint<sup>5</sup> is licensed under License Non-commercial bespoke license.

License for LLaMA 2-7B. The LLaMA 2-7B model [41] with checkpoint<sup>6</sup> is licensed under Llama 2 Community License Agreement.

License for Mistral-7B. The Mistral-7B model [43] with checkpoint<sup>7</sup> is licensed under Apache License 2.0.

License for OPT-6.7B. The OPT-6.7B model [42] with checkpoint<sup>8</sup> is licensed under OPT-175B License Agreement.

License for Qwen3. The Qwen 3 models [58] with checkpoints for Qwen3-1.7B<sup>9</sup>, -8B<sup>10</sup>, and -14B<sup>11</sup> are licensed under the Apache License 2.0.

## A9.2 Computation resources.

The experiments are conducted on a Red Hat Enterprise Linux 8 server equipped with dual NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, each equipped with 96GB VRAM, an AMD Ryzen Threadripper PRO 5975WX CPU @ 3.6-4.5GHz, and 256GB of RAM.

## A10 Discussions

Limitations. A limitation of DictLLM is that coefficient pruning produces unstructured sparsity. While this reduces parameter storage and can be exploited with sparse kernels, its hardware efficiency depends on support for irregular sparse access patterns. Exploring structured coefficient sparsity, such as block sparsity, is a promising direction for improving hardware efficiency and co-design.

Broader impact. DictLLM aims to reduce the memory footprint of large language models, making capable models more accessible on resource-constrained hardware and lowering the energy cost of inference. This may broaden access to LLM deployment beyond large datacenters and enable more private, local, and low-latency applications. At the same time, easier deployment of compressed LLMs can also lower the barrier to misuse, including spam generation, misinformation, or unauthorized automation. Since DictLLM preserves the behavior of the underlying pretrained model as closely as possible, it may also retain that model’s biases, safety risks, and failure modes.