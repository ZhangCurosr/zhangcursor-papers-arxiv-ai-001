# MOAI: MODULE-OPTIMIZING ARCHITECTURE FOR NON-INTERACTIVE SECURE TRANSFORMER INFER-ENCE

Linru Zhang<sup>1</sup>, Xiangning Wang<sup>1</sup>, Jun Jie Sim<sup>2</sup>, Zhicong Huang<sup>3</sup>, Jiahao Zhong<sup>1</sup>, Huaxiong Wang<sup>1</sup>, Pu Duan<sup>2</sup>, Kwok-Yan Lam<sup>1</sup>

<sup>1</sup>Nanyang Technological University, <sup>2</sup>Ant International, <sup>3</sup>Ant Group

## ABSTRACT

Privacy concerns have been raised in Large Language Models (LLM) inference when models are deployed in Cloud Service Providers (CSP). Homomorphic encryption (HE) offers a promising solution by enabling secure inference directly over encrypted inputs. However, the high computational overhead of HE remains a major bottleneck. To address this challenge, we propose MOAI, an efficient HE-based, non-interactive framework for secure transformer inference. MOAI gains significant efficiency improvement from: (1) a novel evaluation flow that combines column and diagonal packing with consistent strategies across all layers, eliminating expensive format conversions. (2) rotation-free algorithms for Softmax and LayerNorm that significantly reduce the number of costly HE rotations, removing 2448 HE rotations in BERT-base inference. (3) Column packing removes rotations in plaintext–ciphertext matrix multiplications and interleaved batching further reduces the rotations in ciphertext–ciphertext matrix multiplications. MOAI uses at least 1.7x fewer HE rotations compared to the state-of-the-art works across all matrix multiplications of BERT-base. As a result, We achieve a 52.8% reduction in evaluation time compared to the state-of-the-art HE-based non-interactive secure transformer inference, THOR (Moon et al., CCS’25). We then apply MOAI on the Powerformer’s framework and achieve a 55.7% reduction in evaluation time compared to Powerformer (Park et al., ACL’25), which approximates Softmax and LayerNorm with simpler functions in transformer and proposes HE-based non-interactive transformer inference. We report an amortized time of 2.36 minutes per input on a single GPU environment. We show the extendibility by applying MOAI in LLaMA-3-8B. Our implementation is publicly available as open source.

## 1 INTRODUCTION

The Transformer architecture Vaswani et al. (2017) powers many NLP tasks but requires heavy computation and memory, motivating deployment on cloud service providers (CSP). There are growing concerns about potential privacy leakage when clients upload sensitive information to CSP.

Fully homomorphic encryption (FHE) Gentry (2009) enables direct, non-interactive computation over encrypted data, making it a promising primitive for protecting clients’ data privacy. In FHE-based AI-as-a-Service works Bourse et al. (2018); Gilad-Bachrach et al. (2016); Mishra et al. (2020); Lam et al. (2024); Zhang et al. (2024b), a client encrypts its input data and uploads the ciphertexts to a cloud service provider (CSP), which performs model evaluation without any decryptions. The CSP then returns the encrypted inference results to the client, ensuring that it learns neither the client’s inputs nor the inference outputs. However, the high computational overhead of homomorphic evaluation significantly limits the practicality of FHE-based inference, especially in scenarios that a large number of inputs has to be processed such as document classification, a representative application of encoder-only models like BERT Devlin et al. (2019).

Related work. Some works Chen et al. (2022); Pang et al. (2024); Lu et al. (2023); Dong et al. (2023); Zeng et al. (2025) targeting interactive privacy-preserving transformer inference, usually on

BERT, relied on a combination of FHE and MPC that splits the user’s data into shares and distributes a share to the server.

The first non-interactive solution NEXUS, based purely on FHE, was proposed by Zhang et al. (2024a) (NDSS ’25). They proposed several FHE-friendly matrix multiplications and approximations for activation functions. However, NEXUS does not provide an end-to-end inference solution. NEXUS only reports performance as an aggregation of microbenchmarks. In addition, due to inconsistent packing formats, its algorithms cannot be directly composed. As also observed by Lim et al. (2025); Moon et al. (2024), the conversions between these formats during secure inference are not explicitly explained and the time cost of format conversions is not included in the reported performance. LEAF Zhang et al. (2025) leverages coefficient-encoding matrix multiplication and functional bootstrapping for inference, but focuses only on the feedforward block rather than providing an end-to-end solution. The state-of-the-art HE-based non-interactive solution THOR Moon et al. (2024) (CCS’25) achieved end-to-end secure transformer inference. THOR provided inference on 128 tokens with the BERT-base model and reported 10 minutes per input on a single GPU. Format conversions are required in THOR to complete ciphertext-ciphertext matrix multiplications. These solutions do not modify the transformer components such as Softmax and LayerNorm, and thus avoid the need for re-training or fine-tuning.

Another line of research focuses on adapting transformers to FHE-friendly functions. Zimerman et al. (2024b) replaced non-polynomial functions (e.g., Softmax) with polynomial alternatives. Power-Softmax Zimerman et al. (2024a) replaced the exponential function in Softmax with a power function and applied conventional fine-tuning. The state-of-the-art solution in this direction, Powerformer Park et al. (2024b) (ACL’25) replaced Softmax and LayerNorm in the BERT-base model with power and linear functions, achieving 5.74 minutes per input on a single GPU. Although this line of work offers improved efficiency, it requires re-training through knowledge distillation, which limits its generalizability to other transformer architectures.

Our contributions. We focus on designing a plug-and-play FHE framework that can be seamlessly integrated into existing transformer architectures. Similar to NEXUS and THOR, we do not modify any transformer components to avoid the need for re-training through knowledge distillation.

In this paper, we present MOAI: Module-Optimizing Architecture for non-interactive secure transformer Inference. MOAI is a FHE-based. We apply MOAI on the BERT-base model and report an amortized time of 2.36 minutes per input on a single GPU environment. We also extend our methods to decoder-only transformers, such as LLaMA-3-8B. MOAI outperforms the state-of-the-art work THOR, achieving 52.8% reduction in evaluation time in the same environment. We also show the extendibility of MOAI by applying MOAI’s evaluation flow to converted FHE-friendly transformer architectures. MOAI outperforms the state-of-the-art work Powerformer, achieving 55.7% reduction in evaluation time when applying MOAI to Powerforer’s FHE-friendly modifications. Our techniques can be summarized as follows:

(1) Consistent matrix packing strategies to avoid extra format conversion. The input ciphertexts and the evaluation of $Q , K , { \bar { V } }$ are in column packing. Then the output of our FHE evaluation of $Q K ^ { T }$ is in diagonal packing, which matches the input format of our FHE evaluation of Softmax. After evaluating Softmax in diagonal packing, a ciphertext - ciphertext matrix multiplication (for softm $\arg { K \bar { \Psi } / \sqrt { d ^ { \prime } } } ) V )$ takes input of one matrix encrypted in diagonal packing $( \mathrm { E n c } ( \mathrm { s o f t m a x } ( Q K ^ { \top } / \sqrt { d ^ { \prime } } ) ) )$ and one matrix encrypted in column packing (Enc(V)) and the output matrix is encrypted in column packing. Resting parts such as feedforward layers are all in column packing, and are ready to be passed to the next transformer layer without format conversion. The details of the evaluation flow is in Figure 2. Our evaluation flow ensures the consistent packing formats across all layers to avoid expensive format conversion between layers. We also remove the need of performing encrypted matrix transpose separately. In THOR, additional format conversions and transpose are necessary.

(2) Rotation-free Softmax and LayerNorm evaluations. Evaluating Softmax and LayerNorm in NEXUS and THOR includes a set of FHE rotations to compute a summation of data packed in a ciphertext. HE rotation is known as a computation intensive operation which affects the efficiency (e.g., Halevi & Shoup (2018)). We propose rotation-free Softmax and LayerNorm evaluation algorithms which eliminate rotation by putting the inputs to the summation in the same slot position of different ciphertexts. These algorithms save 2448 HE rotations in BERT-base inference.

(3) Minimizing HE rotations in matrix multiplication. Column packing is adopted to remove HE rotations in plaintext-ciphertext matrix multiplication. Interleaved batching is applied to further reduces the rotations in ciphertext–ciphertext matrix multiplications. MOAI achieves at least 1.7x fewer HE rotations than the state-of-the-art works across all matrix multiplications of a transformer.

## 2 PRELIMINARIES

We use [n] to represent the set $\{ 0 , 1 , . . . , n - 1 \}$ . We use bold lowercase letters for row vector $\mathbf { v } \in \mathbb { R } ^ { n }$ where $\mathbf { v } = [ v _ { 0 } , v _ { 1 } , . . . , v _ { n - 1 } ]$ . We use $[ v _ { i } ] _ { i \in [ n ] }$ to represent the vector v. Column vectors are represented by $\mathbf { v } ^ { \mathsf { T } }$ . We define the scalar product for $c \in \mathbb { R }$ and $\mathbf { v } \in \mathbb { R } ^ { n }$ , let $c \mathbf { v } = [ c v _ { i } ] _ { i \in [ n ] }$ We also define addition and Hadamard product (entry-wise multiplication) between two vectors with dimension n as follows: $\mathbf { v } + \mathbf { u } : = [ v _ { i } + u _ { i } ] _ { i \in [ n ] } , \mathbf { v } \otimes \mathbf { u } : = [ v _ { i } u _ { i } ] _ { i \in [ n ] }$ . We use ${ \mathrm { R o t } } _ { k } ( \mathbf { v } )$ , for $0 \leq k < n$ , to denote the vector $\left[ v _ { k } , . . . , v _ { n - 1 } , v _ { 0 } , . . . v _ { k - 1 } \right]$ , a left rotation of v. For $- n < k \leq 0$ ${ \mathrm { R o t } } _ { k } ( \mathbf { v } )$ denotes a right rotation.

We use uppercase letters to denote a matrix $A \in \mathbb { R } ^ { m \times d }$ where $A = ( a _ { i j } ) _ { i \in [ m ] , j \in [ d ] }$ . We use $\mathbf { a _ { j } } ^ { \intercal }$ to represent the $j \mathrm { - t h }$ column of A, and we can write $A = ( \mathbf { a _ { 0 } } ^ { \intercal } \parallel \mathbf { a _ { 1 } } \mathsf { \check { \tau } } \parallel \cdot \cdot \cdot \parallel \mathbf { a } _ { d - 1 } ^ { \intercal } )$ . Further, if $A$ is a square matrix with size $m \times m$ , define $\operatorname { D i a g } _ { i } ( A )$ as A’s i-th (upper) diagonal for $i \in [ m ]$ $\operatorname { D i a g } _ { i } ( A ) : = [ a _ { 0 , i } , a _ { 1 , i + 1 } , . . . , a _ { m - i - 1 , m - 1 } , a _ { m - i , 0 } , . . . , a _ { m - 1 , i - 1 } ]$

## 2.1 FULLY HOMOMORPHIC ENCRYPTION - CKKS SCHEME

Fully Homomorphic Encryption (FHE) enables computation on encrypted data. The CKKS scheme Cheon et al. (2017) supports approximate arithmetic over complex vectors. We briefly describe CKKS and refer to Appendix I and Cheon et al. (2017) for details.

Let N be the degree of a polynomial ring, whose value is a power of 2. Q be a product of distinct primes: $\begin{array} { r } { Q = \prod _ { 0 } ^ { L } q _ { i } . } \end{array}$ , where L is called the level of the ciphertext. A fresh ciphertext starts from level $\bar { L }$ and drops to $L - 1$ after a homomorphic multiplication. When remaining level is $0 ,$ decryption fails. The bootstrapping process Cheon et al. (2018) can refresh the ciphertext and restore the remaining level, enabling continued evaluations. The CKKS scheme supports SIMD-style computation by encoding $N / 2$ messages into a single ciphertext, enabling parallel homomorphic operations across all slots. Let $\mathbf { v } , \mathbf { v } ^ { \prime }$ be two vectors in $\mathbb { C } ^ { N / 2 }$ and $\mathbf { c t } = \operatorname { E n c } ( \mathbf { v } )$ and $\mathbf { c } _ { } ^ { \prime } = \operatorname { E n c } ( \mathbf { v } ^ { \prime } )$ . The operations supported by the CKKS scheme are as follows:

$\mathrm { A d d } ( \mathrm { c t , c t ^ { \prime } } )$ . Return a ciphertext that encrypts the vector sum $\mathbf { v } + \mathbf { v } ^ { \prime } .$

$\mathrm { M u l t } ( \mathrm { c t , c t ^ { \prime } } )$ . Return a ciphertext that encrypts the Hadamard product $\mathbf { v } \otimes \mathbf { v } ^ { \prime } .$

$\operatorname { P l a i n M u l t } ( \operatorname { c t } , \mathbf { v } ^ { \prime } )$ . Return a ciphertext that encrypts the Hadamard product $\mathbf { v } \otimes \mathbf { v } ^ { \prime }$

$\mathrm { R o t } _ { k } ^ { \mathrm { H E } } ( \mathrm { c t } )$ . Return a ciphertext that encrypts $\mathrm { R o t } _ { k } ( { \bf v } ) = [ v _ { k } , . . . , v _ { N / 2 - 1 } , v _ { 0 } , . . . v _ { k - 1 } ] ,$

Many previous works state that HE rotation is one of the slowest operations, and they optimize schemes by reducing the number of HE rotation (e.g., Halevi & Shoup (2018); Lu et al. (2021)). Experimental results in (Yang et al. (2024); Agulló-Domingo et al. (2025)) also indicate that HE rotation performs considerably slower, especially as the ciphertext modulus increases.

## 2.2 TRANSFORMERS AND FHE-BASED SECURE TRANSFORMER INFERENCE

The transformer architecture Vaswani et al. (2017) follows an encoder–decoder design. We focus on the encoder (e.g., BERT), commonly used as a benchmark. Each encoder has $\breve { L }$ layers with multi-head self-attention, normalization, and feed-forward blocks. Core computations include matrix multiplication, Softmax, LayerNorm, and GELU. Further details are in Appendix B.

In FHE-based secure transformer inference, the model will take encrypted tokens as input and output encrypted results. The model is evaluated over encrypted data homomorphically, where the FHE evaluations include plaintext-ciphertext matrix multiplication (computing $Q , { \dot { K } } , V$ and feedforward layer), ciphertext-ciphertext matrix multiplication (computing $\overline { { Q } } \bar { K ^ { \dagger } }$ and softmax(·)V), and non-linear evaluations such as Softmax, LayerNorm and GELU.

## 3 MATRIX PACKING, MULTIPLICATIONS AND INTERLEAVED BATCHING

## 3.1 MATRIX PACKING AND MULTIPLICATIONS

The SIMD capability of CKKS scheme allows multiple data to be packed into a single ciphertext, which offers an array-like paradigm to encode matrices for encryption. There are mainly three ways to encode an encrypted matrix in Halevi & Shoup (2014), namely row, column, diagonal packing. The main difficulty in designing an encrypted matrix multiplication pipeline is that the encrypted matrix must be able to fit in and switch between various forms (e.g., the transpose A<sup>⊺</sup>). To this end, we propose to use both the column packing and diagonal packing to realize seamless transition for encrypted matrix multiplication.

Definition 3.1 (Column Packing). Given a matrix $\boldsymbol { X } \in \mathbb { R } ^ { m \times d }$ , the column packing ofX is a set ofd CKKS ciphertexts, denoted as $\operatorname { \mathrm { { E n c } } } _ { c o l } ( X ) : = \left\{ \operatorname { E n c } ( \mathbf { x } _ { 0 } ^ { \intercal } ) , \operatorname { \mathrm { { E n c } } } ( \mathbf { x } _ { 1 } ^ { \intercal } ) , . . . , \operatorname { \mathrm { { E n c } } } ( \mathbf { x } _ { d - 1 } ^ { \intercal } ) \right\}$ , where $\mathbf { x _ { j } } ^ { \mathsf { T } }$ is the j-th column ofX. We say that the ciphertext ofX is in column packing.

Definition 3.2 (Diagonal Packing). Given a square matrix $\begin{array} { r l r } { { \cal X } } & { { } \in } & { \mathbb { R } ^ { m \times m } , } \end{array}$ , the (upper) diagonal packing of X are m CKKS ciphertexts, denoted as $\begin{array} { r l } { \operatorname { E n c } _ { d i a g } ( X ) } & { { } : = } \end{array}$ $\{ \operatorname { E n c } ( \operatorname { D i a g } _ { 0 } ( X ) ) , \operatorname { E n c } ( \operatorname { D i a g } _ { 1 } ( X ) ) , . . . , \operatorname { E n c } ( \operatorname { D i a g } _ { m - 1 } ( X ) ) \}$ , where $\operatorname { D i a g } _ { j } ( X )$ is the j-th (upper) diagonal of X. We use $\operatorname { D i a g } _ { j } ( X )$ as a row vector and say ciphertext of X is in diagonal packing.

Encrypted matrix-vector multiplication for both packing methods were proposed in Halevi & Shoup (2014). We give a brief description of the algorithms for encrypted matrix-vector multiplication for both packing methods. Let $C$ be a square matrix of dimensions $m \times m$ and v be a vector of length m.

For column packing, denote C with its m columns as $\{ \operatorname { C o l } _ { 0 } ( C ) , \operatorname { C o l } _ { 1 } ( C ) , . . . , \operatorname { C o l } _ { m - 1 } ( C ) \}$ . Halevi & Shoup (2014) first proposed an auxiliary algorithm Replicate that takes the vector v and return m-many vectors $\mathbf { v _ { i } }$ , each encrypting $v _ { i }$ , that is $\{ \mathbf { v _ { i } } = [ v _ { i } , v _ { i } , \ldots , v _ { i } ] \} _ { i \in [ m ] }$ . The matrixvector product is computed as $\begin{array} { r } { C v = \sum \mathrm { C o l } _ { i } ( C ) \odot { \bf v _ { i } } } \end{array}$ . We can extend the matrix-vector product to matrix-matrix product by iterating over all columns of the right hand side matrix.

For diagonal packing, denote C with its m diagonals as $\{ \operatorname { D i a g } _ { 0 } ( C ) , \operatorname { D i a g } _ { 1 } ( C ) , . . . , \operatorname { D i a g } _ { m - 1 } ( C ) \}$ The matrix-vector product is computed as $\begin{array} { r } { C v = \sum \mathrm { D i a g } _ { i } ( \dot { C } ) \odot \bar { \mathrm { R o t } } ( \mathbf v , i ) } \end{array}$ . A subsequent improvement in Halevi & Shoup (2021) reduced the number of rotations needed to $O ( { \sqrt { m } } )$ by using the baby-step giant-step method (see Appendix C.1 for details).

We describe our HE matrix multiplications in both column and diagonal packing:

Case I: Calculating Enc(Q), Enc $( K ) , \operatorname { E n c } ( V ) \in \mathbb { R } ^ { m \times d ^ { \prime } }$ with $\operatorname { E n c } ( X ) W _ { i } + \mathbf { 1 } ^ { \intercal } \mathbf { b } _ { i } { \mathrm { f o r } } i \in \{ Q , K , V \}$ We encrypt $X$ in column packing while the weights and biases are not encrypted. We note that since the weights matrices are not encrypted, the matrix product can be interpreted as concatenations of linear combinations of X. $\begin{array} { r } { X W \stackrel {  } { = } [ \sum _ { i \in [ d ^ { \prime } ] } w _ { i , 0 } \mathbf { x } _ { 0 } ^ { \top } \stackrel {  } { = } | | \sum _ { i \in [ d ^ { \prime } ] } w _ { i , 1 } \mathbf { x } _ { 1 } ^ { \top } | | \stackrel {  } { \cdots } | | \sum _ { i \in [ d ^ { \prime } ] } w _ { i , d ^ { \prime } - 1 } \mathbf { x } _ { d ^ { \prime } - 1 } ^ { \top } ] } \end{array}$ The detailed algorithm is given as Algorithm 2 in Appendix C.2.

Case II: Calculating Enc(QK<sup>⊺</sup>). We remind the reader here that $Q$ and K are encrypted in column packing as per the result of Case I. We then apply the following lemma that allows us to compute the matrix product $Q K ^ { \mathsf { T } }$ where both Q and K are encrypted in column packing, without transposing $K .$

Lemma 3.1. Suppose Q, $K \in \mathbb { R } ^ { m \times d ^ { \prime } }$ , so $Q K ^ { \mathsf { T } }$ is square matrix with size m × m. Then $f o r j \in [ m ]$ $\begin{array} { r } { \mathrm { D i a g } _ { j } ( Q K ^ { \intercal } ) = \sum _ { i = 0 } ^ { d ^ { \prime } - 1 } \mathbf { q } _ { i } \otimes R o t _ { j } ( \mathbf { k } _ { i } ) } \end{array}$ , where $\mathbf { q } _ { i }$ and k<sub>i</sub> are the i-column $o f Q$ and K respectively.

The result is a matrix encrypted in diagonal packing. The detailed algorithm is given as Algorithm 3 in Appendix C.2.

Case III: Calculating $\mathrm { E n c ( s o f t m a x } ( Q K ^ { \tau } / \sqrt { d ^ { \prime } } ) V )$ . The encrypted matrix V is encrypted in column packing, as per Case I. For now, we briefly mention that applying the Softmax function on a diagonally packed matrix will return a diagonally packed matrix. Thus, softmax $( Q K ^ { \mathsf { T } } / { \sqrt { d ^ { \prime } } } )$ is encrypted in diagonal packing. For this matrix product, we can apply the matrix-vector product for diagonally packed matrices described earlier Halevi & Shoup (2014; 2021) over all encrypted columns of $\dot { V }$ The detailed algorithm is given as Algorithm 4 in Appendix C.2.

## 3.2 INTERLEAVED BATCHING

In column packing (Definition 3.1) and diagonal packing (Definition 3.2), each ciphertext encrypts a vector of dimension m, i.e., the sequence length of input. Meanwhile, the number of slots $N / 2$ in the CKKS scheme is usually $2 ^ { 1 5 }$ . So batching multiple vectors into one ciphertext will reduce the amortized cost. Interleaved batching, previously discussed in Aharoni et al. (2023); Adir et al. (2024) for evaluating convolutions in neural networks, is adopted in MOAI to further reduce the number of HE rotations in matrix multiplications.

Assume $m | ( N / 2 )$ . Let $N / ( 2 m )$ vectors $\{ \mathbf { x } ^ { ( r ) } \in \mathbb { R } ^ { m } \} _ { r \in [ N / ( 2 m ) ] }$ be:

$$
\mathbf { x } ^ { ( 0 ) } = [ x _ { 0 } ^ { ( 0 ) } , x _ { 1 } ^ { ( 0 ) } , . . . , x _ { m - 1 } ^ { ( 0 ) } ] , . . . , \mathbf { x } ^ { ( N / ( 2 m ) - 1 ) } = [ x _ { 0 } ^ { ( N / ( 2 m ) - 1 ) } , x _ { 1 } ^ { ( N / ( 2 m ) - 1 ) } , . . . , x _ { m - 1 } ^ { ( N / ( 2 m ) - 1 ) } ] ,
$$

interleaved batching is defined by packing the the first entry of each vector from $\{ \mathbf { x } ^ { ( r ) } \in$ $\mathbb { R } ^ { m } \} _ { r \in [ N / ( 2 m ) ] }$ , then the second entry, and at last the last entry. We use $\tilde { \mathbf { x } } \in \mathbb { R } ^ { N / 2 }$ to represent the interleaved batching of $N / ( 2 m )$ vectors in $\mathbb { R } ^ { m }$

$$
\tilde { \mathbf { x } } : = [ x _ { 0 } ^ { ( 0 ) } , x _ { 0 } ^ { ( 1 ) } , x _ { 0 } ^ { ( 2 ) } , . . . , x _ { 0 } ^ { ( N / ( 2 m ) - 1 ) } , . . . x _ { m - 1 } ^ { ( 0 ) } , x _ { m - 1 } ^ { ( 1 ) } , x _ { m - 1 } ^ { ( 2 ) } , . . . , x _ { m - 1 } ^ { ( N / ( 2 m ) - 1 ) } ] .\tag{1}
$$

We use Lemma 3.2 to address the its compatibility with rotation and include its proof in Appendix D.2. The naive batching method in existing works uses two HE rotations to do one “internal rotation”, which only shifts slots within each vector rather than across all slots in a ciphertext. MOAI achieves the same functionality by only one HE rotation, thereby reducing half of HE rotations.

Lemma 3.2. $L e t \tilde { \mathbf { x } } \in \mathbb { R } ^ { N / 2 }$ be the interleaved batching vector of $\{ \mathbf { x } ^ { ( r ) } \in \mathbb { R } ^ { m } \} _ { r \in [ N / ( 2 m ) ] }$ given by Eq. (1). Then $R o t _ { j N / ( 2 m ) } ( \tilde { x } )$ is the interleaved batching vector of $\{ R o t _ { j } ( \mathbf { x } ^ { ( r ) } ) _ { r \in [ N / ( 2 m ) ] } \}$

Finally, we define interleaved column/diagonal packing by merging interleaved batching with our matrix packing methods directly. Please refer to Appendix D.3.

## 3.3 COMPLEXITY COMPARISON

We provide a comparison of our matrix multiplications with matrix multiplications in prior FHE-based secure transformer inference works NEXUS Zhang et al. (2024a)(NDSS’25), THOR Moon et al. (2024)(CCS’25), and Powerformer Park et al. (2024b)(ACL’25). We focus on the total number of HE rotations, which is the most expensive underlying HE operations as discussed in Section 2.1.

In Table 1, we report the total number of HE rotations in all matrix multiplications of a transformer. <sup>1</sup> MOAI requires only 9648 HE rotations in all matrix multiplications in BERT model, achieving 22.9x, 2.3x, and 1.7x fewer HE rotations than NEXUS, THOR, and Powerformer, respectively.

For consistency with prior work, we include a comparison if the total number of key switches in $\mathsf { A p - }$ pendix E. Key switch is the key component in the relinearization process of ciphertext multiplication and the HE rotation. MOAI requires only 834 key switches for all matrix multiplications in a single multi-head attention block, MOAI achieves the least number of key switches.

## 4 NON-LINEAR FUNCTION EVALUATION

In this section we propose optimized evaluations of non-linear functions. Our novelty is that we do not require any rotations in evaluations of Softmax and LayerNorm. This greatly improves the speed of our algorithms. Table 3 gives a comparison between ours and NEXUS Zhang et al. (2024a) (NDSS’25) and THOR Moon et al. (2024) (CCS’25). When substitute BERT’s parameters into the equation, our rotation-free Softmax and LayerNorm save 2448 HE rotations in total.

Softmax. The Softmax function appears in softmax $( Q K ^ { \intercal } / \sqrt { d ^ { \prime } } ) V$ , but we eliminate the $\sqrt { d ^ { \prime } }$ term by folding it into $W _ { K }$ , saving one CKKS level. The input of Softmax is in diagonal packing, and we want the output of the Softmax function to be in diagonal packing, as the input to the next module. We introduce a novel FHE Softmax algorithm (Algorithm 1, Figure 1) that preserves this packing format and eliminates the need for rotations, based on the key insight in Lemma 4.

Table 1: Comparison of number of HE rotations in matrix multiplications. For short, we use σ to denote Softmax and omit $\sqrt { d ^ { \prime } } .$ . t is the intermediate size of the feed-forward layer. H is the number of heads. $c = 1 6$ is from THOR. k = 64 is from Powerformer.
<table><tr><td>Operation</td><td>Function</td><td> $\mathrm { N E X U S } \# \mathrm { R o t } ^ { \mathrm { H E } }$ </td><td>THOR #RotHE</td><td>Powerformer #RotHE</td><td>Our MOAI #RotHE</td></tr><tr><td rowspan="4">Attention block</td><td> $\overline { { \operatorname { E n c } ( X ) W _ { * } ^ { ( h ) } , * = Q , K , V } }$   $X \in \mathbb { R } ^ { m \times d } , W _ { * } ^ { ( h ) } \in \mathbb { R } ^ { d \times d ^ { \prime } } , h \in [ H ]$ </td><td> $\frac { 2 m d } { N / ( 2 m ) }$ </td><td> $\begin{array} { r } { \frac { m } { 2 } + 3 ( H - 2 ) \frac { d ^ { \prime } } { c } + \frac { 3 d ^ { \prime } } { c } } \end{array}$ </td><td> ${ \sqrt { \frac { 5 } { 9 } } } \cdot 2 { \sqrt { \frac { m d ( 3 d ) } { N / 2 } } }$ </td><td>0</td></tr><tr><td> $\overline { { \mathrm { E n c } ( Q ^ { ( h ) } ) \mathrm { E n c } ( ( K ^ { ( h ) } ) ^ { \top } ) } }$   $Q ^ { ( h ) } , \dot { K ^ { ( h ) } } \in \mathbb { R } ^ { m \times d ^ { \prime } }$ </td><td> $\frac { m d ^ { \prime } H } { N / ( 4 m ) }$ </td><td> $\begin{array} { c } { { \frac { m } { 2 c } ( \frac { m } { 2 } + 5 ) + \frac { m } { 2 } } } \\ { { + \frac { m } { 8 } ( \log c + 3 ) } } \end{array}$ </td><td> $( 1 6 + 1 4 { \sqrt { 2 } } ) { \sqrt { k } } + 3 k$ </td><td> $\begin{array} { r } { \frac { m d ^ { \prime } } { N / ( 2 m ) } \cdot H } \end{array}$ </td></tr><tr><td> $\overline { { \mathrm { E n c } ( \sigma ( Q ^ { ( h ) } ( K ^ { ( h ) } ) ^ { \top } ) \mathrm { E n c } ( V ^ { ( h ) } ) } }$   $\sigma ( Q ^ { ( h ) } ( \dot { K } ^ { ( \dot { h } ) } ) ^ { \intercal } ) \in \mathbb { R } ^ { m \times m } , V ^ { ( \dot { h } ) } \in \mathring { \mathbb { R } } ^ { m \times d ^ { \prime } }$ </td><td>N/A</td><td> $m ( { \log c + 3 } ) / { 4 + m { \frac { m + 2 } { 4 c } } }$ </td><td> $( 1 8 + 1 0 { \sqrt { 2 } } ) { \sqrt { k } } + 3 k$ </td><td> $\frac { d ^ { \prime } ( \sqrt { m } + m ) } { ( N / ( 2 m ) ) } \cdot H$ </td></tr><tr><td> $\overrightarrow { \mathrm { E n c } ( ( X ^ { ( 0 ) } \parallel \dots \parallel X ^ { ( H - 1 ) } ) ) W _ { f c 0 } }$   $\boldsymbol { X } ^ { ( \tilde { h } ) } \in \mathbb { R } ^ { \tilde { m } \times d ^ { \prime } } , W _ { f c 0 } \in \mathbb { R } ^ { d \times \tilde { d } }$ </td><td> $\frac { 2 d ^ { 2 } } { N / ( 4 m ) }$ </td><td> $d ^ { \prime } / 2 + 2 ( H / 2$  − 1)m/c</td><td> ${ \sqrt { \frac { 2 } { 3 } } } \cdot 2 { \sqrt { \frac { m d ^ { 2 } } { N / 2 } } }$ </td><td>0</td></tr><tr><td>FC1</td><td> $\operatorname { E n c } ( X ) W _ { f c 1 } X \in \mathbb { R } ^ { m \times d } , W _ { f c 1 } \in \mathbb { R } ^ { d \times t }$ </td><td> $\frac { 2 m d } { N / ( 4 m ) }$ </td><td> $\begin{array} { r } { \frac { m } { 2 } + \frac { 2 t } { c } } \end{array}$ </td><td> $\overline { { \sqrt { \frac { 1 } { 2 } \cdot 2 \sqrt { \frac { m d t } { N / 2 } } } } }$ </td><td>0</td></tr><tr><td>FC2</td><td> $\operatorname { E n c } ( X ) W _ { f c 2 } X \in \mathbb { R } ^ { m \times t } , W _ { f c 2 } \in \mathbb { R } ^ { t \times d }$ </td><td> $\frac { 2 m t } { N / ( 4 m ) }$ </td><td> $\textstyle { \frac { m } { 2 } } + { \frac { 2 d } { c } }$ </td><td> $\textstyle { \frac { \bigwedge { \frac { 1 } { 2 } } \cdot 2 { \sqrt { \frac { m t d } { N / 2 } } } } { \sqrt { \frac { 1 } { N / 2 } } } }$ </td><td>0</td></tr><tr><td colspan="2">Total (Substitute BERT&#x27;s parameters into the equation:  $N = 2 ^ { 1 6 } , m = 1 2 8 , d = 7 6 8 , d ^ { \prime } = 6 4 , H = 1 2 , \dot { 1 } 2 \mathrm { l a y e r s } )$ </td><td> $> 2 2 1 1 8 4$ </td><td>22224</td><td>16740</td><td>9648</td></tr></table>

Table 2: Comparison with NEXUS Zhang et al. (2024a), THOR Moon et al. (2024) and MOAI Softmax and LayerNorm.
<table><tr><td>Function</td><td> $\mathbf { N E X U S # R o t } ^ { \mathrm { H E } }$ </td><td> $\mathrm { T H O R \# R o t ^ { H E } }$ </td><td> $\mathbf { M O A I } \left( \mathbf { o u r s } \right) \# \mathrm { R o t } ^ { \mathrm { H E } }$ </td></tr><tr><td>softmax(X ∈ Rm×m)</td><td> $\overline { { \lceil \log _ { 2 } ( m ) \rceil 4 m \cdot 2 ^ { \lceil \log _ { 2 } ( m ) \rceil } / N } }$ </td><td> $\mathrm { \overline { { { / l o g _ { 2 } ( m ) } } } } \mathrm { \overline { { { | 4 m \cdot \cdot 2 ^ { | \log _ { 2 } ( m ) } | } } } } / { N }$ </td><td>0</td></tr><tr><td> ${ \overline { { \mathrm { L a y e r N o r m } ( X \in \mathbb { R } ^ { m \times d } ) } } }$ </td><td> $\overline { { \lceil \log _ { 2 } ( d ) \rceil 4 d \cdot 2 ^ { \lceil \log _ { 2 } ( d ) \rceil } / N } }$ </td><td> $\overbrace { \lceil \log _ { 2 } ( d ) \rceil 4 d \cdot 2 ^ { \lceil \log _ { 2 } ( d ) \rceil } / N }$ </td><td>0</td></tr><tr><td>Total (Substitute BERT&#x27;s parameters into the equation:  $N = 2 ^ { 1 6 } , m = 1 2 8 , d = 7 6 8 , H = 1 2 , 1 2 \mathrm { l a y e r s } )$ </td><td>2448</td><td>2448</td><td>0</td></tr></table>

Lemma 4.1. $L e t C \in \mathbb { R } ^ { m \times m }$ be a square matrix. The summation of columns equals to the summation of diagonals: $\begin{array} { r } { \sum _ { i } \mathbf { c } _ { i } ^ { \mathsf { T } } = \sum _ { i } \operatorname { D i a g } _ { i } ( \bar { C } ) } \end{array}$ .

![](images/07046ee1243fdba8df8ad0718c3e22e04e82d97d33cd9e497cfcd3674f34a627.jpg)  
Figure 1: Our rotation-free Softmax evaluation whose input and output are both in diagonal packing.

LayerNorm. Note that the input of LayerNorm is in column packing. We first give the SIMD style mean and variance algorithm, given input as $\mathrm { E n c } _ { c o l } ( X )$ where $\boldsymbol { X } \in \mathbb { R } ^ { m \times d }$ . For Enc(mean(X)), compute the summation of all encrypted columns of X and then multiply $1 / d \colon \mathrm { s M u l t } ( 1 / d , \sum _ { i } \bar { \mathrm { E n c } } ( \mathbf { x } _ { i } ^ { \mathsf { T } } ) )$ . For $\operatorname { E n c } ( v a r ( X ) )$ . First compute ${ \mathrm { c t } } _ { x ^ { 2 } } : = \mathrm { { \ M u l t } } ( \mathrm { E n c } ( \mathbf { x } _ { i } ) \ -$ Enc(mean(X)), Enc(x ) − Enc(mean(X))), i ∈ [d]. Then compute sMult $( 1 / d , \textstyle \sum _ { i \in [ d ] } \mathbf { c } \mathbf { i } _ { x ^ { 2 } } )$ We can merge two scalar multiplication of $1 / d$ to save one FHE level. Goldschmidt division algorithm Goldschmidt (1964); Qu & Xu is applied to obtain the inverse square root $( 1 / \sqrt { x } )$ of variance. Please refer to Algorithm 8 for the detailed explanation in Appendix F.

GELU. For the encrypted GELU algorithm using Equation 2, it suffices to approximate tanh(·) function by polynomial. The main challenge is the balance between the size of effective input interval and the polynomial degree. By Cheon et al. (2022), a degree-Ω(R) polynomial is required when using minimax approximation with fixed maximum error on the domain interval $[ - R , R ]$ . In our work, we approximate tanh(·) in Equation 2 using a degree-23 polynomial $P _ { 2 3 } ( x )$ on interval [−20, 10] which is enough in our experiment. If the model needs larger input interval, we apply the domain extension polynomial $\begin{array} { r } { B _ { r } ( \dot { x } ) : = \frac { x } { L ^ { r } } - \frac { 4 } { 2 7 B ^ { 2 } L ^ { 3 r } } x ^ { 3 } } \end{array}$ from Cheon et al. (2022). Let $P ( x )$ be the minimax polynomial of $\operatorname { t a n h } ( x ) \mathrm { o n } [ - B , \tilde { B } ]$ . Then $\overleftarrow { P } \circ B _ { 0 } ( x )$ ) can take input from a larger interval [−LB, LB] for proper L. (Theorem 4 in Cheon et al. (2022)). This can be further extended to input interval $[ - L ^ { s } B , L ^ { s } B ]$ . We also point out that using piecewise polynomial in NEXUS Zhang et al. (2024a) to approximate GELU has precision problem. Please refer to Appendix F.2 for a detailed analysis.

Table 3: Comparison with NEXUS Zhang et al. (2024a), THOR Moon et al. (2024) and MOAI Softmax and LayerNorm.
<table><tr><td>Function</td><td colspan="3"> $\mathrm { N E X U S } \# \mathrm { R o t } ^ { \mathrm { H E } }$ </td><td>MOAI (Ours)  $\# \mathrm { R o t } ^ { \mathrm { H E } }$ </td></tr><tr><td>softmax</td><td> $\overline { { \lceil \log _ { 2 } ( m ) \rceil 4 m \cdot 2 ^ { \lceil \log _ { 2 } ( m ) \rceil } / N } }$ </td><td></td><td></td><td>0</td></tr><tr><td>LayerNorm</td><td></td><td> $\overline { { \lceil \log _ { 2 } ( d ) \rceil 4 d \cdot 2 ^ { \lceil \log _ { 2 } ( d ) \rceil } / N } }$ </td><td></td><td>0</td></tr><tr><td> $\overline { { \# \mathrm { R o t } ^ { \mathrm { H E } } } }$  for BERT</td><td colspan="3">2448</td><td>0</td></tr></table>

```latex
Algorithm 1 HE Softmax evaluation: $\operatorname { E n c } _ { d i a g } ( Q K ^ { \intercal } )  \operatorname { E n c } _ { d i a g } ( \sigma ( Q K ^ { \intercal } ) )$
Input: E $\operatorname { l n c } _ { d i a g } ( Q K ^ { \tau } ) ;$ : Diagonal packing of encrypt matrix $Q K ^ { \intercal } \in \mathbb { R } ^ { m \times m } .$
Output: $\mathrm { E n c } _ { d i a g } ( \mathrm { s o f t m a x } ( \bar { Q } K ^ { \intercal } ) )$ : Diagonal packing of encrypt matrix $\sigma ( Q K ^ { \intercal } ) \in \mathbb { R } ^ { m \times m }$
1: Write $\operatorname { E n c } _ { d i a g } ( Q K ^ { \intercal } )$ as $\{ \mathrm { E n c } ( \mathrm { D i a g } _ { 0 } ( Q K ^ { \widehat { \mathsf { T } } } ) ) , \mathrm { E n c } ( \mathrm { D i a g } _ { 1 } ( \mathsf { \bar { Q } } K ^ { \mathsf { T } } ) ) , . . . , \mathrm { E n c } ( \dot { \mathrm { D i a g } } _ { m - 1 } ( Q K ^ { \mathsf { T } } ) ) \}$
2: Evaluate Enc(exp(Diag $\left( Q K ^ { \mathsf { T } } \right) )$ by SIMD polynomial evaluation $( 1 + x / 2 ^ { r } ) ^ { 2 ^ { r } }$
3: Add ciphertexts to obtain $\begin{array} { r } { \dot { \operatorname { c t } _ { s u m } } = \operatorname { E n c } ( \sum _ { i } \dot { \exp ( \operatorname { D i a g } _ { i } ( Q K ^ { \intercal } ) ) } ) . } \end{array}$
4: Evaluate FHE inverse algorithm on ct<sub>sum</sub> using the Goldschmidt division algorithm Goldschmidt (1964).
Let the result be $\begin{array} { r } { \mathsf { c t } _ { s u m ^ { - 1 } } : = \mathrm { E n c } ( 1 / \sum _ { i } \exp ( \mathrm { D i a g } _ { i } ( Q K ^ { \intercal } ) ) ) . } \end{array}$
5: For $i \in [ m ] ,$ , return Mult $( \operatorname { c t } _ { s u m ^ { - 1 } } , \operatorname { E n c } ( \exp ( \operatorname { D i a g } _ { i } ( Q K ^ { \intercal } ) ) ) )$ as the i-th component.
```

## 5 MOAI: FULL FLOW

The name MOAI also comes from the monolithic human figures on Easter Island. They are huge and stable. They are similar to our scheme: large and stable end-to-end inference, and with very few "rotations". We show our full diagram in Figure 2 of the BERT-base-uncased model, which is implemented and test in Section 6.1. The proposed packing flow, evaluation algorithms and interleaved batching techniques are applicable to other transformer models and even decoder-only models such as Llama3-8B.

We illustrate the dimensions and the packing formats of encrypted matrices in each module. The packing formats include column packing (Col-packing, Definition 3.1) and diagonal packing (Diagpacking, Definition 3.2). CPMM stands for ciphertext-plaintext matrix multiplication, while CCMM stands for ciphertext-ciphertext matrix multiplications. As shown in Figure 2, MOAI does not need conversion between the two packing methods.

Regarding to the number of HE rotations, Table 1 and Table 9 show that MOAI requires only 9,648 HE rotations in the full flow, yielding 23.2×, 2.56×, and 1.46× fewer HE rotations than NEXUS, THOR, and Powerformer, respectively. Unlike Powerformer, which bypasses HE rotations by replacing Softmax and LayerNorm, MOAI preserves both and eliminates all HE rotations with rotation-free evaluation algorithms.

About bootstrapping. Bootstrapping is an expensive operation that resets ciphertext noise, making its placement critical for performance. We make the following two observations: (1) A sufficiently accurate Softmax requires at least 10 iterations of the Goldschmidt division (line 4 in Algorithm 1, based on experimental results in Figure 5), which together with exponentiation and final multiplication amounts to at least 20 layers (8 for $e ^ { x }$ , 11 for division, and 1 for multiplication). (2) The runtime of matrix multiplication increases with noise level, as higher-level ciphertexts have larger moduli.

To address this, we perform bootstrapping on $\operatorname { c t } _ { s u m }$ after the third line of Algorithm 1, refreshing it before Goldschmidt division. This reduces the Softmax depth from 20 to 10 layers, allowing the two preceding matrix multiplications to be executed at a lower level. As a result, the attention block starts from lower layers, and a single bootstrapping operation on one ciphertext substantially decreases both multiplication and overall runtime. The optimized Softmax algorithm is given in Algorithm 9.

Figure 2: The full diagram of MOAI. The parameters are from the BERT-base-uncased model, which has 12 transformer layers and 12 heads in each layer. c is the number of classes.  
![](images/1a63fd00b52666d4c708fe486b727afb259743f3a58da05565b948bbac7b3e88.jpg)

## 6 EXPERIMENTAL RESULTS

Implementation details. We set the degree of the polynomial ring as $N = 2 ^ { 1 6 }$ , yielding 2<sup>15</sup> slots in a ciphertext. We chose a 1743-bit ciphertext modulus of in accordance to the Homomorphic Encryption Standard Albrecht et al. (2021) to achieve 128-bit security. We apply the scale propagation technique Bossuat et al. (2021) to better manage the noise propagation and set the scale to $2 ^ { \dot { 4 } 6 }$ in our implementation. Our codes are available here.<sup>2</sup>

CPU implementation. We implement MOAI in C++, utilizing the SEAL library SEAL for CKKS homomorphic encryption, FHE-MP-CNN<sup>3</sup> for CKKS bootstrapping, and OpenMP Dagum & Menon (1998) for multi-thread programming. All CPU results are evaluated on a Intel(R) Xeon(R) Platinum 8480+ CPU at 2.0 GHz with 56 cores.

GPU implementation. MOAI is implemented using the Phantom library<sup>4</sup> for CKKS encryption and bootstrapping. GPU results are evaluated on a single NVIDIA H200 GPU (end-to-end evaluation in Table 4) or a single NVIDIA A100 GPU (comparison with SOTAs in Table 5 and Table 6).

Model and Dataset. Same as SOTA works THOR Moon et al. (2024) and Powerformer Park et al. (2024b), we test MOAI’s end-to-end performance on BERT-base model. It has 12 attention heads and 12 layers with an embedding dimension of 768. The input token length is 128. In evaluations, we use the pre-trained BERT-base-uncased model fine-tuned on SST-2, QNLI, and RTE tasks.<sup>5</sup>

Microbenchmark. Our optimized algorithms achieve speedups of up to 22× for Softmax and 151× for LayerNorm compared to prior works. We report the microbenchmark results in Appendix H.1.

## 6.1 END-TO-END INFERENCE EVALUATION

Table 4 presents the end-to-end performance (amortized for 256 inputs). The results are obtained from the execution of the BERT model with 12 transformer layers. For evaluating BERT model with input contains up to 128 tokens, we report an amortized time of 9.6 minutes per input on CPU and 2.36 minutes per input on GPU. Beyond encoder-only models, we extend our methods to decoder-only transformers, demonstrating them on LLaMA-3-8B inference and reporting results in Appendix H.2.

Table 4: Breakdown of MOAI encrypted inference, amortized over 256 inputs
<table><tr><td>Component</td><td>Operation in each layer</td><td>Depth</td><td>Packing</td><td>(Data dimensions)× (Number of operations)</td><td>CPU time(s)</td><td>GPU time(s)</td></tr><tr><td rowspan="4">Attention</td><td>Pt-ct MatrixMul</td><td> $1 5  1 4$ </td><td> $\mathrm { C o l } \to \mathrm { C o l }$ </td><td> $( \mathbb { R } ^ { 1 2 8 \times 7 6 8 } \times \mathbb { R } ^ { 7 6 8 \times 6 4 } ) \times 3 6$ </td><td>37.4</td><td>10.2</td></tr><tr><td>Ct-ct MatrixMul</td><td> $1 4  1 3$ </td><td> $\mathbf { C o l } {  } \mathbf { D i a g }$ </td><td> $\left( \mathbb { R } ^ { 1 2 8 \times 6 4 } \times \mathbb { R } ^ { 6 4 \times 1 2 8 } \right) \times 1 2$ </td><td>40.3</td><td>8.55</td></tr><tr><td>Softmax</td><td> $1 3  3$ </td><td> $\mathrm { D i a g } \to \mathrm { D i a g }$ </td><td> $\mathbb { R } ^ { 1 2 8 \times 1 2 8 } \times 1 2$ </td><td>53.3</td><td>0.74</td></tr><tr><td>Ct-ct MatrixMul</td><td>3 → 2</td><td> $\mathrm { D i a g }  \mathrm { C o l }$ </td><td> $( \mathbb { R } ^ { 1 2 8 \times 1 2 8 } \times \mathbb { R } ^ { 1 2 8 \times 6 4 } ) \times 1 2$ </td><td>1.4</td><td>0.26</td></tr><tr><td rowspan="4">Self Output</td><td>Pt-ct MatrixMul</td><td> $2 \to 1$ </td><td> $\mathrm { C o l } \to \mathrm { C o l }$ </td><td> $\left( \mathbb { R } ^ { 1 2 8 \times 7 6 8 } \times \mathbb { R } ^ { 7 6 8 \times 7 6 8 } \right)$ </td><td>1.7</td><td>0.41</td></tr><tr><td>Bootstrapping</td><td> $1  2 1$ </td><td>Col → Col</td><td> $\mathbb { R } ^ { 1 2 8 \times 7 6 8 }$ </td><td>95.4</td><td>26.1</td></tr><tr><td>LayerNorm</td><td>21 → 1</td><td>Col → Col</td><td> $\mathbb { R } ^ { 1 2 8 \times 7 6 8 }$ </td><td>0.6</td><td>0.05</td></tr><tr><td>Bootstrapping</td><td> $1  2 1  1 0$ </td><td>Col → Col</td><td> $\mathbb { R } ^ { 1 2 8 \times 7 6 8 }$ </td><td>95.8</td><td>26.2</td></tr><tr><td rowspan="2">Intermediate</td><td>Pt-ct MatrixMul</td><td>10 → 9</td><td>Col → Col</td><td> $\begin{array} { c } { { ( \mathbb { R } ^ { 1 2 8 \times 7 6 8 } \times \mathbb { R } ^ { 7 6 8 \times 3 0 7 2 } ) } } \\ { { \mathbb { R } ^ { 1 2 8 \times 3 0 7 2 } } } \end{array}$ </td><td>44.1</td><td>12.7</td></tr><tr><td>GELU</td><td>9 → 2</td><td>Col → Col</td><td></td><td>3.3</td><td>1.38</td></tr><tr><td rowspan="4">Final</td><td>Pt-ct MatrixMul</td><td>2 → 1</td><td>Col → Col</td><td> $( \mathbb { R } ^ { 1 2 8 \times 3 0 7 2 } \times \mathbb { R } ^ { 3 0 7 2 \times 7 6 8 } )$ </td><td>7.1</td><td>2.5</td></tr><tr><td>Bootstrapping</td><td>1 → 21</td><td>Col → Col</td><td> $\mathbb { R } ^ { 1 2 8 \times 7 6 8 }$ </td><td>98.8</td><td>26.1</td></tr><tr><td>LayerNorm</td><td>21 → 1</td><td>Col → Col</td><td> $\mathbb { R } ^ { 1 2 8 \times 7 6 8 }$ </td><td>0.6</td><td>0.05</td></tr><tr><td>Bootstrapping</td><td>1 → 21 → 15</td><td>Col → Col</td><td> $\mathbb { R } ^ { 1 2 8 \times 7 6 8 }$ </td><td>94.8</td><td>26.07</td></tr><tr><td>Total</td><td></td><td></td><td></td><td></td><td>574.6</td><td>141.3</td></tr></table>

## 6.2 COMPARISON WITH SOTAS.

We compare MOAI with state-of-the-art works on a single NVIDIA A100 GPU, consistent with THOR Moon et al. (2024) and Powerformer Park et al. (2024b). For fair comparison, we also incorporate Powerformer’s pooler and classification evaluation algorithm into our packing method (column packing) to account for total evaluation time.

Similar to THOR, MOAI is designed as a plug-and-play FHE framework that can be seamlessly integrated into existing transformer architectures. This line of work does not modify the transformer components such as Softmax and LayerNorm, and does not need re-training or fine-tuning. Table 5 reports the layer-wise performance comparison between MOAI and THOR in the same environment with same ring dimension $( N = 2 ^ { 1 6 } )$ and slot size $( n = 2 ^ { 1 5 } )$ . By reducing the number of HE rotations and adopting consistent packing strategies, MOAI achieves a 52.8% reduction in total time.

Furthermore, we apply MOAI’s packing methods and evaluation algorithms to Powerformer which replaces Softmax and LayerNorm with FHE-friendly functions (BPmax and Batch LN) and retrains the model accordingly. Table 6 reports the layer-wise performance comparison between MOAI and Powerformer in the same environment, following Powerformer’s parameter setting, $\mathrm { i . e . , 2 ^ { 1 5 } }$ slots in a ciphertext and 11 multiplicative levels before bootstrapping. MOAI achieves a 55.7% reduction in total evaluation time, demonstrating that our packing methods and evaluation algorithms can significantly improve the efficiency of retrained, FHE-friendly transformer architectures.

<table><tr><td>Operation</td><td>Ours</td><td>THOR</td><td>Diff</td></tr><tr><td>Attention layer</td><td>16.54</td><td>49.77</td><td>33.23</td></tr><tr><td>Attention score</td><td>14.07</td><td>16.25</td><td>2.18</td></tr><tr><td>Softmax</td><td>2.19</td><td>15.53</td><td>13.34</td></tr><tr><td>Attention heads</td><td>0.29</td><td>13.08</td><td>12.79</td></tr><tr><td>Multi-head attention</td><td>0.48</td><td>27.43</td><td>26.95</td></tr><tr><td>LayerNorm1</td><td>0.15</td><td>7.13</td><td>6.98</td></tr><tr><td>FC1</td><td>16.18</td><td>49.8</td><td>33.62</td></tr><tr><td>GELU</td><td>3.3</td><td>29.42</td><td>26.94</td></tr><tr><td>FC2</td><td>2.88</td><td>49.19</td><td>46.31</td></tr><tr><td>LayerNorm2</td><td>0.15</td><td>4.1</td><td>3.95</td></tr><tr><td>Pooler &amp; Classification</td><td>0.7</td><td>2.7</td><td>2</td></tr><tr><td>Bootstrappings</td><td>227.84</td><td>337.86</td><td>110.02</td></tr><tr><td>Total</td><td>283.95</td><td>602.26</td><td>318.31</td></tr><tr><td>Total w/o Pooler &amp; Class.</td><td>283.25</td><td>599.56</td><td>316.31</td></tr></table>

Table 5: Breakdown of execution time on single A100 GPU compared to THOR

<table><tr><td>Operation</td><td>Ours</td><td>Powerformer</td><td>Diff</td></tr><tr><td>Attention layer</td><td>14.29</td><td>57.65</td><td>43.36</td></tr><tr><td>Attention score</td><td>6.61</td><td>28.76</td><td>22.15</td></tr><tr><td>BPmax</td><td>0.06</td><td>0.75</td><td>0.69</td></tr><tr><td>Attention heads</td><td>0.39</td><td>18.95</td><td>18.56</td></tr><tr><td>Multi-head attention</td><td>4.05</td><td>22.54</td><td>18.49</td></tr><tr><td>Batch LN1</td><td>0.004</td><td>0.37</td><td>0.366</td></tr><tr><td>FC1</td><td>4.79</td><td>59.21</td><td>54.42</td></tr><tr><td>GELU</td><td>5.2</td><td>8.31</td><td>5.83</td></tr><tr><td>FC2</td><td>13.79</td><td>43.77</td><td>29.98</td></tr><tr><td>Batch LN2</td><td>0.004</td><td>0.3</td><td>0.296</td></tr><tr><td>Pooler &amp; Classification</td><td>0.7</td><td>0.2</td><td>-0.5</td></tr><tr><td>Bootstrappings</td><td>102.5</td><td>103.72</td><td>1.22</td></tr><tr><td>Total</td><td>152.4</td><td>344.52</td><td>192.12</td></tr><tr><td>Total w/o Pooler &amp; Class.</td><td>151.7</td><td>344.32</td><td>192.62</td></tr></table>

a Table 6: Breakdown of execution time on a single A100 GPU compared to Powerformer

## 6.3 ACCURACY.

We analyze error propagation during encrypted evaluation by recording layer outputs and comparing them with plaintext results (Figure 3). The error grows over the first 6–7 layers but then stabilizes, indicating that it remains bounded. This shows that FHE evaluation does not cause significant accuracy loss. Further details on error variation within a layer are provided in Appendix H.3. Table 7 reports GLUE benchmark scores under both plaintext and encrypted inputs. The results indicate that the additional error introduced by FHE has only a negligible effect on inference accuracy.

![](images/2c1d6a87fbebc672d8335bbb4f12a7e14844c30c99d2be2b009ad63a8e0dd6b6.jpg)  
Figure 3: Error analysis of 12 layers’ output

<table><tr><td>Dataset</td><td>SST-2</td><td>QNLI</td><td>RTE</td></tr><tr><td>Unencrypted input</td><td>93.9</td><td>90.7</td><td>65.7</td></tr><tr><td>Encrypted input</td><td>93.8</td><td>90.5</td><td>65.5</td></tr><tr><td>Match count</td><td>1818</td><td>5437</td><td>2959</td></tr><tr><td>Mismatch count</td><td>4</td><td>27</td><td>42</td></tr></table>

Table 7: Scores on the GLUE benchmarks

## 7 CONCLUSION

In this work, we propose MOAI, an efficient and secure transformer inference framework on encrypted data. MOAI leverages consistent matrix packing strategies, rotation-free Softmax and LayerNorm evaluations, and interleaved batching, removing format conversion, minimizing HE rotations, achieving substantial speedups and outperforming the state-of-the-art HE-based transformer inference work THOR Moon et al. (2024)(CCS’25). The proposed techniques are applicable to other revised transformer-based architectures, like Powerformer Park et al. (2024b) (ACL’25), to speedup the secure inference time.

## 8 REPRODUCIBILITY STATEMENT

Our implementation is publicly available via the anonymous GitHub repository linked in the submission. All datasets used in our experiments are public. Parameter settings, libraries, models, datasets, and hardware are detailed in Section 6.

The execution time of THOR and Powerformer in Table 5 and Table 6 are taken from their paper, which are tested in the same hardware environment as ours (a single NVIDIA A100 GPU). More details can be found in Section 6.2. The execution time of NEXUS in Table 9 are obtained by invoking its Softmax and LayerNorm in its open-source implementation. The GPU execution time of THOR in Table 9 are obtained from their paper, which is tested in the same hardware environment as ours (a single NVIDIA A100 GPU). THOR does not provide an open-source CPU implementation nor report CPU results, so we omit the CPU time for THOR. More details can be found in Appendix H.1.

With these resources, all reported results in this work are reproducible.

## ACKNOWLEDGMENTS

This work is a result of the joint lab collaboration between Ant International and Nanyang Technological University, established as part of a five-year partnership to advance digital trust and privacy-enhancing technologies (PETs), with funding provided by Ant International. This research is supported by the National Research Foundation, Singapore and Infocomm Media Development Authority under its Trust Tech Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore and Infocomm Media Development Authority.

## REFERENCES

Allon Adir, Ehud Aharoni, Nir Drucker, Ronen Levy, Hayim Shaul, and Omri Soceanu. Homomorphic Encryptionfor Data Science (HE4DS). Springer, 2024.

Carlos Agulló-Domingo, Óscar Vera-López, Seyda Guzelhan, Lohit Daksha, Aymane El Jerari, Kaustubh Shivdikar, Rashmi Agrawal, David Kaeli, Ajay Joshi, and José L Abellán. Fideslib: A fully-fledged open-source fhe library for efficient ckks on gpus. In 2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), pp. 1–3. IEEE, 2025.

Ehud Aharoni, Allon Adir, Moran Baruch, Nir Drucker, Gilad Ezov, Ariel Farkash, Lev Greenberg, Ramy Masalha, Guy Moshkowich, Dov Murik, et al. Helayers: A tile tensors framework for large neural networks on encrypted data. Proceedings on Privacy Enhancing Technologies, 2023.

Martin Albrecht, Melissa Chase, Hao Chen, Jintai Ding, Shafi Goldwasser, Sergey Gorbunov, Shai Halevi, Jeffrey Hoffstein, Kim Laine, Kristin Lauter, et al. Homomorphic encryption standard. Protecting privacy through homomorphic encryption, pp. 31–62, 2021.

Jean-Philippe Bossuat, Christian Mouchet, Juan Troncoso-Pastoriza, and Jean-Pierre Hubaux. Efficient bootstrapping for approximate homomorphic encryption with non-sparse keys. In Annual International Conference on the Theory and Applications ofCryptographic Techniques, pp. 587– 617. Springer, 2021.

Florian Bourse, Michele Minelli, Matthias Minihold, and Pascal Paillier. Fast homomorphic evaluation of deep discretized neural networks. In Annual International Cryptology Conference, pp. 483–512. Springer, 2018.

Tianyu Chen, Hangbo Bao, Shaohan Huang, Li Dong, Binxing Jiao, Daxin Jiang, Haoyi Zhou, Jianxin Li, and Furu Wei. THE-X: Privacy-preserving transformer inference with homomorphic encryption. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Findings of the Association for Computational Linguistics: ACL 2022, pp. 3510–3520, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.277. URL https://aclanthology.org/2022.findings-acl.277/.

Jung Hee Cheon, Andrey Kim, Miran Kim, and Yongsoo Song. Homomorphic encryption for arithmetic of approximate numbers. In International Conference on the Theory and Application of Cryptology and Information Security, pp. 409–437. Springer, 2017.

Jung Hee Cheon, Kyoohyung Han, Andrey Kim, Miran Kim, and Yongsoo Song. Bootstrapping for approximate homomorphic encryption. In Advances in Cryptology–EUROCRYPT 2018: 37th Annual International Conference on the Theory and Applications of Cryptographic Techniques, Tel Aviv, Israel, April 29-May 3, 2018 Proceedings, Part I 37, pp. 360–384. Springer, 2018.

Jung Hee Cheon, Wootae Kim, and Jai Hyun Park. Efficient homomorphic evaluation on large intervals. IEEE Transactions on Information Forensics and Security, 17:2553–2568, 2022.

Leonardo Dagum and Ramesh Menon. Openmp: an industry standard api for shared-memory programming. IEEE computational science and engineering, 5(1):46–55, 1998.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Ye Dong, Wen-jie Lu, Yancheng Zheng, Haoqi Wu, Derun Zhao, Jin Tan, Zhicong Huang, Cheng Hong, Tao Wei, and Wenguang Chen. Puma: Secure inference of llama-7b in five minutes. arXiv preprint arXiv:2307.12533, 2023.

Austin Ebel, Karthik Garimella, and Brandon Reagen. Orion: A fully homomorphic encryption framework for deep learning. In Proceedings of the 30th ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems, Volume 2, pp. 734–749, 2025.

Craig Gentry. Fully homomorphic encryption using ideal lattices. In Proceedings of the forty-first annual ACM symposium on Theory ofcomputing, pp. 169–178, 2009.

Ran Gilad-Bachrach, Nathan Dowlin, Kim Laine, Kristin Lauter, Michael Naehrig, and John Wernsing. Cryptonets: Applying neural networks to encrypted data with high throughput and accuracy. In International conference on machine learning, pp. 201–210. PMLR, 2016.

Robert E Goldschmidt. Applications of division by convergence. PhD thesis, Massachusetts Institute of Technology, 1964.

Shai Halevi and Victor Shoup. Algorithms in helib. In Annual Cryptology Conference, pp. 554–571. Springer, 2014.

Shai Halevi and Victor Shoup. Faster homomorphic linear transformations in helib. In Annual International Cryptology Conference, pp. 93–120. Springer, 2018.

Shai Halevi and Victor Shoup. Bootstrapping for helib. Journal ofCryptology, 34(1):7, 2021.

Bobby He, Lorenzo Noci, Daniele Paliotta, Imanol Schlag, and Thomas Hofmann. Understanding and minimising outlier features in neural network training. In NeurIPS 2024, 2024.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

Nandan Kumar Jha and Brandon Reagen. Entropy-guided attention for private llms. arXiv preprint arXiv:2501.03489, 2025.

Kwok-Yan Lam, Xianhui Lu, Linru Zhang, Xiangning Wang, Huaxiong Wang, and Si Qi Goh. Efficient fhe-based privacy-enhanced neural network for trustworthy ai-as-a-service. IEEE Transactions on Dependable and Secure Computing, 21(5):4451–4468, 2024. doi: 10.1109/TDSC.2024. 3353536.

Lawrence Lim, Vikas Kalagi, Divyakant Agrawal, and Amr El Abbadi. Tricycle: Private transformer inference with tricyclic encodings. Cryptology ePrint Archive, Paper 2025/1200, 2025. URL https://eprint.iacr.org/2025/1200.

Wen-jie Lu, Zhicong Huang, Cheng Hong, Yiping Ma, and Hunter Qu. Pegasus: Bridging polynomial and non-polynomial evaluations in homomorphic encryption. In 2021 IEEE Symposium on Security and Privacy (SP), pp. 1057–1073. IEEE, 2021.

Wen-jie Lu, Zhicong Huang, Zhen Gu, Jingyu Li, Jian Liu, Cheng Hong, Kui Ren, Tao Wei, and WenGuang Chen. Bumblebee: Secure two-party inference framework for large transformers. Cryptology ePrint Archive, 2023.

Pratyush Mishra, Ryan Lehmkuhl, Akshayaram Srinivasan, Wenting Zheng, and Raluca Ada Popa. Delphi: A cryptographic inference service for neural networks. In 29th USENIX Security Symposium (USENIX Security 20), pp. 2505–2522, 2020.

Jungho Moon, Dongwoo Yoo, Xiaoqian Jiang, and Miran Kim. THOR: Secure transformer inference with homomorphic encryption. Cryptology ePrint Archive, Paper 2024/1881, 2024. URL https: //eprint.iacr.org/2024/1881.

Qi Pang, Jinhao Zhu, Helen Möllering, Wenting Zheng, and Thomas Schneider. Bolt: Privacypreserving, accurate and efficient inference for transformers. In 2024 IEEE Symposium on Security and Privacy (SP), pp. 4753–4771. IEEE, 2024.

Dongjin Park, Eunsang Lee, and Joon-Woo Lee. Powerformer: Efficient privacy-preserving transformer with batch rectifier-power max function and optimized homomorphic attention. Cryptology ePrint Archive, Paper 2024/1429, 2024a. URL https://eprint.iacr.org/2024/1429.

Dongjin Park, Eunsang Lee, and Joon-Woo Lee. Powerformer: Efficient and high-accuracy privacypreserving language model with homomorphic encryption. Cryptology ePrint Archive, 2024b.

Jai Hyun Park. Ciphertext-ciphertext matrix multiplication: Fast for large matrices. Cryptology ePrint Archive, 2025.

Hongyuan Qu and Guangwu Xu. Improvements of homomorphic evaluation of inverse square root. Available at SSRN 4258571.

Eugene Y Remez. Sur la détermination des polynômes d’approximation de degré donnée. Comm. Soc. Math. Kharkov, 10(196):41–63, 1934.

Donghwan Rho, Taeseong Kim, Minje Park, Jung Woo Kim, Hyunsik Chae, Ernest K Ryu, and Jung Hee Cheon. Encryption-friendly llm architecture. arXiv preprint arXiv:2410.02486, 2024.

SEAL. Microsoft SEAL (release 4.1). https://github.com/Microsoft/SEAL, January 2023. Microsoft Research, Redmond, WA.

Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. arXiv preprint arXiv:2402.17762, 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Hao Yang, Shiyu Shen, Wangchen Dai, Lu Zhou, Zhe Liu, and Yunlei Zhao. Phantom: A cudaaccelerated word-wise homomorphic encryption library. IEEE Transactions on Dependable and Secure Computing, 21(5):4895–4906, 2024.

Wenxuan Zeng, Ye Dong, Jinjin Zhou, Junming Ma, Jin Tan, Runsheng Wang, and Meng Li. Mpcache: Mpc-friendly kv cache eviction for efficient private large language model inference. arXiv e-prints, pp. arXiv–2501, 2025.

Jiawen Zhang, Xinpeng Yang, Lipeng He, Kejia Chen, Wen-jie Lu, Yinghao Wang, Xiaoyang Hou, Jian Liu, Kui Ren, and Xiaohu Yang. Secure transformer inference made non-interactive. Cryptology ePrint Archive, 2024a.

Linru Zhang, Xiangning Wang, Jiabo Wang, Rachael Pung, Huaxiong Wang, and Kwok-Yan Lam. An efficient fhe-enabled secure cloud–edge computing architecture for iomt data protection with its application to pandemic modeling. IEEE Internet ofThings Journal, 11(9):15272–15284, 2024b. doi: 10.1109/JIOT.2023.3348122.

Linru Zhang, Xiangning Wang, Xianhui Lu, Huaxiong Wang, and Kwok Yan Lam. LEAF: A lowlatency evaluation architecture for feedforward block in privacy-preserving transformer inference. Cryptology ePrint Archive, Paper 2025/956, 2025. URL https://eprint.iacr.org/ 2025/956.

Yancheng Zhang, Jiaqi Xue, Mengxin Zheng, Mimi Xie, Mingzhe Zhang, Lei Jiang, and Qian Lou. Cipherprune: Efficient and scalable private transformer inference. In The Thirteenth International Conference on Learning Representations.

Itamar Zimerman, Allon Adir, Ehud Aharoni, Matan Avitan, Moran Baruch, Nir Drucker, Jenny Lerner, Ramy Masalha, Reut Meiri, and Omri Soceanu. Power-softmax: Towards secure llm inference over encrypted data, 2024a. URL https://arxiv.org/abs/2410.09457.

Itamar Zimerman, Moran Baruch, Nir Drucker, Gilad Ezov, Omri Soceanu, and Lior Wolf. Converting transformers to polynomial form for secure inference over homomorphic encryption. In Forty-first International Conference on Machine Learning, 2024b. URL https://openreview.net/ forum?id=9HPoJ6ulgV.

## APPENDIX

## A USAGE OF LARGE LANGUAGE MODELS

ChatGPT is used only for improving grammar and wording in draft preparation. Authors reviewed and revised all content and take full responsibility for the intellectual content of the submission.

## B EXTENDED PRELIMINARIES: THE BERT MODEL

We define the mean and variance of a vector as functions from $\mathbb { R } ^ { n } \to \mathbb { R }$ . We denote the softmax function and layer normalization as functions from $\mathbb { R } ^ { n } \to \{ 0 , 1 \} ^ { n }$ , specifically:

$$
\mathrm { s o f t m a x } ( \mathbf { x } ) = \left[ { \frac { \exp ( x _ { i } ) } { \sum _ { r \in [ n ] } \exp ( x _ { r } ) } } \right] _ { i \in [ n ] } \qquad { \mathrm { L a y e r N o r m } } ( \mathbf { x } ) : = \left[ \gamma \cdot { \frac { x _ { i } - \mathrm { m e a n } ( \mathbf { x } ) } { \sqrt { \operatorname { v a r } ( \mathbf { x } ) } } } + \beta \right] .
$$

where $\gamma , \beta$ are model parameters. The functions softmax and LayerNorm can be directly extended to matrices, by applying the functions to each row of the input matrix.

Embedding. Let the input has m tokens. First it is embedded to m vectorized token embeddings, each of which has dimension d. We denote the it as matrix $X _ { e } \in \mathbb { R } ^ { m \times d }$ . d is also defined as the hidden size of the model. Let $\boldsymbol { P } \in \mathbb { R } ^ { m \times d }$ be the positional embeddings for the input tokens. Then, the embedding input is obtained by computing $\bar { X : = X _ { e } + P }$

Multi-Head Attention. Let $\boldsymbol { X } \in \mathbb { R } ^ { m \times d }$ be the input, $W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { d \times d ^ { \prime } }$ be weights matrices, and $\mathbf { b } _ { Q } , \mathbf { b } _ { K } , \mathbf { b } _ { V }$ be bias vectors. An attention block in head $\textit { h } \in \ [ H ]$ is defined as Attention $( Q , K , V ) =$ softmax $\left( \frac { Q K ^ { \tau } } { \sqrt { d ^ { \prime } } } \right) V \in \mathbb { R } ^ { d \times d ^ { \prime } }$ , where: $Q : = X W _ { Q } + \mathbf { 1 } ^ { \intercal } \mathbf { b } _ { Q } , \ K \ : =$ $X W _ { K } + \mathbf { 1 } ^ { \mathsf { T } } \mathbf { b } _ { K } , \ V : = X W _ { V } + \mathbf { 1 } ^ { \mathsf { T } } \mathbf { b } _ { V }$ . The multi-head attention is the execution of H parallel attention blocks and we set the h-th result to be $X ^ { ( h ) }$ where $d ^ { \prime } : = d / H$ . Then the H matrices are concatenated $\begin{array} { r } { X ^ { \operatorname { C o n c a t } } : = ( X ^ { ( 0 ) } \parallel X ^ { ( 1 ) } \parallel . . . \parallel X ^ { ( H - 1 ) } ) \in \mathbb { R } ^ { m \times d } } \end{array}$ . The attention block ends with a layer normalization procedure with a residual connection: $X : = \mathrm { L a y e r N o r m } ( X + X ^ { \mathrm { C o n c a t } } )$

Feed Forward Network. The feed forward network contains four components. A fully connected layer followed by the Gaussian Error Linear Units (GELU) Hendrycks & Gimpel (2016) activation function, another fully connected layer and LayerNorm. The GELU function is defined as $\begin{array} { r } { \mathrm { G E L U } ( x ) : = \frac { x } { 2 } \left( 1 + \frac { 2 } { \sqrt { \pi } } \int _ { 0 } ^ { x } e ^ { - t ^ { 2 } } d t \right) } \end{array}$ . Since there is no closed-form expression for GELU, Hendrycks & Gimpel (2016) suggested an approximation by tanh:

$$
\operatorname { G E L U } ( x ) \approx 0 . 5 x \left( 1 + \operatorname { t a n h } \left[ { \sqrt { 2 / \pi } } \left( x + 0 . 0 4 4 7 1 5 x ^ { 3 } \right) \right] \right) .\tag{2}
$$

Let $W _ { \mathsf { f c } 1 } , W _ { \mathsf { f c } 2 }$ be weights matrices, and $\mathbf { b } _ { \mathsf { f c } 1 } , \mathbf { b } _ { \mathsf { f c } 2 }$ be bias vectors. The feed forward network can be described as follows: $\begin{array} { r } { X : = \mathrm { L a y e r N o r m } ( X + \mathrm { G E L U } ( X W _ { \mathsf { f c 1 } } + \mathbf { 1 } ^ { \mathsf { T } } \mathbf { b _ { \mathsf { f c 1 } } } ) W _ { \mathsf { f c 2 } } + \mathbf { 1 } ^ { \mathsf { T } } \mathbf { b _ { \mathsf { f c 1 } } } ) \in \mathbb { R } ^ { m \times d } . } \end{array}$ The output of feed forward block is set to be the input of the attention block in the next transformer layer. The BERT-base-uncased model Devlin et al. (2019) is using $H = 1 2$ and it has 12 transformer layers, that is, it contains 12 attention blocks and 12 feed forward blocks.

After 12 transformer layers, the model applies a linear layer with bias and tanh(·) to the first token $\mathrm { ( [ C L S ] ) }$ as the aggregated sequence representation for classification tasks. Let this vector be $\mathbf { x } _ { \mathrm { [ C L S ] } } \in \mathbb { R } ^ { d }$ . The classification layer first computes tanh $\mathsf { \Omega } _ { \mathsf { l } } \bigl ( \mathbf { x } _ { \lceil \mathrm { C L S } \rceil } W + \mathbf { 1 } ^ { \tau } \mathbf { b } \bigr ) \in \mathbb { R } ^ { d }$ , denoted as x. Finally let c is the number of classes, the output is $\mathbf { x } W _ { \mathsf { c l s } } + \mathbf { 1 } ^ { \mathsf { T } } \mathbf { b } _ { \mathsf { c l s } } { } ^ { \mathsf { ' } } \in \mathbb { R } ^ { c }$

## C DETAILS OF MATRIX MULTIPLICATIONS

## C.1 THE BABY-STEP-GIANT-STEP METHOD

In Halevi & Shoup (2021) they optimized the algorithm by baby-step-giant-step method, and the number of HE rotations is reduced to $O ( \sqrt { m } )$ ). The idea is to write $m = g \times b$ . First for $i \in [ b ]$ , we compute $\mathrm { R o t } _ { i } ( \mathbf { v } )$ for b times. To compute rotation ${ \mathrm { R o t } } _ { k } ( \mathbf { v } )$ for $k > = b .$ , we can write $k = \alpha b + r$ with $r < b ,$ and

$$
\mathrm { r o t } _ { k } ( \mathbf { v } ) : = \mathrm { r o t } _ { \alpha b } \big ( \mathrm { r o t } _ { r } ( \mathbf { v } ) \big ) \ .
$$

Combining the above, and with the diagonals of C, we will have

$$
C { \bf v } ^ { \top } : = \sum _ { \alpha \in [ g ] } \mathrm { r o t } _ { \alpha b } ( \sum _ { r \in [ b ] } \mathrm { r o t } _ { m - \alpha b } \big ( \mathrm { D i a g } _ { \alpha b + r } ( C ) \big ) \otimes \mathrm { r o t } _ { r } ( { \bf v } ) ) \ : .\tag{3}
$$

## C.2 PSEUDO CODES IN SECTION 3

Algorithm 2 describes the detailed implementation of CPMM when X is encrypted in column packing.

```latex
Algorithm 2 Ciphertext-plaintext matrix multiplication $\operatorname { M u l t } _ { c t , p t } ( \operatorname { E n c } _ { c o l } ( X ) , W , B ) .$
Input: $\mathrm { E n c } _ { c o l } ( X )$ : Column packing of encrypted $\boldsymbol { X } \in \mathbb { R } ^ { m \times d } ;$ W: Weight matrix with size d $\times \ d ^ { \prime }$
P: Bias matrix with size $m \times d ^ { \prime } .$
Output: $\operatorname { E n c } _ { c o l } ( X W + B ) { \mathrm { : } }$ Column packing of encrypted $\boldsymbol { X } \boldsymbol { W } + \boldsymbol { B } \in \mathbb { R } ^ { m \times d ^ { \prime } }$
1: Let Ans := ∅.
2: Parse Enc $_ { c o l } ( X )$ as  Enc(x<sup>⊺</sup><sub>0</sub>), Enc $( \mathbf { x } _ { 1 } ^ { \mathsf { T } } ) , . . . ,$ , Enc $\big ( \mathbf { x } _ { d - 1 } ^ { \mathsf { T } } \big ) \big \}$
3: for $j \in [ d ^ { \prime } ]$ do
4: Let $c \dot { t _ { j } } : = ( 0 , 0 ) \in R _ { Q } ^ { 2 }$ be a naive ciphertext.
5: for $i \in [ d ]$ do
6: $\begin{array} { r } { c t _ { j } : = \mathrm { A d d } \left( c t _ { j } , \mathrm { s M u l t } ( w _ { i j } , \mathrm { E n c } ( \mathbf { x } _ { i } ^ { \mathsf { T } } ) ) \right) } \end{array}$
7: end for
8: Let $\mathbf { p } _ { j } ^ { \intercal }$ be the j-th column of P.
9: Ans $\dot { : } = A n s \cup \operatorname { A d d } ( c t _ { j } , \mathbf { p } _ { j } ^ { \intercal } )$
10: end for
11: Return Ans.
```

Algorithm 3 describes the detailed implementation for encrypted inputs. For the sake of simplicity, we assume $m = N / 2$ in Algorithm 3 and 4 now. Then the rotation of a vector with dimension m can be directly evaluated using one homomorphic rotation. When $m < N / 2$ and m $\left| \left( N / 2 \right) \right.$ , Halevi & Shoup (2014) uses sparse packing, i.e., using m slots with index $0 , N / ( \dot { 2 } m ) , 2 \cdot \dot { N / } ( \dot { 2 } m ) , . . . , ( m -$ $1 ) \cdot \bar { N } / ( 2 m )$ . We will elaborate the our method of full packing in Section 3.2.

Algorithm 3 Ciphertext-ciphertext matrix multiplication: $\operatorname { M u l t } _ { c o l , c o l } ( \operatorname { E n c } _ { c o l } ( Q ) , \operatorname { E n c } _ { c o l } ( K ) )$   
Input: En $\dot { \bf \Phi } _ { c o l } ( Q )$ , E $\mathrm { { r } } _ { c o l } ( K )$ : Column packing of encrypted matrices $Q , K \in \mathbb { R } ^ { m \times d ^ { \prime } }$   
Output: En $\mathsf { c } _ { d i a g } ( Q K ^ { \mathsf { T } } ) \mathrm { : }$ : Diagonal packing of encrypted matrix $Q K ^ { \tau } \in \mathbb { R } ^ { m \times m }$   
1: Let Ans $: = \varnothing .$   
2: Parse Enc $_ { \mathrm { { c o l } } } ( Q )$ as  Enc(q<sup>⊺</sup>), Enc $( \mathbf { q } _ { 1 } ^ { \intercal } ) , . . . ,$ Enc $\left( \mathbf { q } _ { d ^ { \prime } - 1 } ^ { \mathsf { T } } \right) \}$   
3: Parse $\mathrm { E n c } _ { c o l } ( K )$ as $\left\{ \operatorname { E n c } ( \mathbf { k } _ { 0 } ^ { \mathsf { T } } ) , \operatorname { E n c } ( \mathbf { k } _ { 1 } ^ { \mathsf { T } } ) , . . . , \operatorname { E n c } ( \mathbf { k } _ { d ^ { \prime } - 1 } ^ { \mathsf { T } } ) \right\}$   
4: for $j \in$ [m] do   
5: Let $c \dot { t _ { j } } : = ( 0 , 0 ) \in R _ { Q } ^ { 2 }$ be a naive ciphertext.   
6: for $i \in [ d ^ { \prime } ]$ do   
7: $\mathbf { i f } \ j > 0$ then   
8: $t e m p : = \mathrm { R o t } _ { j } ^ { \mathrm { H E } } ( \mathrm { E n c } ( \mathbf { k } _ { i } ^ { \intercal } ) )$   
9: else   
10: temp := Enc(k<sup>⊺</sup>)   
11: end if   
12: ct<sub>j</sub> := Add  ct<sub>j</sub>, Mult(Enc(q<sup>⊺</sup>), temp).   
13: end for   
14: Ans := Ans ∪ ct<sub>j</sub>.   
15: end for   
16: Return Ans.

```latex
Algorithm 4 Ciphertext-ciphertext matrix multiplication: $\begin{array} { r } { \operatorname { M u l t } _ { d i a g , c o l } ( \operatorname { E n c } _ { d i a g } ( C ) , \operatorname { E n c } _ { c o l } ( V ) ) } \end{array}$
Input: En ${ \dot { z } } _ { d i a g } ( C )$ : Diagonal packing of encrypted matrices $C \in \mathbb { R } ^ { m \times m }$ . Enc ${ \dot { \operatorname { \varepsilon } } } _ { c o l } ( V )$ : Column
packing of encrypted matrix $V \in \mathbb { R } ^ { m \times d ^ { \prime } }$
Output: $\mathrm { E n c } _ { c o l } ( C V )$ : Column packing of encrypted matrix $C V \in \mathbb { R } ^ { m \times d ^ { \prime } }$
1: Let $A n s : = \varnothing .$
2: Parse En $\mathrm { c } _ { d i a g } ( C )$ as $\{ \operatorname { E n c } ( \operatorname { D i a g } _ { 0 } ( C ) )$ , Enc(Diag $( C ) ) , \ldots ,$ Enc $( \mathrm { D i a g } _ { m - 1 } ( C ) ) \big \}$ . Let $c _ { i } : =$
Enc $( \operatorname { D i a g } _ { i } ( C ) )$ , for $i \in [ m ] .$
3: Parse Enc $_ { c o l } ( V )$ as  Enc $\left( \mathbf { v } _ { 0 } ^ { \mathsf { T } } \right)$ , Enc $( \mathbf { v } _ { 1 } ^ { \mathsf { T } } ) , . . . ,$ Enc $\big ( \mathbf { v } _ { d ^ { \prime } - 1 } ^ { \mathsf { T } } \big ) \big \}$
4: Let $b : = \lceil \sqrt { m } \rceil , g : = \lceil m / b \rceil$
5: for $j \in [ d ^ { \prime } ]$ do
6: Baby step: For $i \in [ b ] .$ , let $\beta _ { i } : = \mathrm { R o t } _ { i } ^ { \mathrm { H E } } ( \mathrm { E n c } ( \mathbf { v } _ { j } ^ { \mathsf { T } } ) )$
7: Giant step: Compute the ciphertext of $C \mathbf { v } _ { j } ^ { \intercal } \colon$
$c t _ { j } : = \sum _ { \alpha \in [ g ] } { \mathrm { R o t } } _ { \alpha b } ^ { \mathrm { H E } } \bigg ( \sum _ { r \in [ b ] } \mathrm { M u l t } ( { \mathrm { R o t } } _ { m - \alpha b } ^ { \mathrm { H E } } ( c _ { \alpha b + r } ) , \beta _ { r } ) \bigg ) \ .$
8: Ans := Ans $\cup c t _ { j }$
9: end for
10: Return Ans.
```

## C.3 FURTHER OPTIMIZATION BY HOISTING

Hoisting is a technique to reduce the number of decompositions in HE rotations where the same ciphertext will be rotated multiple times. Halevi & Shoup (2018). More concretely, suppose a ciphertext is rotated N times and let m denote the length of the gadget vector. Without hoisting, performing N HE rotations requires N automorphism, N decomposition and N key switching operations. With hoisting, the decomposition is performed only once and reused for all rotations, replacing the remaining $( { \bar { N } } - 1 )$ decompositions with $( N m - N )$ additional automorphisms. Because decomposition is more expensive than automorphism, hositing yields a efficiency improvement.

Now we report the efficiency improvements that can be obtained through hoisting. In summary, hoisting can accelerate the amortized time about at most 2.29 seconds out of the total amortized time cost of 141.3 seconds, i.e., an improvement of 1.62%. In MOAI, hoisting can accelerate rotations in ciphertext-ciphertext matrix multiplications. We tested the time cost of each operation in CKKS scheme using SEAL library SEAL. The ratio of time for automorphism:decomposition:key switching is roughly equal to $1 : 4 \dot { 0 } : 8 0$ . Therefore the portion of the improvement from hoisting is $\begin{array} { r } { \frac { ( N + 4 0 N \bar { + } 8 0 N ) - \bar { ( } m \bar { N } + 4 \bar { 0 } + 8 0 N ) } { N + 4 0 N + 8 0 N } = \frac { 4 1 - m } { 1 2 1 } - \frac { 4 0 } { 1 2 1 N } < \frac { 4 1 - \bar { m } } { 1 2 1 } } \end{array}$ . In SEAL library, m is usually set in the range of $1 0 - 2 0$ in secure inference of transformers, which indicates a $1 6 \% - 2 6 \%$ improvement when the same ciphertext is rotated multiple times.

Further, for fair comparison with existing works Moon et al. (2024); Park et al. (2024b); Zhang et al. (2024a), which do not indicate using the hoisting technique, all our comparisons $( \mathrm { e . g . }$ , Table 1, Table 3 and Table 8) and experimental results (e.g., Table 4, Table 5, Table 6, Table 10 and Table 12) do not use hoisting.

## D DETAILS OF INTERLEAVED BATCHING

## D.1 NAIVE BATCHING AND INTERLEAVED BATCHING

Suppose we are given $N / ( 2 m )$ vectors $\{ \mathbf { x } ^ { ( r ) } \in \mathbb { R } ^ { m } \} _ { r \in [ N / ( 2 m ) ] } \colon$   
$ { \mathbf { x } } ^ { ( 0 ) } = [ x _ { 0 } ^ { ( 0 ) } , x _ { 1 } ^ { ( 0 ) } , . . . , x _ { m - 1 } ^ { ( 0 ) } ] , . . . ,  { \mathbf { x } } ^ { ( N / ( 2 m ) - 1 ) } = [ x _ { 0 } ^ { ( N / ( 2 m ) - 1 ) } , x _ { 1 } ^ { ( N / ( 2 m ) - 1 ) } , . . . , x _ { m - 1 } ^ { ( N / ( 2 m ) - 1 ) } ] \phantom { . . . \ }$   
the naive idea of batching is to arrange them one by one (e.g., in Zhang et al. (2024a)):   
$\bar { \mathbf { x } } : = [ \mathbf { x } ^ { ( 0 ) } , \mathbf { x } ^ { ( 1 ) } , . . . , \mathbf { x } ^ { ( N / ( 2 m ) - 1 ) } ] = [ x _ { 0 } ^ { ( 0 ) } , . . . , x _ { 0 } ^ { ( 1 ) } , . . . , x _ { 0 } ^ { ( N / ( 2 m ) - 1 ) } , . . . , x _ { m - 1 } ^ { ( N / ( 2 m ) - 1 ) } ] \in \mathbb { R } ^ { N / 2 }$

We call x¯ the naive batching of $N / ( 2 m )$ vectors $\{ \mathbf { x } ^ { ( r ) } \in \mathbb { R } ^ { m } \} _ { r \in [ N / ( 2 m ) ] }$ . For example, consider batching $[ x _ { 0 } , x _ { 1 } , x _ { 2 } ]$ and $[ y _ { 0 } , y _ { 1 } , y _ { 2 } ]$ . The naive batching generates $[ x _ { 0 } , x _ { 1 } , x _ { 2 } , y _ { 0 } , y _ { 1 } , y _ { 2 } ]$ , while the interleaved batching generates $\left[ x _ { 0 } , y _ { 0 } , x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } \right]$ This naive batching method is compatible with the ciphertext-plaintext matrix multiplication (Algorithm 2), but it is not compatible with the ciphertext-ciphertext matrix multiplications (Algorithm 3 and Algorithm 4). More concretely, this naive batching method is not compatible with rotation. When “rotate" the batched vector (e.g., Line 8 of Algorithm 3), the desired output should be a ciphertext encrypting the following “internal rotations":

$$
[ \mathrm { R o t } _ { j } ( { \bf k } ^ { ( 0 ) } ) , \mathrm { R o t } _ { j } ( { \bf k } ^ { ( 1 ) } ) , . . . , \mathrm { R o t } _ { j } ( { \bf k } ^ { ( N / ( 2 m ) - 1 ) } ) ] .
$$

In other words, it should be rotating each $\mathbf { k } ^ { ( i ) }$ “internally", which is different from directly rotating the batched vector by $\mathrm { R o t } _ { j } ( \bar { \bf k } )$ . Moon et al. (2024) proposes a method to obtain internal rotation, using 2 normal CKKS rotations and 2 plaintext-ciphertext multiplications. As a toy example, consider batching $[ x _ { 0 } , x _ { 1 } , x _ { 2 } ]$ and $[ y _ { 0 } , y _ { 1 } , y _ { 2 } ]$ . To “internally" left rotate by 1, we should use the naive batching $\bar { \mathbf { x } } : = [ x _ { 0 } , x _ { 1 } , x _ { 2 } , y _ { 0 } , y _ { 1 } , y _ { 2 } ]$ to generate $[ x _ { 1 } , x _ { 2 } , x _ { 0 } , y _ { 1 } , y _ { 2 } , y _ { 0 } ]$ This can be done by $\mathrm { R o t } _ { 1 } ( \bar { \bf x } )$ ⊗ $[ 1 , 1 , 0 , 1 , 1 , 0 ] = [ x _ { 1 } , x _ { 2 } , 0 , y _ { 1 } , y _ { 2 } , 0 ]$ . Then $\operatorname { R o t } _ { - 2 } ( \bar { \bf x } ) \otimes [ 0 , \bar { 0 } , 1 , 0 , 0 , 1 ] = [ 0 , 0 , x _ { 0 } , 0 , 0 , y _ { 0 } ]$ . By summing them together we obtain $[ x _ { 1 } , x _ { 2 } , x _ { 0 } , y _ { 1 } , y _ { 2 } , y _ { 0 } ]$

## D.2 PROOF OF LEMMA 3.2

Proof. On the one hand, by Eq. (1), we have

$$
\begin{array} { r } { { \mathrm { R o t } _ { j N / ( 2 m ) } ( \tilde { \mathbf { x } } ) : = [ x _ { j } ^ { ( 0 ) } , x _ { j } ^ { ( 1 ) } , x _ { j } ^ { ( 2 ) } , . . . , x _ { j } ^ { ( N / ( 2 m ) - 1 ) } , } } \\ { { x _ { j + 1 } ^ { ( 0 ) } , x _ { j + 1 } ^ { ( 1 ) } , x _ { j + 1 } ^ { ( 2 ) } , . . . , x _ { j + 1 } ^ { ( N / ( 2 m ) - 1 ) } , } } \\ { { . . . } } \\ { { x _ { j - 1 } ^ { ( 0 ) } , x _ { j - 1 } ^ { ( 1 ) } , x _ { j - 1 } ^ { ( 2 ) } , . . . , x _ { j - 1 } ^ { ( N / ( 2 m ) - 1 ) } ] \ . } } \end{array}
$$

On the other hand, $\forall r \in [ N / ( 2 m ) ]$ ], we replace $\mathbf { x } ^ { ( r ) }$ in Eq. (1) by $\mathsf { \Delta t _ { 2 } } ( \mathbf { x } ^ { ( r ) } ) = [ x _ { j } ^ { ( r ) } , x _ { j + 1 } ^ { ( r ) } , . . . , x _ { j - 1 } ^ { ( r ) } ]$ Then we also obtain

$$
\begin{array} { r } { { \mathrm { R o t } _ { j N / ( 2 m ) } ( \tilde { \mathbf { x } } ) : = [ x _ { j } ^ { ( 0 ) } , x _ { j } ^ { ( 1 ) } , x _ { j } ^ { ( 2 ) } , . . . , x _ { j } ^ { ( N / ( 2 m ) - 1 ) } , } } \\ { { x _ { j + 1 } ^ { ( 0 ) } , x _ { j + 1 } ^ { ( 1 ) } , x _ { j + 1 } ^ { ( 2 ) } , . . . , x _ { j + 1 } ^ { ( N / ( 2 m ) - 1 ) } , } } \\ { { . . . } } \\ { { x _ { j - 1 } ^ { ( 0 ) } , x _ { j - 1 } ^ { ( 1 ) } , x _ { j - 1 } ^ { ( 2 ) } , . . . , x _ { j - 1 } ^ { ( N / ( 2 m ) - 1 ) } ] \ . } } \end{array}
$$

This finishes the proof.

## D.3 INTERLEAVED BATCHING FOR MATRICES

Similarly, suppose we are given $N / ( 2 m )$ matrices $\{ X ^ { ( r ) } \in \mathbb { R } ^ { m \times d } \} _ { r \in [ N / ( 2 m ) ] }$ . Let $\tilde { X } \in \mathbb { R } ^ { \frac { N } { 2 } \times d }$ be the interleaved batching matrix of $\{ X ^ { ( r ) } \in \mathbb { R } ^ { m \times d } \} _ { r \in [ N / ( 2 m ) ] }$ . The i-th column of X<sup>˜</sup> is $\tilde { \mathbf { x } } _ { i } ^ { \top } \in \mathbb { R } ^ { N / 2 }$ which is the interleaved batching vector of all the i-th column of each $X ^ { ( r ) }$ , for $r \in [ N / ( 2 m ) ]$ according to Eq. (1) <sup>7</sup>:

$$
\begin{array} { r l } & { \tilde { \mathbf { x } } _ { i } ^ { \intercal } : = [ x _ { i , 0 } ^ { ( 0 ) T } , x _ { i , 0 } ^ { ( 1 ) T } , x _ { i , 0 } ^ { ( 2 ) T } , . . . , x _ { i , 0 } ^ { ( N / ( 2 m ) - 1 ) T } , } \\ & { \qquad x _ { i , 1 } ^ { ( 0 ) T } , x _ { i , 1 } ^ { ( 1 ) T } , x _ { i , 1 } ^ { ( 2 ) T } , . . . , x _ { i , 1 } ^ { ( N / ( 2 m ) - 1 ) T } , } \\ & { \qquad \dots } \\ & { \qquad x _ { i , m - 1 } ^ { ( 0 ) T } , x _ { i , m - 1 } ^ { ( 1 ) T } , x _ { i , m - 1 } ^ { ( 2 ) T } , . . . , x _ { i , m - 1 } ^ { ( N / ( 2 m ) - 1 ) T } ] . } \end{array}\tag{4}
$$

Now we are can formally define the column packing and the diagonal packing for the interleaved batching.

Definition D.1 (Column Packing for Interleaved Batching). Suppose we are given $N / ( 2 m )$ matrices $\{ X ^ { ( r ) } \in \mathbb { R } ^ { m \times d } \} _ { r \in [ N / ( 2 m ) ] }$ , the interleaved batching matrix of them is $\tilde { X } \in \mathbb { R } ^ { \frac { N } { 2 } \times d }$ , whose i-th column is given by Eq. $( 4 ) , f o r i \in [ d ]$ . Then the column packing for $\tilde { X }$ is a set of d CKKS ciphertexts, denoted as:

$$
\mathrm { E n c } _ { c o l } ( \tilde { X } ) : = \left\{ \mathrm { E n c } ( \tilde { \mathbf { x } } _ { 0 } ^ { \mathsf { T } } ) , \mathrm { E n c } ( \tilde { \mathbf { x } } _ { 1 } ^ { \mathsf { T } } ) , . . . , \mathrm { E n c } ( \tilde { \mathbf { x } } _ { d - 1 } ^ { \mathsf { T } } ) \right\} ,
$$

where $\tilde { \mathbf { x } } _ { i } ^ { \mathsf { T } } \in \mathbb { R } ^ { N / 2 }$ is the i-th column $o f { \tilde { X } } .$ . We say that $\mathrm { E n c } _ { c o l } ( \tilde { X } )$ is the interleaved column packing ofencrypted $\{ X ^ { ( r ) } \in \mathbb { R } ^ { m \times d } \} _ { r \in [ N / ( 2 m ) ] } .$

Definition D.2 (Diagonal Packing for Interleaved Batching). Suppose we are given $N / ( 2 m )$ square matrices $\{ X ^ { ( r ) } \in \mathbb { R } ^ { m \times m } \} _ { r \in [ N / ( 2 m ) ] }$ . Define:

$$
\widetilde { d i a g } ( X ) : = \{ \widetilde { d i a g } ( X ) _ { i } \in \mathbb { R } ^ { N / 2 } \} _ { i \in [ m ] } ,
$$

where $\widetilde { d i a g } ( X ) _ { i }$ is the interleaved batching vector given by Eq. (1) using all the i-th (upper) diagonals of $X ^ { ( r ) }$ , for each $r \in [ N / ( 2 m ) ]$ . The (upper) diagonal packing of is a set of m CKKS ciphertexts, denoted as:

$$
\mathrm { E n c } _ { d i a g } ( \widetilde { d i a g } ( X ) ) : = \left\{ \mathrm { E n c } ( \widetilde { d i a g } ( X ) _ { 0 } ) , \mathrm { E n c } ( \widetilde { d i a g } ( X ) _ { 1 } ) , . . . , \mathrm { E n c } ( \widetilde { d i a g } ( X ) _ { m - 1 } ) \right\} .
$$

We use ${ \widetilde { d i a g } } ( X ) _ { i }$ <sub>i</sub> as a row vector. We say that $\operatorname { E n c } _ { d i a g } ( { \widetilde { d i a g } } ( X ) )$ is the interleaved diagonal packing of encrypted $\{ X ^ { ( r ) } \in \mathbb { R } ^ { m \times m } \} _ { r \in [ N / ( 2 m ) ] }$

We conclude that the column/diagonal packing for interleaved batching is compatible with all the three homomorphic matrix multiplication algorithms (Algorithm 2, 3 and 4) using Lemma 3.2. Note that all the input and output ciphertexts of these algorithms will be encrypting the vectors in the form of the interleaved batching. Algorithm 5, 6 and $\bar { 7 }$ are the modified algorithms for homomorphic matrix multiplications using the interleaved batching.

For the HE evaluations of softmax (Algorithm 1), layer normalization ( 8) and GELU, we assert that the algorithm with interleaved batching input remains the same. This is because in these algorithms, the rows of the input matrix are independently evaluated. In the HE evaluations of softmax and layer normalization, if we add a permutation of rows of the input matrix, there will be the same permutation of rows of the output matrix. Note that interleaved batching is effectively a permutation of rows of the matrix. Therefore, Algorithm 1 and 8 remains the same when the input is in interleaved batching.

```latex
Algorithm 5 Mul $\mathbf { \chi } _ { c t , p t } ( \operatorname { E n c } _ { c o l } ( \tilde { X } ) , W , P )$ using interleaved batching.
Input: En $\mathsf { \bar { c } } _ { c o l } ( \tilde { X } )$ : Interleaved column packing of encrypted $\{ X ^ { ( r ) } \ \in \ \mathbb { R } ^ { m \times d } \} _ { r \in [ N / ( 2 m ) ] } ;$ W:
Weight matrix with size $d \times d ^ { \prime } . P \colon$ Bias matrix with size m $\times \ d ^ { \prime }$
Output: $\mathrm { E n c } _ { c o l } ( \tilde { X } W + \tilde { P } )$ : Interleaved column packing of encrypted $\{ X ^ { ( r ) } W + P \in$
$\mathbb { R } ^ { m \times d ^ { \prime } } \} _ { r \in [ N / ( 2 m ) ] }$
1: Let Ans $: = \varnothing .$
2: Parse $\mathrm { E n c } _ { c o l } ( \tilde { X } )$ as  En $\colon ( \tilde { \mathbf { x } } _ { 0 } ^ { \mathsf { T } } )$ , Enc $\mathbf { \Gamma } ( \tilde { \mathbf { x } } _ { 1 } ^ { \mathsf { T } } ) , . . . ,$ Enc $\big ( \tilde { \mathbf { x } } _ { d - 1 } ^ { \mathsf { T } } \big ) \big \}$
3: Let $\tilde { P }$ be the interleaved batching matrix of $N / ( 2 m )$ repeated matrices $P .$
4: for $j \in [ d ^ { \prime } ]$ do
5: Let $c \dot { t _ { j } } : = ( 0 , 0 ) \in R _ { Q } ^ { 2 }$ be a naive ciphertext.
6: for $i \in [ d ]$ do
7: $c t _ { j } : = \mathrm { A d d } \left( c t _ { j } , \mathrm { s M u l t } ( w _ { i j } , \mathrm { E n c } ( \tilde { \mathbf { x } } _ { i } ^ { \mathsf { T } } ) \right)$
8: end for
9: Let $\tilde { \mathbf { p } } _ { j } ^ { \intercal } \in \mathbb { R } ^ { N / 2 }$ be j-th column of ${ \tilde { P } } ,$ i.e., the interleaved batching vector of $N / ( 2 m )$ repeated
j-th column of P.
10: $\overset { \cdot } { A n s } : = A n s \cup \mathrm { A d d } ( c t _ { j } , \tilde { \mathbf { p } } _ { j } ^ { \intercal } )$
11: end for
12: Return Ans.
```

```latex
Algorithm 6 Mul $_ { c o l , c o l } ( \mathrm { E n c } _ { c o l } ( \tilde { Q } )$ , En $\mathrm { \vert c } _ { c o l } ( \tilde { K } ) )$ using interleaved batching.
Input: $\mathrm { E n c } _ { c o l } ( \tilde { Q } ) , \mathrm { E n c } _ { c o l } ( \tilde { K } ) :$ Interleaved column packing of encrypted $\{ Q ^ { ( r ) }$ ∈
$\mathbb { R } ^ { m \times d ^ { \prime } } ]$ $\big \} _ { r \in [ N / ( 2 m ) ] }$ and encrypted $\{ K ^ { ( r ) } \in \mathbb { R } ^ { m \times d ^ { \prime } } \} _ { r \in [ N / ( 2 m ) ] } ,$ , respectively.
Output: $\operatorname { E n c } _ { d i a g } ( \widetilde { d i a g } ( Q K ^ { \mathsf { T } } ) )$ : Interleaved diagonal packing of encrypted $\{ Q ^ { ( r ) } ( K ^ { ( r ) } ) ^ { \intercal } \in$
$\begin{array} { r } { \mathbf { \tilde { \mathbb { R } } } ^ { m \times m } \mathbf { \tilde { \mu } } _ { \Gamma \in \left[ N / ( 2 m ) \right] } . } \end{array}$
1: Let Ans $: = \varnothing .$
2: Parse $\mathrm { E n c } _ { c o l } ( \tilde { Q } )$ as $\left\{ \mathrm { E n c } ( \tilde { \mathbf { q } } _ { 0 } ^ { \mathsf { T } } ) , \mathrm { E n c } ( \tilde { \mathbf { q } } _ { 1 } ^ { \mathsf { T } } ) , . . . , \mathrm { E n c } ( \tilde { \mathbf { q } } _ { d ^ { \prime } - 1 } ^ { \mathsf { T } } ) \right\} .$
3: Parse Enc ${ } _ { ; c o l } ( \tilde { K } )$ as  Enc(k<sup>˜⊺</sup>), Enc(k<sup>˜⊺</sup>), ..., Enc $( \tilde { \mathbf { k } } _ { d ^ { \prime } - 1 } ^ { \top } ) \big \}$
4: for $j \in [ m ]$ do
5: Let $c \dot { t } _ { j } : = ( 0 , 0 ) \in R _ { Q } ^ { 2 }$ be a naive ciphertext.
6: for $i \in [ d ^ { \prime } ]$ do
7: $\mathbf { i f } \ j > 0 $ then
8: $\mathrm { \dot { \ t e m } } p : = \mathbf { R o t } _ { j N / ( 2 m ) } ( \mathrm { E n c } ( \tilde { \mathbf { k } } _ { i } ^ { \top } ) )$
9: else
10: $t e m p : = \mathrm { E n c } ( \tilde { \mathbf { k } } _ { i } ^ { \intercal } )$
11: end if
12: $c t _ { j } : = \mathrm { A d d } \left( c t _ { j } , \mathrm { M u l t } ( \mathrm { E n c } ( \tilde { \mathbf { q } } _ { i } ^ { \mathsf { T } } ) , t e m p ) \right)$
13: end for
14: Ans := Ans ∪ ct<sub>j</sub>.
15: end for
16: Return Ans.
```

Algorithm 7 Mul $\scriptscriptstyle \zeta _ { d i a g , c o l } ( \mathrm { E n c } _ { d i a g } ( \widetilde { d i a g } ( C ) ) , \mathrm { E n c } _ { c o l } ( \tilde { V } ) )$ using interleaved batching.   
Input: En ${ \mathrm { \Sigma } } _ { d i a g } ( \widetilde { d i a g } ( C ) )$ : Interleaved diagonal packing of encrypted $\{ C ^ { ( r ) } \in \mathbb { R } ^ { m \times m } \} _ { r \in [ N / ( 2 m ) ] } .$   
En $\mathrm { c } _ { c o l } ( { \tilde { V } } ) { : }$ : Interleaved column packing of encrypted $\{ V ^ { ( r ) } \in \mathbb { R } ^ { m \times d ^ { \prime } } \} _ { r \in [ N / ( 2 m ) ] } .$   
Output: En ${ \widehat { \mathsf { c } _ { c o l } } } { \widetilde { \mathsf { ( C V ) } } } { \mathrm { : } }$ Interleaved column packing of encrypted $\{ C ^ { ( r ) } V ^ { ( r ) } \in \mathbb { R } ^ { m \times d ^ { \prime } } \} _ { r \in [ N / ( 2 m ) ] } ,$   
1: Let $A n s : = \varnothing .$   
2: Parse $\begin{array} { r } { \overbrace { \mathrm { E n c } _ { d i a g } ( \widetilde { d i a g } ( C ) ) } ^ { \nu \mathrm { e o } \ \cdot \nu \cdot } \equiv \int \mathrm { d } s \ \{ \mathrm { d } s \ \{ \mathrm { d } s \ \{ \mathrm { d } s \ \widetilde { \langle } \widetilde { d i a g } ( C ) _ { 0 } \rangle , \mathrm { E n c } ( \widetilde { d i a g } ( C ) _ { 1 } ) , \dots , \mathrm { E n c } ( \widetilde { d i a g } ( C ) _ { m - 1 } ) } \}  ^ { \nu \mathrm { e o } \ \ \cdot \ \mathrm { d } s }  \end{array}$ . Let   
$c _ { i } : = \operatorname { E n c } ( { \tilde { d i a g } } ( C ) _ { i } ) .$ , for $i \in [ m ] .$   
3: Parse $\mathrm { E n c } _ { c o l } ( \tilde { V } )$ as $\{ \mathrm { E n c } ( \tilde { \mathbf { v } } _ { 0 } ^ { \mathsf { T } } )$ , En $\mathbf { \tilde { \Psi } } ( \tilde { \mathbf { v } } _ { 1 } ^ { \mathsf { T } } ) , . . . ,$ Enc $\left( \tilde { \mathbf { v } } _ { d ^ { \prime } - 1 } ^ { \mathsf { T } } \right) \biggr \}$   
4: Let $b : = \lceil \sqrt { m } \rceil , g : = \lceil m / b \rceil .$   
5: for $j \in [ \dot { d } ^ { \prime } ]$ do   
6: Baby step: For $i \in [ b ] .$ , let $\beta _ { i } : = \mathbf { R o t } i N / ( 2 m ) ( \mathrm { E n c } ( \tilde { \mathbf { v } } _ { j } ^ { \mathsf { T } } ) )$   
7: Giant step: Compute the encrypted interleaved batching vector of $\{ C ^ { r } ( \mathbf { v } _ { j } ^ { ( r ) } ) ^ { \top } \} _ { r \in [ N / ( 2 m ) ] } :$   
$c t _ { j } : = \sum _ { \alpha \in [ g ] } { \mathrm { R o t } } _ { \alpha b N / ( 2 m ) } \Bigg ( \sum _ { r \in [ b ] } \mathrm { M u l t } \big ( { \mathrm { R o t } } _ { ( m - \alpha b ) N / ( 2 m ) } ( c _ { \alpha b + r } ) , \beta _ { r } \big ) \Bigg )$   
8: Ans := Ans $\cup c t _ { j } .$   
9: end for   
10: Return Ans.

## E COMPLEXITY COMPARISON

We provide a detailed comparison of the complexity of our matrix multiplications with prior FHEbased secure transformer inference works: NEXUS Zhang et al. (2024a)(NDSS’25), THOR Moon et al. (2024)(CCS’25), and Powerformer Park et al. (2024b)(ACL’25).

We focus on the number of key switches (KS), which refers the total count of relinearizations in ciphertext-ciphertext multiplication, rotations, conjugations and any other operations requiring a KS. The number of KS is counted in one transformer layer in the BERT-base model.

In Table 8, we provide a detailed comparison of the number of key switches required in the BERT-base model inference. For NEXUS, results are taken from Zhang et al. (2024a) and Table 4 in Moon et al. (2024), and normalized by batch size. <sup>8</sup> The results of THOR are from Table 4 in Moon et al. (2024). For Powerformer, equations are derived from Section C.2 and Algorithm 14, with numbers from Tables 11 and 12 in Park et al. (2024b). According to Algorithm 14 in Park et al. (2024b), Powerformer requires several BlockTrans, BlockSigma and BlockTau operations before each ciphertext–ciphertext matrix multiplication BlockMult’. All the 3 operations involve $\Theta ( { \sqrt { k } } )$ HE rotations (i.e., key switches in this case). The execution time breakdown in Table 6 further validates the results in Table 8. To evaluate all matrix multiplications in a single layer of transformer model, MOAI achieves the least number of key switches.

Table 8: Comparison of HE matrix multiplications in transformer architecture. The concrete parameters are based on BERT-base-uncased model: $m = 1 2 8 , d = 7 6 8 , d ^ { \prime } = 6 4 , t = 3 0 7 2 , H = { \bar { 1 } } 2$ . N is the CKKS ring dimension. $c = 1 6$ is from THOR. k = 64 is from Powerformer.
<table><tr><td rowspan="2">Function</td><td rowspan="2">Scheme</td><td colspan="2">Key switch</td></tr><tr><td>Equation</td><td>Number</td></tr><tr><td rowspan="4"> $\{ X W _ { * } ^ { ( h ) } \} _ { h \in [ H ] }$   $\boldsymbol { X } \in \mathbb { R } ^ { m \times d } , { W _ { * } } \in \dot { \mathbb { R } } ^ { d \times d ^ { \prime } }$   ${ * = Q , K , V }$ </td><td>NEXUS</td><td> $\overline { { 2 m d / ( N / ( 2 m ) ) } }$ </td><td>768</td></tr><tr><td>THOR</td><td> $\begin{array} { r } { \frac { m } { 2 } + 3 ( \underline { { H } } - 2 ) \frac { d ^ { \prime } } { c } + \frac { 3 d ^ { \prime } } { c } } \end{array}$ </td><td>192</td></tr><tr><td>Powerformer</td><td> $\approx { \sqrt { \frac { 5 } { 9 } } } \cdot 2 { \sqrt { \frac { m d ( 3 d ) } { N / 2 } } }$ </td><td>122</td></tr><tr><td>MOAI (ours)</td><td>0</td><td>0</td></tr><tr><td> $\{ Q ^ { ( h ) } ( K ^ { ( h ) } ) ^ { \intercal } \} _ { h \in [ H ] }$ </td><td>NEXUS THOR</td><td> $\overline { { m ( d ^ { \prime } + 1 ) H / ( N / ( 4 m ) ) } }$ </td><td>780</td></tr><tr><td> $Q ^ { ( h ) } \in \mathbb { R } ^ { m \times d ^ { \prime } }$ </td><td></td><td> $\begin{array} { r } { \frac { m } { 2 c } ( \frac { m } { 2 } + 7 ) + \frac { m } { 8 } ( \log c + 3 ) + \frac { m } { 2 } } \end{array}$ </td><td>460</td></tr><tr><td> $K ^ { ( h ) } \in \mathbb { R } ^ { m \times d ^ { \prime } }$ </td><td>Powerformer MOAI (ours)</td><td> $\approx ( 1 7 + 1 4 { \sqrt { 2 } } ) { \sqrt { k } } + 3 k$   $( ( m d ^ { \prime } + m ) / ( N ^ { \prime } ( 2 m ) ) ) \cdot H$ </td><td>486</td></tr><tr><td rowspan="4"> $\{ \mathrm { s o f t m a x } ( Q ^ { ( h ) } ( K ^ { ( h ) } ) ^ { \top } ) V ^ { ( h ) } \} _ { h \in [ H ] }$   $\mathrm { s o f t m a x } ( Q ^ { ( h ) } ( K ^ { ( h ) } ) ^ { \intercal } ) \in \mathbb { R } ^ { m \times \dot { m } }$ </td><td></td><td>N/A</td><td>390</td></tr><tr><td>NEXUS THOR</td><td></td><td>N/A</td></tr><tr><td>Powerformer</td><td> $m ( \log c + 3 ) / 4 + m { \frac { m + 6 } { 4 c } }$ </td><td>492</td></tr><tr><td>MOAI (ours)</td><td> $\approx ( 1 9 + 1 0 { \sqrt { 2 } } ) { \sqrt { k } } + 3 k$   $d ^ { \prime } ( \sqrt { m } + m + \sqrt { d ^ { \prime } } ) / ( N / ( 2 m ) ) \cdot H$ </td><td>457 444</td></tr><tr><td rowspan="4"> $( X ^ { ( 0 ) } \parallel \dots \parallel X ^ { ( H - 1 ) } ) W _ { f c }$   $X ^ { ( h ) } \in \mathbb { R } ^ { m \times d ^ { \prime } }$   $W _ { f c } \in \mathbb { R } ^ { d \times d }$ </td><td>NEXUS</td><td> $\overline { { 2 d ^ { 2 } / ( ( N / ( 4 m ) ) ) } }$ </td><td>9216</td></tr><tr><td>THOR</td><td> $\begin{array} { r } { \frac { d ^ { \prime } } { 2 } + 2 ( \frac { H } { 2 } - 1 ) \frac { m } { c } } \end{array}$ </td><td>122</td></tr><tr><td>Powerformer</td><td> $\approx \sqrt { \frac { 2 } { 3 } \cdot 2 \sqrt { \frac { m d ^ { 2 } } { N / 2 } } }$ </td><td>75</td></tr><tr><td>MOAI (ours)</td><td>0</td><td>0</td></tr><tr><td rowspan="4"> $X W _ { f c 1 }$   $\boldsymbol { X } \in \mathbb { R } ^ { m \times d } , \dot { \boldsymbol { W } } _ { f c 1 } \in \mathbb { R } ^ { d \times t }$ </td><td>NEXUS</td><td> $\overline { { ( 2 m d ) / ( N / ( 4 m ) ) } }$ </td><td>1536</td></tr><tr><td>THOR</td><td> $\textstyle { \frac { m } { 2 } } + { \frac { 2 t } { c } }$ </td><td>448</td></tr><tr><td>Powerformer</td><td> $\approx \sqrt { \frac { 1 } { 2 } } \cdot 2 \sqrt { \frac { m d t } { N / 2 } }$ </td><td>135</td></tr><tr><td>MOAI (ours)</td><td> $_ 0$ </td><td>0</td></tr><tr><td rowspan="4"> $X W _ { f c 2 }$   $\boldsymbol { X } \in \mathbb { R } ^ { m \times t } , \dot { W } _ { f c 2 } \in \mathbb { R } ^ { t \times d }$ </td><td>NEXUS</td><td> $\left( 2 m t \right) / ( N / ( 4 m ) )$ </td><td>6144</td></tr><tr><td>THOR</td><td> $\begin{array} { r } { \frac { m } { 2 } + \frac { 2 d } { c } } \end{array}$ </td><td>160</td></tr><tr><td>Powerformer</td><td></td><td></td></tr><tr><td></td><td> $\approx \sqrt { \frac { 1 } { 2 } } \cdot 2 \sqrt { \frac { m t d } { N / 2 } }$ </td><td>132</td></tr><tr><td rowspan="4">Total (one transformer layer)</td><td>MOAI (ours) NEXUS</td><td>0</td><td>0</td></tr><tr><td>THOR</td><td></td><td>&gt; 18444</td></tr><tr><td>Powerformer</td><td></td><td>1874</td></tr><tr><td>MOAI (ours)</td><td></td><td>1407 834</td></tr></table>

## F DETAILS OF NON-LINEAR EVALUATIONS

## F.1 LAYERNORM DETAILS

In this section we give the detailed implementation of LayerNorm(X). The overview of the flow is shown in Figure 4. Recall that for matrix input, LayerNorm is a “row-to-row"" function applied

![](images/6c17971645114279c04d9b80355b23f67ad506fec247491c6761e0e8aae6a8bb.jpg)  
Figure 4: The flow of HE evaluation of LayerNorm.

to each row of the input. Given one row $\mathbf { x } = [ x _ { i } ] _ { i } \in [ d ]$ from X, the i-th term of LayerNorm(x) is $\gamma ^ { \frac { x _ { i } - \mathrm { m e a n } ( \mathbf { x } ) } { \sqrt { \mathrm { v a r } ( \mathbf { x } ) } } + \beta }$ . After computing mean() and var() we can obtain two different forms of layerNorm where $S : = \textstyle \sum _ { i } x _ { i }$

$$
{ \frac { \gamma } { \sqrt { d } } } { \frac { d x _ { i } - S } { \sqrt { \sum _ { i } ( d x _ { i } - S ) ^ { 2 } / d ^ { 2 } } } } + \beta .\tag{5}
$$

$$
\frac { \gamma } { d } \frac { d x _ { i } - S } { \sqrt { \sum _ { i } ( d x _ { i } - S ) ^ { 2 } / d ^ { 3 } } } + \beta .\tag{6}
$$

We give the detailed implementation in Algorithm 8. The parameter $t \in \{ 1 , 2 \}$ means to choose Equation 5 or Equation 6. It is for the higher precision of the inverse square root $1 / { \sqrt { x } }$ . After attention block, we choose $t = 1$ and compute LayerNorm according to Equation 5. After feed forward block, we choose $t = 2$ and compute LayerNorm according to Equation 6.

Remark. We point out the problem of the HE LayerNorm implementation of NEXUS Zhang et al. (2024a). Comparing with our Equation 5 and 6, they propose:

$$
{ \sqrt { d } } \gamma \cdot { \frac { d x _ { i } - S } { \sqrt { \sum _ { i } ( d x _ { i } - S ) ^ { 2 } } } } + \beta .
$$

Note that the input to the inverse square root evaluation is $\textstyle \sum _ { i } ( d x _ { i } - S ) ^ { 2 }$ , which is about $\Theta ( d ^ { 3 } )$ In the BERT-base-uncased model $d = 7 6 8$ , which makes $\textstyle \sum _ { i } ( d x _ { i } - S ) ^ { 2 }$ extremely large and is out of the supported data range. Therefore their algorithm is unable to compute LayerNorm in BERT-base-uncased model.

## F.2 GELU APPROXIMATION

We approximate tanh(·) in Equation 2 using a degree-23 polynomial $P _ { 2 3 } ( x )$ on interval [−20, 10] which is enough in our experiment. The approximation method is the famous Remez algorithm Remez (1934). It is noteworthy to point out that all polynomial-approximation method only support a specific interval. As suggested by Cheon et al. (2022), a degree-Ω(R) polynomial is required when using Remez or other minimax approximation with fixed maximum error on the domain interval $[ - \bar { R } , R ]$ . If the model needs larger input interval, we apply the domain extension polynomial $\begin{array} { r } { B _ { r } ( x ) \mathrel { \mathop : } = \frac { x } { L ^ { r } } - \frac { 4 } { 2 7 B ^ { 2 } L ^ { 3 r } } x ^ { 3 } } \end{array}$ from Cheon et al. (2022). In their algorithm, let $P ( x )$ be the approximation polynomial of tanh(x) on [−B, B]. Then $P \circ B _ { 0 } ( x )$ ) can take input from a larger interval $[ - L B , L B ]$ for proper L. (Theorem 4 in Cheon et al. (2022)). This can be further extended to input interval $[ - \dot { L ^ { s } } \dot { B _ { \bf \nu } } , L ^ { s } B ]$ , by using P ◦ ${ { B } _ { s - 1 } } \circ \ldots \circ { { B } _ { 0 } } ( x )$

```latex
Algorithm 8 HE evaluation of LayerNorm.
Input: Enc ${ \mathrm { \dot { \prime } } } _ { c o l } ( X )$ : Column packing of encrypted $\boldsymbol { X } \in \mathbb { R } ^ { m \times d } ;$ model parameter $\gamma , \beta \in \mathbb { R } ;$ algo
rithm type $t \in \{ 1 , 2 \}$
Output: Enc<sub>col</sub>(LayerNorm(X)): Column packing of encrypted LayerNorm $( X ) \in \mathbb { R } ^ { m \times d } .$
1: Let $A n s : = \varnothing .$
2: Parse $\mathrm { E n c } _ { c o l } ( X )$ as $\{$ En $\mathbf { \Sigma } : ( \mathbf { x } _ { 0 } ^ { \mathsf { T } } )$ , En $\mathbf { \partial } ; ( \mathbf { x } _ { 1 } ^ { \mathsf { T } } ) , \ldots$ , Enc $\big ( \mathbf { x } _ { d - 1 } ^ { \mathsf { T } } \big ) \big \}$
3: Compute Enc(d · mean(X)), where mean $( X ) \in \mathbb { R } ^ { m }$ is a vector containing the mean of each
row in $X \colon$
$\operatorname { E n c } ( d \cdot \operatorname { m e a n } ( X ) ) : = \sum _ { i \in [ d ] } \operatorname { E n c } ( \mathbf { x } _ { i } ^ { \intercal } ) ,$
i.e., the i-th element in mean $( X )$ is the mean of the i-th row in $X .$
4: Compute $\operatorname { E n c } ( d ^ { 3 } \cdot \operatorname { v a r } ( X ) )$ , where va $\ b { \cdot } ( X ) \in \mathbb { R } ^ { m }$ is a vector containing the variance of each
row in $X \colon$
5: for $j \in [ d ]$ do
6: Compute tmp := sMult(d, Enc(x<sup>⊺</sup>)) − Enc(d · mean(X)).
7: Let $\bar { t m p _ { j } } : = \bar { \mathrm { N } }$ ult(tmp, tmp).
8: end forLet Enc $\begin{array} { r } { ( d ^ { 3 } \cdot \mathrm { v a r } ( X ) ) = \sum _ { j } t m p _ { j } . } \end{array}$
9: $\mathbf { i f } t = 1$ then
10: Compute ct $\therefore = \operatorname { s M u l t } ( { \frac { 1 } { d ^ { 2 } } } , \operatorname { E n c } ( d ^ { 3 } \cdot \operatorname { v a r } ( X ) ) ) .$
11: Use the Goldschmidt division algorithm Goldschmidt (1964); Qu & Xu to obtain the inverse
square root $( 1 / \sqrt { x } )$ of ct, denoted as ct<sub>sqroot</sub>.
12: Compute $\begin{array} { r } { \mathsf { c t } _ { 1 } : = \mathrm { s M u l t } \big ( \frac { \gamma } { \sqrt { d } } , \mathsf { c t } _ { s q r o o t } \big ) . } \end{array}$
13: end if
14: i $\mathbf { f } \ t = 2$ then
15: Compute $\begin{array} { r } { \mathrm { c t } : = \mathrm { s M u l t } ( \frac { 1 } { d ^ { 3 } } } \end{array}$ , Enc $( d ^ { 3 } \cdot \operatorname { v a r } ( X ) ) ) .$
16: Use the Goldschmidt division algorithm Goldschmidt (1964); Qu & Xu to obtain the inverse
square root $( 1 / \sqrt { x } )$ of ct, denoted as $\operatorname { c t } _ { s q r o o t } .$
17: Compute $\begin{array} { r } { \mathsf { c t } _ { 2 } : = \dot { \mathrm { s M u l t } } \bigl ( \frac { \gamma } { d } , \mathsf { c t } _ { s q r o o t } \bigr ) . } \end{array}$
18: end if
19: for $i \in [ d ]$ do
20: $t m p _ { i } : = \mathrm { { M u l t } } \left( \mathsf { c t } _ { t } , ( \mathrm { s M u l t } ( d , \mathrm { E n c } ( \mathbf x _ { i } ^ { \mathsf { T } } ) ) - \mathrm { { E n c } } ( d \cdot \mathrm { { m e a n } } ( X ) ) ) \right) + \beta .$
21: Ans := Ans ∪ tmp<sub>i</sub>.
22: end for
23: Return Ans.
```

Very recently, Rho et al. (2024) proposed another method to control the data range. They pre-fix the desired data range of the output of each component in the transformer. Then they design punish function accordingly and perform pre-training to the model weights.

We also point out that using the piecewise polynomial in NEXUS Zhang et al. (2024a) to approximate GELU has several problems. NEXUS uses a different HE evaluation of GELU which is based on the line of MPC-based solution Lu et al. (2023); Dong et al. (2023). More concretely, NEXUS calculate GELU (R → R) by the following piecewise polynomial:

$$
{ \mathrm { G E L U } } ( x ) : = { \left\{ \begin{array} { l l } { 0 } & { x \leq - 4 } \\ { P _ { 3 } ( x ) } & { - 4 < x \leq - 1 . 9 5 } \\ { P _ { 6 } ( x ) } & { - 1 . 9 5 < x \leq 3 } \\ { x } & { x > 3 } \end{array} \right. }\tag{7}
$$

Here $P _ { 3 } ( x )$ and $P _ { 6 } ( x )$ are polynomials with degree 3 and 6.

First, NEXUS reports the supported input range is [−8, 8]. We find that this range is not enough for BERT-base-uncased model, and in our experiment the range is [−20, 10]. Next, their piecewise polynomial relies heavily on determining which interval does the encrypted x fall in. This is done by HE evaluation of the an approximated sign function. For encrypted number $x ,$ NEXUS uses 4 encrypt bits $b _ { 1 } ( x ) , b _ { 2 } ( x ) , b _ { 3 } ( \bar { x } ) , b _ { 4 } ( x )$ so that $b _ { i } ( x ) = 1$ if and only if x belongs to the i-th segment of Equation 7. Ideally, GELU(x) from Equation 7 can be evaluated by the following for $x \in [ - 8 , 8 ]$

$$
0 \cdot b _ { 1 } ( x ) + P _ { 3 } ( x ) b _ { 2 } ( x ) + P _ { 6 } ( x ) b _ { 3 } ( x ) + 3 b _ { 4 } ( x ) .
$$

But in fact, each $b _ { i }$ are approximated bits and it is a small value ϵ instead of being 0 exactly. This will cause problems. For example, when $x \ = \ - 8$ , in general $P _ { 6 } ( - 8 )$ is huge, and thus $P _ { 6 } ( - 8 ) b _ { 3 } ( - 8 ) \approx \epsilon P _ { 6 } ( - 8 )$ is far from 0. What is more, the actually data range we observed is $[ - 2 0 , 1 0 ]$ , making $P _ { 6 } ( - 2 0 )$ extremely huge. Theoretically one can choose extreme parameters of the CKKS evaluation of sign function so that the error ϵ is small, but this will increase the time cost dramatically.

## F.3 REMARKS ON OTHER NON-LINEAR FUNCTIONS

As introduced in Sun et al. (2024) and He et al. (2024), we must choose approximations carefully, due to the presence of well-known large activations that are critical for maintaining model performance. Compared with existing works NEXUS Zhang et al. (2024a) and THOR Park (2025), we use a different approximation method based on Cheon et al. (2022) which requires less multiplicative levels. Cheon et al. (2022) allows our solution to be extended to much larger intervals (i.e., wider input range) with very few depth consumption. This also accelerate the non-linear evaluations by starting from a lower ciphertext level, compared with NEXUS and THOR. In addition, our framework is designed in a plug-and-play manner, and it is fully compatible with a substitution of non-linear functions. Any activation function can be supported by our solution by simply adapting the corresponding approximation method.

Another line of works try to replace these non-linear functions in transformer to more FHE-friendly or MPC-friendly ones, such as Park et al. (2024b). A very recent work Jha & Reagen (2025) introduced a layerNorm-free solution utilizing HE and MPC. As we can substitute the non-linear functions, our framework can be easily adopted in Park et al. (2024b) and using ReLU in Jha & Reagen (2025).

## G OPTIMIZED SOFTMAX EVALUATION ALGORITHM

In this section we elaborate the optimized softmax algorithm. Figure 5 shows the error of the Goldschmidt division algorithm. Curves in different colors represent different iterations in the Goldschmidt division algorithm. The y-axis is the absolute error while the x-axis is the real data. The absolute error grows rapidly if the input data is near 0. For stable solution and higher precision, we have to ensure that input data range is not close to 0, and we need at least 10 iterations in the Goldschmidt division algorithm.

Next, at a cost of adding a bootstrapping on only one $\operatorname { c t } _ { s u m }$ after the third line of Algorithm 1, it can help to reduce the ciphertext modulus in previous matrix multiplications and can significantly reduce the time cost. The detailed optimized HE evaluation of softmax is shown in Algorithm 9.

![](images/569f3da93bbfec526de7058210ab67af3d9521642c49f37bdaeade8e5b9a699e.jpg)  
(a) Absolute error of Goldschmidt division algorithm in Softmax

![](images/ec5c0b50569dadf5067d8d7b5f8bbf4e8703f1ce08084d67768cd8700340d464.jpg)  
(b) Details of the bottom left corner of the figure

Figure 5: Absolute error of Goldschmidt division algorithm in Softmax  
```latex
Algorithm 9 HE Softmax evaluation: $\operatorname { E n c } _ { d i a g } ( Q K ^ { \intercal } )  \operatorname { E n c } _ { d i a g } ( \sigma ( Q K ^ { \intercal } ) )$
Input: En $\mathsf { z } _ { d i a g } ( Q K ^ { \mathsf { T } } )$ : Diagonal packing of encrypt matrix $Q K ^ { \tau } \in \mathbb { R } ^ { m \times m }$
Output: $\operatorname { E n c } _ { d i a g }$ (softmax $\cdot ( \bar { Q } K ^ { \mathsf { T } } ) \bar { ) }$ : Diagonal packing of encrypt matrix $\sigma ( Q K ^ { \tau } ) \in \mathbb { R } ^ { n }$ m×m
1: Write $\mathrm { E n c } _ { d i a g } ^ { \mathrm { } \mathrm { } \mathrm { } \mathrm { } \omega } ( Q K ^ { \mathsf { T } } ) \mathrm { a s } \{ \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } ( \mathrm { D i a g } _ { 0 } \langle Q K ^ { \mathsf { T } } \rangle ) , \mathrm { E n c } ^ { \mathsf { } } ( \mathrm { D i a g } _ { 1 } ( \dot { Q } K ^ { \mathsf { T } } ) ) , . . . , \mathrm { E n c } ( \mathrm { D i a g } _ { m - 1 } ( Q K ^ { \mathsf { T } } ) ) \}$
2: Evaluate $\operatorname { E n c } ( \exp ( \operatorname { D i a g } _ { i } ( Q K ^ { \intercal } ) ) )$ by SIMD polynomial evaluation $( 1 + x / 2 ^ { r } ) ^ { 2 ^ { r } }$
3: Add ciphertexts to obtain $\begin{array} { r } { \operatorname { c t } _ { s u m } = \operatorname { E n c } ( \sum _ { i } \operatorname { e x p } ( \operatorname { D i a g } _ { i } ( Q K ^ { \intercal } ) ) ) } \end{array}$
4: Bootstrap $\mathtt { c t } _ { s u m }$ to obtain refreshed ciphertext $\begin{array} { r } { \operatorname { c t } _ { s u m } ^ { \prime } = \operatorname { E n c } ( \sum _ { i } \exp ( \operatorname { D i a g } _ { i } ( Q K ^ { \intercal } ) ) ) } \end{array}$
5: Evaluate HE inverse algorithm on $\mathsf { c t } _ { s u m } ^ { \prime }$ using the Goldschmidt division algorithm Goldschmidt
(1964). Let the result be $\mathsf { c t } _ { s u m ^ { - 1 } } ^ { \prime } : = \mathrm { E n c } ( 1 / \sum _ { i } \exp ( \mathrm { D i a g } _ { i } ( Q K ^ { \intercal } ) ) )$
6: For $i \in [ m ]$ , return Mult $\left( \mathrm { c t } _ { s u m ^ { - 1 } } ^ { \prime } \right)$ , Enc ${ \mathrm { ( e x p } } ( { \overline { { \mathrm { D i a g } } } } _ { i } ( { \bar { Q } } K ^ { \mathsf { T } } ) { \bar { ) } } ) { \mathrm { ) } }$ as the i-th component.
```

## H DETAILS OF EXPERIMENTAL RESULTS

## H.1 MICRO-BENCHMARK EVALUATION OF SOFTMAX AND LAYERNORM

We compare our implementation of rotation-free SoftMax and LayerNorm evaluation algorithms with NEXUS Zhang et al. (2024a) (NDSS’25) and THOR Park et al. (2024a) (CCS’25).

All experiments are evaluated with same polynomial ring dimension $( N \ : = \ : 2 ^ { 1 6 } )$ and slot size $( n = 2 ^ { \bar { 1 } 5 } )$ . All CPU experiments are conducted on an Intel(R) Xeon(R) Platinum 8480+ processor using a single-thread setting, and all GPU experiments are performed on a single NVIDIA A100 GPU. We evaluate NEXUS using its open-source implementation<sup>9</sup>, invoking its SoftMax and LayerNorm modules in both CPU and GPU environments. In contrast, THOR does not provide an open-source CPU implementation nor report CPU results; therefore, only GPU timings are included for THOR.

The amortized time taken for both functions are given in Table 9. By eliminating all rotations during evaluation, our SoftMax evaluation algorithm achieves at least 4× speedup over NEXUS and 22× over THOR, while our LayerNorm evaluation algorithm achieves at least 87× and 151× speedup, respectively.

Table 9: Comparison between Softmax and LayerNorm implementations
<table><tr><td rowspan="2">Scheme</td><td colspan="3">Softmax</td><td colspan="3">LayerNorm</td></tr><tr><td>Size</td><td>CPU time(s)</td><td>GPU time(s)</td><td>Size</td><td>CPU time(s)</td><td>GPU time(s)</td></tr><tr><td>MOAI</td><td>128 × 128</td><td>0.62</td><td>0.005</td><td>128 × 768</td><td>0.38</td><td>0.0039</td></tr><tr><td>NEXUS (NDSS’25)</td><td>128 × 128</td><td>2.53</td><td>0.032</td><td>128 × 768</td><td>46.7</td><td>0.34</td></tr><tr><td>THOR (CCS’25)</td><td>128 × 128</td><td></td><td>0.11</td><td>128 × 768</td><td></td><td>0.59</td></tr></table>

## H.2 INFERENCE TIME ON LLAMA3-8B MODEL

In this section, we apply our methods in LLaMA-3-8B model. LLaMA-3-8B is a 32-layer decoderonly Transformer with 4096 hidden size, 32 attention heads (grouped-query attention), a feed-forward dimension of 14,336 (SwiGLU), RoPE positional embeddings, RMSNorm normalization, and a vocabulary size of 128k tokens.

To evaluate LLaMA-3-8B model, we implement rotation-free softmax with a causal mask and RMSNorm evaluations by using the same idea of rotation-free SoftMax and LayerNorm evaluations. SiLU evaluation, similar to GELU evaluation, uses a polynomial to approximate it according to Ebel et al. (2025). We incorporate NEXUS’s ArgMax evaluation algorithm Zhang et al. (2024a) into MOAI to measure total evaluation time. In the extract-token step, we retain only the last token and repack multiple ciphertexts into one, producing row-packed ciphertexts consistent with NEXUS’s ArgMax evaluation. The results in Table 10 aggregate microbenchmarks that cover all components required for LLaMA-3-8B inference. The matrix packing strategies remain consistent, without requiring format conversions or transpositions. We use 8-token input for consistency with PUMA Dong et al. (2023) and NEXUS Zhang et al. (2024a). Note that comparing our results with NEXUS is not appropriate because: (1) NEXUS’s inconsistent packing formats require costly conversions, which are not reflected in its reported performance, as observed in Moon et al. (2024); Lim et al. (2025); and (2) NEXUS does not implement all LLaMA-3-8B components, such as RoPE and SwiGLU.

Accuracy Table 11 reports GLUE benchmark scores under both plaintext and encrypted inputs. The results indicate that the additional error introduced by FHE has only a negligible effect on inference accuracy in LLaMA-3-8B model.

Small input length We give a experiment result of the small input length, since the decoding usually comes with length 1. We list the time cost in Table 12, testing on LLaMA-3-8B model. For clarity, we omit operations whose runtime contributes only a small portion of the total computation time (e.g., less than 0.1 second).

Meanwhile, integrating some popular techniques from plaintext transformer inference (e.g., KVcache, token pruning) with FHE is a useful and interesting direction for future work. A very recent work, Zeng et al. (2025) in NeurIPS’25 has integrated secure multi-party computation (MPC) with KV-cache eviction. Zhang et al. introduced token pruning integrated with MPC, which can further reduce the number of tokens in MPC version inference. While up to our knowledge, no prior work has explored FHE versions of the above techniques.

## H.3 ERROR ANALYSIS IN ONE TRANSFORMER LAYER

We divide one transformer layer into 4 parts: (1) Row 1 to Row 4 in Table 4. We call it attention layer; (2) Row 5 to Row 8 in Table 4. We call it self-output layer; (3) Row 9 to Row 10 in Table 4. We call it intermediate layer; (4) Row 11 to Row 14 in Table 4. We call it final output layer.

Figure 6 illustrated the absolute errors of evaluating the first layer with 44 input tokens. It shows that the error grows slowly along the evaluation and the medians of them fall into [0.01, 0.03] in the final output of the layer, which is consistent with our error analysis of evaluating 12 layers.

Table 10: Amortized breakdown of MOAI encrypted inference of LLaMA-3-8B with 8 input tokens
<table><tr><td>Operation in each layer</td><td>Description</td><td>Depth</td><td>(Data dimensions) × (Number of operations)</td><td>GPU time(s)</td></tr><tr><td>Pt-ct MatrixMul</td><td> $Q = X W _ { Q }$ </td><td>17 → 16</td><td> $\overline { { ( \mathbb { R } ^ { 8 \times 4 0 9 6 } \times \mathbb { R } ^ { 4 0 9 6 \times 4 0 9 6 } ) } }$ </td><td>20.2</td></tr><tr><td>Pt-ct MatrixMul</td><td> $K = \boldsymbol { X } \boldsymbol { W } _ { K } , \boldsymbol { V } \doteq \boldsymbol { X } \boldsymbol { W } _ { V }$ </td><td>17 → 16</td><td> $( \mathbb { R } ^ { 8 \times 4 0 9 6 } \times \mathbb { R } ^ { 4 0 9 6 \times 1 0 2 4 } ) \times 2$ </td><td>10.1</td></tr><tr><td>Pt-ct position-wise Mul</td><td> $\mathrm { R o P E } ( \mathrm { Q } )$ </td><td>16 → 15</td><td> $\mathbb { R } ^ { 8 \times 4 0 9 6 }$ </td><td>0.11</td></tr><tr><td>Pt-ct position-wise Mul</td><td> $\mathrm { R o P E } ( \mathrm { K } )$ </td><td>16 → 15</td><td> $\mathbb { R } ^ { 8 \times 1 0 2 4 }$ </td><td>0.025</td></tr><tr><td>Ct-ct MatrixMul</td><td> $Q K ^ { \mathsf { T } }$ </td><td>15 → 14</td><td> $( \mathbb { R } ^ { 8 \times 1 2 8 } \times \mathbb { R } ^ { 1 2 8 \times 8 } ) \times 3 2$ </td><td>8.6</td></tr><tr><td>Casual masked SoftMax</td><td> $\sigma ^ { \prime } ( Q K ^ { \boldsymbol { \mathsf { T } } } )$ </td><td>14 → 3</td><td> $\mathbb { R } ^ { 8 \times 8 } \times 3 2$ </td><td>0.6</td></tr><tr><td>Ct-ct MatrixMul</td><td> $\sigma ^ { \prime } ( Q K ^ { \boldsymbol { \mathsf { T } } } ) V$ </td><td>3 → 2</td><td> $( \mathbb { R } ^ { 8 \times 8 } \times \mathbb { R } ^ { 8 \times 1 2 8 } ) \times 3 2$ </td><td>0.2</td></tr><tr><td>Pt-ct MatrixMul</td><td> $X W _ { \mathrm { s e l f o u t p u t } }$ </td><td>2 → 1</td><td> $( \mathbb { R } ^ { 8 \times 4 0 9 6 } \times \mathbb { R } ^ { 4 0 9 6 \times 4 0 9 6 } )$ </td><td>1.3</td></tr><tr><td>Bootstrapping</td><td></td><td>1 → 20</td><td> $\mathbb { R } ^ { 8 \times 4 0 9 6 }$ </td><td>8.8</td></tr><tr><td>RMSNorm</td><td> $\mathrm { N o r m a l i z a t i o n }$ </td><td>20 → 1</td><td> $\mathbb { R } ^ { 8 \times 4 0 9 6 }$ </td><td>0.06</td></tr><tr><td>Bootstrapping</td><td></td><td>1 → 20 → 14</td><td> $\mathbb { R } ^ { 8 \times 4 0 9 6 }$ </td><td>8.8</td></tr><tr><td>Pt-ct MatrixMul</td><td> $\mathbf { u } = X W _ { u } , \mathbf { v } = X W _ { v }$ </td><td>14 → 13</td><td> $( \mathbb { R } ^ { 8 \times 4 0 9 6 } \times \mathbb { R } ^ { 4 0 9 6 \times 1 4 3 3 6 } ) \times 2$ </td><td>97.8</td></tr><tr><td>SiLU</td><td> $\mathbf { h } = \mathrm { S i L U } ( \mathbf { u } )$ </td><td>13 → 3</td><td> $\mathbb { R } ^ { 8 \times 1 4 3 3 6 }$ </td><td>6.5</td></tr><tr><td>Position-wise Mul</td><td> $\mathbf { h } = \mathbf { h } \odot \mathbf { v }$ </td><td>3 → 2</td><td> $\mathbb { R } ^ { 8 \times 1 4 3 3 6 }$ </td><td>0.02</td></tr><tr><td>Pt-ct MatrixMul</td><td> $\mathbf { y } = \mathbf { h } W _ { o }$ </td><td>2 → 1</td><td> $( \mathbb { R } ^ { 8 \times 1 4 3 3 6 } \times \mathbb { R } ^ { 1 4 3 3 6 \times 4 0 9 6 } )$ </td><td>48.8</td></tr><tr><td>Bootstrapping</td><td></td><td>1 → 21</td><td> $\mathbb { R } ^ { 8 \times 4 0 9 6 }$ </td><td>8.7</td></tr><tr><td>RMSNorm</td><td>Only in the last layer</td><td>21 → 2</td><td> $\overline { { \mathbb { R } ^ { 8 \times 4 0 9 6 } } }$ </td><td>0.001</td></tr><tr><td>Extract token</td><td>Only in the last layer</td><td>2 → 1</td><td> $\begin{array} { c } { ( \mathbb { R } ^ { 8 \times 4 0 9 6 } \to \mathbb { R } ^ { 1 \times 4 0 9 6 } ) } \\ { \mathbb { R } ^ { 1 \times 4 0 9 6 } } \end{array}$ </td><td>0.07</td></tr><tr><td>Bootstrapping</td><td>Only in the last layer</td><td> $1  2 1  2$ </td><td></td><td>0.015</td></tr><tr><td>Pt-ct Matrix-vector Mul</td><td>Only in the last layer</td><td>2 → 1</td><td> $( \mathbb { R } ^ { 1 \times 4 0 9 6 } \times \mathbb { R } ^ { 4 0 9 6 \times 1 2 8 2 5 6 } )$ </td><td>0.86</td></tr><tr><td>Bootstrapping</td><td>Only in the last layer</td><td> $1  2 1  1 1$ </td><td> $\mathbb { R } ^ { 1 \times 1 2 8 2 5 6 }$ </td><td>0.47</td></tr><tr><td>ArgMax</td><td>Only in the last layer</td><td>11 → 1</td><td>R128256</td><td>2.5</td></tr><tr><td>Total</td><td></td><td></td><td></td><td>224.53</td></tr></table>

Table 11: Scores on the GLUE benchmarks
<table><tr><td>Model</td><td>Dataset</td><td>SST-2</td><td>QNLI</td><td>RTE</td></tr><tr><td>LLaMA-3-8B</td><td>Unencrypted input Encrypted input</td><td>94.9 94.5</td><td>94.2 93.6</td><td>84.2 82.2</td></tr></table>

## I SET UP OPERATIONS OF THE CKKS SCHEME

Finally, we give more details of the CKKS scheme Cheon et al. (2017; 2018). The set up, encryption and decryption of the CKKS scheme can be summerized as follows.

$S e t u p ( 1 ^ { \lambda } )$ . The input is a security parameter λ. The output is a parameter set parms of the scheme, including the ciphertext modulus Q, and polynomial degree N.

Table 12: Amortized breakdown of MOAI encrypted inference of LLaMA-3-8B with input token length = 1
<table><tr><td>Operation in each layer</td><td>Description</td><td>(Data dimensions) × (Number of operations)</td><td>GPU time(s)</td></tr><tr><td>Attention Heads Pt-ct MatrixMul</td><td> $X \to \sigma ^ { \prime } ( Q K ^ { \intercal } ) V$ </td><td> $\mathbb { R } ^ { 1 \times 4 0 9 6 }  ( \mathbb { R } ^ { 1 \times 1 2 8 } \times 3 2 )$   $( \mathbb { R } ^ { 1 \times 4 0 9 6 } \times \mathbf { \dot { \mathbb { R } } } ^ { 4 0 9 6 \times 4 0 9 6 } )$ </td><td>1.57 0.18</td></tr><tr><td>Bootstrapping</td><td> $X W _ { \mathrm { s e l f o u t p u t } }$ </td><td> $\mathbb { R } ^ { 1 \times 4 0 9 6 }$ </td><td>1.18</td></tr><tr><td></td><td></td><td> $\mathbb { R } ^ { 1 \times 4 0 9 6 }$ </td><td></td></tr><tr><td>Bootstrapping after LayerNorm</td><td></td><td></td><td>1.15</td></tr><tr><td>Pt-ct MatrixMul</td><td> $\mathbf { u } = X W _ { u } , \mathbf { v } = X W _ { v }$ </td><td> $( \mathbb { R } ^ { 1 \times 4 0 9 6 } \times \mathbb { R } ^ { 4 0 9 6 \times 1 4 3 3 6 } ) \times 2$ </td><td>12.22</td></tr><tr><td>SiLU</td><td> $\mathbf { h } = \mathrm { S i L U } ( \mathbf { u } )$ </td><td> $\mathbb { R } ^ { 1 \times 1 4 3 3 6 }$ </td><td>0.82</td></tr><tr><td>Pt-ct MatrixMul</td><td> $\mathbf { y } = \mathbf { h } W _ { o }$ </td><td> $( \mathbb { R } ^ { 1 \times 1 4 3 3 6 } \times \mathbb { R } ^ { 1 4 3 3 6 \times 4 0 9 6 } )$ </td><td>5.93</td></tr><tr><td>Bootstrapping</td><td></td><td> $\mathbb { R } ^ { 1 \times 4 0 9 6 }$ </td><td>1.03</td></tr><tr><td>ArgMax</td><td>Only in the last layer</td><td>R128256</td><td>2.5</td></tr><tr><td>Total</td><td></td><td></td><td>26.94</td></tr></table>

![](images/0f00355737f38ab1b67785b9ce2f473e162dbe5c88bd5ffd936fd3bd51eafadc.jpg)  
(a) Absolute error of attention layer

![](images/a7dec3f83f4019e779f48e04dbce42360937bfff62d3d0052022ba5716a7fb60.jpg)  
(b) Absolute error of self-output layer

![](images/612dbbf6455a12daf52ae8301b1d8dc410680de854815dc0b0552d78f3060ce7.jpg)  
(c) Absolute error of intermediate layer

![](images/1b9fc33d78634c832295ecb5d4ed6a227837380091a349fd5f3821d13271e3fb.jpg)  
(d) Absolute error of final output layer  
Figure 6: Box and whisker figure of absolute error in one layer

• KeyGen(parms). The output includes secret key sk, public key pk and other auxiliary data called evaluation keys.

• Enc<sub>pk</sub>(v). Given the vector $\mathbf { v } \in \mathbb { C } ^ { N / 2 }$ , the output ciphertext is ct $\mathbf { \Psi } : = ( a , b ) \in R _ { Q } ^ { 2 }$ . For convenience, we will use Enc(v) when the public key is clear from context. We will use naive ciphertext to refer to $( 0 , \dot { 0 } ) \in R _ { Q } ^ { 2 }$ which is a ciphertext of 0.

$\mathrm { D e c } _ { \mathrm { s k } } ( \mathrm { c t } )$ . It outputs the vector $\mathbf { v } \in \mathbb { C } ^ { N / 2 }$ . sk is omitted when it is clear from context.