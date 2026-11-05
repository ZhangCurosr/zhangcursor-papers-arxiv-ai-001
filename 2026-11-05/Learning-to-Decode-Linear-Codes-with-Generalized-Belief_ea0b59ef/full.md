# Learning to Decode Linear Codes with Generalized Belief

Propagation Network

Samira Malek, Pennsylvania State University, Graduate Student Member, IEEE

Abstract—Error correcting codes are a fundamental component of communication systems, enabling reliable data transmission over noisy channels. Recently, neural decoders have demonstrated notable performance gains over classical iterative algorithms. In this paper, we propose a Generalized Belief Propagation Network (GBPNet) that preserves symmetry conditions while enhancing its learning capability. The resulting architecture implicitly gives rise to fully connected transformations via the generalized variable-node updates across iterations, alongside learned element-wise scaling operation that enhance decoding performance, while retaining the Tanner graph structure. In addition, we introduce two effective training strategies—progressive training and high-SNR training—that significantly improve decoding accuracy. Experimental results on various linear codes show that the proposed GBPNet consistently outperforms conventional BP and state-of-the-art neural decoding.

## I. INTRODUCTION

Error correcting codes play a fundamental role in reliable communication over noisy channels. Recently, deep learning-based decoders have shown promising performance, sometimes surpassing traditional decoding algorithms. Early approaches formulated decoding as a classification problem, where each codeword corresponds to a distinct class. However, because the number of codewords grows exponentially with the code length, the resulting classification task becomes computationally intractable for long codes. Consequently, such methods are generally limited to short codes [1], [2].

A syndrome-based deep learning framework was introduced in [3], enabling training with a single reference codeword while maintaining identical performance across all codewords. Building on this idea, subsequent works employed more expressive architectures, including Transformers [4], [5] and graph-attention networks [6]. However, these methods transform the received sequence into a syndrome-based representation and treat decoding primarily as an error-pattern prediction task, without explicitly exploiting the probabilistic dependencies and iterative message-passing structure of the Tanner graph.

Nachmani et al. [7] proposed a neural belief propagation (BP) decoder by unfolding BP iterations into a deep neural network. The method can be trained using only the all-zero codeword while explicitly exploiting the Tanner graph and iterative message-passing structure. Building on this framework, several BP-based neural decoders have been developed, including Hyper-Graph Network decoders [8], Tensor-Ring Neural BP decoders [9], and Adaptive Learned BP decoders [10], [11], which improve decoding performance through learnable message-update parameters.

Despite their success, existing BP-based neural decoders primarily rely on conventional BP. In contrast, GBPNet introduces generalized variable-node update rules that implicitly realize small fully connected blocks within the BP framework, together with learned scaling messages operation that enables richer message interactions across decoding iterations while preserving the Tanner graph structure. Furthermore, progressive training and high-SNR training strategies are developed to improve decoding performance. The symmetry condition is also proven to be preserved, enabling codeword-independent training. Experimental results on multiple linear codes demonstrate that GBPNet consistently outperforms conventional BP and state-of-the-art deep methods.

The remainder of this paper is organized as follows. After reviewing BP-based neural decoders in Section II, the generalized BP framework and the GBPNet architecture are presented in Section III. The symmetry condition of GBPNet is established in Section IV, showing that training on the allzero codeword yields identical performance for all codewords. Experimental results and comparisons with state-of-the-art methods are provided in Section V. Finally, conclusions are drawn in Section VI.

## II. BACKGROUND

BP is an iterative message-passing algorithm defined on a Tanner graph, where messages are exchanged between variable and check nodes. Fig. 1 illustrates an example Tanner graph. Given the channel output $y _ { v }$ and transmitted bit $x _ { v }$ , the channel loglikelihood ratio (LLR) is initialized as [12]:

$$
m _ { v } ^ { c h } = \log \Big ( \frac { \operatorname* { P r } ( x _ { v } = 1 \mid y _ { v } ) } { \operatorname* { P r } ( x _ { v } = - 1 \mid y _ { v } ) } \Big ) .\tag{1}
$$

A DNN form of BP was proposed in [7], which preserves the BP message-passing structure while introducing trainable weights into the variable-node updates. The variable-to-check message at iteration l is

$$
m _ { v , c } ^ { ( l ) } = \operatorname { t a n h } \left( \frac { 1 } { 2 } \Bigl ( b _ { v , c } ^ { ( l ) } m _ { v } ^ { c h } + \sum _ { c ^ { \prime } \in C _ { v } \backslash \{ c \} } \omega _ { v , c , c ^ { \prime } } ^ { ( l ) } m _ { c ^ { \prime } , v } ^ { ( l - 1 ) } \Bigr ) \right) \Bigr .\tag{2}
$$

where $C _ { v }$ denotes the set of check nodes connected to variable node $v ,$ and $m _ { c , v } ^ { ( - 1 ) } = 0$ for $l = 0$ . The check-node update remains identical to conventional BP:

$$
m _ { c , v } ^ { ( l ) } = 2 \mathrm { a r c t a n h } \left( \prod _ { v ^ { \prime } \in V _ { c } \backslash \{ v \} } m _ { v ^ { \prime } , c } ^ { ( l ) } \right) ,\tag{3}
$$

where $V _ { c }$ is the set of variable nodes connected to check node c. After l iterations, the posterior LLR estimate is computed using Eq. (4). The parameters $b _ { v , c } ^ { ( l ) } , \omega _ { v , c , c ^ { \prime } } ^ { ( l ) } , b _ { v } ^ { \mathrm { o u t } }$ , and $\omega _ { c ^ { \prime } , v } ^ { \mathrm { o u t } }$ are

learned during training, while setting them all to one recovers the classical BP algorithm.

$$
o _ { v } = b _ { v } ^ { \mathrm { o u t } } m _ { v } ^ { c h } + \sum _ { c ^ { \prime } \in C _ { v } } \omega _ { c ^ { \prime } , v } ^ { \mathrm { o u t } } m _ { c ^ { \prime } , v } ^ { ( l ) } .\tag{4}
$$

## III. GENERALIZED BELIEF PROPAGATION NETWORK

We generalize the belief propagation network [7] by replacing the one variable-layer in each decoding iteration with K cascaded variable sublayers, where $K \geq 1$ . Let

$$
v ^ { ( 0 ) } , v ^ { ( 1 ) } , \ldots , v ^ { ( K ) }\tag{5}
$$

denote the intermediate states across the variable-side transformation. For each edge $( v , c )$ , define

$$
m _ { v ^ { ( k - 1 ) } , v ^ { ( k ) } , c } ^ { ( l ) } , \qquad k = 1 , \dots , K ,\tag{6}
$$

as the message produced by the k-th variable sublayer at iteration l. For $l = 0$ , we have:

$$
m _ { v , c } ^ { ( 0 ) } = m _ { v } ,\tag{7}
$$

$$
m _ { c , v } ^ { ( 0 ) } = 2 \mathrm { A r c t a n h } \Bigg ( \prod _ { v ^ { \prime } \in V _ { c } \backslash \{ v \} } \mathrm { t a n h } \Big ( \frac { m _ { v ^ { \prime } , c } ^ { ( 0 ) } } { 2 } \Big ) \Bigg ) ,\tag{8}
$$

For notational consistency, we set

$$
m _ { c , v , v ^ { ( K ) } } ^ { ( 0 ) } \triangleq m _ { c , v } ^ { ( 0 ) } .\tag{9}
$$

For $l \geq 1$ , the first variable sublayer is given by

$$
\begin{array} { l } { m _ { v ^ { ( 0 ) } , v ^ { ( 1 ) } , c } ^ { ( l ) } = \operatorname { t a n h } \Bigg ( \frac { 1 } { 2 } \Big ( b _ { v , c } ^ { ( l , 1 ) } m _ { v } } \\ { + \displaystyle \sum _ { c ^ { \prime } \in C _ { v } \setminus \{ c \} } \omega _ { v , c , c ^ { \prime } } ^ { ( l , 1 ) } m _ { c ^ { \prime } , v , v ^ { ( K ) } } ^ { ( l - 1 ) } \Big ) \Bigg ) } \end{array}\tag{10}
$$

and for $k = 2 , \ldots , K$

$$
\begin{array} { r l } & { m _ { v ^ { ( k - 1 ) } , v ^ { ( k ) } , c } ^ { ( l ) } = \operatorname { t a n h } \Bigg ( \cfrac { 1 } { 2 } \Big ( b _ { v , c } ^ { ( l , k ) } m _ { v } } \\ & { \qquad + \displaystyle \sum _ { c ^ { \prime } \in C _ { v } \backslash \{ c \} } \omega _ { v , c , c ^ { \prime } } ^ { ( l , k ) } m _ { v ^ { ( k - 2 ) } , v ^ { ( k - 1 ) } , c ^ { \prime } } ^ { ( l ) } \Big ) \Bigg ) } \end{array}\tag{11}
$$

We further incorporate a learnable scaling layer after the last variable sublayer. Let $M ^ { ( l ) }$ denote the vector obtained by stacking all messages $m _ { v ^ { ( K - 1 ) } , v ^ { ( K ) } , c } ^ { ( l ) }$ together with the channel messages $m _ { v } ^ { \mathrm { c h } }$ . Let $\mathbf { \mathcal { A } } ^ { ( l ) }$ denote the element-wise scaling vector at iteration l.

$$
\tilde { M } ^ { ( l ) } = \mathcal { A } ^ { ( l ) } \odot M ^ { ( l ) } ,\tag{12}
$$

where ⊙ denotes the Hadamard product. Denote by $\tilde { m } _ { v ^ { ( K - 1 ) } , v ^ { ( K ) } , c } ^ { ( l ) }$ and $\tilde { m } _ { v } ^ { \mathrm { c h } }$ the corresponding components of $\tilde { M } ^ { ( l ) }$ . These reweighted messages are aggregated at the variable nodes as

$$
\begin{array} { l }  { \displaystyle { \hat { m } } _ { v ^ { ( K - 1 ) } , v ^ { ( K ) } , c } ^ { ( l ) } = \mathrm { t a n h } \left( \frac { 1 } { 2 } \Big ( { \tilde { m } } _ { v } ^ { \mathrm { c h } } \right)} \\ { { + \sum _ { c ^ { \prime } \in C _ { v } \setminus \{ c \} } { \tilde { m } } _ { v ^ { ( K - 1 ) } , v ^ { ( K ) } , c ^ { \prime } } ^ { ( l ) } \Big )  } } \end{array}\tag{13}
$$

![](images/50a7e0872b56d9f7b8e18a3821081dabb02478fea845bb03c5a867502c731c29.jpg)  
Fig. 1. Tanner graph with variable nodes $\{ v _ { i } \} _ { i = 1 } ^ { 5 }$ and check nodes $\{ c _ { i } \} _ { i = 1 } ^ { 4 }$

![](images/0f0e2abffa2b6abdb6d9dc39d3376b47486002d5301129fa832158cc61c4f34f.jpg)  
Fig. 2. Architecture of one GBPNet iteration for the Tanner graph in Fig. 1, comprising three cascaded variable sublayers with five fully connected blocks, followed by a learned element-wise scaling operation layer, message recombination, and a check-node update.

The check-layer is then computed as

$$
m _ { c , v , v ^ { ( K ) } } ^ { ( l ) } = 2 \mathrm { A r c t a n h } \bigg ( \prod _ { v ^ { \prime } \in V _ { c } \backslash \{ v \} } \hat { m } _ { v ^ { \prime ( K - 1 ) } , v ^ { \prime ( K ) } , c } ^ { ( l ) } \bigg ) .\tag{14}
$$

Fig. 2 illustrates one iteration of GBPNet for the Tanner graph presented in Fig. 1. The variable-node update is expanded into three cascaded sublayers composed of five small fully connected blocks, followed by a learned message-scaling layer and message recombination before the check-node update. The number of trainable parameters for the scaling layer is E (the number of edges in the Tanner graph), and for each variable sublayer, it is $\sum d _ { v _ { i } } ^ { 2 }$ , where $d _ { v _ { i } }$ is the degree of the i-th variable node. Therefore, each layer has $\sum d _ { v _ { i } } ^ { 2 } \times K + E$ learnable parameters.

## IV. SYMMETERY CONDITION

For block codes, standard message-passing symmetry conditions ensure codeword independent decoding error probability, allowing training on noisy realizations of a single codeword without loss of generality. This section establishes that the proposed generalized decoder preserves these conditions.

At iteration l, the variable-side computation consists of K cascaded variable sublayers, followed by a scaling layer and a recombination layer. We represent this entire block by a single effective variable-node mapping

$$
\Phi _ { v } ^ { ( l ) } : \Big ( m _ { v } , \{ m _ { c ^ { \prime } , v , v ^ { ( K ) } } ^ { ( l - 1 ) } \} _ { c ^ { \prime } \in C _ { v } \backslash \{ c \} } \Big ) \mapsto z _ { v , c } ^ { ( l ) } ,\tag{15}
$$

where $z _ { v , c } ^ { ( l ) }$ denotes the final message passed from variable node v to check node c at iteration l, while the effect of cascaded variable sublayers will be considered in Lemma 1. The check-node update is then given by

$$
m _ { c , v , v ^ { ( K ) } } ^ { ( l ) } = 2 \mathrm { A r c t a n h } \left( \prod _ { v ^ { \prime } \in V _ { c } \backslash \{ v \} } z _ { v ^ { \prime } , c } ^ { ( l ) } \right) .\tag{16}
$$

We next prove that $\Phi _ { v } ^ { ( l ) }$ satisfies the variable-node symmetry condition and that the check-node update satisfies the checknode symmetry condition. As shown in [8], [12], these two conditions are sufficient to ensure that the decoding error probability is independent of the transmitted codeword.

Lemma 1. The effective variable-node mapping $\Phi _ { v } ^ { ( l ) }$ is odd and satisfies the variable-node symmetry condition:

$$
\Phi _ { v } ^ { ( l ) } ( - m _ { v } , \{ - x _ { c ^ { \prime } } \} ) = - \Phi _ { v } ^ { ( l ) } ( m _ { v } , \{ x _ { c ^ { \prime } } \} ) .\tag{17}
$$

Proof. Let

$$
f _ { 1 } ^ { ( l ) } , f _ { 2 } ^ { ( l ) } , \dots , f _ { K } ^ { ( l ) }
$$

denote the K cascaded variable sublayers, let $a ^ { ( l ) }$ denote the scaling block, and let $r ^ { ( l ) }$ denote the recombination layer. Then the overall variable-side mapping can be written as

$$
\Phi _ { v } ^ { ( l ) } = r ^ { ( l ) } \circ a ^ { ( l ) } \circ f _ { K } ^ { ( l ) } \circ \cdots \circ f _ { 2 } ^ { ( l ) } \circ f _ { 1 } ^ { ( l ) } .\tag{18}
$$

For the variable sublayers in (10), (11), and recommbination layer (13), Since there is no bias term, replacing $m _ { v }$ and all incoming messages $m _ { v ^ { ( k - 2 ) } , v ^ { ( k - 1 ) } , c ^ { \prime } } ^ { ( l ) }$ by their negatives changes the argument of tanh by a factor of −1. Because tanh is odd, it follows that $f _ { i } ^ { ( l ) }$ and $r ^ { ( l ) }$ are odd.

Next, the message scaling layer is given by

$$
a ^ { ( l ) } ( M ^ { ( l ) } ) = \mathcal { A } ^ { ( l ) } \odot M ^ { ( l ) } .\tag{19}
$$

Because $\mathbf { \mathcal { A } } ^ { ( l ) }$ is a learned vector independent of $M ^ { ( l ) }$

$$
a ^ { ( l ) } ( - M ^ { ( l ) } ) = \mathcal { A } ^ { ( l ) } \odot ( - M ^ { ( l ) } ) = - \big ( \mathcal { A } ^ { ( l ) } \odot M ^ { ( l ) } \big ) = - a ^ { ( l ) } \big ( M ^ { ( l ) } \odot ( - M ^ { ( l ) } ) \big )\tag{20}
$$

Therefore, the message scaling block is odd. Since the composition of odd mappings is odd, $\Phi _ { v } ^ { ( l ) }$ is odd. Hence,

$$
\Phi _ { v } ^ { ( l ) } ( - m _ { v } , \left\{ - x _ { c ^ { \prime } } \right\} ) = - \Phi _ { v } ^ { ( l ) } ( m _ { v } , \left\{ x _ { c ^ { \prime } } \right\} ) ,
$$

which proves the variable-node symmetry condition.

Lemma 2. The check-node update is given by (16). Then the check-node symmetry condition holds:

$$
m _ { c , v , v ^ { ( K ) } } ^ { ( l ) } ( \{ b _ { v ^ { \prime } } z _ { v ^ { \prime } , c } ^ { ( l ) } \} ) = \left( \prod _ { v ^ { \prime } \in V _ { c } \backslash \{ v \} } b _ { v ^ { \prime } } \right) m _ { c , v , v ^ { ( K ) } } ^ { ( l ) } ( \{ z _ { v ^ { \prime } , c } ^ { ( l ) } \} ) ,
$$

for any $b _ { v ^ { \prime } } \in \{ - 1 , + 1 \}$

(21)

Proof. Let

$$
\Psi ( z _ { 1 } , \dots , z _ { r } ) = 2 \operatorname { A r c t a n h } \left( \prod _ { i = 1 } ^ { r } z _ { i } \right) , \qquad r = | V _ { c } | - 1 .\tag{22}
$$

Then for any sign pattern $b _ { 1 } , \dots , b _ { r } \in \{ - 1 , + 1 \}$

$$
\begin{array} { l } { \displaystyle \Psi ( b _ { 1 } z _ { 1 } , \dots , b _ { r } z _ { r } ) = 2 \ \mathrm { a r c t a n h } \Bigg ( \prod _ { i = 1 } ^ { r } b _ { i } z _ { i } \Bigg ) } \\ { \displaystyle = 2 \ \mathrm { a r c t a n h } \Bigg ( \Bigg ( \prod _ { i = 1 } ^ { r } b _ { i } \Bigg ) \left( \prod _ { i = 1 } ^ { r } z _ { i } \right) \Bigg ) . } \end{array}\tag{23}
$$

Since arctanh(·) is an odd function, Eq. (24) follows, satisfying the check-node symmetry condition.

$$
\Psi ( b _ { 1 } z _ { 1 } , \ldots , b _ { r } z _ { r } ) = \left( \prod _ { i = 1 } ^ { r } b _ { i } \right) \Psi ( z _ { 1 } , \ldots , z _ { r } ) .\tag{24}
$$

## V. EXPERIMENTS

We evaluate GBPNet on BCH, Polar, LDPC, and MacKay codes, comparing its bit error rate (BER) performance with conventional BP and state-of-the-art neural decoders, as reported in Table I. The parity-check matrix of the codes were obtained from [13]. Training samples are generated by transmitting the all-zero codeword using BPSK modulation over an additive white Gaussian noise (AWGN) channel, justified by the message-passing symmetry of GBPNet. A total of $5 \times 1 0 ^ { \circ }$ training samples are used, while testing employs random codewords until at least 500 frame errors are observed at each SNR. The network is optimized using Adam with a learning rate of $1 0 ^ { - 4 }$ and a batch size of 200 using the crossentropy multi-loss in [7].

We further investigate two complementary training strategies: high-SNR training and progressive training. For high-SNR training, samples are generated over 4–10 dB, while evaluation is performed over 1–8 dB or 1–6 dB, depending on the code. For progressive training, the network is initially trained with one decoding iteration. At each subsequent stage, a new iteration is randomly initialized, while previously trained ).iterations retain their learned weights. The entire network is then jointly trained for 200 epochs per stage until the target number of iterations is reached.

For all codes, the number of cascaded variable-node sublayers is fixed at $K = 3$ . GBPNet uses at most five decoding iterations, with L = 2 for BCH codes, $L \ = \ 3$ for Polar and LDPC codes, and $L ~ = ~ 5$ for the MacKay (96, 48) code. For comparison, conventional BP and Hyp BP [8] use five iterations, while the best reported results are used for the state-of-the-art transformer-based decoders ECCT [5] and CrossMPT [4]. As shown in Fig. 3, GBPNet outperforms BP even with regular training, demonstrating the effectiveness of its architecture independently of the proposed training strategies. High-SNR and progressive training yield further BER improvements. Fig. 4 additionally demonstrates GBPNet’s competitive decoding performance against recent methods on the MacKay (96, 48) code. It also shows the effect of the number of cascaded variable-node sublayers, with GBPNet results for $K \ = \ 1 , 2 , 3$ , where increasing K progressively improves the BER which is true for other codes as well.

TABLE I  
QUANTITATIVE COMPARISON OF GBPNET AT SNRS OF 4, 5, AND 6 DB. RESULTS ARE REPORTED AS − ln(BER) (LARGER IS BETTER)
<table><tr><td rowspan="2">Codes</td><td rowspan="2">Parameter</td><td colspan="3">BP</td><td colspan="3">Hyp BP [8]</td><td colspan="3">ECCT [5]</td><td colspan="3">CrossMPT [4]</td><td colspan="3">GBPNet</td></tr><tr><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td></tr><tr><td rowspan="2">BCH</td><td>(63,36)</td><td>4.03</td><td>5.42</td><td>7.26</td><td>4.29</td><td>5.91</td><td>8.01</td><td>4.86</td><td>6.65</td><td>9.10</td><td>5.03</td><td>6.91</td><td>9.37</td><td>6.10</td><td>7.87</td><td>10.21</td></tr><tr><td>(63,51)</td><td>4.50</td><td>5.82</td><td>7.42</td><td>4.80</td><td>6.44</td><td>8.58</td><td>5.66</td><td>7.89</td><td>11.01</td><td>5.78</td><td>8.08</td><td>11.41</td><td>6.42</td><td>8.78</td><td>12.03</td></tr><tr><td rowspan="3">Polar</td><td>(64,32)</td><td>4.26</td><td>5.38</td><td>6.50</td><td>4.59</td><td>6.10</td><td>7.69</td><td>6.99</td><td>9.44</td><td>12.32</td><td>7.50</td><td>9.97</td><td>13.31</td><td>8.24</td><td>10.42</td><td>13.85</td></tr><tr><td>(64,48)</td><td>4.74</td><td>5.94</td><td>7.42</td><td>4.92</td><td>6.44</td><td>8.39</td><td>6.36</td><td>8.46</td><td>11.09</td><td>6.51</td><td>8.70</td><td>11.31</td><td>7.43</td><td>9.36</td><td>11.94</td></tr><tr><td>(128,96)</td><td>4.61</td><td>5.79</td><td>7.08</td><td>4.94</td><td>6.76</td><td>9.09</td><td>6.31</td><td>9.12</td><td>12.47</td><td>7.15</td><td>10.15</td><td>13.13</td><td>7.96</td><td>10.97</td><td>13.89</td></tr><tr><td rowspan="2">LDPC</td><td>(121,60)</td><td>4.82</td><td>7.21</td><td>10.87</td><td>5.22</td><td>8.29</td><td>13.00</td><td>5.17</td><td>8.31</td><td>13.30</td><td>5.74</td><td>9.26</td><td>14.78</td><td>6.23</td><td>10.05</td><td>15.32</td></tr><tr><td>(121,70)</td><td>5.88</td><td>8.76</td><td>13.04</td><td>6.39</td><td>9.81</td><td>14.04</td><td>6.40</td><td>10.21</td><td>16.11</td><td>7.06</td><td>11.39</td><td>17.52</td><td>8.12</td><td>12.21</td><td>18.32</td></tr><tr><td>MacKay</td><td>(96,48)</td><td>6.84</td><td>9.40</td><td>12.57</td><td>7.19</td><td>10.02</td><td>13.16</td><td>7.38</td><td>10.72</td><td>14.83</td><td>7.97</td><td>11.77</td><td>15.52</td><td>9.02</td><td>13.11</td><td>17.03</td></tr></table>

![](images/56899c316d9cd8f38e0f3a5436fdc5e706994ec751a24529be2646f82320d7c3.jpg)  
Fig. 3. Decoding performance comparison for BCH(127, 64), demonstrating that GBPNet outperforms conventional BP even with regular training, highlighting the effectiveness of its architecture, while high-SNR and progressive training provide further improvements.

![](images/0f6b26054a74fd3d7cebee22922fb97c1e6a38d1ceaa17ab226447ba9f594c3c.jpg)  
Fig. 4. BER comparison of different state-of-the-art decoding methods on the MacKay (96,48) code. The figure also shows the effect of using multiple variable-node sublayers, with GBPNet results for $K = 1 , 2 , 3$

## VI. CONCLUSION

We proposed GBPNet, a generalized neural BP decoder incorporating cascaded variable-node sublayers and a learned scaling messages block. We further introduced high-SNR and progressive training strategies. Experiments across various codes demonstrate improved BER performance over existing BP-based decoders. As future work, we aim to employ neural architecture search to optimize the number of variable-node sublayers in each iteration and the total number of decoding iterations.

## REFERENCES

[1] J.-L. Wu, Y.-H. Tseng, and Y.-M. Huang, “Neural network decoders for linear block codes,” International Journal ofComputational Engineering Science, vol. 3, no. 03, pp. 235–255, 2002.

[2] T. Gruber, S. Cammerer, J. Hoydis, and S. Ten Brink, “On deep learningbased channel decoding,” in 2017 51st annual conference on information sciences and systems (CISS). IEEE, 2017, pp. 1–6.

[3] A. Bennatan, Y. Choukroun, and P. Kisilev, “Deep learning for decoding of linear codes-a syndrome-based approach,” in 2018 IEEE International Symposium on Information Theory (ISIT). IEEE, 2018, pp. 1595–1599.

[4] S. J. Park, H. Y. Kwak, S. H. Kim, Y. Kim, and J. S. No, “Crossmpt: Cross-attention message-passing transformer for error correcting codes,” in 13th International Conference on Learning Representations, ICLR, 2025.

[5] Y. Choukroun and L. Wolf, “Error correction code transformer,” Advances in Neural Information Processing Systems, vol. 35, pp. 38 695– 38 705, 2022.

[6] L. Qian, Y. Lu, X. Wang, W. Chen, and B. Ai, “Learning to decode: Graph attention network-gated recurrent unit as channel decoder,” IEEE Wireless Communications Letters, 2026.

[7] E. Nachmani, E. Marciano, L. Lugosch, W. J. Gross, D. Burshtein, and Y. Be’ery, “Deep learning methods for improved decoding of linear codes,” IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp. 119–131, 2018.

[8] E. Nachmani and L. Wolf, “Hyper-graph-network decoders for block codes,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[9] Y. Liang, C.-T. Lam, Q. Wu, B. K. Ng, and S. K. Im, “Low-complexity neural belief propagation decoding algorithm based on tensor ring decomposition,” IEEE Transactions on Cognitive Communications and Networking, vol. 11, no. 3, pp. 1563–1575, 2025.

[10] A. Tasdighi and M. Yousefi, “Adaptive learned belief propagation for decoding error-correcting codes,” Entropy, vol. 27, no. 8, p. 795, 2025.

[11] S. Malek, S. Salehkaleybar, and A. Amini, “Multi variable-layer neural networks for decoding linear codes,” in 2020 Iran Workshop on Communication and Information Theory (IWCIT), 2020, pp. 1–6.

[12] T. Richardson and R. Urbanke, Modern coding theory. Cambridge university press, 2008.

[13] M. Helmling, S. Scholl, F. Gensheimer, T. Dietz, K. Kraft, O. Griebel, S. Ruzika, and N. Wehn, “Database of Channel Codes and ML Simulation Results,” www.rptu.de/channel-codes, 2025.