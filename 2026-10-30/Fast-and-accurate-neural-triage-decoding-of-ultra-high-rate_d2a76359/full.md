# Fast and accurate neural triage decoding of ultra-high-rate quantum LDPC codes

Hong-Ye Hu,<sup>1,</sup> <sup>∗</sup> Zhenlai Wang,<sup>1</sup> Binzhao Luo,<sup>1</sup> Tao Xiang,<sup>1,</sup> <sup>2,</sup> <sup>∗</sup> and Chen Zhao<sup>3,</sup> <sup>∗</sup>

<sup>1</sup>Institute of Physics, Chinese Academy of Sciences, Beijing 100910, China

<sup>2</sup>School of Physical Sciences, University of Chinese Academy of Sciences, Beijing 100049, China

<sup>3</sup>School of Mathematical Sciences, Eastern Institute of Technology, Ningbo, Zhejiang 315200, China

Quantum error correction is essential for fault-tolerant quantum computing. High-rate quantum low-density parity-check (qLDPC) codes have emerged as a promising approach to reducing the substantial physical qubit overhead of quantum error correction, enabling encoding rates exceeding 1/2 in blocks of approximately one thousand physical data qubits. However, their practical implementation requires decoders that combine high accuracy with low latency to keep pace with quantum hardware. Existing decoders of high-rate qLDPC codes can achieve high accuracy but often incur substantial computational latency, whereas neural network decoders have yet to be demonstrated for high-rate qLDPC codes. Here, we introduce neural triage decoding, a framework that uses syndrome consistency to adaptively allocate neural computation. A local neural message-passing network rapidly predicts error corrections, additional iterations reduce residual syndrome violations, and a rarely invoked diversification stage searches for consistent corrections to the remaining shots. We demonstrate that the Triage decoder can decode ultra-high-rate qLDPC codes encoding a few hundred logical qubits, including lifted-product and Kasai codes, achieving state-of-the-art accuracy comparable to the near optimal algorithmic decoders while reducing decoding latency by approximately one to two orders of magnitude. Our framework is inspired by a rigorous analysis of the exact bitwise marginal decoder, revealing a distance-independen syndrome-inconsistent error floor. In addition, we show the robustness of neural network decoders to mismatched error priors. Our results establish adaptive neural decoding as a promising route towards fast, accurate, and scalable quantum error correction in the near term.

## I. INTRODUCTION

Quantum error correction (QEC), which protects quantum information through redundant encoding, is central to building a fault-tolerant quantum computer [1–3]. Recent experiments have demonstrated error suppression and key components of QEC on superconducting processors [4–6], neutral-atom platforms [7, 8], and trapped-ion systems [9–11], marking substantial progress towards fault-tolerant quantum computation. Scaling these advances to practical quantum algorithms [12– 16] requires sufficiently low logical error rates at a manageable physical-qubit cost [17–21]. Recent developments in high-rate quantum low-density parity-check (qLDPC) codes offer a route to reducing this cost by encoding many logical qubits within a single block while retaining sparse parity checks [22–30]. In particular, codes with encoding rates exceeding one half can accommodate hundreds of logical qubits in blocks of approximately one thousand physical data qubits, substantially reducing the encoding overhead [30–35]. Realizing this advantage also requires a fast and accurate classical decoder that infers corrections to logical observables from measured syndromes. Decoding accuracy determines how effectively the code suppresses logical errors, while sustained throughput must keep pace with syndrome generation and latency must meet the timing requirements of logical feedback [6, 36, 37].

Exact decoding is computationally challenging, and finding an optimal correction is NP-hard even for general classical codes [38]. Quantum degeneracy adds a further computational challenge, making optimal decoding of stabilizer codes #P-complete [39]. Practical decoders must therefore balance accuracy, speed, and versatility. Belief propagation (BP) is a natural choice for qLDPC codes because it exploits their sparse parity checks, supports parallel computation, and achieves near-optimal performance for classical LDPC codes [40–42]. In quantum codes, however, degeneracy and short cycles can prevent BP from converging, leaving corrections that do not reproduce the measured syndrome [43– 45]. These difficulties have motivated BP variants such as relay-BP [46], as well as post-processing methods that supplement BP with ordered statistics decoding (OSD) [47–49], localized statistics decoding (LSD) [50], or search-based correction [31, 51–53]. Such approaches can substantially improve accuracy and, in some settings, achieve near-optimal decoding performance [31]. Their computational cost, particularly for most-likely-error (MLE) search, nevertheless remains a barrier to the high throughput required for real-time decoding.

Neural networks offer another route to the versatile, accurate and fast decoder envisioned in Figure 1(a) [54–56]. They can learn directly from experimental data and adapt to devicespecific noise [57–59]. Autoregressive and transformer models can capture global correlations and approach optimal decoding, but often at substantial computational cost [58, 60, 61]. On the other hand, architectures such as Cascade neural decoders exploit code symmetries to achieve fast and accurate decoding for specific code families [62] and the performance for high-rate qLDPC codes is unclear. Graph neural networks (GNNs) offer broader applicability by using Tanner graphs as a common representation while retaining the sparse, parallel message passing of BP [63–66]. However, like BP, they can fail to produce syndrome-consistent corrections [64]; whether they can match the strongest algorithmic decoders across different code families remains unclear. Although neural qLDPC decoders have been demonstrated on blocks encoding tens of

(d)

![](images/66e38ea301144af2c1ef0e4f52114be8df70f861d5812c9862d4692a8ffaa230.jpg)

![](images/a99498de200e8d51e9c8dd03f1276c1d24018dd026cb38941a37184faba331f8.jpg)

![](images/bf9e3dbbe6972673a070e72c3dcfc086eed68866b1fd22703cf9077b23d8a2d5.jpg)  
Figure 1. Neural triage decoding for high-rate quantum LDPC codes. (a) The Triage decoder combines a fast NeuralMP local decoder with iterative decoding strategy. For rare cases when proposed errors cannot resolve all syndrome, it enters diversification stage. The Triage decoder allocate computation resources adaptively to achieve high accuracy, high speed and being versatile simultaneously. (b) NeuralMP performs R internal MP-rounds per evaluation. Then iterative decoding repeatedly applies the network to the residual syndrome for up to K outer iterations. Remaining shots enter diversification stage, which uses prior perturbations, symmetric permutations and syndrome flips to explore alternative correction trajectories with the same trained network. (c) High-rate qLDPC codes are represented by data qubits (red circles), X checks (blue squares) and Z checks (green diamonds). Under the phenomenological error model, data qubits experience Pauli X, Y or Z errors each per measurement round for multiple measurement rounds, while measurement outcomes are flipped with probability $p _ { \mathrm { m } }$ (d) Schematic comparison of decoding latency and logical error rate for neural and algorithmic decoders. Logical error rates are expressed on a per-qubit-per-round basis; the arrow indicates the desired direction of improvement.

logical qubits [65, 66], their performance on ultra-high-rate codes encoding hundreds of logical qubits per block remains unestablished. This regime demands accurate inference over a vast space of logical outcomes: a block encoding 500 logical qubits has $2 ^ { 5 0 0 } \approx 1 0 ^ { 1 5 0 }$ logical classes in a single code block.

In this work, we introduce Neural Triage Decoding (Triage), a versatile, fast, and accurate neural decoder that extends the idea of a local decoder through iterative decoding and adaptive search for syndrome-consistent corrections (see Figure 1 (a)). Its core graph-based message-passing decoder (NeuralMP) predicts physical corrections through local updates on the sparse Tanner graph of a detector error model (DEM). Within each message-passing round (MP-round), updates to nodes of the same type can be evaluated in parallel, making neural inference well suited to GPU execution. Then it outputs a marginal likelihood of error on each error node. We also show the limitations of bitwise marginal decoding and propose iterative decoding through virtual updates on the syndromes, shown as the second step in Figure 1 (b). If a syndrome compatible solution is still not reached, we invoke the last stage of diversification, which combines syndrome flips [67], perturbation of the prior distribution [44], and symmetric permutation [68]. The diversification reuses the same trained neural network as a general strategy to resolve syndrome-error compatibility without retraining.

We evaluate Triage’s performance on the high-rate liftedproduct (LP) code [[1122, 148, $d ~ \leq ~ 2 0 ] ] ~ ( \mathrm { L P } \bar { 2 } 0 )$ [21] and ultra-high-rate Kasai code $[ [ 1 1 5 2 , 5 8 0 , \bar { d } \leq 1 2 ] ]$ [30, 31] with phenomenological error models across r noisy measurement rounds, with a final perfect syndrome extraction.

These two codes are representative of product-based and nonproduct-based constructions of high-rate qLDPC codes, respectively. While being versatile, Triage achieves state-ofthe-art (SOTA) decoding accuracy that is comparable to or better than that of algorithmic decoders, such as BP-OSD and BP-relay-MLE (Tiered) [31], achieving a per-qubit-per-round error rate of around $5 \times 1 0 ^ { - 1 0 }$ at $p = 1 . 5 \%$ physical error rate on the Kasai code (Figure 1d). Alongside this accuracy, Triage reduces decoding latency and increases throughput by factors of 10-100 relative to algorithmic decoders (Figure 1d). Triage thus brings versatility, accuracy, and speed together in a common framework for decoding ultra-highrate qLDPC codes. With further hardware optimization, this framework offers a path towards real-time error correction of hundreds of logical qubits within a single code block.

## II. RESULTS

## Local marginal decoder and iterative decoding

We formulate decoding using a detector error model with F fault mechanisms and m detectors. A fault configuration ⃗e produces a syndrome ${ \vec { s } } = H { \vec { e } }$ and logical-observable flips $\ell = L \vec { e } ,$ with arithmetic over GF(2). NeuralMP predicts one score per fault mechanism by passing messages on the sparse bipartite graph defined by H [65], which is also called the Tanner graph. Syndrome values $s _ { v }$ and mechanism priors $p _ { f }$ are encoded as χ-dimensional detector and fault states, shown as colored bars in Figure $2 ( \mathrm { a } )$ , with learned coordinate embeddings supplying positional information. Each of R MP-rounds first updates fault states using detector states, then updates detector states using the updated fault states (Figure 2(b)). Trainable functions $F _ { \theta } ^ { \bar { v } \to f }$ and $F _ { \theta ^ { \prime } } ^ { f  v }$ combine each node’s current state with the sum of its incoming messages. Parameters are shared across nodes of each type within an MP-round, enabling parallel updates, but differ between MP-rounds. A linear readout of the final fault states gives logits $z _ { f }$ and learned fault scores $q _ { f } = \sigma ( z _ { f } )$ , where $\bar { \sigma ( } x ) = \bar { ( 1 + e ^ { - x } ) ^ { - 1 } }$ is the sigmoid function. Crucially, for $N = m + F$ nodes and E edges, one evaluation costs $\mathcal { O } ( R [ E \chi + N \chi ^ { 2 } ] )$ , which is linear in N when $E = { \mathcal { O } } ( N )$ at fixed $R , \chi$ In contrast, the global decoders that use attention mechanism will have costs that scale as $\mathcal { O } ( N ^ { 2 } )$ [69].

![](images/b2f66a603822cff49da16fad77bdb121df3e051b98a150d8e7717780bf0767ca.jpg)  
Figure 2. NeuralMP architecture, failure-aware training and residual-syndrome scaling. (a) The Tanner graph of the DEM connects fault mechanisms (red circles) to detectors (blue squares) through $H _ { v f } = 1$ . Red colored bars represent node states encoding mechanism priors together with positions embedding and blue colored bars represent detector outcomes embedding. (b) Each MP-round first updates fault states using neighbouring detector states, then updates detector states using the updated fault states. Aggregated messages enter residual multilayer perceptrons (MLPs) with layer normalization. After R MP-rounds, a linear readout followed by a sigmoid produces one marginal estimate per mechanism. (c) Training combines fresh online simulations, including error-weight-conditioned samples, with symmetry-augmented replay from an evolving mistake bank. Logical failures enter the bank, while repeatedly corrected examples are removed. The joint objective supervises fault predictions, logical observables and syndrome consistency. (d) Probability of a nonzero residual syndrome after a single pass of NeuralMP versus physical error rate p for the LP20 and Kasai codes. Orange dashed lines indicate $p ^ { 4 } = p ^ { w _ { \mathrm { m i n } } / 2 }$ reference scaling with $w _ { \mathrm { m i n } }$ as minimal weight of space-time degeneracy. Pale blue and pale red regions highlight regimes dominated by low-weight and high-weight errors, respectively. Error bars show 95% confidence intervals.

During training, the Triage decoder readily learns to correct common error patterns, but rare and difficult decoding cases are the important ones determining its performance in the low error rate regime. To enhance the sparse training reward, we couple online error-weight-conditioned simulation with a dynamically updated mistake bank (Figure 2(c)). We store hard decoding examples in the online mistake bank and remove them from the bank after repeated successes. We also augment replayed examples using code symmetries, jointly permuting fault patterns and syndromes. This failure-aware loop combines targeted data generation with the supervision of fault predictions, logical observables, and syndrome consistency. Implementation details are provided in the Methods.

Independent decisions from marginal probabilities can violate syndrome constraints even when the marginals are exact. To illustrate the ambiguity, consider independent $Z$ noise on n qubits of a Calderbank–Shor–Steane (CSS) code and an evenweight $Z$ stabilizer g of weight w. Errors $e _ { A }$ and $e _ { B } = e _ { A } \oplus g$ occupying complementary halves of its support both have weight $w / 2 .$ . Since $H g \ = \ 0$ and $L g ~ = ~ 0$ , they share the same syndrome and logical labels. Either specified error, with no other errors, occurs with probability $p ^ { w / 2 } ( 1 - p ) ^ { n - w / 2 } =$ $\Theta ( p ^ { w / 2 } )$ . If this pair gives the only minimum-weight explanations of its syndrome, each supported qubit has an exact posterior marginal approaching $1 / 2$ as $p  0$ . Such local ambiguities can arise from stabilizers whose weight does not grow with code distance. In phenomenological noise models, space–time degeneracies involving both data and measurement errors can have lower weight than purely spatial stabilizers.

Theorem 1 (Syndrome inconsistency under phenomenological noise). Consider a CSS code with Z-check matrix $H _ { Z }$ of minimum qubit degree $\nu \geq 2 ,$ with no 4-cycles in its Tanner graph and no nonzero kernel vector of weight below $2 \nu + 2 .$ In the independent $X / Z$ phenomenological model defined in Methods, with $r \geq 5$ noisy measurement rounds, data error rate p, and measurement error rate $q = \Theta ( p )$ , the exact bitwise marginal decoder, using maximum-a-posteriori (MAP)

![](images/2e60cdd5b7869e497d597394237619dd5d2b4a80cdc6d7d567e765a97b1ca260.jpg)

decisions, satisfies

$$
\begin{array} { r l } & { P _ { \mathrm { i n c } } ( p ) : = \mathrm { P r } ( H \hat { e } \neq H e ) = \Omega ( p ^ { \kappa ( \nu ) } ) \qquad ( p \to 0 ) , } \\ & { \qquad \quad \kappa ( \nu ) = \left\{ \begin{array} { l l } { \nu / 2 + 1 , } & { \nu e \nu e n , } \\ { \nu + 1 , } & { \nu o d d . } \end{array} \right. } \end{array}\tag{1}
$$

The LP and Kasai code shots used here have $\nu = 3 ,$ so Theorem 1 yields $P _ { \mathrm { i n c } } ( p ) = \Omega ( p ^ { 4 } )$ for their independent $X / Z$ phenomenological models. Single-pass NeuralMP results are consistent with this mechanism (Figure 2(d)). The proof and finite-instance certificates are given in the Methods and the Supplementary Information.

Motivated by this understanding, we propose the idea of $i t \mathrm { - }$ erative decoding, which addresses residual violations by repeatedly applying the same trained network to the current residual syndrome (Figure 1(b)). Starting from $\hat { e } ^ { ( 0 ) } = 0$ and $\mathbf { \Delta } _ { s } \mathbf { \stackrel { ( 0 ) } {  \mathbf { \sigma } } } = s$ , each outer iteration k samples a proposal $\delta e ^ { ( k ) }$ and updates

$$
\begin{array} { r l } & { \hat { e } ^ { ( k + 1 ) } = \hat { e } ^ { ( k ) } \oplus \delta e ^ { ( k ) } , } \\ & { s _ { \mathrm { r e s } } ^ { ( k + 1 ) } = s _ { \mathrm { r e s } } ^ { ( k ) } \oplus H \delta e ^ { ( k ) } . } \end{array}\tag{2}
$$

Decoding stops when the residual vanishes or reaches the maximum $K _ { \mathrm { m a x } }$ iterations. Practically, we set $K _ { \operatorname* { m a x } } ~ = ~ 4$ Shots unresolved after $K _ { \mathrm { m a x } }$ outer iterations proceed to the final stage of diversification.

## Improved convergence with diversification

Residual-guided iteration can be complemented by diversification, which explores alternative correction trajectories under different inference conditions. The relay-BP decoder follows the same principle by varying local memory strengths to diversify BP trajectories [46]. For a fixed neural decoder, we generate alternative correction candidates through prior perturbations, symmetric permutations, and syndrome flips, using information from the decoder to guide the search (see Figure 1 (b)). In particular, near-zero logit outputs identify uncertain error mechanisms, while repeated corrections at the same locations indicate instability along an iterative trajectory.

Each diversification trajectory starts from the original syndrome and reuses the iterative decoder under modified inference conditions. Prior perturbations modify the mechanism priors supplied to the network, shifting its preferences among competing correction proposals. Symmetric permutations preserve the Tanner graph of the DEM and mechanism priors but change the alignment of the syndrome with the network’s coordinate encodings, allowing different predictions that are mapped back to the original coordinates. For syndrome flips, we select a small set of mechanisms, represented by a binary vector $\vec { t , }$ using marginal uncertainty and repeated corrections as guides. We apply the iterative decoder to the modified syndrome ⃗s ⊕ H<sup>⃗</sup>t with the original priors, then combine its prediction $\vec { e ^ { \prime } }$ with $\vec { t }$ to obtain the candidate ${ \hat { e } } = { \vec { t } } \oplus { \vec { e ^ { \prime } } }$ The resulting candidates are checked against the original syndrome, and syndrome-consistent candidates

Figure 3. Decoding accuracy and amortized latency on LP20 and Kasai codes. (a) Logical error rate (LER) per-qubit-per-round versus physical error rate $p ,$ using approximately 0.04–40.96 million shots per point for LP20 code and 0.24–20.48 million for Kasai code. Error bars quantify finite-sampling uncertainty using 95% Clopper– Pearson intervals. (b) Amortized decoding latency. We benchmark the latency using NVIDIA H200 GPU. NeuralMP (Triage’s frontend) reports the asymptotic frontend cost, whereas Triage shows the composed full-decoder cost at batch size 64, including handoff and diversification. CPU baseline timing uses 1,000–5,000 shots per point. Timing error bars, where shown, are 95% bootstrap confidence intervals from 2,000 resamples and indicate uncertainty in the estimated mean latency. CPU baselines use one thread, except the eight-thread COPT stage [70].

are ranked by their negative log-likelihood under the original mechanism priors.

The effect of a syndrome flip can be seen in the half-support issue described above, where two complementary errors ⃗a and $\vec { b }$ each have weight $w / 2$ and produce the same syndrome. Choosing $\vec { t }$ to contain a single qubit in the support of ⃗a maps these errors to ⃗a⊕<sup>⃗</sup>t and ${ \vec { b } } \oplus { \vec { t } } ,$ both consistent with the modified syndrome but with weights $w / 2 - 1$ and $w / 2 + 1$ , respectively. Under the unchanged independent-error prior with $p < 1 / 2$ the former is more likely by a factor $[ { \bar { ( 1 - p ) / p } } ] ^ { 2 } .$ In the two-candidate example, this imbalance moves the marginals away from $1 / 2$ and favors a consistent choice, which is then mapped back to a correction for the original syndrome. The flip thus changes the preference between candidates during search without changing their relative likelihood in the original decoding problem.

We organize these diversification strategies into a conditional decoding procedure that allocates additional computation to shots with unresolved syndromes. Shots whose residual syndrome vanishes during the initial iterative stage return immediately, and the remainder enter a syndrome-flip stage guided by marginal uncertainty and repeated corrections. If it produces syndrome-consistent candidates, we return the best candidate according to the selection rule above. Shots with no such candidate can proceed to a supplementary search stage which has further sign flip strategy and symmetric permutations combined with prior perturbations. Predictions from different symmetry-related views are mapped back to the original coordinates, and their disagreement identifies locations for additional syndrome flips. Candidates from this stage are evaluated by the same rule. If the search budget is exhausted without a syndrome-consistent candidate, the decoder retains the correction returned by the initial iterative stage, and the shot remains unresolved.

## Decoding high-rate qLDPC codes

We evaluate Triage’s performance on the lifted-product code [[1122, $1 4 8 , \leq 2 0 ] ]$ and the ultra-high-rate Kasai code [[1152, 580, $d \leq 1 2 ] ]$ , spanning encoding rates of 13.2% and 50.3%, respectively. Both benchmarks use phenomenological noise, with data errors and multiple noisy syndrome measurements followed by a final perfect syndrome extraction. For the LP20 code, we simulate five noisy measurement rounds, with independent X and $Z$ errors, each occurring with probability $p$ on every data qubit, and measurement outcomes flipped with probability $p .$ For the Kasai code, we simulate eight noisy measurement rounds under depolarizing noise, with $X , Y$ , and $Z$ errors, each occurring with a probability of $p / 3$ , and measurement outcomes flipped with a probability of $4 p / 3$ . For both codes, diversification begins with syndrome flips. For LP20, unresolved shots proceed to an additional stage combining prior perturbations, symmetric permutations, and further syndrome flips guided by disagreement between symmetry-related predictions; Kasai uses only the initial syndrome-flip stage. Crucially, for both LP20 and Kasai codes, we train Triage at a single physical error rate, $p = 2 \%$ , and evaluate its performance across a range of previously unseen error rates.

We observe that Triage decoder achieves low logical error rates while substantially reducing amortized decoding costs. For example, at $p = 0 . 0 1 5$ , its estimated per-qubit-perround LER is $9 . { \overset { - } { 3 } } 3 \times 1 { \overset { - } { 0 } } ^ { - 1 1 }$ on LP20 and $5 . 0 8 \times \mathrm { \bar { 1 0 } ^ { - 1 0 } }$ on Kasai, compared with $4 . 4 3 \times 1 0 ^ { - 1 0 }$ for BP-OSD and $1 . 7 6 \times 1 0 ^ { - 9 }$ for the BP-relay-MLE Tiered decoder, respectively (see Figure 3 (a)). These accuracies are obtained through iterative NeuralMP decoding with diversification. At the same error rate, the estimated full-decoder costs at batch size 64 are 2.37 ms and 12.3 ms per shot on a single NVIDIA H-series GPU, compared with CPU baseline means of 815 ms and 349 ms for BP-OSD and the Tiered decoder, respectively (see Figure 3 (b)). These benchmark settings yield approximately 343× and 28× speedups.

We further examine its robustness to inaccurate noise calibration by perturbing the priors supplied to the decoder while keeping the physical noise fixed. This is important as the noise distribution can shift from the initial noise learning, and control drifts can also happen during the computation [71, 72]. We perturb the prior error uniform distribution as $p _ { f }  p _ { f } ( 1 + \sigma _ { r } { \mathcal { N } } ( 0 , 1 ) )$ , where $\mathcal { N } ( 0 , 1 )$ is the Gaussian distribution with zero mean and unit variance. We investigate the ratio between the whole block logical error rate with the perturbed prior distribution over that of the original prior distribution. In Figure 4, we show that the four-iteration Triage frontend (NeuralMP) maintains a block-LER ratio close to unity across the tested perturbations. At the largest nominal mismatch strength, $\sigma _ { r } ~ = ~ 0 . 5$ , its ratios are approximately 1.01 for the LP20 code and 0.98 for the Kasai code, whereas the corresponding algorithmic baselines show increases of approximately 1.6–2.1 times relative to matched-prior decoding. These results suggest that neural-network-based decoders can offer passive resilience to uncertainty and fluctuations in noise parameters, analogous to robust quantum controls [73–75].

![](images/af170ef8fe0771f1d72c2a18b13262759f59d1de5e76f493f2a494ab3d28bafd.jpg)

![](images/46e65cfed878db2a8f513064ca96f9fabcf7c04d1e51eb0c9690ed6caec0f7b7.jpg)  
Figure 4. Robustness to mismatched error priors. Block-LER ratio relative to correctly specified priors versus nominal mismatch strength $\sigma _ { r }$ for (a) LP20 code at $p = 0 . 0 2 4$ , and (b) Kasai code at $p \ = \ 0 . 0 1 7 5$ NeuralMP uses four decoding iterations. Each prior realization is evaluated on $9 \times 1 0 ^ { 5 }$ shots for LP20 code and $\mathsf { \bar { 6 } } \times \mathsf { 1 0 } ^ { 5 }$ shots for Kasai code, sharing the same noise samples with the matched-prior baseline. Two prior realizations are used per $\sigma _ { r } .$ except LP20 at $\sigma _ { r } = 0 . 1$ 1 and 0.3, which use four. Markers show geometric means of the per-realization LER ratios. Error bars span the per-realization ratio intervals obtained by transforming conditional 95% Clopper–Pearson intervals.

## III. DISCUSSION

Triage decoder combines a local neural message-passing network with iterative decoding and a rarely invoked diversification to allocate computation according to decoding difficulty. This architecture is inspired by the theoretical understanding of the limitations of the exact bitwise marginal decoder, where there exists a lower bound on the residual syndrome density even when the marginal distribution is exact. The results on lifted-product and Kasai codes show that Triage can achieve SOTA decoding accuracy comparable to and better than that of the algorithmic decoders, with substantially lower amortized decoding costs than the tested algorithmic baselines. In addition, we show the robustness of the Triage decoder against the error prior distribution, which is important and relevant for real-time decoding with hardware fluctuations. These results also suggest several directions for improving the decoder and connecting it to experimental operation.

Symmetric permutations currently enter Triage through diversification. Incorporating these symmetries directly into NeuralMP, for example, through symmetry-respecting parameter sharing as suggested in Cascade [62], could reduce network size and inference cost. The different effectiveness of diversification proposals across the two codes also motivates a more systematic analysis of decoding failures. Relating the response to syndrome flips, prior perturbations, and symmetric permutations to graph structure and degeneracy could reveal which ambiguities each proposal resolves and guide the choice of proposal and computational budget for each shot.

These diagnostics could support recursive selfimprovement (RSI) of Triage. Automated discovery of programs and algorithms with language models [76, 77], together with the discovery of learning rules through meta-learning [78], provides precedents for using machinegenerated proposals and quantitative feedback to improve computational methods. In this setting, AI agents could analyze decoding failures, propose changes to the architecture, training procedure, or diversification strategy, and evaluate successive revisions against logical error rates and total decoding costs. This feedback loop could reduce manual tuning while retaining explicit accuracy and runtime criteria [79].

Finally, adaptive decoding and hardware self-calibration could form a coupled feedback loop. Neural decoders can be adapted to experimental data [58], while recent experimental and theoretical work shows how syndrome-derived signals can guide control optimization during error correction [71, 72]. Triage could update mechanism priors or selectively retrain NeuralMP as the noise changes, while detection-event statistics guide the recalibration of the underlying controls. The observed resilience of the four-iteration frontend to the tested prior perturbations provides motivation for studying this integration. Establishing its ability to track physical drift would require evaluating the full decoder and calibration loop together, including the cost and stability of online updates.

Extending Triage from phenomenological quantum memory to circuit-level logical gadgets and algorithms [7, 53, 80–84], or applying it to experimental data, are central next steps. Real-time operation also requires measuring single-shot latency, its distribution, and sustained throughput against the timing requirements of the hardware [6, 56]. Further work is needed to optimize GPU inference and investigate fieldprogrammable gate array (FPGA) deployment of the trained model [85, 86]. The present GPU and CPU comparisons quantify amortized costs under their stated batching and hardware conditions. These further tests would determine whether adaptive neural decoding can sustain both accurate logical protection and efficient feedback during prolonged operation.

## IV. METHODS

## A. NeuralMP architecture

The detector error model specifies F mechanisms, m detectors, and k logical observables through $H \in \{ 0 , 1 \} ^ { m \times F }$ $L \in \{ 0 , 1 \} ^ { k \times F }$ , and priors $p _ { f }$ , with $N = m + F$ nodes in a bipartite Tanner graph. Error configurations differing by a vector in ker H ∩ ker L share both syndrome and logical labels. Initial states combine learned node-type and coordinate features; fault states additionally receive a linear projection of log $p _ { f } .$ , and detector states receive the encoded binary outcomes. Coordinate features concatenate learned embeddings for individual nodes, followed by a linear projection and a rectified linear unit (ReLU). Messages include a learned sourcetype bias before summation, and detector aggregation uses the newly updated fault states. Each directional update takes the current state and aggregate as the input to a two-layer MLP with a ReLU hidden activation. Its output is added to the current state and layer-normalized. Distinct update modules are used for the two directions and for successive MP-rounds, with each module applied in parallel across its node family. The deployed Kasai network uses hidden dimension $\chi = 6 4$ and R = 80 MP-rounds with distinct MP-round-specific parameters.

Fresh training examples mix natural noise with noise conditioned on selected fault weights. Thresholded predictions with incorrect logical observables enter the replay bank. Verified symmetries transform replayed fault patterns and syndromes together, with logical labels recomputed. The three loss terms are defined in the Training objective subsection. Class weighting and modified sampling distributions mean that the scores are not assumed to be calibrated deployment posteriors.

Each outer iteration is an independent forward pass: hidden states are re-initialized from the current residual syndrome and the original priors, and no hidden state or logit is carried over between iterations. Information propagates between iterations only through the residual syndrome. Proposal bits are sampled independently as $\delta e _ { f } ^ { ( k ) } \sim$ Bernoulli[σ $\cdot ( z _ { f } ^ { ( k ) } / T ) ]$ and accumulated according to Equation (2). Here $T ^ { \ ' } > 0$ is the sampling temperature. The deployed configuration uses $T = 1$ , so $q _ { f } ^ { ( k ) } = \sigma ( \boldsymbol { z } _ { f } ^ { ( k ) } )$ , and a budget of at most $K = 4$ network evaluations per shot before escalation to diversification. Parameters remain fixed throughout inference.

## B. Training objective

We train NeuralMP using physical-error-level, logical, and syndrome supervision. Let $q _ { f } = \sigma ( z _ { f } )$ denote the network output for fault mechanism $f ,$ and let $e , s = H e$ and $\ell =$ $L e$ denote the simulated fault pattern, syndrome and logicalobservable flips, respectively. All parity operations are over GF(2). The total loss is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { f a u l t } } + \mathcal { L } _ { \mathrm { l o g i c a l } } + \mathcal { L } _ { \mathrm { s y n d r o m e } } ,\tag{3}
$$

with unit coefficients for all three terms.

The fault loss is binary cross-entropy (BCE), with a positive-class weight of eight to compensate for sparse faults:

$$
\mathcal { L } _ { \mathrm { f a u l t } } = - \frac { 1 } { F } \sum _ { f } \left[ 8 e _ { f } \log q _ { f } + ( 1 - e _ { f } ) \log ( 1 - q _ { f } ) \right] .\tag{4}
$$

For logical supervision, we construct a differentiable parity estimate under a factorized Bernoulli approximation,

$$
\begin{array} { r } { \widetilde { \ell } _ { j } = \frac { 1 - \prod _ { f : L _ { j f } = 1 } ( 1 - 2 q _ { f } ) } { 2 } , } \\ { \mathcal { L } _ { \mathrm { l o g i c a l } } = \displaystyle \frac { 1 } { k } \sum _ { j } \mathrm { B C E } ( \widetilde { \ell } _ { j } , \ell _ { j } ) } \end{array}\tag{5}
$$

The syndrome-consistency penalty similarly compares predicted fault parities with the measured detectors:

$$
{ \mathcal { L } } _ { \mathrm { s y n d r o m e } } = { \frac { 1 } { m } } \sum _ { c } \left[ 1 - ( - 1 ) ^ { s _ { c } } \prod _ { f : H _ { c f } = 1 } ( 1 - 2 q _ { f } ) \right] .\tag{6}
$$

Here, $F , k ,$ and m are the numbers of fault mechanisms, logical observables, and detectors. These terms encourage accurate fault predictions, correct logical outcomes and syndrome consistency, respectively; exact consistency is checked separately during decoding.

## C. Marginal decoding and syndrome inconsistency

We first establish a criterion for marginal decisions to violate a syndrome constraint. We then derive this criterion from the spatial graph and phenomenological noise assumptions of Theorem 1, and verify those assumptions for the simulated shots. Throughout this subsection, $H \in \mathbb { F } _ { 2 } ^ { m \times F }$ is fixed and finite, and the error mechanisms are independent with probabilities $p _ { f } ( p ) \ = \ \Theta ( p )$ as $p  0$ The exact posterior marginals are $\pi _ { f } ( s ) \ : = \ : \mathrm { P r } ( e _ { f } = 1 \ : | \ : H e = s )$ Bitwise MAP sets $\hat { e } _ { f } ( s ) ~ = ~ \mathrm { \bf \bar { 1 } } \{ \pi _ { f } ( s ) ~ > ~ 1 / 2 \}$ , assigning ties to zero. All matrix operations are over $\mathbb { F } _ { 2 }$ . We write $P _ { \mathrm { i n c } } ( p ) = \mathrm { P r } ( H \hat { e } \neq H e )$

Lemma 1 (Concentration on minimum-weight explanations). For a nonzero attainable syndrome $s _ { * } ,$ , let κ be its minimum compatible error weight, $\mathcal { M } ,$ <sub>∗</sub> the set ofall compatible weightκ errors, and U<sub>∗</sub> the union oftheir supports. Then

$$
\begin{array} { c } { { Z _ { * } : = \mathrm { P r } ( H e = s _ { * } ) = \Theta ( p ^ { \kappa } ) , } } \\ { { \mathrm { P r } ( e \notin { \mathcal { M } } _ { * } \mid H e = s _ { * } ) = O ( p ) . } } \end{array}\tag{7}
$$

In particular, $\pi _ { f } ( s _ { * } ) = O ( p ) f o r e \nu e r y f \notin U _ { * }$

Proof. Independence and $p _ { f } ( p ) = \Theta ( p )$ imply that each fixed weight-t error has probability $\Theta ( p ^ { t } )$ . Since H is finite, the total probability of the minimum-weight explanations is $\Theta ( p ^ { \kappa } )$ while that of all higher-weight compatible errors is $O ( p ^ { \kappa + 1 } )$ Dividing by $Z _ { * }$ proves posterior concentration. An error on a coordinate outside $U _ { * }$ can occur only in a higher-weight explanation, giving the marginal bound. □

For an equally likely disjoint pair $\mathcal { M } _ { * } = \{ a , b \}$ , this lemma gives $\pi _ { f } ( s _ { * } ) = 1 / 2 + O ( p )$ on $U _ { * }$ . The higher-weight explanations determine the signs of these small shifts, which control the hard decisions.

Lemma 2 (Two-bit marginal shifts). Fix a syndrome $s _ { * }$ with $Z _ { * } \ > \ 0$ and distinct bits $i , j .$ . Define $W _ { u v } ( p ) \ = \ $ $\begin{array} { r } { \operatorname* { P r } ( H e = s _ { * } , e _ { i } = u , e _ { j } = v ) f o r u , v \in \{ 0 , 1 \} , \Delta = W _ { 0 0 } - } \end{array}$ $W _ { 1 1 }$ , and $\delta = W _ { 1 0 } - W _ { 0 1 }$ . Then

$$
\begin{array} { l } { \displaystyle \pi _ { i } ( s _ { * } ) - \frac 1 2 = \frac { - \Delta + \delta } { 2 Z _ { * } } , } \\ { \displaystyle \pi _ { j } ( s _ { * } ) - \frac 1 2 = \frac { - \Delta - \delta } { 2 Z _ { * } } . } \end{array}\tag{8}
$$

The two marginals lie strictly on the same side of $1 / 2 \ i f$ and only $i f \left| \Delta \right| > \left| \delta \right|$ . In particular, $\Delta > | \delta |$ places both below $1 / 2 .$

Proof. Use $\pi _ { i } = ( W _ { 1 0 } + W _ { 1 1 } ) / Z _ { * } , \pi _ { j } = ( W _ { 0 1 } + W _ { 1 1 } ) / Z _ { * } . \nonumber$ and $\begin{array} { r } { \dot { Z } _ { * } = \sum _ { u , v } \dot { W } _ { u v } . } \end{array}$ . Subtracting $1 / 2$ gives the identities. Their product is $( \Delta ^ { 2 } - \delta ^ { 2 } ) / ( 4 Z _ { * } ^ { 2 } )$ , which is positive exactly when the shifts have the same nonzero sign. □

Proposition 1 (A two-bit inconsistency criterion). Under the noise assumptions of this subsection, suppose a nonzero syndrome $s _ { * }$ has exactly two minimum-weight explanations $a , b$ ofweight κ with disjoint supports, and set $U = { \mathrm { s u p p } } ( a \oplus b )$ Suppose every compatible error supported on U obeys $e _ { i }$ ⊕ $e _ { j } = 1$ for fixed distinct bits $i , j \in \bar { U } . \ I f \left| \Delta \right| > \left| \delta \right.$ | for these bits at all sufficiently small $p > 0 ,$ , then exact bitwise MAP has $P _ { \mathrm { i n c } } ( p ) = \Omega ( p ^ { \kappa } )$

Proof. By Lemma 1, $Z _ { * } = \Theta ( p ^ { \kappa } )$ and all marginals outside $U$ are $O ( p )$ . Hence the MAP output is supported on $U$ for sufficiently small $p .$ Lemma 2 gives equal MAP decisions on $i , j ,$ violating the required odd parity. Thus MAP is inconsistent whenever $s _ { * }$ is observed, and $\operatorname { \dot { \cal P } } _ { \mathrm { i n c } } ( p ) \ge Z _ { * } = \Theta ( p ^ { \kappa } )$ ， independently of ties at other coordinates. □

The quantities $\Delta$ and δ separate a common shift of the two endpoint marginals from a preference between them: $\pi _ { i } +$ $\pi _ { j } = 1 - \Delta / Z _ { * }$ and $\pi _ { i } - \pi _ { j } = \delta / Z _ { * }$ . The phenomenological construction below gives these two effects a concrete space– time interpretation.

Theorem 1 assumes mutually independent $X / Z$ data faults (probability $p$ each) and measurement flips $( q = \Theta ( p ) )$ , with known initial syndromes and a perfect final readout after $r$ noisy measurement rounds. The code and r are fixed as $p $ 0.

Choose a degree-ν qubit $v _ { * }$ . A data fault on this qubit can be moved by τ measurement rounds without changing the syndrome if a string of measurement faults joins the two times on each neighboring check. Writing $\vec { d } _ { t }$ for its X fault in measurement round t and $\vec { m } _ { h , }$ <sub>t</sub> for a measurement fault connecting detectors in measurement rounds $t , t + 1$ on check $h ,$ this relation is

$$
\vec { g } = \vec { d } _ { 0 } + \vec { d } _ { \tau } + \sum _ { h \in N ( v _ { * } ) } \sum _ { t = 0 } ^ { \tau - 1 } \vec { m } _ { h , t } , \qquad | \vec { g } | = 2 + \tau \nu .
$$

Here $H \vec { g } = L \vec { g } = 0 .$ . We split $U = \operatorname { s u p p } ( \vec { g } )$ into two complementary explanations $a , b ,$ each containing one data endpoint and half of the measurement faults. Taking $\tau = 1$ for even ν and $\tau = 2$ for odd $\nu$ gives weight $\kappa ( \nu ) = 1 + \tau \nu / 2$ . The endpoint bits are $i = d _ { 0 }$ and $j = d _ { \tau } .$ : they refer to the same qubit at different times. The two explanations have equal prior probability even when $p \neq q .$ , and contribute equally to $W _ { 1 0 }$ and $W _ { 0 1 }$

Higher-weight explanations modify this balance in two ways. Redistributing data faults on $v _ { * }$ , with compensating measurement strings, can produce endpoints 00 or 11 and hence change $\Delta$ . For $\nu \ = \ 3$ , the first such pair uses either one data fault at the intermediate time or data faults at all three times, together with four or two measurement faults, respectively. A data fault beyond the endpoint pair gives an additional 00 explanation. Moving the construction from the initial boundary into the interior allows this extension on both sides.

The leading 10, 01 extensions involve a pair of data faults at adjacent times on a neighboring degree-ν qubit. Together with compensating edges on that qubit’s other checks, this pair replaces an occupied measurement edge. These extensions preserve the original endpoint values 10 or 01; unequal numbers of available extensions produce the leading contribution to $\delta .$

For $\nu = 3 .$ , label the neighboring checks 1, 2, 3 and choose

$$
\begin{array} { r l } & { \vec { a } = \vec { d } _ { 0 } + \vec { m } _ { 1 , 0 } + \vec { m } _ { 1 , 1 } + \vec { m } _ { 2 , 1 } , } \\ & { \vec { b } = \vec { d } _ { 2 } + \vec { m } _ { 2 , 0 } + \vec { m } _ { 3 , 0 } + \vec { m } _ { 3 , 1 } . } \end{array}\tag{9}
$$

Their common syndrome activates checks {2, 3}, {2}, and $\{ 1 , 2 \}$ in measurement rounds $0 , 1 , 2 .$ . More generally, for even $\nu = 2 k$ , partition the neighboring checks into sets $A , B$ of size $k ,$ and take

$$
\vec { a } = \vec { d } _ { 0 } + \sum _ { h \in \cal { A } } \vec { m } _ { h , 0 } , \qquad \vec { b } = \vec { d } _ { 1 } + \sum _ { h \in \cal { B } } \vec { m } _ { h , 0 } .
$$

For odd $\nu = 2 k + 1$ , partition them into $A , \{ c \} , B$ with $| { \mathcal { A } } | =$ $| B | = k ,$ , and use

$$
\begin{array} { l } { \displaystyle \vec { a } = \vec { d } _ { 0 } + \sum _ { h \in \mathcal { A } } ( \vec { m } _ { h , 0 } + \vec { m } _ { h , 1 } ) + \vec { m } _ { c , 1 } , } \\ { \displaystyle \vec { b } = \vec { d } _ { 2 } + \sum _ { h \in \mathcal { B } } ( \vec { m } _ { h , 0 } + \vec { m } _ { h , 1 } ) + \vec { m } _ { c , 0 } . } \end{array}
$$

For either degree parity, write $\vec { s } _ { * } = H \vec { a } = H \vec { b } .$ Summing the detector equations over time requires $e _ { i } \oplus e _ { j } = 1$ for every compatible correction supported on $U$ . Once the data endpoint is chosen, the measurement equations fix all remaining bits, so $^ { a , }$ b are the only compatible corrections on $U .$

The spatial assumptions in Theorem 1 control explanations extending outside $\bar { U . }$ The absence of Tanner 4-cycles limits distinct qubits to at most one shared check, while the bound on ker $H _ { Z }$ excludes spatial zero-syndrome patterns of weight below $2 \nu + 2 ,$ , including stabilizers. Together, these conditions establish that $a , b$ are the only minimum-weight explanations and allow the higher-weight contributions to $\Delta , \delta$ to be classified, as stated next and proved in Supplementary Information.

Proposition 2 (Two-rate structure of the phenomenological witnesses). Under the spatial and measurement-round assumptions of Theorem $^ { l , }$ the constructed $\vec { a } , \vec { b }$ are the only minimum-weight explanations of $\vec { s } _ { \ast }$ , with weight $\kappa = \kappa ( \nu )$ Each contains one data fault and $\kappa - 1$ measurement faults, so they remain equally likely for arbitrary $p , q .$ Put $x =$ $p / ( 1 - p ) , y = q / ( 1 - q )$ , and let $F _ { 0 } = 1 + O ( p )$ be the common no-errorfactor ofthe selected sector times the probability of zero syndrome in the other sector. Let $n _ { h }$ count the other degree-ν qubits at check $h ,$ and put $\begin{array} { r } { N _ { A } = \sum _ { h \in \mathcal { A } } n _ { h } , } \end{array}$ $\begin{array} { r } { N _ { B } = \sum _ { h \in B } n _ { h } } \end{array}$ . For even $\nu ,$ the endpoint imbalances obey

$$
\begin{array} { c } { { \Delta = F _ { 0 } x y ^ { 3 \nu / 2 } + { \cal O } ( p ^ { \kappa + \nu + 1 } ) , } } \\ { { \delta = F _ { 0 } ( N _ { A } - N _ { B } ) x ^ { 3 } y ^ { 3 \nu / 2 - 2 } } } \\ { { + { \cal O } ( p ^ { \kappa + \nu + 1 } ) . } } \end{array}\tag{10}
$$

For odd $\nu ,$ use two placements of the construction: $j = 0$ starts in measurement round zero, while $j = 1$ shifts every fault and detector by one measurement round. Both have the same minimum weight and prior composition. Their endpoint imbalances satisfy

$$
\begin{array} { l } { { \Delta _ { j } = F _ { 0 } \big [ x y ^ { \nu - 1 } ( y ^ { 2 } - x ^ { 2 } ) } } \\ { { \qquad + ( 1 + j ) x y ^ { 2 \nu } \big ] + O ( p ^ { 2 \nu + 2 } ) , } } \\ { { \delta _ { j } = 2 F _ { 0 } ( N _ { A } - N _ { B } ) x ^ { 3 } y ^ { 2 \nu - 2 } } } \\ { { \qquad + O ( p ^ { 2 \nu + 2 } ) , \qquad j = 0 , 1 . } } \end{array}\tag{11}
$$

Proof. In Supplementary Information, it classifies all compatible configurations through weight $\kappa + \nu ,$ retaining their separate data and measurement counts. The two minimum explanations have odds weight $x y ^ { \kappa - 1 }$ . For even $\nu ,$ the next configurations comprise one 00 explanation with one data fault and $3 \nu / 2$ measurement faults, and $N _ { A } , N _ { B }$ explanations in classes 10, 01, each with three data faults and $3 \nu / 2 - 2$ measurement faults. This gives Eq. (10).

For odd $\nu ,$ the weight- $\left( \nu + 2 \right)$ explanations in classes 00, 11 have odds weights $x { \bar { y } } ^ { \nu + 1 }$ and $x ^ { 3 } y ^ { \nu - 1 }$ . They need not cancel when $p \neq q .$ . At weight $2 \nu + 1$ , a single data fault one measurement round beyond the endpoint pair gives a 00 explanation with odds weight $x y ^ { 2 \nu }$ . The boundary placement admits one such fault; the interior placement admits two, one on each side. The 10, 01 counts are unchanged by translation and give their stated difference. All omitted configurations have weight at least $2 \nu + 2 .$ , so their total probability is $O ( p ^ { 2 \nu + 2 } )$ because $q = \Theta ( p )$ . The independent other sector contributes the same common factor to both placements. □

ProofofTheorem 1. Choose any minimum-degree qubit. For odd $\nu ,$ fix any central check c. Choose the remaining partition $A , B$ to minimize $| N _ { A } - N _ { B } |$ and orient it so that $\eta = N _ { A } -$ $N _ { B } \geq 0$

If $\eta > 0$ , we show in Supplementary Information that it excludes both compatible outputs on $U .$ . The positive endpoint preference $\delta$ excludes $b .$ For a check in A with the largest $n _ { h }$ a negative measurement marginal bias (or a negative sum of two such biases for odd degree) excludes a. Posterior concentration rejects every bit outside $U .$ , so the boundary syndrome forces inconsistency.

If $\eta = 0$ and ν is even, Eq. (10) gives $\Delta = \Theta ( p ^ { \kappa + \nu } ) > 0$ and $\begin{array} { r } { \delta = O ( p ^ { \kappa + \nu + 1 } ) } \end{array}$ , again forcing inconsistency by Proposition 1.

For $\eta ~ = ~ 0$ and odd $\nu ,$ subtract the two expansions in Eq. (11):

$$
\Delta _ { 1 } - \Delta _ { 0 } = F _ { 0 } x y ^ { 2 \nu } + { \cal O } ( p ^ { 2 \nu + 2 } ) = \Theta ( p ^ { 2 \nu + 1 } ) .
$$

Thus at least one placement satisfies $\begin{array} { r } { | \Delta _ { j } | \geq | \Delta _ { 1 } - \Delta _ { 0 } | / 2 = } \end{array}$ $\Omega ( p ^ { 2 \nu + 1 } )$ , whereas both have $\delta _ { j } = \dot { O } ( p ^ { 2 \nu + 2 } )$ . At each sufficiently small $p ,$ at least one of the two syndromes therefore forces equal MAP decisions on its data endpoints, violating their required odd parity. In every case, each candidate syndrome has probability

$$
\mathrm { P r } \Big ( s _ { * } ^ { ( j ) } \Big ) = 2 F _ { 0 } x y ^ { \kappa - 1 } + { \cal O } ( p ^ { \kappa + 1 } ) = \Theta ( p ^ { \kappa } ) .
$$

Each case therefore supplies a syndrome of probability $\Theta ( p ^ { \kappa } )$ on which MAP is inconsistent, proving $P _ { \mathrm { i n c } } ( p ) ~ = ~ \Omega ( \dot { p } ^ { \kappa } )$ In the balanced odd-degree case, the successful placement may depend on $p ,$ but the constants are uniform over the two choices. No limit or expansion for $q / p$ is needed. □

Corollary 1 (Quartic inconsistency for the LP and Kasai code matrices). For the LP code matrix with $n = 1 1 2 2$ and five noisy measurement rounds, and the Kasai code matrix with $n \ = \ 1 1 5 2$ and eight noisy measurement rounds, the independent X/Z phenomenological model with data-fault probability p and measurement-fault probability $q \ = \ \Theta ( p )$ has $P _ { \mathrm { i n c } } ( p ) \bar { = } \Omega ( p ^ { 4 } )$ under exact marginal MAP.

Proof. The spatial checks provided in the Supplementary Information verify the hypotheses of Theorem 1. Both matrices have a minimum column degree $\nu = 3$ . At the selected qubit, the other degree-3 incidence counts are $( 4 , 4 , 4 )$ for LP and (11, 11, 11) for Kasai, so $N _ { A } = N _ { B }$ . The measurement-round counts admit both placements. Direct enumeration in Supplementary Information confirms the separate data/measurement counts for each placement. □

All asymptotic statements hold at fixed code and measurement-round count. The theorem concerns exact marginals in the independent $X / Z$ model. Approximate NeuralMP decoding and depolarizing simulations are assessed separately by their numerical results.

## D. Diversification

After iterative GNN decoding, almost all remaining failures leave a nonzero residual syndrome; syndrome-consistent but incorrect corrections are comparatively very rare. Similar cases arise in belief propagation and are commonly addressed by generating alternative decoding attempts and keeping the best consistent one [67, 68, 87]. We apply this principle to the trained NeuralMP: for shots whose residual syndrome remains nonzero, the network infers correction candidates with modified versions of the decoding problem and selects the one with the highest confidence.

Syndrome-flip (SF) search. We hypothesize a small set of error mechanisms, indicated by $t ,$ and decode the modified syndrome $s ^ { \prime } = s \oplus H t$ with the same iterative NeuralMP, obtaining $e ^ { \prime } .$ The candidate for the original problem is $e = t \oplus e ^ { \prime } ,$ which satisfies $H e \ = \ s$ whenever $\boldsymbol { H } \boldsymbol { e } ^ { \prime } = \boldsymbol { s } ^ { \prime }$ Because the network re-decodes the modified syndrome, it can revise the whole correction rather than only the hypothesized bits. The hypotheses are guided by the network’s confidence through two signals: absz, which ranks mechanisms by the smallest absolute logit of the first iteration $z _ { 0 , f } |$ (locally ambiguous decisions), and $\mathit { \Pi } \mathit { f l i p } $ , which ranks mechanisms by how often they are corrected across the iterations (decisions that are revised repeatedly, the neural analog of the oscillation signal of BP-SF [67]). For each signal we take the 32 top-ranked mechanisms and form 32 single, 32 pair and 32 triple hypotheses, giving 192 candidates per shot.

Further diversification. For LP20 Code, if all SF candidates have non-sero residual syndrome, a second stage decodes the syndrome in equivalent representations given by code automorphism $( H P _ { g } = D _ { g } H , P _ { g } p = p )$ , for which the network is not exactly equivariant (because of positional encoding) and therefore follows different trajectories, combined with random multiplicative perturbations of the prior inputs, $\tilde { p } _ { f } = \mathrm { c l i p } ( p _ { f } e ^ { \alpha \xi _ { f } } ) , \xi _ { f } \sim \mathcal { N } ( 0 , 1 )$ . After decoding, candidates are mapped back to the original coordinates.

Selection. Among candidates with $H e \ = \ s$ we select e<sup>⋆</sup> = arg min $\sum _ { f } e _ { f } \log { \frac { 1 - p _ { f } } { p _ { f } } }$ under the original noise model; if none is consistent, the output of the fourth iterative decoding is retained. No ground-truth information is used.

The search converts the dominant failure mode, nonconvergence, into a recoverable one: for the Kasai code at $p ~ = ~ 1 . 5 \%$ it resolves 99.6% of the shots left unresolved by iterative decoding, reducing the unresolved fraction from $2 . 6 \times 1 0 ^ { - 4 } \mathrm { t o } 1 . 0 \times 1 0 ^ { - 6 }$ per shot. Since these search methods have similar patterns, with the help of neutral network we can make candidates decoding as parallel batch, its latency can be amortized.

## E. Quantum memory benchmarks

For LP20 code, we simulate $r \ = \ 5$ measurement rounds with independent X and $Z$ data errors, each occurring with a probability $p ,$ and measurement errors occurring with a probability $q = p$ . For Kasai, we use $r = 8$ measurement rounds with depolarizing data errors of probability $p$ and measurement errors of probability $q = 4 p / 3$

For LP20 code, we set the hidden dimension of NeuralMP as $\chi = 1 2 8$ and 12 MP-rounds; for Kasai code, the network has a hidden dimension of $\chi ~ = ~ 6 4$ and 80 MProunds. Both use at most four outer iterations. Shots with a nonzero residual enter syndrome-flip search, which tests 192 iterative-decoding candidates constructed using uncertainty and repeated-correction signals. For LP20 code, shots without a syndrome-consistent candidate enter an additional stage of 396 prior-perturbed, symmetry-transformed candidates and 96 syndrome-flip candidates guided by symmetry-view disagreement. And for the Kasai code, Triage uses no second search stage. Candidates are checked against the original syndrome and ranked using the original noise priors.

For LP20 code, BP-OSD baseline uses 100 min-sum iterations with scaling 0.625 and order-10 combination-sweep OSD. Its relay-BP baseline uses 100 initial BP iterations and up to 300 relay stages of 60 iterations each. For Kasai code, the two-stage baseline uses 100 min-sum iterations followed by up to 300 relay stages of 60 iterations; Tiered adds a COPTbased search, with a limit of 120 seconds per side and eight solver threads.

The per-qubit-per-round LER is $[ 1 - ( 1 - \bar { P } ) ^ { 1 / r } ] / k$ , where $\bar { P }$ is the mean of the block failure probabilities for the two logical sectors and k is the number of logical qubits. Error bars combine 95% Clopper–Pearson intervals with uncertainty in the sector when the failed sector was not recorded; treating sector counts as independent in the interval calculation is an approximation.

For Triage, the plotted amortized time uses a batch size of 64 and combines measured front-end and search costs with observed search frequencies, including handoff and final processing. The neural decoder was measured on one NVIDIA H-series GPU; the algorithmic baselines were measured on the CPU, with single-threaded BP and relay-BP and eight threads for the Tiered solver.

## ACKNOWLEDGEMENTS

We acknowledge the insightful discussion with Yi-Zhuang You. H.Y.H. acknowledges support from a startup fund from IOP, CAS. which initiated the study. C.Z. acknowledges support from a startup fund from Eastern Institute of Technology,

[1] P. W. Shor, Scheme for reducing decoherence in quantum computer memory, Physical Review A 52, R2493 (1995).

[2] A. M. Steane, Error correcting codes in quantum theory, Physical Review Letters 77, 793 (1996).

[3] D. Gottesman, An introduction to quantum error correction and fault-tolerant quantum computation (2009), arXiv:0904.2557 [quant-ph].

[4] S. Krinner, N. Lacroix, A. Remm, A. Di Paolo, E. Genois, C. Leroux, C. Hellings, S. Lazar, F. Swiadek, J. Herrmann, G. J. Norris, C. K. Andersen, M. Muller, A. Blais, C. Eichler, and¨ A. Wallraff, Realizing repeated quantum error correction in a distance-three surface code, Nature 605, 669 (2022).

[5] Google Quantum AI, Suppressing quantum errors by scaling a surface code logical qubit, Nature 614, 676 (2023).

[6] Google Quantum AI and Collaborators, Quantum error correction below the surface code threshold, Nature 638, 920 (2025).

[7] D. Bluvstein, S. J. Evered, A. A. Geim, S. H. Li, H. Zhou, T. Manovitz, S. Ebadi, M. Cain, M. Kalinowski, D. Hangleiter, J. P. Bonilla Ataides, N. Maskara, I. Cong, X. Gao, P. Sales Rodriguez, T. Karolyshyn, G. Semeghini, M. J. Gullans, M. Greiner, V. Vuletic, and M. D. Lukin, Logical quantum´ processor based on reconfigurable atom arrays, Nature 626, 58 (2024).

[8] D. Bluvstein, A. A. Geim, S. H. Li, S. J. Evered, J. P. Bonilla

Ningbo. We gratefully acknowledge the support of the Pioneer Program at the Shanghai Artificial Intelligence Laboratory and thank program managers Xiaoyi Yu and Ting Wang for their support and assistance. H.Y.H. also thanks Shang Liu for the GPU support. Part of the computational work for this research was supported by the High Performance Computing Center at Eastern Institute of Technology, Ningbo. We thank Cardinal Operations for providing an academic license for COPT, which we used to implement the most-likely-error (MLE) decoder. We acknowledge the use of AI in this project and writing.

Hardware specifications: all latency measurements were performed on a single server. NeuralMP’s decoding time was measured on a single NVIDIA H-series GPU, with GPU clocks left at their default dynamic boost. The host has two Intel Xeon Platinum 8481C CPUs (56 cores each, 2.0 GHz base, up to 3.8 GHz turbo, turbo enabled). The BP-based baselines (min-sum BP, relay-BP and BP-OSD) were run singlethreaded, with each process pinned to its own physical core. The exact-solver stage of the Tiered decoder (COPT) was run with 8 threads per call.

Competing interests: All authors declare no competing interests.

Data availability: Source data are available for this paper. All other data supporting the plots within this paper and other study findings are available from the corresponding author upon reasonable request.

Code availability: The code used in this study is available from the corresponding author upon reasonable request.

Note: We will release the fully supplementary information in the second update soon.

Ataides, G. Baranes, A. Gu, T. Manovitz, M. Xu, M. Kalinowski, S. Majidy, C. Kokail, N. Maskara, E. C. Trapp, L. M. Stewart, S. Hollerith, H. Zhou, M. J. Gullans, S. F. Yelin, M. Greiner, V. Vuletic, M. Cain, and M. D. Lukin, A fault-´ tolerant neutral-atom architecture for universal quantum computation, Nature 649, 39 (2026).

[9] L. Egan, D. M. Debroy, C. Noel, A. Risinger, D. Zhu, D. Biswas, M. Newman, M. Li, K. R. Brown, M. Cetina, and C. Monroe, Fault-tolerant control of an error-corrected qubit, Nature 598, 281 (2021).

[10] C. Ryan-Anderson, J. G. Bohnet, K. Lee, D. Gresh, A. Hankin, J. P. Gaebler, D. Francois, A. Chernoguzov, D. Lucchetti, N. C. Brown, T. M. Gatterman, S. K. Halit, K. Gilmore, J. A. Gerber, B. Neyenhuis, D. Hayes, and R. P. Stutz, Realization of realtime fault-tolerant quantum error correction, Physical Review X 11, 041058 (2021).

[11] L. Postler, S. Heußen, I. Pogorelov, M. Rispler, T. Feldker, M. Meth, C. D. Marciniak, R. Stricker, M. Ringbauer, R. Blatt, P. Schindler, M. Muller, and T. Monz, Demonstration of fault-¨ tolerant universal quantum gate operations, Nature 605, 675 (2022).

[12] P. W. Shor, Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer, SIAM Journal on Computing 26, 1484 (1997).

[13] S. Lloyd, Universal quantum simulators, Science 273, 1073 (1996).

[14] A. Aspuru-Guzik, A. D. Dutoi, P. J. Love, and M. Head-Gordon, Simulated quantum computation of molecular energies, Science 309, 1704 (2005).

[15] L. K. Grover, Quantum mechanics helps in searching for a needle in a haystack, Physical Review Letters 79, 325 (1997).

[16] A. W. Harrow, A. Hassidim, and S. Lloyd, Quantum algorithm for linear systems of equations, Physical Review Letters 103, 150502 (2009).

[17] C. Gidney and M. Ekera, How to factor 2048 bit RSA inte-˚ gers in 8 hours using 20 million noisy qubits, Quantum 5, 433 (2021), arXiv:1905.09749.

[18] C. Gidney, How to factor 2048 bit RSA integers with less than a million noisy qubits (2025), arXiv:2505.15917 [quant-ph].

[19] P. Webster, L. Berent, O. Chandra, E. T. Hockings, N. Baspin, F. Thomsen, S. C. Smith, and L. Z. Cohen, The Pinnacle Architecture: Reducing the cost of breaking RSA-2048 to 100 000 physical qubits using quantum LDPC codes (2026), arXiv:2602.11457 [quant-ph].

[20] M. E. Beverland, P. Murali, M. Troyer, K. M. Svore, T. Hoefler, V. Kliuchnikov, G. H. Low, M. Soeken, A. Sundaram, and A. Vaschillo, Assessing requirements to scale to practical quantum advantage (2022), arXiv:2211.07629 [quant-ph].

[21] M. Cain, Q. Xu, R. King, L. R. B. Picard, H. Levine, M. Endres, J. Preskill, H.-Y. Huang, and D. Bluvstein, Shor’s algorithm is possible with as few as 10,000 reconfigurable atomic qubits (2026), arXiv:2603.28627 [quant-ph].

[22] N. P. Breuckmann and J. N. Eberhardt, Quantum low-density parity-check codes, PRX Quantum 2, 040101 (2021).

[23] D. Gottesman, Fault-tolerant quantum computation with constant overhead, Quantum Information and Computation 14, 1338 (2014).

[24] Q. Xu, J. P. Bonilla Ataides, C. A. Pattison, N. Raveendran, D. Bluvstein, J. Wurtz, B. Vasic, M. D. Lukin, L. Jiang, and´ H. Zhou, Constant-overhead fault-tolerant quantum computation with reconfigurable atom arrays, Nature Physics 20, 1084 (2024).

[25] A. Bhardwaj, M. Ma, N. Meister, R. King, D. Bluvstein, J. Preskill, M. Cain, Q. Xu, and H.-Y. Huang, High-rate qLDPC processors (2026), arXiv:2607.28795 [quant-ph].

[26] J.-P. Tillich and G. Zemor, Quantum LDPC codes with positive´ rate and minimum distance proportional to the square root of the blocklength, IEEE Transactions on Information Theory 60, 1193 (2014).

[27] P. Panteleev and G. Kalachev, Asymptotically good quantum and locally testable classical LDPC codes, in Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing (2022) pp. 375–388.

[28] A. Leverrier and G. Zemor, Quantum Tanner codes, in´ 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS) (2022) pp. 872–883.

[29] S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, and T. J. Yoder, High-threshold and low-overhead fault-tolerant quantum memory, Nature 627, 778 (2024), arXiv:2308.07915 [quant-ph].

[30] K. Kasai, Breaking the orthogonality barrier in quantum LDPC codes, Quantum 10, 2205 (2026), arXiv:2601.08824 [quantph].

[31] C. Zhao, C. Duckering, A. Gu, N. Maskara, and H. Zhou, Towards ultra-high-rate quantum error correction with reconfigurable atom arrays, arXiv:2604.16209 (2026), arXiv:2604.16209 [quant-ph].

[32] J. Y. Lee, K. Okada, N. Maskara, K. Kasai, and H. Zhou,

Design principles for ultra-high-rate quantum codes (2026), arXiv:2609.30069 [quant-ph].

[33] W. Yang, C. Duckering, and A. Dua, Designer codes from GALA: Compact, self-dual, and rate-1/2 QEC on reconfigurable atom arrays (2026), arXiv:2608.07431 [quant-ph].

[34] Z. Lu, W. Li, and D.-L. Deng, Quantum error correction at ultralow overhead (2026), arXiv:2608.02773 [quant-ph].

[35] K. Okada and K. Kasai, Pair-partition constructions for CPMbased quantum LDPC codes (2026), arXiv:2607.14091 [quantph].

[36] A. G. Fowler, A. C. Whiteside, and L. C. L. Hollenberg, Towards practical classical processing for the surface code, Physical Review Letters 108, 180501 (2012).

[37] B. M. Terhal, Quantum error correction for quantum memories, Reviews of Modern Physics 87, 307 (2015).

[38] E. R. Berlekamp, R. J. McEliece, and H. C. A. van Tilborg, On the inherent intractability of certain coding problems, IEEE Transactions on Information Theory 24, 384 (1978).

[39] P. Iyer and D. Poulin, Hardness of decoding quantum stabilizer codes, IEEE Transactions on Information Theory 61, 5209 (2015).

[40] R. G. Gallager, Low-density parity-check codes, IRE Transactions on Information Theory 8, 21 (1962).

[41] F. R. Kschischang, B. J. Frey, and H.-A. Loeliger, Factor graphs and the sum-product algorithm, IEEE Transactions on Information Theory 47, 498 (2001).

[42] S.-Y. Chung, G. D. Forney, Jr., T. J. Richardson, and R. Urbanke, On the design of low-density parity-check codes within 0.0045 dB of the Shannon limit, IEEE Communications Letters 5, 58 (2001).

[43] D. J. C. MacKay, G. Mitchison, and P. L. McFadden, Sparsegraph codes for quantum error correction, IEEE Transactions on Information Theory 50, 2315 (2004).

[44] D. Poulin and Y. Chung, On the iterative decoding of sparse quantum codes, Quantum Information and Computation 8, 987 (2008), arXiv:0801.1241.

[45] A. S. Maan, F. M. Garcia Herrero, A. Paler, and V. Savin, Decoding correlated errors in quantum LDPC codes, Nature Communications 17, 3965 (2026).

[46] T. Muller, T. Alexander, M. E. Beverland, M. B¨ uhler, B. R.¨ Johnson, T. Maurer, and D. Vandeth, Improved belief propagation is sufficient for real-time decoding of quantum memory, arXiv:2506.01779 (2025), arXiv:2506.01779 [quant-ph].

[47] M. P. C. Fossorier and S. Lin, Soft-decision decoding of linear block codes based on ordered statistics, IEEE Transactions on Information Theory 41, 1379 (1995).

[48] P. Panteleev and G. Kalachev, Degenerate quantum LDPC codes with good finite length performance, Quantum 5, 585 (2021).

[49] J. Roffe, D. R. White, S. Burton, and E. Campbell, Decoding across the quantum low-density parity-check code landscape, Physical Review Research 2, 043423 (2020).

[50] T. Hillmann, L. Berent, A. O. Quintavalle, J. Eisert, R. Wille, and J. Roffe, Localized statistics decoding for quantum lowdensity parity-check codes, Nature Communications 16, 8214 (2025).

[51] L. Aghababaie Beni, O. Higgott, and N. Shutty, Tesseract: A search-based decoder for quantum error correction (2025), arXiv:2503.10988 [quant-ph].

[52] M. Ye, D. Wecker, and N. Delfosse, Beam search decoder for quantum low-density parity-check codes, PRX Quantum 7, 033002 (2026), arXiv:2512.07057.

[53] M. Cain, C. Zhao, H. Zhou, N. Meister, J. P. Bonilla Ataides, A. Jaffe, D. Bluvstein, and M. D. Lukin, Correlated decoding

of logical algorithms with transversal gates, Physical Review Letters 133, 240602 (2024), arXiv:2403.03272 [quant-ph].

[54] G. Torlai and R. G. Melko, Neural decoder for topological codes, Physical Review Letters 119, 030501 (2017).

[55] N. Maskara, A. Kubica, and T. Jochym-O’Connor, Advantages of versatile neural-network decoding for topological codes, Physical Review A 99, 052351 (2019).

[56] A. W. Senior, T. Edlich, F. J. H. Heras, L. M. Zhang, O. Higgott, J. S. Spencer, T. Applebaum, S. Blackwell, J. Ledford, A. Zemgulyt<sup>ˇ</sup> e, A.˙ Z<sup>ˇ</sup> ´ıdek, N. Shutty, A. Cowie, Y. Li, G. Holland, P. Brooks, C. Beattie, M. Newman, A. Davies, C. Jones, S. Boixo, H. Neven, P. Kohli, and J. Bausch, A scalable and real-time neural decoder for topological quantum codes (2025), arXiv:2512.07737 [quant-ph].

[57] P. Baireuther, T. E. O’Brien, B. Tarasinski, and C. W. J. Beenakker, Machine-learning-assisted correction of correlated qubit errors in a topological code, Quantum 2, 48 (2018).

[58] J. Bausch, A. W. Senior, F. J. H. Heras, T. Edlich, A. Davies, M. Newman, C. Jones, K. Satzinger, M. Y. Niu, S. Blackwell, G. Holland, D. Kafri, J. Atalaya, C. Gidney, D. Hassabis, S. Boixo, H. Neven, and P. Kohli, Learning high-accuracy error decoding for quantum processors, Nature 635, 834 (2024).

[59] M. Lange, P. Havstrom, B. Srivastava, I. Bengtsson, V. Bergen-¨ tall, K. Hammar, O. Heuts, E. van Nieuwenburg, and M. Granath, Data-driven decoding of quantum error correcting codes using graph neural networks, Physical Review Research 7, 023181 (2025), arXiv:2307.01241 [quant-ph].

[60] H. Cao, F. Pan, Y. Wang, and P. Zhang, qecGPT: decoding quantum error-correcting codes with generative pre-trained transformers (2023), arXiv:2307.09025 [quant-ph].

[61] J. P. Bonilla Ataides, A. Gu, S. F. Yelin, and M. D. Lukin, Neural decoders for universal quantum algorithms, PRX Intelligence 1, 013008 (2026).

[62] A. Gu, J. P. Bonilla Ataides, M. D. Lukin, and S. F. Yelin, Scalable neural decoders for practical fault-tolerant quantum computation (2026), arXiv:2604.08358 [quant-ph].

[63] Y.-H. Liu and D. Poulin, Neural belief-propagation decoders for quantum error-correcting codes, Physical Review Letters 122, 200501 (2019).

[64] A. Gong, S. Cammerer, and J. M. Renes, Graph neural networks for enhanced decoding of quantum LDPC codes, in 2024 IEEE International Symposium on Information Theory (ISIT) (2024) pp. 2700–2705, arXiv:2310.17758 [quant-ph].

[65] V. Ninkovic, O. Kundacina, D. Vukobratovic, C. Hager, and¨ A. Graell i Amat, Decoding quantum LDPC codes using graph neural networks, in GLOBECOM 2024 - 2024 IEEE Global Communications Conference (2024) pp. 3479–3484, arXiv:2408.05170 [quant-ph].

[66] A. S. Maan and A. Paler, Machine learning message-passing for the scalable decoding of QLDPC codes, npj Quantum Information 11, 78 (2025), arXiv:2408.07038 [quant-ph].

[67] M. Wang, A. Li, and F. Mueller, Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform BP-OSD (2025), arXiv:2507.00254 [quant-ph].

[68] S. Koutsioumpas, H. Sayginel, M. Webster, and D. E. Browne, Automorphism ensemble decoding of quantum LDPC codes, arXiv:2503.01738 (2025), arXiv:2503.01738 [quant-ph].

[69] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, Attention is all you need, in Advances in Neural Information Processing Systems, Vol. 30 (2017) arXiv:1706.03762 [cs.CL].

[70] D. Ge, Q. Huangfu, Z. Wang, J. Wu, and Y. Ye, Cardinal Optimizer (COPT) user guide (2023).

[71] V. Sivak, A. Morvan, M. Broughton, R. G. Cortinas, J. Bausch,˜

A. W. Senior, et al., Reinforcement learning control of quantum error correction, Nature 655, 879 (2026).

[72] W. Gong and H.-Y. Hu, Provably efficient self-calibrating quantum fault tolerance (2026), arXiv:2608.05686 [quant-ph].

[73] Z.-J. Chen, H. Huang, L. Sun, Q.-X. Jie, J. Zhou, Z. Hua, Y. Xu, W. Wang, G.-C. Guo, C.-L. Zou, L. Sun, and X.-B. Zou, Robust and optimal control of open quantum systems, Science Advances 11, eadr0875 (2025), https://www.science.org/doi/pdf/10.1126/sciadv.adr0875.

[74] H.-Y. Hu, A. McClain Gomez, L. Chen, A. Trowbridge, A. J. Goldschmidt, Z. Manchester, F. T. Chong, A. Jaffe, and S. F. Yelin, Universal Dynamics with Globally Controlled Analog Quantum Simulators, arXiv e-prints , arXiv:2508.19075 (2025), arXiv:2508.19075 [quant-ph].

[75] P. M. Poggi, G. De Chiara, S. Campbell, and A. Kiely, Universally robust quantum control, Phys. Rev. Lett. 132, 193801 (2024).

[76] B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. R. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, P. Kohli, and A. Fawzi, Mathematical discoveries from program search with large language models, Nature 625, 468 (2024).

[77] A. Novikov, N. Vu, M. Eisenberger, E. Dupont, P.-S. Huang,˜ A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. R. Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog, AlphaEvolve: A coding agent for scientific and algorithmic discovery (2025), arXiv:2506.13131 [cs.AI].

[78] J. Oh, G. Farquhar, I. Kemaev, D. A. Calian, M. Hessel, L. Zintgraf, S. Singh, H. van Hasselt, and D. Silver, Discovering stateof-the-art reinforcement learning algorithms, Nature 648, 312 (2025).

[79] S. Arlt, H. Duan, F. Li, S. M. Xie, Y. Wu, and M. Krenn, Metadesigning quantum experiments with language models, Nature Machine Intelligence 8, 148 (2026).

[80] L. Z. Cohen, I. H. Kim, S. D. Bartlett, and B. J. Brown, Low-overhead fault-tolerant quantum computing using longrange connectivity, Science Advances 8, eabn1717 (2022), arXiv:2110.10794 [quant-ph].

[81] T.-G. Zhou, B. Gu, J. Eisert, and C. Zhao, Parallel algebraic surgery for qLDPC codes with constant shuttling depth, arXiv:2609.39874 (2026), arXiv:2609.39874 [quant-ph].

[82] T. J. Yoder, E. Schoute, P. Rall, E. Pritchett, J. M. Gambetta, A. W. Cross, M. Carroll, and M. E. Beverland, Tour de gross: A modular quantum computer based on bivariate bicycle codes (2025), arXiv:2506.03094 [quant-ph].

[83] Z. He, A. Cowtan, D. J. Williamson, and T. J. Yoder, Extractors: QLDPC architectures for efficient Pauli-based computation (2025), arXiv:2503.10390 [quant-ph].

[84] H. Zhou, C. Zhao, M. Cain, D. Bluvstein, N. Maskara, C. Duckering, H.-Y. Hu, S.-T. Wang, A. Kubica, and M. D. Lukin, Low-overhead transversal fault tolerance for universal quantum computation, Nature 646, 303 (2025), arXiv:2406.17653 [quant-ph].

[85] NVIDIA, CUDA-Q QEC documentation: Decoders, accessed 29 October 2026.

[86] A. Cicero, L. Altamura, M. Lange, M. Granath, and P. Trancoso, Low latency GNN accelerator for quantum error correction (2026), arXiv:2603.22149 [quant-ph].

[87] A. Baldelli, M. Baldi, M. Battaglioni, F. Chiaraluce, and P. Santini, Design and analysis of quantum dualcontaining CSS LDPC codes based on quasi-dyadic matrices, arXiv:2605.03631 (2026), arXiv:2605.03631 [cs.IT].