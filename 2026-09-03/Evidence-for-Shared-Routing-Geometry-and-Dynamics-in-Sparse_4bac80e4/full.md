# Evidence for Shared Routing Geometry and Dynamics in Sparse Mixture-of-Experts

Kirill LabzinCentral University, Moscowlabzin.kr@gmail.com

Artem Dzhalilov Innopolis University, Innopolis artem.dzhalilov@gmail.com

Stepan Kulibaba Innopolis University, Innopolis kulibabast@gmail.com

Artem Gorokhov Independent Researcher gorohovartem5534@gmail.com

## Abstract

Sparse mixture-of-experts (MoE) models use an independently parameterized router at each sparse layer to select experts for every token. Prior work has shown that routing decisions across depth can often be predicted from earlier routing signals, suggesting that routing is not fully independent across layers. However, the structure behind this predictability remains unclear. In this work, we provide evidence that routing-relevant states across layers share a common geometric structure that is obscured by layer-specific coordinate systems. We isolate the control subspace of each router and align these spaces into a shared canonical representation using generalized orthogonal Procrustes analysis. After alignment, a single linear transition reaches R<sup>2</sup> = 0.39–0.71 and retains 79–90% of the predictive power of separately fitted layer-specific dynamics, indicating that much of routing-state evolution follows a reusable process across depth. We then ask whether this shared dynamics is specific to routing or simply reflects the smooth evolution of hidden representations. A matched-rank comparison shows that residual representations are often easier to predict across layers, while router-control states preserve the model’s expert choices much more faithfully. This separates generic cross-layer predictability from routing-specific information. Finally, we test whether the predicted canonical states remain meaningful when used in place of native routing states. The transported states preserve local routing behavior, while learned state evolution reduces ∆NLL relative to simple persistence by 15.7% on OLMoE and 6.2% over a 10-router horizon on Phi.

## 1 Introduction

Sparse MoE models increase parameter capacity while activating only a subset of experts for each token. Their routers therefore play a central role in determining which computation is performed, how expert load is distributed, and which experts must be available at inference time. Existing work has studied load balancing, specialization, routing failures, and expert prefetching [5, 7, 8, 12, 14]. Most analyses, however, treat the router at layer ℓ as meaningful only inside that layer’s own expert-index space.

Recent evidence suggests a more structured picture. Cross-layer prefetching methods predict future expert choices from nearby routing signals [4, 13, 15]. Mechanistic work identifies a routing-visible component of the residual stream and observes that it rotates across depth [11]. Router and expert parameters also develop coupled geometry within a layer [2]. Together, these observations motivate a basic question:

What geometric structure underlies the observed predictability of routing across depth, and can routing-relevant states from different layers be aligned into a common representation?

Answering this question requires a representation that can be compared meaningfully across layers. Directly comparing router weights is not sufficient, because hidden representations evolve with depth and expert identities are specific to each layer. We therefore isolate, at each layer, the subspace of hidden-state directions that can change the router’s relative expert logits. This router-control subspace captures the part of the representation that is functionally visible to the router. We then exploit its orthogonal gauge freedom and use generalized orthogonal Procrustes analysis (GPA) to align the layer-wise control spaces into a common canonical coordinate system.

This common coordinate system allows us to test whether cross-layer routing predictability reflects a reusable dynamical structure rather than only local similarity. If such a structure exists, a single low-capacity transition should predict the evolution of canonical routing states across many layers nearly as well as separately fitted layer-specific transitions. We further test whether these states are functionally meaningful by replacing native routing states with transported ones and measuring the resulting change in language-model loss. If the shared structure captures the routing information actually used by the model, these substitutions should cause little degradation. Our experiments support both predictions, while also showing that the strength of the shared structure depends on the architecture and transport horizon.

## Our main contributions are:

• We show that the router-control factorization has an exact orthogonal gauge symmetry and formulate cross-layer comparison on the corresponding quotient space.

• Across Granite, OLMoE, Phi-tiny-MoE, and an IBM shared-expert MoE, one shared linear transition captures 79–90% of the $R ^ { 2 }$ of layer-specific dynamics after gauge fixing.

• Matched-rank, matched-readout probes distinguish generic residual smoothness from information that is specifically useful for expert selection.

• Causal interventions show reproducible local transport. A simple margin bound connects canonicalstate error to exact top-k stability, and learned dynamics beats persistence at longer horizons on Phi and OLMoE.

## 2 Related Work

MoE routing and specialization. Sparse gating underlies large-scale MoE systems such as sparsely gated MoE and Switch Transformers [5, 8]. Later work studies routing schemes and expert specialization in models such as Expert Choice, OLMoE, and OpenMoE [7, 10, 14]. We instead study how routing-relevant representations relate across depth.

Routing structure across depth. Polysemantic Experts, Monosemantic Paths decomposes hidden states into routing-visible control and routing-invisible content, $h _ { \ell } = h _ { \ell } ^ { \mathrm { c t r l } } + h _ { \ell } ^ { \mathrm { c o n t e n t } }$ , and observes that control rotates across layers [11]. Fate and later prefetching methods likewise show that future expert choices are predictable from earlier routing signals [4, 13, 15]. These results reveal cross-layer routing structure, but not whether it reflects shared dynamics or layer-specific coordinates. We address this by aligning router-control states and studying their shared geometry and dynamics.

Representation alignment. CKA and Procrustes-based methods provide tools for comparing neural representations beyond their native coordinate systems [6, 9], while recent work extends orthogonal alignment to multiple representations using GPA [1]. We instead apply this idea to different depths of the same MoE, using the orthogonal non-identifiability of the router-control factorization to define a common coordinate system for routing states.

Layer-wise dynamics. Viewing network depth as discrete time has precedents in dynamicalsystems and Koopman-style analyses [3]. Rather than modeling the full hidden state, we study router-control states and test whether their apparently layer-specific transitions become shared after alignment.

## 3 Method

Figure 1 summarizes the method in three stages. We first extract a low-dimensional router-control state from the hidden representation entering each sparse MoE router. To compare these layerspecific control states across depth, we align them into a common canonical space using generalized orthogonal Procrustes analysis. In this shared space, we model routing-state evolution with a single linear transition and decode the predicted state with the target layer’s own readout to obtain expert logits and routing decisions.

![](images/b93c68e44b56f67138f1d2225d3f8641ca8516132da12e173d71a64666c86307.jpg)  
Figure 1: Method overview. We extract router-control states from layer-wise MoE routers, align them into a shared canonical space, and model their evolution with a shared linear transition. The predicted canonical state is then decoded by the target layer’s own readout to recover expert logits and routing decisions.

## 3.1 Router-control factorization

At layer ℓ, let E denote the number of routed experts and d the residual-stream width. The router weight matrix is $W _ { \ell } \in \mathbb { R } ^ { E \times d }$ , the token representation entering the router is $h _ { \ell , t } \in \mathbb { R } ^ { d }$ , and the router logits are $g _ { \ell , t } = W _ { \ell } h _ { \ell , t } \in \mathbb { R } ^ { E }$ . Let $\mathbf { 1 } _ { E } = ( 1 , \ldots , 1 ) ^ { \top } \in \mathbb { R } ^ { E }$ be the all-ones vector and $I _ { E }$ the $E \times E$ identity. Since softmax probabilities and top-k selection are invariant to a common shift of all E expert logits, we remove this common-logit direction by centering the router across experts:

$$
\widetilde { W } _ { \ell } = \left( I _ { E } - \frac { 1 } { E } \mathbf { 1 } _ { E } \mathbf { 1 } _ { E } ^ { \top } \right) W _ { \ell } = W _ { \ell } - \mathbf { 1 } _ { E } \bar { w } _ { \ell } ^ { \top } ,\tag{1}
$$

where $\bar { w } _ { \ell } = E ^ { - 1 } W _ { \ell } ^ { \top } \mathbf { 1 } _ { E } \in \mathbb { R } ^ { d }$ is the mean router row. Thus $\widetilde { W _ { \ell } }$ is simply $W _ { \ell }$ with the mean expert-weight vector subtracted from every row, and rank $( \widetilde { W } _ { \ell } ) \le E - 1$ . Let

$$
\widetilde { W } _ { \ell } = A _ { \ell } S _ { \ell } U _ { \ell } ^ { \top } = B _ { \ell } U _ { \ell } ^ { \top } , \qquad B _ { \ell } : = A _ { \ell } S _ { \ell } ,\tag{2}
$$

be the compact SVD at rank r. The columns of $U _ { \ell } \in \mathbb { R } ^ { d \times r }$ span exactly the residual-stream directions that can change relative router logits. We therefore define the router-control coordinates

$$
\begin{array} { r } { \boldsymbol { x } _ { \ell , t } = \boldsymbol { U } _ { \ell } ^ { \top } \boldsymbol { h } _ { \ell , t } \in \mathbb { R } ^ { r } , \qquad \widetilde { g } _ { \ell , t } = B _ { \ell } \boldsymbol { x } _ { \ell , t } . } \end{array}\tag{3}
$$

At full numerical rank, Eq. (3) reconstructs centered logits to numerical precision. Across all evaluated models, it also preserves the native top-k set exactly and leaves language-model loss unchanged.

## 3.2 Exact orthogonal gauge symmetry

The basis $U _ { \ell }$ is defined only up to an orthogonal change of coordinates. This freedom is not an artifact of the analysis, but an exact symmetry of the router factorization. Let

$$
{ \mathrm { O } } ( r ) = \{ R \in \mathbb { R } ^ { r \times r } : R ^ { \top } R = I _ { r } \}
$$

denote the set of orthogonal transformations of the r-dimensional control space.

Proposition 1 (Router gauge invariance). For any $R \in \mathrm { O } ( r )$ , define

$$
U _ { \ell } ^ { \prime } = U _ { \ell } R , \qquad B _ { \ell } ^ { \prime } = B _ { \ell } R , \qquad x _ { \ell , t } ^ { \prime } = R ^ { \top } x _ { \ell , t } .\tag{4}
$$

Then

$$
\boldsymbol { B } _ { \ell } ^ { \prime } \boldsymbol { x } _ { \ell , t } ^ { \prime } = \boldsymbol { B } _ { \ell } \boldsymbol { x } _ { \ell , t }\tag{5}
$$

for every token t. Thus, an orthogonal change ofbasis changes the control coordinates but leaves the router output unchanged.

Therefore, the particular coordinates $X _ { \ell }$ are not unique: any rotated representation $X _ { \ell } R$ describes the same router-control state. We collect all such representations into the equivalence class

$$
[ X _ { \ell } ] = \{ X _ { \ell } R : R \in \operatorname { O } ( r ) \} .\tag{6}
$$

Cross-layer comparison should therefore ignore this arbitrary choice of basis. A natural way to do so is the orthogonal Procrustes distance [9],

$$
d _ { \mathrm { P } } ( [ X ] , [ Y ] ) = \operatorname* { m i n } _ { R \in \mathrm { O } ( r ) } \| X R - Y \| _ { F } ,\tag{7}
$$

which first finds the best orthogonal alignment and then measures the remaining discrepancy. This optimization has the closed form

$$
d _ { \mathrm { P } } ^ { 2 } = \| X \| _ { F } ^ { 2 } + \| Y \| _ { F } ^ { 2 } - 2 \| X ^ { \top } Y \| _ { * } .\tag{8}
$$

Thus, Procrustes alignment resolves the orthogonal basis ambiguity inherent in the router-control factorization and enables meaningful cross-layer comparison.

## 3.3 A globally consistent canonical gauge

Pairwise Procrustes fits need not define a single coordinate system across depth. We therefore use generalized Procrustes analysis (GPA). On training tokens, after layer-wise centering and scalar normalization, we solve

$$
\operatorname* { m i n } _ { \{ Q _ { \ell } \} , M } \sum _ { \ell = 1 } ^ { L } \| X _ { \ell } Q _ { \ell } - M \| _ { F } ^ { 2 } , \qquad Q _ { \ell } \in \mathrm { O } ( r ) .\tag{9}
$$

The canonical state of a column vector $x _ { \ell , t }$ is then

$$
z _ { \ell , t } = Q _ { \ell } ^ { \top } \mathrm { n o r m } ( x _ { \ell , t } ) .\tag{10}
$$

For fixed aligned clouds $Z _ { \ell } ~ = ~ X _ { \ell } Q _ { \ell }$ , the optimal template is their mean $\begin{array} { r } { M ^ { * } ~ = ~ L ^ { - 1 } \sum _ { \ell } Z _ { \ell } } \end{array}$ Moreover,

$$
\sum _ { \ell } \| Z _ { \ell } - M ^ { * } \| _ { F } ^ { 2 } = \frac { 1 } { L } \sum _ { \ell < m } \| Z _ { \ell } - Z _ { m } \| _ { F } ^ { 2 } ,\tag{11}
$$

so GPA can equivalently be viewed as minimizing total pairwise disagreement among layer gauges.   
All $Q _ { \ell }$ are fitted on training sequences only.

## 3.4 Shared dynamics as gauge reduction

Suppose raw control coordinates admit layer-specific affine transitions. Ignoring the fixed normalization for notational clarity,

$$
\boldsymbol { x } _ { \ell + 1 } \approx F _ { \ell } \boldsymbol { x } _ { \ell } + \boldsymbol { c } _ { \ell } , \qquad \boldsymbol { z } _ { \ell } = \boldsymbol { Q } _ { \ell } ^ { \top } \boldsymbol { x } _ { \ell } \implies \boldsymbol { x } _ { \ell } = \boldsymbol { Q } _ { \ell } \boldsymbol { z } _ { \ell } ,\tag{12}
$$

where the last relation follows from the orthogonality of $Q _ { \ell }$

After the canonicalization in Eq. (10), the corresponding transition becomes

$$
z _ { \ell + 1 } = Q _ { \ell + 1 } ^ { \top } x _ { \ell + 1 }\tag{13}
$$

$$
\approx Q _ { \ell + 1 } ^ { \top } ( F _ { \ell } x _ { \ell } + c _ { \ell } )\tag{14}
$$

$$
= Q _ { \ell + 1 } ^ { \top } F _ { \ell } Q _ { \ell } z _ { \ell } + Q _ { \ell + 1 } ^ { \top } c _ { \ell } .\tag{15}
$$

Our central dynamical hypothesis is therefore precise: there exists a low-capacity operator A such that

$$
Q _ { \ell + 1 } ^ { \top } F _ { \ell } Q _ { \ell } \approx A \quad \mathrm { f o r m a n y l a y e r s } ~ \ell ,\tag{16}
$$

or equivalently $F _ { \ell } \approx Q _ { \ell + 1 } A Q _ { \ell } ^ { \intercal }$ . Empirically, we fit

$$
\hat { z } _ { \ell + 1 , t } = A z _ { \ell , t } + b\tag{17}
$$

by ridge regression on pooled layer transitions, with regularization selected on validation data. Layer-specific affine models provide an upper bound. We summarize the degree of reuse with

$$
\eta _ { \mathrm { s h a r e } } = \frac { R _ { \mathrm { s h a r e d } } ^ { 2 } } { R _ { \mathrm { l a y e r - s p e c i f i c } } ^ { 2 } } ,\tag{18}
$$

which equals one when a single transition is as predictive as fitting each layer independently.

## 3.5 Routing stability under transport

To evaluate a predicted canonical state, we first map it back to the expert logits of the target layer. Ignoring the fixed normalization transform, which can be absorbed into the decoder and a bias, we have $x _ { \ell } = Q _ { \ell } z _ { \ell }$ and therefore

$$
\begin{array} { r } { \widetilde { g } _ { \ell , t } = B _ { \ell } x _ { \ell , t } = B _ { \ell } Q _ { \ell } z _ { \ell , t } = D _ { \ell } z _ { \ell , t } , \qquad D _ { \ell } : = B _ { \ell } Q _ { \ell } . } \end{array}\tag{19}
$$

Thus, $D _ { \ell }$ is the layer-specific decoder from canonical routing coordinates to centered expert logits. Now let $\hat { z } = z + e$ be a transported or predicted canonical state, where e is its state-space error. Its decoded logits satisfy

$$
\begin{array} { r } { D _ { \ell } \hat { z } = D _ { \ell } z + D _ { \ell } e , } \end{array}\tag{20}
$$

so $D _ { \ell } e$ is exactly the induced error in the centered logits. Let

$$
\widetilde { g } _ { ( 1 ) } \geq \widetilde { g } _ { ( 2 ) } \geq \cdot \cdot \cdot \geq \widetilde { g } _ { ( E ) }
$$

denote the sorted true logits, and define the top-k routing margin

$$
\gamma _ { k } = \widetilde { g } _ { ( k ) } - \widetilde { g } _ { ( k + 1 ) } .\tag{21}
$$

Proposition 2 (Top-k stability). If

$$
\| D _ { \ell } e \| _ { \infty } < \frac { \gamma _ { k } } { 2 } ,\tag{22}
$$

then the selected expert set is unchanged:

$$
\mathrm { T o p K } ( D _ { \ell } \hat { z } ) = \mathrm { T o p K } ( D _ { \ell } z ) .\tag{23}
$$

A sufficient condition directly in canonical state space is

$$
\| e \| _ { 2 } < \frac { \gamma _ { k } } { 2 \| D _ { \ell } \| _ { 2 \to \infty } } , \qquad \| D _ { \ell } \| _ { 2 \to \infty } = \operatorname* { m a x } _ { j \in [ E ] } \| d _ { \ell , j } \| _ { 2 } .\tag{24}
$$

Proposition 2 connects canonical-state prediction error to the discrete routing decision. The same state error can leave routing unchanged when the top-k margin is large, but change the selected experts for a token close to the routing boundary. Geometric prediction quality alone therefore does not determine routing stability. Proofs of Propositions 1 and 2, together with the GPA identity, are given in Appendix A.

## 4 Experimental Setup

We evaluate four sparse MoE architectures chosen to vary depth, number of experts, routing sparsity, and the presence of explicit shared experts (Table 1).

Table 1: Evaluated MoE architectures. Rank is the numerical rank of the centered router.
<table><tr><td>Model</td><td>Layers</td><td>Experts</td><td>Top-k</td><td>Rank</td></tr><tr><td>Granite</td><td>24</td><td>32</td><td>8</td><td>31</td></tr><tr><td>OLMoE-SFT</td><td>16</td><td>64</td><td>8</td><td>63</td></tr><tr><td>Phi-tiny</td><td>32</td><td>16</td><td>2</td><td>15</td></tr><tr><td>IBM Shared</td><td>40</td><td>62</td><td>6</td><td>61</td></tr></table>

Data and splits. We use WikiText-2 raw text as a common natural-language probe corpus. Geometry, alignment baselines, and dynamics ablations use three independent sequence-level splits (seeds 101, 202, 303), each with 16,384 held-out tokens. Final matched-readout validation and targeted causal replications use 50,176 held-out tokens per seed (392 sequences of length 128). Normalization, GPA, dynamics, and readout fitting use training data only, while hyperparameters are selected on validation data.

Metrics. We evaluate geometry using held-out same-token cosine similarity and linear CKA. Dynamics are measured with $R ^ { 2 }$ , normalized MSE, and cosine similarity. Routing fidelity is measured with top-1 accuracy, top-k recall and Jaccard, Jensen–Shannon divergence, and centered-logit MSE. We evaluate causal interventions using ∆NLL.

## 5 Results

We present the results in five parts. First, we test whether aligning router-control states across layers makes their evolution more predictable with a single shared transition. Second, we compare routercontrol states with the residual stream to determine whether this predictability is specific to routing or simply reflects general similarity between nearby layers. Third, we study how the dimensionality of the representation affects predictability and routing accuracy. Fourth, we test whether the aligned states can replace the original routing states without strongly affecting model performance. Finally, we examine where this shared structure breaks down and clarify the limits of our claim.

## 5.1 Gauge alignment exposes reusable dynamics

A shared transition is weak in native coordinates and strong after gauge fixing. Across architectures, the pooled linear transition remains weak in raw coordinates and under generic alignment baselines, but becomes substantially more predictive after orthogonal Procrustes alignment (Fig. 2). Random orthogonal gauges and PCA-basis alignment do not produce the same improvement, and shuffling the correspondence between tokens removes the alignment effect almost entirely (full ablation in Table 7). This suggests that the improvement comes from aligning meaningful cross-layer structure rather than simply applying an additional orthogonal transformation.

![](images/86e7e5d588f6e0241c32005ebbf593e2c45fb4be2df58d4bedc169d81e6127bb.jpg)  
(a) Alignment baselines

![](images/2e1f3bbdc784fc68d0cb066f31e06709176a67c7ff4989908debc25e4b717d07.jpg)  
(b) Shared vs. layer-specific dynamics  
Figure 2: Gauge fixing exposes reusable cross-layer dynamics. (a) Shared prediction remains weak under raw, PCA-basis, random-gauge, and CCA coordinates, then rises sharply after orthogonal Procrustes alignment. (b) One learned transition closes most of the gap from persistence to separately fitted layer-specific dynamics. Error bars show standard deviation across three split seeds.

One shared transition explains most of the predictable cross-layer change. We next compare the shared transition with a stronger baseline that fits a separate transition for every pair of adjacent layers. As shown in Table 2, the shared model reaches 79–90% of the $R ^ { 2 }$ obtained by these layer-specific models across all four architectures, while using far fewer parameters. This shows that the predictable evolution of router-control states is not strongly specific to individual layers. After alignment, much of this evolution can instead be captured by the same transition across depth.

Table 2: Cross-model geometry and dynamics. Values are means over three split seeds. $\mathbf { \dot { \Omega } } ^ { 6 6 } \mathbf { I d } . \mathbf { \Omega } ^ { \mathrm { * } }$ is canonical-state persistence, “Shared” is one pooled linear transition, and “Layer” fits a separate linear transition per depth. $\eta _ { \mathrm { s h a r e } } = R _ { \mathrm { S h a r e d } } ^ { 2 } \dot { / } R _ { \mathrm { L a y e r } } ^ { 2 }$ . Params are shared/layer-specific transition parameters.
<table><tr><td>Model</td><td>GPA cos.</td><td>Id.  $R ^ { 2 }$ </td><td>Shared  $R ^ { 2 }$ </td><td>Layer  $R ^ { 2 }$ </td><td>ηshare</td><td>Params S/L</td></tr><tr><td>Granite</td><td>0.581</td><td>0.378</td><td>0.544</td><td>0.620</td><td>0.88</td><td>0.99k / 22.8k</td></tr><tr><td>OLMoE</td><td>0.548</td><td>0.284</td><td>0.486</td><td>0.553</td><td>0.88</td><td>4.03k / 60.5k</td></tr><tr><td>Phi-tiny</td><td>0.534</td><td>0.613</td><td>0.707</td><td>0.788</td><td>0.90</td><td>0.24k / 7.44k</td></tr><tr><td>IBM Shared</td><td>0.431</td><td>0.138</td><td>0.385</td><td>0.486</td><td>0.79</td><td>3.78k / 147.5k</td></tr></table>

## 5.2 Residual smoothness is not the same as routing specificity

A possible alternative explanation is that canonical router states are predictable simply because hidden representations change smoothly across nearby layers. To test this, we compare them with matched-rank PCA and random residual subspaces. For each representation, we train a decoder with the same parameter budget to reconstruct the true router logits on held-out tokens.

Table 3 shows a clear difference. Residual PCA is much easier to predict across layers, while routercontrol states recover the actual expert choices much more accurately. This shows that cross-layer predictability and routing relevance are different properties: residual representations evolve more smoothly, but router-control states retain the information that is directly used for expert selection.

Table 3: Matched-rank, equal-budget readout comparison. Means over three 50,176-token test splits. Higher is better for both metrics.
<table><tr><td></td><td colspan="2">Next-state  $R ^ { 2 }$ </td><td colspan="2">Top-k recall</td></tr><tr><td>Model</td><td>Router</td><td>PCA</td><td>Router</td><td>PCA</td></tr><tr><td>Granite</td><td>0.542</td><td>0.855</td><td>0.999</td><td>0.798</td></tr><tr><td>OLMoE</td><td>0.500</td><td>0.788</td><td>0.998</td><td>0.669</td></tr><tr><td>Phi-tiny</td><td>0.424</td><td>0.861</td><td>0.989</td><td>0.405</td></tr><tr><td>IBM Shared</td><td>0.371</td><td>0.806</td><td>0.998</td><td>0.589</td></tr></table>

## 5.3 A low-dimensional shared core coexists with routing detail

The rank sweep shows that the most predictable representation is not necessarily the one that best preserves routing behavior. OLMoE provides the clearest example (Table 4): very low-rank states are highly predictable, while higher ranks are needed to recover expert choices accurately and to improve causal transport. This suggests that a small shared component captures much of the cross-layer dynamics, while additional dimensions preserve finer routing information.

Table 4: OLMoE rank trade-off on the screening split (seed 101). Lower causal ∆NLL and higher R<sup>2</sup>/recall are better. The full centered-router rank is 63.
<table><tr><td>Rank</td><td>Shared  $R ^ { 2 }$ </td><td>Top-k recall</td><td>Causal ∆NLL</td></tr><tr><td>2</td><td>0.695</td><td>0.316</td><td>0.593</td></tr><tr><td>8</td><td>0.614</td><td>0.492</td><td>0.309</td></tr><tr><td>16</td><td>0.585</td><td>0.568</td><td>0.230</td></tr><tr><td>31</td><td>0.541</td><td>0.638</td><td>0.176</td></tr><tr><td>63</td><td>0.487</td><td>0.699</td><td>0.118</td></tr></table>

In centered coordinates, the unregularized least-squares shared operator is $A ^ { * } = C _ { 1 0 } C _ { 0 0 } ^ { \dagger }$ , where $C _ { 1 0 } = \mathbb { E } [ z _ { \ell + 1 } z _ { \ell } ^ { \top } ]$ and $C _ { 0 0 } = \mathbb { E } [ z _ { \ell } z _ { \ell } ^ { \top } ]$ . The rank sweep therefore measures how many control dimensions are needed to preserve predictable cross-layer structure and how many are needed for accurate routing decisions. The IBM shared-expert model is less compressible: both predictability and routing fidelity continue to improve as rank increases (Fig. 3).

![](images/f7fe9561cf371f4a48c499ec3daaf52e110c28dcc0f375607a626f9ee968d84b.jpg)  
(a) Shared-dynamics $R ^ { 2 }$

![](images/eeb17259ff232680996c7611710c91ba770c6b4e777ddffa1cb451bdaaa807bc.jpg)  
(b) Top-k routing recall  
Figure 3: A low-dimensional shared structure captures much of the predictable dynamics, while accurate routing requires more dimensions. OLMoE shows this most clearly: low-rank states are easy to predict, but higher ranks are needed to recover expert choices accurately. Rank is normalized by each model’s full centered-router rank.

## 5.4 Causal transport tests functional relevance

Good prediction alone does not show that the aligned states are actually important for model behavior. We therefore replace the native routing states at selected layers with transported states and measure the resulting change in ∆NLL.

Canonicalization is important for causal transport. For Granite layers 9–13, the shared transition causes much less degradation in canonical coordinates than in raw coordinates, with shuffled gauges, or in a random matched-rank subspace (Table 5). Results are averaged over three 50,176-token splits.

Table 5: Granite causal controls, layers 9–13. Mean ∆NLL ± standard deviation across seeds 101/202/303. Lower is better.
<table><tr><td>Transport variant</td><td>∆NLL</td></tr><tr><td>Canonical + shared A</td><td> $0 . 0 0 7 0 \pm 0 . 0 0 1 6$ </td></tr><tr><td>Canonical + identity</td><td> $0 . 0 1 4 2 \pm 0 . 0 0 3 2$ </td></tr><tr><td>Raw coordinates + shared A</td><td> $0 . 1 1 2 0 \pm 0 . 0 0 4 6$ </td></tr><tr><td>Shuffled Q + shared A</td><td> $0 . 1 0 1 2 \pm 0 . 0 0 6 0$ </td></tr><tr><td>Random subspace + shared A</td><td> $0 . 1 4 9 5 \pm 0 . 0 0 9 9$ </td></tr></table>

Learned dynamics outperforms persistence across replications. Table ?? shows consistent gain for OLMoE layers 8–9 and both long Phi horizons. Degradation increases with transport length, while the advantage over persistence becomes more pronounced (Appendix ??). The substantial Phi degradation suggests systematic router dynamics, but not a practical router-skipping method.

Table 6: Targeted causal replications. Means over three fixed 50,176-token splits. The final column is Shared minus Identity; negative values favor learned dynamics.
<table><tr><td>Model</td><td>Block</td><td>Identity</td><td>Shared</td><td>Shared—Id.</td></tr><tr><td>OLMoE</td><td>8-9</td><td>0.0375</td><td>0.0316</td><td>-0.0059</td></tr><tr><td>Phi-tiny</td><td>24-31</td><td>0.2670</td><td>0.2520</td><td>-0.0150</td></tr><tr><td>Phi-tiny</td><td>22-31</td><td>0.3217</td><td>0.3017</td><td>-0.0200</td></tr></table>

The IBM shared-expert model provides an important counterexample to over-generalization. It still exhibits canonical geometry and shared predictability, and selected short blocks can be transported without measurable degradation in replicated screening tests, but learned A is not uniformly better than identity across horizons. The benefit of explicitly evolving the canonical state is therefore architecture- and horizon-dependent.

## 5.5 Negative results define the limits

The negative results show where the shared structure stops being useful. First, directly averaging or sharing router weights strongly degrades model quality, meaning that aligned control states do not make expert identities or router readouts interchangeable across layers. Second, repeatedly skipping routers causes errors to accumulate quickly. We therefore limit our claim to localfunctional transport and reusable cross-layer dynamics.

## 6 Discussion

The experiments support a three-level picture. The residual state $h _ { \ell }$ contains a smooth cross-layer backbone. The control state $x _ { \ell } = U _ { \ell } ^ { \top } h _ { \ell }$ isolates directions that can alter relative expert logits. Orthogonal gauge fixing maps these layer-specific coordinates to $z _ { \ell } ,$ , where a low-capacity transition becomes reusable across depth.

The matched-readout result is central to this interpretation. If our observation were only generic residual smoothness, residual PCA should be an equally good routing representation. It is not. PCA predicts future residual state more accurately, while router-control coordinates preserve expert choices much more faithfully. This separates two properties that are easy to conflate: temporal predictability and routing relevance.

Our results also make the phrase “shared dynamics” more precise. We do not find a universal transition that makes all layers dynamically identical. Rather, Eq. (16) says that apparently different raw transitions are approximately related by layer-specific gauge changes, and one shared transition captures most of the predictive power of independently fitted layer-specific transitions with far fewer parameters. Its causal advantage over identity is clearest at longer horizons on Phi and OLMoE; the explicit shared-expert architecture exhibits weaker universality. The evidence therefore supports reusable cross-layer structure, not exact dynamical equivalence. Proposition 2 further separates representation error from routing error: functional stability depends on the transported-state error measured through the logit decoder relative to the local top-k margin.

Finally, the findings complement cross-layer prefetching work. Prefetching systems exploit futureexpert predictability operationally [4, 13, 15]; our experiments provide a geometric coordinate system in which a simple shared predictor becomes effective. Conversely, the strong residual-PCA baseline shows that cross-layer predictability is broader than the router subspace itself.

## 7 Limitations

The study is empirical and uses one common text corpus for controlled cross-model comparison. Phi-tiny-MoE is instruction-tuned and therefore distribution-mismatched to WikiText perplexity; we use its causal results to compare interventions rather than claim improved language modeling. The IBM shared-expert checkpoint is a research checkpoint rather than a widely deployed production model. The exact symmetry identifies an equivalence class rather than a unique latent coordinate system: GPA fixes one convenient global gauge, but the canonical state remains globally identifiable only up to a common orthogonal transform. Finally, causal transport is used as an analysis tool: recursive open-loop skipping accumulates error, and we make no wall-clock speedup claim.

## 8 Conclusion

Across four sparse MoE architectures, we find that router-control states contain consistent cross-layer structure that becomes visible after orthogonal gauge alignment. In the canonical space, a single linear transition captures most of the predictive power of layer-specific models while using far fewer parameters. This effect is not explained by generic hidden-state smoothness: residual states are often easier to predict, but router-control states preserve expert choices much more accurately. Causal interventions further show that these aligned states remain functionally meaningful, with learned evolution outperforming persistence at longer horizons on Phi and OLMoE. Together, these results suggest that MoE routers share routing-relevant structure and partially reusable dynamics across depth.

## References

[1] Akshit Achara, Tatiana Gaintseva, Mateo Mahaut, Pritish Chakraborty, Viktor Stenby Johansson, Melih Barsbey, Emanuele Rodola, and Donato Crisostomi. Multi-way representation alignment,\` 2026. URL https://arxiv.org/abs/2602.06205.

[2] Sagi Ahrac, Noya Hochwald, and Mor Geva. Routers learn the geometry of their experts: Geometric coupling in sparse mixture-of-experts, 2026. URL https://arxiv.org/abs/ 2605.12476.

[3] Nishant Suresh Aswani, Saif Jabari, and Muhammad Shafique. Representing neural network layers as linear operations via koopman operator theory. In Proceedings of UniReps: the Third Edition of the Workshop on Unifying Representations in Neural Models, volume 322 of Proceedings of Machine Learning Research, pages 70–80. PMLR, 2026. URL https: //proceedings.mlr.press/v322/aswani26a.html.

[4] Zhiyuan Fang, Xingfan Yu, Yuegui Huang, Zicong Hong, Yufeng Lyu, Wuhui Chen, Yue Yu, and Fan Yu. Fate: Fast edge inference of mixture-of-experts models via cross-layer gate. In Proceedings of the ACM Web Conference 2026, 2026. URL https://arxiv.org/abs/2502. 12224.

[5] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23 (120):1–39, 2022. URL https://www.jmlr.org/papers/v23/21-0998.html.

[6] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 3519–3529. PMLR, 2019. URL https://proceedings.mlr.press/v97/kornblith19a.html.

[7] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Evan Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. OLMoE: Open mixtureof-experts language models. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 9b224ace8963c9385ad5e2b5c9039b97-Abstract-Conference.html.

[8] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. In International Conference on Learning Representations, 2017. URL https://arxiv.org/abs/1701.06538.

[9] Alex H. Williams, Erin Kunz, Simon Kornblith, and Scott W. Linderman. Generalized shape metrics on neural representations. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 252a3dbaeb32e7690242ad3b556e626b-Abstract.html.

[10] Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. OpenMoE: An early effort on open mixture-of-experts language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 55625–55655. PMLR, 2024. URL https://proceedings.mlr. press/v235/xue24c.html.

[11] Charles Ye, Bo Yuan, and Lee Sharkey. Polysemantic experts, monosemantic paths: Routing as control in MoEs, 2026. URL https://arxiv.org/abs/2604.17837.

[12] Youngsik Yoon, Siwei Wang, Wei Chen, and Jungseul Ok. When are experts misrouted? counterfactual routing analysis in mixture-of-experts language models, 2026. URL https: //arxiv.org/abs/2605.07260.

[13] Yingnan Zhao, Razvan Bunescu, Ahmed Louri, Avinash Karanth, and Ke Wang. A spatiotemporal expert prefetching framework for efficient MoE-based LLM inference, 2026. URL https://arxiv.org/abs/2606.15453.

[14] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Y. Zhao, Andrew M. Dai, Zhifeng Chen, Quoc V. Le, and James Laudon. Mixture-of-experts with expert choice routing. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html.

[15] Shien Zhu, Samuel Bohl, Robin Oester, and Gustavo Alonso. Pre-attention expert prediction and prefetching for mixture-of-experts large language models, 2025. URL https://arxiv. org/abs/2511.10676.

## A Mathematical Details

Proof of Proposition 1. For $R \in \mathrm { O } ( r )$

$$
B _ { \ell } ^ { \prime } x _ { \ell } ^ { \prime } = ( B _ { \ell } R ) ( R ^ { \top } x _ { \ell } ) = B _ { \ell } ( R R ^ { \top } ) x _ { \ell } = B _ { \ell } x _ { \ell } .\tag{25}
$$

Thus all centered logits, softmax probabilities, and top-k decisions are identical. The same argument applies token-wise to a data matrix $X _ { \ell } ,$ whose coordinate representation transforms as $X _ { \ell } \mapsto X _ { \ell } R .$

Procrustes closed form. For centered $X , Y$ ,

$$
\begin{array} { r l } & { \underset { R \in \mathsf { O } ( r ) } { \operatorname* { m i n } } \| X R - Y \| _ { F } ^ { 2 } = \| X \| _ { F } ^ { 2 } + \| Y \| _ { F } ^ { 2 } - 2 \underset { R \in \mathsf { O } ( r ) } { \operatorname* { m a x } } \mathrm { t r } ( R ^ { \top } X ^ { \top } Y ) } \\ & { \qquad = \| X \| _ { F } ^ { 2 } + \| Y \| _ { F } ^ { 2 } - 2 \| X ^ { \top } Y \| _ { * } , } \end{array}\tag{26}
$$

where the maximum is attained by the standard orthogonal Procrustes solution obtained from the SVD of $X ^ { \top } Y$

GPA pairwise-dispersion identity. Let $\begin{array} { r } { M ^ { * } = L ^ { - 1 } \sum _ { \ell } Z _ { \ell } } \end{array}$ . Expanding squared distances gives

$$
\sum _ { \ell } \| Z _ { \ell } - M ^ { * } \| _ { F } ^ { 2 } = \sum _ { \ell } \| Z _ { \ell } \| _ { F } ^ { 2 } - L \| M ^ { * } \| _ { F } ^ { 2 } = \frac { 1 } { L } \sum _ { \ell < m } \| Z _ { \ell } - Z _ { m } \| _ { F } ^ { 2 } .\tag{27}
$$

Hence minimizing the GPA objective minimizes the average pairwise disagreement among all aligned layer representations.

Proof of Proposition 2. Let i be any selected expert and j any unselected expert. By definition of the top-k margin, $g _ { i } - g _ { j } \ge \gamma _ { k }$ . With logit perturbation $\delta g = D _ { \ell } e$

$$
\hat { g } _ { i } - \hat { g } _ { j } \ge \gamma _ { k } - | \delta g _ { i } | - | \delta g _ { j } | > 0\tag{28}
$$

whenever $\| \delta g \| _ { \infty } < \gamma _ { k } / 2$ . Therefore no selected expert can cross an unselected expert, so the top-k set is unchanged. Finally, $\| D _ { \ell } e \| _ { \infty } \leq \| D _ { \ell } \| _ { 2 \to \infty } \| e \| _ { 2 }$ yields Eq. (24).

Least-squares shared operator. Ignoring the intercept after centering, the pooled objective is $\mathbb { E } \lVert z _ { \ell + 1 } \dot { - } A z _ { \ell } \rVert _ { 2 } ^ { 2 }$ . The minimum-norm solution is

$$
A ^ { * } = C _ { 1 0 } C _ { 0 0 } ^ { \dagger } , \qquad C _ { 1 0 } = \mathbb { E } [ z _ { \ell + 1 } z _ { \ell } ^ { \top } ] , \quad C _ { 0 0 } = \mathbb { E } [ z _ { \ell } z _ { \ell } ^ { \top } ] .\tag{29}
$$

This makes explicit that the shared dynamics are determined by cross-layer covariance expressed in the canonical gauge.

## B Additional Experimental Details

Seed-level targeted causal replications. To characterize how transport degrades with distance, we sweep the number of consecutive transported routers over screened contiguous windows. Figure 4 shows two complementary effects. Absolute ∆NLL generally increases with transport horizon, indicating error accumulation under longer interventions. At the same time, the learned shared transition becomes increasingly competitive with identity persistence at longer horizons, particularly for OLMoE and Phi-tiny. Stars mark the fixed three-seed replications reported in Table 6.

![](images/34c887cc8786751b09871ccfd438bc128a126fb00377a2beb4b6c16fc335ed08.jpg)  
(a) Transport horizon vs. ∆NLL

![](images/e4bba2d29ea198175e1a9157bb28dee92e194bc1cc6b17a87c5ce7315aa0cff8.jpg)  
(b) Learned dynamics vs. persistence  
Figure 4: Causal transport has a clear horizon structure. (a) Median ∆NLL over screened contiguous windows with interquartile ranges. (b) Shared A minus identity ∆NLL; values below zero favor learned dynamics, and stars mark independently replicated three-seed points.

Reconstruction checks. Before every causal run, centered router reconstruction is verified at full numerical rank. Across all layers and models, reconstructed top-k sets agree exactly with native routing. Evaluation scripts also verify model-weight checksums before and after interventions.

Complete alignment controls. Table 7 reports the alignment baselines used in Fig. 2. Values are mean shared-dynamics $R ^ { 2 }$ across the three split seeds. “Random $Q ^ { , , }$ applies independently sampled orthogonal gauges; “PCA basis” aligns via PCA coordinate conventions rather than token-wise Procrustes.

Table 7: Complete alignment ablation. Mean shared-dynamics $R ^ { 2 }$ across three split seeds.
<table><tr><td>Model</td><td>Raw</td><td>PCA basis</td><td>Random Q</td><td>CCA</td><td>Procrustes</td></tr><tr><td>Granite</td><td>0.071</td><td>0.030</td><td>0.081</td><td>0.198</td><td>0.544</td></tr><tr><td>OLMoE</td><td>0.053</td><td>0.034</td><td>0.091</td><td>0.161</td><td>0.486</td></tr><tr><td>Phi-tiny</td><td>0.220</td><td>0.108</td><td>0.224</td><td>0.202</td><td>0.707</td></tr><tr><td>IBM Shared</td><td>0.033</td><td>0.011</td><td>0.033</td><td>0.106</td><td>0.385</td></tr></table>

Matched readout. Router-control and residual-PCA representations use learned linear logit decoders with the same per-layer parameterization (r + 1)E. Exact cross-model means are reported in Table 3; random matched-rank residual subspaces are additionally used as a negative control in the experiment code.

Seed-level targeted causal replications. Table 8 expands Table 6. The blocks are fixed before evaluation on the additional split seeds; they are not re-selected per seed.

Table 8: Seed-level targeted causal replications. Lower ∆NLL is better; Shared−Id. < 0 favors learned dynamics.
<table><tr><td>Model</td><td>Block</td><td>Seed</td><td>Identity</td><td>Shared</td><td>Shared—Id.</td></tr><tr><td>OLMoE</td><td>8-9</td><td>101</td><td>0.0386</td><td>0.0326</td><td>-0.0060</td></tr><tr><td></td><td></td><td>202</td><td>0.0457</td><td>0.0418</td><td>-0.0039</td></tr><tr><td></td><td></td><td>303</td><td>0.0281</td><td>0.0204</td><td>-0.0077</td></tr><tr><td>Phi-tiny</td><td>24-31</td><td>101</td><td>0.2552</td><td>0.2471</td><td>-0.0080</td></tr><tr><td></td><td></td><td>202</td><td>0.2819</td><td>0.2654</td><td>-0.0166</td></tr><tr><td></td><td></td><td>303</td><td>0.2639</td><td>0.2434</td><td>-0.0206</td></tr><tr><td>Phi-tiny</td><td>22-31</td><td>101</td><td>0.3112</td><td>0.3001</td><td>-0.0110</td></tr><tr><td></td><td></td><td>202</td><td>0.3378</td><td>0.3120</td><td>-0.0258</td></tr><tr><td></td><td></td><td>303</td><td>0.3161</td><td>0.2930</td><td>-0.0231</td></tr></table>

Rank sweeps. Rank truncation is applied before gauge alignment. Table 4 shows representative OLMoE points and Fig. 3 the full cross-model curves. We treat the sweep as structural evidence rather than a final benchmark.