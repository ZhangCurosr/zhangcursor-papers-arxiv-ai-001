# FAIRMEAN: PROMOTING FAIRNESS IN DISTRIBUTED LEARNING UNDER LABEL POISONING ATTACKS

Huigan Zheng<sup>1,2</sup>, Jiaojiao Zhang<sup>3</sup>, Yongxiang Liu<sup>2</sup>

<sup>1</sup>Sun Yat-Sen University, <sup>2</sup>Pengcheng Laboratory, <sup>3</sup>Great Bay University

## ABSTRACT

Fairness-aware distributed learning prioritizes clients with large losses to reduce performance disparities, but label poisoning can create large losses, thereby inducing a fairness– robustness conflict. We propose FairMean to manage this conflict. FairMean weights client gradients using a bounded, nondecreasing function of local loss. The increasing weights prioritize high-loss clients to promote fairness, while the upper bound prevents excessive loss-induced amplification of poisoned-client gradients. In the absence of label poisoning, we show that minimizing the FairMean objective is more conducive to solution fairness than minimizing the standard average-loss objective. Under label poisoning, we establish an average-stationarity bound whose attack-dependent term is proportional to the square of the poisoned-client fraction. Experiments show that FairMean promotes fairness by reducing accuracy variance while improving worst-client accuracy.

Index Terms— distributed learning, fairness, label poisoning

## 1. INTRODUCTION

Distributed learning trains a global model from the private data of multiple clients through the coordination of a central server [1–3]. The standard formulation minimizes the average of the local losses. When client data are heterogeneous, the jointly trained model may perform well for most clients but poorly for a minority of clients [4, 5]. Because a larger client loss generally indicates poorer current performance, fairnessaware formulations prioritize clients with large losses to narrow performance disparities across clients [6].

Fairness-aware methods mainly follow two optimization routes. The first transforms or reweights client losses: q-FFL [7] uses a power transformation, DRFL [8] adapts aggregation weights to client losses, and TERM [9] applies exponential tilting. The second treats local losses as multiple objectives. FedMDFG [10] and FedLF [11] introduce dynamically adjusted fairness guidance objectives. AdaFed [12] rescales client gradients before computing a common descent direction, while FairMOO [13] alternates fairness reduction with constrained multi-objective descent.

Label poisoning attacks corrupt the local labels of a subset of clients while leaving their features and the prescribed protocol unchanged [14, 15]; poisoned clients thus transmit misleading updates that raise their local losses and may degrade the global model on clean data. Label poisoning can be viewed as a special case of Byzantine attacks, where poisoned clients follow the prescribed protocol rather than transmitting arbitrary messages [16]; robust aggregators designed for the latter screen updates via robust statistics or outlier detection [17–20], but as we discuss below, such screening can be ineffective under label poisoning on heterogeneous data.

Recent studies incorporate robustness mechanisms into fairness-aware learning. For example, the q-FFL instantiation of H-nobs combines a fairness objective with post-hoc norm-based screening [7,21]. For $q > 0$ , the q-FFL multiplier increases without an explicit upper bound, whereas screening in H-nobs can discard fairness-enhancing gradients under heterogeneity. FedMGDA+ [22] instead normalizes client updates before computing a common descent direction for multiple objectives. Neither mechanism directly resolves the fairness–poisoning conflict: screening may discard atypical regular-client updates under heterogeneity, whereas normalization does not provide loss-dependent prioritization. Consequently, how fairness-aware methods behave when poisoned clients follow the prescribed computation and communication rules remains unclear.

Recent work [15] shows that the mean aggregator can outperform several robust aggregators under label poisoning attacks when client data are sufficiently heterogeneous. Intuitively, robust filtering may discard useful heterogeneous updates, whereas mean aggregation retains them. These observations favor retaining mean aggregation in heterogeneous settings. Plain mean aggregation, however, gives all client gradients equal weights and does not prioritize clients with large losses. Loss-dependent gradient scaling can provide such prioritization, but it creates an inherent conflict: underperforming regular clients and label-poisoned clients may both exhibit large losses. An unbounded scaling rule may therefore amplify gradients induced by corrupted labels. This raises a central question: how can we prioritize regular clients with large losses without excessively scaling gradients from poisoned clients?

To answer this question, we propose FairMean. Our contributions are: (i) a loss transformation with a nondecreasing and uniformly bounded marginal weight that prioritizes largeloss clients to promote fairness while limiting, rather than eliminating, extra loss-induced scaling of poisoned-client gradients, and a clean-case global-optimality guarantee for the proposed loss-dispersion measure; (ii) an explicit bound on the deviation from the gradient of the objective induced by the loss transformation over regular clients and a corresponding average-stationarity guarantee; and (iii) experiments on Fashion-MNIST and CIFAR-10 showing improved fairness and worst-client accuracy under both attacks, with competitive average accuracy.

![](images/9f2f7b4bb4ff38166dfe62da7505ee9c791d8e0864e1eb67c14b7aada5369fad.jpg)

![](images/cb506699d73d0c6652cd4500051e566fd9e6c261b9e10c4a33df3e2faf868519.jpg)  
Fig. 1. Loss transformation and marginal-weight function. (a) The increment $\phi _ { \kappa , \tau } ( z ) - z$ grows with $z , \mathbf { S } \mathbf { O }$ larger losses undergo a stronger transformation. (b) The marginal weight $a _ { \kappa , \tau } ( z )$ is nondecreasing and approaches $1 + \kappa .$ . Solid curves fix $\tau = 1$ and vary $\kappa ;$ the dashed curve uses $( \kappa , \tau ) = ( 1 , 2 )$ . For $\kappa = 1$ , the blue point $( z , a _ { \kappa , \tau } ) = ( 1 , 1 . 5 )$ on the $\tau = 1$ curve and the orange diamond (2, 1.5) on the $\tau = 2$ curve have the same marginal weight; increasing τ shifts the required loss from $z = 1 { \mathrm { ~ t o ~ } } z = 2 .$

## 2. PROBLEM FORMULATION

Consider n clients indexed by $\mathcal { N } = \{ 1 , \ldots , n \}$ and a global model $w \in \mathbb { R } ^ { d }$ . Client $i \in \mathcal N$ has a differentiable local loss $f _ { i } : \mathbb { R } ^ { d }  [ 0 , \infty )$ . Every client is either regular or poisoned. Let R and P denote the corresponding fixed but unknown index sets. We write $r : = | \mathcal { R } | \geq 1$ and $p : = | \mathcal { P } |$ , so that $r + p = n$ , and define the poisoned-client fraction as $\delta : = $ $p / n < 1$ . The nominal objective over the regular clients is

$$
\operatorname* { m i n } _ { w \in \mathbb { R } ^ { d } } F _ { { \mathcal R } } ( w ) , \quad F _ { { \mathcal R } } ( w ) : = \frac { 1 } { r } \sum _ { i \in { \mathcal R } } f _ { i } ( w ) .\tag{1}
$$

Definition 1. Following [15], client $i \in \mathcal { P }$ changes an $\mathit { a r } \mathrm { . }$ bitrary subset of its labels while leaving all other data unchanged. It computes $\widetilde { f _ { i } }$ on the relabeled data andfollows the prescribed protocol; it cannot transmit an arbitrary message.

The transmitted update therefore changes only through relabeling, although corrupted labels may still produce a large loss and a misleading gradient.

Definition 2. Following [7],fairness concerns the uniformity of model performance across clients. Under label poisoning, we assessfairness over the regular clients R.

Regular-client loss dispersion serves as a training-time proxy for performance disparity. Since a large regular-client loss often indicates poor current performance, assigning it a larger marginal weight can promote fairness. However, poisoned labels may also cause large losses, so unrestricted prioritization can amplify poisoned gradients. This conflict motivates Section 3.

## 3. PROPOSED FAIRMEAN

Label poisoning changes labels while leaving client features and the prescribed training procedure unchanged, yet it can produce large losses and misleading gradients. Since mean aggregation can outperform robust alternatives under such attacks in heterogeneous settings [15], we propose FairMean, which retains mean aggregation and introduces a bounded, nondecreasing marginal weight to prioritize large losses while limiting the amplification of poisoned-client gradients.

We seek a differentiable loss transformation whose derivative is (i) nondecreasing to prioritize large regular-client losses, (ii) uniformly bounded to cap extra loss-induced scaling of every client gradient, including gradients produced from poisoned labels, and (iii) Lipschitz continuous to avoid abrupt weight changes. These requirements lead to

$$
\begin{array} { r } { \phi _ { \kappa , \tau } ( z ) : = z + \kappa \left[ z - \tau \log ( 1 + z / \tau ) \right] , \quad \kappa , \tau > 0 . } \end{array}\tag{2}
$$

For analysis, define the regular-client objective induced by the loss transformation as

$$
\Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ) : = \frac { 1 } { r } \sum _ { i \in \mathcal { R } } \phi _ { \kappa , \tau } ( f _ { i } ( w ) ) .\tag{3}
$$

To characterize the gradient of $\Phi _ { \mathcal { R } } ^ { \kappa , \tau }$ , define the corresponding marginal-weight function as

$$
a _ { \kappa , \tau } ( z ) : = 1 + \kappa \frac { z } { z + \tau } .\tag{4}
$$

By the chain rule,

$$
\nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ) = \frac { 1 } { r } \sum _ { i \in \mathcal { R } } a _ { \kappa , \tau } ( f _ { i } ( w ) ) \nabla f _ { i } ( w ) .\tag{5}
$$

Thus, $a _ { \kappa , \tau } ( f _ { i } ( w ) )$ is client i’s marginal weight, namely, the scalar that multiplies $\nabla f _ { i } ( w )$ . For $z \ge 0 , \phi _ { \kappa , \tau } ( z ) \ge 0$ . The marginal weight and its derivative satisfy

$$
\begin{array} { l } { 1 \leq a _ { \kappa , \tau } ( z ) \leq 1 + \kappa , } \\ { 0 \leq a _ { \kappa , \tau } ^ { \prime } ( z ) = \displaystyle \frac { \kappa \tau } { ( z + \tau ) ^ { 2 } } \leq \displaystyle \frac { \kappa } { \tau } . } \end{array}
$$

Thus, $a _ { \kappa , \tau }$ is nondecreasing, bounded by $1 + \kappa ,$ and $\kappa / \tau \mathrm { . }$ Lipschitz continuous. Moreover, $a _ { \kappa , \tau } ( \tau ) = 1 + \kappa / 2 $ so increasing τ shifts the midpoint to a larger loss. The cap $1 + \kappa$ directly limits the additional loss-induced scaling.

To connect the loss transformation with fairness, define $\psi _ { \tau } ( z ) : = z - \tau \log ( 1 + z / \tau )$ and the dispersion term

$$
D _ { \tau } ( w ) : = \frac { 1 } { r } \sum _ { i \in \mathcal { R } } \psi _ { \tau } ( f _ { i } ( w ) ) - \psi _ { \tau } ( F _ { \mathcal { R } } ( w ) ) .
$$

Substituting the loss transformation in (2) into (3) shows that $\Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w )$ in (3) can be equivalently written as

$$
\begin{array} { r } { \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ) = \phi _ { \kappa , \tau } ( F _ { \mathcal { R } } ( w ) ) + \kappa D _ { \tau } ( w ) . } \end{array}\tag{6}
$$

The first term depends only on the mean regular-client loss, while $D _ { \tau } ( w ) \geq 0$ measures loss dispersion and vanishes exactly when all regular-client losses are equal.

Moreover, $\psi _ { \tau } ^ { \prime \prime } ( z ) = \tau / ( z + \tau ) ^ { 2 } . \mathrm { ~ I f ~ } 0 \leq f _ { i } ( w ) \leq B$ for all $i \in { \mathcal { R } } .$ , a second-order Taylor expansion around $F _ { \mathcal { R } } ( w )$ followed by averaging, cancels the linear terms and yields

$$
\frac { \tau } { 2 ( B + \tau ) ^ { 2 } } \mathrm { V a r } _ { \mathscr { R } } ( w ) \leq D _ { \tau } ( w ) \leq \frac { 1 } { 2 \tau } \mathrm { V a r } _ { \mathscr { R } } ( w ) ,\tag{7}
$$

where $\begin{array} { r } { \operatorname { V a r } _ { \mathcal { R } } ( w ) : = r ^ { - 1 } \sum _ { i \in \mathcal { R } } [ f _ { i } ( w ) - F _ { \mathcal { R } } ( w ) ] ^ { 2 } } \end{array}$ . Thus, minimizing the FairMean objective explicitly penalizes dispersion among regular-client losses, linking the loss transformation to fairness.

Algorithm 1 FairMean   
Require: Initial model $w ^ { 0 }$ , number of rounds T, stepsize   
$\alpha > 0 , \kappa > 0 ,$ , and $\tau > 0$   
1: for $t = 0 , \ldots , T - 1$ do   
2: Server broadcasts $w ^ { t }$   
3: for $i = 1 , . . . , n$ in parallel do   
4: Compute $\widehat { f _ { i } } ( w ^ { t } )$ and $\nabla \widehat { f } _ { i } ( w ^ { t } )$   
5: v<sup>t</sup><sub>i</sub> $ a _ { \kappa , \tau } ( \hat { f } _ { i } ( w ^ { t } ) ) \nabla \hat { f } _ { i } ( w ^ { t } )$   
6: Send $\boldsymbol { v } _ { i } ^ { t }$ to the server   
7: end for   
8: $\begin{array} { r } { w ^ { t + 1 }  w ^ { t } - \alpha \frac { 1 } { n } \sum _ { i = 1 } ^ { n } v _ { i } ^ { t } } \end{array}$   
9: end for   
10: return $w ^ { T }$

Because the server does not know R, FairMean applies the loss transformation to every client. Let ${ \widehat { f } } _ { i } = f _ { i }$ for $i \in \mathcal { R }$ and $\widehat { f _ { i } } = \widetilde { f _ { i } } \mathrm { f o r } i \in \mathcal { P }$ , denoting the loss evaluated by client i. At round t, client i transmits

$$
v _ { i } ^ { t } : = a _ { \kappa , \tau } ( \widehat { f } _ { i } ( w ^ { t } ) ) \nabla \widehat { f } _ { i } ( w ^ { t } ) .\tag{8}
$$

Using a fixed stepsize $\alpha > 0$ , the server averages the received vectors and updates

$$
w ^ { t + 1 } : = w ^ { t } - \frac { \alpha } { n } \sum _ { i = 1 } ^ { n } v _ { i } ^ { t } .\tag{9}
$$

## 4. THEORETICAL ANALYSIS

We first show that, without label poisoning, minimizing the proposed FairMean objective in (3) is more conducive to solution fairness than minimizing the original objective in (1). Throughout this paper, ∥ · ∥ denotes the Euclidean norm for vectors and the spectral norm for matrices. Proofs of all theoretical results are provided in Appendix B.

Theorem 1. Assume $p \ : = \ : 0$ and that $F _ { \mathcal { R } }$ and $\Phi _ { \mathcal { R } } ^ { \kappa , \tau }$ admit global minimizers $w _ { \mathrm { a v g } } ^ { \star }$ and $w _ { \mathrm { F M } } ^ { \star } ,$ , respectively. Then

$$
D _ { \tau } ( w _ { \mathrm { F M } } ^ { \star } ) \leq D _ { \tau } ( w _ { \mathrm { a v g } } ^ { \star } ) .
$$

Theorem 1 shows that, without label poisoning, $w _ { \mathrm { F M } } ^ { \star } ,$ a global minimizer of the objective in (3), has no larger $D _ { \tau }$ than $w _ { \mathrm { a v g } } ^ { \star } ,$ a global minimizer of the original objective $F _ { \mathcal { R } }$ in (1). Together with (7), the theorem provides a theoretical guarantee that minimizing the objective induced by $\phi _ { \kappa , \tau }$ in (3) promotes fairness at global optimality.

We now analyze FairMean as inexact gradient descent with respect to $\Phi _ { \mathcal { R } } ^ { \bar { \kappa } , \tau }$

Assumption 1. Let the convex set W contain $\boldsymbol { w ^ { 0 } } , \ldots , \boldsymbol { w ^ { T } }$ For every $i \in \mathcal { R }$ , the regular-client loss $f _ { i }$ is L-smooth on W.

Assumption 2. For some $G , A _ { \mathrm { L P } } \geq 0$ and every $w \in \mathcal W$ , the following bounds holdfor $i \in \mathcal { R }$ and $i \in \mathcal { P }$ , respectively:

$$
\begin{array} { r } { \Vert \nabla f _ { i } ( w ) \Vert \leq G , \quad \Vert \nabla \widetilde { f } _ { i } ( w ) - \nabla F _ { \mathcal { R } } ( w ) \Vert \leq A _ { \mathrm { L P } } . } \end{array}
$$

We next show that Assumption 2 holds for multiclass cross-entropy under label poisoning.

Lemma 1. Suppose all pre- and post-poisoning local losses are empirical cross-entropy losses of a K-class model with logits $\bar { h } ( w ; x ) ~ \in ~ \mathbb { R } ^ { K }$ Assume that the logit Jacobian $\nabla _ { w } h ( w ; x ) \in \mathbb { R } ^ { K \times d }$ satisfies $\| \nabla _ { w } h ( w ; x ) \| \le \Gamma$ on W for every local sample. Also assume $\Vert \nabla f _ { i } ( w ) - \nabla F _ { \mathcal R } ( w ) \Vert \ \leq$ H<sub>P</sub> for every $i \in \mathcal { P }$ and $w \in { \mathcal { W } } .$ . Let $\rho \in [ 0 , 1 ]$ be the largest fraction of relabeled samples at any poisoned client. Then Assumption 2 holds with

$$
G : = \sqrt { 2 } \Gamma , \quad A _ { \mathrm { L P } } : = H _ { \mathcal { P } } + \sqrt { 2 } \rho \Gamma .\tag{10}
$$

Table 1. CIFAR-10 test performance of regular clients in a 10-client network. Arrows indicate preferred directions.
<table><tr><td rowspan="2">Method</td><td colspan="3">No attack</td><td colspan="3">Random flip</td><td colspan="3">Pairwise flip</td></tr><tr><td>Avg.↑</td><td>Fairness↓</td><td>Worst↑</td><td>Avg.↑</td><td>Fairness↓</td><td>Worst↑</td><td>Avg.↑</td><td>Fairness↓</td><td>Worst↑</td></tr><tr><td>FedAvg</td><td>71.54</td><td>40.85</td><td>54.57</td><td>60.59</td><td>177.70</td><td>37.62</td><td>59.98</td><td>203.39</td><td>32.96</td></tr><tr><td>q-FFL</td><td>67.93</td><td>40.50</td><td>57.24</td><td>60.62</td><td>106.12</td><td>46.30</td><td>62.66</td><td>131.22</td><td>38.42</td></tr><tr><td>AdaFed</td><td>70.89</td><td>31.83</td><td>56.18</td><td>61.18</td><td>110.59</td><td>45.98</td><td>61.71</td><td>106.28</td><td>50.53</td></tr><tr><td>FedMGDA+</td><td>70.33</td><td>37.92</td><td>54.41</td><td>55.90</td><td>104.22</td><td>45.18</td><td>63.09</td><td>118.47</td><td>49.04</td></tr><tr><td>H-nobs</td><td>67.93</td><td>40.50</td><td>57.24</td><td>63.38</td><td>108.06</td><td>50.83</td><td>56.35</td><td>161.16</td><td>41.18</td></tr><tr><td>q-FFL+CWTM</td><td>67.93</td><td>40.50</td><td>57.24</td><td>32.71</td><td>406.58</td><td>0.49</td><td>34.29</td><td>379.87</td><td>8.73</td></tr><tr><td>FairMean</td><td>70.15</td><td>34.46</td><td>55.94</td><td>61.54</td><td>85.33</td><td>52.01</td><td>61.37</td><td>99.34</td><td>50.97</td></tr></table>

Lemma 2. Under Assumption 2, define $\begin{array} { r } { e ^ { t } : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } v _ { i } ^ { t } - } \end{array}$ $\nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } )$ . Then (9) can be written as

$$
w ^ { t + 1 } = w ^ { t } - \alpha \big [ \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) + e ^ { t } \big ] .
$$

Moreover, the error satisfies

$$
\begin{array} { r } { \| e ^ { t } \| \le \epsilon ( \delta , \kappa ) , \quad \epsilon ( \delta , \kappa ) : = \delta \big [ ( 1 + \kappa ) A _ { \mathrm { L P } } + 2 \kappa G \big ] . } \end{array}\tag{11}
$$

By (11), the attack contribution to the stationarity bound below contains the explicit factor $\delta ^ { 2 }$

Theorem 2. Under Assumptions 1–2, let $L _ { \Phi } ( \kappa , \tau ) : = ( 1 +$ $\kappa ) L + ( \kappa / \tau ) G ^ { 2 }$ . For $0 < \alpha \leq 1 / L _ { \Phi } ( \kappa , \tau )$ , FairMean satisfies

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) \| ^ { 2 } \leq \frac { 2 \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { 0 } ) } { \alpha T } + \epsilon ( \delta , \kappa ) ^ { 2 } .\tag{12}
$$

The optimization term in (12) decays as $O ( 1 / T )$ . Increasing κ tightens the stepsize bound, whereas increasing τ relaxes it through the term $( \kappa / \tau ) G ^ { 2 }$ , consistent with Fig. 1(b). Thus, more aggressive prioritization of high-loss clients, induced by a larger κ or a smaller τ , requires a more conservative theoretical stepsize.

## 5. EXPERIMENTS

Experimental setup. We use 10 clients, an MLP for Fashion-MNIST, and a 5-layer CNN for CIFAR-10<sup>1</sup>. Their training samples follow Dirichlet partitions with concentrations 0.5 and 0.1, respectively. The main experiments poison two clients. Random flip selects an incorrect label uniformly, while pairwise flip maps class c to 9 − c. Baselines are FedAvg [1], q-FFL [7], AdaFed [12], FedMGDA+ [22], Hnobs [21], and q-FFL+CWTM [17].

Evaluation metrics. Using clean test labels, we evaluate regular clients by accuracy variance, worst-client accuracy, and average accuracy, which measure fairness, poorest-client performance, and overall performance, respectively.

Overall performance. Tables 1 and 2 show that FairMean remains competitive without attacks. Under label poisoning, it achieves the lowest accuracy variance and highest worstclient accuracy in all four dataset–attack settings, while retaining competitive average accuracy.

![](images/92ede524c95a99ae924507534f0df379269e6ce9e2bc1550247fa2dc6cd597c4.jpg)  
Fig. 2. Fashion-MNIST performance under pairwise flip versus the poisoned-client fraction.

Impact of the poisoned-client fraction. On Fashion-MNIST under pairwise flip, we vary the poisoned-client fraction from 0 to 0.3. Figure 2 shows that FairMean retains its fairness advantage. $\mathrm { { A t } \ \delta \delta = \ 0 . 3 }$ , it reduces accuracy variance from 91.99 to 75.97 and raises worst-client accuracy from 56.19% to 60.85% relative to q-FFL.

Future work. Future work will extend FairMean to stochastic client participation and develop mechanisms that account for benign data heterogeneity when mitigating label poisoning.

## 6. REFERENCES

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, “Communication-efficient¨ learning of deep networks from decentralized data,” in Proceedings of the International Conference on Artificial Intelligence and Statistics, 2017.

[2] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning: Challenges, methods, and future directions,” IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 50–60, 2020.

[3] C. Ren, H. Yu, H. Peng, X. Tang, B. Zhao, L. Yi, A. Z. Tan, Y. Gao, A. Li, X. Li, Z. Li, and Q. Yang, “Advances and open challenges in federated foundation models,” IEEE Communications Surveys & Tutorials, vol. 28, pp. 2087–2126, 2026.

[4] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, “A novel framework for the analysis and design of heterogeneous federated learning,” IEEE Transactions on Signal Processing, vol. 69, pp. 5234–5249, 2021.

[5] Y. Shi, H. Yu, and C. Leung, “Towards fairness-aware federated learning,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 9, pp. 11922– 11938, 2024.

[6] N. Mukhtiar, A. Mahmood, and Q. Z. Sheng, “Fairness in federated learning: Trends, challenges, and opportunities,” Advanced Intelligent Systems, vol. 7, no. 6, pp. 2400836, 2025.

[7] T. Li, M. Sanjabi, A. Beirami, and V. Smith, “Fair resource allocation in federated learning,” in Proceedings of the International Conference on Learning Representations, 2020.

[8] Z. Zhao and G. Joshi, “A dynamic reweighting strategy for fair federated learning,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2022.

[9] T. Li, A. Beirami, M. Sanjabi, and V. Smith, “On tilted losses in machine learning: Theory and applications,” Journal ofMachine Learning Research, vol. 24, no. 142, pp. 1–79, 2023.

[10] Z. Pan, S. Wang, C. Li, H. Wang, X. Tang, and J. Zhao, “FedMDFG: Federated learning with multi-gradient descent and fair guidance,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2023.

[11] Z. Pan, C. Li, F. Yu, S. Wang, H. Wang, X. Tang, and J. Zhao, “FedLF: Layer-wise fair federated learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2024.

[12] S. M. Hamidi and E.-H. Yang, “AdaFed: Fair federated learning via adaptive common descent direction,” Transactions on Machine Learning Research, 2024.

[13] H. Zheng, J. Zhang, Y. Liu, and Q. Ling, “Fair-MOO: Achieving fairness in distributed learning via constrained multi-objective optimization,” in Proceedings ofthe IEEE International Conference on Acoustics, Speech and Signal Processing, 2026.

[14] N. M. Jebreel, J. Domingo-Ferrer, D. Sanchez, and´ A. Blanco-Justicia, “LFighter: Defending against the label-flipping attack in federated learning,” Neural Networks, vol. 170, pp. 111–126, 2024.

[15] J. Peng, W. Li, S. Vlaski, and Q. Ling, “Mean aggregator is more robust than robust aggregators under label poisoning attacks on distributed heterogeneous data,” Journal of Machine Learning Research, vol. 26, no. 27, pp. 1–51, 2025.

[16] L. Lamport, R. Shostak, and M. Pease, “The Byzantine generals problem,” ACM Transactions on Programming Languages and Systems, vol. 4, no. 3, pp. 382–401, 1982.

[17] D. Yin, Y. Chen, K. Ramchandran, and P. Bartlett, “Byzantine-robust distributed learning: Towards optimal statistical rates,” in Proceedings ofthe International Conference on Machine Learning, 2018.

[18] K. Pillutla, S. M. Kakade, and Z. Harchaoui, “Robust aggregation for federated learning,” IEEE Transactions on Signal Processing, vol. 70, pp. 1142–1154, 2022.

[19] Q. Xia, Z. Tao, Z. Hao, and Q. Li, “FABA: An algorithm for fast aggregation against Byzantine attacks in distributed neural networks,” in Proceedings of the International Joint Conference on Artificial Intelligence, 2019.

[20] G. Molodtsov, D. Medyakov, S. Skorik, N. Khachaturov, S. Tigranyan, V. Aletov, A. Avetisyan, M. Taka´c, and ˇ A. Beznosikov, “Bant: Byzantine antidote via trial function and trust scores,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

[21] G. Zhou, P. Xu, Y. Wang, and Z. Tian, “H-nobs: Achieving certified fairness and robustness in distributed learning on heterogeneous datasets,” in Advances in Neural Information Processing Systems, 2023.

[22] Z. Hu, K. Shaloudegi, G. Zhang, and Y. Yu, “Federated learning meets multi-objective optimization,” IEEE Transactions on Network Science and Engineering, vol. 9, no. 4, pp. 2039–2051, 2022.

## A. ADDITIONAL EXPERIMENTAL RESULTS

Table 2 shows that FairMean remains competitive without attacks. Under both random and pairwise flips, it achieves the lowest accuracy variance and the highest worst-client accuracy while maintaining competitive average accuracy, consistent with the main results on CIFAR-10.

## B. SUPPORTING PROOFS

Proof of Theorem 1. Since $\phi _ { \kappa , \tau } ^ { \prime } ( z ) = a _ { \kappa , \tau } ( z ) \ge 1$ , the loss transformation is strictly increasing. The optimality of $w _ { \mathrm { a v g } } ^ { \star }$ therefore implies $\phi _ { \kappa , \tau } ( F _ { \mathcal { R } } ( w _ { \mathrm { F M } } ^ { \star } ) ) \geq \phi _ { \kappa , \tau } ( F _ { \mathcal { R } } ( w _ { \mathrm { a v g } } ^ { \star } ) )$ . The optimality of $w _ { \mathrm { F M } } ^ { \star }$ and (6) further give

$$
\begin{array} { r l } & { \kappa \big [ D _ { \tau } ( w _ { \mathrm { a v g } } ^ { \star } ) - D _ { \tau } ( w _ { \mathrm { F M } } ^ { \star } ) \big ] } \\ & { ~ = \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w _ { \mathrm { a v g } } ^ { \star } ) - \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w _ { \mathrm { F M } } ^ { \star } ) } \\ & { ~ + \phi _ { \kappa , \tau } ( F _ { \mathcal { R } } ( w _ { \mathrm { F M } } ^ { \star } ) ) - \phi _ { \kappa , \tau } ( F _ { \mathcal { R } } ( w _ { \mathrm { a v g } } ^ { \star } ) ) \ge 0 . } \end{array}
$$

Dividing by $\kappa > 0$ proves the result.

□

Proof of Lemma 1. For each $i \in \mathcal { R }$ , write its local data as $\{ ( x _ { i j } , y _ { i j } ) \} _ { j = 1 } ^ { m _ { i } }$ . For each $i \in \mathcal { P }$ , write its data before and after poisoning as $\{ ( x _ { i j } , y _ { i j } , \widetilde { y } _ { i j } ) \} _ { j = 1 } ^ { m _ { i } }$ . The largest withinclient relabeling fraction is

$$
\rho : = \operatorname* { m a x } _ { i \in \mathcal { P } } \frac { 1 } { m _ { i } } \big | \{ j : \widetilde { y } _ { i j } \neq y _ { i j } \} \big | .
$$

Define $\pi ( w ; x ) ~ : = ~ \mathrm { s o f t m a x } ( h ( w ; x ) )$ and $\ell ( w ; x , y ) : =$ $- \log \pi _ { y } ( w ; x )$ . For any local sample,

$$
\nabla _ { \boldsymbol { w } } \ell ( \boldsymbol { w } ; \boldsymbol { x } , \boldsymbol { y } ) = \nabla _ { \boldsymbol { w } } h ( \boldsymbol { w } ; \boldsymbol { x } ) ^ { \top } \big ( \pi ( \boldsymbol { w } ; \boldsymbol { x } ) - e _ { \boldsymbol { y } } \big ) ,
$$

where $e _ { y }$ is the y-th standard basis vector in $\mathbb { R } ^ { K }$ . Since $\| \pi ( w ; x ) - e _ { y } \| \leq \sqrt { 2 }$ , the Jacobian bound gives

$$
\begin{array} { r } { \| \nabla _ { w } \ell ( w ; x , y ) \| \leq \| \nabla _ { w } h ( w ; x ) \| \left\| \pi ( w ; x ) - e _ { y } \right\| \leq \sqrt { 2 } \Gamma . } \end{array}
$$

Averaging over the local samples yields

$$
\begin{array} { r } { \| \nabla f _ { i } ( w ) \| \leq \sqrt { 2 } \Gamma , \quad i \in \mathcal { R } , w \in \mathcal { W } . } \end{array}
$$

Thus, the regular-client gradient bound holds with $G = { \sqrt { 2 } } \Gamma$ Fix $i \in \mathcal { P }$ . Because protocol-following label poisoning leaves x unchanged, the softmax terms cancel and

$$
\nabla _ { \boldsymbol { w } } \ell ( \boldsymbol { w } ; \boldsymbol { x } , \widetilde { \boldsymbol { y } } ) - \nabla _ { \boldsymbol { w } } \ell ( \boldsymbol { w } ; \boldsymbol { x } , \boldsymbol { y } ) = \nabla _ { \boldsymbol { w } } h ( \boldsymbol { w } ; \boldsymbol { x } ) ^ { \top } ( \boldsymbol { e } _ { \boldsymbol { y } } - \boldsymbol { e } _ { \widetilde { \boldsymbol { y } } } ) .
$$

For every $j$ satisfying $\widetilde { y } _ { i j } \ \ne \ y _ { i j }$ , the spectral norm bound gives

$$
\begin{array} { r l } & { \left\| \nabla _ { w } h ( w ; x _ { i j } ) ^ { \top } \big ( e _ { y _ { i j } } - e _ { \widetilde { y } _ { i j } } \big ) \right\| } \\ & { \leq \| \nabla _ { w } h ( w ; x _ { i j } ) \| \| e _ { y _ { i j } } - e _ { \widetilde { y } _ { i j } } \| \leq \sqrt { 2 } \Gamma . } \end{array}
$$

Averaging over the local samples of client i, with unchanged labels contributing zero, yields

$$
\| \nabla \widetilde { f } _ { i } ( w ) - \nabla f _ { i } ( w ) \| \leq \frac { \sqrt { 2 } \Gamma } { m _ { i } } | \{ j : \widetilde { y } _ { i j } \neq y _ { i j } \} | \leq \sqrt { 2 } \rho \Gamma .
$$

Adding and subtracting $\nabla f _ { i } ( \boldsymbol { w } )$ therefore yields

$$
\begin{array} { r l } & { \| \nabla \widetilde { f } _ { i } ( w ) - \nabla F _ { \mathcal { R } } ( w ) \| \le \| \nabla \widetilde { f } _ { i } ( w ) - \nabla f _ { i } ( w ) \| } \\ & { \phantom { { \| \nabla \widetilde { f } _ { i } ( w ) - \nabla F _ { \mathcal { R } } ( w ) \| } } + \| \nabla f _ { i } ( w ) - \nabla F _ { \mathcal { R } } ( w ) \| } \\ & { \le \sqrt { 2 } \rho \Gamma + H _ { \mathcal { P } } , } \end{array}
$$

which proves (10).

Proof of Lemma 2. Fix $t \in \{ 0 , \ldots , T - 1 \}$ . For this proof, define the averaged server direction

$$
u ^ { t } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } v _ { i } ^ { t } ,
$$

so that the main-text definition gives $e ^ { t } = u ^ { t } - \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } )$ Also define

$$
g _ { \mathcal { R } } ^ { t } : = \nabla F _ { \mathcal { R } } ( w ^ { t } ) = \frac { 1 } { r } \sum _ { i \in \mathcal { R } } \nabla f _ { i } ( w ^ { t } ) .
$$

Assumption 2 and the triangle inequality imply

$$
\| g _ { \mathcal { R } } ^ { t } \| \leq \frac { 1 } { r } \sum _ { i \in \mathcal { R } } \| \nabla f _ { i } ( w ^ { t } ) \| \leq G .
$$

For each $i \in \mathcal R$ , let $s _ { i } ^ { t } : = \ f _ { i } ( w ^ { t } ) / ( f _ { i } ( w ^ { t } ) + \tau )$ . Since $a _ { \kappa , \tau } ( f _ { i } ( w ^ { t } ) ) = 1 + \kappa s _ { i } ^ { t }$ and $0 \leq s _ { i } ^ { t } \leq 1$ , we obtain

$$
\begin{array} { r l } & { \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) = g _ { \mathcal { R } } ^ { t } + \displaystyle \frac { \kappa } { r } \sum _ { i \in \mathcal { R } } s _ { i } ^ { t } \nabla f _ { i } ( w ^ { t } ) , } \\ & { \| \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) - g _ { \mathcal { R } } ^ { t } \| \leq \displaystyle \frac { \kappa } { r } \sum _ { i \in \mathcal { R } } s _ { i } ^ { t } \| \nabla f _ { i } ( w ^ { t } ) \| \leq \kappa G . } \end{array}
$$

We first consider $p = 0$ . In this case, $n = r$ and $\delta = 0 .$ and the definitions of $\boldsymbol { v } _ { i } ^ { t }$ and $u ^ { t }$ give

$$
u ^ { t } = \frac { 1 } { r } \sum _ { i \in \mathcal { R } } a _ { \kappa , \tau } ( f _ { i } ( w ^ { t } ) ) \nabla f _ { i } ( w ^ { t } ) = \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) .
$$

Thus, $e ^ { t } = 0 .$ , and (11) holds.

Suppose next that $p > 0$ . For $i \in { \mathcal { P } } ,$ define

$$
\widetilde { a } _ { i } ^ { t } : = a _ { \kappa , \tau } ( \widetilde { f } _ { i } ( \boldsymbol { w } ^ { t } ) ) , \quad q _ { \mathcal { P } } ^ { t } : = \frac { 1 } { p } \sum _ { i \in \mathcal { P } } \widetilde { a } _ { i } ^ { t } \nabla \widetilde { f } _ { i } ( \boldsymbol { w } ^ { t } ) .
$$

The nonnegativity of $\widetilde { f } _ { i }$ implies $1 \leq \widetilde { a } _ { i } ^ { t } \leq 1 +$ κ and $0 \leq \widetilde { a } _ { i } ^ { t } -$ $1 \le \kappa$ . Adding and subtracting $\widetilde { a } _ { i } ^ { t } g _ { \mathcal { R } } ^ { t }$ within each summand gives

$$
\begin{array} { r l } & { q _ { \mathcal { P } } ^ { t } - g _ { \mathcal { R } } ^ { t } = \displaystyle \frac { 1 } { p } \sum _ { i \in \mathcal { P } } \widetilde { a } _ { i } ^ { t } \big ( \nabla \widetilde { f } _ { i } ( w ^ { t } ) - g _ { \mathcal { R } } ^ { t } \big ) } \\ & { \quad \quad \quad + \displaystyle \frac { 1 } { p } \sum _ { i \in \mathcal { P } } ( \widetilde { a } _ { i } ^ { t } - 1 ) g _ { \mathcal { R } } ^ { t } . } \end{array}
$$

Therefore, Assumption 2 yields

$$
\begin{array} { r l } { \| q _ { \mathcal { P } } ^ { t } - g _ { \mathcal { R } } ^ { t } \| \le \displaystyle \frac { 1 } { p } \sum _ { i \in \mathcal { P } } \widetilde { a } _ { i } ^ { t } \| \nabla \widetilde { f } _ { i } ( w ^ { t } ) - g _ { \mathcal { R } } ^ { t } \| } & { } \\ { + \displaystyle \frac { 1 } { p } \sum _ { i \in \mathcal { P } } ( \widetilde { a } _ { i } ^ { t } - 1 ) \| g _ { \mathcal { R } } ^ { t } \| } & { } \\ { \le ( 1 + \kappa ) A _ { \mathrm { L P } } + \kappa G . } \end{array}
$$

Table 2. Fashion-MNIST test performance of regular clients in a 10-client network. Arrows indicate preferred directions.
<table><tr><td rowspan="2">Method</td><td colspan="3">No attack</td><td colspan="3">Random flip</td><td colspan="3">Pairwise flip</td></tr><tr><td>Avg.↑</td><td>Fairness↓</td><td>Worst↑</td><td> $\mathrm { A v g . \uparrow }$ </td><td>Fairness↓</td><td>Worst↑</td><td>Avg.↑</td><td>Fairness↓</td><td>Worst↑</td></tr><tr><td>FedAvg</td><td>86.90</td><td>20.00</td><td>79.38</td><td>77.08</td><td>141.43</td><td>58.23</td><td>75.65</td><td>153.12</td><td>55.82</td></tr><tr><td>q-FFL</td><td>88.35</td><td>16.11</td><td>81.33</td><td>81.69</td><td>73.66</td><td>65.97</td><td>82.56</td><td>48.41</td><td>66.34</td></tr><tr><td>AdaFed</td><td>85.62</td><td>12.46</td><td>79.20</td><td>74.55</td><td>129.47</td><td>55.20</td><td>73.10</td><td>139.15</td><td>54.95</td></tr><tr><td>FedMGDA+</td><td>88.62</td><td>15.49</td><td>81.96</td><td>73.68</td><td>126.05</td><td>56.93</td><td>76.93</td><td>155.32</td><td>56.19</td></tr><tr><td>H-nobs</td><td>88.35</td><td>16.11</td><td>81.33</td><td>84.19</td><td>56.79</td><td>69.60</td><td>83.99</td><td>67.89</td><td>67.47</td></tr><tr><td> $q \mathrm { - F F L + C W T M }$ </td><td>88.35</td><td>16.11</td><td>81.33</td><td>54.24</td><td>431.55</td><td>19.09</td><td>63.07</td><td>466.48</td><td>19.37</td></tr><tr><td>FairMean</td><td>87.55</td><td>18.44</td><td>80.36</td><td>81.87</td><td>42.12</td><td>70.55</td><td>82.57</td><td>31.95</td><td>71.29</td></tr></table>

The regular-client contribution equals $r \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } )$ . The identities $r / n = 1 - \delta$ and $p / n = \delta$ therefore give

$$
\begin{array} { l } { { \displaystyle u ^ { t } = \frac { r } { n } \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) + \frac { p } { n } q _ { \mathcal { P } } ^ { t } } } \\ { { \displaystyle ~ = ( 1 - \delta ) \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) + \delta q _ { \mathcal { P } } ^ { t } . } } \end{array}
$$

Subtracting $\nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } )$ from both sides and using the definition of $e ^ { t }$ yields

$$
e ^ { t } = \delta \big ( q _ { \mathcal { P } } ^ { t } - \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) \big ) .
$$

Combining the preceding two bounds gives

$$
\begin{array} { r l } & { \| e ^ { t } \| \leq \delta \big ( \| q _ { \mathcal { P } } ^ { t } - g _ { \mathcal { R } } ^ { t } \| + \| g _ { \mathcal { R } } ^ { t } - \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) \| \big ) } \\ & { \qquad \leq \delta \big [ ( 1 + \kappa ) A _ { \mathrm { L P } } + 2 \kappa G \big ] = \epsilon ( \delta , \kappa ) . } \end{array}
$$

This proves (11) and completes the proof.

□

Proof of Theorem 2. We first establish the smoothness of $\Phi _ { \mathcal { R } } ^ { \kappa , \tau }$ . For any $x , y \in \mathcal { W }$ , convexity of W and the regularclient gradient bound give

$$
\begin{array} { l } { | f _ { i } ( y ) - f _ { i } ( x ) | = \displaystyle \left| \int _ { 0 } ^ { 1 } \langle \nabla f _ { i } ( x + s ( y - x ) ) , y - x \rangle d s \right| } \\ { \leq G \| y - x \| . } \end{array}
$$

For brevity, set $a _ { i } ( w ) : = a _ { \kappa , \tau } ( f _ { i } ( w ) )$ . The derivative bound following (4) gives

$$
| a _ { i } ( x ) - a _ { i } ( y ) | \leq { \frac { \kappa } { \tau } } | f _ { i } ( x ) - f _ { i } ( y ) | \leq { \frac { \kappa G } { \tau } } \| x - y \| .
$$

Define $\begin{array} { r } { h _ { i } ( w ) { \mathbf \Omega } : = { a _ { i } ( w ) \nabla { f _ { i } } ( w ) } } \end{array}$ . Adding and subtracting $a _ { i } ( x ) \nabla f _ { i } ( y )$ and using $1 \leq a _ { i } ( x ) \leq 1 + \kappa$ gives

$$
\begin{array} { r l } & { \| h _ { i } ( x ) - h _ { i } ( y ) \| \leq a _ { i } ( x ) \| \nabla f _ { i } ( x ) - \nabla f _ { i } ( y ) \| } \\ & { \qquad + | a _ { i } ( x ) - a _ { i } ( y ) | \| \nabla f _ { i } ( y ) \| } \\ & { \qquad \leq \left[ ( 1 + \kappa ) L + \frac { \kappa } { \tau } G ^ { 2 } \right] \| x - y \| } \\ & { \qquad = L _ { \Phi } ( \kappa , \tau ) \| x - y \| . } \end{array}
$$

Since $\begin{array} { r } { \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ) = r ^ { - 1 } \sum _ { i \in \mathcal { R } } h _ { i } ( w ) } \end{array}$ , averaging this bound shows that $\bar { \Phi } _ { \mathcal { R } } ^ { \kappa , \tau }$ is $L _ { \Phi } ( \kappa , \tau )$ -smooth on W.

For each $t = 0 , \ldots , T - 1$ , let $g ^ { t } : = \nabla \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } )$ . By Lemma 2, the update satisfies $w ^ { t + 1 } = w ^ { t } - \stackrel {  } { \alpha } ( g ^ { t } + e ^ { t } )$ The segment between $w ^ { t }$ and $w ^ { t + 1 }$ lies in $w ,$ , so the descent lemma gives

$$
\begin{array} { r l } & { \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t + 1 } ) \leq \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) - \alpha \langle g ^ { t } , g ^ { t } + e ^ { t } \rangle } \\ & { \qquad + \frac { L _ { \Phi } \left( \kappa , \tau \right) \alpha ^ { 2 } } { 2 } \| g ^ { t } + e ^ { t } \| ^ { 2 } } \\ & { \qquad \leq \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) - \alpha \langle g ^ { t } , g ^ { t } + e ^ { t } \rangle } \\ & { \qquad + \frac { \alpha } { 2 } \| g ^ { t } + e ^ { t } \| ^ { 2 } , } \end{array}
$$

where the second inequality follows from $\alpha L _ { \Phi } ( \kappa , \tau ) \leq 1$ The identity

$$
- \langle g , g + e \rangle + \frac { 1 } { 2 } \| g + e \| ^ { 2 } = \frac { 1 } { 2 } \big ( \| e \| ^ { 2 } - \| g \| ^ { 2 } \big )
$$

therefore yields

$$
\Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t + 1 } ) \leq \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { t } ) - \frac { \alpha } { 2 } \| g ^ { t } \| ^ { 2 } + \frac { \alpha } { 2 } \| e ^ { t } \| ^ { 2 } .\tag{13}
$$

Rearranging (13) and summing from t = 0 to T − 1 gives

$$
\frac { \alpha } { 2 } \sum _ { t = 0 } ^ { T - 1 } \| g ^ { t } \| ^ { 2 } \leq \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { 0 } ) - \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { T } ) + \frac { \alpha } { 2 } \sum _ { t = 0 } ^ { T - 1 } \| e ^ { t } \| ^ { 2 } .
$$

Since $\lVert e ^ { t } \rVert \leq \epsilon ( \delta , \kappa )$ , the accumulated error term is at most $T \epsilon ( \delta , \dot { \kappa } ) ^ { 2 }$ . Moreover, the nonnegativity of $\phi _ { \kappa , \tau }$ and (3) imply $\Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { T } ) \geq 0$ . Therefore,

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| g ^ { t } \| ^ { 2 } \leq \frac { 2 [ \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { 0 } ) - \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { T } ) ] } { \alpha T } + \epsilon ( \delta , \kappa ) ^ { 2 } } \\ & { \qquad \leq \frac { 2 \Phi _ { \mathcal { R } } ^ { \kappa , \tau } ( w ^ { 0 } ) } { \alpha T } + \epsilon ( \delta , \kappa ) ^ { 2 } . } \end{array}
$$

Recalling the definition of $g ^ { t }$ proves (12).