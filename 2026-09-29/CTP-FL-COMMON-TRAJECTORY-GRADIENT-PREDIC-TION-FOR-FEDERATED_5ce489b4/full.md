# CTP-FL: COMMON-TRAJECTORY GRADIENT PREDIC-TION FOR FEDERATED LEARNING

Junkang Liu<sup>1</sup>   
Tianjin University   
{junkangliukk}@gmail.com

## ABSTRACT

Communication-efficient federated optimization commonly spends several gradient evaluations between server updates. Existing local-update methods use this computation to advance an independent model on each client. Under heterogeneous data, however, these models evaluate gradients at different locations, making the aggregated update difficult to interpret as a gradient of the global objective. We study an alternative use of the same computation budget: evaluate the global objective along a shared, predicted path. We propose Common-Trajectory Predictive Federated Learning (CTP-FL). At each round, all clients construct the same sequence of query points from the current global model and the previous aggregated direction, evaluate K stochastic gradients along this sequence, and upload their average. The server then performs a single global update. Thus, CTP-FL uses K mini-batch gradients per client and one model-sized vector in each communication direction, matching the per-round computation and communication of full-participation FedAvg-M. Shared query points make the aggregated direction an unbiased estimator of the average global gradient along the predicted path. The remaining discrepancy from the gradient at the current model is controlled by the path length, without assuming bounded client-gradient dissimilarity or bounded gradients. For smooth non-convex objectives, we establish an $\mathcal { O } \Big ( \sqrt { L \Delta \sigma ^ { 2 } / ( N K R ) } + L \Delta / R \Big )$ average-stationarity bound under full participation. The analysis isolates a testable trade-off: extending the prediction path provides more forward-looking gradient information but increases its displacement bias.

## 1 INTRODUCTION

Federated learning trains a shared model from decentralized data while limiting communication between clients and a server. Federated Averaging (FedAvg) addresses communication cost by performing multiple client updates before each aggregation (McMahan et al., 2017). This structure has shaped a large body of work on local optimization (Stich, 2019). Yet the role of the computation performed between communications deserves closer examination: K local gradient evaluations need not be used to produce K independent client trajectories.

Where does the difficulty arise? Let $\begin{array} { r } { F ( \pmb { x } ) = N ^ { - 1 } \sum _ { i = 1 } ^ { N } F _ { i } ( \pmb { x } ) } \end{array}$ . In a local-update method, every client begins a round at the same $\pmb { x } ^ { r }$ , but its later gradients are evaluated at client-specific points $\pmb { x } _ { i } ^ { r , k }$ . Even when all clients participate, the average gradient at these different points is generally not a gradient of $F$ at any single common point. This distinction is particularly relevant under heterogeneous client objectives: differences in local gradients alter the trajectories, which in turn alter where subsequent gradients are measured. SCAFFOLD addresses this client drift with control variates (Karimireddy et al., 2020). FedAvg-M instead introduces a global momentum direction into local updates and proves full-participation convergence without a bounded data-heterogeneity assumption (Cheng et al., 2024). These results establish that heterogeneity can be handled without assuming similar client gradients. They also motivate a different question: can the gradient locations themselves be coordinated without increasing per-round communication?

A shared path in place of independent trajectories. We propose Common-Trajectory Predictive Federated Learning (CTP-FL). At round r, the server model $\pmb { x } ^ { r }$ and previous aggregate direction $\pmb { v } ^ { r }$ define K shared query points

$$
z _ { k } ^ { r } = \pmb { x } ^ { r } - t _ { k } \pmb { v } ^ { r } , \qquad t _ { k } \in [ 0 , h ] , \qquad k = 0 , \ldots , K - 1 .\tag{1}
$$

Every client evaluates one mini-batch gradient at each $z _ { k } ^ { r }$ and uploads only their average. The server averages these client vectors to obtain $\bar { \pmb { v } } ^ { r + 1 }$ and updates $\begin{array} { r } { \tilde { \mathbf { \pmb { x } } } ^ { r + 1 } = \mathbf { \pmb { x } } ^ { r } - \gamma \mathbf { \pmb { v } } ^ { r + 1 } } \end{array}$ . The previous direction can be recovered from consecutive global models, so it need not be transmitted as an additional vector.

The distinction from local SGD and FedAvg-M is structural. CTP-FL does not execute K local model updates: the K computations probe a common predicted path before one server update. It therefore gives up the adaptive local trajectories that can make local training effective, while ensuring that clients evaluate the same sequence of model parameters. The prediction path is also different from Lookahead’s interpolation between fast and slow weights (Zhang et al., 2019): here it specifies where distributed gradients are measured, rather than combining locally updated weights.

What does the shared path buy? Conditioned on the history of round $r ,$ independent unbiased mini-batch gradients give

$$
\mathbb { E } [ { \pmb v } ^ { r + 1 } \mid { \mathcal F } _ { r } ] = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F ( \pmb z _ { k } ^ { r } ) .\tag{2}
$$

This identity holds for arbitrarily different client objectives under full participation: client-gradient differences cancel at every common query point. The direction is not, in general, an unbiased estimate of $\nabla F ( { \pmb x } ^ { r } )$ ). If each $F _ { i }$ is L-smooth, its displacement bias satisfies

$$
\left\| \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F ( \boldsymbol { z } _ { k } ^ { r } ) - \nabla F ( \boldsymbol { x } ^ { r } ) \right\| \leq \frac { L } { K } \sum _ { k = 0 } ^ { K - 1 } t _ { k } \| \boldsymbol { v } ^ { r } \| .\tag{3}
$$

For the uniform path from 0 to $h ,$ , the right-hand side equals $L h \| \pmb { v } ^ { r } \| / 2$ as an upper bound. Equations equation 2 and equation 3 expose the method’s central trade-off: h determines how far ahead the clients measure the global gradient field, while also controlling the bias relative to the current iterate.

Guarantee and scope. Under L-smoothness, unbiased stochastic gradients with variance at most $\sigma ^ { 2 }$ , and full client participation, we prove that the choice $h = \gamma$ with an appropriate server stepsize satisfies

$$
\frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } = \mathcal { O } \left( \sqrt { \frac { L \Delta \sigma ^ { 2 } } { N K R } } + \frac { L \Delta } { R } \right) ,\tag{4}
$$

where $\begin{array} { r } { \Delta = F ( \pmb { x } ^ { 0 } ) - \operatorname* { i n f } _ { \pmb { x } } F ( \pmb { x } ) } \end{array}$ . The proof requires neither bounded gradients nor bounded clientgradient dissimilarity. This rate is of the same order as the full-participation FedAvg-M guarantee; it does not establish a theoretical acceleration over FedAvg-M. Importantly, the cancellation in equation 2 relies on full participation. With client sampling, an additional sampling-variance term appears, and the present algorithm does not provide a heterogeneity-independent partial-participation guarantee.

Our contributions are:

• We introduce a communication-neutral allocation of K client gradient evaluations to a shared predicted trajectory, with one model-sized uplink and downlink vector per client per round.

• We characterize the resulting direction as the average global gradient along the shared trajectory and identify its displacement bias separately from stochastic-gradient noise.

• We prove the non-convex full-participation guarantee in equation 4 without bounded gradients or a bounded data-heterogeneity assumption.

## 2 RELATED WORK

Local updates and communication efficiency. FedAvg reduces communication by allowing clients to take multiple optimization steps before model averaging (McMahan et al., 2017). Local SGD provides a broader distributed-optimization perspective on this strategy (Stich, 2019). Its benefit, however, depends on how the saved communication interacts with the errors introduced by independent local trajectories. Woodworth et al. (Woodworth et al., 2020) show that local SGD and mini-batch SGD do not uniformly dominate one another across objective classes. This comparison is especially relevant to our setting: CTP-FL allocates K gradient evaluations per client to K shared query points, rather than to K successive local model updates. When its prediction span is zero, CTP-FL reduces to a synchronized mini-batch-gradient baseline. Thus, any benefit of prediction must be measured against both this baseline and strong local-update methods at matched computation and communication budgets.

Data heterogeneity and client drift. Under heterogeneous client objectives, local models can follow different trajectories even when they start from the same global model. FedProx regularizes local optimization around the current server model (Li et al., 2020), while SCAFFOLD uses client and server control variates to correct client drift (Karimireddy et al., 2020). Mime combines control variates with server-side optimizer statistics to approximate the behavior of centralized optimizers during local training (Karimireddy et al., 2021). FedAvg-M injects a global momentum direction into local updates and establishes full-participation convergence without assuming bounded clientgradient dissimilarity; it also develops momentum variants of SCAFFOLD for partial participation (Cheng et al., 2024). Hence, the absence of a bounded heterogeneity assumption is not, by itself, a distinguishing claim for CTP-FL. Our distinction is where gradients are evaluated: at local, clientdependent iterates in these methods, and at the same predicted positions on every client in CTP-FL. Under full participation, averaging the latter gradients gives an estimator of the global gradient averaged along the shared path. The remaining bias is governed by the path’s displacement from the current model.

Prediction and shared gradient evaluation. Prediction has also been used to organize optimization across multiple steps. Lookahead maintains fast and slow weights and interpolates between them after several optimizer updates (Zhang et al., 2019). CTP-FL uses the previous aggregated direction for a different purpose: it specifies the locations of the current round’s gradient evaluations, without advancing separate client models along that path. The resulting direction is generally biased relative to $\nabla F ( { \pmb x } ^ { r } )$ , even though it is an unbiased estimate of the average global gradient at the queried positions. Our analysis makes this displacement bias explicit and bounds it using smoothness and the prediction span. The present guarantee applies to full participation; subsampling clients introduces additional sampling variance that the basic CTP-FL update does not remove.

## 3 THE PROPOSED ALGORITHM

## 3.1 PROBLEM SETUP

FL seeks to learn a global model collaboratively over clients by minimizing the population risk:

$$
f _ { i } ( \pmb { x } ) : = \mathbb { E } _ { \xi _ { i } \sim \mathcal { D } _ { i } } \big [ F _ { i } ( \pmb { x } ; \xi _ { i } ) \big ] , \qquad f ( \pmb { x } ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f _ { i } ( \pmb { x } ) .\tag{5}
$$

The function $F _ { i }$ is the loss function on client $i . \mathbb { E } _ { \xi _ { i } \sim \mathcal { D } _ { i } } [ \cdot ]$ denotes conditional expectation with respect to the sample $\xi _ { i }$ . N is the number of clients, and x is global model.

## 3.2 FROM LOCAL TRAJECTORIES TO A SHARED GRADIENT PATH

Consider the federated objective

$$
\operatorname* { m i n } _ { \pmb { x } \in \mathbb { R } ^ { d } } F ( \pmb { x } ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F _ { i } ( \pmb { x } ) ,\tag{6}
$$

where $F _ { i }$ is the objective of client i. A communication round typically allows each client to compute $K$ stochastic gradients before uploading one model-sized vector. In local-update methods, these

gradients are evaluated along client-specific iterates $\pmb { x } _ { i } ^ { r , k }$ . Consequently, even under full participation, their aggregate takes the form

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla F _ { i } ( \mathbf { x } _ { i } ^ { r , k } ) ,\tag{7}
$$

which generally differs from $\nabla F ( { \pmb x } )$ at any common model ${ \pmb x } .$ The mismatch arises from the locations of gradient evaluation: heterogeneity first separates the client trajectories, after which subsequent gradients are measured at different locations.

CTP-FL uses the same $K$ gradient evaluations to query a shared predictive path. Instead of constructing N local trajectories, all clients evaluate their objectives at the same sequence of points. This makes the client average at each point an estimate of the corresponding global gradient. The remaining approximation is explicit: the queried global gradients are displaced from the current server model.

## 3.3 COMMON-TRAJECTORY PREDICTIVE FEDERATED LEARNING

At the start of communication round $r ,$ let $\pmb { x } ^ { r }$ be the global model and ${ \pmb v } ^ { r }$ the direction aggregated in the previous round. Set $\mathbf { \boldsymbol { v } } ^ { 0 } = \mathbf { 0 }$ . Given a prediction span $h \geq 0$ , define

$$
t _ { k } = \left\{ \frac { k h } { K - 1 } , \quad K > 1 , \qquad z _ { k } ^ { r } = x ^ { r } - t _ { k } v ^ { r } , \quad k = 0 , \ldots , K - 1 . \right.\tag{8}
$$

The points $z _ { k } ^ { r }$ are identical across clients and are fixed before the current round’s mini-batches are sampled. Client i computes

$$
{ \pmb u } _ { i } ^ { r } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } g _ { i } ( z _ { k } ^ { r } ; \xi _ { i , k } ^ { r } ) ,\tag{9}
$$

where $g _ { i } \big ( z _ { k } ^ { r } ; \xi _ { i , k } ^ { r } \big )$ is a stochastic gradient of $F _ { i }$ at the shared point $\boldsymbol { z } _ { k } ^ { r } .$ . The server aggregates

$$
{ \pmb v } ^ { r + 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \pmb u } _ { i } ^ { r } , \qquad { \pmb x } ^ { r + 1 } = { \pmb x } ^ { r } - \gamma { \pmb v } ^ { r + 1 } .\tag{10}
$$

Algorithm 1 summarizes the procedure. The K query points are gradient-evaluation locations; they are not successive local model updates.

The previous direction requires no separate downlink vector under full participation. A client retaining $\pmb { x } ^ { r - 1 }$ recovers it from

$$
{ \pmb v } ^ { r } = \frac { { \pmb x } ^ { r - 1 } - { \pmb x } ^ { r } } { \gamma } , \qquad r \geq 1 .\tag{11}
$$

Thus, each round uses K mini-batch gradient evaluations per client, one d-dimensional uplink vector $\mathbf { \Delta } \mathbf { \boldsymbol { u } } _ { i } ^ { r }$ , and one d-dimensional downlink model $\pmb { x } ^ { r }$ . The protocol has the same per-round gradient count and vector communication as full-participation FedAvg-M, although its computation does not perform local SGD steps.

## 3.4 WHY SHARED QUERIES CHANGE THE ERROR STRUCTURE

Let ${ \mathcal { F } } _ { r }$ contain the history before the current mini-batches are drawn. Under unbiased stochastic gradients,

$$
\mathbb { E } [ { \pmb v } ^ { r + 1 } \mid { \mathcal F } _ { r } ] = \underbrace { \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F ( \pmb z _ { k } ^ { r } ) } _ { H ^ { r } } .\tag{12}
$$

The equality follows by averaging across all clients at each common point:

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla F _ { i } ( \boldsymbol { z } _ { k } ^ { r } ) = \nabla F ( \boldsymbol { z } _ { k } ^ { r } ) .
$$

Algorithm 1 Common-Trajectory Predictive Federated Learning (CTP-FL)   
Require: Server stepsize γ, prediction span $h ,$ communication rounds $R ,$ gradients per round $K ,$ number of   
clients N   
1: Initialize global model $\pmb { x } ^ { 0 }$ and prediction direction ${ \pmb v } ^ { 0 } = { \bf 0 }$   
2: for $r = 0 , \ldots , R - 1$ do   
3: Server broadcasts ${ \pmb x } ^ { r }$   
4: for each client $i = 1 , \ldots , N$ in parallel do   
5: Recover ${ \pmb v } ^ { r }  ( { \pmb x } ^ { r - 1 } - { \pmb x } ^ { r } ) / \gamma$ if $r > 0 ;$ otherwise use ${ \pmb v } ^ { 0 } = { \bf 0 }$   
6: $\mathbf { \Delta } \mathbf { \mathbf { u } } _ { i } ^ { r } \gets \mathbf { \mathbf { 0 } }$   
7: for $k = 0 , \ldots , K - 1$ do   
8: $t _ { k } \gets k h / ( K - 1 )$ if $K > 1 ;$ otherwise $t _ { 0 } \gets 0$   
9: $\pmb { z } _ { k } ^ { r }  \pmb { x } ^ { r } - t _ { k } \pmb { v } ^ { r }$   
10: Sample mini-batch ${ B } _ { i } ^ { r , k } ; { \pmb { g } } _ { i } ^ { r , k }  \nabla F _ { i } ( { \pmb { z } } _ { k } ^ { r } ; { B } _ { i } ^ { r , k } )$   
11: $\pmb { u } _ { i } ^ { r } \gets \pmb { u } _ { i } ^ { r } + \pmb { g } _ { i } ^ { r , k } / K$   
12: end for   
13: Client i sends $\pmb { u } _ { i } ^ { r }$ to the server   
14: end for   
15: $\begin{array} { r } { \pmb { v } ^ { r + 1 }  \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \pmb { u } _ { i } ^ { r } } \end{array}$   
16: $\pmb { x } ^ { r + 1 }  \pmb { x } ^ { r } - \gamma \pmb { v } ^ { r + 1 }$   
17: end for   
Ensure: Global model $\scriptstyle { \pmb { x } } ^ { R }$

It holds regardless of the magnitude of $\nabla F _ { i } ( z _ { k } ^ { r } ) - \nabla F ( z _ { k } ^ { r } )$ . Accordingly, heterogeneity does not create an additional client-trajectory term in the conditional mean under full participation.

The path-averaged direction ${ \pmb { H } } ^ { r }$ is generally distinct from $\nabla F ( { \pmb x } ^ { r } )$ . For L-smooth $F ,$

$$
\| \pmb { H } ^ { r } - \nabla F ( \pmb { x } ^ { r } ) \| \leq \frac { L } { K } \sum _ { k = 0 } ^ { K - 1 } t _ { k } \| \pmb { v } ^ { r } \| \leq \frac { L h } { 2 } \| \pmb { v } ^ { r } \| .\tag{13}
$$

The prediction span therefore has a precise role: it governs how far the algorithm probes along its previous direction and how much displacement bias that probe can introduce. When $h = 0 ,$ , all queries coincide at $\pmb { x } ^ { r }$ , giving synchronized mini-batch SGD with NK gradient evaluations per round.

A curvature interpretation. The shared path also provides a useful interpretation of what the prediction changes. If $F$ is a quadratic with constant Hessian H, then the uniform schedule in equation 8 yields the exact identity

$$
\pmb { H } ^ { r } = \nabla F ( \pmb { x } ^ { r } ) - \frac { h } { 2 } \pmb { H } \pmb { v } ^ { r } .\tag{14}
$$

Thus, relative to the $h = 0$ direction, the prediction incorporates a Hessian–direction term through gradient evaluations, without constructing a Hessian or communicating extra state. When $\pmb { v } ^ { r }$ approximates the current global gradient, positive-curvature directions are attenuated by this term; for general non-convex objectives, its effect depends on the local curvature and the accuracy of ${ \pmb v } ^ { r }$ Equation equation 14 is an interpretation of the update, not a claim of unconditional acceleration.

Our convergence analysis takes $h = \gamma$ and controls the resulting predictive bias jointly with stochastic gradient noise. It establishes a full-participation stationarity guarantee without a bounded-gradient or bounded-client-dissimilarity assumption. Under partial participation, the identity at each shared point holds only after expectation over client sampling; the sampled direction has an additional client-sampling variance. The basic CTP-FL algorithm does not remove that term.

## 4 THEORETICAL ANALYSIS

We analyze the non-convex objective

$$
F ( \pmb { x } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F _ { i } ( \pmb { x } ) , \qquad F _ { * } = \operatorname* { i n f } _ { \pmb { x } \in \mathbb { R } ^ { d } } F ( \pmb { x } ) > - \infty .\tag{15}
$$

Our analysis separates two sources of error in the server direction: stochastic-gradient noise and the displacement between the current model and the shared predictive trajectory. Under partial participation, client sampling introduces a third source of error. We use the following standard assumptions (Karimireddy et al., 2020; Cheng et al., 2024).

Assumption 4.1 (Smoothness). Each local objective $F _ { i }$ is L-smooth. That is, for every $i \in [ N ]$ and $\ b { x } , \ b { y } \in \mathbb { R } ^ { d }$

$$
\begin{array} { r } { \| \nabla F _ { i } ( { \pmb x } ) - \nabla F _ { i } ( { \pmb y } ) \| \le L \| { \pmb x } - { \pmb y } \| . } \end{array}\tag{16}
$$

Assumption 4.2 (Unbiased stochastic gradients). For every client i and every x, its mini-batch gradient $g _ { i } ( { \pmb x } ; { \pmb \xi } )$ satisfies

$$
\begin{array} { r } { \mathbb { E } _ { \xi } [ g _ { i } ( \pmb { x } ; \xi ) ] = \nabla F _ { i } ( \pmb { x } ) , \qquad \mathbb { E } _ { \xi } \| g _ { i } ( \pmb { x } ; \xi ) - \nabla F _ { i } ( \pmb { x } ) \| ^ { 2 } \le \sigma ^ { 2 } . } \end{array}\tag{17}
$$

The mini-batches used at different query points, clients, and communication rounds are conditionally independent.

Neither assumption bounds individual gradients or the differences $\nabla F _ { i } ( { \pmb x } ) - \nabla F ( { \pmb x } )$

Predictive direction. Let ${ \mathcal { F } } _ { r }$ denote the history before the mini-batches of round r are sampled. For the shared query points

$$
z _ { k } ^ { r } = { \pmb x } ^ { r } - t _ { k } { \pmb v } ^ { r } , \qquad t _ { k } = \left\{ \begin{array} { l l } { k h / ( K - 1 ) , } & { K > 1 , } \\ { 0 , } & { K = 1 , } \end{array} \right.
$$

define

$$
H ^ { r } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F ( z _ { k } ^ { r } ) .\tag{18}
$$

Under full participation, the server direction in Algorithm 1 obeys

$$
\mathbb { E } [ { \pmb v } ^ { r + 1 } \mid { \mathcal F } _ { r } ] = { \pmb H } ^ { r } , \qquad \mathbb { E } \big [ \| { \pmb v } ^ { r + 1 } - { \pmb H } ^ { r } \| ^ { 2 } \mid { \mathcal F } _ { r } \big ] \le \frac { \sigma ^ { 2 } } { N K } .\tag{19}
$$

In particular, heterogeneous client gradients cancel at each shared query point. The resulting direction estimates the average gradient along the predicted path, rather than the gradient at ${ \pmb x } ^ { r }$ . Smoothness bounds this predictive bias by

$$
\| \pmb { H } ^ { r } - \nabla F ( \pmb { x } ^ { r } ) \| \leq \frac { L } { K } \sum _ { k = 0 } ^ { K - 1 } t _ { k } \| \pmb { v } ^ { r } \| \leq \frac { L h } { 2 } \| \pmb { v } ^ { r } \| .\tag{20}
$$

Thus, the path length h controls a bias–variance trade-off without introducing a client-dissimilarity constant.

Theorem 4.3 (Full-participation convergence of CTP-FL). Suppose Assumptions 4.1 and 4.2 hold, and all N clients participate in every round. Initialize $\begin{array} { r } { \pmb { v } ^ { 0 } = \mathbf { 0 } , } \end{array}$ , set $h = \gamma$ , and let

$$
\Delta = F ( { \pmb x } ^ { 0 } ) - F _ { * } , \qquad 0 < \gamma \leq \frac { 1 } { 4 L } .
$$

Then Algorithm 1 satisfies

$$
\boxed { \frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } \leq \frac { 6 \Delta } { \gamma R } + \frac { 5 L \gamma \sigma ^ { 2 } } { N K } . }\tag{21}
$$

$I f \Delta > 0$ and

$$
\gamma = \left( 4 L + \sqrt { \frac { 5 L \sigma ^ { 2 } R } { 6 \Delta N K } } \right) ^ { - 1 } ,\tag{22}
$$

then

$$
\frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } \leq 2 \sqrt { \frac { 3 0 L \Delta \sigma ^ { 2 } } { N K R } } + \frac { 2 4 L \Delta } { R } .\tag{23}
$$

Table 1: Test accuracy (%) after 300 communication rounds on CIFAR-100 and Tiny-ImageNet. Each experiment uses $N = \mathrm { i 0 0 }$ clients, S = 10 participating clients per round, mini-batch size 50, and $K = 5 0$ gradient evaluations per participating client. Data are partitioned with Dirichlet concentrations $\alpha \in \{ 0 . 1 , 0 . 0 5 \}$ Bold denotes the highest accuracy and underlining denotes the second-highest accuracy among the reported methods in each column.
<table><tr><td rowspan="3">Method</td><td colspan="4">ResNet-18</td><td colspan="4">ViT-Tiny</td></tr><tr><td colspan="2">CIFAR-100</td><td colspan="2">Tiny-ImageNet</td><td colspan="2">CIFAR-100</td><td colspan="2">Tiny-ImageNet</td></tr><tr><td>Dir-0.1</td><td>Dir-0.05</td><td>Dir-0.1</td><td>Dir-0.05</td><td>Dir-0.1</td><td>Dir-0.05</td><td>Dir-0.1</td><td>Dir-0.05</td></tr><tr><td>FedAvg</td><td>60.17</td><td>56.75</td><td>47.48</td><td>43.80</td><td>27.24</td><td>23.42</td><td>15.68</td><td>14.05</td></tr><tr><td>SCAFFOLD</td><td>60.69</td><td>56.43</td><td>47.76</td><td>43.92</td><td>26.86</td><td>23.23</td><td>15.70</td><td>14.21</td></tr><tr><td>FedCM</td><td>66.61</td><td>62.65</td><td>41.16</td><td>36.00</td><td>28.23</td><td>25.74</td><td>18.88</td><td>18.15</td></tr><tr><td>CTP-FL</td><td>69.25</td><td>64.16</td><td>55.62</td><td>51.81</td><td>51.16</td><td>47.55</td><td>34.32</td><td>31.33</td></tr></table>

Theorem 4.3 establishes the same $\mathcal { O } \Big ( \sqrt { L \Delta \sigma ^ { 2 } / ( N K R ) } + L \Delta / R \Big )$ order as the full-participation guarantee of FedAvg-M (Cheng et al., 2024). It does not imply a strictly faster convergence rate. The methodological difference lies in how the K gradient evaluations are allocated: FedAvg-M advances client-specific local trajectories, whereas CTP-FL evaluates gradients along a common trajectory. The proof is given in Appendix A.

Scope under partial participation. For completeness, suppose a subset $S _ { r }$ of size $S < N$ is sampled uniformly without replacement and the server averages only the selected clients’ directions. Define

$$
H _ { i } ^ { r } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F _ { i } ( z _ { k } ^ { r } ) , \qquad D _ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| H _ { i } ^ { r } - H ^ { r } \| ^ { 2 } ,
$$

and

$$
\rho _ { S } = \frac { N - S } { S ( N - 1 ) } .
$$

Here $D _ { r }$ is the client dispersion along the observed predictive trajectory; the algorithm neither computes it nor assumes that it is bounded.

Theorem 4.4 (Partial-participation trajectory bound). Under Assumptions 4.1 and 4.2, uniform sampling without replacement, $h = \gamma$ , and $0 < \gamma \leq 1 / ( 4 L )$ ,

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( \pmb { x } ^ { r } ) \| ^ { 2 } \leq \frac { 6 \Delta } { \gamma R } + \frac { 5 L \gamma \sigma ^ { 2 } } { S K } } & { } \\ { \displaystyle + \frac { 5 L \gamma \rho _ { S } } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } [ D _ { r } ] . } \end{array}\tag{24}
$$

The last term is the finite-population variance caused by client sampling. Therefore, Theorem 4.4 does not give a uniform heterogeneity-independent rate when $S < N$ . This limitation is intrinsic to the uncorrected sampled direction: even at a minimizer of $F ,$ individual client gradients may be arbitrarily large and cancel only after full aggregation. The proof and an explicit two-client counterexample are given in Appendix A.

## 5 EXPERIMENTS

Experimental setup. We evaluate CTP-FL on CIFAR-100 (Krizhevsky, 2009) and Tiny-ImageNet (Le & Yang, 2015), using ResNet-18 (He et al., 2016) and ViT-Tiny (Dosovitskiy et al., 2020). To examine statistical heterogeneity, we partition the training data among $N = 1 0 0$ clients using a Dirichlet distribution with concentration $\alpha \in \{ 0 . 1 , 0 . 0 5 \}$ ; the smaller value represents a more concentrated allocation of classes across clients. At each of 300 communication rounds, $S = 1 0$ clients participate, and each participating client computes K = 50 mini-batch gradients with batch size 50. For CTP-FL, these gradients are evaluated at the K shared predictive points in Eq. equation 8; they are not K successive local model updates.

Baselines and comparison budget. We compare against FedAvg (McMahan et al., 2017), SCAF-FOLD (Karimireddy et al., 2020), and FedCM (Xu et al., 2021). All reported methods use the same number of communication rounds and participating clients. The value K = 50 fixes the number of mini-batch gradient evaluations for CTP-FL; for local-update baselines it corresponds to the number of local gradient steps. We report test accuracy at the end of round 300 in Table 1.

Main results. CTP-FL achieves the highest reported accuracy in all eight settings of Table 1. On CIFAR-100 with ResNet-18, it improves over the strongest listed baseline by 2.64 and 1.51 percentage points under Dir-0.1 and Dir-0.05, respectively. On Tiny-ImageNet with ResNet-18, the corresponding gains are 7.86 and 7.89 points. The differences are larger for ViT-Tiny: 22.93 and 21.81 points on CIFAR-100, and 15.44 and 13.18 points on Tiny-ImageNet. These comparisons describe the results at round 300; they do not, on their own, establish faster convergence or identify the cause of the gains.

Theory–experiment scope. These experiments use partial client participation, whereas Theorem 4.3 establishes a heterogeneity-independent rate under full participation. Under partial participation, Theorem 4.4 includes a trajectory-dependent client-sampling term. Consequently, the empirical performance in Table 1 should not be presented as verification of the full-participation rate.

## 6 CONCLUSION

We introduced CTP-FL, which uses the gradient evaluations available between communication rounds to probe a shared predictive trajectory. Because all clients evaluate gradients at the same points, their full-participation average estimates the global gradient along that trajectory, rather than combining gradients measured at divergent local models. The prediction span makes the resulting displacement bias explicit and controllable.

Under smoothness and bounded stochastic-gradient variance, we proved a non-convex fullparticipation stationarity rate of $\mathcal { O } \Big ( \sqrt { L \Delta \sigma ^ { 2 } / ( N K R ) } + L \Delta / R \Big )$ without assuming bounded gradients or bounded client-gradient dissimilarity. Each client computes K mini-batch gradients and uploads one model-sized vector per round. In the reported partial-participation experiments, CTP-FL achieves the highest final accuracy among the listed methods across CIFAR-100 and Tiny-ImageNet with ResNet-18 and ViT-Tiny.

Our theory and experiments have distinct scopes. The heterogeneity-independent rate applies to full participation; sampling clients introduces an additional variance term that the current algorithm does not eliminate. Further work should examine control mechanisms for partial participation and determine when probing a predictive trajectory improves on querying the current model alone under matched computation and communication budgets.

## REFERENCES

Ziheng Cheng, Xinmeng Huang, Pengfei Wu, and Kun Yuan. Momentum benefits non-IID federated learning simply and provably. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=TdhkAcXkRi.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. SCAFFOLD: Stochastic controlled averaging for federated learning. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 5132–5143. PMLR, 2020. URL https://proceedings.mlr.press/v119/karimireddy20a.html.

Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, and Ananda Theertha Suresh. Mime: Mimicking centralized stochastic algorithms in federated learning, 2021. URL https://arxiv.org/abs/2008.03606.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015.

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. In Proceedings ofMachine Learning and Systems, volume 2, 2020. URL https://proceedings.mlsys.org/paper/2020/hash/ 1f5fe83998a09396ebe6477d9475ba0c-Abstract.html.

Junkang Liu, Fanhua Shang, Yuanyuan Liu, Hongying Liu, Yuangang Li, and YunXiang Gong. Fedbcgd: Communication-efficient accelerated block coordinate gradient descent for federated learning. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 2955–2963, 2024.

Junkang Liu, Yuanyuan Liu, Fanhua Shang, Hongying Liu, Jin Liu, and Wei Feng. Improving generalization in federated learning with highly heterogeneous data via momentum-based stochastic controlled weight averaging. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 38894–38939. PMLR, 2025a. URL https://proceedings.mlr.press/v267/liu25am.html.

Junkang Liu, Fanhua Shang, Yuxuan Tian, Hongying Liu, and Yuanyuan Liu. Consistency of local and global flatness for federated learning. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pp. 3875–3883, 2025b. doi: 10.1145/3746027.3755226.

Junkang Liu, Fanhua Shang, Junchao Zhou, Hongying Liu, Yuanyuan Liu, and Jin Liu. FedMuon: Accelerating federated learning with matrix orthogonalization. arXiv preprint arXiv:2510.27403, 2025c. doi: 10.48550/arXiv.2510.27403. URL https://arxiv.org/abs/2510.27403.

Junkang Liu, Yuxuan Tian, Fanhua Shang, Yuanyuan Liu, Hongying Liu, Junchao Zhou, and Daorui Ding. DP-FedPGN: Finding global flat minima for differentially private federated learning via penalizing gradient norm, 2025d. URL https://arxiv.org/abs/2510.27504.

Junkang Liu, Fanhua Shang, Hongying Liu, Jin Liu, Weixin An, and Yuanyuan Liu. Taming preconditioner drift: Unlocking the potential of second-order optimizers for federated learning on non-iid data, 2026a. URL https://arxiv.org/abs/2602.19271.

Junkang Liu, Fanhua Shang, Hongying Liu, Yuxuan Tian, Yuanyuan Liu, Jin Liu, Kewen Zhu, and Zhouchen Lin. FedAdamW: A communication-efficient optimizer with convergence and generalization guarantees for federated large models. Proceedings of the AAAI Conference on Artificial Intelligence, 40:23748–23756, 2026b. doi: 10.1609/aaai.v40i28.39549.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Proceedings ofthe 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pp. 1273–1282. PMLR, 2017. URL https://proceedings. mlr.press/v54/mcmahan17a.html.

Sebastian U. Stich. Local SGD converges fast and communicates little. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= S1g2JnRcFX.

Blake Woodworth, Kumar Kshitij Patel, Sebastian Stich, Zhen Dai, Brian Bullins, Brendan McMahan, Ohad Shamir, and Nathan Srebro. Is local SGD better than minibatch SGD? In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 10334–10343. PMLR, 2020. URL https://proceedings.mlr. press/v119/woodworth20a.html.

Jing Xu, Sen Wang, Liwei Wang, and Andrew Chi-Chih Yao. Fedcm: Federated learning with client-level momentum. arXiv preprint arXiv:2106.10874, 2021.

Michael Zhang, James Lucas, Jimmy Ba, and Geoffrey E. Hinton. Lookahead optimizer: k steps forward, 1 step back. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/ 90fd4f88f588ae64038134f1eeaa023f-Abstract.html.

## A CONVERGENCE ANALYSIS OF CTP-FL

We consider

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } F ( { \pmb x } ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F _ { i } ( { \pmb x } ) , \qquad F _ { * } : = \operatorname* { i n f } _ { { \pmb x } } F ( { \pmb x } ) > - \infty .\tag{25}
$$

The following assumptions match the standard smoothness and stochastic gradient assumptions used in FedAvg-M.

Assumption A.1 (Smoothness). For each client $i , F _ { i }$ is L-smooth:

$$
\begin{array} { r } { \| \nabla F _ { i } ( { \pmb x } ) - \nabla F _ { i } ( { \pmb y } ) \| \leq L \| { \pmb x } - { \pmb y } \| \quad \mathrm { f o r ~ a l l } ~ { \pmb x } , { \pmb y } \in \mathbb { R } ^ { d } . } \end{array}
$$

Assumption A.2 (Stochastic gradients). For each client i and every $^ { \mathbf { \delta x } , }$

$$
\mathbb { E } _ { \xi } [ g _ { i } ( \pmb { x } ; \xi ) ] = \nabla F _ { i } ( \pmb { x } ) , \qquad \mathbb { E } _ { \xi } \| g _ { i } ( \pmb { x } ; \xi ) - \nabla F _ { i } ( \pmb { x } ) \| ^ { 2 } \leq \sigma ^ { 2 } .
$$

Fresh mini-batches are sampled independently across clients, local gradient evaluations, and communication rounds.

CTP-FL update. Initialize $\mathbf { \nabla } \mathbf { v } ^ { 0 } = \mathbf { 0 }$ . At round $r ,$ let $S _ { r }$ contain S clients, sampled uniformly without replacement, independently of the mini-batches. Set

$$
t _ { k } = \left\{ \frac { k \gamma } { K - 1 } , \begin{array} { l l } { { K \geq 2 , } } & { { } } \\ { { 0 , } } & { { K = 1 , } } \end{array} \right. \quad z _ { k } ^ { r } = { \pmb x } ^ { r } - t _ { k } { \pmb v } ^ { r } , \quad k = 0 , \ldots , K - 1 .\tag{26}
$$

Each selected client computes

$$
{ \pmb u } _ { i } ^ { r } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } g _ { i } ( z _ { k } ^ { r } ; \xi _ { i , k } ^ { r } ) .\tag{27}
$$

The server updates

$$
\pmb { v } ^ { r + 1 } = \frac { 1 } { S } \sum _ { i \in \mathcal { S } _ { r } } \pmb { u } _ { i } ^ { r } , \qquad \pmb { x } ^ { r + 1 } = \pmb { x } ^ { r } - \gamma \pmb { v } ^ { r + 1 } .\tag{28}
$$

Full participation corresponds to $S = N$

Define

$$
\boldsymbol { h } _ { i } ^ { r } : = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F _ { i } ( \boldsymbol { z } _ { k } ^ { r } ) , \qquad \boldsymbol { h } ^ { r } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \boldsymbol { h } _ { i } ^ { r } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F ( \boldsymbol { z } _ { k } ^ { r } ) .\tag{29}
$$

The predictive bias is

$$
\pmb { b } ^ { r } : = \pmb { h } ^ { r } - \nabla F ( \pmb { x } ^ { r } ) .
$$

For partial participation, define the observed trajectory dispersion

$$
D _ { r } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \pmb { h } _ { i } ^ { r } - \pmb { h } ^ { r } \| ^ { 2 } .\tag{30}
$$

This is an analysis quantity, not an assumption or an input to CTP-FL.

Lemma A.3 (Bias and conditional variance). Under Assumptions $A . l { - } A . 2 ,$ conditional on the history before client sampling at round $^ { r , }$

$$
\| \pmb { b } ^ { r } \| ^ { 2 } \leq \frac { L ^ { 2 } \gamma ^ { 2 } } { 4 } \| \pmb { v } ^ { r } \| ^ { 2 } .\tag{31}
$$

Moreover,

$$
\mathbb { E } _ { r } [ { \pmb v } ^ { r + 1 } ] = { \pmb h } ^ { r } ,\tag{32}
$$

and

$$
\mathbb { E } _ { r } \| { \pmb v } ^ { r + 1 } - { \pmb h } ^ { r } \| ^ { 2 } \le \frac { \sigma ^ { 2 } } { S K } + \frac { N - S } { S ( N - 1 ) } D _ { r } ,\tag{33}
$$

where the second term is defined to be zero when $S = N$

Proof. Since F is L-smooth and $\begin{array} { r } { K ^ { - 1 } \sum _ { k = 0 } ^ { K - 1 } t _ { k } \le \gamma / 2 } \end{array}$

$$
\begin{array} { r l } & { \| \boldsymbol { b } ^ { r } \| \leq \displaystyle \frac { 1 } { K } \displaystyle \sum _ { k = 0 } ^ { K - 1 } \| \nabla F ( \boldsymbol { z } _ { k } ^ { r } ) - \nabla F ( \boldsymbol { x } ^ { r } ) \| } \\ & { \qquad \leq \displaystyle \frac { L } { K } \displaystyle \sum _ { k = 0 } ^ { K - 1 } t _ { k } \| \boldsymbol { v } ^ { r } \| \leq \displaystyle \frac { L \gamma } { 2 } \| \boldsymbol { v } ^ { r } \| , } \end{array}\tag{34}
$$

which proves equation 31. Unbiasedness of the stochastic gradients and uniform client sampling give equation 32.

For equation 33, first condition on $S _ { r }$ . The independent stochastic gradient errors have mean zero and their average has second moment at most $\sigma ^ { 2 } \overset { \cdot } { / } ( S K )$ . The remaining variance is that of sampling without replacement from the finite population $\{ h _ { i } ^ { r } \} _ { i = 1 } ^ { N } \{$

$$
\mathbb { E } _ { r } \left\| \frac { 1 } { S } \sum _ { i \in S _ { r } } \pmb { h } _ { i } ^ { r } - \pmb { h } ^ { r } \right\| ^ { 2 } = \frac { N - S } { S ( N - 1 ) } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \pmb { h } _ { i } ^ { r } - \pmb { h } ^ { r } \| ^ { 2 } .
$$

The two errors have zero cross term, completing the proof.

Theorem A.4 (Full-participation convergence). Suppose Assumptions A.1–A.2 hold, all N clients participate, and $0 < \bar { \gamma } \le \bar { 1 / ( 4 L ) }$ . Let $\Delta \bar { : } = F ( { \pmb x } ^ { 0 } ) \bar { - } F _ { * }$ . Then CTP-FL satisfies

$$
\boxed { \frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } \leq \frac { 6 \Delta } { \gamma R } + \frac { 5 L \gamma \sigma ^ { 2 } } { N K } . }\tag{35}
$$

In particular, no bounded-gradient or bounded-data-heterogeneity assumption is needed.

Proof. Write

$$
a : = \frac { L ^ { 2 } \gamma ^ { 2 } } 4 , \qquad Z _ { r } : = \mathbb { E } _ { r } \| \pmb { v } ^ { r + 1 } - \pmb { h } ^ { r } \| ^ { 2 } .
$$

Here $a \leq 1 / 6 4$ and, under full participation, $Z _ { r } \le \sigma ^ { 2 } / ( N K )$

Smoothness of $F .$ , the server update, and $\mathbb { E } _ { r } [ { \pmb v } ^ { r + 1 } ] = \nabla F ( { \pmb x } ^ { r } ) + { \pmb b } ^ { r }$ imply

$$
\begin{array} { l } { { \displaystyle \mathbb { E } _ { \boldsymbol { r } } [ F ( { \pmb x } ^ { r + 1 } ) ] \leq F ( { \pmb x } ^ { r } ) - \gamma \langle \nabla F ( { \pmb x } ^ { r } ) , \nabla F ( { \pmb x } ^ { r } ) + { \pmb b } ^ { r } \rangle } } \\ { ~ + \frac { L \gamma ^ { 2 } } { 2 } \left( \| \nabla F ( { \pmb x } ^ { r } ) + { \pmb b } ^ { r } \| ^ { 2 } + Z _ { r } \right) . } \end{array}\tag{36}
$$

Using $- \langle p , q \rangle \leq \| p \| ^ { 2 } / 4 + \| q \| ^ { 2 } , \| p + q \| ^ { 2 } \leq 2 \| p \| ^ { 2 } + 2 \| q \| ^ { 2 }$ , and $L \gamma \leq 1 / 4$ gives the conservative bound

$$
\mathbb { E } _ { r } [ F ( { \pmb x } ^ { r + 1 } ) ] \le F ( { \pmb x } ^ { r } ) - \frac { \gamma } { 4 } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } + \frac { 3 \gamma } { 2 } \| { \pmb b } ^ { r } \| ^ { 2 } + \frac { L \gamma ^ { 2 } } { 2 } Z _ { r } .\tag{37}
$$

For compactness, set

$$
A _ { R } : = \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \Vert \nabla F ( { \pmb x } ^ { r } ) \Vert ^ { 2 } , \quad V _ { R } : = \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \Vert { \pmb v } ^ { r } \Vert ^ { 2 } , \quad \mathcal { Z } _ { R } : = \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } [ Z _ { r } ] .
$$

Summing equation 37, applying $F ( \pmb { x } ^ { R } ) \geq F _ { * }$ , and using Lemma A.3, we obtain

$$
A _ { R } \leq \frac { 4 \Delta } { \gamma } + 6 a V _ { R } + 2 L \gamma \mathcal { Z } _ { R } .\tag{38}
$$

Next, $\mathbf { \boldsymbol { v } } ^ { 0 } = \mathbf { 0 }$ and, for $r \geq 1$

$$
\pmb { v } ^ { r } = \nabla F ( \pmb { x } ^ { r - 1 } ) + \pmb { b } ^ { r - 1 } + \big ( \pmb { v } ^ { r } - \pmb { h } ^ { r - 1 } \big ) .
$$

The final parenthesized term has conditional mean zero. Consequently,

$$
\begin{array} { r l } & { \mathbb { E } \| \pmb { v } ^ { r } \| ^ { 2 } \leq 2 \mathbb { E } \| \nabla F ( \pmb { x } ^ { r - 1 } ) \| ^ { 2 } + 2 \mathbb { E } \| b ^ { r - 1 } \| ^ { 2 } + 2 \mathbb { E } [ Z _ { r - 1 } ] } \\ & { \qquad \leq 2 \mathbb { E } \| \nabla F ( \pmb { x } ^ { r - 1 } ) \| ^ { 2 } + 2 a \mathbb { E } \| \pmb { v } ^ { r - 1 } \| ^ { 2 } + 2 \mathbb { E } [ Z _ { r - 1 } ] . } \end{array}\tag{39}
$$

Summing over r and extending nonnegative sums yields

$$
V _ { R } \leq 2 A _ { R } + 2 a V _ { R } + 2 \mathcal { Z } _ { R } .\tag{40}
$$

Since $a \leq 1 / 6 4 .$

$$
V _ { R } \leq \frac { 2 } { 1 - 2 a } ( A _ { R } + \mathcal { Z } _ { R } ) \leq 3 ( A _ { R } + \mathcal { Z } _ { R } ) .
$$

Substituting this into equation 38 gives

$$
( 1 - 1 8 a ) A _ { R } \leq \frac { 4 \Delta } { \gamma } + ( 1 8 a + 2 L \gamma ) \mathcal { Z } _ { R } .
$$

Because $a \leq 1 / 6 4$

$$
A _ { R } \leq \frac { 6 \Delta } { \gamma } + ( 2 6 a + 3 L \gamma ) \mathcal { Z } _ { R } .\tag{41}
$$

Finally, $\mathcal { Z } _ { R } \leq R \sigma ^ { 2 } / ( N K )$ and $a = ( L \gamma ) ^ { 2 } / 4 \leq L \gamma / 1 6$ , so that $2 6 a + 3 L \gamma < 5 L \gamma$ . Substitution into equation 41 proves equation 35. □

Theorem A.5 (Partial-participation trajectory bound). Suppose Assumptions $A . I { - } A . 2$ hold, $1 \leq S <$ $N ,$ and $0 < \gamma \leq 1 / ( 4 \bar { L } )$ ). With $D _ { r }$ defined in equation 30,

$$
\begin{array} { r l r } {  {  \frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } \leq \frac { 6 \Delta } { \gamma R } + \frac { 5 L \gamma \sigma ^ { 2 } } { S K }  } } \\ & { } & { \qquad +  \frac { 5 L \gamma ( N - S ) } { S ( N - 1 ) R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } [ D _ { r } ] .  } \end{array}\tag{42}
$$

This theorem imposes no bound on $D _ { r } .$ . Accordingly, equation 42 is a valid trajectory-dependent inequality, but does not by itself guarantee a uniform heterogeneity-independent convergence rate.

Proof. The proof of Theorem A.4 up to equation 41 applies unchanged. By Lemma $\mathrm { A . } 3 .$

$$
\mathcal { Z } _ { R } \leq \frac { R \sigma ^ { 2 } } { S K } + \frac { N - S } { S ( N - 1 ) } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } [ D _ { r } ] .
$$

Substitute this bound into equation 41 and use $2 6 a + 3 L \gamma < 5 L \gamma$

Proposition A.6 (Why client sampling cannot be ignored). For the CTP-FL update equation 28 with $S < N$ , Assumptions A.1–A.2 alone cannot yield a uniform stationarity bound depending only on $\Delta , L , \sigma , N , S , \bar { K } , R$ , and $\gamma$ that vanishes when $\Delta = \sigma = 0$

Proof. Take $N = 2 , S = 1 , d = 1$ , and exact gradients $( \sigma = 0 )$ . For any $M > 0$ , define

$$
F _ { 1 } ( x ) = \frac { L } { 2 } x ^ { 2 } + M x , \qquad F _ { 2 } ( x ) = \frac { L } { 2 } x ^ { 2 } - M x .
$$

Then $F ( x ) = L x ^ { 2 } / 2$ . Initialize $x ^ { 0 } = 0$ and $v ^ { 0 } = 0 .$ Thus $x ^ { 0 }$ is the global minimizer and $\Delta = 0 .$ . At round zero, every predictive location equals zero, regardless of K. The single sampled client returns $v ^ { 1 } = M \mathrm { o r } v ^ { 1 } = - M$ , each with probability $1 / 2$ . Hence

$$
x ^ { 1 } = - \gamma v ^ { 1 } , \qquad \mathbb { E } \| \nabla F ( x ^ { 1 } ) \| ^ { 2 } = L ^ { 2 } \gamma ^ { 2 } M ^ { 2 } .
$$

For any $R \ \geq \ 2 ,$ , the average stationarity measure therefore contains the strictly positive term $L ^ { 2 } \gamma ^ { 2 } \dot { M } ^ { 2 } / R$ , which can be made arbitrarily large by increasing M, although $\Delta = \sigma = 0$ □

Theorem A.7 (Convergence of CTP-FL under full participation). Let

$$
F ( \pmb { x } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F _ { i } ( \pmb { x } ) , \qquad \Delta = F ( \pmb { x } ^ { 0 } ) - F _ { * } , \qquad F _ { * } = \operatorname* { i n f } _ { \pmb { x } } F ( \pmb { x } ) > - \infty .
$$

Suppose that every $F _ { i }$ is L-smooth and that the stochastic gradients are unbiased, independent across clients and gradient evaluations, and have variance at most $\sigma ^ { 2 }$ . All N clients participate in every round.

Run CTP-FL with $\mathbf { \boldsymbol { v } } ^ { 0 } = \mathbf { 0 }$ and predictive locations

$$
z _ { k } ^ { r } = { \pmb { x } } ^ { r } - t _ { k } { \pmb v } ^ { r } , \qquad t _ { k } = \left\{ \begin{array} { l l } { k \gamma / ( K - 1 ) , } & { K \ge 2 , } \\ { 0 , } & { K = 1 . } \end{array} \right.
$$

$I f 0 < \gamma \leq 1 / ( 4 L )$ , then

$$
\frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } \leq \frac { 6 \Delta } { \gamma R } + \frac { 5 L \gamma \sigma ^ { 2 } } { N K } .\tag{43}
$$

In particular, for $\Delta > 0 ,$ , choose

$$
\left. \gamma = \left( 4 L + \sqrt { \frac { 5 L \sigma ^ { 2 } R } { 6 \Delta N K } } \right) ^ { - 1 } . \right.\tag{44}
$$

Then

$$
\left| \frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } \leq 2 \sqrt { \frac { 3 0 L \Delta \sigma ^ { 2 } } { N K R } } + \frac { 2 4 L \Delta } { R } = \mathcal { O } \left( \sqrt { \frac { L \Delta \sigma ^ { 2 } } { N K R } } + \frac { L \Delta } { R } \right) . \right|\tag{45}
$$

No bounded-gradient or bounded-data-heterogeneity assumption is imposed.

Proof. Let $\mathcal { F } _ { r }$ denote the history before the fresh mini-batches of round r are sampled. Define

$$
\pmb { h } ^ { r } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F ( \pmb { z } _ { k } ^ { r } ) , \qquad \pmb { b } ^ { r } = \pmb { h } ^ { r } - \nabla F ( \pmb { x } ^ { r } ) , \qquad \ b { a } = \frac { L ^ { 2 } \gamma ^ { 2 } } { 4 } .
$$

Because all clients evaluate their gradients at the same $\boldsymbol { z } _ { k } ^ { r } ,$ , unbiasedness and independence give

$$
\mathbb { E } [ { \pmb v } ^ { r + 1 } \mid { \mathcal F } _ { r } ] = { \pmb h } ^ { r } , \qquad \mathbb { E } \big [ \| { \pmb v } ^ { r + 1 } - { \pmb h } ^ { r } \| ^ { 2 } \bigm | { \mathcal F } _ { r } \big ] \le \frac { \sigma ^ { 2 } } { N K } .\tag{46}
$$

Smoothness and $\textstyle K ^ { - 1 } \sum _ { k = 0 } ^ { K - 1 } t _ { k } \leq \gamma / 2$ yield

$$
\| \pmb { b } ^ { r } \| \leq \frac { L } { K } \sum _ { k = 0 } ^ { K - 1 } t _ { k } \| \pmb { v } ^ { r } \| \leq \frac { L \gamma } { 2 } \| \pmb { v } ^ { r } \| , \qquad \| \pmb { b } ^ { r } \| ^ { 2 } \leq a \| \pmb { v } ^ { r } \| ^ { 2 } .\tag{47}
$$

Since $\pmb { x } ^ { r + 1 } = \pmb { x } ^ { r } - \gamma \pmb { v } ^ { r + 1 }$ , the descent lemma and equation 46 imply

$$
\begin{array} { r l } { \displaystyle \mathbb { E } [ F ( { \pmb x } ^ { r + 1 } ) \mid \mathcal { F } _ { r } ] \leq F ( { \pmb x } ^ { r } ) - \gamma \left. \nabla F ( { \pmb x } ^ { r } ) , \nabla F ( { \pmb x } ^ { r } ) + { \pmb b } ^ { r } \right. } & { } \\ { \displaystyle + \frac { L \gamma ^ { 2 } } { 2 } \left( \| \nabla F ( { \pmb x } ^ { r } ) + { \pmb b } ^ { r } \| ^ { 2 } + \frac { \sigma ^ { 2 } } { N K } \right) . } & { } \end{array}\tag{48}
$$

Apply

$$
- \langle { \pmb p } , { \pmb q } \rangle \le \frac { 1 } { 4 } \| { \pmb p } \| ^ { 2 } + \| { \pmb q } \| ^ { 2 } , \qquad \| { \pmb p } + { \pmb q } \| ^ { 2 } \le 2 \| { \pmb p } \| ^ { 2 } + 2 \| { \pmb q } \| ^ { 2 } .
$$

Because $L \gamma \leq 1 / 4$ , we obtain

$$
\mathbb { E } [ F ( { \pmb x } ^ { r + 1 } ) \mid { \mathcal F } _ { r } ] \le F ( { \pmb x } ^ { r } ) - \frac { \gamma } { 4 } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } + \frac { 3 \gamma } { 2 } \| { \pmb b } ^ { r } \| ^ { 2 } + \frac { L \gamma ^ { 2 } \sigma ^ { 2 } } { 2 N K } .\tag{49}
$$

Set

$$
A _ { R } = \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } , \qquad V _ { R } = \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| { \pmb v } ^ { r } \| ^ { 2 } , \qquad Q _ { R } = \frac { R \sigma ^ { 2 } } { N K } .
$$

Summing equation 49, using $F ( \pmb { x } ^ { R } ) \geq F _ { * }$ , and applying equation 47 give

$$
A _ { R } \leq \frac { 4 \Delta } { \gamma } + 6 a V _ { R } + 2 L \gamma Q _ { R } .\tag{50}
$$

Furthermore, $\mathbf { \boldsymbol { v } } ^ { 0 } = \mathbf { 0 }$ and

$$
\pmb { v } ^ { r } = \nabla F ( \pmb { x } ^ { r - 1 } ) + \pmb { b } ^ { r - 1 } + ( \pmb { v } ^ { r } - \pmb { h } ^ { r - 1 } ) \quad ( r \geq 1 ) .
$$

The last term has conditional mean zero. Thus

$$
V _ { R } \leq 2 A _ { R } + 2 a V _ { R } + 2 Q _ { R } .\tag{51}
$$

Since $\gamma \leq 1 / ( 4 L )$ , we have $a \leq 1 / 6 4$ . Therefore equation 51 implies

$$
V _ { R } \leq \frac { 2 } { 1 - 2 a } ( A _ { R } + Q _ { R } ) \leq 3 ( A _ { R } + Q _ { R } ) .
$$

Substituting into equation 50 gives

$$
( 1 - 1 8 a ) A _ { R } \leq \frac { 4 \Delta } { \gamma } + ( 1 8 a + 2 L \gamma ) Q _ { R } .
$$

As $a \leq 1 / 6 4$

$$
A _ { R } \leq \frac { 6 \Delta } { \gamma } + ( 2 6 a + 3 L \gamma ) Q _ { R } .
$$

Finally, $a = ( L \gamma ) ^ { 2 } / 4 \leq L \gamma / 1 6 .$ , so $2 6 a + 3 L \gamma < 5 L \gamma$ . This proves equation 43.

For the parameter choice equation 44, $\gamma \leq 1 / ( 4 L )$ and

$$
\frac { 6 \Delta } { \gamma R } = \frac { 2 4 L \Delta } R + \sqrt { \frac { 3 0 L \Delta \sigma ^ { 2 } } { N K R } } .
$$

Also,

$$
\frac { 5 L \gamma \sigma ^ { 2 } } { N K } \leq \sqrt { \frac { 3 0 L \Delta \sigma ^ { 2 } } { N K R } } ,
$$

which proves equation 45.

Theorem A.8 (CTP-FL under partial participation). Under the assumptions ofTheorem A.7, suppose each round samples $S < N$ clients uniformly without replacement. Define

$$
\pmb { h } _ { i } ^ { r } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \nabla F _ { i } ( z _ { k } ^ { r } ) , \qquad \pmb { h } ^ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \pmb { h } _ { i } ^ { r } ,
$$

and

$$
D _ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \pmb { h } _ { i } ^ { r } - \pmb { h } ^ { r } \| ^ { 2 } .
$$

For $0 < \gamma \leq 1 / ( 4 L )$

$$
\begin{array} { r l r } {  { \displaystyle \frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \| \nabla F ( { \pmb x } ^ { r } ) \| ^ { 2 } \leq \frac { 6 \Delta } { \gamma R } + \frac { 5 L \gamma \sigma ^ { 2 } } { S K } } } \\ & { } & { \qquad + \frac { 5 L \gamma ( N - S ) } { S ( N - 1 ) R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } [ D _ { r } ] . } \end{array}\tag{52}
$$

No uniform upper bound on $D _ { r }$ is assumed.

Proof. Conditional on the history before client sampling, uniform sampling gives $\mathbb { E } _ { r } [ { \pmb v } ^ { r + 1 } ] = { \pmb h } ^ { r }$ The finite-population variance identity and independent mini-batches give

$$
\mathbb { E } _ { r } \| { \pmb v } ^ { r + 1 } - { \pmb h } ^ { r } \| ^ { 2 } \le \frac { \sigma ^ { 2 } } { S K } + \frac { N - S } { S ( N - 1 ) } D _ { r } .
$$

Repeat the proof of Theorem A.7, replacing $R \sigma ^ { 2 } / ( N K )$ throughout by

$$
\frac { R \sigma ^ { 2 } } { S K } + \frac { N - S } { S ( N - 1 ) } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } [ D _ { r } ] .
$$

The same bias and descent bounds yield equation 52.

## B RELATED WORK

Communication-efficient federated optimization. Communication-efficient federated learning commonly allows clients to perform multiple local updates between aggregation rounds. FedBCGD reduces the amount of information transmitted in each round by updating and communicating parameter blocks, and further incorporates drift control and variance reduction in its accelerated variant (Liu et al., 2024). This approach addresses the size of each message. In contrast, CTP-FL communicates one model-sized vector per participating client and changes where local gradients are evaluated: clients query the same server-defined predictive trajectory, so their updates can be averaged without first combining models that have followed different local trajectories.

Alignment under heterogeneous data. Several recent methods study different forms of local– global misalignment. FedSWA and FedMoSWA use stochastic weight averaging and momentumbased control to improve generalization under highly heterogeneous data (Liu et al., 2025a). FedNSAM examines the mismatch between local and global flatness and uses a global Nesterov direction to improve their consistency (Liu et al., 2025b). These methods primarily target the properties of the resulting solution, including flatness and generalization. Our focus is the geometry of gradient evaluation during a communication round: when all clients evaluate at common points, averaging their gradients estimates the gradient of the global objective at those points, irrespective of how different the individual client gradients are.

Federated adaptive and structured optimizers. FedAdamW combines local correction, decoupled weight decay, and aggregation of second-moment estimates for federated large-model training (Liu et al., 2026b). FedMuon exploits matrix orthogonalization and local–global alignment to improve federated optimization of matrix-structured parameters (Liu et al., 2025c). FedPAC identifies precon ditioner drift as a source of instability when local second-order optimizers induce incompatible client geometries, and proposes preconditioner alignment and update correction (Liu et al., 2026a). Unlike these optimizer-specific mechanisms, CTP-FL applies to stochastic-gradient evaluations without transmitting moments or preconditioners. Its common trajectory aligns the locations of gradient evaluation rather than optimizer states.

Global flatness and privacy. DP-FedPGN encourages globally flat solutions in client-level differentially private federated learning through a global gradient-norm penalty (Liu et al., 2025d). Its objective and privacy accounting are different from ours. We cite it because it likewise illustrates that a quantity defined by the global objective need not be faithfully represented by independently optimized local objectives.