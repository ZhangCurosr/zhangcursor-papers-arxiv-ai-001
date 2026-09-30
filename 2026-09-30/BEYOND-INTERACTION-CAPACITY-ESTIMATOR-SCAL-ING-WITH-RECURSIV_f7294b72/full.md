# BEYOND INTERACTION CAPACITY: ESTIMATOR SCAL-ING WITH RECURSIVE MODELS FOR CTR PREDICTION

Shivang Chopra Georgia Institute of Technology shivangchopra11@gatech.edu

Zsolt Kira Georgia Institute of Technology zkira@gatech.edu

Fotis Iliopoulos Google Research fotisi@google.com

Gaurav Menghani Google Research gmenghani@google.com

## ABSTRACT

Click-Through Rate (CTR) prediction, a core task in recommendation and advertising systems, relies on modeling interactions among sparse categorical features. Explicit cross networks are a central paradigm for CTR prediction, and recent progress has largely come from increasing the interaction capacity of a single predictor through deeper cross networks and more expressive cross operators. We revisit whether continually increasing interaction capacity remains the most effective way to improve predictive performance, and find that its benefits quickly exhibit diminishing returns even as capacity continues to grow. This motivates a complementary scaling direction that we call estimator scaling, where additional resources are used to incorporate multiple related estimators rather than only enlarging a single predictor. Through theoretical analysis and diagnostic experiments, we show that the gains from estimator scaling are governed by the amount of non-shared predictive variation available across estimators. However, exploiting this variation naively can be expensive: independently trained models provide substantial estimator diversity but require deployment cost to grow with ensemble size. This motivates a parameter-efficient realization of estimator scaling that can incorporate diversity from multiple estimator sources without maintaining multiple full models. Building on this view, we introduce RECursive Averaged Predictor (RECAP), a parameter-efficient recursive CTR model that operationalizes estimator scaling at three levels: distillation across independently trained models, exponential moving averaging over training trajectories, and aggregation over inference-time routes within a weight-shared recursive backbone. This design improves predictive performance without requiring parameter growth proportional to the ensemble size. Experiments across multiple CTR benchmarks show that RECAP approaches the accuracy of a five-model ensemble of a strong cross-network baseline at one fifth of its deployed parameters, while doubling the baseline’s interaction capacity recovers only a third of that gain. Overall, these results establish new state-of-the-art predictive performance on standard benchmarks, while placing the proposed approach on a favorable performance–parameter Pareto frontier.

## 1 INTRODUCTION

Click-through rate (CTR) prediction is a central component of modern recommendation and advertising systems, where accurate ranking depends critically on modeling interactions among sparse categorical and numerical features Zhu et al. (2022); Wang et al. (2017). A long line of work has therefore improved CTR models by increasing the interaction capacity of a single predictor through deeper cross networks, and richer interaction operators Wang et al. (2017; 2021a); Guo et al. (2017); Zhu et al. (2023). In explicit cross networks, this capacity can be increased along several architectural dimensions, most directly by making interaction networks deeper or by combining multiple interaction branches with complementary operators. Deep & Cross Networks (DCNs) Wang et al. (2017)

Non-shared variance removed, V(1−1/M)  
![](images/015e19266ea8f3a4af64c2f8c7369f856792b2176ea95048e9b1b1356d8795c3.jpg)

![](images/23f37124d4bbf0ec23ce8960f8b31ccba44d66fc320db4145f855850988e5d37.jpg)

![](images/3f4b487120aadebf4630c2241a005da9c6d889b552c10b82c6af4638c9270fcf.jpg)  
(d) RECAP shifts the deployment frontie

![](images/dee3064515319dea6226b4bb4479d160115f666860fb4468cb49acfd4a6aa8fb.jpg)  
Figure 1: Diagnostic analysis of interaction-capacity and estimator scaling on Criteo. (a) Increasing interaction depth and breadth exhibits diminishing returns. (b) Additional estimators provide further gains, with independent runs outperforming trajectories and shared-model routes. (c) Averaging gains track the non-shared predictive variance available to aggregation (Section 3). (d) RECAP captures much of the ensemble gain at approximately a single-model deployment budget. Diagnosis results on additional datasets can be found in Appendix A.

and their derivatives exemplify this paradigm, explicitly scaling the complexity of feature interactions represented by the model. Recent architectures such as Quadratic Neural Networks (QNN) and Fusing Cross Networks (FCN) push this direction further by constructing interactions whose order grows exponentially with depth Li et al. (2025; 2026). This progression raises a natural question: once a sufficiently expressive interaction model is available, is continually increasing interaction capacity still the most effective way to improve CTR prediction?

We test this assumption by scaling interaction capacity along two concrete architectural axes: depth, through additional cross layers, and breadth, through additional interaction branches. As shown in Fig ure 1(a), increasing interaction depth or adding interaction branches initially improves performance, but the marginal benefit quickly shrinks as interaction capacity increases. This diminishing-return regime raises a second question: if additional resources are no longer best spent increasing the capacity of a single predictor, how else can they improve prediction?

In Section 3, we conduct a theoretical analysis that points to a complementary scaling direction that we call estimator scaling. Rather than enlarging the function class of a single predictor, estimator scaling uses additional training or inference resources to incorporate multiple estimators of the same predictive function. Prior work has theoretically analyzed the underlying benefit of averaging less-correlated estimators Wood et al. (2023); our focus is on its role as a scaling principle for CTR models, where recent progress has predominantly come from increasing feature-interaction capacity. We show that its benefit is governed by the non-shared predictive variation exposed by different estimator sources which offer distinct diversity–deployment-cost trade-offs. This yields a simple scaling principle: gains are larger when additional estimators expose more non-shared variation and diminish as their predictions become increasingly correlated.

As shown in Figure 1(b), our diagnostic experiments support this prediction across three distinct sources of estimators: averaging independently trained models, averaging predictors sampled along a single training trajectory, and averaging alternative inference-time routes through a shared model. Despite their different origins, the improvement from averaging closely tracks the amount of nonshared predictive variation available along each estimator axis (Figure 1(c)). Additional diagnosis results on other datasets can be found in Appendix A. These results reveal both the promise and the challenge of estimator scaling: large gains arise from averaging multiple estimators but maintaining individual copies of the models would incur substantial deployment cost.

To address these challenges, we introduce RECursive Averaged Predictor (RECAP), a parameterefficient recursive CTR model that realizes estimator scaling at three levels. RECAP uses a weightshared recursive interaction backbone, allowing interaction computation to increase without proportional growth in depth-specific parameters. On top of this backbone, RECAP incorporates information across independent training runs through ensemble distillation, across the training trajectory through exponential moving weight averaging, and across inference-time routes through route aggregation. As summarized in Figure 1, these mechanisms expose progressively different sources of estimator variation while keeping the deployed model close to a single-model parameter budget, thereby placing RECAP on a favorable deployment frontier.

Our main contributions are as follows:

• Estimator scaling for CTR prediction. We identify estimator scaling as a complementary axis to interaction-capacity scaling and show that estimator scaling continues to provide gains after additional interaction depth and branch capacity exhibit diminishing returns.

• RECAP: parameter-efficient multi-level estimator scaling. We introduce a recursive CTR model that combines parameter sharing with estimator scaling across training trajectories, independent runs, and inference-time routes through exponential moving weight averaging, ensemble distillation, and route aggregation.

• Improved deployment efficiency. Across multiple CTR benchmarks, RECAP improves the performance–parameter trade-off over strong cross-network baselines and approaches the performance of substantially larger multi-model ensembles while retaining approximately a single-model deployment footprint.

## 2 RELATED WORK

Feature Interaction and Cross-Network Scaling: A central line of CTR research improves predictive accuracy by scaling the learned feature interactions. Early models combine factorization-based interactions with deep networks Guo et al. (2017), while DCN explicitly constructs bounded-degree feature crosses through stacked cross layers Wang et al. (2017). Subsequent work increases or adapts this interaction capacity in different ways: xDeepFM introduces explicit vector-wise high-order interactions Lian et al. (2018), AutoInt uses stacked self-attention to model increasingly high-order combinations Song et al. (2019), and DCNv2 improves the expressiveness of cross networks while retaining computational efficiency through low-rank parameterizations Wang et al. (2021a). Most recently, FCN and QNN use exponential cross networks to explicitly represent interactions spanning a broad range of orders Li et al. (2025; 2026). Collectively, these methods primarily improve CTR prediction by enlarging or selectively allocating the interaction capacity of a predictor. In contrast, we study what should scale once further increases along this axis yield diminishing predictive returns.

Ensembling, Averaging, and Distillation: Our work is also related to methods that combine multiple learned predictors. Deep ensembles improve prediction by aggregating independently trained models Lakshminarayanan et al. (2017), while trajectory-based methods such as stochastic weight averaging (SWA) aggregate solutions encountered during optimization Izmailov et al. (2019). Model Soups and DiWA further study weight averaging across independently trained or fine-tuned models, highlighting the importance of model diversity for successful aggregation Wortsman et al. (2022); Rame et al. (2022). The role of diversity in ensemble performance has also been studied theoretically Wood et al. (2023). Our contribution is not a new averaging operator in isolation. Instead, we formulate estimator scaling as a complementary scaling axis and study its gains, saturation, and cost across independent runs, optimization trajectories, and inference-time routes within a common variance-based framework

Recursive and Parameter-Efficient Computation: A complementary literature seeks to increase computation or interaction complexity without proportional parameter growth. Low-rank cross networks reduce the parameter cost of expressive interaction operators Wang et al. (2021a), while recent LoopCTR introduces loop scaling for CTR prediction, recursively reusing shared layers to decouple additional computation from model size Tang et al. (2026). RECAP shares the principle of recursive parameter reuse but serves a different purpose: recursion provides a compact substrate on which multiple related estimators can be realized. We combine this backbone with cross-run distillation, trajectory averaging, and inference-time route aggregation, using parameter sharing to enable parameter-efficient estimator scaling.

## 3 FROM INTERACTION CAPACITY TO ESTIMATOR SCALING

In this section, we formalize why estimator scaling can provide a complementary source of improvement to increasing the interaction capacity of a single predictor, characterize when such gains are available, and analyze how they combine across different estimator sources.

Consider a CTR dataset $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where $x _ { i }$ contains categorical and numerical features and $y _ { i } \in \{ 0 , 1 \}$ } denotes a click. A model with parameters θ produces a logit z and click probability $p _ { \theta } ( x ) = \sigma ( z )$ , and is trained using binary cross-entropy, which is defined as:

$$
\ell ( y , z ) = \log ( 1 + \exp ( z ) ) - y z .\tag{1}
$$

A fixed model family can yield multiple related predictors through independent training runs, checkpoints along an optimization trajectory, or alternative inference-time routes. We refer to each such predictor as an estimator, and denote these three estimator sources by $S , T$ , and R, respectively. Given M estimators with logits $\{ Z _ { m } ( x ) \} _ { m = 1 } ^ { M }$ , their mean-logit prediction is

$$
\bar { Z } _ { M } ( x ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } Z _ { m } ( x ) .\tag{2}
$$

Two Complementary Scaling Axes As motivated in the introduction, interaction-capacity scaling increases the expressive capacity of a single predictor, whereas estimator scaling increases the number of estimators incorporated into its prediction while largely holding the underlying model family fixed. We next formalize why these two axes can provide distinct sources of improvement.

Let c index the interaction capacity of a model family, let $Z _ { c } ( x )$ denote the logit of a randomly drawn estimator from that family, and let $\mu _ { c } ( x ) = \mathbb { E } [ \dot { Z } _ { c } ( x )$ | x] denote the estimator-family mean. For M estimators, define $\begin{array} { r } { \bar { Z } _ { c , M } ( x ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } Z _ { c , m } ( x ) } \end{array}$ , and let $\mathcal { L } ( c , M )$ denote its expected binary cross-entropy over the data distribution and estimator randomness. We similarly define $\mathcal { L } ( \mu _ { c } ) = \mathbb { E } _ { x , y } [ \ell ( y , \tilde { \mu } _ { c } ( x ) ) ]$ ]. A second-order expansion of Eq. 1 around $\mu _ { c } ( x )$ gives

$$
\mathcal { L } ( c , M ) \approx \mathcal { L } ( \mu _ { c } ) + \frac { 1 } { 2 } \mathbb { E } _ { x } \left[ \sigma ( \mu _ { c } ( x ) ) \left( 1 - \sigma ( \mu _ { c } ( x ) ) \right) \mathrm { V a r } ( \bar { Z } _ { c , M } ( x ) \mid x ) \right] .\tag{3}
$$

Equation 3 separates two mechanisms for improvement. Increasing interaction capacity can improve the mean predictor $\mu _ { c }$ and thereby reduce $\bar { \mathcal { L } ( \mu _ { c } ) }$ , whereas estimator scaling reduces variation around that mean through aggregation. Thus, even when additional interaction capacity yields little further improvement in the mean predictor, estimator scaling can still reduce expected loss whenever reducible estimator variation remains.

Non-Shared Variation Governs Estimator Scaling We next characterize how much estimator variation can be removed by aggregation. For a fixed input x and interaction capacity $c ,$ suppose the M estimators are exchangeable (i.e their joint probability distribution is invariant under any permutation of their indices) , with conditional logit variance $\tau _ { c } ^ { 2 } ( x )$ and common pairwise correlation $\rho _ { c } ( x )$ . The variance of their mean is

$$
\mathrm { V a r } ( \bar { Z } _ { c , M } ( x ) \mid x ) = \tau _ { c } ^ { 2 } ( x ) \left( \rho _ { c } ( x ) + \frac { 1 - \rho _ { c } ( x ) } { M } \right) .\tag{4}
$$

Relative to a single estimator, the variance removed by averaging is therefore

$$
V _ { \mathrm { n s } } ( c , M ; x ) = \tau _ { c } ^ { 2 } ( x ) ( 1 - \rho _ { c } ( x ) ) \left( 1 - \frac { 1 } { M } \right) ,\tag{5}
$$

which we call the non-shared, or reducible, predictive variation. Combining Eqs. 3 and 5, the reduction in expected BCE loss is locally

$$
\Delta \mathcal { L } ( c , M ) \approx \frac { 1 } { 2 } \mathbb { E } _ { x } \left[ \sigma ( \mu _ { c } ( x ) ) \left( 1 - \sigma ( \mu _ { c } ( x ) ) \right) V _ { \mathrm { n s } } ( c , M ; x ) \right] .\tag{6}
$$

Thus, BCE improvement is approximately proportional to the reducible predictive variance, up to the local curvature of the loss. When the estimator families have similar mean predictions, this curvature factor is approximately shared, so differences in averaging gain are primarily determined by $V _ { \mathrm { n s } }$ . Further derivations of the relationship between non-shared variance and estimator scaling are provided in Appendix B.

Combining Estimator Axes Different estimator sources may expose overlapping predictive variation, so their gains need not be additive. Let A and B denote two estimator axes and let $Z _ { A , B } ( x )$ denote the corresponding random logit for a fixed input. By the law of total variance,

$$
\operatorname { V a r } ( Z _ { A , B } \mid x ) = \underbrace { \operatorname { \mathbb { E } } _ { B } \left[ \operatorname { V a r } _ { A } ( Z _ { A , B } \mid B , x ) \right] } _ { V _ { A } ( x ) } + \underbrace { \operatorname { V a r } _ { B } \left[ \operatorname { \mathbb { E } } _ { A } ( Z _ { A , B } \mid B , x ) \right] } _ { V _ { B \mid A } ( x ) } .\tag{7}
$$

RECAP: Recursive Averaged Predictor  
![](images/035397c94ebecb06050945e99db1b52c183b5abd1d32913f124f3ec95da50981.jpg)  
Figure 2: RECAP architecture and multi-level estimator scaling. (a) RECAP builds on the linear and exponential cross-network (LCN and ECN) operators of FCN (Li et al., 2026), which we organize as two recursive interaction towers. Each tower shares its core cross-network parameters across recursive steps, while a route selector chooses lightweight LoRA adaptations from a towerspecific expert bank. The two tower representations are fused to produce the CTR logit. (b) RECAP incorporates estimators at three levels: cross-run information is transferred through ensemble distillation, within-run information is aggregated through exponential moving averaging, and multiple recursive routes are combined through mean-logit aggregation at inference.

Here $V _ { A } ( x )$ is the variation accessible to averaging along axis A, whereas $V _ { B | A } ( x )$ is the residual variation remaining after averaging over A. Up to the same local curvature weighting as in Eq. 6, the standalone reduction in expected loss from an estimator axis is therefore governed by its accessible variation, while the marginal improvement from adding another axis is governed by the residual variation not already removed. This prediction is evaluated empirically in Section 5.2; the full three-axis decomposition for independent runs, trajectories, and routes is given in Appendix C

## 4 RECAP: RECURSIVE AVERAGED PREDICTOR

We now introduce RECursive Averaged Predictor (RECAP), a parameter-efficient realization of the estimator-scaling principle in Section 3. The theory identifies non-shared and residual predictive variation as the quantities governing the benefit of additional estimator axes; RECAP focuses on realizing three such axes: independent runs, optimization trajectories, and inference-time routes, without deploying a separate full model for each. These axes connect to the output-space analysis in different ways: ensemble distillation compresses a cross-run mean-logit predictor into a single model, EMA provides a parameter-space approximation to trajectory prediction averaging, and route aggregation performs the mean-logit averaging analyzed directly in Section 3.

## 4.1 PARAMETER-EFFICIENT RECURSIVE INTERACTION BACKBONE

The estimator-scaling analysis does not prescribe a particular CTR backbone; it specifies that useful estimator axes should expose non-shared predictive variation at acceptable cost. We therefore use recursion for two complementary purposes: to increase interaction computation through parameter sharing, and to expose an additional inference-time estimator axis through alternative computation routes. We instantiate RECAP using the Fusing Cross Network (FCN) architecture (Li et al., 2026), a strong recent explicit-interaction model for CTR prediction. FCN combines a linear cross network (LCN), whose interaction order grows progressively with depth, with an exponential cross network (ECN), which constructs higher-order interactions more rapidly. Given input features x, RECAP first maps sparse categorical fields and numerical features to a shared representation $h _ { 0 }$ , which is then processed by recursive LCN and ECN towers. The two towers use independent parameters, while the core interaction weights within each tower are shared across recursive steps. Lightweight route-specific adaptations allow different adapter sequences to define multiple related predictors over the same shared backbone, providing a third source of estimator variation that can be aggregated at inference without maintaining separate full models. Thus, recursion serves both as a parameterefficient computational substrate and as the mechanism that enables RECAP’s route-scaling axis.

Additionally, the broader estimator-scaling principle is not tied to these particular interaction operators;   
Section 5.3 further examines estimator scaling independently of the recursive architecture.

Let $q \in \{ \mathrm { L } , \mathrm { E } \}$ denote a tower and let $h _ { t } ^ { q }$ be its hidden representation after t recursive steps. RECAP updates

$$
h _ { t + 1 } ^ { q } = F _ { q } \left( h _ { t } ^ { q } , x ; W _ { q } + \Delta W _ { q , r _ { t } } \right) ,\tag{8}
$$

where $W _ { q }$ is the shared interaction operator and $\Delta W _ { q , r _ { t } }$ is a lightweight route-dependent adaptation selected at step t.

Rather than assigning a full independent parameter matrix to every recursive step, RECAP parameterizes each adaptation using a low-rank residual,

$$
\Delta W _ { q , r } = A _ { q , r } B _ { q , r } ^ { \top } , \qquad \mathrm { r a n k } ( \Delta W _ { q , r } ) \leq r _ { \mathrm { L o R A } } .\tag{9}
$$

Each tower maintains a small bank of such residuals, and a route selector determines which adapter is applied at each recursive step. A route is therefore defined by the sequence $\tau _ { q } = ( r _ { 1 } , r _ { 2 } , \ldots , r _ { T } )$ while all routes continue to reuse the same core operator $W _ { q }$

After T recursive steps, the two tower representations are fused, $h _ { \mathrm { f u s e } } = \mathrm { F u s e } \left( h _ { T } ^ { \mathrm { L } } , h _ { T } ^ { \mathrm { E } } \right)$ , and a prediction head produces the CTR logit $z _ { \theta } ( x ) = g ( h _ { \mathrm { f u s e } } )$ This recursive parameterization decouples interaction computation from parameter depth: increasing the number of recursive steps increases computation while reusing the same backbone parameters. The lightweight route residuals additionally allow RECAP to expose multiple related predictors without storing multiple models.

## 4.2 ESTIMATOR SCALING AT THREE LEVELS

RECAP applies the estimator-scaling principle of Section 3 at three complementary levels.

Training-trajectory averaging: During optimization, successive checkpoints provide different estimators from the same training run. RECAP maintains an exponential moving average (EMA) of the model parameters,

$$
\bar { \theta } _ { t } = \beta \bar { \theta } _ { t - 1 } + ( 1 - \beta ) \theta _ { t } ,\tag{10}
$$

where $\beta$ controls the averaging horizon. The EMA parameters are used for validation, checkpoint selection, and final inference. This incorporates information across the optimization trajectory without adding deployed parameters or inference-time computation.

EMA averages parameters rather than logits directly. Its connection to the output-space analysis of Section 3 follows under a local linearization of the predictor. If $\bar { \theta } = \sum _ { t } \alpha _ { t } \theta _ { t }$ denotes the EMA parameters, then for checkpoints lying in a locally approximately linear region,

$$
z _ { \bar { \theta } } ( x ) \approx \sum _ { t } \alpha _ { t } z _ { \theta _ { t } } ( x ) .\tag{11}
$$

Thus, EMA can be viewed as a parameter-efficient approximation to trajectory-level prediction averaging. This approximation need not hold globally, so we empirically evaluate trajectory scaling rather than treating it as an exact consequence of the logit-averaging analysis.

Cross-run averaging through distillation: Independent training runs expose substantially more non-shared predictive variation, but deploying an ensemble of M complete CTR models multiplies storage and inference cost. RECAP instead transfers this cross-run aggregation into a single model through ensemble distillation.

Let $z ^ { ( 1 ) } ( x ) , \dots , z ^ { ( M ) } ( x )$ denote the logits of independently trained teacher models. We form the teacher aggregate

$$
\bar { z } _ { \mathrm { t e a c h } } ( x ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } z ^ { ( m ) } ( x ) ,\tag{12}
$$

and train RECAP to match the corresponding soft prediction in addition to the ground-truth label. The training objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r a i n } } = \mathcal { L } _ { \mathrm { s u p } } + \lambda _ { \mathrm { K D } } \mathcal { L } _ { \mathrm { K D } } , } \end{array}\tag{13}
$$

where

$$
\mathcal { L } _ { \mathrm { K D } } = \mathrm { B C E } \left( \sigma ( \bar { z } _ { \mathrm { t e a c h } } ) , z _ { \theta } ( x ) \right) .\tag{14}
$$

Table 1: Performance comparison of different deep CTR models. Typically, CTR researchers consider an improvement of 0.001 (0.1%) in Logloss and AUC to be practically meaningful Zhu et al. (2021); Li et al. (2026). We conduct a two-tailed t-test over five independent runs to assess the statistical significance between our models and the best baseline $( ^ { * } \colon \mathsf { p } { < } 0 . 0 1 )$ . Best results are highlighted in bold, second-best are underlined.
<table><tr><td rowspan="2">Models</td><td colspan="2">Avazu</td><td colspan="2">Criteo</td><td colspan="2">ML-1M</td><td colspan="2">KDD12</td><td colspan="2">KKBox</td></tr><tr><td>Logloss↓</td><td>AUC(%)↑</td><td>Logloss↓</td><td>AUC(%)↑</td><td>Logloss↓</td><td>AUC(%)↑</td><td>Logloss↓</td><td>AUC(%)↑</td><td>Logloss↓</td><td>AUC(%)↑</td></tr><tr><td>DNN Covington et al. (2016)</td><td>0.3721</td><td>79.27</td><td>0.4380</td><td>81.40</td><td>0.3100</td><td>90.30</td><td>0.1502</td><td>80.52</td><td>0.4811</td><td>85.01</td></tr><tr><td>PNN Qu et al. (2016)</td><td>0.3712</td><td>79.44</td><td>0.4378</td><td>81.42</td><td>0.3070</td><td>90.42</td><td>0.1504</td><td>80.47</td><td>0.4793</td><td>85.15</td></tr><tr><td>Wide &amp; Deep Cheng et al. (2016)</td><td>0.3720</td><td>79.29</td><td>0.4376</td><td>81.42</td><td>0.3056</td><td>90.45</td><td>0.1504</td><td>80.48</td><td>0.4852</td><td>85.04</td></tr><tr><td>DeepFM Guo et al. (2017)</td><td>0.3719</td><td>79.30</td><td>0.4375</td><td>81.43</td><td>0.3073</td><td>90.51</td><td>0.1501</td><td>80.60</td><td>0.4785</td><td>85.31</td></tr><tr><td>DCNv1 Wang et al. (2017)</td><td>0.3719</td><td>79.31</td><td>0.4376</td><td>81.44</td><td>0.3156</td><td>90.38</td><td>0.1501</td><td>80.59</td><td>0.4766</td><td>85.31</td></tr><tr><td>xDeepFM Lian et al. (2018)</td><td>0.3718</td><td>79.33</td><td>0.4376</td><td>81.43</td><td>0.3054</td><td>90.47</td><td>0.1501</td><td>80.62</td><td>0.4772</td><td>85.35</td></tr><tr><td>AutoInt Song et al. (2019)</td><td>0.3746</td><td>79.02</td><td>0.4390</td><td>81.32</td><td>0.3112</td><td>90.45</td><td>0.1502</td><td>80.57</td><td>0.4773</td><td>85.34</td></tr><tr><td>AFN Cheng et al. (2020)</td><td>0.3726</td><td>79.29</td><td>0.4384</td><td>81.38</td><td>0.3048</td><td>90.53</td><td>0.1499</td><td>80.70</td><td>0.4842</td><td>84.89</td></tr><tr><td>DCNv2 Wang et al. (2021a)</td><td>0.3718</td><td>79.31</td><td>0.4376</td><td>81.45</td><td>0.3098</td><td>90.56</td><td>0.1502</td><td>80.59</td><td>0.4787</td><td>85.31</td></tr><tr><td>EDCN Chen et al. (2021)</td><td>0.3716</td><td>79.35</td><td>0.4386</td><td>81.36</td><td>0.3073</td><td>90.48</td><td>0.1501</td><td>80.62</td><td>0.4952</td><td>85.27</td></tr><tr><td>MaskNet Wang et al. (2021b)</td><td>0.3711</td><td>79.43</td><td>0.4387</td><td>81.34</td><td>0.3080</td><td>90.34</td><td>0.1498</td><td>80.79</td><td>0.5003</td><td>84.79</td></tr><tr><td>EulerNet Tian et al. (2023)</td><td>0.3723</td><td>79.22</td><td>0.4379</td><td>81.47</td><td>0.3050</td><td>90.44</td><td>0.1498</td><td>80.78</td><td>0.4922</td><td>84.27</td></tr><tr><td>FinalMLP Mao et al. (2023)</td><td>0.3718</td><td>79.35</td><td>0.4373</td><td>81.45</td><td>0.3058</td><td>90.52</td><td>0.1497</td><td>80.78</td><td>0.4822</td><td>85.10</td></tr><tr><td>FINAL Zhu et al. (2023)</td><td>0.3712</td><td>79.41</td><td>0.4371</td><td>81.49</td><td>0.3035</td><td>90.53</td><td>0.1498</td><td>80.74</td><td>0.4800</td><td>85.14</td></tr><tr><td>RFM Tian et al. (2024)</td><td>0.3723</td><td>79.24</td><td>0.4374</td><td>81.47</td><td>0.3048</td><td>90.51</td><td>0.1506</td><td>80.73</td><td>0.4853</td><td>84.70</td></tr><tr><td>QNN Li et al. (2025)</td><td>0.3712</td><td>79.47</td><td>0.4358</td><td>81.63</td><td>0.2960</td><td>90.87</td><td>0.1506</td><td>80.82</td><td>0.4730</td><td>85.76</td></tr><tr><td>FCN Li et al. (2026)</td><td>0.3702</td><td>79.66</td><td>0.4359</td><td>81.62</td><td>0.3001</td><td>90.74</td><td>0.1494</td><td>80.86</td><td>0.4746</td><td>85.72</td></tr><tr><td>RECAP (ours)</td><td>0.3702*</td><td>79.92*</td><td>0.4345*</td><td>81.76*</td><td>0.2951*</td><td>90.88*</td><td>0.1494*</td><td>81.41*</td><td>0.4669*</td><td>86.06*</td></tr></table>

Thus, distillation transfers the benefit of expensive cross-run averaging into a single deployed parameterization.

This objective has a direct connection to the mean-logit predictor analyzed in Section 3. For a fixed input, BCE with soft target $q = \sigma ( \bar { z } _ { \mathrm { t e a c h } } )$ is minimized when $\sigma ( z _ { \theta } ) = q .$ , or equivalently $z _ { \theta } = \bar { z } _ { \mathrm { t e a c h } }$ Thus, with sufficient student capacity and optimization, distillation compresses the cross-run meanlogit aggregate into a single deployed predictor. In practice this equality is approximate, and the remaining compression error is measured empirically.

Inference-time route aggregation: Finally, the recursive architecture provides a third estimator source at inference. Different LoRA selections define different recursive routes $\tau _ { 1 } , \ldots , \tau _ { K }$ , which produce logits $z _ { \tau _ { 1 } } ( x ) , \dots , z _ { \tau _ { K } } ( x )$

RECAP combines them using mean-logit aggregation,

$$
z _ { \mathrm { R E C A P } } ^ { ( K ) } ( x ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } z _ { \tau _ { k } } ( x ) .\tag{15}
$$

Increasing K therefore increases inference compute but leaves the stored model unchanged. Thi provides a test-time scaling knob that is complementary to both EMA and cross-run distillation.

## 4.3 TRAINING ACROSS RECURSIVE BUDGETS

Because the recursive backbone may be evaluated at different inference depths, we train RECAP across multiple rollout budgets rather than specializing it to a single depth. Let $\mathcal { D } _ { \mathrm { t r a i n } }$ denote the set of training depths. The supervised objective is

$$
\mathcal { L } _ { \mathrm { s u p } } = \sum _ { d \in \mathcal { D } _ { \mathrm { t r a i n } } } \alpha _ { d } \ell \left( y , z _ { \theta } ^ { ( d ) } ( x ) \right) ,\tag{16}
$$

where $\alpha _ { d }$ controls the contribution of rollout depth d. During training, routes are sampled from the LoRA banks so that the shared backbone is exposed to multiple recursive computation paths.

We optimize RECAP with a decaying learning-rate schedule and maintain the EMA parameters from Eq. 10 throughout training. Teacher predictions are precomputed, so ensemble distillation does not require executing the teacher models during student optimization.

## 5 EXPERIMENTS AND RESULTS

We evaluate RECAP with three questions in mind: (1) does estimator scaling improve predictive performance across standard CTR benchmarks;

(2) are these gains explained by the non-shared predictive variation characterized in Section $_ { 3 ; }$ and (3) can RECAP realize these gains without the deployment cost of a conventional ensemble?

<table><tr><td>Model</td><td>EMA</td><td>KD</td><td>Routes</td><td>AUC</td></tr><tr><td>Recursive backbone</td><td></td><td>一</td><td>一</td><td>81.59</td></tr><tr><td>+ trajectory scaling</td><td>√</td><td></td><td></td><td>81.65</td></tr><tr><td>+ cross-run scaling</td><td></td><td>√</td><td></td><td>81.68</td></tr><tr><td>+ route scaling</td><td></td><td></td><td>√</td><td>81.61</td></tr><tr><td>RECAP</td><td>√</td><td>√</td><td>√</td><td>81.76</td></tr></table>

KD denotes ensemble distillation; Routes denotes inference-time route aggregation.

![](images/7f92b0c1d05261c95663868e68669613024ea5c828c2a4ba06c1c0016de1f8c3.jpg)  
Figure 3: Marginal estimator-scaling gains track residual predictive variation. Across trajectory, seed, and inference-time route estimators, the marginal AUC improvement from adding an estimator axis increases with the residual nonshared variance available along that axis.

Table 2: Ablation of estimator-scaling components on the recursive backbone. Trajectory scaling through EMA, cross-run scaling through ensemble distillation, and route scaling each improve performance individually, while combining all three yields the highest AUC.

Datasets and metrics: We evaluate on five public CTR benchmarks: Avazu Zhu et al. (2021), Criteo Zhu et al. (2021), MovieLens-1M (ML-1M) Song et al. (2019), KDD12 Song et al. (2019), and KKBox Zhu et al. (2022). We follow the preprocessing and evaluation protocol defined in the FuxiCTR framework which is used by prior open CTR benchmarks including FCN Zhu et al. (2021); Li et al. (2026); Zhu et al. (2022) . We report Area Under the ROC Curve (AUC), where higher is better, and binary cross-entropy (Logloss), where lower is better. Because improvements in mature CTR benchmarks are often small in absolute magnitude, we report AUC in percentage points throughout the paper.

Baselines: We compare against representative factorization, deep interaction, and explicit crossnetwork models, including DNN Covington et al. (2016), PNN Qu et al. (2016), Wide & Deep Cheng et al. (2016), DeepFM Guo et al. (2017), DCN Wang et al. (2017), xDeepFM Lian et al. (2018), AutoInt Song et al. (2019), AFN Cheng et al. (2020), DCNv2 Wang et al. (2021a), EDCN Chen et al. (2021), MaskNet Wang et al. (2021b), EulerNet Tian et al. (2023), FinalMLP Mao et al. (2023), FINAL Zhu et al. (2023), RFM Tian et al. (2024), QNN Li et al. (2025), FCN Li et al. (2026). FCN is our primary reference architecture because it provides a strong recent explicit-interaction baseline and directly scales interaction capacity through complementary linear and exponential cross networks.

## 5.1 RECAP IMPROVES CTR PREDICTION ACROSS BENCHMARKS

Table 1 compares RECAP with representative CTR prediction models across five public benchmarks. RECAP achieves the highest AUC among the compared methods on all five datasets. Relative to the strongest compared baseline on each dataset, RECAP improves AUC by 0.26, 0.13, 0.01, 0.55, and 0.30 points on Avazu, Criteo, ML-1M, KDD12, and KKBox, respectively. RECAP also improves or matches Logloss on all of the datasets as compared to the next best baseline. These results indicate that the benefit of estimator scaling is not restricted to the Criteo diagnostic setting used in our analysis, but transfers across datasets with substantially different sparsity and scale.

## 5.2 NON-SHARED PREDICTIVE VARIATION EXPLAINS ESTIMATOR-SCALING GAINS

Section 3 makes two related predictions about estimator scaling. First, the standalone benefit of scaling a particular estimator source should depend on the non-shared predictive variation accessible along that axis. Second, when multiple estimator sources are combined, the marginal benefit of adding a new axis should depend on the residual variation that remains after the existing axes have already been incorporated. We test both predictions using the estimator families introduced in Figure 1(b).

Standalone estimator scaling: Figure 1(c) compares the gain from averaging estimators within each axis against the corresponding reducible predictive variation. Despite their different origins, the three estimator families follow a common trend: larger non-shared variation is associated with larger averaging gains. As shown in Figure 1(b), independent training runs occupy the high-variation, high gain regime, checkpoints from a single optimization trajectory expose less reducible variation and provide smaller gains, and alternative inference-time routes through a shared model expose the least variation and provide the smallest gains. Equation 6 formally predicts expected BCE improvement.

Empirically, AUC exhibits the same ordering: estimator families with greater reducible variation provide larger averaging gains. Results on additional datasets can be found in Appendix A.

Marginal gains from combining estimator axes: We next test the marginal improvement prediction from Section 3. Figure 3 plots the marginal AUC improvement obtained by adding an estimator axis to different existing configurations against the residual non-shared variance available to that axis. The same estimator source can provide different gains depending on what has already been averaged: when previous mechanisms remove overlapping variation, less residual variation remains and the incremental benefit decreases. Across seed, trajectory, and route estimators, configurations with greater residual variation consistently yield larger marginal improvements. This relationship explain why the gains from different estimator-scaling mechanisms are complementary but not simply additive, and is consistent with the decomposition in Eq. 7. Table 2 further confirms complementarity within RECAP: each estimator source improves the same recursive backbone individually, while combining trajectory, cross-run, and route scaling yields the strongest performance.

## 5.3 ESTIMATOR SCALING IS NOT SPECIFIC TO THE RECURSIVE BACKBONE

To separate estimator-scaling gains from the recursive architecture, we apply the same training schedule, EMA, and ensemble-distillation objective to the FCN backbone. As shown in Table 3, both trajectory scaling through EMA and cross-run scaling through distillation improve FCN, confirming that estimator scaling is a general principle rather than a RECAPspecific effect. The key contribution of the recursive architecture is to introduce an additional routescaling axis: weight sharing keeps interaction computation parameter-efficient, while lightweight routespecific adaptations define multiple related predictors within a single model. RECAP therefore scales esti-

<table><tr><td>Model</td><td>EMA KD</td><td>AUC</td></tr><tr><td>FCN</td><td>一 一</td><td>81.62</td></tr><tr><td>+ trajectory scaling</td><td>√ 一</td><td>81.67</td></tr><tr><td>+ cross-run scaling</td><td>√ 1</td><td>81.69</td></tr><tr><td>+ cross-run + trajectory scaling</td><td>√ √</td><td>81.74</td></tr></table>

Table 3: Estimator scaling improves the FCN backbone. Trajectory scaling through EMA and cross-run scaling through ensemble distillation each improve AUC individually, while combining both yields the strongest performance.

mators across training trajectories, independent runs, and alternative computation routes, with the latter providing a low-cost source of estimator diversity unavailable to the FCN backbone.

## 5.4 RECAP SHIFTS THE PERFORMANCE–PARAMETER FRONTIER

The performance–parameter analyses in Figures 1(d) and 4 provide complementary views of RECAP’s deployment efficiency. Against a broad set of CTR architectures, RECAP substantially outperforms singlemodel baselines at a comparable parameter budget and approaches the five-model FCN ensemble (81.764 versus 81.771 AUC) while using only a fraction of its deployed parameters. The controlled scaling trajectories in Figure 4 further show that increasing interaction capacity through additional towers quickly yields diminishing returns, whereas independent-model ensembling continues to improve performance but at roughly linear deployment cost. RECAP lies above the interaction-capacity scaling trajectory and reaches the performance regime of substantially larger seed ensembles at approximately a single-model parameter budget. Together, these results suggest that once interaction-capacity scaling becomes inefficient, allocating resources toward estimator scaling provides a more favorable accuracy– parameter trade-off. RECAP captures much of this estimator-scaling benefit while avoiding the deployment cost of maintaining multiple full models.

![](images/ae38087262956dce6d13f93c047bdd9e3410207dd344781f329a589f7134e907.jpg)  
Figure 4: RECAP improves the deployment performance–parameter frontier. Increasing interaction capacity through additional towers yields diminishing returns, while seed ensembling continues to improve AUC at substantially higher deployment cost. RE-CAP reaches near five-model ensemble performance while retaining approximately a single-model parameter footprint.

## 6 CONCLUSION

In this work, we tackle the problem of CTR prediction and show that scaling interaction capacity in CTR models can exhibit diminishing predictive returns even when estimator diversity remains exploitable. This motivates estimator scaling as a complementary axis, with gains governed by the non-shared predictive variation available to aggregation. Building on this view, RECAP combines cross-run distillation, trajectory averaging, and route-based estimator scaling within a parameterefficient recursive backbone. Across five CTR benchmarks, RECAP achieves state-of-the-art performance, while approaching five-model ensemble performance at approximately a single-model parameter budget.

## REFERENCES

Bo Chen, Yichao Wang, Zhirong Liu, Ruiming Tang, Wei Guo, Hongkun Zheng, Weiwei Yao, Muyu Zhang, and Xiuqiang He. Enhancing explicit and implicit feature interactions via information sharing for parallel deep ctr models. In Proceedings ofthe 30th ACM International Conference on Information & Knowledge Management, CIKM ’21, pp. 3757–3766, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450384469. doi: 10.1145/3459637.3481915. URL https://doi.org/10.1145/3459637.3481915.

Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems, DLRS 2016, pp. 7–10, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450347952. doi: 10.1145/2988450.2988454. URL https://doi.org/10.1145/2988450.2988454.

Weiyu Cheng, Yanyan Shen, and Linpeng Huang. Adaptive factorization network: Learning adaptiveorder feature interactions, 2020. URL https://arxiv.org/abs/1909.03276.

Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proceedings ofthe 10th ACM Conference on Recommender Systems, RecSys ’16, pp. 191–198, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450340359. doi: 10.1145/2959100.2959190. URL https://doi.org/10.1145/2959100.2959190.

Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a factorizationmachine based neural network for ctr prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, IJCAI’17, pp. 1725–1731. AAAI Press, 2017. ISBN 9780999241103.

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization, 2019. URL https://arxiv. org/abs/1803.05407.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems, NIPS’17, pp. 6405–6416, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

Honghao Li, Yiwen Zhang, Yi Zhang, Lei Sang, and Jieming Zhu. Revisiting feature interactions from the perspective of quadratic neural networks for click-through rate prediction. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, pp. 1365–1375, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400714542. doi: 10.1145/3711896.3737106. URL https: //doi.org/10.1145/3711896.3737106.

Honghao Li, Yiwen Zhang, Yi Zhang, Hanwei Li, Lei Sang, and Jieming Zhu. Fcn: Fusing exponential and linear cross network for click-through rate prediction. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, KDD ’26, pp. 681–691, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400722585. doi: 10.1145/3770854.3780177. URL https://doi.org/10.1145/3770854.3780177.

Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’18, pp. 1754–1763, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450355520. doi: 10.1145/3219819.3220023. URL https://doi.org/ 10.1145/3219819.3220023.

Kelong Mao, Jieming Zhu, Liangcai Su, Guohao Cai, Yuru Li, and Zhenhua Dong. Finalmlp: an enhanced two-stream mlp model for ctr prediction. In Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23. AAAI Press, 2023. ISBN 978-1-57735-880-0. doi: 10.1609/aaai. v37i4.25577. URL https://doi.org/10.1609/aaai.v37i4.25577.

Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. Product-based neural networks for user response prediction. In 2016 IEEE 16th International Conference on Data Mining (ICDM), pp. 1149–1154, 2016. doi: 10.1109/ICDM.2016.0151.

Alexandre Rame, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord. Diverse weight averaging for out-of-distribution generalization. In NeurIPS, 2022.

Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. Autoint: Automatic feature interaction learning via self-attentive neural networks. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM ’19, pp. 1161–1170, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450369763. doi: 10.1145/3357384.3357925. URL https: //doi.org/10.1145/3357384.3357925.

Jiakai Tang, Runfeng Zhang, Weiqiu Wang, Yifei Liu, Chuan Wang, Xu Chen, Yeqiu Yang, Jian Wu, Yuning Jiang, and Bo Zheng. Loopctr: Unlocking the loop scaling power for click-through rate prediction, 2026. URL https://arxiv.org/abs/2604.19550.

Zhen Tian, Ting Bai, Wayne Xin Zhao, Ji-Rong Wen, and Zhao Cao. Eulernet: Adaptive feature interaction learning via euler’s formula for ctr prediction. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SI-GIR ’23, pp. 1376–1385, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394086. doi: 10.1145/3539618.3591681. URL https://doi.org/10.1145/ 3539618.3591681.

Zhen Tian, Yuhong Shi, Xiangkun Wu, Wayne Xin Zhao, and Ji-Rong Wen. Rotative factorization machines. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, pp. 2912–2923, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704901. doi: 10.1145/3637528.3671740. URL https://doi.org/ 10.1145/3637528.3671740.

Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. Deep & cross network for ad click predictions. In Proceedings of the ADKDD’17, ADKDD’17, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450351942. doi: 10.1145/3124749.3124754. URL https: //doi.org/10.1145/3124749.3124754.

Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings ofthe Web Conference 2021, WWW ’21, pp. 1785–1797, New York, NY, USA, 2021a. Association for Computing Machinery. ISBN 9781450383127. doi: 10.1145/3442381.3450078. URL https://doi.org/10.1145/3442381.3450078.

Zhiqiang Wang, Qingyun She, and Junlin Zhang. Masknet: Introducing feature-wise multiplication to ctr ranking models by instance-guided mask, 2021b. URL https://arxiv.org/abs/ 2102.07619.

Danny Wood, Tingting Mu, Andrew M. Webb, Henry W. J. Reeve, Mikel Luján, and Gavin Brown. A unified theory of diversity in ensemble learning. Journal of Machine Learning Research, 24 (359):1–49, 2023. URL http://jmlr.org/papers/v24/23-0041.html.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 23965–23998. PMLR, 17– 23 Jul 2022. URL https://proceedings.mlr.press/v162/wortsman22a.html.

Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, and Xiuqiang He. Open benchmarking for click-through rate prediction. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, CIKM ’21, pp. 2759–2769, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450384469. doi: 10.1145/3459637.3482486. URL https://doi.org/10.1145/3459637.3482486.

Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, and Rui Zhang. Bars: Towards open benchmarking for recommender systems. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, pp. 2912–2923, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450387323. doi: 10.1145/3477495.3531723. URL https://doi.org/10.1145/ 3477495.3531723.

Jieming Zhu, Qinglin Jia, Guohao Cai, Quanyu Dai, Jingjie Li, Zhenhua Dong, Ruiming Tang, and Rui Zhang. Final: Factorized interaction layer for ctr prediction. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, pp. 2006–2010, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394086. doi: 10.1145/3539618.3591988. URL https://doi.org/10.1145/ 3539618.3591988.

## A DIAGNOSIS EXPERIMENTS ON ADDITIONAL DATASETS

To test whether the diagnostic findings from Criteo generalize beyond a single benchmark, we repeat the same analysis on additional CTR datasets. Figures 5, 6, 7 and 8 report the corresponding results on KKBox, ML-1M, KDD12, and Avazu. As in the main text, these diagnostics compare interaction-capacity scaling against estimator scaling and relate averaging gains to the amount of non-shared predictive variation available across estimator sources.

KKBox

![](images/3b2645622160e689fa40d3c217426e28494af6388191f854add0bcf5895073b4.jpg)  
(b) Estimator scaling

![](images/8c783b8e9fa0735533b5a1eb76758501b549f656dadf279d1d697eb53a85c0c4.jpg)  
(c) Averaging gain tracks non-shared variance

![](images/f3f090a52939b51dfc48ad3af93fbb7ec2c9c6c790093ff802d396a4c1e0440f.jpg)  
Figure 5: Diagnostic analysis of interaction-capacity and estimator scaling on KKBox. The same qualitative picture holds on KKBox: capacity scaling saturates, estimator scaling continues to help, and the gains from averaging track the amount of non-shared predictive variation exposed by each estimator source.  
ML-1M

![](images/2573abfc2334b908723b2038d6ffe6de038602df1814c78056e750b391921824.jpg)

![](images/116baaee1084299feef2f440f7ea68f34f0dd81811897c99207f5959ffddf2d1.jpg)  
(c) Averaging gain tracks non-shared variance

![](images/7b449f6cb18df36eb5eb1dd2704237e97f97012c7215aadb8c6b30867522478a.jpg)  
Figure 6: Diagnostic analysis of interaction-capacity and estimator scaling on ML-1M. ML-1M also exhibits diminishing returns from additional interaction capacity together with continued gains from estimator scaling. The observed averaging improvements remain strongly aligned with the amount of non-shared predictive variation available across estimator families.

(b) Estimator scaling

KDD12

![](images/09b25c5173c1002db0e1830066fdc5d436de9a07dd0329952f156e605f5bf6a6.jpg)

![](images/e249a3a71bc285df61e69b82ecf2500455615c5ed1eba11a8808a798b88f5a2f.jpg)

![](images/69b2d5f0313c2c6f9d7db8bfae21714072202e178aa80d44eab5af208c883bed.jpg)  
Figure 7: Diagnostic analysis of interaction-capacity and estimator scaling on KDD12. Increasing interaction capacity exhibits diminishing returns, whereas estimator scaling continues to provide gains. As in the main text, the improvement from averaging is closely associated with the non-shared predictive variation available to aggregation.  
Avazu

![](images/c0cee33a71719785ece6288c0fb1aec2efbe12881daa54267cb5fdefa80c2e76.jpg)

![](images/f478312872cf5629694d1d3575216c78602a6877fc1720c94424a10f0c45c669.jpg)  
(c) Averaging gain tracks non-shared variance

![](images/0dbcb4ffc5521f42d371533bc3d22bf5b28c37618264a291cb5cf47d15856b78.jpg)  
Figure 8: Diagnostic analysis of interaction-capacity and estimator scaling on Avazu. Increasing interaction capacity exhibits diminishing returns, whereas estimator scaling continues to provide gains. As in the main text, the improvement from averaging is closely associated with the non-shared predictive variation available to aggregation.

Across all four datasets, we observe the same qualitative pattern as on Criteo. First, increasing interaction capacity through greater depth or breadth yields diminishing returns once the underlying interaction model becomes sufficiently expressive. Second, averaging additional estimators continues to improve predictive performance after this capacity-scaling regime begins to saturate. Third, the magnitude of these gains is consistently associated with the amount of non-shared predictive variation exposed by each estimator source. These results support the main claim of the paper: interactioncapacity scaling and estimator scaling are complementary, and the usefulness of estimator scaling is governed by the reducible variation available to aggregation.

At the same time, the relative strength of different estimator sources varies across datasets. On some datasets, independent training runs provide the largest gains, while on others trajectory-based averaging is comparatively stronger. This variation is consistent with the theory in Section 3: the benefit of an estimator axis is determined not by its label (seed, trajectory, or route) but by how much non-shared predictive variation it contributes. This further motivates RECAP’s use of multiple estimator axes rather than relying on a single source of diversity.

Taken together, these additional results show that the diagnosis in Figure 1 is not specific to Criteo. Across diverse CTR benchmarks, increasing the capacity of a single predictor eventually becomes inefficient, while estimator scaling remains beneficial whenever additional non-shared predictive variation is available. This consistency across datasets provides further support for the estimatorscaling perspective developed in the main paper.

## B ADDITIONAL THEORY FOR ESTIMATOR SCALING

This appendix provides the derivations underlying Section 3. We first extend the equicorrelated analysis in the main text to arbitrary estimator covariance, and then derive the local relationship between reducible predictive variation and expected binary cross-entropy. Throughout this appendix, estimator variation is defined conditionally on the input x and for a fixed interaction capacity c.

## B.1 GENERAL COVARIANCE FORMULATION

For a fixed input x and interaction capacity c, let

$$
\mathbf { Z } _ { c } ( x ) = [ Z _ { c , 1 } ( x ) , \ldots , Z _ { c , M } ( x ) ] ^ { \top }\tag{17}
$$

denote the logits of M estimators, with conditional covariance matrix

$$
\Sigma _ { c } ( x ) = \operatorname { C o v } ( \mathbf { Z } _ { c } ( x ) \mid x ) .\tag{18}
$$

Their mean-logit aggregate is

$$
\bar { Z } _ { c , M } ( x ) = \frac { 1 } { M } \mathbf { 1 } ^ { \top } \mathbf { Z } _ { c } ( x ) ,\tag{19}
$$

and therefore

$$
\operatorname { V a r } \left( \bar { Z } _ { c , M } ( x ) \mid x \right) = \frac { 1 } { M ^ { 2 } } \mathbf { 1 } ^ { \top } \Sigma _ { c } ( x ) \mathbf { 1 } .\tag{20}
$$

The average conditional variance of an individual estimator is

$$
V _ { \mathrm { i n d } } ( c ; x ) = \frac { 1 } { M } \mathrm { t r } ( \Sigma _ { c } ( x ) ) .\tag{21}
$$

We define the variation removed by averaging as

$$
V _ { \mathrm { n s } } ( c , M ; x ) = \frac { 1 } { M } \mathrm { t r } ( \Sigma _ { c } ( x ) ) - \frac { 1 } { M ^ { 2 } } \mathbf { 1 } ^ { \top } \Sigma _ { c } ( x ) \mathbf { 1 } .\tag{22}
$$

Equation 22 shows that averaging gains depend jointly on estimator variance and covariance. A collection of estimators can exhibit substantial individual variation while providing little reducible variation if that variation is strongly shared.

For the equicorrelated setting used in the main text, suppose

$$
\mathrm { V a r } \left( Z _ { c , m } ( x ) \mid x \right) = \tau _ { c } ^ { 2 } ( x ) , \qquad \mathrm { C o r r } \left( Z _ { c , i } ( x ) , Z _ { c , j } ( x ) \mid x \right) = \rho _ { c } ( x ) , \quad i \neq j .\tag{23}
$$

Positive semidefiniteness requires

$$
\rho _ { c } ( x ) \geq - \frac { 1 } { M - 1 } .\tag{24}
$$

Substituting into Eq. 20 gives

$$
\mathrm { V a r } \left( \bar { Z } _ { c , M } ( x ) \mid x \right) = \tau _ { c } ^ { 2 } ( x ) \left( \rho _ { c } ( x ) + \frac { 1 - \rho _ { c } ( x ) } { M } \right) .\tag{25}
$$

Since a single estimator has variance $\tau _ { c } ^ { 2 } ( x )$ , the variation removed by averaging is

$$
V _ { \mathrm { n s } } ( c , M ; x ) = \tau _ { c } ^ { 2 } ( x ) ( 1 - \rho _ { c } ( x ) ) \left( 1 - \frac { 1 } { M } \right) ,\tag{26}
$$

recovering Eq. 5.

## B.2 FROM REDUCIBLE VARIATION TO BINARY CROSS-ENTROPY

We next derive Eqs. 3 and 6. Recall that binary cross-entropy in logit space is

$$
\ell ( y , z ) = \log ( 1 + \exp ( z ) ) - y z .\tag{27}
$$

For a fixed x, write the M-estimator aggregate as

$$
\bar { Z } _ { c , M } ( x ) = \mu _ { c } ( x ) + \epsilon _ { c , M } ( x ) , \qquad \mathbb { E } \left[ \epsilon _ { c , M } ( x ) \mid x \right] = 0 ,\tag{28}
$$

where $\mu _ { c } ( x ) = \mathbb { E } [ Z _ { c } ( x ) \mid .$ x] is the estimator-family mean.

A second-order Taylor expansion around $\mu _ { c } ( x )$ gives

$$
\begin{array} { l } { { \ell \left( y , \mu _ { c } + \epsilon _ { c , M } \right) = \ell ( y , \mu _ { c } ) + \ell ^ { \prime } ( y , \mu _ { c } ) \epsilon _ { c , M } } } \\ { { \mathrm { } } } \\ { { \displaystyle \qquad + \frac { 1 } { 2 } \ell ^ { \prime \prime } ( y , \mu _ { c } ) \epsilon _ { c , M } ^ { 2 } + { \cal O } \left( | \epsilon _ { c , M } | ^ { 3 } \right) . } } \end{array}\tag{29}
$$

Taking expectation over estimator randomness conditional on x eliminates the first-order term:

$$
\begin{array} { r } { \mathbb { E } \left[ \ell \left( y , \bar { Z } _ { c , M } ( x ) \right) \mid x \right] \approx \ell \left( y , \mu _ { c } ( x ) \right) \qquad } \\ { \qquad + \frac { 1 } { 2 } \ell ^ { \prime \prime } \left( y , \mu _ { c } ( x ) \right) \operatorname { V a r } \left( \bar { Z } _ { c , M } ( x ) \mid x \right) . } \end{array}\tag{30}
$$

For binary cross-entropy,

$$
\ell ^ { \prime \prime } ( y , z ) = \sigma ( z ) \left( 1 - \sigma ( z ) \right) ,\tag{31}
$$

which is independent of the label y. Averaging over the data distribution therefore gives

$$
\mathcal { L } ( c , M ) \approx \mathcal { L } ( \mu _ { c } ) + \frac { 1 } { 2 } \mathbb { E } _ { x } \left[ \sigma ( \mu _ { c } ( x ) ) \left( 1 - \sigma ( \mu _ { c } ( x ) ) \right) \mathrm { V a r } \left( \bar { Z } _ { c , M } ( x ) \mid x \right) \right] ,\tag{32}
$$

which is Eq. 3 in the main text, up to the omitted third-order remainder.

Comparing a single estimator with an M-estimator aggregate and using Eq. 5, the corresponding reduction in expected BCE is

$$
\Delta \mathcal { L } ( c , M ) \approx \frac { 1 } { 2 } \mathbb { E } _ { x } \left[ \sigma ( \mu _ { c } ( x ) ) \left( 1 - \sigma ( \mu _ { c } ( x ) ) \right) V _ { \mathrm { n s } } ( c , M ; x ) \right] .\tag{33}
$$

Thus, BCE improvement is locally proportional to curvature-weighted reducible predictive variation. When the estimator families being compared have similar mean predictions, the curvature term $\sigma ( \mu _ { c } ( x ) ) ( 1 - \sigma ( \mu _ { c } ( x ) ) )$ is approximately shared, so differences in averaging benefit are primarily governed by $V _ { \mathrm { n s } } ( c , M ; x )$ . This motivates the reducible-variance diagnostic used in the main experiments. The theoretical proportionality applies directly to expected BCE; the corresponding association with AUC reported in the main text is empirical rather than implied by this derivation.

The approximation is most accurate when estimator disagreement around $\mu _ { c } ( x )$ is modest. For broader or heavy-tailed estimator distributions, the omitted third-order and higher-order terms may become non-negligible.

## B.3 EXACT IDENTITY FOR LITERAL MEAN-LOGIT AGGREGATION

For literal mean-logit aggregation, binary cross-entropy also admits an exact inequality. For a fixed example (x, y) and estimator logits $z _ { 1 } , \dots , z _ { M }$ , let

$$
\bar { z } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } z _ { m } .\tag{34}
$$

Since

$$
\ell ( y , z ) = \mathrm { s o f t p l u s } ( z ) - y z ,\tag{35}
$$

we obtain

$$
\begin{array} { r l r } {  { \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \ell ( y , z _ { m } ) - \ell ( y , \bar { z } ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathrm { s o f t p l u s } ( z _ { m } ) - \mathrm { s o f t p l u s } ( \bar { z } ) } } \\ & { } & { \geq 0 , } \end{array}\tag{36}
$$

where the inequality follows from convexity of softplus.

Equation 36 shows that literal mean-logit aggregation cannot have higher BCE than the average BCE of its constituent logits on the same example. This exact result does not, however, quantify the gain in terms of reducible variance, nor does it directly characterize parameter averaging such as EMA or a distilled student that only approximates an ensemble prediction. For this reason, the main text uses the local variance formulation as the common explanatory framework across estimator sources.

## C COMBINING MULTIPLE ESTIMATOR AXES

## C.1 TWO-AXIS DECOMPOSITION

Let A and B denote two estimator sources and let $Z _ { A , B } ( x )$ denote the corresponding random logit for a fixed input x. Applying the law of total variance while treating A as the first averaging axis gives

$$
\operatorname { V a r } \left( Z _ { A , B } ( x ) \mid x \right) = \mathbb { E } _ { B } \left[ \operatorname { V a r } _ { A } \left( Z _ { A , B } ( x ) \mid B , x \right) \right] + \operatorname { V a r } _ { B } \left[ \mathbb { E } _ { A } \left( Z _ { A , B } ( x ) \mid B , x \right) \right] .\tag{37}
$$

Define

$$
V _ { A } ( \boldsymbol { x } ) = \mathbb { E } _ { B } \left[ \operatorname { V a r } _ { A } \left( Z _ { A , B } ( \boldsymbol { x } ) \mid B , \boldsymbol { x } \right) \right] ,\tag{38}
$$

which is the variation directly accessible to averaging along axis A. After averaging over A, the remaining predictor is

$$
\mu _ { A } ( B ; x ) = \operatorname { \mathbb { E } } _ { A } \left[ Z _ { A , B } ( x ) \mid B , x \right] ,\tag{39}
$$

whose residual variation across B is

$$
V _ { B | A } ( x ) = \mathrm { V a r } _ { B } \left[ \mu _ { A } ( B ; x ) \right] .\tag{40}
$$

Hence

$$
\operatorname { V a r } \left( Z _ { A , B } ( x ) \mid x \right) = V _ { A } ( x ) + V _ { B \mid A } ( x ) .\tag{41}
$$

Up to the same local BCE-curvature weighting derived in Appendix B.2, $V _ { A } ( x )$ determines the standalone loss reduction available from averaging along axis $A ,$ , while $V _ { B | A } ( x )$ determines the residual variation available to a subsequent estimator axis B. This is the basis for the marginal-gain analysis in Section 5.2.

The decomposition is order-dependent. Reversing the averaging order gives

$$
\operatorname { V a r } \left( Z _ { A , B } ( x ) \mid x \right) = V _ { B } ( x ) + V _ { A | B } ( x ) ,\tag{42}
$$

and in general

$$
V _ { A } ( x ) \neq V _ { A | B } ( x ) , \qquad V _ { B } ( x ) \neq V _ { B | A } ( x ) .\tag{43}
$$

The total predictive variance is unchanged, but its attribution to the estimator axes depends on the averaging order.

## C.2 THREE-AXIS DECOMPOSITION FOR RECAP

Let S, T, and R denote independent training runs, checkpoints along a training trajectory, and inference-time routes, respectively. For a fixed input $x ,$ let $\dot { Z } _ { S , T , R } ( x )$ denote the resulting random logit. We use the operational nesting order

$$
R \to T \to S ,\tag{44}
$$

corresponding to route averaging first, trajectory averaging next, and cross-run averaging last.

Applying the law of total variance recursively gives

$$
\begin{array} { r l } & { \mathrm { V a r } \left( Z _ { S , T , R } ( x ) \mid x \right) = \underbrace { \mathbb { E } _ { S , T } \left[ \mathrm { V a r } _ { R } \left( Z _ { S , T , R } ( x ) \mid S , T , x \right) \right] } _ { V _ { R } ( x ) } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x } + \underbrace { \mathbb { E } _ { S } \left[ \mathrm { V a r } _ { T } \left( \mathbb { E } _ { R } \left[ Z _ { S , T , R } ( x ) \mid S , T , x \right] \mid S , x \right) \right] } _ { V _ { T \mid R } ( x ) } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x } + \underbrace { \mathrm { V a r } _ { S } \left[ \mathbb { E } _ { T , R } \left( Z _ { S , T , R } ( x ) \mid S , x \right) \right] } _ { V _ { S \mid T , R } ( x ) } . } \end{array}\tag{45}
$$

The three terms have direct operational interpretations:

$V _ { R } ( x )$ is variation accessible to inference-time route averaging;

$V _ { T | R } ( x )$ is trajectory variation remaining after route averaging;

$V _ { S | T , R } ( x )$ is cross-run variation remaining after both route and trajectory averaging.

Equation 45 explains why gains from different estimator sources need not add linearly. If two axes expose overlapping predictive variation, averaging one reduces the residual variation available to the next; if they expose largely distinct variation, their gains can be closer to additive. There are $3 ! = 6$ possible nestings of $( \bar { S , \cal T , R ) }$ . Their component values generally differ, although all decompositions sum to the same total predictive variance. The decomposition is an output-space attribution of the variation associated with the three estimator sources; it should not be interpreted as the literal sequence of operations implemented by RECAP. For a fixed attribution convention, we use the nesting order $R  \bar { T }  S$ , attributing route-accessible variation first, followed by residual trajectory and cross-run variation.

## D HYPERPARAMETER SENSITIVITY

Table 4: Hyperparameter sensitivity of RECAP on Criteo. We vary one method-specific hyperparameter at a time while keeping all other settings fixed. Logloss is the test logloss after validation-fitted affine calibration.
<table><tr><td>Hyperparameter</td><td>Value</td><td>AUC (%) ↑</td><td>Logloss ↓</td></tr><tr><td rowspan="5">EMA decay β</td><td>0.9900</td><td>81.735</td><td>0.43487</td></tr><tr><td>0.9950</td><td>81.744</td><td>0.43482</td></tr><tr><td>0.9990</td><td>81.760</td><td>0.43461</td></tr><tr><td>0.9995</td><td>81.765</td><td>0.43456</td></tr><tr><td>0.9999</td><td>81.770</td><td>0.43450</td></tr><tr><td rowspan="5">KD weight  $\lambda _ { \mathrm { K D } }$ </td><td>0.00</td><td>81.665</td><td>0.43535</td></tr><tr><td>0.25</td><td>81.743</td><td>0.43461</td></tr><tr><td>0.50</td><td>81.766</td><td>0.43443</td></tr><tr><td>1.00</td><td>81.765</td><td>0.43456</td></tr><tr><td>2.00</td><td>81.763</td><td>0.43454</td></tr><tr><td rowspan="5">Teacher seeds  $M _ { \mathrm { s e e d } }$ </td><td>1</td><td>81.704</td><td>0.43502</td></tr><tr><td>2</td><td>81.744</td><td>0.43478</td></tr><tr><td>3</td><td>81.756</td><td>0.43469</td></tr><tr><td>5</td><td>81.765</td><td>0.43456</td></tr><tr><td>8</td><td>81.768</td><td>0.43454</td></tr><tr><td rowspan="4">LoRA rank  $r _ { \mathrm { L o R A } }$ </td><td>24</td><td>81.763</td><td>0.43459</td></tr><tr><td>48</td><td>81.765</td><td>0.43444</td></tr><tr><td>96</td><td>81.765</td><td>0.43456</td></tr><tr><td>192</td><td>81.767</td><td>0.43444</td></tr><tr><td rowspan="4">Route adapters  $M _ { \mathrm { r o u t e } }$ </td><td>2</td><td>81.759</td><td>0.43452</td></tr><tr><td>4</td><td>81.767</td><td>0.43448</td></tr><tr><td>8</td><td>81.765</td><td>0.43456</td></tr><tr><td>16</td><td>81.764</td><td>0.43444</td></tr></table>

We study the sensitivity of RECAP to its main method-specific hyperparameters on Criteo. We vary one hyperparameter at a time while keeping all remaining settings fixed to the default configuration used in the main experiments. We consider five quantities: the EMA decay $\beta ,$ the distillation weight $\lambda _ { \mathrm { K D } }$ , the number of teacher seed models $M _ { \mathrm { s e e d } }$ , the LoRA rank $r _ { \mathrm { L o R A } }$ , and the number of route adapters $M _ { \mathrm { r o u t e } }$ . These parameters control trajectory averaging, the strength and breadth of cross-run distillation, and the capacity of the route-scaling mechanism. Unless otherwise specified, each configuration is evaluated using the same training protocol and data split as the main experiments.

Table 4 reports the resulting test AUC and Logloss. Overall, RECAP is stable across a broad range of settings, with the default configuration lying in regions of consistently strong performance rather than at isolated optima. Performance varies only modestly across EMA decay values, while nonzero distillation weights consistently outperform the no-distillation setting. Increasing the number of teacher seeds improves performance with diminishing returns, consistent with the saturation behavior observed for independent estimators. Likewise, increasing the LoRA rank or the number of route adapters beyond moderate values provides little additional improvement.

These results indicate that RECAP does not depend on narrowly tuned hyperparameters. In particular, its performance remains robust across the trajectory-averaging, cross-run distillation, and routecapacity settings considered here.