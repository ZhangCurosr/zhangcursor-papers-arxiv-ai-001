# ASSOCIATION PROFILE CONDITIONING IN A SET-TEMPORAL TRANSFORMER FOR CROSS-SESSION INTRACORTICAL MOTOR DECODING

Xinyuan Zhang<sup>1,∗</sup>, Handong Mo<sup>2,∗</sup>, Pengfei Wen<sup>2,∗</sup>, Shuang Liang<sup>1</sup>, Jichang Yang<sup>1</sup>, Yan Zeng<sup>1</sup>, Zhongrui Wang<sup>2,†</sup>, Han Wang<sup>1,†</sup>

<sup>1</sup>The University of Hong Kong, Hong Kong SAR, China <sup>2</sup>Southern University of Science and Technology, Shenzhen, China <sup>∗</sup>Equal contribution. <sup>†</sup>Correspondence: wangzr@sustech.edu.cn; hanwang6@hku.hk

## ABSTRACT

Intracortical motor decoders degrade across sessions because the set of recorded units changes and persisting units can alter how their firing relates to behavior. Most existing methods update network weights on each new session or rely on unlabeled activity, which does not directly reveal such changes. We present APST, an Association Profile-conditioned Set-Temporal transformer that adapts to new sessions with all network weights frozen. From a few labeled calibration trials, APST summarizes how each unit’s firing relates to behavior in a four-dimensional association profile computed in closed form. The profiles condition a set-attention encoder that accepts any number and order of units, followed by a causal transformer for streaming decoding. On held-out DANDI688 sessions from two monkeys, APST reaches velocity $R ^ { 2 }$ of 0.78 and 0.81, versus 0.40 and 0.58 for a variant that uses neural activity alone, and matches or exceeds an RNN fine-tuned on the same trials. On FALCON private held-out evaluation, it attains $R ^ { 2 }$ of 0.65, 0.42, and 0.44 on M1, M2, and H1.

Index Terms— Intracortical motor decoding, cross-session adaptation, brain–computer interfaces, neural signal decoding.

## 1. INTRODUCTION

Intracortical motor decoding maps neural activity to continuous motor output, such as limb kinematics or muscle activity. Slight array motion relative to tissue [1,2] changes the set of recorded units (neurons or channels) across sessions (recording days). Units are lost, gained, or drift, and persisting units can change their associations with behavior [3] (Fig. 1). Although the low-dimensional motor manifold is comparatively stable [4], a decoder fit on Day 0 degrades on Day N without target-session calibration. EEG adaptation often assumes a fixed channel layout [5, 6]; in intracortical recordings, a fixed electrode index need not denote the same neural population across sessions.

Existing approaches adapt pretrained decoders through latent alignment [7–9] or target-session fine-tuning [10]. Permutationinvariant architectures accept unit sets that change across sessions [11], and SPINT infers unit embeddings from unlabeled calibration activity without target-session gradient updates [12]. Unlabeled activity, however, cannot resolve all drift: a persisting unit may keep its firing statistics while its association with behavior changes [13]. Behavioral labels can resolve it, and they are routinely available: clinical intracortical BCIs are periodically recalibrated with short blocks of cued movements [14], and FALCON [3] formalizes this setting by releasing a few labeled calibration trials at the start of each held-out session. Labeled few-shot fine-tuning [10] exploits these labels, but only by updating network weights on each new session, which requires backpropagation at deployment and risks overfitting a few trials. Whether a few labeled trials can adapt a decoder whose weights stay frozen remains unclear.

Each unit’s association with behavior can be summarized from a few labeled calibration trials. Directional tuning relates a unit’s firing rate to movement direction and yields a preferred direction and modulation depth [15]. Encoding models relate neural responses to behavioral covariates more generally [16, 17]; for multidimensional kinematics or muscle EMG, behavior-weighted associations are projected onto a source-fitted singular value decomposition (SVD) basis. Either way, each unit receives an association profile, a taskconditioned functional identity that complements the activity signatures estimated from the same trials.

![](images/f8b4f782a5520f9ef110ed1cd07492684345f40d18d18a8be6508f7dcd1a2a85.jpg)  
Fig. 1. Across sessions, electrode drift changes the recorded units, and a decoder fit on Day 0 degrades on Day N without target calibration. APST adapts with frozen weights by conditioning on association profiles estimated from a few labeled calibration trials.

We introduce APST<sup>1</sup>, an Association Profile-conditioned Set-Temporal transformer for cross-session decoding without target-session backpropagation. Specifically: (i) APST estimates a 4-D association profile for each unit from a few labeled calibration trials in closed form, with all source-trained weights frozen. (ii) The profiles condition a permutation-invariant set-attention encoder that accepts unit sets of any size, followed by a sliding-window causal temporal transformer for streaming motor decoding. (iii) We evaluate APST on single-unit recordings from DANDI Archive 000688 (DANDI688; Sub-C and Sub-M) [4, 18] and official FALCON benchmarks (M1, M2, H1) [3]. Ablations show that association profiles raise velocity $R ^ { 2 }$ from 0.40 to 0.78 on Sub-C over an activity-only variant, and that profiles shuffled across units score below activity-only.

## 2. METHOD

Calibration (Fig. 2(a)) gives each observed unit a static profile and a cached identity; streaming decoding (Fig. 2(b)) combines them with live activity and decodes causally.

## 2.1. Notation and calibration boundary

Let s index sessions and $u = 1 , \ldots , N _ { s }$ index observed units (sorted single units or unsorted channels). A target calibration set provides $M _ { s }$ labeled trials $\mathcal { C } _ { s } = \{ ( X _ { s , j } , Y _ { s , j } ) \} _ { j = 1 } ^ { M _ { s } }$ , with neural activity $X _ { s , j }$ and behavioral targets $Y _ { s , j }$ . All network parameters are trained on source sessions and frozen during target deployment. From $\mathcal { C } _ { s } .$ , the system computes a static 4-D association profile $p _ { s , u }$ and unit identity $e _ { s , u }$ without target-session backpropagation.

![](images/1ddaaa8e5c5a00d904fb966d87bf65b760f5bb735566a54a5e7e3811c7985385.jpg)  
Fig. 2. APST overview. (a) Target-session calibration: from M labeled trials, the association estimator produces each unit’s profile $p _ { u } ;$ the profile conditions the activity signature at site ⃝1 through FiLM and concatenation, forming unit identity $e _ { u } .$ . (b) Streaming decoding: live activity is combined with the cached identity and profile at site ⃝2 , aggregated across units by set attention with learned slot queries, then decoded by a sliding-window causal transformer.

## 2.2. Association profiles

An association profile summarizes how a unit’s calibration activity relates to task behavior (activity Directional profiles combine three fitted coefficients with their derived modulation depth, while multidimensional association summaries are projected to the same four-dimensional interface.

Encoding models relate unit responses to behavioral covariates [16, 17]. Let $r _ { s , t , u }$ be the response of unit u at calibration sample t (a trial or time bin). The task supplies $\phi _ { s , t } \in \mathbb { R } ^ { d _ { \phi } }$ and nonnegative weight $\omega _ { s , t }$ . The raw coefficient vector is

$$
a _ { s , u } ^ { \mathrm { r a w } } = \underset { v \in \mathbb { R } ^ { d _ { \phi } } } { \operatorname { a r g m i n } } \sum _ { t } \omega _ { s , t } ( r _ { s , t , u } - \phi _ { s , t } ^ { \top } v ) ^ { 2 } + \eta \| v \| _ { 2 } ^ { 2 } .\tag{1}
$$

The closed-form solution requires one calibration pass and no targetsession gradients. Behavior enters through either the design ϕ or the weights $\omega ,$ and the two estimators below differ in this choice.

## 2.3. Directional tuning profiles (M2 and DANDI688)

For two-dimensional movements, behavior enters the design via trial movement direction $\theta _ { s , j }$ . With $\phi _ { s , j } ~ = ~ [ 1 $ , cos $\theta _ { s , j } ,$ sin $\theta _ { s , j } ]$ , uniform weights, and $\eta ~  ~ 0$ , Eq. (1) corresponds to first-harmonic least-squares tuning [15]:

$$
\widehat { R } _ { s , j , u } = b _ { s , u } + a _ { s , u } \cos \theta _ { s , j } + d _ { s , u } \sin \theta _ { s , j } , \ \rho _ { s , u } = \sqrt { a _ { s , u } ^ { 2 } + d _ { s , u } ^ { 2 } } ,\tag{2}
$$

where $R _ { s , j , u }$ is the movement-window mean count per bin. The profile collects the directional coefficients, their modulation depth, and the baseline response:

$$
p _ { s , u } = \left( \left[ a _ { s , u } , d _ { s , u } , \rho _ { s , u } , b _ { s , u } \right] ^ { \top } - \mu ^ { \mathrm { d i r } } \right) \mathcal { O } \sigma ^ { \mathrm { d i r } } ,\tag{3}
$$

where <sup>⊤</sup> denotes transposition, $\oslash$ element-wise division, and $\mu ^ { \mathrm { d i r } }$ and $\boldsymbol { \sigma } ^ { \mathrm { d i r } }$ the source-session mean and standard deviation.

## 2.4. SVD-compressed association profiles (M1 and H1)

For M1 (16-channel EMG) and H1 (7-D velocity), behavior enters through sample weights $\omega _ { s , t } ~ = ~ w _ { s , t , k }$ for each condition $k ,$ and Eq. (1) is evaluated on standardized unit responses r˜. M1 uses $w _ { s , t , k } = \mathrm { m a x } ( \mathrm { E M G } _ { s , t , k } , 0 ) / \mathrm { R M S } _ { k } ^ { \mathrm { s r c } }$ ; H1 divides velocity by its source RMS, then takes softplus positive and negative states, yielding $K = 1 6$ for M1 and K = 14 for H1.

We normalize unit responses, compute behavior-weighted associations, and project the resulting matrix to four dimensions:

$$
\begin{array} { r l } & { \sigma _ { s , u } ^ { \mathrm { r a t e } } = \frac { \sqrt { \operatorname* { m a x } ( \Delta \bar { r } _ { s , u } , 1 ) } } { \Delta } , \qquad \tilde { r } _ { s , t , u } = \frac { r _ { s , t , u } - \bar { r } _ { s , u } } { \sigma _ { s , u } ^ { \mathrm { r a t e } } } , } \\ & { A _ { s , u , k } = \frac { \sum _ { t } w _ { s , t , k } \tilde { r } _ { s , t , u } } { \sum _ { t } w _ { s , t , k } + \eta } , } \\ & { \qquad P _ { s } = ( A _ { s } V _ { 4 } - \mathbf { 1 } \mu ^ { \top } ) \operatorname * { d i a g } ( \boldsymbol { \nu } ) ^ { - 1 } . } \end{array}\tag{4}
$$

Here $r _ { s , t , u }$ is firing rate (in Hz) with calibration mean $\bar { r } _ { s , u } ,$ and $\boldsymbol { \sigma } _ { s , u } ^ { \mathrm { r a t e } }$ is a Poisson-floor rate scale with effective bin width $\Delta$ . In Eq. (1), fitting each state with scalar design $( \phi \equiv 1 , \omega _ { s , t } = w _ { s , t , k } )$ on r˜ yields association matrix $A _ { s } \in \mathbb { R } ^ { \breve { N } _ { s } \times \breve { K } }$ , shrunk toward zero when $\sum _ { t } w _ { s , t , k }$ is small. Projecting onto the top four right singular vectors $V _ { 4 } ~ \in ~ \mathbb { R } ^ { K \times 4 }$ of uncentered pooled source associations $\boldsymbol { A _ { \mathrm { s r c } } } = \boldsymbol { U } \boldsymbol { \Sigma } \boldsymbol { V } ^ { \intercal }$ and standardizing by source moments $( \mu , \nu )$ outputs profile matrix $P _ { s } = [ p _ { s , 1 } ^ { \top } ; \ldots ; p _ { s , N _ { s } } ^ { \top } ] \in \mathbb { R } ^ { N _ { s } \times 4 }$ , matching the 4-D interface. M1 uses one shared post-projection source RMS scale and H1 coordinatewise source standard deviations.

## 2.5. Profile-conditioned unit identities

Following calibration-based unit representations [12], we combine activity signatures with association profiles to form taskconditioned unit identities. Calibration activity $x _ { s , j , u } ~ \in ~ \mathbb { R } ^ { B _ { \mathfrak { c } } }$ cal (binned counts of unit u in calibration trial $j )$ passes through a shared layer and is trial-averaged into the activity signature $\begin{array} { r } { \overline { { h } } _ { s , u } = M _ { s } ^ { - \bar { 1 } } \sum _ { j = 1 } ^ { M _ { s } } \mathrm { R e L U } ( W _ { a } x _ { s , j , u } + b _ { a } ) } \end{array}$

At identity-conditioning site ⃝1 in Fig. 2(a), profile $p _ { s , u }$ modulates this signature through FiLM [19] and is concatenated before

the identity MLP g:

$$
\begin{array} { r l } & { [ \gamma _ { s , u } ; \beta _ { s , u } ] = W _ { o } \operatorname { R e L U } ( W _ { i } p _ { s , u } + b _ { i } ) + b _ { o } , } \\ & { \qquad \tilde { h } _ { s , u } = ( 1 + \gamma _ { s , u } ) \odot \overline { { h } } _ { s , u } + \beta _ { s , u } , } \\ & { \qquad e _ { s , u } = g ( [ \tilde { h } _ { s , u } ; \ p _ { s , u } ] ) . } \end{array}\tag{5}
$$

Here $[ \cdot ; \cdot ]$ denotes concatenation and ⊙ element-wise multiplication. FiLM lets the profile scale and shift individual activity features, while concatenation passes it to g directly. Following adaLN-Zero [20], $W _ { o }$ and $b _ { o }$ are zero-initialized, so training starts from concatenation alone; the unit identities $e _ { s , u }$ are computed once and cached.

## 2.6. Set-temporal streaming decoder

At token-conditioning site $\textcircled{2}$ in Fig. 2(b), a shared causal convolution over five bins maps live activity to $l _ { s , u , t } = \psi ( x _ { s , u , t - 4 : t } )$ . The projected identity is added to this activity representation, and $p _ { s , u }$ is concatenated before the token MLP f and layer normalization LN form the unit token:

$$
z _ { s , u , t } = \mathrm { L N } ( f ( [ l _ { s , u , t } + W _ { e } e _ { s , u } ; p _ { s , u } ] ) ) .\tag{6}
$$

Eight learned slot queries S aggregate the unit tokens $Z _ { s , t }$ with multi-head set attention [21, 22]. Set aggregation is invariant to jointly permuting each unit’s activity, identity, and association profile. Padding mask $m _ { s }$ excludes padded units before softmax, which permits variable $N _ { s } \colon$

$$
\begin{array} { r l } & { \widetilde { S } _ { s , t } = \mathrm { L N } ( S ) + \mathrm { M H A } ( \mathrm { L N } ( S ) , Z _ { s , t } , Z _ { s , t } ; m _ { s } ) , } \\ & { q _ { s , t } = W _ { q } \mathrm { v e c } \Big ( \widetilde { S } _ { s , t } + \mathrm { F F N } ( \mathrm { L N } ( \widetilde { S } _ { s , t } ) ) \Big ) . } \end{array}\tag{7}
$$

Here MHA computes cross-attention from queries S to unit tokens $Z _ { s , t }$ under mask $m _ { s }$ , and vec flattens the slot matrix before linear projection $W _ { q }$ produces the 256-D population token ${ { q } _ { s , t } }$

Starting from ${ \cal H } _ { s , t } ^ { 0 } = q _ { s , t } ,$ four pre-LN causal transformer layers with eight heads process the population sequence through residual attention and FFN sublayers [23]. At time t, layer ℓ attends to the key/value pairs at t and the preceding $w _ { \ell } - 1$ bins. Within each causal window, attention logits include a learned relative-time bias inspired by ALiBi [24]. These local causal windows define the raw decoder receptive field $\begin{array} { r } { R = 5 + \sum _ { \ell } ( w _ { \ell } - 1 ) } \end{array}$ , including the five-bin convolution. The context length R is set from the temporal structure of each dataset and split evenly across the four causal layers. Perlayer key/value caches bound streaming state by this receptive field rather than the full session history. During source training, wholeunit dropout removes random units from the set.

A final layer normalization and task-specific readout give the raw prediction $\widehat { y } _ { s , t } ~ = ~ r _ { \mathrm { t a s k } } ( \mathrm { L N } ( H _ { s , t } ^ { 4 } ) )$ ), which an inference-only causal EMA smooths as $\widetilde { y } _ { s , t } = \alpha \widetilde { y } _ { s , t - 1 } + ( 1 - \alpha ) \widehat { y } _ { s , t }$ . The EMA $( \alpha = 1 / 3 )$ resets at session boundaries, starting from $\widetilde { y } _ { s , 0 } = \widehat { y } _ { s , 0 }$

## 3. EXPERIMENTS

## 3.1. Datasets and protocol

We evaluate APST across four continuous motor-decoding benchmarks binned at 20 ms: FALCON [3] M1 (rhesus primary motor cortex to 16-channel upper-limb EMG) [25], M2 (rhesus Utah arrays to 2-D finger-group velocity) [26], H1 (human Utah arrays to 7-D robotic arm and hand velocity) [27], and DANDI Archive 000688 [4, 18] (rhesus Utah array to 2-D cursor velocity). Sub-C and Sub-M are two different monkeys recorded under this same configuration. FALCON inputs are unsorted threshold crossings, scored via official EvalAI means on private evaluation data from held-out sessions, with reported SD; normalized latency is processing time divided by evaluation-data duration. DANDI688 uses sorted single units.

On DANDI688, Sub-C uses 18 source, 6 development, and 6 held-out sessions; Sub-M uses 6 source, 2 development, and 3 heldout sessions. Ablations and linear baselines use 32 labeled calibration trials per target session; comparisons with the RNN (Fig. 3(a,b), Table 3) use budgets of 4, 8, 16, and 32 trials. Performance is variance-weighted velocity $R ^ { 2 }$ per session followed by an equalsession mean. Hyperparameters are chosen per dataset on local development splits, primarily tuning the learning rate, the whole-unit dropout rate, the projection dimension of the identity mapping $W _ { e }$ into the decoding stream, and the slope initialization coefficients of the relative-time bias.

## 3.2. Main results

Table 1 reports official FALCON results, with baselines and oracle references from SPINT [12] and FALCON [3]. With all weights frozen, APST achieves $0 . 6 5 \pm 0 . 1 1 , 0 . 4 2 \pm 0 . 1 0 ,$ , and $0 . 4 4 \pm 0 . 1 5$ on M1, M2, and H1. Among methods without target-session gradients, it has the highest mean on M2 and H1 and is within 0.01 of SPINT on M1. It also exceeds the gradient-based NDT2 Multi [10] on M1, comes within 0.01 of it on M2, and matches the private-label RNN oracle on H1. Unlike SPINT, APST uses target-session labels; NDT2 Multi uses the same labeled calibration trials but updates network weights, whereas APST uses them only through closed-form profiles. Normalized latencies range from 0.09 to 0.15.

Table 2 reports official private held-in means (other entries from SPINT Table A1 [12]). Because held-in sessions contribute source-training data, the held-in minus held-out gap $( \mathrm { H I } - \mathrm { H O } )$ measures the accuracy lost on unseen sessions. APST’s held-in $R ^ { 2 }$ (0.77, 0.64, 0.60) matches the strongest baselines (tied best on M1, best on M2, second to NDT2 Multi on H1), so its held-out gains in

Table 1. Official FALCON private held-out performance. Entries are mean ± SD $R ^ { 2 }$ and normalized latency (lat.).
<table><tr><td>Method</td><td>Target adaptation</td><td colspan="2"> $\mathrm { M } 1 \ : ( R ^ { 2 } / \mathrm { l a t . } )$ </td><td colspan="2"> ${ \bf M } 2 ( R ^ { 2 } / \mathrm { l a t . } )$ </td><td colspan="2"> $\mathrm { H } 1 ( R ^ { 2 } / \mathrm { l a t . } )$ </td></tr><tr><td colspan="9">Gradient-free target-session adaptation</td></tr><tr><td>WF</td><td>None</td><td> $0 . 3 4 \pm 0 . 0 6$ </td><td>0.06</td><td> $0 . 0 6 \pm 0 . 0 4$ </td><td>0.08</td><td> $0 . 1 6 \pm 0 . 0 3$ </td><td>0.15</td></tr><tr><td>RNN</td><td>None</td><td> $- 0 . 6 0 \pm 0 . 4 5$ </td><td>0.03</td><td> $- 0 . 0 7 \pm 0 . 2 3$ </td><td>0.01</td><td> $0 . 0 9 \pm 0 . 1 8$ </td><td>0.02</td></tr><tr><td>SPINT</td><td>Unlabeled few-shot</td><td> $0 . 6 6 \pm 0 . 0 7$ </td><td>0.13</td><td> $0 . 2 6 \pm 0 . 1 3$ </td><td>0.13</td><td> $0 . 2 9 \pm 0 . 1 5$ </td><td>0.14</td></tr><tr><td>APST (ours)</td><td>Labeled few-shot</td><td> $\mathbf { 0 . 6 5 \pm 0 . 1 1 }$ </td><td>0.15</td><td> ${ \bf 0 . 4 2 \pm 0 . 1 0 }$ </td><td>0.09</td><td> ${ \bf 0 . 4 4 \pm 0 . 1 5 }$ </td><td>0.15</td></tr><tr><td colspan="8">Gradient-based target adaptation</td></tr><tr><td> $\mathbf { C y c l e G A N + W F }$ </td><td>Unlabeled few-shot</td><td> $0 . 4 3 \pm 0 . 0 4$ </td><td>0.07</td><td> $0 . 2 2 \pm 0 . 0 6$ </td><td>0.09</td><td> $0 . 1 2 \pm 0 . 0 6$ </td><td>0.16</td></tr><tr><td> $\mathrm { N _ { 0 } M A D + W F }$ </td><td>Unlabeled few-shot</td><td> $0 . 4 9 \pm 0 . 0 3$ </td><td>0.99</td><td> $0 . 2 0 \pm 0 . 1 0$ </td><td>0.91</td><td> $0 . 1 3 \pm 0 . 1 0$ </td><td>1.03</td></tr><tr><td>NDT2 Multi</td><td>Labeled few-shot</td><td> $0 . 5 9 \pm 0 . 0 7$ </td><td>0.13</td><td> $0 . 4 3 \pm 0 . 0 8$ </td><td>0.10</td><td> $0 . 5 2 \pm 0 . 0 \dot { 4 }$ </td><td>0.30</td></tr><tr><td colspan="8">Private-label oracle references</td></tr><tr><td>WF</td><td>Private labels</td><td> $0 . 5 3 \pm 0 . 0 4$ </td><td>0.06</td><td> $0 . 2 6 \pm 0 . 0 3$ </td><td>0.08</td><td> $0 . 2 1 \pm 0 . 0 4$ </td><td>0.14</td></tr><tr><td>RNN</td><td>Private labels</td><td> $0 . 7 5 \pm 0 . 0 5$ </td><td>0.04</td><td> $0 . 5 6 \pm 0 . 0 4$ </td><td>0.04</td><td> $0 . 4 4 \pm 0 . 1 3$ </td><td>0.08</td></tr><tr><td>NDT2 Multi</td><td>Private labels</td><td> $0 . 7 8 \pm 0 . 0 4$ </td><td>0.15</td><td> $0 . 5 8 \pm 0 . 0 4$ </td><td>0.10</td><td> $0 . 6 3 \pm 0 . 0 8$ </td><td>2.29</td></tr></table>

Note. Official private held-out evaluation. Bold indicates our method. Other methods from [12].

![](images/1ee1f7c641fdd692a62c2e7aff449407103291046fa407a20b231998b78b0e74.jpg)

(b)  
![](images/7623ca9e5e38960b419cab9084c951075acf58e159f205bfc4f789b50ebe0120.jpg)

![](images/fae5b749b95c8d1cc2f3d83acee259c39cefe69e15d169f6da621a5c20d0cbb5.jpg)

(d)  
![](images/1bc659cc7ab467ea044abb05428bea118fcbf042ae8b5e3f5d34f0746ea2ae7f.jpg)  
Fig. 3. Results on DANDI688 and FALCON (shared y-axis: velocity $R ^ { 2 } ) .$ . (a) DANDI688 held-out sessions at 32 calibration trials. (b) Calibration curves (solid: Sub-C, dashed: Sub-M; blue: APST, pink: RNN-FT). (c) Ablations on DANDI688 held-out and FALCON dev. (d) Conditioning-site ablation on FALCON dev. (c,d) Three-seed means; negative bars are truncated with values annotated.

Table 2. Official private held-in mean $R ^ { 2 } .$ . Parentheses denote heldin minus held-out gap (HI − HO). Bold indicates our method.
<table><tr><td>Method</td><td>M1</td><td>M2</td><td>H1</td></tr><tr><td>WF</td><td>0.46 (0.12)</td><td>0.15 (0.09)</td><td>0.20 (0.04)</td></tr><tr><td>RNN</td><td>0.52 (1.12)</td><td>0.20 (0.27)</td><td>0.31 (0.22)</td></tr><tr><td>SPINT</td><td>0.77 (0.11)</td><td>0.59 (0.33)</td><td>0.47 (0.18)</td></tr><tr><td>CycleGAN + WF</td><td>0.61 (0.18)</td><td>0.32 (0.10)</td><td>0.15 (0.03)</td></tr><tr><td>NoMAD + WF</td><td>0.64 (0.15)</td><td>0.35 (0.15)</td><td>0.21 (0.08)</td></tr><tr><td>NDT2 Multi</td><td>0.77 (0.18)</td><td>0.63 (0.20)</td><td>0.62 (0.10)</td></tr><tr><td>APST (ours)</td><td>0.77 (0.12)</td><td>0.64 (0.22)</td><td>0.60 (0.16)</td></tr></table>

Table 3. Held-out velocity $R ^ { 2 }$ at 32 calibration trials. Act-only: Activity-only (no target labels); Dates in 2015 (same sessions as Fig. 3, different seeds).
<table><tr><td>Sub-C APST Act-only RNN-FT</td><td>11-13 0.80 0.56</td><td>11-16 0.78 0.58</td><td>11-17 0.82 0.65 0.83</td><td>11-19 0.87 -0.02</td><td>11-20 0.83 0.46</td><td>12-01 0.58 0.06</td></tr><tr><td>Sub-M</td><td>0.83</td><td>0.81</td><td></td><td>0.83</td><td>0.85</td><td>0.48</td></tr><tr><td>APST</td><td>06-23 0.84</td><td>06-25 0.79</td><td>06-26 0.80</td><td></td><td></td><td></td></tr><tr><td>Act-only</td><td>0.49</td><td>0.60</td><td>0.66</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RNN-FT</td><td>0.73</td><td>0.69</td><td>0.69</td><td></td><td></td><td></td></tr></table>

Table 1 do not sacrifice within-session accuracy. Against SPINT, the other gradient-free method with competitive held-in accuracy, APST loses less on M2 (0.22 vs. 0.33) and H1 (0.16 vs. 0.18) and a similar amount on M1 (0.12 vs. 0.11). The smaller gaps of WF and CycleGAN + WF reflect their low held-in accuracy rather than robustness.

On DANDI688 held-out sessions (Fig. 3(a)), we compare APST with linear baselines, WF-FSS (Wiener filter with full session support) and PCA-WF (principal component analysis Wiener filter) [3, 28, 29], and an RNN baseline using FALCON’s architecture and search grid [3]. Zero-shot RNN (RNN-ZS) yields negligible accuracy (0.01 on Sub-C, −0.14 on Sub-M) due to unaligned unit channels, but rises to 0.77 and 0.70 when fine-tuned on target calibration trials (RNN-FT). With frozen weights, APST reaches 0.78 on Sub-C and 0.81 on Sub-M, matching or exceeding RNN-FT and outperforming the linear baselines in Fig. 3(a).

Sub-M’s held-out sessions come 8 to 11 days after its last source session, and APST outscores Activity-only and RNN-FT on all three (Table 3). Sub-C’s come 120 to 138 days after, and there RNN-FT is higher on four of six sessions, by at most 0.03. On the most distant session (12-01), where every method declines, APST retains 0.58 versus 0.48 for RNN-FT and 0.06 for Activity-only.

On the calibration-budget curves (Fig. 3(b)), we compare APST with RNN fine-tuning across 4–32 labeled trials. At 8 trials, APST reaches 0.75 on Sub-C and 0.77 on Sub-M, versus 0.64 and 0.47 for RNN-FT. On Sub-C, 8 APST trials exceed the 0.74 that fine-tuning reaches with 16 trials. On Sub-M, APST stays ahead at every budget by 0.11 to 0.38, indicating greater sample efficiency.

Target-session adaptation updates no network weights. With unit sets padded to 100 for both subjects, calibration costs 3.83 to 21.75 million multiply–accumulate operations (MACs) for 4 to 32 trials, of which the closed-form profile fit accounts for < 0.1% (< 15k operations). Fine-tuning the RNN updates all 118k weights on Sub-C and 367k on Sub-M through 512 updates of batch size 128 over 50-bin windows, totaling 383 and 1195 billion forward MACs.

## 3.3. Ablation studies

Fig. 3(c,d) evaluates ablations on DANDI688 held-out sessions and FALCON development splits. Activity-only is trained and calibrated without profiles, conditioning units on activity signatures alone. Static uses one source-trained identity table in place of calibrated identities. Profile shuffle reassigns profiles across units.

In Fig. 3(c), profiles raise performance over Activity-only from 0.40 to 0.78 on Sub-C and from 0.58 to 0.81 on Sub-M, and add 0.03, 0.18, and 0.20 on M1, M2, and H1. Shuffled profiles score below Activity-only in all five settings, reaching 0.00 and 0.03 on Sub-C and Sub-M. Because shuffling leaves each session’s set of profiles unchanged, this drop shows that APST uses each profile as a unit-specific identity rather than as a session-level summary. Static ranges from −0.32 to 0.38 across datasets, and APST exceeds it in every setting, with the smallest margins of 0.06 on M2 and 0.10 on H1. On DANDI688, units are sorted anew in each session, so a single identity table cannot follow them across sessions.

Fig. 3(d) ablates the conditioning site on FALCON over three seeds. Identity-only applies $p _ { s , u }$ only at site ⃝1 , Token-only only at site ⃝2 , and APST at both. Token-only and dual conditioning differ by at most 0.01 on every task, within the seed SD of 0.02, whereas Identity-only falls to 0.40 on H1 against 0.48; token conditioning therefore carries most of the benefit. We retain site ⃝1 because it adds no streaming cost: the identity is computed once during calibration and cached, and the zero-initialized FiLM starts as an identity map. All reported results use dual conditioning.

## 4. CONCLUSION AND DISCUSSION

APST adapts a source-trained set-temporal decoder through unit– behavior association profiles estimated from a few labeled targetsession calibration trials, with one 4-D profile interface shared by planar and high-dimensional tasks. Results on DANDI688 and FALCON support profile conditioning for cross-session decoding without target-session weight updates, while sliding-window causal attention bounds the streaming key/value state. The profile estimators and their source bases are still designed per task, and how the two conditioning sites divide the work across tasks remains to be characterized. Future work will explore unified profile estimators learned jointly with the decoder across datasets and modalities.

## 5. REFERENCES

[1] N. A. Steinmetz et al., “Neuropixels 2.0: A miniaturized high-density probe for stable, long-term brain recordings,” Science, vol. 372, no. 6539, pp. eabf4588, 2021.

[2] A. Yuan et al., “Multi-day neuron tracking in high density electrophysiology recordings using EMD,” bioRxiv preprint 2023.08.03.551724, 2024.

[3] B. Karpowicz et al., “Few-shot algorithms for consistent neural decoding (FALCON) benchmark,” Proc. NeurIPS, vol. 37, 2024.

[4] J. A. Gallego et al., “Long-term stability of cortical population dynamics underlying consistent behavior,” Nat. Neurosci., vol. 23, no. 2, pp. 260–270, 2020.

[5] A. Jiang, S. Hou, Y. Tang, and Y. Zhu, “Joint spatio-temporal filtering of motion imagery EEG signals for data alignment in transfer learning,” in Proc. IEEE ICASSP, 2024, pp. 2235–2239.

[6] G. Rahmani-Sane and S. Haghani, “Cross-session motor imagery classification using very short EEG windows and unsupervised domain adaptation for real-time BCI,” in Proc. IEEE ICASSP, 2026.

[7] A. D. Degenhart et al., “Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity,” Nat. Biomed. Eng., vol. 4, no. 7, pp. 672–685, 2020.

[8] A. Farshchian et al., “Adversarial domain adaptation for stable brain– machine interfaces,” in Proc. ICLR, 2019.

[9] B. M. Karpowicz et al., “Stabilizing brain-computer interfaces through alignment of latent dynamics,” Nat. Commun., vol. 16, pp. 4662, 2025.

[10] J. Ye, J. L. Collinger, L. Wehbe, and R. Gaunt, “Neural data transformer 2: Multi-context pretraining for neural spiking activity,” in Proc. NeurIPS, 2023, vol. 36.

[11] M. Azabou et al., “A unified, scalable framework for neural population decoding,” in Proc. NeurIPS, 2023, vol. 36, pp. 44937–44956.

[12] T. Le et al., “SPINT: Spatial permutation-invariant neural transformer for consistent intracortical motor decoding,” in Proc. NeurIPS, 2025, vol. 38, pp. 29248–29268.

[13] U. Rokni, A. G. Richardson, E. Bizzi, and H. S. Seung, “Motor learning with unstable neural representations,” Neuron, vol. 54, no. 4, pp. 653–666, 2007.

[14] B. Jarosiewicz et al., “Virtual typing by people with tetraplegia using a self-calibrating intracortical brain–computer interface,” Sci. Transl. Med., vol. 7, no. 313, pp. 313ra179, 2015.

[15] A. P. Georgopoulos, J. F. Kalaska, R. Caminiti, and J. T. Massey, “On the relations between the direction of two-dimensional arm movements and cell discharge in primate motor cortex,” J. Neurosci., vol. 2, no. 11, pp. 1527–1537, 1982.

[16] W. Truccolo, U. T. Eden, M. R. Fellows, J. P. Donoghue, and E. N. Brown, “A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects,” J. Neurophysiol., vol. 93, no. 2, pp. 1074–1089, 2005.

[17] N. G. Hatsopoulos, Q. Xu, and Y. Amit, “Encoding of movement fragments in the motor cortex,” J. Neurosci., vol. 27, no. 19, pp. 5105–5114, 2007.

[18] M. G. Perich, L. E. Miller, M. Azabou, and E. L. Dyer, “Longterm recordings of motor and premotor cortical spiking activity during reaching in monkeys,” DANDI Archive 000688, version 0.250122.1735, 2025.

[19] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proc. AAAI Conf. Artif. Intell., 2018, pp. 3942–3951.

[20] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 4195–4205.

[21] J. Lee et al., “Set Transformer: A framework for attention-based permutation-invariant neural networks,” in Proc. ICML, 2019, pp. 3744–3753.

[22] A. Jaegle et al., “Perceiver: General perception with iterative attention,” in Proc. ICML, 2021, pp. 4651–4664.

[23] A. Vaswani et al., “Attention is all you need,” in Proc. NeurIPS, 2017, vol. 30.

[24] O. Press, N. A. Smith, and M. Lewis, “Train short, test long: Attention with linear biases enables input length extrapolation,” in Proc. ICLR, 2022.

[25] A. G. Rouse and M. H. Schieber, “Spatiotemporal distribution of

location and object effects in primary motor cortex neurons during reach-to-grasp,” J. Neurosci., vol. 36, no. 41, pp. 10640–10653, 2016.

[26] S. R. Nason et al., “Real-time linear prediction of simultaneous and independent movements of two finger groups using an intracortical brain-machine interface,” Neuron, vol. 109, no. 19, pp. 3164– 3177.e8, 2021.

[27] J. L. Collinger et al., “High-performance neuroprosthetic control by an individual with tetraplegia,” The Lancet, vol. 381, no. 9866, pp. 557–564, 2013.

[28] J. I. Glaser et al., “Machine learning for neural decoding,” eNeuro, vol. 7, no. 4, 2020.

[29] J. P. Cunningham and B. M. Yu, “Dimensionality reduction for largescale neural recordings,” Nat. Neurosci., vol. 17, no. 11, pp. 1500– 1509, 2014.

## Acknowledgments

This work was supported by the Research Grants Council of Hong Kong under grants AoE/E-101/23-N, C1009-22G, and T45-701/22- R. The authors have no relevant financial or nonfinancial interests to disclose.

## Compliance with Ethical Standards

This study used only publicly released recordings and conducted no new human or animal experiments. The data are the FALCON M1, M2, and H1 tasks [3,25–27] and DANDI Archive dataset 000688 [4, 18]. Ethical approval for the original recordings was obtained by the data collectors.