# Adaptive Pullback: Reweighting Supervision at Penalty Jumps in On-Policy Distillation

Guangyu Liu<sup>\*1</sup> Siyuan Gan<sup>\*1</sup> Shilong Zhou<sup>\*1</sup>

Linjian Meng<sup>2</sup> Boyan Wang<sup>†1</sup> Jing Huo<sup>†1</sup>

<sup>1</sup>State Key Laboratory of Novel Software Technology, Nanjing University, Nanjing, China <sup>2</sup>Shanghai Artificial Intelligence Laboratory, Shanghai, China {liuguangyu, gansiyuan, zsdragon}@smail.nju.edu.cn {boyanwang, huojing}@nju.edu.cn

## Abstract

On-policy distillation (OPD) supervises student-generated responses using sampledtoken student–teacher log-ratios. A positive log-ratio penalizes a token that the student assigns higher probability to than the teacher, but its current value does not distinguish sustained high penalties from abrupt increases along a response. We examine these local changes by comparing consecutive blocks of tokens within the same response. Our diagnostic identifies penalty jumps, abrupt increases in average penalty that persist on held-out tokens. Across these jumps, penalties rise while teacher confidence decreases. We call this pattern penalty–confidence tension. Boundaries into high-penalty blocks do not exhibit comparable confidence changes, showing that local changes provide information beyond the current penalty value. Motivated by this observation, we propose Adaptive Pullback OPD (AP-OPD). At each detected jump, AP-OPD uses the preceding block as a local reference and uniformly scales all supervision coefficients in the following block so that its weighted average penalty matches the preceding block’s original mean. This reference determines the scaling factor without an additional strength constant. AP-OPD improves performance on both mathematical reasoning and code generation.

## 1 Introduction

On-policy distillation (OPD) transfers reasoning capabilities from a teacher model to a student through feedback on the student’s own rollouts. Training on student-generated prefixes brings the training distribution closer to the one visited at inference time and mitigates exposure bias (Bengio et al., 2015; Ranzato et al., 2016; Gu et al., 2024; Agarwal et al., 2024; Song and Zheng, 2026). OPD also provides token-level teacher feedback throughout each response (Gu et al., 2024; Agarwal et al., 2024). How to weight this supervision along a response is therefore an important question for OPD post-training.

In practical token-level OPD, supervision is determined by the sampled-token student–teacher logratio $r _ { t }$ (Fu et al., 2026). A positive log-ratio penalizes a token that the student assigns higher probability to than the teacher. The current penalty value describes how strongly a sampled token is penalized, but does not describe how that penalty changes along the response. A high penalty can persist across neighboring segments or emerge as an abrupt increase. Comparing consecutive blocks of tokens provides a local reference within the same response, allowing us to examine these increases separately from sustained high penalties. We investigate whether local penalty changes carry information relevant to supervision allocation and whether using this information to reweight supervision improves training.

Our diagnostic analysis shows that local penalty changes reveal differences in teacher predictions that the current penalty value alone does not express. We summarize each block by its blockaveraged penalty, the mean positive log-ratio within a block. We call an abrupt increase in the blockaveraged penalty between consecutive blocks a penalty jump. The analysis in Section 3 shows that the increase persists on held-out tokens not used to detect the jump and is concentrated in the first block after it. Across penalty jumps, the blockaveraged penalty increases while teacher confidence decreases, as reflected in rising entropy and falling top-1 probability. We call this observed pattern penalty–confidence tension. Control boundaries entering blocks with high penalties do not exhibit comparable changes in teacher confidence. This tension motivates examining whether supervision in the block after a jump should be reduced.

![](images/a8aa30e2f4c876bea5eebeb87ae4f5d560f51e1058e09d4dd0379e9914bc295e.jpg)

![](images/cca978e6075dbcb53b078fe3482846758be2177d70ede9ef0988d5480eb854be.jpg)  
Figure 1: Penalty–confidence tension at penalty jumps: penalties rise while teacher confidence decreases. Adaptive Pullback uniformly scales post-jump supervision to match the preceding block’s original mean penalty. Numbers are illustrative. $D _ { i } \colon$ block mean penalty; $H \colon$ teacher entropy; $P _ { \mathrm { m a x } } \mathrm { : }$ teacher top-1 probability; w: supervision weight.

Motivated by penalty–confidence tension, we propose Adaptive Pullback OPD (AP-OPD), illustrated in Figure 1. The method uses the preceding block as a local reference and, at each detected jump, uniformly scales down all signed supervision coefficients in the following block so that its weighted block-averaged penalty matches the preceding block’s original mean. This rule uses an adjacent segment of the same response to set the reference. A positive scaling factor preserves the signs and relative proportions of the token coefficients. The reference determines the scaling factor without an additional strength constant. AP-OPD detects jumps using log-probabilities already computed by OPD and requires no additional forward passes.

Training experiments support the value of this intervention. AP-OPD improves performance, whereas a magnitude-cap rule that selects blocks by their penalty values and removes a similar amount of penalty brings no gain over vanilla OPD. The removal amount alone therefore does not explain the improvement. We also compare AP-OPD with FiRe-OPD, which weights supervision using teacher and student confidence (Li et al., 2026b), and IW-OPD, which uses accumulated prefix discrepancy (Xie et al., 2026b). On mathematical reasoning, AP-OPD consistently improves macroaveraged Avg@k over vanilla OPD under both jump detectors and obtains a higher macro average than the compared reweighting methods. On code generation, with a different teacher and student, AP-OPD also achieves the highest macro average among the evaluated methods.

Our contributions are: (1) We identify and diagnose penalty–confidence tension at penalty jumps in OPD, showing that local penalty changes provide information about changes in teacher predictions beyond the current penalty value. (2) We propose Adaptive Pullback OPD, which uniformly scales supervision in the block after a jump using the preceding block as a local reference. (3) We demonstrate gains on mathematical reasoning and code generation, and show that removing a comparable amount of penalty with a magnitude-cap rule does not yield the same benefit.

## 2 Preliminaries

## 2.1 Token-Level OPD

Let x be a prompt, $\pi _ { \theta }$ the student, and $\pi _ { T }$ a frozen teacher. The student samples $\hat { y } _ { 1 : T } \sim \pi _ { \theta } ( \cdot \mid x )$ and $c _ { t } = \left( x , \hat { y } _ { < t } \right)$ denotes the context at position t. With $\ell _ { t } ^ { S } = \log \pi _ { \theta } ( \hat { y } _ { t } \mid c _ { t } )$ and $\ell _ { t } ^ { T } = \log \pi _ { T } ( \hat { y } _ { t } \ |$ $c _ { t } )$ , the sampled log-ratio is

$$
\begin{array} { r } { r _ { t } = \ell _ { t } ^ { S } - \ell _ { t } ^ { T } . } \end{array}\tag{1}
$$

The sequence-level reverse KL is $\mathbb { E } [ \sum _ { t } \boldsymbol { r } _ { t } ]$ (Gu et al., 2024); its policy gradient uses suffix returns (Williams, 1992; Sutton et al., 1999). Token-level OPD uses only the local term, with stop-gradient advantage $A _ { t } = - \operatorname { s g } ( r _ { t } )$ (Fu et al., 2026). Averaging over the $N _ { \mathrm { t o k } }$ valid tokens in a mini-batch (Yu et al., 2025) gives

$$
\mathcal { L } = - \frac { 1 } { N _ { \mathrm { t o k } } } \sum _ { n } \sum _ { t = 1 } ^ { T _ { n } } w _ { n , t } A _ { n , t } \ell _ { n , t } ^ { S } .\tag{2}
$$

Vanilla OPD sets $w _ { n , t } ~ = ~ 1$ . AP-OPD and the compared baselines all change the stop-gradient weights $w _ { n , t }$

Penalizing supervision. A single realization of $r _ { t }$ can take either sign. When $r _ { t } > 0 ,$ , the student assigns the sampled token a higher probability than the teacher, so $A _ { t } < 0$ and the update pushes that token down. This case dominates in practice: in the vanilla OPD rollouts analyzed below, tokens with $r _ { t } > 0$ carry 83.5% of $\textstyle \sum _ { t } | A _ { t } |$ . We define the token penalty as

$$
d _ { t } = \mathrm { s g } ( [ r _ { t } ] _ { + } ) , \quad [ u ] _ { + } = \operatorname* { m a x } ( u , 0 ) .\tag{3}
$$

## 2.2 Block Profile and Jumps

Single log-ratios fluctuate sharply between neighboring tokens, so we summarize a response at the level of contiguous blocks. We partition its valid tokens into blocks $\boldsymbol { B } _ { 1 } , \dots , \boldsymbol { B } _ { M }$ of B tokens; only the last block may be shorter, and $n _ { i } = | \boldsymbol { B } _ { i } |$ . The block-averaged penalty $D _ { i }$ is the mean positive logratio within a block, including zeros for nonpositive values; $s _ { i } ^ { 2 }$ is the within-block sample variance of $d _ { t }$ for $n _ { i } \geq 2 \mathrm { : }$

$$
D _ { i } = \frac { 1 } { n _ { i } } \sum _ { t \in \boldsymbol { B } _ { i } } d _ { t } , \ s _ { i } ^ { 2 } = \frac { 1 } { n _ { i } - 1 } \sum _ { t \in \boldsymbol { B } _ { i } } ( d _ { t } - D _ { i } ) ^ { 2 } .\tag{4}
$$

At boundary i, let $\Delta _ { i } \ : = \ : D _ { i + 1 } \ : - \ : D _ { i }$ . A jump is a positive change judged sufficiently large by a chosen detector. Let $\mathcal { P }$ denote the response’s block penalty profile and required statistics, and $\mathrm { D e t } ( i ; \mathcal { P } ) \in \{ 0 , 1 \}$ the detector’s decision. Over boundaries $i = 1 , \ldots , M - 1$ , the jump set is

$$
\begin{array} { r } { \mathcal { E } _ { \mathrm { D e t } } = \big \{ i : \Delta _ { i } > 0 , ~ \mathrm { D e t } ( i ; \mathcal { P } ) = 1 \big \} . } \end{array}\tag{5}
$$

The default z-gate accepts i when $n _ { i } , n _ { i + 1 } \geq 2$ and $\Delta _ { i } > z \hat { \sigma } _ { i }$ , where

$$
\hat { \sigma } _ { i } = \sqrt { \frac { s _ { i } ^ { 2 } } { n _ { i } } + \frac { s _ { i + 1 } ^ { 2 } } { n _ { i + 1 } } } .\tag{6}
$$

![](images/6f48b60a3fef81bf56d232467ccfc936f37d10b0deca6377dabdbd5be23f292c.jpg)  
Figure 2: Held-out rise in token penalty $d _ { t }$ around penalty jumps, abrupt increases in block-averaged penalty detected with the $z = 2$ gate, and three control boundaries. The rise is relative to the mean penalty of the block before each boundary. Boundaries are selected on one half of each block’s tokens and the curves are measured on the other half. The curves use 8-token bins and average both detection directions; bands for Jump and High-magnitude boundary are 95% bootstrap intervals clustered by prompt. The boundary lies at offset 0, and $B = 6 4$

We fix $z = 2 ;$ because adjacent tokens are correlated, this is a standardized threshold, not a calibrated significance level. For $i \in \mathcal { E } , B _ { i }$ and $B _ { i + 1 }$ are the pre- and post-jump blocks, and $\Delta _ { i }$ is the rise. Blocks only localize where to intervene; the loss in Eq. (2) remains token-level.

## 3 Diagnostic Analysis

We examine whether local changes in blockaveraged penalty reveal changes in teacher predictions beyond the current penalty value. We analyze 512 rollouts saved during vanilla OPD training at steps 70, 80, 90, and 100, one per distinct prompt, with the mathematical-reasoning student, teacher, and data of §5.1. Jumps are detected with the default z-gate at $B = 6 4 { \mathrm { ~ a n d ~ } } z = 2$ , about twice per response. We compare each jump with control boundaries from the same response at a similar relative position: the high-magnitude boundary, which enters the block with the highest blockaveraged penalty among the candidates; a random positive-change boundary with $\Delta _ { i } > 0 ;$ ; and a random boundary. The random boundary measures background variation. The random positive-change boundary asks whether selecting a rise alone explains the effect. The high-magnitude boundary

(a) Teacher entropy

![](images/b25ea0f05f628a01b9ac543f2199262ac0dcf1b060b8a360151cbef01f0c3378.jpg)  
(b) Teacher top-1 probability

![](images/16a0033745feee1014e533cd2c11d3248049c87d3bd16ee39bf4e8a8ad9cb4b2.jpg)

(c) Held-out log-prob. drop  
![](images/b76582f2e5ce14a1d1a830e3e3f8faf4cfb0951e341a0a0541ab0666bdef9933.jpg)  
Figure 3: Teacher prediction changes across penalty jumps, abrupt increases in block-averaged penalty detected with the $z = 2$ gate, and control boundaries $( B = 6 4 )$ . (a) Teacher entropy and (b) teacher top-1 probability, averaged over all tokens in the block before and the block after each boundary, on 153 independent matched groups. Their changes indicate decreasing teacher confidence at jumps, accompanying the penalty increase in the observed penalty–confidence tension. Appendix B.4 also reports all 353 groups. ∆ is after minus before, and the vertical axis in (b) is truncated. (c) Drop in the student’s and the teacher’s log-probability of the sampled tokens (before minus after), measured on held-out tokens from 324 prompts; (a,b) use all block tokens. Error bars are 95% bootstrap intervals.

asks whether entering a block with a high penalty is sufficient to account for the observations. Appendix B gives the full protocol.

## 3.1 The Penalty Jumps

Jumps are selected for a large $\Delta _ { i }$ , so measuring the rise on the same tokens would also count the noise that made the boundary pass the gate. We therefore use a held-out measurement: jumps are detected on the odd-position tokens of each block and the profile is measured on the even-position tokens, and vice versa. The held-out profiles in Figure 2 show an abrupt penalty increase at jumps, larger than at any of the three controls. The rise is concentrated in the first block after a jump. By the second block, the held-out block-averaged penalty is back at the level of a position-matched random boundary. The analysis in Section 5.4 shows that the rise persists after token-level clipping. Appendix B reports the same pattern in two further training runs.

The post-jump block carries about twice its token share of the penalty: 6.5% of tokens hold 13.5% of the penalty mass. The blocks selected by magnitude-cap are even more concentrated, so concentration alone does not single out jumps.

## 3.2 Penalty–Confidence Tension

We measure teacher prediction changes using entropy and top-1 probability separately. For the teacher distribution $p _ { t } ( v ) = \pi _ { T } ( v \mid c _ { t } )$ over vocabulary V, these are $\begin{array} { r } { H _ { t } = - \sum _ { v \in \mathcal { V } } p _ { t } ( v ) } \end{array}$ log p<sub>t</sub>(v)

and $\begin{array} { r c l } { P _ { \mathrm { m a x } , t } } & { = } & { \operatorname* { m a x } _ { v \in \mathcal { V } } p _ { t } ( v ) } \end{array}$ We run fullvocabulary teacher forward passes over the analysis rollouts and report their block means as H and $P _ { \mathrm { m a x } } .$ Panels a and b of Figure 3 show that across the detected jumps, H rises and $P _ { \mathrm { m a x } }$ falls, indicating a decrease in teacher confidence. Together with the increase in block-averaged penalty, this observed pattern constitutes penalty–confidence tension. Compared with high-magnitude boundaries, jumps show an additional entropy increase of 0.261 nats and an additional top-1 probability decrease of 0.083. These paired contrasts compare the postminus-pre changes within matched groups. They quantify how teacher predictions change across the boundary, rather than how confident the teacher is in the following block alone. Local penalty changes therefore reveal differences in teacher predictions beyond entering a block with a high penalty.

Figure 3c shows that, on the held-out tokens, the log-probabilities that both the student and the teacher assign to the sampled tokens fall across a jump. For both models, the drop is larger than across either control. This pattern cannot be explained by an increase in the student’s sampledtoken probability alone.

Implications for the method. Penalty– confidence tension motivates testing whether supervision in the post-jump block should be reduced. We use the preceding block in the same response as a local reference, without assuming that its penalty is correct or optimal. The rule in Section 4 detects penalty jumps rather than teacher uncertainty and does not use teacher confidence as an input. Training experiments in Section 5.3 evaluate the intervention: AP-OPD improves performance, whereas magnitude-cap removes a similar amount of penalty without a gain. This comparison does not establish a causal role for the confidence decrease.

## 4 Methodology

Adaptive Pullback OPD keeps the vanilla advantage $A _ { t }$ and the objective in Eq. (2), and changes only the token weights $w _ { t }$ . For each response, it computes the block profile in Eq. (4) and detects the jumps E using Eq. (5). It then uniformly scales supervision in each post-jump block using the preceding block as a local reference. Algorithm 1 gives the complete procedure.

## 4.1 Pullback Rule

At a jump $\textit { i } \in \mathcal { E }$ , AP-OPD uses the preceding block’s original block-averaged penalty $D _ { i }$ as a local reference. It uniformly scales all signed supervision coefficients in the post-jump block by one factor:

$$
w _ { t } = \left\{ \begin{array} { l l } { D _ { i } / D _ { i + 1 } , t \in \mathcal { B } _ { i + 1 } \mathrm { ~ f o r ~ s o m e ~ } i \in \mathcal { E } , } \\ { 1 , \qquad \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

The weighted block-averaged penalty therefore matches the reference:

$$
\frac { 1 } { n _ { i + 1 } } \sum _ { t \in \mathcal { B } _ { i + 1 } } w _ { t } d _ { t } = \frac { D _ { i } } { D _ { i + 1 } } D _ { i + 1 } = D _ { i } .\tag{8}
$$

A positive weight preserves the signs and relative proportions of the coefficients; a zero weight sets them all to zero. The scaling also reduces rewarding coefficients when $r _ { t } < 0$ . In contrast, setting every $r _ { t }$ to $D _ { i }$ erases token differences and turns negative coefficients into penalties when $D _ { i } > 0 ;$ subtracting $\Delta _ { i }$ turns tokens with $0 < r _ { t } < \Delta _ { i }$ from penalized to rewarded.

The rule triggers on detected local increases in block-averaged penalty. Detection and weights both use the observed, unweighted profile. A block that stays high after a jump therefore keeps weight one. If the next boundary is also a jump, the postjump block serves as its pre-jump block, again with its original, unweighted block-averaged penalty. Each block is weighted at most once.

## 4.2 Properties

Bounded weights. Since $0 \leq D _ { i } < D _ { i + 1 }$ for every jump, post-jump weights lie in [0, 1) and all other weights are one: AP-OPD never upweights a token or flips the sign of a coefficient.

Strength set by the data. The penalty mass removed from a post-jump block is $\textstyle \sum _ { t \in B _ { i + 1 } } ( 1 -$ $w _ { t } ) d _ { t } = n _ { i + 1 } \Delta _ { i }$ . Given the detector and its parameters, the local reference sets the scaling factor without an additional strength constant. For the default detector, $B = 6 4$ and $z = 2$ are fixed before training. The weights have no additional normalization, clipping, or floor. Removed penalty mass is a coefficient statistic, not a gradient or parameterupdate magnitude.

Selective intervention. As reported in Section $5 . 4 ,$ about a quarter of training responses contain no jump and are trained exactly as in vanilla OPD. As quantified in Appendix B, applying the same ratio at every boundary with $\Delta _ { i } > 0$ would remove 36.1% of the penalty mass instead of 11.6% under the default detector on the analysis rollouts.

Ranking detector. As an alternative to the z-gate, the top-α detector ranks the boundaries of each response by $\Delta _ { i }$ . It selects the

$$
K = \operatorname* { m i n } \bigl ( \lceil \alpha C \rceil , \mid \{ i : \Delta _ { i } > 0 \} \vert \bigr )\tag{9}
$$

largest positive changes, where C is the number of block boundaries and $\alpha = 5 \%$ , and applies the same pullback in Eq. (7).

## 4.3 Training Objective and Complexity

Substituting Eq. (7) into Eq. (2) gives the AP-OPD objective. The weights are stop-gradient quantities computed in one $O ( T )$ pass over log-probabilities that OPD already produces; they need no additional rollout, model forward pass, or full-vocabulary statistic, and inference is unchanged. The weights are not renormalized.

## 5 Experiments

## 5.1 Experimental Setup

Training. For mathematical reasoning, the student is Qwen3-1.7B-Base (Yang et al., 2025). The frozen teacher is Qwen3-8B-RL-Math, which is Qwen3-8B-Base trained on mathematics with GSPO (Zheng et al., 2025). Training prompts come from DAPO-Math-17K (Yu et al., 2025), and each method trains for 200 steps. For code generation,

<table><tr><td>Method</td><td>MATH500</td><td>AMC23</td><td>Minerva</td><td>Olympiad</td><td>AIME24</td><td>AIME25</td><td>Macro</td></tr><tr><td>Teacher (Qwen3-8B-RL-Math)</td><td>86.80</td><td>74.22</td><td>35.75</td><td>50.19</td><td>32.08</td><td>21.67</td><td>50.12</td></tr><tr><td>Student (Qwen3-1.7B-Base)</td><td>42.15</td><td>24.06</td><td>11.67</td><td>16.63</td><td>3.54</td><td>1.67</td><td>16.62</td></tr><tr><td>Vanilla OPD</td><td>72.90</td><td>43.75</td><td>20.96</td><td>33.63</td><td>12.40</td><td>7.29</td><td>31.82</td></tr><tr><td>IW-OPD</td><td>71.60</td><td>43.28</td><td>22.15</td><td>32.31</td><td>11.35</td><td>7.19</td><td>31.31</td></tr><tr><td>FiRe-OPD</td><td>72.80</td><td>45.39</td><td>21.74</td><td>32.83</td><td>11.88</td><td>8.44</td><td>32.18</td></tr><tr><td>AP-OPD (top-α)</td><td>72.97</td><td>44.77</td><td>22.06</td><td>34.26</td><td>13.02</td><td>8.65</td><td>32.62</td></tr><tr><td>AP-OPD (z-gate)</td><td>73.08</td><td>46.72</td><td>21.42</td><td>32.94</td><td>12.19</td><td>8.02</td><td>32.39</td></tr></table>

Table 1: Main results at step 200, reported as Avg@k in percent under a shared evaluation protocol, averaged over two training runs. The teacher and the student before distillation are reference evaluations and are not ranked. Bes in bold, second underlined; Macro weights the six benchmarks equally.

Algorithm 1 AP-OPD token weights with the de  
fault z-gate   
Input: $r _ { 1 : T } .$ , block size B, threshold z   
Output: token weights $w _ { 1 : T }$   
1: ${ d _ { t } } \gets \mathrm { s g } ( [ r _ { t } ] _ { + } ) , w _ { t } \gets 1$ for all t ▷ Eq. (3)   
2: Split the tokens into blocks $\boldsymbol { B } _ { 1 : M }$   
3: Compute $D _ { i }$ for each block and $s _ { i } ^ { 2 }$ for each   
block with $n _ { i } \geq 2$ ▷ Eq. (4)   
4: for $i = 1$ to $M - 1$ with $n _ { i } , n _ { i + 1 } \geq 2$ do   
5: Compute $\Delta _ { i } = D _ { i + 1 } - D _ { i }$ and $\hat { \sigma } _ { i } \triangleright  { \mathrm { E q . } }$ . (6)   
6: if $\Delta _ { i } > z \hat { \sigma } _ { i }$ then   
7: $w _ { t } \gets D _ { i } / D _ { i + 1 }$ for $t \in B _ { i + 1 }$ ▷ Eq. (7)   
8: end if   
9: end for   
10: return $w _ { 1 : T }$

Qwen3-4B-Instruct-2507 is distilled into Qwen3- 1.7B on the code subset of Eurus-2-RL-Data (Cui et al., 2026), and each method trains for 100 steps. Within each domain, every method starts from the same student checkpoint. In both domains, our runs use 128 prompts per step, one rollout per prompt of at most 8,192 tokens, and AdamW with a constant learning rate of $1 0 ^ { - 6 }$ . AP-OPD uses B = 64 and z = 2 in both domains, fixed before training. Details are shown in Appendix A.

Baselines. We compare with vanilla token-level OPD (Fu et al., 2026) and two supervision reweighting methods. IW-OPD uses accumulated absolute log-ratios, normalized within each response, to assign decreasing prefix weights (Xie et al., 2026b). FiRe-OPD filters trajectories and reweights tokens by teacher and student confidence (Li et al., 2026b). Both use their authors’ code and default optimization settings, and share the student, teacher, training prompts, number of steps, and evaluation with our runs. On code generation, we compare with vanilla OPD and IW-OPD.

Evaluation. For mathematical reasoning, we report Avg@k, the accuracy averaged over k sampled responses per problem, on MATH500 (Hendrycks et al., 2021; Lightman et al., 2024), AMC23, Minerva (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), AIME24, and AIME25. Answers are graded with the rule-based math checker of verl (Sheng et al., 2025). For code generation, we report Avg@4 on HumanEval+, MBPP+ (Liu et al., 2023), and LiveCodeBench v6 (Jain et al., 2025), running each program against the benchmark’s tests. All models share one decoding setup. Mathematical results average two training runs that differ in data order. Appendix A gives the full protocol, and Appendix C reports each run.

## 5.2 Main Results

Mathematical reasoning. Table 1 reports AP-OPD with each of its two jump detectors, the z-gate of Section 2.2 and the top-α ranking of Eq. (9). Both detectors improve the macro average over vanilla OPD, by 0.80 points for top-α and 0.57 for the z-gate. The improvement holds in each training run (Appendix C). In the two-run mean, top-α improves on every benchmark. Both detectors also obtain higher macro averages than IW-OPD and FiRe-OPD in this mean. FiRe-OPD, the stronger baseline, improves on vanilla OPD by less than either detector. Computing its token weights requires the full-vocabulary entropies of the teacher and the student. AP-OPD uses only the sampled-token log-ratios that OPD already computes.

Choice of detector. The selected boundary sets of the two detectors have a Jaccard overlap of about one third on the analysis rollouts. Since both improve over vanilla OPD, the gain is not tied to one detector. We keep the z-gate as the default as it needs no selection ratio and leaves responses without a clear rise untouched.

<table><tr><td>Method</td><td>HE+</td><td>MBPP+</td><td>LCB</td><td>Macro</td></tr><tr><td>Teacher</td><td>83.69</td><td>72.95</td><td>32.29</td><td>62.97</td></tr><tr><td>Student</td><td>61.59</td><td>56.81</td><td>14.57</td><td>44.32</td></tr><tr><td>Vanilla OPD</td><td>70.43</td><td>63.89</td><td>23.57</td><td>52.63</td></tr><tr><td>IW-OPD</td><td>70.88</td><td>65.28</td><td>25.00</td><td>53.72</td></tr><tr><td>AP-OPD</td><td>73.17</td><td>65.15</td><td>25.71</td><td>54.68</td></tr></table>

Table 2: Code generation uses Avg@4 (%). Qwen3- 4B-Instruct-2507 (Teacher) and undistilled Qwen3-1.7B (Student) are unranked references. HE+ is HumanEval+; LCB is LiveCodeBench v6. Macro is the unweighted mean; best trained scores are bold.

Code generation. Code generation tests AP-OPD with a different student, teacher, and training data. The student is already post-trained. The teacher is a general instruction model. With the default z-gate, AP-OPD obtains the highest macro average, 2.05 points above vanilla OPD and 0.96 above IW-OPD (Table 2). It improves over vanilla OPD on all three benchmarks. It leads on HumanEval+ and on LiveCodeBench, the hardest benchmark, and is close to IW-OPD on MBPP+. The gain over vanilla OPD is larger than on mathematical reasoning. Across both domains, downweighting supervision after detected jumps improves over vanilla OPD. Section 5.3 compares AP-OPD with magnitude-cap to test whether removing a similar amount of penalty is sufficient to obtain the observed gains.

## 5.3 Ablation Studies

Comparison with magnitude-cap. Magnitudecap selects the ⌈αC⌉ post-boundary blocks with the largest $D _ { i + 1 }$ and assigns weight min( $1 , \bar { D } / D _ { i + 1 } )$ where D<sup>¯</sup> is the response’s mean penalty. We measure the removed fraction as $q \ = \ \sum _ { t } ( 1 \ -$ $w _ { t } ) d _ { t } / \sum _ { t } d _ { t }$ , averaged equally over responses with nonzero penalty mass. Magnitude-cap removes 13.75%, between the two AP-OPD detectors, but brings no gain and trails both by about one point (Table 3a; paired intervals in Appendix C). An earlier experiment also gains from a fixed postjump weight of 0.3, whereas pre-jump upweighting brings no gain (Appendix D). The tested gains thus occur with two detectors and two post-jump weights; a similar removal amount alone is insufficient. These rule comparisons do not isolate selection, reference, and scaling depth.

Block size. The block-size ablation in Table 3b shows improvements over vanilla OPD and magnitude-cap for all three block sizes. Larger blocks select fewer jumps but weight longer spans, so the removed mass q stays within 9.5–11.8%. The default B = 64 gives the largest gain. Shorter blocks may make block means noisier, while longer blocks may average over local increases.

<table><tr><td rowspan="2">Variant</td><td colspan="2">Offline</td><td colspan="2">Trained</td></tr><tr><td>Cov. Unsel.</td><td>q</td><td>Macro</td><td>∆</td></tr><tr><td colspan="5">(a) Comparison with magnitude-cap</td></tr><tr><td rowspan="2">Vanilla OPD Magnitude-cap</td><td>0.00</td><td>13.75</td><td>31.55</td><td></td></tr><tr><td>7.17 7.19 0.00</td><td>14.63</td><td>31.40 32.50</td><td>-0.15 +0.95</td></tr><tr><td colspan="2">AP-OPD (top-α) AP-OPD (z-gate) 6.49</td><td>28.71 11.60</td><td>32.50</td><td>+0.95</td></tr><tr><td colspan="5">(b) Block size, z-gate</td></tr><tr><td>B = 32</td><td>4.23</td><td>24.22 9.51</td><td>32.06</td><td>+0.50</td></tr><tr><td>B = 64</td><td>6.49 28.71</td><td>11.60</td><td>32.50</td><td>+0.95</td></tr><tr><td>B = 128</td><td>8.60 36.72</td><td>11.78</td><td>31.96</td><td>+0.40</td></tr></table>

Table 3: Ablations, same number of training steps and evaluation protocol as Table 1, first run only. Macro: macro Avg@k; ∆: difference from vanilla OPD. Offline statistics on the analysis rollouts are reported in percent: Cov., response-mean share of tokens in selected blocks; Unsel., share of responses with no selected boundary; q, response-mean fraction of penalty mass removed.

## 5.4 Analysis

Relation to token-level clipping. The rise is not carried by a few extreme tokens, so clipping each token’s log-ratio, as REOPOLD does (Ko et al., 2026), leaves it largely in place. At a clipping bound that removes as much penalty mass as AP-OPD on the analysis set, jumps are still detected and about 90% of the held-out rise remains; RE-OPOLD’s default bound removes more than twice as much and still keeps about 70%. The two rules also remove different supervision. Appendix E reports that only 14.5% of AP-OPD’s removal lies above the bound, close to the 12.5% expected if its removal were unrelated to token size.

Training statistics. During training, AP-OPD marks 6.8% of boundaries as jumps and leaves 24.2% of responses unweighted. Gradient clipping fires less often than in vanilla OPD. The training statistics in Appendix C show that response length and student entropy stay close to vanilla OPD. The mean token weight is 0.945. Magnitude-cap has a similar mean weight, 0.961, but brings no gain. The removal amount alone therefore does not explain the performance difference.

## 6 Related Work

On-policy distillation. Unlike distillation from fixed or teacher-generated data (Hinton et al., 2015; Kim and Rush, 2016; Hsieh et al., 2023; Wang et al., 2021; Xie et al., 2026a), OPD scores the student’s own rollouts with the teacher (Agarwal et al., 2024). GKD supports on-policy learning with alternative loss functions (Agarwal et al., 2024). MiniLLM minimizes reverse KL with policy gradients to prevent overestimating lowprobability regions of the teacher distribution (Gu et al., 2024). DistiLLM uses an adaptive off-policy approach to reuse student-generated outputs (Ko et al., 2024). Recent work extends OPD to reasoning post-training (Zhu et al., 2026; Zhang et al., 2026a; Ko et al., 2026; Li et al., 2026a) and studies its supervision and optimization (Song and Zheng, 2026; Jang et al., 2026; Jin et al., 2026; Yang et al., 2026a; Zhang et al., 2026b; Wang et al., 2026). RevisitingOPD documents failure modes of sampledtoken comparison, including unreliable guidance on student-drifted prefixes (Fu et al., 2026).

Reweighting, pruning, and down-weighting in OPD. TIP selects tokens by student uncertainty and teacher–student divergence (Xu et al., 2026). FiRe-OPD filters trajectories and softly reweights tokens by teacher and student confidence (Li et al., 2026b). IW-OPD uses accumulated absolute logratios, normalized within each response, to assign decreasing prefix weights (Xie et al., 2026b). Prune-OPD decays the supervision of a suffix once teacher–student support mismatch accumulates (Yang et al., 2026b). REOPOLD clips token rewards at a fixed bound (Ko et al., 2026). Rock Tokens studies tokens whose distillation loss persists over training (Jiang et al., 2026), Blockwise Gating weights 64-token blocks by student policy drift (Zheng and Jiang, 2026), and entropy-aware OPD adds a forward-KL term where teacher entropy is high (Jin et al., 2026). TrOPD gates the penalty by the teacher–student probability ratio (Xing et al., 2026). AP-OPD instead uses changes between consecutive blocks in the same response and scales post-jump supervision using the preceding block as a local reference. Unlike confidence-based weighting, its rule uses penalty changes without measuring teacher confidence.

Temporal information. Return-to-go OPD propagates later log-ratios to earlier positions, at the cost of higher gradient variance (Gu et al., 2024;

Fu et al., 2026). TOPD and AC-OPD obtain nearfuture evidence by generating continuations from selected prefixes (Jiang and Ferraro, 2026; Zhou et al., 2026). LGR evaluates the student’s top-K candidate tokens by the one-step-ahead teacher confidence they induce and adds a group-normalized reward (Liu et al., 2026). AP-OPD uses only changes realized along the completed rollout and needs no additional generation or forward pass.

Choice of baselines. We compare with IW-OPD and FiRe-OPD as supervision reweighting baselines on student-generated rollouts; FiRe-OPD also filters trajectories. Other methods introduce changes to objectives, rollout generation, or token selection. TrOPD and entropy-aware OPD add a forward-KL term, and TrOPD also uses teachergenerated prefixes. LGR, TOPD, and AC-OPD add teacher passes or continuations. Prune-OPD truncates rollouts, and TIP keeps a subset of tokens to reduce training memory. Rock Tokens selects token types by their loss over training. Blockwise Gating gates on the drift between the rollout and the current student, which is zero when each batch is used for one update, as in our setting. REOPOLD also adds stage-dependent token masks, so we compare only its token-level clipping, directly on our rollouts (§5.4). Magnitude-cap selects blocks by their block-averaged penalties and removes a similar amount of penalty mass, trained with the same configuration as AP-OPD (§5.3).

## 7 Conclusion

We studied local penalty changes in on-policy distillation, where the current penalty value does not distinguish sustained high penalties from abrupt increases along a response. Comparing consecutive token blocks reveals penalty jumps that persist on held-out tokens. Across these jumps, penalties rise while teacher confidence decreases, whereas boundaries into high-penalty blocks do not exhibit comparable confidence changes. This penalty–confidence tension shows that local changes provide information beyond the current penalty value. Motivated by this observation, we proposed AP-OPD, which uniformly scales supervision coefficients in each postjump block using the preceding block as a local reference. The weighted average penalty matches the preceding block’s original mean, setting the scaling factor without an additional strength constant. AP-OPD improves performance on mathematical reasoning and code generation.

## Limitations

We evaluate AP-OPD on mathematical reasoning and code generation, and have not yet applied it to other settings such as agentic or multi-turn tasks. Our diagnostic evidence for penalty–confidence tension is observational and does not establish that decreased teacher confidence causes the gains from pullback. AP-OPD also leaves room for improvement in how it forms blocks. It uses a fixed block length, and adapting the block size to each response or aligning blocks with reasoning steps is a natural extension. Finally, AP-OPD responds only to abrupt rises, so it does not address reward-hackinglike behavior in OPD such as repetitive looping, where the penalty fades rather than jumps.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Pi otr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. 2024. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pages 21246–21263.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. 2015. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Processing Systems, volume 28, pages 1171–1179. Curran Associates, Inc.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, and 6 others. 2026. Process reinforcement through implicit rewards. Transactions on Machine Learning Research.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. 2026. Revisiting on-policy distillation: Empirical failure modes and simple fixes. In Third Conference on Language Modeling.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2024. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pages 32694–32717.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, Bangkok, Thailand. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1. Curran Associates, Inc.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. Preprint, arXiv:1503.02531.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8003–8017, Toronto, Canada. Association for Computational Linguistics.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. Live-CodeBench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, volume 2025, pages 58791–58831.

Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggyu Lim, and Taesup Kim. 2026. Stable on-policy distillation through adaptive target reformulation. In Findings of the Association for Computational Linguistics: ACL 2026, pages 42217–42227, San Diego, California, United States. Association for Computational Linguistics.

Yuxuan Jiang and Francis Ferraro. 2026. Bridging reasoning trajectories in on-policy distillation via nearfuture guidance. Preprint, arXiv:2606.00305.

Yuxuan Jiang, Runchao Li, Shubhashis Roy Dipta, Dawei Li, and Zhao Yang. 2026. Cornerstones or stumbling blocks? deciphering the rock tokens in on-policy distillation. Preprint, arXiv:2605.09253.

Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. 2026. Entropy-aware on-policy distillation of language models. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 53926–53943. PMLR.

Yoon Kim and Alexander M. Rush. 2016. Sequencelevel knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, Austin, Texas. Association for Computational Linguistics.

Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. 2026. Scaling reasoning efficiently via relaxed on-policy distillation. Preprint, arXiv:2603.11137.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. 2024. DistiLLM: Towards streamlined distillation for large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 24872–24895. PMLR.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models. In Advances in Neural Information Processing Systems, volume 35, pages 3843–3857. Curran Associates, Inc.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. 2026a. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. Preprint, arXiv:2604.13016.

Yuying Li, Leqi Zheng, Yongzi Yu, Wenrui Zhou, Xuchang Zhong, Xing Hu, Jing Jin, Hangjie Yuan, and Tao Feng. 2026b. Filter, then reweight: Rethinking optimization granularity in on-policy distillation. Preprint, arXiv:2606.02684.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your code generated by Chat-GPT really correct? rigorous evaluation of large language models for code generation. In Advances in Neural Information Processing Systems, volume 36, pages 21558–21572.

Yanjiang Liu, Jie Lou, Xinyan Guan, Yuqiu Ji, Hongyu Lin, Ben He, Xianpei Han, Le Sun, Xing Yu, and Yaojie Lu. 2026. Your teacher can’t help you here: Combating supervision fidelity decay in on-policy distillation. Preprint, arXiv:2605.30833.

Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2016. Sequence level training with recurrent neural networks. In The Fourth International Conference on Learning Representations.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, pages 1279–1297, Rotterdam, Netherlands. Association for Computing Machinery.

Mingyang Song and Mao Zheng. 2026. A survey of on-policy distillation for large language models. Preprint, arXiv:2604.00626.

Richard S. Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour. 1999. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems, volume 12, pages 1057–1063. MIT Press.

Fusheng Wang, Jianhao Yan, Fandong Meng, and Jie Zhou. 2021. Selective knowledge distillation for neural machine translation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6456–6466, Online. Association for Computational Linguistics.

Jiaqi Wang, Wenhao Zhang, Weijie Shi, Yaliang Li, and James Cheng. 2026. Exploring temporal curriculum in on-policy distillation for multi-turn autonomous agents. In Third Conference on Language Modeling.

Ronald J. Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4):229–256.

Xurong Xie, Zhucun Xue, Jiafu Wu, Jian Li, Yabiao Wang, Xiaobin Hu, Yong Liu, and Jiangning Zhang. 2026a. LLM-oriented token-adaptive knowledge distillation. Proceedings of the AAAI Conference on Artificial Intelligence, 40(40):34070–34078.

Yan Xie, Sijie Zhu, Tiansheng Wen, Bo Chen, and Yifei Wang. 2026b. On the position bias of on-policy distillation. Preprint, arXiv:2606.22600.

Xingrun Xing, Haoqing Wang, Boyan Gao, Ziheng Li, and Yehui Tang. 2026. Trust region on-policy distillation. Preprint, arXiv:2606.01249.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. 2026. TIP: Token importance in on-policy distillation. Preprint, arXiv:2604.14084.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. 2026a. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. Preprint, arXiv:2602.12125.

Zhicheng Yang, Zhijiang Guo, Yifan Song, Minrui Xu, Yongxin Wang, Yiwei Wang, Xiaodan Liang, and Jing Tang. 2026b. Prune-OPD: Efficient and reliable on-policy distillation for long-horizon reasoning. Preprint, arXiv:2605.07804.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, et al. 2025. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pages 113222–113244. Curran Associates, Inc.

Dongxu Zhang, Zhichao Yang, Sepehr Janghorbani, Jun Han, Andrew Ressler, II, Qian Qian, Gregory D. Lyng, Sanjit Singh Batra, and Robert E. Tillman. 2026a. Fast and effective on-policy distillation from reasoning prefixes. In Findings of the Association for Computational Linguistics: ACL 2026, pages 25553–25569, San Diego, California, United States. Association for Computational Linguistics.

Yaocheng Zhang, Jiajun Chai, Yuqian Fu, Songjun Tu, Xiaohan Wang, Wei Lin, Guojun Yin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. 2026b. Are full rollouts necessary for on-policy distillation? Preprint, arXiv:2605.31490.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. 2025. Group sequence policy optimization. Preprint, arXiv:2507.18071.

Liwen Zheng and Haiyun Jiang. 2026. Blockwise policy-drift gating for on-policy distillation. Preprint, arXiv:2606.24084.

Jingang Zhou, Yuyi Zhou, Haiyang Guo, Xukai Wang, Shuai Feng, Sirui Gao, Jian Xu, Qingpei Guo, and Xu-Yao Zhang. 2026. Teacher should think ahead: Adaptive continuations for reliable on-policy distillation. Preprint, arXiv:2609.22254.

Wenhong Zhu, Ruobing Xie, Rui Wang, and Pengfei Liu. 2026. Hybrid policy distillation for LLMs. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 167818–167837. PMLR.

## A Experimental Details

## A.1 Student Training

Table 4 lists the mathematical-reasoning configuration shared by vanilla OPD, both AP-OPD detectors, magnitude-cap, and the block-size runs; the code experiment is described in Appendix A.4.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Student initialization</td><td>Qwen3-1.7B-Base</td></tr><tr><td>Frozen teacher</td><td>Qwen3-8B-RL-Math; see Table 5</td></tr><tr><td>Training prompts</td><td>DAPO-Math-17K; 17,398 prompts 2,048 / 8,192 tokens</td></tr><tr><td>Prompt / response limit</td><td></td></tr><tr><td>Steps Batch</td><td>200, from the base checkpoint</td></tr><tr><td></td><td>128 prompts, one rollout each; one update per step</td></tr><tr><td>Rollout sampling</td><td>temperature 1.0, top-p 1.0, no top-k; thinking disabled</td></tr><tr><td>Optimizer</td><td>AdamW, β = (0.9, 0.999), weight decay 0.01</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 6 }$  , constant, no warmup</td></tr><tr><td>Gradient clipping</td><td>global norm 1.0</td></tr><tr><td>Loss</td><td>token mean; no task reward, KL loss, or entropy bonus</td></tr><tr><td>Data order</td><td>not shuffled; run 1: data seed 42; run 2: data seed 43</td></tr><tr><td>AP-OPD</td><td>B = 64, z = 2; top-α uses α = 5%</td></tr></table>

Table 4: Training configuration shared by our mathematical-reasoning runs.

## A.2 Teacher

The mathematical-reasoning teacher, Qwen3-8B-RL-Math, is Qwen3-8B-Base trained with GSPO (Zheng et al., 2025) on DAPO-Math-17K. Table 5 lists its configuration. The teacher is frozen during distillation.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Initialization</td><td>Qwen3-8B-Base</td></tr><tr><td>Training prompts</td><td>DAPO-Math-17K</td></tr><tr><td>Policy optimization</td><td>GSPO</td></tr><tr><td>Steps</td><td>200</td></tr><tr><td>Prompt / response</td><td>2,048 / 4,096 tokens</td></tr><tr><td>limit Batch</td><td>512 prompts, 16 responses each; mini-batch of 32 prompts</td></tr><tr><td>Clipping range</td><td> $3 \times 1 0 ^ { - 4 }$  lower,  $\bar { 4 } \times \bar { 1 0 } ^ { - 4 }$  upper</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 6 } ,$  10 warmup steps</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Reward</td><td>+1 for a correct answer, —1 otherwise; no overlong shaping</td></tr></table>

Table 5: Training configuration of the frozen mathematical-reasoning teacher.

## A.3 Evaluation

All models use the evaluation configuration in Table 6, with 7,388 responses per model. Prompts are capped at 2,048 tokens and responses at 16,384; sampling uses temperature 0.7, top-p 0.8, top-k 20, and seed 42, with thinking mode disabled. For a benchmark with N problems and correctness $c _ { q , j } \in \{ 0 , 1 \}$ of sample j on problem q,

$$
\operatorname { A v g @ } k = \frac { 1 } { N k } \sum _ { q = 1 } ^ { N } \sum _ { j = 1 } ^ { k } c _ { q , j } .\tag{10}
$$

Answers are graded with the rule-based math checker of verl.

Paired bootstrap. For a contrast between two models, we resample the problems of each benchmark with replacement 10,000 times using seed 42, keep the existing k scores of each resampled problem for both models, and recompute the perbenchmark and macro differences. We report percentile intervals. The intervals use the run-1 checkpoints and cover evaluation randomness, not training variance.

<table><tr><td>Benchmark</td><td>Problems</td><td>k</td><td>Metric</td></tr><tr><td>MATH500</td><td>500</td><td>4</td><td>Avg@4</td></tr><tr><td>AMC23</td><td>40</td><td>16</td><td>Avg@16</td></tr><tr><td>Minerva</td><td>272</td><td>4</td><td>Avg@4</td></tr><tr><td>OlympiadBench</td><td>675</td><td>4</td><td>Avg@4</td></tr><tr><td>AIME24</td><td>30</td><td>16</td><td>Avg@16</td></tr><tr><td>AIME25</td><td>30</td><td>16</td><td>Avg@16</td></tr></table>

Table 6: Evaluation benchmarks and samples per problem.

## A.4 Code Generation

The code experiment distills Qwen3-4B-Instruct-2507 into Qwen3-1.7B, with the configuration in Table 7. Training prompts are drawn from the 25,276 training problems in the code subset of Eurus-2-RL-Data (Cui et al., 2026), competitiveprogramming problems with test cases. All methods train for 100 steps from the same student checkpoint, and we evaluate the step-100 checkpoints. We evaluate on 164 HumanEval+ problems, 378 MBPP+ problems, and 175 LiveCodeBench v6 problems, with four samples per problem, responses capped at 16,384 tokens, temperature 0.7, top-p 0.8, top-k 20, and thinking disabled. Each program is executed in a sandbox against the benchmark’s tests, and Avg@4 is the share of passing samples.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Student</td><td>Qwen3-1.7B</td></tr><tr><td>initialization</td><td></td></tr><tr><td>Frozen teacher</td><td>Qwen3-4B-Instruct-2507</td></tr><tr><td>Training prompts</td><td>Eurus-2-RL-Data code subset; 12,800 prompts</td></tr><tr><td>Prompt / response limit</td><td>2,048 / 8,192 tokens</td></tr><tr><td>Steps</td><td>100, one epoch</td></tr><tr><td>Batch</td><td>128 prompts, one rollout each; one update per step</td></tr><tr><td>Rollout sampling</td><td>temperature 1.0, top-p 1.0, no top-k; thinking disabled</td></tr><tr><td>Optimizer</td><td>AdamW,  $\beta = ( 0 . 9 , 0 . 9 9 9 )$  , weight decay 0.01</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 6 } ,$  constant, no warmup</td></tr><tr><td>Gradient clipping</td><td>global norm 1.0</td></tr><tr><td>Task reward</td><td>none</td></tr><tr><td>AP-OPD</td><td>z-gate,  $B = 6 4 , z = 2$ </td></tr></table>

Table 7: Training configuration of our code-generation runs.

## B Diagnostic Analysis Details

## B.1 Rollouts

The analysis set contains 512 rollouts, one per distinct prompt, saved by the vanilla OPD run at steps 70, 80, 90, and 100, with 128 rollouts per step. For replication we use two further training runs with rollouts saved every 10 steps over 300 steps, with 3,840 rollouts per run: a vanilla OPD run and a run trained with the paired rule of Appendix D. All statistics are computed offline and do not affect training.

## B.2 Detector Statistics

Table 8 compares jump detectors and other selection rules on the analysis set. The two detectors select overlapping but different boundaries, with a Jaccard index of 0.33.

## B.3 Split-Half Protocol

Profiles use the window [−2B, 2B) around each boundary, with offset 0 at the first token of the following block, and never cross a mask break or the response boundary. Controls come from the same response and the same decile of relative position, defined as the boundary offset divided by valid length. Within each control type, boundaries are drawn without replacement; the high-magnitude boundary is the candidate with the highest $D _ { i + 1 }$ For split-half measurement, detection uses only the odd-indexed tokens of each 64-token block, including the token counts in the gate, and controls are also selected on odd tokens; all quantities are then measured on even tokens. The reverse direction swaps the roles, and the two directions are averaged with shared bootstrap weights. Estimates are averaged within each response, then over prompts; intervals come from 10,000 prompt-clustered resamples using seed 42. The held-out rise $U _ { a : b }$ is the measured mean of $d _ { t }$ over offsets [a, b) minus the measured mean of the block preceding the selected boundary. For a jump, this reference is the pre-jump block; each control uses its own preceding block. On the analysis set, 910 of the 978 z = 2 jumps have a complete window, covering 346 responses; with odd and even detection there are 478 and 453 complete jumps from 247 and 230 responses, and 324 prompts in the union. Each detection half of a block has only 32 tokens, and adjacent tokens remain correlated, so split-half measurement reduces but does not remove selection effects.

## B.4 Teacher Confidence

Of the 512 analysis prompts, 362 contain an eligible z = 2 jump, and 353 of these yield a complete group with a position-matched high-magnitude boundary and a random boundary. 153 groups share no prompt or control prefix with one another within the subset; the full set comprises 80 connected components. The teacher is run causally over the original student history, without re-encoding prefixes or truncating early context, and H and $P _ { \mathrm { m a x } }$ are averaged over the 64 tokens of each block. Intervals resample the 80 components for all groups and resample groups within training steps for the independent subset, using 10,000 resamples with seed 42. Table 11 reports both sets. These distributional changes support the observed penalty–confidence tension for the default $z = 2$ jump set; they are not held-out measurements or a combined confidence score.

<table><tr><td>Rule</td><td>Selected/resp.</td><td>Unselected (%)</td><td>Token cov. (%)</td><td>q (%)</td></tr><tr><td>Top-α (5%)</td><td>1.85</td><td>0.00</td><td>7.19</td><td>14.63</td></tr><tr><td>z = 1.5</td><td>3.93</td><td>8.98</td><td>13.37</td><td>20.04</td></tr><tr><td> $z = 2 \left( \mathrm { A P \mathrm { - } O P D } \right)$ </td><td>1.91</td><td>28.71</td><td>6.49</td><td>11.60</td></tr><tr><td> $z = 3$ </td><td>0.36</td><td>75.78</td><td>1.10</td><td>2.56</td></tr><tr><td> $\mathrm { A l l } \ \Delta _ { i } > 0$ </td><td></td><td></td><td></td><td>36.11</td></tr><tr><td>Magnitude-cap</td><td>1.85</td><td>0.00</td><td>7.17</td><td>13.75</td></tr></table>

Table 8: Selection rules on the analysis rollouts at B = 64: selected boundaries per response, share of responses without a selection, share of tokens in selected blocks, and removed penalty fraction $q .$ Token coverage and q are averaged equally over responses; q uses responses with nonzero penalty mass. Among the z-gate thresholds, only $z = 2$ is used for training; the other z values are reported for reference. Magnitude-cap selections are not penalty jumps.

<table><tr><td>Detect</td><td>– High-mag. bound.</td><td>– Rand. pos.</td><td>— Rand. bound.</td></tr><tr><td>odd</td><td>0.019 [0.006, 0.031]</td><td>0.041 [0.025, 0.058]</td><td>0.135 [0.105, 0.168]</td></tr><tr><td>even</td><td>0.036 [0.016, 0.057]</td><td>0.050 [0.034, 0.066]</td><td>0.145 [0.116, 0.177]</td></tr></table>

Table 9: Held-out rise $U _ { 0 : 6 4 }$ of $z = 2$ jumps minus that of each matched control, for each detection direction on the analysis set at $B = 6 4 ,$ with 95% intervals.
<table><tr><td>Rollouts</td><td> $U _ { 0 : 6 4 }$ </td><td> $U _ { 6 4 : 1 2 8 }$ </td><td>Ratio</td></tr><tr><td>Analysis set</td><td> $0 . 1 9 0 [ 0 . 1 7 2 , 0 . 2 0 9 ]$ </td><td> $0 . 0 9 6 [ 0 . 0 8 1 , 0 . 1 1 1 ]$ </td><td>50.3%</td></tr><tr><td>Vanilla, 300 steps</td><td>0.188 [0.183, 0.194]</td><td>0.101 [0.096, 0.106]</td><td>53.6%</td></tr><tr><td>Paired rule, 300 steps</td><td>0.190 [0.184, 0.196]</td><td>0.103 [0.098, 0.108]</td><td>54.5%</td></tr></table>

Table 10: Held-out rise of the first and second blocks after $z = 2$ jumps and their ratio, on the analysis set and on two further training runs. Both rises use the pre-jump block as the reference and are measured with the split-half protocol, averaging both detection directions. A positive second-block rise relative to this reference is compatible with returning to the level of position-matched random boundaries in Figure 2.
<table><tr><td>Metric</td><td>Control</td><td>All 353 groups</td><td>Independent 153</td></tr><tr><td>H (nats)</td><td>random</td><td>0.300 [0.275, 0.342]</td><td>0.311 [0.263, 0.365]</td></tr><tr><td>H (nats)</td><td>high-magnitude</td><td>0.172 [0.122, 0.265]</td><td>0.261 [0.212, 0.317]</td></tr><tr><td> $P _ { \mathrm { m a x } }$ </td><td>random</td><td>-0.099 [−0.112, −0.091]</td><td> $- 0 . 1 0 2 \left[ - 0 . 1 1 6 , - 0 . 0 8 8 \right]$ </td></tr><tr><td> $P _ { \mathrm { m a x } }$ </td><td>high-magnitude</td><td> $- 0 . 0 6 1 \ [ - 0 . 0 8 5 , - 0 . 0 5 0 ]$ </td><td> $- 0 . 0 8 3 \left[ - 0 . 0 9 8 , - 0 . 0 6 9 \right]$ </td></tr></table>

Table 11: DID of the teacher’s entropy and top-1 probability, computed as the post-minus-pre change at jumps minus that at control boundaries, with 95% intervals.

Log-probability decomposition. Writing $\Delta S$ and ∆T for the post-minus-pre changes in the block means of $\ell _ { t } ^ { S }$ and $\ell _ { t } ^ { T }$ , the change in the mean signed log-ratio is $\Delta S { - } \Delta T$ . This identity does not directly give the change in block-averaged penalty, which averages $[ r _ { t } ] _ { + }$ . In the held-out setting of Figure 3c, the teacher’s drop exceeds that at highmagnitude boundaries by 0.052 [0.028, 0.077] with odd detection and 0.077 [0.043, 0.112] with even detection; the student’s drop also exceeds that of both controls.

## C Supplementary Results

Table 12 lists the per-benchmark scores of all trained models, and Table 13 gives per-benchmark paired intervals for the contrasts among our own runs. The five methods in the main comparison are reported for each run and for their mean. Magnitude-cap and the additional block sizes were trained in run 1 only. The paired intervals and training dynamics also refer to run 1.

## D Alternative Reweighting Rules

AP-OPD uniformly scales supervision in the postjump block using the preceding block’s original block-averaged penalty as a local reference. Two alternatives are natural: compensating by upweighting the pre-jump block and down-weighting the post-jump block by a fixed constant. We evaluate three such rules at step 100 with 8,192-token evaluation responses, so their numbers are not comparable with Table 1. All three select boundaries with the top-α ranking of Eq. (9) with α = 5%. The paired rule raises each selected pre-jump block and lowers its post-jump block by the same amount, with the weights normalized over the response. The two one-sided rules multiply every selected prejump block by 4 or every selected post-jump block by 0.3.

<table><tr><td>Model</td><td>MATH500</td><td>AMC23</td><td>Minerva</td><td>Olympiad</td><td>AIME24</td><td>AIME25</td><td>Macro</td></tr><tr><td>Mean of the two runs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla OPD</td><td>72.90</td><td>43.75</td><td>20.96</td><td>33.63</td><td>12.40</td><td>7.29</td><td>31.82</td></tr><tr><td>IW-OPD</td><td>71.60</td><td>43.28</td><td>22.15</td><td>32.31</td><td>11.35</td><td>7.19</td><td>31.31</td></tr><tr><td>FiRe-OPD</td><td>72.80</td><td>45.39</td><td>21.74</td><td>32.83</td><td>11.88</td><td>8.44</td><td>32.18</td></tr><tr><td>AP-OPD (top-α)</td><td>72.97</td><td>44.77</td><td>22.06</td><td>34.26</td><td>13.02</td><td>8.65</td><td>32.62</td></tr><tr><td>AP-OPD (z-gate, B = 64)</td><td>73.08</td><td>46.72</td><td>21.42</td><td>32.94</td><td>12.19</td><td>8.02</td><td>32.39</td></tr><tr><td>Run 1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla OPD</td><td>73.50</td><td>43.75</td><td>20.22</td><td>33.30</td><td>12.08</td><td>6.46</td><td>31.55</td></tr><tr><td>IW-OPD</td><td>73.05</td><td>43.91</td><td>21.60</td><td>31.78</td><td>11.46</td><td>8.75</td><td>31.76</td></tr><tr><td>FiRe-OPD</td><td>73.00</td><td>44.22</td><td>21.42</td><td>33.00</td><td>11.46</td><td>8.12</td><td>31.87</td></tr><tr><td>AP-OPD (top-α)</td><td>73.05</td><td>45.94</td><td>21.42</td><td>34.41</td><td>13.12</td><td>7.08</td><td>32.50</td></tr><tr><td>AP-OPD (z-gate, B = 64)</td><td>72.75</td><td>46.56</td><td>21.32</td><td>33.33</td><td>12.71</td><td>8.33</td><td>32.50</td></tr><tr><td>Run 2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla OPD</td><td>72.30</td><td>43.75</td><td>21.69</td><td>33.96</td><td>12.71</td><td>8.12</td><td>32.09</td></tr><tr><td>IW-OPD</td><td>70.15</td><td>42.66</td><td>22.70</td><td>32.85</td><td>11.25</td><td>5.62</td><td>30.87</td></tr><tr><td>FiRe-OPD</td><td>72.60</td><td>46.56</td><td>22.06</td><td>32.67</td><td>12.29</td><td>8.75</td><td>32.49</td></tr><tr><td>AP-OPD (top-α)</td><td>72.90</td><td>43.59</td><td>22.70</td><td>34.11</td><td>12.92</td><td>10.21</td><td>32.74</td></tr><tr><td>AP-OPD (z-gate, B = 64)</td><td>73.40</td><td>46.88</td><td>21.51</td><td>32.56</td><td>11.67</td><td>7.71</td><td>32.29</td></tr><tr><td>Additional ablations</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magnitude-cap</td><td>72.55</td><td>44.53</td><td>21.88</td><td>32.56</td><td>9.58</td><td>7.29</td><td>31.40</td></tr><tr><td>AP-OPD (z-gate, B = 32)</td><td>71.30</td><td>45.16</td><td>22.61</td><td>32.85</td><td>12.29</td><td>8.13</td><td>32.06</td></tr><tr><td>AP-OPD (z-gate, B = 128)</td><td>73.15</td><td>42.81</td><td>22.79</td><td>33.81</td><td>11.25</td><td>7.92</td><td>31.96</td></tr></table>

Table 12: Per-benchmark Avg@k in percent at step 200 under the same evaluation protocol. Mean: arithmetic mean of the two runs, calculated from unrounded scores. Macro: equal-weight mean over the six benchmarks; the additional ablations use run 1 only. Best in bold, second underlined within each block; ties at the reported precision share the same marking.
<table><tr><td>Contrast</td><td>MATH500</td><td>AMC23</td><td>Minerva</td><td>Olympiad</td><td>AIME24</td><td>AIME25</td><td>Macro</td></tr><tr><td>z-gate — Vanilla</td><td>-0.75</td><td>+2.81</td><td>+1.10</td><td>+0.04</td><td>+0.63</td><td>+1.88</td><td>+0.95 [0.02, 1.90]</td></tr><tr><td>Top-α – Vanilla</td><td>[-2.40, 0.80] -0.45 [-2.05, 1.10]</td><td>[-0.47, 6.25] +2.19 [-0.94,5.47]</td><td>[-1.19, 3.31] +1.19 [-0.92, 3.31]</td><td>[-1.44, 1.48] +1.11 [-0.33, 2.52]</td><td>[-1.04, 2.29] +1.04 [-1.67, 4.17]</td><td>[-0.42, 5.00] +0.63 [-1.25, 2.71]</td><td>+0.95 [0.01, 1.90]</td></tr><tr><td>Magnitude-cap — Vanilla</td><td>-0.95 [-2.60, 0.65]</td><td>+0.78 [-1.56, 3.28]</td><td>+1.65 [-0.46, 3.86]</td><td>-0.74 [-2.22, 0.70]</td><td>-2.50 [-5.00, -0.21]</td><td>+0.83 [-1.46, 3.33]</td><td>-0.15 [-0.99, 0.70]</td></tr><tr><td>z-gate — Magnitude-cap</td><td>+0.20 [–1.55, 1.95]</td><td>+2.03 [-0.31, 4.69]</td><td>-0.55 [-2.57, 1.38]</td><td>+0.78 [−0.81, 2.41]</td><td>+3.13 [0.42, 6.46]</td><td>+1.04 [−1.04, 4.17]</td><td>+1.10 [0.20, 2.07]</td></tr><tr><td>Top-α — Magnitude-cap</td><td>+0.50 [-1.15, 2.20]</td><td>+1.41 [-1.72, 4.22]</td><td>-0.46</td><td>+1.85</td><td>+3.54 [0.21, 7.50]</td><td>-0.21</td><td>+1.11 [0.13, 2.12]</td></tr><tr><td>z-gate - Top-α</td><td>-0.30 [-2.00, 1.35]</td><td>+0.63 [-2.19, 3.59]</td><td>[-2.67, 1.75] -0.09 [-2.30, 2.02]</td><td>[0.30, 3.44] -1.07 [-2.56, 0.41]</td><td>-0.42 [-3.75, 2.50]</td><td>[−1.67, 1.04] +1.25 [−1.04, 4.38]</td><td>-0.00 [-1.01, 1.01]</td></tr></table>

Table 13: Paired differences in percentage points among our run-1 models, with 95% paired bootstrap intervals.

Table 15 shows that upweighting the pre-jump block brings no gain. The pre-jump ×4 rule triggers gradient clipping in 88 of 100 steps. The paired rule has a macro average close to vanilla OPD. Down-weighting the post-jump block by a constant 0.3, fixed before training, improves over vanilla OPD and has the best macro average. AP-OPD uses the preceding block’s original mean to determine the post-jump weight, $D _ { i } / D _ { i + 1 }$ , without an additional scaling constant. The fixed-weight result shows that a gain can also occur under another post-jump weighting rule; it does not establish that the ratio is superior, since the protocols differ.

## E Block-Level versus Token-Level Down-Weighting

This appendix gives the protocol and additional numbers for §5.4.

Setup. REOPOLD bounds each token’s penalty from above (Ko et al., 2026); in our notation it replaces $r _ { t }$ by min(r<sub>t</sub>, κ) with

$$
\kappa = { \frac { - \log \lambda } { 1 - \lambda } } .\tag{11}
$$

We use two bounds. In the removal-matched setting, λ = 0.051 gives κ = 3.131, which removes the same 11.6% of penalty mass as AP-OPD on the analysis set. Under REOPOLD’s default, λ = 0.3 gives $\kappa = 1 . 7 2 0$ , which removes 27.0%. On the two 300-step runs of Appendix B we keep both bounds fixed rather than recalibrating them, so the matched bound removes somewhat more than AP-OPD there: 13.2% and 13.4%, compared with 10.7% and 10.6% for AP-OPD, respectively. After clipping, we recompute the block profile, rerun the $z = 2$ detector, and repeat the split-half measurement of Appendix B.3.

<table><tr><td>Model</td><td>Clipped steps (1-50 / 51–200)</td><td>Resp. length (first / 1ast 50)</td><td>Student entropy (first / last 50)</td><td>Weighted bound. (%)</td><td>Unweighted resp. (%)</td><td>Mean weight</td></tr><tr><td>Vanilla OPD</td><td>43 (36 / 7)</td><td>1464 / 2437</td><td>0.764 / 0.168</td><td></td><td></td><td>1</td></tr><tr><td>AP-OPD (top-α)</td><td>30 (30 / 0)</td><td>1441 / 2432</td><td>0.757 / 0.174</td><td>6.52</td><td>1.57</td><td>0.952</td></tr><tr><td>AP-OPD (z-gate)</td><td>33 (33 / 0)</td><td>1485 / 2408</td><td>0.816 / 0.173</td><td>6.79</td><td>24.22</td><td>0.945</td></tr><tr><td>Magnitude-cap</td><td>32 (31 / 1)</td><td>1440 / 2434</td><td>0.726 / 0.175</td><td>6.52</td><td>1.33</td><td>0.961</td></tr></table>

Table 14: Run-1 training dynamics over 200 steps, reported as means of logged values. Clipped steps have preclipping gradient norm above 1.
<table><tr><td>Rule</td><td>MATH500</td><td>AMC23</td><td>Minerva</td><td>Olympiad</td><td>AIME24</td><td>AIME25</td><td>Macro</td><td>Clipped</td><td>Mean w</td></tr><tr><td>Vanilla OPD</td><td>69.90</td><td>43.12</td><td>20.31</td><td>30.67</td><td>11.46</td><td>6.25</td><td>30.29</td><td></td><td>1</td></tr><tr><td>Paired  $( + \mathrm { p r e } , - \mathrm { p o s t } )$ </td><td>70.00</td><td>41.88</td><td>20.77</td><td>30.33</td><td>11.67</td><td>7.71</td><td>30.39</td><td>32 / 100</td><td>1.006</td></tr><tr><td>Pre-jump ×4</td><td>69.15</td><td>41.88</td><td>20.50</td><td>31.26</td><td>12.29</td><td>6.25</td><td>30.22</td><td>88 / 100</td><td>1.200</td></tr><tr><td>Post-jump ×0.3</td><td>70.25</td><td>44.38</td><td>21.23</td><td>31.11</td><td>12.29</td><td>6.88</td><td>31.02</td><td>30 / 100</td><td>0.955</td></tr></table>

Table 15: Alternative reweighting rules at step 100, reported as Avg@k in percent from one training run with 8K evaluation responses. Clipped: steps whose pre-clipping gradient norm exceeds 1; Mean w: mean token weight.
<table><tr><td>Rollouts</td><td>Bound</td><td>Removed</td><td>Jumps kept</td><td>Rise kept</td><td>Overlap</td><td>Expected</td></tr><tr><td>Analysis set</td><td>matched</td><td>11.6</td><td>99.0</td><td>89.8 [86.1, 93.8]</td><td>14.5 [13.1, 15.9]</td><td>12.5</td></tr><tr><td>Analysis set</td><td>default</td><td>27.0</td><td>96.3</td><td>71.1 [67.0, 75.8]</td><td>32.3 [30.9, 33.6]</td><td>28.6</td></tr><tr><td>Vanilla, 300 steps</td><td>matched</td><td>13.2</td><td>98.9</td><td>86.1 [84.8, 87.4]</td><td>13.3 [13.1, 13.6]</td><td>13.1</td></tr><tr><td>Vanilla, 300 steps</td><td>default</td><td>30.1</td><td>95.9</td><td>68.0 [66.7, 69.4]</td><td>31.9 [31.6, 32.2]</td><td>30.2</td></tr><tr><td>Paired rule, 300 steps</td><td>matched</td><td>13.4</td><td>98.7</td><td>84.2 [82.9, 85.5]</td><td>13.7 [13.5, 14.0]</td><td>13.1</td></tr><tr><td>Paired rule, 300 steps</td><td>default</td><td>30.3</td><td>95.5</td><td>66.4 [65.0, 67.8]</td><td>32.5 [32.1, 32.8]</td><td>30.3</td></tr></table>

Table 16: Token-level clipping on the analysis set and the two 300-step runs at B = 64 and $z = 2 .$ Values are percentages; intervals are 95%. Matched: $\kappa = 3 . 1 3 1$ , which removes as much penalty mass as AP-OPD on the analysis set; default: $\mathrm { R E O P O L D ^ { \prime } s } \ \kappa = 1 . 7 2 0$ . Removed: penalty mass removed by clipping, averaged over responses. Jumps kept: original jumps detected again after clipping. Rise kept: held-out rise $U _ { 0 : 6 4 }$ after clipping relative to no clipping. Overlap: share of AP-OPD’s removal that lies above the bound. Expected: share of all penalty mass above the bound, the overlap expected if AP-OPD’s removal were unrelated to token size. Overlap and Expected pool mass over tokens.

Rise after clipping. Table 16 shows that at both bounds and on all three sets of rollouts, nearly all jumps are detected again and most of the held-out rise remains. At the matched bound on the analysis set, the held-out rise still exceeds each of the three matched controls in both detection directions under the paired bootstrap of Appendix B.3, as it does at the default bound. Clipping also changes block variances and creates new jumps. The detector finds 1,124 jumps at the matched bound and 1,296 at the default bound, compared with 978 without clipping. The retained rise therefore compares two detected sets rather than following each original jump.

Overlap of removed mass. With AP-OPD’s selections and weights fixed on the unclipped signal, we split each removed amount $( 1 - w _ { t } ) d _ { t }$ into the part above κ, which clipping would also remove, and the part below it. If AP-OPD’s removal were unrelated to token size, the part above κ would equal the share of all penalty mass that lies above κ. Table 16 shows that the overlap stays close to this reference at both bounds and on all three sets of rollouts. On the two 300-step runs at the matched bound, it nearly coincides with the reference. The two rules therefore remove largely different supervision.