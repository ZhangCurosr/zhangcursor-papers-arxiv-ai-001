# DIVERGENCE-ROUTED ON-POLICYDISTILLATION FOR LANGUAGE MODELS

Tiantian Peng<sup>1</sup> Guannan Lv<sup>3</sup> Hongyang Li<sup>1</sup> Qiuhe Hong<sup>1</sup> Hang Cheng<sup>2</sup>   
Hongjian Dou<sup>3</sup> Xiangyu Wu<sup>3</sup> Yuyang Liu<sup>1∗</sup>   
<sup>1</sup>Peking University   
<sup>2</sup>Tsinghua University   
<sup>3</sup>Kuaishou Technology

## ABSTRACT

On-policy distillation (OPD) transfers knowledge from a teacher model to a student along student-generated trajectories, typically by minimizing the reverse Kullback–Leibler (RKL). However, RKL can provide weak corrective signals for teacher-supported tokens to which the student assigns very low probability. This raises two questions: which positions are most consequential for the behavioral gap between the student and teacher, and what additional supervision is needed at these positions? Through cross-sampling, we find that divergence-based routing more effectively pinpoints positions contributing to the teacher–student performance gap than entropy-based. Further analysis reveals teacher-mode undercoverage at high Jensen–Shannon (JS) divergence positions, where standard OPD increasing the large teacher–student probability gaps. Motivated by this findings, we propose Divergence-Routed On-Policy Distillation (DrOPD). DrOPD retains RKL supervision at all positions, and selectively adds forward Kullback– Leibler (FKL) at positions with the high JS divergence. Experiments across six math benchmarks show that DrOPD improves over standard OPD and entropybased routing. DrOPD also reduces the severe under-coverage cases by 37.9% relative to OPD, supporting targeted FKL as an effective complement to OPD.

## 1 INTRODUCTION

On-policy distillation (OPD) provides an effective way to transfer knowledge from a stronger teacher to a student by providing supervision along student-generated trajectories (Agarwal et al., 2023; Gu et al., 2024; Lu & Lab, 2025), thereby reducing the mismatch between offline distillation supervision and the student’s inference-time state distribution. Standard OPD is commonly implemented by minimizing the reverse Kullback–Leibler (RKL) (Kullback & Leibler, 1951) between the student and teacher next-token distributions at every response position. However, RKL is weighted by the student probability, and can provide weak corrective signals for teacher-supported tokens that already receive very low student probability. As a result, some teacher modes may remain under covered despite improvements in overall alignment.

Recent work has explored selective token-level supervision, often using the student or teacher en tropy to identify important positions (Jin et al., 2026; Zhang et al., 2026; Hou et al., 2026a; Xu et al., 2026; Ko et al., 2026; Liu et al., 2026). Entropy, however, measures uncertainty within a single model, but uncertainty does not necessarily indicate a consequential difference between student and teacher. For example, both models may be uncertain while assigning similar probabilities to the available candidates. This motivates two questions: which positions are most consequential for the teacher–student performance gap, and what additional supervision is needed at these positions?

We use controlled cross-sampling (Meng et al., 2026) to identify routing signals that effectively locate positions contributing to the teacher–student performance gap. The student serves as the default generation policy, while the teacher supplies tokens only at positions selected by a routing criterion. We compare Jensen–Shannon (JS) divergence (Lin, 1991), RKL, and forward Kullback– Leibler (FKL) against student entropy, teacher entropy, and random routing. The results reveal that divergence-based strategies achieve similar intervention efficiency and outperform others. On AMC23, divergence-based strategies reach teacher performance with teacher intervention at fewer than 10% of generated positions. These results identify teacher–student disagreement as an effective signal for locating positions where additional guidance matters most.

We then examine the probability distributions at high-JS positions and find pronounced teachermode under-coverage: on the same contexts, the student assigns substantially lower probabilities than the teacher to the recorded tokens. Standard OPD training further exacerbates severe undercoverage at these positions. This behavior is consistent with the gradient structure of RKL: for a candidate with fixed positive teacher probability, the RKL logit gradient vanishes as the student probability approaches zero, whereas FKL retains a teacher-directed corrective signal. These observations motivate adding FKL supervision at high-disagreement positions to help recover teachersupported tokens that RKL struggles to correct.

Based on this diagnosis, we propose Divergence-Routed On-Policy Distillation (DrOPD). DrOPD retains RKL supervision at all response positions and adds FKL supervision to the fraction $\rho$ of positions with the highest teacher–student JS divergence. This design preserves the standard OPD learning signal while directing complementary mode-covering supervision toward the positions identified by our diagnosis. Experiments across six mathematical reasoning benchmarks show that DrOPD improves mean Avg@8 over standard OPD by 1.84% for the 1.7B–4B student–teacher pairs. It also outperforms entropy-based routing methods, supporting disagreement as a more effective guide for allocating additional supervision. On fixed high-JS contexts, standard OPD increases the fraction of positions with severe probability gaps from 3.83% in the base student to 4.48%. DrOPD reduces this fraction to 2.79%, a relative reduction of 37.9% compared with OPD. These results connect the performance gains to improved recovery from teacher-mode under-coverage.

## Our contributions are threefold:

• Identifying where additional supervision matters. Controlled cross-sampling shows that teacher–student divergence, RKL, FKL and JS divergence, locates behaviorally important positions more effectively than single-model entropy.

• Diagnosing severe teacher-mode under-coverage. Fixed-context analysis reveals severe probability gaps at high-JS positions that become more frequent after standard OPD, motivating complementary FKL supervision through gradient analysis.

• Divergence-Routed On-Policy Distillation. We propose DrOPD, which combines allposition RKL with selectively allocated FKL, improving reasoning performance across multiple settings and and reducing severe under-coverage.

## 2 DIAGNOSING TOKEN IMPORTANCE IN ON-POLICY DISTILLATION

We first use controlled cross-sampling to evaluate signals for identifying positions where teacher intervention improves downstream performance, then examine teacher-mode under-coverage at highdisagreement positions and its connection to KL gradients. Given a prompt x and prefix $y _ { < t }$ , let $p _ { t }$ and $q _ { t }$ denote the student and teacher next-token distributions, respectively, renormalized over the union of their top-16 candidate sets. We compare single-model uncertainty, measured by student entropy $H ( p _ { t } )$ and teacher entropy $H ( q _ { t } )$ , with teacher–student disagreement, measured by RKL $D _ { \mathrm { K L } } ( p _ { t } \Vert q _ { t } )$ , FKL $D _ { \mathrm { K L } } ( q _ { t } \Vert p _ { t } )$ , and JS divergence: $\begin{array} { r } { D _ { \mathrm { J S } } ( p _ { t } , q _ { t } ) = \frac { 1 } { 2 } \tilde { D _ { \mathrm { K L } } } ( p _ { t } \Vert m _ { t } ) + \frac { 1 } { 2 } D _ { \mathrm { K L } } ( \bar { q _ { t } } \Vert m _ { t } ) } \end{array}$ where $m _ { t } = { \textstyle \frac { 1 } { 2 } } ( p _ { t } + q _ { t } )$

## 2.1 CONTROLLED CROSS-SAMPLING

Following Meng et al. (2026), we use controlled cross-sampling to assess how effectively routing signals identify positions that benefit from teacher intervention. At each decoding step, giving a routing score $r _ { t }$ and threshold τ , we define $S _ { t } = \mathbb { I } [ r _ { t } > \tau ]$ and sample

$$
y _ { t } \sim { \left\{ \begin{array} { l l } { q _ { t } , } & { S _ { t } = 1 , } \\ { p _ { t } , } & { S _ { t } = 0 . } \end{array} \right. }\tag{1}
$$

The sampled token is appended to the prefix. We compare routing based on JS divergence, RKL, FKL, student entropy, and teacher entropy, together with a random baseline. For each stategy, we

![](images/63e33fcd8d6008d724c0cdc9e0abc0f580b1df28be39a32aad35c7d5716f41a0.jpg)

![](images/bac1669ee732bdd3d3fcf9e1739713e689e01147e4787d1ba69fc9dbfb5bfc6c.jpg)

![](images/3a750edcf0eeaa9d256ca47e08ab7f9c201fd3bbfe1c8a3e3491efcf49e3ef67.jpg)  
Figure 1: Cross-sampling and teacher-mode under-coverage at high-JS positions. (a) AMC23 accuracy versus teacher-intervention rate; dashed lines indicate standalone Teacher and Base performance. (b) Mean positive probability gap across original JS deciles, computed over recorded tokens with positive gaps. (c) Relative change from Base to OPD in the fraction of positions with gaps at or above each threshold, evaluated on fixed original JS > 0.06 positions.

independently sweep the threshold $\tau ;$ for random routing, we manually change the intervention probability. Each setting is evaluated on 40 AMC23(Li et al., 2024) problems, with eight responses per problem. We measure the actual teacher-intervention ratio as:

$$
\eta = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } S _ { t } ^ { ( i ) } , \qquad N = 3 2 0 ,\tag{2}
$$

where $T _ { i }$ is the length of response i. We compare Avg@8 as a function of $\eta \colon$ higher accuracy at comparable intervention ratios indicates that a routing signal more effectively identifies positions where teacher guidance improves task performance.

## 2.2 DIVERGENCE IDENTIFIES BEHAVIORALLY CRITICAL POSITIONS

Figure 1a shows that divergence-based routing uses teacher intervention more efficiently than others. JS divergence, RKL, and FKL achieve similar performance, reaching near-teacher accuracy with intervention at fewer than 10%. For example, JS routing achieves 60.31% Avg@8 with 9.23% intervention, close to the teacher’s 60.63%. In contrast, student- and teacher-entropy routing achieve only 50.00% and 37.19%, respectively. These results suggest that divergence is an effective signal for locating behaviorally critical positions, while entropy alone is less informative because a model can be uncertain even when the student and teacher make similar predictions.

We use JS divergence for the remaining analysis and the routing mechanism in Section 3 because it is symmetric and bounded. Having identified high-disagreement positions is valuable, we next examine their probability distributions to understand what additional supervision they need.

## 2.3 FROM HIGH DISAGREEMENT TO TEACHER-MODE UNDER-COVERAGE

Measuring under-coverage on fixed contexts. We examine the probability distributions at high JS positions using 320 Base–Teacher cross-sampled responses. To quantify teacher-mode undercoverage, we define the positive teacher–student probability gap as

$$
\delta _ { t } = \big [ q _ { t } ( v ) - p _ { t } ( v ) \big ] _ { + } , \qquad [ a ] _ { + } = \operatorname* { m a x } ( a , 0 ) ,\tag{3}
$$

where v is the recorded token at position t. A larger $\delta _ { t }$ indicates that the student assigns less probability to candidate v than the teacher does.

Figure 1b shows that under-coverage is more pronounced at high-JS positions. For the base student, the mean positive gap increases from 0.014 in the eighth JS decile to 0.406 in the highest. Thus, positions identified as valuable intervention sites also exhibit substantial deficits in the probability assigned to teacher-supported tokens. Standard OPD reduces many probability gaps but exacerbates severe under-coverage. Figure 1c shows the relative change from Base to OPD in the fraction of fixed high-JS positions with $\delta _ { t } \geq \tau$ , where τ is the gap threshold. $\mathrm { { A t } } \tau = 0 . 2$ , this fraction decreases by 29.6%. $\mathrm { A t } \tau = 0 . 8$ , however, it increases from 3.83% to 4.48%, a relative increase of 16.9%. Both comparisons use the same positions with original $\mathrm { J S } > 0 . 0 6$

![](images/b9e2d10e978e74a4d62a14b87e0551970f2ec8e8deea32f6c18177b9a7bf5735.jpg)  
(a) Token-Level Disagreement Scoring

![](images/294ae444fb01c683247654a2ee49428d66f48a30ebc275856c8f666f188baab3.jpg)  
(b) Why add forward supervision

![](images/868c50fe3c6af65a8328150777ac639de1c42d920ed3c64b4517535b7ac2ce35.jpg)  
(c) Divergence-routed OPD  
Figure 2: Overview of DrOPD. (a) We compute teacher–student JS divergence at each rollout position and select the top-20% with the highest disagreement. (b) High-JS positions can exhibit under-coverage of teacher-supported modes, for which RKL provides weak corrective signals. (c) DrOPD retains RKL supervision at all positions and adds FKL supervision at the high-JS positions.

Understanding under-coverage through gradient analysis. The persistence of severe undercoverage is consistent with the gradient structure of RKL. For a student distribution p parameterized by logits $z ( v )$ and a fixed teacher distribution q, the logit gradients are

$$
\frac { \partial D _ { \mathrm { K L } } ( p | | q ) } { \partial z ( v ) } = p ( v ) \left( \log \frac { p ( v ) } { q ( v ) } - D _ { \mathrm { K L } } ( p \| q ) \right) ,\tag{4}
$$

$$
\frac { \partial D _ { \mathrm { K L } } ( q | | p ) } { \partial z ( v ) } = p ( v ) - q ( v ) .\tag{5}
$$

For fixed $q ( v ) > 0 ,$ , the RKL gradient vanishes as $p ( v )  0 .$ , weakening the direct corrective signal for severely under-covered candidates. The FKL gradient instead approaches $- q ( v )$ , so gradient descent continues to increase the student’s logit for these candidates. This complementary gradient behavior motivates adding FKL supervision at high-disagreement positions.

## 3 DIVERGENCE-ROUTED ON-POLICY DISTILLATION

Motivated by the above findings, we propose DrOPD, which retains the RKL throughout studentgenerated trajectories and adds FKL at selected high-JS positions. Figure 2 illustrates the method.

## 3.1 DISAGREEMENT SCORING AND POSITION SELECTION

Set a student policy $\pi _ { \mathrm { o l d } }$ to generate student responses. At position t, let $c _ { t }$ denote the generated prefix. We form the union candidate set from the student’s and teacher’s top-k candidates, and renormalize the two distributions over the union candidate: :

$$
\mathscr { C } _ { t } = \mathrm { T o p K } _ { k } ( p _ { t } ) \cup \mathrm { T o p K } _ { k } ( q _ { t } ) , \quad \widetilde { p } _ { t } ( v ) = \frac { p _ { t } ( v ) } { \sum _ { u \in \mathcal { C } _ { t } } p _ { t } ( u ) } , \quad \widetilde { q } _ { t } ( v ) = \frac { q _ { t } ( v ) } { \sum _ { u \in \mathcal { C } _ { t } } q _ { t } ( u ) } .\tag{6}
$$

This union retains both student-preferred candidates and teacher-supported candidates. We use $k =$ 16 and score disagreement at each position by $d _ { t } = D _ { \mathrm { J S } } ( \widetilde { p } _ { t } , \widetilde { q } _ { t } )$ . Let y denote the collection of

Table 1: Mathematical reasoning results (Avg@8, %). Both panels select and average over all six benchmarks. Results marked with † are reproduced by us.
<table><tr><td>Method</td><td></td><td>AIME24 AIME25</td><td>AMC23 Minerva</td><td></td><td>Olympiad MATH500</td><td></td><td>Mean(6)</td></tr><tr><td colspan="8">Baselines</td></tr><tr><td>1.7B-Base</td><td>0.42</td><td>1.25</td><td>14.38</td><td>6.34</td><td>8.18</td><td>23.88</td><td>9.07</td></tr><tr><td>4B-Base</td><td>10.00</td><td>5.42</td><td>30.00</td><td>6.48</td><td>24.78</td><td>40.38</td><td>19.51</td></tr><tr><td>4B-GRPO</td><td>24.17</td><td>17.50</td><td>60.63</td><td>30.19</td><td>48.53</td><td>83.10</td><td>44.02</td></tr><tr><td>8B-GRPO</td><td>25.83</td><td>21.67</td><td>68.13</td><td>35.29</td><td>51.52</td><td>86.00</td><td>48.07</td></tr><tr><td colspan="8">Student: 1.7B-Base Teacher: 4B-GRPO</td></tr><tr><td>OPD</td><td>12.92</td><td>8.33</td><td>45.31</td><td>21.23</td><td>33.03</td><td>68.63</td><td>31.57</td></tr><tr><td>Stu-entropy</td><td>12.08</td><td>7.08</td><td>47.19</td><td>24.59</td><td>34.55</td><td>70.43</td><td>32.65</td></tr><tr><td>Tch-entropy</td><td>12.92</td><td>7.92</td><td>46.25</td><td>23.44</td><td>34.77</td><td>71.00</td><td>32.72</td></tr><tr><td>EOPD† (2026)</td><td>11.25</td><td>7.50</td><td>45.94</td><td>23.07</td><td>34.72</td><td>69.93</td><td>32.07</td></tr><tr><td>DrOPD</td><td>14.17</td><td>11.67</td><td>46.56</td><td>24.03</td><td>33.88</td><td>70.15</td><td>33.41</td></tr><tr><td colspan="8">Student: 4B-Base Teacher: 8B-GRPO</td></tr><tr><td>OPD</td><td>24.58</td><td>18.33</td><td>61.88</td><td>33.00</td><td>47.92</td><td>82.15</td><td>44.64</td></tr><tr><td>Stu-entropy</td><td>23.33</td><td>19.17</td><td>59.38</td><td>32.86</td><td>47.61</td><td>82.93</td><td>44.21</td></tr><tr><td>Tch-entropy</td><td>20.83</td><td>19.17</td><td>62.19</td><td>33.64</td><td>47.57</td><td>82.53</td><td>44.32</td></tr><tr><td>EOPD† (2026)</td><td>22.08</td><td>18.75</td><td>62.19</td><td>32.67</td><td>48.02</td><td>83.15</td><td>44.48</td></tr><tr><td>DrOPD</td><td>25.00</td><td>19.17</td><td>63.44</td><td>32.54</td><td>48.07</td><td>83.18</td><td>45.23</td></tr></table>

student responses in the current batch, and let $\epsilon _ { \rho } ( y )$ be the $( 1 - \rho )$ -quantile of disagreement scores over its valid response positions. We select high-disagreement positions using

$$
g _ { t } = \mathbb { 1 } [ d _ { t } \geq \epsilon _ { \rho } ( y ) ] .\tag{7}
$$

We set $\rho = 0 . 2$ , allocating additional supervision to the top 20% highest-disagreement positions.

## 3.2 ALL-POSITION RKL WITH ROUTED FKL SUPERVISION

Using $\widetilde { p } _ { t }$ and $\widetilde { q } _ { t }$ , we define the top-k approximate distillation loss at position t as

$$
\ell _ { t } ( \theta ) = \sum _ { v \in \mathcal { C } _ { t } } \widetilde { p } _ { t } ( v ) \log \frac { \widetilde { p } _ { t } ( v ) } { \widetilde { q } _ { t } ( v ) } + g _ { t } \sum _ { v \in \mathcal { C } _ { t } } \widetilde { q } _ { t } ( v ) \log \frac { \widetilde { q } _ { t } ( v ) } { \widetilde { p } _ { t } ( v ) } .\tag{8}
$$

The RKL preserves the standard OPD objective at every response position. When $g _ { t } ~ = ~ 1$ , the additional FKL supplies teacher-weighted supervision to help correct under-covered candidates.

In practice, we optimize a policy-gradient surrogate using the following distillation reward:

$$
R _ { t } ( v ) = - \widetilde p _ { t } ( v ) \log \frac { \widetilde p _ { t } ( v ) } { \widetilde q _ { t } ( v ) } + g _ { t } \widetilde q _ { t } ( v ) , \qquad v \in { \mathcal C } _ { t } .\tag{9}
$$

The first term supplies the standard OPD signal, while the second adds the FKL correction. We use these rewards directly as detached advantages, $A _ { t } ( v ) = \operatorname { s g } [ R _ { t } ( v ) ]$ ], and update the student by maximizing the PPO-style clipped surrogate:

$$
\mathcal { I } ( \theta ) = \frac { 1 } { | \mathcal { B } | } \sum _ { t \in \mathcal { B } } \sum _ { v \in \mathcal { C } _ { t } } \operatorname* { m i n } [ w _ { t } ( v ; \theta ) A _ { t } ( v ) , \operatorname { c l i p } ( w _ { t } ( v ; \theta ) , 1 - \epsilon _ { \mathrm { c l i p } } , 1 + \epsilon _ { \mathrm { c l i p } } ) A _ { t } ( v ) ] .\tag{10}
$$

Here, $w _ { t } ( v ; \theta )$ is the importance-sampling weight, B contains the valid response positions in the batch, and $\epsilon _ { \mathrm { c l i p } }$ is the clipping parameter. The teacher remains frozen during each policy update.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTINGS

Models and training. Our primary pair uses Qwen3-1.7B-Base as the student and Qwen3-4B-Base-GRPO as the teacher. We also evaluate a Qwen3-4B-Base student with a Qwen3-8B-Base-GRPO teacher (Yang et al., 2025; Shao et al., 2024). We use verl(Sheng et al., 2024) as our training framework. Students train on DAPO-Math-17K(Yu et al., 2026a) for one epoch with four studentgenerated responses per prompt, with batch size of 64, and a 7,168-token response limit. DrOPD uses the union of both models’ top-16 candidates and routes added FKL to the top-20% JS positions. All diagnostic and sensitivity analyses use the primary model pair.

![](images/77a22e70b57bb555418e8d39a4f0f99f2e811a3d01439f1ff27fd6f741fa5a43.jpg)

![](images/f9ab168ef6d23d82c7f2d9b7c3ef016801bb983c4ed873661d86bec9b124bb93.jpg)

![](images/0f622fdf0561837cafaf9c2747a7006da084e905b5d4347cb1e4c557dac5b203.jpg)  
Figure 3: Additional forward-KL accelerates early alignment. Changes in (a) FKL and (b) JS divergence during training. (c) Full-softmax teacher mass on the top-16 intersection.

Evaluation. We report AIME24, AIME25 (Balunovic et al., 2025), AMC23 (Li et al., 2024), Minerva (Lewkowycz et al., 2022), Olympiad (He et al., 2024), and MATH500 (Hendrycks et al., 2021). Each problem has eight responses sampled at temperature 0.7 and top-p = 0.95, with a 7,168-token limit. Avg@8 measures mean response correctness.

Compared methods. We compare DrOPD with standard OPD and entropy-based supervision strategies. OPD applies RKL supervision at all response positions. Stu-entropy and Tch-entropy retain all-position RKL and add FKL at the top 20% of positions ranked by the corresponding en tropy. We also reproduce EOPD (Jin et al., 2026), which adds FKL at positions where teacher entropy exceeds 0.8 while retaining RKL supervision at all positions.

## 4.2 MAIN RESULTS

Table 1 shows that DrOPD improves over standard OPD on both model pairs. For the 1.7B–4B pair, mean Avg@8 increases from 31.57% to 33.41%, a gain of 1.84 percentage points, with improvements on all six benchmarks. For the 4B–8B pair, DrOPD raises the mean from 44.64% to 45.23% and improves five of the six benchmarks. These results show that the benefit of targeted FKL supervision extends across both student scales.

DrOPD also outperforms entropy-based routing for both model pairs. For the 1.7B–4B pair, studentropy routing, tch-entropy routing, and the EOPD achieve 32.65%, 32.72%, and 32.07%, respectively, compared with 33.41% for DrOPD. This comparison highlights the value of the routing criterion. Consistent with the cross-sampling results, teacher–student disagreement provides a more effective guide for allocating forward supervision than either model’s uncertainty alone.

## 4.3 TRAINING DYNAMICS

We examine how DrOPD changes alignment during training through FKL, JS divergence, and teacher probability mass on the intersection of the two top-16 candidate sets (Figure 3). With additional FKL supervision, DrOPD maintains lower FKL than OPD over most of training (Figure 3a). DrOPD also reduces JS divergence more rapidly during the early updates (Figure 3b), consistent with directing additional supervision to high-disagreement positions. This faster initial alignment is consistent with the earlier gains and higher peak in reasoning accuracy shown in Figure 4c, although its JS divergence becomes slightly higher than OPD’s later in training. Teacher probability mass on the top-16 intersection provides a complementary view of candidate coverage (Figure 3c). This metric sums the teacher’s probabilities over the top-k shared candidates retained by both models. Under DrOPD, the mass quickly approaches 100% and remains high, indicating that the student’s top-16 candidates capture nearly all teacher probability mass. In contrast, OPD shows larger fluctuations and reaches a comparable level substantially later. These dynamics suggest that routed FKL help earlier coverage of teacher-supported candidates while reducing the distributional discrepancy.

![](images/82680add883fdb027ec1d6c059b5dd60649478b67f40e817096c493787576bf3.jpg)

![](images/4666a80dc34caa733a4cd6e0d0223a644d12450108a4e33d6a6de245c40a6d63.jpg)

![](images/9e47a1ed63240ab035b0cca19418343d50bba7f7d4e040c795cdcb65ac976c8a.jpg)  
Figure 4: Under-coverage recovery and learning progress. (a) Fraction of fixed high-JS positions with probability gaps at or above each threshold. (b) Severe under-coverage $( \delta _ { t } \ge 0 . 8 )$ on the same positions. (c) Mean Avg@8 across six benchmarks at successive training checkpoints.

## 4.4 UNDER-COVERAGE AND LEARNING PROGRESS

Recovering under-covered teacher modes. We evaluate the same recorded tokens on the fixed high-JS contexts from Section 2.3, comparing Base, OPD, and DrOPD on the same prefixes and recorded tokens. Figure 4a reports the fraction of positions with $\delta _ { t } \geq \tau$ as the gap threshold τ varies. DrOPD consistently reduces this proportion relative to OPD for thresholds from 0.5 to 0.9, indicating improved recovery across a range of under-coverage severities. At the severe-gap threshold of 0.8 (Figure 4b), the proportion decreases from 4.48% under OPD to 2.79% under DrOPD, a relative reduction of 37.9%, and falls below the base’s 3.83%. These results support the role of routed FKL in correcting teacher-supported tokens that remain severely underweighted after RKL training.

Faster gains in reasoning accuracy. The mean accuracy in Figure 4c shows that DrOPD also learns useful task behavior earlier. Its mean accuracy of six benchmarks reaches 30.40% at step 40, whereas OPD first exceeds 30% at step 180. DrOPD also achieves a higher peak accuracy, showing that its gains extend beyond the early training stage. Together with the training dynamics in Section 4.3, these results show that improved distributional alignment and reduced severe undercoverage accompany earlier and stronger gains in reasoning performance.

## 4.5 PARAMETER SENSITIVITY

We analysis the effects of the routing fraction $\rho ,$ the candidate count k, and candidate-set construction in Figure 5. Due to computational constraints, these experiments use four benchmarks: AIME24, AIME25, AMC23, and MATH500. Each problem samples eight responses and report mean Avg@8 across these benchmarks. All experiments use the 1.7B–4B model pair.

Routing fraction. Adding FKL supervision to the top 10% of high-JS positions increases mean Avg@8 from 33.80% to 34.63%, and 20% reaches the best, while larger routing fractions yield no further improvement (Figure 5a). These results favor concentrating additional supervision on a moderate fraction of high-disagreement positions rather than extending it broadly across the trajectory.

Candidate count. When k is small, the truncated distribution can differ substantially from the full-vocabulary, limiting the accuracy of the loss approximation. Increasing k improves this approximation. Figure 5b shows that performance gains saturate around k = 16, which we choose as the default to balance accuracy and computational cost.

Candidate-set construction. Using only the student’s candidates can exclude teacher-supported tokens that need FKL correction, while using only the teacher’s candidate may omit studentpreferred tokens relevant to RKL alignment. Their union retains both groups, supporting both RKL alignment and FKL correction. As shown in Figure 5c, the union achieves the highest mean Avg@8 of 35.64%, compared with 34.79% for student-only and 35.23% for teacher-only selection.

![](images/6215e1af89d3204d315ab3ff59c743163a4170caa899f94d58f82c31fae03d0d.jpg)

![](images/cfa66681911f7e10c3edf59052212e863520e9f964b85c610c774d8c9522de13.jpg)

![](images/3e623c696df279ab604082b8f870cc5f57649885e7ef0b09ab3423a8e40f0429.jpg)  
Figure 5: Sensitivity to routing and candidate support. Avg@8 averaged over four benchmarks for (a) the routed fraction, (b) top-k per model, and (c) candidate-set construction. The default uses top-20% JS routing with a top-16 union. The dashed line denotes OPD, equivalent to $\rho = 0$

Table 2: Effect of loss direction and supervision allocation.. Selected positions are the top 20% ranked by JS; remaining positions are the other 80%. “–” denotes no loss at those positions.
<table><tr><td>Method</td><td>Remaining positions</td><td>Selected positions</td><td>Mean(4)</td></tr><tr><td>OPD (Full RKL)</td><td>RKL</td><td>RKL</td><td>33.80</td></tr><tr><td>Full FKL</td><td>FKL</td><td>FKL</td><td>34.16</td></tr><tr><td>Sparse RKL</td><td>一</td><td>RKL</td><td>34.94</td></tr><tr><td>Sparse FKL</td><td>一</td><td>FKL</td><td>33.29</td></tr><tr><td>DrOPD</td><td>RKL</td><td>RKL + FKL</td><td>35.64</td></tr></table>

## 4.6 LOSS DIRECTION AND SUPERVISION ALLOCATION

Our cross-sampling results suggest that a small subset of high-disagreement positions accounts for much of the benefit of teacher intervention. This raises a natural question: can distillation supervision also be restricted to these positions? To investigate this, we rank positions by teacher–student JS divergence and divide them into the bottom 80% and top 20%. We then apply different combinations of RKL and FKL to these two regions, as summarized in Table 2. All variants use the same top-16 union candidate set and are evaluated by mean Avg@8 across the four benchmarks.

Full RKL (standard OPD) and Full FKL apply the respective loss to both regions, whereas Sparse RKL and Sparse FKL supervise only the top-20% high-JS positions. DrOPD retains RKL in both regions and adds FKL only to the top 20%. Sparse RKL improves mean Avg@8 from 33.80% to 34.94%, supporting the value of concentrating supervision at high-disagreement positions. However, the same restriction reduces FKL performance from 34.16% to 33.29%. Thus, the benefit of sparse supervision depends on the loss direction. DrOPD achieves the highest mean Avg@8 of 35.64%. Together, these results favor combining broad RKL alignment with selective FKL supervision over applying either objective alone in the tested configurations.

The strong Sparse RKL also raises a further question: could two directions benefit from different position or allocations that change during training? DrOPD provides a simple effective allocation, while jointly adapting the placement and relative strength of RKL and FKL remains for future work.

## 5 RELATED WORK

On-policy distillation and its mechanisms. OPD trains a student on its own generated trajectories using teacher feedback, reducing the mismatch between training contexts and those encountered at inference. GKD provides a framework for learning from student-generated sequences with differ ent divergence objectives (Agarwal et al., 2023), while MiniLLM develops RKL-based distillation to suppress student probability in regions that the teacher considers unlikely (Gu et al., 2024). Ex-OPD generalizes this formulation through reward extrapolation (Yang et al., 2026). Recent studies examine when this supervision succeeds, highlighting teacher–student compatibility, teacher reliability on student prefixes, and the effects of divergence estimation (Li et al., 2026a; Fu et al., 2026; Zhu et al., 2026). Corresponding improvements include more informative state exploration and calibrated teacher guidance in Uni-OPD (Hou et al., 2026a), variance reduction through control variates in vOPD (Oh et al., 2026), and explicit treatment of omitted probability mass in Tail-Aware Topk OPD (Huang & Wei, 2026). Self-distillation constructs a teacher from the model itself: SDPO conditions the model on textual feedback (Hubotter et al., 2026), while Self-Distilled Reasoner uses¨ verified reasoning traces as privileged context (Zhao et al., 2026). Our work focuses on a fixed external teacher and studies how to supplement RKL at positions where teacher-supported tokens remain severely under-covered.

Token importance and selective supervision. Selective distillation allocates supervision accord ing to the usefulness or reliability of individual tokens. REOPOLD combines entropy-based token sampling with reward clipping (Ko et al., 2026), and SelecTKD uses teacher verification to retain or down-weight token-level distillation losses (Huang et al., 2025). FiRe-OPD filters trajectories using teacher likelihood and then reweights tokens using teacher confidence and student uncertainty (Li et al., 2026b). TIP jointly analyzes student entropy and teacher–student divergence, identifying low-entropy, high-disagreement positions that entropy-only selection can miss (Xu et al., 2026). In self-distillation, PW-OPSD uses position-dependent weighting to account for teacher reliability along reasoning trajectories (Liu et al., 2026), while DASH uses disagreement to adapt the temporal extent of supervision (Hou et al., 2026b). Meng et al. (2026) use cross-sampling between base and RL-trained models to show that sparse high-divergence positions account for much of the performance change after RLVR. Building on this diagnostic, we compare routing signals through their downstream effects at comparable teacher-intervention ratios. We further examine the probability deficits at high-JS positions on fixed contexts.

Divergence objectives and teacher-mode recovery. Several methods adapt the form of supervision to address limitations of a uniform distillation objective. EOPD retains RKL and adds FKL where teacher entropy exceeds a threshold (Jin et al., 2026). AOPD retains reinforcement updates for positive-advantage tokens and replaces non-positive-advantage updates with imitation over teacher top-k candidates (Jia et al., 2026). TIDE independently selects student-excess and student-deficit positions, applying bounded Hellinger corrections to the former and teacher top-k supervision to recover missing probability mass at the latter (Yu et al., 2026b). TRACE applies FKL to annotated key reasoning spans, optionally applies RKL to error spans, and combines distillation with GRPO; it also supports online self-annotation (Wang et al., 2026). In privileged-context self-distillation, DASD uses entropy to choose between attraction toward and repulsion from the self-teacher (Zhang et al., 2026). DrOPD shares the motivation of recovering under-covered teacher modes, but uses a different supervision allocation: all positions retain RKL, and the highest-JS positions receive additional FKL. Our contribution connects this allocation to cross-sampling evidence of behavioral importance and fixed-context evidence of severe under-coverage, without requiring reasoning-span annotations.

## 6 CONCLUSION

We study where standard on-policy distillation fails to effectively transfer teacher knowledge. Controlled cross-sampling shows that teacher–student distributional disagreement identifies behaviorally critical positions more effectively than single-model uncertainty, while further analysis reveals severe teacher-mode under-coverage at many of these positions. Motivated by this diagnosis, we propose Divergence-Routed On-Policy Distillation (DrOPD), which retains RKL supervision throughout student-generated trajectories and selectively adds FKL supervision to the highest-disagreement positions. Experiments across multiple mathematical reasoning benchmarks and student–teacher configurations show that DrOPD consistently improves over standard OPD and entropy-based routing strategies. Our analyses further support targeted mode-covering supervision as an effective way to address teacher knowledge that is difficult to recover with RKL alone.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. arXiv preprint arXiv:2306.13649, 2023.

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovic, and Martin Vechev. MathArena:´ Evaluating LLMs on uncontaminated math competitions. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-0679.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes. arXiv preprint arXiv:2603.25562, 2026. URL https://arxiv.org/abs/2603.25562v2.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pp. 32694–32717, 2024.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024. URL https://aclanthology. org/2024.acl-long.211/.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Wenjin Hou, Shangpin Peng, Weinong Wang, Zheng Ruan, Yue Zhang, Zhenglin Zhou, Mingqi Gao, Yifei Chen, Kaiqi Wang, Hongming Yang, et al. Uni-opd: Unifying on-policy distillation with a dual-perspective recipe. arXiv preprint arXiv:2605.03677, 2026a.

ZhiYan Hou, Xinyu Tang, Hongyan An, Jianjin Zhang, Weizhen Wang, Yunyun Han, Gengsheng Li, Xiangzhao Hao, Haiyun Guo, Wenbin Hu, Jinqiao Wang, and Yafeng Deng. DASH: Divergence-Adaptive Supervision Horizons for On-Policy Self-Distillation of Reasoning Models. arXiv preprint arXiv:2608.06243, 2026b. URL https://arxiv.org/abs/2608.06243v2.

Haiduo Huang, Jiangcheng Song, Yadong Zhang, and Pengju Ren. Selectkd: Selective tokenweighted knowledge distillation for llms, 2025. URL https://arxiv.org/abs/2510. 24021.

Huipeng Huang and Hongxin Wei. Tail-Aware Top-k On-Policy Distillation. arXiv preprint arXiv:2608.14728, 2026. URL https://arxiv.org/abs/2608.14728v1.

Jonas Hubotter, Frederike L¨ ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta,¨ Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. Reinforcement Learning via Self-Distillation. arXiv preprint arXiv:2601.20802, 2026. doi: 10.48550/arXiv.2601.20802. URL https://arxiv.org/abs/2601.20802.

Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, and Zequn Sun. Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level. arXiv preprint arXiv:2605.06387, 2026. URL https: //arxiv.org/abs/2605.06387v4.

Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. arXiv preprint arXiv:2603.07079, 2026.

Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling reasoning efficiently via relaxed on-policy distillation. arXiv preprint arXiv:2603.11137, 2026.

Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics, 22(1):79–86, 1951.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. arXiv preprint arXiv:2206.14858, 2022. URL https://arxiv.org/abs/ 2206.14858.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. NuminaMath. Numina, Hugging Face dataset, 2024. URL https://huggingface.co/datasets/AI-MO/NuminaMath-CoT.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huanang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026a.

Yuying Li, Leqi Zheng, Yongzi Yu, Wenrui Zhou, Xuchang Zhong, Xing Hu, Jing Jin, Hangjie Yuan, and Tao Feng. Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation. arXiv preprint arXiv:2606.02684, 2026b. URL https://arxiv.org/abs/ 2606.02684v2.

Jianhua Lin. Divergence measures based on the shannon entropy. IEEE Transactions on Information theory, 37(1):145–151, 1991.

Xiaogeng Liu, Xinyan Wang, Yingzi Ma, Yechao Zhang, and Chaowei Xiao. When are teacher tokens reliable? position-weighted on-policy self-distillation for reasoning. arXiv preprint arXiv:2605.21606, 2026.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, Bolin Ding, and Jingren Zhou. Sparse but critical: A token-level analysis of distributional shifts in rlvr fine-tuning of llms. arXiv preprint arXiv:2603.22446, 2026.

Minjae Oh, Sangjun Song, Gyubin Choi, Yunho Choi, and Yohan Jo. KL for a KL: On-Policy Distillation with Control Variate Baseline. arXiv preprint arXiv:2605.07865, 2026. URL https: //arxiv.org/abs/2605.07865v1.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Jiaxuan Wang, Xuan Ouyang, Zhiyu Chen, Yulan Hu, Zheng Pan, Xin Li, and Lan-Zhe Guo. Trace: Distilling where it matters via token-routed self on-policy alignment, 2026. URL https:// arxiv.org/abs/2605.10194.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. TIP: Token Importance in On-Policy Distillation. arXiv preprint arXiv:2604.14084, 2026. doi: 10. 48550/arXiv.2604.14084. URL https://arxiv.org/abs/2604.14084.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation. arXiv preprint arXiv:2602.12125, 2026. doi: 10.48550/arXiv.2602.12125. URL https://arxiv.org/ abs/2602.12125.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026a.

Zichao Yu, Chengzhi Yu, Shengze Xu, Yujin Han, Bingqing Jiang, Xu Wang, and Difan Zou. Mismatch Matters: On-Policy Distillation Beyond Token Agreement. arXiv preprint arXiv:2608.09836, 2026b. URL https://arxiv.org/abs/2608.09836v1.

Hongbin Zhang, Chaozheng Wang, Kehai Chen, Youcheng Pan, Yang Xiang, Jinpeng Wang, and Min Zhang. Tailoring teaching to aptitude: Direction-adaptive self-distillation for llm reasoning. arXiv preprint arXiv:2605.22263, 2026.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. arXiv preprint arXiv:2601.18734, 2026. doi: 10.48550/arXiv.2601.18734. URL https://arxiv.org/ abs/2601.18734.

Siqi Zhu, Xuyan Ye, Hongyu Lu, Weiye Shi, and Ge Liu. The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes. arXiv preprint arXiv:2605.11182, 2026. doi: 10.48550/arXiv. 2605.11182. URL https://arxiv.org/abs/2605.11182.

## A EXPERIMENTAL DETAILS

## A.1 MODELS, DATA, AND OPTIMIZATION

We use verl (Sheng et al., 2024) to train the two student–teacher pairs described in Section 4, keeping each teacher frozen. Students train on processed DAPO-Math-17K prompts for one epoch with four responses per prompt. Table 3 summarizes the settings. Student advantages use the distillation signal without adding answer-correctness rewards.

Table 3: Training and evaluation settings. Candidate top-k specifies the support used for distillation, rather than a restriction on rollout sampling.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Training framework / rollout engine</td><td>verl / vLLM</td></tr><tr><td>Training data Training epochs</td><td>DAPO-Math-17K 1</td></tr><tr><td>Prompt batch / PPO minibatch size Responses per training prompt</td><td>64 / 64</td></tr><tr><td></td><td>4  $1 0 ^ { - 6 } / 0 . 0 1$ </td></tr><tr><td>Learning rate / weight decay</td><td></td></tr><tr><td>Learning-rate warmup fraction Maximum prompt / response length</td><td>0 1,024 / 7,168 tokens</td></tr><tr><td>Rollout / teacher temperature</td><td>1.0 / 1.0</td></tr><tr><td>Candidate support Routed fraction / added FKL coefficient</td><td>Top-16 union</td></tr><tr><td></td><td>20% / 1</td></tr><tr><td>PPO epochs / ratio clipping range Loss aggregation</td><td>1 / [0.8, 1.2]</td></tr><tr><td>Number of GPUs</td><td>Candidate sum, valid-response-token mean 8*A800</td></tr><tr><td>Benchmark responses per problem</td><td>8</td></tr><tr><td>Benchmark temperature / top-p</td><td>0.7 / 0.95</td></tr><tr><td>Benchmark response limit</td><td>7,168 tokens</td></tr></table>

## A.2 PROMPTING AND EVALUATION

Evaluation and cross-sampling append “Please reason step by step, and put your final answer within $\{ { \mathrm { b o x e d } } \{ \} . { \mathrm { : } } $ to each problem. We apply the tokenizer’s chat template with a generation prompt and enable thinking=False. The evaluator checks the extracted boxed answer by normalized matching or symbolic equivalence; responses without an extractable boxed answer are incorrect.

For a benchmark D with n problems, let $b _ { i j } \in \{ 0 , 1 \}$ indicate whether response $j$ to problem i is correct. We report

$$
\operatorname { A v g @ 8 } ( \mathcal { D } ) = \frac { 1 } { 8 n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { 8 } b _ { i j } , \qquad \operatorname { M e a n } ( m ) = \frac { 1 } { m } \sum _ { d = 1 } ^ { m } \operatorname { A v g @ 8 } ( \mathcal { D } _ { d } ) .\tag{11}
$$

Benchmarks have equal weight in each mean. Mean (6) includes AIME24, AIME25, AMC23, Minerva, Olympiad, and MATH500. Mean (4), used for sensitivity and loss allocation, includes AIME24, AIME25, AMC23, and MATH500.

## A.3 TRAINING COST

Table 4 reports the logged training cost for the 1.7B–4B student–teacher pair over 279 steps. Mean step time and cumulative step time are computed from the per-step training timers; cumulative time excludes initialization, checkpoint saving, and offline evaluation. Mean response length is averaged over the logged training steps.

## A.4 BASELINE CONFIGURATIONS

OPD uses all-position RKL on the top-16 union. The entropy baselines retain RKL and add FKL at the top-20% positions ranked by student or teacher entropy on the normalized union. EOPD

Table 4: Training cost for the 1.7B–4B pair over 279 steps.
<table><tr><td></td><td>Method</td><td>Mean step time (s)</td><td>Cumulative step time (h)</td><td>Mean response length (tokens)</td></tr><tr><td>Model pair 1.7B-4B</td><td>OPD</td><td>141.43</td><td>10.96</td><td>3117</td></tr><tr><td>1.7B-4B</td><td>DrOPD</td><td>152.62</td><td>11.83</td><td>3733</td></tr></table>

adds FKL where teacher entropy exceeds 0.8. Each added FKL term has coefficient one. Full FKL supervises all positions; Sparse RKL and Sparse FKL supervise only the top-20% JS positions. Candidate-set ablations retain the DrOPD allocation and change only the candidate support.

## B ALGORITHM AND GRADIENT ANALYSIS

## B.1 TRAINING PROCEDURE

Algorithm 1 summarizes DrOPD. Each position receives a distributional learning signal over its candidate set, even though the rollout records only one sampled token. Consequently, the added FKL can assign supervision to teacher-supported candidates that were not sampled at that position.

```latex
Algorithm 1 Divergence-Routed On-Policy Distillation
Require: Student $\pi _ { \theta } .$ frozen teacher $\pi _ { \mathrm { t c h } }$ , prompts $\mathcal { D } , k = 1 6 , \rho = 0 . 2$
1: for each prompt batch from D do
2: Set $\pi _ { \mathrm { o l d } }  \pi _ { \theta }$ and generate four responses per prompt
3: Form contexts $c _ { t }$ and the valid response-position set B
4: for each $t \in B$ do
5: Obtain student and teacher top-k candidates on $c _ { t }$
6: Form their union $\mathcal { C } _ { t }$ and mask duplicate candidate IDs
7: Normalize both distributions on $\mathcal { C } _ { t }$ to obtain $\widetilde { p } _ { t } , \widetilde { q } _ { t }$
8: Compute $d _ { t } = D _ { \mathrm { J S } } ( \widetilde { p } _ { t } , \widetilde { q } _ { t } )$
9: end for
10: Determine the top-ρ cutoff $\epsilon _ { \rho }$ over valid positions
11: Set $g _ { t } = \mathbb { 1 } [ d _ { t } \geq \epsilon _ { \rho } ]$ for each valid position
12: Construct all-position RKL rewards and add the FKL term where $g _ { t } = 1$
13: Detach the candidate advantages as in Equation 9
14: Update θ with the clipped surrogate in Equation 10
15: end for
```

The union contains at most 2k distinct candidates. Candidate IDs, routing decisions, and advantage coefficients remain fixed during each policy update; gradients flow through the student policy ratios. Candidate contributions are summed at each position and the resulting objective is averaged over valid response positions. Prompt tokens and padding do not contribute to this average. We use one PPO epoch and one minibatch per on-policy update.

The percentile router operates over the valid positions available to an actor reward-computation call. It sets $K _ { \rho } = \mathrm { r o u n d } ( \rho | B | )$ and uses the $K _ { \rho }$ -th largest score as the cutoff. Ties at the cutoff are included, so the selected fraction can slightly exceed $\rho .$ An empty population or $K _ { \rho } = 0$ yields no selected positions. Selection is performed over the batch population rather than independently within each response. Routing controls the additional FKL signal; RKL remains active at every valid position.

## B.2 DERIVATION OF THE KL LOGIT GRADIENTS

Let $p ( v )$ = softmax(z)(v) on a fixed support V, and let q be a fixed positive teacher distribution on that support. The softmax Jacobian is $\bar { \partial p ( u ) / \partial z ( v ) } = \bar { p ( u ) } ( \mathbb { 1 } [ u = \bar { v } ] - p ( v ) )$ ). Applying the chain rule gives

$$
\begin{array} { l } { \displaystyle \frac { \partial D _ { \mathrm { K L } } \big ( p \| q \big ) } { \partial z \big ( v \big ) } = \sum _ { u \in \mathcal { V } } p ( u ) \big ( \mathbb { 1 } [ u = v ] - p ( v ) \big ) \left( \log \frac { p ( u ) } { q ( u ) } + 1 \right) } \\ { \displaystyle \qquad = p ( v ) \left( \log \frac { p ( v ) } { q ( v ) } - D _ { \mathrm { K L } } ( p \| q ) \right) , } \end{array}\tag{12}
$$

$$
\frac { \partial D _ { \mathrm { K L } } ( q \| p ) } { \partial z ( v ) } = - \sum _ { u \in \mathcal { V } } q ( u ) \big ( \mathbb { 1 } [ u = v ] - p ( v ) \big ) = p ( v ) - q ( v ) .\tag{13}
$$

For a finite support and fixed positive $q ,$ the RKL is bounded above by $\mathrm { l o g } ( 1 / \operatorname* { m i n } _ { u } q ( u ) )$ . Since $p ( v ) \log p ( v ) \to 0 { \mathrm { ~ a s ~ } } p ( v ) \to 0$ , the RKL gradient in Equation 12 vanishes in this limit. The FKL gradient instead converges $\mathrm { t o } - q ( v )$ , retaining a corrective signal that increases the student’s logit under gradient descent. These identities describe the local gradient behavior at a fixed context and motivate the complementary supervision; the empirical analyses evaluate its effect during training. They also apply to normalized candidate distributions when the support is held fixed.

## B.3 FROM DISTILLATION LOSS TO CANDIDATE REWARDS

We derive the candidate rewards by matching the negative gradient of the distillation loss to a weighted sum of student log-probability gradients. Fix the context $c _ { t }$ , the candidate set $\mathcal { C } _ { t }$ , the teacher distribution $\widetilde { q } _ { t }$ , and the routing gate $g _ { t }$ . The normalized student distribution $\widetilde { p } _ { t }$ depends on θ. The per-position loss in Equation (8) of the main paper is

$$
\ell _ { t } ( \theta ) = \ell _ { \mathrm { R K L } , t } + g _ { t } \ell _ { \mathrm { F K L } , t } = \sum _ { v \in \mathcal { C } _ { t } } \widetilde { p } _ { t } ( v ) \log \frac { \widetilde { p } _ { t } ( v ) } { \widetilde { q _ { t } } ( v ) } + g _ { t } \sum _ { v \in \mathcal { C } _ { t } } \widetilde { q } _ { t } ( v ) \log \frac { \widetilde { q } _ { t } ( v ) } { \widetilde { p _ { t } } ( v ) } .\tag{14}
$$

Our goal is to identify coefficients $R _ { t } ( v )$ such that $\begin{array} { r } { - \nabla _ { \theta } \ell _ { t } = \sum _ { v \in \mathcal { C } _ { \star } } R _ { t } ( v ) \nabla _ { \theta } \log \widetilde { p } _ { t } ( v ) } \end{array}$ . The coefficients are treated as detached weights when applying this gradient; they need not equal the negative loss value.

RKL contribution. Both the weighting distribution and the logarithm in RKL depend on the student parameters. Differentiating and using $\nabla _ { \boldsymbol { \theta } } \widetilde { p } _ { t } ( \boldsymbol { v } ) = \widetilde { p } _ { t } ( \boldsymbol { v } ) \nabla _ { \boldsymbol { \theta } } \log \widetilde { p } _ { t } ( \boldsymbol { v } )$ gives

$$
\begin{array} { r l } & { \nabla _ { \boldsymbol { \theta } } \ell _ { \mathrm { R K L } , t } = \displaystyle \sum _ { \boldsymbol { v } \in \mathcal { C } _ { t } } \left( \log \frac { \widetilde { p } _ { t } ( \boldsymbol { v } ) } { \widetilde { q } _ { t } ( \boldsymbol { v } ) } + 1 \right) \nabla _ { \boldsymbol { \theta } } \widetilde { p } _ { t } ( \boldsymbol { v } ) } \\ & { \qquad = \displaystyle \sum _ { \boldsymbol { v } \in \mathcal { C } _ { t } } \widetilde { p } _ { t } ( \boldsymbol { v } ) \left( \log \frac { \widetilde { p } _ { t } ( \boldsymbol { v } ) } { \widetilde { q } _ { t } ( \boldsymbol { v } ) } + 1 \right) \nabla _ { \boldsymbol { \theta } } \log \widetilde { p } _ { t } ( \boldsymbol { v } ) . } \end{array}\tag{15}
$$

The constant contribution cancels because the student probabilities sum to one on the fixed candidate set:

$$
\begin{array} { r } { \displaystyle \sum _ { v \in \mathcal { C } _ { t } } \widetilde { p } _ { t } ( v ) \nabla _ { \theta } \log \widetilde { p } _ { t } ( v ) = \nabla _ { \theta } \displaystyle \sum _ { v \in \mathcal { C } _ { t } } \widetilde { p } _ { t } ( v ) = 0 . } \end{array}\tag{16}
$$

Therefore,

$$
- \nabla _ { \boldsymbol { \theta } } \ell _ { \mathrm { R K L , } t } = \sum _ { \boldsymbol { v } \in \mathcal { C } _ { t } } \underbrace { \left[ - \widetilde { p } _ { t } ( \boldsymbol { v } ) \log \frac { \widetilde { p } _ { t } ( \boldsymbol { v } ) } { \widetilde { q } _ { t } ( \boldsymbol { v } ) } \right] } _ { R _ { \mathrm { R K L } , t } ( \boldsymbol { v } ) } \nabla _ { \boldsymbol { \theta } } \log \widetilde { p } _ { t } ( \boldsymbol { v } ) .\tag{17}
$$

Thus, the RKL reward coefficient is the negative per-candidate contribution to RKL, including its student-probability weight.

FKL contribution. The teacher distribution is fixed, so FKL decomposes into a constant teacher term and a student cross-entropy term:

$$
\ell _ { \mathrm { F K L } , t } = \underbrace { \sum _ { v \in \mathcal { C } _ { t } } \widetilde { q } _ { t } ( v ) \log \widetilde { q } _ { t } ( v ) } _ { \mathrm { i n d e p e n d e n t ~ o f ~ } \theta } - \sum _ { v \in \mathcal { C } _ { t } } \widetilde { q } _ { t } ( v ) \log \widetilde { p } _ { t } ( v ) .\tag{18}
$$

Only the second term contributes to the gradient, yielding

$$
- \nabla _ { \boldsymbol { \theta } } \ell _ { \mathrm { F K L } , t } = \sum _ { \boldsymbol { v } \in \mathcal { C } _ { t } } \underbrace { \widetilde { q } _ { t } ( \boldsymbol { v } ) } _ { R _ { \mathrm { F K L } , t } ( \boldsymbol { v } ) } \nabla _ { \boldsymbol { \theta } } \log \widetilde { p } _ { t } ( \boldsymbol { v } ) .\tag{19}
$$

The corresponding coefficient is therefore simply the teacher probability, rather than the percandidate FKL value. Combining the two contributions gives

$$
\boxed { R _ { t } ( v ) = - \widetilde { p } _ { t } ( v ) \log \frac { \widetilde { p } _ { t } ( v ) } { \widetilde { q } _ { t } ( v ) } + g _ { t } \widetilde { q } _ { t } ( v ) , \qquad v \in \mathcal { C } _ { t } . }\tag{20}
$$

This recovers Equation (9) of the main paper through the negative-gradient identity above. The different forms of the RKL and FKL rewards follow from differentiating a student-weighted divergence versus a teacher-weighted divergence, rather than from using different definitions of reward.

## C CROSS-SAMPLING PROTOCOL AND ADDITIONAL RESULTS

## C.1 GENERATION AND ROUTING

We evaluate 40 AMC23 problems with eight responses each, giving 320 responses per routing configuration. Both models remain frozen. The default sampler is the base student, and the teacher supplies the next token only when the routing condition is satisfied. Both models subsequently condition on the same updated prefix, including any teacher-sampled token. Routing divergences use temperature-one distributions normalized on the top-16 union. Token generation uses temperature 0.7, a 7,168-token response limit, and full-vocabulary sampling without top-p or top-k truncation. The top-16 approximation therefore controls the routing score, not the set of tokens that can be generated.

For divergence and entropy criteria, teacher intervention occurs when the score exceeds the corresponding threshold. For random routing, each position is assigned to the teacher with probability r, independently of the distributions. Tables 5 and 6 give the numerical threshold sweeps underlying Figure 1a, including points outside its displayed range. Entropy-only cross-sampling uses the indicated model’s full next-token distribution. All entropies and divergences use natural logarithms.

The intervention rate η is the mean of the per-response intervention fractions, as defined in Equation 2. We use the observed rate rather than the nominal threshold or random-routing probability when comparing strategies. Different routing policies can produce different prefixes and response lengths; they are evaluated on the same problems, but do not share a fixed set of decoding contexts. Standalone Base and Teacher achieve 14.38% and 60.63% Avg@8, respectively.

Table 5: Divergence-based cross-sampling on AMC23. τ is the routing threshold; η is the measured intervention rate. Avg@8 is reported in percent.
<table><tr><td colspan="3">JS</td><td colspan="3">RKL</td><td colspan="3">FKL</td></tr><tr><td>T</td><td>η (%)</td><td>Avg@8</td><td>T</td><td>η (%)</td><td>Avg@8</td><td>T</td><td>η (%)</td><td>Avg@8</td></tr><tr><td>0.5</td><td>0.21</td><td>37.81</td><td>3</td><td>0.70</td><td>40.31</td><td>2.2</td><td>0.54</td><td>36.56</td></tr><tr><td>0.4</td><td>0.59</td><td>40.62</td><td>2.6</td><td>0.97</td><td>43.12</td><td>1.9</td><td>0.79</td><td>40.62</td></tr><tr><td>0.3</td><td>1.48</td><td>45.94</td><td>2.2</td><td>1.33</td><td>46.88</td><td>1.6</td><td>1.22</td><td>44.06</td></tr><tr><td>0.2</td><td>3.24</td><td>49.06</td><td>1.8</td><td>1.97</td><td>45.62</td><td>1</td><td>2.80</td><td>46.25</td></tr><tr><td>0.18</td><td>3.93</td><td>51.88</td><td>1.4</td><td>3.11</td><td>50.62</td><td>0.7</td><td>4.41</td><td>52.50</td></tr><tr><td>0.16</td><td>4.46</td><td>54.06</td><td>1</td><td>4.89</td><td>56.88</td><td>0.6</td><td>5.16</td><td>54.06</td></tr><tr><td>0.12</td><td>6.08</td><td>56.88</td><td>0.8</td><td>6.34</td><td>58.13</td><td>0.5</td><td>6.10</td><td>54.37</td></tr><tr><td>0.1</td><td>7.24</td><td>55.94</td><td>0.7</td><td>6.99</td><td>58.44</td><td>0.4</td><td>7.13</td><td>57.19</td></tr><tr><td>0.08</td><td>8.69</td><td>57.81</td><td>0.6</td><td>8.11</td><td>60.62</td><td>0.3</td><td>8.77</td><td>63.12</td></tr><tr><td>0.07</td><td>9.23</td><td>60.31</td><td>0.5</td><td>9.30</td><td>64.06</td><td>0.2</td><td>11.08</td><td>63.44</td></tr><tr><td>0.06</td><td>10.13</td><td>65.00</td><td>0.4</td><td>10.83</td><td>62.50</td><td>0.1</td><td>14.29</td><td>58.75</td></tr><tr><td>0.05</td><td>11.22</td><td>61.88</td><td>0.3</td><td>12.95</td><td>61.25</td><td>0.08</td><td>15.69</td><td>61.56</td></tr><tr><td>0.04</td><td>12.87</td><td>61.88</td><td>0.2</td><td>15.49</td><td>65.00</td><td>0.06</td><td>17.23</td><td>61.56</td></tr><tr><td>0.02</td><td>16.19</td><td>65.31</td><td>0.1</td><td>19.32</td><td>62.81</td><td>0.05</td><td>18.00</td><td>62.81</td></tr><tr><td>0.01</td><td>19.65</td><td>63.44</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr></table>

Table 6: Entropy-based and random cross-sampling on AMC23. r is the random teacherintervention probability; the remaining columns follow Table 5.
<table><tr><td colspan="3">Student entropy</td><td colspan="3">Teacher entropy</td><td colspan="3">Random</td></tr><tr><td>T</td><td>η (%)</td><td>Avg@8</td><td>T</td><td>η (%)</td><td>Avg@8</td><td>r</td><td>η (%)</td><td>Avg@8</td></tr><tr><td>1.8</td><td>0.82</td><td>34.69</td><td>1.6</td><td>3.83</td><td>25.00</td><td>0.04</td><td>3.67</td><td>16.56</td></tr><tr><td>1.6</td><td>1.30</td><td>37.19</td><td>1.4</td><td>4.51</td><td>27.81</td><td>0.06</td><td>5.36</td><td>16.88</td></tr><tr><td>1.4</td><td>2.03</td><td>40.00</td><td>1.2</td><td>5.42</td><td>29.69</td><td>0.08</td><td>7.30</td><td>17.50</td></tr><tr><td>1.2</td><td>3.12</td><td>42.50</td><td>1</td><td>6.28</td><td>37.50</td><td>0.1</td><td>9.03</td><td>20.00</td></tr><tr><td>1</td><td>4.79</td><td>42.19</td><td>0.9</td><td>6.78</td><td>31.25</td><td>0.12</td><td>10.95</td><td>20.31</td></tr><tr><td>0.9</td><td>5.84</td><td>45.94</td><td>0.8</td><td>7.90</td><td>34.06</td><td>0.14</td><td>12.88</td><td>20.94</td></tr><tr><td>0.8</td><td>6.80</td><td>48.44</td><td>0.7</td><td>9.01</td><td>37.19</td><td>0.18</td><td>16.50</td><td>24.38</td></tr><tr><td>0.7</td><td>9.00</td><td>50.00</td><td>0.6</td><td>11.04</td><td>40.94</td><td>0.2</td><td>18.39</td><td>26.56</td></tr><tr><td>0.6</td><td>11.03</td><td>50.62</td><td>0.5</td><td>12.31</td><td>35.31</td><td>0.22</td><td>20.19</td><td>27.81</td></tr><tr><td>0.5</td><td>12.51</td><td>57.19</td><td>0.4</td><td>13.94</td><td>40.94</td><td>0.24</td><td>22.18</td><td>26.56</td></tr><tr><td>0.4</td><td>14.53</td><td>55.62</td><td>0.3</td><td>15.20</td><td>41.88</td><td>一</td><td>一</td><td>一</td></tr><tr><td>0.3</td><td>16.56</td><td>56.88</td><td>0.2</td><td>15.44</td><td>45.00</td><td>一</td><td>一</td><td>一</td></tr><tr><td>0.2</td><td>19.62</td><td>60.00</td><td>0.1</td><td>17.99</td><td>45.62</td><td>一</td><td>一</td><td>一</td></tr></table>

## C.2 ENTROPY REGIONS WITHIN HIGH-JS POSITIONS

We further restrict teacher intervention to one of four entropy regions within positions with JS > 0.06. Student and teacher entropy are each computed on the corresponding model’s normalized top-16 distribution. High and low entropy denote values above and below 0.5, respectively. The region condition is evaluated on the current prefix at every decoding step, together with the JS condition.

Table 7: Cross-sampling restricted to individual entropy regions. Each row is a separate policy on the same 40 AMC23 problems. The column order explicitly distinguishes student and teacher entropy.
<table><tr><td>Student entropy</td><td>Teacher entropy</td><td>Intervention (%)</td><td>Avg@8 (%)</td></tr><tr><td>High</td><td>High</td><td>7.19</td><td>27.81</td></tr><tr><td>High</td><td>Low</td><td>3.82</td><td>47.50</td></tr><tr><td>Low</td><td>High</td><td>1.20</td><td>18.75</td></tr><tr><td>Low</td><td>Low</td><td>0.16</td><td>16.25</td></tr></table>

Table 7 shows that intervention is particularly useful when the student is uncertain and the teacher has a concentrated preference: the high-student/low-teacher region reaches 47.50% Avg@8 at 3.82% intervention. This provides a complementary description of high-disagreement positions. Because the four policies have different realized intervention rates and induce different trajectories, the table does not isolate a causal effect of entropy at a matched budget. DrOPD uses JS directly and does not require an entropy-region classifier.

## D FIXED-CONTEXT UNDER-COVERAGE ANALYSIS

## D.1 SHARED TRAJECTORIES AND RECORDED TOKENS

The fixed-context analysis uses 320 Base–Teacher cross-sampled responses generated with the JS threshold 0.06. The shared archive contains 996,919 response positions, of which 97,122 satisfy the original $\mathrm { J S } > 0 . 0 6$ condition. We hold the prefixes, recorded token IDs, and this original high-JS mask fixed when comparing Base, OPD, and DrOPD. At the selected positions, the recorded token was sampled from the teacher. We check alignment of example IDs, response indices, token IDs, routing decisions, and stored teacher probabilities across the model-specific records.

Each position contributes the probability of its actually recorded token, rather than a maximum gap over candidate tokens. The replay cache stores probabilities normalized over each model’s retained top-16 candidates, assigning zero when the recorded token is outside that set. We compute $\delta _ { t } = [ q _ { t } ( v _ { t } ) - p _ { t } ( v _ { t } ) ]$ <sub>+</sub> from these stored probabilities, following the positive-gap definition in Equation 3. The analysis compares how different students score the same teacher-generated tokens on the same contexts.

## D.2 CONDITIONAL MEANS AND TAIL FRACTIONS

Let $\mathcal { D } _ { j }$ be the j-th decile of positions ranked by their original Base–Teacher JS scores. Figure 1b reports the conditional positive-gap mean

$$
\mu _ { j } ^ { + } = \frac { \sum _ { t \in \mathcal { D } _ { j } } \delta _ { t } \Im [ \delta _ { t } > 0 ] } { \sum _ { t \in \mathcal { D } _ { j } } \Im [ \delta _ { t } > 0 ] } .\tag{21}
$$

Positions with zero positive gap are excluded from this mean. Deciles are determined once using the original JS scores; the highest decile begins at approximately 0.05746. It therefore closely overlaps, but is not identical to, the fixed $\mathrm { J } \mathrm { S } > 0 . { \bar { 0 } } 6$ subset used for the tail comparisons.

Let $\mathcal { H } = \{ t : d _ { t } ^ { \mathrm { B a s e } } > 0 . 0 6 \}$ denote this fixed high-JS subset. For model M and a positive threshold τ , define

$$
F _ { M } ( \tau ) = \frac { 1 } { | \mathcal { H } | } \sum _ { t \in \mathcal { H } } \mathbb { 1 } [ \delta _ { t } ^ { M } \geq \tau ] , \qquad C _ { \mathrm { O P D } } ( \tau ) = 1 0 0 \left( \frac { F _ { \mathrm { O P D } } ( \tau ) } { F _ { \mathrm { B a s e } } ( \tau ) } - 1 \right) .\tag{22}
$$

Figure 4a plots 100 $F _ { M } ( \tau )$ , while Figure 1c plots $C _ { \mathrm { O P D } } ( \tau )$ wherever the Base tail is nonzero. Each tail fraction uses all positions in H as its denominator, including positions without a positive gap. The relative-change curve measures percentage change, rather than a difference in percentage points. The severe-gap bars in Figure 4b are the values at $\tau = 0 . 8$

Table 8: Fraction (%) of fixed original high-JS positions above each recorded-token gap threshold.   
Every row uses the same 97,122 positions.

<table><tr><td>Gap threshold τ</td><td>Base</td><td>OPD</td><td>DrOPD</td></tr><tr><td>0.2</td><td>73.01</td><td>51.40</td><td>56.86</td></tr><tr><td>0.4</td><td>39.97</td><td>30.82</td><td>31.16</td></tr><tr><td>0.5</td><td>26.35</td><td>22.05</td><td>20.68</td></tr><tr><td>0.6</td><td>16.03</td><td>14.78</td><td>12.48</td></tr><tr><td>0.7</td><td>8.72</td><td>8.88</td><td>6.57</td></tr><tr><td>0.8</td><td>3.83</td><td>4.48</td><td>2.79</td></tr><tr><td>0.9</td><td>1.12</td><td>1.65</td><td>0.71</td></tr></table>

Table 8 shows how the comparison changes with deficit severity. Both trained students reduce moderate gaps, while DrOPD also reduces the extreme tail. At threshold 0.8, the fractions are 3.83% for Base, 4.48% for OPD, and 2.79% for DrOPD. The relative reduction from OPD is computed from unrounded fractions. The selected positions comprise 9.74% of all archived tokens; this tokenpooled fraction differs from the mean per-response intervention rate used in the cross-sampling curves.

## E TRAINING DIAGNOSTIC DEFINITIONS

## E.1 DISTRIBUTIONAL ALIGNMENT

The FKL and JS curves in Figure 3 summarize each method’s own student-generated training rollouts. At every valid response position, we form the deduplicated top-16 union and normalize the student and teacher distributions on that support. The logged values average the corresponding divergence over all valid positions, including positions not selected for additional FKL. This differs from the fixed-context analysis in Appendix D, where the prefixes and recorded tokens are shared across models.

The distributions are evaluated using natural logarithms. The displayed training curves show raw step values together with trailing means over at most 15 steps. For a metric $a _ { s }$ at training step $s ,$ the smoothed value is $\begin{array} { r } { \overline { { a } } _ { s } = L _ { s } ^ { - 1 } \sum _ { j = s - L _ { s } + 1 } ^ { s } a _ { j } } \end{array}$ , where $L _ { s } = \operatorname* { m i n } ( 1 5 , s )$ . Benchmark Mean (6) in Figure 4c is computed from the eight-response evaluation results at each available evaluation step and is plotted without smoothing. It is an offline reasoning-accuracy measure, rather than the logged correctness of a training batch.

## E.2 PROBABILITY MASS ON THE TOP-16 INTERSECTION

Let $\mathcal { T } _ { t } = \mathrm { T o p } 1 6 ( p _ { t } ) \cap$ Top1 $. 6 ( q _ { t } )$ and let $B _ { + } = \{ t \in B : | \mathbb { T } _ { t } | > 0 \}$ . Using full-softmax model probabilities, the teacher and student intersection masses are

$$
M _ { q } = \frac { 1 0 0 } { | \mathcal { B } _ { + } | } \sum _ { t \in \mathcal { B } _ { + } } \sum _ { v \in \mathcal { Z } _ { t } } \pi _ { \mathrm { t c h } } ( v \mid c _ { t } ) , \qquad M _ { p } = \frac { 1 0 0 } { | \mathcal { B } _ { + } | } \sum _ { t \in \mathcal { B } _ { + } } \sum _ { v \in \mathcal { Z } _ { t } } \pi _ { \theta } ( v \mid c _ { t } ) .\tag{23}
$$

These probabilities are not renormalized on the intersection. Positions with an empty intersection are excluded from the logged average. Figure 3c reports $M _ { q }$ , and Figure 6 provides the complementary student quantity $M _ { p }$ . The fields are collected on training rollouts even though their logging names begin with $\mathtt { v a l - t o p k }$

High teacher intersection mass means that most teacher probability lies on candidates also included in the student’s top-16 set. High student intersection mass indicates the analogous property for student probability. These statistics measure shared candidate support; FKL and JS additionally assess how probability is distributed within the union. In particular, candidate membership alone does not establish that the two models assign equal probability to those candidates.

![](images/eae6bb7d30d4bf4c0cbc95910f0fce314c6b2c6481eb700ada731a7ed56f3a1a.jpg)  
Figure 6: Student probability mass on the top-16 intersection. Faint lines show raw step values and solid lines show trailing 15-step means. This complements the teacher-mass statistic in Figure 3c using the corresponding student probabilities.

DrOPD establishes high student intersection mass earlier and maintains it through the middle of training, while OPD reaches a similar level later. Together with the teacher-mass curve, this pattern supports the interpretation that additional FKL accelerates the recovery of shared candidate support.

## F DISCLOSURE OF AI USE

AI tools were used to assist with literature searches and the exploration of potential research directions. After the human researchers designed the method, conducted the experiments, and obtained the results, AI tools assisted with drafting manuscript sections, analyzing and visualizing existing experimental results, and preparing mathematical derivations. The manuscript was refined through iterative human review and revision, with AI assistance for language polishing. The authors take responsibility for the final content of the paper, including its claims, analyses, and conclusions.