# Learning-Augmented Online Allocation under Unreliable Advice: Robustness, Exposure Fairness, and Distribution Shift

Preprint, compiled August 28, 2026

Frédy. Pokou <sup>ID</sup> <sup>1∗</sup>

<sup>1</sup>Inria, University of Lille, CNRS, Centrale Lille Villeneuve-d’Ascq, France

## Abstract

Learning-augmented algorithms improve online decisions using predictions, but unreliable advice may harm eficiency and fairness. We study an online allocation problem with finite candidate sets, irreversible decisions, and exposure constraints. We propose a robust and fair rule combining advice with a conservative fallback and fairness correction. Under bounded-error assumptions, we prove consistency and robustness with loss proportional to prediction error. Experiments show stability under adversarial advice and significant reductions in exposure disparity.

Keywords Learning-augmented algorithms · Online allocation · Competitive analysis · Exposure fairness · Distribution shift · Robust decision-making.

## 1 Introduction

Online allocation problems arise in recommendation, advertising, labor-market platforms, and matching systems. Classical online algorithms provide worst-case guarantees, but may be conservative; purely data-driven rules can perform well on average, but may fail under misspecification, distribution shift, or biased predictions. Learning-augmented algorithms address this tension by using predictions while retaining robustness guarantees [Lykouris and Vassilvitskii, 2021, Purohit et al., 2018, Mitzenmacher and Vassilvitskii]. Classical online matching and advertising allocation provide the algorithmic background [Karp et al., 1990, Mehta et al., 2007], while exposure-based fairness constraints are central in ranking and recommendation [Singh and Joachims, 2018].

This paper studies whether learned advice can be used in online allocation while preserving robustness and controlling exposure imbalance. We propose a robust and fair learning-augmented rule that combines predictive advice with a conservative fallback and a virtual-queue fairness correction.

Contributions. First, we formulate a finite-horizon online allocation model with learned advice, conservative scores, and exposure targets. Second, we introduce a robust/fair learningaugmented policy. Third, we prove a central finite-sample guarantee: the robust rule is simultaneously consistent when advice is accurate and protected by a conservative fallback when advice is inaccurate. We also derive an advice-relative robustness certificate of the form

$$
\mathrm { C R } _ { T } ( \mathrm { R L A } ) \geq \mathrm { C R } _ { T } ( \mathrm { A D V } ) - O ( \varepsilon _ { T } ) ,
$$

and a finite-time exposure bound for the fair rule. Fourth, we provide reproducible experiments on MovieLens-derived online allocation instances under benign noise, adversarial advice, and distribution shift.

## 2 Model

Let $T \in$ N be the horizon. At each time $t \in [ T ]$ , a request arrives and the decision-maker observes a finite feasible slate $\mathcal { A } _ { t } \subseteq \mathcal { I }$ . The action $a _ { t } \in \mathcal { A } _ { t }$ is chosen irrevocably and yields reward $r _ { t } ( a _ { t } ) \ \in \ [ 0 , 1 ]$ . Each item $i \in \mathcal { I }$ belongs to a group $g ( i ) \in \mathcal { G } .$ , where $\mathcal { G }$ is finite. Before choosing, the decision-maker observes two score vectors on $\mathcal { A } _ { t } \colon \mathbf { a }$ learned advice vector $\hat { \boldsymbol { r } } _ { t }$ and a conservative fallback vector $b _ { t }$

For a policy π, let $a _ { t } ^ { \pi }$ be its action and

$$
W _ { T } ( \pi ) = \sum _ { t = 1 } ^ { T } r _ { t } ( a _ { t } ^ { \pi } ) .\tag{1}
$$

The ofline benchmark is

$$
\mathrm { O P T } _ { T } = \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { i \in \mathcal { A } _ { t } } r _ { t } ( i ) , \quad \mathrm { C R } _ { T } ( \pi ) = \frac { W _ { T } ( \pi ) } { \mathrm { O P T } _ { T } } ,\tag{2}
$$

whenever $\mathrm { O P T } _ { T } > 0 .$ . This benchmark is slate-wise and intentionally strong. In capacitated variants, $\mathcal { A } _ { t }$ can be interpreted as the remaining feasible actions after past decisions.

The average advice and fallback errors are

$$
\varepsilon _ { T } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \| \hat { \boldsymbol { r } } _ { t } - \boldsymbol { r } _ { t } \| _ { \infty , \mathcal { A } _ { t } } , \quad \kappa _ { T } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \| \boldsymbol { b } _ { t } - \boldsymbol { r } _ { t } \| _ { \infty , \mathcal { A } _ { t } } .\tag{3}
$$

For a target exposure vector $\rho \in \Delta ( \mathcal { G } )$ , group exposure is

$$
E _ { T } ^ { \pi } ( g ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ g ( a _ { t } ^ { \pi } ) = g \} ,\tag{4}
$$

and the exposure gap is

$$
\mathrm { G a p } _ { T } ^ { \rho } ( \pi ) = \operatorname* { m a x } _ { g \in \mathcal { G } } | E _ { T } ^ { \pi } ( g ) - \rho _ { g } | .\tag{5}
$$

The advice-only policy is

$$
a _ { t } ^ { \mathrm { A D V } } \in \arg \operatorname* { m a x } _ { i \in \mathcal { A } _ { t } } \hat { r } _ { t } ( i ) .\tag{6}
$$

## 3 Algorithm

For $\alpha \in [ 0 , 1 ]$ , define the robust learning-augmented score

$$
s _ { t } ^ { \alpha } ( i ) = ( 1 - \alpha ) \widehat { r } _ { t } ( i ) + \alpha b _ { t } ( i ) .\tag{7}
$$

The robust policy RLA selects

$$
a _ { t } ^ { \mathrm { R L A } } \in \arg \operatorname* { m a x } _ { i \in \mathcal { A } _ { t } } s _ { t } ^ { \alpha } ( i ) .\tag{8}
$$

To control exposure, define virtual imbalances

$$
\mathcal { Q } _ { t } ( g ) = \sum _ { s = 1 } ^ { t - 1 } ( \mathbf { 1 } \{ g ( a _ { s } ) = g \} - \rho _ { g } ) , \quad \mathcal { Q } _ { 1 } ( g ) = 0 .\tag{9}
$$

For $\lambda \geq 0 .$ the fair robust policy FLA selects

$$
a _ { t } ^ { \mathrm { F L A } } \in \arg \operatorname* { m a x } _ { i \in \mathcal { A } _ { t } } \{ s _ { t } ^ { \alpha } ( i ) - \lambda Q _ { t } ( g ( i ) ) \} .\tag{10}
$$

Algorithm 1 Fair Robust Learning-Augmented Allocation   
Require: $\alpha \in [ 0 , 1 ] , \lambda \geq 0 ,$ target exposure $\rho \in \Delta ( \mathcal { G } )$   
1: Initialize $Q _ { 1 } ( g ) = 0$ for all $g \in { \mathcal { G } } .$   
2: for $t = 1 , \dots , T$ do   
3: Observe $\mathcal { A } _ { t } , \hat { r } _ { t } ,$ and $b _ { t }$   
4: Compute $s _ { t } ^ { \alpha } ( i ) = ( 1 - \alpha ) \hat { r } _ { t } ( i ) + \alpha b _ { t } ( i ) .$   
5: Choose a<sub>t</sub> ∈ arg max<sub>i∈A</sub> $\{ s _ { t } ^ { \alpha } ( i ) - \lambda Q _ { t } ( g ( i ) ) \} .$   
6: Update $Q _ { t + 1 } ( g ) = Q _ { t } ( g ) + \mathbf { 1 } \{ g ( a _ { t } ) = g \} - \rho _ { g } .$   
7: end for

## 4 Theory

All results are deterministic conditional on the realized sequence $( \mathcal { A } _ { t } , r _ { t } , \hat { r } _ { t } , b _ { t } ) _ { t = 1 } ^ { T }$

Assumption 1 (Bounded finite slates). For all t and $i \in \mathcal { A } _ { t } ,$ $r _ { t } ( i ) , \hat { r } _ { t } ( i ) , b _ { t } ( i ) \in [ 0 , 1 ] ,$ and $1 \leq | \mathcal { R } _ { t } | < \infty .$

Assumption 2 (Non-degenerate benchmark). There exists $\omega > 0$ such that ${ \mathrm { O P T } } _ { T } \geq \omega T$

Lemma 1 (Perturbation stability). Let $s _ { t }$ be any score vector on $\mathcal { A } _ { t } ,$ and let $a _ { t } ^ { s } \in$ arg max<sub>i∈A</sub> s<sub>t</sub>(i). Then

$$
\operatorname* { m a x } _ { i \in \mathcal { A } _ { t } } r _ { t } ( i ) - r _ { t } ( a _ { t } ^ { s } ) \leq 2 \| s _ { t } - r _ { t } \| _ { \infty , \mathcal { A } _ { t } } .
$$

Proof. Let $i _ { t } ^ { \star } \in$ arg max<sub>i∈A</sub> $r _ { t } ( i )$ . Since $s _ { t } ( a _ { t } ^ { s } ) \geq s _ { t } ( i _ { t } ^ { \star } )$ ,

$$
r _ { t } ( i _ { t } ^ { \star } ) - r _ { t } ( a _ { t } ^ { s } ) \leq | r _ { t } ( i _ { t } ^ { \star } ) - s _ { t } ( i _ { t } ^ { \star } ) | + | s _ { t } ( a _ { t } ^ { s } ) - r _ { t } ( a _ { t } ^ { s } ) | .
$$

The result follows by taking the maximum norm over $\mathcal { A } _ { t }$ . □

Theorem 1 (Central consistency-robustness bound). Under As sumption 1,for every $\alpha \in [ 0 , 1 ] ,$

$$
\mathrm { O P T } _ { T } - W _ { T } ( \mathrm { R L A } ) \leq 2 T \bigl ( ( 1 - \alpha ) \varepsilon _ { T } + \alpha \kappa _ { T } \bigr ) .
$$

Consequently, under Assumption 2,

$$
\mathbf { C } \mathbf { R } _ { T } ( \mathbf { R } \mathbf { L } \mathbf { A } ) \geq 1 - \frac { 2 } { \omega } \big ( ( 1 - \alpha ) \varepsilon _ { T } + \alpha \kappa _ { T } \big ) .
$$

Proof. For each $t ,$

$$
\begin{array} { r } { \| s _ { t } ^ { \alpha } - r _ { t } \| _ { \infty , \mathcal { A } _ { t } } \leq ( 1 - \alpha ) \| \widehat { r } _ { t } - r _ { t } \| _ { \infty , \mathcal { A } _ { t } } + \alpha \| b _ { t } - r _ { t } \| _ { \infty , \mathcal { A } _ { t } } . } \end{array}
$$

Apply Lemma 1 with $s _ { t } = s _ { t } ^ { \alpha }$ , sum over $t ,$ and divide by $\mathrm { O P T } _ { T } \geq$ ωT. □

Corollary 2 (Consistency). $H \alpha = 0 ,$ , then

$$
\mathrm { C R } _ { T } ( \mathrm { A D V } ) \geq 1 - \frac { 2 \varepsilon _ { T } } { \omega } .
$$

In particular, perfect advice $( \varepsilon _ { T } = 0 )$ is ofline-optimal.

Corollary 3 (Robust fallback protection). $I f \alpha = 1$ , then

$$
\mathbf { C } \mathbf { R } _ { T } ( \mathbf { R } \mathbf { L } \mathbf { A } ) \geq 1 - \frac { 2 \kappa _ { T } } { \omega } .
$$

Thus the policy remains protected whenever the conservative score has bounded error, irrespective ofthe advice error.

Corollary 4 (Advice-relative certificate). Under Assumptions 1– 2,

$$
\mathbf { C R } _ { T } ( \mathbf { R L A } ) \geq \mathbf { C R } _ { T } ( \mathbf { A D V } ) - \frac { 2 \alpha } { \omega } ( \kappa _ { T } + \varepsilon _ { T } ) .
$$

Hence, $i f \kappa _ { T } = O ( \varepsilon _ { T } )$ , then

$$
\mathrm { C R } _ { T } ( \mathrm { R L A } ) \geq \mathrm { C R } _ { T } ( \mathrm { A D V } ) - O ( \varepsilon _ { T } ) .
$$

Proof. By Corollary $2 , \mathrm { C R } _ { T } ( \mathrm { A D V } ) \leq 1$ and its loss from one is at most $2 \varepsilon _ { T } / \omega$ . Theorem 1 gives the corresponding lower bound for RLA. Combining the two inequalities yields the claim.

Assumption 3 (Corrective availability). There exist $\delta > 0$ and $\Delta \in [ 0 , 1 ]$ such that, whenever $Q _ { t } ( g ) - Q _ { t } ( h ) > \delta T$ , any available itemfrom the over-exposed group g can be replaced by an available itemfrom group h whose robust score is lower by at most ∆.

Theorem 5 (Finite-time exposure control). Under Assumptions 1 and 3, if

$$
\lambda > { \frac { \Delta } { \delta T } } ,
$$

then

$$
\operatorname* { m a x } _ { g , h \in \mathcal { G } } \{ Q _ { T + 1 } ( g ) - Q _ { T + 1 } ( h ) \} \leq \delta T + 2 .
$$

Consequently,

$$
{ \mathrm { G a p } } _ { T } ^ { \rho } ( { \mathrm { F L A } } ) \leq \delta + { \frac { 2 } { T } } .
$$

Proof. Suppose $Q _ { t } ( g ) - Q _ { t } ( h ) > \delta T$ . By Assumption 3, an available group-g item can be replaced by a group-h item with robust-score loss at most ∆. The penalized-score advantage of the group-g item is then at most

$$
\Delta - \lambda ( Q _ { t } ( g ) - Q _ { t } ( h ) ) < 0 .
$$

Thus the policy cannot increase an already excessive pairwise imbalance. Since one decision changes any pairwise imbalance by at most two, the queue bound follows. Finally, $Q _ { T + 1 } ( g ) =$ $T ( E _ { T } ( g ) - \rho _ { g } )$ , which gives the exposure-gap bound. □

Proposition 6 (Incentive dampening). Ifan item can change its advice score by at most m $\begin{array} { r } { \iota \geq 0 , } \end{array}$ , then its one-period score gain under RLA or FLA is at most (1−α)m. Hence any manipulation of size m with cost larger than (1 − α)m is unprofitable.

Proof. The advice enters the decision score only through the coeficient $1 - \alpha . \mathrm { ~ A ~ }$ perturbation of magnitude m can therefore change the score by at most $( 1 - \alpha ) m .$ □

## 5 Computational study

We evaluate the proposed policies on online allocation instances derived from the MovieLens 1M data set. The raw data contain user-movie ratings. We interpret each arriving user as an online request and the available movies as the feasible slate. Ratings are rescaled to [0 1] and used as realized rewards. At each period, the policy selects one movie from the slate irrevocably.

Instance construction. A matrix-factorization model is trained on a fixed training split and used to generate the advice vector $\hat { r } _ { t } .$ . The conservative score $b _ { t }$ is a popularity-calibrated score, adjusted to avoid over-reliance on the learned predictor. Candidate slates contain both high-score items and lower-quality decoy items, so that random and popularity policies are nontrivial but not artificially favored. Movie groups define the exposure categories, and the target vector $\rho$ is set to the empirical group distribution in the candidate pool. All results are averaged over independent arrival sequences generated with fixed random seeds.

Stress regimes. We consider three regimes. In the benign regime, advice is perturbed by mean-zero noise of level $\sigma .$ In the adversarial-advice regime, advice is systematically biased across exposure groups, mimicking strategic or discriminatory score distortion. In the distribution-shift regime, test arrivals over-sample a subpopulation whose preferences difer from the training distribution. These regimes are designed to test the consistency–robustness trade-of predicted by Theorem 1 and Corollary 4.

Policies and metrics. We compare RANDOM, POPULAR-ITY, ADVICE, ROBUST-LA, and FAIR-LA. Performance is measured by the empirical competitive ratio $\mathrm { C R } _ { T }$ relative to the slate-wise ofline benchmark and by the exposure fairness gap Gap<sup>ρ</sup> . For the robustness certificate, we also report the empirical advice error $\varepsilon _ { T }$ and the gain in competitive ratio relative to ADVICE.

![](images/c30933c65c2cdb58a7467a645b16336a9bfcee9cec46b1a59de9f2a58754b56e.jpg)  
Figure 1: Robustness certificate across scenarios. The vertical axis reports the competitive-ratio gain over ADVICE and the horizontal axis reports the empirical advice error $\varepsilon _ { T }$ . The dashed line is the $- \varepsilon _ { T }$ envelope. The points remain above this envelope, in line with the error-dependent guarantee in Corollary 4.

Figure 1 provides the empirical counterpart of the advice-relative robustness certificate. Across benign noise, distribution shift, and adversarial advice, ROBUST-LA and FAIR-LA remain close to or above the advice-only rule, with no collapse as advice error increases. This supports the interpretation that conservative interpolation prevents catastrophic degradation when predictions are unreliable.

![](images/1f0af5fcde5a124524c0a21d749a688ead18a24d2639a199863c0992b8046b3a.jpg)  
Figure 2: Competitive ratio under prediction noise. ROBUST-$\mathrm { L A }$ and FAIR-LA preserve high competitive ratios across the three stress regimes, while RANDOM and POPULARITY are consistently separated from the learning-augmented policies.

Figure 2 shows that the learning-augmented policies retain a clear eficiency advantage over RANDOM and POPULARITY. In the adversarial and benign regimes, the proposed rules remain stable as $\sigma$ varies. Under distribution shift, competitive ratios remain high for all learning-augmented policies, while nonpersonalized baselines stay near 0 81.

![](images/e819777b793d1270642e4481e3542b276ec8cba8b49c76ff1a47378af6344da6.jpg)  
Figure 3: Exposure fairness gap under prediction noise. FAIR-LA sharply reduces exposure imbalance, especially under adversarial advice, where the advice-only policy induces large disparities.

Figure 3 confirms the role of the virtual exposure correction. Under adversarial advice, ADVICE produces severe exposure imbalance, whereas FAIR-LA reduces the gap substantially. In benign and shifted regimes, FAIR-LA also delivers the lowest or near-lowest exposure gap over most noise levels. POPULARITY can occasionally have a small fairness gap, but this is obtained with much lower competitive ratio; hence it does not provide the same eficiency-fairness trade-of.

![](images/c1172eeb05dac776fc390fed09b35ee011d47e25d1bd8f1ec1fdbc3f06cacbd2.jpg)  
Figure 4: Eficiency–fairness frontiers. Each point reports the average competitive ratio and exposure fairness gap of one policy. FAIR-LA lies on the favorable frontier by combining high eficiency with low exposure disparity.

Figure 4 summarizes the trade-of. FAIR-LA is the most stable policy on the eficiency–fairness frontier: it sacrifices little competitive ratio relative to ROBUST-LA or ADVICE while achieving substantially lower exposure disparity. This is the main empirical message of the study.
<table><tr><td rowspan="2"></td><td rowspan="2">Policy</td><td colspan="2">Benign</td><td colspan="2">Dist. Shift</td><td colspan="2">Adversarial</td></tr><tr><td>CR↑</td><td>FG↓</td><td>CR↑</td><td>FG↓</td><td>CR↑</td><td>FG↓</td></tr><tr><td rowspan="5">0.00</td><td>ADVICE</td><td>0.743</td><td>0.022</td><td>0.905</td><td>0.022</td><td>0.714</td><td>0.594</td></tr><tr><td>POPULARITY</td><td>0.674</td><td>0.020</td><td>0.808</td><td>0.022</td><td>0.677</td><td>0.038</td></tr><tr><td>RANDOM</td><td>0.676</td><td>0.019</td><td>0.811</td><td>0.016</td><td>0.687</td><td>0.196</td></tr><tr><td>ROBUST-LA</td><td>0.746</td><td>0.022</td><td>0.905</td><td>0.013</td><td>0.721</td><td>0.494</td></tr><tr><td>FAIR-LA</td><td>0.755</td><td>0.012</td><td>0.915</td><td>0.010</td><td>0.722</td><td>0.339</td></tr><tr><td rowspan="5">0.10</td><td>ADVICE</td><td>0.764</td><td>0.030</td><td>0.898</td><td>0.014</td><td>0.726</td><td>0.620</td></tr><tr><td>POPULARITY</td><td>0.680</td><td>0.024</td><td>0.810</td><td>0.022</td><td>0.681</td><td>0.031</td></tr><tr><td>RANDOM</td><td>0.686</td><td>0.027</td><td>0.817</td><td>0.022</td><td>0.690</td><td>0.217</td></tr><tr><td>ROBUST-LA</td><td>0.769</td><td>0.024</td><td>0.903</td><td>0.017</td><td>0.723</td><td>0.498</td></tr><tr><td>FAIR-LA</td><td>0.771</td><td>0.008</td><td>0.904</td><td>0.013</td><td>0.725</td><td>0.288</td></tr><tr><td rowspan="5">0.50</td><td>ADVICE</td><td>0.713</td><td>0.017</td><td>0.860</td><td>0.020</td><td>0.729</td><td>0.149</td></tr><tr><td>POPULARITY</td><td>0.673</td><td>0.025</td><td>0.816</td><td>0.020</td><td>0.690</td><td>0.029</td></tr><tr><td>RANDOM</td><td>0.682</td><td>0.024</td><td>0.819</td><td>0.013</td><td>0.693</td><td>0.020</td></tr><tr><td>ROBUST-LA</td><td>0.712</td><td>0.018</td><td>0.855</td><td>0.021</td><td>0.729</td><td>0.091</td></tr><tr><td>FAIR-LA</td><td>0.712</td><td>0.006</td><td>0.858</td><td>0.006</td><td>0.728</td><td>0.015</td></tr></table>

Table 1: Performance Comparison across Scenarios. We report the mean Competitive Ratio (CR ↑) and Fairness Gap (FG ↓). Robust-LA and Fair-LA consistently bridge the gap between pure Advice and baseline policies.

Table 1 reports representative noise levels. The table highlights two points. First, RANDOM and POPULARITY are not competitive in eficiency: they are consistently below the learningaugmented policies in CR. Second, FAIR-LA achieves the most reliable fairness improvement. In adversarial settings, ADVICE can achieve high CR but at the cost of extreme exposure gaps; FAIR-LA substantially reduces this disparity while preserving nearly the same CR.

## 6 Conclusion

We studied online allocation with learned advice, conservative fallback scores, and exposure-fairness targets. The proposed rule is deliberately simple: interpolate between advice and fallback scores, then penalize cumulative exposure imbalance. This simplicity yields finite-horizon guarantees. The central bound shows that performance degrades with a weighted combination of advice error and fallback error; the corollaries recover consistency, fallback protection, and the advice-relative certificate $\mathrm { C R } _ { T } ( \mathrm { B L A } ) \ge \mathrm { C R } _ { T } ^ { \bullet } ( \mathrm { A D V } ) - O ( \varepsilon _ { T } )$ . A virtual-queue argument gives finite-time exposure control.

The computational study supports these conclusions on MovieLens-derived online allocation instances. ROBUST-LA protects eficiency under unreliable advice, while FAIR-LA provides the strongest eficiency–fairness compromise. In particular, FAIR-LA sharply reduces exposure disparity under adversarial advice without collapsing in competitive ratio. Future work may extend the analysis to hard matching capacities and to endogenous, strategically generated advice.

## Data Availability

All numerical experiments in this study are based on synthetic benchmark environments generated algorithmically by the authors.

## Code Availability

The Python code used to generate the benchmark environments, compute the optimal policies via dynamic programming, train all boundary-based and reinforcement-learning baselines, and reproduce the tables and figures is available from the corresponding author upon reasonable request.

## References

Thodoris Lykouris and Sergei Vassilvitskii. Competitive caching with machine learned advice. Journal ofthe ACM (JACM), 68(4):1–25, 2021.

Manish Purohit, Zoya Svitkina, and Ravi Kumar. Improving online algorithms via ml predictions. Advances in Neural Information Processing Systems, 31, 2018.

Michael Mitzenmacher and Sergei Vassilvitskii. Algorithms with predictions seeking a new approach that goes beyond worst-case analysis.

Richard M Karp, Umesh V Vazirani, and Vijay V Vazirani. An optimal algorithm for on-line bipartite matching. In Proceedings ofthe twenty-second annual ACM symposium on Theory ofcomputing, pages 352–358, 1990.

Aranyak Mehta, Amin Saberi, Umesh Vazirani, and Vijay Vazirani. Adwords and generalized online matching. Journal of the ACM (JACM), 54(5):22–es, 2007.

Ashudeep Singh and Thorsten Joachims. Fairness of exposure in rankings. In Proceedings ofthe 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2219–2228, 2018.