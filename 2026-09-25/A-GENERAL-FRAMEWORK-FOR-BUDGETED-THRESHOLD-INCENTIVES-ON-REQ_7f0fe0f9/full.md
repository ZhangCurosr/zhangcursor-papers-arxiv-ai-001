# A GENERAL FRAMEWORK FOR BUDGETED THRESHOLD INCENTIVES ON REQUEST

Zhuolin Wu<sup>1</sup>, Chengrui Zhu<sup>1</sup>, Wenhua Nie<sup>1,2</sup>, Kenny Ye Liang<sup>1,3</sup>, Junming Lin<sup>1,4</sup>, Haiyang Li<sup>1</sup>, Zhilin Li<sup>1</sup>, Wenjia Geng<sup>1</sup>, Zeyu Wu<sup>1</sup>, Yinan Wu<sup>1</sup>, Jinghua Hao<sup>1</sup>, Renqing He<sup>1</sup>

<sup>1</sup>Meituan

<sup>2</sup>National Taiwan University

<sup>3</sup>Tsinghua University

<sup>4</sup>Tongji University

## ABSTRACT

On-demand delivery platforms pay riders through incentive activities whose tiers are set from recent completions of riders with a similar history, so that a little extra effort earns a clearly stated reward. Operators request such plans for changing periods, rider populations, payment rules and budgets, days ahead and within minutes, often for holidays or bad weather where randomized trials are scarce and take months to collect. We present a request-driven framework that composes four stages—conditional prediction, population reduction, trajectory integration and budget allocation—through seven replaceable modules that exchange conditional trajectory laws, whose award probabilities and award-marked moments give payment and uplift for any activity rule. To shorten a long randomized campaign, a response-correction step reweights trajectories from abundant no-offer history to match the moments of a short pilot. We prove that, on a fixed plan menu and given the stage errors, the end-to-end value loss is bounded by the sum of four stage terms—synthetic data, moment matching, integration and decision—with constants that cannot be improved from the final tables, and that for every stage there are instances on which omitting it leaves an error floor the others cannot remove. On 3,000 riders over 45 weekly origins, re-drawing trajectories and solving exactly for every request takes 117.8 s per week-long request against 3.07 s for the framework; including the one-off sampling pass, all 127 windows of a week are answered 11.04× faster with identical scenarios and at most 0.92% value lost by the allocation on the audited week-long tables, a gain that comes entirely from reuse, while a point forecast, independent days or an equal budget split each lose accuracy. Cities held out of training keep their plan-cost error within a pre-set margin (change +0.003 [−0.018, +0.020]). On 24 new controlled response laws, with the plan, the optimum and validity all judged within 20% of the budget, the response correction with a one-week pilot lowers regret by 51.2% relative to a trial with the same nominal randomized rider-weeks (53.6% in the stricter pre-specified 10% band), and on these week-long requests with exact summation a four-week pilot, with 4.5 times fewer randomized rider-weeks, comes within +0.007 [−0.012, +0.027] of an 18-week trial. In the registered primary conditions of two studies where windows, populations, rules and binding budgets change from request to request, the framework’s regret is below that of a trial with the same nominal rider-weeks and below that of dose interpolation of the same pilot data at every pilot length, and reusing its one-off preparation answers 60 requests 14.1× and 2.70× faster than re-running the pipeline for each, with identical answers. When the paid window or a selected subpopulation changes behaviour, a window-aware correction lower one-week regret by 0.195 against direct reuse. Against a nine-offer trial fitted with the framework’s own dose curve, one-week regret is 0.055 lower; 73.5% of the gain over per-offer fits of that trial comes from sharing the tilt across offers. Public retail data and three payment rules confirm the transfer with identical exact outputs.

## 1 INTRODUCTION

On-demand meal delivery depends on the people who carry each order from a restaurant to a customer. Large platforms obtain much of this capacity from crowdsourced couriers, usually called riders, who decide for themselves when to log in and how many orders to accept (Savelsbergh & Ulmer, 2022; 2024). Demand, by contrast, peaks at meal times, in bad weather and on holidays. When too few riders are active, orders go unaccepted and are cancelled; one large platform reports several hundred thousand such cancellations a day (Wu et al., 2022). Platforms therefore add budgeted incentive activities to per-order pay. A typical activity reads “complete 40 deliveries this week for an extra reward; complete 55 for more”, possibly with a per-completion rate that steps up at each tier or a condition on the number of active days. The tiers are set from recent completions (here, quantiles of the weekly volume of the rider’s layer), and this is what can make the activity effective. Threshold incentives change behaviour only while the threshold is still ahead (Liu et al., 2023), and effort is shaped by income targets (Chen et al., 2022; Allon et al., 2023): a tier the rider would reach anyway pays for no extra work, and a tier far out of reach is ignored. A well-placed tier sits just above the rider’s usual volume, so that a little extra effort, freely chosen, earns a clearly stated reward, with the aim of serving customers faster when service is scarce. The planning task is to place every tier at that point, for thousands of riders, within a fixed budget.

Operators express this task as a request. They name a rider population, an activity window (any subset of the coming days), an activity rule and a budget band, and they need a plan days before the window opens, within minutes. A week of n days has 2<sup>n</sup> windows, so one model per window is impossible and forecasts must be composable; payments are step functions of completions, attendance and thresholds, and rules differ between activities; and the campaigns that matter most fall on holidays and days of severe weather, where history is thinnest. Placing tiers also requires the response of completions to each candidate offer, and the clean way to measure it, a randomized controlled trial (RCT), is slow. An offer’s effect is observed only after its window has closed and settled, and each group–offer cell needs many rider-weeks: nine offers for five rider groups, with 1,024 randomized observations per cell at 512 riders per group and week, occupy 18 weeks, longer than the campaign the trial should inform. Experimental data are scarce while no-offer history is abundant, a tension well known in causal inference (Kallus et al., 2018; Athey et al., 2020; 2025).

We present a request-driven framework built from four stages and seven replaceable modules (Figure 1). Prediction (M1–M3) turns pre-offer histories into per-rider, per-day completion laws from which any window is composed. Reduction (M4) groups riders into a few activity layers, each receiving one plan, which reduces a portfolio search over individuals to a small multiple-choice problem. Integration (M5–M6) corrects the no-offer law for the offered incentive and compiles joint trajectories into award probabilities: every rule, however complex, reduces to probabilities of nested award events and marked moments, from which expected payment and uplift follow (Section 3). Allocation (M7) selects one plan per layer inside the budget band and returns executable rewards. Every element of the request is a hyperparameter of a downstream stage, so a new period, population, rule or budget reuses every upstream output whose inputs are unchanged. The correction in M5 is designed to shorten the long trial: abundant no-offer history supplies the shape of each group’s trajectory law, a pilot of one to four weeks at three offers supplies a few offer-specific moments, and history reweighted to match those moments stands in for every offer.

Contributions. (i) A general framework whose four stages exchange fixed intermediates—per rider-day laws, activity layers, award probabilities and per-layer plan menus—so that a new period, population, activity rule or budget is answered by changing hyperparameters while the response law it relies on stays valid, and each stage can be replaced and priced on its own (Sections 2–3). (ii) A history-anchored response correction from a short pilot, tested on controlled laws against a trial with the same nominal randomized rider-weeks and an 18-week trial, and in two registered studies in which windows, populations, rules and binding budgets change from request to request, one with a new rider-level generator whose response depends on the rule, and against stronger controls, including the trial fitted with the framework’s own dose curve, and behaviour-changing windows and subpopulations (Section 5). (iii) A four-term loss decomposition—synthetic data, moment matching, integration and decision—with a conditional end-to-end perturbation bound on a fixed plan menu, constants that cannot be improved from the final tables, lower bounds attained by the statistical chain itself, and, for each stage, instances on which omitting it leaves an error floor that the other stages cannot remove; with the ablations, this shows that every stage of the combination is needed (Section 3). (iv) An evaluation against the practices a platform would run instead, reporting the accuracy given up, the time saved and the accuracy on cities left out of training (Sections 4–6).

![](images/8e2895461715cbf641058b92df59716cd67d432a3c0f78ffb4272f4ffe31888e.jpg)  
Figure 1: Request-driven framework: four stages and seven modules. Dashed chips are inputs (request fields, candidate plans, activity rule, budget band, short pilot); arrows show data flow between modules. Predict (M1–M3) gives every eligible rider a conditional completion law; Reduce (M4) merges riders into activity layers with plan menus; Integrate (M5–M6) tilts the no-offer law to short-pilot moments and scores every plan by a weighted sum over stored samples of the corrected law, an exact sum when the trajectory types can be enumerated; Allocate (M7) chooses one plan per layer inside the two-sided budget band and returns the rewards that attain it. The bar names the four terms of the end-to-end bound (Proposition 1); the table marks what a new request recomputes.

## 2 REQUESTS AND THE FOUR-STAGE FRAMEWORK

Request. At issue time $t _ { 0 }$ a request $r = ( \mathcal { T } , W , x , \Gamma , B )$ names an eligible population $\mathcal { T } ,$ an activity window W (a subset of the forecast days), context x (meal periods, weather, supply and demand), an activity rule Γ and a budget set B. A partition $\mathcal { G } = \{ G _ { 1 } , \ldots , \bar { G } _ { L } \}$ of I assigns plan $a _ { l } \in \mathcal A _ { l }$ to group $l ;$ a plan specifies ordered thresholds, rewards or rates, eligible orders and attendance conditions. Let $Z _ { i } ( a )$ be rider $i \ ' s$ activity trajectory in $W$ under plan $a , g _ { i } ( Z _ { i } ( a ) , a )$ the payment defined by the rule, and $\dot { V } _ { i } ( a )$ an outcome such as completed deliveries. With $H _ { t _ { 0 } }$ the pre-offer history,

$$
c _ { l } ( a ) = \sum _ { i \in G _ { l } } \mathbb { E } [ g _ { i } ( Z _ { i } ( a ) , a ) \mid H _ { t _ { 0 } } , x ] , \qquad u _ { l } ( a ) = \sum _ { i \in G _ { l } } \mathbb { E } [ V _ { i } ( a ) - V _ { i } ( 0 ) \mid H _ { t _ { 0 } } , x ] ,\tag{1}
$$

where plan 0 is the no-offer baseline. The planner solves the multiple-choice problem

$$
\operatorname* { m a x } _ { a _ { l } \in A _ { l } } \sum _ { l } \widehat { v } _ { l } ( a _ { l } ) \quad \mathrm { s u b j e c t \ t o } \quad \sum _ { l } \widehat { c } _ { l } ( a _ { l } ) \in \mathcal { B } ,\tag{2}
$$

with $v _ { l } = u _ { l }$ for pure uplift, or v<sub>l</sub> adding a monetary score for the probability that group spend stays within ±20% of its prediction. The band is part of the request: operations tolerate $\breve { B } = \left\lceil 0 . 8 B , 1 . 2 \dot { B } \right\rceil$ a stricter [0.9B, 1.1B] keeps spend closer to the budget, and an upper cap $\boldsymbol { B } = [ 0 , \boldsymbol { B } ]$ is a special case.

Why four stages. Each stage removes one source of hardness. Per rider-day laws remove the $2 ^ { n }$ windows: one forecast composes any window. Layers remove the individual search: L layers with $| \mathcal { A } _ { l } |$ plans each leave $\Pi _ { l } | \ r _ { A _ { l } } |$ portfolios instead of one plan per rider. Award probabilities remove the dependence on the rule: every rule is integrated from the same trajectories. Budget allocation removes the coupling between layers: a single program over the band replaces a search over portfolios. The price of each removal is an approximation error. Section 3 shows that the errors of the estimation and allocation stages add at the allocation interface on a fixed menu; the price of reduction, restricting riders to layer menus, is measured rather than bounded.

Four stages, seven modules. The framework maps a request through fixed intermediates (Algorithm 1 in Appendix I lists the steps),

$$
( H _ { t _ { 0 } } , r ) ~ \xrightarrow { \mathrm { M 1 - M 3 } } ~ \widehat { P } _ { i , d } ~ \xrightarrow { \mathrm { M 4 } } ~ \mathcal { G } , \{ \mathcal { A } _ { l } \} ~ \xrightarrow { \mathrm { M 5 - M 6 } } ~ \{ \widehat { c } _ { l } ( a ) , \widehat { v } _ { l } ( a ) \} ~ \xrightarrow { \mathrm { M 7 } } ~ \widehat { a } .\tag{3}
$$

Prediction. M1 ranks riders by expected response, M2 selects the eligible roster and M3 forecasts a completion distribution for every rider and day. Day-level laws are coupled across days by reordering each rider’s forecast days according to the ranks observed in the rider’s own history, so that any window W is composed from one forecast pass. Reduction. M4 orders riders by recent activity, merges them into contiguous activity layers and shrinks each menu only in ways that cannot remove a feasible optimum: under a sole cap it drops a plan when another costs no more and is worth no less; under a band with a floor only when another plan of the same cost is worth no less, since a cheaper plan may fall below the floor (Lemma 1). Integration. M5 turns the no-offer law of each layer into an offer-conditioned trajectory law (below), and M6 integrates the rule over it as a weighted sum over stored trajectories, which is the exact sum when the trajectory types can be enumerated,

$$
( \widehat { c } _ { l } ( a ) , \widehat { v } _ { l } ( a ) ) = \sum _ { s } \bar { w } _ { s } \big ( g _ { l } ( Z _ { l } ^ { ( s ) } ( a ) , a ) , h _ { l } ( Z _ { l } ^ { ( s ) } ( a ) , a ) \big ) , \quad \bar { w } _ { s } \geq 0 , \quad \sum _ { s } \bar { w } _ { s } = 1 ,\tag{4}
$$

with $\bar { w } _ { s }$ the law’s probability of type s, or the normalized law weight of stored joint sample s when types cannot be enumerated. Allocation. M7 solves Eq. (2) by exact enumeration or by a dynamic program over a budget grid, a multiple-choice knapsack (Sinha & Zoltners, 1979); along affine reward paths the program runs over piecewise-affine score segments and returns the same objective and reward backpointers as dense exact allocation (Appendix I.1). Every returned plan carries reward that attain its reported payment.

Synthetic trajectories from a short pilot. Let $p _ { l } ^ { 0 } ( z )$ be the no-offer law of layer l over trajectory types z, estimated from history, and $\phi ( z )$ low-dimensional trajectory features (normalized completions and active days). For each piloted offer a the correction is the minimum relative-entropy law that reproduces the pilot moments $\bar { \phi } _ { l , a }$

$$
q _ { l , a } ( z ) \propto p _ { l } ^ { 0 } ( z ) \exp \bigl ( \lambda _ { l , a } ^ { \top } \phi ( z ) \bigr ) , \qquad \mathbb { E } _ { q _ { l , a } } [ \phi ] = \bar { \phi } _ { l , a } ,\tag{5}
$$

an exponential tilt solved by Newton’s method on the convex dual (Csiszár, 1975). Offers between pilot points interpolate λ piecewise linearly in the reward increment, with $\lambda = 0$ at the no-offer plan. Integration then sums the rule over ${ { q } _ { l , a } }$ by Eq. (4), with no redraw; redrawing S trajectories from $q _ { l , a }$ only adds Monte Carlo error and is an ablation. History fixes the shape of the law and the pilot fixes how the offer moves it, so a pilot of one to four weeks at three offers is meant to stand in for a tria over every offer (Section 5 measures how far it does).

Generality. A new window selects other day coordinates of the same forecast; a new population re-runs M2–M4 on stored laws; a new budget re-runs M7 only; a new rule changes the payment function in M6 and, when it changes behaviour, the pilot moments in M5. Reusing stored laws is exact for computation; reusing a response correction assumes the response is unchanged, so a new city, season or rule that may change behaviour calls for its own pilot. Any activity whose payment is a function of the trajectory enters through Eq. (4), so fixed prizes, per-completion rates, attendance awards and their combinations share one pipeline. Group-additive allocation assumes no response interference between groups; joint chance constraints would need an additional state in M7.

## 3 AWARD PROBABILITIES AND THE FOUR-TERM LOSS DECOMPOSITION

One intermediate for every rule. For one rider and plan, let $E _ { 1 } \supseteq \cdots \supseteq E _ { J }$ be nested award events, including attendance and eligibility conditions, with award probabilities $p _ { j } ~ = ~ \mathrm { P r } ( E _ { j } ~ |$ $a , H _ { t _ { 0 } } , x )$ and $p _ { J + 1 } = 0$ . If only the highest achieved fixed prize $R _ { j }$ is paid,

$$
\mathbb { E } [ g ] = \sum _ { j = 1 } ^ { J } R _ { j } ( p _ { j } - p _ { j + 1 } ) ;\tag{6}
$$

if the activity pays rate $\rho _ { j }$ on every eligible completion at the highest achieved tier,

$$
\mathbb { E } [ g ] = \sum _ { j = 1 } ^ { J } ( \rho _ { j } - \rho _ { j - 1 } ) \mathbb { E } [ Y \mathbf { 1 } \{ E _ { j } \} ] , \qquad \rho _ { 0 } = 0 ,\tag{7}
$$

with $Y$ the eligible completions. Award probabilities and award-marked moments therefore carry everything a rule needs; the rule itself selects which of them are integrated. Expected spend is additive over riders without any independence assumption, whereas its variance and the probability of staying in a band depend on the joint law, which is why M6 integrates joint trajectories rather than marginals: rules with equal expected payment and award probability can differ in budget risk (Appendix I).

Four-term loss decomposition. For every layer l and plan a the framework passes through a chain of trajectory laws: the true offer law $P ;$ the best synthetic law $P ^ { \mathrm { s y n } }$ , which applies Eq. (5) to the true no-offer law with population moments at the piloted offers and the framework’s own interpolation of λ elsewhere, so that the interpolation bias at unpiloted offers belongs to this term and does not shrink with the pilot; the moment-matched law $P ^ { \mathrm { m m } }$ , which uses the estimated no-offer law and the pilot moments; and the integrated law $P ^ { \mathrm { i n t } }$ , which is $P ^ { \mathrm { m m } }$ under exact summation and a finite-sample law otherwise (weighted stored samples, or $S$ redrawn trajectories in the sampled ablation). Law k induces tables $( c _ { l } ^ { ( k ) } , v _ { l } ^ { ( k ) } )$ , and allocation finally works with tables $( c ^ { \mathrm { d e c } } , v ^ { \mathrm { d e c } } )$ on its budget grid, which equal $( c ^ { \mathrm { i n t } } , v ^ { \mathrm { i n t } } )$ under exact enumeration. For $k \in \{ \mathrm { s y n }$ , mm, int, dec} with predecessor $k ^ { - }$ in this chain, let $\varepsilon _ { k } ^ { c } = \sum _ { l }$ m $\mathfrak { w } _ { a \in \mathcal { A } _ { l } } | c _ { l } ^ { ( k ) } ( a ) - c _ { l } ^ { ( k ^ { - } ) } ( a ) |$ , define $\varepsilon _ { k } ^ { v }$ in the same way, and put $\begin{array} { r } { E _ { c } = \sum _ { k } \varepsilon _ { k } ^ { c } } \end{array}$ and $\begin{array} { r } { E _ { v } = \sum _ { k } \varepsilon _ { k } ^ { v } } \end{array}$ . With $\mathcal { B } _ { \delta } = \left[ \underline { { b } } + \delta , \bar { b } - \delta \right]$ , let $V ^ { \star } ( \delta )$ be the best true value among portfolios whose true expected spend lies in $B _ { \delta }$ , and $\omega ( \delta ) \stackrel { \cdot } { = } V ^ { \star } ( 0 ) - V ^ { \star } ( \delta )$ the value carried by the outer margin δ of the band.

Proposition 1 (Four-term decomposition on a fixed menu). Suppose some portfolio has true expected spend in $\boldsymbol { B } _ { 2 E _ { c } } ,$ , and the planner returns ba with $c ^ { \mathrm { d e c } } ( \widehat { a } ) \ \in \ B _ { E _ { c } }$ and $\bar { v ^ { \mathrm { d e c } } } ( \widehat { a } ) \geq$ max $\{ v ^ { \mathrm { d e c } } ( a ) \ :$ $\bar { c } ^ { \mathrm { d e c } } ( a ) \in { \cal B } _ { E _ { c } } \bar  \} - \eta .$ (i) Upper bound. The true expected spend of ba lies in $B ,$ , and

$$
\begin{array} { r } { V ^ { \star } ( 0 ) - v ( \widehat { a } ) \ \leq \ \omega ( 2 E _ { c } ) + 2 \big ( \varepsilon _ { \mathrm { s y n } } ^ { v } + \varepsilon _ { \mathrm { m m } } ^ { v } + \varepsilon _ { \mathrm { i n t } } ^ { v } + \varepsilon _ { \mathrm { d e c } } ^ { v } \big ) + \eta . } \end{array}\tag{8}
$$

On afinite menu $V ^ { \star }$ is a stepfunction, so the relevant control is a discrete margin: $\omega ( 2 E _ { c } ) = 0$ when an optimal portfolio has true spend in $B _ { 2 E _ { c } }$ , and $i f \omega ( \delta ) \leq \kappa \delta$ on $[ 0 , 2 E _ { c } ]$ the right side is at most $\textstyle \sum _ { k } \mathrm { \hat { 2 } } ( \kappa \varepsilon _ { k } ^ { c } + \varepsilon _ { k } ^ { v } ) + \eta$ . (ii) Lower bound for tables. For any nonnegative stage errors there are table instances on afixed menu with exactly these errors on which Eq. (8) holds with equality and on which every deterministic planner that sees the final tables and keeps the true spend in B for all truths consistent with them loses at least $\omega ( 2 E _ { c } ) \dot { + } 2 E _ { v }$ . Without the tightening, the true spend can leave B by $E _ { c } . ~ ( i i i )$ Error floors of omitted stages. There are instances on which, with the other stages exact, omitting one stage leaves an error that their accuracy cannot reduce: without the pilot correction the entire uplift is invisible; two matched moments without the history shape leave the probability of reaching a threshold t standard deviations above the mean undetermined within $1 / ( 1 \dot { + } t ^ { 2 } ) ,$ ; plug-in means instead oftrajectory integration misprice a threshold prize by half; and an equal budget split canforfeit the entire value.

The proposition is a conditional perturbation bound on true expected spend, given the stage errors and the tightened band. It is relative to the layer menus: the value given up by restricting individuals to layer menus enters only through $V ^ { \star }$ , and on rider records alternative layerings are compared directly (Table 2). Part (ii) shows that no planner seeing only the final tables can improve the constants. For the chain of laws itself, the I-projection attains the constant of the synthetic term exactly, and the pilot and sampling terms are attained in order but not in constant (Proposition 3).

Why the combination is accurate and fast. Costs and pure-uplift values are expectations, hence linear in the trajectory law, so errors of the four stages add at the allocation interface instead of compounding: a value error costs at most twice its size and a cost error at most 2κ times its size under the growth control $\omega ( \delta ) \leq \kappa \delta \left( \mathrm { E q . } \left( 8 \right) \right)$ . Each term is driven by a different resource and shrinks independently: the synthetic term by the part of the response that the matched features do not explain, the moment term as $n ^ { - 1 / 2 }$ in pilot size for a given base (an estimated base adds the term of Lemma 6), the integration term as $S ^ { - 1 / 2 }$ and to zero under exact summation, and the decision term by one grid step per layer with upward rounding, half a step with nearest rounding (Appendix J). Corollary 1 states the rates in expectation: they hold for entries with a fixed integrand, such as uplift and payment, and, under the margin condition of Corollary 1(b), for a stability score centred at the estimated mean. The stage errors belong to the compiled tables, not to a request, so every budget and weight answered from the same tables inherits the same bound; this is why paying for the upstream stages once gives large speed-ups at no additional loss. On a given instance errors may partly cancel, so the ablations below measure what each stage buys in practice.

## 4 RIDER RECORDS: ACCURACY GIVEN UP AND TIME GAINED

Data and protocol. The rider panel holds daily completed deliveries of 3,000 riders over 595 days in 2025–2026 across 17 separately modelled cities and one pooled node. We replay the framework at 45 weekly origins with three sampling seeds: models are refit at each origin on data before it, and each seven-day window is planned before its outcomes are read. Plans combine three tier shapes, six active-day requirements and six reward levels, 108 plans per layer, so five activity layers give $1 0 8 ^ { 5 } \approx 1 . 5 \stackrel { \cdot } { \times } 1 0 ^ { 1 0 }$ portfolios; budgets are upper caps at $\beta = 0 . 5 , 1 . 0$ and 1.5 times the realized cost of a reference plan, so they are set in hindsight. Because every plan is scored against the completions riders actually recorded, the replay measures what a plan must get right before any behavioural response: who attains which tier and what the plan costs. A plan’s value is the number of deliveries completed by riders who reach an award, its cost the reward paid; plan-cost error is the median absolute relative error between predicted and realized plan cost over plans with nonzero realized cost (the 810 of 72,900 plans that pay nothing are excluded), and regret is measured against the exact optimum on the realized records. Recorded completions do not respond to a plan, so value does not depend on the reward level, and the budget binds in hindsight only in the Spring Festival week (3 of 135 units at $\beta = 0 . 5 )$ . The replay therefore tests pricing and computation (M3: gradient-boosted quantiles with rider shares, Appendix A); Section 5 tests the choice of reward level and allocation under a binding budget. Intervals resample origins (Appendix A).

What the framework is compared with. We compare with the most accurate practices a platform would otherwise run, each exact or unbiased for the step it replaces: re-drawing trajectories for the requested window and solving exactly, request by request; one exact plan per rider; a mixed-integer program per request (Sinha & Zoltners, 1979) solved to optimality (Huangfu & Hall, 2018); and a trial long enough to observe every offer. Table 1 reports the accuracy given up and the time saved.

Near-lossless and much faster. Against re-drawing and solving exactly for every request, the framework gives up nothing in the trajectories and almost nothing in the allocation, while a week-long request takes 3.07 s instead of 117.8 s. The 3.07 s follow a one-off sampling pass of 114.2 s per origin, the only cold-start cost. Charging that pass to the framework, eight windows are answered 1.81× faster and all 127 windows of a week take 12.2 instead of 131.1 minutes, 11.04× faster (Figure 2b; 9 timed origins, fixed before the run); solving one plan per rider is out of reach, which is what population reduction buys. The gain is reuse of the sampling pass, and it follows from the interface: the per-day laws are fixed before a request arrives, so one forecast serves every request (Table 5).

Every stage earns its place. Table 2 replaces one stage at a time by the cheaper practice it displaces. Distributional prediction is the largest single gain: a point forecast raises plan-cost error from 0.215 to 0.660 and median award-rate error from 4.0 to 66.9 percentage points. Joint integration is next: composing days independently raises plan-cost error by 1.51×, and a simplified normal baseline defined here (window totals normal, tiers priced without the truncated-moment term) by 2.46×; its award rates for five- and six-day plans fall 10.31 points below those of the joint law, exactly where attendance persists across days. Equal-width buckets change the layer menus and raise regret against the same hindsight optimum from 0.37% to 0.78%, and splitting the budget equally across layers gives up 15.7% of the value found by the budget program. On the realized five-layer records the grid program equals the exact optimum in 134 of 135 main-grid instances (largest loss 0.69%; Appendix I.1).

Cities left out of training. The replay refits every model weekly on all cities, as the framework does in operation. A held-out condition fixed before the runs removes the evaluated cities from training (Appendix A.1): cities removed in four folds (45 origins) and forecast from their riders’ own history keep plan-cost error at 0.343 against 0.340 for the reference that trains on them (paired change +0.003 [−0.018, +0.020]), within the pre-set margin of 0.045. On these riders every stage replacement again loses accuracy (Table 8).

Table 1: Accuracy given up and time gained against the most accurate way to answer the same request without the framework, on the same inputs. Speed-up: median of paired ratios (mixed-integer program: ratio of geometric-mean times); brackets: interquartile range or 95% interval over laws.
<table><tr><td>Alternative per request</td><td>Accuracy the framework gives up</td><td>Time: alternative → framework</td><td>Speed-up</td></tr><tr><td colspan="2">Rider records: 3,000 riders, 45 weekly origins, five activity layers Re-draw trajectories and none in the scenarios (1,431/1,431</td><td colspan="2">117.8 s → 3.07 s per</td></tr><tr><td>solve exactly</td><td>requests identical); allocation loses value in 3/405 instances, at most 0.92%</td><td>week-long request</td><td>38.6× [30.8, 43.0]; 11.0× over 127 requests with the one-off pass</td></tr><tr><td>One plan per rider, exact not measurable: the exact plan did</td><td>not finish within 3 hours and 190 GiB per table of memory</td><td>unfinished → 4.05 s</td><td></td></tr><tr><td>Controlled response laws: 18 laws × 8 replicates, 33 budgets each (Section 5) Randomized trial over all offers</td><td>relative regret (planned and scored within 20% of budget) 0.113 → 0.121 nominal randomized (+0.007 [−0.012, +0.027]) with a 4-week pilot; a trial with the pilot&#x27;s rider-weeks: 0.421 with 1 week (framework 0.205) and 0.225 with 4</td><td>9,216 → 2,048 rider-weeks per group (18 → 4 weeks)</td><td>4.5× fewer</td></tr><tr><td colspan="2">Public retail (M5) and three payment rules (Section 6) Re-draw and solve</td><td>84.8 s → 4.36 s per</td><td>20.0× [18.9, 21.2];</td></tr><tr><td>exactly (M5, 3,000 series)</td><td>none: identical scenarios, value lost in 0/405 tables</td><td>request</td><td>7.11× over 127 with preparation 1, 9, 33 requests:</td></tr><tr><td>Mixed-integer program per request, model kept between requests (396 conditions)</td><td>none: the budget program is exact on paired on conditions 65,340/65,340 requests; the MIP on1 65,315 (raw) and 65,335 (envelope)</td><td>both solve exactly</td><td>1.08, 7.6, 20.9× (envelope 1.06, 2.5, 5.5×)</td></tr></table>

Table 2: Rider records, 45 weekly origins × 3 seeds, five activity layers. Each row replaces one stage of the full framework by a cheaper alternative. Ratios are of origin means with 95% intervals over origins. Regret at budget fraction 0.5. <sup>∗</sup>45 origins × 3 budget fractions.
<table><tr><td>Stage</td><td>Full framework</td><td>Cheaper alternative</td><td>Accuracy change</td></tr><tr><td></td><td>Prediction per rider-day distribution point forecast</td><td></td><td>plan-cost error ×3.08 [2.48, 3.81]</td></tr><tr><td></td><td>Reduction activity layers</td><td>equal-width buckets</td><td>regret 0.37% → 0.78%</td></tr><tr><td></td><td>Integration joint law across days</td><td>independent days simplified normal</td><td>plan-cost error ×1.51 [1.33, 1.74] ×2.46 [2.02, 3.01]; 5–6-day award -10.31 pp</td></tr><tr><td></td><td></td><td>plug-in mean</td><td>×1.85 [1.58, 2.19]</td></tr><tr><td></td><td>Allocation budget-grid program</td><td>equal budget split exact solver (reference) loss 0 in 134/135*; max 0.69%</td><td>value -15.7% [13.8, 17.4]</td></tr></table>

## 5 SYNTHETIC TRAJECTORIES AND A SHORT PILOT VERSUS A LONG TRIAL

Design. Behavioural response cannot be read from logged completions, so we test the responsecorrection step M5 on controlled response laws where the truth is known; every other stage is fixed. We draw 24 new laws in four families of six (saturating, threshold and hump responses combined; saturating only; threshold only; none), each with five rider groups under three payment rules, nine offers per group and 32 weekly trajectory types; allocation enumerates all 9<sup>5</sup> portfolios for 33 budgets, and every request covers the whole week and all five groups. Every replicate (eight per law) gives each method its own data: 4,096 no-offer weeks per group; a long RCT of 1,024 randomized observations per group and offer, which at 512 riders per group and week takes 18 weeks; a same-calendar RCT that spreads the rider-weeks of a pilot evenly over all nine offers; and the framework, which runs the pilot at three offers for one, two or four weeks, corrects history by Eq. (5) and, since the 32 types can be enumerated, sums the rule exactly over the corrected law, which removes the integration term (Lemma 7); drawing $S = 2 \AA , 0 4 8$ trajectories instead is an ablation. Samples are independent across indices, each with one weekly shock shared by all cells; weeks convert nominal rider-weeks at 512 riders per group. The endpoint is relative regret of pure uplift against the true optimum in the request’s band; an abstention scores zero, and so does a plan whose true expected spend leaves the band. We report the operating band [0.8B, 1.2B], in which every method plans and is scored and the optimum is taken (chosen after the runs), and the pre-specified, stricter [0.9B, 1.1B] alongside, which the appendix tables of this design use. Laws are the unit of inference (18 laws with a response; the 6 null laws check validity only), with 95% intervals by bootstrap over laws. Arms were specified before the runs, after one disclosed development run whose exclusion changes no conclusion (Appendix B).

Table 3: Relative regret (lower is better), 18 response laws × 8 replicates × 33 budgets, pilot of 1/2/4 weeks; planning, optimum and scoring within 20% of the budget or 10% (pre-specified). Indented rows change one stage (others: Table 9). Validity (%): issued plans with true expected spend in the 10% band, all 24 laws.
<table><tr><td rowspan="2">Method</td><td colspan="3">Within 20% of budget</td><td colspan="3">Within 10% (pre-specified)</td></tr><tr><td>1 wk</td><td>2wk</td><td>4wk</td><td>1 wk</td><td>2wk</td><td>4wk</td></tr><tr><td>Long RCT, 18 weeks (business alternative)</td><td></td><td>0.113</td><td></td><td></td><td>0.130</td><td></td></tr><tr><td>Same-calendar RCT</td><td>0.421</td><td>0.313</td><td>0.225</td><td>0.542</td><td>0.383</td><td>0.264</td></tr><tr><td>Framework (tilt, exact sum)</td><td>0.205</td><td>0.159</td><td>0.121</td><td>0.252</td><td>0.186</td><td>0.136</td></tr><tr><td>sampled integration  $( S = 2 , 0 4 8 )$ </td><td>0.245</td><td>0.197</td><td>0.163</td><td>0.286</td><td>0.223</td><td>0.189</td></tr><tr><td>without history (pilot frequencies)</td><td>0.603</td><td>0.560</td><td>0.527</td><td>0.658</td><td>0.611</td><td>0.569</td></tr><tr><td>Validity (framework / same-calendar)</td><td></td><td></td><td></td><td>80.3/53.2</td><td>83.5/69.0</td><td>87.5/76.5</td></tr></table>

Table 4: Two registered studies with changing requests and binding budgets (registered primary condition, 36 response laws each; relative regret within 20% of the budget). A: fresh laws from the generator of Table 3; B: a new rider-level generator with a rule-dependent response. Speed-up: 60 requests, one-off preparation charged, against re-running the pipeline per request.
<table><tr><td rowspan="2">Pilot</td><td colspan="3">A: integrated study</td><td colspan="3">B: new generator</td></tr><tr><td>1 wk</td><td>2wk</td><td>4wk</td><td>1wk</td><td>2wk</td><td>4wk</td></tr><tr><td>Same-calendar RCT</td><td>0.391</td><td>0.296</td><td>0.216</td><td>0.477</td><td>0.326</td><td>0.240</td></tr><tr><td>Framework</td><td>0.220</td><td>0.172</td><td>0.148</td><td>0.188</td><td>0.150</td><td>0.116</td></tr><tr><td>Dose interpolation (same information)</td><td>0.227</td><td>0.177</td><td>0.154</td><td>0.210</td><td>0.171</td><td>0.131</td></tr><tr><td>Independent days (least costly replacement)</td><td>0.311</td><td>0.292</td><td>0.296</td><td>0.198</td><td>0.158</td><td>0.124</td></tr><tr><td>Budgets that bind (%, all 48 laws)</td><td></td><td>72.3</td><td></td><td></td><td>75.0</td><td></td></tr><tr><td>Speed-up, 60 requests</td><td>14.1×</td><td>14.2×</td><td>14.3×</td><td>2.70×</td><td>2.72×</td><td>2.78×</td></tr></table>

Results. Table 3 gives three findings; the pre-specified 10% band is in parentheses and is the one plotted in Figure 2a (Appendix B). First, for the same nominal randomized rider-weeks the framework is far better than randomizing the pilot over all offers: regret falls by 0.215 [−0.261, −0.173] with a one-week pilot, 51.2% lower (53.6%), by 0.155 [−0.198, −0.115] with two weeks and by 0.105 $[ - 0 . 1 4 0 , - \bar { 0 } . 0 7 1 ]$ with four (at 10%, intervals also below zero). Second, with exact summation on these week-long requests, a four-week pilot, with 4.5 times fewer randomized rider-weeks, comes within $+ 0 . 0 \bar { 0 } 7 [ \bar { - } 0 . 0 1 2 , + 0 . 0 2 7 ]$ of the 18-week trial $( + 0 . 0 0 6 [ - 0 . 0 1 3 , + 0 . 0 2 4 ] ) \colon$ ; sampled integration, the pre-specified arm: Table 22. Third, each stage is needed against the replacement tested for it (10% scoring): dropping history, moment matching or joint integration raises one-week regret by $+ 0 . 4 0 6 , + 0 . 4 0 4 \ \mathrm { { a n d } + 0 . \bar { 3 } 3 1 }$ , and an equal budget split on the exact tables raises it by +0.610, each interval excluding zero; drawing 2,048 trajectories instead of summing exactly raises it by +0.039 $\left[ + 0 . 0 1 7 , + 0 . 0 6 { \bar { 4 } } \right] ( + 0 . 0 3 5 \left[ + { \bar { 0 . } } 0 1 0 , + 0 . { \bar { 0 } } 6 1 \right] )$ . With the truth known, only the moment-matching term of Eq. (8) shrinks with the pilot, at the $\sqrt { 2 }$ rate per doubling of Corollary 1(a) (Table 10).

Changing requests, binding budgets and a new generator. The design above fixes the request. Two further registered studies let behavioural response, changing requests and binding budgets occur together (Table 4; Appendices C and D). Each replicate issues 60 requests with a random window among the 127 subsets of the week, a random population and rule, and budgets at 0.3, 0.5 and 0.7 times the cost of the request’s unconstrained optimum; the framework prepares its laws once per origin, and the preparation is charged. Study A draws 48 fresh laws from the generator above. Study B uses a rider-level generator unrelated to it: six groups of 16 latent rider types with day-of-week activity, a response that depends on the rule shown, and weeks that cannot be enumerated, so every request is integrated as a weighted sum over 4,096 stored weeks per group. In the registered primary condition of both studies, both registered hypotheses hold (Table 22): regret is below the samecalendar trial’s at every pilot length (by −0.171 [−0.201, −0.140] and −0.289 [−0.311, −0.267] with one week), and reusing the prepared laws for 60 requests is faster than re-running the pipeline for each, with identical answers. Every stage replacement loses accuracy, and dose interpolation of the same history and pilot data, a same-information control, has higher regret at every pilot length $( + 0 . 0 0 7 \ [ + 0 . 0 \dot { 0 } 4 , + 0 . 0 1 0 ] \ \mathrm { a n d } + 0 . 0 2 2 \ [ + 0 . 0 1 8 , + 0 . 0 2 7 ]$ with one week). In three more registered studies all hypotheses hold (Appendix E): at matched cache the framework beats a history-anchored tilt of the trial fitted per offer and a normal law with exact truncated moments, it beats the trial under shared calendar-week shocks, and request-aware reuse beats direct reuse when windows or subpopulations change behaviour. The third fits the trial jointly with the framework’s own dose curve (Table 17): at the same parameterization the framework’s regret is lower by 0.055 [0.051, 0.059] at one week (0.052 and 0.060 at two and four), and of its one-week advantage over the per-offer tilt, sharing the tilt across offers accounts for 73.5% [72.1, 74.8] and placing the pilot at three offers for 26.5% (68.3% at four weeks). In a fourth, five history layers retain 94.6% [94.4, 94.8] of an individual-level optimum, against 82.5% for one plan per group.

## 6 TRANSFER ACROSS DOMAINS AND RULES

The same modules serve other domains and rules by changing inputs and hyperparameters only; both sides of every comparison get the same laws, menus and budget bands, and every output is checked against exact allocation (Appendix G). M5 sales (Makridakis et al., 2022a;b) follow the same rule as a supplier’s tiered rebate to a store. Designed on riders and run unchanged on 3,000 item–store series, law reuse cuts the median request from 84.8 s to 4.36 s (paired 20.0×) with identical scenarios and identical exact values on all 405 tables; charging the 80.63 s preparation to the framework, it is 1.66×, 3.92× and 7.11× faster for 8, 32 and 127 requests per origin. An active-day rule reuses the same laws through joint histograms, with all 14,580 answers identical to direct settlement. Across fixed prizes, per-completion rates and attendance awards (396 conditions), the budget program reuses its value curve and resolves all 65,340 requests exactly, whereas a mixed-integer program per request leaves 25 (raw) and 5 (envelope) unresolved and is 20.9× and 5.5× slower at 33 requests (Table 1).

## 7 RELATED WORK

Response or uplift models feed budgeted allocation in marketing, sometimes trained with the decision or summarized by budget–value curves (Zhao et al., 2019; Albert & Goldenberg, 2022; Ai et al., 2022; Zhou et al., 2024; Sun et al., 2024; Zhang et al., 2025; Yang et al., 2026; Cong et al., 2025); delivery bonuses, driver subsidies and rider incentives are allocated in the same spirit (Wu et al., 2022; Chen et al., 2024; Yang et al., 2025; Chen et al., 2026; 2025). Here the decision unit is a multi-day threshold activity whose payment depends on the whole trajectory, and every request may change window, population, rule and budget. Our allocation stage extends multiple-choice knapsacks and graphical Bellman algorithms (Sinha & Zoltners, 1979; Kameshwaran & Narahari, 2009; Gafarov et al., 2014) to two-sided budget bands and reward traceback. Where small experiments are combined with observational data to estimate effects (Kallus et al., 2018; Athey et al., 2020; 2025), our pilot corrects history for planning by entropy balancing (Hainmueller, 2012) on offer-conditioned trajectories coupled across days by rank (Clark et al., 2004; Schefzik et al., 2013), and its error is priced as its own term of an end-to-end bound.

## 8 CONCLUSION

Four stages exchanging conditional trajectory laws answer changing requests while the response law stays reusable; a window-aware correction covers behaviour-changing windows and subpopulations.

The four-term decomposition prices every stage, every cheaper replacement loses accuracy, and reuse answers all windows of a week 11.04× faster at almost no loss. In registered controlled studies the framework beats equal-effort trials, same-information controls and the trial fitted with its own dose curve. Rider records support pricing and computation; response evidence comes from controlled studies.

## REFERENCES

Meng Ai, Biao Li, Heyang Gong, Qingwei Yu, Shengjie Xue, Yuan Zhang, Yunzhou Zhang, and Peng Jiang. LBCF: A large-scale budget-constrained causal forest algorithm. In Proceedings ofthe ACM Web Conference 2022 (WWW ’22), pp. 2310–2319, 2022. doi: 10.1145/3485447.3512103.

Javier Albert and Dmitri Goldenberg. E-commerce promotions personalization via online multiplechoice knapsack with uplift modeling. In Proceedings ofthe 31st ACM International Conference on Information & Knowledge Management (CIKM ’22), pp. 2863–2872, 2022. doi: 10.1145/ 3511808.3557100. arXiv:2108.13298.

Gad Allon, Maxime C. Cohen, and Wichinpong Park Sinchaisri. The impact of behavioral and economic drivers on gig economy workers. Manufacturing & Service Operations Management, 25 (4):1376–1393, 2023. doi: 10.1287/msom.2023.1191.

Susan Athey, Raj Chetty, and Guido Imbens. Using experiments to correct for selection in observational studies. arXiv preprint arXiv:2006.09676, 2020. Revised May 2025.

Susan Athey, Raj Chetty, Guido W. Imbens, and Hyunseung Kang. The surrogate index: Combining short-term proxies to estimate long-term treatment effects more rapidly and precisely. The Review ofEconomic Studies, 93(4):2284–2312, 2025. doi: 10.1093/restud/rdaf087. Published online 30 September 2025; July 2026 issue.

Bobby Chen, Siyu Chen, Jason Dowlatabadi, Yu Xuan Hong, Vinayak Iyer, Uday Mantripragada, Rishabh Narang, Apoorv Pandey, Zijun Qin, Abrar Sheikh, Hongtao Sun, Jiaqi Sun, Matthew Walker, Kaichen Wei, Chen Xu, Jingnan Yang, Allen T. Zhang, and Guoqing Zhang. Practical marketplace optimization at Uber using causally-informed machine learning. In 2nd Workshop on Causal Inference and Machine Learning in Practice, KDD 2024, 2024. arXiv:2407.19078.

Juhua Chen, Karson Shi, Jialing He, North Chen, and Kele Jiang. MMCE: A framework for deep monotonic modeling of multiple causal effects. arXiv preprint arXiv:2504.03753, 2025.

Taijie Chen, Rui Su, Siyuan Feng, Laoming Zhang, Hongyang Zhang, Haijiao Wang, Zhaofeng Ma, Jintao Ke, and Li Ma. D<sup>3</sup>-Subsidy: Online and sequential driver subsidy decision-making for large-scale ride-hailing market. arXiv preprint arXiv:2605.20036, 2026.

Xirong Chen, Zheng Li, Liu Ming, and Weiming Zhu. The incentive game under target effects in ridesharing: A structural econometric analysis. Manufacturing & Service Operations Management, 24(2):972–992, 2022. doi: 10.1287/msom.2021.1002.

Martyn Clark, Subhrendu Gangopadhyay, Lauren Hay, Balaji Rajagopalan, and Robert Wilby. The Schaake shuffle: A method for reconstructing space–time variability in forecasted precipitation and temperature fields. Journal ofHydrometeorology, 5(1):243–262, 2004.

Yu Cong, Chao Xu, and Yi Zhou. Large-scale trade-off curve computation for incentive allocation with cardinality and matroid constraints. In Proceedings ofthe Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25), pp. 2575–2582, 2025. doi: 10.24963/ijcai.2025 287.

Imre Csiszár. I-divergence geometry of probability distributions and minimization problems. The Annals ofProbability, 3(1):146–158, 1975. doi: 10.1214/aop/1176996454.

Miroslav Dudík, John Langford, and Lihong Li. Doubly robust policy evaluation and learning. In Proceedings ofthe 28th International Conference on Machine Learning (ICML), 2011.

Evgeny R. Gafarov, Alexandre Dolgui, Alexander A. Lazarev, and Frank Werner. A graphical approach to solve an investment optimization problem. Journal ofMathematical Modelling and Algorithms in Operations Research, 13(4):597–614, 2014. doi: 10.1007/s10852-013-9248-2.

Evgeny R. Gafarov, Alexandre Dolgui, Alexander A. Lazarev, and Frank Werner. A new effective dynamic program for an investment optimization problem. Automation and Remote Control, 77(9): 1633–1648, 2016. doi: 10.1134/S0005117916090101.

Jens Hainmueller. Entropy balancing for causal effects: A multivariate reweighting method to produce balanced samples in observational studies. Political Analysis, 20(1):25–46, 2012.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal ofthe American Statistical Association, 58(301):13–30, 1963. doi: 10.1080/01621459.1963.10500830.

Q. Huangfu and J. A. J. Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10(1):119–142, 2018. doi: 10.1007/s12532-017-0130-5. URL https://doi.org/10.1007/s12532-017-0130-5.

Nathan Kallus, Aahlad Manas Puli, and Uri Shalit. Removing hidden confounding by experimental grounding. In Advances in Neural Information Processing Systems 31 (NeurIPS 2018), 2018. URL https://proceedings.neurips.cc/paper/2018/ hash/566f0ea4f6c2e947f36795c8f58ba901-Abstract.html.

S. Kameshwaran and Y. Narahari. Nonconvex piecewise linear knapsack problems. European Journal of Operational Research, 192(1):56–68, 2009. doi: 10.1016/j.ejor.2007.08.044.

Tianming Liu, Zhengtian Xu, Daniel Vignon, Yafeng Yin, Qingyang Li, and Zhiwei Qin. Effects of threshold-based incentives on drivers’ labor supply behavior. Transportation Research Part C: Emerging Technologies, 152:104140, 2023. doi: 10.1016/j.trc.2023.104140.

Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. M5 accuracy competition: Results, findings, and conclusions. International Journal ofForecasting, 38(4):1346–1364, 2022a.

Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, Zhi Chen, Anil Gaba, Ilia Tsetlin, and Robert L. Winkler. The M5 uncertainty competition: Results, findings and conclusions. International Journal of Forecasting, 38(4):1365–1385, 2022b.

Martin Savelsbergh and Marlin W. Ulmer. Challenges and opportunities in crowdsourced delivery planning and operations—an update. Annals ofOperations Research, 343(2):639–661, 2024. doi: 10.1007/s10479-024-06249-1.

Martin W. P. Savelsbergh and Marlin W. Ulmer. Challenges and opportunities in crowdsourced delivery planning and operations. 4OR, 20(1):1–21, 2022. doi: 10.1007/s10288-021-00500-2.

Roman Schefzik, Thordis L. Thorarinsdottir, and Tilmann Gneiting. Uncertainty quantification in complex simulation models using ensemble copula coupling. Statistical Science, 28(4):616–640, 2013.

Prabhakant Sinha and Andris A. Zoltners. The multiple-choice knapsack problem. Operations Research, 27(3):503–515, 1979. doi: 10.1287/opre.27.3.503.

Zexu Sun, Hao Yang, Dugang Liu, Yunpeng Weng, Xing Tang, and Xiuqiang He. End-to-end costeffective incentive recommendation under budget constraint with uplift modeling. In Proceedings of the 18th ACM Conference on Recommender Systems (RecSys ’24), pp. 560–569, 2024. doi: 10.1145/3640457.3688147. arXiv:2408.11623.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, New York, 2009. doi: 10.1007/b13794.

Zhuolin Wu, Li Wang, Fangsheng Huang, Linjun Zhou, Yu Song, Chengpeng Ye, Pengyu Nie, Hao Ren, Jinghua Hao, Renqing He, and Zhizhao Sun. A framework for multi-stage bonus allocation in meal delivery platform. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22), pp. 4195–4203, 2022. doi: 10.1145/3534678.3539202. arXiv:2202.10695.

Jiaqi Yang, Lexiao Chen, Zicheng Su, Wanjing Ma, Zhichao Zou, and Kun An. Decision-focused learning for optimal subsidy allocation in ride-hailing services. Transportation Research Part C: Emerging Technologies, 180:105301, 2025. doi: 10.1016/j.trc.2025.105301.

Siyun Yang, Shixiao Yang, Jian Wang, Di Fan, Kehe Cai, Haoyan Fu, Jiaming Zhang, Wenjin Wu, and Peng Jiang. Jointly optimizing debiased CTR and uplift for coupons marketing: A unified causal framework. arXiv preprint arXiv:2602.12972, 2026. Accepted at CIKM 2026 (arXiv v3 comment).

Shuli Zhang, Hao Zhou, Jiaqi Zheng, Guibin Jiang, Bing Cheng, Wei Lin, and Guihai Chen. Bi-level decision-focused causal learning for large-scale marketing optimization: Bridging observational and experimental data. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), 2025. arXiv:2510.19517.

Kui Zhao, Junhao Hua, Ling Yan, Qi Zhang, Huan Xu, and Cheng Yang. A unified framework for marketing budget allocation. In Proceedings ofthe 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19), pp. 1820–1830, 2019. doi: 10.1145/3292500. 3330700. arXiv:1902.01128.

Hao Zhou, Rongxiao Huang, Shaoming Li, Guibin Jiang, Jiaqi Zheng, Bing Cheng, and Wei Lin. Decision focused causal learning for direct counterfactual marketing optimization. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), pp. 6368–6379, 2024. doi: 10.1145/3637528.3672353. arXiv:2407.13664.

## A RIDER REPLAY: PROTOCOL AND TIMING

Panel and calendar. The panel holds daily completed deliveries of 3,000 riders from 2025-01-01 to 2026-08-18 (595 days), recorded per rider, city and day; a missing rider-day means no recorded completion. Seventeen cities with enough riders are modelled as separate nodes and the rest are pooled. No feature uses information from after the origin. Forecast origins are 45 consecutive weeks; every model is refit at each origin on data up to the origin, and a request window is any subset of the seven days that follow.

Stages as run. The replay instance of M3 combines quantile gradient-boosted forecasts of node totals with per-rider share and concentration models into per rider-day completion distributions (Table 21 lists a recurrent network as another instance of M3); R joint samples per origin and seed are coupled across days by reordering each rider’s forecast days according to the ranks observed in the rider’s own 182-day history. Reduction forms five activity layers (or three coarser ones) from participation in the preceding 28 days. Integration averages each plan’s payment and value over the R joint samples. Allocation solves the multiple-choice problem on a 400-point budget grid with costs rounded up, so every portfolio it finds respects the cap; the exact reference enumerates each layer’s menu after cost–value dominance, which is exact here because every budget is a sole cap (Lemma 1). The main-grid instances in which the grid program loses value fall in the Spring Festival week.

Plans, value and budget. A layer’s plan is a ladder of three rungs, an active-day requirement $d \in \{ 0 , 2 , 3 , 4 , 5 , 6 \}$ and a reward level $\rho \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 , 1 . 5 , 2 \}$ (108 plans; a coarse grid keeps 18 and a day-requirement grid keeps $d \in \{ 5 , 6 \} ;$ ). The rungs are the 50/70/85%, 60/80/92% or 40/65/85% quantiles of the layer’s seven-day totals over the 182 days before the origin. With $Y _ { i }$ the rider’s window completions, $D _ { i }$ its active days and $J _ { i }$ the highest rung reached when $D _ { i } \geq d$ $( J _ { i } = 0$ otherwise), the plan pays $\rho m _ { J _ { i } } Y _ { i }$ with $m = ( 0 , 1 , 1 . 6 , 2 . 4 )$ . The value is the award-covered volume $\begin{array} { r } { \sum _ { i } \mathbf { 1 } \{ J _ { i } \geq 1 \} \dot { Y } _ { i } } \end{array}$ . It is a pricing target, not the uplift of Eq. (1): with recorded completions held fixed, $\rho$ changes only the cost. No stability weight enters the objective. The budget is a cap $B = \beta$ times the realized cost of a reference portfolio (middle ladder, $d = 3 , \rho = 1$ in every layer); because this cost is read from the evaluated week, the budget is retrospective, whereas operationally a request fixes B in advance.

Endpoints and uncertainty. Plan-cost error is $| \widehat { c } - c | / c$ for predicted and realized plan cost, summarized by its median over the plans of a unit; the 810 plans (1.1% of 72,900) with $c = 0$ are excluded and counted, all in the dormant layer. Award-rate error is the predicted minus realized share of riders with $J _ { i } \geq 1$ , over all plans. Regret is $( V ^ { \star } - V ( \widehat { a } ) ) / V ^ { \star }$ , with $V ( \widehat { a } )$ the realized value of the portfolio chosen on the predicted table and V<sup>⋆</sup> the exact optimum of the realized table of the full framework’s layering at the same budget; $V ^ { \star } > 0$ in every unit. Because every replacement is graded against this one optimum, regret can be negative: a portfolio that overspends can exceed $V ^ { \star }$ and a replacement with other layers has other menus. The framework’s regret is nonzero only in the two weeks around the Spring Festival (0.37% on average at $\beta = 0 . 5 )$ . Equal split gives every layer $B / L$ and takes, per layer, the plan of largest predicted value whose predicted cost fits the share. The replacements are: a point forecast (rounded per rider-day mean); equal-width buckets of the 28-day active-day count; independent days (the same per-day draws without coupling); a two-moment normal shortcut (per-rider normal window total; each tier’s payment and covered volume priced as the mean times the normal tier probability, without the truncated-mean correction; the day requirement judged on mean active days); and a plug-in mean. Each origin is first averaged over its three seeds; 95% intervals use an origin-block bootstrap (blocks of one and two origins, 20,000 draws, the wider interval reported). Every origin, including the four Spring Festival weeks, is kept.

Timing. All timings use one CPU core at 45 origins, seed 1 (Table 5). A framework request reduces the stored per-day draws to the requested days, integrates the rule over the 540 plans and allocates. The alternative re-draws the per-day samples of the requested days with the same random streams, re-couples them and runs the same integration and exact allocation; the re-drawn samples equal the stored ones bit for bit at every request, so the comparison is of time only. Model fitting is needed by both sides once per origin and is shown separately. The first request takes 2.97 s against 3.07 s for later ones, so the only cold-start cost is the 114.2 s sampling pass. K = 8 uses all 45 origins; $K = 3 2$ and 127 use origins 0, 5, 10, 15, 20, 25, 30, 35, 40, fixed before the run. The complete per-request alternative of Table 1 (117.8 s) also recomputes the rank template; re-drawing alone takes 114.5 s. By window length, the framework and the alternative take 2.39 and 18.5 s for one day and 5.63 and 101.1 s for six days.

Table 5: Where the time of rider requests goes (seconds; medians over origins, one CPU core). Top: one week-long request, by step. Bottom: K requests per origin in the fixed order (whole week, seven single days, remaining subsets); the framework total includes its once-per-origin sampling pass; speed-ups are medians of per-origin ratios. <sup>a</sup>Gradient-boosted node quantiles (29.8 s) plus share and concentration models (189.1 s), needed equally by both approaches. <sup>b</sup>Speed-up when model fitting is charged once to both sides.
<table><tr><td colspan="4">Step Framework Re-draw per request</td></tr><tr><td colspan="2">Model fitting (once per origin)ª</td><td>221.3 once</td><td>221.3 once</td></tr><tr><td colspan="2">Per-day draws, 7 days</td><td>108.7 once</td><td>108.7</td></tr><tr><td colspan="2">Cross-day rank template</td><td>3.5 once 1.6 once</td><td>reused</td></tr><tr><td colspan="2">Cross-day coupling</td><td colspan="2">0.54</td><td>1.8</td></tr><tr><td colspan="4">Window reduction</td><td>0.54</td></tr><tr><td colspan="4">Integration, 540 plans 2.52 Allocation 0.44 ms (grid)</td><td>2.52</td></tr><tr><td colspan="4">3.07</td><td>0.50 ms (exact)</td></tr><tr><td colspan="4">Per request, once-per-origin steps excluded Same, first request 2.97</td><td>114.5</td></tr><tr><td colspan="5"></td></tr><tr><td>K Origins</td><td>First</td><td>Total</td><td>Per req. Re-draw</td><td>Ratio</td><td>+Fitting6 b</td></tr><tr><td>8</td><td>45 117.8</td><td>136.1</td><td>17.01</td><td>245.0 1.81</td><td>1.31</td></tr><tr><td>32</td><td>9117.7</td><td>240.4</td><td>7.51</td><td>1410.7 5.87</td><td>3.58</td></tr><tr><td>127</td><td>9117.7 734.5</td><td></td><td>5.78</td><td>7864.2 11.04</td><td>8.82</td></tr></table>

Time saved by cheaper shortcuts. Table 6 adds, for the shortcuts of Table 2, the time each would save. Drawing and coupling trajectories run once per origin; window reduction, integration (2.52 of the 3.07 s of a week-long request) and allocation run for every request.

Table 6: Cheaper shortcuts the framework declines, on rider records (five layers, main plan grid). Accuracy: plan-cost error of the shortcut over the framework’s, or value lost (95% intervals over origins). Time: the framework’s stage time over the shortcut’s.
<table><tr><td>Stage</td><td>Shortcut</td><td>Accuracy cost of the shortcut</td><td>Time the shortcut saves</td></tr><tr><td>Prediction</td><td>point forecast</td><td>plan-cost error ×3.08 [2.48, 3.81]</td><td>not timed separately</td></tr><tr><td>Integration</td><td>independent days</td><td>plan-cost error × 1.51 [1.33, 1.74]</td><td>1.4% of sampling; integration ×1.001</td></tr><tr><td>Integration</td><td>simplified normal</td><td>plan-cost error ×2.46 [2.02, 3.01]; 5–6-day plans ×3.29</td><td>integration 85.8× (3.86 s per table)</td></tr><tr><td>Allocation</td><td>equal budget split</td><td>value -15.7% [13.8, 17.4]</td><td>0.51 → 0.01 ms</td></tr></table>

## A.1 CITIES LEFT OUT OF TRAINING

The reference refits the forecasting models at every weekly origin on all cities. For the held-out condition the 17 separately modelled city nodes are ranked by median daily participation and assigned to four folds (5/4/4/4 nodes); the pooled node of small cities always stays in training. For each fold and each of the 45 origins, all training rows of the fold’s cities, and the all-city aggregate rows, are removed, together with the fold’s contexts in the share and concentration models, and the fold’s riders are forecast from their own history only. All later stages, menus and budgets are unchanged, and the seed is that of the reference, so both use the same random streams. At each origin the held-out result is paired with the reference on the same riders and outcomes; intervals resample origins (B = 20,000), averaging folds within an origin. The pre-set rule calls a loss immaterial if the upper 95% bound of the paired increase in plan-cost error at budget fraction 1 is at most 0.045 (the half-width of the reference’s own interval) and that of selection regret at most 0.01. All 180 runs finished; nothing is dropped, and a control run that reuses the reference models at one origin reproduces the reference window exactly. Table 7 gives the result, and Table 8 the stage replacements on the held-out riders. The design fixed before the runs also contained a condition with models fitted once and left unrefit; every pre-specified rule and its outcome is listed in Table 22.

Table 7: Cities left out of training, rider records (five-band system, budget fraction 1). Plan-cost error is the median absolute relative error of predicted plan cost; change = held-out minus reference on the same riders and outcomes, 95% interval over origins. Pre-set margin: upper bound ≤ 0.045.
<table><tr><td>Condition</td><td>Origins / runs</td><td>Riders per run</td><td>Plan-cost error Held-out Reference</td><td></td><td>Change</td><td>Within margin</td></tr><tr><td>Unseen cities (four folds)</td><td>45 /180</td><td>311</td><td>0.343</td><td>0.340</td><td>+0.003 [-0.018, +0.020]</td><td>yes</td></tr></table>

Table 8: Stage replacements on held-out riders: replacement minus full framework, budget fraction 1, 95% interval over origins. Prediction and integration are scored by plan-cost error, reduction and allocation by selection regret. Unseen cities: the condition of Table $7 ;$ fit once: models fitted at the origin of 2026-04-21 and used for the next 16 weekly origins.
<table><tr><td>Replacement</td><td>Endpoint</td><td>Fit once</td><td>Unseen cities</td></tr><tr><td>Point forecast</td><td>plan-cost error</td><td>+0.416[+0.359, +0.472]</td><td>+0.380 [+0.292, +0.467]</td></tr><tr><td>Plug-in integration</td><td>plan-cost error</td><td>+0.317 [+0.235, +0.382]</td><td>+0.123 [+0.097, +0.148]</td></tr><tr><td>Two-moment normal</td><td>plan-cost error</td><td>+0.459 [+0.382, +0.523]</td><td>+0.205 [+0.167, +0.240]</td></tr><tr><td>Independent days</td><td>plan-cost error</td><td>+0.202 [+0.142, +0.250]</td><td>+0.058 [+0.035, +0.080]</td></tr><tr><td>Equal-width buckets</td><td>regret</td><td>+0.0017[+0.0006, +0.0028]</td><td>+0.0046[+0.0024, +0.0070]</td></tr><tr><td>Equal budget split</td><td>regret</td><td>+0.048[+0.021, +0.080]</td><td>+0.122 [+0.099, +0.145]</td></tr></table>

## B SYNTHETIC TRAJECTORIES AND A SHORT PILOT: DESIGN AND FURTHER RESULTS

Laws. The 24 laws use the support, offers, payment rules and budgets of Appendix H (five groups, 32 weekly trajectory types, nine offers per group whose reward-rate increments form a 3 × 3 grid, group rules fixed award, per-completion, attendance, fixed award, per-completion) with their own response generator and a fresh seed. With dose $x = 2 b + t$ of offer $a = 3 b + t ,$ , completions $C _ { k }$ and active days $D _ { k }$ of type $k ,$ base weights $w _ { g k } \in \{ 1 , \dots , 1 7 \}$ and a weekly shock $z \in \{ 0 , 1 \}$ (probability $\textstyle { \frac { 1 } { 2 } } ,$ drawn independently for every sample index and shared by all groups and offers at that index), the no-offer law is proportional to $w _ { g k } s _ { g k } ( z )$ with

![](images/4a768a631ef4c13ee1696113969b094ecf9f4740726a3af7c75985026aac5262.jpg)

![](images/9010b45ae3f3424e5fd0dfe0ebc412385df610914ae28a856a9ffd4554990393.jpg)  
Figure 2: (a) Relative regret against pilot length, 18 response laws, 10% scoring; bands: 95% intervals over laws. (b) Total time to answer $K$ windows per origin on rider records, one-off sampling pass charged; labels: median paired speed-ups.

$$
s _ { g k } ( z ) = 6 4 + u _ { g } [ z C _ { k } + ( 1 - z ) ( C _ { \operatorname* { m a x } } - C _ { k } ) ] ,
$$

and the law under dose x to

$$
\begin{array} { c } { { w _ { g k } s _ { g k } ( z ) \left[ 2 5 6 + a _ { g } \operatorname* { m i n } ( x , \tau _ { g } ) C _ { k } + b _ { g } ( x - \tau _ { g } ) _ { + } ^ { 2 } D _ { k } \right. } } \\ { { \left. + c _ { g } x ( 6 - x ) ( C _ { \mathrm { m a x } } - C _ { k } ) \right] , } } \end{array}
$$

with integers $a _ { g } \in \{ 1 , \ldots , 4 \} , b _ { g } \in \{ 1 , \ldots , 6 \} , c _ { g } \in \{ 0 , \ldots , 3 \} , \tau _ { g } \in \{ 1 , \ldots , 4 \}$ and $u _ { g } \in \{ 1 , 2 , 3 \}$ The combined family keeps all three response terms; the saturating family sets $b _ { g } = c _ { g } = 0 .$ , the threshold family $a _ { g } = c _ { g } = 0$ and the null family $a _ { g } = b _ { g } = c _ { g } = 0$ . The response form is unknown to every method.

Methods. All methods share exact enumeration over $9 ^ { 5 }$ portfolios, the same 33 budgets $B \in$ $\{ 0 , 5 , \ldots , 1 6 0 \}$ and the band [0.9B, 1.1B]. The long RCT uses offer-conditional frequencies from 1,024 observations per group and offer. The same-calendar RCT uses the pilot’s rider-weeks spread uniformly over the nine offers. Budgets are nominal: with integer division over offers, the nominal 512, 1,024 and 2,048 rider-weeks per group give 510, 1,023 and 2,046 pilot observations and 504, 1,017 and 2,043 same-calendar observations. The framework pilots offers with increments 0.2, 0.6 and 1.0, solves Eq. (5) exactly for two moments (normalized completions and active days) at each piloted offer, interpolates the tilt linearly in the increment through zero at the no-offer plan, and sums payment and value exactly over the 32 types. Ablations change one stage each: sampled integration averages 2,048 draws per group and offer from the same corrected law; without history uses the pilot frequencies directly (unpiloted offers take the nearest piloted offer; called “without synthesis” in the pre-specified rules); without moment matching uses the history law for every offer; withoutjoint integration evaluates payment and value on the mean trajectory; without allocation splits the budget equally and lets each group choose its best offer within its share; the pre-specified version ran on the sampled tables (Table 22), and Section 5 and Table 9 apply it to the exact-sum tables, a re-analysis of the stored laws without new draws (within 20% of the budget it raises one-week regret by +0.455 [+0.344, +0.571]).

Endpoint. For one replicate and pilot length, relative regret is $\begin{array} { r } { \sum _ { r } ( U _ { r } ^ { \star } - U _ { r } ) / \sum _ { r } U _ { r } ^ { \star } } \end{array}$ over the budgets r whose true optimum is feasible, with $U _ { r } ^ { \star }$ the optimal true uplift and $U _ { r }$ the true uplift of the issued plan, set to zero for an abstention or a plan whose true expected spend leaves the band.

Budgets without a feasible portfolio enter only the validity count; a replicate whose optimal uplifts sum to zero, as in the null laws, has no regret. Replicates are averaged within a law.

Pre-specified design. Design, endpoints, arms, unit of inference and decision rules were fixed before the runs, with one exception. After a single development run on one law (one replicate, one-week pilot), the operator was changed from a single tilt with a linear dose to one tilt per piloted offer at three offers; that law remains in the analysis. The pre-specified primary arm integrated 2,048 sampled trajectories; the exact sum over the 32 types was pre-specified as the finite-pilot arm of the stage decomposition, and we use it as the default because it removes the integration term (Lemma 7); the pre-specified rules and their outcomes for the sampled arm are listed in Table 22. Excluding the development law (17 response laws) leaves every conclusion unchanged: the framework minus the same-calendar $\mathrm { R C T \ i s - \dot { 0 } . 2 9 6 \ [ - 0 . 3 4 5 , - 0 . 2 4 \dot { 8 } ] , - 0 . 1 9 8 \ [ - 0 . 2 4 3 , \ - 0 . 1 5 7 ] \ a n d - 0 . 1 3 5 \ [ - 0 . 1 6 9 \dot { 8 } ] }$ −0.101] at one, two and four weeks; the gap to the long RCT at four weeks is +0.004 [−0.015, +0.023]; and the ablations raise one-week regret by +0.386, +0.391, +0.328 and, for sampling, +0.034 [+0.008, +0.061].

The decomposition, measured. Because the truth is known, each term of Eq. (8) can be computed; Table 10 does so for the sampled variant, which carries the synthetic, moment-matching and integra tion terms; allocation enumerates exactly, so the decision term is zero, and exact summation also sets the integration term to zero. As a separate allocation diagnostic, the budget-grid program is within 0.0015 relative regret of exact enumeration. Lengthening the pilot shrinks only the moment-matching term: each doubling divides it by 1.40 and 1.41, the $\sqrt { 2 } \approx 1 . 4 1$ rate of Corollary 1(a) for uplift; the synthetic term, which includes the interpolation bias, does not depend on the pilot and is the fixed price of shortening experimentation. As a heuristic, the decomposition suggests lengthening a pilot until the moment term is comparable to the synthetic term, which happens at two weeks here; realized regret keeps falling with a longer pilot (Table 3). Proposition 1 needs the true cost error, so it is an after-the-fact error analysis; on the 26, 140 and 499 requests (one-, two- and four-week pilots) whose comparator fits in the band tightened by twice that error it holds without exception.

Table 9: Further stage replacements of the framework (penalized relative regret, pre-specified 10% scoring, 18 response laws; the first two arms were not rescored at 20%, and the equal split on the exact tables is also given at 20% in the description of the methods above). Their differences from the framework are in Section 5.
<table><tr><td>Replacement</td><td>1wk</td><td>2wk</td><td>4wk</td></tr><tr><td>Without moment matching (history only)</td><td>0.656</td><td>0.656</td><td>0.656</td></tr><tr><td>Without joint integration (plug-in mean)</td><td>0.583</td><td>0.599</td><td>0.616</td></tr><tr><td>Without budget allocation (equal split, exact tables)</td><td>0.861</td><td>0.860</td><td>0.855</td></tr></table>

## B.1 EMPIRICAL CHECK OF THE DECOMPOSITION

In panel A each stage term is measured against the preceding law, exactly as in the proposition, and the direct error is below the sum of stage terms, as the triangle inequality allows. The synthetic and integration terms do not change with the pilot, while the moment term falls by factors 1.40 and 1.41 per doubling of the pilot (cost: 1.44 and 1.44), close to the $\sqrt { 2 } \approx 1 . 4 1$ of Corollary 1(a). Part (i) held on every request where its comparator exists. The check tightens by the direct error $\bar { E } _ { c } ,$ which needs the truth, so it is an after-the-fact error analysis. Among the 4,968 requests whose band holds a portfolio of true cost, the tightened band still holds one in 26, 140 and 499 (0.5%, 2.8% and 10.0% at one, two and four weeks), and the bound is below the comparator value, the trivial bound, in 16, 62 and 290 (0.3%, 1.2% and 5.8%); a rule held to the tightened band would abstain on the other 4,942, 4,828 and 4,469; realized regrets are reported in Section 5. The grid program is indistinguishable from enumeration. In panel B the signs follow the mechanisms of Appendix J: composing days independently understates the 5–6-day requirements and overstates lower ones, the crossing expected under shared daily shocks; the normal law with attendance evaluated at its mean fails where attendance binds, the plug-in floor of part (iii); and dropping distributional prediction or joint allocation costs most. Panel B concerns quantities that can be replayed under the offered plans; it involves no behavioural response.

Table 10: Empirical check of Proposition 1. A: controlled response laws with known truth (24 laws × 8 replicates, 33 budgets each), sampled variant $( S = 2 , 0 4 8 )$ , which carries the synthetic, moment-matching and integration terms; under exact summation the integration term is zero and the other terms are unchanged; allocation enumerates exactly, so $\varepsilon _ { \mathrm { d e c } } = 0$ , and the last two rows of A are an allocation diagnostic comparing the budget-grid program with enumeration; value errors are uplift in completions and cost errors are currency units, both summed over five layers (the median optimal uplift of a request is 14.1). B: rider records, 45 weekly origins × 3 seeds, five activity layers; each row replaces one stage of the framework and is compared with the framework on the realized records; intervals resample origins.
<table><tr><td colspan="2">A Pilot length</td><td>1 week 5.92 / 5.53</td><td>2 weeks</td><td>4 weeks</td></tr><tr><td colspan="3">Synthetic term  $\varepsilon _ { \mathrm { s y n } } ^ { v } / \varepsilon _ { \mathrm { s y n } } ^ { c }$ </td><td>5.92 / 5.53</td><td>5.92 / 5.53</td></tr><tr><td colspan="2">Moment term  $\varepsilon _ { \mathrm { m m } } ^ { v } / \varepsilon _ { \mathrm { m m } } ^ { c ^ { \ast } }$ </td><td>8.82 / 7.67</td><td>6.29 / 5.31</td><td>4.48 / 3.69</td></tr><tr><td colspan="2">Integration term  $\varepsilon _ { \mathrm { i n t } } ^ { v } / \varepsilon _ { \mathrm { i n t } } ^ { c } \left( S = 2 , 0 4 8 \right)$ </td><td>3.43 / 3.00</td><td>3.44 / 3.00</td><td>3.51 / 3.00</td></tr><tr><td colspan="2">Sum of stage terms</td><td>18.16 / 16.20</td><td>15.65 / 13.83</td><td>13.90 / 12.22</td></tr><tr><td colspan="2">Direct error  $\bar { E } _ { v } / \bar { E } _ { c }$  (truth vs. final tables)</td><td>11.79 / 10.88</td><td>10.06 / 9.35</td><td>8.89 / 8.24</td></tr><tr><td colspan="2">Requests with a comparator in  $B _ { 2 \bar { E } _ { c } }$ </td><td>26/6336</td><td>140/6336</td><td>499/6336</td></tr><tr><td colspan="2">share of the 4,968 feasible requests</td><td>0.5%</td><td>2.8%</td><td>10.0%</td></tr><tr><td colspan="2">bound below the comparator value</td><td>16</td><td>62</td><td>290</td></tr><tr><td colspan="2">Violations of part (i), tightening by  $\bar { E } _ { c }$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td colspan="2">Grid program – enumeration, relative regret</td><td>0.0005</td><td>0.0015</td><td>0.0009</td></tr><tr><td colspan="2">95% interval over laws</td><td>[-0.0014, 0.0028] [-0.0005,0.0038]</td><td></td><td>[-0.0014,0.0033]</td></tr><tr><td>B Stage replaced by</td><td>Mechanism</td><td>Observed on rider records</td><td></td><td></td></tr><tr><td>Prediction: point forecast Integration: independent</td><td>plug-in floor, part (iii) cross-day</td><td>award-rate error award rate —3.07pp [—3.28, -2.87] on 5–6-day</td><td>plan-cost error ×3.08 [2.48, 3.81]; median  $4 . 0  6 6 . 9 \mathrm { p p }$ </td><td></td></tr><tr><td>days</td><td>dependence</td><td>error ×1.51</td><td>plans, +3.32 pp [2.93, 3.69] on the others†; cost</td><td></td></tr><tr><td>Integration: per-rider normal, two moments; attendance at its mean</td><td>plug-in gate</td><td>×2.46</td><td>-10.31 pp [−10.97, -9.69] on 5–6-day plans, +0.23 pp [−0.19, 0.57] on the others†; cost error</td><td></td></tr><tr><td>Integration: plug-in mean</td><td>plug-in floor, part (iii)</td><td></td><td>+10.17 pp [9.31, 11.03] on the others†, —4.11 pp on 5–6-day plans; cost error × 1.85</td><td></td></tr><tr><td>Decision: budget grid vs. exact optimum</td><td>Lemma 8</td><td></td><td>zero loss in 134 of 135 five-layer main-grid instances (45 origins × 3 budgets), max 0.69%</td><td></td></tr><tr><td>Decision: equal budget split</td><td>allocation floor, part (iii)</td><td>(realized-record tables)</td><td>value —15.7% [13.8, 17.4] at budget fraction 0.5</td><td></td></tr></table>

<sup>†</sup>Plans requiring 0, 2, 3 or 4 active days, derived exactly from the all-plan and 5–6-day means. Award-rate entries are differences from the framework’s joint law in percentage points.

## C INTEGRATED STUDY: CHANGING REQUESTS AND BINDING BUDGETS

Design, endpoints and decision rules were registered, with the simulation code, before the run. One pre-run check on the first six requests of one law and replicate is disclosed with the registration and changed nothing. The analysis script was completed after the run and pooled all 48 laws for timing; H2 is reported here, as registered, on the 36 response laws.

Laws and requests. The 48 laws are fresh draws of the generator of Appendix B, twelve per family (36 with a response; the 12 null laws check validity and timing only). Behaviour responds to the offer at the week level, and the request decides which days are paid and counted. Every replicate issues its own stream of 60 requests. Each request draws, uniformly, an activity window among the 127 non-empty subsets of the seven days, a population among the 26 subsets of at least two of the five groups, and one rule for all its groups (fixed award, per-completion rate or attendance). Tier thresholds scale with the window, max $( 1 , \mathrm { r o u n d } ( t | W | / 7 ) )$ for $t \in \{ 1 4 , 2 8 , 4 2 \}$ completions or {2, 4, 6} active days; uplift is the change in completions inside the window. With $c _ { \mathrm { r e f } }$ the true expected cost of the cheapest portfolio that maximizes uplift without a budget, the request carries three budgets $B = \beta c _ { \mathrm { r e f } } , \beta \in \{ 0 . 3 , 0 . 5 , 0 . 7 \}$ , which gives 180 budgeted requests per replicate; the budgets are set from the true law so that they bind, and every method receives the same ones. A budget binds when its band optimum is below the unconstrained maximum; 72.3% of budgeted requests bind (over all 48 laws).

Data and methods. Each of eight replicates per law draws 4,096 no-offer weeks per group, a pilot of one, two or four weeks at 512 riders per group and week spread over offers 2, 5 and 8, and a same-calendar RCT that spreads the same nominal rider-weeks over all nine offers (with integer division per cell, 510, 1,023 and 2,046 pilot and 504, 1,017 and 2,043 same-calendar observations per group for one, two and four weeks); in the registered primary condition reported here, every sample index has its own weekly shock shared by all cells. Theframework prepares its laws once per origin, from history and pilot only: the history law, the moment-matched tilt of Eq. (5) at the piloted offers interpolated in the reward increment, and the corrected law of every offer. Each request then reuses these laws: its tables come from the exact sum over the 32 trajectory types for the request’s window, population and rule, and its plan from exact enumeration over the 9<sup>|population|</sup> portfolios in the band. The same-calendar RCT uses its empirical offer laws with the same integration and allocation. Stage replacements change one stage: the history law for every offer (without moment matching); pilot frequencies at the nearest piloted offer (without history); payment at the mean window completions and active days (plug-in integration); days composed independently from per-day marginals (independent days); the band split equally over the population’s groups (equal split). We also report the budget-grid program with 1,024 states in place of enumeration and, as a same-information control, the history and pilot laws interpolated in the reward increment (dose interpolation). For timing, every request is also answered by re-running the whole pipeline, including the preparation, from the raw data; its answers must equal the framework’s.

Endpoints and decision rules. Relative regret is computed per replicate over the budgeted requests whose band optimum exists, scoring an abstention or a plan whose true expected spend leaves the band as zero; replicates are averaged within a law and 95% intervals come from 20,000 bootstrap resamples of the 36 response laws. The primary band is [0.8B, 1.2B], with [0.9B, 1.1B] alongside. The confirmatory family (Table 22) is H1, the framework has lower regret than the same-calendar RCT at each pilot length (upper bound below zero), and H2, answering all 60 requests of an origin by reuse with the preparation charged is faster than re-running the pipeline per request (lower bound of the law-mean log speed-up over the 36 response laws above zero, every timed answer identical). The breakdowns of Table 12 are descriptive; a subgroup is summarized over the laws in which it contains a request with a feasible band optimum (at β = 0.3, two laws have none), a correction of the registered summary made after the run.

Table 11: Integrated study: relative regret (lower is better) on 36 response laws × 8 replicates × 180 budgeted requests, with each request’s window, population, rule and budget drawn afresh. Indented rows change one stage of the framework; every difference from the framework has a 95% interval above zero. Validity: share of the framework’s issued plans whose true expected spend lies in the band, over all 48 laws.
<table><tr><td rowspan="2">Method</td><td colspan="3">Within 20% of budget</td><td colspan="3">Within 10%</td></tr><tr><td>1wk</td><td>2wk</td><td>4wk</td><td>1wk</td><td>2wk</td><td>4wk</td></tr><tr><td>Same-calendar RCT</td><td>0.391</td><td>0.296</td><td>0.216</td><td>0.527</td><td>0.388</td><td>0.269</td></tr><tr><td>Framework</td><td>0.220</td><td>0.172</td><td>0.148</td><td>0.258</td><td>0.198</td><td>0.164</td></tr><tr><td>without moment matching</td><td>0.592</td><td>0.592</td><td>0.592</td><td>0.626</td><td>0.626</td><td>0.626</td></tr><tr><td>without history</td><td>0.471</td><td>0.433</td><td>0.404</td><td>0.561</td><td>0.509</td><td>0.470</td></tr><tr><td>plug-in integration</td><td>0.592</td><td>0.604</td><td>0.631</td><td>0.669</td><td>0.672</td><td>0.690</td></tr><tr><td>independent days</td><td>0.311</td><td>0.292</td><td>0.296</td><td>0.368</td><td>0.340</td><td>0.339</td></tr><tr><td>equal split</td><td>0.699</td><td>0.692</td><td>0.684</td><td>0.902</td><td>0.898</td><td>0.892</td></tr><tr><td>Dose interpolation (same information)</td><td>0.227</td><td>0.177</td><td>0.154</td><td>0.265</td><td>0.205</td><td>0.169</td></tr><tr><td>Validity of the framework (%)</td><td>93.2</td><td>94.0</td><td>94.1</td><td>87.7</td><td>90.0</td><td>91.1</td></tr></table>

Results. Framework minus same-calendar RCT is −0.171 [−0.201, −0.140], −0.124 [−0.160, −0.089] and −0.069 [−0.104, −0.035] within 20% (43.8%, 42.0% and 31.8% lower), and −0.269 [−0.297, −0.240], −0.190 [−0.225, −0.156] and −0.105 [−0.139, −0.073] within 10%. With a

one-week pilot, the stage replacements raise regret by $+ 0 . 3 7 3 [ + 0 . 3 3 3 , + 0 . 4 1 2 ]$ (moment matching), +0.251 [+0.191, +0.312] (history), +0.372 [+0.328, +0.417] (plug-in integration), +0.091 [+0.068, +0.116] (independent days) and +0.479 [+0.392, +0.570] (equal split). The budget-grid program’s regret differs from that of enumeration by at most 0.001 in every cell, and dose interpolation has higher regret by $+ 0 . 0 0 7 [ + 0 . 0 0 4 , + 0 . 0 1 0 ] , + 0 . 0 0 5 [ + 0 . 0 0 3 , + 0 . 0 0 7 ] \mathrm { a n d } + 0 . 0 0 6 [ + 0 . 0 0 3 , + 0 . 0 0 9 ]$ at one, two and four weeks. Table 12 shows that the gain over the same-calendar RCT holds in every rule, window length, population size and budget level, and Table 13 gives the preparation-charged timing.

Table 12: Integrated study by request type (within 20% of budget, descriptive): framework regret and framework minus same-calendar RCT with 95% intervals over the laws in which the subgroup is defined.
<table><tr><td></td><td></td><td></td><td colspan="2">1-week pilot</td><td colspan="2">2-week pilot</td></tr><tr><td></td><td>Level</td><td>Laws</td><td>Framework</td><td>minus same-calendar</td><td>Framework</td><td>minus same-calendar</td></tr><tr><td>Rule</td><td>fixed award</td><td>36</td><td>0.225</td><td>-0.175[-0.208, -0.142]</td><td>0.176</td><td>-0.127 [−0.166, −0.089]</td></tr><tr><td></td><td>per completion</td><td>36</td><td>0.229</td><td>-0.172 [-0.206, -0.138]</td><td>0.180</td><td>-0.125[-0.165, -0.087]</td></tr><tr><td></td><td>attendance</td><td>36</td><td>0.205</td><td>-0.165 [-0.192, -0.139]</td><td>0.162</td><td>-0.120 [-0.151, -0.089]</td></tr><tr><td>Window days</td><td>1-2</td><td>36</td><td>0.221</td><td>-0.193 [-0.226, −0.159]</td><td>0.173</td><td>-0.146[-0.184, -0.110]</td></tr><tr><td></td><td>3-5</td><td>36</td><td>0.219</td><td>-0.171 [-0.200, -0.141]</td><td>0.171</td><td>-0.123 [-0.159, -0.088]</td></tr><tr><td></td><td>6-7</td><td>36</td><td>0.226</td><td>-0.155 [−0.194, -0.113]</td><td>0.172</td><td>-0.119[-0.159, -0.082]</td></tr><tr><td>Groups</td><td>2</td><td>36</td><td>0.216</td><td>-0.214[-0.238, −0.190]</td><td>0.171</td><td>-0.152[-0.183, −0.123]</td></tr><tr><td></td><td>3</td><td>36</td><td>0.215</td><td>-0.160 [-0.193, -0.126]</td><td>0.169</td><td>-0.116[-0.154, -0.076]</td></tr><tr><td></td><td>4</td><td>36</td><td>0.230</td><td>-0.153 [-0.190, -0.114]</td><td>0.183</td><td>-0.111[-0.154, -0.067]</td></tr><tr><td></td><td>5</td><td>36</td><td>0.223</td><td>-0.148 [-0.187, -0.111]</td><td>0.186</td><td>-0.105 [-0.148, -0.063]</td></tr><tr><td>Budget β</td><td>0.3</td><td>34</td><td>0.305</td><td>-0.290 [-0.346, -0.237]</td><td>0.255</td><td>-0.205 [-0.253, -0.160]</td></tr><tr><td></td><td>0.5</td><td>36</td><td>0.228</td><td>-0.197 [-0.258, -0.151]</td><td>0.180</td><td>-0.151[-0.216, -0.101]</td></tr><tr><td></td><td>0.7</td><td>36</td><td>0.176</td><td>-0.149 [-0.181, -0.118]</td><td>0.131</td><td> $- 0 . 1 0 9 \left[ - 0 . 1 4 5 , - 0 . 0 7 5 \right]$ </td></tr></table>

Table 13: Integrated study, preparation charged (milliseconds on one core; medians over replicates for the one-off preparation and over requests otherwise). Speed-up: time to answer the first K requests of a replicate by re-running the whole pipeline for each, over the preparation plus the reusing requests; geometric mean over the 36 response laws (replicates averaged on the log scale) with 95% intervals. Answers are identical on all 51,840 timed requests of these laws and on all 69,120 of all 48 laws.
<table><tr><td>Pilot</td><td>Preparation</td><td>Request (reuse)</td><td>Request (re-run)</td><td> $K = 8$ </td><td> $K = 3 2$ </td><td> $K = 6 0$ </td></tr><tr><td>1 week</td><td>7.27</td><td>0.22</td><td>7.57</td><td>5.8×</td><td>11.9×</td><td> $1 4 . 1 \times \left[ 1 3 . 8 , 1 4 . 5 \right]$ </td></tr><tr><td>2 weeks</td><td>7.40</td><td>0.22</td><td>7.57</td><td>5.8×</td><td>11.9×</td><td> $1 4 . 2 \times [ 1 3 . 8 , 1 4 . 5 ]$ </td></tr><tr><td>4 weeks</td><td>6.25</td><td>0.21</td><td>6.61</td><td>5.9×</td><td>12.0×</td><td> $1 4 . 3 \times [ 1 4 . 0 , 1 4 . 7 ]$ </td></tr></table>

## D RESPONSE VALIDATION WITH A NEW GENERATOR

This study answers whether the results of Appendix C depend on the generator that produced its laws. Its design, endpoints, decision rules, simulation code and analysis script were all registered before the run. One disclosed pre-run check on four requests changed nothing.

Generator. None of it is shared with the 32-type construction of Appendix B. Each of 48 laws (twelve per family) has six rider groups, and each group is a mixture of 16 latent rider types with Dirichlet weights.

• Activity. A type is active on day d with probability $\mathrm { l o g i t } ^ { - 1 } ( \alpha _ { k } + \gamma _ { d } )$ , where $\gamma _ { d }$ is a day-ofweek effect of the group.

• Completions. An active day has $1 + \mathrm { P o i s s o n } ( \lambda _ { k } - 1 )$ completions, truncated at 15, and a weekly shock multiplies $\lambda _ { k }$ by 1 ± δ.

• Response. The response depends on the rule shown. Under dose x of rule $\rho ,$ activity becomes $p \dot { + } ( 1 - p ) \min \{ 0 . 9 , A _ { k } \hat { S } _ { \rho } ^ { A } f ( x ) \}$ and intensity $\lambda ( 1 + B _ { k } S _ { \rho } ^ { B } f ( x ) )$ . The sensitivities are $( S ^ { A } , S ^ { B } ) = ( 0 . 4 , 0 . 8 )$ for a fixed award, (0.2, 1) for a per-completion rate and (1, 0.2) for attendance.

• Families. Both channels with a saturating f; intensity only; a threshold f in which only types whose usual week lies between the first and third tier respond fully; no response.

Weeks take 16<sup>7</sup> daily patterns per rider, so the framework cannot enumerate them. The truth is exact: a mixture over types and shocks of day-by-day convolutions of the law of window completions and active days.

Requests, data and methods. Requests follow Appendix C, with populations of two to four of the six groups; 75.0% of budgeted requests bind (over all 48 laws). Each of eight replicates per law draws:

• 4,096 no-offer weeks per group;

• a pilot of one, two or four weeks at 512 riders per group and week, spread over the three rules times offers 2, 5 and 8;

• a same-calendar RCT that spreads the same nominal rider-weeks over the three rules times all nine offers. With integer division per cell the actual observations per group are 504, 1,017 and 2,043 in the pilot and 486, 999 and 2,025 in the same-calendar RCT for one, two and four weeks, so the trial has slightly fewer.

For every group and rule the framework tilts the stored weeks to the pilot moments by Eq. (5) (interpolated in the reward increment). It does this once per origin; every request then reuses the weights and integrates its window rule as a weighted sum over the stored weeks (Algorithm 1, lines 5–6). The stage replacements are the same as in Appendix C: history weeks for every offer; pilot weeks at the nearest piloted offer; payment at the weighted mean; days composed independently; equal split. Dose interpolation mixes history and pilot weeks with interpolation weights in the reward increment. For timing, each request re-fits the tilts it needs from the raw data. The endpoints, the unit of inference (36 response laws) and H1–H2 are as in Appendix C.

Table 14: New generator: relative regret (lower is better) on 36 response laws × 8 replicates × 180 budgeted requests. Indented rows change one stage of the framework; every difference from the framework, and that of dose interpolation, has a 95% interval above zero. Validity: share of the framework’s issued plans whose true expected spend lies in the band, over all 48 laws.
<table><tr><td rowspan="2">Method</td><td colspan="3">Within 20% of budget</td><td colspan="3">Within 10%</td></tr><tr><td>1wk</td><td>2wk</td><td>4wk</td><td>1wk</td><td>2wk</td><td>4wk</td></tr><tr><td>Same-calendar RCT</td><td>0.477</td><td>0.326</td><td>0.240</td><td>0.765</td><td>0.601</td><td>0.415</td></tr><tr><td>Framework</td><td>0.188</td><td>0.150</td><td>0.116</td><td>0.270</td><td>0.197</td><td>0.140</td></tr><tr><td>without moment matching</td><td>0.433</td><td>0.433</td><td>0.433</td><td>0.781</td><td>0.781</td><td>0.781</td></tr><tr><td>without history</td><td>0.415</td><td>0.354</td><td>0.309</td><td>0.610</td><td>0.517</td><td>0.448</td></tr><tr><td>plug-in integration</td><td>0.447</td><td>0.479</td><td>0.510</td><td>0.576</td><td>0.577</td><td>0.591</td></tr><tr><td>independent days</td><td>0.198</td><td>0.158</td><td>0.124</td><td>0.314</td><td>0.222</td><td>0.162</td></tr><tr><td>equal split</td><td>0.498</td><td>0.477</td><td>0.466</td><td>0.793</td><td>0.775</td><td>0.760</td></tr><tr><td>Dose interpolation (same information)</td><td>0.210</td><td>0.171</td><td>0.131</td><td>0.305</td><td>0.223</td><td>0.159</td></tr><tr><td>Validity of the framework (%)</td><td>93.1</td><td>94.6</td><td>95.7</td><td>83.5</td><td>88.9</td><td>92.7</td></tr></table>

## Results.

• Same-calendar RCT. Framework minus same-calendar RCT is −0.289 [−0.311, −0.267], −0.176 [−0.192, −0.160] and −0.123 [−0.140, −0.108] within 20% (60.6%, 53.9% and 51.4% lower), and −0.495 [−0.516, −0.473], −0.404 [−0.426, −0.383] and −0.276 [−0.296, −0.257] within 10%. The gain holds in every rule, window length, population size and budget level (Table 15).

• Stage replacements. With a one-week pilot they raise regret by:

– +0.245 [+0.214, +0.275] without moment matching;

– +0.228 [+0.206, +0.252] without history;

– +0.260 [+0.233, +0.287] with plug-in integration;

– +0.011 [+0.009, +0.013] with independent days;

– +0.311 [+0.269, +0.355] with an equal split.

• Dose interpolation. Its regret is higher by +0.022 [+0.018, +0.027], +0.020 [+0.016, +0.024] and +0.015 [+0.010, +0.019] at one, two and four weeks.

• Allocation. The budget-grid program’s regret is within 0.001 of that of enumeration.

• Timing. Here the per-request alternative re-fits only the tilts the request needs. With the preparation charged, answering the first 8, 32 and 60 requests of an origin is 1.22×, 2.33× and 2.70× [2.67, 2.73] faster (one-week pilot; 2.72× and 2.78× at two and four weeks). Medians are 52.4 ms for the preparation, 4.1 ms per reusing request and 13.7 ms per re-fitting request. Answers are identical on all 51,840 timed requests of the response laws.

Table 15: New generator by request type (within 20% of budget, descriptive): framework regret and framework minus same-calendar RCT with 95% intervals over laws.
<table><tr><td></td><td></td><td></td><td colspan="3">1-week pilot</td><td colspan="2">4-week pilot</td></tr><tr><td></td><td>Level</td><td>Laws</td><td>Framework</td><td>minus same-calendar</td><td>Framework</td><td></td><td>minus same-calendar</td></tr><tr><td>Rule</td><td>fixed award</td><td>36</td><td>0.179</td><td>-0.318 [-0.343, -0.293]</td><td></td><td>0.120</td><td>-0.127 [-0.143, -0.112]</td></tr><tr><td></td><td>per completion</td><td>36</td><td>0.196</td><td>-0.322 [-0.343, -0.300]</td><td></td><td>0.111</td><td>-0.127[-0.144, -0.111]</td></tr><tr><td></td><td>attendance</td><td>36</td><td>0.202</td><td>-0.183 [-0.213, -0.155]</td><td></td><td>0.132</td><td>-0.114[-0.137, -0.093]</td></tr><tr><td>Window days</td><td>1-2</td><td>36</td><td>0.186</td><td>-0.340 [-0.365, -0.315]</td><td></td><td>0.119</td><td>-0.154[-0.175, -0.133]</td></tr><tr><td></td><td>3-5</td><td>36</td><td>0.188</td><td>-0.285 [-0.308, -0.263]</td><td></td><td>0.116</td><td>-0.121 [-0.138, -0.106]</td></tr><tr><td></td><td>6-7</td><td>36</td><td>0.189</td><td>-0.269 [-0.294, -0.242]</td><td></td><td>0.122</td><td>-0.111[-0.130, −0.092]</td></tr><tr><td>Groups</td><td>2</td><td>36</td><td>0.211</td><td>-0.301 [-0.318, -0.285]</td><td></td><td>0.131</td><td>-0.154 [-0.168, -0.140]</td></tr><tr><td></td><td>3</td><td>36</td><td>0.188</td><td></td><td>-0.294 [-0.315, -0.273]</td><td>0.117</td><td>-0.121[-0.139, -0.104]</td></tr><tr><td></td><td>4</td><td>36</td><td>0.176</td><td></td><td>-0.277 [-0.306, -0.248]</td><td>0.110</td><td>-0.111 [-0.131, -0.093]</td></tr><tr><td>Budget β</td><td>0.3</td><td>36</td><td>0.261</td><td>-0.275 [-0.300, -0.252]</td><td></td><td>0.159</td><td>-0.182[-0.217, -0.150]</td></tr><tr><td></td><td>0.5</td><td>36</td><td>0.181</td><td></td><td>-0.288 [-0.312, -0.264]</td><td>0.114</td><td>-0.118[-0.138, -0.100]</td></tr><tr><td></td><td>0.7</td><td>36</td><td>0.170</td><td></td><td>-0.301 [-0.326, -0.276]</td><td>0.106</td><td>-0.106[-0.121, -0.093]</td></tr></table>

## E STRONGER CONTROLS, WEIGHT DIAGNOSTICS, BEHAVIOUR-CHANGING REQUESTS AND POPULATION REDUCTION

Four further studies were registered, with designs, decision rules, simulation code and analysis scripts fixed before the runs. All use the rider-level generator of Appendix D with fresh seeds, 60 changing requests per replicate, budgets at 0.3, 0.5 and 0.7 times the unconstrained optimum, and the law as the unit of inference (95% intervals from a law bootstrap, Holm correction within each family); studies C–E use eight replicates per law. Regret is relative regret of pure uplift within 20% of the budget, as in Table 4.

Study C: stronger same-information controls at matched cache. It draws 384 laws (96 per family; 288 with a response). The pilot and the same-calendar trial have the same number of observations per group (486, 999 and 2,025 at one, two and four weeks). Every arm reads the same cached 4,096 stored weeks per group and the same data, and every arm is solved by the same exact enumeration.

• History-anchored tilt ofthe trial. The framework’s tilt is fitted at each of the eight non-zero offers to that offer’s cell of the same-calendar trial. This is the strongest control that uses the same history and the same number of randomized observations.

• Normal law with exact truncated moments. It keeps the framework’s weights and replaces only the law of the paid statistic by a normal with the weighted mean and variance. Every tier is priced by

$$
\mathbb { E } [ X \mathbf { 1 } \{ \ell \leq X < h \} ] = \mu \{ \bar { \Phi } ( z _ { \ell } ) - \bar { \Phi } ( z _ { h } ) \} + \sigma \{ \varphi ( z _ { \ell } ) - \varphi ( z _ { h } ) \} ,
$$

with $z = ( t - 0 . 5 - \mu ) / \sigma$ at every integer threshold t. This separates distribution shape from the moment computation.

• Calendar condition. A rider keeps its type through all observed weeks. One shock per group and calendar week is shared by every rider observed in that week, and the pilot, the trial and 512 concurrent no-offer riders share the pilot weeks’ shocks. The framework then anchors its moment targets to the concurrent riders (pilot mean minus concurrent mean plus history mean), and the trial measures uplift against the same concurrent riders.

Table 16: Study C: relative regret within 20% of the budget, 288 response laws $\times 8$ replicates × 60 requests × 3 budgets. Every arm uses the same stored weeks, data and exact enumeration. Calendar: repeated riders with one shock per group and calendar week.
<table><tr><td>Condition</td><td>Method</td><td>1wk</td><td>2 wk</td><td>4wk</td></tr><tr><td>Independent</td><td>Framework</td><td>0.196</td><td>0.154</td><td>0.126</td></tr><tr><td></td><td>Dose interpolation (same information)</td><td>0.219</td><td>0.171</td><td>0.138</td></tr><tr><td></td><td>Normal law, exact truncated moments</td><td>0.250</td><td>0.208</td><td>0.179</td></tr><tr><td></td><td>History-anchored tilt of the trial</td><td>0.405</td><td>0.291</td><td>0.211</td></tr><tr><td></td><td>Same-calendar RCT</td><td>0.480</td><td>0.340</td><td>0.241</td></tr><tr><td>Calendar</td><td>Framework, concurrent anchor</td><td>0.223</td><td>0.215</td><td>0.207</td></tr><tr><td></td><td>Same-calendar RCT, concurrent control</td><td>0.490</td><td>0.429</td><td>0.386</td></tr></table>

## Results of study C. All seven registered hypotheses hold (Table 22).

• History anchoring helps the trial as well: the tilted trial has lower regret than the raw trial by 0.075 [0.072, 0.078] at one week.

• The framework’s regret is below that of every same-information control at every pilot length (Table 16). At one week it is below dose interpolation by 0.022 [0.021, 0.024], below the normal law with exact truncated moments by 0.053 [0.051, 0.055], and below the tilted trial by 0.209 [0.201, 0.217]. Study E below splits this last difference into its two sources. The normal comparison isolates the distribution shape, since the weights and the tier-moment formula are exact in both arms.

• Under calendar shocks shared by the riders of a week, the concurrently anchored framework has lower regret than the concurrently controlled trial by 0.267 [0.259, 0.274] at one week and 0.214 [0.206, 0.222] at two weeks.

• Reuse gives identical answers in all 184,320 comparisons per condition and pilot length. Charging the one-off preparation, it answers the 60 requests of an origin 2.73× [2.71, 2.74] faster than re-running the pipeline per request at one week (2.71× and 2.73× at two and four weeks).

Study E: placement of the observations versus sharing across offers. The tilted trial differs from the framework in two ways: its randomized observations are spread over nine offers instead of three, and its tilt is fitted separately at every offer instead of along one dose curve. Study E adds a joint fit ofthe trial that keeps the trial’s data and removes the second difference. It uses the framework’s own parameterization, λ(s) piecewise linear in the reward increment s with $\lambda ( 0 ) = 0$ and nodes $\theta = ( \theta _ { 2 } , \theta _ { 5 } , \theta _ { 8 } )$ at the three piloted offers, and fits θ to all eight non-zero trial cells by maximum likelihood of the tilted-history model:

$$
\widehat { \theta } = \arg \operatorname* { m a x } _ { \theta } \sum _ { a = 1 } ^ { 8 } n _ { a } \Big [ \lambda _ { \theta } \big ( s _ { a } \big ) ^ { \top } m _ { a }
$$

$$
- \log \sum _ { i = 1 } ^ { N } \exp \Bigl \{ \lambda _ { \theta } \bigl ( s _ { a } \bigr ) ^ { \top } \phi _ { i } \Bigr \} \Bigr ] ,
$$

where $m _ { a }$ is the mean feature vector of the $n _ { a }$ trial observations at offer a and $\phi _ { i }$ are the features of the N stored weeks. The problem is concave, and when the data sit at the three nodes only it separates into the framework’s per-node moment matching. The difference between the framework and the tilted trial then splits into two contrasts,

$$
R _ { \mathrm { F W } } - R _ { \mathrm { t i l t } } = \underbrace { ( R _ { \mathrm { F W } } - R _ { \mathrm { j o i n t } } ) } _ { \mathrm { p l a c e m e n t } } + \underbrace { ( R _ { \mathrm { j o i n t } } - R _ { \mathrm { t i l t } } ) } _ { \mathrm { s h a r i n g a c r o s s ~ o f f e r s } } ,
$$

the first at the same parameterization and the second on the same data. The study reuses the 384 laws of study C with fresh requests and data, the same observation counts, stored weeks, cache and exact enumeration. In the first run, the registered check that the joint fit reproduces the framework’s per-node fit on pilot data found a numerical fault in the joint-fit implementation; it was corrected under an amendment registered before the re-run, the study was re-run in full with the same laws, data streams and analysis script, and all results reported here come from the re-run.

Table 17: Study E: relative regret within 20% of the budget, 288 response laws × 8 replicates × 60 requests × 3 budgets; share of the difference to the tilted trial due to placing the observations at three offers, with 95% law-bootstrap intervals.
<table><tr><td>Method</td><td>1wk</td><td>2wk</td><td>4wk</td></tr><tr><td>Framework (pilot at three offers)</td><td>0.198</td><td>0.154</td><td>0.123</td></tr><tr><td>Joint fit of the trial, framework&#x27;s dose curve</td><td>0.253</td><td>0.206</td><td>0.183</td></tr><tr><td>History-anchored tilt of the trial, per offer</td><td>0.405</td><td>0.289</td><td>0.211</td></tr><tr><td>Same-calendar RCT</td><td>0.479</td><td>0.337</td><td>0.241</td></tr><tr><td colspan="4">Placement share 0.265 [0.252, 0.279] 0.382 [0.361, 0.405] 0.683 [0.623, 0.745]</td></tr></table>

## Results of study E. All five registered hypotheses hold (Table 22).

• Sharing the tilt across offers along the framework’s dose curve improves the trial by 0.152 [0.146, 0.158] at one week.

• At the same parameterization, the framework’s regret is below that of the joint fit by 0.055 [0.051, 0.059], 0.052 [0.048, 0.056] and 0.060 [0.053, 0.066] at one, two and four weeks, and its cost tables are more accurate (relative cost-table error 0.076 against 0.090 at one week; difference 0.014 [0.014, 0.015]).

• At one week, sharing across offers accounts for 73.5% [72.1, 74.8] of the framework’s advantage over the tilted trial and placement for 26.5% [25.2, 27.9]. The placement share grows with the pilot length (Table 17) as the per-offer fits improve.

• By risk component at one week (Table 18), the framework’s plans leave the band less often than the joint fit’s, below (3.4% against 6.5% of requests) and above (4.1% against 4.4%), and by less (3.5% against 4.6% of the budget), and its valid plans lose less value (13.6% against 16.5%).

Table 18: Risk components at one week within 20% of the budget (study E, 288 response laws; law means). Every arm issues a plan for more than 99.99% of the requests. Below/above: share of requests whose plan has true expected spend below/above the band. Excursion: mean distance outside the band, in units of the budget, over the plans that leave it. Valid-plan loss: value lost by the plans inside the band relative to the band optimum.
<table><tr><td>Method</td><td>Below</td><td>Above</td><td>Excursion</td><td>Valid-plan loss</td></tr><tr><td>Framework</td><td>0.034</td><td>0.041</td><td>0.035</td><td>0.136</td></tr><tr><td>Joint fit of the trial</td><td>0.065</td><td>0.044</td><td>0.046</td><td>0.165</td></tr><tr><td>Dose interpolation (same information)</td><td>0.044</td><td>0.045</td><td>0.041</td><td>0.149</td></tr><tr><td>History-anchored tilt of the trial, per offer</td><td>0.179</td><td>0.010</td><td>0.070</td><td>0.266</td></tr><tr><td>Same-calendar RCT</td><td>0.272</td><td>0.010</td><td>0.082</td><td>0.288</td></tr></table>

Weight diagnostics. For every group, rule and piloted offer, the study records the effective sample size $\mathrm { E S S } = ( \sum _ { i } w _ { i } ) ^ { 2 } / \sum _ { i } \stackrel {  } { w _ { i } ^ { 2 } }$ of the tilt over the $N = 4 { , } 0 9 6$ stored weeks. It also records N max<sub>i</sub> w<sub>i</sub>, whether the moment target lies in the coordinate-wise range (box) of the stored weeks’ features, a necessary condition for lying in their convex hull, whether the fit reaches its tolerance, and the condition number of the feature covariance. Percentiles below are over all such cells at one week, independent condition.

• ESS/N: median 0.951, 5th percentile 0.600, 1st percentile 0.422.

• N max<sub>i</sub> w<sub>i</sub>: median 2.05, 95th percentile 12.6, 99th percentile 26.4, so even at the 99th percentile no stored week carries more than 0.65% of the mass.

• The target lies outside the feature box in 0% of the cells, and no fit fails to reach its tolerance.

• The covariance condition number has median 7.8, 95th percentile 31.3 and 99th percentile 47.4.

Both registered diagnostic hypotheses hold. Within a law, the ESS deficit 1 − ESS/N is rankcorrelated with the cost-table error of the framework (Spearman 0.415 [0.400, 0.431]). Fitting the tilt to exact pilot moments on 1,024 rather than 16,384 stored weeks raises the relative cost-table error by 0.006 [0.006, 0.006]. These are heuristic stability diagnostics: fits converge, weights are not concentrated, and the ESS deficit tracks the cost-table error. They do not check the interior-hull and eigenvalue conditions of Lemma 6 along the parameter path (a condition number is not an eigenvalue bound), which remain sufficient conditions of the analysis.

Study D: requests that change behaviour. It draws 360 laws: 216 window-dependent laws (three families), 72 laws whose response does not depend on the window, and 72 null laws. In the windowdependent families, a short paid window concentrates effort on its days, riders shift some activity away from unpaid days, and less active riders respond more. A request may also target the riders above or below the group’s median of the previous no-offer week, whose response differs from the group’s.

• Direct reuse keeps the full-week, group-level calibration of Appendix D for every request.

• Request-aware reuse runs the pilot with a random paid window per rider. It then fits a window-relative tilt on four features (completions and active days inside and outside the paid window), pooled over the pilot riders’ windows. Stored weeks and pilot riders are conditioned on the request’s selection. The tilt is fitted once per origin for every group, rule, selection and piloted offer, and reused by every request.

The objective is the uplift in weekly completions, and the payment follows the request’s window and rule. The truth is exact.

Table 19: Study D: relative regret within 20% of the budget on the 216 window-dependent laws (8 replicates × 60 requests × 3 budgets). Selected: requests that target a subpopulation chosen on the previous week.
<table><tr><td>Method</td><td>1wk</td><td>2wk</td><td>4wk</td><td>Selected, 1 wk</td></tr><tr><td>Request-aware reuse</td><td>0.380</td><td>0.329</td><td>0.297</td><td>0.395</td></tr><tr><td>Direct reuse of the full-week calibration</td><td>0.576</td><td>0.584</td><td>0.596</td><td>0.742</td></tr></table>

Results of study D. All five registered hypotheses hold (Table 22).

• When the window changes behaviour, request-aware reuse lowers regret against direct reuse by 0.195 [0.189, 0.202], 0.255 [0.249, 0.261] and 0.300 [0.293, 0.306] at one, two and four weeks. For requests that target a selected subpopulation the reduction is 0.347 [0.338, 0.356] at one week.

• On all 288 response laws, request-aware reuse has lower regret than the same-calendar trial by 0.129 [0.123, 0.136] at one week.

• The reused tilt returns the same answers as a tilt re-fitted for every request in all 18,432 comparisons at each pilot length. With preparation charged it is 3.13× [3.09, 3.16] faster over 60 requests at one week (3.07× and 3.09× at two and four weeks).

The study makes the reuse rule explicit. A cached calibration is valid for the windows and populations on which the response law was measured. When the paid window or the selection changes behaviour, the cache is keyed by window-relative features and by the selection, and the pilot must randomize windows. Its cost is the pilot’s calendar weeks, which are charged above.

Study F: value retained by population reduction. Study F prices the reduction stage against a common individual-level optimum. It draws 3,072 laws from the rider-level generator (1,024 in each response family) and scores every method with the exact per-type tables, so estimation error is absent and only the loss from giving one plan to a whole layer remains. Each request takes a window, one rule and two of the six groups, with three budgets; every budget binds. A rider’s history statistic is its total completions over the previous four no-offer weeks. The individual-level optimum lets every rider receive its own offer; since riders of one latent type are exchangeable, it is a linear programme over types, computed exactly as the sup-convolution of the types’ upper concave hulls, and every layered plan is a feasible point of it. The framework’s reduction cuts each group into L equal-share quantile layers of the history statistic with one offer per layer, and the layered optimum is found by exact enumeration. The endpoint is the retained share: the value of a method summed over a law’s requests, divided by the individual optimum summed over the same requests.

Table 20: Study F: share of the individual-level optimum retained by L history-quantile layers per group, within 20% of the budget; 3,072 laws, 60 requests × 3 budgets each; 95% law-bootstrap intervals. By family: saturating response on both channels, intensity only, threshold response near the goal.
<table><tr><td>Layers per group</td><td>All laws</td><td>Both channels</td><td>Intensity</td><td>Threshold</td></tr><tr><td>1 (group-level plan)</td><td>0.825 [0.820, 0.831]</td><td>0.938</td><td>0.944</td><td>0.594</td></tr><tr><td>2</td><td>0.900 [0.897, 0.904]</td><td>0.966</td><td>0.971</td><td>0.764</td></tr><tr><td>3</td><td>0.926 [0.923, 0.928]</td><td>0.974</td><td>0.978</td><td>0.825</td></tr><tr><td>4</td><td>0.938 [0.936, 0.940]</td><td>0.977</td><td>0.982</td><td>0.856</td></tr><tr><td>5</td><td>0.946 [0.944, 0.948]</td><td>0.979</td><td>0.984</td><td>0.875</td></tr></table>

## Results of study F.

• Five history layers retain 94.6% [94.4, 94.8] of the individual-level optimum, against 82.5% [82.0, 83.1] for one group-level plan: 0.120 [0.116, 0.125] more, which recovers 68.9% [68.8, 69.0] of the group-level loss (Table 20).

• Every added layer raises the retained share: five layers retain 0.046 [0.044, 0.047] more than two, and the fourth and fifth layers add 0.013 [0.012, 0.013] and 0.008 [0.007, 0.008].

• Retention is highest when the response is smooth (97.9% and 98.4% with five layers) and lowest under a threshold response near the goal (87.5%), where the plan that pays depends most on a rider’s distance to the tier.

• No layered plan abstains in this band and every budget binds, so the loss measured here is the structural cost of reduction; estimation error, which studies A–E price, comes on top of it.

## F PUBLIC RETAIL DATA

M5. M5 (Makridakis et al., 2022a) provides 3,000 series (300 per store) from an 873-day panel, 45 weekly origins and seven-day targets; eligibility uses the preceding 182 days and five groups use positive-sales days in the preceding 28 days. Each forecast is trained on 13 earlier weekly origins, and 2,048 aligned scenarios keep cross-day ranks. Scale tests use all 30,490 series at three origins. Sales carry no incentive response, so M5 tests computation and settlement transfer only.

## G COMPUTATION AND APPROXIMATION BOUNDARIES

Complete time runs from loading the inputs to returning every requested plan and includes all preparation; model fitting is reported separately. All methods in a comparison receive the same laws, menus, budget bands and the same permission to reuse inputs.

Reuse across requests. On 3,000 M5 series and 45 origins, reusing a valid law instead of regenerating it for each request lowers the median request time from 84.78 to 4.36 seconds (paired 20.00×)

with identical scenarios and identical exact values on all 405 tables; charging the 80.63-second preparation to the framework, 8, 32 and 127 requests per origin are 1.66×, 3.92× and 7.11× faster. On all 30,490 series the times are 82.62 and 1,146.49 seconds (13.88×), and all nine optima agree. With the 1,063.87-second preparation charged, eight requests are 1.59× faster (median over the three origins).

Per-request mixed-integer programming. The multiple-choice piecewise-linear MIP uses binaries $y _ { s }$ and integers t with

$$
\begin{array} { r l r l } & { \sum _ { s \in g } y _ { s } = 1 , \qquad } & & { 0 \leq t _ { s } \leq ( U _ { s } - L _ { s } ) y _ { s } , } \\ & { x = \sum _ { s } ( L _ { s } y _ { s } + t _ { s } ) , \qquad } & & { v = \sum _ { s } [ ( a _ { s } L _ { s } + b _ { s } ) y _ { s } + a _ { s } t _ { s } ] / D , } \end{array}
$$

solved to zero gap with a time limit per request; every answer is checked in exact arithmetic. Over 396 conditions (180 M5, 216 controlled) with 33 budget bands each, the dynamic program reuses its value curve across requests and is 5.52× faster than the envelope MIP at 33 requests (raw dense allocation: 5.47×), with the same optimal value on every request the MIP resolves. Each condition’s 33 budgets were answered in five balanced method orders, a timing repetition, which gives 65,340 requests per method; the raw and envelope MIP formulations leave 25 and 5 unresolved, while all dynamic-programming variants resolve every request. The gain over per-request MIP comes from reusing the value curve.

New payment rule with the same laws. For volume tier $J _ { i s }$ and active days $D _ { i s } .$ , the active-day rule pays $7 D _ { i s } \rho { { J } _ { i , { s } } }$ when $D _ { i s } \geq d$ and $J _ { i s } > 0$ . The histogram

$$
H _ { s } ( k , j ) = \sum _ { i \in G _ { l } } \mathbf { 1 } \{ D _ { i s } = k , J _ { i s } = j \}
$$

gives the exact coefficients

$$
A _ { s j } ( d ) = 7 \sum _ { k = d } ^ { 7 } k H _ { s } ( k , j )
$$

for all eight attendance requirements. Over 45 origins, 2,048 scenarios, five groups and 270 reward paths, all 14,580 answers equal direct settlement, and shared preparation cuts complete time at every origin, by 70.98% on average (67.41–74.08%).

## H CONTROLLED RESPONSE INSTANCE

Support, settlement and objective. Five groups share 32 weekly trajectory labels (29 distinct trajectories) with daily completions

$$
y _ { k d } = { \left\{ \begin{array} { l l } { 2 ( \lfloor k / 8 \rfloor + 1 ) + ( ( d + k ) { \bmod { 3 } } ) , } & { d < k { \bmod { 8 } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. }
$$

and $\begin{array} { r } { C _ { k } = \sum _ { d } y _ { k d } , D _ { k } = \sum _ { d } { \mathbf { 1 } } \{ y _ { k d } > 0 \} } \end{array}$ . Groups use fixed-prize, per-completion, attendance, fixed-prize and per-completion rules. Each group has nine offers $a = 3 b + t , b , t \in \{ 0 , 1 , 2 \}$ with rung rates $( \ell _ { j } + 4 0 \dot { b } + 1 0 t ) / 1 0 0 , \ell = ( 1 \bar { 0 } , 1 5 , 2 0 )$ . Fixed prizes and per-completion rates use completion rungs (14, 28, 42) and pay rung r<sub>j</sub> or $C _ { k } r _ { j } ;$ attendance pays $7 D _ { k } r _ { j }$ at rungs of (2, 4, 6) active days; below the first rung the payment is zero. With uplift $U _ { g }$ , mean payment $M _ { g }$ and stability $A _ { g } = \mathrm { P r } ( 0 . 8 M _ { g } \leq Y _ { g } \leq 1 . 2 \bar { M } _ { g } )$ , the composite objective is

$$
J = \sum _ { g } [ U _ { g } + M _ { g } A _ { g } ] / 1 0 4 ;
$$

requests are $B = 0 , 5 , \ldots ,$ 160 with band [0.9B, 1.1B] on expected total payment, over the $9 ^ { 5 } = { }$ 59,049 portfolios.

Response laws. With $x = 2 b + t ,$ base weights $w _ { g k } \in \{ 1 , \dots , 1 7 \}$ , elasticity $\boldsymbol { e } _ { g } ,$ , threshold $\tau _ { g }$ and rule index $j ,$ , offer-conditioned masses are proportional to ${ w _ { g k } f _ { g k } ( x ) }$ with

$$
\begin{array} { r l } & { f ^ { \mathrm { n u l l } } = 1 , } \\ & { ~ f ^ { \mathrm { s a t } } = 1 2 8 + e _ { g } ( j + 1 ) \frac { x } { 2 + x } C _ { k } , } \\ & { ~ f ^ { \mathrm { t h r } } = 1 2 8 + e _ { g } ( j + 1 ) ( x - \tau _ { g } ) _ { + } ^ { 2 } C _ { k } , } \\ & { ~ f ^ { \mathrm { t r a d e } } = 2 5 6 + e _ { g } \big [ ( j + 1 ) \frac { 1 6 x } { 1 + x } D _ { k } + \frac { x ( 6 - x ) } { 6 } C _ { k } \big ] . } \end{array}
$$

The null and threshold families add a shared weekly shock z through the factor

$$
3 2 + u _ { g } \big [ z C _ { k } + ( 1 - z ) ( C _ { \operatorname* { m a x } } - C _ { k } ) \big ] ;
$$

the no-offer law omits $f .$

## I SETTLEMENT INTERFACE AND EXACT ALLOCATION

Table 21 lists the seven modules of Figure 1 (the figure’s short names in parentheses), their inputs and outputs, and the instance used on delivery data. Each module can be replaced by any estimator with the same input and output; allocation exactness says nothing about the accuracy of Modules 1–5.

Algorithm 1 Answering a request $r = ( \mathcal { T } , W , x , \Gamma , B )$   
1: Once per origin: fit M3 on pre-origin history; draw R joint per-day samples for every rider (cross-day   
ranks from each rider’s history); store.   
2: Once per rule family: run a pilot of n rider-weeks at three offers; fit $\lambda _ { l , a }$ by Eq. (5).   
3: Select roster I (M1–M2) and form layers $\mathcal { G }$ with plan menus $\boldsymbol { A } _ { l }$ from recent activity (M4).   
4: for each layer l and plan a $\in \mathcal { A } _ { l }$ do   
5: Compose window W from stored samples; tilt to $q _ { l , a }$ by reweighting the stored samples, or the trajectory   
types when they can be enumerated (M5).   
6: Compute award probabilities and $( \widehat { c } _ { l } ( a ) , \widehat { v } _ { l } ( a ) )$ for rule Γ as the weighted sum of Eq. (4) over the stored   
samples, the exact sum when types are enumerable; no redraw (M6).   
7: end for   
8: Reduce menus by the rule admissible for B (M4); solve Eq. (2) on B by exact enumeration or the budget-grid   
program (M7).   
9: return plans with attaining rewards, or report that no plan satisfies B.

Table 21: Four stages and seven modules. The last column gives the instance used on the delivery platform.  
Stage Module Input → output Delivery instance   
Predict 1. Response ranking pre-decision context and past offers → ranking doubly robust learner with a 5% no-offer   
(Rank) by incremental activity group (Dudík et al., 2011)   
2. Roster (Recall) ranking and eligibility → population fixed before at least one delivery in the preceding 182   
attendance, incl. zero-activity riders days   
3. Baseline forecast histories and context → no-offer outcome law recurrent network, 60-day history, five   
(Forecast) for the requested window day zero-inflated Beta shares   
Reduce 4. Grouping (Activity baseline capacity → groups and non-dominated ordered buckets merged to balance disper  
layers) plan menus sion and spending stability   
Integrate 5. Response correc- no-offer law and pilot or context-matched out- exponential tilt, Eq. (5); residual matching   
tion (Correct) comes → offer-conditioned joint trajectories on incentive, time and weather   
6. Settlement (Score) trajectories, payment rule, attainable rewards → Eqs. (6)–(7)   
cost, award probability, score and reward witness   
Allocate 7. Budget allocation group domains and both budget bounds → re- two-sided Bellman recursion, $\operatorname { E q . } \left( 1 1 \right)$   
(Allocate) ward portfolio or infeasibility

Settlement identities. For nested award events $E _ { 1 } \supseteq \cdots \supseteq E _ { J }$ and $E _ { J + 1 } = \emptyset$ , a highest-rung prize pays

$$
g = \sum _ { j } R _ { j } { \bf 1 } \{ E _ { j } \backslash E _ { j + 1 } \} ;
$$

taking expectations gives Eq. (6) without independence across riders. A rate paid on every completion satisfies, by telescoping,

$$
g = \sum _ { j } ( \rho _ { j } - \rho _ { j - 1 } ) Y \mathbf { 1 } \{ E _ { j } \} ,
$$

which gives Eq. (7). For integrable quantile functions,

$$
\mathbb { E } Y ( a ) - \mathbb { E } Y ( 0 ) = \int _ { 0 } ^ { 1 } [ Q _ { a } ( v ) - Q _ { 0 } ( v ) ] d v ,\tag{9}
$$

so uplift is integrated once on the final response curve, which avoids counting overlapping rung effects twice; the identity is an accounting statement, not causal identification. Mean payment and award probability do not determine spending stability: if one or two completions are equally likely, a fixed prize $3 / 2$ and a unit rate 1 have the same mean payment and award probability, but they fall in [1.2, 1.8] with probability one and zero, respectively. This is why M6 integrates joint trajectories.

## I.1 EXACT ALLOCATION OVER ATTAINING SCORE SEGMENTS

Fix a conditional trajectory law and uplift $u _ { l q }$ on each group/path $( l , q )$ . On a straight reward path $r ( t ) = r ^ { - } + t ( r ^ { + } - \bar { r } ^ { - } ) , \bar { 0 } \leq t \leq 1$ , with expected-payment endpoints $\mu ^ { - } < \mu ^ { + }$ ，

$$
\begin{array} { r } { t ( \mu ) = \displaystyle \frac { \mu - \mu ^ { - } } { \mu ^ { + } - \mu ^ { - } } , } \\ { S _ { s } ( \mu ) = k _ { s } ^ { \top } r ( t ( \mu ) ) , } \end{array}\tag{10}
$$

where $k _ { s }$ is the payment coefficient vector of scenario s under the rule. Each closed event $0 . 8 \mu \leq$ $S _ { s } ( \mu ) \leq 1 . 2 \mu$ is an interval, a point or empty, so its probability $p _ { l q } ( \mu )$ is piecewise constant and the score

$$
v _ { l q } ( \mu ) = u _ { l q } / T _ { 0 } + \lambda \mu p _ { l q } ( \mu ) / T _ { 0 }
$$

is piecewise affine $( T _ { 0 }$ is a common normalizer). Gaps between paths are kept, never interpolated, and a path is used only inside one response branch.

Same-spend reduction. For attainable sets $I _ { l q }$ keep the envelope $F _ { l } ( b ) = \operatorname* { m a x } _ { q : b \in I _ { l q } } v _ { l q } ( b )$ with an attaining reward. With additive scores this preserves the optimum under any constraint that depends only on the group amounts: each candidate can be replaced by a no-worse witness at the same amount, and every witness was feasible. Lemma 1 states which menu reductions are admissible for which budget.

Lemma 1 (Admissible menu reductions). Let each layer l choose one planfrom afinite menu A with cost $c _ { l } ( a ) \geq 0$ and value $v _ { l } ( a )$ , and let a portfolio befeasible when $\dot { \textstyle \sum _ { l } } c _ { l } ( a _ { l } ) \in B$ . (i) $I f B = [ 0 , \bar { b } ]$ (a sole cap), deleting a plan a for which some $a ^ { \prime } \in \mathcal { A } _ { l }$ has $c _ { l } ( a ^ { \prime } ) \leq \overline { { c } } _ { l } ( a )$ and $v _ { l } ( a ^ { \prime } ) \geq v _ { l } ( a )$ (ties broken by afixed order) preservesfeasibility and the optimal value. (ii) For any B, deleting a when some a<sup>′</sup> has $c _ { l } ( a ^ { \prime } ) = c _ { l } ( a )$ and $v _ { l } ( a ^ { \prime } ) \geq v _ { l } ( a )$ preserves feasibility and the optimal value; in both cases the retained plan is an attaining witness. (iii) Rule (i) is not admissible once B has a positive floor.

Proof. (i) Replacing $a _ { l }$ by $a ^ { \prime }$ in a feasible portfolio lowers total cost weakly, so it stays in $[ 0 , { \bar { b } } ]$ and raises value weakly; repeating the replacement ends on retained plans, since the fixed tie order excludes cycles. (ii) The replacement leaves total cost unchanged, so membership in any $\boldsymbol { B }$ is unchanged. (iii) One layer with plans $( c , v ) = ( 1 , 2 ) , ( 2 , 1 )$ and $\pmb { \cal B } = [ 1 . 8 , 2 . 2 ]$ : rule (i) deletes the second plan, which is the only feasible one. □

Which rule each experiment uses, and an independent check. Rider-record budgets are sole caps, so M4 applies rule (i) and the exact reference enumerates the products of the rule-(i) menus. The controlled study of Section 5 has two-sided bands and applies no reduction: it enumerates all $9 ^ { 5 }$ portfolios. The segment and envelope allocators below use only rule (ii). To rule out an error that a reduction shared by solver and reference would hide, we re-solved every rider table small enough for complete enumeration (at most $2 \times 1 0 ^ { 8 }$ portfolios: 675 of the 810 realized and 12,150 of the 14,580 predicted tables) by a separately written complete enumeration without any reduction: feasibility and optimal value agree with the reference on every one of them (the returned plan can differ among portfolios of equal value). The budget-grid program loses value on 3 of the 405 predicted five-layer main-grid instances (largest 0.92%) and never exceeds the cap. In the controlled study, 10,800 enumeration calls covering 356,400 budget bands differ in $0 ,$ and rerunning the study with the independent solver’s choices reproduces every stored result bitwise.

Segment recursion. Let $\delta = T _ { 0 } / h , o _ { l } = \operatorname* { m i n } _ { q } \mu _ { l q } ^ { - } / \delta$ and $\mu _ { l } = \delta ( x _ { l } + o _ { l } )$ with integer x . With R equally weighted scenarios and stability count $n ,$ candidate $q$ of group l attains score $\alpha x + \beta$ , with

$$
\begin{array} { l } { { \alpha = \lambda n / ( R h ) , } } \\ { { \beta = u _ { l q } / T _ { 0 } + \alpha o _ { l } , } } \end{array}
$$

at every integer spend index $x \in [ L , U ]$ of a segment. A physical band $[ B ^ { - } , B ^ { + } ]$ becomes the integer interval

$$
\begin{array} { r } { \left[ \operatorname* { m a x } \bigl \{ 0 , \lceil B ^ { - } / \delta - \sum _ { l } o _ { l } \rceil \bigr \} , \lfloor B ^ { + } / \delta - \sum _ { l } o _ { l } \rfloor \right] . } \end{array}
$$

Each segment contributes

$$
\begin{array} { c } { { D _ { l } ( b ) = \displaystyle { \operatorname* { m a x } _ { q , s } } \bigg \{ \alpha _ { q s } b + \beta _ { q s } } } \\ { { + \mathrm { \small { ~ \displaystyle { \operatorname* { m a x } _ { k \in [ b - U _ { q s } , b - L _ { q s } ] , \ 0 \leq k < N } } } } [ D _ { l - 1 } ( k ) - \alpha _ { q s } k ] \bigg \} . } } \end{array}\tag{11}
$$

Minimum-cost options, closed endpoints, gaps and unreachable states are retained, and no no-offer action is added.

Proposition 2 (Segment and dense allocation coincide). For a fixed law, additive scores, group-wise menus and exact rational arithmetic, the recursion (11) returns the same value curve and the same canonical rewards as dense allocation on the same grid.

Proof. Step 1: the recursion. Set $D _ { 0 } ( 0 ) = 0$ and all unreachable values to $- \infty$ . On a segment, substitute $x = b - k$ in $D _ { l - 1 } ( b - x ) +$ αx + β; the feasible predecessor interval gives Eq. (11), and maximizing over segments keeps every action.

Step 2: ties. Ties within a segment keep the largest k (smallest current x); ties across segments keep the smallest x, then the smallest candidate index. By induction the choices equal the dense ones.

Step 3: rewards. Backtracking from the smallest optimal feasible total through Eq. (10) returns rewards with the same score and physical spend; an unreachable band is infeasible for both. □

A monotone queue evaluates each window maximum in $O ( N )$ , so K segments, L groups and N budget states need $O ( K N + L K + L N )$ operations against $O ( L N ^ { 2 } + \mathbf { \bar {  { K } } } N + L K )$ for dense allocation; both store $O ( L N )$ choices. If each group’s candidate intervals are pairwise disjoint, the envelope equals the raw menu and its construction can be skipped without changing the solution. Compared with graphical dynamic programming for piecewise-linear investment under an upper cap (Gafarov et al., 2014; 2016) and nonconvex piecewise-linear knapsacks (Kameshwaran & Narahari, 2009), the recursion handles reward domains with gaps, two-sided budgets, rule-dependent event scores and a reward witness for every scored spend.

## J FOUR-TERM LOSS DECOMPOSITION: PROOFS AND STAGE BOUNDS

Setting. Layers $l = 1 , \ldots , L$ have finite plan menus $\boldsymbol { A } _ { l } ;$ a portfolio is $a = ( a _ { 1 } , \ldots , a _ { L } )$ with

$$
\begin{array} { l } { { c ( a ) = \displaystyle \sum _ { l } c _ { l } ( a _ { l } ) , } } \\ { { v ( a ) = \displaystyle \sum _ { l } v _ { l } ( a _ { l } ) . } } \end{array}
$$

Tables without a superscript are the true ones; $( c ^ { ( k ) } , v ^ { ( k ) } )$ for $k = \mathrm { s y n }$ , mm, int, dec are the successive tables of the framework and $( \widehat { c } , \widehat { v } ) = ( c ^ { \mathrm { d e c } } , \widehat { v } ^ { \mathrm { d e c } } )$ . Stage errors $\varepsilon _ { k } ^ { c } , \varepsilon _ { k } ^ { v }$ and their sums $E _ { c } , E _ { \imath }$ are as in Section 3. The band and the value it carries are

$$
\begin{array} { c } { { \mathcal { B } = [ \underline { { { b } } } , \bar { { b } } ] , } } \\ { { \mathcal { B } _ { \delta } = [ \underline { { { b } } } + \delta , \bar { { b } } - \delta ] , } } \\ { { V ^ { \star } ( \delta ) = \operatorname* { m a x } \{ v ( a ) : c ( a ) \in { \mathcal { B } _ { \delta } } \} , } } \\ { { \omega ( \delta ) = V ^ { \star } ( 0 ) - V ^ { \star } ( \delta ) \geq 0 , } } \end{array}
$$

with $V ^ { \star } ( \delta ) = - \infty$ if no portfolio qualifies. Cost and uplift entries are expectations of bounded functions of the trajectory (payment, completions), or differences of two such expectations, so they are linear in the law; the same holds for a stability indicator with a fixed band. Parts (i)–(ii) use only table errors and therefore also cover the mean-centred stability score, which is not linear in the law. In particular, if a layer law is the mixture of its members’ laws, $n _ { l }$ times its expectation equals the sum of the members’ expectations exactly; population reduction therefore adds no error to expected tables and acts only through the menus, which enter $V ^ { \star }$ and ω.

## J.1 PROOF OF PROPOSITION 1

(i) Upper bound. Step 1: table errors. For every portfolio, the triangle inequality along the chain gives

$$
\begin{array} { r l } & { | c ( a ) - \widehat { c } ( a ) | \leq \displaystyle \sum _ { l } \displaystyle \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } _ { l } } | c _ { l } ( a ^ { \prime } ) - \widehat { c } _ { l } ( a ^ { \prime } ) | } \\ & { \qquad \leq \displaystyle \sum _ { l } \displaystyle \sum _ { k } \displaystyle \operatorname* { m a x } _ { a ^ { \prime } } | c _ { l } ^ { ( k ) } ( a ^ { \prime } ) - c _ { l } ^ { ( k ^ { - } ) } ( a ^ { \prime } ) | } \\ & { \qquad = E _ { c } , } \end{array}
$$

and likewise $| v ( a ) - \widehat { v } ( a ) | \leq E _ { v }$

Step 2: feasibility. Since $\widehat { c } ( \widehat { a } ) \in \mathcal { B } _ { E _ { c } }$ c

$$
c ( \widehat { a } ) \in [ \widehat { c } ( \widehat { a } ) - E _ { c } , \widehat { c } ( \widehat { a } ) + E _ { c } ] \subseteq B .
$$

Step 3: value. Let $a ^ { \circ }$ attain $V ^ { \star } ( 2 E _ { c } )$ . Then $\widehat { c } ( a ^ { \circ } ) \in \mathcal { B } _ { E _ { c } }$ , so $a ^ { \circ }$ is admissible for the planner, the admissible set is nonempty, and $\widehat { v } ( \widehat { a } ) \geq \widehat { v } ( a ^ { \circ } ) - \eta$ . Hence

$$
\begin{array} { r l r l } & { v ( \widehat { a } ) \geq \widehat { v } ( \widehat { a } ) - E _ { v } \quad } & & { ( \mathrm { S t e p ~ 1 } ) } \\ & { \qquad \geq \widehat { v } ( a ^ { \circ } ) - \eta - E _ { v } \quad } & & { ( \eta \mathrm { - o p t i m a l i t y } ) } \\ & { \qquad \geq v ( a ^ { \circ } ) - 2 E _ { v } - \eta \quad } & & { ( \mathrm { S t e p ~ 1 } ) } \\ & { \qquad = V ^ { \star } ( 2 E _ { c } ) - 2 E _ { v } - \eta . } \end{array}
$$

Step 4: conclusion. Subtracting from $V ^ { \star } ( 0 )$ gives Eq. (8). If $\omega ( \delta ) \leq \kappa \delta$ on $[ 0 , 2 E _ { c } ]$

$$
\omega ( 2 E _ { c } ) \leq 2 \kappa E _ { c } = \sum _ { k } 2 \kappa \varepsilon _ { k } ^ { c } .
$$

If the comparator ranges over a finer reward domain than the planner’s menu, the difference between the two optima at $2 E _ { c }$ is added to η. The proof uses only $| c ( a ) - \widehat { c } ( a ) | \leq E _ { c }$ and $| v ( a ) - \widehat { v } ( a ) | \leq E _ { v }$ so (i) also holds with the direct errors

$$
\begin{array} { l l l } { \displaystyle \bar { E } _ { c } = \sum _ { l } \operatorname* { m a x } _ { a } \left| c _ { l } ( a ) - \widehat { c } _ { l } ( a ) \right| \leq E _ { c } , } \\ { \displaystyle \bar { E } _ { v } \leq E _ { v } } \end{array}
$$

in their place.

(ii) Lower bound. Step 1: the instance. Fix stage errors $( \varepsilon _ { k } ^ { c } , \varepsilon _ { k } ^ { v } ) _ { k }$ with $E _ { c } > 0$ (the case $E _ { c } = 0$ is treated in Step 5), any $\Delta > 0$ , and a band with $\bar { b } - \underline { { b } } > 4 E _ { c }$ and midpoint m. Order the stages as syn, mm, int, dec and write $\begin{array} { r } { s _ { k } ( \varepsilon ) = \sum _ { j < k } \varepsilon _ { j } } \end{array}$ for the partial sums along this order. Layer 1 has plans s and h with true $( c , v ) = ( m , 0 )$ and $( \bar { b } - E _ { c } , \Delta )$ . Plan s is exact at every stage; plan h has exact value and cost

$$
c ^ { ( k ) } ( h ) = \bar { b } - E _ { c } + s _ { k } ( \varepsilon ^ { c } ) ,
$$

so that ${ \widehat { c } } ( h ) = { \bar { b } }$ . Layer 2 has plans $x ,$ y of zero cost at every stage, true values 0 and $2 E _ { v }$ , and

$$
\begin{array} { l } { { v ^ { ( k ) } ( x ) = s _ { k } ( \varepsilon ^ { v } ) , } } \\ { { v ^ { ( k ) } ( y ) = 2 E _ { v } - s _ { k } ( \varepsilon ^ { v } ) , } } \end{array}
$$

so that $\widehat { v } ( x ) = \widehat { v } ( y ) = E _ { \iota }$ . Each stage error is exactly the prescribed one.

Step 2: the benchmark. In $\boldsymbol { B } _ { 2 E _ { c } }$ plan h is excluded. For this truth

$$
\begin{array} { r l } { V ^ { \star } ( 0 ) = \Delta + 2 E _ { v } \qquad } & { \mathrm { ( a t t a i n e d ~ b y ~ } ( h , y ) \mathrm { ~ a t ~ s p e n d ~ } \bar { b } - E _ { c } \in \mathcal { B } \mathrm { ) , } } \\ { V ^ { \star } ( 2 E _ { c } ) = 2 E _ { v } \qquad } & { \mathrm { ( v i a ~ } ( s , y ) ) , } \\ { \omega ( 2 E _ { c } ) = \Delta . } \end{array}
$$

Step 3: equalityfor the planner of(i). With $\eta = 0$ it cannot use h because $\widehat c ( h ) = \bar { b } \notin \mathcal B _ { E _ { c } }$ ; breaking the tie in layer 2 toward x it earns 0, so Eq. (8) holds with equality.

Step 4: every planner. Now take any deterministic planner that sees the final tables and the stage-error magnitudes and keeps the true spend in B for every consistent truth. The truth with $c ( h ) = \mathsf { \bar { b } } + E _ { c }$ (the same final tables, reached by decreasing stage steps) makes every portfolio with h infeasible, so the planner never selects h. In layer 2 its choice is a function of identical final tables, so the truth that gives value 0 to the chosen plan and $2 E _ { v }$ to the other is consistent; the planner earns 0 against $V ^ { \star } ( \breve { 0 } ) = \omega ( 2 E _ { c } ) + 2 E _ { v } .$ Abstaining also earns 0. A randomized planner loses at least $\omega ( 2 E _ { c } ) \overline { { + E _ { v } } }$ in expectation, by averaging over the two value truths. Without tightening, a planner that maximizes vb over ${ \widehat { c } } \in B$ selects $( h , \cdot )$ with estimated spend ${ \bar { b } } ,$ whose true spend is $\bar { b } + E _ { c }$ under the second truth. Adding a layer with two exact zero-cost plans of value 0 and $\eta ,$ an η-optimal planner may choose the first, so equality also holds for $\eta > 0$

Step 5: the case $E _ { c } = 0$ . Then $\begin{array} { r } { B _ { E _ { c } } = B , } \end{array}$ plan h would stay admissible and $\omega ( 0 ) = 0 ,$ , so drop h: layer 1 has only the exact plan s and layer 2 is as above. Now every cost is exact and $V ^ { \star } ( 0 ) = 2 E _ { v }$ attained via $( s , y )$ at spend $m \in B$ . The planner of (i) breaking the tie toward x earns 0, and by the argument for layer 2 any deterministic planner earns 0 against one of the two value truths. Hence the bound

$$
\omega ( 0 ) + 2 E _ { v } = 2 E _ { v }
$$

is attained.

(iii) Error floors of each omitted stage. All instances use exact tables for the stages that are kept.

Pilot correction. Take one layer with two offers whose payment functions coincide, true offer laws $P _ { a _ { 1 } }$ equal to the no-offer law and $P _ { a _ { 2 } }$ with expected completions larger by $U > 0$ , and a band containing both costs. Without the correction both synthetic laws equal the no-offer law, the two offers have identical tables and estimated uplift 0, and relabelling the offers makes any deterministic planner choose $a _ { 1 } \colon$ it loses the entire optimal uplift U.

History shape. By Lemma 2 two laws with the same mean and variance can differ by $1 / ( 1 + t ^ { 2 } )$ in the probability of reaching $\mu + t \sigma ,$ , so any estimate computed from the two moments alone errs by at least $1 / ( 2 ( \dot { 1 } + t ^ { 2 } ) )$ ) on one of them, and a fixed prize R at that threshold is mispriced by at least $R / ( 2 ( 1 + \dot { t } ^ { 2 } ) )$

Trajectory integration. Let $Y \in \{ 0 , 2 \tau \}$ with probability $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ each, known exactly, and a prize R paid when $Y \geq \tau$ . Evaluating the rule at the mean gives

$$
R \mathbf { 1 } \{ \mathbb { E } Y \geq \tau \} = R ,
$$

whereas

$$
\mathbb { E } [ R \mathbf { 1 } \{ Y \geq \tau \} ] = R / 2 .
$$

The probability that payment lies within ±20% of its mean $R / 2$ is 0, while the plug-in value is 1.

Joint allocation. Under the cap $\boldsymbol { B } = [ 0 , B ]$ with $L \geq 2$ layers, let layer 1 offer $( 0 , 0 )$ and $( B , U )$ and all other layers offer only $( 0 , 0 )$ . The joint optimum earns $U ;$ splitting the budget equally gives layer 1 at most $\dot { B } / L < B$ , so it earns 0. □

Lemma 2 (Two moments do not fix a threshold probability). For $t > 0$

$$
\operatorname { s u p } \left| \operatorname* { P r } ( S \geq \mu + t \sigma ) - \operatorname* { P r } ( S ^ { \prime } \geq \mu + t \sigma ) \right| = { \frac { 1 } { 1 + t ^ { 2 } } }
$$

over pairs of laws with common mean µ and variance $\sigma ^ { 2 } > 0 .$ , and the supremum is attained. For integer laws, $\{ 0 { : } \frac { 1 } { 2 } , 2 { : } \frac { 1 } { 2 } \}$ and $\{ 0 { : } { \frac { 1 } { 3 } } , 1 { : } { \frac { 1 } { 2 } } , \dot { 3 } { : } { \frac { 1 } { 6 } } \}$ both have mean and variance 1 but $\operatorname* { P r } ( S \geq 2 )$ equal to ${ \begin{array} { l } { { \frac { 1 } { 2 } } } \end{array} } a n d { \begin{array} { l } { { \frac { 1 } { 6 } } } \end{array} }$

Proof. Upper bound. For $u > 0$

$$
\operatorname* { P r } ( S - \mu \geq t \sigma ) \leq \frac { \mathbb { E } [ ( S - \mu + u ) ^ { 2 } ] } { ( t \sigma + u ) ^ { 2 } } = \frac { \sigma ^ { 2 } + u ^ { 2 } } { ( t \sigma + u ) ^ { 2 } } ,
$$

which equals $1 / ( 1 + t ^ { 2 } )$ at $u = \sigma / t ;$ both probabilities lie in $[ 0 , 1 / ( 1 + t ^ { 2 } ) ]$

Equality. $S = \mu + t \sigma$ with probability $1 / ( 1 + t ^ { 2 } )$ and $S = \mu - \sigma / t$ otherwise has mean $\mu$ and variance $\sigma ^ { 2 } ; S ^ { \prime } \mathrm { o n } \left\{ \mu + \alpha , \mu - \sigma ^ { 2 } / \alpha \right\}$ with $0 < \alpha <$ tσ and mass $\sigma ^ { 2 } / ( \alpha ^ { 2 } + \sigma ^ { 2 } )$ at $\mu + \alpha$ has the same moments and never reaches $\mu + t \sigma$ . The integer example is checked directly. □

## J.2 BOUNDS ON THE FOUR STAGE TERMS

Lemma 3 (Award probabilities carry every threshold rule). Let $Y \in \{ 0 , \ldots , M \}$ count eligible completions, D be a common attendance or eligibility event, $E _ { j } = \{ Y \geq \tau _ { j } \} \cap$ D with integers $1 \leq \tau _ { 1 } < \cdot \cdot \cdot < \tau _ { J } ,$ , and $G _ { F } ( k ) = F ( Y \geq k , D )$ . For laws F, F<sup>′</sup> write $\Delta ( k ) \stackrel { \smile } { = } | G _ { F } ( k ) - G _ { F ^ { \prime } } ( k ) |$ Forfixed prizes $0 = R _ { 0 } \leq R _ { 1 } \leq \cdot \cdot \cdot \leq R _ { J }$ paid at the highest rung reached,

$$
| \mathbb { E } _ { F } g - \mathbb { E } _ { F ^ { \prime } } g | \le \sum _ { j } ( R _ { j } - R _ { j - 1 } ) \Delta ( \tau _ { j } ) \le R _ { J } \operatorname* { m a x } _ { j } \Delta ( \tau _ { j } ) .
$$

For rates $0 = \rho _ { 0 } \le \cdots \le \rho _ { J }$ paid on all eligible completions at the highest rung reached,

$$
\begin{array} { c l } { \displaystyle \big | \mathbb { E } _ { F } g - \mathbb { E } _ { F ^ { \prime } } g \big | \leq \sum _ { j } ( \rho _ { j } - \rho _ { j - 1 } ) \Big \{ \tau _ { j } \Delta ( \tau _ { j } ) + \sum _ { k > \tau _ { j } } \Delta ( k ) \Big \} } \\ { \leq \rho _ { J } M \displaystyle \operatorname* { m a x } _ { k \geq \tau _ { 1 } } \Delta ( k ) . } \end{array}
$$

Expected completions satisfy

$$
| \mathbb { E } _ { F } Y - \mathbb { E } _ { F ^ { \prime } } Y | \leq \sum _ { k = 1 } ^ { M } | F ( Y \geq k ) - F ^ { \prime } ( Y \geq k ) | .
$$

Proof. Nested events give

$$
g = \sum _ { j } ( R _ { j } - R _ { j - 1 } ) \mathbf { 1 } \{ E _ { j } \}
$$

(fixed prizes),

$$
g = \sum _ { j } ( \rho _ { j } - \rho _ { j - 1 } ) Y \mathbf { 1 } \{ E _ { j } \}
$$

(rates).

With $p ( y ) = F ( Y = y , D )$ , summation by parts gives

$$
\begin{array} { l } { { \displaystyle \mathbb { E } _ { F } [ Y \mathbf { 1 } \{ E _ { j } \} ] = \sum _ { y \ge \tau _ { j } } y p ( y ) } } \\ { ~ } \\ { { \displaystyle = \tau _ { j } G _ { F } ( \tau _ { j } ) + \sum _ { k > \tau _ { j } } G _ { F } ( k ) } , } \end{array}
$$

and, in the same way,

$$
\mathbb { E } _ { F } Y = \sum _ { k \geq 1 } F ( Y \geq k ) .
$$

Take differences; at most $M - \tau _ { j }$ terms follow $\tau _ { j }$

Thus each table error of a threshold rule is controlled by errors in award probabilities $G ( k )$ . When D requires attendance on several days, G depends on the joint law of completions and attendance, not on their separate marginals: over two days with D = {both days active} and $\tau _ { 1 } = 3$ , the laws $\begin{array} { r } { \frac { 1 } { 2 } \delta _ { ( 2 , 1 ) } + \frac { 1 } { 2 } \delta _ { ( 3 , 2 ) } } \end{array}$ and the uniform law on $\{ ( \bar { 2 , 1 } ) , ( 2 , 2 ) , ( 3 , \mathsf { \bar { 1 } } ) , ( 3 , 2 \mathsf { \bar { ) } } \}$ of (completions, active days) share both marginals but give $\begin{array} { r } { G ( 3 ) = \frac { 1 } { 2 } } \end{array}$ and $\textstyle { \frac { 1 } { 4 } }$ . This is why integration uses joint trajectories.

Lemma 4 (Synthetic term). On a finite trajectory space let $p _ { 0 }$ be the no-offer law, ϕ the matched features, P a true offer law with $P \ll p _ { 0 } ,$ , and $Q _ { \lambda } \propto p _ { 0 } e ^ { \lambda ^ { \prime } \phi } . \ : I f \lambda ^ { \star }$ solves $\mathbb { E } _ { Q _ { \lambda ^ { \star } } } \phi = \mathbb { E } _ { P } \phi ,$ , then

$$
\mathrm { K L } ( P \| p _ { 0 } ) = \mathrm { K L } ( P \| Q _ { \lambda ^ { \star } } ) + \mathrm { K L } ( Q _ { \lambda ^ { \star } } \| p _ { 0 } ) ,
$$

and for every function f with range osc(f) = max f − min $f ,$

$$
\begin{array} { r } { | \mathbb { E } _ { P } f - \mathbb { E } _ { Q _ { \lambda ^ { \star } } } f | \le \operatorname { o s c } ( f ) \sqrt \frac 1 2 \{ \mathrm { K L } ( P \| p _ { 0 } ) - \mathrm { K L } ( Q _ { \lambda ^ { \star } } \| p _ { 0 } ) \} . } \end{array}
$$

Proof. Step 1: the identity.

$$
\begin{array} { r } { \mathrm { K L } ( P \| p _ { 0 } ) - \mathrm { K L } ( P \| Q _ { \lambda ^ { \star } } ) = \mathbb { E } _ { P } \log ( Q _ { \lambda ^ { \star } } / p _ { 0 } ) } \\ { = \lambda ^ { \star \top } \mathbb { E } _ { P } \phi - \log Z , } \end{array}
$$

and $\begin{array} { r } { \mathrm { K L } ( Q _ { \lambda ^ { \star } } | | p _ { 0 } ) = \lambda ^ { \star \top } \mathbb { E } _ { Q _ { \lambda ^ { \star } } \phi } - \log Z } \end{array}$ is the same number.

Step 2: the bound. Then $| \mathbb { E } _ { P } f - \mathbb { E } _ { Q } f | \le \sec ( f ) \operatorname { T V } ( P , Q )$ and Pinsker’s inequality $\mathrm { T V } \leq \sqrt { \mathrm { K L } / 2 }$ (Tsybakov, 2009, Lemma 2.5). □

The same identity applied to any $Q$ with the matched moments shows $\mathrm { K L } ( Q \| p _ { 0 } ) \geq \mathrm { K L } ( Q _ { \lambda ^ { \star } } \| p _ { 0 } )$ the tilt is the least change of history that reproduces the experiment (Csiszár, 1975). Without the correction the same argument gives only os $: ( f ) \sqrt { \mathrm { K L } ( P \| p _ { 0 } ) / 2 } ;$ matching removes $\mathrm { K L } ( Q _ { \lambda ^ { \star } } | | p _ { 0 } )$ from the radicand, and the term vanishes when the response is a tilt in $\phi .$ The bound covers piloted offers. At an unpiloted offer, $P ^ { \mathrm { s y n } }$ uses the interpolated parameter

$$
\lambda _ { t } ^ { \star } = ( 1 - t ) \lambda _ { j } ^ { \star } + t \lambda _ { k } ^ { \star }
$$

of its piloted neighbours $j ,$ , k (Section 3), so the synthetic term also contains the interpolation bias $| \mathbb { E } _ { P } \bar { f } - \mathbb { E } _ { Q _ { \lambda _ { \mathtt { f } } ^ { \star } } } f | .$ , which the lemma does not bound and which persists however large the pilot. For example, with $Y \sim \mathrm { B e r n o u l l i } ( \frac { 1 } { 2 } )$ under no offer, $\phi ( Y ) = Y$ and a true response $P _ { a } ( Y = 1 ) = \sigma ( a ^ { 2 } )$ inside the tilt family (σ the logistic function), pilots at $a = 0 . 2$ and 0.6 give $\lambda ^ { \star } = 0 . 0 4$ and 0.36 exactly, but the interpolated $\bar { \lambda = } 0 . 2 0$ at $a = 0 . 4$ 4 differs from the true 0.16, leaving an error of $\sigma ( 0 . 2 \dot { 0 } ) - \sigma ( 0 . 1 6 ) \approx 0 . 0 0 9 9$ in $\mathbb { E } Y$ with exact moments and exact integration. More pilot offers, or smoothness of λ in the offer, reduce this part.

Lemma 5 (Moment-matching term). Keep p<sub>0</sub> fixed and let $\Sigma _ { \lambda } = \operatorname * { C o v } _ { Q _ { \lambda } } ( \phi )$ satisfy $\kappa _ { \phi } I \preceq \Sigma _ { \lambda } \preceq$ $\Lambda _ { \phi } I , \kappa _ { \phi } > 0$ , on the segment between $\lambda ^ { \star }$ and the fitted $\widehat { \lambda } ,$ , whose moments mb are the pilot means. Then for every $f ,$

$$
\begin{array} { r l } & { \left. \mathbb { E } _ { Q _ { \widehat { \lambda } } } f - \mathbb { E } _ { Q _ { \lambda ^ { \star } } } f \right. \leq \frac { 1 } { 2 } \mathrm { o s c } ( f ) \Lambda _ { \phi } ^ { 1 / 2 } \lVert \widehat { \lambda } - \lambda ^ { \star } \rVert _ { 2 } } \\ & { \qquad \leq \frac { 1 } { 2 } \mathrm { o s c } ( f ) \frac { \Lambda _ { \phi } ^ { 1 / 2 } } { \kappa _ { \phi } } \left. \widehat { m } - m ^ { \star } \right. _ { 2 } . } \end{array}
$$

Ifeach of $J _ { p }$ piloted offers per layer receives n independent trajectories and $\phi \in [ 0 , 1 ] ^ { d }$ , then with probability at least $1 - \delta ,$ , simultaneouslyfor all layers and piloted offers,

$$
\lVert \widehat { m } - m ^ { \star } \rVert _ { 2 } \leq \sqrt { d \log ( 2 d L J _ { p } / \delta ) / ( 2 n ) } .
$$

Offers whose λ is interpolated linearly between piloted $o f f e r s ,$ , or between $\lambda = 0$ at the no-offer plan and a piloted offer, satisfy thefirst inequality (given the covariance bounds on their own segment) with $\lambda ^ { \star }$ the interpolated synthetic parameter and $\lVert \widehat { \lambda } - \lambda ^ { \star } \rVert _ { 2 }$ at most the larger error of their two neighbours; the comparison is with $P ^ { \mathrm { s y n } }$ , not with the projection of the offer’s own population moments.

Proof. Step 1: derivative along the segment. With $u = \widehat { \lambda } - \lambda ^ { \star }$ and ${ \lambda } _ { t } = { \lambda } ^ { \star } + t u .$

$$
\begin{array} { r l } & { \left| \frac { d } { d t } \mathbb { E } _ { Q _ { \lambda _ { t } } } f \right| = | \mathrm { C o v } _ { \lambda _ { t } } ( f , u ^ { \top } \phi ) | } \\ & { \qquad \leq \mathrm { s d } ( f ) \left( u ^ { \top } \Sigma _ { \lambda _ { t } } u \right) ^ { 1 / 2 } } \\ & { \qquad \leq \frac { 1 } { 2 } \mathrm { o s c } ( f ) \Lambda _ { \phi } ^ { 1 / 2 } \| u \| . } \end{array}
$$

Step 2: from parameters to moments. Also

$$
\widehat { m } - m ^ { \star } = \int _ { 0 } ^ { 1 } \Sigma _ { \lambda _ { t } } d t u
$$

with

$$
\int _ { 0 } ^ { 1 } \ d \Sigma _ { \lambda _ { t } } d t \succeq \kappa _ { \phi } I ,
$$

so $\| u \| \leq \| \widehat { m } - m ^ { \star } \| / \kappa _ { \phi }$

Step 3: the rate. Hoeffding’s inequality (Hoeffding, 1963) for each of the $d L J _ { p }$ coordinates with a union bound gives the rate.

Step 4: interpolated offers. Interpolation weights are convex, so the interpolated error is a convex combination of the neighbours’ errors. □

If the no-offer law is estimated from $N _ { 0 }$ history weeks, changing the base at fixed λ moves every expectation by at most

$$
\begin{array} { r } { 2 \sec ( f ) e ^ { s ( \lambda ) } \mathrm { T V } ( \widehat { p } _ { 0 } , p _ { 0 } ) , } \end{array}
$$

where

$$
s ( \lambda ) = \operatorname* { m a x } _ { z } \lambda ^ { \top } \phi ( z ) - \operatorname* { m i n } _ { z } \lambda ^ { \top } \phi ( z ) ,
$$

since, for $w = e ^ { \lambda ^ { \top } \phi }$

$$
\| Q - Q ^ { \prime } \| _ { 1 } \leq 2 \sum _ { z } w | p _ { 0 } - \widehat { p _ { 0 } } | / \mathbb { E } _ { p _ { 0 } } w ;
$$

and $\begin{array} { r } { \mathbb { E } \operatorname { T V } ( \widehat { p } _ { 0 } , p _ { 0 } ) \leq \frac { 1 } { 2 } \sqrt { | \mathcal { Z } | / N _ { 0 } } } \end{array}$ . With abundant history $( N _ { 0 } \gg n )$ this part is small; the measured moment term includes it.

This bound holds at fixed λ. The method, however, re-fits the tilt on $\widehat { p } _ { 0 }$ so that it matches the pilot moments again, and the change of λ must be controlled as well.

Lemma 6 (Estimated base and re-fitted tilt). Let $Q = Q _ { \lambda } [ p _ { 0 } ]$ be the tilt of $p _ { 0 }$ with moments $m ,$ let $Q ^ { \prime } = Q _ { \lambda } [ \widehat { p _ { 0 } } ]$ keep λ on the estimated base, and let $\widehat { Q } = Q _ { \widehat { \lambda } } [ \widehat { p } _ { 0 } ]$ be re-fitted to the same moments m. Assume support overlap, that $i s ,$ m lies in the interior of the convex hull of ϕ(supp pb0). Let the feature covariance of the tilts of pb0 satisfy $\widehat { \kappa } I \preceq \Sigma \preceq \widehat { \Lambda } I , \widehat { \kappa } > 0 ,$ , on the segment between λ and $\widehat { \lambda } .$ Put

$$
\tau = 2 e ^ { s ( \lambda ) } \mathrm { T V } ( \widehat { p } _ { 0 } , p _ { 0 } ) , \qquad D _ { \phi } = \Bigl ( \sum _ { i } \operatorname { o s c } ( \phi _ { i } ) ^ { 2 } \Bigr ) ^ { 1 / 2 } .
$$

Then for every $f ,$

$$
\big | \mathbb { E } _ { \widehat { Q } } f - \mathbb { E } _ { Q } f \big | \le \operatorname { o s c } \left( f \right) \tau \Big ( 1 + \frac { 1 } { 2 } D _ { \phi } \frac { \widehat { \Lambda } ^ { 1 / 2 } } { \widehat { \kappa } } \Big ) .
$$

Proof. Step 1: base change at fixed λ. By the display above, $\mathrm { T V } ( Q ^ { \prime } , Q ) \leq \tau$ , hence $| \mathbb { E } _ { Q ^ { \prime } } f - \mathbb { E } _ { Q } f | \le$ $\sec ( f ) \tau$

Step $2 \div$ moment shift. Let $m ^ { \prime }$ be the moments of $Q ^ { \prime }$ . Each coordinate moves by at most osc $( \phi _ { i } ) \tau ,$ , so $\| \bar { m ^ { \prime } } - m \| _ { 2 } \leq D _ { \phi } \bar { \tau }$

Step 3: re-fit on the estimated base. $Q ^ { \prime }$ and $\widehat { Q }$ are tilts of the same base $\widehat { p } _ { 0 }$ with moments $m ^ { \prime }$ and $m$ Lemma 5, whose proof uses only a fixed base and the covariance bounds on the segment, gives

$$
\begin{array} { r l } & { \displaystyle \left. \mathbb { E } _ { \widehat { Q } } f - \mathbb { E } _ { Q ^ { \prime } } f \right. \leq \frac { 1 } { 2 } \mathrm { o s c } ( f ) \frac { \widehat { \Lambda } ^ { 1 / 2 } } { \widehat { \kappa } } \left. m - m ^ { \prime } \right. _ { 2 } } \\ & { \qquad \leq \frac { 1 } { 2 } \mathrm { o s c } ( f ) \frac { \widehat { \Lambda } ^ { 1 / 2 } } { \widehat { \kappa } } D _ { \phi } \tau . } \end{array}
$$

The triangle inequality concludes.

The three conditions are the quantities that the diagnostics of Appendix E measure: support overlap (the target inside the range of the stored weeks), the weight range $e ^ { s ( \lambda ) }$ , which bounds N max $w _ { i }$ on N equally weighted stored weeks, and the conditioning $\widehat { \Lambda } / \widehat { \kappa }$ of the feature covariance. If the estimated base loses support on which the matched moments rely, κb becomes small or $\widehat { \lambda }$ fails to exist, and the bound grows accordingly; stability is claimed only where these quantities are controlled. The rates of Corollary 1 are stated for a given base; on the event that the conditions hold, the estimated base adds this term. When the corrected law is the tilt of the empirical stored-week law, as in Appendices D and E, the weighted sum over the stored weeks is exact for that law, and the finite history enters through pb in this lemma rather than through the integration term.

Experiment length. Lemma 5 needs the experiment to fix d moments at $J _ { p }$ offers per layer, so its error scales as $\sqrt { d / n }$ . A trial that estimates the complete law of every offer directly avoids the synthetic term, but the analogous bound for empirical frequencies is $\scriptstyle { \frac { 1 } { 2 } } { \sqrt { | { \mathcal { Z } } | / n _ { \mathrm { R } } } }$ per offer and must hold at all $| \mathcal { A } _ { l } |$ offers. Comparing these sufficient conditions, the trial needs roughly

$$
\frac { | \mathcal { A } _ { l } | | \mathcal { Z } | } { J _ { p } d }
$$

times as many rider-weeks for the same estimation error (48 in the controlled design with $| \mathcal { Z } | = 3 2$ $| \mathcal { A } _ { l } | = 9 , J _ { p } = 3 , d = 2 )$ , up to logarithms and $\Lambda _ { \phi } ^ { 1 / 2 } / \kappa _ { \phi }$ . The price is the synthetic term, which does not shrink with experiment length; this is a comparison of upper bounds, not a lower bound for trials. Lemma 7 (Integration term). Ifintegration averages S independent drawsfrom the moment-matched law for each of the $N = { \dot { \sum _ { l } | A _ { l } | } }$ layer–plan pairs, then with probability at least $1 - \delta$ every cost and value entry is within osc $\because \sqrt { \log ( 4 N / \delta ) / ( 2 S ) }$ of its exact value, where $\mathrm { o s c } _ { l }$ is the range of the integrandfor layer l; hence

$$
\varepsilon _ { \mathrm { i n t } } ^ { c } , \varepsilon _ { \mathrm { i n t } } ^ { v } \leq \sum _ { l } \mathrm { o s c } _ { l } \sqrt { \log ( 4 N / \delta ) / ( 2 S ) } .
$$

Summing exactly over afinite trajectory space gives $\varepsilon _ { \mathrm { i n t } } = 0 .$

Proof. Hoeffding’s inequality for each of the 2N entries and a union bound. Draws may be shared across plans (common random numbers): the bound is per entry and needs no independence across entries. □

Scope of the rates. Lemmas 4–7 concern fixed integrands; Appendix J.3 gives the rates in expectation, including the conditions under which they extend to the mean-centred stability score and an example in which they fail. The controlled experiment uses pure uplift as its primary endpoint for this reason.

Lemma 8 (Decision term). A budget-grid program with step h that rounds each layer cost to the nearest grid point and optimizes exactly over the rounded tables is an exact planner $( \eta = 0 )$ on tables with $\varepsilon _ { \mathrm { d e c } } ^ { c } \leq L h / 2$ and $\varepsilon _ { \mathrm { d e c } } ^ { v } = 0 .$ . Rounding increments above each layer’s minimum upward gives $0 \leq c ^ { \mathrm { d e c } } - c ^ { \mathrm { i n t } } < h$ per layer, so $\varepsilon _ { \mathrm { d e c } } ^ { c } \leq L h$ ; exact enumeration and the segment program on its declared monetary grid have $\varepsilon _ { \mathrm { d e c } } = 0$

Proof. Rounding moves each layer cost by at most $h / 2$ (respectively less than h) and leaves values unchanged; the program maximizes exactly over the rounded problem. □

Cross-day dependence. Suppose a member’s day-activity indicators are conditionally independent Bernoull $\mathrm { i } ( p ( \xi ) )$ ) given a shared factor $\xi$ over $H$ days, so A counts active days, and let $\hat { A } ^ { \mathrm { i n d } } \sim$ Bin(H, Ep(ξ)) be the independent-day law with the same daily marginals. Then, by Jensen’s inequality,

$$
\operatorname* { P r } ( A = H ) = \operatorname { \mathbb { E } } p ( \xi ) ^ { H } \geq ( \operatorname { \mathbb { E } } p ( \xi ) ) ^ { H } = \operatorname* { P r } ( A ^ { \mathrm { i n d } } = H ) ,
$$

strictly unless $p ( \xi )$ is constant. Since

$$
\mathbb { E } A = \mathbb { E } A ^ { \mathrm { i n d } } = \sum _ { d \geq 1 } \operatorname* { P r } ( A \geq d ) ,
$$

the two survival functions must cross: composing days independently understates the full-attendance requirement and overstates some lower one.

## J.3 RATES OF THE MOMENT-MATCHING AND INTEGRATION TERMS

The rates quoted in Section 3 hold in expectation for every table entry with a fixed integrand. They do not hold for a stability score whose band is centred at the law’s own mean payment: in the two examples below such an entry keeps an error of constant size while the pilot moments converge. This subsection states the rates with the conditions they need.

Setting. Fix a layer and a piloted offer. The trajectory space $\mathcal { Z }$ is finite, the base $p _ { 0 }$ has full support, and the features $\dot { \phi } : \mathcal { Z } \to [ \dot { 0 } , 1 ] ^ { d }$ do not all lie in one hyperplane. Write

$$
\begin{array} { r } { \begin{array} { c } { Q _ { \lambda } \propto p _ { 0 } e ^ { \lambda ^ { \top } \phi } , } \\ { \Sigma _ { \lambda } = \mathrm { C o v } _ { Q _ { \lambda } } ( \phi ) , } \\ { m ( \lambda ) = \mathbb E _ { Q _ { \lambda } } \phi . } \end{array} } \end{array}
$$

The pilot mean $\widehat { m }$ averages $\phi$ over n independent trajectories drawn from the true offer law $P ;$ when riders of one week share a shock, n counts independent weeks or blocks, not rider-weeks. With $m ^ { \star } = \mathbb { E } _ { P } \phi$

$$
\begin{array} { r c l } { { } } & { { } } & { { P ^ { \mathrm { s y n } } = Q _ { \lambda ( m ^ { \star } ) } , } } \\ { { } } & { { } } & { { P ^ { \mathrm { m m } } = Q _ { \lambda ( \widehat { m } ) } ; } } \end{array}
$$

the base is held fixed, and its estimation adds the term stated after Lemma 5. The closed ball $\bar { B } ( m ^ { \star } , \rho )$ lies in the interior of conv $\phi ( { \mathcal { Z } } )$ , and $\kappa _ { \rho }$ is the smallest eigenvalue of $\Sigma _ { \lambda ( m ) }$ over that ball. If mb is not interior, the fit is replaced by any law or by abstention, and the entry stays in its range.

The map $\lambda \mapsto m ( \lambda )$ is a bijection from $\mathbb { R } ^ { d }$ onto the interior of conv $\phi ( { \mathcal { Z } } )$ with a smooth inverse, so $\lambda ( m )$ is well defined and $\kappa _ { \rho } > 0$ . Indeed, $u ^ { \top } \Sigma _ { \lambda } u = \mathrm { V a r } _ { Q _ { \lambda } } ( u ^ { \top } \phi ) > 0$ for $u \ne 0$ . For interior m, the strictly convex function

$$
\psi ( \lambda ) = \log \sum _ { z } p _ { 0 } ( z ) e ^ { \lambda ^ { \top } \phi ( z ) } - \lambda ^ { \top } m
$$

satisfies, for every unit vector $u ,$

$$
\psi ( t u ) \geq t \{ \operatorname* { m a x } _ { z } u ^ { \top } \phi ( z ) - u ^ { \top } m \} + \log \operatorname* { m i n } _ { z } p _ { 0 } ( z ) \to \infty ,
$$

so it has a unique minimizer, where $m ( \lambda ) = m$ . The inverse function theorem gives smoothness, and continuity on the compact ball gives $\kappa _ { \rho } > 0$

Mean-centred stability. For a law q and a payment y $: \mathcal { Z } \to [ 0 , \bar { y } ]$ let

$$
\begin{array} { r l } & { M ( q ) = \mathbb { E } _ { q } y , } \\ & { \ S ( q ) = \{ z : \alpha M ( q ) \leq y ( z ) \leq \beta M ( q ) \} } \end{array}
$$

with $\begin{array} { r } { 0 < \alpha < 1 < \beta } \end{array}$ (the paper uses 0.8 and 1.2). The monetary stability score is $F ( q ) =$ $M ( q ) q ( S ( q ) )$ , and its margin

$$
\gamma ( q ) = \operatorname* { m i n } _ { z } \operatorname* { m i n } \{ | y ( z ) - \alpha M ( q ) | , | y ( z ) - \beta M ( q ) | \}
$$

is the distance from the band’s endpoints to the nearest payment value.

Corollary 1 (Rates of the moment-matching and integration terms). In the setting above let $M ^ { \star } =$ $M ( P ^ { \mathrm { s y n } } ) , \gamma = \gamma ( P ^ { \mathrm { s y n } } ) a n d r = \operatorname* { m i n } \{ \rho , \gamma \kappa _ { \rho } ^ { 1 / 2 } / ( \beta \bar { y } ) \}$

(a) Fixed integrand. For every f (payment, completions, uplift, an award indicator, or stability in a band with a fixed centre),

$$
\mathbb { E } | \mathbb { E } _ { P ^ { \mathrm { m m } } } f - \mathbb { E } _ { P ^ { \mathrm { s y n } } } f | \leq \mathrm { o s c } ( f ) \big \{ \sqrt { d } / ( 4 \sqrt { \kappa _ { \rho } n } ) + 2 d e ^ { - 2 n \rho ^ { 2 } / d } \big \} ,
$$

andfor $S$ independent drawsfrom $P ^ { \mathrm { m m } }$

$$
\mathbb { E } \big | \mathbb { E } _ { P ^ { \mathrm { i n t } } } f - \mathbb { E } _ { P ^ { \mathrm { m m } } } f \big | \le \mathrm { o s c } ( f ) / ( 2 \sqrt { S } ) .
$$

Summed over the finite menu, $\mathbb { E } \varepsilon _ { \mathrm { m m } } = O ( n ^ { - 1 / 2 } )$ and $\mathbb { E } \varepsilon _ { \mathrm { i n t } } = O ( S ^ { - 1 / 2 } )$ for these entries; exact summation gives $\varepsilon _ { \mathrm { i n t } } = 0$

(b) Centred band with a margin. $I f \gamma > 0$ , then

$$
\mathbb { E } | F ( P ^ { \mathrm { m m } } ) - F ( P ^ { \mathrm { s y n } } ) | \leq \bar { y } \big \{ \sqrt { d } / ( 2 \sqrt { \kappa _ { \rho } n } ) + 2 d e ^ { - 2 n r ^ { 2 } / d } \big \} .
$$

Given P<sup>mm</sup> with margin $\gamma ^ { \prime } > 0 ,$

$$
\mathbb { E } \lvert F ( P ^ { \mathrm { i n t } } ) - F ( P ^ { \mathrm { m m } } ) \rvert \le \bar { y } \bigl \{ 1 / \sqrt { S } + 2 e ^ { - 2 S \gamma ^ { \prime 2 } / ( \beta \bar { y } ) ^ { 2 } } \bigr \} .
$$

(c) No margin. $H \gamma = 0 ,$ , Cov $P \mathrm { s y n } \left( \phi , y \right) \neq 0$ and $\mathrm { C o v } _ { P } ( \phi )$ is nonsingular, then, as $n \to \infty$

$$
\begin{array} { r } { \mathbb { E } | F ( P ^ { \mathrm { m m } } ) - F ( P ^ { \mathrm { s y n } } ) |  \frac { 1 } { 2 } M ^ { \star } ( w _ { - } + w _ { + } ) > 0 , } \end{array}
$$

where $w _ { - }$ and $w _ { + }$ are the $P ^ { \mathrm { s y n } }$ -masses of the trajectories that pay exactly α $M ^ { \star }$ and $\beta M ^ { \star }$ .

For an offer whose tilt is interpolated between piloted offers, (a) and (b) hold with the constants of Lemma $g _ { ( c ) }$

Under (b) the rate is $n ^ { - 1 / 2 }$ , but it sets in only once $n \gtrsim d \beta ^ { 2 } \bar { y } ^ { 2 } / ( \kappa _ { \rho } \gamma ^ { 2 } )$ , so a small margin delays it. Neither (b) nor (c) affects the upper bound of Proposition 1(i), which uses the measured table errors, whatever their size. Centring the band at a quantity fixed before estimation, such as the requested spend share or the mean issued with the plan, turns the score into a fixed-integrand entry covered by (a).

Lemma 9 (Moment-matching term in expectation). Let $g _ { f } ( m ) = \mathbb { E } _ { Q _ { \lambda ( m ) } } f .$ . (a) For $m , m ^ { \prime } \in$ $\bar { B } ( m ^ { \star } , \rho )$

$$
\begin{array} { r } { | g _ { f } ( m ^ { \prime } ) - g _ { f } ( m ) | \leq \frac { 1 } { 2 } \mathrm { o s c } ( f ) \kappa _ { \rho } ^ { - 1 / 2 } \| m ^ { \prime } - m \| _ { 2 } . } \end{array}
$$

(b) $\mathbb { E } \| \widehat { m } - m ^ { \star } \| _ { 2 } \leq \sqrt { d / ( 4 n ) }$ , and $\operatorname* { P r } ( \| \widehat { m } - m ^ { \star } \| _ { 2 } > s ) \leq 2 d e ^ { - 2 n s ^ { 2 } / d } f o r s > 0 .$ . (c) Let an offer’s tilt be $\lambda _ { t } = ( 1 - t ) \lambda _ { j } ^ { \cdot } + t \lambda _ { k }$ for piloted neighbours $j , k$ (at the no-offer plan $\lambda = 0$ is exact), and let both pilot means lie in their balls, with $\kappa _ { \rho }$ the smaller ofthe two constants. Then

$$
\begin{array} { r } { \vert \mathbb { E } _ { Q _ { \widehat { \lambda } _ { t } } } f - \mathbb { E } _ { Q _ { \lambda _ { t } ^ { \star } } } f \vert \leq \frac { \sqrt { d } } { 4 } \mathrm { o s c } ( f ) \kappa _ { \rho } ^ { - 1 } \displaystyle \operatorname* { m a x } _ { i \in \{ j , k \} } \| \widehat { m } _ { i } - m _ { i } ^ { \star } \| _ { 2 } . } \end{array}
$$

Proof. (a) By the chain rule, $\nabla g _ { f } ( m ) = \Sigma ^ { - 1 } c$ with $\Sigma = \Sigma _ { \lambda ( m ) }$ and $c = \operatorname { C o v } _ { Q _ { \lambda ( m ) } } ( \phi , f )$ . Since $c ^ { \top } \Sigma ^ { - 1 } c$ is the variance of the best linear predictor of f from ϕ,

$$
\begin{array} { r l } & { \| \Sigma ^ { - 1 } c \| _ { 2 } ^ { 2 } \leq \kappa _ { \rho } ^ { - 1 } c ^ { \top } \Sigma ^ { - 1 } c } \\ & { \qquad \leq \kappa _ { \rho } ^ { - 1 } \mathrm { V a r } ( f ) } \\ & { \qquad \leq \kappa _ { \rho } ^ { - 1 } \mathrm { o s c } ( f ) ^ { 2 } / 4 . } \end{array}
$$

Integrate along the segment from m to $m ^ { \prime }$ , which stays in the ball.

(b) Since $\phi \in [ 0 , 1 ] ^ { d }$

$$
\begin{array} { r } { \begin{array} { l } { \mathbb { E } \| \widehat { m } - m ^ { \star } \| _ { 2 } ^ { 2 } = \operatorname { t r } \operatorname { C o v } _ { P } ( \phi ) / n } \\ { \leq d / ( 4 n ) . } \end{array} } \end{array}
$$

For the tail apply Hoeffding’s inequality (Hoeffding, 1963) to each coordinate at level $s / \sqrt { d }$ and take a union bound.

(c) Step 1: covariance bound. For $\phi \in [ 0 , 1 ] ^ { d }$

$$
\begin{array} { r } { \Sigma \preceq ( \operatorname { t r } \Sigma ) I \preceq \frac { d } { 4 } I . } \end{array}
$$

Along $\lambda _ { t } ^ { \star } + \tau u$ with $u = \widehat { \lambda } _ { t } - \lambda _ { t } ^ { \star }$ , the derivative of the expectation is $\mathrm { C o v } ( f , u ^ { \top } \phi )$ , and

$$
\begin{array} { r l } & { | \mathrm { C o v } ( f , u ^ { \top } \phi ) | \leq \frac { 1 } { 2 } \mathrm { o s c } ( f ) ( u ^ { \top } \Sigma u ) ^ { 1 / 2 } } \\ & { \qquad \leq \frac { 1 } { 2 } \mathrm { o s c } ( f ) \frac { \sqrt { d } } { 2 } \| u \| . } \end{array}
$$

Step 2: integrate along the segment. By convexity,

$$
\| u \| \leq \operatorname* { m a x } _ { i } \| \widehat { \lambda } _ { i } - \lambda _ { i } ^ { \star } \| .
$$

Integrating $d \lambda / d m = \Sigma ^ { - 1 }$ along the segment from m<sup>⋆</sup><sub>i</sub> to $\widehat { m } _ { i }$ gives

$$
\begin{array} { r } { \| \widehat { \lambda } _ { i } - \lambda _ { i } ^ { \star } \| \leq \kappa _ { \rho } ^ { - 1 } \| \widehat { m } _ { i } - m _ { i } ^ { \star } \| . } \end{array}
$$

Lemma 10 (Stability of the centred band). Let $q , q ^ { \prime }$ be laws on $\mathcal { Z } . \ ( i ) I f \beta | M ( q ^ { \prime } ) - M ( q ) | < \gamma ( q )$ then $S ( q ^ { \prime } ) = S ( q )$ and

$$
| F ( q ^ { \prime } ) - F ( q ) | \leq | M ( q ^ { \prime } ) - M ( q ) | + M ( q ) | q ^ { \prime } ( S ( q ) ) - q ( S ( q ) ) | .
$$

(ii) Let q have full support and ${ \cal M } ( q ) > 0 ,$ , let $Z _ { - }$ and $Z _ { + }$ be the trajectories paying exactly α $\iota M ( q )$ and $\beta M ( q )$ , and let $w _ { \pm } = q ( Z _ { \pm } ) . \ I f q _ { k } \to q$ with $M ( q _ { k } ) > M ( q )$ for all $k ,$ then

$$
F ( q _ { k } )  M ( q ) \{ q ( S ( q ) ) - w _ { - } \} ;
$$

$i f M ( q _ { k } ) < M ( q )$ for all k, then $F ( q _ { k } ) \to M ( q ) \{ q ( S ( q ) ) - w _ { + } \}$

Proof. (i) Write $M = M ( q ) , M ^ { \prime } = M ( q ^ { \prime } ) , \gamma = \gamma ( q )$ and $\Delta = | M ^ { \prime } - M | < \gamma / \beta$ . A trajectory in $S ( q )$ has αM $+ \gamma \leq y \leq \beta M - \gamma$ , so

$$
\begin{array} { c } { { \alpha M ^ { \prime } \leq \alpha M + \alpha \Delta < \alpha M + \gamma \leq y , } } \\ { { y \leq \beta M - \gamma < \beta M - \beta \Delta \leq \beta M ^ { \prime } . } } \end{array}
$$

A trajectory outside $S ( q )$ has either

$$
y \le \alpha M - \gamma < \alpha M ^ { \prime }
$$

or

$$
y \ge \beta M + \gamma > \beta M ^ { \prime } .
$$

Hence $S ( q ^ { \prime } ) = S ( q ) = : S$ and

$$
F ( q ^ { \prime } ) - F ( q ) = \left( M ^ { \prime } - M \right) q ^ { \prime } ( S ) + M \{ q ^ { \prime } ( S ) - q ( S ) \} .
$$

(ii) A trajectory with $y \notin \{ \alpha M , \beta M \}$ keeps its membership once $\beta | M ( q _ { k } ) - M |$ is below its distance to the endpoints, as in (i). If $M ( q _ { k } ) > \bar { M }$ , then α $M ( q _ { k } ) > y$ on $Z _ { - }$ and $\beta M ( q _ { k } ) > y \mathrm { o n } Z _ { + }$ , so eventually $S ( q _ { k } ) = S ( q ) \backslash Z _ { - }$ and

$$
F ( q _ { k } ) = M ( q _ { k } ) q _ { k } ( S ( q ) \setminus Z _ { - } ) \to M \{ q ( S ( q ) ) - w _ { - } \} .
$$

The other case is symmetric.

Proof of Corollary 1. (a) On $\{ \| { \widehat { m } } - m ^ { \star } \| \leq \rho \}$ , Lemma $^ { 9 ( \mathrm { a } , \mathrm { b } ) }$ bounds the expected difference by $\frac { 1 } { 2 } \mathrm { o s c } ( f ) \kappa _ { \rho } ^ { - 1 / 2 } \sqrt { d / ( 4 n ) }$ . Off this event the difference is at most $\operatorname { o s c } ( f )$ , and the event has probability at most $2 d e ^ { - 2 n \rho ^ { 2 } / d }$ . Given $P ^ { \mathrm { m m } } , \mathbb { E } _ { P ^ { \mathrm { i n t } } } f$ averages $S$ independent values with variance at most $\mathrm { o s c } ( f ) ^ { 2 } / 4$ . The interpolated case uses Lemma 9(c) in the same way.

(b) Moment term. On $G = \left\{ \left\| { \widehat { m } } - m ^ { \star } \right\| \leq r \right\}$ , Lemma $9 ( \mathrm { a } )$ with $f = y$ gives $\beta | M ( P ^ { \mathrm { m m } } ) - M ^ { \star } | \leq$ $\gamma / 2 < \gamma .$ Lemma 10(i), with Lemma 9(a) for $f = y$ and $f = \mathbf { 1 } _ { S ( P ^ { \mathrm { s y n } } ) }$ , then gives

$$
\begin{array} { r l } & { | F ( P ^ { \mathrm { m m } } ) - F ( P ^ { \mathrm { s y n } } ) | \leq \frac { 1 } { 2 } ( \bar { y } + M ^ { \star } ) \kappa _ { \rho } ^ { - 1 / 2 } \| \widehat { m } - m ^ { \star } \| } \\ & { \qquad \leq \bar { y } \kappa _ { \rho } ^ { - 1 / 2 } \| \widehat { m } - m ^ { \star } \| . } \end{array}
$$

Off G the difference is at most ${ \bar { y } } ,$ since $0 \leq F \leq \bar { y }$

Integration term. Hoeffding’s inequality gives

$$
\mathrm { P r } \big ( \beta | M ( P ^ { \mathrm { i n t } } ) - M ( P ^ { \mathrm { m m } } ) | \geq \gamma ^ { \prime } \big ) \leq 2 e ^ { - 2 S \gamma ^ { \prime 2 } / ( \beta \bar { y } ) ^ { 2 } } ;
$$

otherwise Lemma 10(i) applies, with $\mathbb { E } | \Delta M | \le \bar { y } / ( 2 \sqrt { S } )$ and $\mathbb { E } | \Delta q ( S ) | \le 1 / ( 2 \sqrt { S } )$

(c) Step 1: asymptotic normality. Almost surely $\widehat { m }  m ^ { \star }$ , and $\sqrt { n } ( \widehat { m } - m ^ { \star } )$ is asymptotically normal with covariance $\operatorname { C o v } _ { P } ( \phi )$

Step 2: delta method. The gradient of M with respect to the moments at $m ^ { \star }$ is $v =$ $\Sigma ^ { - \mathrm { 1 } } \mathrm { C o v } _ { P ^ { \mathrm { s y n } } } ( \phi , y ) \ \ne \ 0 ,$ , so $\bar { \sqrt { n } } \{ M ( P ^ { \mathrm { m m } } ) ~ - ~ M ^ { \star } \}$ is asymptotically normal with variance $v ^ { \top } \mathrm { C o v } _ { P } ( \phi ) v > 0$ , and each sign has limiting probability $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$

Step 3: the limit. By Lemma 10(ii) and continuity of $m \mapsto Q _ { \lambda ( m ) }$ , with $\Delta M = M ( P ^ { \mathrm { m m } } ) - M ^ { \star }$

$$
| F ( P ^ { \mathrm { m m } } ) - F ( P ^ { \mathrm { s y n } } ) | - M ^ { \star } \{ w _ { - } { \bf 1 } ( \Delta M > 0 ) + w _ { + } { \bf 1 } ( \Delta M < 0 ) \}  0
$$

in probability. The difference is bounded by y¯, so its expectation converges to $\scriptstyle { \frac { 1 } { 2 } } M ^ { \star } ( w _ { - } + w _ { + } )$ .

Two examples. Both fall under (c).

First example. If $Y \in \{ 1 , \frac { 3 } { 2 } \}$ with equal probabilities, then $\begin{array} { r } { M ^ { \star } = \frac { 5 } { 4 } } \end{array}$ , the band is $[ 1 , \textstyle { \frac { 3 } { 2 } } ]$ , and both payment values sit on its endpoints $\begin{array} { r } { ( w _ { - } = w _ { + } = \frac { 1 } { 2 } ) } \end{array}$ : shifting mass $t \neq 0$ between them gives

$$
\begin{array} { r } { F = ( \frac { 5 } { 4 } + \frac { t } { 2 } ) ( \frac { 1 } { 2 } + | t | ) \to \frac { 5 } { 8 } , } \end{array}
$$

whereas $\begin{array} { r } { F = \frac { 5 } { 4 } \mathrm { a t } t = 0 . } \end{array}$

Second example. If a unit prize is won with probability $\begin{array} { r l } { p ^ { \star } = } & { { } \frac { 5 } { 6 } } \end{array}$ , the prize sits on the upper endpoint $\begin{array} { r } { \beta M ^ { \star } = 1 ( w _ { + } = \frac { 5 } { 6 } ) } \end{array}$ , and the limit is $\begin{array} { r } { \frac { 1 } { \gamma } \cdot \frac { 5 } { 6 } \cdot \frac { 5 } { 6 } = \frac { 2 5 } { 7 \cdot } } \end{array}$ . This example runs through Eq. (5) with a nonsingular covariance: four trajectories $( \stackrel { \triangledown } { \boldsymbol { C } } , \stackrel { \triangledown } { \boldsymbol { D } } ) \in \stackrel { \iota \mathfrak { A } } { \left\{ ( 1 , 1 ) , ( 2 , 1 ) , ( 3 , 2 ) , ( 4 , 2 ) \right\} }$ , features $( C / 4 , D / 2 )$ , a uniform base, a prize for $\dot { C } \geq 3 ,$ , and truth $Q _ { \lambda }$ ⋆ with $\lambda ^ { \star } = ( 0 , 2 \log 5 )$

## J.4 WHAT THE LOWER BOUNDS COVER

Proposition 1(ii) is a statement about tables. Its instances perturb the entries of a fixed menu directly, and the truths consistent with the final tables range over all tables within the stage errors, not over laws produced by an I-projection, a finite pilot and finite sampling. The next proposition asks what the statistical chain itself attains.

Proposition 3 (Lower bounds attained by the statistical chain). Take one layer with two plans $x , y$ that pay nothing, so that every portfolio isfeasible and $\omega \equiv 0 ,$ , and a value integrand h.

(a) Synthetic stage, constant attained. Let p<sub>0</sub> and ϕ be as in Appendix J.3, and let $h : { \mathcal { Z } } \to [ 0 , 1 ]$ not be affine in $\phi .$ There $i s \bar { \varepsilon } > 0$ such that for every $\varepsilon \in ( 0 , \bar { \varepsilon } ]$ there are offer laws $P _ { x } , P _ { y }$ with the same I-projection $Q ^ { \star }$ and

$$
\mathbb { E } _ { P _ { x } } h - \mathbb { E } _ { Q ^ { \star } } h = \varepsilon = \mathbb { E } _ { Q ^ { \star } } h - \mathbb { E } _ { P _ { y } } h .
$$

With population moments, exact summation and exact enumeration, $\varepsilon _ { \mathrm { s y n } } ^ { v } = \varepsilon$ and the other terms vanish, and every deterministic planner that uses the laws only through their ϕ-moments has regret 2ε under one ofthe two labellings. Thus Eq. (8) holds with equalityfor the actual I-projection.

(b) Moment-matching stage, order attained. Let $\mathcal { Z } = \{ 0 , 1 \} , \phi ( z ) = h ( z ) = z$ and anyfull-support base, so that the tiltfamily contains every nondegenerate Bernoulli law and $P ^ { \mathrm { m m } }$ is the pilot frequency. Let $P _ { x } , P _ { y }$ be Bernou $l l i \dot { ( \frac { 1 } { 2 } \pm \delta ) }$ with $\dot { \delta } = ( 3 / ( 1 2 8 n ) ) ^ { 1 / 2 }$ , and let each plan receive n independent pilot trajectories. Every rule that picks a planfrom the pilot data has expected regret at least

$$
\delta / 2 = ( 3 / ( 5 1 2 n ) ) ^ { 1 / 2 }
$$

under one of the two labellings.

(c) Integration stage, order attained by sampling only. The construction of (b), with S independent draws from $P ^ { \mathrm { m m } }$ in place of the pilot, gives the S-draw integrator expected regret at least $( 3 / ( 5 1 2 S ) ) ^ { 1 / 2 }$ on some instance. Exact summation over a finite $\mathcal { Z }$ has no integration error (Lemma 7).

Proof. (a) Step 1: a direction invisible to the moments. Since h $\notin$ span $\{ 1 , \phi _ { 1 } , . . . , \phi _ { d } \}$ in $\mathbb { R } ^ { \mathcal { Z } }$ , its residual w after orthogonal projection onto that span is nonzero, with

$$
\begin{array} { c } { { \displaystyle \sum _ { z } w ( z ) = 0 , } } \\ { { \displaystyle \sum _ { z } w ( z ) \phi ( z ) = 0 , } } \\ { { \langle w , h \rangle = \| w \| ^ { 2 } ; } } \end{array}
$$

rescale w so that $\langle w , h \rangle = 1$

Step 2: the two laws. Let $Q ^ { \star } = Q _ { \lambda }$ for any λ and $\bar { \varepsilon } = \operatorname* { m i n } _ { w ( z ) \neq 0 } Q ^ { \star } ( z ) / | w ( z ) |$ . Then

$$
\begin{array} { l } { { P _ { x } = Q ^ { \star } + \varepsilon w , } } \\ { { P _ { y } = Q ^ { \star } - \varepsilon w } } \end{array}
$$

are laws with moments $\mathbb { E } _ { Q ^ { \star } } \phi .$ . By the identity of Lemma 4, for every q with these moments,

$$
\mathrm { K L } ( q \| p _ { 0 } ) = \mathrm { K L } ( q \| Q ^ { \star } ) + \mathrm { K L } ( Q ^ { \star } \| p _ { 0 } ) ,
$$

so $Q ^ { \star }$ is the I-projection of both, and their values are $\mathbb { E } _ { Q ^ { \star } } h \pm \varepsilon .$

Step 3: the regret. All later tables of x and y coincide, so a deterministic planner’s choice depends only on the labels, and the labelling under which it picks the plan with law $P _ { y }$ has regret 2ε. A randomized planner has regret at least ε averaged over the two labellings.

(b) Step 1: two-point reduction. Let $\mathsf { P } _ { 1 } , \mathsf { P } _ { 2 }$ be the laws of the 2n pilot outcomes under the two labellings. A rule ψ has regrets $2 \delta \mathsf { P } _ { 1 } ( \psi = y )$ and $2 \delta \mathsf { P } _ { 2 } ( \psi = x )$ , whose sum is at least

$$
2 \delta \{ 1 - \mathrm { T V } ( \mathsf { P } _ { 1 } , \mathsf { P } _ { 2 } ) \} .
$$

Step 2: bounding the distance. For $\delta \leq { \frac { 1 } { 4 } }$

$$
\mathrm { K L } ( \mathsf { P } _ { 1 } \| \mathsf { P } _ { 2 } ) = 4 n \delta \log \frac { 1 + 2 \delta } { 1 - 2 \delta } \leq \frac { 6 4 } { 3 } n \delta ^ { 2 } ,
$$

so Pinsker’s inequality gives

$$
\mathrm { T V } \leq \left( \frac { 3 2 } { 3 } n \delta ^ { 2 } \right) ^ { 1 / 2 } = \frac { 1 } { 2 } .
$$

(c) This is (b) with the $S$ draws as the data.

Part (a) shows that the I-projection itself attains the constant 2 of the synthetic term, and that this term is the price of compressing the pilot to d moments: $P _ { x }$ and $P _ { y }$ have different trajectory histograms, which a trial over every offer would observe. Parts (b) and (c) show that the $n ^ { - 1 / 2 }$ and $S ^ { - 1 / 2 }$ orders of Corollary 1 cannot be improved in general: the first by no use of the same pilot, the second only by giving up sampling.

Scope of the decomposition. Four kinds of loss are distinct, and only some are bounded. (1) Proposition 1(i) is a perturbation bound: it compares the planner with the best portfolio over the same layer menus. The value lost by restricting individuals to layer menus,

$$
\Delta _ { \mathrm { m e n u } } = V _ { \mathrm { i n d } } ^ { \star } - V ^ { \star } ( 0 ) ,
$$

with $V _ { \mathrm { i n d } } ^ { \star }$ the best true value over individual plans, is not bounded here, and neither are the ranking and roster choices of M1–M2; they are measured only against the layerings that were tested. (2) The loss from tightening the budget, $\omega ( 2 E _ { c } )$ , is bounded only under a margin or growth condition: on a finite menu $\bar { V } ^ { \star }$ is a step function, so $\omega ( 2 E _ { c } ) = 0$ when an optimal portfolio’s true spend lies in $B _ { 2 E _ { c } }$ and $\omega ( 2 E _ { c } ) \leq 2 \kappa E _ { c }$ when $\omega ( \delta ) \leq \kappa \dot { \delta }$ . The tightening itself needs $E _ { c } ,$ which is unknown when a plan is issued unless a certificate supplies it, and Algorithm 1 uses the untightened band. (3) Part (ii) of the proposition holds at the level of tables; Proposition 3 is the statement for the chain of laws. (4) Part (iii) and Proposition 3 assert that certain instances exist. They do not say that on a given instance an accurate stage cannot partly offset another: errors can cancel, and the direct error in Table 10 is below the sum of the stage errors. Nor do they say that realized regret grows whenever one stage term grows. That a stage is needed in practice rests on the ablations and holds relative to the replacements tested there.

K REGISTERED TEST FAMILIES

Table 22: Pre-specified decision rules and their outcomes (rules fixed before the runs, and except for the integrated study also the analysis scripts). Rider held-out validation (Appendix A.1): paired change, held-out minus reference, of plan-cost error and selection regret at budget fraction 1, 95% intervals over origins; the rule requires upper bounds ≤ 0.045 and ≤ 0.01. Synthetic trajectories and a short pilot (Section 5): the pre-specified primary arm is the framework with sampled integration (S = 2,048); differences of penalized relative regret at 10% scoring on the 18 response laws, 95% intervals over laws. Integrated study (Appendix C): the registered confirmatory family on 36 response laws; H1 is the framework minus the same-calendar trial in relative regret (band within 20% of the budget), H2 the geometric-mean speed-up of answering all 60 requests of an origin by reuse, one-off preparation charged, over re-running the whole pipeline per request; 95% intervals over laws. Its analysis script was completed after the run; H2 is computed on the registered 36 response laws. New generator (Appendix D): the same family on its 36 response laws, with the rules and the analysis script fixed before the run. Stronger controls, behaviour change, joint fit and population reduction (Appendix E, studies C–F): registered families of seven, five, five and three one-sided tests, Holm step-down at level 0.025 within each family; estimates with 95% law-bootstrap intervals, relative regret within 20% of the budget (study F: retained share of the individual-level optimum). Study E: full re-run after the correction described in Appendix E.
<table><tr><td>Held-out condition</td><td>Change in plan-cost error</td><td>Change in regret</td><td>Met</td></tr><tr><td>Unseen period (fit once, 16 origins)</td><td>+0.051[+0.002, +0.098]</td><td>+0.000 [+0.000, +0.000]</td><td>no</td></tr><tr><td>Unseen cities (four folds, 45 origins)</td><td>+0.003[-0.018, +0.020]</td><td>+0.000 [+0.000, +0.000]</td><td>yes</td></tr><tr><td>Pilot rule</td><td>1-week pilot</td><td>2-week pilot</td><td>4-week pilot</td></tr><tr><td>Near-lossless vs. 18-week trial (upper ≤ 0.05)</td><td>+0.156 [+0.114, +0.201] (no)</td><td>+0.093 [+0.061, +0.126] (no)</td><td>+0.059 [+0.039, +0.077] (no)</td></tr><tr><td>Beats same-calendar trial (upper &lt; 0)</td><td>-0.256 [—0.290, −0.222] (yes)</td><td>-0.160 [-0.186, -0.134] (yes) -0.076 [-0.109, -0.043] (yes)</td><td></td></tr><tr><td>Without history is worse (lower &gt; 0)</td><td>+0.371 [+0.266, +0.488] (yes) +0.388 [+0.261, +0.525] (yes) +0.380 [+0.258, +0.514] (yes)</td><td></td><td></td></tr><tr><td>Without moment matching is worse (lower &gt; 0)</td><td>+0.369 [+0.235, +0.499] (yes) +0.433 [+0.302, +0.561] (yes) +0.467 [+0.359, +0.574] (yes)</td><td></td><td></td></tr><tr><td>Without joint integration is worse (lower &gt; 0)</td><td></td><td>+0.296 [+0.226, +0.368] (yes) +0.376 [+0.299, +0.458] (yes) +0.428 [+0.374, +0.487] (yes)</td><td></td></tr><tr><td>Without budget allocation is worse (lower &gt; 0)</td><td>+0.581 [+0.516, +0.652] (yes) +0.643 [+0.568, +0.719] (yes) +0.676 [+0.612, +0.737] (yes)</td><td></td><td></td></tr><tr><td>Integrated-study rule</td><td>1-week pilot</td><td>2-week pilot</td><td>4-week pilot</td></tr><tr><td>H1: beats same-calendar trial within 20% (upper &lt; 0)</td><td>-0.171 [-0.201, -0.140] (yes)</td><td>—0.124[—0.160, -0.089] (yes)</td><td>-0.069[-0.104, -0.035] (yes)</td></tr><tr><td>H2: 60 requests faster with preparation (lower &gt; 1, same answers)</td><td>14.1 × [13.8, 14.5] (yes)</td><td>14.2× [13.8, 14.5] (yes)</td><td>14.3× [14.0, 14.7] (yes)</td></tr><tr><td>New-generator rule</td><td>1-week pilot</td><td>2-week pilot</td><td>4-week pilot</td></tr><tr><td>H1: beats same-calendar trial within 20% (upper &lt; 0) H2: 60 requests faster with</td><td>-0.289 [—0.311, -0.267] (yes)</td><td>-0.176 [—0.192, -0.160] (yes)</td><td>) -0.123 [−0.140, −0.108] (yes)</td></tr><tr><td>preparation (lower &gt; 1, same answers)</td><td>2.70×[2.67, 2.73] (yes)</td><td>2.72× [2.69, 2.75] (yes)</td><td>2.78× [2.76, 2.81] (yes)</td></tr><tr><td>Stronger-control rule</td><td>1-week pilot</td><td>2-week pilot</td><td>4-week pilot</td></tr><tr><td>H6.1: tilted trial minus trial (&lt; 0)</td><td>-0.075 [−0.078, −0.072] (yes)</td><td></td><td></td></tr><tr><td>H6.2: minus dose interpolation (&lt; 0)</td><td>-0.022 [−0.024, −0.021] (yes)</td><td></td><td></td></tr><tr><td>H6.3: minus normal, exact truncated moments (&lt; 0)</td><td>-0.053 [-0.055, -0.051] (yes)</td><td></td><td></td></tr><tr><td>H6.4: calendar shocks, minus trial (&lt; 0)</td><td>—0.267 [—0.274, -0.259] (yes)</td><td>-0.214[−0.222, -0.206] (yes)</td><td></td></tr><tr><td>H6.5: ESS deficit vs. cost error, Spearman (&gt; 0)</td><td>+0.415 [+0.400, +0.431] (yes)</td><td></td><td></td></tr><tr><td>H6.6: support 1,024 minus 16,384 weeks (&gt; 0)</td><td>+0.006 [+0.006, +0.006] (yes)</td><td></td><td></td></tr><tr><td>Behaviour-change rule</td><td>1-week pilot</td><td>2-week pilot</td><td>4-week pilot</td></tr><tr><td>H7.1–H7.3: request-aware minus direct reuse (&lt; 0)</td><td>-0.195 [—0.202, -0.189] (yes)</td><td>−0.255 [−0.261, -0.249] (yes) -0.300 [−0.306, -0.293] (yes)</td><td></td></tr><tr><td>H7.4: same, selected subpopulations (&lt; 0)</td><td>-0.347 [-0.356, -0.338] (yes)</td><td></td><td></td></tr><tr><td>H7.5: request-aware minus</td><td>-0.129 [−0.136, −0.123] (yes)</td><td></td><td></td></tr><tr><td>trial, 288 laws (&lt; 0) Joint-fit rule (study E)</td><td>1-week pilot</td><td>2-week pilot</td><td>4-week pilot</td></tr><tr><td>H9.1: framework minus joint</td><td>t −0.055 [−0.059, −0.051] (yes) 420.052 [−0.056, −0.048] (yes) −0.060 [−0.066, −0.053] (yes)</td><td></td><td></td></tr><tr><td>fit of the trial (&lt; 0) H9.2: joint fit minus per-offer —0.152 [—0.158, —0.146] (yes) tilt of the trial (&lt; 0)</td><td></td><td></td><td></td></tr></table>