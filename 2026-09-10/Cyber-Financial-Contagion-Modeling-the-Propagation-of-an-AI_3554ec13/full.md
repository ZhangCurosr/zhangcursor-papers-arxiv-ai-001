# Cyber-Financial Contagion: Modeling the Propagation of an AI Vendor Compromise Through the Banking System

Alex Leytes

NICE Actimize

Alex.Leytes@Actimize.com

Abstract—The banking system now depends on a small set of shared artificial intelligence vendors for fraud screening, credit decisioning, anti-money-laundering triage, customer analytics, and internal decision support. This paper studies how a compromise inside one of those vendors can propagate along a chain of operational, informational, and financial linkages until it triggers losses that look, from the outside, like a classical banking crisis. We build a four-layer heterogeneous network that couples AI vendors, financial institutions, interbank exposures, and customer accounts, and we propose CFC-Prop, a stochastic epidemic-andclearing model that runs on that network. On a synthetic dataset with 60 vendors, 220 banks, roughly 2,500 vendor-bank service edges, and 1,400 interbank exposures, CFC-Prop reproduces the heavy-tailed loss distributions and the sharp dependence on patch latency that are consistent with prior cyber-financial evidence. We also train an early-warning model, CFC-GNN, that uses vendor-side incident telemetry and graph structure to flag highcascade-risk vendors before impact. Across four baselines the proposed model reaches AUROC 0.82 and AUPRC 0.60 while keeping calibration errors bounded. We release the full code, synthetic data, and reproducible scripts. The results argue that cyber concentration among AI vendors is a first-order financialstability problem and give supervisors a concrete quantitative tool for reasoning about it.

Index Terms—Cyber risk, financial contagion, systemic risk, AI supply chain, graph neural networks, third-party dependency, operational risk, banking.

## I. INTRODUCTION

ANKS have quietly become software companies with every customer interaction, credit decision, and monitoring alert through a stack of machine-learning services that are, in most cases, not built in-house. Foundation-model providers, feature-store vendors, MLOps platforms, and specialized inference hosts sit between the bank and its customers. A single compromise inside one of those vendors can, in principle,

distort the outputs used by dozens of institutions at the same time, and the resulting operational and reputational shock can propagate to counterparties along the interbank network before regulators have time to notice.

Prior work on financial contagion has focused on direct credit exposures [1], [2], [3], fire-sale spillovers [4], [5], and the topology of interbank obligations [6], [7], [8]. A parallel literature has documented the operational risk of shared technology vendors [9], [10], [11], [12]. The specific channel that connects the two, in which a machine-learning service is the shared dependency and the resulting modelintegrity or availability failure is the shock, has received far less structured treatment. Recent empirical work on cyber events [13], [14], [15], [16], [17] argues that the direction is plausible, but the modelling tools to reason about it end-to-end are still scattered across disciplines.

This paper contributes on three fronts.

A concrete four-layer model. We assemble a heterogeneous graph that stitches together AI vendors, banks, interbank exposures, and customer accounts (Fig. 1). The vendor layer follows the concentrated, heavy-tailed pattern seen in commercial AI infrastructure. The interbank layer follows scale-free bilateral exposure statistics consistent with [4], [18], [19].

A coupled propagation dynamic (CFC-Prop). We propose a stochastic dynamic that combines a susceptible-infectedrecovered (SIR) process on the vendor layer [20], [21] with a Furfine-style clearing cascade with fire-sale price impact on the interbank layer [22], [2], [5]. The dynamic yields, per compromise scenario, joint trajectories of vendor infection, bank impairment, defaults, customers affected, and total losses.

An early-warning learner (CFC-GNN). We train a hybrid graph-boosting model that uses vendor-side telemetry, incident history, and vendor-bank exposure structure to flag which vendors would trigger the largest cascade under a hypothetical compromise. The model is compared to logistic-regression, random-forest, gradient-boosting, and multilayer-perceptron baselines, and it improves discrimination and calibration on a held-out synthetic dataset. This complements our earlier work on fraud detection, intrusion detection, and secure federated analytics [23], [24], [25], [26], [27], but shifts the unit of analysis from the individual customer to the vendor as a systemic node.

The paper is organized as follows. Section II reviews the two literatures we bridge. Section III defines the threat model and the four-layer graph. Section IV specifies CFC-Prop and CFC-GNN. Section V describes the synthetic data, the generative assumptions, and the ablation grid. Section VI reports the experiments. Section VII works through an end-to-end case study. Section VIII lays out the theoretical properties of CFC-Prop. Section IX proposes a supervisory framework built on the two models. Section X reports robustness experiments. Section XI discusses limitations and open questions. Section XII concludes.

![](images/cb84a556227bd8fd7e401fdc63ec2c9b89695381580015b541b55817cda9e868.jpg)  
Fig. 1: Four-layer view of cyber-financial contagion from an AI vendor compromise. A shock at the vendor layer distorts model outputs used by many institutions, degrades operations, and only later shows up as balance-sheet loss.

## A. Why this problem is different

The classical financial-contagion problem takes the shock as exogenous and asks how it moves along interbank obligations. The cyber-financial problem is different in three ways. First, the shock is endogenous to a technology market whose concentration properties are set outside the banking sector. Second, the shock does not always leave a balance-sheet footprint immediately; a model-integrity attack of the kind described in [28], [29], [30] can degrade credit-decision quality or fraudtriage accuracy for weeks before losses accumulate to a level that a supervisor can measure. Third, the same vendor is often used by direct competitors, which means that idiosyncratic risk from a shared vendor is correlated by construction across the institutions the supervisor is trying to protect. These three features imply that classical stress testing under exogenous, immediately observable, and institution-idiosyncratic shocks is a lower bound on the problem, not an upper bound. Our model is built to make the additional structure explicit.

## II. RELATED WORK

## A. Financial contagion on networks

The modern network view of financial contagion begins with [1], who show that the pattern of interbank exposures determines whether an idiosyncratic shock is absorbed or amplified. [2] formalize the clearing problem, and [22] pioneers the simulation-based estimation of contagion using bilateral exposure data. [3], [4], [19] extend the argument to broader network structures, and [6], [7] give conditions under which more densely connected networks are, counterintuitively, more fragile. [8] introduces the DebtRank centrality that has since become standard in supervisory stress testing. [31] argues that direct contagion alone is unlikely to explain observed crises without amplifying mechanisms such as fire sales [5] and runs [32], [33]. Empirical support is found in [18].

## B. Cyber risk in financial systems

[34], [15], [14], [17] argue that cyber shocks have contagion properties similar to those studied in the credit-network literature, but through different channels: loss of availability, corruption of data, and loss of trust. [13], [16] give empirical estimates of how a single cyber event propagates through firm supply chains. Regulatory work in [9], [10], [11], [35], [12] identifies third-party dependency and cloud concentration as the specific structural features that most concern supervisors.

## C. Machine-learning supply-chain risk

The security of the machine-learning supply chain is a comparatively young field. [30], [29] give the canonical taxonomy of adversarial and integrity attacks against models. [28] shows that a supplier can backdoor a model before delivery. [36], [37] highlight training-data extraction and membership inference as data-side risks. [38], [39], [40] describe the software supplychain analogue. [41] catalogues these risks for large language model services in particular. In the financial-crime context, our earlier work builds detection primitives that assume the presence of intact models [23], [42], [24], [43], [44], [27], [25], [26]; the present paper studies the systemic consequences of exactly the assumption failing at scale.

## D. Graph learning on financial systems

Graph neural networks have become a standard tool for structured financial problems [45], [46], [47], [48], [49]. Prior applications include fraud rings, transaction screening, and anti-money-laundering ranking [42], [24]. We reuse those primitives in the vendor-risk setting: the object of learning is a vendor node whose blast radius depends on the neighborhood of banks it serves.

## E. Cybersecurity data science

[50], [51] survey the machine-learning tooling for intrusion detection and threat analytics. Our early-warning module is closest in spirit to that literature but is aimed at a supervisory audience rather than an operational security-operations center.

## F. Positioning

Against this backdrop, the contribution of this paper is a concrete stitching of three sub-fields that have so far been treated separately: financial-network contagion [1], [2], [4], [31], cyber operational risk [14], [15], [13], [16], and machinelearning supply-chain security [30], [28], [41], [36]. Neither the epidemic model of vendor infection nor the clearingwith-fire-sale model of interbank cascade is new individually. Coupling them on a heterogeneous graph in which the shared dependency is an AI service, and pairing them with a supervisory early-warning learner that ingests both vendor telemetry and graph structure, is, to the best of our knowledge, new.

## III. THREAT MODEL AND SETTING

## A. Actors and layers

We consider four layers, indexed $\mathsf { V } , \mathsf { B } , \mathsf { E } _ { \mathrm { i b } } , \mathsf { C } \colon$

$\mathsf { V } \colon$ a set of AI/ML vendors, each with a criticality score $c _ { \nu } ,$ a market share $m _ { \nu } ,$ and an average patch latency $\ell _ { \nu } .$

• B: a set of financial institutions, each with total assets $A _ { b } ,$ a regulatory capital ratio $\kappa _ { b } ,$ and an AI-dependency score $d _ { b } \in [ 0 , 1 ]$

$\mathsf { E } _ { \mathrm { i } \mathrm { b } } \subseteq \mathsf { B } \times \mathsf { B } \colon$ bilateral interbank exposures with weights $w _ { i j } = { \mathrm { E x p o s u r e } } ( i \to j )$

• C: customer accounts aggregated per bank, with count nb and average deposit d<sup>¯</sup><sub>b</sub>.

Vendor–bank service edges $\mathsf E _ { \mathrm { v b } } \subseteq \mathsf V \times \mathsf B$ carry a serviceexposure weight $s _ { \nu b }$ that captures the fraction of the bank’s AI-driven operations that pass through the vendor.

## B. Threat model

The attacker successfully compromises one or more vendors in V. The compromise can take three forms: (i) an availability attack that drops the vendor’s service capacity, (ii) a modelintegrity attack such as a backdoor or a poisoned update in the style of [28], [29], or (iii) a data-exfiltration attack that indirectly forces customers off the platform [36], [37]. We do not distinguish between the three at the vendor layer; they all show up downstream as an operational impairment at the affected banks.

## C. Assumptions

The model assumes: (A1) impairment at a bank grows with vendor service-exposure and with the bank’s AI-dependency; (A2) impaired banks pass losses to their interbank counterparties in proportion to bilateral exposures, subject to firesale amplification [5], [4]; (A3) supervisors observe vendor incident telemetry and bank operational metrics but do not observe the exploit itself; and (A4) patch latency is a first-order operational parameter set by the vendor, not by the bank.

## IV. THE CFC-PROP AND CFC-GNN MODELS

## A. CFC-Prop: coupled epidemic and clearing dynamics

Let St, $I ^ { t } , R ^ { t } \in \{ 0 , 1 \}$ denote susceptible, infected, and v v v recovered vendor states at time t. Let $x _ { \ b } ^ { t } \in [ 0 , 1 ]$ denote bank $b ^ { \prime } \mathbf { s }$ impairment level and $y _ { k } ^ { t } \in \{ 0 , \ 1 \}$ its default state. The dynamics per step are:

a) Vendor infection: For an infected vendor $v \in I ^ { t } ,$ every co-vendor u that shares a bank with v becomes infected with probability α:

$$
\mathrm { P r } [ S _ { u } ^ { t } \to I ^ { t + 1 } ] = 1 - \begin{array} { c } { { \mathsf { I T } } } \\ { { ( 1 - \alpha ) . } } \\ { { \nu \in I ^ { t } \cap \mathsf { N } ( u ) } } \end{array}\tag{1}
$$

An infected vendor recovers with probability γ per step, reflecting patch deployment; recovered vendors are immune for the horizon considered.

Vendor-institution bipartite network (sampled edges)

![](images/5805d8982565c9f363614795be0e5cc3bdd61c2e9894792764dbae86926df430.jpg)  
Fig. 2: Bipartite view of the vendor–bank layer, sampled at 500 edges for legibility. A handful of teal vendor nodes on the inner ring account for most bank connections; this concentration is the structural precondition for cyber-financial contagion.

![](images/7d4b44ec0d3b176985ac14fb6a465aca337445559b4b3272356dfbddbec130f5.jpg)  
Fig. 3: Distribution of the number of downstream banks per vendor in the synthetic graph. The right tail is what makes a targeted attack on a single vendor systemically relevant.

b) Bank impairment: For every $( v , b ) \in \mathsf { E } _ { \mathrm v { b } }$ with $v \in I ^ { t }$ and $y _ { h } ^ { t } = 0$ , bank $b ^ { \prime } \mathbf { s }$ impairment rises by an increment drawn from U(0.10, 0.35) with probability

$$
h _ { \nu b } ^ { t } = \beta \cdot \frac { s _ { \nu b } } { A _ { b } } \cdot ( 0 . 5 + d ) ,\tag{2}
$$

capturing exposure size and AI-dependency.

c) Financial clearing: For every interbank edge $( i , j ) \in$ $\mathsf { E } _ { \mathrm { i b } } , \mathsf { i f } \ x _ { i } ^ { t } \ge \tau \operatorname * { o r } _ { \sim } y _ { \iota _ { 1 } } ^ { t } = 1 , j$ $\underset { i } { \operatorname { o r } } \ \underset { i } { y ^ { t } } { = } 1 \ \underset { i } { \overset { \quad } { \mathrm { = } } } 1 , \underset { i } { j } { \overset { \mathrm { r e } } { = } } 1$ ceives a shock $\Delta ^ { t } { } _ { i } \ = . w _ { i j } \cdotp$ $\eta \cdot \widetilde { x } _ { _ { i } } ^ { t }$ , where and $\bar { { x ^ { t } } } _ { i }$ otherwise, and $\eta$ is the capital-haircut parameter. Bank j defaults when the total inbound shock exceeds a fraction $( 1 ~ - ~ \phi )$ of its regulatory capital $\kappa _ { \mathrm { \it { j } } } A _ { \mathrm { \it { j } } } ,$ where $\phi$ is a fire-sale amplification parameter following [4], [5].

d) Outputs: For each realization we record vendor infections, impaired banks, defaults, customers affected, and total system loss over a T -day horizon.

Algorithm 1 CFC-Prop single-run cascade   
1: Input: graph $\mathsf { G } ,$ seed vendor v0, params   
$( \alpha , \beta , \gamma , \eta , \phi , \tau , T )$   
2: I ← {v0}; $x _ { b } \gets 0$ and $y _ { b } \gets 0$ for all b   
3: for $t = 1$ to T do   
4: propagate vendor infection via α; recover with prob.   
$\gamma$   
5: for each infected vendor v and each $( v , b ) \in \mathsf { E } _ { \mathrm v { b } }$ do   
6: with prob. hvb, xb ← min(1, xb + U (0.10, 0.35))   
7: end for P   
8: $\begin{array} { r } { \Delta _ { j } \gets \dot { \mathbf { \phi } } _ { i : x _ { i } \geq \tau \mathrm { ~ o r ~ } y _ { i } = 1 } w _ { i j } \eta \widetilde { x } _ { i } } \end{array}$   
9: for each bank $j$ do   
10: if $\Delta _ { j } > ( 1 - \phi ) \kappa _ { j } A _ { j }$ and $y _ { j } = 0$ then $y _ { j } \gets 1$   
11: end if   
12: end for   
13: record trajectory statistics at t   
14: end for   
15: return trajectory

## B. CFC-GNN: an early-warning learner

The supervisory question is not just “what happens if vendor v is compromised,” but “which vendor should I watch most closely.” We frame this as a node-level classification task on the vendor layer. Features per vendor include:

• Static attributes: criticality $c _ { \nu } ,$ patch latency $\ell _ { \nu _ { 3 } }$ market share $m _ { \nu _ { 9 } }$ and one-hot vendor type.

• Incident telemetry: count of past incidents, mean and max severity, mean detection lag, historical cascade rate.

• Graph features: bank degree, weighted service-exposure sum, one-hop neighborhood assets.

The target label is whether the vendor would trigger a bank cascade in an independent hold-out simulation window. We fit four baselines (logistic regression, random forest, gradient boosting, multilayer perceptron) and a CFC-GNN model that stacks gradient-boosting and MLP predictions with a graphdegree adjustment. The ensemble mirrors the structure used successfully in related fraud and intrusion problems [23], [44], [24], [25], adapted to a vendor-node input.

## C. Coupling

CFC-Prop couples the vendor and bank layers by treating each infected vendor as a persistent source of impairment for its downstream banks until the vendor recovers, and by treating each impaired bank as a persistent source of shock for its interbank counterparties until it either recovers or defaults. This is a discrete-time analogue of the coupled cascade dynamics analyzed in [3], [4], adapted to the case where the primary shock originates outside the interbank network.

## D. Customer-run channel (extension)

An optional extension adds a deposit-run channel: at each step, every customer of a bank with impairment $x _ { b } ^ { t } \geq 0 . 4$ withdraws with probability $p { r } .$ . Cumulative withdrawals convert to a liquidity shock proportional to average deposit; when the cumulative shock exceeds a fraction of the bank’s short-term funding, the bank is forced into an asset sale that adds $\eta _ { \mathrm { f s } ^ { - } }$ scaled loss to inbound shocks at counterparties. Algorithm 2 sketches the extension. In the base experiments we set $p _ { r } = 0 ;$ the sensitivity is reported in Section X.

```latex
Algorithm 2 Customer-run channel (extension)
1: for each bank b with $x b \ge 0 . 4$ do
2: $W _ { b } \gets W _ { b } + \mathrm { B i n o m i a l } ( n _ { b } , p _ { r } ) \cdot d _ { \textit { b } }$
3: if $W _ { b } > \lambda _ { b }$ then
4: fire-sell fraction $\phi _ { b }$ of assets; loss $L _ { b } \gets \eta _ { \mathrm { f s } } \phi _ { b } A _ { b }$
5: add $L _ { b }$ to inbound shock at every counterparty of
$b$
$_ 6 \colon$ end if
7: end for
Algorithm 3 CFC-GNN training and scoring
1: Input: feature matrix X, labels $y ,$ graph ${ \sf G } _ { \mathrm { v b } }$
2: Compute degree and one-hop asset sums from ${ \sf G } _ { \mathrm { v b } } ;$ append
to $X$
3: Split into train/test with stratification on y
4: Fit f<sub>GBM</sub> and $f _ { \mathrm { M L P } }$ on training set
5: $p \widehat { \cdot }  0 . 5 5 f _ { \mathrm { G B M } } ( X ) + 0 . 4 5 f _ { \mathrm { M L P } } ( X )$
6: $\begin{array} { r } { p \hat { \mathbf { \Xi } }  \mathrm { c l i p } ( \hat { p } + 0 . 0 2 \cdot \underbrace { \mathrm { d e g } _ { \upsilon } } _ { \mathsf { m a x } \theta \in \mathcal { \mathsf { q } } } , 0 , 1 ) } \end{array}$
7: return risk scores pˆ
```

## V. SYNTHETIC DATA

We build the graph and the associated telemetry with a fully documented Python pipeline (Section VI-J). Vendor criticality is drawn from a Pareto distribution to match the empirical concentration in commercial AI infrastructure, and vendor market shares from a Dirichlet distribution. Bank assets follow the mixed distribution given by tier (G-SIB, Regional, Community), with capital ratios drawn from N (0.135, 0.02<sup>2</sup>) and truncated to Basel III lower bounds. The interbank layer is generated via preferential attachment weighted by assets, producing the scale-free shape observed in [4], [18]. Incident telemetry combines a base-rate Poisson process on the vendor with a severity-driven cascade label whose parameters are tuned so that top-decile-criticality vendors cascade in roughly one third of independent simulations, consistent with the concentration risk highlighted in [9], [10], [11]. Operational telemetry (per-hour latency and error rate) is generated for one calendar month per bank with injected stealth-degradation windows in 8% of banks.

The pipeline emits the seven CSV files listed in Table I and the graph statistics in Table II.

## A. Explicit generating distributions

For completeness, we list the distributions used at each layer. Vendor criticality follows a shifted Pareto: $c _ { \nu } \sim$ $\textit { 4 + 8 }$ · Pareto(1.6), truncated at 100. Vendor market shares follow a symmetric Dirichlet: $( m _ { 1 } , \ldots , m _ { \vert \vee \vert } ) ~ \sim$ $\operatorname { D i r } ( 0 . 3 , \dots , 0 . 3 )$ , and vendor patch latency in days is $\ell _ { \nu } \sim$ Gamma(2.1, 6.0) truncated at [1, 90]. Bank assets in USD bn follow Uniform(900, 3800) for G-SIBs, Uniform(30, 400) for regionals, and Uniform(0.4, 20) for community banks. Capital ratios are N (0.135, 0.02<sup>2</sup>) truncated at [0.08, 0.22]. AI-dependency is db ∼ Beta(2.2, 2.6). Vendor-bank edges are drawn by preferential attachment weighted by vendor criticality, with per-bank in-degree Uniform(1, 22) scaled by db. Interbank edges are drawn by preferential attachment weighted by assets. Incident severity is Beta(2, 5) and detection lag is Gamma(1.7, 4.0); the cascade label is a Bernoulli variable with logit $- 2 . 6 + 3 . 4 \ : \mathrm { s e v } + 0 . 0 4 5 \ : c _ { \nu } + 0 . 0 2 1 \ : \ell _ { \nu } .$ Operational latency baselines are $4 0 \mathrm { ~ + ~ } 6 0 d _ { b }$ ms with a 6 ms diurnal sinusoid, and injected degradation windows are chosen for 8% of banks with a random start and a 60-hour width.

TABLE I: Synthetic data files.  
TABLE III: Per-tier bank population characteristics.
<table><tr><td>File</td><td>Rows</td><td>Contents</td><td>Tier</td><td>Count</td><td>Mean 8 sets (bn)</td><td>Mean cap. ratio</td><td>Mean AI de endency</td></tr><tr><td>nodes vendors.csv</td><td>60</td><td>vendor attributes</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathtt { n o d e s \_ i n s t i t u t i o n s . c s v }$ </td><td>220 220</td><td>bank attributes</td><td>G-SIB</td><td>13</td><td>2,341</td><td>0.132</td><td>0.61</td></tr><tr><td>nodes customers.csv</td><td>2,476</td><td>per-bank customer aggregRaetegsional</td><td></td><td>75</td><td>214</td><td>0.135</td><td>0.49</td></tr><tr><td>edges vendor to bank.csv</td><td>1,400</td><td>service exposures</td><td>Community</td><td>132</td><td>10.2</td><td>0.137</td><td>0.42</td></tr><tr><td>edges interbank.csv incidēnts.csv</td><td>3,200</td><td>bilateral obligations vendor incident telemetry</td><td colspan="5">TABLE IV: Per-vendor-type population characteristics.</td></tr><tr><td>timeseries signals.csv</td><td>158,400</td><td>per-bank per-hour signals</td><td colspan="5"></td></tr></table>

TABLE II: Graph statistics of the synthetic system.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Vendors (||)</td><td>60</td></tr><tr><td>Banks (|B|) Vendor-bank edges (|Evb|)</td><td>220</td></tr><tr><td>Interbank edges (|Eib|)</td><td>2,476 1,400</td></tr><tr><td>Mean bank degree (vendors used)</td><td>11.3</td></tr><tr><td>Median vendor degree (banks served)</td><td>32</td></tr><tr><td>Top-1 vendor degree</td><td>194</td></tr><tr><td>Interbank density</td><td>2.9%</td></tr><tr><td>G-SIB share of assets</td><td></td></tr><tr><td>Mean AI dependency</td><td>71% 0.46</td></tr></table>

## B. What the synthetic data does and does not represent

The purpose of the synthetic data is to give a controllable, reproducible substrate on which the coupled dynamic and the early-warning learner can be studied. The distributional choices above are consistent with widely reported statistics on cloud- and AI-vendor concentration [9], [11], [10], [12], on interbank-exposure heavy tails [4], [18], [19], and on operational-error signatures in production ML systems [23], [44], [27]. They are not calibrated to a specific jurisdiction or to a particular vendor, and no numerical result in this paper should be read as a forecast for a real institution.

## VI. EXPERIMENTS

## A. Illustrative cascade

Fig. 4 shows one representative CFC-Prop trajectory seeded at the top-critical vendor. The vendor-infection curve peaks quickly and then decays as patch cycles recover the population; the bank-impairment curve rises with a lag and reaches a much higher fraction of the institutional population because a single infected vendor typically serves many banks. Defaults appear only when impairment on hub banks crosses the capital-adjusted threshold and are highly stochastic, which is consistent with the intuition in [31], [6].

<table><tr><td>Type</td><td>Count</td><td>Mean criticality</td><td>Mean patch latency (d)</td></tr><tr><td>FoundationModel</td><td>11</td><td>21.4</td><td>15.2</td></tr><tr><td>MLOps</td><td>15</td><td>14.6</td><td>11.4</td></tr><tr><td>DataPipeline</td><td>12</td><td>11.9</td><td>10.1</td></tr><tr><td>InferenceHost</td><td>13</td><td>18.7</td><td>13.6</td></tr><tr><td>FeatureStore</td><td>9</td><td>9.5</td><td>8.9</td></tr></table>

## B. Monte Carlo across vendor tiers

We run 6 realizations per seed vendor across three tiers of vendor criticality (top, mid, low), producing 150 trajectories. Fig. 5 shows the resulting loss distribution: median losses are near zero for low-criticality vendors and long-tailed for topcritical vendors, consistent with the reasoning in [1], [4], [3]. Table V summarizes the outcomes.

A useful way to read Table V is that the ratio of mean loss between the top and mid tiers (≈ 8.3×) is larger than the ratio of vendor criticality between the same tiers (≈ 3.1×). This super-linearity is the empirical fingerprint of the coupled cascade: an incremental unit of vendor criticality produces a disproportionately larger financial loss, because the extra downstream bank connections tend to include more G-SIBs, whose default in turn propagates further. The pattern is the reason a supervisor cannot rely on a linear vendor-criticality score alone.

## C. Patch latency sensitivity

Patch latency is the single lever that vendors and their supervisors can most directly move. We sweep a patch-latency multiplier from 0.25× to 3.0× baseline and re-run 8 seeds per multiplier. Fig. 6 shows that both the mean peak of impaired banks and the 90th-percentile loss are convex in the multiplier, arguing for aggressive patch-cycle service-level objectives at systemically important vendors along the lines of [9], [11].

## D. Early-warning detection benchmark

Table VI and Fig. 7 report AUROC, AUPRC, and Brier score for four baselines and CFC-GNN. CFC-GNN improves discrimination and calibration over the strongest baseline while remaining a simple, auditable model, which matters for supervisory adoption. The improvement over gradient boosting is modest but consistent, echoing the finding in [24], [23] that graph structure gives a small but reliable lift on top of strong tabular learners.

![](images/ac1aa1295e3e06c52ed8b7f0ce64bb16384e4e3df754ed43b13bed2a3def2daf.jpg)  
Fig. 4: An illustrative CFC-Prop cascade from a compromise at the top-critical vendor. Impairment saturates the bank population well after the vendor-infection curve has decayed.

Loss distribution scales with vendor criticality tier  
![](images/07e9da3cbdee6a8d9108598c1d57e7302729c6d98c4a8d5a54c1bd40e0807778.jpg)  
Fig. 5: Loss distribution across vendor criticality tiers. The heavy right tail for top-critical vendors is where supervisory attention should concentrate.

## E. Hyperparameter sensitivity of CFC-GNN

Table VII reports the effect of small perturbations to the two mixing weights and the graph-degree correction. Model quality is stable inside a wide neighborhood of the chosen defaults, which is the kind of robustness a supervisor would want before adopting the score for regulatory purposes.

## F. Ablation

Table VIII isolates the contribution of each block in CFC-GNN. Removing graph features drops AUROC by roughly 0.020, and removing incident telemetry drops it by 0.043; both channels are necessary. This mirrors the pattern in [42], [24], where structural signals and behavioral signals are complementary rather than redundant.

## G. Stress scenario: simultaneous compromise of two hubs

As a stress case, we seed two of the top-five critical vendors simultaneously. The peak impaired-banks count rises from a mean of 128 under a single hub to a mean of 187, and the

TABLE V: Monte Carlo cascade outcomes by vendor criticality tier (n = 150 runs).
<table><tr><td>Tier</td><td>Peak vendors</td><td>Peak banks</td><td>Final defaults</td><td>Final loss (USD bn)</td></tr><tr><td>Top-critical</td><td>43</td><td>128</td><td>8.1</td><td>1,612</td></tr><tr><td>Mid-critical</td><td>22</td><td>71</td><td>1.4</td><td>194</td></tr><tr><td>Low-critical</td><td>6</td><td>12</td><td>0.0</td><td>0</td></tr></table>

![](images/d77db45ad6f03626ddb0232667d933e2856daf72dd651c11e090086cb7cd19d4.jpg)  
Fig. 6: Sensitivity of peak impairment and tail loss to patch-latency multiplier. Convexity is the case for treating patch cycles as a supervisory KPI.

95th-percentile loss roughly doubles. This is the joint-cyber scenario that regulators cite most often [9], [10], [11], [14], and it is arguably the correct planning case for a modern crisismanagement manual.

## H. Scenario catalog

Table IX summarizes six scenarios that would be reasonable candidates for a supervisory stress-test menu. Each row reports the mean and 95th-percentile outcomes across 20 CFC-Prop realizations at the parameters in Appendix B. The relative severity ordering is stable across seeds and matches the intuition in [14], [13], [15].

## I. Operational telemetry as a leading signal

Fig. 8 shows one month of per-hour latency and errorrate telemetry for two illustrative banks. Bank $B _ { \mathrm { l o } }$ has no injected degradation; bank $B _ { \mathrm { h i } }$ has a stealth degradation window inserted into the middle of the month. The window is visually detectable in both series, and the same window is what CFC-GNN turns into a numeric feature (max severity, mean detection lag) through the incident-summary aggregation in Section IV. In practice, this is the operational telemetry that supervisors would receive under DORA [11], and it is precisely the data that lets a supervisory model reach the discrimination levels reported in Table VI.

## J. Reproducibility

All code, synthetic data, figures, and scripts are provided with the supplementary package. The full pipeline runs in about two minutes on a laptop with 8 GB RAM and produces the exact figures and tables in this paper. Random seeds are fixed; every seed used in the paper is listed in the supplementary README. This is the same reproducibility discipline we apply to related fraud, credit, and federatedlearning work [23], [44], [26], [27].

TABLE VI: Early-warning classifier comparison on the vendorcascade task.
<table><tr><td>Model</td><td>AUROC</td><td>AUPRC</td><td>Brier</td></tr><tr><td>Logistic Regression</td><td>0.697</td><td>0.478</td><td>0.135</td></tr><tr><td>Random Forest</td><td>0.805</td><td>0.573</td><td>0.124</td></tr><tr><td>Gradient Boosting</td><td>0.799</td><td>0.577</td><td>0.122</td></tr><tr><td>MLP (2×64)</td><td>0.767</td><td>0.531</td><td>0.127</td></tr><tr><td>CFC-GNN (ours)</td><td>0.817</td><td>0.597</td><td>0.111</td></tr></table>

![](images/d6d384bee90556d571280ba170eed7db0e77cb301bb20ebfd3cb9d093f23f2f7.jpg)  
Fig. 7: Detection quality across baselines and CFC-GNN. The proposed model wins on all three metrics.

## VII. END-TO-END CASE STUDY

We walk one hypothetical scenario through the full pipeline to make the model concrete. The seed vendor is $V _ { 0 0 3 } .$ the highest-criticality node in the synthetic system, with 194 downstream banks and a patch latency in the 90th percentile.

## A. Day 0: the compromise

An adversary compromises V<sub>003</sub> via a poisoned update to its inference container, in the style of the model-integrity attack described in [28], [29]. No bank is aware yet. Vendor telemetry shows a 9% jump in inference-error rate on a subset of routing keys.

## B. Days 1-3: silent bank impairment

CFC-Prop simulates rapid impairment across banks with high service-exposure to V003. Latency and error-rate telemetry at the affected banks begins to drift (Fig. 8). Fraud-screening false-negative rates rise; a small number of laundering typologies of the kind flagged in [42], [24] slip through undetected. No hard financial loss is booked yet, and the supervisor sees only elevated operational-risk telemetry.

TABLE VII: Hyperparameter sensitivity of CFC-GNN. AUROC on held-out set.
<table><tr><td>GBM weight wG</td><td>MLP weight  $w _ { M }$ </td><td>Degree bonus</td><td>AUROC</td></tr><tr><td>0.50</td><td>0.50</td><td>0.02</td><td>0.813</td></tr><tr><td>0.55</td><td>0.45</td><td>0.02</td><td>0.817</td></tr><tr><td>0.60</td><td>0.40</td><td>0.02</td><td>0.815</td></tr><tr><td>0.55</td><td>0.45</td><td>0.00</td><td>0.804</td></tr><tr><td>0.55</td><td>0.45</td><td>0.04</td><td>0.816</td></tr><tr><td>0.70</td><td>0.30</td><td>0.02</td><td>0.808</td></tr></table>

TABLE VIII: Feature ablation for CFC-GNN.
<table><tr><td>Variant</td><td>AUROC</td><td>AUPRC</td><td>Brier</td></tr><tr><td>Full model</td><td>0.817</td><td>0.597</td><td>0.111</td></tr><tr><td>- graph features</td><td>0.797</td><td>0.579</td><td>0.117</td></tr><tr><td>- incident telemetry</td><td>0.774</td><td>0.541</td><td>0.122</td></tr><tr><td>- vendor-type dummies</td><td>0.812</td><td>0.593</td><td>0.113</td></tr><tr><td>Only static attributes</td><td>0.703</td><td>0.484</td><td>0.132</td></tr></table>

## C. Days 4-8: cascade ignition

By day 4, roughly 40% of banks in the system are impaired at level xb ≥ 0.4 (Fig. 4). A subset of G-SIBs cross the impairment threshold that triggers interbank shock propagation. Their counterparties see a jump in inbound capital shocks and, for the most concentrated exposures, cross the default threshold.

## D. Days 9-30: recovery

As patches roll out at V003 and its co-vendors, the vendorinfection curve decays. Impairment plateaus in the surviving banks but does not increase further. Defaults are a small, essentially irreversible tail of the process. The total system loss in this realization is close to the mean loss for the topcritical seed tier in Table V.

## E. What supervisors would have seen in real time

CFC-GNN, trained on prior scenarios, would have flagged V003 as a top-decile risk well before the compromise, based on the combination of high criticality, long patch latency, and heavy downstream degree. The point of the model is not to predict the specific attack but to identify the vendors whose compromise would be systemic. Table X summarizes the case study.

## VIII. THEORETICAL PROPERTIES OF CFC-PROP

Although the paper is primarily empirical, three properties of CFC-Prop are useful for interpretation.

## A. Basic reproduction number

For the vendor layer, treat co-vendor infection as a discretetime SIR process on the vendor-vendor projection graph. Let k be the mean co-vendor degree (the average number of vendors that share at least one bank with a given vendor). Following the classical result of [20], [21], the basic reproduction number is

$$
R _ { 0 } = \frac { \alpha k } { \gamma } \cdot\tag{3}
$$

TABLE IX: Scenario catalog. Losses and impaired-bank counts are means (p95 in parentheses).
<table><tr><td>Scenario</td><td>Impaired banks</td><td>Loss (USD bn)</td></tr><tr><td>Single top-critical vendor</td><td>128 (156)</td><td>1,612 (3,104)</td></tr><tr><td>Single mid-critical vendor</td><td>71 (99)</td><td>194 (612)</td></tr><tr><td>Single low-critical vendor</td><td>12 (28)</td><td>0 (0)</td></tr><tr><td>Joint two top vendors</td><td>187 (211)</td><td>3,021 (5,890)</td></tr><tr><td>Top vendor + slow patch  $( 2 \times )$ </td><td>152 (188)</td><td>2,342 (4,774)</td></tr><tr><td>Top vendor + dense interbank</td><td>141 (177)</td><td>2,113 (4,418)</td></tr></table>

![](images/448b88fc1e18a3e211836a8101e29177614fcd9761312d320ca18d9d1d6b4bf0.jpg)  
Fig. 8: Illustrative per-hour operational telemetry for two banks. The injected stealth-degradation window in the top-most series is what a supervisory early-warning model must recover.

In our synthetic graph, $k ^ { ^ { - } } \approx 4 1 ;$ with $\alpha = 0 . 1 0$ and $\gamma = 0 . 1 8$ this gives $R _ { 0 } \approx 2 2 . 8$ , well above the epidemic threshold. This is why single-seed compromises reliably ignite in Section VI and why patch latency is the operationally meaningful lever.

## B. Expected impairment

Conditional on vendor v being infected at time t, the expected one-step impairment increment of bank b satisfies

$$
\mathsf { E } [ \Delta x _ { b } \mid v \in I ^ { t } , ( v , b ) \in \mathsf { E } _ { \mathrm { v b } } ] = 0 . 2 2 5 \cdot h _ { \phantom { } { \nu } \nu b } ^ { t } ,\tag{4}
$$

where 0.225 is the mean of the U(0.10, 0.35) increment. $\mathbf { A } \mathbf { g }$ gregating across infected vendors and using the union bound gives an upper envelope on impairment that is monotone in vendor criticality, service-exposure, and bank AI-dependency. This is the formal counterpart of the empirical monotonicity in Fig. 5.

## C. Default sufficient condition

Bank j defaults at time t when its aggregate inbound shock exceeds $( 1 - \phi ) \kappa _ { j } A _ { j } .$ A sufficient condition for default of j is that a single counterparty i with $\widetilde { x } _ { i } ^ { t } \ge \widetilde { x } ^ { * }$ satisfies

$$
w i j \eta \widetilde { x } ^ { * } \geq ( 1 - \phi ) \kappa _ { j } A _ { j } .\tag{5}
$$

The condition isolates the interbank concentration that makes a given bank fragile even under mild counterparty impairment; it is the discrete analogue of the “too-central-to-fail” argument in [8], [6].

TABLE X: End-to-end case study, seed vendor V<sub>003</sub>, top-critical tier.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Seed vendor downstream degree</td><td>194</td></tr><tr><td>Day of vendor-infection peak</td><td>5</td></tr><tr><td>Day of bank-impairment peak</td><td>12</td></tr><tr><td>Peak impaired banks (level ≥ 0.3)</td><td>128</td></tr><tr><td>Final defaulted banks Customers affected (mn)</td><td>8</td></tr><tr><td>System loss (USD bn, mean)</td><td>132</td></tr><tr><td>CFC-GNN pre-compromise risk decile</td><td>1,612 10th (highest)</td></tr></table>

## D. Convexity in patch latency

Holding the graph fixed, the expected number of impaired banks in one epidemic cycle scales roughly as $k / \gamma$ , and $\gamma$ is inversely proportional to patch latency. Therefore expected impairment scales linearly in latency, while the fire-sale cascade adds a super-linear tail because default of one hub raises inbound shock at all its neighbors. This yields the convex empirical pattern in Fig. 6 and is the theoretical justification for treating patch latency as a supervisory KPI.

## IX. A SUPERVISORY FRAMEWORK

CFC-Prop and CFC-GNN are analytical tools; on their own they do not deliver policy. Combining them with the exposure disclosures that supervisors are beginning to receive under DORA [11] and the EBA outsourcing guidelines [10] suggests a concrete four-step framework.

## A. Step 1: Vendor concentration mapping

Supervisors receive per-bank vendor-dependency reports, aggregate them into a system-wide bipartite graph, and compute per-vendor degree, weighted service-exposure sum, and R0 under a plausible α. The output is a heatmap of vendor systemic importance, directly analogous to the G-SIB scoring under Basel III [12].

## B. Step 2: CFC-Prop scenario runs

For each vendor above a systemic-importance threshold, supervisors run CFC-Prop with parameters drawn from a jurisdiction-calibrated distribution. Outputs feed into the operational-risk pillar of the capital adequacy assessment and into the crisis-management manual.

## C. Step 3: CFC-GNN early warning

CFC-GNN is applied to real-time vendor telemetry to generate a rolling watch list. The list is shared with the affected banks under strict confidentiality, mirroring the approach used for suspicious-transaction alerts [42], [24].

## D. Step 4: Patch-cycle service-level objectives

Given the convexity in Fig. 6, supervisors negotiate hard service-level objectives on patch cycles with the most systemic vendors. This is analogous to the liquidity-coverage ratio: an operational parameter with a hard floor is more informative than a soft guideline.

![](images/7b28c9c6157cd3fdf3216434ff2d055e629c55a98a57d15023c219d47b7802d6.jpg)  
Fig. 9: Reliability curves. CFC-GNN sits closer to the diagonal, especially in the mid-probability regime that a supervisor would use for a watch-list threshold.

![](images/a5467436c795c1aa15e116b83ccb08c6ff1b8542b56a0af83ae2f4ab877cd4fd.jpg)  
Fig. 10: ROC curves on the vendor-cascade detection task.

## X. ROBUSTNESS AND ADDITIONAL EXPERIMENTS

## A. Calibration of CFC-GNN

A useful early-warning score is calibrated, not merely discriminative. Fig. 9 shows the reliability curves for CFC-GNN and the strongest baseline (gradient boosting). CFC-GNN’s expected calibration error (ECE) is 0.041 versus 0.058 for gradient boosting, and its Brier score is lower (Table VI). The lift is small but consistent with the calibration findings we report in the credit-scoring setting [44], where boostingplus-uncertainty stacks routinely beat single-model baselines by roughly this margin. Fig. 10 shows the ROC curves for the same two models, giving a complementary picture of the discrimination gain.

## B. Adversarial perturbation of vendor telemetry

We stress-test CFC-GNN by injecting Gaussian noise of standard deviation σ ∈ {0.05, 0.10, 0.20} into the incidenttelemetry features at inference time. AUROC degrades from 0.817 to 0.811, 0.798, 0.771 respectively. Gradient boosting degrades from 0.799 to 0.790, 0.771, 0.734, a slightly steeper slope. This is the same qualitative pattern observed in the intrusion-detection setting of [25], [27]: ensemble stacks are marginally more robust to feature noise than their strongest single component.

![](images/0f06a1f74cdcc16b2e19cd848232e92d4a38f600243deaf3442585a0a53d4ae5.jpg)  
Fig. 11: Adding a stylized customer-run channel with per-day withdrawal probability p<sub>r</sub> amplifies the tail loss.

## C. Cross-graph generalization

We regenerate the synthetic system with a different random seed (holding all distributions fixed) and evaluate CFC-GNN trained on graph A on graph B. AUROC drops from 0.817 to 0.791; AUPRC drops from 0.597 to 0.552. The drop is meaningful but modest, arguing that the model learns a signal that generalizes across graph realizations from the same generating process, not merely across permutations of a single graph.

## D. A minimal customer-run channel

We add a stylized customer-run channel: with probability pr per customer per day, a customer of an impaired bank withdraws its deposit. Deposit outflows accumulate and, above a liquidity threshold, force a fire sale of the bank’s assets that adds to inbound shock at counterparties. Fig. 11 plots the mean and 95th-percentile system loss as $p _ { r }$ varies from 0 to 0.02. With $p _ { r } = 0 . 0 0 5$ , the 95th-percentile loss rises by roughly 18% in the top-critical seed scenario. This is the trust channel of [32], [33], [15]; a full treatment is left to future work.

## E. Sensitivity to interbank density

We re-run the top-critical seed scenario with the interbank density lowered by 50% and raised by 50%. Consistent with [6], [7], densely connected systems absorb small shocks better but suffer worse tail losses under large shocks. Under our seed, the 95th-percentile loss falls by 22% in the sparser system and rises by 31% in the denser system.

## F. Comparison against a network-only baseline

A reasonable strawman is a purely structural score: rank vendors by weighted service-exposure to the top-k G-SIBs. This baseline reaches AUROC 0.744 and AUPRC 0.502, well below CFC-GNN. The gap confirms that incident telemetry and vendor-level attributes carry non-trivial predictive power beyond static graph topology.

TABLE XI: Robustness summary: CFC-GNN AUROC under stress conditions.
<table><tr><td>Condition</td><td>AUROC</td><td>∆ vs. base</td></tr><tr><td>Base (in-graph)</td><td>0.817</td><td></td></tr><tr><td>Gaussian noise σ = 0.05</td><td>0.811</td><td>-0.006</td></tr><tr><td>Gaussian noise σ = 0.10</td><td>0.798</td><td>-0.019</td></tr><tr><td>Gaussian noise σ = 0.20</td><td>0.771</td><td>-0.046</td></tr><tr><td>Cross-graph generalization</td><td>0.791</td><td>-0.026</td></tr><tr><td>Interbank density ×0.5</td><td>0.809</td><td>-0.008</td></tr><tr><td>Interbank density ×1.5</td><td>0.815</td><td>-0.002</td></tr></table>

## G. Comparison against an epidemic-only baseline

A second strawman ignores learned components entirely and ranks vendors by the CFC-Prop simulated peak impairedbank count under a compromise of each vendor in isolation. This is a computationally expensive but transparent supervisory rule. It reaches AUROC 0.786 and AUPRC 0.548, better than the pure network baseline but below CFC-GNN. The reason is that the epidemic-only score ignores incidentside telemetry (severity history, detection lag), which is where CFC-GNN’s marginal gain comes from.

## H. Ranking stability under label perturbation

We perturb the cascade labels by flipping 2% of them and refit. The top-10 vendor ranking induced by CFC-GNN changes by at most two positions in more than 90% of trials. This is the kind of ranking stability that matters for a supervisory watch list: an early-warning score that reorders wildly under small label perturbations is difficult to defend to the vendors it targets.

## I. Alternative scenario: fragmented compromise

As a contrast to the single-hub scenario, we also seed compromises at ten randomly chosen low-criticality vendors. The peak impaired-bank count is roughly 27, an order of magnitude below the top-critical single-hub case, and the mean loss is essentially zero. The scenario shows that vendor concentration, not vendor count, drives the cyber-financial tail, echoing the qualitative argument in [9], [11].

## J. Feature importance and interpretability

Gradient boosting exposes a natural feature-importance ranking. In the trained model, the five most important features are, in order, vendor bank-degree, mean incident severity, patch latency, criticality, and detection lag. This ordering is consistent with the theoretical decomposition in Section VIII: bank-degree drives R0; incident severity drives the per-step impairment increment; patch latency drives γ; and criticality and detection lag interact with both. Feature attributions of this shape are what makes the model defensible in a supervisory conversation with the vendor being scored.

## K. Runtime and scalability

One CFC-Prop run over a 30-day horizon takes under one second on a single CPU thread. Training and scoring CFC-GNN over the full synthetic system takes under 10 seconds. The whole experiment section, including all 20-run scenario tables, completes in under two minutes on a laptop. This makes the tool practical for exploratory supervisory analysis, where dozens of scenarios might be run per day.

## XI. DISCUSSION

## A. Concentration is a first-order systemic variable

Fig. 3 shows that a small number of vendors already serve a large majority of the banks in the synthetic system, and this pattern is qualitatively consistent with public statements from supervisors on cloud and AI-vendor concentration [9], [11], [10]. Under CFC-Prop, this concentration is what turns an operational incident into a systemic event. The policy implication is that vendor-level concentration limits, similar in spirit to large-exposure limits in credit, are worth serious study.

## B. Patch latency deserves KPI status

Fig. 6 shows a convex relationship between patch latency and tail loss. This is the standard shape that motivates hard supervisory limits rather than soft guidelines and is consistent with the argument in [34], [14], [17].

## C. Model integrity is a balance-sheet concern

The three attack modes in Section III correspond to the OWASP LLM top-10 list [41] and the classical adversarialmachine-learning taxonomy of [30], [29], [28], [36], [37]. The novel point is that under high AI-dependency, model-integrity failures no longer stay inside the model-risk-management team; they show up in operational-risk capital and, at the tail, in credit losses. The regulatory infrastructure for treating model integrity as a balance-sheet concern is still nascent [35], [12].

## D. Runs and trust

The trust channel emphasized in [32], [33], [15] interacts with the cyber channel: a widely publicized model-integrity event at a foundation-model vendor could plausibly precipitate a run at any bank publicly known to depend on that vendor. Our model captures this only indirectly, through customercount aggregates, and we flag it as an important extension.

## E. From alerts to capital

A piece missing from most cyber-financial narratives is how a supervisor should turn a CFC-GNN score into a capital addon. One reasonable operational bridge is to treat the score as a multiplier on the operational-risk component of Pillar 1 capital: a vendor in the top decile of CFC-GNN, and above a hard degree threshold, triggers a proportional buffer at every bank whose weighted service-exposure to the vendor exceeds a set fraction of Tier 1 capital. This mirrors the concentration surcharges used in the credit-risk pillar [12]. It also lines up with the model-risk arguments in [23], [44], where a wellcalibrated risk score is more useful for capital purposes than a strictly higher-AUROC but poorly calibrated one.

## F. Interaction with anti-money-laundering pipelines

Many of the affected AI services in the scenario of Section VII are AML pipelines: transaction screening, alert ranking, and entity resolution [42], [24]. A quiet degradation of alert-ranking quality does not show up as a loss immediately, but it produces two second-order effects. First, laundering typologies pass undetected for longer, which is a financial-crimecompliance liability under Bank Secrecy Act enforcement. Second, downstream fraud losses accumulate more slowly, so the size of the eventual balance-sheet impact is a lagged function of the impairment period. Our model captures the first-order impairment; the compliance and reputational tail is left to future work.

## G. Interaction with intrusion-detection stacks

Bank-side intrusion detection [25], [50], [51] is the operational counterpart to the supervisory early-warning learner. In the scenario walk-through, the intrusion-detection stack would ideally have caught the poisoned-update signal before it reached vendor customers. In practice, poisoned updates that pass the vendor’s own release checks are difficult for downstream defenders to spot [28], [36], which is why the supervisory layer matters.

## H. Limitations

The synthetic dataset is designed to be qualitatively realistic, not calibrated to a specific jurisdiction. The clearing dynamic follows Furfine-style logic [22], [2] rather than the fixedpoint pricing of [4]. The GNN component is deliberately kept simple; deeper heterogeneous-graph architectures [48], [49], [47] would very likely improve discrimination. Adversarial robustness of the early-warning learner itself is not assessed here; we treat it in a related manuscript on adversarial stress tests for financial machine learning [27]. The customer-run channel is modelled at the aggregate level; a heterogeneouscustomer treatment along the lines of [32] would give a finer picture of the trust-mediated tail. Finally, the exposure disclosures needed to calibrate the model at scale are still being built out under DORA and the EBA outsourcing guidelines [11], [10]; supervisory adoption depends on the pace of that disclosure effort.

## I. Threats to validity

The main threats to the validity of the empirical results are three. First, the vendor-vendor projection graph is a simplifying assumption: real co-vendor lateral movement depends on shared code paths and identity systems, not just on shared customers. Second, we treat impairment as a scalar $x _ { b } \in [ 0 , 1 ] ;$ in practice it is a vector across business lines, and different business lines have different capital treatments. Third, the labels used to train CFC-GNN come from the same simulator used to generate the shocks, which biases the evaluation upward. Cross-graph generalization results in Section X partially address this concern, but real-world validation, once vendorexposure disclosures allow it, is the ultimate test.

## J. Cost of adoption

Deploying CFC-Prop and CFC-GNN at a supervisor is a modest engineering exercise. The pipeline runs on commodity hardware, does not require GPUs, and produces auditable intermediate artifacts. The heavy lift is in exposure disclosure: a supervisor needs per-bank vendor-dependency reports at reasonable granularity, and vendors need a common ontology for service categories. Both are underway under DORA and equivalent regimes [11], [10], and both are areas where the finance industry can meaningfully accelerate the timeline by adopting shared taxonomies.

## K. Model risk of the early-warning learner

Any model used in a supervisory capacity itself becomes a model-risk problem. CFC-GNN’s ensemble structure is deliberately simple, deterministic given fixed seeds, and expressible in a few pages of code, which is roughly the auditability standard the model-risk-management practice recommends for supervisory tooling [44]. Extensions to deeper GNN backbones [48], [49], [47], [45], [46] would improve discrimination but should be weighed against the additional model-risk overhead.

## L. Future work

We see four directions. First, calibration to real vendorexposure disclosures under DORA and comparable regimes [11], [10]. Second, a finer-grained customer-run channel that separates retail from wholesale funding [32], [33], [15]. Third, an adversarial-training version of CFC-GNN with certified robustness guarantees along the lines of [27]. Fourth, integration with secure multi-party analytics so that supervisors can pool vendor-exposure data across jurisdictions without exposing bank-level trade secrets, which is the technical direction we develop in [26].

## XII. CONCLUSION

This paper studies cyber-financial contagion through a specific and increasingly important channel: the shared AI vendor. We build a four-layer heterogeneous graph, propose a coupled epidemic-and-clearing dynamic (CFC-Prop), and train an early-warning learner (CFC-GNN) that flags vendors whose compromise would produce the largest downstream cascade. Experiments on synthetic data show heavy-tailed loss distributions, strong sensitivity to patch latency, and a modest but consistent improvement of CFC-GNN over strong tabular baselines. The full pipeline is reproducible and released with the paper. The findings support treating AI-vendor concentration, patch cycles, and model integrity as first-order supervisory variables. Future work will incorporate a customer-run channel, richer heterogeneous-graph architectures, adversarial robustness of the early-warning model, and calibration to the vendor-exposure disclosures that are beginning to become available under the DORA regime [11], [10].

• The pipeline is: python code/generate\_data.py python

APPENDIX A NOTATION SUMMARY
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $V , B$ </td><td>vendor set, bank set</td></tr><tr><td> $\dot { E _ { \mathrm { v b } } } , E _ { \mathrm { i b } }$ </td><td>vendor-bank, interbank edge sets</td></tr><tr><td> $c _ { \nu } , m _ { \nu } , \ell _ { \nu }$ </td><td>vendor criticality, market share, patch latency</td></tr><tr><td> $A _ { b } , \kappa _ { b } , d _ { b }$ </td><td>bank assets, capital ratio, AI-dependency</td></tr><tr><td> $\mathsf { \Pi } _ { { S } ^ { t } , \Pi ^ { t } , \Pi ^ { t } } ^ { s _ { p b } , \varpi w i j }$  v ν ν</td><td>vençor-bąnk service exposure, interbank exposure vendor SIR states</td></tr><tr><td> $x _ { \mathit { k } } ^ { t } , y _ { \mathit { b } } ^ { t }$ </td><td>bank impairment level, default flag</td></tr><tr><td> $\alpha , \beta , \gamma$ </td><td>vendor infection, bank hazard, recovery rates</td></tr><tr><td> $\eta , \phi , \tau$ </td><td>haircut, fire-sale amp., impairment threshold</td></tr><tr><td> $R _ { 0 }$ </td><td>basic reproduction number on the vendor layer</td></tr></table>

APPENDIX B  
DEFAULT PARAMETER VALUES
<table><tr><td>Symbol</td><td>Description</td><td>Default</td></tr><tr><td>α</td><td>vendor-vendor lateral movement rate</td><td>0.10</td></tr><tr><td>β</td><td>bank-hazard multiplier</td><td>6.0</td></tr><tr><td>γ</td><td>vendor recovery rate per step</td><td>0.18</td></tr><tr><td>η</td><td>interbank capital haircut</td><td>0.55</td></tr><tr><td>φ</td><td>fire-sale amplification</td><td>0.06</td></tr><tr><td>τ</td><td>impairment threshold for shock propagation</td><td>0.40</td></tr><tr><td>T</td><td>horizon (days)</td><td>25-30</td></tr><tr><td>Seeds</td><td>fixed per figure/table in supplementary README</td><td></td></tr></table>

## APPENDIX C

## DERIVATION OF $R _ { 0 }$

The co-vendor projection graph $G _ { \mathrm { p r o j } }$ has an edge (u, v) if vendors u and v share at least one bank. In discrete-time SIR on $G _ { \mathrm { p r o j } } ,$ a newly infected vendor has, in expectation, k susceptible neighbors and infects each with probability α. Its expected infectious lifespan is $1 / \gamma$ steps. Multiplying yields $R 0 = \alpha k / \gamma \ [ 2 1 ]$ . For $k \ = 4 1 , \alpha = 0 . 1 0 , \gamma = 0 . 1 8 , R 0 \approx 2 2 . 8 ,$ comfortably above unity. In practice, saturation on $G _ { \mathrm { p r o j } }$ means the effective reproduction number decays quickly, which is why the vendor-infection curve in Fig. 4 peaks and then decays.

## APPENDIX D

## REPRODUCIBILITY CHECKLIST

• All code is Python 3.11 and depends only on numpy, pandas, scikit-learn, and matplotlib.

• Random seeds are fixed at the top of generate\_data.py (2026-09-08) and per-experiment inside experiments.py.

code/experiments.py → python code/make\_figures.py.

• Every figure and table in this paper is regenerated by that pipeline; run time is under two minutes on a laptop.

• Data files, code files, and this manuscript are shipped together as the supplementary bundle.

## DATA AND CODE AVAILABILITY

The synthetic data, complete Python pipeline, and reproducible scripts that generate every figure and table in this paper are available in the supplementary package attached to this submission.

## REFERENCES

[1] F. Allen and D. Gale, “Financial contagion,” Journal of Political Economy, vol. 108, no. 1, pp. 1–33, 2000.

[2] L. Eisenberg and T. Noe, “Systemic risk in financial systems,” Management Science, vol. 47, no. 2, pp. 236–249, 2001.

[3] P. Gai and S. Kapadia, “Contagion in financial networks,” Proc. Royal Society A, vol. 466, no. 2120, pp. 2401–2423, 2010.

[4] R. Cont, A. Moussa, and E. Santos, “Network structure and systemic risk in banking systems,” Handbook on Systemic Risk, pp. 327–368, 2013.

[5] A. Shleifer and R. Vishny, “Fire sales in finance and macroeconomics,” Journal of Economic Perspectives, vol. 25, no. 1, pp. 29–48, 2011.

[6] D. Acemoglu, A. Ozdaglar, and A. Tahbaz-Salehi, “Systemic risk and stability in financial networks,” American Economic Review, vol. 105, no. 2, pp. 564–608, 2015.

[7] M. Elliott, B. Golub, and M. O. Jackson, “Financial networks and contagion,” American Economic Review, vol. 104, no. 10, pp. 3115– 3153, 2014.

[8] S. Battiston, M. Puliga, R. Kaushik, P. Tasca, and G. Caldarelli, “DebtRank: Too central to fail? financial networks, the FED and systemic risk,” Scientific Reports, vol. 2, p. 541, 2012.

[9] Financial Stability Board, “Third-party dependencies in cloud services: Considerations on financial stability implications,” FSB Report, 2019.

[10] European Banking Authority, “EBA guidelines on outsourcing arrangements,” EBA/GL/2019/02, 2019.

[11] European Supervisory Authorities, “Digital operational resilience act (DORA) overview,” ESA Joint Committee, 2022.

[12] Basel Committee on Banking Supervision, “Principles for the sound management of operational risk,” Bank for International Settlements, 2011.

[13] M. Crosignani, M. Macchiavelli, and A. F. Silva, “Pirates without borders: The propagation of cyberattacks through firms’ supply chains,” Journal of Financial Economics, vol. 147, no. 2, pp. 432–448, 2023.

[14] T. M. Eisenbach, A. Kovner, and M. J. Lee, “Cyber risk and the U.S. financial system: A pre-mortem analysis,” Journal of Financial Economics, vol. 145, no. 3, pp. 802–826, 2022.

[15] D. Duffie and J. Younger, “Cyber runs,” Hutchins Center Working Paper, 2019.

[16] I. Aldasoro, J. Frost, L. Gambacorta, and D. Whyte, “Covid-19 and cyber risk in the financial sector,” BIS Bulletin, no. 37, 2021.

[17] A. K. Kashyap and A. Wetherilt, “Some principles for regulating cyber risk,” AEA Papers and Proceedings, vol. 109, pp. 482–487, 2019.

[18] R. Iyer and J.-L. Peydro, “Interbank contagion at work: Evidence from a natural experiment,” Review of Financial Studies, vol. 24, no. 4, pp. 1337–1377, 2011.

[19] C. Upper, “Simulation methods to assess the danger of contagion in interbank markets,” Journal of Financial Stability, vol. 7, no. 3, pp. 111– 125, 2011.

[20] W. O. Kermack and A. G. McKendrick, “A contribution to the mathematical theory of epidemics,” Proc. Royal Society A, vol. 115, no. 772, pp. 700–721, 1927.

[21] R. Pastor-Satorras, C. Castellano, P. Van Mieghem, and A. Vespignani, “Epidemic processes in complex networks,” Reviews ofModern Physics, vol. 87, no. 3, pp. 925–979, 2015.

[22] C. Furfine, “Interbank exposures: Quantifying the risk of contagion,” Journal of Money, Credit and Banking, vol. 35, no. 1, pp. 111–128, 2003.

[23] S. Nayak and A. R. Bushara, “Uncertainty-aware fraud detection using hybrid transformer with gated token mixing and conformal risk control,” IEEE Access, vol. 14, pp. 77557–77573, 2026.

[24] S. Nayak, “FraudGNN: Self-supervised graph neural anomaly detection for real-time financial fraud with adversarial robustness and explainable reasoning,” in Proc. IEEE 5th Int. Conf. AI in Cybersecurity (ICAIC), (Houston, TX, USA), pp. 1–7, 2026.

[25] S. Nayak, “ThreatFormer-IDS: Robust transformer intrusion detection with zero-day generalization and explainable attribution,” in Proc. 4th Cognitive Models and Artificial Intelligence Conf. (AICCONF), (Prague, Czech Republic), pp. 1–8, 2026.

[26] S. Nayak, “TrustFed-He: Trustworthy federated fraud analytics for banks with homomorphic secure aggregation and poisoning-resilient training,” in Proc. 14th Int. Symp. Digital Forensics and Security (ISDFS), (Boston, MA, USA), pp. 1–7, 2026.

[27] S. Nayak, “BHF-Guard: Breaking and hardening financial ML with adversarial stress tests and certified robustness checks,” in Proc. 14th Int. Symp. Digital Forensics and Security (ISDFS), (Boston, MA, USA), pp. 1–7, 2026.

[28] T. Gu, B. Dolan-Gavitt, and S. Garg, “BadNets: Identifying vulnerabilities in the machine learning model supply chain,” in NIPS Workshop on Machine Learning and Computer Security, 2017.

[29] B. Biggio and F. Roli, “Wild patterns: Ten years after the rise of adversarial machine learning,” Pattern Recognition, vol. 84, pp. 317– 331, 2018.

[30] N. Papernot, P. McDaniel, A. Sinha, and M. Wellman, “SoK: Security and privacy in machine learning,” Proc. IEEE EuroS&P, 2018.

[31] P. Glasserman and H. P. Young, “How likely is contagion in financial networks?,” Journal of Banking & Finance, vol. 50, pp. 383–399, 2015.

[32] D. W. Diamond and P. H. Dybvig, “Bank runs, deposit insurance, and liquidity,” Journal of Political Economy, vol. 91, no. 3, pp. 401–419, 1983.

[33] M. K. Brunnermeier, “Deciphering the liquidity and credit crunch 2007– 2008,” Journal of Economic Perspectives, vol. 23, no. 1, pp. 77–100, 2009.

[34] A. W. Lo, “Moore’s law vs. murphy’s law in the financial system: Who’s winning?,” Journal ofInvestment Management, vol. 14, no. 1, pp. 17–29, 2016.

[35] National Institute of Standards and Technology, “Framework for improving critical infrastructure cybersecurity, version 1.1,” NIST, 2018.

[36] N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson, et al., “Extracting training data from large language models,” USENIX Security Symposium, 2021.

[37] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership inference attacks against machine learning models,” IEEE Symp. Security and Privacy, 2017.

[38] M. Ohm, A. Sykosch, and M. Meier, “Backstabber’s knife collection: A review of open source software supply chain attacks,” Proc. Int. Conf. Detection of Intrusions and Malware, and Vulnerability Assessment, 2020.

[39] A. Cencini, K. Yu, and T. Chan, “Software vulnerabilities: Full-, responsible-, and non-disclosure,” Univ. of Washington Report, 2005.

[40] C. Herley and P. van Oorschot, “SoK: Science, security, and the elusive goal of security as a scientific pursuit,” IEEE Symp. Security and Privacy, 2017.

[41] OWASP Foundation, “OWASP Top 10 for large language model applications,” OWASP Project, 2023.

[42] S. Nayak and R. Kumar, “Quantum-enhanced AML: Hybrid quantum– classical graph learning with entity resolution and evidence subgraph discovery for transaction screening,” IEEE Access, vol. 14, pp. 74837– 74850, 2026.

[43] S. Nayak, “SecurePayChain-AI: Smart-contract-verified secure payments with DID-assisted hybrid fraud mining,” in Proc. 5th Asia Conf. Algorithms, Computing and Machine Learning (CACML), (Guangzhou, China), pp. 1–8, 2026.

[44] S. Nayak, “Calibrated credit intelligence: Shift-robust and fair risk scoring with Bayesian uncertainty and gradient boosting,” in Proc. IEEE Int. Research Conf. Smart Computing and Systems Engineering (SCSE), (Kelaniya, Sri Lanka), pp. 1–6, 2026.

[45] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learning Representations (ICLR), 2017.

[46] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2017.

[47] P. Velicˇkovic´, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, “Graph attention networks,” in Proc. Int. Conf. Learning Representations (ICLR), 2018.

[48] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling relational data with graph convolutional networks,” European Semantic Web Conference, 2018.

[49] C. Zhang, D. Song, C. Huang, A. Swami, and N. V. Chawla, “Heterogeneous graph neural network,” in Proc. KDD, 2019.

[50] I. H. Sarker, A. S. M. Kayes, S. Badsha, H. Alqahtani, P. Watters, and A. Ng, “Cybersecurity data science: An overview from machine learning perspective,” Journal of Big Data, vol. 7, no. 1, p. 41, 2020.

[51] A. L. Buczak and E. Guven, “A survey of data mining and machine learning methods for cyber security intrusion detection,” IEEE Communications Surveys & Tutorials, vol. 18, no. 2, pp. 1153–1176, 2016.