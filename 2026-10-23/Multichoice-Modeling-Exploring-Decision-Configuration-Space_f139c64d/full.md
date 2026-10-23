# Multichoice Modeling: Exploring Decision Configuration Space under Uncertainty

Mariia Kozlova<sup>1,\*</sup>, Roman Stepanov<sup>1</sup>, Alexander Walzer<sup>2</sup>, Andrea Saltelli<sup>3</sup>

<sup>1</sup>LUT University, LUT Business School, Lappeenranta, Finland

<sup>2</sup>University of Applied Sciences and Arts Northwestern Switzerland FHNW, Switzerland

<sup>3</sup>UPF Barcelona School of Management, Barcelona, Spain

Corresponding author: mariia.kozlova@lut.fi

## Abstract

Decision support has become sophisticated at exploring uncertainty, but much less so at representing the architecture of the decision itself. As a result, the decision space is often narrowed before uncertainty is analyzed, leaving interactions among choices and the efects of alternatives they imply largely invisible. This paper introduces multichoice modeling, a framework for making large decision configuration spaces explicit and computationally explorable under uncertainty. Drawing on the logic of rugged landscapes, we conceptualize such problems as contingent performance landscapes, where interdependent choices generate configuration-specific outcomes whose value changes across uncertain futures. The computational framework combines multidimensional decision and uncertainty spaces in a single simulation. Further, variance-based sensitivity analysis is used for identifying influential choices, and conditional distribution analysis for comparing configuration-specific performance profiles. We demonstrate the framework through three case studies: a manufacturing process upgrade, M&A strategy, and mortgage design. By contrasting conventional uncertainty-only analysis with multichoice modeling, the cases show how opening the decision configuration space can reveal unexpected complementarities, near-best configurations with diferent risk profiles, and strategies that were not apparent as alternatives at the outset.

Keywords: Decision support; uncertainty; strategy table; Monte Carlo simulation; global sensitivity analysis

“We shape our tools and thereafter our tools shape us”

Various authors, attributed to Marshall McLuhan (Kuskis 2013)

## 1 Introduction

In investment appraisal, cost-benefit analysis, operations, finance, and risk management, actions must be evaluated under uncertain future conditions. Therefore, the treatment of uncertainty has become a central aspect of decision support (Saltelli et al. 2020, Scoones 2024). Yet a form of residual determinism remains in the specification of the decision architecture itself. Exploratory modeling often examines how a single system or policy behaves under uncertainty (Bankes 1993, Kwakkel et al. 2013). Robust decision-making evaluates a limited set of candidate strategies across many plausible futures (Lempert et al. 2006, Marchau et al. 2019). Real options analysis frequently focuses on valuing or optimizing either a single or a limited set of generic flexibilities (Dixit and Pindyck 1994, Trigeorgis 1996, Kozlova 2017). Consequently, decision support often remains confined to narrowly defined decision situations, in which only a small subset of the available strategic degrees of freedom is examined.

There is an emerging call for opening up the decision space itself (H¨am¨al¨ainen et al. 2024, Stepanov et al. 2024, Colorni and Tsouki\`as 2020). Unlike the decision analysis field, strategy research has long recognized that a strategy is not a single decision, but a configuration of multiple interdependent choices (Levinthal 1997, Rivkin 2000). Drawing on the concept of rugged landscapes, this literature shows that when choices interact, the performance contribution of any choice depends on the broader configuration in which it is embedded. Yet the rugged-landscape perspective is primarily used as a behavioral and evolutionary lens in explaining how firms search and adapt, whereas strategy evaluation raises a qualitatively diferent decision-support problem.

Consider an example of a battery supply strategy for an electric vehicle (EV) manufacturer. The firm faces multiple interdependent choices, including whether to make or buy batteries, whether to own upstream mining assets, which chemistries to support, which suppliers and regions to rely on, which transportation routes to choose, how much inventory bufer to arrange, and whether to use long-term contracts or spot purchasing. These choices are interdependent and interact with each other: battery chemistry determines suppliers and production assets; vertical integration alters the value of diversification and contracting decisions; transportation, inventory, and sourcing choices jointly determine supply-chain resilience and cost. External uncertainties, such as EV demand, raw-material prices, regulation, geopolitical disruptions, and transport bottlenecks, further complicate the decision landscape by altering which configurations prove robust, fragile, or unexpectedly valuable. There could be thousands of possible combinations of choices that form distinct strategies, most of which are never evaluated or even formulated. The relevant question is therefore no longer how a given strategy performs under uncertainty, but which combinations of choices give rise to attractive strategies across multiple uncertain futures.

We conceptualize strategic decision-making under uncertainty as navigation of an uncertaintycontingent rugged landscape: a recombinatory decision space in which the value of each choice depends on both the surrounding configuration of choices and uncertain external conditions. Choice interdependencies are not abstract assumptions, but arise directly from the structure of the problem itself: physical constraints, operational dependencies, and economic mechanisms.

We further propose multichoice modeling as a methodological framework for making such decision-uncertainty spaces explicit and computationally explorable. The approach represents a decision problem as a set of admissible choice configurations. Choices may be independent or conditional, mutually exclusive or multiple-selection, and may include option-like alternatives. The choice configurations are simulated jointly with uncertainty realizations, producing a dataset that links choices, uncertainties, and performance outcomes. Eficient variance-based sensitivity analysis and conditional projections are then used to identify influential choices and compare the performance profiles induced by their combinations. Rather than framing the problem as the fixed set of alternatives that existing methods can handle (Quade 1980), strategies can emerge as combinations of interdependent choices under uncertainty.

The remainder of the paper is organized as follows. Section 2 positions multichoice modeling within the related literature on decision-making under uncertainty. Section 3 introduces the mathematical formulation and computational implementation of multichoice modeling. Section 4 illustrates the framework through three case studies: a manufacturing process upgrade conducted i a project with Hyperion Robotics, M&A strategy consulted by M&A experts, and a household finance decision. Section 5 proceeds with discussion, and Section 6 concludes.

## 2 Related literature

Our work contributes to the literature on decision-making under uncertainty by combining explicit construction of the decision space with exploratory analysis. We therefore position it by examining how existing approaches represent the decision and uncertainty space and whether they optimize or explore it.

The fundamentals of decision analysis distinguish decision alternatives (controllable choices) from uncertainty (uncontrollable circumstances) (Howard 1988). This conceptual separation has often been operationalized through a narrow decision space: in cost-benefit analysis, investment analysis, risk assessment, and multi-criteria decision-making approaches, analysts first define a limited set of alternatives and then examine how uncertainty afects their consequences (Boardman et al. 2018, Morgan et al. 1990, Clemen and Reilly 1999, Keeney and Raifa 1993).

When the decision space itself is made explicit, it is commonly represented as a feasible set over which an objective is optimized. This is the case in stochastic programming, where decisions are optimized over scenarios or probability distributions (Kleywegt et al. 2002, Shapiro et al. 2009), and in simulation optimization, where stochastic simulation is embedded within a search for high-performing or optimal decisions (Fu 2015). Optimization makes large decision spaces tractable, yet it treats the surrounding configuration space mainly as a feasible region to be searched rather than as an object of interpretation.

Strategy literature has conceptualized configurational complexity through the notion of rugged landscapes (Levinthal 1997, Rivkin 2000). In NK-type models, the complexity of the landscape is characterized by the number of decision elements N and the extent of their interdependence K, such that the performance contribution of each choice depends on the configuration of other choices. As interdependence increases, the landscape becomes increasingly rugged. This is precisely what makes adaptive search and optimization more dificult (Ganco and Hoetker 2009). The appropriate functional form for representing interdependence remains debated (Rahmandad 2019). Subsequent work has extended rugged landscapes to evolving ecosystems and dynamic interdependencies (Ganco et al. 2020). However, these approaches rarely show how external uncertainty conditions the performance landscape itself. The primary focus remains on how actors search rugged landscapes, rather than on decision-makers’ ability to interpret a recombinatory decision space in which performance changes across uncertain futures.

Several related streams of literature shift the emphasis from optimization to exploration. Exploratory modeling and analysis (EMA) uses simulation to examine system behavior across many plausible assumptions and uncertainties (Bankes 1993, Kwakkel et al. 2013). Robust decision-making (RDM) builds on this logic by evaluating candidate strategies across multiple possible futures and identifying strategies that perform satisfactorily (Lempert et al. 2006, Marchau et al. 2019). Scenario discovery provides a key computational technique in this tradition: it works backward from outcomes of interest to interpretable regions of the simulation input space associated with those outcomes (Lempert et al. 2008, Bryant and Lempert 2010, ter Horst et al. 2026). Global sensitivity analysis (GSA) is similarly based on exploring simulation experiments, but its main purpose is to quantify the contribution of input variables and model structures to output variation (Saltelli et al. 2008, Tarantola et al. 2024). Some GSA applications extend this logic beyond parametric uncertainty by treating structural modeling choices, competing scenarios, or alternative policy-relevant representations as sampled inputs (Saltelli et al. 2000). Despite their exploratory nature, these approaches usually treat the decision side as a single architecture, a predefined set of candidate policies, or a limited number of alternatives. Therefore, they capture the broader space of possible decision configurations only partially.

Real options theory ofers a powerful way of thinking about decision design under uncertainty: flexibility embedded in the initial design can create value by enabling delayed responses to unfolding uncertainty (Dixit and Pindyck 1994, Trigeorgis 1996). Its relevance to strategic management has been articulated through the role of flexibility, staged commitment, and strategic investment timing under uncertainty (Trigeorgis and Reuer 2017). In practice, however, real options models often focus on valuing or optimizing a single flexibility mechanism specified prior to the numerical analysis, such as investment timing or capacity choice (Alam et al. 2024), while the broader space of interdependent strategic choices remains at best partially explored due to methodological constraints.

Portfolio decision analysis (PDA) moves closer to a formal treatment of large decision spaces under uncertainty by representing alternatives as portfolios of actions subject to resource and feasibility constraints (Salo et al. 2011, Liesi¨o et al. 2021). Recent work also allows selected actions to afect the probabilities of uncertainty scenarios (Vilkkumaa et al. 2025). The purpose, however, remains portfolio recommendation rather than exploration of the configuration space. Moreover, the additive scenario-wise utility representation supports tractable dominance analysis but is less suited to rugged landscapes.

Multichoice modeling contributes to decision-making under uncertainty by combining three dimensions that are rarely brought together: (i) a large decision space represented through multiple actionable choices; (ii) a rich uncertainty space that both influences and is influenced by choices, jointly shaping the performance of choice configurations; and (iii) an exploratory orientation aimed at discovering performance profiles across choice configurations.

## 3 Multichoice modeling

A multichoice model represents a decision problem under uncertainty by forming a configuration space from interdependent choices, overlaying it with uncertainty realizations, and mapping this decision-uncertainty input space onto performance metrics. The framework

is exploratory: rather than identifying an optimal choice configuration, it characterizes the behavior of performance metrics across the space of feasible decision designs. The following subsections formalize the framework and describe its implementation.

## 3.1 Mathematical formulation

A multichoice model consists of a set of choices ${ \mathcal { C } } = \{ 1 , \ldots , C \}$ , each with alternatives $A _ { c } ,$ a dependency structure ${ \mathcal { G } } .$ an uncertainty space $x ,$ a performance-measure space $\mathcal { V } ,$ and a model function $f .$ . Formally,

$$
{ \mathcal { M } } = ( { \mathcal { C } } , { \mathcal { A } } , { \mathcal { G } } , { \mathcal { X } } , { \mathcal { Y } } , f ) ,\tag{1}
$$

where $\mathcal { G }$ governs admissible combinations of alternatives and f maps selected alternatives and uncertainty realizations onto performance outcomes.

A configuration of choices is represented by a binary vector $z = ( z _ { a } ) _ { a \in \mathcal { A } }$ , with $z _ { a } \in \{ 0 , 1 \}$ , where $z _ { a } = 1$ indicates that alternative a is selected. The set of all such vectors forms the unconstrained configuration space $\bar { \cal S } = \{ 0 , 1 \} ^ { | { \cal A } | }$

Admissibility is governed by a dependency structure $\mathcal { G }$ , defined as a directed hypergraph $\mathcal { G } = ( \mathcal { A } , \mathcal { E } )$ , where each hyperedge $e = ( U , V , \phi _ { e } )$ links a set of antecedent alternatives $U \subseteq A$ to a set of afected alternatives $V \subseteq A$ through a set-valued logical mapping $\phi _ { e } : \{ 0 , 1 \} ^ { | U | } $ $2 ^ { \{ 0 , 1 \} ^ { | V | } }$ . The mapping specifies which selections in $V$ are admissible for each selection pattern in $U$ . The dependency structure $\mathcal { G }$ includes both constraints across choices and selection logic within each $A _ { c } ,$ such as exactly-one or multiple-selection rules. A configuration $z \in \bar { S }$ is admissible if, for every hyperedge $e = ( U , V , \phi _ { e } ) , z _ { V } \in \phi _ { e } ( z _ { U } )$ . The admissible configuration space is therefore

$$
{ \mathcal { S } } = \Big \{ z \in { \bar { \mathcal { S } } } : z _ { V } \in \phi _ { e } ( z _ { U } ) \ { \mathrm { f o r ~ a l l } } \ e \in { \mathcal { E } } \Big \} .\tag{2}
$$

Uncertainty is represented by a random element $X \in { \mathcal { X } }$ . The uncertainty space may include scalar variables $X ^ { ( s ) }$ and time-indexed stochastic processes $X ^ { ( t ) } = \{ X _ { t } \} _ { t = 1 } ^ { T }$ . When uncertainty exposure depends on the choice configuration, a family of conditional distributions is built

$$
X \sim P ( \cdot | z ) .\tag{3}
$$

Given a configuration $z \in S$ and a realization $x \in \mathcal { X }$ , performance outcomes $Y \in \mathcal { V } \subseteq \mathbb { R } ^ { M }$ are generated by

$$
Y = f ( z , x ) .\tag{4}
$$

Dynamic and conditional decisions can be handled either as ex ante choices included in $z \mathrm { \Omega o r }$ as contingent operational rules embedded in $f .$

The simulation experiment and exploratory analysis are presented next as part of the

computational framework.

## 3.2 Implementation workflow

The implementation workflow for multichoice modeling consists of three main steps: constructing the logical model of available choices, translating it into a computational model, and exploring the resulting simulation outcomes (Figure 1). The following subsections describe each step.

![](images/7baa8d8c136f91ceb8d49b9f337db4e7efe1b7fdadbf830b14c591ceeb44587a.jpg)  
Figure 1: Implementation workflow for multichoice modeling.

## 3.2.1 Logical model: choice elicitation.

The strategy table (Howard 1988, Gregory et al. 2012) provides the elicitation layer for multichoice modeling. Its role is to specify the decision space and the links among choices, model parameters, and uncertainty. For multichoice modeling, the table records:

• Decision areas; thematic groups of choices relevant to the decision situation.

• Choices and alternatives; the actions available within each decision area.

• Choice logic; specifying whether alternatives are mutually exclusive, allow multiple selections, introduce flexibility, or involve allocation and sizing decisions.

• Dependencies among choices; including enabling conditions, incompatibilities, sequencing requirements, and other logical or operational constraints.

• Efects on model parameters; such as capital expenditure, operating cost, capacity, or eficiency. When several choices afect the same parameter, their joint efect must be specified.

• Efects on uncertainty exposure; indicating whether a choice introduces, removes, or modifies uncertain factors.

## 3.2.2 Computational model and simulation.

The strategy table is translated into the formal representation M by specifying C, A<sub>c</sub>, G, X , Y, and f, so that the computational model reflects the elicited choice logic, parameter efects, uncertainty exposure, and dependencies.

Sampling and distribution types. A direct way to examine the configuration space is to enumerate all admissible combinations of choices. This becomes impractical as the number of choices grows. Instead, the design distribution q is induced by randomly sampling alternatives for each choice uniformly, subject to the admissibility constraints encoded in G. This sampling scheme is not intended to represent managerial beliefs; it rather serves as a designof-experiments mechanism that provides broad coverage of the admissible configuration space without enumerating the full factorial set of possible configurations. Uncertainty sources are sampled from the distributions specified in the model, with uniform distributions used as neutral exploratory specifications when no probabilistic information is available. Both decision configurations and uncertainty realizations vary in the generated experiment, while retaining diferent interpretations: q is a design distribution over configurations, whereas P(· | z) represents uncertainty conditional on the selected configuration.

Sampling strategy. Variance-based sensitivity analysis is often implemented with a quasirandom sampling strategy to improve the precision of estimated indices (Saltelli et al. 2008). However, the admissible domain in multichoice modeling is generally non-rectangular because choices may enable, exclude, or condition other choices. For such constrained decision– uncertainty domains, simple random sampling is recommended over others (Kucherenko et al. 2011). Moreover, the simple binning estimator (Kozlova et al. 2025) used in the subsequent analysis has been shown to perform well under both simple random and quasirandom sampling, including such small sample size as $1 0 ^ { 2 }$ (Helo et al. 2026). In multichoice modeling, we therefore use simple random sampling.

Sample size. In computer experiments, the required sample size is typically formulated as a function of input dimensions d. A widely cited rule of thumb for initial designs prescribes $N = 1 0 d$ (Loeppky et al. 2009). Later work emphasizes that adequate sample size also depends on response-surface complexity, efect sparsity, and the purpose of the experiment (Harari et al. 2018). In multichoice modeling, however, sampling pressure is not determined by input dimensionality alone, because the analysis compares outcome distributions across admissible choice configurations. We therefore report sample size relative to the cardinality of the admissible configuration space |S| and the number of uncertainty sources $d _ { X }$ . As a practical benchmark, for analyses intended to compare full admissible configurations, we recommend

$$
N = 1 0 0 | S | ( d _ { X } + 1 ) .\tag{5}
$$

Larger samples are advised when the model includes stochastic process uncertainty, tail behavior, or rarely occurring configurations induced by choice dependencies. Smaller samples may still support low-dimensional conditional analysis, but should not be interpreted as stable evidence for every full configuration.

Simulation dataset. Each simulation run draws one feasible choice configuration $z _ { r } \in S$ from a design distribution $q ( z )$ and one uncertainty realization $x _ { r } \sim P ( \cdot \mid z _ { r } )$ , producing model outputs $y _ { r } = f ( z _ { r } , x _ { r } )$ :

$$
{ D _ { N } } = \{ ( z _ { r } , x _ { r } , y _ { r } ) \} _ { r = 1 } ^ { N } .\tag{6}
$$

The resulting dataset, with choice and uncertainty variables treated as analyzed inputs W, forms the basis for subsequent sensitivity-guided exploratory analysis.

## 3.2.3 Exploratory analysis.

Multichoice modeling examines how configurations of interdependent choices interact with uncertainty to shape performance outcomes. Rather than identifying a single optimal configuration, the framework explores the simulated mapping $( z , X ) \mapsto Y$ to identify influential choices, uncertainty conditions, and partial configurations associated with distinct outcome distributions.

Variance-based sensitivity analysis. First, global sensitivity analysis is used to screen the analyzed inputs W, which consists of choice variables and uncertainty variables, according to their contribution to output variation. The eficient binning estimator of Sobol’ indices is employed because it can be applied to dependent data and does not require a specific experimental design, unlike mainstream estimators (Kozlova et al. 2025). For an input variable $W _ { i } \in { \mathcal { W } }$ , the first-order efect measures the variance explained by conditioning on $W _ { i }$ alone:

$$
S _ { i } = \frac { \operatorname { V a r } _ { W _ { i } } \left( \mathbb { E } [ Y \mid W _ { i } ] \right) } { \operatorname { V a r } ( Y ) } .\tag{7}
$$

For a pair $( W _ { i } , W _ { j } )$ , the second-order efect measures the additional variance explained by their joint conditioning beyond the individual efects:

$$
S _ { i j } = \frac { \operatorname { V a r } _ { W _ { i } , W _ { j } } \left( \mathbb { E } [ Y \mid W _ { i } , W _ { j } ] \right) } { \operatorname { V a r } ( Y ) } - S _ { i } - S _ { j } .
$$

Combined sensitivity indices $C _ { i }$ aggregate these efects into an importance measure for ranking variables

$$
C _ { i } = S _ { i } + \frac { 1 } { 2 } \sum _ { j \neq i } S _ { i j } .\tag{8}
$$

In practice, often only a small subset of inputs explains the majority of output variance in computational models (Loeppky et al. 2009). The sensitivity results therefore serve as a navigation device: they indicate which choice combinations should be examined first.

Conditional outcome distributions. Second, the framework examines conditional outcome distributions associated with influential choices identified by the sensitivity ranking. Direct comparison of all configuration-level distributions $P ( \boldsymbol { Y } \mid z )$ is typically infeasible when $s$ is combinatorially large. Instead, the analysis compares conditional projections of the form

$$
P ( Y \mid z _ { B } = \alpha _ { B } ) ,\tag{9}
$$

where $B \subseteq { \mathcal { C } }$ is a selected subset of influential choices, $z _ { B }$ denotes the components of $z$ associated with the alternatives pertaining to those choices, and $\alpha _ { B }$ ranges over admissible partial configurations of $B$ . Thus, each conditional distribution corresponds to one admissible combination of alternatives for the selected choices, while uncertainty realizations and all non-selected choices continue to vary over their admissible ranges within the corresponding subset of the simulation dataset. The comparison therefore does not evaluate isolated deterministic strategies, but conditional projections of the full choice–uncertainty space. The algorithms for sensitivity screening and conditional distribution analysis are available as open-source implementations in Python, R, and Matlab, and through a free no-code web dashboard (Kozlova et al. 2024, Roy and Kozlova 2024).

Results interpretation. Choice configuration performance is assessed through the location and shape of the conditional outcome distributions. Diferences in location indicate partial configurations associated with either higher or lower average performance, while diferences in spread show how much variation remains after conditioning on the selected choices. The distributions also reveal choice interactions, since the efect of one choice may difer depending on the alternatives selected for other choices, rendering some alternatives attractive only in specific configurations. They can further reveal uncertainty-dependent performance, where a choice performs well only in particular regions of the uncertainty space, as opposed to configurations that remain satisfactory across a wider range of conditions. The same decomposition logic can be applied to secondary outputs to assess tradeofs across performance dimensions.

Iterative analysis. Once appealing, robust, or problematic configuration regions have been identified, the dataset can be filtered by those regions of the decision space and the sensitivity and conditional distribution analysis repeated. A new sensitivity profile may then emerge, because the importance of choices and uncertainties can vary across regions of the decision–uncertainty space. Importantly, zooming in is possible by fixing choices that are not contingent on uncertainty. Otherwise, the procedure filters out entire regions of uncertain futures instead of isolating the efects of the selected decision configurations. Finally, the process may also trigger model reformulation. If the observed performance profiles suggest missing design degrees of freedom, additional choices can be added and the multichoice modeling procedure repeated from the beginning.

## 4 Cases

This section presents three applications of multichoice modeling. We begin with a simple mortgage case that is accessible to readers regardless of background, and then proceed to two more specialized settings: a manufacturing line upgrade case and an M&A integration strategy case.

In each case, we first perform a conventional analysis with a fixed decision architecture and then introduce multichoice modeling to demonstrate how decision support is enhanced when the decision configuration space is made explicit and explored jointly with uncertainty.

The Discussion section synthesizes the insights that emerge from this shift in perspective and positions them relative to established decision-support approaches.

## 4.1 Mortgage

In Finland and many other countries, mortgage negotiations typically focus on reducing the interest-rate margin across ofers from several banks, while the remaining terms are often treated as fixed. This case considers a typical mortgage ofer, based on the authors’ personal experience of mortgage negotiation in 2024. For a loan of 250,000 €, the negotiated margin was 0.6%, implying a monthly payment of 1,199 €, as stated in the bank’s ofer (Table 1).

Table 1: Mortgage details
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Loan</td><td>€250,000</td></tr><tr><td>Reference rate</td><td>2.5%</td></tr><tr><td>Margin</td><td>0.6%</td></tr><tr><td>Duration</td><td>25 years</td></tr><tr><td>Monthly payment</td><td>€1,199</td></tr></table>

Table 2: Interest-rate parameters
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Normal mean level</td><td>1%</td></tr><tr><td>Mean reversion speed</td><td>10%</td></tr><tr><td>Volatility</td><td>0.2%</td></tr><tr><td>Crisis mean level</td><td>6.5%</td></tr><tr><td>Crisis duration</td><td>5 years</td></tr><tr><td>Crisis occurrence</td><td>binary 50/50</td></tr></table>

This adds up to €359,572 paid over the 25-year loan period, corresponding to €109,572 in interest on top of the €250,000 principal. However, this figure is only a point estimate: the ofer is based on a floating interest rate, so the actual interest expense depends on the future path of the reference rate. Estimating the range of possible interest expenses therefore requires modeling interest-rate uncertainty.

## 4.1.1 Conventional analysis.

Although the mortgage choices are made at the outset, their consequences unfold through monthly payments over the loan period. To represent this time-dependent execution, the interest rate is modeled as a stochastic process rather than as a static range. Specifically, we use the mean-reverting Vasicek process $r _ { t + 1 } = r _ { t } + a ( b - r _ { t } ) \Delta t + \sigma \sqrt { \Delta t } \varepsilon _ { t }$ , where $r _ { t }$ denotes the interest rate at time t, a is the speed of mean reversion, b is the long-term mean level, $\sigma$ is the volatility, $\Delta t$ is the time step, and $\varepsilon _ { t } \sim \mathcal { N } ( 0 , 1 )$ . For monthly simulations, the annual mean-reversion speed and volatility are adjusted by dividing them by 12 and ${ \sqrt { 1 2 } }$ respectively.

To reflect divergent expert estimates about future interest-rate paths, and to evaluate the resilience of alternative mortgage tactics, we model two scenarios for interest-rate evolution:

“normal times” and “crisis”, see Table 2. This bimodal stochastic process is visualized in Figure 2, left.

The resulting distribution of cumulative interest expenses has two clear peaks (Figure 2, right), which is a direct consequence of the uncertainty scenario specification. The left peak corresponds to normal times, where cumulative interest expenses fall roughly between €50,000 and €100,000. The right peak reflects the simulated crisis scenario, in which expectedly total interest expenses can rise to approximately €160,000, or 64% of the loan principal. Knowing the possible extent of future costs, however, does not yet tell us what to do about them.

![](images/e5d6fe578280cfe4435ae9d033bf1c70254cf207f41bb96c64720441aa99c342.jpg)

![](images/73016a33859cf9e37c681bdfae05b6cce3779300f115764fc3f1af5644194630.jpg)  
Figure 2: Ten simulated paths of interest rate evolution under the mean-reverting process (left) and the resulting distribution of cumulative interest-rate expenses (right).

## 4.1.2 Multichoice modeling.

Once the decision is framed as a choice architecture rather than a fixed ofer, several additional options become visible, even though they were not explicitly presented by the bank (Table 3).

Table 3: Strategy table for mortgage choices
<table><tr><td>Choice</td><td>Dependency</td><td>Effect in the model</td></tr><tr><td>Rate type</td><td>None</td><td>Fixed 4.1% / stochastic process</td></tr><tr><td>Interest rate cap</td><td>Variable rate only</td><td>5% cap at 0.05% premium / no cap</td></tr><tr><td>Payment scheme</td><td>Variable rate only</td><td>Fixed payment / fixed term</td></tr><tr><td>Reference rate</td><td>Variable rate only</td><td>3-month / 12-month Euribor</td></tr></table>

The first choice does not require simulation, since a fixed-rate mortgage produces a deterministic outcome that can be calculated directly using the model already developed. We therefore embed the remaining three choices into the model with interest-rate uncertainty.

For this model structure, Equation 5 implies a benchmark sample size of 1,600 simulation runs. With 5,000 simulation runs, the case provides adequate sampling density for comparing full admissible configurations. The resulting simulation dataset records the binary input variables, namely the uncertainty scenario and the mortgage choices, together with the output variable, cumulative interest-rate expenses.

Table 4: Additional inputs for the multichoice model
<table><tr><td>Input</td><td>Value</td></tr><tr><td>Interest rate cap, no/yes</td><td>Binary 50/50</td></tr><tr><td>Cap level</td><td>5%</td></tr><tr><td>Cap fee</td><td>0.05%</td></tr><tr><td>Reference rate, 3-month/12-month</td><td>Binary 50/50</td></tr><tr><td>3-month rate mean reversion speed</td><td>20%</td></tr><tr><td>3-month rate volatility</td><td>0.3%</td></tr><tr><td>Fixed payment/term</td><td>Binary 50/50</td></tr></table>

Table 5: Sensitivity indices
<table><tr><td>Variable</td><td>Index</td></tr><tr><td>Crisis yes/no</td><td>85.30%</td></tr><tr><td>Fixed payment/term</td><td>1.61%</td></tr><tr><td>Reference rate, 3m/12m</td><td>1.16%</td></tr><tr><td>Interest rate cap, no/yes</td><td>0.02%</td></tr></table>

Variance-based sensitivity analysis identifies the uncertainty scenario as the dominant driver of interest-rate expenses. This is expected, given the intentionally strong contrast between the normal-times and crisis scenarios. More importantly for decision support, two mortgage choices receive non-zero sensitivity indices, as shown in Table 5.

Figure 3 depicts the conditional distributions produced by combinations of payment schemes and reference-rate maturities under the normal-times (blue) and crisis (red) scenarios. Although the corresponding sensitivity indices are low, the choices remain economically meaningful: even under stable conditions, the diference between the best and worst strategies is approximately €20,000.

![](images/460584d5125d458f568df81d94dcadd04edacac77921b85dacfb32966ac0f57a.jpg)

<table><tr><td rowspan="2">Color</td><td rowspan="2">Crisis</td><td rowspan="2">Type of payment</td><td rowspan="2">Reference rate</td><td colspan="3">Interest expenses, €</td></tr><tr><td>min</td><td>mean</td><td>max</td></tr><tr><td>8</td><td rowspan="4">no</td><td rowspan="2">fixed term</td><td>12-month</td><td>56 802</td><td>82 460</td><td>109 536</td></tr><tr><td>7</td><td>3-month</td><td>41 656</td><td>70 708</td><td>98 967</td></tr><tr><td>6</td><td>fixed</td><td>12-month</td><td>50 109</td><td>76 982</td><td>111 520</td></tr><tr><td>5</td><td>payment</td><td>3-month</td><td>37 916</td><td>64 188</td><td>93 319</td></tr><tr><td>4</td><td rowspan="4">yes</td><td rowspan="2">fixed term</td><td>12-month</td><td>99 100</td><td>127 005</td><td>155 456</td></tr><tr><td>3</td><td>3-month</td><td>101 207</td><td>128 296</td><td>158 210</td></tr><tr><td>2</td><td rowspan="2">fixed payment</td><td>12-month</td><td>91 825</td><td>139 200</td><td>194 611</td></tr><tr><td>1</td><td>3-month</td><td>103 086</td><td>142 415</td><td>184 674</td></tr></table>

Figure 3: Conditional distributions produced by the combination of the dominant uncertainty variable and two mortgage choices.

The relative performance of the payment schemes reverses across scenarios. Under crisis conditions, the fixed-payment scheme generates higher cumulative interest expenses than the fixed-term plan because rising rates shift a larger share of payments toward interest, extend the loan term, and increase total borrowing costs. Under normal conditions, declining rates shorten the loan under the fixed-payment scheme, making it less costly than the fixed-term alternative. Thus, fixed payment performs better in stable environments, whereas fixed term is more robust under adverse interest-rate paths.

Nonlinear efects are also observed for reference-rate maturity. Under crisis conditions, reference-rate maturity has almost no efect. Under normal conditions, however, the 3-month rate yields consistently lower interest costs than the 12-month rate. This follows from its faster mean reversion, despite its higher volatility. Historical Euribor data are consistent with this pattern: in the observed data, the 3-month rate has been lower than the 12-month rate in more than 95% of observations. The cost advantage of the shorter maturity is therefore empirically supported.

The interest-rate cap, omitted from the decomposition in Figure 3, has near-zero importance (Table 5). Because the ofered 5% cap is rarely triggered in the simulated interest-rate paths (Figure 2, left), the added margin premium provides little protection while increasing expected costs.

Finally, a fixed-rate mortgage at 4.1% results in total interest expenses of approximately €150,000, exceeding most simulated outcomes, including many crisis cases. It is therefore not competitive in this setting.

Overall, the fixed-payment and 3-month reference-rate combination performs best under normal conditions but becomes one of the worst options in crisis. In contrast, the fixed-term and 3-month reference-rate configuration performs well in both uncertainty scenarios. It is therefore the most robust choice across the modeled futures, unlike the default ofer based on fixed payments and a 12-month reference rate.

The shift from conventional mortgage evaluation to multichoice modeling not only makes the available choices explicit but also crucially changes the recommendation. A conventional analysis asks how risky the ofered mortgage is under possible interest-rate paths. The multichoice model asks which contract configuration is preferred once payment scheme, referencerate maturity, and protection mechanisms are treated as explicit choices. The resulting recommendation is a reconfiguration of the decision object itself.

## 4.2 Manufacturing process upgrade

This case is based on a project with Hyperion Robotics, an innovative Finnish construction company. The project was conducted at the end of 2023 and documented in detail by Walzer et al. (2026). In this paper, we present a reduced model for ease of exposition while preserving the relevant complexity of the decision space, the actual progression of the project, and the insights derived at each stage. Some details are anonymized in accordance with the company’s requirements.

The company already had a deterministic model for computing the unit cost of its 3Dprinted construction element. The model links production parameters to the associated cost components, with unit cost defined as the ratio of total daily cost to daily output. The production process consists of three stages: preparation, printing, and post-processing. Preparation and post-processing activities together require 5.6 hours per operational day. During the printing stage, the robotic system operates at a nominal material deposition rate of 2 tons per hour, corresponding to $0 . 8 3 ~ \mathrm { m ^ { 3 } / h }$ for a concrete density of $\mathrm { 2 . 4 ~ t / m ^ { 3 } }$ . The intermittent installation of steel reinforcement elements reduces efective productivity to 60% of the nominal printing rate.

With these assumptions (Table 6), the model returns a total daily cost of €4,389. Given a daily output of 0.96 units, this corresponds to a current unit cost of €4,572 per unit.

Table 6: Input variables in the unit cost model  
Table 7: Improvement potential in input variables
<table><tr><td>Input variable</td><td>Value</td></tr><tr><td>Production Non-printing time, h/day</td><td>5.6</td></tr><tr><td>Volume,  $\mathrm { m ^ { 3 } / u n i t }$  Printing speed, ton/h</td><td>1.25 2</td></tr><tr><td>Robotic system productivity</td><td>60%</td></tr><tr><td>Costs</td><td></td></tr><tr><td>Workers, persons</td><td>6</td></tr><tr><td>Salary,  $\mathrm { \epsilon \epsilon ~ } \mathrm { \epsilon } \mathrm { / h }$ </td><td>22</td></tr><tr><td>Buffer/waste, % concrete use</td><td>100%</td></tr><tr><td>Concrete price,  $\mathrm { \overline { { e } } / t o n }$ </td><td>441</td></tr><tr><td>Steel reinforcement, % volume</td><td>1%</td></tr><tr><td></td><td></td></tr><tr><td>Steel reinforcement price,  $\mathrm { { \overline { { e } } / k g } }$ </td><td>2</td></tr><tr><td>Self-compacting concrete,  $\mathrm { m ^ { 3 } / u n i t }$  Self-compacting concrete price,  $\mathrm { { \overline { { e } } / m ^ { 3 } } }$ </td><td>0.63 1000</td></tr></table>

<table><tr><td>Variable</td><td>Current</td><td>Better</td></tr><tr><td>Concrete price, €/ton</td><td>441</td><td>223</td></tr><tr><td>Buffer/waste</td><td>100%</td><td>50%</td></tr><tr><td>Self-compacting concrete price,  $\mathrm { { \bf { \mathrm { E } } / \mathrm { { m ^ { 3 } } } } }$ </td><td>1000</td><td>300</td></tr><tr><td>Preparation time, h/day</td><td>5.6</td><td>2</td></tr><tr><td>Printing speed, ton/h</td><td>2</td><td>4</td></tr><tr><td>Robotic system productivity</td><td>60%</td><td>100%</td></tr></table>

## 4.2.1 Conventional analysis.

The initial intuition was to map which input variables in the existing model could be improved, and by how much, and then to perform two analyses: (i) a one-at-a-time sensitivity analysis to identify which variables had the strongest efect on unit cost, and (ii) a Monte Carlo simulation to examine their joint efect. The company provided estimates of the feasible improvement ranges for several model inputs, as shown in Table 7. These values were first used in a basic sensitivity analysis, varying one input at a time. For the Monte Carlo simulation, uniform distributions between the current and improved values were assumed, since these parameter ranges represented possible improvements rather than probabilistic uncertainty. The results are presented in Figure 4.

![](images/4b48ce4cf7b3297b226bcf3be4f85e6c9b009697af5e49f27f9171aa2aabc1a5.jpg)

![](images/3cc7e5b95142a553a59582245a2baa059e86fc2e00ccfa72025c2c02776cbef6.jpg)  
Figure 4: One-at-a-time sensitivity analysis (left) and Monte Carlo simulation (right) results.

The results confirmed the company’s intuition that the concrete price was the most influential variable. More unexpectedly, the analysis showed that the combined improvement of the selected inputs could reduce the unit cost by roughly half.

However, these results were not suficient to guide an investment decision. Each variable in the one-at-a-time sensitivity analysis was, in practice, the result of one or more possible upgrades, and these upgrades could afect several variables simultaneously.

## 4.2.2 Multichoice modeling.

To support the decision on which upgrades to implement, the upgrades themselves needed to be incorporated into the model. Each upgrade afects specific subsets of model variables in predefined ways. For instance, material cost is no longer represented by a single parameter; instead, the model includes three alternative material choices, each with a distinct unit price. Their availability depends on the hardware configuration, since certain materials can only be used if specific hardware upgrades are selected. The production system is also subject to a space-related capacity constraint, which currently limits output to two units per day. This capacity can be increased through additional hardware upgrades that modify the corresponding capacity parameter. Finally, each upgrade has its own investment cost, so improvements in unit cost must be evaluated against the required capital expenditure.

Listing the upgrades together with their efects on model variables and their interdependencies naturally produced a strategy table, shown in Table 8.

To integrate investment costs into the unit cost calculation, the former are converted into daily annuities using a discount rate of 5% and a time horizon of three years. All technology upgrades are modeled as binary choices. The company first wanted to examine the upgrade choice architecture alone, while keeping the workflow simulation-ready so that uncertainty sources could be introduced in subsequent analyses.

Equation 5 implies a benchmark sample size of 44,800 simulation runs for such a model. However, this case is not intended to support comparison of all full admissible configurations.

Table 8: Upgrade options, efects, and computed sensitivity indices
<table><tr><td>Area/upgrade</td><td>Cost, K€ Effect</td><td></td><td>Sensitivity index</td></tr><tr><td>Material</td><td></td><td></td><td>3%</td></tr><tr><td>Concrete 2</td><td></td><td>40 Price 301 €/ton; available with Mixer upgrade.</td><td></td></tr><tr><td>Concrete 3</td><td></td><td>40 Price 223 €/ton; available with Mixer and Pump upgrades.</td><td></td></tr><tr><td>Hardware</td><td></td><td></td><td>20%</td></tr><tr><td>Mixer</td><td></td><td>90 Enables self-compacting concrete up to 300; enables Concrete 2.</td><td></td></tr><tr><td>Printing head 1 Printing head 2</td><td></td><td>50 Reduces labor by one worker. 100 Reduces buffer/waste to 10%;</td><td>0% 22%</td></tr><tr><td></td><td></td><td>changes space constraint to 4 units per day; reduces non-printing time by 0.5 h.</td><td></td></tr><tr><td>Robot</td><td>300</td><td>Changes space constraint to 6 units per day.</td><td>1%</td></tr><tr><td>Pump</td><td>100</td><td>Reduces non-printing time by 1.5 h; reduces buffer/waste to 10%; reduces labor by two workers; increases pro- ductivity to 100%; enables Concrete</td><td>49%</td></tr><tr><td>Steel reinforcement</td><td>150</td><td>3 if Mixer upgrade is selected. Reduces non-printing time by 0.5 h; reduces labor by one worker; changes</td><td>2%</td></tr><tr><td>Software</td><td></td><td>productivity to 80%.</td><td></td></tr><tr><td>Print speed software</td><td></td><td>25 Increases printing speed, ton/h, by a</td><td>1%</td></tr><tr><td>Software library</td><td></td><td>factor of 1.5. 200 Reduces non-printing time by 1.6 h.</td><td>1%</td></tr></table>

Its purpose is to identify the first moves that most strongly shape performance. We therefore use an approximately one-tenth of the benchmark size, 5,000 runs. A robustness check with a larger sample led to the same conclusion.

Variance-based sensitivity analysis identified Pump upgrade, Printing head upgrade 2 and Mixer upgrade as the most influential, Table 8, last column. All other choices gained negligible indices. Interestingly, Material yielded a small sensitivity index, even though it directly defines the concrete price, which was flagged as the most influential model input in the one-at-a-time analysis of a non-multichoice model.

Figure 5 decomposes the simulation results into conditional distributions defined by these three upgrades. The analysis of upgrade combinations confirms that the configuration with all three key upgrades (Pump upgrade, Printing head upgrade 2, and Mixer upgrade) yields the lowest observed unit cost, while the configuration with none of them yields the highest. Between these extremes, however, the efects are strongly nonlinear.

![](images/dff2cf790e6473bc3a51187ffa3428424dea4d5d523610e678312794decc79fa.jpg)

<table><tr><td rowspan="2">Color</td><td rowspan="2">Pump upgrade</td><td rowspan="2">Printing head upgrade 2</td><td rowspan="2">Mixer upgrade</td><td colspan="3">Unit cost, €/unit</td></tr><tr><td>min</td><td>mean</td><td>max</td></tr><tr><td>8</td><td rowspan="4">yes</td><td rowspan="2">yes</td><td>yes</td><td>1349</td><td>1791</td><td>2283</td></tr><tr><td>7</td><td>no</td><td>2476</td><td>2571</td><td>2693</td></tr><tr><td>6</td><td rowspan="2">no</td><td>yes</td><td>1340</td><td>1853</td><td>2401</td></tr><tr><td>5</td><td>no</td><td>2479</td><td>2636</td><td>2797</td></tr><tr><td>4</td><td rowspan="4">no</td><td rowspan="2">yes</td><td>yes</td><td>1719</td><td>2380</td><td>3176</td></tr><tr><td>3</td><td>no</td><td>2590</td><td>2940</td><td>3539</td></tr><tr><td>2</td><td rowspan="2">no</td><td>yes</td><td>2626</td><td>3526</td><td>4540</td></tr><tr><td>1</td><td>no</td><td>3870</td><td>4212</td><td>4887</td></tr></table>

Figure 5: Conditional distributions of unit cost across configurations defined by the three most influential choices.

The Printing head upgrade 2 substantially reduces unit cost when the Pump upgrade is not implemented. In this case, the corresponding boxplots, cases 3 and 4, are shifted markedly to the left relative to the baseline without the Printing head upgrade 2, as in cases 1 and 2. When the Pump upgrade is already in place, however, the marginal efect of the Printing head upgrade 2 becomes negligible: the distributions with and without it, cases 5 versus 7 and 6 versus 8, are nearly indistinguishable. This indicates a strong overlap in the efect mechanisms of the Pump upgrade and Printing head upgrade 2. Moreover, a relatively low unit cost can be achieved without relaxing the space constraint, corresponding to production remaining below the two-units-per-day limit.

The Mixer upgrade has a substantial efect even when the Pump upgrade is already implemented. Without the Mixer upgrade, the unit cost distribution is narrower and remains at a higher level, as shown by cases 5 and 7. Activating the Mixer upgrade widens the distribution and shifts it downward. This occurs because the Mixer upgrade enables the use of Concrete 2 ; when combined with the Pump upgrade, it also enables Concrete 3. Because these materials have lower prices, they widen the conditional unit cost distributions by extending them towards lower values.

In the variance-based sensitivity analysis across all simulated scenarios, the Material variable has a relatively low sensitivity index of 3%, despite its evident importance in certain configurations. This pattern is consistent with highly nonlinear models, where a factor may have strong influence within a restricted subset of the design space but a diluted average impact across all combinations. When the analysis is conditioned on scenarios in which both the Pump upgrade and Mixer upgrade are active, cases 6 and 8, material choice becomes the dominant driver of unit cost with a sensitivity index of 92%. In this subset, Concrete 3 yields the lowest unit costs, with values below €1,700 per unit, Figure 6.

![](images/092ebaa7a1a17678d534ca0fbb54b5c1adff5ce600a649dcde20cd6b889bd88d.jpg)

<table><tr><td rowspan="2">Color</td><td rowspan="2">Material</td><td colspan="3">Unit cost, €/unit</td></tr><tr><td>min</td><td>mean</td><td>max</td></tr><tr><td>3</td><td>Concrete 3</td><td>1340</td><td>1489</td><td>1702</td></tr><tr><td>2</td><td>Concrete 2</td><td>1597</td><td>1747</td><td>1959</td></tr><tr><td>1</td><td>Concrete 1</td><td>2053</td><td>2194</td><td>2401</td></tr></table>

Figure 6: Conditional distributions for material choices within the subset of configurations with Pump and Mixer upgrades.

The analysis indicates that the largest reductions in unit cost are obtained by implementing the Pump upgrade, followed by the Mixer upgrade, and then adopting Concrete 3. This sequence targets the largest unit cost reductions while keeping the implementation path focused on a small number of simultaneous improvements. Although Printing head upgrade 2 has a high sensitivity index, its efect is material only in configurations where the Pump upgrade is absent. Given its limited marginal contribution under the recommended upgrade path, it is therefore not included. Conversely, Material has a low sensitivity index in the full simulated design space, but becomes the primary driver of unit cost within the subset of scenarios where both the Pump upgrade and Mixer upgrade are active. Therefore, Concrete 3 is included as the final step once these enabling upgrades are in place.

This case presents two diferent models of the same underlying production system. The first follows a conventional modeling practice: it computes unit cost as a function of operational parameters and supports basic model analysis, but it does not directly indicate which actions the company should take. The second is a multichoice model: it encodes the behavior of the unit cost model as a function of available upgrade choices, including their dependencies and enabling conditions. As a result, this level of analysis reveals not only which variables matter, but which choices need to be undertaken, in what sequence, and under which configurations. Although both models compute unit cost, the emphasis of the decision support outcome is entirely diferent. The conventional model asks what would happen if selected input parameters changed. The multichoice model asks what the company can actually do to make those changes happen. This distinction is central because actionable decision support requires moving from sensitivity to variables toward analysis of the choice architecture.

## 4.3 M&A strategy

This synthetic case was constructed based on consultations with two independent senior M&A experts based in Finland. In practice, acquisitions are rarely analyzed as a space of alternative choices. Instead, decision makers typically formulate a single acquisition strategy, define an integration approach, and evaluate the transaction through identified synergies. A discounted cash flow valuation is then used to compare the estimated value of the target with the proposed purchase price and support the final decision.

In the modeled case, an equipment manufacturing company considers acquiring a target firm operating in the equipment service and refurbishment market. The acquisition is motivated by two potential synergies. First, the buyer expects market synergies through cross-selling opportunities between the customer bases of the two businesses. Second, integration may create operational flexibility, allowing capacity to be shifted between manufacturing and service activities as their relative attractiveness changes over time. The baseline characteristics of the businesses are summarized in Table 9.

## 4.3.1 Conventional analysis.

To incorporate the value of production-mix flexibility, the attractiveness of the equipment and service markets is modeled stochastically through random walk. This allows the two markets to develop distinctly over time, creating opportunities for capacity reallocation. The equipment market is modeled as more volatile compared to the service market, as illustrated in Figure 7 (left). Other uncertainties include market synergy potential and integration success. Numeric assumptions are summarized in Table 10.

Table 9: Baseline assumptions of the M&A case  
Table 10: Uncertainty assumptions in the M&A case
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Deal price, M€</td><td>80</td><td>Integration success</td><td>U(0.4, 1.0)</td></tr><tr><td>Existing business EBITDA, M€/year</td><td>25</td><td>Market synergy potential, M€/year</td><td>U(0, 10)</td></tr><tr><td>Service EBITDA, M€/year</td><td>10</td><td>Equipment market attractiveness</td><td>Random walk</td></tr><tr><td>Equipment capacity, units</td><td>80</td><td>Initial value</td><td>1.0</td></tr><tr><td>Service capacity, units</td><td>20</td><td>Volatility</td><td>0.20</td></tr><tr><td></td><td></td><td>Service market attractiveness</td><td>Random walk</td></tr><tr><td></td><td></td><td>Initial value</td><td>1.0</td></tr><tr><td></td><td></td><td>Volatility</td><td>0.10</td></tr></table>

![](images/5c8c0916c7a474dffaca4824c3d5794273d5816ff5ccd17115b42664aaf84bc7.jpg)

![](images/c9569684ed7fdd2957ae5a3c023283da81692094ed5965ba77dc07dda1a97694.jpg)  
Figure 7: Five sample realizations of the stochastic equipment and service market attractiveness processes (left) and distribution of simulated net present value (NPV) outcomes for the M&A case (right).

The value of operational flexibility is captured through the possibility of reallocating production capacity between equipment manufacturing and service activities. The maximum share of capacity that can be reallocated is determined by integration success, $F = 0 . 5$ integration success, reflecting the idea that successful integration improves the organization’s ability to coordinate resources across business units.

In each year t, profitability per unit of capacity is calculated separately for equipment manufacturing and service activities as $\pi _ { i , t } = A _ { i , t } E B I T D A _ { i } / C _ { i }$ , where $A _ { i , t }$ is market attractiveness, EBIT DA<sub>i</sub> is the baseline EBITDA of business i, and $C _ { i }$ is its production capacity.

The annual capacity reallocation is then determined as

$$
\Delta C _ { t } = F \frac { \pi _ { E , t } - \pi _ { S , t } } { \pi _ { E , t } + \pi _ { S , t } } .\tag{10}
$$

Positive values of $\Delta C _ { t }$ indicate capacity reallocation towards equipment manufacturing, whereas negative values indicate reallocation in favor of service activities.

Enabling operational flexibility requires a one-time investment of €10M. Annual flexibility costs are assumed equal to 10(1 − integration success) M€, representing coordination ineficiencies and organizational friction associated with operating a flexible production system.

Integration costs are modeled as 5+30(1−integration success) M€, reflecting the greater efort and disruption associated with less successful integrations. Market development incurs a deterministic cost of €2M per year.

The profitability of the acquisition is evaluated over a five-year horizon using a discounted cash flow model with a 5% discount rate. A terminal value equal to four times the final-year EBITDA is added at the end of the planning horizon.

These modeling assumptions translate into a distribution of M&A value concentrated between −50M€ and 150M€ (Figure 7, right). While the overall outlook is reasonably favorable, with only about one-third of outcomes resulting in negative values, the analysis remains limited to evaluating a predefined acquisition strategy under uncertainty. It cannot reveal whether the strategy itself is appropriate, which strategic choices drive performance, and whether the downside risk can be mitigated through alternative acquisition, integration, or market development approaches.

## 4.3.2 Multichoice modeling.

The uncertainty analysis above assumes that the acquisition strategy is fixed. However, acquisition performance depends not only on uncertain future conditions but also on how the deal is structured and managed. To make these strategic degrees of freedom explicit, the model is reformulated as a multichoice model. The acquisition is represented as a set of interacting choices concerning acquisition staging, information use, integration, operational flexibility, and market development, Table 11.

Some of these choices have been previously discussed in the real options literature, where acquisitions may be viewed as options to expand, defer, stage, or abandon investments (Alvarez and Stenbacka 2006, Collan and Kinnunen 2009). However, these options are typically analyzed individually or within a predefined acquisition strategy. Multichoice modeling represents available real options jointly as interacting strategic choices whose performance can be explored jointly under uncertainty.

Only operational flexibility fits the classical real-options setup directly: switching between manufacturing and service activities is exercised in response to evolving uncertainty. Investment-stage real options, such as staging and learning, are typically modeled by optimizing over future uncertain payofs. In multichoice modeling, however, such optimization would conflict with the premise of a rugged decision landscape, where the value of an option may depend on the surrounding configuration of choices. We therefore do not optimize the staged acquisition option directly. Instead, we reformulate it as a choice over decision triggers: if the acquisition is staged, the remaining stake is acquired when integration success is known to have reached a low, medium, or high threshold.

Table 11: Choice structure of the M&A case
<table><tr><td>Choices</td><td>Dependencies</td><td>Effects</td><td>Numeric assumptions</td></tr><tr><td>Staged acquisition • No • Yes</td><td>None</td><td>Delays full commitment and First-stage price is €35M. enables learning about inte- Learned integration success gration success before the re- takes values low (40–60%), maining stake is acquired.</td><td>medium (60–80%), or high (80–100%). Buying trigger Relevant only if Determines the threshold of Directly tied to the levels</td></tr><tr><td>• Low • Medium • High</td><td>= Yes</td><td>staged acquisition learned integration success re- of learned integration suc- quisition.</td><td>quired to proceed with full ac- cess. If learned integration success ≥ trigger, then buy the remaining stake and in- tegrate fully. Second-stage price is €55M with a one- year delay.</td></tr><tr><td>Integration • Partial • Full</td><td>possible</td><td>If buy remaining Full integration increases Full stake = Yes, only market synergy realization (M€): Full integration isandenablesoperational flexibility, but incurs higher integration costs.</td><td>integrationcost 5 + 30(1 - 1 integration success).</td></tr><tr><td>Operational flexibility • No • Yes</td><td></td><td>Available only if Allows capacity reallocation Investment cost is €10M. integration = Full between equipment manufac- Recurring in response to changing mar- M€/year. ket conditions; incurs recur- ring flexibility costs depen- dent on integration success.</td><td>costs are turing and service activities 10(1 - integration success)</td></tr><tr><td>Market development • Passive • Active</td><td>None</td><td>which potential market syn- opment is €2M/year. ergies are pursued through cross-selling and market ex- pansion; active development incurs additional costs.</td><td>Determines the extent to Cost of active market devel-</td></tr></table>

To facilitate interpretation, the stochastic market trajectories are aggregated into a categorical variable, Market. The variable takes values 0 (equipment-dominated), 1 (balanced), and 2 (service-dominated), based on the diference between the cumulative equipment and service market-attractiveness indices. Diferences within ±1 are classified as balanced, while larger deviations are assigned to the market with the higher cumulative attractiveness.

In addition to acquisition NPV, the value of operational flexibility is computed as the incremental cash flows generated by reallocating production capacity between equipment manufacturing and service activities. An ample simulation design of 100,000 runs was used.

Figure 8 compares the distributions and sensitivity profiles of both performance measures. Conditional distributions are constructed out of the two most influential variables for both output measures: uncertain market and the choice to enable operational flexibility.

![](images/d78e5d18b9b184b0a09ee8a1e8a243fd9863956b9713545c046d66bc6a393089.jpg)

![](images/3ecebe9d57134bd5c83ab3ad36545aa6ddfa0c2604102726ae25296a974ad142.jpg)  
Figure 8: Joint analysis of M&A profitability and operational flexibility value: simulated outcomes (left) and sensitivity indices (right).

Figure 8 reveals that operational flexibility creates value primarily when service markets outperform equipment markets, which also generate the most profitable acquisition outcomes. At the same time, many simulation outcomes remain profitable even without operational flexibility, provided that service markets dominate. Balanced and equipmentdominated markets can also result in positive acquisition value, but they are associated with substantially greater downside risk.

The second most important choice afecting acquisition NPV, after operational flexibility, is the integration approach. Conditioning NPV jointly on integration and operational flexibility choices (Figure 9) further supports the conclusion that operational flexibility is the primary source of value creation. Full integration without operational flexibility erodes acquisition value, suggesting that integration is not a value creator in itself but an enabling mechanism for operational flexibility.

![](images/e9f63430385af5b73253a98807b484f1f29a7007729084a10389753c726e6443.jpg)

<table><tr><td>Color</td><td>Integration</td><td>Operational flexibility</td></tr><tr><td>1</td><td rowspan="2">Full</td><td>Enabled</td></tr><tr><td>2</td><td>None</td></tr><tr><td>3</td><td rowspan="2">Partial</td><td>Enabled</td></tr><tr><td>4</td><td>None</td></tr></table>

Figure 9: Conditional distributions of acquisition NPV by integration and operational flexibility choices.

Since market conditions and market synergy potential are the dominant sources of uncertainty, the choice configurations created by operational flexibility and integration were further examined across diferent states of uncertainty. The observed pattern remained stable, suggesting that the conclusion is robust to variations in market conditions and synergy realization.

Another important set of choices, although not highlighted by the sensitivity indices, concerns acquisition staging and the trigger for acquiring the remaining stake. Unlike the previously discussed choices, the performance of these configurations changes across the types of market uncertainty. Therefore, they are analyzed conditional on market type, as shown in Figure 10.

![](images/9d66cdd6a58126f26615eebcfcae39ef0091d4f335809d134bd572919191b019.jpg)

<table><tr><td rowspan="2">Color</td><td rowspan="2">Market</td><td rowspan="2">Staged acquisition</td><td rowspan="2">Buying trigger</td><td colspan="4">NPV, M€</td></tr><tr><td>std</td><td>min</td><td>mean</td><td>max</td></tr><tr><td>1</td><td rowspan="5">Service</td><td rowspan="3">Staged</td><td>High</td><td>26.1</td><td>-55.6</td><td>10.4</td><td>176.8</td></tr><tr><td>2</td><td>Medium</td><td>32.7</td><td>-59.4</td><td>13.8</td><td>238.9</td></tr><tr><td>3</td><td>Low</td><td>36.2</td><td>-61.4</td><td>13.4</td><td>200.0</td></tr><tr><td>4</td><td rowspan="3">One-off</td><td>High</td><td>31.5</td><td>-53.3</td><td>18.0</td><td>260.9</td></tr><tr><td>5</td><td></td><td>Medium</td><td>31.7</td><td>-52.8 18.8</td><td>226.5</td></tr><tr><td>6</td><td></td><td>Low</td><td>30.0</td><td>-46.6</td><td>18.0</td></tr><tr><td>7</td><td rowspan="6">Balanced</td><td rowspan="3">Staged</td><td>High</td><td>15.7</td><td>-67.5</td><td>0.7</td><td>199.7 102.4</td></tr><tr><td>8</td><td>Medium</td><td>20.7</td><td>-71.4</td><td>-1.6</td><td>94.5</td></tr><tr><td>9</td><td>Low</td><td>24.8</td><td>-74.2</td><td>-6.2</td><td>103.0</td></tr><tr><td>10</td><td rowspan="3">One-off</td><td>High</td><td>21.5</td><td>-75.6</td><td>4.3</td><td>111.2</td></tr><tr><td>11</td><td>Medium</td><td>21.8</td><td>-59.6</td><td>4.8</td><td>109.8</td></tr><tr><td>12</td><td>Low</td><td>21.8</td><td>-63.3</td><td>4.8</td><td>110.1</td></tr><tr><td>13</td><td rowspan="6">Equipment</td><td rowspan="3">Staged</td><td>High</td><td>15.3</td><td>-70.1</td><td>-4.3</td><td>85.2</td></tr><tr><td>14</td><td>Medium</td><td>20.4</td><td>-69.5</td><td>-8.5</td><td>81.4</td></tr><tr><td>15</td><td>Low</td><td>24.2</td><td>-74.4</td><td>-14.7</td><td>84.4</td></tr><tr><td>16</td><td rowspan="3">One-off</td><td>High</td><td>21.1</td><td>-63.7</td><td>-3.1</td><td>86.1</td></tr><tr><td>17</td><td>Medium</td><td>22.1</td><td>-69.3</td><td>-3.1</td><td>91.0</td></tr><tr><td>18</td><td>Low</td><td>21.3</td><td>-63.7</td><td>-2.3</td><td>86.1</td></tr></table>

Figure 10: Conditional distributions of acquisition NPV by market type, acquisition staging, and buying trigger choices.

Figure 10 first separates the results into three market types, highlighted by the color groups. Within each market type, the lower three boxes correspond to acquiring the entire target company immediately. In these configurations, the trigger choice is irrelevant because no learning takes place before the acquisition; consequently, the three boxes are nearly identical, with only minor diferences attributable to the simulation noise. The top three boxes correspond to staged acquisition. Here, each box represents a diferent trigger level that determines whether the remaining stake is acquired after observing the realized integration success.

In service-dominated markets, lower acquisition triggers generate the highest expected value (as seen in the comparison of boxes 3 to 1) by preserving access to the substantial benefits of operational flexibility. In contrast, balanced and equipment-dominated markets favor more conservative triggers (as seen in the comparison of boxes 7 to 9 and 13 to 15), which reduce exposure to unfavorable acquisitions. Consequently, no globally optimal trigger exists. A trigger that improves performance under one market type can degrade it under another, illustrating why optimizing individual real options in isolation may be misleading in a rugged decision-uncertainty space.

Independent of market type, higher trigger levels reduce the volatility of acquisition outcomes by screening out a larger share of unfavorable acquisitions. Nevertheless, all trigger levels produce lower expected value than immediate acquisition. Thus, despite the availability of staging and learning options, the preferred strategy under the considered uncertainties and choice configurations is to acquire the target company immediately.

The final choice concerns market development. Having established the preferred acquisition, integration, and operational flexibility configurations, the analysis now conditions on these choices and examines whether active market development contributes to value (Figure 11).

Within the preferred acquisition configurations, active market development improves both the expected value and the upside potential of the deal (boxes 1 and 2 in Figure 11). As expected, however, the additional investment also increases downside exposure by introducing recurring development costs. Interestingly, the value of market development depends on the surrounding configuration of other choices. Under partial integration and without operational flexibility, passive market development becomes preferable (as seen in the comparison of boxes 7 and 8, as well as 15 and 16). Thus, even this seemingly straightforward choice does not have a universally preferred alternative; its attractiveness depends on the broader configuration of strategic choices.

The overall strategy recommended by the multichoice analysis is to acquire the target company immediately, perform full integration, invest in operational flexibility, and pursue active market development. Notably, this is the same strategy that was evaluated in the preceding uncertainty-only analysis. However, the multichoice analysis provides a much stronger justification for this recommendation. Rather than merely quantifying the uncertainty surrounding a predefined strategy, it systematically explores alternative acquisition architectures and demonstrates why competing configurations are inferior.

![](images/fcde9bad6997046f90bfcae5c46d323df7a96db0e59dfda67f88823c41e4547e.jpg)

<table><tr><td rowspan="2">Color</td><td rowspan="2">Staged acquisition</td><td rowspan="2">Integration</td><td rowspan="2">Operational flexibility</td><td rowspan="2">Market development std</td><td colspan="4">NPV, M€</td></tr><tr><td>min</td><td></td><td>mean</td><td>max</td></tr><tr><td>1</td><td rowspan="6">Staged</td><td rowspan="3">Full</td><td>Enabled</td><td>Active</td><td>37.2</td><td>-74.2</td><td>10.9</td><td>238.9</td></tr><tr><td>2 3</td><td></td><td>Passive Active</td><td>33.2 25.8</td><td>-69.6 -73.9</td><td>8.3 -8.2</td><td>200.0 93.5</td></tr><tr><td>4</td><td></td><td>None</td><td>Passive</td><td>18.2</td><td>-74.4</td><td>-12.5</td><td>54.9</td></tr><tr><td>5</td><td rowspan="3">Partial</td><td rowspan="2">Enabled</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td></td></tr><tr><td>6</td><td></td></tr><tr><td>7</td><td rowspan="2"></td><td rowspan="2">None</td><td>Active</td><td>7.5</td><td>-24.1</td><td>-0.8</td><td>27.6</td></tr><tr><td>8</td><td></td><td>Passive</td><td>6.5</td><td>-19.4</td><td>1.8</td><td>25.9</td></tr><tr><td>9 10</td><td rowspan="6">One-off</td><td rowspan="3">Full</td><td rowspan="2">Enabled</td><td>Active Passive</td><td>37.7 33.3</td><td>-63.7</td><td>22.9</td><td>260.9</td></tr><tr><td></td><td></td><td>27.3</td><td>-58.4 -75.6</td><td>19.7</td><td>223.3</td></tr><tr><td>11</td><td rowspan="2">None</td><td rowspan="2">Active Passive</td><td rowspan="2">19.2</td><td rowspan="2"></td><td rowspan="2">-69.3</td><td rowspan="2">3.1 -0.5</td><td rowspan="2">99.5 75.7</td></tr><tr><td>12</td><td></td></tr><tr><td>13 14</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Enabled</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Partial</td><td>Active</td><td>17.3</td><td>-60.8</td><td>-1.5</td><td>70.7</td></tr><tr><td>15</td><td rowspan="4"></td><td rowspan="4"></td><td rowspan="2">None</td><td>Passive</td><td>14.9</td><td>-50.5</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>4.0</td><td>74.1</td></tr><tr><td>16</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Figure 11: Conditional distributions of acquisition NPV by acquisition staging, integration, operational flexibility, and market development choices.

This case illustrates three important points. First, real options such as acquisition staging, learning, and operational flexibility can be naturally represented within a multichoice modeling framework. Second, optimizing individual options in isolation may be misleading on a rugged decision landscape, as their attractiveness depends on the surrounding configuration of choices and uncertainties. Third, both strategic choices and real options can either create or destroy value depending on how they interact with one another and with the evolution of uncertain future conditions.

## 4.4 Discussion

The three cases highlight how multichoice modeling extends established decision-support perspectives by shifting the focus from evaluating predefined strategies to exploring decision architectures under uncertainty, Table 12.

All three cases demonstrated that the decision space can be substantially richer than the predefined strategies that serve as inputs to uncertainty-only analysis. The manufacturing case provides the most striking example: the existing valuation model was unable to support the decision because the company’s actual decision levers were not represented in the fieldstandard model.

Interesting insights emerged for the rugged landscape concept. Combining a structured decision space with external uncertainty can make the performance landscape even more rugged, as illustrated by the mortgage and M&A cases. Moreover, exploration, rather than optimization, reframes the task from ’searching for the best strategy’ to ’getting the most important choices right’, as was specifically requested by the company in the manufacturing case.

The cases show that global sensitivity indices alone cannot provide the full picture. A choice that appears unimportant globally may matter within specific choice configurations, an observation made in every case. Regional sensitivity analysis can partly address this problem by quantifying variable importance within regions of the output. However, it can still disregard important relationships when the configuration in which a choice matters overlaps in performance with other configurations in which it does not. More generally, sensitivity indices do not reveal the direction or shape of input-output relationships. For this reason, complementing global sensitivity analysis with exploratory visualization of conditiona distributions becomes critical for strategy design.

All three cases demonstrate that the notion of the best strategy is conditional on: uncertainty, other choices, and the desired performance characteristics that can emerge during the analysis. In some settings, low variance may be more attractive than a higher mean; in others, robustness across uncertainty scenarios may dominate expected performance in one scenario. The M&A case adds a further insight: some choices are efectively real options, but optimizing these options in isolation would hide their nonlinear performance across choice configurations and uncertainty states.

Conventional uncertainty-only analysis is often met with skepticism by practitioners, a pattern we observed during the development of both the manufacturing and M&A cases. The common rationale is the garbage-in-garbage-out principle: if the numerical assumptions are questionable, the resulting decision recommendation is equally questionable. Indeed, the outcome of the uncertainty-only analysis is often a crisp decision threshold, such as a maximum acceptable probability of losses. Such a number can be highly sensitive to the underlying assumptions.

Multichoice modeling shifts the focus from evaluating a single decision under uncertainty against a numerical threshold to exploring patterns of system behavior. As a result, the analysis becomes less dependent on any particular numerical outcome and more concerned with structural relationships, such as which choices matter, how they jointly shape the performance, and how they interact with uncertainty. These patterns are often driven by the structure and mechanics of the model rather than by the precise parameter values alone. Moreover, the identified inflection points can be further examined, challenged, or refined through targeted information gathering and expert elicitation.

Table 12: How the cases extend established decision-support perspectives
<table><tr><td>Case</td><td>analysis</td><td>Uncertainty-only Rugged landscape Global sensitivity Optimization</td><td>analysis</td><td></td><td></td></tr><tr><td>Mortgage</td><td>Limits attention to Even the default decision choices can change zero sensitivity index configuration is not configuration</td><td>their relative per- may still matter if necessarily the most formance uncertainty</td><td>most variance</td><td>independent A choice with a near- The best-performing under uncertainty explains robust across uncer-</td><td>tainty</td></tr><tr><td>ManufacturingCannot represent the Focus on partial con- A choice with high The best-performing upgrade</td><td></td><td>decision space, as figurations prioritizes sensitivity is not nec- configuration each uncertain model the most critical essarily advisable if it implementing</td><td></td><td>or its positive effect practical decision</td><td>is all has a negative effect upgrades, while the</td></tr><tr><td></td><td>input is influenced by choices first multiple alternative actions.</td><td></td><td>uration)</td><td>can still be critical lo- the improvement cally (within an ap- pealing partial config-</td><td>overlaps with another problem is prioritiz- influential choice; a ing a smaller subset low-sensitivity choice that creates most of</td></tr><tr><td>M&amp;A</td><td>single strategy</td><td>Limits attention to a Same choice configu- Choices may earn low Optimality of a learn- rations can perform sensitivity indices if ing threshold may de- differently in different they only make a dif- pend on uncertainty regions of uncertainty ference in a certain and choice configura-</td><td>choices</td><td>configuration of other tion</td><td></td></tr></table>

In each case, the central question shifted from “How profitable is this strategy under uncertainty?” to “Which choices matter under which conditions, and which configurations of these choices form attractive strategies?” Accordingly, the purpose of the analysis shifted from evaluating a predefined strategy to exploring the space of available choices and designing a strategy from them. This observation highlights the importance of how decision situations are framed and the role analytical tools play in shaping that framing (Quade 1980). It echoes the quotation that opened this paper:

“We shape our tools and thereafter our tools shape us.”

Various authors, commonly attributed to Marshall McLuhan (Kuskis 2013).

The three cases suggest that many established decision-support approaches implicitly constrain attention to a predefined set of alternatives, variables, or optimal solutions. By making the decision architecture explicit and computationally explorable, multichoice modeling shifts the focus toward discovering, comparing, and designing alternative configurations. In this sense, it expands not only the analytical space available to the decision maker, but also the range of strategies that can be meaningfully considered.

## 5 Conclusion

Decision-support approaches have become increasingly sophisticated in representing uncertainty, yet they often operate with a restricted representation of the decision space, typically evaluating one or a small number of predefined strategies. Strategy research, by contrast, has long recognized that strategic performance emerges from rugged landscapes of multiple interdependent choices. This paper connects these perspectives by conceptualizing decision support as the exploration of an uncertainty-contingent rugged decision landscape and ofering a practical workflow to investigate joint uncertainty-decision space.

The three cases demonstrate how the focus of analysis shifts from the conventional valuation of a predefined strategy to the design and discovery of strategies from a richer decision space. Forcing the explicit representation of choices available to the decision-maker enlarges the set of strategies considered and can lead to configurations that difer materially from the predefined default.

Future research could develop more systematic procedures for eliciting choice architectures, adaptive sampling designs for large configuration spaces, and applications in settings where strategic choices and uncertainty are deeply intertwined.

## References

Alam AN, Kozlova M, Yeomans JS (2024) Use of sensitivity analysis in real options studies. Proceedings of the Annual International Conference on Real Options (Bologna, Italy), presented July 11–13, 2024.

Alvarez LH, Stenbacka R (2006) Takeover timing, implementation uncertainty, and embedded divestment options. Review of Finance 10(3):417–441.

Bankes S (1993) Exploratory modeling for policy analysis. Operations research 41(3):435–449.

Boardman AE, Greenberg DH, Vining AR, Weimer DL (2018) Cost-Benefit Analysis: Concepts and Practice (Cambridge: Cambridge University Press), 5 edition.

Bryant BP, Lempert RJ (2010) Thinking inside the box: A participatory, computer-assisted approach to scenario discovery. Technological Forecasting and Social Change 77(1):34–49.

Clemen RT, Reilly T (1999) Making hard decisions with DecisionTools Suite (Duxbury).

Collan M, Kinnunen J (2009) Acquisition strategy and real options. The IUP Journal of Business Strategy VI(3–4):45–65, URL https://ssrn.com/abstract=1532063, date written: January 6, 2010.

Colorni A, Tsouki\`as A (2020) Designing alternatives in decision problems. Journal of Multi-Criteria Decision Analysis 27(3-4):150–158.

Dixit AK, Pindyck RS (1994) Investment Under Uncertainty (Princeton, NJ: Princeton University Press).

Fu MC, ed. (2015) Handbook of Simulation Optimization (New York: Springer), URL http://dx. doi.org/10.1007/978-1-4939-1384-8.

Ganco M, Hoetker G (2009) Nk modeling methodology in the strategy literature: Bounded search on a rugged landscape. Research Methodology in Strategy and Management, volume 5, 237–268 (Emerald Group Publishing Limited).

Ganco M, Kapoor R, Lee GK (2020) From rugged landscapes to rugged ecosystems: Structure of interdependencies and firms’ innovative search. Academy of Management Review 45(3):646– 674.

Gregory R, Failing L, Harstone M, Long G, McDaniels T, Ohlson D (2012) Structured decision making: a practical guide to environmental management choices (John Wiley & Sons).

H¨am¨al¨ainen RP, Lahtinen TJ, Virtanen K (2024) Generating policy alternatives for decision making: A process model, behavioural issues, and an experiment. EURO Journal on Decision Processes 12:100050.

Harari O, Bingham D, Dean A, Higdon D (2018) Computer experiments: Prediction accuracy, sample size and model complexity revisited. Statistica Sinica 899–919.

Helo J, Kozlova M, Roy P, Yeomans JS (2026) Regional variance-based sensitivity analysis for complex models, manuscript under review, Risk Analysis.

Howard RA (1988) Decision analysis: Practice and promise. Management science 34(6):679–695.

Keeney RL, Raifa H (1993) Decisions with multiple objectives: preferences and value trade-ofs (Cambridge University Press).

Kleywegt AJ, Shapiro A, Homem-de Mello T (2002) The sample average approximation method for stochastic discrete optimization. SIAM Journal on optimization 12(2):479–502.

Kozlova M (2017) Real option valuation in renewable energy literature: Research focus, trends and design. Renewable and Sustainable Energy Reviews 80:180–196.

Kozlova M, Ahola A, Roy P, Yeomans JS (2025) Simple binning algorithm and simdec visualization for comprehensive sensitivity analysis of complex computational models. Journal of Environmental Informatics Letters 13(1), URL http://dx.doi.org/10.3808/jeil.202400149.

Kozlova M, Moss RJ, Yeomans JS, Caers J (2024) Uncovering heterogeneous efects in computational models for sustainable decision-making. Environmental Modelling & Software 171:105898.

Kucherenko S, Feil B, Shah N, Mauntz W (2011) The identification of model efective dimensions using global sensitivity analysis. Reliability Engineering & System Safety 96(4):440–449.

Kuskis A (2013) “We shape our tools and thereafter our tools shape us”. URL https://mcluhangalaxy.wordpress.com/2013/04/01/ we-shape-our-tools-and-thereafter-our-tools-shape-us/.

Kwakkel JH, Auping WL, Pruyt E (2013) Dynamic scenario discovery under deep uncertainty: The future of copper. Technological Forecasting and Social Change 80(4):789–800.

Lempert RJ, Bryant BP, Bankes SC (2008) Comparing algorithms for scenario discovery. RAND, Santa Monica, CA .

Lempert RJ, Groves DG, Popper SW, Bankes SC (2006) A general, analytic method for generating robust strategies and narrative scenarios. Management science 52(4):514–528.

Levinthal DA (1997) Adaptation on rugged landscapes. Management science 43(7):934–950.

Liesi¨o J, Salo A, Keisler JM, Morton A (2021) Portfolio decision analysis: Recent developments and future prospects. European Journal of Operational Research 293(3):811–825.

Loeppky JL, Sacks J, Welch WJ (2009) Choosing the sample size of a computer experiment: A practical guide. Technometrics 51(4):366–376.

Marchau VAWJ, Walker WE, Bloemen PJTM, Popper SW, eds. (2019) Decision Making Under Deep Uncertainty: From Theory to Practice (Cham: Springer), URL http://dx.doi.org/ 10.1007/978-3-030-05252-2.

Morgan MG, Henrion M, Small M (1990) Uncertainty: a guide to dealing with uncertainty in quantitative risk and policy analysis (Cambridge university press).

Quade ES (1980) Pitfalls in formulation and modeling. Pitfalls of Analysis, 23–43 (John Wiley & Sons).

Rahmandad H (2019) Interdependence, complementarity, and ruggedness of performance landscapes. Strategy Science 4(3):234–249.

Rivkin JW (2000) Imitation of complex strategies. Management science 46(6):824–844.

Roy PT, Kozlova M (2024) Simulation decomposition in python. Journal of Open Source Software 9(98):6713.

Salo A, Keisler J, Morton A (2011) An invitation to portfolio decision analysis. Portfolio decision analysis: Improved methods for resource allocation, 3–27 (Springer).

Saltelli A, Bammer G, Bruno I, Charters E, Di Fiore M, Didier E, Nelson Espeland W, Kay J, Lo Piano S, Mayo D, et al. (2020) Five ways to ensure that models serve society: a manifesto. Nature 582(7813):482–484.

Saltelli A, Ratto M, Andres T, Campolongo F, Cariboni J, Gatelli D, Saisana M, Tarantola S (2008) Global Sensitivity Analysis: The Primer (Chichester, UK: John Wiley & Sons).

Saltelli A, Tarantola S, Campolongo F (2000) Sensitivity analysis as an ingredient of modeling. Statistical Science 15(4):377–395, ISSN 0883-4237.

Scoones I (2024) Navigating Uncertainty: Radical Rethinking for a Turbulent World (Cambridge: Polity).

Shapiro A, Dentcheva D, Ruszczy´nski A (2009) Lectures on Stochastic Programming: Modeling and Theory (Philadelphia, PA: Society for Industrial and Applied Mathematics), URL http: //dx.doi.org/10.1137/1.9780898718751.

Stepanov R, Kozlova M, Yeomans JS (2024) Unlocking actionability in financial modelling with simulation decomposition. Kozlova M, Yeomans JS, eds., Sensitivity Analysis for Business, Technology, and Policymaking: Made Easy with Simulation Decomposition (SimDec) (London: Routledge), URL http://dx.doi.org/10.4324/9781003453789-6.

Tarantola S, Ferretti F, Piano SL, Kozlova M, Lachi A, Rosati R, Puy A, Roy P, Vannucci G, Kuc-Czarnecka M, et al. (2024) An annotated timeline of sensitivity analysis. Environmental Modelling & Software 174:105977.

ter Horst JT, Steinmann P, Kwakkel JH (2026) Evaluating rule induction algorithms for scenario discovery. Environmental Modelling & Software 107006, URL http://dx.doi.org/10.1016/ j.envsoft.2026.107006, article in press.

Trigeorgis L (1996) Real Options: Managerial Flexibility and Strategy in Resource Allocation (Cambridge, MA: MIT Press).

Trigeorgis L, Reuer JJ (2017) Real options theory in strategic management. Strategic management journal 38(1):42–63.

Vilkkumaa E, Rikkinen P, Liesi¨o J (2025) A framework for selecting action portfolios with incomplete and action-dependent scenario probabilities. Management Science .

Walzer AN, Kozlova M, Mohite A, Yeomans JS, Hall D (2026) How to scale a 3d concrete print-

ing facility? a stochastic decision-support framework for production investment. Journal of Information Technology in Construction 31:180–200.