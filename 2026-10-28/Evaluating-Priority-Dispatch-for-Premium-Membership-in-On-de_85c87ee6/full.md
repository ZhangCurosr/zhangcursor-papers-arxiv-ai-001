# Evaluating Priority Dispatch for Premium Membership in On-demand Taxi-sharing

Nozomi Yamamoto<sup>[0009−0005−5770−5524]</sup>, Hiroki Sakaji<sup>[0000−0001−5030−625X]</sup>

, and Itsuki Noda<sup>[0000−0003−1987−5336]</sup>

Hokkaido University, Sapporo 060-0808, Japan yamamoto.nozomi.i2@elms.hokudai.ac.jp; {sakaji, i.noda}@ist.hokudai.ac.jp

Abstract. In this paper, we analyze the dispatch-level feasibility of introducing premium membership into an on-demand taxi-sharing service using a sequential optimal insertion method. We propose two dispatchlevel methods: deadline shortening for premium users, and deadline shortening with dedicated premium vehicles. We compare them with a baseline dispatch method through SUMO-based simulations under three demand scenarios: steady uniform demand, time-varying demand with a peak period, and urban-focused demand with concentrated peak trafic. Our results show that the suitable priority mechanism depends on operating conditions and premium ratio, suggesting that condition-dependent method switching may provide premium benefits while keeping nonpremium-user burden smaller for overall optimization.

Keywords: On-demand mobility · Priority dispatch · MaaS

## 1 Introduction

On-demand taxi-sharing services support shared trips among users and aim to provide lower fares than conventional taxis. Such services combine the flexibility of taxis with the eficiency of shared public transport. In recent years, transit deserts, which are gaps between public transport supply and travel demand, have been discussed internationally. Such services are expected to help fill such gaps, though their efectiveness and equity implications depend on local conditions and institutional design [8, 3]. They also often depend on subsidies or incentives to secure supply, making long-term profitability dificult to achieve [7, 5]. One possible design option is to provide a premium service to time-sensitive users at a higher fare. Unlike air or rail travel where premium services focus on comfort, a premium service in a taxi-sharing system can potentially change travel time itself by changing dispatch priority. It remains unclear whether such diferences in travel time can be achieved in practice, and whether this can be accomplished without placing an excessive time burden on non-premium users.

In this paper, we examine the introduction of premium membership into ondemand taxi-sharing services. Specifically, we focus on whether a measurable time advantage can be created operationally, leaving fare design for open issue.

Premium users are given priority so that their total travel time from demand generation to arrival can be shortened. We also examine the impact on nonpremium users to identify conditions under which premium service substantially degrades their service quality. To examine this trade-of, we propose and compare two dispatch mechanisms by using SUMO-based simulations: (1) a method that introduces deadline shortening into the sequential optimal insertion method, and (2) a method that extends the first method by reserving part of the fleet for premium users. The main contributions of this paper are as follows.

– We show that deadline shortening creates a travel-time gap mainly when the fleet is under pressure, while its efect becomes limited when vehicle capacity is suficient.

– We show that dedicated premium vehicles can create a clearer gap under ofpeak conditions, but at the cost of reducing the efective capacity available to non-premium users under congestion.

– We show that premium ratio and demand pattern afect both the choice of priority mechanism and the appropriate premium ratio, helping operators design dispatch rules for their operating conditions.

## 2 Related Work

Research on ride-hailing and ride-pooling has mainly developed around dynamic vehicle assignment. Santi et al. quantified pooling potential using shareability networks [12], and Alonso-Mora et al. proposed a real-time optimization framework for large-scale ride-pooling [1]. However, these studies generally assume homogeneous users and do not explicitly address service diferentiation.

Although few studies examine a membership-based premium scheme of the type considered in this study, the underlying issue is closely related to the price–quality-of-service (QoS) tradeof in on-demand mobility. Prior studies have shown that ride-pooling adoption and platform performance are shaped by the tradeof among price, travel time, service quality, and system eficiency [6, 11, 14]. More explicit diferentiation has also been studied through waiting-time differentiation, service quality contracts, and heterogeneous service classes [4, 2, 9]. These studies provide important foundations for diferentiated on-demand mobility, but they mainly focus on pricing or theoretical service design and often abstract away from real-time spatial dispatch interactions.

In contrast, this study isolates the dispatch-level mechanism: before optimizing fares, we evaluate whether priority dispatch can produce a measurable travel-time advantage and what burden this imposes on non-premium users under diferent demand patterns [13].

## 3 Problem Formulation

This study investigates the trade-of inherent in introducing premium membership to on-demand taxi-sharing services, using the Smart Access Vehicle Service (SAVS) — a real-time dispatch platform — as a case study.

## 3.1 Premium and Non-Premium Users

This study classifies users into premium and non-premium groups. Premium users are time-sensitive travelers willing to accept an additional burden, such as a higher fare, in exchange for shorter travel time. The purpose of this classification is to examine dispatch behavior under non-uniform user treatment. Here, the value of premium membership is defined as a reduction in total travel time from demand generation to arrival. Therefore, the analysis considers both the benefit to premium users and the degradation experienced by non-premium users. A desirable premium system should provide clear premium-user benefits while keeping non-premium user burden small.

## 3.2 Allocation Method

SAVS uses the sequential optimal insertion method for dispatching. The method keeps, for each vehicle, an ordered list of waypoints corresponding to the origins and destinations of assigned demands. Once a waypoint sequence is stored, the order of existing waypoints is not changed. Fig. 1 illustrates the concept.

![](images/97d7e58d29c095f0fb29aced779cb6ad4c0e9cc7c044860b7582217b4c3f9f51.jpg)  
Fig. 1. Concept of the sequential optimal insertion method.

The procedure is based on the method described by Noda et al. [10]. First, each demand is divided into two waypoints: an origin and a destination. Each vehicle in the fleet maintains a waypoint list for the demands assigned to it. Second, each vehicle calculates the expected completion times of its assigned demands. Third, when a new demand appears, each vehicle tries to insert the origin and destination of the new demand into all feasible positions in its waypoint list. For each possible insertion, the system calculates the total delay caused at all waypoints from the expected completion times. The insertion cost is defined as this total delay plus the expected completion time of the newly inserted demand.

![](images/66052c3bfcaef6b9e48dadf4bc03550adad1513d9d35e29b3327a3e69cd271ba.jpg)  
Fig. 2. Concept of the proposed dispatch methods. Shorter-Deadline recalculates the allowable deadline interval for premium requests before insertion.

Insertions that cause an existing or new demand to miss its deadline, defined in Section 4.1, are removed from consideration. Finally, the system compares the minimum insertion cost over the fleet and assigns the demand to the vehicle with the smallest cost.

## 3.3 Eficiency Metric

Travel time inherently depends on trip distance. Therefore, we use time per distance (TPD) as the primary metric, defined for each user i as ${ { \tau } _ { i } } \mathrm { ~ = ~ } ( { { w } _ { i } } \mathrm { ~ + ~ }$ $r _ { i } ) / d _ { i }$ , where $w _ { i } , \boldsymbol { r } _ { i } ,$ and $d _ { i }$ represent waiting time, in-vehicle travel time, and travel distance, respectively. Since waiting and travel times are only defined for completed trips, this metric is calculated exclusively for users whose demands were successfully fulfilled. We also use the cancel rate, defined as the proportion of canceled demands among all generated demands. In our analysis, we evaluate the performance using the ensemble average of time per distance, denoted as $\bar { \tau } ^ { P }$ for premium users and $\bar { \tau } ^ { R }$ for non-premium users. Specifically, we first calculate the mean $\tau _ { i }$ for completed demands within each of the 30 independent simulation runs (varied by diferent random seeds), and then average these values across all runs to obtain the final representative metrics. A desirable outcome is one that minimizes $\bar { \tau } ^ { P }$ while preventing an excessive increase in $\bar { \tau } ^ { R }$

## 4 Proposed Dispatch Methods

This section explains the baseline and the two proposed methods. The baseline uses the ordinary sequential optimal insertion method without considering premium membership. Premium labels are retained only for evaluation, and both user groups use the same deadline setting.

## 4.1 Shorter-Deadline

The sequential optimal insertion method incorporates a deadline variable that specifies the maximum allowable travel time from demand generation to arrival. Demands whose predicted arrival times exceed this deadline are rejected and treated as cancellations. This deadline is used as a dispatch feasibility constraint based on predicted travel time, not as an absolute guarantee of arrival time under all real-world trafic conditions.

In this study, we set the baseline allowable time interval $t _ { \mathrm { d e a d l i n e } }$ in proportion to trip distance, defined as: $\begin{array} { r } { t _ { \mathrm { d e a d l i n e } } = \frac { d } { v _ { \mathrm { s l o w e s t } } } } \end{array}$ . Here, d is the estimated travel distance and $v _ { \mathrm { s l o w e s t } }$ is a deadline slack parameter expressed in speed units. The Shorter-Deadline prioritizes premium users by applying a tighter time constraint. This is achieved by recalculating the deadline using an increased speed parameter $v _ { \mathrm { p r e m } } ~ ( > ~ v _ { \mathrm { s l o w e s t } } )$ . To avoid excessively strict deadlines, we incorporate the average vehicle speed $v _ { \mathrm { s a v } }$ and a lower-bound adjustment $N$ , defining the premium deadline as: $\begin{array} { r } { t _ { \mathrm { d e a d l i n e } } = \operatorname* { m a x } \left( \frac { d } { v _ { \mathrm { p r e m } } } , \frac { d } { v _ { \mathrm { s a v } } } + N \right) } \end{array}$ . Here, $v _ { \mathrm { s l o w e s t } }$ and $v _ { \mathrm { p r e m } }$ function as slack parameters rather than physical speeds. Increasing $v _ { \mathrm { p r e m } }$ strengthens priority by treating demands with higher urgency, though excessively strict deadlines may increase cancellations.

## 4.2 Dedicated-Fleet

Dedicated-Fleet extends Shorter-Deadline by dedicating part of the fleet to premium users. In this method, two types of vehicles are prepared: dedicated vehicles that can be used only by premium users, and shared vehicles that can be used by both premium and non-premium users. Premium demands are dispatched using both vehicle groups, whereas non-premium demands are dispatched using only shared vehicles. The deadline setting for premium users is the same as in Shorter-Deadline. Dedicated vehicles can assign premium users to less crowded vehicles, shortening their travel time and creating a clearer diference from nonpremium users. At the same time, non-premium users can use fewer vehicles than in the baseline or Shorter-Deadline.

## 5 Experimental Settings

This section presents the experimental settings. We first describe the map, fleet, and simulation horizon, then explain the demand scenarios, and finally define the evaluation metrics.

## 5.1 Map, Fleet, and Simulation Horizon

The experiments use a synthetic 15 × 15 grid map shown in Fig. 3. The node interval is 400 m, and the entire area is 5.6 km × 5.6 km. An urban area of 1.6 km × 1.6 km is defined at the center of the map. The simulator uses two zones: entire area, which covers the whole map, and urban area, which covers the central area.

![](images/fd0b057315103dbf2d0a2c190e05c5364541ed488b794a41fa4f27fe35887f4b.jpg)  
Fig. 3. Synthetic grid map used in the experiments.

We compare three methods: Baseline, Shorter-Deadline, and Dedicated-Fleet. In Baseline and Shorter-Deadline, all 12 vehicles are shared by all users. In Dedicated-Fleet, ⌊ratio × savN⌋ of the 12 vehicles are dedicated to premium users. Premium users are assigned randomly at a fixed ratio among all demands. Each condition is simulated with 30 random seeds, and methods are compared under the same condition and the same seed. For details, see Table 1.

Table 1. Fleet and parameter settings. All deadline terms are converted to seconds in the implementation.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Number of vehicles savN</td><td>12</td></tr><tr><td>Vehicle deployment interval</td><td>60 s</td></tr><tr><td>Average vehicle speed vsav</td><td>30 km/h</td></tr><tr><td>Baseline deadline slack parameter vslowest 0.1 m/s</td><td></td></tr><tr><td>Premium deadline slack parameter vprem</td><td>0.1, 0.3, 0.5, 0.7, 1.0 m/s</td></tr><tr><td>Lower-bound adjustment value N</td><td>600 s 10%,20%,30%,40%,50%</td></tr><tr><td>Premium ratio ratio Simulation time</td><td></td></tr><tr><td>Demand generation period</td><td>43,200 s</td></tr><tr><td>Peak period</td><td>1,800–37,800 s 16,200–23,400 s</td></tr><tr><td></td><td></td></tr><tr><td>Random seeds</td><td>30</td></tr></table>

## 5.2 Demand Scenarios

This study evaluates three demand scenarios. S1 (steady) has uniform demand over the entire area. S2 (time-varying) has uniform demand over the entire area and a higher demand rate during the peak period. S3 (urban-focused) has a center-focused demand structure and adds peak demand related to the urban area. Table 2 summarizes the scenarios.

Table 2. Summary of Demand Scenarios (S1–S3). OD denotes an origin–destination zone pair. Approximate rates result from combining OD-specific demand streams.
<table><tr><td>Scenario Phase</td><td></td><td>Distribution</td><td> $\mathbf { R e q . } / \mathbf { h }$ </td><td>OD Composition</td></tr><tr><td>S1</td><td></td><td>off-peak Uniform</td><td>150</td><td>All → All</td></tr><tr><td>S2</td><td></td><td>off-peak Uniform</td><td>150</td><td>All → All</td></tr><tr><td>S3</td><td>peak off-peak Center-focused</td><td>Uniform</td><td>300</td><td>All → All</td></tr><tr><td></td><td></td><td></td><td>~151</td><td>Center → Center (30%) Periph. → Center (30%)</td></tr><tr><td></td><td></td><td></td><td></td><td>Center → Periph. (20%)</td></tr><tr><td></td><td></td><td></td><td></td><td>Periph. → Periph. (20%)</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>peak</td><td>Center-heavy</td><td>~300</td><td>Center → Center (40%)</td></tr><tr><td></td><td></td><td></td><td></td><td>Periph. → Center (30%)</td></tr><tr><td></td><td></td><td></td><td></td><td>Center → Periph. (20%)</td></tr><tr><td></td><td></td><td></td><td></td><td>Periph. → Periph. (10%)</td></tr></table>

## 5.3 Evaluation Metrics

The main metric are TPD and cancel rate, as defined in Section 3.3. We evaluate premium users’ improvement and non-premium users’ degradation from the baseline, where positive values indicate a reduction in premium users’ TPD and an increase in non-premium users’ TPD. Additionally, we report the service gap δ, defined as $\delta = \bar { \tau } ^ { \bar { R } } - \bar { \tau } ^ { P }$

## 6 Results

This section presents the simulation results. We first examine δ. We then compare the average TPD for each scenario and method at $v _ { \mathrm { p r e m } } = 0 . 5$ , examine cancellations, and Finally, we compare premium user benefit and non-premium user degradation in the trade-of space.

## 6.1 Efect of Shorter-Deadline and Dedicated-Fleet

Fig. 4 shows the 30-seed distribution of δ for the scenarios with peak periods. The figure compares peak and of-peak periods and is used to examine whether Shorter-Deadline can create a diference between non-premium and premium users. The results show that Shorter-Deadline induces a disparity in TPD between the two user groups during peak periods. This means that deadline shortening works as a priority rule when the fleet is under pressure. In contrast, δ becomes small during of-peak periods. A likely reason is that the number of vehicles is suficient relative to demand, so many demands can be served almost one-to-one and ride-sharing occurs less often.

Fig. 5 shows the 30-seed distribution of δ in the steady without a peak period. The purpose is to examine whether dedicated premium vehicles can create a service gap in of-peak period. The results indicate that introducing dedicated vehicles contributes to increasing δ under of-peak conditions. This is diferent from Shorter-Deadline, whose efect becomes small when there is enough vehicle capacity. Dedicated-Fleet creates a service gap by restricting non-premium users to shared vehicles while allowing premium users access to the entire fleet.

![](images/15463fee6bb049ca8ab19581e4dfc97648877df895a2d4fe96f51f9708c9a996.jpg)

![](images/0c2a7fbfb680f632ab77cdd27a0bfb9f28e185a8e27958128bce4c4ce2652716.jpg)  
Fig. 4. Distribution of δ for Shorter-Deadline in scenarios with peak periods.

![](images/28b18621ab239a9e510fb762d937862198acc5c1142b5bdc65f221da770d2e05.jpg)  
Fig. 5. Distribution of δ for Dedicated-Fleet in the steady.

## 6.2 Average Time per Distance for the Two User Groups

Fig. 6 shows the mean TPD at $v _ { \mathrm { p r e m } } = 0 . 5$ . The figure reports the 30-seed mean and 95% confidence interval for premium and non-premium users under each scenario and method. In the steady scenario, diferences among methods are limited. In the time-varying and urban-focused scenarios, higher premium ratios reduce the remaining benefit for premium users and increase the burden on non-premium users. This efect is stronger in Dedicated-Fleet, because a larger premium ratio directly reallocates more vehicles away from non-premium users.

## 6.3 Cancellation Rate and Improvement-Rate Space

Fig. 7 shows the cancellation rate for each scenario, dispatch method, and user group. The figure illustrates the relationship between $v _ { \mathrm { p r e m } }$ and cancellations. In

![](images/6339658b4b95efedbdd83850a5f4be62365d9c5e7e21fafaf2257bb95719e0e0.jpg)

Fig. 6. Mean time per distance at $v _ { \mathrm { p r e m } } = 0 . 5 .$  
![](images/534b8c814b0f9f8f4606bead10eb44a780f39cd2ee69190eb7ca9ec67bf4c5f7.jpg)

Fig. 7. Cancellation rate as a function of $v _ { \mathrm { p r e m } }$ . The cancellation rate remained zero under all conditions in Steady.  
![](images/f5d12c1f90848e9abeec3e1df7004276bc268330d574f12b53acfced1a1a7a82.jpg)  
Fig. 8. Relationship between premium benefit and non-premium-user protection.Each polyline corresponds to one premium ratio, and points on a polyline correspond to diferent values of $v _ { \mathrm { p r e m } }$

Shorter-Deadline, a larger $v _ { \mathrm { p r e m } }$ makes the deadline stricter. As a result, more demands cannot be served before their shortened deadline, and the cancellation rate tends to increase. In contrast, Dedicated-Fleet tends to keep the cancellation rate lower because dedicated vehicles absorb some strict premium demands.

Fig 8 shows the trade-of space between premium users’ improvement rate and non-premium users’ degradation rate from the baseline for each scenario, method, and $v _ { \mathrm { p r e m } }$ . Points are connected by $v _ { \mathrm { p r e m } }$ . Moving upward means that premium users receive a larger benefit, whereas moving rightward indicates greater degradation for non-premium users. In Shorter-Deadline, a U-turn pattern appears in the trade-of space under strict deadline settings such as $v _ { \mathrm { p r e m } } =$ 0.7 and 1.0. The triangle series sometimes shows 50% cases with a smaller apparent non-premium burden than 30% cases. Since TPD is computed only for completed trips, excessive cancellations can make the remaining trips appear more favorable. In Dedicated-Fleet, dedicated vehicles absorb some strict premium demands, keeping cancellations lower and preventing this U-turn pattern. The figure also shows a clear trade-of between premium user benefit and non-premium user impact. Among the tested points, limiting the increase in non-premium users’ TPD to 5% allows a maximum reduction of approximately 20% in premium users’ time per distance. When the acceptable deterioration for non-premium users is relaxed to 15%, the maximum premium-user improvement among the tested points increases to approximately 30%. Thus, rather than identifying a single universally optimal setting, the results suggest that the appropriate dispatch policy should be selected according to the operator’s acceptable level of non-premium service degradation.

## 7 Discussion

These results indicate that the proposed methods should be interpreted as diferent control policies whose suitability depends on operating conditions. Shorter-Deadline preserves access to the full fleet for non-premium users and is useful when the fleet is under pressure. However, its efect becomes limited when capacity is suficient. Dedicated-Fleet can create a clearer service gap under of-peak conditions, but it can also reduce the efective capacity available to non-premium users under congestion. This pattern appears across demand scenarios despite diferences in spatial demand structure.

The results suggest that a fixed dedicated fleet is too rigid under congestion. When the system is congested, premium priority may be better implemented by deadline tightening through Shorter-Deadline, while dedicated vehicles are more suitable under of-peak conditions. Thus, the results should be interpreted as guidance for choosing dispatch rules that create service diferentiation while accounting for degradation among non-premium users.

## 8 Conclusion

We evaluated premium membership in on-demand taxi-sharing using two dispatch mechanisms: Shorter-Deadline, which shortens the deadline for premium users in the sequential optimal insertion method, and Dedicated-Fleet, which reserves part of the fleet for premium users while applying the same premium deadline rule.

The results do not imply that premium membership is universally beneficial. Rather, they show that the suitable priority mechanism depends on operating conditions: Shorter-Deadline is useful under congestion, whereas Dedicated-Fleet is more suitable under of-peak conditions. Several open issues remain, including pricing, hybrid dispatch policies, non-random premium demand, additional demand scenarios, user-facing verification of priority, and validation under diferent map and fleet settings.

## Acknowledgement

This work was supported by JST-CREST Program Grant Number JPMJCR2564, Japan.

## References

1. Alonso-Mora, J., Samaranayake, S., Wallar, A., Frazzoli, E., Rus, D.: Ondemand high-capacity ride-sharing via dynamic trip-vehicle assignment. Proceedings of the National Academy of Sciences 114(3), 462–467 (2017). https://doi.org/10.1073/pnas.1611675114

2. Beirigo, B.A., Negenborn, R.R., Alonso-Mora, J., Schulte, F.: A business class for autonomous mobility-on-demand: Modeling service quality contracts in dynamic ridesharing systems. Transportation Research Part C: Emerging Technologies 136, 103520 (2022). https://doi.org/10.1016/j.trc.2021.103520

3. Chen, Y., Jiao, J.: Are there transit deserts in europe? a study focusing on four european cases through publicly available data. Sustainability 14(20), 13182 (2022). https://doi.org/10.3390/su142013182

4. De Munck, T., Chevalier, P., Tancrez, J.S.: Managing priorities on on-demand service platforms with waiting time diferentiation. International Journal of Production Economics 266, 109053 (2023). https://doi.org/10.1016/j.ijpe.2023.109053

5. Fang, Z., Huang, L., Wierman, A.: Prices and subsidies in the sharing economy. Performance Evaluation 136, 102037 (2019). https://doi.org/10.1016/j.peva.2019.102037

6. Hansen, T., Sener, I.: Strangers on this road we are on: A literature review of pooling in on-demand mobility services. Transportation Research Record: Journal of the Transportation Research Board 2677, 036119812211238 (09 2022). https://doi.org/10.1177/03611981221123801

7. International Transport Forum: Regulating app-based mobility services: Summary and conclusions. Tech. Rep. 175, OECD Publishing, Paris (2019). https://doi.org/10.1787/94d27a3a-en

8. Jiao, J., Dillivan, M.: Transit deserts: The gap between demand and supply. Journal of Public Transportation 16(3), 23–39 (2013). https://doi.org/10.5038/2375- 0901.16.3.2

9. Ma, L., Tao, Z., Wei, Q.: On-demand ride service platform with diferentiated services. PLOS ONE 19, 1–27 (01 2024). https://doi.org/10.1371/journal.pone.0296732, https://doi.org/10.1371/ journal.pone.0296732

10. Noda, I., Ohta, M., Shinoda, K., Kumada, Y., Nakashima, H.: Evaluation of usability of dial-a-ride systems by social simulation. In: International Workshop on Multi-Agent Systems and Agent-Based Simulation. pp. 167–181. Springer (2003)

11. de Ruijter, A., Cats, O., Alonso-Mora, J., Hoogendoorn, S.: Ride-pooling matching with a compensatory cost function: Implications for adoption, eficiency and level of service (2021), https://arxiv.org/abs/2107.06710

12. Santi, P., Resta, G., Szell, M., Sobolevsky, S., Strogatz, S.H., Ratti, C.: Quantifying the benefits of vehicle pooling with shareability networks. Proceedings of the National Academy of Sciences 111(37), 13290–13294 (2014). https://doi.org/10.1073/pnas.1403657111

13. Soza-Parra, J., Kucharski, R., Cats, O.: The shareability potential of ride-pooling under alternative spatial demand patterns. Transportmetrica A: Transport Science 20(2), 2140022 (2024). https://doi.org/10.1080/23249935.2022.2140022

14. Wang, X., Zhang, R.: Carpool services for ride-sharing platforms: Price and welfare implications. Naval Research Logistics (NRL) 69 (10 2021). https://doi.org/10.1002/nav.22030