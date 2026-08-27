# Choose Your Game Wisely: Measuring Game-Theoretic Structures in

Real-World Vehicle Interactions

Yueyuan Li, Rongcheng Nie, Weijie Xi, Mingyang Jiang, Songan Zhang, Hanyang Zhuang, and Ming Yang

Abstract—Game-theoretic models provide principled frameworks for modeling vehicle interactions, but their underlying temporal assumptions have not been systematically examined against real-world driving behavior. In particular, it remains unclear how simultaneous, sequential, and asymmetric interaction structures can be measured from vehicle trajectories. This paper develops a trajectory-based interaction measurement framework to identify interaction events and quantify behavioral change onset, temporal organization, post-onset response dynamics, and ordering stability. The framework uses behavioral deviations to verify candidate interactions. We evaluate the framework on six real-world trajectory datasets, including INTERACTION, highD, inD, rounD, Waymo Open Motion, and nuPlan, covering diverse road geometries, traffic environments, and interaction types. The results show that concurrent and sequential behavioral changes both constitute substantial proportions of observed following, merging, and conflicting interactions. Among sequential interactions, stable ordering is more prevalent than alternating ordering, indicating that persistent asymmetric roles are a common interaction structure. Importantly, temporal precedence does not necessarily coincide with a measurable behavioral response, indicating that temporal ordering alone may not be sufficient to characterize behavioral dependence. These findings show that real-world interactions exhibit concurrent, sequential, and persistently ordered temporal structures. Different game-theoretic formulations are therefore better regarded as complementary modeling abstractions for different interaction regimes rather than as a universal structure governing all vehicle interactions.

Index Terms—Driving behavior; Game theory; Vehicle interaction; Trajectory-based measurement; Behavioral measurement.

## I. INTRODUCTION

Vehicle behavior in complex traffic is often shaped by surrounding vehicles rather than determined independently [1]. Such interactions are particularly evident in highways, ramps, intersections, and roundabouts [2, 3], where a behavioral change by one vehicle may induce a subsequent response from another. Real-world trajectory data provide direct measurements of vehicle motion and have been widely used to identify interacting vehicles [4], characterize interaction patterns [3, 5], and model or predict vehicle behavior [6]. However, existing studies primarily detect or model observed behavior, while the temporal and directional organization of behavioral responses remains less systematically measured. In particular, proximity or temporal precedence alone does not establish whether an observed change constitutes a measurable behavioral response to another vehicle.

Game theory provides explicit models of interactive decision making by specifying how interacting vehicles make decisions. Both cooperative and non-cooperative formulations have been considered in driving applications [7, 8], with Nash and Stackelberg formulations among the commonly used approaches [9, 10]. Nash formulations typically assume simultaneous decisions, whereas Stackelberg formulations impose asymmetric leader–follower roles [11, 12]. These formulations therefore imply distinct temporal and directional structures in observable behavior. Whether real-world interactions exhibit these structures, however, has not been systematically measured.

To address this gap, we develop a trajectory-based interaction measurement framework that identifies interaction events and quantifies behavioral change onset, temporal organization, response dynamics, and ordering stability. Interaction directionality captures which vehicle changes its behavior first and whether a subsequent measurable response occurs, while temporal organization and ordering stability characterize the timing and persistence of these roles. We then use these measurements to assess which game-theoretic temporal structures best characterize different observed interaction regimes.

Rather than proposing another interaction model, this study establishes an empirical basis for choosing between alternative temporal interaction structures when modeling real-world driving. The contributions of this paper are as follows:

• An interaction measurement framework that separates behavioral onset, temporal organization, response dynamics, and ordering stability, enabling interaction structures to be quantified directly from real-world trajectories.

• A cross-dataset characterization showing that concurrent and sequential behavioral changes coexist across following, merging, and conflicting interactions, while sequential interactions are predominantly characterized by stable rather than alternating ordering.

• An empirical assessment showing that concurrent, sequential, and persistently ordered interaction structures coexist in real-world driving, supporting a complementary view of simultaneous-move, sequential-move, and leader– follower formulations across interaction regimes.

## II. RELATED WORKS

## A. Interaction Behavior Analysis

The availability of large-scale real-world trajectory datasets has enabled vehicle interactions to be studied directly from observed driving behavior. Datasets such as INTERACTION, nu-Plan, and the Waymo Open Motion Dataset (WOMD) contain numerous interactive traffic scenarios and have been widely used for interaction analysis and motion prediction [13–15]. Vehicle interaction has also been studied from behavioral, cognitive, optimization-based, learning-based, and game-theoretic perspectives [1, 16]. For this study, trajectory-based analysis is especially relevant because it provides direct observations of how interacting vehicles adjust their behavior over time.

A number of studies have used trajectory data to identify interaction patterns. Zhang and Wang [17] extracted vehicleto-vehicle interaction patterns at signalized intersections using driving primitives and unsupervised clustering. Zhang et al. [3] considered how interaction patterns evolve during mandatory and discretionary lane changes using graph structures and hidden Markov models. Jiang et al. [4] developed InterHub to extract and organize dense interaction events from large-scale driving datasets. Other work has focused more directly on identifying which surrounding vehicles are behaviorally relevant. Hazard et al. [18] ranked surrounding agents according to their effect on the ego vehicle’s planned trajectory, while Wang et al. [19] introduced a Vehicle Interaction Level to quantify how strongly the behavior of one vehicle is constrained by another. These studies move beyond detecting interaction events and provide measures of interaction relevance or strength.

Interaction has also been modeled explicitly in trajectory prediction. Some early methods incorporated semantic interaction features or graph-based relations between surrounding vehicles [20, 21]. Lee et al. [22] jointly predicted trajectories and pairwise interaction modes, and Kumar et al. [23] represented temporal interaction types as edges in a hybrid traffic graph. M2I [24] modeled directed dependence by dividing interacting agents into influencer–reactor pairs and conditioning the reactor’s prediction on the influencer. Recent work continues to introduce more explicit forms of vehicle dependence into motion prediction [6].

Overall, trajectory-based studies have become increasingly detailed in describing whether vehicles interact, which vehicles are involved, and how strongly their behaviors are related. In most cases, however, these descriptions are introduced to support interaction identification or trajectory prediction. How the behavioral dependence between two vehicles develops within an interaction episode is less often examined directly from observed trajectories.

## B. Interaction Modeling Based on Game Theory

Game theory has been widely used to model interactive driving because it explicitly accounts for the interdependence between vehicle decisions. In a game-theoretic formulation, each vehicle is typically associated with an objective that reflects factors such as safety, efficiency, and comfort [10]. The objective may also incorporate the outcomes of other road users, allowing different social preferences, ranging from predominantly self-interested behavior to more cooperative behavior [25].

Game-theoretic models differ not only in vehicle objectives but also in how the interaction between vehicles is structured. Nash formulations generally model mutually dependent decisions within the same decision stage [10, 26]. Stackelberg formulations instead introduce an asymmetric and sequential relationship, in which one vehicle is treated as the leader and another responds as the follower [27, 28]. Hierarchical formulations introduce additional reasoning levels or decision stages to describe more involved interactions [29, 30]. Nash and Stackelberg formulations have also been considered within the same driving decision framework to examine how different game structures affect decision making [28, 31].

Some game-based methods allow parts of the interaction to change as more observations become available. Zhang et al. [32], for example, estimated driver aggressiveness online within a Stackelberg-based controller for mandatory lane changing. Li et al. [33] used Bayesian inference together with cognitive hierarchy theory to update the ego vehicle’s strategy according to the inferred behavior of another driver. Liu et al. [34] treated pairwise leader–follower relationships as uncertain in a partially observable game and estimated them from observed trajectories during forced merging. In these models, driver type, intention, or role does not have to remain fixed throughout the interaction, although such adaptation is still performed within a specified game formulation.

These formulations have been applied to a variety of driving scenarios, including lane changing, merging, intersections, roundabouts, and racing [35]. Although the specific assumptions differ, the interaction structure is generally specified as part of the model formulation [36], while real-world or simulated driving data are mainly used to calibrate model parameters, reproduce observed behavior, or evaluate model performance [27, 28]. Consequently, empirical evidence is often used to assess behavior generated under a predefined interaction structure rather than to identify that structure directly from observed interactions.

Taken together, existing studies either represent interaction for behavioral analysis and prediction or prescribe its structure within a behavioral model. This leaves the structure observed in real driving behavior less well understood.

## III. METHOD

## A. Preliminary

Before introducing the interaction measurement framework, we review the observable temporal implications of assumptions underlying commonly used game-theoretic models for vehicle interaction.

1) Simultaneous-move assumption: In simultaneous-move games, players select their actions within the same decision stage without observing the current actions of other players. The Nash equilibrium provides a classical solution concept for such games, where each player’s strategy is optimal given the strategies selected by others [11]. In vehicle interactions, this corresponds to concurrent behavioral changes without a measurable temporal precedence between the vehicles.

![](images/e8ccdc4f3dab5e7f291b9bad8c4ccea977f48b0ce8d2a1241bd012b6392820e7.jpg)  
Fig. 1. Examples of the three interaction types: following, merging, and conflicting, from the INTERACTION dataset [13].

2) Sequential-move assumption: Sequential-move games describe interactions in which players make decisions at different stages, creating an ordered decision process where the action of one player precedes and can be observed before the subsequent decision of another player [37]. In vehicle interactions, this corresponds to a temporally ordered sequence of observable behavioral changes.

3) Leader–follower structure: Stackelberg games represent a sequential interaction with asymmetric leader–follower roles, where the follower responds after observing the leader’s action [27]. In vehicle interactions, this structure implies persistent asymmetric ordering, which can be empirically examined from observed behavioral changes.

## B. Interaction Event Extraction

Following the concept of interaction in [38], we consider an interaction as a situation in which two vehicles intend to occupy the same spatial region at the same time in the near future. Such interactions may be expressed through changes in vehicle motion as well as non-kinematic cues such as vehicle lights, gaze, or gestures. In this study, we focus on interactions observable from vehicle trajectories.

We categorize trajectory-level relations into three interaction types: following, merging, and conflicting, consistent with common interaction patterns considered in previous studies [1, 3, 17]. Following (F) describes vehicles traveling in the same direction on the same lane. Merging (M) describes vehicles traveling in the same direction on different lanes, with one vehicle moving toward the other vehicle’s lane. Conflicting (C) describes vehicles whose future paths intersect or converge from different directions, including both intersecting and opposing-path configurations. Figure 1 illustrates representative examples of these interaction types in different traffic scenarios, including highways, urban roads, and roundabouts.

For potential interaction detection, each vehicle is assigned a reference trajectory that maintains its current speed and follows its lane route toward the observed destination. The observed and reference trajectories are converted into vehicleoccupancy-aware motion corridors. Potential following, merging, and conflicting interactions are detected from road topology, spatial compatibility, and temporal compatibility [3, 17, 19, 21, 23, 24], and frame-level detections are aggregated into candidate interaction events.

Spatiotemporal compatibility alone, however, does not establish a behavioral response. We therefore use speed deviation as additional behavioral evidence. For each candidate event window W, we construct a constant-velocity reference by holding vehicle i’s speed at the beginning of W while following its lane route, denoted by $v _ { i } ^ { w } ( t )$ , and quantify the deviation of the observed speed $v _ { i } ^ { \mathrm { o b s } } ( t )$ from this reference within the window. Because vehicles differ in typical speed variability, we normalize this deviation against a vehiclespecific empirical baseline: for control windows in which vehicle i is not involved in any candidate relation, we compute the same deviation and take its mean $\mu _ { i }$ and standard deviation $\sigma _ { i } .$ . The normalized deviation is

$$
z _ { i } = \frac { \mathrm { m e d i a n } _ { t \in W } \left| v _ { i } ^ { \mathrm { o b s } } ( t ) - v _ { i } ^ { w } ( t ) \right| - \mu _ { i } } { \sigma _ { i } + \epsilon _ { b } } ,\tag{1}
$$

where $\epsilon _ { b } = 0 . 5$ m/s is a stabilization term. A candidate pair (i, j) is retained if at least one participant exhibits a sufficiently large normalized deviation:

$$
\operatorname* { m a x } ( z _ { i } , z _ { j } ) > \tau ,\tag{2}
$$

where $\tau = 2 . 0$ is the deviation threshold. This criterion retains asymmetric cases in which only one participant exhibits a marked behavioral adjustment.

For merging and conflicting candidates, we require the vehicles to encounter the shared spatial region within a temporal gap of at most 1.0 s, preventing temporally unrelated behaviors from being associated with the same event.

## C. Temporal Characterization of Interaction

1) Behavioral Change Onset: We first identify behavioral change onset relative to a map-constrained reference trajectory. This reference trajectory assumes that the vehicle follows its lane route toward the observed destination while accounting for road geometry and planned deceleration, such as curvature and stop-line constraints. For vehicle i, the speed residual at time t is defined as

$$
r _ { i } ( t ) = v _ { i } ^ { \mathrm { o b s } } ( t ) - v _ { i } ^ { \mathrm { r e f } } ( t ) ,\tag{3}
$$

where $v _ { i } ^ { \mathrm { o b s } } ( t )$ and $v _ { i } ^ { \mathrm { r e f } } ( t )$ denote the observed and mapconstrained reference speeds, respectively. To suppress trajectory noise, a behavioral change must exceed a residual

threshold continuously for a minimum duration. The behavioral change onset is defined as

$$
t _ { i } ^ { \mathrm { o n s e t } } = \operatorname* { i n f } \left\{ t : | r _ { i } ( t ^ { \prime } ) | > \delta , \forall t ^ { \prime } \in [ t , t + d _ { \operatorname* { m i n } } ) \right\} ,\tag{4}
$$

where the residual threshold is set to $\delta = 0 . 3$ m/s and the minimum persistence duration is set to $d _ { \operatorname* { m i n } } = 4 0 0$ ms.

2) Temporal Organization: We use the relative timing of behavioral change onsets to characterize whether the two participants exhibit temporally aligned or temporally ordered behavioral changes. For an interaction event involving vehicles i and j, the onset difference is defined as

$$
\Delta t _ { i j } = t _ { i } ^ { \mathrm { o n s e t } } - t _ { j } ^ { \mathrm { o n s e t } } .\tag{5}
$$

When both onsets are detected, an event is classified as concurrent if $| \Delta t _ { i j } | ~ \le ~ \tau _ { \mathrm { t o l } }$ and as sequential otherwise, where $\tau _ { \mathrm { t o l } } = 4 0 0$ ms. Thus, concurrent denotes no measurable temporal precedence at this tolerance. For sequential events, the sign of $\Delta t _ { i j }$ indicates which vehicle exhibits an observable behavioral change first. An event is classified as one-sided when only one onset is detected and as unresolved when neither onset is detected. All proportions use the full set of verified interaction events; one-sided and unresolved events are therefore counted as non-concurrent.

3) Role Stability: The first-onset comparison captures only the initial temporal relationship between the two vehicles. We therefore examine the complete sequence of detected behavioral changes within each interaction.

Events whose detected changes fall within the temporal tolerance are classified as concurrent; otherwise, changes are ordered chronologically according to the corresponding vehicle. Events are classified as stable-ordering when fewer than two ordering reversals occur, and as alternating when two or more reversals are observed. Events with changes detected for only one vehicle are classified as one-sided, while events without detectable changes are classified as unresolved. The sequence-level concurrent class is distinct from the firstonset classification above; all temporal-organization statistics and the GLMM use the sequence-level classification.

4) Post-onset Response Timing: For each directed pair $i $ $j ,$ we examine whether vehicle j exhibits a new behavioral change after the onset of vehicle i. The response threshold for the target vehicle is defined as

$$
\theta _ { j } = \mathrm { m a x } \left( \delta , \mu _ { j , \mathrm { p r e } } + k \sigma _ { j , \mathrm { p r e } } \right) ,\tag{6}
$$

where $\mu _ { j , \mathrm { p r e } }$ and $\sigma _ { j , \mathrm { p r e } }$ are the mean and standard deviation of $| r _ { j } ( t )$ | before the source change, respectively, and $k = 2$

Each directed pair is classified as response, preactive, noresponse, or right-censored. Directions without a valid source onset or with fewer than three pre-onset baseline frames are recorded separately. A response is detected when the target exhibits a sustained residual above $\theta _ { j }$ after the source change, whereas a preactive target has already exhibited behavioral change before the source onset.

Let $t _ { j } ^ { \mathrm { r e s p o n s e } }$ denote the first sustained post-onset change of vehicle $j$ satisfying the response criterion. For response directions, the response lag is

$$
\Delta t _ { i  j } = t _ { j } ^ { \mathrm { r e s p o n s e } } - t _ { i } ^ { \mathrm { o n s e t } } .\tag{7}
$$

Response-lag statistics are computed only over directions classified as response. The post-onset search window is limited to 2000 ms; directions without a qualifying response within this window are treated according to the corresponding noresponse or right-censored criteria.

## D. Statistical Analysis

1) Cluster-aware Uncertainty Estimation: Confidence intervals are estimated using scene-level cluster bootstrap resampling. For each bootstrap replicate, scenes are sampled with replacement and all interaction events belonging to the sampled scenes are retained. We use 2,000 bootstrap replicates and report percentile-based 95% confidence intervals.

2) Interaction-type Heterogeneity: To compare temporal organization across interaction types while accounting for eventand scene-level heterogeneity, we fit a logistic mixed-effects model of concurrent organization on interaction type, event duration, and number of vehicles, with a scene-level random intercept. Interaction-type effects are reported as adjusted odds ratios with 95% confidence intervals.

## IV. EXPERIMENTS

## A. Dataset and Experimental Setup

We evaluate the temporal structures associated with gametheoretic assumptions against six widely used trajectory datasets: INTERACTION (abbreviated as INT. in tables) [13], highD [39], inD [40], rounD [41], the Waymo Open Motion Dataset (WOMD) [15], and nuPlan [14]. These datasets provide substantial variation in data sources, road geometries, traffic densities, and driving environments, covering scenarios such as intersections, roundabouts, highways, and urban streets. For all datasets, interaction events are extracted following the method in Sec. III-B. Raw trajectories are processed using Tactics2D-based parsers where supported, together with dataset-specific adapters for Waymo and nuPlan [42]. Datasetspecific statistics of the interaction events are summarized in Table I. The number of extracted events is lower than that reported by InterHub [4], as our definition additionally requires measurable behavioral change from at least one interacting vehicle.

TABLE I  
EXTRACTED INTERACTION EVENTS FROM DIFFERENT DATASETS.
<table><tr><td>Dataset</td><td># Event</td><td>#F</td><td>#M</td><td># C</td><td>Med. Duration (s)</td></tr><tr><td>INT.</td><td>765</td><td>104</td><td>225</td><td>436</td><td>2.30</td></tr><tr><td>highD</td><td>1,387</td><td>1,324</td><td>63</td><td>0</td><td>3.48</td></tr><tr><td>inD</td><td>225</td><td>12</td><td>92</td><td>121</td><td>2.44</td></tr><tr><td>rounD</td><td>1,371</td><td>207</td><td>258</td><td>906</td><td>2.24</td></tr><tr><td>Waymo</td><td>5,565</td><td>3,078</td><td>1,996</td><td>491</td><td>2.70</td></tr><tr><td>nuPlan</td><td>2,566</td><td>1,588</td><td>662</td><td>316</td><td>2.65</td></tr></table>

## B. Temporal Organization of Vehicle Interactions

Using the complete sequence of detected behavioral changes described in Sec. III-C, each event is classified as concurrent, sequential, or one-sided. Concurrent events have all behavioral changes within the 400 ms tolerance, whereas sequential events show a temporal ordering between the two vehicles. The results are shown in Table II.

TABLE II  
TEMPORAL ORGANIZATION OF VEHICLE INTERACTIONS ACROSS DATASETS AND INTERACTION TYPES.
<table><tr><td>Dataset</td><td>Interaction type</td><td>Concurrent (%)</td><td>Sequential (%)</td><td>One-sided (%)</td></tr><tr><td rowspan="3">INT.</td><td>F</td><td>48.08</td><td>46.15</td><td>5.77</td></tr><tr><td>M</td><td>58.67</td><td>37.33</td><td>4.00</td></tr><tr><td>C</td><td>43.98</td><td>48.61</td><td>7.18</td></tr><tr><td rowspan="2">highD</td><td>F</td><td>50.68</td><td>42.15</td><td>7.18</td></tr><tr><td>M</td><td>44.44</td><td>47.62</td><td>7.94</td></tr><tr><td rowspan="3">inD</td><td>F</td><td>45.45</td><td>54.55</td><td>0.00</td></tr><tr><td>M</td><td>54.26</td><td>44.68</td><td>1.06</td></tr><tr><td>C</td><td>46.15</td><td>50.43</td><td>3.42</td></tr><tr><td rowspan="3">rounD</td><td>F</td><td>49.04</td><td>47.60</td><td>3.37</td></tr><tr><td>M</td><td>50.78</td><td>44.57</td><td>4.65</td></tr><tr><td>C</td><td>33.81</td><td>52.77</td><td>13.30</td></tr><tr><td rowspan="3">Waymo</td><td>F</td><td>48.92</td><td>43.93</td><td>6.66</td></tr><tr><td>M</td><td>52.49</td><td>41.06</td><td>6.20</td></tr><tr><td>C</td><td>52.25</td><td>40.57</td><td>6.15</td></tr><tr><td rowspan="3">nuPlan</td><td>F</td><td>52.09</td><td>40.96</td><td>6.83</td></tr><tr><td>M</td><td>57.51</td><td>33.69</td><td>8.80</td></tr><tr><td>C</td><td>55.41</td><td>34.08</td><td>10.51</td></tr></table>

Concurrent events are common across datasets and interaction types, but they do not consistently account for the majority of interactions. Across the reported cases, the concurrent proportion ranges from 33.81% to 58.67%. Sequential events are also frequent, accounting for 33.69%–54.55% of events, and are more frequent than concurrent events in, for example, InD following and RounD conflicting interactions. One-sided events are relatively rare, reaching at most 13.30% for conflicting interactions in RounD.

Figure 2 (a) shows the distribution of temporal organization across interaction types. The corresponding GLMM results are shown in Fig. 2 (b), where the reported odds ratios are adjusted for event duration, the number of vehicles, and scene-level heterogeneity. Together, these results show that the temporal organization of vehicle interactions varies across interaction types, with both concurrent and sequential changes occurring frequently.

## C. Response Dynamics and Interaction Directionality

For each temporally ordered pair, the vehicle that changes first is treated as the source and the other as the target. A response is identified from the target’s post-onset behavioral change, with the response lag measured from the source onset to the target response onset.

Figure 3(a) shows the response probability among assessable directed pairs, defined as directions classified as either response or preactive. Directions with no valid source onset, insufficient pre-onset baseline frames, no response, or rightcensoring are excluded. The response probability is 40.0% in INTERACTION, 42.8% in InD, 47.5% in RounD, 47.6% in HighD, 27.1% in Waymo, and 43.2% in nuPlan. With equal weighting across datasets, the mean response probability is 41.4% (95% CI: 35.2–46.0%).

![](images/4af5fc8c1a2eca6e3b49c2669aae755c36c3b72e210d92588ac87dc644f93f81.jpg)

![](images/59345cd76d64564e1010b2b24d6d2f155a0d5cb2375c0539d6f82fcbfffbfa99.jpg)  
(a) Temporal organization  
(b) Adjusted odds ratios  
Fig. 2. Temporal organization of vehicle interactions and adjusted odds ratios across interaction types.

![](images/5c19706820a223ba78fa373441e76fcdd7bb942c67d826bfaf4babbdb8293b78.jpg)

![](images/62d71249b9fba4f811d4d2ba0d31983a24304315ca9a642a417e92a3b4f6ae4c.jpg)  
(a) Response probability among as- (b) Dataset-averaged empirical cumusessable directed pairs lative distribution of response lags  
Fig. 3. Response dynamics across datasets.

Figure 3 (b) shows the dataset-averaged empirical cumulative distribution function (ECDF) of response-lag. About 64% of detected responses occur within the 400 ms concurrent tolerance, ranging from 56.4% to 69.5% across datasets.

These results show that temporal precedence does not necessarily lead to a measurable behavioral response: only about 41% of assessable directed pairs exhibit a response. When a response does occur, however, it is often observed within the 400 ms temporal tolerance.

## D. Ordering Stability

We next examine whether a clear temporal ordering, once established, is maintained throughout the interaction. Restricting attention to the sequential events identified in Sec. IV-B, we ask whether the ordering is stable or alternating. The results are reported in Table III.

Stable ordering accounts for 80.46%–100.00% of sequential events across datasets and relation types, with alternating ordering accounting for the remainder. Stable ordering reaches 100.00% for following interactions in InD and exceeds 95% for following and merging interactions in HighD and for merging and conflicting interactions in RounD. Alternating ordering is most frequent in Waymo, accounting for 19.54%,

TABLE III
<table><tr><td>Dataset</td><td>Interaction type</td><td></td><td>Stable (%) Alternating (%)</td></tr><tr><td rowspan="3">INT.</td><td>F</td><td>87.50</td><td>12.50</td></tr><tr><td>M</td><td>86.90</td><td>13.10</td></tr><tr><td>C</td><td>87.14</td><td>12.86</td></tr><tr><td rowspan="2">highD</td><td>F</td><td>98.39</td><td>1.61</td></tr><tr><td>M</td><td>96.67</td><td>3.33</td></tr><tr><td rowspan="3">inD</td><td>F</td><td>100.00</td><td>0.00</td></tr><tr><td>M</td><td>80.95</td><td>19.05</td></tr><tr><td>C</td><td>84.75</td><td>15.25</td></tr><tr><td rowspan="3">rounD</td><td>F</td><td>91.92</td><td>8.08</td></tr><tr><td>M</td><td>95.65</td><td>4.35</td></tr><tr><td>C</td><td>95.80</td><td>4.20</td></tr><tr><td rowspan="3">Waymo</td><td>F</td><td>80.46</td><td>19.54</td></tr><tr><td>M</td><td>80.61</td><td>19.39</td></tr><tr><td>C</td><td>82.83</td><td>17.17</td></tr><tr><td rowspan="3">nuPlan</td><td>F</td><td>83.80</td><td>16.20</td></tr><tr><td>M</td><td>91.89</td><td>8.11</td></tr><tr><td>C</td><td>88.79</td><td>11.21</td></tr></table>

19.39%, and 17.17% of sequential following, merging, and conflicting interactions, respectively. Overall, temporal ordering is predominantly stable once a sequential interaction is established, while role alternation remains a minority pattern.

## E. Implications for Game-Theoretic Interaction Models

The empirical findings provide the following implications for the game-theoretic assumptions introduced in Sec. III-A.

1) Simultaneous-move game assumption: Concurrent behavioral changes are common across datasets and interaction types, indicating that the simultaneous-move structure underlying Nash formulations represents a meaningful class of vehicle interactions.

2) Sequential-move game assumption: Sequential behavioral changes are also prevalent, with their proportion sometimes exceeding that of concurrent interactions. This finding shows that sequential-move formulations are likewise relevant to observed vehicle interactions.

3) Leader–follower structure: Among sequential interactions, stable ordering is substantially more common than alternating ordering across datasets. This suggests that persistent asymmetric ordering is a meaningful structure within sequential interactions, supporting the use of leader–follower formulations.

Overall, the results suggest a mixture of interaction structures rather than a single universal temporal organization. Concurrent, sequential, and stably ordered behavioral patterns are all observed across different interaction contexts and datasets, indicating that different game-theoretic formulations provide complementary descriptions of real-world vehicle interactions.

## V. CONCLUSION

This work presents a measurement framework for examining the temporal interaction structures associated with gametheoretic formulations using real-world vehicle trajectories. Across six real-world trajectory datasets, vehicle interactions show substantial variation in temporal organization. Concurrent and sequential behavioral changes both occur across following, merging, and conflicting scenarios, while sequential interactions are more often characterized by stable than alternating ordering. These results indicate that real-world driving does not follow a single temporal interaction pattern. Concurrent behavioral changes support simultaneous-move temporal formulations, whereas sequential interactions with stable ordering support leader–follower temporal formulations. The same geometric relation can exhibit different temporal organizations across datasets, suggesting that interaction geometry alone is not sufficient to determine the appropriate interaction structure. The response analysis also shows that temporal precedence does not necessarily imply a measurable behavioral response. Temporal ordering and behavioral dependence should therefore be considered separately when interpreting sequential interactions. This evidence concerns the temporal organization of observable behavior. It does not by itself establish vehicles’ decision-making processes, information sets, utility functions, strategies, or equilibrium mechanisms. Future interaction models should account for such variation in temporal organization and response behavior rather than assuming a single interaction mechanism.

## REFERENCES

[1] W. Wang, L. Wang, C. Zhang, C. Liu, and L. Sun, “Social interactions for autonomous driving: A review and perspectives,” Foundations and Trends® in Robotics, vol. 10, no. 3-4, pp. 198–377, 2022.

[2] M. S. Shirazi and B. T. Morris, “Looking at intersections: a survey of intersection monitoring, behavior and safety analysis of recent studies,” IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 1, pp. 4–24, 2016.

[3] Y. Zhang, Y. Zou, Y. Xie, and L. Chen, “Identifying dynamic interaction patterns in mandatory and discretionary lane changes using graph structure,” Computer-Aided Civil and Infrastructure Engineering, vol. 39, no. 5, pp. 638–655, 2024.

[4] X. Jiang, X. Zhao, Y. Liu, Z. Li, P. Hang, L. Xiong, and J. Sun, “A naturalistic trajectory dataset with dense interaction for autonomous driving,” Scientific Data, vol. 12, no. 1, p. 1084, 2025.

[5] A. Abarghooei and M. Ahmadi, “Driving behaviour classification and risk quantification via multi-sensor machine learning: From simulation to reality,” IEEE Transactions on Instrumentation and Measurement, 2026.

[6] J. Liang, C. Tan, L. Yan, J. Zhou, G. Yin, and K. Yang, “Interaction-aware trajectory prediction for safe motion planning in autonomous driving: A transformer-transfer learning approach,” IEEE Transactions on Intelligent Transportation Systems, 2025.

[7] P. Hang, C. Lv, C. Huang, Y. Xing, and Z. Hu, “Cooperative decision making of connected automated vehicles at multi-lane merging zone: A coalitional game approach,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 4, pp. 3829–3841, 2021.

[8] A. Liniger and J. Lygeros, “A noncooperative game approach to autonomous racing,” IEEE Transactions on Control Systems Technology, vol. 28, no. 3, pp. 884–897, 2019.

[9] A. Talebpour, H. S. Mahmassani, and S. H. Hamdar, “Modeling lane-changing behavior in a connected environment: A game theory approach,” Transportation Research Procedia, vol. 7, pp. 420–440, 2015.

[10] M. Wang, S. P. Hoogendoorn, W. Daamen, B. Van Arem, and R. Happee, “Game theoretic approach for predictive lane-changing and car-following control,” Transportation Research Part C: Emerging Technologies, vol. 58, pp. 73–92, 2015.

[11] J. F. Nash et al., “Equilibrium points in n-person games,” Proceedings of the national academy of sciences, vol. 36, no. 1, pp. 48–49, 1950.

[12] H. Von Stackelberg, Market structure and equilibrium. Springer Science & Business Media, 2010.

[13] W. Zhan, L. Sun, D. Wang, H. Shi, A. Clausse, M. Naumann, J. Kummerle, H. Konigshof, C. Stiller, A. de La Fortelle et al., “Interaction dataset: An international, adversarial and cooperative motion dataset in interactive driving scenarios with semantic maps,” arXiv preprint arXiv:1910.03088, 2019.

[14] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari, “nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles,” arXiv preprint arXiv:2106.11810, 2021.

[15] S. Ettinger, S. Cheng, B. Caine, C. Liu, H. Zhao, S. Pradhan, Y. Chai, B. Sapp, C. R. Qi, Y. Zhou et al., “Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 9710–9719.

[16] L. Crosato, K. Tian, H. P. Shum, E. S. Ho, Y. Wang, and C. Wei, “Social interaction-aware dynamical models and decision-making for autonomous vehicles,” Advanced Intelligent Systems, vol. 6, no. 3, p. 2300575, 2024.

[17] W. Zhang and W. Wang, “Learning v2v interactive driving patterns at signalized intersections,” Transportation Research Part C: Emerging Technologies, vol. 108, pp. 151–166, 2019.

[18] C. Hazard, A. Bhagat, B. R. Buddharaju, Z. Liu, Y. Shao, L. Lu, S. Omari, and H. Cui, “Importance is in your attention: agent importance prediction for autonomous driving,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2022, pp. 2531–2534.

[19] J. Wang, G. Lu, A. Yang, Z. Zhang, M. Liu, and J. Liang, “Quantitative identification of strong-interaction vehicles in autonomous driving,” IEEE Transactions on Intelligent Transportation Systems, 2026.

[20] J. Hong, B. Sapp, and J. Philbin, “Rules of the road: Predicting driving behavior with a convolutional model of semantic interactions,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019, pp. 8446–8454.

[21] X. Li, X. Ying, and M. C. Chuah, “Grip: Graph-based interaction-aware trajectory prediction,” in 2019 IEEE intelligent transportation systems conference (ITSC). IEEE, 2019, pp. 3960–3966.

[22] D. Lee, Y. Gu, J. Hoang, and M. Marchetti-Bowick, “Joint interaction and trajectory prediction for autonomous driving using graph neural networks,” arXiv preprint arXiv:1912.07882, 2019.

[23] S. Kumar, Y. Gu, J. Hoang, G. C. Haynes, and M. Marchetti-Bowick, “Interaction-based trajectory prediction over a hybrid traffic graph,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 5530–5535.

[24] Q. Sun, X. Huang, J. Gu, B. C. Williams, and H. Zhao, “M2i: From factored marginal trajectory prediction to interactive prediction,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 6533–6542.

[25] W. Schwarting, A. Pierson, J. Alonso-Mora, S. Karaman, and D. Rus, “Social behavior for autonomous vehicles,” Proceedings of the National Academy of Sciences, vol. 116, no. 50, pp. 24 972–24 978, 2019.

[26] E. L. Zhu and F. Borrelli, “A sequential quadratic programming approach to the solution of open-loop generalized nash equilibria,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 3211–3217.

[27] J. H. Yoo and R. Langari, “A stackelberg game theoretic driver model for merging,” in Dynamic Systems and Control Conference, vol. 56130. American Society of Mechanical Engineers, 2013, p. V002T30A003.

[28] P. Hang, C. Lv, Y. Xing, C. Huang, and Z. Hu, “Humanlike decision making for autonomous driving: A noncooperative game theoretic approach,” IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 4, pp. 2076–2087, 2020.

[29] J. F. Fisac, E. Bronstein, E. Stefansson, D. Sadigh, S. S. Sastry, and A. D. Dragan, “Hierarchical game-theoretic planning for autonomous vehicles,” in 2019 International conference on robotics and automation (ICRA). IEEE, 2019, pp. 9590–9596.

[30] P. Huang, H. Ding, Z. Sun, and H. Chen, “A gamebased hierarchical model for mandatory lane change of autonomous vehicles,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 9, pp. 11 256– 11 268, 2024.

[31] P. Hang, C. Huang, Z. Hu, and C. Lv, “Driving conflict resolution of autonomous vehicles at unsignalized intersections: A differential game approach,” IEEE/ASME Transactions on Mechatronics, vol. 27, no. 6, pp. 5136– 5146, 2022.

[32] Q. Zhang, R. Langari, H. E. Tseng, D. Filev, S. Szwabowski, and S. Coskun, “A game theoretic model predictive controller with aggressiveness estimation for mandatory lane change,” IEEE Transactions on Intelligent Vehicles, vol. 5, no. 1, pp. 75–89, 2019.

[33] S. Li, N. Li, A. Girard, and I. Kolmanovsky, “Decision making in dynamic and interactive environments based

on cognitive hierarchy theory, bayesian inference, and predictive control,” in 2019 IEEE 58th Conference on Decision and Control (CDC). IEEE, 2019, pp. 2181– 2187.

[34] K. Liu, N. Li, H. E. Tseng, I. Kolmanovsky, and A. Girard, “Interaction-aware trajectory prediction and planning for autonomous vehicles in forced merge scenarios,” IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 1, pp. 474–488, 2022.

[35] Z. Lin and Z. Tian, “Scenario-based decision-making using game theory for interactive autonomous driving: A survey,” arXiv preprint arXiv:2509.05777, 2025.

[36] X. Di, X. Chen, and E. Talley, “Liability design for autonomous vehicles and human-driven vehicles: A hierarchical game-theoretic approach,” Transportation research part C: emerging technologies, vol. 118, p. 102710, 2020.

[37] D. Fudenberg and J. Tirole, Game theory. MIT press, 1991.

[38] G. Markkula, R. Madigan, D. Nathanael, E. Portouli, Y. M. Lee, A. Dietrich, J. Billington, A. Schieben, and N. Merat, “Defining interactions: A conceptual framework for understanding interactive behaviour in human and automated road traffic,” Theoretical Issues in Ergonomics Science, vol. 21, no. 6, pp. 728–752, 2020.

[39] R. Krajewski, J. Bock, L. Kloeker, and L. Eckstein, “The highd dataset: A drone dataset of naturalistic vehicle trajectories on german highways for validation of highly automated driving systems,” in 2018 21st international conference on intelligent transportation systems (ITSC). IEEE, 2018, pp. 2118–2125.

[40] J. Bock, R. Krajewski, T. Moers, S. Runde, L. Vater, and L. Eckstein, “The ind dataset: A drone dataset of naturalistic road user trajectories at german intersections,” in 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020, pp. 1929–1934.

[41] R. Krajewski, T. Moers, J. Bock, L. Vater, and L. Eckstein, “The round dataset: A drone dataset of road user trajectories at roundabouts in germany,” in 2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2020, pp. 1–6.

[42] Y. Li, S. Zhang, M. Jiang, X. Chen, J. Yang, Y. Qian, C. Wang, and M. Yang, “Tactics2d: A highly modular and extensible simulator for driving decision-making,” IEEE Transactions on Intelligent Vehicles, vol. 9, no. 5, pp. 4840–4844, 2024.

![](images/f2db2852312c4fff3ab8b559ccdeb6b053df1c9c3334862b44f0055d6173d254.jpg)

Rongcheng NIE is currently pursuing a Bachelor’s degree in Automation with the School of Airspace Science and Engineering, Shandong University. His research interests include autonomous driving, robotics and embodied intelligence.

Weijie XI received a Bachelor’s degree in engineering from Jiangnan University, Jiangsu, China, in 2026. He is working towards a Ph.D. degree in Control Science and Engineering from Shanghai Jiao Tong University. His research interests include learning-based planning and control for intelligent vehicles and mobile robots.

![](images/ba2c44fbfc4827e0ac34f475b2701f9c837368b94f3e62d5df904c629ce51e47.jpg)

![](images/ea12142f345f56f8a2f094ccfd59de2947a1233b0c4710fc9f318b7689e2b84d.jpg)

Mingyang JIANG received a Bachelor’s degree in engineering from Shanghai Jiao Tong University in 2023, and a Master’s degree in Control Science and Engineering from Shanghai Jiao Tong University in 2026. His main research interests are end-to-end planning, driving decision-making, and reinforcement learning for autonomous vehicles.

![](images/ca89628f3a54676e1890b4db8085190161f682f6ea0f2bd8cd6804a91ddeed1a.jpg)

Songan ZHANG received B.S. and M.S. degrees in automotive engineering from Tsinghua University in 2013 and 2016, respectively. Then, she went to the University of Michigan, Ann Arbor, and received a Ph.D. in mechanical engineering in 2021. After graduation, she worked as a research scientist on the Robotics Research Team at Ford Motor Company. Presently, she is an assistant professor at the Global Institute of Future Technology (GIFT) at Shanghai Jiao Tong University. Her research interests include accelerated evaluation of autonomous vehicles, model-based reinforcement learning, and meta-reinforcement learning for autonomous vehicle decision-making.

![](images/ffddc6c64157c4782e86be9ff19c695e029c58d228d873393c7023a928cc81d9.jpg)

![](images/b323ce208852ba83626e76f28f80fd459eea4de63b07677ac9f046aa652c014b.jpg)  
Yueyuan LI received a Bachelor’s degree in Electrical and Computer Engineering from the University of Michigan-Shanghai Jiao Tong University Joint Institute in 2020. She is pursuing a Ph.D. degree in Control Science and Engineering from Shanghai Jiao Tong University. Her main fields of interest are the security of the autonomous driving system and driving decision-making. Her research activities include reinforcement learning, behavior modeling, and simulation.

Hanyang ZHUANG received the Ph.D. degree from Shanghai Jiao Tong University, Shanghai, China, in 2018. He has worked as a postdoctoral researcher at Shanghai Jiao Tong University from 2020 to 2022. He is currently an assistant research professor at Shanghai Jiao Tong University implementing research works related to intelligent vehicles. His research interest is in autonomous driving and cooperative driving systems.

![](images/861976c69ab70a040da118cc8328bfa673c0caaa6895b065fe680f31ffa1e6be.jpg)

Ming YANG received his Master’s and Ph.D. degrees from Tsinghua University, Beijing, China, in 1999 and 2003, respectively. Presently, he holds the position of Distinguished Professor at Shanghai Jiao Tong University, also serving as the Director of the Innovation Center of Intelligent Connected Vehicles. Dr. Yang has been engaged in the research of intelligent vehicles for more than 25 years.