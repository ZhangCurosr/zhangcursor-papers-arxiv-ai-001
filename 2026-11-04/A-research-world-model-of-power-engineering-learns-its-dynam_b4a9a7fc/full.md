# A research world model of power engineering learns its dynamics and reveals what-ifs

Fanping Sui<sup>1</sup>, Ying Zhang<sup>2,\*</sup> and Liwei Lin<sup>1</sup>

<sup>1</sup>Department of Mechanical Engineering, University of California, Berkeley, Berkeley, CA, USA <sup>2</sup>School of Electrical and Computer Engineering, Oklahoma State University, Stillwater, OK, USA Corresponding author. Email: y.zhang@okstate.edu

## Abstract

Decisions about the direction of research in power engineering determine which technologies mature and, through them, shape infrastructure that lasts decades, yet their consequences cannot be observed before they are taken, and existing forecasts of research fields extrapolate a single trajectory. Here we introduce research world models, which treat a research field as a dynamical system that can be acted upon, and establish one for power engineering from 1.1 million papers. A distilled language model reads each paper into role-typed concepts, two sophisticated, high-fidelity graph–temporal neural network cores learn how the resulting concept network evolves, and operators learned from natural experiments in the publication record estimate the efects of interventions. In temporally held-out tests, the model forecast years ahead which concepts would grow and which method–target pairs would be joined for the first time, outperforming tuned machine-learning baselines in most settings, and its estimated intervention efects tracked those realized in natural experiments it had not seen. Its training scheme and three of its operators replicated in two fields withheld from development. Questions posed in natural language are compiled by a domain-specialized 27-billion-parameter language model into executable actions and answered as paired simulations with and without the intervention. The answers were rated 4.1 of 5 for faithfulness by an independent language-model judge. By revealing the latent dynamics of a research field and its response to interventions, research world models ofer a new way to simulate and test research decisions before they are made.

## Introduction

Power engineering research supports infrastructure that is long-lived, capital-intensive and coupled to nearly every sector of the economy. The technologies and methods it develops today therefore shape the options of energy systems for decades<sup>1</sup>. Because such systems are strongly path-dependent, with early technological choices becoming locked in through investment and institutions <sup>2,3</sup>, misallocated research efort is costly and dificult to reverse. Even the technologies now central to the energy transition illustrate these timescales: photovoltaics and lithium-ion batteries required several decades of research and development to progress from laboratory demonstration to deployment at scale, and their costs declined only as cumulative production grew<sup>4,5</sup>. The research itself evolves through internal dynamics, in which attention concentrates on established topics and new problem–method combinations emerge only gradually <sup>6,7</sup>, and it is repeatedly redirected by general-purpose technologies, from electrification to semiconductors and, most recently, machine learning, each of which changed both the problems the field addresses and the methods it uses<sup>8,9</sup>. Shifts in policy priorities and funding act on the same trajectory <sup>10,11</sup>. Deciding on an intervention in such a field therefore requires both a forecast of its course and an estimate of how that course would change under the intervention. Yet this counterfactual is inaccessible to direct observation: each intervention is enacted only once, and the trajectory it forecloses is never realized, an instance of the fundamental problem of causal inference <sup>12</sup>. The instruments currently available, expert elicitation and bibliometric indicators, are retrospective by construction and silent on such counterfactuals <sup>13,14</sup>.

In machine learning, the problem of anticipating how a system evolves under alternative actions is the province of world models. A world model infers a latent state from incomplete observations and learns

$$
S _ { t + 1 } = T ( S _ { t } , a _ { t } ) ,\tag{1}
$$

where $S _ { t }$ is the latent state at step t and $a _ { t }$ an action<sup>15,16</sup>. Because forecasting and counterfactual evaluation draw on a single learned transition, the two questions are answered within one model. Learned simulators of this kind now outperform operational numerical systems in medium-range weather prediction<sup>17,18</sup> and have been extended to the Earth system<sup>19</sup> and to human physiology<sup>20</sup>.

A research field has a latent state of this kind: the problems it pursues and the methods brought to bear on them are never recorded as such, but are reflected, incompletely and with delay, in its publications. The science of science has forecast topic growth and the emergence of links between concepts <sup>21,22</sup>, and dynamic knowledge-graph pipelines have been proposed for the same purpose<sup>23</sup>, yet these approaches extrapolate a single trajectory: they do not learn a transition of the field under interventions and have not been validated against the observed efects of interventions. Such validation is dificult because concept growth is dominated by persistence and mean reversion, which makes naive baselines formidable, and because interventions are unlabeled, so that their efects are identifiable only from natural experiments <sup>24</sup>. Artifacts of the record, such as duplicated entries, can also be mistaken for genuine dynamics.

Here we introduce the research world model, a world model in which the state $S _ { t }$ of equation (1) is a research field, represented by its typed concept hypergraph and the latent embedding of that hypergraph, the transition $T$ is learned from the publication record, the actions $a _ { t }$ are research interventions, such as a burst of papers, a new term or the adoption of a method, and the answer to a what-if question is a pair of simulations with and without the action. To our knowledge, no earlier model of a research field has learned its transition under interventions. We build a research world model for power engineering, with semiconductors and materials science as replication fields. We test whether its latent state forecasts concept growth and new method–target links more accurately than tuned baselines, whether its training scheme and operators transfer to fields not used in development, and whether its operators recover held-out natural experiments. The experiments also establish how far each answer is identified: bursts of papers on first-time method–target pairs have identified efects, and releases, retractions and policy events have partly identified efects. We further test whether natural-language questions can be answered without a language model performing any computation. Duplicate records (4.5 % of records but 83 % of naively detected bursts) were removed from the forecasting evaluations and from the detection of bursts. Each evaluation followed a pre-registered decision rule <sup>25</sup>, and comparisons of trained components used at least five independently trained runs per arm with a run-level bootstrap. A forecast made from December 2025 was frozen for later scoring against observed data.

## Results

## Latent state from the paper stream

The research world model reads 1.20 million power-engineering abstracts (1.14 million papers after deduplication), the 11.1 million citation events of the citation layer and the use of their terms in papers from seven other fields (Fig. 1a). A distilled language-model reader extracts from each abstract its role-typed concepts and their relations, which form a monthly typed hypergraph of 204,287 concepts with one hyperedge per deduplicated paper. This hypergraph is read by two temporal graph networks, the growth core and the link core. The hypergraph and the concept states that these networks compute from it form the state $S _ { t }$ of equation (1). The growth core forecasts the log growth of each concept 12, 24, 36 and 60 months ahead (Fig. 1a,b,c(i)). On the field map, four links between regions that the link core ranked at December 2019 (ranks 1,551–3,996) were first named in a title or abstract in 2020–2023 (Fig. 1b). On the states of the link core, a link readout forecasts which concepts will be linked for the first time (Fig. 1c(ii)), for all 15 pairs of the five concept roles (problem, method, object, setting and metric).

Five operators read the trained core states and answer what-if questions according to their subject. The entity-insertion operator adds a term that does not yet exist and forecasts its trajectory (Fig. 1c(iii)), and the observation-assimilation operator revises that forecast once the first papers using the term have appeared. The event-readout operator predicts the citations, pair reuse and disruption <sup>26</sup> of a single paper. The intervention operator estimates how a burst of injected activity changes concept states, and check and whose concepts are grounded. The deterministic action executor runs scenario and baseline on shared random numbers and returns historical analogs and a validity badge. The grounded answer generator (a fine-tuned writer) produces the answer from the computed output alone, and the independent judge model audits each claim. b, Forecast concept map (t-SNE <sup>27</sup>, December 2025) colored by 24-month forecast. Hexagons, forecast change in share of mentions. Circles, concepts born in 2023 or later. Arrows, four links between regions ranked at December 2019 before any title or abstract named both terms (ranks 1,551, 1,650, 3,239 and 3,996; first named 2020–2023). c, Running example. c(i), growth forecasts of the growth core (80 % intervals). c(ii), a pair ranked 128 of 2,046,140 by the link core at December 2019 that formed later. c(iii), a hypothetical new term. c(iv), executor result for the question of a: 24-month efects on three concepts (lines, 80 % intervals of the three injected pairs per concept; box, level band; point, combined efect) and the entity-insertion forecast for the new concept (papers in 24 months, 80 % interval). d, Top-ranked unlinked pairs formed within 24 months, against shared-neighbor ranking and the base rate. e, Intervention operator on 574 test bursts: share of top-ranked bursts with a top-third realized efect (top tenth: method 75 %, 2.3 times the base rate of 33.4 %; target 63 %, ×1.9).

a Research world model architecture  
![](images/c516b3b6539a8281f08272aed832df8e8da3f6e13eb70004dcb1f0ceb0cde482.jpg)

![](images/2b164d70de9192cfe736d492fd816a63d8d4eeadef55b4749e36e74fc1e13d02.jpg)

![](images/f4a23f2a854a600ce3a9612bfce5936f8e26109cb6628948c3f18e9d34b9d0fd.jpg)  
ii link core

![](images/2abb056a253dbdebe25611c5f8298ae88980c3d77dbf315aba937c6359005690.jpg)

![](images/3fb6decee98a89fd5afcd0c96c3b5f088ccb850a369d9a453b5e05e3651e3d2d.jpg)

![](images/9b9e7c5cb7a3613c26b1a6d6493e03e94514df1b300fd107c64b268adb25a3c3.jpg)

![](images/392230324fa1e58ee21f41c5e7e3505ff8d7ce774a6228d6260d1048b870358d.jpg)

![](images/96b70234eccfdba1d1419cc6da2e7ac458cb6511a628da02be480f7ba47ee57c.jpg)  
Fig. 1 | A research world model of power engineering. a, Architecture (schematic, with tensor dimensions as implemented). (1) A 7-billion-parameter reader distilled from a 27-billion-parameter teacher converts 1.20 million abstracts into role-typed concepts. Citations and use in seven other fields form the citation-attention and external-supply layers. (2) The field state, a monthly typed hypergraph (real subgraph around the running example). (3) The growth and link cores (stacked cards) share one architecture, in which a graph encoder and a temporal transformer return a 128-dimensional concept state, and are trained separately. The link core adds a graph stream with a learned gate, concept text and a typed-link head. One link readout on the link-core states scores pairs of all 15 role combinations, and five operators (teal shades) read the trained core states of both cores. (4) Interfaces, shown for a what-if question (box, condensed). In the action model, a fine-tuned decoder compiles the question into an action program over state primitives, which passes a schema

the estimators for the other what-if events are its variants. The transition operator advances the whole field as paired scenario and baseline trajectories.

A question in natural language reaches the operators through a fine-tuned decoder, which compiles it into an action program, either a registered action of one of 18 types or a composition of six state primitives, and grounds every action to existing concepts. The program is run by the deterministic action executor once as scenario and once as baseline with shared random numbers. It returns their diference with intervals, historical analogs and a validity badge that states how far the efect is identified (“validated”, “partial”, “extrapolation”, “not supported” or, at level 2, “generated”). A grounded answer generator (a fine-tuned writer) produces the answer from this output alone (labeled “computed output” in Fig. 1a), and an independent judge model audits the answer claim by claim. The what-if question of Fig. 1a lies outside the registry. The fine-tuned decoder compiles it, at level 2 of the action model, into an added concept and bursts of papers (Methods), and the executor returns 24-month efects of 1.50–1.70 times the baseline on three concepts and 6.6 papers for the new concept (Fig. 1c(iv)). Fig. 1d,e summarizes the link-formation and intervention results reported below: the share of top-ranked unlinked pairs that formed within 24 months, against shared-neighbor ranking and the base rate, and the share of the top-ranked test bursts of the intervention operator whose realized efect fell in the top third.

## Forecasting concept growth

Growth forecasts from the learned state were compared with those of strong conventional regressors at five walk-forward cutofs (2013–2021) and four horizons, in 19 cells with complete labels. Each growth head is trained with a horizon-specific cutof and learns from every past month (anchor) whose outcome at its own horizon had been observed by the cutof (Fig. 2a and Methods).

The growth forecaster blends the ranking of ten horizon-specific runs with that of a tuned multilayer perceptron, and it is evaluated on the 15 cells of the cutofs other than December 2019. In 13 of these cells it was better than both a multilayer perceptron and LightGBM <sup>28</sup> tuned on the same node features, each time with a 95 % interval above zero (Fig. 2b). It also ranked growth better than a zero-shot language model. Its top-minus-bottom decile median of realized growth was +0.32, against +0.27 and +0.23 for the tuned perceptron and LightGBM (Fig. 2c), and its top-100 enrichment was higher in 37 of 38 comparisons (Fig. 2d). It exceeded default LightGBM at all 393 test anchors and, over all 19 cells, both tuned baselines in 16, with no interval entirely below zero.

## Forecasting new links between concepts

Links between concepts depend on pair-level structure that node-feature regressors do not represent. The link core adds a gated graph stream, a typed-link head and concept text to represent it, and one link readout on its trained states scores pairs of every role combination. At all five cutofs in power engineering, the link core had a higher typed average precision than the growth core, by +0.023 to +0.049, with every interval above zero. Growth and links are therefore forecast by separate cores, each trained for its own task (Methods). The link core is trained with horizon-specific cutofs, under which its primary metric, hard-negative typed average precision, exceeds that of the default scheme by +0.023 [+0.018, +0.029].

A retrospective test was run at December 2019 on every unlinked pair among the 1,500 most active methods and the 1,500 most active problems and objects, about 2.05 million pairs. Of the 100 pairs

## a Training and validation design

![](images/97b730cb41ec696ef7f1e1f690aa95960568d44333d479008b56bb009bdbd28f.jpg)  
⑤ Tuned baselines

![](images/7d20c5bea3bb8d20b16809f48d95b0ccd8d4d0bd9c5296c27e5c94cb914779ef.jpg)

![](images/a936e9226e6a712fcc4b05f329ea94c99915ce920e681aae75f0ab70764a17d0.jpg)

![](images/5eda9507ad942a03b12f1b4696a4e8483e48b1f06b95a65770a30e1847ab01cf.jpg)

![](images/066cdcb7805873c773c542745dc19d7099baa478abc444c4ada7f7bb607d6791.jpg)

b Growth forecast skill  
![](images/962955e77036d9aaa8504fbead894c5b2e354d2a4c4a50fb9157197b1d22a1c5.jpg)

![](images/f04afdd5b6e54b9e7a689dfb58261686a5d6b6e77a2e08cbb11a1bc0d557bc33.jpg)

C Ordering by forecast decile  
![](images/c088f9542c415c67688a1677067479a32410c6fb0cc2dee14a275bdf21e64c4d.jpg)

d Top-100 enrichment  
![](images/dcbf477e939194b6edf6c970fff720c2ee1289ad23957f6ee6a27681649456d8.jpg)

f Kinds of links formed  
![](images/56db9d5450edbc0fe7fdd6f40459315db957d3989cc5711899833fa20a0eeccc.jpg)

![](images/763cd8b853dd0cc71ce841e7268e966bec8cc128436ff5e0d47d08f45d6d6779.jpg)

g Forecast reliability  
![](images/d5f76a08c5f43a4edb339e1c488f8a02bfdcd505671dae70f72a3d38f99f5652.jpg)

![](images/a5575a5b6ff91cf16dc96e38fb063908d16b6caafbc050da53aaef334443d041.jpg)

![](images/502e3048f39cf7f3e6f6bf361c15ccdac5f113759b0de1c9fe60f027e8a57db1.jpg)  
Fig. 2 | Training, validation and test of growth and link forecasting. a, Design, with mini-plots of results. (1) A training anchor with masked growth losses and a typed-link loss. (2) Horizon-specific training cutofs. (3) Ten independent runs. (4) Blend weight chosen on validation cores trained at the cutof minus the horizon. (5) Tuned baselines on the same 26 node features. (6) Walk-forward test at five cutofs. b, b(i), test Spearman correlation over the 1,000 most active concepts by horizon for three cutofs, growth forecaster (solid) against the stronger tuned baseline (dashed). b(ii), gain in each of 19 cells over tuned LightGBM and over the stronger tuned baseline (mean and 95 % interval where above 0). The forecaster exceeds tuned LightGBM in all 19 cells and both tuned baselines in 13 of the 15 unseen cells. December 2019 cells, seen during development, are shown for comparison. $\mathbf { c } , \mathbf { c } ( \mathrm { i } )$ , realized log growth by forecast decile for the growth forecaster and the two tuned baselines on identical anchors. c(ii), top-minus-bottom decile median (+0.32, +0.27 and +0.23) $\mathbf { d } ,$ Top-100 enrichment over chance, growth forecaster against each tuned baseline per cell (37 of 38 points above the identity line). e, Share of the top 1,000 unlinked pairs formed within 24 months (filled) and by March 2026 (open), by rank, target role, shared neighbors and activity. f, f(i), December 2019 top 1,000 by map region of method and target. f(ii), months to

formation per typed relation. f(iii), first relation extracted. g, Reliability. g(i), hindcast direction (96, 92, 88, 83 %) and Spearman correlation (0.90, 0.92, 0.89, 0.82) by horizon. g(ii), origin stability of the five-run forecast (Spearman 0.888). g(iii), agreement between origins.

that the link core ranked highest, 72 % formed within 24 months, against a base rate of 1.75 % (Table 1). Over both cutofs (December 2015 and 2019) and two seed sets, the top 100 formed at 35–41 times the base rate (Fig. 1d). Under the strictest alias map, which also merges aliases matched by embeddings, the December 2019 share is 65 % instead of 72 %.

A stricter test was run on new pairs, whose two concepts no earlier paper mentions together under any variant of their names and for which no earlier paper contains concepts whose names include both. Among the top 100 new pairs, the share that formed within 24 months was 36 % at December 2015 and 50 % at December 2019, and the share that formed by March 2026 was 89 % and 92 % (Table 1). Of the 1,000 highest-ranked unlinked pairs, 39 % (December 2015) and 57 % (December 2019) formed within 24 months (Fig. 2e,f(i)) and 86 % and 90 % by March 2026. At December 2019, formed pairs with an extracted relation took a median of 13–16 months, against 23 months for co-mention only (Fig. 2f(ii),(iii)).

First links between concepts of any two roles, for example between two methods, between a problem and a setting, or between an object and a metric, are scored by the same readout. In all 75 combinations of role pair and cutof, the readout’s average precision exceeded that of the strongest of 11 baselines, with an interval above zero in 74 (Methods).

In 24 hindcasts of concept growth from four earlier origins, the forecast direction was correct in 96, 92, 88 and 83 % of cases at 12, 24, 36 and 60 months (22 of 24 at 24 months; 18 hindcasts at 60 months; Fig. 2g(i)), and the five-run forecast agreed between origins at a Spearman correlation of 0.888 at 24 months (Fig. 2g(ii)). Between the two origins, the Spearman correlation was 0.886–0.908 at 12–60 months, and 78–83 % of the 100 highest-ranked risers were shared (Fig. 2g(iii)).

## Operators on natural experiments

Each operator was validated on natural experiments not used in training, against the strongest available baseline. The intervention operator learns from genuine bursts of papers on first-time method–target pairs, each matched to five control concepts in a diference-in-diferences design <sup>30</sup>. On 186 blind test bursts, its predictions correlated with the realized efects at r = 0.557, which is 0.78 of the noise ceiling set by the spread of the matched controls (Fig. 3a). Its squared errors were smaller than those of injecting the burst into the core or of assuming a constant average efect. Across all 574 test bursts, Pearson correlations of the per-burst predictions were 0.48–0.56 (method) and 0.32–0.40 (target, over two seed sets), and the share with the predicted sign rose from 55 % to 87 % (method) and from 63 % to 80 % (target) between the lowest and the highest thirds of absolute predicted efect (Fig. 3a). On the method side, three-quarters of its top tenth had realized efects in the top third, 2.3 times the base rate (Fig. 1e). Predicted means also tracked realized means within strata defined on training bursts, with correlations of 0.96 (2015) and 0.79 (2019) against a criterion of 0.6 (Fig. 3b).

The transition operator was tested by replaying the test bursts. Two ways of propagating a burst (paths; Fig. 3c(i)), feeding its papers into the state and carrying the intervention efect forward, ranked the realized efects with Spearman correlations of 0.33–0.38 and intervals above zero, as did the direct growth head. The method-side ranking criterion held at every evaluable horizon from 12 to 60 months, and the follow-on first-time pairs per burst (+11 method-side and +15 target-side pairs in the training era, +55 and +44 in the test era) met the pre-registered criterion for a growing graph (Fig. 3c(ii)).

Because a new term has no history of its own, the entity-insertion operator was compared with a structural prior taken from the link cores. It ranked 24-month growth better than the prior at all four cutofs, with gains in Spearman correlation of +0.085 to +0.226 (Fig. 3d,e). The operator’s 80 % conformal intervals<sup>31</sup> covered 0.774–0.792 of outcomes.

Once the first papers on a term have appeared, the observation-assimilation operator is applied instead. Three and six months after birth, it improved at all four cutofs on the growth head applied to the term’s real core state (Fig. 3f). After one year, its 24-month forecast reached a Spearman correlation of 0.66 (Fig. 3e(ii)). Its 80 % intervals covered 0.77–0.82 of outcomes at every cutof. For single papers, the event-readout operator predicted 36-month citations, reuse and disruption better than LightGBM with text features in every run, and the operator results held under new seeds.

ii  
![](images/16318879a9d16bbd34aa6960697618ab38af104765ee660842167e73cdb9771d.jpg)

![](images/b04ce0da68b94e5a7efd34ddbc4edf4b53b1d26bd34f5f2f14eb624a60cd7874.jpg)

![](images/dc1756c719439d34f909ee0f69074e13bbb25d03ff77f481595e3262a5dc3c8c.jpg)

b Stratum agreement  
![](images/5eb1d7d659f2001945de9312fafff33f5396fe9b5b9ccee24ccc7bb3051821d3.jpg)  
d Beyond the structural prior

C Replayed bursts  
![](images/489d9b689634014b9b8a4d78ba1a1e0796aae2b0654166daa0934e9c62f68052.jpg)  
e Newborn-term growth

ii  
![](images/5871a7956c50eb61e059359c355c138ec57feeac159fd42b65ff32307319a8d1.jpg)  
f Assimilation gain

![](images/4d62fa4d1558dce6ebd9ee3e9ac4a5ab505c30a003f673a62ae2ae73737221a1.jpg)

![](images/0e7697d1872bff87fcc13cdc4c12cd5e846ab274a44ce1970ec9069fd19f1dd5.jpg)

![](images/d2f46a416c57dbb44beebe36560454278172077b2e6f6afba28dc55f8cd04681.jpg)

![](images/f710854a3913f08b2d6297130f767ddcee6a59c6bc62791526ff72894fb9009c.jpg)  
Fig. 3 | Operators on held-out natural experiments (power engineering). ${ \mathbf { a } } ,$ Intervention operator on 574 test bursts (filled, December 2019 cutof; open, December 2015), predicted against realized 24-month efect relative to five matched controls. a(i), method side (Pearson 0.48 and 0.56). a(ii), target side (0.40 in 2019; 0.32–0.40 in 2015 across two seed sets). a(iii), share with the predicted sign by thirds of absolute predicted efect (all 574 bursts; method $5 5 \% $ 87 %, target $6 3 \%  8 0 \%$ . $\mathbf { b } ,$ Stratum means, predicted against realized (correlation 0.96 in 2015, 0.79 in 2019; criterion $\geq 0 . 6 ) . \ \mathbf { c } ,$ Transition operator replaying the bursts. c(i), realized efect by decile of the simulated paired diference for two paths. c(ii), follow-on first-time pairs per burst, training and test eras. d, Entity-insertion operator, 3,472 terms first seen after December 2019. $\mathrm { d } ( \mathrm { i } )$ , realized growth by structural-prior quintile and operator tertile. d(ii), Spearman correlation at four cutofs and interval coverage. e, Newborn forecasts at birth (e(i), entity insertion; Spearman 0.29 against 0.21 for the prior) and after 12 months (e(ii), observation assimilation; Spearman $0 . 6 6 , n = 8 5 7 )$ . e(iii), coverage by quintile f, Observation-assimilation gain over the growth head of the real core state by age and cutof.

Table 1 | Forecasting new method–target links
<table><tr><td></td><td colspan="2">December 2015</td><td colspan="2">December 2019</td></tr><tr><td></td><td>within 24 months</td><td>by March 2026</td><td>within 24 months</td><td>by March 2026</td></tr><tr><td>All unlinked pairs, precision at 100 (%)</td><td></td><td></td><td></td><td></td></tr><tr><td>Link readout, horizon-specific link cores</td><td>49 [39, 59]</td><td>88 [82, 94]</td><td>72 [63,81]</td><td>91 [85,96]</td></tr><tr><td>Link readout, default-scheme link cores</td><td>55 [45, 64]</td><td>91 [85, 96]</td><td>67 [58, 76]</td><td>87 [80, 93]</td></tr><tr><td>Adamic-Adar</td><td>30 [22, 39]</td><td>71 [62, 79]</td><td>23 [15, 31]</td><td>51 [41, 61]</td></tr><tr><td>Base rate</td><td>1.40</td><td>6.80</td><td>1.75</td><td>5.47</td></tr><tr><td>Enrichment, horizon-specific link cores (×)</td><td>35.0</td><td>12.9</td><td>41.1</td><td>16.6</td></tr><tr><td>New pairs, precision at 100 (%)</td><td></td><td></td><td></td><td></td></tr><tr><td>Link readout, horizon-specific link cores</td><td>36 [26, 45]</td><td>89 [82, 95]</td><td>50 [41, 59]</td><td>92 [86,97]</td></tr><tr><td>Link readout, default-scheme link cores</td><td>29 [21, 38]</td><td>83 [75,90]</td><td>42 [33, 52]</td><td>85 [78,92]</td></tr><tr><td>Adamic-Adar</td><td>18 [11, 26]</td><td>56 [46, 66]</td><td>15 [8, 23]</td><td>40 [30, 50]</td></tr><tr><td>Base rate</td><td>0.81</td><td>4.63</td><td>1.02</td><td>3.59</td></tr><tr><td>Enrichment, horizon-specific link cores (×)</td><td>44.4</td><td>19.2</td><td>49.0</td><td>25.6</td></tr></table>

Power engineering. Candidates, pairs of the 1,500 most active methods and the 1,500 most active problem or object concepts without a typed link at the cutof (2,099,192 at December 2015 and 2,046,140 at December 2019). All unlinked pairs, all candidates. New pairs, candidates whose two concepts no earlier paper mentions together under any variant of their names and for which no earlier paper contains concepts whose names include both, ranked within this pool. Precision at 100, share of the 100 highest-ranked pairs that formed a typed method–target link under the exact concept names within 24 months of the cutof or by March 2026. Link readout, scores of the readout on link cores trained with horizon-specific or with default-scheme training cutofs (mean of five runs). Adamic–Adar <sup>29</sup>, on the co-occurrence graph at the cutof. Base rate, share of the candidates of the block that formed (%). Enrichment, precision at 100 divided by the base rate. Square brackets, 95 % bootstrap intervals over the top 100 pairs.

## Natural-language what-if questions

Each of the 17 what-if types in the registry, and the observation type, is labeled with the extent to which natural experiments identify its efect. For the demonstration observation question, the executor returned 24-month ratios to the baseline of 0.62–1.54 for three grounded actions, and the judge-model audits under three sampling seeds supported 54 of 56 claims. For the release of a dataset, code or tool, the model estimates the uptake of the concept that the release serves at +0.03 log, an efect confirmed on test events at both validation cutofs. The model estimates that retracted papers receive in-field citations 0.25 log below those of matched papers. Both efects are partly identified.

Questions are compiled by the fine-tuned decoder at two levels: a question matched to a registered type is executed by the module of that type (level 1), and any other question is compiled into an action program over six state primitives, each executed by an existing operator (level 2; Fig. 4a). On 449 new questions, 136 of them outside the registry, the decoder routed 0.878 of the questions exactly, against 0.768 for a general-purpose language model given the same registry (diference +0.109 [+0.069, +0.149]), and per-type routing thresholds raised exact routing on 187 further questions (diference +0.021 [+0.005, +0.043]; Fig. 4b). The independent judge model passed 0.71 of 1,550 programs in 31 categories on all five of its checks (Fig. 4c). Where historical events exist, the realized efects of programs were reproduced for adoption targets, whose precision exceeded that of a ranking by target popularity by +0.058 [+0.017, +0.096] over 117 test-era waves, and, in level, for dataset or tool releases (bias −0.066 [−0.171, +0.032]; Fig. 4d).

For registered actions, the fine-tuned decoder uses two low-rank adapters <sup>32,33</sup> on its base language model, one for actions and one for clarification requests. In end-to-end tests, the fine-tuned decoder grounded both method and target to the correct concept in 0.788 of questions, against 0.560 for the base model without adapters, and decoded more multi-action questions exactly (Fig. 4e). On a new set

c Program audit by category

fine-tuned decoder against base model O base model fine-tuned decoder

ii 187 questions, 77 outside the registry  
![](images/f9ea5c9c3fe899b4c458f7777ed6ad16c40ed06932eda8a50269579026e8e912.jpg)

![](images/152c60bedd6d6c627e82a920a9fe224ff19a01ba35347304069bdb0061b7f43c.jpg)

a From question to computed answer  
![](images/3af5db8bcca7cce96a7fbebf0d9a54fa2b1c10bfe5e18c7f2a8901df7a62556f.jpg)

## b Routing of new questions

![](images/4070c5869ac351c4d7919c15d53fb7274a4065cd9582493c785f31d43d65482a.jpg)

![](images/d0be9354b841563b33889784dd3b29d455d346d1cec9f4e65b745e8737214a5f.jpg)

## e Decoding and grounding

![](images/25c3f838643b112c11fdbc03d10b62f99e0f0c85b334a14aa1ce0e4ba432027e.jpg)  
f Faithfulness of answers

![](images/0547960f9b1d7bb4d9a5bcf8825f8e8cbf45501dc3f031e9716c2fa55222afd8.jpg)

![](images/c450ba446193433d89bfd7d5b06149527bd10f540085d4a54ef64b9c122bdfc7.jpg)

## d Realized effects of programs

![](images/53bfff3d0570fb9604de5e452014c0343957c320c16ab3d37cf01887d3173dca.jpg)  
g Target proposer

![](images/629d8d5ad0854f50185b25f97608bdc4415a1268d212aa97e455a2fdfb773450.jpg)  
ii grounding confidence of concept slots

![](images/ca20f8d876d7fb6492b2faa9494fa0455e54ddbbc54f588f545ad5e112d222ad.jpg)

![](images/39265a70f2421707fe9842f70bb3bdaba8fc3f43fbca2aeba91828c7de8e69d6.jpg)

![](images/86425186e0859fc1240ca588de0533f9e456bc0099b55267059784b60db39b37.jpg)  
partial or validatedextrapolationnot supported

![](images/4dc371da9204bc69507fcad4b41dd1f96ecc8ad2a491f73ace28f2df4e037814.jpg)  
Fig. 4 | From a what-if question to a faithful answer. a, From question to computed answer (schematic with system outputs). Left, training of the fine-tuned decoder (adapter on Qwen3.8-27B; 5,003 rows written and labeled by a general-purpose language model; routing threshold 0.5, −0.25 for six action types) and of the grounded answer generator (fine-tuned writer; supervised fine-tuning, self-play and four rounds of expert iteration, the last two ranked by judges distilled from labels of the independent judge model). Right, route margins of the 449 routing-test questions (at the threshold 0.5, the decoder routes 275 to level 1 and 174 to level 2). Example 1 is matched to a registered type, and example 2 is composed from state primitives, its removal of future output executed by the intervention operator as a burst with the

opposite sign. The deterministic action executor runs scenario and baseline through the trained cores and the transition operator in 6-month steps with shared random numbers. Historical analogs, the five most similar past genuine bursts for example 2. The judge model checks each claim against the computed output. b, Routing of new questions (95 % Wilson intervals). b(i), 449 questions, fine-tuned decoder against the general-purpose language model (exact routing 0.878 against 0.768, diference +0.109 [+0.069, +0.149]). b(ii), 187 further questions, global against per-type thresholds (exact routing +0.021 [+0.005, +0.043]). c, Level-2 programs passing all five checks of the judge model, 50 programs per category (95 % Wilson intervals; dotted, 0.70; gray, all 1,550 programs, 0.71). Heat map, share of programs meeting each check. d, Realized efects of the two validated level-2 program types on historical events, with the event type hidden from the decoder. d(i), precision of the adoption targets against a ranking by target popularity (117 test-era and 133 earlier waves). d(ii), per-wave diference in target precision, programs minus popularity, over the 117 test-era waves (mean +0.058 [+0.017, +0.096]; share of waves higher, equal and lower). d(iii), realized 24-month efects of 60 dataset or tool releases against the constant release efect of the programs (+0.007; bias −0.066 [−0.171, +0.032]). e, Decoding and grounding of registered actions against the base model. e(i), both slots grounded correctly (3,868 questions; 0.560 → 0.788), exact multi-action set (210; 0.762 → 0.881, adapter run used in the system) and no needless clarification (157; 0.490 → 0.975). e(ii), grounding confidence of 365 filled concept slots. e(iii), Spearman correlation of burst efects predicted from decoded questions with realized efects (bars, gain; whisker, 95 % interval of the gain; black line, gold actions). f, Faithfulness of answers audited by the judge model, three sampling seeds. Top, the 18 demonstration questions (4.44 [4.28, 4.61]) and 215 new questions (4.13 [4.07, 4.19]). Rows, the 18 action types: badge mix of the new answers, per-question means (dots), type mean with 95 % bootstrap interval (diamond), demonstration question (star) and claims supported. g, Target proposer on 21 historical adoption waves. g(i), per wave and mean with 95 % interval (circles, precision at 8; squares, recall at 50). g(ii), precision at 8 (0.464 against 0.321 for popularity) and recall at 50 (0.686 against 0.429). g(iii), share of waves with higher, equal or lower values than popularity (precision at 8: 15, 5 and 1).

whose labels were verified by the independent judge model, the decoder asked for clarification on as many ambiguous questions as the base model but on only 4 of 157 clear ones, where the base model asked 80 times. Burst efects predicted from its decoded questions also correlated more strongly with realized efects, consistent with its better grounding of method and target (Fig. 4e(iii)).

On 215 new questions covering every action type, the independent judge model compared each answer with the executor output it was written from. Faithfulness was 4.13 [4.07, 4.19] of 5 and 90 % of claims were supported (Fig. 4f), and faithfulness was 4.10–4.17 in every identification class. The 18 demonstration answers, one per what-if type plus observation, scored 4.44 (Fig. 4f).

For adoption questions, a target proposer, learned from link-core scores with a correction for target popularity, selects the targets of the method, and the intervention operator computes the efect on each method–target pair. The executor reports method-side quantities and target efects as separately labeled entries. The grounded answer generator writes each answer from these entries and states extrapolated efects by their direction. Answers to 62 adoption and group-move questions (one sampling seed) scored a faithfulness of 4.13, with 93 % of claims supported and an over-claiming rate of 0.048. Answers to 23 questions whose actions are all extrapolations scored 4.12, with an over-claiming rate of 0.10.

Questions about the adoption of a method are routed to this target proposer. Over 21 historical adoption waves, its precision at 8, the share of its top 8 targets that the method reached, exceeded that of a ranking by target popularity by +0.143 [+0.083, +0.202] (Fig. 4g).

## Field-scale scenarios

Each scenario was run from the validation origins December 2015 and December 2019 as 12 paired trajectories under several mechanism families, with full dose curves (Fig. 5a,b). Of 25 candidate directions, 19 met the robustness criterion (Fig. 5a,c), and a flood of machine-generated papers changed five-year field entropy by more than run-to-run noise.

Moving papers to first-time method–target pairs produced more five-year growers than repackaging the same number of papers at both validation origins, 2.8 times as many at 2019 (Fig. 5b(i)). When a fifth of method mentions was re-allocated toward growing methods, boom–bust methods became 190 % more numerous at ten years (2019 origin, mechanism family without feedback; Fig. 5b(ii)). Faster method turnover and a halved computer-science inflow raised field entropy, a disruptive method shock lowered it (Fig. 5b(iii)–(v)), and the flood concentrated the observed literature. Retracting part of the papers of a hot topic reduced its five-year output roughly in proportion to the share removed.

From the forecast origin of December 2025, every direction that met the robustness criterion keeps its validation-origin sign, except innovation against repackaging in two of its five mechanism families. The

scenario scenario arm minus same-state baseline arm, 2019 origin

![](images/4f82ecb33442c599acfcff8b8ac294b9c9dfb915fbb537f914fee06b37a4603d.jpg)

![](images/33a68657c032ab99d4ddcb042dd0615f96e9ce69732e37edf9f2bed0c05c3c85.jpg)  
b Paired rollouts  
baseline arm, same state.

i  
![](images/29e7bed9ea75c78ea4c389efd1c6db53a9df324e07e7830ef6a42760945d50b7.jpg)  
ii

![](images/3d071987c21a08dea23f1fe29820a914936ae70c2b4691441a97edbaf59dacdb.jpg)  
Growth-chasing herding re-allocation to fast growers

![](images/368d7a94fca9184a8dec72b79bb1a4c7c67f26a89a8264182d12fd8f544eab15.jpg)  
Boom-bust change, 2015 (%) papers fed back +54+122 observed only +252 additional papers +65+136 +100 +200 +300

iii  
![](images/11eeceacf6915b174b3bedaa4ef85d172d215e801e8911a72d15df8a239ef5c1.jpg)  
iv

![](images/a279874a7727a11c847a691d954b679952064ea1ee9a18c1b42154ce6778a503.jpg)

V  
![](images/01fe8c742fa08db2696b3e4d150da4bf0e828e8fcc5ff62a813a9e281cddc557.jpg)

<table><tr><td>Concepts, inflow ×0.5 deep Q-learning</td></tr><tr><td>-0.28 cloud storage</td></tr><tr><td>-0.12</td></tr><tr><td>fog node -0.13</td></tr><tr><td>planar inductor 0 +0.08</td></tr></table>

![](images/6e8a722d00bcc0245373260314fa7e9c79ac8251616ed8b0ec8696b8912b2f52.jpg)

<table><tr><td>Target concepts gaining short-term electricity load forecasting</td><td>+0.65</td></tr><tr><td>time series prediction</td><td>+0.56</td></tr><tr><td>sky image</td><td>+0.55</td></tr><tr><td>over-fitting</td><td>+0.43</td></tr></table>

## C Scenario robustness

201520192025extrapolation

<table><tr><td colspan="2" rowspan="2">Scenario outcome, reference dose</td><td rowspan="2">Mechanism family</td><td colspan="2">Effect, 2019</td></tr><tr><td colspan="2">95 % interval includes zero reference dose</td></tr><tr><td rowspan="2">4</td><td rowspan="2">five-year field entropy shock size ×1</td><td>Disruptive method shock intervention effects, fixed total target mentions, additional</td><td>■</td><td>-0.0115</td></tr><tr><td>target mentions, fixed total</td><td>o</td><td>-0.0080 -0.0067</td></tr><tr><td colspan="3">心 Machine-generated flood Field entropy</td><td></td><td></td></tr><tr><td rowspan="10">50 % of human output</td><td rowspan="10"></td><td>0</td><td rowspan="10"></td><td></td></tr><tr><td>papers fed back</td><td>-0.61</td></tr><tr><td>crowding out</td><td>-0.54</td></tr><tr><td>observed only</td><td>-0.21</td></tr><tr><td>Top-1 % share (points)</td><td></td></tr><tr><td>papers fed back</td><td>+12.6</td></tr><tr><td>crowding out</td><td>+11.4</td></tr><tr><td>observed only</td><td>K +4.5</td></tr><tr><td>additional newborns</td><td>+0.0418</td></tr><tr><td colspan="2" rowspan="2"></td></tr><tr><td rowspan="4">five-year field entropy newborn rate ×2</td></tr><tr><td>zero-sum on parents</td><td>+0.0392</td></tr><tr><td></td><td>+0.0333</td></tr><tr><td>-30-10 -20</td><td>2 10 30</td></tr></table>

## d Forward scenarios to 2035

![](images/155443233693bf70254c46aebf2d3a6a71e168fef849d0e770cbfdd3825103d5.jpg)  
e Concept-level change by size  
Paired effect ÷ standard error

![](images/f537f6093026f0db2481e24b4faafc2775754684ea2b04c76662b44fd68d672f.jpg)

![](images/e2ccd8b8bdcd5e5b50bdb3cd20c57dbb7d222c996bd8a06b287a35b0b6688327.jpg)  
Fig. 5 | Field-scale scenarios with the transition operator. a, Simulation design. The trained state (typed hypergraph field state with the trained growth and link cores) at the validation origins December 2015 and December 2019 and at the forecast origin December 2025 is perturbed by seven scenario types, each run under several mechanism families (tiles) and a dose ladder (bars; outlined, reference dose). The scenario arm and a baseline arm from the same state are rolled forward by the transition operator in 6-month steps with shared random numbers, giving 12 paired trajectories (3 checkpoints × 4 noise streams). Their scenario-minus-baseline diferences are shown for faster method turnover (field entropy, 2019 origin; thin lines colored by checkpoint, mean and 95 % interval of the mean). A direction meets the robustness criterion only when every mechanism family gives the same sign with a 95 % interval excluding zero at both validation origins (example: five-year field-entropy efect of faster method turnover for its three families; square, 2015; circle, 2019). b, Paired rollouts at the 2019 origin (thin, pairs; thick, mean). b(i), innovation shift (5 % of papers to first-time pairs), extra five-year growers against equal-dose repackaging (×2.8); below, innovation ÷ repackaging (median over families) against the share shifted at both origins. b(ii), growth-chasing herding, boom–bust methods (+190 % at ten years with a fifth of method mentions re-allocated; family without feedback), with ten-year crashes per family and the 2015 replication below. b(iii)–b(v), field-entropy change under faster method turnover, halved computer-science inflow (and its mirror ×2) and a disruptive method shock (dashed, 2015 mean), with cumulative first-time pairs (b(iii)) and named concepts with the largest changes (b(iv), b(v)) below. Every plotted direction meets the robustness criterion of a. c, Paired efect over its standard error at the reference dose,

per family and origin (herding at exponent 4; gray, interval includes zero). d, Forward scenarios from December 2025 to 2035 (paired diferences only). d(i), field-entropy change (AI boom, computer-science inflow doubled; AI-adoption wave, deep-learning-sized adoption path). d(ii), frontier concepts and the machine-learning-forecasting share. e, Machine-generated flood. e(i), five-year change of human papers per concept (papers fed back, 2019 origin). e(ii), change in the top-1 % concept share against flood size per mechanism family at three origins (ringed, reference dose).

other three still favor innovation. From this origin, the flood lowered field entropy by 0.47 at five years and a doubled computer-science inflow by 0.25 at ten, and faster turnover raised it (Fig. 5d). Efects changed monotonically with dose, and in the families with feedback herding reversed sign at growth exponents of 0 and 1. A flood of half the human output raised the top-1 % concept share by 4.5–15.0 percentage points at the validation origins and by 3.3–9.7 from December 2025 (Fig. 5e).

## A prospective forecast

From December 2025, the model forecasts 20,740 active power-engineering concepts (Fig. 6a,b), each with an 80 % interval and a separately reported era bias band. The largest gain in share of mentions to 2030 is forecast for the machine-learning-forecasting region (+5.2 percentage points; Fig. 6a(i),c), led by retrieval-augmented generation, explainable AI, hybrid frameworks and computational overhead, whereas the power-electronics, power-quality and electrical-machines regions lose most. Large language models and quantum computing are forecast to reach 3.6 and 2.2 times their 2025 output by 2030 (Fig. 6b(i)). In the transition-operator baseline, the top-1 % concept share reaches a simulated 48.6 % by 2035.

The link core forecasts method–target links that no earlier paper had made, among them Kolmogorov– Arnold networks for extreme weather events and cascading failures, proximal policy optimization for soft open points and isolation forests for transient stability assessment (Fig. 6a). Of the 200 highest-ranked pairs, 83 % have never been mentioned together in the corpus under their exact concept names, 33 % under any spelling or synonym variant and 25 % when name containment is also counted, and 82 % join concepts of one map region (Fig. 6d). From the March 2026 state, the entity-insertion operator ranks large language model, retrieval-augmented generation, YOLOv8 and low-rank adaptation first among recent newborn terms (Fig. 6e). Among the newborn terms of 2024 and 2025, hallucination and generative artificial intelligence lead the problem and method roles (Fig. 6f). At 24 months, the median forecast of every role is close to no change, at 0.90–0.99 times the 2025 level (Fig. 6g(i)), and the share of concepts forecast to grow rises with the horizon, to 51–68 % at 60 months (Fig. 6g(ii)).

## Discussion

Research world models, as introduced here, draw forecasts and what-if answers from one transition of a research field, learned from its publication record. Earlier forecasts of research fields predicted new links in concept networks built from the literature. Krenn et al. <sup>21</sup> forecast which pairs of artificial-intelligence concepts would first appear together in papers, and Marwitz et al. <sup>22</sup> used concept graphs extracted by language models to propose new research directions in materials science, which domain experts then assessed. Arman et al. <sup>23</sup> combined temporal graph embeddings and language models in a dynamic knowledge graph and evaluated the pipeline on simulated corpora. The first two studies join concepts that occur together in a paper, and none of the three studies estimated how a field would respond to an intervention. Our hypergraph types each concept by its role in the paper. A forecast therefore names which method will reach which problem, and these forecasts were scored against hard negatives and with spelling and synonym variants merged. The operators add what-if answers drawn from the same core states as the forecasts, so that both can be tested against one publication record. The training scheme and three of the operators were also validated in two further fields, semiconductors and materials science.

The baselines were matched to the training window of the forecaster, and two families of them, LightGBM and a multilayer perceptron, were tuned, because forecasting margins shrink as baselines become stronger. The publication record also has to be cleaned before it can support event studies. Most bursts detected before deduplication were re-indexed copies of one work, and event studies on the uncleaned record would largely measure indexing. Bibliographic databases that merge preprints, proceedings and journal versions are likely to need the same step.

![](images/c7c93eb292d66ab7ee002be19aa892cfe63910f9ea57778c095357d57087586a.jpg)

![](images/5e8026473d4404300071759c3e8ed2ecf80f5717839c13f5d6ce652e7b8ce2b7.jpg)

## b Concept forecasts

i Hot topics, observed and forecast  
![](images/7346c98d6260648a1a8288f4a0f04242a492a45a279be6d76a1424dcb1542d0b.jpg)

<table><tr><td>Established topics</td><td>×0.5 ×1 ×2 ×4</td><td>Predicted new link</td></tr><tr><td>data center</td><td>×1.13</td><td>← direct load control</td></tr><tr><td>large language model</td><td>×3.57</td><td>→ LFP battery</td></tr><tr><td>robotics</td><td>5 ×0.90</td><td>← deadbeat predictive</td></tr><tr><td></td><td>1</td><td>current control</td></tr><tr><td>quantum computing</td><td>×2.16</td><td>→ deception attack</td></tr><tr><td>quantum annealing</td><td>东 ×0.95</td><td>→ inverter-based resource</td></tr></table>

![](images/9b02ab9e39455326e7902cbce1d7be6f4be30d70101198f8da9163614a3077bb.jpg)

![](images/e7a34682381a177b4acb133d54ed9e9be5422530b365e88c3485042908764a88.jpg)

C Regional outlook to 2030  
![](images/5551faf4d377f51450330a2eff8c60630156d4d06a7527a581f8d506a5141091.jpg)  
f Emerging concepts born 2024–2025, entity-insertion operator

![](images/bb7516d15d46dbed13f5f9ed84d93ddb4fd00ab2e90fe7df303fe6d29c0eaade.jpg)

d Link outlook  
![](images/9fe451afc0ff6346583a3e46b98e9851f5111e7ab41f70763ee85806a14f2a82.jpg)  
e Newborn-term ranking

![](images/e962f0d7286a708226fc1cf6d7b41f6fb46ee554749b1c3076dac59ebd2ff119.jpg)

![](images/3ec7ec6627a8d936dff76c9cffbf2edaac83243f0af425871f9fa1ecc85c03a2.jpg)

g Forecasts by role  
![](images/c1e764ecdc587a6d61028cde00115a84dd5d39e04f102416c5373dfab0167454.jpg)

![](images/ecf5a7664434362d718d1837dcaf3c4392f97952c50f54a5367bfd0338f60190.jpg)

![](images/95022974c65679315355a8ab64bc8a2c5e18d8cc9fad7806d14dfdc64c846daa.jpg)  
Fig. 6 | Forecasts of power engineering. Forward outputs are from December 2025 (second origin March 2026) Rankings are those of the growth forecaster, and levels, intervals and era bias bands those of the five-run ensemble. a, a(i), zoom of the field map (layout as Fig. 1b) on the window with the most forecast new links (base map, current forecast). Color, 24-month forecast versus 2025. Size, papers in 2025. 1–4, the highest-ranked concepts with at least 20 papers in 2025. A–F, method→target links forecast at December 2025 that no paper before 2026 connects (KAN, Kolmogorov–Arnold network; PPO, proximal policy optimization). Fill, run spread; strips, sub-region forecasts at 12–60 months; rings, concepts with predicted new links. a(ii), share forecast to rise. b, b(i), hot topics, observed and forecast with 80 % interval and era bias band. b(ii), established topics and emerging terms first used in power and energy research in 2023 or later (entity-insertion operator), with each topic’s top predicted new link at December 2025 (arrow, from method to target; new, target outside the 5 % of targets linked to the most methods, connected to the topic by no paper before 2026). LFP, lithium iron phosphate; LLM, large language model. b(iii), highest- and lowest-ranked concept per role. c, Regional outlook to 2030: share of mentions relative to 2025, 2030 multiplier and share change by region and sub-region (abbreviations: batt., battery; comput., computational; conv., converter; dyn., dynamics; elem., element; optim., optimization; part., particle; pwr., power; stor., storage; sys., system). d, The 200 likeliest unlinked pairs. d(i), regions connected. d(ii), share within one region (82 %) and share of the top 200 unseen together in the corpus under three nested criteria (exact names; variants merged, spelling and synonym variants; no name containment, also no paper with concepts whose names contain both). d(iii), d(iv), methods and targets with most links. $\mathbf { e } ,$ Newborn terms of 2024–2025 (corpus) whose first use in power and energy research is 2023 or later (literature check), ranked by the entity-insertion operator from March 2026: take-of percentile in the birth cohort (dotted, median) and rank of the 24-month growth forecast. Diamonds, two hypothetical terms. f, Newborns of 2024–2025 (corpus) whose first use in power and energy research is 2023 or later (literature check). f(i), top newborn per role. f(ii), distribution. g, g(i), 24-month forecasts by role. g(ii), share forecast to rise by horizon.

An analysis of funding shows why a forecast alone cannot answer a what-if question. The apparent efect of a funder entering a topic was matched by its placebo, which indicates that funders enter topics that are already growing, and the registry returns no estimate for funding. To be answered with an identified efect, a type of action needs its own natural experiment and a statement of how far its efect is identified. Questions outside the registry are answered by programs over state primitives, and these answers carry the badge “generated” (Methods). Faithfulness was highest when the executor produced every number and the language models only translated questions and wrote answers from its output. The same division of labor may suit other systems that put a language interface on a quantitative model.

For users of the answers, the main limitation is that the operators rank efects well but underestimate their size. Efects and simulated scenarios are reported as rankings and directions. Concept intervals are calibrated only for the most active concepts. The faithfulness audits rest on an independent judge model, which may favor fluent text<sup>34,35</sup>. Two further judges, one of them from an unrelated model family, rated the answers higher but agreed only weakly with it on individual answers. The faithfulness scores therefore depend on a single judge model, whereas the forecasts were scored against observed data. Changes in inflow from neighboring fields are answered only as extrapolations, and answers to questions outside the registry are not validated against natural experiments. The full system exists only for power engineering. Links between concepts other than a method and its target are forecast only as first co-occurrence, without a relation type, and there the readout’s margin over LightGBM on the same pair features is small. For new terms the gap is larger: the entity-insertion operator ranks the fast-growing terms of the AI era first but underestimates how large they become, and three pre-registered corrections did not change this. We attribute the gap to a take-of that lies outside the support of the training data. We will score the December 2025 forecast against the papers published in 2026 and 2027 as these data become available.

## Methods

## Corpus and field state

Power-engineering works were harvested from 33 OpenAlex topics <sup>36</sup>, with every article or preprint published from 2000 onwards. After cleaning and excluding seven records dated after 2026, 1,196,720 abstracts remain. Deduplication by canonical work reduces them to 1,142,498 papers, 1,108,269 of them published in 2000–2025. With the citation layer the corpus holds 1.55 million papers and 11.13 million within-corpus citation events. The reader, a Qwen2.5-7B <sup>37</sup> student low-rank-adapted from a 27-billion-parameter teacher, extracts concepts from the title and abstract, each with one of five roles (problem, method, object, setting, metric), and the typed relations between them, using greedy decoding and up to 400 new tokens. Concepts with at least five papers form a vocabulary of 204,287. Deduplication removed 54,222 duplicate records, and year-only dates were treated as a completeness problem.

The state $S _ { t }$ of the research world model (equation (1)) comprises the deduplicated typed hypergraph at month t and the concept states that the growth and link cores compute from it. The hypergraph holds role-specific concept counts, typed method→problem and method→object edges and relations, an external-supply layer (each term’s use in 58.5 million papers of seven other OpenAlex fields) and a citation-attention layer (the citations received by each concept’s papers). Concept text embeddings are time-respecting yearly 64-component principal-component projections of SPECTER2 vectors <sup>38</sup>.

## Growth and link cores

Both cores read the 100,000 most active concepts in K = 8 graph snapshots spaced 6 months apart. Two message-passing layers over co-occurrence and typed-link neighbors <sup>39</sup> feed a temporal transformer<sup>40,41</sup> over node states. Training runs for 6 epochs, and each head is calibrated separately. The growth core has heads at H = 12, 24, 36 and 60 months. The link core adds a graph stream, joined to the temporal stream through a learned gate, concept text and a typed-link head that adds a linear floor on 60 pair features to a residual network on the two concept states. The cores are trained separately because joint training degraded both.

In the default scheme every head is trained only on anchors with $t + 6 0 \leq c$ . Horizon-specific training cutofs extend the training anchors to $t + 1 2 \leq c$ for cutof c. The loss of each H-head is then masked to anchors with $t + H \leq c \ ( \mathrm { l i n k s : ~ } t + 2 4 \leq c )$ , and each head is calibrated on its last 12 observed anchors.

One readout on the trained states of each link-core run scores candidate pairs for all 15 combinations of the five concept roles, and the scores of the five runs are averaged. For method→problem and method→object links the score is the core’s typed-link score plus α times a residual network (two layers of 256 units, last layer initialized at zero). The residual receives the products and absolute diferences of the two concepts’ graph-stream states, 62 pair-law features, the core’s score and the logarithm of its rank among the candidates of the anchor, and SPECTER2 embeddings of the two concept names. It is trained with a softmax loss over the candidate list of each of 11 anchors from 84 to 24 months before the cutof. Each list is a uniform sample of 200,000 pairs from the full candidate list, and positives therefore keep their natural rate. The weight $\alpha \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ } is chosen at the anchor 24 months before the cutof, with the residual refitted on earlier anchors. The other 13 role pairs, and the first co-occurrence of methods with problems or objects, are scored by additive pair heads on the same states.

## Operators

The operators apply the actions $a _ { t }$ of equation (1) to the state or read their efects from it. Because a new term has no history, the entity-insertion operator, a five-seed bagged LightGBM, uses early activity, external supply, the references of the first papers and a structural prior: the growth head of the link cores applied after the term is attached to its three text-nearest parents of the same role. Conformalized quantile regression<sup>31</sup> gives its 80 % intervals, and take-of is reported only as a cohort percentile. Once papers on the term exist, the observation-assimilation operator, also a five-seed LightGBM, reads its real core state at ages 0, 3, 6 and 12 months, early activity, supply and the out-of-fold entity-insertion prior. Single papers are routed to the event-readout operator, a text-aware network that predicts 36-month citations, reuse of first-use pairs and disruption <sup>26,42</sup>.

The intervention operator averages diference-in-diferences scores from LightGBM, a perceptron and counterfactual regression <sup>43</sup>, all trained on genuine bursts (≥ 2 distinct works in 3 months, 5 matched controls). Its level is a doubly robust<sup>44</sup> estimate anchored to a fixed pool of 388 reference events (186 at the December 2015 origin). Its predictions are deterministic. Bursts are detected on the deduplicated record, and their outcomes, like the default-scheme core states that the operator reads, are derived from the graph before deduplication. The transition operator is a separately trained network with the core architecture (graph encoder and temporal transformer), applied autoregressively in negative-binomial 6-month steps with constant drift and shared random numbers. Its graph grows at a follow-on rate calibrated in the training era (11.3 method-side and 14.7 target-side pairs per genuine burst). No operator is gradient-coupled to the cores. The simulations use cores trained in the default scheme.

## Interfaces

The fine-tuned decoder is one base model, Qwen3.8-27B<sup>45,46</sup>, a 27-billion-parameter language model, with three low-rank adapters: actions, clarification, and routing and compiling (described below). For registered types it is served with vLLM <sup>47</sup> in FP8 and prompted with the schema, rules and six examples. In two passes, the actions adapter (QLoRA<sup>33</sup> rank 16, trained on 3,084 rows that include multi-action questions) supplies the actions and the clarification adapter supplies the clarification note. A name is grounded by exact match or, failing that, to the five most similar concepts of the same role, with similarity weighted 0.6 on character-n-gram TF-IDF and 0.4 on SPECTER2 embeddings. A language-model variant judge maps spelling and synonym variants of new terms to existing concepts. The deterministic action executor runs scenario and baseline with shared random numbers, retrieves k = 5 historical analogs and attaches the badge of the action type. It describes changes in words $( | d | < 0 . 0 5$ “roughly flat”, 0.05–0.18 “slight”, ≥ 0.18 “clear”) and labels the era bias band explicitly as a bias. The grounded answer generator (fine-tuned writer) is the same base model fine-tuned by supervised fine-tuning, self-play and four rounds of expert iteration<sup>48</sup>, the last two ranked by judges distilled from the independent judge model. It writes six fixed sections conditioned only on the executor’s result. The independent judge model (Claude Opus, Anthropic) scored faithfulness (1–5), claim support, over-claiming and section completeness against the executor output, seeing each answer in a separate call without the question or the arm label. It also audited the level-2 programs, and none of the reported efect sizes or forecasts was computed by it. Questions outside the registry are routed by the routing and compiling adapter of the fine-tuned decoder (rank 16) to programs over six state primitives (injected papers, removed papers, scaled inflow, an added concept, adoption targets from the target proposer and a constant release efect). The adapter was trained on 5,003 routes and programs written by a general-purpose language model. The operators execute these programs as paired scenarios under the badge “generated”. An added concept is forecast by the entity-insertion operator, and answers carry the historical analogs of each injected pair, retrieved as for registered actions. On 449 new questions its routing was exact for 0.878, against 0.768 for a general-purpose language model given the same registry. Realized efects of the programs were validated for adoption targets and, in level, for releases. These tests are counted separately from the 42 tests of the evaluation protocol.

## Evaluation protocol

The cores were evaluated at walk-forward cutofs from December 2013 to December 2021 in two-year steps. The operators were evaluated at December 2015 and 2019, with 2013 and 2017 as supplements. The evaluation cutofs, the last month with complete data (March 2026) and the origin of the prospective forecast (December 2025) lie on one walk-forward time axis (Fig. 2a). “Test” denotes events after the cutof, used neither for training nor for any choice of model or hyperparameters. The growth metric is the Spearman correlation over the 1,000 most active concepts. Restricting it to these concepts excludes the regression to the mean that dominates the growth rates of rarely mentioned concepts. For links the metric is typed average precision. Positives are first-time typed method→problem or method→object links formed within 24 months of an anchor, and negatives are sampled method–problem or method–object pairs not linked by then, either overall or restricted to hard negatives (pairs without a common neighbor at the anchor). For links among all role pairs, the metric is average precision over all new candidate pairs, the pairs among the 1,000 most active concepts of each role that no earlier paper had joined under any variant of their names, with a link counted at the first paper that mentions both concepts within 24 months. Labels are scored only if $t + H \leq$ March 2026, because truncated windows inflate long-horizon advantages (by about 0.01–0.03). The baselines are default, horizon-matched and tuned LightGBM <sup>28</sup> and a tuned perceptron on the same node features, 11 link-prediction baselines for links among all role pairs and Qwen3.8-27B<sup>45,46</sup> used zero-shot.

All training runs used deterministic kernels, at about 1–1.5 % extra run time, except three growth-core runs at each of the December 2015, 2019 and 2021 cutofs. Single runs of the horizon-specific growth core have a standard deviation of 0.012–0.054 (median 0.023). Every comparison therefore bootstraps over training runs (re-ensembled) and anchors $( B = 2 , 0 0 0 { - } 4 , 0 0 0 )$ . Non-inferiority bounds were −0.005 for switching operator inputs to horizon-specific core states and −0.01 for horizon-specific cutofs at horizons not required to improve. Before every evaluation, the question, data, primary metric, decision criterion and expected power were written into a version-controlled, time-stamped pre-registration log. The reported claims rest on 42 pre-registered tests, of which 28 met their criterion, 4 met it in part and 6 failed, and the other 4 were planned analyses without a criterion. In the forecast from December 2025, concepts are ranked by the growth forecaster (a rank blend of a ten-run horizon-specific ensemble with the tuned perceptron), and levels, intervals and the era bias band (the range of calibrated bias over the validation cutofs, not a forecast range) are taken from a five-run ensemble of cores trained to March 2026. New pairs are pairs whose two concepts no earlier paper mentions together under any variant of their names and for which no earlier paper contains concepts whose names include both. Pairs and newborn terms named in the figures and text were also checked against the literature dated before 2026 (Semantic Scholar, OpenAlex and a web search, judged by a language model on the retrieved records): no earlier paper connects a named pair, and each named newborn term was first used in power and energy research in 2023 or later.

## Code availability

The complete source code for this work, including the corpus construction and concept-resolution pipeline, the world-model cores and operators, the fine-tuned decoder and executor, the evaluation and simulation scripts, and the scripts that generate the figures and tables, is available from the corresponding author upon reasonable request.

## Acknowledgments

The authors thank Oklahoma State University for providing the computational resources used in this work.

## Author contributions

F.S. conceived the idea, designed the study, developed the methods, built the data pipeline and the models, performed the experiments and analyses, and wrote the manuscript. Y.Z. and L.L. supervised the work and reviewed the manuscript.

## Competing interests

The authors declare no competing interests.

## Additional information

Correspondence and requests for materials should be addressed to Ying Zhang (y.zhang@okstate.edu).

## Supplementary Information

The Supplementary Information will be released upon publication.

## References

[1] International Energy Agency. Net Zero by 2050: A Roadmap for the Global Energy Sector. OECD Publishing, Paris, 2021. doi: 10.1787/c8328405-en.

[2] W. Brian Arthur. Competing technologies, increasing returns, and lock-in by historical events. The Economic Journal, 99(394):116–131, 1989. doi: 10.2307/2234208.

[3] Gregory C. Unruh. Understanding carbon lock-in. Energy Policy, 28(12):817–830, 2000. doi: 10.1016/S0301-4215(00)00070-7.

[4] Micah S. Ziegler and Jessika E. Trancik. Re-examining rates of lithium-ion battery technology improvement and cost decline. Energy & Environmental Science, 14(4):1635–1651, 2021. doi: 10.1039/D0EE02681F.

[5] Rupert Way, Matthew C. Ives, Penny Mealy, and J. Doyne Farmer. Empirically grounded technology forecasts and the energy transition. Joule, 6(9):2057–2082, 2022. doi: 10.1016/j.joule.2022.08.009.

[6] Jacob G. Foster, Andrey Rzhetsky, and James A. Evans. Tradition and innovation in scientists’ research strategies. American Sociological Review, 80:875–908, 2015.

[7] Brian Uzzi, Satyam Mukherjee, Michael Stringer, and Ben Jones. Atypical combinations and scientific impact. Science, 342:468–472, 2013.

[8] Timothy F. Bresnahan and Manuel Trajtenberg. General purpose technologies ‘Engines of growth’? Journal of Econometrics, 65(1):83–108, 1995. doi: 10.1016/0304-4076(94)01598-T.

[9] Iain M. Cockburn, Rebecca Henderson, and Scott Stern. The impact of artificial intelligence on innovation: an exploratory analysis. In Ajay Agrawal, Joshua Gans, and Avi Goldfarb, editors, The Economics of Artificial Intelligence: An Agenda, pages 115–148. University of Chicago Press, Chicago, 2019. doi: 10.7208/chicago/9780226613475.003.0004.

[10] Nick Johnstone, Ivan Haščič, and David Popp. Renewable energy policies and technological innovation: evidence based on patent counts. Environmental and Resource Economics, 45(1):133–155, 2010. doi: 10.1007/s10640-009-9309-1.

[11] Kyle Myers. The elasticity of science. American Economic Journal: Applied Economics, 12(4): 103–134, 2020. doi: 10.1257/app.20180518.

[12] Paul W. Holland. Statistics and causal inference. Journal of the American Statistical Association, 81 (396):945–960, 1986. doi: 10.1080/01621459.1986.10478354.

[13] Elena Verdolini, Laura Díaz Anadón, Erin Baker, Valentina Bosetti, and Lara Aleluia Reis. Future prospects for energy technologies: insights from expert elicitations. Review of Environmental Economics and Policy, 12(1):133–153, 2018. doi: 10.1093/reep/rex028.

[14] Santo Fortunato, Carl T. Bergstrom, Katy Börner, James A. Evans, Dirk Helbing, Staša Milojević, Alexander M. Petersen, Filippo Radicchi, Roberta Sinatra, Brian Uzzi, Alessandro Vespignani, Ludo Waltman, Dashun Wang, and Albert-László Barabási. Science of science. Science, 359:eaao0185, 2018.

[15] David Ha and Jürgen Schmidhuber. World models. arXiv:1803.10122, 2018.

[16] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025. doi: 10.1038/s41586-025-08744-2.

[17] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Learning skillful mediumrange global weather forecasting. Science, 382(6677):1416–1421, 2023. doi: 10.1126/science.adi2336.

[18] Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R. Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, et al. Probabilistic weather forecasting with machine learning. Nature, 637(8044):84–90, 2025. doi: 10.1038/s41586-024-08252-9.

[19] Cristian Bodnar, Wessel P. Bruinsma, Ana Lucic, Megan Stanley, Anna Allen, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan A. Weyn, Haiyu Dong, et al. A foundation model for the Earth system. Nature, 641(8065):1180–1187, 2025. doi: 10.1038/s41586-025-09005-y.

[20] Zirui Wang, Odin Zhang, Jiaqi Wang, Yijia Jiang, Zijian Carl Ma, Minghao Guo, Saleem Al Dajani, Alec Eames, Alibek Moldakozhayev, Jesse R. Poganik, et al. A world model simulates the latent dynamics of human health. Preprint at medRxiv, 2026. doi: 10.64898/2026.09.19.26363460.

[21] Mario Krenn, Lorenzo Bufoni, Bruno Coutinho, Sagi Eppel, Jacob Gates Foster, Andrew Gritsevskiy, Harlin Lee, Yichao Lu, João P. Moutinho, Nima Sanjabi, et al. Forecasting the future of artificial intelligence with machine learning-based link prediction in an exponentially growing knowledge network. Nature Machine Intelligence, 5:1326–1335, 2023. doi: 10.1038/s42256-023-00735-0.

[22] Thomas Marwitz, Alexander Colsmann, Ben Breitung, Christoph Brabec, Christoph Kirchlechner, Eva Blasco, Gabriel Cadilha Marques, Horst Hahn, Michael Hirtz, Pavel A. Levkin, Yolita M. Eggeler, Tobias Schlöder, and Pascal Friederich. Predicting new research directions in materials science using large language models and concept graphs. Nature Machine Intelligence, 8:535–544, 2026. doi: 10.1038/s42256-026-01206-y.

[23] Mithila Arman, Md. Farhad Kabir, Fakir Mashuque Alamgir, Kazi Aklima, Mohammad Kasedullah, Efaz Kabir, Abdullah Rakib Akand, Abdullah Al Imtihan, and Miguel Ángel Rodríguez Sánchez. Forecasting the evolution of scientific knowledge: a dynamic knowledge graph approach integrating temporal embeddings and large language models. Machine Learning, 115:148, 2026. doi: 10.1007/ s10994-026-07095-x.

[24] Guido W. Imbens and Donald B. Rubin. Causal Inference for Statistics, Social, and Biomedical Sciences. Cambridge University Press, 2015.

[25] Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, and David T. Mellor. The preregistration revolution. Proceedings of the National Academy of Sciences, 115:2600–2606, 2018.

[26] Russell J. Funk and Jason Owen-Smith. A dynamic network measure of technological change. Management Science, 63:791–817, 2017.

[27] Laurens van der Maaten and Geofrey Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research, 9:2579–2605, 2008.

[28] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. LightGBM: a highly eficient gradient boosting decision tree. In Advances in Neural Information Processing Systems 30, 2017.

[29] Lada A. Adamic and Eytan Adar. Friends and neighbors on the Web. Social Networks, 25:211–230, 2003.

[30] David Card and Alan B. Krueger. Minimum wages and employment: a case study of the fast-food industry in New Jersey and Pennsylvania. American Economic Review, 84:772–793, 1994.

[31] Yaniv Romano, Evan Patterson, and Emmanuel Candès. Conformalized quantile regression. In Advances in Neural Information Processing Systems 32, 2019.

[32] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

[33] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: eficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems 36, 2023.

[34] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36, 2023.

[35] Arjun Panickssery, Samuel R. Bowman, and Shi Feng. LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems 37, 2024.

[36] Jason Priem, Heather Piwowar, and Richard Orr. OpenAlex: a fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833, 2022.

[37] Qwen Team. Qwen2.5 technical report. arXiv:2412.15115, 2024.

[38] Amanpreet Singh, Mike D’Arcy, Arman Cohan, Doug Downey, and Sergey Feldman. SciRepEval: a multi-format benchmark for scientific document representations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

[39] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems 30, 2017.

[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30, 2017.

[41] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael Bronstein. Temporal graph networks for deep learning on dynamic graphs. arXiv:2006.10637, 2020.

[42] Michael Park, Erin Leahey, and Russell J. Funk. Papers and patents are becoming less disruptive over time. Nature, 613:138–144, 2023.

[43] Uri Shalit, Fredrik D. Johansson, and David Sontag. Estimating individual treatment efect: generalization bounds and algorithms. In Proceedings of the 34th International Conference on Machine Learning, 2017.

[44] James M. Robins, Andrea Rotnitzky, and Lue Ping Zhao. Estimation of regression coeficients when some regressors are not always observed. Journal of the American Statistical Association, 89:846–866, 1994.

[45] Qwen Team. Qwen3.8-27B-FP8: model card. Hugging Face, https://huggingface.co/Qwen/ Qwen3.8-27B-FP8, 2026.

[46] Qwen Team. Qwen3 technical report. arXiv:2505.09388, 2025.

[47] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, 2023.

[48] Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. In Advances in Neural Information Processing Systems 30, 2017.