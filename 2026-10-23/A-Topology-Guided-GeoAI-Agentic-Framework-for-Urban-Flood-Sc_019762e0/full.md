# A Topology-Guided GeoAI Agentic Framework for Urban Flood Screening

Yutian Tao, Haoan Feng, Xin Xu, and Leila De Floriani

{tyt16888, hfengac, xinxu629, deflo}@umd.edu

University of Maryland

College Park, Maryland, USA

## Abstract

Open-world geographic artificial intelligence (GeoAI) agents can transform public geospatial data into accessible decision support for real-world challenges, but must balance autonomous reasoning with reliable spatial analysis and human oversight. We present a topology-guided GeoAI agentic framework for urban flood screening that takes a place name, drawn region, or uploaded boundary and returns ranked candidate basins with an interactive map and report. A deterministic controller retrieves public elevation, rainfall, soil, land-cover, exposure, and infrastructure layers, extracts terrain depressions with discrete Morse theory, and ranks basins with a rel ative flood-priority score. The language model acts only at recorded decision points: selecting a persistence threshold from a precomputed candidate table, drafting recommendations from computed evidence, and reviewing each recommendation against the same evidence. Three human-in-the-loop checkpoints let users revise the threshold, factor weights, or recommendation. In a College Park, Maryland case study, the selected threshold reduced 278 raw terrain minima to 70 basins, with 77% of these basins’ lowest points also identified as sinks by a separately computed eight-direction (D8) flow-routing model. Across four further study areas, agreement ranged from 61% to 83%. By combining open data, explicit spatial reasoning, traceable evidence, and human oversight, the framework enables transparent first-pass flood screening while keeping consequential recommendations inspectable and user-controlled.

## CCS Concepts

• Information systems → Geographic information systems; • Computing methodologies → Planning and scheduling; • Applied computing → Environmental sciences.

## Keywords

GeoAI, agentic frameworks, urban flood screening, topological data analysis, human-in-the-loop systems

## ACM Reference Format:

Yutian Tao, Haoan Feng, Xin Xu, and Leila De Floriani. 2026. A Topology-Guided GeoAI Agentic Framework for Urban Flood Screening. In The 1st ACM SIGSPATIAL Student Challenge on Open Agents with Spatial Intelligence for Social Good (OASIS ’26), November 03–06, 2026, Riverside, CA, USA. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3849739.3856761

![](images/8ad00bd54f9153dfd2babac35f16a1e13781a2b6e1291870810d8fc391deef22.jpg)

## 1 Introduction

Urban flooding can disrupt roads, buildings, and drainage infrastructure when heavy rainfall overwhelms local drainage capacity. A terrain-based flood assessment usually requires specialist hydrological modeling, licensed software, and data preparation that smaller municipalities, campuses, and community organizations may not be able to support [8, 9]. These institutions need a screening tool that identifies locations worth closer inspection and exposes the assumptions behind each ranking.

Terrain-based screening begins with a digital elevation model (DEM). Topological data analysis summarizes the terrain basins and drainage structure of this elevation field. In the discrete Morse analysis used here, every local minimum of the DEM except the global minimum is paired with the saddle at which its basin merges with a deeper basin [3, 4]. The elevation diference between the paired minimum and saddle is the persistence of that basin. A persistence threshold is the smallest persistence at which a basin is retained. Basins below the threshold are removed as noise. After simplification, the cells that drain toward each retained minimum form a Morse basin, and we call it a retained basin when the distinction from removed depressions matters.

Choosing the threshold is the first technical gap. A DEM of an urban area contains hundreds of shallow basins produced by measurement noise and small surface features. Because persistence depends on terrain relief, DEM resolution, and noise characteristics, no single threshold applies across study areas [1]. The threshold must therefore be chosen per study area, and the choice must be recorded, as it determines every downstream count and ranking.

The second gap concerns hydrological relevance. Persistence captures topological prominence but does not ensure physically plausible runof. We therefore use D8 flow routing as an independent check [2, 7]. For each retained basin, we test whether its minimum is also a D8 sink, a cell with no downslope neighbor, and compare its Morse contributing area, the area assigned to that basin, with the D8flow-accumulation area, the upstream area whose D8 flow path passes through the same minimum. Together, these checks measure consistency between two representations derived from the same DEM.

Existing agentic approaches (e.g., ReAct[15], Plan-and-Solve[14], and Reflexion[10]) allow language models to interleave reasoning, planning, and self-critique. For flood screening, however, openended reasoning can make numerical results depend on model outputs. We therefore keep geospatial computation deterministic and restrict the language model to recorded decision points for threshold selection, evidence-grounded recommendation, self-review, and question answering. Users can intervene at three human-in-theloop (HITL) checkpoints. Our contributions are threefold:

![](images/400b04c585105a568d266ab4e7a2c63465c499a61bb341b1a2fb1c709e03a0a4.jpg)  
Figure 1: Workflow of the framework with intermediate results. Deterministic stages retrieve and align data, extract and rank Morse basins, and compare them with D8 routing and with re-ranking under alternative weights. Separate language-model calls select the persistence threshold, draft the recommendation, review it against the recorded evidence, and answer later user questions. Purple arrows mark the three human-in-the-loop (HITL) checkpoints.

• A GeoAI agentic framework that combines predefined queries to public data services, deterministic terrain analysis, and languagemodel decisions restricted to recorded choice points.

• A topology-guided screening method that selects a persistence threshold from a candidate table, ranks basins with a relative flood-priority score, and checks the result against a D8 flowrouting model of the same DEM.

• An evaluation across five study areas that reports threshold selection, D8 agreement, first-rank stability under four weighting schemes, and the behavior of self-review and user feedback.

## 2 GeoAI Agentic Framework

## 2.1 System Overview and Data Acquisition

Figure 1 summarizes the framework. A run starts from a place name, user-drawn polygon, or uploaded boundary, with optional priorities, ranking weights, and design-storm return period (default: 100 years). The framework retrieves and aligns public data, extracts and ranks terrain basins, performs consistency checks, and generates an evidence-grounded report. The language model is invoked only at designated decision points, while computed quantities and intermediate results are stored in an evidence bundle. For fixed inputs, numerical stages are deterministic.

Data are retrieved through predefined adapters to the public sources listed in Table 1. Elevation is required; missing optional layers are omitted or replaced by documented defaults. All layers are aligned to the elevation grid, and substitutions are recorded in the evidence bundle and report.

## 2.2 Topology-Guided Basin Extraction

The framework computes terrain topology using the Topology ToolKit (TTK) [11]. A preliminary pass collects the persistence of finite-persistence minima in the DEM. The controller then constructs candidate persistence thresholds from the 25th, 40th, 50th, 60th, 70th, 80th, 90th, 95th, and 99th percentiles of this distribution, together with a default equal to 5% of the DEM elevation range. Each candidate is associated with the number of minima it would retain.

Threshold selection is the first language-model decision. Given the candidate table, study-area name, and user-specified priority, the model selects one candidate and justifies the choice using the corresponding change in basin count. The controller then performs a second TTK pass at the selected threshold to extract the retained basins. Both the candidate-based estimate and the achieved basin count are recorded.

## 2.3 Basin Prioritization

To rank the retained basins by flood priority, each retained basin � is characterized by six normalized factors: topological prominence, contributing area, rainfall intensity, inverse local slope, exposure, and socioeconomic vulnerability. The Morse contributing area �<sub>�</sub> is the number of cells assigned to the basin multiplied by cell area. Topological prominence is derived from minimum persistence, contributing area is normalized by the largest retained basin, rainfall intensity by the maximum value in the design-storm table, and exposure by the range of the exposure raster. Inverse slope gives higher values to flatter basin floors. Missing slope, exposure, or poverty-rate values receive 0.5, representing a neutral value that avoids treating missing data as either low or high risk.

Given nonnegative weights �<sub>�</sub> summing to one, the relative flood-priority score is

$$
P _ { b } = \sum _ { j \in \{ t , a , r , s , e , v \} } w _ { j } x _ { b , j } .\tag{1}
$$

The score provides a within-run ranking rather than a calibrated flood probability or inundation estimate. By default, weights are derived using the entropy weight method [16], which assigns greater weight to factors with greater variation across retained basins. Because rainfall is spatially uniform within a run, it receives zero entropy-derived weight. Users may instead select a fixed weight preset or provide custom weights. A weight-sensitivity check rescores the same retained basins under three fixed weighting presets (balanced, life-safety, infrastructure-protection) and the entropy weights above, then reports whether the top-ranked basin is stable across all four schemes.

Table 1: Public data sources used by the framework.
<table><tr><td>Data category</td><td>U.S. source</td><td>Non-U.S. source</td></tr><tr><td>Elevation</td><td>USGS 3DEP</td><td>Copernicus GLO-30</td></tr><tr><td>Land cover</td><td>NLCD</td><td>Sentinel-2 10m</td></tr><tr><td>Exposure</td><td>Census block-group population density</td><td>WorldPop 100m</td></tr><tr><td>Infrastructure</td><td>OSM roads, buildings, water</td><td>OSM roads, buildings, water</td></tr><tr><td>Rainfall</td><td>NOAA Atlas 14</td><td>N/A</td></tr><tr><td>Soil</td><td>USDA SSURGO</td><td>N/A</td></tr><tr><td>Socioeconomic vulnerability</td><td>Census-tract poverty rate (ACS)</td><td>N/A</td></tr><tr><td>Stormwater pipes</td><td>Local government data</td><td>N/A</td></tr><tr><td>Flood zones</td><td>FEMA flood-hazard data</td><td>N/A</td></tr><tr><td>Stream gauges</td><td>USGS stream-gauge service</td><td>N/A</td></tr></table>

## 2.4 Screening-Level Runof and Consistency Checks

The framework additionally estimates screening-level peak runof to support recommendation generation, while this quantity is not included in the priority score in Equation 1. For each basin, peak discharge under return period � is estimated using the Rational Method [6] as $Q _ { b } ( T ) = C _ { b } I ( T ) A _ { b } / 3 . 6$ in $\mathrm { m } ^ { 3 } / s ,$ , where � is the storm return period in years, �(�) is the corresponding design-storm rainfall intensity in mm/h, $A _ { b }$ is the basin’s contributing area in $\mathrm { k m } ^ { 2 } .$ , and $C _ { b }$ is a dimensionless runof coeficient derived from soil and land cover using the SCS Curve Number procedure [13]. The framework integrates $Q _ { b } ( T )$ across the 2-, 10-, 25-, 50-, 100-, and 500-year scenarios using the trapezoidal rule with AEP, 1/�, as the integration variable, following the approach used in the expected annual damage framework [12]. The resulting metric is provided as supporting evidence for recommendation generation rather than as a component of the flood-priority score.

For consistency checking, the controller computes D8 flow routing on the same elevation grid. It compares whether retained Morse minima coincide with D8 sinks and compares Morse contributing area with D8 flow-accumulation area at the corresponding minimum cells. The first comparison is summarized by the D8 sink-agreement rate; the second uses Pearson correlation and the coeficient of determination (�<sup>2</sup>) after log transformation. These checks provide independent diagnostics of the topology-derived basins and are also performed across additional study areas.

## 2.5 Recommendation and Human Oversight

Recommendation generation and evidence-grounded self-review are separate language-model calls. The drafting call receives the evidence bundle and user priority, while the review call checks numerical consistency, unsupported certainty, and omitted data limitations. Detected problems trigger either textual correction or a bounded re-analysis of threshold selection or basin ranking, followed by regeneration and review.

Human oversight is provided through three checkpoints covering threshold selection, basin ranking, and recommendation text. Users can revise the selected threshold, factor weights, or recommendation, respectively. Language-model failures do not modify completed numerical results: threshold selection falls back to the default candidate, failed reviews are marked as unreviewed, and unavailable analytical checks are explicitly reported.

## 3 Evaluation

We evaluate the framework along three dimensions: terrain simplification, cross-model spatial consistency, and robustness of the final ranking and recommendation. College Park, Maryland serves as the end-to-end case study, while spatial consistency is additionally evaluated in four study areas: Annapolis and Baltimore, Maryland, and Richmond and Alexandria, Virginia. The language model calls used Google Gemini-3.5-flash [5].

The College Park run used the user-specified priority "protecting the university campus and nearby residential streets from flooding." Ten data categories returned data except stream gauges: a 192×235 USGS 3DEP elevation grid, NLCD land cover, 26 FEMA flood-zone features, a 100-year rainfall intensity of 79 mm/h, hydrologic soil group C, US Census block-group population density, US Censustract poverty rate, 5,936 stormwater pipe segments, 12,138 roads, 20,948 buildings, and 291 water features. The USGS gauge query returned no active gauge within the study area, which the report records as a verified empty result.

The first TTK pass identified 278 raw local minima. Ten candidates were evaluated, spanning the 25th–99th persistence percentiles plus a default threshold of 5% of the elevation range, with estimated basin counts ranging from 207 to 3. The language model selected the 70th-percentile candidate, 0.508 m, with an estimated 83 retained minima. Its justification cited the drop in basin count from 83 to 55 at the next candidate (0.774 m), describing 0.508 m as capturing meaningful depressions associated with surface ponding while filtering out minor surface irregularities.

D8 routing on the same grid identified 222 raw sinks, compared with 278 raw Morse minima. Of the 70 retained basins, 77% had a minimum cell that was also a D8 sink. The minimum cells of the remaining 23% of basins lie mostly on flat areas, where D8 assigns a downslope neighbor to a cell that Morse analysis retains as a minimum. The same D8 consistency checks were also performed across the additional study areas. Figure 2 summarizes sink agreement across these areas, while Figure 3 compares the D8 flowaccumulation area with the Morse contributing area.

Under entropy weighting, catchment 8 (38.977°N, 76.919°W) ranked first, with a contributing area of 0.249 km<sup>2</sup> and a relative flood-priority score of 0.61. The basin lies within FEMA’s Special Flood Hazard Area and 6.8 m from the nearest mapped water feature, providing independent spatial evidence consistent with the topological ranking. It remained first under all four weighting schemes tested: balanced, life-safety, infrastructure-protection, and entropy.

![](images/c6485ae6a302356e0e103f63d22b835eb8a58df9ca99c80895142d391409ee0a.jpg)

Figure 2: D8 sink status of retained basin minima in four study areas. Each point is one retained basin, ordered by its rank under entropy-derived weights. The vertical axis gives the D8 flow-accumulation area at the basin minimum on a logarithmic scale. Each panel reports the D8 sink-agreement rate $A _ { \mathrm { s i n k } }$ (the fraction of that area’s retained basins whose minimum coincides with a D8 sink) and the number of retained basins �.  
![](images/fc468b5698e341ef71edca7261c5a66e2a79aa27ad5d3054f1a86584711d31ca.jpg)  
Figure 3: D8 flow-accumulation area (horizontal) against Morse contributing area (vertical) for the retained basins of Figure 2, both on logarithmic scales, with a linear fit in log space, $r _ { \log } , R ^ { 2 }$ , and � per panel.

## 4 Conclusion

We presented a topology-guided GeoAI agentic framework for urban flood screening that integrates public geospatial data, deterministic spatial analysis, LLM-based reasoning, and human-in-the-loop review. Across five study areas, the framework showed consistent agreement between Morse-based terrain structures and independent D8 routing, while sensitivity analysis and self-review helped identify unstable rankings and unsupported statements.

Beyond the technical evaluation, the framework is designed to make preliminary flood-risk screening more accessible and transparent. It provides a low-cost first-pass analysis from public data, keeps recommendations traceable to their underlying spatial evidence, and allows users to inspect results through an interactive map, report, and LLM interface. Human checkpoints further ensure that consequential decisions remain reviewable rather than being delegated entirely to the language model.

The experiments also highlight important lessons and limitations. LLM-generated recommendations can overstate available evidence, reinforcing the need for explicit verification and human oversight. The current evaluation uses a limited number of study areas, U.S.- specific data services, simplified exposure representation, and no observed flood events as ground truth. Future work will therefore evaluate more geographic settings, incorporate broader and more complete data sources, compare multiple LLMs, and further assess the framework against observed or simulated flood information.

## References

[1] Omer Bobrowski and Primoz Skraba. 2023. A Universal Null-Distribution for Topological Data Analysis. Sci. Rep. 13 (2023), 12274.

[2] Sandra Dávila-Hernández, Julián González-Trinidad, Hugo Enrique Júnez-Ferreira, Carlos Francisco Bautista-Capetillo, Heriberto Morales de Ávila, Juana Cázares Escareño, Jennifer Ortiz-Letechipia, Cruz Octavio Robles Rovelo, and Enrique A. López-Baltazar. 2022. Efects of the Digital Elevation Model and Hydrological Processing Algorithms on the Geomorphological Parameterization. Water 14, 15 (2022), 2363. doi:10.3390/w14152363

[3] Herbert Edelsbrunner and John Harer. 2010. Computational Topology: An Introduction. American Mathematical Society.

[4] Robin Forman. 2002. A User’s Guide to Discrete Morse Theory. Sém. Lothar. Combin. 48 (2002), B48c.

[5] Google DeepMind. 2026. Gemini 3.5 Flash Model Card. https://deepmind.google/ models/model-cards/gemini-3-5-flash/. Accessed: 2026-09-29.

[6] Emil Kuichling. 1889. The Relation between the Rainfall and the Discharge of Sewers in Populous Districts. Trans. ASCE 20 (1889), 1–56.

[7] John F. O’Callaghan and David M. Mark. 1984. The Extraction of Drainage Networks from Digital Elevation Data. Comput. Vis. Graph. Image Process. 28, 3 (1984), 323–344.

[8] Louise Petersson, Marie-Claire ten Veldhuis, Govert Verhoeven, Zoran Kapelan, Innocent Maholi, and Hessel C. Winsemius. 2020. Community Mapping Supports Comprehensive Urban Flood Modeling for Flood Risk Management in a Data-Scarce Environment. Front. Earth Sci. 8 (2020). doi:10.3389/feart.2020.00304

[9] Dina Pirone, Roberta Padulano, and Giuseppe Del Giudice. 2026. Approaching Urban Flood Modelling According to Available Data: A State-of-the-Art Review and Future Challenges. J. Flood Risk Manag. 19, 1 (2026), e70184. doi:10.1111/jfr3. 70184

[10] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366 (2023).

[11] Julien Tierny, Guillaume Favelier, Joshua A. Levine, Charles Gueunet, and Michael Michaux. 2017. The Topology ToolKit. IEEE Trans. Vis. Comput. Graph. 24, 1 (2017), 832–842.

[12] U.S. Army Corps of Engineers. 1996. Risk-Based Analysis for Flood Damage Reduction Studies. Technical Report EM 1110-2-1619. USACE.

[13] USDA Natural Resources Conservation Service. 1986. Urban Hydrology for Small Watersheds. Technical Report TR-55. USDA NRCS.

[14] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Plan-and-Solve Prompting: Improving Zero-Shot Chainof-Thought Reasoning by Large Language Models. In Proc. 61st Annu. Meeting Assoc. Comput. Linguist. (ACL). 2609–2634.

[15] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629 (2022).

[16] Yuxin Zhu, Dazuo Tian, and Feng Yan. 2020. Efectiveness of Entropy Weight Method in Decision-Making. Math. Probl. Eng. 2020 (2020), 3564835.