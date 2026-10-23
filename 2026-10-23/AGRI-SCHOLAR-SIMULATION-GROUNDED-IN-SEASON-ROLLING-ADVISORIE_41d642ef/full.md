# AGRI-SCHOLAR: SIMULATION-GROUNDED IN-SEASON ROLLING ADVISORIES FOR AGRICULTURAL CROPS USING LLM-BOUNDED OPTIMIZATION

Geetha Charan, Vedant Balasubramaniam, Manojkumar Patil, V. Priyanka Indian Institute of Science Bengaluru, India {geethacharan, vedantb, pmanojkumar, priyankav}@iisc.ac.in

Kodur Sai Vinay Sathvik Indian Institute of Science Bengaluru, India sathvik2004@gmail.com

Rohit P Suresh   
University of Michigan   
Ann Arbor, USA   
rohitps@umich.edu

Mohammad Sufyan Azam, Y. Narahari Indian Institute of Science Bengaluru, India {mohammadazam, narahari}@iisc.ac.in

## ABSTRACT

Existing agricultural advisories to farmers suffer from several limitations: (a) remain largely expertdriven, (b) do not adequately account for in-season weather, and (c) do not scale to individual farms. In this paper, our objective is to close these gaps by developing Agri-SCHOLAR (Simulation-Grounded Crop Harvest Optimization with LLM-Guided In-Season Advisory Rollouts). Agri-SCHOLAR is a rolling, in-season advisory framework that issues, at intervals of every few days or every week, a personalized advisory using the following information: soil and crop properties, observed weather to date, short-range forecast of weather, and a season extrapolation of weather based on historical analogs. The novelty and depth of Agri-SCHOLAR lies in each advisory being produced by a closed-loop system which couples multiple LLM agents with a powerful crop simulator (such as APSIM in our case). The simulator crucially serves as an evaluation oracle. In Agri-SCHOLAR, the LLM agents propose agronomically sound and feasible bounds on actionable parameters and an optimizer searches within those bounds against a multi-objective reward spanning yield, resource utilization, and economic feasibility. We conduct a detailed 11-year analysis of the maize crop in two selected locations of South Asia and show that Agri-SCHOLAR consistently outperforms the region’s standard packages of practice and satisfactorily addresses the limitations of existing agricultural advisories.

Keywords Personalized in-season advisories · Multi-agent LLM systems · Agricultural advisory · Crop simulation Simulation-grounded reasoning · Sustainable agriculture · AI for social good

## 1 Introduction

Agricultural advisories in much of the developing world are still produced by small teams of agronomists who issue broad guidance across large, dispersed regions. This approach does not scale, as even farms in the same advisory zone differ in soil, cultivar, sowing date, and management history, and they experience the season’s weather differently. So a single recommendation is rarely optimal for an individual farm. Personalized, AI-driven advisories are a natural way to close this gap. For farmers to act on them, however, such advisories must be agronomically and physiologically feasible for the selected crop under the conditions of that particular farm.

Recent works [1] address this requirement by using crop simulators like Decision Support System for Agrotechnology Transfer (DSSAT) and Agricultural Production Systems sIMulator (APSIM) to validate LLM-generated season-long advisories. This resolves the plausibility problem but introduces a new practical problem: evaluating a season-long plan in a simulator requires the weather for the entire season in advance. No forecasting system provides this at the required fidelity for a crop cycle, which can exceed several months, so such a pipeline cannot be deployed prospectively.

Operational advisory services avoid this problem by not planning the whole season at once. For example, India’s Agro-Meteorological Field Units (AMFUs) issue advisories twice a week based on observed weather and short-range forecasts, and reissue them as the season unfolds; the agro-meteorological services of China and Vietnam follow similar practices. The underlying principle is that advice should be revised repeatedly as new weather information arrives. Existing operational advisories, however, remain expert-driven, are issued at the level of an administrative region rather than a farm, and are not checked against a mechanistic model of crop response.

Inspired by this practice, we introduce Agri-SCHOLAR, a rolling, in-season advisory framework that combines the simulator-based validation with the reissue-as-you-go principle of operational services. At each advisory step (every few days to a week), Agri-SCHOLAR evaluates candidate actions in APSIM using a weather trajectory assembled from three sources: (i) observed weather to date, (ii) the available short-range forecast, and (iii) an extrapolation of the remainder of the season based on atmospherically analogous previous years. The advisory is then recomputed at the next step with updated observations of weather, so errors due to extrapolated weather in the previous step do not accumulate over the season.

A two-stage, closed-loop pipeline produces each advisory. In the first stage, LLM agents reason about the farm’s soil, crop state, and current weather context to propose agronomically sound and feasible bounds on the actionable parameters (in this paper, the timing and amount of irrigation and fertilizer). In the second stage, an optimizer searches within these bounds to maximize a multi-objective reward that captures yield, nitrogen, and water use efficiency, and economic feasibility, with APSIM serving as the evaluation oracle for every candidate recommendation. This division of labor allows the LLM to perform semantic, context-dependent reasoning about which actions are appropriate for a given farm, while the simulator and optimizer quantitatively evaluate and compare those actions within the feasible decision space. The resulting advisories are therefore both farm-specific and constrained by the crop and environmental conditions under which they must remain feasible.

We evaluate Agri-SCHOLAR for maize in South Asia over 11 years, comparing it with the region’s standard package of practices. Across seasons, Agri-SCHOLAR consistently outperforms this baseline on the reward components above, which indicates that rolling, simulator-grounded advisories can address the limitations of existing advisory systems without requiring an accurate season-long weather forecast.

Contributions: This paper makes the following contributions.

1. We introduce Agri-SCHOLAR, which is an eminently deployable, rolling, farm-specific crop advisory system that clearly improves upon classical season-long, generic, simulation-based advisories. Agri-SCHOLAR makes information constrained decision-making with weekly, farm-specific advisories produced by an LLMbounded, simulation-validated optimization. Each weekly decision uses weather information available at that time (observations, short-range and long-range forecasts) and is re-optimized as conditions evolve, directly overcoming the fundamental foresight limitation of prior simulation-grounded advisory systems.

2. The key component of Agri-SCHOLAR is a novel LLM-bound optimization mechanism that assigns complementary roles to language agents and numerical optimization. LLMs define agronomically feasible action bounds, while the optimizer searches within those bounds using APSIM as an evaluation oracle. This jointly optimizes productivity, profitability, resource use efficiency, and sustainability, as well as prevents any gaming of the simulator by the optimizer. This jointly optimizes productivity, profitability, resource use efficiency, and sustainability while preventing optimizer gaming. Backed by a dynamic RAG base, it doesn’t require expert re-specification for new crops, regions, or seasons.

3. We provide a substantive empirical demonstration across 11 maize crop seasons in two contrasting irrigated and rainfed South Asian environments. The study demonstrates yield gains and shows improved economic and environmental outcomes. An ablation study showcases the value of agronomic LLM bounds, demonstrating that the proposed architecture drives practical, sustainable improvements.

## 2 Related Work

Most LLM-based agricultural advisory systems are grounded in retrieved text. ShizishanGPT [2] and AgriGPT [3] pair general-purpose LLMs with knowledge graphs and multi-channel retrieval. AgriRegion [4] and AgroAskAI [5] improve localization through geospatial re-ranking and multi-agent governance, respectively, to reduce contextual hallucination. Even benchmarks such as AGXQA [6], AgriBench [7], AgMMU [8], and AgEval [9] evaluate these systems on question-answering accuracy and multimodal reasoning. None of them checks whether a recommendation is physiologically viable for the crop.

A smaller line of work grounds LLM outputs in process-based crop simulators. DSSAT-LM [10] takes a neurosymbolic view, using DSSAT as a deterministic symbolic backbone while the LLM interprets questions and generates answers. Agri-SAGE [1] couples multi-agent LLM reasoning with APSIM [11] and compares Plan-and-Solve, Tree of Thoughts, and Reflexion as reasoning strategies over a 10-year retrospective in a single region, substantially improving physiological grounding over text-only advisory.

Agri-SCHOLAR builds on Agri-SAGE but differs in three ways. First, Agri-SAGE produces one season-long plan and refines it in APSIM using the observed weather of the entire season, which is unknown when the plan is made, so it cannot be deployed before or during a real season; Agri-SCHOLAR decides each week using only the information available on that date and re-optimizes as new observations arrive. Second, in Agri-SAGE the LLM proposes the management actions themselves, so the continuous space of doses and dates is explored through a handful of proposals; in Agri-SCHOLAR the LLM only sets agronomically admissible bounds, and a numerical optimizer evaluates many candidates within them in APSIM. Third, Agri-SAGE maximizes grain yield alone, whereas Agri-SCHOLAR maximizes net profit, accounting for production costs and monetized environmental penalties. Beyond these design differences, we evaluate Agri-SCHOLAR in two contrasting water regimes, irrigated and rainfed, rather than one.

Another body of work uses crop simulators as environments for learning management policies. gym-DSSAT [12] wraps DSSAT in a standard Reinforcement Learning interface, and later work trains nitrogen and irrigation policies on top of it [13, 14]. Wu et al. [15] verbalize the simulator’s state variables as text and train a language-model-based policy with deep Q-learning. Wang et al. [16] show that policies trained under normal conditions must be retrained for extreme weather, highlighting the value of adapting to in-season conditions rather than committing to a fixed season-long policy. These systems output numeric fertilization or irrigation schedules rather than explainable, farmer-facing advisories.

The unobserved remainder of a season cannot be reliably forecast either. Even the strongest machine-learned weather models, including GraphCast [17], Pangu-Weather [18], GenCast [19], and Aurora [20], are skillful only for roughly two weeks, well short of the 90-150 days of a crop cycle. Operational advisory services have adapted to this ceiling by issuing short-horizon guidance and revising it as the season unfolds. India’s Gramin Krishi Mausam Sewa issues district-level advisories twice a week from roughly 130 Agro-Met Field Units, combining observed weather, short-range forecasts, and crop stage [21]. China’s meteorological services follows a similar rolling cadence, nowcasting precipitation at 0-3 days and extending guidance to 30 days through provincial and district agrometeorological service centers. Vietnam’s agroclimatic bulletin program covering 13 provinces and roughly 290,000 farmers, issues comparable short-horizon, crop-stage-specific advisories [22]. These services are expert-driven and issued at the level of an administrative region.

Agri-SCHOLAR combines these threads. Like retrieval-grounded systems, it produces natural-language, farmer-facing advisories, but grounds them in a crop simulator. Like Agri-SAGE and DSSAT-LM, it uses a process-based simulator. However, it evaluates candidate actions on a trajectory built from observed weather, a short-range forecast, and an analog-year extrapolation, rather than assuming full-season foresight. Agri-SCHOLAR also optimizes against a simulator while restricting the search to LLM-proposed, agronomically feasible bounds, and scores candidates using a multi-objective reward that spans yield, resource-use efficiency, and economic feasibility, ensuring the outputs remain interpretable. Finally, it reissues advisories on a rolling cycle that mirrors how operational agro-meteorological units operate, while adding farm-level personalization and mechanistic validation that those services lack.

## 3 Methodology

Figure 1 shows the architecture of Agri-SCHOLAR. The pipeline has five agents. (1) The Context Retrieval Agent (CRA), (2) PreSeason Agent (PSA), (3) Advisory Bounding Agent (ABA), (4) Optimization Agent (OA), and (5) Verification and Feedback Agent (VFA).

The PSA operates at the beginning of each cropping season to establish the initial seasonal context and constraints, while the remaining four agents jointly produce a short-term advisory each week using a constructed weather. Once the advisory has been optimized and verified, it passes through a cutoff layer, is issued to the farmer, and is logged as a historical action that conditions later cycles.

## 3.1 Context Retrieval Agent (CRA)

The CRA converts numerical weather, soil, and crop-stage features into a natural-language query. It then runs hybrid (sparse and dense) retrieval over a vector database of agronomic literature and regional guidance. Its output is a compact natural-language context bundle that the downstream agents reason over.

![](images/10d1d78c4d673761a517f437408108e72cdf60525cd81b5461c5a33db31cd94e.jpg)  
Figure 1: Agri-SCHOLAR architecture.

## 3.2 PreSeason Agent (PSA)

The PSA sets the week-0 plan: the sowing date, mulch, and basal Di-Ammonium Phosphate (DAP) application. The sowing date is chosen by a rainfall-onset rule. The rule scans the period from June 1 to July 15 using a sliding three-day window and selects the first window with cumulative rainfall of at least 20 mm. The last day of that window is taken as the onset day, which serves as the sowing date (the agronomic planting day).

## 3.3 Advisory Bounding Agent (ABA)

Given the location details and the retrieved context, the ABA generates the advisory together with bounds for each management operation under consideration (e.g., plausible fertilizer amounts at the current crop stage). These bounds serve two purposes. First, they shrink the search space, because the OA searches only within the ABA’s bounds rather than the full numerically feasible space. Second, they stop the OA from exploiting the simulator, for example: unrealistically large nitrogen applications that inflate simulated yield without being physiologically or practically plausible. Grounding the bounds in literature before APSIM is queried keeps the search within actions a farmer could realistically carry out.

## 3.4 Optimization Agent (OA)

At each weekly decision point, the OA searches within the ABA’s bounds for the management action that maximizes expected economic return. Each candidate is evaluated by the VFA on the constructed weather file for the current cycle. This weather file contains only the information an advisory system would have at that point in the season, not the full observed record. The resulting APSIM simulation gives the projected crop response, from which the reward is computed.

Let $\pi = \left\{ A _ { 1 } , A _ { 2 } , \ldots , A _ { K } \right\}$ be a management policy, where $A _ { k }$ is the set of fertilizer and irrigation decisions for week k of a season with K weekly decision points. The OA seeks the policy that maximizes cumulative profit:

$$
\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } R ( \pi ) ,\tag{1}
$$

where $R ( \pi )$ is the profit per hectare over the season under π, as simulated by APSIM.

Eq. 1 is solved each week t using Optuna. At week t, all previously issued actions are fixed, and the OA jointly optimizes this week’s urea and irrigation within the ABA’s bounds together with a tentative nitrogen and irrigation schedules for the remaining weeks, constrained by the season structure set by the PSA (remaining splits, quantity, and minimum application). Each candidate is scored by a full APSIM run from sowing to harvest on the constructed weather $W _ { t }$ , using a Tree-structured Parzen Estimator with $2 5 + 4 ( K - t - 1 )$ candidate evaluations per ABA iteration. Only the week-t action is committed and issued; the tentative schedule is discarded and re-optimized the following week with updated observations.

Internally, fertilizer actions are expressed in kg $\mathrm { \Delta N h a ^ { - 1 } }$ and denoted UreaN (the quantity of elemental nitrogen applied), so UreaN is 100% nitrogen by definition. Commercial urea, in contrast, contains only 46% nitrogen by mass. So for farmer-facing advisories, each UreaN dose is therefore converted to commercial urea as follows

$$
m _ { \mathrm { u r e a } } = { \frac { m _ { \mathrm { U r e a N } } } { 0 . 4 6 } } ,\tag{2}
$$

For example, 46 kg $\mathrm { \Delta N h a ^ { - 1 } }$ of UreaN corresponds to 100 kg $\mathrm { \ h a ^ { - 1 } }$ of commercial urea. All nitrogen constraints in the optimization (minimum dose, remaining splits, and season budget) are stated in UreaN units. Throughout this paper, we use UreaN for explanations and observations, and convert to commercial urea only in the farmer-facing advisories.

## 3.4.1 Profit-Based Objective

OA optimizes for farm profit based on yield, costs and environmental impact. Environmental impacts enter as monetary sustainability penalties. The reward per hectare is

$$
R = R _ { \mathrm { r e v } } - C _ { \mathrm { p r o d } } - P _ { \mathrm { s u s t } } ,\tag{3}
$$

where $R _ { \mathrm { r e v } }$ is gross grain revenue, $C _ { \mathrm { p r o d } }$ is the production cost, and $P _ { \mathrm { s u s t } }$ is the sustainability penalty. All monetary quantities are expressed per hectare in the local currency.

Revenue and production costs. Revenue is the simulated grain yield $Y \ ( \mathrm { { k g } h a ^ { - 1 } ) }$ multiplied by the grain price. Production cost is the sum of fertilizer, irrigation, and labour costs, all of which follow directly from the actions chosen by the OA:

$$
R _ { \mathrm { r e v } } = p _ { g } Y ,\tag{4}
$$

$$
C _ { \mathrm { p r o d } } = C _ { F } + C _ { I } + C _ { L } ,\tag{5}
$$

$$
C _ { F } = \sum _ { i = 1 } ^ { n _ { F } } A _ { i } ^ { F } p _ { i } ^ { F } ,\tag{6}
$$

$$
C _ { I } = \sum _ { j = 1 } ^ { n _ { I } } A _ { j } ^ { I } p _ { w } ,\tag{7}
$$

$$
C _ { L } = \left( n _ { F } + n _ { I } \right) p _ { L } .\tag{8}
$$

Here $C _ { F } , C _ { I }$ , and $C _ { L }$ are the fertilizer, irrigation, and labour costs. $n _ { F }$ and $n _ { I }$ are the numbers of fertilizer and irrigation operations in the season. $A _ { i } ^ { F }$ is the amount of fertilizer applied in the i-th fertilizer operation (kg ha $^ { - 1 } )$ , and $A _ { i } ^ { I }$ is the amount of water applied in the j-th irrigation operation (mm). The unit prices are $p _ { g }$ for grain (per kg), $p _ { i } ^ { F }$ for fertilizer in operation i (per kg), $p _ { w }$ for irrigation water (per mm), and $p _ { L }$ for labour (per operation).

Sustainability penalties. The sustainability penalty $P _ { \mathrm { s u s t } }$ combines impacts that occur during the season $( P _ { \mathrm { s e a s o n } } )$ with impacts that appear after it $( P _ { \mathrm { p o s t } } )$

$$
P _ { \mathrm { s u s t } } = P _ { \mathrm { s e a s o n } } + P _ { \mathrm { p o s t } } ,\tag{9}
$$

$$
P _ { \mathrm { s e a s o n } } = S _ { w l } I _ { w l } + S _ { n o } I _ { n o } ,\tag{10}
$$

$$
P _ { \mathrm { p o s t } } = S _ { d d } I _ { d d } + S _ { n l } I _ { n l } + S _ { c l } I _ { c l } .\tag{11}
$$

Each term multiplies an environmental indicator I , computed from APSIM outputs, by a shadow cost $S .$ , the monetary cost assigned to one unit of that impact. The subscripts denote waterlogging (wl), nitrogen oversupply (no), deep drainage (dd), nitrogen leaching (nl), and soil organic carbon loss (cl). Waterlogging and nitrogen oversupply are penalized in-season because they directly harm the crop and waste inputs. The other three are penalized after the season because they represent water and nutrient losses beyond the root zone and the depletion of long-term soil fertility.

With T the number of days in the simulated season, the indicators are

$$
I _ { w l } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { m a x } \bigg ( 0 , ~ \frac { S W _ { t } - D U L _ { t } } { S A T _ { t } - D U L _ { t } } \bigg ) ,\tag{12}
$$

$$
I _ { n o } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { m a x } ( 0 , N _ { \mathrm { a p p l i e d } , t } - N _ { \mathrm { d e m a n d } , t } ) \ : ,\tag{13}
$$

$$
I _ { d d } = D _ { \mathrm { c m l } } ,\tag{14}
$$

$$
I _ { n l } = L _ { N O _ { 3 } } + L _ { N H _ { 4 } } ,\tag{15}
$$

$$
I _ { c l } = \operatorname * { m a x } ( 0 , S O C _ { 0 } - S O C _ { H } ) .\tag{16}
$$

Economic quantities come from the OA’s selected actions and external prices, whereas productivity and environmental indicators come exclusively from APSIM. Table 1 lists the details of each quantity.

Table 1: Quantities used in the reward and their sources.
<table><tr><td>Symbol</td><td>Meaning</td><td>Source</td></tr><tr><td>Y</td><td>Grain yield</td><td>APSIM:[Maize].Grain.Wt</td></tr><tr><td> $S W _ { t } , D U L _ { t } , S A T _ { t }$ </td><td>Soil water, drained upper limit, and saturation of the profile on day t</td><td>APSIM: [Soil].Physical. [SWmm/DULmm/SATmm]</td></tr><tr><td> $N _ { \mathrm { d e m a n d } }$ </td><td>Crop nitrogen demand, summed over leaf, stem, root, and grain organs</td><td>APSIM: sum of NDemand. Total over the organs</td></tr><tr><td> $D _ { \mathrm { c m l } }$ </td><td>Cumulative deep drainage</td><td>APSIM: [WaterBalance].Drainage</td></tr><tr><td> $L _ { N O _ { 3 } } , L _ { N H _ { 4 } }$ </td><td>Cumulative nitrate and ammonium leaching</td><td>APSIM: [WaterBalance].</td></tr><tr><td> $S O C _ { 0 } , S O C _ { H }$ </td><td>Soil organic carbon (SOC) at the start and end of</td><td>[LeachN03/LeachNH4] APSIM: [Soil].Organic.Carbon</td></tr><tr><td> $C _ { F } , C _ { I } , C _ { L }$ </td><td>the horizon Fertilizer, irrigation, and labour costs</td><td>OA: computed from applied inputs, number of</td></tr><tr><td></td><td></td><td>operations, and unit prices</td></tr><tr><td> $p _ { g } , p _ { i } ^ { F } , p _ { w } , p _ { L }$ </td><td>Unit prices of grain, fertilizer, water, and labour</td><td>External: regional or market prices</td></tr></table>

## 3.5 Verification and Feedback Agent (VFA)

The VFA translates each candidate action into APSIM tool calls and runs the simulation on the constructed weather file for the current cycle. It returns the resulting metrics (grain yield, soil water status, nitrogen balance, drainage, and leaching) to the OA, which computes the reward. It also returns them to the ABA, so that bounds can be tightened or relaxed if they repeatedly yield infeasible or dominated candidates.

## 3.6 Iterative Loops

Agri-SCHOLAR runs three nested loops at different frequencies.

• Inner loop (OA-VFA). The OA proposes a candidate, the VFA scores it in APSIM, and the OA updates its search.

• Middle loop (ABA-OA/VFA). When the inner loop converges, or fails to find a satisfactory candidate, the ABA uses the VFA’s feedback to revise the bounds through self-reflection, and the inner loop is repeated within the revised bounds.

• Outer loop (weekly). The full pipeline reruns from scratch each week with the past week’s observed weather, a new short-range forecast, and an updated analog tail, producing a new advisory.

## 3.7 Weather Array Construction

To emulate the information available to an operational advisory system, we construct the weather as follows: weather up to current date, a noisy seven-day forecast, and an analog-based estimate for the remainder of the season.

Let the crop be sown on day S, with season length D days. At week $t \geq 0$ after sowing, the weather file supplied to APSIM is

$$
\begin{array} { r } { W _ { t } = C _ { \mathrm { p r e } } \parallel O _ { 1 : 7 t } \parallel F _ { 7 t + 1 : 7 t + 7 } \parallel \hat { A } _ { 7 t + 8 : D } , } \end{array}\tag{17}
$$

where

$C _ { \mathrm { p r e } }$ is observed weather from January 1 to day $S - 1$

$O _ { 1 : 7 t }$ is observed weather from sowing through the current day;

$F _ { 7 t + 1 : 7 t + 7 }$ is a synthetic seven-day forecast obtained by perturbing the true weather;

$\hat { A } _ { 7 t + 8 : D }$ is the analog tail, an observed weather sequence from the historically most similar season.

The segment F mimics a real seven-day forecast by adding errors to the observed weather that scale with lead time. All other segments rely strictly on data available on the decision date, preventing the model from seeing any seasonal information beyond the seven-day horizon.

## 3.7.1 Synthetic Seven Day Forecast

The forecast segment $F _ { 7 t + 1 : 7 t + 7 }$ is generated by perturbing the true weather with stochastic errors that grow with lead time. For forecast lead time $\tau \in \{ 1 , \ldots , 7 \}$ days, precipitation is simulated first because it conditions the temperature and radiation errors.

Precipitation. Precipitation follows the wet/dry occurrence framework of Richardson’s stochastic weather generator [23]. First, the day’s wet/dry state is flipped with probability

$$
P _ { \mathrm { f l i p } } ( \tau ) = 0 . 0 5 \tau ^ { 0 . 7 } .\tag{18}
$$

If the resulting day is dry, precipitation is set to zero. Otherwise, the amount is sampled from a Gamma distribution, following the Weather Generator (WGEN) [24]:

$$
\mathrm { P } _ { f } ( \tau ) \sim \mathrm { G a m m a } \Bigg ( \mathrm { s h a p e } = \frac { 1 } { \sigma _ { p } ^ { 2 } \tau ^ { b } } , \mathrm { s c a l e } = \mathrm { P } _ { \mathrm { t r u e } } \sigma _ { p } ^ { 2 } \tau ^ { b } \Bigg ) ,\tag{19}
$$

where $\mathrm { ~ P ~ } _ { f }$ and $\mathrm { P _ { t r u e } }$ denote forecast and true precipitation. This parameterization preserves the expected precipitation, $\mathbb { E } [ \mathrm { P } _ { f } ( \tau ) ] = \mathrm { P } _ { \mathrm { t r u e } } ,$ while letting the relative uncertainty grow with lead time.

To propagate precipitation uncertainty to the other variables, we define the outcome $\delta _ { \mathrm { p r e c i p } } ( \tau ) \in \{ - 1 , 0 , + 1 \}$ . Here +1 denotes a false alarm or overprediction, −1 a missed or underpredicted event, and 0 a preserved precipitation state.

Temperature. Temperature errors are additive Gaussian with lead-time-dependent variance and a small mean shift conditioned on $\delta _ { \mathrm { p r e c i p } } ( \tau )$

$$
\begin{array} { r l } & { T _ { f } ( \tau ) = T _ { \mathrm { t r u e } } ( \tau ) - \delta _ { \mathrm { p r e c i p } } ( \tau ) \mu _ { T } ( \tau ) + \epsilon _ { T } ( \tau ) , } \\ & { \qquad \epsilon _ { T } ( \tau ) \sim \mathcal { N } \bigl ( 0 , \sigma _ { T } ^ { 2 } \tau ^ { 2 \beta } \bigr ) . } \end{array}\tag{20}
$$

The parameters $\sigma _ { T }$ and $\beta$ are estimated from the empirical validation of the U-Cast probabilistic forecaster [25]. Converting the reported day-1 and day-7 temperature Continuous Ranked Probability Score into equivalent standard deviations gives a base uncertainty of $\sigma _ { T } \approx \mathrm { { \bar { 0 } } } . 9 3 ^ { \circ } \mathrm { { C } }$ and a lead-time growth exponent of $\beta \approx 0 . 3 \bar { 6 }$ . The mean shift $\mu _ { T } ( \tau )$ (approximately $0 . 3 { - } 0 . 8 ^ { \circ } \mathrm { C } )$ is calibrated in this study and applied only when the precipitation state changes, reflecting the effect of cloud cover on surface temperature.

Solar radiation. Solar radiation is perturbed in clear-sky-index space rather than directly in irradiance. The clear-sky index is

$$
\kappa = { \frac { \mathrm { R a d } _ { \mathrm { t r u e } } } { \mathrm { R a d } _ { \mathrm { c s } } } } ,\tag{21}
$$

where $\mathrm { R a d _ { c s } }$ is the clear-sky irradiance from the Ineichen-Perez model [26]. Working in this normalized space removes predictable seasonal and astronomical variation and keeps perturbed values physically bounded. Gaussian forecast error is then applied:

$$
\kappa _ { f } ( \tau ) = \mathrm { c l i p } \Big ( \kappa _ { \mathrm { t r u e } } ( \tau ) - \delta _ { \mathrm { p r e c i p } } ( \tau ) \mu _ { \kappa } ( \tau ) + \epsilon _ { \kappa } ( \tau ) , 0 , 1 . 1 \Big ) ,\tag{22}
$$

$$
\epsilon _ { \kappa } ( \tau ) \sim { \mathcal N } \big ( 0 , \sigma _ { \kappa } ^ { 2 } \tau ^ { 2 \beta } \big ) ,\tag{23}
$$

and the forecast radiation is recovered as

$$
\operatorname { R a d } _ { f } ( \tau ) = \kappa _ { f } ( \tau ) \operatorname { R a d } _ { \csc } ( \tau ) .\tag{24}
$$

As with temperature, the shift $\mu _ { \kappa } ( \tau )$ is calibrated for this study. It lowers the clear-sky index on forecast wet days and raises it on forecast dry days, preserving the physical link between cloud cover, precipitation, and incoming radiation.

## 3.7.2 Analog Tail Selection

The weather beyond the 7-day forecast is filled in with observed weather from a historical year that resembles the current season. The analog is selected using only information available up to the decision date. For a target season in year $y ,$ the candidate pool is $\mathcal { H } _ { y } = \{ 1 9 8 0 , \dots , y - 1 \}$ : the target year and all later years are excluded, so neither the realized weather of the season nor any subsequent record can enter the analog tail. For example, an advisory for the 2015 season compares the weather observed up to its decision date with the same calendar window in each year from 1980 to 2014.

At each cycle t, we describe the current season with a climate descriptor

$$
X _ { t } = [ \mathrm { O N I } , \mathrm { P } , \Delta T _ { \mathrm { m a x } } ] ,\tag{25}
$$

which captures the large-scale climate state and the seasonal buildup before sowing. Here ONI is the Oceanic Niño Index, the running three-month mean of sea-surface-temperature anomalies. P and $\Delta T _ { \mathrm { m a x } }$ are the cumulative precipitation and the maximum-temperature anomaly over the previous weeks.

Because these features have different units and scales, we standardize them. Let H be the historical feature matrix, with row i a year in $\mathcal { H } _ { y }$ and column j a feature, and let $\mu _ { j }$ and $\sigma _ { j }$ be the mean and standard deviation of feature j over $\mathcal { H } _ { y }$ Then

$$
Z _ { i , j } = \frac { H _ { i , j } - \mu _ { j } } { \sigma _ { j } } , \qquad z _ { t , j } = \frac { X _ { t , j } - \mu _ { j } } { \sigma _ { j } } .\tag{26}
$$

Each feature carries a fixed agronomic importance weight, with $\omega = [ \omega _ { 1 } , \ldots , \omega _ { k } ]$ and $\textstyle \sum _ { j = 1 } ^ { k } \omega _ { j } = 1$ . The dissimilarity between the current season and historical year i is the weighted Euclidean distance

$$
d _ { i } = \sqrt { \sum _ { j = 1 } ^ { k } \omega _ { j } \left( Z _ { i , j } - z _ { t , j } \right) ^ { 2 } } .\tag{27}
$$

We select the year $i ^ { * } = \arg \operatorname* { m i n } _ { i \in \mathcal { H } _ { y } } d _ { i }$

## 4 Experimental Results

This section begins with the experimental setup (Section 4.1). It then compares Agri-SCHOLAR against the regional package of practices on yield (Section 4.2), profit (Section 4.3), and sustainability (Section 4.4). Next, it examines a single season in detail as a case study (Section 4.5) and closes with an ablation study of the Advisory Bounding Agent (Section 4.6).

## 4.1 Experimental Setup

We evaluate Agri-SCHOLAR for maize in two regions of South Asia that differ in their water regimes: one irrigated and one rainfed. For each region, we simulate 11 consecutive kharif seasons (2015-2025). Although we study maize, the framework is crop-agnostic: it can be applied to any crop the simulator supports, and another process-based simulator can replace APSIM. All language agents use google/gemma-4-E4B-it.

## 4.1.1 Data

Daily weather (rainfall, minimum and maximum temperature, solar radiation) is taken from ERA5 for 1980-2025 at the coordinates of each study site and serves as the “observed” weather. The full record is used only to evaluate each season; advisories use only data available up to the decision date (Section 3.7). Local agricultural experts provided soil profiles for both regions. The knowledge base used by the CRA consists of agronomic literature, institutional reports, and the regional package of practices for maize. The expert-curated costs for grain, fertilizer, water, and labor, along with sustainability shadow costs and references, are detailed in Appendix 1.

## 4.1.2 Baseline

Our baseline is the regional package of practices (PoP), the main channel through which agronomic guidance reaches farmers. The PoP is a fixed calendar of fertilizer and irrigation operations, and we simulate it in APSIM under the actual weather of each season, representing a farmer who follows it diligently. Both methods sow on the same day, as given by the rainfall-onset rule of the PSA (Section 3.2): from June 1 to July 15, we slide a three-day window and take the last day of the first window with at least 20 mm of cumulative rainfall. Differences in outcome, therefore, come from in-season management rather than from the sowing date.

## 4.1.3 Evaluation Protocol and Metrics

In each season, Agri-SCHOLAR issues one advisory per week, from sowing (week 0) until the crop reaches physiological maturity. Each advisory is chosen using only the constructed weather $W _ { t }$ of that week (Eq. 17), never the future observed weather. The advisories issued over the season form the executed management plan, which is fixed once issued. The full season is then replayed in APSIM under the true weather to obtain the realized outcome. The PoP is evaluated on the same true weather, so both methods are scored on identical seasons, and Agri-SCHOLAR’s results reflect decisions made without knowledge of the weather beyond the seven-day forecast.

We report grain yield, net profit R (Eq. 3), and its components: revenue, production cost, and sustainability penalty. Monetary values are converted from local currency to US dollars at purchasing power parity (Appendix 1). Means are taken over the 11 seasons of each region, and improvements are stated relative to the PoP.

## 4.2 Yield

![](images/1a1ba4f03983fef1be5263985392ac2a73ea5bd00a37292462372fba8d5b2838.jpg)  
(a) Irrigated region

![](images/fed7b27e25dbdbb2cb4784e3a9f9ae6d06e720a2fe55fe73a219b3948465a731.jpg)  
(b) Rainfed region  
Figure 2: Grain yield per season under PoP baseline and Agri-SCHOLAR.

Figure 2 shows grain yield per season in the irrigated and rainfed regions. Agri-SCHOLAR outperforms the PoP in every season of the study in both regions. In the irrigated region, it raises mean yield by 15.18%, with per-season gains ranging from 6.57% to 29.45%. In the rainfed region, the mean improvement is 22.43%, with per-season gains ranging from 15.14% to 27.72%.

The size of the gain depends on how the season’s rainfall is distributed. In the irrigated region, the gain is largest in 2023 $( + 2 , 5 0 9 \mathrm { k g } \mathrm { h a } ^ { - 1 } , + 2 9 . { \overset { . } { 4 } } 5 \% )$ , a season with a long mid-season dry spell. The PoP’s fixed calendar cannot respond to such a spell, whereas Agri-SCHOLAR detects it from the observed weather and the short-range forecast and irrigates when soil moisture falls (Section 4.5). The gain is smallest in 2018 (+428 kg ha<sup>−1</sup>, +6.57%), a season with well-distributed rainfall, in which the PoP’s calendar happens to suit the weather and leaves little room for improvement.

![](images/c1e6e208148b69339b90db8ea350508ee20ee6c7806837f9d584375c0e63ad3c.jpg)  
Figure 3: Net profit per season under PoP baseline and Agri-SCHOLAR (irrigated region).

## 4.3 Profit and Resource Use

Figure 3 reports net profit R per season in the irrigated region, that is, grain revenue minus production cost and sustainability penalty (Eq. 3). Agri-SCHOLAR earns a higher net profit than the PoP in every season in both regions, and it increases mean net profit by 17.2% in the irrigated region and by 22.8% in the rainfed region. Table 2 summarizes mean yield, profit, production cost, and sustainability penalty.

Table 2 also shows where the profit gain comes from. Agri-SCHOLAR spends more on inputs than the PoP: its mean production cost is $\$ 185.66\mathrm { { h } \bar { a } ^ { - 1 } }$ higher in the irrigated region and $\$ 64.46\mathrm { { h a } ^ { - 1 } }$ higher in the rainfed region. This extra spending is recovered several times over by the additional grain. At the grain price of $\$ 1.04\mathrm { k g } ^ { - 1 }$ (Table 4), the mean yield gain of $9 8 3 \mathrm { ~ k g } \mathrm { h a } ^ { - 1 }$ in the irrigated region is worth about $\$ 1,023,456 a$ , and the sustainability penalty is $\$ 228\thinspace\dot { \mathrm { h a } } ^ { - 1 }$ lower. Together these account for the mean profit gain of $\$ 1,066\mathrm { { h a } ^ { - 1 } }$ . In the rainfed region, the yield gain of $1 , 3 2 5 \mathrm { { k g } h a ^ { - 1 } }$ is worth about $\$ 1,378\mathrm{ h a } ^ { - 1 }$ , while the penalty is only $\$ 8\ : \mathrm { h a } ^ { - 1 }$ lower, so almost the entire profit gain of $\$ 1,322\hslash \mathrm { a } ^ { - 1 }$ comes from higher yield. Agri-SCHOLAR therefore does not raise profit by cutting inputs. It spends more on inputs when the resulting yield gain outweighs their cost, and in the irrigated region it also reduces environmental losses, as discussed next.

Table 2: Mean yield, net profit, production cost, and sustainability penalty over 11 seasons. Monetary values are in PPP-adjusted \$/ha. Best value per column and region in bold (highest for yield and profit, lowest for cost and penalty).
<table><tr><td>Region</td><td>Method</td><td>Yield(kg/ha)</td><td>Profit($/ha)</td><td>Prod. cost</td><td>Sust. penalty</td></tr><tr><td rowspan="2">Irrigated</td><td>PoP</td><td>6,477.55</td><td>5,959.33</td><td>284.02</td><td>495.33</td></tr><tr><td>Agri-SCHOLAR</td><td>7,461.00</td><td>7,025.09</td><td>469.68</td><td>267.05</td></tr><tr><td rowspan="2">Rainfed</td><td>PoP</td><td>5,909.36</td><td>5715.15</td><td>149.63</td><td>282.81</td></tr><tr><td>Agri-SCHOLAR</td><td>7,234.55</td><td>7,037.19</td><td>214.09</td><td>274.95</td></tr></table>

![](images/e181890b0f49a801ea5cf1d3609c0763230c3fd540e8dbccc00d256b8125e50a.jpg)  
(a) Cumulative Deep Drainage

![](images/432a6ba00675bf383c1e64162ea9351bbfdd78281438559d1d338a642817f7bb.jpg)  
(b) Cumulative Nitrate Leaching  
Figure 4: Drainage and Leaching per season (irrigated region).

## 4.4 Sustainability

Figure 4 compares cumulative deep drainage and nitrate leaching per season in the irrigated region. Deep drainage is the water that moves below the root zone and is lost to the crop, and nitrate leaching is the nitrogen carried below the root zone with it. Agri-SCHOLAR reduces mean drainage by 52.8% and mean $\mathrm { N O _ { 3 } }$ leaching by 52.7%. The reduction holds in every season, ranging from 40.73% to 82.49% for drainage and from 33.64% to 84.13% for leaching, and it lowers the mean sustainability penalty by 46.1% (Table 2).

These reductions follow from the structure of the reward. Drainage and leaching carry shadow costs (Eq. 11), so irrigating soil that is already wet is penalized twice, through the cost of the water and through the drainage it causes. The optimizer therefore irrigates only when soil moisture is low. Similarly, nitrogen applied before the crop can take it up remains in the soil and is exposed to leaching, so the optimizer times UreaN doses to crop demand. The PoP irrigates and fertilizes on a fixed calendar and has no such feedback. The case study in Section 4.5 shows both effects.

In the rainfed region, the mean penalty falls by only 2.8% (Table 2). A likely reason is that, without irrigation, drainage in this region is driven mainly by rainfall, which management cannot change, so there is less room for improvement.

## 4.5 Case Study

To show how the rolling cycle behaves, we follow the 2023 kharif season in the irrigated region, chosen because it showed the largest gain over the PoP (+2,509 kg ha<sup>−1</sup>, +29.45%). Figure 6 provides week-by-week information on Fertiliser (UreaN, DAP) and irrigation applied per week. It also shows how the projected yield changes over the season, and Listing 5 presents selected natural-language advisories issued by the pipeline. The complete week-by-week advisories are provided in Appendix 2.

The pipeline recommended sowing on 4 July, once the monsoon had set in, with basal DAP, residue mulching, and reduced tillage. In week 0, it added a starter UreaN dose, and the crop-model check predicted no water or nitrogen stress. The early season was wet (332 mm by week 4, including 101 mm in week 3), so the advisories withheld irrigation and extra UreaN, since the soil was moist and added nitrogen would have leached. In week 4, as the crop entered rapid growth, it prescribed the main top-dress of 97 kg $\mathrm { { h a ^ { - 1 } } }$ UreaN. The most consequential decision came in week 6, when only 7 mm of rain had fallen in the past week and the dry spell had reached 11 days: a single 63 mm irrigation halved the projected water stress. By week 8 (tasselling and silking) the dry spell had grown to 23 days, yet no further irrigation was issued because the soil still held enough moisture. Over the season, the pipeline applied $1 8 0 \mathrm { k g } \mathrm { h a } ^ { - 1 }$ of UreaN in three splits and one irrigation, yielding $1 1 , \mathsf { 0 2 8 \ k g h a ^ { - 1 } }$ with 12.9 kg N ha<sup>−1</sup> leached.

# PRE-SOWING (land preparation) # WEEK 6 | 15 Aug | 42 DAS   
sow: 04 Jul 2023 (monsoon set in; soil moist enough stage: knee-high, rapid vegetative growth   
for germination) weather: rain since 1 Jun = 351 mm (7 mm past week);   
basal: 200 kg/ha DAP (4 bags of 50 kg) in the longest dry spell = 11 d   
seed furrow at sowing action: one irrigation of 63 mm (631,000 L/ha;   
mulch: 26 kg/ha residue/straw between rows to roughly one furrow irrigation)   
save soil moisture why: soil moisture is running low for the current   
tillage: light/reduced tillage (20% surface disturbed) crop stage (dry spells of up to 11 d this season);   
model check: minor stress remains (water 5.4 d,   
# WEEK 0 | 04 Jul | 0 Days After Sowing (DAS) N 0.0 d)   
stage: sowing / germination   
weather: rain since 1 Jun = 88 mm; longest dry spell = 10 d # WEEK 8 | 29 Aug | 56 DAS   
action: apply 128.3 kg/ha of urea (2.85 bags of 45 kg) stage: tasselling and silking (most water-sensitive)   
as side-dressing near plant rows on moist weather: rain since 1 Jun = 362 mm (5 mm past week);   
soil; cover with soil longest dry spell = 23 d   
why: starter urea with basal DAP supports early root action: no fertiliser or irrigation this week; continue   
and leaf growth; model check: no water or N routine scouting (weeds, fall armyworm, leaf blight)   
stress expected under this plan   
why: only 5 mm rain last week, but soil still holds   
# WEEK 4 | 01 Aug | 28 DAS enough moisture; week-7 urea still available,   
stage: seedling to early vegetative extra urea now would mostly leach; model check:   
weather: rain since 1 Jun = 332 mm (42 mm past week); minor stress remains (water 5.6 d, N 0.0 d)   
longest dry spell = 10 d   
action: apply 210.9 kg/ha of urea (4.69 bags of 45 kg) # WEEK 16 | 24 Oct | 112 DAS   
as side-dressing near plant rows on moist stage: physiological maturity / harvest preparation   
soil; cover with soil weather: rain since 1 Jun = 451 mm (0 mm past week);   
why: crop is entering its fastest growth phase and longest dry spell = 23 d   
takes up most of its N now; top-dressing avoids action: no fertiliser or irrigation this week   
leaf yellowing; model check: no water or N why: crop is maturing, so fertiliser or water will no   
stress expected under this plan longer raise yield; model check: minor stress   
remains (water 2.9 d, N 0.0 d)  
Figure 5: Selected weekly advisories issued by Agri-SCHOLAR in the 2023 irrigated-region case study.

![](images/e528eba47854abcc339cdf2f35cc8ea0b27e6918ee13c37599da013235908f86.jpg)

![](images/8c7f4e3bea41cb2ae3f5776739e10e6af8a74ba8bdbf6e14f955c2952220c81d.jpg)

![](images/0c945542d186b25c4b280c662cea0b7911c642367d18042916c6d8aad03b6daf.jpg)  
Figure 6: Case study for irrigated region, 2023 season. (a) UreaN applied per week (b) Irrigation per week (c) Projected Yield

In comparison, the Package of Practices (PoP) follows a fixed schedule: a basal dose of 75 kg $\mathrm { P _ { 2 } O _ { 5 } }$ , 40 kg $\mathrm { K _ { 2 } O _ { 3 } }$ and $5 0 \mathrm { { k g } \tilde { N } \ h a ^ { - 1 } }$ , then $5 0 \mathrm { { k g } N h a ^ { - 1 } }$ at weeks 4 and $8 ( 1 5 0 \mathrm { { k g } N h a ^ { - 1 } }$ in total), plus six irrigations at weeks 0, 2, 4, 6, 8, and 10. Being tied to the calendar, it cannot respond to the season as it unfolds. It ignores rainfall (332 mm had fallen by week 4, yet it still irrigates at weeks 0, 2, and 4), fixes total nitrogen regardless of crop need, which raises cost and leaching in wet seasons, and spreads irrigation evenly, whereas Agri-SCHOLAR concentrated its single irrigation at week 6, when soil moisture was actually low. As a result, the PoP uses five more irrigation events and about 77% more nitrogen, yet yields $8 , 5 1 9 \mathrm { { k g } h a ^ { - 1 } }$ , which is $2 { , } 5 0 9 \mathrm { { k g } h a ^ { - 1 } }$ less than Agri-SCHOLAR. A farmer may adjust the PoP from experience, but it remains a single predefined plan, whereas Agri-SCHOLAR adapts to the evolving crop, soil moisture, and weather on each farm and season.

## 4.6 Ablation: Advisory Bounding Agent

The central design choice of Agri-SCHOLAR is that the optimizer does not search the action space on its own: the ABA first uses the retrieved agronomic context to decide which actions are admissible at the current crop stage, and the

OA searches only within those LLM-generated bounds. To test this choice, we keep the simulator, reward, optimizer, weather information, and season structure fixed, and replace the ABA’s bounds in two ways.

• Loose bounds. The OA searches a single fixed range for each action, identical in every week and every season: urea-N in $[ 0 , \overline { { N } } ]$ kg $\mathrm { { N h a ^ { - 1 } } }$ per application and irrigation in [0, I] mm per week, where N and I are upper limits. These limits are hardcoded using the regional package of practices, so they rely on expert-curated, region-specific knowledge and naturally do not adapt to crop stage, soil conditions, or weather. This makes the pipeline hard to scale, since each new region or crop would require its own manually specified limits, much like the expert-guided advisories.

• No bounds. The upper limits are removed, so the OA may propose any non-negative amount of UreaN and irrigation in any week.

Table 3 reports the mean of each metric over the 11 seasons. The number below each mean, after ±, is the standard deviation of that metric across the 11 seasons. It measures how much the outcome varies from one season to the next, it is not an uncertainty over repeated runs.

With loose bounds, mean yield rises by 2.1%, profit by 2.5%, and production cost falls by 14.6%, but the sustainability penalty rises by 20.1%. These gains come from actions that a farmer could not or should not carry out, such as doses too small to apply (2.8 kg $\mathrm { { h a ^ { - 1 } } }$ of UreaN at week 8 of 2024) or a single large dose at planting $( 1 9 0 \mathrm { { k g } h a ^ { - 1 } }$ of UreaN in 2019) that the young crop cannot yet take up and that is therefore exposed to leaching. The optimizer produces these schedules because neither APSIM nor the reward encodes whether a dose suits the crop’s growth stage or can be applied with standard equipment (Appendix 3.2).

With no bounds, performance collapses. Mean yield falls by 5.1%, production cost rises about 9.5-fold and the sustainability penalty about 4.6-fold, and mean profit drops by 76%, with negative profit in four of the 11 seasons. In 2019, the OA applied 6,062 kg ha<sup>−1</sup> of UreaN and about 6,000 mm of irrigation in week 0, for a projected reward of about $\mathbb { S } 6 , 4 2 1 . 1 \bar { 0 } \mathrm { { h a } ^ { - 1 } }$ on the constructed weather. Under the realized weather, 3,271 kg ha<sup>−1</sup> of nitrate leached, the sustainability penalty reached about $\$ 4,778.5\mathrm { h a } ^ { - 1 }$ , and the season ended with a net loss of \$2,887 ha<sup>−1</sup> (Table 6).

These results show that the bounds guard against simulator exploitation and keep recommendations agronomically practical. The ABA also addresses the scaling problem: at finer spatial and temporal resolution, hand-specifying bounds for every crop stage, farm, and season becomes infeasible even for experts, whereas the ABA generates them automatically from the retrieved context for each new setting.

Table 3: ABA ablation: irrigated region, mean over 11 seasons.
<table><tr><td>Variant</td><td>Yield (kg/ha)</td><td>Profit ($/ha)</td><td>Prod. cost</td><td>Sust. penalty</td></tr><tr><td>Agri-SCHOLAR</td><td>7,461.00</td><td>7,025.09</td><td>469.68</td><td>267.05</td></tr><tr><td>Loose bounds</td><td>7,614.82</td><td>7,199.86</td><td>401.24</td><td>320.73</td></tr><tr><td></td><td>±1,785.82</td><td>±1,823.07</td><td>±61.40</td><td>±107.33</td></tr><tr><td>No bounds</td><td>7,082.64</td><td>1,682.69</td><td>4,451.40</td><td>1,234.11</td></tr><tr><td></td><td>±2,153.72</td><td>±3,117.25</td><td>±2,521.46</td><td>±1,503.77</td></tr></table>

## 5 Conclusion and Future Work

We introduced Agri-SCHOLAR, a rolling, in-season framework for farm-specific crop advisories. Prior simulationgrounded advisory systems need the whole season’s weather in advance, which no forecasting system can provide. Agri-SCHOLAR instead rebuilds its weather trajectory every week from observations to date, a short-range forecast, and an analog-year tail. Each advisory comes from a closed loop in which LLM agents propose agronomically feasible bounds on irrigation and fertilizer actions. An optimizer then searches within those bounds, with APSIM as the evaluation oracle, to maximize a profit-based reward that includes sustainability penalties.

Over 11 maize seasons (2015-2025) in one irrigated and one rainfed region of South Asia, Agri-SCHOLAR beat the regional package of practices (PoP) in every season, even though the PoP was simulated with perfect knowledge of the true weather. Mean yield rose by 15.2% in the irrigated region and 22.4% in the rainfed region, and mean net profit rose by 17.2% and 22.8%. In the irrigated region, deep drainage and nitrate leaching each fell by about 50%. The 2023 case study shows where the gains come from. The pipeline withheld irrigation and extra nitrogen during a wet early season and concentrated its single irrigation at the week when soil moisture was low. Finally, we showed that the bounds from the Advisory Bounding Agent keep the optimizer from exploiting the simulator with infeasible actions. Taken together, these results suggest that rolling, simulator-validated advisories can overcome the main limitations of expert-driven, region-level guidance without an accurate season-long forecast.

Future work. We see several directions for extending this work. First, we will explore reinforcement learning as an alternative or complement to the bounded optimizer, while keeping the resulting advisories interpretable. Second, we will study sowing-date decisions, which our experiments suggest matter considerably but which require much better long-range weather prediction than is available today. Third, we will replace the emulated forecasts with operational numerical weather prediction products and test how sensitive the advisories are to forecast error. Fourth, we will calibrate and validate the simulator against field trials, and learn the shadow costs and reward weights. Finally, we plan prospective pilots with farmers, delivering advisories through channels they already use, such as SMS or voice in local languages, to measure adoption and real-world outcomes and to feed their feedback back into the system.

## Acknowledgments

The authors thank Prof. Soma Dhawala for his valuable suggestions and agronomic insights. The authors also thank Dr.   
Richard Magala for his helpful discussion regarding the apsimNGpy framework.

The authors acknowledge the use of various AI tools and assistants, such as GPT and Claude, to enhance the writing quality and provide summaries of source materials. However, the authors diligently verified all information provided by these AI tools to ensure accuracy and factual integrity.

## References

[1] Vedant Balasubramaniam, Geetha Charan, Manojkumar Patil, Rohit P Suresh, V Priyanka, Kodur Sai Vinay Sathvik, and Y Narahari. Agri-sage: Simulation-grounded multi-agent llm for context-aware agricultural advisory generation. arXiv preprint arXiv:2607.00454, 2026.

[2] Shuting Yang, Zehui Liu, Wolfgang Mayer, Ningpei Ding, Ying Wang, Yu Huang, Pengfei Wu, Wanli Li, Lin Li, Hong-Yu Zhang, et al. Shizishangpt: An agricultural large language model integrating tools and resources. In International Conference on Web Information Systems Engineering, pages 284–298. Springer, 2024.

[3] Bo Yang, Yu Zhang, Lanfei Feng, Yunkui Chen, Jianyu Zhang, Xiao Xu, Nueraili Aierken, Yurui Li, Yuxuan Chen, Guijun Yang, et al. Agrigpt: A large language model ecosystem for agriculture. arXiv preprint arXiv:2508.08632, 2025.

[4] Mesafint Fanuel, Mahmoud Nabil Mahmoud, Crystal Cook Marshal, Vishal Lakhotia, Biswanath Dari, Kaushik Roy, and Shaohu Zhang. Agriregion: Region-aware retrieval for high-fidelity agricultural advice. arXiv preprint arXiv:2512.10114, 2025.

[5] Nadine Angela Palanca Cantonjos and Arpita Biswas. Agroaskai: a multi-agentic ai framework for supporting smallholder farmers’ enquiries globally. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 40217–40223, 2026.

[6] Josué Kpodo, Parisa Kordjamshidi, and A Pouyan Nejadhashemi. Agxqa: A benchmark for advanced agricultural extension question answering. Computers and Electronics in Agriculture, 225:109349, 2024.

[7] Yutong Zhou and Masahiro Ryo. Agribench: A hierarchical agriculture benchmark for multimodal large language models. In European Conference on Computer Vision, pages 207–223. Springer, 2024.

[8] Aruna Gauba, Irene Pi, Yunze Man, Ziqi Pang, Vikram S Adve, and Yu-Xiong Wang. Agmmu: A comprehensive agricultural multimodal understanding and reasoning benchmark. arXiv e-prints, pages arXiv–2504, 2025.

[9] Muhammad Arbab Arshad, Talukder Zaki Jubery, Tirtho Roy, Rim Nassiri, Asheesh Singh, Arti Singh, Chinmay Hegde, Baskar Ganapathysubramanian, Aditya Balu, Adarsh Krishnamurthy, et al. Ageval: A benchmark for zero-shot and few-shot plant stress phenotyping with multimodal llms. arXiv, 2024.

[10] Josué Kpodo, A Pouyan Nejadhashemi, Parisa Kordjamshidi, Rasu Eeswaran, and Younsuk Dong. Closing the gap between crop simulation and practical decision-making. Computers and Electronics in Agriculture, 247: 111655, 2026.

[11] Robert L McCown, Graeme L Hammer, John Norman Gresham Hargreaves, Dean P Holzworth, and David M Freebairn. Apsim: a novel software system for model development, model testing and simulation in agricultural systems research. Agricultural systems, 50(3):255–271, 1996.

[12] Romain Gautron, Emilio J Padrón, Philippe Preux, Julien Bigot, Odalric-Ambrym Maillard, and David Emukpere. gym-dssat: a crop model turned into a reinforcement learning environment. arXiv preprint arXiv:2207.03270, 2022.

[13] Jing Wu, Ran Tao, Pan Zhao, Nicolas F Martin, and Naira Hovakimyan. Optimizing nitrogen management with deep reinforcement learning and crop simulations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1712–1720, 2022.

[14] Ran Tao, Pan Zhao, Jing Wu, Nicolas F Martin, Matthew T Harrison, Carla Ferreira, Zahra Kalantari, and Naira Hovakimyan. Optimizing crop management with reinforcement learning and imitation learning. arXiv preprint arXiv:2209.09991, 2022.

[15] Jing Wu, Zhixin Lai, Suiyao Chen, Ran Tao, Pan Zhao, and Naira Hovakimyan. The new agronomists: Language models are experts in crop management. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5346–5356, 2024.

[16] Zhaoan Wang, Shaoping Xiao, Junchao Li, and Jun Wang. Learning-based agricultural management in partially observable environments subject to climate variability. Scientific reports, 2026.

[17] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Graphcast: Learning skillful medium-range global weather forecasting. arXiv preprint arXiv:2212.12794, 2022.

[18] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate medium-range global weather forecasting with 3d neural networks. Nature, 619(7970):533–538, 2023.

[19] Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, et al. Probabilistic weather forecasting with machine learning. Nature, 637(8044):84–90, 2025.

[20] Cristian Bodnar, Wessel P Bruinsma, Ana Lucic, Megan Stanley, Anna Allen, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan A Weyn, Haiyu Dong, et al. A foundation model for the earth system. Nature, 641(8065):1180–1187, 2025.

[21] KK Singh, Kripan Ghosh, SC Bhan, Priyanka Singh, Lata Vishnoi, R Balasubramanian, SD Attri, Sheshakumar Goroshi, and R Singh. Decision support system for digitally climate informed services to farmers in india. Journal ofAgrometeorology, 25(2):205–214, 2023.

[22] Tam Thi Le and Cornelis Swaans. Vietnamese government scales agroclimatic bulletins across 13 provinces in the Mekong Delta, reaching more than 290,000 farmers. Study ASI-2405, Bioversity International and CIAT, Rome, Italy and Cali, Colombia, 2025. 4 p.

[23] Clarence W Richardson. Stochastic simulation of daily precipitation, temperature, and solar radiation. Water resources research, 17(1):182–190, 1981.

[24] Clarence W Richardson and David A Wright. Wgen: A model for generating daily weather variables. Report ARS-8 August 1984. 83 p, 3 Fig, 12 Tab, 13 Ref, 4 App., 1984.

[25] Salva Rühling Cachay, Duncan Watson-Parris, and Rose Yu. U-cast: A surprisingly simple and efficient frontier probabilistic ai weather forecaster. In ICML 2026 AIfor Science Workshop, 2026.

[26] Pierre Ineichen and Richard Perez. A new airmass independent formulation for the linke turbidity coefficient. Solar Energy, 73(3):151–157, 2002.

[27] International Comparison Program, World Bank. PPP conversion factor, GDP (local currency unit per international \$), indicator PA.NUS.PPP. World Development Indicators. Sources: International Comparison Program (World Bank), Eurostat PPP Programme, OECD PPP Programme. Licence CC BY-4.0, 2024. Published 30 May 2024. https://data.worldbank.org/indicator/PA.NUS.PPP. Accessed 2026.

[28] Commission for Agricultural Costs and Prices. Price policy for kharif crops: Marketing season 2023–24. Technical report, Government of India, 2023. Maize: support price 2090 per quintal; projected yield 36.96 q/ha; cost of production A2 1086, A2+FL 1394, C2 1797 per quintal.

[29] Press Information Bureau. 12 years of modi governance: Transforming india into an atmanirbhar fertilizer hub amid global uncertanities, June 2026. URL https://www.pib.gov.in/PressReleasePage.aspx?PRID= 2272694&reg=3&lang=1. Press release, Release ID 2272694.

[30] CIMMYT. Maize stover: An underutilized resource for rainfed india. https://www.cimmyt.org/news/ maize-stover-an-underutilized-resource-for-rainfed-india/, September 2013. CIMMYT and ILRI.

[31] MG Chandrakanth. Water resource economics: towards a sustainable use of water for irrigation in India. Springer, 2015.

[32] Arshdeep Singh and Poonam Kataria. Cost competitiveness of maize cultivation in india: A state-wise analysis. Indian Journal of Economics and Development, 21:477–487, 2025.

[33] Press Information Bureau. Increase in minimum wages, March 2017. URL https://www.pib.gov.in PressReleasePage.aspx?PRID=1485016&reg=48&lang=2. Press release, Release ID 1485016.

[34] Manjunath Mukkannawar. Economic analysis of surface and subsurface drainage system in canal commands of north karnataka. Master’s thesis, University of Agricultural Sciences, Dharwad, 2011. URL https:// krishikosh.egranth.ac.in/items/891c3b32-43e4-4d28-8586-213a3aecfd16.

[35] J. Boonstra, H. P. Ritzema, W. Wolters, R. J. Oosterbaan, L. Res, and A. M. van Lieshout. Research on the Control ofWaterlogging and Salinization in Irrigated Agricultural Lands: Recommendations on Waterlogging and Salinity Control Based on Pilot Area Drainage Research. Central Soil Salinity Research Institute (CSSRI), Karnal, India, 2002.

[36] V. Singh and A. Srivastava. Performance of different maturity groups of maize hybrids under temporary waterlog ging condition in a mollisol. Indian Journal of Agricultural Sciences, 2013.

[37] Y. P. Abrol, T. K. Adhya, V. P. Aneja, N. Raghuram, H. Pathak, U. Kulshrestha, C. Sharma, and B. Singh, editors. The Indian Nitrogen Assessment: Sources ofReactive Nitrogen, Environmental and Climate Effects, Management Options, and Policies. Elsevier, 2017. Editor list to be verified.

[38] Saghar K. Motarjemi, Merete E. Styczen, Rasmus J. Petersen, Kasper S. J. Jensen, and Finn Plauborg. Effects of different drainage conditions on nitrogen losses of an agricultural sandy loam soil. Journal ofEnvironmental Management, 325:116267, 2023. doi: 10.1016/j.jenvman.2022.116267.

[39] et al. Zhang. Drip fertigation sustains crop productivity while mitigating reactive nitrogen losses in chinese agricultural systems: Evidence from a meta-analysis. Science ofthe Total Environment, 886:163804, 2023. doi: 10.1016/j.scitotenv.2023.163804.

[40] Anning Zhu, Jiabao Zhang, Bingzi Zhao, Zhuhua Cheng, and Liping Li. Water balance and nitrate leaching losses under intensive crop production with ochric aquic cambosols in north china plain. Environment International, 31 (6):904–912, 2005. doi: 10.1016/j.envint.2005.05.038.

[41] Sung-Tae Lee and Sun-Moon Jung. Monetizing environmental impacts of nuclear power: Cost-benefit analysis using social cost of carbon. Sage Open, 14(4):21582440241301468, 2024.

[42] Katharine Ricke, Laurent Drouet, Ken Caldeira, and Massimo Tavoni. Country-level social cost of carbon. Nature Climate Change, 8:895–900, 2018. doi: 10.1038/s41558-018-0282-y.

## Appendix

## 1 Cost and Penalty Parameters

The simulator uses the following fixed cost and value parameters. All values are expressed in local currency units (LCU). The corresponding dollar values are also reported using Purchasing Power Parity (PPP) [27]. The simulator itself uses the LCU values during the experimentation, we convert the values post-experimentation into dollars.

Table 4: Cost and penalty parameters used in Optimization Agent (OA).
<table><tr><td>Parameter</td><td>Notation</td><td>LCU</td><td>$</td><td>Source/Basis</td></tr><tr><td>Grain price (per kg)</td><td> $p _ { \underline { { { g } } } }$ </td><td>20.90</td><td>1.04</td><td>Support price [28]</td></tr><tr><td>UreaN (per kg N)</td><td> $p _ { 1 } ^ { r }$ </td><td>13.04</td><td>0.65</td><td>Retail price [29]</td></tr><tr><td>DAP (per kg)</td><td> $p _ { 2 } ^ { F }$ </td><td>27.00</td><td>1.34</td><td>Retail price [29]</td></tr><tr><td>Mulch (per kg)</td><td>pM</td><td>2.00</td><td>0.10</td><td>Residue and fodder prices [30]</td></tr><tr><td>Water (per mm)</td><td> $p _ { w }$ </td><td>2.50</td><td>0.12</td><td>Irrigation charges [31, 32]</td></tr><tr><td>Labour (per operation)</td><td> $p _ { L }$ </td><td>350.00</td><td>17.42</td><td>Rural wage [33]</td></tr><tr><td>Waterlogging (per ha-season)</td><td> $S _ { \mathrm { w l } }$ </td><td>5,000</td><td>248.88</td><td>Drainage cost [34–36]</td></tr><tr><td>N oversupply (per ha)</td><td> $S _ { \mathrm { n o } }$ </td><td>2,000</td><td>99.55</td><td>Excess-N penalty [37]</td></tr><tr><td>Deep drainage (per mm)</td><td> $S _ { \mathrm { d d } }$ </td><td>15.00</td><td>0.75</td><td>Water-loss and drainage impacts [34, 38]</td></tr><tr><td>N leaching (per kg N)</td><td> $S _ { \mathrm { n l } }$ </td><td>25.00</td><td>1.24</td><td>Nutrient-loss and environmental impacts [37- 40]</td></tr><tr><td>SOC loss (per kg C)</td><td> $S _ { \mathrm { c l } }$ </td><td>150.00</td><td>7.47</td><td>Social cost of carbon [41, 42]</td></tr><tr><td>Crop failure (per ha)</td><td> $S _ { \mathrm { c f } }$ </td><td>50,000</td><td>2,488.80</td><td>Cost of production [28]</td></tr></table>

## 1.1 Price Parameters

The price parameters are based on prevailing market, administered, or farmer-paid prices reported in the literature and official sources. Where applicable, these prices reflect the prevailing subsidy and pricing policies in the country of study during the corresponding period. Thus, the parameters represent the effective prices relevant to the production decision.

Grain $( p _ { g } ) .$ The maize support price for 2023-24 was 2,090 LCU per quintal (100 kg) [28], giving a value of 20.90 LCU per kg.

UreaN $( p _ { 1 } ^ { F } )$ . The reported farmer-paid price of a 45 kg Urea bag is 266.5 LCU and the bag contains 46% N [29]. We express the N fertilizer cost on an 100 % elemental-N basis and use 13.04 LCU per kg of N in the simulator, reflecting the effective fertilizer price under the prevailing pricing and subsidy system.

DAP $( p _ { 2 } ^ { F } )$ ). A 50 kg bag of DAP is reported at 1,350 LCU [29], giving an effective price of 27.00 LCU per kg. This reflects the prevailing farmer-paid price under the applicable fertilizer pricing system.

Labour $( p _ { L } ) .$ . Labour is charged as one person-day per operation. The reported rural wage was 316 LCU in 2023-24 and 349 LCU in 2024-25 [33]. We use 350 LCU per operation.

Water $( p _ { w } )$ . Reported irrigation charges for maize are approximately 254 LCU per ha [32]. Assuming about 100 mm of irrigation per season gives approximately 2.54 LCU per mm. We use 2.50 LCU per mm as the effective farmer-paid water cost.

Mulch (p<sub>M</sub>). Stover has been reported at approximately 3.8 LCU per kg [30]. We use 2.00 LCU per kg as the effective value of crop residue retained in the field.

## 1.2 Environmental Parameters

The environmental parameters convert simulated resource losses into monetary values.

Waterlogging management $( S _ { \mathrm { w l } } )$ . Waterlogging can reduce crop productivity and cause economic losses through yield losses and soil degradation. Reported estimates of waterlogging-related production losses range from 4,400 to 9,400 LCU per ha for individual crops, with a reported annual loss of 13,800 LCU per ha [34–36]. We use 5,000 LCU per ha-season as a conservative partial-cost parameter for waterlogging.

Deep drainage $( S _ { \mathrm { d d } } )$ . Deep drainage represents water that moves below the crop root zone and is therefore no longer available for crop uptake. Excessive deep drainage can also transport nutrients below the root zone, increasing nutrient losses and reducing nutrient-use efficiency [38, 40]. Such losses can reduce crop productivity and may require additional inputs to maintain crop growth. We use 15.00 LCU per mm of deep drainage as a model parameter representing the combined economic effect of excess water and associated nutrient losses.

N leaching $( S _ { \mathrm { n l } } ) .$ . Nitrogen loss through leaching reduces nitrogen-use efficiency and can affect crop productivity while contributing to environmental pollution [37–40]. We use 25.00 LCU per kg of leached N as a conservative model parameter representing the economic penalty associated with nitrogen loss.

N oversupply $( S _ { \mathrm { n o } } )$ . The N oversupply parameter assigns a penalty to nitrogen applied beyond the crop’s estimated nitrogen requirement. A value of 2,000 LCU is used as the penalty parameter, reflecting the economic and environmental costs associated with excess nitrogen application [37].

SOC loss $( S _ { \mathrm { c l } } )$ . SOC loss represents a reduction in carbon stored in the soil. Declining SOC can reduce soil fertility, water-holding capacity, and nutrient availability, thereby affecting crop productivity in subsequent seasons. It also represents a loss of carbon from the soil that may contribute to atmospheric $\mathrm { C O _ { 2 } }$ emissions. We therefore assign an economic value using the social cost of carbon. The global median social cost of carbon is estimated at US\$417 per tonne of $\mathrm { C O _ { 2 } }$ [41, 42]. Converting this value from $\mathrm { C \bar { O } _ { 2 } }$ to elemental carbon and into LCU gives approximately 146 LCU per kg of C. We use 150 LCU per kg of C as the SOC-loss parameter.

Crop failure. The projected cost of maize production in 2023–24 was 1,394 LCU per quintal, with a projected yield of 36.96 q/ha [28]. This gives approximately 51,522 LCU per ha. We use 50,000 LCU per ha as the crop-failure parameter.

## 2 Case Study: Full Weekly Advisories

This section provides the natural language advisories provided by Agri-SCHOLAR during the 2023 season in the irrigated region.

# PRE-SOWING (land preparation) # WEEK 2 | 18 Jul | 14 DAS   
sow: 04 Jul 2023 (monsoon set in; soil moist enough stage: seedling to early vegetative (V2-V6)   
for germination) weather: rain since 1 Jun = 189 mm (36 mm past week);   
basal: 200 kg/ha DAP (4 bags of 50 kg) in the   
longest dry spell = 10 d   
seed furrow at sowing action: no fertiliser or irrigation this week; continue   
mulch: 26 kg/ha residue/straw between rows to routine scouting (weeds, fall armyworm, leaf blight   
save soil moisture )   
tillage: light/reduced tillage (20% surface disturbed) why: 36 mm rain last week kept soil moist; week-0 urea   
still available, extra urea now would mostly   
# WEEK 0 | 04 Jul | 0 DAS leach; model check: no water or N stress   
stage: sowing / germination   
expected under this plan   
weather: rain since 1 Jun = 88 mm; longest dry spell = 10 d   
action: apply 128.3 kg/ha of urea (2.85 bags of 45 kg) # WEEK 3 | 25 Jul | 21 DAS   
as side-dressing near plant rows on moist stage: seedling to early vegetative (V2-V6)   
soil; cover with soil   
weather: rain since 1 Jun = 290 mm (101 mm past week);   
why: starter urea with basal DAP supports early root longest dry spell = 10 d   
and leaf growth; model check: no water or N action: no fertiliser or irrigation this week; continue   
stress expected under this plan routine scouting (weeds, fall armyworm, leaf blight   
)   
# WEEK 1 | 11 Jul | 7 DAS why: 101 mm rain last week kept soil moist; week-0 urea   
stage: emer ence / establishment   
still available, extra urea now would mostly   
weather: rain since 1 Jun = 153 mm (65 mm past week); leach; model check: no water or N stress   
longest dry spell = 10 d expected under this plan   
action: no fertiliser or irrigation this week; continue   
routine scouting (weeds, fall armyworm, leaf blight # WEEK 4 | 01 Aug | 28 DAS   
) stage: seedling to early vegetative (V2-V6)   
why: 65 mm rain last week kept soil moist; week-0 urea weather: rain since 1 Jun = 332 mm (42 mm past week);   
still available, extra urea now would mostly longest dry spell = 10 d   
leach; model check: no water or N stress action: apply 210.9 kg/ha of urea (4.69 bags of 45 kg)   
expected under this plan as side-dressing near plant rows on moist

<table><tr><td>why:</td><td>soil; cover with soil crop is entering its fastest growth phase and takes up most of its N now; top-dressing avoids</td><td>) why:</td><td>routine scouting (weeds, fall armyworm, leaf blight</td></tr><tr><td></td><td>leaf yellowing; model check: no water or N stress expected under this plan</td><td></td><td>only 9 mm rain last week, but soil still holds enough moisture; week-7 urea still available,</td></tr><tr><td></td><td></td><td></td><td>extra urea now would mostly leach; model check:</td></tr><tr><td># WEEK 5 | 08 Aug I 35 DAS</td><td></td><td></td><td>no water or N stress expected under this plan</td></tr><tr><td>stage:</td><td>knee-high, rapid vegetative growth (V8-V12)</td><td></td><td># WEEK 12 | 26 Sep I 84 DAS</td></tr><tr><td></td><td>weather: rain since 1 Jun = 344 mm (12 mm past week);</td><td>stage:</td><td>grain filling (milk to dough)</td></tr><tr><td></td><td>longest dry spell = 10 d</td><td></td><td>weather: rain since 1 Jun = 422 mm (19 mm past week);</td></tr><tr><td>action:</td><td>no fertiliser or irrigation this week; continue</td><td></td><td>longest dry spell = 23 d</td></tr><tr><td>)</td><td>routine scouting (weeds, fall armyworm, leaf blight</td><td>action:</td><td>no fertiliser or irrigation this week; continue</td></tr><tr><td>why:</td><td>only 12 mm rain last week, but soil still holds</td><td>)</td><td>routine scouting (weeds, fall armyworm, leaf blight</td></tr><tr><td></td><td>enough moisture; week-4 urea still available,</td><td>why:</td><td></td></tr><tr><td></td><td>extra urea now would mostly leach; model check:</td><td></td><td>19 mm rain last week, soil still holds enough moisture; week-7 urea still available, extra</td></tr><tr><td></td><td>no water or N stress expected under this plan</td><td></td><td>urea now would mostly leach; model check: no</td></tr><tr><td></td><td></td><td></td><td>water or N stress expected under this plan</td></tr><tr><td># WEEK 6 | 15 Aug | 42 DAS</td><td></td><td></td><td></td></tr><tr><td>stage:</td><td>knee-high, rapid vegetative growth (V8-V12)</td><td></td><td># WEEK 13 | 03 0ct | 91 DAS</td></tr><tr><td></td><td>weather: rain since 1 Jun = 351 mm (7 mm past week);</td><td>stage:</td><td>grain filling (milk to dough)</td></tr><tr><td>action:</td><td>longest dry spell = 11 d</td><td>weather:</td><td>rain since 1 Jun = 444 mm (22 mm past week);</td></tr><tr><td></td><td>one irrigation of 63 mm (631,000 L/ha;</td><td>action:</td><td>longest dry spell = 23 d</td></tr><tr><td>why:</td><td>roughly one furrow irrigation) soil moisture is running low for the current</td><td></td><td>no fertiliser or irrigation this week; continue</td></tr><tr><td></td><td>crop stage (dry spells of up to 11 d this season);</td><td>)</td><td>routine scouting (weeds, fall armyworm, leaf blight</td></tr><tr><td></td><td>model check: minor stress remains (water 5.4 d,</td><td>why:</td><td>22 mm rain last week kept soil moist; week-7 urea</td></tr><tr><td></td><td>N0.0d)</td><td></td><td>still available, extra urea now would mostly</td></tr><tr><td></td><td></td><td></td><td>leach; model check: minor stress remains</td></tr><tr><td># WEEK 7 | 22 Aug I 49 DAS</td><td></td><td></td><td>(water 2.1 d, N 0.0 d)</td></tr><tr><td>stage:</td><td>pre-tasselling (V14-VT)</td><td></td><td></td></tr><tr><td></td><td>weather: rain since 1 Jun = 357 mm (6 mm past week);</td><td></td><td># WEEK 14 | 10 0ct | 98 DAS</td></tr><tr><td>action:</td><td>longest dry spell = 18 d apply 52.2 kg/ha of urea (1.16 bags of 45 kg)</td><td></td><td>stage: dough to dent, grain hardening</td></tr><tr><td></td><td>as side-dressing near plant rows on moist</td><td></td><td>weather: rain since 1 Jun = 446 mm (2 mm past week); longest dry spell = 23 d</td></tr><tr><td></td><td>soil; cover with soil</td><td>action:</td><td>no fertiliser or irrigation this week; continue</td></tr><tr><td>why:</td><td>urea top-up keeps the crop green through the</td><td></td><td>routine scouting (weeds, fall armyworm, leaf blight</td></tr><tr><td></td><td>reproductive phase; model check: minor stress</td><td>)</td><td></td></tr><tr><td></td><td>remains (water 4.7 d, N 0.0 d)</td><td>why:</td><td>crop is maturing, so fertiliser or water will no</td></tr><tr><td># WEEK 8 | 29 Aug I 56 DAS</td><td></td><td></td><td>longer raise yield; model check: minor stress</td></tr><tr><td>stage:</td><td>tasselling and silking (most water-sensitive)</td><td></td><td>remains (water 1.6 d, N 0.0 d)</td></tr><tr><td></td><td>weather: rain since 1 Jun = 362 mm (5 mm past week);</td><td></td><td># WEEK 15 | 17 0ct | 105 DAS</td></tr><tr><td></td><td>longest dry spell = 23 d</td><td>stage:</td><td>dough to dent, grain hardening</td></tr><tr><td>action:</td><td>no fertiliser or irrigation this week; continue</td><td>weather:</td><td>rain since 1 Jun = 451 mm (5 mm past week);</td></tr><tr><td></td><td>routine scouting (weeds, fall armyworm, leaf blight</td><td>action:</td><td>longest dry spell = 23 d</td></tr><tr><td>）</td><td></td><td></td><td>no fertiliser or irrigation this week; continue</td></tr><tr><td>why:</td><td>only 5 mm rain last week, but soil still holds enough moisture; week-7 urea still available,</td><td>)</td><td>routine scouting (weeds, fall armyworm, leaf blight</td></tr><tr><td></td><td>extra urea now would mostly leach; model check:</td><td>why:</td><td>crop is maturing, so fertiliser or water will no</td></tr><tr><td></td><td>minor stress remains (water 5.6 d, N 0.0 d)</td><td></td><td>longer raise yield; model check: minor stress</td></tr><tr><td></td><td></td><td></td><td>remains (water 2.8 d, N 0.0 d)</td></tr><tr><td># WEEK 9 I 05 Sep I 63 DAS</td><td></td><td></td><td></td></tr><tr><td>stage:</td><td>tasselling and silking (most water-sensitive)</td><td></td><td># WEEK 16 | 24 0ct | 112 DAS</td></tr><tr><td></td><td>weather: rain since 1 Jun = 382 mm (20 mm past week);</td><td>stage:</td><td>physiological maturity / harvest preparation</td></tr><tr><td>action:</td><td>longest dry spell = 23 d no fertiliser or irrigation this week; continue</td><td></td><td>weather: rain since 1 Jun = 451 mm (0 mm past week); longest dry spell = 23 d</td></tr><tr><td></td><td>routine scouting (weeds, fall armyworm, leaf blight</td><td>action:</td><td>no fertiliser or irrigation this week; continue</td></tr><tr><td>)</td><td></td><td></td><td>routine scouting (weeds, fall armyworm, leaf blight</td></tr><tr><td>why:</td><td>20 mm rain last week kept soil moist; week-7 urea</td><td>)</td><td></td></tr><tr><td></td><td>still available, extra urea now would mostly</td><td>why:</td><td>crop is maturing, so fertiliser or water will no</td></tr><tr><td></td><td>leach; model check: no water or N stress</td><td></td><td>longer raise yield; model check: minor stress</td></tr><tr><td></td><td>expected under this plan</td><td></td><td>remains (water 2.9 d, N 0.0 d)</td></tr><tr><td></td><td># WEEK 10 I 12 Sep I 70 DAS</td><td></td><td></td></tr><tr><td>stage:</td><td>tasselling and silking (most water-sensitive)</td><td>stage:</td><td># WEEK 17 | 31 0ct | 119 DAS physiological maturity / harvest preparation</td></tr><tr><td></td><td>weather: rain since 1 Jun = 394 mm (12 mm past week);</td><td>weather:</td><td>rain since 1 Jun = 465 mm (14 mm past week);</td></tr><tr><td></td><td>longest dry spell = 23 d</td><td></td><td></td></tr><tr><td>action:</td><td>no fertiliser or irrigation this week; continue</td><td>action:</td><td>longest dry spell = 27 d</td></tr><tr><td></td><td></td><td></td><td>no fertiliser or irrigation this week; continue</td></tr><tr><td>)</td><td>routine scouting (weeds, fall armyworm, leaf blight</td><td>)</td><td>routine scouting (weeds, fall armyworm, leaf blight</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>why:</td><td>only 12 mm rain last week, but soil still holds</td><td>why:</td><td>crop is maturing, so fertiliser or water will no</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>enough moisture; week-7 urea still available,</td><td></td><td>longer raise yield; model check: minor stress</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>extra urea now would mostly leach; model check:</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>remains (water 2.9 d, N 0.0 d)</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>no water or N stress expected under this plan</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td># SEASON SUMMARY</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>inputs:</td><td></td></tr><tr><td></td><td># WEEK 11 | 19 Sep | 77 DAS</td><td></td><td>urea 391.4 kg/ha (8.7 bags of 45 kg) in 3 splits</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>stage: grain filling (milk to dough)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>(wk 0, 4, 7); irrigation 63 mm (1 event, wk 6); rest from rain</td></tr><tr><td>weather: rain since 1 Jun = 403 mm (9 mm past week);</td></table>

<table><tr><td>leached: 12.9 kg N/ha nitrate</td></tr></table>

## 3 Ablation of the Advisory Bounding Agent

This section gives the full results of the ablation in Section 4.6. We compare the Agri-SCHOLAR pipeline (With ABA) against two variants in which the ABA is removed and an Optuna-based OA searches on its own:

• Loose Bounds: the optimizer is limited only to fixed, numerically feasible ranges for each action, without the agronomic bounds the ABA derives from the farm context and retrieved literature.

• No Bounds: even these ranges are removed, so fertilizer and irrigation amounts have no upper limit.

The question is whether a purely profit-driven optimizer, given the same simulator and reward, can find plans a farmer could actually follow. All results are for the irrigated region over 2015–2025, and every plan is evaluated in APSIM under the true weather of the season.

In Tables 5 to 8, values for Loose Bounds and No Bounds are shown in red square brackets. The brackets mark outcomes of agronomically infeasible advisories (Appendix 3.2). They are what the simulator predicts, these are reported only to show how far an unconstrained optimizer can push the simulated metrics but these are not achievable.

## 3.1 Results

Loose Bounds. On simulated yield and profit alone, Loose Bounds is competitive with Agri-SCHOLAR. It has the highest yield in 4 of 11 seasons and the highest net profit in 7 seasons, with its largest gains in 2016 and 2025 (over 3,000 kg ha<sup>−1</sup> above Agri-SCHOLAR). Environmental totals are similar: Agri-SCHOLAR has the lowest drainage in 6 seasons and the lowest leaching in 5, against 3 and 6 for Loose Bounds.

The totals are weather-dependent, when heavy rain follows a large early dose, losses are pronounced: in 2017 Loose Bounds leached 30.73 kg ha<sup>−1</sup> of nitrate against 5.80 for Agri-SCHOLAR (over five times as much), and in 2023 it leached 22.52 against 12.90. Across all seasons, its sustainability penalty is 20.1% higher than that of Agri-SCHOLAR (Table 3).

No Bounds. Removing all bounds is far worse. Net profit is negative in four seasons (2019, 2022, 2024, 2025) and ranges from −\$2,887 to \$9,208 ha<sup>−1</sup>. No Bounds reaches the highest yield in 2015 and 2024, but severe penalty costs leave its profit well below that of the bounded variants. Nitrate leaching is the highest of the three variants in every season, reaching 694.18 kg ha<sup>−1</sup> in 2025 and 3,270.66 kg ha<sup>−1</sup> in 2019. Deep drainage exceeds 800 mm in 2019 and 1,000 mm in 2020. It comes close to the bounded variants only in seasons such as 2023, where the realized weather happened to suit its large early applications.

Table 5: Yield (kg/ha) per season: Agri-SCHOLAR vs. without ABA (irrigated region, 2015–2025). Highest value per season in bold. Red values in [ ] come from agronomically infeasible advisories.
<table><tr><td rowspan="2">Year</td><td colspan="3">w/o ABA</td></tr><tr><td>With ABA</td><td>Loose Bounds</td><td>No Bounds</td></tr><tr><td>2015</td><td>9,649</td><td>[10,070]</td><td>[10,170]</td></tr><tr><td>2016</td><td>5,819</td><td>[9,200]</td><td>[9,022]</td></tr><tr><td>2017</td><td>8,798</td><td>[5,419]</td><td>[4,979]</td></tr><tr><td>2018</td><td>6,945</td><td>[7,162]</td><td>[4,208]</td></tr><tr><td>2019</td><td>6,927</td><td>[6,927]</td><td>[6,927]</td></tr><tr><td>2020</td><td>6,401</td><td>[4,679]</td><td>[4,356]</td></tr><tr><td>2021</td><td>9,031</td><td>[7,667]</td><td>[7,968]</td></tr><tr><td>2022</td><td>7,854</td><td>[8,127]</td><td>[7,181]</td></tr><tr><td>2023</td><td>11,028</td><td>[10,544]</td><td>[10,405]</td></tr><tr><td>2024</td><td>4,524</td><td>[5,805]</td><td>[6,467]</td></tr><tr><td>2025</td><td>5,095</td><td>[8,163]</td><td>[6,226]</td></tr></table>

Table 6: Net profit (\$/ha) per season: Agri-SCHOLAR vs. without ABA (irrigated region, 2015–2025). Highest value per season in bold. Red values in [ ] come from agronomically infeasible advisories.
<table><tr><td rowspan="2">Year</td><td colspan="3">w/o ABA</td></tr><tr><td>With ABA</td><td>Loose Bounds</td><td>No Bounds</td></tr><tr><td>2015</td><td>9,470.27</td><td>[9,966.65]</td><td>[3,733.94]</td></tr><tr><td>2016</td><td>5,295.06</td><td>[8,883.95]</td><td>[2,188.71]</td></tr><tr><td>2017</td><td>8,705.98</td><td>[4,831.18]</td><td>[2,767.43]</td></tr><tr><td>2018</td><td>6,509.34</td><td>[6,788.55]</td><td>[2,320.83]</td></tr><tr><td>2019</td><td>6,571.44</td><td>[6,581.54]</td><td>[-2,887.00]</td></tr><tr><td>2020</td><td>5,792.05</td><td>[4,231.49]</td><td>[1,045.02]</td></tr><tr><td>2021</td><td>8,633.07</td><td>[7,311.67]</td><td>[1,287.78]</td></tr><tr><td>2022</td><td>7,338.90</td><td>[7,724.62]</td><td>[-103.31]</td></tr><tr><td>2023</td><td>10,745.40</td><td>[9,857.63]</td><td>[9,208.20]</td></tr><tr><td>2024</td><td>3,753.07</td><td>[5,324.62]</td><td>[-750.11]</td></tr><tr><td>2025</td><td>4,461.40</td><td>[7,696.51]</td><td>[-301.95]</td></tr></table>

Table 7: Cumulative deep drainage (mm) per season: Agri-SCHOLAR vs. without ABA (irrigated region, 2015– 2025). Lowest value per season in bold. Red values in [ ] come from agronomically infeasible advisories.
<table><tr><td rowspan="2">Year</td><td colspan="2">w/o ABA</td></tr><tr><td>With ABA Loose Bounds</td><td>No Bounds</td></tr><tr><td>2015</td><td>45.00</td><td>[42.63]</td><td>[42.61]</td></tr><tr><td>2016</td><td>282.00</td><td>[228.95]</td><td>[233.58]</td></tr><tr><td>2017</td><td>84.00</td><td>[297.67]</td><td>[386.83]</td></tr><tr><td>2018</td><td>197.00</td><td>[201.08]</td><td>[569.57]</td></tr><tr><td>2019</td><td>194.00</td><td>[202.90]</td><td>[818.60]</td></tr><tr><td>2020</td><td>382.00</td><td>[304.65]</td><td>[1,089.61]</td></tr><tr><td>2021</td><td>291.00</td><td>[269.60]</td><td>[228.84]</td></tr><tr><td>2022</td><td>299.00</td><td>[303.53]</td><td>[339.13]</td></tr><tr><td>2023</td><td>151.00</td><td>[263.82]</td><td>[342.66]</td></tr><tr><td>2024</td><td>407.00</td><td>[335.67]</td><td>[340.41]</td></tr><tr><td>2025</td><td>349.00</td><td>[363.58]</td><td>[598.73]</td></tr></table>

Table 8: Cumulative $\Nu { 0 _ { 3 } }$ leaching (kg/ha) per season: Agri-SCHOLAR vs. without ABA (irrigated region, 2015– 2025). Lowest value per season in bold. Red values in [ ] come from agronomically infeasible advisories.
<table><tr><td rowspan="2"></td><td colspan="3">w/o ABA</td></tr><tr><td>With ABA</td><td>Loose Bounds</td><td>No Bounds</td></tr><tr><td>2015</td><td>3.10</td><td>[2.95]</td><td>[3.11]</td></tr><tr><td>2016</td><td>23.70</td><td>[18.76]</td><td>[76.44]</td></tr><tr><td>2017</td><td>5.80</td><td>[30.73]</td><td>[33.60]</td></tr><tr><td>2018</td><td>16.40</td><td>[15.79]</td><td>[54.83]</td></tr><tr><td>2019</td><td>17.90</td><td>[26.34]</td><td>[3,270.66]</td></tr><tr><td>2020</td><td>34.70</td><td>[20.94]</td><td>[275.83]</td></tr><tr><td>2021</td><td>28.70</td><td>[34.59]</td><td>[102.15]</td></tr><tr><td>2022</td><td>28.10</td><td>[25.10]</td><td>[296.62]</td></tr><tr><td>2023</td><td>12.90</td><td>[22.52]</td><td>[36.55]</td></tr><tr><td>2024</td><td>40.10</td><td>[28.90]</td><td>[364.21]</td></tr><tr><td>2025</td><td>36.50</td><td>[36.68]</td><td>[694.18]</td></tr></table>

## 3.2 Observations

The optimizer exploits blind spots in the simulator and reward. Without the ABA, the optimizer treats APSIM as a black box and maximizes the reward, which is reliable only if the simulator and reward capture everything that matters in the field. They do not. APSIM does not model the smallest dose that can be spread evenly, or the effort of a field operation beyond the labour cost. The reward penalizes leaching, deep drainage, waterlogging, and nitrogen oversupply, but has no term for whether a dose suits the crop stage or can be applied at all. Because each application carries a fixed labour cost $( p _ { L } = 3 5 0 \mathrm { L C U } )$ , the optimizer is drawn to two kinds of schedules that look efficient under the reward but are unsound on the farm

• Excessive nitrogen at planting. In 2019, the optimizer front-loaded nitrogen into a single application, Loose Bounds applied 190 kg ha<sup>−1</sup> of UreaN at planting, and No Bounds applied $\mathrm { 6 , 0 6 2 . 2 \ k g h { a } ^ { - 1 } }$ . A maize seedling has almost no root system and takes up little nitrogen, so such a dose sits in the soil exposed to leaching, and urea near germinating seed risks ammonia injury that APSIM never penalizes.

• Doses too small to apply. The optimizer also adds token top-ups of $2 { - } 4 \ \mathrm { k g } \mathrm { h a } ^ { - 1 }$ , such as $2 . 8 \mathrm { { k g } h a ^ { - 1 } }$ of UreaN at week 8 of 2024. These buy a marginal simulated yield gain while staying below the leaching penalty, but 2.8 kg is about 6% of a single 45 kg bag. It cannot be calibrated or spread evenly with standard equipment, and no farmer would spend a person-day on it.

Simulated gains are not evidence of better advice. Large single doses are also a bet that the following weeks will not bring heavy rain, since the optimizer plans on the constructed weather available at advisory time. The 2019 No Bounds plan shows the failure: under the true weather it leached 3,270.66 kg ha<sup>−1</sup> of nitrate yet yielded $6 , 9 2 7 \mathrm { { k g } h a ^ { - 1 } }$ , identical to both bounded variants. The extra nitrogen bought no yield and produced a net loss of \$2,887 ha<sup>−1</sup>. Loose Bounds’ apparent advantage in several seasons is likewise an artefact of a model and reward that cannot judge agronomic quality. The scores reflect skill at exploiting the model, not the quality of the advice.

The ABA closes these gaps. By restricting the search to doses and timings that agronomic guidance supports for the current crop stage, the ABA yields plans a farmer can follow. For example, In the 2023 case study (Appendix 2), Agri-SCHOLAR splits 180 kg ha<sup>−1</sup> of UreaN into a starter dose at sowing (59 kg ha<sup>−1</sup>), a main top-dress at rapid vegetative growth $( 9 7 \mathrm { { k g } h a ^ { - 1 } ) }$ , and a top-up before tasselling (24 kg ha<sup>−1</sup>). Each dose matches crop demand at that stage and can be applied in whole or half bags. The cost is a small loss in simulated profit in some seasons, in exchange for avoiding the leaching and profit losses seen when the bounds are removed.