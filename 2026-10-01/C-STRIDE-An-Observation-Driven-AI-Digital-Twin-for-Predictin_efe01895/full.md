# C-STRIDE: An Observation-Driven AI Digital Twin for Predicting Basin-Wide Flood Fields from Sparse Stream-Gauge Histories

Yanjie Tong<sup>a</sup>, Phillip Si<sup>a</sup>, Yuan Qiu<sup>a</sup>, Peng Chen<sup>a,</sup>∗

<sup>a</sup>School of Computational Science and Engineering, Georgia Institute of Technology, 756 West Peachtree Street Northwest, Atlanta, 30308, GA, USA

## Abstract

Emergency managers need to know where floodwater is, how deep it is, and how it will change over the coming hours across an entire river basin. During a flood, however, real-time measurements come from only a handful of stream gauges, and high-resolution hydrodynamic models are too costly to rerun each time new data arrive or to run as large ensembles. We present C-STRIDE, an observation-driven AI digital twin that turns short records from a few stream gauges, together with terrain and rainfall, into basin-wide maps of water depth and extends these predictions up to a day ahead. It is trained on simulations from a calibrated two-dimensional hydrodynamic model and needs no separate data-assimilation step. In the Des Plaines River basin near Chicago, six gauges inform predictions over 4.2 million 30-m grid cells. Terrain improves the predictions most, rainfall keeps errors from growing over longer horizons, and together they reduce errors by about 40% compared with gauge records alone. When future rainfall is known, errors remain near 15% one day ahead, compared with nearly 40% without rainfall. Given real instead of simulated gauge records, the model shifts its predictions toward the observed hydrographs at three of six gauges without retraining, and it runs about 150 times faster than the hydrodynamic model. These results show how sparse gauges, terrain, and rainfall can be combined into fast, continuously updated flood predictions, a step toward operational flood digital twins that still requires testing with real-time data and rainfall forecasts.

Keywords: AI-driven digital twins, Flood prediction, Sparse stream-gauge observations, Implicit neural representations, Surrogate modelling, Shallow water equations

## 1. Introduction

## 1.1. Hydrological digital twins for flood decision support

Flooding threatens lives, infrastructure, and economic activity in cities and regions worldwide (Rentschler et al., 2022; Tellman et al., 2021). Efective response requires information about where water is accumulating, how deep it is, and how inundation may evolve as an event develops. These questions connect local measurements to spatially distributed predictions: a gauge records conditions at one location, while emergency planning requires flood information across roads, neighborhoods, and critical facilities. A useful forecasting system must therefore combine the physical structure of flood dynamics with observations and rapidly updated predictions.

Digital twins provide a framework for this integration. A digital twin is commonly described as a virtual representation of a physical system that is updated with data from that system, has predictive capability, and informs decisions, with two-way interaction between the virtual and physical systems as a central element (National Academies of Sciences, Engineering, and Medicine, 2024). In hydrology, Digital

Twin Earth Hydrology connects Earth-observation products with hydrological modeling to investigate watercycle conditions and flood-risk scenarios (Brocca et al., 2024), observation-coupled hydraulic models support real-time monitoring of drainage systems (Bartos and Kerkez, 2021), and coupled hydrologic–hydrodynamic twins support flood forecasting in data-scarce regions (Rápalo et al., 2024). For the flood-prediction problem considered here, we organize the required capabilities around three complementary pillars:

(P1) Physics-anchored simulation. A reference model represents the governing hydrodynamics and is evaluated against observations, providing the physical target for a learned surrogate.

(P2) Fast surrogate replication. A computationally eficient model reproduces spatial flood evolution, making repeated forecasts and scenario evaluation practical.

(P3) Observation-driven state estimation. An observation interface uses incoming measurements to update the estimated basin state and its associated flood field.

Integrating these capabilities is particularly demanding for basin-wide, high-resolution prediction, where sparse measurements must constrain millions of spatial degrees of freedom. In this paper, we develop an observation-driven flood digital twin: a model that provides fast spatial prediction (P2) and observationdriven state estimation (P3), trained on a previously evaluated simulator that supplies the physical reference (P1). Calibrated uncertainty and decision feedback remain necessary for an operational twin but are outside the present scope; they are discussed in Section 5.

## 1.2. Hydrodynamic models, learned surrogates, and observation integration

Hydrodynamic models and data assimilation. Hydrodynamic solvers remain central to flood modeling be cause they represent transport, topographic controls, and friction through the shallow water equations. GPU acceleration, including the SynxFlow framework (Xia et al., 2017; Xia and Liang, 2018; Xia et al., 2019), makes high-resolution simulations increasingly accessible. Nevertheless, repeated basin-wide simulations remain expensive when forecasts must be updated frequently, ensembles are required, or many scenarios must be evaluated. Connecting such models to observations has a long history in flood forecasting. Realtime updating corrects river-model states using gauge data (Madsen and Skotner, 2005); ensemble Kalman filters update inundation models with spatially distributed water-level measurements (Neal et al., 2007); satellite-based assimilation updates river-network and inundation models using observed water levels and flood extents (García-Pintado et al., 2015; Hostache et al., 2018); and Pipedream combines hydraulic simulation with Kalman filtering to estimate and forecast drainage-network states (Bartos and Kerkez, 2021). These methods integrate observations through a physics-based model. Here, we examine whether a learned model can map sparse gauge histories directly to the next distributed flood field at lower computational cost.

Learning-based flood surrogates. Deep learning has reduced the cost of several flood-modeling tasks (Bentivoglio et al., 2022). Convolutional networks emulate flood-depth maps for urban pluvial (Löwe et al., 2021; Guo et al., 2021) and fluvial (Kabir et al., 2020) flooding from rainfall, inflow, and topographic inputs. SWE-GNN instead advances hydraulic variables on a computational graph using terrain and current flow conditions, with a propagation rule motivated by finite-volume hydraulics (Bentivoglio et al., 2023). At the scale of individual gauges, long short-term memory (LSTM) networks driven by meteorological forcing rival or exceed conceptual rainfall–runof models (Kratzert et al., 2018, 2019) and underpin operational flood forecasting at large scales (Nevo et al., 2022; Nearing et al., 2024). These models difer both in what they predict (event-maximum inundation, time-varying hydraulics, or streamflow at gauges) and in what they need at run time, so rainfall-driven, state-driven, and observation-driven surrogates address diferent problems.

Field reconstruction from sparse sensors. A separate line of work reconstructs full spatial fields from a few point sensors. Shallow decoders map instantaneous sensor values to fluid-flow fields (Erichson et al., 2020); Voronoi-tessellation inputs let convolutional networks handle arbitrary sensor layouts (Fukami et al.,

2021); the Senseiver encodes sparse observations with attention and decodes the field at arbitrary query coordinates (Santos et al., 2023); and SHRED encodes sensor time histories with an LSTM to reconstruct spatiotemporal fields (Williams et al., 2024). Coordinate-based decoders conditioned on latent dynamics have also been used to forecast solutions of partial diferential equations on irregular geometries (Yin et al., 2023; Serrano et al., 2023). These methods have mostly been demonstrated on fluid-dynamics, climate, or synthetic benchmarks rather than on basin-wide flood fields constrained by real gauge networks.

Conditional latent dynamics for metropolitan floods. A particularly relevant reference is the Conditional Latent Dynamics Network (CLDNet) (Si et al., 2026). It combines rainfall-driven latent dynamics with a coordinate-based decoder conditioned on elevation, slope, and Manning roughness. Pointwise decoding supports irregular watersheds and queries at gauge coordinates without requiring a dense output grid during training. On the Des Plaines River basin, CLDNet demonstrates metropolitan-scale surrogate modeling using a SynxFlow reference evaluated against USGS observations. We reuse this case study and compare against the CLDNet surrogate and against its combination with latent ensemble score filtering (LD-EnSF) (Xiao et al., 2026), which assimilates the same six gauges.

The observation-to-field problem. Emulating a simulator and estimating the current flood state are related but diferent tasks. A simulator-style surrogate advances a given state under given forcing, whereas an observation-driven model must infer the distributed state from incomplete measurements. A maximumdepth map summarizes event severity but cannot describe when inundation arrives or how quickly it recedes; a time-dependent field can, but requires a representation of the evolving state. Rainfall information remains valuable in either case, particularly for forecasting, but does not replace measurements of the evolving hydraulic response. Conversely, gauge histories constrain recent basin behavior but do not determine future precipitation. An efective architecture should combine these information sources while making their respective contributions explicit. Table 1 summarizes representative approaches by the information they require at run time and the output they produce.

## 1.3. From sparse gauge histories to spatial flood fields

Stream gauges provide a practical observation interface because they record the evolving river response at a small number of mainstem and tributary sites. Yet inferring a flood field from these records is underdetermined at any single instant. Similar local depths can occur during diferent event phases and correspond to diferent conditions elsewhere in the basin. A temporal history provides information about the direction and rate of change, whereas terrain describes persistent spatial controls that cannot be inferred directly from a few gauge values. Forecasting adds another requirement: the inferred state must evolve consistently with the forcing supplied beyond the observation window.

The STRIDE framework (Tong and Chen, 2026) factorizes sparse-observation reconstruction into a temporal encoder and a coordinate-conditional decoder. The encoder maps an observation history to a compact latent state, and the decoder evaluates the field at a requested spatial coordinate. The rationale comes from delay-embedding theory: if a short history of sensor readings is enough to identify the system state, a network can learn the map from that history to the field. STRIDE makes this precise under a stable delay-observability assumption (Section 3.1). The assumption is conditional: a finite sensor history need not identify the full state for arbitrary sensor placements or event distributions, so its adequacy here must be examined empirically through field-prediction, gauge, and window-length tests.

Applying this formulation to floods raises three issues. First, local depth depends on elevation, slope, and roughness, so the decoder should receive explicit terrain features alongside the latent basin state. Second, observations and precipitation have diferent temporal roles: gauge measurements describe the current hydraulic response, whereas rainfall can influence downstream depth after a delay. A model that uses both must keep the direct gauge signal while adding the rainfall information needed to predict what happens next. Third, the evaluation must distinguish agreement with simulator-generated fields from agreement with measured hydrographs. Matching a simulator that has been checked against gauges is useful evidence, but it does not validate every surrogate output against observations.

<table><tr><td>Approach</td><td>Run-time inputs</td><td>Output</td><td>Spatial representation</td></tr><tr><td>DTE Hydrology (Brocca</td><td>Earth observations, hydrological forcing</td><td>Water-cycle states and scenarios</td><td>Gridded</td></tr><tr><td>et al., 2024) Pipedream (Bartos</td><td>Hydraulic model, sensor data</td><td>Drainage-network states and</td><td>1-D network</td></tr><tr><td>and Kerkez, 2021) Inundation data assimilation (Neal</td><td>Hydraulic model, gauge or satellite data</td><td>forecasts Updated water levels and extent</td><td>Model grid</td></tr><tr><td>et al., 2007; Hostache et al., 2018) U-FLOOD (Löwe</td><td>Rainfall, terrain</td><td>Event-maximum depth</td><td>Raster</td></tr><tr><td>et al., 2021) SWE-</td><td>Current hydraulic state,</td><td>Time-evolving hydraulic state</td><td>Mesh graph</td></tr><tr><td>GNN (Bentivoglio et al., 2023) SHRED, Senseiver (Williams</td><td>terrain Sparse sensor histories or values</td><td>Full field</td><td>Grid or continuous queries</td></tr><tr><td>et al., 2024; Santos et al., 2023) CLDNet (Si et al.,</td><td>Rainfall, terrain</td><td>Time-evolving depth</td><td>Continuous queries</td></tr><tr><td>2026) C-STRIDE (this</td><td></td><td></td><td></td></tr><tr><td>work)</td><td>optional rainfall</td><td>Sparse gauge histories, terrain, Next-step and multi-step depth predictions</td><td>Continuous queries</td></tr></table>

Table 1: Representative hydrological digital-twin, data-assimilation, flood-surrogate, and sparse-sensing approaches, summa rized by the information they require at run time, the output they produce, and how they represent space. Entries describe modeling interfaces rather than a common accuracy benchmark.

We therefore develop Conditional STRIDE (C-STRIDE), which combines an LSTM encoder, a terrainconditioned FMMNN decoder, and a second LSTM that forecasts the gauge readings so that predictions can be extended beyond one step. The encoder receives sparse water-depth histories, augmented with precipitation in the full configuration. The decoder combines the latent state with query coordinates, elevation, slope magnitude, and Manning roughness to predict water depth one step beyond the observation window. For multi-step forecasts, predicted gauge values and the supplied future precipitation are fed back step by step. Terrain-only and forcing-only variants isolate the contributions of the two conditioning sources.

The coordinate decoder suits an irregular watershed, where the active domain occupies only part of the enclosing raster. During training, many locations can be sampled for the same latent basin state, which keeps memory use manageable. At inference, the same representation can be evaluated over an event-specific reduced grid or at a selected set of locations. The same trained model can therefore be tested at both field and gauge scales, without assuming that accuracy at observed sites implies accuracy across the basin.

The trained encoder provides an observation-driven state estimate directly, without an assimilation filter. The estimate is deterministic: the model does not maintain a posterior covariance or produce calibrated uncertainty intervals. Likewise, the coordinate decoder provides flexible spatial evaluation, but querying a finer grid does not by itself establish accuracy beyond the resolution of the reference data.

## 1.4. Contributions and evaluation scope

The study evaluates C-STRIDE in the Des Plaines River basin in the greater Chicago area, using 4,188,840 active cells at 30 m resolution and six USGS gauge locations. The dataset described in Section 2.3 contains 94 storm-driven simulations, with 90 used for training and four held out for testing (Si et al., 2026). The April 2013 flood, the flood of record at several of these gauges, connects the simulator reference, the learned prediction, and the observed gauge hydrographs for the same event. We organize the study around a question relevant to any flood digital twin: what do sparse gauge histories, terrain, and precipitation each contribute to next-step and multi-step flood prediction? The contributions are:

(C1) An observation-driven AI digital twin for basin-wide flood prediction. We adapt STRIDE to combine gauge-history encoding with terrain-conditioned continuous decoding and precipitation-conditioned forecasting. The model provides fast spatial prediction (P2) and observation-driven state estimation (P3), is trained on a simulator evaluated against USGS observations (P1), and can be queried at individual locations or over the basin.

(C2) An empirical separation of spatial and temporal conditioning. Terrain gives the larger individual improvement in next-step field prediction, while precipitation limits error growth during multi-step forecasting. Joint conditioning reduces four-event mean relative depth error by approximately 41% compared with vanilla STRIDE. Full C-STRIDE reaches 14.48% error at a 24-h horizon with reference future rainfall, and a perturbation experiment tests sensitivity to that assumption.

(C3) Evaluation across fields, gauges, and observed inputs. We compare predicted depth and flood extent with SynxFlow on event-specific reduced grids, test hydrograph consistency at mapped gauge cells, replace simulated histories with USGS observations at inference time, and evaluate tributary gauges withheld from the encoder inputs. Additional tests vary history length and train models with reduced, missing, or noisy inputs.

(C4) An assessment of computational cost. We report field and six-gauge inference times, multi-step evaluation cost, and ofline training cost. On the same NVIDIA L40S GPU, next-step prediction of depth fields of a 96-h event is approximately 150× faster than a SynxFlow simulation, although the two computations difer in scope.

The remainder of the paper is organized as follows. Section 2 describes the study area, data, and problem setup. Section 3 presents the C-STRIDE model, its training, and the evaluation metrics. Section 4 evaluates next-step prediction, hydrographs, flood extent, forecasting, sensitivity, and computational cost. Section 5 interprets the results, discusses operational implications, and states limitations. Section 6 concludes.

## 2. Study area, data, and problem setup

This section describes the simulator that generates the reference data (Section 2.1), defines the prediction problem (Section 2.2), and summarizes the Des Plaines River basin dataset and gauge network (Section 2.3).

## 2.1. High-fidelity simulator for generating the synthetic dataset

We generated the synthetic training and test datasets using the open-source library SynxFlow (Xia et al., 2017; Xia and Liang, 2018; Xia et al., 2019; SynxFlow Developers, 2023). The simulator solves the twodimensional shallow-water equations using a first-order Godunov-type finite-volume method on a uniform rectangular grid. Cells outside the watershed are masked, so the rectangular grid can represent an irregular domain. SynxFlow uses surface reconstruction to preserve well-balanced treatment of bed elevation, the HLLC Riemann solver to compute interface fluxes, and a minmod limiter to reconstruct bed gradients (Xia and Liang, 2018). It discretizes the flux and bed-slope terms explicitly and treats the stif friction term implicitly. The time step is chosen adaptively according to the CFL condition.

The simulator takes as input the initial water depth $h _ { 0 } .$ , the initial unit-width discharges $h _ { 0 } u _ { 0 }$ and $h _ { 0 } v _ { 0 }$ in the x- and y-directions (where $u _ { 0 }$ and $v _ { 0 }$ are depth-averaged velocities), bed elevation b, Manning’s roughness coeficient $n ,$ and a spatially varying, discrete-time precipitation series $\{ r _ { t } \}$ . It outputs the state $\left( h _ { t } , h _ { t } u _ { t } , h _ { t } v _ { t } \right)$ at future times t.

## 2.2. Sparse-observation prediction problem

Let $\Omega \subset \mathbb { R } ^ { 2 }$ denote the physical domain, and let $[ 0 , T ]$ be the simulation horizon. We uniformly partition the interval $[ 0 , T ]$ into $N _ { T }$ subintervals of length $\Delta t = T / N _ { T }$ . We consider snapshots of $( h , h u , h v )$ at times $t _ { k } = k \Delta t$ , for $k = 1 , \dots , N _ { T }$ . The model is trained to predict only water depth $h ( t _ { k } , \xi )$ at locations $\xi \in \Omega$ rather than the unit-width discharges $( h u , h v )$

We measure water depth through $N _ { s }$ point sensors located at $\{ \xi ^ { ( 1 ) } , \dots , \xi ^ { ( N _ { s } ) } \} \subset \Omega$ . The observation vector at time $t _ { k }$ is

$$
y _ { k } : = \bigl ( h ( t _ { k } , \xi ^ { ( 1 ) } ) , \dots , h ( t _ { k } , \xi ^ { ( N _ { s } ) } ) \bigr ) \in \mathbb { R } ^ { N _ { s } } .\tag{1}
$$

Other observed quantities, such as water-surface elevation (WSE) or discharge, could be used in the same way with suitable preprocessing and training. We write $y _ { k - K : k } = ( y _ { k - K } , \dots , y _ { k } )$ for a window of $K + 1$ consecutive observations and $p _ { k } \in \mathbb { R } ^ { d _ { p } }$ for the precipitation field at time $t _ { k }$ . Given a forecast origin $k ,$ a horizon $m \geq 1$ , and a query location $\xi ,$ the model estimates

$$
\tilde { h } ( t _ { k + m } , \xi ) \ = \ S _ { \theta } \big ( y _ { k - K : k } , \ p _ { k - K : k + m - 1 } , \ \xi , \ \phi ( \xi ) \big ) ,\tag{2}
$$

where $\phi ( \xi )$ is a static terrain feature vector (Section 3.2). We consider horizons $m = 1 , 2 , \hdots , 2 4$ . For $m = 1$ , we predict water depth at the next time step from historical observations $y _ { k - K : k }$ and precipitation $p _ { k - K : k }$ . For $m \geq 2$ , the model additionally requires future precipitation $p _ { k + 1 } , \ldots , p _ { k + m - 1 }$ . The default multi-step evaluation supplies the reference future precipitation used to drive SynxFlow; Section 4.4 also tests perturbed rainfall. Operational use would instead require precipitation forecasts.

## 2.3. Des Plaines River basin case study

We reuse the Des Plaines River basin (HUC8 07120004) simulation dataset and the six USGS gauges of Si et al. (2026). SynxFlow solves the two-dimensional shallow-water equations on a fixed 30 m USGS 3DEP digital elevation model. The computational domain contains 4,188,840 active watershed cells within a $5 , 0 7 5 \times 1 , 6 6 1$ rectangular grid. NLCD 2021 land cover (Dewitz, 2023) determines the Manning coeficients, which are 0.02 for water-covered cells and 0.05 for land-covered cells. We refer to Si et al. (2026) for more details about the simulation setup and data.

The dataset contains 94 simulations driven by spatially varying NCEP Stage IV precipitation fields from 2002–2024 (Lin and Mitchell, 2005). The rainfall fields come from a broader Midwestern storm archive, so they do not all represent storms that occurred over the Des Plaines basin. At each hourly step, $\mathrm { ~ a ~ 3 9 ~ } \times \mathrm { ~ 1 3 ~ }$ precipitation grid supplies 507 forcing values. Each simulation starts from a shared spun-up flow state and runs for 96 h, producing hourly water depth and discharge fields. We predict water depth only and discard the initial state, retaining the states at original time indices $1 , \ldots , 9 6$ . Each state is paired with the precipitation at the preceding index $( 0 , \ldots , 9 5 )$ , so each precipitation input covers the hour before its paired state.

We use 90 trajectories for training and four for testing, including the April 2013 flood trajectory, which we use for comparison with USGS observations. Our sparse-observation network uses four Des Plaines mainstem gauges and two tributary gauges from Si et al. (2026). Unless stated otherwise, gauge observations are noisefree simulated water depth sampled at the sensor cells. With $N _ { s } = 6$ sensors and the default window of $K + 1 = 1 2$ stored snapshots, each window contains 72 gauge values. When precipitation is included, each time step contains $6 + 5 0 7 = 5 1 3$ features, giving an input tensor of size $B \times ( K + 1 ) \times 5 1 3$ for a batch of B windows. We summarize the dataset and sparse-observation configuration in Table 2. Section 4.5.2 also considers missing tributary observations and noisy observations.

## 3. The C-STRIDE model

This section develops the Conditional STRIDE (C-STRIDE) model, summarized in Fig. 1. We start from the original STRIDE factorization (Section 3.1), introduce terrain conditioning, precipitation input, and the multi-step forecasting scheme (Section 3.2), describe training on the basin-wide, high-resolution domain (Section 3.3), and define the inference protocol and evaluation metrics (Section 3.4).

<table><tr><td>Property</td><td>Value</td></tr><tr><td>Domain shape</td><td>Non-rectangular (NaN-masked)</td></tr><tr><td>Grid size</td><td>5,075 × 1,661</td></tr><tr><td>Active in-domain cells</td><td>4,188,840</td></tr><tr><td>Spatial resolution</td><td>30m</td></tr><tr><td>DEM source</td><td>USGS 3DEP</td></tr><tr><td>Manning coefficient</td><td>0.02 (water) / 0.05 (land)</td></tr><tr><td>Precipitation source</td><td>Stage IV QPE, 2002–2024</td></tr><tr><td>Precipitation grid</td><td>39 × 13</td></tr><tr><td># precipitation features  $d _ { p }$ </td><td>507 per time step</td></tr><tr><td># trajectories</td><td>94 (90 train, 4 test including 1 USGS comparison)</td></tr><tr><td>Simulation horizon</td><td>96 h 1h</td></tr><tr><td>Output time interval</td><td></td></tr><tr><td>Snapshots per trajectory</td><td> $N _ { T } = 9 6$  after discarding the initial state</td></tr><tr><td>Retained state indices</td><td> $1 , 2 , \ldots , 9 6$  (original time indices)</td></tr><tr><td>Paired precipitation indices Learned output</td><td> $0 , 1 , \ldots , 9 5$  (original time indices) Water depth</td></tr><tr><td></td><td></td></tr><tr><td>Sensor count  $N _ { s }$ </td><td>6 USGS gauges (4 mainstem, 2 tributary)</td></tr><tr><td>Observed quantity</td><td>Water depth</td></tr><tr><td>Forecast horizons m</td><td>1–24 steps (reported at time step  $1 , 4 , 6 , 1 2 , 2 4 )$ </td></tr><tr><td>Simulator evaluation</td><td>6 USGS gauges, April 2013 flood</td></tr></table>

Table 2: Des Plaines River basin dataset and sparse-observation configuration.

## 3.1. STRIDE recap: delay-embedded continuous decoding

The STRIDE framework (Tong and Chen, 2026) reconstructs a continuous spatiotemporal field from a short window of sparse point measurements by factorizing the reconstruction operator into a temporal encoder G and a coordinate-conditional spatial decoder ${ \mathcal { F } } \mathrm { : }$ :

$$
\tilde { h } ( t _ { k } , \xi ) = \mathcal { F } \big ( \xi , z _ { k } \big ) , \qquad z _ { k } = \mathcal { G } ( y _ { k - K : k } ) ,\tag{3}
$$

where $z _ { k } ~ \in ~ \mathbb { R } ^ { d _ { z } }$ is a latent state and $K + 1$ is the window length. The rationale comes from delayembedding theory. For autonomous systems, delay-coordinate maps built from a generic observable embed the system’s invariant set (Takens, 1981; Mañé, 1981); Stark (1999) extended such results to systems driven by deterministic forcing, and Botvinick-Greenhouse et al. (2025) gave a measure-theoretic formulation. Building on these results, Tong and Chen (2026) show that, when the state on a finite-dimensional parametric invariant set is stably identifiable from K+1 delayed observations (stable delay observability), the map in (3) can be approximated to arbitrary accuracy by a recurrent encoder and a coordinate decoder. STRIDE’s default instantiation pairs a Long Short-Term Memory (LSTM) encoder (Hochreiter and Schmidhuber, 1997) with a modulated Fourier Multi-Component and Multi-Layer Neural Network (FMMNN) decoder (Zhang et al., 2026) whose frozen random Fourier basis acts as an implicit regularizer.

Two features of the original STRIDE limit its direct use for basin-wide flood prediction. First, the decoder sees only the query coordinate and the latent state, without static terrain features such as elevation, slope, and roughness, and the encoder sees only the sensor values, without external forcing. Flood dynamics are driven by precipitation, which the gauges record only indirectly and with a delay. Delay-embedding results for forced systems treat the forcing as part of the system state (Stark, 1999), which motivates explicitly supplying the precipitation history as input. Second, the original formulation estimates the field at the end of the observation window and does not specify how to predict beyond it. C-STRIDE trains the encoder– decoder for the next field and uses an auxiliary gauge forecaster for longer horizons.

![](images/6282d2418178e206402fce66c000797146f591967608941eac46faafd1dac32c.jpg)  
Figure 1: Overview of the C-STRIDE architecture. Sparse observation histories $y _ { k - K : k }$ and precipitation histories $p _ { k - K : k }$ are encoded by G into a latent state $z _ { k }$ , which conditions the decoder ${ \mathcal F } .$ The decoder is queried at spatial coordinates ξ together with terrain feature vectors $\phi ( \xi )$ , comprising elevation $b _ { z } ( \xi )$ , slope $b _ { g } ( \xi )$ , and Manning coeficient $\tilde { n } ( \xi )$ , to produce the next-step depth $\tilde { h } ( t _ { k + 1 } , \xi )$ . The example panels illustrate sensor observations, precipitation snapshots, query locations, terrain features, and a predicted water-depth field.

## 3.2. C-STRIDE architecture

Recurrent encoder. The encoder G is an LSTM that processes the observation window sequentially. In forcing-aware configurations, each observation is concatenated with the paired precipitation field, $\tilde { y } _ { i } ~ =$ $[ y _ { i } , p _ { i } ] \in \mathbb { R } ^ { N _ { s } + d _ { p } } ;$ otherwise $\tilde { y } _ { i } = y _ { i }$ . Writing $s _ { i }$ for the LSTM hidden state, the update is

$$
s _ { i } = g \bigl ( s _ { i - 1 } , \tilde { y } _ { i } \bigr ) , \qquad i = k - K , \ldots , k ,\tag{4}
$$

where $g$ is the LSTM cell, and the latent state is $z _ { k } : = s _ { k } \in \mathbb { R } ^ { d _ { z } }$ (the hidden state of the final LSTM layer).   
Hidden and cell states are reset to zero for each window.

Gauge forecaster and multi-step prediction. Unlike (3), which estimates the field at the end of the window, the C-STRIDE encoder–decoder uses the window ending at k to predict the depth field at k+1. A separately trained LSTM, G′, forecasts the gauge depths one step ahead. Starting from the last observed index $k ,$ define $y _ { j } ^ { \star } = y _ { j }$ for $j \le k$ and $y _ { j } ^ { \star } = \hat { y } _ { \mathcal { I } }$ <sub>j</sub> thereafter. At forecast horizon ℓ,

$$
\begin{array} { r } { \hat { y } _ { k + \ell } = \mathcal { G } ^ { \prime } \mathopen { } \mathclose \bgroup \left( \left\{ \left[ y _ { j } ^ { \star } , p _ { j } \right] \right\} _ { j = k + \ell - K - 1 } ^ { k + \ell - 1 } \aftergroup \egroup \right) , \qquad \ell = 1 , \ldots , M , } \end{array}\tag{5}
$$

and the encoder recomputes the latent state from the same window,

$$
z _ { k + \ell - 1 } = \mathcal { G } \Big ( \big \{ [ y _ { j } ^ { \star } , p _ { j } ] \big \} _ { j = k + \ell - K - 1 } ^ { k + \ell - 1 } \Big ) , \qquad \ell = 1 , \dots , M .\tag{6}
$$

Precipitation can be included in or omitted from each model independently. Both LSTMs re-encode their fixed-length windows from zero hidden and cell states at every step. The encoder–decoder first predicts the field at $k + \ell ;$ the predicted gauge depths and the supplied rainfall for that step are then appended to the window, so rainfall at $k + \ell$ first afects the field at $k + \ell + 1$

No new gauge observations are used during a multi-step forecast. This feedback scheme resembles SHRED (Williams et al., 2024), except that the latent state is recomputed from a sliding window at each step. The encoder provides an observation-driven state-estimation interface (P3) without maintaining an explicit posterior covariance.

Terrain-conditioned implicit decoder. The decoder $\mathcal { F }$ is a modulated coordinate-based implicit neural representation (INR), conditioned jointly on the latent state and on a static terrain feature vector:

$$
\tilde { h } ( t _ { k + \ell } , \xi ) = \mathcal { F } \big ( [ \gamma ( \xi ) , \phi ( \xi ) ] , z _ { k + \ell - 1 } \big ) , \qquad \ell = 1 , \dots , m ,\tag{7}
$$

where $\gamma ( \xi )$ is a coordinate feature mapping (Tancik et al., 2020), set to the identity here, and $\phi ( \xi ) \in \mathbb { R } ^ { 3 }$ collects the terrain features used by CLDNet (Si et al., 2026):

$$
{ \phi } ( \xi ) ~ = ~ \big ( b _ { z } ( \xi ) , ~ b _ { g } ( \xi ) , ~ \tilde { n } ( \xi ) \big ) ,\tag{8}
$$

with ground elevation $b _ { z } ,$ , slope magnitude $b _ { g } ,$ , and Manning coeficient n˜, each scaled to [−1, 1]. Because the case study uses two Manning classes, n˜ acts as a binary water/land indicator. The concatenation $[ \gamma ( \xi ) , \phi ( \xi ) ]$ is the decoder input, and a single afine layer maps the latent state to the shift modulations. Each FMMNN layer is defined as in Zhang et al. (2026):

$$
\sigma _ { i } ( \eta _ { i } ) \ = \ A _ { i } \sin \bigl ( W _ { i } \eta _ { i } + b _ { i } \bigr ) + c _ { i } + \varphi _ { i } ,\tag{9}
$$

where the frequencies $W _ { i }$ and phases $b _ { i }$ are frozen at random initialization, $A _ { i }$ is a trainable dense mixing matrix, $c _ { i }$ is a trainable bias, and $\varphi _ { i }$ is the latent-derived shift modulation (omitted in the final block). Tong and Chen (2026) report that the frozen random Fourier basis acts as an implicit regularizer and trains more stably than fully trainable SIREN networks (Sitzmann et al., 2020).

The conditioning in (7) mirrors that of CLDNet but is attached to an observation-driven rather than a rainfall-driven latent state. The coordinate decoder supplies fast spatial prediction (P2), with queries at selected grid points or other supplied coordinates.

Training objectives. For each training window from trajectory j, the encoder–decoder receives the observed steps $k - K , \ldots , k$ and predicts the field at $k + 1$ . At sampled coordinates $\left\{ \xi _ { q } \right\} _ { q = 1 } ^ { N _ { \xi } }$ , the encoder–decoder parameters θ minimize normalized-depth mean-squared error,

$$
L _ { \mathrm { p r e d } } ( \theta ) = \frac { 1 } { N _ { \xi } } \sum _ { q = 1 } ^ { N _ { \xi } } \left( \tilde { h } _ { \theta } ( t _ { k + 1 } , \xi _ { q } ) - h ( t _ { k + 1 } , \xi _ { q } ) \right) ^ { 2 } .\tag{10}
$$

SynxFlow depths provide the physical reference (P1) (Si et al., 2026). The gauge forecaster $\mathcal { G } ^ { \prime }$ has separate parameters $\bar { \theta ^ { \prime } }$ and minimizes the one-step mean-squared error of the gauge depths,

$$
L _ { \mathrm { o b s } } ( \theta ^ { \prime } ) = \frac { 1 } { N _ { s } N _ { \mathrm { w i n } } } \sum _ { j , k } \left. \hat { y } _ { k + 1 } ^ { ( j ) } - y _ { k + 1 } ^ { ( j ) } \right. _ { 2 } ^ { 2 } .\tag{11}
$$

Here, $N _ { \mathrm { w i n } }$ is the number of observation-model training windows. The two models are trained separately.

## 3.3. Training strategy and scalability

Because the decoder (7) is evaluated pointwise, training cost and memory scale with the number of query points in the loss rather than with the full grid size. As in CLDNet (Si et al., 2026), we exploit this by training on random subsets of cells.

Spatial subsampling. Training and evaluation use, for each event, a reduced grid that keeps only the cells reaching a depth of at least 0.1 m in that event’s reference simulation. The evaluation domain is therefore defined by each event’s reference depths. The four test grids contain 809,551–993,797 cells. Each training window uses $1 0 ^ { 5 }$ sampled points, with four windows per minibatch. The latent state is shared across all queries within a window.

Optimization. The encoder–decoder minimizes (10) using SOAP (Vyas et al., 2025) with initial learning rate $2 \times 1 0 ^ { - 3 }$ , momentum coeficients (0.95, 0.95), weight decay 0.01, epsilon $1 0 ^ { - 8 }$ , and preconditioner updates every ten optimizer steps. The learning rate decreases on training-loss plateaus, with patience of 4 epochs, a factor of 0.4, a relative threshold of 1%, and a minimum of 10−<sup>6</sup>.

The observation models use AdamW with learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 512, and gradient-norm clipping at 1.0. The vanilla and terrain-only configurations use a gauge-only forecaster, and the forcing-only and full configurations use a rainfall-conditioned forecaster. Table 3 summarizes the architecture and training settings.

Input and output normalization. Sensor observations, precipitation, terrain features, and target depths are scaled using training-set minima and maxima to [−1, 1]. Coordinates retain their stored scaling. The decoder predicts normalized depth, which is mapped back to meters before evaluation; its outputs are not constrained to the normalization interval.

<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Observation window K + 1</td><td>12 snapshots</td></tr><tr><td>Encoder G</td><td>Two-layer LSTM, 256 hidden units</td></tr><tr><td>Auxiliary model  $\mathcal { G } ^ { \prime }$ </td><td>Two-layer LSTM, 128 hidden units; linear 128 → 6 head</td></tr><tr><td>Decoder  $\mathcal { F }$ </td><td>8 FMMNN blocks, width 1024, rank 128</td></tr><tr><td>Decoder inputs</td><td>2 coordinates and 3 terrain features</td></tr><tr><td>Latent modulation</td><td>Affine 256 → 896 shift projection</td></tr><tr><td>Parameters (full model)</td><td>3.40 M total, 2.47 M trainable (excluding  $\mathcal { G } ^ { \prime } )$ </td></tr><tr><td>Query points per step  $N _ { \xi }$ </td><td> $\mathrm { 1 0 ^ { 5 } }$ </td></tr><tr><td>Minibatch</td><td>4 windows</td></tr><tr><td>Optimizer</td><td>SOAP, learning rate  $2 \times 1 0 ^ { - 3 }$  , weight decay 0.01</td></tr><tr><td>Scheduler</td><td>ReduceLROnPlateau, patience 4, factor 0.4</td></tr><tr><td>Epochs</td><td>180 completed</td></tr></table>

Table 3: Architecture and training settings of the full C-STRIDE encoder–decoder and the separately trained gauge forecasters. The four encoder–decoder configurations share the same architecture and optimization settings.

## 3.4. Digital-twin inference protocol and evaluation metrics

Digital-twin inference protocol. At run time, incoming gauge observations update a fixed-length history, which the encoder–decoder uses to predict the depth field one step ahead. Starting from observations through k, the model predicts depth at k + 1 and then generates later fields using predicted gauge values to advance the history window, following (5)–(7). At each step, the recurrent states are reset to zero, and the updated window is re-encoded into a latent state that conditions the coordinate decoder. Depth can then be predicted at grid points or gauge coordinates; accuracy, however, is established only on the evaluated reduced grids, and querying finer than 30 m adds no information beyond the training resolution.

Evaluation metrics. We assess performance along four complementary axes:

1. Field accuracy. For each evaluation trajectory, we compute the snapshot-wise relative $L _ { 2 }$ error $\varepsilon _ { h }$ and root-mean-squared error ${ \mathrm { R M S E } _ { h } }$ of water depth over the evaluation cells $\Omega _ { \mathrm { e v a l } }$ (the trajectoryspecific reduced grid described in Section 3.3) and average them over the evaluation snapshots. With zero-based indices for the retained sequence, $\mathcal { T } _ { \mathrm { e v a l } } = \{ K + 1 , \ldots , N _ { T } - 1 \}$ corresponds to original state indices $K + 2 , \ldots , N _ { T }$ in Section 2.3:

$$
\varepsilon _ { h } = \frac { 1 } { \left| { \mathcal T } _ { \mathrm { e v a l } } \right| } \sum _ { k \in { \mathcal T } _ { \mathrm { e v a l } } } \frac { \left\| \tilde { h } ( t _ { k } , \cdot ) - h ( t _ { k } , \cdot ) \right\| _ { \Omega _ { \mathrm { e v a l } } } } { \left\| \tilde { h } ( t _ { k } , \cdot ) \right\| _ { \Omega _ { \mathrm { e v a l } } } } , \qquad \mathrm { R M S E } _ { h } = \frac { 1 } { \left| { \mathcal T } _ { \mathrm { e v a l } } \right| } \sum _ { k \in { \mathcal T } _ { \mathrm { e v a l } } } \frac { \left\| \tilde { h } ( t _ { k } , \cdot ) - h ( t _ { k } , \cdot ) \right\| _ { \Omega _ { \mathrm { e v a l } } } } { \sqrt { \left| { \Omega _ { \mathrm { e v a l } } } \right| } } ,\tag{12}
$$

where $\begin{array} { r } { \| f \| _ { \Omega _ { \mathrm { e v a l } } } = \big ( \sum _ { \xi \in \Omega _ { \mathrm { e v a l } } } f ( \xi ) ^ { 2 } \big ) ^ { 1 / 2 } } \end{array}$ . Because $\varepsilon _ { h }$ is an $L _ { 2 }$ measure, it is weighted toward deep cells such as channels and reservoirs. Both metrics are computed for each snapshot and then averaged, rather than pooled over space and time.

2. Hydrograph metrics at USGS gauges. For the depth or WSE time series at each mapped gauge cell, we compute the Nash–Sutclife eficiency (Nash and Sutclife, 1970), $\begin{array} { r } { \mathrm { N S E } \ = \ 1 - \sum _ { k } ( \tilde { h } _ { k } \ - \ } \end{array}$ $\begin{array} { r } { h _ { k } ) ^ { 2 } / \sum _ { k } ( h _ { k } ~ { \bf \bar { \Phi } } - { \bf \Phi } \bar { h } ) ^ { 2 } ; } \end{array}$ ; the Kling–Gupta eficiency (2009 formulation) (Gupta et al., 2009), KGE = $1 - \sqrt { ( r - 1 ) ^ { 2 } + ( \alpha - 1 ) ^ { 2 } + ( \beta - 1 ) ^ { 2 } }$ , where r is the linear correlation, α the ratio of standard deviations, and β the ratio of means of the predicted and reference series; and the relative peak-depth error $\varepsilon _ { h _ { \mathrm { p e a k } } } = | \operatorname* { m a x } _ { k } \tilde { h } _ { k } - \operatorname* { m a x } _ { k } h _ { k } | / \operatorname* { m a x } _ { k } h _ { k }$

3. Flood-extent metrics. For a depth threshold τ, predicted and reference depths are converted to binary masks $\mathbf { 1 } [ h ( t , \xi ) \ \geq \ \tau ]$ . Summing true positives (TP), false positives (FP), and false negatives (FN) over all evaluation cells and snapshots, we report the critical success index $\mathrm { C S I } = \mathrm { T P } / ( \mathrm { T P } + \mathrm { F P } + $ FN) (Schaefer, 1990), precision $\mathrm { T P / ( T P + F P ) }$ , recall $\mathrm { T P } / ( \mathrm { T P } + \mathrm { F N } )$ , and frequency bias (TP + $\mathrm { F P ) / ( T P + F N ) }$ , which exceeds one when flooding is over-predicted. Following Si et al. (2026), we use $\tau = 0 . 5 \mathrm { m }$

4. Computational cost. We report the measured evaluation time with its computational scope.

For forecasting, we report $\varepsilon _ { h }$ at the forecast target times as a function of the horizon $m \in \{ 1 , 4 , 6 , 1 2 , 2 4 \}$

## 4. Results

This section evaluates the two capabilities that C-STRIDE contributes to the digital twin: fast spatial prediction (P2), assessed through field prediction and flood extent, and observation-driven state estimation (P3), assessed through gauge hydrographs and observed USGS inputs. Physical anchoring (P1) is inherited from the evaluation of SynxFlow against USGS observations by Si et al. (2026) and is not re-evaluated here. We also examine multi-step forecasting, sensitivity to the observation interface, and computational cost.

Compared configurations. The full model uses both terrain and precipitation inputs. We compare it with three reduced variants and two CLDNet-based references:

• Vanilla STRIDE: no terrain or precipitation input; the STRIDE-FMMNN configuration of Tong and Chen (2026), adapted to the observation model of Section 2.2.

• C-STRIDE (terrain): terrain input only.

• C-STRIDE (forcing): precipitation input only.

• C-STRIDE (full): terrain and precipitation inputs.

• CLDNet: the rainfall-driven CLDNet surrogate (Si et al., 2026), which uses terrain and rainfall without a gauge-assimilation step.

• CLDNet + LD-EnSF: CLDNet coupled with latent ensemble score filtering (Xiao et al., 2026), which assimilates the gauges without access to the true rainfall.

The four STRIDE configurations share the same architecture and training settings and difer only in these inputs.

<table><tr><td>Model</td><td>Gauge history</td><td>Past rainfall</td><td>Future rainfall</td><td>Terrain</td></tr><tr><td>Vanilla STRIDE</td><td>√</td><td>一</td><td>一</td><td>一</td></tr><tr><td>C-STRIDE (terrain)</td><td>√</td><td>一</td><td>一</td><td>√</td></tr><tr><td>C-STRIDE (forcing)</td><td>√</td><td>√</td><td>√</td><td>一</td></tr><tr><td>C-STRIDE (full)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>CLDNet</td><td>一</td><td>√</td><td>n/a</td><td>√</td></tr><tr><td>CLDNet + LD-EnSF</td><td>√</td><td></td><td>n/a</td><td>√</td></tr></table>

Table 4: Information available to each model at run time. Future rainfall is used for forecasting. $\mathrm { n / a } \mathrm { : }$ not evaluated.
<table><tr><td rowspan="2">Model</td><td colspan="2">April 2013 flood</td><td colspan="2">Four-event average</td></tr><tr><td>εh (%)</td><td>RMSEh (m)</td><td>εh (%)</td><td>RMSEh (m)</td></tr><tr><td>Vanilla STRIDE</td><td>13.69</td><td>0.0961</td><td>18.14</td><td>0.0995</td></tr><tr><td>C-STRIDE (terrain)</td><td>9.49</td><td>0.0667</td><td>14.23</td><td>0.0811</td></tr><tr><td>C-STRIDE (forcing)</td><td>13.59</td><td>0.0960</td><td>15.73</td><td>0.0879</td></tr><tr><td>C-STRIDE (full)</td><td>8.38</td><td>0.0589</td><td>10.63</td><td>0.0631</td></tr><tr><td>CLDNet</td><td>13.46</td><td>0.0930</td><td>16.13</td><td>0.0867</td></tr><tr><td>CLDNet + LD-EnSF</td><td>16.99</td><td>0.1188</td><td>20.88</td><td>0.1130</td></tr></table>

Table 5: Mean snapshot depth errors for the April 2013 flood and the four-event average.

## 4.1. Aggregate next-step prediction accuracy

Table 5 reports depth prediction errors for the April 2013 flood and the four-event average. Each estimate is one step ahead of a 12-step observation history. The relative error normalizes the depth error by the reference field, and the RMSE gives its magnitude in meters.

Both conditioning sources improve the aggregate estimates, with terrain alone giving the larger reduction. Full C-STRIDE combines their benefits, reducing the relative error from 18.14% to 10.63% (approximately 41%) and yielding the lowest RMSE. Adding rainfall to the terrain-conditioned model further reduces error from 14.23% to 10.63%. Full C-STRIDE also improves on the rainfall-driven CLDNet surrogate (16.13%, 0.0867 m), although it additionally receives gauge histories. Terrain-only C-STRIDE provides the more relevant comparison with CLDNet + LD-EnSF: its four-event relative error is 14.23% versus 20.88%, approximately 32% lower, with RMSE decreasing from 0.1130 to 0.0811 m. Terrain and rainfall therefore provide complementary information: terrain gives the larger individual gain, and rainfall further improves the terrain-conditioned estimate.

The April 2013 results show the same ordering among the STRIDE variants. Terrain conditioning accounts for most of the single-event gain, and the full model reduces relative error from 13.69% to 8.38% and RMSE from 0.0961 to 0.0589 m. Full C-STRIDE also has lower errors than CLDNet (13.46%, 0.0930 m), while terrain-only C-STRIDE improves on CLDNet + LD-EnSF (9.49% versus 16.99%).

Figure 2 shows the mean prediction error per time step and its variability across events. Full C-STRIDE has the lowest mean error over most of the event, although the curves cross near its end. The separation between configurations varies over time, and wider bands mark periods when prediction dificulty difers more between events.

Aggregate errors can obscure localized behavior. Figure 3 compares full C-STRIDE, terrain-only C-STRIDE, and vanilla STRIDE at eight locations chosen along northing rows, alternating between the largest peak depth, which tests the magnitude of inundation, and the largest temporal total variation, which tests the rise and recession.

Full C-STRIDE and vanilla STRIDE reproduce the broad rise, peak, and recession at the peak-depth locations, with most residual error concentrated around rapid transitions. The clearest diference occurs at location H, the most dynamically variable site. There, vanilla STRIDE underestimates the peak by about 1.31 m and misses the recession, whereas full C-STRIDE recovers the overall magnitude and shape but oscillates near the crest and overshoots it by about 0.27 m. Terrain-only C-STRIDE overshoots the same crest by 0.61 m. Location H lies about 25 km from the nearest gauge, which may leave its rapidly varying local dynamics weakly constrained by the sensor histories. The other high-variation locations show smaller improvements in rising-limb timing and recession.

![](images/bc7be44f8fcdbc122bebcad5cb13bc463e38a02fed51313306645e2362df806e.jpg)  
Figure 2: Efect of terrain and forcing conditioning on depth prediction. Each curve shows the mean snapshot-wise relative $L _ { 2 }$ error for one STRIDE configuration. Shading denotes one standard deviation across events.

Figure 4 shows the spatial prediction at peak total reference depth. The snapshot RMSE is 0.0684 m, the mean signed error is 0.0045 m, and the maximum absolute local error is 1.47 m. The full model reproduces the spatial pattern of inundation, and the small positive mean error indicates slight overall over-prediction; larger errors remain at individual locations.

## 4.2. Hydrograph fidelity at USGS gauges

We first compare predicted gauge hydrographs with SynxFlow (Table 6), which tests whether the predicted field is consistent with the input histories. We then compare predictions with USGS observations, first with simulated and then with observed gauge inputs (Figs. 5 and 6), and finally evaluate two tributary gauges withheld from the encoder inputs (Fig. 7).

For the USGS comparisons, gauge-specific ofsets are computed as the event-wide mean diference between observed and simulated WSE. The same ofsets are used to align the WSE series and to convert USGS observations to model-depth inputs. These comparisons therefore assess hydrograph timing and shape after retrospective mean alignment.

<table><tr><td></td><td colspan="2">NSE↑</td><td colspan="2">KGE↑</td><td colspan="2">(%) ↓  $\varepsilon _ { h _ { \mathrm { p e a k } } }$ </td></tr><tr><td>Model</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>Vanilla STRIDE</td><td>0.949</td><td>0.937</td><td>0.898</td><td>0.876</td><td>3.10</td><td>3.06</td></tr><tr><td>C-STRIDE (terrain)</td><td>0.976</td><td>0.984</td><td>0.936</td><td>0.952</td><td>2.75</td><td>1.66</td></tr><tr><td>C-STRIDE (forcing)</td><td>0.931</td><td>0.955</td><td>0.881</td><td>0.880</td><td>4.90</td><td>0.93</td></tr><tr><td>C-STRIDE (full)</td><td>0.951</td><td>0.960</td><td>0.915</td><td>0.934</td><td>6.33</td><td>3.29</td></tr></table>

Table 6: Gauge hydrograph fidelity relative to SynxFlow during the April 2013 flood. Mean and median are over six gauges, while bold values identify the best configuration.

Terrain-only C-STRIDE gives the best gauge fidelity. It improves NSE over vanilla STRIDE at all six sites and achieves the best mean NSE, KGE, and peak-depth error. Full C-STRIDE improves NSE at five of the six sites, but the gains at Gurnee and Des Plaines are negligible $\left( \le 0 . 0 0 2 \right)$ , its mean NSE is essentially unchanged (0.951 versus 0.949), and its mean peak-depth error is about twice that of vanilla STRIDE (6.33% versus 3.10%). The larger mean is mainly due to DuPage (22.04%) and Salt Creek (8.13%). By median, which is less sensitive to DuPage, full C-STRIDE improves NSE (0.960 versus 0.937) and KGE over vanilla STRIDE but still has a slightly higher peak error. One possible reason is that the 507 precipitation values outnumber the 6 gauge values in the encoder input and may weaken the direct gauge signal, particularly because rainfall afects gauge depth with a delay. Full C-STRIDE is thus best for field prediction, whereas terrain-only conditioning gives the best average fidelity at the gauges.

![](images/14d033883d9031f3700796fc388106f619ed612287de6fa4a96e28b618793d7b.jpg)  
Figure 3: April 2013 depth hydrographs at diagnostic locations A–H. Left panels show peak-depth locations (circles), and right panels show high-total-variation locations (squares), selected along alternating northing rows. Curves compare SynxFlow, full C-STRIDE, terrain-only C-STRIDE, and vanilla STRIDE. The center panel shows the DEM, diagnostic locations, and gauges 1–6 (red triangles).

Figure 5 separates surrogate error from simulator-to-observation discrepancy. Full C-STRIDE closely follows SynxFlow at all six sites, while agreement with USGS is strongest for the broad rise and crest at the three upstream mainstem gauges (1–3). The larger timing and recession discrepancies at Riverside (5) and at the tributary gauges (4, 6) are largely shared by C-STRIDE and SynxFlow, indicating that they originate mainly in the simulator rather than in surrogate replication. DuPage (6) is the clearest surrogate-specific exception, where C-STRIDE overshoots the simulated crest by about 0.19 m. At both tributary gauges, the SynxFlow WSE is flat at the start of the event while the observed stage is already rising, showing a discrepancy in the early response.

Replacing simulated histories with USGS observations improves NSE from 0.880 to 0.970 at Des Plaines, from 0.162 to 0.603 at Salt Creek, and from 0.118 to 0.840 at Riverside (Fig. 6 and Table 7). NSE decreases at Russell (0.780 → 0.731), Gurnee (0.944 → 0.932), and DuPage (0.588 → 0.567). At DuPage, the observedinput prediction peaks 11 hours early and overshoots the observed crest by approximately 0.31 m. At Salt Creek and Riverside, the observed histories shift the rise and recession toward the measurements, while at Des Plaines, they improve the late recession. The predictions thus respond to the observed inputs, but not

![](images/061d9337b67daf670bad6652221cdb9e8378e883f0aff18989b2b4da0b6da6af.jpg)

![](images/da704c4d2135bf208682dffd8d0fddabe5cfe0dd1a56cbc1ef17d68bd18dcd6e.jpg)

![](images/0128c7da5638b9c106317d93e0dc38fce75becf25e87afd5e995c8c9d652166a.jpg)  
Figure 4: Spatial prediction of the April 2013 flood at peak total reference depth. Panels show SynxFlow depth (left), full C-STRIDE depth (center), and prediction minus reference (right) over the evaluation domain.

equally well at every gauge.
<table><tr><td rowspan="2">Gauge</td><td colspan="2">SynxFlow</td><td colspan="2">C-STRIDE (sim. input)</td><td colspan="2">C-STRIDE (USGS input)</td></tr><tr><td>NSE</td><td> $\Delta t _ { \mathrm { p e a k } } ~ ( \mathrm { h } )$ </td><td>NSE</td><td> $\Delta t _ { \mathrm { p e a k } } ~ ( \mathrm { h } )$ </td><td>NSE</td><td> $\Delta t _ { \mathrm { p e a k } } ~ ( \mathrm { h } )$ </td></tr><tr><td>Russell (05527800)</td><td>0.883</td><td>+6</td><td>0.780</td><td>+7</td><td>0.731</td><td>+3</td></tr><tr><td>Gurnee (05528000)</td><td>0.893</td><td>-25</td><td>0.944</td><td>+2</td><td>0.932</td><td>+2</td></tr><tr><td>Des Plaines (05529000)</td><td>0.889</td><td>-10</td><td>0.880</td><td>-11</td><td>0.970</td><td>0</td></tr><tr><td>Salt Creek (05531500)</td><td>0.163</td><td>+4</td><td>0.162</td><td>+6</td><td>0.603</td><td>+4</td></tr><tr><td>Riverside (05532500)</td><td>0.182</td><td>+1</td><td>0.118</td><td>0</td><td>0.840</td><td>0</td></tr><tr><td>DuPage (05540130)</td><td>0.712</td><td>-1</td><td>0.588</td><td>-1</td><td>0.567</td><td>-11</td></tr><tr><td>Held out: Salt Creek</td><td>0.163</td><td>+4</td><td>0.004</td><td>+7</td><td>0.346</td><td>+7</td></tr><tr><td>Held out: DuPage</td><td>0.712</td><td>-1</td><td>0.388</td><td>+1</td><td>0.387</td><td>-7</td></tr></table>

Table 7: Agreement with USGS mean-aligned WSE. Positive peak-time errors indicate late predictions. The first maximum defines peak time. At Gurnee, the observed crest is nearly flat, so peak-time errors there are sensitive to small fluctuations. The last two rows use gauges withheld from encoder inputs.

Held-out tributary gauges. The four-mainstem-gauge model does not use Salt Creek or DuPage as inputs, although both locations remain in the simulated training fields. The response at these two held-out tributaries is mixed (Fig. 7). With USGS inputs, Salt Creek NSE improves from 0.004 to 0.346, but its peak remains 7 hours late. At DuPage, NSE is essentially unchanged (0.388 versus 0.387), and the peak shifts from 1 hour late to 7 hours early. Observed mainstem histories can therefore improve a tributary prediction without that tributary’s own record, but not reliably.

![](images/cb5bfd44c580e1917c06682673a6a0bc84b4a4b4511613cf9d46dbc46942d07b.jpg)  
Figure 5: Gauge-scale comparison during the April 2013 flood using simulated histories as C-STRIDE inputs. The left panel shows the reference depth and the six gauge locations. Hydrographs compare mean-aligned WSE from full C-STRIDE (blue dashed), SynxFlow (blue solid), and USGS (black solid).

## 4.3. Flood-extent metrics

Beyond pointwise depth accuracy, a flood digital twin must identify the spatial extent of decision-relevant inundation. Following Si et al. (2026), Table 8 reports CSI, precision, recall, and frequency bias at the τ = 0.5 m depth threshold for the 2013 event, aggregated over all valid snapshots and evaluation cells.

<table><tr><td>Model</td><td>CSI ↑</td><td>Precision ↑</td><td>Recall ↑</td><td>Bias (→ 1)</td></tr><tr><td>Vanilla STRIDE</td><td>0.888</td><td>0.947</td><td>0.935</td><td>0.987</td></tr><tr><td>C-STRIDE (terrain)</td><td>0.925</td><td>0.959</td><td>0.963</td><td>1.004</td></tr><tr><td>C-STRIDE (forcing)</td><td>0.894</td><td>0.951</td><td>0.938</td><td>0.986</td></tr><tr><td>C-STRIDE (full)</td><td>0.933</td><td>0.963</td><td>0.967</td><td>1.004</td></tr><tr><td>CLDNet</td><td>0.871</td><td>0.926</td><td>0.936</td><td>1.011</td></tr><tr><td>CLDNet + LD-EnSF</td><td>0.843</td><td>0.901</td><td>0.929</td><td>1.032</td></tr></table>

Table 8: Flood-extent metrics for the April 2013 flood at a 0.5 m depth threshold. Frequency bias above one indicates over prediction.

Full C-STRIDE has the highest CSI, precision, and recall, with CSI reaching 0.933. For this single event, terrain conditioning accounts for most of the improvement in flood extent, while precipitation alone has only a modest efect. Adding precipitation to the terrain-conditioned model yields the best overall inundation mask. Precision and recall show the same pattern. Vanilla STRIDE slightly under-predicts flooding (bias 0.987), and terrain conditioning mainly corrects these omissions without adding many false alarms. Full C-STRIDE is nearly unbiased, whereas both CLDNet references slightly over-predict. Because cells that stay wet throughout the event, such as river channels, count as hits in these pooled scores, the high absolute CSI values partly reflect easy cells (Stephens et al., 2014).

![](images/6ebb4591dd903094228a85ed6fcf0b512a45b5a6f7ae126126bd7853d407271f.jpg)  
Figure 6: Response of full C-STRIDE to replacing simulated gauge histories with USGS observations at inference time, without retraining. Dashed curves show predictions with simulated (blue) or USGS (black) inputs. Solid curves show SynxFlow (blue) and USGS (black). All series use the same retrospective mean alignment, allowing comparison of hydrograph timing and shape.

## 4.4. Multi-step forecast skill

Each multi-step forecast predicts 24 successive hourly depth fields without new gauge observations. We evaluate 40 forecasts, from 10 randomly chosen origins per event, using the same origins for all configurations. The rainfall-conditioned configurations receive the observed future rainfall. Because predicted gauge values replace the unavailable observations as the window advances, errors can accumulate in both the gauge forecaster and the field prediction, so the results test the complete forecasting procedure.

Full C-STRIDE yields the lowest mean error among the four unperturbed configurations for each horizon (Table 9). Terrain improves early forecasts, but its advantage over vanilla STRIDE has disappeared by 24 hours. Rainfall conditioning limits this error growth: forcing-only C-STRIDE reaches 18.28%, while the full model reaches 14.48%, about 62% below vanilla STRIDE. Full-field persistence has lower error at horizon one (2.73% versus 8.06%), while full C-STRIDE is better at the reported horizons from 4 onward and reaches 14.48% versus 36.37% at horizon 24. Persistence uses the complete previous reference field, unlike the sparse-input learned models.

The one-step errors in Tables 9 (8.06%) and 5 (10.63%) use diferent evaluation snapshots. The former averages over 40 randomly selected forecast origins (ten per event), restricted to time steps 12–72 to accommodate the 24-step forecast, whereas the latter averages over all 336 valid snapshots at time steps 12–95 across the four events. For the gauge forecasts alone, the error of the gauge-only forecaster rises from 1.27% at the first step to 42.63% at 24 h, compared with 9.30% and 16.48% for the rainfall-conditioned forecaster. Rainfall therefore helps long-horizon gauge forecasting but not the first step.

![](images/eedea2912f786dbcdd590a74819b55aeaec8791dc5207746b2dfdfc29b0ec989.jpg)

Figure 7: Gauge hydrographs from the four-mainstem-input model. Panels marked “Input” supply histories to the encoder, while Salt Creek and DuPage, marked “Held out”, do not supply inputs. Simulated- and USGS-input predictions are compared with SynxFlow and observations using the line styles in Fig. 6.
<table><tr><td>Model</td><td>m = 1</td><td>m = 4</td><td>m = 6</td><td> $m = 1 2$ </td><td>m = 24</td></tr><tr><td>Vanilla STRIDE</td><td> $1 7 . 2 6 \pm 8 . 7 2$ </td><td> $1 8 . 1 5 \pm 9 . 1 2$ </td><td> $2 1 . 0 5 \pm 1 1 . 3 5$ </td><td> $2 7 . 6 9 \pm 1 3 . 4 2$ </td><td> $3 8 . 1 4 \pm 1 2 . 6 3$ </td></tr><tr><td>C-STRIDE (terrain)</td><td> $1 1 . 7 2 \pm 9 . 7 5$ </td><td> $1 3 . 4 9 \pm 9 . 8 4$ </td><td> $1 7 . 4 2 \pm 1 2 . 3 0$ </td><td> $2 6 . 8 8 \pm 1 4 . 8 1$ </td><td> $3 8 . 7 4 \pm 1 3 . 6 8$ </td></tr><tr><td>C-STRIDE (forcing)</td><td> $1 4 . 7 0 \pm 4 . 2 9$ </td><td> $1 5 . 3 3 \pm 4 . 5 3$ </td><td> $1 5 . 7 6 \pm 4 . 5 2$ </td><td> $1 6 . 8 6 \pm 5 . 0 7$ </td><td> $1 8 . 2 8 \pm 4 . 2 6$ </td></tr><tr><td>C-STRIDE (full)</td><td> ${ \bf 8 . 0 6 \pm 5 . 0 9 }$ </td><td> ${ \bf 8 . 7 2 \pm 4 . 9 7 }$ </td><td> ${ \bf 9 . 3 8 \pm 4 . 7 5 }$ </td><td> ${ \bf 1 1 . 7 5 \pm 5 . 8 8 }$ </td><td> ${ \bf 1 4 . 4 8 \pm 4 . 6 7 }$ </td></tr><tr><td>Full, rainfall mismatch  $\alpha = 1$ </td><td> $8 . 0 6 \pm 5 . 0 9$ </td><td> $8 . 7 4 \pm 5 . 0 1$ </td><td> $9 . 4 4 \pm 4 . 8 5$ </td><td> $1 2 . 0 5 \pm 6 . 1 6$ </td><td> $1 4 . 7 2 \pm 4 . 6 3$ </td></tr><tr><td>Full-field persistence</td><td> $2 . 7 3 \pm 3 . 8 5$ </td><td> $9 . 5 6 \pm 1 0 . 4 3$ </td><td> $1 3 . 0 2 \pm 1 2 . 9 2$ </td><td> $2 1 . 7 4 \pm 1 6 . 2 1$ </td><td> $3 6 . 3 7 \pm 1 7 . 7 9$ </td></tr></table>

Table 9: Mean ± standard deviation of relative depth error (%) at forecast horizons $m \in \{ 1 , 4 , 6 , 1 2 , 2 4 \}$ across 40 forecasts. Persistence uses the complete preceding reference field.

Sensitivity to future precipitation. To test sensitivity to the reference-rainfall assumption, we perturb the normalized future precipitation $p _ { k + \ell } \in [ - 1 , 1 ] ^ { d _ { p } }$ at future step $\ell \in \{ 1 , \ldots , M \}$ as

$$
\begin{array} { r l } & { p _ { k + \ell } ^ { \mathrm { m i s } } = \mathrm { c l i p } _ { [ - 1 , 1 ] } \bigg ( p _ { k + \ell } + \alpha \frac { \ell } { M } \sigma _ { k + \ell } \epsilon _ { k + \ell } \bigg ) , } \\ & { \sigma _ { k + \ell } = \mathrm { s t d } [ p _ { k + \ell } ] , \qquad \epsilon _ { k + \ell } \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{13}
$$

The perturbation grows linearly with lead time and scales with the spatial variability of the true rainfall field. We use $\alpha = 1 ~ ( \mathrm { { ^ { 6 4 } N o i s y } ^ { 3 } } )$ and apply the same realization to all components that consume future precipitation. Observed histories are unchanged. Because rainfall is appended after each field prediction, the perturbation first afects the following predicted field.

Figure 8 compares the mean rainfall intensity with the mean absolute mismatch. Averaged over the 507 rainfall values (not area-weighted), the true intensity is 0.812 mm/h, the absolute mismatch is 0.277 mm/h, and the signed mismatch is +0.072 mm/h. The band shows how the size of the perturbation varies between forecast origins.

![](images/1a532c684a76b1a15d04caae458cb15d6b61deef44413043f9e91bef053da314.jpg)  
Figure 8: Mean rainfall intensity and mean absolute mismatch for $\alpha = 1$ over the forecast horizon. The solid curve shows true precipitation and the dashed curve shows the imposed mismatch, averaged equally over the 507 rainfall values and then over forecasts. Shading denotes one standard deviation of the mismatch across forecasts.

Rainfall mismatch increases the full-model horizon-24 error from 14.48% to 14.72%, approximately 0.24 percentage points (Fig. 9). Horizon one is unchanged because future rainfall enters after the first prediction. The model is therefore only weakly sensitive to this type of perturbation. The true- and noisy-rainfall curves stay close throughout, whereas the configurations without rainfall show much larger error growth; the tested perturbation matters far less than omitting rainfall altogether.

## 4.5. Sensitivity to the observation interface

We test two properties of the observation interface: how much temporal history is needed to identify the flood state, and how performance changes when models are trained and evaluated with missing or noisy inputs.

## 4.5.1. Window length

For window lengths $K + 1 = 1 , 4 , 6 , 1 2 , 2 4$ , mean relative errors on common prediction times are 18.72%, 15.15%, 14.69%, 11.44%, and 8.61%, respectively (Fig. 10). Corresponding RMSEs are 0.1134, 0.0929, 0.0901, 0.0701, and 0.0508 m. Both metrics decrease steadily with history length, and the 24-step history outperforms the 12-step default. Comparing at common prediction times ensures that the diferences reflect the available history rather than the part of the event evaluated. We retain the 12-step window as the default for consistency with the main experiments; the 24-step result indicates that longer histories could further improve accuracy, at the cost of a longer record before the first prediction.

![](images/ec60735c4f01595656300ab6325ed77ae5ad554fa15e381c461f4ce5cd0339e5.jpg)

Figure 9: Relative depth error over a 24-step forecast. “True” and “Noisy” denote full C-STRIDE with observed and perturbed future rainfall, respectively. Terrain-only and vanilla receive no rainfall. Persistence holds the complete preceding reference field fixed. Curves show means over the forecasts, with shading denoting one standard deviation. Forecasts can overlap in time, so this spread does not represent uncertainty across independent training runs.  
![](images/83993cbefc2105e1586019475e479c1b413b32c206c50c84c9384d11dbb4b556.jpg)  
Figure 10: Efect of observation-history length K + 1 on full C-STRIDE prediction. The models are compared at common prediction times, so each curve covers the same part of the flood evolution. Curves show mean snapshot relative L depth error, and shading denotes one standard deviation across events.

## 4.5.2. Sensor robustness

Table 10 compares sensor configurations, each with its own trained model. In the 50% dropout configuration, the four mainstem gauges are always available, while the two tributary records are removed together at selected times and filled with the nearest available observation in time. The four-gauge configuration omits the tributary gauges entirely. The noise configurations add zero-mean Gaussian noise to the normalized inputs during training and evaluation. At each time step, its standard deviation is the noise level times the standard deviation across the six gauge values for gauge inputs, and across the 507 rainfall values for rainfall inputs.

Using four mainstem gauges gives nearly the same error as the six-gauge model (10.67% versus 10.63%), while tributary dropout increases it to 11.15%. Error rises with noise level, reaching 12.43% at level 0.20, approximately 17% above the default. The near-identical four- and six-gauge averages contrast with the mixed results at the held-out tributaries: similar field-level accuracy can hide local diferences. Accuracy degrades gradually as the noise level increases.

## 4.6. Computational cost

Training the full encoder–decoder took approximately 26.2 h, a one-time cost. On one NVIDIA L40S GPU, next-step prediction of all fields of the April 2013 event takes 21.99 s, or approximately 0.262 s per field. Restricting the queries to the six gauge locations reduces this to 3.55 ms per snapshot. The 40 forecasts of 24 steps take 247.37 s in total, including forecast preparation and scoring.

<table><tr><td>Configuration</td><td>εh (%)</td><td>RMSEh (m)</td></tr><tr><td>Full, six gauges</td><td>10.63</td><td>0.0631</td></tr><tr><td>50% tributary dropout</td><td>11.15</td><td>0.0665</td></tr><tr><td>Four mainstem gauges</td><td>10.67</td><td>0.0636</td></tr><tr><td>Noise level 0.05</td><td>11.77</td><td>0.0700</td></tr><tr><td>Noise level 0.10</td><td>12.17</td><td>0.0730</td></tr><tr><td>Noise level 0.20</td><td>12.43</td><td>0.0741</td></tr></table>

Table 10: Mean relative depth error and RMSE for the sensor configurations.

For reference, the reported CLDNet surrogate evaluation time is 28.8 s per event (Si et al., 2026), while SynxFlow requires approximately 3,300 s to simulate a 96-h event. These measurements have diferent scopes: SynxFlow advances depth and momentum over the full computational grid, whereas the learned models predict depth at selected locations, and the C-STRIDE forecast timing covers multiple forecast origins with overlapping target times. Because SynxFlow was also run on an NVIDIA L40S GPU, next-step prediction of the April 2013 event (21.99 s) is approximately 150 times faster than the simulation, subject to these diferences in scope.

## 5. Discussion

## 5.1. What gauges, terrain, and precipitation each contribute

The model variants separate three information sources with distinct roles. Terrain conditioning gives the largest single gain, about half of the total reduction in four-event next-step error achieved by the full model (18.14% → 14.23%, compared with 18.14% → 10.63%), and most of the improvement in inundation extent for the 2013 event. Two explanations, not mutually exclusive, are plausible. Elevation, slope, and roughness carry physical information about where water collects and how fast it moves. They also give the decoder high-resolution positional information: with only two coordinates, a pointwise decoder must represent detail on a 5,075 × 1,661 raster, whereas terrain features vary on the 30 m grid itself, and the binary Manning feature marks channels and open water directly. Removing terrain features one at a time, and giving vanilla STRIDE a richer coordinate encoding, would distinguish these explanations. This distinction matters when assessing whether the learned terrain relationships transfer to other basins.

Precipitation alone gives a smaller next-step improvement than terrain, but its contribution becomes more pronounced at longer horizons. Added to terrain, it still reduces the four-event next-step error from 14.23% to 10.63%. At the next-step horizon, the gauges already record the hydraulic response to past rainfall, so rainfall adds information mainly in parts of the basin that the gauges do not see. For forecasting, the auxiliary model cannot anticipate new rainfall from gauge histories alone, and the configurations without precipitation accumulate substantially larger errors over 24 h. The cost of precipitation input appears at the gauges, where full C-STRIDE has larger peak errors than the terrain-only model, particularly at the tributaries. Because the 507 precipitation features outnumber the 6 gauge features, compressing rainfall before it enters the encoder, for example into sub-catchment averages, is a natural next step.

Gauge histories provide the state information. A single snapshot is substantially less informative than a sequence. The larger late-event errors with shorter windows suggest that longer histories help distinguish evolving flood states, consistent with the delay-observability motivation in Section 3.1. Removing the tributary gauges changes basin-wide error only slightly, which shows that aggregate metrics are not suficient for judging a sensor network; local objectives, such as hydrograph fidelity on the tributaries, should be evaluated separately.

## 5.2. From simulated to observed gauge inputs

C-STRIDE is trained only on simulated gauge histories, so USGS records test how it behaves when the inputs difer systematically from those seen in training. These diferences arise from simulator errors in timing and recession, visible where SynxFlow departs from the USGS hydrographs; from the conversion of gauge stage to model depth, which depends on the station datum and on how the DEM represents the channel bed; from the initial state of each simulation; and from measurement noise. The USGS-input experiment shows that the learned state responds to observed timing and magnitude at several gauges without retraining. It also shows that such inputs can produce artifacts, such as the spurious early rise at Russell and the crest overshoot at DuPage. At the two tributary gauges withheld from the inputs (Fig. 7), the response is mixed: observed mainstem histories raise NSE at Salt Creek from 0.004 to 0.346, although its peak remains 7 h late, but leave DuPage essentially unchanged. Fine-tuning on observed histories, training with perturbed or bias-shifted inputs, and estimating datum ofsets from pre-event baseflow are direct ways to narrow this gap.

## 5.3. Toward an operational flood digital twin

In the pillar framework of Section 1.1, C-STRIDE supplies fast spatial prediction (P2) and observationdriven state estimation (P3) and inherits physical anchoring (P1) from SynxFlow. Several elements of an operational digital twin remain to be added.

Operational forcing. Long-horizon skill depends on future rainfall. The perturbation experiment tests one specific error model. Operational precipitation forecasts also have storm-displacement, timing, and systematic intensity errors, as well as missed and false storms. Prospective evaluation with real-time gauge streams and archived precipitation forecasts or ensembles is the most important next test. Independent inundation observations, including remotely sensed flood extents, would complement the gauge comparisons and assess spatial accuracy beyond agreement with SynxFlow.

Uncertainty and data assimilation. C-STRIDE produces deterministic estimates. The spread across evaluation samples in the figures describes variability in performance, not predictive uncertainty for an individual forecast. The compact latent state could initialize a latent ensemble filter such as LD-EnSF (Xiao et al., 2026) or a latent ensemble-variational smoother such as LEVDA (Si and Chen, 2026), with the coordinate decoder acting as the observation operator at gauge locations, subject to a suitable measurement-error model. Combined with precipitation ensembles and estimates of surrogate error, this would support probabilistic forecasts of depth and extent. A complete operational twin would additionally require a decision interface and feedback to the physical system (National Academies of Sciences, Engineering, and Medicine, 2024).

Changing sensor networks. The current encoder assumes a fixed set of gauges. An encoder that represents sensor locations and availability explicitly, as in attention-based sparse-sensing models (Santos et al., 2023), could accommodate outages and new sensors without retraining. Future tests should distinguish isolated missing values, contiguous outages, and permanent sensor loss, using only information available at each forecast origin. Sensor-placement studies could then examine where additional measurements most improve local hydrograph fidelity or inundation extent.

Computational role. Because SynxFlow already runs faster than real time for single events, the surrogate’s advantage lies in workloads with many evaluations: ensembles, assimilation cycles, and scenario analysis. The measured inference costs support these repeated-use settings, but a break-even estimate would require matched computational scopes and an accounting of both training-data generation and model training (Section 4.6).

Transfer and broader outputs. Multi-basin training and basin-conditioned encoders could test whether terrain-aware decoding reduces the data needed to adapt to a new watershed. Such studies should distinguish transfer across terrain, storm regimes, and sensor layouts. Extending the predicted state to velocity and discharge, together with conservation-aware training, would support hazard measures beyond depth and inundation extent. Coupling the surrogate with upstream hydrologic predictions would allow evaluation under time-varying inflow conditions beyond the precipitation-driven experiments presented here.

## 5.4. Limitations

The evaluation covers one basin, four held-out test events, and a six-gauge network, together with a retrained four-gauge configuration. Each configuration was trained once, so small diferences between configurations may fall within run-to-run variation. The missing-observation experiment uses a separately trained model with 50% dropout at two tributary gauges and nearest-neighbor temporal filling. Such filling may use later observations, so it does not establish robustness to causal, real-time outages. The reducednetwork experiment requires retraining and does not demonstrate adaptation to arbitrary sensor layouts or permanent gauge loss during deployment. Field metrics are evaluated on event-specific masks derived from reference depths, not on the full computational domain. These masks select cells that reach 0.1 m in each event and would not be known in advance during deployment; accuracy outside them is not established.

The surrogate learns water depth from simulated targets and inherits the simulator’s limitations: the representation of terrain and channel geometry at 30 m, two-class Manning roughness, and the initial and boundary conditions of each event. It can also inherit errors in gauge-to-grid alignment. The USGS comparisons provide a complementary observational assessment, but the WSE comparisons aligned using event-wide mean ofsets do not establish absolute elevation accuracy or prospective deployment performance. Evaluation at gauges that also supply the input histories is an observation-consistency test; the held-out-gauge experiment covers only two tributary gauges in one event. Training on a physical simulator does not explicitly enforce mass conservation or hydraulic consistency in the learned predictions, and velocity and discharge are outside the present scope.

Forecasts remain conditional on precipitation information. The mismatch experiment tests a specific perturbation of the reference rainfall rather than the full range of errors in operational precipitation forecasts, such as storm displacement, timing errors, and persistent bias. The model also produces deterministic estimates without calibrated predictive uncertainty.

## 6. Conclusions

We presented Conditional STRIDE (C-STRIDE), an observation-driven AI digital twin for basin-wide flood prediction from sparse stream gauges. A recurrent encoder summarizes gauge and, optionally, precipitation histories into a latent basin state, and a terrain-conditioned coordinate decoder predicts water depth one step beyond the observation window. A separately trained gauge forecaster supports multi-step prediction when new observations are unavailable. Trained on SynxFlow simulations of the Des Plaines River basin, the model combines rapid spatial prediction with observation-driven state estimation, while inheriting its physical reference from the simulator.

The experiments support four conclusions.

1. Terrain and precipitation provide complementary information for next-step prediction. Terrain gives the larger individual gain, while full conditioning reduces four-event mean relative depth error from 18.14% for vanilla STRIDE to 10.63%. For the April 2013 flood, full C-STRIDE achieves CSI 0.933 at the 0.5 m depth threshold on the assessed domain.

2. Precipitation limits error growth during multi-step prediction. With reference future rainfall, full C-STRIDE reaches 14.48% mean relative error at 24 h, compared with 38.14% for vanilla STRIDE. The tested rainfall mismatch raises the full-model error to 14.72%. Full-field persistence is more accurate at the first step, whereas C-STRIDE is more accurate at the reported horizons from four steps onward.

3. Field accuracy and gauge fidelity are distinct objectives. Full conditioning gives the lowest aggregate field error, whereas terrain-only conditioning gives the best average hydrograph agreement with SynxFlow at the six gauges. Similar field errors for four- and six-gauge configurations can therefore conceal local diferences.

4. Predictions respond to observed histories without retraining, but the gains are site dependent. Replacing simulated histories with USGS observations improves NSE at three of six gauges, including an increase from 0.118 to 0.840 at Riverside. The mixed response at the two tributary gauges withheld from encoder inputs further limits claims of general observational correction. These comparisons use gauge-specific, event-wide mean ofsets and assess the retrospective response to observed inputs.

Full C-STRIDE predicts the April 2013 depth fields in 21.99 s on one NVIDIA L40S GPU, about 150 times faster than SynxFlow on the same hardware. The results are limited to one basin, event-specific reference-derived evaluation grids, and deterministic predictions. Priorities for further development are prospective tests with operational precipitation forecasts and causal gauge preprocessing, independent inundation observations, calibrated uncertainty, and evaluation across basins and sensor networks.

## Data and code availability

Code, input data, and trained models for C-STRIDE will be available at https://github.com/ Flood-Digital-Twins/CSTRIDE, organized as in the CLDNet repository for the same Des Plaines River basin dataset (https://github.com/Flood-Digital-Twins/CLDNet; Si et al., 2026). The simulated flow fields will not be hosted but can be regenerated from the released inputs with the open-source SynxFlow solver (SynxFlow Developers, 2023). USGS gauge records were obtained from the National Water Information System (U.S. Geological Survey, 2016).

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

We thank Dr. Omar Sallam (Argonne National Laboratory) for calibrating the SynxFlow model of the Des Plaines River basin, and Dr. Eugene Yan (Argonne National Laboratory), Prof. Barnali Dixon (University of South Florida), and Prof. Subhrajit Guhathakurta (Georgia Institute of Technology) for helpful discussions. This work was supported by the National Science Foundation under Awards CNS-2325631 and DMS-2245111, by the U.S. Department of Energy under Contract Nos. DE-AC05-00OR22725 and DE-AC02-06CH11357, and by the 2025 IDEaS + Cloud Hub with support from Microsoft at the Georgia Institute of Technology.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the authors used Claude and ChatGPT to assist with manuscript editing, table revision, and language polishing. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## References

Bartos, M., Kerkez, B., 2021. Pipedream: An interactive digital twin model for natural and urban drainage systems. Environmental Modelling & Software 144, 105120. doi:10.1016/j.envsoft.2021.105120.

Bentivoglio, R., Isufi, E., Jonkman, S.N., Taormina, R., 2022. Deep learning methods for flood mapping: a review of existing applications and future research directions. Hydrology and Earth System Sciences 26, 4345–4378. doi:10.5194/hess-26-4345-2022.

Bentivoglio, R., Isufi, E., Jonkman, S.N., Taormina, R., 2023. Rapid spatio-temporal flood modelling via hydraulics-based graph neural networks. Hydrology and Earth System Sciences 27, 4227–4246. doi:10. 5194/hess-27-4227-2023.

Botvinick-Greenhouse, J., Oprea, M., Maulik, R., Yang, Y., 2025. Measure-theoretic time-delay embedding. Journal of Statistical Physics 192, 171. doi:10.1007/s10955-025-03555-1.

Brocca, L., Barbetta, S., Camici, S., Ciabatta, L., Dari, J., Filippucci, P., Massari, C., Modanesi, S., Tarpanelli, A., Bonaccorsi, B., Mosafa, H., Wagner, W., Vreugdenhil, M., Quast, R., Alfieri, L., Gabellani, S., Avanzi, F., Rains, D., Miralles, D.G., Mantovani, S., Briese, C., Domeneghetti, A., Jacob, A., Castelli, M., Camps-Valls, G., Volden, E., Fernandez, D., 2024. A Digital Twin of the terrestrial water cycle: a glimpse into the future through high-resolution Earth observations. Frontiers in Science 1, 1190191. doi:10.3389/fsci.2023.1190191.

Dewitz, J., 2023. National Land Cover Database (NLCD) 2021 Products. U.S. Geological Survey data release. doi:10.5066/P9JZ7AO3

Erichson, N.B., Mathelin, L., Yao, Z., Brunton, S.L., Mahoney, M.W., Kutz, J.N., 2020. Shallow neural networks for fluid flow reconstruction with limited sensors. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 476, 20200097. doi:10.1098/rspa.2020.0097.

Fukami, K., Maulik, R., Ramachandra, N., Fukagata, K., Taira, K., 2021. Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning. Nature Machine Intelligence 3, 945–951. doi:10.1038/s42256-021-00402-2.

García-Pintado, J., Mason, D.C., Dance, S.L., Cloke, H.L., Neal, J.C., Freer, J., Bates, P.D., 2015. Satellitesupported flood forecasting in river networks: A real case study. Journal of Hydrology 523, 706–724. doi:10.1016/j.jhydrol.2015.01.084.

Guo, Z., Leitão, J.P., Simões, N.E., Moosavi, V., 2021. Data-driven flood emulation: Speeding up urban flood predictions by deep convolutional neural networks. Journal of Flood Risk Management 14, e12684. doi:10.1111/jfr3.12684.

Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F., 2009. Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. Journal of Hydrology 377, 80–91. doi:10.1016/j.jhydrol.2009.08.003.

Hochreiter, S., Schmidhuber, J., 1997. Long Short-Term Memory. Neural Computation 9, 1735–1780. doi:10.1162/neco.1997.9.8.1735.

Hostache, R., Chini, M., Giustarini, L., Neal, J., Kavetski, D., Wood, M., Corato, G., Pelich, R.M., Matgen, P., 2018. Near-real-time assimilation of SAR-derived flood maps for improving flood forecasts. Water Resources Research 54, 5516–5535. doi:10.1029/2017wr022205.

Kabir, S., Patidar, S., Xia, X., Liang, Q., Neal, J., Pender, G., 2020. A deep convolutional neural network model for rapid prediction of fluvial flood inundation. Journal of Hydrology 590, 125481. doi:10.1016/ j.jhydrol.2020.125481.

Kratzert, F., Klotz, D., Brenner, C., Schulz, K., Herrnegger, M., 2018. Rainfall–runof modelling using Long Short-Term Memory (LSTM) networks. Hydrology and Earth System Sciences 22, 6005–6022. doi:10.5194/hess-22-6005-2018.

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A.K., Hochreiter, S., Nearing, G.S., 2019. Toward improved predictions in ungauged basins: Exploiting the power of machine learning. Water Resources Research 55, 11344–11354. doi:10.1029/2019wr026065.

Lin, Y., Mitchell, K.E., 2005. The NCEP Stage II/IV hourly precipitation analyses: Development and applications, in: Preprints, 19th Conference on Hydrology, American Meteorological Society, San Diego, CA. Paper 1.2.

Löwe, R., Böhm, J., Jensen, D.G., Leandro, J., Rasmussen, S.H., 2021. U-FLOOD – Topographic deep learning for predicting urban pluvial flood water depth. Journal of Hydrology 603, 126898. doi:10.1016/ j.jhydrol.2021.126898.

Madsen, H., Skotner, C., 2005. Adaptive state updating in real-time river flow forecasting—a combined filtering and error forecasting procedure. Journal of Hydrology 308, 302–312. doi:10.1016/j.jhydrol. 2004.10.030.

Mañé, R., 1981. On the dimension of the compact invariant sets of certain nonlinear maps, in: Rand, D.A., Young, L.S. (Eds.), Dynamical Systems and Turbulence, Warwick 1980. Springer. volume 898 of Lecture Notes in Mathematics, pp. 230–242. doi:10.1007/bfb0091916.

Nash, J.E., Sutclife, J.V., 1970. River flow forecasting through conceptual models part I—A discussion of principles. Journal of Hydrology 10, 282–290. doi:10.1016/0022-1694(70)90255-6.

National Academies of Sciences, Engineering, and Medicine, 2024. Foundational Research Gaps and Future Directions for Digital Twins. The National Academies Press, Washington, DC. doi:10.17226/26894.

Neal, J.C., Atkinson, P.M., Hutton, C.W., 2007. Flood inundation model updating using an ensemble Kalman filter and spatially distributed measurements. Journal of Hydrology 336, 401–415. doi:10.1016/ j.jhydrol.2007.01.012.

Nearing, G., Cohen, D., Dube, V., Gauch, M., Gilon, O., Harrigan, S., Hassidim, A., Klotz, D., Kratzert, F., Metzger, A., et al., 2024. Global prediction of extreme floods in ungauged watersheds. Nature 627, 559–563. doi:10.1038/s41586-024-07145-1.

Nevo, S., Morin, E., Gerzi Rosenthal, A., Metzger, A., Barshai, C., Weitzner, D., Voloshin, D., Kratzert, F., Elidan, G., Dror, G., et al., 2022. Flood forecasting with machine learning models in an operational framework. Hydrology and Earth System Sciences 26, 4013–4032. doi:10.5194/hess-26-4013-2022.

Rápalo, L.M.C., Gomes, Jr., M.N., Mendiondo, E.M., 2024. Developing an open-source flood forecasting system adapted to data-scarce regions: A digital twin coupled with hydrologic-hydrodynamic simulations. Journal of Hydrology 644, 131929. doi:10.1016/j.jhydrol.2024.131929.

Rentschler, J., Salhab, M., Jafino, B.A., 2022. Flood exposure and poverty in 188 countries. Nature Communications 13, 3527. doi:10.1038/s41467-022-30727-4.

Santos, J.E., Fox, Z.R., Mohan, A., O’Malley, D., Viswanathan, H., Lubbers, N., 2023. Development of the Senseiver for eficient field reconstruction from sparse observations. Nature Machine Intelligence 5, 1317–1325. doi:10.1038/s42256-023-00746-x.

Schaefer, J.T., 1990. The critical success index as an indicator of warning skill. Weather and Forecasting 5, 570–575. doi:10.1175/1520-0434(1990)005<0570:tcsiaa>2.0.co;2.

Serrano, L., Le Boudec, L., Kassaï Koupaï, A., Wang, T.X., Yin, Y., Vittaut, J.N., Gallinari, P., 2023. Operator learning with neural fields: Tackling PDEs on general geometries, in: Advances in Neural Information Processing Systems, pp. 70581–70611. doi:10.52202/075280-3093.

Si, P., Chen, P., 2026. LEVDA: Latent ensemble variational data assimilation via diferentiable dynamics, in: Advances in Neural Information Processing Systems. arXiv:2602.19406.

Si, P., Qiu, Y., Sallam, O., Feinstein, J., He, Z., Yan, E., Chen, P., 2026. Toward AI-driven digital twins for metropolitan floods: A conditional latent dynamics network surrogate of the shallow water equations. Journal of Hydrology , 136461doi:10.1016/j.jhydrol.2026.136461.

Sitzmann, V., Martel, J., Bergman, A., Lindell, D., Wetzstein, G., 2020. Implicit neural representations with periodic activation functions, in: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (Eds.), Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 7462–7473.

Stark, J., 1999. Delay embeddings for forced systems. I. Deterministic forcing. Journal of Nonlinear Science 9, 255–332. doi:10.1007/s003329900072.

Stephens, E., Schumann, G., Bates, P., 2014. Problems with binary pattern measures for flood model evaluation. Hydrological Processes 28, 4928–4937. doi:10.1002/hyp.9979.

SynxFlow Developers, 2023. SynxFlow: Simulates flood, landslide and debris flow dynamically using GPUs. https://github.com/SynxFlow/SynxFlow.

Takens, F., 1981. Detecting strange attractors in turbulence, in: Rand, D.A., Young, L.S. (Eds.), Dynamical Systems and Turbulence, Warwick 1980. Springer. volume 898 of Lecture Notes in Mathematics, pp. 366– 381. doi:10.1007/bfb0091924.

Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J.T., Ng, R., 2020. Fourier features let networks learn high frequency functions in low dimensional domains, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 7537–7547.

Tellman, B., Sullivan, J.A., Kuhn, C., Kettner, A.J., Doyle, C.S., Brakenridge, G.R., Erickson, T.A., Slayback, D.A., 2021. Satellite imaging reveals increased proportion of population exposed to floods. Nature 596, 80–86. doi:10.1038/s41586-021-03695-w.

Tong, Y., Chen, P., 2026. From sparse sensors to continuous fields: STRIDE for spatiotemporal reconstruction. arXiv preprint arXiv:2602.04201 .

U.S. Geological Survey, 2016. National Water Information System data available on the World Wide Web (USGS Water Data for the Nation). https://waterdata.usgs.gov/nwis. doi:10.5066/F7P55KJN.

Vyas, N., Morwani, D., Zhao, R., Shapira, I., Brandfonbrener, D., Janson, L., Kakade, S., 2025. SOAP: Improving and stabilizing Shampoo using Adam for language modeling, in: Yue, Y., Garg, A., Peng, N., Sha, F., Yu, R. (Eds.), International Conference on Learning Representations, pp. 93423–93444.

Williams, J.P., Zahn, O., Kutz, J.N., 2024. Sensing with shallow recurrent decoder networks. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 480, 20240054. doi:10.1098/ rspa.2024.0054.

Xia, X., Liang, Q., 2018. A new eficient implicit scheme for discretising the stif friction terms in the shallow water equations. Advances in Water Resources 117, 87–97. doi:10.1016/j.advwatres.2018.05.004.

Xia, X., Liang, Q., Ming, X., 2019. A full-scale fluvial flood modelling framework based on a highperformance integrated hydrodynamic modelling system (HiPIMS). Advances in Water Resources 132, 103392. doi:10.1016/j.advwatres.2019.103392.

Xia, X., Liang, Q., Ming, X., Hou, J., 2017. An eficient and stable hydrodynamic model with novel source term discretization schemes for overland flow and flood simulations. Water Resources Research 53, 3730–3759. doi:10.1002/2016wr020055.

Xiao, P., Si, P., Chen, P., 2026. LD-EnSF: Synergizing latent dynamics with ensemble score filters for fast data assimilation with sparse observations, in: International Conference on Learning Representations.

Yin, Y., Kirchmeyer, M., Franceschi, J.Y., Rakotomamonjy, A., Gallinari, P., 2023. Continuous PDE dynamics forecasting with implicit neural representations, in: International Conference on Learning Representations.

Zhang, S., Zhao, H., Zhong, Y., Zhou, H., 2026. Fourier multi-component and multi-layer neural networks: Unlocking high-frequency potential. Neural Networks 204, 109268. doi:10.1016/j.neunet.2026.109268.