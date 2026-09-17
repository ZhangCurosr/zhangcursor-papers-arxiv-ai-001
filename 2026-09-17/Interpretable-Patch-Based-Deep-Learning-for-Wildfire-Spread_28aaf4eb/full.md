# Interpretable Patch-Based Deep Learning for Wildfire Spread Prediction from Ensemble Simulations

Marcin Lawenda<sup>1</sup>, Aleksandra Krasicka<sup>1</sup>, David Caballero<sup>2</sup>, Luis Torres<sup>2</sup>, and Łukasz Szustak<sup>3</sup>

<sup>1</sup> Poznan Supercomputing and Networking Center, Poznań, Poland

lawenda@man.poznan.pl, akrasicka@man.poznan.pl 2 MeteoGrid, Madrid, Spain

david@meteogrid.com, luis@meteogrid.com

3 Technical University of Czestochowa, Częstochowa, Poland lukasz.szustak@pcz.pl

Abstract. Wildfire spread is traditionally predicted using physics-based simulators, which are physically interpretable but whose cost increases with each additional ensemble member. We ask how well deep learning surrogates can reproduce these simulations at a fraction of this cost, training them on 10,584 fire spread simulations at 2 m resolution for the Rectoret region in Catalonia, Spain. Four architectures are compared: a patch-based U-Net, a transfer-learned ResNet-50, a physics-informed network constrained by the wind-driven advection equation and a Swin-Unet transformer. Among the terrain and vegetation variables, only surface fuel load predicts burn probability with any strength (r = 0.27) and including it lowers prediction error by 21%. The remaining variables correlate weakly and are highly duplicative. Next, an experiment with saliency, occlusion and rotation demonstrates the models’ learning. Convolutional models rely primarily on distance from the current fire front, while Swin-Unet assigns more weight to fuel and terrain, a finding also noted in an unrelated wildfire dataset. When applied without retraining to the second region, Pedriza, all three convolutional models still predict fire spread, losing accuracy by a small but systematic margin.

Keywords: wildfire spread prediction · deep learning · physics-informed neural networks · U-Net · burn probability · high-performance data analytics

## 1 Introduction

Wildfires are among the most devastating natural hazards afecting Mediterranean landscapes, and their increasing frequency and severity are linked to climate change and land-use changes in the wildland-urban interface. Accurate, spatially precise fire spread prediction influences evacuation planning, firefighting resource allocation and fuel-break design. Traditional approaches rely on physically based or semi-empirical simulators, particularly cellular automata (CA) models based on Rothermel’s fire-spread equations [23] (e.g. FARSITE [7], FlamMap [8]). These simulators are physically interpretable and well-validated. However, their cost increases directly with the number of simulations. A burn probability (BP) map is generated from multiple simulations, each representing a separate, complete run with a new combination of wind and ignition point.

Producing a reliable BP map therefore multiplies the cost of a single run many times over.

This cost motivates machine learning surrogates: once trained, the surrogate reproduces the simulator’s output in a single forward pass, making inference virtually free. Whether it also performs outside the terrain and fuel on which it was trained is a separate and much more dificult question, which we treat here only as a preliminary comparison (Section 8). Our dataset consists of a set of fire spread simulations for the Rectoret region in Catalonia, recording the minute the fire reaches each cell under varying wind and ignition conditions, along with the terrain and fuel descriptors that shape it.

Our contribution has three parts, moving from what the data itself shows, to how closely the surrogates reproduce it, to what they have learned in doing so. First, we characterise the statistical relationship between terrain/fuel features and burn probability using correlation, Moran’s I spatial autocorrelation and XGBoost feature importance. Second, we design and compare four deep learningbased models to predict fire spread 30 minutes in advance on 128 × 128 patches: a custom U-Net (108-configuration architecture/loss search), a transfer-learned ResNet-50, a physics-informed variant penalising violations of a wind-driven advection equation and a Swin-Unet transformer baseline. Third, we apply complementary interpretability techniques (saliency, occlusion, local perturbations) to characterise what each model has learned and compare performance on a second region (Pedriza).

## 2 Related Work

The work related to this paper spans two traditions: physics-based fire propagation simulation, which provides the ground truth used here, and trained surrogates, which approximate it at a fraction of the inference cost. We discuss four streams in turn: physics-based simulators, which solve the fire-spread equations, CNN/U-Net surrogates, physics-informed networks, which instead learn from data under a penalty for violating them and transformer-based architectures. Each of these corresponds to one of the models compared in Section 5. We conclude by contrasting our own contributions with them.

Physics-based simulators. Operational fire propagation prediction has historically relied on semi-empirical models based on Rothermel’s equations [23], which underpin simulators such as FARSITE [7] and FlamMap [8], which propagate the fire front by expanding the Huygens wavelet. CA variants discretise the landscape into a grid and propagate the ignition state between neighbouring cells, ofering a more parallelisable alternative. Recent, diferentiable, GPUaccelerated CA simulators achieve millisecond-per-step computations and can still be calibrated using gradient descent. They also demonstrate better interregion transfer than purely supervised surrogates [27]. This is a useful warning for Section 8. The burn probability (BP) formula used here is the fraction of the ensemble in which a cell burns. This is a standard CA simulator output, widely used for fuel management and risk zoning.

CNN/U-Net surrogates. The Next Day Wildfire Spread dataset and its CNN baseline [12] established the standard approach to image-to-image segmentation based on stacked terrain, weather and fire status channels, which was later refined in two-branch CNNs by processing fuel and weather separately [10]. Closer to our setup, ConvLSTM models were trained directly on simulated wildfire data from the mathematical analog model [3], setting a precedent for surrogate evaluation based on simulation results rather than observed fires.

Transfer learning from backbone networks pretrained on ImageNet [11] forms the basis for our ResNet-50 baseline model, and patch-based training handles 1500 × 1500-pixel rasters, from which we assemble 11-channel input tensors.

Physics-informed neural networks. PINNs add diferential equations to the training loss, guiding the model toward physically consistent predictions, even with limited labelled data. Wind-driven fire front propagation naturally fits this model, as an advection equation for the fire state field. The closest approximation to our model is [26], which adds mass and energy conservation constraints to learn forest fire spread parameters from both simulated and real thermal data, and shows that training with physical constraints allows the recovery of significant parameters even from noisy observations. This motivates the advection penalty in Section 5.

Transformer-based architectures. Vision-transformer architectures have recently been applied to forest fire prediction as an alternative to purely convolutional designs, including Swin-based encoder-decoder variants for next-day spread, whose accuracy depends significantly on weights pre-trained on ImageNet [15]. The most relevant element of our Swin-Unet comparison is [29], which compares Autoencoder, ResNet, U-Net and Swin-Unet using a decade of California remote sensing data, with Swin-Unet and U-Net outperforming the other two. Grad-CAM analysis also showed that Swin-Unet assigns a higher weight to vegetation and drought than U-Net. We observe the same diference in Section 7.

This paper is unique in combining these threads. A data-driven CNN, a transfer-based framework, a physics-informed variant and a transformer are compared within a single dataset and a single patching scheme. Interpretability is examined using three separate methods and a direct comparison is performed across two regions. Most of the papers cited above examine a single model family or a single region, but rarely both simultaneously.

## 3 Study Area and Data

Simulation ensemble. The dataset comprises $\begin{array} { l l l } { N S } & { = } & { 1 0 { , } 5 8 4 } \end{array}$ stochastic fire-spread simulations for the Rectoret region, Catalonia, Spain (ETRS89 UTM31N), at 2 m resolution, bounded by 41°24’48.09”N, 2°4’44.72”E (lower left) and 41°26’26.69”N, 2°6’52.50”E (upper right). Each simulation is a TA<Number>.asc raster encoding the simulated minute of fire arrival at each cell (capped at 180 minutes), parameterised by wind speed (WS), wind direction (WD) and ignition ofset (OX, OY). A companion Burn Probability raster summarises the ensemble as

$$
B P = 1 0 0 \cdot { \frac { N F } { N S } } ,\tag{1}
$$

where NF is the number of simulations in which fire passed through a given cell.

The four generating parameters were sampled on a regular full-factorial grid (Table 1): three wind speeds, eight wind directions at $4 5 ^ { \circ }$ increments and a $2 1 \times 2 1$ grid of ignition points spaced 100 m apart across 500–2500 m in each direction. That gives $3 \times 8 = 2 4$ distinct wind configurations, each realised at all $2 1 ^ { 2 } = 4 4 1$ ignition points, for $2 4 \times 4 4 1 = 1 0 { , } 5 8 4$ simulations in total. The design is therefore exactly balanced, since every wind configuration is represented by the same ignition points and every ignition point by the same winds.

Terrain and fuel covariates. The terrain and fuel descriptors come from the Institut Cartogràfic de Catalunya in ArcGIS ASCII format. They comprise slope and aspect, a continuous fuel-load map (cmb01) with its binary counterpart (cmb), five vegetation-continuity indices computed at several scales from the WUIX index (bio3x3n, bio5x5n, cont6m, cont10m, CONT\_NORM\_2\_20) and finally surface fuel load and crown fuel. Surface fuel load later proves to be the single most informative covariate for BP prediction (Section 4). It is obtained by translating each BEHAVE-Anderson fuel class [2] into a total fuel load in $\mathrm { k g } \mathrm { m } ^ { - 2 }$ , which for the 2 m grid corresponds to four times the per-square-metre value. In the deep-learning pipeline this raster (fuel\_load) is one of the eleven input channels, min–max normalised to [0, 1] along with the other static layers. Crown fuel is not used as a model input (Section 5). The fuel-load raster is derived from the same fuel-model classification the simulator uses to propagate fire, so it is not an independent predictor.

Table 1. Sampling of the four ensemble-generating parameters. The design is a full factorial grid, giving $3 \times 8 \times 2 1 \times 2 1 = 1 0 { , } 5 8 4$ simulations.
<table><tr><td>Parameter</td><td></td></tr><tr><td>Wind speed (WS)  $5 , 1 0 , 2 0 \ \mathrm { k m h ^ { - 1 } }$ </td><td>3</td></tr><tr><td>Wind direction (WD)  $0 ^ { \circ } , 4 5 ^ { \circ } , \ldots , 3 1 5 ^ { \circ } \ : ( 4 5 ^ { \circ } \ : \mathrm { s t e p s ) }$ </td><td>8</td></tr><tr><td>Ignition offset OX 500, 600, . . . , 2500 m (100 m steps)</td><td>21</td></tr><tr><td>Ignition offset OY 500, 600, . . . , 2500 m (100 m steps)</td><td>21</td></tr><tr><td colspan="2">Total ensemble size NS</td></tr></table>

Second region: Pedriza. For a cross-region comparison, a second region was used: La Pedriza de Manzanares, Community of Madrid, Spain (ETRS89 UTM30N, EPSG:25830), bounded by 40°42’21.13”N, 3°57’27.98”W (lower left) and 40°48’6.63”N, 3°48’54.53”W (upper right). It is centred on a granite massif in the southern Sierra de Guadarrama rising from about 890 m to 2,029 m, with Mediterranean vegetation of pine, holm oak, juniper and scrubland. The raster is 1217 × 1054 cells at 10 m resolution, so the domain spans 12.17 km east–west by 10.54 km north–south, about 128 km<sup>2</sup>, roughly fourteen times the area of the Rectoret domain. Pedriza therefore shares the broad fuel and terrain regime of Rectoret while difering in elevation, geology and in covering a much larger and more varied area. One diference matters for everything that follows. Rectoret is mapped at 2 m and Pedriza at 10 m, so a 128 × 128 patch covers 256 m in Rectoret but 1280 m in Pedriza, twenty-five times the ground area. The region was used in two ways: descriptively, aggregated to a 100 m/200 m mesh with per-cell mean, standard deviation, minimum and maximum for slope, aspect, bio5x5n, cont10m, CONT\_NORM\_2\_20, fuel load and BP, and as a model-transfer test bed at native patch resolution.

## 4 Exploratory Analysis

Correlation and spatial autocorrelation. Both BP and the vegetationrelated features are moderately clustered in space (Moran’s I [21]), with the vegetation-continuity and fuel descriptors all falling between I ≈ 0.25 and 0.35: bio3x3 0.35, bio5x5 0.33, surface fuel 0.32, cont6m 0.32, cont10m 0.30, crown fuel 0.28 and CONT\_NORM\_2\_20 0.25. Slope (0.10) and aspect (−0.02) show almost none at this scale (Fig. 1).

Correlation with BP tells a similar story: at the pixel level nothing correlates strongly on its own, and aspect, slope and the continuity descriptors all sit at $| r | \leq 0 . 0 7$ . The one exception is surface fuel load $( r = 0 . 2 7 )$ , which is why we single it out below (Fig. 2).

![](images/fe2480c127e57ad6f6d7925e1416d111c31b018fd099dce247638884a6dbb2c7.jpg)  
Fig. 1. Moran’s I spatial autocorrelation with Burn Probability for each terrain and fuel covariate. The vegetation-continuity and fuel descriptors cluster moderately, while slope and aspect are close to zero.

![](images/9b703a99cbdc7916f1d73eaf12e6ff9b26349fc15ac5d00c531d9cde9773ec09.jpg)  
Fig. 2. Pearson correlation of each terrain and fuel covariate with Burn Probability, at the pixel level. Surface fuel load is a clear outlier.

At the coarser aggregated-cell level, the vegetation-continuity descriptors correlate strongly with each other, mostly above $r = 0 . 9 5$ . They therefore carry largely the same information. The correlation between aspect\_max and the continuity and biomass maxima is more moderate, at $r \approx 0 . 7 3 – 0 . 8 3$

Feature importance via gradient boosting. We trained an XGBoost [4] model to predict BP from the terrain and fuel covariates, once with surface fuel load and once without it. Including surface fuel load reduces the mean squared error (MSE) from 386.19 to 305.32, a 21% reduction. It therefore improves the prediction, rather than merely correlating with BP by coincidence (Fig. 3).

With surface fuel load included it absorbs almost all of the model’s importance (0.700), leaving the remaining covariates marginal (CONT\_NORM\_2\_20 0.069, aspect 0.086, cont10m 0.035, bio5x5n 0.023, cont6m 0.016, bio3x3 0.007) and crown fuel at efectively zero (0.000). Removing it redistributes importance towards CONT\_NORM\_2\_20 (0.260), aspect (0.210), slope (0.150) and cont10m (0.140), (Fig. 4).

Wind sensitivity. Because wind speed and direction are constant within a simulation, they cannot enter the pixel-level analysis directly. Grouping simulations instead by direction at fixed speed and by speed at fixed direction (180°), shows the efect: BP rises markedly for wind from 315°, and higher speeds enlarge and merge the high-BP regions. This sits awkwardly beside the small importance the models assign to wind (Section 7), a tension we return to in Section 9.

![](images/9dd3ab010182ae369e9300744e2e95741b311e19592b7d91203df21170461cd9.jpg)  
Fig. 3. BP prediction MSE with and without surface fuel load. Including it reduces MSE from 386.19 to 305.32, a 21% reduction.

![](images/ca322c36bafdfcbc8f88c55fcf4608f0e8b399feb51e8440a2c2a8d2782368d2.jpg)  
Fig. 4. XGBoost feature importance for BP prediction, with and without surface fuel load. Surface fuel load dominates when present (0.700). Removing it redistributes importance to $\mathrm { C O N T \_ N O R M \_ 2 \_ 2 0 }$ and aspect. Crown fuel contributes essentially nothing.

## 5 Methods

Problem formulation. We frame fire-spread prediction as image-to-image translation: given a 128 × 128 patch describing the current fire state, the terrain and fuel covariates and the wind, the model predicts the binary fire state 30 minutes later. Patches are extracted from the 1500 × 1500-pixel simulation domain with a 64-pixel stride, so neighbouring training patches overlap by half. That domain holds the single-channel arrival-time raster, which we combine with the static terrain and fuel layers and with the wind. Eleven channels make up the input. One carries the current fire state as a binary mask and two more carry the distance to the fire front, at the current time and 30 minutes earlier. Six hold the static terrain and fuel layers, namely aspect, slope, fuel load, bio5x5n, cont10m and CONT\_NORM\_2\_20. The remaining two carry wind speed and direction, each broadcast across the patch as a constant matrix. Patches with < 1% fire-active pixels are discarded, with no-spread patches retained at only 0.1% probability to limit class imbalance.

Custom U-Net. We selected the structure and hyperparameters by grid search on a 1,000-simulation subset. The search covered 3 activation functions (ReLU, Swish, Leaky ReLU), 6 loss functions and 6 filter-depth configurations, giving 108 variants trained for 10 epochs at batch size 10. The loss functions were binary cross-entropy (BCE), Matthews correlation [19,5], Dice [6,20], Hausdorf [13], Tversky [24] and Focal Tversky [1]. All variants follow a U-Net [22] encoder–bottleneck–decoder structure with a final 128 × 128 sigmoid output. Ranked first was Leaky ReLU with base filter width 64, depth 2 and a combined 0.5 BCE + 0.5 Dice loss. The configuration carried forward to all later experiments, named “Custom” in the tables and figures below, is the same one at base filter width 32 (depth 2, [32, 64]), which ranked third and difers only marginally on the combined score. Shallower networks and Dice-based losses generally did better, which we attribute to the strong class imbalance in fire segmentation, where burned pixels are rare. The BCE–Dice combination is an established choice for such tasks [25].

ResNet-50 baseline. We adapted an ImageNet-pretrained ResNet-50 [11] (include\_top=False, $1 2 8 \times 1 2 8 \times 3 ~ \mathrm { i n p u t } )$ . A learned $1 \times 1$ convolution (the input\_projection layer) first projects the 11 input channels onto 3. A decoder then restores the 128 × 128 output. It has five nearest-neighbour 2× upsampling stages, each followed by a $3 \times 3$ ReLU convolution with $5 1 2  2 5 6  1 2 8 $ $6 4  3 2  1 6$ filters and ends with $\mathrm { ~ a ~ } 1 \times 1$ sigmoid convolution. Unlike the Custom U-Net, the decoder has no encoder–decoder skip connections. The backbone stays frozen for the first 40 epochs and is then unfrozen for fine-tuning, at which point the Adam [14] learning rate drops from $1 0 ^ { - 4 } ~ \mathrm { t o } ~ 1 0 ^ { - 5 }$ . Activation is ReLU throughout, with the same 0.5 BCE + 0.5 Dice loss used for Custom.

Physics-informed neural network (PINN). The PINN shares the Custom architecture but augments the loss with a physics-inspired penalty encoding wind-driven advective transport of the fire-state field $u ( x , y , t )$ . With $\theta = 2 \pi \cdot W D$ , wind velocity components are

$$
v _ { x } = W S \cos { \theta } , \qquad v _ { y } = W S \sin { \theta } ,\tag{2}
$$

and spatial gradients are approximated by finite diferences. The governing advection equation $\partial u / \partial t + v _ { x } \partial u / \partial x + v _ { y } \partial u / \partial y = 0$ is enforced via

$$
\begin{array} { r } { L = L _ { d a t a } + \lambda _ { 1 } L _ { s m o o t h } + \lambda _ { 2 } L _ { a d v } , \quad L _ { a d v } = \left( v _ { x } \frac { \partial u } { \partial x } + v _ { y } \frac { \partial u } { \partial y } \right) ^ { 2 } , } \end{array}\tag{3}
$$

combined with a spatial-smoothness penalty $L _ { s m o o t h }$ and a mean-squared-error data-fitting term $L _ { d a t a }$ . In the released implementation $\lambda _ { 1 } = 0 . 0 5$ weights the smoothness term and $\lambda _ { 2 } = 0 . 1$ the advection term. In every other respect the PINN matches the Custom U-Net (base\_filters=32, depth 2, Leaky ReLU), trained for 100 epochs at batch size 10 through a custom training step. A sweep over $\lambda _ { 2 } \in \{ 0 , 0 . 0 5 , 0 . 1 , 0 . 2 \}$ across three seeds leaves the test Dice loss between 0.086 and 0.090, with no significant efect (Friedman, $p = 0 . 4 6 )$ and seed-to-seed variation larger than the spread between settings.

Swin-Unet. The fourth model replaces the convolutional encoder with a transformer one. We use SwinTransformerV2-Tiny [17] with window size 8, pretrained on ImageNet: four stages of [2, 2, 6, 2] blocks with [3, 6, 12, 24] attention heads, embedding dimension 96 and a stem patch size of 4. Apart from the backbone it is built exactly like the ResNet-50 baseline, sharing the same learned 1×1 channel projection, the same decoder of five upsampling stages, the same absence of skip connections and the same 40-epoch backbone freeze. Since the two difer only in their encoder, the comparison between them is a controlled one. Following earlier transformer-based wildfire work [15,29], we include it to test what changes when the encoder attends globally, both in accuracy and in what the network learns. This is a Swin encoder with a convolutional decoder rather than the symmetric transformer decoder of the original Swin-Unet. We keep the name “Swin-Unet” for consistency with the figures.

## 6 Experimental Setup

Training used mini-batches of size 10 per epoch, with structure/hyperparameters fixed per Section 5 and applied consistently across models. The ensemble is split by simulation. A simulation goes into the test set if its ignition ofset (OX, OY) falls on a 200 m grid, which represents every wind configuration equally. All remaining simulations form the training set. This gives $1 0 \times 1 0 = 1 0 0$ test ignition points per wind configuration, so $2 4 \times 1 0 0 = 2 { , } 4 0 0$ test simulations (22.7%) and 8,184 training simulations (77.3%). Patches are extracted within each set separately, so no simulation contributes patches to both. At prediction time, patches are re-assembled into a full-domain map by averaging overlapping predictions, then thresholded at 0.5 to a binary classification. We report five metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Dice loss, the Matthews correlation coeficient (MCC) [19,5] and a custom weighted error. Dice loss is overlap-based and suited to class imbalance. MCC summarises the whole confusion matrix as a single correlation between prediction and truth, from −1 to +1, and stays informative under class imbalance. The weighted error is

$$
\mathrm { E r r o r } = \frac { w _ { T P } T P + w _ { T N } T N - w _ { F P } F P - w _ { F N } F N } { T P + T N + F P + F N } ,\tag{4}
$$

where TP, TN, FP and FN are pixel counts of true and false positives and negatives. The four weights, which let false positives and false negatives be penalised diferently, are configurable. All values reported here use one fixed setting, so weighted errors are comparable across models but not on an absolute scale. Evaluation is stratified by fire size, with simulations grouped into four bins by final burned-area fraction (0–0.54%, 0.55–1.11%, 1.12–1.62% and 1.70–2.24%).

## 7 Results

## 7.1 Quantitative model comparison

Custom and PINN are efectively tied at the front. A paired Wilcoxon signedrank test over the 16 shared evaluation runs finds no significant diference between them $( W = 6 4 , p = 0 . 8 6$ ; mean weighted error 0.230 against 0.230). The PINN is the Custom U-Net with the physics terms added to its loss and nothing else changed, so the penalty does not measurably improve accuracy at this scale. The two also use 0.47 M parameters against 34.6 M for ResNet-50 and 32.7 M for Swin-Unet, so roughly seventy times fewer parameters give better accuracy. They lead on every metric and train in the least time (Table 2). ResNet-50 is behind (MCC 0.767) and Swin-Unet further still (MCC 0.499), despite both being the slowest to train. The weighted error gives the same ordering: 0.277 for Custom, 0.284 for PINN, 0.861 for ResNet-50 and 2.353 for Swin-Unet, averaged over the four fire-size groups at fire stages 30–120 min with a 25% patch step. Error is somewhat higher for the smallest and largest fire-size groups than for intermediate ones.

Table 2. Model comparison on 1,500 held-out patches, all four models evaluated on the same data. Training time is for 100 epochs on the GPU used for the original runs.
<table><tr><td>Model</td><td>Params MAE ↓ MSE ↓ Dice ↓ MCC ↑ Train (h)</td></tr><tr><td></td><td>0.043 0.945</td></tr><tr><td>0.019 5.8</td><td>Custom (U-Net) 0.47M 0.024</td></tr><tr><td>34.6 M</td><td>7.5</td></tr><tr><td>ResNet-50 PINN 0.47M 0.031</td><td>0.104 0.078 0.187 0.767</td></tr><tr><td>0.018</td><td>0.057 0.946 5.9</td></tr><tr><td>Swin-Unet 32.7M 0.216</td><td></td></tr><tr><td>0.194</td><td>0.361 0.499 8.7</td></tr></table>

## 7.2 Interpretability analysis

Gradient-based saliency and occlusion sensitivity give a clear ranking for Custom and PINN. The distance to the current fire front dominates (occlusion importance 0.326 and 0.324), followed by the current fire mask (0.149, 0.146), then fuel load and bio5x5n at around 0.09–0.10. ResNet-50 behaves diferently, relying almost entirely on the raw fire mask (0.289, half its total importance) while essentially ignoring the distance channels, for which it records the lowest value anywhere in the analysis (0.004 for dist). It never learned to exploit the distancetransform representation the other convolutional models depend on, which may be why it is the weakest of them. An earlier version of this analysis omitted the binary fire mask and including it matters: the mask ranks second for Custom and PINN and first for ResNet-50, so its omission had left out the top-ranked channel for one of the four models. Adding Swin-Unet (Fig. 5) reveals a split between the two architectures. Swin-Unet shifts importance toward fuel and terrain: its four highest channels are cont10m (0.403), bio5x5n (0.331), fuel load (0.299) and aspect (0.245), while the current fire-front distance falls to 0.061. The convolutional models show the opposite ordering. The transformer therefore appears to learn a diferent representation, one that relies more on static terrain and fuel. The same efect has been reported on a diferent wildfire dataset, where Swin-Unet’s Grad-CAM attributions also favoured vegetation and drought over a plain U-Net’s [29]. Finding it in two separate datasets and pipelines suggests it is a genuine property of Swin-style attention.

![](images/8f2752265288759e5d32495b2451114fd3264fc3c18762e7c80a66d348062a98.jpg)  
Fig. 5. Occlusion importance by model across all eleven input channels, measured as the mean change in predicted fire state when each channel is zeroed. Computed on 200 held-out patches. dist\_-30 and dist are the distances to the fire front 30 minutes earlier and at the current time.

We also perturbed each of the six static terrain and fuel channels locally, along the fire perimeter (Fig. 6). For the PINN and the custom U-Net, fuel\_load is by far the most influential: perturbing it produces the widest change in fire coverage. Slope produces a smaller response, but one with a consistent sign. The vegetation-continuity descriptors (cont10m, bio5x5n, CONT\_NORM\_2\_20) fall in between. How pronounced this is depends strongly on the model: Swin-Unet responds broadly across all six channels, while ResNet-50 is comparatively insensitive to all of them. Taken together the three methods agree only in part. Perturbation points to fuel load and for Swin-Unet to terrain and fuel more generally, whereas occlusion points to fire-front distance for the convolutional models. Saliency spreads importance more evenly than either. Wind has almost no importance in the saliency and occlusion analyses. We attribute this to how wind is encoded: it is constant across a patch, so methods that look for spatial variation have little to detect. Wind was not part of the perturbation analysis, which covers only the static terrain and fuel channels. To separate the two explanations we rotated the wind direction by a known amount and measured how far the prediction moved, taking the local spread direction from the displacement of the fire front between the two input stages. The result is unambiguous. Every model responds and the response grows monotonically with the rotation: for the Custom U-Net the mean absolute change in predicted fire state is 0.030 at 45°, 0.052 at 90°and 0.066 at 180°, against 0.045 for simply zeroing the channel. PINN behaves almost identically (0.069 at 180°), Swin-Unet is the most windsensitive (0.123) and ResNet-50 the least (0.027). Reversing the wind changes the prediction most, which is the physically expected ordering.

Wind is therefore not ignored and describing its importance as negligible was too strong. What the earlier analysis measured was limited by the encoding: because wind is constant across a patch, methods that look for spatially varying signal understate it. The ranking itself does not change, however. Even a full reversal moves the prediction by about 0.07 for the convolutional models, against 0.33 for occluding the fire-front distance, so wind remains a secondary factor behind the current fire state.

![](images/cdf97929e35d18f800b6eb659ebcdde4f672c5060a85295be3341e1cb8101d33.jpg)  
Fig. 6. Local perturbation along the fire perimeter: mean absolute change in predicted fire coverage when each terrain or fuel channel is perturbed. PINN and the custom U-Net respond overwhelmingly to fuel load, Swin-Unet responds broadly across channels and ResNet-50 is comparatively insensitive to all of them.

## 7.3 How much comes from the fire front?

The distance channels dominate the occlusion ranking, raising the question of whether the surrogate learns environmental drivers or mainly extrapolates the existing front. We retrained the Custom U-Net on reduced inputs under the same protocol, three seeds each (Table 3). These are interpretability probes, not deployment candidates.

Dropping either distance channel costs almost nothing. Dropping the fire mask as well halves accuracy. The fire mask and the two distance layers are partially redundant representations of the same fire-state information rather than three independent predictors. dist is a distance transform of the fire mask, so removing it leaves the geometry in the mask. Occlusion confirms this, fire-mask importance doubling from 0.121 to 0.248 while the eight environmental channels stay flat in total (0.399 to 0.342), the model shifting to another encoding of the same information. The distance channels are causally informative for near-term spread, not shortcuts in the pejorative sense, but their dominance limits what accuracy reveals about environmental sensitivity. Terrain, fuel and wind alone still reach MCC 0.42, so that signal is real but second-order.

Table 3. Input ablation on the Custom U-Net, mean ± standard deviation over three seeds. Geometry means the fire mask and both distance channels.
<table><tr><td>Input</td><td>Ch.</td><td>MSE↓</td><td>Dice↓ MCC ↑</td></tr><tr><td>Full</td><td>11</td><td> $0 . 0 2 5 \pm 0 . 0 0 1 0 . 0 5 7 \pm 0 . 0 0 3 \mathbf { 0 . 9 2 0 } \pm 0 . 0 0 5$ </td><td></td></tr><tr><td>No dist</td><td>10</td><td> $0 . 0 2 9 \pm 0 . 0 0 1 0 . 0 6 3 \pm 0 . 0 0 1 0 . 9 1 4 \pm 0 . 0 0 1$ </td><td></td></tr><tr><td>No dist_-30</td><td></td><td>10 0.026 ± 0.003 0.060 ± 0.006 0.919 ± 0.006</td><td></td></tr><tr><td>No distance channels 9</td><td></td><td> $0 . 0 2 8 \pm 0 . 0 0 1 0 . 0 6 1 \pm 0 . 0 0 1 0 . 9 1 7 \pm 0 . 0 0 2$ </td><td></td></tr><tr><td>No geometry</td><td>8</td><td> $0 . 1 6 2 \pm 0 . 0 0 7 0 . 4 0 2 \pm 0 . 0 1 5 0 . 4 2 4 \pm 0 . 0 1 5$ </td><td></td></tr></table>

## 8 Comparison with a Second Region

Because AI-based fire-spread models are often region-specific and degrade in unseen environments, we applied the trained models to Pedriza as a first check. It difers from Rectoret in terrain and environment, but one diference matters more than the rest. Because Pedriza is mapped at 10 m against Rectoret’s 2 m, a patch of the same pixel size covers twenty-five times the ground area (Section 3), so the models were trained at one scale and applied at another. Any gap reported below therefore mixes a change of region with a change of spatial scale and the present data cannot separate the two. Setting scale aside, the correlation structure of the aggregated cells is similar in the two regions, so they are comparable in statistical structure even though their absolute values difer. Quantitatively, we measured the histogram intersection of each shared covariate’s value distribution between the two regions, on a common 100-bin range. The mean overlap is 0.63, ranging from 0.47 for bio5x5n to 0.77 for bio3x3n, with slope at 0.56 and aspect at 0.71. The regions are therefore similar in the shape of their covariate distributions but far from identical: about a third of the distribution mass does not overlap. Pedriza is also steeper on average (mean slope 30.5 against 22.2). The models were first applied to Pedriza zero-shot, with no retraining. Table 4 reports that comparison, together with the efect of fine-tuning on Pedriza itself.

At the model level (Fig. 7), the trained surrogates were evaluated on Pedriza against their Rectoret performance across fire stage (40–120 min), using an MCC-based score and a boundary Dice-loss-based overlap error. Both metrics show the same pattern. The Pedriza curves follow the shape of the Rectoret curves, ofset by a roughly constant amount rather than collapsing. Table 4 gives the per-model figures. Zero-shot Dice loss rises from 0.469–0.487 on Rectoret to

0.552–0.637 on Pedriza. Fine-tuning on Pedriza recovers most of that gap for Custom and PINN, both reaching 0.481 after 50 epochs. ResNet-50 behaves differently. It improves only up to epoch 2 and then degrades slowly to 0.519. It is also the model that transfers best zero-shot, even though it is the weakest of the three on Rectoret.

Table 4. Cross-region comparison, averaged over fire stages 30–120 min, with ±1 standard deviation across evaluation simulations. Zero-shot means the Rectoret-trained model applied to Pedriza without retraining. The last row is Pedriza after 50 fine-tuning epochs on Pedriza.
<table><tr><td>Region</td><td>Metric</td><td>Custom</td><td> $\mathrm { R e s N e t - 5 0 }$ </td><td>PINN</td></tr><tr><td>Rectoret</td><td>MAE</td><td> $0 . 1 4 3 \pm 0 . 0 7 5$ </td><td> $0 . 1 4 5 \pm 0 . 0 7 7$ </td><td> $0 . 1 4 3 \pm 0 . 0 7 5$ </td></tr><tr><td></td><td>Dice</td><td> $0 . 4 6 9 \pm 0 . 0 4 7$ </td><td> $0 . 4 8 7 \pm 0 . 0 4 5$ </td><td> $0 . 4 6 9 \pm 0 . 0 4 7$ </td></tr><tr><td rowspan="2">Pedriza (zero-shot)</td><td>MAE Dice</td><td> $0 . 1 7 9 \pm 0 . 0 8 0$   $0 . 6 3 7 \pm 0 . 1 8 6$ </td><td> $0 . 1 9 8 \pm 0 . 0 8 5$   $\mathbf { 0 . 5 5 2 \pm 0 . 0 6 6 }$ </td><td> $0 . 1 8 5 \pm 0 . 0 8 0$   $0 . 5 9 8 \pm 0 . 1 4 9$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Pedriza (fine-tuned) Dice</td><td></td><td> $\mathbf { 0 . 4 8 1 \pm 0 . 0 5 5 }$ </td><td> $0 . 5 1 9 \pm 0 . 0 6 1$ </td><td> $\mathbf { 0 . 4 8 1 \pm 0 . 0 5 3 }$ </td></tr></table>

We report this as a descriptive comparison only. Establishing that the models generalise would require a second region mapped at the same 2 m resolution, so that scale and region are not varied together, and further regions beyond that.

![](images/fbcd659df0824af999ca23c34789e54b7206cc8efdb68954db73f82384799c3d.jpg)  
Fig. 7. Cross-region performance against fire stage, per model, evaluated zero-shot. Left: MAE. Right: boundary Dice loss. Solid lines are Rectoret, dashed lines Pedriza. Shaded bands are ±1 standard deviation across evaluation simulations. Swin-Unet is absent from the region logs.

Other cross-region transfer studies report the same pattern: a roughly constant ofset instead of a collapse. A cross-county wildfire-risk study found that transfer depends sharply on ecological similarity [16]. Transfer was near chance between dissimilar counties and strong between similar ones. The Rectoret– Pedriza ofset should therefore be read against how similar the two regions are, which we have not yet quantified. An attention-based ConvLSTM study found that pairwise self-attention transferred poorly across regions, while patchwise, window-local self-attention transferred better [18]. This bears directly on Swin-Unet, whose attention is also window-based and connects back to the interpretability results above.

## 9 Discussion

The main finding is that surface fuel load alone accounts for most of the explainable variance in burn probability at the pixel level. The vegetation-continuity descriptors add little, because they largely repeat one another. In practice, this suggests that fuel management should focus specifically on surface fuel loading rather than the entire suite of continuity metrics. This result requires one important caveat. The surface fuel load raster comes from the same BEHAVE-Anderson fuel classification that the simulator uses for fire spread. Therefore, it is not independent of the process that generated our target variable and some of its apparent dominance may reflect this common source rather than a purely physical efect. Confirming this finding with an independently measured fuel load product is a natural next test.

The wind seemed inconsistent at first. It has a strong impact at the simulation level, but was rated low in the channel-specific importance analysis. The rotation experiment solves this problem. The models use wind direction and the previous low ranking was partly an artefact of encoding wind as a constant across the area, which weakens methods that look for spatial variability. The physical efect is real, but secondary: wind reversal shifts the forecast by about a fifth compared to removing the distance from the fire front.

On generalisation, two further directions from the literature are relevant. First, our models give a single point prediction. A recent denoising-difusion surrogate for a stochastic CA burn-probability simulator samples an ensemble instead [28]. It beats a deterministic model of the same architecture on accuracy, spatial coherence and distributional quality at once. Our dataset is itself a stochastic ensemble, so a probabilistic surrogate is a natural next step. Second, work on diferentiable-Eikonal fire spread treats cross-region transfer as a test of whether the learned covariate–physics relationships are universal. It trains on 11 wildfire scenes and tests on 4 held-out scenes that never appear in training [9]. That protocol is stricter than our single train/test region split and worth adopting if more regions become available. Two limitations remain. The training data is completely synthetic, without validation against observed fires and covers one type of fuel.

## 10 Conclusions and Future Work

This paper compares four deep learning models for predicting wildfire spread in an ensemble of high-resolution simulations: a custom U-Net, a ResNet-50 network with transfer learning, a physics-informed model and a Swin-Unet transformer. This is combined with exploratory analysis and interpretability. Three results stand out. First, surface fuel load is the single strongest driver of burn probability, well ahead of all other covariates. Second, there is a clear separation between the architectures: convolutional models rely primarily on distance from the fire front, while Swin-Unet assigns more weight to terrain and fuel, consistent with results obtained in an unrelated dataset. Third, the comparison with Pedriza shows a consistent shift, not a decrease in accuracy. This is encouraging for application beyond a single region, although it does not yet prove model transferability. Two further steps must then be taken. The first is to test the surface fuel load against an independently measured fuel product, as the raster used here is from the same classification used by the simulator. The second is to further analyse the 2 m resolution region, which would allow us to separate the area change from the scale change, which we cannot do in the current comparison.

## References

1. Abraham, N., Khan, N.M.: A novel focal tversky loss function with improved attention u-net for lesion segmentation. In: 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). pp. 683–687 (2019). https://doi.org/10. 1109/ISBI.2019.8759329

2. Anderson, H.E.: Aids to determining fuel models for estimating fire behavior. Tech. Rep. General Technical Report INT-122, USDA Forest Service, Intermountain Forest and Range Experiment Station, Ogden, UT (1982). https://doi.org/10. 2737/INT-GTR-122

3. Burge, J., Bonanni, M., Ihme, M., Hu, R.L.: Convolutional lstm neural networks for modeling wildland fire dynamics (2020), arXiv preprint arXiv:2012.06679, https: //arxiv.org/abs/2012.06679

4. Chen, T., Guestrin, C.: Xgboost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. p. 785–794. KDD ’16, Association for Computing Machinery, New York, NY, USA (2016). https://doi.org/10.1145/2939672.2939785

5. Chicco, D., Jurman, G.: The advantages of the Matthews correlation coeficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics 21, 6 (2020). https://doi.org/10.1186/s12864-019-6413-7

6. Dice, L.R.: Measures of the amount of ecologic association between species. Ecology 26(3), 297–302 (1945). https://doi.org/10.2307/1932409, https:// esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1932409

7. Finney, M.A.: FARSITE: Fire area simulator – model development and evaluation. Tech. Rep. Research Paper RMRS-RP-4, USDA Forest Service, Rocky Mountain Research Station, Ogden, UT (1998). https://doi.org/10.2737/RMRS-RP-4

8. Finney, M.A.: An overview of FlamMap fire modeling capabilities. In: Fuels Management – How to Measure Success: Conference Proceedings. pp. 213–220. Proceedings RMRS-P-41, USDA Forest Service, Rocky Mountain Research Station (2006), https://research.fs.usda.gov/treesearch/25948

9. Gahtan, B., Shpund, J., Bronstein, A.M.: Diferentiable Randers-Finsler eikonal solvers (2026), arXiv preprint arXiv:2603.00035, https://arxiv.org/abs/2603. 00035

10. Han, J., Lee, J., Han, H., Na, Y., Lee, J.J.: Firesensenet: A dual-branch cnn with cross-attentive feature interaction for next-day wildfire spread prediction (2026), arXiv preprint arXiv:2604.07675, https://arxiv.org/abs/2604.07675

11. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90

12. Huot, F., Hu, R.L., Goyal, N., Sankar, T., Ihme, M., Chen, Y.F.: Next day wildfire spread: A machine learning dataset to predict wildfire spreading from remotesensing data. IEEE Transactions on Geoscience and Remote Sensing 60, 1–13 (2022). https://doi.org/10.1109/TGRS.2022.3192974

13. Karimi, D., Salcudean, S.E.: Reducing the hausdorf distance in medical image segmentation with convolutional neural networks. IEEE Transactions on Medical Imaging 39(2), 499–513 (2020). https://doi.org/10.1109/TMI.2019.2930068

14. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: International Conference on Learning Representations (ICLR) (2015). https://doi.org/ 10.48550/arXiv.1412.6980

15. Lahrichi, S., Bova, J., Johnson, J., Malof, J.: Improved wildfire spread prediction with time-series data and the WSTS+ benchmark. In: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2026), arXiv:2502.12003

16. Liu, C., Mostafavi, A.: Wildfiregenome: Interpretable machine learning reveals local drivers of wildfire risk and their cross-county variation (2025), arXiv preprint arXiv:2511.11589, https://arxiv.org/abs/2511.11589

17. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin Transformer: Hierarchical vision transformer using shifted windows. In: IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9992–10002 (2021). https://doi.org/10.1109/ICCV48922.2021.00986

18. Masrur, A., Yu, M., Taylor, A.: Capturing and interpreting wildfire spread dynamics: attention-based spatiotemporal models using convlstm networks. Ecological Informatics 82, 102760 (2024). https://doi.org/10.1016/j.ecoinf.2024.102760

19. Matthews, B.: Comparison of the predicted and observed secondary structure of t4 phage lysozyme. Biochimica et Biophysica Acta (BBA) - Protein Structure 405(2), 442–451 (1975). https://doi.org/10.1016/0005-2795(75)90109-9, https://www.sciencedirect.com/science/article/pii/0005279575901099

20. Milletari, F., Navab, N., Ahmadi, S.A.: V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In: Fourth International Conference on 3D Vision (3DV). pp. 565–571 (2016). https://doi.org/10.1109/3DV.2016.79

21. Moran, P.A.P.: Notes on continuous stochastic phenomena. Biometrika 37(1/2), 17–23 (1950). https://doi.org/10.2307/2332142

22. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI). LNCS, vol. 9351, pp. 234–241 (2015). https://doi.org/ 10.1007/978-3-319-24574-4\_28

23. Rothermel, R.C.: A mathematical model for predicting fire spread in wildland fuels. Tech. Rep. Research Paper INT-115, USDA Forest Service, Intermountain Forest and Range Experiment Station, Ogden, UT (1972). https://doi.org/10.2737/ INT-RP-115

24. Salehi, S.S.M., Erdogmus, D., Gholipour, A.: Tversky loss function for image segmentation using 3d fully convolutional deep networks. In: Wang, Q., Shi, Y., Suk, H.I., Suzuki, K. (eds.) Machine Learning in Medical Imaging. pp. 379– 387. Springer International Publishing, Cham (2017). https://doi.org/10.1007/ 978-3-319-67389-9\_44

25. Shahid, M., Chen, S.F., Hsu, Y.L., Chen, Y.Y., Chen, Y.L., Hua, K.L.: Forest fire segmentation via temporal transformer from aerial images. Forests 14(3), 563 (2023). https://doi.org/10.3390/f14030563

26. Vogiatzoglou, K., Papadimitriou, C., Bontozoglou, V., Ampountolas, K.: Physicsinformed neural networks for parameter learning of wildfire spreading. Computer Methods in Applied Mechanics and Engineering 434, 117545 (2025). https:// doi.org/10.1016/j.cma.2024.117545

27. Xia, Z., Cheng, S.: Pytorchfire: A gpu-accelerated wildfire simulator with diferentiable cellular automata. Environmental Modelling & Software 188, 106401 (2025). https://doi.org/10.1016/j.envsoft.2025.106401

28. Yu, W., Ghosh, A., Finn, T.S., Arcucci, R., Bocquet, M., Cheng, S.: A probabilistic approach to wildfire spread prediction using a denoising difusion surrogate model. Geoscientific Model Development 19, 1027–1054 (2026). https: //doi.org/10.5194/gmd-19-1027-2026

29. Zhou, Y., Kong, R., Xu, Z., Xu, L., Cheng, S.: Comparative and interpretative analysis of cnn and transformer models in predicting wildfire spread using remote sensing data. Journal of Geophysical Research: Machine Learning and Computation 2(2), e2024JH000409 (2025). https://doi.org/10.1029/2024JH000409